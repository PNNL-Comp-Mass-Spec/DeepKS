# Imports
import torch, torch.nn as nn, pandas as pd, numpy as np, sys, pathlib, os
from ..tools.NNInterface import NNInterface
from ..tools.tensorize import gather_data

join_first = (
    lambda levels, x: x
    if os.path.isabs(x)
    else os.path.join(pathlib.Path(__file__).parent.resolve(), *[".."] * levels, x)
)

# Constants
KIN_GROUPS = ["AGC", "ATYPICAL", "CK1", "CAMK", "CMGC", "OTHER", "STE", "TK", "TKL"]
NUM_KIN_GROUPS = len(KIN_GROUPS)
NUM_TOK = 22
PADDED_SIZE = 4128

PAD_IDX = None


def set_pad_idx(pad_idx):
    global PAD_IDX
    PAD_IDX = pad_idx


class SeqGC(nn.Module):
    """
    RNN to classify kinases into groups based on sequence
    """

    def __init__(self, num_features=NUM_TOK, hidden_features=NUM_TOK * 4, num_recur=3, linear_layer_sizes=[]) -> None:
        """Initialize RNN

        Args:
            kwargs (dict): Dictionary of hyperparameters including
                            * num_features (int): the number of features for each amino acid token
                            * hidden_features (int): the number of hidden state features
                            * num_recur (int): the number of recurrent layers
                            * linear_layer_sizes (list[int]): list of sizes of additional linear layers between output
                                                                of LSTM and final output vector
        """
        super().__init__()
        # Hyperparameters
        self.num_features: int = num_features
        self.hidden_features: int = hidden_features
        self.num_recur: int = num_recur
        self.linear_layer_sizes: list[int] = linear_layer_sizes

        # Layers
        self.lstm = nn.LSTM(self.num_features, self.hidden_features, self.num_recur, batch_first=True)
        self.emb = nn.Embedding(NUM_TOK, self.num_features, padding_idx=PAD_IDX)
        self.fltn = nn.Flatten()

        # Create linear layers
        self.linear_layer_sizes.insert(0, self.hidden_features)
        self.linear_layer_sizes.append(NUM_KIN_GROUPS)
        # At this point, the linear layer (input, output) shapes will be
        # (hidden_features, X), (X, Y), (Y, Z), (Z, NUM_KIN_GROUPS),
        # if X,Y, and Z are specified.

        # Put linear layers into Sequential module
        lls = []
        for i in range(len(self.linear_layer_sizes) - 1):
            lls.append(nn.Linear(self.linear_layer_sizes[i], self.linear_layer_sizes[i + 1]))

        self.linears = nn.Sequential(*lls)

    def forward(self, _, kin_seq_input):
        """Forward pass of RNN

        Args:
            _ (torch.LongTensor): placeholder for site sequence, which is not used in this model
            kin_seq_input (torch.LongTensor): Tensor of kinase sequence inputs of shape
                                                (batch_size, padded_seq_len, num_emb_features)

        Returns:
            torch.LongTensor: Output tensor of scores of shape (batch_size, num_kin_groups)
        """
        kin_seq = self.emb(kin_seq_input)  # Embed kinase sequence
        _, (_, c_out) = self.lstm(kin_seq)  # Run embedded sequence through LSTM
        out = self.linears(c_out[-1])  # Pass cell output of last cell to the linear layer(s)
        return out


def get_kin_seq_to_group_dict(seq_and_uniprot_df, uniprot_and_group_df) -> dict:
    """Helper function to get true group labels

    Args:
        seq_and_uniprot_df (str): Path of csv containing sequence-uniprot mappings
        uniprot_and_group_df (str): Path of csv containing uniprot-group mappings

    Returns:
        dict: Map from kinase sequence to group labels
    """
    seq_to_uniprot = pd.read_csv(seq_and_uniprot_df).set_index("kinase_seq").to_dict()["kinase"]
    uniprot_to_group = pd.read_csv(uniprot_and_group_df)
    uniprot_to_group = uniprot_to_group.set_index("Uniprot").to_dict()["Group"]

    seq_to_group = {x: uniprot_to_group[seq_to_uniprot[x]] for x in seq_to_uniprot.keys()}

    return seq_to_group


def get_relevant_data_loaders(df_filenames: list[str], batch_size: int = 64):
    # Get dataloaders
    dfs = []
    for df_filename in df_filenames:
        df = pd.read_csv(df_filename)
        df.drop_duplicates(subset=["Kinase Sequence"], inplace=True, keep="first")
        dfs.append(df)

    kin_seq_to_group_dict = get_kin_seq_to_group_dict(
        join_first(1, "data/raw_data/kinase_seq_826.csv"),
        join_first(1, "data/preprocessing/kin_to_fam_to_grp_826.csv"),
    )
    common_args = {
        "tuf": 0,
        "train_batch_size": batch_size,
        "n_gram": 1,
        "maxsize": PADDED_SIZE,
        "kin_seq_to_group": kin_seq_to_group_dict,
        # "subsample_num": 100,  # Can make dataset smaller here
    }
    (train_loader, _, _, _), train_info = list(gather_data(dfs[0], trf=1, vf=0, tef=0, **common_args))[0]
    (_, val_loader, _, _), _ = list(gather_data(dfs[1], trf=0, vf=1, tef=0, **common_args))[0]
    (_, _, _, test_loader), _ = list(gather_data(dfs[2], trf=0, vf=0, tef=1, **common_args))[0]

    set_pad_idx(train_info["tok_dict"]["<PADDING>"])

    return train_loader, val_loader, test_loader


def get_NNInterface(model, batch_size):
    """Obtain Neural Network Training/Val/Test Interface and write model summary

    Args:
        model (torch.nn.Module): The NN model in question
        batch_size (int): batch size for training

    Returns:
        DeepKS.Tools.NNInterface.NNInterface: object that allows for training, validation, and testing
    """
    inp_size = [(batch_size, 15), (batch_size, PADDED_SIZE)]
    inp_types = [torch.long, torch.long]
    interface = NNInterface(
        model,
        nn.CrossEntropyLoss(),
        torch.optim.Adam(model.parameters()),
        inp_size,
        inp_types,
        "RNN-GC-arch.txt",
        torch.device("cpu"),
    )
    interface.write_model_summary()
    return interface


def main():
    # Setup hyperparameters
    model_HPs = {"num_features": 32, "hidden_features": 64, "num_recur": 10, "linear_layer_sizes": [30, 24, 15]}
    train_options = {"num_epochs": 20}
    batch_size = 128

    # Obtain dataloaders
    train_loader, val_loader, test_loader = get_relevant_data_loaders(
        [
            join_first(1, "data/raw_data_31834_formatted_65_26610.csv"),
            join_first(1, "data/raw_data_6500_formatted_95_5698.csv"),
            join_first(1, "data/raw_data_6406_formatted_95_5616.csv"),
        ]
    )

    # Initialize Model
    model = SeqGC(**model_HPs)

    # Initialize Training/Testing Interface and create model summary diagram
    interface = get_NNInterface(model, batch_size)

    # Train, test, and create roc_curve
    interface.train(train_loader, val_dl=val_loader, **train_options)
    interface.test(test_loader)


if __name__ == "__main__":
    main()
