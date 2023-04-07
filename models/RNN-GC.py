### SETUP ----
# Imports
from termcolor import colored

print(colored("Status: Importing libraries...", "green"))
import torch, torch.nn as nn, pandas as pd, numpy as np, sys, pathlib, os
from ..tools.NNInterface import NNInterface
from ..tools.tensorize import gather_data

join_first = lambda levels, x: (
    x if os.path.isabs(x) else os.path.join(pathlib.Path(__file__).parent.resolve(), *[".."] * levels, x)
)

# Constants
KIN_GROUPS = ["AGC", "ATYPICAL", "CK1", "CAMK", "CMGC", "OTHER", "STE", "TK", "TKL"]
NUM_KIN_GROUPS = len(KIN_GROUPS)
NUM_TOK = 22
PADDED_SIZE = 4128

PAD_IDX = None


def set_pad_idx(pad_idx):
    """Small helper function"""
    global PAD_IDX
    PAD_IDX = pad_idx


### RNN DEFINITION ----
class SeqGC(nn.Module):
    """RNN to classify kinases into groups based on sequence"""

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
        self.activation = nn.ELU()

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
            lls.append(self.activation)

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
        kin_seq = self.activation(kin_seq)  # Pass through activation function

        __, (_, c_out) = self.lstm(
            kin_seq
        )  # Run embedded sequence through LSTM, get cell output, throw other stuff away
        c_out = self.activation(c_out)  # Pass cell output through activation function

        out = self.linears(c_out[-1])  # Pass cell output of last cell to the linear layer(s)

        # This does not change any of the tensors, but it is a workaround for a bug in PyTorch,
        # if the device is an MPS device (e.g. Apple M1)
        if "mps" in str(self.parameters().__next__().device):
            out = _mps_workaround(__, out)

        return out


### HELPER FUNCTIONS ----
def _mps_workaround(__, out):
    """Workaround function for a bug in PyTorch, when using metal performance shaders (e.g. Apple M1)

    Args:
        __ (torch.Tensor): "Output" (what PyTorch calls "output") of LSTM
        out (torch.Tensor): Final cell state of LSTM
    """
    zeroed = __.zero_()
    return out + torch.sum(zeroed)


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


def get_relevant_data_loaders(df_filenames: list[str], batch_size: int = 64, device=torch.device("cpu")):
    """Get dataloaders for training, validation, and test sets

    Args:
        df_filenames (list[str]): List of dataframes to turn into dataloaders
        batch_size (int): Training batch size.
        device (torch.device): NN device.

    Returns:
        tuple[dataloader, dataloader, dataloader]: Train, val, and test dataloaders
    """

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
        "device": device
        # "subsample_num": 100,  # Can make dataset smaller here
    }
    (train_loader, _, _, _), train_info = list(gather_data(dfs[0], trf=1, vf=0, tef=0, **common_args))[0]
    (_, val_loader, _, _), _ = list(gather_data(dfs[1], trf=0, vf=1, tef=0, **common_args))[0]
    (_, _, _, test_loader), _ = list(gather_data(dfs[2], trf=0, vf=0, tef=1, **common_args))[0]

    set_pad_idx(train_info["tok_dict"]["<PADDING>"])

    return train_loader, val_loader, test_loader


def get_NNInterface(model, batch_size, device=torch.device("cpu"), lr=0.1):
    """Obtain Neural Network Training/Val/Test Interface and write model summary

    Args:
        model (torch.nn.Module): The NN model in question
        batch_size (int): batch size for training
        device (torch.device): device to train on
        lr (float): learning rate

    Returns:
        DeepKS.Tools.NNInterface.NNInterface: object that allows for training, validation, and testing
    """
    inp_size = [(batch_size, 15), (batch_size, PADDED_SIZE)]
    inp_types = [torch.long, torch.long]
    interface = NNInterface(
        model,
        nn.CrossEntropyLoss(),
        torch.optim.Adam(model.parameters(), lr=lr),
        inp_size,
        inp_types,
        "RNN-GC-arch.txt",
        device,
    )
    interface.write_model_summary()
    return interface


### MAIN ENTRY POINT ----
def main():
    """Main entry point for training, validation, and testing"""
    DEVICE = torch.device("cpu")
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    model_HPs = {"num_features": 8, "hidden_features": 8, "num_recur": 8, "linear_layer_sizes": [32, 16, 8]}
    train_options = {"num_epochs": 8, "metric": "acc"}
    batch_size = 64

    # Obtain dataloaders
    train_loader, val_loader, test_loader = get_relevant_data_loaders(
        [
            join_first(1, "data/raw_data_31834_formatted_65_26610.csv"),
            join_first(1, "data/raw_data_6500_formatted_95_5698.csv"),
            join_first(1, "data/raw_data_6406_formatted_95_5616.csv"),
        ],
        device=DEVICE,
        batch_size=batch_size,
    )

    print(hash(tuple(train_loader.dataset.target.data.numpy().ravel().tolist())))

    # Initialize Model
    model = SeqGC(**model_HPs)

    # Initialize Training/Testing Interface and create model summary diagram
    interface = get_NNInterface(model, batch_size, device=DEVICE, lr=0.01)

    # Train and test
    interface.train(train_loader, val_dl=val_loader, **train_options)
    interface.test(test_loader, print_sample_predictions=True, metric="acc")


if __name__ == "__main__":
    main()
