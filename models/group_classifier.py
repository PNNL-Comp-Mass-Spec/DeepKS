import json, torch, torch.nn as nn, sys, os
sys.path.append("../tools")
from tensorize_group_classifier import gather_data
from NNInterface import NNInterface
from parse import parsing
from pprint import pprint
from formal_layers import Transpose

os.getcwd()
pprint(None)

class GroupClassifier(nn.Module):
    def __init__(self, **kwargs):
        """
        Required kwargs:
            input_size: int
            embedding_dim: int
            num_classes: int
            kernel_size: int
            batch_norm: bool
            stride: int
            dropout: float
            activation_fn: nn.Module
        """
        try:
            embedding_dim = kwargs["embedding_dim"]
            num_classes = kwargs["num_classes"]
            kernel_size = kwargs["kernel_size"]
            batch_norm = kwargs["batch_norm"]
            stride = kwargs["stride"]
            dropout = kwargs["dropout"]
            activation_fn = kwargs["activation_fn"]
        except KeyError as ke:
            print(ke, "not found in kwargs of GroupClassifier.__init__!")
            raise

        super().__init__()

        self.embedding = nn.Embedding(NUM_AA, embedding_dim=embedding_dim)
        self.transpose = Transpose(-1, -2)
        self.CNN = nn.Conv1d(
            in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=kernel_size, stride=stride
        )
        self.do_batch_norm = batch_norm
        self.batch_norm = nn.BatchNorm1d(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = activation_fn
        self.ll1 = nn.Linear(embedding_dim, embedding_dim)
        self.ll2 = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transpose(x)
        x = self.CNN(x)
        x = self.relu(x)
        if self.do_batch_norm:
            x = self.batch_norm(x)
        x = self.dropout(x)
        x = torch.mean(x, dim=2)
        x = self.ll1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.ll2(x)
        return x

def main(config):
    try:
        num_epochs = config["num_epochs"]
        batch_size = config["batch_size"]
        lr_decay_amount = config["lr_decay_amount"]
        lr_decay_freq = config["lr_decay_freq"]
        metric = config["metric"]
        n_gram = config["n_gram"]
    except KeyError as ke:
        print(ke, "not found in config of GroupClassifier.main!")
        raise

    model = GroupClassifier(**config)

    tokdict = json.load(open("json/tok_dict.json", "rb"))
    tokdict["-"] = tokdict["<PADDING>"]

    (train_loader, _, _, _), _ = gather_data(
        train_filename,
        trf=1,
        vf=0,
        tuf=0,
        tef=0,
        train_batch_size=batch_size,
        n_gram=n_gram,
        tokdict=tokdict,
        device=torch.device(device),
        maxsize=KIN_LEN,
    )
    (_, val_loader, _, _), _ = gather_data(
        val_filename,
        trf=0,
        vf=1,
        tuf=0,
        tef=0,
        n_gram=config["n_gram"],
        tokdict=tokdict,
        device=torch.device(device),
        maxsize=KIN_LEN,
    )
    (_, _, _, test_loader), tl_info = gather_data(
        test_filename,
        trf=0,
        vf=0,
        tuf=0,
        tef=1,
        n_gram=config["n_gram"],
        tokdict=tokdict,
        device=torch.device(device),
        maxsize=KIN_LEN,
    )

    cm_lab_dict = tl_info["remapping_class_label_dict_inv"]

    interface = NNInterface(
        model,
        nn.CrossEntropyLoss(),
        torch.optim.Adam(model.parameters(), lr=0.001),
        inp_size=NNInterface.get_input_size(train_loader),
        inp_types=NNInterface.get_input_types(train_loader),
        model_summary_name="GroupClassifier",
        device=device,
    )

    interface.train(
        train_loader,
        num_epochs=num_epochs,
        lr_decay_amount=lr_decay_amount,
        lr_decay_freq=lr_decay_freq,
        val_dl=val_loader,
        metric=metric
    )

    interface.test(test_loader, metric = metric)
    interface.get_all_conf_mats(test_loader, savefile="Group_Classifier_CM.png", metric=metric, cm_labels=cm_lab_dict)


args = parsing()
device = args["device"]
train_filename = args["train"]
val_filename = args["val"]
test_filename = args["test"]
KIN_LEN = 4128
NUM_AA = 22

if __name__ == "__main__":
    config = {
        "embedding_dim": 32,
        "num_classes": 10,
        "kernel_size": 100,
        "batch_norm": True,
        "stride": 10,
        "dropout": 0.5,
        "activation_fn": nn.ReLU(),
        "lr_decay_amount": 0.1,
        "lr_decay_freq": 10,
        "metric": "acc",
        'num_epochs': 10,
        'batch_size': 32,
        'n_gram': 1
    }

    main(config)
