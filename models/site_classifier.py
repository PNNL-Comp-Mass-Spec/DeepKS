import pathlib
import re
from matplotlib import pyplot as plt
import numpy as np
import sklearn.cluster
import sklearn.metrics
import os, pandas as pd
import sklearn
from termcolor import colored
import torch.nn as nn, torch.utils.data
import torch

num_aa = 21
seq_len = 15
aas = list("ACDEFGHIKLMNPQRSTVWYX")
aa_inv = {aa: i for i, aa in enumerate(aas)}
num_classes = 10
from ..tools.NNInterface import NNInterface
from .KinaseSubstrateRelationship import MultipleCNN
from ..config.root_logger import get_logger

logger = get_logger()

os.chdir(pathlib.Path(__file__).resolve().parent.parent.parent)


class DeepSC(nn.Module):
    def __init__(self, emb_dim, linear_layer_sizes, drop_p=0.5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.emb_dim = emb_dim
        self.emb = nn.Embedding(num_aa, emb_dim)
        # self.cnn = MultipleCNN(1, 6, 2, num_aa, True, True)
        self.act = nn.ELU()
        self.drop = nn.Dropout(drop_p)

        self.linear_layer_sizes: list[int] = linear_layer_sizes if linear_layer_sizes is not None else [10]

        # Create linear layers
        self.linear_layer_sizes.insert(0, 15 * emb_dim)
        self.linear_layer_sizes.append(num_classes)

        # Put linear layers into Sequential module
        lls = []
        for i in range(len(self.linear_layer_sizes) - 1):
            lls.append(nn.Linear(self.linear_layer_sizes[i], self.linear_layer_sizes[i + 1]))
            lls.append(self.act)
            lls.append(self.drop)

        self.linears = nn.Sequential(*lls)

    def forward(self, x):
        x = self.emb(x)
        x = torch.flatten(x, 1, -1)
        x = self.linears(x)
        return x

    @staticmethod
    def embedding(seq):
        return np.asarray([aa_inv[aa] for aa in seq])

    class SiteDataSet(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    @staticmethod
    def format_data(*fns, kin_fam_grp, sample_size=None):
        dataloaders: list[torch.utils.data.DataLoader] = []
        kin_to_grp = pd.read_csv(kin_fam_grp)[["Kinase", "Uniprot", "Group"]]
        kin_to_grp["Symbol"] = (
            kin_to_grp["Kinase"].apply(lambda x: re.sub(r"[\(\)\*]", "", x)) + "|" + kin_to_grp["Uniprot"]
        )
        kin_to_grp = kin_to_grp.set_index("Symbol").to_dict()["Group"]
        grp_to_idx = {grp: i for i, grp in enumerate(sorted(list(set(kin_to_grp.values()))))}
        for fn in fns:
            dff = pd.read_csv(fn)[["Site Sequence", "Gene Name of Kin Corring to Provided Sub Seq"]]
            df = dff.values
            X = []
            y = [grp_to_idx[kin_to_grp[y]] for y in df[:, 1]]
            for seq in df[:, 0]:
                X.append(DeepSC.embedding(seq))
            Xnd: torch.Tensor = torch.from_numpy(np.asarray(X).astype(int))
            ynd: torch.Tensor = torch.from_numpy(np.asarray(y).astype(int))
            np.random.seed(42)
            dataloaders.append(
                torch.utils.data.DataLoader(DeepSC.SiteDataSet(Xnd, ynd), batch_size=10000, shuffle=True)
            )

        return tuple(dataloaders)

    @staticmethod
    def get_SC(
        train_file,
        val_file,
        kin_fam_grp_file,
    ):
        model = DeepSC(emb_dim=4, linear_layer_sizes=[100])
        interface = NNInterface(model, nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=0.01))
        train_loader, val_loader = DeepSC.format_data(train_file, val_file, kin_fam_grp=kin_fam_grp_file)
        interface.train(train_loader, num_epochs=50, metric="acc", val_dl=val_loader)


class SiteClassifier:
    @staticmethod
    def one_hot_aa_embedding(seq):
        assert len(seq) == seq_len
        patch = np.zeros((len(seq), num_aa))
        aa_dict = {aa: i for i, aa in enumerate(aas)}
        for i, aa in enumerate(seq):
            patch[i, aa_dict[aa]] = 1
        embedding = patch.ravel()
        return embedding

    @staticmethod
    def format_data(*fns, kin_fam_grp, sample_size=None):
        X_y_tuples: list[tuple[np.ndarray, np.ndarray]] = []
        kin_to_grp = pd.read_csv(kin_fam_grp)[["Kinase", "Uniprot", "Group"]]
        kin_to_grp["Symbol"] = (
            kin_to_grp["Kinase"].apply(lambda x: re.sub(r"[\(\)\*]", "", x)) + "|" + kin_to_grp["Uniprot"]
        )
        kin_to_grp = kin_to_grp.set_index("Symbol").to_dict()["Group"]
        for fn in fns:
            dff = pd.read_csv(fn)[["Site Sequence", "Gene Name of Kin Corring to Provided Sub Seq"]]
            df = dff.values
            X = []
            y = [kin_to_grp[y] for y in df[:, 1]]
            for seq in df[:, 0]:
                X.append(SiteClassifier.one_hot_aa_embedding(seq))
            Xnd: np.ndarray = np.asarray(X).astype(int)
            ynd: np.ndarray = np.asarray(y)
            np.random.seed(42)
            if isinstance(sample_size, int):
                shuffled_ids = np.random.permutation(len(y))[:sample_size]
            else:
                shuffled_ids = np.random.permutation(len(y))
            Xnd = Xnd[shuffled_ids]
            ynd = ynd[shuffled_ids]
            X_y_tuples.append((Xnd, ynd))
        return tuple(X_y_tuples)

    @staticmethod
    def get_group_classifier(
        train_file,
        kin_fam_grp_file,
        grp_to_cls={},
        gc_hyperparameters={
            "base_classifier": "ensemble.RandomForestClassifier",
            "n_estimators": 10000,
            "max_depth": 3,
        },
    ):
        cpu_count_raw = os.cpu_count()
        cpu_count = max(1, cpu_count_raw // 2) if cpu_count_raw is not None else 1
        exec(f"import sklearn.{gc_hyperparameters['base_classifier'].split('.')[0]}")
        hps = {k: v for k, v in gc_hyperparameters.items() if k != "base_classifier"}
        try:
            gc = eval(gc_hyperparameters["base_classifier"])(**hps, random_state=42, n_jobs=cpu_count)
        except TypeError:
            gc = eval(gc_hyperparameters["base_classifier"])(**hps, n_jobs=cpu_count)
        (tr,) = SiteClassifier.format_data(train_file, kin_fam_grp=kin_fam_grp_file)
        logger.status("Training Group Classifier")
        gc.fit(tr[0], tr[1])
        return gc

    @staticmethod
    def performance(model, X, y, metric_fn=sklearn.metrics.accuracy_score):
        y_true, y_pred = y, model.predict(X)
        return round(metric_fn(y_true, y_pred), 4)

    @staticmethod
    def format_data_for_clusters(*fns, site_grp):
        X_y_tuples: list[tuple[np.ndarray, np.ndarray]] = []
        for fn in fns:
            dff = pd.read_csv(fn)["Site Sequence"]
            df = dff.values
            X = []
            y = [site_grp[y] for y in df]
            for seq in df:
                X.append(SiteClassifier.one_hot_aa_embedding(seq))
            Xnd: np.ndarray = np.asarray(X).astype(int)
            ynd: np.ndarray = np.asarray(y)
            np.random.seed(42)
            shuffled_ids = np.random.permutation(len(y))
            Xnd = Xnd[shuffled_ids]
            ynd = ynd[shuffled_ids]
            X_y_tuples.append((Xnd, ynd))
        return tuple(X_y_tuples)

    @staticmethod
    def get_clustered_classifier(train_file, site_to_group, cluster_hps={"n_clusters": 9, "random_state": 42}):
        (tr,) = SiteClassifier.format_data_for_clusters(train_file, site_grp=site_to_group)
        cluster = sklearn.cluster.KMeans(**cluster_hps)
        cluster.fit(tr[0])
        return cluster


if __name__ == "__main__":
    fd = (
        "DeepKS/data/raw_data_31834_formatted_65_26610.csv",
        "DeepKS/data/raw_data_6500_formatted_95_5698.csv",
        "DeepKS/data/raw_data_6406_formatted_95_5616.csv",
    )
    kfg = "DeepKS/data/preprocessing/kin_to_fam_to_grp_826.csv"

    # deepsc = DeepSC.get_SC(*fd[:2], kin_fam_grp_file=kfg)
    site_to_grp = pd.read_csv("DeepKS/data/preprocessing/site_to_group_9108.csv").set_index("Site").to_dict()["Group"]
    clf = SiteClassifier.get_clustered_classifier(fd[0], site_to_grp, cluster_hps={"n_clusters": 9, "random_state": 42})

    tr, vl, te = SiteClassifier.format_data_for_clusters(*fd, site_grp=site_to_grp)
    # clf = SiteClassifier.get_group_classifier(fd[0], kfg, gc_hyperparameters={'base_classifier': 'neighbors.KNeighborsClassifier', 'n_neighbors': 11, 'metric': 'cosine', 'weights': lambda x: 1 / (x + 0.01)})

    print("Train Acc: ", SiteClassifier.performance(clf, *tr))
    print("Val Acc: ", SiteClassifier.performance(clf, *vl))
    print("Test Acc: ", SiteClassifier.performance(clf, *te))
    cm = sklearn.metrics.confusion_matrix(te[1], clf.predict(te[0]))
    fig, ax = plt.subplots(figsize=(7, 7))
    sklearn.metrics.ConfusionMatrixDisplay(cm, display_labels=sorted(list(set(te[1])))).plot(ax=ax)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.show()
