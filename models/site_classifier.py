import pathlib  # pragma: no cover
import re  # pragma: no cover
from matplotlib import pyplot as plt  # pragma: no cover
import numpy as np  # pragma: no cover
import sklearn.cluster  # pragma: no cover
import sklearn.metrics  # pragma: no cover
import os, pandas as pd  # pragma: no cover
import sklearn  # pragma: no cover
from termcolor import colored  # pragma: no cover
import torch.nn as nn, torch.utils.data  # pragma: no cover
import torch  # pragma: no cover

# pragma: no cover
num_aa = 21  # pragma: no cover
seq_len = 15  # pragma: no cover
aas = list("ACDEFGHIKLMNPQRSTVWYX")  # pragma: no cover
aa_inv = {aa: i for i, aa in enumerate(aas)}  # pragma: no cover
num_classes = 10  # pragma: no cover
from ..tools.NNInterface import NNInterface  # pragma: no cover
from .KinaseSubstrateRelationshipClassic import MultipleCNN  # pragma: no cover
from ..config.logging import get_logger  # pragma: no cover

# pragma: no cover
logger = get_logger()  # pragma: no cover
# pragma: no cover
os.chdir(pathlib.Path(__file__).resolve().parent.parent.parent)  # pragma: no cover


# pragma: no cover
# pragma: no cover
class DeepSC(nn.Module):  # pragma: no cover
    def __init__(self, emb_dim, linear_layer_sizes, drop_p=0.5, *args, **kwargs) -> None:  # pragma: no cover
        super().__init__(*args, **kwargs)  # pragma: no cover
        self.emb_dim = emb_dim  # pragma: no cover
        self.emb = nn.Embedding(num_aa, emb_dim)  # pragma: no cover
        # self.cnn = MultipleCNN(1, 6, 2, num_aa, True, True) # pragma: no cover
        self.act = nn.ELU()  # pragma: no cover
        self.drop = nn.Dropout(drop_p)  # pragma: no cover
        # pragma: no cover
        if linear_layer_sizes is not None:  # pragma: no cover
            self.linear_layer_sizes: list[int] = linear_layer_sizes  # pragma: no cover
        else:  # pragma: no cover
            self.linear_layer_sizes: list[int] = [10]  # pragma: no cover
        # pragma: no cover
        # Create linear layers # pragma: no cover
        self.linear_layer_sizes.insert(0, 15 * emb_dim)  # pragma: no cover
        self.linear_layer_sizes.append(num_classes)  # pragma: no cover
        # pragma: no cover
        # Put linear layers into Sequential module # pragma: no cover
        lls = []  # pragma: no cover
        for i in range(len(self.linear_layer_sizes) - 1):  # pragma: no cover
            lls.append(nn.Linear(self.linear_layer_sizes[i], self.linear_layer_sizes[i + 1]))  # pragma: no cover
            lls.append(self.act)  # pragma: no cover
            lls.append(self.drop)  # pragma: no cover
        # pragma: no cover
        self.linears = nn.Sequential(*lls)  # pragma: no cover

    # pragma: no cover
    def forward(self, x):  # pragma: no cover
        x = self.emb(x)  # pragma: no cover
        x = torch.flatten(x, 1, -1)  # pragma: no cover
        x = self.linears(x)  # pragma: no cover
        return x  # pragma: no cover

    # pragma: no cover
    @staticmethod  # pragma: no cover
    def embedding(seq):  # pragma: no cover
        return np.asarray([aa_inv[aa] for aa in seq])  # pragma: no cover

    # pragma: no cover
    class SiteDataSet(torch.utils.data.Dataset):  # pragma: no cover
        def __init__(self, X, y):  # pragma: no cover
            self.X = X  # pragma: no cover
            self.y = y  # pragma: no cover

        # pragma: no cover
        def __len__(self):  # pragma: no cover
            return len(self.y)  # pragma: no cover

        # pragma: no cover
        def __getitem__(self, idx):  # pragma: no cover
            return self.X[idx], self.y[idx]  # pragma: no cover

    # pragma: no cover
    @staticmethod  # pragma: no cover
    def format_data(*fns, kin_fam_grp, sample_size=None):  # pragma: no cover
        dataloaders: list[torch.utils.data.DataLoader] = []  # pragma: no cover
        kin_to_grp = pd.read_csv(kin_fam_grp)[["Kinase", "Uniprot", "Group"]]  # pragma: no cover
        kin_to_grp["Symbol"] = (  # pragma: no cover
            kin_to_grp["Kinase"].apply(lambda x: re.sub(r"[\(\)\*]", "", x))
            + "|"
            + kin_to_grp["Uniprot"]  # pragma: no cover
        )  # pragma: no cover
        kin_to_grp = kin_to_grp.set_index("Symbol").to_dict()["Group"]  # pragma: no cover
        grp_to_idx = {grp: i for i, grp in enumerate(sorted(list(set(kin_to_grp.values()))))}  # pragma: no cover
        for fn in fns:  # pragma: no cover
            dff = pd.read_csv(fn)[["Site Sequence", "Gene Name of Kin Corring to Provided Sub Seq"]]  # pragma: no cover
            df = dff.values  # pragma: no cover
            X = []  # pragma: no cover
            y = [grp_to_idx[kin_to_grp[y]] for y in df[:, 1]]  # pragma: no cover
            for seq in df[:, 0]:  # pragma: no cover
                X.append(DeepSC.embedding(seq))  # pragma: no cover
            Xnd: torch.Tensor = torch.from_numpy(np.asarray(X).astype(int))  # pragma: no cover
            ynd: torch.Tensor = torch.from_numpy(np.asarray(y).astype(int))  # pragma: no cover
            np.random.seed(42)  # pragma: no cover
            dataloaders.append(  # pragma: no cover
                torch.utils.data.DataLoader(
                    DeepSC.SiteDataSet(Xnd, ynd), batch_size=10000, shuffle=True
                )  # pragma: no cover
            )  # pragma: no cover
        # pragma: no cover
        return tuple(dataloaders)  # pragma: no cover

    # pragma: no cover
    @staticmethod  # pragma: no cover
    def get_SC(  # pragma: no cover
        train_file,  # pragma: no cover
        val_file,  # pragma: no cover
        kin_fam_grp_file,  # pragma: no cover
    ):  # pragma: no cover
        model = DeepSC(emb_dim=4, linear_layer_sizes=[100])  # pragma: no cover
        interface = NNInterface(
            model, nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=0.01)
        )  # pragma: no cover
        train_loader, val_loader = DeepSC.format_data(
            train_file, val_file, kin_fam_grp=kin_fam_grp_file
        )  # pragma: no cover
        interface.train(train_loader, num_epochs=50, metric="acc", val_dl=val_loader)  # pragma: no cover


# pragma: no cover
# pragma: no cover
class SiteClassifier:  # pragma: no cover
    @staticmethod  # pragma: no cover
    def one_hot_aa_embedding(seq):  # pragma: no cover
        assert len(seq) == seq_len  # pragma: no cover
        patch = np.zeros((len(seq), num_aa))  # pragma: no cover
        aa_dict = {aa: i for i, aa in enumerate(aas)}  # pragma: no cover
        for i, aa in enumerate(seq):  # pragma: no cover
            patch[i, aa_dict[aa]] = 1  # pragma: no cover
        embedding = patch.ravel()  # pragma: no cover
        return embedding  # pragma: no cover

    # pragma: no cover
    @staticmethod  # pragma: no cover
    def format_data(*fns, kin_fam_grp, sample_size=None):  # pragma: no cover
        X_y_tuples: list[tuple[np.ndarray, np.ndarray]] = []  # pragma: no cover
        kin_to_grp = pd.read_csv(kin_fam_grp)[["Kinase", "Uniprot", "Group"]]  # pragma: no cover
        kin_to_grp["Symbol"] = (  # pragma: no cover
            kin_to_grp["Kinase"].apply(lambda x: re.sub(r"[\(\)\*]", "", x))
            + "|"
            + kin_to_grp["Uniprot"]  # pragma: no cover
        )  # pragma: no cover
        kin_to_grp = kin_to_grp.set_index("Symbol").to_dict()["Group"]  # pragma: no cover
        for fn in fns:  # pragma: no cover
            dff = pd.read_csv(fn)[["Site Sequence", "Gene Name of Kin Corring to Provided Sub Seq"]]  # pragma: no cover
            df = dff.values  # pragma: no cover
            X = []  # pragma: no cover
            y = [kin_to_grp[y] for y in df[:, 1]]  # pragma: no cover
            for seq in df[:, 0]:  # pragma: no cover
                X.append(SiteClassifier.one_hot_aa_embedding(seq))  # pragma: no cover
            Xnd: np.ndarray = np.asarray(X).astype(int)  # pragma: no cover
            ynd: np.ndarray = np.asarray(y)  # pragma: no cover
            np.random.seed(42)  # pragma: no cover
            if isinstance(sample_size, int):  # pragma: no cover
                shuffled_ids = np.random.permutation(len(y))[:sample_size]  # pragma: no cover
            else:  # pragma: no cover
                shuffled_ids = np.random.permutation(len(y))  # pragma: no cover
            Xnd = Xnd[shuffled_ids]  # pragma: no cover
            ynd = ynd[shuffled_ids]  # pragma: no cover
            X_y_tuples.append((Xnd, ynd))  # pragma: no cover
        return tuple(X_y_tuples)  # pragma: no cover

    # pragma: no cover
    @staticmethod  # pragma: no cover
    def get_group_classifier(  # pragma: no cover
        train_file,  # pragma: no cover
        kin_fam_grp_file,  # pragma: no cover
        grp_to_cls={},  # pragma: no cover
        gc_hyperparameters={  # pragma: no cover
            "base_classifier": "ensemble.RandomForestClassifier",  # pragma: no cover
            "n_estimators": 10000,  # pragma: no cover
            "max_depth": 3,  # pragma: no cover
        },  # pragma: no cover
    ):  # pragma: no cover
        cpu_count_raw = os.cpu_count()  # pragma: no cover
        if cpu_count_raw is not None:  # pragma: no cover
            cpu_count = max(1, cpu_count_raw // 2)  # pragma: no cover
        else:  # pragma: no cover
            cpu_count = 1  # pragma: no cover
        exec(f"import sklearn.{gc_hyperparameters['base_classifier'].split('.')[0]}")  # pragma: no cover
        hps = {k: v for k, v in gc_hyperparameters.items() if k != "base_classifier"}  # pragma: no cover
        try:  # pragma: no cover
            gc = eval(gc_hyperparameters["base_classifier"])(
                **hps, random_state=42, n_jobs=cpu_count
            )  # pragma: no cover
        except TypeError:  # pragma: no cover
            gc = eval(gc_hyperparameters["base_classifier"])(**hps, n_jobs=cpu_count)  # pragma: no cover
        (tr,) = SiteClassifier.format_data(train_file, kin_fam_grp=kin_fam_grp_file)  # pragma: no cover
        logger.status("Training Group Classifier")  # pragma: no cover
        gc.fit(tr[0], tr[1])  # pragma: no cover
        return gc  # pragma: no cover

    # pragma: no cover
    @staticmethod  # pragma: no cover
    def performance(model, X, y, metric_fn=sklearn.metrics.accuracy_score):  # pragma: no cover
        y_true, y_pred = y, model.predict(X)  # pragma: no cover
        return round(metric_fn(y_true, y_pred), 4)  # pragma: no cover

    # pragma: no cover
    @staticmethod  # pragma: no cover
    def format_data_for_clusters(*fns, site_grp):  # pragma: no cover
        X_y_tuples: list[tuple[np.ndarray, np.ndarray]] = []  # pragma: no cover
        for fn in fns:  # pragma: no cover
            dff = pd.read_csv(fn)["Site Sequence"]  # pragma: no cover
            df = dff.values  # pragma: no cover
            X = []  # pragma: no cover
            y = [site_grp[y] for y in df]  # pragma: no cover
            for seq in df:  # pragma: no cover
                X.append(SiteClassifier.one_hot_aa_embedding(seq))  # pragma: no cover
            Xnd: np.ndarray = np.asarray(X).astype(int)  # pragma: no cover
            ynd: np.ndarray = np.asarray(y)  # pragma: no cover
            np.random.seed(42)  # pragma: no cover
            shuffled_ids = np.random.permutation(len(y))  # pragma: no cover
            Xnd = Xnd[shuffled_ids]  # pragma: no cover
            ynd = ynd[shuffled_ids]  # pragma: no cover
            X_y_tuples.append((Xnd, ynd))  # pragma: no cover
        return tuple(X_y_tuples)  # pragma: no cover

    # pragma: no cover
    @staticmethod  # pragma: no cover
    def get_clustered_classifier(
        train_file, site_to_group, cluster_hps={"n_clusters": 9, "random_state": 42}
    ):  # pragma: no cover
        (tr,) = SiteClassifier.format_data_for_clusters(train_file, site_grp=site_to_group)  # pragma: no cover
        cluster = sklearn.cluster.KMeans(**cluster_hps)  # pragma: no cover
        cluster.fit(tr[0])  # pragma: no cover
        return cluster  # pragma: no cover


# pragma: no cover
# pragma: no cover
if __name__ == "__main__":  # pragma: no cover
    fd = (  # pragma: no cover
        "DeepKS/data/raw_data_31834_formatted_65_26610.csv",  # pragma: no cover
        "DeepKS/data/raw_data_6500_formatted_95_5698.csv",  # pragma: no cover
        "DeepKS/data/raw_data_6406_formatted_95_5616.csv",  # pragma: no cover
    )  # pragma: no cover
    kfg = "DeepKS/data/preprocessing/kin_to_fam_to_grp_826.csv"  # pragma: no cover
    # pragma: no cover
    # deepsc = DeepSC.get_SC(*fd[:2], kin_fam_grp_file=kfg) # pragma: no cover
    site_to_grp = (
        pd.read_csv("DeepKS/data/preprocessing/site_to_group_9108.csv").set_index("Site").to_dict()["Group"]
    )  # pragma: no cover
    clf = SiteClassifier.get_clustered_classifier(
        fd[0], site_to_grp, cluster_hps={"n_clusters": 9, "random_state": 42}
    )  # pragma: no cover
    # pragma: no cover
    tr, vl, te = SiteClassifier.format_data_for_clusters(*fd, site_grp=site_to_grp)  # pragma: no cover
    # clf = SiteClassifier.get_group_classifier(fd[0], kfg, gc_hyperparameters={'base_classifier': 'neighbors.KNeighborsClassifier', 'n_neighbors': 11, 'metric': 'cosine', 'weights': lambda x: 1 / (x + 0.01)}) # pragma: no cover
    # pragma: no cover
    print("Train Acc: ", SiteClassifier.performance(clf, *tr))  # pragma: no cover
    print("Val Acc: ", SiteClassifier.performance(clf, *vl))  # pragma: no cover
    print("Test Acc: ", SiteClassifier.performance(clf, *te))  # pragma: no cover
    cm = sklearn.metrics.confusion_matrix(te[1], clf.predict(te[0]))  # pragma: no cover
    fig, ax = plt.subplots(figsize=(7, 7))  # pragma: no cover
    sklearn.metrics.ConfusionMatrixDisplay(cm, display_labels=sorted(list(set(te[1])))).plot(ax=ax)  # pragma: no cover
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=90) # pragma: no cover
    plt.show()  # pragma: no cover
