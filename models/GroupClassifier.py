from __future__ import annotations
import numpy as np, pandas as pd, abc, warnings, re, collections, pickle, os, pathlib
from typing import Union
from termcolor import colored

AA = set(list("ACDEFGHIKLMNPQRSTVWXY"))

from ..config.root_logger import get_logger

logger = get_logger()
if __name__ == "__main__":
    logger.status("Loading Modules")

join_first = lambda levels, x: os.path.join(pathlib.Path(__file__).parent.resolve(), *[".."] * levels, x)


def smart_save_gc(group_classifier: GroupClassifier):
    bin_ = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), "bin")
    max_version = 0
    for file in os.listdir(bin_):
        if v := re.search(r"(UNITTESTVERSION|)deepks_gc_weights\.((|-)\d+)\.cornichon", file):
            max_version = max(max_version, int(v.group(2)) + 1)
    save_path = os.path.join(bin_, f"deepks_gc_weights.{max_version}.cornichon")
    logger.status(f"Serializing and Saving Group Classifier to Disk. ({save_path})")
    with open(save_path, "wb") as f:
        pickle.dump(group_classifier, f)


class GCPrediction(str):
    pass


class GroupClassifier(abc.ABC):
    @abc.abstractmethod
    def __init__(self, sequences: list[str], ground_truth: list[str]) -> None:
        self.sequences = sequences
        self.ground_truth = ground_truth

        self.sequences_to_ground_truths = dict(zip(self.sequences, self.ground_truth))

        self.all_groups = list(set(self.ground_truth))

    @staticmethod
    @abc.abstractmethod
    def get_ground_truth(self_: GroupClassifier, X: Union[np.ndarray, list[str]]) -> list[GCPrediction]:
        pass

    @staticmethod
    @abc.abstractmethod
    def simulated_predict(
        self_: GroupClassifier, X: Union[np.ndarray, list[str]], simulated_acc: float = 0.8
    ) -> list[GCPrediction]:
        pass

    @staticmethod
    @abc.abstractmethod
    def predict(self_, X: Union[np.ndarray, list[str]]) -> list[GCPrediction]:
        pass


class SiteGroupClassifier(GroupClassifier, abc.ABC):
    pass


class KinGroupClassifier(GroupClassifier, abc.ABC):
    pass


class PseudoSiteGroupClassifier(SiteGroupClassifier):
    def __init__(self, sequences, ground_truth) -> None:
        super().__init__(sequences, ground_truth)

    @staticmethod
    def simulated_predict(
        self_: GroupClassifier, X: Union[np.ndarray, list[str]], simulated_acc: float = 0.8
    ) -> list[GCPrediction]:
        return []

    @staticmethod
    def is_aa(x: str) -> bool:
        return all([xi in AA for xi in x])

    @staticmethod
    def get_ground_truth(self_: GroupClassifier, X: Union[np.ndarray, list[str]]):
        warnings.warn(colored("Warning: Using ground truth groups. (Normal for training/val/simulated gc)", "yellow"))
        return PseudoSiteGroupClassifier.predict(self_, X)

    @staticmethod
    def predict(self_: GroupClassifier, X: Union[np.ndarray, list[str]]):
        all(PseudoSiteGroupClassifier.is_aa(x) for x in X)
        list_form = [x for x in X]
        assert all(isinstance(x, str) for x in list_form)
        assert all(len(x) == 15 for x in list_form)
        list_form = [x.upper() for x in list_form]
        # stk_aa = {'S', 'T'}
        stk_grp = "NON-TK"
        tk_aa = {"Y"}
        tk_grp = "TK"
        # acceptable_center_aa = set.union(stk_aa, tk_aa)
        # assert all(x[7] in acceptable_center_aa for x in list_form)
        return [GCPrediction(tk_grp) if x[7] in tk_aa else GCPrediction(stk_grp) for x in list_form]

    @staticmethod
    def general_package(train_file: str, kin_fam_grp: str):
        fd = join_first(1, train_file)
        # fd = "/Users/druc594/Library/CloudStorage/OneDrive-PNNL/Desktop/DeepKS_/DeepKS/data/raw_data_31834_formatted_65_26610.csv"
        fddf = pd.read_csv(fd)
        kin_to_grp_df = pd.read_csv(join_first(1, kin_fam_grp))[["Kinase", "Uniprot", "Group"]]
        kin_to_grp_df["Symbol"] = (
            kin_to_grp_df["Kinase"].apply(lambda x: re.sub(r"[\(\)\*]", "", x)) + "|" + kin_to_grp_df["Uniprot"]
        )
        kin_to_grp = kin_to_grp_df.set_index("Symbol").to_dict()["Group"]
        site_to_grp = collections.defaultdict(set)
        for _, r in fddf.iterrows():
            site_to_grp[r["Site Sequence"]].add(kin_to_grp[r["Gene Name of Kin Corring to Provided Sub Seq"]])

        pgc = PseudoSiteGroupClassifier(
            list(site_to_grp.keys()), ["TK" if x.upper()[7] == "Y" else "NON-TK" for x in list(site_to_grp.keys())]
        )
        smart_save_gc(pgc)


if __name__ == "__main__":
    PseudoSiteGroupClassifier.general_package(
        join_first(1, "data/raw_data/raw_data_45176_formatted_65.csv"),
        join_first(1, "data/preprocessing/kin_to_fam_to_grp_826.csv"),
    )
