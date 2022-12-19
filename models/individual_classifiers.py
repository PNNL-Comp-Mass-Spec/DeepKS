import functools, warnings, scipy, pandas as pd, json, re, os, pathlib, numpy as np, sklearn.model_selection, matplotlib
import json, torch, tqdm, collections
from .main import KinaseSubstrateRelationshipNN
from ..tools.NNInterface import NNInterface
from ..tools.tensorize import gather_data
from ..tools.parse import parsing
from typing import Any, Union

DEL_DECOR = lambda x: re.sub(r"[\(\)\*]", "", x).upper()
MAX_SIZE_DS = 4128


def RAISE_ASSERTION_ERROR(x):
    raise AssertionError(x)

def printdb(*X: Any):
    print("🪲:", *X)

class IndividualClassifiers:
    def __init__(
        self,
        grp_to_model_args: dict[str, dict[str, Union[bool, str, int, float]]],
        grp_to_interface_args: dict[str, dict[str, Union[bool, str, int, float, type]]],
        grp_to_training_args: dict[str, dict[str, Union[bool, str, int, float]]],
        device: str,
        kin_fam_grp_file: str = "../data/preprocessing/kin_to_fam_to_grp_817.csv",
    ):
        for group in grp_to_model_args:
            assert group in grp_to_interface_args, "No interface args for group %s" % group
            assert group in grp_to_training_args, "No training args for group %s" % group
        for group in grp_to_interface_args:
            assert group in grp_to_model_args, "No model args for group %s" % group
            assert group in grp_to_training_args, "No training args for group %s" % group
        for group in grp_to_training_args:
            assert group in grp_to_model_args, "No model args for group %s" % group
            assert group in grp_to_interface_args, "No interface args for group %s" % group

        self.device = torch.device(device)
        self.grp_to_training_args = grp_to_training_args
        self.individual_classifiers = {
            group: KinaseSubstrateRelationshipNN(**grp_to_model_args[group]) for group in grp_to_model_args
        }
        self.grp_to_interface_args = grp_to_interface_args
        gia = grp_to_interface_args
        self.interfaces = {
            grp: (
                NNInterface(
                    model_to_train=self.individual_classifiers[grp],
                    loss_fn=gia[grp]["loss_fn"]()
                    if isinstance(gia[grp]["loss_fn"], type)
                    else RAISE_ASSERTION_ERROR("Loss function must be a class, not an instance"),
                    optim=gia[grp]["optim"](
                        self.individual_classifiers[grp].parameters(), lr=self.grp_to_interface_args[grp]["lr"]
                    )
                    if isinstance(gia[grp]["optim"], type)
                    else RAISE_ASSERTION_ERROR("Optimizer must be a class, not an instance"),
                    inp_size=None,
                    inp_types=None,
                    model_summary_name=None,
                    device=self.device,
                )
            )
            for grp in gia
        }
        self.default_tok_dict = json.load(open("./json/tok_dict.json"))
        self.kin_symbol_to_grp = pd.read_csv(kin_fam_grp_file)
        symbol_to_grp = pd.read_csv(kin_fam_grp_file)
        symbol_to_grp["Symbol"] = symbol_to_grp["Kinase"].apply(DEL_DECOR) + "|" + symbol_to_grp["Uniprot"]
        self.symbol_to_grp_dict = symbol_to_grp.set_index("Symbol").to_dict()["Group"]

    def _run_dl_core(self, which_groups: list[str], Xy_formatted_input_file: str):
        which_groups.sort()
        Xy = pd.read_csv(Xy_formatted_input_file)
        Xy["Group"] = [self.symbol_to_grp_dict[x] for x in Xy["lab_name"].apply(DEL_DECOR)]
        group_df: dict[str, pd.DataFrame] = {group: Xy[Xy["Group"] == group] for group in which_groups}
        for group in tqdm.tqdm(which_groups, desc="Group Progress", leave=False, position=0):
            yield group, group_df[group]

    def train(
        self,
        which_groups: list[str],
        Xy_formatted_train_file: str,
        Xy_formatted_val_file: str,
    ):
        gen_train = self._run_dl_core(which_groups, Xy_formatted_train_file)
        gen_val = self._run_dl_core(which_groups, Xy_formatted_val_file)
        for (group_tr, partial_group_df_tr), (group_vl, partial_group_df_vl) in zip(gen_train, gen_val):
            assert group_tr == group_vl, "Group mismatch: %s != %s" % (group_tr, group_vl)
            (train_loader, _, _, _), _ = gather_data(
                partial_group_df_tr,
                trf=1,
                vf=0,
                tuf=0,
                tef=0,
                tokdict=self.default_tok_dict,
                train_batch_size=self.grp_to_interface_args[group_tr]['batch_size'],
                n_gram=self.grp_to_interface_args[group_tr]['n_gram'] if isinstance(self.grp_to_interface_args[group_tr]['n_gram'], int) else RAISE_ASSERTION_ERROR('n_gram is not an integer.'),
                device=self.device,
                maxsize=MAX_SIZE_DS
            )
            (_, val_loader, _, _), _ = gather_data(
                partial_group_df_vl,
                trf=0,
                vf=1,
                tuf=0,
                tef=0,
                tokdict=self.default_tok_dict,
                n_gram=self.grp_to_interface_args[group_tr]['n_gram'] if isinstance(self.grp_to_interface_args[group_tr]['n_gram'], int) else RAISE_ASSERTION_ERROR('n_gram is not an integer.'),
                device=self.device, 
                maxsize=MAX_SIZE_DS
            )
            self.interfaces[group_tr].inp_size = self.interfaces[group_tr].get_input_size(train_loader)
            self.interfaces[group_tr].inp_types = self.interfaces[group_tr].get_input_types(train_loader)
            self.interfaces[group_tr].model_summary_name = (
                str("../architectures/" + self.grp_to_interface_args[group_tr]["model_summary_name"]) + "-" + group_tr.upper()
            )
            self.interfaces[group_tr].write_model_summary()
            self.interfaces[group_tr].train(
                train_loader,
                val_dl=val_loader,
                **self.grp_to_training_args[group_tr],
                extra_description="(Group: %s)" % group_tr.upper()
            )

    def evaluate(
        self,
        which_groups: list[str],
        Xy_formatted_input_file: str,
        evaluation_kwargs={"verbose": False, "savefile": False, "metric": "acc"},
    ):
        gen_te = self._run_dl_core(which_groups, Xy_formatted_input_file)
        for (group_tr, partial_group_df_te) in gen_te:
            (_, _, _, test_loader), _ = gather_data(
                partial_group_df_te,
                trf=0,
                vf=0,
                tuf=0,
                tef=1,
                tokdict=self.default_tok_dict,
                n_gram=self.grp_to_interface_args[group_tr]['n_gram'] if isinstance(self.grp_to_interface_args[group_tr]['n_gram'], int) else RAISE_ASSERTION_ERROR('n_gram is not an integer.'),
                device=self.device, 
                maxsize=MAX_SIZE_DS
            )
            assert 'text' not in evaluation_kwargs, "Should not specify `text` output text in `evaluation_kwargs`."
            self.interfaces[group_tr].test(test_loader, text = f"Test Accuracy of the model (Group = {group_tr})", **evaluation_kwargs)


if __name__ == "__main__":
    args = parsing()
    train_filename = args["train"]
    val_filename = args["val"]
    test_filename = args["test"]
    device = args["device"]
    groups: list[str] = [
        "TK",
        "AGC",
        "TKL",
    ]  # pd.read_csv("../data/preprocessing/kin_to_fam_to_grp_817.csv")["Group"].unique().tolist()
    default_grp_to_model_args = {
        "ll1_size": 50,
        "ll2_size": 25,
        "emb_dim": 22,
        "num_conv_layers": 1,
        "dropout_pr": 0.4,
        "site_param_dict": {"kernels": [8], "out_lengths": [8], "out_channels": [20]},
        "kin_param_dict": {"kernels": [100], "out_lengths": [8], "out_channels": [20]},
    }

    default_grp_to_interface_args = {
        "loss_fn": torch.nn.BCEWithLogitsLoss,
        "optim": torch.optim.Adam,
        "model_summary_name": "../architectures/architecture (IC-XX).txt",
        "lr": 0.003,
        "batch_size": 64,
        device: device,
        "n_gram": 1,
    }

    default_training_args = {
        "lr_decay_amount": 0.7,
        "lr_decay_freq": 3,
        "num_epochs": 5,
        "metric": 'roc'
    }

    fat_model = IndividualClassifiers(
        grp_to_model_args={group: default_grp_to_model_args for group in groups},
        grp_to_interface_args={group: default_grp_to_interface_args for group in groups},
        grp_to_training_args={group: default_training_args for group in groups},
        device=device,
        kin_fam_grp_file="../data/preprocessing/kin_to_fam_to_grp_817.csv",
    )

    fat_model.train(which_groups=groups, Xy_formatted_train_file=train_filename, Xy_formatted_val_file=val_filename)
    print("==== EVALUATION ====")
    fat_model.evaluate(which_groups=groups, Xy_formatted_input_file=test_filename)
