from __future__ import annotations

print("Progress: Loading Modules")
# import cProfile
# pr = cProfile.Profile()
# pr.enable()
import pandas as pd, json, re, json, torch, tqdm, torch.utils
import torch.utils.data, datetime, pickle, pstats  # type: ignore
from .main import KinaseSubstrateRelationshipNN
from ..tools.NNInterface import NNInterface
from ..tools.tensorize import gather_data
from ..tools.parse import parsing
from typing import Callable, Generator, Union, Tuple
from pprint import pprint  # type: ignore

# s = io.StringIO()
# load_stats = pstats.Stats(pr, stream=s).sort_stats('tottime')
# pr.disable()
# print("Import Stats...")
# load_stats.print_stats()
# print("\n".join(s.getvalue().split("\n")[:10]))

torch.use_deterministic_algorithms(True)
torch.manual_seed(42)

DEL_DECOR = lambda x: re.sub(r"[\(\)\*]", "", x).upper()
MAX_SIZE_DS = 4128


def RAISE_ASSERTION_ERROR(x):
    raise AssertionError(x)


#### ===================================================================== ####


class IndividualClassifiers:
    def __init__(
        self,
        grp_to_model_args: dict[str, dict[str, Union[bool, str, int, float, Callable]]],
        grp_to_interface_args: dict[str, dict[str, Union[bool, str, int, float, Callable]]],
        grp_to_training_args: dict[str, dict[str, Union[bool, str, int, float, Callable]]],
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
        optims: list[Callable] = []
        loss_fns: list[Callable] = []
        for grp in gia:
            o = gia[grp]["optim"]
            assert isinstance(o, Callable), "Optimizer must be a callable"
            optims.append(o)
            l = gia[grp]["loss_fn"]
            assert isinstance(l, Callable), "Loss function must be a callable"
            loss_fns.append(l)

        self.interfaces = {
            grp: (
                NNInterface(
                    model_to_train=self.individual_classifiers[grp],
                    loss_fn=loss_fns[i](),
                    optim=optims[i](
                        self.individual_classifiers[grp].parameters(), lr=self.grp_to_interface_args[grp]["lr"]
                    ),
                    inp_size=None,
                    inp_types=None,
                    model_summary_name=None,
                    device=self.device,
                )
            )
            for i, grp in enumerate(gia)
        }
        self.evaluations: dict[
            str, dict[str, dict[str, list[Union[int, float]]]]
        ] = {}  # Group -> Tr/Vl/Te -> outputs/labels -> list
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
            b = self.grp_to_interface_args[group_tr]["batch_size"]
            ng = self.grp_to_interface_args[group_tr]["n_gram"]
            assert isinstance(b, int), "Batch size must be an integer"
            assert isinstance(ng, int), "N-gram must be an integer"
            (train_loader, _, _, _), _ = gather_data(
                partial_group_df_tr,
                trf=1,
                vf=0,
                tuf=0,
                tef=0,
                tokdict=self.default_tok_dict,
                train_batch_size=b,
                n_gram=ng,
                device=self.device,
                maxsize=MAX_SIZE_DS,
            )
            (_, val_loader, _, _), _ = gather_data(
                partial_group_df_vl,
                trf=0,
                vf=1,
                tuf=0,
                tef=0,
                tokdict=self.default_tok_dict,
                n_gram=ng,
                device=self.device,
                maxsize=MAX_SIZE_DS,
            )
            self.interfaces[group_tr].inp_size = self.interfaces[group_tr].get_input_size(train_loader)
            self.interfaces[group_tr].inp_types = self.interfaces[group_tr].get_input_types(train_loader)
            msm = self.grp_to_interface_args[group_tr]["model_summary_name"]
            assert isinstance(msm, str), "Model summary name must be a string"
            self.interfaces[group_tr].model_summary_name = msm + "-" + group_tr.upper()
            self.interfaces[group_tr].write_model_summary()
            self.interfaces[group_tr].train(
                train_loader,
                val_dl=val_loader,
                **self.grp_to_training_args[group_tr],
                extra_description="(Group: %s)" % group_tr.upper(),
            )

    def evaluate(
        self,
        which_groups: list[str],
        Xy_formatted_input_file: str,
        evaluation_kwargs={"verbose": False, "savefile": False, "metric": "acc"},
        retain_evals=False,
    ) -> Generator[Tuple[str, torch.utils.data.DataLoader], Tuple[str, pd.DataFrame], None]:
        gen_te = self._run_dl_core(which_groups, Xy_formatted_input_file)
        count = 0
        for (group_te, partial_group_df_te) in gen_te:
            ng = self.grp_to_interface_args[group_te]["n_gram"]
            assert isinstance(ng, int), "N-gram must be an integer"
            (_, _, _, test_loader), _ = gather_data(
                partial_group_df_te,
                trf=0,
                vf=0,
                tuf=0,
                tef=1,
                tokdict=self.default_tok_dict,
                n_gram=ng,
                device=self.device,
                maxsize=MAX_SIZE_DS,
            )
            assert "text" not in evaluation_kwargs, "Should not specify `text` output text in `evaluation_kwargs`."
            self.interfaces[group_te].test(
                test_loader, text=f"Test Accuracy of the model (Group = {group_te})", **evaluation_kwargs
            )
            assert test_loader is not None
            count += 1
            yield group_te, test_loader
        if count > 0:
            pass
        else:
            raise AssertionError("`evaluate` did not iterate through any groups!")

    @staticmethod
    def save_all(individualClassifiers: IndividualClassifiers, path):
        f = open(path, "wb")
        pickle.dump(individualClassifiers, f)

    @staticmethod
    def load_all(path) -> IndividualClassifiers:
        f = open(path, "rb")
        return pickle.load(f)


if __name__ == "__main__":
    print("Progress: Parsing Args")
    args = parsing()
    test_filename = args["test"]
    if args["train"] is not None:  # AKA, not loading from pickle
        print("Progress: Preparing Training Data")
        train_filename = args["train"]
        val_filename = args["val"]
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

        default_training_args = {"lr_decay_amount": 0.7, "lr_decay_freq": 3, "num_epochs": 5, "metric": "roc"}

        fat_model = IndividualClassifiers(
            grp_to_model_args={group: default_grp_to_model_args for group in groups},
            grp_to_interface_args={group: default_grp_to_interface_args for group in groups},
            grp_to_training_args={group: default_training_args for group in groups},
            device=device,
            kin_fam_grp_file="../data/preprocessing/kin_to_fam_to_grp_817.csv",
        )
        print("Progress: About to Train")
        fat_model.train(which_groups=groups, Xy_formatted_train_file=train_filename, Xy_formatted_val_file=val_filename)
    else:  # AKA, loading from file
        fat_model = IndividualClassifiers.load_all(args["load"] if args['load_include_eval'] is None else args['load_include_eval'])
        groups = list(fat_model.interfaces.keys())
    print("Progress: Evaluation")
    if args["load_include_eval"] is None:
        grp_to_loaders = {
            grp: loader for grp, loader in fat_model.evaluate(which_groups=groups, Xy_formatted_input_file=test_filename)
        }
        print("Progress: ROC")
        NNInterface.get_combined_rocs_from_individual_models(
            fat_model.interfaces,
            grp_to_loaders,
            "../images/Evaluation and Results/ROC_indiv_tues_night",
            retain_evals=fat_model.evaluations,
        )
    else:
        print("Progress: ROC")
        NNInterface.get_combined_rocs_from_individual_models(
            savefile = "../images/Evaluation and Results/ROC_indiv_tues_night",
            from_loaded = fat_model.evaluations
        )
    if args["s"]:
        print("Progress: Saving State to Disk")
        IndividualClassifiers.save_all(
            fat_model, "../bin/Saved State Dicts/indivudial_classifiers_tues_night_" + datetime.datetime.now().isoformat() + ".pkl"
        )
