from __future__ import annotations
import collections

if __name__ == "__main__":
    from ..splash.write_splash import write_splash

    write_splash("nn_trainer")
    print("Progress: Loading Modules", flush=True)

# import cProfile
# pr = cProfile.Profile()
# pr.enable()
import pandas as pd, json, re, torch, tqdm, torch.utils, io, psutil, numpy as np
import torch.utils.data, cloudpickle as pickle, pickle as orig_pickle, pstats  # type: ignore
from .main import KinaseSubstrateRelationshipNN
from ..tools.NNInterface import NNInterface
from ..tools.tensorize import gather_data
from ..tools.parse import parsing
from ..tools import file_names
from typing import Callable, Generator, Union, Tuple
from pprint import pprint  # type: ignore
from termcolor import colored

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
memory_multiplier = 2**6
EVAL_BATCH_SIZE = 0


def RAISE_ASSERTION_ERROR(x):
    raise AssertionError(x)


#### ===================================================================== ####


class IndividualClassifiers:
    def __init__(
        self,
        grp_to_model_args: dict[str, dict[str, Union[bool, str, int, float, Callable, type]]],
        grp_to_interface_args: dict[str, dict[str, Union[bool, str, int, float, type]]],
        grp_to_training_args: dict[str, dict[str, Union[bool, str, int, float, Callable, type]]],
        device: str,
        args: dict[str, Union[str, None]],
        groups: list[str],
        kin_fam_grp_file: str = "../data/preprocessing/kin_to_fam_to_grp_826.csv",
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

        self.args = args
        self.groups = groups
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

        (
            self.default_tok_dict,
            self.kin_symbol_to_grp,
            self.symbol_to_grp_dict,
        ) = IndividualClassifiers.get_symbol_to_grp_dict(kin_fam_grp_file)

    @staticmethod
    def get_symbol_to_grp_dict(kin_fam_grp_file: str):
        default_tok_dict = json.load(open("./json/tok_dict.json", "r"))
        kin_symbol_to_grp = pd.read_csv(kin_fam_grp_file)
        symbol_to_grp = pd.read_csv(kin_fam_grp_file)
        symbol_to_grp["Symbol"] = symbol_to_grp["Kinase"].apply(DEL_DECOR) + "|" + symbol_to_grp["Uniprot"]
        symbol_to_grp_dict = symbol_to_grp.set_index("Symbol").to_dict()["Group"]
        return default_tok_dict, kin_symbol_to_grp, symbol_to_grp_dict

    def _run_dl_core(
        self,
        which_groups: list[str],
        Xy_formatted_input_file: str,
        pred_groups: Union[None, dict[str, str]] = None,
        tqdm_passthrough: list[tqdm.tqdm] = [],
        cartesian_product: bool = False,
    ):
        which_groups_ordered = collections.OrderedDict(sorted({x: -1 for x in which_groups}.items()))
        Xy: Union[pd.DataFrame, dict]
        if not cartesian_product:
            Xy = pd.read_csv(Xy_formatted_input_file)
        else:
            Xy = json.load(open(Xy_formatted_input_file))
        if pred_groups is None:  # Training
            print("Warning: Using ground truth groups. (Normal for training)")
            Xy["Group"] = [self.symbol_to_grp_dict[x] for x in Xy["orig_lab_name"].apply(DEL_DECOR)]
        else:  # Evaluation CHECK
            if "orig_lab_name" in Xy.columns if isinstance(Xy, pd.DataFrame) else "orig_lab_name" in Xy.keys():  # Test
                Xy["Group"] = [pred_groups[y] for y in [DEL_DECOR(x) for x in Xy["orig_lab_name"]]]
            else:  # Prediction
                Xy["Group"] = [pred_groups[x] for x in Xy["lab_name"].apply(DEL_DECOR)]
        group_df: dict[str, Union[pd.DataFrame, dict]]
        if not cartesian_product:
            assert isinstance(Xy, pd.DataFrame)
            group_df = {group: Xy[Xy["Group"] == group] for group in which_groups_ordered}
        else:
            group_df = {}
            for group in tqdm.tqdm(which_groups_ordered, colour='cyan', leave=False, desc=colored("Info: Formatting data for each group", "cyan")):
                group_df_inner = {}
                put_in_indices = [i for i, x in enumerate(Xy["Group"]) if x == group]

                group_df_inner['lab_name'] = Xy['lab_name']
                group_df_inner['orig_lab_name'] = [Xy['orig_lab_name'][i] for i in put_in_indices]
                group_df_inner['lab'] = [Xy['lab'][i] for i in put_in_indices]
                group_df_inner['seq'] = Xy['seq']
                group_df_inner['pair_id'] = [] # [Xy['pair_id'][j] for j in range(i, i + len(Xy['seq'])) for i in put_in_indices]
                for i in put_in_indices:
                    group_df_inner['pair_id'] += Xy['pair_id'][i*len(Xy['seq']):(i+1)*len(Xy['seq'])]
                group_df_inner['class'] = Xy['class']
                group_df_inner['num_seqs'] = ['N/A']
                
                group_df[group] = group_df_inner

        for group in (
            pb := tqdm.tqdm(
                which_groups_ordered.keys(),
                leave=True,
                position=1,
                desc=colored(f"Overall Group Evaluation Progress", "cyan"),
                colour="cyan"
            )
        ):  # Do only if Verbose
            if len(tqdm_passthrough) == 1:
                tqdm_passthrough[0] = pb
            yield group, group_df[group]
        print("\r", end="\r")

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
        pred_groups: Union[None, dict[str, str]] = None,
        info_dict_passthrough={},
    ) -> Generator[Tuple[str, torch.utils.data.DataLoader], Tuple[str, pd.DataFrame], None]:
        assert len(info_dict_passthrough) == 0, "Info dict passthrough must be empty for passing in"
        tqdm_passthrough = [None]
        gen_te = self._run_dl_core(
            which_groups,
            Xy_formatted_input_file,
            pred_groups=pred_groups,
            tqdm_passthrough=tqdm_passthrough, # type: ignore
            cartesian_product=evaluation_kwargs["cartesian_product"]
            if "cartesian_product" in evaluation_kwargs
            else False,
        )
        count = 0
        for (group_te, partial_group_df_te) in gen_te:
            if len(partial_group_df_te) == 0:
                print("Info: No inputs to evaluate for group =", group_te)
                continue
            ng = self.grp_to_interface_args[group_te]["n_gram"]
            assert isinstance(ng, int), "N-gram must be an integer"
            for (_, _, _, test_loader), info_dict in gather_data(
                partial_group_df_te,
                trf=0,
                vf=0,
                tuf=0,
                tef=1,
                tokdict=self.default_tok_dict,
                n_gram=ng,
                device=self.device
                if "predict_mode" not in evaluation_kwargs or not evaluation_kwargs["predict_mode"]
                else evaluation_kwargs["device"],
                maxsize=MAX_SIZE_DS,
                eval_batch_size= 2**10 if evaluation_kwargs["device"] == "cpu" else 2**16,
                cartesian_product=evaluation_kwargs["cartesian_product"]
                if "cartesian_product" in evaluation_kwargs
                else False,
                tqdm_passthrough=tqdm_passthrough
            ):
                info_dict_passthrough[group_te] = info_dict
                info_dict_passthrough['on_chunk'] = info_dict['on_chunk']
                info_dict_passthrough['total_chunks'] = info_dict['total_chunks']
                assert "text" not in evaluation_kwargs, "Should not specify `text` output text in `evaluation_kwargs`."
                if "predict_mode" not in evaluation_kwargs or not evaluation_kwargs["predict_mode"]:
                    self.interfaces[group_te].test(
                        test_loader, text=f"Test Accuracy of the model (Group = {group_te})", **evaluation_kwargs
                    )
                assert test_loader is not None
                count += 1
                yield group_te, test_loader
                # if len(tqdm_passthrough) == 1:
                #     assert isinstance(tqdm_passthrough[0], tqdm.tqdm)
                #     tqdm_passthrough[0].write("\r" + " " * os.get_terminal_size()[0], end="\r")
        if count > 0:
            pass
        else:
            raise AssertionError("`evaluate` did not iterate through any groups!")

    @staticmethod
    def save_all(individualClassifiers: IndividualClassifiers, path):
        f = open(path, "wb")
        torch.save(individualClassifiers, f)

    @staticmethod
    def load_all(path, device=None) -> IndividualClassifiers:
        with open(path, "rb") as f:
            if device is None or "cuda" in device:
                ic: IndividualClassifiers = pickle.load(f)
                ic.individual_classifiers = {k: v.to(device) for k, v in ic.individual_classifiers.items()}
                return ic
            else:  # Workaround from https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219
                assert device == "cpu"

                class CPU_Unpickler(orig_pickle.Unpickler):
                    def find_class(self, module, name):
                        if module == "torch.storage" and name == "_load_from_bytes":
                            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
                        else:
                            return super().find_class(module, name)

                return CPU_Unpickler(f).load()

    def roc_evaluation(self, new_args, pred_groups, true_groups, predict_mode):
        if "test" in new_args:
            test_filename = new_args["test"]
        elif "test_json" in new_args:
            test_filename = new_args["test_json"]
        else:
            raise AssertionError("Must specify either `test` or `test_json` in `new_args`.")
        if "load_include_eval" in new_args and new_args["load_include_eval"] is None and not predict_mode:
            grp_to_info_pass_through_info_dict = {}
            grp_to_loaders = {
                grp: loader
                for grp, loader in self.evaluate(
                    which_groups=self.groups,
                    Xy_formatted_input_file=test_filename,
                    pred_groups=pred_groups,
                    info_dict_passthrough=grp_to_info_pass_through_info_dict,
                )
            }

            # if False:
            #     pickled_version = pickle.load(open("/people/druc594/ML/DeepKS/bin/saved_state_dicts/test_grp_to_loaders_2023-01-11T17:49:21.981786.pkl", "rb"))
            #     unfurled_pickle = {g: (l.dataset.class_.tolist(), l.dataset.data.tolist(), l.dataset.target.tolist()) for g, l in pickled_version.items()}

            # unfurled_local = {g: (l.dataset.class_.tolist(), l.dataset.data.tolist(), l.dataset.target.tolist()) for g, l in grp_to_loaders.items()} # type: ignore

            print("Progress: ROC")
            NNInterface.get_combined_rocs_from_individual_models(
                self.interfaces,
                grp_to_loaders,
                f"../images/Evaluation and Results/ROC_{file_names.get()}",
                retain_evals=self.evaluations,
                grp_to_loader_true_groups={
                    grp: [true_groups[x] for x in grp_to_info_pass_through_info_dict[grp]["orig_symbols_order"]["test"]]
                    for grp in grp_to_info_pass_through_info_dict
                },
            )

        elif not predict_mode:
            print("Progress: ROC")
            NNInterface.get_combined_rocs_from_individual_models(
                savefile=f"../bin/saved_state_dicts/indivudial_classifiers_{file_names.get()}"
                + file_names.get()
                + ".pkl",
                from_loaded=self.evaluations,
            )
        else:
            all_predictions_outputs = {}
            info_dict_passthrough = {}
            for grp, loader in self.evaluate(
                which_groups=list(pred_groups.values()),
                Xy_formatted_input_file=test_filename,
                pred_groups=pred_groups,
                evaluation_kwargs={
                    "predict_mode": True,
                    "device": new_args["device"],
                    "cartesian_product": "test_json" in new_args,
                },
                info_dict_passthrough=info_dict_passthrough,
            ):
                # print(f"Progress: Predicting for {grp}") # TODO: Only if verbose
                jumbled_predictions = self.interfaces[grp].predict(
                    loader,int(info_dict_passthrough['on_chunk'] + 1), int(info_dict_passthrough['total_chunks']),cutoff=0.5, device=new_args["device"], group=grp
                )  # TODO: Make adjustable cutoff
                del loader
                if "cuda" in str(self.device):
                    torch.cuda.empty_cache()
                new_info = info_dict_passthrough[grp]["PairIDs"]["test"]
                all_predictions_outputs.update(
                    {
                        pair_id: (jumbled_predictions[0][i], jumbled_predictions[1][i], jumbled_predictions[2][i])
                        for pair_id, i in zip(new_info, range(len(new_info)))
                    }
                )

            pred_items = sorted(
                all_predictions_outputs.items(), key=lambda x: int(re.sub("Pair # ([0-9]+)", "\\1", x[0]))
            )  # Pair # {i}
            return pred_items  # TODO: Enable saving

        if self.args["s"]:
            print("Progress: Saving State to Disk")
            IndividualClassifiers.save_all(
                self, f"../bin/saved_state_dicts/indivudial_classifiers_{file_names.get()}" + ".pkl"
            )


def main():
    print("Progress: Parsing Args")
    args = parsing()
    if args["train"] is not None:  # AKA, not loading from pickle
        print("Progress: Preparing Training Data")
        train_filename = args["train"]
        val_filename = args["val"]
        device = args["device"]
        assert device is not None
        # groups: list[str] = [
        #     "TK",
        #     "AGC",
        #     "TKL",
        # ]
        groups: list[str] = (
            pd.read_csv("../data/preprocessing/kin_to_fam_to_grp_826.csv")["Group"].unique().tolist()
        )  # FIXME - all - make them configurable
        default_grp_to_model_args = {
            "ll1_size": 50,
            "ll2_size": 25,
            "emb_dim": 22,
            "num_conv_layers": 1,
            "dropout_pr": 0.4,
            "site_param_dict": {"kernels": [8], "out_lengths": [8], "out_channels": [20]},
            "kin_param_dict": {"kernels": [100], "out_lengths": [8], "out_channels": [20]},
        }

        default_grp_to_interface_args: dict[str, Union[type, str, float, int]] = {
            "loss_fn": torch.nn.BCEWithLogitsLoss,
            "optim": torch.optim.Adam,
            "model_summary_name": "../architectures/architecture (IC-XX).txt",
            "lr": 0.003,
            "batch_size": 64,
            device: device,
            "n_gram": 1,
        } 

        default_training_args = {"lr_decay_amount": 0.7, "lr_decay_freq": 3, "num_epochs": 5, "metric": "roc"}
        assert device is not None
        fat_model = IndividualClassifiers(
            grp_to_model_args={group: default_grp_to_model_args for group in groups},
            grp_to_interface_args={group: default_grp_to_interface_args for group in groups}, # type: ignore # FIXME
            grp_to_training_args={group: default_training_args for group in groups},
            device=device,
            args=args,
            groups=groups,
            kin_fam_grp_file="../data/preprocessing/kin_to_fam_to_grp_826.csv",
        )
        print("Progress: About to Train")
        assert val_filename is not None
        fat_model.train(
            which_groups=groups, Xy_formatted_train_file=train_filename, Xy_formatted_val_file=val_filename
        )
    else:  # AKA, loading from file
        fat_model = IndividualClassifiers.load_all(
            args["load"] if args["load_include_eval"] is None else args["load_include_eval"]
        )
        groups = list(fat_model.interfaces.keys())
    if args["load_include_eval"] is None and args["train"] is None:
        assert fat_model.args["test"] is not None
        grp_to_loaders = {
            grp: loader
            for grp, loader in fat_model.evaluate(
                which_groups=groups, Xy_formatted_input_file=fat_model.args["test"]
            )
        }
        # Working dataloaders
        with open(
            f"../bin/saved_state_dicts/test_grp_to_loaders_{file_names.get()}.pkl", "wb"
        ) as f:
            pickle.dump(grp_to_loaders, f)
        # raise RuntimeError("Done")
        print("Progress: ROC")
        NNInterface.get_combined_rocs_from_individual_models(
            fat_model.interfaces,
            grp_to_loaders,
            f"../images/Evaluation and Results/ROC_indiv_{file_names.get()}",
            retain_evals=fat_model.evaluations,
        )
    elif args["train"] is None:
        print("Progress: ROC")
        NNInterface.get_combined_rocs_from_individual_models(
            savefile=f"../images/Evaluation and Results/ROC_indiv_{file_names.get()}",
            from_loaded=fat_model.evaluations,
        )
    if args["s"]:
        print("Progress: Saving State to Disk")
        IndividualClassifiers.save_all(
            fat_model, f"../bin/indivudial_classifiers_{file_names.get()}.pkl"
        )


if __name__ == "__main__":
    main()
