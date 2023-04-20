from __future__ import annotations

import numpy as np

if __name__ == "__main__":
    from ..splash.write_splash import write_splash

    write_splash("main_nn_trainer")
    print("Progress: Loading Modules", flush=True)

import pandas as pd, json, re, torch, tqdm, torch.utils, io, warnings, dill, argparse, torch.utils.data
import cloudpickle as pickle, socket, pathlib, os, itertools
from .KinaseSubstrateRelationship import KinaseSubstrateRelationshipNN
from ..tools.NNInterface import NNInterface
from ..tools.tensorize import gather_data
from ..tools import file_names
from typing import Callable, Generator, Union, Tuple
from pprint import pprint  # type: ignore
from termcolor import colored
from ..tools.file_names import get as get_file_name
from copy import deepcopy

torch.use_deterministic_algorithms(True)
torch.manual_seed(42)

DEL_DECOR = lambda x: re.sub(r"[\(\)\*]", "", x).upper()
MAX_SIZE_DS = 4128
memory_multiplier = 2**6
EVAL_BATCH_SIZE = 0

join_first = lambda levels, x: os.path.join(pathlib.Path(__file__).parent.resolve(), *[".."] * levels, x)


def RAISE_ASSERTION_ERROR(x):
    raise AssertionError(x)


def smart_save_nn(individual_classifier: IndividualClassifiers):
    bin_ = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), "bin")
    max_version = -1
    for file in os.listdir(bin_):
        if v := re.search(r"(UNITTESTVERSION|)deepks_nn_weights\.((|-)\d+)\.cornichon", file):
            max_version = max(max_version, int(v.group(2)) + 1)
    savepath = os.path.join(bin_, f"deepks_nn_weights.{max_version}.cornichon")
    print(colored(f"Status: Serializing and Saving Neural Networks to Disk. ({savepath})", "green"))
    IndividualClassifiers.save_all(individual_classifier, savepath)


class IndividualClassifiers:
    def __init__(
        self,
        grp_to_model_args: dict[str, dict[str, Union[bool, str, int, float, Callable, type]]],
        grp_to_interface_args: dict[str, dict[str, Union[bool, str, int, float, type]]],
        grp_to_training_args: dict[str, dict[str, Union[bool, str, int, float, Callable, type]]],
        device: str,
        args: dict[str, Union[str, None, list[str]]],
        groups: list[str],
        kin_fam_grp_file: str,
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
            grp: NNInterface(
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
            for i, grp in enumerate(gia)
        }
        self.evaluations: dict[str, dict[str, dict[str, list[Union[int, float]]]]] = (
            {}
        )  # Group -> Tr/Vl/Te -> outputs/labels -> list

        (
            self.default_tok_dict,
            self.kin_symbol_to_grp,
            self.symbol_to_grp_dict,
        ) = IndividualClassifiers.get_symbol_to_grp_dict(kin_fam_grp_file)

    @staticmethod
    def get_symbol_to_grp_dict(kin_fam_grp_file: str):
        with open(join_first(0, "json/tok_dict.json"), "r") as f:
            default_tok_dict = json.load(f)
        kin_symbol_to_grp = pd.read_csv(join_first(0, kin_fam_grp_file))
        kin_symbol_to_grp["Symbol"] = kin_symbol_to_grp["Kinase"].apply(DEL_DECOR) + "|" + kin_symbol_to_grp["Uniprot"]
        symbol_to_grp_dict = kin_symbol_to_grp.set_index("Symbol").to_dict()["Group"]
        return default_tok_dict, kin_symbol_to_grp, symbol_to_grp_dict

    @staticmethod
    def _run_dl_core(
        which_groups: list[str],
        Xy_formatted_input_file: str,
        group_classifier=None,
        cartesian_product: bool = False,
        symbol_to_grp_dict: dict = {},
    ):
        which_groups_ordered = sorted(list(set(which_groups)))
        Xy_formatted_input_file = join_first(1, Xy_formatted_input_file)
        Xy: Union[pd.DataFrame, dict]
        if not cartesian_product:
            Xy = pd.read_csv(Xy_formatted_input_file)
        else:
            with open(Xy_formatted_input_file, "r") as f:
                Xy = json.load(f)
        if group_classifier is None or symbol_to_grp_dict:  # Training
            print(colored("Warning: Using ground truth groups. (Normal for training/val/simulated GC)", "yellow"))
            Xy["Group"] = [
                symbol_to_grp_dict[x] for x in Xy["Gene Name of Kin Corring to Provided Sub Seq"].apply(DEL_DECOR)
            ]
        else:  # Evaluation CHECK
            if (
                "Gene Name of Kin Corring to Provided Sub Seq" in Xy.columns
                if isinstance(Xy, pd.DataFrame)
                else "Gene Name of Kin Corring to Provided Sub Seq" in Xy.keys()
            ):  # Test
                Xy["Group"] = (
                    group_classifier.predict([DEL_DECOR(x) for x in Xy["Gene Name of Kin Corring to Provided Sub Seq"]])
                    if hasattr(group_classifier, "predict")
                    else [
                        group_classifier[xx]
                        for xx in [DEL_DECOR(x) for x in Xy["Gene Name of Kin Corring to Provided Sub Seq"]]
                    ]
                )
            else:  # Prediction
                Xy["Group"] = (
                    group_classifier.predict(Xy["Gene Name of Kin Corring to Provided Sub Seq"].apply(DEL_DECOR))
                    if hasattr(group_classifier, "predict")
                    else [
                        group_classifier[xx]
                        for xx in [DEL_DECOR(x) for x in Xy["Gene Name of Kin Corring to Provided Sub Seq"]]
                    ]
                )
        group_df: dict[str, Union[pd.DataFrame, dict]]
        if not cartesian_product:
            Xy["pair_id"] = [f"Pair # {i}" for i in range(len(Xy))]
            assert isinstance(Xy, pd.DataFrame)
            group_df = {group: Xy[Xy["Group"] == group] for group in which_groups_ordered}
        else:
            group_df = {}
            for group in tqdm.tqdm(
                which_groups_ordered,
                colour="cyan",
                leave=False,
                desc=colored("Info: Formatting data for each group", "cyan"),
            ):
                group_df_inner = {}
                put_in_indices = [i for i, x in enumerate(Xy["Group"]) if x == group]

                group_df_inner["Gene Name of Kin Corring to Provided Sub Seq"] = Xy[
                    "Gene Name of Kin Corring to Provided Sub Seq"
                ]
                group_df_inner["Gene Name of Kin Corring to Provided Sub Seq"] = [
                    Xy["Gene Name of Kin Corring to Provided Sub Seq"][i] for i in put_in_indices
                ]
                group_df_inner["Kinase Sequence"] = [Xy["Kinase Sequence"][i] for i in put_in_indices]
                group_df_inner["Site Sequence"] = Xy["Site Sequence"]
                group_df_inner["pair_id"] = (
                    []
                )  # [Xy['pair_id'][j] for j in range(i, i + len(Xy['Site Sequence'])) for i in put_in_indices]
                for i in put_in_indices:
                    group_df_inner["pair_id"] += Xy["pair_id"][
                        i * len(Xy["Site Sequence"]) : (i + 1) * len(Xy["Site Sequence"])
                    ]
                group_df_inner["Class"] = Xy["Class"]
                group_df_inner["Num Seqs in Orig Kin"] = ["N/A"]
                group_df[group] = group_df_inner

        for group in which_groups_ordered:
            yield group, group_df[group]

    def train(
        self,
        which_groups: list[str],
        Xy_formatted_train_file: str,
        Xy_formatted_val_file: str,
    ):
        pass_through_scores = []
        gen_train = self._run_dl_core(which_groups, Xy_formatted_train_file, symbol_to_grp_dict=self.symbol_to_grp_dict)
        gen_val = self._run_dl_core(which_groups, Xy_formatted_val_file, symbol_to_grp_dict=self.symbol_to_grp_dict)
        group_tr = "TK"
        for (group_tr, partial_group_df_tr), (group_vl, partial_group_df_vl) in tqdm.tqdm(
            zip(gen_train, gen_val), desc="Training Group Progress", total=len(set(which_groups))
        ):
            assert group_tr == group_vl, "Group mismatch: %s != %s" % (group_tr, group_vl)
            b = self.grp_to_interface_args[group_tr]["batch_size"]
            ng = self.grp_to_interface_args[group_tr]["n_gram"]
            assert isinstance(b, int), "Batch size must be an integer"
            assert isinstance(ng, int), "N-gram must be an integer"
            (train_loader, _, _, _), _ = list(
                gather_data(
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
            )[0]
            (_, val_loader, _, _), _ = list(
                gather_data(
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
            )[0]
            if len(train_loader) != 0:
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
                    pass_through_scores=pass_through_scores,
                )
            else:
                warnings.warn(
                    f"No data for group {group_tr}, skipping training for this group. Neural network weights will be"
                    " random."
                )
        weighted = sum([x[0] * x[1] for x in pass_through_scores]) / sum([x[1] for x in pass_through_scores])
        print(
            colored(
                (
                    f"Train/Validation Info: Overall Weighted {self.grp_to_training_args[group_tr]['metric']} ---"
                    f" {weighted:3.4f}"
                ),
                "blue",
            )
        )

    def obtain_group_and_loader(
        self,
        which_groups: list[str],
        Xy_formatted_input_file: str,
        evaluation_kwargs={"verbose": False, "savefile": False, "metric": "acc"},
        group_classifier=None,
        info_dict_passthrough={},
        seen_groups_passthrough=[],
        bypass_gc={},
        cartesian_product=False,
    ) -> Generator[Tuple[str, torch.utils.data.DataLoader], Tuple[str, pd.DataFrame], None]:
        assert len(info_dict_passthrough) == 0, "Info dict passthrough must be empty for passing in"
        gen_te = self._run_dl_core(
            which_groups,
            Xy_formatted_input_file,
            group_classifier=group_classifier,
            cartesian_product=bool(
                sum(
                    [
                        evaluation_kwargs["cartesian_product"] if "cartesian_product" in evaluation_kwargs else False,
                        cartesian_product,
                    ]
                )
            ),
            symbol_to_grp_dict=bypass_gc,
        )
        count = 0
        for group_te, partial_group_df_te in gen_te:
            if len(partial_group_df_te) == 0:
                print("Info: No inputs to evaluate for group =", group_te)
                continue
            ng = self.grp_to_interface_args[group_te]["n_gram"]
            assert isinstance(ng, int), "N-gram must be an integer"
            seen_groups_passthrough.append(group_te)
            for (_, _, _, test_loader), info_dict in gather_data(
                partial_group_df_te,
                trf=0,
                vf=0,
                tuf=0,
                tef=1,
                tokdict=self.default_tok_dict,
                n_gram=ng,
                device=(
                    self.device
                    if "predict_mode" not in evaluation_kwargs or not evaluation_kwargs["predict_mode"]
                    else evaluation_kwargs["device"]
                ),
                maxsize=MAX_SIZE_DS,
                eval_batch_size=1,
                cartesian_product=bool(
                    sum(
                        [
                            (
                                evaluation_kwargs["cartesian_product"]
                                if "cartesian_product" in evaluation_kwargs
                                else False
                            ),
                            cartesian_product,
                        ]
                    )
                ),
            ):
                info_dict_passthrough[group_te] = info_dict
                info_dict_passthrough["on_chunk"] = info_dict["on_chunk"]
                info_dict_passthrough["total_chunks"] = info_dict["total_chunks"]
                assert "text" not in evaluation_kwargs, "Should not specify `text` output text in `evaluation_kwargs`."
                assert test_loader is not None
                count += 1
                yield group_te, test_loader
        if count > 0:
            pass
        else:
            raise AssertionError("`evaluate` did not iterate through any groups!")

    @staticmethod
    def save_all(individualClassifiers: IndividualClassifiers, path):
        with open(path, "wb") as f:
            pickle.dump(
                individualClassifiers,
                f,
            )

    @staticmethod
    def load_all(path, target_device="cpu") -> IndividualClassifiers:
        if not isinstance(target_device, str):
            target_device = str(target_device)
        with open(join_first(1, path), "rb") as f:
            if "cuda" in target_device:
                ic: IndividualClassifiers = pickle.load(f)
            elif (
                target_device == "cpu"
            ):  # Workaround from https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219

                class CPU_Unpickler(dill.Unpickler):
                    def find_class(self, module, name):
                        if module == "torch.storage" and name == "_load_from_bytes":
                            unpickler_lambda = lambda b: torch.load(io.BytesIO(b), map_location="cpu")
                            return unpickler_lambda
                        else:
                            return super().find_class(module, name)

                ic = CPU_Unpickler(f).load()
            else:
                raise NotImplementedError("Invalid `target_device`")

            assert (
                ic.__class__.__name__ == "IndividualClassifiers"
            ), f"Loaded object must be an `IndividualClassifiers` object. It is a {type(ic)}"
            ic.individual_classifiers = {k: v.to(target_device) for k, v in ic.individual_classifiers.items()}
            ic.args = {}
            setattr(ic, "repickle_loc", path)
            ic.device = target_device
            for interface in ic.interfaces:
                ic.interfaces[interface].device = torch.device(target_device)
            return ic

    def evaluation(
        self,
        addl_args: dict,
        pred_groups: dict,
        true_groups: dict,
        predict_mode: bool,
        get_emp_eqn: bool = True,
        emp_eqn_kwargs: dict = {"plot_emp_eqn": True, "print_emp_eqn": True},
        cartesian_product: bool = False,
    ) -> Union[None, list, Callable]:
        """Get predictions or ROC curves from model

        Args:
            @arg addl_args: Additional arguments outside of `self.args` to be passed in.
            @arg pred_groups: Dict mapping kinase symbol to predicted group
            @arg true_groups: Dict mapping kinase symbol to ground truth group
            @arg predict_mode: Whether or not to run in pure prediction mode
            @arg get_emp_eqn: Whether or not to get empirical equation mapping raw score to empirical probability
            @arg emp_eqn_kwargs: Arguments for the empirical equation.

        Raises:
            ValueError: If neither `test` nor `test_json` is specified in `addl_args`.

        Returns:
            Union[None, dict[str, Callable], list]: None, if no empirical equation is requested. If empirical
            equation IS requested, contains lambda which maps raw score to empirical probability. If prediction mode, list of prediction items.
        """
        if "test" in addl_args:
            test_filename = addl_args["test"]  # Dataframe-based evaluation
        elif "test_json" in addl_args:
            test_filename = addl_args["test_json"]  # Json-based evaluation
        else:
            raise ValueError("Must specify either `test` or `test_json` in `addl_args`.")

        if predict_mode:
            all_predictions_outputs = {}
            info_dict_passthrough = {}
            for grp, loader in self.obtain_group_and_loader(
                which_groups=list(pred_groups.values()),
                Xy_formatted_input_file=test_filename,
                group_classifier=pred_groups,
                evaluation_kwargs={
                    "predict_mode": True,
                    "device": addl_args["device"],
                    "cartesian_product": "test_json" in addl_args,
                },
                info_dict_passthrough=info_dict_passthrough,
                cartesian_product=cartesian_product,
            ):
                # print(f"Progress: Predicting for {grp}") # TODO: Only if verbose
                jumbled_predictions = self.interfaces[grp].predict(
                    loader,
                    int(info_dict_passthrough["on_chunk"] + 1),
                    int(info_dict_passthrough["total_chunks"]),
                    cutoff=0.5,
                    group=grp,
                )  # TODO: Make adjustable cutoff
                # jumbled_predictions = list[predictions], list[output scores], list[group]
                del loader
                if "cuda" in str(self.device):
                    torch.cuda.empty_cache()
                new_info = info_dict_passthrough[grp]["PairIDs"]["test"]
                try:
                    all_predictions_outputs.update(
                        {
                            pair_id: (
                                jumbled_predictions[0][i],
                                jumbled_predictions[1][i],
                                jumbled_predictions[2][i],
                                jumbled_predictions[3][i],
                                self.__dict__["grp_to_emp_eqn"].get(grp) if get_emp_eqn else None,
                            )
                            for pair_id, i in zip(new_info, range(len(new_info)))
                        }
                    )
                except KeyError:
                    raise AttributeError(
                        "`--convert-raw-to-prob` was set but the attribute `grp_to_emp_eqn` (aka a"
                        " dictionary that maps a kinase group to a function capible of converting raw"
                        " scores to empirical probabilities) was not found in the pretrained neural"
                        " network file. (To change the neural network file, use the `--pre-trained-nn`"
                        " command line option.)"
                    ) from None
            key_lambda = lambda x: int(re.sub("Pair # ([0-9]+)", "\\1", x[0]))
            pred_items = sorted(all_predictions_outputs.items(), key=key_lambda)  # Pair # {i}
            if self.args.get("s"):
                smart_save_nn(self)
            return pred_items

        else:  # Not predict mode
            if addl_args["load_include_eval"] is None:  # Need to eval
                grp_to_info_pass_through_info_dict = {}
                grp_to_loaders = {
                    grp: loader
                    for grp, loader in self.obtain_group_and_loader(
                        which_groups=self.groups,
                        Xy_formatted_input_file=test_filename,
                        group_classifier=pred_groups,
                        info_dict_passthrough=grp_to_info_pass_through_info_dict,
                    )
                }

                print(colored("Status: Creating combined roc from individual models.", "green"))
                NNInterface.get_combined_rocs_from_individual_models(
                    self.interfaces,
                    grp_to_loaders,
                    f"../images/Evaluation and Results/{file_names.get(f'ROC_{len(grp_to_loaders)}', 'pdf')}",
                    retain_evals=self.evaluations,
                    grp_to_loader_true_groups={
                        grp: [
                            true_groups[x]
                            for x in grp_to_info_pass_through_info_dict[grp]["orig_symbols_order"]["test"]
                        ]
                        for grp in grp_to_info_pass_through_info_dict
                        if grp in grp_to_loaders
                    },
                    get_emp_eqn=get_emp_eqn,
                    emp_eqn_kwargs=emp_eqn_kwargs,
                    _ic_ref=self,
                    grp_to_idents={
                        grp: grp_to_info_pass_through_info_dict[grp]["orig_symbols_order"]["test"]
                        for grp in grp_to_loaders
                    },
                )
            #     pick_out_kinase = "HIPK2|Q9H2X6"
            #     # assert pick_out_kinase in true_groups
            #     # pick_out_group = true_groups[pick_out_kinase]
            #     pick_out_group = "CMGC"
            #     if pick_out_kinase == "ABL1|P00519":
            #         assert pick_out_group == "TK"
            #     if pick_out_kinase == "HIPK2|Q9H2X6":
            #         assert pick_out_group == "CMGC"
            #     print(colored(f"Status: Creating ROC curves stratified by kinase family {pick_out_group}.", "green"))
            #     roc = DeepKS_evaluation.SplitIntoKinasesROC()
            #     scores = self.evaluations[pick_out_group]["test"]["outputs"]
            #     labels = self.evaluations[pick_out_group]["test"]["labels"]
            #     kinase_identities = grp_to_info_pass_through_info_dict[pick_out_group]["orig_symbols_order"]["test"]
            #     # assert pick_out_kinase in kinase_identities
            #     roc.make_plot(
            #         scores,
            #         labels,
            #         kinase_identities,
            #         pick_out_group,
            #         plotting_kwargs={
            #             "plot_markers": True,
            #             "plot_unified_line": True,
            #             "jitter_amount": 0,
            #             "diff_by_color": False,
            #             "diff_by_opacity": True,
            #         },
            #     )
            #     roc.save_plot(get_file_name(f"ROC_{pick_out_group}", "pdf"))

            # else:  # Eval already completed
            #     print("Progress: ROC")
            #     NNInterface.get_combined_rocs_from_individual_models(
            #         savefile=f"../bin/saved_state_dicts/individual_classifiers_{file_names.get()}"
            #         + file_names.get()
            #         + ".pkl",
            #         from_loaded=self.evaluations,
            #     )
            if self.args.get("s"):
                smart_save_nn(self)


def main():
    print(colored("Progress: Parsing Args", "green"))
    args = parse_args()
    print(colored("Progress: Preparing Training Data", "green"))
    train_filename = args["train"]
    val_filename = args["val"]
    device = args["device"]
    kin_fam_grp = pd.read_csv(kfg_file := join_first(1, args["kin_fam_grp"]))
    groups: list[str] = list(args["groups"]) if args["groups"] is not None else kin_fam_grp["Group"].unique().tolist()

    assert device is not None

    with open(join_first(0, args["ksr_params"])) as f:
        grp_to_model_args = json.load(f)
        default_grp_to_model_args = grp_to_model_args["default"]
    with open(join_first(0, args["nni_params"])) as f:
        grp_to_interface_args = json.load(f)
        for grp in grp_to_interface_args:
            grp_to_interface_args[grp]["loss_fn"] = eval(str(grp_to_interface_args[grp]["loss_fn"]))
            grp_to_interface_args[grp]["optim"] = eval(str(grp_to_interface_args[grp]["optim"]))
            grp_to_interface_args[grp]["device"] = device
        default_grp_to_interface_args = grp_to_interface_args["default"]

    with open(join_first(0, args["ksr_training_params"])) as f:
        grp_to_training_args = json.load(f)
        default_training_args = grp_to_training_args["default"]

    gtma = {group: grp_to_model_args.get(group, deepcopy(default_grp_to_model_args)) for group in groups}
    gtia = {group: grp_to_interface_args.get(group, deepcopy(default_grp_to_interface_args)) for group in groups}
    gtta = {group: grp_to_training_args.get(group, deepcopy(default_training_args)) for group in groups}

    classifier = IndividualClassifiers(gtma, gtia, gtta, str(device), args, groups, kfg_file)

    print(colored("Status: About to Train", "green"))
    assert val_filename is not None
    classifier.train(
        which_groups=groups,
        Xy_formatted_train_file=join_first(1, train_filename),
        Xy_formatted_val_file=join_first(1, val_filename),
    )

    if args["s"]:
        smart_save_nn(classifier)


def device_eligibility(arg_value):
    try:
        assert bool(re.search("^cuda(:|)[0-9]*$", arg_value)) or bool(re.search("^cpu$", arg_value))
        if "cuda" in arg_value:
            if arg_value == "cuda":
                return arg_value
            cuda_num = int(re.findall("([0-9]+)", arg_value)[0])
            assert 0 <= cuda_num <= torch.cuda.device_count()
    except Exception:
        raise argparse.ArgumentTypeError(
            f"Device '{arg_value}' does not exist on this machine (hostname: {socket.gethostname()}).\n"
            f"Choices are {sorted(set(['cpu']).union([f'cuda:{i}' for i in range(torch.cuda.device_count())]))}."
        )


def parse_args() -> dict[str, Union[str, None, list[str]]]:
    print(colored("Progress: Parsing Arguments", "green"))

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        type=str,
        help="Specify device. Choices are {'cpu', 'cuda:<gpu number>'}.",
        metavar="<device>",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--train", type=str, help="Specify train file name", required=True, metavar="<train_file_name.csv>"
    )
    parser.add_argument(
        "--val", type=str, help="Specify validation file name", required=True, metavar="<val_file_name.csv>"
    )

    parser.add_argument(
        "--ksr-params",
        type=str,
        help="Specify Kinase Substrate Relationship hyperparameters file name",
        required=False,
        default=join_first(0, "KSR_params.json"),
        metavar="<ksr_params.json>",
    )

    parser.add_argument(
        "--ksr-training-params",
        type=str,
        help="Specify Kinase Substrate Relationship training options file name",
        required=False,
        default=join_first(0, "KSR_training_params.json"),
        metavar="<ksr_params.json>",
    )

    parser.add_argument(
        "--nni-params",
        type=str,
        help="Specify Nerual Net Interface options file name",
        required=False,
        default=join_first(0, "NNI_params.json"),
        metavar="<ksr_params.json>",
    )

    parser.add_argument(
        "--kin-fam-grp",
        type=str,
        help="Specify Kinase-Family-Group file name",
        required=False,
        default=join_first(1, "data/preprocessing/kin_to_fam_to_grp_826.csv"),
        metavar="<kin_fam_grp.csv>",
    )

    parser.add_argument(
        "--groups",
        type=str,
        nargs="+",
        help="Specify groups to train on",
        required=False,
        metavar="<group> <group> ...",
    )

    parser.add_argument("-s", action="store_true", help="Include to save state", required=False)

    try:
        args = vars(parser.parse_args())
    except Exception as e:
        print(e)
        exit(1)

    device_eligibility(args["device"])

    for fn in ["train", "val", "ksr_params", "ksr_training_params", "nni_params"]:
        f = str(args[fn])
        try:
            assert "formatted" in f or f.endswith(
                ".json"
            ), "'formatted' is not in the train or filename. Did you select the correct file?"
        except AssertionError as e:
            warnings.warn(str(e), UserWarning)
        assert os.path.exists(join_first(1, f)), f"Input file '{join_first(1, f)}' does not exist."

    return args


if __name__ == "__main__":
    main()
