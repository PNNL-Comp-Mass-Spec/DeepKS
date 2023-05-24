"""Contains class definition of `IndividualClassifiers`, which is used to train/validate group-specific neural network classifiers."""

from __future__ import annotations

import numpy as np

if __name__ == "__main__":  # pragma: no cover
    from ..tools.splash.write_splash import write_splash

    write_splash("main_nn_trainer")

from ..config.logging import get_logger

logger = get_logger()
"""Logger for this module"""
if __name__ == "__main__":  # pragma: no cover
    logger.status("Loading Modules...")


import pandas as pd, json, re, torch, tqdm, torch.utils, io, warnings, argparse, torch.utils.data, more_itertools
import socket, pathlib, os, itertools, functools, numpy as np, pickle
from ..tools.NNInterface import NNInterface
from ..tools.tensorize import data_to_tensor
from ..tools.model_utils import KSDataset
from typing import Callable, Generator, Literal, Protocol, Union, Tuple, Any

import pprint
from termcolor import colored
from ..tools.custom_tqdm import CustomTqdm
from copy import deepcopy
from .GroupClassifier import (
    GroupClassifier,
    GCPrediction,
    SiteGroupClassifier,
    KinGroupClassifier,
    PseudoSiteGroupClassifier,
)
from .KSRProtocol import KSR
import __main__

setattr(__main__, "PseudoSiteGroupClassifier", PseudoSiteGroupClassifier)


torch.use_deterministic_algorithms(True)
torch.manual_seed(42)

DEL_DECOR = lambda x: re.sub(r"[\(\)\*]", "", x).upper()
"""Simple lambda to remove parentheses and asterisks from a string and convert it to uppercase."""
MAX_SIZE_DS = 4128
memory_multiplier = 2**6
EVAL_BATCH_SIZE = 0
from ..config.join_first import join_first


def smart_save_nn(individual_classifier: IndividualClassifiers, optional_idx: int | None = None):
    bin_ = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), "bin")
    max_version = -1
    for file in os.listdir(bin_):
        if v := re.search(r"deepks_nn_weights\.((|-)\d+)\.cornichon", file):
            max_version = max(max_version, int(v.group(1)) + 1)
    if optional_idx is not None:
        file_name_numb = optional_idx
    else:
        file_name_numb = max_version
    savepath = os.path.join(bin_, f"deepks_nn_weights.{file_name_numb}.cornichon")
    logger.status(f"Serializing and Saving Neural Networks to Disk. ({savepath})")
    IndividualClassifiers.save_all(individual_classifier, savepath)


class IndividualClassifiers:
    """Class to train and validate group-specific neural network classifiers."""

    def __init__(
        self,
        grp_to_model_args: dict[str, dict[str, Any]],
        grp_to_interface_args: dict[str, dict[str, Any]],
        grp_to_training_args: dict[str, dict[str, Any]],
        device: str,
        args: dict[str, Union[str, None, list[str]]],
        groups: list[str],
    ):
        """Initialize an IndividualClassifiers object.

        Parameters
        ----------
        grp_to_model_args : dict[str, dict[str, Any]]
            A dictionary mapping group names to dictionaries of model-related arguments (hyperparameters).
        grp_to_interface_args : dict[str, dict[str, Any]]
            A dictionary mapping group names to dictionaries of `NNInterface`-related arguments.
        grp_to_training_args : dict[str, dict[str, Any]]
            A dictionary mapping group names to dictionaries of training-related arguments.
        device : str
            The device to train the neural networks on.
        args : dict[str, Union[str, None, list[str]]]
            Should Be Depricated.
        groups : list[str]
            The groups to train neural networks for.
        """
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
        for group in grp_to_model_args:
            exec(f"from .{grp_to_model_args[group]['model_class']} import {grp_to_model_args[group]['model_class']}")
        self.individual_classifiers = {}
        for group in grp_to_model_args:
            clss = eval(grp_to_model_args[group]["model_class"])
            assert isinstance(clss, Callable)
            self.individual_classifiers[group] = eval(grp_to_model_args[group]["model_class"])(
                **{k: v for k, v in grp_to_model_args[group].items() if k != "model_class"}
            )
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
        self.evaluations: dict[
            str, dict[str, dict[str, list[Union[int, float]]]]
        ] = {}  # Group -> Tr/Vl/Te -> outputs/labels -> list

        self.default_tok_dict = {
            "M": 0,
            "E": 1,
            "D": 2,
            "Y": 3,
            "T": 4,
            "K": 5,
            "I": 6,
            "G": 7,
            "V": 8,
            "R": 9,
            "H": 10,
            "Q": 11,
            "A": 12,
            "L": 13,
            "S": 14,
            "P": 15,
            "N": 16,
            "F": 17,
            "C": 18,
            "W": 19,
            "X": 20,
            "<PADDING>": 21,
            "<N-GRAM>": 1,
        }

    @staticmethod
    def get_symbol_to_grp_dict(kin_fam_grp_file: str):
        with open(join_first("json/tok_dict.json", 0, __file__), "r") as f:
            default_tok_dict = json.load(f)
        kin_symbol_to_grp = pd.read_csv(join_first(kin_fam_grp_file, 0, __file__))
        kin_symbol_to_grp["Symbol"] = kin_symbol_to_grp["Kinase"].apply(DEL_DECOR) + "|" + kin_symbol_to_grp["Uniprot"]
        symbol_to_grp_dict = kin_symbol_to_grp.set_index("Symbol").to_dict()["Group"]
        return default_tok_dict, kin_symbol_to_grp, symbol_to_grp_dict

    @staticmethod
    def get_group_dataframes(
        which_groups: list[str],
        Xy_formatted_input_file: str,
        group_classifier: GroupClassifier,
        group_classifier_method: Callable[[GroupClassifier, Union[np.ndarray, list[str]]], list[GCPrediction]],
        cartesian_product: bool = False,
    ):
        which_groups_ordered = sorted(list(set(which_groups)))
        Xy_formatted_input_file = join_first(Xy_formatted_input_file, 1, __file__)
        Xy: Union[pd.DataFrame, dict]
        if not cartesian_product:
            Xy = pd.read_csv(Xy_formatted_input_file)
        else:
            with open(Xy_formatted_input_file, "r") as f:
                Xy = json.load(f)

        if isinstance(group_classifier, SiteGroupClassifier):
            group_by = "site"
        else:
            group_by = "kin"
        if group_by == "site":
            col = "Site Sequence"
        else:
            col = "Gene Name of Provided Kin Seq"
        Xy["Group"] = group_classifier_method(group_classifier, [x for x in Xy[col]])
        if which_groups_ordered == ["All - Use Preds"]:
            which_groups_ordered = sorted(list(set(Xy["Group"])))
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

                group_df_inner["Gene Name of Provided Kin Seq"] = Xy["Gene Name of Provided Kin Seq"]
                group_df_inner["Gene Name of Kin Corring to Provided Sub Seq"] = [
                    Xy["Gene Name of Kin Corring to Provided Sub Seq"][i] for i in put_in_indices
                ]

                if col == "Site Sequence":  # Classifying the sites into groups
                    group_df_inner["Site Sequence"] = [Xy["Site Sequence"][i] for i in put_in_indices]
                    group_df_inner["Kinase Sequence"] = Xy["Kinase Sequence"]
                    group_df_inner["Gene Name of Provided Kin Seq"] = Xy["Gene Name of Provided Kin Seq"]
                    group_df_inner["Gene Name of Kin Corring to Provided Sub Seq"] = [
                        Xy["Gene Name of Kin Corring to Provided Sub Seq"][i] for i in put_in_indices
                    ]
                else:  # Classifying the kinases into groups
                    group_df_inner["Site Sequence"] = Xy["Site Sequence"]
                    group_df_inner["Kinase Sequence"] = [Xy["Kinase Sequence"][i] for i in put_in_indices]
                    group_df_inner["Gene Name of Kin Corring to Provided Sub Seq"] = Xy[
                        "Gene Name of Kin Corring to Provided Sub Seq"
                    ]
                    group_df_inner["Gene Name of Provided Kin Seq"] = [
                        Xy["Gene Name of Provided Kin Seq"][i] for i in put_in_indices
                    ]
                if group_by == "site":
                    oposite_grp = "Kinase Sequence"
                else:
                    oposite_grp = "Site Sequence"
                segment_size = len(Xy[oposite_grp])
                group_df_inner["pair_id"] = [
                    Xy["pair_id"][i * segment_size + j] for i in put_in_indices for j in range(segment_size)
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
        group_classifier: GroupClassifier,
        cartesian_product: bool = False,
        **training_kwargs,
    ):
        notes = ""
        pass_through_scores = {}

        gen_train = self.get_group_dataframes(
            which_groups,
            Xy_formatted_train_file,
            group_classifier,
            group_classifier.get_ground_truth,
            cartesian_product=cartesian_product,
        )
        gen_val = self.get_group_dataframes(
            which_groups,
            Xy_formatted_val_file,
            group_classifier,
            group_classifier.get_ground_truth,
            cartesian_product=cartesian_product,
        )
        group_tr = ""
        for (group_tr, partial_group_df_tr), (group_vl, partial_group_df_vl) in (
            pb := CustomTqdm(
                zip(gen_train, gen_val),
                total=len(set(which_groups)),
            )
        ):
            pb.set_description(f"Status: Training Group Progress (Currently on {group_tr})")
            assert group_tr == group_vl, "Group mismatch: %s != %s" % (group_tr, group_vl)
            b = self.grp_to_interface_args[group_tr]["batch_size"]
            ng = self.grp_to_interface_args[group_tr]["n_gram"]
            assert isinstance(b, int), "Batch size must be an integer"
            assert isinstance(ng, int), "N-gram must be an integer"
            dummy = list(
                data_to_tensor(
                    partial_group_df_vl,
                    tokdict=self.default_tok_dict,
                    n_gram=ng,
                    device=self.device,
                    maxsize=MAX_SIZE_DS,
                )
            )[0]
            self.interfaces[group_tr].inp_size = self.interfaces[group_tr].get_input_size(dummy[0])
            self.interfaces[group_tr].inp_types = self.interfaces[group_tr].get_input_types(dummy[0])
            bpi, bc = self.interfaces[group_tr].get_bytes_per_input(batch_size=b)
            try:
                val_loader, _ = list(
                    data_to_tensor(
                        partial_group_df_vl,
                        tokdict=self.default_tok_dict,
                        n_gram=ng,
                        device=self.device,
                        maxsize=MAX_SIZE_DS,
                        bytes_per_input=bpi,
                        bytes_constant=bc,
                    )
                )[0]
            except ValueError as e:
                if str(e) == "Input data is empty":
                    logger.warning(f"No validation data for group {group_vl}.")
                val_loader = torch.utils.data.DataLoader(KSDataset([], [], []), batch_size=1)
            train_generator = more_itertools.peekable(
                data_to_tensor(
                    partial_group_df_tr,
                    tokdict=self.default_tok_dict,
                    batch_size=b,
                    n_gram=ng,
                    device=self.device,
                    maxsize=MAX_SIZE_DS,
                    bytes_per_input=bpi,
                    bytes_constant=bc,
                )
            )
            msm = self.grp_to_interface_args[group_tr]["model_summary_name"]
            assert isinstance(msm, str), "Model summary name must be a string"
            self.interfaces[group_tr].model_summary_name = msm + "-" + group_tr.upper()
            self.interfaces[group_tr].write_model_summary()
            err = self.interfaces[group_tr].train(
                train_generator,
                val_dl=val_loader,
                **self.grp_to_training_args[group_tr],
                extra_description="(Group: %s)" % group_tr.upper(),
                pass_through_scores=pass_through_scores,
                step_batch_size=b,
                **training_kwargs,
            )
            if err:
                notes = (
                    f"Stopped early after {-err} epochs since train loss was not decreasing and val score not"
                    " increasing."
                )

        weighted = sum([x[0] * x[1] for x in pass_through_scores.values()]) / sum(
            [x[1] for x in pass_through_scores.values()]
        )
        logger.valinfo(f"Overall Weighted {self.grp_to_training_args[group_tr]['metric']} → {weighted:3.4f}")
        return weighted, notes

    def obtain_group_and_loader(
        self,
        which_groups: list[str],
        group_classifier: GroupClassifier,
        Xy_formatted_input_file: str,
        device: torch.device,
        info_dict_passthrough={},
        seen_groups_passthrough=[],
        cartesian_product=False,
        simulated_acc=1,  # AKA regular predict
    ) -> Generator[Tuple[str, torch.utils.data.DataLoader], Tuple[str, pd.DataFrame], None]:
        assert len(info_dict_passthrough) == 0, "Info dict passthrough must be empty for passing in"
        gen_te = self.get_group_dataframes(
            which_groups,
            Xy_formatted_input_file,
            group_classifier=group_classifier,
            group_classifier_method=(
                group_classifier.predict
                if simulated_acc == 1
                else functools.partial(group_classifier.simulated_predict, simulated_acc=simulated_acc)
            ),
            cartesian_product=cartesian_product,
        )
        count = 0
        for group_te, partial_group_df_te in gen_te:
            if len(partial_group_df_te) == 0:
                print("Info: No inputs to evaluate for group =", group_te)
                continue
            ng = self.grp_to_interface_args[group_te]["n_gram"]
            assert isinstance(ng, int), "N-gram must be an integer"
            seen_groups_passthrough.append(group_te)
            dummy = list(
                data_to_tensor(
                    partial_group_df_te,
                    tokdict=self.default_tok_dict,
                    n_gram=ng,
                    device=device,
                    maxsize=MAX_SIZE_DS,
                    cartesian_product=cartesian_product,
                )
            )[0]
            self.interfaces[group_te].inp_size = self.interfaces[group_te].get_input_size(dummy[0])
            self.interfaces[group_te].inp_types = self.interfaces[group_te].get_input_types(dummy[0])
            bpi, bc = self.interfaces[group_te].get_bytes_per_input(batch_size=256)
            for test_loader, info_dict in data_to_tensor(
                partial_group_df_te,
                tokdict=self.default_tok_dict,
                n_gram=ng,
                device=device,
                maxsize=MAX_SIZE_DS,
                cartesian_product=cartesian_product,
                bytes_per_input=bpi,
                bytes_constant=bc,
            ):
                info_dict_passthrough[group_te] = info_dict
                info_dict_passthrough["on_chunk"] = info_dict["on_chunk"]
                info_dict_passthrough["total_chunks"] = info_dict["total_chunks"]
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
        with open(join_first(path, 1, __file__), "rb") as f:
            if "cuda" in target_device:
                ic: IndividualClassifiers = pickle.load(f)
            elif (
                target_device == "cpu"
            ):  # Workaround from https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219

                class CPU_Unpickler(pickle.Unpickler):
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
        group_classifier: GroupClassifier,
        predict_mode: bool,
        device: torch.device,
        get_emp_eqn: bool = True,
        emp_eqn_kwargs: dict = {"plot_emp_eqn": True, "print_emp_eqn": True},
        cartesian_product: bool = False,
    ) -> Union[None, list, Callable]:
        """Get predictions or ROC curves from model

        Parameters
        ----------
            addl_args :
                Additional arguments outside of `args` to be passed in.
            predict_mode :
                Whether or not to run in pure prediction mode
            get_emp_eqn : optional
                Whether or not to get empirical equation mapping raw score to empirical probability, by default True
            emp_eqn_kwargs : optional
                Arguments for the empirical equation, by default ``{"plot_emp_eqn": True, "print_emp_eqn": True}``
            cartesian_product : optional
                Whether or not to use cartesian product (i.e., cross product of sites and kinases), by default False

        Raises
        ------
            ValueError: If neither ``test`` nor ``test_json`` is specified in ``addl_args``.

        Returns
        -------
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
                which_groups=["All - Use Preds"],
                Xy_formatted_input_file=test_filename,
                group_classifier=group_classifier,
                device=addl_args["device"],
                info_dict_passthrough=info_dict_passthrough,
                cartesian_product=cartesian_product,
            ):
                jumbled_predictions = self.interfaces[grp].predict(
                    loader,
                    int(info_dict_passthrough["on_chunk"] + 1),
                    int(info_dict_passthrough["total_chunks"]),
                    cutoff=0.5,
                    group=grp,
                )  # TODO: Make adjustable cutoff
                # jumbled_predictions = list[predictions], list[output scores], list[group]
                del loader
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                new_info: list = info_dict_passthrough[grp]["PairIDs"]
                try:
                    if get_emp_eqn:
                        grp_to_emp_eqn = self.__dict__["grp_to_emp_eqn"].get(grp)
                    else:
                        grp_to_emp_eqn = None
                    for pair_id, i in zip(new_info, range(len(new_info))):  # type: ignore  # Why does this give warning?
                        all_predictions_outputs.update(
                            {
                                pair_id: (
                                    jumbled_predictions[0][i],
                                    jumbled_predictions[1][i],
                                    jumbled_predictions[2][i],
                                    jumbled_predictions[3][i],
                                    grp_to_emp_eqn,
                                )
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
                grp_to_info_pass_through_info_dict = {}  # TODO
                grp_to_loaders = {  # type: ignore
                    grp: loader
                    for grp, loader in self.obtain_group_and_loader(
                        which_groups=self.groups,
                        Xy_formatted_input_file=test_filename,
                        group_classifier=group_classifier,
                        device=device,
                        info_dict_passthrough=grp_to_info_pass_through_info_dict,
                    )
                }

                logger.status("Creating combined roc from individual models.")
                # TODO: Put new version of get_combined_rocs_from_individual_models here

            if self.args.get("s"):
                smart_save_nn(self)


def main(args_pass_in: Union[None, list[str]] = None, **training_kwargs) -> tuple[float, str]:
    """Main function for training the neural network(s).

    Parameters
    ----------
    args_pass_in: Union[None, list[str]], optional
        If None, uses sys.argv[1:]. Otherwise, uses the passed in list of strings as the command line arguments.
    training_kwargs: dict, optional, keyword only, variable length
        Keyword arguments to pass to the training function. See `train` for more details.

    Returns
    -------
        The first element is the final-epoch validation performance. The second element is a string containing any notes from the training process.

    Raises
    ------
    ValueError
        If the ``--pre-trained-gc`` argument is not passed in.

    """
    logger.status("Parsing Args")
    args = parse_args(args_pass_in)
    logger.status("Preparing Training Data")
    train_filename = args["train"]
    val_filename = args["val"]
    device = args["device"]

    gc_file = args["pre_trained_gc"]
    with open(join_first(gc_file, 1, __file__), "rb") as f:
        group_classifier: GroupClassifier = pickle.load(f)

    if isinstance(args["groups"], list):
        groups = args["groups"]
    elif args["groups"] is None:
        groups = group_classifier.all_groups
    else:
        raise ValueError("Groups must be a list or None.")

    assert device is not None

    with open(join_first(args["ksr_params"], 1, __file__)) as f:
        grp_to_model_args = json.load(f)
        default_grp_to_model_args = grp_to_model_args.get("default", grp_to_model_args.values().__iter__())
    with open(join_first(args["nni_params"], 1, __file__)) as f:
        grp_to_interface_args = json.load(f)
        for grp in grp_to_interface_args:
            grp_to_interface_args[grp]["loss_fn"] = eval(str(grp_to_interface_args[grp]["loss_fn"]))
            grp_to_interface_args[grp]["optim"] = eval(str(grp_to_interface_args[grp]["optim"]))
            grp_to_interface_args[grp]["device"] = device
        default_grp_to_interface_args = grp_to_interface_args.get("default", grp_to_model_args.values().__iter__())

    with open(join_first(args["ksr_training_params"], 1, __file__)) as f:
        grp_to_training_args = json.load(f)
        default_training_args = grp_to_training_args.get("default", grp_to_model_args.values().__iter__())

    gtma = {group: grp_to_model_args.get(group, deepcopy(default_grp_to_model_args)) for group in groups}
    gtia = {group: grp_to_interface_args.get(group, deepcopy(default_grp_to_interface_args)) for group in groups}
    gtta = {group: grp_to_training_args.get(group, deepcopy(default_training_args)) for group in groups}

    classifier = IndividualClassifiers(gtma, gtia, gtta, str(device), args, groups)
    if args["dry_run"]:
        logger.status("Dry Run Successful; Exiting after printing hyperparameter configurations:")
        logger.info(f"Model Args: {pprint.pformat(gtma, compact=True)}")
        logger.info(f"Interface Args: {pprint.pformat(gtia, compact=True)}")
        logger.info(f"Training Args: {pprint.pformat(gtta, compact=True)}")
        return -1, ""
    logger.status("About to Train")
    assert val_filename is not None
    weighted, notes = classifier.train(
        which_groups=groups,
        Xy_formatted_train_file=join_first(train_filename, 1, __file__),
        Xy_formatted_val_file=join_first(val_filename, 1, __file__),
        group_classifier=group_classifier,
        cartesian_product=False,
        **training_kwargs,
    )

    if args["s"] or args["s_test"]:
        if args["s_test"]:
            opt_idx = -1
        else:
            opt_idx = None
        smart_save_nn(classifier, opt_idx)
    return weighted, notes


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


def parse_args(args_pass_in: Union[None, list[str]] = None) -> dict[str, Union[str, None]]:
    """Argument parser for training a neural network.

    Parameters
    ----------
    args_pass_in : Union[None, list[str]], optional
        A list of arguments to parse. If None, defaults to sys.argv[1:].

    Returns
    -------
    dict[str, Union[str, None]]
        A dictionary of arguments mapping to their values.
    """
    logger.status("Parsing Arguments")

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
        default=join_first("models/hyperparameters/KSR_params.json", 1, __file__),
        metavar="<ksr_params.json>",
    )

    parser.add_argument(
        "--ksr-training-params",
        type=str,
        help="Specify Kinase Substrate Relationship training options file name",
        required=False,
        default=join_first("models/hyperparameters/KSR_training_params.json", 1, __file__),
        metavar="<ksr_params.json>",
    )

    parser.add_argument(
        "--nni-params",
        type=str,
        help="Specify Nerual Net Interface options file name",
        required=False,
        default=join_first("models/hyperparameters/NNI_params.json", 1, __file__),
        metavar="<ksr_params.json>",
    )

    parser.add_argument(
        "--groups",
        type=str,
        nargs="+",
        help="Specify groups to train on",
        required=False,
        metavar="<group> <group> ...",
    )

    parser.add_argument(
        "--pre-trained-gc",
        help="The path to the pre-trained group classifier.",
        required=True,
        metavar="<pre-trained group classifier file>",
    )

    parser.add_argument(
        "--dry-run",
        help="If included, will not train the model, but will instead print out the hyperparameters.",
        action="store_true",
        required=False,
    )

    parser.add_argument("-s", action="store_true", help="Include to save state", required=False)
    parser.add_argument("--s-test", action="store_true", help="Include to save state while testing", required=False)

    try:
        if args_pass_in is None:
            args = vars(parser.parse_args())
        else:
            args = vars(parser.parse_args(args_pass_in))
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
        assert os.path.exists(join_first(f, 1, __file__)), f"Input file '{join_first(f, 1, __file__)}' does not exist."

    return args


if __name__ == "__main__":  # pragma: no cover
    main()
