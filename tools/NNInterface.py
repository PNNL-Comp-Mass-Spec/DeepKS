from __future__ import annotations
import json, torch, re, torch.nn, torch.utils.data, sklearn.metrics, numpy as np, pandas as pd, collections, tqdm, io
import logging
import tempfile, os, scipy, itertools, warnings, pathlib, typing
from matplotlib import lines, pyplot as plt, rcParams
from typing import Tuple, Union, Callable, Literal, Sequence
from prettytable import PrettyTable
from torchinfo_modified import summary
from roc_comparison_modified.auc_delong import delong_roc_test
from termcolor import colored
from .roc_helpers import ROCHelpers

protected_roc_auc_score = ROCHelpers.protected_roc_auc_score

from ..data.preprocessing.PreprocessingSteps.get_kin_fam_grp import HELD_OUT_FAMILY
from ..tools.raw_score_to_prob import raw_score_to_probability

rcParams["font.family"] = "monospace"
rcParams["font.size"] = 12
# rcParams["font.serif"] = ["monospace", "Times", "Times New Roman", "Gentinum", "URW Bookman", "Roman", "Nimbus Roman"]

join_first = lambda levels, x: os.path.join(pathlib.Path(__file__).parent.resolve(), *[".."] * levels, x)
heaviside_cutoff = lambda outputs, cutoff: torch.heaviside(
    torch.sigmoid(outputs.data.cpu()).cpu() - cutoff, values=torch.tensor([0.0])
)
expand_metric = lambda met: "ROC AUC" if met == "roc" else "Accuracy" if met == "acc" else met

from ..config.root_logger import get_logger
from ..tools.custom_logging import CustomLogger

logger = get_logger()


class NNInterface:
    def __init__(
        self,
        model_to_train: torch.nn.Module,
        loss_fn: torch.nn.modules.loss._Loss,
        optim: torch.optim.Optimizer,
        inp_size=None,
        inp_types=None,
        model_summary_name=None,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        """Initializes the NNInterface class

        Args:
            @arg model_to_train: The model to train
            @arg loss_fn: The loss function to use
            @arg optim: The optimizer to use
            @arg inp_size: The input size of the model
            @arg inp_types: The input types of the model
            @arg model_summary_name: The name of the file to write the model summary to
            @arg device: The device to use for training

        Raises:
            AssertionError: If the loss function is not `torch.nn.BCEWithLogitsLoss` or `torch.nn.CrossEntropyLoss`
        """
        self.model: torch.nn.Module = model_to_train.to(device)
        self.criterion = loss_fn
        assert type(self.criterion) in [
            torch.nn.BCEWithLogitsLoss,
            torch.nn.CrossEntropyLoss,
        ], "Loss function must be either `torch.nn.BCEWithLogitsLoss` or `torch.nn.CrossEntropyLoss`."
        self.optimizer: torch.optim.Optimizer = optim
        self.device = device
        self.inp_size = inp_size
        self.inp_types = inp_types
        self.model_summary_name = model_summary_name
        if inp_size is None and inp_types is None and model_summary_name is None:
            return
        if inp_size is not None and inp_types is None and isinstance(self.model_summary_name, str):
            self.write_model_summary()

    def write_model_summary(self):
        logger.info(f"Writing model summary to file {self.model_summary_name}.")
        self.model_summary_name = join_first(0, self.model_summary_name)
        if isinstance(self.model_summary_name, str):
            with open(self.model_summary_name, "w", encoding="utf-8") as f:
                str_rep = str(self)
                f.write(str_rep)

        elif isinstance(self.model_summary_name, io.StringIO):
            self.model_summary_name.write(str(self))

    def __str__(self):
        try:
            self.representation = (
                "\n"
                + "--- Model Summary ---\n"
                + str(
                    ms := summary(
                        self.model,
                        device=self.device,
                        input_size=self.inp_size,
                        dtypes=self.inp_types,
                        col_names=["input_size", "output_size", "num_params", "trainable"],
                        row_settings=["var_names"],
                        verbose=0,
                        col_width=50,
                    )
                )
                + "\n"
            )
            self.model_summary = ms
            torch.cuda.empty_cache()
        except Exception as e:
            print("Failed to run model summary:", flush=True)
            print(e, flush=True)
            raise
        return self.representation

    def report_progress(
        self,
        paradigm,
        epoch,
        num_epochs,
        batch_num,
        chunk_num,
        total_step,
        loss,
        score,
        metric,
        print_every=1,
        retain=False,
    ):
        if not (isinstance(batch_num, str) or batch_num % print_every == 0):
            return
        if "validation" in paradigm.lower():
            method = CustomLogger.valinfo
        elif "train" in paradigm.lower():
            method = CustomLogger.trinfo
        else:
            method = CustomLogger.info

        method(
            logger,
            (
                f"{'(Means) ' if 'mean' in paradigm.lower() else ''}Chunk [{chunk_num + 1}] | Epoch"
                f" [{epoch + 1}/{num_epochs}] | Batch"
                f" [{batch_num + 1 if isinstance(batch_num, int) else 'All'}/{total_step}] | Loss {loss:.4f} |"
                f" {expand_metric(metric)} {score:.2f}"
            ),
        )

    def train(
        self,
        loader_generator,
        num_epochs=50,
        lr_decay_amount=1.0,
        lr_decay_freq=1,
        threshold=None,
        val_dl=None,
        cutoff=0.5,
        metric: Literal["roc", "acc"] = "roc",
        extra_description="",
        pass_through_scores=None,
        **training_kwargs,
    ):
        assert isinstance(cutoff, (int, float)), "Cutoff needs to be a number."
        assert (
            type(self.criterion) == torch.nn.BCEWithLogitsLoss or type(self.criterion) == torch.nn.CrossEntropyLoss
        ), "Loss function must be either `torch.nn.BCEWithLogitsLoss` or `torch.nn.CrossEntropyLoss`."

        logger.status(f"Training {extra_description}")

        lowest_loss = float("inf")
        total_step = float("nan")
        threshold = threshold if threshold is not None else float("inf")
        epoch = 0

        # While loop to train the model
        cycler = itertools.cycle(loader_generator)
        while not ((lowest_loss < threshold and epoch >= num_epochs) or epoch >= 2 * num_epochs):
            train_scores = []
            losses = []
            self.model.train()

            # Decrease learning rate, if so desired
            if epoch % lr_decay_freq == 0 and epoch > 0:
                for param in self.optimizer.param_groups:
                    param["lr"] *= lr_decay_amount

            # Batch loop
            chunk_num = -1
            for (train_loader, _, _, _), info_dict in cycler:
                chunk_num += 1
                if chunk_num == info_dict["total_chunks"]:
                    chunk_num -= 1
                    break
                batch_num = 0
                for batch_num, (*X, labels) in enumerate(train_loader):
                    total_step = len(train_loader)
                    X = [x.to(self.device) for x in X]
                    labels = labels.to(self.device)

                    # Forward pass
                    logger.vstatus("Train Step A - Forward propogating.")
                    outputs = self.model.forward(*X)
                    outputs = outputs if outputs.size() != torch.Size([]) else outputs.reshape([1])
                    torch.cuda.empty_cache()

                    # Compute loss
                    print(
                        logger.vstatus("Train Step B - Computing loss."),
                        flush=True,
                        end="\r",
                    )
                    loss: torch.Tensor
                    match self.criterion:
                        case torch.nn.CrossEntropyLoss():
                            loss = self.criterion.__call__(outputs, labels.long())
                        case torch.nn.BCEWithLogitsLoss():
                            loss = self.criterion.__call__(outputs, labels.float())

                    # Backward and optimize
                    self.optimizer.zero_grad()
                    logger.vstatus("Train Step C - Backpropogating.")
                    loss.backward()
                    logger.vstatus("Train Step D - Stepping in ∇'s direction.")
                    self.optimizer.step()

                    # Report Progress
                    performance_score = None
                    performance_score, _ = self._get_acc_or_auc_and_predictions(outputs, labels, metric, cutoff)
                    self.report_progress(
                        "Train",
                        epoch,
                        num_epochs,
                        batch_num,
                        chunk_num,
                        total_step,
                        torch.mean(loss).item(),
                        performance_score,
                        metric,
                        print_every=1,
                        retain=True,
                    )

                    lowest_loss = min(lowest_loss, loss.item())
                    losses.append(loss.item())
                    train_scores.append(performance_score)
            try:
                assert len(train_scores) != 0, "No Data Trained"
            except AssertionError as e:
                logger.warning(str(e))

            mean_loss = (sum(losses) / len(losses)) if losses else float("NaN")
            self.report_progress(
                "Mean Train",
                epoch,
                num_epochs,
                "",
                chunk_num,
                total_step,
                mean_loss,
                (sum(train_scores) / len(train_scores)) if train_scores else float("NaN"),
                expand_metric(metric),
                retain=True,
            )

            score, all_outputs = -1, [-1]
            # Validate
            if val_dl is not None:
                print(" " * os.get_terminal_size().columns, end="\r")
                logger.status("Validating")
                total_step = len(val_dl)
                score, val_loss, _, all_outputs, _ = self.eval(val_dl, cutoff, metric, display_pb=False)
                self.report_progress("Validation", epoch, num_epochs, 0, 0, 1, val_loss, score, metric, retain=True)
            assert score >= 0

            logger.status(f" ---------< Epoch {epoch + 1}/{num_epochs} Done >---------\n")
            epoch += 1
            if pass_through_scores is not None and val_dl is not None:
                pass_through_scores[extra_description] = (score, len(all_outputs))

            # logger.debug(f"{training_kwargs=}")
            if (
                training_kwargs.get("loss_chances") is not None
                and training_kwargs.get("loss_below") is not None
                and training_kwargs.get("val_le") is not None
            ):
                # logger.debug(
                #     f"{mean_loss=}, {training_kwargs['loss_below']=}, {epoch=}, {training_kwargs['loss_chances']=},"
                #     f" {score=}, {training_kwargs['val_le']=}"
                # )
                # logger.debug(f"{mean_loss >= training_kwargs['loss_below']=}; {epoch >= training_kwargs['loss_chances'] - 1=}")
                if (
                    mean_loss >= training_kwargs["loss_below"]
                    and epoch == training_kwargs["loss_chances"]
                    and score < training_kwargs["val_le"]
                ):
                    logger.warning("Stopping early because train loss is not decreasing.")
                    return -epoch
            if chunk_num == -1:
                logger.warning(
                    f"No data for {extra_description}, skipping training for this group. Neural network weights will"
                    " be random."
                )

    def predict(
        self,
        dataloader,
        on_chunk,
        total_chunks,
        cutoff=0.5,
        group="UNKNOWN GROUP",
        metric: Literal["roc", "acc"] = "roc",
        display_pb: bool = False,
    ) -> Tuple[list, list, list, list]:
        """
        Returns a list of predictions, a list of outputs, a list of groups, and a list of labels.
        """

        metadata = {"group": group, "on_chunk": on_chunk, "total_chunks": total_chunks}
        _, _, outputs, labels, predictions = self.eval(
            dataloader, cutoff, metric, predict_mode=True, metadata=metadata, display_pb=display_pb
        )

        return [bool(x) for x in predictions], outputs, [group] * len(outputs), labels

    def _get_acc_or_auc_and_predictions(
        self, outputs: torch.Tensor, labels: torch.Tensor, metric: Literal["acc", "roc"], cutoff=0.5
    ):
        match self.criterion:
            case torch.nn.CrossEntropyLoss():
                predictions = torch.argmax(outputs.data.cpu(), dim=1).cpu()
            case torch.nn.BCEWithLogitsLoss():
                outputs = torch.sigmoid(outputs.data.cpu())
                predictions = heaviside_cutoff(outputs, cutoff)
            case _:
                raise NotImplementedError(f"Loss function {self.criterion} not implemented.")
        match metric:
            case "acc":
                performance = sklearn.metrics.accuracy_score(labels.cpu(), predictions)
            case "roc":
                scores = outputs.data.cpu()
                performance = protected_roc_auc_score(labels.cpu(), scores)

        return performance, predictions

    def eval(
        self,
        dataloader,
        cutoff=0.5,
        metric: typing.Literal["roc", "acc"] = "roc",
        predict_mode=False,
        expected_input_tuple_len=3,
        metadata={"group": "<?>", "on_chunk": 1, "total_chunks": 1},
        display_pb=True,
    ) -> Tuple[float, float, list[float], list[int], list[int]]:
        """
        Returns:
            (mean performance, mean loss, all outputs, all labels, all predictions, all outputs--sigmoided)
        """
        assert (
            type(self.criterion) == torch.nn.BCEWithLogitsLoss or type(self.criterion) == torch.nn.CrossEntropyLoss
        ), "Loss function must be either `torch.nn.BCEWithLogitsLoss` or `torch.nn.CrossEntropyLoss`."

        all_labels, all_outputs, all_preds, avg_perf, avg_loss = [], [], [], [], []
        self.model.eval()
        outputs = torch.Tensor([-1])
        # if self.device == torch.device('cpu'):
        #     input("!!! Warning: About to evaluate using CPU."
        #           "This may be lengthy and/or crash your computer (due to high RAM use)."
        #           "Press any key to continue anyway (ctrl+c to abort): ")
        dll = list(dataloader)
        # for x in dll:
        #     assert (
        #         len(x) == expected_input_tuple_len
        #     ), f"Input tuple length {len(x)} was not what was expected ({expected_input_tuple_len})."

        ible = dll
        if display_pb:
            logger.warning("Progress bars in `eval` are currently not implemented, but display_pb was set.")

        with torch.no_grad():
            for b, (*X, labels) in enumerate(ible):
                # if predict_mode:
                #     assert torch.equal(
                #         labels, torch.Tensor([-1] * len(labels)).to(self.device)
                #     ), "Labels must be -1 for prediction."
                X = [x.to(self.device) for x in X]

                if not isinstance(ible, tqdm.tqdm):
                    logger.vstatus(f"Status: Forward propogating through NN -- Batch [{b + 1}/{len(dll)}]")

                labels = labels.to(self.device)
                outputs = self.model.forward(*X)
                outputs = outputs.unsqueeze(0) if outputs.dim() == 0 else outputs

                match self.criterion:
                    case torch.nn.CrossEntropyLoss():
                        labels = labels.long()
                    case torch.nn.BCEWithLogitsLoss():
                        labels = labels.float()

                loss = self.criterion(outputs, labels)
                outputs = torch.sigmoid(outputs)

                if "cuda" in str(self.device):
                    torch.cuda.empty_cache()

                performance, predictions = self._get_acc_or_auc_and_predictions(outputs, labels, metric, cutoff)

                all_labels += labels.cpu().numpy().tolist()
                all_outputs += outputs.data.cpu().numpy().tolist()
                all_preds += predictions.long().cpu().numpy().tolist()
                avg_perf += [performance] * len(labels)
                avg_loss += [loss.item()] * len(labels)

            avg_perf = sum(avg_perf) / len(avg_perf) if len(avg_perf) > 0 else 0.0
            avg_loss = sum(avg_loss) / len(avg_loss) if len(avg_loss) > 0 else 0.0

            return avg_perf, avg_loss, all_outputs, all_labels, all_preds

    @staticmethod
    def get_combined_rocs_from_individual_models(
        grp_to_interface: dict[str, NNInterface] = {},
        grp_to_loader: dict[str, torch.utils.data.DataLoader] = {},
        savefile="",
        retain_evals=None,
        from_loaded=None,
        grp_to_loader_true_groups=None,
        get_emp_eqn=True,
        emp_eqn_kwargs={},
        get_avg_roc_line=True,
        _ic_ref=None,
        grp_to_idents=[],
    ) -> Union[None, dict]:
        assert grp_to_loader_true_groups is not None
        if from_loaded is None:
            grp_to_interface = {k: grp_to_interface[k] for k in grp_to_loader}
            assert grp_to_interface.keys() == grp_to_loader.keys(), (
                "The groups for the provided models are not equal to the groups for the provided loaders."
                f" Respectively, {grp_to_interface.keys()} != {grp_to_loader.keys()}"
            )
        orig_order = ["CAMK", "AGC", "CMGC", "CK1", "OTHER", "TK", "TKL", "STE", "ATYPICAL", "<UNANNOTATED>"]
        fig = plt.figure(figsize=(24, 24))
        plt.plot(
            [0, 1],
            [0, 1],
            color="black",
            linestyle="dashed",
            alpha=1,
            linewidth=1,
            label=" Random Model ┆ AUC 0.5   ┆" + " " * 25 + "┆ n = ∞",
        )
        groups = grp_to_interface.keys() if from_loaded is None else from_loaded.keys()
        true_grp_to_outputs_and_labels = collections.defaultdict(lambda: collections.defaultdict(list))
        grp_to_emp_eqn = {}
        for grp in groups:
            if from_loaded is None:
                interface = grp_to_interface[grp]
                loader = grp_to_loader[grp]
                _, _, outputs, labels, _ = interface.eval(
                    loader, cutoff=0.5, metric="roc", predict_mode=False, display_pb=False
                )
                if retain_evals is not None:
                    # Group -> Tr/Vl/Te -> outputs/labels -> list
                    retain_evals.update({grp: {"test": {"outputs": outputs, "labels": labels}}})
            else:
                outputs = from_loaded[grp]["test"]["outputs"]
                labels = from_loaded[grp]["test"]["labels"]
            agst = np.argsort(outputs)
            booled_labels = [bool(x) for x in labels]
            if get_emp_eqn:
                emp_prob_lambda = raw_score_to_probability(sorted(outputs), [booled_labels[i] for i in agst], **emp_eqn_kwargs)  # type: ignore
                grp_to_emp_eqn[grp] = emp_prob_lambda
            else:

                def emp_prob_lambda(l: list[Union[float, int]]):
                    raise AssertionError(
                        "For some reason, get_emp_eqn was set to False, but the emp_prob_lambda was called."
                    )

            for i in range(len(grp_to_loader_true_groups[grp])):
                true_grp_to_outputs_and_labels[grp_to_loader_true_groups[grp][i]]["outputs"].append(outputs[i])
                true_grp_to_outputs_and_labels[grp_to_loader_true_groups[grp][i]]["labels"].append(labels[i])
                true_grp_to_outputs_and_labels[grp_to_loader_true_groups[grp][i]]["group_model_used"].append(grp)
                if get_emp_eqn:
                    true_grp_to_outputs_and_labels[grp_to_loader_true_groups[grp][i]]["outputs_emp_prob"].append(
                        emp_prob_lambda([outputs[i]])[0]
                    )

            assert len(outputs) == len(labels), (
                "Something is wrong in NNInterface.get_all_rocs_by_group; the length of outputs, the number of labels,"
                f" and the number of kinases in `kinase_order` are not equal. (Respectively, {len(outputs)},"
                f" {len(labels)}"
            )

        if get_emp_eqn:
            setattr(_ic_ref, "grp_to_emp_eqn", grp_to_emp_eqn)
            # Repickle
            if _ic_ref is not None:
                try:
                    with tempfile.NamedTemporaryFile() as tf:
                        _ic_ref.save_all(_ic_ref, tf.name)
                        print(colored("Info Re-saved Individual Classifiers with empirical equation.", "blue"))

                        with open(_ic_ref.repickle_loc.replace(".pkl", "_with_emp_eqn.pkl"), "wb") as f:
                            f.write(tf.read())
                except Exception as e:
                    warnings.warn(f"Couldn't repickle Individual Classifiers with empirical equation: {e}", UserWarning)
            else:
                warnings.warn(
                    (
                        "No repickle location found for Individual Classifiers. Not repickling; can't save empirical"
                        " equation."
                    ),
                    UserWarning,
                )

        points_fpr = []
        points_tpr = []
        aucs = []
        for group in set(all_obs := list(itertools.chain(*[list(x) for x in grp_to_loader_true_groups.values()]))):
            outputs, labels = (
                true_grp_to_outputs_and_labels[group]["outputs_emp_prob" if get_emp_eqn else "outputs"],
                true_grp_to_outputs_and_labels[group]["labels"],
            )
            orig_i = orig_order.index(group)

            plt.ioff()
            res = NNInterface.roc_core(
                outputs,
                labels,
                orig_i,
                line_labels=[f"{group}"],
                linecolor=None,
                total_obs=len(all_obs),
            )
            # if group in ['AGC', 'TK', 'CMGC']:
            points_fpr.append(res[0])
            points_tpr.append(res[1])
            aucs.append(res[2])

        plt.legend(
            loc="lower right",
            title="Legend — 'Solid line' indicates significant (ROC ≠ 0.5) model.)",
            title_fontproperties={"weight": "bold", "size": "medium"},
            fontsize="small",
        )

        if savefile:
            fig.savefig(savefile, bbox_inches="tight")

    @staticmethod
    def roc_core(
        outputs: Sequence[float],
        labels,
        i,
        linecolor=(0.5, 0.5, 0.5),
        line_labels=["Train Set", "Validation Set", "Test Set", f"Held Out Family — {HELD_OUT_FAMILY}"],
        alpha=0.05,
        total_obs: int = 1,
    ):
        assert len(outputs) == len(labels), (
            f"Something is wrong in NNInterface.roc_core; the length of outputs ({len(outputs)}) does not equal the"
            f" number of labels ({len(labels)})"
        )
        assert len(line_labels) > 0, "`line_labels` length is 0."
        if linecolor is None:
            linecolor = plt.rcParams["axes.prop_cycle"].by_key()["color"][i]

        roc_data = sklearn.metrics.roc_curve(labels, np.asarray(outputs))
        aucscore = protected_roc_auc_score(labels, outputs)

        labels, outputs, rand_outputs = NNInterface.create_random(
            np.array(labels), np.array(outputs), group=line_labels[0].split(" ")[0]
        )  # type: ignore
        if len(labels):
            random_auc = protected_roc_auc_score(labels, rand_outputs)
            assert abs(random_auc - 0.5) < 1e-16, f"Random ROC not equal to 0.5! (Was {random_auc})"
            _, se, diff = delong_roc_test(labels, rand_outputs, outputs)
            quant = -scipy.stats.norm.ppf(alpha / 2)
            is_signif = (diff + quant * se + 0.5) > (diff - quant * se + 0.5) > 0.5
        else:
            is_signif = False
            se = diff = quant = float("nan")
        label = (
            f"{line_labels[0] if len(line_labels) == 1 else line_labels[i]:>13} ┆ AUC {aucscore:.3f} ┆"
            f" {100-int(alpha*100)}% CI = [{max(0, (diff - quant*se + 0.5)):.2f},"
            f" {min((diff + quant*se + 0.5), 1):.2f}] {'❋' if is_signif else ' '} ┆ n = {len(labels)} —"
            f" {len(labels)*100/total_obs:3.2f}%"
        )
        sklearn.metrics.RocCurveDisplay(fpr=roc_data[0], tpr=roc_data[1]).plot(
            color=linecolor,
            linestyle="solid" if is_signif else "dashed",
            linewidth=2 if i is not None else 3,
            ax=plt.gca(),
            label=label,
            alpha=1 if is_signif else 0.25,
        )
        plt.gca().set_aspect(1)
        if aucscore > 1:
            NNInterface.inset_auc()
        plt.title("ROC Curves (with DeLong Test 95% confidence intervals)")
        plt.xticks([x / 100 for x in range(0, 110, 10)])
        plt.yticks([x / 100 for x in range(0, 110, 10)])

        return roc_data[0], roc_data[1], aucscore

    @staticmethod
    def inset_auc():
        ax: plt.Axes = plt.gca()
        ax_inset = ax.inset_axes([0.65, 0.1, 0.5, 0.5])
        x1, x2, y1, y2 = 0, 0.15, 0.85, 1
        ax_inset.set_xlim(x1, x2)
        ax_inset.set_ylim(y1, y2)
        ax.indicate_inset_zoom(ax_inset, linestyle="dashed")
        ax_inset.set_xticks([x1, x2])
        ax_inset.set_yticks([y1, y2])
        ax_inset.set_xticklabels([x1, x2])
        ax_inset.set_yticklabels([y1, y2])
        rel_line: lines.Line2D = ax.lines[-1]
        ax_inset.plot(
            rel_line.get_xdata(),
            rel_line.get_ydata(),
            linestyle=rel_line.get_linestyle(),
            linewidth=rel_line.get_linewidth(),
            color=rel_line.get_color(),
        )

    @staticmethod
    def create_random(labels: np.ndarray, outputs: np.ndarray, group=""):
        ast: list[int] = np.argsort(labels).astype(int).tolist()
        num_zeros = len(ast) - sum(labels)
        num_ones = sum(labels)
        if num_zeros % 2:  # number of zeros is not even
            print(
                "Info: Trimming a correct true decoy instance for even size of"
                f" sets{f' — {group}' if group != '' else ''}."
            )
            key_: Callable[[Tuple[float, float]], Tuple[float, float]] = lambda x: (x[0], x[1])
            rm_idx = sorted(list(zip(labels, outputs)), key=key_)[0]
            rm_idx = list(zip(labels, outputs)).index(rm_idx)
            rm_idx = ast.index(rm_idx)
            ast = ast[:rm_idx] + ast[rm_idx + 1 :]
        if num_ones % 2:  # number of ones is not even
            print(
                "Info: Trimming a correct true target instance for even size of"
                f" sets{f' — {group}' if group != '' else ''}."
            )
            rm_idx = sorted(list(zip(labels, outputs)), key=lambda x: (-x[0], -x[1]))[0]
            rm_idx = list(zip(labels, outputs)).index(rm_idx)
            rm_idx = ast.index(rm_idx)
            ast = ast[:rm_idx] + ast[rm_idx + 1 :]

        asts: list[int] = sorted(ast)
        if len(asts) > 0:
            labels2: np.ndarray = labels[np.array(asts)]
            outputs2: np.ndarray = outputs[np.array(asts)]

            assert not len(labels2[(labels2 == 1)]) % 2, "319"
            assert not len(labels2[(labels2 == 0)]) % 2, "320"

            ast2 = np.argsort(labels2)
            rand_outputs = np.zeros_like(labels2)
            for i in range(len(ast2)):
                rand_outputs[ast2[i]] = 1 if i % 2 else 0

            return labels2, outputs2, rand_outputs
        else:
            return [], [], []

    def test(
        self,
        test_loader,
        print_sample_predictions: bool = False,
        cutoff: float = 0.5,
        metric: Literal["roc", "acc"] = "roc",
        int_label_to_str_label: dict[int, str] = {},
    ) -> Tuple[list, list, list]:
        """
        Returns: list of predictions, list of outputs, list of ground-truth labels
        """

        print(" " * os.get_terminal_size().columns, end="\r")
        logger.status("Testing")
        performance, _, outputs, labels, predictions = self.eval(
            test_loader, cutoff, metric, predict_mode=False, display_pb=False
        )
        print(
            colored(
                f"""{f"Test Info: {expand_metric(metric)}: {performance:.3f} {'%' if metric == 'acc' else ''}"}""",
                "blue",
            )
        )

        if print_sample_predictions:
            tab = PrettyTable(["Input Index", "Ground Truth Label", "Prediction"], min_width=20)
            num_rows = min(10, len(labels))
            tab.title = f"Info: First {num_rows} labels mapped to predictions"
            for i in range(num_rows):
                pred_print = predictions[i] if not int_label_to_str_label else int_label_to_str_label[predictions[i]]
                lab_print = labels[i] if not int_label_to_str_label else int_label_to_str_label[labels[i]]
                tab.add_row([i, lab_print, pred_print])
            logger.teinfo(str(tab))

        return predictions, outputs, labels

    def save_model(self, path):
        torch.save(self.model.state_dict(), open(path, "wb"))

    def save_eval_results(self, loader, path, kin_order=None):
        eval_res = self.eval(dataloader=loader)
        outputs = eval_res[-1]
        labels = eval_res[3]
        results_dict = {"labels": labels, "outputs": outputs}
        if kin_order is not None:
            results_dict.update({"kin_order": kin_order})
        json.dump(results_dict, open(path, "w"), indent=4)

    def load_model(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)

    @staticmethod
    def get_input_size(dl: torch.utils.data.DataLoader, leave_out_last=True):
        inp_sizes = []
        assert isinstance(dl, torch.utils.data.DataLoader), "dl must be a DataLoader, was {}".format(type(dl))
        dll = list(dl)
        iterab = dll[0] if not leave_out_last else dll[0][:-1]
        for X in iterab:
            assert isinstance(X, torch.Tensor)
            inp_sizes.append(list(X.size()))
        return inp_sizes

    @staticmethod
    def get_input_types(dl: torch.utils.data.DataLoader, leave_out_last=True):
        types = []
        assert isinstance(dl, torch.utils.data.DataLoader)
        dll = list(dl)
        iterab = dll[0] if not leave_out_last else dll[0][:-1]
        for X in iterab:
            assert isinstance(X, torch.Tensor)
            types.append(X.dtype)
        return types
