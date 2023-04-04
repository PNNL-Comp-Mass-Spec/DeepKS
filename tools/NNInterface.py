from __future__ import annotations
import json, torch, re, torch.nn, torch.utils.data, sklearn.metrics, numpy as np, pandas as pd, collections, tqdm, io
import tempfile, os, scipy, itertools, warnings, pathlib, typing
from matplotlib import ArrayLike, lines, pyplot as plt, rcParams
from typing import Collection, Tuple, Union, Callable, Literal, Sequence
from prettytable import PrettyTable
from torchinfo_modified import summary
from roc_comparison_modified.auc_delong import delong_roc_test
from termcolor import colored
from sklearn.metrics import roc_auc_score

from .roc_lambda import get_avg_roc
from ..data.preprocessing.PreprocessingSteps.get_kin_fam_grp import HELD_OUT_FAMILY
from ..tools.raw_score_to_prob import raw_score_to_probability
from ..tools.model_utils import KSDataset

rcParams["font.family"] = "monospace"
rcParams["font.size"] = 12
# rcParams["font.serif"] = ["monospace", "Times", "Times New Roman", "Gentinum", "URW Bookman", "Roman", "Nimbus Roman"]

join_first = lambda levels, x: os.path.join(pathlib.Path(__file__).parent.resolve(), *[".."] * levels, x)
heaviside_cutoff = lambda outputs, cutoff: torch.heaviside(
    torch.sigmoid(outputs.data.cpu()).cpu() - cutoff, values=torch.tensor([0.0])
)
expand_metric = lambda met: "ROC AUC" if met == "roc" else "Accuracy" if met == "acc" else met


def protected_roc_auc_score(y_true: ArrayLike, y_score: ArrayLike, *args, **kwargs) -> float:
    """Wrapper for sklearn.metrics.roc_auc_score that handles edge cases

    Args:
        @arg y_true: Iterable of (integer) true labels
        @arg y_score: Iterable of predicted scores
        @arg *args: Additional arguments to pass to roc_auc_score
        @arg **kwargs: Additional keyword arguments to pass to roc_auc_score

    Raises:
        e: Error that is not a multi class error or single class present error

    Returns:
        float: The roc_auc_score
    """
    try:
        return float(roc_auc_score(y_true, y_score, *args, **kwargs))
    except Exception as e:
        match str(e):
            case "Only one class present in y_true. ROC AUC score is not defined in that case.":
                warnings.warn(f"Setting roc_auc_score to 0.0 since there is only one class present in y_true.")
                return 0.0
            case "multi_class must be in ('ovo', 'ovr')":
                # softmax in case of multi-class
                y_score = torch.nn.functional.softmax(torch.as_tensor(y_score), dim=1)
                return protected_roc_auc_score(
                    y_true,
                    y_score,
                    *args,
                    **({"multi_class": "ovo", "average": "macro", "labels": list(range(y_score.shape[-1]))} | kwargs),
                )
            case _:
                raise e


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
        self.model: torch.nn.Module = model_to_train
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
        print(colored(f"Info: Writing model summary to file {self.model_summary_name}.", "blue"))
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

    def report_progress(self, paradigm, epoch, num_epochs, batch_num, total_step, loss, score, metric):
        print(colored(f"{f'{paradigm} Info:':20}", "blue"), end="")
        print(colored(f"Epoch [{epoch + 1}/{num_epochs}], ", "blue"), end="")
        print(colored(f"Batch [{batch_num + 1}/{total_step}], ", "blue"), end="")
        print(colored(f"{paradigm} Loss: {loss:.4f}, {paradigm} {expand_metric(metric)}: {score:.2f}", "blue"))

    def train(
        self,
        train_loader,
        num_epochs=50,
        lr_decay_amount=1.0,
        lr_decay_freq=1,
        threshold=None,
        val_dl=None,
        cutoff=0.5,
        metric: Literal["roc", "acc"] = "roc",
        extra_description="",
    ):
        print_every = 1  # max(len(train_loader) // 10, 1)
        assert isinstance(cutoff, (int, float)), "Cutoff needs to be a number."
        assert (
            type(self.criterion) == torch.nn.BCEWithLogitsLoss or type(self.criterion) == torch.nn.CrossEntropyLoss
        ), "Loss function must be either `torch.nn.BCEWithLogitsLoss` or `torch.nn.CrossEntropyLoss`."

        print(
            colored(f"Status: Training{extra_description}{'' if extra_description == '' else ' '}." + "\n", "green"),
            flush=True,
        )

        lowest_loss = float("inf")
        threshold = threshold if threshold is not None else float("inf")
        epoch = 0

        # While loop to train the model
        while not ((lowest_loss < threshold and epoch >= num_epochs) or epoch >= 2 * num_epochs):
            train_scores = []
            self.model.train()

            # Decrease learning rate, if so desired
            if epoch % lr_decay_freq == 0 and epoch > 0:
                for param in self.optimizer.param_groups:
                    param["lr"] *= lr_decay_amount

            # Batch loop
            batch_num = 0
            for batch_num, (*X, labels) in enumerate(list(train_loader)):
                total_step = len(train_loader)
                X = [x.to(self.device) for x in X]
                labels = labels.to(self.device)

                # Forward pass
                print(colored(f"{'Train Status:':20}Step A - Forward pass.", "green"), flush=True, end="\r")
                outputs = self.model.forward(*X)
                outputs = outputs if outputs.size() != torch.Size([]) else outputs.reshape([1])
                torch.cuda.empty_cache()

                # Compute loss
                print(colored(f"{'Train Status:':20}Step B - Computing loss.", "green"), flush=True, end="\r")
                loss: torch.Tensor
                match self.criterion:
                    case torch.nn.CrossEntropyLoss():
                        loss = self.criterion.__call__(outputs, labels.long())
                    case torch.nn.BCEWithLogitsLoss():
                        loss = self.criterion.__call__(outputs, labels.float())

                # Backward and optimize
                self.optimizer.zero_grad()
                print(colored(f"{'Train Status:':20}Step C - Computing gradients.", "green"), flush=True, end="\r")
                loss.backward()
                print(
                    colored(f"{'Train Status:':20}Step D - Stepping in the direction of the gradient.", "green"),
                    flush=True,
                    end="\r",
                )
                self.optimizer.step()

                # Report Progress
                score = -1.0
                if (batch_num) % print_every == 0:
                    score, train_loss, _, _, _, _ = self.eval(
                        torch.utils.data.DataLoader(KSDataset(X[0], X[1], labels), batch_size=len(labels)),
                        cutoff,
                        metric,
                        display_pb=False,
                    )
                    self.report_progress("Train", epoch, num_epochs, batch_num, total_step, train_loss, score, metric)

                lowest_loss = min(lowest_loss, loss.item())
                train_scores.append(score)

            print(colored(f"{'Train Info:':20}Mean Train {expand_metric(metric)} ", "blue"), end="", flush=True)
            print(colored(f"for Epoch [{epoch + 1}/{num_epochs}] was ", "blue"), end="", flush=True)
            print(colored(f"{sum(train_scores)/len(train_scores):.3f}", "blue"), flush=True)

            # Validate
            if val_dl is not None:
                total_step = len(val_dl)
                score, val_loss, _, _, _, _ = self.eval(val_dl, cutoff, metric, display_pb=False)
                self.report_progress("Validation", epoch, num_epochs, 0, 1, val_loss, score, metric)

            print(colored(f"Status: ---------< Epoch {epoch + 1}/{num_epochs} Done >---------\n", "green"), flush=True)
            epoch += 1

    def predict(
        self,
        dataloader,
        on_chunk,
        total_chunks,
        cutoff=0.5,
        group="UNKNOWN GROUP",
        report_performance_test_mode=False,
        show_sample_output_test_mode=False,
        metric: Literal["roc", "acc"] = "roc",
    ) -> Tuple[list, list, list, list]:
        """
        Returns a list of predictions, a list of outputs, a list of groups, and a list of labels.
        """

        metadata = {"group": group, "on_chunk": on_chunk, "total_chunks": total_chunks}
        performance, _, outputs, labels, predictions, _ = self.eval(
            dataloader, cutoff, metric, predict_mode=not report_performance_test_mode, metadata=metadata
        )

        print(colored(f"Info: Test {expand_metric(metric)} ({group}): {performance:.3f} %\n", "blue"))

        if show_sample_output_test_mode:
            tab = PrettyTable(["Index", "Label", "Prediction"], min_width=20)
            num_rows = min(10, len(labels))
            tab.title = f"Info: First {num_rows} labels mapped to predictions"
            for i in range(num_rows):
                tab.add_row([i, labels[i], predictions[i]])
            print(colored(str(tab), color="blue"), flush=True)

        return [bool(x) for x in predictions], outputs, [group] * len(outputs), labels

    def eval(
        self,
        dataloader,
        cutoff=0.5,
        metric: typing.Literal["roc", "acc"] = "roc",
        predict_mode=False,
        expected_input_tuple_len=3,
        metadata={"group": "<?>", "on_chunk": 1, "total_chunks": 1},
        display_pb=True,
    ) -> Tuple[float, float, list[float], list[int], list[float], list[float]]:
        """
        Returns:
            (mean performance, mean loss, all outputs, all labels, all predictions, all outputs--sigmoided)
        """
        group = metadata.get("group", "<?>")
        on_chunk = metadata.get("on_chunk", -1)
        total_chunks = metadata.get("total_chunks", -1)
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
        for x in dll:
            assert (
                len(x) == expected_input_tuple_len
            ), f"Input tuple length {len(x)} was not what was expected {expected_input_tuple_len}."

        if not display_pb:
            ible = dll
        else:
            description = (
                f"Status: Eval Progress of {group} " + f"[Chunk {on_chunk}/{total_chunks}]"
                if on_chunk == total_chunks == 0
                else ""
            )
            ible = tqdm.tqdm(dll, desc=colored(description, "cyan"), position=0, leave=False, colour="cyan")

        with torch.no_grad():
            for *X, labels in ible:
                if predict_mode:
                    assert torch.equal(
                        labels, torch.Tensor([-1] * len(labels)).to(self.device)
                    ), "Labels must be -1 for prediction."
                X = [x.to(self.device) for x in X]

                labels = labels.to(self.device)
                outputs = self.model.forward(*X)

                match self.criterion:
                    case torch.nn.CrossEntropyLoss():
                        labels = labels.long()
                        predictions = torch.argmax(outputs.data.cpu(), dim=1).cpu()
                    case torch.nn.BCEWithLogitsLoss():
                        labels = labels.float()
                        predictions = heaviside_cutoff(outputs, cutoff)
                        outputs = torch.sigmoid(outputs.data.cpu())

                outputs = outputs.unsqueeze(0) if outputs.dim() == 0 else outputs
                loss = self.criterion(outputs, labels)

                if "cuda" in str(self.device):
                    torch.cuda.empty_cache()

                match metric:
                    case "acc":
                        performance = sklearn.metrics.accuracy_score(labels.cpu(), predictions)
                    case "roc":
                        scores = outputs.data.cpu()
                        performance = protected_roc_auc_score(labels.cpu(), scores)

                all_labels += labels.cpu().numpy().tolist()
                all_outputs += outputs.data.cpu().numpy().tolist()
                all_preds += predictions.cpu().numpy().tolist()
                avg_perf += [performance] * len(labels)
                avg_loss += [loss.item()] * len(labels)

            avg_perf = sum(avg_perf) / len(avg_perf) if len(avg_perf) > 0 else 0.0
            avg_loss = sum(avg_loss) / len(avg_loss) if len(avg_loss) > 0 else 0.0
            sigmoided = torch.sigmoid(torch.Tensor(all_outputs)).data.cpu().numpy().tolist()

            return avg_perf, avg_loss, all_outputs, all_labels, all_preds, sigmoided

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
                eval_res = interface.eval(loader)
                outputs: list[float] = eval_res[-1]
                labels: list[int] = eval_res[3]
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
        if get_avg_roc_line:
            get_avg_roc(points_fpr, points_tpr, aucs, True)

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
        print_sample_outputs: bool = True,
        cutoff: float = 0.5,
        metric: Literal["roc", "acc"] = "roc",
        group: str = "Test Set",
    ) -> Tuple[list, list, list]:
        """
        Returns: list of predictions, list of outputs, list of ground-truth labels
        """
        print(colored("Status: Testing", "green"))
        predictions, outputs, _, labels = self.predict(
            test_loader,
            1,
            1,
            cutoff=cutoff,
            group=group,
            report_performance_test_mode=True,
            show_sample_output_test_mode=print_sample_outputs,
            metric=metric,
        )
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
    def get_input_size(dl, leave_out_last=True):
        inp_sizes = []
        assert isinstance(dl, torch.utils.data.DataLoader)
        dl = list(dl)
        iterab = dl[0] if not leave_out_last else dl[0][:-1]
        for X in iterab:
            assert isinstance(X, torch.Tensor)
            inp_sizes.append(list(X.size()))
        return inp_sizes

    @staticmethod
    def get_input_types(dl, leave_out_last=True):
        types = []
        assert isinstance(dl, torch.utils.data.DataLoader)
        dl = list(dl)
        iterab = dl[0] if not leave_out_last else dl[0][:-1]
        for X in iterab:
            assert isinstance(X, torch.Tensor)
            types.append(X.dtype)
        return types
