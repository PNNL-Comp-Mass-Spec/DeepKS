"""Contains `NNInterface`, A flexible class that allows for training, validation, testing, and summarization of a `torch.nn.Module`-based neural network
"""

from __future__ import annotations
from cProfile import label
import json, torch, torch.nn, torch.utils.data, sklearn.metrics, numpy as np, tqdm, io
import os, itertools, pathlib, typing
from typing import Any, Tuple, Union, Literal, Iterable
from prettytable import PrettyTable
from torchinfo_modified import summary
from termcolor import colored
from .roc_helpers import ROCHelpers

protected_roc_auc_score = ROCHelpers.protected_roc_auc_score
"""See `ROCHelpers.protected_roc_auc_score`"""

from ..config.join_first import join_first


def heaviside_cutoff(outputs: torch.Tensor, cutoff: float) -> torch.Tensor:
    """A helper function to convert raw outputs to binary predictions

    Parameters
    ----------
    outputs :
        The raw outputs
    cutoff :
        The cutoff to use

    Returns
    -------
        The binary predictions
    """
    return torch.heaviside(torch.sigmoid(outputs.data.cpu()).cpu() - cutoff, values=torch.tensor([0.0]))


def expand_metric(met: str) -> str:
    """A helper function to expand a metric ("roc" or "acc") abbreviation to its full name

    Parameters
    ----------
    met :
        The metric to expand

    Returns
    -------
        The expanded metric
    """
    return "ROC AUC" if met == "roc" else "Accuracy" if met == "acc" else met


expand_metric = lambda met: "ROC AUC" if met == "roc" else "Accuracy" if met == "acc" else met

from ..config.logging import get_logger
from ..tools.custom_logging import CustomLogger

logger = get_logger()
"""The logger for this module."""


class NNInterface:
    """A flexible class that allows for training, validation, testing, and summarization of a `torch.nn.Module`-based neural network
    """

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

        Parameters
        ----------
        model_to_train:
            The model to train
        loss_fn:
            The loss function to use
        optim:
            The optimizer to use
        inp_size:
            The input size of the model
        inp_types:
            The input types of the model
        model_summary_name:
            The name of the file to write the model summary to
        device:
            The device to use for training

        Raises
        ------
            AssertionError: If the loss function is not `torch.nn.BCEWithLogitsLoss` or `torch.nn.CrossEntropyLoss`
        """
        self.model: torch.nn.Module = model_to_train.to(device)
        """The model to train"""
        self.criterion = loss_fn
        """The loss function to use"""
        assert type(self.criterion) in [
            torch.nn.BCEWithLogitsLoss,
            torch.nn.CrossEntropyLoss,
        ], "Loss function must be either `torch.nn.BCEWithLogitsLoss` or `torch.nn.CrossEntropyLoss`."
        self.optimizer: torch.optim.Optimizer = optim
        """The optimizer to use"""
        self.device = device
        """The device with which to compute"""
        self.inp_size = inp_size
        """The input shape for the model"""
        self.inp_types = inp_types
        """The input types for the model"""
        self.model_summary_name = model_summary_name
        """The name of the file to write the model summary to"""
        if inp_size is None and inp_types is None and model_summary_name is None:
            return
        if inp_size is not None and inp_types is None and isinstance(self.model_summary_name, str):
            self.write_model_summary()

    def write_model_summary(self):
        """Writes the model summary (calls `NNInterface.__str__`) to a file specified by `model_summary_name`."""
        logger.info(f"Writing model summary to file {self.model_summary_name}.")
        self.model_summary_name = join_first(self.model_summary_name, 0, __file__)
        if isinstance(self.model_summary_name, str):
            with open(self.model_summary_name, "w", encoding="utf-8") as f:
                str_rep = str(self)
                f.write(str_rep)

        elif isinstance(self.model_summary_name, io.StringIO):
            self.model_summary_name.write(str(self))

    def __str__(self):
        """Gets the model summary as a string by using `torchinfo.summary` and returns it.

        Returns
        -------
        str:
            The model summary as a string
        """
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
            """The model summary as a string; the string representation of ``self``"""
            self.model_summary = ms
            """The model summary as a `torchinfo.summary.Summary` object"""
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception as e:
            print("Failed to run model summary:", flush=True)
            print(e, flush=True)
            raise
        return self.representation

    def report_prog(
        self,
        paradigm: str,
        epoch: int,
        num_epochs: int,
        batch_num: Union[int, str],
        chunk_num: int,
        total_step: int,
        loss: float,
        score: float,
        metric: Literal["acc", "roc"],
        print_every: int = 1,
    ):
        """Reports the progress of the training/validation process, including the loss, score, and metric.

        Parameters
        ----------
        paradigm :
            Training or Validation
        epoch :
            The epoch we are on
        num_epochs :
            The total number of epochs we are training for
        batch_num :
            The batch number we are on or "All" if we are reporting the average loss over the batch
        chunk_num :
            The chunk number we are on. There may be multiple "chunks" per epoch if the dataset is too large to fit in memory.
        total_step :
            The total number of batches we are training for
        loss :
            The resultant loss
        score :
            The resultant score
        metric :
            The metric of ``score``
        print_every : int, optional
            Returns after doing nothing if ``batch_num % print_every != 0``, by default 1
        """
        if not (isinstance(batch_num, str) or batch_num % print_every == 0):
            return
        if "validation" in paradigm.lower():
            method = CustomLogger.valinfo
        elif "train" in paradigm.lower():
            method = CustomLogger.trinfo
        else:
            method = CustomLogger.info

        if isinstance(batch_num, str):
            batch_str = "All"
            prepend_str = "(Means) "
        elif isinstance(batch_num, int):
            batch_str = batch_num + 1
            prepend_str = ""
        else:
            raise TypeError(f"batch_num must be either a string or an int, not {type(batch_num)}")

        method(
            logger,
            (
                f"{prepend_str}Chunk [{chunk_num + 1}] | Epoch [{epoch + 1}/{num_epochs}] | Batch"
                f" [{batch_str}/{total_step}] | Loss {loss:.4f} | {expand_metric(metric)} {score:.2f}"
            ),
        )

    def train(
        self,
        loader_iterable: Iterable[tuple[torch.utils.data.DataLoader, dict[str, Any]]],
        num_epochs: int = 50,
        lr_decay_amount: float = 1.0,
        lr_decay_freq: int = 1,
        threshold: float = float("inf"),
        val_dl: torch.utils.data.DataLoader | None = None,
        cutoff: float = 0.5,
        metric: Literal["roc", "acc"] = "roc",
        extra_description: str = "",
        pass_through_scores: dict | None = None,
        **training_kwargs,
    ):
        """Trains a `torch.nn.Module` -based neural network model.

        Parameters
        ----------
        loader_iterable :
            A generator that yields `torch.utils.data.DataLoader` s
        num_epochs : optional
            The number of epochs for which to train, by default 50
        lr_decay_amount : float, optional
            The amount by which the learning rate will be multiplied for every ``lr_decay_freq``'th epoch, by default 1.0
        lr_decay_freq : optional
            For an epoch this often, the learning rate gets multiplied by ``lr_decay_amount``, by default 1
        threshold : optional
            The training process will stop if the lowest loss encountered by a batch goes below this value, by default float("inf")
        val_dl : optional
            The validation `torch.utils.data.DataLoader`, by default `None`
        cutoff : optional
            If ``metric`` is ``acc`` the score cutoff for deciding label, by default 0.5
        metric : optional
            The scoring metric, by default "roc"
        extra_description :  optional
            An extra descriptor used to identify which data being trained, by default ""
        pass_through_scores : optional
            If a `dict`, an empty `dict` can be passed in here to collect the resultant validation score, as this function will map ``extra_description`` to the resultant validation score, by default `None`
        training_kwargs : optional
            Additional, option keyword arguments. They can be any of the following:
                - ``loss_below``: The loss below which the training process ends early
                - ``loss_chances``: The number of chances the loss has to go below ``loss_below`` before the training process ends early
                - ``val_le``: In order to stop training early, the validation score must be less than this value

        Returns
        -------
        None
            Trains, but does not return anything.
        """
        assert isinstance(cutoff, (int, float)), "Cutoff needs to be a number."
        assert (
            type(self.criterion) == torch.nn.BCEWithLogitsLoss or type(self.criterion) == torch.nn.CrossEntropyLoss
        ), "Loss function must be either `torch.nn.BCEWithLogitsLoss` or `torch.nn.CrossEntropyLoss`."

        logger.status(f"Training {extra_description}")

        lowest_loss = float("inf")
        total_step = -1
        epoch = 0

        # While loop to train the model
        cycler = itertools.cycle(loader_iterable)
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
            for train_loader, info_dict in cycler:
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
                    if outputs.size() == torch.Size([]):
                        outputs = outputs.reshape([1])
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

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
                    logger.debug(f"{self.device=}")
                    self.optimizer.step()

                    # Report Progress
                    performance_score = None
                    performance_score, _ = self._get_acc_or_auc_and_predictions(outputs, labels, metric, cutoff)
                    self.report_prog(
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
                    )

                    lowest_loss = min(lowest_loss, loss.item())
                    losses.append(loss.item())
                    train_scores.append(performance_score)
            try:
                assert len(train_scores) != 0, "No Data Trained"
            except AssertionError as e:
                logger.warning(str(e))

            if losses:
                mean_loss = sum(losses) / len(losses)
            else:
                mean_loss = float("NaN")

            if train_scores:
                mean_score = sum(train_scores) / len(train_scores)
            else:
                mean_score = float("NaN")

            self.report_prog("Mean Train", epoch, num_epochs, "", chunk_num, total_step, mean_loss, mean_score, metric)

            score, all_outputs = -1, [-1]
            # Validate
            if val_dl is not None:
                print(" " * os.get_terminal_size().columns, end="\r")
                logger.status("Validating")
                total_step = len(val_dl)
                score, val_loss, _, all_outputs, _ = self.eval(val_dl, cutoff, metric, display_pb=False)
                self.report_prog("Validation", epoch, num_epochs, 0, 0, 1, val_loss, score, metric)
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
        dataloader: torch.utils.data.DataLoader,
        on_chunk: int,
        total_chunks: int,
        cutoff: float = 0.5,
        group: str = "UNKNOWN GROUP",
        metric: Literal["roc", "acc"] = "roc",
        display_pb: bool = False,
    ) -> Tuple[list, list, list, list]:
        """Obtains prediction from `model`, of a specified dataloader. Essentially a special case of `NNInterface.eval`.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            The dataloader to obtain predictions from.
        on_chunk : int
            The chunk number of the dataloader. There may be multiple "chunks" we want to obtain predictions for, if the dataset is too large to fit in memory.
        total_chunks : int
            The total number of chunks we will be obtaining predictions for.
        cutoff : float, optional
            If ``metric`` is ``acc`` the score cutoff for deciding label, by default 0.5
        group : str, optional
            The group name we are obtaining predictions for, by default "UNKNOWN GROUP"
        metric : optional
            The scoring metric, by default "roc"
        display_pb : bool, optional
            Whether to display a progress bar, by default False

        Returns
        -------
            A list of predictions, a list of outputs, a list of groups, and a list of labels.
        """

        _, _, outputs, labels, predictions = self.eval(dataloader, cutoff, metric, display_pb=display_pb)

        return [bool(x) for x in predictions], outputs, [group] * len(outputs), labels

    def _get_acc_or_auc_and_predictions(
        self, outputs: torch.Tensor, labels: torch.Tensor, metric: Literal["acc", "roc"], cutoff=0.5
    ) -> Tuple[float, np.ndarray]:
        """Obtains the accuracy or AUC score, and predictions, from a set of outputs and labels.

        Parameters
        ----------
        outputs : torch.Tensor
            The outputs from the model.
        labels : torch.Tensor
            The labels from the dataset.
        metric : Literal[&quot;acc&quot;, &quot;roc&quot;]
            The metric to use.
        cutoff : float, optional
            The cutoff for the metric, by default 0.5

        Returns
        -------
            Performance score, and predictions.

        Raises
        ------
        NotImplementedError
            If `criterion` does not have an implementated case in this function.
        """
        predictions: np.ndarray
        match self.criterion:
            case torch.nn.CrossEntropyLoss():
                predictions = torch.argmax(outputs.data.cpu(), dim=1).cpu().numpy()
            case torch.nn.BCEWithLogitsLoss():
                outputs = torch.sigmoid(outputs.data.cpu())
                predictions = heaviside_cutoff(outputs, cutoff).cpu().numpy()
            case _:
                raise NotImplementedError(f"Loss function {self.criterion} not implemented.")
        match metric:
            case "acc":
                performance = sklearn.metrics.accuracy_score(labels.cpu(), predictions)
            case "roc":
                scores = outputs.data.cpu()
                performance = protected_roc_auc_score(labels.cpu(), scores)

        return float(performance), predictions

    def eval(
        self,
        dataloader: torch.utils.data.DataLoader,
        cutoff: float = 0.5,
        metric: typing.Literal["roc", "acc"] = "roc",
        display_pb: bool = False,
    ) -> Tuple[float, float, list[float], list[int], list[int]]:
        """Obtains prediction from `model`, of a specified dataloader.

        Parameters
        ----------
        dataloader :
            The dataloader to obtain predictions from.
        cutoff : optional
            If ``metric`` is ``acc`` the score cutoff for deciding label, by default 0.5
        metric : optional
            The scoring metric, by default "roc"
        display_pb : optional
            Whether to display a progress bar, by default False

        Returns
        -------
            Average performance across all batches of ``dataloader``, average loss across all batches of ``dataloader``, all output scores, all true labels (if legitimate true labels are put into ``dataloader``), all predictions.
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

                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)

                match self.criterion:
                    case torch.nn.CrossEntropyLoss():
                        labels = labels.long()
                    case torch.nn.BCEWithLogitsLoss():
                        labels = labels.float()

                loss = self.criterion(outputs, labels)
                outputs = torch.sigmoid(outputs)

                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

                performance, predictions = self._get_acc_or_auc_and_predictions(outputs, labels, metric, cutoff)

                all_labels += labels.cpu().numpy().tolist()
                all_outputs += outputs.data.cpu().numpy().tolist()
                all_preds += predictions.tolist()
                avg_perf += [performance] * len(labels)
                avg_loss += [loss.item()] * len(labels)

            if len(avg_perf) > 0:
                avg_perf = sum(avg_perf) / len(avg_perf)
            else:
                avg_perf = 0.0

            if len(avg_loss) > 0:
                avg_loss = sum(avg_loss) / len(avg_loss)
            else:
                avg_loss = 0.0

            return avg_perf, avg_loss, all_outputs, all_labels, all_preds

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
        performance, _, outputs, labels, predictions = self.eval(test_loader, cutoff, metric, display_pb=False)
        if metric == "acc":
            pct = "%"
        else:
            pct = ""
        logger.teinfo(f"{expand_metric(metric)}: {performance:.3f} {pct}")

        if print_sample_predictions:
            tab = PrettyTable(["Input Index", "Ground Truth Label", "Prediction"], min_width=20)
            num_rows = min(10, len(labels))
            tab.title = f"Info: First {num_rows} labels mapped to predictions"
            for i in range(num_rows):
                if not int_label_to_str_label:
                    pred_print = predictions[i]
                else:
                    pred_print = int_label_to_str_label[predictions[i]]

                if not int_label_to_str_label:
                    lab_print = labels[i]
                else:
                    lab_print = int_label_to_str_label[labels[i]]
                tab.add_row([i, lab_print, pred_print])
            logger.teinfo(str(tab))

        return predictions, outputs, labels

    def save_eval_results(
        self, loader: torch.utils.data.DataLoader, path: str, kin_order: Union[list[str], None] = None
    ):
        """Save evaluation results to a JSON file.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            The data loader to evaluate on.
        path : str
            The path to save the results to.
        kin_order : optional
            The order of kinases, corresponding to the predictions in the output, by default None
        """
        eval_res = self.eval(dataloader=loader)
        outputs = eval_res[-3]
        labels = eval_res[-2]
        results_dict = {"labels": labels, "outputs": outputs}
        if kin_order is not None:
            results_dict.update({"kin_order": kin_order})
        with open(path, "w") as f:
            json.dump(results_dict, f, indent=4)

    @staticmethod
    def get_input_size(dl: torch.utils.data.DataLoader, leave_out_last=True) -> list[list[int]]:
        """Get the input size of tensors in a `torch.utils.data.DataLoader`.

        Parameters
        ----------
        dl : torch.utils.data.DataLoader
            The dataloader to get the input size of.
        leave_out_last : bool, optional
            Whether to leave out the last batch, as this may have a different size, by default True

        Returns
        -------
            A list of lists of ints, where each list of ints is the shape of a(n) input tensor.
        """
        inp_sizes: list[list[int]] = []
        assert isinstance(dl, torch.utils.data.DataLoader), "dl must be a DataLoader, was {}".format(type(dl))
        dll = list(dl)
        if not leave_out_last:
            iterab = dll[0]
        else:
            iterab = dll[0][:-1]
        for X in iterab:
            assert isinstance(X, torch.Tensor)
            inp_sizes.append(list(X.size()))
        return inp_sizes

    @staticmethod
    def get_input_types(dl: torch.utils.data.DataLoader, leave_out_last=True):
        """Get the input types of tensors in a `torch.utils.data.DataLoader`.

        Parameters
        ----------
        dl : torch.utils.data.DataLoader
            The DataLoader to get the input types of.
        leave_out_last : bool, optional
            Whether to leave out the last batch, for consistency with `NNInterface.get_input_size`, by default True

        Returns
        -------
            A list of types, where each type is the type of a(n) input tensor.
        """
        types = []
        assert isinstance(dl, torch.utils.data.DataLoader)
        dll = list(dl)
        if not leave_out_last:
            iterab = dll[0]
        else:
            iterab = dll[0][:-1]

        for X in iterab:
            assert isinstance(X, torch.Tensor)
            types.append(X.dtype)
        return types
