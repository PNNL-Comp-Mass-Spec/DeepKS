import numpy as np, torch, sys, inspect
from typing import Any, Callable, Protocol
from ..config.logging import get_logger
from torchinfo import summary

logger = get_logger()
"""The logger for this module"""


class MemoryCalculator(Protocol):
    """Determine the amount of memory needed for a model and input, for a backward and forward pass, on a given device
    """

    @staticmethod
    def calculate_memory(
        model: torch.nn.Module,
        input_like_no_batch: list[torch.Tensor],
        calculating_batch_size: int = 100,
        safety_factor: float = 1,
        device: torch.device = torch.device("cpu"),
    ) -> int:
        """Calculate the memory needed for a model and input, for a backward and forward pass

        Parameters
        ----------
        model :
            The model for which to calculate memory requirements
        input_like_no_batch :
            A list of Tensors of inputs to the model with the desired input shape, without the batch dimension
        loss_steps :
            A function that takes the output of the model and performs the loss calculation and backward pass
        calculating_batch_size :
            The batch size to use when calculating memory requirements
        safety_factor :
            A factor to multiply the result by, to account for any additional memory requirements


        Returns
        -------
            The number of bytes of memory needed for a model and input, for a backward and forward pass

        """
        try:
            input_ = [
                torch.randn(calculating_batch_size, *inp.shape, dtype=input_like_no_batch[0].dtype, device=device)
                for inp in input_like_no_batch
            ]
        except RuntimeError as e:
            if "not implemented for 'Int'" in str(e):
                input_ = [
                    torch.randint(
                        0, 1, (calculating_batch_size, *inp.shape), dtype=input_like_no_batch[0].dtype, device=device
                    )
                    for inp in input_like_no_batch
                ]
            else:
                raise e from None

        summary_args = {"model": model.to(device), "input_data": input_, "verbose": False, "device": device}
        summary_locals = get_locals(summary, **summary_args)
        model_stats = summary_locals["results"]
        input_bytes, output_bytes, param_bytes = (
            model_stats.total_input,
            model_stats.total_output_bytes,
            model_stats.total_param_bytes,
        )

        logger.debug(f"input bytes: {input_bytes} | param bytes: {param_bytes} | output bytes: {output_bytes}")
        full_size = input_bytes + output_bytes + param_bytes
        res = full_size // calculating_batch_size
        needed = int(np.ceil(np.mean(res) * safety_factor))
        logger.info(f"MB Needed Per Input = {needed/1024/1024:,.3f}")
        return needed


class persistent_locals(object):
    """Class to enable wrapping a function so that its inner local variables are exposed.

    Notes
    -----
    Based on https://stackoverflow.com/a/9187022/16158339
    """

    def __init__(self, func):
        self._locals = {}
        self.func = func

    def __call__(self, *args, **kwargs):
        def tracer(frame, event, arg):
            if event == "return":
                self._locals = frame.f_locals.copy()

        # tracer is activated on next call, return or exception
        sys.setprofile(tracer)
        try:
            # trace the function call
            res = self.func(*args, **kwargs)
        finally:
            # disable tracer and replace with old one
            sys.setprofile(None)
        return res

    def clear_locals(self):
        self._locals = {}

    @property
    def locals(self):
        return self._locals


def get_locals(any_function: Callable, *args, **kwargs) -> dict[str, Any]:
    """Function that works with `persistent_locals` to obtain local variables of a function.

    Parameters
    ----------
    any_function :
        The function from which to obtain local variables.
    args :
        Positional arguments to pass to ``any_function``.
    kwargs :
        Keyword arguments to pass to ``any_function``.

    Returns
    -------
        A dictionary mapping variable names to variable values
    """
    dec_func = persistent_locals(any_function)
    dec_func(*args, **kwargs)

    func_args = inspect.signature(any_function).parameters.keys()
    all_locals = dec_func.locals.items()
    user_defined_vars = {str(var): val for var, val in all_locals if var not in func_args}
    return user_defined_vars


def main():
    class SampleModel0(torch.nn.Module):
        def __init__(self, ll_size_1, ll_size_2, out_size):
            super().__init__()
            self.ll1 = torch.nn.Linear(ll_size_1, out_size, dtype=torch.float32)
            self.ll2 = torch.nn.Linear(ll_size_2, out_size, dtype=torch.float32)

            class Concat(torch.nn.Module):
                def __init__(self, *args, **kwargs) -> None:
                    super().__init__(*args, **kwargs)

                def forward(self, x, y):
                    return torch.concat([x, y], dim=1)

            class Squeeze(torch.nn.Module):
                def __init__(self, *args, **kwargs) -> None:
                    super().__init__(*args, **kwargs)

                def forward(self, X):
                    return torch.squeeze(X, dim=-1)

            self.concat = Concat()
            self.squeeze = Squeeze()
            self.final_fc = torch.nn.Linear(2 * out_size, 1)

        def forward(self, X1, X2):
            x1 = self.ll1(X1)
            x2 = self.ll2(X2)
            out = self.concat(x1, x2)
            final = self.final_fc(out)
            final = self.squeeze(final)
            return final

    model = SampleModel0(10000, 5000, 1)
    torch.random.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    input1 = torch.randn(10000, dtype=torch.float32)
    input2 = torch.randn(5000, dtype=torch.float32)
    return MemoryCalculator.calculate_memory(model, [input1, input2], calculating_batch_size=10)


def main2():
    class SampleModel(torch.nn.Module):
        def __init__(self, ll_size, out_size):
            super().__init__()
            self.ll = torch.nn.Linear(ll_size, out_size, dtype=torch.float32)

        def forward(self, X):
            return self.ll(X).squeeze(1)

    model = SampleModel(10000, 1)
    torch.random.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    input = torch.randn(10000, dtype=torch.float32)
    return MemoryCalculator.calculate_memory(model, [input], calculating_batch_size=10)
