import tempfile
import numpy as np, torch, os, sys, ast
from typing import Callable, Protocol
from torch.profiler import profile
import inspect
from copy import deepcopy

from DeepKS.config.join_first import join_first
import tracemalloc
from ..config.logging import get_logger
import memory_profiler

logger = get_logger()
"""The logger for this module"""


class MemoryCalculator(Protocol):
    """Determine the amount of memory needed for a model and input, for a backward and forward pass, on a given device
    """

    @staticmethod
    def calculate_memory(
        model: torch.nn.Module,
        input_like_no_batch: list[torch.Tensor],
        loss_steps: Callable[[torch.Tensor], None],
        calculating_batch_size: int = 100,
        reps: int = 1,
        safety_factor: float = 1.25,
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
        reps :
            The number of times to repeat the calculation, before taking the mean
        safety_factor :
            A factor to multiply the result by, to account for any additional memory requirements


        Returns
        -------
            The number of bytes of memory needed for a model and input, for a backward and forward pass

        """
        pass_through_cuda = []
        runit = lambda: MemoryCalculator._calculate_memory_core(
            model, input_like_no_batch, loss_steps, calculating_batch_size, device, pass_through_cuda
        )
        mp = memory_profiler.memory_usage(proc=runit, interval=0.01)  # type: ignore

        if not pass_through_cuda:
            res = max(mp) - min(mp)
        else:
            res = pass_through_cuda[0]
        logger.info(f"MB Needed Per Input = {res:,.3f}")
        return res

    @staticmethod
    def _calculate_memory_core(
        model: torch.nn.Module,
        input_like_no_batch: list[torch.Tensor],
        loss_steps: Callable[[torch.Tensor], None],
        calculating_batch_size: int = 5,
        device: torch.device = torch.device("cpu"),
        pass_through_cuda: list[int] = [],
    ):
        """
        Calculate the memory needed for a model and input, for a backward and forward pass. This function implements the core functionality of the calculate_memory function, but does not perform any averaging or safety factor multiplication.

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
        device :
            The device to use when calculating memory requirements"""
        print("here")
        if "cuda" in str(device):
            torch.cuda.reset_peak_memory_stats()
        try:
            input_ = [
                torch.randn(calculating_batch_size, *inp.shape, dtype=input_like_no_batch[0].dtype, device=device)
                for inp in input_like_no_batch
            ]
        except RuntimeError as e:
            if "Int" in str(e):
                input_ = [
                    torch.randint(
                        0,
                        1,
                        (calculating_batch_size, *inp.shape),
                        dtype=input_like_no_batch[0].dtype,
                        device=device,
                    )
                    for inp in input_like_no_batch
                ]
            else:
                raise e from None

        lt = (model.to(device))(*input_)
        loss_steps(lt)

        if "cuda" in str(device):
            pass_through_cuda.append(torch.cuda.max_memory_allocated())


def assert_unique_variable_names(func: Callable) -> bool:
    """Inspects a function and ensures that all assignment statements use unique variable names.

    Parameters
    ----------
    func :
        The function to check

    Raises
    ------
    AssertionError :
        A variable assignment is repeated

    Returns
    -------
        True, if all assignment statements are, in fact, unique.

    Notes
    -----
    Based on Chat GPT starter function.
    """
    tree = ast.parse(inspect.getsource(func).strip())
    assignment_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]

    assigned_variables = set()
    for node in assignment_nodes:
        for target in node.targets:
            if isinstance(target, ast.Name):
                if target.id in assigned_variables:
                    raise AssertionError(
                        f"In function `{func.__self__.__class__.__name__}.{func.__name__}`, the variable `{target.id}`"
                        " is reassigned. The function that calculates the required memory for the model requires that"
                        " all assignments in forward functions are to unique variable names. Please change the"
                        f" definition of `{func.__self__.__class__.__name__}.{func.__name__}` to fix this."
                    )
                assigned_variables.add(target.id)

    return True


class persistent_locals(object):
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


def get_local_tensors(any_function, *args, **kwargs):
    dec_func = persistent_locals(any_function)
    dec_func(*args, **kwargs)

    func_args = inspect.signature(any_function).parameters.keys()
    all_locals = dec_func.locals.items()
    user_defined_vars = {
        str(var): val for var, val in all_locals if var not in func_args and isinstance(val, torch.Tensor)
    }
    return user_defined_vars


class SampleModel(torch.nn.Module):
    def __init__(self, ll_size, out_size):
        super().__init__()
        self.ll = torch.nn.Linear(ll_size, out_size, dtype=torch.float32)

    def forward(self, X):
        return self.ll(X).squeeze(1)


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


import types


def get_user_defined_variables():
    frame = inspect.currentframe()
    assert frame is not None
    frame = frame.f_back
    assert frame is not None
    func_args = inspect.signature(types.FunctionType(frame.f_code, {})).parameters.keys()
    local_vars = frame.f_locals.items()
    user_defined_vars = {var: val for var, val in local_vars if var not in func_args and isinstance(val, torch.Tensor)}
    return user_defined_vars


def main():
    model = SampleModel0(1000, 500, 1)
    torch.random.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    input1 = torch.randn(1000, dtype=torch.float32)
    input2 = torch.randn(500, dtype=torch.float32)

    def loss(outputs):
        l = torch.nn.BCEWithLogitsLoss()
        ll = l(outputs, torch.rand(outputs.shape[0]))
        ll.backward()

    return MemoryCalculator.calculate_memory(model, [input1, input2], loss, calculating_batch_size=1000) / 1024 / 1024


def main2():
    model = SampleModel(100000, 1)
    torch.random.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    input = torch.randn(10000, dtype=torch.float32)

    def loss(outputs):
        l = torch.nn.BCEWithLogitsLoss()
        ll = l(outputs, torch.rand(outputs.shape[0]))
        ll.backward()

    return MemoryCalculator.calculate_memory(model, [input], loss, calculating_batch_size=100) / 1024 / 1024


if __name__ == "__main__":
    main()
