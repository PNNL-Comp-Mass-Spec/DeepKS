import gc
import random
import tempfile
import time
import psutil
import numpy as np, torch, os, sys, ast
from typing import Callable, Protocol
from torch.profiler import profile
import inspect
from copy import deepcopy
import multiprocessing, contextlib
from sympy import factorint

from DeepKS.config.join_first import join_first
import tracemalloc
from ..config.logging import get_logger
import memory_profiler

logger = get_logger()
"""The logger for this module"""

import numpy as np


@contextlib.contextmanager
def no_output():
    devnull = open(os.devnull, "w")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = devnull
    sys.stderr = devnull
    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        devnull.close()


def mem_mon(v, q, thsh=15, interv=0.001):
    i = 0
    init_avail = psutil.virtual_memory().available + psutil.swap_memory().free
    peak_mem_usage = 0
    try:
        while True:
            time.sleep(interv)
            cur_avail = psutil.virtual_memory().available + psutil.swap_memory().free
            peak_mem_usage = max((init_avail - cur_avail), peak_mem_usage)
            v.value = int(peak_mem_usage)

            pct_rem = 100 - (100 * ((init_avail - cur_avail) / init_avail))

            # Check if memory usage exceeds the limit
            i += 1
            if i == int(0.1 / interv):
                i = 1
            if pct_rem < thsh:
                raise RuntimeError("Memory threshold exceeded.")
    except Exception as e:
        q.put(e)


def rep_mem_wrapper(fn, wrapper_kwargs={}, *args, **kwargs):
    assert (
        multiprocessing.get_start_method() == "spawn"
    ), f"Multiprocessing start method needs to be 'spawn'. It was {multiprocessing.get_start_method()}."
    calculating_batch_size = wrapper_kwargs["calculating_batch_size"]
    while True:
        m = multiprocessing.Manager()
        v = m.Value("d", 0)
        q = m.Queue(maxsize=1)
        gc.collect()
        if True:
            mem_proc = multiprocessing.Process(target=mem_mon, args=(v, q))
            calc_proc = multiprocessing.Process(target=fn, args=args, kwargs=kwargs)
            other_err = ValueError("Initialization of error -- should never see this")
            try:
                mem_proc.start()
                calc_proc.start()  # inner(fn, v, *args, **kwargs)
                calc_proc.join()
                assert q.empty(), "Assert Memory threshold exceeded."
            except Exception as e:
                if (
                    str(e) == "Assert Memory threshold exceeded."
                    and str(other_err := q.get()) == "Memory threshold exceeded."
                ):
                    calc_proc.kill()
                    mem_proc.kill()
                    calc_proc.join()
                    mem_proc.join()
                    calc_proc.close()
                    mem_proc.close()
                    logger.warning(
                        "The memory threshold was exceeded which means the amount of memory required exceed the system"
                        " limits"
                    )
                    if calculating_batch_size == 1:
                        raise ValueError("The batch size reduced to 1, meaning one input cannot fit into memory.")
                    calculating_batch_size //= min(list(factorint(calculating_batch_size).keys()))
                    logger.warning(f"Reducing batch size to {calculating_batch_size}")
                    largs = list(args)
                    largs[3] = calculating_batch_size
                    args = tuple(largs)
                else:
                    print(str(e))
                    print(other_err)
                    raise
            else:
                calc_proc.close()
                mem_proc.kill()
                mem_proc.join()
                mem_proc.close()
                break
    return v.value, calculating_batch_size


class MemoryCalculator(Protocol):
    """Determine the amount of memory needed for a model and input, for a backward and forward pass, on a given device"""

    @staticmethod
    def calculate_memory(*args, **kwargs) -> tuple[int, int]:
        return (1, 1)

    @staticmethod
    def calculate_memory_todo(
        model: torch.nn.Module,
        input_like_no_batch: list[torch.Tensor],
        loss_steps: Callable[[torch.Tensor], None],
        calculating_batch_size: int = 250,
        reps: int = 1,
        safety_factor: float = 1.333,
        device: torch.device = torch.device("cpu"),
        cpu_no_compute=os.getenv("DOCKER_CONTAINER") is not None,
    ) -> tuple[int, int]:
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
            Tuple of (the number of bytes of memory needed for each additional input to a model, the number of bytes of memory needed for a single input --- aka the constant memory requirement.)

        """
        pass_through_cuda = []
        # if str(device) == "cpu" and cpu_no_compute:
        #     logger.info("Skipping memory estimation test (since we're on CPU). Assuming memory requirement values.")
        #     return int(5e6), int(5e7)
        calculating_batch_size_eventual = calculating_batch_size
        if False: ...
        elif str(device) == "cpu":
            logger.status("Calculating memory for batch_size == 1")
            one_tot_mem, _ = rep_mem_wrapper(
                MemoryCalculator._calculate_memory_core,
                {"calculating_batch_size": calculating_batch_size},
                model,
                input_like_no_batch,
                loss_steps,
                1,
                device,
                pass_through_cuda,
            )
            logger.status(f"Calculating memory for batch_size == {calculating_batch_size}")
            many_tot_mem, calculating_batch_size_eventual = rep_mem_wrapper(
                MemoryCalculator._calculate_memory_core,
                {"calculating_batch_size": calculating_batch_size},
                model,
                input_like_no_batch,
                loss_steps,
                calculating_batch_size,
                device,
                pass_through_cuda,
            )
            assert many_tot_mem > one_tot_mem, f"{many_tot_mem}, {one_tot_mem}"

        else:
            logger.status("Calculating memory for batch_size == 1")
            MemoryCalculator._calculate_memory_core(
                model, input_like_no_batch, loss_steps, 1, device, pass_through_cuda
            )
            one_tot_mem = pass_through_cuda[0]
            del pass_through_cuda[0]
            logger.status(f"Calculating memory for batch_size == {calculating_batch_size}")
            MemoryCalculator._calculate_memory_core(
                model, input_like_no_batch, loss_steps, calculating_batch_size, device, pass_through_cuda
            )
            many_tot_mem = pass_through_cuda[0]

        ## one_tot_mem = constant_mem + var_mem
        ## many_tot_mem = constant_mem + var_mem * calculating_batch_size
        ## many_tot_mem - one_tot_mem = var_mem * (calculating_batch_size - 1)
        ## var_mem = (many_tot_mem - one_tot_mem) / (calculating_batch_size - 1)
        var_mem = int((many_tot_mem - one_tot_mem) / (calculating_batch_size_eventual - 1)) + 1
        constant_mem = int(one_tot_mem - var_mem) + 1
        logger.info(f"Memory Needed as a Base for the Model: {constant_mem / 1024 / 1024:,.3f} MiB")
        logger.info(f"Memory Needed Per Additional Input: {var_mem / 1024 / 1024:,.3f} MiB")
        return var_mem, constant_mem

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

        if "cuda" in str(device):
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(device)
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
            pass_through_cuda.append(torch.cuda.max_memory_allocated(device))


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


# class SampleModel(torch.nn.Module):
#     def __init__(self, ll_size, out_size):
#         super().__init__()
#         self.ll = torch.nn.Linear(ll_size, out_size, dtype=torch.float32)

#     def forward(self, X):
#         return self.ll(X).squeeze(1)


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


class SampleModel0(torch.nn.Module):
    def __init__(self, ll_size_1, ll_size_2, out_size):
        super().__init__()
        self.ll1 = torch.nn.Linear(ll_size_1, out_size, dtype=torch.float32)
        self.ll2 = torch.nn.Linear(ll_size_2, out_size, dtype=torch.float32)

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


def loss(outputs):
    l = torch.nn.BCEWithLogitsLoss()
    ll = l(outputs.cpu(), torch.rand(outputs.shape[0], device="cpu"))
    ll.backward()


def main():
    model = SampleModel0(12345, 54321, 1)
    torch.random.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    input1 = torch.randn(12345, dtype=torch.float32)
    input2 = torch.randn(54321, dtype=torch.float32)
    device = torch.device("cpu")
    # device = torch.device("cuda:4")
    return (
        MemoryCalculator.calculate_memory(model, [input1, input2], loss, device=device, calculating_batch_size=100)[0]
        / 1024
        / 1024
    )


# def main2():
#     model = SampleModel(100000, 1)
#     torch.random.manual_seed(0)
#     torch.use_deterministic_algorithms(True)
#     input = torch.randn(10000, dtype=torch.float32)

#     def loss(outputs):
#         l = torch.nn.BCEWithLogitsLoss()
#         ll = l(outputs, torch.rand(outputs.shape[0]))
#         ll.backward()

#     return MemoryCalculator.calculate_memory(model, [input], loss, calculating_batch_size=100)[0] / 1024 / 1024


def main3():
    from ..models.KinaseSubstrateRelationshipATTN import KinaseSubstrateRelationshipATTN

    model = KinaseSubstrateRelationshipATTN()
    input1 = torch.randint(0, 22, (15,), dtype=torch.int32)
    input2 = torch.randint(0, 22, (2064,), dtype=torch.int32)
    device = torch.device("cpu")
    # device = torch.device("cuda:4")
    return (
        MemoryCalculator.calculate_memory(model, [input1, input2], loss, device=device, calculating_batch_size=200)[0]
        / 1024
        / 1024
    )


if __name__ == "__main__":
    main3()
