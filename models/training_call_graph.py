from ..tools.make_call_graph import DeepKSCallGraph
from ..models.individual_classifiers import main
from pycallgraph2.output import GephiOutput, GraphvizOutput
import tqdm, contextlib, sys, time, os
from tqdm.contrib import DummyTqdmFile

@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    orig_err = sys.stderr
    sys.stderr = sys.stdout
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err
        sys.stderr = orig_err

# def test_reg_print(i):
#     print("This is a standard print statement.", i)
#     print("This is an stderr print statement.", i, file=sys.stderr)

# def test_bar():
#     with std_out_err_redirect_tqdm() as orig_stdout:
#         with tqdm.tqdm(total=100, dynamic_ncols=True, position=0, leave=False) as pbar:
#             for _ in range(100):
#                 pbar.update(1)
#                 time.sleep(0.5)
#                 if _ % 5 == 0:
#                     test_reg_print(_)

#     print("Done!")


# test_bar()
DeepKSCallGraph(other_config={'include_stdlib': True, 'groups': True}, other_output={'output_file': os.path.abspath(os.path.expanduser('~/Desktop/call_flow.pdf')), 'output_type': 'pdf'}, exclude_globs=["__len__", "__getitem__", r"torch._tensor.Tensor.__array__", "torch._tensor.Tensor.storage"], include_globs=[r"torch\._tensor\.Tensor\..*", r"torch\.optim\.optimizer\.[^\.]+$"]).make_call_graph(
    main,
    [
        "--train",
        "~/Desktop/DeepKS_/DeepKS/data/raw_data_trunc_200.csv",
        "--val",
        "~/Desktop/DeepKS_/DeepKS/data/raw_data_trunc_105.csv",
        "--device",
        "cpu"
    ],
    output_class=GraphvizOutput
)
