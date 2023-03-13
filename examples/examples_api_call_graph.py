import os, pathlib
from ..config.cfg import API_IMPORT_MODE
open(str(pathlib.Path(__file__).parent.resolve())+"/../config/API_IMPORT_MODE.json", "w").write("true")

from .examples import EXAMPLES

from ..tools.make_call_graph import DeepKSCallGraph
from ..api import main
from pycallgraph2.output import GraphvizOutput


DeepKSCallGraph(
    other_config={"include_stdlib": True, "groups": True},
    other_output={
        "output_file": os.path.abspath(os.path.expanduser("~/Desktop/call_flow_api.pdf")),
        "output_type": "pdf",
    },
    exclude_globs=["__len__", "__getitem__", r"torch._tensor.Tensor.__array__", "torch._tensor.Tensor.storage"],#, r"\.<module>$"],
    include_globs=[
        r"torch\._tensor\.Tensor\.*",
        r"torch\.optim\.optimizer\.[^\.]+$",
        r"torch\.nn\.modules\.module\.Module\.[^\.]+$",
    ],
).make_call_graph(
    lambda: main.make_predictions(**main.parse_api()),
    EXAMPLES[3][1:],
    output_class=GraphvizOutput,
)
