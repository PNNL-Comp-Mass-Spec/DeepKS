from pycallgraph2 import PyCallGraph
from pycallgraph2.output import GraphvizOutput
from pycallgraph2.config import Config
from pycallgraph2.globbing_filter import GlobbingFilter
from . import custom_tree_maker
import sys


class DeepKSCallGraph:
    def __init__(
        self,
        exclude_globs=["pycallgraph.*", "*.<listcomp>", "*.<dictcomp>"],
        include_globs=["*"],
        other_config={},
        other_output={},
    ):
        self.exclude_globs = exclude_globs
        self.include_globs = include_globs
        self.other_config = other_config
        self.other_output = other_output

    def make_call_graph(self, function, cmdline_args):
        with PyCallGraph(
            output=GraphvizOutput(**self.other_output),
            config=Config(trace_filter=GlobbingFilter(exclude=self.exclude_globs, include=self.include_globs)),
        ):
            sys.argv += cmdline_args
            function()


if __name__ == "__main__":
    DeepKSCallGraph().make_call_graph(
        custom_tree_maker.main,
        [
            "-f",
            "-d",
            "/Users/druc594/Library/CloudStorage/OneDrive-PNNL/Desktop/DeepKS_/DeepKS/docs/tree_description.txt",
        ],
    )
