from pycallgraph2 import PyCallGraph
from pycallgraph2.output import GraphvizOutput
from pycallgraph2.config import Config
import sys, re


class GlobbingFilter(object):
    """Filter module names using a set of globs.

    Objects are matched against the exclude list first, then the include list.
    Anything that passes through without matching either, is excluded.
    """

    def __init__(self, include=None, exclude=None):
        if include is None and exclude is None:
            include = [".*"]
            exclude = []
        elif include is None:
            include = [".*"]
        elif exclude is None:
            exclude = []

        self.include = include
        self.exclude = exclude

    def __call__(self, full_name=""):
        assert isinstance(self.exclude, list)
        for pattern in self.exclude:
            if isinstance(pattern, str):
                if re.search(pattern, full_name):
                    return False

        for pattern in self.include:
            if re.search(pattern, full_name):
                return True

        return False


class DeepKSCallGraph:
    def __init__(
        self,
        keep_all_from="DeepKS",
        exclude_globs=[
            # r"pycallgraph\..*",
            # r".*<listcomp>.*",
            # r".*<dictcomp>.*",
            # r".*<setcomp>.*",
            # r"^_[^_].*",
            # r"__(?!(init|str)).*__",
            # r"^<lambda>$"
        ],
        include_globs=[r".*"],
        other_config={},
        other_output={},
    ):
        self.exclude_globs = exclude_globs
        outer_module = re.sub(r"<module '(.*)' from.*", r"\1", str(sys.modules[__name__])).split(".")[0]
        # print(f"{outer_module=}")
        self.include_globs = include_globs# + [r"(^__main__$|^" + outer_module + r"\..*$)"]
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
    from . import make_call_graph_demonstrator

    DeepKSCallGraph(other_output={'output_file': './demonstrated_call_graph.png'}).make_call_graph(
        make_call_graph_demonstrator.dummy_fn,
        [],
    )

    from . import custom_tree_maker

    DeepKSCallGraph(other_output={'output_file': './demonstrated_call_graph_tree.png'}).make_call_graph(
        custom_tree_maker.main,
        [
            "-f",
            "-d",
            "/Users/druc594/Library/CloudStorage/OneDrive-PNNL/Desktop/DeepKS_/DeepKS/docs/tree_description.txt",
        ],
    )

    print("Uncomment to view examples")
