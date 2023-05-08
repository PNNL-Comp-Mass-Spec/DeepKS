import pandas as pd, re, pathlib, os
from psankey_modified.sankey import sankey
from matplotlib import rcParams
from ...tools.system_tools import os_system_and_get_stdout

os.chdir(pathlib.Path(__file__).parent.resolve())


def make_sankey():
    rcParams["font.family"] = "monospace"

    flows = pd.read_csv("./flows.csv")

    plot_order = flows.set_index("target").to_dict()["order"]

    label_dict = {x: {"label": re.sub("Derangement..", "Derange", x)} for x in flows["source"] if "Derangement" in x}
    label_dict.update(
        {
            x: {"label": re.sub("Removed Overlap.*", "Removed\nOverlaps", x)}
            for x in flows["source"]
            if "Removed Overlap" in x
        }
    )

    mod = {"D": dict(facecolor="green", edgecolor="black", alpha=1, label="D1", yPush=1)}
    mod.update(label_dict)
    _, fig, _ = sankey(
        flows,
        aspect_ratio=3,
        nodelabels=True,
        linklabels=False,
        labelsize=9,
        nodecmap="viridis",
        nodecolorby="level",
        nodealpha=0.5,
        nodeedgecolor="white",
        nodemodifier=mod,
        plotOrder=plot_order,
    )
    # fig.savefig("big_sankey.svg", bbox_inches = "tight", pad_inches = 0)
    # fig.set_size_inches(fig.get_size_inches()[0], fig.get_size_inches()[1]*1.5)
    fig.savefig("big_sankey.pdf", transparent=True, bbox_inches="tight")
    # plt.show()
    os_system_and_get_stdout("inkscape -D big_sankey.svg -o big_sankey.svg", prepend="[inkscape] ")


if __name__ == "__main__": # pragma: no cover
    make_sankey()
