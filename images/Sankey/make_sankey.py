import pandas as pd
from psankey_modified.sankey import sankey
import re
from matplotlib import rcParams
from matplotlib import pyplot as plt

rcParams['font.family'] = 'Palatino'

size = dict(width=2000, height=300)

flows = pd.read_csv("flows.csv")

plot_order = flows.set_index('target').to_dict()['order']

label_dict = {x: {'label': re.sub("Derangement..", "Derange", x)} for x in flows['source'] if "Derangement" in x}
label_dict.update({x: {'label': re.sub("Removed Overlap.*", "Removed\nOverlaps", x)} for x in flows['source'] if "Removed Overlap" in x})

mod = {'D': dict(facecolor='green', edgecolor='black', alpha=1, label='D1', yPush=1)}
mod.update(label_dict)
nodes, fig, ax = sankey(flows, aspect_ratio=3, nodelabels=True, linklabels=False, labelsize=9, nodecmap='viridis', nodecolorby='level', nodealpha=0.5, nodeedgecolor='white', nodemodifier=mod, plotOrder=plot_order)
# fig.savefig("big_sankey.svg", bbox_inches = "tight", pad_inches = 0)
# fig.set_size_inches(fig.get_size_inches()[0], fig.get_size_inches()[1]*1.5)
fig.savefig("big_sankey.svg", transparent=True, bbox_inches='tight')
plt.show()