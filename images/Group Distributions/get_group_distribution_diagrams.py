import itertools, collections
import matplotlib as mpl, matplotlib.pyplot as plt, textwrap
from brokenaxes import brokenaxes
import pandas as pd
import sys
sys.path.append("../../data/preprocessing/")
from PreprocessingSteps import split_into_sets

mpl.rcParams['font.family'] = "Palatino"

def get_group_distribution_diagrams(split_dict):
    global labs
    labs = sorted(list(set(itertools.chain(*[split_dict[s]['group distribution'] for s in split_dict])))) + ["TOTALS"]
    res = []
    for split in split_dict:
        grp_cnt = collections.defaultdict(int)
        for x in split_dict[split]['num kins']:
            grp_cnt[x[1]] += x[0]

        total = sum(grp_cnt.values())

        grp_cnt.update({"TOTALS": total})
        res.append(grp_cnt)
    return tuple(res)

def get_overall_plot(counts, savefig = False):
    cnt_tr = counts[0]
    cnt_vl = counts[1]
    cnt_te = counts[2]


    fig = plt.figure(figsize=(12, 5.5))
    baxes = brokenaxes(ylims = ((0, 8000), (15000, 22000)), hspace = 0.1)
    baxes.bar(labs, align = 'center', height=[cnt_tr[la] for la in labs], color = "green", alpha = 1, label = "Train")
    baxes.bar(labs, align = 'center', height=[cnt_vl[la] for la in labs], color = "aqua", alpha = 1, bottom = [cnt_tr[la] for la in labs], label = "Dev")
    baxes.bar(labs, align = 'center', height=[cnt_te[la] for la in labs], color = "blue", alpha = 1, bottom = [cnt_tr[la] + cnt_vl[la] for la in labs], label = "Test")
    baxes.legend(loc = "upper left")
    baxes.set_xlabel("Kinase Group")
    baxes.set_ylabel("Number of input pairs")
    baxes.set_title("Distribution of model inputs by kinase group")
    baxes.set_xticks(list(range(10)))
    baxes.axs[0].set_yticks(list(range(15000, 23000, 1000)))
    baxes.axs[1].set_yticks(list(range(0, 9000, 1000)))
    if savefig:
        fig.savefig("./relative_distributions.svg")
    plt.show()

def get_proportional_plot(counts, savefig = False):
    cnt_tr = counts[0]
    cnt_vl = counts[1]
    cnt_te = counts[2]

    plt.figure(figsize=(10, 7.5))
    plt.subplots_adjust(bottom=0.3)
    r = list(range(len(labs)))
    raw_data = {'greenBars': [cnt_tr[la] for la in labs], 'orangeBars': [cnt_vl[la] for la in labs],'blueBars': [cnt_te[la] for la in labs]}
    df = pd.DataFrame(raw_data)
    
    # From raw value to percentage
    totals = [i+j+k for i,j,k in zip(df['greenBars'], df['orangeBars'], df['blueBars'])]
    greenBars = [i / j * 100 for i,j in zip(df['greenBars'], totals)]
    orangeBars = [i / j * 100 for i,j in zip(df['orangeBars'], totals)]
    blueBars = [i / j * 100 for i,j in zip(df['blueBars'], totals)]
    
    # plot
    barWidth = 0.85
    names = labs
    plt.xticks(r, names, rotation = 30, ha = 'right', va = 'top')
    plt.yticks(list(range(0, 105, 5)), list(range(0, 105, 5)))
    # Create green Bars
    plt.bar(r, greenBars, color= (55/255, 126/255, 34/255), width=barWidth, label = "train")
    # Create orange Bars
    plt.bar(r, orangeBars, bottom=greenBars, color=(117/255, 251/255, 253/255), width=barWidth, label = "val")
    # Create blue Bars
    plt.bar(r, blueBars, bottom=[i+j for i,j in zip(greenBars, orangeBars)], color=(0, 0, 245/255), width=barWidth, label = "test")

    plt.hlines(70, -100, 100, ['black']*2, ['dashed']*2, linewidth = 1, label = textwrap.fill("Train set proportion goal", 15))
    plt.hlines(85, -100, 100, ['black']*2, ['dotted']*2, linewidth = 1, label = textwrap.fill("Dev set proportion goal", 15))
    plt.xlabel("Kinase Group")
    plt.ylabel("% Proportion of sites")
    plt.title("Result of Repeated Simulated Annealing")
    plt.legend(loc='lower right')
    plt.xlim(-0.5, 10.5)
    if savefig:
        plt.savefig("./proportional_distribution.svg")
    plt.show()

if __name__ == "__main__":
    import os, pathlib
    
    where_am_i = pathlib.Path(__file__).parent.resolve()
    os.chdir(where_am_i)

    data = get_group_distribution_diagrams(split_into_sets.get_assignment_info_dict("../../data/preprocessing/kin_to_fam_to_grp_817.csv", "../../data/preprocessing/raw_data_22473.csv", *[f"../../data/preprocessing/{x}_kins.json" for x in ["tr", "vl", "te"]]))

    get_overall_plot(data, savefig = True)
    get_proportional_plot(data, savefig = True)