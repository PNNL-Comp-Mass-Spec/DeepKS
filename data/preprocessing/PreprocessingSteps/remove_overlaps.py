"""Contains functionality to assign each site to exactly one of the train, test, or val sets"""

import pandas as pd, random, collections, re, os, pathlib
from ....config.join_first import join_first

random.seed(0)


def remove_legitimate_duplicates(*input_files, rel_sizes):
    in_dict = collections.defaultdict(list)
    for inpf in input_files:
        seqs = pd.read_csv(inpf)["Site Sequence"].to_list()
        for i in range(len(seqs)):
            in_dict[seqs[i]].append((inpf, i))

    to_drop = collections.defaultdict(list)

    for seq in in_dict:
        if len(s := set([x[0] for x in in_dict[seq]])) > 1:
            ls = list(s)
            keep = random.sample(
                [i for i in range(len(s))],
                k=1,
                counts=[int(y / min([1 / rel_sizes[x] for x in ls])) for y in [1 / rel_sizes[x] for x in ls]],
            )[0]
            for instance in in_dict[seq]:
                if instance[0] != ls[keep]:
                    # print("delete:", instance)
                    to_drop[instance[0]].append(instance[1])
                else:
                    pass
                    # print("keep:", instance)
            # print("---------------------------")

    to_validate = []

    for fn in to_drop:
        df = pd.read_csv(fn)
        df.drop(to_drop[fn], axis=0, inplace=True)
        print("Outputted files:", re.sub(r"\.csv$", f"_{len(df)}.csv", fn))
        df.to_csv(tv := re.sub(r"\.csv$", f"_{len(df)}.csv", fn), index=False)
        to_validate.append(tv)
        os.unlink(fn)

    return to_validate


def validate_data(input_files):
    val_dict = collections.defaultdict(list)
    for inpf in input_files:
        df = pd.read_csv(inpf)
        seqs = df["Site Sequence"].to_list()
        original_kinases = df["Kinase Sequence"].to_list()
        for seq, original_kinase in zip(seqs, original_kinases):
            val_dict[(seq, original_kinase)].append(inpf)

    for entry in val_dict.values():
        assert len(set(entry)) == 1

def main(*files):
    rel_sizes = {
        y: int(x) for y, x in [(z, re.sub(r".*raw_data_(.*)_formatted_.*\.csv", "\\1", z)) for z in files]
    }
    in_dict = remove_legitimate_duplicates(*files, rel_sizes=rel_sizes)
    validate_data(in_dict)