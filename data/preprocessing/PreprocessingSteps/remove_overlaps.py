
import pandas as pd, random, collections, re, os
random.seed(0)

def remove_legitimate_duplicates(input_files, rel_sizes):
    in_dict = collections.defaultdict(list)
    for inpf in input_files:
        seqs = pd.read_csv(inpf)['seq'].to_list()
        for i in range(len(seqs)):
            in_dict[seqs[i]].append((inpf, i))
    
    rld = in_dict
    to_drop = collections.defaultdict(list)

    for seq in rld:
        if len(s := set([x[0] for x in rld[seq]])) > 1:
            ls = list(s)
            cntr = collections.Counter(ls)
            keep = random.sample([i for i in range(len(s))], k = 1, counts = [int(y/min([1/rel_sizes[x] for x in ls])) for y in [1/rel_sizes[x] for x in ls]])[0]
            for instance in rld[seq]:
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
        print("Outputted files:", re.sub("\.csv$", f"_{len(df)}.csv", fn))
        df.to_csv(tv := re.sub("\.csv$", f"_{len(df)}.csv", fn), index=False)
        to_validate.append(tv)
        os.unlink(fn)
    
    return to_validate

def validate_data(input_files):
    val_dict = collections.defaultdict(list)
    for inpf in input_files:
        df = pd.read_csv(inpf)
        seqs = df['seq'].to_list()
        original_kinases = df['lab'].to_list()
        for seq, original_kinase in zip(seqs, original_kinases):
            val_dict[(seq, original_kinase)].append(inpf)
    
    bad_entry = None
    try:
        for entry in val_dict.values():
            bad_entry = entry
            assert len(set(entry)) == 1
            bad_entry = None
    except AssertionError:
        print(bad_entry)

if __name__ == "__main__":
    import os, pathlib
    where_am_i = pathlib.Path(__file__).parent.resolve()
    os.chdir(where_am_i)

    inp_list = ["../raw_data_31834_formatted_65.csv", "../raw_data_6500_formatted_95.csv", "../raw_data_6406_formatted_95.csv"]
    rel_sizes = {y: int(x) for y, x in [(z, re.sub("\.\.\/raw_data_(.*)_formatted_.*\.csv", "\\1", z)) for z in inp_list]}
    rld = remove_legitimate_duplicates(inp_list, rel_sizes)
    validate_data(rld)