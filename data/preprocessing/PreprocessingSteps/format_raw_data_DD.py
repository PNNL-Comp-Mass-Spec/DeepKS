# %%
import pathlib, os

where_am_i = pathlib.Path(__file__).parent.resolve()
os.chdir(where_am_i)

import pandas as pd
import sys
sys.path.append("../../../tools")
sys.path.append("../../../config")
import config
sys.path.append("../../")
from bipartite_derangement import get_groups_derangement4
import random
import re
import json
import pickle

mode = config.get_mode()
random.seed(0)

# %%
def my_pop(df, index):
    """
    Based on https://stackoverflow.com/a/58548326
    """
    row = df.iloc[index]
    df.drop(index, inplace=True)
    return row

def remove_untrue_duplicates(data):
    print("--- Removing Spurious Decoys --- ")
    duplicates = pd.read_csv("../raw_data/duplicates.csv")
    data_copy = data[data['class'] == 0].copy()

    suspects_datafile = []
    # suspects_duplicates = []

    for i, r in data_copy.iterrows():
        duplicates_kinases_idx = duplicates[duplicates['flank_seq'] == r['seq']].index.values.tolist()
        duplicates_kinases = duplicates.loc[duplicates_kinases_idx]['kinase'].tolist()
        if r['lab_name'] in duplicates_kinases and r['orig_lab_name'] in duplicates_kinases:
            suspects_datafile.append(i)
            # suspects_duplicates.append(duplicates_kinases_idx)
    
    dd = data.drop(labels = suspects_datafile, axis= 0)
    print(f"After removing spurious decoys, there are {len(dd)} inputs.")
    return dd

def generate_real(input_fn, kin_seq_file, config):
    try:
        assert("held_out_percentile" in config)
        assert("train_percentile" in config)
        assert("num_held_out_kins" in config)
        assert("pre_split_or_post_split" in config)
        held_out_percentile = config['held_out_percentile']
        train_percentile = config['train_percentile']
        num_held_out_kins = config['num_held_out_kins']
        pre_split_or_post_split = config['pre_split_or_post_split']
    except AssertionError as ae:
        print("Assertion Error:", ae)
        exit(1)

    # assert "no_overlaps" in input_fn, "You probably mean to call `generate_real` on a file with filename containing \"no_overlaps\"."
    all_data = pd.read_csv(input_fn).sort_values(by=['num_sites', 'lab', 'seq'], ascending=[False, True, True]).reset_index(drop=True).applymap(lambda x: x.upper() if isinstance(x, str) else x)
    # specific_kins = json.load(open("../../models/json/kinase_folds.json"))
    # held_out_kins = specific_kins['4']
    # training_kins = specific_kins['train_kins']
    if pre_split_or_post_split == "pre":
        for ML_set, percentile in [("../te_kins.json", held_out_percentile), ('../vl_kins.json', held_out_percentile), ('../tr_kins.json', train_percentile)]:
            kins = [x.replace("(", "").replace(")", "").replace("*", "") for x in json.load(open(ML_set, "r"))]
            all_df = pd.merge(pd.DataFrame({"lab_updated": kins, "uniprot_id": [x.split("|")[1] for x in kins]}), all_data, how = "left", on = "uniprot_id")\
                     .drop('lab', axis = 1, inplace = False).rename(columns={'lab_updated': 'lab'}, inplace=False)
            generate_real_helper(all_df, input_fn, kin_seq_file, percentile)

    # elif pre_split_or_post_split == "post":
    #     df, df_orig = generate_real_helper(all_data, input_fn, kin_seq_file, train_percentile, write_file=True)
    #     training_df = df[df['lab_name'].isin(training_kins)]
    #     held_out_df = df[df['lab_name'].isin(held_out_kins)]
    #     training_df_orig = df_orig[df_orig['lab'].isin(training_kins)]
    #     held_out_df_orig = df_orig[df_orig['lab'].isin(held_out_kins)]
    #     extra = "_50-separate"
    #     training_df.to_csv(fn:=re.sub("([0-9]+)", f"{len(training_df)}", input_fn).replace(".csv", "") + f"_formatted{extra}.csv", index=False)
    #     training_df_orig.to_csv(fn.replace("_formatted", ""), index=False)
    #     held_out_df.to_csv(fn:=re.sub("([0-9]+)", f"{len(held_out_df)}", input_fn).replace(".csv", "") + f"_formatted{extra}.csv", index=False)
    #     held_out_df_orig.to_csv(fn.replace("_formatted", ""), index=False)
    #     print(len(training_df), len(held_out_df))
    else:
        raise RuntimeError("`pre_split_or_post_split` is not one of `pre` or `post`.")
# %%

def generate_real_helper(section_df: pd.DataFrame, input_fn: str, kin_seq_file: str, percentile: float, write_file=True):
    all_data = section_df.reset_index(drop=True)
    start_dict = {}
    end_dict = {}
    for i, r in all_data.iterrows():
        if r['lab'] not in start_dict:
            start_dict[r['lab']] = i
        end_dict[r['lab']] = i
    
    sizes = []
    order = []
    to_pop_inds = []

    print("assigning target and decoy...")
    for kin in start_dict:
        k = end_dict[kin] - start_dict[kin] + 1
        sizes.append(k)
        order.append(kin)
        # assert(sum(sizes) == len(to_pop_inds))

    # decoy = my_pop(all_data, to_pop_inds).reset_index(drop=True)
    target = all_data.copy()
    decoy = all_data.copy()

    target['original_kinase'] = target['lab']
    decoy['original_kinase'] = decoy['lab']
    print("done assigning target and decoy.")

    print("processing derangement...")
    derangement = [x if x is not None else len(decoy) for x in get_groups_derangement4(order, sizes, kin_seq_file, re.sub("[^0-9]*([0-9]+)[^0-9]*", "../mtx_\\1.csv", kin_seq_file),percentile)]
    decoy.loc[len(decoy)] = ['NOLAB', 'NOSITE', 'NOSEQ', 'NOID', 'NONUM', 'NOCLASS', 'NOORIG']
    decoy_seqs = decoy.iloc[derangement][['seq', 'original_kinase', 'num_sites']].reset_index().drop("index", axis=1).squeeze()

    print("done processing derangement.\n")

    print("assembling final data...")
    decoy = pd.DataFrame({"lab": decoy['lab'], "seq": decoy_seqs['seq'], "original_kinase": decoy_seqs['original_kinase'], "num_sites": decoy_seqs['num_sites'], "class": 0})
    decoy = decoy[(decoy['seq'] != "NOSEQ") & (~decoy['seq'].isna())]
    for i, r in decoy.iterrows(): # weak version
        assert(r['lab'] != r['original_kinase'])
    all_data = pd.concat([target, decoy], ignore_index=True)
    # xl = pd.read_csv("../raw_data/relevant_GENE_KIN_ACC_ID.csv")
    # kin_name_dict = {str(kai) : str(k).upper() for k, kai in zip(xl['GENE'], xl['KIN_ACC_ID'])}

    kin_seqs_dict = pd.read_csv(kin_seq_file, sep = "\t")
    kin_seqs_dict['kinase'] = kin_seqs_dict['gene_name'].apply(lambda x: x.upper() if isinstance(x, str) else x) + "|" + kin_seqs_dict['kinase']
    kin_seqs_dict = kin_seqs_dict.set_index('kinase').to_dict()['kinase_seq']
    
    all_data_w_seqs = pd.DataFrame({"orig_lab_name": all_data['original_kinase'], "lab": all_data['lab'], "kin_seq": [kin_seqs_dict[k][:] for k in all_data['lab']], "seq": all_data['seq'], "class": all_data['class'], "num_seqs": all_data['num_sites']})
    all_data_w_seqs.rename(columns={"lab": "lab_name", "kin_seq": "lab"}, inplace=True)
    num = int("".join([x if x.isdigit() else "" for x in input_fn]))

    extra = "" if mode == "no_alin" else "_alin"
    extra += f"_{percentile}"
    all_data_w_seqs = all_data_w_seqs.sort_values(by=['class', 'lab_name', 'orig_lab_name', 'seq'], ascending = [False, True, True, True])
    # orig_data = pd.read_csv(input_fn).sort_values(by=['num_sites', 'lab', 'seq'], ascending=[False, True, True]).reset_index(drop=True)
    # orig_data = orig_data[orig_data['seq'].isin(all_data_w_seqs['seq'])]
    if write_file:
        all_data_w_seqs.to_csv(fn:=re.sub("([0-9]+)", f"{len(all_data_w_seqs)}", input_fn).replace(".csv", "") + f"_formatted{extra}.csv", index=False)
        # orig_data.to_csv(fn.replace("_formatted", ""), index=False)
    print(f"Size: {len(all_data_w_seqs)}")
    # return all_data_w_seqs #orig_data

if __name__ == "__main__":
    if mode == "no_alin":
        kin_seq_file = "../../raw_data/kinase_seq_822.txt"
    else:
        kin_seq_file = "../../raw_data/kinase_seq_alin_792.txt"

    generate_real("../raw_data_22473.csv", kin_seq_file, {
                                                                        "held_out_percentile": 95,
                                                                        "train_percentile": 65,
                                                                        "num_held_out_kins": 60,
                                                                        "pre_split_or_post_split": 'pre' 
                                                                    }
                 )