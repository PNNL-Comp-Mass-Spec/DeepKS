import pathlib, os, traceback
from numbers import Number
from typing import Union

where_am_i = pathlib.Path(__file__).parent.resolve()
os.chdir(where_am_i)

import pandas as pd
import sys
from ....config import cfg
from ....tools.bipartite_derangement import get_groups_derangement4
import random
import re
import json

mode = cfg.get_mode()
random.seed(0)


def my_pop(df, index):
    """
    Based on https://stackoverflow.com/a/58548326
    """
    row = df.iloc[index]
    df.drop(index, inplace=True)
    return row


def get_input_dataframe(input_fn, kin_seq_file, distance_matrix_file, config):
    try:
        assert "held_out_percentile" in config or (
            "dataframe_generation_mode" in config and config["dataframe_generation_mode"] == "tr-all"
        )
        assert "train_percentile" in config
        assert "dataframe_generation_mode" in config
        if "held_out_percentile" in config:
            held_out_percentile = config["held_out_percentile"]
        else:
            held_out_percentile = None
        train_percentile = config["train_percentile"]
        dataframe_generation_mode = config["dataframe_generation_mode"]
    except AssertionError:
        _, _, tb = sys.exc_info()
        tb_info = traceback.extract_tb(tb)
        file, line, func, text = tb_info[-1]

        print(
            f"An assertion error occurred in file {file}, on line {line}, in function {func}, in assertion « {text} »"
        )
        exit(1)

    all_data = (
        pd.read_csv(input_fn)
        .sort_values(by=["num_sites", "lab", "seq"], ascending=[False, True, True])
        .reset_index(drop=True)
        .applymap(lambda x: x.upper() if isinstance(x, str) else x)
    )
    if dataframe_generation_mode == "tr-val-te":
        for ML_set, percentile in [
            ("../te_kins.json", held_out_percentile),
            ("../vl_kins.json", held_out_percentile),
            ("../tr_kins.json", train_percentile),
        ]:
            kins = [x.replace("(", "").replace(")", "").replace("*", "") for x in json.load(open(ML_set, "r"))]
            all_df = (
                pd.merge(
                    pd.DataFrame({"lab_updated": kins, "uniprot_id": [x.split("|")[1] for x in kins]}),
                    all_data,
                    how="left",
                    on="uniprot_id",
                )
                .drop("lab", axis=1, inplace=False)
                .rename(columns={"lab_updated": "lab"}, inplace=False)
            )
            get_input_dataframe_helper(all_df, input_fn, kin_seq_file, distance_matrix_file, percentile)

    elif dataframe_generation_mode == "tr-all":
        get_input_dataframe_helper(
            section_df=all_data,
            input_fn=input_fn,
            kin_seq_file=kin_seq_file,
            percentile=train_percentile,
            distance_matrix_file=distance_matrix_file,
            write_file=True,
        )

    else:
        raise ValueError("`dataframe_generation_mode` is not one of `tr-val-te` or `tr-all`.")


def get_input_dataframe_helper(
    section_df: pd.DataFrame,
    input_fn: str,
    kin_seq_file: str,
    percentile: Union[int, float],
    distance_matrix_file: str,
    write_file=True,
):
    all_data = section_df.sort_values(by=["num_sites", "lab", "uniprot_id"], ascending=[False, True, True]).reset_index(
        drop=True
    )
    start_dict = {}
    end_dict = {}
    for i, r in all_data.iterrows():
        if r["lab"] + "|" + r["uniprot_id"] not in start_dict:
            start_dict[r["lab"] + "|" + r["uniprot_id"]] = i
        end_dict[r["lab"] + "|" + r["uniprot_id"]] = i

    sizes = []
    order = []
    to_pop_inds = []

    print("Status: Assigning target and decoy with bipartite graph algorithm.")
    max_seen_end = 0
    max_seen_start = 0
    for kin in start_dict:
        if (end_dict[kin] <= max_seen_end and max_seen_end != 0) or (start_dict[kin] <= max_seen_start and max_seen_start != 0):
            raise ValueError()
        max_seen_end = end_dict[kin]
        max_seen_start = start_dict[kin]
        k = end_dict[kin] - start_dict[kin] + 1
        sizes.append(k)
        order.append(kin)
        # assert(sum(sizes) == len(to_pop_inds))

    # decoy = my_pop(all_data, to_pop_inds).reset_index(drop=True)
    target = all_data.copy()
    decoy = all_data.copy()

    target["original_kinase"] = target["lab"]
    decoy["original_kinase"] = decoy["lab"]

    derangement = json.load(open("/home/ubuntu/DeepKS/data/preprocessing/.gitig-derangement.json"))
    # derangement = [
    #     x if x is not None else len(decoy)
    #     for x in get_groups_derangement4(order, sizes, kin_seq_file, distance_matrix_file, percentile)
    # ]
    decoy.loc[len(decoy)] = ["NOLAB", "NOSITE", "NOSEQ", "NOID", "NONUM", "NOCLASS", "NOORIG", "NOORIGKIN"]  # type: ignore
    decoy_seqs = (
        decoy.iloc[derangement].reset_index().drop("index", axis=1).squeeze()
    )

    print("done processing derangement.\n")

    print("assembling final data...")
    decoy = pd.DataFrame(
        {
            "lab": decoy["lab"],
            "seq": decoy_seqs["seq"],
            "original_kinase": decoy_seqs["original_kinase"],
            "uniprot_id": decoy_seqs["uniprot_id"],
            "num_sites": decoy_seqs["num_sites"],
            "class": 0,
        }
    )
    decoy = decoy[(decoy["seq"] != "NOSEQ") & (~decoy["seq"].isna())]
    for i, r in decoy.iterrows():  # weak version
        assert r["lab"] != r["original_kinase"]
    all_data = pd.concat([target, decoy], ignore_index=True)
    # xl = pd.read_csv("../raw_data/relevant_GENE_KIN_ACC_ID.csv")
    # kin_name_dict = {str(kai) : str(k).upper() for k, kai in zip(xl['GENE'], xl['KIN_ACC_ID'])}

    kin_seqs_dict = pd.read_csv(kin_seq_file)
    kin_seqs_dict["kinase"] = (
        kin_seqs_dict["gene_name"].apply(lambda x: x.upper() if isinstance(x, str) else x)
        + "|"
        + kin_seqs_dict["kinase"]
    )
    kin_seqs_dict = kin_seqs_dict.set_index("kinase").to_dict()["kinase_seq"]

    all_data_w_seqs = pd.DataFrame(
        {
            "orig_lab_name": all_data["original_kinase"],
            "lab": all_data["lab"],
            "kin_seq": [kin_seqs_dict[k][:] for k in all_data["original_kinase"] + "|" + all_data["uniprot_id"]],
            "seq": all_data["seq"],
            "class": all_data["class"],
            "num_seqs": all_data["num_sites"],
        }
    )
    all_data_w_seqs.rename(columns={"lab": "lab_name", "kin_seq": "lab"}, inplace=True)
    num = int("".join([x if x.isdigit() else "" for x in input_fn]))

    extra = "" if mode == "no_alin" else "_alin"
    extra += f"_{percentile}"
    all_data_w_seqs = all_data_w_seqs.sort_values(
        by=["class", "lab_name", "orig_lab_name", "seq"], ascending=[False, True, True, True]
    )
    # orig_data = pd.read_csv(input_fn).sort_values(by=['num_sites', 'lab', 'seq'], ascending=[False, True, True]).reset_index(drop=True)
    # orig_data = orig_data[orig_data['seq'].isin(all_data_w_seqs['seq'])]
    if write_file:
        all_data_w_seqs.to_csv(
            fn := re.sub("([0-9]+)", f"{len(all_data_w_seqs)}", input_fn).replace(".csv", "")
            + f"_formatted{extra}.csv",
            index=False,
        )
        # orig_data.to_csv(fn.replace("_formatted", ""), index=False)
    print(f"Size: {len(all_data_w_seqs)}")
    # return all_data_w_seqs #orig_data


if __name__ == "__main__":
    if mode == "no_alin":
        kin_seq_file = "../../raw_data/kinase_seq_822.txt"
    else:
        raise RuntimeError("mode not supported")

    get_input_dataframe(
        "../../raw_data/raw_data_22473.csv",
        kin_seq_file,
        "../pairwise_mtx_826.csv",
        {
            "held_out_percentile": 95,
            "train_percentile": 65,
            "num_held_out_kins": 60,
            "dataframe_generation_mode": "pre",
        },
    )
