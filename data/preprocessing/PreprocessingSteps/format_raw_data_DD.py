import pathlib, os, traceback
from typing import Union
from termcolor import colored

where_am_i = pathlib.Path(__file__).parent.resolve()
os.chdir(where_am_i)

import pandas as pd
import sys
from ....config import cfg
from ....tools.bipartite_derangement import get_derangement
import random
import re
import json

mode = cfg.get_mode()
random.seed(0)

from ....config.logging import get_logger

logger = get_logger()
if __name__ == "__main__":
    logger.status("Loading Modules")

join_first = lambda levels, x: os.path.join(pathlib.Path(__file__).parent.resolve(), *[".."] * levels, x)

def my_pop(df, index):
    """
    Based on https://stackoverflow.com/a/58548326
    """
    row = df.iloc[index]
    df.drop(index, inplace=True)
    return row


def get_input_dataframe(input_fn, kin_seq_file, distance_matrix_file, config):
    assert "held_out_percentile" in config or config.get("dataframe_generation_mode") == "tr-all"
    assert "train_percentile" in config
    assert "dataframe_generation_mode" in config
    if "held_out_percentile" in config:
        held_out_percentile = config["held_out_percentile"]
    else:
        held_out_percentile = None
    train_percentile = config["train_percentile"]
    dataframe_generation_mode = config["dataframe_generation_mode"]

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
            get_input_dataframe_core(all_df, input_fn, kin_seq_file, distance_matrix_file, percentile)

    elif dataframe_generation_mode == "tr-all":
        get_input_dataframe_core(
            section_df=all_data,
            input_fn=input_fn,
            kin_seq_file=kin_seq_file,
            percentile=train_percentile,
            distance_matrix_file=distance_matrix_file,
            write_file=True,
        )

    else:
        raise ValueError("`dataframe_generation_mode` is not one of `tr-val-te` or `tr-all`.")


def get_input_dataframe_core(
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

    logger.status("Assigning target and decoy with bipartite graph algorithm.")
    max_seen_end = 0
    max_seen_start = 0
    for kin in start_dict:
        if (end_dict[kin] <= max_seen_end and max_seen_end != 0) or (
            start_dict[kin] <= max_seen_start and max_seen_start != 0
        ):
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
    decoy["Class"] = 0

    for df in [target, decoy]:
        df["original_kinase"] = target["lab"]
        df["original_uniprot_id"] = target["uniprot_id"]

    if os.path.exists(f"{sum(sizes)}.derangement"):
        derangement = json.load(open(f"{sum(sizes)}.derangement"))
        logger.info("Using derangement found in cache.")
    else:
        logger.warning("Computing derangement instead of using cache.")
        derangement = [
            x if x is not None else len(decoy)
            for x in get_derangement(
                order, sizes, kin_seq_file, distance_matrix_file, percentile, cache_derangement=True
            )
        ]

    logger.status("Done processing derangement.")
    logger.status("assembling final data.")

    for col in ["original_kinase", "original_uniprot_id", "seq"]:
        decoy[col] = decoy[col].loc[derangement].reset_index(drop=True)

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
            "Gene Name of Kin Corring to Provided Sub Seq": all_data["original_kinase"] + "|" + all_data["original_uniprot_id"],
            "Gene Name of Provided Kin Seq": all_data["lab"] + "|" + all_data["uniprot_id"],
            "Num Seqs in Orig Kin": all_data["num_sites"],
            "Class": all_data["Class"],
            "Site Sequence": all_data["seq"],
            "Kinase Sequence": [kin_seqs_dict[k][:] for k in all_data["lab"] + "|" + all_data["uniprot_id"]],
        }
    )

    extra = "" if mode == "no_alin" else "_alin"
    extra += f"_{percentile}"
    all_data_w_seqs = all_data_w_seqs.sort_values(
        by=["Class", "Kinase Sequence", "Gene Name of Kin Corring to Provided Sub Seq", "Site Sequence"],
        ascending=[False, True, True, True],
    )

    if write_file:
        all_data_w_seqs.to_csv(
            "/".join(
                input_fn.split("/")[:-1]
                + [re.sub("([0-9]+)", f"{len(all_data_w_seqs)}", input_fn.split("/")[-1]).replace(".csv", "")]
            )
            + f"_formatted{extra}.csv",
            index=False,
        )
        # orig_data.to_csv(fn.replace("_formatted", ""), index=False)
    logger.info("Outputting formatted data file with size: {len(all_data_w_seqs)}")


if __name__ == "__main__":
    if mode == "no_alin":
        kin_seq_file = (
            join_first(3, "data/raw_data/kinase_seq_826.csv")
        )
    else:
        raise RuntimeError("mode not supported")

    get_input_dataframe(
        "../../raw_data/raw_data_22588.csv",
        kin_seq_file,
        "../pairwise_mtx_826.csv",
        {"train_percentile": 65, "dataframe_generation_mode": "tr-all"},
    )
