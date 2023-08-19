import pandas as pd
import re
import numpy as np
from typing import Union


def format_for_excel(mtx_file, raw_data_file):
    mtx = pd.read_csv(mtx_file, index_col=0).applymap(
        lambda x: np.round(x * 100, 2) if isinstance(x, Union[int, float]) else x
    )
    uniprot_to_organism = (
        pd.read_csv(raw_data_file)[["uniprot_id", "organism"]].set_index("uniprot_id").to_dict()["organism"]
    )

    gene_names = [re.sub(r"[\\(\\)\\*]", "", str(x)).split("|")[0] for x in mtx.index]
    uniprots = [re.sub(r"[\\(\\)\\*]", "", str(x)).split("|")[1] for x in mtx.index]
    organisms = [uniprot_to_organism[uniprots[i]] for i in range(len(uniprots))]

    mtx["uniprot_id"] = uniprots
    mtx.rename(
        columns={"Unnamed: 0": "uniprot_id"} | {current: new for current, new in zip(mtx.columns, uniprots)},
        inplace=True,
    )

    col_join_df = pd.DataFrame({"gene_name": gene_names, "organism": organisms, "uniprot_id": uniprots})
    mtx = pd.merge(col_join_df, mtx, how="left", on="uniprot_id")

    row_concat_df = pd.DataFrame(
        {
            "gene_name": 3 * [pd.NA] + gene_names,
            "organism": 3 * [pd.NA] + organisms,
            "uniprot_id": 3 * [pd.NA] + uniprots,
        }
    ).T
    row_concat_df.columns = mtx.columns.tolist()
    mtx = pd.concat([row_concat_df, mtx], axis=0)
    mtx.columns = np.array(["gene_name", "organism", "uniprot_id"] + [f"K{i}" for i in range(len(mtx.columns) - 3)])
    mtx.index = pd.Index(mtx.columns)
    mtx.to_csv("excel_art_formatted.csv")


if __name__ == "__main__":  # pragma: no cover
    format_for_excel("../../data/preprocessing/mtx_822.csv", "../../data/raw_data/raw_data_22473.csv")
