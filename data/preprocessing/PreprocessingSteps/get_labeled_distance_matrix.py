import os, pandas as pd, tempfile as tf, textwrap as tw, re, io
from pandas import _typing as pd_typing
from typing import Tuple

format_for_needle = (
    lambda x: x.replace("|", "_")
    .replace("(", "")
    .replace(")", "")
    .replace("/", "--")
    .replace(" ", "---")
    .replace(":", "----")
)
eldeen_rof_tamrof = (
    lambda x: x.replace("_", "|")
    .replace("--", "/")
    .replace("---", " ")
    .replace("----", ":")
)

def make_fasta(df_in: str, fasta_out: str) -> str:
    df = pd.read_csv(df_in, sep="," if "," in open(df_in, "r").read() else "\t").iloc[:]
    cols = set(df.columns)
    assert "kinase" in cols, 'Input df to `make_fasta` must have column "kinase."'
    assert "kinase_seq" in cols, 'Input df to `make_fasta` must have column "kinase_seq."'
    assert "gene_name" in cols, 'Input df to `make_fasta` must have column "gene_name."'
    rows = []
    for _ in df[df["gene_name"].isna()].index:
        raise RuntimeError("NA for gene_name!")
    df['kinase'] = df['kinase'].replace(float('NaN'), "")

    for _, r in df.iterrows():
        rows.append(format_for_needle(">" + r["gene_name"] + ("|" if r['kinase'] != "" else "") + r["kinase"]))
        rows.append("\n")
        rows.append(tw.fill(r["kinase_seq"]))
        rows.append("\n")
    with open(fasta_out, "w") as tfa:
        tfa.write("".join(rows))

    return fasta_out

