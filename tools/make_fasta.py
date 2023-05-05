"""Make a fasta file from a dataframe with biological sequences."""

import pandas as pd, textwrap as tw

format_for_needle = (
    lambda x: x.replace("(", "")
    .replace(")", "")
    .replace("|", "_")
    .replace("/", "--")
    .replace(" ", "---")
    .replace(":", "----")
)
"""Simple lambda to filter out characters that cause problems for ``needle``"""

eldeen_rof_tamrof = lambda x: x.replace("----", ":").replace("---", " ").replace("--", "/").replace("_", "|")
"""Simple lambda to reverse ``format_for_needle``"""


def make_fasta(df_in: str, fasta_out: str) -> str:
    """Make a fasta file from a dataframe with biological sequences.

    Parameters
    ----------
    df_in : str
        The input dataframe. Must have columns "kinase" -- the UniProt ID, "kinase_seq" -- the sequence itself, and "gene_name" -- the UniProt gene name.
    fasta_out : str
        The output fasta filename

    Returns
    -------
    str
        The output fasta filename (unmodified from the input parameter)

    Raises
    ------
    RuntimeError
        If there is an NA in the "gene_name" column of the input dataframe.
    """
    with open(df_in, "r") as f:
        if "," in f.read():
            sep = ","
        else:
            sep = "\t"
        df = pd.read_csv(df_in, sep=sep).iloc[:]
    cols = set(df.columns)
    assert "kinase" in cols, 'Input df to `make_fasta` must have column "kinase."'
    assert "kinase_seq" in cols, 'Input df to `make_fasta` must have column "kinase_seq."'
    assert "gene_name" in cols, 'Input df to `make_fasta` must have column "gene_name."'
    rows = []
    for _ in df[df["gene_name"].isna()].index:
        raise RuntimeError("NA for gene_name!")
    df["kinase"] = df["kinase"].replace(float("NaN"), "")

    for _, r in df.iterrows():
        rows.append(format_for_needle(">" + r["gene_name"] + ("|" if r["kinase"] != "" else "") + r["kinase"]))
        rows.append("\n")
        rows.append(tw.fill(r["kinase_seq"]))
        rows.append("\n")
    with open(fasta_out, "w") as tfa:
        tfa.write("".join(rows))

    return fasta_out
