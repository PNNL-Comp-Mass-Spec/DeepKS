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

def get_labeled_distance_matrix(input_fasta: str, dir=None) -> Tuple[str, str]:
    raise DeprecationWarning("Don't use this.")
    if dir is None:
        tf_output = tf.NamedTemporaryFile().name
        tf_matrix = tf.NamedTemporaryFile().name
    else:
        tf_output = os.path.join(dir, "clustalo_output.txt")
        tf_matrix = os.path.join(dir, "clustalo_matrix.txt")

    clustalo_cmd = (
        f"clustalo --infile {input_fasta} --verbose --distmat-out {tf_matrix} --full --full-iter --outfmt clustal"
        f" --resno --outfile {tf_output} --output-order input-order --seqtype protein --force"
    )

    print("Clustalo Aligning...")
    os.system(clustalo_cmd)
    return tf_matrix, tf_output


def extract_data(tf_matrix: str, tf_output: str, outfile="../mtx_@XX.csv", sort = True):
    raise DeprecationWarning("Don't use this.")
    with open(tf_matrix, "r") as mat_file:
        unformatted_mtx = mat_file.read().split("\n")[1:-1]
        print(f"[Distance matrix stored at {tf_matrix}")
        print(f"[Verbose output file stored at {tf_output}")

    csved_mtx = "\n".join([re.sub(" +", ",", x) for x in unformatted_mtx])
    mtx = pd.read_csv(io.StringIO(csved_mtx), index_col=0, header=None)
    mtx = mtx.applymap(lambda x: round(1 - x, 8))
    mtx.index = pd.Index([x.upper().replace("---", " ") for x in mtx.index.tolist()])
    mtx.columns = mtx.index.tolist()
    if sort:
        mtx = mtx.sort_index(axis=0).sort_index(axis=1)
    if outfile:
        if "@XX" in outfile:
            outfile = outfile.replace("@XX", str(mtx.shape[0]))
        mtx.to_csv(outfile)
    return mtx


def make_fasta(df_in: str, fasta_out: str) -> str:
    df = pd.read_csv(df_in, sep="," if "," in open(df_in, "r").read() else "\t").iloc[:]
    cols = set(df.columns)
    assert "kinase" in cols, 'Input df to `make_fasta` must have column "kinase."'
    assert "kinase_seq" in cols, 'Input df to `make_fasta` must have column "kinase_seq."'
    assert "gene_name" in cols, 'Input df to `make_fasta` must have column "gene_name."'
    rows = []
    for _ in df[df["gene_name"].isna()].index:
        raise RuntimeError("NA for gene_name!")

    for _, r in df.iterrows():
        rows.append(format_for_needle(">" + r["gene_name"] + "|" + r["kinase"]))
        rows.append("\n")
        rows.append(tw.fill(r["kinase_seq"]))
        rows.append("\n")
    with open(fasta_out, "w") as tfa:
        tfa.write("".join(rows))

    return fasta_out


# if __name__ == "__main__":
#     f = make_fasta("../../raw_data/kinase_seq_822.txt", "tmp.txt")
#     matrix, out = get_labeled_distance_matrix(f)
#     extract_data(matrix, out)
