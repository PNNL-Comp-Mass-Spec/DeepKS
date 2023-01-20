import os, pandas as pd, tempfile as tf, textwrap as tw, re, io
from typing import Tuple


def get_labeled_distance_matrix(input_fasta: str, dir=None) -> Tuple[str, str]:
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
        mtx = mtx.sort_index(0).sort_index(1)
    if outfile:
        if "@XX" in outfile:
            outfile = outfile.replace("@XX", str(mtx.shape[0]))
        mtx.to_csv(outfile)
    return mtx


def make_fasta(df_in: str, fasta_out: str) -> str:
    df = pd.read_csv(df_in, sep="\t").iloc[:]
    assert "kinase" in df.columns, 'Input df to `make_fasta` must have column "kinase."'
    assert "kinase_seq" in df.columns, 'Input df to `make_fasta` must have column "kinase_seq."'
    assert "gene_name" in df.columns, 'Input df to `make_fasta` must have column "gene_name."'
    rows = []
    for _ in df[df["gene_name"].isna()].index:
        raise RuntimeError("NA for gene_name!")

    for _, r in df.iterrows():
        rows.append(">" + r["gene_name"].replace(" ", "---") + "|" + r["kinase"])
        rows.append("\n")
        rows.append(tw.fill(r["kinase_seq"]))
        rows.append("\n")
    with open(fasta_out, "w") as tfa:
        tfa.write("".join(rows))

    return fasta_out


if __name__ == "__main__":
    f = make_fasta("../../raw_data/kinase_seq_822.txt")
    matrix, out = get_labeled_distance_matrix(f)
    extract_data(matrix, out)
