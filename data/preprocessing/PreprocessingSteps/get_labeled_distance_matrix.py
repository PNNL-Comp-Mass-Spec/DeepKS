import os, pandas as pd, tempfile as tf, textwrap as tw, re, io

def get_labeled_distance_matrix(input_fasta):
    tf_output = tf.NamedTemporaryFile().name
    tf_matrix = tf.NamedTemporaryFile().name

    clustalo_cmd = f"clustalo --infile {input_fasta} --verbose --distmat-out {tf_matrix} --full --full-iter --outfmt clustal --resno --outfile {tf_output} --output-order input-order --seqtype protein"

    print("Clustalo Aligning...")
    os.system(clustalo_cmd)

    with open(tf_matrix, "r") as mat_file:
        unformatted_mtx = mat_file.read().split("\n")[1:-1]

    csved_mtx = "\n".join([re.sub(" +", ",", x) for x in unformatted_mtx])
    mtx = pd.read_csv(io.StringIO(csved_mtx), index_col=0, header=None)
    mtx = mtx.applymap(lambda x: round(1-x, 8))
    mtx.index = pd.Index([x.upper() for x in mtx.index.tolist()])
    mtx.columns = mtx.index.tolist()
    mtx = mtx.sort_index(0).sort_index(1)
    mtx.to_csv(f"../mtx_{len(mtx)}.csv")

    print(f"[Distance matrix stored at {tf_matrix}")
    print(f"[Verbose output file stored at {tf_output}")

def make_fasta(df_in):
    df = pd.read_csv(df_in, sep = "\t").iloc[:]
    assert 'kinase' in df.columns, "Input df to `make_fasta` must have column \"kinase.\""
    assert 'kinase_seq' in df.columns, "Input df to `make_fasta` must have column \"kinase_seq.\""
    assert 'gene_name' in df.columns, "Input df to `make_fasta` must have column \"gene_name.\""
    rows = []
    for i in df[df['gene_name'].isna()].index:
        df.at[i, 'gene_name'] = f"!{df.at[i, 'kinase']}!"

    for _, r in df.iterrows():
        rows.append(">"+r['gene_name']+"|"+r['kinase'])
        rows.append("\n")
        rows.append(tw.fill(r['kinase_seq']))
        rows.append("\n")
    temp_fasta = tf.NamedTemporaryFile().name
    with open(temp_fasta, "w") as tfa:
        tfa.write("".join(rows))
        
    return temp_fasta

if __name__ == "__main__":
    get_labeled_distance_matrix(make_fasta("../../raw_data/kinase_seq_822.txt"))