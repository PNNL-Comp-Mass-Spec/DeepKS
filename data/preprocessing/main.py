import os, sys, re, pathlib, pprint, tempfile as tf, pandas as pd
from . import PreprocessingSteps as PS
from ...tools import system_tools, get_needle_pairwise as get_pairwise
from .PreprocessingSteps import get_labeled_distance_matrix as mtx_utils

# Change working directory to the directory of this file
where_am_i = pathlib.Path(__file__).parent.resolve()
os.chdir(where_am_i)
DEBUGGING = False if "DEBUGGING" not in os.environ else True if os.environ["DEBUGGING"] == "1" else False

if not DEBUGGING:
    print("Step 1: Download most recent version of the PhosphositePlus database.")
    PS.download_psp.get_phospho(outfile="../raw_data/PSP_script_download.xlsx")

    print("Step 2: Download sequences using the Uniprot REST API. Using R.")
    print("Step 2a: Ensuring `Rscript` is in PATH.")
    if os.system("Rscript --version 1>/dev/null 2>/dev/null") != 0:
        print("Rscript not found in PATH. Please install R and ensure it is in PATH.")
        exit(1)
    print("Step 2b: Running R scripts.")
    print("\n~~~~ R MESSAGES ~~~~\n")
    seq_filename, data_filename = tuple(
        system_tools.os_system_and_get_stdout("Rscript PreprocessingSteps/ML_data_pipeline.R")
    )
    print("\n~~~~ END R MESSAGES ~~~~\n")

    print("Step 3: Determining Kinase Family and Group Classifications (from http://www.kinhub.org/kinases.html).")
    kin_fam_grp_filename = PS.get_kin_fam_grp.get_kin_to_fam_to_grp(seq_filename).split("/")[:-1]

# For debugging
else:
    seq_filename, data_filename, kin_fam_grp_filename = (
        "../raw_data/kinase_seq_826.txt",
        "../raw_data/raw_data_22588.csv",
        "./kin_to_fam_to_grp_821.csv",
    )
print(
    "Step 4: Getting pairwise distance matrix for all sequences to assess similarity and to be used later in the kinase"
    " group classifier."
)

with tf.NamedTemporaryFile() as tmp:
    cur_mtx_files = sorted([x for x in os.listdir() if re.match(r"pairwise_mtx_[0-9]+.csv", x)], reverse=True)
    new_mtx_file = f"./pairwise_mtx_{re.sub('[^0-9]', '', seq_filename)}.csv"
    if len(cur_mtx_files) > 0:
        if DEBUGGING:
            cur_mtx_file = "./pairwise_mtx_822.csv"
        else:
            cur_mtx_file = cur_mtx_files[0]
        cur_mtx = pd.read_csv(cur_mtx_file, index_col=0)
        new_seq_df = pd.read_csv(seq_filename)
        existing = set(cur_mtx.index)
        not_existing = set(new_seq_df["gene_name"] + "|" + new_seq_df["kinase"]) - existing
        symbols_unk = new_seq_df[(new_seq_df["gene_name"] + "|" + new_seq_df["kinase"]).isin(not_existing)]
        with tf.NamedTemporaryFile() as small_fasta:
            symbols_unk.to_csv(small_fasta.name, index=False)
            small_fasta = mtx_utils.make_fasta(df_in=small_fasta.name, fasta_out=small_fasta.name)
            with tf.NamedTemporaryFile() as needle_out:
                square_mtx = get_pairwise.get_needle_pairwise_mtx(small_fasta, needle_out.name, num_procs=1)
                pass

        # Get restricted combinations
        all_fasta = mtx_utils.make_fasta(df_in=seq_filename, fasta_out=tmp.name)
        thin_mtx = get_pairwise.get_needle_pairwise_mtx(
            tmp.name, new_mtx_file, num_procs=4, restricted_combinations=[list(existing), list(not_existing)]
        )

        # Put it all together
        wide = pd.merge(cur_mtx, thin_mtx.T, how='left', left_index=True, right_index=True)
        missing_corner = pd.concat([wide, thin_mtx], axis=0)
        missing_corner.loc[square_mtx.index, square_mtx.columns] = square_mtx
        new_mtx = missing_corner
        new_mtx.to_csv(new_mtx_file)

    else:
        get_pairwise.get_needle_pairwise_mtx(tmp.name, new_mtx_file, num_procs=4, restricted_combinations=[])

pass
print("\nStep 5: Creating Table of Targets and computing Decoys.\n")
input_good = False
eval_or_train_on_all = ""
while not input_good:
    eval_or_train_on_all = input(
        "[E]valuation Mode (splits into train/val/test split) or [T]raining Mode (retrains model on all available"
        " data)? [Type E or T]: "
    )
    if eval_or_train_on_all.lower() == "e":
        input_good = True
        print("Step 4a: Must obtain train/val/test splits.")
        PS.split_into_sets_individual_deterministic_top_k.split_into_sets(
            kin_fam_grp_filename,
            data_filename,
            tgt=0.3,
            get_restart=True,
            num_restarts=600,
        )
    elif eval_or_train_on_all.lower() == "t":
        input_good = True
    else:
        print("Bad input. Trying again...")

train_on_all = eval_or_train_on_all.lower() == "t"
data_gen_conf = {"held_out_percentile": 95, "train_percentile": 65}
print(
    "\nInfo: Using the following thresholds for similarity: (In the future this will be configurable.)"
)  # TODO: Make this configurable
pprint.pprint(data_gen_conf)
