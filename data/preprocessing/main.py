import os, re, pathlib, pprint, tempfile as tf, pandas as pd
from . import PreprocessingSteps as PS
from ...tools import system_tools, get_needle_pairwise as get_pairwise
from ...tools import make_fasta as mtx_utils
from termcolor import colored

# Change working directory to the directory of this file
where_am_i = pathlib.Path(__file__).parent.resolve()
os.chdir(where_am_i)
DEBUGGING = False if "DEBUGGING" not in os.environ else True if os.environ["DEBUGGING"] == "1" else False
USE_XL_CACHE = False if "USE_XL_CACHE" not in os.environ else True if os.environ["USE_XL_CACHE"] == "1" else False

def main():
    # FOR DEBUGGING:
    seq_filename, seq_filename2, kin_fam_grp_filename, data_filename, cur_mtx_file, new_mtx_file, eval_or_train_on_all = (
            "../raw_data/kinase_seq_826.csv", 
            "../raw_data/kinase_seq_494.csv",
            "kin_to_fam_to_grp_826.csv",
            "../raw_data/raw_data_22588.csv",
            "./pairwise_mtx_826.csv",
            "pairwise_mtx_826.csv", 
            "T"
        )

    if not (DEBUGGING or USE_XL_CACHE):
        print("Step 1: Download most recent version of the PhosphositePlus database.")
        PS.download_psp.get_phospho(outfile="../raw_data/PSP_script_download.xlsx")
    else:
        print(colored("Warning: Using cached version of PSP database.", "yellow"))

    if not DEBUGGING:
        print("Step 2: Download sequences using the Uniprot REST API. Using R.")
        print("Step 2a: Ensuring `Rscript` is in PATH.")
        if os.system("Rscript --version 1>/dev/null 2>/dev/null") != 0:
            print("Rscript not found in PATH. Please install R and ensure it is in PATH.")
            exit(1)
        print("Step 2b: Running R scripts.")
        print("\n~~~~ R MESSAGES ~~~~\n")
        seq_filename, data_filename = tuple(
            system_tools.os_system_and_get_stdout("Rscript PreprocessingSteps/ML_data_pipeline.R", prepend = "[R] ")
        )
        print("\n~~~~ END R MESSAGES ~~~~\n")

    if not DEBUGGING:
        print("Step 3: Determining Kinase Family and Group Classifications (from http://www.kinhub.org/kinases.html).")
        kin_fam_grp_filename = PS.get_kin_fam_grp.get_kin_to_fam_to_grp(seq_filename).split("/")[:-1]

    if not DEBUGGING:
        print(
            "Step 4: Getting pairwise distance matrix for all sequences to assess similarity and to be used later in the kinase"
            " group classifier."
        )

        with tf.NamedTemporaryFile() as tmp:
            cur_mtx_files = sorted([x for x in os.listdir() if re.search(r"^pairwise_mtx_[0-9]+.csv$", x)], reverse=True)
            new_mtx_file = f"./pairwise_mtx_{re.sub('[^0-9]', '', seq_filename.split('/')[-1])}.csv"
            if len(cur_mtx_files) > 0:
                try:
                    cur_mtx_file = cur_mtx_files[0]
                    cur_mtx = pd.read_csv(cur_mtx_file, index_col=0)
                    dfs = [pd.read_csv(seq_f) for seq_f in [seq_filename, seq_filename2]]
                    dfs = [df.drop(columns='symbol') if 'symbol' in df.columns else df for df in dfs]
                    new_seq_df = pd.concat(dfs, ignore_index=True).drop_duplicates(keep='first').reset_index(drop=True)
                    existing = set(cur_mtx.index)
                    not_existing = set(new_seq_df["gene_name"] + "|" + new_seq_df["kinase"]) - existing
                    if len(not_existing) == 0:
                        raise InterruptedError()
                    symbols_unk = new_seq_df[(new_seq_df["gene_name"] + "|" + new_seq_df["kinase"]).isin(not_existing)]
                    new_seq_df.to_csv("temp.csv")
                    # Get restricted combinations
                    mtx_utils.make_fasta(df_in="temp.csv", fasta_out=tmp.name)
                    thin_mtx = get_pairwise.get_needle_pairwise_mtx(
                        tmp.name, new_mtx_file, num_procs=8, restricted_combinations=[list(existing), list(not_existing)]
                    )
                    with tf.NamedTemporaryFile() as small_fasta:
                        symbols_unk.to_csv(small_fasta.name, index=False)
                        small_fasta = mtx_utils.make_fasta(df_in=small_fasta.name, fasta_out=small_fasta.name)
                        with tf.NamedTemporaryFile() as needle_out:
                            square_mtx = get_pairwise.get_needle_pairwise_mtx(small_fasta, needle_out.name, num_procs=8)
                            pass
                    

                    # Put it all together
                    wide = pd.merge(cur_mtx, thin_mtx.T, how="left", left_index=True, right_index=True)
                    missing_corner = pd.concat([wide, thin_mtx], axis=0)
                    missing_corner.loc[square_mtx.index, square_mtx.columns] = square_mtx
                    new_mtx = missing_corner
                    new_mtx.to_csv(new_mtx_file)
                    new_mtx = cur_mtx
                except InterruptedError:
                    print(colored("Info: No new sequences to add to the pairwise distance matrix."))
            else:
                get_pairwise.get_needle_pairwise_mtx(tmp.name, new_mtx_file, num_procs=4, restricted_combinations=[])

    if not DEBUGGING:
        print("\nStep 5: Creating Table of Targets and computing Decoys.\n")
        input_good = False
        while not input_good:
            # eval_or_train_on_all = input(
            #     "[E]valuation Mode (splits into train/val/test split) or [T]raining Mode (retrains model on all available"
            #     " data)? [Type E or T]: "
            # ) # FOR DEBUGGING
            # TODO - for debugging. Uncomment out.
            if eval_or_train_on_all.lower() == "e":
                input_good = True
                print("Step 5a: Must obtain train/val/test splits.")
                PS.split_into_sets_individual_deterministic_top_k.split_into_sets(
                    kin_fam_grp_filename,
                    data_filename,
                    tgt=0.3,
                    get_restart=True,
                    num_restarts=600
                )
                data_gen_conf = {"held_out_percentile": 95, "train_percentile": 65, "dataframe_generation_mode": "tr-all"}
                print(
                    "\nInfo: Using the following thresholds for similarity: (In the future this will be configurable.)"
                )  # TODO: Make this configurable
                pprint.pprint(data_gen_conf)
            elif eval_or_train_on_all.lower() == "t":
                input_good = True
                data_gen_conf = {"train_percentile": 65, "dataframe_generation_mode": "tr-all"}
                print(colored("Info: Generating dataframe with the following configuration dictionary:", "blue"))
                pprint.pprint(data_gen_conf)
                PS.format_raw_data_DD.get_input_dataframe(
                    input_fn=data_filename,
                    kin_seq_file=seq_filename,
                    distance_matrix_file=new_mtx_file,
                    config=data_gen_conf
                )
            else:
                print("Bad input. Trying again...")


if __name__ == "__main__":
    main()