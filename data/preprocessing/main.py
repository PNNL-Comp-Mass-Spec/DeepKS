"""The main script for preprocessing the data."""

import os, re, pathlib, pprint, tempfile as tf, pandas as pd, sys
from . import PreprocessingSteps
from ...tools import system_tools, get_needle_pairwise as get_pairwise
from ...tools import make_fasta as mtx_utils
from termcolor import colored

from ...config.logging import get_logger

logger = get_logger()
"""The logger for this script."""
if __name__ == "__main__":
    logger.status("Loading Modules")

# Change working directory to the directory of this file
os.chdir(pathlib.Path(__file__).parent.resolve())  # TODO: Do a better version of this


DEBUGGING = False if "DEBUGGING" not in os.environ else True if os.environ["DEBUGGING"] == "1" else False
"""Whether or not to run in debugging mode; need to provide steps in `perform_steps` if True."""

debug_var_names = [
    "seq_filename_A",
    "seq_filename_B",
    "kin_fam_grp_filename",
    "data_filename",
    "new_mtx_file",
    "eval_or_train_on_all",
]
"""If one only wants to run a single step, one can use the variables here."""

debug_var_vals = [
    "../raw_data/kinase_seq_826.csv",
    "../raw_data/kinase_seq_494.csv",
    "kin_to_fam_to_grp_826.csv",
    "../raw_data/raw_data_22588.csv",
    "pairwise_mtx_826.csv",
    "T",
]
"""The values for the variables in `debug_var_names`."""

debugging_variables = dict(zip(debug_var_names, debug_var_vals))
"""The debugging variables as a dictionary."""


USE_XL_CACHE = False if "USE_XL_CACHE" not in os.environ else True if os.environ["USE_XL_CACHE"] == "1" else False
"""Whether or not to use the cached version of the PSP database."""

perform_steps = eval(sys.argv[sys.argv.index("--perform-steps") + 1]) if "--perform-steps" in sys.argv else {}
"""The steps to perform. If empty, all steps will be performed."""


def step_1_download_psp(outfile="../raw_data/PSP_script_download.xlsx"):
    """Wrapper for `PreprocessingSteps.download_psp.get_phospho`. Same Parameters."""
    if (not (DEBUGGING or USE_XL_CACHE)) or 1 in perform_steps:
        logger.status("Step 1: Download most recent version of the PhosphositePlus database.")
        PreprocessingSteps.download_psp.get_phospho(outfile=outfile)
    else:
        logger.warning("Using cached version of PSP database.")


def step_2_download_uniprot():
    """Wrapper for `PreprocessingSteps/ML_data_pipeline.R`."""
    if not DEBUGGING or 2 in perform_steps:
        logger.status("Step 2: Download sequences using the Uniprot REST API. Using R.")
        logger.status("Step 2a: Ensuring `Rscript` is in PATH.")
        if os.system("Rscript --version 1>/dev/null 2>/dev/null") != 0:
            logger.status("Rscript not found in PATH. Please install R and ensure it is in PATH.")
            exit(1)
        logger.status("Step 2b: Running R scripts.")
        print("\n~~~~ R MESSAGES ~~~~\n")
        seq_filename_A, data_filename = tuple(
            system_tools.os_system_and_get_stdout("Rscript PreprocessingSteps/ML_data_pipeline.R", prepend="[R] ")
        )
        print("\n~~~~ END R MESSAGES ~~~~\n")
        return seq_filename_A, data_filename
    return None, None


def step_3_get_kin_to_fam_to_grp(seq_filename_A):
    """Wrapper for `PreprocessingSteps.get_kin_fam_grp.get_kin_to_fam_to_grp`. Same parameters."""
    if not DEBUGGING or 3 in perform_steps:
        logger.status(
            "Step 3: Determining Kinase Family and Group Classifications (from http://www.kinhub.org/kinases.html)."
        )
        kin_fam_grp_filename = PreprocessingSteps.get_kin_fam_grp.get_kin_to_fam_to_grp(seq_filename_A).split("/")[:-1]
        return kin_fam_grp_filename
    return None


def step_4_get_pairwise_mtx(seq_filename_A, seq_filename_B):
    """Wrapper for `get_pairwise.get_needle_pairwise_mtx`."""
    if not DEBUGGING or 4 in perform_steps:
        logger.status(
            "Step 4: Getting pairwise distance matrix for all sequences to assess similarity and to be used later in"
            " the kinase group classifier."
        )

        with tf.NamedTemporaryFile() as tmp:
            cur_mtx_files = sorted(
                [x for x in os.listdir() if re.search(r"^pairwise_mtx_[0-9]+.csv$", x)], reverse=True
            )
            new_mtx_file = f"./pairwise_mtx_{re.sub('[^0-9]', '', seq_filename_A.split('/')[-1])}.csv"
            if len(cur_mtx_files) > 0:
                try:
                    cur_mtx_file = cur_mtx_files[0]
                    cur_mtx = pd.read_csv(cur_mtx_file, index_col=0)
                    dfs = [pd.read_csv(seq_f) for seq_f in [seq_filename_A, seq_filename_B]]
                    dfs = [df.drop(columns="symbol") if "symbol" in df.columns else df for df in dfs]
                    new_seq_df = pd.concat(dfs, ignore_index=True).drop_duplicates(keep="first").reset_index(drop=True)
                    existing = set(cur_mtx.index)
                    not_existing = set(new_seq_df["gene_name"] + "|" + new_seq_df["kinase"]) - existing
                    if len(not_existing) == 0:
                        raise InterruptedError()
                    symbols_unk = new_seq_df[(new_seq_df["gene_name"] + "|" + new_seq_df["kinase"]).isin(not_existing)]
                    new_seq_df.to_csv("temp.csv")
                    # Get restricted combinations
                    mtx_utils.make_fasta(df_in="temp.csv", fasta_out=tmp.name)
                    thin_mtx = get_pairwise.get_needle_pairwise_mtx(
                        tmp.name,
                        new_mtx_file,
                        num_procs=8,
                        restricted_combinations=[list(existing), list(not_existing)],
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


def step_5_get_train_val_test_split(
    kin_fam_grp_filename,
    data_filename,
    seq_filename_A,
    new_mtx_file,
    eval_or_train_on_all,
    data_gen_conf={
        "held_out_percentile": 95,
        "train_percentile": 65,
        "dataframe_generation_mode": "tr-all",
    },
):
    """Wrapper for `PreprocessingSteps.split_into_sets_individual_deterministic_top_k.split_into_sets` and `PreprocessingSteps.format_raw_data_DD.get_input_dataframe`."""
    if not DEBUGGING or 5 in perform_steps:
        logger.status("Step 5: Creating Table of Targets and computing Decoys.")
        input_good = False
        while not input_good:
            if eval_or_train_on_all.lower() == "e":
                input_good = True
                logger.status("Step 5a: Must obtain train/val/test splits.")
                PreprocessingSteps.split_into_sets_individual_deterministic_top_k.split_into_sets(
                    kin_fam_grp_filename, data_filename, tgt=0.3, get_restart=True, num_restarts=600
                )
                pprint.pprint(data_gen_conf)
            elif eval_or_train_on_all.lower() == "t":
                input_good = True
                data_gen_conf = {"train_percentile": 65, "dataframe_generation_mode": "tr-all"}
                logger.info(
                    "Generating dataframe with the following configuration"
                    f" dictionary:\n{pprint.pformat(data_gen_conf)}"
                )
                PreprocessingSteps.format_raw_data_DD.get_input_dataframe(
                    input_fn=data_filename,
                    kin_seq_file=seq_filename_A,
                    distance_matrix_file=new_mtx_file,
                    config=data_gen_conf,
                )
            else:
                logger.warning("Bad input. Trying again...")


def main(
    seq_filename_B,
    data_filename,
    new_mtx_file,
    eval_or_train_on_all,
    seq_filename_A=None,
    kin_fam_grp_filename=None,
):
    """The main function for preprocessing the data. Does the following:
    1. Downloads PSP
    2. Downloads Uniprot data
    3. Maps kinases to families and groups
    4. Gets pairwise distance matrix
    5. Gets train/val/test split
    """

    step_1_download_psp()
    seq_filename_A, data_filename = step_2_download_uniprot()
    kin_fam_grp_filename = step_3_get_kin_to_fam_to_grp(seq_filename_A)
    step_4_get_pairwise_mtx(seq_filename_A, seq_filename_B)
    step_5_get_train_val_test_split(
        kin_fam_grp_filename, data_filename, seq_filename_A, new_mtx_file, eval_or_train_on_all
    )


if __name__ == "__main__":
    main(**debugging_variables)
