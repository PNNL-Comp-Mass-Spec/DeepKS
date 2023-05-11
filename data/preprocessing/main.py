"""The main script for preprocessing the data."""

import os, re, pathlib, pprint, tempfile as tf, pandas as pd, sys
from typing import Literal
from . import PreprocessingSteps
from ...tools import system_tools, get_needle_pairwise as get_pairwise
from ...tools import make_fasta as mtx_utils
from ...config.join_first import join_first
from termcolor import colored

from ...config.logging import get_logger

logger = get_logger()
"""The logger for this script."""
if __name__ == "__main__":  # pragma: no cover
    logger.status("Loading Modules")

# Change working directory to the directory of this file
os.chdir(pathlib.Path(__file__).parent.resolve())  # TODO: Do a better version of this


if "USE_XL_CACHE" not in os.environ:
    USE_XL_CACHE = False
else:
    if os.environ["USE_XL_CACHE"] == "1":
        USE_XL_CACHE = True
    else:
        USE_XL_CACHE = False
"""Whether or not to use the cached version of the PSP database."""

if "--perform-steps" in sys.argv:
    perform_steps = eval(sys.argv[sys.argv.index("--perform-steps") + 1])
else:
    perform_steps = {}
"""The steps to perform. If empty, all steps will be performed."""

print(sys.argv)

def step_1_download_psp(outfile="../raw_data/PSP_script_download.xlsx"):
    """Wrapper for `PreprocessingSteps.download_psp.get_phospho`. Same Parameters."""
    logger.status("Step 1: Download most recent version of the PhosphositePlus database.")
    PreprocessingSteps.download_psp.get_phospho(outfile=outfile)


def step_2_download_uniprot():
    """Wrapper for ``PreprocessingSteps/ML_data_pipeline.R``."""
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


def step_3_get_kin_to_fam_to_grp(seq_filename_A):
    """Wrapper for `PreprocessingSteps.get_kin_fam_grp.get_kin_to_fam_to_grp`. Same parameters."""
    logger.status(
        "Step 3: Determining Kinase Family and Group Classifications (from http://www.kinhub.org/kinases.html)."
    )
    res = PreprocessingSteps.get_kin_fam_grp.get_kin_to_fam_to_grp(seq_filename_A)
    kin_fam_grp_filename = res
    return kin_fam_grp_filename


def step_4_get_pairwise_mtx(seq_filename_A, *addl_seq_filenames):
    """Wrapper for `get_pairwise.get_needle_pairwise_mtx`."""
    logger.status(
        "Step 4: Getting pairwise distance matrix for all sequences to assess similarity and to be used later in"
        " the kinase group classifier."
    )

    dfs_orig = [pd.read_csv(seq_f) for seq_f in list(addl_seq_filenames) + [seq_filename_A]]

    dfs = []
    for df in dfs_orig:
        if "symbol" in df.columns:
            df = df.drop(columns="symbol")
        else:
            df = df
        dfs.append(df)
    new_seq_df = pd.concat(dfs, ignore_index=True).drop_duplicates(keep="first").reset_index(drop=True)

    with tf.NamedTemporaryFile() as tmp, tf.NamedTemporaryFile() as tmp2:
        cur_mtx_files = sorted(
            [x for x in os.listdir() if re.search(r"^pairwise_mtx_[0-9]+.csv$", x)],
            reverse=True,
            key=lambda x: int(re.sub("[^0-9]", "", x)),
        )
        if len(cur_mtx_files) > 0:
            cur_mtx_file = cur_mtx_files[0]
            cur_mtx = pd.read_csv(cur_mtx_file, index_col=0)
            existing = set(cur_mtx.index) - (
                set(cur_mtx.index) - set(new_seq_df["gene_name"] + "|" + new_seq_df["kinase"])
            )
            not_existing = set(new_seq_df["gene_name"] + "|" + new_seq_df["kinase"]) - existing
            new_mtx_file = f"./pairwise_mtx_{len(existing) + len(not_existing)}.csv"
            if len(not_existing) == 0:
                return cur_mtx_file
            symbols_unk = new_seq_df[(new_seq_df["gene_name"] + "|" + new_seq_df["kinase"]).isin(not_existing)]
            new_seq_df.to_csv(tmp2.name)
            # Get restricted combinations
            mtx_utils.make_fasta(df_in=tmp2.name, fasta_out=tmp.name)
            thin_mtx = get_pairwise.get_needle_pairwise_mtx(
                tmp.name,
                new_mtx_file,
                num_processes=8,
                restricted_combinations=(list(existing), list(not_existing)),
            )
            with tf.NamedTemporaryFile() as small_fasta:
                symbols_unk.to_csv(small_fasta.name, index=False)
                small_fasta = mtx_utils.make_fasta(df_in=small_fasta.name, fasta_out=small_fasta.name)
                with tf.NamedTemporaryFile() as needle_out:
                    square_mtx = get_pairwise.get_needle_pairwise_mtx(small_fasta, needle_out.name, num_processes=8)
                    pass

            # Put it all together
            wide = pd.merge(cur_mtx, thin_mtx.T, how="left", left_index=True, right_index=True)
            missing_corner = pd.concat([wide, thin_mtx], axis=0)
            missing_corner.loc[square_mtx.index, square_mtx.columns] = square_mtx
            new_mtx = missing_corner
        else:
            new_seq_df.to_csv(tmp2.name)
            symbols_unk = new_seq_df[(new_seq_df["gene_name"] + "|" + new_seq_df["kinase"])]
            new_mtx_file = f"./pairwise_mtx_{len(symbols_unk)}.csv"
            with tf.NamedTemporaryFile() as small_fasta:
                symbols_unk.to_csv(small_fasta.name, index=False)
                small_fasta = mtx_utils.make_fasta(df_in=small_fasta.name, fasta_out=small_fasta.name)
                with tf.NamedTemporaryFile() as needle_out:
                    square_mtx = get_pairwise.get_needle_pairwise_mtx(small_fasta, needle_out.name, num_processes=8)
                    pass
            new_mtx = square_mtx

        new_mtx.to_csv(new_mtx_file)
        return new_mtx_file


def step_5_get_train_val_test_split(
    kin_fam_grp_filename,
    data_filename,
    seq_filename_A,
    new_mtx_file,
    part: Literal["a", "b", "c"],
    data_gen_conf={
        "held_out_percentile": 95,
        "train_percentile": 65,
        "dataframe_generation_mode": "tr-all",
    },
    num_restarts=60,
) -> tuple[str, str, str] | None:
    """Wrapper for `PreprocessingSteps.split_into_sets_individual_deterministic_top_k.split_into_sets` and `PreprocessingSteps.format_raw_data.get_input_dataframe`."""
    match part:
        case "a":
            logger.status("Step 5a: Obtaining train/val/test splits through hill climbing.")
            PreprocessingSteps.split_into_sets_individual_deterministic_top_k.split_into_sets(
                kin_fam_grp_filename,
                data_filename,
                tgt=0.3,
                get_restart=True,
                num_restarts=num_restarts,  # TODO: change back to 600
            )
            pprint.pprint(data_gen_conf)
        case "b":
            data_gen_conf = {"train_percentile": 65, "dataframe_generation_mode": "tr-all"}
            logger.status(
                "Step 5b: Generating dataframe with the following configuration"
                f" dictionary:\n{pprint.pformat(data_gen_conf)}"
            )
            return PreprocessingSteps.format_raw_data.get_input_dataframe(
                input_fn=data_filename,
                kin_seq_file=seq_filename_A,
                distance_matrix_file=new_mtx_file,
                config=data_gen_conf,
            )
        case "c":
            logger.status("Step 5c: Splitting into train/val/test files.")
            return PreprocessingSteps.format_raw_data.get_input_dataframe(
                input_fn=data_filename,
                kin_seq_file=seq_filename_A,
                distance_matrix_file=new_mtx_file,
                config=data_gen_conf,
            )

def step_6_drop_overlapping_sites(train_file: str, val_file: str, test_file: str):
    PreprocessingSteps.remove_overlaps.main(train_file, val_file, test_file)


def main():
    """The main function for preprocessing the data. Does the following:
    1. Downloads PSP
    2. Downloads Uniprot data
    3. Maps kinases to families and groups
    4. Gets pairwise distance matrix
    5a. Gets train/val/test split through hill climbing
    5b. Provides one file with all inputs
    5c. Provides, tr, vl, te files (analagous to 5b)
    6. Drops overlapping sites
    """

    step_1_download_psp()
    seq_filename_A, data_filename = step_2_download_uniprot()
    kin_fam_grp_filename = step_3_get_kin_to_fam_to_grp(seq_filename_A)
    new_mtx_file = step_4_get_pairwise_mtx(seq_filename_A)
    step_5_get_train_val_test_split(
        kin_fam_grp_filename, data_filename, seq_filename_A, new_mtx_file, part='a'
    )
    step_5_get_train_val_test_split(
        kin_fam_grp_filename, data_filename, seq_filename_A, new_mtx_file, part='b'
    )
    
    fn_res = step_5_get_train_val_test_split(
        kin_fam_grp_filename, data_filename, seq_filename_A, new_mtx_file, part='c'
    )
    assert fn_res is not None
    step_6_drop_overlapping_sites(*fn_res)
