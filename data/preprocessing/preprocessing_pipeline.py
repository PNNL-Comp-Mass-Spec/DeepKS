import os, re, prettytable as prt
from turtle import update
import sys
import pathlib
import os
import subprocess
import pprint

where_am_i = pathlib.Path(__file__).parent.resolve()
os.chdir(where_am_i)
raise DeprecationWarning("Use data.preprocessing.main instead.")
from PreprocessingSteps import remove_overlaps, format_raw_data_DD, get_kin_fam_grp, get_labeled_distance_matrix, remove_overlaps, split_into_sets, format_raw_data
from config import get_mode, set_mode

import subprocess
import time
import tempfile

def os_system_and_get_stdout(cmd):
    """
    cmd: command to run

    NOTE: output line must contain `[@python_capture_output]` to be presented in output
    """
    TAG = "[@python_capture_output]"

    tf = tempfile.TemporaryFile()
    p = subprocess.Popen(cmd, shell=True, stdout=tf)

    where = 0
    res_out = []
    while p.poll() is None:
        time.sleep(0.01)
        tf.seek(where)
        o = tf.read()
        where += len(o)
        ol = o.decode("UTF-8").split("\n")
        for line in ol:
            if line != "" and TAG not in line:
                print(line)
            elif TAG in line:
                res_out.append(re.sub(f"({TAG})".replace("[", r"\[").replace("]", r"\]"), "", line))
    
    return res_out

def main():
    mode = input("Process [a]ligned or [u]naligned sequences? [a|u]: ")
    if mode.lower() == "a":
        set_mode("alin")
    elif mode.lower() == "u":
        set_mode("no_alin")
    else:
        print("Bad input. Exiting.")
        exit(1)
    
    print("\n--- Getting sequence information and alignments [Running R scripts] ---")
    seq_filename, data_filename = tuple(os_system_and_get_stdout("Rscript ML_data_pipeline.R"))
    print("\n--- [Done with R] ---")


    print("\n--- Removing Duplicates --- ")
    new_fn = remove_all_duplicates.main(data_filename)

    print("\n--- Converting into a convenient format for the model and arranging target/decoy data ---")
    data_gen_conf = {
        "held_out_percentile": 95,
        "train_percentile": 50,
        "num_held_out_kins": 60
    }

    print(f"The default preprocessing settings are:\n{pprint.pformat(data_gen_conf)}")
    print("See readme.md for a guide on these settings.")
    print("To leave these settings at default, press <RETURN>. Otherwise, enter new key-value pairs below.")
    new_config = input(">>> ")

    if new_config != "":
        update_dict = eval(new_config)
        for k in update_dict:
            if 'num' in k:
                update_dict[k] = int(update_dict[k])

        data_gen_conf.update(update_dict)

    print(f"The new config is \n{pprint.pformat(data_gen_conf)}")
    
    format_raw_data.generate_real(new_fn, seq_filename, data_gen_conf)
    print("\n--- Completed Preprocessing! ---")

if __name__ == "__main__":
    main()