import sys
sys.path.append("../tools/")
sys.path.append("../data/preprocessing/")

from SimpleTuner import SimpleTuner
import sys
import torch
from NNInterface import *
import multiprocessing
from attention_one_layer_k_fold import perform_k_fold
import json

num_classes = 1

# torch.use_deterministic_algorithms(True)

def my_tune(hp_ss, num_samples = 50, num_sim_procs = 1, collapse=[[]]):
    st = SimpleTuner(num_sim_procs, perform_k_fold, hp_ss, num_samples, collapse=collapse)
    st.go()


if __name__ == "__main__":

    print("Is CUDA available?", torch.cuda.is_available(), flush=True)
    # assert torch.cuda.is_available()
    multiprocessing.set_start_method("spawn")

    # Main Call
    cf = SimpleTuner.get_config_dict('cf_new_data')
    my_tune(cf, collapse = [['site_param_dict', 'kin_param_dict']], num_sim_procs=4, num_samples=35)
