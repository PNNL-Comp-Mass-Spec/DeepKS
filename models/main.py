import os
import pathlib
import re

where_am_i = pathlib.Path(__file__).parent.resolve()
os.chdir(where_am_i)

import sys
sys.path.append("../config/")
sys.path.append("../tools/")
sys.path.append("../data/preprocessing/")

import torch 
import torch.nn as nn
from tensorize import gather_data
from NNInterface import NNInterface
from matplotlib import rcParams
from formal_layers import Concatenation, Multiply, Transpose
import numpy as np
import argparse
import pickle
from SimpleTuner import SimpleTuner
from model_utils import cNNUtils as U
import config

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 13

NUM_EMBS = 22
SITE_LEN = 15


# Device configuration
# device = torch.device("cpu")
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def batch_dot(x, y):
    return torch.einsum("ij,ij->i", x, y)

class Attention(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.attn = nn.Linear(in_features, out_features)

    def forward(self, *X):
        x_a = self.attn(X[0])
        y_a = self.attn(X[1])
        weights = torch.tanh(batch_dot(x_a, y_a)).unsqueeze(1)
        return weights


class CNN(nn.Module):
    def __init__(self, out_channels, conv_kernel_size, pool_kernel_size, in_channels=NUM_EMBS, do_flatten=False, do_transpose=False):
        super().__init__()
        self.do_transpose = do_transpose
        if self.do_transpose:
            self.transpose = Transpose(1, 2)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size = conv_kernel_size)
        self.activation = nn.ELU() # nn.Tanh()
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size)
        self.do_flatten = do_flatten
        if self.do_flatten:
            self.flat = nn.Flatten(1)      
    
    def forward(self, x, ret_size = False):
        if self.do_transpose:
            out = self.transpose(x)
        else:
            out = x
        out = self.conv(out)
        out = self.activation(out)
        out = self.pool(out)
        if self.do_flatten:
            out = self.flat(out)
        if ret_size:
            return out.size()
        return out

class CNNW(nn.Module):
    def __init__(self, module_list):
        super().__init__()
        self.module_list = module_list
    def forward(self, x):
        out = x
        for mod in self.module_list:
            out = mod(out)
        return out

# Convolutional neural network (two convolutional layers)
class KinaseSubstrateRelationshipNN(nn.Module):
    def __init__(self, inp_size, num_classes=1, ll1_size = 200, ll2_size = 200, emb_dim = 30, num_conv_layers = 1, site_param_dict = {"kernels": [4], "out_lengths":[12], "out_channels":[11]}, kin_param_dict = {"kernels": [10], "out_lengths":[12], "out_channels":[11]}, dropout_pr=0.3):
        super().__init__()
        spvals = site_param_dict.values()
        kpvals = kin_param_dict.values()
        assert all(num_conv_layers == len(x) for x in list(spvals) + list(kpvals)), "Layer parameter lists do not all equal `num_conv_layers`"
        self.inp_size = inp_size

        self.emb_site = nn.Embedding(NUM_EMBS, emb_dim)
        self.emb_kin = nn.Embedding(NUM_EMBS, emb_dim)
        self.site_param_dict = site_param_dict
        self.kin_param_dict = kin_param_dict

        try:
           calculated_pools_site, calculated_pools_kin, calculated_in_channels_site, calculated_in_channels_kin, calculated_do_flatten_site, calculated_do_flatten_kin, calculated_do_transpose_site, calculated_do_transpose_kin = self.calculate_cNN_params(num_conv_layers)
        except RuntimeError as e:
            raise AssertionError(str(e) + "\n" + "S - {} K - {}".format(site_param_dict, kin_param_dict))

        self.site_cnns = nn.ModuleList([CNN(site_param_dict['out_channels'][i], site_param_dict['kernels'][i], calculated_pools_site[i], calculated_in_channels_site[i], calculated_do_flatten_site[i], calculated_do_transpose_site[i]) for i in range(num_conv_layers)])
        self.kin_cnns = nn.ModuleList([CNN(kin_param_dict['out_channels'][i], kin_param_dict['kernels'][i], calculated_pools_kin[i], calculated_in_channels_kin[i], calculated_do_flatten_kin[i], calculated_do_transpose_kin[i]) for i in range(num_conv_layers)])


        linear_size_sum, linear_size = self.get_size()

        self.attn = Attention(linear_size[0], linear_size[0])
        self.mult = Multiply()
        self.cat = Concatenation()
        
        self.linear = nn.Linear(linear_size_sum, ll1_size)
        
        self.activation = nn.ELU() # nn.Tanh()
        self.dropout = nn.Dropout(dropout_pr)
        self.intermediate = nn.Linear(ll1_size, ll2_size)
        self.final = nn.Linear(ll2_size, num_classes)
    
    def calculate_cNN_params(self, num_conv_layers):
        calculated_pools_site = []
        calculated_pools_kin = []
        calculated_in_channels_site = []
        calculated_in_channels_kin = []
        calculated_do_flatten_site = []
        calculated_do_flatten_kin = []
        calculated_do_transpose_site = []
        calculated_do_transpose_kin = []

        for i in range(num_conv_layers):
            if num_conv_layers == 1:
                calculated_pools_site.append(U.desired_conv_then_pool_shape(SITE_LEN, None, self.site_param_dict["out_lengths"][i], None, self.site_param_dict['kernels'][i], err_message = "S")[0])
                calculated_pools_kin.append(U.desired_conv_then_pool_shape(KIN_LEN, None, self.kin_param_dict["out_lengths"][i], None, self.kin_param_dict['kernels'][i], err_message = "K")[0])
                calculated_in_channels_site.append(NUM_EMBS)
                calculated_in_channels_kin.append(NUM_EMBS)
                calculated_do_flatten_site.append(True)
                calculated_do_flatten_kin.append(True)
                calculated_do_transpose_site.append(True)
                calculated_do_transpose_kin.append(True)
            else:
                if i == 0:
                    calculated_pools_site.append(U.desired_conv_then_pool_shape(SITE_LEN, None,  self.site_param_dict["out_lengths"][i], None,  self.site_param_dict['kernels'][i], err_message = "S")[0])
                    calculated_pools_kin.append(U.desired_conv_then_pool_shape(KIN_LEN, None,  self.kin_param_dict["out_lengths"][i], None,  self.kin_param_dict['kernels'][i], err_message = "K")[0])
                    calculated_in_channels_site.append(NUM_EMBS)
                    calculated_in_channels_kin.append(NUM_EMBS)
                    calculated_do_flatten_site.append(False)
                    calculated_do_flatten_kin.append(False)
                    calculated_do_transpose_site.append(True)
                    calculated_do_transpose_kin.append(True)
                if num_conv_layers - 1 > i > 0:
                    calculated_pools_site.append(U.desired_conv_then_pool_shape( self.site_param_dict["out_lengths"][i-1], None,  self.site_param_dict["out_lengths"][i], None,  self.site_param_dict['kernels'][i], err_message = "S")[0])
                    calculated_pools_kin.append(U.desired_conv_then_pool_shape( self.kin_param_dict["out_lengths"][i-1], None,  self.kin_param_dict["out_lengths"][i], None,  self.kin_param_dict['kernels'][i], err_message = "K")[0])
                    calculated_in_channels_kin.append( self.kin_param_dict['out_channels'][i-1])
                    calculated_in_channels_site.append( self.site_param_dict['out_channels'][i-1])
                    calculated_do_flatten_site.append(False)
                    calculated_do_flatten_kin.append(False)
                    calculated_do_transpose_site.append(False)
                    calculated_do_transpose_kin.append(False)
                if i == num_conv_layers - 1:
                    calculated_pools_site.append(U.desired_conv_then_pool_shape( self.site_param_dict["out_lengths"][i-1], None,  self.site_param_dict["out_lengths"][i], None,  self.site_param_dict['kernels'][i], err_message = "S")[0])
                    calculated_pools_kin.append(U.desired_conv_then_pool_shape( self.kin_param_dict["out_lengths"][i-1], None,  self.kin_param_dict["out_lengths"][i], None,  self.kin_param_dict['kernels'][i], err_message = "K")[0])
                    calculated_in_channels_kin.append( self.kin_param_dict['out_channels'][i-1])
                    calculated_in_channels_site.append( self.site_param_dict['out_channels'][i-1])
                    calculated_do_flatten_site.append(True)
                    calculated_do_flatten_kin.append(True)
                    calculated_do_transpose_site.append(False)
                    calculated_do_transpose_kin.append(False)
        
        return calculated_pools_site, calculated_pools_kin, calculated_in_channels_site, calculated_in_channels_kin, calculated_do_flatten_site, calculated_do_flatten_kin, calculated_do_transpose_site, calculated_do_transpose_kin
    
    def get_size(self):
        return (self.site_param_dict['out_channels'][-1] * self.site_param_dict['out_lengths'][-1] + self.kin_param_dict['out_channels'][-1] * self.kin_param_dict['out_lengths'][-1]), [self.site_param_dict['out_channels'][-1] * self.site_param_dict['out_lengths'][-1], self.kin_param_dict['out_channels'][-1] * self.kin_param_dict['out_lengths'][-1]]
    
    def forward(self, site_seq, kin_seq, ret_size=False):
        emb_site = self.emb_site(site_seq)
        emb_kin = self.emb_kin(kin_seq)

        out_site = emb_site
        out_kin = emb_kin
        for cnn in self.site_cnns:
            out_site = cnn(out_site)
        for cnn in self.kin_cnns:
            out_kin = cnn(out_kin)

        if ret_size:
            return np.sum([np.array(out_site.size()), np.array(out_kin.size())], axis=0)[-1], np.array([np.array(out_site.size()), np.array(out_kin.size())])[:, -1]

        weights = self.attn(out_site, out_kin)
        weights = torch.softmax(weights/out_site.size(-1)**(0.5), dim=-1)
        out_site = self.mult(out_site, weights)
        out_kin = self.mult(out_kin, weights)

        
        out = self.cat(out_site, out_kin)

        out =  self.linear(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.intermediate(out)
        out = self.activation(out)
        out = self.dropout(out)
        return self.final(out).squeeze()

torch.use_deterministic_algorithms(True)

def perform_k_fold(config, display_within_train = False, process_device = "cpu"):
    print(f"Using Device > {process_device} <")
    global NUM_EMBS
    
    # Hyperparameters
    for x in config:
        exec(f"{x} = {config[x]}")

    tokdict = pickle.load(open("../bin/tok_dict.pkl", "rb"))
    tokdict['-'] = tokdict['<PADDING>']

    (train_loader, _, _, _), info_dict = gather_data(train_filename, trf=1, vf=0, tuf=0, tef=0, train_batch_size=config['batch_size'], n_gram=config['n_gram'], tokdict=tokdict, device=process_device, maxsize=KIN_LEN)
    ( _, val_loader, _, _), info_dict = gather_data(val_filename, trf=0, vf=1, tuf=0, tef=0, train_batch_size=config['batch_size'], n_gram=config['n_gram'], tokdict=tokdict, device=process_device, maxsize=KIN_LEN)
    NUM_EMBS = 22
    
    results = []
    (_, _, _, test_loader), _ = gather_data(test_filename, trf=0, vf=0, tuf=0, tef=1, n_gram=config['n_gram'], tokdict=tokdict, device=process_device, maxsize=KIN_LEN)
    
    crit = torch.nn.BCEWithLogitsLoss()
    if isinstance(crit, torch.nn.BCEWithLogitsLoss):
        num_classes = 1
    elif isinstance(crit, torch.nn.CrossEntropyLoss):
        num_classes = 2

    torch.manual_seed(3)
    try:
        model = KinaseSubstrateRelationshipNN(num_classes=num_classes, inp_size=NNInterface.get_input_size(train_loader), ll1_size=config['ll1_size'], ll2_size=config['ll2_size'], emb_dim=config['emb_dim'], num_conv_layers=config['num_conv_layers'], site_param_dict=config['site_param_dict'], kin_param_dict=config['kin_param_dict'], dropout_pr=config['dropout_pr']).to(process_device)
    except AssertionError as ae:
        print("WARNING: AssertionError for the parameter set: ")
        print(ae)
        return tuple(["N/A"]*5)
    the_nn = NNInterface(model, crit, torch.optim.Adam(model.parameters(), lr=config['learning_rate']), inp_size=NNInterface.get_input_size(train_loader), model_summary_name="../architectures/architecture (id-%d).txt" %(U.id_params(config)), device=process_device)

    cutoff = 0.4
    metric = 'acc'
    
    if process_device == 'cpu':
        input("WARNING: Running without CUDA. Are you sure you want to proceed? Press any key to proceed. (ctrl + c to quit)\n")

    results.append(the_nn.train(train_loader, lr_decay_amount=config['lr_decay_amt'], lr_decay_freq=config['lr_decay_freq'], num_epochs=config['num_epochs'], include_val = True, val_dl = val_loader, fold = 0, maxfold=0, cutoff = cutoff, metric = metric)) 

    # the_nn.save_model(model.to('cpu'), "bin/current_best_model.pkl")
    # pickle.dump(train_loader, open("bin/train_loader.pkl", "wb"))
    # pickle.dump(val_loader, open("bin/val_loader.pkl", "wb"))

    
    the_nn.test(test_loader, roc = False, verbose = False, cutoff = cutoff, text=f"Test {metric} on fully held out for model.", metric = metric)
    
    del model, the_nn
    torch.cuda.empty_cache()

    results = np.array(results)
    if display_within_train:
        SimpleTuner.table_intermediates(config, results[:, 0].tolist(), np.mean(results[:, 0]), np.mean(results[:, 1]), np.std(results[:, 0]), np.std(results[:, 1]))
    return results[:, 0].tolist(), np.mean(results[:, 0]), np.mean(results[:, 1]), np.std(results[:, 0]), np.std(results[:, 1])  # accuracy, loss, acc_std, loss_std

mode = config.get_mode()
if mode == "no_alin":
    KIN_LEN = 4128
else:
    KIN_LEN = 9264

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    def device(arg_value):
        try:
            assert(bool(re.search("^cuda(:|)[0-9]*$", arg_value)) or bool(re.search("^cpu$", arg_value)))
            if "cuda" in arg_value:
                if arg_value == "cuda":
                    return arg_value
                cuda_num = int(re.findall("([0-9]+)", arg_value)[0])
                assert(0 <= cuda_num <= torch.cuda.device_count())
        except Exception:
            raise argparse.ArgumentTypeError(f"Device '{arg_value}' does not exist. Choices are {'cpu', 'cuda[:<gpu #>]'}.")
        
        return arg_value
        
    parser.add_argument("--device", type=device, help="Specify device. Choices are {'cpu', 'cuda:<gpu#>'}.", metavar='<device>', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--train", type=str, help="Specify train file name", required=True, metavar='<train_file_name.csv>')
    parser.add_argument("--val", type=str, help="Specify validation file name", required=True, metavar='<val_file_name.csv>')
    parser.add_argument("--test", type=str, help="Specify test file name", required=True, metavar='<test_file_name.csv>')

    args = vars(parser.parse_args())
    train_filename = args['train']
    val_filename = args['val']
    test_filename = args['test']

    assert 'formatted' in train_filename, "'formatted' is not in the train filename. Did you select the correct file?"
    assert 'formatted' in val_filename, "'formatted' is not in the test filename. Did you select the correct file?"
    assert 'formatted' in test_filename, "'formatted' is not in the test filename. Did you select the correct file?"    
    assert os.path.exists(train_filename), f"Train file '{train_filename}' does not exist."
    assert os.path.exists(test_filename), f"Val file '{val_filename}' does not exist."
    assert os.path.exists(test_filename), f"Test file '{test_filename}' does not exist."

    # torch.use_deterministic_algorithms(True)
    cf = {
        "learning_rate": 0.003,
        "batch_size": 64,
        "ll1_size": 50,
        "ll2_size": 25,
        "emb_dim": 22,
        "num_epochs": 10,
        "n_gram": 1,
        "lr_decay_amt": 0.4,
        "lr_decay_freq": 3,
        "num_conv_layers": 1,
        "dropout_pr": 0.4,
        "site_param_dict": {"kernels": [8], "out_lengths": [8], "out_channels": [20]},
        "kin_param_dict": {"kernels": [80], "out_lengths": [8], "out_channels": [20]},
    }
    

    perform_k_fold(cf, display_within_train = True, process_device=args['device'])
