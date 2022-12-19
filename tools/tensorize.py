from typing import Tuple, Union
import torch
import pandas as pd
import collections
import json
import random
import model_utils
import torch.utils.data
import sys

def get_tok_dict(data, n_gram = 3, verbose = False, include_metadata = False):
    tok_dict = {}
    tok_occurrence = collections.Counter()
    all_iters = 0
    def iter_through(iterable):
        nonlocal all_iters
        for x in iterable:
            for i in range(len(x) - n_gram + 1):
                all_iters += 1
                if x[i:i+n_gram] not in tok_dict:
                    tok_dict[x[i:i+n_gram]] = len(tok_dict)
                tok_occurrence[x[i:i+n_gram]] += 1

    iter_through(data["lab"].unique())
    iter_through(data["seq"].unique())
    assert(sum(tok_occurrence.values()) == all_iters)
    if verbose:
        print("Token frequency (top 50):", tok_occurrence.most_common(50))
        print("Total iterations: ", all_iters)
        print(f"Total unique {n_gram}-grams: ", len(tok_dict))

    if include_metadata:
        tok_dict['<PADDING>'] = len(tok_dict)
        tok_dict['<N-GRAM>'] = n_gram

    return dict(tok_dict)

def encode_seq(seq, mapping_dict):
    res = []
    n_gram = mapping_dict['<N-GRAM>']
    for i in range(len(seq) - n_gram + 1):
        res.append(mapping_dict[seq[i:i+n_gram]])
    return res

def gather_data(
    input_t: Union[str, pd.DataFrame],
    input_d="",
    trf=0.7,
    vf=0.1,
    tuf=0.1,
    tef=0.1,
    train_batch_size=3,
    clusterfile=None,
    mc = False,
    maxsize = None, 
    tokdict = None,
    subsample_num = None, 
    ret_info = False, 
    n_gram = 3,
    device = torch.device("cpu")) -> Union[Tuple[Tuple[int, None, None, None], dict], Tuple[Tuple[Union[torch.utils.data.DataLoader,None],Union[torch.utils.data.DataLoader,None],Union[torch.utils.data.DataLoader,None],Union[torch.utils.data.DataLoader,None]],dict]]:
    """
    input_t: target set filename
    input_d: decoy set filename
    trf: training set fraction
    vf: validation set fraction
    tuf: tune set fraction
    tef: test set fraction
    train_batch_size: batch size for training set (all other sets come in 1 batch)
    clusterfile: filename of json defining kinase clusters
    """
    assert abs(sum([trf, vf, tuf, tef, -1]) < 1e-16)


    if isinstance(input_t, str):
        if input_d == "":
            data = pd.read_csv(input_t)
        else:
            data_t = pd.read_csv(input_t)
            data_d = pd.read_csv(input_d)
            data = pd.concat([data_t, data_d], axis=0)
    else:
        data = input_t

    if subsample_num is not None:
        try:
            subsample_num = int(subsample_num)
            data = data.sample(n=subsample_num)
        except Exception as e:
            print("Not subsampling (error: {})".format(e), flush=True)
    
    if maxsize is None:
        max_kin_length = max([len(x) for x in data["lab"].values]) - n_gram + 1
    else:
        max_kin_length = maxsize

    if tokdict is None:
        tok_dict = get_tok_dict(data, verbose=False, n_gram=n_gram, include_metadata=False)
        tok_dict["<PADDING>"] = len(tok_dict)
        tok_dict["<N-GRAM>"] = n_gram
    else:
        tok_dict = tokdict
    
    
    if ret_info:
        return (max_kin_length, None, None, None), tok_dict
    
    

    if clusterfile is not None:
        clust_dict = collections.defaultdict(list)
        clust_dict_json = json.load(open("data/clusters.json", "r"))
        for k, v in clust_dict_json.items():
            clust_dict[v].append(k)
        clust_dict = {", ".join(clust_dict[k]): v for k, v in dict(clust_dict).items()}
        final_clust_dict = {}
        for k, v in clust_dict.items():
            for vv in v:
                final_clust_dict[vv] = k
        clust_dict = final_clust_dict
        data["lab"] = data["lab"].apply(lambda x: clust_dict[x])

    class_col = "lab" if mc else "class"
    class_labels = pd.unique(data[class_col])
    classes = len(class_labels)
    if set(class_labels) in [{0, 1}, {1}, {0}]:
        remapping_class_label_dict = {0: 0, 1: 1}
        remapping_class_label_dict_inv = {0: "Decoy", 1: "Real"}
        remapping_class_label_dict_inv = remapping_class_label_dict_inv
        data[class_col] = data[class_col].apply(lambda x: remapping_class_label_dict[x])
    else:
        remapping_class_label_dict = {
            class_labels[i]: i for i in range(len(class_labels))
        }
        remapping_class_label_dict_inv = {
            i: list(class_labels)[i] for i in range(len(class_labels))
        }
        remapping_class_label_dict_inv = remapping_class_label_dict_inv
        data[class_col] = data[class_col].apply(lambda x: remapping_class_label_dict[x])

    random.seed(99354)
    shuffled_ids = list(range(len(data)))
    random.shuffle(shuffled_ids)
    tot_len = len(shuffled_ids)
    train_ids = shuffled_ids[: int(tot_len * trf)]
    val_ids = shuffled_ids[int(tot_len * trf) : int(tot_len * (trf + vf))]
    tune_ids = shuffled_ids[int(tot_len * (trf + vf)) : int(tot_len * (trf + vf + tuf))]
    test_ids = shuffled_ids[int(tot_len * (trf + vf + tuf)) :]

    assert train_ids + val_ids + tune_ids + test_ids == shuffled_ids

    data_train = data.iloc[train_ids]
    data_val = data.iloc[val_ids]
    data_tune = data.iloc[tune_ids]
    data_test = data.iloc[test_ids]


    X_train, X_val, X_tune, X_test = tuple(
        [
            torch.IntTensor([encode_seq(x, tok_dict) for x in d_set["seq"].values]).to(device)
            for d_set in [data_train, data_val, data_tune, data_test]
        ]
    )
    if not mc:
        X_train_kin, X_val_kin, X_tune_kin, X_test_kin = tuple(
            [
                torch.IntTensor([pad(encode_seq(x, tok_dict), max_kin_length, tok_dict) for x in d_set["lab"].values]).to(device)
                for d_set in [data_train, data_val, data_tune, data_test]
            ]
        )
    else:
        X_train_kin, X_val_kin, X_tune_kin, X_test_kin = tuple([None]*4)
    y_train, y_val, y_tune, y_test = tuple(
        [
            torch.IntTensor(d_set[class_col].values).to(device)
            for d_set in [data_train, data_val, data_tune, data_test]
        ]
    )

    if trf > 0:
        train_loader = torch.utils.data.DataLoader(
            model_utils.KSDataset(X_train, X_train_kin, y_train),
            batch_size=train_batch_size,
        )
    else:
        train_loader = None
    if vf > 0:
        val_loader = torch.utils.data.DataLoader(
            model_utils.KSDataset(X_val, X_val_kin, y_val), batch_size=len(X_val)
        )
    else:
        val_loader = None
    if tuf > 0:
        tune_loader = torch.utils.data.DataLoader(
            model_utils.KSDataset(X_tune, X_tune_kin, y_tune), batch_size=len(X_tune)
        )
    else:
        tune_loader = None
    if tef > 0:
        test_loader = torch.utils.data.DataLoader(
            model_utils.KSDataset(X_test, X_test_kin, y_test), batch_size=len(X_test)
        )
    else:
        test_loader = None

    return (train_loader, val_loader, tune_loader, test_loader), {
        "kin_orders": {'train': data.loc[train_ids]['lab_name'].to_list(), 'val': data.loc[val_ids]['lab_name'].to_list(), 'test':data.loc[test_ids]['lab_name'].to_list()},
        "classes": classes,
        "class_labels": class_labels,
        "remapping_class_label_dict_inv": remapping_class_label_dict_inv,
        "maxsize": max_kin_length, 
        "tok_dict": tok_dict
    }

def pad(tok_list, max_len, map_dict):
    return tok_list + [map_dict['<PADDING>'] for _ in range(max_len - len(tok_list))]

def get_info(target_file, decoy_file="", n_gram=1):
    (max_length0, _, _, _), tok_dict0 = gather_data(target_file, ret_info=True, n_gram=n_gram)
    if decoy_file != "":
        max_length1, tok_dict1 = gather_data(decoy_file, ret_info=True, n_gram=n_gram)
    else:
        max_length1, tok_dict1 = 0, {}

    assert isinstance(max_length0, int)
    max_length = max(max_length0, max_length1)
    for k in tok_dict1: 
        if k not in tok_dict0:
            # print("Adding", k)
            tok_dict0[k] = len(tok_dict0)
    
    tok_dict = tok_dict0
    if "<PADDING>" not in tok_dict:
        tok_dict["<PADDING>"] = len(tok_dict)
    if "<N-GRAM>" not in tok_dict:
        tok_dict["<N-GRAM>"] = n_gram

    return max_length, tok_dict

if __name__ == "__main__":
    n = 3
    max_kinase_length, tok_dict = get_info("data/raw_data_train_8151_formatted.csv", "data/raw_data_held_out_998_formatted.csv", n)
    (a, b, c, d), _ = gather_data("data/raw_data_train_8151_formatted.csv", maxsize=max_kinase_length, tokdict=tok_dict, n_gram=n)
    print()