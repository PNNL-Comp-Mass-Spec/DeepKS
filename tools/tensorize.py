import numpy as np, psutil, torch, pandas as pd, collections, json, tqdm, random, torch.utils.data
from typing import Generator, Union
from ..tools import model_utils
from termcolor import colored


def get_tok_dict(data, n_gram=3, verbose=False, include_metadata=False):
    tok_dict = {}
    tok_occurrence = collections.Counter()
    all_iters = 0

    def iter_through(iterable):
        nonlocal all_iters
        for x in iterable:
            for i in range(len(x) - n_gram + 1):
                all_iters += 1
                if x[i : i + n_gram] not in tok_dict:
                    tok_dict[x[i : i + n_gram]] = len(tok_dict)
                tok_occurrence[x[i : i + n_gram]] += 1

    iter_through(data["Kinase Sequence"].unique())
    iter_through(data["Site Sequence"].unique())
    assert sum(tok_occurrence.values()) == all_iters
    if verbose:
        print("Token frequency (top 50):", tok_occurrence.most_common(50))
        print("Total iterations: ", all_iters)
        print(f"Total unique {n_gram}-grams: ", len(tok_dict))

    if include_metadata:
        tok_dict["<PADDING>"] = len(tok_dict)
        tok_dict["<N-GRAM>"] = n_gram

    return dict(tok_dict)


def encode_seq(seq: str, mapping_dict: dict[str, int]) -> list[int]:
    res = []
    assert isinstance(mapping_dict["<N-GRAM>"], int)
    n_gram: int = mapping_dict["<N-GRAM>"]
    for i in range(len(seq) - n_gram + 1):
        res.append(mapping_dict[seq[i : i + n_gram]])
    return res


def gather_data(
    input_t: Union[str, pd.DataFrame, dict[str, Union[list[str], list[int]]]],
    input_d="",
    trf=0.7,
    vf=0.1,
    tuf=0.1,
    tef=0.1,
    train_batch_size=3,
    clusterfile=None,
    mc=False,
    maxsize=None,
    tokdict=None,
    subsample_num=None,
    ret_info=False,
    n_gram=3,
    device=torch.device("cpu"),
    eval_batch_size=None,
    cartesian_product=False,
    tqdm_passthrough=[None],
    kin_seq_to_group: dict = {},
) -> Generator:
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
    if len(tqdm_passthrough) == 1:
        tq = tqdm_passthrough[0]
        # tqdm_passthrough[0].write("\r" + " " * os.get_terminal_size()[0], end="\r")
        if tq is not None:
            tq.write(colored("...(Re)loading Tensors into Device for Next Chunk...", "blue"), end="\r")
    assert abs(sum([trf, vf, tuf, tef, -1]) < 1e-16)

    # ===== #
    def package(desired_length=None, desired_chunk_pos=None):
        nonlocal eval_batch_size, X_train, X_train_kin, y_train, X_val, X_val_kin, y_val, X_tune, X_tune_kin, y_tune, X_test, X_test_kin, y_test, device
        if trf > 0:
            train_loader = torch.utils.data.DataLoader(
                model_utils.KSDataset(X_train, X_train_kin, y_train),
                batch_size=train_batch_size,
            )
        else:
            train_loader = None

        # eval_batch_size = (
        #     len(X_val)
        #     if vf > 0 and eval_batch_size is None
        #     else len(X_tune)
        #     if tuf > 0 and eval_batch_size is None
        #     else len(X_test)
        #     if tef > 0 and eval_batch_size is None
        #     else eval_batch_size
        # )

        eval_batch_size = 10000 if "cuda" in str(device) else 100  # max(
        #     [
        #         len(X) // (1 if "cuda" in str(device) else (9 if len(X) % 10 != 0 else 10))
        #         for X in [X_val, X_tune, X_test]
        #     ]
        #     + [1]
        # )

        if vf > 0:
            val_loader = torch.utils.data.DataLoader(
                model_utils.KSDataset(X_val, X_val_kin, y_val), batch_size=eval_batch_size
            )
        else:
            val_loader = None
        if tuf > 0:
            tune_loader = torch.utils.data.DataLoader(
                model_utils.KSDataset(X_tune, X_tune_kin, y_tune), batch_size=eval_batch_size
            )
        else:
            tune_loader = None
        if tef > 0:
            test_loader = torch.utils.data.DataLoader(
                model_utils.KSDataset(X_test, X_test_kin, y_test), batch_size=eval_batch_size
            )
        else:
            test_loader = None
        if isinstance(data, pd.DataFrame):
            ret_info_dict = {
                "kin_orders": {
                    "train": data.loc[train_ids]["Gene Name of Kin Corring to Provided Sub Seq"].to_list(),
                    "val": data.loc[val_ids]["Gene Name of Kin Corring to Provided Sub Seq"].to_list(),
                    "test": data.loc[test_ids]["Gene Name of Kin Corring to Provided Sub Seq"].to_list(),
                },
                "orig_symbols_order": {
                    "train": data.loc[train_ids]["Gene Name of Kin Corring to Provided Sub Seq"].to_list(),
                    "val": data.loc[val_ids]["Gene Name of Kin Corring to Provided Sub Seq"].to_list(),
                    "test": data.loc[test_ids]["Gene Name of Kin Corring to Provided Sub Seq"].to_list(),
                },
                "PairIDs": {
                    "train": (
                        data.loc[train_ids]["pair_id"].to_list()
                        if ("pair_id" in (data.columns if isinstance(data, pd.DataFrame) else data))
                        else [f"N/A # {i}" for i in range(len(train_ids))]
                    ),
                    "val": (
                        data.loc[val_ids]["pair_id"].to_list()
                        if ("pair_id" in (data.columns if isinstance(data, pd.DataFrame) else data))
                        else [f"N/A # {i}" for i in range(len(val_ids))]
                    ),
                    "test": (
                        data.loc[test_ids]["pair_id"].to_list()
                        if ("pair_id" in (data.columns if isinstance(data, pd.DataFrame) else data))
                        else [f"N/A # {i}" for i in range(len(test_ids))]
                    ),
                },
            }
        else:
            desired_length = len(data["Site Sequence"]) if desired_length is None else desired_length
            desired_chunk_pos = 0 if desired_chunk_pos is None else desired_chunk_pos
            ret_info_dict = {
                "kin_orders": {
                    "train": ["N/A"],  # CHECK -- may not be correct in non-predict mode
                    "val": ["N/A"],
                    "test": ["N/A"],
                },
                "orig_symbols_order": {
                    "train": [
                        data["Gene Name of Kin Corring to Provided Sub Seq"][i // len(data["Site Sequence"])]
                        for i in train_ids
                    ][desired_length * desired_chunk_pos : desired_length * (desired_chunk_pos + 1)],
                    "val": [
                        data["Gene Name of Kin Corring to Provided Sub Seq"][i // len(data["Site Sequence"])]
                        for i in val_ids
                    ][desired_length * desired_chunk_pos : desired_length * (desired_chunk_pos + 1)],
                    "test": [
                        data["Gene Name of Kin Corring to Provided Sub Seq"][i // len(data["Site Sequence"])]
                        for i in test_ids
                    ][desired_length * desired_chunk_pos : desired_length * (desired_chunk_pos + 1)],
                },
                "PairIDs": {  # FIXME!
                    "train": data["pair_id"][
                        desired_length * desired_chunk_pos : desired_length * (desired_chunk_pos + 1)
                    ],
                    "val": data["pair_id"][
                        desired_length * desired_chunk_pos : desired_length * (desired_chunk_pos + 1)
                    ],
                    "test": data["pair_id"][
                        desired_length * desired_chunk_pos : desired_length * (desired_chunk_pos + 1)
                    ],
                },
            }
        update_dict = {
            "classes": classes,
            "class_labels": class_labels,
            "remapping_class_label_dict_inv": remapping_class_label_dict_inv,
            "maxsize": max_kin_length,
            "tok_dict": tok_dict,
            "on_chunk": desired_chunk_pos,
            "total_chunks": num_partitions,
        }
        ret_info_dict.update(update_dict)

        return (train_loader, val_loader, tune_loader, test_loader), ret_info_dict

    # ===== #

    data: Union[pd.DataFrame, dict[str, Union[list[str], list[int]]]]
    if isinstance(input_t, str):
        if input_d == "":
            data = pd.read_csv(input_t)
        else:
            data_t = pd.read_csv(input_t)
            data_d = pd.read_csv(input_d)
            data = pd.concat([data_t, data_d], axis=0)
    else:
        data = input_t
    if subsample_num is not None and isinstance(data, pd.DataFrame):  # TODO Could improve this
        try:
            subsample_num = int(subsample_num)
            data = data.sample(n=subsample_num)
        except Exception as e:
            print("Not subsampling (error: {})".format(e), flush=True)

    if maxsize is None:
        assert all([isinstance(x, str) for x in data["Kinase Sequence"]]), "All kinase seqs must be strings"
        max_kin_length = max([len(str(x)) for x in data["Kinase Sequence"]]) - n_gram + 1
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
        data["Kinase Sequence"] = [clust_dict[x] for x in data["Kinase Sequence"]]

    if kin_seq_to_group:
        data["Group"] = [kin_seq_to_group[x] for x in data["Kinase Sequence"]]
        if isinstance(data, pd.DataFrame):
            data = data[data["Group"] != "<UNANNOTATED>"]
        else:
            data_DFed = pd.DataFrame(data)
            data_DFed = data_DFed[data_DFed["Group"] != "<UNANNOTATED>"]
            data_intermediary = data_DFed.to_dict("list")
            data_intermediary = {str(k): v for k, v in data_intermediary.items()}
            data = data_intermediary

    data = data.copy()
    class_col = ("Kinase Sequence" if mc else "Class") if not kin_seq_to_group else "Group"
    class_labels = sorted(list(set(data[class_col])))
    classes = len(class_labels)
    if set(class_labels) in [{0, 1}, {1}, {0}]:
        remapping_class_label_dict = {0: 0, 1: 1}
        remapping_class_label_dict_inv = {0: "Decoy", 1: "Real"}
        remapping_class_label_dict_inv = remapping_class_label_dict_inv
        assert all(isinstance(x, int) for x in class_labels) or all(
            isinstance(x, str) and x.isdigit() for x in class_labels
        ), f"Class labels must be integers ({class_labels[:10]=})"
        data[class_col] = [remapping_class_label_dict[int(x)] for x in data[class_col]]
    elif set(class_labels) != {-1}:  # I.e., not predict mode
        remapping_class_label_dict = {class_labels[i]: i for i in range(len(class_labels))}
        remapping_class_label_dict_inv = {i: list(class_labels)[i] for i in range(len(class_labels))}
        remapping_class_label_dict_inv = remapping_class_label_dict_inv
        data[class_col] = [remapping_class_label_dict[x] for x in data[class_col]]
    else:
        remapping_class_label_dict = None
        remapping_class_label_dict_inv = None

    random.seed(99354)
    if isinstance(data, pd.DataFrame):
        data = data.reset_index(drop=True)
        possible_ids = list(range(len(data)))
    else:
        possible_ids = list(range(len(data["Kinase Sequence"]) * len(data["Site Sequence"])))

    shuffled_ids = possible_ids.copy()

    random.shuffle(shuffled_ids)
    tot_len = len(shuffled_ids)
    train_ids: list[int] = shuffled_ids[: int(tot_len * trf)]
    val_ids: list[int] = shuffled_ids[int(tot_len * trf) : int(tot_len * (trf + vf))]
    tune_ids: list[int] = shuffled_ids[int(tot_len * (trf + vf)) : int(tot_len * (trf + vf + tuf))]
    test_ids: list[int] = shuffled_ids[int(tot_len * (trf + vf + tuf)) :]

    assert train_ids + val_ids + tune_ids + test_ids == shuffled_ids

    data_train, data_val, data_tune, data_test, kinase_seq_to_tensor_data, site_seq_to_tensor_data = tuple(6 * [{}])
    if isinstance(data, pd.DataFrame):
        data_train = data.loc[train_ids]
        data_val = data.loc[val_ids]
        data_tune = data.loc[tune_ids]
        data_test = data.loc[test_ids]
    else:
        assert all([isinstance(x, str) for x in data["Kinase Sequence"]]), "All kinase seqs must be strings"
        assert all([isinstance(x, str) for x in data["Site Sequence"]]), "All site seqs must be strings"
        kinase_seq_to_tensor_data: dict[str, torch.IntTensor] = {
            str(ks): torch.IntTensor(pad(encode_seq(str(ks), tok_dict), max_len=max_kin_length, map_dict=tok_dict))
            for ks in data["Kinase Sequence"]
        }
        site_seq_to_tensor_data: dict[str, torch.IntTensor] = {
            str(ss): torch.IntTensor(encode_seq(str(ss), tok_dict)) for ss in data["Site Sequence"]
        }

    assert all(isinstance(x, str) for x in data["Kinase Sequence"]), "Kinase names must be strings"
    assert all(isinstance(x, str) for x in data["Site Sequence"]), "Site sequences must be strings"

    BYTES_PER_PAIR = 1.4e6  # The approximate size of one kinase-site pair in bytes + forward pass size.
    BYTES_PER_PAIR_MULTIPLIER = 1  # Optionally pretend data is larger than the original data
    BYTES_PER_PAIR *= BYTES_PER_PAIR_MULTIPLIER

    if str(device) == str(torch.device("cpu")) or "mps" in str(device):
        free_ram_and_swap_B = (psutil.virtual_memory().available + psutil.swap_memory().free) / 1.25
        # print(colored(f"{psutil.virtual_memory().available=}{psutil.swap_memory().free=}", "yellow"))
    else:
        assert "cuda" in str(device), "Device must be either 'cpu' or a cuda device."
        free_ram_and_swap_B = torch.cuda.mem_get_info(device)[0]

    num_pairs_can_be_stored_per_dl = int(free_ram_and_swap_B / BYTES_PER_PAIR)

    assert (
        len(data["Kinase Sequence"]) == len(data["Site Sequence"]) or cartesian_product
    ), "Length of kinase and site lists must be equal."

    assert num_pairs_can_be_stored_per_dl > 0, "Can't fit one pair in memory. Check system memory usage."
    num_partitions = max(
        [int(np.ceil(len(x) / num_pairs_can_be_stored_per_dl)) for x in [train_ids, val_ids, tune_ids, test_ids]] + [1]
    )
    assert num_partitions > 0, "num_partitions <= 0. Something went wrong."
    # assert num_partitions <= len(data['Kinase Sequence'), f"{num_partitions=} > {len(data['Kinase Sequence')=}. Something went wrong."

    # tqdm_passthrough[0].write(colored(f"Status: Partitioning data into {num_partitions} partitions", "green"))
    partition_size = min(
        num_pairs_can_be_stored_per_dl,
        (
            len(data["Kinase Sequence"]) * len(data["Site Sequence"])
            if cartesian_product
            else len(data["Kinase Sequence"])
        ),
    )
    for partition_id in range(int(num_partitions)):
        final_kin_tensor_chunks = []
        final_site_tensor_chunks = []
        final_class_chunks = []
        # print(colored(f"Status: Writing Tensor Data to RAM", "green"))
        begin_idx = partition_size * partition_id
        end_idx = partition_size * (partition_id + 1)
        train_ids_subset = train_ids[begin_idx:end_idx]
        val_ids_subset = val_ids[begin_idx:end_idx]
        tune_ids_subset = tune_ids[begin_idx:end_idx]
        test_ids_subset = test_ids[begin_idx:end_idx]

        actual_partition_size = end_idx - begin_idx

        if isinstance(data, pd.DataFrame):
            assert data_train is not None
            assert data_val is not None
            assert data_tune is not None
            assert data_test is not None
            X_train, X_val, X_tune, X_test = tuple(
                [
                    torch.IntTensor([encode_seq(x, tok_dict) for x in d_set["Site Sequence"].values]).to(device)
                    for d_set in [data_train, data_val, data_tune, data_test]
                ]
            )
            if not mc:
                X_train_kin, X_val_kin, X_tune_kin, X_test_kin = tuple(
                    [
                        torch.IntTensor(
                            [
                                pad(encode_seq(x, tok_dict), max_kin_length, tok_dict)
                                for x in d_set["Kinase Sequence"].values
                            ]
                        ).to(device)
                        for d_set in [data_train, data_val, data_tune, data_test]
                    ]
                )
            else:
                X_train_kin, X_val_kin, X_tune_kin, X_test_kin = tuple([None] * 4)
            y_train, y_val, y_tune, y_test = tuple(
                [
                    torch.IntTensor(d_set[class_col].values).to(device)
                    for d_set in [data_train, data_val, data_tune, data_test]
                ]
            )
        else:
            rand_idx_to_kin_idx_site_idx = lambda rand_idx: (
                int(rand_idx // len(data["Site Sequence"])),
                int(rand_idx // len(data["Kinase Sequence"])),
            )
            for _, idx_set in enumerate([train_ids_subset, val_ids_subset, tune_ids_subset, test_ids_subset]):
                kin_tensor_data: list[torch.Tensor] = []
                site_tensor_data: list[torch.Tensor] = []
                class_tensor_data: list[torch.Tensor] = []
                for idx in idx_set:
                    kin_tensor_idx, site_tensor_idx = rand_idx_to_kin_idx_site_idx(idx)

                    kin_tensor, site_tensor = (
                        kinase_seq_to_tensor_data[str(data["Kinase Sequence"][kin_tensor_idx])],
                        site_seq_to_tensor_data[str(data["Site Sequence"][site_tensor_idx])],
                    )
                    kin_tensor_data.append(kin_tensor)
                    site_tensor_data.append(site_tensor)
                    class_tensor_data.append(torch.IntTensor([-1]))  # CHECK -- May not be right for non-predict mode.
                stacked_kin_tensors = (
                    torch.stack(kin_tensor_data).to(device) if len(kin_tensor_data) > 0 else torch.IntTensor()
                )
                stacked_site_tensors = (
                    torch.stack(site_tensor_data).to(device) if len(site_tensor_data) > 0 else torch.IntTensor()
                )
                stacked_class_tensors = (
                    torch.cat(class_tensor_data).to(device) if len(class_tensor_data) > 0 else torch.IntTensor()
                )
                final_kin_tensor_chunks.append(stacked_kin_tensors)
                final_site_tensor_chunks.append(stacked_site_tensors)
                final_class_chunks.append(stacked_class_tensors)
                # print(colored(f"Status: Done writing to Tensor", "green"))
            X_train_kin, X_val_kin, X_tune_kin, X_test_kin = tuple(final_kin_tensor_chunks)
            X_train, X_val, X_tune, X_test = tuple(final_site_tensor_chunks)
            y_train, y_val, y_tune, y_test = tuple(final_class_chunks)
        yield package(actual_partition_size, partition_id)


def pad(tok_list: list[int], max_len: int, map_dict: dict[str, int]) -> list[int]:
    return tok_list + [map_dict["<PADDING>"] for _ in range(max_len - len(tok_list))]


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
    max_kin_len, tok_dict = get_info(
        "data/raw_data_train_8151_formatted.csv", "data/raw_data_held_out_998_formatted.csv", n
    )
    (a, b, c, d), _ = gather_data(
        "data/raw_data_train_8151_formatted.csv", maxsize=max_kin_len, tokdict=tok_dict, n_gram=n
    )
    print()
