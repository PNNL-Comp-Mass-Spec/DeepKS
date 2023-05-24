"""Functions to convert textual biological sequences into tensors for deep learning models."""

import numpy as np, psutil, torch, pandas as pd, collections, json, tqdm, random, torch.utils.data
from sympy import cartes
from typing import Generator, Union, Literal, Any
from ..tools import model_utils
from termcolor import colored

from ..config.logging import get_logger

logger = get_logger()
"""The logger for this module."""
if __name__ == "__main__":  # pragma: no cover
    logger.status("Loading Modules...")


def encode_label(labels: list[int], mode: Literal["mlt_cls", "scalar"]) -> Union[list[list[int]], list[int]]:
    """Encode labels into a one-hot vector or leave as is

    Parameters
    ----------
    labels :
        The list of labels to encode
    mode :
        The mode to use for encoding. If ``mlt_cls``, then the labels will be encoded into a one-hot vector. If ``scalar``, then the labels will be left as is

    Returns
    -------
        The encoded labels
    """
    match mode:
        case "mlt_cls":
            unq_labels = sorted(list(set(labels)))
            label_dim = len(unq_labels)
            new_labels = []
            for j in labels:
                inner = []
                for i in range(label_dim):
                    if i == j:
                        x = 0
                    else:
                        x = 1
                    inner.append(x)
                new_labels.append(inner)
            return new_labels
        case "scalar":
            return labels


def get_tok_dict(
    data: Union[pd.DataFrame, dict[str, Any]], n_gram: int = 3, include_metadata: bool = False
) -> dict[str, int]:
    """Get a token dictionary from a dataset

    Parameters
    ----------
    data :
        The dataset to get the token dictionary from. Gets unique characters from ``data["Kinase Sequence"]`` and ``data["Site Sequence"]``
    n_gram :
        The n-gram size to use, by default 3
    include_metadata :
        Whether to include the metadata tokens ``tok_dict["<PADDING>"]`` and ``tok_dict["<N-GRAM>"]`` in the token dictionary, by default `False`

    Returns
    -------
    dict[str, int]
        The token dictionary, mapping string tokens to their indices
    """
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
    logger.info("Token frequency (top 50):", tok_occurrence.most_common(50))
    logger.debug("Total iterations: ", all_iters)
    logger.info(f"Total unique {n_gram}-grams: ", len(tok_dict))

    if include_metadata:
        tok_dict["<PADDING>"] = len(tok_dict)
        tok_dict["<N-GRAM>"] = n_gram

    return dict(tok_dict)


def encode_seq(seq: str, mapping_dict: dict[str, int]) -> list[int]:
    """Encode a sequence into a list of integers

    Parameters
    ----------
    seq :
        The sequence to encode
    mapping_dict :
        The mapping dictionary to use, which maps tokens to their indices

    Raises
    ------
    AssertionError
        If "<N-GRAM>" is not in ``mapping_dict`` or if ``mapping_dict["<N-GRAM>"]`` is not an integer
    AssertionError
        If a provided token is not in ``mapping_dict``

    Returns
    -------
    list[int]
        The encoded sequence
    """
    res = []
    assert "<N-GRAM>" in mapping_dict
    assert isinstance(mapping_dict["<N-GRAM>"], int)
    n_gram: int = mapping_dict["<N-GRAM>"]
    for i in range(len(seq) - n_gram + 1):
        assert seq[i : i + n_gram] in mapping_dict
        res.append(mapping_dict[seq[i : i + n_gram]])
    return res


def package(
    X_site: torch.Tensor,
    X_kin: torch.Tensor,
    y: torch.Tensor,
    data: pd.DataFrame | dict[str, Any],
    ids: list[int],
    batch_size: int,
    group_by: Literal["kin", "site"],
    num_chunks: int,
    max_kin_length: int,
    remapping_class_label_dict_inv: dict | None,
    class_labels: list[str | int],
    tok_dict: dict[str, int],
    chunk_size: int | None,
    chunk_position: int = 0,
) -> tuple[torch.utils.data.DataLoader, dict[str, Any]]:
    """Take the data tensors and package them into `torch.utils.data.Dataset` objects, and then `torch.utils.data.DataLoader` objects.

    Parameters
    ----------
    X_site :
        The site sequence data, in tensor form
    X_kin :
        The kinase sequence data, in tensor form
    y :
        The labels, in tensor form
    data :
        The original data passed to `data_to_tensor`
    ids :
        The indices of the data to use
    batch_size :
        The batch size to use
    group_by : optional
        Whether or not the groups in the group classifier are classifying by site or kinase, by default "site"
    num_chunks :
        The number of partitions to be used
    max_kin_length :
        The maximum kinase sequence length
    remapping_class_label_dict_inv :
        The remapping dictionary to use, that maps class integers back to their original labels. If `None`, then no remapping will be done
    class_labels :
        The class labels that are used
    tok_dict :
        The token dictionary that was used
    chunk_size : optional
        The actual size of the current chunk. If `None`, then the chunk size will be the size of the entire dataset.
    chunk_position : optional
        Position (index) of the desired chunk within all the data, by default 0.

    Returns
    -------
        dataloader, metadata dictionary
    """
    loader = torch.utils.data.DataLoader(model_utils.KSDataset(X_site, X_kin, y), batch_size=batch_size)
    if group_by == "kin":
        other_group = "Site Sequence"
    else:
        other_group = "Kinase Sequence"

    if isinstance(data, pd.DataFrame):
        if "pair_id" in data.columns:
            pair_ids = data.loc[ids]["pair_id"].to_list()
        else:
            raise AssertionError()
        ret_info_dict = {"PairIDs": pair_ids}
    else:
        if chunk_size is None:
            chunk_size = len(data[other_group])

        ret_info_dict = {
            "PairIDs": [data["pair_id"][i] for i in ids],
        }

    update_dict = {
        "classes": len(class_labels),
        "class_labels": class_labels,
        "remapping_class_label_dict_inv": remapping_class_label_dict_inv,
        "maxsize": max_kin_length,
        "tok_dict": tok_dict,
        "on_chunk": chunk_position,
        "total_chunks": num_chunks,
    }
    ret_info_dict.update(update_dict)

    assert loader is not None
    return loader, ret_info_dict


def data_to_tensor(
    input_data: Union[str, pd.DataFrame, dict[str, Union[list[str], list[int]]]],
    batch_size: int = 256,
    mc=False,
    maxsize=None,
    tokdict=None,
    subsample_num=None,
    n_gram: int = 1,
    device: torch.device = torch.device("cpu"),
    cartesian_product: bool = False,
    group_by: Literal["site", "kin"] = "site",
    kin_seq_to_group: dict = {},
    bytes_per_input: float = 2e6,
    bytes_constant: float = 100e6,
) -> Generator[tuple[torch.utils.data.DataLoader, dict[str, Any]], None, None]:
    """Takes an input dataframe of kinase/substrate data and obtains a tensor representation of it.

    Parameters
    ----------
    input_data :
        Input dataframe of kinase/substrate data. If a string, it is assumed to be a path to a CSV file.
    batch_size : optional
        Batch size for dataloader(s), by default 256.
    mc : optional
        Whether or not to produce "multi_classification" output shape (i.e., labels are n_classes dimentional rather than 0-dimensional), by default False
    maxsize : optional
        The maximum size of an input kinase, by default None. If None, the maximum size is the length of the longest kinase in the dataset.
    tokdict : optional
        A specific token dictionary to use, by default None. If None, a new token dictionary will be generated from the data.
    subsample_num : optional
        The number of datapoints in ``input_data``, to subsample, by default None. If None, no subsampling will be done.
    n_gram : optional
        The n-gram length, by default 3
    device : optional
        The device with which to use for the tensors, by default torch.device("cpu")
    cartesian_product : optional
        Whether or not the kinases and sites in ``input_data`` should have cartesian product performed on them, by default False. For example, if True and ``input_data`` has 2 kinases and 3 sites, the output will have 6 kinases and 6 sites. If false, the input data should have the same number of kinases and sites; this will be the number of kinases/sites in the output.
    group_by : optional
        Whether or not the groups in the group classifier are classifying by site or kinase, by default "site"
    kin_seq_to_group : optional
        _description_, by default {}
    bytes_per_input : optional
        The number of bytes per input, by default 1_400_000
    bytes_per_input_multiplier : optional
        The multiplier for the number of bytes per input, by default 1. Used to increase the number of bytes per input, which is a "safety" measure to prevent OOM errors.

    Yields
    ------
        A generator of tuples of dataloaders and metadata dictionaries.
    """

    logger.vstatus("(Re)loading Tensors into Device for Next Chunk...")

    data: Union[pd.DataFrame, dict[str, Union[list[str], list[int]]]]
    if isinstance(input_data, str):
        data = pd.read_csv(input_data)
    else:
        data = input_data
    if len(data) == 0:
        raise ValueError("Input data is empty")
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
        tok_dict = get_tok_dict(data, n_gram=n_gram, include_metadata=False)
        tok_dict["<PADDING>"] = len(tok_dict)
        tok_dict["<N-GRAM>"] = n_gram
    else:
        tok_dict = tokdict

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

    if kin_seq_to_group:
        class_col = "Group"
    else:
        if mc:
            class_col = "Kinase Sequence"
        else:
            class_col = "Class"

    class_labels = sorted(list(set(data[class_col])))
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
    if not cartesian_product:
        data = pd.DataFrame(data).reset_index(drop=True)
        possible_ids = list(range(len(data)))
    else:
        possible_ids = list(range(len(data["Kinase Sequence"]) * len(data["Site Sequence"])))

    ids = possible_ids.copy()
    random.shuffle(ids)

    kinase_seq_to_tensor_data: dict[str, torch.IntTensor] = {}
    site_seq_to_tensor_data: dict[str, torch.IntTensor] = {}

    if isinstance(data, pd.DataFrame):
        data_shuffled = data.loc[ids]
    else:
        data_shuffled = None
        assert all([isinstance(x, str) for x in data["Kinase Sequence"]]), "All kinase seqs must be strings"
        assert all([isinstance(x, str) for x in data["Site Sequence"]]), "All site seqs must be strings"
        kinase_seq_to_tensor_data = {
            str(ks): torch.IntTensor(pad(encode_seq(str(ks), tok_dict), max_len=max_kin_length, map_dict=tok_dict))
            for ks in data["Kinase Sequence"]
        }
        site_seq_to_tensor_data = {
            str(ss): torch.IntTensor(encode_seq(str(ss), tok_dict)) for ss in data["Site Sequence"]
        }

    assert all(isinstance(x, str) for x in data["Kinase Sequence"]), "Kinase names must be strings"
    assert all(isinstance(x, str) for x in data["Site Sequence"]), "Site sequences must be strings"

    free_ram_and_swap_B = (
        psutil.virtual_memory().available + psutil.swap_memory().free
    ) - bytes_constant  # amount of CPU memory available
    if "cuda" in str(device):
        free_GPU_mem = torch.cuda.mem_get_info(device)[0] - bytes_constant
        # Make sure gpu can fit whole batch
        max_batch_len_GPU = free_GPU_mem // (bytes_per_input)
        try:
            assert max_batch_len_GPU >= batch_size, (
                "Cannot store one batch in GPU memory at a time. Please choose a smaller batch size, as splitting"
                " batches into chunks is not yet implemented."
            )
        except Exception as e:
            raise NotImplementedError(str(e)) from None

    num_batches_can_be_stored_per_dl_CPU = int(free_ram_and_swap_B / (bytes_per_input * batch_size))

    assert (
        len(data["Kinase Sequence"]) == len(data["Site Sequence"]) or cartesian_product
    ), "Length of kinase and site lists must be equal."

    # Make sure cpu can fit whole batch
    try:
        assert num_batches_can_be_stored_per_dl_CPU > 0, (
            "Cannot store one batch in CPU memory at a time. Please choose a smaller batch size, as splitting batches"
            " into chunks is not yet implemented."
        )
    except Exception as e:
        raise NotImplementedError(str(e)) from None

    num_batches = int(np.ceil(len(ids) / batch_size))
    num_chunks = int(np.ceil(num_batches / num_batches_can_be_stored_per_dl_CPU))
    assert num_chunks > 0, "num_chunks <= 0. Something went wrong."

    if cartesian_product:
        num_inputs = len(data["Kinase Sequence"]) * len(data["Site Sequence"])
    else:
        num_inputs = len(data["Kinase Sequence"])

    chunk_size = min(num_batches_can_be_stored_per_dl_CPU * batch_size, num_inputs)
    if chunk_size >= batch_size:
        chunk_size -= chunk_size % batch_size

    assert (
        chunk_size % batch_size == 0 or num_inputs < num_batches_can_be_stored_per_dl_CPU * batch_size
    ), "The chunk size is not a multiple of the batch size."

    for partition_id in range(int(num_chunks)):
        begin_idx = chunk_size * partition_id
        end_idx = min(chunk_size * (partition_id + 1), num_inputs)
        actual_chunk_size = end_idx - begin_idx
        if isinstance(data_shuffled, pd.DataFrame):
            X_site = torch.IntTensor(
                [encode_seq(x, tok_dict) for x in data_shuffled["Site Sequence"].values[begin_idx:end_idx]]
            )
            # X_site = X_site.to(device)
            X_kin = torch.IntTensor(
                [
                    pad(encode_seq(x, tok_dict), max_kin_length, tok_dict)
                    for x in data_shuffled["Kinase Sequence"].values[begin_idx:end_idx]
                ]
            )
            # X_kin = X_kin.to(device)
            if mc:
                encoding = "mlt_cls"
            else:
                encoding = "scalar"
            y = torch.IntTensor(encode_label(data_shuffled[class_col].values[begin_idx:end_idx].tolist(), encoding))
            # y = y.to(device)

        else:  # data is a dict
            rand_idx_to_kin_idx_site_idx = lambda rand_idx: (
                int(rand_idx // len(data["Site Sequence"])),
                int(rand_idx // len(data["Kinase Sequence"])),
            )
            kin_tensor_data: list[torch.Tensor] = []
            site_tensor_data: list[torch.Tensor] = []
            class_tensor_data: list[torch.Tensor] = []
            for id in ids:
                kin_tensor_idx, site_tensor_idx = rand_idx_to_kin_idx_site_idx(id)

                kin_tensor, site_tensor = (
                    kinase_seq_to_tensor_data[str(data["Kinase Sequence"][kin_tensor_idx])],
                    site_seq_to_tensor_data[str(data["Site Sequence"][site_tensor_idx])],
                )
                kin_tensor_data.append(kin_tensor)
                site_tensor_data.append(site_tensor)
                class_tensor_data.append(torch.IntTensor([-1]))  # CHECK -- May not be right for non-predict mode.

            blank = torch.IntTensor()
            if len(kin_tensor_data) == 0:
                X_kin = blank
            else:
                X_kin = torch.stack(kin_tensor_data)  # .to(device)
            if len(site_tensor_data) == 0:
                X_site = blank
            else:
                X_site = torch.stack(site_tensor_data)  # .to(device)
            if len(class_tensor_data) == 0:
                y = blank
            else:
                y = torch.cat(class_tensor_data)  # .to(device)

        yield package(
            X_site=X_site,
            X_kin=X_kin,
            y=y,
            data=data,
            ids=ids,
            batch_size=batch_size,
            group_by=group_by,
            num_chunks=num_chunks,
            max_kin_length=max_kin_length,
            remapping_class_label_dict_inv=remapping_class_label_dict_inv,
            class_labels=class_labels,
            tok_dict=tok_dict,
            chunk_size=actual_chunk_size,
            chunk_position=partition_id,
        )


def pad(tok_list: list[int], max_len: int, map_dict: dict[str, int]) -> list[int]:
    """Pads a list of tokens to a given length.

    Parameters
    ----------
    tok_list :
        The list of tokens to pad.
    max_len :
        The length to pad to.
    map_dict :
        The dictionary mapping tokens to integers. In particular, the padding token should be mapped to an integer.

    Raises
    ------
    AssertionError
        If the padding token is not in the map_dict.

    Returns
    -------
    list[int]
        The padded list of tokens.
    """
    assert "<PADDING>" in map_dict, "Padding token not in map_dict."
    return tok_list + [map_dict["<PADDING>"] for _ in range(max_len - len(tok_list))]
