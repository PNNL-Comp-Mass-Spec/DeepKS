"""Functions to convert textual biological sequences into tensors for deep learning models."""

from ast import Tuple
import time
import numpy as np, psutil, torch, pandas as pd, collections, json, tqdm, random, torch.utils.data
from sympy import factorint
from typing import Generator, Hashable, Protocol, Union, Literal, Any
from ..tools import model_utils
from ..ProtTrans.get_T5_remote import GetT5

from ..config.logging import get_logger

logger = get_logger()
"""The logger for this module."""
if __name__ == "__main__":  # pragma: no cover
    logger.status("Loading Modules...")


def pad(tok_list: list[str], max_len: int, map_dict: dict[str, int]) -> list[str]:
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
    pads = [str(map_dict["<PADDING>"]) for _ in range(max_len - len(tok_list))]
    return tok_list + pads


def compute_memory_requirements(
    batch_size: int,
    device: torch.device,
    data: dict,
    bytes_constant: int | None,
    bytes_per_input: int | None,
    ids: list[int],
    cartesian_product: bool,
) -> tuple[int, int, int]:
    """
    Returns
    -------
        num_batches, num_chunks, chunk_size
    """
    num_batches = int(np.ceil(len(ids) / batch_size))
    if bytes_constant is None or bytes_per_input is None:
        num_chunks = 1
        chunk_size = len(ids)
        return num_batches, num_chunks, chunk_size
    max_batch_len_GPU: float = float("inf")
    free_ram_and_swap_B = (
        (psutil.virtual_memory().available + psutil.swap_memory().free) - bytes_constant
    ) * 0.66  # amount of CPU memory available
    if "cuda" in str(device):
        free_GPU_mem = torch.cuda.mem_get_info(device)[0] - bytes_constant
        # Make sure gpu can fit whole batch
        max_batch_len_GPU = free_GPU_mem // (bytes_per_input)
        while max_batch_len_GPU < batch_size:
            factors: list[int] = list(factorint(batch_size).keys())
            max_batch_len_GPU //= min(factors)
            if batch_size == 0:
                raise ValueError("The batch size reduced to zero meaning one input cannot fit into memory.")
            logger.warning(f"Reducing batch size to {batch_size}")

    num_batches_can_be_stored_per_dl_CPU = int(free_ram_and_swap_B / (bytes_per_input * batch_size))

    assert (
        len(data["Kinase Sequence"]) == len(data["Site Sequence"]) or cartesian_product
    ), "Length of kinase and site lists must be equal."

    # Make sure cpu can fit whole batch
    while num_batches_can_be_stored_per_dl_CPU < 1:
        try:
            assert num_batches_can_be_stored_per_dl_CPU >= 1, (
                "Cannot store one batch in CPU memory at a time. Reducing the batch size, but if/when training, I will"
                " still only `.step()` after the originally provided batch size."
            )
        except Exception as e:
            logger.warning(str(e))
            factors: list[int] = list(factorint(batch_size).keys())
            batch_size //= min(factors)
            if batch_size == 0:
                raise ValueError("The batch size reduced to zero meaning one input cannot fit into memory.")
            logger.warning(f"Reducing batch size to {batch_size}")
            num_batches_can_be_stored_per_dl_CPU = int(free_ram_and_swap_B / (bytes_per_input * batch_size))

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
    assert chunk_size == int(chunk_size)
    chunk_size = int(chunk_size)
    return num_batches, num_chunks, chunk_size


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
    data: dict[Hashable, list], n_gram: int = 1, include_metadata: bool = True, alphabetize: bool = True
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
    alphabetize :
        Whether to alphabetize the resultant tok dict like ``{"A": 0, "C": 1, "D": 2`` and so on

    Returns
    -------
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

    iter_through(np.unique(data["Kinase Sequence"]))
    iter_through(np.unique(data["Site Sequence"]))
    assert sum(tok_occurrence.values()) == all_iters
    logger.debug(f"Token frequency (top 50): {tok_occurrence.most_common(50)}")
    logger.debug(f"Total iterations: {all_iters}")
    logger.debug(f"Total unique {n_gram}-grams: {len(tok_dict)}")

    if include_metadata:
        tok_dict["<PADDING>"] = len(tok_dict)
        tok_dict["<N-GRAM>"] = n_gram

    d: dict[str, int] = dict(tok_dict)
    if alphabetize:
        d_keys: list[str] = sorted(list(d.keys()))
        d_vals: list[int] = sorted(list(d.values()))
        d = {k: v for k, v in zip(d_keys, d_vals)}

    return d


def encode_seqs(seqs: list[str], mapping_dict: dict[str, int]) -> torch.IntTensor:
    """Encode a list of string sequences into a list of integers

    Parameters
    ----------
    seqs :
        The list of sequences to encode
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
        The encoded sequence array
    """
    assert isinstance(mapping_dict["<N-GRAM>"], int)
    n_gram: int = mapping_dict["<N-GRAM>"]
    max_len: int = max([len(s) for s in seqs])
    s = time.time()
    seqs_np = np.full((len(seqs), max_len), int(mapping_dict["<PADDING>"]))
    for i in range(len(seqs)):
        seq = seqs[i]
        for j in range(len(seq)):
            the_n_gram = seq[j : j + n_gram]
            seqs_np[i, j] = mapping_dict[the_n_gram]
    e = time.time()
    logger.info(f"{e - s:.3f}")

    tensor_version = torch.IntTensor(seqs_np)
    return tensor_version


def package(
    loader: torch.utils.data.DataLoader,
    data: dict[Hashable, list],
    ids: list[int],
    num_chunks: int,
    max_kin_length: int,
    remapping_class_label_dict_inv: dict | None,
    class_labels: list,
    tok_dict: dict[str, int],
    chunk_position: int = 0,
) -> tuple[torch.utils.data.DataLoader, dict[str, Any]]:
    """Take the data tensors and package them into `torch.utils.data.Dataset` objects, and then `torch.utils.data.DataLoader` objects.

    Parameters
    ----------
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
    ret_info_dict: dict[str, list | dict] = {
        "PairIDs": [data["pair_id"][i] for i in range(len(data["pair_id"]))],
    }

    update_dict = {
        "classes": len(class_labels),
        "class_labels": class_labels,
        "remapping_class_label_dict_inv": remapping_class_label_dict_inv,
        "maxsize": max_kin_length,
        "tok_dict": tok_dict,
        "on_chunk": chunk_position,
        "total_chunks": num_chunks,
        "site": data["Site Sequence"],
        "kin": data["Gene Name of Provided Kin Seq"],
    }
    ret_info_dict.update(update_dict)
    ret_info_dict.update({"original_data": data})
    return loader, ret_info_dict


def get_remapping_dict(data: dict[Hashable, list], class_labels: list) -> tuple[dict | None, dict | None]:
    if set(class_labels) in [{0, 1}, {1}, {0}]:
        remapping_class_label_dict = {0: 0, 1: 1}
        remapping_class_label_dict_inv = {0: "Decoy", 1: "Target"}
        remapping_class_label_dict_inv = remapping_class_label_dict_inv
        assert all(isinstance(x, int) for x in class_labels) or all(
            isinstance(x, str) and x.isdigit() for x in class_labels
        ), f"Class labels must be integers ({class_labels[:10]=})"
        data["Class"] = [remapping_class_label_dict[int(x)] for x in data["Class"]]
    elif set(class_labels) != {-1}:  # I.e., not predict mode
        remapping_class_label_dict = {class_labels[i]: i for i in range(len(class_labels))}
        remapping_class_label_dict_inv = {i: list(class_labels)[i] for i in range(len(class_labels))}
        remapping_class_label_dict_inv = remapping_class_label_dict_inv
        data["Class"] = [remapping_class_label_dict[x] for x in data["Class"]]
    else:
        remapping_class_label_dict, remapping_class_label_dict_inv = None, None
    return remapping_class_label_dict, remapping_class_label_dict_inv


def data_to_tensor(
    input_data: str | pd.DataFrame | dict[Hashable, list],
    batch_size: int = 256,
    maxsize: int | None = None,
    tokdict: dict[str, int] | None = None,
    n_gram: int = 1,
    device: torch.device = torch.device("cpu"),
    cartesian_product: bool = False,
    compute_memory: bool = False,
    shuffling_seed: int = 99354,
    bytes_constant: int | None = None,
    bytes_per_input: int | None = None,
) -> Generator[tuple[torch.utils.data.DataLoader, dict[str, Any]], None, None]:
    """Takes an input dataframe of kinase/substrate data and obtains a tensor representation of it.

    Parameters
    ----------
    input_data :
        Input dataframe of kinase/substrate data. If a string, it is assumed to be a path to a CSV file.
    batch_size : optional
        Batch size for dataloader(s), by default 256.
    maxsize : optional
        The maximum size of an input kinase, by default None. If None, the maximum size is the length of the longest kinase in the dataset.
    tokdict : optional
        A specific token dictionary to use, by default None. If None, a new token dictionary will be generated from the data.
    n_gram : optional
        The n-gram length, by default 3
    device : optional
        The device with which to use for the tensors, by default torch.device("cpu")
    cartesian_product : optional
        Whether or not the kinases and sites in ``input_data`` should have cartesian product performed on them, by default False. For example, if True and ``input_data`` has 2 kinases and 3 sites, the output will have 6 kinases and 6 sites. If false, the input data should have the same number of kinases and sites; this will be the number of kinases/sites in the output.
    compute_memory : optionsl
        Whether or not to compute memory requirements. If ``False``, it is assumed that the machine has infinite memory (a reasonable assumption, usually)

    Yields
    ------
        A generator of tuples of dataloaders and metadata dictionaries.
    """

    logger.vstatus("Converting AA sequence data into Tensors...")

    data: dict[Hashable, list]

    match input_data:
        case str():
            data = pd.read_csv(input_data).to_dict(orient="list")
        case pd.DataFrame():
            data = input_data.to_dict(orient="list")
        case dict():
            data = input_data

    max_kin_length = maxsize
    if max_kin_length is None:
        assert all([isinstance(x, str) for x in data["Kinase Sequence"]]), "All kinase seqs must be strings"
        max_kin_length = max([len(str(x)) for x in data["Kinase Sequence"]]) - n_gram + 1

    tok_dict = tokdict
    if tok_dict is None:
        tok_dict = get_tok_dict(data, n_gram=n_gram, include_metadata=False)
        tok_dict["<PADDING>"] = len(tok_dict)
        tok_dict["<N-GRAM>"] = n_gram

    data = data.copy()

    class_labels = sorted(list(set(data["Class"])))

    _, remapping_class_label_dict_inv = get_remapping_dict(data, class_labels)

    random.seed(shuffling_seed)
    if not cartesian_product:
        possible_ids = list(range(len(data["Kinase Sequence"])))

    else:
        possible_ids = list(range(len(data["Kinase Sequence"]) * len(data["Site Sequence"])))

    _ = len(possible_ids)

    ids = possible_ids.copy()
    random.shuffle(ids)
    if not cartesian_product:
        data = {k: [v[i] for i in ids] for k, v in data.items()}

    if not compute_memory:
        bytes_constant = None
        bytes_per_input = None

    _, num_chunks, _ = compute_memory_requirements(
        batch_size, device, data, bytes_constant, bytes_per_input, ids, cartesian_product
    )

    kin_tensors = encode_seqs(data["Kinase Sequence"], tok_dict)
    site_tensors = encode_seqs(data["Site Sequence"], tok_dict)
    labels = torch.IntTensor(data["Class"])

    dataset: model_utils.KSDataset
    if cartesian_product:
        dataset = model_utils.CartesianKSDataset(site_tensors, kin_tensors, labels)
    else:
        dataset = model_utils.TandemKSDataset(site_tensors, kin_tensors, labels)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    yield package(
        loader,
        data=data,
        ids=ids,
        num_chunks=num_chunks,
        max_kin_length=max_kin_length,
        remapping_class_label_dict_inv=remapping_class_label_dict_inv,
        class_labels=class_labels,
        tok_dict=tok_dict,
        chunk_position=1,  # TODO: for now
    )


# def data_to_tensor_T5(
#     input_data: Union[str, pd.DataFrame, dict[str, Union[list[str], list[int]]]],
#     batch_size: int = 256,
#     mc=False,
#     maxsize=None,
#     tokdict=None,
#     subsample_num=None,
#     device: torch.device = torch.device("cpu"),
#     cartesian_product: bool = False,
#     group_by: Literal["site", "kin"] = "site",
#     kin_seq_to_group: dict = {},
#     bytes_per_input: float = 2e6,
#     bytes_constant: float = 100e6,
# ) -> Generator[tuple[torch.utils.data.DataLoader, dict[str, Any]], None, None]:
#     """Takes an input dataframe of kinase/substrate data and obtains a tensor representation of it.

#     Parameters
#     ----------
#     input_data :
#         Input dataframe of kinase/substrate data. If a string, it is assumed to be a path to a CSV file.
#     batch_size : optional
#         Batch size for dataloader(s), by default 256.
#     mc : optional
#         Whether or not to produce "multi_classification" output shape (i.e., labels are n_classes dimentional rather than 0-dimensional), by default False
#     maxsize : optional
#         The maximum size of an input kinase, by default None. If None, the maximum size is the length of the longest kinase in the dataset.
#     tokdict : optional
#         A specific token dictionary to use, by default None. If None, a new token dictionary will be generated from the data.
#     subsample_num : optional
#         The number of datapoints in ``input_data``, to subsample, by default None. If None, no subsampling will be done.
#     n_gram : optional
#         The n-gram length, by default 3
#     device : optional
#         The device with which to use for the tensors, by default torch.device("cpu")
#     cartesian_product : optional
#         Whether or not the kinases and sites in ``input_data`` should have cartesian product performed on them, by default False. For example, if True and ``input_data`` has 2 kinases and 3 sites, the output will have 6 kinases and 6 sites. If false, the input data should have the same number of kinases and sites; this will be the number of kinases/sites in the output.
#     group_by : optional
#         Whether or not the groups in the group classifier are classifying by site or kinase, by default "site"
#     kin_seq_to_group : optional
#         _description_, by default {}
#     bytes_per_input : optional
#         The number of bytes per input, by default 1_400_000
#     bytes_per_input_multiplier : optional
#         The multiplier for the number of bytes per input, by default 1. Used to increase the number of bytes per input, which is a "safety" measure to prevent OOM errors.

#     Yields
#     ------
#         A generator of tuples of dataloaders and metadata dictionaries.
#     """

#     logger.vstatus("(Re)loading Tensors into Device for Next Chunk...")

#     data: Union[pd.DataFrame, dict[str, Union[list[str], list[int]]]]
#     if isinstance(input_data, str):
#         data = pd.read_csv(input_data)
#     else:
#         data = input_data
#     if len(data) == 0:
#         raise ValueError("Input data is empty")
#     if subsample_num is not None and isinstance(data, pd.DataFrame):  # TODO Could improve this
#         try:
#             subsample_num = int(subsample_num)
#             data = data.sample(n=subsample_num)
#         except Exception as e:
#             print("Not subsampling (error: {})".format(e), flush=True)

#     if kin_seq_to_group:
#         data["Group"] = [kin_seq_to_group[x] for x in data["Kinase Sequence"]]
#         if isinstance(data, pd.DataFrame):
#             data = data[data["Group"] != "<UNANNOTATED>"]
#         else:
#             data_DFed = pd.DataFrame(data)
#             data_DFed = data_DFed[data_DFed["Group"] != "<UNANNOTATED>"]
#             data_intermediary = data_DFed.to_dict("list")
#             data_intermediary = {str(k): v for k, v in data_intermediary.items()}
#             data = data_intermediary

#     data = data.copy()

#     if kin_seq_to_group:
#         class_col = "Group"
#     else:
#         if mc:
#             class_col = "Kinase Sequence"
#         else:
#             class_col = "Class"

#     class_labels = sorted(list(set(data[class_col])))
#     if set(class_labels) in [{0, 1}, {1}, {0}]:
#         remapping_class_label_dict = {0: 0, 1: 1}
#         remapping_class_label_dict_inv = {0: "Decoy", 1: "Real"}
#         remapping_class_label_dict_inv = remapping_class_label_dict_inv
#         assert all(isinstance(x, int) for x in class_labels) or all(
#             isinstance(x, str) and x.isdigit() for x in class_labels
#         ), f"Class labels must be integers ({class_labels[:10]=})"
#         data[class_col] = [remapping_class_label_dict[int(x)] for x in data[class_col]]
#     elif set(class_labels) != {-1}:  # I.e., not predict mode
#         remapping_class_label_dict = {class_labels[i]: i for i in range(len(class_labels))}
#         remapping_class_label_dict_inv = {i: list(class_labels)[i] for i in range(len(class_labels))}
#         remapping_class_label_dict_inv = remapping_class_label_dict_inv
#         data[class_col] = [remapping_class_label_dict[x] for x in data[class_col]]
#     else:
#         remapping_class_label_dict = None
#         remapping_class_label_dict_inv = None

#     random.seed(99354)
#     if not cartesian_product:
#         data = pd.DataFrame(data).reset_index(drop=True)
#         possible_ids = list(range(len(data)))
#     else:
#         possible_ids = list(range(len(data["Kinase Sequence"]) * len(data["Site Sequence"])))

#     ids = possible_ids.copy()
#     random.shuffle(ids)

#     kinase_seq_to_tensor_data: dict[str, torch.Tensor] = {}
#     site_seq_to_tensor_data: dict[str, torch.Tensor] = {}
#     getT5 = GetT5()

#     if isinstance(data, pd.DataFrame):
#         data_shuffled = data.loc[ids]
#     else:
#         data_shuffled = None
#         assert all([isinstance(x, str) for x in data["Kinase Sequence"]]), "All kinase seqs must be strings"
#         assert all([isinstance(x, str) for x in data["Site Sequence"]]), "All site seqs must be strings"
#         # kinase_seq_to_tensor_data = {
#         #     str(ks): getT5.get_t5([str(ks)], device=device)
#         #     for ks in data["Kinase Sequence"]
#         # }
#         kin_seqs = [str(ks) for ks in data["Kinase Sequence"]]
#         site_seqs = [str(ss) for ss in data["Site Sequence"]]
#         kin_t5_embs = getT5.get_t5(kin_seqs, device=device)
#         site_t5_embs = getT5.get_t5(site_seqs, device=device)
#         kinase_seq_to_tensor_data = dict(zip(kin_seqs, kin_t5_embs))
#         site_seq_to_tensor_data = dict(zip(site_seqs, site_t5_embs))

#         site_seq_to_tensor_data = {str(ss): getT5.get_t5([str(ss)], device=device) for ss in data["Site Sequence"]}

#     assert all(isinstance(x, str) for x in data["Kinase Sequence"]), "Kinase names must be strings"
#     assert all(isinstance(x, str) for x in data["Site Sequence"]), "Site sequences must be strings"

#     max_batch_len_GPU = float("inf")
#     free_ram_and_swap_B = (
#         (psutil.virtual_memory().available + psutil.swap_memory().free) - bytes_constant
#     ) * 0.66  # amount of CPU memory available
#     if "cuda" in str(device):
#         free_GPU_mem = torch.cuda.mem_get_info(device)[0] - bytes_constant
#         # Make sure gpu can fit whole batch
#         max_batch_len_GPU = free_GPU_mem // (bytes_per_input)
#         while max_batch_len_GPU < batch_size:
#             max_batch_len_GPU //= min(list(factorint(batch_size).keys()))
#             if batch_size == 0:
#                 raise ValueError("The batch size reduced to zero meaning one input cannot fit into memory.")
#             logger.warning(f"Reducing batch size to {batch_size}")

#     num_batches_can_be_stored_per_dl_CPU = int(free_ram_and_swap_B / (bytes_per_input * batch_size))

#     assert (
#         len(data["Kinase Sequence"]) == len(data["Site Sequence"]) or cartesian_product
#     ), "Length of kinase and site lists must be equal."

#     # Make sure cpu can fit whole batch
#     while num_batches_can_be_stored_per_dl_CPU < 1:
#         try:
#             assert num_batches_can_be_stored_per_dl_CPU >= 1, (
#                 "Cannot store one batch in CPU memory at a time. Reducing the batch size, but if/when training, I will"
#                 " still only `.step()` after the originally provided batch size."
#             )
#         except Exception as e:
#             logger.warning(str(e))
#             batch_size //= min(list(factorint(batch_size).keys()))
#             if batch_size == 0:
#                 raise ValueError("The batch size reduced to zero meaning one input cannot fit into memory.")
#             logger.warning(f"Reducing batch size to {batch_size}")
#             num_batches_can_be_stored_per_dl_CPU = int(free_ram_and_swap_B / (bytes_per_input * batch_size))

#     num_batches = int(np.ceil(len(ids) / batch_size))
#     num_chunks = int(np.ceil(num_batches / num_batches_can_be_stored_per_dl_CPU))
#     assert num_chunks > 0, "num_chunks <= 0. Something went wrong."

#     if cartesian_product:
#         num_inputs = len(data["Kinase Sequence"]) * len(data["Site Sequence"])
#     else:
#         num_inputs = len(data["Kinase Sequence"])

#     chunk_size = min(num_batches_can_be_stored_per_dl_CPU * batch_size, num_inputs)
#     if chunk_size >= batch_size:
#         chunk_size -= chunk_size % batch_size

#     assert (
#         chunk_size % batch_size == 0 or num_inputs < num_batches_can_be_stored_per_dl_CPU * batch_size
#     ), "The chunk size is not a multiple of the batch size."
#     assert chunk_size == int(chunk_size)
#     chunk_size = int(chunk_size)
#     for partition_id in range(int(num_chunks)):
#         begin_idx = chunk_size * partition_id
#         end_idx = min(chunk_size * (partition_id + 1), num_inputs)
#         actual_chunk_size = end_idx - begin_idx
#         if isinstance(data_shuffled, pd.DataFrame):
#             kin_seqs = [str(ks) for ks in data_shuffled["Kinase Sequence"].values[begin_idx:end_idx]]
#             site_seqs = [str(ss) for ss in data_shuffled["Site Sequence"].values[begin_idx:end_idx]]
#             X_kin = getT5.get_t5(kin_seqs, device=device)
#             X_site = getT5.get_t5(site_seqs, device=device)

#             if mc:
#                 encoding = "mlt_cls"
#             else:
#                 encoding = "scalar"
#             y = torch.IntTensor(encode_label(data_shuffled[class_col].values[begin_idx:end_idx].tolist(), encoding))
#             # y = y.to(device)

#         else:  # data is a dict
#             rand_idx_to_kin_idx_site_idx = lambda rand_idx: (
#                 int(rand_idx // len(data["Site Sequence"])),
#                 int(rand_idx // len(data["Kinase Sequence"])),
#             )
#             kin_tensor_data: list[torch.Tensor] = []
#             site_tensor_data: list[torch.Tensor] = []
#             class_tensor_data: list[torch.Tensor] = []
#             for id in ids:
#                 kin_tensor_idx, site_tensor_idx = rand_idx_to_kin_idx_site_idx(id)

#                 kin_tensor, site_tensor = (
#                     kinase_seq_to_tensor_data[str(data["Kinase Sequence"][kin_tensor_idx])],
#                     site_seq_to_tensor_data[str(data["Site Sequence"][site_tensor_idx])],
#                 )
#                 kin_tensor_data.append(kin_tensor)
#                 site_tensor_data.append(site_tensor)
#                 class_tensor_data.append(torch.IntTensor([-1]))  # CHECK -- May not be right for non-predict mode.

#             blank = torch.IntTensor()
#             if len(kin_tensor_data) == 0:
#                 X_kin = blank
#             else:
#                 X_kin = torch.stack(kin_tensor_data)  # .to(device)
#             if len(site_tensor_data) == 0:
#                 X_site = blank
#             else:
#                 X_site = torch.stack(site_tensor_data)  # .to(device)
#             if len(class_tensor_data) == 0:
#                 y = blank
#             else:
#                 y = torch.cat(class_tensor_data)  # .to(device)

#         yield package(
#             X_site=X_site,
#             X_kin=X_kin,
#             y=y,
#             data=data,
#             ids=ids,
#             batch_size=batch_size,
#             group_by=group_by,
#             num_chunks=num_chunks,
#             max_kin_length=-1,
#             remapping_class_label_dict_inv=remapping_class_label_dict_inv,
#             class_labels=class_labels,
#             tok_dict={},
#             chunk_size=actual_chunk_size,
#             chunk_position=partition_id,
#         )
