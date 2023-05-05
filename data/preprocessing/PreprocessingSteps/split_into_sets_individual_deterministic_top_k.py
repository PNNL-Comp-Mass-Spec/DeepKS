"""Use a hill climbing algorithm to split the data into train/val/test sets."""

import collections, re, sys, mlrose_hiive as mlr, pandas as pd, numpy as np, json
from types import NoneType
from tqdm import tqdm
from typing import Iterable, Union
from numpy.ma import core as np_ma_core

from ....config.logging import get_logger
from ....config.join_first import join_first
from ....tools.custom_tqdm import CustomTqdm

logger = get_logger()
"""The logger for this script."""

# newstream = open("debug_non_det.log", "a")
np.set_printoptions(linewidth=240)

def get_kin_assignments_from_state(
    states: dict[str, list[list[int]]], which_seed: int, fam_to_kin: dict[str, list[int]]
) -> tuple[list[str], list[str], list[str]]:
    """Get the actual kinase gene name from state vector

    Parameters
    ----------
    states : dict[str, list[list[int]]]
        Dictionary mapping kinase groups to lists of lists that are the states for several restarts
    which_seed :
        Which seed's state should we extract from the list of states?
    fam_to_kin :
        A dictionary mapping a kinase family to a list of its members

    Returns
    -------
        Three lists of kinase gene names, one for each of train, val, and test
    """
    states2 = {k: v[min(which_seed, len(v))] for k, v in states.items()}
    results = [[] for _ in range(3)]
    for group in states2:
        for i in range(3):
            fams_inner = [grp_to_fam[group][j] for j in range(len(states2[group])) if states2[group][j] == i]
            for f in fams_inner:
                results[i] += fam_to_kin[f]
    for r in results:
        r.sort(key=lambda x: re.sub(r"[\(\)\*]", "", x))
    return tuple(results)


def core_ai(tgt: float, random_state: int, group: str) -> tuple[list[int], float]:
    """Run the core components of the hill climbing algorithm

    Parameters
    ----------
    tgt :
        The target proportion of the PSP data dedicated to val and test sets (split evenly)
    random_state :
        The random state to use in `mlr.simulated_annealing`
    group :
        The group for which we are trying to optimize

    Returns
    -------
        The best state list, the best fitness score achieved
    """
    num_fams = len(grp_to_fam[group])
    fit = mlr.CustomFitness(
        hill_climbing_score, problem_type="discrete", targets=[1 - tgt, tgt / 2, tgt / 2], group=group
    )
    opt = mlr.DiscreteOpt(num_fams, fit, maximize=False, max_val=3)
    best_state, best_fitness, _ = mlr.simulated_annealing(
        opt, max_attempts=1000, max_iters=1000, random_state=random_state, init_state=[0 for _ in range(num_fams)]
    )
    best_state = best_state.tolist()
    assert isinstance(best_state, list)
    assert isinstance(best_fitness, float)
    return best_state, best_fitness


def run_restarts(
    tgt: float, group: str, num_restarts: int = 100, top_k: int = 5
) -> tuple[np.ma.MaskedArray, np.ma.MaskedArray, float, list[list[int]]]:
    """Run the hill climbing algorithm for a given group, for a specified number of restarts

    Parameters
    ----------
    tgt :
        The target proportion of the PSP data dedicated to val and test sets (split evenly)
    group :
        The group for which we are trying to optimize
    num_restarts : optional
        The number of restarts to complete, by default 100
    top_k : optional
        Return data about the ``top_k`` fitness scores, by default 5

    Returns
    -------
        Masked array of top k indices (masked where there are fewer than k unique scores),\
        masked array of top k corresponding scores, standard deviation of the corresponding scores,\
        list of the corresponding states, themselves lists of integers.
    """
    scores = []
    states = []
    for r in CustomTqdm(range(1, num_restarts + 1), position=0, leave=True, desc="Group Restart Progress"):
        present = core_ai(tgt, r, group)
        states.append(present[0])
        scores.append(present[1])
    argsorted = np.argsort(scores, kind="stable")
    scores = np.array(sorted(scores))
    unq_top_k_ids = get_inds_of_unq_top_k(scores, top_k)
    ma = ma_from_inds(scores, unq_top_k_ids)  # ma = min(scores)
    mai = np.ma.MaskedArray(
        [argsorted[i] if type(i) == np.int64 else None for i in unq_top_k_ids],
        mask=[False if type(i) == np.int64 else True for i in unq_top_k_ids],
    )  # mai = scores.index(ma)
    sd: float = float(np.std(scores))
    state = [states[argsorted[i]] for i in range(len(argsorted))]
    logger.info(
        "Best scores for group {} were {} achieved by random states {} and the restart scores had an SD of"
        " {:3.2f}".format(group, ma, mai, sd)
    )
    return mai, ma, sd, state


def run_groups(tgt: float, grp_to_num_sites: dict[str, int], num_restarts: int = 100) -> dict[str, list[list[int]]]:
    """Run the hill climbing algorithm for a given group, for a specified number of restarts

    Parameters
    ----------
    tgt : 
        The target proportion of the PSP data dedicated to val and test sets (split evenly)
    grp_to_num_sites : 
        A dictionary mapping a kinase group to the number of sites it has in PSP
    num_restarts : int, optional
        The number of hill climbing restarts to complete, by default 100

    Returns
    -------
        A dictionary mapping a kinase group to a list of its top k states, where k is the number of unique scores. \
            Each state is a list of integers.
    """
    group_scores = []
    group_r = []
    group_sd = []
    group_states = {}
    for _, group in enumerate(tq := tqdm(sorted(list(grp_to_num_sites.keys())), position=0, leave=True, colour="cyan")):
        tq.set_description(f"Processing Group {group}")
        group_result = run_restarts(tgt, group, num_restarts=num_restarts)
        group_r.append(group_result[0])
        group_scores.append(group_result[1])
        group_sd.append(group_result[2])
        group_states[group] = group_result[3]

    # print(f"Overall Top k Scores = {[sum([group_scores[j][i] for j in range(len(group_scores))]) for i in range(len(group_scores[0]))]}")
    return group_states


def ma_from_inds(base_ar: np.ndarray, inds: Iterable[int]) -> np.ma.MaskedArray:
    """Create a masked array from a base array and a list of indices.

    Parameters
    ----------
    base_ar :
        The array with which to create a masked array at the specified indices.
    inds :
        The indices to mask in the original ``base_ar``.

    Returns
    -------
        The resultant masked array
    """
    initial = np.ma.MaskedArray([base_ar[i] if not np.ma.is_masked(i) else np_ma_core.MaskedConstant() for i in inds])
    if isinstance(base_ar, np.ndarray):
        post_processing = lambda x: x.astype(base_ar.dtype) 
    else:
        post_processing = x
    return post_processing(initial)


def get_inds_of_unq_top_k(ls: np.ndarray | list, top_k: int, verify_sorted: bool = True) -> np.ma.MaskedArray:
    """Get the indices of the top k unique elements in a list.

    Parameters
    ----------
    ls :
        The input list/array from which to get the top k unique elements.
    top_k :
        The top :math:`k` unique elements to get.
    verify_sorted : optional
        Whether or not to verify that the input list ``ls`` is sorted, by default True. \
            (If it is not sorted, the function will not work correctly.)

    Returns
    -------
        The top :math:`k` unique elements in the list, with the final :math:`\\max\\{0,~k - m\\}` elements being \
            masked if there are :math:`m` unique elements in ``ls``.
    """
    assert isinstance(ls, np.ndarray) or isinstance(ls, list), "Input is not a numpy array/list!"
    assert isinstance(top_k, int), "top_k is not an integer!"
    if verify_sorted:
        assert all(np.equal(sorted(ls), ls)), "List is not sorted!"
    inds = []
    if len(ls) == 0:
        return np.ma.MaskedArray(data=[0 for _ in range(top_k)], mask=[True for _ in range(top_k)])
    inds.append(0)
    i = 1
    i_last = 0
    while i < len(ls) and len(inds) < top_k:
        if ls[i] != ls[i_last]:
            inds.append(i)
            i_last = i
        i += 1
    return np.ma.MaskedArray(
        inds + [0 for _ in range(top_k - len(inds))],
        mask=[False for _ in range(len(inds))] + [True for _ in range(top_k - len(inds))],
    )


def split_into_sets(
    kin_fam_grp_file: str,
    raw_input_file: str,
    tgt: float | int = 0.3,
    get_restart: bool | None = True,
    num_restarts: int = 25,
) -> None:
    """Uses a hill climbing algorithm to split the data into train/val/test sets.

    Parameters
    ----------
    kin_fam_grp_file :
        A file containing the kinase-family-group mapping.
    raw_input_file :
        A file containing raw PSP data
    tgt : optional
        The target proportion of the PSP data dedicated to val and test sets (split evenly), by default 0.3
    get_restart : optional
        Whether or not to determine the most optimal hill-climbing restart, by default True
    num_restarts : optional
        If ``get_restarts == True``, the number of restarts to complete, by default 25
    """
    del_decor = lambda x: re.sub(r"[\(\)\*]", "", x)
    global fam_to_grp, kin_to_fam, kin_to_num_seq, grp_to_fam, fam_to_num_seq, fams
    kin_fam_grp = pd.read_csv(kin_fam_grp_file).applymap(del_decor)
    grp_to_fam = collections.defaultdict(list)
    for _, r in kin_fam_grp[["Family", "Group"]].drop_duplicates(keep="first").iterrows():
        grp_to_fam[r["Group"].upper()].append(r["Family"].upper())

    fam_to_kin = collections.defaultdict(list)  # {f: k for f, k in zip(kin_fam_grp['Family'], kin_fam_grp['Kinase'])}
    for _, r in kin_fam_grp.iterrows():
        fam_to_kin[r["Family"].upper()].append(f"{r['Kinase'].upper()}|{r['Uniprot']}")
        # print(sum([len(x) for x in fam_to_kin.values()]), _)

    fam_to_kin = collections.OrderedDict(sorted(fam_to_kin.items(), key=lambda x: x[0]))
    grp_to_fam = collections.OrderedDict(sorted(grp_to_fam.items(), key=lambda x: x[0]))

    accumulator = 0
    kin_to_fam = {}
    for f in fam_to_kin:
        for k in fam_to_kin[f]:
            if k in kin_to_fam:
                logger.warning(f"Kin {k} already in!")
            kin_to_fam[k] = f

    fam_to_grp = {}
    for g in grp_to_fam:
        for f in grp_to_fam[g]:
            fam_to_grp[f] = g

    for f in fam_to_kin:
        fam_to_kin[f].sort()

    raw = (
        pd.read_csv(raw_input_file)[["lab", "num_sites", "uniprot_id"]]
        .rename(columns={"lab": "Kinase", "uniprot_id": "Uniprot"})
        .drop_duplicates()
        .applymap(lambda x: x.upper() if isinstance(x, str) else x)
    )
    num_sites_join = pd.merge(kin_fam_grp, raw, how="left", on="Uniprot")
    assert not any(num_sites_join["num_sites"].isna())
    assert num_sites_join["Kinase_x"].equals(num_sites_join["Kinase_y"])
    num_sites_join = num_sites_join.drop(axis="columns", labels=["Kinase_y"]).rename(columns={"Kinase_x": "Kinase"})
    kin_to_num_seq = {
        f"{l}|{u}": n
        for l, u, n in zip(num_sites_join["Kinase"], num_sites_join["Uniprot"], num_sites_join["num_sites"])
    }
    fam_to_num_seq = collections.defaultdict(int)
    kin_to_num_seq_new = {}
    for k in kin_to_num_seq:
        k_prime = k
        if k not in kin_to_fam:
            if f"({k.split('|')[0]})|{k.split('|')[1]}" in kin_to_fam:
                k_prime = f"({k.split('|')[0]})|{k.split('|')[1]}"
            elif f"(({k.split('|')[0]}))|{k.split('|')[1]}" in kin_to_fam:
                k_prime = f"(({k.split('|')[0]}))|{k.split('|')[1]}"
            elif f"*{k}" in kin_to_fam:
                k_prime = f"*{k}"
            else:
                logger.warning(f"Error with {k}")
                continue
        fam_to_num_seq[kin_to_fam[k_prime]] += kin_to_num_seq[k]
        kin_to_num_seq_new[k_prime] = kin_to_num_seq[k]

    kin_to_num_seq = kin_to_num_seq_new
    grp_to_num_sites = {g: sum([fam_to_num_seq[f] for f in grp_to_fam[g]]) for g in grp_to_fam}

    fams = sorted(fam_to_num_seq.keys())
    pd.DataFrame({"fam": fam_to_num_seq.keys(), "num_seq": fam_to_num_seq.values()}).to_csv("fam_to_num_seq.csv")
    if get_restart is None or not get_restart:
        return

    if get_restart:
        tks, vks, teks = {}, {}, {}
        states = run_groups(tgt, grp_to_num_sites, num_restarts=num_restarts)
        for seed_rank in [0, 1, 2, 3, 4]:
            tk, vk, tek = get_kin_assignments_from_state(states, seed_rank, fam_to_kin)
            accum_tek = collections.defaultdict(int)
            accum_vk = collections.defaultdict(int)
            for g in grp_to_fam.keys():
                for f in grp_to_fam[g]:
                    for k in fam_to_kin[f]:
                        if k in tek:
                            accum_tek[g] += kin_to_num_seq[k]
                        if k in vk:
                            accum_vk[g] += kin_to_num_seq[k]

            true_acc = 0
            for g in sorted(list(set(accum_vk.keys()) | set(accum_tek.keys()))):
                for accumulator in [accum_vk, accum_tek]:
                    assert tgt is not None
                    ideal = round(tgt / 2 * grp_to_num_sites[g], 0)
                    actual = accumulator[g]
                    diff = abs(ideal - actual)
                    # print(f"Diff for group {g} is {diff}")
                    true_acc += diff

            logger.info(
                f"Unsquared Score (Seed Rank {seed_rank}) ="
                f" {true_acc} ({true_acc/sum(grp_to_num_sites.values())*100:2.2f}%)"
            )
            tks[f"seed rank {seed_rank}"] = tk
            vks[f"seed rank {seed_rank}"] = vk
            teks[f"seed rank {seed_rank}"] = tek

        json.dump(tks, open(join_first("tr_kins_large.json", 1, __file__), "w"), indent=4)
        json.dump(vks, open(join_first("vl_kins_large.json", 1, __file__), "w"), indent=4)
        json.dump(teks, open(join_first("te_kins_large.json", 1, __file__), "w"), indent=4)
        json.dump(tks[f"seed rank {0}"], open(join_first("tr_kins.json", 1, __file__), 'w'), indent=4)
        json.dump(vks[f"seed rank {0}"], open(join_first("vl_kins.json", 1, __file__), 'w'), indent=4)
        json.dump(teks[f"seed rank {0}"], open(join_first("te_kins.json", 1, __file__), 'w'), indent=4)


def hill_climbing_score(
    state: list[int], group: str = "", targets: tuple[float, float, float] = (0.7, 0.15, 0.15)
) -> float:
    """The fitness function for the hill climbing algorithm.

    Parameters
    ----------
    state :
        The discrete vector state to be scored. (0 = train, 1 = val, 2 = test)
    group :
        The kinase group we are scoring.
    targets :
        The target fractions for each dataset, respectively for train, val, test, by default ``(0.7, 0.15, 0.15)``.

    Returns
    -------
    The fitness score. The formula is:
        .. math::
            \\text{fitness}_\\text{vl, te assignments}(grp,~dv,~dt) = \
            \\Bigl(\\text{ideal}(grp,~dv) - \\text{actual}_\\text{vl assig}(grp,~dv)\\big)^2 + \
            \\big(\\text{ideal}(grp,~dt) - \\text{actual}_\\text{te assig}(grp,~dt)\\Bigl)^2
    """
    global grp_to_fam, fam_to_num_seq
    scores = np.zeros((3,))
    grp_fams = grp_to_fam[group]
    for i in range(3):
        fams_inner = [grp_fams[j] for j in range(len(state)) if state[j] == i]
        for f in fams_inner:
            scores[i] += fam_to_num_seq[f]

    ideal_1 = np.array(targets[1] * np.sum(scores), dtype=int)
    ideal_2 = np.array(targets[2] * np.sum(scores), dtype=int)
    unsquared_1 = np.sum(np.abs(ideal_1 - scores[1]))
    unsquared_2 = np.sum(np.abs(ideal_2 - scores[2]))
    squared_1 = unsquared_1**2
    squared_2 = unsquared_2**2
    if ideal_1 > scores[1]:
        squared_1 *= 20
    if ideal_2 > scores[2]:
        squared_2 *= 20

    return squared_1 + squared_2


def get_assignment_info_dict(kin_fam_grp_file, raw_input_file, jt, jv, jte):
    """Pending moving this function."""
    split_into_sets(kin_fam_grp_file, raw_input_file, get_restart=None)
    tk = json.load(open(jt, "r"))
    vk = json.load(open(jv, "r"))
    tek = json.load(open(jte, "r"))

    tr_dist = [fam_to_grp[kin_to_fam[k]] for k in tk]
    vl_dist = [fam_to_grp[kin_to_fam[k]] for k in vk]
    te_dist = [fam_to_grp[kin_to_fam[k]] for k in tek]
    tr_num_kins = [(kin_to_num_seq[k], fam_to_grp[kin_to_fam[k]]) for k in tk]
    vl_num_kins = [(kin_to_num_seq[k], fam_to_grp[kin_to_fam[k]]) for k in vk]
    te_num_kins = [(kin_to_num_seq[k], fam_to_grp[kin_to_fam[k]]) for k in tek]

    ret_dict = {
        "train": {"kinases": tk, "group distribution": tr_dist, "num kins": tr_num_kins},
        "val": {"kinases": vk, "group distribution": vl_dist, "num kins": vl_num_kins},
        "test": {"kinases": tek, "group distribution": te_dist, "num kins": te_num_kins},
    }

    return ret_dict


if __name__ == "__main__":
    import os, pathlib

    where_am_i = pathlib.Path(__file__).parent.resolve()
    os.chdir(where_am_i)
    # print(os.getcwd())
    split_into_sets(
        "../kin_to_fam_to_grp_817.csv",
        "../../../data/raw_data/raw_data_22473.csv",
        tgt=0.3,
        get_restart=True,
        num_restarts=600,
    )
    # print("=====================\n\n")
