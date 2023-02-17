import collections
import re, sys
from types import NoneType
import pandas as pd
import numpy as np
import json
from sklearn.utils import indexable
from tqdm import tqdm
import mlrose_hiive as mlr
from typing import Union
from numpy.ma import core as np_ma_core

newstream = open("debug_non_det.log", "a")
np.set_printoptions(linewidth=240)

def ma_from_inds(base_ar, inds):
    initial = np.ma.MaskedArray([base_ar[i] if not np.ma.is_masked(i) else np_ma_core.MaskedConstant() for i in inds])
    post_processing = lambda x: x.astype(base_ar.dtype) if isinstance(base_ar, np.ndarray) else x
    return post_processing(initial)


def get_inds_of_unq_top_k(ls, top_k, verify_sorted=True):
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
    kin_fam_grp_file,
    raw_input_file,
    tgt: Union[NoneType, float, int] = 0.3,
    get_restart: Union[NoneType, bool] = True,
    restart_idx=None,
    num_restarts=25,
):
    global fam_to_grp, kin_to_fam, kin_to_num_seq, grp_to_fam, fam_to_num_seq, fams
    kin_fam_grp = pd.read_csv(kin_fam_grp_file)
    grp_to_fam = collections.defaultdict(list)
    for _, r in kin_fam_grp[["Family", "Group"]].drop_duplicates(keep="first").iterrows():
        grp_to_fam[r["Group"].upper()].append(r["Family"].upper())

    fam_to_kin = collections.defaultdict(list)  # {f: k for f, k in zip(kin_fam_grp['Family'], kin_fam_grp['Kinase'])}
    for _, r in kin_fam_grp.iterrows():
        fam_to_kin[r["Family"].upper()].append(f"{r['Kinase'].upper()}|{r['Uniprot']}")
        # print(sum([len(x) for x in fam_to_kin.values()]), _)

    fam_to_kin = collections.OrderedDict(sorted(fam_to_kin.items(), key=lambda x: x[0]))
    grp_to_fam = collections.OrderedDict(sorted(grp_to_fam.items(), key=lambda x: x[0]))

    accum = 0
    kin_to_fam = {}
    for f in fam_to_kin:
        for k in fam_to_kin[f]:
            if k in kin_to_fam:
                print("Kin already in!")
                print(k)
            kin_to_fam[k] = f

    fam_to_grp = {}
    for g in grp_to_fam:
        for f in grp_to_fam[g]:
            fam_to_grp[f] = g

    for f in fam_to_kin:
        fam_to_kin[f].sort()

    raw = (
        pd.read_csv(raw_input_file)[["lab", "num_sites", "uniprot_id"]]
        .rename(columns={"Kinase": "lab", "uniprot_id": "Uniprot"})
        .drop_duplicates()
        .applymap(lambda x: x.upper() if isinstance(x, str) else x)
    )
    num_sites_join = pd.merge(kin_fam_grp, raw, how="left", on="Uniprot")
    assert len(num_sites_join["lab"]) == len(num_sites_join["Uniprot"]) == len(num_sites_join["num_sites"])
    kin_to_num_seq = {
        f"{l}|{u}": n for l, u, n in zip(num_sites_join["lab"], num_sites_join["Uniprot"], num_sites_join["num_sites"])
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
                print("Error with", k)
                continue
        fam_to_num_seq[kin_to_fam[k_prime]] += kin_to_num_seq[k]
        kin_to_num_seq_new[k_prime] = kin_to_num_seq[k]

    kin_to_num_seq = kin_to_num_seq_new
    grp_to_num_sites = {g: sum([fam_to_num_seq[f] for f in grp_to_fam[g]]) for g in grp_to_fam}

    fams = sorted(fam_to_num_seq.keys())
    pd.DataFrame({"fam": fam_to_num_seq.keys(), "num_seq": fam_to_num_seq.values()}).to_csv("fam_to_num_seq.csv")
    if get_restart is None or not get_restart:
        return

    def get_kin_assignments_from_state(states, which_seed):
        states = {k: v[min(which_seed, len(v))] for k, v in states.items()}
        results = [[] for _ in range(3)]
        for group in states:
            for i in range(3):
                fams_inner = [grp_to_fam[group][j] for j in range(len(states[group])) if states[group][j] == i]
                for f in fams_inner:
                    results[i] += fam_to_kin[f]
        for r in results:
            r.sort(key=lambda x: re.sub(r"[\(\)\*]", "", x))
        return tuple(results)

    def core_ai(tgt, random_state, group, ret_score=True):
        num_fams = len(grp_to_fam[group])
        if ret_score == True:
            fit = mlr.CustomFitness(
                hill_climbing_score, problem_type="discrete", targets=[1 - tgt, tgt / 2, tgt / 2], group=group
            )
        else:
            fit = mlr.CustomFitness(
                hill_climbing_score, problem_type="discrete", targets=[1 - tgt, tgt / 2, tgt / 2], group=group
            )
        opt = mlr.DiscreteOpt(num_fams, fit, maximize=False, max_val=3)
        res = mlr.simulated_annealing(
            opt, max_attempts=1000, max_iters=1000, random_state=random_state, init_state=[0 for _ in range(num_fams)]
        )
        if group == "<UNANNOTATED>" and random_state == 1:
            print(res, file = newstream)
        return res

    def algorithm4(tgt, group, verbose=True, num_restarts=100, top_k=5):
        scores = []
        states = []
        for r in tqdm(range(1, num_restarts + 1), position=0, leave=True, desc="Group Restart Progress: ", file=sys.stderr, colour = 'cyan'):
            present = core_ai(tgt, r, group)
            states.append(present[0])
            scores.append(present[1])
        argst = np.argsort(scores, kind="stable")
        scores = np.array(sorted(scores))
        unq_top_k_ids = get_inds_of_unq_top_k(scores, top_k)
        ma = ma_from_inds(scores, unq_top_k_ids)  # ma = min(scores)
        mai = np.ma.MaskedArray(
            [argst[i] if type(i) == np.int64 else None for i in unq_top_k_ids],
            mask=[False if type(i) == np.int64 else True for i in unq_top_k_ids],
        )  # mai = scores.index(ma)
        sd = np.std(scores)
        state = [states[argst[i]] for i in range(len(argst))]
        if verbose:
            tqdm.write(
                "Best scores for group {} were {} achieved by random states {} and the restart scores had an SD of"
                " {:3.2f}".format(group, ma, mai, sd),
                sys.stdout,
            )
        return mai, ma, sd, state

    def algorithm5(tgt, verbose=True, num_restarts=100):
        group_scores = []
        group_r = []
        group_sd = []
        group_states = {}
        for i, group in enumerate(
            tq := tqdm(sorted(list(grp_to_num_sites.keys())), position=0, leave=True, file=sys.stderr, colour = 'cyan')
        ):
            tq.set_description(f"Processing Group {group}")
            group_result = algorithm4(tgt, group, verbose, num_restarts=num_restarts)
            group_r.append(group_result[0])
            group_scores.append(group_result[1])
            group_sd.append(group_result[2])
            group_states[group] = group_result[3]

        # print(f"Overall Top k Scores = {[sum([group_scores[j][i] for j in range(len(group_scores))]) for i in range(len(group_scores[0]))]}")
        return group_states

    if get_restart:
        tks, vks, teks = {}, {}, {}
        states = algorithm5(tgt, num_restarts=num_restarts, verbose=True)
        for seed_rank in [0, 1, 2, 3, 4]:
            tk, vk, tek = get_kin_assignments_from_state(states, seed_rank)
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
                for accum in [accum_vk, accum_tek]:
                    assert tgt is not None
                    ideal = round(tgt / 2 * grp_to_num_sites[g], 0)
                    actual = accum[g]
                    diff = abs(ideal - actual)
                    # print(f"Diff for group {g} is {diff}")
                    true_acc += diff

            print(f"Unsquared Score (Seed Rank {seed_rank}) = {true_acc} ({true_acc/sum(grp_to_num_sites.values())*100:2.2f}%)")
            tks[f'seed rank {seed_rank}'] = tk
            vks[f'seed rank {seed_rank}'] = vk
            teks[f'seed rank {seed_rank}'] = tek

        json.dump(tks, open("../tr_kins_large.json", "w"), indent=4)
        json.dump(vks, open("../vl_kins_large.json", "w"), indent=4)
        json.dump(teks, open("../te_kins_large.json", "w"), indent=4)
        json.dump(tks[f'seed rank {0}'], open("../tr_kins.json", "w"), indent=4)
        json.dump(vks[f'seed rank {0}'], open("../vl_kins.json", "w"), indent=4)
        json.dump(teks[f'seed rank {0}'], open("../te_kins.json", "w"), indent=4)



def hill_climbing_score(state, **kwargs):
    global grp_to_fam, fam_to_num_seq
    scores = np.zeros((3,))
    grp_fams = grp_to_fam[kwargs["group"]]
    for i in range(3):
        fams_inner = [grp_fams[j] for j in range(len(state)) if state[j] == i]
        for f in fams_inner:
            scores[i] += fam_to_num_seq[f]

    ideal_1 = np.array(kwargs["targets"][1] * np.sum(scores), dtype=int)
    ideal_2 = np.array(kwargs["targets"][2] * np.sum(scores), dtype=int)
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
    split_into_sets(kin_fam_grp_file, raw_input_file, tgt=None, get_restart=None)
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
    print("=====================\n\n", file=newstream)
