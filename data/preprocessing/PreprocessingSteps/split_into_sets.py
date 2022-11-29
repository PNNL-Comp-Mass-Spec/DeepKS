import collections
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import mlrose_hiive as mlr

def split_into_sets(kin_fam_grp_file, raw_input_file, tgt = 0.3, get_restart = True, restart_idx = None):
    global fam_to_grp, kin_to_fam, kin_to_num_seq
    kin_fam_grp = pd.read_csv(kin_fam_grp_file)
    grp_to_fam = collections.defaultdict(list)
    for _, r in kin_fam_grp[['Family', 'Group']].drop_duplicates(keep='first').iterrows():
        grp_to_fam[r['Group'].upper()].append(r['Family'].upper())

    fam_to_kin = collections.defaultdict(list) # {f: k for f, k in zip(kin_fam_grp['Family'], kin_fam_grp['Kinase'])}
    for _, r in kin_fam_grp.iterrows():
        fam_to_kin[r['Family'].upper()].append(f"{r['Kinase'].upper()}|{r['Uniprot']}")
        # print(sum([len(x) for x in fam_to_kin.values()]), _)

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

    raw = pd.read_csv(raw_input_file)[['lab', 'num_sites', 'uniprot_id']].rename(columns={'Kinase': 'lab', 'uniprot_id': 'Uniprot'}).drop_duplicates().applymap(lambda x: x.upper() if isinstance(x, str) else x)
    num_sites_join = pd.merge(kin_fam_grp, raw, how = 'left', on = 'Uniprot')
    assert(len(num_sites_join['lab']) == len(num_sites_join['Uniprot']) == len(num_sites_join['num_sites']))
    kin_to_num_seq = {f"{l}|{u}": n for l, u, n in zip(num_sites_join['lab'], num_sites_join['Uniprot'], num_sites_join['num_sites'])}
    fam_to_num_seq = collections.defaultdict(int)
    kin_to_num_seq_new = {}
    for k in kin_to_num_seq:
        k_prime = k
        if k not in kin_to_fam:
            if f"({k.split('|')[0]})|{k.split('|')[1]}" in kin_to_fam:
                k_prime = f"({k.split('|')[0]})|{k.split('|')[1]}"
            elif f"*{k}" in kin_to_fam:
                k_prime = f"*{k}"
            else:
                print("Error with", k)
                continue
        fam_to_num_seq[kin_to_fam[k_prime]] += kin_to_num_seq[k]
        kin_to_num_seq_new[k_prime] = kin_to_num_seq[k]

    kin_to_num_seq = kin_to_num_seq_new
    grp_to_num_sites = {g: sum([fam_to_num_seq[f] for f in grp_to_fam[g]]) for g in grp_to_fam}

    # %% [markdown]
    # ### Algorithm 4
    fams = sorted(fam_to_num_seq.keys())
    groups = sorted(grp_to_num_sites.keys())
    num_fams = len(fams)
    if get_restart is None:
        return
    def get_kin_assignments_from_state(state):
        results = [[] for _ in range(3)]
        for i in range(3):
            fams_inner = [fams[j] for j in range(len(state)) if state[j] == i]
            for f in fams_inner:
                results[i] += fam_to_kin[f]
        return tuple(results)

    def hill_climbing_score(state, **kwargs):
        scores = np.zeros((3, 10))
        for i in range(3):
            fams_inner = [fams[j] for j in range(len(state)) if state[j] == i]
            for f in fams_inner:
                scores[i][groups.index(fam_to_grp[f])] += fam_to_num_seq[f]

        ideal_1 = np.array(kwargs['targets'][1]*np.sum(scores, axis = 0), dtype=int)
        ideal_2 = np.array(kwargs['targets'][2]*np.sum(scores, axis = 0), dtype=int)
        diff_ideal_1 = np.sum(np.abs(ideal_1 - scores[1])**2)
        diff_ideal_2 = np.sum(np.abs(ideal_2 - scores[2])**2)
        unsquared_1 = np.sum(np.abs(ideal_1 - scores[1]))
        unsquared_2 = np.sum(np.abs(ideal_2 - scores[2]))
        if kwargs['unsquared'] != [-1]:
            kwargs['unsquared'].append((unsquared_1 + unsquared_2) * (1 if np.min(scores) > 1 else 1.5))
        
        return (diff_ideal_1 + diff_ideal_2) * (1 if np.min(scores) > 1 else 1.5)

    def core_ai(tgt, random_state, ret_score = True):
        if ret_score == True:
            fit = mlr.CustomFitness(hill_climbing_score, problem_type = 'discrete', targets = [1-tgt, tgt/2, tgt/2], unsquared = [-1])
        else:
            us = []
            fit = mlr.CustomFitness(hill_climbing_score, problem_type = 'discrete', targets = [1-tgt, tgt/2, tgt/2], unsquared = us)
        opt = mlr.DiscreteOpt(num_fams, fit, maximize = False, max_val = 3)
        res = mlr.simulated_annealing(opt, max_attempts=1000, max_iters=10000, random_state = random_state, init_state=[0 for _ in range(num_fams)])
        if ret_score:
            return res[1]
        else:
            return res[0], us[-1]

    def algorithm4(tgt, verbose = True):
        best_r = None
        best_score = float("inf")
        for r in tqdm(range(100), desc="Simulated Annealing Restarts"):
            pres = core_ai(tgt, r)
            if pres < best_score:
                best_r = r
                best_score = pres
        if verbose:
            print("Best score was {} achieved by random state {}".format(best_score, best_r))
        return best_r, best_score

    if get_restart:
        preresult = algorithm4(tgt)
        restart_idx = preresult[0]
    
    result, us = core_ai(tgt, restart_idx, ret_score = False)
    print("Unsquared Score:", us)

    tk, vk, tek = get_kin_assignments_from_state(result)

    json.dump(tk, open("../tr_kins.json", "w"), indent = 4)
    json.dump(vk, open("../vl_kins.json", "w"), indent = 4)
    json.dump(tek, open("../te_kins.json", "w"), indent = 4)

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

    ret_dict =  {'train': 
                    {'kinases': tk, 
                    'group distribution': tr_dist, 
                    'num kins': tr_num_kins},
                'val':
                    {'kinases': vk,
                    'group distribution': vl_dist,
                    'num kins': vl_num_kins},
                'test':
                    {'kinases': tek,
                    'group distribution': te_dist,
                    'num kins': te_num_kins}
                }

    return ret_dict

if __name__ == "__main__":
    import os, pathlib
    
    where_am_i = pathlib.Path(__file__).parent.resolve()
    os.chdir(where_am_i)
    print(os.getcwd())
    split_into_sets("../kin_to_fam_to_grp_817.csv", "../raw_data_22473.csv", tgt=0.3, get_restart = False, restart_idx = 41)