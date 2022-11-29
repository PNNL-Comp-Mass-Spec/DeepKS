import warnings
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csgraph
import itertools
from get_array_percentile import get_array_percentile

def get_groups_derangement2(NUM_KINS, SUBS_PER_KIN):
    np.random.seed(0)
    k = NUM_KINS
    m = SUBS_PER_KIN 

    graph = np.ones((k*m, k*m))
    for i in range(k*m):
        for j in range(k*m):
            if i // m == j // m:
                graph[i][j] = 0

    neworderc = np.random.permutation(k*m)
    graph_new = graph[neworderc, :]
    graph_new = graph_new[:, neworderc]

    ma = csgraph.maximum_bipartite_matching(csr_matrix(graph_new), perm_type= 'column')

    crypted = zip(range(len(ma)), ma)
    pi = lambda p: neworderc[p]
    decoded = ((pi(x), pi(y)) for x, y in crypted)
    unscrambled = [x[1] for x in sorted(decoded)]

    chunks = [range(i*SUBS_PER_KIN, (i+1)*SUBS_PER_KIN) for i in range(NUM_KINS)]
    for j in range(NUM_KINS*SUBS_PER_KIN):
        chunk = j // SUBS_PER_KIN
        assert(unscrambled[j] not in chunks[chunk])

    return(unscrambled)

def get_groups_derangement3(group_lengths):
    np.random.seed(0)
    total_nodes = sum(group_lengths)

    running_ind = 0
    chunks = []
    for gl in group_lengths:
        chunks.append(range(running_ind, running_ind + gl))
        running_ind += gl

    graph = np.ones((total_nodes, total_nodes))
    for c in chunks:
        for x in c:
            for y in c:
                graph[x][y] = 0

    neworderc = np.random.permutation(total_nodes)
    graph_new = graph[neworderc, :]
    graph_new = graph_new[:, neworderc]

    ma = csgraph.maximum_bipartite_matching(csr_matrix(graph_new), perm_type= 'column')

    crypted = zip(range(len(ma)), ma)
    pi = lambda p: neworderc[p]
    decoded = ((pi(x), pi(y)) for x, y in crypted)
    unscrambled = [x[1] for x in sorted(decoded)]

    chunks_new = [[unscrambled[j] for j in c] for c in chunks]
    for c1, c2 in zip(chunks, chunks_new):
        for x in c1:
            assert(x not in c2)

    return(unscrambled)

def get_groups_derangement4(order, sizes, kin_seq_fn, distance_matrix_file, percentile = 90):
    graph = None
    np.random.seed(0)

    d = pd.read_csv(distance_matrix_file, index_col=0)
    np_version = d.values


    a = get_array_percentile(np_version, percentile, 1)
    
    graph = np.array(list(itertools.chain(*[[list(itertools.chain(*[[a[i][j]]*sizes[j] for j in range(len(sizes))])) for k in range(sizes[i])] for i in range(len(sizes))])), dtype = np.uint8)


    neworderc = np.random.permutation(sum(sizes))
    graph_new = graph[neworderc, :]
    graph_new = graph_new[:, neworderc]

    ma = csgraph.maximum_bipartite_matching(csr_matrix(graph_new), perm_type= 'column')
    matched_len = len([x for x in ma.tolist() if x != -1])
    try:
        assert matched_len == sum(sizes), f"The bipartite matching derangement was not able to produce an assignment for all decoys. (Max # of assignments = {matched_len} and # of decoys = {sum(sizes)}."
    except AssertionError as ae:
        raise RuntimeError(str(ae))

    ma = [x if x != -1 else None for x in ma]
    crypted = list(zip(range(len(ma)), ma))
    pi = lambda p: neworderc[p] if p is not None else None
    decoded = [(pi(x), pi(y)) for x, y in crypted]
    unscrambled = [x[1] for x in sorted(decoded)]

    for i in range(graph.shape[0]):
        assert(unscrambled[i] is None or graph[i][unscrambled[i]] == 1)

    return unscrambled