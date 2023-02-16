import numpy as np, json, pandas as pd, itertools
from scipy.sparse import csr_matrix, csgraph, vstack
from typing import Union
from .get_array_percentile import get_array_percentile


def get_groups_derangement2(NUM_KINS, SUBS_PER_KIN):
    np.random.seed(0)
    k = NUM_KINS
    m = SUBS_PER_KIN

    graph = np.ones((k * m, k * m))
    for i in range(k * m):
        for j in range(k * m):
            if i // m == j // m:
                graph[i][j] = 0

    neworder = np.random.permutation(k * m)
    graph_new = graph[neworder, :]
    graph_new = graph_new[:, neworder]

    ma = csgraph.maximum_bipartite_matching(csr_matrix(graph_new), perm_type="column")

    permuted = zip(range(len(ma)), ma)
    pi = lambda p: neworder[p]
    decoded = ((pi(x), pi(y)) for x, y in permuted)
    unscrambled = [x[1] for x in sorted(decoded)]

    chunks = [range(i * SUBS_PER_KIN, (i + 1) * SUBS_PER_KIN) for i in range(NUM_KINS)]
    for j in range(NUM_KINS * SUBS_PER_KIN):
        chunk = j // SUBS_PER_KIN
        assert unscrambled[j] not in chunks[chunk]

    return unscrambled


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

    neworder = np.random.permutation(total_nodes)
    graph_new = graph[neworder, :]
    graph_new = graph_new[:, neworder]

    ma = csgraph.maximum_bipartite_matching(csr_matrix(graph_new), perm_type= 'column')

    permuted = zip(range(len(ma)), ma)
    pi = lambda p: neworder[p]
    decoded = ((pi(x), pi(y)) for x, y in permuted)
    unscrambled = [x[1] for x in sorted(decoded)]

    chunks_new = [[unscrambled[j] for j in c] for c in chunks]
    for c1, c2 in zip(chunks, chunks_new):
        for x in c1:
            assert x not in c2

    return unscrambled


def get_groups_derangement4(
    order, sizes, kin_seq_fn, distance_matrix_file, percentile: Union[int, float] = 90, cache_derangement=False
):
    graph = None
    np.random.seed(0)

    d = pd.read_csv(distance_matrix_file, index_col=0).loc[order, order]
    np_version = d.values

    a = get_array_percentile(np_version, percentile, 1).astype(np.ubyte)
    graph = np.repeat(np.repeat(a, sizes, axis=0), sizes, axis=1)

    neworder = np.random.permutation(sum(sizes)).tolist()
    graph_new = graph[neworder, :]
    graph_new = graph_new[:, neworder]

    ma = csgraph.maximum_bipartite_matching(memory_efficient_np_to_sparse(graph_new), perm_type="column")
    matched_len = len([x for x in ma.tolist() if x != -1])
    try:
        assert matched_len == sum(sizes), (
            "The bipartite matching derangement was not able to produce an assignment for all decoys. (Max # of"
            f" assignments = {matched_len} and # of decoys = {sum(sizes)}."
        )
    except AssertionError as ae:
        raise RuntimeError(str(ae))

    ma = [x if x != -1 else None for x in ma]
    permuted: list[tuple[int, Union[int, None]]] = list(zip(range(len(ma)), ma))
    def pi(p: Union[int, None]) -> Union[int, None]:
        return neworder[p] if p is not None else None

    decoded:list[tuple[Union[None,int], Union[None,int]]] = [(pi(x), pi(y)) for x, y in permuted]
    unscrambled = [x[1] for x in sorted(decoded)]

    for i in range(graph.shape[0]):
        assert unscrambled[i] is None or graph[i][unscrambled[i]] == 1

    if cache_derangement:
        with open(f"{len(unscrambled)}.derangement", "w") as f:
            json.dump(unscrambled, f)
    return unscrambled

def memory_efficient_np_to_sparse(base: np.ndarray, chunk_multiplier: int = 10):
    assert base.shape[0] == base.shape[1], "The matrix must be square."
    if base.shape[0] % chunk_multiplier != 0:
        chunk_multiplier = 1
    s = base.shape[0]
    base = base.reshape((s * chunk_multiplier, -1))
    increment = s
    chunks = [base[i] for i in range(increment*chunk_multiplier)]
    sparse_tiles = [csr_matrix(x) for x in chunks]
    sparse_concatted = vstack(sparse_tiles)
    reshaped = sparse_concatted.reshape((s, s))
    del chunks
    del sparse_tiles
    del base
    del sparse_concatted
    return reshaped

def test_mem_eff_np_to_sparse():
    a = np.random.randint(0, 100, (1000, 1000))
    b = memory_efficient_np_to_sparse(a)
    assert np.all(a == b.toarray())

if __name__ == "__main__":
    test_mem_eff_np_to_sparse()