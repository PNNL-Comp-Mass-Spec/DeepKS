"""Contains functions to generate derangements of bipartite sets, using maximum bipartite matching"""

import numpy as np, json, pandas as pd
from scipy.sparse import csr_matrix, csgraph, vstack
from typing import Union
from .get_array_percentile import get_array_percentile


def get_derangement(
    order: list[str],
    sizes: list[int],
    distance_matrix_file: str,
    percentile: int | float = 90,
    cache_derangement: bool = False,
) -> list[int | None]:
    """Takes a set of N elements grouped into M groups. The M groups have a similarity matrix. This function returns a derangement of the N elements, such that none of the N elements are paired with an element whose group similarity is above a certain percentile.

    Parameters
    ----------
    order :
        The input order of element groups to be deranged
    sizes :
        The order of sizes of each element group
    distance_matrix_file :
        Path to the distance matrix file
    percentile : optional
        Maximum column/row-wise percentile similarity to be considered acceptable, by default 90
    cache_derangement : optional
        Whether or not to save the derangement data (it can take a while for big inputs), by default False

    Returns
    -------
        List of indices, which range from 0 to :math:`\\sum` ``sizes`` represent the derangement of the input order. I.e., If sizes was ``[3, 3]`` and ``order`` was ``["A", "B"]``, then the derangement could be ``[3, 4, 5, 1, 0, 2]`` (if A kinases were not to be A kinases, and B kinases were not to be B kinases, based on ``percentile`` and ``distance_matrix_file``).

    Raises
    ------
    RuntimeError
        TODO


    Examples
    --------
    Suppose we have the following symmetric distance matrix with kinases A, B, and C:

    .. code-block:: plain

                              A    B    C
        distance matrix =  [[1.0, 0.4, 0.9], A
                            [0.4, 1.0, 0.0], B
                            [0.9, 0.0, 1.0]] C

    Suppose we have 4 sites phosphorylated by A, 2 sites phosphorylated by B, and 4 sites phosphorylated by C.

    And we say that in order for a kinase-site pair to be considered a "decoy", the percentile of similarity between the kinase a site's phosphorylating kinase must be less than 0.666. Hence, A kinases can be matched with B sites; B kinases can be matched with C sites; C kinases can be matched with A sites, based on the distance matrix. So a possible matching might be the following:

    .. code-block:: plain

        {
            A1: C9,
            A2: C8,
            A3: C10,
            A4: B6,
            B5: C7,  ↖︎
                      Symmetric
            B6: A4,  ↙︎
            C7: B5,
            C8: A2,
            C9: A1,
            C10: A3
        }

    In this function, the flow network is constructed as follows:

    .. code-block:: plain

                                            <dummy source>

                                    0   0   0   0   0   0   0   0   0
                                    |   |   |   |   |   |   |   |   |

                                    A1  A2  A3  A4  B5  B6  C7  C8  C9
                               [[ 0 , 0 , 0 , 0 | 1 , 1 | 1 , 1 , 1 ], A1    — 0
                                [ 0 , 0 , 0 , 0 | 1 , 1 | 1 , 1 , 1 ], A2    — 0
                                [ 0 , 0 , 0 , 0 | 1 , 1 | 1 , 1 , 1 ], A3    — 0
                                [ 0 , 0 , 0 , 0 | 1 , 1 | 1 , 1 , 1 ], A4    — 0
                                -------------------------------------
        adjacency weights   =   [ 1 , 1 , 1 , 1 | 0 , 0 | 1 , 1 , 1 ], B5    — 0    <dummy sink>
                                [ 1 , 1 , 1 , 1 | 0 , 0 | 1 , 1 , 1 ], B6    — 0
                                ------------------------------------
                                [ 1 , 1 , 1 , 1 | 1 , 1 | 0 , 0 , 0 ], C7    — 0
                                [ 1 , 1 , 1 , 1 | 1 , 1 | 0 , 0 , 0 ], C8    — 0
                                [ 1 , 1 , 1 , 1 | 1 , 1 | 0 , 0 , 0 ]] C9    — 0

    We would call this function in the following way:

    .. code-block:: python

        >>> get_derangement(['A', 'B', 'C'], [4, 2, 4], 'path/to/distance/mtx.csv', 0.666)

    And the output could be:

    .. code-block:: python

        [9, 8, 10, 6, 7, 4, 5, 2, 1, 3]
    """
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

    ma_tmp = []
    for x in ma:
        if x == -1:
            ma_tmp.append(None)
        else:
            ma_tmp.append(x)
    ma = ma_tmp
    permuted: list[tuple[int, Union[int, None]]] = list(zip(range(len(ma)), ma))

    def pi(p: Union[int, None]) -> Union[int, None]:
        if p is not None:
            return neworder[p]
        else:
            return None

    decoded: list[tuple[Union[None, int], Union[None, int]]] = [(pi(x), pi(y)) for x, y in permuted]
    unscrambled = [x[1] for x in sorted(decoded)]

    for i in range(graph.shape[0]):
        assert unscrambled[i] is None or graph[i][unscrambled[i]] == 1

    if cache_derangement:
        with open(f"{len(unscrambled)}.derangement", "w") as f:
            json.dump(unscrambled, f)
    return unscrambled


def memory_efficient_np_to_sparse(base: np.ndarray, chunk_multiplier: int = 10) -> csr_matrix:
    """Converts a numpy array to a sparse matrix in a memory efficient way.

    Parameters
    ----------
    base : np.ndarray
        The array to convert to a sparse matrix.
    chunk_multiplier : int, optional
        The pseudo-aspect ratio the input array gets reshaped to before chunking, by default 10

    Returns
    -------
        Sparse matrix representation of the input array.
    """
    assert base.shape[0] == base.shape[1], "The matrix must be square."
    if base.shape[0] % chunk_multiplier != 0:
        chunk_multiplier = 1
    s = base.shape[0]
    base = base.reshape((s * chunk_multiplier, -1))
    increment = s
    chunks = [base[i] for i in range(increment * chunk_multiplier)]
    sparse_tiles = [csr_matrix(x) for x in chunks]
    sparse_concatted = vstack(sparse_tiles)
    reshaped = csr_matrix(sparse_concatted.reshape((s, s)))
    del chunks
    del sparse_tiles
    del base
    del sparse_concatted
    return reshaped


if __name__ == "__main__":  # pragma: no cover

    def test_mem_eff_np_to_sparse():
        a = np.random.randint(0, 100, (1000, 1000))
        b = memory_efficient_np_to_sparse(a)
        assert np.all(a == b.toarray())

    test_mem_eff_np_to_sparse()

"""

    
    


    """
