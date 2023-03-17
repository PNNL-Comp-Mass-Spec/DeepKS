import json, itertools, pandas as pd, random


def my_jaccard(x: set, y: set):
    if len(x.union(y)) == 0:
        return 0
    return round(100 * len(x.intersection(y)) / len(x.union(y)), 2)


def main(tr_file, vl_file, te_file):
    tr = json.load(open(tr_file, "r"))
    vl = json.load(open(vl_file, "r"))
    te = json.load(open(te_file, "r"))

    rankings = [set(tr.keys()), set(vl.keys()), set(te.keys())]
    assert all(rankings[0] == rankings[i] for i in range(1, len(rankings))), "Rankings provided are not the same."

    rankings_single = sorted(list(rankings[0]))
    rankings = list(itertools.combinations(rankings_single, 2))
    for ml_set, ml_set_name in zip([tr, vl, te], ["tr", "vl", "te"]):
        print(f"Jaccard Similarities for {ml_set_name}")
        mtx = pd.DataFrame(columns=rankings_single, index=rankings_single)
        for r, c in rankings:
            set_1 = set(ml_set[r])
            set_2 = set(ml_set[c])
            mtx.loc[r, c] = my_jaccard(set_1, set_2)
            mtx.loc[c, r] = mtx.loc[r, c]
            mtx.loc[r, r] = 100
            mtx.loc[c, c] = 100
        print(mtx, "\n")


def empirical_expected_value(size, fraction):
    all = list(range(size))
    part_a = set(random.sample(all, int(size * fraction)))
    part_b = set(random.sample(all, int(size * fraction)))
    # print(f"Jaccard Sim for {int(size * fraction)} samples of a pool of size {size}: {my_jaccard(part_a, part_b)}")
    return my_jaccard(part_a, part_b)


if __name__ == "__main__":
    main(*[f"../{x}_kins_large.json" for x in ["tr", "vl", "te"]])
    # X = []
    # Y = []
    # for size in [int(x) for x in [1e6]]:
    #     for fraction in [0] + np.linspace(0.01, 0.1, num=5).tolist() + np.linspace(0.1, 1, endpoint=True, num=10).tolist():
    #         iee = empirical_expected_value(size, fraction)
    #         X.append(fraction); Y.append(iee)
    # Y2 = [55.555*x**2 + 38.888*x for x in X]
    # plt.plot(X, Y, "bo-")
    # plt.plot(X, Y2, 'ro-')
    # plt.xlim(0, 1)
    # # plt.yticks(list(range(0, 110, 10)), list(range(0, 110, 10)))
    # plt.ylim(0, 100)
    # plt.show()
