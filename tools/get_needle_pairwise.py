import itertools
import tqdm
import os, re, tempfile, time, pathlib
import threading
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd, tqdm

matplotlib.rcParams["font.family"] = "Palatino"
from multiprocessing.pool import ThreadPool
import subprocess
from ..data.preprocessing.PreprocessingSteps import get_labeled_distance_matrix as mtx_utils

where_am_i = pathlib.Path(__file__).parent.resolve()
os.chdir(where_am_i)
tq = tqdm.tqdm(
    total=100,
    desc="Approximate Progress",
    unit="%",
    leave=False,
    position=0,
    bar_format="{desc}: {percentage:.3f}%|{bar}| {n:.3f}/{total_fmt} [{elapsed}<{remaining}",
)


def prog_bar_worker(num_seqs, done, num_workers):
    global tq
    estimator = lambda x: ((0.0003799 * x**2 + 0.007835 * x)*8/num_workers)**1.08
    estimated_total_secs = estimator(num_seqs)
    tq.reset()
    tq.write("Estimated total time: " + f"{estimated_total_secs:.2f} seconds")
    while not done[0]:
        time.sleep(0.25)
        current = tq.n
        if current + 0.25 * 100 / estimated_total_secs >= 100:
            # tq.write("~Taking a little longer~" + str(time.time()))
            break
        tq.update(0.25 * 100 / estimated_total_secs)  # type: ignore


def worker(fasta_chunks: list[str]) -> dict[str, dict[str, float]]:
    assert len(fasta_chunks) == 3
    fasta_chunks_a = fasta_chunks[0]
    fasta_chunks_b = fasta_chunks[1]
    outfile = fasta_chunks[2]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as tmp_a, tempfile.NamedTemporaryFile(
        mode="w", suffix=".fasta"
    ) as tmp_b:
        tmp_a.write(fasta_chunks_a)
        tmp_a.flush()
        tmp_b.write(fasta_chunks_b)
        tmp_b.flush()
        cmd = f"needleall -asequence {tmp_a.name} -bsequence {tmp_b.name} -auto -aformat3 markx3 -outfile {outfile}"
        # tq.write("Running command")
        subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).wait()
    with open(outfile, "r") as f:
        results = {}
        output_lines = f.read()
        rgx = (
            r"#={39}\n#\n# Aligned_sequences: 2\n# 1: (.*)\n# 2: (.*)(\n.*){5}\n# Identity:\s+[0-9]+\/[0-9]+"
            r" \(([0-9\.\s]+)%\)"
        )
        for match in (re.findall(rgx, output_lines)):
            if not match[0] in results:
                results[match[0]] = {}
            results[match[0]][match[1]] = float(match[3])
    return results


def get_needle_pairwise_mtx(fasta: str, outfile: str, subset: int = -1, num_procs: int = 4):
    fasta_all = re.findall(r">.*\n[^>]+", open(fasta, "r").read())
    if subset == -1:
        subset = len(fasta_all)
    fastas = fasta_all[:subset]
    combos = list(itertools.combinations(fastas, 2))
    assert len(combos) == len(fastas) * (len(fastas) - 1) / 2, "Wrong number of combinations"
    fastas_a = [i[0] for i in combos]
    fastas_b = [i[1] for i in combos]
    assert len(fastas_a) == len(fastas_b), "Unequal number of strings"
    group_size = len(combos) // num_procs
    strs_a = []
    strs_b = []
    for i in range(0, len(combos), group_size):
        strs_a.append("".join(list(set(fastas_a[i : i + group_size]))).replace("|", "_"))
        strs_b.append("".join(list(set(fastas_b[i : i + group_size]))).replace("|", "_"))
    assert len(strs_a) == len(strs_b) > 0, "Unequal number of strings/len of strings is zero"
    args = [[strs_a[i], strs_b[i], f"{i}-" + outfile] for i in range(len(strs_a))]
    done = [False]
    progress_thread = threading.Thread(target=prog_bar_worker, args=(subset, done, num_procs))
    progress_thread.start()
    with ThreadPool(num_procs) as p:
        results = p.map(worker, args)
    done[0] = True
    final_results = {}
    for chunk_dict in results:
        for k1 in chunk_dict:
            if not k1 in final_results:
                final_results[k1] = {}
            for k2 in chunk_dict[k1]:
                final_results[k1][k2] = chunk_dict[k1][k2]
    rc_names = list(
        set(list(itertools.chain(*[i.keys() for i in final_results.values()]))).union(set(final_results.keys()))
    )
    df_results = pd.DataFrame(data="NA", index=rc_names, columns=rc_names).sort_index(axis=0).sort_index(axis=1)
    for k1 in final_results:
        for k2 in final_results[k1]:
            df_results.at[k1, k2] = final_results[k1][k2]
            df_results.at[k2, k1] = final_results[k1][k2]
    i = df_results.index
    np.fill_diagonal(df_results.values, 100.0)
    df_results = df_results.astype(float)
    prepare = lambda x: x.replace("_", "|").upper()
    df_results.columns = [prepare(x) for x in df_results.columns.to_list()]
    df_results.index = pd.Index([prepare(x) for x in df_results.index.tolist()])
    assert df_results.columns.tolist() == df_results.index.tolist(), "df_results.columns != df_results.index"
    assert df_results.shape[0] == df_results.shape[1] == subset, "Matrix is not square/the right size"
    assert not pd.isna(df_results.values).any(), "Matrix contains NaNs"

    return df_results


def benchmark_performance(fasta_a_file, test_subsets):
    times = []
    for sub in test_subsets:
        tq.write(f"Running alignment for {sub} sequences...")
        start = time.time()
        results = get_needle_pairwise_mtx(fasta_a_file, "tmp.txt", sub, num_procs=8)
        results.to_csv(f"pairwise_mtx_{sub}.csv")
        tq.reset()
        # tq.write(results)
        # exit(1)
        time_elapsed = time.time() - start
        times.append(time_elapsed)
        tq.write("".join(["Actual time: ", str(time_elapsed), " seconds"]))

    # for x, y in zip(test_subsets, times):
    #     tq.write(x, ",", y)
    # plt.plot(test_subsets, times, "bo-")
    # plt.xlabel("Number of sequences")
    # plt.ylabel("Time (s)")
    # plt.title("Needle pairwise alignment benchmark")
    # plt.show()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        fn = mtx_utils.make_fasta("../data/raw_data/kinase_seq_822.txt", td)
        test_subsets = [822]
        benchmark_performance(fn, test_subsets)
