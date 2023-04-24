import itertools, warnings, tqdm, os, re, tempfile, time, pathlib, matplotlib, numpy as np, pandas as pd, subprocess
from multiprocessing.pool import ThreadPool

matplotlib.rcParams["font.family"] = "monospace"

format_for_needle = (
    lambda x: x.replace("|", "_")
    .replace("(", "")
    .replace(")", "")
    .replace("/", "--")
    .replace(" ", "---")
    .replace(":", "----")
)
eldeen_rof_tamrof = lambda x: x.replace("_", "|").replace("----", ":").replace("---", " ").replace("--", "/")

where_am_i = pathlib.Path(__file__).parent.resolve()
os.chdir(where_am_i)
tq: tqdm.tqdm


# def instantiate_tq():
#     global tq
#     tq = tqdm.tqdm(
#         total=100,
#         desc="Approximate Progress",
#         unit="%",
#         leave=False,
#         position=0,
#         bar_format="{desc}: {percentage:.3f}%|{bar}| {n:.3f}/{total_fmt} [{elapsed}<{remaining}",
#     )


# def prog_bar_worker(num_seqs, done, num_workers):
#     global tq
#     instantiate_tq()
#     estimator = lambda x: ((0.0003799 * x**2 + 0.007835 * x) * 8 / num_workers) ** 1.08
#     estimated_total_secs = estimator(num_seqs)
#     tq.reset()
#     tq.write("Estimated total time: " + f"{estimated_total_secs:.2f} seconds")
#     while not done[0]:
#         time.sleep(0.25)
#         current = tq.n
#         if current + 0.25 * 100 / estimated_total_secs >= 100:
#             # tq.write("~Taking a little longer~" + str(time.time()))
#             break
#         tq.update(0.25 * 100 / estimated_total_secs)  # type: ignore


def worker(fasta_chunks: list[str]) -> dict[str, dict[str, float]]:
    assert len(fasta_chunks) == 4
    fasta_chunks_a = fasta_chunks[0]
    fasta_chunks_b = fasta_chunks[1]
    outfile = fasta_chunks[2] + f"-TID-{fasta_chunks[3]}"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as tmp_a, tempfile.NamedTemporaryFile(
        mode="w", suffix=".fasta"
    ) as tmp_b:
        tmp_a.write(fasta_chunks_a)
        tmp_a.flush()
        tmp_b.write(fasta_chunks_b)
        tmp_b.flush()
        cmd = f"needleall -asequence {tmp_a.name} -bsequence {tmp_b.name} -auto -aformat3 markx3 -outfile {outfile}"
        # tq.write("Running command")
        with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
            retcode = proc.wait()
    if retcode != 0:
        raise RuntimeError(
            "Error: The `needleall` command for running pairwise alignments did not exit with success. For reference,"
            f" the failed command was:\n\n{cmd}"
        )
    with open(outfile, "r") as f:
        results = {}
        output_lines = f.read()
        rgx = (
            r"#={39}\n#\n# Aligned_sequences: 2\n# 1: (.*)\n# 2: (.*)(\n.*){5}\n# Identity:\s+[0-9]+\/[0-9]+"
            r" \(([0-9\.\s]+)%\)"
        )
        for match in re.findall(rgx, output_lines):
            if not match[0] in results:
                results[match[0]] = {}
            results[match[0]][match[1]] = float(match[3])
    os.unlink(outfile)
    return results


def get_needle_pairwise_mtx(
    fasta: str, outfile: str, subset: int = -1, num_procs: int = 4, restricted_combinations: list = []
):
    assert len(restricted_combinations) in [0, 2], "Restricted combinations must be a list of two iterables or empty."
    if len(restricted_combinations) == 2 and (
        len(restricted_combinations[0]) == 0 or len(restricted_combinations[1]) == 0
    ):
        return pd.DataFrame()
    with open(fasta, "r") as f:
        fasta_all = re.findall(r">.*\n[^>]+", f.read())
    if subset == -1:
        subset = len(fasta_all)
    fastas = [format_for_needle(x) for x in fasta_all[:subset]]
    assert len(fastas) >= 2, "Need at least two sequences to compute pairwise distances."

    strs_a = []
    strs_b = []

    if len(restricted_combinations) == 0:
        combos = list(itertools.combinations(fastas, 2))
        assert len(combos) == len(fastas) * (len(fastas) - 1) / 2, "Wrong number of combinations"
        fastas_a = [i[0] for i in combos]
        fastas_b = [i[1] for i in combos]
        assert len(fastas_a) == len(fastas_b), "Unequal number of strings"
        group_size = int(np.ceil(len(combos) / num_procs))
        for i in range(0, len(combos), group_size):
            strs_a.append(format_for_needle("".join(list(set(fastas_a[i : i + group_size])))))
            strs_b.append(format_for_needle("".join(list(set(fastas_b[i : i + group_size])))))
    else:
        name_to_seq = {
            format_for_needle(x.split("\n")[0].split(">")[1].upper()): "\n".join(x.split("\n")[1:]).strip()
            for x in fastas
        }
        fastas_a = [format_for_needle(x) for x in restricted_combinations[0]]
        fastas_b = [format_for_needle(x) for x in restricted_combinations[1]]
        fastas_a = [f">{x}\n{name_to_seq[x]}\n" for x in fastas_a]
        fastas_b = [f">{x}\n{name_to_seq[x]}\n" for x in fastas_b]
        smaller = fastas_a if len(fastas_a) < len(fastas_b) else fastas_b
        larger = fastas_a if len(fastas_a) >= len(fastas_b) else fastas_b
        group_size_a = int(np.ceil(len(larger) / num_procs))
        for i in range(0, len(fastas_a), group_size_a):
            strs_a.append("".join(larger[i : i + group_size_a]))
        strs_b = ["".join(smaller) for _ in range(len(strs_a))]

    assert len(strs_a) == len(strs_b) > 0, "Unequal number of strings/len of strings is zero"
    args = [[strs_a[i], strs_b[i], outfile, i] for i in range(len(strs_a))]
    done = [False]
    # progress_thread = threading.Thread(target=prog_bar_worker, args=(subset, done, num_procs))
    # progress_thread.start() TODO: Only have progress bar if verbose
    with ThreadPool(num_procs) as p:
        results = p.map(worker, args)
    done[0] = True
    # progress_thread.join()
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
    if len(restricted_combinations) == 0:
        df_results = pd.DataFrame(data="NA", index=rc_names, columns=rc_names).sort_index(axis=0).sort_index(axis=1)
        for k1 in final_results:
            for k2 in final_results[k1]:
                df_results.at[k1, k2] = final_results[k1][k2]
                df_results.at[k2, k1] = final_results[k1][k2]
        np.fill_diagonal(df_results.values, 100)
    else:
        df_results = pd.DataFrame(final_results)
    df_results = df_results.astype(float)
    df_results.columns = [eldeen_rof_tamrof(x) for x in df_results.columns.to_list()]
    df_results.index = pd.Index([eldeen_rof_tamrof(x) for x in df_results.index.tolist()])
    assert (
        np.asarray(df_results.columns).tolist() == df_results.index.tolist() or len(restricted_combinations) > 0
    ), "df_results.columns != df_results.index"
    assert (
        df_results.shape[0] == df_results.shape[1] == subset or len(restricted_combinations) > 0
    ), "Matrix is not square/the right size"
    if pd.isna(df_results.values).any():
        with warnings.catch_warnings():  # TODO: Show warnings if Verbose
            pd.set_option("display.max_rows", 1000)
            pd.set_option("display.max_columns", 1000)
            pd.set_option("display.width", 140)
            warnings.formatwarning = warning_on_one_line
            warnings.warn(
                (
                    "There are NA values in the results matrix. This may simply mean that some sequences had zero"
                    " overlap. Replacing these instances with 0.0. To enable checking, printing a subset of the matrix"
                    " with"
                    f" NAs:\n{df_results[df_results.isna().any(axis=1)][df_results.isna().any(axis=0).index[df_results.isna().any(axis=0)]]}\n"
                ),
                RuntimeWarning,
            )
            df_results = df_results.fillna(0.0)
            pd.set_option("display.max_rows", 50)
            pd.set_option("display.max_columns", 15)
            pd.set_option("display.width", 120)

    return df_results


def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return f"{filename}:{lineno}: {category.__name__}: {message}"


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
        # fn = mtx_utils.make_fasta("../data/raw_data/kinase_seq_822.txt", td)
        # test_subsets = [822]
        # benchmark_performance(fn, test_subsets)
        get_needle_pairwise_mtx(
            "test_selected_combos.fasta",
            "tmp.txt",
            -1,
            num_procs=1,
            restricted_combinations=[
                # itertools.product(, )
                ["S" + str(i) for i in range(1, 26)],
                ["S" + str(i) for i in range(26, 41)],
            ],
        )
