import collections
import multiprocessing, itertools, random, time, os, sigfig, signal, warnings, re, json, pathlib, functools
import pandas as pd
import tempfile, pprint, tabulate
import numpy as np
from typing import Collection, MutableSequence, Iterable
from prettytable import PrettyTable
from beautifultable import BeautifulTable
from numpyencoder import NumpyEncoder

from ..config.root_logger import get_logger
from .file_names import get as get_file_name

logger = get_logger()

join_first = lambda levels, x: os.path.join(pathlib.Path(__file__).parent.resolve(), *[".."] * levels, x)


def sigfig_iter(iterable, sigfigs=3):
    res = []
    for x in iterable:
        if isinstance(x, float):
            res.append(sigfig.round(x, sigfigs))
        else:
            raise ValueError(f"sigfig_iter only works on floats, not {type(x)}")
    return res


class SimpleTuner:
    def __init__(
        self,
        num_sim_procs,
        train_fn,
        config_dict,
        which_map,
        num_samples,
        random_seed=84,
        collapse=[[]],
        num_gpu=None,
        max_output_width=512,
    ):
        if num_gpu is None:
            self.num_gpu = num_sim_procs
        else:
            self.num_gpu = num_sim_procs
        self.max_output_width = max_output_width

        self.num_sim_procs = num_sim_procs
        self.train_fn = train_fn
        self.config_dict = config_dict

        assert all(isinstance(x, list) for x in collapse), "`collapse` must be a list of lists."
        if collapse != [[]]:
            for set_ in collapse:
                config_dict[f'group__[[{"+".join(set_)}]]__'] = list(zip(*[list(config_dict[x]) for x in set_]))
                for x in set_:
                    del config_dict[x]

        assert all(isinstance(x, list) or isinstance(x, np.ndarray) for x in config_dict.values()), (
            "Not all hyperparameters are in a list or numpy array."
            f" ({'; '.join([f'{k}:{v}' for k, v in config_dict.items() if not (isinstance(v, list) or isinstance(v, np.ndarray))])})"
        )

        random.seed(random_seed)
        self.sampled_config_dicts = []
        for _ in range(num_samples):
            running = {}
            for k, v in config_dict.items():
                running[k] = random.choice(v if isinstance(v, list) else v.tolist())
            self.sampled_config_dicts.append(running)

        logger.info(
            f"{num_samples:,} configurations out of a possible ~"
            f" 10^{np.sum([np.log10(len(v)) for v in config_dict.values()]):,.2f} were randomly sampled."
        )

        self.model_params = [
            {
                "default": {
                    k: int(v) if type(v) == np.int64 else v
                    for k, v in sampled_config_dict.items()
                    if k in which_map["model_params"]
                }
            }
            for sampled_config_dict in self.sampled_config_dicts
        ]
        self.training_params = [
            {
                "default": {
                    k: int(v) if type(v) == np.int64 else v
                    for k, v in sampled_config_dict.items()
                    if k in which_map["training_params"]
                }
            }
            for sampled_config_dict in self.sampled_config_dicts
        ]
        self.interface_params = [
            {
                "default": {
                    k: int(v) if type(v) == np.int64 else v
                    for k, v in sampled_config_dict.items()
                    if k in which_map["interface_params"]
                }
            }
            for sampled_config_dict in self.sampled_config_dicts
        ]

        assert len(self.model_params) == len(self.training_params) == len(self.interface_params) == num_samples

        # k_outer = -1
        # for d in self.sampled_config_dicts:
        #     splits = []
        #     for k in d:
        #         if re.search(r"group__\[\[(.+\+.+)\]\]__", k):
        #             ksub = re.sub(r"group__\[\[(.+\+.+)\]\]__", r"\1", k)
        #             splits = ksub.split("+")
        #         k_outer = k

        #     for i, hp in enumerate(splits):
        #         d[hp] = d[k_outer][i]

        #     if len(splits) != 0:
        #         del d[k_outer]

        self._print_combinations()

    def generate_args_for_go(self, base_args, tempdir):
        all_args = []
        for i in range(len(self.model_params)):
            model_params = self.model_params[i]
            training_params = self.training_params[i]
            interface_params = self.interface_params[i]
            with tempfile.NamedTemporaryFile("w", dir=tempdir, delete=False) as mpf, tempfile.NamedTemporaryFile(
                "w", dir=tempdir, delete=False
            ) as tpf, tempfile.NamedTemporaryFile("w", dir=tempdir, delete=False) as ipf:
                json.dump(model_params, mpf, cls=NumpyEncoder)
                json.dump(training_params, tpf, cls=NumpyEncoder)
                json.dump(interface_params, ipf, cls=NumpyEncoder)
                mpf.flush()
                tpf.flush()
                ipf.flush()
                self.model_params_file = mpf.name
                self.training_params_file = tpf.name
                self.interface_params_file = ipf.name
            param_args = [
                "--ksr-params",
                self.model_params_file,
                "--ksr-training-params",
                self.training_params_file,
                "--nni-params",
                self.interface_params_file,
            ]
            full_args = base_args + param_args
            all_args.append(full_args)
        return all_args

    def _print_combinations(self):
        preview = {f"cfg # {i}": v for i, v in enumerate(self.sampled_config_dicts[:20])}
        df = pd.DataFrame.from_dict(preview, orient="index")
        df["cfg #"] = df.index
        df.insert(0, "cfg #", df.pop("cfg #"))
        # info = json.dumps(, indent=3, cls=NumpyEncoder)
        bt = BeautifulTable()
        include_cols = [i for i in range(len(df.columns)) if df.iloc[:, i].apply(lambda x: str(x)).nunique() > 1]

        bt.columns.header = df.columns[include_cols].tolist()
        for row in df.values[:, include_cols].tolist():
            bt.rows.append(row)
        bt.columns.width = (os.get_terminal_size().columns - 1 - len(bt.columns.header)) // (len(bt.columns.header))

        logger.info("\n" + str(bt))

    def init_pool(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def go(self, args_collection):
        scores = []
        best_score = 0
        best_score_index = None
        # with multiprocessing.Pool(self.num_sim_procs, initializer=self.init_pool) as pool:
        #     try:
        #         results = pool.starmap(
        #             self.display_intermediates,
        #             [[args_collection[i]] for i in range(len(self.sampled_config_dicts))],
        #         )
        #     except KeyboardInterrupt:
        #         logger.status("\nQuitting...\n")
        #         pool.terminate()
        #         pool.join()
        #         exit(1)
        #     except Exception:
        #         logger.error("Error in multiprocessing")
        #         pool.terminate()
        #         pool.join()
        #         exit(1)
        #     ret = [x for x in results]
        for i, args in enumerate(args_collection):
            new_score = self.display_intermediates(args)
            scores.append(new_score)
            if best_score < new_score:
                best_score = new_score
                best_score_index = i

        res_json = []
        res_json.append({"Best Score": best_score, "Best Score Index": best_score_index, "Scores": scores})
        res_json += args_collection
        with open(get_file_name("tuning", "json", join_first(1, "")), "w") as f:
            json.dump(res_json, f, indent=3)

    def display_final_results(self, results):
        cols = (
            ["dispatch_num"]
            + list(self.sampled_config_dicts[0].keys())
            + ["acc_vals", "acc", "loss", "acc_std", "loss_std"]
        )
        for chunk in range(len(results)):
            for i in range(len(results[chunk][0])):
                results[chunk][0][i] = sigfig.round(results[chunk][0][i], sigfigs=3)
        pt = PrettyTable(cols, title=" --- Final Results --- ")
        pt.max_table_width = self.max_output_width
        rows = [
            [i]
            + list(self.sampled_config_dicts[i].values())
            + [results[i][0]]
            + [sigfig.round(x, sigfigs=3) for x in list(results[i])[1:]]
            for i in range(len(self.sampled_config_dicts))
        ]
        rows.sort(key=lambda x: (-x[-4], x[-2], x[-3], x[-1]))
        pt.add_rows(rows)
        pt.sortby
        logger.valinfo(pt)

    @staticmethod
    def table_intermediates(config, *args):
        cols = list(config.keys()) + ["acc_vals", "acc", "loss", "acc_std", "loss_std"]
        pt = PrettyTable(cols, title=" --- Config Results --- ")
        acc_vals, acc, loss, acc_std, loss_std = args
        for i in range(len(acc_vals)):
            acc_vals[i] = sigfig.round(acc_vals[i], sigfigs=3)
        pt.add_row(
            list(config.values()) + [acc_vals] + [sigfig.round(x, sigfigs=3) for x in [acc, loss, acc_std, loss_std]]
        )
        logger.info(pt)

    @staticmethod
    def get_config_dict(config_key):
        oldcwd = os.getcwd()
        where_am_i = pathlib.Path(__file__).parent.resolve()
        os.chdir(where_am_i)
        os.chdir("../models/json")
        with open("hp_tuning_grids.json", "r") as f:
            d = json.load(f)

        os.chdir(oldcwd)

        desired_conf = d[config_key]

        for hp in desired_conf:
            if "_@pyex_" in desired_conf[hp]:
                loc = {}
                exec(desired_conf[hp].replace("_@pyex_", "temp = "), globals(), loc)
                desired_conf[hp] = loc["temp"]

        return desired_conf

    def display_intermediates(self, cmdl):
        try:
            # cols = ["dispatch_num"] + list(conf.keys()) + ["acc_vals", "acc", "loss", "acc_std", "loss_std"]
            # pt = PrettyTable(cols, title=" --- Intermediate Results --- ")
            # pt.max_table_width = self.max_output_width
            # kwargs = {}
            # acc_vals, acc, loss, acc_std, loss_std
            res = self.train_fn(cmdl)
            # for i in range(len(acc_vals)):
            #     acc_vals[i] = sigfig.round(acc_vals[i], sigfigs=3)
            # pt.add_row(
            #     [dispatch_num]
            #     + list(conf.values())
            #     + [acc_vals]
            #     + [sigfig.round(x, sigfigs=3) for x in [acc, loss, acc_std, loss_std]]
            # )
            # logger.status(pt)
            return res  # acc_vals, acc, loss, acc_std, loss_std
        except KeyboardInterrupt:
            logger.status("My KeyboardInterrupt")
            raise KeyboardInterrupt


def ll_sizes():
    LL_SIZES = [1, 10, 25, 50, 100, 250, 500]
    chunks = [[[x] for x in LL_SIZES]]
    MAX_LAYERS = 3
    partials = [p for p in chunks[0]]
    for _ in range(1, MAX_LAYERS):
        new = []
        for p in partials:
            for ll_size in LL_SIZES:
                new.append(p + [ll_size])
        chunks.append(new)
        partials = new

    out = list(itertools.chain(*[chunk * len(LL_SIZES) ** (MAX_LAYERS - c - 1) for c, chunk in enumerate(chunks)]))

    theoretical_length = MAX_LAYERS * (len(LL_SIZES) ** MAX_LAYERS)
    assert len(out) == theoretical_length, f"got {len(out)} but wanted {theoretical_length}"
    return out


def get_one_layer_cnn(seq_len: int, kern_range: Collection, out_range: Collection, channels: Collection):
    kern_options = []
    pool_options = []
    channel_options = []

    for kern in kern_range:
        assert int(kern) == kern
        assert kern >= 1
        for out in out_range:
            assert int(out) == out
            assert out >= 1
            kern_out = seq_len - kern + 1
            pool_stride_l = (kern_out) / out
            pool_stride_r = (kern_out) / (out + 1 - 1e-8)
            if int(pool_stride_l) != int(pool_stride_r):
                kern_options.append(kern)
                pool_options.append(out)

    kern_options *= len(channels)
    pool_options *= len(channels)
    assert len(kern_options) == len(pool_options), f"got {len(kern_options)=} != {len(pool_options)=}"

    channel_options = [x for x in channels for _ in range(len(kern_options) // len(channels))]

    assert len(kern_options) == len(channel_options), f"{len(kern_options)=} != {len(channel_options)=}"

    return list(zip(kern_options, pool_options, channel_options))


if __name__ == "__main__":
    from ..models.individual_classifiers import main as train_main

    from ..models.GroupClassifier import (
        PseudoSiteGroupClassifier,
    )

    import __main__

    setattr(__main__, "PseudoSiteGroupClassifier", PseudoSiteGroupClassifier)

    cnn_kin_one_layer_options = get_one_layer_cnn(4128, *[np.unique(np.logspace(4, 8, 5, base=2, dtype=int))] * 3)
    cnn_site_one_layer_options = get_one_layer_cnn(
        15, list(range(1, 16, 3)), list(range(1, 16, 3)), np.unique(np.logspace(4, 8, 5, base=2, dtype=int))
    )

    model_params = {
        "model_class": ["KinaseSubstrateRelationshipLSTM"],
        "linear_layer_sizes": ll_sizes(),
        "emb_dim_kin": [1] + list(range(2, 65, 10)),
        "emb_dim_site": [1] + list(range(2, 65, 10)),
        "dropout_pr": sigfig_iter(np.arange(0, 0.66, 0.05), 3),
        "attn_out_features": list(set(np.linspace(64, 640, 5).astype(int))),
        "site_param_dict": [
            {"kernels": [kern], "out_lengths": [out], "out_channels": [chan]}
            for kern, out, chan in cnn_site_one_layer_options
        ],
        "kin_param_dict": [
            {"kernels": [kern], "out_lengths": [out], "out_channels": [chan]}
            for kern, out, chan in cnn_kin_one_layer_options
        ],
        "num_recur_kin": list(range(1, 8, 2)),
        "num_recur_site": list(range(1, 8, 2)),
        "hidden_features_site": [1] + list(range(5, 105, 20)),
        "hidden_features_kin": [1] + list(range(5, 105, 20)),
    }

    training_params = {
        "lr_decay_amount": sigfig_iter(np.arange(0.5, 1.05, 0.05), 2),
        "lr_decay_freq": list(range(1, 9)),
        "num_epochs": list(range(5, 11)),
        "metric": ["roc"],
    }
    interface_params = {
        "loss_fn": ["torch.nn.BCEWithLogitsLoss"],
        "optim": ["torch.optim.Adam"],
        "model_summary_name": ["../architectures/architecture (IC-DEFAULT).txt"],
        "lr": sigfig_iter(np.logspace(-5, -1, 5), 3),
        "batch_size": sorted(list(set(np.logspace(3, 10, 8, base=2).astype(int)))),
        "n_gram": [1],
    }

    all_params = training_params | interface_params | model_params
    which_map = collections.defaultdict(list)
    for k, v in all_params.items():
        for param_dict in ["model_params", "training_params", "interface_params"]:
            if k in eval(param_dict):
                which_map[param_dict].append(k)

    cmdl = [
        "--train",
        "data/raw_data_31834_formatted_65_26610.csv",
        "--val",
        "data/raw_data_6500_formatted_95_5698.csv",
        "--device",
        "cpu",
        "--pre-trained-gc",
        "bin/deepks_gc_weights.1.cornichon",
        "--groups",
        "TK",
    ]

    st = SimpleTuner(
        num_sim_procs=1,
        train_fn=train_main,
        config_dict=all_params,
        which_map=which_map,
        num_samples=1,
        random_seed=84,
        collapse=[[]],
        num_gpu=None,
        max_output_width=512,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        all_args = st.generate_args_for_go(cmdl, tmpdir)
        st.go(all_args)
