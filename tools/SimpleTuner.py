import collections
import multiprocessing, itertools, random, time, os, sigfig, signal, warnings, re, json, pathlib, functools
import numpy as np
from typing import Iterable, MutableSequence
from prettytable import PrettyTable
from pprint import pprint

from ..config.root_logger import get_logger

logger = get_logger()


class SimpleTuner:
    def __init__(
        self,
        num_sim_procs,
        train_fn,
        config_dict,
        which_map,
        num_samples,
        random_seed=42,
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
            "Not all hyperparameters are in a mutable sequence or numpy array."
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
            f"{num_samples:,} configurations out of a possible"
            f" {np.product([len(v) for v in config_dict.values()]):,} were randomly sampled."
        )

        k_outer = -1
        for d in self.sampled_config_dicts:
            splits = []
            for k in d:
                if re.search(r"group__\[\[(.+\+.+)\]\]__", k):
                    ksub = re.sub(r"group__\[\[(.+\+.+)\]\]__", r"\1", k)
                    splits = ksub.split("+")
                k_outer = k

            for i, hp in enumerate(splits):
                d[hp] = d[k_outer][i]

            if len(splits) != 0:
                del d[k_outer]

        self._print_combinations()

    def _print_combinations(self):
        pt = PrettyTable(["dispatch_num"] + list(self.sampled_config_dicts[0].keys()), title="Configs to Run")
        rows = [[i] + list(self.sampled_config_dicts[i].values()) for i in range(len(self.sampled_config_dicts))]
        pt.max_table_width = self.max_output_width
        pt.add_rows(rows)
        logger.valinfo(pt)

    def init_pool(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def go(self, *args):
        ret = None
        with multiprocessing.Pool(self.num_sim_procs, initializer=self.init_pool) as pool:
            try:
                results = pool.starmap(
                    self.display_intermediates,
                    [
                        [cd, dispatch_num, *args]
                        for cd, dispatch_num in zip(self.sampled_config_dicts, range(len(self.sampled_config_dicts)))
                    ],
                )
            except KeyboardInterrupt:
                logger.status("\nQuitting...\n")
                pool.terminate()
                pool.join()
                exit(1)
            ret = [x for x in results]

        self.display_final_results(ret)

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

    def display_intermediates(self, conf, dispatch_num, *args):
        try:
            cols = ["dispatch_num"] + list(conf.keys()) + ["acc_vals", "acc", "loss", "acc_std", "loss_std"]
            pt = PrettyTable(cols, title=" --- Intermediate Results --- ")
            pt.max_table_width = self.max_output_width
            kwargs = {"process_device": f"cuda:{dispatch_num % self.num_gpu}"}
            acc_vals, acc, loss, acc_std, loss_std = self.train_fn(conf, *args, **kwargs)
            for i in range(len(acc_vals)):
                acc_vals[i] = sigfig.round(acc_vals[i], sigfigs=3)
            pt.add_row(
                [dispatch_num]
                + list(conf.values())
                + [acc_vals]
                + [sigfig.round(x, sigfigs=3) for x in [acc, loss, acc_std, loss_std]]
            )
            logger.status(pt)
            return acc_vals, acc, loss, acc_std, loss_std
        except KeyboardInterrupt:
            logger.status("My KeyboardInterrupt", flush=True)
            raise KeyboardInterrupt


def ll_sizes():
    LL_SIZES = [1, 10, 25, 50, 100, 250, 500]
    possible_ll_sizes = [[x] for x in LL_SIZES]
    MAX_LAYERS = 3
    partials = [p for p in possible_ll_sizes]
    for _ in range(MAX_LAYERS - 1):  # [1, >2]
        new = []
        for p in partials:
            for ll_size in LL_SIZES:
                new.append(p + [ll_size])
        possible_ll_sizes += new
        partials = new

    theoretical_length = 0
    for i in range(1, MAX_LAYERS + 1):
        theoretical_length += len(LL_SIZES) ** i
    assert len(possible_ll_sizes) == theoretical_length, f"got {len(possible_ll_sizes)} but wanted {theoretical_length}"
    return possible_ll_sizes


def get_one_layer_cnn(seq_len: int, kern_range: Iterable, out_range: Iterable, pass_through_channel_range: Iterable):
    kern_options = []
    pool_options = []

    for kern in kern_range:
        assert int(kern) == kern
        for out in out_range:
            assert int(out) == out
            kern_out = seq_len - kern
            pool_stride_l = (kern_out - 1) / out
            pool_stride_r = (kern_out - 1) / (out + 1)
            if int(pool_stride_l) != int(pool_stride_r):
                kern_options.append(kern)
                pool_options.append(out)

    return list(zip(kern_options, pool_options, [x for x in pass_through_channel_range]))


get_one_layer_cnn(4128, *[np.unique(np.logspace(0, 8, 16, base=2, dtype=int))] * 3)


if __name__ == "__main__":
    from ..models.individual_classifiers import main as train_main

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

    cnn_one_layer_options = get_one_layer_cnn(4128, *[np.unique(np.logspace(0, 8, 16, base=2, dtype=int))] * 3)

    model_params = {
        "model_class": ["KinaseSubstrateRelationshipLSTM"],
        "linear_layer_sizes": ll_sizes(),
        "emb_dim_kin": [1] + list(range(2, 65, 4)),
        "emb_dim_site": [1] + list(range(2, 65, 4)),
        "dropout_pr": np.round(np.arange(0, 0.66, 0.05), 2),
        "attn_out_features": list(set(np.linspace(16, 320, 10).astype(int))),
        "site_param_dict": [
            {"kernels": [kern], "out_lengths": [out], "out_channels": [chan]}
            for kern, out, chan in cnn_one_layer_options
        ],
        "kin_param_dict": [
            {"kernels": [kern], "out_lengths": [out], "out_channels": [chan]}
            for kern, out, chan in cnn_one_layer_options
        ],
        "num_recur_kin": list(range(1, 21, 2)),
        "num_recur_site": list(range(1, 21, 2)),
        "hidden_features_site": [1] + list(range(5, 105, 10)),
        "hidden_features_kin": [1] + list(range(5, 105, 10)),
    }

    training_params = {
        "lr_decay_amount": np.round(np.arange(0.5, 1.05, 0.05), 2),
        "lr_decay_freq": list(range(10)),
        "num_epochs": list(range(10)),
        "metric": ["roc"],
    }
    interface_params = {
        "loss_fn": ["torch.nn.BCEWithLogitsLoss"],
        "optim": ["torch.optim.Adam"],
        "model_summary_name": ["../architectures/architecture (IC-DEFAULT).txt"],
        "lr": np.logspace(-5, 0, 15),
        "batch_size": sorted(list(set(np.logspace(0, 12, 10, base=1.8).astype(int)))),
        "n_gram": [1],
    }

    all_params = training_params | interface_params | model_params
    which_map = collections.defaultdict(list)
    for k, v in all_params.items():
        for param_dict in ["model_params", "training_params", "interface_params"]:
            if k in eval(param_dict):
                which_map[param_dict].append(k)

    st = SimpleTuner(
        num_sim_procs=1,
        train_fn=functools.partial(train_main, cmdl),
        config_dict=all_params,
        which_map=which_map,
        num_samples=100,
        random_seed=42,
        collapse=[[]],
        num_gpu=None,
        max_output_width=512,
    )
    st.go()
