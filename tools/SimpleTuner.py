import multiprocessing
import itertools
import random
import time
import os
from prettytable import PrettyTable
import sigfig
import signal
import warnings
import re
import json
import pathlib
from . import file_names


class SimpleTuner:
    def __init__(
        self,
        num_sim_procs,
        train_fn,
        config_dict,
        num_samples,
        random_seed=0,
        collapse=[[]],
        num_gpu=None,
        max_output_width=512,
    ):
        random.seed(random_seed)
        if num_gpu is None:
            self.num_gpu = num_sim_procs
        else:
            self.num_gpu = num_sim_procs
        self.max_output_width = max_output_width
        random.seed(0)
        self.num_sim_procs = num_sim_procs
        self.train_fn = train_fn
        self.config_dict = config_dict

        assert all(isinstance(x, list) for x in collapse), "`collapse` must be a list of lists."
        if collapse != [[]]:
            for _set_ in collapse:
                config_dict[f'group__[[{"+".join(_set_)}]]__'] = list(zip(*[list(config_dict[x]) for x in _set_]))
                for x in _set_:
                    del config_dict[x]

        assert all(
            isinstance(x, list) for x in config_dict.values()
        ), "Not all hyperparameters are in a list. (E.g., `'num_epochs':1` instead of `'num_epochs':[1]`)"

        cart_prod = list(itertools.product(*list(config_dict.values())))
        self.num_samples = min(num_samples, len(cart_prod))
        if self.num_samples != num_samples:
            warnings.warn(
                "num_samples exceeds the number of possible combinations. Using all possible combinations (%d) instead."
                % len(cart_prod)
            )
        samples = random.sample(cart_prod, k=self.num_samples)
        self.sampled_config_dicts = [
            {k: v for k, v in zip(self.config_dict.keys(), samples[i])} for i in range(len(samples))
        ]
        not_chosen = [x for x in cart_prod if x not in samples]
        not_chosen_config_dicts = [
            {k: v for k, v in zip(self.config_dict.keys(), not_chosen[i])} for i in range(len(not_chosen))
        ]
        print(
            f"{len(samples)} out of {len(samples) + len(not_chosen)} were picked. See"
            f" {(fn := f'../logs/leftover-configs-{file_names.get()}.log')} for leftover configurations."
        )
        with open(fn, "w") as f:
            f.write("\n".join([str(x) for x in not_chosen_config_dicts]) + "\n")
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
        print(pt, flush=True)

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
                print("\nQuitting...\n", flush=True)
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
        print(pt, flush=True)

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
        print("", flush=True)
        print(pt, flush=True)
        print("", flush=True)

    @staticmethod
    def get_config_dict(config_key):
        oldcwd = os.getcwd()
        where_am_i = pathlib.Path(__file__).parent.resolve()
        os.chdir(where_am_i)
        os.chdir("../models/json")
        with open("hp_tuning_grids.json", "r") as f:
            d = json.load(f)

        os.chdir(oldcwd)

        try:
            desired_conf = d[config_key]
        except AssertionError as ae:
            print(ae)
            raise RuntimeError()

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
            print("", flush=True)
            print(pt, flush=True)
            print("", flush=True)
            return acc_vals, acc, loss, acc_std, loss_std
        except KeyboardInterrupt:
            print("MY KeyboardInterrupt", flush=True)
            raise KeyboardInterrupt


def mainfn(cd):
    strr = (
        f"My batch size is {cd['batch_size']}; "
        f"My epochs is {cd['num_epochs']}; "
        f"My lldim is {cd['ll_dim']}; "
        f"My lr is {cd['learning_rate']}"
    )
    time.sleep(random.randrange(10, 30))
    print("My pid is", os.getpid(), flush=True)
    return strr


def mainfn2(X):
    time.sleep(random.randrange(1, 3))
    return random.randrange(50, 100), random.randrange(50, 100), random.randrange(50, 100)


if __name__ == "__main__":
    cf = {
        "learning_rate": [0.003],
        "batch_size": [43, 46, 50, 54, 57],
        "ll_size": [20, 25, 30],
        "emb_dim": [22],
        "num_epochs": [15, 20],
        "n_gram": [1],
        "lr_decay_amt": [0.95, 0.96, 0.97],
        "lr_decay_freq": [1],
        "num_conv_layers": [1],
        # "site_param_dict": [
        #     {"kernels": [skern], "out_lengths": [o], "out_channels": [c]}
        #     for skern, o, c in [(6, 10, 14), (8, 8, 20), (10, 6, 22)]
        # ],
        # "kin_param_dict": [
        #     {"kernels": [kkern], "out_lengths": [o], "out_channels": [c]}
        #     for kkern, o, c in [(60, 10, 14), (80, 8, 20), (100, 6, 22)]
        # ],
        "dropout_pr": [0.3, 0.32, 0.34, 0.37, 0.4],
    }

    st = SimpleTuner(1, None, cf, 20)  # collapse=[["site_param_dict", "kin_param_dict"]])
    st.go()
