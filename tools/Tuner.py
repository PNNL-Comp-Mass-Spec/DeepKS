"""Module to perform hyperparameter tuning on `torch.nn.Module` -based neural networks."""

import collections
import itertools, random, os, sigfig, signal, json, pathlib
import pandas as pd
import tempfile
import numpy as np
from typing import Any, Callable, Collection, Iterable, Protocol, Union
from prettytable import PrettyTable
from beautifultable import BeautifulTable
from numpyencoder import NumpyEncoder
from abc import ABC, abstractmethod

from DeepKS.tools.custom_tqdm import CustomTqdm

from ..config.logging import get_logger
from .file_names import get as get_file_name

logger = get_logger()
"""The logger for this module."""

join_first = lambda levels, x: os.path.join(pathlib.Path(__file__).parent.resolve(), *[".."] * levels, x)


def sigfig_iter(iterable: Iterable, sigfigs: int = 3):
    """Rounds all floats in an iterable to a certain number of significant figures.

    Parameters
    ----------
    iterable :
        The iterable to round.
    sigfigs : int, optional
        The number of significant figures to round the elements to, by default 3

    Returns
    -------
    list
        The rounded iterable.

    Raises
    ------
    ValueError
        If the iterable contains non-floats.
    """
    res = []
    for x in iterable:
        if isinstance(x, float):
            res.append(sigfig.round(x, sigfigs))
        else:
            raise ValueError(f"sigfig_iter only works on floats, not {type(x)}")
    return res


class Tuner(Protocol):
    num_gpu: int
    max_output_width: int
    num_sim_procs: int
    train_fn: Callable
    config_dict: dict[str, Any]
    sampled_config_dicts: list

    def __init__(
        self,
        num_sim_procs: int,
        train_fn: Callable,
        config_dict: dict[str, Any],
        num_samples: int,
        random_seed: int = 84,
        collapse=[[]],
        num_gpu: Union[None, int] = None,
        max_output_width=512,
        start_at=0,
        **kwargs,
    ) -> None:
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
        for i in range(start_at, num_samples):
            running = {}
            for k, v in config_dict.items():
                if isinstance(v, np.ndarray):
                    choice_ = random.choice(v.tolist())
                else:
                    choice_ = random.choice(v)
                running[k] = choice_
            if i >= start_at:
                self.sampled_config_dicts.append(running)

        if num_samples > 1:
            addl_s = "s"
        else:
            addl_s = ""
        logger.info(
            f"{num_samples:,} configuration{addl_s} out of a possible ~"
            f" 10^{np.sum([np.log10(len(v)) for v in config_dict.values()]):,.1f} were randomly sampled."
        )

        self.print_combinations()

    def print_combinations(self):
        preview = {f"cfg # {i}": v for i, v in enumerate(self.sampled_config_dicts[:20])}
        df = pd.DataFrame.from_dict(preview, orient="index")
        # logger.debug(f"{df=}")
        df["cfg #"] = df.index
        df.insert(0, "cfg #", df.pop("cfg #"))
        bt = BeautifulTable()
        include_cols = [
            i for i in range(len(df.columns)) if df.iloc[:, i].apply(lambda x: str(x)).nunique() > 1 or len(df) == 1
        ]

        bt.columns.header = df.columns[include_cols].tolist()
        for row in df.values[:, include_cols].tolist():
            bt.rows.append(row)
        assert len(bt.columns.header)
        bt.columns.width = (os.get_terminal_size().columns - 1 - len(bt.columns.header)) // (len(bt.columns.header))

        logger.info("\n" + str(bt))

    def init_pool(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    @abstractmethod
    def go(self, args_collection: list[list], **kwargs):
        ...

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

    def get_result_from_config(self, cmdl, **kwargs):
        try:
            res = self.train_fn(cmdl, **kwargs)
            return res
        except KeyboardInterrupt:
            logger.status("My KeyboardInterrupt")
            raise KeyboardInterrupt


######## General Tuner Above ########
####### Specific Tuner(s) Below #######


class BasicTuner(Tuner):
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
        start_at=0,
    ):
        super().__init__(
            num_sim_procs,
            train_fn,
            config_dict,
            num_samples,
            random_seed,
            collapse,
            num_gpu,
            max_output_width,
            start_at,
        )

        self.model_params = []
        for sampled_config_dict in self.sampled_config_dicts:
            d = {}
            for k, v in sampled_config_dict.items():
                if k in which_map["model_params"]:
                    if type(v) == np.int64:
                        d[k] = int(v)
                    else:
                        d[k] = v
            self.model_params.append({"default": d})

        self.training_params = []
        for sampled_config_dict in self.sampled_config_dicts:
            d = {}
            for k, v in sampled_config_dict.items():
                if k in which_map["training_params"]:
                    if type(v) == np.int64:
                        d[k] = int(v)
                    else:
                        d[k] = v
            self.training_params.append({"default": d})

        self.interface_params = []
        for sampled_config_dict in self.sampled_config_dicts:
            d = {}
            for k, v in sampled_config_dict.items():
                if k in which_map["interface_params"]:
                    if type(v) == np.int64:
                        d[k] = int(v)
                    else:
                        d[k] = v
            self.interface_params.append({"default": d})

        assert (
            len(self.model_params) == len(self.training_params) == len(self.interface_params) == num_samples - start_at
        )

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

    def go(self, args_collection, **kwargs):
        scores = []
        first_file_name = get_file_name(
            prefix="a_tuning", suffix="json", directory=os.path.join(join_first(0, ""), "tunings")
        )
        # with multiprocessing.Pool(self.num_sim_procs, initializer=self.init_pool) as pool:
        #     try:
        #         results = pool.starmap(
        #             self.get_result_from_config,
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
        for i, args in CustomTqdm(
            enumerate(args_collection), desc="Tuning Progress", total=len(args_collection), position=5
        ):
            new_score, notes = self.get_result_from_config(args, **kwargs)
            scores.append((i, new_score, notes))
            scores.sort(key=lambda x: x[1], reverse=True)
            # logger.debug(f"{scores=}")
            res_json = {
                f"cfg # {scores[j][0]}": {
                    "Score": scores[j][1],
                    "Model Params": self.model_params[j],
                    "Training Params": self.training_params[j],
                    "Interface Params": self.interface_params[j],
                    "Notes": scores[j][2],
                }
                for j in range(i + 1)
            }
            # logger.debug(f"{res_json=}")
            logger.status("Saving safety temp file")
            with open(f"{first_file_name}.gitig-temp", "w") as tf:
                json.dump(res_json, tf, indent=3, cls=NumpyEncoder)
            logger.status("Overwriting main file")
            with open(first_file_name, "w") as f:
                json.dump(res_json, f, indent=3, cls=NumpyEncoder)
        logger.status("Done!")

    @staticmethod
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

    @staticmethod
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


def main(cmdl: list[str], train_fn: Callable, num_samples: int = 200, max_epoch: int = 20):
    cnn_kin_one_layer_options = BasicTuner.get_one_layer_cnn(
        4128, *[np.unique(np.logspace(3, 7, 5, base=2, dtype=int))] * 3
    )
    cnn_site_one_layer_options = BasicTuner.get_one_layer_cnn(
        15, list(range(1, 16, 3)), list(range(1, 16, 3)), np.unique(np.logspace(4, 8, 5, base=2, dtype=int))
    )

    model_params = {
        "model_class": ["KinaseSubstrateRelationshipLSTM"],
        "linear_layer_sizes": BasicTuner.ll_sizes(),
        "emb_dim_kin": [4, 24, 44, 64, 84],
        "emb_dim_site": [4, 24, 44, 64, 84],
        "dropout_pr": [0, 0.1, 0.3, 0.5, 0.7],
        "attn_out_features": [64, 128, 256, 512],
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
        "lr_decay_amount": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "lr_decay_freq": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "num_epochs": sorted(list(set([min(x, max_epoch) for x in [1, 3, 5, 10, 15, 20]]))),
        "metric": ["roc"],
    }
    interface_params = {
        "loss_fn": ["torch.nn.BCEWithLogitsLoss"],
        "optim": ["torch.optim.Adam"],
        "model_summary_name": ["../architectures/architecture (IC-DEFAULT).txt"],
        "lr": sigfig_iter(np.logspace(-5, -1, 5), 3),
        "batch_size": [8, 32, 128, 256, 512],
        "n_gram": [1],
    }

    all_params = training_params | interface_params | model_params
    which_map = collections.defaultdict(list)
    for k in all_params.keys():
        for param_dict in ["model_params", "training_params", "interface_params"]:
            if k in eval(param_dict):
                which_map[param_dict].append(k)

    st = BasicTuner(
        num_sim_procs=1,
        train_fn=train_fn,
        config_dict=all_params,
        which_map=which_map,
        num_samples=num_samples,
        start_at=0,
        random_seed=142,
        collapse=[[]],
        num_gpu=None,
        max_output_width=512,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        all_args = st.generate_args_for_go(cmdl, tmpdir)
        st.go(all_args, loss_chances=5, loss_below=-np.emath.logn(np.e, 0.5), val_le=0.52)


if __name__ == "__main__":
    from ..models.individual_classifiers import main as train_main

    from ..models.GroupClassifier import (
        PseudoSiteGroupClassifier,
    )

    import __main__

    setattr(__main__, "PseudoSiteGroupClassifier", PseudoSiteGroupClassifier)
    cmdl = [
        "--train",
        "data/raw_data_31834_formatted_65_26610.csv",
        "--val",
        "data/raw_data_6500_formatted_95_5698.csv",
        "--device",
        "cuda:4",
        "--pre-trained-gc",
        "bin/deepks_gc_weights.1.cornichon",
        "--groups",
        "NON-TK",
    ]
    main(cmdl, train_main, num_samples=3, max_epoch=5)
