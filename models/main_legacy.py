import os, pathlib, json, torch, numpy as np, torch.nn as nn

from ..tools.tensorize import gather_data
from ..tools.NNInterface import NNInterface
from ..tools import file_names
from ..tools.SimpleTuner import SimpleTuner
from ..tools.model_utils import cNNUtils as cNNUtils
from ..tools.parse import parsing


def perform_k_fold(config, display_within_train=False, process_device="cpu"):
    print(f"Using Device > {process_device} <")
    global NUM_EMBS

    # Hyperparameters
    for x in config:
        exec(f"{x} = {config[x]}")

    tokdict = json.load(open("json/tok_dict.json", "rb"))
    tokdict["-"] = tokdict["<PADDING>"]
    assert train_filename is not None
    (train_loader, _, _, _), info_dict_tr = gather_data(
        train_filename,
        trf=1,
        vf=0,
        tuf=0,
        tef=0,
        train_batch_size=config["batch_size"],
        n_gram=config["n_gram"],
        tokdict=tokdict,
        device=torch.device(process_device),
        maxsize=KIN_LEN,
    )
    assert val_filename is not None
    (_, val_loader, _, _), info_dict_vl = gather_data(
        val_filename,
        trf=0,
        vf=1,
        tuf=0,
        tef=0,
        train_batch_size=config["batch_size"],
        n_gram=config["n_gram"],
        tokdict=tokdict,
        device=torch.device(process_device),
        maxsize=KIN_LEN,
    )
    NUM_EMBS = 22

    results = []
    assert test_filename is not None
    (_, _, _, test_loader), info_dict_te = gather_data(
        test_filename,
        trf=0,
        vf=0,
        tuf=0,
        tef=1,
        n_gram=config["n_gram"],
        tokdict=tokdict,
        device=torch.device(process_device),
        maxsize=KIN_LEN,
    )

    kinase_order = [
        info_dict_tr["kin_orders"]["train"],
        info_dict_vl["kin_orders"]["val"],
        info_dict_te["kin_orders"]["test"],
    ]

    crit = torch.nn.BCEWithLogitsLoss()
    if isinstance(crit, torch.nn.BCEWithLogitsLoss):
        num_classes = 1
    elif isinstance(crit, torch.nn.CrossEntropyLoss):
        num_classes = 2
    else:
        raise RuntimeError("Don't know how many classes to output.")

    torch.manual_seed(3)
    model = KinaseSubstrateRelationshipNN(
        num_classes=num_classes,
        inp_size=NNInterface.get_input_size(train_loader),
        ll1_size=config["ll1_size"],
        ll2_size=config["ll2_size"],
        emb_dim=config["emb_dim"],
        num_conv_layers=config["num_conv_layers"],
        site_param_dict=config["site_param_dict"],
        kin_param_dict=config["kin_param_dict"],
        dropout_pr=config["dropout_pr"],
    ).to(process_device)
    the_nn = NNInterface(
        model,
        crit,
        torch.optim.Adam(model.parameters(), lr=config["learning_rate"]),
        inp_size=NNInterface.get_input_size(train_loader),
        inp_types=NNInterface.get_input_types(train_loader),
        model_summary_name="../architectures/architecture (id-%d).txt" % (cNNUtils.id_params(config)),
        device=torch.device(process_device),
    )

    cutoff = 0.4
    metric = "roc"

    if process_device == "cpu":
        input(
            "WARNING: Running without CUDA. Are you sure you want to proceed? Press any key to proceed. (ctrl + c to"
            " quit)\n"
        )

    results.append(
        the_nn.train(
            train_loader,
            lr_decay_amount=config["lr_decay_amt"],
            lr_decay_freq=config["lr_decay_freq"],
            num_epochs=config["num_epochs"],
            val_dl=val_loader,
            cutoff=cutoff,
            metric=metric,
        )
    )

    the_nn.test(
        test_loader,
        print_sample_predictions=False,
        cutoff=cutoff,
        metric=metric,
    )

    the_nn.save_model(f"../bin/saved_state_dicts/{(fn := file_names.get())}.pkl")
    the_nn.save_eval_results(test_loader, f"../res/{fn}.json", kin_order=kinase_order[2])
    # Legacy
    # the_nn.get_all_rocs(
    #     train_loader,
    #     val_loader,
    #     test_loader,
    #     test_loader,
    #     savefile="../images/Evaluation and Results/ROC/Preliminary_ROC_Test",
    # )
    # the_nn.get_all_rocs_by_group(
    #     test_loader,
    #     kinase_order[2],
    #     savefile="../images/Evaluation and Results/ROC/ROC_by_group",
    #     kin_fam_grp_file="../data/preprocessing/kin_to_fam_to_grp_817.csv",
    # )
    # the_nn.get_all_conf_mats(
    #     train_loader,
    #     val_loader,
    #     test_loader,
    #     test_loader,
    #     savefile="../images/Evaluation and Results/ROC/CM_",
    #     cutoffs=[0.3, 0.4, 0.5, 0.6],
    # )

    del model, the_nn
    torch.cuda.empty_cache()

    results = np.array(results)
    if display_within_train:
        SimpleTuner.table_intermediates(
            config,
            results[:, 0].tolist(),
            np.mean(results[:, 0]),
            np.mean(results[:, 1]),
            np.std(results[:, 0]),
            np.std(results[:, 1]),
        )
    return (
        results[:, 0].tolist(),
        np.mean(results[:, 0]),
        np.mean(results[:, 1]),
        np.std(results[:, 0]),
        np.std(results[:, 1]),
    )  # accuracy, loss, acc_std, loss_std


if __name__ == "__main__":
    args = parsing()
    train_filename = args["train"]
    val_filename = args["val"]
    test_filename = args["test"]

    cf = {
        "learning_rate": 0.003,
        "batch_size": 64,
        "ll1_size": 50,
        "ll2_size": 25,
        "emb_dim": 22,
        "num_epochs": 1,
        "n_gram": 1,
        "lr_decay_amt": 0.35,
        "lr_decay_freq": 3,
        "num_conv_layers": 1,
        "dropout_pr": 0.4,
        "site_param_dict": {"kernels": [8], "out_lengths": [8], "out_channels": [20]},
        "kin_param_dict": {"kernels": [100], "out_lengths": [8], "out_channels": [20]},
    }
    assert args["device"] is not None
    perform_k_fold(cf, display_within_train=True, process_device=args["device"])
