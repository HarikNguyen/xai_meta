import torch
import torch.nn as nn
from models import Conv4
from loaders import get_dataloader


def warm_up(config):
    # Parse config
    ds_cfg = config["dataset"]
    dl_cfg = config["dataloader"]
    algo_cfg = config["algo"]

    # Define device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define datasets
    train_loader = get_dataloader(
        data_root=ds_cfg["train_root"],
        dataset=ds_cfg["train_name"],
        dataset_type="train",
        num_workers=dl_cfg["num_workers"],
        sample={
            "metatrain_iterations": dl_cfg["metatrain_iterations"],
            "n_way": dl_cfg["n_way"],
            "k_shot": dl_cfg["k_shot"],
            "k_query": dl_cfg["k_query"],
            "meta_batch_size": dl_cfg["meta_batch_size"],
            "shuffle": True,
        },
    )

    val_loader = get_dataloader(
        data_root=ds_cfg["val_root"],
        dataset=ds_cfg["val_name"],
        dataset_type="val",
        num_workers=dl_cfg["num_workers"],
        sample={
            "metatrain_iterations": dl_cfg["metatrain_iterations"] // dl_cfg["val_after"] + 1,
            "n_way": dl_cfg["n_way"],
            "k_shot": dl_cfg["k_shot"],
            "k_query": dl_cfg["k_query"],
            "meta_batch_size": dl_cfg["meta_batch_size"],
            "shuffle": True,
        },
    )

    metatest_iterations = dl_cfg["metatest_iterations"] // dl_cfg["metatest_batch_size"] # each task only use for 1 iteration
    def _get_test_loader(root, name):
        return get_dataloader(
            data_root=root,
            dataset=name,
            dataset_type="test",
            num_workers=dl_cfg["num_workers"],
            sample={
                "metatrain_iterations": metatest_iterations,
                "n_way": dl_cfg["n_way"],
                "k_shot": dl_cfg["k_shot"],
                "k_query": dl_cfg["test_k_query"],
                "meta_batch_size": dl_cfg["metatest_batch_size"],  # really equal (metatrain_iterations = 600 || meta_batch_size = 1)
                "shuffle": True,
            },
            seed=42,
        )
    
    test_loader = _get_test_loader(ds_cfg["test_root"], ds_cfg["test_name"])
    if ds_cfg.get("ood_explain_root") and ds_cfg.get("explain_root"):
        explain_loader = _get_test_loader(ds_cfg["explain_root"], ds_cfg["explain_name"])
        ood_explain_loader = _get_test_loader(ds_cfg["ood_explain_root"], ds_cfg["ood_explain_name"])
    else:
        explain_loader = None
        ood_explain_loader = None

    # Define model conf
    baselearner_args = {
        "device": device,
        "train_classes": dl_cfg["n_way"],
        "criterion": nn.CrossEntropyLoss(),
    }

    algo_conf = algo_cfg.copy() # copy from yaml
    algo_conf.update({
        "baselearner_fn": Conv4,
        "baselearner_args": baselearner_args,
        "optim_fn": torch.optim.Adam,
        "device": device,
        "train_batch_size": dl_cfg["meta_batch_size"],
        "test_batch_size": 1,
    })

    return train_loader, val_loader, test_loader, explain_loader, ood_explain_loader, algo_conf
