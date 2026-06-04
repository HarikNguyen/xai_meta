import torch
import torch.nn as nn
from models import Conv4
from loaders import get_dataloader


def warm_up():
    # Parse args

    # Define device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define datasets
    metatrain_iterations = 60000
    val_after = 1000
    metatest_batch_size = 4
    metatest_iterations = 600 // metatest_batch_size # each task only use for 1 iteration

    train_loader = get_dataloader(
        data_root="miniImagenet",
        dataset="miniImagenet",
        dataset_type="train",
        num_workers=2,
        sample={
            "metatrain_iterations": metatrain_iterations,
            "n_way": 5,
            "k_shot": 1,
            "k_query": 15,
            "meta_batch_size": 4,
            "shuffle": True,
        },
    )

    val_loader = get_dataloader(
        data_root="miniImagenet",
        dataset="miniImagenet",
        dataset_type="val",
        num_workers=2,
        sample={
            "metatrain_iterations": metatrain_iterations // val_after + 1,
            "n_way": 5,
            "k_shot": 1,
            "k_query": 15,
            "meta_batch_size": 4,
            "shuffle": True,
        },
    )

    test_loader = get_dataloader(
        data_root="miniImagenet",
        dataset="miniImagenet",
        dataset_type="test",
        num_workers=2,
        seed=42,
        sample={
            "metatrain_iterations": metatest_iterations,
            "n_way": 5,
            "k_shot": 1,
            "k_query": 1,
            "meta_batch_size": metatest_batch_size,  # Really equal (metatrain_iterations = 600 || meta_batch_size = 1)
            "shuffle": True,
        },
    )

    # Define model conf
    baselearner_args = {
        "device": device,
        "train_classes": 5,
        "criterion": nn.CrossEntropyLoss(),
    }

    algo_conf = {
        "train_base_lr": 0.01,
        "base_lr": 0.01,
        "second_order": False,
        "meta_batch_size": 4,
        "baselearner_fn": Conv4,
        "baselearner_args": baselearner_args,
        "optim_fn": torch.optim.Adam,
        "T": 5,
        "T_val": 1,  # Eval on at least one update f
        "T_test": 10,
        "train_batch_size": 4,
        "test_batch_size": 1,
        "lr": 0.001,
        "device": device,
        "batching_eps": False,
        "test_adam": False,
        "grad_clip": 10,
    }

    return train_loader, val_loader, test_loader, algo_conf
