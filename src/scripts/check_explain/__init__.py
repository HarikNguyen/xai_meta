import os
import shutil
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from interpreters import MAMLPostHocExplainer

from .bi_adt import compute_bidirectional_faithfulness


def check_explain(
    algo,
    algo_class,
    test_loader,
    algo_conf,
    method=None,
    use_best=False,
    use_last=True,
    checkpoint_dir="checkpoints",
    log_dir="logs",
):

    # Inits
    algo_mgr = algo_class(**algo_conf)
    T = algo_mgr.T_test
    device = algo_conf.get("device", "cpu")

    # Create dir for plots saving
    plots_dir = os.path.join(log_dir, "plots")
    if os.path.exists(plots_dir):
        shutil.rmtree(plots_dir)
    os.makedirs(plots_dir)

    # Load checkpoint
    if use_best:
        checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")
    elif use_last:
        checkpoint_path = os.path.join(checkpoint_dir, "last_checkpoint.pt")
    else:
        raise ValueError("Please specify --use_last or --use_best")

    print("Loading checkpoint from", checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    algo_mgr.read_file(checkpoint_path)

    # Define explainer
    if algo_class.__name__ == "MAML":
        explainer = FAMAExplainer(algo_mgr, device=device)
    else:
        raise NotImplementedError(f"Algorithm {algo} can not be explained.")

    if method == "biADT":
        pdas, ndas, combineds = compute_bidirectional_faithfulness(
            explainer, test_loader, T=T
        )

        for i in range(len(pdas)):
            print(f"PDAS: {pdas[i]}, NDAS: {ndas[i]}, Combined: {combineds[i]}")
