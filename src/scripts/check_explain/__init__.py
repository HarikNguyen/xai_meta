import os
import shutil
import math
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from interpreters import FAMAExplainer

from .bi_adt import compute_bidirectional_faithfulness
from .sanity_params import sanity_check_params
from .sanity_support_set import sanity_check_support_set

def check_explain(
    algo,
    algo_class,
    test_loader,
    ood_test_loader,
    algo_conf,
    method=None,
    use_best=False,
    use_last=True,
    checkpoint_dir="checkpoints",
    log_dir="logs",
):
    if test_loader is None:
        raise ValueError("Test loader is None. Please provide a valid test loader.")
    
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
        res_df = pd.DataFrame({
            "PDAS": pdas,
            "NDAS": ndas,
            "Combined": combineds
        })
        res_df.to_csv(os.path.join(log_dir, "biADT_results.csv"), index=False)
        res_df.mean().to_csv(os.path.join(log_dir, "biADT_results_mean.csv"), index=True)

    elif method == "sanity_params":
        results = sanity_check_params(explainer, test_loader, T=T)
        res_df = pd.DataFrame(results)
        mean_res_df = pd.DataFrame(res_df.apply(lambda col: np.mean(col.to_list(), axis=0)))
        res_df.to_csv(os.path.join(log_dir, "sanity_params_results.csv"), index=False)
        res_df.mean().to_csv(os.path.join(log_dir, "sanity_params_results_mean.csv"), index=True)

    elif method == "sanity_support_set":
        if ood_test_loader is None:
            raise ValueError("OOD test loader is None. Please provide a valid OOD test loader.")
        results = sanity_check_support_set(explainer, test_loader, ood_test_loader, T=T)
        noisy_check_df = pd.DataFrame(results["noisy_check"])
        hard_check_df = pd.DataFrame(results["hard_check"])
        ood_check_df = pd.DataFrame(results["ood_check"])

        noisy_check_df.to_csv(os.path.join(log_dir, "sanity_support_set_noisy_check_results.csv"), index=False)
        hard_check_df.to_csv(os.path.join(log_dir, "sanity_support_set_hard_check_results.csv"), index=False)
        ood_check_df.to_csv(os.path.join(log_dir, "sanity_support_set_ood_check_results.csv"), index=False)

        noisy_check_df.mean().to_csv(os.path.join(log_dir, "sanity_support_set_noisy_check_results_mean.csv"), index=True)
        hard_check_df.mean().to_csv(os.path.join(log_dir, "sanity_support_set_hard_check_results_mean.csv"), index=True)
        ood_check_df.mean().to_csv(os.path.join(log_dir, "sanity_support_set_ood_check_results_mean.csv"), index=True)

    else:
        raise NotImplementedError(f"Method {method} not implemented.")
