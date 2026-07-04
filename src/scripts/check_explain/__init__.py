import os
import numpy as np
import pandas as pd

from ..utils import load_checkpoint, prepare_plots_dir, build_explainer
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
    prepare_plots_dir(log_dir)

    # Load checkpoint
    load_checkpoint(algo_mgr, checkpoint_dir, use_best, use_last)

    # Define explainer
    explainer = build_explainer(algo, algo_class, algo_mgr, device)

    if method == "biADT":
        pdas, ndas, combineds = compute_bidirectional_faithfulness(
            explainer, test_loader, T=T
        )
        res_df = pd.DataFrame({
            "PDAS": pdas,
            "NDAS": ndas,
            "Combined": combineds
        })
        _save_results_to_csv(res_df, "biADT", log_dir)

    elif method == "sanity_params":
        results = sanity_check_params(explainer, test_loader, T=T)
        res_df = pd.DataFrame(results)
        mean_res_df = pd.DataFrame(res_df.apply(lambda col: np.mean(col.to_list(), axis=0)))

        _save_results_to_csv(res_df, "sanity_params", log_dir, mean_df=mean_res_df)

    elif method == "sanity_support_set":
        if ood_test_loader is None:
            raise ValueError("OOD test loader is None. Please provide a valid OOD test loader.")
        results = sanity_check_support_set(explainer, test_loader, ood_test_loader, T=T)
        noisy_check_df = pd.DataFrame(results["noisy_check"])
        hard_check_df = pd.DataFrame(results["hard_check"])
        ood_check_df = pd.DataFrame(results["ood_check"])

        _save_results_to_csv(noisy_check_df, "sanity_support_set_noisy", log_dir)
        _save_results_to_csv(hard_check_df, "sanity_support_set_hard", log_dir)
        _save_results_to_csv(ood_check_df, "sanity_support_set_ood", log_dir)

    else:
        raise NotImplementedError(f"Method {method} not implemented.")

def _save_results_to_csv(res_df, name, log_dir, mean_df=None):
    res_df.to_csv(os.path.join(log_dir, f"{name}_results.csv"), index=False)
    (mean_df if mean_df is not None else res_df.mean()).to_csv(
        os.path.join(log_dir, f"{name}_results_mean.csv"), index=True
    )
