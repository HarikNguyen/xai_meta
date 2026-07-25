import os
import numpy as np
import pandas as pd

from ..utils import load_checkpoint, prepare_plots_dir, build_explainer, shutdown_executors
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
    illustrate_label=None,
    illustrate_n_tasks=3,
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

    # Illustrative plots (masking grid / corrupted-layer grid / perturbation
    # grid) are opt-in via --illustrate_label, since saving one per task would
    # mean hundreds of images for a full metatest_iterations run. When set,
    # only the first illustrate_n_tasks tasks get a plot, written to
    # check_explain_storage/<illustrate_label>/<method>/ regardless of
    # --log_dir (so runs against different backbones/checkpoints land in the
    # same top-level folder for side-by-side comparison).
    illustrate_dir = None
    if illustrate_label is not None:
        illustrate_dir = os.path.join("check_explain_storage", illustrate_label, method)
        os.makedirs(illustrate_dir, exist_ok=True)

    if method == "biADT":
        pdas, ndas, combineds = compute_bidirectional_faithfulness(
            explainer, test_loader, T=T,
            illustrate_dir=illustrate_dir, illustrate_n_tasks=illustrate_n_tasks,
        )
        res_df = pd.DataFrame({
            "PDAS": pdas,
            "NDAS": ndas,
            "Combined": combineds
        })
        _save_results_to_csv(res_df, "biADT", log_dir)

    elif method == "sanity_params":
        results = sanity_check_params(
            explainer, test_loader, T=T,
            illustrate_dir=illustrate_dir, illustrate_n_tasks=illustrate_n_tasks,
        )
        res_df = pd.DataFrame(results)
        mean_res_df = pd.DataFrame(res_df.apply(lambda col: np.mean(col.to_list(), axis=0)))

        _save_results_to_csv(res_df, "sanity_params", log_dir, mean_df=mean_res_df)

    elif method == "sanity_support_set":
        if ood_test_loader is None:
            raise ValueError("OOD test loader is None. Please provide a valid OOD test loader.")
        results = sanity_check_support_set(
            explainer, test_loader, ood_test_loader, T=T,
            illustrate_dir=illustrate_dir, illustrate_n_tasks=illustrate_n_tasks,
        )
        noisy_check_df = pd.DataFrame(results["noisy_check"])
        hard_check_df = pd.DataFrame(results["hard_check"])
        ood_check_df = pd.DataFrame(results["ood_check"])

        _save_results_to_csv(noisy_check_df, "sanity_support_set_noisy", log_dir)
        _save_results_to_csv(hard_check_df, "sanity_support_set_hard", log_dir)
        _save_results_to_csv(ood_check_df, "sanity_support_set_ood", log_dir)

    else:
        raise NotImplementedError(f"Method {method} not implemented.")

    # Wait for any pending background plot-saving jobs (sanity_params) to
    # finish writing to disk before the process exits.
    shutdown_executors()

def _save_results_to_csv(res_df, name, log_dir, mean_df=None):
    res_df.to_csv(os.path.join(log_dir, f"{name}_results.csv"), index=False)
    (mean_df if mean_df is not None else res_df.mean()).to_csv(
        os.path.join(log_dir, f"{name}_results_mean.csv"), index=True
    )
