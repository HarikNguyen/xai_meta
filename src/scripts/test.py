import os
import csv
from collections import defaultdict
from tqdm import tqdm

from .utils import log_to_csv, compute_stats, load_trained_algo, _write_csv
from loaders.utils import boT_to_stack


def run_test(args, algo_class, test_loader, algo_conf, use_best=False, use_last=True, checkpoint_dir="checkpoints", log_dir="logs"):
    # define algo_obj for manage training and validating strategies
    algo_mgr = load_trained_algo(algo_class, algo_conf, checkpoint_dir, use_best, use_last)

    # test on whole test set
    all_sup_losses, all_que_losses, all_sup_accs, all_que_accs = test_on_wholeset(algo_mgr, test_loader)

    # compute stats
    num_steps = len(all_sup_losses)
    total_tasks = len(all_sup_losses[0])

    metrics_dict = {
        "SUPPORT LOSS": all_sup_losses,
        "QUERY LOSS": all_que_losses,
        "SUPPORT ACCURACY": all_sup_accs,
        "QUERY ACCURACY": all_que_accs
    }

    # print and save
    print_n_log_test(metrics_dict, num_steps, total_tasks, log_dir)
    # save details
    save_details(metrics_dict, num_steps, total_tasks, log_dir)

    # save image paths
    save_paths(all_sup_outpaths, "support", log_dir)
    save_paths(all_que_outpaths, "query", log_dir)

############################################################################################
### Helper Funcs
############################################################################################

def test_on_wholeset(algo_mgr, test_loader):
    all_sup_losses = defaultdict(list)
    all_que_losses = defaultdict(list)
    all_sup_accs = defaultdict(list)
    all_que_accs = defaultdict(list)
    all_sup_outpaths = []
    all_que_outpaths = []

    test_pbar = tqdm(test_loader, desc="Testing", leave=True)
    for boT in test_pbar:
        # fast-adaptation for each task in meta-batch

        sup_x, sup_y, que_x, que_y, sup_outpath, que_outpath = boT_to_stack(boT, has_outpath=True) # stack of meta_batch_size tasks
        sup_losses, que_losses, sup_accs, que_accs = algo_mgr.test(sup_x, sup_y, que_x, que_y)
        
        # store results
        for step in range(len(sup_losses)):
            all_sup_losses[step].extend(sup_losses[step].detach().cpu().numpy())
            all_que_losses[step].extend(que_losses[step].detach().cpu().numpy())
            all_sup_accs[step].extend(sup_accs[step].detach().cpu().numpy())
            all_que_accs[step].extend(que_accs[step].detach().cpu().numpy())

        all_sup_outpaths.append(sup_outpath)
        all_que_outpaths.append(que_outpath)

    return all_sup_losses, all_que_losses, all_sup_accs, all_que_accs

def print_n_log_test(metrics_dict, num_steps, total_tasks, log_dir="logs"):
    print(f"\n\n{'#'*70}\nMETA-TESTING RESULTS OVER {total_tasks} TASKS\n{'#'*70}")

    final_results = {}

    for metric_name, data_by_step in metrics_dict.items():
        print(f"\n\n{'='*60}\nTABLE: {metric_name}\n{'='*60}")
        print(f"{'Step':<15} | {'Mean':<10} | {'Std':<10} | {'CI 95%':<10}\n{'-'*60}")

        csv_filename = f"test_{metric_name.replace(' ', '_')}.csv"
        all_rows = []
        for step in range(num_steps):
            mean, std, ci95 = compute_stats(data_by_step[step])
            step_label = "Pre-update" if step == 0 else f"Update {step}"
            print(f"{step_label:<15} | {mean:<10.4f} | {std:<10.4f} | ± {ci95:<10.4f}")
            all_rows.append([step_label, mean, std, ci95])
        _write_csv(csv_filename, ["Step", "Mean", "Std", "CI_95"], all_rows, log_dir)

    print(f"\n{'='*60}\n")

def save_details(metrics_dict, num_steps, total_tasks, log_dir="logs"):
    print(f"\n{'='*70}\n SAVING DETAILED METRICS PER TASK\n{'='*70}")
    for metric_name, data_by_step in metrics_dict.items():
        header = ["task_id", "pre_update"] + [f"update_{step}" for step in range(1, num_steps)]
        csv_filename = f"detailed_test_{metric_name.replace(' ', '_').lower()}.csv"
        rows = [[t_id] + [f"{data_by_step[s][t_id]:.6f}" for s in range(num_steps)] for t_id in range(total_tasks)]
        
        path = _write_csv(csv_filename, header, rows, log_dir)
        print(f"[*] Saved detailed {metric_name} -> {path}")

def save_paths(all_outpaths, data_type, log_dir="logs"):
    """
    Save image paths to a csv file.

    Args:
        all_outpaths (list): A list of image paths.
        data_type (str): The type of data (support or query).
        log_dir (str): The directory to save the csv file.
    """
    print(f"\n{'='*70}\n SAVING IMAGE PATHS TO {data_type.upper()}\n{'='*70}")
    num_imgs = len(all_outpaths[0]) if all_outpaths else 0
    header = ["task_id"] + [f"img_{i}" for i in range(num_imgs)]
    rows = [[t_id] + list(paths) for t_id, paths in enumerate(all_outpaths)]

    path = _write_csv(f"{data_type}.csv", header, rows, log_dir)
    print(f"[*] Saved image paths -> {path}")
