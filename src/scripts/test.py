import os
import csv
import torch
from collections import defaultdict
from tqdm import tqdm

from .utils import log_to_csv, compute_stats
from loaders.utils import boT_to_stack


def run_test(args, algo_class, test_loader, algo_conf, use_best=False, use_last=True, checkpoint_dir="checkpoints", log_dir="logs"):
    # define algo_obj for manage training and validating strategies
    algo_mgr = algo_class(**algo_conf)

    # load checkpoint
    if use_best:
        checkpoint_path = os.path.join(checkpoint_dir, f"best_checkpoint.pt")
    elif use_last:
        checkpoint_path = os.path.join(checkpoint_dir, f"last_checkpoint.pt")
    else:
        raise ValueError("Please specify a checkpoint to load (--use_last or --use_best)")
    print("Loading checkpoint from", checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please run training first.")
    algo_mgr.read_file(checkpoint_path)

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

############################################################################################
### Helper Funcs
############################################################################################

def test_on_wholeset(algo_mgr, test_loader):
    all_sup_losses = defaultdict(list)
    all_que_losses = defaultdict(list)
    all_sup_accs = defaultdict(list)
    all_que_accs = defaultdict(list)

    test_pbar = tqdm(test_loader, desc="Testing", leave=True)
    for boT in test_pbar:
        # fast-adaptation for each task in meta-batch
        sup_x, sup_y, que_x, que_y = boT_to_stack(boT) # stack of meta_batch_size tasks
        sup_losses, que_losses, sup_accs, que_accs = algo_mgr.test(sup_x, sup_y, que_x, que_y)
        
        # store results
        for step in range(len(sup_losses)):
            all_sup_losses[step].extend(sup_losses[step].detach().cpu().numpy())
            all_que_losses[step].extend(que_losses[step].detach().cpu().numpy())
            all_sup_accs[step].extend(sup_accs[step].detach().cpu().numpy())
            all_que_accs[step].extend(que_accs[step].detach().cpu().numpy())

    return all_sup_losses, all_que_losses, all_sup_accs, all_que_accs

def print_n_log_test(metrics_dict, num_steps, total_tasks, log_dir="logs"):
    print("\n\n" + "#" * 70)
    print(f"META-TESTING RESULTS OVER {total_tasks} TASKS")
    print("#" * 70)

    final_results = {}

    for metric_name, data_by_step in metrics_dict.items():
        print(f"\n\n{'='*60}")
        print(f" TABLE: {metric_name} ")
        print(f"{'='*60}")
        print(f"{'Step':<15} | {'Mean':<10} | {'Std':<10} | {'CI 95%':<10}")
        print(f"{'-'*60}")
        
        csv_filename = f"test_{metric_name.replace(' ', '_')}.csv"
        csv_path = os.path.join(log_dir, csv_filename)
        if os.path.exists(csv_path):
            os.remove(csv_path)
        csv_header = ["Step", "Mean", "Std", "CI_95"]

        for step in range(num_steps):
            mean, std, ci95 = compute_stats(data_by_step[step])
            
            step_label = "Pre-update" if step == 0 else f"Update {step}"
            
            print(f"{step_label:<15} | {mean:<10.4f} | {std:<10.4f} | ± {ci95:<10.4f}")

            log_row = [step_label, mean, std, ci95]
            log_to_csv(csv_path, log_row, header=csv_header)
            
    print(f"\n{'='*60}\n")

def save_details(metrics_dict, num_steps, total_tasks, log_dir="logs"):
    print("\n" + "=" * 70)
    print(" SAVING DETAILED METRICS PER TASK")
    print("=" * 70)

    os.makedirs(log_dir, exist_ok=True)

    for metric_name, data_by_step in metrics_dict.items():
        csv_filename = f"detailed_test_{metric_name.replace(' ', '_').lower()}.csv"
        csv_path = os.path.join(log_dir, csv_filename)
        header = ["task_id", "pre_update"] + [f"update_{step}" for step in range(1, num_steps)]
        
        with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            for task_id in range(total_tasks):
                row = [task_id]
                for step in range(num_steps):
                    val = data_by_step[step][task_id]
                    row.append(f"{val:.6f}")
                
                writer.writerow(row)
                
        print(f"[*] Saved detailed {metric_name:<18} -> {csv_path}")
