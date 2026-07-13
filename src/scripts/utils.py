import csv
import os
import shutil

import numpy as np
import scipy.stats as stats

from interpreters import FAMAExplainer

def _write_csv(filename, header, rows, log_dir="logs"):
    """Helper function to make writing CSV files easier."""
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, filename)
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return csv_path

def log_to_csv(csv_path, log, header=None):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)

        # write header if file does not exist and header is provided
        if not file_exists and header is not None:
            writer.writerow(header)

        # write log values
        writer.writerow(log)

def compute_stats(data):
    """Compute mean, std, ci95 for a list of numbers."""
    arr = np.array(data)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)  # sample standard deviation
    n = len(data)
    se = std / np.sqrt(n) # standard error
    ci95 = stats.t.ppf(0.975, n-1) * se # ci95 using t-distribution
    return mean, std, ci95

def correlation_sample_wise(A, A_prime):
    """Compute the mean Pearson and Spearman correlation between two batches of
    tensors, computed independently for each sample in the batch."""
    N = A.shape[0]
    a_np = A.detach().cpu().numpy().reshape(N, -1)
    a_prime_np = A_prime.detach().cpu().numpy().reshape(N, -1)
    pearson_list = []
    spearman_list = []
 
    for i in range(N):
        img = a_np[i]
        img_prime = a_prime_np[i]
 
        p_corr, _ = stats.pearsonr(img, img_prime)
        p_corr = 0.0 if np.isnan(p_corr) else p_corr
        pearson_list.append(p_corr)
 
        s_corr, _ = stats.spearmanr(img, img_prime)
        s_corr = 0.0 if np.isnan(s_corr) else s_corr
        spearman_list.append(s_corr)
 
    return {
        "pearson": np.mean(pearson_list).item(),
        "spearman": np.mean(spearman_list).item()
    }
 
def resolve_checkpoint_path(checkpoint_dir, use_best=False, use_last=True):
    """Resolve the checkpoint file path based on the use_best/use_last flags."""
    if use_best:
        return os.path.join(checkpoint_dir, "best_checkpoint.pt")
    if use_last:
        return os.path.join(checkpoint_dir, "last_checkpoint.pt")
    raise ValueError("Please specify a checkpoint to load (--use_best or --use_last)")
 
def load_checkpoint(algo_mgr, checkpoint_dir, use_best=False, use_last=True):
    """Load weights into an existing algo manager from the selected checkpoint."""
    checkpoint_path = resolve_checkpoint_path(checkpoint_dir, use_best, use_last)
 
    print("Loading checkpoint from", checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please run training first.")
    algo_mgr.read_file(checkpoint_path)
 
    return algo_mgr
 
def load_trained_algo(algo_class, algo_conf, checkpoint_dir, use_best=False, use_last=True):
    """Instantiate the algo manager and load its weights from the selected checkpoint."""
    algo_mgr = algo_class(**algo_conf)
    return load_checkpoint(algo_mgr, checkpoint_dir, use_best, use_last)
 
def build_explainer(algo, algo_class, algo_mgr, device):
    """Create the interpreter/explainer for a given algorithm."""
    if algo_class.__name__ == "MAML":
        return FAMAExplainer(algo_mgr, device=device)
    raise NotImplementedError(f"Algorithm {algo} can not be explained.")
 
def prepare_plots_dir(log_dir):
    """(Re)create a clean directory to store generated plots."""
    plots_dir = os.path.join(log_dir, "plots")
    if os.path.exists(plots_dir):
        shutil.rmtree(plots_dir)
    os.makedirs(plots_dir)
    return plots_dir
