import csv
import os
import shutil

import torch
import numpy as np
import scipy.stats as stats
import torchvision.transforms.functional as vF
from collections import Counter

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

def blur_sup(sup_x, kernel_size=7, sigma=3.0):
    """Apply a Gaussian blur to the support set."""
    sup_x_blurred = sup_x.clone()
    sup_x_blurred = vF.gaussian_blur(
        sup_x_blurred, 
        kernel_size=[kernel_size, kernel_size], 
        sigma=[sigma, sigma]
    )
    return sup_x_blurred

def permute_label(sup_y, flip_ratio=0.6):
    """Randomly permute (flip) the labels in the support set."""
    N, C = sup_y.shape
    device = sup_y.device
    sup_y_np = sup_y.clone().detach().cpu().numpy()

    # 1. Randomly pick the indices whose labels will be shuffled
    num_flip = int(N * flip_ratio)
    if num_flip <= 1:
        # Not enough elements to permute
        return torch.from_numpy(sup_y_np).to(device)

    flip_indices = np.random.choice(N, num_flip, replace=False)

    # 2. Get the labels at the selected positions
    # For the optimal-shift algorithm to work, convert labels to integer class ids (0, 1, 2... C-1)
    # If sup_y_np already holds integer labels (N, 1), skip argmax. Here we assume one-hot format (N, C)
    labels = np.argmax(sup_y_np[flip_indices], axis=1)

    # 3. Apply the Sort & Shift algorithm
    # Keep the original index within the flip group so we can map values back
    indexed_labels = sorted(enumerate(labels), key=lambda x: x[1])

    # Count occurrences of the most frequent label in this group
    counts = Counter(labels)
    max_freq = max(counts.values())

    # Circularly shift the sorted array by max_freq positions
    # This shift pushes identical labels as far apart from each other as possible
    shifted_indexed = indexed_labels[-max_freq:] + indexed_labels[:-max_freq]

    # 4. Write the optimally permuted labels back into sup_y_np
    # Keep a temporary copy of the original label vectors before they get overwritten
    temp_targets = sup_y_np[flip_indices].copy()

    for i in range(num_flip):
        original_pos_in_flip = indexed_labels[i][0]
        # Actual position in the sup_y_np matrix
        actual_global_idx = flip_indices[original_pos_in_flip]

        # Get the label vector from the shifted-to position
        from_pos_in_flip = shifted_indexed[i][0]

        # Overwrite the label vector (one-hot or probability distribution)
        sup_y_np[actual_global_idx] = temp_targets[from_pos_in_flip]

    # 5. Convert back to a tensor on the original device
    sup_y_np = torch.from_numpy(sup_y_np).to(device)
    return sup_y_np
