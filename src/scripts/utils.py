import csv
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
import scipy.stats as stats
import torchvision.transforms.functional as vF
from collections import Counter

from interpreters import FAMAExplainer

# Bounded, shared thread pools reused across explain.py / check_explain/*.py so
# concurrency stays capped regardless of how many call sites use it (avoids a
# "pool of pools" explosion). Sized for a single RTX 4080S (16GB VRAM, so GPU
# work is capped at a handful of concurrent interpret() calls) + 12 CPU cores
# (IO/CPU-bound work like SLIC segmentation and matplotlib rendering can use more).
_GPU_WORKERS = 4
_IO_WORKERS = 8
# matplotlib's pyplot keeps global figure-manager state (Gcf) that is not
# thread-safe across concurrent calls, and GUI backends additionally require
# the main thread -- so plot-saving gets its own single dedicated worker
# instead of sharing the general IO pool. This still keeps the main loop
# (GPU inference for the next task) from blocking on rendering/disk I/O.
_PLOT_WORKERS = 1

_gpu_executor = None
_io_executor = None
_plot_executor = None

def _write_csv(filename, header, rows, log_dir="logs"):
    """Helper function to make writing CSV files easier."""
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, filename)
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return csv_path

_csv_handles = {}

def log_to_csv(csv_path, log, header=None):
    # Training calls this once per iteration (e.g. 60000x for the meta-loss
    # log) -- reopening + os.path.isfile()-checking the file every single
    # call adds real, easily-avoidable syscall overhead over a long run.
    # Keep one open (writer, file) pair per path instead, flushing after each
    # write so a crash still only loses at most the in-flight row (same
    # durability as before, just without the repeated open/close).
    cached = _csv_handles.get(csv_path)
    if cached is None:
        file_exists = os.path.isfile(csv_path)
        f = open(csv_path, mode='a', newline='')
        writer = csv.writer(f)
        if not file_exists and header is not None:
            writer.writerow(header)
        _csv_handles[csv_path] = (f, writer)
    else:
        f, writer = cached

    writer.writerow(log)
    f.flush()

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

def get_gpu_executor():
    """Shared thread pool for GPU-bound work (explainer.interpret calls)."""
    global _gpu_executor
    if _gpu_executor is None:
        _gpu_executor = ThreadPoolExecutor(max_workers=_GPU_WORKERS)
    return _gpu_executor

def get_io_executor():
    """Shared thread pool for CPU/IO-bound work (SLIC segmentation, plot saving)."""
    global _io_executor
    if _io_executor is None:
        _io_executor = ThreadPoolExecutor(max_workers=_IO_WORKERS)
    return _io_executor

def get_plot_executor():
    """Single dedicated background thread for matplotlib plot-saving jobs."""
    global _plot_executor
    if _plot_executor is None:
        _plot_executor = ThreadPoolExecutor(max_workers=_PLOT_WORKERS)
    return _plot_executor

def shutdown_executors(wait=True):
    """Release the shared thread pools. Call once a top-level mode (explain /
    check_explain) has finished all its work."""
    global _gpu_executor, _io_executor, _plot_executor, _cuda_stream_pool
    if _gpu_executor is not None:
        _gpu_executor.shutdown(wait=wait)
        _gpu_executor = None
    if _io_executor is not None:
        _io_executor.shutdown(wait=wait)
        _io_executor = None
    if _plot_executor is not None:
        _plot_executor.shutdown(wait=wait)
        _plot_executor = None
    _cuda_stream_pool = None

_cuda_stream_pool = None

def _get_cuda_stream_pool(n):
    """Lazily create a small, bounded, REUSED pool of CUDA streams (at most
    _GPU_WORKERS of them, ever). Streams are cheap to reuse but each fresh
    torch.cuda.Stream() carries its own allocator bookkeeping -- creating one
    per task in a long-running loop (e.g. metatest_batch_size=1 -> one brand
    new stream per metabatch, hundreds/thousands over a run) leaks/fragments
    VRAM slowly instead of releasing it, since the caching allocator doesn't
    always eagerly reclaim blocks cached against an abandoned stream."""
    global _cuda_stream_pool
    if _cuda_stream_pool is None:
        _cuda_stream_pool = [torch.cuda.Stream() for _ in range(min(n, _GPU_WORKERS))]
    return _cuda_stream_pool

def _is_cuda_device(device):
    return (isinstance(device, torch.device) and device.type == "cuda") or (
        isinstance(device, str) and device.startswith("cuda")
    )

def _run_with_stream(fn, device, stream=None):
    """Run fn(), optionally on a given (reused) CUDA stream so independent
    calls can overlap on the GPU instead of serializing; no-op passthrough on
    CPU or when no stream is given (nothing to overlap with)."""
    if stream is None or not _is_cuda_device(device):
        return fn()
    with torch.cuda.stream(stream):
        result = fn()
    stream.synchronize()
    return result

def parallel_map(fns, device):
    """Run a list of zero-arg callables concurrently on the shared GPU executor,
    each on a reused CUDA stream from a small bounded pool, and return their
    results in submission order."""
    if len(fns) == 1:
        # Nothing to overlap with a single task -- run inline, skip the
        # executor/stream machinery entirely (also avoids the per-call stream
        # churn described above for configs with only 1 task per metabatch).
        return [_run_with_stream(fns[0], device, stream=None)]

    executor = get_gpu_executor()
    streams = _get_cuda_stream_pool(len(fns)) if _is_cuda_device(device) else None
    futures = [
        executor.submit(_run_with_stream, fn, device, streams[i % len(streams)] if streams else None)
        for i, fn in enumerate(fns)
    ]
    return [f.result() for f in futures]

def submit_io_task(fn, *args, **kwargs):
    """Fire off a CPU/IO-bound job (e.g. SLIC segmentation) on the shared IO
    thread pool without blocking the caller; returns a Future the caller can
    wait on. Do NOT use this for matplotlib work -- see submit_plot_task."""
    return get_io_executor().submit(fn, *args, **kwargs)

def submit_plot_task(fn, *args, **kwargs):
    """Fire off a matplotlib plot-saving job on the single dedicated plot
    thread, without blocking the caller."""
    return get_plot_executor().submit(fn, *args, **kwargs)

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
