import copy
import functools
import math
import os

import matplotlib
matplotlib.use("Agg")  # plot saving runs on a background thread; GUI backends need the main thread
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from ..utils import correlation_sample_wise, parallel_map, submit_plot_task
from models.utils import get_layer_parameters_map


def randomize_layer(weight):
    # apply Kaiming Uniform for weight.dim >= 2
    if weight is None:
        return
    if weight.dim() >= 2:
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    # if weight is the bias layer or weight.dim == 1
    else:
        nn.init.uniform_(weight, -0.1, 0.1)

def save_full_nxm_grid(images_tensor, orig_saliencies, corrupted_data, save_path, alpha=0.5):
    """
    Draw a single figure containing an N (images) x M (corruption states) grid.
    """
    num_samples = images_tensor.shape[0]        # number of rows (N)
    num_cols = 1 + len(corrupted_data)          # number of columns (M = original + L corrupted layers)

    # 1. Create a large figure. Adjust figsize based on N and number of layers.
    # e.g. each cell is 3x3 inches.
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(num_cols * 3, num_samples * 3))

    # 2. Handle the case of a single sample or a single column so axes is always a 2D array (num_samples, num_cols)
    if num_samples == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_samples == 1:
        axes = np.array([axes])
    elif num_cols == 1:
        axes = np.array([[ax] for ax in axes])

    # 3. Loop over each cell in the grid
    for i in range(num_samples):
        # A. Preprocess the i-th original image for display (H, W, C)
        img = images_tensor[i].squeeze().cpu().detach().numpy()
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))

        img_min, img_max = img.min(), img.max()
        if img_max - img_min > 0:
            img = (img - img_min) / (img_max - img_min)

        cmap_img = 'gray' if img.ndim == 2 or img.shape[-1] == 1 else None

        # --- Column 0: original saliency of image i ---
        ax_orig = axes[i, 0]
        sal_orig = orig_saliencies[i].squeeze().cpu().detach().numpy()

        ax_orig.imshow(img, cmap=cmap_img)
        ax_orig.imshow(sal_orig, cmap='jet', alpha=alpha)

        # Only show the title on the first row
        if i == 0:
            ax_orig.set_title("Original", fontsize=10, fontweight='bold')

        # Show the image index on the leftmost column
        if num_cols > 0:
            ax_orig.set_ylabel(f"Img {i}", fontsize=10, fontweight='bold')
            ax_orig.set_yticks([])  # hide Y ticks but keep the Y label

        ax_orig.set_xticks([])  # hide X ticks
        # ax_orig.axis('off')  # using .axis('off') would also hide set_ylabel, so hide manually instead

        # --- Following columns: corrupted layers ---
        for j, (layer_idx, all_new_saliencies) in enumerate(corrupted_data):
            ax_corr = axes[i, j + 1]
            sal_corr = all_new_saliencies[i].squeeze().cpu().detach().numpy()

            ax_corr.imshow(img, cmap=cmap_img)
            ax_corr.imshow(sal_corr, cmap='jet', alpha=alpha)

            # Only show the layer title on the first row
            if i == 0:
                ax_corr.set_title(f"Corr Layer {layer_idx}", fontsize=10)

            ax_corr.axis('off')  # fully hide axes on inner cells

    # 4. Final touches and save
    fig.subplots_adjust(wspace=0.05, hspace=0.05)  # or adjust manually for tighter spacing
    plt.savefig(save_path, bbox_inches='tight', dpi=150)  # higher dpi for a sharper image
    plt.close(fig)

def check_on_task(explainer, theta_0, net_layers, sup_x, sup_y, que_x, que_y, T, log_dir, metabatch_id, task_id):
    # Each concurrent task gets its own shallow copy of the explainer so its
    # theta_0 override doesn't clash with other tasks running at the same time
    # (interpret() reads self.theta_0, and this check works by mutating it).
    local_explainer = copy.copy(explainer)
    local_explainer.theta_0 = [p.clone().detach() for p in theta_0]

    task_pearson = []
    task_spearman = []

    _, orig_saliency_map = local_explainer.interpret(sup_x, sup_y, que_x, que_y, T)

    corrupted_saliencies = []
    corrupted_theta_grouped = copy.deepcopy(net_layers)
    # Cascading randomization: each step corrupts one more layer on top of the
    # previous steps' corruption, so this inner loop must stay sequential.
    for layer_idx in range(len(corrupted_theta_grouped) - 1, -1, -1):
        layer = corrupted_theta_grouped[layer_idx]
        # destroy layer
        for param_tensor in layer["params"]:
            randomize_layer(param_tensor)
        corrupted_theta = []
        for l in corrupted_theta_grouped:
            corrupted_theta.extend(l["params"])
        local_explainer.theta_0 = [p.clone().detach() for p in corrupted_theta]
        _, new_saliency_map = local_explainer.interpret(sup_x, sup_y, que_x, que_y, T)

        scores = correlation_sample_wise(orig_saliency_map, new_saliency_map)
        task_pearson.append(scores["pearson"])
        task_spearman.append(scores["spearman"])

        corrupted_saliencies.append((layer_idx, new_saliency_map))

    save_path = os.path.join(log_dir, "plots", f"sanity_params_task{metabatch_id}-{task_id}_grid.png")
    # Detach + move to CPU before queuing -- otherwise every backlogged plot
    # job (single-worker executor, much slower than GPU inference) keeps its
    # tensors resident in VRAM until it's actually drawn, growing VRAM usage
    # monotonically over a long run instead of releasing it per-task.
    submit_plot_task(
        save_full_nxm_grid,
        sup_x.detach().cpu(),                # (N, C, H, W)
        orig_saliency_map.detach().cpu(),     # (N, H, W)
        [(layer_idx, sal.detach().cpu()) for layer_idx, sal in corrupted_saliencies],  # List of (layer_idx, (N, H, W))
        save_path,
        0.5,
    )
    return task_pearson, task_spearman

def sanity_check_params(explainer, test_loader, T, log_dir="logs"):
    test_loader_pbar = tqdm(
        test_loader, desc="Sanity Check", position=0, leave=True, unit="boT"
    )
    theta_0 = [p.clone().detach() for p in explainer.algo_mgr.theta_0]
    net_layers = get_layer_parameters_map(explainer.algo_mgr.baselearner, theta_0)
    results = {
        "pearson": [],
        "spearman": []
    }
    for metabatch_id, boT in enumerate(test_loader_pbar):
        # Tasks within a metabatch are independent of each other -> run them
        # concurrently instead of one at a time (only the per-task cascading
        # corruption loop above has to stay sequential).
        fns = [
            functools.partial(
                check_on_task, explainer, theta_0, net_layers,
                support[0], support[1], query[0], query[1], T,
                log_dir, metabatch_id, task_id,
            )
            for task_id, (support, query) in enumerate(boT)
        ]
        task_results = parallel_map(fns, device=explainer.device)

        for task_pearson, task_spearman in task_results:
            results["pearson"].append(task_pearson)
            results["spearman"].append(task_spearman)

    return results
