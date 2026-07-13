import copy
import math

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from ..utils import correlation_sample_wise


def randomize_layer(weight):
    # apply Kaiming Uniform for weight.dim >= 2
    if weight is not None and weight.dim() >= 2:
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    # if weight is the bias layer or weight.dim == 1
    elif weight is not None:
        nn.init.uniform_(weight, -0.1, 0.1)

    return weight

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
    # plt.tight_layout(pad=0.5)  # reduce spacing between cells
    fig.subplots_adjust(wspace=0.05, hspace=0.05)  # or adjust manually for tighter spacing
    plt.savefig(save_path, bbox_inches='tight', dpi=150)  # higher dpi for a sharper image
    plt.close()

def get_layer_parameters_map(baselearner, theta_0):
    """
    Map the flat theta_0 parameter list back onto the actual layers of the baselearner.
    """
    param_iterator = iter(theta_0)
    net_info_with_params = []

    # Iterate over sub-modules holding parameters, in PyTorch's own order
    for name, module in baselearner.named_modules():
        # Only consider modules with local parameters (excludes wrapping parent modules)
        local_params = list(module.parameters(recurse=False))
        if len(local_params) > 0:
            layer_dict = {
                "name": name,
                "type": module.__class__.__name__,
                "params": []
            }
            # Pull exactly as many tensors from theta_0 as this layer has parameters
            for _ in range(len(local_params)):
                layer_dict["params"].append(next(param_iterator))

            net_info_with_params.append(layer_dict)

    return net_info_with_params

def check_on_task(explainer, theta_0, net_layers, sup_x, sup_y, que_x, que_y, T):
    task_pearson = []
    task_spearman = []

    explainer.theta_0 = [p.clone().detach() for p in theta_0]
    _, orig_saliency_map = explainer.interpret(sup_x, sup_y, que_x, que_y, T)

    corrupted_saliencies = []
    corrupted_theta_grouped = copy.deepcopy(net_layers)
    for layer_idx in range(len(corrupted_theta_grouped) - 1, -1, -1):
        layer = corrupted_theta_grouped[layer_idx]
        # destroy layer
        for param_tensor in layer["params"]:
            param_tensor = randomize_layer(param_tensor)
        corrupted_theta = []
        for l in corrupted_theta_grouped:
            corrupted_theta.extend(l["params"])
        explainer.theta_0 = [p.clone().detach() for p in corrupted_theta]
        _, new_saliency_map = explainer.interpret(sup_x, sup_y, que_x, que_y, T)

        scores = correlation_sample_wise(orig_saliency_map, new_saliency_map)
        task_pearson.append(scores["pearson"])
        task_spearman.append(scores["spearman"])

        corrupted_saliencies.append((layer_idx, new_saliency_map))
        
    save_full_nxm_grid(
        images_tensor=sup_x,              # (N, C, H, W)
        orig_saliencies=orig_saliency_map, # (N, H, W)
        corrupted_data=corrupted_saliencies,     # List of (layer_idx, (N, H, W))
        save_path=f"task_{sup_x.shape[0]}_saliency_grid.png", 
        alpha=0.5
    )
    return task_pearson, task_spearman

def sanity_check_params(explainer, test_loader, T):
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
        boT_pbar = tqdm(
            boT, desc=f"Batch {metabatch_id}", position=1, leave=False, unit="task"
        )
        for task_id, (support, query) in enumerate(boT_pbar):
            sup_x, sup_y, _ = support
            que_x, que_y, _ = query

            task_pearson, task_spearman = check_on_task(explainer, theta_0, net_layers, sup_x, sup_y, que_x, que_y, T)
            results["pearson"].append(task_pearson)
            results["spearman"].append(task_spearman)

    return results
