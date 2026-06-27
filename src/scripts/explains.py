import os
import shutil
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from interpreters import MAMLPostHocExplainer


def explain(algo, algo_class, test_loader, algo_conf, use_best=False, use_last=True,
            checkpoint_dir="checkpoints", log_dir="logs"):
    
    algo_mgr = algo_class(**algo_conf)
    T = algo_mgr.T_test
    device = algo_conf.get("device", "cpu")

    # Xóa và tạo thư mục plots
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
        explainer = MAMLPostHocExplainer(algo_mgr, device=device)
    else:
        raise NotImplementedError(f"Algorithm {algo} can not be explained.")

    # Explain each task
    test_loader_pbar = tqdm(test_loader, desc="Explaining", position=0, leave=True, unit="boT")
    for metabatch_id, boT in enumerate(test_loader_pbar):
        boT_pbar = tqdm(boT, desc=f"Batch {metabatch_id}", position=1, leave=False, unit="task")
        for task_id, (support, query) in enumerate(boT_pbar):
            sup_x, sup_y = support
            que_x, que_y = query

            adaptation_gain, saliency_map = explainer.interpret(
                sup_x, sup_y, que_x, que_y, T=T,
            )

            show_explaination(
                sup_x, saliency_map, adaptation_gain,
                algo, log_dir, metabatch_id, task_id, T
            )


def show_explaination(sup_x, saliency_map, adaptation_gain,
                      algo, log_dir, metabatch_id, task_id, t):
    cols = 2
    rows = sup_x.shape[0]
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4.2 * rows))
    
    if rows == 1:
        axes = axes[np.newaxis, :]

    for shot_idx in range(num_shot_to_plot):
        # original image 
        original_img_tensor = sup_x[shot_idx].cpu().detach()
        img_min, img_max = original_img_tensor.min(), original_img_tensor.max()
        original_img_np = (original_img_tensor - img_min) / (img_max - img_min + 1e-8)
        original_img_np = original_img_np.permute(1, 2, 0).numpy()
        
        is_gray = original_img_np.shape[-1] == 1
        cmap_img = 'gray' if is_gray else None
        img_to_show = original_img_np[:, :, 0] if is_gray else original_img_np

        # saliency map
        saliency_tensor = saliency_map[shot_idx].cpu().detach()
        heatmap = saliency_tensor.sum(dim=0).numpy()
        max_abs = np.max(np.abs(heatmap)) + 1e-8 # normalization for colorbar

        # overlay pic (first col)
        ax_overlay = axes[shot_idx, 0]
        ax_overlay.imshow(img_to_show, cmap=cmap_img)
        # red = positive gain (tốt), blue = negative gain (nhiễu)
        img_overlay = ax_overlay.imshow(heatmap, cmap='rdbu_r', alpha=0.75, vmin=-max_abs, vmax=max_abs)
        ax_overlay.set_title(f"shot {shot_idx+1} saliency map", fontsize=8)
        ax_overlay.axis('off')

        # original pic (second col)
        ax_orig = axes[shot_idx, 1]
        ax_orig.imshow(img_to_show, cmap=cmap_img)
        ax_orig.set_title(f"original shot {shot_idx+1}", fontsize=8)
        ax_orig.axis('off')

    # add colorbar to show color meaning
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(img_overlay, cax=cbar_ax, label="feature contribution\n(red: helpful, blue: harmful)")

    # title
    gain_text = f"adaptation gain: {adaptation_gain:.2f}%" if adaptation_gain is not None else ""
    plt.suptitle(
        f"explaination - task {metabatch_id}-{task_id}\n{gain_text}",
        fontsize=18, fontweight='bold', y=0.98
    )
    plt.subplots_adjust(right=0.9, top=0.92, wspace=0.1, hspace=0.3)

    save_path = os.path.join(log_dir, "plots", 
                            f"{algo}_task{metabatch_id}-{task_id}_fama_trajectory.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
