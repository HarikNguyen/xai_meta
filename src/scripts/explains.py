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
                      algo, log_dir, metabatch_id, task_id, T):
    """
    Vẽ ảnh Saliency Map Bidirectional theo từng bước adaptation.
    Red: Positive contribution (Giảm loss), Blue: Negative (Tăng loss).
    """
    # saliency_map shape expected: [T, num_shots, C, H, W] or similar
    if len(saliency_map.shape) == 4:  # [T, shots, H, W]
        num_steps = saliency_map.shape[0]
    else:
        num_steps = 1
        saliency_map = saliency_map.unsqueeze(0)

    num_shots_to_plot = min(sup_x.shape[0], 5)
    
    # Create figure: 2 rows per step (Overlay + Original)
    rows = num_steps * 2
    cols = num_shots_to_plot
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.8 * rows))
    
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for step_idx in range(num_steps):
        for shot_idx in range(num_shots_to_plot):
            row_base = step_idx * 2
            col_idx = shot_idx

            # --- Original Image ---
            original_img_tensor = sup_x[shot_idx].cpu().detach()
            img_min, img_max = original_img_tensor.min(), original_img_tensor.max()
            original_img_np = (original_img_tensor - img_min) / (img_max - img_min + 1e-8)
            original_img_np = original_img_np.permute(1, 2, 0).numpy()

            is_gray = original_img_np.shape[-1] == 1
            cmap_img = 'gray' if is_gray else None
            img_to_show = original_img_np[:, :, 0] if is_gray else original_img_np

            # --- Saliency for this step ---
            saliency_tensor = saliency_map[step_idx, shot_idx].cpu().detach()
            heatmap = saliency_tensor.sum(dim=0).numpy()  # sum over channels

            # Symmetric normalization for bidirectional meaning
            max_abs = np.max(np.abs(heatmap)) + 1e-8

            # Overlay (Top row)
            ax_overlay = axes[row_base, col_idx]
            ax_overlay.imshow(img_to_show, cmap=cmap_img)
            img_overlay = ax_overlay.imshow(
                heatmap, 
                cmap='RdBu_r', 
                alpha=0.6, 
                vmin=-max_abs, 
                vmax=max_abs
            )
            ax_overlay.set_title(f"Step {step_idx+1} | Shot {shot_idx+1}")
            ax_overlay.axis('off')

            # Original (Bottom row)
            ax_orig = axes[row_base + 1, col_idx]
            ax_orig.imshow(img_to_show, cmap=cmap_img)
            ax_orig.set_title(f"Original Shot {shot_idx+1}")
            ax_orig.axis('off')

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(img_overlay, cax=cbar_ax, label="Feature Contribution\n(Red: Helpful, Blue: Harmful)")

    gain_text = f"Adaptation Gain: {adaptation_gain:.2f}%" if adaptation_gain is not None else ""
    plt.suptitle(
        f"Bidirectional FAMA Trajectory - Task {metabatch_id}-{task_id}\n{gain_text}",
        fontsize=18, fontweight='bold', y=0.98
    )
    
    plt.subplots_adjust(right=0.9, top=0.92, wspace=0.15, hspace=0.35)
    
    save_path = os.path.join(log_dir, "plots",
                            f"{algo}_task{metabatch_id}-{task_id}_fama_trajectory.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
