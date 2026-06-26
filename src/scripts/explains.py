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

            # Gọi interpret với cả gain và trajectory
            result = explainer.interpret(
                sup_x, sup_y, que_x, que_y, T=T,
                return_gain=True,
                # return_saliency=False,      # Không cần total nếu chỉ xem trajectory
                # return_trajectory=True
            )

            if isinstance(result, tuple):
                adaptation_gain, trajectory_saliencies = result
            else:
                adaptation_gain = None
                trajectory_saliencies = result

            show_explaination(
                sup_x, trajectory_saliencies, adaptation_gain,
                algo, log_dir, metabatch_id, task_id, T
            )


def show_explaination(sup_x, trajectory_saliencies, adaptation_gain,
                      algo, log_dir, metabatch_id, task_id, T):
    """
    Vẽ một ảnh lớn chứa saliency map theo từng bước adaptation (T steps).
    """
    num_shot = sup_x.shape[0]
    cols = min(num_shot, 5)
    rows = T * 2  # Mỗi step có 2 hàng: Overlay + Original

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    
    # Đảm bảo axes là mảng 2D
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for step_idx in range(T):
        saliency_step = trajectory_saliencies[step_idx]  # saliency tại bước này

        for shot_idx in range(num_shot):
            row_base = step_idx * 2
            col_idx = shot_idx % cols

            # --- Original Image ---
            original_img_tensor = sup_x[shot_idx].cpu().detach()
            img_min, img_max = original_img_tensor.min(), original_img_tensor.max()
            original_img_np = (original_img_tensor - img_min) / (img_max - img_min + 1e-8)
            original_img_np = original_img_np.permute(1, 2, 0).numpy()
            
            is_gray = original_img_np.shape[-1] == 1
            cmap_img = 'gray' if is_gray else None
            img_to_show = original_img_np[:, :, 0] if is_gray else original_img_np

            # --- Overlay (Hàng trên) ---
            ax_overlay = axes[row_base, col_idx]
            saliency_tensor = saliency_step[shot_idx].cpu().detach()
            heatmap = torch.abs(saliency_tensor).sum(dim=0).numpy()   # sum channels nếu multi-channel
            heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

            ax_overlay.imshow(img_to_show, cmap=cmap_img)
            ax_overlay.imshow(heatmap_norm, cmap='jet', alpha=0.6)
            ax_overlay.set_title(f"Step {step_idx+1} | Shot {shot_idx+1}")
            ax_overlay.axis('off')

            # --- Original Image (Hàng dưới) ---
            ax_orig = axes[row_base + 1, col_idx]
            ax_orig.imshow(img_to_show, cmap=cmap_img)
            ax_orig.set_title(f"Original Shot {shot_idx+1}")
            ax_orig.axis('off')

    # Tắt các ô trống (nếu num_shot < cols)
    for i in range(rows):
        for j in range(cols):
            if (i // 2) * cols + j >= num_shot:
                axes[i, j].axis('off')

    gain_text = f"Adaptation Gain: {adaptation_gain:.4f}" if adaptation_gain is not None else ""
    plt.suptitle(
        f"Saliency Trajectory - Task {metabatch_id}-{task_id} (T={T})\n{gain_text}",
        fontsize=16, y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = os.path.join(log_dir, "plots", 
                            f"{algo}_task{metabatch_id}-{task_id}_trajectory.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
