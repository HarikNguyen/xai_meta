import os
import shutil
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from interpreters import MAMLPostHocExplainer

def explain(algo, algo_class, test_loader, algo_conf, use_best=False, use_last=True, checkpoint_dir="checkpoints", log_dir="logs"):
    
    # define algo_obj for manage training and validating strategies
    algo_mgr = algo_class(**algo_conf)
    T = algo_mgr.T_test
    device = algo_conf.get("device", "cpu")
    if os.path.exists(os.path.join(log_dir, "plots")):
        shutil.rmtree(os.path.join(log_dir, "plots"))
    os.makedirs(os.path.join(log_dir, "plots"))

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

    # define explainer
    if algo_class.__name__ == "MAML":
        explainer = MAMLPostHocExplainer(algo_mgr, device=device)
    else:
        raise NotImplementedError(f"Algorithm {algo} can not be explained.")

    # explain each task
    test_loader_pbar = tqdm(test_loader, desc="Training", position=0, leave=True, unit="boT")
    for metabatch_id, boT in enumerate(test_loader_pbar):
        boT_pbar = tqdm(boT, desc=f"Batch {metabatch_id}", position=1, leave=False, unit="task")
        for task_id, (support, query) in enumerate(boT_pbar):
            sup_x, sup_y = support
            que_x, que_y = query

            saliency_maps = explainer.interpret(
                sup_x, sup_y, que_x, que_y, T=T
            )

            show_explaination(sup_x, saliency_maps, algo, log_dir, metabatch_id, task_id, T)

def show_explaination(sup_x, saliency_maps, algo, log_dir, metabatch_id, task_id, T):
    # inits
    num_shot = sup_x.shape[0]
    cols = min(num_shot, 5)
    # Tăng gấp đôi số hàng: một hàng cho overlay, một hàng cho ảnh gốc
    rows = math.ceil(num_shot / cols) * 2 
    
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    
    # Đảm bảo axes luôn là một mảng 2D (rows, cols)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for shot_idx in range(num_shot):
        # Tính toán vị trí hàng/cột
        row_idx = (shot_idx // cols) * 2
        col_idx = shot_idx % cols
        
        # --- Xử lý dữ liệu ảnh ---
        original_img_tensor = sup_x[shot_idx].cpu().detach()
        img_min, img_max = original_img_tensor.min(), original_img_tensor.max()
        original_img_np = (original_img_tensor - img_min) / (img_max - img_min + 1e-8)
        original_img_np = original_img_np.permute(1, 2, 0).numpy()

        is_gray = original_img_np.shape[-1] == 1
        cmap_img = 'gray' if is_gray else None
        img_to_show = original_img_np[:, :, 0] if is_gray else original_img_np

        # 1. Vẽ ảnh Overlay (Hàng trên)
        ax_overlay = axes[row_idx, col_idx]
        saliency_tensor = saliency_maps[shot_idx].cpu().detach()
        heatmap = torch.abs(saliency_tensor).sum(dim=0).numpy()
        heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        ax_overlay.imshow(img_to_show, cmap=cmap_img)
        ax_overlay.imshow(heatmap_normalized, cmap='jet', alpha=0.5)
        ax_overlay.set_title(f"Overlay {shot_idx + 1}")
        ax_overlay.axis('off')

        # 2. Vẽ ảnh Gốc (Hàng dưới)
        ax_orig = axes[row_idx + 1, col_idx]
        ax_orig.imshow(img_to_show, cmap=cmap_img)
        ax_orig.set_title(f"Original {shot_idx + 1}")
        ax_orig.axis('off')

    # Tắt các ô trống nếu có
    for i in range(rows):
        for j in range(cols):
            if (i // 2) * cols + j >= num_shot:
                axes[i, j].axis('off')

    plt.suptitle(f"Saliency Overlay & Original - Task {metabatch_id}-{task_id} (T={T})", fontsize=16, y=1.02)
    plt.tight_layout()

    plot_dir = os.path.join(log_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True) # Đảm bảo thư mục tồn tại
    save_path = os.path.join(plot_dir, f"{algo}_task{metabatch_id}-{task_id}_saliency_overlay.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
