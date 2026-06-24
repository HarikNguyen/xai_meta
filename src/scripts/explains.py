import os
import shutil
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
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
    total_tasks = len(test_loader) * len(test_loader.dataset)
    with tqdm(total=len(test_loader) * len(test_loader.dataset), desc="Processing Tasks") as pbar:
        for metabatch_id, boT in enumerate(test_loader):
            for task_id, (support, query) in enumerate(boT):
                sup_x, sup_y = support
                que_x, que_y = query

                saliency_maps = explainer.saliency_x(
                    sup_x, sup_y, que_x, que_y, T=T
                )

                show_explaination(sup_x, saliency_maps, algo, log_dir, metabatch_id, task_id, T)
                pbar.update(1)

def show_explaination(sup_x, saliency_maps, algo, log_dir, metabatch_id, task_id, T):
    # inits
    num_shot = sup_x.shape[0]
    cols = min(num_shot, 5)
    rows = math.ceil(num_shot/ cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if num_shot == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes).flatten()

    for shot_idx in range(num_shot):
        ax = axes[shot_idx]

        original_img_tensor = sup_x[shot_idx].cpu().detach()
        img_min, img_max = original_img_tensor.min(), original_img_tensor.max()
        original_img_np = (original_img_tensor - img_min) / (img_max - img_min + 1e-8)
        original_img_np = original_img_np.permute(1, 2, 0).numpy()

        is_gray = original_img_np.shape[-1] == 1
        if is_gray:
            img_to_show = original_img_np[:, :, 0]
            cmap_img = 'gray'
        else:
            img_to_show = original_img_np
            cmap_img = None

        saliency_tensor = saliency_maps[shot_idx].cpu().detach()
        heatmap = torch.abs(saliency_tensor).sum(dim=0).numpy()
        hm_min, hm_max = heatmap.min(), heatmap.max()
        heatmap_normalized = (heatmap - hm_min) / (hm_max - hm_min + 1e-8)

        ax.imshow(img_to_show, cmap=cmap_img)
        ax.imshow(heatmap_normalized, cmap='jet', alpha=0.5)

        ax.set_title(f"Shot {shot_idx + 1}")
        ax.axis('off')

    for idx in range(num_shot, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f"Saliency Overlay - Task {metabatch_id}-{task_id} (T={T})", fontsize=16, y=1.02)
    plt.tight_layout()

    plot_dir = os.path.join(log_dir, "plots")
    save_path = os.path.join(plot_dir, f"{algo}_task{metabatch_id}-{task_id}_saliency_overlay.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
