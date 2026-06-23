import os
import shutil
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
    for metabatch_id, boT in enumerate(test_loader):
        for task_id, (support, query) in enumerate(boT):
            sup_x, sup_y = support
            que_x, que_y = query

            saliency_maps = explainer.saliency_x(
                sup_x, sup_y, que_x, que_y, T=T
            )

            shot_idx = 0
            original_img_tensor = sup_x[shot_idx].cpu().detach()
            saliency_tensor = saliency_maps[shot_idx].cpu().detach()

            img_min, img_max = original_img_tensor.min(), original_img_tensor.max()
            original_img_np = (original_img_tensor - img_min) / (img_max - img_min + 1e-8)
            original_img_np = original_img_np.permute(1, 2, 0).numpy()

            heatmap = torch.abs(saliency_tensor).sum(dim=0).numpy()
            hm_min, hm_max = heatmap.min(), heatmap.max()
            heatmap_normalized = (heatmap - hm_min) / (hm_max - hm_min + 1e-8)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            if original_img_np.shape[-1] == 1:
                axes[0].imshow(original_img_np[:, :, 0], cmap='gray')
            else:
                axes[0].imshow(original_img_np)
            axes[0].set_title(f"Support Input (Task {metabatch_id} - {task_id})")
            axes[0].axis('off')

            im = axes[1].imshow(heatmap_normalized, cmap='jet')
            axes[1].set_title(f"Feature Saliency Map (T={T})")
            axes[1].axis('off')

            fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

            save_path = os.path.join(log_dir, "plots", f"{algo}_task{metabatch_id}-{task_id}_saliency.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
