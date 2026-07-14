import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .utils import load_checkpoint, prepare_plots_dir, build_explainer, _write_csv, permute_label


def explain(
    algo,
    algo_class,
    test_loader,
    algo_conf,
    use_best=False,
    use_last=True,
    checkpoint_dir="checkpoints",
    log_dir="logs",
    flip_ratio=None,
):
    # Inits
    algo_mgr = algo_class(**algo_conf)
    T = algo_mgr.T_test
    device = algo_conf.get("device", "cpu")

    # Create dir for plots saving
    prepare_plots_dir(log_dir)

    # Load checkpoint
    load_checkpoint(algo_mgr, checkpoint_dir, use_best, use_last)

    # Define explainer
    explainer = build_explainer(algo, algo_class, algo_mgr, device)

    # Explain each task
    test_loader_pbar = tqdm(
        test_loader, desc="Explaining", position=0, leave=True, unit="boT"
    )
    ad_gains = []
    sup_paths = []
    que_paths = []
    for metabatch_id, boT in enumerate(test_loader_pbar):
        boT_pbar = tqdm(
            boT, desc=f"Batch {metabatch_id}", position=1, leave=False, unit="task"
        )
        for task_id, (support, query) in enumerate(boT_pbar):
            sup_x, sup_y, sup_outpath = support
            que_x, que_y, que_outpath = query

            # Random flip label in the support set if flip_ratio is not None
            if flip_ratio is not None:
                sup_y = permute_label(sup_y, flip_ratio=flip_ratio)

            adaptation_gain, saliency_map = explainer.interpret(
                sup_x, sup_y, que_x, que_y, T=T
            )

            ad_gains.append((metabatch_id, task_id, adaptation_gain))
            sup_paths.append((metabatch_id, task_id, sup_outpath))
            que_paths.append((metabatch_id, task_id, que_outpath))

            show_explaination(
                sup_x,
                saliency_map,
                adaptation_gain,
                algo,
                log_dir,
                metabatch_id,
                task_id,
                T,
            )

    _write_csv("adaptation_gain.csv", ["metabatch_id", "task_id", "adaptation_gain"], ad_gains, log_dir)
    _write_csv("S_paths.csv", ["metabatch_id", "task_id", "support_path"], sup_paths, log_dir)
    _write_csv("Q_paths.csv", ["metabatch_id", "task_id", "query_path"], que_paths, log_dir)

def show_explaination(
    sup_x, saliency_map, adaptation_gain, algo, log_dir, metabatch_id, task_id, t
):
    cols = 2
    rows = sup_x.shape[0]
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4.2 * rows))

    if rows == 1:
        axes = axes[np.newaxis, :]

    for shot_idx in range(rows):
        # original image
        original_img_tensor = sup_x[shot_idx].cpu().detach()
        img_min, img_max = original_img_tensor.min(), original_img_tensor.max()
        original_img_np = (original_img_tensor - img_min) / (img_max - img_min + 1e-8)
        original_img_np = original_img_np.permute(1, 2, 0).numpy()

        is_gray = original_img_np.shape[-1] == 1
        cmap_img = "gray" if is_gray else None
        img_to_show = original_img_np[:, :, 0] if is_gray else original_img_np

        # saliency map
        saliency_tensor = saliency_map[shot_idx].cpu().detach()
        heatmap = saliency_tensor.sum(dim=0).numpy()
        max_abs = np.max(np.abs(heatmap)) + 1e-8  # normalization for colorbar

        # overlay pic (first col)
        ax_overlay = axes[shot_idx, 0]
        ax_overlay.imshow(img_to_show, cmap=cmap_img)
        # red = positive gain (helpful), blue = negative gain (noisy)
        img_overlay = ax_overlay.imshow(
            heatmap, cmap="RdBu_r", alpha=0.75, vmin=-max_abs, vmax=max_abs
        )
        ax_overlay.set_title(f"shot {shot_idx+1} saliency map", fontsize=8)
        ax_overlay.axis("off")

        # original pic (second col)
        ax_orig = axes[shot_idx, 1]
        ax_orig.imshow(img_to_show, cmap=cmap_img)
        ax_orig.set_title(f"original shot {shot_idx+1}", fontsize=8)
        ax_orig.axis("off")

    # add colorbar to show color meaning
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(
        img_overlay,
        cax=cbar_ax,
        label="feature contribution\n(red: helpful, blue: harmful)",
    )

    # title
    gain_text = (
        f"adaptation gain: {adaptation_gain:.2f}%"
        if adaptation_gain is not None
        else ""
    )
    plt.suptitle(
        f"explaination - task {metabatch_id}-{task_id}\n{gain_text}",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    plt.subplots_adjust(right=0.9, top=0.92, wspace=0.1, hspace=0.3)

    save_path = os.path.join(
        log_dir, "plots", f"{algo}_task{metabatch_id}-{task_id}_fama_trajectory.png"
    )
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
