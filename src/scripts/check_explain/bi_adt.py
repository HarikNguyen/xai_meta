import functools
import os

import torch
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib
matplotlib.use("Agg")  # plot saving runs on a background thread; GUI backends need the main thread
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import ndimage
from skimage.segmentation import slic

from ..utils import parallel_map, submit_io_task, submit_plot_task


def _segment_and_score(img_np, sal_np, n_segs, compactness):
    """Run SLIC once for ONE image and compute the mean saliency per superpixel.
    Mode-independent: pos/neg/random only differ in how these scores get
    ranked, so this is computed once and reused by all three.
    """
    segs = slic(
        img_np, n_segments=n_segs, compactness=compactness,
        sigma=1.0, start_label=0, channel_axis=-1 if img_np.shape[-1] == 3 else None,
        enforce_connectivity=True
    )
    unique_sps = np.unique(segs)
    sp_scores = ndimage.mean(sal_np, labels=segs, index=unique_sps)
    return segs, unique_sps, sp_scores


def compute_rank_bases(sup_x, saliency_map, n_segs=150, compactness=10.0):
    """Cache the SLIC segmentation + per-segment saliency score for every image
    in the support set, ONCE per task (was previously recomputed from scratch
    for each of the pos/neg/random modes)."""
    imgs_np = sup_x.detach().cpu().numpy().transpose(0, 2, 3, 1)
    sal_np = saliency_map.detach().cpu().squeeze(1).numpy()
    # SLIC per image has no shared mutable state, so it's safe to fan out across the shared IO pool
    futures = [
        submit_io_task(_segment_and_score, imgs_np[i], sal_np[i], n_segs, compactness)
        for i in range(sup_x.shape[0])
    ]
    return [f.result() for f in futures]


def _rank_map_from_scores(segs, unique_sps, sp_scores, mode):
    """Turn cached (segments, per-segment score) into a per-pixel rank map for
    one mode, without re-running SLIC."""
    if mode == "pos":
        order = np.argsort(sp_scores)[::-1]   # remove the most positive (red) first
    elif mode == "neg":
        order = np.argsort(sp_scores)         # remove the most negative (blue) first
    else:
        order = np.random.permutation(len(unique_sps))  # remove in random order

    num_sps = len(unique_sps)
    sp_to_rank = np.empty(int(unique_sps.max()) + 1, dtype=np.float32)
    sp_to_rank[unique_sps[order]] = np.arange(1, num_sps + 1) / num_sps
    return sp_to_rank[segs]


def rank_tensor_from_bases(rank_bases, mode, device):
    """Build the [N, 1, H, W] GPU rank tensor for one mode from cached rank bases."""
    rank_maps = np.stack([
        _rank_map_from_scores(segs, unique_sps, sp_scores, mode)
        for segs, unique_sps, sp_scores in rank_bases
    ])
    return torch.from_numpy(rank_maps).unsqueeze(1).to(device)


def apply_mask_fast(sup_x, blurred_baseline, rank_tensor, ratio, blur_sigma=5.0):
    """Build the mask and apply it directly on the GPU using vectorized ops (no for-loop)."""
    mask = (rank_tensor <= ratio).float()

    if blur_sigma > 0:
        ksize = int(blur_sigma * 4) | 1
        mask = TF.gaussian_blur(mask, kernel_size=[ksize, ksize], sigma=[blur_sigma, blur_sigma])
        mask = torch.clamp(mask, 0.0, 1.0)

    return sup_x * (1 - mask) + blurred_baseline * mask


def _interpret_gain(explainer, sup_x_masked, sup_y, que_x, que_y, T):
    # only the scalar gain is used, so skip interpret()'s expensive saliency machinery
    return explainer.compute_gain_only(sup_x_masked, sup_y, que_x, que_y, T)


def _montage(images_tensor):
    """Concatenate a (N, C, H, W) tensor of support images side by side into
    one (H, N*W, C) numpy array in [0,1], for a single-cell preview."""
    imgs = images_tensor.detach().cpu()
    n = imgs.shape[0]
    strip = torch.cat([imgs[i] for i in range(n)], dim=-1)  # (C, H, N*W)
    arr = strip.permute(1, 2, 0).numpy()
    lo, hi = arr.min(), arr.max()
    if hi - lo > 0:
        arr = (arr - lo) / (hi - lo)
    return arr


def save_biadt_mask_grid(sup_x, mode_step_masked, mode_step_gain, ratios, save_path):
    """Grid: rows = pos/neg/random deletion modes, columns = original + a few mask
    ratios; each cell is a support-set montage annotated with its adaptation gain."""
    modes = ("pos", "neg", "random")
    mode_titles = {
        "pos": "pos (remove most helpful first)",
        "neg": "neg (remove most harmful first)",
        "random": "random",
    }
    cols = [0.0] + ratios
    fig, axes = plt.subplots(len(modes), len(cols), figsize=(3 * len(cols), 3.2 * len(modes)))
    if len(modes) == 1:
        axes = axes[np.newaxis, :]

    orig_montage = _montage(sup_x)
    for row, mode in enumerate(modes):
        for col, ratio in enumerate(cols):
            ax = axes[row, col]
            if ratio == 0.0:
                ax.imshow(orig_montage)
                gain_text = f"gain={mode_step_gain[(mode, 'base')]:.2f}%"
            else:
                ax.imshow(_montage(mode_step_masked[(mode, ratio)]))
                gain_text = f"gain={mode_step_gain[(mode, ratio)]:.2f}%"
            ax.set_title(gain_text, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(mode_titles[mode], fontsize=9, fontweight="bold")

    fig.suptitle("biADT -- deletion direction vs. adaptation gain", fontsize=13, fontweight="bold")
    fig.subplots_adjust(wspace=0.05, hspace=0.25, top=0.9)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def compute_bidirectional_faithfulness(
    explainer, test_loader, T, n_segs=150, compactness=10.0, blur_sigma=5.0, num_steps=10,
    illustrate_dir=None, illustrate_n_tasks=0,
):
    # illustrate_n_tasks unused: always illustrates max/min-gain tasks, kept for a uniform signature
    test_loader_pbar = tqdm(test_loader, desc="BiDAT", position=0, leave=True, unit="boT")
    pdas, ndas, combines = [], [], []
    ratios = [step / num_steps for step in range(1, num_steps + 1)]
    # a handful of evenly-spaced ratios to render (all of them would make the grid too wide)
    display_ratios = [ratios[i] for i in np.linspace(0, len(ratios) - 1, min(5, len(ratios))).astype(int)]

    # track the max/min adaptation_gain tasks; only their display data is kept in memory
    champions = {"max": None, "min": None}

    for metabatch_id, boT in enumerate(test_loader_pbar):
        boT_pbar = tqdm(boT, desc=f"Batch {metabatch_id}", position=1, leave=False, unit="task")

        for task_id, (support, query) in enumerate(boT_pbar):
            sup_x, sup_y, _ = support
            que_x, que_y, _ = query

            # Base gain + saliency map, shared by all 3 modes below
            adapt_gain_base, saliency_map = explainer.interpret(sup_x, sup_y, que_x, que_y, T)

            # Mode-independent pre-computation, done ONCE (was 3x before)
            rank_bases = compute_rank_bases(sup_x, saliency_map, n_segs, compactness)
            blurred_baseline = TF.gaussian_blur(sup_x, kernel_size=[11, 11], sigma=[5.0, 5.0])

            # Flatten pos/neg/random x num_steps into ONE job list for the shared executor
            jobs = []
            for mode in ("pos", "neg", "random"):
                rank_tensor = rank_tensor_from_bases(rank_bases, mode, sup_x.device)
                for ratio in ratios:
                    masked = apply_mask_fast(sup_x, blurred_baseline, rank_tensor, ratio, blur_sigma)
                    jobs.append((mode, ratio, masked))

            fns = [
                functools.partial(_interpret_gain, explainer, masked, sup_y, que_x, que_y, T)
                for _, _, masked in jobs
            ]
            gains = parallel_map(fns, device=sup_x.device)

            aucs = {}
            mode_step_gain = {}
            mode_step_masked = {}
            for mode in ("pos", "neg", "random"):
                mode_step_gain[(mode, "base")] = adapt_gain_base
                mode_gains = [adapt_gain_base]
                for (m, ratio, masked), g in zip(jobs, gains):
                    if m != mode:
                        continue
                    mode_gains.append(g)
                    if illustrate_dir is not None and ratio in display_ratios:
                        mode_step_gain[(mode, ratio)] = g
                        mode_step_masked[(mode, ratio)] = masked.detach().cpu()
                mode_ratios = [0.0] + ratios
                aucs[mode] = np.trapezoid(mode_gains, mode_ratios)

            if illustrate_dir is not None:
                if champions["max"] is None or adapt_gain_base > champions["max"]["gain"]:
                    champions["max"] = {
                        "gain": adapt_gain_base, "metabatch_id": metabatch_id, "task_id": task_id,
                        "sup_x": sup_x.detach().cpu(), "mode_step_gain": dict(mode_step_gain),
                        "mode_step_masked": dict(mode_step_masked),
                    }
                if champions["min"] is None or adapt_gain_base < champions["min"]["gain"]:
                    champions["min"] = {
                        "gain": adapt_gain_base, "metabatch_id": metabatch_id, "task_id": task_id,
                        "sup_x": sup_x.detach().cpu(), "mode_step_gain": dict(mode_step_gain),
                        "mode_step_masked": dict(mode_step_masked),
                    }

            pda = aucs["random"] - aucs["pos"]
            nda = aucs["neg"] - aucs["random"]
            combined = pda + nda

            pdas.append(pda)
            ndas.append(nda)
            combines.append(combined)

    if illustrate_dir is not None:
        for label, champ in champions.items():
            if champ is None:
                continue
            save_path = os.path.join(
                illustrate_dir,
                f"biadt_{label}gain_task{champ['metabatch_id']}-{champ['task_id']}.png",
            )
            submit_plot_task(
                save_biadt_mask_grid,
                champ["sup_x"], champ["mode_step_masked"], champ["mode_step_gain"], display_ratios, save_path,
            )

    return pdas, ndas, combines
