import functools
import os

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # plot saving runs on a background thread; GUI backends need the main thread
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..utils import correlation_sample_wise, blur_sup, permute_label, parallel_map, submit_plot_task

def mix_set(task_source, task_another, num_mixed_classes=2):
    (task_ssx, task_ssy), (task_sqx, task_sqy) = task_source
    (task_asx, task_asy), (task_aqx, task_aqy) = task_another

    mixed_sx = task_ssx.clone()
    mixed_sy = task_ssy.clone()

    C_source = task_ssy.size(1)
    C_another = task_asy.size(1)
    if num_mixed_classes >= C_source:
        raise ValueError("num_mixed_classes must be less than the number of classes in the source task.")

    target_classes_in_source = torch.randperm(C_source)[:num_mixed_classes]
    source_classes_from_another = torch.randperm(C_another)[:num_mixed_classes]

    ood_indices = []
    for i in range(num_mixed_classes):
        tgt_c = target_classes_in_source[i].item()
        src_c = source_classes_from_another[i].item()

        idx_tgt = torch.where(task_ssy[:, tgt_c] == 1)[0]
        idx_src = torch.where(task_asy[:, src_c] == 1)[0]

        num_replace = min(len(idx_tgt), len(idx_src))
        idx_tgt = idx_tgt[:num_replace]
        idx_src = idx_src[:num_replace]

        mixed_sx[idx_tgt] = task_asx[idx_src]
        ood_indices.extend(idx_tgt.tolist())

    ood_indices.sort()
    task_mixed = ((mixed_sx, mixed_sy), (task_sqx, task_sqy))

    return task_mixed

def check_on_noisy_task(explainer, sup_x, sup_y, que_x, que_y, T, orig_saliency_map):
    sup_y_noisy = permute_label(sup_y, flip_ratio=0.8)
    _, noisy_saliency_map = explainer.interpret(sup_x, sup_y_noisy, que_x, que_y, T)
    scores = correlation_sample_wise(orig_saliency_map, noisy_saliency_map)
    return scores, sup_x, noisy_saliency_map

def check_on_hard_task(explainer, sup_x, sup_y, que_x, que_y, T, orig_saliency_map):
    sup_x_hard = blur_sup(sup_x, kernel_size=7, sigma=3.0)
    _, hard_saliency_map = explainer.interpret(sup_x_hard, sup_y, que_x, que_y, T)
    scores = correlation_sample_wise(orig_saliency_map, hard_saliency_map)
    return scores, sup_x_hard, hard_saliency_map

def check_on_mixed_task(explainer, source_task, another_task, T, orig_saliency_map):
    (sup_x, sup_y, _), (que_x, que_y, _) = source_task
    (a_sup_x, a_sup_y, _), (a_que_x, a_que_y, _) = another_task

    (ood_sup_x, ood_sup_y), (ood_que_x, ood_que_y) = mix_set(
            ((sup_x, sup_y), (que_x, que_y)),
            ((a_sup_x, a_sup_y), (a_que_x, a_que_y)),
            num_mixed_classes=2)

    _, mixed_saliency_map = explainer.interpret(ood_sup_x, ood_sup_y, ood_que_x, ood_que_y, T)
    scores = correlation_sample_wise(orig_saliency_map, mixed_saliency_map)
    return scores, ood_sup_x, mixed_saliency_map


def save_support_set_grid(orig_sup_x, orig_sal, variants, save_path, alpha=0.5):
    """One figure: rows = support images, columns = original + each perturbed
    variant (noisy-label / hard-blurred / OOD-mixed), overlaying that
    variant's saliency map and annotating the pearson/spearman correlation
    (vs. the original saliency) in the column title.
    variants: list of (name, sup_x_variant, saliency_variant, pearson, spearman)
    """
    num_rows = orig_sup_x.shape[0]
    num_cols = 1 + len(variants)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3.2 * num_rows))
    if num_rows == 1:
        axes = axes[np.newaxis, :]

    def _prep(img_t):
        img = img_t.squeeze().cpu().detach().numpy()
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        lo, hi = img.min(), img.max()
        if hi - lo > 0:
            img = (img - lo) / (hi - lo)
        return img

    for i in range(num_rows):
        ax0 = axes[i, 0]
        ax0.imshow(_prep(orig_sup_x[i]))
        ax0.imshow(orig_sal[i].squeeze().cpu().detach().numpy(), cmap="jet", alpha=alpha)
        ax0.set_xticks([]); ax0.set_yticks([])
        if i == 0:
            ax0.set_title("Original", fontsize=10, fontweight="bold")

        for j, (name, var_sup_x, var_sal, pearson, spearman) in enumerate(variants):
            ax = axes[i, j + 1]
            ax.imshow(_prep(var_sup_x[i]))
            ax.imshow(var_sal[i].squeeze().cpu().detach().numpy(), cmap="jet", alpha=alpha)
            ax.axis("off")
            if i == 0:
                ax.set_title(f"{name}\npearson={pearson:.2f} spearman={spearman:.2f}", fontsize=9)

    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

def sanity_check_support_set(explainer, test_loader, ood_test_loader, T, illustrate_dir=None, illustrate_n_tasks=0):
    test_loader_pbar = tqdm(
        test_loader, desc="Sanity Check", position=0, leave=True, unit="boT"
    )

    noisy_check_results = {
        "pearson": [],
        "spearman": []
    }
    hard_check_results = {
        "pearson": [],
        "spearman": []
    }
    ood_check_results = {
        "pearson": [],
        "spearman": []
    }

    # Create the OOD iterator ONCE outside the loop: recreating it every
    # metabatch (as before) always restarted it from the first batch, so the
    # OOD check silently reused the same batch instead of advancing through
    # ood_test_loader.
    ood_iter = iter(ood_test_loader)

    # Illustrate the max-gain and min-gain tasks (highest/lowest raw
    # adaptation_gain across the whole run), not an arbitrary first-N subset
    # -- those extremes are what's actually informative to inspect. Only the
    # current champions' data is kept in memory (replaced whenever a more
    # extreme task is found).
    champions = {"max": None, "min": None}

    for metabatch_id, boT in enumerate(test_loader_pbar):
        boT_pbar = tqdm(
            boT, desc=f"Batch {metabatch_id}", position=1, leave=False, unit="task"
        )
        boT_ood = next(ood_iter)
        for task_id, (support, query) in enumerate(boT_pbar):
            sup_x, sup_y, _ = support
            que_x, que_y, _ = query

            # Computed once per task and shared by all 3 checks below (was
            # previously recomputed independently inside each of them).
            gain, orig_saliency_map = explainer.interpret(sup_x, sup_y, que_x, que_y, T)

            boT_ood_task = boT_ood[task_id]

            # noisy / hard / mixed checks are independent of each other -> run concurrently
            fns = [
                functools.partial(check_on_noisy_task, explainer, sup_x, sup_y, que_x, que_y, T, orig_saliency_map),
                functools.partial(check_on_hard_task, explainer, sup_x, sup_y, que_x, que_y, T, orig_saliency_map),
                functools.partial(check_on_mixed_task, explainer, (support, query), boT_ood_task, T, orig_saliency_map),
            ]
            (noisy_scores, noisy_sup_x, noisy_sal), (hard_scores, hard_sup_x, hard_sal), \
                (ood_scores, ood_sup_x, ood_sal) = parallel_map(fns, device=explainer.device)

            noisy_check_results["pearson"].append(noisy_scores["pearson"])
            noisy_check_results["spearman"].append(noisy_scores["spearman"])

            hard_check_results["pearson"].append(hard_scores["pearson"])
            hard_check_results["spearman"].append(hard_scores["spearman"])

            ood_check_results["pearson"].append(ood_scores["pearson"])
            ood_check_results["spearman"].append(ood_scores["spearman"])

            if illustrate_dir is not None:
                variants = [
                    ("Noisy-label", noisy_sup_x.detach().cpu(), noisy_sal.detach().cpu(),
                     noisy_scores["pearson"], noisy_scores["spearman"]),
                    ("Hard (blurred)", hard_sup_x.detach().cpu(), hard_sal.detach().cpu(),
                     hard_scores["pearson"], hard_scores["spearman"]),
                    ("OOD-mixed", ood_sup_x.detach().cpu(), ood_sal.detach().cpu(),
                     ood_scores["pearson"], ood_scores["spearman"]),
                ]
                entry = {
                    "gain": gain, "metabatch_id": metabatch_id, "task_id": task_id,
                    "sup_x": sup_x.detach().cpu(), "orig_saliency_map": orig_saliency_map.detach().cpu(),
                    "variants": variants,
                }
                if champions["max"] is None or gain > champions["max"]["gain"]:
                    champions["max"] = entry
                if champions["min"] is None or gain < champions["min"]["gain"]:
                    champions["min"] = entry

    if illustrate_dir is not None:
        for label, champ in champions.items():
            if champ is None:
                continue
            save_path = os.path.join(
                illustrate_dir,
                f"sanity_supportset_{label}gain_task{champ['metabatch_id']}-{champ['task_id']}.png",
            )
            submit_plot_task(
                save_support_set_grid,
                champ["sup_x"], champ["orig_saliency_map"], champ["variants"], save_path,
            )

    results = {
        "noisy_check": noisy_check_results,
        "hard_check": hard_check_results,
        "ood_check": ood_check_results,
    }

    return results
