import numpy as np
import torch
from tqdm import tqdm

from ..utils import correlation_sample_wise, blur_sup, permute_label

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

def sanity_check_support_set(explainer, test_loader, ood_test_loader, T):
    test_loader_pbar = tqdm(
        test_loader, desc="Sanity Check", position=0, leave=True, unit="boT"
    )
    theta_0 = [p.clone().detach() for p in explainer.algo_mgr.theta_0]
    
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

    for metabatch_id, boT in enumerate(test_loader_pbar):
        boT_pbar = tqdm(
            boT, desc=f"Batch {metabatch_id}", position=1, leave=False, unit="task"
        )
        ood_iter = iter(ood_test_loader)
        boT_ood = next(ood_iter)
        for task_id, (support, query) in enumerate(boT_pbar):
            sup_x, sup_y, _ = support
            que_x, que_y, _ = query
            
            # check on noisy task
            scores = check_on_noisy_task(explainer, sup_x, sup_y, que_x, que_y, T)
            noisy_check_results["pearson"].append(scores["pearson"])
            noisy_check_results["spearman"].append(scores["spearman"])

            # check on hard task
            scores = check_on_hard_task(explainer, sup_x, sup_y, que_x, que_y, T)
            hard_check_results["pearson"].append(scores["pearson"])
            hard_check_results["spearman"].append(scores["spearman"])
            
            # check on mixed task (ood task)
            boT_ood_task = boT_ood[task_id]
            scores = check_on_mixed_task(explainer, (support, query), boT_ood_task, T)
            ood_check_results["pearson"].append(scores["pearson"])
            ood_check_results["spearman"].append(scores["spearman"])

    results = {
        "noisy_check": noisy_check_results,
        "hard_check": hard_check_results,
        "ood_check": ood_check_results,
    }

    return results

def check_on_noisy_task(explainer, sup_x, sup_y, que_x, que_y, T):
    sup_y_noisy = permute_label(sup_y, flip_ratio=0.8)

    _, orig_saliency_map = explainer.interpret(sup_x, sup_y, que_x, que_y, T)
    _, noisy_saliency_map = explainer.interpret(sup_x, sup_y_noisy, que_x, que_y, T)

    scores = correlation_sample_wise(orig_saliency_map, noisy_saliency_map)
    return scores

def check_on_hard_task(explainer, sup_x, sup_y, que_x, que_y, T):
    sup_x_hard = blur_sup(sup_x, kernel_size=7, sigma=3.0)
    _, orig_saliency_map = explainer.interpret(sup_x, sup_y, que_x, que_y, T)
    _, hard_saliency_map = explainer.interpret(sup_x_hard, sup_y, que_x, que_y, T)

    scores = correlation_sample_wise(orig_saliency_map, hard_saliency_map)
    return scores

def check_on_mixed_task(explainer, source_task, another_task, T):
    (sup_x, sup_y, _), (que_x, que_y, _) = source_task
    (a_sup_x, a_sup_y, _), (a_que_x, a_que_y, _) = another_task

    (ood_sup_x, ood_sup_y), (ood_que_x, ood_que_y) = mix_set(
            ((sup_x, sup_y), (que_x, que_y)),
            ((a_sup_x, a_sup_y), (a_que_x, a_que_y)),
            num_mixed_classes=2)
    
    _, orig_saliency_map = explainer.interpret(sup_x, sup_y, que_x, que_y, T)
    _, mixed_saliency_map = explainer.interpret(ood_sup_x, ood_sup_y, ood_que_x, ood_que_y, T)

    scores = correlation_sample_wise(orig_saliency_map, mixed_saliency_map)
    return scores
