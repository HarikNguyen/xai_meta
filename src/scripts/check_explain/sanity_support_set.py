import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

def hard_task_mining():
    pass

def permute_label(sup_y, flip_ratio=0.6):
    N, C = sup_y.shape
    device = sup_y.device
    sup_y_np = sup_y.clone().detach().cpu().numpy()

    num_flip = int(N * flip_ratio)
    flip_indices = np.random.choice(N, num_flip, replace=False)
    avail_indices = list(flip_indices)

    for idx, flip_idx in enumerate(flip_indices):
        orig_val = sup_y[flip_idx]
        avail_indices.pop(idx)
        
        while True:
            per_idx = np.random.choice(avail_indices)
            per_val = sup_y[per_idx]
            print(per_val)
            print(orig_val)
            print()
            if not torch.equal(per_val, orig_val):
                sup_y_np[idx] = per_val
                break

    sup_y_np = torch.from_numpy(sup_y_np).to(device)
    return sup_y_np

def mix_set():
    pass

def correlation_sample_wise(A, A_prime):
    N = A.shape[0]
    a_np = A.detach().cpu().numpy().reshape(N, -1)
    a_prime_np = A_prime.detach().cpu().numpy().reshape(N, -1)
    pearson_list = []
    spearman_list = []

    for i in range(N):
        img = a_np[i]
        img_prime = a_prime_np[i]
        
        # Pearson
        p_corr, _ = pearsonr(img, img_prime)
        p_corr = 0.0 if np.isnan(p_corr) else p_corr
        pearson_list.append(p_corr)
        
        # Spearman
        s_corr, _ = spearmanr(img, img_prime)
        s_corr = 0.0 if np.isnan(s_corr) else s_corr
        spearman_list.append(s_corr)

    return {
        "pearson": np.mean(pearson_list),
        "spearman": np.mean(spearman_list)
    }

def sanity_check_support_set(explainer, test_loader, T):
    test_loader_pbar = tqdm(
        test_loader, desc="Sanity Check", position=0, leave=True, unit="boT"
    )
    theta_0 = [p.clone().detach() for p in explainer.algo_mgr.theta_0]
    noisy_check_results = {
        "pearson": [],
        "spearman": []
    }
    for metabatch_id, boT in enumerate(test_loader_pbar):
        boT_pbar = tqdm(
            boT, desc=f"Batch {metabatch_id}", position=1, leave=False, unit="task"
        )
        for task_id, (support, query) in enumerate(boT_pbar):
            sup_x, sup_y = support
            que_x, que_y = query
            
            # check noisy task
            scores = check_on_noisy_task(explainer, sup_x, sup_y, que_x, que_y, T)
            noisy_check_results["pearson"].append(scores["pearson"])
            noisy_check_results["spearman"].append(scores["spearman"])
            print(f"Task {task_id}: Pearson={scores['pearson']:.4f}, Spearman={scores['spearman']:.4f}")

        break 
    results = {
        "noisy_check": noisy_check_results
    }

    return results

def check_on_noisy_task(explainer, sup_x, sup_y, que_x, que_y, T):
    sup_y_noisy = permute_label(sup_y, flip_ratio=0.6)
    print(sup_y, sup_y_noisy)
    
    _, orig_saliency_map = explainer.interpret(sup_x, sup_y, que_x, que_y, T)
    _, noisy_saliency_map = explainer.interpret(sup_x, sup_y_noisy, que_x, que_y, T)

    scores = correlation_sample_wise(orig_saliency_map, noisy_saliency_map)
    return scores
