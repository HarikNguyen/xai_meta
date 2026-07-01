import copy
import torch
import torch.nn as nn
import math
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr


def randomize_layer(weight):
    # apply Kaiming Uniform for weight.dim >= 2
    if weight is not None and weight.dim() >= 2:
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    
    # if weight is the bias layer or weight.dim == 1
    elif weight is not None:
        nn.init.uniform_(weight, -0.1, 0.1)

    return weight

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

def check_on_task(explainer, theta_0, sup_x, sup_y, que_x, que_y, T):
    task_pearson = []
    task_spearman = []

    explainer.theta_0 = [p.clone().detach() for p in theta_0]
    _, orig_saliency_map = explainer.interpret(sup_x, sup_y, que_x, que_y, T)

    corrupted_theta = [p.clone().detach() for p in theta_0]
    for param_idx in range(len(corrupted_theta) - 1, -1, -1):
        destroyed_weight = randomize_layer(corrupted_theta[param_idx])
        corrupted_theta[param_idx] = destroyed_weight
        explainer.theta_0 = [p.clone().detach() for p in corrupted_theta]
        _, new_saliency_map = explainer.interpret(sup_x, sup_y, que_x, que_y, T)

        scores = correlation_sample_wise(orig_saliency_map, new_saliency_map)
        task_pearson.append(scores["pearson"])
        task_spearman.append(scores["spearman"])
    
    return task_pearson, task_spearman


def sanity_check(explainer, test_loader, T):
    test_loader_pbar = tqdm(
        test_loader, desc="Sanity Check", position=0, leave=True, unit="boT"
    )
    theta_0 = [p.clone().detach() for p in explainer.algo_mgr.theta_0]

    for metabatch_id, boT in enumerate(test_loader_pbar):
        boT_pbar = tqdm(
            boT, desc=f"Batch {metabatch_id}", position=1, leave=False, unit="task"
        )
        for task_id, (support, query) in enumerate(boT_pbar):
            sup_x, sup_y = support
            que_x, que_y = query

            task_pear, task_spear = check_on_task(explainer, theta_0, sup_x, sup_y, que_x, que_y, T)
            
            for idx, (p, s) in enumerate(zip(mean_pearson, mean_spearman)):
                print(f"Step {idx+1} (Randomized to Layer Index {len(theta_0)-1-idx}): Pearson={p:.4f}, Spearman={s:.4f}")

    return {
        "mean_pearson_steps": mean_pearson,
        "mean_spearman_steps": mean_spearman
    }
