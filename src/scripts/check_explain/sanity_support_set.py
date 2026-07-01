import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms.functional as vF
import math
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from collections import Counter

def blur_sup(sup_x, kernel_size=7, sigma=3.0):
    sup_x_blurred = sup_x.clone()
    sup_x_blurred = vF.gaussian_blur(
        sup_x_blurred, 
        kernel_size=[kernel_size, kernel_size], 
        sigma=[sigma, sigma]
    )
    return sup_x_blurred

def permute_label(sup_y, flip_ratio=0.6):
    N, C = sup_y.shape
    device = sup_y.device
    sup_y_np = sup_y.clone().detach().cpu().numpy()

    # 1. Lấy ngẫu nhiên các chỉ số (indices) cần xáo trộn nhãn
    num_flip = int(N * flip_ratio)
    if num_flip <= 1:
        # Không đủ phần tử để hoán vị
        return torch.from_numpy(sup_y_np).to(device)
        
    flip_indices = np.random.choice(N, num_flip, replace=False)

    # 2. Lấy ra các nhãn tương ứng của các vị trí cần xáo trộn
    # Để thuật toán tối ưu hoạt động, ta chuyển nhãn sang dạng số nguyên (0, 1, 2... C-1)
    # Nếu sup_y_np là nhãn dạng số nguyên sẵn (N, 1), bỏ qua argmax. Ở đây giả định dạng (N, C)
    labels = np.argmax(sup_y_np[flip_indices], axis=1)

    # 3. Áp dụng thuật toán Sắp xếp & Dịch chuyển (Sort & Shift)
    # Lưu lại index gốc trong nhóm cần flip để khôi phục
    indexed_labels = sorted(enumerate(labels), key=lambda x: x[1])
    
    # Đếm số lần xuất hiện của nhãn phổ biến nhất trong nhóm này
    counts = Counter(labels)
    max_freq = max(counts.values())
    
    # Dịch chuyển vòng tròn mảng đã sắp xếp một khoảng bằng max_freq
    # Việc dịch chuyển này ép các nhãn trùng nhau rời xa vị trí của nhau nhất có thể
    shifted_indexed = indexed_labels[-max_freq:] + indexed_labels[:-max_freq]
    
    # 4. Ghi đè nhãn mới đã hoán vị tối ưu ngược lại mảng sup_y_np
    # Tạo một bản sao lưu tạm thời của các vector nhãn gốc trước khi bị ghi đè
    temp_targets = sup_y_np[flip_indices].copy()
    
    for i in range(num_flip):
        original_pos_in_flip = indexed_labels[i][0]
        # Vị trí thực tế trên ma trận sup_y_np
        actual_global_idx = flip_indices[original_pos_in_flip] 
        
        # Lấy vector nhãn từ vị trí được dịch chuyển tới
        from_pos_in_flip = shifted_indexed[i][0]
        
        # Ghi đè vector nhãn (one-hot hoặc xác suất)
        sup_y_np[actual_global_idx] = temp_targets[from_pos_in_flip]

    # 5. Chuyển lại về Tensor và trả về thiết bị cũ
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
    hard_check_results = {
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
            
            # check on noisy task
            scores = check_on_noisy_task(explainer, sup_x, sup_y, que_x, que_y, T)
            noisy_check_results["pearson"].append(scores["pearson"])
            noisy_check_results["spearman"].append(scores["spearman"])
            print(f"Task {task_id}: Pearson={scores['pearson']:.4f}, Spearman={scores['spearman']:.4f}")

            # check on hard task
            scores = check_on_hard_task(explainer, sup_x, sup_y, que_x, que_y, T)
            hard_check_results["pearson"].append(scores["pearson"])
            hard_check_results["spearman"].append(scores["spearman"])
            print(f"Task {task_id}: Pearson={scores['pearson']:.4f}, Spearman={scores['spearman']:.4f}")


    results = {
        "noisy_check": noisy_check_results
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
