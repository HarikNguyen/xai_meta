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

def max_change_permutation(lo_tuple):
    if len(lo_tuple) < 2:
        return lo_tuple  # Không thể hoán vị nếu có ít hơn 2 phần tử

    n = len(lo_tuple)
    
    # 1. Đếm tần suất xuất hiện của các nhãn (label nằm ở vị trí tuple[1])
    labels = [item[1].item() if isinstance(item[1], torch.Tensor) else item[1] for item in lo_tuple]
    counts = Counter(labels)
    
    # 2. Sắp xếp danh sách tuple dựa trên tần suất của nhãn (giảm dần)
    # Điều này đưa các tuple có nhãn giống nhau đứng cạnh nhau
    sorted_tuple = sorted(lo_tuple, key=lambda x: (counts[x[1].item() if isinstance(x[1], torch.Tensor) else x[1]], x[1]), reverse=True)
    
    # 3. Tính toán độ dịch chuyển K tối ưu
    max_freq = counts.most_common(1)[0][1]
    shift = n // 2
    if max_freq > shift:
        shift = max_freq  # Bắt buộc dịch chuyển bằng số lượng nhãn phổ biến nhất nếu nó chiếm đa số
        
    # 4. Tạo danh sách nhãn mới bằng cách dịch vòng (chỉ lấy phần nhãn ở vị trí [1])
    permuted_labels = [None] * n
    for i in range(n):
        permuted_labels[(i + shift) % n] = sorted_tuple[i][1]
        
    # 5. Ghép nhãn mới đã hoán vị lại với các chỉ số ban đầu (index nằm ở vị trí [0])
    # final_tuple[original_idx] sẽ chứa nhãn mới
    result_tuple = []
    for i in range(n):
        original_idx = sorted_tuple[i][0]
        new_label = permuted_labels[i]
        result_tuple.append((original_idx, new_label))
        
    return result_tuple

def flip_label(sup_y, flip_ratio=0.6):
    num_samples = sup_y.size(0)
    sup_y_noisy = sup_y.clone()

    num_flip = int(num_samples * flip_ratio)
    if num_flip == 0:
        return sup_y_noisy

    # 1. Lựa chọn ngẫu nhiên các chỉ số để tiến hành làm nhiễu (flip)
    flip_indices = torch.randperm(num_samples)[:num_flip].tolist()
    flip_lo_tuple = [(idx, sup_y_noisy[idx]) for idx in flip_indices]
    
    # 2. Gọi hàm hoán vị tối đa nhãn trên các chỉ số đã chọn
    permuted_lo_tuple = max_change_permutation(flip_lo_tuple)
    
    # 3. Cập nhật các nhãn mới đã hoán vị vào tensor sup_y_noisy
    for idx, new_label in permuted_lo_tuple:
        sup_y_noisy[idx] = new_label

    return sup_y_noisy   


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

            
    results = {
        "noisy_check": noisy_check_results
    }

    return results

def check_on_noisy_task(explainer, sup_x, sup_y, que_x, que_y, T):
    sup_y_noisy = flip_label(sup_y, flip_ratio=0.6)
    
    _, orig_saliency_map = explainer.interpret(sup_x, sup_y, que_x, que_y, T)
    _, noisy_saliency_map = explainer.interpret(sup_x, sup_y_noisy, que_x, que_y, T)

    scores = correlation_sample_wise(orig_saliency_map, noisy_saliency_map)
    return scores
