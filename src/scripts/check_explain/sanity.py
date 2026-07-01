import copy
import matplotlib.pyplot as plt
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

    corrupted_saliencies = []
    corrupted_theta = [p.clone().detach() for p in theta_0]
    for param_idx in range(len(corrupted_theta) - 1, -1, -1):
        destroyed_weight = randomize_layer(corrupted_theta[param_idx])
        corrupted_theta[param_idx] = destroyed_weight
        explainer.theta_0 = [p.clone().detach() for p in corrupted_theta]
        _, new_saliency_map = explainer.interpret(sup_x, sup_y, que_x, que_y, T)

        scores = correlation_sample_wise(orig_saliency_map, new_saliency_map)
        task_pearson.append(scores["pearson"])
        task_spearman.append(scores["spearman"])

        corrupted_saliencies.append((param_idx, new_saliency_map[0]))
        
    save_task_saliency_grid(sup_x[0], orig_saliency_map[0], corrupted_saliencies, save_path=f"task_{sup_x.shape[0]}_saliency_grid.png", alpha=0.5)
    return task_pearson, task_spearman

def save_task_saliency_grid(image_tensor, orig_saliency, corrupted_saliencies, save_path, alpha=0.5):
    """
    Lưu ảnh gốc và tất cả saliency map (original + corrupted) vào cùng 1 ảnh (grid).
    
    Args:
        image_tensor: Tensor ảnh gốc (C, H, W)
        orig_saliency: Tensor saliency map gốc
        corrupted_saliencies: List chứa các tuple (layer_idx, saliency_tensor)
        save_path: Đường dẫn lưu ảnh tổng
        alpha: Độ trong suốt của saliency map
    """
    # 1. Xử lý ảnh gốc để hiển thị
    img = image_tensor.squeeze().cpu().detach().numpy()
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))
        
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 0:
        img = (img - img_min) / (img_max - img_min)
        
    cmap_img = 'gray' if img.ndim == 2 or img.shape[-1] == 1 else None

    # 2. Tính toán lưới (grid) hiển thị
    total_plots = 1 + len(corrupted_saliencies) # 1 original + N corrupted
    cols = min(5, total_plots)                  # Tối đa 5 cột cho dễ nhìn
    rows = math.ceil(total_plots / cols)        # Tính số hàng cần thiết

    # 3. Khởi tạo figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    
    # Đưa axes về dạng list 1 chiều để dễ lặp (xử lý trường hợp chỉ có 1 hàng)
    if total_plots > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    def plot_overlay(ax, saliency_tensor, title):
        sal = saliency_tensor.squeeze().cpu().detach().numpy()
        ax.imshow(img, cmap=cmap_img)
        ax.imshow(sal, cmap='jet', alpha=alpha)
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    # 4. Vẽ ảnh Original vào ô đầu tiên
    plot_overlay(axes[0], orig_saliency, "Original Saliency")

    # 5. Vẽ các ảnh Corrupted vào các ô tiếp theo
    for i, (layer_idx, sal_tensor) in enumerate(corrupted_saliencies):
        plot_overlay(axes[i + 1], sal_tensor, f"Corrupted Layer {layer_idx}")

    # 6. Ẩn các ô trống dư thừa (nếu grid lớn hơn số ảnh thực tế)
    for i in range(total_plots, len(axes)):
        axes[i].axis('off')

    # 7. Lưu lại thành 1 ảnh duy nhất
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def sanity_check(explainer, test_loader, T):
    test_loader_pbar = tqdm(
        test_loader, desc="Sanity Check", position=0, leave=True, unit="boT"
    )
    theta_0 = [p.clone().detach() for p in explainer.algo_mgr.theta_0]

    results = {
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

            task_pearson, task_spearman = check_on_task(explainer, theta_0, sup_x, sup_y, que_x, que_y, T)
            results["pearson"].append(task_pearson)
            results["spearman"].append(task_spearman)
            break

    return results
