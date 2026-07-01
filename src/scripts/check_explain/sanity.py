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

def save_full_nxm_grid(images_tensor, orig_saliencies, corrupted_data, save_path, alpha=0.5):
    """
    Vẽ 1 ảnh duy nhất chứa lưới N (ảnh) x M (trạng thái corrupt).
    """
    num_samples = images_tensor.shape[0]        # Số hàng (N)
    num_cols = 1 + len(corrupted_data)          # Số cột (M = Original + L corrupted layers)
    
    # 1. Khởi tạo figure lớn. Điều chỉnh figsize tùy thuộc vào N và số layer.
    # Ví dụ: mỗi ô 3x3 inch.
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(num_cols * 3, num_samples * 3))
    
    # 2. Xử lý trường hợp chỉ có 1 sample hoặc 1 cột để đảm bảo axes là mảng 2D (num_samples, num_cols)
    if num_samples == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_samples == 1:
        axes = np.array([axes])
    elif num_cols == 1:
        axes = np.array([[ax] for ax in axes])

    # 3. Vòng lặp vẽ từng ô trong lưới
    for i in range(num_samples):
        # A. Tiền xử lý ảnh gốc thứ i để hiển thị (H, W, C)
        img = images_tensor[i].squeeze().cpu().detach().numpy()
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))
            
        img_min, img_max = img.min(), img.max()
        if img_max - img_min > 0:
            img = (img - img_min) / (img_max - img_min)
            
        cmap_img = 'gray' if img.ndim == 2 or img.shape[-1] == 1 else None

        # --- Vẽ Cột 0: Original Saliency của ảnh i ---
        ax_orig = axes[i, 0]
        sal_orig = orig_saliencies[i].squeeze().cpu().detach().numpy()
        
        ax_orig.imshow(img, cmap=cmap_img)
        ax_orig.imshow(sal_orig, cmap='jet', alpha=alpha)
        
        # Chỉ hiện tiêu đề ở hàng đầu tiên
        if i == 0:
            ax_orig.set_title("Original", fontsize=10, fontweight='bold')
        
        # Hiện index ảnh ở cột đầu tiên bên trái
        if num_cols > 0:
            ax_orig.set_ylabel(f"Img {i}", fontsize=10, fontweight='bold')
            ax_orig.set_yticks([]) # Ẩn vạch tick Y nhưng giữ Label Y

        ax_orig.set_xticks([]) # Ẩn tick X
        # ax_orig.axis('off') # Nếu dùng .axis('off') sẽ ẩn luôn set_ylabel, ta nên ẩn thủ công

        # --- Vẽ các cột tiếp theo: Corrupted layers ---
        for j, (layer_idx, all_new_saliencies) in enumerate(corrupted_data):
            ax_corr = axes[i, j + 1]
            sal_corr = all_new_saliencies[i].squeeze().cpu().detach().numpy()
            
            ax_corr.imshow(img, cmap=cmap_img)
            ax_corr.imshow(sal_corr, cmap='jet', alpha=alpha)
            
            # Chỉ hiện tiêu đề layer ở hàng đầu tiên
            if i == 0:
                ax_corr.set_title(f"Corr Layer {layer_idx}", fontsize=10)
            
            ax_corr.axis('off') # Ẩn trục hoàn toàn ở các ô bên trong

    # 4. Tinh chỉnh và lưu
    # plt.tight_layout(pad=0.5) # Giảm khoảng cách giữa các ô
    fig.subplots_adjust(wspace=0.05, hspace=0.05) # Hoặc chỉnh thủ công hẹp hơn
    plt.savefig(save_path, bbox_inches='tight', dpi=150) #dpi cao hơn để ảnh sắc nét
    plt.close()

def get_layer_parameters_map(baselearner, theta_0):
    """
    Map danh sách phẳng theta_0 vào các Layer thực tế của baselearner
    """
    param_iterator = iter(theta_0)
    net_info_with_params = []
    
    # Lặp qua các sub-modules chứa param theo đúng thứ tự của PyTorch
    for name, module in baselearner.named_modules():
        # Chỉ xét module có parameter cục bộ (không tính module cha bọc ngoài)
        local_params = list(module.parameters(recurse=False))
        if len(local_params) > 0:
            layer_dict = {
                "name": name,
                "type": module.__class__.__name__,
                "params": []
            }
            # Bốc chính xác các tensor từ theta_0 ra theo đúng số lượng param của layer đó
            for _ in range(len(local_params)):
                layer_dict["params"].append(next(param_iterator))
                
            net_info_with_params.append(layer_dict)
            
    return net_info_with_params

def check_on_task(explainer, theta_0, net_layers, sup_x, sup_y, que_x, que_y, T):
    task_pearson = []
    task_spearman = []

    explainer.theta_0 = [p.clone().detach() for p in theta_0]
    _, orig_saliency_map = explainer.interpret(sup_x, sup_y, que_x, que_y, T)

    corrupted_saliencies = []
    corrupted_theta_grouped = copy.deepcopy(net_layers)
    for layer_idx in range(len(corrupted_theta_grouped) - 1, -1, -1):
        layer = randomize_layer(corrupted_theta_grouped[layer_idx])
        # destroy layer
        for param_tensor in layer["params"]:
            param_tensor = randomize_layer(param_tensor)
        corrupted_theta = []
        for l in corrupted_theta_grouped:
            corrupted_theta.extend(l["params"])
        explainer.theta_0 = [p.clone().detach() for p in corrupted_theta]
        _, new_saliency_map = explainer.interpret(sup_x, sup_y, que_x, que_y, T)

        scores = correlation_sample_wise(orig_saliency_map, new_saliency_map)
        task_pearson.append(scores["pearson"])
        task_spearman.append(scores["spearman"])

        corrupted_saliencies.append((layer_idx, new_saliency_map))
        
    save_full_nxm_grid(
        images_tensor=sup_x,              # (N, C, H, W)
        orig_saliencies=orig_saliency_map, # (N, H, W)
        corrupted_data=corrupted_saliencies,     # List of (layer_idx, (N, H, W))
        save_path=f"task_{sup_x.shape[0]}_saliency_grid.png", 
        alpha=0.5
    )
    return task_pearson, task_spearman

def sanity_check(explainer, test_loader, T):
    test_loader_pbar = tqdm(
        test_loader, desc="Sanity Check", position=0, leave=True, unit="boT"
    )
    theta_0 = [p.clone().detach() for p in explainer.algo_mgr.theta_0]
    net_layers = get_layer_parameters_map(explainer.algo_mgr.baselearner, theta_0) 
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

            task_pearson, task_spearman = check_on_task(explainer, theta_0, net_layers, sup_x, sup_y, que_x, que_y, T)
            results["pearson"].append(task_pearson)
            results["spearman"].append(task_spearman)
        break

    return results
