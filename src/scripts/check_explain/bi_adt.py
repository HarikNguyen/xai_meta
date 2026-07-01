import copy
import torch
import torchvision.transforms.functional as TF
import numpy as np
from tqdm import tqdm
from skimage.segmentation import slic

def get_per_image_rank_tensor(sup_x, saliency_map, mode, n_segs=150, compactness=10.0):
    """
    Chạy SLIC trên CPU một lần duy nhất.
    Trả về một Tensor trên GPU [N, 1, H, W] chứa "Thứ hạng" (Rank từ 0.0 -> 1.0) của từng pixel.
    - Pixel thuộc Superpixel quan trọng nhất sẽ có Rank gần 0.0.
    - Pixel thuộc Superpixel ít quan trọng nhất sẽ có Rank gần 1.0.
    """
    N, C, H, W = sup_x.shape
    device = sup_x.device
    imgs_np = sup_x.detach().cpu().numpy().transpose(0, 2, 3, 1)
    sal_np = saliency_map.detach().cpu().squeeze(1).numpy()

    # Ma trận chứa hạng của từng pixel
    rank_maps = np.zeros((N, H, W), dtype=np.float32)

    for i in range(N):
        # 1. Chia Superpixel
        segs = slic(
            imgs_np[i], n_segments=n_segs, compactness=compactness,
            sigma=1.0, start_label=0, channel_axis=-1 if C == 3 else None,
            enforce_connectivity=True
        )
        
        # 2. Tính Saliency trung bình cho từng Superpixel
        unique_sps = np.unique(segs)
        sp_list = []
        for sp in unique_sps:
            avg_sal = sal_np[i][segs == sp].mean()
            sp_list.append((sp, avg_sal))

        # 3. Sắp xếp ĐỘC LẬP TỪNG ẢNH (Local Ranking)
        if mode == "pos":
            sp_list.sort(key=lambda x: x[1], reverse=True)   # Đỏ nhất xóa trước
        elif mode == "neg":
            sp_list.sort(key=lambda x: x[1], reverse=False)  # Xanh nhất xóa trước
        else:
            np.random.shuffle(sp_list)                       # Xóa bừa

        # 4. Gán Percentile Rank (0.0 -> 1.0) cho từng vùng
        num_sps = len(sp_list)
        for rank_idx, (sp, _) in enumerate(sp_list):
            percentile_rank = (rank_idx + 1) / num_sps
            rank_maps[i][segs == sp] = percentile_rank

    # Chuyển lên GPU để siêu tăng tốc cho các bước sau
    return torch.from_numpy(rank_maps).unsqueeze(1).to(device)


def apply_mask_fast(sup_x, blurred_baseline, rank_tensor, ratio, blur_sigma=5.0):
    """
    Tạo Mask và áp dụng trực tiếp trên GPU bằng toán tử Vector hóa.
    (Không cần vòng lặp for nào cả).
    """
    # Tạo mask đồng thời cho TẤT CẢ các ảnh: pixel nào có hạng <= ratio thì bị xóa (bằng 1)
    mask = (rank_tensor <= ratio).float()

    # Làm mờ viền Mask để tránh OOD artifacts
    if blur_sigma > 0:
        ksize = int(blur_sigma * 4) | 1
        mask = TF.gaussian_blur(mask, kernel_size=[ksize, ksize], sigma=[blur_sigma, blur_sigma])
        mask = torch.clamp(mask, 0.0, 1.0)

    # Pha trộn (Lerp) ảnh gốc và ảnh nhiễu nền
    return sup_x * (1 - mask) + blurred_baseline * mask


def adt(
    explainer, sup_x, sup_y, que_x, que_y, T, adapt_gain_base, saliency_map,
    mode="pos", blur_sigma=5.0, n_segs=150, compactness=10.0, num_steps=10
):
    __MODES = ["pos", "neg", "random"]
    if mode not in __MODES:
        raise ValueError(f"Invalid mode: {mode}.")

    # 1. Lấy ma trận xếp hạng (Rank Tensor) đã được tính toán độc lập cho từng ảnh
    rank_tensor = get_per_image_rank_tensor(sup_x, saliency_map, mode, n_segs, compactness)
    
    # 2. Tạo Baseline Blur 1 lần duy nhất trên GPU
    blurred_baseline = TF.gaussian_blur(sup_x, kernel_size=[11, 11], sigma=[5.0, 5.0])

    gains = [adapt_gain_base]
    pixel_ratios = [0.0]

    # 3. Vòng lặp Xóa % (Siêu nhanh vì Mask được tạo = Tensor Thresholding)
    for step in range(1, num_steps + 1):
        ratio = step / num_steps
        
        # Xóa đồng thời ratio% diện tích trên TẤT CẢ các ảnh
        sup_x_masked = apply_mask_fast(sup_x, blurred_baseline, rank_tensor, ratio, blur_sigma)
        
        # Đánh giá lại mô hình MAML
        adapt_gain, _ = explainer.interpret(sup_x_masked, sup_y, que_x, que_y, T)
        
        gains.append(adapt_gain)
        pixel_ratios.append(ratio)

    auc = np.trapezoid(gains, pixel_ratios)
    return auc


def compute_bidirectional_faithfulness(
    explainer, test_loader, T, n_segs=150, compactness=10.0, blur_sigma=5.0, num_steps=10
):
    test_loader_pbar = tqdm(test_loader, desc="BiDAT", position=0, leave=True, unit="boT")
    pdas, ndas, combines = [], [], []

    for metabatch_id, boT in enumerate(test_loader_pbar):
        boT_pbar = tqdm(boT, desc=f"Batch {metabatch_id}", position=1, leave=False, unit="task")
        
        for task_id, (support, query) in enumerate(boT_pbar):
            sup_x, sup_y = support
            que_x, que_y = query

            # Tính Gain gốc và Saliency
            adapt_gain_base, saliency_map = explainer.interpret(sup_x, sup_y, que_x, que_y, T)
            
            kwargs = {
                "explainer": explainer, "sup_x": sup_x, "sup_y": sup_y, 
                "que_x": que_x, "que_y": que_y, "T": T, 
                "adapt_gain_base": adapt_gain_base, "saliency_map": saliency_map,
                "blur_sigma": blur_sigma, "n_segs": n_segs, 
                "compactness": compactness, "num_steps": num_steps
            }

            auc_pos = adt(mode="pos", **kwargs)
            auc_neg = adt(mode="neg", **kwargs)
            auc_random = adt(mode="random", **kwargs)

            pda = auc_random - auc_pos
            nda = auc_neg - auc_random
            combined = pda + nda
            
            pdas.append(pda)
            ndas.append(nda)
            combines.append(combined)

    return pdas, ndas, combines
