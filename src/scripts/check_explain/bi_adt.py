import torch
import torchvision.transforms.functional as TF
import numpy as np
from tqdm import tqdm
from skimage.segmentation import slic

def get_per_image_rank_tensor(sup_x, saliency_map, mode, n_segs=150, compactness=10.0):
    """
    Run SLIC on the CPU exactly once.
    Return a GPU Tensor [N, 1, H, W] holding the "Rank" (0.0 -> 1.0) of each pixel.
    - Pixels belonging to the most important superpixel get a rank close to 0.0.
    - Pixels belonging to the least important superpixel get a rank close to 1.0.
    """
    N, C, H, W = sup_x.shape
    device = sup_x.device
    imgs_np = sup_x.detach().cpu().numpy().transpose(0, 2, 3, 1)
    sal_np = saliency_map.detach().cpu().squeeze(1).numpy()

    # Matrix holding the rank of each pixel
    rank_maps = np.zeros((N, H, W), dtype=np.float32)

    for i in range(N):
        # 1. Split into superpixels
        segs = slic(
            imgs_np[i], n_segments=n_segs, compactness=compactness,
            sigma=1.0, start_label=0, channel_axis=-1 if C == 3 else None,
            enforce_connectivity=True
        )

        # 2. Compute the average saliency for each superpixel
        unique_sps = np.unique(segs)
        sp_list = []
        for sp in unique_sps:
            avg_sal = sal_np[i][segs == sp].mean()
            sp_list.append((sp, avg_sal))

        # 3. Sort INDEPENDENTLY PER IMAGE (local ranking)
        if mode == "pos":
            sp_list.sort(key=lambda x: x[1], reverse=True)   # remove the most positive (red) first
        elif mode == "neg":
            sp_list.sort(key=lambda x: x[1], reverse=False)  # remove the most negative (blue) first
        else:
            np.random.shuffle(sp_list)                       # remove in random order

        # 4. Assign a percentile rank (0.0 -> 1.0) to each region
        num_sps = len(sp_list)
        for rank_idx, (sp, _) in enumerate(sp_list):
            percentile_rank = (rank_idx + 1) / num_sps
            rank_maps[i][segs == sp] = percentile_rank

    # Move to GPU to greatly speed up the following steps
    return torch.from_numpy(rank_maps).unsqueeze(1).to(device)


def apply_mask_fast(sup_x, blurred_baseline, rank_tensor, ratio, blur_sigma=5.0):
    """
    Build the mask and apply it directly on the GPU using vectorized ops.
    (No for-loop needed at all.)
    """
    # Build the mask for ALL images at once: pixels with rank <= ratio are masked out (set to 1)
    mask = (rank_tensor <= ratio).float()

    # Blur the mask edges to avoid OOD artifacts
    if blur_sigma > 0:
        ksize = int(blur_sigma * 4) | 1
        mask = TF.gaussian_blur(mask, kernel_size=[ksize, ksize], sigma=[blur_sigma, blur_sigma])
        mask = torch.clamp(mask, 0.0, 1.0)

    # Blend (lerp) the original image with the blurred baseline
    return sup_x * (1 - mask) + blurred_baseline * mask


def adt(
    explainer, sup_x, sup_y, que_x, que_y, T, adapt_gain_base, saliency_map,
    mode="pos", blur_sigma=5.0, n_segs=150, compactness=10.0, num_steps=10
):
    __MODES = ["pos", "neg", "random"]
    if mode not in __MODES:
        raise ValueError(f"Invalid mode: {mode}.")

    # 1. Get the rank tensor, computed independently for each image
    rank_tensor = get_per_image_rank_tensor(sup_x, saliency_map, mode, n_segs, compactness)

    # 2. Build the blurred baseline once on the GPU
    blurred_baseline = TF.gaussian_blur(sup_x, kernel_size=[11, 11], sigma=[5.0, 5.0])

    gains = [adapt_gain_base]
    pixel_ratios = [0.0]

    # 3. Progressive removal loop (fast, since the mask is just tensor thresholding)
    for step in range(1, num_steps + 1):
        ratio = step / num_steps

        # Remove ratio% of the area simultaneously across ALL images
        sup_x_masked = apply_mask_fast(sup_x, blurred_baseline, rank_tensor, ratio, blur_sigma)

        # Re-evaluate the MAML model
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
            sup_x, sup_y, _= support
            que_x, que_y, _ = query

            # Compute the base gain and saliency map
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
