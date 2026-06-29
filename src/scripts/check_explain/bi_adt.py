import copy
import torch
import torchvision.transforms.functional as TF
import numpy as np
from tqdm import tqdm
from skimage.segmentation import slic  # hoặc felzenszwalb


def split_to_superpixels(sup_x, saliency_map, n_segs=150, compactness=10.0):
    N, C, H, W = sup_x.shape
    device = sup_x.device
    imgs_np = sup_x.detach().cpu().numpy().transpose(0, 2, 3, 1)  # [N, H, W, C]
    sal_np = saliency_map.detach().cpu().squeeze(1).numpy()  # [N, H, W]
    print(saliency_map.shape)

    segments_list = []  # for each image
    sp_info = []

    for i in range(N):
        # create superpixels
        segments = slic(
            imgs_np[i],
            n_segments=n_segs,
            compactness=compactness,
            sigma=1.0,
            start_label=0,
            channel_axis=-1 if C == 3 else None,
            enforce_connectivity=True,
        )
        segments_list.append(segments)

        # locate superpixel info
        unique_sps = np.unique(segments)
        for spl_label in unique_sps:
            sp_mask_np = segments == spl_label
            avg_sal = float(sal_np[i][sp_mask_np].mean())

            sp_info.append(
                {
                    "image_idx": i,
                    "spl_label": spl_label,  # superpixel location label
                    "avg_sal": avg_sal,
                    "size": int(sp_mask_np.sum()),
                }
            )

    return {
        "segments_list": segments_list,
        "sp_info": sp_info,
        "shape": (N, H, W),
        "device": device,
    }


def blur_mask_sup(sup_x, segs, blur_sps, blur_sigma=5.0):
    N, C, H, W = sup_x.shape
    device = sup_x.device

    mask = torch.zeros((N, 1, H, W), dtype=torch.float32, device=device)
    for sp in blur_sps:
        img_idx = sp["image_idx"]
        segments = segs[img_idx]
        sp_mask_np = segments == sp["spl_label"]
        mask[img_idx, 0] += torch.from_numpy(sp_mask_np.astype(np.float32)).to(device)

    # Blur mask
    if blur_sigma > 0:
        ksize = int(blur_sigma * 4) | 1
        mask = TF.gaussian_blur(mask, kernel_size=(ksize, ksize), sigma=blur_sigma)
        mask = torch.clamp(mask, 0.0, 1.0)

    # Blur original image for baseline
    blurred_baseline = TF.gaussian_blur(sup_x, kernel_size=(11, 11), sigma=5.0)

    # Apply mask
    sup_x_masked = sup_x * (1 - mask) + blurred_baseline * mask
    return sup_x_masked


def adt(
    explainer,
    sup_x,
    sup_y,
    que_x,
    que_y,
    T,
    adapt_gain_base,
    saliency_map,
    scale="all",
    mode="pos",
    blur_sigma=5.0,
    n_segs=150,
    compactness=10.0,
):
    __MODES = ["pos", "neg", "random"]
    if mode not in __MODES:
        raise ValueError(f"Invalid mode: {mode}. Must be one of {__MODE}")
    __SCALES = ["all", "same"]
    if scale not in __SCALES:
        raise ValueError(f"Invalid scale: {scale}. Must be one of {__SCALES}")

    # split image to superpixels
    sps = split_to_superpixels(
        sup_x, saliency_map, n_segs=n_segs, compactness=compactness
    )
    sp_info = sps["sp_info"]

    # sort by mode (asc or desc or random of salience)
    if mode == "pos":
        positive_sps = [sp for sp in sp_info if sp["avg_sal"] > 0]
        positive_sps.sort(key=lambda x: x["avg_sal"], reverse=True)
        sp_sorted = positive_sps
    elif mode == "neg":
        negative_sps = [sp for sp in sp_info if sp["avg_sal"] < 0]
        negative_sps.sort(key=lambda x: x["avg_sal"], reverse=False)
        sp_sorted = negative_sps
    else:
        sp_sorted = copy.deepcopy(sp_info)
        np.random.shuffle(sp_sorted)

    # delete one by one superpixel with order (support set scale)
    gains = [adapt_gain_base]
    pixel_ratios = [0.0]
    if scale == "all":
        for sp_id, _ in enumerate(sp_sorted):
            sup_x_masked = blur_mask_sup(sup_x, sps["segments_list"], sp_sorted[: sp_id + 1], blur_sigma=blur_sigma)
            adapt_gain, _ = explainer.interpret(sup_x_masked, sup_y, que_x, que_y, T)
            gains.append(adapt_gain)
            pixel_ratios.append(float(sp_id + 1) / len(sp_sorted))

    # cumpute the Area Under the Curve
    auc = np.trapz(gains, pixel_ratios)
    return auc


def compute_bidirectional_faithfulness(
    explainer, test_loader, T, scale="all", n_segs=150, compactness=10.0, blur_sigma=5.0
):
    test_loader_pbar = tqdm(
        test_loader, desc="Explaining", position=0, leave=True, unit="boT"
    )
    pdas = []
    ndas = []
    combines = []

    for metabatch_id, boT in enumerate(test_loader_pbar):
        boT_pbar = tqdm(
            boT, desc=f"Batch {metabatch_id}", position=1, leave=False, unit="task"
        )
        for task_id, (support, query) in enumerate(boT_pbar):
            sup_x, sup_y = support
            que_x, que_y = query

            adapt_gain_base, saliency_map = explainer.interpret(
                sup_x, sup_y, que_x, que_y, T
            )
            auc_pos = adt(explainer, sup_x, sup_y, que_x, que_y, T,
                adapt_gain_base=adapt_gain_base,
                saliency_map=saliency_map,
                scale=scale,
                mode="pos",
                blur_sigma=blur_sigma,
                n_segs=n_segs,
                compactness=compactness,
            )
            auc_neg = adt(explainer, sup_x, sup_y, que_x, que_y, T,
                adapt_gain_base=adapt_gain_base,
                saliency_map=saliency_map,
                scale="all",
                mode="neg",
                blur_sigma=blur_sigma,
                n_segs=n_segs,
                compactness=compactness,
            )
            auc_random = adt(explainer, sup_x, sup_y, que_x, que_y, T,
                adapt_gain_base=adapt_gain_base,
                saliency_map=saliency_map,
                scale="all",
                mode="random",
                blur_sigma=blur_sigma,
                n_segs=n_segs,
                compactness=compactness,
            )

            pda = auc_pos - auc_neg
            nda = auc_neg - auc_random
            combined = auc_pos - auc_random

            pdas.append(pda)
            ndas.append(nda)
            combines.append(combined)

    return pdas, ndas, combines
