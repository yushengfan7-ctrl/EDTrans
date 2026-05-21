"""Evaluation metrics: IoU, Dice, Precision, Recall, HD95, and model complexity."""

import numpy as np
import torch


def compute_hd95(pred, target, img_size=256):
    """Compute the mean 95th-percentile Hausdorff Distance (HD95) for a batch.

    Args:
        pred (torch.Tensor): Binary prediction tensor (B, 1, H, W).
        target (torch.Tensor): Binary ground-truth tensor (B, 1, H, W).
        img_size (int): Image size used for distance normalisation.

    Returns:
        float: Mean HD95 over valid (non-empty) samples in the batch.
    """
    from scipy.ndimage import distance_transform_edt

    norm_factor = np.sqrt(2) * img_size / 25.0
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    hd95_sum, valid = 0.0, 0

    for i in range(pred_np.shape[0]):
        p = pred_np[i, 0]
        t = target_np[i, 0]
        if p.sum() == 0 or t.sum() == 0:
            continue

        d_pt = distance_transform_edt(~t.astype(bool))[p.astype(bool)]
        d_tp = distance_transform_edt(~p.astype(bool))[t.astype(bool)]

        if len(d_pt) == 0 or len(d_tp) == 0:
            continue

        hd95_sum += max(np.percentile(d_pt, 95), np.percentile(d_tp, 95)) / norm_factor
        valid += 1

    return hd95_sum / valid if valid > 0 else 0.0


def calculate_metrics(pred, target, threshold=0.5, img_size=256):
    """Compute segmentation metrics for a batch of predictions.

    Args:
        pred (torch.Tensor): Probability map (B, 1, H, W), values in [0, 1].
        target (torch.Tensor): Binary ground-truth mask (B, 1, H, W).
        threshold (float): Binarisation threshold.
        img_size (int): Image size for HD95 normalisation.

    Returns:
        tuple[float, float, float, float, float]:
            IoU, Dice, Precision, Recall, HD95.
    """
    pred = torch.clamp(pred, 0, 1)
    target = torch.clamp(target, 0, 1)
    pred_bin = (pred > threshold).float()

    tp = (pred_bin * target).sum()
    fp = pred_bin.sum() - tp
    fn = target.sum() - tp

    iou = (tp + 1e-7) / (tp + fp + fn + 1e-7)
    dice = (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)
    precision = (tp + 1e-7) / (tp + fp + 1e-7)
    recall = (tp + 1e-7) / (tp + fn + 1e-7)

    try:
        hd95 = compute_hd95(pred_bin, target, img_size)
    except Exception:
        hd95 = 0.0

    return iou.item(), dice.item(), precision.item(), recall.item(), hd95


def calculate_complexity(model, input_size=(1, 3, 256, 256), device="cuda"):
    """Estimate model parameters and FLOPs using thop.

    Args:
        model (nn.Module): Model to profile.
        input_size (tuple): Input tensor shape (B, C, H, W).
        device (str): Device for the dummy forward pass.

    Returns:
        tuple[float, float]: Raw parameter count and FLOPs from thop.
    """
    from thop import profile

    model = model.to(device).eval()
    dummy = torch.randn(input_size).to(device)
    try:
        flops, params = profile(model, inputs=(dummy,), verbose=False)
    except Exception:
        flops, params = 0, 0
    return params, flops
