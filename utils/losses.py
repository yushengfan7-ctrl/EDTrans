"""Loss functions for polyp segmentation training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StructureLoss(nn.Module):
    """Weighted BCE + Weighted IoU loss for boundary-sensitive segmentation.

    Applies spatially-adaptive weighting that up-weights pixels near object
    boundaries, following the formulation used in PraNet and related work.
    """

    def forward(self, pred, mask):
        """
        Args:
            pred (torch.Tensor): Predicted probability map (B, 1, H, W), values in [0, 1].
            mask (torch.Tensor): Binary ground-truth mask (B, 1, H, W), values in {0, 1}.

        Returns:
            torch.Tensor: Scalar combined loss value.
        """
        weight = 1 + 5 * torch.abs(
            F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
        )
        wbce = (F.binary_cross_entropy(pred, mask, reduction="none") * weight).mean()

        inter = (pred * mask).sum(dim=(2, 3))
        union = (pred + mask).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        return wbce + wiou.mean()
