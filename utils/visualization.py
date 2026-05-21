"""Visualization utilities for segmentation predictions and training diagnostics."""

import matplotlib.pyplot as plt
import numpy as np


def save_prediction_grid(images, masks, preds, output_path, num_samples=5):
    """Save a side-by-side grid of images, ground-truth masks, and predictions.

    Args:
        images (list[np.ndarray]): Original RGB images (H, W, 3).
        masks (list[np.ndarray]): Ground-truth binary masks (H, W), values in {0, 1}.
        preds (list[np.ndarray]): Predicted probability maps (H, W), values in [0, 1].
        output_path (str): File path for the saved figure.
        num_samples (int): Number of samples to include in the grid.
    """
    n = min(num_samples, len(images))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    titles = ["Image", "Ground Truth", "Prediction"]
    for i in range(n):
        for j, (data, cmap) in enumerate(zip(
            [images[i], masks[i], preds[i]],
            [None, "gray", "gray"],
        )):
            axes[i][j].imshow(data, cmap=cmap, vmin=0, vmax=1 if j > 0 else None)
            axes[i][j].set_title(titles[j])
            axes[i][j].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_history(history, output_path):
    """Plot and save training loss and validation metric curves.

    Args:
        history (dict): Keys 'train_loss', 'val_loss', 'val_dice', 'val_iou',
            each mapping to a list of per-epoch scalar values.
        output_path (str): File path for the saved figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["val_dice"], label="Val Dice")
    axes[1].plot(history["val_iou"], label="Val IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation Metrics")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def overlay_mask(image, mask, alpha=0.4, color=(0, 255, 0)):
    """Overlay a binary mask on an RGB image.

    Args:
        image (np.ndarray): RGB image (H, W, 3), dtype uint8.
        mask (np.ndarray): Binary mask (H, W), values in {0, 1}.
        alpha (float): Mask transparency (0 = transparent, 1 = opaque).
        color (tuple[int, int, int]): RGB colour for the mask overlay.

    Returns:
        np.ndarray: Blended image (H, W, 3), dtype uint8.
    """
    overlay = image.copy().astype(np.float32)
    mask_bool = mask.astype(bool)
    for c, val in enumerate(color):
        overlay[mask_bool, c] = (
            (1 - alpha) * overlay[mask_bool, c] + alpha * val
        )
    return np.clip(overlay, 0, 255).astype(np.uint8)
