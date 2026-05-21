"""Single-image inference demo for EDTrans."""

import argparse
import os

import cv2
import numpy as np
import torch
import yaml

from data.transforms import get_transforms
from models import EDTrans


def run_inference(image_path, checkpoint, cfg, output_path=None):
    """Run EDTrans on a single image and optionally save the predicted mask.

    Args:
        image_path (str): Path to the input image file.
        checkpoint (str): Path to the trained model checkpoint (.pth).
        cfg (dict): Configuration dictionary loaded from config.yaml.
        output_path (str | None): If given, the predicted mask is written here.

    Returns:
        np.ndarray: Predicted binary mask (H, W), dtype uint8, values in {0, 255}.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = cfg["data"]["img_size"]

    model = EDTrans(
        encoder_name=cfg["model"]["encoder_name"],
        encoder_weights=None,
        in_channels=cfg["model"]["in_channels"],
        classes=cfg["model"]["num_classes"],
        use_edge_guided=cfg["model"]["use_edge_guided"],
        use_dual_attention=cfg["model"]["use_dual_attention"],
        use_feature_transformer=cfg["model"]["use_feature_transformer"],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    tensor = get_transforms("val", img_size)(image=img_rgb)["image"]
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(tensor)
        if isinstance(pred, tuple):
            pred = pred[0]
        prob = pred.squeeze().cpu().numpy()

    mask = (prob > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    if output_path:
        cv2.imwrite(output_path, mask)
        print(f"Prediction saved to: {output_path}")

    return mask


def main():
    parser = argparse.ArgumentParser(description="EDTrans single-image inference")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output", default="prediction.png",
                        help="Output path for the predicted mask")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_inference(args.image, args.checkpoint, cfg, args.output)


if __name__ == "__main__":
    main()
