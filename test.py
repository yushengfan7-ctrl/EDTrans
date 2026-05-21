"""Evaluation script: runs a trained EDTrans checkpoint on the held-out test split."""

import argparse
import os

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.polyp_dataset import PolypDataset, prepare_dataset_split
from data.transforms import get_transforms
from models import EDTrans
from utils.losses import StructureLoss
from utils.metrics import calculate_metrics, calculate_complexity


@torch.no_grad()
def evaluate(model, loader, criterion, device, img_size):
    """Run full evaluation over a DataLoader.

    Returns:
        dict: Per-metric averages keyed by metric name.
    """
    model.eval()
    totals = dict(loss=0.0, iou=0.0, dice=0.0, precision=0.0, recall=0.0, hd95=0.0)

    for images, masks in tqdm(loader, desc="Evaluating"):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        outputs = torch.clamp(outputs, 0, 1)

        totals["loss"] += criterion(outputs, masks).item() * images.size(0)
        iou, dice, prec, rec, hd95 = calculate_metrics(outputs, masks, img_size=img_size)
        n = images.size(0)
        totals["iou"] += iou * n
        totals["dice"] += dice * n
        totals["precision"] += prec * n
        totals["recall"] += rec * n
        totals["hd95"] += hd95 * n

    num = len(loader.dataset)
    return {k: v / num for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser(description="Evaluate EDTrans")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint file")
    parser.add_argument("--dataset", default=None,
                        help="Override config dataset (Kvasir-SEG | CVC-ClinicDB)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = args.dataset or cfg["data"]["dataset"]
    img_size = cfg["data"]["img_size"]

    _, _, _, _, test_imgs, test_msks = prepare_dataset_split(
        dataset_name, root_dir=cfg["data"]["data_dir"], seed=cfg["seed"]
    )
    test_loader = DataLoader(
        PolypDataset(test_imgs, test_msks, get_transforms("val", img_size)),
        batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=0,
    )

    model = EDTrans(
        encoder_name=cfg["model"]["encoder_name"],
        encoder_weights=None,
        in_channels=cfg["model"]["in_channels"],
        classes=cfg["model"]["num_classes"],
        use_edge_guided=cfg["model"]["use_edge_guided"],
        use_dual_attention=cfg["model"]["use_dual_attention"],
        use_feature_transformer=cfg["model"]["use_feature_transformer"],
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    metrics = evaluate(model, test_loader, StructureLoss(), device, img_size)

    print(f"\n=== Test Results on {dataset_name} ===")
    for k, v in metrics.items():
        print(f"  {k.capitalize():12s}: {v:.4f}")

    params, flops = calculate_complexity(model, device=device)
    print(f"  Params:       {params / 1e6:.2f} M")
    print(f"  FLOPs:        {flops / 1e9:.2f} G")

    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    out_path = os.path.join(cfg["paths"]["results_dir"], f"result_{dataset_name}.csv")
    pd.DataFrame([{"dataset": dataset_name, **metrics}]).to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
