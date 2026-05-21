"""Training script for EDTrans on polyp segmentation datasets."""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.polyp_dataset import PolypDataset, prepare_dataset_split
from data.transforms import get_transforms
from models import EDTrans
from utils.losses import StructureLoss
from utils.metrics import calculate_metrics
from utils.visualization import plot_training_history


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, criterion, device, cfg):
    model.train()
    total_loss = 0.0
    edge_w = cfg["training"]["edge_loss_weight"]
    grad_clip = cfg["training"]["grad_clip_norm"]

    for images, masks in tqdm(loader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        outputs, edge_pred = model(images)
        outputs = torch.clamp(outputs, 0, 1)
        masks = torch.clamp(masks, 0, 1)

        loss = criterion(outputs, masks)

        if edge_pred is not None:
            edge_y = torch.abs(masks[:, :, 1:, :] - masks[:, :, :-1, :])
            edge_x = torch.abs(masks[:, :, :, 1:] - masks[:, :, :, :-1])
            edge_gt = (F.pad(edge_y, (0, 0, 0, 1)) + F.pad(edge_x, (0, 1, 0, 0)) > 0).float()
            if edge_pred.shape != edge_gt.shape:
                edge_gt = F.interpolate(edge_gt, size=edge_pred.shape[2:], mode="nearest")
            loss = loss + edge_w * F.binary_cross_entropy(edge_pred, edge_gt)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device, img_size):
    model.eval()
    val_loss = val_iou = val_dice = 0.0

    for images, masks in tqdm(loader, desc="Validating", leave=False):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        outputs = torch.clamp(outputs, 0, 1)

        val_loss += criterion(outputs, masks).item() * images.size(0)
        iou, dice, *_ = calculate_metrics(outputs, masks, img_size=img_size)
        val_iou += iou * images.size(0)
        val_dice += dice * images.size(0)

    n = len(loader.dataset)
    return val_loss / n, val_iou / n, val_dice / n


def main():
    parser = argparse.ArgumentParser(description="Train EDTrans")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dataset", default=None,
                        help="Override config dataset (Kvasir-SEG | CVC-ClinicDB)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = args.dataset or cfg["data"]["dataset"]
    img_size = cfg["data"]["img_size"]

    os.makedirs(cfg["paths"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)

    print(f"Dataset : {dataset_name}")
    print(f"Device  : {device}")

    train_imgs, train_msks, val_imgs, val_msks, _, _ = prepare_dataset_split(
        dataset_name, root_dir=cfg["data"]["data_dir"], seed=cfg["seed"]
    )
    print(f"Train: {len(train_imgs)} | Val: {len(val_imgs)}")

    train_loader = DataLoader(
        PolypDataset(train_imgs, train_msks, get_transforms("train", img_size)),
        batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        PolypDataset(val_imgs, val_msks, get_transforms("val", img_size)),
        batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=0,
    )

    model = EDTrans(
        encoder_name=cfg["model"]["encoder_name"],
        encoder_weights=cfg["model"]["encoder_weights"],
        in_channels=cfg["model"]["in_channels"],
        classes=cfg["model"]["num_classes"],
        use_edge_guided=cfg["model"]["use_edge_guided"],
        use_dual_attention=cfg["model"]["use_dual_attention"],
        use_feature_transformer=cfg["model"]["use_feature_transformer"],
    ).to(device)

    criterion = StructureLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)

    best_dice = 0.0
    patience_counter = 0
    patience = cfg["training"]["early_stopping_patience"]
    checkpoint_path = os.path.join(cfg["paths"]["checkpoint_dir"], "best_model.pth")
    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}

    for epoch in range(cfg["training"]["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, cfg)
        val_loss, val_iou, val_dice = validate(model, val_loader, criterion, device, img_size)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        print(
            f"Epoch [{epoch+1:03d}/{cfg['training']['epochs']}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Checkpoint saved  (Dice: {best_dice:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    plot_training_history(
        history,
        os.path.join(cfg["paths"]["results_dir"], "training_history.png"),
    )
    print(f"\nTraining complete. Best Val Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
