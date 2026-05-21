"""Dataset classes and data-loading utilities for polyp segmentation benchmarks."""

import glob
import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class PolypDataset(Dataset):
    """PyTorch Dataset for polyp segmentation images and binary masks.

    Args:
        images (list[str]): Absolute paths to input images.
        masks (list[str] | None): Absolute paths to ground-truth mask images.
            Pass None for inference-only mode (no mask returned).
        transforms (albumentations.Compose | None): Augmentation pipeline.
    """

    def __init__(self, images, masks=None, transforms=None):
        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.images[idx]), cv2.COLOR_BGR2RGB)

        if self.masks is not None:
            mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.float32)

            if self.transforms:
                aug = self.transforms(image=img, mask=mask)
                img = aug["image"]
                mask = aug["mask"].unsqueeze(0)

            return img, mask

        if self.transforms:
            img = self.transforms(image=img)["image"]
        return img


def get_dataset_files(dataset_name, root_dir="./dataset"):
    """Return matched image and mask file paths for a named dataset.

    Supported dataset layouts:

    - **Kvasir-SEG**: ``<root_dir>/Kvasir-SEG/images/*.jpg`` and
      ``<root_dir>/Kvasir-SEG/masks/*.jpg``.
    - **CVC-ClinicDB**: ``<root_dir>/CVC-ClinicDB/Original/`` and
      ``<root_dir>/CVC-ClinicDB/Ground Truth/`` (*.tif or *.png).
    - **ETIS-LaribPoly**: ``<root_dir>/ETIS-LaribPoly/images/`` and
      ``<root_dir>/ETIS-LaribPoly/masks/`` (*.tif, *.jpg, or *.png).

    Args:
        dataset_name (str): One of 'Kvasir-SEG', 'CVC-ClinicDB', 'ETIS-LaribPoly'.
        root_dir (str): Root directory containing dataset sub-folders.

    Returns:
        tuple[list[str], list[str]]: Matched lists of image paths and mask paths.

    Raises:
        ValueError: If dataset_name is not recognised.
    """
    dataset_path = os.path.join(root_dir, dataset_name)

    if dataset_name == "Kvasir-SEG":
        if not os.path.exists(dataset_path):
            dataset_path = root_dir
        images = sorted(glob.glob(os.path.join(dataset_path, "images", "*.jpg")))
        masks = sorted(glob.glob(os.path.join(dataset_path, "masks", "*.jpg")))

    elif dataset_name == "CVC-ClinicDB":
        img_dir = os.path.join(dataset_path, "Original")
        mask_dir = os.path.join(dataset_path, "Ground Truth")
        if not os.path.exists(img_dir):
            print(f"Warning: {img_dir} not found.")
            return [], []
        images = sorted(glob.glob(os.path.join(img_dir, "*.tif"))) + \
                 sorted(glob.glob(os.path.join(img_dir, "*.png")))
        masks = sorted(glob.glob(os.path.join(mask_dir, "*.tif"))) + \
                sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    elif dataset_name == "ETIS-LaribPoly":
        img_dir = os.path.join(dataset_path, "images")
        mask_dir = os.path.join(dataset_path, "masks")
        exts = ("*.tif", "*.jpg", "*.png")
        images = sum((sorted(glob.glob(os.path.join(img_dir, e))) for e in exts), [])
        masks = sum((sorted(glob.glob(os.path.join(mask_dir, e))) for e in exts), [])

    else:
        raise ValueError(f"Unknown dataset: '{dataset_name}'. "
                         "Supported: 'Kvasir-SEG', 'CVC-ClinicDB', 'ETIS-LaribPoly'.")

    if len(images) != len(masks):
        print(f"Warning: {len(images)} images vs {len(masks)} masks — matching by filename.")
        img_map = {os.path.splitext(os.path.basename(p))[0]: p for p in images}
        mask_map = {os.path.splitext(os.path.basename(p))[0]: p for p in masks}
        common = sorted(set(img_map) & set(mask_map))
        images = [img_map[k] for k in common]
        masks = [mask_map[k] for k in common]
        print(f"Matched {len(images)} image-mask pairs.")

    return images, masks


def prepare_dataset_split(dataset_name, root_dir="./dataset", seed=42):
    """Load a dataset and split it 70 / 15 / 15 into train, val, and test subsets.

    Args:
        dataset_name (str): Dataset identifier (see get_dataset_files).
        root_dir (str): Root directory for datasets.
        seed (int): Random seed for reproducible splits.

    Returns:
        tuple[list, list, list, list, list, list]:
            train_images, train_masks, val_images, val_masks, test_images, test_masks.
    """
    images, masks = get_dataset_files(dataset_name, root_dir)

    if not images:
        return [], [], [], [], [], []

    train_imgs, temp_imgs, train_msks, temp_msks = train_test_split(
        images, masks, test_size=0.3, random_state=seed
    )
    val_imgs, test_imgs, val_msks, test_msks = train_test_split(
        temp_imgs, temp_msks, test_size=0.5, random_state=seed
    )
    return train_imgs, train_msks, val_imgs, val_msks, test_imgs, test_msks
