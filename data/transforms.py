"""Albumentations-based augmentation pipelines for training and validation."""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(phase, img_size=256):
    """Build an augmentation pipeline for the given phase.

    Args:
        phase (str): 'train' for augmented pipeline; any other value gives the
            validation/test pipeline (resize + normalise only).
        img_size (int): Target square image size in pixels.

    Returns:
        albumentations.Compose: Composed augmentation pipeline.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if phase == "train":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
            ], p=0.3),
            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5
                ),
                A.CLAHE(clip_limit=2.0, p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
            ], p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
