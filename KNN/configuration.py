from pathlib import Path

from albumentations import (
    Compose,
    Rotate,
    HorizontalFlip,
    VerticalFlip,
    RandomBrightnessContrast,
    GaussNoise,
    RandomCrop,
    ShiftScaleRotate,
    Blur,
)


SEED = 42

# Dataset routing (relative to this file)
KNN_DIR = Path(__file__).parent
RAW_DATASET_DIR = KNN_DIR / "dataset"

SPLIT_DATASET_DIR = KNN_DIR / "data_augmentation" / "splitted-dataset"
TRAIN_DIR = SPLIT_DATASET_DIR / "train"
VAL_DIR = SPLIT_DATASET_DIR / "val"

# Augmentation / split settings
TARGET_COUNT_PER_CLASS = 500  # target count for TRAIN only
IMG_SIZE = 224
TRAIN_RATIO = 0.7  # train/val only (no test)

# Define per-class augmentations (NO Normalize; matches KNN/test2.ipynb)
CLASS_AUGMENTATIONS = {
    "cardboard": Compose(
        [
            Rotate(limit=15, p=0.2),
            HorizontalFlip(p=0.2),
            RandomBrightnessContrast(p=0.2),
            GaussNoise(p=0.2),
            RandomCrop(width=200, height=200, p=0.1),
        ]
    ),
    "glass": Compose(
        [
            Rotate(limit=15, p=0.1),
            HorizontalFlip(p=0.1),
            RandomBrightnessContrast(p=0.1),
            Blur(blur_limit=3, p=0.1),
            GaussNoise(p=0.1),
        ]
    ),
    "metal": Compose(
        [
            ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.3),
            HorizontalFlip(p=0.3),
            RandomBrightnessContrast(p=0.3),
            GaussNoise(p=0.3),
            RandomCrop(width=200, height=200, p=0.3),
        ]
    ),
    "paper": Compose(
        [
            Rotate(limit=10, p=0.2),
            HorizontalFlip(p=0.2),
            RandomBrightnessContrast(p=0.2),
            GaussNoise(p=0.2),
        ]
    ),
    "plastic": Compose(
        [
            ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=15, p=0.2),
            HorizontalFlip(p=0.2),
            RandomBrightnessContrast(p=0.2),
            GaussNoise(p=0.2),
            RandomCrop(width=200, height=200, p=0.2),
        ]
    ),
    "trash": Compose(
        [
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=25, p=0.6),
            HorizontalFlip(p=0.6),
            VerticalFlip(p=0.6),
            RandomBrightnessContrast(p=0.6),
            GaussNoise(p=0.6),
            RandomCrop(width=200, height=200, p=0.6),
        ]
    ),
}