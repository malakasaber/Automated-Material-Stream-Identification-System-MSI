from albumentations import (
    Compose, Rotate, HorizontalFlip, VerticalFlip, RandomBrightnessContrast,
    GaussNoise, RandomCrop, ShiftScaleRotate, Blur, Normalize
)

# ----------------------------
# (1) CONFIGURATION
# ----------------------------
RAW_DATASET_DIR = "C:\Users\Malak\Year4_Term1\Machine Learning\Project\Dataset\dataset"   # folder containing class subfolders
AUG_DATASET_DIR = "C:\Users\Malak\Year4_Term1\Machine Learning\Project\Dataset\dataset_augmented"
TARGET_COUNT_PER_CLASS = 500
IMG_SIZE = 224
TRAIN_RATIO = 0.8

# Define per-class augmentations
CLASS_AUGMENTATIONS = {
    "cardboard": Compose([
        Rotate(limit=15, p=0.7),
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(p=0.5),
        GaussNoise(p=0.3),
        RandomCrop(width=200, height=200, p=0.3),
        Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ]),
    "glass": Compose([
        Rotate(limit=15, p=0.5),
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(p=0.5),
        Blur(blur_limit=3, p=0.3),
        GaussNoise(p=0.2),
        Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ]),
    "metal": Compose([
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.7),
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(p=0.5),
        GaussNoise(p=0.3),
        RandomCrop(width=200, height=200, p=0.3),
        Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ]),
    "paper": Compose([
        Rotate(limit=10, p=0.5),
        HorizontalFlip(p=0.3),
        RandomBrightnessContrast(p=0.3),
        GaussNoise(p=0.1),
        Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ]),
    "plastic": Compose([
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.6),
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(p=0.5),
        GaussNoise(p=0.2),
        RandomCrop(width=200, height=200, p=0.3),
        Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ]),
    "trash": Compose([
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=25, p=0.9),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.3),
        RandomBrightnessContrast(p=0.7),
        GaussNoise(p=0.4),
        RandomCrop(width=200, height=200, p=0.5),
        Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ])
}