import os
import random
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm

from configuration import (
    RAW_DATASET_DIR,
    SPLIT_DATASET_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TRAIN_RATIO,
    TARGET_COUNT_PER_CLASS,
    CLASS_AUGMENTATIONS,
    SEED,
    IMG_SIZE,
)

IMG_EXTS = (".jpg", ".jpeg", ".png")

def _list_images(folder: Path):
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

def _save_rgb_image(rgb, out_path: Path):
    # Albumentations returns uint8 images when no Normalize is used.
    if rgb.dtype != "uint8":
        rgb = rgb.clip(0, 255).astype("uint8")
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), bgr)


def _augment_class_to_target(cls: str, cls_train_dir: Path, target_count: int, rng: random.Random):
    if cls not in CLASS_AUGMENTATIONS:
        raise KeyError(f"Missing augmentation config for class '{cls}'")

    # Only sample from original (non-augmented) images to avoid compounding artifacts.
    originals = [p for p in _list_images(cls_train_dir) if not p.name.startswith("aug_")]
    current_count = len(_list_images(cls_train_dir))
    needed = max(0, target_count - current_count)

    if needed == 0:
        return current_count
    if not originals:
        return current_count

    pbar = tqdm(total=needed, desc=f"Augmenting {cls}", leave=False)
    while needed > 0:
        src = rng.choice(originals)
        img_bgr = cv2.imread(str(src))
        if img_bgr is None:
            continue
        img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        aug = CLASS_AUGMENTATIONS[cls](image=img_rgb)["image"]
        out_path = cls_train_dir / f"aug_{rng.randrange(0, 1_000_000_000)}.jpg"
        _save_rgb_image(aug, out_path)
        needed -= 1
        pbar.update(1)
    pbar.close()

    return len(_list_images(cls_train_dir))


def main():
    rng = random.Random(SEED)

    raw_root = Path(RAW_DATASET_DIR)
    split_root = Path(SPLIT_DATASET_DIR)
    train_root = Path(TRAIN_DIR)
    val_root = Path(VAL_DIR)

    if not raw_root.exists():
        raise FileNotFoundError(f"RAW_DATASET_DIR not found: {raw_root}")

    # Rebuild split output folders (as requested)
    if split_root.exists():
        shutil.rmtree(split_root)
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    classes = sorted([p.name for p in raw_root.iterdir() if p.is_dir()])
    if not classes:
        raise ValueError(f"No class folders found under: {raw_root}")

    print(f"Raw dataset root: {raw_root}")
    print(f"Split output root: {split_root}")
    print(f"Classes ({len(classes)}): {classes}")

    print("\n=== Splitting raw data into train/val ===")
    for cls in classes:
        src_cls_dir = raw_root / cls
        imgs = _list_images(src_cls_dir)
        rng.shuffle(imgs)

        split_idx = int(len(imgs) * TRAIN_RATIO)
        train_imgs = imgs[:split_idx]
        val_imgs = imgs[split_idx:]

        dst_train_cls = train_root / cls
        dst_val_cls = val_root / cls
        dst_train_cls.mkdir(parents=True, exist_ok=True)
        dst_val_cls.mkdir(parents=True, exist_ok=True)

        for p in train_imgs:
            shutil.copy2(p, dst_train_cls / p.name)
        for p in val_imgs:
            shutil.copy2(p, dst_val_cls / p.name)

        print(f"{cls}: Train={len(train_imgs)}, Val={len(val_imgs)}")

    print("\n=== Augmenting training data only ===")
    report = {}
    for cls in classes:
        dst_train_cls = train_root / cls
        report[cls] = _augment_class_to_target(
            cls=cls,
            cls_train_dir=dst_train_cls,
            target_count=TARGET_COUNT_PER_CLASS,
            rng=rng,
        )

    print("\nAugmented counts (train):")
    for cls in classes:
        print(f"  {cls}: {report[cls]}")

    print("\nDone.")


if __name__ == "__main__":
    main()