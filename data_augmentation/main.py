import os
import random
import cv2
from tqdm import tqdm
import shutil

from data_augmentation.configuration import RAW_DATASET_DIR, AUG_DATASET_DIR, TARGET_COUNT_PER_CLASS, CLASS_AUGMENTATIONS
from functions import load_images_from_folder, save_image

# ----------------------------
# (3) MAIN AUGMENTATION SCRIPT
# ----------------------------

if os.path.exists(AUG_DATASET_DIR):
    shutil.rmtree(AUG_DATASET_DIR)
os.makedirs(AUG_DATASET_DIR, exist_ok=True)

report = {}

for cls in os.listdir(RAW_DATASET_DIR):
    class_folder = os.path.join(RAW_DATASET_DIR, cls)
    images = load_images_from_folder(class_folder)
    current_count = len(images)
    needed = TARGET_COUNT_PER_CLASS - current_count

    print(f"\nProcessing class '{cls}' - current: {current_count}, target: {TARGET_COUNT_PER_CLASS}")
    cls_aug_dir = os.path.join(AUG_DATASET_DIR, cls)
    os.makedirs(cls_aug_dir, exist_ok=True)

    # Save original images first
    for img, fname in images:
        save_path = os.path.join(cls_aug_dir, fname)
        save_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), save_path)

    # Apply augmentation to reach target count
    pbar = tqdm(total=needed)
    while needed > 0:
        img, _ = random.choice(images)
        augmented = CLASS_AUGMENTATIONS[cls](image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        aug_img = augmented['image']
        save_path = os.path.join(cls_aug_dir, f"aug_{random.randint(0,1e9)}.jpg")
        save_image(aug_img, save_path)
        needed -= 1
        pbar.update(1)
    pbar.close()
    report[cls] = len(os.listdir(cls_aug_dir))
