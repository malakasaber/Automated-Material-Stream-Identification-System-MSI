import os
import random
import shutil

from configuration import AUG_DATASET_DIR, TRAIN_RATIO

# ----------------------------
# (4) TRAIN/VALIDATION SPLIT
# ----------------------------
train_dir = os.path.join(AUG_DATASET_DIR, "train")
val_dir = os.path.join(AUG_DATASET_DIR, "val")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

for cls in os.listdir(AUG_DATASET_DIR):
    cls_path = os.path.join(AUG_DATASET_DIR, cls)
    if not os.path.isdir(cls_path):
        continue
    images = os.listdir(cls_path)
    random.shuffle(images)
    split_idx = int(len(images) * TRAIN_RATIO)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    cls_train_dir = os.path.join(train_dir, cls)
    cls_val_dir = os.path.join(val_dir, cls)
    os.makedirs(cls_train_dir, exist_ok=True)
    os.makedirs(cls_val_dir, exist_ok=True)

    for img in train_imgs:
        shutil.move(os.path.join(cls_path, img), os.path.join(cls_train_dir, img))
    for img in val_imgs:
        shutil.move(os.path.join(cls_path, img), os.path.join(cls_val_dir, img))
    shutil.rmtree(cls_path)  # remove original class folder