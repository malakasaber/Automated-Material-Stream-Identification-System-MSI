import os

from train_val_split import train_dir, val_dir

# ----------------------------
# (5) FINAL REPORT
# ----------------------------
print("\n=== Balanced Dataset Counts ===")
for cls in sorted(os.listdir(train_dir)):
    train_count = len(os.listdir(os.path.join(train_dir, cls)))
    val_count = len(os.listdir(os.path.join(val_dir, cls)))
    print(f"{cls}: train={train_count}, val={val_count}, total={train_count+val_count}")