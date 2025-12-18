import os
from pathlib import Path

import numpy as np


def _build_model(device: str):
    import torch
    from torchvision.models import resnet50, ResNet50_Weights

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    return model, weights


def _build_loaders(train_dir: Path, val_dir: Path, batch_size: int, num_workers: int):
    import torch
    from torchvision import datasets
    from torchvision.models import ResNet50_Weights

    # Use the exact preprocessing expected by the pretrained weights.
    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()

    train_ds = datasets.ImageFolder(str(train_dir), transform=preprocess)
    val_ds = datasets.ImageFolder(str(val_dir), transform=preprocess)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_ds, val_ds, train_loader, val_loader


def _extract_features(model, loader, device: str):
    import torch

    all_feats = []
    all_y = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            feat = model(xb)
            all_feats.append(feat.detach().cpu())
            all_y.append(yb.detach().cpu())

    X = torch.cat(all_feats, dim=0).numpy().astype(np.float32, copy=False)
    y = torch.cat(all_y, dim=0).numpy().astype(np.int64, copy=False)
    return X, y


def main():
    # Defaults (same behavior as prior CLI defaults)
    dataset_root = Path(__file__).parent / "data_augmentation" / "splitted-dataset"
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    out_dir = Path(__file__).parent / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, weights = _build_model(device)
    train_ds, val_ds, train_loader, val_loader = _build_loaders(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=32,
        num_workers=0,
    )

    print(f"Classes ({len(train_ds.classes)}): {train_ds.classes}")
    if train_ds.classes != val_ds.classes:
        raise ValueError("Train/val class folders differ; cannot build consistent labels.")

    print("Extracting train features...")
    X_train, y_train = _extract_features(model, train_loader, device=device)
    print("Extracting val features...")
    X_val, y_val = _extract_features(model, val_loader, device=device)

    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_val.npy", X_val)
    np.save(out_dir / "y_val.npy", y_val)



    print(f"Saved: {out_dir / 'X_train.npy'} {X_train.shape}")
    print(f"Saved: {out_dir / 'X_val.npy'} {X_val.shape}")
    print(f"Saved: {out_dir / 'cnn_features_meta.json'}")


if __name__ == "__main__":

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()


