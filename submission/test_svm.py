# test.py
import torch
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import numpy as np
import joblib
from pathlib import Path
import os

UNKNOWN_CLASS_ID = 6
UNKNOWN_PROBA_THRESHOLD = 0.50

CLASS_IDS = {
    "cardboard": 0,
    "glass": 1,
    "metal": 2,
    "paper": 3,
    "plastic": 4,
    "trash": 5,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _build_model(device: str):
    """Build ResNet-18 feature extractor for SVM (512-dim features)"""
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()  # remove final classification layer
    model.eval()
    model.to(device)
    return model, weights

def _extract_features_single(model, img_paths, preprocess, device: str):
    """Extract features for a list of image paths"""
    feats = []
    with torch.no_grad():
        for path in img_paths:
            img = Image.open(path).convert("RGB")
            tensor = preprocess(img).unsqueeze(0).to(device)
            feat = model(tensor)
            feats.append(feat.cpu().numpy())
    return np.vstack(feats)

def predict(dataFilePath, bestModelPath):
    """Predict function for SVM model"""
    data_path = Path(dataFilePath)
    img_paths = sorted([p for p in data_path.iterdir() if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])
    if not img_paths:
        raise ValueError(f"No images found in {dataFilePath}")

    # Load trained SVM pipeline (scaler + SVM)
    svm_pipeline = joblib.load(bestModelPath)

    # Feature extractor (ResNet-18)
    model, weights = _build_model(DEVICE)
    preprocess = weights.transforms()

    # Extract features
    X_test = _extract_features_single(model, img_paths, preprocess, DEVICE)

    if hasattr(svm_pipeline, "predict_proba"):
        proba = svm_pipeline.predict_proba(X_test)
        y_pred = proba.argmax(axis=1)
        conf = proba.max(axis=1)
        # Unknown-class rejection
        y_pred[conf < UNKNOWN_PROBA_THRESHOLD] = UNKNOWN_CLASS_ID
    else:
        y_pred = svm_pipeline.predict(X_test)
        conf = np.ones_like(y_pred)
        y_pred = np.where(conf < UNKNOWN_PROBA_THRESHOLD, UNKNOWN_CLASS_ID, y_pred)

    return y_pred.tolist()

if __name__ == "__main__":
    preds = predict(
        r"C:\Users\Malak\Year4_Term1\Machine Learning\Project_Again\MSI\submission\ew_test_data",
        r"C:\Users\Malak\Year4_Term1\Machine Learning\Project_Again\MSI\SVM\models\svm_resnet18_unknown_optimized.pkl"
    )
    print(preds)