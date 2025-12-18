# test.py
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
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
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    return model, weights

def _extract_features_single(model, img_paths, preprocess, device: str):
    feats = []
    with torch.no_grad():
        for path in img_paths:
            img = Image.open(path).convert("RGB")
            tensor = preprocess(img).unsqueeze(0).to(device)
            feat = model(tensor)
            feats.append(feat.cpu().numpy())
    return np.vstack(feats)

def predict(dataFilePath, bestModelPath):
    data_path = Path(dataFilePath)
    img_paths = sorted([p for p in data_path.iterdir() if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])
    if not img_paths:
        raise ValueError(f"No images found in {dataFilePath}")

    # Load model
    knn_model = joblib.load(bestModelPath)

    # Feature extractor
    model, weights = _build_model(DEVICE)
    preprocess = weights.transforms()

    # Extract features
    X_test = _extract_features_single(model, img_paths, preprocess, DEVICE)

    # Predict probabilities
    proba = knn_model.predict_proba(X_test)
    y_pred = proba.argmax(axis=1)
    conf = proba.max(axis=1)

    # Unknown-class rejection
    y_pred[conf < UNKNOWN_PROBA_THRESHOLD] = UNKNOWN_CLASS_ID

    return y_pred.tolist()

if __name__ == "__main__":
    preds = predict(
        r"C:\Users\Malak\Year4_Term1\Machine Learning\Project_Again\MSI\submission\ew_test_data",
        r"C:\Users\Malak\Year4_Term1\Machine Learning\Project_Again\MSI\KNN\models\knn_models.pkl"
    )
    print(preds)
