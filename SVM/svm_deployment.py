import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import joblib

# =======================
# CONFIG
# =======================
IMG_SIZE = 224
UNKNOWN_THRESHOLD = 0.6

CLASS_LABELS = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]
UNKNOWN_LABEL = "Uncertain"

MODEL_DIR = Path(__file__).parent / "models"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# =======================
# LOAD FEATURE EXTRACTOR
# =======================
def load_feature_extractor(device):
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = nn.Sequential(*list(resnet.children())[:-1])  # remove classifier
    model.eval().to(device)

    preprocess = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return model, preprocess

# =======================
# LOAD SVM + SCALER
# =======================
def load_models():
    svm_path = MODEL_DIR / "svm_resnet18_unknown_optimized.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"

    if not svm_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("SVM or Scaler model not found!")

    svm = joblib.load(svm_path)
    scaler = joblib.load(scaler_path)
    return svm, scaler

# =======================
# FEATURE EXTRACTION
# =======================
def extract_features(frame, model, preprocess, device):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(img)
        feat = feat.flatten(1)

    return feat.cpu().numpy()

def main():
    feature_model, preprocess = load_feature_extractor(DEVICE)
    svm_model, scaler = load_models()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # -------- Feature Extraction --------
        features = extract_features(frame, feature_model, preprocess, DEVICE)

        # -------- Scaling --------
        features_scaled = scaler.transform(features)

        # -------- Prediction --------
        proba = svm_model.predict_proba(features_scaled)
        pred_class = np.argmax(proba)
        confidence = np.max(proba)

        if confidence < UNKNOWN_THRESHOLD:
            label = UNKNOWN_LABEL
            color = (0, 0, 255)  
        else:
            label = CLASS_LABELS[pred_class]
            color = (0, 255, 0)  

        # -------- FPS --------
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # -------- Overlay --------
        cv2.putText(frame, f"Label: {label}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("SVM Real-Time Deployment", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()