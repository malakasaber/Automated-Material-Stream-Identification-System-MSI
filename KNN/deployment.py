import cv2
import torch
import numpy as np
from pathlib import Path
import joblib
from time import time
from PIL import Image

# ------------------------------
# CONFIG
# ------------------------------
UNKNOWN_LABEL = "Uncertain"
UNKNOWN_THRESHOLD = 0.50

CLASS_LABELS = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

MODEL_DIR = Path(__file__).parent / "models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# LOAD CNN FEATURE EXTRACTOR
# ------------------------------
def load_cnn_model(device=DEVICE):
    from torchvision.models import resnet50, ResNet50_Weights

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()  # remove classifier head
    model.eval()
    model.to(device)

    preprocess = weights.transforms()  # correct preprocessing
    return model, preprocess

# ------------------------------
# LOAD KNN MODEL (Pipeline)
# ------------------------------
def load_knn_model():
    path = MODEL_DIR / "knn_models.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found! Checked path: {path}")
    return joblib.load(path)

# ------------------------------
# FEATURE EXTRACTION
# ------------------------------
def extract_feature(frame, cnn_model, preprocess, device=DEVICE):
    # BGR -> RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    # Preprocess (resize + normalize)
    x = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = cnn_model(x)

    return feat.cpu().numpy().reshape(1, -1)

def main():
    print("Using device:", DEVICE)

    print("Loading CNN feature extractor...")
    cnn_model, preprocess = load_cnn_model()

    print("Loading kNN model...")
    knn_pipe = load_knn_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open camera")
        return

    prev_time = time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract features
        feat = extract_feature(frame, cnn_model, preprocess)

        # Prediction
        proba = knn_pipe.predict_proba(feat)
        pred_class = proba.argmax(axis=1)[0]
        confidence = proba.max()

        # Unknown rejection
        if confidence < UNKNOWN_THRESHOLD:
            label = UNKNOWN_LABEL
            color = (0, 0, 255)  
        else:
            label = CLASS_LABELS[pred_class]
            color = (0, 255, 0)  

        # FPS calculation
        curr_time = time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        # Overlay text
        cv2.putText(
            frame,
            f"Label: {label}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
        )
        cv2.putText(
            frame,
            f"Confidence: {confidence:.2f}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
        )

        cv2.imshow("Real-Time Waste Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
