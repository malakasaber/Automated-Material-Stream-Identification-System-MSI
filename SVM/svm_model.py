import os
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from pathlib import Path




# ----------------- CONFIGURATION -----------------
TRAIN_DIR = r"C:\Users\Malak\Year4_Term1\Machine Learning\Project_Again\MSI\SVM\dataset_train_val-20251214T215240Z-1-0011\dataset_train_val\train"
VAL_DIR   = r"C:\Users\Malak\Year4_Term1\Machine Learning\Project_Again\MSI\SVM\dataset_train_val-20251214T215240Z-1-0011\dataset_train_val\val"
OUTPUT_DIR = Path(__file__).parent / "models"

BATCH_SIZE = 32
PROB_THRESHOLD = 0.6
UNKNOWN_LABEL = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------- SAFE IMAGE LOADER -----------------
def pil_loader_safe(path):
    try:
        return Image.open(path).convert("RGB")
    except UnidentifiedImageError:
        print(f"Skipping corrupted image: {path}")
        return Image.new("RGB", (224, 224))

class ImageFolderSafe(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = pil_loader_safe(path)
        if self.transform:
            sample = self.transform(sample)
        return sample, target

# ----------------- TRANSFORMS -----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------------- LOAD DATA -----------------
train_dataset = ImageFolderSafe(TRAIN_DIR, transform=transform)
val_dataset = ImageFolderSafe(VAL_DIR, transform=transform)

print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")

# ----------------- FEATURE EXTRACTION -----------------
print("Loading ResNet18 for feature extraction...")
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
feature_extractor.eval().to(device)

def extract_features(dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    feats_list, labels_list = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            feats = feature_extractor(imgs).flatten(1)
            feats_list.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())

    return np.vstack(feats_list), np.concatenate(labels_list)

print("Extracting train features...")
X_train, y_train = extract_features(train_dataset)

print("Extracting validation features...")
X_val, y_val = extract_features(val_dataset)

# ----------------- SCALING -----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# ----------------- GRID SEARCH FOR SVM -----------------
print("Performing Grid Search for best SVM hyperparameters...")
param_grid = {
    'C': [0.1, 1, 10, 50],
    'gamma': ['scale', 0.01, 0.1, 1],
    'kernel': ['rbf']
}

svm = SVC(class_weight='balanced', probability=True)
grid_search = GridSearchCV(
    estimator=svm,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

print("\nBest parameters found:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

best_svm = grid_search.best_estimator_

# ================= TRAIN METRICS =================
train_probs = best_svm.predict_proba(X_train_scaled)
train_max_probs = train_probs.max(axis=1)
train_preds = best_svm.predict(X_train_scaled)

train_preds_safe = np.where(train_max_probs < PROB_THRESHOLD, UNKNOWN_LABEL, train_preds)

train_acc_incl = accuracy_score(y_train, train_preds_safe)
mask_train_known = train_preds_safe != UNKNOWN_LABEL
train_acc_excl = accuracy_score(y_train[mask_train_known], train_preds_safe[mask_train_known])

print("\n================ TRAIN RESULTS ================")
print(f"Train Accuracy (including Unknowns): {train_acc_incl:.4f}")
print(f"Train Accuracy (excluding Unknowns): {train_acc_excl:.4f}")

# ================= VALIDATION METRICS =================
val_probs = best_svm.predict_proba(X_val_scaled)
val_max_probs = val_probs.max(axis=1)
val_preds = best_svm.predict(X_val_scaled)

val_preds_safe = np.where(val_max_probs < PROB_THRESHOLD, UNKNOWN_LABEL, val_preds)

val_acc_incl = accuracy_score(y_val, val_preds_safe)
mask_val_known = val_preds_safe != UNKNOWN_LABEL
val_acc_excl = accuracy_score(y_val[mask_val_known], val_preds_safe[mask_val_known])

print("\n============= VALIDATION RESULTS =============")
print(f"Validation Accuracy (including Unknowns): {val_acc_incl:.4f}")
print(f"Validation Accuracy (excluding Unknowns): {val_acc_excl:.4f}")

print("\nClassification Report (excluding Unknowns):")
print(classification_report(y_val[mask_val_known], val_preds_safe[mask_val_known]))

print("\nConfusion Matrix (excluding Unknowns):")
print(confusion_matrix(y_val[mask_val_known], val_preds_safe[mask_val_known]))

unknown_count = np.sum(val_preds_safe == UNKNOWN_LABEL)
print(f"\nUnknown predictions: {unknown_count} ({unknown_count/len(y_val):.2%})")

# ================= GENERALIZATION GAP =================
gap = train_acc_excl - val_acc_excl
print("\n================ GENERALIZATION GAP ================")
print(f"Train – Validation Accuracy Gap: {gap:.4f}")

# ----------------- SAVE MODEL -----------------
OUTPUT_DIR.mkdir(exist_ok=True)

joblib.dump(best_svm, OUTPUT_DIR / "svm_resnet18_unknown_optimized.pkl")
joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")

print("\nOptimized SVM model and scaler saved successfully.")
