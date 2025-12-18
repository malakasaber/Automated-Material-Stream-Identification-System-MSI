import numpy as np
import json
from typing import List
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

FEATURE_DIR = str(Path(__file__).parent / "features")

UNKNOWN_CLASS_ID = 6
UNKNOWN_PROBA_THRESHOLD = 0.50
CLASS_LABELS = ["cardboard", "glass", "metal", "paper", "plastic", "trash", "unknown"]

def _print_confusion_matrix(cm: np.ndarray, labels: List[str]):
    width = max(7, max(len(s) for s in labels) + 2)
    header = " " * width + "".join(f"{s:>{width}}" for s in labels)
    print(header)
    for i, row in enumerate(cm):
        print(f"{labels[i]:>{width}}" + "".join(f"{int(v):>{width}d}" for v in row))

def load_features():
    X_train = np.load(f"{FEATURE_DIR}/X_train.npy")
    y_train = np.load(f"{FEATURE_DIR}/y_train.npy")
    X_val   = np.load(f"{FEATURE_DIR}/X_val.npy")
    y_val   = np.load(f"{FEATURE_DIR}/y_val.npy")
    return X_train, y_train, X_val, y_val

def main():
    X_train, y_train, X_val, y_val = load_features()
    print("Train:", X_train.shape, "Val:", X_val.shape)

    # Pipeline: scale → k-NN 
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(algorithm="brute"))
    ])

    # Feature-friendly grid (cosine is a strong baseline for pretrained CNN features).
    param_grid = [
        {
            "scaler": [StandardScaler(), MinMaxScaler()],
            "knn__n_neighbors": [3, 5, 7, 9, 11],
            "knn__weights": ["distance", "uniform"],
            "knn__metric": ["cosine"],
        },
        {
            "scaler": [StandardScaler(), MinMaxScaler()],
            "knn__n_neighbors": [3, 5, 7, 9, 11],
            "knn__weights": ["distance", "uniform"],
            "knn__metric": ["minkowski"],
            "knn__p": [2],
        },
    ]

   

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="accuracy",
        cv=2,
        n_jobs=-1,
        verbose=1,
    )

    grid.fit(X_train, y_train)
    print("Best params:", grid.best_params_)
    print("Best CV score:", grid.best_score_)

    try:
        out_path = Path(__file__).parent / "best_gridsearch_params.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_params": grid.best_params_,
                    "best_score": float(grid.best_score_),
                    "unknown_class_id": int(UNKNOWN_CLASS_ID),
                    "unknown_proba_threshold": float(UNKNOWN_PROBA_THRESHOLD),
                },
                f,
                indent=2,
                default=str,
            )
        print(f"Best parameters saved to '{out_path.name}'")
    except Exception as e:
        print(f"Warning: failed to save best params json: {e}")

    best_model = grid.best_estimator_

    #save model for deployment
    import joblib
    joblib.dump(best_model,"models/knn_models.pkl")

    # Use one probability matrix for both predicted class and confidence.
    proba = best_model.predict_proba(X_val)
    y_pred = proba.argmax(axis=1)
    conf = proba.max(axis=1)

    # Rejection mechanism: low confidence => Unknown (ID 6)
    y_pred_reject = y_pred.copy()
    y_pred_reject[conf < UNKNOWN_PROBA_THRESHOLD] = UNKNOWN_CLASS_ID

    accepted_mask = conf >= UNKNOWN_PROBA_THRESHOLD
    coverage = float(np.mean(accepted_mask)) if conf.size else 0.0 
    rejected = int(np.sum(~accepted_mask)) if conf.size else 0

    print("\n=== Unknown-class rejection (KNN) ===")
    print(f"Unknown class id: {UNKNOWN_CLASS_ID}")
    print(f"Probability threshold: {UNKNOWN_PROBA_THRESHOLD}")
    print(f"Coverage (accepted fraction): {coverage:.4f}")
    print(f"Rejected (as Unknown): {rejected}")
    print("Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_val, y_pred_reject, labels=[0, 1, 2, 3, 4, 5, 6])
    _print_confusion_matrix(cm, CLASS_LABELS)
    print("Classification report (with Unknown=6):")
    report_text = classification_report(
        y_val,
        y_pred_reject,
        labels=[0, 1, 2, 3, 4, 5, 6],
        target_names=CLASS_LABELS,
        zero_division=0,
    )
    print(report_text)

    # Dump classification report to disk for reporting/reuse.
    try:
        report_dict = classification_report(
            y_val,
            y_pred_reject,
            labels=[0, 1, 2, 3, 4, 5, 6],
            target_names=CLASS_LABELS,
            zero_division=0,
            output_dict=True,
        )
        report_out_path = Path(__file__).parent / "classification_report_unknown.json"
        with open(report_out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "labels": CLASS_LABELS,
                    "unknown_class_id": int(UNKNOWN_CLASS_ID),
                    "unknown_proba_threshold": float(UNKNOWN_PROBA_THRESHOLD),
                    "coverage": float(coverage),
                    "rejected": int(rejected),
                    "report": report_dict,
                },
                f,
                indent=2,
            )
        print(f"Classification report dumped to '{report_out_path.name}'")
    except Exception as e:
        print(f"Warning: failed to dump classification report: {e}")

if __name__ == "__main__":
    main()