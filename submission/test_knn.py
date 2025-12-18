import os
from pathlib import Path
import numpy as np
import pickle
import cv2
import torch
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.preprocessing import StandardScaler


def predict(dataFilePath, bestModelPath):
    """
    Predict material classes for images in the given folder.
    
    Parameters:
    -----------
    dataFilePath : str
        Path to folder containing images to classify
    bestModelPath : str
        Path to the saved k-NN model (pickle file)
    
    Returns:
    --------
    list
        List of predicted class IDs (0-6) for each image
    """
    
    # Configuration constants
    UNKNOWN_CLASS_ID = 6
    UNKNOWN_PROBA_THRESHOLD = 0.50
    IMG_SIZE = 224
    IMG_EXTS = ('.jpg', '.jpeg', '.png')
    
    # 1. Load the trained k-NN model
    with open(bestModelPath, 'rb') as f:
        model = pickle.load(f)
    
    # 2. Initialize ResNet50 feature extractor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = ResNet50_Weights.DEFAULT
    feature_extractor = resnet50(weights=weights)
    feature_extractor.fc = torch.nn.Identity()
    feature_extractor.eval()
    feature_extractor.to(device)
    
    # Get preprocessing transform
    preprocess = weights.transforms()
    
    # 3. Load and sort images from dataFilePath
    data_folder = Path(dataFilePath)
    image_files = sorted([
        f for f in data_folder.iterdir() 
        if f.is_file() and f.suffix.lower() in IMG_EXTS
    ])
    
    if not image_files:
        return []
    
    # 4. Extract features for all images
    all_features = []
    
    with torch.no_grad():
        for img_path in image_files:
            # Load and preprocess image
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                # Handle corrupted images - add zero features
                all_features.append(np.zeros(2048, dtype=np.float32))
                continue
            
            # Resize to expected size
            img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE), 
                                interpolation=cv2.INTER_AREA)
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Convert numpy array to PIL Image (required by torchvision transforms)
            img_pil = Image.fromarray(img_rgb)
            
            # Apply ResNet50 preprocessing and convert to tensor
            img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
            
            # Extract features
            features = feature_extractor(img_tensor)
            features = features.detach().cpu().numpy().flatten()
            all_features.append(features)
    
    # Convert to numpy array
    X_test = np.array(all_features, dtype=np.float32)
    
    # 5. Make predictions with the k-NN model
    # Get probability predictions
    proba = model.predict_proba(X_test)
    y_pred = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    
    # 6. Apply rejection mechanism for unknown class
    # If confidence is below threshold, classify as Unknown (ID 6)
    y_pred_final = y_pred.copy()
    y_pred_final[conf < UNKNOWN_PROBA_THRESHOLD] = UNKNOWN_CLASS_ID
    
    # 7. Return predictions as a list
    return y_pred_final.tolist()


if __name__ == "__main__":
    # Example usage for testing
    test_folder = r"C:\Users\Malak\Year4_Term1\Machine Learning\Project_Again\MSI\submission\ew_test_data"
    model_path = r"C:\Users\Malak\Year4_Term1\Machine Learning\Project_Again\MSI\KNN\models\knn_models.pkl"
    
    predictions = predict(test_folder, model_path)
    print(f"Predictions: {predictions}")

