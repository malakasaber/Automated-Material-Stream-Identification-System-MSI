# Automated Material Stream Identification (MSI) System

### Feature-Based Machine Learning Classifiers for Waste Material Recognition

## Overview

This project implements an **Automated Material Stream Identification (MSI) System** that classifies post-consumer waste into six recyclable categories plus one "Unknown" class. The system follows the complete classical Machine Learning pipeline: **Data Preprocessing**, **Feature Extraction**, **Model Training**, **Performance Evaluation**, and **Real-Time Deployment**.

The goal is to build a fully functional, feature-based vision system using **SVM** and **k-NN** classifiers, compare their performance, and deploy the best model in a **real-time camera-based application**.

---
## Dataset 
[Dataset](https://drive.google.com/drive/folders/1nZIc0PvX5mh6IiN1potWtPopxAHgthKy?usp=drive_link)
[Augmented Dataset](https://drive.google.com/drive/folders/1BNrisqSbppShLq2q7xq1PLECSm8wqLD-?usp=drive_link)
[Split Train Validation Augmented Dataset](https://drive.google.com/drive/folders/1ARaLmSgDnVauqe2L7wQCpCTv4rZB4ADH?usp=drive_link)
---

## Project Objectives

### 1. **Data Augmentation & Feature Extraction**

* Implement a pipeline to convert raw images into fixed-length numerical feature vectors.
* Apply data augmentation to increase dataset size by at least **30%**.
* Justify chosen augmentation techniques (rotation, flipping, scale, color jitter...).

### 2. **Classifier Training**

Train and evaluate two foundational ML models:

* **Variant A:** Support Vector Machine (SVM)
* **Variant B:** k-Nearest Neighbors (k-NN)

Both models use extracted feature vectors as input.

### 3. **Architecture Comparison**

Analyze and report trade-offs between:

* SVM vs. k-NN
* Chosen feature extraction approaches

### 4. **Robust Classification**

* Achieve **≥ 0.85 validation accuracy** across all six primary classes.
* Implement a rejection mechanism to detect out-of-distribution or uncertain inputs (class **ID 6: Unknown**).

### 5. **System Deployment**

Deploy the highest-performing model into an application that:

* Processes **live camera frames**
* Performs **real-time waste classification**
* Displays predicted class labels continuously

---

## Material Classes

The model must classify images into **7 categories**:

| ID | Class     | Description                                 |
| -- | --------- | ------------------------------------------- |
| 0  | Glass     | Bottles, jars, amorphous silicate materials |
| 1  | Paper     | Newspapers, office paper                    |
| 2  | Cardboard | Multi-layer cellulose packaging             |
| 3  | Plastic   | Bottles, containers, films                  |
| 4  | Metal     | Aluminum cans, steel pieces                 |
| 5  | Trash     | Contaminated/reject waste                   |
| 6  | Unknown   | Out-of-distribution or blurred items        |

### Dataset Structure

```
dataset/
│── glass/
│── paper/
│── cardboard/
│── plastic/
│── metal/
│── trash/
```

Each folder contains image samples representing its class.

---

## Technical Implementation

### 1. Data Augmentation

Mandatory techniques (minimum +30% data increase).
operations:

* Rotation
* Horizontal/vertical flip
* Scaling
* Color jitter
* Contrast/brightness adjustments

Justifications documented in the technical report.

---

### 2. Feature Extraction

Convert raw 2D/3D images into **1D fixed-length feature vectors**.

Examples of feature descriptors:

* Color histograms
* Edge histograms (Sobel, Canny)
* Texture descriptors (LBP, Haralick features)
* Shape descriptors (Hu moments)

All chosen descriptors are explained and justified.

---

### 3. Classifier Implementation

#### **Support Vector Machine (SVM)**

* Input: extracted feature vector
* Chose and justified kernel (linear, RBF, polynomial, etc.)
* Tuned hyperparameters (C, gamma, kernel degree)

#### **k-Nearest Neighbors (k-NN)**

* Input: extracted feature vector
* Chose value of **k**
* Select weighting scheme:

  * Uniform
  * Distance-based

#### Rejection Mechanism (Class 6)

Implemented per-model method:

* Threshold on decision scores (SVM)
* Distance threshold to nearest neighbors (k-NN)

---

### 4. Real-Time System Deployment

A final application:

* Captures frames from a live camera
* Extracts features in real-time
* Classify via best model
* Displays the predicted class on-screen

---

## Deliverables

### Source Code Repository

Contains code for:

* Data preprocessing + augmentation
* Feature extraction pipeline
* SVM & k-NN training and evaluation
* Real-time inference application

### Trained Model Files

Saved model weights (e.g., `.pkl` or `.joblib`).
Used for the hidden test set competition.

### Technical Report (PDF)

Includes:

* Feature extraction explanation
* Classifier analysis
* Performance comparison
* Justification of all design choices

---

## Hidden Test Set Competition

* A private dataset is used to rank all submitted models.
* Accuracy determines competition ranking
* A private leaderboard may be updated during evaluation.

---

## Repository Structure

```
MSI-Project/
│── src/
│   ├── augmentation/
│   ├── feature_extraction/
│   ├── classifiers/
│   ├── utils/
│   └── real_time_app/
│── models/
│── dataset/
│── results/
│── README.md
│── report.pdf
```
