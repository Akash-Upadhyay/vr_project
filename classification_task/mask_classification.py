#!/usr/bin/env python3
"""
Traditional ML-based Face Mask Classification
- Implementation of traditional machine learning approaches
- Feature extraction using HOG and LBP
- Classification using SVM and Neural Network
"""

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import hog, local_binary_pattern
from tqdm import tqdm  # Add progress bar
import json

# Set paths
DATASET_DIR = os.path.join('classification_task', 'dataset')
WITH_MASK_DIR = os.path.join(DATASET_DIR, 'with_mask')
WITHOUT_MASK_DIR = os.path.join(DATASET_DIR, 'without_mask')
OUTPUT_DIR = os.path.join('classification_task', 'output', 'traditional_ml')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameters for feature extraction
IMG_SIZE = 64  # Resize images to 64x64
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
LBP_POINTS = 24
LBP_RADIUS = 3

def load_dataset():
    """Load and preprocess the dataset"""
    data = []
    labels = []
    
    # Load images with masks
    print("Loading images with masks...")
    mask_images = [f for f in os.listdir(WITH_MASK_DIR) 
                  if f.endswith(('.jpg', '.jpeg', '.png'))]
    for img_name in tqdm(mask_images, desc="Processing mask images"):
        img_path = os.path.join(WITH_MASK_DIR, img_name)
        try:
            if os.path.exists(img_path):
                data.append(img_path)
                labels.append(1)  # 1 for mask
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    # Load images without masks
    print("\nLoading images without masks...")
    no_mask_images = [f for f in os.listdir(WITHOUT_MASK_DIR) 
                     if f.endswith(('.jpg', '.jpeg', '.png'))]
    for img_name in tqdm(no_mask_images, desc="Processing no-mask images"):
        img_path = os.path.join(WITHOUT_MASK_DIR, img_name)
        try:
            if os.path.exists(img_path):
                data.append(img_path)
                labels.append(0)  # 0 for no mask
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    print(f"\nTotal images found: {len(data)}")
    print(f"Images with mask: {sum(labels)}")
    print(f"Images without mask: {len(labels) - sum(labels)}")
    
    return np.array(data), np.array(labels)

def extract_features(image_path):
    """Extract HOG and LBP features from an image"""
    # Read and preprocess image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Extract HOG features
    hog_features = hog(gray, orientations=HOG_ORIENTATIONS,
                      pixels_per_cell=HOG_PIXELS_PER_CELL,
                      cells_per_block=HOG_CELLS_PER_BLOCK,
                      block_norm='L2-Hys')
    
    # Extract LBP features
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3),
                          range=(0, LBP_POINTS + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    
    # Calculate statistical features
    mean = gray.mean()
    std = gray.std()
    
    # Combine all features
    features = np.concatenate([hog_features, hist, [mean, std]])
    
    return features

def prepare_data(image_paths, labels):
    """Prepare features and labels for training"""
    features = []
    valid_labels = []
    
    print("Extracting features from images...")
    for img_path, label in tqdm(zip(image_paths, labels), total=len(image_paths), 
                               desc="Feature extraction"):
        try:
            features.append(extract_features(img_path))
            valid_labels.append(label)
        except Exception as e:
            print(f"Error extracting features from {img_path}: {str(e)}")
            continue
    
    return np.array(features), np.array(valid_labels)

def train_svm(X_train, y_train):
    """Train SVM classifier"""
    print("\nTraining SVM classifier...")
    svm = SVC(kernel='rbf', probability=True, verbose=True)
    with tqdm(total=1, desc="SVM training") as pbar:
        svm.fit(X_train, y_train)
        pbar.update(1)
    return svm

def train_neural_network(X_train, y_train):
    """Train Neural Network classifier"""
    print("\nTraining Neural Network classifier...")
    nn = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),  # Larger architecture for more data
        activation='relu',
        solver='adam',
        max_iter=1000,
        verbose=True,
        batch_size='auto',  # Will use min(200, n_samples)
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.1
    )
    with tqdm(total=1, desc="Neural Network training") as pbar:
        nn.fit(X_train, y_train)
        pbar.update(1)
    return nn

def evaluate_model(model, X_test, y_test, model_name, run_dir):
    """Evaluate model and print results"""
    print(f"\nEvaluating {model_name}...")
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate and print metrics
    print(f"\n{model_name} Results:")
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred)
    print(report)
    
    # Save classification report
    with open(os.path.join(run_dir, f'{model_name.lower()}_classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Create and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(run_dir, f'{model_name.lower()}_confusion_matrix.png'))
    plt.close()
    
    return y_pred

def main():
    """Main execution function"""
    # Create timestamp for this run
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(OUTPUT_DIR, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    # Save script parameters
    params = {
        'IMG_SIZE': IMG_SIZE,
        'HOG_ORIENTATIONS': HOG_ORIENTATIONS,
        'HOG_PIXELS_PER_CELL': HOG_PIXELS_PER_CELL,
        'HOG_CELLS_PER_BLOCK': HOG_CELLS_PER_BLOCK,
        'LBP_POINTS': LBP_POINTS,
        'LBP_RADIUS': LBP_RADIUS,
        'DATASET_DIR': DATASET_DIR
    }
    with open(os.path.join(run_dir, 'parameters.json'), 'w') as f:
        json.dump(params, f, indent=4)
    
    # Load dataset
    print("Loading dataset...")
    image_paths, labels = load_dataset()
    
    # Split dataset with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # Save dataset split information
    split_info = {
        'train_size': len(X_train),
        'test_size': len(X_test),
        'total_size': len(image_paths),
        'mask_ratio': sum(labels) / len(labels)
    }
    with open(os.path.join(run_dir, 'dataset_info.json'), 'w') as f:
        json.dump(split_info, f, indent=4)
    
    # Prepare features
    print("\nPreparing training features...")
    X_train_features, y_train = prepare_data(X_train, y_train)
    print("\nPreparing testing features...")
    X_test_features, y_test = prepare_data(X_test, y_test)
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_test_scaled = scaler.transform(X_test_features)
    
    # Train and evaluate SVM
    print("\nTraining and evaluating SVM...")
    svm_model = train_svm(X_train_scaled, y_train)
    svm_pred = evaluate_model(svm_model, X_test_scaled, y_test, "SVM", run_dir)
    
    # Train and evaluate Neural Network
    print("\nTraining and evaluating Neural Network...")
    nn_model = train_neural_network(X_train_scaled, y_train)
    nn_pred = evaluate_model(nn_model, X_test_scaled, y_test, "Neural Network", run_dir)
    
    # Compare models
    plt.figure(figsize=(10, 5))
    models = ['SVM', 'Neural Network']
    accuracies = [
        (svm_pred == y_test).mean(),
        (nn_pred == y_test).mean()
    ]
    
    plt.bar(models, accuracies)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    plt.savefig(os.path.join(run_dir, 'model_comparison.png'))
    plt.close()
    
    # Save results summary
    results = [
        {'model': 'SVM', 'accuracy': accuracies[0]},
        {'model': 'Neural Network', 'accuracy': accuracies[1]}
    ]
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(run_dir, 'results_summary.csv'), index=False)
    
    print(f"\nResults saved to: {run_dir}")
    print("Training complete!")

if __name__ == "__main__":
    main() 