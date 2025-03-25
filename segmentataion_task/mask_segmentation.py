#!/usr/bin/env python3
"""
Mask Region Segmentation using Traditional Techniques
- Implementation of region-based segmentation methods for mask detection
- Visualization and evaluation of results
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from skimage import filters, segmentation, color, measure
from skimage.morphology import closing, opening, disk, dilation
import glob

# Set paths to dataset directories
DATASET_DIR = os.path.join(os.path.dirname(__file__), 'MSFD')
DATASET_CSV = os.path.join(DATASET_DIR, '1', 'dataset.csv')
IMAGES_DIR = os.path.join(DATASET_DIR, '1', 'img')
FACE_CROP_DIR = os.path.join(DATASET_DIR, '1', 'face_crop')
SEGMENTATION_DIR = os.path.join(DATASET_DIR, '1', 'face_crop_segmentation')

# Create output directory for results
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output', 'traditional')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')

# Create necessary directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_dataset():
    """Load the dataset CSV file"""
    df = pd.read_csv(DATASET_CSV)
    # Filter for faces with masks
    masked_faces = df[df['with_mask'] == True]
    return masked_faces

def threshold_based_segmentation(image):
    """
    Apply threshold-based segmentation to detect mask regions
    
    Args:
        image: Input color image
        
    Returns:
        Binary mask highlighting the mask region
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    opening_result = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    closing_result = cv2.morphologyEx(opening_result, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    return closing_result

def edge_based_segmentation(image):
    """
    Apply edge-based segmentation to detect mask regions
    
    Args:
        image: Input color image
        
    Returns:
        Binary mask highlighting the mask region
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges using Canny edge detector
    edges = cv2.Canny(blurred, 30, 100)
    
    # Dilate edges to connect broken lines
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Fill in the regions using morphological closing
    closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel, iterations=5)
    
    return closed_edges

def color_based_segmentation(image):
    """
    Apply color-based segmentation to detect mask regions
    
    Args:
        image: Input color image
        
    Returns:
        Binary mask highlighting the mask region
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for blue/white/light colored masks
    # These values may need adjustment depending on your dataset
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 30, 255])
    
    # Create masks for different colors
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Combine masks
    combined_mask = cv2.bitwise_or(blue_mask, white_mask)
    
    # Apply morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    opening_result = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    closing_result = cv2.morphologyEx(opening_result, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    return closing_result

def evaluate_segmentation(segmented, ground_truth):
    """
    Calculate IoU (Intersection over Union) score for evaluation
    
    Args:
        segmented: Binary segmentation result
        ground_truth: Binary ground truth mask
        
    Returns:
        IoU score
    """
    # Ensure both masks are binary (0 or 1)
    segmented_bin = (segmented > 0).astype(np.uint8)
    ground_truth_bin = (ground_truth > 0).astype(np.uint8)
    
    # Calculate intersection and union
    intersection = cv2.bitwise_and(segmented_bin, ground_truth_bin).sum()
    union = cv2.bitwise_or(segmented_bin, ground_truth_bin).sum()
    
    # Avoid division by zero
    if union == 0:
        return 0.0
    
    return intersection / union

def process_face_images():
    """Process a batch of face images for mask segmentation"""
    # Find face crop images that have corresponding segmentation ground truth
    face_crops = glob.glob(os.path.join(FACE_CROP_DIR, '*.jpg'))
    
    results = []
    
    for face_idx, face_path in enumerate(face_crops[:10]):  # Process first 10 images for demonstration
        # Get image basename
        basename = os.path.basename(face_path)
        image_id = basename.split('_')[0]
        face_id = basename.split('.')[0]
        
        # Check if corresponding segmentation mask exists
        segmentation_paths = glob.glob(os.path.join(SEGMENTATION_DIR, f"{face_id}*.jpg"))
        
        if not segmentation_paths:
            continue
            
        # Read the face image
        face_img = cv2.imread(face_path)
        if face_img is None:
            print(f"Could not read image: {face_path}")
            continue
            
        # Apply segmentation methods
        threshold_result = threshold_based_segmentation(face_img)
        edge_result = edge_based_segmentation(face_img)
        color_result = color_based_segmentation(face_img)
        
        # Create a combined result (ensemble approach)
        combined_result = cv2.bitwise_or(
            cv2.bitwise_or(threshold_result, edge_result),
            color_result
        )
        
        # Save results
        cv2.imwrite(os.path.join(RESULTS_DIR, f"{face_id}_threshold.jpg"), threshold_result)
        cv2.imwrite(os.path.join(RESULTS_DIR, f"{face_id}_edge.jpg"), edge_result)
        cv2.imwrite(os.path.join(RESULTS_DIR, f"{face_id}_color.jpg"), color_result)
        cv2.imwrite(os.path.join(RESULTS_DIR, f"{face_id}_combined.jpg"), combined_result)
        
        # Visualization output
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        plt.title('Original Face Image')
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(threshold_result, cmap='gray')
        plt.title('Threshold Segmentation')
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.imshow(edge_result, cmap='gray')
        plt.title('Edge Segmentation')
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        plt.imshow(color_result, cmap='gray')
        plt.title('Color Segmentation')
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.imshow(combined_result, cmap='gray')
        plt.title('Combined Result')
        plt.axis('off')
        
        plt.subplot(2, 3, 6)
        plt.imshow(cv2.imread(segmentation_paths[0], cv2.IMREAD_GRAYSCALE), cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{face_id}_comparison.png"))
        plt.close()
        
        # Calculate IoU for each method
        ground_truth = cv2.imread(segmentation_paths[0], cv2.IMREAD_GRAYSCALE)
        threshold_iou = evaluate_segmentation(threshold_result, ground_truth)
        edge_iou = evaluate_segmentation(edge_result, ground_truth)
        color_iou = evaluate_segmentation(color_result, ground_truth)
        combined_iou = evaluate_segmentation(combined_result, ground_truth)
        
        results.append({
            'image_id': image_id,
            'face_id': face_id,
            'threshold_iou': threshold_iou,
            'edge_iou': edge_iou,
            'color_iou': color_iou,
            'combined_iou': combined_iou
        })
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, 'segmentation_results.csv'), index=False)
    
    # Print average IoU scores
    print("\nAverage IoU Scores:")
    print(f"Threshold-based: {df['threshold_iou'].mean():.4f}")
    print(f"Edge-based: {df['edge_iou'].mean():.4f}")
    print(f"Color-based: {df['color_iou'].mean():.4f}")
    print(f"Combined: {df['combined_iou'].mean():.4f}")

if __name__ == "__main__":
    print("Starting mask region segmentation...")
    process_face_images()
    print(f"Results saved to {OUTPUT_DIR} directory") 