#!/usr/bin/env python3
"""
Improved Mask Region Segmentation using Traditional Techniques
- Implementation of region-based segmentation methods for mask detection
- Additional post-processing for improved results
- Visualization and evaluation of results
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from skimage import filters, segmentation, color, measure
from skimage.morphology import closing, opening, disk, dilation, erosion
import glob
from tqdm import tqdm
import albumentations as A

# Constants
DATASET_DIR = "segmentataion_task/MSFD"
OUTPUT_DIR = "segmentataion_task/output/improved directory"
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
AUGMENT = True  # Enable data augmentation

# Set paths to dataset directories
DATASET_CSV = os.path.join(DATASET_DIR, '1', 'dataset.csv')
IMAGES_DIR = os.path.join(DATASET_DIR, '1', 'img')
FACE_CROP_DIR = os.path.join(DATASET_DIR, '1', 'face_crop')
SEGMENTATION_DIR = os.path.join(DATASET_DIR, '1', 'face_crop_segmentation')

# Create necessary directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def get_augmentation():
    """Create data augmentation pipeline"""
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5),
        ], p=0.3),
    ])

def load_and_preprocess_image(image_path, mask_path, augment=False):
    """Load and preprocess image with optional augmentation"""
    # Read image and mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize to standard size
    image = cv2.resize(image, (256, 256))
    mask = cv2.resize(mask, (256, 256))
    
    if augment:
        # Apply augmentation
        aug = get_augmentation()
        augmented = aug(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
    
    return image, mask

def threshold_based_segmentation(image):
    """Enhanced threshold-based segmentation"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Apply morphological operations
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return thresh

def edge_based_segmentation(image):
    """Enhanced edge-based segmentation with improved preprocessing"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Apply bilateral filter for edge-preserving smoothing
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply Sobel edge detection in both directions with larger kernel
    sobelx = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=5)
    
    # Combine Sobel results with magnitude
    edges = np.sqrt(sobelx**2 + sobely**2)
    edges = np.uint8(edges * 255 / np.max(edges))
    
    # Apply Canny edge detection with automatic threshold
    median = np.median(smooth)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    edges_canny = cv2.Canny(smooth, lower, upper)
    
    # Apply Laplacian edge detection
    edges_lap = cv2.Laplacian(smooth, cv2.CV_64F)
    edges_lap = np.uint8(np.absolute(edges_lap))
    
    # Combine all edge detection results
    edges = cv2.bitwise_or(edges, edges_canny)
    edges = cv2.bitwise_or(edges, edges_lap)
    
    # Dilate edges with larger kernel
    kernel = np.ones((7,7), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    
    # Fill holes using floodfill
    filled = edges.copy()
    h, w = edges.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(filled, mask, (0,0), 255)
    filled_inv = cv2.bitwise_not(filled)
    edges = edges | filled_inv
    
    # Apply morphological operations
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    
    return edges

def color_based_segmentation(image):
    """Enhanced color-based segmentation"""
    # Convert to multiple color spaces
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Create masks for different color ranges
    # HSV masks for light colors
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 30, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    # LAB mask for light colors
    l_thresh = 200
    mask_lab = cv2.inRange(lab, np.array([l_thresh, 0, 0]), np.array([255, 255, 255]))
    
    # Combine masks
    mask = cv2.bitwise_or(mask_white, mask_lab)
    
    # Apply morphological operations
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def post_process_mask(mask, image):
    """Enhanced post-processing of the segmentation mask"""
    # Apply morphological operations
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area and aspect ratio
    valid_contours = []
    max_area = 0
    max_contour = None
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Minimum area threshold
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h
            if 0.2 < aspect_ratio < 5.0:  # More relaxed aspect ratio threshold
                valid_contours.append(contour)
                if area > max_area:
                    max_area = area
                    max_contour = contour
    
    # Create new mask from valid contours
    mask = np.zeros_like(mask)
    cv2.drawContours(mask, valid_contours, -1, 255, -1)
    
    # If no valid contours found, use the largest contour
    if len(valid_contours) == 0 and max_contour is not None:
        cv2.drawContours(mask, [max_contour], -1, 255, -1)
    
    # Apply watershed segmentation for refinement
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.25*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Find unknown region
    sure_bg = cv2.dilate(mask, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Apply watershed
    markers = cv2.watershed(image, markers)
    mask = np.uint8(markers == 2) * 255
    
    # Final cleanup with multiple morphological operations
    kernel_small = np.ones((3,3), np.uint8)
    kernel_large = np.ones((7,7), np.uint8)
    
    # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    
    # Fill holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)
    
    # Smooth boundaries
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_large)
    
    return mask

def calculate_iou(pred_mask, gt_mask):
    """Calculate Intersection over Union"""
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def combine_masks(thresh_mask, edge_mask, color_mask):
    """Combine different segmentation masks with weighted voting"""
    # Convert masks to binary
    thresh_binary = thresh_mask > 127
    edge_binary = edge_mask > 127
    color_binary = color_mask > 127
    
    # Create weighted combination
    combined = np.zeros_like(thresh_mask, dtype=np.float32)
    
    # Edge-based gets highest weight (0.7)
    # Threshold-based gets medium weight (0.2)
    # Color-based gets lowest weight (0.1)
    combined[edge_binary] += 0.7
    combined[thresh_binary] += 0.2
    combined[color_binary] += 0.1
    
    # Threshold the weighted combination
    combined = (combined > 0.35).astype(np.uint8) * 255
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((7,7), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    
    # Apply watershed segmentation for refinement
    dist_transform = cv2.distanceTransform(combined, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.25*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Find unknown region
    sure_bg = cv2.dilate(combined, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Apply watershed
    markers = cv2.watershed(cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR), markers)
    combined = np.uint8(markers == 2) * 255
    
    # Final cleanup with multiple morphological operations
    kernel_small = np.ones((3,3), np.uint8)
    kernel_large = np.ones((7,7), np.uint8)
    
    # Remove small noise
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_small)
    
    # Fill holes
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_large)
    
    # Smooth boundaries
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_large)
    
    return combined

def process_face_images():
    """Process face images and generate segmentation masks"""
    # Get all face crop files
    face_crop_dir = os.path.join(DATASET_DIR, '1', 'face_crop')
    segmentation_dir = os.path.join(DATASET_DIR, '1', 'face_crop_segmentation')
    
    if not os.path.exists(face_crop_dir) or not os.path.exists(segmentation_dir):
        print(f"Error: Required directories not found. Please check paths:\n{face_crop_dir}\n{segmentation_dir}")
        return
    
    # Get all face files and sort them
    face_files = sorted([f for f in os.listdir(face_crop_dir) if f.endswith('.jpg')])
    face_files = face_files[:50]  # Process first 50 files
    
    results = []
    processed_count = 0
    
    for face_file in tqdm(face_files, desc="Processing images"):
        try:
            # Get base name without extension
            base_name = os.path.splitext(face_file)[0]
            
            # Read face image
            face_path = os.path.join(face_crop_dir, face_file)
            face_img = cv2.imread(face_path)
            if face_img is None:
                print(f"Could not read image: {face_file}")
                continue
                
            # Try to find corresponding mask file
            mask_found = False
            for mask_file in os.listdir(segmentation_dir):
                if mask_file.startswith(base_name):
                    mask_path = os.path.join(segmentation_dir, mask_file)
                    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask_img is not None:
                        mask_found = True
                        break
            
            if not mask_found:
                print(f"Mask not found for {face_file}")
                continue
                
            # Apply data augmentation if enabled
            if AUGMENT:
                augmented = get_augmentation()(image=face_img, mask=mask_img)
                face_img = augmented['image']
                mask_img = augmented['mask']
            
            # Apply segmentation methods
            thresh_mask = threshold_based_segmentation(face_img)
            edge_mask = edge_based_segmentation(face_img)
            color_mask = color_based_segmentation(face_img)
            
            # Combine results
            final_mask = combine_masks(thresh_mask, edge_mask, color_mask)
            
            # Calculate IoU scores
            thresh_iou = calculate_iou(mask_img, thresh_mask)
            edge_iou = calculate_iou(mask_img, edge_mask)
            color_iou = calculate_iou(mask_img, color_mask)
            final_iou = calculate_iou(mask_img, final_mask)
            
            # Store results
            results.append({
                'filename': face_file,
                'thresh_iou': thresh_iou,
                'edge_iou': edge_iou,
                'color_iou': color_iou,
                'final_iou': final_iou
            })
            
            # Save segmentation results
            output_path = os.path.join(RESULTS_DIR, f"{base_name}_segmented.jpg")
            cv2.imwrite(output_path, final_mask)
            
            # Create visualization
            vis_img = np.hstack([
                face_img,
                cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
            ])
            vis_path = os.path.join(PLOTS_DIR, f"{base_name}_visualization.jpg")
            cv2.imwrite(vis_path, vis_img)
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {face_file}: {str(e)}")
            continue
    
    if not results:
        print("No results to process!")
        return
        
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'segmentation_results.csv'), index=False)
    
    # Print average IoU scores
    print("\nAverage IoU Scores:")
    print(f"Threshold-based: {results_df['thresh_iou'].mean():.4f}")
    print(f"Edge-based: {results_df['edge_iou'].mean():.4f}")
    print(f"Color-based: {results_df['color_iou'].mean():.4f}")
    print(f"Final: {results_df['final_iou'].mean():.4f}")
    print(f"\nProcessed {processed_count} images")
    print(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    print("Starting improved mask region segmentation...")
    process_face_images() 