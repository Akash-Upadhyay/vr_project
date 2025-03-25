# Face Mask Detection and Segmentation Project

## Introduction
This project focuses on developing and comparing different approaches for face mask detection and segmentation in images. The main objectives are:
1. Accurately classify whether a person is wearing a mask or not
2. Precisely segment the mask region in the image
3. Compare traditional computer vision techniques with deep learning methods

## Dataset
The project uses the MSFD (Masked Face Detection) dataset, which contains:
- Original face images
- Face crop images (cropped face regions)
- Segmentation masks for face masks
- Classification labels (masked/unmasked)

### Dataset Structure
```
project_root/
├── classification_task/
│   ├── dataset/
│   │   ├── with_mask/              # Images of people wearing masks
│   │   └── without_mask/           # Images of people not wearing masks
│   ├── cnn_mask_classification.py  # CNN-based classification script
│   ├── mask_classification.py      # Traditional ML classification script
│   └── output/
│       ├── cnn/                    # CNN model results
│       └── traditional_ml/         # Traditional ML results
│
└── segmentation_task/
    ├── MSFD/
    │   └── 1/
    │       ├── face_crop/           # Cropped face images
    │       ├── face_crop_segmentation/  # Ground truth masks
    │       └── dataset.csv          # Dataset annotations
    └── output/
        ├── traditional/             # Traditional approach results
        └── improved/                # Improved approach results
```

## Methodology

### Classification Task
The project implements multiple classification approaches:

1. **Traditional Computer Vision Features**
   - HOG (Histogram of Oriented Gradients)
   - LBP (Local Binary Patterns)
   - Color histograms
   - Combined feature vector

2. **Machine Learning Models**
   - Support Vector Machine (SVM)
   - Neural Network

3. **Deep Learning Models**
   - CNN with different configurations
   - Data augmentation techniques

### Segmentation Task
The project implements three main segmentation techniques:

1. **Threshold-based Segmentation**
   - Uses CLAHE for contrast enhancement
   - Applies Otsu's thresholding
   - Includes morphological operations for refinement

2. **Edge-based Segmentation**
   - Combines multiple edge detection methods:
     - Sobel edge detection (x and y directions)
     - Canny edge detection with automatic thresholding
     - Laplacian edge detection
   - Uses bilateral filtering for edge-preserving smoothing
   - Implements hole-filling and morphological operations

3. **Color-based Segmentation**
   - Works in multiple color spaces (HSV and LAB)
   - Creates masks for light-colored regions
   - Applies morphological operations for refinement

### Mask Combination Strategy
The final segmentation is achieved through a weighted combination:
- Edge-based: 70% weight
- Threshold-based: 20% weight
- Color-based: 10% weight

### Post-processing Pipeline
1. Morphological operations for noise removal
2. Contour filtering based on area and aspect ratio
3. Watershed segmentation for refinement
4. Multi-scale morphological operations for final cleanup

## Results

### Classification Performance

#### Traditional ML Models
1. **Support Vector Machine (SVM)**
   - Accuracy: 94.62%
   - Precision: 0.95
   - Recall: 0.94
   - F1-Score: 0.95
   - Best performing traditional ML model

2. **Neural Network**
   - Accuracy: 94.01%
   - Precision: 0.94
   - Recall: 0.94
   - F1-Score: 0.94
   - Good balance of performance

#### CNN Models with Different Configurations
1. **Configuration 1**
   - Validation Accuracy: 96.33%
   - Validation Loss: 0.1595
   - Parameters:
     - Image Size: 128x128
     - Epochs: 10
     - Batch Size: 32

2. **Configuration 2**
   - Validation Accuracy: 97.07%
   - Validation Loss: 0.0937
   - Parameters:
     - Image Size: 128x128
     - Epochs: 10
     - Batch Size: 32
   - Best performing configuration

3. **Configuration 3**
   - Validation Accuracy: 96.70%
   - Validation Loss: 0.0945
   - Parameters:
     - Image Size: 128x128
     - Epochs: 10
     - Batch Size: 32

### Model Comparison
- **Best Overall Performance**: CNN Configuration 2 (97.07% accuracy)
- **Best Traditional ML**: SVM (94.62% accuracy)
- **Neural Network**: 94.01% accuracy

### Segmentation Performance

#### Traditional Approach
1. **Individual Methods**
   - Threshold-based: 0.2251 IoU
   - Edge-based: 0.2897 IoU
   - Color-based: 0.2529 IoU

2. **Final Combined Result**
   - Combined approach: 0.3304 IoU

#### Improved Approach
1. **Individual Methods**
   - Threshold-based: 0.2138 IoU
   - Edge-based: 0.3707 IoU
   - Color-based: 0.1524 IoU

2. **Final Combined Result**
   - Combined approach: 0.3708 IoU

### Analysis of Results
- CNN models achieved the best classification performance
- SVM showed strong performance among traditional ML methods
- Improved segmentation approach showed better performance than traditional approach
- Edge-based segmentation performed best in both approaches
- The improved approach showed better edge detection (0.3707 vs 0.2897 IoU)
- Color-based segmentation in improved approach is more selective (0.1524 vs 0.2529 IoU)

## How to Run the Code

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running Classification
```bash
python3 classification_task/cnn_mask_classification.py
python3 classification_task/mask_classification.py
```

### Running Segmentation
```bash
python3 segmentataion_task/mask_segmentation.py
python3 segmentataion_task/improved_mask_segmentation.py
```

### Expected Outputs

#### Classification Task Outputs
```
classification_task/output/
├── cnn/
│   ├── overall_results.json        # Performance metrics for all CNN configs
│   ├── model_config_*.h5           # Saved CNN models
│   └── training_history_*.png      # Training curves
└── traditional_ml/
    └── run_*/                      # Results for each run
        ├── results_summary.csv     # Performance metrics
        ├── model_comparison.png    # Model comparison plots
        └── classification_reports/ # Detailed classification reports
```

#### Segmentation Task Outputs
```
segmentation_task/output/
├── traditional/
│   ├── results/                    # Generated segmentation masks
│   ├── plots/                      # Visualization plots
│   └── segmentation_results.csv    # IoU scores and metrics
└── improved/
    ├── results/                    # Generated segmentation masks
    ├── plots/                      # Visualization plots
    └── segmentation_results.csv    # IoU scores and metrics
```

## Observations and Analysis

### Classification Task
1. **Strengths**
   - High accuracy across different models
   - Robust to variations in lighting and pose
   - Fast inference time

2. **Challenges**
   - False positives in low-light conditions
   - Difficulty with partially occluded faces
   - Performance degradation with extreme angles

### Segmentation Task
1. **Strengths**
   - Robust to different lighting conditions through CLAHE
   - Effective edge detection using multiple methods
   - Good handling of various mask types
   - Efficient post-processing pipeline

2. **Challenges**
   - Inconsistent mask boundaries
   - False positives in color-based segmentation
   - Noise in edge detection

### Solutions Implemented
1. **Classification**
   - Data augmentation for better generalization
   - Multiple CNN configurations for optimal performance
   - Feature normalization for better performance

2. **Segmentation**
   - Multi-scale morphological operations
   - Weighted combination strategy
   - Bilateral filtering and hole-filling

### Future Improvements
1. **Classification**
   - Implement attention mechanisms
   - Add more sophisticated data augmentation
   - Explore transformer-based architectures

2. **Segmentation**
   - Implement adaptive thresholding
   - Add more sophisticated post-processing
   - Explore deep learning approaches
   - Optimize hyperparameters

## Dependencies
- OpenCV (cv2)
- NumPy
- scikit-image
- scikit-learn
- albumentations
- pandas
- matplotlib
- tqdm
- PyTorch (for deep learning models)
- TensorFlow (optional, for additional models)