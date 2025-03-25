#!/usr/bin/env python3
"""
CNN-based Face Mask Classification
Simple and effective implementation for mask detection
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

# Set paths
DATASET_DIR = os.path.join('classification_task', 'dataset')
WITH_MASK_DIR = os.path.join(DATASET_DIR, 'with_mask')
WITHOUT_MASK_DIR = os.path.join(DATASET_DIR, 'without_mask')
OUTPUT_DIR = os.path.join('classification_task', 'output', 'cnn')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Image Parameters
IMG_SIZE = (128, 128)
EPOCHS = 10
BATCH_SIZE = 32
MEMORY_LIMIT = 1000  # Process images in chunks to save memory

def load_images_from_folder(folder, label):
    """Load and preprocess images from a folder"""
    data = []
    labels = []
    
    image_files = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Process images in chunks to save memory
    for i in range(0, len(image_files), MEMORY_LIMIT):
        chunk = image_files[i:i + MEMORY_LIMIT]
        for filename in tqdm(chunk, desc=f"Processing {'mask' if label == 1 else 'no mask'} images (chunk {i//MEMORY_LIMIT + 1})"):
            file_path = os.path.join(folder, filename)
            try:
                with Image.open(file_path) as img:
                    img = img.convert("RGB").resize(IMG_SIZE)
                    img_array = np.array(img) / 255.0  # Normalize
                    data.append(img_array)
                    labels.append(label)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    return np.array(data), np.array(labels)

def create_cnn_model(config_num):
    """Create CNN model with different configurations"""
    if config_num == 1:
        # Configuration 1: Original simple architecture
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        optimizer = 'adam'
        
    elif config_num == 2:
        # Configuration 2: Deeper network with more filters
        model = keras.Sequential([
            layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(512, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        optimizer = 'adam'
        
    else:  # config_num == 3
        # Configuration 3: Wider network with batch normalization
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        optimizer = 'adam'
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history, config_num):
    """Plot and save training history for each configuration"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.title(f'Training and Validation Accuracy (Config {config_num})')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title(f'Training and Validation Loss (Config {config_num})')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'training_history_config_{config_num}.png'))
    plt.close()

def plot_configuration_comparison(results):
    """Plot comparison of different configurations"""
    plt.figure(figsize=(10, 5))
    
    configs = [f'Config {r["config"]}' for r in results]
    accuracies = [r['validation_accuracy'] for r in results]
    
    plt.bar(configs, accuracies)
    plt.title('Model Configuration Comparison')
    plt.ylabel('Validation Accuracy')
    plt.ylim(0, 1)
    
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'configuration_comparison.png'))
    plt.close()

def main():
    """Main execution function"""
    # Load mask images
    print("Loading dataset...")
    print("Loading mask images...")
    mask_data, mask_labels = load_images_from_folder(WITH_MASK_DIR, 1)
    
    print("\nLoading no-mask images...")
    no_mask_data, no_mask_labels = load_images_from_folder(WITHOUT_MASK_DIR, 0)
    
    # Combine datasets
    print("\nCombining datasets...")
    data = np.concatenate([mask_data, no_mask_data], axis=0)
    labels = np.concatenate([mask_labels, no_mask_labels], axis=0)
    
    # Free some memory
    del mask_data, mask_labels, no_mask_data, no_mask_labels
    import gc
    gc.collect()
    
    print(f"\nTotal images loaded: {len(data)}")
    print(f"Images with mask: {sum(labels)}")
    print(f"Images without mask: {len(labels) - sum(labels)}")
    
    # Split dataset
    print("\nSplitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )
    
    # Free more memory
    del data, labels
    gc.collect()
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # Train and evaluate different configurations
    results = []
    for config_num in range(1, 4):
        print(f"\nTraining Configuration {config_num}...")
        
        # Create and train model
        model = create_cnn_model(config_num)
        
        # Save model summary
        with open(os.path.join(OUTPUT_DIR, f'model_summary_config_{config_num}.txt'), 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1
        )
        
        # Plot training history
        plot_training_history(history, config_num)
        
        # Evaluate model
        val_loss, val_acc = model.evaluate(X_test, y_test)
        print(f"\nConfiguration {config_num} Validation Accuracy: {val_acc * 100:.2f}%")
        
        # Save results
        config_results = {
            'config': config_num,
            'validation_accuracy': float(val_acc),
            'validation_loss': float(val_loss),
            'parameters': {
                'img_size': IMG_SIZE,
                'epochs': EPOCHS,
                'batch_size': BATCH_SIZE
            }
        }
        
        with open(os.path.join(OUTPUT_DIR, f'results_config_{config_num}.json'), 'w') as f:
            import json
            json.dump(config_results, f, indent=4)
        
        # Save model
        model.save(os.path.join(OUTPUT_DIR, f'model_config_{config_num}.h5'))
        
        results.append(config_results)
        
        # Clear memory
        del model
        gc.collect()
    
    # Plot configuration comparison
    plot_configuration_comparison(results)
    
    # Save overall results
    with open(os.path.join(OUTPUT_DIR, 'overall_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("Training complete!")

if __name__ == "__main__":
    main() 