"""
Utility functions for the FreshHarvest project.
Contains helper functions for seeding, visualization, and logging.
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import logging
from datetime import datetime

from config import LOGS_DIR, SEED

def set_seed(seed=SEED):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return

def setup_logger(name, log_file=None, level=logging.INFO):
    """Set up logger with specified name and file"""
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        log_path = LOGS_DIR / log_file
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_timestamp_str():
    """Get current timestamp as string for filenames"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def imshow(img, title=None, mean=None, std=None):
    """Display image with optional denormalization"""
    # Clone the image to avoid modifying the original
    img = img.clone()
    
    # Denormalize if mean and std are provided
    if mean is not None and std is not None:
        mean = torch.tensor(mean).view(3, 1, 1)
        std = torch.tensor(std).view(3, 1, 1)
        img = img * std + mean
    
    # Convert tensor to numpy array for matplotlib
    img = img.cpu().numpy().transpose((1, 2, 0))
    
    # Clip values to valid range [0, 1]
    img = np.clip(img, 0, 1)
    
    # Display image
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def visualize_batch(images, labels, class_names, mean=None, std=None, max_images=16, figsize=(15, 10)):
    """Visualize a batch of images with their labels"""
    # Calculate grid size
    num_images = min(max_images, images.size(0))
    num_rows = (num_images + 3) // 4  # Ceiling division
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot each image
    for i in range(num_images):
        plt.subplot(num_rows, 4, i + 1)
        imshow(images[i], title=class_names[labels[i]], mean=mean, std=std)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, figsize=(10, 8)):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
    return cm

def plot_training_history(history, figsize=(12, 4)):
    """Plot training history (loss and accuracy curves)"""
    plt.figure(figsize=figsize)
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def print_classification_report(y_true, y_pred, class_names):
    """Print classification report with class names"""
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))