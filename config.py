"""
Configuration module for the FreshHarvest project.
Contains all configurable parameters for training and evaluation.
"""

import os
import torch
from pathlib import Path

# Fix for OpenMP multiple runtime error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Basic config
SEED = 42
BASE_DIR = Path("C:/FreshHarvest")
DATASET_PATH = BASE_DIR / "data" / "FreshHarvest_Dataset" / "FRUIT-16K"
OUTPUT_DIR = BASE_DIR / "model_outputs"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
LOGS_DIR.mkdir(exist_ok=True, parents=True)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4  # Number of workers for data loading

# Training hyperparameters
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15  # Calculated automatically
NUM_EPOCHS = 5
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 5

# Data normalization (ImageNet stats for pre-trained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Class mapping
FRESHNESS_CLASSES = {
    "fresh": 0,
    "spoiled": 1
}

# Mapping from folder prefixes to class names
FOLDER_TO_CLASS = {
    "F_": "fresh",
    "S_": "spoiled"
}

def get_fruit_types():
    """Get list of fruit types from the dataset directory"""
    if not DATASET_PATH.exists():
        return []
    
    fruit_types = set()
    for folder in DATASET_PATH.iterdir():
        if folder.is_dir():
            # Extract fruit name (remove F_ or S_ prefix)
            fruit_name = folder.name[2:]
            fruit_types.add(fruit_name)
    
    return sorted(list(fruit_types))

# Model configuration
MODEL_CONFIG = {
    "architecture": "efficientnet_b0",  # Options: efficientnet_b0, resnet50, mobilenet_v2
    "pretrained": True,
    "freeze_backbone": False,
    "dropout_rate": 0.2
}

print(f"Using device: {DEVICE}")