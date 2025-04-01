"""
Data module for the FreshHarvest project.
Contains dataset class, data loading, and augmentation functionality.
"""

import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision.utils
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from config import (
    DATASET_PATH, BATCH_SIZE, NUM_WORKERS, TRAIN_SPLIT, VAL_SPLIT,
    IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, FOLDER_TO_CLASS
)


# Define transforms
def get_transforms(train=True):
    """
    Get data transforms for training or validation/testing

    Args:
        train (bool): Whether to include training augmentations

    Returns:
        transforms.Compose: Composition of transforms
    """
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


class FruitDataset(Dataset):
    """Dataset class for the FreshHarvest fruit dataset"""

    def __init__(self, data_frame, transform=None):
        """
        Initialize the dataset

        Args:
            data_frame (pd.DataFrame): DataFrame with image paths and labels
            transform (callable, optional): Transform to apply to the images
        """
        self.data_frame = data_frame
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Get image path and label
        img_path = self.data_frame.iloc[idx]['image_path']
        label = self.data_frame.iloc[idx]['label']

        # Load and transform image
        try:
            img = Image.open(img_path).convert('RGB')

            if self.transform:
                img = self.transform(img)

            return img, label

        except Exception as e:
            # Handle corrupt images by returning a black image
            print(f"Error loading image {img_path}: {e}")
            if self.transform:
                dummy_img = torch.zeros(3, IMG_SIZE, IMG_SIZE)
            else:
                dummy_img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='black')

            return dummy_img, label


def create_dataset_dataframe(dataset_path=DATASET_PATH):
    """
    Create a DataFrame with image paths and labels

    Args:
        dataset_path (Path or str): Path to the dataset directory

    Returns:
        pd.DataFrame: DataFrame with image paths and labels
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    # Initialize lists for image paths and labels
    image_paths = []
    labels = []
    fruit_types = []
    freshness_labels = []

    # Valid image extensions
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    # Scan all directories
    for folder in dataset_path.iterdir():
        if folder.is_dir():
            folder_name = folder.name

            # Determine class (fresh or spoiled) from folder prefix
            if folder_name.startswith("F_"):
                freshness = "fresh"
                freshness_label = 0
                fruit_type = folder_name[2:]  # Remove F_ prefix
            elif folder_name.startswith("S_"):
                freshness = "spoiled"
                freshness_label = 1
                fruit_type = folder_name[2:]  # Remove S_ prefix
            else:
                # Skip folders not following the naming convention
                continue

            # Process all images in the folder
            for img_path in folder.glob("**/*"):
                # Check if file is an image
                if img_path.suffix.lower() in valid_extensions:
                    image_paths.append(str(img_path))
                    labels.append(freshness_label)
                    fruit_types.append(fruit_type)
                    freshness_labels.append(freshness)

    # Create DataFrame
    df = pd.DataFrame({
        'image_path': image_paths,
        'label': labels,
        'fruit_type': fruit_types,
        'freshness': freshness_labels
    })

    return df


def split_dataset(df, train_split=TRAIN_SPLIT, val_split=VAL_SPLIT, stratify=True):
    """
    Split the dataset into train, validation, and test sets

    Args:
        df (pd.DataFrame): DataFrame with image paths and labels
        train_split (float): Proportion of data for training
        val_split (float): Proportion of data for validation
        stratify (bool): Whether to stratify the split by label

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Calculate test_split
    test_split = 1.0 - train_split - val_split

    if stratify:
        stratify_col = df['label']
    else:
        stratify_col = None

    # First split: separate train from the rest
    train_df, temp_df = train_test_split(
        df,
        train_size=train_split,
        random_state=42,
        stratify=stratify_col
    )

    if stratify:
        stratify_col = temp_df['label']

    # Second split: separate validation and test from the rest
    # Calculate the relative size for validation from the remaining data
    relative_val_size = val_split / (val_split + test_split)

    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_size,
        random_state=42,
        stratify=stratify_col
    )

    return train_df, val_df, test_df


def get_data_loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """
    Create data loaders for training, validation, and testing

    Args:
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of workers for dataloaders

    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names)
    """
    # Create dataset DataFrame
    df = create_dataset_dataframe()

    # Split dataset
    train_df, val_df, test_df = split_dataset(df)

    # Print dataset statistics
    print(f"Total images: {len(df)}")
    print(f"Training: {len(train_df)} images")
    print(f"Validation: {len(val_df)} images")
    print(f"Testing: {len(test_df)} images")

    # Get class distribution
    class_dist = df['freshness'].value_counts()
    print(f"Class distribution: {class_dist.to_dict()}")

    # Get fruit type distribution
    fruit_dist = df['fruit_type'].value_counts()
    print(f"Fruit type distribution: {fruit_dist.to_dict()}")

    # Create datasets
    train_dataset = FruitDataset(train_df, transform=get_transforms(train=True))
    val_dataset = FruitDataset(val_df, transform=get_transforms(train=False))
    test_dataset = FruitDataset(test_df, transform=get_transforms(train=False))

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Get class names (sorted by label)
    class_names = df['freshness'].unique().tolist()
    class_names.sort(key=lambda x: 0 if x == "fresh" else 1)

    return train_loader, val_loader, test_loader, class_names


def visualize_dataset_samples(dataloader, class_names, num_samples=8):
    """
    Visualize samples from the dataset

    Args:
        dataloader (DataLoader): DataLoader for the dataset
        class_names (list): List of class names
        num_samples (int): Number of samples to visualize
    """
    # Get a batch of samples
    images, labels = next(iter(dataloader))

    # Limit the number of samples to visualize
    images = images[:num_samples]
    labels = labels[:num_samples]

    # Create a grid of images
    grid = torchvision.utils.make_grid(images)

    # Denormalize images for visualization
    grid = grid.permute(1, 2, 0).numpy()
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    grid = std * grid + mean
    grid = np.clip(grid, 0, 1)

    # Plot the grid
    plt.figure(figsize=(12, 6))
    plt.imshow(grid)

    # Add class labels as titles
    class_names_list = [class_names[label.item()] for label in labels]
    plt.title(f"Samples: {', '.join(class_names_list)}")

    plt.axis('off')
    plt.tight_layout()
    plt.show()


def get_dataset_statistics():
    """
    Get statistics for the dataset (useful for normalization)

    Returns:
        tuple: (mean, std) calculated on the entire dataset
    """
    # Create dataset DataFrame
    df = create_dataset_dataframe()

    # Create a dataset with minimal transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    dataset = FruitDataset(df, transform=transform)

    # Sample a subset for efficiency
    sample_size = min(1000, len(dataset))
    indices = torch.randperm(len(dataset))[:sample_size]

    # Calculate mean and std
    mean = torch.zeros(3)
    std = torch.zeros(3)

    for idx in indices:
        img, _ = dataset[idx]
        mean += img.mean([1, 2])
        std += img.std([1, 2])

    mean /= sample_size
    std /= sample_size

    return mean.tolist(), std.tolist()