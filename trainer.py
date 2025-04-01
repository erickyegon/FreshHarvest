"""
Training module for the FreshHarvest project.
Contains training, validation, and testing functionality.
"""

import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import os

from config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, OUTPUT_DIR
)
from utils import (
    get_timestamp_str, plot_training_history,
    plot_confusion_matrix, print_classification_report, setup_logger
)

# Set up logger
logger = setup_logger("trainer", "trainer.log")


class EarlyStopping:
    """Early stopping class to prevent overfitting"""

    def __init__(self, patience=EARLY_STOPPING_PATIENCE, min_delta=0, verbose=True):
        """
        Initialize early stopping

        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement
            verbose (bool): If True, prints a message when early stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf  # Changed from np.Inf to np.inf

    def __call__(self, val_loss, model, path):
        """
        Check if validation loss has improved and save model if it has

        Args:
            val_loss (float): Validation loss
            model (nn.Module): Model to save
            path (str or Path): Path to save model to
        """
        score = -val_loss

        if self.best_score is None:
            # First epoch
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.min_delta:
            # Validation loss has increased
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Validation loss has decreased
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        """
        Save model checkpoint when validation loss decreases

        Args:
            val_loss (float): Validation loss
            model (nn.Module): Model to save
            path (str or Path): Path to save model to
        """
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...")
        model.save(path)
        self.val_loss_min = val_loss


def train_one_epoch(model, dataloader, criterion, optimizer, device=DEVICE):
    """
    Train the model for one epoch

    Args:
        model (nn.Module): Model to train
        dataloader (DataLoader): DataLoader for training data
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to use for training

    Returns:
        tuple: (epoch_loss, epoch_acc)
    """
    # Set model to training mode
    model.train()

    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    # Iterate over training data
    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        # Move inputs and labels to device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        # Update statistics
        running_loss += loss.item() * inputs.size(0)
        correct_preds += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)

    # Calculate epoch loss and accuracy
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_preds.double() / total_samples

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device=DEVICE):
    """
    Validate the model on the validation set

    Args:
        model (nn.Module): Model to validate
        dataloader (DataLoader): DataLoader for validation data
        criterion (nn.Module): Loss function
        device (torch.device): Device to use for validation

    Returns:
        tuple: (val_loss, val_acc)
    """
    # Set model to evaluation mode
    model.eval()

    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    # Iterate over validation data
    for inputs, labels in tqdm(dataloader, desc="Validation", leave=False):
        # Move inputs and labels to device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass with no gradient calculation
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        # Update statistics
        running_loss += loss.item() * inputs.size(0)
        correct_preds += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)

    # Calculate validation loss and accuracy
    val_loss = running_loss / total_samples
    val_acc = correct_preds.double() / total_samples

    return val_loss, val_acc


def predict(model, dataloader, device=DEVICE):
    """
    Make predictions on a dataset

    Args:
        model (nn.Module): Model to use for predictions
        dataloader (DataLoader): DataLoader for data to predict on
        device (torch.device): Device to use for predictions

    Returns:
        tuple: (all_preds, all_labels, all_probs)
    """
    # Set model to evaluation mode
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    # Iterate over data
    for inputs, labels in tqdm(dataloader, desc="Predicting"):
        # Move inputs to device
        inputs = inputs.to(device)

        # Forward pass with no gradient calculation
        with torch.no_grad():
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

        # Gather predictions and labels
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS,
                lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, device=DEVICE):
    """
    Train the model

    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        num_epochs (int): Number of epochs to train for
        lr (float): Learning rate
        weight_decay (float): Weight decay
        device (torch.device): Device to use for training

    Returns:
        tuple: (model, history)
    """
    # Initialize criterion, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # Create model checkpoint directory
    timestamp = get_timestamp_str()
    checkpoint_dir = OUTPUT_DIR / f"checkpoint_{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_model.pt"

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, verbose=True)

    # Initialize history dictionary
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'lr': []
    }

    # Initialize best model weights and best accuracy
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Track training time
    start_time = time.time()

    # Train for the specified number of epochs
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")

        # Train one epoch and validate
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_loss)

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc.item())
        history['val_acc'].append(val_acc.item())
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Log metrics
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Check if this is the best model so far
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())

        # Early stopping check
        early_stopping(val_loss, model, checkpoint_path)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break

    # Calculate total training time
    total_time = time.time() - start_time
    logger.info(f"Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")
    logger.info(f"Best validation accuracy: {best_acc:.4f}")

    # Load best model weights
    model.load_state_dict(best_model_weights)

    return model, history


def evaluate_model(model, test_loader, class_names, device=DEVICE):
    """
    Evaluate the model on the test set

    Args:
        model (nn.Module): Model to evaluate
        test_loader (DataLoader): DataLoader for test data
        class_names (list): List of class names
        device (torch.device): Device to use for evaluation

    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Get predictions
    y_pred, y_true, y_prob = predict(model, test_loader, device)

    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Log metrics
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test Precision: {precision:.4f}")
    logger.info(f"Test Recall: {recall:.4f}")
    logger.info(f"Test F1 Score: {f1:.4f}")

    # Plot confusion matrix
    cm = plot_confusion_matrix(y_true, y_pred, class_names)

    # Print classification report
    print_classification_report(y_true, y_pred, class_names)

    # Return metrics
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_true': y_true,
        'y_prob': y_prob
    }


# Additional utilities for more advanced training scenarios

def train_with_mixup(model, train_loader, criterion, optimizer, alpha=1.0, device=DEVICE):
    """
    Train one epoch with mixup data augmentation

    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): DataLoader for training data
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        alpha (float): Mixup alpha parameter
        device (torch.device): Device to use for training

    Returns:
        tuple: (epoch_loss, epoch_acc)
    """
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    for inputs, labels in tqdm(train_loader, desc="Training (Mixup)", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        # Generate mixup parameters
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
        index = torch.randperm(inputs.size(0)).to(device)

        # Create mixed inputs and targets
        mixed_inputs = lam * inputs + (1 - lam) * inputs[index]

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(mixed_inputs)

        # Mixup loss
        loss = lam * criterion(outputs, labels) + (1 - lam) * criterion(outputs, labels[index])

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # For accuracy calculation (using original labels)
        _, preds = torch.max(outputs, 1)

        # Update statistics
        running_loss += loss.item() * inputs.size(0)
        correct_preds += (lam * torch.sum(preds == labels) +
                          (1 - lam) * torch.sum(preds == labels[index]))
        total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_preds.double() / total_samples

    return epoch_loss, epoch_acc


def lr_finder(model, train_loader, criterion, optimizer, start_lr=1e-7, end_lr=10, num_iterations=100, device=DEVICE):
    """
    Learning rate finder to help determine the optimal learning rate

    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): DataLoader for training data
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        start_lr (float): Starting learning rate
        end_lr (float): Ending learning rate
        num_iterations (int): Number of iterations for the search
        device (torch.device): Device to use for training

    Returns:
        tuple: (lr_list, loss_list) - Lists of learning rates and corresponding losses
    """
    # Save original model state
    original_state = copy.deepcopy(model.state_dict())

    # Set model to training mode
    model.train()

    # Initialize learning rate and lists to store results
    lr_list = []
    loss_list = []
    best_loss = float('inf')

    # Reset optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr

    # Calculate learning rate multiplier
    lr_multiplier = (end_lr / start_lr) ** (1 / num_iterations)

    # Training loop
    iterator = iter(train_loader)
    for iteration in tqdm(range(num_iterations), desc="LR Finder"):
        try:
            inputs, labels = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            inputs, labels = next(iterator)

        inputs, labels = inputs.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Check if loss is exploding
        if torch.isnan(loss) or torch.isinf(loss) or (loss.item() > 4 * best_loss):
            break

        # Update best loss
        if loss.item() < best_loss:
            best_loss = loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

        # Store current learning rate and loss
        current_lr = optimizer.param_groups[0]['lr']
        lr_list.append(current_lr)
        loss_list.append(loss.item())

        # Increase learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_multiplier

    # Restore original model state
    model.load_state_dict(original_state)

    # Plot learning rate vs loss
    plt.figure(figsize=(10, 6))
    plt.plot(lr_list, loss_list)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.grid(True)
    plt.show()

    return lr_list, loss_list