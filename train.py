"""
Main training script for the FreshHarvest project.
Run this script to train the model on the FreshHarvest dataset.
"""

import argparse
import torch
import json
import matplotlib.pyplot as plt
from pathlib import Path

from config import (
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, MODEL_CONFIG, DEVICE, OUTPUT_DIR
)
from utils import set_seed, setup_logger, get_timestamp_str, plot_training_history
from data import get_data_loaders, visualize_dataset_samples
from model import FruitFreshnessClassifier, get_model_summary
from trainer import train_model, evaluate_model

# Set up logger
logger = setup_logger("train", "train.log")

def parse_args():
    parser = argparse.ArgumentParser(description="Train the FreshHarvest model")
    
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help=f"Number of epochs (default: {NUM_EPOCHS})")
    
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help=f"Learning rate (default: {LEARNING_RATE})")
    
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help=f"Weight decay (default: {WEIGHT_DECAY})")
    
    parser.add_argument("--architecture", type=str, default=MODEL_CONFIG["architecture"],
                        choices=["efficientnet_b0", "resnet50", "mobilenet_v2"],
                        help=f"Model architecture (default: {MODEL_CONFIG['architecture']})")
    
    parser.add_argument("--pretrained", action="store_true", default=MODEL_CONFIG["pretrained"],
                        help="Use pretrained model")
    
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained",
                        help="Don't use pretrained model")
    
    parser.add_argument("--freeze-backbone", action="store_true", default=MODEL_CONFIG["freeze_backbone"],
                        help="Freeze backbone layers")
    
    parser.add_argument("--no-freeze-backbone", action="store_false", dest="freeze_backbone",
                        help="Don't freeze backbone layers")
    
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize dataset samples before training")
    
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    logger.info(f"Set random seed to {args.seed}")
    
    # Update model config based on arguments
    MODEL_CONFIG["architecture"] = args.architecture
    MODEL_CONFIG["pretrained"] = args.pretrained
    MODEL_CONFIG["freeze_backbone"] = args.freeze_backbone
    
    # Create output directory with timestamp
    timestamp = get_timestamp_str()
    run_dir = OUTPUT_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = {
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "model_config": MODEL_CONFIG,
        "device": str(DEVICE),
        "seed": args.seed
    }
    
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    logger.info(f"Configuration saved to {run_dir}/config.json")
    
    # Get data loaders
    logger.info("Loading dataset...")
    train_loader, val_loader, test_loader, class_names = get_data_loaders()
    
    # Visualize dataset samples if requested
    if args.visualize:
        logger.info("Visualizing dataset samples...")
        visualize_dataset_samples(train_loader, class_names)
    
    # Create model
    logger.info(f"Creating model (architecture: {args.architecture})...")
    model = FruitFreshnessClassifier(num_classes=len(class_names), architecture=args.architecture)
    
    # Log model summary
    model_summary = get_model_summary(model)
    logger.info(f"Model summary:\n{model_summary}")
    
    # Train model
    logger.info(f"Starting training for {args.epochs} epochs...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Save trained model
    model_path = run_dir / "final_model.pt"
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save training history
    history_path = run_dir / "history.json"
    with open(history_path, "w") as f:
        # Convert tensors to Python floats for JSON serialization
        serializable_history = {}
        for key, value in history.items():
            if isinstance(value, list) and all(isinstance(x, torch.Tensor) for x in value):
                serializable_history[key] = [float(x) for x in value]
            else:
                serializable_history[key] = value
                
        json.dump(serializable_history, f, indent=4)
    
    logger.info(f"Training history saved to {history_path}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plot_training_history(history)
    history_plot_path = run_dir / "training_history.png"
    plt.savefig(history_plot_path)
    logger.info(f"Training history plot saved to {history_plot_path}")
    
    # Evaluate model on test set
    logger.info("Evaluating model on test set...")
    metrics = evaluate_model(model, test_loader, class_names)
    
    # Save evaluation metrics
    metrics_path = run_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        # Remove non-serializable items
        serializable_metrics = {k: v for k, v in metrics.items() if k not in ['confusion_matrix', 'y_pred', 'y_true', 'y_prob']}
        json.dump(serializable_metrics, f, indent=4)
    
    logger.info(f"Test metrics saved to {metrics_path}")
    
    # Save confusion matrix
    cm_path = run_dir / "confusion_matrix.png"
    plt.savefig(cm_path)
    logger.info(f"Confusion matrix saved to {cm_path}")
    
    logger.info("Training and evaluation complete!")

if __name__ == "__main__":
    main()