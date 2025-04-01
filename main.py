"""
Main script for the FreshHarvest project.
Entry point for the entire project.
"""

import argparse
import time
import os
import sys
import subprocess
from pathlib import Path

# Configure paths based on your specific system layout
PROJECT_ROOT = Path("C:/FreshHarvest")
DATA_PATH = PROJECT_ROOT / "data" / "FreshHarvest_Dataset" / "FRUIT-16K"
OUTPUT_DIR = PROJECT_ROOT / "model_outputs"

# Import custom modules
try:
    from config import DEVICE, set_seed
    from utils import setup_logger
    from data import get_data_loaders, visualize_dataset_samples
    from model import FruitFreshnessClassifier, get_model_summary
    from trainer import train_model, evaluate_model
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Make sure you're running this script from the project root directory")
    sys.exit(1)

# Set up logger
logger = setup_logger("main", "main.log")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="FreshHarvest - Fruit Freshness Detection")

    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'evaluate', 'predict', 'serve'],
                        help='Operation mode')

    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to saved model (for evaluate/predict/serve modes)')

    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs for training')

    parser.add_argument('--visualize', action='store_true',
                        help='Visualize dataset samples')

    parser.add_argument('--input', type=str, default=None,
                        help='Input image or directory for prediction')

    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory for results')

    parser.add_argument('--data-path', type=str, default=str(DATA_PATH),
                        help=f'Path to dataset (default: {DATA_PATH})')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    parser.add_argument('--save-annotated', action='store_true',
                        help='Save annotated images with predictions')

    return parser.parse_args()


def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        import cv2
        from PIL import Image
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info("Please install required packages with:")
        logger.info(
            "pip install torch torchvision numpy pandas matplotlib opencv-python pillow tqdm seaborn scikit-learn")
        return False


def train(args):
    """Train the model"""
    logger.info("Starting training mode")

    # Update config with custom data path if provided
    if args.data_path and args.data_path != str(DATA_PATH):
        from config import config
        config['dataset_path'] = args.data_path
        logger.info(f"Using custom dataset path: {args.data_path}")

    # Get data loaders
    train_loader, val_loader, test_loader, class_names = get_data_loaders()

    # Visualize dataset samples if requested
    if args.visualize:
        visualize_dataset_samples(train_loader, class_names)

    # Create model
    model = FruitFreshnessClassifier()
    logger.info(get_model_summary(model))

    # Train model
    start_time = time.time()
    logger.info(f"Training for {args.epochs} epochs...")

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs
    )

    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time // 60:.0f}m {training_time % 60:.0f}s")

    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "final_model.pt"
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")

    # Evaluate model on test set
    logger.info("Evaluating model on test set...")
    metrics = evaluate_model(model, test_loader, class_names)

    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Test Precision: {metrics['precision']:.4f}")
    logger.info(f"Test Recall: {metrics['recall']:.4f}")
    logger.info(f"Test F1 Score: {metrics['f1']:.4f}")

    logger.info("Training mode completed")


def evaluate(args):
    """Evaluate the model"""
    logger.info("Starting evaluation mode")

    if not args.model_path:
        logger.error("Model path not provided")
        return

    # Load model
    logger.info(f"Loading model from {args.model_path}")
    try:
        model = FruitFreshnessClassifier.load(args.model_path)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    # Update config with custom data path if provided
    if args.data_path and args.data_path != str(DATA_PATH):
        from config import config
        config['dataset_path'] = args.data_path
        logger.info(f"Using custom dataset path: {args.data_path}")

    # Get data loaders
    _, _, test_loader, class_names = get_data_loaders()

    # Evaluate model
    logger.info("Evaluating model on test set...")
    metrics = evaluate_model(model, test_loader, class_names)

    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Test Precision: {metrics['precision']:.4f}")
    logger.info(f"Test Recall: {metrics['recall']:.4f}")
    logger.info(f"Test F1 Score: {metrics['f1']:.4f}")

    logger.info("Evaluation mode completed")


def predict(args):
    """Make predictions using the model"""
    logger.info("Starting prediction mode")

    if not args.model_path:
        logger.error("Model path not provided")
        return

    if not args.input:
        logger.error("Input path not provided")
        return

    # Check if input path exists - if not, try to fix relative paths
    input_path = Path(args.input)
    if not input_path.exists():
        # Check if it might be relative to the project root or data directory
        potential_paths = [
            PROJECT_ROOT / args.input,
            DATA_PATH / args.input,
            DATA_PATH / "F_Banana" / args.input,  # Try common subdirectories
            DATA_PATH / "S_Banana" / args.input,
        ]

        for path in potential_paths:
            if path.exists():
                args.input = str(path)
                logger.info(f"Found image at {args.input}")
                break
        else:
            logger.error(f"Input path does not exist: {args.input}")
            logger.info(f"Please provide a valid path to an image or directory.")
            logger.info(f"Your data is located at: {DATA_PATH}")
            # List some example files from the dataset
            try:
                some_files = list(DATA_PATH.glob('*//*'))[:5]  # Get a few example files
                if some_files:
                    logger.info("Example files from your dataset:")
                    for file in some_files:
                        logger.info(f"  - {file}")
            except Exception:
                pass
            return

    # Check dependencies
    if not check_dependencies():
        return

    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run the predict.py script
    cmd = [
        sys.executable, "predict.py",
        "--model-path", args.model_path,
        "--input", args.input,
        "--output-dir", str(output_dir)
    ]

    if args.visualize:
        cmd.append("--visualize")

    if args.save_annotated:
        cmd.append("--save-annotated")

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        # Run the prediction script
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Stream output in real-time
        for line in process.stdout:
            print(line, end='')

        # Wait for the process to complete
        process.wait()

        # Check for errors
        if process.returncode != 0:
            for line in process.stderr:
                print(line, end='')
            logger.error(f"Prediction script exited with error code {process.returncode}")
        else:
            logger.info("Prediction completed successfully")
    except Exception as e:
        logger.error(f"Error running prediction script: {e}")

    logger.info("Prediction mode completed")


def serve(args):
    """Start the web server for model serving"""
    logger.info("Starting server mode")

    if not args.model_path:
        logger.error("Model path not provided")
        return

    # Check if model exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model not found at {args.model_path}")
        return

    # Check for required packages
    try:
        import streamlit
    except ImportError:
        logger.error("Streamlit not found. Installing...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"], check=True)
            logger.info("Streamlit installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install streamlit: {e}")
            logger.info("Please install streamlit manually: pip install streamlit")
            return

    # Update the model path in app.py
    app_path = Path("app_streamlit.py")
    if not app_path.exists():
        logger.error(f"App file not found: {app_path}")
        return

    try:
        with open(app_path, "r") as f:
            app_py = f.read()

        # Replace the model path
        app_py = app_py.replace('DEFAULT_MODEL_PATH = "model_outputs/final_model.pt"',
                                f'DEFAULT_MODEL_PATH = "{args.model_path}"')
        app_py = app_py.replace('DEFAULT_DATA_PATH = "data/FreshHarvest_Dataset/FRUIT-16K"',
                                f'DEFAULT_DATA_PATH = "{DATA_PATH}"')

        with open(app_path, "w") as f:
            f.write(app_py)

        logger.info(f"Updated app configuration with model path: {args.model_path}")
    except Exception as e:
        logger.error(f"Error updating app configuration: {e}")

    # Run the streamlit app
    logger.info(f"Starting web server with model {args.model_path}")
    logger.info("Access the web interface at http://localhost:8501")

    try:
        subprocess.run([
            "streamlit", "run", str(app_path),
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except Exception as e:
        logger.error(f"Error running streamlit app: {e}")

    logger.info("Server mode completed")


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Run the selected mode
    if args.mode == 'train':
        train(args)
    elif args.mode == 'evaluate':
        evaluate(args)
    elif args.mode == 'predict':
        predict(args)
    elif args.mode == 'serve':
        serve(args)
    else:
        logger.error(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()