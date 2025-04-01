"""
Evaluation script for the FreshHarvest project.
Run this script to evaluate a trained model on the test set or custom images.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from PIL import Image

from config import DEVICE, OUTPUT_DIR
from utils import set_seed, setup_logger, plot_confusion_matrix, print_classification_report
from data import get_data_loaders, get_transforms
from model import FruitFreshnessClassifier
from trainer import predict, evaluate_model

# Set up logger
logger = setup_logger("evaluate", "evaluate.log")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained FreshHarvest model")
    
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the trained model")
    
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save evaluation results (default: same as model directory)")
    
    parser.add_argument("--image-path", type=str, default=None,
                        help="Path to a single image for prediction")
    
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Directory of images for batch prediction")
    
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    return parser.parse_args()

def predict_single_image(model, image_path, class_names, device=DEVICE):
    """
    Make a prediction on a single image
    
    Args:
        model (nn.Module): Model to use for prediction
        image_path (str or Path): Path to the image
        class_names (list): List of class names
        device (torch.device): Device to use for prediction
        
    Returns:
        dict: Prediction results
    """
    # Load and transform image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None
    
    # Apply transformation
    transform = get_transforms(train=False)
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
    
    # Get prediction details
    pred_idx = preds.item()
    pred_class = class_names[pred_idx]
    confidence = probs[0][pred_idx].item()
    
    # Create visualization
    plt.figure(figsize=(6, 6))
    
    # Show image
    plt.imshow(image)
    
    # Add prediction info as title
    title = f"Prediction: {pred_class} (Confidence: {confidence:.2f})"
    plt.title(title)
    
    plt.axis('off')
    
    # Return results
    return {
        'image_path': str(image_path),
        'predicted_class': pred_class,
        'confidence': confidence,
        'probabilities': probs[0].cpu().numpy().tolist()
    }

def predict_batch_images(model, image_dir, class_names, device=DEVICE):
    """
    Make predictions on a batch of images
    
    Args:
        model (nn.Module): Model to use for predictions
        image_dir (str or Path): Directory containing images
        class_names (list): List of class names
        device (torch.device): Device to use for predictions
        
    Returns:
        list: List of prediction results
    """
    image_dir = Path(image_dir)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(list(image_dir.glob(f"*{ext}")))
        image_paths.extend(list(image_dir.glob(f"*{ext.upper()}")))
    
    if not image_paths:
        logger.error(f"No image files found in {image_dir}")
        return []
    
    logger.info(f"Found {len(image_paths)} images in {image_dir}")
    
    # Make predictions
    results = []
    
    for image_path in image_paths:
        result = predict_single_image(model, image_path, class_names, device)
        if result:
            results.append(result)
    
    return results

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Determine output directory
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir) if args.output_dir else model_path.parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    try:
        model = FruitFreshnessClassifier.load(model_path)
        model.eval()
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Get data loaders and class names
    _, _, test_loader, class_names = get_data_loaders()
    
    # If a single image is provided, make a prediction
    if args.image_path:
        logger.info(f"Making prediction on single image: {args.image_path}")
        result = predict_single_image(model, args.image_path, class_names)
        
        if result:
            logger.info(f"Prediction: {result['predicted_class']} (Confidence: {result['confidence']:.2f})")
            
            # Save visualization
            vis_path = output_dir / f"prediction_{Path(args.image_path).stem}.png"
            plt.savefig(vis_path)
            logger.info(f"Visualization saved to {vis_path}")
            
            # Save result as JSON
            result_path = output_dir / f"prediction_{Path(args.image_path).stem}.json"
            with open(result_path, "w") as f:
                json.dump(result, f, indent=4)
            logger.info(f"Prediction result saved to {result_path}")
        
    # If an image directory is provided, make batch predictions
    elif args.image_dir:
        logger.info(f"Making predictions on images in directory: {args.image_dir}")
        results = predict_batch_images(model, args.image_dir, class_names)
        
        if results:
            # Save results as JSON
            results_path = output_dir / "batch_predictions.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)
            logger.info(f"Batch prediction results saved to {results_path}")
            
            # Calculate statistics
            predictions = [r['predicted_class'] for r in results]
            pred_counts = {c: predictions.count(c) for c in class_names}
            
            # Create visualization of prediction distribution
            plt.figure(figsize=(10, 6))
            plt.bar(pred_counts.keys(), pred_counts.values())
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.title('Prediction Distribution')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save visualization
            vis_path = output_dir / "prediction_distribution.png"
            plt.savefig(vis_path)
            logger.info(f"Prediction distribution visualization saved to {vis_path}")
    
    # Otherwise, evaluate on the test set
    else:
        logger.info("Evaluating model on test set")
        metrics = evaluate_model(model, test_loader, class_names)
        
        # Save evaluation metrics
        metrics_path = output_dir / "test_metrics.json"
        with open(metrics_path, "w") as f:
            # Remove non-serializable items
            serializable_metrics = {k: v for k, v in metrics.items() if k not in ['confusion_matrix', 'y_pred', 'y_true', 'y_prob']}
            json.dump(serializable_metrics, f, indent=4)
        
        logger.info(f"Test metrics saved to {metrics_path}")
        
        # Save confusion matrix
        cm_path = output_dir / "confusion_matrix.png"
        plt.savefig(cm_path)
        logger.info(f"Confusion matrix saved to {cm_path}")
    
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main()