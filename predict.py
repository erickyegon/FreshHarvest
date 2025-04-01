"""
Prediction script for the FreshHarvest project.
Provides a simple interface for making predictions on new images.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import json
import cv2
from PIL import Image
from tqdm import tqdm

from config import DEVICE, IMAGENET_MEAN, IMAGENET_STD
from utils import set_seed, imshow
from data import get_transforms
from model import FruitFreshnessClassifier

def parse_args():
    parser = argparse.ArgumentParser(description="Make predictions with a trained FreshHarvest model")
    
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the trained model")
    
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input image or directory")
    
    parser.add_argument("--output-dir", type=str, default="predictions",
                        help="Directory to save predictions (default: predictions)")
    
    parser.add_argument("--visualize", action="store_true",
                        help="Create visualizations of predictions")
    
    parser.add_argument("--save-annotated", action="store_true",
                        help="Save annotated images with predictions")
    
    parser.add_argument("--class-names", type=str, default="fresh,spoiled",
                        help="Comma-separated list of class names (default: fresh,spoiled)")
    
    return parser.parse_args()

def predict_image(model, image_path, transform, class_names, device=DEVICE):
    """
    Make a prediction on a single image
    
    Args:
        model (nn.Module): Model to use for prediction
        image_path (str or Path): Path to the image
        transform (callable): Transform to apply to the image
        class_names (list): List of class names
        device (torch.device): Device to use for prediction
        
    Returns:
        dict: Prediction results
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Transform image
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
    
    # Return result
    return {
        'image_path': str(image_path),
        'predicted_class': pred_class,
        'predicted_idx': pred_idx,
        'confidence': confidence,
        'probabilities': probs[0].cpu().numpy().tolist(),
        'image': image
    }

def create_visualization(result, class_names, output_path=None):
    """
    Create a visualization of a prediction
    
    Args:
        result (dict): Prediction result
        class_names (list): List of class names
        output_path (str or Path, optional): Path to save visualization
        
    Returns:
        None
    """
    # Extract data
    image = result['image']
    pred_class = result['predicted_class']
    confidence = result['confidence']
    
    # Create figure
    plt.figure(figsize=(6, 6))
    
    # Show image
    plt.imshow(image)
    
    # Set title
    plt.title(f"Prediction: {pred_class} (Confidence: {confidence:.2f})")
    
    # Remove axis
    plt.axis('off')
    
    # Save if output_path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

def create_annotated_image(result, output_path=None):
    """
    Create an annotated image with prediction overlay
    
    Args:
        result (dict): Prediction result
        output_path (str or Path, optional): Path to save annotated image
        
    Returns:
        numpy.ndarray: Annotated image
    """
    # Extract data
    image = np.array(result['image'])
    pred_class = result['predicted_class']
    confidence = result['confidence']
    
    # Convert to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Define annotation parameters
    if pred_class == 'fresh':
        color = (0, 255, 0)  # Green for fresh
    else:
        color = (0, 0, 255)  # Red for spoiled
    
    # Add border
    h, w = image_bgr.shape[:2]
    annotated = cv2.copyMakeBorder(image_bgr, 40, 10, 10, 10, cv2.BORDER_CONSTANT, value=color)
    
    # Add prediction text
    cv2.putText(annotated, f"{pred_class.upper()}", (15, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add confidence text
    cv2.putText(annotated, f"Confidence: {confidence:.2f}", (15 + w // 2, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Save if output_path is provided
    if output_path:
        cv2.imwrite(str(output_path), annotated)
    
    return annotated

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(42)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse class names
    class_names = args.class_names.split(',')
    
    # Load model
    print(f"Loading model from {args.model_path}")
    try:
        model = FruitFreshnessClassifier.load(args.model_path)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Get transform
    transform = get_transforms(train=False)
    
    # Check if input is a file or directory
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Process a single image
        print(f"Processing single image: {input_path}")
        
        # Make prediction
        result = predict_image(model, input_path, transform, class_names)
        
        # Print result
        print(f"Prediction: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2f}")
        
        # Save result as JSON
        result_copy = result.copy()
        result_copy.pop('image')  # Remove image from result for JSON serialization
        with open(output_dir / f"{input_path.stem}_prediction.json", 'w') as f:
            json.dump(result_copy, f, indent=4)
        
        # Create visualization if requested
        if args.visualize:
            vis_path = output_dir / f"{input_path.stem}_visualization.png"
            create_visualization(result, class_names, vis_path)
            print(f"Visualization saved to {vis_path}")
        
        # Create annotated image if requested
        if args.save_annotated:
            annotated_path = output_dir / f"{input_path.stem}_annotated.png"
            create_annotated_image(result, annotated_path)
            print(f"Annotated image saved to {annotated_path}")
        
    elif input_path.is_dir():
        # Process all images in directory
        print(f"Processing images in directory: {input_path}")
        
        # Get image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(input_path.glob(f"*{ext}")))
            image_files.extend(list(input_path.glob(f"*{ext.upper()}")))
        
        if not image_files:
            print(f"No images found in {input_path}")
            return
        
        print(f"Found {len(image_files)} images")
        
        # Create subdirectories for outputs
        if args.visualize:
            vis_dir = output_dir / "visualizations"
            vis_dir.mkdir(exist_ok=True)
        
        if args.save_annotated:
            annotated_dir = output_dir / "annotated"
            annotated_dir.mkdir(exist_ok=True)
        
        # Process each image
        results = []
        
        for image_file in tqdm(image_files, desc="Processing images"):
            # Make prediction
            result = predict_image(model, image_file, transform, class_names)
            
            # Save result
            result_copy = result.copy()
            result_copy.pop('image')  # Remove image for JSON serialization
            results.append(result_copy)
            
            # Create visualization if requested
            if args.visualize:
                vis_path = vis_dir / f"{image_file.stem}_visualization.png"
                create_visualization(result, class_names, vis_path)
            
            # Create annotated image if requested
            if args.save_annotated:
                annotated_path = annotated_dir / f"{image_file.stem}_annotated.png"
                create_annotated_image(result, annotated_path)
        
        # Save all results as JSON
        with open(output_dir / "predictions.json", 'w') as f:
            json.dump(results, f, indent=4)
        
        # Calculate statistics
        pred_counts = {}
        for result in results:
            pred_class = result['predicted_class']
            pred_counts[pred_class] = pred_counts.get(pred_class, 0) + 1
        
        print("\nPrediction statistics:")
        for class_name, count in pred_counts.items():
            print(f"  {class_name}: {count} ({count/len(results)*100:.1f}%)")
        
        # Create summary visualization
        plt.figure(figsize=(8, 6))
        plt.bar(pred_counts.keys(), pred_counts.values())
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Prediction Distribution')
        plt.tight_layout()
        plt.savefig(output_dir / "prediction_distribution.png")
        print(f"Summary visualization saved to {output_dir}/prediction_distribution.png")
        
    else:
        print(f"Input path does not exist: {input_path}")

if __name__ == "__main__":
    main()