"""
Inference module for the FreshHarvest project.
Contains functions for deploying the model and making inferences.
"""

import torch
import numpy as np
import cv2
import time
from PIL import Image
import io
import base64
from pathlib import Path

from config import DEVICE, IMG_SIZE
from data import get_transforms
from model import FruitFreshnessClassifier

class FruitFreshnessInference:
    """Inference class for fruit freshness detection"""
    
    def __init__(self, model_path, device=DEVICE, class_names=None):
        """
        Initialize the inference class
        
        Args:
            model_path (str or Path): Path to the trained model
            device (torch.device): Device to use for inference
            class_names (list, optional): List of class names
        """
        self.device = device
        
        # Set default class names if not provided
        self.class_names = class_names if class_names else ['fresh', 'spoiled']
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Get transforms
        self.transform = get_transforms(train=False)
        
    def _load_model(self, model_path):
        """
        Load the model from path
        
        Args:
            model_path (str or Path): Path to the trained model
            
        Returns:
            nn.Module: Loaded model
        """
        try:
            model = FruitFreshnessClassifier.load(model_path, self.device)
            model.eval()
            return model
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")
    
    def preprocess_image(self, image):
        """
        Preprocess image for inference
        
        Args:
            image: PIL Image, numpy array, or file path
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Convert to PIL Image if needed
        if isinstance(image, str) or isinstance(image, Path):
            # Load from file path
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Convert from numpy array (OpenCV) to PIL
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        # Apply transforms and add batch dimension
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        return input_tensor
    
    def predict(self, image, return_probs=False):
        """
        Make a prediction on an image
        
        Args:
            image: PIL Image, numpy array, or file path
            return_probs (bool): Whether to return class probabilities
            
        Returns:
            dict: Prediction results
        """
        # Preprocess image
        start_time = time.time()
        input_tensor = self.preprocess_image(image)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        # Get prediction details
        pred_idx = preds.item()
        pred_class = self.class_names[pred_idx]
        confidence = probs[0][pred_idx].item()
        
        # Create result dictionary
        result = {
            'prediction': pred_class,
            'confidence': confidence,
            'inference_time': inference_time
        }
        
        # Add probabilities if requested
        if return_probs:
            result['probabilities'] = {
                self.class_names[i]: prob.item() 
                for i, prob in enumerate(probs[0])
            }
        
        return result
    
    def predict_batch(self, images, return_probs=False):
        """
        Make predictions on a batch of images
        
        Args:
            images (list): List of images (PIL, numpy, or file paths)
            return_probs (bool): Whether to return class probabilities
            
        Returns:
            list: List of prediction results
        """
        # Process each image
        results = []
        for image in images:
            result = self.predict(image, return_probs)
            results.append(result)
        
        return results
    
    def create_annotated_image(self, image, result):
        """
        Create an annotated image with prediction overlay
        
        Args:
            image: PIL Image, numpy array, or file path
            result (dict): Prediction result
            
        Returns:
            numpy.ndarray: Annotated image
        """
        # Convert to numpy array if needed
        if isinstance(image, str) or isinstance(image, Path):
            # Load from file path
            image = cv2.imread(str(image))
        elif isinstance(image, Image.Image):
            # Convert from PIL to numpy
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Get prediction information
        pred_class = result['prediction']
        confidence = result['confidence']
        
        # Define annotation parameters
        if pred_class == 'fresh':
            color = (0, 255, 0)  # Green for fresh
        else:
            color = (0, 0, 255)  # Red for spoiled
        
        # Add border
        h, w = image.shape[:2]
        annotated = cv2.copyMakeBorder(image, 40, 10, 10, 10, cv2.BORDER_CONSTANT, value=color)
        
        # Add prediction text
        cv2.putText(annotated, f"{pred_class.upper()}", (15, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add confidence text
        cv2.putText(annotated, f"Confidence: {confidence:.2f}", (15 + w // 2, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return annotated
    
    def image_to_base64(self, image):
        """
        Convert image to base64 string
        
        Args:
            image: PIL Image, numpy array, or file path
            
        Returns:
            str: Base64 encoded image
        """
        # Convert to numpy array if needed
        if isinstance(image, str) or isinstance(image, Path):
            # Load from file path
            image = cv2.imread(str(image))
        elif isinstance(image, Image.Image):
            # Convert from PIL to numpy
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Encode image to base64
        _, buffer = cv2.imencode('.jpg', image)
        base64_str = base64.b64encode(buffer).decode('utf-8')
        
        return base64_str