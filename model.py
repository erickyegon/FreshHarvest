"""
Model module for the FreshHarvest project.
Contains model architecture definitions and related utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from config import MODEL_CONFIG, DEVICE

def create_model(num_classes=2, architecture=None):
    """
    Create a model with the specified architecture
    
    Args:
        num_classes (int): Number of output classes
        architecture (str, optional): Architecture to use. If None, uses the one from MODEL_CONFIG
        
    Returns:
        nn.Module: Model with the specified architecture
    """
    if architecture is None:
        architecture = MODEL_CONFIG["architecture"]
    
    pretrained = MODEL_CONFIG["pretrained"]
    freeze_backbone = MODEL_CONFIG["freeze_backbone"]
    dropout_rate = MODEL_CONFIG["dropout_rate"]
    
    # Pick weights parameter based on pretrained option
    weights = 'DEFAULT' if pretrained else None
    
    if architecture == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=weights)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
                
        # Modify classifier for our task
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
    elif architecture == 'resnet50':
        model = models.resnet50(weights=weights)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
                
        # Modify classifier for our task
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        
    elif architecture == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=weights)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
                
        # Modify classifier for our task
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    # Move model to device
    model = model.to(DEVICE)
    
    return model

def count_trainable_parameters(model):
    """
    Count the number of trainable parameters in a model
    
    Args:
        model (nn.Module): Model to count parameters for
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_summary(model):
    """
    Get a summary of the model architecture and parameters
    
    Args:
        model (nn.Module): Model to summarize
        
    Returns:
        str: Summary of the model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_trainable_parameters(model)
    non_trainable_params = total_params - trainable_params
    
    summary = [
        f"Model Architecture: {model.__class__.__name__}",
        f"Total Parameters: {total_params:,}",
        f"Trainable Parameters: {trainable_params:,}",
        f"Non-trainable Parameters: {non_trainable_params:,}"
    ]
    
    return "\n".join(summary)

class FruitFreshnessClassifier(nn.Module):
    """Custom classifier for fruit freshness detection"""
    
    def __init__(self, num_classes=2, architecture='efficientnet_b0'):
        super(FruitFreshnessClassifier, self).__init__()
        self.model = create_model(num_classes, architecture)
        
    def forward(self, x):
        return self.model(x)
    
    def save(self, path):
        """Save model to path"""
        torch.save({
            'architecture': MODEL_CONFIG['architecture'],
            'state_dict': self.state_dict(),
            'config': MODEL_CONFIG
        }, path)
    
    @classmethod
    def load(cls, path, device=DEVICE):
        """Load model from path"""
        checkpoint = torch.load(path, map_location=device)
        architecture = checkpoint.get('architecture', MODEL_CONFIG['architecture'])
        model = cls(num_classes=2, architecture=architecture)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        return model