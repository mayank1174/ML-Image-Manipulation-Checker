"""
classifier.py — Core ML classification module.

Loads pre-trained CNNs (ResNet50, MobileNetV2) from torchvision and
provides preprocessing + inference utilities for ImageNet classification.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Standard preprocessing pipeline for ImageNet models
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_model(model_name: str = "resnet50") -> nn.Module:
    """
    Load a pre-trained CNN model.

    Args:
        model_name: One of 'resnet50' or 'mobilenetv2'.

    Returns:
        A pre-trained model set to evaluation mode.
    """
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    elif model_name == "mobilenetv2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Choose 'resnet50' or 'mobilenetv2'.")

    model.eval()
    return model


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess a PIL Image for model input.

    Resizes, center-crops to 224x224, converts to tensor,
    and normalizes with ImageNet statistics.

    Args:
        image: A PIL Image in RGB mode.

    Returns:
        A tensor of shape (1, 3, 224, 224).
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    tensor = preprocess(image)
    return tensor.unsqueeze(0)  # Add batch dimension


def classify(model: nn.Module, input_tensor: torch.Tensor, top_k: int = 5) -> list[dict]:
    """
    Run inference and return top-k predictions.

    Args:
        model: A pre-trained CNN model.
        input_tensor: Preprocessed image tensor of shape (1, 3, 224, 224).
        top_k: Number of top predictions to return.

    Returns:
        A list of dicts with keys 'class_index', 'class_name', 'confidence'.
    """
    from models.labels import IMAGENET_LABELS

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    top_probs, top_indices = torch.topk(probabilities, top_k)

    results = []
    for i in range(top_k):
        idx = top_indices[i].item()
        results.append({
            "class_index": idx,
            "class_name": IMAGENET_LABELS.get(idx, f"Class {idx}"),
            "confidence": round(top_probs[i].item() * 100, 2),
        })

    return results


def denormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalization and convert a tensor to a NumPy uint8 image.

    Args:
        tensor: A normalized tensor of shape (1, 3, H, W) or (3, H, W).

    Returns:
        A NumPy array of shape (H, W, 3) with uint8 values in [0, 255].
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to HWC uint8
    img_np = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return img_np
