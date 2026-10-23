"""
manipulation.py — Image manipulation utilities.

Contains both benign image transformations (brightness, rotation, noise,
JPEG compression) and the adversarial FGSM attack implementation.
"""

import io
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image


# ============================================================================
# Benign Transformations
# ============================================================================

def adjust_brightness(image: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """
    Adjust image brightness.

    Args:
        image: Input image as NumPy array (H, W, 3), uint8.
        factor: Brightness multiplier. >1 brightens, <1 darkens.

    Returns:
        Brightness-adjusted image as uint8 NumPy array.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def adjust_contrast(image: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """
    Adjust image contrast.

    Args:
        image: Input image as NumPy array (H, W, 3), uint8.
        factor: Contrast multiplier. >1 increases, <1 decreases.

    Returns:
        Contrast-adjusted image as uint8 NumPy array.
    """
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    adjusted = mean + factor * (image.astype(np.float32) - mean)
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def rotate_image(image: np.ndarray, angle: float = 15.0) -> np.ndarray:
    """
    Rotate image by a given angle around its center.

    Args:
        image: Input image as NumPy array (H, W, 3), uint8.
        angle: Rotation angle in degrees (positive = counter-clockwise).

    Returns:
        Rotated image as uint8 NumPy array (same dimensions, with black fill).
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), borderValue=(0, 0, 0))
    return rotated


def add_gaussian_noise(image: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """
    Add Gaussian noise to an image.

    Args:
        image: Input image as NumPy array (H, W, 3), uint8.
        sigma: Standard deviation of the Gaussian noise.

    Returns:
        Noisy image as uint8 NumPy array.
    """
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def jpeg_compress(image: np.ndarray, quality: int = 10) -> np.ndarray:
    """
    Apply JPEG compression artifacts to an image.

    Args:
        image: Input image as NumPy array (H, W, 3), uint8.
        quality: JPEG quality (1-100). Lower = more artifacts.

    Returns:
        JPEG-compressed image as uint8 NumPy array.
    """
    pil_img = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer).convert("RGB")
    return np.array(compressed)


# ============================================================================
# Adversarial Attack: FGSM
# ============================================================================

def fgsm_attack(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    epsilon: float = 0.03,
) -> torch.Tensor:
    """
    Perform the Fast Gradient Sign Method (FGSM) adversarial attack.

    Computes the gradient of the cross-entropy loss with respect to the
    input image, then perturbs the image in the direction of the gradient
    sign, scaled by epsilon.

    Args:
        model: A pre-trained CNN model (in eval mode).
        input_tensor: Preprocessed image tensor of shape (1, 3, 224, 224).
                      Will be cloned internally; the original is NOT modified.
        target_class: The ground-truth (or predicted) class index to attack.
        epsilon: Perturbation magnitude. Larger values = stronger attack
                 but more visible noise. Typical range: 0.005 – 0.1.

    Returns:
        Adversarial image tensor of shape (1, 3, 224, 224) (normalized space).
    """
    # Clone and enable gradient tracking
    perturbed = input_tensor.clone().detach().requires_grad_(True)

    # Forward pass
    output = model(perturbed)

    # Compute loss against the target class
    loss = nn.CrossEntropyLoss()(output, torch.tensor([target_class]))

    # Backward pass to get input gradients
    model.zero_grad()
    loss.backward()

    # Generate the adversarial perturbation
    gradient_sign = perturbed.grad.data.sign()
    adversarial = perturbed.data + epsilon * gradient_sign

    return adversarial.detach()


# ============================================================================
# Benign Transform Registry (for dashboard dropdown)
# ============================================================================

BENIGN_TRANSFORMS = {
    # Brightness / Contrast: ±10% to ±50% (Table I)
    "Brightness (+10%)": lambda img: adjust_brightness(img, factor=1.1),
    "Brightness (+50%)": lambda img: adjust_brightness(img, factor=1.5),
    "Brightness (-10%)": lambda img: adjust_brightness(img, factor=0.9),
    "Brightness (-50%)": lambda img: adjust_brightness(img, factor=0.5),
    "Contrast (+10%)": lambda img: adjust_contrast(img, factor=1.1),
    "Contrast (+50%)": lambda img: adjust_contrast(img, factor=1.5),
    "Contrast (-10%)": lambda img: adjust_contrast(img, factor=0.9),
    "Contrast (-50%)": lambda img: adjust_contrast(img, factor=0.5),
    # Rotation: 5° to 45° (Table I)
    "Rotation (5°)": lambda img: rotate_image(img, angle=5),
    "Rotation (15°)": lambda img: rotate_image(img, angle=15),
    "Rotation (45°)": lambda img: rotate_image(img, angle=45),
    # Gaussian noise: σ = 5 to 25 (Table I)
    "Gaussian Noise (σ=5)": lambda img: add_gaussian_noise(img, sigma=5),
    "Gaussian Noise (σ=15)": lambda img: add_gaussian_noise(img, sigma=15),
    "Gaussian Noise (σ=25)": lambda img: add_gaussian_noise(img, sigma=25),
    # JPEG compression: Quality 90 to 10 (Table I)
    "JPEG Compression (Q=90)": lambda img: jpeg_compress(img, quality=90),
    "JPEG Compression (Q=50)": lambda img: jpeg_compress(img, quality=50),
    "JPEG Compression (Q=10)": lambda img: jpeg_compress(img, quality=10),
}
