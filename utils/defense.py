"""
defense.py — Defense mechanisms against adversarial perturbations.

Implements spatial filtering and other denoising techniques to mitigate
the effects of adversarial attacks on image classifiers.
"""

import io
import cv2
import numpy as np
from PIL import Image


def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply a median spatial filter to remove adversarial noise.

    The median filter is effective against FGSM-style perturbations
    because it replaces each pixel with the median of its neighborhood,
    smoothing out the high-frequency salt-and-pepper-like noise
    introduced by the sign gradient.

    Args:
        image: Input image as NumPy array (H, W, 3), uint8.
        kernel_size: Size of the median filter kernel. Must be odd.
                     Default is 3 (a 3×3 neighborhood).

    Returns:
        Filtered image as uint8 NumPy array.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd kernel size

    return cv2.medianBlur(image, kernel_size)


def gaussian_blur(image: np.ndarray, kernel_size: int = 3, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian blur as a defense mechanism.

    Smooths out high-frequency adversarial perturbations using a
    Gaussian kernel.

    Args:
        image: Input image as NumPy array (H, W, 3), uint8.
        kernel_size: Size of the Gaussian kernel. Must be odd.
        sigma: Standard deviation of the Gaussian kernel.

    Returns:
        Blurred image as uint8 NumPy array.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def bilateral_filter(image: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """
    Apply bilateral filtering to denoise while preserving edges.

    The bilateral filter is edge-aware, making it effective for
    removing adversarial perturbations while maintaining image sharpness.

    Args:
        image: Input image as NumPy array (H, W, 3), uint8.
        d: Diameter of each pixel neighborhood.
        sigma_color: Filter sigma in the color space.
        sigma_space: Filter sigma in the coordinate space.

    Returns:
        Filtered image as uint8 NumPy array.
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def jpeg_recompression(image: np.ndarray, quality: int = 75) -> np.ndarray:
    """
    Apply JPEG recompression to remove high-frequency adversarial perturbations.

    Args:
        image: Input image as NumPy array (H, W, 3), uint8.
        quality: JPEG compression quality (1-100). Default is 75.

    Returns:
        Recompressed image as uint8 NumPy array.
    """
    pil_img = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")
    return np.array(recompressed)


# ============================================================================
# Defense Registry (for dashboard dropdown)
# ============================================================================

DEFENSE_METHODS = {
    "Median Filter (3×3)": lambda img: median_filter(img, kernel_size=3),
    "Gaussian Blur (3×3, σ=1.0)": lambda img: gaussian_blur(img, kernel_size=3, sigma=1.0),
    "JPEG Recompression (Q=75)": lambda img: jpeg_recompression(img, quality=75),
    "Bilateral Filter (d=9)": lambda img: bilateral_filter(img, d=9),
}

