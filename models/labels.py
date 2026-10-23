"""
labels.py — ImageNet class labels.

Provides a mapping from class index (0-999) to human-readable label names
for the ImageNet-1K classification dataset.

This file contains the full 1000-class label set used by torchvision
pre-trained models (IMAGENET1K_V1 / V2).
"""

# We dynamically load labels from torchvision's built-in weights metadata
# at first import. This ensures we always match the model's class ordering.

_LABELS_CACHE: dict[int, str] | None = None


def _load_labels() -> dict[int, str]:
    """Load ImageNet labels from torchvision weights metadata."""
    try:
        from torchvision.models import ResNet50_Weights
        categories = ResNet50_Weights.IMAGENET1K_V2.meta["categories"]
        return {i: name for i, name in enumerate(categories)}
    except Exception:
        # Fallback: return a minimal dict; model will still work
        # but labels will show as "Class N"
        return {}


def get_labels() -> dict[int, str]:
    """Get the ImageNet label mapping, loading once and caching."""
    global _LABELS_CACHE
    if _LABELS_CACHE is None:
        _LABELS_CACHE = _load_labels()
    return _LABELS_CACHE


# Module-level alias for easy import: `from models.labels import IMAGENET_LABELS`
IMAGENET_LABELS = get_labels()
