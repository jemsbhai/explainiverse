# src/explainiverse/adapters/__init__.py
"""
Model adapters - wrappers that provide a consistent interface for different ML frameworks.

Available adapters:
- SklearnAdapter: For scikit-learn models (always available)
- PyTorchAdapter: For PyTorch nn.Module models (requires torch)
"""

from explainiverse.adapters.base_adapter import BaseModelAdapter
from explainiverse.adapters.sklearn_adapter import SklearnAdapter

# Conditionally import PyTorchAdapter if torch is available
try:
    from explainiverse.adapters.pytorch_adapter import PyTorchAdapter, TORCH_AVAILABLE
    __all__ = ["BaseModelAdapter", "SklearnAdapter", "PyTorchAdapter", "TORCH_AVAILABLE"]
except ImportError:
    TORCH_AVAILABLE = False
    __all__ = ["BaseModelAdapter", "SklearnAdapter", "TORCH_AVAILABLE"]
