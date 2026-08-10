# src/explainiverse/adapters/__init__.py
"""
Model adapters with framework-specific prediction contracts.

Available adapters:
- SklearnAdapter: For supported scikit-learn estimator contracts (always available)
- PyTorchAdapter: For PyTorch nn.Module models (requires torch)
"""

from explainiverse.adapters.base_adapter import BaseModelAdapter
from explainiverse.adapters.sklearn_adapter import SklearnAdapter

# Conditionally import PyTorchAdapter if torch is available
try:
    from explainiverse.adapters.pytorch_adapter import TORCH_AVAILABLE, PyTorchAdapter

    __all__ = ["BaseModelAdapter", "SklearnAdapter", "PyTorchAdapter", "TORCH_AVAILABLE"]
except ImportError:
    TORCH_AVAILABLE = False
    __all__ = ["BaseModelAdapter", "SklearnAdapter", "TORCH_AVAILABLE"]
