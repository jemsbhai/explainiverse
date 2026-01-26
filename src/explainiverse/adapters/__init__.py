# src/explainiverse/adapters/__init__.py
"""
Model adapters - wrappers that provide a consistent interface for different ML frameworks.
"""

from explainiverse.adapters.base_adapter import BaseModelAdapter
from explainiverse.adapters.sklearn_adapter import SklearnAdapter

__all__ = ["BaseModelAdapter", "SklearnAdapter"]
