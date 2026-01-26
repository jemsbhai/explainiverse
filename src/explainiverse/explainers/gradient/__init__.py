# src/explainiverse/explainers/gradient/__init__.py
"""
Gradient-based explainers for neural networks.

These explainers require models that support gradient computation,
typically via the PyTorchAdapter.
"""

from explainiverse.explainers.gradient.integrated_gradients import IntegratedGradientsExplainer

__all__ = ["IntegratedGradientsExplainer"]
