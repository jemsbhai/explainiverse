# src/explainiverse/explainers/gradient/__init__.py
"""
Gradient-based explainers for neural networks.

These explainers require models that support gradient computation,
typically via the PyTorchAdapter.
"""

from explainiverse.explainers.gradient.integrated_gradients import IntegratedGradientsExplainer
from explainiverse.explainers.gradient.gradcam import GradCAMExplainer
from explainiverse.explainers.gradient.deeplift import DeepLIFTExplainer, DeepLIFTShapExplainer
from explainiverse.explainers.gradient.smoothgrad import SmoothGradExplainer
from explainiverse.explainers.gradient.saliency import SaliencyExplainer

__all__ = [
    "IntegratedGradientsExplainer", 
    "GradCAMExplainer",
    "DeepLIFTExplainer",
    "DeepLIFTShapExplainer",
    "SmoothGradExplainer",
    "SaliencyExplainer",
]
