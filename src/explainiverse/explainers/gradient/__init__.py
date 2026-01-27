# src/explainiverse/explainers/gradient/__init__.py
"""
Gradient-based explainers for neural networks.

These explainers require models that support gradient computation,
typically via the PyTorchAdapter.

Explainers:
    - IntegratedGradientsExplainer: Axiomatic attributions via path integration
    - GradCAMExplainer: Visual explanations for CNNs
    - DeepLIFTExplainer: Reference-based attribution
    - DeepLIFTShapExplainer: DeepLIFT + SHAP combination
    - SmoothGradExplainer: Noise-averaged gradients
    - SaliencyExplainer: Basic gradient attribution
    - TCAVExplainer: Concept-based explanations (TCAV)
    - LRPExplainer: Layer-wise Relevance Propagation
"""

from explainiverse.explainers.gradient.integrated_gradients import IntegratedGradientsExplainer
from explainiverse.explainers.gradient.gradcam import GradCAMExplainer
from explainiverse.explainers.gradient.deeplift import DeepLIFTExplainer, DeepLIFTShapExplainer
from explainiverse.explainers.gradient.smoothgrad import SmoothGradExplainer
from explainiverse.explainers.gradient.saliency import SaliencyExplainer
from explainiverse.explainers.gradient.tcav import TCAVExplainer, ConceptActivationVector
from explainiverse.explainers.gradient.lrp import LRPExplainer

__all__ = [
    "IntegratedGradientsExplainer", 
    "GradCAMExplainer",
    "DeepLIFTExplainer",
    "DeepLIFTShapExplainer",
    "SmoothGradExplainer",
    "SaliencyExplainer",
    "TCAVExplainer",
    "ConceptActivationVector",
    "LRPExplainer",
]
