# src/explainiverse/explainers/gradient/__init__.py
"""
Gradient-based explainers for neural networks.

These explainers require models that support gradient computation,
typically via the PyTorchAdapter.

Explainers:
    - IntegratedGradientsExplainer: Numerical straight-line path-gradient attribution
    - GradCAMExplainer: Paper-defined Grad-CAM (Grad-CAM++ is not exposed)
    - HiResCAMExplainer: Element-wise HiResCAM formula (Draelos & Carin, 2021)
    - XGradCAMExplainer: XGrad-CAM formula (Fu et al., 2020)
    - LayerCAMExplainer: Hierarchical spatial-weighted CAM (Jiang et al., 2021)
    - EigenCAMExplainer: SVD-based gradient-free CAM (Muhammad & Yeasin, 2020)
    - ScoreCAMExplainer: Score-weighted gradient-free CAM (Wang et al., 2020)
    - EigenGradCAMExplainer: pytorch-grad-cam SVD-on-grad*act library variant
    - GradCAMElementWiseExplainer: pytorch-grad-cam element-wise library variant
    - AblationCAMExplainer: Gradient-free ablation-based CAM (Desai & Ramaswamy, 2020)
    - DeepLIFTExplainer: Reference-based attribution
    - DeepLIFTShapExplainer: DeepLIFT + SHAP combination
    - SmoothGradExplainer: Noise-averaged gradients
    - SaliencyExplainer: Basic gradient attribution
    - TCAVExplainer: Concept-based explanations (TCAV)
    - LRPExplainer: Layer-wise Relevance Propagation
"""

from explainiverse.explainers.gradient.cam_variants import (
    AblationCAMExplainer,
    BaseCAMExplainer,
    EigenCAMExplainer,
    EigenGradCAMExplainer,
    GradCAMElementWiseExplainer,
    HiResCAMExplainer,
    LayerCAMExplainer,
    ScoreCAMExplainer,
    XGradCAMExplainer,
)
from explainiverse.explainers.gradient.deeplift import DeepLIFTExplainer, DeepLIFTShapExplainer
from explainiverse.explainers.gradient.gradcam import GradCAMExplainer
from explainiverse.explainers.gradient.integrated_gradients import IntegratedGradientsExplainer
from explainiverse.explainers.gradient.lrp import LRPExplainer
from explainiverse.explainers.gradient.saliency import SaliencyExplainer
from explainiverse.explainers.gradient.smoothgrad import SmoothGradExplainer
from explainiverse.explainers.gradient.tcav import ConceptActivationVector, TCAVExplainer

__all__ = [
    "IntegratedGradientsExplainer",
    "GradCAMExplainer",
    "BaseCAMExplainer",
    "HiResCAMExplainer",
    "XGradCAMExplainer",
    "LayerCAMExplainer",
    "EigenCAMExplainer",
    "ScoreCAMExplainer",
    "EigenGradCAMExplainer",
    "GradCAMElementWiseExplainer",
    "AblationCAMExplainer",
    "DeepLIFTExplainer",
    "DeepLIFTShapExplainer",
    "SmoothGradExplainer",
    "SaliencyExplainer",
    "TCAVExplainer",
    "ConceptActivationVector",
    "LRPExplainer",
]
