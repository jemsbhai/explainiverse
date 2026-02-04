# src/explainiverse/__init__.py
"""
Explainiverse - A unified, extensible explainability framework.

Supports 18 state-of-the-art XAI methods including LIME, SHAP, TreeSHAP,
Integrated Gradients, DeepLIFT, DeepSHAP, LRP, GradCAM, TCAV, Anchors,
Counterfactuals, Permutation Importance, PDP, ALE, SAGE, and ProtoDash
through a consistent interface.

Quick Start:
    from explainiverse import default_registry
    
    # List available explainers
    print(default_registry.list_explainers())
    
    # Create an explainer
    explainer = default_registry.create("lime", model=adapter, training_data=X, ...)
    explanation = explainer.explain(instance)
    
For PyTorch models:
    from explainiverse import PyTorchAdapter  # Requires torch
    adapter = PyTorchAdapter(model, task="classification")
"""

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation
from explainiverse.core.registry import (
    ExplainerRegistry,
    ExplainerMeta,
    default_registry,
    get_default_registry,
)
from explainiverse.adapters.sklearn_adapter import SklearnAdapter
from explainiverse.adapters import TORCH_AVAILABLE
from explainiverse.engine.suite import ExplanationSuite

__version__ = "0.8.10"

__all__ = [
    # Core
    "BaseExplainer",
    "Explanation",
    # Registry
    "ExplainerRegistry",
    "ExplainerMeta",
    "default_registry",
    "get_default_registry",
    # Adapters
    "SklearnAdapter",
    "TORCH_AVAILABLE",
    # Engine
    "ExplanationSuite",
]

# Conditionally export PyTorchAdapter if torch is available
if TORCH_AVAILABLE:
    from explainiverse.adapters import PyTorchAdapter
    __all__.append("PyTorchAdapter")
