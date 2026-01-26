# src/explainiverse/__init__.py
"""
Explainiverse - A unified, extensible explainability framework.

Supports multiple XAI methods including LIME, SHAP, Anchors, Counterfactuals,
Permutation Importance, PDP, ALE, and SAGE through a consistent interface.

Quick Start:
    from explainiverse import default_registry
    
    # List available explainers
    print(default_registry.list_explainers())
    
    # Create an explainer
    explainer = default_registry.create("lime", model=adapter, training_data=X, ...)
    explanation = explainer.explain(instance)
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
from explainiverse.engine.suite import ExplanationSuite

__version__ = "0.2.0"

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
    # Engine
    "ExplanationSuite",
]
