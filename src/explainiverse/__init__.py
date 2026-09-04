# src/explainiverse/__init__.py
"""
Explainiverse - a unified, extensible explainability framework.

The package exposes a registry of local, global, gradient, concept, rule,
counterfactual-search, and example-based methods together with evaluation
metrics and diagnostics. Registry metadata carries an explicit accuracy-audit
status and claim scope; method/count leadership claims are intentionally not
made here.

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

from explainiverse.adapters import TORCH_AVAILABLE
from explainiverse.adapters.sklearn_adapter import SklearnAdapter
from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import EXPLANATION_WIRE_SCHEMA_VERSION, Explanation
from explainiverse.core.registry import (
    ExplainerMeta,
    ExplainerRegistry,
    default_registry,
    get_default_registry,
)
from explainiverse.core.scaled_detail import (
    SCALED_DETAIL_SCHEMA_VERSION,
    DetailRepresentationError,
    decode_scaled_detail,
    encode_scaled_detail,
)
from explainiverse.engine.suite import ExplanationSuite

__version__ = "0.15.2"

__all__ = [
    # Core
    "BaseExplainer",
    "Explanation",
    "EXPLANATION_WIRE_SCHEMA_VERSION",
    "DetailRepresentationError",
    "SCALED_DETAIL_SCHEMA_VERSION",
    "encode_scaled_detail",
    "decode_scaled_detail",
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
    from explainiverse.adapters import PyTorchAdapter as PyTorchAdapter

    __all__.append("PyTorchAdapter")
