# src/explainiverse/core/__init__.py
"""
Explainiverse core components.
"""

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

__all__ = [
    "BaseExplainer",
    "Explanation",
    "EXPLANATION_WIRE_SCHEMA_VERSION",
    "ExplainerRegistry",
    "ExplainerMeta",
    "default_registry",
    "get_default_registry",
    "DetailRepresentationError",
    "SCALED_DETAIL_SCHEMA_VERSION",
    "encode_scaled_detail",
    "decode_scaled_detail",
]
