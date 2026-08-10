# src/explainiverse/explainers/example_based/__init__.py
"""
Example-based explanation methods.

The exported ProtoDash implementation selects examples and non-negative
weights by its documented kernel objective. This module does not assert that
the selected examples are representative to a person or for a downstream use.

Methods:
- ProtoDash: Select examples with kernel-objective weights
"""

from explainiverse.explainers.example_based.protodash import ProtoDashExplainer

__all__ = [
    "ProtoDashExplainer",
]
