# src/explainiverse/explainers/example_based/__init__.py
"""
Example-based explanation methods.

These methods explain models by identifying representative examples
from the training data, rather than computing feature attributions.

Methods:
- ProtoDash: Select prototypical examples with importance weights
- (Future) Influence Functions: Identify training examples that most affect predictions
- (Future) MMD-Critic: Find prototypes and criticisms
"""

from explainiverse.explainers.example_based.protodash import ProtoDashExplainer

__all__ = [
    "ProtoDashExplainer",
]
