# src/explainiverse/explainers/global_explainers/__init__.py
"""
Global explainers - model-level explanations.

These explainers compute dataset-level or model-level quantities under the
contracts documented by each implementation.
"""

from explainiverse.explainers.global_explainers.ale import ALEExplainer
from explainiverse.explainers.global_explainers.partial_dependence import PartialDependenceExplainer
from explainiverse.explainers.global_explainers.permutation_importance import (
    PermutationImportanceExplainer,
)
from explainiverse.explainers.global_explainers.sage import SAGEExplainer

__all__ = [
    "PermutationImportanceExplainer",
    "PartialDependenceExplainer",
    "ALEExplainer",
    "SAGEExplainer",
]
