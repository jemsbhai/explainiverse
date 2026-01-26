# src/explainiverse/explainers/__init__.py
"""
Explainiverse Explainers - comprehensive XAI method implementations.

Local Explainers (instance-level):
- LIME: Local Interpretable Model-agnostic Explanations
- SHAP: SHapley Additive exPlanations
- Anchors: High-precision rule-based explanations
- Counterfactual: Diverse counterfactual explanations

Global Explainers (model-level):
- Permutation Importance: Feature importance via permutation
- Partial Dependence: Marginal feature effects (PDP)
- ALE: Accumulated Local Effects (unbiased for correlated features)
- SAGE: Shapley Additive Global importancE
"""

from explainiverse.explainers.attribution.lime_wrapper import LimeExplainer
from explainiverse.explainers.attribution.shap_wrapper import ShapExplainer
from explainiverse.explainers.rule_based.anchors_wrapper import AnchorsExplainer
from explainiverse.explainers.counterfactual.dice_wrapper import CounterfactualExplainer
from explainiverse.explainers.global_explainers.permutation_importance import PermutationImportanceExplainer
from explainiverse.explainers.global_explainers.partial_dependence import PartialDependenceExplainer
from explainiverse.explainers.global_explainers.ale import ALEExplainer
from explainiverse.explainers.global_explainers.sage import SAGEExplainer

__all__ = [
    # Local explainers
    "LimeExplainer",
    "ShapExplainer",
    "AnchorsExplainer",
    "CounterfactualExplainer",
    # Global explainers
    "PermutationImportanceExplainer",
    "PartialDependenceExplainer",
    "ALEExplainer",
    "SAGEExplainer",
]
