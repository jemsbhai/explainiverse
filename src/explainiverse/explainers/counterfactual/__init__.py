# src/explainiverse/explainers/counterfactual/__init__.py
"""
Counterfactual-search entry points for model-valid input changes.
"""

from explainiverse.explainers.counterfactual.dice_wrapper import CounterfactualExplainer

__all__ = ["CounterfactualExplainer"]
