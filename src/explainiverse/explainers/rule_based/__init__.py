# src/explainiverse/explainers/rule_based/__init__.py
"""
Rule-based explainers - explanations in the form of if-then rules.
"""

from explainiverse.explainers.rule_based.anchor_tabular import AnchorTabularExplainer
from explainiverse.explainers.rule_based.anchors_wrapper import AnchorsExplainer

__all__ = ["AnchorTabularExplainer", "AnchorsExplainer"]
