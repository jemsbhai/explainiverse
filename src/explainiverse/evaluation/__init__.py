# src/explainiverse/evaluation/__init__.py
"""
Evaluation metrics for explanation quality.
"""

from explainiverse.evaluation.metrics import compute_aopc, compute_roar

__all__ = ["compute_aopc", "compute_roar"]
