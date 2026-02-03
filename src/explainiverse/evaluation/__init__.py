# src/explainiverse/evaluation/__init__.py
"""
Evaluation metrics for explanation quality.

Includes:
- Faithfulness metrics (PGI, PGU, Comprehensiveness, Sufficiency, Faithfulness Estimate)
- Stability metrics (RIS, ROS, Lipschitz)
- Perturbation metrics (AOPC, ROAR)
- Extended faithfulness metrics (Phase 1 expansion)
"""

from explainiverse.evaluation.metrics import (
    compute_aopc,
    compute_batch_aopc,
    compute_roar,
    compute_roar_curve,
)

from explainiverse.evaluation.faithfulness import (
    compute_pgi,
    compute_pgu,
    compute_faithfulness_score,
    compute_comprehensiveness,
    compute_sufficiency,
    compute_faithfulness_correlation,
    compare_explainer_faithfulness,
    compute_batch_faithfulness,
)

from explainiverse.evaluation.stability import (
    compute_ris,
    compute_ros,
    compute_lipschitz_estimate,
    compute_stability_metrics,
    compute_batch_stability,
    compare_explainer_stability,
)

from explainiverse.evaluation.faithfulness_extended import (
    compute_faithfulness_estimate,
    compute_batch_faithfulness_estimate,
    compute_monotonicity,
    compute_batch_monotonicity,
)

__all__ = [
    # Perturbation metrics (existing)
    "compute_aopc",
    "compute_batch_aopc",
    "compute_roar",
    "compute_roar_curve",
    # Faithfulness metrics (core)
    "compute_pgi",
    "compute_pgu",
    "compute_faithfulness_score",
    "compute_comprehensiveness",
    "compute_sufficiency",
    "compute_faithfulness_correlation",
    "compare_explainer_faithfulness",
    "compute_batch_faithfulness",
    # Stability metrics
    "compute_ris",
    "compute_ros",
    "compute_lipschitz_estimate",
    "compute_stability_metrics",
    "compute_batch_stability",
    "compare_explainer_stability",
    # Extended faithfulness metrics (Phase 1)
    "compute_faithfulness_estimate",
    "compute_batch_faithfulness_estimate",
    "compute_monotonicity",
    "compute_batch_monotonicity",
]
