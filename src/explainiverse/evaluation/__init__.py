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
    compute_monotonicity_nguyen,
    compute_batch_monotonicity_nguyen,
    compute_pixel_flipping,
    compute_batch_pixel_flipping,
    compute_region_perturbation,
    compute_batch_region_perturbation,
    compute_selectivity,
    compute_batch_selectivity,
    compute_sensitivity_n,
    compute_sensitivity_n_multi,
    compute_batch_sensitivity_n,
    compute_irof,
    compute_irof_multi_segment,
    compute_batch_irof,
    compute_infidelity,
    compute_infidelity_multi_perturbation,
    compute_batch_infidelity,
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
    "compute_monotonicity_nguyen",
    "compute_batch_monotonicity_nguyen",
    "compute_pixel_flipping",
    "compute_batch_pixel_flipping",
    "compute_region_perturbation",
    "compute_batch_region_perturbation",
    "compute_selectivity",
    "compute_batch_selectivity",
    "compute_sensitivity_n",
    "compute_sensitivity_n_multi",
    "compute_batch_sensitivity_n",
    "compute_irof",
    "compute_irof_multi_segment",
    "compute_batch_irof",
    "compute_infidelity",
    "compute_infidelity_multi_perturbation",
    "compute_batch_infidelity",
]
