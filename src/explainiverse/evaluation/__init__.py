# src/explainiverse/evaluation/__init__.py
"""
Evaluation metrics for explanation quality.

Includes:
- Faithfulness metrics (PGI, PGU, Comprehensiveness, Sufficiency, Faithfulness Estimate)
- Stability metrics (RIS, ROS, Lipschitz)
- Perturbation metrics (AOPC, ROAR)
- Extended faithfulness metrics (Phase 1 expansion)
- Insertion/Deletion AUC (Petsiuk et al., 2018)
- Robustness metrics (Max-Sensitivity, Avg-Sensitivity, Continuity) — Phase 2
- Complexity metrics (Sparseness, Complexity, Effective Complexity) — Phase 4
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

from explainiverse.evaluation.robustness import (
    compute_max_sensitivity,
    compute_batch_max_sensitivity,
    compute_avg_sensitivity,
    compute_batch_avg_sensitivity,
    compute_continuity,
    compute_batch_continuity,
)

from explainiverse.evaluation.complexity import (
    compute_sparseness,
    compute_batch_sparseness,
    compute_complexity,
    compute_batch_complexity,
    compute_effective_complexity,
    compute_batch_effective_complexity,
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
    compute_road,
    compute_road_combined,
    compute_batch_road,
    compute_deletion_auc,
    compute_batch_deletion_auc,
    compute_insertion_auc,
    compute_batch_insertion_auc,
    compute_insertion_deletion_auc,
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
    "compute_road",
    "compute_road_combined",
    "compute_batch_road",
    # Insertion/Deletion AUC (Petsiuk et al., 2018)
    "compute_deletion_auc",
    "compute_batch_deletion_auc",
    "compute_insertion_auc",
    "compute_batch_insertion_auc",
    "compute_insertion_deletion_auc",
    # Robustness metrics (Phase 2)
    "compute_max_sensitivity",
    "compute_batch_max_sensitivity",
    "compute_avg_sensitivity",
    "compute_batch_avg_sensitivity",
    "compute_continuity",
    "compute_batch_continuity",
    # Complexity metrics (Phase 4)
    "compute_sparseness",
    "compute_batch_sparseness",
    "compute_complexity",
    "compute_batch_complexity",
    "compute_effective_complexity",
    "compute_batch_effective_complexity",
]
