# src/explainiverse/evaluation/__init__.py
"""
Evaluation metrics and diagnostics for explanations.

Includes:
- Faithfulness diagnostics (deterministic PGI/PGU specializations, tabular ERASER
  adaptations, Bhatt subset correlation, and separately scoped extended metrics)
- Stability diagnostics (paper/Quantus-scoped RIS, RRS, and ROS finite-sample
  estimates plus compatibility entry points)
- Perturbation metrics (AOPC, ROAR)
- Robustness diagnostics (sampled Yeh Max-Sensitivity, a noncanonical mean-sensitivity
  compatibility heuristic, a finite-sample local Lipschitz proxy, and Dasgupta
  consistency)
- Complexity metrics (Sparseness, Bhatt Complexity, Attribution Threshold Count;
  historical Effective Complexity names are warning aliases)
- Agreement metrics (Feature Agreement, Rank Agreement)
- Localisation metrics (Pointing Game, Attribution Localisation, Top-K Intersection,
  Relevance Mass Accuracy, Relevance Rank Accuracy, AUC, Energy-Based Pointing Game,
  Focus, and a library-defined Attribution IoU diagnostic)
- Randomisation metrics (MPRT, Random Logit, Smooth MPRT, Efficient MPRT,
  Data Randomisation)
- Axiomatic checks and explicitly limited diagnostics (Completeness, Nguyen
  Non-Sensitivity, translation Input Invariance checks, and conditional Symmetry)
- Fairness-related audits (group metric disparity, cross-group Lipschitz,
  sensitive-attribution change/gap, exact fidelity gaps, and prediction-conditioned
  metric disparity). Diagnostic gaps are not standalone fairness verdicts.
"""

from explainiverse.evaluation.agreement import (
    compute_batch_feature_agreement,
    compute_batch_rank_agreement,
    compute_feature_agreement,
    compute_rank_agreement,
)
from explainiverse.evaluation.axiomatic import (
    compute_batch_completeness,
    compute_batch_input_invariance,
    compute_batch_input_invariance_pytorch,
    compute_batch_non_sensitivity,
    compute_batch_symmetry,
    compute_completeness,
    compute_completeness_score,
    compute_input_invariance,
    compute_input_invariance_pytorch,
    compute_non_sensitivity,
    compute_non_sensitivity_score,
    compute_symmetry,
    compute_symmetry_score,
)
from explainiverse.evaluation.complexity import (
    compute_attribution_threshold_count,
    compute_batch_attribution_threshold_count,
    compute_batch_complexity,
    compute_batch_effective_complexity,
    compute_batch_sparseness,
    compute_complexity,
    compute_effective_complexity,
    compute_sparseness,
)
from explainiverse.evaluation.fairness import (
    FairnessMetricMeta,
    FairnessMetricRegistry,
    compute_attribution_parity,
    compute_batch_group_fairness,
    compute_conditional_fairness,
    compute_counterfactual_fairness,
    compute_cross_group_lipschitz_diagnostic,
    compute_fidelity_disparity,
    compute_fidelity_gap,
    compute_group_fairness,
    compute_group_fairness_score,
    compute_group_metric_disparity,
    compute_individual_fairness,
    compute_prediction_conditioned_metric_disparity,
    compute_sensitive_attribution_change,
    compute_sensitive_attribution_gap,
    get_default_fairness_registry,
)
from explainiverse.evaluation.faithfulness import (
    compare_explainer_faithfulness,
    compute_batch_faithfulness,
    compute_comprehensiveness,
    compute_faithfulness_correlation,
    compute_faithfulness_score,
    compute_pgi,
    compute_pgu,
    compute_sufficiency,
)
from explainiverse.evaluation.faithfulness_extended import (
    compute_batch_deletion_auc,
    compute_batch_faithfulness_estimate,
    compute_batch_infidelity,
    compute_batch_insertion_auc,
    compute_batch_irof,
    compute_batch_monotonicity,
    compute_batch_monotonicity_nguyen,
    compute_batch_pixel_flipping,
    compute_batch_region_perturbation,
    compute_batch_road,
    compute_batch_selectivity,
    compute_batch_sensitivity_n,
    compute_deletion_auc,
    compute_faithfulness_estimate,
    compute_infidelity,
    compute_infidelity_multi_perturbation,
    compute_insertion_auc,
    compute_insertion_deletion_auc,
    compute_irof,
    compute_irof_multi_segment,
    compute_monotonicity,
    compute_monotonicity_nguyen,
    compute_pixel_flipping,
    compute_region_perturbation,
    compute_road,
    compute_road_combined,
    compute_selectivity,
    compute_sensitivity_n,
    compute_sensitivity_n_multi,
)
from explainiverse.evaluation.localisation import (
    LocalisationMask,
    compute_attribution_iou,
    compute_attribution_localisation,
    compute_auc,
    compute_batch_attribution_iou,
    compute_batch_attribution_localisation,
    compute_batch_auc,
    compute_batch_energy_based_pointing_game,
    compute_batch_focus,
    compute_batch_pointing_game,
    compute_batch_relevance_mass_accuracy,
    compute_batch_relevance_rank_accuracy,
    compute_batch_top_k_intersection,
    compute_energy_based_pointing_game,
    compute_focus,
    compute_pointing_game,
    compute_relevance_mass_accuracy,
    compute_relevance_rank_accuracy,
    compute_top_k_intersection,
)
from explainiverse.evaluation.metrics import (
    compute_aopc,
    compute_batch_aopc,
    compute_roar,
    compute_roar_curve,
)
from explainiverse.evaluation.randomisation import (
    compute_batch_data_randomisation,
    compute_batch_efficient_mprt,
    compute_batch_mprt,
    compute_batch_random_logit,
    compute_batch_smooth_mprt,
    compute_data_randomisation,
    compute_data_randomisation_score,
    compute_efficient_mprt,
    compute_mprt,
    compute_mprt_score,
    compute_random_logit,
    compute_random_logit_score,
    compute_smooth_mprt,
)
from explainiverse.evaluation.registry import MetricMeta, MetricRegistry, build_metric_registry
from explainiverse.evaluation.robustness import (
    compare_consistency_results,
    compute_avg_sensitivity,
    compute_batch_avg_sensitivity,
    compute_batch_consistency,
    compute_batch_continuity,
    compute_batch_max_sensitivity,
    compute_batch_relative_input_stability,
    compute_batch_relative_output_stability,
    compute_batch_relative_representation_stability,
    compute_batch_relative_stability,
    compute_consistency,
    compute_continuity,
    compute_max_sensitivity,
    compute_relative_input_stability,
    compute_relative_output_stability,
    compute_relative_representation_stability,
    compute_relative_stability,
)
from explainiverse.evaluation.stability import (
    compare_explainer_stability,
    compute_batch_stability,
    compute_lipschitz_estimate,
    compute_ris,
    compute_ros,
    compute_stability_metrics,
)
from explainiverse.evaluation.uncertainty import (
    compare_intervention_sensitivity_reports,
    evaluate_intervention_sensitivity,
    run_seeded_replicates,
    summarize_replicate_estimates,
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
    # Extended faithfulness diagnostics
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
    # Robustness diagnostics
    "compute_max_sensitivity",
    "compute_batch_max_sensitivity",
    "compute_avg_sensitivity",
    "compute_batch_avg_sensitivity",
    "compute_continuity",
    "compute_batch_continuity",
    "compute_consistency",
    "compute_batch_consistency",
    "compare_consistency_results",
    "summarize_replicate_estimates",
    "run_seeded_replicates",
    "evaluate_intervention_sensitivity",
    "compare_intervention_sensitivity_reports",
    # Relative Stability metrics (Agarwal et al., 2022)
    "compute_relative_input_stability",
    "compute_batch_relative_input_stability",
    "compute_relative_representation_stability",
    "compute_batch_relative_representation_stability",
    "compute_relative_output_stability",
    "compute_batch_relative_output_stability",
    "compute_relative_stability",
    "compute_batch_relative_stability",
    # Agreement metrics (Krishna et al., 2022)
    "compute_feature_agreement",
    "compute_batch_feature_agreement",
    "compute_rank_agreement",
    "compute_batch_rank_agreement",
    # Complexity diagnostics
    "compute_sparseness",
    "compute_batch_sparseness",
    "compute_complexity",
    "compute_batch_complexity",
    "compute_attribution_threshold_count",
    "compute_batch_attribution_threshold_count",
    "compute_effective_complexity",
    "compute_batch_effective_complexity",
    # Localisation diagnostics
    "LocalisationMask",
    "compute_pointing_game",
    "compute_batch_pointing_game",
    "compute_attribution_localisation",
    "compute_batch_attribution_localisation",
    "compute_top_k_intersection",
    "compute_batch_top_k_intersection",
    "compute_relevance_mass_accuracy",
    "compute_batch_relevance_mass_accuracy",
    "compute_relevance_rank_accuracy",
    "compute_batch_relevance_rank_accuracy",
    "compute_auc",
    "compute_batch_auc",
    "compute_energy_based_pointing_game",
    "compute_batch_energy_based_pointing_game",
    "compute_focus",
    "compute_batch_focus",
    "compute_attribution_iou",
    "compute_batch_attribution_iou",
    # Randomisation diagnostics
    "compute_mprt",
    "compute_mprt_score",
    "compute_batch_mprt",
    "compute_random_logit",
    "compute_random_logit_score",
    "compute_batch_random_logit",
    "compute_smooth_mprt",
    "compute_batch_smooth_mprt",
    "compute_efficient_mprt",
    "compute_batch_efficient_mprt",
    "compute_data_randomisation",
    "compute_data_randomisation_score",
    "compute_batch_data_randomisation",
    # Axiomatic checks and diagnostics
    "compute_completeness",
    "compute_completeness_score",
    "compute_batch_completeness",
    "compute_non_sensitivity",
    "compute_non_sensitivity_score",
    "compute_batch_non_sensitivity",
    "compute_input_invariance",
    "compute_input_invariance_pytorch",
    "compute_batch_input_invariance",
    "compute_batch_input_invariance_pytorch",
    "compute_symmetry",
    "compute_symmetry_score",
    "compute_batch_symmetry",
    # Fairness-related metrics and disparity diagnostics
    "FairnessMetricRegistry",
    "FairnessMetricMeta",
    "get_default_fairness_registry",
    "compute_group_metric_disparity",
    "compute_group_fairness",
    "compute_group_fairness_score",
    "compute_batch_group_fairness",
    "compute_cross_group_lipschitz_diagnostic",
    "compute_individual_fairness",
    "compute_sensitive_attribution_change",
    "compute_counterfactual_fairness",
    "compute_fidelity_gap",
    "compute_fidelity_disparity",
    "compute_sensitive_attribution_gap",
    "compute_attribution_parity",
    "compute_prediction_conditioned_metric_disparity",
    "compute_conditional_fairness",
]


# Build the trust inventory only after every public evaluation callable exists
# in this namespace.  The exact-coverage assertion means that exporting a new
# ``compute_*`` endpoint without metadata fails immediately in tests/imports
# rather than silently creating an unaudited public surface.
default_metric_registry = build_metric_registry(globals(), __all__)


def get_default_metric_registry() -> MetricRegistry:
    """Return the process-wide registry for public evaluation endpoints."""

    return default_metric_registry


__all__.extend(
    [
        "MetricMeta",
        "MetricRegistry",
        "default_metric_registry",
        "get_default_metric_registry",
    ]
)
