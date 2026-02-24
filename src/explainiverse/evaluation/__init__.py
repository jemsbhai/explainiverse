# src/explainiverse/evaluation/__init__.py
"""
Evaluation metrics for explanation quality.

Includes:
- Faithfulness metrics (PGI, PGU, Comprehensiveness, Sufficiency, Faithfulness Estimate)
- Stability metrics (RIS, ROS, Lipschitz)
- Perturbation metrics (AOPC, ROAR)
- Extended faithfulness metrics (Phase 1 expansion)
- Insertion/Deletion AUC (Petsiuk et al., 2018)
- Robustness metrics (Max-Sensitivity, Avg-Sensitivity, Continuity, Consistency,
  Relative Input/Representation/Output Stability) — Phase 2
- Complexity metrics (Sparseness, Complexity, Effective Complexity) — Phase 4
- Agreement metrics (Feature Agreement, Rank Agreement) — Phase 2
- Localisation metrics (Pointing Game, Attribution Localisation, Top-K Intersection,
  Relevance Mass Accuracy, Relevance Rank Accuracy, AUC, Energy-Based Pointing Game,
  Focus, Attribution IoU) — Phase 3
- Randomisation metrics (MPRT, Random Logit, Smooth MPRT, Efficient MPRT,
  Data Randomisation) — Phase 5
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
    compute_consistency,
    compute_batch_consistency,
    compute_relative_input_stability,
    compute_batch_relative_input_stability,
    compute_relative_representation_stability,
    compute_batch_relative_representation_stability,
    compute_relative_output_stability,
    compute_batch_relative_output_stability,
    compute_relative_stability,
    compute_batch_relative_stability,
)

from explainiverse.evaluation.agreement import (
    compute_feature_agreement,
    compute_batch_feature_agreement,
    compute_rank_agreement,
    compute_batch_rank_agreement,
)

from explainiverse.evaluation.complexity import (
    compute_sparseness,
    compute_batch_sparseness,
    compute_complexity,
    compute_batch_complexity,
    compute_effective_complexity,
    compute_batch_effective_complexity,
)

from explainiverse.evaluation.randomisation import (
    compute_mprt,
    compute_mprt_score,
    compute_batch_mprt,
    compute_random_logit,
    compute_random_logit_score,
    compute_batch_random_logit,
    compute_smooth_mprt,
    compute_batch_smooth_mprt,
    compute_efficient_mprt,
    compute_batch_efficient_mprt,
    compute_data_randomisation,
    compute_data_randomisation_score,
    compute_batch_data_randomisation,
)

from explainiverse.evaluation.localisation import (
    LocalisationMask,
    compute_pointing_game,
    compute_batch_pointing_game,
    compute_attribution_localisation,
    compute_batch_attribution_localisation,
    compute_top_k_intersection,
    compute_batch_top_k_intersection,
    compute_relevance_mass_accuracy,
    compute_batch_relevance_mass_accuracy,
    compute_relevance_rank_accuracy,
    compute_batch_relevance_rank_accuracy,
    compute_auc,
    compute_batch_auc,
    compute_energy_based_pointing_game,
    compute_batch_energy_based_pointing_game,
    compute_focus,
    compute_batch_focus,
    compute_attribution_iou,
    compute_batch_attribution_iou,
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
    "compute_consistency",
    "compute_batch_consistency",
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
    # Complexity metrics (Phase 4)
    "compute_sparseness",
    "compute_batch_sparseness",
    "compute_complexity",
    "compute_batch_complexity",
    "compute_effective_complexity",
    "compute_batch_effective_complexity",
    # Localisation metrics (Phase 3)
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
    # Randomisation metrics (Phase 5)
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
]
