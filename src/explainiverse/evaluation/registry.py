"""Validated discovery metadata for public evaluation endpoints.

The explainer registry distinguishes verified, quarantined, and unverified
implementations.  Evaluation functions need the same trust boundary: a public
function name alone does not say whether it is a paper transcription, a scoped
adaptation, or a library-defined diagnostic.

This module deliberately contains no imports from the metric implementation
modules.  ``evaluation.__init__`` builds the default registry only after all
public functions have been imported, avoiding circular imports while ensuring
that every exported ``compute_*`` endpoint is represented.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional


@dataclass(frozen=True)
class MetricMeta:
    """Audit and interpretation metadata for one evaluation endpoint.

    ``stochasticity`` describes randomness introduced or explicitly
    orchestrated by the endpoint itself:
    ``"deterministic"`` introduces none, ``"conditional"`` uses a random or
    potentially random branch only for some arguments/data, and
    ``"stochastic"`` always performs sampling/randomisation. It cannot certify
    hidden randomness inside arbitrary caller-supplied models, explainers, or
    callbacks beyond that declared protocol. The legacy ``stochastic`` boolean
    remains true for both non-deterministic categories.
    """

    family: str
    level: str
    claim_status: str
    claim_scope: str
    canonical_claim: bool = False
    score_direction: str = "contextual"
    stochastic: bool = False
    paper_reference: Optional[str] = None
    stochasticity: str = "deterministic"

    _VALID_LEVELS = {"instance", "batch", "dataset"}
    _VALID_STATUSES = {"verified", "quarantined", "unverified"}
    _VALID_DIRECTIONS = {"higher", "lower", "zero", "contextual", "none"}
    _VALID_STOCHASTICITY = {"deterministic", "conditional", "stochastic"}

    def __post_init__(self) -> None:
        for name in ("family", "claim_scope"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise TypeError(f"{name} must be a non-empty string")
        if self.level not in self._VALID_LEVELS:
            raise ValueError(f"level must be one of {sorted(self._VALID_LEVELS)}")
        if self.claim_status not in self._VALID_STATUSES:
            raise ValueError(f"claim_status must be one of {sorted(self._VALID_STATUSES)}")
        if self.score_direction not in self._VALID_DIRECTIONS:
            raise ValueError(f"score_direction must be one of {sorted(self._VALID_DIRECTIONS)}")
        for name in ("canonical_claim", "stochastic"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if self.stochasticity not in self._VALID_STOCHASTICITY:
            raise ValueError(f"stochasticity must be one of {sorted(self._VALID_STOCHASTICITY)}")
        # Preserve the historical boolean while exposing whether randomness is
        # unconditional or activated only by a mode/data-dependent branch.
        if self.stochastic and self.stochasticity == "deterministic":
            object.__setattr__(self, "stochasticity", "stochastic")
        elif not self.stochastic and self.stochasticity != "deterministic":
            raise ValueError("non-deterministic stochasticity requires stochastic=True")
        if self.paper_reference is not None and (
            not isinstance(self.paper_reference, str) or not self.paper_reference.strip()
        ):
            raise TypeError("paper_reference must be None or a non-empty string")
        if self.claim_status == "unverified" and self.canonical_claim:
            raise ValueError("an unverified endpoint cannot make a canonical claim")

    def matches(
        self,
        *,
        family: Optional[str] = None,
        level: Optional[str] = None,
        claim_status: Optional[str] = None,
        canonical_claim: Optional[bool] = None,
        stochasticity: Optional[str] = None,
    ) -> bool:
        """Return whether this entry matches validated discovery criteria."""

        if family is not None and (not isinstance(family, str) or not family.strip()):
            raise TypeError("family must be None or a non-empty string")
        if level is not None and level not in self._VALID_LEVELS:
            raise ValueError(f"level must be one of {sorted(self._VALID_LEVELS)}")
        if claim_status is not None and claim_status not in self._VALID_STATUSES:
            raise ValueError(f"claim_status must be one of {sorted(self._VALID_STATUSES)}")
        if canonical_claim is not None and not isinstance(canonical_claim, bool):
            raise TypeError("canonical_claim must be a boolean or None")
        if stochasticity is not None and stochasticity not in self._VALID_STOCHASTICITY:
            raise ValueError(
                f"stochasticity must be one of {sorted(self._VALID_STOCHASTICITY)} or None"
            )
        return (
            (family is None or self.family == family)
            and (level is None or self.level == level)
            and (claim_status is None or self.claim_status == claim_status)
            and (canonical_claim is None or self.canonical_claim == canonical_claim)
            and (stochasticity is None or self.stochasticity == stochasticity)
        )


class MetricRegistry:
    """Defensive registry of evaluation callables and their audit metadata."""

    def __init__(self) -> None:
        self._registry: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        metric_fn: Callable[..., Any],
        meta: MetricMeta,
        *,
        override: bool = False,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Metric name must be a non-empty string")
        if not callable(metric_fn):
            raise TypeError("metric_fn must be callable")
        if not isinstance(meta, MetricMeta):
            raise TypeError("meta must be a MetricMeta instance")
        if not isinstance(override, bool):
            raise TypeError("override must be a boolean")
        if name in self._registry and not override:
            raise ValueError(f"Metric {name!r} is already registered")

        # Reconstruct the frozen dataclass to validate even objects produced by
        # unusual deserialisation paths, then isolate the stored snapshot.
        validated = MetricMeta(**meta.__dict__)
        self._registry[name] = {"fn": metric_fn, "meta": copy.deepcopy(validated)}

    def unregister(self, name: str) -> None:
        if name not in self._registry:
            raise KeyError(f"Metric {name!r} is not registered")
        del self._registry[name]

    def get(self, name: str) -> Dict[str, Any]:
        if name not in self._registry:
            raise KeyError(f"Metric {name!r} is not registered. Available: {list(self._registry)}")
        entry = self._registry[name]
        return {"fn": entry["fn"], "meta": copy.deepcopy(entry["meta"])}

    def get_meta(self, name: str) -> MetricMeta:
        return self.get(name)["meta"]

    def list_metrics(self, *, with_meta: bool = False) -> Any:
        if not isinstance(with_meta, bool):
            raise TypeError("with_meta must be a boolean")
        if not with_meta:
            return list(self._registry)
        return {name: self.get(name) for name in self._registry}

    def filter(
        self,
        *,
        family: Optional[str] = None,
        level: Optional[str] = None,
        claim_status: Optional[str] = None,
        canonical_claim: Optional[bool] = None,
        stochasticity: Optional[str] = None,
    ) -> List[str]:
        # Validate even when the registry is empty.
        probe = MetricMeta(
            family="probe",
            level="instance",
            claim_status="unverified",
            claim_scope="probe",
        )
        probe.matches(
            family=family,
            level=level,
            claim_status=claim_status,
            canonical_claim=canonical_claim,
            stochasticity=stochasticity,
        )
        return [
            name
            for name, entry in self._registry.items()
            if entry["meta"].matches(
                family=family,
                level=level,
                claim_status=claim_status,
                canonical_claim=canonical_claim,
                stochasticity=stochasticity,
            )
        ]

    def evaluate(self, name: str, *args: Any, **kwargs: Any) -> Any:
        return self.get(name)["fn"](*args, **kwargs)

    def validate_inventory(self, public_names: Iterable[str]) -> None:
        """Require exact coverage of a supplied public ``compute_*`` inventory."""

        expected = {
            name for name in public_names if isinstance(name, str) and name.startswith("compute_")
        }
        actual = set(self._registry)
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing or unexpected:
            raise ValueError(
                "Metric registry inventory mismatch: " f"missing={missing}, unexpected={unexpected}"
            )

    def summary(self) -> str:
        counts: Dict[str, int] = {}
        for entry in self._registry.values():
            status = entry["meta"].claim_status
            counts[status] = counts.get(status, 0) + 1
        details = ", ".join(f"{key}={counts[key]}" for key in sorted(counts))
        return f"MetricRegistry(total={len(self._registry)}, {details})"


_FAMILY_BY_MODULE = {
    "agreement": "agreement",
    "axiomatic": "axiomatic",
    "complexity": "complexity",
    "fairness": "fairness-related",
    "faithfulness": "core faithfulness",
    "faithfulness_extended": "extended faithfulness",
    "localisation": "localisation",
    "metrics": "perturbation/retraining",
    "randomisation": "randomisation",
    "robustness": "robustness",
    "stability": "relative stability",
}


_ALL_REVIEWED_ENDPOINTS = (
    "compute_aopc",
    "compute_batch_aopc",
    "compute_roar",
    "compute_roar_curve",
    "compute_pgi",
    "compute_pgu",
    "compute_faithfulness_score",
    "compute_comprehensiveness",
    "compute_sufficiency",
    "compute_faithfulness_correlation",
    "compute_batch_faithfulness",
    "compute_ris",
    "compute_ros",
    "compute_lipschitz_estimate",
    "compute_stability_metrics",
    "compute_batch_stability",
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
    "compute_deletion_auc",
    "compute_batch_deletion_auc",
    "compute_insertion_auc",
    "compute_batch_insertion_auc",
    "compute_insertion_deletion_auc",
    "compute_max_sensitivity",
    "compute_batch_max_sensitivity",
    "compute_avg_sensitivity",
    "compute_batch_avg_sensitivity",
    "compute_continuity",
    "compute_batch_continuity",
    "compute_consistency",
    "compute_batch_consistency",
    "compute_relative_input_stability",
    "compute_batch_relative_input_stability",
    "compute_relative_representation_stability",
    "compute_batch_relative_representation_stability",
    "compute_relative_output_stability",
    "compute_batch_relative_output_stability",
    "compute_relative_stability",
    "compute_batch_relative_stability",
    "compute_feature_agreement",
    "compute_batch_feature_agreement",
    "compute_rank_agreement",
    "compute_batch_rank_agreement",
    "compute_sparseness",
    "compute_batch_sparseness",
    "compute_complexity",
    "compute_batch_complexity",
    "compute_attribution_threshold_count",
    "compute_batch_attribution_threshold_count",
    "compute_effective_complexity",
    "compute_batch_effective_complexity",
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
)

# Level describes the unit represented by one result, not a spelling pattern.
# In particular, aggregate randomisation tests and Consistency consume a cohort
# and are dataset-level even though some historical names omit/contain "batch".
_BATCH_ENDPOINTS = {
    "compute_batch_aopc",
    "compute_batch_faithfulness",
    "compute_batch_stability",
    "compute_batch_faithfulness_estimate",
    "compute_batch_monotonicity",
    "compute_batch_monotonicity_nguyen",
    "compute_batch_pixel_flipping",
    "compute_batch_region_perturbation",
    "compute_batch_selectivity",
    "compute_batch_sensitivity_n",
    "compute_batch_irof",
    "compute_batch_infidelity",
    "compute_batch_road",
    "compute_batch_deletion_auc",
    "compute_batch_insertion_auc",
    "compute_batch_max_sensitivity",
    "compute_batch_avg_sensitivity",
    "compute_batch_continuity",
    "compute_batch_relative_input_stability",
    "compute_batch_relative_representation_stability",
    "compute_batch_relative_output_stability",
    "compute_batch_relative_stability",
    "compute_batch_feature_agreement",
    "compute_batch_rank_agreement",
    "compute_batch_sparseness",
    "compute_batch_complexity",
    "compute_batch_attribution_threshold_count",
    "compute_batch_effective_complexity",
    "compute_batch_pointing_game",
    "compute_batch_attribution_localisation",
    "compute_batch_top_k_intersection",
    "compute_batch_relevance_mass_accuracy",
    "compute_batch_relevance_rank_accuracy",
    "compute_batch_auc",
    "compute_batch_energy_based_pointing_game",
    "compute_batch_focus",
    "compute_batch_attribution_iou",
    "compute_batch_mprt",
    "compute_batch_random_logit",
    "compute_batch_smooth_mprt",
    "compute_batch_efficient_mprt",
    "compute_batch_data_randomisation",
    "compute_batch_completeness",
    "compute_batch_non_sensitivity",
    "compute_batch_input_invariance",
    "compute_batch_input_invariance_pytorch",
    "compute_batch_symmetry",
    "compute_batch_group_fairness",
}

_DATASET_ENDPOINTS = {
    "compute_roar",
    "compute_roar_curve",
    "compute_consistency",
    "compute_batch_consistency",
    "compute_mprt",
    "compute_random_logit",
    "compute_smooth_mprt",
    "compute_efficient_mprt",
    "compute_data_randomisation",
    "compute_group_metric_disparity",
    "compute_group_fairness",
    "compute_group_fairness_score",
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
}

_STOCHASTIC_ENDPOINTS = {
    "compute_ris",
    "compute_ros",
    "compute_lipschitz_estimate",
    "compute_stability_metrics",
    "compute_batch_stability",
    "compute_faithfulness_estimate",
    "compute_batch_faithfulness_estimate",
    "compute_sensitivity_n",
    "compute_sensitivity_n_multi",
    "compute_batch_sensitivity_n",
    "compute_infidelity",
    "compute_infidelity_multi_perturbation",
    "compute_batch_infidelity",
    "compute_road",
    "compute_road_combined",
    "compute_batch_road",
    "compute_max_sensitivity",
    "compute_batch_max_sensitivity",
    "compute_avg_sensitivity",
    "compute_batch_avg_sensitivity",
    "compute_relative_input_stability",
    "compute_batch_relative_input_stability",
    "compute_relative_representation_stability",
    "compute_batch_relative_representation_stability",
    "compute_relative_output_stability",
    "compute_batch_relative_output_stability",
    "compute_relative_stability",
    "compute_batch_relative_stability",
    "compute_mprt",
    "compute_batch_mprt",
    "compute_random_logit",
    "compute_batch_random_logit",
    "compute_smooth_mprt",
    "compute_batch_smooth_mprt",
    "compute_efficient_mprt",
    "compute_batch_efficient_mprt",
}

_CONDITIONALLY_STOCHASTIC_ENDPOINTS = {
    "compute_roar",
    "compute_roar_curve",
    "compute_faithfulness_correlation",
    "compute_consistency",
    "compute_batch_consistency",
    "compute_input_invariance",
    "compute_input_invariance_pytorch",
    "compute_batch_input_invariance",
    "compute_batch_input_invariance_pytorch",
    "compute_cross_group_lipschitz_diagnostic",
    "compute_individual_fairness",
}

if len(_ALL_REVIEWED_ENDPOINTS) != len(set(_ALL_REVIEWED_ENDPOINTS)):
    raise RuntimeError("reviewed metric endpoint manifest contains duplicates")
if _BATCH_ENDPOINTS & _DATASET_ENDPOINTS:
    raise RuntimeError("a metric endpoint cannot have both batch and dataset level")
if _STOCHASTIC_ENDPOINTS & _CONDITIONALLY_STOCHASTIC_ENDPOINTS:
    raise RuntimeError("metric stochasticity classifications must be disjoint")
for _classification in (
    _BATCH_ENDPOINTS,
    _DATASET_ENDPOINTS,
    _STOCHASTIC_ENDPOINTS,
    _CONDITIONALLY_STOCHASTIC_ENDPOINTS,
):
    if not _classification <= set(_ALL_REVIEWED_ENDPOINTS):
        raise RuntimeError("metric behavior classification names an unreviewed endpoint")

_ENDPOINT_BEHAVIOR = {
    endpoint: (
        (
            "dataset"
            if endpoint in _DATASET_ENDPOINTS
            else "batch" if endpoint in _BATCH_ENDPOINTS else "instance"
        ),
        (
            "stochastic"
            if endpoint in _STOCHASTIC_ENDPOINTS
            else (
                "conditional"
                if endpoint in _CONDITIONALLY_STOCHASTIC_ENDPOINTS
                else "deterministic"
            )
        ),
    )
    for endpoint in _ALL_REVIEWED_ENDPOINTS
}


# Direction is intentionally assigned only where the scalar's mathematical
# ordering is unambiguous.  ``contextual`` is safer for structured returns,
# signed effects, adaptations, and diagnostics that are not quality rankings.
_LOWER_IS_BETTER = {
    "compute_deletion_auc",
    "compute_batch_deletion_auc",
    "compute_infidelity",
    "compute_infidelity_multi_perturbation",
    "compute_batch_infidelity",
    "compute_lipschitz_estimate",
    "compute_max_sensitivity",
    "compute_batch_max_sensitivity",
    "compute_avg_sensitivity",
    "compute_batch_avg_sensitivity",
    "compute_non_sensitivity",
    "compute_non_sensitivity_score",
    "compute_batch_non_sensitivity",
    "compute_symmetry",
    "compute_symmetry_score",
    "compute_batch_symmetry",
}


_HIGHER_IS_BETTER = {
    "compute_feature_agreement",
    "compute_batch_feature_agreement",
    "compute_rank_agreement",
    "compute_batch_rank_agreement",
    "compute_faithfulness_correlation",
    "compute_insertion_auc",
    "compute_batch_insertion_auc",
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
}


_NONCANONICAL_SCOPES: Mapping[str, str] = {
    "compute_attribution_iou": (
        "Library-defined attribution/mask IoU diagnostic under its documented "
        "binarisation contract; it is not a literature-canonical IoU attribution metric."
    ),
    "compute_batch_attribution_iou": (
        "Batch aggregation of the library-defined attribution/mask IoU diagnostic."
    ),
    "compute_avg_sensitivity": (
        "Library-defined finite-sample mean sensitivity heuristic; not Yeh et al.'s "
        "sampled maximum sensitivity."
    ),
    "compute_batch_avg_sensitivity": (
        "Batch aggregation of the library-defined mean sensitivity heuristic."
    ),
    "compute_effective_complexity": (
        "Compatibility alias for attribution threshold count; not a separate "
        "verified canonical method."
    ),
    "compute_batch_effective_complexity": (
        "Batch compatibility alias for attribution threshold count."
    ),
    "compute_pgi": (
        "Deterministic baseline-replacement PGI specialisation under an explicit "
        "fixed-target contract; not the OpenXAI noisy-expectation benchmark."
    ),
    "compute_pgu": (
        "Deterministic baseline-replacement PGU specialisation under an explicit "
        "fixed-target contract; not the OpenXAI noisy-expectation benchmark."
    ),
    "compute_batch_faithfulness": (
        "Batch aggregation of deterministic baseline-replacement PGI/PGU variants "
        "with an explicit independent background contract."
    ),
    "compute_aopc": (
        "Generalised signed MoRF AOPC formula under configurable feature-level "
        "replacement; not Samek et al.'s complete image-region experiment."
    ),
    "compute_batch_aopc": ("Batch aggregation of the generalised signed MoRF AOPC formula."),
    "compute_symmetry": (
        "Conditional attribution-disparity diagnostic for caller-certified symmetric "
        "feature pairs; it does not establish functional symmetry."
    ),
    "compute_batch_symmetry": (
        "Batch conditional disparity for caller-certified symmetric feature pairs."
    ),
}


_QUARANTINED_ENDPOINTS = {
    "compute_effective_complexity",
    "compute_batch_effective_complexity",
}


def infer_metric_meta(name: str, metric_fn: Callable[..., Any]) -> MetricMeta:
    """Create conservative, validated metadata for a public metric callable."""

    if not isinstance(name, str) or not name.startswith("compute_"):
        raise ValueError("metric names must start with 'compute_'")
    if not callable(metric_fn):
        raise TypeError("metric_fn must be callable")
    if name not in _ENDPOINT_BEHAVIOR:
        raise ValueError(
            f"Metric {name!r} has no reviewed level/stochasticity metadata; "
            "add it to the explicit endpoint manifest before registration"
        )
    module_name = getattr(metric_fn, "__module__", "").rsplit(".", 1)[-1]
    family = _FAMILY_BY_MODULE.get(module_name, "unclassified evaluation")
    level, stochasticity = _ENDPOINT_BEHAVIOR[name]

    if name in _NONCANONICAL_SCOPES:
        scope = _NONCANONICAL_SCOPES[name]
        canonical = False
    else:
        scope = (
            f"Verified {family} {level} endpoint only under the exact target, "
            "baseline, perturbation, aggregation, and unsupported-domain contract "
            "documented by the callable."
        )
        # Formula verification does not by itself establish that every default,
        # aggregation, or experimental protocol reproduces a paper wholesale.
        canonical = False

    status = "quarantined" if name in _QUARANTINED_ENDPOINTS else "verified"
    stochastic = stochasticity != "deterministic"
    if name in _LOWER_IS_BETTER:
        direction = "lower"
    elif name in _HIGHER_IS_BETTER:
        direction = "higher"
    else:
        direction = "contextual"

    return MetricMeta(
        family=family,
        level=level,
        claim_status=status,
        claim_scope=scope,
        canonical_claim=canonical and status == "verified",
        score_direction=direction,
        stochastic=stochastic,
        stochasticity=stochasticity,
    )


def build_metric_registry(
    namespace: Mapping[str, Any], public_names: Iterable[str]
) -> MetricRegistry:
    """Build and inventory-check a registry from an evaluation namespace."""

    registry = MetricRegistry()
    names = [name for name in public_names if isinstance(name, str) and name.startswith("compute_")]
    for name in names:
        if name not in namespace:
            raise ValueError(f"Public metric {name!r} is missing from the namespace")
    missing_review = sorted(set(names) - set(_ENDPOINT_BEHAVIOR))
    stale_review = sorted(set(_ENDPOINT_BEHAVIOR) - set(names))
    if missing_review or stale_review:
        raise ValueError(
            "Reviewed metric metadata inventory mismatch: "
            f"missing_review={missing_review}, stale_review={stale_review}"
        )
    for name in names:
        metric_fn = namespace[name]
        registry.register(name, metric_fn, infer_metric_meta(name, metric_fn))
    registry.validate_inventory(names)
    return registry
