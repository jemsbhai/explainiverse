"""Group-disparity diagnostics for explanation evaluations.

This module intentionally distinguishes measurable disparities from fairness
verdicts.  A gap in a scalar explanation property can be evidence for an
audit, but it does not by itself establish that an explainer or model is fair.

The canonical paper-backed implementation in this module is
``compute_fidelity_gap``.  It implements Balagopalan et al. (2022),
Definitions 3.3 and 3.4, for *supplied per-instance fidelity scores*.  The
remaining functions are clearly labelled diagnostics or compatibility names:

* ``compute_group_metric_disparity`` applies the group-comparison framework of
  Dai et al. (2022) to a caller-selected scalar property.  Its default is only
  attribution L1 magnitude, not a validated explanation-quality measure.
* the cross-group Lipschitz, sensitive-intervention, sensitive-attribution,
  and prediction-conditioned functions are useful audit probes.  They are not
  implementations of Dwork individual fairness, Kusner counterfactual
  fairness, or Hardt equality of opportunity.

References
----------
Dai, J., Upadhyay, S., Aivodji, U., Bach, S. H., & Lakkaraju, H. (2022).
Fairness via Explanation Quality: Evaluating Disparities in the Quality of
Post hoc Explanations. AIES. https://doi.org/10.1145/3514094.3534159

Balagopalan, A., Zhang, H., Hamidieh, K., Hartvigsen, T., Rudzicz, F., &
Ghassemi, M. (2022). The Road to Explainability is Paved with Bias: Measuring
the Fairness of Explanations. FAccT.
https://doi.org/10.1145/3531146.3533179
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation
from explainiverse.evaluation._utils import (
    _exact_percentile,
    _percentile_mask,
    _stable_difference_of_means,
    _stable_mean,
    _stable_sum,
)

ScalarMetric = Callable[[np.ndarray], float]


def _attribution_l1_magnitude(attr_vector: np.ndarray) -> float:
    """Return attribution L1 magnitude; this is not a quality or fairness score."""
    result = float(_stable_sum(np.abs(attr_vector)))
    if not np.isfinite(result):
        raise FloatingPointError("attribution L1 magnitude is not representable")
    return result


def _finite_mean(values: np.ndarray, context: str) -> float:
    """Return a cancellation-safe mean or fail when it cannot be represented."""
    result = float(_stable_mean(np.asarray(values, dtype=np.float64)))
    if not np.isfinite(result):
        raise FloatingPointError(f"{context} mean is not representable")
    return result


def _finite_difference(left: float, right: float, context: str) -> float:
    """Subtract two finite scalars without overflowing an intermediate sum."""
    result = float(_stable_sum(np.asarray([left, -right], dtype=np.float64)))
    if not np.isfinite(result):
        raise FloatingPointError(f"{context} is not representable")
    return result


def _finite_gap(left: float, right: float, context: str) -> float:
    return abs(_finite_difference(left, right, context))


def _finite_vector_difference(left: np.ndarray, right: np.ndarray, context: str) -> np.ndarray:
    """Subtract finite vectors and reject a genuinely out-of-range coordinate."""
    with np.errstate(over="ignore", invalid="ignore"):
        difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    if not np.all(np.isfinite(difference)):
        raise FloatingPointError(f"{context} vector difference is not representable")
    return difference


def _finite_l2_norm(values: np.ndarray, context: str) -> float:
    """Evaluate a Euclidean norm without overflow or tiny-value underflow."""
    vector = np.asarray(values, dtype=np.float64)
    scale = float(np.max(np.abs(vector)))
    if scale == 0.0:
        return 0.0
    scaled = vector / scale
    unit_norm = float(np.sqrt(np.dot(scaled, scaled)))
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(scale * unit_norm)
    if not np.isfinite(result):
        raise FloatingPointError(f"{context} Euclidean norm is not representable")
    return result


# Kept private-name compatible for callers that imported it despite the underscore.
_default_inner_metric = _attribution_l1_magnitude


def _validate_matrix(values: Any, name: str) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a numeric two-dimensional array.")
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be convertible to a numeric array: {exc}") from exc
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {array.shape}.")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    if array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validate_attributions(attributions: Any) -> np.ndarray:
    return _validate_matrix(attributions, "attributions")


def _normalise_label(value: Any, name: str) -> Any:
    label = value.item() if isinstance(value, np.generic) else value
    if label is None:
        raise ValueError(f"{name} must not contain missing labels.")
    if isinstance(label, (float, np.floating)) and not np.isfinite(label):
        raise ValueError(f"{name} must not contain NaN or infinite labels.")
    try:
        hash(label)
    except TypeError as exc:
        raise TypeError(f"Every {name} label must be hashable, got {label!r}.") from exc
    return label


def _validate_labels(labels: Any, n_samples: int, name: str) -> np.ndarray:
    if isinstance(labels, (str, bytes)):
        raise TypeError(f"{name} must be a one-dimensional array of labels.")
    try:
        array = np.asarray(labels, dtype=object)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be array-like: {exc}") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {array.shape}.")
    if len(array) != n_samples:
        raise ValueError(f"{name} length ({len(array)}) does not match sample count ({n_samples}).")
    if len(array) == 0:
        raise ValueError(f"{name} must not be empty.")
    return np.asarray([_normalise_label(v, name) for v in array], dtype=object)


def _validate_sensitive_features(sensitive_features: Any, n_samples: int) -> np.ndarray:
    return _validate_labels(sensitive_features, n_samples, "sensitive_features")


def _partition_by_group(labels: np.ndarray) -> Dict[Any, np.ndarray]:
    groups: Dict[Any, List[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)
    return {label: np.asarray(indices, dtype=np.int64) for label, indices in groups.items()}


def _require_multiple_groups(groups: Dict[Any, np.ndarray], context: str) -> None:
    if len(groups) < 2:
        raise ValueError(
            f"{context} requires at least two observed groups; a one-group sample "
            "cannot identify a between-group disparity."
        )


def _validate_scores(scores: Any, n_samples: int, name: str) -> np.ndarray:
    if isinstance(scores, (str, bytes)):
        raise TypeError(f"{name} must be a numeric one-dimensional array.")
    try:
        array = np.asarray(scores, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric: {exc}") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {array.shape}.")
    if len(array) != n_samples:
        raise ValueError(f"{name} length ({len(array)}) does not match sample count ({n_samples}).")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _score_rows(attributions: np.ndarray, metric: ScalarMetric) -> np.ndarray:
    if not callable(metric):
        raise TypeError("inner_metric must be callable.")
    scores: List[float] = []
    for index, row in enumerate(attributions):
        try:
            raw_score = metric(row.copy())
        except Exception as exc:
            raise ValueError(f"inner_metric failed for row {index}: {exc}") from exc
        value = np.asarray(raw_score)
        if value.ndim != 0 or value.dtype.kind not in "iuf":
            raise TypeError(
                "inner_metric must return one real scalar per row; "
                f"row {index} returned shape {value.shape}."
            )
        score = float(value)
        if not np.isfinite(score):
            raise ValueError(f"inner_metric returned a non-finite value for row {index}.")
        scores.append(score)
    return np.asarray(scores, dtype=np.float64)


def _mann_whitney_u(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Return a two-sided Mann-Whitney U p-value when it is defined.

    If every pooled observation is identical, the tie-corrected asymptotic
    variance is zero.  SciPy versions have returned either ``1.0`` or ``NaN``
    for that degenerate case; this API reports ``None`` instead of presenting a
    version-dependent or undefined value as statistical evidence.
    """
    if len(a) == 0 or len(b) == 0:
        raise ValueError("Mann-Whitney U requires two non-empty samples.")
    pooled = np.concatenate((a, b))
    if np.all(pooled == pooled[0]):
        return None
    from scipy.stats import mannwhitneyu

    result = mannwhitneyu(a, b, alternative="two-sided")
    p_value = float(result.pvalue)
    if not np.isfinite(p_value):
        raise ValueError("Mann-Whitney U returned a non-finite p-value.")
    return p_value


def _cohens_d(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Return pooled Cohen's d, or ``None`` when sample variance is unavailable."""
    if len(a) < 2 or len(b) < 2:
        return None
    with localcontext() as context:
        context.prec = 3000 + len(str(len(a) + len(b)))
        decimal_a = [Decimal.from_float(float(value)) for value in a]
        decimal_b = [Decimal.from_float(float(value)) for value in b]
        mean_a = sum(decimal_a, start=Decimal(0)) / Decimal(len(decimal_a))
        mean_b = sum(decimal_b, start=Decimal(0)) / Decimal(len(decimal_b))
        mean_difference = mean_a - mean_b
        pooled_sum_squares = sum(
            ((value - mean_a) * (value - mean_a) for value in decimal_a),
            start=Decimal(0),
        ) + sum(
            ((value - mean_b) * (value - mean_b) for value in decimal_b),
            start=Decimal(0),
        )
        pooled_variance = pooled_sum_squares / Decimal(len(a) + len(b) - 2)
        if pooled_variance == 0:
            if mean_difference == 0:
                return 0.0
            return float(np.copysign(np.inf, float(mean_difference)))
        exact = mean_difference / pooled_variance.sqrt()
        result = float(exact)
    if not np.isfinite(result) or (result == 0.0 and exact != 0):
        raise FloatingPointError("Cohen's d is not representable")
    return result


def _group_statistics(
    scores: np.ndarray,
    groups: Dict[Any, np.ndarray],
) -> Dict[str, Any]:
    group_scores = {label: scores[indices] for label, indices in groups.items()}
    group_means = {
        label: _finite_mean(values, f"group {label!r}") for label, values in group_scores.items()
    }
    pairwise_gaps: Dict[Tuple[Any, Any], float] = {}
    pairwise_p_values: Dict[Tuple[Any, Any], Optional[float]] = {}
    pairwise_effect_sizes: Dict[Tuple[Any, Any], Optional[float]] = {}
    for left, right in combinations(groups, 2):
        pair = (left, right)
        pairwise_gaps[pair] = abs(
            _stable_difference_of_means(group_scores[left], group_scores[right])
        )
        pairwise_p_values[pair] = _mann_whitney_u(group_scores[left], group_scores[right])
        pairwise_effect_sizes[pair] = _cohens_d(group_scores[left], group_scores[right])

    effects = [abs(value) for value in pairwise_effect_sizes.values() if value is not None]
    available_p_values = [value for value in pairwise_p_values.values() if value is not None]
    unavailable_p_value_pairs = [pair for pair, value in pairwise_p_values.items() if value is None]
    return {
        "disparity": float(max(pairwise_gaps.values())),
        "group_means": group_means,
        "pairwise_gaps": pairwise_gaps,
        "pairwise_p_values": pairwise_p_values,
        "pairwise_effect_sizes": pairwise_effect_sizes,
        # Compatibility summaries.  For >2 groups, p_value is explicitly an
        # uncorrected minimum and must not be treated as a family-wise p-value.
        "p_value": float(min(available_p_values)) if available_p_values else None,
        "p_value_unavailable_pairs": unavailable_p_value_pairs,
        "effect_size": float(max(effects)) if effects else None,
        "p_value_adjustment": "none",
        "p_value_summary": (
            "two-sided Mann-Whitney U; None marks completely tied pooled samples"
            if len(groups) == 2
            else (
                "minimum available uncorrected pairwise two-sided Mann-Whitney U; "
                "None marks completely tied pooled samples"
            )
        ),
    }


_FAIRNESS_LEVELS = frozenset({"group", "individual", "conditional"})


@dataclass(frozen=True)
class FairnessMetricMeta:
    """Discovery metadata for a registered disparity diagnostic."""

    level: str
    composable: bool = False
    description: str = ""
    paper_reference: Optional[str] = None
    canonical_claim: bool = False
    claim_scope: str = "diagnostic only"

    def __post_init__(self) -> None:
        if not isinstance(self.level, str) or self.level.strip() not in _FAIRNESS_LEVELS:
            raise ValueError(f"level must be one of {sorted(_FAIRNESS_LEVELS)}, got {self.level!r}")
        if not isinstance(self.composable, bool):
            raise TypeError("composable must be bool")
        if not isinstance(self.canonical_claim, bool):
            raise TypeError("canonical_claim must be bool")
        if not isinstance(self.description, str):
            raise TypeError("description must be a string")
        if self.paper_reference is not None and (
            not isinstance(self.paper_reference, str) or not self.paper_reference.strip()
        ):
            raise ValueError("paper_reference must be a non-empty string or None")
        if not isinstance(self.claim_scope, str) or not self.claim_scope.strip():
            raise ValueError("claim_scope must be a non-empty string")
        object.__setattr__(self, "level", self.level.strip())
        object.__setattr__(self, "description", self.description.strip())
        object.__setattr__(self, "claim_scope", self.claim_scope.strip())
        if self.paper_reference is not None:
            object.__setattr__(self, "paper_reference", self.paper_reference.strip())

    def matches(self, level: Optional[str] = None) -> bool:
        return level is None or self.level == level


class FairnessMetricRegistry:
    """Registry for fairness-related metrics and disparity diagnostics."""

    def __init__(self) -> None:
        self._registry: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _validated_name(name: str) -> str:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Metric name must be a non-empty string.")
        return name.strip()

    def register(
        self,
        name: str,
        metric_fn: Callable[..., Dict[str, Any]],
        meta: FairnessMetricMeta,
        override: bool = False,
    ) -> None:
        name = self._validated_name(name)
        if not callable(metric_fn):
            raise TypeError("metric_fn must be callable.")
        if not isinstance(meta, FairnessMetricMeta):
            raise TypeError("meta must be a FairnessMetricMeta instance.")
        if not isinstance(override, bool):
            raise TypeError("override must be bool.")
        if name in self._registry and not override:
            raise ValueError(
                f"Fairness metric '{name}' is already registered. " "Use override=True to replace."
            )
        self._registry[name] = {"fn": metric_fn, "meta": meta}

    def unregister(self, name: str) -> None:
        name = self._validated_name(name)
        if name not in self._registry:
            raise KeyError(f"Fairness metric '{name}' is not registered.")
        del self._registry[name]

    def get(self, name: str) -> Dict[str, Any]:
        name = self._validated_name(name)
        if name not in self._registry:
            raise KeyError(
                f"Fairness metric '{name}' is not registered. " f"Available: {list(self._registry)}"
            )
        # Return a detached entry. The metadata itself is frozen, so callers
        # cannot mutate registry discovery contracts through either surface.
        entry = self._registry[name]
        return {"fn": entry["fn"], "meta": entry["meta"]}

    def get_meta(self, name: str) -> FairnessMetricMeta:
        return self.get(name)["meta"]

    def list_metrics(self, with_meta: bool = False) -> Any:
        if not isinstance(with_meta, bool):
            raise TypeError("with_meta must be bool")
        return (
            {
                name: {"fn": entry["fn"], "meta": entry["meta"]}
                for name, entry in self._registry.items()
            }
            if with_meta
            else list(self._registry)
        )

    def filter(self, level: Optional[str] = None) -> List[str]:
        if level is not None and level not in _FAIRNESS_LEVELS:
            raise ValueError(f"level must be one of {sorted(_FAIRNESS_LEVELS)} or None")
        return [
            name for name, entry in self._registry.items() if entry["meta"].matches(level=level)
        ]

    def evaluate(
        self,
        name: str,
        attributions: np.ndarray,
        sensitive_features: np.ndarray,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self.get(name)["fn"](attributions, sensitive_features, **kwargs)

    def register_decorator(self, name: str, meta: FairnessMetricMeta) -> Callable:
        def decorator(function: Callable) -> Callable:
            self.register(name, function, meta)
            return function

        return decorator

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "Explainiverse - Registered Fairness-Related Diagnostics",
            "=" * 60,
            "",
        ]
        for name, entry in self._registry.items():
            meta: FairnessMetricMeta = entry["meta"]
            lines.append(f"  {name} [{meta.level}]: {meta.description or '(no description)'}")
        lines.extend(["", f"Total: {len(self._registry)} metrics", "=" * 60])
        return "\n".join(lines)


def compute_group_metric_disparity(
    attributions: Any,
    sensitive_features: Any,
    inner_metric: Optional[ScalarMetric] = None,
) -> Dict[str, Any]:
    """Compare a scalar attribution property across observed groups.

    This follows the compositional audit pattern of Dai et al. (2022), but the
    interpretation is only as valid as ``inner_metric``.  When omitted, the
    function compares attribution L1 magnitude and makes no explanation-quality
    or fairness claim.
    """
    attrs = _validate_attributions(attributions)
    labels = _validate_sensitive_features(sensitive_features, len(attrs))
    groups = _partition_by_group(labels)
    _require_multiple_groups(groups, "Group metric disparity")

    supplied_metric = inner_metric is not None
    metric: ScalarMetric = inner_metric if inner_metric is not None else _attribution_l1_magnitude
    scores = _score_rows(attrs, metric)
    result = _group_statistics(scores, groups)
    result.update(
        {
            "metric_name": (
                getattr(metric, "__name__", "custom_metric")
                if supplied_metric
                else "attribution_l1_magnitude"
            ),
            "inner_metric_supplied": supplied_metric,
            "canonical_explanation_quality": False,
            "canonical_fairness_metric": False,
            "interpretation": (
                "Between-group difference in the selected scalar property; "
                "not a standalone fairness verdict."
            ),
        }
    )
    return result


def compute_group_fairness(
    attributions: Any,
    sensitive_features: Any,
    inner_metric: Optional[ScalarMetric] = None,
) -> Dict[str, Any]:
    """Compatibility name for :func:`compute_group_metric_disparity`.

    The return value is a disparity diagnostic, not a binary or normative
    determination of fairness.
    """
    return compute_group_metric_disparity(attributions, sensitive_features, inner_metric)


def _extract_tabular_attribution(
    explanation: Explanation,
    n_features: int,
    expected_names: Optional[Tuple[str, ...]],
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    if not isinstance(explanation, Explanation):
        raise TypeError("explainer.explain() must return an Explanation instance.")
    mapping = explanation.explanation_data.get("feature_attributions")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Explanation must contain a non-empty feature_attributions mapping.")
    if explanation.feature_names is None:
        raise ValueError(
            "Explanation.feature_names is required to align tabular attributions safely."
        )
    names = tuple(explanation.feature_names)
    if len(names) != n_features or len(set(names)) != n_features:
        raise ValueError(
            "Explanation.feature_names must contain one unique name per input feature."
        )
    if set(mapping) != set(names):
        raise ValueError("feature_attributions keys must match Explanation.feature_names exactly.")
    if expected_names is not None and names != expected_names:
        raise ValueError("Explanation feature order changed between samples.")
    try:
        vector = np.asarray([mapping[name] for name in names], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Attribution values must be numeric: {exc}") from exc
    if not np.all(np.isfinite(vector)):
        raise ValueError("Explanation attributions must be finite.")
    return vector, names


def compute_group_fairness_score(
    explainer: BaseExplainer,
    inputs: Any,
    sensitive_features: Any,
    inner_metric: Optional[ScalarMetric] = None,
) -> Dict[str, Any]:
    """Generate aligned tabular attributions, then compute group disparity."""
    data = _validate_matrix(inputs, "inputs")
    rows: List[np.ndarray] = []
    expected_names: Optional[Tuple[str, ...]] = None
    for row in data:
        explanation = explainer.explain(row.copy())
        vector, expected_names = _extract_tabular_attribution(
            explanation, data.shape[1], expected_names
        )
        rows.append(vector)
    return compute_group_metric_disparity(np.vstack(rows), sensitive_features, inner_metric)


def compute_batch_group_fairness(
    batch_attributions: List[np.ndarray],
    batch_sensitive_features: List[Any],
    inner_metric: Optional[ScalarMetric] = None,
) -> List[Dict[str, Any]]:
    """Compute group scalar disparities for equally sized batch lists."""
    if len(batch_attributions) != len(batch_sensitive_features):
        raise ValueError("batch_attributions and batch_sensitive_features must have equal lengths.")
    if len(batch_attributions) == 0:
        raise ValueError("Batch inputs must not be empty.")
    return [
        compute_group_metric_disparity(attrs, labels, inner_metric)
        for attrs, labels in zip(batch_attributions, batch_sensitive_features)
    ]


def compute_cross_group_lipschitz_diagnostic(
    inputs: Any,
    attributions: Any,
    sensitive_features: Any,
    distance_threshold: Optional[float] = None,
    n_pairs: int = 500,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Measure attribution/input distance ratios for nearby cross-group pairs.

    Distances depend on feature scaling and the chosen representation.  This
    empirical ratio is therefore not Dwork et al.'s task-specific individual
    fairness guarantee.
    """
    data = _validate_matrix(inputs, "inputs")
    attrs = _validate_attributions(attributions)
    if len(data) != len(attrs):
        raise ValueError("inputs and attributions must have the same row count.")
    labels = _validate_sensitive_features(sensitive_features, len(data))
    groups = _partition_by_group(labels)
    _require_multiple_groups(groups, "Cross-group Lipschitz diagnostic")
    if isinstance(n_pairs, (bool, np.bool_)) or not isinstance(n_pairs, (int, np.integer)):
        raise TypeError("n_pairs must be a positive integer.")
    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive.")
    if isinstance(random_state, (bool, np.bool_)) or not isinstance(
        random_state, (int, np.integer)
    ):
        raise TypeError("random_state must be an integer.")
    if distance_threshold is not None:
        if isinstance(distance_threshold, (bool, np.bool_)):
            raise TypeError("distance_threshold must be a non-negative finite number.")
        distance_threshold = float(distance_threshold)
        if not np.isfinite(distance_threshold) or distance_threshold < 0:
            raise ValueError("distance_threshold must be non-negative and finite.")

    pairs: List[Tuple[int, int]] = []
    for left, right in combinations(groups, 2):
        pairs.extend((int(i), int(j)) for i in groups[left] for j in groups[right])
    feature_distances = np.asarray(
        [
            _finite_l2_norm(
                _finite_vector_difference(data[i], data[j], "cross-group input"),
                "cross-group input",
            )
            for i, j in pairs
        ],
        dtype=np.float64,
    )
    attribution_distances = np.asarray(
        [
            _finite_l2_norm(
                _finite_vector_difference(attrs[i], attrs[j], "cross-group attribution"),
                "cross-group attribution",
            )
            for i, j in pairs
        ],
        dtype=np.float64,
    )
    if distance_threshold is None:
        exact_threshold = _exact_percentile(feature_distances, 25.0)
        selected_threshold = float(exact_threshold)
        qualifying = np.flatnonzero(
            _percentile_mask(feature_distances, 25.0, comparison="at_or_below")
        )
        threshold_decimal = str(exact_threshold)
    else:
        selected_threshold = distance_threshold
        qualifying = np.flatnonzero(feature_distances <= selected_threshold)
        threshold_decimal = None
    if len(qualifying) == 0:
        raise ValueError(
            "No cross-group pairs satisfy distance_threshold; the requested local "
            "diagnostic is not estimable."
        )
    if len(qualifying) > n_pairs:
        rng = np.random.default_rng(int(random_state))
        qualifying = np.sort(rng.choice(qualifying, size=n_pairs, replace=False))

    feature_selected = feature_distances[qualifying]
    attribution_selected = attribution_distances[qualifying]
    ratios = np.empty_like(feature_selected)
    zero_distance = feature_selected == 0.0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ratios[~zero_distance] = (
            attribution_selected[~zero_distance] / feature_selected[~zero_distance]
        )
    if not np.all(np.isfinite(ratios[~zero_distance])):
        raise FloatingPointError("cross-group attribution/input ratio is not representable")
    ratios[zero_distance & (attribution_selected == 0.0)] = 0.0
    ratios[zero_distance & (attribution_selected > 0.0)] = np.inf

    mean_ratio = (
        float(np.inf) if np.any(np.isinf(ratios)) else _finite_mean(ratios, "cross-group ratio")
    )

    return {
        "score": mean_ratio,
        "max_ratio": float(np.max(ratios)),
        "n_pairs_evaluated": int(len(qualifying)),
        "distance_threshold": selected_threshold,
        "distance_threshold_exact_decimal": threshold_decimal,
        "canonical_individual_fairness": False,
        "interpretation": (
            "Cross-group attribution sensitivity in the supplied, scale-dependent "
            "input and attribution representations."
        ),
    }


def compute_individual_fairness(
    inputs: Any,
    attributions: Any,
    sensitive_features: Any,
    distance_threshold: Optional[float] = None,
    n_pairs: int = 500,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Compatibility name for a cross-group Lipschitz diagnostic."""
    return compute_cross_group_lipschitz_diagnostic(
        inputs,
        attributions,
        sensitive_features,
        distance_threshold,
        n_pairs,
        random_state,
    )


def compute_sensitive_attribution_change(
    inputs: Any,
    attributions: Any,
    sensitive_feature_idx: int,
    counterfactual_explainer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, Any]:
    """Measure attribution change under an intervention or matched-group proxy.

    Changing one observed column while holding all others fixed is not a
    structural-causal counterfactual.  Nearest-opposite-group matching is
    observational rather than counterfactual.  Both modes are reported as
    diagnostics, never as Kusner counterfactual fairness.
    """
    data = _validate_matrix(inputs, "inputs")
    attrs = _validate_attributions(attributions)
    if len(data) != len(attrs):
        raise ValueError("inputs and attributions must have the same row count.")
    if isinstance(sensitive_feature_idx, (bool, np.bool_)) or not isinstance(
        sensitive_feature_idx, (int, np.integer)
    ):
        raise TypeError("sensitive_feature_idx must be an integer.")
    index = int(sensitive_feature_idx)
    if index < 0 or index >= data.shape[1]:
        raise ValueError(
            f"sensitive_feature_idx {index} is out of bounds for {data.shape[1]} features."
        )
    if counterfactual_explainer is not None and not callable(counterfactual_explainer):
        raise TypeError("counterfactual_explainer must be callable.")

    sensitive_values = np.unique(data[:, index])
    if len(sensitive_values) != 2:
        raise ValueError(
            "Sensitive-attribution change requires exactly two observed sensitive "
            "values; it cannot be computed for a one-group or ambiguous multi-group sample."
        )

    per_instance: List[float] = []
    match_distances: Optional[List[float]] = None
    if counterfactual_explainer is not None:
        if set(sensitive_values.tolist()) != {0.0, 1.0}:
            raise ValueError(
                "One-feature intervention currently requires binary values encoded as 0 and 1."
            )
        for row_index, row in enumerate(data):
            intervened = row.copy()
            intervened[index] = 1.0 - intervened[index]
            try:
                changed = np.asarray(counterfactual_explainer(intervened), dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"counterfactual_explainer returned invalid values for row {row_index}: {exc}"
                ) from exc
            if changed.shape != (attrs.shape[1],):
                raise ValueError(
                    "counterfactual_explainer must return one attribution vector of "
                    f"shape {(attrs.shape[1],)}, got {changed.shape}."
                )
            if not np.all(np.isfinite(changed)):
                raise ValueError("counterfactual_explainer returned non-finite attributions.")
            per_instance.append(
                _finite_l2_norm(
                    _finite_vector_difference(
                        attrs[row_index], changed, "counterfactual attribution"
                    ),
                    "counterfactual attribution",
                )
            )
        method = "one_feature_intervention"
    else:
        non_sensitive = np.delete(data, index, axis=1)
        match_distances = []
        sensitive_column = data[:, index]
        for row_index, value in enumerate(sensitive_column):
            candidates = np.flatnonzero(sensitive_column != value)
            distances = np.asarray(
                [
                    _finite_l2_norm(
                        _finite_vector_difference(
                            non_sensitive[candidate],
                            non_sensitive[row_index],
                            "counterfactual matching input",
                        ),
                        "counterfactual matching input",
                    )
                    for candidate in candidates
                ],
                dtype=np.float64,
            )
            match_offset = int(np.argmin(distances))
            match_index = int(candidates[match_offset])
            match_distances.append(float(distances[match_offset]))
            per_instance.append(
                _finite_l2_norm(
                    _finite_vector_difference(
                        attrs[row_index],
                        attrs[match_index],
                        "counterfactual matching attribution",
                    ),
                    "counterfactual matching attribution",
                )
            )
        method = "nearest_opposite_group_matching"

    return {
        "score": _finite_mean(np.asarray(per_instance), "sensitive-attribution change"),
        "per_instance_scores": per_instance,
        "method": method,
        "match_distances": match_distances,
        "canonical_counterfactual_fairness": False,
        "requires_structural_causal_model_for_counterfactual_claim": True,
        "interpretation": "Attribution-change diagnostic; lower change is not proof of fairness.",
    }


def compute_counterfactual_fairness(
    inputs: Any,
    attributions: Any,
    sensitive_feature_idx: int,
    counterfactual_explainer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, Any]:
    """Compatibility name for :func:`compute_sensitive_attribution_change`."""
    return compute_sensitive_attribution_change(
        inputs, attributions, sensitive_feature_idx, counterfactual_explainer
    )


def compute_fidelity_gap(
    fidelity_scores: Any,
    sensitive_features: Any,
    *,
    higher_is_better: bool = True,
) -> Dict[str, Any]:
    """Compute Balagopalan et al. fidelity gaps from per-instance scores.

    ``max_gap_from_average`` implements Definition 3.3.  With
    ``higher_is_better=True`` it is ``max(overall_mean - group_mean)``; for an
    error/loss where lower is better the direction is reversed.
    ``mean_group_gap`` implements Definition 3.4, the mean absolute difference
    over unordered subgroup-mean pairs.
    """
    if not isinstance(higher_is_better, (bool, np.bool_)):
        raise TypeError("higher_is_better must be a boolean.")
    raw_scores = np.asarray(fidelity_scores)
    if raw_scores.ndim != 1:
        raise ValueError(f"fidelity_scores must be 1D, got shape {raw_scores.shape}.")
    scores = _validate_scores(raw_scores, len(raw_scores), "fidelity_scores")
    labels = _validate_sensitive_features(sensitive_features, len(scores))
    groups = _partition_by_group(labels)
    _require_multiple_groups(groups, "Fidelity gap")

    group_scores = {label: scores[indices] for label, indices in groups.items()}
    group_means = {
        label: _finite_mean(values, f"fidelity group {label!r}")
        for label, values in group_scores.items()
    }
    overall_mean = _finite_mean(scores, "overall fidelity")
    if higher_is_better:
        deficits = {
            label: _finite_difference(overall_mean, mean, f"fidelity deficit for group {label!r}")
            for label, mean in group_means.items()
        }
    else:
        deficits = {
            label: _finite_difference(mean, overall_mean, f"fidelity deficit for group {label!r}")
            for label, mean in group_means.items()
        }
    max_gap_from_average = float(max(deficits.values()))

    pairwise_gaps: Dict[Tuple[Any, Any], float] = {}
    pairwise_p_values: Dict[Tuple[Any, Any], Optional[float]] = {}
    for left, right in combinations(groups, 2):
        pair = (left, right)
        pairwise_gaps[pair] = _finite_gap(
            group_means[left], group_means[right], f"gap for groups {left!r} and {right!r}"
        )
        pairwise_p_values[pair] = _mann_whitney_u(group_scores[left], group_scores[right])
    mean_group_gap = _finite_mean(
        np.asarray(list(pairwise_gaps.values()), dtype=np.float64), "pairwise fidelity gap"
    )

    return {
        "max_gap_from_average": max_gap_from_average,
        "mean_group_gap": mean_group_gap,
        # Backward-compatible keys now have the paper's exact meanings.
        "max_gap": max_gap_from_average,
        "mean_gap": mean_group_gap,
        "overall_mean": overall_mean,
        "group_means": group_means,
        "group_deficits_from_average": deficits,
        "pairwise_gaps": pairwise_gaps,
        "pairwise_p_values": pairwise_p_values,
        "p_value_unavailable_pairs": [
            pair for pair, value in pairwise_p_values.items() if value is None
        ],
        "higher_is_better": bool(higher_is_better),
        "canonical_definition": "Balagopalan et al. (2022), Definitions 3.3 and 3.4",
    }


def compute_fidelity_disparity(
    attributions: Any,
    sensitive_features: Any,
    inner_metric: Optional[ScalarMetric] = None,
    *,
    higher_is_better: bool = True,
) -> Dict[str, Any]:
    """Compatibility adapter requiring an explicit per-row fidelity metric.

    Attributions alone do not determine fidelity.  ``inner_metric`` must return
    a genuine fidelity/performance/error score for each row.  Prefer calling
    :func:`compute_fidelity_gap` with already computed fidelity scores.
    """
    if inner_metric is None:
        raise ValueError(
            "inner_metric is required: attribution magnitude is not fidelity. "
            "Prefer compute_fidelity_gap(per_instance_fidelity_scores, ...)."
        )
    attrs = _validate_attributions(attributions)
    scores = _score_rows(attrs, inner_metric)
    result = compute_fidelity_gap(scores, sensitive_features, higher_is_better=higher_is_better)
    result["fidelity_metric_name"] = getattr(inner_metric, "__name__", "custom_metric")
    return result


def compute_sensitive_attribution_gap(
    attributions: Any,
    sensitive_features: Any,
    sensitive_feature_idx: int,
) -> Dict[str, Any]:
    """Compare signed sensitive-feature attribution means across groups."""
    attrs = _validate_attributions(attributions)
    labels = _validate_sensitive_features(sensitive_features, len(attrs))
    if isinstance(sensitive_feature_idx, (bool, np.bool_)) or not isinstance(
        sensitive_feature_idx, (int, np.integer)
    ):
        raise TypeError("sensitive_feature_idx must be an integer.")
    index = int(sensitive_feature_idx)
    if index < 0 or index >= attrs.shape[1]:
        raise ValueError(
            f"sensitive_feature_idx {index} is out of bounds for {attrs.shape[1]} features."
        )
    groups = _partition_by_group(labels)
    _require_multiple_groups(groups, "Sensitive-attribution gap")
    statistics = _group_statistics(attrs[:, index], groups)
    return {
        "divergence": statistics["disparity"],
        "group_sensitive_means": statistics["group_means"],
        "pairwise_gaps": statistics["pairwise_gaps"],
        "pairwise_p_values": statistics["pairwise_p_values"],
        "p_value": statistics["p_value"],
        "p_value_adjustment": statistics["p_value_adjustment"],
        "canonical_fairness_metric": False,
        "interpretation": (
            "Between-group gap in signed attribution assigned to the selected feature; "
            "zero gap is not proof of model or explanation fairness."
        ),
    }


def compute_attribution_parity(
    attributions: Any,
    sensitive_features: Any,
    sensitive_feature_idx: int,
) -> Dict[str, Any]:
    """Compatibility name for :func:`compute_sensitive_attribution_gap`."""
    return compute_sensitive_attribution_gap(
        attributions, sensitive_features, sensitive_feature_idx
    )


def compute_prediction_conditioned_metric_disparity(
    attributions: Any,
    sensitive_features: Any,
    predictions: Any,
    inner_metric: Optional[ScalarMetric] = None,
) -> Dict[str, Any]:
    """Compare a scalar attribution property within prediction strata.

    Conditioning on model predictions is not Hardt et al. equality of
    opportunity, which conditions error-rate comparisons on ground-truth
    outcomes.  Strata containing fewer than two groups are reported as not
    comparable and are not assigned a fabricated zero disparity.
    """
    attrs = _validate_attributions(attributions)
    labels = _validate_sensitive_features(sensitive_features, len(attrs))
    prediction_labels = _validate_labels(predictions, len(attrs), "predictions")
    supplied_metric = inner_metric is not None
    metric: ScalarMetric = inner_metric if inner_metric is not None else _attribution_l1_magnitude
    scores = _score_rows(attrs, metric)

    prediction_groups = _partition_by_group(prediction_labels)
    per_class_disparity: Dict[Any, Optional[float]] = {}
    per_class_group_means: Dict[Any, Dict[Any, float]] = {}
    non_comparable_classes: List[Any] = []
    comparable_values: List[float] = []
    for prediction_class, indices in prediction_groups.items():
        class_scores = scores[indices]
        class_sensitive = labels[indices]
        class_groups = _partition_by_group(class_sensitive)
        means = {
            group: _finite_mean(
                class_scores[group_indices],
                f"prediction {prediction_class!r}, group {group!r}",
            )
            for group, group_indices in class_groups.items()
        }
        per_class_group_means[prediction_class] = means
        if len(class_groups) < 2:
            per_class_disparity[prediction_class] = None
            non_comparable_classes.append(prediction_class)
            continue
        disparity = float(
            max(
                _finite_gap(
                    means[left],
                    means[right],
                    f"prediction-conditioned gap for groups {left!r} and {right!r}",
                )
                for left, right in combinations(means, 2)
            )
        )
        per_class_disparity[prediction_class] = disparity
        comparable_values.append(disparity)
    if not comparable_values:
        raise ValueError(
            "No prediction stratum contains at least two sensitive groups; a "
            "prediction-conditioned disparity is not estimable."
        )

    return {
        "disparity": float(max(comparable_values)),
        "per_class_disparity": per_class_disparity,
        "per_class_group_means": per_class_group_means,
        "non_comparable_classes": non_comparable_classes,
        "metric_name": (
            getattr(metric, "__name__", "custom_metric")
            if supplied_metric
            else "attribution_l1_magnitude"
        ),
        "condition": "model_prediction",
        "canonical_equal_opportunity": False,
        "canonical_fairness_metric": False,
        "interpretation": (
            "Prediction-conditioned scalar-property disparity; not Hardt equality "
            "of opportunity and not a standalone fairness verdict."
        ),
    }


def compute_conditional_fairness(
    attributions: Any,
    sensitive_features: Any,
    predictions: Any,
    inner_metric: Optional[ScalarMetric] = None,
) -> Dict[str, Any]:
    """Compatibility name for prediction-conditioned metric disparity."""
    return compute_prediction_conditioned_metric_disparity(
        attributions, sensitive_features, predictions, inner_metric
    )


_default_fairness_registry: Optional[FairnessMetricRegistry] = None


def _create_default_fairness_registry() -> FairnessMetricRegistry:
    registry = FairnessMetricRegistry()
    registry.register(
        "group_fairness",
        compute_group_fairness,
        FairnessMetricMeta(
            level="group",
            composable=True,
            description="Group scalar-property disparity diagnostic (Dai-inspired)",
            paper_reference="Dai et al. (2022), AIES",
            claim_scope="Only the caller-supplied scalar property is compared",
        ),
    )

    def individual_wrapper(attributions: Any, sensitive_features: Any, **kwargs: Any):
        if "inputs" not in kwargs:
            raise ValueError("individual_fairness registry evaluation requires inputs=...")
        inputs = kwargs.pop("inputs")
        return compute_individual_fairness(inputs, attributions, sensitive_features, **kwargs)

    registry.register(
        "individual_fairness",
        individual_wrapper,
        FairnessMetricMeta(
            level="individual",
            description="Scale-dependent cross-group Lipschitz diagnostic",
            paper_reference="Dwork et al. (2012) motivates, but does not define, this diagnostic",
            claim_scope="Diagnostic only; no task metric or fairness guarantee",
        ),
    )

    def intervention_wrapper(attributions: Any, sensitive_features: Any, **kwargs: Any):
        del sensitive_features
        if "inputs" not in kwargs or "sensitive_feature_idx" not in kwargs:
            raise ValueError(
                "counterfactual_fairness registry evaluation requires inputs=... "
                "and sensitive_feature_idx=..."
            )
        inputs = kwargs.pop("inputs")
        index = kwargs.pop("sensitive_feature_idx")
        return compute_counterfactual_fairness(inputs, attributions, index, **kwargs)

    registry.register(
        "counterfactual_fairness",
        intervention_wrapper,
        FairnessMetricMeta(
            level="individual",
            description="Sensitive-feature intervention or matched-group attribution change",
            paper_reference="Not a Kusner et al. (2017) SCM counterfactual-fairness test",
            claim_scope="Attribution-change diagnostic only",
        ),
    )
    registry.register(
        "fidelity_disparity",
        compute_fidelity_disparity,
        FairnessMetricMeta(
            level="group",
            composable=True,
            description="Maximum-from-average and mean subgroup fidelity gaps",
            paper_reference="Balagopalan et al. (2022), Definitions 3.3 and 3.4",
            canonical_claim=True,
            claim_scope="Requires a genuine caller-supplied fidelity metric",
        ),
    )

    def attribution_wrapper(attributions: Any, sensitive_features: Any, **kwargs: Any):
        if "sensitive_feature_idx" not in kwargs:
            raise ValueError("attribution_parity requires sensitive_feature_idx=...")
        index = kwargs.pop("sensitive_feature_idx")
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")
        return compute_attribution_parity(attributions, sensitive_features, index)

    registry.register(
        "attribution_parity",
        attribution_wrapper,
        FairnessMetricMeta(
            level="group",
            description="Signed sensitive-feature attribution gap diagnostic",
            claim_scope="Novel diagnostic; zero is not proof of fairness",
        ),
    )

    def conditional_wrapper(attributions: Any, sensitive_features: Any, **kwargs: Any):
        if "predictions" not in kwargs:
            raise ValueError("conditional_fairness requires predictions=...")
        predictions = kwargs.pop("predictions")
        return compute_conditional_fairness(attributions, sensitive_features, predictions, **kwargs)

    registry.register(
        "conditional_fairness",
        conditional_wrapper,
        FairnessMetricMeta(
            level="conditional",
            composable=True,
            description="Prediction-conditioned scalar-property disparity diagnostic",
            paper_reference="Not Hardt et al. (2016) equality of opportunity",
            claim_scope="Diagnostic only; conditions on predictions, not ground truth",
        ),
    )
    return registry


def get_default_fairness_registry() -> FairnessMetricRegistry:
    """Return the lazily constructed default diagnostic registry."""
    global _default_fairness_registry
    if _default_fairness_registry is None:
        _default_fairness_registry = _create_default_fairness_registry()
    return _default_fairness_registry
