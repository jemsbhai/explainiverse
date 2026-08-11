"""Perturbation-based diagnostics for feature-attribution faithfulness.

The public ``compute_pgi`` and ``compute_pgu`` functions implement a
*deterministic baseline-replacement variant* of the predictive-faithfulness
metrics in OpenXAI.  OpenXAI estimates an expectation over noisy local
perturbations (and reports an AUC over values of ``k``); consequently, values
from these functions must not be presented as numerically equivalent to the
OpenXAI benchmark.

``compute_comprehensiveness`` and ``compute_sufficiency`` adapt ERASER's
same-output, signed probability differences from token erasure to numeric
feature replacement.  This tabular adaptation is likewise not an ERASER
benchmark score.

Every diagnostic resolves one numeric model output from explicit or recorded
explanation identity and keeps it fixed. Multi-output calls fail when an
explanation label cannot be mapped; the model's current argmax is never used as
a substitute for an unknown explanation target.

Batch APIs require exact row/explanation pairing. Statistical and callable
baselines require explicit ``background_data`` and never derive replacement
values from the evaluated rows. Aggregate results distinguish the ratio of
mean PGI/PGU from the mean of per-row ratios.

References
----------
* Agarwal et al. (2022), OpenXAI, Appendix A.2, equations (1)--(2):
  https://proceedings.neurips.cc/paper_files/paper/2022/file/
  65398a0eba88c9b4a1c38ae405b125ef-Paper-Datasets_and_Benchmarks.pdf
* DeYoung et al. (2020), ERASER, equations (1)--(3):
  https://aclanthology.org/2020.acl-main.408.pdf
* Bhatt, Weller, and Moura (2020), Definition 3 (Faithfulness):
  https://arxiv.org/abs/2005.00631
"""

from __future__ import annotations

import math
import re
import warnings
from decimal import Decimal
from itertools import combinations
from numbers import Integral, Real
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from explainiverse.core.explanation import Explanation
from explainiverse.evaluation._utils import (
    _get_prediction_proba_vector,
    _model_task,
    _stable_difference_of_means,
    _stable_mean,
    _stable_pearson,
    _stable_pearson_affine,
    _stable_pearson_decimal_affine,
    _stable_ratio_of_means,
    _stable_std,
    _stable_sum,
    apply_feature_mask,
    compute_baseline_values,
    get_prediction_value,
)

Baseline = Union[str, float, np.ndarray, Callable]
KValue = Union[int, float]

_AOPC_DEFAULT_K_VALUES: tuple[float, ...] = (0.01, 0.05, 0.1, 0.2, 0.5)
_MAX_ENUMERATED_SUBSETS = 100_000
_INDEXED_FEATURE = re.compile(r"^(?:feature_|feat_|f|x)(\d+)$", re.IGNORECASE)


def _validate_instance(instance: np.ndarray) -> np.ndarray:
    """Return one finite numeric feature vector without silently flattening."""
    raw = np.asarray(instance)
    if raw.ndim != 1:
        raise ValueError(f"instance must be one-dimensional; got shape {raw.shape}")
    if raw.size == 0:
        raise ValueError("instance must contain at least one feature")
    if np.issubdtype(raw.dtype, np.bool_):
        raise TypeError("instance must contain numeric feature values, not booleans")
    try:
        values = raw.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("instance must contain numeric feature values") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError("instance must contain only finite values")
    return values


def _validate_dataset(X: np.ndarray) -> np.ndarray:
    """Validate a non-empty finite numeric sample-by-feature matrix."""
    raw = np.asarray(X)
    if raw.ndim != 2:
        raise ValueError(f"X must be two-dimensional; got shape {raw.shape}")
    if raw.shape[0] == 0 or raw.shape[1] == 0:
        raise ValueError("X must contain at least one sample and one feature")
    if np.issubdtype(raw.dtype, np.bool_):
        raise TypeError("X must contain numeric feature values, not booleans")
    try:
        values = raw.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("X must contain numeric feature values") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError("X must contain only finite values")
    return values


def _coerce_attribution(value: object, feature_name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"attribution for {feature_name!r} must be numeric, not boolean")
    array = np.asarray(value)
    if array.ndim != 0:
        raise TypeError(f"attribution for {feature_name!r} must be a scalar")
    try:
        result = float(array)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"attribution for {feature_name!r} must be numeric") from exc
    if not np.isfinite(result):
        raise ValueError(f"attribution for {feature_name!r} must be finite")
    return result


def _extract_attribution_vector(explanation: Explanation, n_features: int) -> np.ndarray:
    """Map every attribution to its input column without positional guesses."""
    if not isinstance(explanation, Explanation):
        raise TypeError("explanation must be an Explanation instance")

    if not isinstance(explanation.explanation_data, Mapping):
        raise TypeError("explanation.explanation_data must be a mapping")
    attributions = explanation.explanation_data.get("feature_attributions")
    if not isinstance(attributions, Mapping) or not attributions:
        raise ValueError("explanation must contain a non-empty feature_attributions mapping")

    feature_names = explanation.feature_names
    if feature_names is not None:
        if len(feature_names) != n_features:
            raise ValueError(
                "explanation.feature_names length must equal the number of input features"
            )
        if not all(isinstance(name, str) and name for name in feature_names):
            raise TypeError("explanation.feature_names must contain non-empty strings")
        if len(set(feature_names)) != n_features:
            raise ValueError("explanation.feature_names must be unique")
        if set(attributions) != set(feature_names):
            missing = [name for name in feature_names if name not in attributions]
            extra = [name for name in attributions if name not in set(feature_names)]
            raise ValueError(
                "feature_attributions must cover feature_names exactly; "
                f"missing={missing}, extra={extra}"
            )
        return np.asarray(
            [_coerce_attribution(attributions[name], name) for name in feature_names],
            dtype=float,
        )

    # Without feature_names, only explicit, unambiguous indexed keys are safe.
    indexed: dict[int, float] = {}
    for name, value in attributions.items():
        if not isinstance(name, str):
            raise TypeError("feature attribution keys must be strings")
        match = _INDEXED_FEATURE.fullmatch(name)
        if match is None:
            raise ValueError(
                "feature_names are required unless attribution keys use explicit "
                "indices such as feature_0, feat_0, f0, or x0"
            )
        index = int(match.group(1))
        if index >= n_features:
            raise ValueError(f"attribution key {name!r} refers to an out-of-range feature")
        if index in indexed:
            raise ValueError(f"multiple attribution keys refer to feature index {index}")
        indexed[index] = _coerce_attribution(value, name)

    expected = set(range(n_features))
    if set(indexed) != expected:
        raise ValueError(
            "indexed feature_attributions must cover every input feature exactly; "
            f"missing indices={sorted(expected - set(indexed))}"
        )
    return np.asarray([indexed[index] for index in range(n_features)], dtype=float)


def _resolve_feature_count(k: KValue, n_features: int) -> int:
    """Resolve a positive count or a fraction in ``(0, 1]`` to a count."""
    if isinstance(k, (bool, np.bool_)):
        raise TypeError("k must be an integer count or a real fraction, not boolean")
    if isinstance(k, Integral):
        count = int(k)
        if count <= 0 or count > n_features:
            raise ValueError(f"integer k must be in [1, {n_features}]; got {k}")
        return count
    if isinstance(k, Real):
        fraction = float(k)
        if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError(f"fractional k must be finite and in (0, 1]; got {k}")
        return max(1, int(fraction * n_features))
    raise TypeError("k must be an integer count or a real fraction")


def _resolve_baseline(
    baseline: Baseline,
    background_data: Optional[np.ndarray],
    n_features: int,
) -> np.ndarray:
    if isinstance(baseline, (bool, np.bool_)):
        raise TypeError("baseline must not be boolean")

    validated_background = background_data
    if background_data is not None:
        validated_background = _validate_dataset(background_data)
        if validated_background.shape[1] != n_features:
            raise ValueError("background_data must have the same number of features as instance")

    values = compute_baseline_values(baseline, validated_background, n_features)
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.shape[0] != n_features:
        raise ValueError(f"baseline must resolve to shape ({n_features},); got {raw.shape}")
    if np.issubdtype(raw.dtype, np.bool_):
        raise TypeError("baseline must contain numeric values, not booleans")
    try:
        result = raw.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("baseline must contain numeric values") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError("baseline must contain only finite values")
    return result


def _candidate_from_label(model, explanation: Explanation) -> Optional[int]:
    label = explanation.target_class
    if label is None:
        return None
    if isinstance(label, (bool, np.bool_)) or not isinstance(label, (str, Integral)):
        raise TypeError("explanation.target_class must be a string, integer output index, or None")

    class_names = getattr(model, "class_names", None)
    if class_names is not None:
        matches = [
            index
            for index, name in enumerate(class_names)
            if label == name or str(label) == str(name)
        ]
        if len(matches) > 1:
            raise ValueError("model.class_names does not identify a unique target output")
        if matches:
            return matches[0]

    wrapped_model = getattr(model, "model", None)
    classes = getattr(model, "classes_", None)
    if classes is None and wrapped_model is not None:
        classes = getattr(wrapped_model, "classes_", None)
    if classes is not None:
        matches = [
            index
            for index, class_label in enumerate(np.asarray(classes).reshape(-1))
            if label == class_label or str(label) == str(class_label)
        ]
        if len(matches) > 1:
            raise ValueError("model.classes_ does not identify a unique target output")
        if matches:
            return matches[0]

    # ``Explanation.target_class`` is presentation identity first. An integer
    # therefore maps through declared class labels above (important for labels
    # such as [10, 20]) and acts as a raw output index only when no label matches.
    if isinstance(label, Integral):
        return int(label)

    if isinstance(label, str):
        match = re.fullmatch(r"(?:class|output)_(\d+)", label)
        if match is not None:
            return int(match.group(1))
    return None


def _resolve_target_output(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    target_class: Optional[int],
) -> int:
    outputs = np.asarray(_get_prediction_proba_vector(model, instance), dtype=float).reshape(-1)
    if not np.all(np.isfinite(outputs)):
        raise ValueError("model returned a non-finite prediction")

    candidates: list[tuple[str, int]] = []
    if target_class is not None:
        if not isinstance(target_class, Integral) or isinstance(target_class, (bool, np.bool_)):
            raise TypeError("target_class must be an integer output index or None")
        candidates.append(("target_class", int(target_class)))

    metadata = explanation.metadata
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("explanation.metadata must be a mapping")
    containers = (
        ("explanation.metadata", metadata),
        ("explanation.explanation_data", explanation.explanation_data),
    )
    for container_name, container in containers:
        if not isinstance(container, Mapping):
            continue
        for key in ("output_index", "target_class_index", "class_index"):
            if key not in container:
                continue
            output_index = container[key]
            if not isinstance(output_index, Integral) or isinstance(output_index, (bool, np.bool_)):
                raise TypeError(f"{container_name}[{key!r}] must be an integer")
            candidates.append((f"{container_name}[{key!r}]", int(output_index)))

    label_candidate = _candidate_from_label(model, explanation)
    if label_candidate is not None:
        candidates.append(("explanation.target_class", label_candidate))

    if candidates:
        unique = {candidate for _, candidate in candidates}
        if len(unique) != 1:
            details = ", ".join(f"{source}={candidate}" for source, candidate in candidates)
            raise ValueError(f"conflicting target output metadata: {details}")
        resolved = candidates[0][1]
    else:
        if outputs.size > 1:
            if _model_task(model) == "regression":
                raise ValueError("multi-output regression requires an explicit target output index")
            raise ValueError(
                "cannot establish which model output the explanation describes; pass "
                "target_class or record a numeric output index/class label that maps "
                "through model.class_names or model.classes_"
            )
        resolved = 0

    if resolved < 0 or resolved >= outputs.size:
        raise ValueError(
            f"target output index {resolved} is invalid for {outputs.size} model output(s)"
        )
    return resolved


def _prediction_for_target(model, instance: np.ndarray, target_class: int) -> float:
    value = float(get_prediction_value(model, instance, target_class=target_class))
    if not np.isfinite(value):
        raise ValueError("model returned a non-finite prediction")
    return value


def _rank_feature_indices(
    attributions: np.ndarray,
    k: int,
    *,
    absolute: bool,
) -> np.ndarray:
    scores = np.abs(attributions) if absolute else attributions
    order = np.argsort(-scores, kind="stable")
    if 0 < k < attributions.size and scores[order[k - 1]] == scores[order[k]]:
        raise ValueError(
            "the attribution ranking is undefined because a tie crosses the top-k cutoff"
        )
    return order


def _prepare_metric_inputs(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    baseline: Baseline,
    background_data: Optional[np.ndarray],
    target_class: Optional[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    values = _validate_instance(instance)
    attributions = _extract_attribution_vector(explanation, values.size)
    baseline_values = _resolve_baseline(baseline, background_data, values.size)
    resolved_target = _resolve_target_output(model, values, explanation, target_class)
    original_value = _prediction_for_target(model, values, resolved_target)
    return values, attributions, baseline_values, resolved_target, original_value


def compute_pgi(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    k: KValue = 0.2,
    baseline: Baseline = "mean",
    background_data: Optional[np.ndarray] = None,
    target_class: Optional[int] = None,
) -> float:
    """Compute a deterministic baseline-replacement PGI value at one ``k``.

    The top-``k`` features by absolute attribution are replaced with their
    baseline values, and the absolute change in one fixed model output is
    returned. If no output index is supplied, the explanation's output identity
    is used. Ambiguous or conflicting multi-output identities fail explicitly.

    This is a deterministic specialization of OpenXAI PGI, not the canonical
    noisy-expectation/AUC benchmark estimator.  A larger result only indicates
    greater sensitivity under the chosen replacement intervention; it is not,
    by itself, proof that an explanation is faithful.
    """
    (
        values,
        attributions,
        baseline_values,
        resolved_target,
        original_value,
    ) = _prepare_metric_inputs(
        model, instance, explanation, baseline, background_data, target_class
    )
    k_int = _resolve_feature_count(k, values.size)
    order = _rank_feature_indices(attributions, k_int, absolute=True)
    perturbed = apply_feature_mask(values, order[:k_int].tolist(), baseline_values)
    perturbed_value = _prediction_for_target(model, perturbed, resolved_target)
    return abs(_finite_score_difference(original_value, perturbed_value, "PGI"))


def compute_pgu(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    k: KValue = 0.2,
    baseline: Baseline = "mean",
    background_data: Optional[np.ndarray] = None,
    target_class: Optional[int] = None,
) -> float:
    """Compute a deterministic baseline-replacement PGU value at one ``k``.

    OpenXAI PGU holds the top-``k`` features fixed and perturbs **all non-top-k
    features**.  Accordingly, this function replaces the complete complement
    of the top-``k`` absolute-attribution set, rather than replacing only
    ``k`` bottom-ranked features.

    This is a deterministic specialization of OpenXAI PGU, not the canonical
    noisy-expectation/AUC benchmark estimator.  A smaller result means greater
    invariance only under the chosen replacement intervention.
    """
    (
        values,
        attributions,
        baseline_values,
        resolved_target,
        original_value,
    ) = _prepare_metric_inputs(
        model, instance, explanation, baseline, background_data, target_class
    )
    k_int = _resolve_feature_count(k, values.size)
    order = _rank_feature_indices(attributions, k_int, absolute=True)
    non_top_k = order[k_int:]
    perturbed = apply_feature_mask(values, non_top_k.tolist(), baseline_values)
    perturbed_value = _prediction_for_target(model, perturbed, resolved_target)
    return abs(_finite_score_difference(original_value, perturbed_value, "PGU"))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        raise ValueError("faithfulness ratio requires finite PGI and PGU values")
    if denominator == 0.0:
        raise ValueError("faithfulness ratio is undefined because PGU is zero")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = float(numerator / denominator)
    if not np.isfinite(result) or (result == 0.0 and numerator != 0.0):
        raise FloatingPointError("faithfulness ratio is not representable")
    return result


def _finite_score_mean(values: Sequence[float], context: str) -> float:
    result = float(_stable_mean(np.asarray(values, dtype=np.float64)))
    if not np.isfinite(result):
        raise FloatingPointError(f"{context} mean is not representable")
    return result


def _finite_score_std(values: Sequence[float], context: str) -> float:
    result = float(_stable_std(np.asarray(values, dtype=np.float64)))
    if not np.isfinite(result):
        raise FloatingPointError(f"{context} standard deviation is not representable")
    return result


def _finite_score_difference(left: float, right: float, context: str) -> float:
    result = float(_stable_sum(np.asarray([left, -right], dtype=np.float64)))
    if not np.isfinite(result):
        raise FloatingPointError(f"{context} is not representable")
    return result


def compute_faithfulness_score(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    k: KValue = 0.2,
    baseline: Baseline = "mean",
    background_data: Optional[np.ndarray] = None,
    epsilon: Optional[float] = None,
    target_class: Optional[int] = None,
) -> Dict[str, float]:
    """Return PGI, PGU, and two project-specific descriptive contrasts.

    ``faithfulness_ratio`` and ``faithfulness_diff`` are not metrics defined by
    OpenXAI.  They are retained as compatibility diagnostics and must not be
    cited as literature metrics. The ratio is undefined when PGU is zero and
    raises explicitly; an epsilon is not added because doing so invents a
    scale-dependent finite value.
    """
    if epsilon is not None:
        if isinstance(epsilon, (bool, np.bool_)) or not isinstance(epsilon, Real):
            raise TypeError("epsilon must be a finite non-negative real number or None")
        if not np.isfinite(float(epsilon)) or float(epsilon) < 0.0:
            raise ValueError("epsilon must be a finite non-negative real number or None")
        warnings.warn(
            "epsilon is deprecated and ignored; the exact PGI/PGU ratio is returned",
            FutureWarning,
            stacklevel=2,
        )

    pgi = compute_pgi(model, instance, explanation, k, baseline, background_data, target_class)
    pgu = compute_pgu(model, instance, explanation, k, baseline, background_data, target_class)
    return {
        "pgi": pgi,
        "pgu": pgu,
        "faithfulness_ratio": _safe_ratio(pgi, pgu),
        "faithfulness_diff": float(pgi - pgu),
    }


def _validate_k_values(
    k_values: Optional[Sequence[KValue]],
) -> tuple[KValue, ...]:
    if k_values is None:
        return _AOPC_DEFAULT_K_VALUES
    if isinstance(k_values, (str, bytes)):
        raise TypeError("k_values must be a non-empty sequence of k values")
    try:
        values = tuple(k_values)
    except TypeError as exc:
        raise TypeError("k_values must be a non-empty sequence of k values") from exc
    if not values:
        raise ValueError("k_values must not be empty")
    identities: list[tuple[str, Union[int, float]]] = []
    fractional_values: list[Real] = []
    for value in values:
        identity: tuple[str, Union[int, float]]
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("k_values entries must not be boolean")
        if isinstance(value, Integral):
            if int(value) <= 0:
                raise ValueError("integer k_values entries must be positive")
            identity = ("count", int(value))
        elif isinstance(value, Real):
            fraction = float(value)
            if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
                raise ValueError("fractional k_values entries must be finite and in (0, 1]")
            if any(bool(value == previous) for previous in fractional_values):
                raise ValueError("k_values must not contain semantically duplicate entries")
            fractional_values.append(value)
            identity = ("fraction", fraction)
        else:
            raise TypeError("k_values entries must be integer counts or real fractions")
        if identity in identities:
            raise ValueError("k_values must not contain semantically duplicate entries")
        identities.append(identity)
    return values


def _k_result_suffix(k: KValue) -> str:
    """Return one canonical, collision-free suffix for a validated k value."""
    if isinstance(k, Integral):
        return str(int(k))
    return repr(float(k))


def _compute_eraser_adaptation(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    k_values: Optional[Sequence[KValue]],
    baseline: Baseline,
    background_data: Optional[np.ndarray],
    target_class: Optional[int],
    *,
    keep_top: bool,
) -> Dict[str, float]:
    (
        values,
        attributions,
        baseline_values,
        resolved_target,
        original_value,
    ) = _prepare_metric_inputs(
        model, instance, explanation, baseline, background_data, target_class
    )
    requested_k = _validate_k_values(k_values)
    scores: Dict[str, float] = {}

    for k in requested_k:
        k_int = _resolve_feature_count(k, values.size)
        # ERASER selects the highest importance scores, not the largest
        # magnitudes: negative scores oppose the target class.
        order = _rank_feature_indices(attributions, k_int, absolute=False)
        top_k = set(int(index) for index in order[:k_int])
        if keep_top:
            indices_to_mask = [index for index in range(values.size) if index not in top_k]
            prefix = "suff"
        else:
            indices_to_mask = sorted(top_k)
            prefix = "comp"
        perturbed = apply_feature_mask(values, indices_to_mask, baseline_values)
        perturbed_value = _prediction_for_target(model, perturbed, resolved_target)
        scores[f"{prefix}_k{_k_result_suffix(k)}"] = _finite_score_difference(
            original_value, perturbed_value, prefix
        )

    aggregate_name = "sufficiency" if keep_top else "comprehensiveness"
    scores[aggregate_name] = _finite_score_mean(
        list(scores.values()), f"{aggregate_name} aggregate"
    )
    return scores


def compute_comprehensiveness(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    k_values: Optional[List[KValue]] = None,
    baseline: Baseline = "mean",
    background_data: Optional[np.ndarray] = None,
    target_class: Optional[int] = None,
) -> Dict[str, float]:
    """Compute a tabular baseline-replacement adaptation of ERASER COMP.

    For each ``k``, this returns ``f_j(x) - f_j(x without top-k)`` for one
    fixed output ``j``.  The difference is deliberately signed: a negative
    value means removing the selected features increased that output.  The
    aggregate is the arithmetic mean over the requested thresholds, matching
    the released ERASER scorer's AOPC aggregation.  Defaults are ERASER's
    ``[1%, 5%, 10%, 20%, 50%]`` thresholds.
    """
    return _compute_eraser_adaptation(
        model,
        instance,
        explanation,
        k_values,
        baseline,
        background_data,
        target_class,
        keep_top=False,
    )


def compute_sufficiency(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    k_values: Optional[List[KValue]] = None,
    baseline: Baseline = "mean",
    background_data: Optional[np.ndarray] = None,
    target_class: Optional[int] = None,
) -> Dict[str, float]:
    """Compute a tabular baseline-replacement adaptation of ERASER SUFF.

    For each ``k``, this returns ``f_j(x) - f_j(top-k only)`` for one fixed
    output ``j``.  Lower values indicate that the selected features preserve
    more of that output under this intervention; negative values are valid and
    are not converted to magnitudes.
    """
    return _compute_eraser_adaptation(
        model,
        instance,
        explanation,
        k_values,
        baseline,
        background_data,
        target_class,
        keep_top=True,
    )


def _sample_feature_subsets(
    n_features: int,
    subset_size: int,
    n_subsets: Optional[int],
    random_state: Optional[int],
) -> list[tuple[int, ...]]:
    if random_state is not None and (
        isinstance(random_state, (bool, np.bool_)) or not isinstance(random_state, Integral)
    ):
        raise TypeError("random_state must be an integer or None")

    total = math.comb(n_features, subset_size)
    if total < 2:
        raise ValueError("faithfulness correlation requires at least two distinct subsets")

    if n_subsets is None:
        if total > _MAX_ENUMERATED_SUBSETS:
            raise ValueError(
                f"there are {total} subsets; provide n_steps to request an explicit sample"
            )
        return list(combinations(range(n_features), subset_size))

    if isinstance(n_subsets, (bool, np.bool_)) or not isinstance(n_subsets, Integral):
        raise TypeError("n_steps must be an integer or None")
    count = int(n_subsets)
    if count < 2:
        raise ValueError("n_steps must be at least 2 for a correlation")
    if count > total:
        raise ValueError(f"n_steps={count} exceeds the {total} distinct feature subsets")
    rng = np.random.default_rng(None if random_state is None else int(random_state))

    if total <= _MAX_ENUMERATED_SUBSETS:
        candidates = list(combinations(range(n_features), subset_size))
        indices = rng.choice(total, size=count, replace=False)
        return [candidates[int(index)] for index in indices]

    sampled: set[tuple[int, ...]] = set()
    while len(sampled) < count:
        subset = tuple(
            sorted(int(index) for index in rng.choice(n_features, subset_size, replace=False))
        )
        sampled.add(subset)
    return sorted(sampled)


def compute_faithfulness_correlation(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    baseline: Baseline = "mean",
    background_data: Optional[np.ndarray] = None,
    n_steps: Optional[int] = None,
    subset_size: KValue = 1,
    random_state: Optional[int] = None,
    target_class: Optional[int] = None,
) -> float:
    """Estimate Bhatt et al.'s fixed-size-subset faithfulness correlation.

    For subsets ``S`` of one fixed size, this computes Pearson correlation
    between ``sum(attribution[S])`` and
    ``f_j(x) - f_j(x with S replaced by baseline)``.  Both quantities remain
    signed.  With ``n_steps=None`` all subsets are enumerated (up to a safety
    limit); otherwise ``n_steps`` unique subsets are sampled with a local RNG.
    The historical ``n_steps`` name is retained for API compatibility and now
    denotes the number of subsets, not the number of ranked features.

    Pearson correlation is mathematically undefined when either vector is
    constant, or when fewer than two subsets exist; those cases raise instead
    of fabricating a score of zero.
    """
    (
        values,
        attributions,
        baseline_values,
        resolved_target,
        original_value,
    ) = _prepare_metric_inputs(
        model, instance, explanation, baseline, background_data, target_class
    )
    subset_count = _resolve_feature_count(subset_size, values.size)
    subsets = _sample_feature_subsets(values.size, subset_count, n_steps, random_state)

    attribution_sums: list[Decimal] = []
    perturbed_values: list[float] = []
    for subset in subsets:
        attribution_sums.append(
            sum(
                (Decimal.from_float(float(value)) for value in attributions[list(subset)]),
                start=Decimal(0),
            )
        )
        perturbed = apply_feature_mask(values, list(subset), baseline_values)
        perturbed_value = _prediction_for_target(model, perturbed, resolved_target)
        perturbed_values.append(perturbed_value)

    perturbed_array = np.asarray(perturbed_values, dtype=float)
    try:
        return _stable_pearson_decimal_affine(attribution_sums, original_value, perturbed_array)
    except ValueError as exc:
        raise ValueError(
            "Pearson faithfulness correlation is undefined when attribution sums "
            "or output changes are constant"
        ) from exc


def _validate_explanation_collection(
    X: np.ndarray,
    explanations: Sequence[Explanation],
) -> None:
    if isinstance(explanations, (str, bytes)) or not isinstance(explanations, Sequence):
        raise TypeError("explanations must be a sequence of Explanation objects")
    if len(explanations) != X.shape[0]:
        raise ValueError(
            f"expected one explanation per row of X ({X.shape[0]}); got {len(explanations)}"
        )


def compare_explainer_faithfulness(
    model,
    X: np.ndarray,
    explanations: Dict[str, List[Explanation]],
    k: KValue = 0.2,
    baseline: Baseline = "mean",
    max_samples: Optional[int] = None,
    background_data: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Compare deterministic PGI/PGU variants without suppressing failures.

    Statistical or callable baselines require explicit ``background_data``;
    evaluated rows are never silently reused as the background distribution.
    ``ratio_of_means`` and ``mean_of_sample_ratios`` are reported separately
    because these aggregations are generally unequal. ``mean_ratio`` remains
    as a compatibility alias for ``ratio_of_means``.
    """
    values = _validate_dataset(X)
    if not isinstance(explanations, Mapping):
        raise TypeError("explanations must map explainer names to explanation sequences")
    if max_samples is not None:
        if isinstance(max_samples, (bool, np.bool_)) or not isinstance(max_samples, Integral):
            raise TypeError("max_samples must be a positive integer or None")
        if int(max_samples) <= 0:
            raise ValueError("max_samples must be a positive integer or None")
        n_samples = min(int(max_samples), values.shape[0])
    else:
        n_samples = values.shape[0]

    columns = [
        "explainer",
        "mean_pgi",
        "std_pgi",
        "mean_pgu",
        "std_pgu",
        "ratio_of_means",
        "mean_of_sample_ratios",
        "mean_ratio",
        "mean_diff",
        "n_samples",
    ]
    results: list[dict[str, object]] = []
    for explainer_name, explainer_values in explanations.items():
        if not isinstance(explainer_name, str) or not explainer_name:
            raise TypeError("explainer names must be non-empty strings")
        _validate_explanation_collection(values, explainer_values)
        pgi_scores: list[float] = []
        pgu_scores: list[float] = []
        for index in range(n_samples):
            scores = compute_faithfulness_score(
                model,
                values[index],
                explainer_values[index],
                k,
                baseline,
                background_data,
            )
            pgi_scores.append(scores["pgi"])
            pgu_scores.append(scores["pgu"])

        mean_pgi = _finite_score_mean(pgi_scores, "PGI")
        mean_pgu = _finite_score_mean(pgu_scores, "PGU")
        ratio_of_means = _stable_ratio_of_means(np.asarray(pgi_scores), np.asarray(pgu_scores))
        mean_of_sample_ratios = _finite_score_mean(
            [_safe_ratio(pgi, pgu) for pgi, pgu in zip(pgi_scores, pgu_scores)],
            "faithfulness sample ratio",
        )
        results.append(
            {
                "explainer": explainer_name,
                "mean_pgi": mean_pgi,
                "std_pgi": _finite_score_std(pgi_scores, "PGI"),
                "mean_pgu": mean_pgu,
                "std_pgu": _finite_score_std(pgu_scores, "PGU"),
                "ratio_of_means": ratio_of_means,
                "mean_of_sample_ratios": mean_of_sample_ratios,
                "mean_ratio": ratio_of_means,
                "mean_diff": _stable_difference_of_means(
                    np.asarray(pgi_scores), np.asarray(pgu_scores)
                ),
                "n_samples": n_samples,
            }
        )
    return pd.DataFrame(results, columns=columns)


def compute_batch_faithfulness(
    model,
    X: np.ndarray,
    explanations: List[Explanation],
    k: KValue = 0.2,
    baseline: Baseline = "mean",
    background_data: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Aggregate deterministic PGI/PGU variants over a validated batch.

    Statistical or callable baselines require an explicit background dataset.
    ``ratio_of_means`` is the ratio of the two aggregate means, while
    ``mean_of_sample_ratios`` averages the per-row ratios. ``mean_ratio`` is a
    compatibility alias for the former.
    """
    values = _validate_dataset(X)
    _validate_explanation_collection(values, explanations)

    pgi_scores: list[float] = []
    pgu_scores: list[float] = []
    for index, explanation in enumerate(explanations):
        scores = compute_faithfulness_score(
            model, values[index], explanation, k, baseline, background_data
        )
        pgi_scores.append(scores["pgi"])
        pgu_scores.append(scores["pgu"])

    mean_pgi = _finite_score_mean(pgi_scores, "PGI")
    mean_pgu = _finite_score_mean(pgu_scores, "PGU")
    ratio_of_means = _stable_ratio_of_means(np.asarray(pgi_scores), np.asarray(pgu_scores))
    return {
        "mean_pgi": mean_pgi,
        "std_pgi": _finite_score_std(pgi_scores, "PGI"),
        "mean_pgu": mean_pgu,
        "std_pgu": _finite_score_std(pgu_scores, "PGU"),
        "ratio_of_means": ratio_of_means,
        "mean_of_sample_ratios": _finite_score_mean(
            [_safe_ratio(pgi, pgu) for pgi, pgu in zip(pgi_scores, pgu_scores)],
            "faithfulness sample ratio",
        ),
        "mean_ratio": ratio_of_means,
        "mean_diff": _stable_difference_of_means(np.asarray(pgi_scores), np.asarray(pgu_scores)),
        "n_samples": len(pgi_scores),
    }
