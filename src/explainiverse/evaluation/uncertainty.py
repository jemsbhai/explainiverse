"""Owned helpers for finite-estimator uncertainty and intervention sensitivity."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal, localcontext
from fractions import Fraction
from numbers import Integral, Real
from typing import Any, Optional, Union

import numpy as np
from scipy import stats

from explainiverse.evaluation._utils import _stable_mean, _stable_std, _stable_sum

_INTERVENTION_FINGERPRINT_SCHEMA = "explainiverse.intervention-reference.sha256.v1"
Binary64Scalar = Union[int, float, Decimal, Fraction, np.generic]


def _lossless_binary64(value: Any, *, name: str) -> float:
    """Coerce an explicitly supported scalar without changing its exact value."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar, not a boolean")
    if type(value) is float:
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value
    if isinstance(value, np.floating):
        if value.dtype.itemsize > np.dtype(np.float64).itemsize:
            raise TypeError(f"{name} NumPy floating dtype {value.dtype} is wider than binary64")
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"{name} must be finite")
        return numeric
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"{name} must be finite")
        try:
            numeric = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"{name} is not losslessly representable as binary64") from exc
        if not np.isfinite(numeric) or Decimal.from_float(numeric) != value:
            raise ValueError(f"{name} is not losslessly representable as binary64")
        return numeric
    if isinstance(value, Fraction):
        try:
            numeric = float(value)
        except OverflowError as exc:
            raise ValueError(f"{name} is not losslessly representable as binary64") from exc
        if not np.isfinite(numeric) or Fraction.from_float(numeric) != value:
            raise ValueError(f"{name} is not losslessly representable as binary64")
        return numeric
    if isinstance(value, np.integer):
        integer = int(value)
        numeric = float(integer)
        if not np.isfinite(numeric) or int(numeric) != integer:
            raise ValueError(f"{name} is not losslessly representable as binary64")
        return numeric
    if type(value) is int:
        try:
            numeric = float(value)
        except OverflowError as exc:
            raise ValueError(f"{name} is not losslessly representable as binary64") from exc
        if not np.isfinite(numeric) or int(numeric) != value:
            raise ValueError(f"{name} is not losslessly representable as binary64")
        return numeric
    if isinstance(value, Real):
        raise TypeError(
            f"{name} real scalar type {type(value).__name__} has no lossless binary64 contract"
        )
    raise TypeError(
        f"{name} must be a supported Python/NumPy integer or float, Decimal, or Fraction"
    )


def _canonical_reference(value: Any, path: str, active: set[int]) -> Any:
    """Build an exact, deterministic JSON value for reference fingerprinting."""

    if value is None:
        return ["null"]
    if type(value) is bool:
        return ["boolean", "python:bool", value]
    if isinstance(value, np.bool_):
        return ["boolean", f"numpy:{value.dtype.str}", bool(value)]
    if isinstance(value, str):
        return ["string", value]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"{path} must be finite for deterministic fingerprinting")
        decimal_tuple = value.as_tuple()
        return [
            "decimal",
            decimal_tuple.sign,
            "".join(str(digit) for digit in decimal_tuple.digits),
            decimal_tuple.exponent,
        ]
    if isinstance(value, Fraction):
        return ["fraction", str(value.numerator), str(value.denominator)]
    if isinstance(value, np.integer):
        return ["integer", f"numpy:{value.dtype.str}", str(int(value))]
    if isinstance(value, Integral):
        return ["integer", "python:int", str(int(value))]
    if isinstance(value, np.floating):
        if value.dtype.itemsize > np.dtype(np.float64).itemsize:
            raise TypeError(
                f"{path} NumPy floating scalar dtype {value.dtype} is wider than binary64; "
                "use Decimal, Fraction, or an exact ndarray reference"
            )
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"{path} must be finite for deterministic fingerprinting")
        return ["real", f"numpy:float{value.dtype.itemsize * 8}", numeric.hex()]
    if type(value) is float:
        if not np.isfinite(value):
            raise ValueError(f"{path} must be finite for deterministic fingerprinting")
        return ["real", "python:float", value.hex()]
    if isinstance(value, Real):
        raise TypeError(
            f"{path} real scalar type {type(value).__name__} has no supported exact "
            "fingerprint; use Python int/float, Decimal, Fraction, or a supported NumPy type"
        )
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject or value.dtype.kind not in "biufSU":
            raise TypeError(
                f"{path} ndarray dtype {value.dtype} has no supported exact fingerprint"
            )
        if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
            raise ValueError(f"{path} must be finite for deterministic fingerprinting")
        contiguous = np.ascontiguousarray(value)
        return [
            "ndarray",
            value.dtype.str,
            list(value.shape),
            contiguous.tobytes(order="C").hex(),
        ]
    if type(value) not in {list, tuple, dict}:
        raise TypeError(
            f"{path} must use supported finite scalars, arrays, lists, tuples, or dictionaries"
        )

    identity = id(value)
    if identity in active:
        raise TypeError(f"{path} must not contain cyclic references")
    active.add(identity)
    try:
        if type(value) is list:
            return [
                "list",
                [
                    _canonical_reference(item, f"{path}[{index}]", active)
                    for index, item in enumerate(value)
                ],
            ]
        if type(value) is tuple:
            return [
                "tuple",
                [
                    _canonical_reference(item, f"{path}[{index}]", active)
                    for index, item in enumerate(value)
                ],
            ]
        if any(type(key) is not str for key in value):
            raise TypeError(f"{path} dictionary keys must be strings")
        entries = []
        for key in sorted(value):
            entries.append([key, _canonical_reference(value[key], f"{path}.{key}", active)])
        return ["dictionary", entries]
    finally:
        active.remove(identity)


def _reference_fingerprint(value: Any) -> str:
    canonical = _canonical_reference(value, "intervention reference", set())
    serialized = json.dumps(
        canonical,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(serialized).hexdigest()


def _fraction_as_decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def _fused_student_t_interval(
    values: np.ndarray,
    critical: float,
) -> tuple[Optional[float], list[float]]:
    """Evaluate SE and CI endpoints before any subnormal intermediate rounding."""

    exact_values = [Fraction.from_float(float(value)) for value in values]
    count = len(exact_values)
    exact_mean = sum(exact_values, start=Fraction(0)) / count
    squared_deviations = sum(
        ((value - exact_mean) ** 2 for value in exact_values),
        start=Fraction(0),
    )
    exact_variance = squared_deviations / (count - 1)
    with localcontext() as context:
        context.prec = 3000 + len(str(count))
        mean_decimal = _fraction_as_decimal(exact_mean)
        standard_error_decimal = (_fraction_as_decimal(exact_variance) / Decimal(count)).sqrt()
        margin_decimal = Decimal.from_float(critical) * standard_error_decimal
        lower = float(mean_decimal - margin_decimal)
        upper = float(mean_decimal + margin_decimal)
        rounded_standard_error = float(standard_error_decimal)
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise FloatingPointError("replicate confidence interval is not representable")
    standard_error = (
        None
        if rounded_standard_error == 0.0 and standard_error_decimal != 0
        else rounded_standard_error
    )
    return standard_error, [lower, upper]


def _validated_estimates(estimates: Sequence[Binary64Scalar]) -> np.ndarray:
    if isinstance(estimates, (str, bytes)) or not isinstance(estimates, Sequence):
        raise TypeError("estimates must be a non-empty sequence of finite real scalars")
    if not estimates:
        raise ValueError("estimates must not be empty")
    return np.asarray(
        [
            _lossless_binary64(value, name=f"estimates[{index}]")
            for index, value in enumerate(estimates)
        ],
        dtype=np.float64,
    )


def _validated_seeds(seeds: Sequence[int], count: int) -> list[int]:
    if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Sequence):
        raise TypeError("seeds must be a sequence of non-negative integers")
    if len(seeds) != count:
        raise ValueError("seeds must contain exactly one entry per estimate")
    if any(
        isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or int(seed) < 0
        for seed in seeds
    ):
        raise ValueError("seeds must contain only non-negative integers")
    result = [int(seed) for seed in seeds]
    if len(set(result)) != len(result):
        raise ValueError("seeds must be unique so replicates identify distinct RNG streams")
    return result


def _positive_sample_count(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("sample_count must be a positive integer")
    if int(value) <= 0:
        raise ValueError("sample_count must be a positive integer")
    return int(value)


def summarize_replicate_estimates(
    estimates: Sequence[Binary64Scalar],
    *,
    seeds: Sequence[int],
    sample_count: int,
    confidence_level: Binary64Scalar = 0.95,
    convergence_tolerance: Optional[Binary64Scalar] = None,
) -> dict[str, Any]:
    """Summarize independent seeded estimates without claiming global validity.

    The Student-t interval describes the mean across the supplied independent
    RNG streams.  It is not a confidence statement about unsampled inputs,
    alternative interventions, model classes, or deployment behavior.
    Cumulative-mean change is reported as an order-dependent convergence
    diagnostic and becomes a pass/fail statement only when the caller supplies
    an explicit tolerance.
    """

    values = _validated_estimates(estimates)
    validated_seeds = _validated_seeds(seeds, len(values))
    sample_count = _positive_sample_count(sample_count)
    validated_confidence_level = _lossless_binary64(confidence_level, name="confidence_level")
    if not 0.0 < validated_confidence_level < 1.0:
        raise ValueError("confidence_level must be finite and lie in (0, 1)")
    validated_convergence_tolerance: Optional[float] = None
    if convergence_tolerance is not None:
        validated_convergence_tolerance = _lossless_binary64(
            convergence_tolerance, name="convergence_tolerance"
        )
        if validated_convergence_tolerance < 0.0:
            raise ValueError("convergence_tolerance must be finite and non-negative")

    mean = float(_stable_mean(values))
    cumulative_means = [float(_stable_mean(values[: index + 1])) for index in range(len(values))]
    exact_values = [Fraction.from_float(float(value)) for value in values]
    exact_cumulative_means: list[Fraction] = []
    exact_running_total = Fraction(0)
    for index, value in enumerate(exact_values):
        exact_running_total += value
        exact_cumulative_means.append(exact_running_total / (index + 1))
    terminal_change: Optional[float]
    exact_terminal_change: Optional[Fraction]
    if len(cumulative_means) < 2:
        terminal_change = None
        exact_terminal_change = None
        terminal_change_defined = False
        terminal_change_reason = "at_least_two_independent_replicates_are_required"
        terminal_change_computation = None
    else:
        exact_terminal_change = abs(exact_cumulative_means[-1] - exact_cumulative_means[-2])
        rounded_terminal_change = float(exact_terminal_change)
        terminal_change_computation = "exact_binary64_rational_cumulative_means"
        if not np.isfinite(rounded_terminal_change):
            terminal_change = None
            terminal_change_defined = False
            terminal_change_reason = "terminal_change_is_not_representable_as_float64"
        elif rounded_terminal_change == 0.0 and exact_terminal_change != 0:
            terminal_change = None
            terminal_change_defined = False
            terminal_change_reason = "positive_terminal_change_is_not_representable_as_float64"
        else:
            terminal_change = rounded_terminal_change
            terminal_change_defined = True
            terminal_change_reason = None

    converged_under_tolerance: Optional[bool]
    if validated_convergence_tolerance is None or exact_terminal_change is None:
        converged_under_tolerance = None
    else:
        converged_under_tolerance = exact_terminal_change <= Fraction.from_float(
            validated_convergence_tolerance
        )

    standard_deviation: Optional[float]
    standard_error: Optional[float]
    standard_error_reason: Optional[str]
    confidence_interval: Optional[list[float]]
    confidence_interval_reason: Optional[str]
    confidence_interval_computation: Optional[str]
    if len(values) < 2:
        standard_deviation = None
        standard_error = None
        standard_error_defined = False
        standard_error_reason = "at_least_two_independent_replicates_are_required"
        confidence_interval = None
        confidence_interval_defined = False
        confidence_interval_reason = "at_least_two_independent_replicates_are_required"
        confidence_interval_computation = None
    else:
        standard_deviation = float(_stable_std(values, ddof=1))
        # Use the upper tail directly: ``(1 + confidence_level) / 2`` rounds
        # to 1.0 for the largest valid confidence levels even though the
        # complementary tail remains positive and has a finite quantile.
        upper_tail_probability = (1.0 - validated_confidence_level) / 2.0
        critical = float(stats.t.isf(upper_tail_probability, len(values) - 1))
        if not np.isfinite(critical) or critical < 0.0:
            raise FloatingPointError("Student-t critical value is not representable")
        direct_standard_error = standard_deviation / np.sqrt(len(values))
        direct_margin = critical * direct_standard_error
        needs_fused_interval = (
            (standard_deviation > 0.0 and direct_standard_error == 0.0)
            or (standard_deviation == 0.0 and np.any(values != values[0]))
            or (direct_standard_error > 0.0 and direct_margin == 0.0)
            or not np.isfinite(direct_margin)
        )
        if needs_fused_interval:
            standard_error, confidence_interval = _fused_student_t_interval(values, critical)
            standard_error_defined = standard_error is not None
            standard_error_reason = (
                None
                if standard_error_defined
                else "positive_standard_error_is_not_representable_as_float64"
            )
            confidence_interval_computation = (
                "high_precision_fused_student_t_from_exact_binary64_replicates"
            )
        else:
            standard_error = direct_standard_error
            margin = critical * standard_error
            lower = mean - margin
            upper = mean + margin
            if not np.isfinite(lower) or not np.isfinite(upper):
                raise FloatingPointError("replicate confidence interval is not representable")
            confidence_interval = [float(lower), float(upper)]
            standard_error_defined = True
            standard_error_reason = None
            confidence_interval_computation = "direct_float64_student_t"
        confidence_interval_defined = True
        confidence_interval_reason = None

    return {
        "estimate": mean,
        "replicate_estimates": values.tolist(),
        "replicate_count": len(values),
        "seeds": validated_seeds,
        "sample_count_per_replicate": sample_count,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "standard_error_defined": standard_error_defined,
        "standard_error_reason": standard_error_reason,
        "confidence_level": validated_confidence_level,
        "confidence_interval": confidence_interval,
        "confidence_interval_defined": confidence_interval_defined,
        "confidence_interval_reason": confidence_interval_reason,
        "confidence_interval_computation": confidence_interval_computation,
        "confidence_interval_kind": "student_t_mean_of_independent_seeded_replicates",
        "confidence_interval_scope": "mean_over_the_supplied_rng_streams_only",
        "cumulative_means": cumulative_means,
        "terminal_cumulative_mean_change": terminal_change,
        "terminal_cumulative_mean_change_defined": terminal_change_defined,
        "terminal_cumulative_mean_change_reason": terminal_change_reason,
        "terminal_cumulative_mean_change_computation": terminal_change_computation,
        "convergence_tolerance": validated_convergence_tolerance,
        "converged_under_declared_tolerance": converged_under_tolerance,
        "convergence_diagnostic_only": True,
        "finite_estimate_is_global_proof": False,
    }


def run_seeded_replicates(
    estimator: Callable[..., Binary64Scalar],
    *,
    seeds: Sequence[int],
    sample_count: int,
    confidence_level: Binary64Scalar = 0.95,
    convergence_tolerance: Optional[Binary64Scalar] = None,
) -> dict[str, Any]:
    """Run ``estimator(seed=...)`` for explicit independent RNG streams."""

    if not callable(estimator):
        raise TypeError("estimator must be callable")
    # Validate before the first potentially expensive or stateful estimator call.
    if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Sequence):
        raise TypeError("seeds must be a sequence of non-negative integers")
    validated_seeds = _validated_seeds(seeds, len(seeds))
    sample_count = _positive_sample_count(sample_count)
    estimates: list[float] = []
    for seed in validated_seeds:
        try:
            value = estimator(seed=seed)
        except Exception as exc:
            raise RuntimeError(f"estimator failed for seed {seed}: {exc}") from exc
        estimates.append(_lossless_binary64(value, name=f"estimator result for seed {seed}"))
    return summarize_replicate_estimates(
        estimates,
        seeds=validated_seeds,
        sample_count=sample_count,
        confidence_level=confidence_level,
        convergence_tolerance=convergence_tolerance,
    )


def evaluate_intervention_sensitivity(
    interventions: Mapping[str, Any],
    evaluator: Callable[[Any], Binary64Scalar],
    *,
    intervention_contract: str,
) -> dict[str, Any]:
    """Evaluate named references and fingerprint their exact supported values."""

    if not isinstance(interventions, Mapping) or len(interventions) < 2:
        raise ValueError("interventions must map at least two names to prespecified references")
    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    if not isinstance(intervention_contract, str) or not intervention_contract.strip():
        raise ValueError("intervention_contract must be a non-empty caller declaration")

    scores: dict[str, float] = {}
    fingerprints: dict[str, str] = {}
    for name, intervention in interventions.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("intervention names must be non-empty strings")
        fingerprints[name] = _reference_fingerprint(intervention)
        try:
            value = evaluator(copy.deepcopy(intervention))
        except Exception as exc:
            raise RuntimeError(f"evaluator failed for intervention {name!r}: {exc}") from exc
        numeric = _lossless_binary64(value, name=f"evaluator result for intervention {name!r}")
        scores[name] = numeric

    ordered_values = np.asarray(list(scores.values()), dtype=np.float64)
    minimum = float(np.min(ordered_values))
    maximum = float(np.max(ordered_values))
    sensitivity_range = float(_stable_sum(np.asarray([maximum, -minimum])))
    return {
        "intervention_contract": intervention_contract.strip(),
        "intervention_names": list(scores),
        "intervention_reference_fingerprint_schema": _INTERVENTION_FINGERPRINT_SCHEMA,
        "intervention_reference_fingerprints": fingerprints,
        "scores": scores,
        "minimum": minimum,
        "maximum": maximum,
        "sensitivity_range": sensitivity_range,
        "conclusion_invariant_across_prespecified_interventions": bool(
            np.all(ordered_values == ordered_values[0])
        ),
        "universal_default_claimed": False,
        "shared_intervention_contract_required_for_comparison": True,
    }


def compare_intervention_sensitivity_reports(
    reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare reports only under one contract, name order, and reference identity."""

    if not isinstance(reports, Mapping) or len(reports) < 2:
        raise ValueError("reports must contain at least two named sensitivity reports")
    contracts = set()
    intervention_names = set()
    reference_fingerprint_sequences = set()
    collected: dict[str, dict[str, float]] = {}
    for name, report in reports.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("report names must be non-empty strings")
        if not isinstance(report, Mapping):
            raise TypeError("every sensitivity report must be a mapping")
        contract = report.get("intervention_contract")
        names = report.get("intervention_names")
        scores = report.get("scores")
        fingerprint_schema = report.get("intervention_reference_fingerprint_schema")
        fingerprints = report.get("intervention_reference_fingerprints")
        if not isinstance(contract, str) or not contract.strip():
            raise ValueError("every sensitivity report must record an intervention_contract")
        if not isinstance(names, list) or any(not isinstance(value, str) for value in names):
            raise ValueError("every sensitivity report must record ordered intervention_names")
        if not isinstance(scores, Mapping) or list(scores) != names:
            raise ValueError("every sensitivity report must align scores to intervention_names")
        if fingerprint_schema != _INTERVENTION_FINGERPRINT_SCHEMA:
            raise ValueError(
                "every sensitivity report must record the supported reference fingerprint schema"
            )
        if not isinstance(fingerprints, Mapping) or list(fingerprints) != names:
            raise ValueError(
                "every sensitivity report must align reference fingerprints to "
                "intervention_names"
            )
        fingerprint_sequence = []
        for intervention_name, fingerprint in fingerprints.items():
            if (
                not isinstance(fingerprint, str)
                or not fingerprint.startswith("sha256:")
                or len(fingerprint) != len("sha256:") + 64
                or any(character not in "0123456789abcdef" for character in fingerprint[7:])
            ):
                raise ValueError("reference fingerprints must be canonical SHA-256 strings")
            fingerprint_sequence.append((intervention_name, fingerprint))
        numeric_scores: dict[str, float] = {}
        for intervention_name, value in scores.items():
            numeric_scores[intervention_name] = _lossless_binary64(
                value,
                name=f"sensitivity report score {intervention_name!r}",
            )
        contracts.add(contract.strip())
        intervention_names.add(tuple(names))
        reference_fingerprint_sequences.add(tuple(fingerprint_sequence))
        collected[name] = numeric_scores
    if len(contracts) != 1:
        raise ValueError("mixed intervention_contract values are scientifically incomparable")
    if len(intervention_names) != 1:
        raise ValueError("sensitivity reports use different prespecified interventions")
    if len(reference_fingerprint_sequences) != 1:
        raise ValueError(
            "sensitivity reports use different intervention reference values or identities"
        )
    return {
        "intervention_contract": next(iter(contracts)),
        "intervention_names": list(next(iter(intervention_names))),
        "intervention_reference_fingerprint_schema": _INTERVENTION_FINGERPRINT_SCHEMA,
        "intervention_reference_fingerprints": dict(next(iter(reference_fingerprint_sequences))),
        "scores_by_report": collected,
        "automatic_best_estimator_selected": False,
    }
