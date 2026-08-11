"""Checks and diagnostics related to attribution axioms.

The functions in this module deliberately distinguish exact, pointwise checks
from global axioms:

* :func:`compute_completeness` computes the pointwise residual in the
  completeness identity for one *specified scalar model output and baseline*.
* :func:`compute_non_sensitivity` implements Nguyen and Martinez's cardinality
  metric when the caller supplies the reference set of non-dependent features.
  A finite perturbation helper is available only as a local diagnostic; it
  cannot prove mathematical independence.
* :func:`compute_input_invariance_pytorch` tests the particular compensated
  input-translation construction of Kindermans et al. for a deliberately
  restricted model family.  :func:`compute_input_invariance` is retained as an
  explicitly warned, uncompensated translation-sensitivity diagnostic.
* :func:`compute_symmetry` measures attribution disparity for pairs whose
  symmetry must already have been established by the caller.  It does not prove
  that a model is symmetric.

None of these finite checks establishes Sundararajan et al.'s global
Implementation Invariance axiom, which quantifies over functionally equivalent
implementations and every input.

This module also does not implement Sensitivity(a). Completeness implies
Sensitivity(a) only in its special one-differing-feature antecedent, and a
finite completeness residual does not prove that universal property. Nguyen
and Martinez's symmetric-difference Non-Sensitivity metric is merely aligned
with Sensitivity(b) (Dummy): it additionally penalises zero attribution on a
feature outside the supplied ``X0`` set.

Primary references
------------------
Sundararajan, Taly, and Yan (2017), "Axiomatic Attribution for Deep
Networks", ICML.  https://proceedings.mlr.press/v70/sundararajan17a.html

Kindermans et al. (2017), "The (Un)reliability of Saliency Methods".
https://arxiv.org/abs/1711.00867

Nguyen and Martinez (2020), "On Quantitative Aspects of Model
Interpretability".  https://arxiv.org/abs/2007.07584
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Mapping, Sequence
from decimal import Decimal, localcontext
from numbers import Real
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation
from explainiverse.evaluation._utils import _stable_mean, _stable_std, _stable_sum


def _as_finite_vector(
    value,
    name: str,
    *,
    expected_length: Optional[int] = None,
) -> np.ndarray:
    """Return a non-empty, finite, one-dimensional float vector."""
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric one-dimensional array") from exc
    if vector.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {vector.shape}")
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty")
    if expected_length is not None and vector.size != expected_length:
        raise ValueError(
            f"{name} length ({vector.size}) must match expected length " f"({expected_length})"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _coerce_scalar_output(value, name: str = "model_fn output") -> float:
    """Accept exactly one finite numeric output; never select an output silently."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        array = np.asarray(value)
    except Exception as exc:  # pragma: no cover - unusual third-party objects
        raise ValueError(f"{name} must be scalar") from exc
    if array.size != 1:
        raise ValueError(
            f"{name} must contain exactly one scalar value; got shape "
            f"{array.shape}. Select the target/output explicitly."
        )
    try:
        scalar = float(array.reshape(-1)[0])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _validate_model_fn(model_fn: Callable) -> None:
    if not callable(model_fn):
        raise TypeError(f"model_fn must be callable, got {type(model_fn).__name__}")


def _validate_bool(value, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def _safe_model_output(
    model_fn: Callable,
    x: np.ndarray,
    output_func: Optional[Callable] = None,
) -> float:
    """Evaluate a model and require an explicitly selected scalar output.

    ``output_func``, when supplied, receives the model's raw return value.  It
    may therefore select a class/output and transform its score space.  The
    attributions being evaluated must have been computed for that exact scalar.
    """
    _validate_model_fn(model_fn)
    if output_func is not None and not callable(output_func):
        raise TypeError("output_func must be callable or None")
    result = model_fn(x)
    if output_func is not None:
        result = output_func(result)
    return _coerce_scalar_output(result)


def _extract_attribution_vector(explanation: Explanation) -> np.ndarray:
    """Extract an ordered attribution vector without inventing missing values."""
    if not isinstance(explanation, Explanation):
        raise TypeError("explainer.explain() must return an Explanation")
    attributions = explanation.explanation_data.get("feature_attributions")
    if not isinstance(attributions, Mapping) or not attributions:
        raise ValueError("No feature attributions found in explanation")
    feature_names = explanation.feature_names
    if not feature_names:
        raise ValueError("Explanation.feature_names is required to establish attribution order")
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("Explanation.feature_names must be unique")
    missing = [name for name in feature_names if name not in attributions]
    extra = [name for name in attributions if name not in feature_names]
    if missing or extra:
        raise ValueError(
            "feature_attributions must match feature_names exactly; "
            f"missing={missing}, extra={extra}"
        )
    return _as_finite_vector([attributions[name] for name in feature_names], "attributions")


def _get_explanation_vector(
    explainer: BaseExplainer,
    instance: np.ndarray,
    n_features: int,
    explain_kwargs: Optional[Dict[str, object]] = None,
    verify_determinism: bool = True,
) -> np.ndarray:
    verify_determinism = _validate_bool(verify_determinism, "verify_determinism")
    if not hasattr(explainer, "explain") or not callable(explainer.explain):
        raise TypeError("explainer must provide a callable explain() method")
    kwargs = {} if explain_kwargs is None else dict(explain_kwargs)
    explanation = explainer.explain(instance, **kwargs)
    vector = _extract_attribution_vector(explanation)
    if vector.size != n_features:
        raise ValueError(
            f"attributions length ({vector.size}) must match instance length " f"({n_features})"
        )
    if verify_determinism:
        repeated = _extract_attribution_vector(explainer.explain(instance, **kwargs))
        if repeated.size != n_features:
            raise ValueError(
                f"repeated attributions length ({repeated.size}) must match "
                f"instance length ({n_features})"
            )
        _assert_deterministic(vector, repeated, "explainer.explain()")
    return vector


def _validate_batch_inputs(X, max_instances: Optional[int]) -> Tuple[np.ndarray, int]:
    try:
        batch = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must be a finite numeric 2D array") from exc
    if batch.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {batch.shape}")
    if batch.shape[0] == 0 or batch.shape[1] == 0:
        raise ValueError("X must contain at least one instance and one feature")
    if not np.all(np.isfinite(batch)):
        raise ValueError("X must contain only finite values")
    if max_instances is None:
        return batch, batch.shape[0]
    if isinstance(max_instances, bool) or not isinstance(max_instances, (int, np.integer)):
        raise TypeError("max_instances must be a positive integer or None")
    if max_instances <= 0:
        raise ValueError("max_instances must be positive")
    return batch, min(batch.shape[0], int(max_instances))


def _validate_attribution_batch(attributions_list, n: int) -> None:
    if attributions_list is None:
        return
    if not isinstance(attributions_list, Sequence) and not isinstance(
        attributions_list, np.ndarray
    ):
        raise TypeError("attributions_list must be a sequence")
    if len(attributions_list) < n:
        raise ValueError(
            f"attributions_list has {len(attributions_list)} entries but {n} are required"
        )


def _summarize(scores: List[float]) -> Dict[str, object]:
    values = _as_finite_vector(scores, "scores")
    mean = float(_stable_mean(values))
    std = float(_stable_std(values))
    if not np.isfinite(mean) or not np.isfinite(std):
        raise FloatingPointError("score summary is not representable")
    return {
        "mean": mean,
        "std": std,
        "max": float(np.max(values)),
        "min": float(np.min(values)),
        "scores": [float(value) for value in values],
        "n_evaluated": int(values.size),
    }


def _rms_difference(left: np.ndarray, right: np.ndarray, context: str) -> float:
    """Compute RMS(left-right) without materializing overflowing squares."""
    with np.errstate(over="ignore", invalid="ignore"):
        difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    if np.all(np.isfinite(difference)):
        scale = float(np.max(np.abs(difference)))
        if scale == 0.0:
            return 0.0
        normalized = difference / scale
        with np.errstate(over="ignore", invalid="ignore"):
            result = float(scale * np.sqrt(float(_stable_mean(normalized * normalized))))
        exact_nonzero = True
    else:
        with localcontext() as decimal_context:
            decimal_context.prec = 1600
            squared = sum(
                (
                    (Decimal.from_float(float(left_value)) - Decimal.from_float(float(right_value)))
                    ** 2
                    for left_value, right_value in zip(left, right)
                ),
                start=Decimal(0),
            )
            exact = (squared / Decimal(len(left))).sqrt()
            result = float(exact)
            exact_nonzero = exact != 0
    if not np.isfinite(result) or (result == 0.0 and exact_nonzero):
        raise FloatingPointError(f"{context} RMS is not representable")
    return result


def _mean_absolute_pair_difference(values: np.ndarray, pairs: List[Tuple[int, int]]) -> float:
    """Average pair disparities using exact fallback for overflowing differences."""
    with np.errstate(over="ignore", invalid="ignore"):
        differences = np.asarray(
            [abs(values[first] - values[second]) for first, second in pairs],
            dtype=np.float64,
        )
    if np.all(np.isfinite(differences)):
        result = float(_stable_mean(differences))
        exact_nonzero = bool(np.any(differences != 0.0))
    else:
        with localcontext() as decimal_context:
            decimal_context.prec = 1600
            total = sum(
                (
                    abs(
                        Decimal.from_float(float(values[first]))
                        - Decimal.from_float(float(values[second]))
                    )
                    for first, second in pairs
                ),
                start=Decimal(0),
            )
            exact = total / Decimal(len(pairs))
            result = float(exact)
            exact_nonzero = exact != 0
    if not np.isfinite(result) or (result == 0.0 and exact_nonzero):
        raise FloatingPointError("symmetry disparity is not representable")
    return result


def compute_completeness(
    attributions: np.ndarray,
    model_fn: Callable,
    instance: np.ndarray,
    baseline: Optional[Union[np.ndarray, float]] = None,
    output_func: Optional[Callable] = None,
) -> float:
    """Return ``abs(sum(attributions) - (F(x) - F(baseline)))``.

    This is the pointwise completeness residual from Sundararajan et al.
    ``model_fn`` (or ``output_func(model_fn(...))``) must return the exact
    scalar target and score space explained by ``attributions``.  Multi-output
    results are rejected rather than silently taking their first element.

    ``baseline=None`` means an all-zero baseline.  The caller is responsible
    for supplying the same baseline that the attribution method used; this
    cannot be inferred from a bare attribution vector.
    """
    attrs = _as_finite_vector(attributions, "attributions")
    x = _as_finite_vector(instance, "instance")
    if attrs.size != x.size:
        raise ValueError(
            f"attributions length ({attrs.size}) must match instance length " f"({x.size})"
        )
    if baseline is None:
        baseline_vector = np.zeros_like(x)
    elif isinstance(baseline, Real) and not isinstance(baseline, bool):
        if not np.isfinite(float(baseline)):
            raise ValueError("baseline must be finite")
        baseline_vector = np.full_like(x, float(baseline))
    else:
        baseline_vector = _as_finite_vector(baseline, "baseline", expected_length=x.size)
    output_at_x = _safe_model_output(model_fn, x, output_func)
    output_at_baseline = _safe_model_output(model_fn, baseline_vector, output_func)
    residual = float(
        _stable_sum(np.concatenate((attrs, np.array([-output_at_x, output_at_baseline]))))
    )
    if not np.isfinite(residual):
        raise FloatingPointError("completeness residual is not representable")
    return abs(residual)


def compute_completeness_score(
    explainer: BaseExplainer,
    model_fn: Callable,
    instance: np.ndarray,
    baseline: Optional[Union[np.ndarray, float]] = None,
    output_func: Optional[Callable] = None,
    explain_kwargs: Optional[Dict[str, object]] = None,
    verify_determinism: bool = True,
) -> float:
    """Generate attributions and compute their completeness residual.

    Use ``explain_kwargs`` to fix the explainer's target when required.  The
    caller must ensure that target, score space, and baseline match ``model_fn``
    and ``output_func``.
    """
    x = _as_finite_vector(instance, "instance")
    attrs = _get_explanation_vector(explainer, x, x.size, explain_kwargs, verify_determinism)
    return compute_completeness(attrs, model_fn, x, baseline, output_func)


def compute_batch_completeness(
    attributions_list: Optional[List[np.ndarray]] = None,
    explainer: Optional[BaseExplainer] = None,
    model_fn: Optional[Callable] = None,
    X: Optional[np.ndarray] = None,
    baseline: Optional[Union[np.ndarray, float]] = None,
    output_func: Optional[Callable] = None,
    max_instances: Optional[int] = None,
    explain_kwargs: Optional[Dict[str, object]] = None,
    verify_determinism: bool = True,
) -> Dict[str, object]:
    """Compute completeness residuals for a batch, failing on any bad row."""
    verify_determinism = _validate_bool(verify_determinism, "verify_determinism")
    if model_fn is None:
        raise ValueError("model_fn is required for Completeness evaluation")
    if X is None:
        raise ValueError("X (input data) is required for Completeness evaluation")
    if attributions_list is None and explainer is None:
        raise ValueError("Either attributions_list or explainer must be provided")
    batch, n = _validate_batch_inputs(X, max_instances)
    _validate_attribution_batch(attributions_list, n)
    scores: List[float] = []
    for index in range(n):
        if attributions_list is not None:
            score = compute_completeness(
                attributions_list[index], model_fn, batch[index], baseline, output_func
            )
        else:
            assert explainer is not None
            score = compute_completeness_score(
                explainer,
                model_fn,
                batch[index],
                baseline,
                output_func,
                explain_kwargs,
                verify_determinism,
            )
        scores.append(score)
    return _summarize(scores)


def _detect_non_sensitive_features(
    model_fn: Callable,
    instance: np.ndarray,
    n_perturbations: int = 10,
    perturbation_scale: float = 0.1,
    tolerance: float = 1e-5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Return a *local finite-perturbation proxy* for non-sensitive features.

    A ``True`` entry only means that none of the sampled local interventions
    changed the selected scalar output beyond ``tolerance``.  It does not prove
    the global functional independence in Sensitivity(b), nor does it implement
    the distributional expectation defining Nguyen and Martinez's ``X0``.
    Consequently this helper is never used implicitly by
    :func:`compute_non_sensitivity`.
    """
    _validate_model_fn(model_fn)
    x = _as_finite_vector(instance, "instance")
    if isinstance(n_perturbations, bool) or not isinstance(n_perturbations, (int, np.integer)):
        raise TypeError("n_perturbations must be a positive integer")
    if n_perturbations <= 0:
        raise ValueError("n_perturbations must be positive")
    if not isinstance(perturbation_scale, Real) or perturbation_scale <= 0:
        raise ValueError("perturbation_scale must be positive")
    if not np.isfinite(float(perturbation_scale)):
        raise ValueError("perturbation_scale must be finite")
    if not isinstance(tolerance, Real) or tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be a finite non-negative number")

    rng = np.random.RandomState(seed)
    original = _safe_model_output(model_fn, x)
    mask = np.ones(x.size, dtype=bool)
    for feature in range(x.size):
        scale = float(perturbation_scale) * max(1.0, abs(float(x[feature])))
        for _ in range(int(n_perturbations)):
            perturbed = x.copy()
            perturbed[feature] += rng.normal(0.0, scale)
            if abs(_safe_model_output(model_fn, perturbed) - original) > tolerance:
                mask[feature] = False
                break
    return mask


def compute_non_sensitivity(
    attributions: np.ndarray,
    model_fn: Callable,
    instance: np.ndarray,
    non_sensitive_features: Optional[np.ndarray] = None,
    n_perturbations: int = 10,
    perturbation_scale: float = 0.1,
    tolerance: float = 1e-5,
    normalize: bool = False,
    seed: Optional[int] = None,
    attribution_tolerance: float = 0.0,
) -> float:
    """Compute Nguyen and Martinez's non-sensitivity metric ``|A0 Δ X0|``.

    ``A0`` is the set of features assigned zero attribution and ``X0`` is the
    supplied ``non_sensitive_features`` mask.  The unnormalised result is the
    cardinality of their symmetric difference; with ``normalize=True`` it is
    divided by the number of features.

    The caller must supply ``X0``.  Finite perturbations cannot establish the
    required functional non-dependence and are therefore not substituted
    automatically.  The legacy perturbation arguments remain in the signature
    for compatibility but are ignored here; use
    :func:`_detect_non_sensitive_features` explicitly for a labelled proxy.
    ``model_fn`` is likewise retained for API compatibility and callable
    validation; the set formula itself does not evaluate the model once ``X0``
    is supplied.
    """
    normalize = _validate_bool(normalize, "normalize")
    _validate_model_fn(model_fn)
    attrs = _as_finite_vector(attributions, "attributions")
    x = _as_finite_vector(instance, "instance")
    if attrs.size != x.size:
        raise ValueError(
            f"attributions length ({attrs.size}) must match instance length " f"({x.size})"
        )
    if non_sensitive_features is None:
        raise ValueError(
            "non_sensitive_features must be supplied; finite local perturbations "
            "cannot prove the reference set X0 required by Non-Sensitivity"
        )
    mask = np.asarray(non_sensitive_features)
    if mask.ndim != 1 or mask.size != attrs.size:
        raise ValueError(
            f"non_sensitive_features length ({mask.size}) must match instance "
            f"length ({attrs.size})"
        )
    if mask.dtype.kind != "b":
        raise TypeError("non_sensitive_features must be a boolean mask")
    if not isinstance(attribution_tolerance, Real) or attribution_tolerance < 0:
        raise ValueError("attribution_tolerance must be non-negative")
    if not np.isfinite(float(attribution_tolerance)):
        raise ValueError("attribution_tolerance must be finite")
    attribution_zero = np.abs(attrs) <= float(attribution_tolerance)
    mismatches = int(np.count_nonzero(np.logical_xor(attribution_zero, mask)))
    return float(mismatches / attrs.size if normalize else mismatches)


def compute_non_sensitivity_score(
    explainer: BaseExplainer,
    model_fn: Callable,
    instance: np.ndarray,
    non_sensitive_features: Optional[np.ndarray] = None,
    n_perturbations: int = 10,
    perturbation_scale: float = 0.1,
    tolerance: float = 1e-5,
    normalize: bool = False,
    seed: Optional[int] = None,
    attribution_tolerance: float = 0.0,
    explain_kwargs: Optional[Dict[str, object]] = None,
    verify_determinism: bool = True,
) -> float:
    """Generate attributions and compute ``|A0 Δ X0|``."""
    normalize = _validate_bool(normalize, "normalize")
    verify_determinism = _validate_bool(verify_determinism, "verify_determinism")
    x = _as_finite_vector(instance, "instance")
    attrs = _get_explanation_vector(explainer, x, x.size, explain_kwargs, verify_determinism)
    return compute_non_sensitivity(
        attrs,
        model_fn,
        x,
        non_sensitive_features,
        n_perturbations,
        perturbation_scale,
        tolerance,
        normalize,
        seed,
        attribution_tolerance,
    )


def compute_batch_non_sensitivity(
    attributions_list: Optional[List[np.ndarray]] = None,
    explainer: Optional[BaseExplainer] = None,
    model_fn: Optional[Callable] = None,
    X: Optional[np.ndarray] = None,
    non_sensitive_features: Optional[np.ndarray] = None,
    n_perturbations: int = 10,
    perturbation_scale: float = 0.1,
    tolerance: float = 1e-5,
    normalize: bool = False,
    seed: Optional[int] = None,
    max_instances: Optional[int] = None,
    attribution_tolerance: float = 0.0,
    explain_kwargs: Optional[Dict[str, object]] = None,
    verify_determinism: bool = True,
) -> Dict[str, object]:
    """Compute non-sensitivity mismatch counts, failing on any bad row."""
    normalize = _validate_bool(normalize, "normalize")
    verify_determinism = _validate_bool(verify_determinism, "verify_determinism")
    if model_fn is None:
        raise ValueError("model_fn is required for Non-Sensitivity evaluation")
    if X is None:
        raise ValueError("X (input data) is required for Non-Sensitivity evaluation")
    if attributions_list is None and explainer is None:
        raise ValueError("Either attributions_list or explainer must be provided")
    if non_sensitive_features is None:
        raise ValueError("non_sensitive_features must be supplied")
    batch, n = _validate_batch_inputs(X, max_instances)
    _validate_attribution_batch(attributions_list, n)
    scores: List[float] = []
    for index in range(n):
        if attributions_list is not None:
            attrs = attributions_list[index]
        else:
            assert explainer is not None
            attrs = _get_explanation_vector(
                explainer,
                batch[index],
                batch.shape[1],
                explain_kwargs,
                verify_determinism,
            )
        scores.append(
            compute_non_sensitivity(
                attrs,
                model_fn,
                batch[index],
                non_sensitive_features,
                n_perturbations,
                perturbation_scale,
                tolerance,
                normalize,
                None if seed is None else seed + index,
                attribution_tolerance,
            )
        )
    return _summarize(scores)


def _build_shift(
    instance: np.ndarray,
    shift: Optional[Union[np.ndarray, float]],
    seed: Optional[int],
) -> np.ndarray:
    if shift is None:
        shift_vector = np.random.RandomState(seed).uniform(-1.0, 1.0, instance.size)
    elif isinstance(shift, Real) and not isinstance(shift, bool):
        shift_vector = np.full(instance.size, float(shift), dtype=np.float64)
    else:
        shift_vector = _as_finite_vector(shift, "shift", expected_length=instance.size)
    if not np.all(np.isfinite(shift_vector)):
        raise ValueError("shift must contain only finite values")
    if not np.any(shift_vector != 0.0):
        raise ValueError("shift must contain at least one non-zero value")
    return shift_vector


def _explanation_result(
    explain_func: Callable,
    instance: np.ndarray,
    *,
    expected_length: int,
    label: str,
) -> np.ndarray:
    return _as_finite_vector(explain_func(instance), label, expected_length=expected_length)


def _assert_deterministic(first: np.ndarray, second: np.ndarray, label: str) -> None:
    if not np.allclose(first, second, rtol=1e-7, atol=1e-10):
        raise RuntimeError(
            f"{label} is stochastic across repeated identical calls; control the "
            "explainer RNG or set verify_determinism=False and treat the result "
            "as one noisy realization"
        )


def _translation_sensitivity(
    explain_func: Callable,
    instance: np.ndarray,
    shift: Optional[Union[np.ndarray, float]],
    seed: Optional[int],
    verify_determinism: bool,
) -> float:
    verify_determinism = _validate_bool(verify_determinism, "verify_determinism")
    if not callable(explain_func):
        raise TypeError(f"explain_func must be callable, got {type(explain_func).__name__}")
    x = _as_finite_vector(instance, "instance")
    shift_vector = _build_shift(x, shift, seed)
    original = _explanation_result(
        explain_func, x, expected_length=x.size, label="original attributions"
    )
    shifted_x = x + shift_vector
    if not np.all(np.isfinite(shifted_x)):
        raise FloatingPointError("shifted instance is not representable")
    shifted = _explanation_result(
        explain_func,
        shifted_x,
        expected_length=x.size,
        label="shifted attributions",
    )
    if verify_determinism:
        original_again = _explanation_result(
            explain_func,
            x,
            expected_length=x.size,
            label="repeated original attributions",
        )
        shifted_again = _explanation_result(
            explain_func,
            shifted_x,
            expected_length=x.size,
            label="repeated shifted attributions",
        )
        _assert_deterministic(original, original_again, "explain_func")
        _assert_deterministic(shifted, shifted_again, "explain_func")
    return _rms_difference(original, shifted, "input-invariance")


def compute_input_invariance(
    explain_func: Callable,
    instance: np.ndarray,
    shift: Optional[Union[np.ndarray, float]] = None,
    seed: Optional[int] = None,
    verify_determinism: bool = True,
) -> float:
    """Compute an uncompensated input-translation sensitivity diagnostic.

    The result is the per-feature root-mean-square change between ``E(x)`` and
    ``E(x + shift)``.  Because no corresponding transformed model is supplied,
    this function does **not** test Kindermans et al.'s Input Invariance axiom:
    the model prediction itself may have changed.  A warning is emitted on each
    direct call to prevent the proxy from being reported as the canonical test.
    """
    warnings.warn(
        "compute_input_invariance is an uncompensated translation-sensitivity "
        "diagnostic, not a test of the Input Invariance axiom; use "
        "compute_input_invariance_pytorch for its supported compensated case",
        RuntimeWarning,
        stacklevel=2,
    )
    return _translation_sensitivity(explain_func, instance, shift, seed, verify_determinism)


def _torch_attributions(
    explain_func: Callable,
    model,
    instance: np.ndarray,
    n_features: int,
    label: str,
) -> np.ndarray:
    return _as_finite_vector(explain_func(model, instance), label, expected_length=n_features)


def compute_input_invariance_pytorch(
    model,
    explain_func: Callable,
    instance: np.ndarray,
    shift: Optional[Union[np.ndarray, float]] = None,
    seed: Optional[int] = None,
    verify_determinism: bool = True,
) -> float:
    """Test the compensated constant-input-shift construction for PyTorch.

    To make the functional compensation algebra checkable, support is limited
    to an ``nn.Sequential`` model whose first module is ``nn.Linear`` and whose
    input is a one-dimensional feature vector.  The model must be in evaluation
    mode.  The function deep-copies the model, sets ``b2 = b1 - W @ shift``,
    verifies equality of the complete model outputs at the paired inputs, then
    returns the RMS attribution difference.

    A zero result is evidence only for this transformation and tested input.  It
    does not establish global Input Invariance or Implementation Invariance.
    """
    verify_determinism = _validate_bool(verify_determinism, "verify_determinism")
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("PyTorch is required for compute_input_invariance_pytorch") from exc
    if not callable(explain_func):
        raise TypeError(f"explain_func must be callable, got {type(explain_func).__name__}")
    if not isinstance(model, nn.Sequential) or len(model) == 0 or type(model[0]) is not nn.Linear:
        raise TypeError(
            "model must be nn.Sequential with nn.Linear as its first module; "
            "arbitrary module registration order cannot prove compensation"
        )
    if any(module.training for module in model.modules()):
        raise ValueError("model must be in evaluation mode to exclude stochastic layers")

    x = _as_finite_vector(instance, "instance")
    first_layer = cast(Any, model[0])
    if x.size != first_layer.in_features:
        raise ValueError(
            f"instance length ({x.size}) must match first layer in_features "
            f"({first_layer.in_features})"
        )
    shift_vector = _build_shift(x, shift, seed)

    # Attribution callbacks receive private copies so even a stateful or
    # ill-behaved callback cannot mutate the caller's model.
    reference_model = copy.deepcopy(model)
    compensated_model = copy.deepcopy(model)
    compensated_layer = cast(Any, compensated_model[0])
    with torch.no_grad():
        shift_tensor = torch.as_tensor(
            shift_vector,
            dtype=compensated_layer.weight.dtype,
            device=compensated_layer.weight.device,
        )
        adjustment = compensated_layer.weight @ shift_tensor
        if compensated_layer.bias is None:
            compensated_layer.bias = nn.Parameter(
                -adjustment.clone(),
                requires_grad=compensated_layer.weight.requires_grad,
            )
        else:
            compensated_layer.bias.sub_(adjustment)

        input_tensor = torch.as_tensor(
            x, dtype=first_layer.weight.dtype, device=first_layer.weight.device
        ).unsqueeze(0)
        shifted_tensor = torch.as_tensor(
            x + shift_vector,
            dtype=compensated_layer.weight.dtype,
            device=compensated_layer.weight.device,
        ).unsqueeze(0)
        original_output = reference_model(input_tensor)
        compensated_output = compensated_model(shifted_tensor)
        if not isinstance(original_output, torch.Tensor) or not isinstance(
            compensated_output, torch.Tensor
        ):
            raise TypeError("model must return a Tensor for equivalence verification")
        if (
            not torch.isfinite(original_output).all()
            or not torch.isfinite(compensated_output).all()
        ):
            raise ValueError("model outputs must be finite")
        if original_output.shape != compensated_output.shape or not torch.allclose(
            original_output, compensated_output, rtol=1e-5, atol=1e-6
        ):
            raise RuntimeError("compensated model output does not match original model output")

    original = _torch_attributions(
        explain_func, reference_model, x, x.size, "original attributions"
    )
    if verify_determinism:
        original_again = _torch_attributions(
            explain_func,
            reference_model,
            x,
            x.size,
            "repeated original attributions",
        )
        _assert_deterministic(original, original_again, "explain_func")

    shifted_x = x + shift_vector
    if not np.all(np.isfinite(shifted_x)):
        raise FloatingPointError("shifted instance is not representable")
    shifted = _torch_attributions(
        explain_func,
        compensated_model,
        shifted_x,
        x.size,
        "compensated attributions",
    )
    if verify_determinism:
        shifted_again = _torch_attributions(
            explain_func,
            compensated_model,
            shifted_x,
            x.size,
            "repeated compensated attributions",
        )
        _assert_deterministic(shifted, shifted_again, "explain_func")
    return _rms_difference(original, shifted, "compensated input-invariance")


def compute_batch_input_invariance(
    explain_func: Callable,
    X: np.ndarray,
    shift: Optional[Union[np.ndarray, float]] = None,
    seed: Optional[int] = None,
    max_instances: Optional[int] = None,
    verify_determinism: bool = True,
) -> Dict[str, object]:
    """Batch uncompensated translation-sensitivity diagnostic."""
    verify_determinism = _validate_bool(verify_determinism, "verify_determinism")
    warnings.warn(
        "compute_batch_input_invariance is an uncompensated translation-"
        "sensitivity diagnostic, not a test of the Input Invariance axiom",
        RuntimeWarning,
        stacklevel=2,
    )
    batch, n = _validate_batch_inputs(X, max_instances)
    # One batch represents one coordinate transformation, so every sample
    # receives the same constant shift vector.
    shared_shift = _build_shift(batch[0], shift, seed)
    scores = [
        _translation_sensitivity(
            explain_func,
            batch[index],
            shared_shift,
            None,
            verify_determinism,
        )
        for index in range(n)
    ]
    return _summarize(scores)


def compute_batch_input_invariance_pytorch(
    model,
    explain_func: Callable,
    X: np.ndarray,
    shift: Optional[Union[np.ndarray, float]] = None,
    seed: Optional[int] = None,
    max_instances: Optional[int] = None,
    verify_determinism: bool = True,
) -> Dict[str, object]:
    """Batch compensated input-shift tests, failing on any bad row."""
    verify_determinism = _validate_bool(verify_determinism, "verify_determinism")
    batch, n = _validate_batch_inputs(X, max_instances)
    shared_shift = _build_shift(batch[0], shift, seed)
    scores = [
        compute_input_invariance_pytorch(
            model,
            explain_func,
            batch[index],
            shared_shift,
            None,
            verify_determinism,
        )
        for index in range(n)
    ]
    return _summarize(scores)


def _validate_symmetric_pairs(
    symmetric_pairs: List[Tuple[int, int]], n_features: int
) -> List[Tuple[int, int]]:
    if symmetric_pairs is None:
        raise TypeError("symmetric_pairs must be a list of index pairs")
    pairs: List[Tuple[int, int]] = []
    seen = set()
    for pair in symmetric_pairs:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise TypeError("each symmetric pair must contain exactly two indices")
        first, second = pair
        if (
            isinstance(first, bool)
            or isinstance(second, bool)
            or not isinstance(first, (int, np.integer))
            or not isinstance(second, (int, np.integer))
        ):
            raise TypeError("symmetric pair indices must be integers")
        first, second = int(first), int(second)
        if first < 0 or first >= n_features:
            raise ValueError(f"Feature index {first} out of bounds for {n_features} features")
        if second < 0 or second >= n_features:
            raise ValueError(f"Feature index {second} out of bounds for {n_features} features")
        if first == second:
            raise ValueError("a symmetric pair must contain two distinct features")
        canonical = tuple(sorted((first, second)))
        if canonical in seen:
            raise ValueError(f"duplicate symmetric pair {canonical}")
        seen.add(canonical)
        pairs.append((first, second))
    return pairs


def compute_symmetry(
    attributions: np.ndarray,
    symmetric_pairs: List[Tuple[int, int]],
    instance: Optional[np.ndarray] = None,
    baseline: Optional[np.ndarray] = None,
    precondition_tolerance: float = 0.0,
) -> float:
    """Return mean attribution disparity for caller-certified symmetric pairs.

    Sundararajan et al.'s symmetry-preserving condition applies only when each
    pair is symmetric in the function *and* the two variables have equal values
    in both the input and baseline.  This function cannot prove functional
    symmetry.  Supplying ``instance`` and ``baseline`` makes it validate the
    equal-value preconditions; omitting both treats the pairs as fully certified
    by the caller and returns only the conditional disparity diagnostic.
    """
    attrs = _as_finite_vector(attributions, "attributions")
    pairs = _validate_symmetric_pairs(symmetric_pairs, attrs.size)
    if not pairs:
        raise ValueError("symmetry is undefined without at least one symmetric pair")
    if (instance is None) != (baseline is None):
        raise ValueError("instance and baseline must be supplied together")
    if (
        isinstance(precondition_tolerance, (bool, np.bool_))
        or not isinstance(precondition_tolerance, Real)
        or precondition_tolerance < 0
    ):
        raise ValueError("precondition_tolerance must be non-negative")
    if not np.isfinite(float(precondition_tolerance)):
        raise ValueError("precondition_tolerance must be finite")
    if instance is not None:
        x = _as_finite_vector(instance, "instance", expected_length=attrs.size)
        reference = _as_finite_vector(baseline, "baseline", expected_length=attrs.size)
        for first, second in pairs:
            if not np.isclose(x[first], x[second], rtol=0.0, atol=precondition_tolerance):
                raise ValueError(
                    f"instance values for symmetric pair {(first, second)} are " "not equal"
                )
            if not np.isclose(
                reference[first],
                reference[second],
                rtol=0.0,
                atol=precondition_tolerance,
            ):
                raise ValueError(
                    f"baseline values for symmetric pair {(first, second)} are " "not equal"
                )
    return _mean_absolute_pair_difference(attrs, pairs)


def compute_symmetry_score(
    explainer: BaseExplainer,
    instance: np.ndarray,
    symmetric_pairs: List[Tuple[int, int]],
    baseline: Optional[np.ndarray] = None,
    precondition_tolerance: float = 0.0,
    explain_kwargs: Optional[Dict[str, object]] = None,
    verify_determinism: bool = True,
) -> float:
    """Generate attributions and compute conditional pair disparity.

    If ``baseline`` is supplied, equal input/baseline values are checked.  If
    omitted, this is only a caller-certified pair-disparity diagnostic.
    """
    x = _as_finite_vector(instance, "instance")
    attrs = _get_explanation_vector(explainer, x, x.size, explain_kwargs, verify_determinism)
    if baseline is None:
        return compute_symmetry(attrs, symmetric_pairs)
    return compute_symmetry(
        attrs,
        symmetric_pairs,
        x,
        baseline,
        precondition_tolerance,
    )


def compute_batch_symmetry(
    symmetric_pairs: List[Tuple[int, int]],
    attributions_list: Optional[List[np.ndarray]] = None,
    explainer: Optional[BaseExplainer] = None,
    X: Optional[np.ndarray] = None,
    max_instances: Optional[int] = None,
    baseline: Optional[np.ndarray] = None,
    precondition_tolerance: float = 0.0,
    explain_kwargs: Optional[Dict[str, object]] = None,
    verify_determinism: bool = True,
) -> Dict[str, object]:
    """Compute conditional attribution-pair disparities for a batch."""
    verify_determinism = _validate_bool(verify_determinism, "verify_determinism")
    if attributions_list is None and (explainer is None or X is None):
        raise ValueError("Either attributions_list or (explainer + X) must be provided")
    if attributions_list is not None:
        if len(attributions_list) == 0:
            raise ValueError("attributions_list must not be empty")
        if max_instances is None:
            n = len(attributions_list)
        else:
            if isinstance(max_instances, bool) or not isinstance(max_instances, (int, np.integer)):
                raise TypeError("max_instances must be a positive integer or None")
            if max_instances <= 0:
                raise ValueError("max_instances must be positive")
            n = min(len(attributions_list), int(max_instances))
        if baseline is None:
            batch = None
        else:
            if X is None:
                raise ValueError(
                    "X is required to validate symmetry preconditions when "
                    "baseline is supplied with pre-computed attributions"
                )
            batch, n_from_x = _validate_batch_inputs(X, max_instances)
            if n_from_x < n:
                raise ValueError(
                    f"X has only {n_from_x} usable rows but {n} attribution "
                    "vectors are being evaluated"
                )
    else:
        batch, n = _validate_batch_inputs(X, max_instances)

    scores: List[float] = []
    for index in range(n):
        if attributions_list is not None:
            if baseline is None:
                score = compute_symmetry(attributions_list[index], symmetric_pairs)
            else:
                assert batch is not None
                score = compute_symmetry(
                    attributions_list[index],
                    symmetric_pairs,
                    batch[index],
                    baseline,
                    precondition_tolerance,
                )
        else:
            assert explainer is not None
            assert batch is not None
            score = compute_symmetry_score(
                explainer,
                batch[index],
                symmetric_pairs,
                baseline,
                precondition_tolerance,
                explain_kwargs,
                verify_determinism,
            )
        scores.append(score)
    return _summarize(scores)
