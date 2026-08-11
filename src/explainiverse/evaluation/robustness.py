# src/explainiverse/evaluation/robustness.py
"""
Finite-sample robustness and stability diagnostics for explanations.

Implements:
- Max-Sensitivity (Yeh et al., 2019)
- Mean-sensitivity compatibility heuristic (not defined by Yeh et al.)
- Finite-sample local Lipschitz estimate (Alvarez-Melis & Jaakkola, 2018)
- Consistency (Dasgupta et al., 2022)
- Relative Input Stability — RIS (Agarwal et al., 2022, Eq 2)
- Relative Representation Stability — RRS (Agarwal et al., 2022, Eq 3)
- Relative Output Stability — ROS (Agarwal et al., 2022, Eq 5)

These functions report declared finite-sample instability or agreement
statistics. They do not certify stability outside the sampled inputs and
perturbations.

References:
    Yeh, C. K., Hsieh, C. Y., Suggala, A. S., Inouye, D. I., & Ravikumar, P.
    (2019). On the (In)fidelity and Sensitivity of Explanations. NeurIPS.
    https://proceedings.neurips.cc/paper/2019/hash/a7471fdc77b3435276507cc8f2571547-Abstract.html

    Alvarez-Melis, D., & Jaakkola, T. S. (2018). On the Robustness of
    Interpretability Methods. ICML Workshop on Human Interpretability in
    Machine Learning (WHI).
    https://arxiv.org/abs/1806.08049

    Dasgupta, S., Frost, N., & Moshkovitz, M. (2022). Framework for
    Evaluating Faithfulness of Local Explanations. ICML.
    https://proceedings.mlr.press/v162/dasgupta22a.html

    Agarwal, C., Johnson, N., Pawelczyk, M., Krishna, S., Saxena, E.,
    Zitnik, M., & Lakkaraju, H. (2022). Rethinking Stability for
    Attribution-based Explanations. arXiv:2203.06877.
"""

import inspect
import warnings
from collections.abc import Mapping
from decimal import Decimal, localcontext
from numbers import Integral, Real
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial.distance import cdist

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation
from explainiverse.evaluation._utils import _stable_mean, _stable_std, _stable_sum
from explainiverse.evaluation.faithfulness import _candidate_from_label

# =============================================================================
# Internal Helpers
# =============================================================================

_UNSET_TARGET = object()


def _extract_attribution_vector(explanation: Explanation) -> np.ndarray:
    """
    Extract attribution values as a numpy array from an Explanation.

    Preserves the declared feature order. If feature names are absent, the
    mapping must use the complete explicit sequence ``feature_0``,
    ``feature_1``, ...; dictionary insertion order is not treated as a feature
    identity contract.

    Args:
        explanation: Explanation object with feature_attributions

    Returns:
        1D numpy array of attribution values

    Raises:
        ValueError: If no feature attributions are found
    """
    if not isinstance(explanation, Explanation):
        raise TypeError("explanation must be an Explanation object.")
    attributions = explanation.explanation_data.get("feature_attributions", {})
    if not isinstance(attributions, Mapping):
        raise TypeError("feature_attributions must be a mapping from feature names to values.")
    if not attributions:
        raise ValueError("No feature attributions found in explanation.")
    if any(not isinstance(name, str) or not name for name in attributions):
        raise TypeError("Feature attribution keys must be non-empty strings.")

    feature_names = getattr(explanation, "feature_names", None)
    if feature_names is not None and len(feature_names) > 0:
        feature_names = list(feature_names)
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("Explanation feature_names must be unique.")
        missing = set(feature_names) - set(attributions)
        extra = set(attributions) - set(feature_names)
        if missing or extra:
            raise ValueError(
                "Explanation feature_attributions must map exactly one value per "
                f"declared feature; missing={sorted(missing)!r}, extra={sorted(extra)!r}."
            )
        values = [attributions[name] for name in feature_names]
    else:
        feature_names = [f"feature_{index}" for index in range(len(attributions))]
        if set(attributions) != set(feature_names):
            raise ValueError(
                "Explanation without feature_names must use a complete zero-based "
                "feature_<index> attribution mapping."
            )
        values = [attributions[name] for name in feature_names]

    if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
        raise TypeError("Feature attributions must be real numeric scalars.")
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError("Feature attributions must be a non-empty 1D vector.")
    if not np.all(np.isfinite(result)):
        raise ValueError("Feature attributions must contain only finite values.")
    return result


def _validate_instance(instance: np.ndarray, name: str = "instance") -> np.ndarray:
    """Return a strict, finite one-dimensional feature vector."""
    result = np.asarray(instance, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _validate_batch(X: np.ndarray, name: str = "X") -> np.ndarray:
    """Return a strict, finite two-dimensional feature matrix."""
    result = np.asarray(X, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D array.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _validate_sampling_parameters(
    *,
    n_samples: int,
    radius: Optional[float] = None,
    noise_scale: Optional[float] = None,
) -> None:
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
        raise TypeError("The number of perturbations must be an integer.")
    if int(n_samples) <= 0:
        raise ValueError("The number of perturbations must be greater than zero.")
    if radius is not None:
        if isinstance(radius, (bool, np.bool_)) or not isinstance(
            radius, (int, float, np.integer, np.floating)
        ):
            raise TypeError("radius must be a real number.")
        if not np.isfinite(radius) or radius < 0:
            raise ValueError("radius must be finite and non-negative.")
    if noise_scale is not None:
        if isinstance(noise_scale, (bool, np.bool_)) or not isinstance(
            noise_scale, (int, float, np.integer, np.floating)
        ):
            raise TypeError("noise_scale must be a real number.")
        if not np.isfinite(noise_scale) or noise_scale < 0:
            raise ValueError("noise_scale must be finite and non-negative.")


def _validate_norm_order(norm_ord: Union[int, float]) -> None:
    if isinstance(norm_ord, (bool, np.bool_)) or norm_ord not in {1, 2, np.inf}:
        raise ValueError("norm_ord must be 1, 2, or np.inf.")


def _validate_bool(value, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean.")
    return bool(value)


def _vector_norm(values: np.ndarray, norm_ord: Union[int, float]) -> float:
    """Return a finite vector norm without overflow or tiny-value underflow."""
    vector = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise FloatingPointError("vector norm input must be finite")
    magnitudes = np.abs(vector)
    if norm_ord == np.inf:
        return float(np.max(magnitudes))
    if norm_ord == 1:
        result = float(_stable_sum(magnitudes))
    else:
        scale = float(np.max(magnitudes))
        if scale == 0.0:
            return 0.0
        scaled = vector / scale
        with np.errstate(over="ignore", invalid="ignore"):
            result = float(scale * np.sqrt(np.dot(scaled, scaled)))
    if not np.isfinite(result):
        raise FloatingPointError("vector norm is not representable")
    return result


def _scaled_norm_components(values: np.ndarray, norm_ord: Union[int, float]) -> Tuple[float, float]:
    """Represent a norm as ``scale * unit_norm`` without materializing it."""
    vector = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise FloatingPointError("vector norm input must be finite")
    scale = float(np.max(np.abs(vector)))
    if scale == 0.0:
        return 0.0, 0.0
    normalized = vector / scale
    if norm_ord == np.inf:
        unit_norm = 1.0
    elif norm_ord == 1:
        unit_norm = float(_stable_sum(np.abs(normalized)))
    else:
        unit_norm = float(np.sqrt(_stable_sum(normalized * normalized)))
    return scale, unit_norm


def _vector_norm_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    norm_ord: Union[int, float],
    context: str,
) -> float:
    """Divide vector norms even when each separate norm is outside float64."""
    numerator_scale, numerator_unit = _scaled_norm_components(numerator, norm_ord)
    denominator_scale, denominator_unit = _scaled_norm_components(denominator, norm_ord)
    if denominator_scale == 0.0:
        raise ValueError(f"{context} denominator norm is zero")
    if numerator_scale == 0.0:
        return 0.0
    with localcontext() as decimal_context:
        decimal_context.prec = 1600
        exact = (
            Decimal.from_float(numerator_scale)
            * Decimal.from_float(numerator_unit)
            / Decimal.from_float(denominator_scale)
            / Decimal.from_float(denominator_unit)
        )
        result = float(exact)
    if not np.isfinite(result) or (result == 0.0 and exact != 0):
        raise FloatingPointError(f"{context} ratio is not representable")
    return result


def _vector_difference(left: np.ndarray, right: np.ndarray, context: str) -> np.ndarray:
    """Subtract finite vectors, rejecting an out-of-range coordinate."""
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError(f"{context} vector difference is not representable")
    return result


def _finite_mean(values: Union[List[float], np.ndarray], context: str) -> float:
    result = float(_stable_mean(np.asarray(values, dtype=np.float64)))
    if not np.isfinite(result):
        raise FloatingPointError(f"{context} mean is not representable")
    return result


def _finite_std(values: Union[List[float], np.ndarray], context: str) -> float:
    result = float(_stable_std(np.asarray(values, dtype=np.float64)))
    if not np.isfinite(result):
        raise FloatingPointError(f"{context} standard deviation is not representable")
    return result


def _finite_median(values: Union[List[float], np.ndarray], context: str) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    midpoint = ordered.size // 2
    if ordered.size % 2:
        return float(ordered[midpoint])
    return _finite_mean(ordered[midpoint - 1 : midpoint + 1], context)


def _finite_ratio(numerator: float, denominator: float, context: str) -> float:
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        result = float(numerator / denominator)
    if not np.isfinite(result) or (result == 0.0 and numerator != 0.0):
        raise FloatingPointError(f"{context} ratio is not representable")
    return result


def _require_scalar_result(result: Union[float, dict], metric_name: str) -> float:
    """Narrow a detail-capable metric call made with details disabled."""
    if isinstance(result, dict):
        raise RuntimeError(f"{metric_name} unexpectedly returned detail data")
    return float(result)


def _supports_target_class(explainer: BaseExplainer) -> bool:
    """Whether ``explain`` accepts a fixed output-column target."""
    try:
        parameters = inspect.signature(explainer.explain).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == "target_class" or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _get_explanation(
    explainer: BaseExplainer,
    instance: np.ndarray,
    target_class: Optional[int] = None,
) -> Explanation:
    kwargs: Dict[str, int] = {}
    if target_class is not None:
        if not isinstance(target_class, Integral) or isinstance(target_class, (bool, np.bool_)):
            raise TypeError("target_class must be an integer output index or None.")
        target_class = int(target_class)
        if target_class < 0:
            raise ValueError("target_class must be non-negative.")
        if not _supports_target_class(explainer):
            raise ValueError(
                "This explainer cannot accept target_class, so the metric cannot "
                "guarantee a fixed explained output."
            )
        kwargs["target_class"] = target_class
    explanation = explainer.explain(instance, **kwargs)
    if not isinstance(explanation, Explanation):
        raise TypeError("explainer.explain() must return an Explanation object.")
    if target_class is not None:
        identity_source = getattr(explainer, "model", explainer)
        returned_target = _candidate_from_label(identity_source, explanation)
        if returned_target is None:
            raise ValueError(
                "The explainer returned no target identity that can verify the "
                "requested target_class."
            )
        if returned_target != target_class:
            raise ValueError(
                "The explainer did not honor the requested target_class "
                f"({target_class} requested, {returned_target} returned)."
            )
    return explanation


def _generate_perturbations_l2(
    instance: np.ndarray,
    radius: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate uniform random perturbations within an L2 ball.

    Samples directions uniformly on the unit sphere, then scales by
    a radius drawn uniformly from [0, r] (with volume correction via
    r^{1/d} for uniform density in the ball).

    Args:
        instance: Center point (1D array)
        radius: Radius of the L2 ball
        n_samples: Number of perturbations to generate
        rng: NumPy random generator

    Returns:
        Array of shape (n_samples, n_features) — perturbed instances
    """
    d = len(instance)
    # Sample direction uniformly on unit sphere
    directions = rng.standard_normal((n_samples, d))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # Avoid division by zero
    directions = directions / norms

    # Sample radius with volume-correction: r ~ U[0, R]^{1/d}
    radii = rng.uniform(0, 1, size=(n_samples, 1)) ** (1.0 / d) * radius
    with np.errstate(over="ignore", invalid="ignore"):
        perturbations = instance + directions * radii
    if not np.all(np.isfinite(perturbations)):
        raise FloatingPointError("L2 perturbation is not representable")
    return perturbations


def _generate_perturbations_linf(
    instance: np.ndarray,
    radius: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate uniform random perturbations within an L-infinity ball.

    Each feature is independently perturbed by a uniform value in [-r, r].

    Args:
        instance: Center point (1D array)
        radius: Radius of the L∞ ball
        n_samples: Number of perturbations to generate
        rng: NumPy random generator

    Returns:
        Array of shape (n_samples, n_features) — perturbed instances
    """
    d = len(instance)
    if radius <= np.finfo(np.float64).max / 2.0:
        noise = rng.uniform(-radius, radius, size=(n_samples, d))
    else:
        noise = (2.0 * rng.random((n_samples, d)) - 1.0) * radius
    with np.errstate(over="ignore", invalid="ignore"):
        perturbations = instance + noise
    if not np.all(np.isfinite(perturbations)):
        raise FloatingPointError("L-infinity perturbation is not representable")
    return perturbations


def _get_explanation_vector(
    explainer: BaseExplainer,
    instance: np.ndarray,
    n_features: int,
    target_class: Optional[int] = None,
    expected_target=_UNSET_TARGET,
) -> np.ndarray:
    """
    Get attribution vector for a single instance.

    Args:
        explainer: Explainer instance
        instance: Input (1D array)
        n_features: Expected number of features

    Returns:
        1D numpy array of attributions
    """
    exp = _get_explanation(explainer, instance, target_class=target_class)
    values = _extract_attribution_vector(exp)
    if values.size != n_features:
        raise ValueError(
            f"Explanation has {values.size} attribution values; expected "
            f"{n_features}, one per input feature."
        )
    if expected_target is not _UNSET_TARGET and exp.target_class != expected_target:
        raise ValueError(
            "The explainer changed target_class across perturbations "
            f"({expected_target!r} -> {exp.target_class!r}). Pass an explicit "
            "target_class to an explainer that supports fixed targets."
        )
    return values


def _get_explanation_vector_and_target(
    explainer: BaseExplainer,
    instance: np.ndarray,
    n_features: int,
    target_class: Optional[int] = None,
):
    exp = _get_explanation(explainer, instance, target_class=target_class)
    values = _extract_attribution_vector(exp)
    if values.size != n_features:
        raise ValueError(
            f"Explanation has {values.size} attribution values; expected "
            f"{n_features}, one per input feature."
        )
    return values, exp.target_class


# =============================================================================
# Max-Sensitivity (Yeh et al., 2019)
# =============================================================================


def compute_max_sensitivity(
    explainer: BaseExplainer,
    instance: np.ndarray,
    radius: float = 0.1,
    n_samples: int = 50,
    norm_ord: Union[int, float] = 2,
    perturb_norm: str = "l2",
    normalize: bool = False,
    seed: Optional[int] = None,
    target_class: Optional[int] = None,
) -> float:
    """
    Compute Max-Sensitivity of an explanation method.

    Max-Sensitivity measures the worst-case change in explanation when the
    input is perturbed within a small ball of radius r:

        MaxSens(E, x, r) = max_{||δ||_p ≤ r} ||E(x + δ) - E(x)||_q

    A lower score means a smaller sampled explanation change under this
    perturbation and norm contract; it is not a global robustness certificate.

    Args:
        explainer: Explainer instance with .explain() method.
        instance: Input instance (1D array of shape (n_features,)).
        radius: Radius of the perturbation ball. Default: 0.1.
            For unnormalized features, scale this to the feature range.
        n_samples: Number of Monte Carlo samples to approximate the max.
            Additional samples reduce under-sampling risk but do not certify
            the true neighbourhood maximum. Default: 50.
        norm_ord: Norm order for measuring explanation change.
            2 for L2 (default), 1 for L1, np.inf for L∞.
        perturb_norm: Norm for the perturbation ball.
            "l2" (default) or "linf".
        normalize: If True, divide by the norm of the original explanation.
            This is a noncanonical relative variant; Yeh et al.'s definition
            is the unnormalised change, so the default is False.
        seed: Seed for this function's local perturbation generator only.
            Stochastic explainers/models must be controlled separately.
        target_class: Optional fixed output-column index. If supplied, the
            explainer must accept ``target_class``.

    Returns:
        Finite-sample Monte Carlo maximum (float). It is a sampled lower
        estimate of the neighbourhood supremum; lower = more robust under the
        declared sampling contract.
        If ``normalize=True``, a zero-norm original explanation is undefined
        and raises ``ValueError``.

    Example:
        >>> from explainiverse.evaluation import compute_max_sensitivity
        >>> score = compute_max_sensitivity(explainer, instance, radius=0.1)
        >>> print(f"Max-Sensitivity: {score:.4f}")

    Reference:
        Yeh et al. (2019). On the (In)fidelity and Sensitivity of
        Explanations. NeurIPS.
    """
    instance = _validate_instance(instance)
    _validate_sampling_parameters(n_samples=n_samples, radius=radius)
    _validate_norm_order(norm_ord)
    if not isinstance(normalize, (bool, np.bool_)):
        raise TypeError("normalize must be a boolean.")
    n_features = len(instance)
    rng = np.random.default_rng(seed)

    # Get original explanation
    original_attr, explained_target = _get_explanation_vector_and_target(
        explainer, instance, n_features, target_class=target_class
    )
    original_norm = _vector_norm(original_attr, norm_ord)

    if normalize and original_norm == 0:
        raise ValueError(
            "Normalized sensitivity is undefined for a zero-norm original "
            "explanation. Use normalize=False for canonical Max-Sensitivity."
        )

    # Generate perturbations
    if perturb_norm == "l2":
        perturbed = _generate_perturbations_l2(instance, radius, n_samples, rng)
    elif perturb_norm == "linf":
        perturbed = _generate_perturbations_linf(instance, radius, n_samples, rng)
    else:
        raise ValueError(f"perturb_norm must be 'l2' or 'linf', got '{perturb_norm}'")

    # Compute explanation distances
    diffs = []
    for i in range(n_samples):
        perturbed_attr = _get_explanation_vector(
            explainer,
            perturbed[i],
            n_features,
            target_class=target_class,
            expected_target=explained_target,
        )
        diffs.append(
            _vector_norm(
                _vector_difference(original_attr, perturbed_attr, "explanation"),
                norm_ord,
            )
        )

    max_diff = max(diffs)

    if normalize:
        return _finite_ratio(max_diff, original_norm, "normalized max-sensitivity")
    return float(max_diff)


def compute_batch_max_sensitivity(
    explainer: BaseExplainer,
    X: np.ndarray,
    radius: float = 0.1,
    n_samples: int = 50,
    norm_ord: Union[int, float] = 2,
    perturb_norm: str = "l2",
    normalize: bool = False,
    max_instances: Optional[int] = None,
    seed: Optional[int] = None,
    target_class: Optional[int] = None,
) -> dict:
    """
    Compute Max-Sensitivity over a batch of instances.

    Args:
        explainer: Explainer instance.
        X: Input data (2D array of shape (n_instances, n_features)).
        radius: Perturbation radius.
        n_samples: Perturbation samples per instance.
        norm_ord: Norm order for explanation differences.
        perturb_norm: Perturbation ball norm ("l2" or "linf").
        normalize: If True, normalize by original explanation norm.
        max_instances: Maximum number of instances to evaluate (None = all).
        seed: Base seed for per-instance local perturbation generators. It
            does not seed a stochastic explainer or model.

    Returns:
        Dictionary with:
            - "mean": Mean Max-Sensitivity across instances
            - "std": Standard deviation
            - "max": Worst-case Max-Sensitivity
            - "min": Minimum observed per-instance score
            - "scores": List of per-instance scores
            - "n_evaluated": Number of instances evaluated
    """
    X = _validate_batch(X)
    n = len(X)
    if max_instances is not None:
        if isinstance(max_instances, bool) or not isinstance(max_instances, (int, np.integer)):
            raise TypeError("max_instances must be an integer or None.")
        if max_instances <= 0:
            raise ValueError("max_instances must be greater than zero.")
        n = min(n, max_instances)

    scores = []
    for i in range(n):
        score = compute_max_sensitivity(
            explainer,
            X[i],
            radius=radius,
            n_samples=n_samples,
            norm_ord=norm_ord,
            perturb_norm=perturb_norm,
            normalize=normalize,
            seed=seed + i if seed is not None else None,
            target_class=target_class,
        )
        scores.append(score)

    if not scores:
        return {
            "mean": float("nan"),
            "std": 0.0,
            "max": float("nan"),
            "min": float("nan"),
            "scores": [],
            "n_evaluated": 0,
            "n_attempted": n,
            "n_undefined": n,
        }

    return {
        "mean": _finite_mean(scores, "batch max-sensitivity"),
        "std": _finite_std(scores, "batch max-sensitivity"),
        "max": float(np.max(scores)),
        "min": float(np.min(scores)),
        "scores": scores,
        "n_evaluated": len(scores),
        "n_attempted": n,
        "n_undefined": n - len(scores),
    }


# =============================================================================
# Relative Stability Helpers (Agarwal et al., 2022)
# =============================================================================


def _generate_mixed_perturbations(
    instance: np.ndarray,
    n_perturbations: int,
    noise_scale: float,
    rng: np.random.Generator,
    feature_types: Optional[np.ndarray] = None,
    discrete_flip_prob: float = 0.03,
) -> np.ndarray:
    """
    Generate perturbations with support for mixed feature types.

    Continuous features: additive Gaussian noise N(0, noise_scale).
    Discrete (binary) features: independent Bernoulli(p) replacement draws.

    Following Appendix B of Agarwal et al. (2022):
    - Continuous perturbations: x' = x + N(0, 0.05)
    - Discrete perturbations: replace values with Bernoulli(p) draws, p=0.03

    The paper's Appendix B text literally specifies replacement. The authors'
    later OpenXAI code instead implements a Bernoulli-triggered binary flip,
    so discrete scores from the two perturbation contracts are not directly
    comparable. This function follows the written paper contract and exposes
    it in metric diagnostics.

    Args:
        instance: 1D input array.
        n_perturbations: Number of perturbed copies to generate.
        noise_scale: Standard deviation of Gaussian noise for continuous
            features. The public metric default of 0.05 matches the paper's
            experimental setting.
        rng: NumPy random generator.
        feature_types: 1D array of strings, one per feature. Values:
            "continuous" (default) or "discrete". If None, all continuous.
        discrete_flip_prob: Bernoulli success probability for replacement
            draws. The historical parameter name is retained for API
            compatibility; this is not a per-value flip probability.

    Returns:
        Array of shape (n_perturbations, n_features).
    """
    instance = _validate_instance(instance)
    _validate_sampling_parameters(n_samples=n_perturbations, noise_scale=noise_scale)
    if isinstance(discrete_flip_prob, (bool, np.bool_)) or not isinstance(
        discrete_flip_prob, (int, float, np.integer, np.floating)
    ):
        raise TypeError("discrete_flip_prob must be a real number.")
    if not np.isfinite(discrete_flip_prob) or not 0 <= discrete_flip_prob <= 1:
        raise ValueError("discrete_flip_prob must be in [0, 1].")

    d = len(instance)
    perturbed = np.tile(instance, (n_perturbations, 1))

    if feature_types is None:
        # All continuous — simple Gaussian perturbation
        noise = rng.normal(0, noise_scale, size=(n_perturbations, d))
        perturbed = perturbed + noise
    else:
        feature_types = np.asarray(feature_types)
        if feature_types.ndim != 1 or feature_types.size != d:
            raise ValueError("feature_types must be a 1D array with one entry per feature.")
        unknown = set(feature_types.tolist()) - {"continuous", "discrete"}
        if unknown:
            raise ValueError(
                "feature_types entries must be 'continuous' or 'discrete'; "
                f"got {sorted(unknown)!r}."
            )
        continuous_mask = feature_types == "continuous"
        discrete_mask = feature_types == "discrete"
        if np.any(discrete_mask) and not np.all(np.isin(instance[discrete_mask], [0.0, 1.0])):
            raise ValueError(
                "Features marked 'discrete' must be binary 0/1 values for "
                "the declared Bernoulli replacement contract."
            )

        # Continuous features: additive Gaussian noise
        if np.any(continuous_mask):
            n_cont: int = int(np.sum(continuous_mask))
            noise = rng.normal(0, noise_scale, size=(n_perturbations, n_cont))
            perturbed[:, continuous_mask] += noise

        # Discrete features: Bernoulli replacement draws.
        if np.any(discrete_mask):
            n_disc: int = int(np.sum(discrete_mask))
            # Appendix B replaces each discrete dimension by an independent
            # Bernoulli(p) draw; it does not describe an XOR/flip operation.
            perturbed[:, discrete_mask] = rng.binomial(
                1,
                discrete_flip_prob,
                size=(n_perturbations, n_disc),
            )

    return perturbed


def _element_wise_percent_change(
    original: np.ndarray,
    perturbed: np.ndarray,
    epsilon_min: float = 1e-7,
) -> np.ndarray:
    """
    Compute element-wise percent change: (original - perturbed) / original.

    Replaces exact zero denominator entries with ``epsilon_min``, matching the
    official Quantus objective implementation.

    Args:
        original: 1D array (the reference vector).
        perturbed: 1D array (the perturbed vector).
        epsilon_min: Positive replacement for exact zero denominator entries.

    Returns:
        1D array of element-wise percent changes.
    """
    original = np.asarray(original, dtype=np.float64)
    perturbed = np.asarray(perturbed, dtype=np.float64)
    if original.shape != perturbed.shape:
        raise ValueError("original and perturbed vectors must have the same shape.")
    if isinstance(epsilon_min, (bool, np.bool_)) or not isinstance(
        epsilon_min, (int, float, np.integer, np.floating)
    ):
        raise TypeError("epsilon_min must be a real number.")
    if not np.isfinite(epsilon_min) or epsilon_min <= 0:
        raise ValueError("epsilon_min must be finite and greater than zero.")
    safe_denom = np.copy(original)
    safe_denom[original == 0] = epsilon_min
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        result = (original - perturbed) / safe_denom
    fallback = (~np.isfinite(result)) | ((result == 0.0) & (original != perturbed))
    for index in np.flatnonzero(fallback):
        with localcontext() as context:
            context.prec = 1500
            exact = (
                Decimal.from_float(float(original[index]))
                - Decimal.from_float(float(perturbed[index]))
            ) / Decimal.from_float(float(safe_denom[index]))
            rounded = float(exact)
        if not np.isfinite(rounded) or (rounded == 0.0 and exact != 0):
            raise FloatingPointError("element-wise percent change is not representable")
        result[index] = rounded
    return result


def _aggregate_perturbation_scores(
    scores: List[float],
    aggregation: str = "max",
) -> dict:
    """
    Aggregate per-perturbation scores and build diagnostic dict.

    Args:
        scores: List of per-perturbation ratio scores.
        aggregation: "max" (canonical), "mean", or "median". The latter two
            are explicit noncanonical summaries.

    Returns:
        Dict with score, max, mean, median, perturbation_scores.

    Raises:
        ValueError: If aggregation is not one of the valid options.
    """
    valid_aggs = {"max", "mean", "median"}
    if aggregation not in valid_aggs:
        raise ValueError(f"aggregation must be one of {valid_aggs}, got '{aggregation}'")

    if not scores:
        return {
            "score": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
            "perturbation_scores": [],
        }

    arr = np.array(scores, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise FloatingPointError("perturbation score is not representable")
    result = {
        "max": float(np.max(arr)),
        "mean": _finite_mean(arr, "perturbation score"),
        "median": _finite_median(arr, "perturbation score median"),
        "perturbation_scores": scores,
    }
    result["score"] = result[aggregation]
    return result


def _validate_relative_parameters(
    *,
    n_perturbations: int,
    noise_scale: float,
    norm_ord: Union[int, float],
    epsilon_min: float,
    aggregation: str,
) -> None:
    _validate_sampling_parameters(n_samples=n_perturbations, noise_scale=noise_scale)
    _validate_norm_order(norm_ord)
    if isinstance(epsilon_min, (bool, np.bool_)) or not isinstance(
        epsilon_min, (int, float, np.integer, np.floating)
    ):
        raise TypeError("epsilon_min must be a real number.")
    if not np.isfinite(epsilon_min) or epsilon_min <= 0:
        raise ValueError("epsilon_min must be finite and greater than zero.")
    if aggregation not in {"max", "mean", "median"}:
        raise ValueError(
            "aggregation must be one of {'max', 'mean', 'median'}, " f"got {aggregation!r}"
        )


def _evaluate_single_vector_fn(
    function: Callable[[np.ndarray], np.ndarray],
    instance: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Evaluate a representation/logit callable for exactly one instance."""
    result = np.asarray(function(instance), dtype=np.float64)
    if result.ndim >= 2:
        if result.shape[0] != 1:
            raise ValueError(f"{name} must return exactly one sample.")
        result = result[0]
    result = result.reshape(-1)
    if result.size == 0:
        raise ValueError(f"{name} must return a non-empty vector.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must return only finite values.")
    return result


def _class_from_single_prediction(output, *, probabilistic: bool):
    """Interpret one model result without coercing arbitrary labels to int."""
    values = np.asarray(output)
    if values.ndim == 0:
        return values.item()

    # Remove only the single-sample batch dimension. Extra sample rows are a
    # contract error rather than something to flatten and silently reinterpret.
    if values.ndim >= 2:
        if values.shape[0] != 1:
            raise ValueError("Model prediction for one instance must have exactly one row.")
        values = values[0]

    values = np.asarray(values)
    if values.ndim == 0 or values.size == 1:
        scalar = values.reshape(-1)[0].item()
        if probabilistic:
            probability = float(scalar)
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(
                    "A one-column predict_proba output must be a probability " "in [0, 1]."
                )
            if not np.isclose(probability, 1.0):
                raise ValueError(
                    "A one-column probabilistic output is ambiguous unless it "
                    "represents a one-class distribution with probability 1. "
                    "Return two class-score columns for binary classification."
                )
            return 0
        return scalar

    if values.ndim != 1:
        raise ValueError("Model output for one instance must be a 1D vector.")
    numeric = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("Model output must contain only finite values.")
    if probabilistic:
        if np.any((numeric < 0) | (numeric > 1)) or not np.isclose(
            np.sum(numeric), 1.0, rtol=1e-6, atol=1e-8
        ):
            raise ValueError(
                "Probabilistic model output must contain values in [0, 1] "
                "that sum to one for the single sample."
            )
    return int(np.argmax(numeric))


def _get_predicted_class(
    model,
    instance: np.ndarray,
):
    """
    Get predicted class label for a single instance.

    Handles both predict_proba() and predict() interfaces.

    Args:
        model: Model adapter.
        instance: 1D input array.

    Returns:
        A predicted class label or output-column index.
    """
    known_task = getattr(model, "task", None)
    estimator_type = getattr(model, "_estimator_type", None)
    wrapped_model = getattr(model, "model", None)
    wrapped_estimator_type = getattr(wrapped_model, "_estimator_type", None)
    if (
        known_task == "regression"
        or estimator_type == "regressor"
        or wrapped_estimator_type == "regressor"
    ):
        raise ValueError(
            "This metric requires categorical model predictions; a known "
            "regression model has no same-predicted-class contract."
        )

    instance = _validate_instance(instance)
    instance_2d = instance.reshape(1, -1)

    if callable(getattr(model, "predict_proba", None)):
        proba = model.predict_proba(instance_2d)
        return _class_from_single_prediction(proba, probabilistic=True)

    if callable(getattr(model, "predict", None)):
        pred = model.predict(instance_2d)
        # Explainiverse adapters return a row of classification scores from
        # predict(), while raw sklearn classifiers return one label. The shape
        # distinguishes these contracts; never take the first probability and
        # cast it to int.
        return _class_from_single_prediction(
            pred,
            probabilistic=getattr(model, "task", None) == "classification",
        )

    raise ValueError("Model must have a predict() or predict_proba() method.")


# =============================================================================
# Relative Input Stability (Agarwal et al., 2022 — Equation 2)
# =============================================================================


def compute_relative_input_stability(
    explainer: BaseExplainer,
    model,
    instance: np.ndarray,
    n_perturbations: int = 50,
    noise_scale: float = 0.05,
    norm_ord: Union[int, float] = 2,
    epsilon_min: float = 1e-7,
    aggregation: str = "max",
    feature_types: Optional[np.ndarray] = None,
    discrete_flip_prob: float = 0.03,
    seed: Optional[int] = None,
    return_details: bool = False,
    representation_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    target_class: Optional[int] = None,
) -> Union[float, dict]:
    """
    Compute Relative Input Stability (Agarwal et al., 2022, Equation 2).

    RIS measures the instability of an explanation by computing the maximum
    ratio of percent change in explanation to percent change in input across
    perturbations that preserve the predicted class:

        RIS(x) = max_{x': ŷ_x = ŷ_x'}
                 ||(e_x − e_x') / e_x||_p
                 / max(||(x − x') / x||_p, ε_min)

    A higher score is greater measured instability under this equation and
    perturbation contract.

    With finitely many generated perturbations, ``aggregation="max"`` is a
    sampled maximum and need not equal the full-neighbourhood maximum.

    The numerator uses element-wise percent change in explanation, enabling
    comparison across explanation methods with different magnitude ranges.
    The denominator normalises by the percent change in input.

    This follows the written equations and Quantus objective implementation.
    The authors' later OpenXAI benchmark code instead computes
    ``||e-e'||/||e||`` and ``||x-x'||/||x||``; those scores are not directly
    comparable to this element-wise-percent-change contract.

    Args:
        explainer: Explainer instance with .explain() method.
        model: Model adapter with predict() or predict_proba().
        instance: Input instance (1D array of shape (n_features,)).
        n_perturbations: Number of perturbations to generate. Default: 50,
            matching the paper's experimental setting (not a guarantee).
        noise_scale: Standard deviation of Gaussian noise for continuous
            features. Default: 0.05, matching the paper's experiments. The
            Gaussian is unbounded and applied in the supplied (typically
            preprocessed) feature space; this function does not clip to data
            bounds or enforce domain constraints.
        norm_ord: Norm order for computing vector norms. Default: 2 (L2).
            Supports 1, 2, np.inf.
        epsilon_min: Floor to prevent division by zero in element-wise
            percent change and in the denominator norm. Default: 1e-7.
        aggregation: How to aggregate per-perturbation scores.
            "max" (canonical worst-case instability), or the explicitly
            noncanonical "mean" and "median" summaries.
        feature_types: 1D array of "continuous" or binary "discrete" per feature.
            If None, all features are treated as continuous.
            Discrete features are replaced by Bernoulli draws.
        discrete_flip_prob: Bernoulli replacement success probability. The
            historical name is retained for compatibility. Default: 0.03.
        seed: Seed for the local perturbation generator only. It does not seed
            a stochastic explainer, model, representation_fn, or logit_fn.
        return_details: If True, return a dict with full diagnostics.
            If False (default), return a single float score.
        representation_fn: Optional callable mapping input → hidden
            representation. If provided and return_details=True, the
            sampled right-hand-side estimate for Equation 4 is included. It
            is not a certified theoretical bound.
        target_class: Optional fixed output-column index.

    Returns:
        If return_details=False: float score (using chosen aggregation).
            Returns NaN if no perturbations pass the same-class filter.
        If return_details=True: dict with keys:
            - "score": float (aggregated score)
            - "max": float
            - "mean": float
            - "median": float
            - "n_valid": int (perturbations passing same-class filter)
            - "n_total": int (total perturbations generated)
            - "perturbation_scores": list of per-perturbation ratios
            - "empirical_bound_estimate": sampled RHS of Equation 4
            - "theoretical_bound": always None (compatibility tombstone)

    Example:
        >>> from explainiverse.evaluation import compute_relative_input_stability
        >>> score = compute_relative_input_stability(
        ...     explainer, model, instance, n_perturbations=50, seed=42
        ... )
        >>> print(f"RIS: {score:.4f}")

    Reference:
        Agarwal, C., Johnson, N., Pawelczyk, M., Krishna, S., Saxena, E.,
        Zitnik, M., & Lakkaraju, H. (2022). Rethinking Stability for
        Attribution-based Explanations. arXiv:2203.06877. Equation 2.
    """
    return_details = _validate_bool(return_details, "return_details")
    instance = _validate_instance(instance)
    _validate_relative_parameters(
        n_perturbations=n_perturbations,
        noise_scale=noise_scale,
        norm_ord=norm_ord,
        epsilon_min=epsilon_min,
        aggregation=aggregation,
    )
    n_features = len(instance)
    rng = np.random.default_rng(seed)

    # Get original explanation and prediction
    original_attr, explained_target = _get_explanation_vector_and_target(
        explainer, instance, n_features, target_class=target_class
    )
    original_class = _get_predicted_class(model, instance)

    # Generate perturbations
    perturbed = _generate_mixed_perturbations(
        instance,
        n_perturbations,
        noise_scale,
        rng,
        feature_types=feature_types,
        discrete_flip_prob=discrete_flip_prob,
    )

    # Pre-compute the representation used by the optional sampled Equation 4
    # right-hand-side diagnostic.
    repr_orig = None
    if representation_fn is not None:
        repr_orig = _evaluate_single_vector_fn(
            representation_fn, instance, name="representation_fn"
        )

    # Evaluate each perturbation
    per_perturbation_scores: List[float] = []
    # Retain the exact evaluations used in the main pass. Re-evaluating a
    # stochastic/stateful representation function would mix incompatible
    # samples into the diagnostic.
    rrs_scores_for_bound: List[float] = []
    representation_differences: List[Tuple[np.ndarray, np.ndarray]] = []

    for i in range(n_perturbations):
        x_prime = perturbed[i]

        # Same-class filter: ŷ_x = ŷ_x'
        pred_class = _get_predicted_class(model, x_prime)
        if pred_class != original_class:
            continue

        # Get perturbed explanation
        perturbed_attr = _get_explanation_vector(
            explainer,
            x_prime,
            n_features,
            target_class=target_class,
            expected_target=explained_target,
        )

        # Numerator: ||(e_x - e_x') / e_x||_p  (element-wise percent change)
        numerator_vec = _element_wise_percent_change(original_attr, perturbed_attr, epsilon_min)
        numerator = _vector_norm(numerator_vec, norm_ord)

        # Denominator: max(||(x - x') / x||_p, epsilon_min)
        denom_vec = _element_wise_percent_change(instance, x_prime, epsilon_min)
        denom = max(_vector_norm(denom_vec, norm_ord), epsilon_min)

        ratio = _finite_ratio(numerator, denom, "relative input stability")
        per_perturbation_scores.append(ratio)

        # Collect components for the sampled Equation 4 RHS diagnostic.
        if representation_fn is not None:
            if repr_orig is None:
                raise RuntimeError("representation_fn anchor evaluation is unavailable")
            repr_pert = _evaluate_single_vector_fn(
                representation_fn, x_prime, name="representation_fn"
            )
            if repr_pert.shape != repr_orig.shape:
                raise ValueError("representation_fn changed output shape across perturbations.")
            repr_pct = _element_wise_percent_change(repr_orig, repr_pert, epsilon_min)
            repr_denom = max(_vector_norm(repr_pct, norm_ord), epsilon_min)
            rrs_ratio = _finite_ratio(numerator, repr_denom, "relative representation stability")
            rrs_scores_for_bound.append(rrs_ratio)
            representation_differences.append(
                (
                    _vector_difference(instance, x_prime, "input"),
                    _vector_difference(repr_orig, repr_pert, "representation"),
                )
            )

    n_valid = len(per_perturbation_scores)

    # Emit an explicitly heuristic low-count diagnostic.
    if 0 < n_valid < 5:
        warnings.warn(
            f"Only {n_valid}/{n_perturbations} perturbations passed the "
            f"same-class filter. RIS has fewer than five valid draws; this "
            f"warning threshold is a library diagnostic, not a paper criterion. "
            f"Consider increasing n_perturbations or decreasing noise_scale.",
            stacklevel=2,
        )

    agg_result = _aggregate_perturbation_scores(per_perturbation_scores, aggregation)

    # Equation 4's RHS is λ₁ · L₁ · RRS, where
    # λ₁ = ||L(x)||_p / ||x||_p. Here both L₁ and RRS are finite sampled
    # maxima, so this is only a diagnostic estimate of that RHS. It is not a
    # certified Lipschitz constant or a theoretical upper bound.
    empirical_bound_estimate = None
    if representation_fn is not None and rrs_scores_for_bound:
        if repr_orig is None:
            raise RuntimeError("representation_fn anchor evaluation is unavailable")
        repr_norm = _vector_norm(repr_orig, norm_ord)
        input_norm = _vector_norm(instance, norm_ord)
        l1_estimates: List[float] = []
        for input_delta, representation_delta in representation_differences:
            input_diff = _vector_norm(input_delta, norm_ord)
            if input_diff == 0:
                continue
            representation_diff = _vector_norm(representation_delta, norm_ord)
            l1_estimates.append(
                _finite_ratio(representation_diff, input_diff, "sampled representation Lipschitz")
            )
        # λ₁ is undefined for a zero-norm anchor. Do not manufacture a finite
        # value with epsilon regularisation and call it Equation 4's RHS.
        if input_norm != 0 and l1_estimates:
            lambda_1 = _finite_ratio(repr_norm, input_norm, "Equation 4 lambda")
            l1_est = float(np.max(l1_estimates))
            max_rrs = float(np.max(rrs_scores_for_bound))
            empirical_bound_estimate = float(lambda_1 * l1_est * max_rrs)

    if not return_details:
        return agg_result["score"]

    return {
        "score": agg_result["score"],
        "max": agg_result["max"],
        "mean": agg_result["mean"],
        "median": agg_result["median"],
        "n_valid": n_valid,
        "n_total": n_perturbations,
        "perturbation_scores": agg_result["perturbation_scores"],
        "empirical_bound_estimate": empirical_bound_estimate,
        "theoretical_bound": None,
        "aggregation_is_canonical": aggregation == "max",
        "perturbation_space": "preprocessed_input_space",
        "discrete_perturbation_contract": "paper_text_bernoulli_replacement",
    }


def compute_batch_relative_input_stability(
    explainer: BaseExplainer,
    model,
    X: np.ndarray,
    n_perturbations: int = 50,
    noise_scale: float = 0.05,
    norm_ord: Union[int, float] = 2,
    epsilon_min: float = 1e-7,
    aggregation: str = "max",
    feature_types: Optional[np.ndarray] = None,
    discrete_flip_prob: float = 0.03,
    max_instances: Optional[int] = None,
    seed: Optional[int] = None,
    target_class: Optional[int] = None,
) -> dict:
    """
    Compute Relative Input Stability over a batch of instances.

    Args:
        explainer: Explainer instance.
        model: Model adapter.
        X: Input data (2D array of shape (n_instances, n_features)).
        n_perturbations: Perturbations per instance.
        noise_scale: Unbounded Gaussian noise standard deviation for
            continuous features in the supplied feature space.
        norm_ord: Norm order.
        epsilon_min: Division-by-zero floor.
        aggregation: ``"max"`` implements Equation 2; ``"mean"`` and
            ``"median"`` are noncanonical summaries.
        feature_types: Feature type array ("continuous"/"discrete").
        discrete_flip_prob: Bernoulli replacement success probability.
        max_instances: Maximum instances to evaluate (None = all).
        seed: Base seed for per-instance local perturbation generators only.

    Returns:
        Dictionary with statistics over defined scores, plus ``n_attempted``
        and ``n_undefined`` so same-class-filter failures are explicit.
    """
    X = _validate_batch(X)
    n = len(X)
    if max_instances is not None:
        if isinstance(max_instances, bool) or not isinstance(max_instances, (int, np.integer)):
            raise TypeError("max_instances must be an integer or None.")
        if max_instances <= 0:
            raise ValueError("max_instances must be greater than zero.")
        n = min(n, max_instances)

    scores: List[float] = []
    for i in range(n):
        result = compute_relative_input_stability(
            explainer,
            model,
            X[i],
            n_perturbations=n_perturbations,
            noise_scale=noise_scale,
            norm_ord=norm_ord,
            epsilon_min=epsilon_min,
            aggregation=aggregation,
            feature_types=feature_types,
            discrete_flip_prob=discrete_flip_prob,
            seed=seed + i if seed is not None else None,
            target_class=target_class,
        )
        score = _require_scalar_result(result, "compute_relative_input_stability")
        if not np.isnan(score):
            scores.append(score)

    if not scores:
        return {
            "mean": float("nan"),
            "std": 0.0,
            "max": float("nan"),
            "min": float("nan"),
            "scores": [],
            "n_evaluated": 0,
            "n_attempted": n,
            "n_undefined": n,
        }

    return {
        "mean": _finite_mean(scores, "batch relative-input stability"),
        "std": _finite_std(scores, "batch relative-input stability"),
        "max": float(np.max(scores)),
        "min": float(np.min(scores)),
        "scores": scores,
        "n_evaluated": len(scores),
        "n_attempted": n,
        "n_undefined": n - len(scores),
    }


# =============================================================================
# Relative Representation Stability (Agarwal et al., 2022 — Equation 3)
# =============================================================================


def compute_relative_representation_stability(
    explainer: BaseExplainer,
    model,
    instance: np.ndarray,
    representation_fn: Callable[[np.ndarray], np.ndarray],
    n_perturbations: int = 50,
    noise_scale: float = 0.05,
    norm_ord: Union[int, float] = 2,
    epsilon_min: float = 1e-7,
    aggregation: str = "max",
    feature_types: Optional[np.ndarray] = None,
    discrete_flip_prob: float = 0.03,
    seed: Optional[int] = None,
    return_details: bool = False,
    target_class: Optional[int] = None,
) -> Union[float, dict]:
    """
    Compute Relative Representation Stability (Agarwal et al., 2022, Eq 3).

    RRS measures explanation instability relative to changes in the model's
    internal representations:

        RRS(x) = max_{x': ŷ_x = ŷ_x'}
                 ||(e_x − e_x') / e_x||_p
                 / max(||(L_x − L_x') / L_x||_p, ε_min)

    where L(·) denotes the internal model representation (e.g., hidden layer
    embeddings). This metric captures instability that arises when the model
    uses different internal logic paths for similar inputs.

    This implementation uses the paper/Quantus element-wise percent-change
    interpretation. The later OpenXAI author implementation uses ratios of
    whole-vector norms instead, so its numeric scores are not interchangeable.

    A higher score is greater measured instability under this equation and
    perturbation contract.

    With finitely many generated perturbations, ``aggregation="max"`` is a
    sampled maximum and need not equal the full-neighbourhood maximum.

    Args:
        explainer: Explainer instance with .explain() method.
        model: Model adapter with predict() or predict_proba().
        instance: Input instance (1D array).
        representation_fn: Callable receiving one 1D input and returning an
            internal representation as either a 1D vector or one-row 2D
            array. E.g., the pre-ReLU output of the first hidden layer.
        n_perturbations: Number of perturbations. Default: 50.
        noise_scale: Gaussian noise standard deviation in the supplied feature
            space. It is unbounded; no data-bound clipping is applied.
        norm_ord: Norm order. Default: 2.
        epsilon_min: Division-by-zero floor. Default: 1e-7.
        aggregation: ``"max"`` implements Equation 3; ``"mean"`` and
            ``"median"`` are noncanonical summaries.
        feature_types: Feature type array ("continuous"/"discrete").
        discrete_flip_prob: Bernoulli replacement success probability.
        seed: Seed for the local perturbation generator only; it does not seed
            the explainer, model, or representation_fn.
        return_details: If True, return diagnostic dict.

    Returns:
        float or dict (see compute_relative_input_stability for dict format,
        excluding theoretical_bound).

    Reference:
        Agarwal et al. (2022). Equation 3.
    """
    return_details = _validate_bool(return_details, "return_details")
    instance = _validate_instance(instance)
    _validate_relative_parameters(
        n_perturbations=n_perturbations,
        noise_scale=noise_scale,
        norm_ord=norm_ord,
        epsilon_min=epsilon_min,
        aggregation=aggregation,
    )
    n_features = len(instance)
    rng = np.random.default_rng(seed)

    # Get original explanation, prediction, and representation
    original_attr, explained_target = _get_explanation_vector_and_target(
        explainer, instance, n_features, target_class=target_class
    )
    original_class = _get_predicted_class(model, instance)
    repr_orig = _evaluate_single_vector_fn(representation_fn, instance, name="representation_fn")

    # Generate perturbations
    perturbed = _generate_mixed_perturbations(
        instance,
        n_perturbations,
        noise_scale,
        rng,
        feature_types=feature_types,
        discrete_flip_prob=discrete_flip_prob,
    )

    # Evaluate each perturbation
    per_perturbation_scores = []

    for i in range(n_perturbations):
        x_prime = perturbed[i]

        # Same-class filter
        pred_class = _get_predicted_class(model, x_prime)
        if pred_class != original_class:
            continue

        # Get perturbed explanation and representation
        perturbed_attr = _get_explanation_vector(
            explainer,
            x_prime,
            n_features,
            target_class=target_class,
            expected_target=explained_target,
        )
        repr_pert = _evaluate_single_vector_fn(representation_fn, x_prime, name="representation_fn")
        if repr_pert.shape != repr_orig.shape:
            raise ValueError("representation_fn changed output shape across perturbations.")

        # Numerator: ||(e_x - e_x') / e_x||_p
        numerator_vec = _element_wise_percent_change(original_attr, perturbed_attr, epsilon_min)
        numerator = _vector_norm(numerator_vec, norm_ord)

        # Denominator: max(||(L_x - L_x') / L_x||_p, epsilon_min)
        denom_vec = _element_wise_percent_change(repr_orig, repr_pert, epsilon_min)
        denom = max(_vector_norm(denom_vec, norm_ord), epsilon_min)

        ratio = _finite_ratio(numerator, denom, "relative representation stability")
        per_perturbation_scores.append(ratio)

    n_valid = len(per_perturbation_scores)

    # Emit an explicitly heuristic low-count diagnostic.
    if 0 < n_valid < 5:
        warnings.warn(
            f"Only {n_valid}/{n_perturbations} perturbations passed the "
            f"same-class filter. RRS has fewer than five valid draws; this "
            f"warning threshold is a library diagnostic, not a paper criterion. "
            f"Consider increasing n_perturbations or decreasing noise_scale.",
            stacklevel=2,
        )

    agg_result = _aggregate_perturbation_scores(per_perturbation_scores, aggregation)

    if not return_details:
        return agg_result["score"]

    return {
        "score": agg_result["score"],
        "max": agg_result["max"],
        "mean": agg_result["mean"],
        "median": agg_result["median"],
        "n_valid": n_valid,
        "n_total": n_perturbations,
        "perturbation_scores": agg_result["perturbation_scores"],
        "aggregation_is_canonical": aggregation == "max",
        "perturbation_space": "preprocessed_input_space",
        "discrete_perturbation_contract": "paper_text_bernoulli_replacement",
    }


def compute_batch_relative_representation_stability(
    explainer: BaseExplainer,
    model,
    X: np.ndarray,
    representation_fn: Callable[[np.ndarray], np.ndarray],
    n_perturbations: int = 50,
    noise_scale: float = 0.05,
    norm_ord: Union[int, float] = 2,
    epsilon_min: float = 1e-7,
    aggregation: str = "max",
    feature_types: Optional[np.ndarray] = None,
    discrete_flip_prob: float = 0.03,
    max_instances: Optional[int] = None,
    seed: Optional[int] = None,
    target_class: Optional[int] = None,
) -> dict:
    """
    Compute Relative Representation Stability over a batch of instances.

    Args:
        explainer: Explainer instance.
        model: Model adapter.
        X: Input data (2D array).
        representation_fn: Model representation extractor.
        n_perturbations: Perturbations per instance.
        noise_scale: Unbounded Gaussian noise standard deviation in the
            supplied feature space.
        norm_ord: Norm order.
        epsilon_min: Division-by-zero floor.
        aggregation: ``"max"`` implements Equation 3; ``"mean"`` and
            ``"median"`` are noncanonical summaries.
        feature_types: Feature type array.
        discrete_flip_prob: Bernoulli replacement success probability.
        max_instances: Maximum instances to evaluate.
        seed: Base seed for per-instance local perturbation generators only.

    Returns:
        Dictionary with statistics over defined scores, plus ``n_attempted``
        and ``n_undefined`` so same-class-filter failures are explicit.
    """
    X = _validate_batch(X)
    n = len(X)
    if max_instances is not None:
        if isinstance(max_instances, bool) or not isinstance(max_instances, (int, np.integer)):
            raise TypeError("max_instances must be an integer or None.")
        if max_instances <= 0:
            raise ValueError("max_instances must be greater than zero.")
        n = min(n, max_instances)

    scores: List[float] = []
    for i in range(n):
        result = compute_relative_representation_stability(
            explainer,
            model,
            X[i],
            representation_fn=representation_fn,
            n_perturbations=n_perturbations,
            noise_scale=noise_scale,
            norm_ord=norm_ord,
            epsilon_min=epsilon_min,
            aggregation=aggregation,
            feature_types=feature_types,
            discrete_flip_prob=discrete_flip_prob,
            seed=seed + i if seed is not None else None,
            target_class=target_class,
        )
        score = _require_scalar_result(result, "compute_relative_representation_stability")
        if not np.isnan(score):
            scores.append(score)

    if not scores:
        return {
            "mean": float("nan"),
            "std": 0.0,
            "max": float("nan"),
            "min": float("nan"),
            "scores": [],
            "n_evaluated": 0,
            "n_attempted": n,
            "n_undefined": n,
        }

    return {
        "mean": _finite_mean(scores, "batch relative-representation stability"),
        "std": _finite_std(scores, "batch relative-representation stability"),
        "max": float(np.max(scores)),
        "min": float(np.min(scores)),
        "scores": scores,
        "n_evaluated": len(scores),
        "n_attempted": n,
        "n_undefined": n - len(scores),
    }


# =============================================================================
# Relative Output Stability (Agarwal et al., 2022 — Equation 5)
# =============================================================================


def compute_relative_output_stability(
    explainer: BaseExplainer,
    model,
    instance: np.ndarray,
    logit_fn: Callable[[np.ndarray], np.ndarray],
    n_perturbations: int = 50,
    noise_scale: float = 0.05,
    norm_ord: Union[int, float] = 2,
    epsilon_min: float = 1e-7,
    aggregation: str = "max",
    feature_types: Optional[np.ndarray] = None,
    discrete_flip_prob: float = 0.03,
    seed: Optional[int] = None,
    return_details: bool = False,
    target_class: Optional[int] = None,
) -> Union[float, dict]:
    """
    Compute Relative Output Stability (Agarwal et al., 2022, Equation 5).

    ROS measures explanation instability relative to changes in the model's
    output logits:

        ROS(x) = max_{x': ŷ_x = ŷ_x'}
                 ||(e_x − e_x') / e_x||_p
                 / max(||h(x) − h(x')||_p, ε_min)

    where h(x) denotes the output logits (pre-softmax scores). Unlike RIS
    and RRS, the denominator uses absolute difference of logits, NOT percent
    change. This is for black-box models where internal representations are
    not accessible.

    The written Equation 5 and Quantus objective use this absolute-logit
    denominator. The later OpenXAI author implementation applies its generic
    relative-norm helper to logits; its scores are therefore not directly
    comparable.

    A higher score is greater measured instability under this equation and
    perturbation contract.

    With finitely many generated perturbations, ``aggregation="max"`` is a
    sampled maximum and need not equal the full-neighbourhood maximum.

    Args:
        explainer: Explainer instance with .explain() method.
        model: Model adapter with predict() or predict_proba().
        instance: Input instance (1D array).
        logit_fn: Callable receiving one 1D input and returning logits as a
            1D vector or one-row 2D array. E.g., a pre-softmax layer output.
        n_perturbations: Number of perturbations. Default: 50.
        noise_scale: Gaussian noise standard deviation in the supplied feature
            space. It is unbounded; no data-bound clipping is applied.
        norm_ord: Norm order. Default: 2.
        epsilon_min: Division-by-zero floor. Default: 1e-7.
        aggregation: ``"max"`` implements Equation 5; ``"mean"`` and
            ``"median"`` are noncanonical summaries.
        feature_types: Feature type array ("continuous"/"discrete").
        discrete_flip_prob: Bernoulli replacement success probability.
        seed: Seed for the local perturbation generator only; it does not seed
            the explainer, model, or logit_fn.
        return_details: If True, return diagnostic dict.

    Returns:
        float or dict (see compute_relative_input_stability for dict format,
        excluding theoretical_bound).

    Reference:
        Agarwal et al. (2022). Equation 5.
    """
    return_details = _validate_bool(return_details, "return_details")
    instance = _validate_instance(instance)
    _validate_relative_parameters(
        n_perturbations=n_perturbations,
        noise_scale=noise_scale,
        norm_ord=norm_ord,
        epsilon_min=epsilon_min,
        aggregation=aggregation,
    )
    n_features = len(instance)
    rng = np.random.default_rng(seed)

    # Get original explanation, prediction, and logits
    original_attr, explained_target = _get_explanation_vector_and_target(
        explainer, instance, n_features, target_class=target_class
    )
    original_class = _get_predicted_class(model, instance)
    logits_orig = _evaluate_single_vector_fn(logit_fn, instance, name="logit_fn")

    # Generate perturbations
    perturbed = _generate_mixed_perturbations(
        instance,
        n_perturbations,
        noise_scale,
        rng,
        feature_types=feature_types,
        discrete_flip_prob=discrete_flip_prob,
    )

    # Evaluate each perturbation
    per_perturbation_scores = []

    for i in range(n_perturbations):
        x_prime = perturbed[i]

        # Same-class filter
        pred_class = _get_predicted_class(model, x_prime)
        if pred_class != original_class:
            continue

        # Get perturbed explanation and logits
        perturbed_attr = _get_explanation_vector(
            explainer,
            x_prime,
            n_features,
            target_class=target_class,
            expected_target=explained_target,
        )
        logits_pert = _evaluate_single_vector_fn(logit_fn, x_prime, name="logit_fn")
        if logits_pert.shape != logits_orig.shape:
            raise ValueError("logit_fn changed output shape across perturbations.")

        # Numerator: ||(e_x - e_x') / e_x||_p  (element-wise percent change)
        numerator_vec = _element_wise_percent_change(original_attr, perturbed_attr, epsilon_min)
        numerator = _vector_norm(numerator_vec, norm_ord)

        # Denominator: max(||h(x) - h(x')||_p, epsilon_min)
        # NOTE: Equation 5 uses ABSOLUTE difference, not percent change
        logit_diff = _vector_difference(logits_orig, logits_pert, "logit")
        denom = max(_vector_norm(logit_diff, norm_ord), epsilon_min)

        ratio = _finite_ratio(numerator, denom, "relative output stability")
        per_perturbation_scores.append(ratio)

    n_valid = len(per_perturbation_scores)

    # Emit an explicitly heuristic low-count diagnostic.
    if 0 < n_valid < 5:
        warnings.warn(
            f"Only {n_valid}/{n_perturbations} perturbations passed the "
            f"same-class filter. ROS has fewer than five valid draws; this "
            f"warning threshold is a library diagnostic, not a paper criterion. "
            f"Consider increasing n_perturbations or decreasing noise_scale.",
            stacklevel=2,
        )

    agg_result = _aggregate_perturbation_scores(per_perturbation_scores, aggregation)

    if not return_details:
        return agg_result["score"]

    return {
        "score": agg_result["score"],
        "max": agg_result["max"],
        "mean": agg_result["mean"],
        "median": agg_result["median"],
        "n_valid": n_valid,
        "n_total": n_perturbations,
        "perturbation_scores": agg_result["perturbation_scores"],
        "aggregation_is_canonical": aggregation == "max",
        "output_space": "logits",
        "perturbation_space": "preprocessed_input_space",
        "discrete_perturbation_contract": "paper_text_bernoulli_replacement",
    }


def compute_batch_relative_output_stability(
    explainer: BaseExplainer,
    model,
    X: np.ndarray,
    logit_fn: Callable[[np.ndarray], np.ndarray],
    n_perturbations: int = 50,
    noise_scale: float = 0.05,
    norm_ord: Union[int, float] = 2,
    epsilon_min: float = 1e-7,
    aggregation: str = "max",
    feature_types: Optional[np.ndarray] = None,
    discrete_flip_prob: float = 0.03,
    max_instances: Optional[int] = None,
    seed: Optional[int] = None,
    target_class: Optional[int] = None,
) -> dict:
    """
    Compute Relative Output Stability over a batch of instances.

    Args:
        explainer: Explainer instance.
        model: Model adapter.
        X: Input data (2D array).
        logit_fn: Model logit extractor.
        n_perturbations: Perturbations per instance.
        noise_scale: Unbounded Gaussian noise standard deviation in the
            supplied feature space.
        norm_ord: Norm order.
        epsilon_min: Division-by-zero floor.
        aggregation: ``"max"`` implements Equation 5; ``"mean"`` and
            ``"median"`` are noncanonical summaries.
        feature_types: Feature type array.
        discrete_flip_prob: Bernoulli replacement success probability.
        max_instances: Maximum instances to evaluate.
        seed: Base seed for per-instance local perturbation generators only.

    Returns:
        Dictionary with statistics over defined scores, plus ``n_attempted``
        and ``n_undefined`` so same-class-filter failures are explicit.
    """
    X = _validate_batch(X)
    n = len(X)
    if max_instances is not None:
        if isinstance(max_instances, bool) or not isinstance(max_instances, (int, np.integer)):
            raise TypeError("max_instances must be an integer or None.")
        if max_instances <= 0:
            raise ValueError("max_instances must be greater than zero.")
        n = min(n, max_instances)

    scores: List[float] = []
    for i in range(n):
        result = compute_relative_output_stability(
            explainer,
            model,
            X[i],
            logit_fn=logit_fn,
            n_perturbations=n_perturbations,
            noise_scale=noise_scale,
            norm_ord=norm_ord,
            epsilon_min=epsilon_min,
            aggregation=aggregation,
            feature_types=feature_types,
            discrete_flip_prob=discrete_flip_prob,
            seed=seed + i if seed is not None else None,
            target_class=target_class,
        )
        score = _require_scalar_result(result, "compute_relative_output_stability")
        if not np.isnan(score):
            scores.append(score)

    if not scores:
        return {
            "mean": float("nan"),
            "std": 0.0,
            "max": float("nan"),
            "min": float("nan"),
            "scores": [],
            "n_evaluated": 0,
            "n_attempted": n,
            "n_undefined": n,
        }

    return {
        "mean": _finite_mean(scores, "batch relative-output stability"),
        "std": _finite_std(scores, "batch relative-output stability"),
        "max": float(np.max(scores)),
        "min": float(np.min(scores)),
        "scores": scores,
        "n_evaluated": len(scores),
        "n_attempted": n,
        "n_undefined": n - len(scores),
    }


# =============================================================================
# Relative Stability — All-in-One Convenience (Agarwal et al., 2022)
# =============================================================================


def compute_relative_stability(
    explainer: BaseExplainer,
    model,
    instance: np.ndarray,
    representation_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    logit_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    n_perturbations: int = 50,
    noise_scale: float = 0.05,
    norm_ord: Union[int, float] = 2,
    epsilon_min: float = 1e-7,
    aggregation: str = "max",
    feature_types: Optional[np.ndarray] = None,
    discrete_flip_prob: float = 0.03,
    seed: Optional[int] = None,
    return_details: bool = False,
    target_class: Optional[int] = None,
) -> dict:
    """
    Compute all applicable Relative Stability metrics in a single pass.

    This convenience function computes RIS (always), RRS (if representation_fn
    is provided), and ROS (if logit_fn is provided) using shared perturbations
    and explanation computations, avoiding redundant work.

    Args:
        explainer: Explainer instance.
        model: Model adapter.
        instance: Input instance (1D array).
        representation_fn: Optional callable for RRS. Maps input → hidden
            representation.
        logit_fn: Optional callable for ROS. Maps input → output logits.
        n_perturbations: Number of perturbations. Default: 50.
        noise_scale: Unbounded Gaussian noise standard deviation in the
            supplied feature space. No data-bound clipping is applied.
        norm_ord: Norm order. Default: 2.
        epsilon_min: Division-by-zero floor. Default: 1e-7.
        aggregation: ``"max"`` implements Equations 2/3/5; ``"mean"`` and
            ``"median"`` are noncanonical summaries.
        feature_types: Feature type array.
        discrete_flip_prob: Bernoulli replacement success probability.
        seed: Seed for the local perturbation generator only; it does not seed
            stochastic explainers, models, or supplied callables.
        return_details: If True, each metric value is a diagnostic dict.
            If False, each is a float.

    Returns:
        Dict with keys:
            - "ris": float or dict (always computed)
            - "rrs": float, dict, or None (if representation_fn not given)
            - "ros": float, dict, or None (if logit_fn not given)

    Example:
        >>> result = compute_relative_stability(
        ...     explainer, model, instance,
        ...     representation_fn=repr_fn, logit_fn=logit_fn,
        ...     n_perturbations=50, seed=42,
        ... )
        >>> print(f"RIS={result['ris']:.4f}, RRS={result['rrs']:.4f}")
    """
    return_details = _validate_bool(return_details, "return_details")
    instance = _validate_instance(instance)
    _validate_relative_parameters(
        n_perturbations=n_perturbations,
        noise_scale=noise_scale,
        norm_ord=norm_ord,
        epsilon_min=epsilon_min,
        aggregation=aggregation,
    )
    n_features = len(instance)
    rng = np.random.default_rng(seed)

    # Get original explanation and prediction
    original_attr, explained_target = _get_explanation_vector_and_target(
        explainer, instance, n_features, target_class=target_class
    )
    original_class = _get_predicted_class(model, instance)

    # Get original representation and logits if needed
    repr_orig = None
    if representation_fn is not None:
        repr_orig = _evaluate_single_vector_fn(
            representation_fn, instance, name="representation_fn"
        )

    logits_orig = None
    if logit_fn is not None:
        logits_orig = _evaluate_single_vector_fn(logit_fn, instance, name="logit_fn")

    # Generate perturbations (shared across all metrics)
    perturbed = _generate_mixed_perturbations(
        instance,
        n_perturbations,
        noise_scale,
        rng,
        feature_types=feature_types,
        discrete_flip_prob=discrete_flip_prob,
    )

    # Collect per-perturbation scores for each metric
    ris_scores: List[float] = []
    rrs_scores: List[float] = []
    ros_scores: List[float] = []

    for i in range(n_perturbations):
        x_prime = perturbed[i]

        # Same-class filter
        pred_class = _get_predicted_class(model, x_prime)
        if pred_class != original_class:
            continue

        # Get perturbed explanation (shared)
        perturbed_attr = _get_explanation_vector(
            explainer,
            x_prime,
            n_features,
            target_class=target_class,
            expected_target=explained_target,
        )

        # Shared numerator: ||(e_x - e_x') / e_x||_p
        numerator_vec = _element_wise_percent_change(original_attr, perturbed_attr, epsilon_min)
        numerator = _vector_norm(numerator_vec, norm_ord)

        # RIS denominator: max(||(x - x') / x||_p, epsilon_min)
        ris_denom_vec = _element_wise_percent_change(instance, x_prime, epsilon_min)
        ris_denom = max(_vector_norm(ris_denom_vec, norm_ord), epsilon_min)
        ris_scores.append(_finite_ratio(numerator, ris_denom, "relative input stability"))

        # RRS denominator: max(||(L_x - L_x') / L_x||_p, epsilon_min)
        if representation_fn is not None:
            if repr_orig is None:
                raise RuntimeError("representation_fn anchor evaluation is unavailable")
            repr_pert = _evaluate_single_vector_fn(
                representation_fn, x_prime, name="representation_fn"
            )
            if repr_pert.shape != repr_orig.shape:
                raise ValueError("representation_fn changed output shape across perturbations.")
            rrs_denom_vec = _element_wise_percent_change(repr_orig, repr_pert, epsilon_min)
            rrs_denom = max(_vector_norm(rrs_denom_vec, norm_ord), epsilon_min)
            rrs_scores.append(
                _finite_ratio(numerator, rrs_denom, "relative representation stability")
            )

        # ROS denominator: max(||h(x) - h(x')||_p, epsilon_min)
        if logit_fn is not None:
            if logits_orig is None:
                raise RuntimeError("logit_fn anchor evaluation is unavailable")
            logits_pert = _evaluate_single_vector_fn(logit_fn, x_prime, name="logit_fn")
            if logits_pert.shape != logits_orig.shape:
                raise ValueError("logit_fn changed output shape across perturbations.")
            logit_diff = _vector_difference(logits_orig, logits_pert, "logit")
            ros_denom = max(_vector_norm(logit_diff, norm_ord), epsilon_min)
            ros_scores.append(_finite_ratio(numerator, ros_denom, "relative output stability"))

    # Aggregate
    ris_agg = _aggregate_perturbation_scores(ris_scores, aggregation)
    rrs_agg = (
        _aggregate_perturbation_scores(rrs_scores, aggregation)
        if representation_fn is not None
        else None
    )
    ros_agg = (
        _aggregate_perturbation_scores(ros_scores, aggregation) if logit_fn is not None else None
    )

    n_valid = len(ris_scores)

    # Emit an explicitly heuristic low-count diagnostic.
    if 0 < n_valid < 5:
        warnings.warn(
            f"Only {n_valid}/{n_perturbations} perturbations passed the "
            f"same-class filter. Relative stability has fewer than five valid "
            f"draws; this warning threshold is a library diagnostic, not a "
            f"paper criterion. Consider increasing "
            f"n_perturbations or decreasing noise_scale.",
            stacklevel=2,
        )

    if return_details:

        def _build_detail(agg, n_v, n_t):
            if agg is None:
                return None
            return {
                "score": agg["score"],
                "max": agg["max"],
                "mean": agg["mean"],
                "median": agg["median"],
                "n_valid": n_v,
                "n_total": n_t,
                "perturbation_scores": agg["perturbation_scores"],
                "aggregation_is_canonical": aggregation == "max",
                "perturbation_space": "preprocessed_input_space",
                "discrete_perturbation_contract": "paper_text_bernoulli_replacement",
            }

        return {
            "ris": _build_detail(ris_agg, n_valid, n_perturbations),
            "rrs": _build_detail(
                rrs_agg,
                len(rrs_scores),
                n_perturbations,
            ),
            "ros": _build_detail(
                ros_agg,
                len(ros_scores),
                n_perturbations,
            ),
        }

    return {
        "ris": ris_agg["score"],
        "rrs": rrs_agg["score"] if rrs_agg is not None else None,
        "ros": ros_agg["score"] if ros_agg is not None else None,
    }


def compute_batch_relative_stability(
    explainer: BaseExplainer,
    model,
    X: np.ndarray,
    representation_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    logit_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    n_perturbations: int = 50,
    noise_scale: float = 0.05,
    norm_ord: Union[int, float] = 2,
    epsilon_min: float = 1e-7,
    aggregation: str = "max",
    feature_types: Optional[np.ndarray] = None,
    discrete_flip_prob: float = 0.03,
    max_instances: Optional[int] = None,
    seed: Optional[int] = None,
    target_class: Optional[int] = None,
) -> dict:
    """
    Compute all applicable Relative Stability metrics over a batch.

    Args:
        explainer: Explainer instance.
        model: Model adapter.
        X: Input data (2D array).
        representation_fn: Optional callable for RRS.
        logit_fn: Optional callable for ROS.
        n_perturbations: Perturbations per instance.
        noise_scale: Unbounded Gaussian noise standard deviation in the
            supplied feature space.
        norm_ord: Norm order.
        epsilon_min: Division-by-zero floor.
        aggregation: ``"max"`` implements Equations 2/3/5; ``"mean"`` and
            ``"median"`` are noncanonical summaries.
        feature_types: Feature type array.
        discrete_flip_prob: Bernoulli replacement success probability.
        max_instances: Maximum instances to evaluate.
        seed: Base seed for per-instance local perturbation generators only.

    Returns:
        Dict with keys "ris", "rrs", "ros". Each is a dict with
        mean, std, max, min, scores, n_evaluated (or None if not computed).
    """
    X = _validate_batch(X)
    n = len(X)
    if max_instances is not None:
        if isinstance(max_instances, bool) or not isinstance(max_instances, (int, np.integer)):
            raise TypeError("max_instances must be an integer or None.")
        if max_instances <= 0:
            raise ValueError("max_instances must be greater than zero.")
        n = min(n, max_instances)

    ris_scores: List[float] = []
    rrs_scores: List[float] = []
    ros_scores: List[float] = []

    for i in range(n):
        result = compute_relative_stability(
            explainer,
            model,
            X[i],
            representation_fn=representation_fn,
            logit_fn=logit_fn,
            n_perturbations=n_perturbations,
            noise_scale=noise_scale,
            norm_ord=norm_ord,
            epsilon_min=epsilon_min,
            aggregation=aggregation,
            feature_types=feature_types,
            discrete_flip_prob=discrete_flip_prob,
            seed=seed + i if seed is not None else None,
            target_class=target_class,
        )
        if not np.isnan(result["ris"]):
            ris_scores.append(result["ris"])
        if result["rrs"] is not None:
            if not np.isnan(result["rrs"]):
                rrs_scores.append(result["rrs"])
        if result["ros"] is not None:
            if not np.isnan(result["ros"]):
                ros_scores.append(result["ros"])

    def _batch_stats(scores_list: Optional[List[float]]) -> Optional[dict]:
        if scores_list is None:
            return None
        if not scores_list:
            return {
                "mean": float("nan"),
                "std": 0.0,
                "max": float("nan"),
                "min": float("nan"),
                "scores": [],
                "n_evaluated": 0,
                "n_attempted": n,
                "n_undefined": n,
            }
        return {
            "mean": _finite_mean(scores_list, "batch relative stability"),
            "std": _finite_std(scores_list, "batch relative stability"),
            "max": float(np.max(scores_list)),
            "min": float(np.min(scores_list)),
            "scores": scores_list,
            "n_evaluated": len(scores_list),
            "n_attempted": n,
            "n_undefined": n - len(scores_list),
        }

    return {
        "ris": _batch_stats(ris_scores),
        "rrs": _batch_stats(rrs_scores if representation_fn is not None else None),
        "ros": _batch_stats(ros_scores if logit_fn is not None else None),
    }


# =============================================================================
# Consistency (Dasgupta et al., 2022 — ICML)
# =============================================================================


def _get_top_k_features(
    attribution_vector: np.ndarray,
    k: int,
    tie_policy: str = "stable_order",
    *,
    return_tie_incidence: bool = False,
) -> Union[frozenset, Tuple[frozenset, bool]]:
    """
    Extract the indices of the top-k features by absolute attribution magnitude.

    Returns a frozenset for hashable set comparison.

    Args:
        attribution_vector: 1D array of attribution values.
        k: Number of top features to select.

    Returns:
        frozenset of integer indices.
    """
    if tie_policy not in {"stable_order", "reject", "include_all"}:
        raise ValueError("tie_policy must be 'stable_order', 'reject', or 'include_all'.")
    k = min(k, len(attribution_vector))
    magnitudes = np.abs(attribution_vector)
    feature_indices = np.arange(len(attribution_vector))
    ordered = np.lexsort((feature_indices, -magnitudes))
    cutoff = magnitudes[ordered[k - 1]]
    strictly_above = int(np.sum(magnitudes > cutoff))
    tied_at_cutoff = np.flatnonzero(magnitudes == cutoff)
    tie_spans_cutoff = strictly_above < k < strictly_above + len(tied_at_cutoff)

    if tie_policy == "reject" and tie_spans_cutoff:
        raise ValueError(
            "top-k attribution magnitudes contain a tie spanning the cutoff; "
            "choose tie_policy='stable_order' or 'include_all' explicitly"
        )
    if tie_policy == "include_all":
        indices = np.flatnonzero(magnitudes >= cutoff)
    else:
        indices = ordered[:k]
    # The discretisation needs exactly k features. Ties are resolved by the
    # declared feature order (lowest index first), an explicit deterministic
    # policy rather than an accidental consequence of NumPy's sort algorithm.
    selected = frozenset(indices.tolist())
    if return_tie_incidence:
        return selected, tie_spans_cutoff
    return selected


def _get_model_prediction(
    model,
    instance: np.ndarray,
):
    """
    Get the predicted class label for a single instance.

    Handles both predict() and predict_proba() adapters.

    Args:
        model: Model adapter with predict or predict_proba method.
        instance: 1D input array.

    Returns:
        Predicted label or output-column index. Arbitrary label dtypes are
        preserved when the model's ``predict`` method returns labels.
    """
    return _get_predicted_class(model, instance)


def compute_consistency(
    explainer: BaseExplainer,
    model,
    X: np.ndarray,
    top_k: int = 3,
    max_pairs: Optional[int] = None,
    seed: Optional[int] = None,
    tie_policy: str = "stable_order",
    return_details: bool = False,
) -> Union[float, Dict[str, object]]:
    """
    Compute a top-k-discretised empirical Consistency estimate.

    Dasgupta et al. define global consistency as the expected local
    probability that a second input receiving the same explanation also
    receives the same prediction. Their Section 4.2 explicitly permits a
    discretisation ``psi`` for continuous explanation spaces. This endpoint
    chooses ``psi`` to be the unordered set of top-k absolute-attribution
    feature indices:

        m_c = E_x[P(f(x') = f(x) | psi(E(x')) = psi(E(x)))].

    This is one declared operationalisation, not a unique canonical
    discretisation. It intentionally ignores attribution sign and magnitude.

    The exact branch implements the paper's finite-sample estimator: it
    computes one leave-self-out local score per query and averages those
    scores with equal query weight. A query whose discretised explanation is
    unique contributes zero because its consistency is unverifiable from the
    sample. A higher value is greater empirical conditional prediction
    agreement under this discretisation.

    The metric is operationalised by:
      1. Computing explanations for all instances in X.
      2. Discretising each explanation to its top-k feature set.
      3. Grouping instances that share the same top-k set.
      4. Computing prediction agreement with every same-group peer for each
         query, excluding the query itself.
      5. Averaging the query-local scores, not pooling unordered pairs.

    Args:
        explainer: Explainer instance with .explain() method.
        model: Model adapter with predict() or predict_proba().
        X: Input data (2D array of shape (n_instances, n_features)).
            At least two instances are required for a defined result. Repeated
            top-k sets determine how many peer comparisons are available.
        top_k: Number of top features to define the discrete explanation.
            Default: 3. Must be an integer smaller than n_features. The
            number of possible top-k sets is combinatorial and is not
            monotone in k, so k must be treated as a reported
            discretisation parameter.
            Magnitude ties are resolved by declared feature order (lowest
            feature index first), so the selected set is reproducible.
        max_pairs: Historical parameter name. If None, compute the exact
            finite-sample estimator. If set, perform this many with-replacement
            Monte Carlo draws: sample a query uniformly, then sample one of
            its same-explanation peers uniformly. This is an unbiased estimate
            of the same query-weighted target.
        seed: Random seed for Monte Carlo sampling when max_pairs is set.
        tie_policy: ``"stable_order"`` selects exactly ``top_k`` features and
            breaks a cutoff tie by feature index; ``"reject"`` refuses such a
            tie; ``"include_all"`` includes every feature tied at the cutoff.
        return_details: If true, return the score together with the selected
            tie policy, cutoff-tie incidence, and selected-set sizes.

    Returns:
        Consistency score (float) in [0, 1]. Higher = more consistent.
        Returns NaN only when X contains fewer than two instances. If no
        explanation repeats, the conservative finite-sample estimate is 0.

    Example:
        >>> from explainiverse.evaluation import compute_consistency
        >>> score = compute_consistency(explainer, model, X_test, top_k=3)
        >>> print(f"Consistency: {score:.4f}")

    Reference:
        Dasgupta, S., Frost, N., & Moshkovitz, M. (2022). Framework for
        Evaluating Faithfulness of Local Explanations. ICML.
        https://proceedings.mlr.press/v162/dasgupta22a.html
    """
    X = _validate_batch(X)
    n_instances, n_features = X.shape

    if tie_policy not in {"stable_order", "reject", "include_all"}:
        raise ValueError("tie_policy must be 'stable_order', 'reject', or 'include_all'.")
    if not isinstance(return_details, (bool, np.bool_)):
        raise TypeError("return_details must be a boolean.")
    return_details = bool(return_details)

    if isinstance(top_k, (bool, np.bool_)) or not isinstance(top_k, (int, np.integer)):
        raise TypeError("top_k must be an integer.")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if top_k >= n_features:
        raise ValueError(
            f"top_k ({top_k}) must be < n_features ({n_features}). "
            "With top_k == n_features, all explanations would trivially "
            "match and the metric becomes uninformative."
        )
    if n_instances < 2:
        score = float("nan")
        if not return_details:
            return score
        return {
            "score": score,
            "tie_policy": tie_policy,
            "requested_top_k": int(top_k),
            "cutoff_tie_count": 0,
            "cutoff_tie_fraction": 0.0,
            "selected_feature_counts": [],
            "n_instances": n_instances,
            "estimator": "undefined_fewer_than_two_instances",
        }
    if max_pairs is not None:
        if isinstance(max_pairs, bool) or not isinstance(max_pairs, (int, np.integer)):
            raise TypeError("max_pairs must be an integer or None.")
        if max_pairs <= 0:
            raise ValueError("max_pairs must be greater than zero.")

    rng = np.random.default_rng(seed)

    # Step 1: Compute explanations and predictions for all instances
    top_k_sets = []  # frozenset per instance
    cutoff_ties: List[bool] = []
    predictions = []  # predicted class per instance
    for i in range(n_instances):
        attr = _get_explanation_vector(explainer, X[i], n_features)
        pred = _get_model_prediction(model, X[i])
        selected, cutoff_tie = _get_top_k_features(
            attr,
            top_k,
            tie_policy,
            return_tie_incidence=True,
        )
        top_k_sets.append(selected)
        cutoff_ties.append(cutoff_tie)
        predictions.append(pred)

    # Step 2: Group instances by their top-k explanation set
    groups: Dict[frozenset, List[int]] = {}
    for idx, topk_set in enumerate(top_k_sets):
        if topk_set not in groups:
            groups[topk_set] = []
        groups[topk_set].append(idx)

    # Definition 1 is E_x[P(f(x')=f(x) | e(x')=e(x))]. Therefore the exact
    # empirical estimator first computes a local score for every query x and
    # then gives every query equal weight. Pooling unordered pairs instead
    # incorrectly overweights large explanation groups.
    if max_pairs is None:
        local_scores = []
        for i, explanation_label in enumerate(top_k_sets):
            peers = [j for j in groups[explanation_label] if j != i]
            if not peers:
                # The paper's Section 4.1 estimator multiplies singleton
                # explanation groups by zero because they are unverifiable.
                local_scores.append(0.0)
                continue
            local_scores.append(float(np.mean([predictions[j] == predictions[i] for j in peers])))
        score = float(np.mean(local_scores))
        estimator = "exact_query_weighted"
    else:
        # Unbiased Monte Carlo approximation of the same query-weighted quantity:
        # draw x uniformly, then x' uniformly from x's same-explanation peers.
        sampled_scores = []
        query_indices = rng.integers(0, n_instances, size=max_pairs)
        for i in query_indices:
            peers = [j for j in groups[top_k_sets[i]] if j != i]
            if not peers:
                sampled_scores.append(0.0)
                continue
            j = peers[int(rng.integers(0, len(peers)))]
            sampled_scores.append(float(predictions[j] == predictions[i]))
        score = float(np.mean(sampled_scores))
        estimator = "monte_carlo_query_peer"

    if not return_details:
        return score
    tie_count = int(sum(cutoff_ties))
    return {
        "score": score,
        "tie_policy": tie_policy,
        "requested_top_k": int(top_k),
        "cutoff_tie_count": tie_count,
        "cutoff_tie_fraction": tie_count / n_instances,
        "selected_feature_counts": [len(features) for features in top_k_sets],
        "n_instances": n_instances,
        "n_explanation_groups": len(groups),
        "estimator": estimator,
        "max_pairs": None if max_pairs is None else int(max_pairs),
        "seed": seed,
    }


def compute_batch_consistency(
    explainer: BaseExplainer,
    model,
    X: np.ndarray,
    top_k_values: Optional[List[int]] = None,
    max_pairs: Optional[int] = None,
    seed: Optional[int] = None,
    tie_policy: str = "stable_order",
) -> dict:
    """
    Compute Consistency across multiple top-k values.

    This reports the finite-sample consistency statistic under each selected
    top-k discretisation. The cross-k mean is descriptive and does not establish
    that an explanation captures a decision boundary.

    Args:
        explainer: Explainer instance.
        model: Model adapter.
        X: Input data (2D array).
        top_k_values: List of top-k values to evaluate.
            By default, use the entries from [1, 2, 3, 5] that are strictly
            smaller than the feature count.
        max_pairs: Historical name for the number of Monte Carlo query-peer
            draws per top-k evaluation; None computes the exact estimator.
        seed: Seed for Monte Carlo query-peer sampling only.
        tie_policy: Shared cutoff-tie policy for every selected ``top_k``.

    Returns:
        Dictionary with:
            - "scores": Dict mapping each k to its consistency score
            - "mean": Mean consistency across all k values
            - "top_k_values": List of k values evaluated
            - "n_instances": Number of instances in X
    """
    X = _validate_batch(X)
    n_features = X.shape[1]

    if tie_policy not in {"stable_order", "reject", "include_all"}:
        raise ValueError("tie_policy must be 'stable_order', 'reject', or 'include_all'.")

    if top_k_values is None:
        top_k_values = [k for k in [1, 2, 3, 5] if k < n_features]
    else:
        top_k_values = list(top_k_values)
        for k in top_k_values:
            if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
                raise TypeError("Every top_k_values entry must be an integer.")
            if k < 1 or k >= n_features:
                raise ValueError(
                    "Every top_k_values entry must be at least 1 and smaller "
                    "than the number of features."
                )
        if len(set(int(k) for k in top_k_values)) != len(top_k_values):
            raise ValueError("top_k_values must not contain duplicate entries.")

    if not top_k_values:
        return {
            "scores": {},
            "mean": float("nan"),
            "top_k_values": [],
            "n_instances": len(X),
            "tie_policy": tie_policy,
            "details": {},
        }

    scores = {}
    details = {}
    for k in top_k_values:
        detail = compute_consistency(
            explainer,
            model,
            X,
            top_k=k,
            max_pairs=max_pairs,
            seed=seed,
            tie_policy=tie_policy,
            return_details=True,
        )
        if not isinstance(detail, dict):
            raise RuntimeError("compute_consistency unexpectedly returned a scalar")
        details[k] = detail
        score_value = detail.get("score")
        if isinstance(score_value, (bool, np.bool_)) or not isinstance(score_value, Real):
            raise RuntimeError("compute_consistency detail payload has no real score")
        scores[k] = float(score_value)

    valid_scores = [s for s in scores.values() if not np.isnan(s)]
    mean_score = float(np.mean(valid_scores)) if valid_scores else float("nan")

    return {
        "scores": scores,
        "mean": mean_score,
        "top_k_values": top_k_values,
        "n_instances": len(X),
        "tie_policy": tie_policy,
        "details": details,
    }


def compare_consistency_results(
    results: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    """Validate and collect detailed Consistency results under one tie policy.

    The helper deliberately refuses scalar-only payloads and mixed policies;
    without the recorded discretisation policy, numeric scores are not a
    defensible comparison.
    """

    if not isinstance(results, Mapping) or not results:
        raise ValueError("results must be a non-empty mapping of named detail payloads")
    scores: Dict[str, float] = {}
    policies = set()
    for name, payload in results.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("result names must be non-empty strings")
        if not isinstance(payload, Mapping):
            raise TypeError("every consistency result must be a detail mapping")
        policy = payload.get("tie_policy")
        if policy not in {"stable_order", "reject", "include_all"}:
            raise ValueError("every consistency result must record a valid tie_policy")
        score = payload.get("score")
        if isinstance(score, (bool, np.bool_)) or not isinstance(score, Real):
            raise TypeError("every consistency result must contain a real scalar score")
        numeric_score = float(score)
        if not np.isfinite(numeric_score):
            raise ValueError("consistency comparison scores must be finite")
        policies.add(policy)
        scores[name] = numeric_score
    if len(policies) != 1:
        raise ValueError("consistency results with mixed tie_policy values are incomparable")
    return {
        "tie_policy": next(iter(policies)),
        "scores": scores,
        "comparison_contract": "same_consistency_cutoff_tie_policy",
    }


# =============================================================================
# Mean-sensitivity heuristic (noncanonical compatibility endpoint)
# =============================================================================


def compute_avg_sensitivity(
    explainer: BaseExplainer,
    instance: np.ndarray,
    radius: float = 0.1,
    n_samples: int = 50,
    norm_ord: Union[int, float] = 2,
    perturb_norm: str = "l2",
    normalize: bool = False,
    seed: Optional[int] = None,
    target_class: Optional[int] = None,
) -> float:
    """
    Compute a Monte Carlo mean-sensitivity heuristic.

    This function measures the expected change in explanation when the
    input is uniformly perturbed within a ball of radius r:

        AvgSens(E, x, r) = E_{||δ||_p ≤ r} [ ||E(x + δ) - E(x)||_q ]

    Unlike Max-Sensitivity, this captures typical rather than worst-case
    behaviour. Yeh et al. (2019) define Max-Sensitivity, not this average;
    the name is retained for compatibility with evaluation toolkits. A lower
    value is a smaller mean sampled explanation change under this contract.

    Args:
        explainer: Explainer instance with .explain() method.
        instance: Input instance (1D array).
        radius: Radius of the perturbation ball. Default: 0.1.
        n_samples: Number of Monte Carlo samples. Default: 50.
        norm_ord: Norm order for explanation differences.
        perturb_norm: Perturbation ball norm ("l2" or "linf").
        normalize: Optional noncanonical relative normalization. Default False.
        seed: Seed for the local perturbation generator only. It does not seed
            a stochastic explainer.
        target_class: Optional fixed output-column index.

    Returns:
        Mean sampled explanation change (float).

    Relation:
        Uses Yeh et al.'s perturbation-neighbourhood contrast, but replaces
        their canonical maximum with a mean; it must not be reported as the
        paper's Max-Sensitivity.
    """
    instance = _validate_instance(instance)
    _validate_sampling_parameters(n_samples=n_samples, radius=radius)
    _validate_norm_order(norm_ord)
    if not isinstance(normalize, (bool, np.bool_)):
        raise TypeError("normalize must be a boolean.")
    n_features = len(instance)
    rng = np.random.default_rng(seed)

    # Get original explanation
    original_attr, explained_target = _get_explanation_vector_and_target(
        explainer, instance, n_features, target_class=target_class
    )
    original_norm = _vector_norm(original_attr, norm_ord)

    if normalize and original_norm == 0:
        raise ValueError(
            "Normalized sensitivity is undefined for a zero-norm original "
            "explanation. Use normalize=False."
        )

    # Generate perturbations
    if perturb_norm == "l2":
        perturbed = _generate_perturbations_l2(instance, radius, n_samples, rng)
    elif perturb_norm == "linf":
        perturbed = _generate_perturbations_linf(instance, radius, n_samples, rng)
    else:
        raise ValueError(f"perturb_norm must be 'l2' or 'linf', got '{perturb_norm}'")

    # Compute mean explanation distance
    diffs = []
    for i in range(n_samples):
        perturbed_attr = _get_explanation_vector(
            explainer,
            perturbed[i],
            n_features,
            target_class=target_class,
            expected_target=explained_target,
        )
        diff = _vector_norm(
            _vector_difference(original_attr, perturbed_attr, "explanation"), norm_ord
        )
        diffs.append(diff)

    mean_diff = _finite_mean(diffs, "average sensitivity")

    if normalize:
        return _finite_ratio(mean_diff, original_norm, "normalized average sensitivity")
    return float(mean_diff)


def compute_batch_avg_sensitivity(
    explainer: BaseExplainer,
    X: np.ndarray,
    radius: float = 0.1,
    n_samples: int = 50,
    norm_ord: Union[int, float] = 2,
    perturb_norm: str = "l2",
    normalize: bool = False,
    max_instances: Optional[int] = None,
    seed: Optional[int] = None,
    target_class: Optional[int] = None,
) -> dict:
    """
    Compute Avg-Sensitivity over a batch of instances.

    Args:
        explainer: Explainer instance.
        X: Input data (2D array).
        radius: Perturbation radius.
        n_samples: Perturbation samples per instance.
        norm_ord: Norm order for explanation differences.
        perturb_norm: Perturbation ball norm.
        normalize: If True, normalize by original explanation norm.
        max_instances: Maximum instances to evaluate.
        seed: Base seed for per-instance local perturbation generators only.

    Returns:
        Dictionary with mean, std, max, min, scores, n_evaluated.
    """
    X = _validate_batch(X)
    n = len(X)
    if max_instances is not None:
        if isinstance(max_instances, bool) or not isinstance(max_instances, (int, np.integer)):
            raise TypeError("max_instances must be an integer or None.")
        if max_instances <= 0:
            raise ValueError("max_instances must be greater than zero.")
        n = min(n, max_instances)

    scores = []
    for i in range(n):
        score = compute_avg_sensitivity(
            explainer,
            X[i],
            radius=radius,
            n_samples=n_samples,
            norm_ord=norm_ord,
            perturb_norm=perturb_norm,
            normalize=normalize,
            seed=seed + i if seed is not None else None,
            target_class=target_class,
        )
        scores.append(score)

    if not scores:
        return {
            "mean": float("nan"),
            "std": 0.0,
            "max": float("nan"),
            "min": float("nan"),
            "scores": [],
            "n_evaluated": 0,
        }

    return {
        "mean": _finite_mean(scores, "batch average sensitivity"),
        "std": _finite_std(scores, "batch average sensitivity"),
        "max": float(np.max(scores)),
        "min": float(np.min(scores)),
        "scores": scores,
        "n_evaluated": len(scores),
    }


# =============================================================================
# Finite-sample local Lipschitz estimate (compatibility name: Continuity)
# =============================================================================


def compute_continuity(
    explainer: BaseExplainer,
    instance: np.ndarray,
    X_reference: np.ndarray,
    k_neighbors: int = 5,
    norm_ord: Union[int, float] = 2,
    input_distance: str = "euclidean",
    seed: Optional[int] = None,
    target_class: Optional[int] = None,
) -> float:
    """Compute a k-neighbour finite-sample local Lipschitz estimate.

    The prior implementation under this public name was an uncited Spearman
    rank-distance heuristic and has been removed. This compatibility endpoint
    now evaluates

        max_j ||E(x) - E(x_j)|| / d(x, x_j).

    Alvarez-Melis & Jaakkola (2018, Equation 2) use the same fixed-anchor
    finite-sample maximum over an epsilon neighbourhood with Euclidean norms.
    Here the k nearest non-identical reference points define an adaptive
    finite-sample neighbourhood. ``input_distance='euclidean'`` and
    ``norm_ord=2`` recover their distance choices; other explicit choices are
    generalized norm/distance variants. This is not the image-translation
    ``Continuity`` metric implemented by Quantus. ``seed`` is retained for
    compatibility and is unused. Lower values indicate greater local
    stability.
    """
    del seed
    instance = _validate_instance(instance)
    X_reference = _validate_batch(X_reference, "X_reference")
    _validate_norm_order(norm_ord)
    if X_reference.shape[1] != instance.size:
        raise ValueError("X_reference must have the same feature count as instance.")
    if isinstance(k_neighbors, bool) or not isinstance(k_neighbors, (int, np.integer)):
        raise TypeError("k_neighbors must be an integer.")
    if k_neighbors <= 0:
        raise ValueError("k_neighbors must be greater than zero.")

    k_neighbors = min(k_neighbors, len(X_reference))

    if input_distance == "euclidean":
        input_dists = np.asarray(
            [
                _vector_norm(_vector_difference(instance, row, "continuity input"), 2)
                for row in X_reference
            ],
            dtype=np.float64,
        )
    else:
        input_dists = cdist(instance.reshape(1, -1), X_reference, metric=input_distance).reshape(-1)
    finite_nonzero = np.flatnonzero(np.isfinite(input_dists) & (input_dists > 0))
    selected = finite_nonzero[np.argsort(input_dists[finite_nonzero])][:k_neighbors]
    if selected.size == 0:
        return float("nan")

    n_features = instance.size
    original_attr, explained_target = _get_explanation_vector_and_target(
        explainer, instance, n_features, target_class=target_class
    )
    ratios = []
    for index in selected:
        neighbor_attr = _get_explanation_vector(
            explainer,
            X_reference[index],
            n_features,
            target_class=target_class,
            expected_target=explained_target,
        )
        numerator = _vector_norm(
            _vector_difference(original_attr, neighbor_attr, "continuity attribution"),
            norm_ord,
        )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            ratio = _finite_ratio(numerator, input_dists[index], "continuity")
        ratios.append(ratio)
    return float(np.max(ratios))


def compute_batch_continuity(
    explainer: BaseExplainer,
    X: np.ndarray,
    k_neighbors: int = 5,
    norm_ord: Union[int, float] = 2,
    input_distance: str = "euclidean",
    max_instances: Optional[int] = None,
    seed: Optional[int] = None,
    target_class: Optional[int] = None,
) -> dict:
    """
    Compute k-neighbour local Lipschitz estimates over a batch.

    Each instance is evaluated against the remaining instances in X as the
    reference set (leave-one-out).

    Args:
        explainer: Explainer instance.
        X: Input data (2D array).
        k_neighbors: Number of nearest neighbours per instance.
        norm_ord: Norm order for explanation distances.
        input_distance: Input-space distance metric.
        max_instances: Maximum instances to evaluate.
        seed: Retained for compatibility; unused by this deterministic
            reference-neighbour calculation.

    Returns:
        Dictionary with mean, std, max, min, scores, n_evaluated.
    """
    X = _validate_batch(X)
    n = len(X)
    if max_instances is not None:
        if isinstance(max_instances, bool) or not isinstance(max_instances, (int, np.integer)):
            raise TypeError("max_instances must be an integer or None.")
        if max_instances <= 0:
            raise ValueError("max_instances must be greater than zero.")
        n = min(n, max_instances)

    if len(X) < 2:
        return {
            "mean": float("nan"),
            "std": 0.0,
            "max": float("nan"),
            "min": float("nan"),
            "scores": [],
            "n_evaluated": 0,
            "n_attempted": n,
            "n_undefined": n,
        }

    scores = []
    for i in range(n):
        # Use all other instances as reference.
        reference = np.delete(X, i, axis=0)
        score = compute_continuity(
            explainer,
            X[i],
            reference,
            k_neighbors=k_neighbors,
            norm_ord=norm_ord,
            input_distance=input_distance,
            seed=seed,
            target_class=target_class,
        )
        if not np.isnan(score):
            scores.append(score)

    if not scores:
        return {
            "mean": float("nan"),
            "std": 0.0,
            "max": float("nan"),
            "min": float("nan"),
            "scores": [],
            "n_evaluated": 0,
            "n_attempted": n,
            "n_undefined": n,
        }

    return {
        "mean": _finite_mean(scores, "batch continuity"),
        "std": _finite_std(scores, "batch continuity"),
        "max": float(np.max(scores)),
        "min": float(np.min(scores)),
        "scores": scores,
        "n_evaluated": len(scores),
        "n_attempted": n,
        "n_undefined": n - len(scores),
    }
