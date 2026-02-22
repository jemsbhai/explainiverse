# src/explainiverse/evaluation/robustness.py
"""
Robustness evaluation metrics for explanations (Phase 2).

Implements:
- Max-Sensitivity (Yeh et al., 2019)
- Avg-Sensitivity (Yeh et al., 2019)
- Continuity (Montavon et al., 2018; Alvarez-Melis & Jaakkola, 2018)
- Consistency (Dasgupta et al., 2022)
- Relative Input Stability — RIS (Agarwal et al., 2022, Eq 2)
- Relative Representation Stability — RRS (Agarwal et al., 2022, Eq 3)
- Relative Output Stability — ROS (Agarwal et al., 2022, Eq 5)

These metrics evaluate whether explanations are stable under small input
perturbations and whether similar inputs receive similar explanations.

References:
    Yeh, C. K., Hsieh, C. Y., Suggala, A. S., Inber, D. I., & Ravikumar, P.
    (2019). On the (In)fidelity and Sensitivity of Explanations. NeurIPS.
    https://proceedings.neurips.cc/paper/2019/hash/a7471fdc77b3435276507cc8f2571547-Abstract.html

    Montavon, G., Samek, W., & Müller, K. R. (2018). Methods for interpreting
    and understanding deep neural networks. Digital Signal Processing, 73, 1-15.

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
import warnings

import numpy as np
from typing import Union, Callable, List, Dict, Optional
from scipy import stats
from scipy.spatial.distance import cdist
from itertools import combinations

from explainiverse.core.explanation import Explanation
from explainiverse.core.explainer import BaseExplainer


# =============================================================================
# Internal Helpers
# =============================================================================

def _extract_attribution_vector(explanation: Explanation) -> np.ndarray:
    """
    Extract attribution values as a numpy array from an Explanation.

    Preserves feature order from explanation.feature_names if available,
    otherwise uses dictionary iteration order.

    Args:
        explanation: Explanation object with feature_attributions

    Returns:
        1D numpy array of attribution values

    Raises:
        ValueError: If no feature attributions are found
    """
    attributions = explanation.explanation_data.get("feature_attributions", {})
    if not attributions:
        raise ValueError("No feature attributions found in explanation.")

    feature_names = getattr(explanation, 'feature_names', None)
    if feature_names:
        values = [attributions.get(fn, 0.0) for fn in feature_names]
    else:
        values = list(attributions.values())

    return np.array(values, dtype=np.float64)


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
    perturbations = instance + directions * radii

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
    noise = rng.uniform(-radius, radius, size=(n_samples, d))
    return instance + noise


def _get_explanation_vector(
    explainer: BaseExplainer,
    instance: np.ndarray,
    n_features: int,
) -> np.ndarray:
    """
    Get attribution vector for a single instance.

    Sets feature_names on the explanation if not present.

    Args:
        explainer: Explainer instance
        instance: Input (1D array)
        n_features: Expected number of features

    Returns:
        1D numpy array of attributions
    """
    exp = explainer.explain(instance)
    if not getattr(exp, 'feature_names', None):
        exp.feature_names = [f"feature_{i}" for i in range(n_features)]
    return _extract_attribution_vector(exp)


# =============================================================================
# Max-Sensitivity (Yeh et al., 2019)
# =============================================================================

def compute_max_sensitivity(
    explainer: BaseExplainer,
    instance: np.ndarray,
    radius: float = 0.1,
    n_samples: int = 50,
    norm_ord: Union[int, float, str] = 2,
    perturb_norm: str = "l2",
    normalize: bool = True,
    seed: int = None,
) -> float:
    """
    Compute Max-Sensitivity of an explanation method.

    Max-Sensitivity measures the worst-case change in explanation when the
    input is perturbed within a small ball of radius r:

        MaxSens(E, x, r) = max_{||δ||_p ≤ r} ||E(x + δ) - E(x)||_q

    A lower score indicates a more robust explanation.

    Args:
        explainer: Explainer instance with .explain() method.
        instance: Input instance (1D array of shape (n_features,)).
        radius: Radius of the perturbation ball. Default: 0.1.
            For unnormalized features, scale this to the feature range.
        n_samples: Number of Monte Carlo samples to approximate the max.
            More samples give a tighter estimate. Default: 50.
        norm_ord: Norm order for measuring explanation change.
            2 for L2 (default), 1 for L1, np.inf for L∞.
        perturb_norm: Norm for the perturbation ball.
            "l2" (default) or "linf".
        normalize: If True, normalize by the norm of the original explanation
            to produce a relative (scale-invariant) sensitivity. Default: True.
        seed: Random seed for reproducibility.

    Returns:
        Max-Sensitivity score (float). Lower = more robust.
        Returns 0.0 if the original explanation is zero and normalize=True.

    Example:
        >>> from explainiverse.evaluation import compute_max_sensitivity
        >>> score = compute_max_sensitivity(explainer, instance, radius=0.1)
        >>> print(f"Max-Sensitivity: {score:.4f}")

    Reference:
        Yeh et al. (2019). On the (In)fidelity and Sensitivity of
        Explanations. NeurIPS.
    """
    instance = np.asarray(instance, dtype=np.float64).flatten()
    n_features = len(instance)
    rng = np.random.default_rng(seed)

    # Get original explanation
    original_attr = _get_explanation_vector(explainer, instance, n_features)
    original_norm = np.linalg.norm(original_attr, ord=norm_ord)

    if normalize and original_norm < 1e-12:
        return 0.0

    # Generate perturbations
    if perturb_norm == "l2":
        perturbed = _generate_perturbations_l2(instance, radius, n_samples, rng)
    elif perturb_norm == "linf":
        perturbed = _generate_perturbations_linf(instance, radius, n_samples, rng)
    else:
        raise ValueError(f"perturb_norm must be 'l2' or 'linf', got '{perturb_norm}'")

    # Compute explanation distances
    max_diff = 0.0
    for i in range(n_samples):
        try:
            perturbed_attr = _get_explanation_vector(
                explainer, perturbed[i], n_features
            )
            diff = np.linalg.norm(original_attr - perturbed_attr, ord=norm_ord)
            if diff > max_diff:
                max_diff = diff
        except Exception:
            continue

    if normalize and original_norm > 1e-12:
        return float(max_diff / original_norm)
    return float(max_diff)


def compute_batch_max_sensitivity(
    explainer: BaseExplainer,
    X: np.ndarray,
    radius: float = 0.1,
    n_samples: int = 50,
    norm_ord: Union[int, float, str] = 2,
    perturb_norm: str = "l2",
    normalize: bool = True,
    max_instances: int = None,
    seed: int = None,
) -> Dict[str, float]:
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
        seed: Random seed.

    Returns:
        Dictionary with:
            - "mean": Mean Max-Sensitivity across instances
            - "std": Standard deviation
            - "max": Worst-case Max-Sensitivity
            - "min": Best-case Max-Sensitivity
            - "scores": List of per-instance scores
            - "n_evaluated": Number of instances evaluated
    """
    X = np.asarray(X)
    n = len(X)
    if max_instances is not None:
        n = min(n, max_instances)

    scores = []
    for i in range(n):
        try:
            score = compute_max_sensitivity(
                explainer, X[i], radius=radius, n_samples=n_samples,
                norm_ord=norm_ord, perturb_norm=perturb_norm,
                normalize=normalize,
                seed=seed + i if seed is not None else None,
            )
            scores.append(score)
        except Exception:
            continue

    if not scores:
        return {"mean": float("nan"), "std": 0.0, "max": float("nan"),
                "min": float("nan"), "scores": [], "n_evaluated": 0}

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "max": float(np.max(scores)),
        "min": float(np.min(scores)),
        "scores": scores,
        "n_evaluated": len(scores),
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
    Discrete (binary) features: independent Bernoulli flip with probability p.

    Following Appendix B of Agarwal et al. (2022):
    - Continuous perturbations: x' = x + N(0, 0.05)
    - Discrete perturbations: flip with p = 0.03 (small p to avoid OOD)

    Args:
        instance: 1D input array.
        n_perturbations: Number of perturbed copies to generate.
        noise_scale: Standard deviation of Gaussian noise for continuous
            features. Default paper value: 0.05.
        rng: NumPy random generator.
        feature_types: 1D array of strings, one per feature. Values:
            "continuous" (default) or "discrete". If None, all continuous.
        discrete_flip_prob: Probability of flipping each discrete feature.
            Default paper value: 0.03.

    Returns:
        Array of shape (n_perturbations, n_features).
    """
    d = len(instance)
    perturbed = np.tile(instance, (n_perturbations, 1))

    if feature_types is None:
        # All continuous — simple Gaussian perturbation
        noise = rng.normal(0, noise_scale, size=(n_perturbations, d))
        perturbed = perturbed + noise
    else:
        feature_types = np.asarray(feature_types)
        continuous_mask = feature_types == "continuous"
        discrete_mask = feature_types == "discrete"

        # Continuous features: additive Gaussian noise
        if np.any(continuous_mask):
            n_cont = np.sum(continuous_mask)
            noise = rng.normal(0, noise_scale, size=(n_perturbations, n_cont))
            perturbed[:, continuous_mask] += noise

        # Discrete features: Bernoulli flip
        if np.any(discrete_mask):
            n_disc = np.sum(discrete_mask)
            flip_mask = rng.random(size=(n_perturbations, n_disc)) < discrete_flip_prob
            # Flip: 0 → 1, 1 → 0 (XOR with flip mask)
            perturbed[:, discrete_mask] = np.where(
                flip_mask,
                1.0 - perturbed[:, discrete_mask],
                perturbed[:, discrete_mask],
            )

    return perturbed


def _element_wise_percent_change(
    original: np.ndarray,
    perturbed: np.ndarray,
    epsilon_min: float = 1e-7,
) -> np.ndarray:
    """
    Compute element-wise percent change: (original - perturbed) / original.

    Applies an epsilon floor to the denominator to prevent division by zero
    when elements of the original vector are zero or near-zero.

    Args:
        original: 1D array (the reference vector).
        perturbed: 1D array (the perturbed vector).
        epsilon_min: Floor for absolute value of denominator elements.

    Returns:
        1D array of element-wise percent changes.
    """
    # Replace near-zero denominator elements with epsilon_min to prevent
    # division by zero. We use positive epsilon for near-zero values;
    # sign does not matter here because the result is wrapped in a norm.
    safe_denom = np.copy(original)
    near_zero = np.abs(original) < epsilon_min
    safe_denom[near_zero] = epsilon_min
    return (original - perturbed) / safe_denom


def _aggregate_perturbation_scores(
    scores: List[float],
    aggregation: str = "max",
) -> dict:
    """
    Aggregate per-perturbation scores and build diagnostic dict.

    Args:
        scores: List of per-perturbation ratio scores.
        aggregation: "max" (paper default), "mean", or "median".

    Returns:
        Dict with score, max, mean, median, perturbation_scores.

    Raises:
        ValueError: If aggregation is not one of the valid options.
    """
    valid_aggs = {"max", "mean", "median"}
    if aggregation not in valid_aggs:
        raise ValueError(
            f"aggregation must be one of {valid_aggs}, got '{aggregation}'"
        )

    if not scores:
        return {
            "score": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
            "perturbation_scores": [],
        }

    arr = np.array(scores, dtype=np.float64)
    result = {
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "perturbation_scores": scores,
    }
    result["score"] = result[aggregation]
    return result


def _get_predicted_class(
    model,
    instance: np.ndarray,
) -> int:
    """
    Get predicted class label for a single instance.

    Handles both predict_proba() and predict() interfaces.

    Args:
        model: Model adapter.
        instance: 1D input array.

    Returns:
        Integer predicted class label.
    """
    instance_2d = instance.reshape(1, -1)

    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(instance_2d)
        if isinstance(proba, np.ndarray):
            if proba.ndim == 2:
                return int(np.argmax(proba[0]))
            return int(np.argmax(proba))

    if hasattr(model, 'predict'):
        pred = model.predict(instance_2d)
        if isinstance(pred, np.ndarray):
            pred = pred.flatten()[0]
        return int(pred)

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
) -> Union[float, dict]:
    """
    Compute Relative Input Stability (Agarwal et al., 2022, Equation 2).

    RIS measures the instability of an explanation by computing the maximum
    ratio of percent change in explanation to percent change in input across
    perturbations that preserve the predicted class:

        RIS(x) = max_{x': ŷ_x = ŷ_x'}
                 ||(e_x − e_x') / e_x||_p
                 / max(||(x − x') / x||_p, ε_min)

    A higher score indicates higher instability. Lower is better.

    The numerator uses element-wise percent change in explanation, enabling
    comparison across explanation methods with different magnitude ranges.
    The denominator normalises by the percent change in input.

    Args:
        explainer: Explainer instance with .explain() method.
        model: Model adapter with predict() or predict_proba().
        instance: Input instance (1D array of shape (n_features,)).
        n_perturbations: Number of perturbations to generate. Default: 50
            (paper recommendation).
        noise_scale: Standard deviation of Gaussian noise for continuous
            features. Default: 0.05 (paper recommendation).
        norm_ord: Norm order for computing vector norms. Default: 2 (L2).
            Supports 1, 2, np.inf.
        epsilon_min: Floor to prevent division by zero in element-wise
            percent change and in the denominator norm. Default: 1e-7.
        aggregation: How to aggregate per-perturbation scores.
            "max" (paper default — worst-case instability),
            "mean" (expected instability), or "median" (robust estimate).
        feature_types: 1D array of "continuous" or "discrete" per feature.
            If None, all features are treated as continuous.
            Discrete features are perturbed via Bernoulli flips.
        discrete_flip_prob: Probability of flipping each discrete feature.
            Default: 0.03 (paper recommendation).
        seed: Random seed for reproducibility.
        return_details: If True, return a dict with full diagnostics.
            If False (default), return a single float score.
        representation_fn: Optional callable mapping input → hidden
            representation. If provided and return_details=True, the
            theoretical upper bound (Equation 4) is included.

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
            - "theoretical_bound": float or None (Equation 4, if
              representation_fn is provided)

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
    instance = np.asarray(instance, dtype=np.float64).flatten()
    n_features = len(instance)
    rng = np.random.default_rng(seed)

    # Validate aggregation early
    valid_aggs = {"max", "mean", "median"}
    if aggregation not in valid_aggs:
        raise ValueError(
            f"aggregation must be one of {valid_aggs}, got '{aggregation}'"
        )

    # Get original explanation and prediction
    original_attr = _get_explanation_vector(explainer, instance, n_features)
    original_class = _get_predicted_class(model, instance)

    # Generate perturbations
    perturbed = _generate_mixed_perturbations(
        instance, n_perturbations, noise_scale, rng,
        feature_types=feature_types, discrete_flip_prob=discrete_flip_prob,
    )

    # Pre-compute representation for theoretical bound (avoid redundant calls)
    repr_orig = None
    if representation_fn is not None:
        repr_orig = np.asarray(
            representation_fn(instance), dtype=np.float64
        ).flatten()

    # Evaluate each perturbation
    per_perturbation_scores = []
    # For theoretical bound: collect RRS ratios if representation_fn given
    rrs_scores_for_bound = [] if representation_fn is not None else None

    for i in range(n_perturbations):
        x_prime = perturbed[i]

        # Same-class filter: ŷ_x = ŷ_x'
        try:
            pred_class = _get_predicted_class(model, x_prime)
        except Exception:
            continue
        if pred_class != original_class:
            continue

        # Get perturbed explanation
        try:
            perturbed_attr = _get_explanation_vector(
                explainer, x_prime, n_features
            )
        except Exception:
            continue

        # Numerator: ||(e_x - e_x') / e_x||_p  (element-wise percent change)
        numerator_vec = _element_wise_percent_change(
            original_attr, perturbed_attr, epsilon_min
        )
        numerator = np.linalg.norm(numerator_vec, ord=norm_ord)

        # Denominator: max(||(x - x') / x||_p, epsilon_min)
        denom_vec = _element_wise_percent_change(
            instance, x_prime, epsilon_min
        )
        denom = max(np.linalg.norm(denom_vec, ord=norm_ord), epsilon_min)

        ratio = float(numerator / denom)
        per_perturbation_scores.append(ratio)

        # Collect RRS ratio for theoretical bound
        if representation_fn is not None:
            try:
                repr_pert = np.asarray(
                    representation_fn(x_prime), dtype=np.float64
                ).flatten()
                repr_pct = _element_wise_percent_change(
                    repr_orig, repr_pert, epsilon_min
                )
                repr_denom = max(
                    np.linalg.norm(repr_pct, ord=norm_ord), epsilon_min
                )
                rrs_ratio = float(numerator / repr_denom)
                rrs_scores_for_bound.append(rrs_ratio)
            except Exception:
                pass

    n_valid = len(per_perturbation_scores)

    # Warn if too few valid perturbations for statistically reliable results
    if 0 < n_valid < 5:
        warnings.warn(
            f"Only {n_valid}/{n_perturbations} perturbations passed the "
            f"same-class filter. RIS score may be statistically unreliable. "
            f"Consider increasing n_perturbations or noise_scale.",
            stacklevel=2,
        )

    agg_result = _aggregate_perturbation_scores(
        per_perturbation_scores, aggregation
    )

    # Compute theoretical bound (Equation 4): RIS ≤ λ₁ · L₁ · RRS
    # where λ₁ = ||L(x)||_p / ||x||_p and L₁ is the local Lipschitz
    # constant estimated as max ||L(x) - L(x')||_p / ||x - x'||_p.
    theoretical_bound = None
    if representation_fn is not None and rrs_scores_for_bound:
        try:
            # repr_orig already computed before the perturbation loop
            repr_norm = np.linalg.norm(repr_orig, ord=norm_ord)
            input_norm = np.linalg.norm(instance, ord=norm_ord)
            lambda_1 = repr_norm / max(input_norm, epsilon_min)

            max_rrs = float(np.max(rrs_scores_for_bound))

            # Estimate L₁ from valid perturbation pairs
            l1_estimates = []
            for i_pert in range(n_perturbations):
                x_p = perturbed[i_pert]
                try:
                    pred_p = _get_predicted_class(model, x_p)
                except Exception:
                    continue
                if pred_p != original_class:
                    continue
                try:
                    repr_p = np.asarray(
                        representation_fn(x_p), dtype=np.float64
                    ).flatten()
                    input_diff = np.linalg.norm(
                        instance - x_p, ord=norm_ord
                    )
                    repr_diff = np.linalg.norm(
                        repr_orig - repr_p, ord=norm_ord
                    )
                    if input_diff > epsilon_min:
                        l1_estimates.append(repr_diff / input_diff)
                except Exception:
                    continue
            if l1_estimates:
                l1_est = float(np.max(l1_estimates))
                theoretical_bound = float(lambda_1 * l1_est * max_rrs)
        except Exception:
            theoretical_bound = None

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
        "theoretical_bound": theoretical_bound,
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
) -> Dict[str, Union[float, list]]:
    """
    Compute Relative Input Stability over a batch of instances.

    Args:
        explainer: Explainer instance.
        model: Model adapter.
        X: Input data (2D array of shape (n_instances, n_features)).
        n_perturbations: Perturbations per instance.
        noise_scale: Gaussian noise std for continuous features.
        norm_ord: Norm order.
        epsilon_min: Division-by-zero floor.
        aggregation: Aggregation mode ("max", "mean", "median").
        feature_types: Feature type array ("continuous"/"discrete").
        discrete_flip_prob: Flip probability for discrete features.
        max_instances: Maximum instances to evaluate (None = all).
        seed: Random seed.

    Returns:
        Dictionary with mean, std, max, min, scores, n_evaluated.
    """
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    if max_instances is not None:
        n = min(n, max_instances)

    scores = []
    for i in range(n):
        try:
            score = compute_relative_input_stability(
                explainer, model, X[i],
                n_perturbations=n_perturbations,
                noise_scale=noise_scale, norm_ord=norm_ord,
                epsilon_min=epsilon_min, aggregation=aggregation,
                feature_types=feature_types,
                discrete_flip_prob=discrete_flip_prob,
                seed=seed + i if seed is not None else None,
            )
            if not np.isnan(score):
                scores.append(score)
        except Exception:
            continue

    if not scores:
        return {"mean": float("nan"), "std": 0.0, "max": float("nan"),
                "min": float("nan"), "scores": [], "n_evaluated": 0}

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "max": float(np.max(scores)),
        "min": float(np.min(scores)),
        "scores": scores,
        "n_evaluated": len(scores),
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

    A higher score indicates higher instability. Lower is better.

    Args:
        explainer: Explainer instance with .explain() method.
        model: Model adapter with predict() or predict_proba().
        instance: Input instance (1D array).
        representation_fn: Callable that maps input (1D or 2D array) to
            internal model representation (1D or 2D array). E.g., the
            pre-ReLU output of the first hidden layer.
        n_perturbations: Number of perturbations. Default: 50.
        noise_scale: Gaussian noise std. Default: 0.05.
        norm_ord: Norm order. Default: 2.
        epsilon_min: Division-by-zero floor. Default: 1e-7.
        aggregation: "max" (default), "mean", or "median".
        feature_types: Feature type array ("continuous"/"discrete").
        discrete_flip_prob: Flip probability for discrete features.
        seed: Random seed.
        return_details: If True, return diagnostic dict.

    Returns:
        float or dict (see compute_relative_input_stability for dict format,
        excluding theoretical_bound).

    Reference:
        Agarwal et al. (2022). Equation 3.
    """
    instance = np.asarray(instance, dtype=np.float64).flatten()
    n_features = len(instance)
    rng = np.random.default_rng(seed)

    valid_aggs = {"max", "mean", "median"}
    if aggregation not in valid_aggs:
        raise ValueError(
            f"aggregation must be one of {valid_aggs}, got '{aggregation}'"
        )

    # Get original explanation, prediction, and representation
    original_attr = _get_explanation_vector(explainer, instance, n_features)
    original_class = _get_predicted_class(model, instance)
    repr_orig = np.asarray(
        representation_fn(instance), dtype=np.float64
    ).flatten()

    # Generate perturbations
    perturbed = _generate_mixed_perturbations(
        instance, n_perturbations, noise_scale, rng,
        feature_types=feature_types, discrete_flip_prob=discrete_flip_prob,
    )

    # Evaluate each perturbation
    per_perturbation_scores = []

    for i in range(n_perturbations):
        x_prime = perturbed[i]

        # Same-class filter
        try:
            pred_class = _get_predicted_class(model, x_prime)
        except Exception:
            continue
        if pred_class != original_class:
            continue

        # Get perturbed explanation and representation
        try:
            perturbed_attr = _get_explanation_vector(
                explainer, x_prime, n_features
            )
            repr_pert = np.asarray(
                representation_fn(x_prime), dtype=np.float64
            ).flatten()
        except Exception:
            continue

        # Numerator: ||(e_x - e_x') / e_x||_p
        numerator_vec = _element_wise_percent_change(
            original_attr, perturbed_attr, epsilon_min
        )
        numerator = np.linalg.norm(numerator_vec, ord=norm_ord)

        # Denominator: max(||(L_x - L_x') / L_x||_p, epsilon_min)
        denom_vec = _element_wise_percent_change(
            repr_orig, repr_pert, epsilon_min
        )
        denom = max(np.linalg.norm(denom_vec, ord=norm_ord), epsilon_min)

        ratio = float(numerator / denom)
        per_perturbation_scores.append(ratio)

    n_valid = len(per_perturbation_scores)

    # Warn if too few valid perturbations for statistically reliable results
    if 0 < n_valid < 5:
        warnings.warn(
            f"Only {n_valid}/{n_perturbations} perturbations passed the "
            f"same-class filter. RRS score may be statistically unreliable. "
            f"Consider increasing n_perturbations or noise_scale.",
            stacklevel=2,
        )

    agg_result = _aggregate_perturbation_scores(
        per_perturbation_scores, aggregation
    )

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
) -> Dict[str, Union[float, list]]:
    """
    Compute Relative Representation Stability over a batch of instances.

    Args:
        explainer: Explainer instance.
        model: Model adapter.
        X: Input data (2D array).
        representation_fn: Model representation extractor.
        n_perturbations: Perturbations per instance.
        noise_scale: Gaussian noise std.
        norm_ord: Norm order.
        epsilon_min: Division-by-zero floor.
        aggregation: Aggregation mode.
        feature_types: Feature type array.
        discrete_flip_prob: Flip probability for discrete features.
        max_instances: Maximum instances to evaluate.
        seed: Random seed.

    Returns:
        Dictionary with mean, std, max, min, scores, n_evaluated.
    """
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    if max_instances is not None:
        n = min(n, max_instances)

    scores = []
    for i in range(n):
        try:
            score = compute_relative_representation_stability(
                explainer, model, X[i],
                representation_fn=representation_fn,
                n_perturbations=n_perturbations,
                noise_scale=noise_scale, norm_ord=norm_ord,
                epsilon_min=epsilon_min, aggregation=aggregation,
                feature_types=feature_types,
                discrete_flip_prob=discrete_flip_prob,
                seed=seed + i if seed is not None else None,
            )
            if not np.isnan(score):
                scores.append(score)
        except Exception:
            continue

    if not scores:
        return {"mean": float("nan"), "std": 0.0, "max": float("nan"),
                "min": float("nan"), "scores": [], "n_evaluated": 0}

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "max": float(np.max(scores)),
        "min": float(np.min(scores)),
        "scores": scores,
        "n_evaluated": len(scores),
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

    A higher score indicates higher instability. Lower is better.

    Args:
        explainer: Explainer instance with .explain() method.
        model: Model adapter with predict() or predict_proba().
        instance: Input instance (1D array).
        logit_fn: Callable that maps input (1D or 2D array) to output
            logits (1D or 2D array). E.g., the pre-softmax layer output.
        n_perturbations: Number of perturbations. Default: 50.
        noise_scale: Gaussian noise std. Default: 0.05.
        norm_ord: Norm order. Default: 2.
        epsilon_min: Division-by-zero floor. Default: 1e-7.
        aggregation: "max" (default), "mean", or "median".
        feature_types: Feature type array ("continuous"/"discrete").
        discrete_flip_prob: Flip probability for discrete features.
        seed: Random seed.
        return_details: If True, return diagnostic dict.

    Returns:
        float or dict (see compute_relative_input_stability for dict format,
        excluding theoretical_bound).

    Reference:
        Agarwal et al. (2022). Equation 5.
    """
    instance = np.asarray(instance, dtype=np.float64).flatten()
    n_features = len(instance)
    rng = np.random.default_rng(seed)

    valid_aggs = {"max", "mean", "median"}
    if aggregation not in valid_aggs:
        raise ValueError(
            f"aggregation must be one of {valid_aggs}, got '{aggregation}'"
        )

    # Get original explanation, prediction, and logits
    original_attr = _get_explanation_vector(explainer, instance, n_features)
    original_class = _get_predicted_class(model, instance)
    logits_orig = np.asarray(
        logit_fn(instance), dtype=np.float64
    ).flatten()

    # Generate perturbations
    perturbed = _generate_mixed_perturbations(
        instance, n_perturbations, noise_scale, rng,
        feature_types=feature_types, discrete_flip_prob=discrete_flip_prob,
    )

    # Evaluate each perturbation
    per_perturbation_scores = []

    for i in range(n_perturbations):
        x_prime = perturbed[i]

        # Same-class filter
        try:
            pred_class = _get_predicted_class(model, x_prime)
        except Exception:
            continue
        if pred_class != original_class:
            continue

        # Get perturbed explanation and logits
        try:
            perturbed_attr = _get_explanation_vector(
                explainer, x_prime, n_features
            )
            logits_pert = np.asarray(
                logit_fn(x_prime), dtype=np.float64
            ).flatten()
        except Exception:
            continue

        # Numerator: ||(e_x - e_x') / e_x||_p  (element-wise percent change)
        numerator_vec = _element_wise_percent_change(
            original_attr, perturbed_attr, epsilon_min
        )
        numerator = np.linalg.norm(numerator_vec, ord=norm_ord)

        # Denominator: max(||h(x) - h(x')||_p, epsilon_min)
        # NOTE: Equation 5 uses ABSOLUTE difference, not percent change
        logit_diff = logits_orig - logits_pert
        denom = max(np.linalg.norm(logit_diff, ord=norm_ord), epsilon_min)

        ratio = float(numerator / denom)
        per_perturbation_scores.append(ratio)

    n_valid = len(per_perturbation_scores)

    # Warn if too few valid perturbations for statistically reliable results
    if 0 < n_valid < 5:
        warnings.warn(
            f"Only {n_valid}/{n_perturbations} perturbations passed the "
            f"same-class filter. ROS score may be statistically unreliable. "
            f"Consider increasing n_perturbations or noise_scale.",
            stacklevel=2,
        )

    agg_result = _aggregate_perturbation_scores(
        per_perturbation_scores, aggregation
    )

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
) -> Dict[str, Union[float, list]]:
    """
    Compute Relative Output Stability over a batch of instances.

    Args:
        explainer: Explainer instance.
        model: Model adapter.
        X: Input data (2D array).
        logit_fn: Model logit extractor.
        n_perturbations: Perturbations per instance.
        noise_scale: Gaussian noise std.
        norm_ord: Norm order.
        epsilon_min: Division-by-zero floor.
        aggregation: Aggregation mode.
        feature_types: Feature type array.
        discrete_flip_prob: Flip probability for discrete features.
        max_instances: Maximum instances to evaluate.
        seed: Random seed.

    Returns:
        Dictionary with mean, std, max, min, scores, n_evaluated.
    """
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    if max_instances is not None:
        n = min(n, max_instances)

    scores = []
    for i in range(n):
        try:
            score = compute_relative_output_stability(
                explainer, model, X[i],
                logit_fn=logit_fn,
                n_perturbations=n_perturbations,
                noise_scale=noise_scale, norm_ord=norm_ord,
                epsilon_min=epsilon_min, aggregation=aggregation,
                feature_types=feature_types,
                discrete_flip_prob=discrete_flip_prob,
                seed=seed + i if seed is not None else None,
            )
            if not np.isnan(score):
                scores.append(score)
        except Exception:
            continue

    if not scores:
        return {"mean": float("nan"), "std": 0.0, "max": float("nan"),
                "min": float("nan"), "scores": [], "n_evaluated": 0}

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "max": float(np.max(scores)),
        "min": float(np.min(scores)),
        "scores": scores,
        "n_evaluated": len(scores),
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
        noise_scale: Gaussian noise std. Default: 0.05.
        norm_ord: Norm order. Default: 2.
        epsilon_min: Division-by-zero floor. Default: 1e-7.
        aggregation: "max" (default), "mean", or "median".
        feature_types: Feature type array.
        discrete_flip_prob: Flip probability for discrete features.
        seed: Random seed.
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
    instance = np.asarray(instance, dtype=np.float64).flatten()
    n_features = len(instance)
    rng = np.random.default_rng(seed)

    valid_aggs = {"max", "mean", "median"}
    if aggregation not in valid_aggs:
        raise ValueError(
            f"aggregation must be one of {valid_aggs}, got '{aggregation}'"
        )

    # Get original explanation and prediction
    original_attr = _get_explanation_vector(explainer, instance, n_features)
    original_class = _get_predicted_class(model, instance)

    # Get original representation and logits if needed
    repr_orig = None
    if representation_fn is not None:
        repr_orig = np.asarray(
            representation_fn(instance), dtype=np.float64
        ).flatten()

    logits_orig = None
    if logit_fn is not None:
        logits_orig = np.asarray(
            logit_fn(instance), dtype=np.float64
        ).flatten()

    # Generate perturbations (shared across all metrics)
    perturbed = _generate_mixed_perturbations(
        instance, n_perturbations, noise_scale, rng,
        feature_types=feature_types, discrete_flip_prob=discrete_flip_prob,
    )

    # Collect per-perturbation scores for each metric
    ris_scores = []
    rrs_scores = [] if representation_fn is not None else None
    ros_scores = [] if logit_fn is not None else None

    for i in range(n_perturbations):
        x_prime = perturbed[i]

        # Same-class filter
        try:
            pred_class = _get_predicted_class(model, x_prime)
        except Exception:
            continue
        if pred_class != original_class:
            continue

        # Get perturbed explanation (shared)
        try:
            perturbed_attr = _get_explanation_vector(
                explainer, x_prime, n_features
            )
        except Exception:
            continue

        # Shared numerator: ||(e_x - e_x') / e_x||_p
        numerator_vec = _element_wise_percent_change(
            original_attr, perturbed_attr, epsilon_min
        )
        numerator = np.linalg.norm(numerator_vec, ord=norm_ord)

        # RIS denominator: max(||(x - x') / x||_p, epsilon_min)
        ris_denom_vec = _element_wise_percent_change(
            instance, x_prime, epsilon_min
        )
        ris_denom = max(
            np.linalg.norm(ris_denom_vec, ord=norm_ord), epsilon_min
        )
        ris_scores.append(float(numerator / ris_denom))

        # RRS denominator: max(||(L_x - L_x') / L_x||_p, epsilon_min)
        if representation_fn is not None:
            try:
                repr_pert = np.asarray(
                    representation_fn(x_prime), dtype=np.float64
                ).flatten()
                rrs_denom_vec = _element_wise_percent_change(
                    repr_orig, repr_pert, epsilon_min
                )
                rrs_denom = max(
                    np.linalg.norm(rrs_denom_vec, ord=norm_ord), epsilon_min
                )
                rrs_scores.append(float(numerator / rrs_denom))
            except Exception:
                pass

        # ROS denominator: max(||h(x) - h(x')||_p, epsilon_min)
        if logit_fn is not None:
            try:
                logits_pert = np.asarray(
                    logit_fn(x_prime), dtype=np.float64
                ).flatten()
                logit_diff = logits_orig - logits_pert
                ros_denom = max(
                    np.linalg.norm(logit_diff, ord=norm_ord), epsilon_min
                )
                ros_scores.append(float(numerator / ros_denom))
            except Exception:
                pass

    # Aggregate
    ris_agg = _aggregate_perturbation_scores(ris_scores, aggregation)
    rrs_agg = (
        _aggregate_perturbation_scores(rrs_scores, aggregation)
        if rrs_scores is not None else None
    )
    ros_agg = (
        _aggregate_perturbation_scores(ros_scores, aggregation)
        if ros_scores is not None else None
    )

    n_valid = len(ris_scores)

    # Warn if too few valid perturbations for statistically reliable results
    if 0 < n_valid < 5:
        warnings.warn(
            f"Only {n_valid}/{n_perturbations} perturbations passed the "
            f"same-class filter. Relative stability scores may be "
            f"statistically unreliable. Consider increasing "
            f"n_perturbations or noise_scale.",
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
            }

        return {
            "ris": _build_detail(ris_agg, n_valid, n_perturbations),
            "rrs": _build_detail(
                rrs_agg,
                len(rrs_scores) if rrs_scores is not None else 0,
                n_perturbations,
            ),
            "ros": _build_detail(
                ros_agg,
                len(ros_scores) if ros_scores is not None else 0,
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
        noise_scale: Gaussian noise std.
        norm_ord: Norm order.
        epsilon_min: Division-by-zero floor.
        aggregation: Aggregation mode.
        feature_types: Feature type array.
        discrete_flip_prob: Flip probability for discrete features.
        max_instances: Maximum instances to evaluate.
        seed: Random seed.

    Returns:
        Dict with keys "ris", "rrs", "ros". Each is a dict with
        mean, std, max, min, scores, n_evaluated (or None if not computed).
    """
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    if max_instances is not None:
        n = min(n, max_instances)

    ris_scores = []
    rrs_scores = [] if representation_fn is not None else None
    ros_scores = [] if logit_fn is not None else None

    for i in range(n):
        try:
            result = compute_relative_stability(
                explainer, model, X[i],
                representation_fn=representation_fn,
                logit_fn=logit_fn,
                n_perturbations=n_perturbations,
                noise_scale=noise_scale, norm_ord=norm_ord,
                epsilon_min=epsilon_min, aggregation=aggregation,
                feature_types=feature_types,
                discrete_flip_prob=discrete_flip_prob,
                seed=seed + i if seed is not None else None,
            )
            if not np.isnan(result["ris"]):
                ris_scores.append(result["ris"])
            if rrs_scores is not None and result["rrs"] is not None:
                if not np.isnan(result["rrs"]):
                    rrs_scores.append(result["rrs"])
            if ros_scores is not None and result["ros"] is not None:
                if not np.isnan(result["ros"]):
                    ros_scores.append(result["ros"])
        except Exception:
            continue

    def _batch_stats(scores_list):
        if scores_list is None:
            return None
        if not scores_list:
            return {"mean": float("nan"), "std": 0.0, "max": float("nan"),
                    "min": float("nan"), "scores": [], "n_evaluated": 0}
        return {
            "mean": float(np.mean(scores_list)),
            "std": float(np.std(scores_list)),
            "max": float(np.max(scores_list)),
            "min": float(np.min(scores_list)),
            "scores": scores_list,
            "n_evaluated": len(scores_list),
        }

    return {
        "ris": _batch_stats(ris_scores),
        "rrs": _batch_stats(rrs_scores),
        "ros": _batch_stats(ros_scores),
    }


# =============================================================================
# Consistency (Dasgupta et al., 2022 — ICML)
# =============================================================================

def _get_top_k_features(
    attribution_vector: np.ndarray,
    k: int,
) -> frozenset:
    """
    Extract the indices of the top-k features by absolute attribution magnitude.

    Returns a frozenset for hashable set comparison.

    Args:
        attribution_vector: 1D array of attribution values.
        k: Number of top features to select.

    Returns:
        frozenset of integer indices.
    """
    k = min(k, len(attribution_vector))
    indices = np.argsort(np.abs(attribution_vector))[::-1][:k]
    return frozenset(indices.tolist())


def _get_model_prediction(
    model,
    instance: np.ndarray,
) -> int:
    """
    Get the predicted class label for a single instance.

    Handles both predict() and predict_proba() adapters.

    Args:
        model: Model adapter with predict or predict_proba method.
        instance: 1D input array.

    Returns:
        Integer predicted class label.
    """
    instance_2d = instance.reshape(1, -1)

    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(instance_2d)
        if isinstance(proba, np.ndarray) and proba.ndim >= 1:
            if proba.ndim == 2:
                return int(np.argmax(proba[0]))
            else:
                return int(np.argmax(proba))

    if hasattr(model, 'predict'):
        pred = model.predict(instance_2d)
        if isinstance(pred, np.ndarray):
            pred = pred.flatten()[0]
        return int(pred)

    raise ValueError(
        "Model must have a predict() or predict_proba() method."
    )


def compute_consistency(
    explainer: BaseExplainer,
    model,
    X: np.ndarray,
    top_k: int = 3,
    max_pairs: int = None,
    seed: int = None,
) -> float:
    """
    Compute Consistency of an explanation method (Dasgupta et al., 2022).

    Consistency measures the probability that two inputs receiving the
    **same explanation** (identical top-k important features) also receive
    the **same prediction**:

        Consistency = P( f(x) = f(x') | E(x) = E(x') )

    where E(x) = E(x') means the top-k features by absolute attribution
    magnitude are the same set, and f(x) = f(x') means the predicted
    class labels match.

    A higher score indicates more faithful explanations — the explanation
    captures the model's decision boundary well. A score of 1.0 means
    that whenever two instances have the same important features, the
    model always predicts the same class for them.

    The metric is operationalised by:
      1. Computing explanations for all instances in X.
      2. Discretising each explanation to its top-k feature set.
      3. Grouping instances that share the same top-k set.
      4. For every pair within each group, checking prediction agreement.
      5. Returning the fraction of agreeing pairs across all groups.

    Args:
        explainer: Explainer instance with .explain() method.
        model: Model adapter with predict() or predict_proba().
        X: Input data (2D array of shape (n_instances, n_features)).
            Should contain enough instances for meaningful pair comparisons.
        top_k: Number of top features to define the discrete explanation.
            Default: 3. Should be < n_features. Smaller k gives stricter
            explanation equivalence (fewer matching pairs but higher
            precision); larger k gives looser matching.
        max_pairs: Maximum total pairs to evaluate (None = all pairs).
            Useful for large datasets where the number of pairs can
            grow quadratically. If set, pairs are sampled uniformly.
        seed: Random seed for pair sampling (only used when max_pairs
            is set).

    Returns:
        Consistency score (float) in [0, 1]. Higher = more consistent.
        Returns NaN if no matching explanation pairs are found.

    Example:
        >>> from explainiverse.evaluation import compute_consistency
        >>> score = compute_consistency(explainer, model, X_test, top_k=3)
        >>> print(f"Consistency: {score:.4f}")

    Reference:
        Dasgupta, S., Frost, N., & Moshkovitz, M. (2022). Framework for
        Evaluating Faithfulness of Local Explanations. ICML.
        https://proceedings.mlr.press/v162/dasgupta22a.html
    """
    X = np.asarray(X, dtype=np.float64)
    n_instances, n_features = X.shape

    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if top_k >= n_features:
        raise ValueError(
            f"top_k ({top_k}) must be < n_features ({n_features}). "
            "With top_k == n_features, all explanations would trivially "
            "match and the metric becomes uninformative."
        )
    if n_instances < 2:
        return float("nan")

    rng = np.random.default_rng(seed)

    # Step 1: Compute explanations and predictions for all instances
    top_k_sets = []   # frozenset per instance
    predictions = []  # predicted class per instance
    valid_indices = []

    for i in range(n_instances):
        try:
            attr = _get_explanation_vector(explainer, X[i], n_features)
            pred = _get_model_prediction(model, X[i])
            top_k_sets.append(_get_top_k_features(attr, top_k))
            predictions.append(pred)
            valid_indices.append(i)
        except Exception:
            continue

    if len(valid_indices) < 2:
        return float("nan")

    # Step 2: Group instances by their top-k explanation set
    groups = {}  # frozenset -> list of indices into valid_indices
    for idx, topk_set in enumerate(top_k_sets):
        if topk_set not in groups:
            groups[topk_set] = []
        groups[topk_set].append(idx)

    # Step 3: Evaluate prediction agreement within each group
    agree_count = 0
    total_pairs = 0

    # Collect all within-group pairs
    all_pairs = []
    for group_indices in groups.values():
        if len(group_indices) < 2:
            continue
        for i, j in combinations(group_indices, 2):
            all_pairs.append((i, j))

    if not all_pairs:
        return float("nan")

    # Optionally subsample pairs
    if max_pairs is not None and len(all_pairs) > max_pairs:
        pair_indices = rng.choice(
            len(all_pairs), size=max_pairs, replace=False
        )
        all_pairs = [all_pairs[p] for p in pair_indices]

    for i, j in all_pairs:
        if predictions[i] == predictions[j]:
            agree_count += 1
        total_pairs += 1

    return float(agree_count / total_pairs)


def compute_batch_consistency(
    explainer: BaseExplainer,
    model,
    X: np.ndarray,
    top_k_values: List[int] = None,
    max_pairs: int = None,
    seed: int = None,
) -> Dict[str, Union[float, Dict[int, float]]]:
    """
    Compute Consistency across multiple top-k values.

    Evaluating consistency at different levels of explanation granularity
    (different k values) gives a more complete picture of how well the
    explanation captures the model's decision boundary.

    Args:
        explainer: Explainer instance.
        model: Model adapter.
        X: Input data (2D array).
        top_k_values: List of top-k values to evaluate.
            Default: [1, 2, 3, 5] (if n_features > 5).
        max_pairs: Maximum pairs per top-k evaluation.
        seed: Random seed.

    Returns:
        Dictionary with:
            - "scores": Dict mapping each k to its consistency score
            - "mean": Mean consistency across all k values
            - "top_k_values": List of k values evaluated
            - "n_instances": Number of instances in X
    """
    X = np.asarray(X)
    n_features = X.shape[1]

    if top_k_values is None:
        top_k_values = [k for k in [1, 2, 3, 5] if k < n_features]
    else:
        top_k_values = [k for k in top_k_values if k < n_features]

    if not top_k_values:
        return {
            "scores": {},
            "mean": float("nan"),
            "top_k_values": [],
            "n_instances": len(X),
        }

    scores = {}
    for k in top_k_values:
        scores[k] = compute_consistency(
            explainer, model, X, top_k=k,
            max_pairs=max_pairs,
            seed=seed,
        )

    valid_scores = [s for s in scores.values() if not np.isnan(s)]
    mean_score = float(np.mean(valid_scores)) if valid_scores else float("nan")

    return {
        "scores": scores,
        "mean": mean_score,
        "top_k_values": top_k_values,
        "n_instances": len(X),
    }


# =============================================================================
# Avg-Sensitivity (Yeh et al., 2019)
# =============================================================================

def compute_avg_sensitivity(
    explainer: BaseExplainer,
    instance: np.ndarray,
    radius: float = 0.1,
    n_samples: int = 50,
    norm_ord: Union[int, float, str] = 2,
    perturb_norm: str = "l2",
    normalize: bool = True,
    seed: int = None,
) -> float:
    """
    Compute Avg-Sensitivity of an explanation method.

    Avg-Sensitivity measures the expected change in explanation when the
    input is uniformly perturbed within a ball of radius r:

        AvgSens(E, x, r) = E_{||δ||_p ≤ r} [ ||E(x + δ) - E(x)||_q ]

    Unlike Max-Sensitivity, this captures the typical (rather than
    worst-case) robustness behaviour. A lower score is better.

    Args:
        explainer: Explainer instance with .explain() method.
        instance: Input instance (1D array).
        radius: Radius of the perturbation ball. Default: 0.1.
        n_samples: Number of Monte Carlo samples. Default: 50.
        norm_ord: Norm order for explanation differences.
        perturb_norm: Perturbation ball norm ("l2" or "linf").
        normalize: If True, normalize by original explanation norm.
        seed: Random seed.

    Returns:
        Avg-Sensitivity score (float). Lower = more robust.

    Reference:
        Yeh et al. (2019). On the (In)fidelity and Sensitivity of
        Explanations. NeurIPS.
    """
    instance = np.asarray(instance, dtype=np.float64).flatten()
    n_features = len(instance)
    rng = np.random.default_rng(seed)

    # Get original explanation
    original_attr = _get_explanation_vector(explainer, instance, n_features)
    original_norm = np.linalg.norm(original_attr, ord=norm_ord)

    if normalize and original_norm < 1e-12:
        return 0.0

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
        try:
            perturbed_attr = _get_explanation_vector(
                explainer, perturbed[i], n_features
            )
            diff = np.linalg.norm(original_attr - perturbed_attr, ord=norm_ord)
            diffs.append(diff)
        except Exception:
            continue

    if not diffs:
        return float("nan")

    mean_diff = np.mean(diffs)

    if normalize and original_norm > 1e-12:
        return float(mean_diff / original_norm)
    return float(mean_diff)


def compute_batch_avg_sensitivity(
    explainer: BaseExplainer,
    X: np.ndarray,
    radius: float = 0.1,
    n_samples: int = 50,
    norm_ord: Union[int, float, str] = 2,
    perturb_norm: str = "l2",
    normalize: bool = True,
    max_instances: int = None,
    seed: int = None,
) -> Dict[str, float]:
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
        seed: Random seed.

    Returns:
        Dictionary with mean, std, max, min, scores, n_evaluated.
    """
    X = np.asarray(X)
    n = len(X)
    if max_instances is not None:
        n = min(n, max_instances)

    scores = []
    for i in range(n):
        try:
            score = compute_avg_sensitivity(
                explainer, X[i], radius=radius, n_samples=n_samples,
                norm_ord=norm_ord, perturb_norm=perturb_norm,
                normalize=normalize,
                seed=seed + i if seed is not None else None,
            )
            if not np.isnan(score):
                scores.append(score)
        except Exception:
            continue

    if not scores:
        return {"mean": float("nan"), "std": 0.0, "max": float("nan"),
                "min": float("nan"), "scores": [], "n_evaluated": 0}

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "max": float(np.max(scores)),
        "min": float(np.min(scores)),
        "scores": scores,
        "n_evaluated": len(scores),
    }


# =============================================================================
# Continuity (Montavon et al., 2018; Alvarez-Melis & Jaakkola, 2018)
# =============================================================================

def compute_continuity(
    explainer: BaseExplainer,
    instance: np.ndarray,
    X_reference: np.ndarray,
    k_neighbors: int = 5,
    norm_ord: Union[int, float, str] = 2,
    input_distance: str = "euclidean",
    seed: int = None,
) -> float:
    """
    Compute Continuity of an explanation method.

    Continuity measures whether nearby inputs receive similar explanations.
    It computes the Spearman rank correlation between input-space distances
    and explanation-space distances for the k nearest neighbours of the
    query instance:

        Continuity(E, x, X) = SpearmanCorr(
            [ ||x − x_i|| ]_{i=1}^k,
            [ ||E(x) − E(x_i)|| ]_{i=1}^k
        )

    A score of +1 indicates perfect continuity: as inputs get closer, their
    explanations also get proportionally closer.  A score near 0 or negative
    indicates that explanation distances do not track input distances.

    Args:
        explainer: Explainer instance with .explain() method.
        instance: Query instance (1D array).
        X_reference: Reference dataset to find neighbours in (2D array).
            Typically the training or validation set.
        k_neighbors: Number of nearest neighbours. Default: 5.
            Must be ≥ 3 for a meaningful rank correlation.
        norm_ord: Norm order for explanation-space distances.
        input_distance: Distance metric for input space.
            Any metric accepted by scipy.spatial.distance.cdist.
            Default: "euclidean".
        seed: Random seed (unused, reserved for future stochastic variants).

    Returns:
        Spearman rank correlation coefficient (float) in [-1, 1].
        Higher = more continuous explanations.
        Returns NaN if fewer than 3 valid neighbours are found.

    Reference:
        Montavon, G., Samek, W., & Müller, K. R. (2018). Methods for
        interpreting and understanding deep neural networks.

        Alvarez-Melis, D., & Jaakkola, T. S. (2018). On the Robustness of
        Interpretability Methods.
    """
    instance = np.asarray(instance, dtype=np.float64).flatten()
    X_reference = np.asarray(X_reference, dtype=np.float64)
    n_features = len(instance)

    if len(X_reference) < k_neighbors:
        k_neighbors = len(X_reference)

    if k_neighbors < 3:
        return float("nan")

    # Find k nearest neighbours by input distance
    input_dists = cdist(
        instance.reshape(1, -1), X_reference, metric=input_distance
    ).flatten()

    # Exclude the instance itself if it appears in the reference set
    # (distance ≈ 0)
    neighbor_indices = np.argsort(input_dists)
    selected_indices = []
    for idx in neighbor_indices:
        if input_dists[idx] > 1e-12:
            selected_indices.append(idx)
        if len(selected_indices) == k_neighbors:
            break

    if len(selected_indices) < 3:
        return float("nan")

    # Get original explanation
    original_attr = _get_explanation_vector(explainer, instance, n_features)

    # Compute input and explanation distances for each neighbour
    d_input = []
    d_explanation = []

    for idx in selected_indices:
        neighbor = X_reference[idx]
        try:
            neighbor_attr = _get_explanation_vector(explainer, neighbor, n_features)
        except Exception:
            continue

        d_input.append(input_dists[idx])
        d_explanation.append(
            np.linalg.norm(original_attr - neighbor_attr, ord=norm_ord)
        )

    if len(d_input) < 3:
        return float("nan")

    # Spearman rank correlation
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)
        correlation, _ = stats.spearmanr(d_input, d_explanation)

    if np.isnan(correlation):
        # Can happen if all distances are identical
        return 0.0

    return float(correlation)


def compute_batch_continuity(
    explainer: BaseExplainer,
    X: np.ndarray,
    k_neighbors: int = 5,
    norm_ord: Union[int, float, str] = 2,
    input_distance: str = "euclidean",
    max_instances: int = None,
    seed: int = None,
) -> Dict[str, float]:
    """
    Compute Continuity over a batch of instances.

    Each instance is evaluated against the remaining instances in X as the
    reference set (leave-one-out).

    Args:
        explainer: Explainer instance.
        X: Input data (2D array).
        k_neighbors: Number of nearest neighbours per instance.
        norm_ord: Norm order for explanation distances.
        input_distance: Input-space distance metric.
        max_instances: Maximum instances to evaluate.
        seed: Random seed.

    Returns:
        Dictionary with mean, std, max, min, scores, n_evaluated.
    """
    X = np.asarray(X)
    n = len(X)
    if max_instances is not None:
        n = min(n, max_instances)

    scores = []
    for i in range(n):
        try:
            # Use all other instances as reference
            reference = np.delete(X, i, axis=0)
            score = compute_continuity(
                explainer, X[i], reference,
                k_neighbors=k_neighbors, norm_ord=norm_ord,
                input_distance=input_distance, seed=seed,
            )
            if not np.isnan(score):
                scores.append(score)
        except Exception:
            continue

    if not scores:
        return {"mean": float("nan"), "std": 0.0, "max": float("nan"),
                "min": float("nan"), "scores": [], "n_evaluated": 0}

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "max": float(np.max(scores)),
        "min": float(np.min(scores)),
        "scores": scores,
        "n_evaluated": len(scores),
    }
