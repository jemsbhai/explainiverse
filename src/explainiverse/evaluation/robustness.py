# src/explainiverse/evaluation/robustness.py
"""
Robustness evaluation metrics for explanations (Phase 2).

Implements:
- Max-Sensitivity (Yeh et al., 2019)
- Avg-Sensitivity (Yeh et al., 2019)
- Continuity (Montavon et al., 2018; Alvarez-Melis & Jaakkola, 2018)

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
"""
import numpy as np
from typing import Union, Callable, List, Dict, Optional, Tuple
from scipy import stats
from scipy.spatial.distance import cdist

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
    import warnings
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
