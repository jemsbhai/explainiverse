# src/explainiverse/evaluation/complexity.py
"""
Complexity evaluation metrics for explanations (Phase 4).

Implements:
- Sparseness (Chalasani et al., 2020) — Gini Index of absolute attributions
- Complexity (Bhatt et al., 2020) — Shannon entropy of fractional contributions
- Effective Complexity (Nguyen & Martínez, 2020) — threshold-based feature count

These metrics evaluate how concise and interpretable an explanation is.
Sparser, lower-entropy, and fewer-feature explanations are generally
considered more human-interpretable.

References:
    Chalasani, P., Chen, J., Chowdhury, A. R., Wu, X., & Jha, S. (2020).
    Concise Explanations of Neural Networks using Adversarial Training.
    ICML.

    Bhatt, U., Weller, A., & Moura, J. M. F. (2020). Evaluating and
    Aggregating Feature-based Model Explanations. IJCAI.
    https://arxiv.org/abs/2005.00631

    Nguyen, A. P., & Martínez, M. R. (2020). On Quantitative Aspects
    of Model Interpretability. arXiv:2007.07584.
"""
import numpy as np
from typing import Union, Dict, Optional

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


def _compute_gini_index(values: np.ndarray) -> float:
    """
    Compute the Gini index of a 1D array of non-negative values.

    Uses the efficient sorted-values formula:

        G = (2 * Σᵢ (i+1) * x_sorted[i]) / (n * Σx) - (n+1) / n

    where x_sorted is sorted in ascending order.

    Properties:
        - G = 0 when all values are equal (perfect equality)
        - G → 1 when all value is concentrated on one element
        - G = (n-1)/n for a single non-zero element among n
        - Scale-invariant: G(c*x) = G(x) for c > 0
        - Permutation-invariant

    Args:
        values: 1D numpy array of non-negative values.

    Returns:
        Gini index in [0, 1]. Returns 0.0 if sum is zero or n <= 1.
    """
    n = len(values)
    if n <= 1:
        return 0.0

    total = np.sum(values)
    if total < 1e-300:
        return 0.0

    sorted_vals = np.sort(values)
    # Indices 1..n (1-based) for the sorted formula
    indices = np.arange(1, n + 1, dtype=np.float64)
    gini = (2.0 * np.sum(indices * sorted_vals)) / (n * total) - (n + 1.0) / n

    return float(gini)


def _compute_entropy(values: np.ndarray) -> float:
    """
    Compute the Shannon entropy (base 2) of a 1D probability distribution.

    This is a reusable utility for computing entropy of attribution
    distributions. Used internally by compute_complexity() and available
    for future metrics (e.g., Efficient MPRT).

    Args:
        values: 1D array of non-negative values. Will be normalized to
                sum to 1 (i.e., treated as unnormalized probabilities).

    Returns:
        Shannon entropy in bits. Range: [0, log2(n)] where n is the
        number of non-zero elements. Returns 0.0 if sum is zero or
        only one non-zero element exists.
    """
    total = np.sum(values)
    if total < 1e-300:
        return 0.0

    # Normalize to probability distribution
    p = values / total

    # Filter out zeros to avoid log(0)
    p_nonzero = p[p > 0]

    if len(p_nonzero) <= 1:
        return 0.0

    entropy = -np.sum(p_nonzero * np.log2(p_nonzero))
    return float(entropy)


# =============================================================================
# Sparseness (Chalasani et al., 2020)
# =============================================================================

def compute_sparseness(
    explainer: BaseExplainer,
    instance: np.ndarray,
) -> float:
    """
    Compute Sparseness of an explanation using the Gini Index.

    Sparseness measures what fraction of features carry meaningful
    attribution weight. It uses the Gini index of the absolute
    attribution values:

        Sparseness(E, x) = Gini(|E(x)|)

    A higher score indicates a sparser (more concentrated) explanation,
    which is generally considered more interpretable. A score of 0
    means all features have equal attribution (least sparse).

    Properties:
        - Range: [0, 1] (0 = uniform, approaches 1 = maximally sparse)
        - Perfectly sparse (1 of n features): (n-1)/n
        - Scale-invariant: independent of attribution magnitude
        - Permutation-invariant: independent of feature ordering

    Args:
        explainer: Explainer instance with .explain() method.
        instance: Input instance (1D array of shape (n_features,)).

    Returns:
        Sparseness score (float) in [0, 1]. Higher = sparser.
        Returns 0.0 for all-zero or single-feature attributions.

    Example:
        >>> from explainiverse.evaluation import compute_sparseness
        >>> score = compute_sparseness(explainer, instance)
        >>> print(f"Sparseness (Gini): {score:.4f}")

    Reference:
        Chalasani, P., Chen, J., Chowdhury, A. R., Wu, X., & Jha, S.
        (2020). Concise Explanations of Neural Networks using Adversarial
        Training. ICML.
    """
    instance = np.asarray(instance, dtype=np.float64).flatten()
    n_features = len(instance)

    attr = _get_explanation_vector(explainer, instance, n_features)
    abs_attr = np.abs(attr)

    return _compute_gini_index(abs_attr)


def compute_batch_sparseness(
    explainer: BaseExplainer,
    X: np.ndarray,
    max_instances: int = None,
) -> Dict[str, object]:
    """
    Compute Sparseness over a batch of instances.

    Args:
        explainer: Explainer instance.
        X: Input data (2D array of shape (n_instances, n_features)).
        max_instances: Maximum number of instances to evaluate (None = all).

    Returns:
        Dictionary with:
            - "mean": Mean Sparseness across instances
            - "std": Standard deviation
            - "max": Maximum Sparseness
            - "min": Minimum Sparseness
            - "scores": List of per-instance scores
            - "n_evaluated": Number of instances evaluated

    Example:
        >>> from explainiverse.evaluation import compute_batch_sparseness
        >>> result = compute_batch_sparseness(explainer, X_test)
        >>> print(f"Mean Sparseness: {result['mean']:.4f}")

    Reference:
        Chalasani et al. (2020). Concise Explanations of Neural Networks
        using Adversarial Training. ICML.
    """
    X = np.asarray(X)
    n = len(X)
    if max_instances is not None:
        n = min(n, max_instances)

    scores = []
    for i in range(n):
        try:
            score = compute_sparseness(explainer, X[i])
            scores.append(score)
        except Exception:
            continue

    if not scores:
        return {
            "mean": float("nan"), "std": 0.0, "max": float("nan"),
            "min": float("nan"), "scores": [], "n_evaluated": 0,
        }

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "max": float(np.max(scores)),
        "min": float(np.min(scores)),
        "scores": scores,
        "n_evaluated": len(scores),
    }


# =============================================================================
# Effective Complexity (Nguyen & Martínez, 2020)
# =============================================================================

def compute_effective_complexity(
    explainer: BaseExplainer,
    instance: np.ndarray,
    threshold: float = 1e-5,
    threshold_type: str = "absolute",
    normalize: bool = False,
) -> float:
    """
    Compute Effective Complexity of an explanation.

    Effective Complexity counts the number of features whose absolute
    attribution exceeds a relevance threshold ε:

        EC(E, x, ε) = |{ i : |a_i| > ε }|

    Fewer features above the threshold means a simpler, more focused
    explanation. This metric complements Sparseness (Gini) and
    Complexity (entropy) by providing a direct, interpretable count
    of "active" features.

    Supports two threshold modes:
        - "absolute": feature counts if |a_i| > threshold
        - "relative": feature counts if |a_i| > threshold * max(|a|)

    Properties:
        - Range: [0, n] (unnormalized) or [0, 1] (normalized)
        - EC = 0 when all attributions are below threshold
        - EC = n when all attributions exceed threshold
        - Lower is simpler (fewer relevant features)
        - Monotonically non-increasing in threshold

    Args:
        explainer: Explainer instance with .explain() method.
        instance: Input instance (1D array of shape (n_features,)).
        threshold: Relevance threshold. Default: 1e-5.
            For "absolute": features with |a_i| > threshold are counted.
            For "relative": features with |a_i| > threshold * max(|a|)
            are counted. Typical relative values: 0.01 to 0.1.
        threshold_type: "absolute" or "relative". Default: "absolute".
        normalize: If True, return EC / n (fraction in [0, 1]).
            Default: False (return raw count).

    Returns:
        Effective Complexity score (float).
        Unnormalized: integer-valued float in [0, n].
        Normalized: float in [0, 1].
        Returns 0.0 for all-zero attributions.

    Raises:
        ValueError: If threshold_type is not "absolute" or "relative".

    Example:
        >>> from explainiverse.evaluation import compute_effective_complexity
        >>> # Absolute threshold
        >>> ec = compute_effective_complexity(explainer, instance, threshold=0.01)
        >>> # Relative threshold (1% of max attribution)
        >>> ec = compute_effective_complexity(
        ...     explainer, instance, threshold=0.01, threshold_type="relative"
        ... )
        >>> # Normalized to [0, 1]
        >>> ec_norm = compute_effective_complexity(
        ...     explainer, instance, normalize=True
        ... )

    Reference:
        Nguyen, A. P., & Martínez, M. R. (2020). On Quantitative Aspects
        of Model Interpretability. arXiv:2007.07584.
    """
    if threshold_type not in ("absolute", "relative"):
        raise ValueError(
            f"threshold_type must be 'absolute' or 'relative', "
            f"got '{threshold_type}'"
        )

    instance = np.asarray(instance, dtype=np.float64).flatten()
    n_features = len(instance)

    attr = _get_explanation_vector(explainer, instance, n_features)
    abs_attr = np.abs(attr)

    # Compute effective threshold
    if threshold_type == "relative":
        max_attr = np.max(abs_attr)
        if max_attr < 1e-300:
            # All attributions are effectively zero
            return 0.0
        effective_threshold = threshold * max_attr
    else:
        effective_threshold = threshold

    # Count features exceeding threshold
    count = int(np.sum(abs_attr > effective_threshold))

    if normalize:
        return float(count) / float(n_features) if n_features > 0 else 0.0
    return float(count)


def compute_batch_effective_complexity(
    explainer: BaseExplainer,
    X: np.ndarray,
    threshold: float = 1e-5,
    threshold_type: str = "absolute",
    normalize: bool = False,
    max_instances: int = None,
) -> Dict[str, object]:
    """
    Compute Effective Complexity over a batch of instances.

    Args:
        explainer: Explainer instance.
        X: Input data (2D array of shape (n_instances, n_features)).
        threshold: Relevance threshold.
        threshold_type: "absolute" or "relative".
        normalize: If True, return EC / n per instance.
        max_instances: Maximum number of instances to evaluate (None = all).

    Returns:
        Dictionary with:
            - "mean": Mean Effective Complexity across instances
            - "std": Standard deviation
            - "max": Maximum Effective Complexity
            - "min": Minimum Effective Complexity
            - "scores": List of per-instance scores
            - "n_evaluated": Number of instances evaluated

    Example:
        >>> from explainiverse.evaluation import compute_batch_effective_complexity
        >>> result = compute_batch_effective_complexity(
        ...     explainer, X_test, threshold=0.01, threshold_type="relative"
        ... )
        >>> print(f"Mean EC: {result['mean']:.2f} features")

    Reference:
        Nguyen & Martínez (2020). On Quantitative Aspects of Model
        Interpretability. arXiv:2007.07584.
    """
    X = np.asarray(X)
    n = len(X)
    if max_instances is not None:
        n = min(n, max_instances)

    scores = []
    for i in range(n):
        try:
            score = compute_effective_complexity(
                explainer, X[i],
                threshold=threshold,
                threshold_type=threshold_type,
                normalize=normalize,
            )
            scores.append(score)
        except Exception:
            continue

    if not scores:
        return {
            "mean": float("nan"), "std": 0.0, "max": float("nan"),
            "min": float("nan"), "scores": [], "n_evaluated": 0,
        }

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "max": float(np.max(scores)),
        "min": float(np.min(scores)),
        "scores": scores,
        "n_evaluated": len(scores),
    }


# =============================================================================
# Complexity (Bhatt et al., 2020)
# =============================================================================

def compute_complexity(
    explainer: BaseExplainer,
    instance: np.ndarray,
) -> float:
    """
    Compute Complexity of an explanation using Shannon entropy.

    Complexity measures the entropy of the fractional contribution
    distribution over features:

        p_i = |a_i| / \u03a3|a_j|
        Complexity(E, x) = H(p) = -\u03a3 p_i \u00b7 log\u2082(p_i)

    A lower score indicates a simpler (more concentrated) explanation.
    A higher score means attribution is spread across many features,
    making the explanation harder for humans to interpret.

    Properties:
        - Range: [0, log\u2082(n)] where n is the number of features
        - H = 0 when all weight is on one feature (simplest)
        - H = log\u2082(n) when weight is uniform (most complex)
        - Scale-invariant: independent of attribution magnitude
        - Lower is better (simpler explanation)

    Args:
        explainer: Explainer instance with .explain() method.
        instance: Input instance (1D array of shape (n_features,)).

    Returns:
        Complexity score (float) in [0, log\u2082(n)]. Lower = simpler.
        Returns 0.0 for all-zero or single-feature attributions.

    Example:
        >>> from explainiverse.evaluation import compute_complexity
        >>> score = compute_complexity(explainer, instance)
        >>> print(f"Complexity (entropy): {score:.4f} bits")

    Reference:
        Bhatt, U., Weller, A., & Moura, J. M. F. (2020). Evaluating and
        Aggregating Feature-based Model Explanations. IJCAI.
        https://arxiv.org/abs/2005.00631
    """
    instance = np.asarray(instance, dtype=np.float64).flatten()
    n_features = len(instance)

    attr = _get_explanation_vector(explainer, instance, n_features)
    abs_attr = np.abs(attr)

    return _compute_entropy(abs_attr)


def compute_batch_complexity(
    explainer: BaseExplainer,
    X: np.ndarray,
    max_instances: int = None,
) -> Dict[str, object]:
    """
    Compute Complexity over a batch of instances.

    Args:
        explainer: Explainer instance.
        X: Input data (2D array of shape (n_instances, n_features)).
        max_instances: Maximum number of instances to evaluate (None = all).

    Returns:
        Dictionary with:
            - "mean": Mean Complexity across instances
            - "std": Standard deviation
            - "max": Maximum Complexity
            - "min": Minimum Complexity
            - "scores": List of per-instance scores
            - "n_evaluated": Number of instances evaluated

    Example:
        >>> from explainiverse.evaluation import compute_batch_complexity
        >>> result = compute_batch_complexity(explainer, X_test)
        >>> print(f"Mean Complexity: {result['mean']:.4f} bits")

    Reference:
        Bhatt et al. (2020). Evaluating and Aggregating Feature-based
        Model Explanations. IJCAI.
    """
    X = np.asarray(X)
    n = len(X)
    if max_instances is not None:
        n = min(n, max_instances)

    scores = []
    for i in range(n):
        try:
            score = compute_complexity(explainer, X[i])
            scores.append(score)
        except Exception:
            continue

    if not scores:
        return {
            "mean": float("nan"), "std": 0.0, "max": float("nan"),
            "min": float("nan"), "scores": [], "n_evaluated": 0,
        }

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "max": float(np.max(scores)),
        "min": float(np.min(scores)),
        "scores": scores,
        "n_evaluated": len(scores),
    }
