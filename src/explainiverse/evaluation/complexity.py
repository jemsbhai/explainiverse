# src/explainiverse/evaluation/complexity.py
"""
Attribution concentration and support-size diagnostics.

Implements:
- Sparseness (Chalasani et al., 2020) — Gini Index of absolute attributions
- Complexity (Bhatt et al., 2020) — Shannon entropy of fractional contributions
- Attribution Threshold Count — number of attribution magnitudes above a threshold

The historical ``compute_effective_complexity`` name is retained as a
compatibility alias for Attribution Threshold Count. It is not Nguyen &
Martínez's Effective Complexity, which requires measuring model performance
while conditioning on successively larger top-feature sets.

These metrics summarize attribution concentration, inequality, or thresholded
support size. They do not by themselves establish human interpretability.

References:
    Chalasani, P., Chen, J., Chowdhury, A. R., Wu, X., & Jha, S. (2020).
    Concise Explanations of Neural Networks using Adversarial Training.
    ICML.

    Bhatt, U., Weller, A., & Moura, J. M. F. (2020). Evaluating and
    Aggregating Feature-based Model Explanations. IJCAI.
    https://arxiv.org/abs/2005.00631

    Nguyen, A. P., & Martínez, M. R. (2020). On Quantitative Aspects
    of Model Interpretability. arXiv:2007.07584. This source is cited to
    distinguish its model-conditional Effective Complexity from the simpler
    attribution threshold count retained here for compatibility.
"""
import warnings
from typing import Dict, Optional

import numpy as np

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation

# =============================================================================
# Internal Helpers
# =============================================================================


def _extract_attribution_vector(
    explanation: Explanation,
    expected_n_features: Optional[int] = None,
) -> np.ndarray:
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

    feature_names = getattr(explanation, "feature_names", None)
    if feature_names:
        if len(feature_names) != len(set(feature_names)):
            raise ValueError("Explanation feature_names must be unique.")
        missing = [name for name in feature_names if name not in attributions]
        unexpected = [name for name in attributions if name not in feature_names]
        if missing or unexpected:
            raise ValueError(
                "feature_attributions must match feature_names exactly; "
                f"missing={missing}, unexpected={unexpected}."
            )
        values = [attributions[name] for name in feature_names]
    else:
        values = list(attributions.values())

    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError("Feature attributions must be a non-empty one-dimensional vector.")
    if expected_n_features is not None and result.size != expected_n_features:
        raise ValueError(
            f"Explanation returned {result.size} attributions for an input with "
            f"{expected_n_features} features."
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("Feature attributions must contain only finite values.")
    return result


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
    return _extract_attribution_vector(exp, expected_n_features=n_features)


def _validate_instance(instance: np.ndarray) -> np.ndarray:
    """Validate a single tabular attribution-metric input."""
    result = np.asarray(instance, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError("instance must be a non-empty one-dimensional feature vector.")
    if not np.all(np.isfinite(result)):
        raise ValueError("instance must contain only finite values.")
    return result


def _validate_batch(X: np.ndarray, max_instances: Optional[int]) -> tuple[np.ndarray, int]:
    """Validate batch shape and return the number of rows to evaluate."""
    result = np.asarray(X)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError("X must be a non-empty two-dimensional feature matrix.")
    if not np.all(np.isfinite(result)):
        raise ValueError("X must contain only finite values.")
    if max_instances is not None:
        if (
            not isinstance(max_instances, (int, np.integer))
            or isinstance(max_instances, (bool, np.bool_))
            or max_instances < 1
        ):
            raise ValueError("max_instances must be a positive integer or None.")
        return result, min(result.shape[0], int(max_instances))
    return result, result.shape[0]


def _summarise_scores(scores: list[float]) -> Dict[str, object]:
    """Return the shared deterministic batch summary."""
    values = np.asarray(scores, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
        "scores": scores,
        "n_evaluated": len(scores),
    }


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
    if total == 0.0:
        return 0.0

    sorted_vals = np.sort(values)
    # Indices 1..n (1-based) for the sorted formula
    indices = np.arange(1, n + 1, dtype=np.float64)
    gini = (2.0 * np.sum(indices * sorted_vals)) / (n * total) - (n + 1.0) / n

    return float(gini)


def _compute_entropy(values: np.ndarray) -> float:
    """
    Compute natural-log Shannon entropy of a 1D probability distribution.

    This is a reusable utility for computing entropy of attribution
    distributions. Used internally by compute_complexity() and available
    for future metrics (e.g., Efficient MPRT).

    Args:
        values: 1D array of non-negative values. Will be normalized to
                sum to 1 (i.e., treated as unnormalized probabilities).

    Returns:
        Shannon entropy in nats. Range: [0, ln(n)] where n is the
        number of non-zero elements. Returns 0.0 if sum is zero or
        only one non-zero element exists.
    """
    total = np.sum(values)
    if total == 0.0:
        raise ValueError(
            "Complexity is undefined for an all-zero attribution vector because "
            "the fractional contribution distribution cannot be formed."
        )

    # Normalize to probability distribution
    p = values / total

    # Filter out zeros to avoid log(0)
    p_nonzero = p[p > 0]

    if len(p_nonzero) <= 1:
        return 0.0

    entropy = -np.sum(p_nonzero * np.log(p_nonzero))
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

    Sparseness measures concentration across feature-attribution magnitudes.
    It uses the Gini index of the absolute attribution values:

        Sparseness(E, x) = Gini(|E(x)|)

    A higher score indicates a sparser (more concentrated) attribution vector.
    A score of 0 means all features have equal attribution magnitude. No human-
    interpretability conclusion follows from this statistic alone.

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
    instance = _validate_instance(instance)
    n_features = len(instance)

    attr = _get_explanation_vector(explainer, instance, n_features)
    abs_attr = np.abs(attr)

    return _compute_gini_index(abs_attr)


def compute_batch_sparseness(
    explainer: BaseExplainer,
    X: np.ndarray,
    max_instances: Optional[int] = None,
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
    X, n = _validate_batch(X, max_instances)

    scores = []
    for i in range(n):
        scores.append(compute_sparseness(explainer, X[i]))

    return _summarise_scores(scores)


# =============================================================================
# Attribution Threshold Count (historically mislabeled Effective Complexity)
# =============================================================================


def compute_attribution_threshold_count(
    explainer: BaseExplainer,
    instance: np.ndarray,
    threshold: float = 1e-5,
    threshold_type: str = "absolute",
    normalize: bool = False,
) -> float:
    """
    Count attribution magnitudes above a configured threshold.

    Attribution Threshold Count counts the number of features whose absolute
    attribution exceeds a relevance threshold ε:

        EC(E, x, ε) = |{ i : |a_i| > ε }|

    Fewer features above the threshold means a simpler, more focused
    explanation. This metric complements Sparseness (Gini) and
    Complexity (entropy) by providing a direct thresholded count
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
        Attribution Threshold Count score (float).
        Unnormalized: integer-valued float in [0, n].
        Normalized: float in [0, 1].
        Returns 0.0 for all-zero attributions.

    Raises:
        ValueError: If threshold_type is not "absolute" or "relative".

    Example:
        >>> from explainiverse.evaluation import compute_attribution_threshold_count
        >>> # Absolute threshold
        >>> count = compute_attribution_threshold_count(explainer, instance, threshold=0.01)
        >>> # Relative threshold (1% of max attribution)
        >>> count = compute_attribution_threshold_count(
        ...     explainer, instance, threshold=0.01, threshold_type="relative"
        ... )
        >>> # Normalized to [0, 1]
        >>> count_norm = compute_attribution_threshold_count(
        ...     explainer, instance, normalize=True
        ... )

    Note:
        This threshold statistic is not Effective Complexity as defined by
        Nguyen & Martínez (2020), which depends on conditional model-loss
        evaluations for successively larger top-feature sets.
    """
    if threshold_type not in ("absolute", "relative"):
        raise ValueError(
            f"threshold_type must be 'absolute' or 'relative', " f"got '{threshold_type}'"
        )

    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be a finite non-negative number.")
    if not isinstance(normalize, (bool, np.bool_)):
        raise TypeError("normalize must be boolean.")

    instance = _validate_instance(instance)
    n_features = len(instance)

    attr = _get_explanation_vector(explainer, instance, n_features)
    abs_attr = np.abs(attr)

    # Compute effective threshold
    if threshold_type == "relative":
        max_attr = np.max(abs_attr)
        if max_attr == 0.0:
            # All attributions are exactly zero.
            return 0.0
        effective_threshold = threshold * max_attr
    else:
        effective_threshold = threshold

    # Count features exceeding threshold
    count = int(np.sum(abs_attr > effective_threshold))

    if normalize:
        return float(count) / float(n_features) if n_features > 0 else 0.0
    return float(count)


def compute_effective_complexity(
    explainer: BaseExplainer,
    instance: np.ndarray,
    threshold: float = 1e-5,
    threshold_type: str = "absolute",
    normalize: bool = False,
) -> float:
    """Compatibility alias for :func:`compute_attribution_threshold_count`.

    This historical name does not implement Nguyen & Martínez's
    model-conditional Effective Complexity. Call the accurately named function
    for new code.
    """
    warnings.warn(
        "compute_effective_complexity computes Attribution Threshold Count, not "
        "Nguyen & Martínez's model-conditional Effective Complexity; use "
        "compute_attribution_threshold_count for this statistic.",
        FutureWarning,
        stacklevel=2,
    )
    return compute_attribution_threshold_count(
        explainer,
        instance,
        threshold=threshold,
        threshold_type=threshold_type,
        normalize=normalize,
    )


def compute_batch_attribution_threshold_count(
    explainer: BaseExplainer,
    X: np.ndarray,
    threshold: float = 1e-5,
    threshold_type: str = "absolute",
    normalize: bool = False,
    max_instances: Optional[int] = None,
) -> Dict[str, object]:
    """
    Compute Attribution Threshold Count over a batch of instances.

    Args:
        explainer: Explainer instance.
        X: Input data (2D array of shape (n_instances, n_features)).
        threshold: Relevance threshold.
        threshold_type: "absolute" or "relative".
        normalize: If True, return EC / n per instance.
        max_instances: Maximum number of instances to evaluate (None = all).

    Returns:
        Dictionary with:
            - "mean": Mean threshold count across instances
            - "std": Standard deviation
            - "max": Maximum threshold count
            - "min": Minimum threshold count
            - "scores": List of per-instance scores
            - "n_evaluated": Number of instances evaluated

    Example:
        >>> from explainiverse.evaluation import compute_batch_attribution_threshold_count
        >>> result = compute_batch_attribution_threshold_count(
        ...     explainer, X_test, threshold=0.01, threshold_type="relative"
        ... )
        >>> print(f"Mean threshold count: {result['mean']:.2f} features")

    Note:
        This is not Nguyen & Martínez's model-conditional Effective Complexity.
    """
    X, n = _validate_batch(X, max_instances)

    scores = []
    for i in range(n):
        scores.append(
            compute_attribution_threshold_count(
                explainer,
                X[i],
                threshold=threshold,
                threshold_type=threshold_type,
                normalize=normalize,
            )
        )

    return _summarise_scores(scores)


def compute_batch_effective_complexity(
    explainer: BaseExplainer,
    X: np.ndarray,
    threshold: float = 1e-5,
    threshold_type: str = "absolute",
    normalize: bool = False,
    max_instances: Optional[int] = None,
) -> Dict[str, object]:
    """Compatibility alias for batch Attribution Threshold Count."""
    warnings.warn(
        "compute_batch_effective_complexity computes Attribution Threshold Count, "
        "not Nguyen & Martínez's model-conditional Effective Complexity; use "
        "compute_batch_attribution_threshold_count for this statistic.",
        FutureWarning,
        stacklevel=2,
    )
    return compute_batch_attribution_threshold_count(
        explainer,
        X,
        threshold=threshold,
        threshold_type=threshold_type,
        normalize=normalize,
        max_instances=max_instances,
    )


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

        p_i = |a_i| / sum_j |a_j|
        Complexity(E, x) = H(p) = -sum_i p_i * ln(p_i)

    A lower score means attribution magnitude is more concentrated; a higher
    score means it is more dispersed. The statistic does not determine human
    interpretability.

    Properties:
        - Range: [0, ln(n)] where n is the number of features
        - H = 0 when all magnitude is on one feature
        - H = ln(n) when magnitude is uniform
        - Scale-invariant: independent of attribution magnitude

    Args:
        explainer: Explainer instance with .explain() method.
        instance: Input instance (1D array of shape (n_features,)).

    Returns:
        Complexity score (float) in [0, ln(n)] nats. Lower is more concentrated.
        All-zero attributions raise because no fractional-contribution
        probability distribution exists.

    Example:
        >>> from explainiverse.evaluation import compute_complexity
        >>> score = compute_complexity(explainer, instance)
        >>> print(f"Complexity (entropy): {score:.4f} nats")

    Reference:
        Bhatt, U., Weller, A., & Moura, J. M. F. (2020). Evaluating and
        Aggregating Feature-based Model Explanations. IJCAI.
        https://arxiv.org/abs/2005.00631
    """
    instance = _validate_instance(instance)
    n_features = len(instance)

    attr = _get_explanation_vector(explainer, instance, n_features)
    abs_attr = np.abs(attr)

    return _compute_entropy(abs_attr)


def compute_batch_complexity(
    explainer: BaseExplainer,
    X: np.ndarray,
    max_instances: Optional[int] = None,
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
        >>> print(f"Mean Complexity: {result['mean']:.4f} nats")

    Reference:
        Bhatt et al. (2020). Evaluating and Aggregating Feature-based
        Model Explanations. IJCAI.
    """
    X, n = _validate_batch(X, max_instances)

    scores = []
    for i in range(n):
        scores.append(compute_complexity(explainer, X[i]))

    return _summarise_scores(scores)
