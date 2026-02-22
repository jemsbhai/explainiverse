# src/explainiverse/evaluation/agreement.py
"""
Pairwise agreement metrics for comparing explanations (Phase 2).

Implements:
- Feature Agreement (Krishna et al., 2022) — top-k feature set overlap
- Rank Agreement (Krishna et al., 2022) — top-k rank position match

These metrics quantify the extent to which two different explanation methods
agree on which features are most important (Feature Agreement) and whether
they rank them in the same order (Rank Agreement). They address the
"Disagreement Problem" in explainable ML, where different post-hoc methods
may produce conflicting explanations for the same model prediction.

Both metrics are computed per-instance and operate on the top-k features
by absolute attribution value. Higher values indicate stronger agreement
(range [0, 1]).

Note:
    Unlike most other evaluation metrics in Explainiverse, these are
    *pairwise* metrics that compare two attribution vectors directly.
    They do not require access to the underlying model.

Reference:
    Krishna, S., Han, T., Gu, A., Pombra, J., Jabbari, S., Wu, S.,
    & Lakkaraju, H. (2022). The Disagreement Problem in Explainable
    Machine Learning: A Practitioner's Perspective. Transactions on
    Machine Learning Research (TMLR).
    https://arxiv.org/abs/2202.01602
"""
import warnings
import numpy as np
from typing import Union, List, Optional

from explainiverse.core.explanation import Explanation


# =============================================================================
# Internal Helpers
# =============================================================================

def _extract_attribution_array(
    attributions: Union[np.ndarray, "Explanation"],
) -> np.ndarray:
    """
    Extract a 1-D numpy attribution vector from various input types.

    Accepts:
        - 1-D numpy array directly
        - Explanation object (extracts feature_attributions values)

    Args:
        attributions: Attribution values or Explanation object.

    Returns:
        1-D numpy array of float64 attribution values.

    Raises:
        TypeError: If input is not a supported type.
        ValueError: If attributions are empty.
    """
    if isinstance(attributions, Explanation):
        attr_dict = attributions.explanation_data.get("feature_attributions", {})
        if not attr_dict:
            raise ValueError("No feature attributions found in Explanation.")
        feature_names = getattr(attributions, "feature_names", None)
        if feature_names:
            values = [attr_dict.get(fn, 0.0) for fn in feature_names]
        else:
            values = list(attr_dict.values())
        return np.array(values, dtype=np.float64)

    if isinstance(attributions, np.ndarray):
        return attributions.astype(np.float64).ravel()

    raise TypeError(
        f"Expected np.ndarray or Explanation, got {type(attributions).__name__}"
    )


def _validate_pair(
    attr_a: np.ndarray,
    attr_b: np.ndarray,
    k: int,
) -> None:
    """
    Validate a pair of attribution arrays and the k parameter.

    Args:
        attr_a: First attribution vector.
        attr_b: Second attribution vector.
        k: Number of top features.

    Raises:
        ValueError: If shapes mismatch, arrays are empty, or k is invalid.
    """
    if attr_a.ndim != 1 or attr_b.ndim != 1:
        raise ValueError(
            f"Attributions must be 1-D arrays, got shapes "
            f"{attr_a.shape} and {attr_b.shape}."
        )
    if attr_a.shape[0] != attr_b.shape[0]:
        raise ValueError(
            f"Attribution vectors must have the same length, got "
            f"{attr_a.shape[0]} and {attr_b.shape[0]}."
        )
    n = attr_a.shape[0]
    if n == 0:
        raise ValueError("Attribution vectors must not be empty.")
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k}.")
    if k > n:
        raise ValueError(
            f"k={k} exceeds number of features n={n}."
        )


def _top_k_indices(attr: np.ndarray, k: int) -> np.ndarray:
    """
    Return indices of top-k features by absolute attribution value.

    Ties are broken by the order returned from np.argsort (last
    occurrence for equal absolute values gets higher rank). The result
    is sorted from rank 1 (most important) to rank k.

    Args:
        attr: 1-D attribution vector.
        k: Number of top features.

    Returns:
        1-D array of length k with feature indices, ordered from
        most important (rank 1) to k-th most important (rank k).
    """
    # argsort ascending, take last k, reverse to descending
    order = np.argsort(np.abs(attr))
    return order[-k:][::-1]


# =============================================================================
# Feature Agreement
# =============================================================================

def compute_feature_agreement(
    attributions_a: Union[np.ndarray, "Explanation"],
    attributions_b: Union[np.ndarray, "Explanation"],
    k: int = 5,
) -> float:
    """
    Feature Agreement (Krishna et al., 2022).

    Measures the fraction of top-k features (by absolute attribution) that
    are shared between two explanations for the same instance:

        FA(e_a, e_b, k) = |top_k(e_a) ∩ top_k(e_b)| / k

    A value of 1.0 means both explanations identify exactly the same
    top-k features; 0.0 means no overlap at all.

    Args:
        attributions_a: Attribution vector from method A (1-D array or
            Explanation object).
        attributions_b: Attribution vector from method B (1-D array or
            Explanation object).
        k: Number of top features to compare (default 5). Must satisfy
            1 ≤ k ≤ n_features.

    Returns:
        Float in [0, 1]. Higher = more agreement.

    Raises:
        ValueError: If shapes mismatch, k is invalid, or arrays are empty.
        TypeError: If inputs are not np.ndarray or Explanation.

    Example:
        >>> import numpy as np
        >>> a = np.array([0.5, 0.1, 0.3, 0.8, 0.2])
        >>> b = np.array([0.4, 0.05, 0.6, 0.7, 0.15])
        >>> compute_feature_agreement(a, b, k=2)
        1.0  # Both identify features 3 and 0 as top-2

    Reference:
        Krishna, S., Han, T., Gu, A., Pombra, J., Jabbari, S., Wu, S.,
        & Lakkaraju, H. (2022). The Disagreement Problem in Explainable
        Machine Learning: A Practitioner's Perspective. TMLR.
    """
    attr_a = _extract_attribution_array(attributions_a)
    attr_b = _extract_attribution_array(attributions_b)
    _validate_pair(attr_a, attr_b, k)

    top_a = set(_top_k_indices(attr_a, k))
    top_b = set(_top_k_indices(attr_b, k))

    return len(top_a & top_b) / k


def compute_batch_feature_agreement(
    attributions_a_batch: List[Union[np.ndarray, "Explanation"]],
    attributions_b_batch: List[Union[np.ndarray, "Explanation"]],
    k: int = 5,
) -> List[float]:
    """
    Batch Feature Agreement (Krishna et al., 2022).

    Computes Feature Agreement for each corresponding pair of
    attributions across two batches.

    Args:
        attributions_a_batch: List of attribution vectors from method A.
        attributions_b_batch: List of attribution vectors from method B.
        k: Number of top features to compare (default 5).

    Returns:
        List of float scores, one per instance.

    Raises:
        ValueError: If batch sizes differ, or any individual pair fails
            validation.
    """
    if len(attributions_a_batch) != len(attributions_b_batch):
        raise ValueError(
            f"Batch sizes must match: got {len(attributions_a_batch)} "
            f"and {len(attributions_b_batch)}."
        )
    return [
        compute_feature_agreement(a, b, k=k)
        for a, b in zip(attributions_a_batch, attributions_b_batch)
    ]


# =============================================================================
# Rank Agreement
# =============================================================================

def compute_rank_agreement(
    attributions_a: Union[np.ndarray, "Explanation"],
    attributions_b: Union[np.ndarray, "Explanation"],
    k: int = 5,
) -> float:
    """
    Rank Agreement (Krishna et al., 2022).

    Measures the fraction of top-k rank positions where both explanations
    place the same feature. For each rank position i ∈ {1, …, k}, check
    whether the feature ranked i-th by method A is also ranked i-th by
    method B:

        RA(e_a, e_b, k) = (1/k) Σ_{i=1}^{k} 𝟙[top_k(e_a)[i] = top_k(e_b)[i]]

    This is strictly more demanding than Feature Agreement: not only must
    the same features be in the top-k set, they must occupy the same rank
    positions. In particular, RA ≤ FA always holds.

    Args:
        attributions_a: Attribution vector from method A (1-D array or
            Explanation object).
        attributions_b: Attribution vector from method B (1-D array or
            Explanation object).
        k: Number of top features to compare (default 5). Must satisfy
            1 ≤ k ≤ n_features.

    Returns:
        Float in [0, 1]. Higher = more agreement.

    Raises:
        ValueError: If shapes mismatch, k is invalid, or arrays are empty.
        TypeError: If inputs are not np.ndarray or Explanation.

    Example:
        >>> import numpy as np
        >>> a = np.array([0.5, 0.1, 0.3, 0.8, 0.2])
        >>> b = np.array([0.4, 0.05, 0.6, 0.7, 0.15])
        >>> compute_rank_agreement(a, b, k=2)
        0.5  # Both rank feature 3 first, but disagree on rank 2

    Reference:
        Krishna, S., Han, T., Gu, A., Pombra, J., Jabbari, S., Wu, S.,
        & Lakkaraju, H. (2022). The Disagreement Problem in Explainable
        Machine Learning: A Practitioner's Perspective. TMLR.
    """
    attr_a = _extract_attribution_array(attributions_a)
    attr_b = _extract_attribution_array(attributions_b)
    _validate_pair(attr_a, attr_b, k)

    ranking_a = _top_k_indices(attr_a, k)
    ranking_b = _top_k_indices(attr_b, k)

    # Count rank positions where both place the same feature
    matches = np.sum(ranking_a == ranking_b)

    return float(matches) / k


def compute_batch_rank_agreement(
    attributions_a_batch: List[Union[np.ndarray, "Explanation"]],
    attributions_b_batch: List[Union[np.ndarray, "Explanation"]],
    k: int = 5,
) -> List[float]:
    """
    Batch Rank Agreement (Krishna et al., 2022).

    Computes Rank Agreement for each corresponding pair of
    attributions across two batches.

    Args:
        attributions_a_batch: List of attribution vectors from method A.
        attributions_b_batch: List of attribution vectors from method B.
        k: Number of top features to compare (default 5).

    Returns:
        List of float scores, one per instance.

    Raises:
        ValueError: If batch sizes differ, or any individual pair fails
            validation.
    """
    if len(attributions_a_batch) != len(attributions_b_batch):
        raise ValueError(
            f"Batch sizes must match: got {len(attributions_a_batch)} "
            f"and {len(attributions_b_batch)}."
        )
    return [
        compute_rank_agreement(a, b, k=k)
        for a, b in zip(attributions_a_batch, attributions_b_batch)
    ]
