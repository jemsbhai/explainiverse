# src/explainiverse/evaluation/fairness.py
"""
Fairness evaluation metrics for explanations (Phase 7).

Implements 6 fairness metrics for evaluating whether post hoc explanations
exhibit group-based or individual-based disparities:

1. Group Fairness (Dai et al., 2022) — composable disparity across demographic groups
2. Individual Fairness (Dwork et al., 2012; adapted) — similar individuals get
   similar explanations regardless of protected attribute
3. Counterfactual Explanation Fairness (Kusner et al., 2017; adapted) — explanations
   should not change when only the protected attribute is flipped
4. Fidelity Disparity (Balagopalan et al., 2022) — max/mean explanation quality
   gaps across subgroup pairs
5. Attribution Parity (novel synthesis) — whether the protected feature itself
   receives disproportionate attribution across groups
6. Conditional Fairness (Hardt et al., 2016; adapted) — explanation quality
   equality conditioned on model prediction

Also provides FairnessMetricRegistry for extensibility — users can register
custom fairness metrics with the same pattern used by the ExplainerRegistry.

References:
    Dai, J., Upadhyay, S., Aïvodji, U., Bach, S. H., & Lakkaraju, H. (2022).
    Fairness via Explanation Quality: Evaluating Disparities in the Quality
    of Post hoc Explanations. AIES. https://doi.org/10.1145/3514094.3534159

    Balagopalan, A., Zhang, H., Hamidieh, K., Hartvigsen, T., Rudzicz, F., &
    Ghassemi, M. (2022). The Road to Explainability is Paved with Bias:
    Measuring the Fairness of Explanations. FAccT.
    https://doi.org/10.1145/3531146.3533179

    Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R. (2012).
    Fairness Through Awareness. ITCS.

    Kusner, M. J., Loftus, J., Russell, C., & Silva, R. (2017).
    Counterfactual Fairness. NeurIPS.

    Hardt, M., Price, E., & Srebro, N. (2016). Equality of Opportunity in
    Supervised Learning. NeurIPS.

    Aïvodji, U., Arai, H., Fortineau, O., Gambs, S., Hara, S., & Tapp, A.
    (2019). Fairwashing: the risk of rationalization. ICML.
"""

import warnings
import numpy as np
from dataclasses import dataclass, field
from itertools import combinations
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from explainiverse.core.explanation import Explanation
from explainiverse.core.explainer import BaseExplainer


# =============================================================================
# Default Inner Metrics
# =============================================================================

def _default_inner_metric(attr_vector: np.ndarray) -> float:
    """
    Default per-instance explanation quality metric: L1 norm.

    This measures the total magnitude of the attribution vector, which
    serves as a simple proxy for explanation "intensity". Disparity in
    this value across groups indicates that some groups receive explanations
    with systematically different magnitudes.

    Args:
        attr_vector: 1D numpy array of attribution values for one instance.

    Returns:
        Scalar quality score for this instance.
    """
    return float(np.sum(np.abs(attr_vector)))


# =============================================================================
# Input Validation Helpers
# =============================================================================

def _validate_attributions(attributions: Any) -> np.ndarray:
    """
    Validate and convert attributions to a 2D numpy array.

    Args:
        attributions: Array-like of shape (n_samples, n_features).

    Returns:
        2D numpy float64 array.

    Raises:
        TypeError: If not array-like.
        ValueError: If wrong dimensionality or empty.
    """
    if isinstance(attributions, str):
        raise TypeError(
            f"attributions must be array-like, got {type(attributions).__name__}"
        )
    try:
        arr = np.asarray(attributions, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"attributions must be convertible to a numeric numpy array: {exc}"
        ) from exc

    if arr.ndim != 2:
        raise ValueError(
            f"attributions must be 2D (n_samples, n_features), got shape {arr.shape}"
        )
    if arr.shape[0] == 0:
        raise ValueError("attributions must not be empty (0 samples).")
    return arr


def _validate_sensitive_features(
    sensitive_features: Any,
    n_samples: int,
) -> np.ndarray:
    """
    Validate sensitive features array and coerce to group labels.

    Accepts int, float, or string arrays. Float arrays are cast to int
    when they contain whole numbers.

    Args:
        sensitive_features: 1D array-like of group labels.
        n_samples: Expected number of samples (must match).

    Returns:
        1D numpy array of group labels (dtype varies: int or str).

    Raises:
        ValueError: If length mismatch or empty.
    """
    sf = np.asarray(sensitive_features)
    if sf.ndim != 1:
        raise ValueError(
            f"sensitive_features must be 1D, got shape {sf.shape}"
        )
    if len(sf) != n_samples:
        raise ValueError(
            f"sensitive_features length ({len(sf)}) does not match "
            f"attributions rows ({n_samples}). Length mismatch."
        )
    if len(sf) == 0:
        raise ValueError("sensitive_features must not be empty.")

    # Coerce float -> int when possible (e.g. 0.0 -> 0)
    if sf.dtype.kind == 'f':
        if np.all(sf == sf.astype(int)):
            sf = sf.astype(int)
    return sf


def _partition_by_group(
    sensitive_features: np.ndarray,
) -> Dict[Any, np.ndarray]:
    """
    Partition sample indices by group label.

    Args:
        sensitive_features: 1D array of group labels.

    Returns:
        Dict mapping group_label -> array of indices belonging to that group.
    """
    groups: Dict[Any, List[int]] = {}
    for i, g in enumerate(sensitive_features):
        key = g.item() if hasattr(g, 'item') else g
        groups.setdefault(key, []).append(i)
    return {k: np.array(v) for k, v in groups.items()}


# =============================================================================
# Statistical Testing Helpers
# =============================================================================

def _mann_whitney_u(a: np.ndarray, b: np.ndarray) -> float:
    """
    Mann-Whitney U test p-value for two independent samples.

    Returns 1.0 if either sample has fewer than 1 element or all values
    are identical (no variation).

    Args:
        a: 1D array of metric values for group A.
        b: 1D array of metric values for group B.

    Returns:
        Two-sided p-value.
    """
    if len(a) < 1 or len(b) < 1:
        return 1.0
    # If both samples are constant and equal, no test needed
    if np.std(a) == 0 and np.std(b) == 0 and np.mean(a) == np.mean(b):
        return 1.0
    try:
        from scipy.stats import mannwhitneyu
        _, p = mannwhitneyu(a, b, alternative='two-sided')
        return float(p)
    except Exception:
        return 1.0


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cohen's d effect size between two samples.

    Uses pooled standard deviation. Returns 0.0 if both samples have
    zero variance.

    Args:
        a: 1D array of metric values for group A.
        b: 1D array of metric values for group B.

    Returns:
        Cohen's d (positive = group A has higher mean).
    """
    n_a, n_b = len(a), len(b)
    if n_a < 1 or n_b < 1:
        return 0.0
    mean_a, mean_b = np.mean(a), np.mean(b)
    var_a, var_b = np.var(a, ddof=1) if n_a > 1 else 0.0, np.var(b, ddof=1) if n_b > 1 else 0.0
    pooled_std = np.sqrt(
        ((n_a - 1) * var_a + (n_b - 1) * var_b)
        / max(n_a + n_b - 2, 1)
    )
    if pooled_std < 1e-15:
        return 0.0
    return float((mean_a - mean_b) / pooled_std)


# =============================================================================
# FairnessMetricMeta & FairnessMetricRegistry
# =============================================================================

@dataclass
class FairnessMetricMeta:
    """
    Metadata for a fairness metric, used for discovery and filtering.

    Attributes:
        level: Granularity — "group", "individual", or "conditional".
        composable: Whether the metric accepts a user-supplied inner metric.
        description: Human-readable description.
        paper_reference: Citation for the original paper.
    """
    level: str  # "group", "individual", "conditional"
    composable: bool = False
    description: str = ""
    paper_reference: Optional[str] = None

    def matches(self, level: Optional[str] = None) -> bool:
        """Check if this metadata matches the given filter criteria."""
        if level is not None and self.level != level:
            return False
        return True


class FairnessMetricRegistry:
    """
    Extensible registry for fairness evaluation metrics.

    Modelled on the ExplainerRegistry pattern. Allows registration of
    custom fairness metrics via programmatic or decorator-based APIs.

    Example:
        from explainiverse.evaluation.fairness import (
            FairnessMetricRegistry, FairnessMetricMeta
        )

        registry = FairnessMetricRegistry()

        @registry.register_decorator(
            name="my_metric",
            meta=FairnessMetricMeta(level="group", composable=False),
        )
        def my_custom_fairness(attributions, sensitive_features, **kwargs):
            ...
            return {"score": 0.42}

        result = registry.evaluate("my_metric", attributions, sensitive_features)
    """

    def __init__(self):
        self._registry: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        metric_fn: Callable,
        meta: FairnessMetricMeta,
        override: bool = False,
    ) -> None:
        """
        Register a fairness metric function with metadata.

        Args:
            name: Unique identifier (e.g. "group_fairness").
            metric_fn: Callable(attributions, sensitive_features, **kw) -> dict.
            meta: Metadata describing the metric.
            override: If True, allows overwriting an existing registration.

        Raises:
            ValueError: If name already registered and override=False.
        """
        if name in self._registry and not override:
            raise ValueError(
                f"Fairness metric '{name}' is already registered. "
                "Use override=True to replace."
            )
        self._registry[name] = {"fn": metric_fn, "meta": meta}

    def unregister(self, name: str) -> None:
        """
        Remove a fairness metric from the registry.

        Raises:
            KeyError: If the metric is not registered.
        """
        if name not in self._registry:
            raise KeyError(
                f"Fairness metric '{name}' is not registered."
            )
        del self._registry[name]

    def get(self, name: str) -> Dict[str, Any]:
        """
        Retrieve a registered metric entry by name.

        Returns:
            Dict with "fn" and "meta" keys.

        Raises:
            KeyError: If the metric is not registered.
        """
        if name not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(
                f"Fairness metric '{name}' is not registered. "
                f"Available: {available}"
            )
        return self._registry[name]

    def get_meta(self, name: str) -> FairnessMetricMeta:
        """Get just the metadata for a metric."""
        return self.get(name)["meta"]

    def list_metrics(self, with_meta: bool = False) -> Any:
        """
        List all registered fairness metrics.

        Args:
            with_meta: If True, return dict with metadata.

        Returns:
            List of names, or dict of {name: {"fn": ..., "meta": ...}}.
        """
        if with_meta:
            return dict(self._registry)
        return list(self._registry.keys())

    def filter(self, level: Optional[str] = None) -> List[str]:
        """
        Filter metrics by level (group, individual, conditional).

        Returns:
            List of matching metric names.
        """
        results = []
        for name, entry in self._registry.items():
            if entry["meta"].matches(level=level):
                results.append(name)
        return results

    def evaluate(
        self,
        name: str,
        attributions: np.ndarray,
        sensitive_features: np.ndarray,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Evaluate a registered fairness metric by name.

        Args:
            name: Metric name.
            attributions: 2D array (n_samples, n_features).
            sensitive_features: 1D array of group labels.
            **kwargs: Additional keyword arguments for the metric.

        Returns:
            Dict of metric results.
        """
        entry = self.get(name)
        return entry["fn"](attributions, sensitive_features, **kwargs)

    def register_decorator(
        self,
        name: str,
        meta: FairnessMetricMeta,
    ) -> Callable:
        """
        Decorator for registering a fairness metric function.

        Usage:
            @registry.register_decorator(
                name="my_metric",
                meta=FairnessMetricMeta(level="group"),
            )
            def my_metric(attributions, sensitive_features, **kwargs):
                return {"score": 0.0}
        """
        def decorator(fn: Callable) -> Callable:
            self.register(name, fn, meta)
            return fn
        return decorator

    def summary(self) -> str:
        """Generate a human-readable summary of all registered metrics."""
        lines = [
            "=" * 60,
            "Explainiverse — Registered Fairness Metrics",
            "=" * 60,
            "",
        ]
        for name, entry in self._registry.items():
            meta: FairnessMetricMeta = entry["meta"]
            desc = meta.description or "(no description)"
            lines.append(f"  {name} [{meta.level}]: {desc}")
        lines.append("")
        lines.append(f"Total: {len(self._registry)} metrics")
        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# Metric 1: Group Fairness (Dai et al., 2022)
# =============================================================================

def compute_group_fairness(
    attributions: Any,
    sensitive_features: Any,
    inner_metric: Optional[Callable[[np.ndarray], float]] = None,
) -> Dict[str, Any]:
    """
    Compute group-based explanation fairness disparity.

    Measures whether explanation quality differs across demographic groups
    by computing an inner quality metric per instance, averaging per group,
    and reporting the disparity (max absolute difference between group means),
    along with statistical tests.

    For binary groups this is the absolute difference of means; for multi-group
    it is the maximum pairwise absolute difference.

    Args:
        attributions: Array of shape (n_samples, n_features) — pre-computed
            attribution vectors.
        sensitive_features: 1D array of group labels (int, float, or str).
            Length must match attributions rows.
        inner_metric: Optional callable that takes a 1D attribution vector
            and returns a scalar quality score. Defaults to L1 norm.

    Returns:
        Dict with keys:
            - "disparity": float — maximum pairwise gap in group means.
            - "group_means": dict — {group_label: mean_score}.
            - "p_value": float — Mann-Whitney U p-value (binary groups) or
              minimum pairwise p-value (multi-group).
            - "effect_size": float — Cohen's d (binary) or max absolute
              Cohen's d (multi-group).

    References:
        Dai, J., et al. (2022). Fairness via Explanation Quality. AIES.
    """
    attrs = _validate_attributions(attributions)
    sf = _validate_sensitive_features(sensitive_features, attrs.shape[0])

    if inner_metric is None:
        inner_metric = _default_inner_metric

    # Compute per-instance quality scores
    scores = np.array([inner_metric(attrs[i]) for i in range(attrs.shape[0])])

    # Partition into groups
    groups = _partition_by_group(sf)

    if len(groups) < 2:
        # Single group — no disparity possible
        single_key = list(groups.keys())[0]
        return {
            "disparity": 0.0,
            "group_means": {single_key: float(np.mean(scores))},
            "p_value": 1.0,
            "effect_size": 0.0,
        }

    # Per-group means
    group_means = {}
    group_scores = {}
    for label, indices in groups.items():
        gs = scores[indices]
        group_means[label] = float(np.mean(gs))
        group_scores[label] = gs

    # Pairwise comparisons
    labels = list(groups.keys())
    max_disparity = 0.0
    min_p = 1.0
    max_effect = 0.0

    for la, lb in combinations(labels, 2):
        gap = abs(group_means[la] - group_means[lb])
        if gap > max_disparity:
            max_disparity = gap
        p = _mann_whitney_u(group_scores[la], group_scores[lb])
        if p < min_p:
            min_p = p
        d = abs(_cohens_d(group_scores[la], group_scores[lb]))
        if d > max_effect:
            max_effect = d

    return {
        "disparity": float(max_disparity),
        "group_means": group_means,
        "p_value": float(min_p),
        "effect_size": float(max_effect),
    }


def compute_group_fairness_score(
    explainer: BaseExplainer,
    inputs: np.ndarray,
    sensitive_features: Any,
    inner_metric: Optional[Callable[[np.ndarray], float]] = None,
) -> Dict[str, Any]:
    """
    Explainer-based API for group fairness: generates attributions, then evaluates.

    Args:
        explainer: An Explainiverse explainer instance.
        inputs: 2D array (n_samples, n_features).
        sensitive_features: 1D array of group labels.
        inner_metric: Optional inner quality metric.

    Returns:
        Same as compute_group_fairness().
    """
    from explainiverse.evaluation.axiomatic import (
        _extract_attribution_vector,
    )

    inputs = np.asarray(inputs, dtype=np.float64)
    n_features = inputs.shape[1]

    attr_rows = []
    for i in range(inputs.shape[0]):
        exp = explainer.explain(inputs[i])
        if not getattr(exp, 'feature_names', None):
            exp.feature_names = [f"feature_{j}" for j in range(n_features)]
        vec = _extract_attribution_vector(exp)
        attr_rows.append(vec)

    attributions = np.array(attr_rows, dtype=np.float64)
    return compute_group_fairness(attributions, sensitive_features, inner_metric)


def compute_batch_group_fairness(
    batch_attributions: List[np.ndarray],
    batch_sensitive_features: List[Any],
    inner_metric: Optional[Callable[[np.ndarray], float]] = None,
) -> List[Dict[str, Any]]:
    """
    Batch computation of group fairness across multiple attribution sets.

    Args:
        batch_attributions: List of 2D arrays.
        batch_sensitive_features: List of 1D group-label arrays.
        inner_metric: Optional inner quality metric.

    Returns:
        List of result dicts (one per batch element).
    """
    results = []
    for attrs, sf in zip(batch_attributions, batch_sensitive_features):
        results.append(compute_group_fairness(attrs, sf, inner_metric))
    return results


# =============================================================================
# Metric 2: Individual Fairness (Dwork et al., 2012; adapted)
# =============================================================================

def compute_individual_fairness(
    inputs: Any,
    attributions: Any,
    sensitive_features: Any,
    distance_threshold: Optional[float] = None,
    n_pairs: int = 500,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Compute individual fairness of explanations.

    Measures whether similar individuals from different groups receive similar
    explanations. For cross-group pairs within a feature-space distance
    threshold, computes the Lipschitz ratio:
        explanation_distance / feature_distance.

    A high ratio indicates unfairness: similar individuals are receiving
    very different explanations.

    Args:
        inputs: 2D array (n_samples, n_features) — original feature values.
        attributions: 2D array (n_samples, n_features) — attribution vectors.
        sensitive_features: 1D group-label array.
        distance_threshold: Maximum L2 feature distance for a pair to be
            considered "similar". If None, uses the 25th percentile of all
            cross-group distances.
        n_pairs: Maximum number of cross-group pairs to sample.
        random_state: Seed for reproducibility.

    Returns:
        Dict with keys:
            - "score": float — mean Lipschitz ratio across qualifying pairs.
              0 = perfectly fair, higher = more unfair.
            - "max_ratio": float — worst-case Lipschitz ratio.
            - "n_pairs_evaluated": int — how many cross-group pairs were used.

    References:
        Dwork, C., et al. (2012). Fairness Through Awareness. ITCS.
    """
    X = _validate_attributions(np.asarray(inputs, dtype=np.float64))
    A = _validate_attributions(np.asarray(attributions, dtype=np.float64))
    sf = _validate_sensitive_features(sensitive_features, X.shape[0])

    if X.shape != A.shape:
        raise ValueError(
            f"inputs shape {X.shape} does not match attributions shape {A.shape}."
        )

    groups = _partition_by_group(sf)
    labels = list(groups.keys())

    if len(labels) < 2:
        return {"score": 0.0, "max_ratio": 0.0, "n_pairs_evaluated": 0}

    # Build cross-group pairs
    rng = np.random.RandomState(random_state)
    cross_pairs: List[Tuple[int, int]] = []
    for la, lb in combinations(labels, 2):
        idx_a = groups[la]
        idx_b = groups[lb]
        for ia in idx_a:
            for ib in idx_b:
                cross_pairs.append((ia, ib))

    if len(cross_pairs) == 0:
        return {"score": 0.0, "max_ratio": 0.0, "n_pairs_evaluated": 0}

    # Subsample if too many pairs
    if len(cross_pairs) > n_pairs:
        indices = rng.choice(len(cross_pairs), size=n_pairs, replace=False)
        cross_pairs = [cross_pairs[i] for i in indices]

    # Compute distances
    feat_dists = np.array([
        np.linalg.norm(X[i] - X[j]) for i, j in cross_pairs
    ])
    attr_dists = np.array([
        np.linalg.norm(A[i] - A[j]) for i, j in cross_pairs
    ])

    # Determine threshold
    if distance_threshold is None:
        if len(feat_dists) > 0 and np.max(feat_dists) > 0:
            distance_threshold = float(np.percentile(feat_dists, 25))
        else:
            distance_threshold = 1e10  # accept all

    # Filter to "similar" pairs
    mask = feat_dists <= distance_threshold
    if not np.any(mask):
        # No similar cross-group pairs — relax to all pairs
        mask = np.ones(len(cross_pairs), dtype=bool)

    filtered_feat = feat_dists[mask]
    filtered_attr = attr_dists[mask]

    # Compute Lipschitz ratios
    eps = 1e-15
    ratios = filtered_attr / (filtered_feat + eps)

    return {
        "score": float(np.mean(ratios)),
        "max_ratio": float(np.max(ratios)),
        "n_pairs_evaluated": int(np.sum(mask)),
    }


# =============================================================================
# Metric 3: Counterfactual Explanation Fairness (Kusner et al., 2017; adapted)
# =============================================================================

def compute_counterfactual_fairness(
    inputs: Any,
    attributions: Any,
    sensitive_feature_idx: int,
    counterfactual_explainer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Compute counterfactual explanation fairness.

    Measures whether explanations change when the protected attribute is
    flipped (all else equal). If pre-computed attributions are constant
    (don't depend on sensitive feature), score is 0.

    When a ``counterfactual_explainer`` is provided, it is called for each
    instance's counterfactual (sensitive feature flipped) to obtain the
    counterfactual attribution. Otherwise, a simple within-dataset matching
    strategy is used: for each instance, find the nearest instance with
    the opposite sensitive-feature value and use its attribution.

    Args:
        inputs: 2D array (n_samples, n_features).
        attributions: 2D array (n_samples, n_features) — pre-computed.
        sensitive_feature_idx: Column index of the sensitive feature.
        counterfactual_explainer: Optional callable that takes a 1D instance
            (with sensitive feature already flipped) and returns a 1D
            attribution vector.

    Returns:
        Dict with keys:
            - "score": float — mean L2 distance between original and
              counterfactual attributions. Lower = fairer.
            - "per_instance_scores": list of float — per-instance distances.

    References:
        Kusner, M. J., et al. (2017). Counterfactual Fairness. NeurIPS.
    """
    X = np.asarray(inputs, dtype=np.float64)
    A = np.asarray(attributions, dtype=np.float64)

    if X.ndim != 2 or A.ndim != 2:
        raise ValueError("inputs and attributions must be 2D arrays.")
    if X.shape[0] != A.shape[0]:
        raise ValueError(
            f"inputs rows ({X.shape[0]}) != attributions rows ({A.shape[0]})."
        )
    if sensitive_feature_idx < 0 or sensitive_feature_idx >= X.shape[1]:
        raise ValueError(
            f"sensitive_feature_idx {sensitive_feature_idx} out of bounds "
            f"for {X.shape[1]} features."
        )

    per_instance = []

    if counterfactual_explainer is not None:
        # Use the provided explainer to get counterfactual attributions
        for i in range(X.shape[0]):
            cf_input = X[i].copy()
            # Flip the sensitive feature (binary: 0<->1)
            cf_input[sensitive_feature_idx] = 1.0 - cf_input[sensitive_feature_idx]
            cf_attr = counterfactual_explainer(cf_input)
            dist = float(np.linalg.norm(A[i] - cf_attr))
            per_instance.append(dist)
    else:
        # Matching strategy: for each instance, find nearest with opposite
        # sensitive value and compare attributions
        sens_vals = X[:, sensitive_feature_idx]
        unique_vals = np.unique(sens_vals)

        if len(unique_vals) < 2:
            # All same sensitive value — can't compute counterfactual
            return {
                "score": 0.0,
                "per_instance_scores": [0.0] * X.shape[0],
            }

        # Precompute feature distances excluding sensitive feature
        non_sens_cols = [c for c in range(X.shape[1]) if c != sensitive_feature_idx]
        X_non_sens = X[:, non_sens_cols]

        for i in range(X.shape[0]):
            my_val = sens_vals[i]
            # Find indices with opposite sensitive value
            opp_mask = sens_vals != my_val
            opp_indices = np.where(opp_mask)[0]

            if len(opp_indices) == 0:
                per_instance.append(0.0)
                continue

            # Find nearest neighbour in non-sensitive features
            dists = np.linalg.norm(
                X_non_sens[opp_indices] - X_non_sens[i], axis=1
            )
            nearest = opp_indices[np.argmin(dists)]
            dist = float(np.linalg.norm(A[i] - A[nearest]))
            per_instance.append(dist)

    return {
        "score": float(np.mean(per_instance)),
        "per_instance_scores": per_instance,
    }


# =============================================================================
# Metric 4: Fidelity Disparity (Balagopalan et al., 2022)
# =============================================================================

def compute_fidelity_disparity(
    attributions: Any,
    sensitive_features: Any,
    inner_metric: Optional[Callable[[np.ndarray], float]] = None,
) -> Dict[str, Any]:
    """
    Compute fidelity disparity: max and mean explanation-quality gaps.

    Extends Group Fairness to report pairwise gaps for all subgroup pairs,
    enabling multi-group worst-case analysis.

    Args:
        attributions: 2D array (n_samples, n_features).
        sensitive_features: 1D group-label array.
        inner_metric: Optional per-instance quality metric. Defaults to L1 norm.

    Returns:
        Dict with keys:
            - "max_gap": float — maximum pairwise gap across groups.
            - "mean_gap": float — mean pairwise gap across groups.
            - "pairwise_gaps": dict — {(g_a, g_b): gap} for all pairs.
            - "group_means": dict — {group: mean_score}.
            - "pairwise_p_values": dict — {(g_a, g_b): p_value}.

    References:
        Balagopalan, A., et al. (2022). The Road to Explainability is
        Paved with Bias. FAccT.
    """
    attrs = _validate_attributions(attributions)
    sf = _validate_sensitive_features(sensitive_features, attrs.shape[0])

    if inner_metric is None:
        inner_metric = _default_inner_metric

    scores = np.array([inner_metric(attrs[i]) for i in range(attrs.shape[0])])
    groups = _partition_by_group(sf)

    group_means = {}
    group_scores = {}
    for label, indices in groups.items():
        gs = scores[indices]
        group_means[label] = float(np.mean(gs))
        group_scores[label] = gs

    if len(groups) < 2:
        single_key = list(groups.keys())[0]
        return {
            "max_gap": 0.0,
            "mean_gap": 0.0,
            "pairwise_gaps": {},
            "group_means": {single_key: group_means[single_key]},
            "pairwise_p_values": {},
        }

    labels = sorted(groups.keys(), key=str)
    pairwise_gaps = {}
    pairwise_pvals = {}

    for la, lb in combinations(labels, 2):
        gap = abs(group_means[la] - group_means[lb])
        pairwise_gaps[(la, lb)] = float(gap)
        pairwise_pvals[(la, lb)] = _mann_whitney_u(
            group_scores[la], group_scores[lb]
        )

    gaps_list = list(pairwise_gaps.values())

    return {
        "max_gap": float(max(gaps_list)),
        "mean_gap": float(np.mean(gaps_list)),
        "pairwise_gaps": pairwise_gaps,
        "group_means": group_means,
        "pairwise_p_values": pairwise_pvals,
    }


# =============================================================================
# Metric 5: Attribution Parity (novel synthesis)
# =============================================================================

def compute_attribution_parity(
    attributions: Any,
    sensitive_features: Any,
    sensitive_feature_idx: int,
) -> Dict[str, Any]:
    """
    Compute Attribution Parity: whether the protected feature itself receives
    disproportionate attribution across groups.

    Extracts the attribution assigned to the sensitive feature column for each
    instance, partitions by group, and computes the absolute difference in
    group-mean attributions of that column plus a statistical divergence score.

    A high divergence indicates that the explanation method attributes
    importance to the sensitive feature differently depending on group
    membership — a potential signal of "fairwashing" or hidden bias.

    Args:
        attributions: 2D array (n_samples, n_features).
        sensitive_features: 1D group-label array.
        sensitive_feature_idx: Column index of the sensitive feature in
            the attribution matrix.

    Returns:
        Dict with keys:
            - "divergence": float — max absolute difference in group-mean
              attribution of the sensitive feature. 0 = fair.
            - "group_sensitive_means": dict — {group: mean_attribution_of_sensitive}.
            - "p_value": float — Mann-Whitney U p-value.

    References:
        Aïvodji, U., et al. (2019). Fairwashing: the risk of rationalization. ICML.
        Dai, J., et al. (2022). Fairness via Explanation Quality. AIES.
    """
    attrs = _validate_attributions(attributions)
    sf = _validate_sensitive_features(sensitive_features, attrs.shape[0])

    if sensitive_feature_idx < 0 or sensitive_feature_idx >= attrs.shape[1]:
        raise ValueError(
            f"sensitive_feature_idx {sensitive_feature_idx} out of bounds "
            f"for {attrs.shape[1]} features."
        )

    # Extract the column for the sensitive feature
    sens_attr = attrs[:, sensitive_feature_idx]

    groups = _partition_by_group(sf)
    group_means: Dict[Any, float] = {}
    group_vals: Dict[Any, np.ndarray] = {}

    for label, indices in groups.items():
        vals = sens_attr[indices]
        group_means[label] = float(np.mean(vals))
        group_vals[label] = vals

    if len(groups) < 2:
        single_key = list(groups.keys())[0]
        return {
            "divergence": 0.0,
            "group_sensitive_means": {single_key: group_means[single_key]},
            "p_value": 1.0,
        }

    # Max pairwise absolute difference of means
    labels = list(groups.keys())
    max_div = 0.0
    min_p = 1.0

    for la, lb in combinations(labels, 2):
        div = abs(group_means[la] - group_means[lb])
        if div > max_div:
            max_div = div
        p = _mann_whitney_u(group_vals[la], group_vals[lb])
        if p < min_p:
            min_p = p

    return {
        "divergence": float(max_div),
        "group_sensitive_means": group_means,
        "p_value": float(min_p),
    }


# =============================================================================
# Metric 6: Conditional Fairness (Hardt et al., 2016; adapted)
# =============================================================================

def compute_conditional_fairness(
    attributions: Any,
    sensitive_features: Any,
    predictions: Any,
    inner_metric: Optional[Callable[[np.ndarray], float]] = None,
) -> Dict[str, Any]:
    """
    Compute Conditional Fairness (Equalized Explanation Quality).

    Measures explanation quality disparity across groups *conditioned on the
    model's prediction class*. This separates explanation fairness from
    prediction fairness — a group might have worse explanations simply
    because the model makes different predictions for them.

    For each prediction class, computes group-level disparity, then
    aggregates (max over classes).

    Args:
        attributions: 2D array (n_samples, n_features).
        sensitive_features: 1D group-label array.
        predictions: 1D array of predicted class labels (same length).
        inner_metric: Optional per-instance quality metric. Defaults to L1 norm.

    Returns:
        Dict with keys:
            - "disparity": float — maximum across-class disparity.
            - "per_class_disparity": dict — {class_label: disparity_within_class}.
            - "per_class_group_means": dict — {class_label: {group: mean}}.

    References:
        Hardt, M., Price, E., & Srebro, N. (2016). Equality of Opportunity
        in Supervised Learning. NeurIPS.
    """
    attrs = _validate_attributions(attributions)
    sf = _validate_sensitive_features(sensitive_features, attrs.shape[0])
    preds = np.asarray(predictions)

    if preds.ndim != 1 or len(preds) != attrs.shape[0]:
        raise ValueError(
            f"predictions length ({len(preds)}) must match "
            f"attributions rows ({attrs.shape[0]})."
        )

    if inner_metric is None:
        inner_metric = _default_inner_metric

    # Per-instance scores
    scores = np.array([inner_metric(attrs[i]) for i in range(attrs.shape[0])])

    # Partition by prediction class
    pred_classes = np.unique(preds)
    per_class_disparity: Dict[Any, float] = {}
    per_class_group_means: Dict[Any, Dict[Any, float]] = {}

    for pc in pred_classes:
        pc_key = pc.item() if hasattr(pc, 'item') else pc
        class_mask = preds == pc
        class_scores = scores[class_mask]
        class_sf = sf[class_mask]

        class_groups = _partition_by_group(class_sf)

        if len(class_groups) < 2:
            # Only one group in this prediction class — no disparity
            per_class_disparity[pc_key] = 0.0
            gm = {}
            for label, indices in class_groups.items():
                gm[label] = float(np.mean(class_scores[indices]))
            per_class_group_means[pc_key] = gm
            continue

        # Compute per-group means within this class
        gm = {}
        for label, indices in class_groups.items():
            gm[label] = float(np.mean(class_scores[indices]))
        per_class_group_means[pc_key] = gm

        # Max pairwise gap within this class
        means_list = list(gm.values())
        max_gap = 0.0
        for i_m in range(len(means_list)):
            for j_m in range(i_m + 1, len(means_list)):
                gap = abs(means_list[i_m] - means_list[j_m])
                if gap > max_gap:
                    max_gap = gap
        per_class_disparity[pc_key] = float(max_gap)

    # Overall disparity: max across classes
    if per_class_disparity:
        overall = max(per_class_disparity.values())
    else:
        overall = 0.0

    return {
        "disparity": float(overall),
        "per_class_disparity": per_class_disparity,
        "per_class_group_means": per_class_group_means,
    }


# =============================================================================
# Default Fairness Registry (lazy)
# =============================================================================

_default_fairness_registry: Optional[FairnessMetricRegistry] = None


def _create_default_fairness_registry() -> FairnessMetricRegistry:
    """Create and populate the default fairness metric registry."""
    registry = FairnessMetricRegistry()

    registry.register(
        name="group_fairness",
        metric_fn=compute_group_fairness,
        meta=FairnessMetricMeta(
            level="group",
            composable=True,
            description=(
                "Group-based explanation quality disparity "
                "(Dai et al., 2022, AIES)"
            ),
            paper_reference=(
                "Dai, J., Upadhyay, S., Aïvodji, U., Bach, S. H., & "
                "Lakkaraju, H. (2022). Fairness via Explanation Quality. AIES."
            ),
        ),
    )

    # Individual fairness needs inputs as well, so we wrap it
    def _individual_fairness_wrapper(attributions, sensitive_features, **kwargs):
        inputs = kwargs.pop("inputs", attributions)
        return compute_individual_fairness(
            inputs=inputs,
            attributions=attributions,
            sensitive_features=sensitive_features,
            **kwargs,
        )

    registry.register(
        name="individual_fairness",
        metric_fn=_individual_fairness_wrapper,
        meta=FairnessMetricMeta(
            level="individual",
            composable=False,
            description=(
                "Individual-level explanation fairness via Lipschitz ratio "
                "(Dwork et al., 2012, adapted)"
            ),
            paper_reference=(
                "Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R. "
                "(2012). Fairness Through Awareness. ITCS."
            ),
        ),
    )

    def _counterfactual_wrapper(attributions, sensitive_features, **kwargs):
        inputs = kwargs.pop("inputs", attributions)
        sensitive_feature_idx = kwargs.pop("sensitive_feature_idx", 0)
        return compute_counterfactual_fairness(
            inputs=inputs,
            attributions=attributions,
            sensitive_feature_idx=sensitive_feature_idx,
            **kwargs,
        )

    registry.register(
        name="counterfactual_fairness",
        metric_fn=_counterfactual_wrapper,
        meta=FairnessMetricMeta(
            level="individual",
            composable=False,
            description=(
                "Counterfactual explanation fairness — explanations should not "
                "change when protected attribute is flipped "
                "(Kusner et al., 2017, adapted)"
            ),
            paper_reference=(
                "Kusner, M. J., Loftus, J., Russell, C., & Silva, R. (2017). "
                "Counterfactual Fairness. NeurIPS."
            ),
        ),
    )

    registry.register(
        name="fidelity_disparity",
        metric_fn=compute_fidelity_disparity,
        meta=FairnessMetricMeta(
            level="group",
            composable=True,
            description=(
                "Max/mean explanation quality gaps across subgroup pairs "
                "(Balagopalan et al., 2022, FAccT)"
            ),
            paper_reference=(
                "Balagopalan, A., Zhang, H., et al. (2022). The Road to "
                "Explainability is Paved with Bias. FAccT."
            ),
        ),
    )

    def _attribution_parity_wrapper(attributions, sensitive_features, **kwargs):
        sensitive_feature_idx = kwargs.pop("sensitive_feature_idx", 0)
        return compute_attribution_parity(
            attributions=attributions,
            sensitive_features=sensitive_features,
            sensitive_feature_idx=sensitive_feature_idx,
        )

    registry.register(
        name="attribution_parity",
        metric_fn=_attribution_parity_wrapper,
        meta=FairnessMetricMeta(
            level="group",
            composable=False,
            description=(
                "Whether the protected feature receives disproportionate "
                "attribution across groups (Aïvodji et al., 2019 + Dai et al., 2022)"
            ),
            paper_reference=(
                "Aïvodji, U., et al. (2019). Fairwashing. ICML. "
                "Dai, J., et al. (2022). Fairness via Explanation Quality. AIES."
            ),
        ),
    )

    def _conditional_wrapper(attributions, sensitive_features, **kwargs):
        predictions = kwargs.pop("predictions", np.zeros(len(sensitive_features), dtype=int))
        inner_metric = kwargs.pop("inner_metric", None)
        return compute_conditional_fairness(
            attributions=attributions,
            sensitive_features=sensitive_features,
            predictions=predictions,
            inner_metric=inner_metric,
        )

    registry.register(
        name="conditional_fairness",
        metric_fn=_conditional_wrapper,
        meta=FairnessMetricMeta(
            level="conditional",
            composable=True,
            description=(
                "Equalized explanation quality conditioned on prediction class "
                "(Hardt et al., 2016, adapted)"
            ),
            paper_reference=(
                "Hardt, M., Price, E., & Srebro, N. (2016). Equality of "
                "Opportunity in Supervised Learning. NeurIPS."
            ),
        ),
    )

    return registry


def get_default_fairness_registry() -> FairnessMetricRegistry:
    """Get the default global fairness metric registry (lazy initialization)."""
    global _default_fairness_registry
    if _default_fairness_registry is None:
        _default_fairness_registry = _create_default_fairness_registry()
    return _default_fairness_registry
