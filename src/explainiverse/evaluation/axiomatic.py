# src/explainiverse/evaluation/axiomatic.py
"""
Axiomatic evaluation metrics for explanations (Phase 6).

Implements:
- Completeness (Sundararajan et al., 2017) — sum of attributions ≈ f(x) - f(baseline)
- Non-Sensitivity (Nguyen & Martínez, 2020; Sundararajan et al., 2017) — non-influential
  features should receive zero attribution
- Input Invariance (Kindermans et al., 2017) — explanations should not change under
  constant shift of input (when model compensates)
- Symmetry (Sundararajan et al., 2017) — symmetric features should receive equal
  attribution

These metrics evaluate whether attribution methods satisfy fundamental axiomatic
properties that are desirable from a theoretical perspective. Violations of these
axioms indicate potential unreliability of the explanation method.

References:
    Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for
    Deep Networks. ICML. https://arxiv.org/abs/1703.01365

    Kindermans, P.-J., Hooker, S., Adebayo, J., Alber, M., Schütt, K. T.,
    Dähne, S., Erhan, D., & Kim, B. (2017). The (Un)reliability of Saliency
    Methods. arXiv:1711.00867.

    Nguyen, A. P., & Martínez, M. R. (2020). On Quantitative Aspects of Model
    Interpretability. arXiv:2007.07584.
"""
import copy
import warnings
import numpy as np
from typing import Union, Dict, Optional, List, Tuple, Callable

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


def _validate_model_fn(model_fn: Callable) -> None:
    """Validate that model_fn is callable."""
    if not callable(model_fn):
        raise TypeError(
            f"model_fn must be callable, got {type(model_fn).__name__}"
        )


def _safe_model_output(model_fn: Callable, x: np.ndarray) -> float:
    """
    Call model_fn and ensure scalar float output.

    Args:
        model_fn: Callable that takes input and returns scalar prediction.
        x: Input array.

    Returns:
        Scalar float output.
    """
    result = model_fn(x)
    # Handle numpy arrays, tensors, etc.
    if hasattr(result, 'item'):
        return float(result.item())
    if isinstance(result, np.ndarray):
        return float(result.flatten()[0])
    return float(result)


# =============================================================================
# Completeness (Sundararajan et al., 2017)
# =============================================================================

def compute_completeness(
    attributions: np.ndarray,
    model_fn: Callable,
    instance: np.ndarray,
    baseline: Optional[Union[np.ndarray, float]] = None,
    output_func: Optional[Callable] = None,
) -> float:
    """
    Compute the Completeness score for pre-computed attributions.

    Completeness (Sundararajan et al., 2017) is the axiom that the sum of
    attributions should equal the difference between the model output at the
    input and the model output at the baseline:

        Σᵢ aᵢ = F(x) - F(x')

    This metric measures the absolute deviation from this axiom:

        completeness_score = |Σᵢ aᵢ - (F(x) - F(x'))|

    A score of 0.0 indicates perfect completeness. Higher values indicate
    greater violation of the axiom.

    Also known as:
        - "Summation to Delta" (Shrikumar et al., 2017)
        - "Conservation" (Montavon et al., 2018)

    Properties:
        - Range: [0, ∞) — lower is better (0.0 = perfect)
        - Integrated Gradients satisfies this axiom by construction
        - SHAP values satisfy this axiom (they sum to f(x) - E[f(x)])
        - Gradient × Input generally does NOT satisfy this axiom

    Args:
        attributions: 1D numpy array of attribution values, shape (n_features,).
        model_fn: Callable that takes a 1D input array and returns a scalar
            prediction (e.g., probability for a target class, or raw logit).
        instance: Input instance, 1D array of shape (n_features,).
        baseline: Baseline input for the completeness check. Can be:
            - None: uses zero vector (default)
            - float: constant value broadcast to all features
            - np.ndarray: explicit baseline of shape (n_features,)
        output_func: Optional transformation applied to model output before
            computing the difference. E.g., lambda x: x to use raw output,
            or a softmax function. If None, model_fn output is used directly.

    Returns:
        Completeness score (float) in [0, ∞). Lower = better. 0.0 = perfect.

    Raises:
        ValueError: If attributions and instance have different lengths.
        TypeError: If model_fn is not callable.

    Example:
        >>> import numpy as np
        >>> from explainiverse.evaluation import compute_completeness
        >>> # Assume attributions from Integrated Gradients
        >>> score = compute_completeness(
        ...     attributions=ig_attrs,
        ...     model_fn=lambda x: model.predict_proba(x.reshape(1, -1))[0, 1],
        ...     instance=x_test[0],
        ... )
        >>> print(f"Completeness deviation: {score:.6f}")

    Reference:
        Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution
        for Deep Networks. ICML. https://arxiv.org/abs/1703.01365
    """
    _validate_model_fn(model_fn)

    attributions = np.asarray(attributions, dtype=np.float64).flatten()
    instance = np.asarray(instance, dtype=np.float64).flatten()

    if len(attributions) != len(instance):
        raise ValueError(
            f"attributions length ({len(attributions)}) must match "
            f"instance length ({len(instance)})"
        )

    n_features = len(instance)

    # Construct baseline
    if baseline is None:
        baseline_vec = np.zeros(n_features, dtype=np.float64)
    elif isinstance(baseline, (int, float)):
        baseline_vec = np.full(n_features, float(baseline), dtype=np.float64)
    else:
        baseline_vec = np.asarray(baseline, dtype=np.float64).flatten()
        if len(baseline_vec) != n_features:
            raise ValueError(
                f"baseline length ({len(baseline_vec)}) must match "
                f"instance length ({n_features})"
            )

    # Compute model outputs
    f_x = _safe_model_output(model_fn, instance)
    f_baseline = _safe_model_output(model_fn, baseline_vec)

    # Apply output_func if provided
    if output_func is not None:
        f_x = float(output_func(f_x))
        f_baseline = float(output_func(f_baseline))

    # Completeness deviation
    attribution_sum = float(np.sum(attributions))
    output_diff = f_x - f_baseline

    return float(np.abs(attribution_sum - output_diff))


def compute_completeness_score(
    explainer: BaseExplainer,
    model_fn: Callable,
    instance: np.ndarray,
    baseline: Optional[Union[np.ndarray, float]] = None,
    output_func: Optional[Callable] = None,
) -> float:
    """
    Compute Completeness using an explainer (high-level API).

    Generates attributions via the explainer, then checks the Completeness
    axiom: Σᵢ aᵢ ≈ F(x) - F(x').

    Args:
        explainer: Explainer instance with .explain() method.
        model_fn: Callable that takes a 1D input array and returns a scalar
            prediction.
        instance: Input instance (1D array of shape (n_features,)).
        baseline: Baseline for completeness check (None = zero vector).
        output_func: Optional transformation applied to model output.

    Returns:
        Completeness score (float) in [0, ∞). Lower = better. 0.0 = perfect.

    Example:
        >>> from explainiverse.evaluation import compute_completeness_score
        >>> score = compute_completeness_score(explainer, model_fn, instance)
        >>> print(f"Completeness: {score:.6f}")

    Reference:
        Sundararajan et al. (2017). Axiomatic Attribution for Deep Networks. ICML.
    """
    instance = np.asarray(instance, dtype=np.float64).flatten()
    n_features = len(instance)
    attr = _get_explanation_vector(explainer, instance, n_features)

    return compute_completeness(
        attributions=attr,
        model_fn=model_fn,
        instance=instance,
        baseline=baseline,
        output_func=output_func,
    )


def compute_batch_completeness(
    attributions_list: Optional[List[np.ndarray]] = None,
    explainer: Optional[BaseExplainer] = None,
    model_fn: Optional[Callable] = None,
    X: Optional[np.ndarray] = None,
    baseline: Optional[Union[np.ndarray, float]] = None,
    output_func: Optional[Callable] = None,
    max_instances: Optional[int] = None,
) -> Dict[str, object]:
    """
    Compute Completeness over a batch of instances.

    Supports two modes:
        1. Pre-computed: provide attributions_list, model_fn, and X.
        2. Explainer-based: provide explainer, model_fn, and X.

    Args:
        attributions_list: List of 1D attribution arrays (one per instance).
            If provided, used directly. Otherwise, explainer generates them.
        explainer: Explainer instance (used if attributions_list is None).
        model_fn: Callable for model predictions (required).
        X: Input data, 2D array of shape (n_instances, n_features) (required).
        baseline: Baseline for completeness check (None = zero vector).
        output_func: Optional transformation applied to model output.
        max_instances: Maximum number of instances to evaluate (None = all).

    Returns:
        Dictionary with:
            - "mean": Mean Completeness score across instances
            - "std": Standard deviation
            - "max": Maximum score (worst completeness)
            - "min": Minimum score (best completeness)
            - "scores": List of per-instance scores
            - "n_evaluated": Number of instances evaluated

    Raises:
        ValueError: If neither attributions_list nor explainer is provided,
            or if model_fn or X is missing.

    Example:
        >>> from explainiverse.evaluation import compute_batch_completeness
        >>> result = compute_batch_completeness(
        ...     explainer=explainer, model_fn=model_fn, X=X_test
        ... )
        >>> print(f"Mean Completeness: {result['mean']:.6f}")

    Reference:
        Sundararajan et al. (2017). Axiomatic Attribution for Deep Networks. ICML.
    """
    if model_fn is None:
        raise ValueError("model_fn is required for Completeness evaluation.")
    if X is None:
        raise ValueError("X (input data) is required for Completeness evaluation.")
    if attributions_list is None and explainer is None:
        raise ValueError(
            "Either attributions_list or explainer must be provided."
        )

    X = np.asarray(X)
    n = len(X)
    if max_instances is not None:
        n = min(n, max_instances)

    scores = []
    for i in range(n):
        try:
            if attributions_list is not None:
                score = compute_completeness(
                    attributions=attributions_list[i],
                    model_fn=model_fn,
                    instance=X[i],
                    baseline=baseline,
                    output_func=output_func,
                )
            else:
                score = compute_completeness_score(
                    explainer=explainer,
                    model_fn=model_fn,
                    instance=X[i],
                    baseline=baseline,
                    output_func=output_func,
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
# Non-Sensitivity (Nguyen & Martínez, 2020; Sundararajan et al., 2017)
# =============================================================================

def _detect_non_sensitive_features(
    model_fn: Callable,
    instance: np.ndarray,
    n_perturbations: int = 10,
    perturbation_scale: float = 0.1,
    tolerance: float = 1e-5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Detect features that do not influence model output via perturbation.

    For each feature i, we perturb only feature i by adding random noise
    drawn from N(0, perturbation_scale * |x_i| + perturbation_scale) and
    check if the model output changes beyond tolerance. If the output never
    changes for any of the perturbations, feature i is classified as
    non-sensitive.

    This implements the Sensitivity(b) / "Dummy" axiom detection from
    Sundararajan et al. (2017): if the function does not depend on a variable,
    perturbations of that variable should not change the output.

    Args:
        model_fn: Callable that takes a 1D input and returns scalar output.
        instance: Input instance (1D array, shape (n_features,)).
        n_perturbations: Number of perturbations per feature (default: 10).
        perturbation_scale: Scale of perturbation noise (default: 0.1).
        tolerance: Output change threshold below which a feature is
            considered non-sensitive (default: 1e-5).
        seed: Random seed for reproducibility (default: None).

    Returns:
        Boolean array of shape (n_features,). True = non-sensitive.
    """
    rng = np.random.RandomState(seed)
    n_features = len(instance)
    f_original = _safe_model_output(model_fn, instance)

    is_non_sensitive = np.ones(n_features, dtype=bool)

    for i in range(n_features):
        for _ in range(n_perturbations):
            perturbed = instance.copy()
            noise_scale = perturbation_scale * (np.abs(instance[i]) + perturbation_scale)
            perturbed[i] += rng.normal(0, noise_scale)

            f_perturbed = _safe_model_output(model_fn, perturbed)
            if np.abs(f_perturbed - f_original) > tolerance:
                is_non_sensitive[i] = False
                break

    return is_non_sensitive


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
) -> float:
    """
    Compute Non-Sensitivity for pre-computed attributions.

    Non-Sensitivity (Nguyen & Martínez, 2020) evaluates whether features
    that do not influence the model output receive zero attribution, as
    required by the Sensitivity(b) / "Dummy" axiom (Sundararajan et al.,
    2017; Friedman, 2004):

        If F does not depend on feature i, then aᵢ = 0.

    This metric measures the total absolute attribution assigned to
    non-sensitive features:

        non_sensitivity = Σᵢ |aᵢ|  for all i where feature i is non-sensitive

    A score of 0.0 indicates perfect adherence to the axiom. Higher values
    indicate the method is assigning attribution to features that provably
    do not affect the model's prediction.

    Non-sensitive features are identified either:
        (a) by user-provided boolean mask (non_sensitive_features), or
        (b) automatically via perturbation-based detection.

    Properties:
        - Range: [0, ∞) unnormalized, [0, 1] normalized — lower is better
        - Score = 0.0 means no attribution is wasted on irrelevant features
        - Perturbation-based detection uses multiple random perturbations
          per feature to robustly identify non-influential features

    Args:
        attributions: 1D numpy array of attribution values, shape (n_features,).
        model_fn: Callable that takes a 1D input array and returns a scalar
            prediction.
        instance: Input instance (1D array of shape (n_features,)).
        non_sensitive_features: Optional boolean array of shape (n_features,).
            True indicates the feature is non-sensitive (should have zero
            attribution). If None, features are detected automatically via
            perturbation.
        n_perturbations: Number of perturbations per feature for auto-detection
            (default: 10). Ignored if non_sensitive_features is provided.
        perturbation_scale: Scale of perturbation noise for auto-detection
            (default: 0.1). Ignored if non_sensitive_features is provided.
        tolerance: Output change threshold for auto-detection (default: 1e-5).
            Ignored if non_sensitive_features is provided.
        normalize: If True, normalize by the total absolute attribution
            (returns fraction of attribution wasted on non-sensitive features).
            Default: False.
        seed: Random seed for perturbation-based detection (default: None).

    Returns:
        Non-Sensitivity score (float). Lower = better. 0.0 = perfect.
        Unnormalized: sum of |aᵢ| for non-sensitive features.
        Normalized: fraction of total |attribution| on non-sensitive features.
        Returns 0.0 if no non-sensitive features are detected.

    Raises:
        ValueError: If attributions and instance have different lengths.
        TypeError: If model_fn is not callable.

    Example:
        >>> import numpy as np
        >>> from explainiverse.evaluation import compute_non_sensitivity
        >>> score = compute_non_sensitivity(
        ...     attributions=attrs,
        ...     model_fn=lambda x: model.predict_proba(x.reshape(1, -1))[0, 1],
        ...     instance=x_test[0],
        ... )
        >>> print(f"Non-Sensitivity: {score:.6f}")

    Reference:
        Nguyen, A. P., & Martínez, M. R. (2020). On Quantitative Aspects of
        Model Interpretability. arXiv:2007.07584.

        Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution
        for Deep Networks. ICML. (Sensitivity(b) / Dummy axiom)
    """
    _validate_model_fn(model_fn)

    attributions = np.asarray(attributions, dtype=np.float64).flatten()
    instance = np.asarray(instance, dtype=np.float64).flatten()

    if len(attributions) != len(instance):
        raise ValueError(
            f"attributions length ({len(attributions)}) must match "
            f"instance length ({len(instance)})"
        )

    # Detect or validate non-sensitive features
    if non_sensitive_features is not None:
        ns_mask = np.asarray(non_sensitive_features, dtype=bool).flatten()
        if len(ns_mask) != len(instance):
            raise ValueError(
                f"non_sensitive_features length ({len(ns_mask)}) must match "
                f"instance length ({len(instance)})"
            )
    else:
        ns_mask = _detect_non_sensitive_features(
            model_fn=model_fn,
            instance=instance,
            n_perturbations=n_perturbations,
            perturbation_scale=perturbation_scale,
            tolerance=tolerance,
            seed=seed,
        )

    # If no non-sensitive features found, score is 0 (vacuously satisfied)
    if not np.any(ns_mask):
        return 0.0

    abs_attr = np.abs(attributions)
    ns_attribution = float(np.sum(abs_attr[ns_mask]))

    if normalize:
        total_attribution = float(np.sum(abs_attr))
        if total_attribution < 1e-300:
            return 0.0
        return ns_attribution / total_attribution

    return ns_attribution


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
) -> float:
    """
    Compute Non-Sensitivity using an explainer (high-level API).

    Generates attributions via the explainer, then checks the Non-Sensitivity
    axiom.

    Args:
        explainer: Explainer instance with .explain() method.
        model_fn: Callable for model predictions.
        instance: Input instance (1D array).
        non_sensitive_features: Optional boolean mask of non-sensitive features.
        n_perturbations: Number of perturbations for auto-detection.
        perturbation_scale: Scale of perturbation noise.
        tolerance: Output change threshold for auto-detection.
        normalize: If True, normalize by total attribution.
        seed: Random seed.

    Returns:
        Non-Sensitivity score (float). Lower = better.

    Reference:
        Nguyen & Martínez (2020). On Quantitative Aspects of Model
        Interpretability. arXiv:2007.07584.
    """
    instance = np.asarray(instance, dtype=np.float64).flatten()
    n_features = len(instance)
    attr = _get_explanation_vector(explainer, instance, n_features)

    return compute_non_sensitivity(
        attributions=attr,
        model_fn=model_fn,
        instance=instance,
        non_sensitive_features=non_sensitive_features,
        n_perturbations=n_perturbations,
        perturbation_scale=perturbation_scale,
        tolerance=tolerance,
        normalize=normalize,
        seed=seed,
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
) -> Dict[str, object]:
    """
    Compute Non-Sensitivity over a batch of instances.

    Non-sensitive features are detected per-instance (unless a global mask
    is provided via non_sensitive_features).

    Args:
        attributions_list: List of 1D attribution arrays (one per instance).
        explainer: Explainer instance (used if attributions_list is None).
        model_fn: Callable for model predictions (required).
        X: Input data (2D array, required).
        non_sensitive_features: Optional boolean mask shared across all
            instances. If None, auto-detection is performed per instance.
        n_perturbations: Number of perturbations for auto-detection.
        perturbation_scale: Scale of perturbation noise.
        tolerance: Output change threshold.
        normalize: If True, normalize scores.
        seed: Random seed.
        max_instances: Maximum instances to evaluate (None = all).

    Returns:
        Dictionary with mean, std, max, min, scores, n_evaluated.

    Reference:
        Nguyen & Martínez (2020). On Quantitative Aspects of Model
        Interpretability. arXiv:2007.07584.
    """
    if model_fn is None:
        raise ValueError("model_fn is required for Non-Sensitivity evaluation.")
    if X is None:
        raise ValueError("X (input data) is required for Non-Sensitivity evaluation.")
    if attributions_list is None and explainer is None:
        raise ValueError(
            "Either attributions_list or explainer must be provided."
        )

    X = np.asarray(X)
    n = len(X)
    if max_instances is not None:
        n = min(n, max_instances)

    scores = []
    for i in range(n):
        try:
            # Use per-instance seed for reproducibility
            instance_seed = (seed + i) if seed is not None else None

            if attributions_list is not None:
                score = compute_non_sensitivity(
                    attributions=attributions_list[i],
                    model_fn=model_fn,
                    instance=X[i],
                    non_sensitive_features=non_sensitive_features,
                    n_perturbations=n_perturbations,
                    perturbation_scale=perturbation_scale,
                    tolerance=tolerance,
                    normalize=normalize,
                    seed=instance_seed,
                )
            else:
                score = compute_non_sensitivity_score(
                    explainer=explainer,
                    model_fn=model_fn,
                    instance=X[i],
                    non_sensitive_features=non_sensitive_features,
                    n_perturbations=n_perturbations,
                    perturbation_scale=perturbation_scale,
                    tolerance=tolerance,
                    normalize=normalize,
                    seed=instance_seed,
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
# Input Invariance (Kindermans et al., 2017)
# =============================================================================

def compute_input_invariance(
    explain_func: Callable,
    instance: np.ndarray,
    shift: Optional[Union[np.ndarray, float]] = None,
    seed: Optional[int] = None,
) -> float:
    """
    Compute Input Invariance (simplified version, model-agnostic).

    Input Invariance (Kindermans et al., 2017) tests whether an explanation
    method changes its output when a constant is added to the input. The
    key insight is: if two networks compute the same function (one on
    original inputs, one on shifted inputs with compensating bias), their
    explanations should be identical.

    This simplified version measures:

        input_invariance = ||E(x) - E(x + c)||₂ / n

    where c is a constant shift vector and n is the number of features
    (normalization for comparability across different dimensionalities).

    This tests whether the explanation method is inherently sensitive to
    input translation. Note: this does NOT adjust the model — it measures
    raw sensitivity of the explain_func to input shift. For the full
    axiom-faithful version with model compensation, use
    compute_input_invariance_pytorch().

    Properties:
        - Range: [0, ∞) — lower is better (0.0 = perfectly invariant)
        - Gradient methods are input invariant by construction
        - Gradient × Input is NOT input invariant (the shift carries through)
        - Integrated Gradients with zero baseline is NOT input invariant

    Args:
        explain_func: Callable that takes a 1D input array and returns a 1D
            array of attributions. E.g.:
            ``lambda x: explainer.explain(x).get_attributions()``
        instance: Input instance (1D array of shape (n_features,)).
        shift: Constant shift to add to the input. Can be:
            - None: random shift from U(-1, 1) per feature (default)
            - float: constant shift broadcast to all features
            - np.ndarray: explicit shift vector of shape (n_features,)
        seed: Random seed for shift generation (used when shift is None).

    Returns:
        Input Invariance score (float) in [0, ∞). Lower = better.
        Normalized by number of features.

    Raises:
        TypeError: If explain_func is not callable.

    Example:
        >>> from explainiverse.evaluation import compute_input_invariance
        >>> def explain_fn(x):
        ...     exp = explainer.explain(x)
        ...     return np.array(list(exp.explanation_data[
        ...         "feature_attributions"].values()))
        >>> score = compute_input_invariance(explain_fn, instance)
        >>> print(f"Input Invariance: {score:.6f}")

    Reference:
        Kindermans, P.-J., Hooker, S., Adebayo, J., et al. (2017). The
        (Un)reliability of Saliency Methods. arXiv:1711.00867.
    """
    if not callable(explain_func):
        raise TypeError(
            f"explain_func must be callable, got {type(explain_func).__name__}"
        )

    instance = np.asarray(instance, dtype=np.float64).flatten()
    n_features = len(instance)

    # Generate shift
    if shift is None:
        rng = np.random.RandomState(seed)
        shift_vec = rng.uniform(-1.0, 1.0, size=n_features)
    elif isinstance(shift, (int, float)):
        shift_vec = np.full(n_features, float(shift), dtype=np.float64)
    else:
        shift_vec = np.asarray(shift, dtype=np.float64).flatten()
        if len(shift_vec) != n_features:
            raise ValueError(
                f"shift length ({len(shift_vec)}) must match "
                f"instance length ({n_features})"
            )

    # Compute attributions for original and shifted inputs
    attr_original = np.asarray(explain_func(instance), dtype=np.float64).flatten()
    shifted_instance = instance + shift_vec
    attr_shifted = np.asarray(explain_func(shifted_instance), dtype=np.float64).flatten()

    # L2 norm of difference, normalized by number of features
    diff = attr_original - attr_shifted
    l2_diff = float(np.sqrt(np.sum(diff ** 2)))

    return l2_diff / n_features


def compute_input_invariance_pytorch(
    model,
    explain_func: Callable,
    instance: np.ndarray,
    shift: Optional[Union[np.ndarray, float]] = None,
    seed: Optional[int] = None,
) -> float:
    """
    Compute Input Invariance with model compensation (PyTorch-only).

    This is the full axiom-faithful implementation from Kindermans et al.
    (2017). It:
        1. Creates a shifted input x' = x + c
        2. Creates a compensated model by adjusting the first layer's bias
           to absorb the shift (so the compensated model on x' produces the
           same output as the original model on x)
        3. Computes attributions for both (original model, x) and
           (compensated model, x')
        4. Measures the difference: ||E(x) - E'(x')||₂ / n

    A method satisfying Input Invariance will produce score ≈ 0 because
    the explanations should be identical for functionally equivalent
    model-input pairs.

    Properties:
        - Range: [0, ∞) — lower is better (0.0 = perfectly invariant)
        - Requires PyTorch model with accessible first-layer weights
        - Deep copy is used to never modify the user's model

    Args:
        model: PyTorch nn.Module. Must have a first linear/conv layer
            whose bias can be adjusted.
        explain_func: Callable with signature (model, instance) -> attributions.
            Takes a PyTorch model and a 1D numpy input, returns a 1D numpy
            array of attributions. E.g.:
            ``lambda m, x: my_explain(m, x)``
        instance: Input instance (1D array of shape (n_features,)).
        shift: Constant shift. None = random U(-1,1), float = broadcast,
            ndarray = explicit.
        seed: Random seed.

    Returns:
        Input Invariance score (float) in [0, ∞). Lower = better.

    Raises:
        ImportError: If PyTorch is not installed.
        RuntimeError: If the model's first layer cannot be identified or
            does not support bias compensation.

    Example:
        >>> from explainiverse.evaluation import compute_input_invariance_pytorch
        >>> def explain_fn(model, x):
        ...     # Your attribution computation
        ...     return attributions_array
        >>> score = compute_input_invariance_pytorch(
        ...     model, explain_fn, instance
        ... )

    Reference:
        Kindermans, P.-J., Hooker, S., Adebayo, J., et al. (2017). The
        (Un)reliability of Saliency Methods. arXiv:1711.00867.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError(
            "PyTorch is required for compute_input_invariance_pytorch. "
            "Install with: pip install torch"
        )

    if not callable(explain_func):
        raise TypeError(
            f"explain_func must be callable, got {type(explain_func).__name__}"
        )

    instance = np.asarray(instance, dtype=np.float64).flatten()
    n_features = len(instance)

    # Generate shift
    if shift is None:
        rng = np.random.RandomState(seed)
        shift_vec = rng.uniform(-1.0, 1.0, size=n_features)
    elif isinstance(shift, (int, float)):
        shift_vec = np.full(n_features, float(shift), dtype=np.float64)
    else:
        shift_vec = np.asarray(shift, dtype=np.float64).flatten()
        if len(shift_vec) != n_features:
            raise ValueError(
                f"shift length ({len(shift_vec)}) must match "
                f"instance length ({n_features})"
            )

    # Compute attributions for original model + original input
    attr_original = np.asarray(
        explain_func(model, instance), dtype=np.float64
    ).flatten()

    # Create compensated model via deep copy
    compensated_model = copy.deepcopy(model)

    # Find the first layer (Linear or Conv1d/Conv2d) and adjust bias
    first_layer = None
    for module in compensated_model.modules():
        if isinstance(module, nn.Linear):
            first_layer = module
            break
        elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
            first_layer = module
            break

    if first_layer is None:
        raise RuntimeError(
            "Could not identify the first Linear or Conv layer in the model. "
            "Input Invariance (PyTorch) requires a model with an accessible "
            "first layer whose bias can be adjusted."
        )

    # Adjust first-layer bias to compensate for the shift
    # For a linear layer: y = Wx + b → y' = W(x+c) + b' = Wx + Wc + b'
    # To keep output the same: b' = b - Wc
    with torch.no_grad():
        shift_tensor = torch.tensor(shift_vec, dtype=first_layer.weight.dtype,
                                    device=first_layer.weight.device)

        if isinstance(first_layer, nn.Linear):
            # weight shape: (out_features, in_features)
            # bias_adjustment = W @ c
            bias_adjustment = first_layer.weight @ shift_tensor
        elif isinstance(first_layer, (nn.Conv1d, nn.Conv2d)):
            # For Conv layers, the shift is per-channel (simplified case)
            # This applies when shift is uniform across spatial dimensions
            # weight shape: (out_channels, in_channels, *kernel_size)
            # Sum over spatial dims and in_channels
            if isinstance(first_layer, nn.Conv1d):
                # (out_ch, in_ch, kernel) × shift(in_ch) → sum over in_ch, kernel
                w_sum = first_layer.weight.sum(dim=2)  # (out_ch, in_ch)
                bias_adjustment = w_sum @ shift_tensor
            else:
                # Conv2d: (out_ch, in_ch, kH, kW)
                w_sum = first_layer.weight.sum(dim=(2, 3))  # (out_ch, in_ch)
                bias_adjustment = w_sum @ shift_tensor

        if first_layer.bias is None:
            first_layer.bias = nn.Parameter(
                -bias_adjustment, requires_grad=False
            )
        else:
            first_layer.bias.data -= bias_adjustment

    # Compute attributions for compensated model + shifted input
    shifted_instance = instance + shift_vec
    attr_shifted = np.asarray(
        explain_func(compensated_model, shifted_instance), dtype=np.float64
    ).flatten()

    # Clean up
    del compensated_model

    # L2 norm of difference, normalized by number of features
    diff = attr_original - attr_shifted
    l2_diff = float(np.sqrt(np.sum(diff ** 2)))

    return l2_diff / n_features


def compute_batch_input_invariance(
    explain_func: Callable,
    X: np.ndarray,
    shift: Optional[Union[np.ndarray, float]] = None,
    seed: Optional[int] = None,
    max_instances: Optional[int] = None,
) -> Dict[str, object]:
    """
    Compute Input Invariance (simplified) over a batch of instances.

    Args:
        explain_func: Callable that takes 1D input → 1D attributions.
        X: Input data (2D array of shape (n_instances, n_features)).
        shift: Constant shift (None = random per instance).
        seed: Random seed.
        max_instances: Maximum instances to evaluate.

    Returns:
        Dictionary with mean, std, max, min, scores, n_evaluated.

    Reference:
        Kindermans et al. (2017). The (Un)reliability of Saliency Methods.
    """
    X = np.asarray(X)
    n = len(X)
    if max_instances is not None:
        n = min(n, max_instances)

    scores = []
    for i in range(n):
        try:
            instance_seed = (seed + i) if seed is not None else None
            score = compute_input_invariance(
                explain_func=explain_func,
                instance=X[i],
                shift=shift,
                seed=instance_seed,
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


def compute_batch_input_invariance_pytorch(
    model,
    explain_func: Callable,
    X: np.ndarray,
    shift: Optional[Union[np.ndarray, float]] = None,
    seed: Optional[int] = None,
    max_instances: Optional[int] = None,
) -> Dict[str, object]:
    """
    Compute Input Invariance (PyTorch, with model compensation) over a batch.

    Args:
        model: PyTorch nn.Module.
        explain_func: Callable with signature (model, instance) -> attributions.
        X: Input data (2D array).
        shift: Constant shift.
        seed: Random seed.
        max_instances: Maximum instances.

    Returns:
        Dictionary with mean, std, max, min, scores, n_evaluated.

    Reference:
        Kindermans et al. (2017). The (Un)reliability of Saliency Methods.
    """
    X = np.asarray(X)
    n = len(X)
    if max_instances is not None:
        n = min(n, max_instances)

    scores = []
    for i in range(n):
        try:
            instance_seed = (seed + i) if seed is not None else None
            score = compute_input_invariance_pytorch(
                model=model,
                explain_func=explain_func,
                instance=X[i],
                shift=shift,
                seed=instance_seed,
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
# Symmetry (Sundararajan et al., 2017)
# =============================================================================

def compute_symmetry(
    attributions: np.ndarray,
    symmetric_pairs: List[Tuple[int, int]],
) -> float:
    """
    Compute Symmetry for pre-computed attributions.

    Symmetry (Sundararajan et al., 2017) is the axiom that if two input
    features are functionally symmetric — swapping their values does not
    change the model output for any input — then they should receive
    identical attributions:

        If features i and j are symmetric, then aᵢ = aⱼ

    This metric measures the mean absolute difference in attribution
    between specified symmetric feature pairs:

        symmetry = (1/|P|) Σ_{(i,j) ∈ P} |aᵢ - aⱼ|

    where P is the set of symmetric feature pairs.

    A score of 0.0 indicates perfect symmetry preservation. Higher values
    indicate the method assigns different attributions to features that
    should, by construction, receive equal attribution.

    Properties:
        - Range: [0, ∞) — lower is better (0.0 = perfect)
        - Integrated Gradients (with straight-line path) preserves symmetry
        - Other path methods may NOT preserve symmetry
        - NOT in Quantus — this is an Explainiverse differentiator

    Note:
        Symmetric features must be identified by the user. Auto-detection
        is model-specific and computationally expensive (requires testing
        all feature permutations). Common cases include:
        - Features with identical roles (e.g., x₁ and x₂ in f = x₁ + x₂)
        - Duplicate/redundant features

    Args:
        attributions: 1D numpy array of attribution values, shape (n_features,).
        symmetric_pairs: List of (i, j) tuples specifying pairs of feature
            indices that are symmetric. Indices are 0-based.

    Returns:
        Symmetry score (float) in [0, ∞). Lower = better. 0.0 = perfect.
        Returns 0.0 if symmetric_pairs is empty.

    Raises:
        ValueError: If any index in symmetric_pairs is out of bounds.

    Example:
        >>> import numpy as np
        >>> from explainiverse.evaluation import compute_symmetry
        >>> # Features 0 and 1 are symmetric (e.g., f = x0 + x1)
        >>> score = compute_symmetry(attrs, symmetric_pairs=[(0, 1)])
        >>> print(f"Symmetry violation: {score:.6f}")

    Reference:
        Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution
        for Deep Networks. ICML. https://arxiv.org/abs/1703.01365
    """
    attributions = np.asarray(attributions, dtype=np.float64).flatten()
    n_features = len(attributions)

    if not symmetric_pairs:
        return 0.0

    # Validate indices
    for i, j in symmetric_pairs:
        if i < 0 or i >= n_features:
            raise ValueError(
                f"Feature index {i} out of bounds for {n_features} features."
            )
        if j < 0 or j >= n_features:
            raise ValueError(
                f"Feature index {j} out of bounds for {n_features} features."
            )

    # Compute mean absolute difference across symmetric pairs
    diffs = []
    for i, j in symmetric_pairs:
        diffs.append(np.abs(attributions[i] - attributions[j]))

    return float(np.mean(diffs))


def compute_symmetry_score(
    explainer: BaseExplainer,
    instance: np.ndarray,
    symmetric_pairs: List[Tuple[int, int]],
) -> float:
    """
    Compute Symmetry using an explainer (high-level API).

    Generates attributions via the explainer, then checks the Symmetry axiom.

    Args:
        explainer: Explainer instance with .explain() method.
        instance: Input instance (1D array).
        symmetric_pairs: List of (i, j) tuples of symmetric feature indices.

    Returns:
        Symmetry score (float) in [0, ∞). Lower = better.

    Reference:
        Sundararajan et al. (2017). Axiomatic Attribution for Deep Networks. ICML.
    """
    instance = np.asarray(instance, dtype=np.float64).flatten()
    n_features = len(instance)
    attr = _get_explanation_vector(explainer, instance, n_features)

    return compute_symmetry(
        attributions=attr,
        symmetric_pairs=symmetric_pairs,
    )


def compute_batch_symmetry(
    symmetric_pairs: List[Tuple[int, int]],
    attributions_list: Optional[List[np.ndarray]] = None,
    explainer: Optional[BaseExplainer] = None,
    X: Optional[np.ndarray] = None,
    max_instances: Optional[int] = None,
) -> Dict[str, object]:
    """
    Compute Symmetry over a batch of instances.

    Supports two modes:
        1. Pre-computed: provide attributions_list.
        2. Explainer-based: provide explainer and X.

    Args:
        symmetric_pairs: List of (i, j) tuples of symmetric feature indices.
        attributions_list: List of 1D attribution arrays.
        explainer: Explainer instance (used if attributions_list is None).
        X: Input data (required if using explainer).
        max_instances: Maximum instances to evaluate.

    Returns:
        Dictionary with mean, std, max, min, scores, n_evaluated.

    Raises:
        ValueError: If neither attributions_list nor (explainer + X) provided.

    Reference:
        Sundararajan et al. (2017). Axiomatic Attribution for Deep Networks. ICML.
    """
    if attributions_list is None and (explainer is None or X is None):
        raise ValueError(
            "Either attributions_list or (explainer + X) must be provided."
        )

    if attributions_list is not None:
        n = len(attributions_list)
    else:
        X = np.asarray(X)
        n = len(X)

    if max_instances is not None:
        n = min(n, max_instances)

    scores = []
    for i in range(n):
        try:
            if attributions_list is not None:
                score = compute_symmetry(
                    attributions=attributions_list[i],
                    symmetric_pairs=symmetric_pairs,
                )
            else:
                score = compute_symmetry_score(
                    explainer=explainer,
                    instance=X[i],
                    symmetric_pairs=symmetric_pairs,
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
