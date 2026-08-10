# src/explainiverse/evaluation/faithfulness_extended.py
"""Extended, numeric-feature faithfulness diagnostics.

Only Infidelity and Sensitivity-n below are direct estimators of equations in
their cited papers (subject to the documented perturbation and scalar-output
choices).  The image-only metrics expose explicitly labelled one-dimensional
adaptations; their values are not interchangeable with published image
benchmarks.  ROAD is deliberately labelled as inspired by ROAD because this
module does not implement ROAD's spatial noisy-linear imputation or its
dataset-level accuracy curve.

Diagnostics resolve one numeric output from an explicit ``target_class``,
recorded output-index metadata, or an explanation label that maps through the
model's class labels. A one-output model is unambiguous. Multi-output calls fail
when that identity cannot be established; model argmax is never substituted
for an unmappable explanation target.

Batch APIs require exactly one explanation for every row of ``X`` before
optionally shortening evaluation with ``max_samples``. Statistical and
callable baselines require an explicit, separately supplied
``background_data`` matrix; evaluated rows are never silently reused as their
own background distribution.
Where a paper leaves ranking ties unspecified, this module uses a stable
original-feature/segment-index tie convention; scores are not claimed to be
tie-invariant.
"""

import re
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats

# NumPy 2.0 compatibility: np.trapz was renamed to np.trapezoid
try:
    _trapezoid = np.trapezoid
except AttributeError:
    _trapezoid = getattr(np, "trapz")

from explainiverse.core.explanation import Explanation
from explainiverse.evaluation._utils import (
    apply_feature_mask,
    compute_baseline_values,
    get_prediction_value,
)
from explainiverse.evaluation.faithfulness import _resolve_target_output


def _as_feature_vector(instance: np.ndarray) -> np.ndarray:
    """Return one finite, non-empty numeric feature vector without reshaping."""
    try:
        values = np.asarray(instance, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("instance must contain real numeric values") from exc
    if values.ndim != 1:
        raise ValueError(f"instance must be one-dimensional, got shape {values.shape}")
    if values.size == 0:
        raise ValueError("instance must contain at least one feature")
    if not np.all(np.isfinite(values)):
        raise ValueError("instance must contain only finite values")
    return values.copy()


def _validate_positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_nonnegative_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _validate_bool(value, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def _validate_nonnegative_real(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _defined_pearson(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Compute Pearson's r, rejecting samples for which it is undefined."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or a.size < 2:
        raise ValueError("Pearson correlation requires two paired observations")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Pearson correlation inputs must be finite")
    if np.ptp(a) == 0.0 or np.ptp(b) == 0.0:
        raise ValueError("Pearson correlation is undefined for a constant input")
    correlation, p_value = stats.pearsonr(a, b)
    if not np.isfinite(correlation) or not np.isfinite(p_value):
        raise ValueError("Pearson correlation is undefined for these observations")
    return float(correlation), float(p_value)


def _defined_spearman(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Compute Spearman's rho, rejecting samples for which it is undefined."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or a.size < 2:
        raise ValueError("Spearman correlation requires two paired observations")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Spearman correlation inputs must be finite")
    if np.ptp(a) == 0.0 or np.ptp(b) == 0.0:
        raise ValueError("Spearman correlation is undefined for a constant input")
    correlation, p_value = stats.spearmanr(a, b)
    if not np.isfinite(correlation):
        raise ValueError("Spearman correlation is undefined for these observations")
    return float(correlation), float(p_value) if np.isfinite(p_value) else np.nan


def _validated_baseline_values(
    baseline: Union[str, float, np.ndarray, Callable],
    background_data: Optional[np.ndarray],
    n_features: int,
) -> np.ndarray:
    """Resolve a baseline and enforce the numeric-feature contract."""
    if isinstance(baseline, (bool, np.bool_)):
        raise TypeError("baseline must not be boolean")
    validated_background = background_data
    if background_data is not None:
        try:
            validated_background = np.asarray(background_data, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError("background_data must contain real numeric values") from exc
        if (
            validated_background.ndim != 2
            or validated_background.shape[0] == 0
            or validated_background.shape[1] != n_features
        ):
            raise ValueError(
                "background_data must be a non-empty 2D array with "
                f"{n_features} columns; got shape {validated_background.shape}"
            )
        if not np.all(np.isfinite(validated_background)):
            raise ValueError("background_data must contain only finite values")

    values = np.asarray(
        compute_baseline_values(baseline, validated_background, n_features),
        dtype=float,
    )
    if values.shape != (n_features,):
        raise ValueError(f"baseline must resolve to shape ({n_features},), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("baseline must resolve to finite values")
    return values.copy()


def _validate_batch_inputs(
    X: np.ndarray,
    explanations: Sequence[Explanation],
    max_samples: Optional[int],
) -> Tuple[np.ndarray, int]:
    """Validate the exact one-explanation-per-row batch contract."""
    try:
        X_array = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("X must contain real numeric values") from exc
    if X_array.ndim != 2 or X_array.shape[0] == 0 or X_array.shape[1] == 0:
        raise ValueError("X must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(X_array)):
        raise ValueError("X must contain only finite values")
    if isinstance(explanations, (str, bytes)) or not isinstance(explanations, Sequence):
        raise TypeError("explanations must be a sequence")
    if len(explanations) != X_array.shape[0]:
        raise ValueError(
            f"expected exactly one explanation per row of X ({X_array.shape[0]}); "
            f"got {len(explanations)}"
        )
    n_samples = X_array.shape[0]
    if max_samples is not None:
        n_samples = min(n_samples, _validate_positive_int(max_samples, "max_samples"))
    return X_array, n_samples


def _summarize_scores(scores: Sequence[float]) -> Dict[str, float]:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("batch evaluation produced no finite scores")
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n_samples": int(values.size),
    }


def _require_scalar_result(result: Union[float, Mapping[str, object]], metric_name: str) -> float:
    """Narrow a detail-capable metric call made with details disabled."""
    if isinstance(result, Mapping):
        raise RuntimeError(f"{metric_name} unexpectedly returned detail data")
    return float(result)


def _extract_attribution_array(explanation: Explanation, n_features: int) -> np.ndarray:
    """
    Extract attribution values as a numpy array in feature index order.

    Args:
        explanation: Explanation object with feature_attributions
        n_features: Expected number of features

    Returns:
        1D numpy array of attribution values ordered by feature index
    """
    if isinstance(n_features, bool) or not isinstance(n_features, Integral) or n_features <= 0:
        raise ValueError("n_features must be a positive integer")
    if not isinstance(explanation, Explanation):
        raise TypeError("explanation must be an Explanation")
    attributions = explanation.explanation_data.get("feature_attributions")
    if not isinstance(attributions, Mapping) or not attributions:
        raise ValueError("No feature attributions found in explanation.")

    feature_names = getattr(explanation, "feature_names", None)
    if feature_names is not None:
        if isinstance(feature_names, (str, bytes)):
            raise TypeError("feature_names must be a sequence of strings")
        try:
            feature_names = list(feature_names)
        except TypeError as exc:
            raise TypeError("feature_names must be a sequence of strings") from exc
        if len(feature_names) != n_features:
            raise ValueError("feature_names length must equal the number of input features")
        if any(not isinstance(name, str) or not name for name in feature_names):
            raise ValueError("feature_names must contain non-empty strings")
        if len(set(feature_names)) != n_features:
            raise ValueError("feature_names must be unique")

    attr_array = np.zeros(int(n_features), dtype=float)
    assigned_indices = set()

    def resolve_index(feature_key: str) -> int:
        if feature_names is not None:
            if feature_key in feature_names:
                return feature_names.index(feature_key)
            # Legacy LIME payloads may encode one declared name inside a
            # numeric interval/inequality. Match only the complete supported
            # condition grammar; plain substring containment can map ``age``
            # incorrectly from an unrelated key such as ``mortgage``.
            number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
            operator = r"(?:<=|>=|<|>|=)"
            candidates = []
            for index, name in enumerate(feature_names):
                escaped_name = re.escape(name)
                patterns = (
                    rf"{escaped_name}\s*{operator}\s*{number}",
                    rf"{number}\s*{operator}\s*{escaped_name}",
                    rf"{number}\s*{operator}\s*{escaped_name}\s*{operator}\s*{number}",
                )
                if any(re.fullmatch(pattern, feature_key.strip()) for pattern in patterns):
                    candidates.append(index)
            if len(candidates) == 1:
                return candidates[0]
            if len(candidates) > 1:
                raise ValueError(f"ambiguous attribution feature name {feature_key!r}")

        patterns = (
            r"feature[_\s]*(\d+)",
            r"feat[_\s]*(\d+)",
            r"f(\d+)",
            r"x(\d+)",
        )
        for pattern in patterns:
            match = re.fullmatch(pattern, feature_key, re.IGNORECASE)
            if match:
                index = int(match.group(1))
                if index >= n_features:
                    raise ValueError(f"attribution index {index} is outside {n_features} features")
                return index
        raise ValueError(f"cannot map attribution feature {feature_key!r} to an input index")

    for feature_key, value in attributions.items():
        if not isinstance(feature_key, str) or not feature_key:
            raise ValueError("attribution feature names must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("attribution values must be real numeric scalars")
        numeric_value = float(value)
        if not np.isfinite(numeric_value):
            raise ValueError("attribution values must be finite")
        index = resolve_index(feature_key)
        if index in assigned_indices:
            raise ValueError(f"multiple attribution entries map to feature index {index}")
        assigned_indices.add(index)
        attr_array[index] = numeric_value

    expected_indices = set(range(int(n_features)))
    if assigned_indices != expected_indices:
        raise ValueError(
            "feature_attributions must cover every input feature exactly; "
            f"missing indices={sorted(expected_indices - assigned_indices)}"
        )
    return attr_array


# =============================================================================
# Noisy Linear Imputation (helper for ROAD)
# =============================================================================


def _noisy_linear_impute(
    instance: np.ndarray,
    removed_indices: List[int],
    remaining_indices: List[int],
    background_data: np.ndarray,
    noise_scale: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Impute removed tabular features using a noisy fitted linear regression.

    This is a library-defined tabular operator, not the spatial Noisy Linear
    Imputation from ROAD.  ROAD solves a sparse system based on direct and
    diagonal image-pixel neighbours with fixed interpolation weights.  Here,
    for each removed feature j, a least-squares regression is fitted from all
    remaining tabular features:
        x_j = w^T * x_remaining + b + epsilon
    where epsilon ~ N(0, (noise_scale * sigma_residual)^2).

    Noise may make the imputation mask less obvious, but this function does not
    establish ROAD's Minimally Revealing Imputation condition.

    Args:
        instance: Input instance (1D array)
        removed_indices: Indices of features to impute
        remaining_indices: Indices of features to use as predictors
        background_data: Training data for fitting linear models (2D array)
        noise_scale: Scale factor for residual noise (default: 1.0).
            This controls noise magnitude; no monotone leakage guarantee is
            established for this adapted operator.
        seed: Random seed for reproducibility

    Returns:
        Imputed instance with removed features replaced by noisy linear predictions
    """
    instance = _as_feature_vector(instance)
    n_features = instance.size
    background_data = np.asarray(background_data, dtype=float)
    if (
        background_data.ndim != 2
        or background_data.shape[0] == 0
        or background_data.shape[1] != n_features
    ):
        raise ValueError(
            "background_data must be a non-empty 2D array with " f"{n_features} columns"
        )
    if not np.all(np.isfinite(background_data)):
        raise ValueError("background_data must contain only finite values")
    noise_scale = _validate_nonnegative_real(noise_scale, "noise_scale")
    if seed is not None:
        seed = _validate_nonnegative_int(seed, "seed")
    rng = np.random.default_rng(seed)

    removed_indices = list(removed_indices)
    remaining_indices = list(remaining_indices)
    all_indices = removed_indices + remaining_indices
    if any(isinstance(i, bool) or not isinstance(i, Integral) for i in all_indices):
        raise TypeError("feature indices must be integers")
    all_indices = [int(i) for i in all_indices]
    if any(i < 0 or i >= n_features for i in all_indices):
        raise ValueError("feature indices are outside the instance")
    if len(set(all_indices)) != len(all_indices):
        raise ValueError("removed_indices and remaining_indices must be disjoint")
    if set(all_indices) != set(range(n_features)):
        raise ValueError("removed_indices and remaining_indices must partition all features")

    imputed = instance.copy()

    if len(removed_indices) == 0:
        return imputed

    if len(remaining_indices) == 0:
        # No remaining features to predict from - use column means + noise
        for j in removed_indices:
            col_mean = np.mean(background_data[:, j])
            col_std = np.std(background_data[:, j])
            imputed[j] = col_mean + rng.normal(0, noise_scale * col_std)
        return imputed

    # Extract remaining features from background data and instance
    X_remaining = background_data[:, remaining_indices]
    x_remaining = instance[remaining_indices].reshape(1, -1)

    # For each removed feature, fit a linear model and impute with noise
    for j in removed_indices:
        y_target = background_data[:, j]

        # Fit linear regression: y_target = X_remaining @ w + b
        # Using least squares with intercept via augmented matrix
        n_samples = X_remaining.shape[0]
        X_aug = np.column_stack([X_remaining, np.ones(n_samples)])

        try:
            # Use least squares (handles rank-deficient cases)
            result = np.linalg.lstsq(X_aug, y_target, rcond=None)
            coeffs = result[0]

            # Predict for the instance
            x_aug = np.column_stack([x_remaining, np.ones(1)])
            predicted = float((x_aug @ coeffs).item())

            # Compute residual standard deviation
            y_pred_train = X_aug @ coeffs
            residuals = y_target - y_pred_train
            residual_std = np.std(residuals)

            # Add calibrated noise
            noise = rng.normal(0, noise_scale * max(residual_std, 1e-10))
            imputed[j] = predicted + noise

        except np.linalg.LinAlgError:
            # Fallback: use column mean + noise if linear fit fails
            col_mean = np.mean(y_target)
            col_std = np.std(y_target)
            imputed[j] = col_mean + rng.normal(0, noise_scale * max(col_std, 1e-10))

    return imputed


# =============================================================================
# Metric 10: local ROAD-inspired diagnostic (Rong et al., 2022)
# =============================================================================


def compute_road(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    background_data: np.ndarray,
    target_class: Optional[int] = None,
    percentages: Optional[List[float]] = None,
    order: str = "morf",
    noise_scale: float = 1.0,
    use_absolute: bool = True,
    seed: Optional[int] = None,
    return_details: bool = False,
) -> Union[
    float,
    Dict[str, Union[float, np.ndarray, List[float], List[int], str]],
]:
    """
    Compute a local, ROAD-inspired prediction-change diagnostic.

    This implementation uses the tabular regression imputer documented in
    :func:`_noisy_linear_impute` for one instance, then measures the fixed
    target output's change. It does not implement ROAD's spatial imputer or
    canonical dataset-level classification-accuracy evaluation. Its score must
    not be reported as ROAD or compared with published ROAD values.

    At each removal percentage p, the top-p% features (by attribution) are
    removed and replaced using Noisy Linear Imputation fitted on the
    background data. The model's prediction change on the imputed sample
    is recorded. The final score is the mean prediction change across
    all percentages.

    Two ordering strategies are available: MoRF removes the largest ranked
    features first and LeRF removes the smallest first. The returned value is
    the mean signed output drop, so larger values mean larger drops under that
    ordering; this alone is not a universal explanation-quality guarantee.

    The regression imputation operator fits a linear regression from
    remaining features to each removed feature using the training data, then
    adds Gaussian noise with standard deviation
    ``noise_scale * residual_std``. This local operator has no established
    information-leakage guarantee.

    Args:
        model: Model adapter with predict/predict_proba method
        instance: Input instance (1D array)
        explanation: Explanation object with feature_attributions
        background_data: Training data for fitting imputation models (2D array).
            Required for noisy linear imputation.
        target_class: Explicit output index. If omitted, resolve it from the
            explanation; ambiguous or conflicting multi-output identities fail.
        percentages: List of removal percentages as fractions in (0, 1).
            Default: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        order: Feature removal order:
            - "morf": Most Relevant First (descending attribution)
            - "lerf": Least Relevant First (ascending attribution)
        noise_scale: Scale factor for imputation noise (default: 1.0).
            Controls the amount of Gaussian noise added to linear predictions.
        use_absolute: If True, use a library-defined magnitude ranking;
            False ranks signed attribution values.
        seed: Random seed for reproducibility
        return_details: If True, return detailed results

    Returns:
        If return_details=False: Mean signed prediction change across
            percentages (float).
        If return_details=True: Dictionary with:
            - 'score': float - Mean prediction change across percentages
            - 'prediction_changes': np.ndarray - Change at each percentage
            - 'predictions': np.ndarray - Prediction at each percentage
            - 'percentages': list - Removal percentages used
            - 'n_removed': list - Number of features removed at each step
            - 'feature_order': np.ndarray - Order in which features are removed
            - 'order': str - Removal order used ('morf' or 'lerf')
            - 'original_prediction': float - Original prediction value

    References:
        Rong, Y., Leemann, T., Borisov, V., Kasneci, G., & Kasneci, E. (2022).
        A Consistent and Efficient Evaluation Strategy for Attribution Methods.
        Proceedings of the 39th International Conference on Machine Learning
        (ICML), PMLR 162, 18770-18795.
    """
    instance = _as_feature_vector(instance)
    n_features = len(instance)
    use_absolute = _validate_bool(use_absolute, "use_absolute")
    return_details = _validate_bool(return_details, "return_details")

    # Validate background_data
    if background_data is None:
        raise ValueError(
            "background_data is required for the local regression-imputation " "diagnostic"
        )
    background_data = np.asarray(background_data, dtype=float)
    if (
        background_data.ndim != 2
        or background_data.shape[0] == 0
        or background_data.shape[1] != n_features
    ):
        raise ValueError(
            f"background_data must be 2D with {n_features} columns and non-empty, "
            f"got shape {background_data.shape}."
        )
    if not np.all(np.isfinite(background_data)):
        raise ValueError("background_data must contain only finite values")
    noise_scale = _validate_nonnegative_real(noise_scale, "noise_scale")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("seed must be a non-negative integer or None")
        seed = int(seed)

    # Validate order
    if order not in ("morf", "lerf"):
        raise ValueError(f"order must be 'morf' or 'lerf', got '{order}'.")

    # Extract attributions as array
    attr_array = _extract_attribution_array(explanation, n_features)

    # Sort features by attribution
    if use_absolute:
        sort_values = np.abs(attr_array)
    else:
        sort_values = attr_array

    if order == "morf":
        # Most Relevant First: descending order
        sorted_indices = np.argsort(-sort_values, kind="stable")
    else:
        # Least Relevant First: ascending order
        sorted_indices = np.argsort(sort_values, kind="stable")

    # Default percentages
    if percentages is None:
        percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Validate percentages
    if isinstance(percentages, (str, bytes)) or not isinstance(percentages, Sequence):
        raise TypeError("percentages must be a non-empty sequence")
    validated_percentages: List[float] = []
    for percentage in percentages:
        if isinstance(percentage, bool) or not isinstance(percentage, Real):
            raise TypeError("each percentage must be a real scalar")
        percentage = float(percentage)
        if not np.isfinite(percentage) or not 0.0 < percentage < 1.0:
            raise ValueError("percentages must contain only finite values in (0, 1)")
        validated_percentages.append(percentage)
    if not validated_percentages:
        raise ValueError("percentages must not be empty")
    if len(set(validated_percentages)) != len(validated_percentages):
        raise ValueError("percentages must not contain duplicates")
    percentages = sorted(validated_percentages)

    # Resolve the explanation's declared output once, then keep it fixed.
    target_class = _resolve_target_output(model, instance, explanation, target_class)
    original_value = _get_target_class_prediction(model, instance.reshape(1, -1), target_class)

    # Evaluate at each removal percentage
    prediction_changes = []
    predictions = []
    n_removed_list = []

    for p in percentages:
        # Number of features to remove at this percentage
        n_remove = max(1, int(round(p * n_features)))
        n_remove = min(n_remove, n_features)
        n_removed_list.append(n_remove)

        # Determine removed and remaining feature indices
        removed_indices = sorted_indices[:n_remove].tolist()
        remaining_indices = sorted_indices[n_remove:].tolist()

        # Compute seed for this step (deterministic per percentage)
        step_seed = None
        if seed is not None:
            step_seed = seed + int(p * 1000)

        # Impute removed features using noisy linear imputation
        imputed = _noisy_linear_impute(
            instance,
            removed_indices,
            remaining_indices,
            background_data,
            noise_scale=noise_scale,
            seed=step_seed,
        )

        # Get prediction on imputed sample
        imputed_value = _get_target_class_prediction(model, imputed.reshape(1, -1), target_class)

        predictions.append(imputed_value)
        # Signed output drop (positive = tracked output decreased)
        prediction_changes.append(original_value - imputed_value)

    prediction_change_array = np.asarray(prediction_changes, dtype=float)
    prediction_array = np.asarray(predictions, dtype=float)

    # Score: mean prediction change across percentages
    score = float(np.mean(prediction_change_array))

    if return_details:
        return {
            "score": score,
            "prediction_changes": prediction_change_array,
            "predictions": prediction_array,
            "percentages": percentages,
            "n_removed": n_removed_list,
            "feature_order": sorted_indices,
            "order": order,
            "original_prediction": original_value,
        }

    return score


def compute_road_combined(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    background_data: np.ndarray,
    target_class: Optional[int] = None,
    percentages: Optional[List[float]] = None,
    noise_scale: float = 1.0,
    use_absolute: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Compute both orderings of the local ROAD-inspired diagnostic.

    The arithmetic gap is a library-defined summary, not a statistic defined
    or validated by Rong et al. Because the two orderings use separate noise
    streams, its magnitude has no universal interpretation.

    Args:
        model: Model adapter with predict/predict_proba method
        instance: Input instance (1D array)
        explanation: Explanation object with feature_attributions
        background_data: Training data for fitting imputation models
        target_class: Explicit output index. If omitted, resolve it from the
            explanation; ambiguous or conflicting multi-output identities fail.
        percentages: List of removal percentages as fractions in (0, 1)
        noise_scale: Scale factor for imputation noise
        use_absolute: If True, use the local magnitude-ranking variant;
            False ranks signed attribution values.
        seed: Random seed for reproducibility

    Returns:
        Dictionary with:
            - 'morf': float - local diagnostic in MoRF order
            - 'lerf': float - local diagnostic in LeRF order
            - 'gap': float - MoRF minus LeRF (library-defined)
            - 'scores': Dict[str, float] - Both scores by name
    """
    if seed is not None:
        seed = _validate_nonnegative_int(seed, "seed")
    morf_seed = seed
    lerf_seed = seed + 10000 if seed is not None else None

    morf_score = _require_scalar_result(
        compute_road(
            model,
            instance,
            explanation,
            background_data=background_data,
            target_class=target_class,
            percentages=percentages,
            order="morf",
            noise_scale=noise_scale,
            use_absolute=use_absolute,
            seed=morf_seed,
        ),
        "compute_road",
    )

    lerf_score = _require_scalar_result(
        compute_road(
            model,
            instance,
            explanation,
            background_data=background_data,
            target_class=target_class,
            percentages=percentages,
            order="lerf",
            noise_scale=noise_scale,
            use_absolute=use_absolute,
            seed=lerf_seed,
        ),
        "compute_road",
    )

    return {
        "morf": float(morf_score),
        "lerf": float(lerf_score),
        "gap": float(morf_score - lerf_score),
        "scores": {"morf": float(morf_score), "lerf": float(lerf_score)},
    }


def compute_batch_road(
    model,
    X: np.ndarray,
    explanations: List[Explanation],
    background_data: Optional[np.ndarray] = None,
    max_samples: Optional[int] = None,
    percentages: Optional[List[float]] = None,
    order: str = "morf",
    noise_scale: float = 1.0,
    use_absolute: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute the mean local ROAD-inspired diagnostic over paired instances.

    Args:
        model: Model adapter
        X: Input data (2D array)
        explanations: List of Explanation objects (one per instance)
        background_data: Explicit training/background data for imputation.
        max_samples: Maximum number of samples to evaluate
        percentages: Removal percentages (default: [0.1, ..., 0.9])
        order: Feature removal order ('morf' or 'lerf')
        noise_scale: Scale factor for imputation noise
        use_absolute: If True, use the local magnitude-ranking variant;
            False ranks signed attribution values.
        seed: Random seed for reproducibility

    Returns:
        Dictionary with mean, std, min, max, and count of evaluated scores
    """
    X, n_instances = _validate_batch_inputs(X, explanations, max_samples)
    if seed is not None:
        seed = _validate_nonnegative_int(seed, "seed")
    if background_data is None:
        raise ValueError("background_data is required for batch ROAD evaluation")

    scores = []
    for i in range(n_instances):
        current_seed = seed + i if seed is not None else None
        result = compute_road(
            model,
            X[i],
            explanations[i],
            background_data=background_data,
            percentages=percentages,
            order=order,
            noise_scale=noise_scale,
            use_absolute=use_absolute,
            seed=current_seed,
        )
        scores.append(_require_scalar_result(result, "compute_road"))
    return _summarize_scores(scores)


# =============================================================================
# Metric 11: Deletion AUC (Petsiuk et al., 2018)
# =============================================================================


def _get_target_class_prediction(
    model,
    instance_2d: np.ndarray,
    target_class: int,
) -> float:
    """
    Get the finite scalar model output for a specific target index.

    Handles both raw scikit-learn models and explainiverse adapters, including
    one-column binary probabilities represented as ``P(class 1)``.

    Args:
        model: Model with predict or predict_proba method
        instance_2d: Input instance reshaped to 2D (1, n_features)
        target_class: Target-output index to extract

    Returns:
        Scalar value for the target output (float)
    """
    if isinstance(target_class, bool) or not isinstance(target_class, Integral) or target_class < 0:
        raise ValueError("target_class must be a non-negative integer")
    value = get_prediction_value(
        model,
        instance_2d,
        target_class=int(target_class),
    )
    if not np.isfinite(value):
        raise ValueError("model prediction must be finite")
    return float(value)


def compute_deletion_auc(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    background_data: Optional[np.ndarray] = None,
    target_class: Optional[int] = None,
    use_absolute: bool = True,
    n_steps: Optional[int] = None,
    return_curve: bool = False,
) -> Union[float, Dict[str, Union[float, np.ndarray]]]:
    """
    Compute a one-dimensional ranked-feature Deletion AUC adaptation.

    Progressively removes features from the original input in order of
    decreasing ranked attribution, recording the model's fixed-target scalar
    output at each step. The Deletion
    AUC is integrated over the fraction of features removed. The original
    image method used black-pixel deletion; configurable numeric baselines and
    one-dimensional features are an adaptation. Values are comparable only
    when model-output and baseline conventions are held fixed.

    Under the Petsiuk deletion-test interpretation, a useful ranking causes a
    rapid drop in the tracked class probability, so lower AUC is preferred.
    That direction is conditional on the documented output, baseline, and
    perturbation convention.

    Both this function and the corrected Pixel-Flipping adaptation record raw
    fixed-target outputs. Their differences are the baseline/ranking contracts
    documented by their respective APIs.

    When ``n_steps`` is specified the metric uses percentage-based steps
    (removing a fraction of features at each step) rather than one feature
    at a time, which is useful for high-dimensional inputs.

    Args:
        model: Model adapter with predict/predict_proba method
        instance: Input instance (1D array)
        explanation: Explanation object with feature_attributions
        baseline: Baseline for feature removal ("mean", "median", scalar,
            array, or callable). Removed features are replaced with the
            corresponding baseline value.
        background_data: Reference data for computing baseline
            (required when ``baseline`` is "mean" or "median")
        target_class: Explicit output index whose score is tracked. If omitted,
            resolve it from the explanation; ambiguous or conflicting identities fail.
        use_absolute: If True (default), rank attribution magnitudes. This
            agrees with non-negative RISE saliency maps; for signed vector
            attributions it is a library-defined ranking convention. False
            ranks signed values.
        n_steps: Number of evenly spaced removal steps. If None, removes
            one feature at a time (n_steps = n_features). If specified,
            cumulative removal counts are spaced approximately evenly from
            zero through all features.
        return_curve: If True, return full curve details

    Returns:
        If return_curve=False:
            Deletion AUC (float). For probability outputs it lies in [0, 1].
            A smaller value means a lower tracked-output curve under this
            deletion intervention; it is not a general quality verdict.
        If return_curve=True:
            Dictionary with:
            - 'auc': float — Area under deletion curve
            - 'curve': np.ndarray — Fixed-target scalar output at each step
            - 'fractions': np.ndarray — Fraction of features removed at
              each step (from 0 to 1)
            - 'feature_order': np.ndarray — Feature indices in removal
              order
            - 'n_features': int — Total number of features
            - 'target_class': int — Class tracked
            - 'original_prediction': float — Prediction before any removal

    References:
        Petsiuk, V., Das, A., & Saenko, K. (2018). RISE: Randomized Input
        Sampling for Explanation of Black-box Models. Proceedings of the
        British Machine Vision Conference (BMVC).
    """
    instance = _as_feature_vector(instance)
    n_features = len(instance)
    use_absolute = _validate_bool(use_absolute, "use_absolute")
    return_curve = _validate_bool(return_curve, "return_curve")

    # Get baseline values
    baseline_values = _validated_baseline_values(baseline, background_data, n_features)

    # Extract and sort attributions
    attr_array = _extract_attribution_array(explanation, n_features)
    if use_absolute:
        sorted_indices = np.argsort(-np.abs(attr_array), kind="stable")
    else:
        sorted_indices = np.argsort(-attr_array, kind="stable")

    # Resolve the explanation's declared output once, then keep it fixed.
    target_class = _resolve_target_output(model, instance, explanation, target_class)

    # Determine removal schedule
    if n_steps is not None:
        # Percentage-based steps: remove features in chunks
        n_steps = _validate_positive_int(n_steps, "n_steps")
        if n_steps > n_features:
            raise ValueError("n_steps cannot exceed the number of features")
        step_sizes = np.round(np.linspace(0, n_features, n_steps + 1)).astype(int)
        # step_sizes[i] = cumulative number of features removed after step i
        removal_counts = step_sizes[1:]  # exclude the 0
    else:
        # One feature at a time
        removal_counts = np.arange(1, n_features + 1)

    # Get original prediction (step 0: no features removed)
    original_pred = _get_target_class_prediction(model, instance.reshape(1, -1), target_class)

    # Build the deletion curve
    predictions = [original_pred]
    fractions = [0.0]
    current = instance.copy()
    prev_count = 0

    for count in removal_counts:
        # Remove features from prev_count to count
        for idx in sorted_indices[prev_count:count]:
            current[idx] = baseline_values[idx]
        prev_count = count

        # Record prediction
        pred_val = _get_target_class_prediction(model, current.reshape(1, -1), target_class)
        predictions.append(pred_val)
        fractions.append(float(count) / float(n_features))

    prediction_array = np.asarray(predictions, dtype=float)
    fraction_array = np.asarray(fractions, dtype=float)

    # Compute AUC using trapezoidal rule
    # x-axis: fraction of features removed [0, 1]
    # y-axis: raw fixed-target output
    auc = float(_trapezoid(prediction_array, fraction_array))

    if return_curve:
        return {
            "auc": auc,
            "curve": prediction_array,
            "fractions": fraction_array,
            "feature_order": sorted_indices,
            "n_features": n_features,
            "target_class": target_class,
            "original_prediction": original_pred,
        }

    return auc


def compute_batch_deletion_auc(
    model,
    X: np.ndarray,
    explanations: List[Explanation],
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    max_samples: Optional[int] = None,
    use_absolute: bool = True,
    n_steps: Optional[int] = None,
    background_data: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute average Deletion AUC over a batch of instances.

    Args:
        model: Model adapter
        X: Input data (2D array, one row per instance)
        explanations: List of Explanation objects (one per instance)
        baseline: Baseline for feature removal
        max_samples: Maximum number of samples to evaluate
        use_absolute: If True, use the magnitude-ranking variant; False ranks
            signed attribution values.
        n_steps: Number of removal steps per instance (None = one per feature)

    Returns:
        Dictionary with mean, std, min, max, and n_samples of evaluated scores.
        Lower mean means faster degradation under the documented deletion
        convention.
    """
    X, n_samples = _validate_batch_inputs(X, explanations, max_samples)
    scores = []
    for i in range(n_samples):
        result = compute_deletion_auc(
            model,
            X[i],
            explanations[i],
            baseline=baseline,
            background_data=background_data,
            use_absolute=use_absolute,
            n_steps=n_steps,
        )
        scores.append(_require_scalar_result(result, "compute_deletion_auc"))
    return _summarize_scores(scores)


# =============================================================================
# Metric 12: Insertion AUC (Petsiuk et al., 2018)
# =============================================================================


def compute_insertion_auc(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    background_data: Optional[np.ndarray] = None,
    target_class: Optional[int] = None,
    use_absolute: bool = True,
    n_steps: Optional[int] = None,
    return_curve: bool = False,
) -> Union[float, Dict[str, Union[float, np.ndarray]]]:
    """
    Compute a one-dimensional ranked-feature Insertion AUC adaptation.

    Starts from a baseline input (all features at baseline values) and
    progressively inserts features in decreasing ranked-attribution order,
    recording the model's fixed-target scalar output at each step. The
    Insertion AUC is the Area Under
    this recovery curve. Petsiuk et al. start from a blurred image; the
    configurable numeric baseline used here is a general-vector adaptation.
    Values are comparable only under the same baseline and output conventions.

    Under the Petsiuk insertion-test interpretation, a useful ranking causes a
    rapid rise in the tracked class probability, so higher AUC is preferred.
    That direction is conditional on the documented output, baseline, and
    perturbation convention.

    The insertion and deletion curves probe opposite perturbation paths. They
    remain baseline-sensitive diagnostics and do not establish causal necessity
    or sufficiency by themselves.

    When ``n_steps`` is specified the metric uses percentage-based steps
    (inserting a fraction of features at each step) rather than one feature
    at a time.

    Args:
        model: Model adapter with predict/predict_proba method
        instance: Input instance (1D array)
        explanation: Explanation object with feature_attributions
        baseline: Baseline for the starting state ("mean", "median",
            scalar, array, or callable). The initial input is set entirely
            to baseline values.
        background_data: Reference data for computing baseline
            (required when ``baseline`` is "mean" or "median")
        target_class: Explicit output index whose score is tracked. If omitted,
            resolve it from the explanation; ambiguous or conflicting identities fail.
        use_absolute: If True (default), rank attribution magnitudes. This
            agrees with non-negative RISE saliency maps; for signed vector
            attributions it is a library-defined ranking convention. False
            ranks signed values.
        n_steps: Number of evenly spaced insertion steps. If None, inserts
            one feature at a time (n_steps = n_features). If specified,
            cumulative insertion counts are spaced approximately evenly from
            zero through all features.
        return_curve: If True, return full curve details

    Returns:
        If return_curve=False:
            Insertion AUC (float). For probability outputs it lies in [0, 1].
            A larger value means a higher tracked-output curve under this
            insertion intervention; it is not a general quality verdict.
        If return_curve=True:
            Dictionary with:
            - 'auc': float — Area under insertion curve
            - 'curve': np.ndarray — Fixed-target scalar output at each step
            - 'fractions': np.ndarray — Fraction of features inserted at
              each step (from 0 to 1)
            - 'feature_order': np.ndarray — Feature indices in insertion
              order
            - 'n_features': int — Total number of features
            - 'target_class': int — Class tracked
            - 'baseline_prediction': float — Prediction from baseline state
            - 'final_prediction': float — Prediction after all features
              inserted (should match original prediction)

    References:
        Petsiuk, V., Das, A., & Saenko, K. (2018). RISE: Randomized Input
        Sampling for Explanation of Black-box Models. Proceedings of the
        British Machine Vision Conference (BMVC).
    """
    instance = _as_feature_vector(instance)
    n_features = len(instance)
    use_absolute = _validate_bool(use_absolute, "use_absolute")
    return_curve = _validate_bool(return_curve, "return_curve")

    # Get baseline values
    baseline_values = _validated_baseline_values(baseline, background_data, n_features)

    # Extract and sort attributions
    attr_array = _extract_attribution_array(explanation, n_features)
    if use_absolute:
        sorted_indices = np.argsort(-np.abs(attr_array), kind="stable")
    else:
        sorted_indices = np.argsort(-attr_array, kind="stable")

    # Resolve the explanation's declared output once, then keep it fixed.
    target_class = _resolve_target_output(model, instance, explanation, target_class)

    # Determine insertion schedule
    if n_steps is not None:
        n_steps = _validate_positive_int(n_steps, "n_steps")
        if n_steps > n_features:
            raise ValueError("n_steps cannot exceed the number of features")
        step_sizes = np.round(np.linspace(0, n_features, n_steps + 1)).astype(int)
        insertion_counts = step_sizes[1:]  # cumulative features inserted
    else:
        insertion_counts = np.arange(1, n_features + 1)

    # Start from baseline (all features at baseline)
    current = baseline_values.copy()

    # Get baseline prediction (step 0: no features from original)
    baseline_pred = _get_target_class_prediction(model, current.reshape(1, -1), target_class)

    # Build the insertion curve
    predictions = [baseline_pred]
    fractions = [0.0]
    prev_count = 0

    for count in insertion_counts:
        # Insert features from prev_count to count
        for idx in sorted_indices[prev_count:count]:
            current[idx] = instance[idx]
        prev_count = count

        # Record prediction
        pred_val = _get_target_class_prediction(model, current.reshape(1, -1), target_class)
        predictions.append(pred_val)
        fractions.append(float(count) / float(n_features))

    prediction_array = np.asarray(predictions, dtype=float)
    fraction_array = np.asarray(fractions, dtype=float)

    # Compute AUC using trapezoidal rule
    # x-axis: fraction of features inserted [0, 1]
    # y-axis: raw fixed-target output
    auc = float(_trapezoid(prediction_array, fraction_array))

    if return_curve:
        return {
            "auc": auc,
            "curve": prediction_array,
            "fractions": fraction_array,
            "feature_order": sorted_indices,
            "n_features": n_features,
            "target_class": target_class,
            "baseline_prediction": baseline_pred,
            "final_prediction": float(prediction_array[-1]),
        }

    return auc


def compute_batch_insertion_auc(
    model,
    X: np.ndarray,
    explanations: List[Explanation],
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    max_samples: Optional[int] = None,
    use_absolute: bool = True,
    n_steps: Optional[int] = None,
    background_data: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute average Insertion AUC over a batch of instances.

    Args:
        model: Model adapter
        X: Input data (2D array, one row per instance)
        explanations: List of Explanation objects (one per instance)
        baseline: Baseline for feature insertion starting state
        max_samples: Maximum number of samples to evaluate
        use_absolute: If True, use the magnitude-ranking variant; False ranks
            signed attribution values.
        n_steps: Number of insertion steps per instance (None = one per feature)

    Returns:
        Dictionary with mean, std, min, max, and n_samples of evaluated scores.
        Higher mean means faster recovery under the documented insertion
        convention.
    """
    X, n_samples = _validate_batch_inputs(X, explanations, max_samples)
    scores = []
    for i in range(n_samples):
        result = compute_insertion_auc(
            model,
            X[i],
            explanations[i],
            baseline=baseline,
            background_data=background_data,
            use_absolute=use_absolute,
            n_steps=n_steps,
        )
        scores.append(_require_scalar_result(result, "compute_insertion_auc"))
    return _summarize_scores(scores)


# =============================================================================
# Combined Insertion-Deletion convenience function
# =============================================================================


def compute_insertion_deletion_auc(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    background_data: Optional[np.ndarray] = None,
    target_class: Optional[int] = None,
    use_absolute: bool = True,
    n_steps: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute both Insertion and Deletion AUC in a single call.

    The returned arithmetic difference is a library convenience. Petsiuk et
    al. define the individual insertion and deletion curves, not this combined
    scalar, so ``delta`` has no separately verified interpretation.

    Args:
        model: Model adapter with predict/predict_proba method
        instance: Input instance (1D array)
        explanation: Explanation object with feature_attributions
        baseline: Baseline for feature operations
        background_data: Reference data for computing baseline
        target_class: Explicit output index. If omitted, resolve it from the
            explanation; ambiguous or conflicting multi-output identities fail.
        use_absolute: If True, use the magnitude-ranking variant; False ranks
            signed attribution values.
        n_steps: Number of steps per curve (None = one per feature)

    Returns:
        Dictionary with:
        - 'insertion_auc': float — Insertion AUC (higher means faster recovery
          under the insertion convention)
        - 'deletion_auc': float — Deletion AUC (lower means faster degradation
          under the deletion convention)
        - 'delta': float — insertion_auc minus deletion_auc

    References:
        Petsiuk, V., Das, A., & Saenko, K. (2018). RISE: Randomized Input
        Sampling for Explanation of Black-box Models. BMVC.
    """
    insertion_result = compute_insertion_auc(
        model,
        instance,
        explanation,
        baseline=baseline,
        background_data=background_data,
        target_class=target_class,
        use_absolute=use_absolute,
        n_steps=n_steps,
    )
    deletion_result = compute_deletion_auc(
        model,
        instance,
        explanation,
        baseline=baseline,
        background_data=background_data,
        target_class=target_class,
        use_absolute=use_absolute,
        n_steps=n_steps,
    )

    insertion_auc = _require_scalar_result(insertion_result, "compute_insertion_auc")
    deletion_auc = _require_scalar_result(deletion_result, "compute_deletion_auc")
    return {
        "insertion_auc": insertion_auc,
        "deletion_auc": deletion_auc,
        "delta": insertion_auc - deletion_auc,
    }


# =============================================================================
# Metric 1: AIX360-style faithfulness correlation proxy
# =============================================================================


def compute_faithfulness_estimate(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    background_data: Optional[np.ndarray] = None,
    subset_size: Optional[int] = None,
    n_subsets: int = 100,
    seed: Optional[int] = None,
    target_class: Optional[int] = None,
) -> float:
    """
    Compute the AIX360-style feature-removal faithfulness correlation.

    For ``subset_size=1`` this is Pearson's correlation between each signed
    attribution and the fixed-target output drop when that feature is replaced
    by its baseline. This is algebraically equivalent to the proxy implemented
    by IBM AIX360 (which negates correlation with the post-removal output).
    AIX360 cites Alvarez-Melis & Jaakkola (2018), but that paper does not define
    this evaluation formula; the method is therefore not claimed as a metric
    from that paper.

    ``subset_size > 1`` is a library-defined sampled-subset extension. It must
    not be confused with Ancona et al.'s separately exposed Sensitivity-n.

    Args:
        model: Model adapter with predict/predict_proba method
        instance: Input instance (1D array)
        explanation: Explanation object with feature_attributions
        baseline: Baseline for feature replacement ("mean", "median", scalar, array, callable)
        background_data: Reference data for computing baseline (required for "mean"/"median")
        subset_size: Size of random subsets to perturb (default: 1 for single-feature)
        n_subsets: Number of random subsets to evaluate (used when subset_size > 1)
        seed: Random seed for reproducibility
        target_class: Optional numeric output index. It must agree with any
            target identity recorded by the explanation.

    Returns:
        Pearson correlation in [-1, 1]. Raises ``ValueError`` when correlation
        is undefined (for example, a constant attribution or effect vector).

    References:
        Arya, V., et al. (2019). One Explanation Does Not Fit All: A Toolkit
        and Taxonomy of AI Explainability Techniques. arXiv:1909.03012.
        Trusted-AI/AIX360, ``aix360.metrics.local_metrics.faithfulness_metric``.
    """
    if seed is not None:
        seed = _validate_nonnegative_int(seed, "seed")
    rng = np.random.default_rng(seed)

    instance = _as_feature_vector(instance)
    n_features = len(instance)

    # Get baseline values
    baseline_values = _validated_baseline_values(baseline, background_data, n_features)

    # Extract attributions as array
    attr_array = _extract_attribution_array(explanation, n_features)

    # Default subset_size is 1 (single-feature perturbation)
    if subset_size is None:
        subset_size = 1
    subset_size = _validate_positive_int(subset_size, "subset_size")
    if subset_size > n_features:
        raise ValueError("subset_size cannot exceed the number of features")
    n_subsets = _validate_positive_int(n_subsets, "n_subsets")

    target_class = _resolve_target_output(model, instance, explanation, target_class)
    original_value = _get_target_class_prediction(model, instance.reshape(1, -1), target_class)

    if subset_size == 1:
        # Single-feature perturbation: evaluate each feature individually
        prediction_changes = []
        attribution_values = []

        for i in range(n_features):
            # Perturb single feature
            perturbed = apply_feature_mask(instance, [i], baseline_values)
            perturbed_value = _get_target_class_prediction(
                model, perturbed.reshape(1, -1), target_class
            )

            prediction_changes.append(original_value - perturbed_value)
            attribution_values.append(attr_array[i])

        corr, _ = _defined_pearson(np.asarray(attribution_values), np.asarray(prediction_changes))
        return corr

    else:
        # Random subset perturbation
        prediction_changes = []
        attribution_sums = []

        for _ in range(n_subsets):
            # Sample random subset of features
            subset_indices = rng.choice(
                n_features, size=min(subset_size, n_features), replace=False
            )

            # Perturb subset
            perturbed = apply_feature_mask(instance, subset_indices.tolist(), baseline_values)

            perturbed_value = _get_target_class_prediction(
                model, perturbed.reshape(1, -1), target_class
            )
            change = original_value - perturbed_value

            # Sum of attributions in subset
            attr_sum = np.sum(attr_array[subset_indices])

            prediction_changes.append(change)
            attribution_sums.append(attr_sum)

        corr, _ = _defined_pearson(np.asarray(attribution_sums), np.asarray(prediction_changes))
        return corr


def compute_batch_faithfulness_estimate(
    model,
    X: np.ndarray,
    explanations: List[Explanation],
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    max_samples: Optional[int] = None,
    seed: Optional[int] = None,
    background_data: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute average Faithfulness Estimate over a batch of instances.

    Args:
        model: Model adapter
        X: Input data (2D array)
        explanations: List of Explanation objects (one per instance)
        baseline: Baseline for feature replacement
        max_samples: Maximum number of samples to evaluate
        seed: Random seed

    Returns:
        Dictionary with mean, std, min, max, and count of evaluated scores
    """
    X, n_samples = _validate_batch_inputs(X, explanations, max_samples)
    if seed is not None:
        seed = _validate_nonnegative_int(seed, "seed")
    scores = []
    for i in range(n_samples):
        current_seed = seed + i if seed is not None else None
        scores.append(
            compute_faithfulness_estimate(
                model,
                X[i],
                explanations[i],
                baseline=baseline,
                background_data=background_data,
                seed=current_seed,
            )
        )
    return _summarize_scores(scores)


# =============================================================================
# Metric 8: IROF - Iterative Removal of Features (Rieger & Hansen, 2020)
# =============================================================================


def compute_irof(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    background_data: Optional[np.ndarray] = None,
    target_class: Optional[int] = None,
    segment_size: Optional[int] = None,
    use_absolute: bool = True,
    return_details: bool = False,
) -> Union[float, Dict[str, Union[float, np.ndarray, List]]]:
    """
    Compute a one-dimensional, per-instance IROF adaptation.

    Canonical IROF segments natural images into superpixels and defines segment
    importance using mean L1 relevance. Its experiments retain positive
    evidence, replace segments with a dataset mean, normalize the class-score
    curve by the original score, compute area over that curve, and average over
    samples. This function instead groups adjacent entries of one numeric
    vector and returns one sample's normalized AOC. It must not be compared
    directly with published IROF values.

    The local AOC measures how quickly the normalized fixed-target output drops
    under this grouping and baseline convention. A larger value means a faster
    decline on this path, not a general proof of faithfulness.

    For tabular data, each feature can be treated as a segment (segment_size=1),
    or features can be grouped into larger segments.

    AOC = ∫₀¹ [1 - f(x_perturbed) / f(x)] d(fraction_removed)

    where the integral is computed using the trapezoidal rule over the
    normalized perturbation curve.

    Args:
        model: Model adapter with predict/predict_proba method
        instance: Input instance (1D array)
        explanation: Explanation object with feature_attributions
        baseline: Baseline for feature removal ("mean", "median", scalar, array, callable)
        background_data: Reference data for computing baseline (required for "mean"/"median")
        target_class: Explicit output index. If omitted, resolve it from the
            explanation; ambiguous or conflicting multi-output identities fail.
        segment_size: Number of features per segment. If None, defaults to 1
            (each feature is its own segment). For image-like data, this groups
            features into spatial regions.
        use_absolute: If True (default), rank by mean absolute (L1) relevance.
            False enables a signed-mean variant for this vector adaptation.
        return_details: If True, return detailed results including degradation curve

    Returns:
        If return_details=False: normalized local AOC (float)
        If return_details=True: Dictionary with:
            - 'aoc': float - Area Over the perturbation Curve
            - 'curve': np.ndarray - Normalized prediction drop at each step
            - 'predictions': np.ndarray - Raw predictions at each step
            - 'segment_order': list - Order in which segments were removed
            - 'segments': list - List of feature indices in each segment
            - 'segment_importance': np.ndarray - Aggregated importance per segment
            - 'n_segments': int - Number of segments
            - 'original_prediction': float - Original prediction value

    References:
        Rieger, L., & Hansen, L. K. (2020). IROF: A Low Resource Evaluation
        Metric for Explanation Methods. Workshop AI for Affordable Healthcare
        at ICLR 2020.
    """
    instance = _as_feature_vector(instance)
    n_features = len(instance)
    use_absolute = _validate_bool(use_absolute, "use_absolute")
    return_details = _validate_bool(return_details, "return_details")

    # Get baseline values
    baseline_values = _validated_baseline_values(baseline, background_data, n_features)

    # Extract attributions as array
    attr_array = _extract_attribution_array(explanation, n_features)

    # Determine segment size (default: 1 = each feature is a segment)
    if segment_size is None:
        segment_size = 1
    segment_size = _validate_positive_int(segment_size, "segment_size")
    if segment_size > n_features:
        raise ValueError("segment_size cannot exceed the number of features")

    # Create non-overlapping segments
    segments = []
    for start_idx in range(0, n_features, segment_size):
        end_idx = min(start_idx + segment_size, n_features)
        segments.append(list(range(start_idx, end_idx)))

    n_segments = len(segments)

    # IROF's segment equation uses mean L1 relevance. The signed branch is an
    # explicit option of this one-dimensional adaptation.
    segment_importance = np.zeros(n_segments)
    for i, segment in enumerate(segments):
        if use_absolute:
            segment_importance[i] = np.mean(np.abs(attr_array[segment]))
        else:
            segment_importance[i] = np.mean(attr_array[segment])

    # Sort segments by importance (descending - most important first)
    sorted_segment_indices = np.argsort(-segment_importance, kind="stable")

    # Resolve the explanation's declared output once, then keep it fixed.
    target_class = _resolve_target_output(model, instance, explanation, target_class)
    original_value = _get_target_class_prediction(model, instance.reshape(1, -1), target_class)
    if original_value == 0.0:
        raise ValueError("IROF normalization is undefined when f(x) is zero")

    # Start with original instance
    current = instance.copy()

    # Track predictions and prediction drops at each step
    # Step 0: no segments removed
    predictions = [original_value]

    # Iteratively remove segments (most important first)
    for segment_idx in sorted_segment_indices:
        segment = segments[segment_idx]

        # Remove features in this segment (replace with baseline)
        for feat_idx in segment:
            current[feat_idx] = baseline_values[feat_idx]

        # Get prediction
        current_pred = _get_target_class_prediction(model, current.reshape(1, -1), target_class)

        predictions.append(current_pred)

    prediction_array = np.asarray(predictions, dtype=float)
    normalised_predictions = prediction_array / original_value
    prediction_drops = 1.0 - normalised_predictions

    # Per-instance normalized area over the curve.
    x = np.linspace(0, 1, len(prediction_array))
    aoc = _trapezoid(prediction_drops, x)

    if return_details:
        return {
            "aoc": float(aoc),
            "curve": prediction_drops,
            "predictions": prediction_array,
            "normalised_predictions": normalised_predictions,
            "segment_order": sorted_segment_indices.tolist(),
            "segments": segments,
            "segment_importance": segment_importance,
            "n_segments": n_segments,
            "original_prediction": original_value,
        }

    return float(aoc)


def compute_irof_multi_segment(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    background_data: Optional[np.ndarray] = None,
    target_class: Optional[int] = None,
    segment_sizes: Optional[List[int]] = None,
    use_absolute: bool = True,
) -> Dict[str, Union[float, Dict[int, float], List[int]]]:
    """
    Compute IROF for multiple segment sizes and return average.

    This library-defined sensitivity analysis averages the local adaptation
    across group sizes. It is not part of canonical IROF and is not inherently
    more statistically robust.

    Args:
        model: Model adapter with predict/predict_proba method
        instance: Input instance (1D array)
        explanation: Explanation object with feature_attributions
        baseline: Baseline for feature removal
        background_data: Reference data for computing baseline
        target_class: Explicit output index. If omitted, resolve it from the
            explanation; ambiguous or conflicting multi-output identities fail.
        segment_sizes: List of segment sizes to evaluate. If None, uses
            [1, n//4, n//2] where n is the number of features.
        use_absolute: If True (default), rank by mean absolute (L1) relevance;
            False uses the signed-mean variant.

    Returns:
        Dictionary with:
            - 'mean': float - Average AOC across all segment sizes
            - 'scores': Dict[int, float] - AOC for each segment size
            - 'segment_sizes': List[int] - Segment sizes evaluated
    """
    instance = _as_feature_vector(instance)
    n_features = len(instance)

    # Determine segment sizes to evaluate
    if segment_sizes is None:
        segment_sizes = [
            1,
            max(1, n_features // 4),
            max(1, n_features // 2),
        ]
        # Remove duplicates and sort
        segment_sizes = sorted(set(segment_sizes))

    if isinstance(segment_sizes, (str, bytes)) or not isinstance(segment_sizes, Sequence):
        raise TypeError("segment_sizes must be a non-empty sequence")
    if not segment_sizes:
        raise ValueError("segment_sizes must not be empty")
    validated_segment_sizes = []
    for segment_size in segment_sizes:
        segment_size = _validate_positive_int(segment_size, "segment_sizes item")
        if segment_size > n_features:
            raise ValueError("segment_sizes items cannot exceed the number of features")
        validated_segment_sizes.append(segment_size)
    if len(set(validated_segment_sizes)) != len(validated_segment_sizes):
        raise ValueError("segment_sizes must not contain duplicates")
    segment_sizes = validated_segment_sizes

    scores = {}
    for seg_size in segment_sizes:
        score = _require_scalar_result(
            compute_irof(
                model,
                instance,
                explanation,
                baseline=baseline,
                background_data=background_data,
                target_class=target_class,
                segment_size=seg_size,
                use_absolute=use_absolute,
            ),
            "compute_irof",
        )
        scores[seg_size] = score

    mean_score = np.mean(list(scores.values()))

    return {
        "mean": float(mean_score),
        "scores": scores,
        "segment_sizes": segment_sizes,
    }


def compute_batch_irof(
    model,
    X: np.ndarray,
    explanations: List[Explanation],
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    max_samples: Optional[int] = None,
    segment_size: Optional[int] = None,
    use_absolute: bool = True,
    background_data: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute average IROF score over a batch of instances.

    Args:
        model: Model adapter
        X: Input data (2D array)
        explanations: List of Explanation objects (one per instance)
        baseline: Baseline for feature removal
        max_samples: Maximum number of samples to evaluate
        segment_size: Number of features per segment (default: 1)
        use_absolute: If True (default), rank by mean absolute (L1) relevance;
            False uses the signed-mean variant.

    Returns:
        Dictionary with mean, std, min, max, and count of evaluated scores
    """
    X, n_samples = _validate_batch_inputs(X, explanations, max_samples)
    scores = []
    for i in range(n_samples):
        result = compute_irof(
            model,
            X[i],
            explanations[i],
            baseline=baseline,
            background_data=background_data,
            segment_size=segment_size,
            use_absolute=use_absolute,
        )
        scores.append(_require_scalar_result(result, "compute_irof"))
    return _summarize_scores(scores)


# =============================================================================
# Metric 9: Infidelity (Yeh et al., 2019)
# =============================================================================


def compute_infidelity(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    background_data: Optional[np.ndarray] = None,
    target_class: Optional[int] = None,
    perturbation_type: str = "gaussian",
    noise_scale: float = 0.1,
    n_samples: int = 100,
    subset_size: Optional[int] = None,
    seed: Optional[int] = None,
    return_details: bool = False,
) -> Union[float, Dict[str, Union[float, np.ndarray, str]]]:
    """
    Compute Infidelity score (Yeh et al., 2019).

    Infidelity measures agreement between an attribution-based linear change
    estimate and the model's actual change under a selected perturbation
    distribution. Its evaluation premise prefers smaller disagreement; that
    premise is specific to this diagnostic and perturbation choice.

    The metric computes:

        INFD(φ, f, x) = E_{I ~ μ}[(φ(x)ᵀ · I - (f(x) - f(x - I)))²]

    where:
    - φ(x) are the attributions (explanation)
    - I is a perturbation vector sampled from distribution μ
    - f(x) is the model output for the target class
    - The expectation is estimated via Monte Carlo sampling

    The intuition is that if attributions correctly identify feature importance,
    then the dot product φ(x)ᵀ · I (expected prediction change based on explanation)
    should match the actual prediction change f(x) - f(x - I).

    The attribution vector must have units compatible with an inner product
    against input-space perturbations. The paper's released image evaluation
    also applies an optimal scalar calibration before scoring; this API does
    not, and its vector perturbations are not benchmark-equivalent to that code.

    Lower infidelity is better for the selected perturbation distribution;
    zero is the minimum possible squared-error value.

    Three numeric-vector perturbation distributions are supported:
    - "gaussian": Continuous Gaussian noise I ~ N(0, σ²I)
    - "square": legacy name for an independent Bernoulli feature mask; it is
      not the square image-patch perturbation in the reference implementation
    - "subset": Random subset of features are perturbed

    Args:
        model: Model adapter with predict/predict_proba method
        instance: Input instance (1D array)
        explanation: Explanation object with feature_attributions
        baseline: Baseline for perturbation ("mean", "median", scalar, array, callable).
            For masked coordinates, ``I = x - baseline`` so that ``x - I``
            equals the baseline.
        background_data: Reference data for computing baseline (required for "mean"/"median")
        target_class: Explicit output index. If omitted, resolve it from the
            explanation; ambiguous or conflicting multi-output identities fail.
        perturbation_type: Type of perturbation distribution:
            - "gaussian": Gaussian noise scaled by noise_scale
            - "square": Binary mask perturbation (features replaced with baseline)
            - "subset": Random subset of features perturbed to baseline
        noise_scale: Gaussian standard deviation, or (for ``"square"``) the
            Bernoulli probability of replacing each feature. It is unused by
            the fixed-cardinality ``"subset"`` distribution.
        n_samples: Number of Monte Carlo samples for expectation (default: 100)
        subset_size: For "subset" perturbation, number of features to perturb.
            If None, defaults to max(1, n_features // 4)
        seed: Random seed for reproducibility
        return_details: If True, return detailed results

    Returns:
        If return_details=False: Infidelity mean-squared residual. Smaller
            values indicate closer fit under the selected perturbation
            distribution, and zero is the minimum.
        If return_details=True: Dictionary with:
            - 'infidelity': float - Mean squared error
            - 'squared_errors': np.ndarray - Squared error for each sample
            - 'expected_changes': np.ndarray - φ(x)ᵀ · I for each sample
            - 'actual_changes': np.ndarray - f(x) - f(x-I) for each sample
            - 'n_samples': int - Number of Monte Carlo samples
            - 'perturbation_type': str - Type of perturbation used

    References:
        Yeh, C. K., Hsieh, C. Y., Suggala, A. S., Inouye, D. I., &
        Ravikumar, P. (2019). On the (In)fidelity and Sensitivity of
        Explanations. NeurIPS 2019.
    """
    if perturbation_type not in {"gaussian", "square", "subset"}:
        raise ValueError(
            f"Unknown perturbation_type: {perturbation_type}. "
            "Choose from 'gaussian', 'square', 'subset'."
        )
    if seed is not None:
        seed = _validate_nonnegative_int(seed, "seed")
    rng = np.random.default_rng(seed)
    n_samples = _validate_positive_int(n_samples, "n_samples")
    noise_scale = _validate_nonnegative_real(noise_scale, "noise_scale")
    if perturbation_type == "square" and noise_scale > 1.0:
        raise ValueError("noise_scale must lie in [0, 1] for 'square'")
    return_details = _validate_bool(return_details, "return_details")

    instance = _as_feature_vector(instance)
    n_features = len(instance)

    # Get baseline values
    if perturbation_type in {"square", "subset"}:
        baseline_values = _validated_baseline_values(baseline, background_data, n_features)
    else:
        baseline_values = np.zeros(n_features, dtype=float)

    # Extract attributions as array
    attr_array = _extract_attribution_array(explanation, n_features)

    # Resolve the explanation's declared output once, then keep it fixed.
    target_class = _resolve_target_output(model, instance, explanation, target_class)
    original_value = _get_target_class_prediction(model, instance.reshape(1, -1), target_class)

    # Determine subset size for subset perturbation
    if subset_size is None:
        subset_size = max(1, n_features // 4)
    subset_size = _validate_positive_int(subset_size, "subset_size")
    if subset_size > n_features:
        raise ValueError("subset_size cannot exceed the number of features")

    # Monte Carlo sampling for expectation
    squared_errors = []
    expected_changes = []
    actual_changes = []

    for _ in range(n_samples):
        if perturbation_type == "gaussian":
            # Gaussian perturbation: I ~ N(0, σ²I)
            # For continuous perturbation, we add noise directly
            perturbation = rng.normal(0, noise_scale, n_features)
            perturbed = instance - perturbation

            # Expected change: φ(x)ᵀ · I
            expected_change = np.dot(attr_array, perturbation)

        elif perturbation_type == "square":
            # Square/binary mask perturbation
            # Each feature has probability noise_scale of being perturbed
            mask = rng.random(n_features) < noise_scale

            # Perturbation vector: difference between original and baseline for masked features
            perturbation = np.zeros(n_features)
            perturbed = instance.copy()

            for i in range(n_features):
                if mask[i]:
                    perturbation[i] = instance[i] - baseline_values[i]
                    perturbed[i] = baseline_values[i]

            # Expected change: φ(x)ᵀ · I (using the actual perturbation magnitude)
            expected_change = np.dot(attr_array, perturbation)

        elif perturbation_type == "subset":
            # Random subset perturbation
            subset_indices = rng.choice(n_features, size=subset_size, replace=False)

            # Perturbation vector: 1 for perturbed features, 0 otherwise
            # Use the actual applied value difference in the defining dot product.
            perturbation = np.zeros(n_features)
            perturbed = instance.copy()

            for idx in subset_indices:
                perturbation[idx] = instance[idx] - baseline_values[idx]
                perturbed[idx] = baseline_values[idx]

            # Expected change: φ(x)ᵀ · I
            expected_change = np.dot(attr_array, perturbation)

        # Get perturbed prediction
        perturbed_value = _get_target_class_prediction(
            model, perturbed.reshape(1, -1), target_class
        )

        # Actual change: f(x) - f(x - I)
        actual_change = original_value - perturbed_value

        # Squared error: (expected - actual)²
        sq_error = (expected_change - actual_change) ** 2

        squared_errors.append(sq_error)
        expected_changes.append(expected_change)
        actual_changes.append(actual_change)

    squared_error_array = np.asarray(squared_errors, dtype=float)
    expected_change_array = np.asarray(expected_changes, dtype=float)
    actual_change_array = np.asarray(actual_changes, dtype=float)

    # Infidelity is the mean squared error
    infidelity = float(np.mean(squared_error_array))
    if not np.isfinite(infidelity):
        raise ValueError("infidelity is non-finite")

    if return_details:
        return {
            "infidelity": infidelity,
            "squared_errors": squared_error_array,
            "expected_changes": expected_change_array,
            "actual_changes": actual_change_array,
            "n_samples": n_samples,
            "perturbation_type": perturbation_type,
            "original_prediction": original_value,
        }

    return infidelity


def compute_infidelity_multi_perturbation(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    background_data: Optional[np.ndarray] = None,
    target_class: Optional[int] = None,
    perturbation_types: Optional[List[str]] = None,
    noise_scale: float = 0.1,
    n_samples: int = 100,
    seed: Optional[int] = None,
) -> Dict[str, Union[float, Dict[str, float], List[str]]]:
    """
    Compute Infidelity separately for several perturbation distributions.

    Yeh et al. define Infidelity relative to one chosen perturbation
    distribution. The returned arithmetic mean across distributions is a
    library convenience and is not a canonical, scale-invariant, or
    necessarily more robust statistic. Inspect the per-distribution scores.

    Args:
        model: Model adapter with predict/predict_proba method
        instance: Input instance (1D array)
        explanation: Explanation object with feature_attributions
        baseline: Baseline for perturbation
        background_data: Reference data for computing baseline
        target_class: Target class index for probability
        perturbation_types: List of perturbation types to evaluate.
            If None, uses ["gaussian", "square", "subset"]
        noise_scale: Standard deviation/probability for perturbations
        n_samples: Number of Monte Carlo samples per perturbation type
        seed: Random seed for reproducibility

    Returns:
        Dictionary with:
            - 'mean': float - Average infidelity across perturbation types
            - 'scores': Dict[str, float] - Infidelity for each perturbation type
            - 'perturbation_types': List[str] - Types evaluated
    """
    if perturbation_types is None:
        perturbation_types = ["gaussian", "square", "subset"]
    if isinstance(perturbation_types, (str, bytes)) or not isinstance(perturbation_types, Sequence):
        raise TypeError("perturbation_types must be a non-empty sequence")
    if not perturbation_types:
        raise ValueError("perturbation_types must not be empty")
    if any(
        perturbation_type not in {"gaussian", "square", "subset"}
        for perturbation_type in perturbation_types
    ):
        raise ValueError("perturbation_types may contain only 'gaussian', 'square', and 'subset'")
    if len(set(perturbation_types)) != len(perturbation_types):
        raise ValueError("perturbation_types must not contain duplicates")
    if seed is not None:
        seed = _validate_nonnegative_int(seed, "seed")

    scores = {}
    for i, ptype in enumerate(perturbation_types):
        current_seed = seed + i if seed is not None else None
        score = _require_scalar_result(
            compute_infidelity(
                model,
                instance,
                explanation,
                baseline=baseline,
                background_data=background_data,
                target_class=target_class,
                perturbation_type=ptype,
                noise_scale=noise_scale,
                n_samples=n_samples,
                seed=current_seed,
            ),
            "compute_infidelity",
        )
        scores[ptype] = score

    mean_score = float(np.mean(list(scores.values())))
    if not np.isfinite(mean_score):
        raise ValueError("multi-perturbation mean is non-finite")

    return {
        "mean": mean_score,
        "scores": scores,
        "perturbation_types": perturbation_types,
    }


def compute_batch_infidelity(
    model,
    X: np.ndarray,
    explanations: List[Explanation],
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    max_samples: Optional[int] = None,
    perturbation_type: str = "gaussian",
    noise_scale: float = 0.1,
    n_perturbations: int = 100,
    seed: Optional[int] = None,
    background_data: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute average Infidelity score over a batch of instances.

    Args:
        model: Model adapter
        X: Input data (2D array)
        explanations: List of Explanation objects (one per instance)
        baseline: Baseline for perturbation
        max_samples: Maximum number of samples to evaluate
        perturbation_type: Type of perturbation ("gaussian", "square", "subset")
        noise_scale: Perturbation scale parameter
        n_perturbations: Number of Monte Carlo samples per instance
        seed: Random seed for reproducibility

    Returns:
        Dictionary with mean, std, min, max, and count of evaluated scores
    """
    X, n_instances = _validate_batch_inputs(X, explanations, max_samples)
    if (isinstance(baseline, str) or callable(baseline)) and background_data is None:
        raise ValueError("background_data is required for statistical or callable batch baselines")
    if seed is not None:
        seed = _validate_nonnegative_int(seed, "seed")
    scores = []
    for i in range(n_instances):
        current_seed = seed + i if seed is not None else None
        result = compute_infidelity(
            model,
            X[i],
            explanations[i],
            baseline=baseline,
            background_data=background_data,
            perturbation_type=perturbation_type,
            noise_scale=noise_scale,
            n_samples=n_perturbations,
            seed=current_seed,
        )
        scores.append(_require_scalar_result(result, "compute_infidelity"))
    return _summarize_scores(scores)


# =============================================================================
# Metric 6: Selectivity (Montavon et al., 2018)
# =============================================================================


def compute_selectivity(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    background_data: Optional[np.ndarray] = None,
    target_class: Optional[int] = None,
    n_steps: Optional[int] = None,
    use_absolute: bool = True,
    return_details: bool = False,
) -> Union[float, Dict[str, Union[float, np.ndarray]]]:
    """
    Compute a one-dimensional, per-instance AOPC selectivity proxy.

    Measures how quickly a fixed scalar model output drops when removing features
    with the highest attributed values. Computed as the Area Over the
    Perturbation Curve (AOPC), which is the average prediction drop across
    all perturbation steps.

    AOPC = (1/(K+1)) * Σₖ₌₀ᴷ [f(x) - f(x_{1..k})]

    where:
    - f(x) is the original prediction for the target class
    - f(x_{1..k}) is the prediction after removing the top-k most important features
    - K is the total number of perturbation steps (default: n_features)

    Samek et al.'s AOPC averages these drops over a dataset and uses image
    regions with a specified perturbation operator. Montavon et al. describe
    selectivity/pixel-flipping as the AUC of recorded function values. This
    function's scalar, baseline-replacement result is therefore a local numeric
    adaptation, not a published benchmark score. Larger values mean a larger
    average drop under the chosen perturbation convention; drops may be
    negative for non-monotone models.

    Args:
        model: Model adapter with predict/predict_proba method
        instance: Input instance (1D array)
        explanation: Explanation object with feature_attributions
        baseline: Baseline for feature removal ("mean", "median", scalar, array, callable)
        background_data: Reference data for computing baseline (required for "mean"/"median")
        target_class: Explicit output index. If omitted, resolve it from the
            explanation; ambiguous or conflicting multi-output identities fail.
        n_steps: Number of perturbation steps (default: n_features, max features to remove)
        use_absolute: If True, use a library-defined magnitude ranking;
            False ranks signed attribution values.
        return_details: If True, return detailed results including prediction drops per step

    Returns:
        If return_details=False: local mean signed output drop (float)
        If return_details=True: Dictionary with:
            - 'aopc': float - Area Over the Perturbation Curve (average drop)
            - 'prediction_drops': np.ndarray - Drop at each step [f(x) - f(x_{1..k})]
            - 'predictions': np.ndarray - Predictions at each step
            - 'feature_order': np.ndarray - Order in which features were removed
            - 'n_steps': int - Number of perturbation steps

    References:
        Montavon, G., Samek, W., & Müller, K. R. (2018). Methods for Interpreting
        and Understanding Deep Neural Networks. Digital Signal Processing, 73, 1-15.

        Samek, W., Binder, A., Montavon, G., Lapuschkin, S., & Müller, K. R. (2016).
        Evaluating the Visualization of What a Deep Neural Network has Learned.
        IEEE Transactions on Neural Networks and Learning Systems, 28(11), 2660-2673.
    """
    instance = _as_feature_vector(instance)
    n_features = len(instance)
    use_absolute = _validate_bool(use_absolute, "use_absolute")
    return_details = _validate_bool(return_details, "return_details")

    # Get baseline values
    baseline_values = _validated_baseline_values(baseline, background_data, n_features)

    # Extract attributions as array
    attr_array = _extract_attribution_array(explanation, n_features)

    # Determine number of steps (default: all features)
    if n_steps is None:
        n_steps = n_features
    n_steps = _validate_positive_int(n_steps, "n_steps")
    if n_steps > n_features:
        raise ValueError("n_steps cannot exceed the number of features")

    # Sort features by attribution (descending - most important first)
    if use_absolute:
        sorted_indices = np.argsort(-np.abs(attr_array), kind="stable")
    else:
        sorted_indices = np.argsort(-attr_array, kind="stable")

    # Resolve the explanation's declared output once, then keep it fixed.
    target_class = _resolve_target_output(model, instance, explanation, target_class)
    original_value = _get_target_class_prediction(model, instance.reshape(1, -1), target_class)

    # Start with original instance
    current = instance.copy()

    # Track predictions and drops at each step
    # Step 0: no features removed (drop = 0)
    predictions = [original_value]
    prediction_drops = [0.0]  # f(x) - f(x) = 0

    # Remove features one by one (most important first)
    for k in range(n_steps):
        idx = sorted_indices[k]
        # Remove this feature (replace with baseline)
        current[idx] = baseline_values[idx]

        # Get prediction
        current_pred = _get_target_class_prediction(model, current.reshape(1, -1), target_class)

        predictions.append(current_pred)
        # Prediction drop: f(x) - f(x_{1..k})
        prediction_drops.append(original_value - current_pred)

    prediction_array = np.asarray(predictions, dtype=float)
    prediction_drop_array = np.asarray(prediction_drops, dtype=float)

    # Compute AOPC: average of prediction drops across all steps
    # AOPC = (1/(K+1)) * Σₖ₌₀ᴷ [f(x) - f(x_{1..k})]
    aopc = np.mean(prediction_drop_array)

    if return_details:
        return {
            "aopc": float(aopc),
            "prediction_drops": prediction_drop_array,
            "predictions": prediction_array,
            "feature_order": sorted_indices[:n_steps],
            "n_steps": n_steps,
            "original_prediction": original_value,
        }

    return float(aopc)


def compute_batch_selectivity(
    model,
    X: np.ndarray,
    explanations: List[Explanation],
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    max_samples: Optional[int] = None,
    n_steps: Optional[int] = None,
    use_absolute: bool = True,
    background_data: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute average Selectivity (AOPC) over a batch of instances.

    Args:
        model: Model adapter
        X: Input data (2D array)
        explanations: List of Explanation objects (one per instance)
        baseline: Baseline for feature removal
        max_samples: Maximum number of samples to evaluate
        n_steps: Number of perturbation steps per instance
        use_absolute: If True, use the local magnitude-ranking variant;
            False ranks signed attribution values.

    Returns:
        Dictionary with mean, std, min, max, and count of evaluated scores
    """
    X, n_samples = _validate_batch_inputs(X, explanations, max_samples)
    scores = []
    for i in range(n_samples):
        result = compute_selectivity(
            model,
            X[i],
            explanations[i],
            baseline=baseline,
            background_data=background_data,
            n_steps=n_steps,
            use_absolute=use_absolute,
        )
        scores.append(_require_scalar_result(result, "compute_selectivity"))
    return _summarize_scores(scores)


# =============================================================================
# Metric 7: Sensitivity-n (Ancona et al., 2018)
# =============================================================================


def compute_sensitivity_n(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    background_data: Optional[np.ndarray] = None,
    target_class: Optional[int] = None,
    n: Optional[int] = None,
    n_subsets: int = 100,
    use_absolute: bool = False,
    seed: Optional[int] = None,
    return_details: bool = False,
) -> Union[float, Dict[str, Union[float, np.ndarray, List]]]:
    """
    Compute Sensitivity-n score (Ancona et al., 2018).

    Measures the correlation between the sum of attributions for a subset
    of n features and the prediction change when those features are removed.
    Ancona et al. define the property with signed attributions and the signed
    variation in a target pre-softmax activation, using a zero baseline. This
    function estimates the same Pearson-correlation equation for the fixed
    scalar output exposed by ``model`` and a caller-selected numeric baseline.
    Probability outputs and nonzero baselines are explicit adaptations.

    For a random subset S of size n:
    - Sum of attributions: Σᵢ∈S aᵢ
    - Prediction change: f(x) - f(x_S) where x_S has features in S removed

    The metric computes Pearson correlation across many random subsets.

    Args:
        model: Model adapter with predict/predict_proba method
        instance: Input instance (1D array)
        explanation: Explanation object with feature_attributions
        baseline: Baseline for feature removal ("mean", "median", scalar, array, callable)
        background_data: Reference data for computing baseline (required for "mean"/"median")
        target_class: Explicit output index. If omitted, resolve it from the
            explanation; ambiguous or conflicting multi-output identities fail.
        n: Subset size. If None, defaults to max(1, n_features // 4)
        n_subsets: Number of random subsets to sample (default: 100)
        use_absolute: If False (default), sum signed attributions as in the
            paper. True enables a non-canonical magnitude variant.
        seed: Random seed for reproducibility
        return_details: If True, return detailed results including all subset data

    Returns:
        If return_details=False: Sensitivity-n Pearson correlation in [-1, 1].
            Larger values mean stronger positive association under the sampled
            subset and baseline contract, not universally better explanations.
        If return_details=True: Dictionary with:
            - 'correlation': float - Pearson correlation coefficient
            - 'p_value': float - p-value of the correlation
            - 'attribution_sums': np.ndarray - Sum of attributions for each subset
            - 'prediction_drops': np.ndarray - Prediction drop for each subset
            - 'subsets': list - List of subset indices sampled
            - 'n': int - Subset size used
            - 'n_subsets': int - Number of subsets sampled

    References:
        Ancona, M., Ceolini, E., Öztireli, C., & Gross, M. (2018). Towards Better
        Understanding of Gradient-based Attribution Methods for Deep Neural Networks.
        ICLR 2018.
    """
    if seed is not None:
        seed = _validate_nonnegative_int(seed, "seed")
    rng = np.random.default_rng(seed)
    n_subsets = _validate_positive_int(n_subsets, "n_subsets")
    if n_subsets < 2:
        raise ValueError("n_subsets must be at least 2 for correlation")
    use_absolute = _validate_bool(use_absolute, "use_absolute")
    return_details = _validate_bool(return_details, "return_details")

    instance = _as_feature_vector(instance)
    n_features = len(instance)

    # Get baseline values
    baseline_values = _validated_baseline_values(baseline, background_data, n_features)

    # Extract attributions as array
    attr_array = _extract_attribution_array(explanation, n_features)

    # Determine subset size
    if n is None:
        n = max(1, n_features // 4)
    n = _validate_positive_int(n, "n")
    if n > n_features:
        raise ValueError("n cannot exceed the number of features")

    # Resolve the explanation's declared output once, then keep it fixed.
    target_class = _resolve_target_output(model, instance, explanation, target_class)
    original_value = _get_target_class_prediction(model, instance.reshape(1, -1), target_class)

    # Sample random subsets and compute correlations
    attribution_sums = []
    prediction_drops = []
    subsets = []

    for _ in range(n_subsets):
        # Sample random subset of size n
        subset = rng.choice(n_features, size=n, replace=False)
        subsets.append(subset.tolist())

        # Compute sum of attributions in subset
        if use_absolute:
            attr_sum = np.sum(np.abs(attr_array[subset]))
        else:
            attr_sum = np.sum(attr_array[subset])
        attribution_sums.append(attr_sum)

        # Create perturbed instance with subset features removed
        perturbed = instance.copy()
        for idx in subset:
            perturbed[idx] = baseline_values[idx]

        # Get prediction for perturbed instance
        perturbed_value = _get_target_class_prediction(
            model, perturbed.reshape(1, -1), target_class
        )

        # Prediction drop (positive = removing features decreased prediction)
        drop = original_value - perturbed_value
        prediction_drops.append(drop)

    attribution_sum_array = np.asarray(attribution_sums, dtype=float)
    prediction_drop_array = np.asarray(prediction_drops, dtype=float)

    corr, p_value = _defined_pearson(attribution_sum_array, prediction_drop_array)

    if return_details:
        return {
            "correlation": float(corr),
            "p_value": float(p_value),
            "attribution_sums": attribution_sum_array,
            "prediction_drops": prediction_drop_array,
            "subsets": subsets,
            "n": n,
            "n_subsets": len(subsets),
        }

    return float(corr)


def compute_sensitivity_n_multi(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    background_data: Optional[np.ndarray] = None,
    target_class: Optional[int] = None,
    n_values: Optional[List[int]] = None,
    n_subsets: int = 100,
    use_absolute: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, Union[float, Dict[int, float], List[int]]]:
    """
    Compute Sensitivity-n for multiple subset sizes and return average.

    This arithmetic mean across subset sizes is a library-defined convenience,
    not a recommendation or aggregate defined by Ancona et al.

    Args:
        model: Model adapter with predict/predict_proba method
        instance: Input instance (1D array)
        explanation: Explanation object with feature_attributions
        baseline: Baseline for feature removal
        background_data: Reference data for computing baseline
        target_class: Explicit output index. If omitted, resolve it from the
            explanation; ambiguous or conflicting multi-output identities fail.
        n_values: List of subset sizes to evaluate. If None, uses [1, n//4, n//2, 3n//4]
        n_subsets: Number of random subsets per n value (default: 100)
        use_absolute: If False (default), sum signed attributions as in the
            paper; True uses the non-canonical magnitude variant.
        seed: Random seed for reproducibility

    Returns:
        Dictionary with:
            - 'mean': float - Average correlation across all n values
            - 'scores': Dict[int, float] - Correlation for each n value
            - 'n_values': List[int] - Subset sizes evaluated
    """
    instance = _as_feature_vector(instance)
    n_features = len(instance)

    # Determine n values to evaluate
    if n_values is None:
        n_values = [
            1,
            max(1, n_features // 4),
            max(1, n_features // 2),
            max(1, 3 * n_features // 4),
        ]
        # Remove duplicates and sort
        n_values = sorted(set(n_values))

    if isinstance(n_values, (str, bytes)) or not isinstance(n_values, Sequence):
        raise TypeError("n_values must be a non-empty sequence")
    if not n_values:
        raise ValueError("n_values must not be empty")
    validated_n_values = []
    for n_value in n_values:
        n_value = _validate_positive_int(n_value, "n_values item")
        if n_value > n_features:
            raise ValueError("n_values items cannot exceed the number of features")
        validated_n_values.append(n_value)
    if len(set(validated_n_values)) != len(validated_n_values):
        raise ValueError("n_values must not contain duplicates")
    n_values = validated_n_values
    if seed is not None:
        seed = _validate_nonnegative_int(seed, "seed")

    scores = {}
    for n in n_values:
        score = _require_scalar_result(
            compute_sensitivity_n(
                model,
                instance,
                explanation,
                baseline=baseline,
                background_data=background_data,
                target_class=target_class,
                n=n,
                n_subsets=n_subsets,
                use_absolute=use_absolute,
                seed=seed + n if seed is not None else None,
            ),
            "compute_sensitivity_n",
        )
        scores[n] = score

    mean_score = np.mean(list(scores.values()))

    return {
        "mean": float(mean_score),
        "scores": scores,
        "n_values": n_values,
    }


def compute_batch_sensitivity_n(
    model,
    X: np.ndarray,
    explanations: List[Explanation],
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    max_samples: Optional[int] = None,
    n: Optional[int] = None,
    n_subsets: int = 100,
    use_absolute: bool = False,
    seed: Optional[int] = None,
    background_data: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute average Sensitivity-n over a batch of instances.

    Args:
        model: Model adapter
        X: Input data (2D array)
        explanations: List of Explanation objects (one per instance)
        baseline: Baseline for feature removal
        max_samples: Maximum number of samples to evaluate
        n: Subset size (default: n_features // 4)
        n_subsets: Number of random subsets per instance
        use_absolute: If False (default), sum signed attributions as in the
            paper; True uses the non-canonical magnitude variant.
        seed: Random seed for reproducibility

    Returns:
        Dictionary with mean, std, min, max, and count of evaluated scores
    """
    X, n_samples = _validate_batch_inputs(X, explanations, max_samples)
    if seed is not None:
        seed = _validate_nonnegative_int(seed, "seed")
    scores = []
    for i in range(n_samples):
        current_seed = seed + i if seed is not None else None
        result = compute_sensitivity_n(
            model,
            X[i],
            explanations[i],
            baseline=baseline,
            background_data=background_data,
            n=n,
            n_subsets=n_subsets,
            use_absolute=use_absolute,
            seed=current_seed,
        )
        scores.append(_require_scalar_result(result, "compute_sensitivity_n"))
    return _summarize_scores(scores)


# =============================================================================
# Metric 5: Region Perturbation (Samek et al., 2015)
# =============================================================================


def compute_region_perturbation(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    background_data: Optional[np.ndarray] = None,
    target_class: Optional[int] = None,
    region_size: Optional[int] = None,
    use_absolute: bool = True,
    return_curve: bool = False,
) -> Union[float, Dict[str, Union[float, np.ndarray]]]:
    """
    Compute a one-dimensional contiguous-group perturbation diagnostic.

    Samek et al.'s Region Perturbation is an image-heatmap procedure that
    iteratively replaces a 9x9 neighbourhood with uniform random values and
    reports dataset-averaged AOPC. This function instead partitions a numeric
    vector into fixed contiguous, non-overlapping groups, uses a deterministic
    caller-selected baseline, normalizes by the original scalar output, and
    returns AUC. It is not a canonical implementation of the cited metric.

    Lower AUC means faster relative degradation under this local convention.
    The ratio and AUC are not guaranteed to lie in [0, 1].

    Args:
        model: Model adapter with predict/predict_proba method
        instance: Input instance (1D array)
        explanation: Explanation object with feature_attributions
        baseline: Baseline for feature removal ("mean", "median", scalar, array, callable)
        background_data: Reference data for computing baseline (required for "mean"/"median")
        target_class: Explicit output index. If omitted, resolve it from the
            explanation; ambiguous or conflicting multi-output identities fail.
        region_size: Number of features per region. If None, defaults to max(1, n_features // 4)
            For image-like data, this would correspond to patch size.
        use_absolute: If True, use a library-defined magnitude aggregation;
            False sums signed attributions.
        return_curve: If True, return full degradation curve and details

    Returns:
        If return_curve=False: local relative-output AUC (float)
        If return_curve=True: Dictionary with:
            - 'auc': float - Area under the perturbation curve
            - 'curve': np.ndarray - Normalized prediction values at each step
            - 'predictions': np.ndarray - Raw prediction values
            - 'region_order': list - Order in which regions were perturbed
            - 'regions': list - List of feature indices in each region
            - 'n_regions': int - Number of regions
            - 'region_size': int - Size of each region

    References:
        Samek, W., Binder, A., Montavon, G., Lapuschkin, S., & Müller, K. R. (2015).
        Evaluating the Visualization of What a Deep Neural Network has Learned.
        arXiv preprint arXiv:1509.06321.
    """
    instance = _as_feature_vector(instance)
    n_features = len(instance)
    use_absolute = _validate_bool(use_absolute, "use_absolute")
    return_curve = _validate_bool(return_curve, "return_curve")

    # Get baseline values
    baseline_values = _validated_baseline_values(baseline, background_data, n_features)

    # Extract attributions as array
    attr_array = _extract_attribution_array(explanation, n_features)

    # Determine region size
    if region_size is None:
        # Default: divide features into ~4 regions
        region_size = max(1, n_features // 4)
    region_size = _validate_positive_int(region_size, "region_size")
    if region_size > n_features:
        raise ValueError("region_size cannot exceed the number of features")

    # Create non-overlapping regions
    regions = []
    for start_idx in range(0, n_features, region_size):
        end_idx = min(start_idx + region_size, n_features)
        regions.append(list(range(start_idx, end_idx)))

    n_regions = len(regions)

    # Compute region importance (sum of attributions in each region)
    region_importance = []
    for region in regions:
        if use_absolute:
            importance = np.sum(np.abs(attr_array[region]))
        else:
            importance = np.sum(attr_array[region])
        region_importance.append(importance)

    # Sort regions by importance (descending - most important first)
    sorted_region_indices = np.argsort(-np.array(region_importance), kind="stable")

    # Resolve the explanation's declared output once, then keep it fixed.
    target_class = _resolve_target_output(model, instance, explanation, target_class)
    original_value = _get_target_class_prediction(model, instance.reshape(1, -1), target_class)
    if original_value == 0.0:
        raise ValueError(
            "relative region-perturbation normalization is undefined when f(x) is zero"
        )

    # Start with original instance
    current = instance.copy()

    # Track predictions as regions are perturbed
    predictions = [original_value]

    # Perturb regions one by one (most important first)
    for region_idx in sorted_region_indices:
        region = regions[region_idx]

        # Replace all features in this region with baseline
        for feat_idx in region:
            current[feat_idx] = baseline_values[feat_idx]

        # Get prediction
        predictions.append(
            _get_target_class_prediction(model, current.reshape(1, -1), target_class)
        )

    prediction_array = np.asarray(predictions, dtype=float)

    # Form a relative-output curve. Ratios need not lie in [0, 1].
    curve = prediction_array / original_value

    # Compute AUC using trapezoidal rule
    # x-axis: fraction of regions perturbed (0 to 1)
    # y-axis: fixed-target output divided by the original output
    x = np.linspace(0, 1, len(prediction_array))
    auc = _trapezoid(curve, x)

    if return_curve:
        return {
            "auc": float(auc),
            "curve": curve,
            "predictions": prediction_array,
            "region_order": sorted_region_indices.tolist(),
            "regions": regions,
            "n_regions": n_regions,
            "region_size": region_size,
        }

    return float(auc)


def compute_batch_region_perturbation(
    model,
    X: np.ndarray,
    explanations: List[Explanation],
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    max_samples: Optional[int] = None,
    region_size: Optional[int] = None,
    use_absolute: bool = True,
    background_data: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute average Region Perturbation score over a batch of instances.

    Args:
        model: Model adapter
        X: Input data (2D array)
        explanations: List of Explanation objects (one per instance)
        baseline: Baseline for feature removal
        max_samples: Maximum number of samples to evaluate
        region_size: Number of features per region (default: n_features // 4)
        use_absolute: If True, sum attribution magnitudes within each region;
            False sums signed attributions.

    Returns:
        Dictionary with mean, std, min, max, and count of evaluated scores
    """
    X, n_samples = _validate_batch_inputs(X, explanations, max_samples)
    scores = []
    for i in range(n_samples):
        result = compute_region_perturbation(
            model,
            X[i],
            explanations[i],
            baseline=baseline,
            background_data=background_data,
            region_size=region_size,
            use_absolute=use_absolute,
        )
        scores.append(_require_scalar_result(result, "compute_region_perturbation"))
    return _summarize_scores(scores)


# =============================================================================
# Metric 4: Pixel Flipping (Bach et al., 2015)
# =============================================================================


def compute_pixel_flipping(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    background_data: Optional[np.ndarray] = None,
    target_class: Optional[int] = None,
    use_absolute: bool = False,
    return_curve: bool = False,
) -> Union[float, Dict[str, Union[float, np.ndarray]]]:
    """
    Compute a one-dimensional baseline-replacement Pixel-Flipping adaptation.

    Montavon et al.'s reviewed procedure records raw function values while
    iteratively removing the feature with highest relevance and returns their
    AUC. The original applications operate on images (and later text). This
    function applies that raw-output curve to a numeric vector with a selected
    baseline. It explicitly includes both the unperturbed and fully replaced
    endpoints when integrating over the fraction removed; this endpoint
    convention is part of the adaptation. The result is not directly
    comparable with published image results.

    Lower AUC means faster degradation under the chosen output and baseline.
    It lies in [0, 1] only when the tracked model output does.

    Args:
        model: Model adapter with predict/predict_proba method
        instance: Input instance (1D array)
        explanation: Explanation object with feature_attributions
        baseline: Baseline for feature removal ("mean", "median", scalar, array, callable)
        background_data: Reference data for computing baseline (required for "mean"/"median")
        target_class: Explicit output index. If omitted, resolve it from the
            explanation; ambiguous or conflicting multi-output identities fail.
        use_absolute: If False (default), rank signed relevance as in the
            published procedure. True enables a magnitude-ranking variant.
        return_curve: If True, return full degradation curve and predictions

    Returns:
        If return_curve=False: raw-output AUC score (float)
        If return_curve=True: Dictionary with 'auc', 'curve', 'predictions', 'feature_order'

    References:
        Bach, S., et al. (2015). On Pixel-Wise Explanations for Non-Linear
        Classifier Decisions by Layer-Wise Relevance Propagation. PLOS ONE.
        Montavon, G., Samek, W., & Müller, K. R. (2018). Methods for
        Interpreting and Understanding Deep Neural Networks. Digital Signal
        Processing, 73, 1-15.
    """
    instance = _as_feature_vector(instance)
    n_features = len(instance)
    use_absolute = _validate_bool(use_absolute, "use_absolute")
    return_curve = _validate_bool(return_curve, "return_curve")

    # Get baseline values
    baseline_values = _validated_baseline_values(baseline, background_data, n_features)

    # Extract attributions as array
    attr_array = _extract_attribution_array(explanation, n_features)

    # Sort features by attribution (descending - most important first)
    if use_absolute:
        sorted_indices = np.argsort(-np.abs(attr_array), kind="stable")
    else:
        sorted_indices = np.argsort(-attr_array, kind="stable")

    # Resolve the explanation's declared output once, then keep it fixed.
    target_class = _resolve_target_output(model, instance, explanation, target_class)
    original_value = _get_target_class_prediction(model, instance.reshape(1, -1), target_class)

    # Start with original instance
    current = instance.copy()

    # Track predictions as features are removed
    predictions = [original_value]

    # Remove features one by one (most important first)
    for idx in sorted_indices:
        # Remove this feature (replace with baseline)
        current[idx] = baseline_values[idx]

        # Get prediction
        predictions.append(
            _get_target_class_prediction(model, current.reshape(1, -1), target_class)
        )

    prediction_array = np.asarray(predictions, dtype=float)

    # The reviewed pixel-flipping procedure integrates recorded raw function
    # values; it does not divide the curve by f(x).
    curve = prediction_array.copy()

    # Compute AUC using trapezoidal rule
    # x-axis: fraction of features removed (0 to 1)
    # y-axis: raw fixed-target output
    x = np.linspace(0, 1, len(prediction_array))
    auc = _trapezoid(curve, x)

    if return_curve:
        return {
            "auc": float(auc),
            "curve": curve,
            "predictions": prediction_array,
            "feature_order": sorted_indices,
            "n_features": n_features,
        }

    return float(auc)


def compute_batch_pixel_flipping(
    model,
    X: np.ndarray,
    explanations: List[Explanation],
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    max_samples: Optional[int] = None,
    use_absolute: bool = False,
    background_data: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute average Pixel Flipping score over a batch of instances.

    Args:
        model: Model adapter
        X: Input data (2D array)
        explanations: List of Explanation objects (one per instance)
        baseline: Baseline for feature removal
        max_samples: Maximum number of samples to evaluate
        use_absolute: If False (default), rank signed relevance; True uses the
            magnitude-ranking variant.

    Returns:
        Dictionary with mean, std, min, max, and count of evaluated scores
    """
    X, n_samples = _validate_batch_inputs(X, explanations, max_samples)
    scores = []
    for i in range(n_samples):
        result = compute_pixel_flipping(
            model,
            X[i],
            explanations[i],
            baseline=baseline,
            background_data=background_data,
            use_absolute=use_absolute,
        )
        scores.append(_require_scalar_result(result, "compute_pixel_flipping"))
    return _summarize_scores(scores)


# =============================================================================
# Metric 3: Monotonicity-Nguyen (Nguyen et al., 2020)
# =============================================================================


def compute_monotonicity_nguyen(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    background_data: Optional[np.ndarray] = None,
    target_class: Optional[int] = None,
    use_absolute: bool = True,
) -> float:
    """
    Compute a single-baseline proxy for Nguyen--Martinez Monotonicity.

    Nguyen & Rodriguez Martinez define Spearman correlation between absolute
    attributions and an expected loss obtained by varying each feature over a
    specified marginal or conditional distribution. This function substitutes
    one absolute fixed-target output change at a caller-selected baseline for
    that expectation. It is a deterministic local proxy, not the paper's
    estimator and not directly comparable with its results.

    Unlike AIX360 Monotonicity (sequential feature addition), this proxy
    evaluates each feature independently and uses rank correlation to
    measure agreement between attributed importance and actual impact.

    Args:
        model: Model adapter with predict/predict_proba method
        instance: Input instance (1D array)
        explanation: Explanation object with feature_attributions
        baseline: Baseline for feature removal ("mean", "median", scalar, array, callable)
        background_data: Reference data for computing baseline (required for "mean"/"median")
        target_class: Explicit output index. If omitted, resolve it from the
            explanation; ambiguous or conflicting multi-output identities fail.
        use_absolute: If True (default), use absolute attributions as in the
            paper's definition; False is a signed-attribution extension.

    Returns:
        Spearman correlation in [-1, 1]. Larger values mean a more increasing
        response along this baseline-addition path, not a general quality verdict.

    References:
        Nguyen, A. P., & Rodriguez Martinez, M. (2020). On Quantitative
        Aspects of Model Interpretability. arXiv:2007.07584.
    """
    instance = _as_feature_vector(instance)
    n_features = len(instance)
    use_absolute = _validate_bool(use_absolute, "use_absolute")

    # Get baseline values
    baseline_values = _validated_baseline_values(baseline, background_data, n_features)

    # Extract attributions as array
    attr_array = _extract_attribution_array(explanation, n_features)

    # Resolve the explanation's declared output once, then keep it fixed.
    target_class = _resolve_target_output(model, instance, explanation, target_class)
    original_value = _get_target_class_prediction(model, instance.reshape(1, -1), target_class)

    # Compute prediction change for each feature when removed
    prediction_changes = []
    attribution_values = []

    for i in range(n_features):
        # Create perturbed instance with feature i replaced by baseline
        perturbed = instance.copy()
        perturbed[i] = baseline_values[i]

        # Get prediction for perturbed instance
        perturbed_value = _get_target_class_prediction(
            model, perturbed.reshape(1, -1), target_class
        )

        # Absolute fixed-target output change after replacing this feature.
        change = original_value - perturbed_value
        prediction_changes.append(abs(change))

        # Attribution value
        if use_absolute:
            attribution_values.append(abs(attr_array[i]))
        else:
            attribution_values.append(attr_array[i])

    prediction_change_array = np.asarray(prediction_changes, dtype=float)
    attribution_array = np.asarray(attribution_values, dtype=float)

    corr, _ = _defined_spearman(attribution_array, prediction_change_array)
    return corr


def compute_batch_monotonicity_nguyen(
    model,
    X: np.ndarray,
    explanations: List[Explanation],
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    max_samples: Optional[int] = None,
    use_absolute: bool = True,
    background_data: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute average Monotonicity-Nguyen over a batch of instances.

    Args:
        model: Model adapter
        X: Input data (2D array)
        explanations: List of Explanation objects (one per instance)
        baseline: Baseline for feature removal
        max_samples: Maximum number of samples to evaluate
        use_absolute: If True (default), use absolute attributions as in the
            paper's definition; False is a signed-attribution extension.

    Returns:
        Dictionary with mean, std, min, max, and count of evaluated scores
    """
    X, n_samples = _validate_batch_inputs(X, explanations, max_samples)
    scores = []
    for i in range(n_samples):
        scores.append(
            compute_monotonicity_nguyen(
                model,
                X[i],
                explanations[i],
                baseline=baseline,
                background_data=background_data,
                use_absolute=use_absolute,
            )
        )
    return _summarize_scores(scores)


# =============================================================================
# Metric 2: Monotonicity (Arya et al., 2019)
# =============================================================================


def compute_monotonicity(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    background_data: Optional[np.ndarray] = None,
    target_class: Optional[int] = None,
    use_absolute: bool = False,
    tolerance: float = 0.0,
) -> float:
    """
    Compute the binary AIX360 Monotonicity diagnostic.

    This follows the AIX360 reference implementation described by Arya et al.:
    start at the baseline, add features in increasing signed-attribution order,
    and test whether the fixed target output is nondecreasing after successive
    additions. The result is binary (1.0 or 0.0), not the fraction of successful
    transitions. Arya et al.'s prose motivates the test for positive evidence
    and notes an independence assumption, while the released AIX360 function
    iterates over every coefficient; this implementation follows that code.
    Passing this finite diagnostic is not a proof of global monotonicity.

    Args:
        model: Model adapter with predict/predict_proba method
        instance: Input instance (1D array)
        explanation: Explanation object with feature_attributions
        baseline: Baseline for masked features ("mean", "median", scalar, array, callable)
        background_data: Reference data for computing baseline (required for "mean"/"median")
        target_class: Explicit output index. If omitted, resolve it from the
            explanation; ambiguous or conflicting multi-output identities fail.
        use_absolute: If False (default), use the signed ordering in AIX360.
            True enables a non-canonical magnitude-ordering variant.
        tolerance: Optional non-negative relaxation. The default 0.0 matches
            the AIX360 comparison; positive values are a library extension.

    Returns:
        1.0 if every evaluated transition is nondecreasing, otherwise 0.0.

    References:
        Arya, V., et al. (2019). One Explanation Does Not Fit All: A Toolkit and
        Taxonomy of AI Explainability Techniques. arXiv:1909.03012.
    """
    instance = _as_feature_vector(instance)
    n_features = len(instance)
    use_absolute = _validate_bool(use_absolute, "use_absolute")
    tolerance = _validate_nonnegative_real(tolerance, "tolerance")

    # Get baseline values
    baseline_values = _validated_baseline_values(baseline, background_data, n_features)

    # Extract attributions as array
    attr_array = _extract_attribution_array(explanation, n_features)

    # AIX360 adds features from least to most attributed.
    if use_absolute:
        sorted_indices = np.argsort(np.abs(attr_array), kind="stable")
    else:
        sorted_indices = np.argsort(attr_array, kind="stable")

    # Resolve the explanation's declared output once, then keep it fixed.
    target_class = _resolve_target_output(model, instance, explanation, target_class)

    # Start from baseline (all features masked)
    current = baseline_values.copy()

    # Track outputs after each feature is revealed. AIX360 does not include a
    # baseline-only output in the monotonicity comparison.
    predictions = []

    # Add features one by one
    revealed_features = []
    for idx in sorted_indices:
        # Reveal this feature (set to original value)
        revealed_features.append(idx)
        current[idx] = instance[idx]

        # Get prediction
        predictions.append(
            _get_target_class_prediction(model, current.reshape(1, -1), target_class)
        )

    return float(np.all(np.diff(np.asarray(predictions)) >= -tolerance))


def compute_batch_monotonicity(
    model,
    X: np.ndarray,
    explanations: List[Explanation],
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    max_samples: Optional[int] = None,
    use_absolute: bool = False,
    background_data: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute average Monotonicity over a batch of instances.

    Args:
        model: Model adapter
        X: Input data (2D array)
        explanations: List of Explanation objects (one per instance)
        baseline: Baseline for masked features
        max_samples: Maximum number of samples to evaluate
        use_absolute: If False (default), use the signed AIX360 ordering;
            True uses the non-canonical magnitude-ordering variant.

    Returns:
        Dictionary with mean, std, min, max, and count of evaluated scores
    """
    X, n_samples = _validate_batch_inputs(X, explanations, max_samples)
    scores = []
    for i in range(n_samples):
        scores.append(
            compute_monotonicity(
                model,
                X[i],
                explanations[i],
                baseline=baseline,
                background_data=background_data,
                use_absolute=use_absolute,
            )
        )
    return _summarize_scores(scores)
