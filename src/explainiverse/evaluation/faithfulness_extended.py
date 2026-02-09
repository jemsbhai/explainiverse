# src/explainiverse/evaluation/faithfulness_extended.py
"""
Extended faithfulness evaluation metrics.

Phase 1 metrics for exceeding OpenXAI/Quantus:
- Faithfulness Estimate (Alvarez-Melis et al., 2018)
- Monotonicity (Arya et al., 2019)
- Monotonicity-Nguyen (Nguyen et al., 2020)
- Pixel Flipping (Bach et al., 2015)
- Region Perturbation (Samek et al., 2015)
- Selectivity (Montavon et al., 2018)
- Sensitivity-n (Ancona et al., 2018)
- IROF (Rieger & Hansen, 2020)
- Infidelity (Yeh et al., 2019)
- ROAD (Rong et al., 2022)
- Insertion AUC (Petsiuk et al., 2018)
- Deletion AUC (Petsiuk et al., 2018)

All 12 Phase 1 faithfulness metrics are now complete.
"""
import numpy as np
import re
from typing import Union, Callable, List, Dict, Optional, Tuple
from scipy import stats

# NumPy 2.0 compatibility: np.trapz was renamed to np.trapezoid
_trapezoid = getattr(np, 'trapezoid', np.trapz)

from explainiverse.core.explanation import Explanation
from explainiverse.evaluation._utils import (
    get_sorted_feature_indices,
    compute_baseline_values,
    apply_feature_mask,
    resolve_k,
    get_prediction_value,
    compute_prediction_change,
)


def _extract_attribution_array(
    explanation: Explanation,
    n_features: int
) -> np.ndarray:
    """
    Extract attribution values as a numpy array in feature index order.
    
    Args:
        explanation: Explanation object with feature_attributions
        n_features: Expected number of features
        
    Returns:
        1D numpy array of attribution values ordered by feature index
    """
    attributions = explanation.explanation_data.get("feature_attributions", {})
    feature_names = getattr(explanation, 'feature_names', None)
    
    if not attributions:
        raise ValueError("No feature attributions found in explanation.")
    
    # Build attribution array in feature order
    attr_array = np.zeros(n_features)
    
    if feature_names is not None:
        for fname, value in attributions.items():
            # Try to find the index for this feature name
            for i, fn in enumerate(feature_names):
                if fn == fname or fn in fname or fname in fn:
                    attr_array[i] = value
                    break
            else:
                # Try extracting index from name pattern
                for pattern in [r'feature[_\s]*(\d+)', r'feat[_\s]*(\d+)', r'^f(\d+)', r'^x(\d+)']:
                    match = re.search(pattern, fname, re.IGNORECASE)
                    if match:
                        idx = int(match.group(1))
                        if 0 <= idx < n_features:
                            attr_array[idx] = value
                        break
    else:
        # No feature names - try to extract indices from keys
        for fname, value in attributions.items():
            for pattern in [r'feature[_\s]*(\d+)', r'feat[_\s]*(\d+)', r'^f(\d+)', r'^x(\d+)']:
                match = re.search(pattern, fname, re.IGNORECASE)
                if match:
                    idx = int(match.group(1))
                    if 0 <= idx < n_features:
                        attr_array[idx] = value
                    break
    
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
    seed: int = None,
) -> np.ndarray:
    """
    Impute removed features using Noisy Linear Imputation (Rong et al., 2022).
    
    For each removed feature j, fits a linear regression from the remaining
    features using the background data:
        x_j = w^T * x_remaining + b + epsilon
    where epsilon ~ N(0, noise_scale * sigma^2_residual).
    
    The noise prevents class information leakage through the imputed values,
    which is the key insight of the ROAD framework.
    
    Args:
        instance: Input instance (1D array)
        removed_indices: Indices of features to impute
        remaining_indices: Indices of features to use as predictors
        background_data: Training data for fitting linear models (2D array)
        noise_scale: Scale factor for residual noise (default: 1.0).
            Higher values add more noise, reducing information leakage.
        seed: Random seed for reproducibility
        
    Returns:
        Imputed instance with removed features replaced by noisy linear predictions
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
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
# Metric 10: ROAD - RemOve And Debias (Rong et al., 2022)
# =============================================================================

def compute_road(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    background_data: np.ndarray,
    target_class: int = None,
    percentages: List[float] = None,
    order: str = "morf",
    noise_scale: float = 1.0,
    use_absolute: bool = True,
    seed: int = None,
    return_details: bool = False,
) -> Union[float, Dict[str, Union[float, np.ndarray, List]]]:
    """
    Compute ROAD (RemOve And Debias) score (Rong et al., 2022).
    
    ROAD evaluates explanation faithfulness using noisy linear imputation
    instead of simple baseline replacement, addressing the out-of-distribution
    problem and class information leakage in perturbation-based evaluation.
    
    At each removal percentage p, the top-p% features (by attribution) are
    removed and replaced using Noisy Linear Imputation fitted on the
    background data. The model's prediction change on the imputed sample
    is recorded. The final score is the mean prediction change across
    all percentages.
    
    Two ordering strategies:
    - **MoRF** (Most Relevant First): Remove important features first.
      Higher score = better explanation (important features truly matter).
    - **LeRF** (Least Relevant First): Remove unimportant features first.
      Lower score = better explanation (unimportant features truly don't matter).
    
    The Noisy Linear Imputation operator fits a linear regression from
    remaining features to each removed feature using the training data,
    then adds calibrated Gaussian noise (epsilon ~ N(0, sigma^2_residual))
    to provably remove information while preserving inter-feature dependencies.
    
    Args:
        model: Model adapter with predict/predict_proba method
        instance: Input instance (1D array)
        explanation: Explanation object with feature_attributions
        background_data: Training data for fitting imputation models (2D array).
            Required for noisy linear imputation.
        target_class: Target class index for probability (default: predicted class)
        percentages: List of removal percentages as fractions in (0, 1).
            Default: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        order: Feature removal order:
            - "morf": Most Relevant First (descending attribution)
            - "lerf": Least Relevant First (ascending attribution)
        noise_scale: Scale factor for imputation noise (default: 1.0).
            Controls the amount of Gaussian noise added to linear predictions.
        use_absolute: If True, sort features by absolute attribution value
        seed: Random seed for reproducibility
        return_details: If True, return detailed results
        
    Returns:
        If return_details=False: Mean prediction change across percentages (float).
            For MoRF: higher is better (removing important features hurts prediction).
            For LeRF: lower is better (removing unimportant features has little effect).
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
    instance = np.asarray(instance).flatten()
    n_features = len(instance)
    
    # Validate background_data
    if background_data is None:
        raise ValueError(
            "background_data is required for ROAD metric "
            "(needed for noisy linear imputation)."
        )
    background_data = np.asarray(background_data)
    if background_data.ndim != 2 or background_data.shape[1] != n_features:
        raise ValueError(
            f"background_data must be 2D with {n_features} columns, "
            f"got shape {background_data.shape}."
        )
    
    # Validate order
    if order not in ("morf", "lerf"):
        raise ValueError(
            f"order must be 'morf' or 'lerf', got '{order}'."
        )
    
    # Extract attributions as array
    attr_array = _extract_attribution_array(explanation, n_features)
    
    # Sort features by attribution
    if use_absolute:
        sort_values = np.abs(attr_array)
    else:
        sort_values = attr_array
    
    if order == "morf":
        # Most Relevant First: descending order
        sorted_indices = np.argsort(-sort_values)
    else:
        # Least Relevant First: ascending order
        sorted_indices = np.argsort(sort_values)
    
    # Default percentages
    if percentages is None:
        percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Validate percentages
    percentages = [p for p in percentages if 0 < p < 1]
    if not percentages:
        raise ValueError("percentages must contain values in (0, 1).")
    percentages = sorted(percentages)
    
    # Determine target class
    if target_class is None:
        pred = get_prediction_value(model, instance.reshape(1, -1))
        if isinstance(pred, np.ndarray) and pred.ndim > 0:
            target_class = int(np.argmax(pred))
        else:
            target_class = 0
    
    # Get original prediction for the target class
    original_pred = get_prediction_value(model, instance.reshape(1, -1))
    if isinstance(original_pred, np.ndarray) and original_pred.ndim > 0 and len(original_pred) > target_class:
        original_value = original_pred[target_class]
    else:
        original_value = float(original_pred)
    
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
        imputed_pred = get_prediction_value(model, imputed.reshape(1, -1))
        if isinstance(imputed_pred, np.ndarray) and imputed_pred.ndim > 0 and len(imputed_pred) > target_class:
            imputed_value = imputed_pred[target_class]
        else:
            imputed_value = float(imputed_pred)
        
        predictions.append(imputed_value)
        # Prediction change: drop in confidence (positive = prediction decreased)
        prediction_changes.append(original_value - imputed_value)
    
    prediction_changes = np.array(prediction_changes)
    predictions = np.array(predictions)
    
    # Score: mean prediction change across percentages
    score = float(np.mean(prediction_changes))
    
    if return_details:
        return {
            "score": score,
            "prediction_changes": prediction_changes,
            "predictions": predictions,
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
    target_class: int = None,
    percentages: List[float] = None,
    noise_scale: float = 1.0,
    use_absolute: bool = True,
    seed: int = None,
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Compute combined ROAD-MoRF and ROAD-LeRF scores.
    
    This variant evaluates both removal orderings and provides a combined
    assessment. A good explanation should have high MoRF score (removing
    important features hurts) and low LeRF score (removing unimportant
    features doesn't hurt), so a large gap (MoRF - LeRF) indicates a
    faithful explanation.
    
    Args:
        model: Model adapter with predict/predict_proba method
        instance: Input instance (1D array)
        explanation: Explanation object with feature_attributions
        background_data: Training data for fitting imputation models
        target_class: Target class index for probability (default: predicted class)
        percentages: List of removal percentages as fractions in (0, 1)
        noise_scale: Scale factor for imputation noise
        use_absolute: If True, sort features by absolute attribution value
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with:
            - 'morf': float - ROAD-MoRF score (higher is better)
            - 'lerf': float - ROAD-LeRF score (lower is better)
            - 'gap': float - MoRF minus LeRF (higher gap = better explanation)
            - 'scores': Dict[str, float] - Both scores by name
    """
    morf_seed = seed
    lerf_seed = seed + 10000 if seed is not None else None
    
    morf_score = compute_road(
        model, instance, explanation,
        background_data=background_data,
        target_class=target_class,
        percentages=percentages,
        order="morf",
        noise_scale=noise_scale,
        use_absolute=use_absolute,
        seed=morf_seed,
    )
    
    lerf_score = compute_road(
        model, instance, explanation,
        background_data=background_data,
        target_class=target_class,
        percentages=percentages,
        order="lerf",
        noise_scale=noise_scale,
        use_absolute=use_absolute,
        seed=lerf_seed,
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
    background_data: np.ndarray = None,
    max_samples: int = None,
    percentages: List[float] = None,
    order: str = "morf",
    noise_scale: float = 1.0,
    use_absolute: bool = True,
    seed: int = None,
) -> Dict[str, float]:
    """
    Compute average ROAD score over a batch of instances.
    
    Args:
        model: Model adapter
        X: Input data (2D array)
        explanations: List of Explanation objects (one per instance)
        background_data: Training data for imputation (default: uses X)
        max_samples: Maximum number of samples to evaluate
        percentages: Removal percentages (default: [0.1, ..., 0.9])
        order: Feature removal order ('morf' or 'lerf')
        noise_scale: Scale factor for imputation noise
        use_absolute: If True, sort features by absolute attribution value
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with mean, std, min, max, and count of valid scores
    """
    if background_data is None:
        background_data = X
    
    n_instances = len(explanations)
    if max_samples:
        n_instances = min(n_instances, max_samples)
    
    scores = []
    
    for i in range(n_instances):
        try:
            current_seed = seed + i if seed is not None else None
            score = compute_road(
                model, X[i], explanations[i],
                background_data=background_data,
                percentages=percentages,
                order=order,
                noise_scale=noise_scale,
                use_absolute=use_absolute,
                seed=current_seed,
            )
            if not np.isnan(score):
                scores.append(score)
        except Exception:
            continue
    
    if not scores:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n_samples": 0}
    
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "n_samples": len(scores),
    }


# =============================================================================
# Metric 11: Deletion AUC (Petsiuk et al., 2018)
# =============================================================================


def _get_target_class_prediction(
    model,
    instance_2d: np.ndarray,
    target_class: int,
) -> float:
    """
    Get the predicted probability for a specific target class.

    Handles both raw scikit-learn models and explainiverse adapters.
    Unlike get_prediction_value (which returns the max probability),
    this function extracts the probability for a *specific* class index.

    Args:
        model: Model with predict or predict_proba method
        instance_2d: Input instance reshaped to 2D (1, n_features)
        target_class: Class index to extract probability for

    Returns:
        Probability value for the target class (float)
    """
    instance_2d = np.asarray(instance_2d)
    if instance_2d.ndim == 1:
        instance_2d = instance_2d.reshape(1, -1)

    # Try predict_proba first (raw sklearn model)
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(instance_2d)
        if isinstance(proba, np.ndarray):
            if proba.ndim == 2 and proba.shape[1] > target_class:
                return float(proba[0, target_class])
            elif proba.ndim == 1 and len(proba) > target_class:
                return float(proba[target_class])
        return float(np.max(proba))

    # Fall back to predict (adapter returns probs from predict)
    pred = model.predict(instance_2d)
    if isinstance(pred, np.ndarray):
        if pred.ndim == 2 and pred.shape[1] > target_class:
            return float(pred[0, target_class])
        elif pred.ndim == 1 and len(pred) > target_class:
            return float(pred[target_class])
    return float(pred)


def _resolve_target_class(
    model,
    instance: np.ndarray,
) -> int:
    """
    Determine the target class as the model's predicted class for this instance.

    Args:
        model: Model adapter
        instance: 1D input instance

    Returns:
        Predicted class index
    """
    instance_2d = np.asarray(instance).reshape(1, -1)

    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(instance_2d)
        if isinstance(proba, np.ndarray):
            if proba.ndim == 2:
                return int(np.argmax(proba[0]))
            return int(np.argmax(proba))

    pred = model.predict(instance_2d)
    if isinstance(pred, np.ndarray):
        if pred.ndim == 2:
            return int(np.argmax(pred[0]))
        elif pred.ndim == 1 and len(pred) > 1:
            return int(np.argmax(pred))
    return 0


def compute_deletion_auc(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    background_data: np.ndarray = None,
    target_class: int = None,
    use_absolute: bool = True,
    n_steps: int = None,
    return_curve: bool = False,
) -> Union[float, Dict[str, Union[float, np.ndarray]]]:
    """
    Compute Deletion AUC score (Petsiuk et al., 2018).

    Progressively removes features from the original input in order of
    decreasing attribution (most important first), recording the model's
    predicted probability for the target class at each step. The Deletion
    AUC is the Area Under this degradation Curve (AUC), normalized to [0, 1].

    A faithful explanation identifies features that, when removed, cause a
    rapid drop in the target class probability. Therefore, **lower Deletion
    AUC indicates a better explanation**.

    The metric differs from Pixel Flipping in its formulation:
    - Pixel Flipping normalizes predictions relative to the original value.
    - Deletion AUC tracks raw class probabilities on the y-axis (as in the
      original RISE paper), giving a more interpretable curve whose y-axis
      represents actual prediction confidence.

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
        target_class: Target class index whose probability is tracked.
            Default: model's predicted class for this instance.
        use_absolute: If True, sort features by absolute attribution value
        n_steps: Number of evenly spaced removal steps. If None, removes
            one feature at a time (n_steps = n_features). If specified,
            features are removed in chunks of size ceil(n_features/n_steps).
        return_curve: If True, return full curve details

    Returns:
        If return_curve=False:
            Deletion AUC (float, 0 to 1). **Lower is better.**
        If return_curve=True:
            Dictionary with:
            - 'auc': float — Area under deletion curve
            - 'curve': np.ndarray — Predicted probability at each step
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
    instance = np.asarray(instance).flatten()
    n_features = len(instance)

    # Get baseline values
    baseline_values = compute_baseline_values(
        baseline, background_data, n_features
    )

    # Extract and sort attributions
    attr_array = _extract_attribution_array(explanation, n_features)
    if use_absolute:
        sorted_indices = np.argsort(-np.abs(attr_array))
    else:
        sorted_indices = np.argsort(-attr_array)

    # Determine target class
    if target_class is None:
        target_class = _resolve_target_class(model, instance)

    # Determine removal schedule
    if n_steps is not None:
        # Percentage-based steps: remove features in chunks
        n_steps = max(1, min(n_steps, n_features))
        step_sizes = np.round(
            np.linspace(0, n_features, n_steps + 1)
        ).astype(int)
        # step_sizes[i] = cumulative number of features removed after step i
        removal_counts = step_sizes[1:]  # exclude the 0
    else:
        # One feature at a time
        removal_counts = np.arange(1, n_features + 1)

    # Get original prediction (step 0: no features removed)
    original_pred = _get_target_class_prediction(
        model, instance.reshape(1, -1), target_class
    )

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
        pred_val = _get_target_class_prediction(
            model, current.reshape(1, -1), target_class
        )
        predictions.append(pred_val)
        fractions.append(float(count) / float(n_features))

    predictions = np.array(predictions)
    fractions = np.array(fractions)

    # Compute AUC using trapezoidal rule
    # x-axis: fraction of features removed [0, 1]
    # y-axis: target class probability
    auc = float(_trapezoid(predictions, fractions))

    if return_curve:
        return {
            "auc": auc,
            "curve": predictions,
            "fractions": fractions,
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
    max_samples: int = None,
    use_absolute: bool = True,
    n_steps: int = None,
) -> Dict[str, float]:
    """
    Compute average Deletion AUC over a batch of instances.

    Args:
        model: Model adapter
        X: Input data (2D array, one row per instance)
        explanations: List of Explanation objects (one per instance)
        baseline: Baseline for feature removal
        max_samples: Maximum number of samples to evaluate
        use_absolute: If True, sort features by absolute attribution value
        n_steps: Number of removal steps per instance (None = one per feature)

    Returns:
        Dictionary with mean, std, min, max, and n_samples of valid scores.
        Lower mean indicates better explanations on average.
    """
    n_samples = len(explanations)
    if max_samples:
        n_samples = min(n_samples, max_samples)

    scores = []

    for i in range(n_samples):
        try:
            score = compute_deletion_auc(
                model, X[i], explanations[i],
                baseline=baseline, background_data=X,
                use_absolute=use_absolute,
                n_steps=n_steps,
            )
            if not np.isnan(score):
                scores.append(score)
        except Exception:
            continue

    if not scores:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n_samples": 0}

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "n_samples": len(scores),
    }


# =============================================================================
# Metric 12: Insertion AUC (Petsiuk et al., 2018)
# =============================================================================


def compute_insertion_auc(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    background_data: np.ndarray = None,
    target_class: int = None,
    use_absolute: bool = True,
    n_steps: int = None,
    return_curve: bool = False,
) -> Union[float, Dict[str, Union[float, np.ndarray]]]:
    """
    Compute Insertion AUC score (Petsiuk et al., 2018).

    Starts from a baseline input (all features at baseline values) and
    progressively inserts features in order of decreasing attribution
    (most important first), recording the model's predicted probability
    for the target class at each step. The Insertion AUC is the Area Under
    this recovery Curve, normalized to [0, 1].

    A faithful explanation identifies features that, when inserted, cause a
    rapid rise in the target class probability. Therefore, **higher Insertion
    AUC indicates a better explanation**.

    The Insertion metric is the natural complement to the Deletion metric.
    Together they provide a comprehensive view of explanation faithfulness:
    Deletion measures whether attributed features are truly needed, while
    Insertion measures whether they are truly sufficient.

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
        target_class: Target class index whose probability is tracked.
            Default: model's predicted class for the original instance.
        use_absolute: If True, sort features by absolute attribution value
        n_steps: Number of evenly spaced insertion steps. If None, inserts
            one feature at a time (n_steps = n_features). If specified,
            features are inserted in chunks of size ceil(n_features/n_steps).
        return_curve: If True, return full curve details

    Returns:
        If return_curve=False:
            Insertion AUC (float, 0 to 1). **Higher is better.**
        If return_curve=True:
            Dictionary with:
            - 'auc': float — Area under insertion curve
            - 'curve': np.ndarray — Predicted probability at each step
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
    instance = np.asarray(instance).flatten()
    n_features = len(instance)

    # Get baseline values
    baseline_values = compute_baseline_values(
        baseline, background_data, n_features
    )

    # Extract and sort attributions
    attr_array = _extract_attribution_array(explanation, n_features)
    if use_absolute:
        sorted_indices = np.argsort(-np.abs(attr_array))
    else:
        sorted_indices = np.argsort(-attr_array)

    # Determine target class from the ORIGINAL instance
    if target_class is None:
        target_class = _resolve_target_class(model, instance)

    # Determine insertion schedule
    if n_steps is not None:
        n_steps = max(1, min(n_steps, n_features))
        step_sizes = np.round(
            np.linspace(0, n_features, n_steps + 1)
        ).astype(int)
        insertion_counts = step_sizes[1:]  # cumulative features inserted
    else:
        insertion_counts = np.arange(1, n_features + 1)

    # Start from baseline (all features at baseline)
    current = baseline_values.copy()

    # Get baseline prediction (step 0: no features from original)
    baseline_pred = _get_target_class_prediction(
        model, current.reshape(1, -1), target_class
    )

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
        pred_val = _get_target_class_prediction(
            model, current.reshape(1, -1), target_class
        )
        predictions.append(pred_val)
        fractions.append(float(count) / float(n_features))

    predictions = np.array(predictions)
    fractions = np.array(fractions)

    # Compute AUC using trapezoidal rule
    # x-axis: fraction of features inserted [0, 1]
    # y-axis: target class probability
    auc = float(_trapezoid(predictions, fractions))

    if return_curve:
        return {
            "auc": auc,
            "curve": predictions,
            "fractions": fractions,
            "feature_order": sorted_indices,
            "n_features": n_features,
            "target_class": target_class,
            "baseline_prediction": baseline_pred,
            "final_prediction": float(predictions[-1]),
        }

    return auc


def compute_batch_insertion_auc(
    model,
    X: np.ndarray,
    explanations: List[Explanation],
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    max_samples: int = None,
    use_absolute: bool = True,
    n_steps: int = None,
) -> Dict[str, float]:
    """
    Compute average Insertion AUC over a batch of instances.

    Args:
        model: Model adapter
        X: Input data (2D array, one row per instance)
        explanations: List of Explanation objects (one per instance)
        baseline: Baseline for feature insertion starting state
        max_samples: Maximum number of samples to evaluate
        use_absolute: If True, sort features by absolute attribution value
        n_steps: Number of insertion steps per instance (None = one per feature)

    Returns:
        Dictionary with mean, std, min, max, and n_samples of valid scores.
        Higher mean indicates better explanations on average.
    """
    n_samples = len(explanations)
    if max_samples:
        n_samples = min(n_samples, max_samples)

    scores = []

    for i in range(n_samples):
        try:
            score = compute_insertion_auc(
                model, X[i], explanations[i],
                baseline=baseline, background_data=X,
                use_absolute=use_absolute,
                n_steps=n_steps,
            )
            if not np.isnan(score):
                scores.append(score)
        except Exception:
            continue

    if not scores:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n_samples": 0}

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "n_samples": len(scores),
    }


# =============================================================================
# Combined Insertion-Deletion convenience function
# =============================================================================


def compute_insertion_deletion_auc(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    background_data: np.ndarray = None,
    target_class: int = None,
    use_absolute: bool = True,
    n_steps: int = None,
) -> Dict[str, float]:
    """
    Compute both Insertion and Deletion AUC in a single call.

    Provides a comprehensive faithfulness assessment by combining the two
    complementary metrics from Petsiuk et al. (2018). Also returns their
    difference (Insertion − Deletion), which summarizes overall explanation
    quality as a single number: higher difference indicates better faithfulness.

    Args:
        model: Model adapter with predict/predict_proba method
        instance: Input instance (1D array)
        explanation: Explanation object with feature_attributions
        baseline: Baseline for feature operations
        background_data: Reference data for computing baseline
        target_class: Target class index (default: predicted class)
        use_absolute: If True, sort features by absolute attribution value
        n_steps: Number of steps per curve (None = one per feature)

    Returns:
        Dictionary with:
        - 'insertion_auc': float — Insertion AUC (higher is better)
        - 'deletion_auc': float — Deletion AUC (lower is better)
        - 'delta': float — insertion_auc − deletion_auc (higher is better)

    References:
        Petsiuk, V., Das, A., & Saenko, K. (2018). RISE: Randomized Input
        Sampling for Explanation of Black-box Models. BMVC.
    """
    ins = compute_insertion_auc(
        model, instance, explanation,
        baseline=baseline,
        background_data=background_data,
        target_class=target_class,
        use_absolute=use_absolute,
        n_steps=n_steps,
    )
    dele = compute_deletion_auc(
        model, instance, explanation,
        baseline=baseline,
        background_data=background_data,
        target_class=target_class,
        use_absolute=use_absolute,
        n_steps=n_steps,
    )

    return {
        "insertion_auc": float(ins),
        "deletion_auc": float(dele),
        "delta": float(ins) - float(dele),
    }


# =============================================================================
# Metric 1: Faithfulness Estimate (Alvarez-Melis & Jaakkola, 2018)
# =============================================================================

def compute_faithfulness_estimate(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    background_data: np.ndarray = None,
    subset_size: int = None,
    n_subsets: int = 100,
    seed: int = None,
) -> float:
    """
    Compute Faithfulness Estimate (Alvarez-Melis & Jaakkola, 2018).
    
    Measures the correlation between feature attributions and the actual
    impact on predictions when individual features are perturbed. For each
    feature, computes the prediction change when that feature is replaced
    with baseline, then correlates these changes with attribution magnitudes.
    
    Higher correlation indicates the explanation correctly identifies
    which features actually matter for the prediction.
    
    Args:
        model: Model adapter with predict/predict_proba method
        instance: Input instance (1D array)
        explanation: Explanation object with feature_attributions
        baseline: Baseline for feature replacement ("mean", "median", scalar, array, callable)
        background_data: Reference data for computing baseline (required for "mean"/"median")
        subset_size: Size of random subsets to perturb (default: 1 for single-feature)
        n_subsets: Number of random subsets to evaluate (used when subset_size > 1)
        seed: Random seed for reproducibility
        
    Returns:
        Faithfulness estimate score (Pearson correlation, -1 to 1, higher is better)
        
    References:
        Alvarez-Melis, D., & Jaakkola, T. S. (2018). Towards Robust Interpretability 
        with Self-Explaining Neural Networks. NeurIPS.
    """
    if seed is not None:
        np.random.seed(seed)
    
    instance = np.asarray(instance).flatten()
    n_features = len(instance)
    
    # Get baseline values
    baseline_values = compute_baseline_values(
        baseline, background_data, n_features
    )
    
    # Extract attributions as array
    attr_array = _extract_attribution_array(explanation, n_features)
    
    # Default subset_size is 1 (single-feature perturbation)
    if subset_size is None:
        subset_size = 1
    
    if subset_size == 1:
        # Single-feature perturbation: evaluate each feature individually
        prediction_changes = []
        attribution_values = []
        
        for i in range(n_features):
            # Skip features with zero attribution (they won't affect correlation)
            if abs(attr_array[i]) < 1e-10:
                continue
            
            # Perturb single feature
            perturbed = apply_feature_mask(instance, [i], baseline_values)
            
            # Compute prediction change
            change = compute_prediction_change(model, instance, perturbed, metric="absolute")
            
            prediction_changes.append(change)
            attribution_values.append(abs(attr_array[i]))
        
        if len(prediction_changes) < 2:
            return 0.0  # Not enough data points for correlation
        
        # Check for constant arrays (would produce undefined correlation)
        if np.std(attribution_values) < 1e-10 or np.std(prediction_changes) < 1e-10:
            return 0.0
        
        # Compute Pearson correlation
        corr, _ = stats.pearsonr(attribution_values, prediction_changes)
        
        return float(corr) if not np.isnan(corr) else 0.0
    
    else:
        # Random subset perturbation
        prediction_changes = []
        attribution_sums = []
        
        for _ in range(n_subsets):
            # Sample random subset of features
            subset_indices = np.random.choice(
                n_features, size=min(subset_size, n_features), replace=False
            )
            
            # Perturb subset
            perturbed = apply_feature_mask(instance, subset_indices.tolist(), baseline_values)
            
            # Compute prediction change
            change = compute_prediction_change(model, instance, perturbed, metric="absolute")
            
            # Sum of attributions in subset
            attr_sum = np.sum(np.abs(attr_array[subset_indices]))
            
            prediction_changes.append(change)
            attribution_sums.append(attr_sum)
        
        if len(prediction_changes) < 2:
            return 0.0
        
        # Compute Pearson correlation
        corr, _ = stats.pearsonr(attribution_sums, prediction_changes)
        
        return float(corr) if not np.isnan(corr) else 0.0


def compute_batch_faithfulness_estimate(
    model,
    X: np.ndarray,
    explanations: List[Explanation],
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    max_samples: int = None,
    seed: int = None,
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
        Dictionary with mean, std, min, max, and count of valid scores
    """
    n_samples = len(explanations)
    if max_samples:
        n_samples = min(n_samples, max_samples)
    
    scores = []
    
    for i in range(n_samples):
        try:
            score = compute_faithfulness_estimate(
                model, X[i], explanations[i],
                baseline=baseline, background_data=X,
                seed=seed
            )
            if not np.isnan(score):
                scores.append(score)
        except Exception:
            continue
    
    if not scores:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n_samples": 0}
    
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "n_samples": len(scores),
    }


# =============================================================================
# Metric 8: IROF - Iterative Removal of Features (Rieger & Hansen, 2020)
# =============================================================================

def compute_irof(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    background_data: np.ndarray = None,
    target_class: int = None,
    segment_size: int = None,
    use_absolute: bool = True,
    return_details: bool = False,
) -> Union[float, Dict[str, Union[float, np.ndarray, List]]]:
    """
    Compute IROF (Iterative Removal of Features) score (Rieger & Hansen, 2020).
    
    IROF measures explanation faithfulness by iteratively removing features
    (or segments of features) in order of attributed importance and tracking
    prediction degradation. Features are organized into segments, and segments
    are removed from most to least important based on the sum of their
    attributed relevance scores.
    
    The metric computes the Area Over the Perturbation Curve (AOC), which
    measures how quickly the model's prediction for the target class drops
    as important features are removed. Higher AOC indicates better faithfulness
    (the explanation correctly identifies features important for classification).
    
    For tabular data, each feature can be treated as a segment (segment_size=1),
    or features can be grouped into larger segments.
    
    AOC = ∫₀¹ [original_pred - f(x_perturbed)] d(fraction_removed)
    
    where the integral is computed using the trapezoidal rule over the
    normalized perturbation curve.
    
    Args:
        model: Model adapter with predict/predict_proba method
        instance: Input instance (1D array)
        explanation: Explanation object with feature_attributions
        baseline: Baseline for feature removal ("mean", "median", scalar, array, callable)
        background_data: Reference data for computing baseline (required for "mean"/"median")
        target_class: Target class index for probability (default: predicted class)
        segment_size: Number of features per segment. If None, defaults to 1
            (each feature is its own segment). For image-like data, this groups
            features into spatial regions.
        use_absolute: If True, sort segments by absolute attribution sum (default: True)
        return_details: If True, return detailed results including degradation curve
        
    Returns:
        If return_details=False: AOC score (float, higher is better)
        If return_details=True: Dictionary with:
            - 'aoc': float - Area Over the perturbation Curve
            - 'curve': np.ndarray - Prediction drop at each step (original - perturbed)
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
    instance = np.asarray(instance).flatten()
    n_features = len(instance)
    
    # Get baseline values
    baseline_values = compute_baseline_values(
        baseline, background_data, n_features
    )
    
    # Extract attributions as array
    attr_array = _extract_attribution_array(explanation, n_features)
    
    # Determine segment size (default: 1 = each feature is a segment)
    if segment_size is None:
        segment_size = 1
    segment_size = max(1, min(segment_size, n_features))
    
    # Create non-overlapping segments
    segments = []
    for start_idx in range(0, n_features, segment_size):
        end_idx = min(start_idx + segment_size, n_features)
        segments.append(list(range(start_idx, end_idx)))
    
    n_segments = len(segments)
    
    # Compute segment importance (sum of attributions in each segment)
    segment_importance = np.zeros(n_segments)
    for i, segment in enumerate(segments):
        if use_absolute:
            segment_importance[i] = np.sum(np.abs(attr_array[segment]))
        else:
            segment_importance[i] = np.sum(attr_array[segment])
    
    # Sort segments by importance (descending - most important first)
    sorted_segment_indices = np.argsort(-segment_importance)
    
    # Determine target class
    if target_class is None:
        pred = get_prediction_value(model, instance.reshape(1, -1))
        if isinstance(pred, np.ndarray) and pred.ndim > 0:
            target_class = int(np.argmax(pred))
        else:
            target_class = 0
    
    # Get original prediction for the target class
    original_pred = get_prediction_value(model, instance.reshape(1, -1))
    if isinstance(original_pred, np.ndarray) and original_pred.ndim > 0 and len(original_pred) > target_class:
        original_value = original_pred[target_class]
    else:
        original_value = float(original_pred)
    
    # Start with original instance
    current = instance.copy()
    
    # Track predictions and prediction drops at each step
    # Step 0: no segments removed
    predictions = [original_value]
    prediction_drops = [0.0]  # original - original = 0
    
    # Iteratively remove segments (most important first)
    for segment_idx in sorted_segment_indices:
        segment = segments[segment_idx]
        
        # Remove features in this segment (replace with baseline)
        for feat_idx in segment:
            current[feat_idx] = baseline_values[feat_idx]
        
        # Get prediction
        pred = get_prediction_value(model, current.reshape(1, -1))
        if isinstance(pred, np.ndarray) and pred.ndim > 0 and len(pred) > target_class:
            current_pred = pred[target_class]
        else:
            current_pred = float(pred)
        
        predictions.append(current_pred)
        # Prediction drop: original_value - current_pred
        # Positive drop means removing features decreased prediction
        prediction_drops.append(original_value - current_pred)
    
    predictions = np.array(predictions)
    prediction_drops = np.array(prediction_drops)
    
    # Compute Area Over the Curve (AOC)
    # x-axis: fraction of segments removed (0 to 1)
    # y-axis: prediction drop (original - perturbed)
    # Higher AOC = more area above baseline = better explanation
    x = np.linspace(0, 1, len(prediction_drops))
    aoc = _trapezoid(prediction_drops, x)
    
    if return_details:
        return {
            "aoc": float(aoc),
            "curve": prediction_drops,
            "predictions": predictions,
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
    background_data: np.ndarray = None,
    target_class: int = None,
    segment_sizes: List[int] = None,
    use_absolute: bool = True,
) -> Dict[str, Union[float, Dict[int, float]]]:
    """
    Compute IROF for multiple segment sizes and return average.
    
    This variant evaluates IROF across different segment granularities,
    providing a more robust assessment that is less sensitive to the
    choice of segment size.
    
    Args:
        model: Model adapter with predict/predict_proba method
        instance: Input instance (1D array)
        explanation: Explanation object with feature_attributions
        baseline: Baseline for feature removal
        background_data: Reference data for computing baseline
        target_class: Target class index for probability (default: predicted class)
        segment_sizes: List of segment sizes to evaluate. If None, uses
            [1, n//4, n//2] where n is the number of features.
        use_absolute: If True, sort segments by absolute attribution sum
        
    Returns:
        Dictionary with:
            - 'mean': float - Average AOC across all segment sizes
            - 'scores': Dict[int, float] - AOC for each segment size
            - 'segment_sizes': List[int] - Segment sizes evaluated
    """
    instance = np.asarray(instance).flatten()
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
    
    scores = {}
    for seg_size in segment_sizes:
        score = compute_irof(
            model, instance, explanation,
            baseline=baseline, background_data=background_data,
            target_class=target_class,
            segment_size=seg_size,
            use_absolute=use_absolute,
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
    max_samples: int = None,
    segment_size: int = None,
    use_absolute: bool = True,
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
        use_absolute: If True, sort segments by absolute attribution sum
        
    Returns:
        Dictionary with mean, std, min, max, and count of valid scores
    """
    n_samples = len(explanations)
    if max_samples:
        n_samples = min(n_samples, max_samples)
    
    scores = []
    
    for i in range(n_samples):
        try:
            score = compute_irof(
                model, X[i], explanations[i],
                baseline=baseline, background_data=X,
                segment_size=segment_size,
                use_absolute=use_absolute,
            )
            if not np.isnan(score):
                scores.append(score)
        except Exception:
            continue
    
    if not scores:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n_samples": 0}
    
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "n_samples": len(scores),
    }


# =============================================================================
# Metric 9: Infidelity (Yeh et al., 2019)
# =============================================================================

def compute_infidelity(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    background_data: np.ndarray = None,
    target_class: int = None,
    perturbation_type: str = "gaussian",
    noise_scale: float = 0.1,
    n_samples: int = 100,
    subset_size: int = None,
    seed: int = None,
    return_details: bool = False,
) -> Union[float, Dict[str, Union[float, np.ndarray]]]:
    """
    Compute Infidelity score (Yeh et al., 2019).
    
    Infidelity measures how well an explanation predicts the change in model
    output when the input is perturbed. A faithful explanation should accurately
    predict that perturbing important features causes proportional prediction changes.
    
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
    
    **Lower infidelity = better explanation** (0 is perfect).
    
    Three perturbation strategies are supported:
    - "gaussian": Continuous Gaussian noise I ~ N(0, σ²I)
    - "square": Binary mask (1s for perturbed features, 0s otherwise)
    - "subset": Random subset of features are perturbed
    
    Args:
        model: Model adapter with predict/predict_proba method
        instance: Input instance (1D array)
        explanation: Explanation object with feature_attributions
        baseline: Baseline for perturbation ("mean", "median", scalar, array, callable)
            Used to determine perturbed values: x - I becomes baseline where I=1
        background_data: Reference data for computing baseline (required for "mean"/"median")
        target_class: Target class index for probability (default: predicted class)
        perturbation_type: Type of perturbation distribution:
            - "gaussian": Gaussian noise scaled by noise_scale
            - "square": Binary mask perturbation (features replaced with baseline)
            - "subset": Random subset of features perturbed to baseline
        noise_scale: Standard deviation for Gaussian perturbations (default: 0.1)
            For "square" and "subset", controls probability of perturbation per feature.
        n_samples: Number of Monte Carlo samples for expectation (default: 100)
        subset_size: For "subset" perturbation, number of features to perturb.
            If None, defaults to max(1, n_features // 4)
        seed: Random seed for reproducibility
        return_details: If True, return detailed results
        
    Returns:
        If return_details=False: Infidelity score (float, lower is better, 0 is perfect)
        If return_details=True: Dictionary with:
            - 'infidelity': float - Mean squared error
            - 'squared_errors': np.ndarray - Squared error for each sample
            - 'expected_changes': np.ndarray - φ(x)ᵀ · I for each sample
            - 'actual_changes': np.ndarray - f(x) - f(x-I) for each sample
            - 'n_samples': int - Number of Monte Carlo samples
            - 'perturbation_type': str - Type of perturbation used
        
    References:
        Yeh, C. K., Hsieh, C. Y., Suggala, A., Inber, D. I., Ravikumar, P. K., 
        Ravikumar, P., & Dhillon, I. S. (2019). On the (In)fidelity and Sensitivity 
        of Explanations. NeurIPS 2019.
    """
    if seed is not None:
        np.random.seed(seed)
    
    instance = np.asarray(instance).flatten()
    n_features = len(instance)
    
    # Get baseline values
    baseline_values = compute_baseline_values(
        baseline, background_data, n_features
    )
    
    # Extract attributions as array
    attr_array = _extract_attribution_array(explanation, n_features)
    
    # Determine target class
    if target_class is None:
        pred = get_prediction_value(model, instance.reshape(1, -1))
        if isinstance(pred, np.ndarray) and pred.ndim > 0:
            target_class = int(np.argmax(pred))
        else:
            target_class = 0
    
    # Get original prediction for the target class
    original_pred = get_prediction_value(model, instance.reshape(1, -1))
    if isinstance(original_pred, np.ndarray) and original_pred.ndim > 0 and len(original_pred) > target_class:
        original_value = original_pred[target_class]
    else:
        original_value = float(original_pred)
    
    # Determine subset size for subset perturbation
    if subset_size is None:
        subset_size = max(1, n_features // 4)
    subset_size = max(1, min(subset_size, n_features))
    
    # Monte Carlo sampling for expectation
    squared_errors = []
    expected_changes = []
    actual_changes = []
    
    for _ in range(n_samples):
        if perturbation_type == "gaussian":
            # Gaussian perturbation: I ~ N(0, σ²I)
            # For continuous perturbation, we add noise directly
            perturbation = np.random.normal(0, noise_scale, n_features)
            perturbed = instance - perturbation
            
            # Expected change: φ(x)ᵀ · I
            expected_change = np.dot(attr_array, perturbation)
            
        elif perturbation_type == "square":
            # Square/binary mask perturbation
            # Each feature has probability noise_scale of being perturbed
            mask = np.random.random(n_features) < noise_scale
            
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
            subset_indices = np.random.choice(n_features, size=subset_size, replace=False)
            
            # Perturbation vector: 1 for perturbed features, 0 otherwise
            # But we scale by the actual value difference for meaningful comparison
            perturbation = np.zeros(n_features)
            perturbed = instance.copy()
            
            for idx in subset_indices:
                perturbation[idx] = instance[idx] - baseline_values[idx]
                perturbed[idx] = baseline_values[idx]
            
            # Expected change: φ(x)ᵀ · I
            expected_change = np.dot(attr_array, perturbation)
            
        else:
            raise ValueError(f"Unknown perturbation_type: {perturbation_type}. "
                           f"Choose from 'gaussian', 'square', 'subset'.")
        
        # Get perturbed prediction
        perturbed_pred = get_prediction_value(model, perturbed.reshape(1, -1))
        if isinstance(perturbed_pred, np.ndarray) and perturbed_pred.ndim > 0 and len(perturbed_pred) > target_class:
            perturbed_value = perturbed_pred[target_class]
        else:
            perturbed_value = float(perturbed_pred)
        
        # Actual change: f(x) - f(x - I)
        actual_change = original_value - perturbed_value
        
        # Squared error: (expected - actual)²
        sq_error = (expected_change - actual_change) ** 2
        
        squared_errors.append(sq_error)
        expected_changes.append(expected_change)
        actual_changes.append(actual_change)
    
    squared_errors = np.array(squared_errors)
    expected_changes = np.array(expected_changes)
    actual_changes = np.array(actual_changes)
    
    # Infidelity is the mean squared error
    infidelity = np.mean(squared_errors)
    
    if return_details:
        return {
            "infidelity": float(infidelity),
            "squared_errors": squared_errors,
            "expected_changes": expected_changes,
            "actual_changes": actual_changes,
            "n_samples": n_samples,
            "perturbation_type": perturbation_type,
            "original_prediction": original_value,
        }
    
    return float(infidelity)


def compute_infidelity_multi_perturbation(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    background_data: np.ndarray = None,
    target_class: int = None,
    perturbation_types: List[str] = None,
    noise_scale: float = 0.1,
    n_samples: int = 100,
    seed: int = None,
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Compute Infidelity across multiple perturbation types.
    
    This variant provides a more robust assessment by evaluating infidelity
    under different perturbation strategies and returning the average.
    
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
    
    scores = {}
    for i, ptype in enumerate(perturbation_types):
        current_seed = seed + i if seed is not None else None
        score = compute_infidelity(
            model, instance, explanation,
            baseline=baseline, background_data=background_data,
            target_class=target_class,
            perturbation_type=ptype,
            noise_scale=noise_scale,
            n_samples=n_samples,
            seed=current_seed,
        )
        scores[ptype] = score
    
    mean_score = np.mean(list(scores.values()))
    
    return {
        "mean": float(mean_score),
        "scores": scores,
        "perturbation_types": perturbation_types,
    }


def compute_batch_infidelity(
    model,
    X: np.ndarray,
    explanations: List[Explanation],
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    max_samples: int = None,
    perturbation_type: str = "gaussian",
    noise_scale: float = 0.1,
    n_perturbations: int = 100,
    seed: int = None,
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
        Dictionary with mean, std, min, max, and count of valid scores
    """
    n_instances = len(explanations)
    if max_samples:
        n_instances = min(n_instances, max_samples)
    
    scores = []
    
    for i in range(n_instances):
        try:
            current_seed = seed + i if seed is not None else None
            score = compute_infidelity(
                model, X[i], explanations[i],
                baseline=baseline, background_data=X,
                perturbation_type=perturbation_type,
                noise_scale=noise_scale,
                n_samples=n_perturbations,
                seed=current_seed,
            )
            if not np.isnan(score):
                scores.append(score)
        except Exception:
            continue
    
    if not scores:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n_samples": 0}
    
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "n_samples": len(scores),
    }


# =============================================================================
# Metric 6: Selectivity (Montavon et al., 2018)
# =============================================================================

def compute_selectivity(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    background_data: np.ndarray = None,
    target_class: int = None,
    n_steps: int = None,
    use_absolute: bool = True,
    return_details: bool = False,
) -> Union[float, Dict[str, Union[float, np.ndarray]]]:
    """
    Compute Selectivity score using AOPC (Montavon et al., 2018).
    
    Measures how quickly the prediction function drops when removing features
    with the highest attributed values. Computed as the Area Over the
    Perturbation Curve (AOPC), which is the average prediction drop across
    all perturbation steps.
    
    AOPC = (1/(K+1)) * Σₖ₌₀ᴷ [f(x) - f(x_{1..k})]
    
    where:
    - f(x) is the original prediction for the target class
    - f(x_{1..k}) is the prediction after removing the top-k most important features
    - K is the total number of perturbation steps (default: n_features)
    
    Higher AOPC indicates better selectivity - the explanation correctly
    identifies features whose removal causes the largest prediction drop.
    
    Args:
        model: Model adapter with predict/predict_proba method
        instance: Input instance (1D array)
        explanation: Explanation object with feature_attributions
        baseline: Baseline for feature removal ("mean", "median", scalar, array, callable)
        background_data: Reference data for computing baseline (required for "mean"/"median")
        target_class: Target class index for probability (default: predicted class)
        n_steps: Number of perturbation steps (default: n_features, max features to remove)
        use_absolute: If True, sort features by absolute attribution value (default: True)
        return_details: If True, return detailed results including prediction drops per step
        
    Returns:
        If return_details=False: AOPC score (float, higher is better)
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
    instance = np.asarray(instance).flatten()
    n_features = len(instance)
    
    # Get baseline values
    baseline_values = compute_baseline_values(
        baseline, background_data, n_features
    )
    
    # Extract attributions as array
    attr_array = _extract_attribution_array(explanation, n_features)
    
    # Determine number of steps (default: all features)
    if n_steps is None:
        n_steps = n_features
    n_steps = min(n_steps, n_features)
    
    # Sort features by attribution (descending - most important first)
    if use_absolute:
        sorted_indices = np.argsort(-np.abs(attr_array))
    else:
        sorted_indices = np.argsort(-attr_array)
    
    # Determine target class
    if target_class is None:
        pred = get_prediction_value(model, instance.reshape(1, -1))
        if isinstance(pred, np.ndarray) and pred.ndim > 0:
            target_class = int(np.argmax(pred))
        else:
            target_class = 0
    
    # Get original prediction for the target class
    original_pred = get_prediction_value(model, instance.reshape(1, -1))
    if isinstance(original_pred, np.ndarray) and original_pred.ndim > 0 and len(original_pred) > target_class:
        original_value = original_pred[target_class]
    else:
        original_value = float(original_pred)
    
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
        pred = get_prediction_value(model, current.reshape(1, -1))
        if isinstance(pred, np.ndarray) and pred.ndim > 0 and len(pred) > target_class:
            current_pred = pred[target_class]
        else:
            current_pred = float(pred)
        
        predictions.append(current_pred)
        # Prediction drop: f(x) - f(x_{1..k})
        prediction_drops.append(original_value - current_pred)
    
    predictions = np.array(predictions)
    prediction_drops = np.array(prediction_drops)
    
    # Compute AOPC: average of prediction drops across all steps
    # AOPC = (1/(K+1)) * Σₖ₌₀ᴷ [f(x) - f(x_{1..k})]
    aopc = np.mean(prediction_drops)
    
    if return_details:
        return {
            "aopc": float(aopc),
            "prediction_drops": prediction_drops,
            "predictions": predictions,
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
    max_samples: int = None,
    n_steps: int = None,
    use_absolute: bool = True,
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
        use_absolute: If True, sort features by absolute attribution value
        
    Returns:
        Dictionary with mean, std, min, max, and count of valid scores
    """
    n_samples = len(explanations)
    if max_samples:
        n_samples = min(n_samples, max_samples)
    
    scores = []
    
    for i in range(n_samples):
        try:
            score = compute_selectivity(
                model, X[i], explanations[i],
                baseline=baseline, background_data=X,
                n_steps=n_steps,
                use_absolute=use_absolute
            )
            if not np.isnan(score):
                scores.append(score)
        except Exception:
            continue
    
    if not scores:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n_samples": 0}
    
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "n_samples": len(scores),
    }


# =============================================================================
# Metric 7: Sensitivity-n (Ancona et al., 2018)
# =============================================================================

def compute_sensitivity_n(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    background_data: np.ndarray = None,
    target_class: int = None,
    n: int = None,
    n_subsets: int = 100,
    use_absolute: bool = True,
    seed: int = None,
    return_details: bool = False,
) -> Union[float, Dict[str, Union[float, np.ndarray, List]]]:
    """
    Compute Sensitivity-n score (Ancona et al., 2018).
    
    Measures the correlation between the sum of attributions for a subset
    of n features and the prediction change when those features are removed.
    A faithful explanation should show high correlation - removing features
    with high total attribution should cause proportionally larger prediction drops.
    
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
        target_class: Target class index for probability (default: predicted class)
        n: Subset size. If None, defaults to max(1, n_features // 4)
        n_subsets: Number of random subsets to sample (default: 100)
        use_absolute: If True, use absolute attribution values in sum (default: True)
        seed: Random seed for reproducibility
        return_details: If True, return detailed results including all subset data
        
    Returns:
        If return_details=False: Sensitivity-n score (Pearson correlation, -1 to 1, higher is better)
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
        np.random.seed(seed)
    
    instance = np.asarray(instance).flatten()
    n_features = len(instance)
    
    # Get baseline values
    baseline_values = compute_baseline_values(
        baseline, background_data, n_features
    )
    
    # Extract attributions as array
    attr_array = _extract_attribution_array(explanation, n_features)
    
    # Determine subset size
    if n is None:
        n = max(1, n_features // 4)
    n = max(1, min(n, n_features))  # Clamp to valid range
    
    # Determine target class
    if target_class is None:
        pred = get_prediction_value(model, instance.reshape(1, -1))
        if isinstance(pred, np.ndarray) and pred.ndim > 0:
            target_class = int(np.argmax(pred))
        else:
            target_class = 0
    
    # Get original prediction for the target class
    original_pred = get_prediction_value(model, instance.reshape(1, -1))
    if isinstance(original_pred, np.ndarray) and original_pred.ndim > 0 and len(original_pred) > target_class:
        original_value = original_pred[target_class]
    else:
        original_value = float(original_pred)
    
    # Sample random subsets and compute correlations
    attribution_sums = []
    prediction_drops = []
    subsets = []
    
    for _ in range(n_subsets):
        # Sample random subset of size n
        subset = np.random.choice(n_features, size=n, replace=False)
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
        perturbed_pred = get_prediction_value(model, perturbed.reshape(1, -1))
        if isinstance(perturbed_pred, np.ndarray) and perturbed_pred.ndim > 0 and len(perturbed_pred) > target_class:
            perturbed_value = perturbed_pred[target_class]
        else:
            perturbed_value = float(perturbed_pred)
        
        # Prediction drop (positive = removing features decreased prediction)
        drop = original_value - perturbed_value
        prediction_drops.append(drop)
    
    attribution_sums = np.array(attribution_sums)
    prediction_drops = np.array(prediction_drops)
    
    # Handle edge cases
    if len(attribution_sums) < 2:
        if return_details:
            return {
                "correlation": 0.0,
                "p_value": 1.0,
                "attribution_sums": attribution_sums,
                "prediction_drops": prediction_drops,
                "subsets": subsets,
                "n": n,
                "n_subsets": len(subsets),
            }
        return 0.0
    
    # Check for constant arrays
    if np.std(attribution_sums) < 1e-10 or np.std(prediction_drops) < 1e-10:
        if return_details:
            return {
                "correlation": 0.0 if np.std(attribution_sums) < 1e-10 or np.std(prediction_drops) < 1e-10 else 1.0,
                "p_value": 1.0,
                "attribution_sums": attribution_sums,
                "prediction_drops": prediction_drops,
                "subsets": subsets,
                "n": n,
                "n_subsets": len(subsets),
            }
        return 0.0
    
    # Compute Pearson correlation
    corr, p_value = stats.pearsonr(attribution_sums, prediction_drops)
    
    if np.isnan(corr):
        corr = 0.0
        p_value = 1.0
    
    if return_details:
        return {
            "correlation": float(corr),
            "p_value": float(p_value),
            "attribution_sums": attribution_sums,
            "prediction_drops": prediction_drops,
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
    background_data: np.ndarray = None,
    target_class: int = None,
    n_values: List[int] = None,
    n_subsets: int = 100,
    use_absolute: bool = True,
    seed: int = None,
) -> Dict[str, Union[float, Dict[int, float]]]:
    """
    Compute Sensitivity-n for multiple subset sizes and return average.
    
    This is the recommended way to use Sensitivity-n, as averaging across
    multiple subset sizes provides a more robust assessment of faithfulness.
    
    Args:
        model: Model adapter with predict/predict_proba method
        instance: Input instance (1D array)
        explanation: Explanation object with feature_attributions
        baseline: Baseline for feature removal
        background_data: Reference data for computing baseline
        target_class: Target class index for probability (default: predicted class)
        n_values: List of subset sizes to evaluate. If None, uses [1, n//4, n//2, 3n//4]
        n_subsets: Number of random subsets per n value (default: 100)
        use_absolute: If True, use absolute attribution values
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with:
            - 'mean': float - Average correlation across all n values
            - 'scores': Dict[int, float] - Correlation for each n value
            - 'n_values': List[int] - Subset sizes evaluated
    """
    instance = np.asarray(instance).flatten()
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
    
    scores = {}
    for n in n_values:
        if seed is not None:
            np.random.seed(seed + n)  # Different seed per n for reproducibility
        
        score = compute_sensitivity_n(
            model, instance, explanation,
            baseline=baseline, background_data=background_data,
            target_class=target_class, n=n,
            n_subsets=n_subsets, use_absolute=use_absolute,
            seed=None  # Already set above
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
    max_samples: int = None,
    n: int = None,
    n_subsets: int = 100,
    use_absolute: bool = True,
    seed: int = None,
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
        use_absolute: If True, use absolute attribution values
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with mean, std, min, max, and count of valid scores
    """
    n_samples = len(explanations)
    if max_samples:
        n_samples = min(n_samples, max_samples)
    
    scores = []
    
    for i in range(n_samples):
        try:
            current_seed = seed + i if seed is not None else None
            score = compute_sensitivity_n(
                model, X[i], explanations[i],
                baseline=baseline, background_data=X,
                n=n, n_subsets=n_subsets,
                use_absolute=use_absolute, seed=current_seed
            )
            if not np.isnan(score):
                scores.append(score)
        except Exception:
            continue
    
    if not scores:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n_samples": 0}
    
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "n_samples": len(scores),
    }


# =============================================================================
# Metric 5: Region Perturbation (Samek et al., 2015)
# =============================================================================

def compute_region_perturbation(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    background_data: np.ndarray = None,
    target_class: int = None,
    region_size: int = None,
    use_absolute: bool = True,
    return_curve: bool = False,
) -> Union[float, Dict[str, Union[float, np.ndarray]]]:
    """
    Compute Region Perturbation score (Samek et al., 2015).
    
    Similar to Pixel Flipping, but operates on regions (groups) of features
    rather than individual features. Features are divided into non-overlapping
    regions, and regions are perturbed in order of their cumulative importance
    (sum of attributions within the region).
    
    This metric is particularly relevant for image data where local spatial
    correlations exist, but is also applicable to tabular data with groups
    of related features.
    
    The score is the Area Under the perturbation Curve (AUC), normalized
    to [0, 1]. Lower AUC indicates better faithfulness (faster degradation
    when important regions are removed first).
    
    Args:
        model: Model adapter with predict/predict_proba method
        instance: Input instance (1D array)
        explanation: Explanation object with feature_attributions
        baseline: Baseline for feature removal ("mean", "median", scalar, array, callable)
        background_data: Reference data for computing baseline (required for "mean"/"median")
        target_class: Target class index for probability (default: predicted class)
        region_size: Number of features per region. If None, defaults to max(1, n_features // 4)
            For image-like data, this would correspond to patch size.
        use_absolute: If True, sort regions by absolute attribution sum (default: True)
        return_curve: If True, return full degradation curve and details
        
    Returns:
        If return_curve=False: AUC score (float, 0 to 1, lower is better)
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
    instance = np.asarray(instance).flatten()
    n_features = len(instance)
    
    # Get baseline values
    baseline_values = compute_baseline_values(
        baseline, background_data, n_features
    )
    
    # Extract attributions as array
    attr_array = _extract_attribution_array(explanation, n_features)
    
    # Determine region size
    if region_size is None:
        # Default: divide features into ~4 regions
        region_size = max(1, n_features // 4)
    region_size = max(1, min(region_size, n_features))  # Clamp to valid range
    
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
    sorted_region_indices = np.argsort(-np.array(region_importance))
    
    # Determine target class
    if target_class is None:
        pred = get_prediction_value(model, instance.reshape(1, -1))
        if isinstance(pred, np.ndarray) and pred.ndim > 0:
            target_class = int(np.argmax(pred))
        else:
            target_class = 0
    
    # Get original prediction for the target class
    original_pred = get_prediction_value(model, instance.reshape(1, -1))
    if isinstance(original_pred, np.ndarray) and original_pred.ndim > 0 and len(original_pred) > target_class:
        original_value = original_pred[target_class]
    else:
        original_value = float(original_pred)
    
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
        pred = get_prediction_value(model, current.reshape(1, -1))
        if isinstance(pred, np.ndarray) and pred.ndim > 0 and len(pred) > target_class:
            predictions.append(pred[target_class])
        else:
            predictions.append(float(pred))
    
    predictions = np.array(predictions)
    
    # Normalize predictions to [0, 1] relative to original
    # curve[i] = prediction after perturbing i regions / original prediction
    if abs(original_value) > 1e-10:
        curve = predictions / original_value
    else:
        # Handle zero original prediction
        curve = predictions
    
    # Compute AUC using trapezoidal rule
    # x-axis: fraction of regions perturbed (0 to 1)
    # y-axis: relative prediction value
    x = np.linspace(0, 1, len(predictions))
    auc = _trapezoid(curve, x)
    
    if return_curve:
        return {
            "auc": float(auc),
            "curve": curve,
            "predictions": predictions,
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
    max_samples: int = None,
    region_size: int = None,
    use_absolute: bool = True,
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
        use_absolute: If True, sort regions by absolute attribution sum
        
    Returns:
        Dictionary with mean, std, min, max, and count of valid scores
    """
    n_samples = len(explanations)
    if max_samples:
        n_samples = min(n_samples, max_samples)
    
    scores = []
    
    for i in range(n_samples):
        try:
            score = compute_region_perturbation(
                model, X[i], explanations[i],
                baseline=baseline, background_data=X,
                region_size=region_size,
                use_absolute=use_absolute
            )
            if not np.isnan(score):
                scores.append(score)
        except Exception:
            continue
    
    if not scores:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n_samples": 0}
    
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "n_samples": len(scores),
    }


# =============================================================================
# Metric 4: Pixel Flipping (Bach et al., 2015)
# =============================================================================

def compute_pixel_flipping(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    background_data: np.ndarray = None,
    target_class: int = None,
    use_absolute: bool = True,
    return_curve: bool = False,
) -> Union[float, Dict[str, Union[float, np.ndarray]]]:
    """
    Compute Pixel Flipping score (Bach et al., 2015).
    
    Sequentially removes features in order of attributed importance (most
    important first) and measures the cumulative prediction degradation.
    A faithful explanation should cause rapid prediction drop when the
    most important features are removed first.
    
    The score is the Area Under the perturbation Curve (AUC), normalized
    to [0, 1]. Lower AUC indicates better faithfulness (faster degradation).
    
    Args:
        model: Model adapter with predict/predict_proba method
        instance: Input instance (1D array)
        explanation: Explanation object with feature_attributions
        baseline: Baseline for feature removal ("mean", "median", scalar, array, callable)
        background_data: Reference data for computing baseline (required for "mean"/"median")
        target_class: Target class index for probability (default: predicted class)
        use_absolute: If True, sort features by absolute attribution value
        return_curve: If True, return full degradation curve and predictions
        
    Returns:
        If return_curve=False: AUC score (float, 0 to 1, lower is better)
        If return_curve=True: Dictionary with 'auc', 'curve', 'predictions', 'feature_order'
        
    References:
        Bach, S., et al. (2015). On Pixel-Wise Explanations for Non-Linear 
        Classifier Decisions by Layer-Wise Relevance Propagation. PLOS ONE.
    """
    instance = np.asarray(instance).flatten()
    n_features = len(instance)
    
    # Get baseline values
    baseline_values = compute_baseline_values(
        baseline, background_data, n_features
    )
    
    # Extract attributions as array
    attr_array = _extract_attribution_array(explanation, n_features)
    
    # Sort features by attribution (descending - most important first)
    if use_absolute:
        sorted_indices = np.argsort(-np.abs(attr_array))
    else:
        sorted_indices = np.argsort(-attr_array)
    
    # Determine target class
    if target_class is None:
        pred = get_prediction_value(model, instance.reshape(1, -1))
        if isinstance(pred, np.ndarray) and pred.ndim > 0:
            target_class = int(np.argmax(pred))
        else:
            target_class = 0
    
    # Get original prediction for the target class
    original_pred = get_prediction_value(model, instance.reshape(1, -1))
    if isinstance(original_pred, np.ndarray) and original_pred.ndim > 0 and len(original_pred) > target_class:
        original_value = original_pred[target_class]
    else:
        original_value = float(original_pred)
    
    # Start with original instance
    current = instance.copy()
    
    # Track predictions as features are removed
    predictions = [original_value]
    
    # Remove features one by one (most important first)
    for idx in sorted_indices:
        # Remove this feature (replace with baseline)
        current[idx] = baseline_values[idx]
        
        # Get prediction
        pred = get_prediction_value(model, current.reshape(1, -1))
        if isinstance(pred, np.ndarray) and pred.ndim > 0 and len(pred) > target_class:
            predictions.append(pred[target_class])
        else:
            predictions.append(float(pred))
    
    predictions = np.array(predictions)
    
    # Normalize predictions to [0, 1] relative to original
    # curve[i] = prediction after removing i features / original prediction
    if abs(original_value) > 1e-10:
        curve = predictions / original_value
    else:
        # Handle zero original prediction
        curve = predictions
    
    # Compute AUC using trapezoidal rule
    # x-axis: fraction of features removed (0 to 1)
    # y-axis: relative prediction value
    x = np.linspace(0, 1, len(predictions))
    auc = _trapezoid(curve, x)
    
    if return_curve:
        return {
            "auc": float(auc),
            "curve": curve,
            "predictions": predictions,
            "feature_order": sorted_indices,
            "n_features": n_features,
        }
    
    return float(auc)


def compute_batch_pixel_flipping(
    model,
    X: np.ndarray,
    explanations: List[Explanation],
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    max_samples: int = None,
    use_absolute: bool = True,
) -> Dict[str, float]:
    """
    Compute average Pixel Flipping score over a batch of instances.
    
    Args:
        model: Model adapter
        X: Input data (2D array)
        explanations: List of Explanation objects (one per instance)
        baseline: Baseline for feature removal
        max_samples: Maximum number of samples to evaluate
        use_absolute: If True, sort features by absolute attribution value
        
    Returns:
        Dictionary with mean, std, min, max, and count of valid scores
    """
    n_samples = len(explanations)
    if max_samples:
        n_samples = min(n_samples, max_samples)
    
    scores = []
    
    for i in range(n_samples):
        try:
            score = compute_pixel_flipping(
                model, X[i], explanations[i],
                baseline=baseline, background_data=X,
                use_absolute=use_absolute
            )
            if not np.isnan(score):
                scores.append(score)
        except Exception:
            continue
    
    if not scores:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n_samples": 0}
    
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "n_samples": len(scores),
    }


# =============================================================================
# Metric 3: Monotonicity-Nguyen (Nguyen et al., 2020)
# =============================================================================

def compute_monotonicity_nguyen(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    background_data: np.ndarray = None,
    target_class: int = None,
    use_absolute: bool = True,
) -> float:
    """
    Compute Monotonicity Correlation (Nguyen et al., 2020).
    
    Measures the Spearman rank correlation between attribution magnitudes
    and the prediction changes when each feature is individually removed
    (replaced with baseline). A faithful explanation should show that
    features with higher attributions cause larger prediction changes
    when removed.
    
    Unlike Arya's Monotonicity (sequential feature addition), this metric
    evaluates each feature independently and uses rank correlation to
    measure agreement between attributed importance and actual impact.
    
    Args:
        model: Model adapter with predict/predict_proba method
        instance: Input instance (1D array)
        explanation: Explanation object with feature_attributions
        baseline: Baseline for feature removal ("mean", "median", scalar, array, callable)
        background_data: Reference data for computing baseline (required for "mean"/"median")
        target_class: Target class index for probability (default: predicted class)
        use_absolute: If True, use absolute attribution values (default: True)
        
    Returns:
        Monotonicity correlation score (Spearman rho, -1 to 1, higher is better)
        
    References:
        Nguyen, A. P., & Martinez, M. R. (2020). Quantitative Evaluation of 
        Machine Learning Explanations: A Human-Grounded Benchmark. 
        arXiv:2010.07455.
    """
    instance = np.asarray(instance).flatten()
    n_features = len(instance)
    
    # Get baseline values
    baseline_values = compute_baseline_values(
        baseline, background_data, n_features
    )
    
    # Extract attributions as array
    attr_array = _extract_attribution_array(explanation, n_features)
    
    # Determine target class
    if target_class is None:
        pred = get_prediction_value(model, instance.reshape(1, -1))
        if isinstance(pred, np.ndarray) and pred.ndim > 0:
            target_class = int(np.argmax(pred))
        else:
            target_class = 0
    
    # Get original prediction for the target class
    original_pred = get_prediction_value(model, instance.reshape(1, -1))
    if isinstance(original_pred, np.ndarray) and original_pred.ndim > 0 and len(original_pred) > target_class:
        original_value = original_pred[target_class]
    else:
        original_value = float(original_pred)
    
    # Compute prediction change for each feature when removed
    prediction_changes = []
    attribution_values = []
    
    for i in range(n_features):
        # Create perturbed instance with feature i replaced by baseline
        perturbed = instance.copy()
        perturbed[i] = baseline_values[i]
        
        # Get prediction for perturbed instance
        perturbed_pred = get_prediction_value(model, perturbed.reshape(1, -1))
        if isinstance(perturbed_pred, np.ndarray) and perturbed_pred.ndim > 0 and len(perturbed_pred) > target_class:
            perturbed_value = perturbed_pred[target_class]
        else:
            perturbed_value = float(perturbed_pred)
        
        # Prediction change (drop in confidence when feature is removed)
        # Positive change means removing the feature decreased prediction
        change = original_value - perturbed_value
        prediction_changes.append(abs(change))
        
        # Attribution value
        if use_absolute:
            attribution_values.append(abs(attr_array[i]))
        else:
            attribution_values.append(attr_array[i])
    
    prediction_changes = np.array(prediction_changes)
    attribution_values = np.array(attribution_values)
    
    # Handle edge cases
    if len(prediction_changes) < 2:
        return 0.0
    
    # Check for constant arrays (would cause division by zero in correlation)
    if np.std(prediction_changes) < 1e-10 or np.std(attribution_values) < 1e-10:
        # If both are constant, consider it perfect correlation
        if np.std(prediction_changes) < 1e-10 and np.std(attribution_values) < 1e-10:
            return 1.0
        # If only one is constant, correlation is undefined
        return 0.0
    
    # Compute Spearman rank correlation
    corr, _ = stats.spearmanr(attribution_values, prediction_changes)
    
    return float(corr) if not np.isnan(corr) else 0.0


def compute_batch_monotonicity_nguyen(
    model,
    X: np.ndarray,
    explanations: List[Explanation],
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    max_samples: int = None,
    use_absolute: bool = True,
) -> Dict[str, float]:
    """
    Compute average Monotonicity-Nguyen over a batch of instances.
    
    Args:
        model: Model adapter
        X: Input data (2D array)
        explanations: List of Explanation objects (one per instance)
        baseline: Baseline for feature removal
        max_samples: Maximum number of samples to evaluate
        use_absolute: If True, use absolute attribution values
        
    Returns:
        Dictionary with mean, std, min, max, and count of valid scores
    """
    n_samples = len(explanations)
    if max_samples:
        n_samples = min(n_samples, max_samples)
    
    scores = []
    
    for i in range(n_samples):
        try:
            score = compute_monotonicity_nguyen(
                model, X[i], explanations[i],
                baseline=baseline, background_data=X,
                use_absolute=use_absolute
            )
            if not np.isnan(score):
                scores.append(score)
        except Exception:
            continue
    
    if not scores:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n_samples": 0}
    
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "n_samples": len(scores),
    }


# =============================================================================
# Metric 2: Monotonicity (Arya et al., 2019)
# =============================================================================

def compute_monotonicity(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    background_data: np.ndarray = None,
    target_class: int = None,
    use_absolute: bool = True,
    tolerance: float = 1e-6,
) -> float:
    """
    Compute Monotonicity (Arya et al., 2019).
    
    Measures whether sequentially adding features in order of their attributed
    importance monotonically increases the model's prediction confidence.
    Starting from a baseline (all features masked), features are revealed
    one-by-one in descending order of attribution. A faithful explanation
    should show monotonically increasing predictions.
    
    Args:
        model: Model adapter with predict/predict_proba method
        instance: Input instance (1D array)
        explanation: Explanation object with feature_attributions
        baseline: Baseline for masked features ("mean", "median", scalar, array, callable)
        background_data: Reference data for computing baseline (required for "mean"/"median")
        target_class: Target class index for probability (default: predicted class)
        use_absolute: If True, sort features by absolute attribution value
        tolerance: Small value for numerical stability in monotonicity check
        
    Returns:
        Monotonicity score (0 to 1, higher is better)
        1.0 means perfectly monotonic increase
        
    References:
        Arya, V., et al. (2019). One Explanation Does Not Fit All: A Toolkit and 
        Taxonomy of AI Explainability Techniques. arXiv:1909.03012.
    """
    instance = np.asarray(instance).flatten()
    n_features = len(instance)
    
    # Get baseline values
    baseline_values = compute_baseline_values(
        baseline, background_data, n_features
    )
    
    # Extract attributions as array
    attr_array = _extract_attribution_array(explanation, n_features)
    
    # Sort features by attribution (descending - most important first)
    if use_absolute:
        sorted_indices = np.argsort(-np.abs(attr_array))
    else:
        sorted_indices = np.argsort(-attr_array)
    
    # Determine target class
    if target_class is None:
        # Use predicted class
        pred = get_prediction_value(model, instance.reshape(1, -1))
        if isinstance(pred, np.ndarray) and pred.ndim > 0:
            target_class = int(np.argmax(pred))
        else:
            target_class = 0
    
    # Start from baseline (all features masked)
    current = baseline_values.copy()
    
    # Track predictions as features are revealed
    predictions = []
    
    # Get initial prediction (baseline state)
    pred = get_prediction_value(model, current.reshape(1, -1))
    if isinstance(pred, np.ndarray) and pred.ndim > 0 and len(pred) > target_class:
        predictions.append(pred[target_class])
    else:
        predictions.append(float(pred))
    
    # Add features one by one
    revealed_features = []
    for idx in sorted_indices:
        # Reveal this feature (set to original value)
        revealed_features.append(idx)
        current[idx] = instance[idx]
        
        # Get prediction
        pred = get_prediction_value(model, current.reshape(1, -1))
        if isinstance(pred, np.ndarray) and pred.ndim > 0 and len(pred) > target_class:
            predictions.append(pred[target_class])
        else:
            predictions.append(float(pred))
    
    # Count monotonic increases
    # A step is monotonic if: pred[i+1] >= pred[i] - tolerance
    n_steps = len(predictions) - 1
    if n_steps == 0:
        return 1.0
    
    monotonic_steps = 0
    for i in range(n_steps):
        if predictions[i + 1] >= predictions[i] - tolerance:
            monotonic_steps += 1
    
    return float(monotonic_steps) / float(n_steps)


def compute_batch_monotonicity(
    model,
    X: np.ndarray,
    explanations: List[Explanation],
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    max_samples: int = None,
    use_absolute: bool = True,
) -> Dict[str, float]:
    """
    Compute average Monotonicity over a batch of instances.
    
    Args:
        model: Model adapter
        X: Input data (2D array)
        explanations: List of Explanation objects (one per instance)
        baseline: Baseline for masked features
        max_samples: Maximum number of samples to evaluate
        use_absolute: If True, sort features by absolute attribution value
        
    Returns:
        Dictionary with mean, std, min, max, and count of valid scores
    """
    n_samples = len(explanations)
    if max_samples:
        n_samples = min(n_samples, max_samples)
    
    scores = []
    
    for i in range(n_samples):
        try:
            score = compute_monotonicity(
                model, X[i], explanations[i],
                baseline=baseline, background_data=X,
                use_absolute=use_absolute
            )
            if not np.isnan(score):
                scores.append(score)
        except Exception:
            continue
    
    if not scores:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n_samples": 0}
    
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "n_samples": len(scores),
    }
