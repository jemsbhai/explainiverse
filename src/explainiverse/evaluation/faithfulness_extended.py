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
"""
import numpy as np
import re
from typing import Union, Callable, List, Dict, Optional, Tuple
from scipy import stats

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
    auc = np.trapz(curve, x)
    
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
    auc = np.trapz(curve, x)
    
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
