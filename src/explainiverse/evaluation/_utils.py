# src/explainiverse/evaluation/_utils.py
"""
Shared utility functions for evaluation metrics.
"""
import numpy as np
import re
from typing import Union, Callable, List, Tuple
from explainiverse.core.explanation import Explanation


def _extract_base_feature_name(feature_str: str) -> str:
    """
    Extract the base feature name from LIME-style feature strings.
    
    LIME returns strings like "petal width (cm) <= 0.80" or "feature_2 > 3.5".
    This extracts just the feature name part.
    
    Args:
        feature_str: Feature string possibly with conditions
        
    Returns:
        Base feature name
    """
    # Remove comparison operators and values
    # Pattern matches: name <= value, name < value, name >= value, name > value, name = value
    patterns = [
        r'^(.+?)\s*<=\s*[\d\.\-]+$',
        r'^(.+?)\s*>=\s*[\d\.\-]+$',
        r'^(.+?)\s*<\s*[\d\.\-]+$',
        r'^(.+?)\s*>\s*[\d\.\-]+$',
        r'^(.+?)\s*=\s*[\d\.\-]+$',
    ]
    
    for pattern in patterns:
        match = re.match(pattern, feature_str.strip())
        if match:
            return match.group(1).strip()
    
    # No match found, return as-is
    return feature_str.strip()


def _match_feature_to_index(
    feature_key: str,
    feature_names: List[str]
) -> int:
    """
    Match a feature key (possibly with LIME conditions) to its index.
    
    Args:
        feature_key: Feature name from explanation (may include conditions)
        feature_names: List of original feature names
        
    Returns:
        Index of the matching feature, or -1 if not found
    """
    # Try exact match first
    if feature_key in feature_names:
        return feature_names.index(feature_key)
    
    # Try extracting base name
    base_name = _extract_base_feature_name(feature_key)
    if base_name in feature_names:
        return feature_names.index(base_name)
    
    # Try partial matching (feature name is contained in key)
    for i, fname in enumerate(feature_names):
        if fname in feature_key:
            return i
    
    # Try index extraction from patterns like "feature_2" or "f2" or "feat_2"
    patterns = [
        r'feature[_\s]*(\d+)',
        r'feat[_\s]*(\d+)',
        r'^f(\d+)$',
        r'^x(\d+)$',
    ]
    for pattern in patterns:
        match = re.search(pattern, feature_key, re.IGNORECASE)
        if match:
            idx = int(match.group(1))
            if 0 <= idx < len(feature_names):
                return idx
    
    return -1


def get_sorted_feature_indices(
    explanation: Explanation,
    descending: bool = True
) -> List[int]:
    """
    Extract feature indices sorted by absolute attribution value.
    
    Handles various feature naming conventions:
    - Clean names: "sepal length", "feature_0"
    - LIME-style: "sepal length <= 5.0", "feature_0 > 2.3"
    - Indexed: "f0", "x1", "feat_2"
    
    Args:
        explanation: Explanation object with feature_attributions
        descending: If True, sort from most to least important
        
    Returns:
        List of feature indices sorted by importance
    """
    attributions = explanation.explanation_data.get("feature_attributions", {})
    
    if not attributions:
        raise ValueError("No feature attributions found in explanation.")
    
    # Sort features by absolute importance
    sorted_features = sorted(
        attributions.items(),
        key=lambda x: abs(x[1]),
        reverse=descending
    )
    
    # Map feature names to indices
    feature_indices = []
    feature_names = getattr(explanation, 'feature_names', None)
    
    for i, (fname, _) in enumerate(sorted_features):
        if feature_names is not None:
            idx = _match_feature_to_index(fname, feature_names)
            if idx >= 0:
                feature_indices.append(idx)
            else:
                # Fallback: use position in sorted list
                feature_indices.append(i % len(feature_names))
        else:
            # No feature_names available - try to extract index from name
            patterns = [
                r'feature[_\s]*(\d+)',
                r'feat[_\s]*(\d+)',
                r'^f(\d+)',
                r'^x(\d+)',
            ]
            found = False
            for pattern in patterns:
                match = re.search(pattern, fname, re.IGNORECASE)
                if match:
                    feature_indices.append(int(match.group(1)))
                    found = True
                    break
            if not found:
                feature_indices.append(i)
    
    return feature_indices


def compute_baseline_values(
    baseline: Union[str, float, np.ndarray, Callable],
    background_data: np.ndarray = None,
    n_features: int = None
) -> np.ndarray:
    """
    Compute per-feature baseline values for perturbation.
    
    Args:
        baseline: Baseline specification - one of:
            - "mean": Use mean of background_data
            - "median": Use median of background_data
            - float/int: Use this value for all features
            - np.ndarray: Use these values directly (must match n_features)
            - Callable: Function that takes background_data and returns baseline array
        background_data: Reference data for computing statistics (required for "mean"/"median")
        n_features: Number of features (required if baseline is scalar)
        
    Returns:
        1D numpy array of baseline values, one per feature
    """
    if isinstance(baseline, str):
        if background_data is None:
            raise ValueError(f"background_data required for baseline='{baseline}'")
        if baseline == "mean":
            return np.mean(background_data, axis=0)
        elif baseline == "median":
            return np.median(background_data, axis=0)
        else:
            raise ValueError(f"Unsupported string baseline: {baseline}")
    
    elif callable(baseline):
        if background_data is None:
            raise ValueError("background_data required for callable baseline")
        result = baseline(background_data)
        return np.asarray(result)
    
    elif isinstance(baseline, np.ndarray):
        return baseline
    
    elif isinstance(baseline, (float, int, np.number)):
        if n_features is None:
            raise ValueError("n_features required for scalar baseline")
        return np.full(n_features, baseline)
    
    else:
        raise ValueError(f"Invalid baseline type: {type(baseline)}")


def apply_feature_mask(
    instance: np.ndarray,
    feature_indices: List[int],
    baseline_values: np.ndarray
) -> np.ndarray:
    """
    Replace specified features with baseline values.
    
    Args:
        instance: Original instance (1D array)
        feature_indices: Indices of features to replace
        baseline_values: Per-feature baseline values
        
    Returns:
        Modified instance with specified features replaced
    """
    modified = instance.copy()
    for idx in feature_indices:
        if idx < len(modified) and idx < len(baseline_values):
            modified[idx] = baseline_values[idx]
    return modified


def resolve_k(k: Union[int, float], n_features: int) -> int:
    """
    Resolve k to an integer number of features.
    
    Args:
        k: Either an integer count or a float fraction (0-1)
        n_features: Total number of features
        
    Returns:
        Integer number of features
    """
    if isinstance(k, float) and 0 < k <= 1:
        return max(1, int(k * n_features))
    elif isinstance(k, int) and k > 0:
        return min(k, n_features)
    else:
        raise ValueError(f"k must be positive int or float in (0, 1], got {k}")


def get_prediction_value(
    model,
    instance: np.ndarray,
    output_type: str = "probability"
) -> float:
    """
    Get a scalar prediction value from a model.
    
    Works with both raw sklearn models and explainiverse adapters.
    For adapters, .predict() typically returns probabilities.
    
    Args:
        model: Model adapter with predict/predict_proba methods
        instance: Single instance (1D array)
        output_type: "probability" (max prob) or "class" (predicted class)
        
    Returns:
        Scalar prediction value
    """
    instance_2d = instance.reshape(1, -1)
    
    if output_type == "probability":
        # Try predict_proba first (raw sklearn model)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(instance_2d)
            if isinstance(proba, np.ndarray):
                if proba.ndim == 2:
                    return float(np.max(proba[0]))
                return float(np.max(proba))
            return float(np.max(proba[0]))
        
        # Fall back to predict (adapter returns probs from predict)
        pred = model.predict(instance_2d)
        if isinstance(pred, np.ndarray):
            if pred.ndim == 2:
                return float(np.max(pred[0]))
            elif pred.ndim == 1:
                return float(np.max(pred))
        return float(pred[0]) if hasattr(pred, '__getitem__') else float(pred)
    
    elif output_type == "class":
        # For class prediction, use argmax of probabilities
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(instance_2d)
            return float(np.argmax(proba[0]))
        pred = model.predict(instance_2d)
        if isinstance(pred, np.ndarray) and pred.ndim == 2:
            return float(np.argmax(pred[0]))
        return float(pred[0]) if hasattr(pred, '__getitem__') else float(pred)
    
    else:
        raise ValueError(f"Unknown output_type: {output_type}")


def compute_prediction_change(
    model,
    original: np.ndarray,
    perturbed: np.ndarray,
    metric: str = "absolute"
) -> float:
    """
    Compute the change in prediction between original and perturbed instances.
    
    Args:
        model: Model adapter
        original: Original instance
        perturbed: Perturbed instance
        metric: "absolute" for |p1 - p2|, "relative" for |p1 - p2| / p1
        
    Returns:
        Prediction change value
    """
    orig_pred = get_prediction_value(model, original)
    pert_pred = get_prediction_value(model, perturbed)
    
    if metric == "absolute":
        return abs(orig_pred - pert_pred)
    elif metric == "relative":
        if abs(orig_pred) < 1e-10:
            return abs(pert_pred)
        return abs(orig_pred - pert_pred) / abs(orig_pred)
    else:
        raise ValueError(f"Unknown metric: {metric}")
