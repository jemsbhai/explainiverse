# src/explainiverse/evaluation/metrics.py
"""
Legacy evaluation metrics: AOPC and ROAR.

For comprehensive evaluation, prefer the metrics in faithfulness.py
and stability.py which have better edge case handling.
"""

import numpy as np
import re
from typing import List, Dict, Optional, Union, Callable
from explainiverse.core.explanation import Explanation
from sklearn.metrics import accuracy_score
import copy


def _extract_feature_index(
    feature_name: str,
    feature_names: Optional[List[str]] = None,
    fallback_index: int = 0
) -> int:
    """
    Extract feature index from a feature name string.
    
    Handles various naming conventions including LIME-style conditions
    like "feature_0 <= 5.0".
    
    Args:
        feature_name: Feature name (possibly with conditions)
        feature_names: Optional list of canonical feature names
        fallback_index: Index to return if extraction fails
        
    Returns:
        Feature index
    """
    # Try exact match first
    if feature_names is not None:
        if feature_name in feature_names:
            return feature_names.index(feature_name)
        
        # Extract base name (remove LIME-style conditions)
        base_name = re.sub(r'\s*[<>=!]+\s*[\d.\-]+$', '', feature_name).strip()
        if base_name in feature_names:
            return feature_names.index(base_name)
        
        # Try partial match (feature name contained in key)
        for i, fname in enumerate(feature_names):
            if fname in feature_name:
                return i
    
    # Try extracting index from patterns like "feature_2", "f2", "x2"
    patterns = [
        r'feature[_\s]*(\d+)',
        r'feat[_\s]*(\d+)',
        r'^f(\d+)$',
        r'^x(\d+)$',
    ]
    for pattern in patterns:
        match = re.search(pattern, feature_name, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    return fallback_index


def compute_aopc(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    num_steps: int = 10,
    baseline_value: float = 0.0
) -> float:
    """
    Compute Area Over the Perturbation Curve (AOPC).
    
    AOPC measures explanation faithfulness by iteratively removing
    the most important features and measuring prediction change.

    Args:
        model: Model adapter with .predict() method
        instance: Input sample (1D array)
        explanation: Explanation object with feature_attributions
        num_steps: Number of top features to remove
        baseline_value: Value to replace removed features with

    Returns:
        AOPC score (higher = more faithful explanation)
    """
    instance = np.asarray(instance).flatten()
    n_features = len(instance)
    
    base_pred = model.predict(instance.reshape(1, -1))[0]
    if hasattr(base_pred, '__len__') and len(base_pred) > 1:
        base_pred = float(np.max(base_pred))
    else:
        base_pred = float(base_pred)
    
    attributions = explanation.explanation_data.get("feature_attributions", {})
    if not attributions:
        raise ValueError("No feature attributions found in explanation.")

    # Sort features by absolute importance (most important first)
    sorted_features = sorted(
        attributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    # Get feature_names from explanation (may be None)
    feature_names = getattr(explanation, 'feature_names', None)
    
    # Map feature names to indices
    feature_indices = []
    for i, (fname, _) in enumerate(sorted_features):
        idx = _extract_feature_index(fname, feature_names, fallback_index=i)
        if 0 <= idx < n_features:
            feature_indices.append(idx)

    deltas = []
    modified = instance.copy()

    for i in range(min(num_steps, len(feature_indices))):
        idx = feature_indices[i]
        modified[idx] = baseline_value
        
        new_pred = model.predict(modified.reshape(1, -1))[0]
        if hasattr(new_pred, '__len__') and len(new_pred) > 1:
            new_pred = float(np.max(new_pred))
        else:
            new_pred = float(new_pred)
        
        delta = abs(base_pred - new_pred)
        deltas.append(delta)

    return float(np.mean(deltas)) if deltas else 0.0


def compute_batch_aopc(
    model,
    X: np.ndarray,
    explanations: Dict[str, List[Explanation]],
    num_steps: int = 10,
    baseline_value: float = 0.0
) -> Dict[str, float]:
    """
    Compute average AOPC across multiple explainers and instances.

    Args:
        model: Model adapter
        X: 2D input array (n_samples, n_features)
        explanations: Dict mapping explainer names to lists of Explanation objects
        num_steps: Number of top features to remove
        baseline_value: Value to replace features with

    Returns:
        Dict mapping explainer names to mean AOPC scores
    """
    results = {}

    for explainer_name, expl_list in explanations.items():
        scores = []
        for i, exp in enumerate(expl_list):
            if i >= len(X):
                break
            try:
                score = compute_aopc(model, X[i], exp, num_steps, baseline_value)
                scores.append(score)
            except Exception:
                continue
        results[explainer_name] = float(np.mean(scores)) if scores else 0.0

    return results


def compute_roar(
    model_class,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    explanations: List[Explanation],
    top_k: int = 3,
    baseline_value: Union[str, float, np.ndarray, Callable] = 0.0,
    model_kwargs: Optional[Dict] = None
) -> float:
    """
    Compute ROAR (Remove And Retrain) score.
    
    ROAR retrains the model after removing top-k important features
    and measures the accuracy drop.

    Args:
        model_class: Uninstantiated model class (e.g., LogisticRegression)
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        explanations: List of Explanation objects (one per training instance)
        top_k: Number of top features to remove
        baseline_value: Replacement value for removed features:
            - float/int: constant value
            - "mean": per-feature mean from X_train
            - "median": per-feature median from X_train
            - np.ndarray: per-feature values
            - callable: function(X_train) -> per-feature values
        model_kwargs: Optional kwargs for model_class

    Returns:
        Accuracy drop (baseline_acc - retrained_acc)
    """
    model_kwargs = model_kwargs or {}
    n_features = X_train.shape[1]

    # Train baseline model
    baseline_model = model_class(**model_kwargs)
    baseline_model.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, baseline_model.predict(X_test))

    # Collect top-k feature indices via voting across explanations
    feature_votes: Dict[int, int] = {}
    
    for exp in explanations:
        attributions = exp.explanation_data.get("feature_attributions", {})
        if not attributions:
            continue
        
        # Get feature_names from explanation
        feature_names = getattr(exp, 'feature_names', None)
        
        # Get top-k features by absolute importance
        sorted_attrs = sorted(
            attributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_k]
        
        for i, (fname, _) in enumerate(sorted_attrs):
            idx = _extract_feature_index(fname, feature_names, fallback_index=i)
            if 0 <= idx < n_features:
                feature_votes[idx] = feature_votes.get(idx, 0) + 1

    # Select most voted features
    top_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_indices = [idx for idx, _ in top_features]
    
    if not top_indices:
        return 0.0

    # Compute baseline values
    if isinstance(baseline_value, str):
        if baseline_value == "mean":
            feature_baseline = np.mean(X_train, axis=0)
        elif baseline_value == "median":
            feature_baseline = np.median(X_train, axis=0)
        else:
            raise ValueError(f"Unsupported baseline: {baseline_value}")
    elif callable(baseline_value):
        feature_baseline = baseline_value(X_train)
    elif isinstance(baseline_value, np.ndarray):
        feature_baseline = baseline_value
    else:
        feature_baseline = np.full(n_features, float(baseline_value))

    # Remove features
    X_train_mod = X_train.copy()
    X_test_mod = X_test.copy()
    
    for idx in top_indices:
        X_train_mod[:, idx] = feature_baseline[idx]
        X_test_mod[:, idx] = feature_baseline[idx]

    # Retrain and evaluate
    retrained_model = model_class(**model_kwargs)
    retrained_model.fit(X_train_mod, y_train)
    retrained_acc = accuracy_score(y_test, retrained_model.predict(X_test_mod))

    return float(baseline_acc - retrained_acc)


def compute_roar_curve(
    model_class,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    explanations: List[Explanation],
    max_k: int = 5,
    baseline_value: Union[str, float, np.ndarray, Callable] = "mean",
    model_kwargs: Optional[Dict] = None
) -> Dict[int, float]:
    """
    Compute ROAR scores for k=1 to max_k.

    Returns:
        Dict mapping k to accuracy drop
    """
    model_kwargs = model_kwargs or {}
    curve = {}

    for k in range(1, max_k + 1):
        acc_drop = compute_roar(
            model_class=model_class,
            X_train=X_train.copy(),
            y_train=y_train.copy(),
            X_test=X_test.copy(),
            y_test=y_test.copy(),
            explanations=explanations,
            top_k=k,
            baseline_value=baseline_value,
            model_kwargs=model_kwargs
        )
        curve[k] = acc_drop

    return curve
