# src/explainiverse/evaluation/faithfulness.py
"""
Faithfulness evaluation metrics for explanations.

Implements:
- PGI (Prediction Gap on Important features)
- PGU (Prediction Gap on Unimportant features)
- Faithfulness Correlation
- Comprehensiveness and Sufficiency
"""
import numpy as np
import pandas as pd
from typing import Union, Callable, List, Dict, Optional
from explainiverse.core.explanation import Explanation
from explainiverse.evaluation._utils import (
    get_sorted_feature_indices,
    compute_baseline_values,
    apply_feature_mask,
    resolve_k,
    get_prediction_value,
    compute_prediction_change,
)


def compute_pgi(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    k: Union[int, float] = 0.2,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    background_data: np.ndarray = None,
) -> float:
    """
    Compute Prediction Gap on Important features (PGI).
    
    Measures prediction change when removing the top-k most important features.
    Higher PGI indicates the explanation correctly identified important features.
    
    Args:
        model: Model adapter with predict/predict_proba method
        instance: Input instance (1D array)
        explanation: Explanation object with feature_attributions
        k: Number of top features to remove (int) or fraction (float 0-1)
        baseline: Baseline for feature replacement ("mean", "median", scalar, array, callable)
        background_data: Reference data for computing baseline (required for "mean"/"median")
        
    Returns:
        PGI score (higher = explanation identified truly important features)
    """
    instance = np.asarray(instance).flatten()
    n_features = len(instance)
    
    # Resolve k to integer
    k_int = resolve_k(k, n_features)
    
    # Get feature indices sorted by importance (most important first)
    sorted_indices = get_sorted_feature_indices(explanation, descending=True)
    top_k_indices = sorted_indices[:k_int]
    
    # Compute baseline values
    baseline_values = compute_baseline_values(
        baseline, background_data, n_features
    )
    
    # Perturb instance by removing top-k important features
    perturbed = apply_feature_mask(instance, top_k_indices, baseline_values)
    
    # Compute prediction change
    return compute_prediction_change(model, instance, perturbed, metric="absolute")


def compute_pgu(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    k: Union[int, float] = 0.2,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    background_data: np.ndarray = None,
) -> float:
    """
    Compute Prediction Gap on Unimportant features (PGU).
    
    Measures prediction change when removing the bottom-k least important features.
    Lower PGU indicates the explanation correctly identified unimportant features.
    
    Args:
        model: Model adapter with predict/predict_proba method
        instance: Input instance (1D array)
        explanation: Explanation object with feature_attributions
        k: Number of bottom features to remove (int) or fraction (float 0-1)
        baseline: Baseline for feature replacement ("mean", "median", scalar, array, callable)
        background_data: Reference data for computing baseline (required for "mean"/"median")
        
    Returns:
        PGU score (lower = explanation correctly identified unimportant features)
    """
    instance = np.asarray(instance).flatten()
    n_features = len(instance)
    
    # Resolve k to integer
    k_int = resolve_k(k, n_features)
    
    # Get feature indices sorted by importance (least important first for PGU)
    sorted_indices = get_sorted_feature_indices(explanation, descending=False)
    bottom_k_indices = sorted_indices[:k_int]
    
    # Compute baseline values
    baseline_values = compute_baseline_values(
        baseline, background_data, n_features
    )
    
    # Perturb instance by removing bottom-k unimportant features
    perturbed = apply_feature_mask(instance, bottom_k_indices, baseline_values)
    
    # Compute prediction change
    return compute_prediction_change(model, instance, perturbed, metric="absolute")


def compute_faithfulness_score(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    k: Union[int, float] = 0.2,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    background_data: np.ndarray = None,
    epsilon: float = 1e-7,
) -> Dict[str, float]:
    """
    Compute combined faithfulness metrics.
    
    Args:
        model: Model adapter
        instance: Input instance (1D array)
        explanation: Explanation object
        k: Number/fraction of features for PGI/PGU
        baseline: Baseline for feature replacement
        background_data: Reference data for baseline computation
        epsilon: Small constant to avoid division by zero
        
    Returns:
        Dictionary containing:
            - pgi: Prediction Gap on Important features
            - pgu: Prediction Gap on Unimportant features  
            - faithfulness_ratio: PGI / (PGU + epsilon) - higher is better
            - faithfulness_diff: PGI - PGU - higher is better
    """
    pgi = compute_pgi(model, instance, explanation, k, baseline, background_data)
    pgu = compute_pgu(model, instance, explanation, k, baseline, background_data)
    
    return {
        "pgi": pgi,
        "pgu": pgu,
        "faithfulness_ratio": pgi / (pgu + epsilon),
        "faithfulness_diff": pgi - pgu,
    }


def compute_comprehensiveness(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    k_values: List[Union[int, float]] = None,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    background_data: np.ndarray = None,
) -> Dict[str, float]:
    """
    Compute comprehensiveness - how much prediction drops when removing important features.
    
    This is essentially PGI computed at multiple k values and averaged.
    Higher comprehensiveness = better explanation.
    
    Args:
        model: Model adapter
        instance: Input instance
        explanation: Explanation object
        k_values: List of k values to evaluate (default: [0.1, 0.2, 0.3])
        baseline: Baseline for feature replacement
        background_data: Reference data
        
    Returns:
        Dictionary with per-k scores and mean comprehensiveness
    """
    if k_values is None:
        k_values = [0.1, 0.2, 0.3]
    
    scores = {}
    for k in k_values:
        score = compute_pgi(model, instance, explanation, k, baseline, background_data)
        scores[f"comp_k{k}"] = score
    
    scores["comprehensiveness"] = np.mean(list(scores.values()))
    return scores


def compute_sufficiency(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    k_values: List[Union[int, float]] = None,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    background_data: np.ndarray = None,
) -> Dict[str, float]:
    """
    Compute sufficiency - how much prediction is preserved when keeping only important features.
    
    Lower sufficiency = the important features alone are sufficient for prediction.
    
    Args:
        model: Model adapter
        instance: Input instance
        explanation: Explanation object  
        k_values: List of k values (fraction of features to KEEP)
        baseline: Baseline for feature replacement
        background_data: Reference data
        
    Returns:
        Dictionary with per-k scores and mean sufficiency
    """
    if k_values is None:
        k_values = [0.1, 0.2, 0.3]
    
    instance = np.asarray(instance).flatten()
    n_features = len(instance)
    
    # Get baseline values
    baseline_values = compute_baseline_values(baseline, background_data, n_features)
    
    # Get sorted indices (most important first)
    sorted_indices = get_sorted_feature_indices(explanation, descending=True)
    
    scores = {}
    for k in k_values:
        k_int = resolve_k(k, n_features)
        
        # Keep only top-k features, replace rest with baseline
        top_k_set = set(sorted_indices[:k_int])
        indices_to_mask = [i for i in range(n_features) if i not in top_k_set]
        
        perturbed = apply_feature_mask(instance, indices_to_mask, baseline_values)
        change = compute_prediction_change(model, instance, perturbed, metric="absolute")
        scores[f"suff_k{k}"] = change
    
    scores["sufficiency"] = np.mean([v for k, v in scores.items() if k.startswith("suff_k")])
    return scores


def compute_faithfulness_correlation(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    background_data: np.ndarray = None,
    n_steps: int = None,
) -> float:
    """
    Compute faithfulness correlation between attributions and prediction changes.
    
    Measures correlation between feature importance ranking and actual impact
    on predictions when features are removed one at a time.
    
    Args:
        model: Model adapter
        instance: Input instance
        explanation: Explanation object
        baseline: Baseline for feature replacement
        background_data: Reference data
        n_steps: Number of features to evaluate (default: all features)
        
    Returns:
        Pearson correlation coefficient (-1 to 1, higher is better)
    """
    instance = np.asarray(instance).flatten()
    n_features = len(instance)
    
    if n_steps is None:
        n_steps = n_features
    n_steps = min(n_steps, n_features)
    
    # Get attributions
    attributions = explanation.explanation_data.get("feature_attributions", {})
    sorted_indices = get_sorted_feature_indices(explanation, descending=True)[:n_steps]
    
    # Get baseline
    baseline_values = compute_baseline_values(baseline, background_data, n_features)
    
    # Compute importance values and prediction changes for each feature
    importance_values = []
    prediction_changes = []
    
    feature_names = getattr(explanation, 'feature_names', None)
    
    for idx in sorted_indices:
        # Get importance value for this feature
        if feature_names and idx < len(feature_names):
            fname = feature_names[idx]
        else:
            # Try common naming patterns
            for pattern in [f"feature_{idx}", f"f{idx}", f"feat_{idx}"]:
                if pattern in attributions:
                    fname = pattern
                    break
            else:
                fname = list(attributions.keys())[sorted_indices.index(idx)] if idx < len(attributions) else None
        
        if fname and fname in attributions:
            importance_values.append(abs(attributions[fname]))
        else:
            continue
        
        # Compute prediction change when removing this single feature
        perturbed = apply_feature_mask(instance, [idx], baseline_values)
        change = compute_prediction_change(model, instance, perturbed, metric="absolute")
        prediction_changes.append(change)
    
    if len(importance_values) < 2:
        return 0.0  # Not enough data points
    
    # Compute Pearson correlation
    return float(np.corrcoef(importance_values, prediction_changes)[0, 1])


def compare_explainer_faithfulness(
    model,
    X: np.ndarray,
    explanations: Dict[str, List[Explanation]],
    k: Union[int, float] = 0.2,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
    max_samples: int = None,
) -> pd.DataFrame:
    """
    Compare multiple explainers on faithfulness metrics across a dataset.
    
    Args:
        model: Model adapter
        X: Input data (2D array, n_samples x n_features)
        explanations: Dict mapping explainer names to lists of Explanation objects
        k: Number/fraction of features for PGI/PGU
        baseline: Baseline for feature replacement
        max_samples: Limit number of samples to evaluate (None = all)
        
    Returns:
        DataFrame with columns: [explainer, mean_pgi, std_pgi, mean_pgu, std_pgu, 
                                 mean_ratio, mean_diff, n_samples]
    """
    results = []
    
    for explainer_name, expl_list in explanations.items():
        n_samples = len(expl_list)
        if max_samples:
            n_samples = min(n_samples, max_samples)
        
        pgi_scores = []
        pgu_scores = []
        
        for i in range(n_samples):
            instance = X[i]
            exp = expl_list[i]
            
            try:
                scores = compute_faithfulness_score(
                    model, instance, exp, k, baseline, X
                )
                pgi_scores.append(scores["pgi"])
                pgu_scores.append(scores["pgu"])
            except Exception as e:
                # Skip instances that fail
                continue
        
        if pgi_scores:
            results.append({
                "explainer": explainer_name,
                "mean_pgi": np.mean(pgi_scores),
                "std_pgi": np.std(pgi_scores),
                "mean_pgu": np.mean(pgu_scores),
                "std_pgu": np.std(pgu_scores),
                "mean_ratio": np.mean(pgi_scores) / (np.mean(pgu_scores) + 1e-7),
                "mean_diff": np.mean(pgi_scores) - np.mean(pgu_scores),
                "n_samples": len(pgi_scores),
            })
    
    return pd.DataFrame(results)


def compute_batch_faithfulness(
    model,
    X: np.ndarray,
    explanations: List[Explanation],
    k: Union[int, float] = 0.2,
    baseline: Union[str, float, np.ndarray, Callable] = "mean",
) -> Dict[str, float]:
    """
    Compute average faithfulness metrics over a batch of instances.
    
    Args:
        model: Model adapter
        X: Input data (2D array)
        explanations: List of Explanation objects (one per instance)
        k: Number/fraction of features for PGI/PGU
        baseline: Baseline for feature replacement
        
    Returns:
        Dictionary with aggregated metrics
    """
    pgi_scores = []
    pgu_scores = []
    
    for i, exp in enumerate(explanations):
        try:
            scores = compute_faithfulness_score(
                model, X[i], exp, k, baseline, X
            )
            pgi_scores.append(scores["pgi"])
            pgu_scores.append(scores["pgu"])
        except Exception:
            continue
    
    if not pgi_scores:
        return {"mean_pgi": 0.0, "mean_pgu": 0.0, "mean_ratio": 0.0, "n_samples": 0}
    
    return {
        "mean_pgi": np.mean(pgi_scores),
        "std_pgi": np.std(pgi_scores),
        "mean_pgu": np.mean(pgu_scores),
        "std_pgu": np.std(pgu_scores),
        "mean_ratio": np.mean(pgi_scores) / (np.mean(pgu_scores) + 1e-7),
        "mean_diff": np.mean(pgi_scores) - np.mean(pgu_scores),
        "n_samples": len(pgi_scores),
    }
