# src/explainiverse/evaluation/stability.py
"""
Stability evaluation metrics for explanations.

Implements:
- RIS (Relative Input Stability) - sensitivity to input perturbations
- ROS (Relative Output Stability) - consistency with similar predictions
- Lipschitz Estimate - local smoothness of explanations
"""
import numpy as np
from typing import Union, Callable, List, Dict, Optional, Tuple
from explainiverse.core.explanation import Explanation
from explainiverse.core.explainer import BaseExplainer
from explainiverse.evaluation._utils import get_prediction_value


def _extract_attribution_vector(explanation: Explanation) -> np.ndarray:
    """
    Extract attribution values as a numpy array from an Explanation.
    
    Args:
        explanation: Explanation object with feature_attributions
        
    Returns:
        1D numpy array of attribution values
    """
    attributions = explanation.explanation_data.get("feature_attributions", {})
    if not attributions:
        raise ValueError("No feature attributions found in explanation.")
    
    # Get values in consistent order
    feature_names = getattr(explanation, 'feature_names', None)
    if feature_names:
        values = [attributions.get(fn, 0.0) for fn in feature_names]
    else:
        values = list(attributions.values())
    
    return np.array(values, dtype=float)


def _normalize_vector(v: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(v)
    if norm < epsilon:
        return v
    return v / norm


def compute_ris(
    explainer: BaseExplainer,
    instance: np.ndarray,
    n_perturbations: int = 10,
    noise_scale: float = 0.01,
    seed: int = None,
) -> float:
    """
    Compute Relative Input Stability (RIS).
    
    Measures how stable explanations are to small perturbations in the input.
    Lower RIS indicates more stable explanations.
    
    RIS = mean(||E(x) - E(x')|| / ||x - x'||) for perturbed inputs x'
    
    Args:
        explainer: Explainer instance with .explain() method
        instance: Original input instance (1D array)
        n_perturbations: Number of perturbed samples to generate
        noise_scale: Standard deviation of Gaussian noise (relative to feature range)
        seed: Random seed for reproducibility
        
    Returns:
        RIS score (lower = more stable)
    """
    if seed is not None:
        np.random.seed(seed)
    
    instance = np.asarray(instance).flatten()
    n_features = len(instance)
    
    # Get original explanation
    original_exp = explainer.explain(instance)
    original_exp.feature_names = getattr(original_exp, 'feature_names', None) or \
                                  [f"feature_{i}" for i in range(n_features)]
    original_attr = _extract_attribution_vector(original_exp)
    
    ratios = []
    
    for _ in range(n_perturbations):
        # Generate perturbed input
        noise = np.random.normal(0, noise_scale, n_features)
        perturbed = instance + noise * np.abs(instance + 1e-10)  # Scale noise by feature magnitude
        
        # Get perturbed explanation
        try:
            perturbed_exp = explainer.explain(perturbed)
            perturbed_exp.feature_names = original_exp.feature_names
            perturbed_attr = _extract_attribution_vector(perturbed_exp)
        except Exception:
            continue
        
        # Compute ratio of changes
        attr_diff = np.linalg.norm(original_attr - perturbed_attr)
        input_diff = np.linalg.norm(instance - perturbed)
        
        if input_diff > 1e-10:
            ratios.append(attr_diff / input_diff)
    
    if not ratios:
        return float('inf')
    
    return float(np.mean(ratios))


def compute_ros(
    explainer: BaseExplainer,
    model,
    instance: np.ndarray,
    reference_instances: np.ndarray,
    n_neighbors: int = 5,
    prediction_threshold: float = 0.05,
) -> float:
    """
    Compute Relative Output Stability (ROS).
    
    Measures how similar explanations are for instances with similar predictions.
    Higher ROS indicates more consistent explanations.
    
    Args:
        explainer: Explainer instance with .explain() method
        model: Model adapter with predict/predict_proba method
        instance: Query instance
        reference_instances: Pool of reference instances to find neighbors
        n_neighbors: Number of neighbors to compare
        prediction_threshold: Maximum prediction difference to consider "similar"
        
    Returns:
        ROS score (higher = more consistent for similar predictions)
    """
    instance = np.asarray(instance).flatten()
    n_features = len(instance)
    
    # Get prediction for query instance
    query_pred = get_prediction_value(model, instance)
    
    # Find instances with similar predictions
    similar_instances = []
    for ref in reference_instances:
        ref = np.asarray(ref).flatten()
        ref_pred = get_prediction_value(model, ref)
        if abs(query_pred - ref_pred) <= prediction_threshold:
            similar_instances.append(ref)
    
    if len(similar_instances) < 2:
        return 1.0  # Perfect stability if no similar instances
    
    # Limit to n_neighbors
    similar_instances = similar_instances[:n_neighbors]
    
    # Get explanation for query
    query_exp = explainer.explain(instance)
    query_exp.feature_names = getattr(query_exp, 'feature_names', None) or \
                               [f"feature_{i}" for i in range(n_features)]
    query_attr = _normalize_vector(_extract_attribution_vector(query_exp))
    
    # Get explanations for similar instances and compute similarity
    similarities = []
    for ref in similar_instances:
        try:
            ref_exp = explainer.explain(ref)
            ref_exp.feature_names = query_exp.feature_names
            ref_attr = _normalize_vector(_extract_attribution_vector(ref_exp))
            
            # Cosine similarity
            similarity = np.dot(query_attr, ref_attr)
            similarities.append(similarity)
        except Exception:
            continue
    
    if not similarities:
        return 1.0
    
    return float(np.mean(similarities))


def compute_lipschitz_estimate(
    explainer: BaseExplainer,
    instance: np.ndarray,
    n_samples: int = 20,
    radius: float = 0.1,
    seed: int = None,
) -> float:
    """
    Estimate local Lipschitz constant of the explanation function.
    
    The Lipschitz constant bounds how fast explanations can change:
    ||E(x) - E(y)|| <= L * ||x - y||
    
    Lower L indicates smoother, more stable explanations.
    
    Args:
        explainer: Explainer instance
        instance: Center point for local estimate
        n_samples: Number of sample pairs to evaluate
        radius: Radius of ball around instance to sample from
        seed: Random seed
        
    Returns:
        Estimated local Lipschitz constant (lower = smoother)
    """
    if seed is not None:
        np.random.seed(seed)
    
    instance = np.asarray(instance).flatten()
    n_features = len(instance)
    
    max_ratio = 0.0
    
    for _ in range(n_samples):
        # Generate two random points in a ball around instance
        direction1 = np.random.randn(n_features)
        direction1 = direction1 / np.linalg.norm(direction1)
        r1 = np.random.uniform(0, radius)
        point1 = instance + r1 * direction1
        
        direction2 = np.random.randn(n_features)
        direction2 = direction2 / np.linalg.norm(direction2)
        r2 = np.random.uniform(0, radius)
        point2 = instance + r2 * direction2
        
        try:
            exp1 = explainer.explain(point1)
            exp1.feature_names = [f"feature_{i}" for i in range(n_features)]
            attr1 = _extract_attribution_vector(exp1)
            
            exp2 = explainer.explain(point2)
            exp2.feature_names = exp1.feature_names
            attr2 = _extract_attribution_vector(exp2)
        except Exception:
            continue
        
        attr_diff = np.linalg.norm(attr1 - attr2)
        input_diff = np.linalg.norm(point1 - point2)
        
        if input_diff > 1e-10:
            ratio = attr_diff / input_diff
            max_ratio = max(max_ratio, ratio)
    
    return float(max_ratio)


def compute_stability_metrics(
    explainer: BaseExplainer,
    model,
    instance: np.ndarray,
    background_data: np.ndarray,
    n_perturbations: int = 10,
    noise_scale: float = 0.01,
    n_neighbors: int = 5,
    seed: int = None,
) -> Dict[str, float]:
    """
    Compute comprehensive stability metrics for a single instance.
    
    Args:
        explainer: Explainer instance
        model: Model adapter
        instance: Query instance
        background_data: Reference data for ROS computation
        n_perturbations: Number of perturbations for RIS
        noise_scale: Noise scale for RIS
        n_neighbors: Number of neighbors for ROS
        seed: Random seed
        
    Returns:
        Dictionary with RIS, ROS, and Lipschitz estimate
    """
    return {
        "ris": compute_ris(explainer, instance, n_perturbations, noise_scale, seed),
        "ros": compute_ros(explainer, model, instance, background_data, n_neighbors),
        "lipschitz": compute_lipschitz_estimate(explainer, instance, seed=seed),
    }


def compute_batch_stability(
    explainer: BaseExplainer,
    model,
    X: np.ndarray,
    n_perturbations: int = 10,
    noise_scale: float = 0.01,
    max_samples: int = None,
    seed: int = None,
) -> Dict[str, float]:
    """
    Compute average stability metrics over a batch of instances.
    
    Args:
        explainer: Explainer instance
        model: Model adapter
        X: Input data (2D array)
        n_perturbations: Number of perturbations per instance
        noise_scale: Noise scale for perturbations
        max_samples: Maximum number of samples to evaluate
        seed: Random seed
        
    Returns:
        Dictionary with mean and std of stability metrics
    """
    n_samples = len(X)
    if max_samples:
        n_samples = min(n_samples, max_samples)
    
    ris_scores = []
    ros_scores = []
    
    for i in range(n_samples):
        instance = X[i]
        
        try:
            ris = compute_ris(explainer, instance, n_perturbations, noise_scale, seed)
            if not np.isinf(ris):
                ris_scores.append(ris)
            
            ros = compute_ros(explainer, model, instance, X, n_neighbors=5)
            ros_scores.append(ros)
        except Exception:
            continue
    
    results = {"n_samples": len(ris_scores)}
    
    if ris_scores:
        results["mean_ris"] = np.mean(ris_scores)
        results["std_ris"] = np.std(ris_scores)
    else:
        results["mean_ris"] = float('inf')
        results["std_ris"] = 0.0
    
    if ros_scores:
        results["mean_ros"] = np.mean(ros_scores)
        results["std_ros"] = np.std(ros_scores)
    else:
        results["mean_ros"] = 0.0
        results["std_ros"] = 0.0
    
    return results


def compare_explainer_stability(
    explainers: Dict[str, BaseExplainer],
    model,
    X: np.ndarray,
    n_perturbations: int = 5,
    noise_scale: float = 0.01,
    max_samples: int = 20,
    seed: int = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compare stability metrics across multiple explainers.
    
    Args:
        explainers: Dict mapping explainer names to explainer instances
        model: Model adapter
        X: Input data
        n_perturbations: Number of perturbations per instance
        noise_scale: Noise scale
        max_samples: Max samples to evaluate per explainer
        seed: Random seed
        
    Returns:
        Dict mapping explainer names to their stability metrics
    """
    results = {}
    
    for name, explainer in explainers.items():
        metrics = compute_batch_stability(
            explainer, model, X, n_perturbations, noise_scale, max_samples, seed
        )
        results[name] = metrics
    
    return results
