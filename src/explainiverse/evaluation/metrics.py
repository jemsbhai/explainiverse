import numpy as np
from explainiverse.core.explanation import Explanation

def compute_aopc(
    model,
    instance: np.ndarray,
    explanation: Explanation,
    num_steps: int = 10,
    baseline_value: float = 0.0
) -> float:
    """
    Computes Area Over the Perturbation Curve (AOPC) by iteratively removing top features.

    Args:
        model: wrapped model with .predict() method
        instance: input sample (1D array)
        explanation: Explanation object
        num_steps: number of top features to remove
        baseline_value: value to replace removed features with (e.g., 0, mean)

    Returns:
        AOPC score (higher means explanation is more faithful)
    """
    base_pred = model.predict(instance.reshape(1, -1))[0]
    attributions = explanation.explanation_data.get("feature_attributions", {})

    if not attributions:
        raise ValueError("No feature attributions found in explanation.")

    # Sort features by abs importance
    sorted_features = sorted(
        attributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    # Try to map feature names to indices
    feature_indices = []
    for i, (fname, _) in enumerate(sorted_features):
        try:
            idx = explanation.feature_names.index(fname)
        except Exception:
            idx = i  # fallback: assume order
        feature_indices.append(idx)

    deltas = []
    modified = instance.copy()

    for i in range(min(num_steps, len(feature_indices))):
        idx = feature_indices[i]
        modified[idx] = baseline_value
        new_pred = model.predict(modified.reshape(1, -1))[0]
        delta = abs(base_pred - new_pred)
        deltas.append(delta)

    return np.mean(deltas)


def compute_batch_aopc(
    model,
    X: np.ndarray,
    explanations: dict,
    num_steps: int = 10,
    baseline_value: float = 0.0
) -> dict:
    """
    Compute average AOPC for multiple explainers over a batch of instances.

    Args:
        model: wrapped model
        X: 2D input array
        explanations: dict of {explainer_name: list of Explanation objects}
        num_steps: number of top features to remove
        baseline_value: value to replace features with

    Returns:
        Dict of {explainer_name: mean AOPC score}
    """
    results = {}

    for explainer_name, expl_list in explanations.items():
        scores = []
        for i, exp in enumerate(expl_list):
            instance = X[i]
            score = compute_aopc(model, instance, exp, num_steps, baseline_value)
            scores.append(score)
        results[explainer_name] = np.mean(scores)

    return results

