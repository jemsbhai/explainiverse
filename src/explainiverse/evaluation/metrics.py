import numpy as np
from explainiverse.core.explanation import Explanation
from sklearn.metrics import accuracy_score
import copy


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


def compute_roar(
    model_class,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    explanations: list,
    top_k: int = 3,
    baseline_value: float = 0.0,
    model_kwargs: dict = None
) -> float:
    """
    Compute ROAR (Remove And Retrain) using top-k important features from explanations.

    Args:
        model_class: uninstantiated model class (e.g. LogisticRegression)
        X_train: full training data
        y_train: training labels
        X_test: test features
        y_test: test labels
        explanations: list of Explanation objects (one per train instance)
        top_k: number of top features to remove
        baseline_value: what to set removed features to
        model_kwargs: optional kwargs to pass to model_class

    Returns:
        Accuracy drop (baseline_acc - retrained_acc)
    """
    model_kwargs = model_kwargs or {}

    # Baseline model
    baseline_model = model_class(**model_kwargs)
    baseline_model.fit(X_train, y_train)
    baseline_preds = baseline_model.predict(X_test)
    baseline_acc = accuracy_score(y_test, baseline_preds)

    # Compute top-k feature indices from attributions (use mode)
    feature_counts = {}
    for exp in explanations:
        for fname, val in sorted(exp.explanation_data["feature_attributions"].items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]:
            try:
                idx = exp.feature_names.index(fname)
                feature_counts[idx] = feature_counts.get(idx, 0) + 1
            except:
                continue

    top_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_feature_indices = [idx for idx, _ in top_features]

    # Remove top-k from training and test data
    X_train_mod = copy.deepcopy(X_train)
    X_test_mod = copy.deepcopy(X_test)
    
    # Prepare feature-wise baselines
    # Compute or assign feature-wise baseline values
    if not isinstance(
    baseline_value,
        (str, float, int, np.number, np.ndarray)
    ) and not callable(baseline_value):
        raise ValueError(f"Invalid baseline_value type: {type(baseline_value)}")
    
    if isinstance(baseline_value, str):
        if baseline_value == "mean":
            feature_baseline = np.mean(X_train, axis=0)
        elif baseline_value == "median":
            feature_baseline = np.median(X_train, axis=0)
        else:
            raise ValueError(f"Unsupported string baseline: {baseline_value}")
    elif callable(baseline_value):
        feature_baseline = baseline_value(X_train)
    elif isinstance(baseline_value, np.ndarray):
        if baseline_value.shape != (X_train.shape[1],):
            raise ValueError("baseline_value ndarray must match number of features")
        feature_baseline = baseline_value
    elif isinstance(baseline_value, (float, int, np.number)):
        feature_baseline = np.full(X_train.shape[1], baseline_value)
    else:
        raise ValueError(f"Invalid baseline_value type: {type(baseline_value)}")
    
    for idx in top_feature_indices:
        X_train_mod[:, idx] = feature_baseline[idx]
        X_test_mod[:, idx] = feature_baseline[idx]
        # X_train_mod[:, idx] = baseline_value
        # X_test_mod[:, idx] = baseline_value

    # Retrain and evaluate
    retrained_model = model_class(**model_kwargs)
    retrained_model.fit(X_train_mod, y_train)
    retrained_preds = retrained_model.predict(X_test_mod)
    retrained_acc = accuracy_score(y_test, retrained_preds)

    return baseline_acc - retrained_acc


def compute_roar_curve(
    model_class,
    X_train,
    y_train,
    X_test,
    y_test,
    explanations,
    max_k=5,
    baseline_value="mean",
    model_kwargs=None
) -> dict:
    """
    Compute ROAR accuracy drops across a range of top-k features removed.

    Args:
        model_class: model type (e.g. LogisticRegression)
        X_train, y_train, X_test, y_test: full dataset
        explanations: list of Explanation objects
        max_k: maximum top-k to try
        baseline_value: string, scalar, ndarray, or callable
        model_kwargs: passed to model class

    Returns:
        Dict of {k: accuracy drop} for k in 1..max_k
    """
    from copy import deepcopy

    model_kwargs = model_kwargs or {}
    curve = {}

    for k in range(1, max_k + 1):
        acc_drop = compute_roar(
            model_class=model_class,
            X_train=deepcopy(X_train),
            y_train=deepcopy(y_train),
            X_test=deepcopy(X_test),
            y_test=deepcopy(y_test),
            explanations=deepcopy(explanations),
            top_k=k,
            baseline_value=baseline_value,
            model_kwargs=deepcopy(model_kwargs)
        )
        curve[k] = acc_drop

    return curve