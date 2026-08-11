"""Verified perturbation and remove-and-retrain evaluation metrics.

The AOPC implementation follows Equation 12 of Samek et al. (2017):

    AOPC = (1 / (L + 1)) * sum_{k=0}^L [f(x) - f(x^(k))]

For classification, ``f`` is one fixed output identified by explicit or
mappable explanation metadata. An unknown multi-output target fails; model
argmax is never substituted for explanation identity or reselected after a
perturbation.

The ROAR implementation follows the core protocol in Hooker et al. (2019): an
importance ranking is required for every input in both the training and held-out
test sets; each row is masked by its own ranking; a fresh model is trained on the
masked training set and evaluated on the correspondingly masked test set.

References
----------
Samek et al., "Evaluating the Visualization of What a Deep Neural Network Has
Learned", IEEE TNNLS 28(11), 2017. DOI: 10.1109/TNNLS.2016.2599820.

Hooker et al., "A Benchmark for Interpretability Methods in Deep Neural
Networks", NeurIPS 2019.
"""

from __future__ import annotations

import re
from decimal import Decimal, localcontext
from numbers import Integral, Real
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, cast

import numpy as np
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.metrics import accuracy_score

from explainiverse.core.explanation import Explanation
from explainiverse.evaluation._utils import (
    _stable_difference_of_means,
    _stable_mean,
    _stable_mean_difference,
    _stable_std,
    _stable_sum,
    compute_baseline_values,
)

Baseline = Union[str, float, np.ndarray, Callable[[np.ndarray], np.ndarray]]
Target = Optional[Union[int, float, str]]


def _finite_mean(values, context: str) -> float:
    result = float(_stable_mean(np.asarray(values, dtype=np.float64)))
    if not np.isfinite(result):
        raise FloatingPointError(f"{context} mean is not representable")
    return result


def _finite_std(values, context: str) -> float:
    result = float(_stable_std(np.asarray(values, dtype=np.float64)))
    if not np.isfinite(result):
        raise FloatingPointError(f"{context} standard deviation is not representable")
    return result


def _finite_difference(left: float, right: float, context: str) -> float:
    result = float(_stable_sum(np.asarray([left, -right], dtype=np.float64)))
    if not np.isfinite(result):
        raise FloatingPointError(f"{context} is not representable")
    return result


def _extract_feature_index(
    feature_name: str,
    feature_names: Optional[List[str]] = None,
    fallback_index: Optional[int] = 0,
) -> Optional[int]:
    """Resolve an attribution key to a feature index.

    ``fallback_index`` remains for backwards compatibility with callers of this
    helper.  Metric implementations pass ``None`` and therefore fail rather
    than silently assigning an unknown attribution to an unrelated feature.
    """
    if not isinstance(feature_name, str):
        return fallback_index

    key = feature_name.strip()
    if feature_names is not None:
        if key in feature_names:
            return feature_names.index(key)

        # LIME conditions commonly look like ``age <= 30``.  Only strip a
        # trailing numeric comparison; substring matching is intentionally not
        # used because names such as ``age`` and ``age_squared`` are ambiguous.
        base_name = re.sub(
            r"\s*(?:<=|>=|==|=|!=|<|>)\s*" r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s*$",
            "",
            key,
        ).strip()
        if base_name in feature_names:
            return feature_names.index(base_name)

    patterns = (
        r"^feature[_\s]*(\d+)$",
        r"^feat[_\s]*(\d+)$",
        r"^f(\d+)$",
        r"^x(\d+)$",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, key, re.IGNORECASE)
        if match:
            return int(match.group(1))

    return fallback_index


def _validate_task(task: Optional[str]) -> Optional[str]:
    if task not in {None, "classification", "regression"}:
        raise ValueError("task must be 'classification', 'regression', or None")
    return task


def _wrapped_model(model: Any) -> Any:
    return getattr(model, "model", model)


def _infer_model_task(model: Any, task: Optional[str] = None) -> Optional[str]:
    task = _validate_task(task)
    if task is not None:
        return task

    for candidate in (model, _wrapped_model(model)):
        candidate_task = getattr(candidate, "task", None)
        if candidate_task in {"classification", "regression"}:
            return candidate_task
        estimator_type = getattr(candidate, "_estimator_type", None)
        if estimator_type == "classifier":
            return "classification"
        if estimator_type == "regressor":
            return "regression"
        if hasattr(candidate, "classes_") or hasattr(candidate, "predict_proba"):
            return "classification"

    return None


def _model_class_label_sequences(model: Any) -> list[list[Any]]:
    """Return label sequences while preserving each sequence's output indices."""
    sequences: list[list[Any]] = []
    for candidate in (model, _wrapped_model(model)):
        for attribute in ("class_names", "classes_"):
            values = getattr(candidate, attribute, None)
            if values is not None:
                labels = np.asarray(values, dtype=object).reshape(-1).tolist()
                if labels and labels not in sequences:
                    sequences.append(labels)
    return sequences


def _raw_predictions(model: Any, X: np.ndarray, task: str) -> tuple[np.ndarray, str]:
    """Return a numerical ``(samples, outputs)`` matrix and its output space."""
    raw_model = _wrapped_model(model)
    if task == "classification" and hasattr(model, "predict_proba"):
        predictions = model.predict_proba(X)
    elif task == "classification" and model is raw_model and hasattr(raw_model, "predict_proba"):
        predictions = raw_model.predict_proba(X)
    else:
        if not hasattr(model, "predict"):
            raise TypeError("model must expose predict or predict_proba")
        predictions = model.predict(X)

    predictions = np.asarray(predictions)
    if predictions.ndim == 0:
        predictions = predictions.reshape(1, 1)
    elif predictions.ndim == 1:
        if predictions.shape[0] != X.shape[0]:
            # A single sample may be returned as an output vector.
            if X.shape[0] == 1:
                predictions = predictions.reshape(1, -1)
            else:
                raise ValueError("model returned the wrong number of predictions")
        else:
            predictions = predictions.reshape(-1, 1)
    elif predictions.ndim != 2:
        raise ValueError("model predictions must be one- or two-dimensional")

    if predictions.shape[0] != X.shape[0]:
        raise ValueError("model returned the wrong number of predictions")

    if task == "regression":
        try:
            values = predictions.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("regression predictions must be numerical") from exc
        if not np.all(np.isfinite(values)):
            raise ValueError("model returned non-finite regression predictions")
        return values, "regression_output"

    # A predict-only classifier can expose labels rather than scores.  Convert
    # them to fixed one-hot indicators when classes_ is available, and disclose
    # that coarse output space in return_details.
    has_probability_api = hasattr(model, "predict_proba") or (
        model is raw_model and hasattr(raw_model, "predict_proba")
    )
    classes = getattr(raw_model, "classes_", getattr(model, "classes_", None))
    if predictions.shape[1] == 1 and classes is not None and not has_probability_api:
        labels = predictions[:, 0]
        classes = np.asarray(classes)
        matches = labels[:, None] == classes[None, :]
        if not np.all(matches.sum(axis=1) == 1):
            raise ValueError("classifier predicted a label not present in classes_")
        return matches.astype(float), "hard_label_indicator"

    try:
        values = predictions.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("classification outputs must be numerical scores") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError("model returned non-finite classification outputs")

    if values.shape[1] == 1:
        if not np.all((0.0 <= values) & (values <= 1.0)):
            raise ValueError("one-column classification output must contain P(class 1) in [0, 1]")
        positive = values[:, 0]
        values = np.column_stack([1.0 - positive, positive])

    is_probability = np.all((0.0 <= values) & (values <= 1.0)) and np.allclose(
        values.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8
    )
    return values, "probability" if is_probability else "class_score"


def _predict_outputs(model: Any, X: np.ndarray, task: Optional[str]) -> tuple[np.ndarray, str, str]:
    resolved_task = _infer_model_task(model, task)
    if resolved_task is not None:
        outputs, output_space = _raw_predictions(model, X, resolved_task)
        return outputs, output_space, resolved_task

    # For an untagged object, a multi-column output is a classification score
    # matrix.  A scalar/one-column output is treated as regression; callers can
    # override this with task='classification' for a sigmoid model.
    if not hasattr(model, "predict"):
        raise TypeError("model must expose a predict method")
    probe = np.asarray(model.predict(X))
    inferred = "classification" if probe.ndim == 2 and probe.shape[1] > 1 else "regression"
    outputs, output_space = _raw_predictions(model, X, inferred)
    return outputs, output_space, inferred


def _resolve_target_index(
    model: Any,
    explanation: Explanation,
    original_outputs: np.ndarray,
    task: str,
    explicit_target: Target,
) -> tuple[int, str]:
    n_outputs = original_outputs.shape[1]

    index_metadata_sources = {
        "metadata.class_index",
        "metadata.target_class_index",
        "metadata.output_index",
    }

    def resolve_candidate(candidate: Target, source: str) -> Optional[int]:
        if isinstance(candidate, bool):
            raise TypeError(f"{source} must be an output index or class label, not bool")

        # These metadata fields explicitly promise an index.  All other target
        # sources are labels first, including integer labels such as classes_
        # == [1, 2].  Falling back to an integer index keeps index-only model
        # APIs usable when the model exposes no label vocabulary.
        if source in index_metadata_sources:
            if not isinstance(candidate, Integral):
                raise TypeError(f"{source} must be an integer output index")
            return int(candidate)

        matching_indices: set[int] = set()
        for labels in _model_class_label_sequences(model):
            for index, label in enumerate(labels):
                try:
                    exact_match = bool(candidate == label)
                except (TypeError, ValueError):
                    exact_match = False
                string_match = isinstance(candidate, str) and candidate == str(label)
                if exact_match or string_match:
                    matching_indices.add(index)
        if len(matching_indices) > 1:
            raise ValueError(
                f"{source}={candidate!r} maps to conflicting model output indices "
                f"{sorted(matching_indices)}"
            )
        if matching_indices:
            return next(iter(matching_indices))

        if isinstance(candidate, Integral):
            return int(candidate)
        if not isinstance(candidate, (str, Real)):
            raise TypeError(f"{source} must be an output index or scalar class label")
        if not isinstance(candidate, str):
            return None
        parsed = re.fullmatch(r"(?:class|output)[_\s]*(\d+)", candidate, re.IGNORECASE)
        if parsed:
            return int(parsed.group(1))
        if task == "regression" and n_outputs == 1:
            return 0
        return None

    candidates: list[tuple[str, Target]] = []
    if explicit_target is not None:
        candidates.append(("argument", explicit_target))
    for key in ("class_index", "target_class_index", "output_index"):
        if key in explanation.metadata:
            candidates.append((f"metadata.{key}", explanation.metadata[key]))

    label = getattr(explanation, "target_class", None)
    informative_label = label is not None and not (
        isinstance(label, str) and label in {"", "output", "regression"}
    )
    resolved: list[tuple[str, int]] = []
    for source, candidate in candidates:
        index = resolve_candidate(candidate, source)
        if index is None:
            raise ValueError(f"Cannot map {source}={candidate!r} to a model output")
        resolved.append((source, index))

    if informative_label:
        label_index = resolve_candidate(label, "explanation.target_class")
        if label_index is not None:
            resolved.append(("explanation.target_class", label_index))
        elif not resolved:
            raise ValueError(f"Cannot map target_class={label!r} to a model output")

    if resolved:
        unique_indices = {index for _, index in resolved}
        if len(unique_indices) != 1:
            details = ", ".join(f"{source}={index}" for source, index in resolved)
            raise ValueError(f"conflicting target output metadata: {details}")
        index = resolved[0][1]
        source = "+".join(source for source, _ in resolved)
    else:
        if n_outputs != 1:
            descriptor = "multi-output regression" if task == "regression" else "multi-output"
            raise ValueError(
                f"{descriptor} evaluation requires an explicit or mappable explanation target"
            )
        index = 0
        source = "single_output"

    if index < 0 or index >= n_outputs:
        raise ValueError(f"target output index {index} is invalid for {n_outputs} model output(s)")
    return index, source


def _baseline_values(
    baseline_value: Baseline,
    n_features: int,
    background_data: Optional[np.ndarray],
) -> np.ndarray:
    try:
        return compute_baseline_values(baseline_value, background_data, n_features)
    except ValueError as exc:
        if "baseline must resolve to shape" in str(exc):
            raise ValueError(
                f"baseline_value must provide exactly one value per feature; {exc}"
            ) from exc
        raise


def _validate_ranking(ranking: str) -> str:
    if ranking not in {"descending", "absolute"}:
        raise ValueError("ranking must be 'descending' or 'absolute'")
    return ranking


def _sorted_feature_indices(
    explanation: Explanation,
    n_features: int,
    ranking: str = "descending",
) -> list[int]:
    if not isinstance(explanation, Explanation):
        raise TypeError("each explanation must be an Explanation instance")
    ranking = _validate_ranking(ranking)
    attributions = explanation.explanation_data.get("feature_attributions", {})
    if not isinstance(attributions, dict) or not attributions:
        raise ValueError("No feature attributions found in explanation.")

    feature_names = getattr(explanation, "feature_names", None)
    if feature_names is not None:
        feature_names = list(feature_names)
        if len(feature_names) != n_features:
            raise ValueError("explanation.feature_names must contain one name per feature")
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("explanation.feature_names must be unique")

    validated: list[tuple[str, float]] = []
    for name, value in attributions.items():
        if not isinstance(name, str):
            raise TypeError("feature attribution keys must be strings")
        if not isinstance(value, Real) or isinstance(value, bool):
            raise ValueError(f"attribution for {name!r} must be a finite number")
        numeric_value = float(value)
        if not np.isfinite(numeric_value):
            raise ValueError(f"attribution for {name!r} must be a finite number")
        validated.append((name, numeric_value))

    if ranking == "absolute":
        validated.sort(key=lambda item: abs(item[1]), reverse=True)
    else:
        validated.sort(key=lambda item: item[1], reverse=True)
    indices: list[int] = []
    for name, _ in validated:
        index = _extract_feature_index(name, feature_names, fallback_index=None)
        if index is None:
            raise ValueError(f"Cannot map attribution key {name!r} to a feature index")
        if index < 0 or index >= n_features:
            raise ValueError(f"Attribution key {name!r} maps to out-of-range feature index {index}")
        if index in indices:
            raise ValueError(f"Multiple attribution keys map to the same feature index {index}")
        indices.append(index)
    return indices


def _validate_positive_steps(value: int, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def compute_aopc(
    model: Any,
    instance: np.ndarray,
    explanation: Explanation,
    num_steps: int = 10,
    baseline_value: Baseline = 0.0,
    *,
    target_class: Target = None,
    background_data: Optional[np.ndarray] = None,
    task: Optional[str] = None,
    ranking: str = "descending",
    return_details: bool = False,
) -> Union[float, Dict[str, Any]]:
    """Compute a generalized feature-wise MoRF AOPC contribution.

    The aggregation follows Samek et al.'s Equation 12, while the intervention
    is deterministic per-feature baseline replacement rather than the paper's
    image-region perturbation generator. Features are cumulatively replaced
    most-relevant-first. By default the
    estimator's descending relevance order is preserved; use
    ``ranking='absolute'`` only when magnitude is the intended importance
    definition.  The returned value is a *signed* output drop, so a
    perturbation that raises the fixed explained output can produce a negative
    AOPC.

    For the paper's dataset-level quantity, use :func:`compute_batch_aopc`,
    which averages these per-input contributions.
    """
    if not isinstance(return_details, (bool, np.bool_)):
        raise TypeError("return_details must be a boolean")
    return_details = bool(return_details)
    instance = np.asarray(instance)
    if instance.ndim != 1 or instance.size == 0:
        raise ValueError("instance must be a non-empty one-dimensional array")
    try:
        instance = instance.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("instance must be numerical") from exc
    if not np.all(np.isfinite(instance)):
        raise ValueError("instance must contain only finite values")

    num_steps = _validate_positive_steps(num_steps, "num_steps")
    ranking = _validate_ranking(ranking)
    n_features = instance.size
    feature_order = _sorted_feature_indices(explanation, n_features, ranking)
    effective_steps = min(num_steps, n_features)
    if len(feature_order) < effective_steps:
        raise ValueError(
            f"explanation provides {len(feature_order)} unique features, but "
            f"AOPC requires {effective_steps} for this input and num_steps"
        )

    baseline = _baseline_values(baseline_value, n_features, background_data)
    original_matrix, output_space, resolved_task = _predict_outputs(
        model, instance.reshape(1, -1), task
    )
    target_index, target_source = _resolve_target_index(
        model, explanation, original_matrix, resolved_task, target_class
    )
    original_value = float(original_matrix[0, target_index])

    prediction_values = [original_value]
    modified = instance.copy()
    for index in feature_order[:effective_steps]:
        modified[index] = baseline[index]
        matrix, perturbed_space, perturbed_task = _predict_outputs(
            model, modified.reshape(1, -1), resolved_task
        )
        if perturbed_task != resolved_task or perturbed_space != output_space:
            raise ValueError("model output contract changed after perturbation")
        if target_index >= matrix.shape[1]:
            raise ValueError("model output count changed after perturbation")
        value = float(matrix[0, target_index])
        prediction_values.append(value)

    # Equation 12 includes k=0.  Its drop is zero but it contributes to the
    # L+1 denominator. Aggregate predictions before subtracting so an
    # out-of-range individual drop cannot hide a representable signed mean.
    aopc = _stable_mean_difference(original_value, np.asarray(prediction_values))
    if not return_details:
        return aopc

    try:
        prediction_drops = [
            _finite_difference(original_value, value, "AOPC drop") for value in prediction_values
        ]
    except FloatingPointError as exc:
        raise FloatingPointError(
            "return_details cannot represent an individual AOPC prediction_drop; "
            "use return_details=False for the representable aggregate"
        ) from exc

    return {
        "aopc": aopc,
        "formula": "sum_k=0^L(f(x)-f(x_k))/(L+1)",
        "aggregation": "signed_mean_including_k_zero",
        "ranking": ranking,
        "ranking_transformation_applied": ranking == "absolute",
        "task": resolved_task,
        "output_space": output_space,
        "target_index": target_index,
        "target_source": target_source,
        "original_value": original_value,
        "prediction_values": prediction_values,
        "prediction_drops": prediction_drops,
        "feature_order": feature_order[:effective_steps],
        "requested_steps": num_steps,
        "effective_steps": effective_steps,
        "baseline_values": baseline.tolist(),
        "perturbation_protocol": "deterministic_per_feature_baseline_replacement",
        "paper_random_perturbation_protocol_used": False,
        "canonical_samek_output_contract": (
            resolved_task == "classification" and output_space in {"probability", "class_score"}
        ),
        "samek_formula_contract_met": True,
        "samek_region_perturbation_contract_met": False,
        "random_order_control_included": False,
        "claim_scope": "generalized_feature_morf_aopc",
    }


def compute_batch_aopc(
    model: Any,
    X: np.ndarray,
    explanations: Dict[str, List[Explanation]],
    num_steps: int = 10,
    baseline_value: Baseline = 0.0,
    *,
    background_data: Optional[np.ndarray] = None,
    task: Optional[str] = None,
    ranking: str = "descending",
) -> Dict[str, float]:
    """Average verified per-input AOPC contributions for each explainer.

    Every method must provide exactly one valid explanation per row.  Errors are
    surfaced rather than silently skipped, because skipping difficult rows makes
    scores between explainers incomparable.
    """
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a non-empty two-dimensional array")
    if not isinstance(explanations, dict) or not explanations:
        raise ValueError("explanations must map at least one explainer to a list")

    results: Dict[str, float] = {}
    for explainer_name, explainer_explanations in explanations.items():
        if len(explainer_explanations) != len(X):
            raise ValueError(
                f"{explainer_name!r} must provide the same number of explanations as X rows"
            )
        scores = [
            cast(
                float,
                compute_aopc(
                    model,
                    X[row],
                    explainer_explanations[row],
                    num_steps=num_steps,
                    baseline_value=baseline_value,
                    background_data=background_data,
                    task=task,
                    ranking=ranking,
                ),
            )
            for row in range(len(X))
        ]
        results[explainer_name] = _finite_mean(scores, f"AOPC for {explainer_name!r}")
    return results


def _validate_roar_arrays(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("X_train and X_test must be two-dimensional")
    if X_train.shape[0] == 0 or X_test.shape[0] == 0 or X_train.shape[1] == 0:
        raise ValueError("training and test data must be non-empty")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("X_train and X_test must have the same number of features")
    if y_train.ndim == 0 or y_test.ndim == 0:
        raise ValueError("y_train and y_test must retain a sample dimension")
    if len(y_train) != len(X_train) or len(y_test) != len(X_test):
        raise ValueError("X and y sample counts must match within each split")
    if X_train.shape == X_test.shape and np.array_equal(X_train, X_test):
        raise ValueError("ROAR requires a held-out test split; X_test is identical to X_train")
    try:
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("ROAR currently supports numerical feature matrices") from exc
    if not np.all(np.isfinite(X_train)) or not np.all(np.isfinite(X_test)):
        raise ValueError("ROAR feature matrices must contain only finite values")
    return X_train, y_train, X_test, y_test


def _resolve_top_k(top_k: Union[int, float], n_features: int) -> int:
    if isinstance(top_k, Integral) and not isinstance(top_k, bool):
        count = int(top_k)
        if count <= 0 or count > n_features:
            raise ValueError(f"top_k must be between 1 and {n_features}")
        return count
    if isinstance(top_k, Real) and not isinstance(top_k, bool):
        fraction = float(top_k)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fractional top_k must be in (0, 1]")
        return min(n_features, int(np.ceil(fraction * n_features)))
    raise TypeError("top_k must be a positive feature count or fraction in (0, 1]")


def _validate_explanation_alignment(
    explanations: Sequence[Explanation],
    n_rows: int,
    split_name: str,
) -> None:
    if len(explanations) != n_rows:
        raise ValueError(
            f"explanations must align with all {split_name} rows: expected {n_rows}, "
            f"received {len(explanations)}"
        )


def _mask_dataset(
    X: np.ndarray,
    explanations: Sequence[Explanation],
    top_k: int,
    baseline: np.ndarray,
    ranking: str,
) -> tuple[np.ndarray, np.ndarray]:
    dtype = np.result_type(X.dtype, baseline.dtype)
    masked = np.asarray(X, dtype=dtype).copy()
    masks = np.zeros(X.shape, dtype=bool)
    for row, explanation in enumerate(explanations):
        order = _sorted_feature_indices(explanation, X.shape[1], ranking)
        if len(order) < top_k:
            raise ValueError(
                f"explanation at row {row} provides {len(order)} unique features, "
                f"but top_k={top_k}"
            )
        indices = np.asarray(order[:top_k], dtype=int)
        masked[row, indices] = baseline[indices]
        masks[row, indices] = True
    return masked, masks


def _new_estimator(model_spec: Any, model_kwargs: Dict[str, Any]) -> Any:
    if isinstance(model_spec, type):
        return model_spec(**model_kwargs)
    if hasattr(model_spec, "fit"):
        if model_kwargs:
            estimator = clone(model_spec)
            estimator.set_params(**model_kwargs)
            return estimator
        return clone(model_spec)
    if callable(model_spec):
        return model_spec(**model_kwargs)
    raise TypeError("model_class must be an estimator class, factory, or sklearn estimator")


def _estimator_task(estimator: Any, explicit_task: Optional[str]) -> str:
    explicit_task = _validate_task(explicit_task)
    if explicit_task is not None:
        return explicit_task
    if is_classifier(estimator) or getattr(estimator, "_estimator_type", None) == "classifier":
        return "classification"
    if is_regressor(estimator) or getattr(estimator, "_estimator_type", None) == "regressor":
        return "regression"
    raise ValueError("Cannot infer estimator task; pass task='classification' or task='regression'")


def _set_random_state_if_supported(estimator: Any, seed: Optional[int]) -> Any:
    if seed is None or not hasattr(estimator, "get_params"):
        return estimator
    params = estimator.get_params(deep=True)
    random_state_names = sorted(
        name for name in params if name == "random_state" or name.endswith("__random_state")
    )
    if random_state_names:
        estimator.set_params(**{name: seed for name in random_state_names})
    return estimator


def _random_state_parameter_names(estimator: Any) -> list[str]:
    if not hasattr(estimator, "get_params"):
        return []
    params = estimator.get_params(deep=True)
    return sorted(
        name for name in params if name == "random_state" or name.endswith("__random_state")
    )


def _score_predictions(
    y_true: np.ndarray,
    predictions: np.ndarray,
    task: str,
    scoring: Optional[Union[str, Callable[[np.ndarray, np.ndarray], float]]],
    scoring_greater_is_better: Optional[bool],
) -> tuple[float, str, bool]:
    if callable(scoring):
        if not isinstance(scoring_greater_is_better, bool):
            raise ValueError(
                "scoring_greater_is_better must be explicitly True or False for callable scoring"
            )
        value = scoring(y_true, predictions)
        name = getattr(scoring, "__name__", "callable")
        greater_is_better = scoring_greater_is_better
    else:
        if scoring_greater_is_better not in {None, True}:
            raise ValueError("built-in accuracy and r2 scorers are greater-is-better")
        name = scoring or ("accuracy" if task == "classification" else "r2")
        if name == "accuracy":
            if task != "classification":
                raise ValueError("accuracy scoring is only valid for classification")
            value = accuracy_score(y_true, predictions)
        elif name == "r2":
            if task != "regression":
                raise ValueError("r2 scoring is only valid for regression")
            value = _stable_r2_score(y_true, predictions)
        else:
            raise ValueError("scoring must be None, 'accuracy', 'r2', or a callable")
        greater_is_better = True
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"scoring function {name!r} returned a non-finite value")
    return value, str(name), greater_is_better


def _stable_r2_score(y_true: np.ndarray, predictions: np.ndarray) -> float:
    """Match sklearn's uniform-average, force-finite R2 without square overflow."""
    actual = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    if actual.shape != predicted.shape or actual.ndim not in (1, 2) or actual.shape[0] < 2:
        raise ValueError("r2 inputs must have equal 1-D/2-D shape with at least two rows")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(predicted)):
        raise ValueError("r2 inputs must contain only finite values")
    if actual.ndim == 1:
        actual = actual[:, None]
        predicted = predicted[:, None]

    with localcontext() as context:
        context.prec = 3000 + len(str(actual.shape[0]))
        scores = []
        for output_index in range(actual.shape[1]):
            actual_values = [Decimal.from_float(float(value)) for value in actual[:, output_index]]
            predicted_values = [
                Decimal.from_float(float(value)) for value in predicted[:, output_index]
            ]
            mean = sum(actual_values, start=Decimal(0)) / Decimal(len(actual_values))
            residual_sum = sum(
                (
                    (actual_value - predicted_value) * (actual_value - predicted_value)
                    for actual_value, predicted_value in zip(actual_values, predicted_values)
                ),
                start=Decimal(0),
            )
            total_sum = sum(
                ((value - mean) * (value - mean) for value in actual_values),
                start=Decimal(0),
            )
            if total_sum == 0:
                scores.append(Decimal(1) if residual_sum == 0 else Decimal(0))
            else:
                scores.append(Decimal(1) - residual_sum / total_sum)
        exact = sum(scores, start=Decimal(0)) / Decimal(len(scores))
        result = float(exact)
    if not np.isfinite(result) or (result == 0.0 and exact != 0):
        raise FloatingPointError("r2 score is not representable")
    return result


def compute_roar(
    model_class: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    explanations: List[Explanation],
    top_k: Union[int, float] = 0.1,
    baseline_value: Baseline = "mean",
    model_kwargs: Optional[Dict[str, Any]] = None,
    *,
    test_explanations: Optional[List[Explanation]] = None,
    n_repeats: int = 5,
    random_state: Optional[int] = 0,
    task: Optional[str] = None,
    scoring: Optional[Union[str, Callable[[np.ndarray, np.ndarray], float]]] = None,
    scoring_greater_is_better: Optional[bool] = None,
    ranking: str = "descending",
    return_details: bool = False,
) -> Union[float, Dict[str, Any]]:
    """Compute one threshold of the per-sample ROAR retraining protocol.

    ``test_explanations`` is required because Hooker et al. compute an
    importance estimate for every input in both splits.  The former global
    vote over a partial training explanation set has been removed: it was a
    global feature-ablation diagnostic, not ROAR.

    The scalar return value is a positive-oriented paired performance drop.
    For a lower-is-better callable loss, set
    ``scoring_greater_is_better=False``; the subtraction is then reversed.
    For the paper's full benchmark, use several removal fractions, at least
    five independent retraining runs, and compare against a random-ranking
    control.  ``return_details=True`` exposes whether those contracts were met.
    """
    if not isinstance(return_details, (bool, np.bool_)):
        raise TypeError("return_details must be a boolean")
    return_details = bool(return_details)
    X_train, y_train, X_test, y_test = _validate_roar_arrays(X_train, y_train, X_test, y_test)
    if test_explanations is None:
        raise ValueError(
            "test_explanations is required for ROAR because test rows need their own rankings"
        )
    _validate_explanation_alignment(explanations, len(X_train), "training")
    _validate_explanation_alignment(test_explanations, len(X_test), "test")

    n_features = X_train.shape[1]
    top_k_count = _resolve_top_k(top_k, n_features)
    ranking = _validate_ranking(ranking)
    n_repeats = _validate_positive_steps(n_repeats, "n_repeats")
    if random_state is not None:
        if (
            not isinstance(random_state, Integral)
            or isinstance(random_state, bool)
            or int(random_state) < 0
        ):
            raise ValueError("random_state must be a non-negative integer or None")
        random_state = int(random_state)

    # Statistical/callable replacement values are derived from training data
    # only.  This avoids test-statistic leakage into the modified datasets.
    baseline = _baseline_values(baseline_value, n_features, X_train)
    X_train_masked, train_mask = _mask_dataset(
        X_train, explanations, top_k_count, baseline, ranking
    )
    X_test_masked, test_mask = _mask_dataset(
        X_test, test_explanations, top_k_count, baseline, ranking
    )

    model_kwargs = dict(model_kwargs or {})
    probe_estimator = _new_estimator(model_class, model_kwargs)
    resolved_task = _estimator_task(probe_estimator, task)
    random_state_parameters = _random_state_parameter_names(probe_estimator)
    nested_random_state_parameters = [name for name in random_state_parameters if "__" in name]
    baseline_scores: list[float] = []
    retrained_scores: list[float] = []
    scoring_name: Optional[str] = None
    scoring_direction: Optional[bool] = None

    for repeat in range(n_repeats):
        seed = None if random_state is None else random_state + repeat
        clean_model = _set_random_state_if_supported(
            _new_estimator(model_class, model_kwargs), seed
        )
        masked_model = _set_random_state_if_supported(
            _new_estimator(model_class, model_kwargs), seed
        )

        try:
            clean_model.fit(X_train, y_train)
            clean_predictions = clean_model.predict(X_test)
        except Exception as exc:
            raise RuntimeError(f"clean ROAR fit/evaluation failed on repeat {repeat}") from exc
        try:
            masked_model.fit(X_train_masked, y_train)
            masked_predictions = masked_model.predict(X_test_masked)
        except Exception as exc:
            raise RuntimeError(f"masked ROAR fit/evaluation failed on repeat {repeat}") from exc

        clean_score, current_name, current_direction = _score_predictions(
            y_test, clean_predictions, resolved_task, scoring, scoring_greater_is_better
        )
        masked_score, masked_name, masked_direction = _score_predictions(
            y_test, masked_predictions, resolved_task, scoring, scoring_greater_is_better
        )
        if current_name != masked_name or current_direction != masked_direction:
            raise RuntimeError("clean and masked models used different scoring contracts")
        scoring_name = current_name
        scoring_direction = current_direction
        baseline_scores.append(clean_score)
        retrained_scores.append(masked_score)

    baseline_mean = _finite_mean(baseline_scores, "ROAR baseline score")
    retrained_mean = _finite_mean(retrained_scores, "ROAR retrained score")
    if scoring_direction is None:  # Defensive: n_repeats is validated positive.
        raise RuntimeError("ROAR scoring direction was not resolved")
    score_drop = (
        _stable_difference_of_means(np.asarray(baseline_scores), np.asarray(retrained_scores))
        if scoring_direction
        else _stable_difference_of_means(np.asarray(retrained_scores), np.asarray(baseline_scores))
    )
    if not return_details:
        return float(score_drop)

    return {
        "score_drop": float(score_drop),
        "baseline_score": baseline_mean,
        "retrained_score": retrained_mean,
        "baseline_scores": baseline_scores,
        "retrained_scores": retrained_scores,
        "baseline_score_std": _finite_std(baseline_scores, "ROAR baseline score"),
        "retrained_score_std": _finite_std(retrained_scores, "ROAR retrained score"),
        "scoring": scoring_name,
        "scoring_greater_is_better": scoring_direction,
        "score_drop_semantics": "positive_means_masking_hurt_performance",
        "task": resolved_task,
        "protocol": "per_sample_remove_and_retrain",
        "ranking": ranking,
        "ranking_transformation_applied": ranking == "absolute",
        "top_k": top_k_count,
        "fraction_removed": top_k_count / n_features,
        "baseline_values": baseline.tolist(),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "n_repeats": n_repeats,
        "random_state": random_state,
        "per_row_train_masks": bool(np.all(train_mask.sum(axis=1) == top_k_count)),
        "per_row_test_masks": bool(np.all(test_mask.sum(axis=1) == top_k_count)),
        "canonical_core_contract": resolved_task == "classification" and scoring_name == "accuracy",
        "repeat_seeds": (
            None if random_state is None else [random_state + repeat for repeat in range(n_repeats)]
        ),
        "random_state_parameters": random_state_parameters,
        "random_state_parameters_controlled": (
            random_state is not None and bool(random_state_parameters)
        ),
        "nested_random_state_parameters": nested_random_state_parameters,
        "nested_random_state_parameters_controlled": (
            random_state is not None and bool(nested_random_state_parameters)
        ),
        "paired_clean_masked_initialisation": (
            random_state is not None and bool(random_state_parameters)
        ),
        "independent_repeat_initialisations_controlled": (
            random_state is not None and bool(random_state_parameters) and n_repeats > 1
        ),
        "paper_repetition_count_met": n_repeats >= 5,
        "paper_repetition_contract_met": (
            n_repeats >= 5 and random_state is not None and bool(random_state_parameters)
        ),
        "random_ranking_control_included": False,
        "requires_control_comparison_for_method_quality_claim": True,
    }


def compute_roar_curve(
    model_class: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    explanations: List[Explanation],
    max_k: int = 5,
    baseline_value: Baseline = "mean",
    model_kwargs: Optional[Dict[str, Any]] = None,
    *,
    test_explanations: Optional[List[Explanation]] = None,
    n_repeats: int = 5,
    random_state: Optional[int] = 0,
    task: Optional[str] = None,
    scoring: Optional[Union[str, Callable[[np.ndarray, np.ndarray], float]]] = None,
    scoring_greater_is_better: Optional[bool] = None,
    ranking: str = "descending",
) -> Dict[int, float]:
    """Compute verified ROAR score drops for feature counts 1 through ``max_k``."""
    X_train_array = np.asarray(X_train)
    if X_train_array.ndim != 2:
        raise ValueError("X_train must be two-dimensional")
    max_k = _validate_positive_steps(max_k, "max_k")
    if max_k > X_train_array.shape[1]:
        raise ValueError("max_k cannot exceed the feature count")

    return {
        k: float(
            cast(
                float,
                compute_roar(
                    model_class=model_class,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    explanations=explanations,
                    top_k=k,
                    baseline_value=baseline_value,
                    model_kwargs=model_kwargs,
                    test_explanations=test_explanations,
                    n_repeats=n_repeats,
                    random_state=random_state,
                    task=task,
                    scoring=scoring,
                    scoring_greater_is_better=scoring_greater_is_better,
                    ranking=ranking,
                ),
            )
        )
        for k in range(1, max_k + 1)
    }
