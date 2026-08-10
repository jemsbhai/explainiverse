"""Shared, fail-fast helpers for tabular evaluation diagnostics."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from numbers import Integral, Real
from typing import Any, Optional, Union

import numpy as np

from explainiverse.core.explanation import Explanation

_INDEX_PATTERNS = (
    re.compile(r"feature[_\s]*(\d+)", re.IGNORECASE),
    re.compile(r"feat[_\s]*(\d+)", re.IGNORECASE),
    re.compile(r"f(\d+)", re.IGNORECASE),
    re.compile(r"x(\d+)", re.IGNORECASE),
)
_TRAILING_COMPARISON = re.compile(
    r"^(.+?)\s*(?:<=|>=|<|>|=)\s*" r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$"
)
_LEADING_COMPARISON = re.compile(
    r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s*" r"(?:<=|>=|<|>|=)\s*(.+)$"
)


def _extract_base_feature_name(feature_str: str) -> str:
    """Remove numeric bounds from a legacy LIME feature label.

    LIME may emit either a one-sided label such as ``"age <= 30"`` or a
    discretized interval such as ``"20 < age <= 30"``. Only complete numeric
    comparisons are removed; arbitrary substrings are never treated as names.
    """
    if not isinstance(feature_str, str) or not feature_str.strip():
        raise TypeError("feature_str must be a non-empty string")
    stripped = feature_str.strip()
    trailing_match = _TRAILING_COMPARISON.fullmatch(stripped)
    without_trailing = trailing_match.group(1).strip() if trailing_match else stripped
    leading_match = _LEADING_COMPARISON.fullmatch(without_trailing)
    return leading_match.group(1).strip() if leading_match else without_trailing


def _explicit_feature_index(feature_key: str) -> Optional[int]:
    """Return an index only for a complete, recognized index-style key."""
    for pattern in _INDEX_PATTERNS:
        match = pattern.fullmatch(feature_key.strip())
        if match:
            return int(match.group(1))
    return None


def _match_feature_to_index(feature_key: str, feature_names: Sequence[str]) -> int:
    """Resolve a feature key without substring or positional fallbacks."""
    if not isinstance(feature_key, str) or not feature_key.strip():
        raise TypeError("feature_key must be a non-empty string")
    if isinstance(feature_names, (str, bytes)) or not isinstance(feature_names, Sequence):
        raise TypeError("feature_names must be a sequence of strings")
    names = list(feature_names)
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("feature_names must contain non-empty strings")
    if len(names) != len(set(names)):
        raise ValueError("feature_names must be unique")

    if feature_key in names:
        return names.index(feature_key)

    base_name = _extract_base_feature_name(feature_key)
    if base_name in names:
        return names.index(base_name)

    explicit_index = _explicit_feature_index(base_name)
    if explicit_index is not None and explicit_index < len(names):
        return explicit_index
    return -1


def get_sorted_feature_indices(
    explanation: Explanation,
    descending: bool = True,
) -> list[int]:
    """Return attribution feature indices ranked by absolute magnitude.

    Feature keys must resolve exactly through ``explanation.feature_names`` or
    use a complete index-style key such as ``feature_2``/``f2``. The helper
    intentionally rejects unknown or duplicate mappings instead of inventing
    positions from attribution order.
    """
    if not isinstance(explanation, Explanation):
        raise TypeError("explanation must be an Explanation")
    if not isinstance(descending, bool):
        raise TypeError("descending must be a boolean")

    attributions = explanation.get_attributions()
    if not attributions:
        raise ValueError("No feature attributions found in explanation")
    ranked = explanation.get_top_features(k=len(attributions), absolute=True)
    ranked.sort(key=lambda item: abs(item[1]), reverse=descending)

    indices: list[int] = []
    if explanation.feature_names is not None:
        names = explanation.feature_names
        for feature_key, _ in ranked:
            index = _match_feature_to_index(feature_key, names)
            if index < 0:
                raise ValueError(
                    f"Attribution feature {feature_key!r} does not resolve through feature_names"
                )
            indices.append(index)
    else:
        for feature_key, _ in ranked:
            explicit_index = _explicit_feature_index(feature_key)
            if explicit_index is None:
                raise ValueError(
                    f"Attribution feature {feature_key!r} has no explicit index and "
                    "explanation.feature_names is unavailable"
                )
            indices.append(explicit_index)

    if len(indices) != len(set(indices)):
        raise ValueError("Attribution keys do not map one-to-one to feature indices")
    return indices


def _validate_n_features(n_features: Optional[int]) -> Optional[int]:
    if n_features is None:
        return None
    if isinstance(n_features, bool) or not isinstance(n_features, Integral):
        raise TypeError("n_features must be a positive integer or None")
    if int(n_features) <= 0:
        raise ValueError("n_features must be positive")
    return int(n_features)


def _finite_numeric_array(value: Any, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain real numeric values")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{name} must contain real numeric values")
    array = np.asarray(array, dtype=float)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if any(size == 0 for size in array.shape):
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def compute_baseline_values(
    baseline: Union[str, float, np.ndarray, Callable[[np.ndarray], Any]],
    background_data: Optional[np.ndarray] = None,
    n_features: Optional[int] = None,
) -> np.ndarray:
    """Resolve a finite one-dimensional replacement baseline.

    ``"mean"``/``"median"`` and callable baselines require a finite, nonempty
    two-dimensional background matrix. Returned values are copied and must
    contain exactly ``n_features`` entries when that count is supplied.
    """
    expected_features = _validate_n_features(n_features)

    background: Optional[np.ndarray] = None
    if isinstance(baseline, str) or callable(baseline):
        if background_data is None:
            descriptor = repr(baseline) if isinstance(baseline, str) else "callable baseline"
            raise ValueError(f"background_data is required for {descriptor}")
        background = _finite_numeric_array(background_data, "background_data", ndim=2)
        if expected_features is not None and background.shape[1] != expected_features:
            raise ValueError("background_data column count must equal n_features")

    if isinstance(baseline, str):
        assert background is not None
        if baseline == "mean":
            result = np.mean(background, axis=0)
        elif baseline == "median":
            result = np.median(background, axis=0)
        else:
            raise ValueError("baseline string must be 'mean' or 'median'")
    elif callable(baseline):
        assert background is not None
        result = baseline(background.copy())
    elif isinstance(baseline, np.ndarray):
        result = baseline
    elif isinstance(baseline, Real) and not isinstance(baseline, bool):
        if expected_features is None:
            raise ValueError("n_features is required for a scalar baseline")
        if not np.isfinite(float(baseline)):
            raise ValueError("scalar baseline must be finite")
        result = np.full(expected_features, float(baseline), dtype=float)
    else:
        raise TypeError(
            "baseline must be 'mean', 'median', a finite real scalar, a numpy array, "
            "or a callable"
        )

    values = _finite_numeric_array(result, "resolved baseline", ndim=1)
    if expected_features is not None and values.size != expected_features:
        raise ValueError(
            f"baseline must resolve to shape ({expected_features},); got {values.shape}"
        )
    if background is not None and values.size != background.shape[1]:
        raise ValueError("resolved baseline length must equal background_data columns")
    return values.copy()


def apply_feature_mask(
    instance: np.ndarray,
    feature_indices: Iterable[int],
    baseline_values: np.ndarray,
) -> np.ndarray:
    """Copy one finite feature vector and replace the specified positions."""
    values = _finite_numeric_array(instance, "instance", ndim=1)
    baseline = _finite_numeric_array(baseline_values, "baseline_values", ndim=1)
    if baseline.shape != values.shape:
        raise ValueError("baseline_values must have exactly the same shape as instance")
    if isinstance(feature_indices, (str, bytes)) or not isinstance(feature_indices, Iterable):
        raise TypeError("feature_indices must be an iterable of integers")

    indices = list(feature_indices)
    if any(isinstance(index, bool) or not isinstance(index, Integral) for index in indices):
        raise TypeError("feature_indices must contain only integers")
    normalized = [int(index) for index in indices]
    if len(normalized) != len(set(normalized)):
        raise ValueError("feature_indices must not contain duplicates")
    if any(index < 0 or index >= values.size for index in normalized):
        raise ValueError("feature_indices contain an index outside the instance")

    modified = values.copy()
    modified[normalized] = baseline[normalized]
    return modified


def resolve_k(k: Union[int, float], n_features: int) -> int:
    """Resolve a positive count or a fraction in ``(0, 1]`` to a feature count."""
    validated_features = _validate_n_features(n_features)
    assert validated_features is not None
    if isinstance(k, bool) or not isinstance(k, Real):
        raise TypeError("k must be a positive integer count or real fraction in (0, 1]")
    if not np.isfinite(float(k)):
        raise ValueError("k must be finite")
    if isinstance(k, Integral):
        if int(k) <= 0:
            raise ValueError("integer k must be positive")
        return min(int(k), validated_features)
    if 0.0 < float(k) <= 1.0:
        return max(1, int(float(k) * validated_features))
    raise ValueError("fractional k must lie in (0, 1]")


def _model_task(model: Any) -> Optional[str]:
    """Return explicit or conventionally inferred model-task identity."""
    task = getattr(model, "task", None)
    wrapped_model = getattr(model, "model", None)
    estimator_type = getattr(model, "_estimator_type", None)
    if estimator_type is None and wrapped_model is not None:
        estimator_type = getattr(wrapped_model, "_estimator_type", None)

    inferred = None
    if estimator_type == "classifier":
        inferred = "classification"
    elif estimator_type == "regressor":
        inferred = "regression"
    if task in {"classification", "regression"} and inferred not in {None, task}:
        raise ValueError("model task metadata conflicts with estimator type")
    if task in {"classification", "regression"}:
        return task
    if inferred is not None:
        return inferred

    classes = getattr(model, "classes_", None)
    predict_proba = getattr(model, "predict_proba", None)
    if wrapped_model is not None:
        if classes is None:
            classes = getattr(wrapped_model, "classes_", None)
        if not callable(predict_proba):
            predict_proba = getattr(wrapped_model, "predict_proba", None)
    if classes is not None or callable(predict_proba):
        return "classification"
    return None


def _scalar_values_equal(left: Any, right: Any) -> bool:
    """Compare two prospective class labels without accepting array truthiness."""
    try:
        result = left == right
    except (TypeError, ValueError):
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _model_classes(model: Any) -> Optional[np.ndarray]:
    classes = getattr(model, "classes_", None)
    if classes is None:
        wrapped_model = getattr(model, "model", None)
        classes = getattr(wrapped_model, "classes_", None)
    if classes is None:
        return None
    array = np.asarray(classes)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("model.classes_ must be a non-empty one-dimensional array")
    labels = array.tolist()
    if any(
        _scalar_values_equal(labels[left], labels[right])
        for left in range(len(labels))
        for right in range(left + 1, len(labels))
    ):
        raise ValueError("model.classes_ must contain unique labels")
    return array.copy()


def _validate_single_instance(instance: Any) -> np.ndarray:
    """Return one finite input row as a defensive two-dimensional copy."""
    instance_array = np.asarray(instance)
    if instance_array.dtype == np.bool_ or not np.issubdtype(instance_array.dtype, np.number):
        raise TypeError("instance must contain real numeric values")
    if np.issubdtype(instance_array.dtype, np.complexfloating):
        raise TypeError("instance must contain real numeric values")
    instance_array = np.asarray(instance_array, dtype=float)
    if instance_array.ndim == 1:
        if instance_array.size == 0:
            raise ValueError("instance must not be empty")
        instance_2d = instance_array.reshape(1, -1)
    elif instance_array.ndim == 2 and instance_array.shape[0] == 1:
        if instance_array.shape[1] == 0:
            raise ValueError("instance must not be empty")
        instance_2d = instance_array
    else:
        raise ValueError("instance must represent exactly one row")
    if not np.all(np.isfinite(instance_2d)):
        raise ValueError("instance must contain only finite values")
    return instance_2d.copy()


def _single_raw_output(raw_output: Any, source: str) -> np.ndarray:
    """Check that a model returned exactly one scalar or vector for one row."""
    array = np.asarray(raw_output)
    if array.ndim == 0:
        values = array.reshape(1)
    elif array.ndim == 1:
        values = array
    elif array.ndim == 2 and array.shape[0] == 1:
        values = array[0]
    else:
        raise ValueError(
            f"model {source} must return one scalar/vector for one input; got shape {array.shape}"
        )
    values = values.reshape(-1)
    if values.size == 0:
        raise ValueError(f"model {source} returned an empty output vector")
    return values


def _single_prediction_vector(raw_output: Any, source: str) -> np.ndarray:
    array = _single_raw_output(raw_output, source)
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"model {source} output must be real numeric")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"model {source} output must be real numeric")
    array = np.asarray(array, dtype=float)
    values = array.reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"model {source} output must be finite")
    return values


def _hard_label_vector(raw_output: Any, classes: np.ndarray) -> Optional[np.ndarray]:
    """Convert one class label returned by ``predict`` to an indicator vector."""
    raw_values = _single_raw_output(raw_output, "predict")
    if raw_values.size != 1:
        return None
    label = raw_values[0]
    matches = [
        index
        for index, candidate in enumerate(classes.tolist())
        if _scalar_values_equal(label, candidate)
    ]
    if not matches:
        return None
    result: np.ndarray = np.zeros(classes.size, dtype=float)
    result[matches[0]] = 1.0
    return result


def _validate_probability_vector(
    values: np.ndarray,
    classes: Optional[np.ndarray],
) -> np.ndarray:
    """Validate a complete finite classification-probability vector."""
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("classification probabilities must lie in [0, 1]")
    if not np.isclose(float(np.sum(values)), 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError("classification probabilities must sum to 1")
    if classes is not None and classes.size != values.size:
        raise ValueError("classification output width does not match model.classes_")
    return values.copy()


def _get_prediction_proba_vector(model: Any, instance: np.ndarray) -> np.ndarray:
    """Return one finite model-output vector for one finite instance.

    For classifiers exposing a single binary probability, the result is
    expanded to ``[P(class 0), P(class 1)]``. Classifiers whose ``predict``
    method returns one label are converted to a one-hot vector when
    ``classes_`` makes that mapping unambiguous. Unknown ``predict`` outputs
    are rejected because classification and regression semantics cannot be
    inferred from shape alone.
    """
    instance_2d = _validate_single_instance(instance)
    task = _model_task(model)

    predict_proba = getattr(model, "predict_proba", None)
    predict = getattr(model, "predict", None)
    if task == "regression":
        if not callable(predict):
            raise TypeError("a regression model must expose a callable predict method")
        raw_output = predict(instance_2d.copy())
        values = _single_prediction_vector(raw_output, "predict")
        return values.copy()
    if callable(predict_proba):
        raw_output = predict_proba(instance_2d.copy())
        source_is_probability = True
    elif callable(predict):
        raw_output = predict(instance_2d.copy())
        source_is_probability = False
    else:
        raise TypeError("model must expose a callable predict or predict_proba method")

    classes = _model_classes(model)
    if not source_is_probability and task != "classification":
        raise ValueError(
            "model predict output is ambiguous; expose task='classification' or "
            "task='regression', or provide predict_proba"
        )
    if not source_is_probability and classes is not None:
        hard_label_values = _hard_label_vector(raw_output, classes)
        if hard_label_values is not None:
            return hard_label_values

    values = _single_prediction_vector(
        raw_output,
        "predict_proba" if source_is_probability else "predict",
    )
    if values.size == 1:
        if classes is not None and classes.size == 1:
            return _validate_probability_vector(values, classes)
        if classes is not None and classes.size != 2:
            raise ValueError("one-column binary classification output requires exactly two classes")
        positive_probability = float(values[0])
        if not 0.0 <= positive_probability <= 1.0:
            raise ValueError("one-column classification output must be P(class 1) in [0, 1]")
        expanded = np.array([1.0 - positive_probability, positive_probability])
        return _validate_probability_vector(expanded, classes)

    return _validate_probability_vector(values, classes)


def resolve_target_class(model: Any, instance: np.ndarray) -> int:
    """Resolve one original-input output index and keep it fixed thereafter."""
    task = _model_task(model)
    outputs = _get_prediction_proba_vector(model, instance)
    if task == "regression":
        if outputs.size != 1:
            raise ValueError("multi-output regression requires an explicit output index")
        return 0
    return int(np.argmax(outputs)) if outputs.size > 1 else 0


def get_prediction_value(
    model: Any,
    instance: np.ndarray,
    output_type: str = "probability",
    target_class: Optional[int] = None,
) -> float:
    """Return a finite scalar model output or a classifier's predicted index.

    ``output_type="probability"`` is the historical name for scalar-output
    selection. For regression it returns the selected raw regression output;
    for classification adapters it returns a probability. Comparisons across
    perturbations must pass one fixed ``target_class``/output index.
    """
    if output_type not in {"probability", "class"}:
        raise ValueError("output_type must be 'probability' or 'class'")
    if target_class is not None and (
        isinstance(target_class, bool) or not isinstance(target_class, Integral)
    ):
        raise TypeError("target_class must be an integer or None")
    if output_type == "class" and target_class is not None:
        raise ValueError("target_class is not used with output_type='class'")

    task = _model_task(model)
    if output_type == "class" and task == "regression":
        raise ValueError("output_type='class' is undefined for regression")
    outputs = _get_prediction_proba_vector(model, instance)
    if output_type == "class":
        return float(np.argmax(outputs)) if outputs.size > 1 else 0.0

    if target_class is None:
        if task == "regression" and outputs.size != 1:
            raise ValueError("multi-output regression requires an explicit output index")
        index = int(np.argmax(outputs)) if outputs.size > 1 else 0
    else:
        index = int(target_class)
        if index < 0 or index >= outputs.size:
            raise ValueError(
                f"target_class={index} is invalid for a model with {outputs.size} output value(s)"
            )
    return float(outputs[index])


def compute_prediction_change(
    model: Any,
    original: np.ndarray,
    perturbed: np.ndarray,
    metric: str = "absolute",
) -> float:
    """Compare one fixed original-input output across two instances.

    ``metric="relative"`` implements ``|f(x)-f(x')| / |f(x)|`` exactly.
    A zero denominator returns ``NaN`` for ``0/0`` and positive infinity for a
    nonzero numerator instead of applying epsilon smoothing.
    """
    if metric not in {"absolute", "relative"}:
        raise ValueError("metric must be 'absolute' or 'relative'")
    original_instance = _validate_single_instance(original)
    perturbed_instance = _validate_single_instance(perturbed)
    if original_instance.shape != perturbed_instance.shape:
        raise ValueError("original and perturbed must have the same feature shape")

    task = _model_task(model)
    original_outputs = _get_prediction_proba_vector(model, original_instance)
    if task == "regression":
        if original_outputs.size != 1:
            raise ValueError("multi-output regression requires an explicit output index")
        original_index = 0
    else:
        original_index = int(np.argmax(original_outputs)) if original_outputs.size > 1 else 0

    perturbed_outputs = _get_prediction_proba_vector(model, perturbed_instance)
    if perturbed_outputs.size != original_outputs.size:
        raise ValueError("model output width changed between original and perturbed inputs")

    original_value = float(original_outputs[original_index])
    perturbed_value = float(perturbed_outputs[original_index])
    numerator = abs(original_value - perturbed_value)
    if metric == "absolute":
        return numerator
    if original_value == 0.0:
        return float("nan") if numerator == 0.0 else float("inf")
    return numerator / abs(original_value)
