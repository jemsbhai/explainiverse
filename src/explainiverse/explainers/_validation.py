"""Shared validation helpers for tabular explainer boundaries.

These helpers intentionally live below the explainer package rather than in an
adapter.  Adapters retain their model-specific output contracts; explainers use
the helpers to enforce the narrower contracts required by their algorithms.
"""

from typing import Any, Iterable, List, Optional, Sequence, cast

import numpy as np

_PREDICTION_OUTPUT_KINDS = frozenset(
    {"probabilities", "class_labels", "scores", "regression_values"}
)


def _contains_complex_values(values: np.ndarray) -> bool:
    """Return whether an array contains native or object-wrapped complex values."""

    object_values = cast(Iterable[Any], values.flat)
    return bool(
        np.iscomplexobj(values)
        or (
            values.dtype == object
            and any(isinstance(value, (complex, np.complexfloating)) for value in object_values)
        )
    )


def as_real_array(
    values,
    *,
    name: str,
    dtype=None,
    require_finite: bool = False,
) -> np.ndarray:
    """Convert an array-like value without silently discarding imaginary parts.

    NumPy permits several complex-to-real casts with only a warning. Explainer
    algorithms in this package are defined over real-valued inputs, so their
    public and method-local numeric boundaries must reject complex values before
    requesting a real dtype.
    """

    try:
        raw_values = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an array-like value") from exc
    if _contains_complex_values(raw_values):
        raise ValueError(f"{name} must not contain complex values")

    try:
        array = np.asarray(raw_values, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} could not be converted to the required numeric values") from exc

    if require_finite:
        try:
            finite = np.isfinite(array)
        except TypeError as exc:
            raise ValueError(f"{name} must contain finite numerical values") from exc
        if not np.all(finite):
            raise ValueError(f"{name} must contain only finite values")
    return array


def validate_name_sequence(
    names,
    *,
    name: str,
    allow_none: bool = False,
    allow_empty: bool = False,
) -> Optional[List[str]]:
    """Return validated display names while preserving their exact spelling."""

    if names is None:
        if allow_none:
            return None
        raise ValueError(f"{name} must be a non-empty sequence")
    if isinstance(names, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of non-empty strings")
    try:
        values = list(names)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of non-empty strings") from exc
    if not values and not allow_empty:
        raise ValueError(f"{name} must be non-empty")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be unique")
    return values


def validate_single_tabular_instance(
    instance,
    n_features: int,
    *,
    name: str = "instance",
    dtype=None,
    require_finite: bool = True,
) -> np.ndarray:
    """Return one feature vector without flattening batch dimensions.

    A single instance may be supplied as ``(n_features,)`` or
    ``(1, n_features)``.  Other shapes are rejected even when their total
    element count happens to equal ``n_features``.
    """

    values = as_real_array(instance, name=name, dtype=dtype)

    if values.ndim == 1:
        row = values
    elif values.ndim == 2 and values.shape[0] == 1:
        row = values[0]
    else:
        raise ValueError(f"{name} must be a 1D feature vector or a single-row 2D array")

    if row.shape != (int(n_features),):
        raise ValueError(f"{name} must contain exactly one value per feature")

    if require_finite:
        try:
            finite_values = np.asarray(row, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain finite numerical values") from exc
        if not np.all(np.isfinite(finite_values)):
            raise ValueError(f"{name} must contain only finite values")

    return row


def ensure_classification_task(model, *, context: str) -> Optional[str]:
    """Reject declared regression semantics for a classifier-only explainer.

    Black-box models without task metadata remain supported.  When task
    metadata exists on an adapter or wrapped estimator, contradictions and
    regression semantics fail before any prediction is reinterpreted.
    """

    declared_tasks = []
    model_task = getattr(model, "task", None)
    if model_task is not None:
        if model_task not in {"classification", "regression"}:
            raise ValueError("model.task must be 'classification' or 'regression'")
        declared_tasks.append(model_task)

    raw_model = getattr(model, "model", model)
    estimator_type = getattr(raw_model, "_estimator_type", None)
    if estimator_type in {"classifier", "regressor"}:
        declared_tasks.append("classification" if estimator_type == "classifier" else "regression")

    if len(set(declared_tasks)) > 1:
        raise ValueError("model.task conflicts with the wrapped estimator semantics")
    if declared_tasks and declared_tasks[0] == "regression":
        raise ValueError(f"{context} requires a classification model; got regression semantics")
    return declared_tasks[0] if declared_tasks else None


def get_prediction_output_kind(model) -> Optional[str]:
    """Return an explicitly declared ``predict`` output contract, if present.

    Adapters may expose ``prediction_output_kind`` as ``"probabilities"``,
    ``"class_labels"``, ``"scores"``, or ``"regression_values"``. The marker
    removes the otherwise irreducible ambiguity between numerical hard labels
    and endpoint probabilities such as ``[0, 1]``. Legacy models without a
    marker continue through the historical shape/metadata heuristics.
    """

    kind = getattr(model, "prediction_output_kind", None)
    if kind is None:
        wrapped = getattr(model, "model", None)
        if wrapped is not None and wrapped is not model:
            kind = getattr(wrapped, "prediction_output_kind", None)
    if kind is None:
        return None
    if not isinstance(kind, str):
        raise TypeError("model.prediction_output_kind must be a string or None")
    if kind not in _PREDICTION_OUTPUT_KINDS:
        supported = ", ".join(sorted(_PREDICTION_OUTPUT_KINDS))
        raise ValueError(
            f"Unknown model.prediction_output_kind {kind!r}; " f"supported values are: {supported}"
        )
    return kind


def _model_class_labels(model) -> Optional[np.ndarray]:
    """Return the one-dimensional class order exposed by a model, if any."""

    raw_model = getattr(model, "model", model)
    classes = getattr(model, "classes_", None)
    if classes is None:
        classes = getattr(raw_model, "classes_", None)
    if classes is None:
        return None
    if isinstance(classes, (list, tuple)) and any(np.asarray(value).ndim > 0 for value in classes):
        raise ValueError("multi-output classification is not supported")
    labels = np.asarray(classes)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("model.classes_ must be a non-empty one-dimensional array")
    if np.unique(labels).size != labels.size:
        raise ValueError("model.classes_ must contain unique labels")
    return labels


def _class_labels_to_indicator(
    raw: np.ndarray,
    *,
    model_labels: Optional[np.ndarray],
    display_names: Optional[List[str]],
    context: str,
) -> np.ndarray:
    """Map an explicitly or heuristically identified hard-label vector."""

    if model_labels is not None:
        matches = raw[:, None] == model_labels[None, :]
        if np.all(matches.sum(axis=1) == 1):
            return matches.astype(float)
        unknown = raw[matches.sum(axis=1) != 1]
        raise ValueError(
            f"{context} returned class labels not present in model.classes_: " f"{unknown.tolist()}"
        )

    if display_names is not None:
        display_array = np.asarray(display_names, dtype=object)
        direct_matches = raw[:, None] == display_array[None, :]
        if np.all(direct_matches.sum(axis=1) == 1):
            return direct_matches.astype(float)
        try:
            numeric_labels = np.asarray(raw, dtype=float)
        except (TypeError, ValueError):
            numeric_labels = None
        if numeric_labels is not None:
            integer_labels = numeric_labels.astype(int)
            if np.all(numeric_labels == integer_labels) and np.all(
                (integer_labels >= 0) & (integer_labels < len(display_names))
            ):
                return np.eye(len(display_names), dtype=float)[integer_labels]

    raise ValueError(
        f"{context} cannot map hard class labels to output columns; expose "
        "model.classes_ or class_names"
    )


def normalize_classifier_outputs(
    model,
    X: np.ndarray,
    *,
    context: str,
    class_names: Optional[Sequence[str]] = None,
    require_probabilities: bool,
    allow_label_predictions: bool = True,
) -> np.ndarray:
    """Normalize supported classifier outputs to ``(samples, classes)``.

    One-column and one-dimensional numerical outputs use the documented binary
    positive-class probability convention.  One-dimensional hard labels are
    mapped through ``model.classes_`` when allowed.  Multi-column outputs may be
    arbitrary finite class scores for rule precision, while probability-based
    algorithms can request bounds and row-sum validation.
    """

    ensure_classification_task(model, context=context)
    matrix = np.asarray(X)
    if matrix.ndim != 2:
        raise ValueError(f"{context} model inputs must be a two-dimensional array")
    if not hasattr(model, "predict"):
        raise TypeError(f"{context} requires a model with a batched predict method")

    declared_output_kind = get_prediction_output_kind(model)
    if declared_output_kind == "regression_values":
        raise ValueError(
            f"{context} requires classifier outputs, but model.predict declares "
            "regression_values"
        )

    used_predict_proba = require_probabilities and hasattr(model, "predict_proba")
    prediction_method = model.predict_proba if used_predict_proba else model.predict
    output_kind = "probabilities" if used_predict_proba else declared_output_kind
    raw = np.asarray(prediction_method(matrix))
    if raw.ndim == 0:
        raise ValueError(f"{context} model.predict must retain a sample dimension")
    if raw.shape[0] != matrix.shape[0]:
        raise ValueError(f"{context} model.predict returned the wrong number of rows")

    model_labels = _model_class_labels(model)
    display_names = validate_name_sequence(
        class_names,
        name="class_names",
        allow_none=True,
    )
    if model_labels is not None and display_names is not None:
        if len(model_labels) != len(display_names):
            raise ValueError("class_names length must match model.classes_")

    if _contains_complex_values(raw):
        raise ValueError(f"{context} classifier outputs must not contain complex values")

    normalized = None
    if raw.ndim == 1:
        if output_kind == "scores":
            raise ValueError(f"{context} class-score outputs must contain one column per class")
        if output_kind == "class_labels":
            if not allow_label_predictions:
                raise ValueError(
                    f"{context} requires probabilities, not hard class-label predictions"
                )
            normalized = _class_labels_to_indicator(
                raw,
                model_labels=model_labels,
                display_names=display_names,
                context=context,
            )
        elif output_kind is None:
            # Backward-compatible heuristic: known class values are labels.
            # Endpoint probability vectors are inherently ambiguous here;
            # adapters can declare ``prediction_output_kind='probabilities'``
            # to select probability semantics deterministically.
            if model_labels is not None:
                label_matches = raw[:, None] == model_labels[None, :]
                if np.all(label_matches.sum(axis=1) == 1):
                    if not allow_label_predictions:
                        raise ValueError(
                            f"{context} requires probabilities, not hard class-label predictions; "
                            "declare prediction_output_kind='probabilities' when endpoint "
                            "values are probabilities"
                        )
                    normalized = label_matches.astype(float)

            if normalized is None and allow_label_predictions and display_names is not None:
                try:
                    normalized = _class_labels_to_indicator(
                        raw,
                        model_labels=None,
                        display_names=display_names,
                        context=context,
                    )
                except ValueError:
                    normalized = None

        if normalized is None:
            try:
                positive = np.asarray(raw, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{context} requires numerical probabilities or mappable class labels"
                ) from exc
            normalized = np.column_stack((1.0 - positive, positive))
    elif raw.ndim == 2:
        if raw.shape[1] == 0:
            raise ValueError(f"{context} model.predict returned no output columns")
        if output_kind == "class_labels":
            raise ValueError(f"{context} hard class-label outputs must be one-dimensional")
        try:
            numerical = raw.astype(float, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context} classifier outputs must be numerical") from exc
        if numerical.shape[1] == 1:
            if output_kind == "scores":
                raise ValueError(f"{context} class-score outputs must contain one column per class")
            positive = numerical[:, 0]
            normalized = np.column_stack((1.0 - positive, positive))
        else:
            normalized = numerical
    else:
        raise ValueError(f"{context} model.predict must return a 1D or 2D array")

    normalized = np.asarray(normalized, dtype=float)
    if not np.all(np.isfinite(normalized)):
        raise ValueError(f"{context} model.predict returned non-finite classifier outputs")

    expected_width = None
    if model_labels is not None:
        expected_width = len(model_labels)
    elif display_names is not None:
        expected_width = len(display_names)
    if expected_width is not None and normalized.shape[1] != expected_width:
        raise ValueError(
            f"{context} model returned {normalized.shape[1]} class columns but "
            f"class metadata describes {expected_width} classes"
        )

    # Binary vectors are always interpreted as probabilities, even for an
    # algorithm that otherwise permits arbitrary multi-class scores.
    binary_probability_input = raw.ndim == 1 or (raw.ndim == 2 and raw.shape[1] == 1)
    if require_probabilities and output_kind == "scores":
        raise ValueError(f"{context} requires probabilities, not arbitrary class scores")
    if require_probabilities or output_kind == "probabilities" or binary_probability_input:
        if np.any(normalized < -1e-8) or np.any(normalized > 1.0 + 1e-8):
            raise ValueError(f"{context} requires probabilities in [0, 1]")
        if not np.allclose(normalized.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6):
            raise ValueError(f"{context} requires probability rows that sum to 1")
        normalized = np.clip(normalized, 0.0, 1.0)

    return normalized
