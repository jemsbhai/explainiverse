"""
First- and second-order partial dependence.

The explainer averages a declared prediction output over the empirical
reference rows. Classification versus regression is determined by an explicit
task contract, not by prediction array dimensionality.

Reference:
    Friedman, J.H. (2001). Greedy function approximation: A gradient boosting
    machine. Annals of Statistics, 29(5), 1189-1232.
"""

from numbers import Integral
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation
from explainiverse.explainers._validation import as_real_array, validate_name_sequence

Feature = Union[int, str]
FeatureSpec = Union[Feature, Tuple[Feature, Feature]]
_VALID_TASKS = {"classification", "regression"}


def _resolve_task(model, task: Optional[str]) -> str:
    """Resolve a declared task without guessing from prediction shape."""
    if task is not None and task not in _VALID_TASKS:
        raise ValueError(f"task must be 'classification' or 'regression'; got {task!r}")
    model_task = getattr(model, "task", None)
    if model_task is not None and model_task not in _VALID_TASKS:
        raise ValueError(
            "model.task must be 'classification' or 'regression'; " f"got {model_task!r}"
        )
    if task is not None and model_task is not None and task != model_task:
        raise ValueError(f"task={task!r} conflicts with model.task={model_task!r}")
    resolved = task or model_task
    if resolved is None:
        raise ValueError(
            "The model task is ambiguous. Pass task='classification' or "
            "task='regression', or use an adapter that declares model.task."
        )
    return resolved


def _normalize_predictions(predictions: np.ndarray, n_samples: int) -> np.ndarray:
    predictions = as_real_array(
        predictions,
        name="model predictions",
        require_finite=True,
    )
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    if predictions.ndim != 2 or predictions.shape[0] != n_samples:
        raise ValueError(
            "model.predict must return shape (n_samples,), (n_samples, 1), "
            "or (n_samples, n_outputs)"
        )
    if predictions.shape[1] == 0:
        raise ValueError("model.predict returned no output columns")
    return predictions


def _validate_output_index(output_index: Optional[int], n_outputs: int) -> int:
    if output_index is None:
        raise ValueError("An explicit output index is required")
    if not isinstance(output_index, Integral) or isinstance(output_index, bool):
        raise TypeError("target_class must be an integer or None")
    output_index = int(output_index)
    if output_index < 0 or output_index >= n_outputs:
        raise ValueError(f"target_class must be in [0, {n_outputs - 1}]; got {output_index}")
    return output_index


def _resolve_output_index(predictions: np.ndarray, task: str, target_class: Optional[int]) -> int:
    """Resolve the class/output column under the declared task."""
    n_outputs = predictions.shape[1]
    if task == "classification":
        # A one-column classifier is the documented P(class 1) convention.
        if n_outputs == 1:
            target = 1 if target_class is None else target_class
            if not isinstance(target, Integral) or isinstance(target, bool):
                raise TypeError("target_class must be an integer or None")
            target = int(target)
            if target not in (0, 1):
                raise ValueError("A one-column binary classifier only supports target_class 0 or 1")
            return target

        # Preserve the documented positive/second-class default while making
        # the selected output explicit in the returned metadata.
        target = 1 if target_class is None else target_class
        return _validate_output_index(target, n_outputs)

    # Regression: a single response needs no explicit index; a multi-output
    # response does, because averaging an arbitrary column is not meaningful.
    if n_outputs == 1:
        if target_class is None:
            return 0
        return _validate_output_index(target_class, 1)
    return _validate_output_index(target_class, n_outputs)


def _select_prediction_output(
    predictions: np.ndarray,
    task: str,
    output_index: int,
) -> np.ndarray:
    """Select a response/class score with one-column binary semantics."""
    if task == "classification" and predictions.shape[1] == 1:
        positive_probability = as_real_array(
            predictions[:, 0],
            name="one-column classification output",
            dtype=float,
            require_finite=True,
        )
        if np.any((positive_probability < 0.0) | (positive_probability > 1.0)):
            raise ValueError(
                "A one-column classification output must contain P(class 1) " "values in [0, 1]"
            )
        return positive_probability if output_index == 1 else 1.0 - positive_probability

    if output_index >= predictions.shape[1]:
        raise ValueError("model.predict changed its number of output columns during explanation")
    selected = as_real_array(
        predictions[:, output_index],
        name="selected model output",
        dtype=float,
        require_finite=True,
    )
    return selected


class PartialDependenceExplainer(BaseExplainer):
    """Compute brute-force partial dependence over reference observations."""

    def __init__(
        self,
        model,
        X: np.ndarray,
        feature_names: List[str],
        grid_resolution: int = 50,
        percentile_range: Tuple[float, float] = (5, 95),
        task: Optional[str] = None,
        categorical_features: Optional[Iterable[Feature]] = None,
    ):
        """
        Initialize the PDP explainer.

        ``categorical_features`` must explicitly identify categorical columns.
        Their grids contain observed categories only; the explainer never
        interpolates synthetic categories. Numeric columns with no more unique
        values than ``grid_resolution`` also use their observed values, matching
        the conventional PDP grid rule for low-cardinality features.
        """
        super().__init__(model)
        self.X: np.ndarray = np.asarray(X)
        validated_names = validate_name_sequence(feature_names, name="feature_names")
        assert validated_names is not None
        self.feature_names: List[str] = validated_names
        self.task: str = _resolve_task(model, task)

        if self.X.ndim != 2 or self.X.shape[0] == 0 or self.X.shape[1] == 0:
            raise ValueError("X must be a non-empty 2D array")
        if len(self.feature_names) != self.X.shape[1]:
            raise ValueError("feature_names length must equal the number of columns in X")
        if not isinstance(grid_resolution, Integral) or isinstance(grid_resolution, bool):
            raise TypeError("grid_resolution must be an integer")
        if grid_resolution < 2:
            raise ValueError("grid_resolution must be at least 2")
        if len(percentile_range) != 2:
            raise ValueError("percentile_range must contain exactly two values")
        lower, upper = (float(percentile_range[0]), float(percentile_range[1]))
        if not (0.0 <= lower < upper <= 100.0):
            raise ValueError("percentile_range must satisfy 0 <= lower < upper <= 100")

        self.grid_resolution: int = int(grid_resolution)
        self.percentile_range: Tuple[float, float] = (lower, upper)
        self.categorical_features: set[int] = self._normalize_categorical_features(
            categorical_features
        )

    def _normalize_categorical_features(
        self, categorical_features: Optional[Iterable[Feature]]
    ) -> set[int]:
        if categorical_features is None:
            return set()

        values = list(categorical_features)
        if values and all(isinstance(value, (bool, np.bool_)) for value in values):
            if len(values) != self.X.shape[1]:
                raise ValueError(
                    "A categorical_features boolean mask must have one entry " "per feature"
                )
            return {idx for idx, is_categorical in enumerate(values) if is_categorical}

        result: set[int] = set()
        for feature in values:
            result.add(self._get_feature_idx(feature))
        return result

    def _get_feature_idx(self, feature: Feature) -> int:
        """Convert a validated feature name/index to its column index."""
        if isinstance(feature, str):
            try:
                return self.feature_names.index(feature)
            except ValueError as exc:
                raise ValueError(f"Unknown feature name: {feature!r}") from exc
        if not isinstance(feature, Integral) or isinstance(feature, bool):
            raise TypeError("feature must be an integer index or feature name")
        feature_idx = int(feature)
        if feature_idx < 0 or feature_idx >= self.X.shape[1]:
            raise ValueError(
                f"feature index must be in [0, {self.X.shape[1] - 1}]; " f"got {feature_idx}"
            )
        return feature_idx

    def _create_grid(self, feature_idx: int) -> np.ndarray:
        """Create an observed categorical grid or a numeric PDP grid."""
        values = self.X[:, feature_idx]
        if feature_idx in self.categorical_features:
            return np.unique(values)

        try:
            numeric_values = as_real_array(
                values,
                name=f"Feature {self.feature_names[feature_idx]!r}",
                dtype=float,
            )
        except ValueError as exc:
            if "complex values" in str(exc):
                raise
            raise TypeError(
                f"Feature {self.feature_names[feature_idx]!r} is non-numeric. "
                "Declare it in categorical_features to use observed categories."
            ) from exc
        if not np.all(np.isfinite(numeric_values)):
            raise ValueError(
                f"Feature {self.feature_names[feature_idx]!r} contains non-finite values"
            )

        unique_values = np.unique(numeric_values)
        if len(unique_values) <= self.grid_resolution:
            return unique_values

        lower, upper = np.percentile(numeric_values, self.percentile_range)
        return np.linspace(lower, upper, self.grid_resolution)

    def _selected_predictions(self, X: np.ndarray, output_index: int) -> np.ndarray:
        predictions = _normalize_predictions(self.model.predict(X), X.shape[0])
        return _select_prediction_output(predictions, self.task, output_index)

    def _grid_copy(self, feature_indices: Tuple[int, ...]) -> np.ndarray:
        """Copy X without truncating continuous grid values into integer columns."""
        has_continuous_feature = any(
            feature_idx not in self.categorical_features for feature_idx in feature_indices
        )
        if has_continuous_feature and np.issubdtype(self.X.dtype, np.integer):
            return self.X.astype(float, copy=True)
        return self.X.copy()

    def _compute_pdp_1d(
        self, feature_idx: int, target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute one-dimensional partial dependence."""
        reference = _normalize_predictions(self.model.predict(self.X[:1]), 1)
        output_index = _resolve_output_index(reference, self.task, target_class)
        grid = self._create_grid(feature_idx)
        pdp_values: np.ndarray = np.empty(len(grid), dtype=float)

        for grid_idx, value in enumerate(grid):
            X_temp = self._grid_copy((feature_idx,))
            X_temp[:, feature_idx] = value
            pdp_values[grid_idx] = np.mean(self._selected_predictions(X_temp, output_index))
        return grid, pdp_values

    def _compute_pdp_2d(
        self,
        feature_idx1: int,
        feature_idx2: int,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute two-dimensional partial dependence."""
        if feature_idx1 == feature_idx2:
            raise ValueError("A PDP interaction requires two distinct features")
        reference = _normalize_predictions(self.model.predict(self.X[:1]), 1)
        output_index = _resolve_output_index(reference, self.task, target_class)
        grid1 = self._create_grid(feature_idx1)
        grid2 = self._create_grid(feature_idx2)
        pdp_values: np.ndarray = np.empty((len(grid1), len(grid2)), dtype=float)

        for i, value1 in enumerate(grid1):
            for j, value2 in enumerate(grid2):
                X_temp = self._grid_copy((feature_idx1, feature_idx2))
                X_temp[:, feature_idx1] = value1
                X_temp[:, feature_idx2] = value2
                pdp_values[i, j] = np.mean(self._selected_predictions(X_temp, output_index))
        return grid1, grid2, pdp_values

    def explain(  # type: ignore[override]
        self,
        features: List[FeatureSpec],
        target_class: Optional[int] = None,
        **kwargs,
    ) -> Explanation:
        """Compute partial dependence for the requested features."""
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unexpected}")
        if not isinstance(features, (list, tuple)) or not features:
            raise ValueError("features must be a non-empty list or tuple")

        reference = _normalize_predictions(self.model.predict(self.X[:1]), 1)
        output_index = _resolve_output_index(reference, self.task, target_class)
        pdp_results: dict[str, object] = {}
        grid_results: dict[str, object] = {}
        grid_types: dict[str, object] = {}

        for feature in features:
            if isinstance(feature, tuple):
                if len(feature) != 2:
                    raise ValueError("A PDP interaction tuple must contain two features")
                idx1 = self._get_feature_idx(feature[0])
                idx2 = self._get_feature_idx(feature[1])
                grid1, grid2, pdp = self._compute_pdp_2d(idx1, idx2, output_index)
                key = f"{self.feature_names[idx1]}_x_{self.feature_names[idx2]}"
                pdp_results[key] = pdp.tolist()
                grid_results[key] = {
                    "grid1": grid1.tolist(),
                    "grid2": grid2.tolist(),
                }
                grid_types[key] = [
                    "categorical" if idx1 in self.categorical_features else "numeric",
                    "categorical" if idx2 in self.categorical_features else "numeric",
                ]
            else:
                idx = self._get_feature_idx(feature)
                grid, pdp = self._compute_pdp_1d(idx, output_index)
                key = self.feature_names[idx]
                pdp_results[key] = pdp.tolist()
                grid_results[key] = grid.tolist()
                grid_types[key] = "categorical" if idx in self.categorical_features else "numeric"

        target_name = (
            f"class_{output_index}" if self.task == "classification" else f"output_{output_index}"
        )
        output_space = (
            "classification_score" if self.task == "classification" else "regression_response"
        )
        return Explanation(
            explainer_name="PartialDependence",
            target_class=target_name,
            feature_names=self.feature_names,
            explanation_data={
                "pdp_values": pdp_results,
                "grid_values": grid_results,
                "grid_types": grid_types,
                "features_analyzed": [str(feature) for feature in features],
                "interaction": any(isinstance(feature, tuple) for feature in features),
                "task": self.task,
                "output_index": output_index,
                "output_space": output_space,
            },
            metadata={
                "task": self.task,
                "output_index": output_index,
                "output_space": output_space,
                "prediction_output": "model.predict",
                "averaging": "empirical_reference_mean",
                "percentile_range": self.percentile_range,
                "grid_resolution": self.grid_resolution,
            },
        )
