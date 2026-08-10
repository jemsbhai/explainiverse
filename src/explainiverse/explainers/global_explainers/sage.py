# src/explainiverse/explainers/global_explainers/sage.py
"""SAGE (Shapley Additive Global importancE) explainer.

This module implements permutation sampling for the SAGE loss game described
by Covert, Lundberg, and Lee (2020). Missing features are handled with a
*marginal imputer*: model predictions are averaged over a background data set
before the loss is evaluated. Averaging predictions before applying the loss
is essential; evaluating the loss on individual random completions defines a
different cooperative game and does not satisfy SAGE efficiency.

Reference:
    Covert, I., Lundberg, S., & Lee, S.I. (2020). Understanding Global Feature
    Contributions with Additive Importance Measures. NeurIPS 2020.
"""

from numbers import Integral
from typing import Callable, List, Literal, Optional, TypeAlias

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation
from explainiverse.explainers._validation import as_real_array, validate_name_sequence

SAGETask: TypeAlias = Literal["classification", "regression"]
LossValue: TypeAlias = float | np.ndarray
LossFunction: TypeAlias = Callable[[np.ndarray, np.ndarray], LossValue]


class SAGEExplainer(BaseExplainer):
    """
    SAGE: Shapley Additive Global importancE.

    Compute global Shapley values for predictive loss.

    For a coalition ``S``, this implementation evaluates the restricted model
    ``f_S(x_S) = E[f(x_S, X_not_S)]`` with an empirical marginal expectation
    over ``background_data``. It then estimates the Shapley values of the loss
    reduction game by sampling feature permutations.

    Attributes:
        model: Model adapter with .predict() method
        X: Feature matrix
        y: Target values
        feature_names: List of feature names
        n_permutations: Number of permutation samples for approximation
        loss_fn: Loss function (default: zero-one loss for classification and
            mean squared error for regression)
    """

    def __init__(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        n_permutations: int = 100,
        loss_fn: Optional[LossFunction] = None,
        task: SAGETask = "classification",
        random_state: int = 42,
        background_data: Optional[np.ndarray] = None,
    ):
        """
        Initialize the SAGE explainer.

        Args:
            model: Model adapter with .predict() method
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            feature_names: List of feature names
            n_permutations: Number of permutations for approximation
            loss_fn: Custom loss function (lower is better)
            task: "classification" or "regression"
            random_state: Random seed.
            background_data: Empirical marginal distribution used to average
                out missing features. Defaults to ``X``.
        """
        super().__init__(model)
        self.X: np.ndarray = np.asarray(X)
        self.y: np.ndarray = np.asarray(y)
        validated_names = validate_name_sequence(feature_names, name="feature_names")
        assert validated_names is not None
        self.feature_names: List[str] = validated_names
        if not isinstance(n_permutations, Integral) or isinstance(n_permutations, bool):
            raise TypeError("n_permutations must be an integer")
        self.n_permutations: int = int(n_permutations)
        self.task: SAGETask = task
        if not isinstance(random_state, Integral) or isinstance(random_state, bool):
            raise TypeError("random_state must be an integer")
        self.random_state: int = int(random_state)
        self.background_data: np.ndarray = (
            self.X.copy() if background_data is None else np.asarray(background_data)
        )

        self._validate_inputs()
        self.class_labels: Optional[np.ndarray] = (
            self._resolve_class_labels() if task == "classification" else None
        )
        self.loss_fn: LossFunction
        self.loss_name: str
        self.loss_direction: str
        self.loss_is_custom: bool

        if loss_fn is None:
            if task == "classification":
                self.loss_fn = self._zero_one_loss
                self.loss_name = "zero_one_loss"
                # Validate the probability-column contract before the expensive
                # permutation loop rather than failing on an arbitrary coalition.
                self._zero_one_loss(self.y[:1], self.model.predict(self.X[:1]))
            else:
                self.loss_fn = self._mean_squared_error
                self.loss_name = "mean_squared_error"
            self.loss_is_custom = False
        else:
            self.loss_fn = loss_fn
            self.loss_name = getattr(loss_fn, "__name__", type(loss_fn).__name__)
            self.loss_is_custom = True
        self.loss_direction = "lower_is_better"

    def _validate_inputs(self) -> None:
        """Validate the SAGE game before performing expensive model calls."""
        if self.X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if self.X.shape[0] == 0 or self.X.shape[1] == 0:
            raise ValueError("X must contain at least one sample and feature")
        if self.y.ndim == 0 or self.y.shape[0] != self.X.shape[0]:
            raise ValueError("y must contain one target row per X row")
        if self.task == "classification" and self.y.ndim != 1:
            raise ValueError("classification y must be a one-dimensional label array")
        if len(self.feature_names) != self.X.shape[1]:
            raise ValueError("feature_names length must match the number of X columns")
        if self.n_permutations <= 0:
            raise ValueError("n_permutations must be a positive integer")
        if self.task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
        model_task = getattr(self.model, "task", None)
        if model_task in {"classification", "regression"} and model_task != self.task:
            raise ValueError(f"task={self.task!r} conflicts with the model's {model_task} task")
        if self.random_state < 0 or self.random_state > 2**32 - 1:
            raise ValueError("random_state must be between 0 and 2**32 - 1")
        if self.background_data.ndim != 2:
            raise ValueError("background_data must be a 2D array")
        if self.background_data.shape[0] == 0:
            raise ValueError("background_data must contain at least one row")
        if self.background_data.shape[1] != self.X.shape[1]:
            raise ValueError("background_data must have the same columns as X")

    def _resolve_class_labels(self) -> Optional[np.ndarray]:
        """Resolve probability-column labels from model metadata when available."""
        raw_model = getattr(self.model, "model", self.model)
        classes = getattr(raw_model, "classes_", None)
        if classes is None:
            classes = getattr(self.model, "classes_", None)
        if classes is None:
            return None
        if isinstance(classes, (list, tuple)):
            raise ValueError("SAGE does not support multi-output classification")
        labels = np.asarray(classes)
        if labels.ndim != 1 or labels.size == 0:
            raise ValueError("model.classes_ must be a non-empty one-dimensional array")
        if np.unique(labels).size != labels.size:
            raise ValueError("model.classes_ must contain unique labels")
        return labels

    def _labels_for_probability_width(self, width: int) -> np.ndarray:
        """Return labels ordered like prediction columns, validating the width."""
        if self.class_labels is not None:
            if len(self.class_labels) != width:
                raise ValueError(
                    f"classification model returned {width} columns but model.classes_ "
                    f"contains {len(self.class_labels)} labels"
                )
            return self.class_labels

        default_labels = np.arange(width)
        targets = np.asarray(self.y)
        if np.all(np.isin(targets, default_labels)):
            return default_labels
        raise ValueError(
            "classification probability columns cannot be mapped to non-index labels; "
            "the model must expose classes_"
        )

    def _zero_one_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Return zero-one loss after mapping output columns to model class labels."""
        predictions = as_real_array(
            y_pred,
            name="classification predictions",
            require_finite=True,
        )
        targets = np.asarray(y_true)
        if predictions.ndim != 2 or predictions.shape[0] != targets.shape[0]:
            raise ValueError("classification predictions must have shape (n_samples, n_outputs)")
        if predictions.shape[1] == 0:
            raise ValueError("classification predictions must contain output columns")
        try:
            predictions = as_real_array(
                predictions,
                name="classification predictions",
                dtype=float,
                require_finite=True,
            )
        except ValueError as exc:
            raise ValueError("classification predictions must be finite real numbers") from exc

        if predictions.shape[1] == 1:
            class_labels = self._labels_for_probability_width(2)
            positive = predictions[:, 0]
            if np.any((positive < 0.0) | (positive > 1.0)):
                raise ValueError("one-column classification outputs must be probabilities")
            labels = class_labels[(positive >= 0.5).astype(int)]
        else:
            class_labels = self._labels_for_probability_width(predictions.shape[1])
            labels = class_labels[np.argmax(predictions, axis=1)]
        return float(1.0 - accuracy_score(targets, labels))

    @staticmethod
    def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Return MSE while accepting the standard ``(n, 1)`` adapter shape."""
        targets = as_real_array(
            y_true,
            name="regression targets",
            dtype=float,
            require_finite=True,
        )
        predictions = as_real_array(
            y_pred,
            name="regression predictions",
            dtype=float,
            require_finite=True,
        )
        if targets.ndim == 1 and predictions.ndim == 2 and predictions.shape[1] == 1:
            predictions = predictions[:, 0]
        return float(mean_squared_error(targets, predictions))

    def _loss_from_predictions(self, predictions: np.ndarray) -> float:
        """Evaluate and scalarize the configured loss."""
        loss = self.loss_fn(self.y, predictions)
        loss_array = as_real_array(
            loss,
            name="loss_fn result",
            dtype=float,
            require_finite=True,
        )
        if loss_array.size == 0 or not np.all(np.isfinite(loss_array)):
            raise ValueError("loss_fn must return finite values")
        return float(np.mean(loss_array))

    def _predict_restricted(self, feature_mask: np.ndarray) -> np.ndarray:
        """Evaluate ``E[f(X) | X_S]`` using the empirical marginal imputer.

        The expectation is taken over model outputs, not losses. This matches
        the marginal imputer in the official SAGE implementation.
        """
        mask = np.asarray(feature_mask, dtype=bool)
        if mask.shape != (self.X.shape[1],):
            raise ValueError("feature_mask must contain one value per feature")

        if np.all(mask):
            return np.asarray(self.model.predict(self.X))

        n_samples = self.X.shape[0]
        n_background = self.background_data.shape[0]

        if not np.any(mask):
            background_predictions = np.asarray(self.model.predict(self.background_data))
            mean_prediction = np.mean(background_predictions, axis=0, keepdims=True)
            return np.repeat(mean_prediction, n_samples, axis=0)

        completed = np.repeat(self.X, n_background, axis=0)
        tiled_background = np.tile(self.background_data, (n_samples, 1))
        completed[:, ~mask] = tiled_background[:, ~mask]

        predictions = np.asarray(self.model.predict(completed))
        expected_rows = n_samples * n_background
        if predictions.ndim == 0 or predictions.shape[0] != expected_rows:
            raise ValueError("model.predict must return one prediction row per input row")

        output_shape = predictions.shape[1:]
        reshaped = predictions.reshape(n_samples, n_background, *output_shape)
        return np.mean(reshaped, axis=1)

    def _compute_coalition_loss(self, feature_mask: np.ndarray) -> float:
        """Compute the SAGE loss for one feature coalition."""
        return self._loss_from_predictions(self._predict_restricted(feature_mask))

    def _marginal_contribution(
        self, feature_idx: int, feature_order: List[int], position: int
    ) -> float:
        """
        Compute marginal contribution of a feature given a feature ordering.

        The marginal contribution is the change in loss when adding the feature
        to the set of features that come before it in the ordering.
        """
        mask_without = np.zeros(self.X.shape[1], dtype=bool)
        mask_without[feature_order[:position]] = True
        mask_with = mask_without.copy()
        mask_with[feature_idx] = True

        loss_without = self._compute_coalition_loss(mask_without)
        loss_with = self._compute_coalition_loss(mask_with)

        return loss_without - loss_with

    def explain(self, **kwargs) -> Explanation:
        """
        Compute SAGE values for all features.

        Uses permutation sampling to approximate the Shapley values.

        Returns:
            Explanation object with global feature importance (SAGE values)
        """
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

        n_features = self.X.shape[1]
        sage_values = np.zeros(n_features)

        baseline_mask = np.zeros(n_features, dtype=bool)
        full_mask = np.ones(n_features, dtype=bool)
        baseline_loss = self._compute_coalition_loss(baseline_mask)
        full_loss = self._compute_coalition_loss(full_mask)

        # Reset for reproducible repeated calls.
        rng = np.random.RandomState(self.random_state)

        for _ in range(self.n_permutations):
            order = rng.permutation(n_features)
            mask = np.zeros(n_features, dtype=bool)
            previous_loss = baseline_loss

            for feature_idx in order:
                mask[feature_idx] = True
                current_loss = self._compute_coalition_loss(mask)
                sage_values[feature_idx] += previous_loss - current_loss
                previous_loss = current_loss

        sage_values /= self.n_permutations

        attributions = {fname: float(sage_values[i]) for i, fname in enumerate(self.feature_names)}

        total_value = baseline_loss - full_loss
        efficiency_error = float(np.sum(sage_values) - total_value)

        return Explanation(
            explainer_name="SAGE",
            target_class="global",
            feature_names=self.feature_names,
            explanation_data={
                "feature_attributions": attributions,
                "n_permutations": self.n_permutations,
                "task": self.task,
                "loss_name": self.loss_name,
                "loss_direction": self.loss_direction,
                "loss_is_custom": self.loss_is_custom,
                "imputer": "marginal",
                "background_size": int(self.background_data.shape[0]),
                "baseline_loss": baseline_loss,
                "full_loss": full_loss,
                "total_value": total_value,
                "efficiency_error": efficiency_error,
            },
            metadata={
                "task": self.task,
                "loss_name": self.loss_name,
                "loss_direction": self.loss_direction,
                "loss_is_custom": self.loss_is_custom,
                "imputer": "marginal",
                "n_permutations": self.n_permutations,
                "random_state": self.random_state,
            },
        )
