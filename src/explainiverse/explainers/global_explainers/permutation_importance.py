"""
Permutation feature importance.

The implementation follows the score-drop definition used by scikit-learn:
the baseline score is compared with scores after independently permuting each
feature.  Classification and regression are selected from an explicit task
contract, never from the dimensionality of ``model.predict``.

Reference:
    Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
"""

from numbers import Integral
from typing import Callable, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, r2_score

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation
from explainiverse.explainers._validation import as_real_array, validate_name_sequence

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


class PermutationImportanceExplainer(BaseExplainer):
    """Measure feature importance as the decrease in predictive score."""

    def __init__(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        n_repeats: int = 10,
        scoring_fn: Optional[Callable] = None,
        random_state: int = 42,
        task: Optional[str] = None,
    ):
        """
        Initialize permutation importance.

        Args:
            model: Adapter with ``predict`` and a declared ``task`` attribute.
            X: Evaluation features of shape ``(n_samples, n_features)``.
            y: Evaluation targets.
            feature_names: Unique feature names in column order.
            n_repeats: Positive number of permutations per feature.
            scoring_fn: Optional ``(y_true, model_prediction) -> score`` callable
                where higher is better. For backward compatibility a custom
                scorer receives the unmodified result of ``model.predict``.
            random_state: Seed for reproducible permutations.
            task: Explicit task override. It must agree with ``model.task`` if
                the model declares one.

        The default score is accuracy for classification and R-squared for
        regression. These choices match the conventional default estimator
        scores used by scikit-learn permutation importance.
        """
        super().__init__(model)
        self.X: np.ndarray = np.asarray(X)
        self.y: np.ndarray = np.asarray(y)
        validated_names = validate_name_sequence(feature_names, name="feature_names")
        assert validated_names is not None
        self.feature_names: List[str] = validated_names
        self.task: str = _resolve_task(model, task)

        if self.X.ndim != 2 or self.X.shape[0] == 0 or self.X.shape[1] == 0:
            raise ValueError("X must be a non-empty 2D array")
        if self.y.ndim == 0 or self.y.shape[0] != self.X.shape[0]:
            raise ValueError("y must contain one target row per row of X")
        if len(self.feature_names) != self.X.shape[1]:
            raise ValueError("feature_names length must equal the number of columns in X")
        if not isinstance(n_repeats, Integral) or isinstance(n_repeats, bool):
            raise TypeError("n_repeats must be a positive integer")
        if n_repeats < 1:
            raise ValueError("n_repeats must be at least 1")
        if not isinstance(random_state, Integral) or isinstance(random_state, bool):
            raise TypeError("random_state must be an integer")
        if scoring_fn is not None and not callable(scoring_fn):
            raise TypeError("scoring_fn must be callable or None")

        self.n_repeats: int = int(n_repeats)
        self.scoring_fn: Optional[Callable] = scoring_fn
        self.random_state: int = int(random_state)
        self.scoring_name: str = (
            getattr(scoring_fn, "__name__", scoring_fn.__class__.__name__)
            if scoring_fn is not None
            else ("accuracy" if self.task == "classification" else "r2")
        )
        self.score_input: str = (
            "model_prediction"
            if scoring_fn is not None
            else ("class_labels" if self.task == "classification" else "regression_response")
        )

    def _class_values(self, n_classes: int) -> np.ndarray:
        """Return the labels corresponding to prediction columns."""
        candidates = (
            getattr(self.model, "classes_", None),
            getattr(getattr(self.model, "model", None), "classes_", None),
        )
        classes = next((value for value in candidates if value is not None), None)
        if classes is None:
            classes = np.unique(self.y)
        classes = np.asarray(classes)
        if classes.ndim != 1 or len(classes) != n_classes:
            raise ValueError(
                f"Cannot map {n_classes} prediction columns to class labels. "
                "Expose classes_ on the model/adapter or ensure y contains all classes."
            )
        return classes

    def _classification_labels(self, predictions: np.ndarray) -> np.ndarray:
        """Convert adapter prediction scores to labels without label-index guesses."""
        predictions = as_real_array(
            predictions,
            name="classification predictions",
            require_finite=True,
        )
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        if predictions.ndim != 2 or predictions.shape[0] != self.X.shape[0]:
            raise ValueError(
                "Classification predictions must have shape "
                "(n_samples,), (n_samples, 1), or (n_samples, n_classes)"
            )

        if predictions.shape[1] == 1:
            probabilities = as_real_array(
                predictions[:, 0],
                name="one-column classification output",
                dtype=float,
                require_finite=True,
            )
            if np.any((probabilities < 0.0) | (probabilities > 1.0)):
                raise ValueError(
                    "A one-column classification output must contain P(class 1) " "values in [0, 1]"
                )
            classes = self._class_values(2)
            return classes[(probabilities >= 0.5).astype(int)]

        classes = self._class_values(predictions.shape[1])
        return classes[np.argmax(predictions, axis=1)]

    def _regression_values(self, predictions: np.ndarray) -> np.ndarray:
        """Match standardized regression outputs to the target array shape."""
        predictions = as_real_array(
            predictions,
            name="regression predictions",
            require_finite=True,
        )
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        if predictions.ndim != 2 or predictions.shape[0] != self.X.shape[0]:
            raise ValueError(
                "Regression predictions must have shape (n_samples,) or " "(n_samples, n_outputs)"
            )

        if self.y.ndim == 1:
            if predictions.shape[1] != 1:
                raise ValueError("One-dimensional y is incompatible with multi-output predictions")
            return predictions[:, 0]
        if self.y.ndim == 2 and predictions.shape == self.y.shape:
            return predictions
        raise ValueError(
            f"Regression prediction shape {predictions.shape} does not match "
            f"target shape {self.y.shape}"
        )

    def _score_predictions(self, predictions: np.ndarray) -> float:
        if self.scoring_fn is not None:
            score_input = np.asarray(predictions)
            scorer = self.scoring_fn
        elif self.task == "classification":
            score_input = self._classification_labels(predictions)
            scorer = accuracy_score
        else:
            score_input = self._regression_values(predictions)
            scorer = r2_score

        raw_score = as_real_array(
            scorer(self.y, score_input),
            name="scoring function result",
            dtype=float,
            require_finite=True,
        )
        if raw_score.size != 1:
            raise ValueError("The scoring function must return exactly one score")
        score = float(raw_score.reshape(-1)[0])
        if not np.isfinite(score):
            raise ValueError(f"The scoring function returned a non-finite score: {score}")
        return score

    def _compute_baseline_score(self) -> float:
        """Compute model performance on unperturbed data."""
        return self._score_predictions(self.model.predict(self.X))

    def explain(self, **kwargs) -> Explanation:  # type: ignore[override]
        """Compute reproducible permutation feature importances."""
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

        baseline_score = self._compute_baseline_score()

        # Match scikit-learn's deterministic permutation stream: generate one
        # seed and initialize the same stream independently for each feature.
        seed_rng = np.random.RandomState(self.random_state)
        permutation_seed = seed_rng.randint(np.iinfo(np.int32).max + 1)

        importances: dict[str, float] = {}
        stds: dict[str, float] = {}
        raw_importances: dict[str, List[float]] = {}

        for idx, feature_name in enumerate(self.feature_names):
            feature_rng = np.random.RandomState(permutation_seed)
            row_indices = np.arange(self.X.shape[0])
            X_permuted = self.X.copy()
            score_drops: List[float] = []

            for _ in range(self.n_repeats):
                feature_rng.shuffle(row_indices)
                X_permuted[:, idx] = X_permuted[row_indices, idx]
                score = self._score_predictions(self.model.predict(X_permuted))
                score_drops.append(baseline_score - score)

            score_drops_array = np.asarray(score_drops, dtype=float)
            importances[feature_name] = float(np.mean(score_drops_array))
            stds[feature_name] = float(np.std(score_drops_array))
            raw_importances[feature_name] = score_drops_array.tolist()

        return Explanation(
            explainer_name="PermutationImportance",
            target_class="global",
            feature_names=self.feature_names,
            explanation_data={
                "feature_attributions": importances,
                "std": stds,
                "importances": raw_importances,
                "baseline_score": baseline_score,
                "task": self.task,
                "scoring": self.scoring_name,
                "score_direction": "higher_is_better",
                "score_input": self.score_input,
            },
            metadata={
                "task": self.task,
                "scoring": self.scoring_name,
                "score_direction": "higher_is_better",
                "score_input": self.score_input,
                "prediction_output": "model.predict",
                "n_repeats": self.n_repeats,
                "random_state": self.random_state,
            },
        )
