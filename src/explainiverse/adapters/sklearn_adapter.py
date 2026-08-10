# src/explainiverse/adapters/sklearn_adapter.py

import numpy as np
from sklearn.base import is_classifier, is_regressor

from .base_adapter import BaseModelAdapter


class SklearnAdapter(BaseModelAdapter):
    """
    Adapter for scikit-learn classifiers and regressors.

    ``class_names`` is presentation metadata only; it never determines the
    estimator task. By default the task is inferred from scikit-learn estimator
    tags, with ``classes_``/``predict_proba`` used for compatible third-party
    estimators. Pass ``task`` explicitly when wrapping an estimator that does
    not expose any of those signals.
    """

    def __init__(self, model, feature_names=None, class_names=None, task=None):
        if task not in {None, "classification", "regression"}:
            raise ValueError(f"task must be 'classification', 'regression', or None; got {task!r}")

        super().__init__(model, feature_names)
        self.class_names = list(class_names) if class_names is not None else None
        if self.class_names is not None and len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class_names must be unique")

        inferred_task = self._infer_task(model)
        if task is None:
            if inferred_task is None:
                raise ValueError(
                    "Could not infer estimator task. Pass task='classification' or "
                    "task='regression' explicitly."
                )
            self.task = inferred_task
        else:
            if inferred_task is not None and inferred_task != task:
                raise ValueError(
                    f"task={task!r} conflicts with the estimator's {inferred_task} semantics"
                )
            self.task = task

        if self.task == "classification":
            self._validate_class_metadata()

    @staticmethod
    def _infer_task(model):
        """Infer task from estimator semantics, never from display metadata."""
        if is_classifier(model):
            return "classification"
        if is_regressor(model):
            return "regression"
        if hasattr(model, "classes_") or hasattr(model, "predict_proba"):
            return "classification"
        return None

    def _classification_classes(self):
        """Return a one-dimensional class vector or reject multi-output classifiers."""
        classes = getattr(self.model, "classes_", None)
        if classes is None:
            return None
        if isinstance(classes, (list, tuple)):
            raise ValueError(
                "SklearnAdapter does not support multi-output classification; "
                "wrap one output at a time"
            )
        classes = np.asarray(classes)
        if classes.ndim != 1 or classes.size == 0:
            raise ValueError(
                "model.classes_ must be a non-empty one-dimensional array; "
                "multi-output classification is not supported"
            )
        return classes

    def _validate_class_metadata(self, output_width=None):
        classes = self._classification_classes()
        if classes is not None and self.class_names is not None:
            if len(self.class_names) != len(classes):
                raise ValueError(
                    f"class_names has {len(self.class_names)} entries but the model "
                    f"has {len(classes)} classes"
                )
        if output_width is not None:
            expected = len(classes) if classes is not None else None
            if expected is None and self.class_names is not None:
                expected = len(self.class_names)
            if expected is not None and int(output_width) != expected:
                raise ValueError(
                    f"model returned {output_width} probability columns but class "
                    f"metadata describes {expected} classes"
                )
        return classes

    def _labels_to_indicator(self, predictions: np.ndarray) -> np.ndarray:
        """Convert predicted labels to columns ordered by ``model.classes_``."""
        classes = self._classification_classes()
        if classes is None:
            raise ValueError(
                "A classifier without predict_proba must expose classes_ so "
                "predicted labels can be mapped to output columns."
            )
        self._validate_class_metadata(output_width=len(classes))

        predictions = np.asarray(predictions)
        if predictions.ndim != 1:
            raise ValueError(
                "SklearnAdapter does not support multi-output classification; "
                "model.predict must return one label per sample"
            )
        matches = predictions[:, None] == classes[None, :]
        if not np.all(matches.sum(axis=1) == 1):
            unknown = predictions[matches.sum(axis=1) != 1]
            raise ValueError(f"Model predicted labels not present in classes_: {unknown.tolist()}")
        return matches.astype(float)

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Return classification probabilities/indicators or regression outputs.

        Args:
            data: A 2D numpy array of inputs.

        Returns:
            Classification: ``(n_samples, n_classes)`` ordered by
            ``model.classes_``. Regression: ``(n_samples, n_outputs)``.
        """
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError("data must be a two-dimensional array")
        n_samples = data.shape[0]

        if self.task == "classification":
            if hasattr(self.model, "predict_proba"):
                raw_probabilities = self.model.predict_proba(data)
                if isinstance(raw_probabilities, (list, tuple)):
                    raise ValueError(
                        "SklearnAdapter does not support multi-output classification; "
                        "predict_proba returned one matrix per output"
                    )
                probabilities = np.asarray(raw_probabilities)
                if probabilities.ndim != 2:
                    raise ValueError(
                        "classification predict_proba must return a two-dimensional "
                        "(n_samples, n_classes) array"
                    )
                if probabilities.shape[0] != n_samples or probabilities.shape[1] == 0:
                    raise ValueError("predict_proba returned an invalid output shape")
                self._validate_class_metadata(output_width=probabilities.shape[1])
                try:
                    probabilities = probabilities.astype(float, copy=False)
                except (TypeError, ValueError) as exc:
                    raise ValueError("predict_proba outputs must be numerical") from exc
                if not np.all(np.isfinite(probabilities)):
                    raise ValueError("predict_proba returned non-finite values")
                if np.any(probabilities < -1e-8) or np.any(probabilities > 1.0 + 1e-8):
                    raise ValueError("predict_proba values must lie in [0, 1]")
                if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6):
                    raise ValueError("predict_proba rows must sum to 1")
                return probabilities
            predictions = np.asarray(self.model.predict(data))
            if predictions.ndim != 1 or predictions.shape[0] != n_samples:
                raise ValueError("model.predict must return one class label per input row")
            return self._labels_to_indicator(predictions)

        predictions = np.asarray(self.model.predict(data))
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        elif predictions.ndim != 2:
            raise ValueError(
                "regression predict must return a one- or two-dimensional sample-first array"
            )
        if predictions.shape[0] != n_samples or predictions.shape[1] == 0:
            raise ValueError("regression predict returned an invalid output shape")
        try:
            predictions = predictions.astype(float, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("regression predictions must be numerical") from exc
        if not np.all(np.isfinite(predictions)):
            raise ValueError("regression predict returned non-finite values")
        return predictions
