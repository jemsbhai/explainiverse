"""Model-agnostic KernelSHAP explanations.

This wrapper delegates the weighted-linear-regression estimator to the
official :mod:`shap` package and makes the output/target contract explicit.

Reference:
    Lundberg, S.M. & Lee, S.I. (2017). A Unified Approach to Interpreting
    Model Predictions. NeurIPS 2017. https://arxiv.org/abs/1705.07874
"""

from __future__ import annotations

import re
import threading
from numbers import Integral, Real
from typing import List, Optional

import numpy as np

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation
from explainiverse.explainers._validation import as_real_array, validate_name_sequence

# Lazy import check -- importing SHAP is relatively expensive.
_SHAP_AVAILABLE = None

# KernelExplainer currently samples with NumPy's process-global RNG and also
# mutates state on the explainer object while solving. Serializing calls lets a
# seeded wrapper be repeatable without changing the caller's RNG stream.
_KERNEL_SHAP_LOCK = threading.RLock()


def _check_shap_available():
    """Raise a useful error if the optional SHAP dependency is unavailable."""
    global _SHAP_AVAILABLE

    if _SHAP_AVAILABLE is None:
        try:
            import shap  # noqa: F401

            _SHAP_AVAILABLE = True
        except ImportError:
            _SHAP_AVAILABLE = False

    if not _SHAP_AVAILABLE:
        raise ImportError("SHAP is required for ShapExplainer. Install it with: pip install shap")


def _python_scalar(value):
    """Return a metadata-friendly Python scalar when NumPy supplied one."""
    return value.item() if isinstance(value, np.generic) else value


class ShapExplainer(BaseExplainer):
    """Model-agnostic KernelSHAP wrapper.

    The object explains the numerical outputs returned by the supplied model.
    Classifier adapters are expected to return one column per class. A binary
    classifier returning only positive-class probabilities is normalized to
    ``[1 - p, p]`` before SHAP sees it.

    ``l1_reg`` defaults to ``0.0`` deliberately. SHAP 0.47 changed its own
    default to ``"num_features(10)"``, which performs feature selection rather
    than estimating a value for every feature. Callers can opt into that
    sparsification explicitly.
    """

    def __init__(
        self,
        model,
        background_data: np.ndarray,
        feature_names: List[str],
        class_names: Optional[List[str]] = None,
        *,
        task: Optional[str] = None,
        link: str = "identity",
        nsamples: str | int = "auto",
        l1_reg: str | float | bool = 0.0,
        random_state: Optional[int] = 42,
    ):
        """Initialize an official SHAP ``KernelExplainer``.

        Args:
            model: Adapter or model exposing batched predictions. Raw
                classifiers with ``predict_proba`` are supported directly.
            background_data: Non-empty 2D reference distribution.
            feature_names: Unique names, one per input column.
            class_names: Optional display names ordered like model outputs.
            task: ``"classification"`` or ``"regression"``. By default this
                is inferred from adapter/model semantics.
            link: KernelSHAP link function, ``"identity"`` or ``"logit"``.
            nsamples: Default SHAP coalition-sample budget.
            l1_reg: Feature-selection rule/strength. The default ``0.0`` keeps
                all features in the weighted regression.
            random_state: Seed for repeatable coalition sampling. ``None`` uses
                the ambient NumPy RNG.
        """
        _check_shap_available()
        import shap as shap_module

        super().__init__(model)

        self.background_data = np.asarray(background_data)
        if self.background_data.ndim != 2:
            raise ValueError("background_data must be a two-dimensional array")
        if self.background_data.shape[0] == 0:
            raise ValueError("background_data must contain at least one row")
        if self.background_data.shape[1] == 0:
            raise ValueError("background_data must contain at least one feature")

        validated_features = validate_name_sequence(feature_names, name="feature_names")
        assert validated_features is not None
        self.feature_names = validated_features
        if len(self.feature_names) != self.background_data.shape[1]:
            raise ValueError("feature_names length must match the background feature count")
        self.class_names = validate_name_sequence(
            class_names,
            name="class_names",
            allow_none=True,
        )

        self.raw_model = getattr(model, "model", model)
        n_model_features = getattr(self.raw_model, "n_features_in_", None)
        if n_model_features is not None and int(n_model_features) != self.background_data.shape[1]:
            raise ValueError("background_data feature count must match model.n_features_in_")

        self.task = self._resolve_task(task)
        if link not in {"identity", "logit"}:
            raise ValueError("link must be 'identity' or 'logit'")
        self.link = link
        self.nsamples = self._validate_nsamples(nsamples)
        self.l1_reg = self._validate_l1_reg(l1_reg)
        self.random_state = self._validate_random_state(random_state)

        # Validate the entire background because it also defines the output
        # space represented by the resulting expected value.
        probe = self._predict_for_shap(self.background_data)
        self.n_outputs = probe.shape[1]
        if self._task_semantics_ambiguous and self.n_outputs > 1:
            raise ValueError(
                "task is required for a model with ambiguous multi-output predictions; "
                "class_names are display metadata and do not establish regression semantics"
            )
        if self.class_names is not None and len(self.class_names) != self.n_outputs:
            raise ValueError("class_names length must match the normalized model output count")
        if self.task == "classification" and self.n_outputs < 2:
            raise ValueError(
                "classification predictions must expose at least two normalized " "output columns"
            )

        self.output_space = self._infer_output_space(probe)
        if link == "logit" and self.output_space != "probability":
            raise ValueError("link='logit' requires model outputs that are probabilities")

        self.explainer = shap_module.KernelExplainer(
            self._predict_for_shap,
            self.background_data,
            feature_names=self.feature_names,
            link=link,
        )

    def _resolve_task(self, task: Optional[str]) -> str:
        """Infer the task from estimator semantics, not display metadata."""
        if task not in {None, "classification", "regression"}:
            raise ValueError("task must be 'classification', 'regression', or None")

        adapter_task = getattr(self.model, "task", None)
        if adapter_task not in {None, "classification", "regression"}:
            raise ValueError("model.task must be 'classification' or 'regression'")

        estimator_task = None
        estimator_type = getattr(self.raw_model, "_estimator_type", None)
        if estimator_type == "classifier":
            estimator_task = "classification"
        elif estimator_type == "regressor":
            estimator_task = "regression"
        elif hasattr(self.raw_model, "classes_") or hasattr(self.raw_model, "predict_proba"):
            estimator_task = "classification"

        semantic_task = adapter_task or estimator_task
        if (
            adapter_task is not None
            and estimator_task is not None
            and adapter_task != estimator_task
        ):
            raise ValueError("model.task conflicts with the wrapped estimator semantics")
        if task is not None and semantic_task is not None and task != semantic_task:
            raise ValueError(f"task={task!r} conflicts with the model's {semantic_task} task")

        self._task_semantics_ambiguous = semantic_task is None and task is None
        return task or semantic_task or "regression"

    @staticmethod
    def _validate_nsamples(nsamples: str | int) -> str | int:
        if nsamples == "auto":
            return nsamples
        if not isinstance(nsamples, Integral) or isinstance(nsamples, bool) or int(nsamples) <= 0:
            raise ValueError("nsamples must be 'auto' or a positive integer")
        return int(nsamples)

    @staticmethod
    def _validate_l1_reg(l1_reg: str | float | bool) -> str | float | bool:
        if l1_reg is False:
            return False
        if isinstance(l1_reg, str):
            if l1_reg in {"auto", "aic", "bic"}:
                return l1_reg
            match = re.fullmatch(r"num_features\(([1-9][0-9]*)\)", l1_reg)
            if match:
                return l1_reg
            raise ValueError(
                "l1_reg must be 0/False, a non-negative float, 'auto', "
                "'aic', 'bic', or 'num_features(k)'"
            )
        if isinstance(l1_reg, bool) or not isinstance(l1_reg, Real):
            raise TypeError("l1_reg must be a supported string, float, or False")
        value = float(l1_reg)
        if not np.isfinite(value) or value < 0:
            raise ValueError("l1_reg numeric values must be finite and non-negative")
        return value

    @staticmethod
    def _validate_random_state(random_state: Optional[int]) -> Optional[int]:
        if random_state is None:
            return None
        if not isinstance(random_state, Integral) or isinstance(random_state, bool):
            raise TypeError("random_state must be a non-negative integer or None")
        value = int(random_state)
        if value < 0 or value > 2**32 - 1:
            raise ValueError("random_state must be between 0 and 2**32 - 1")
        return value

    def _expected_class_count(self) -> Optional[int]:
        classes = getattr(self.raw_model, "classes_", None)
        if classes is not None:
            if isinstance(classes, (list, tuple)):
                raise ValueError("KernelSHAP does not support multi-output classification")
            classes = np.asarray(classes)
            if classes.ndim != 1 or classes.size == 0:
                raise ValueError("model.classes_ must be a non-empty one-dimensional array")
            return len(classes)
        if self.class_names is not None:
            return len(self.class_names)
        return None

    def _predict_for_shap(self, X: np.ndarray) -> np.ndarray:
        """Return a validated ``(samples, outputs)`` numerical matrix."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("model inputs must be a two-dimensional array")
        if X.shape[1] != len(self.feature_names):
            raise ValueError("model input feature count does not match feature_names")

        used_label_predictions = False
        if self.task == "classification" and hasattr(self.model, "predict_proba"):
            predictions = self.model.predict_proba(X)
        elif (
            self.task == "classification"
            and self.model is self.raw_model
            and hasattr(self.raw_model, "predict_proba")
        ):
            predictions = self.raw_model.predict_proba(X)
        else:
            if not hasattr(self.model, "predict"):
                raise TypeError("model must expose a batched predict method")
            predictions = self.model.predict(X)
            used_label_predictions = (
                self.task == "classification"
                and self.model is self.raw_model
                and hasattr(self.raw_model, "classes_")
                and not hasattr(self.model, "predict_proba")
            )

        predictions = np.asarray(predictions)
        if predictions.ndim == 0:
            raise ValueError("model predictions must retain a sample dimension")

        if self.task == "classification" and used_label_predictions:
            labels = predictions.reshape(-1)
            if labels.shape[0] != X.shape[0]:
                raise ValueError("model returned the wrong number of predictions")
            classes = np.asarray(self.raw_model.classes_)
            matches = labels[:, None] == classes[None, :]
            if not np.all(matches.sum(axis=1) == 1):
                raise ValueError("model predicted labels not found in model.classes_")
            predictions = matches.astype(float)
        elif predictions.ndim == 1:
            if predictions.shape[0] != X.shape[0]:
                raise ValueError("model returned the wrong number of predictions")
            predictions = predictions.reshape(-1, 1)
        elif predictions.ndim == 2:
            if predictions.shape[0] != X.shape[0]:
                raise ValueError("model returned the wrong number of predictions")
        else:
            raise ValueError("model predictions must be one- or two-dimensional")

        try:
            predictions = as_real_array(
                predictions,
                name="model outputs",
                dtype=float,
                require_finite=True,
            )
        except ValueError as exc:
            raise ValueError("model outputs must be numerical") from exc

        if self.task == "classification" and predictions.shape[1] == 1:
            expected_classes = self._expected_class_count()
            if expected_classes not in {None, 2}:
                raise ValueError("one-column classification outputs require exactly two classes")
            positive = predictions[:, 0]
            if np.any((positive < 0.0) | (positive > 1.0)):
                raise ValueError("one-column binary outputs must be positive-class probabilities")
            predictions = np.column_stack((1.0 - positive, positive))

        expected_classes = self._expected_class_count()
        if (
            self.task == "classification"
            and expected_classes is not None
            and predictions.shape[1] != expected_classes
        ):
            raise ValueError("classification output count does not match model/classes metadata")
        return predictions

    def _infer_output_space(self, predictions: np.ndarray) -> str:
        if self.task != "classification":
            return "model_output"
        is_probability = bool(np.all((predictions >= 0.0) & (predictions <= 1.0)))
        is_probability = is_probability and bool(
            np.allclose(predictions.sum(axis=1), 1.0, rtol=1e-7, atol=1e-9)
        )
        return "probability" if is_probability else "model_output"

    def _resolve_output_index(
        self, instance: np.ndarray, target_class: Optional[int]
    ) -> tuple[int, np.ndarray]:
        predictions = self._predict_for_shap(instance)
        if target_class is not None:
            if not isinstance(target_class, Integral) or isinstance(target_class, bool):
                raise TypeError("target_class must be an integer output index")
            output_index = int(target_class)
        elif self.task == "classification":
            output_index = int(np.argmax(predictions[0]))
        elif self.n_outputs == 1:
            output_index = 0
        else:
            raise ValueError("target_class is required for multi-output regression")

        if not 0 <= output_index < self.n_outputs:
            raise ValueError("target_class is outside the model output range")
        return output_index, predictions

    def _label_for_output(self, output_index: int) -> tuple[str, object | None]:
        model_label = None
        classes = getattr(self.raw_model, "classes_", None)
        if self.task == "classification" and classes is not None:
            model_label = _python_scalar(np.asarray(classes)[output_index])

        if self.class_names is not None:
            label_name = str(self.class_names[output_index])
        elif model_label is not None:
            label_name = str(model_label)
        elif self.task == "regression" and self.n_outputs == 1:
            label_name = "output"
        else:
            label_name = f"output_{output_index}"
        return label_name, model_label

    def _extract_output_values(self, shap_values, output_index: int) -> np.ndarray:
        """Extract one output without flattening features and outputs together."""
        if isinstance(shap_values, list):
            if output_index >= len(shap_values):
                raise ValueError("KernelSHAP output does not cover target_class")
            values = np.asarray(shap_values[output_index])
            if values.ndim == 2:
                values = values[0]
        else:
            values_array = np.asarray(shap_values)
            if values_array.ndim == 3:
                if output_index >= values_array.shape[2]:
                    raise ValueError("KernelSHAP output does not cover target_class")
                values = values_array[0, :, output_index]
            elif values_array.ndim == 2:
                if self.n_outputs != 1:
                    raise ValueError(
                        "KernelSHAP returned a single-output array for a " "multi-output model"
                    )
                values = values_array[0]
            elif values_array.ndim == 1:
                if self.n_outputs != 1:
                    raise ValueError(
                        "KernelSHAP returned a single-output array for a " "multi-output model"
                    )
                values = values_array
            else:
                raise ValueError(f"Unexpected KernelSHAP values shape: {values_array.shape}")

        values = as_real_array(
            values,
            name="KernelSHAP values",
            dtype=float,
            require_finite=True,
        ).reshape(-1)
        if values.size != len(self.feature_names):
            raise ValueError("KernelSHAP value count does not match the input feature count")
        return values

    def _expected_value(self, output_index: int) -> float:
        expected = as_real_array(
            self.explainer.expected_value,
            name="KernelSHAP expected value",
            require_finite=True,
        )
        if expected.ndim == 0:
            if output_index != 0:
                raise ValueError("KernelSHAP expected value does not cover target_class")
            return float(expected)
        expected = expected.reshape(-1)
        if output_index >= expected.size:
            raise ValueError("KernelSHAP expected value does not cover target_class")
        return float(expected[output_index])

    def _compute_shap_values(self, instance: np.ndarray):
        kwargs = {
            "nsamples": self.nsamples,
            "l1_reg": self.l1_reg,
            "silent": True,
        }
        with _KERNEL_SHAP_LOCK:
            if self.random_state is None:
                return self.explainer.shap_values(instance, **kwargs)

            state = np.random.get_state()
            try:
                np.random.seed(self.random_state)
                return self.explainer.shap_values(instance, **kwargs)
            finally:
                np.random.set_state(state)

    def explain(
        self,
        instance: np.ndarray,
        top_labels: int = 1,
        target_class: Optional[int] = None,
    ) -> Explanation:
        """Explain exactly one instance and one model-output column.

        ``target_class`` is an output-column index. If omitted, classifiers use
        the largest original-instance output and single-output regressors use
        output zero. ``top_labels`` remains for API compatibility but must be
        one because this method returns exactly one :class:`Explanation`.
        """
        if (
            not isinstance(top_labels, Integral)
            or isinstance(top_labels, bool)
            or int(top_labels) != 1
        ):
            raise ValueError(
                "KernelSHAP explain returns one output; top_labels must be 1. "
                "Use target_class to select a model output."
            )

        instance = np.asarray(instance)
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)
        elif instance.ndim != 2 or instance.shape[0] != 1:
            raise ValueError("explain accepts exactly one instance")
        if instance.shape[1] != len(self.feature_names):
            raise ValueError("instance feature count does not match feature_names")

        output_index, predictions = self._resolve_output_index(instance, target_class)
        shap_values = self._compute_shap_values(instance)
        output_values = self._extract_output_values(shap_values, output_index)
        expected_value = self._expected_value(output_index)
        label_name, model_label = self._label_for_output(output_index)

        attributions = {
            name: float(output_values[index]) for index, name in enumerate(self.feature_names)
        }
        model_value = float(self.explainer.link.f(float(predictions[0, output_index])))
        explained_value = expected_value + float(output_values.sum())
        explanation_space = "log_odds" if self.link == "logit" else self.output_space

        return Explanation(
            explainer_name="SHAP",
            target_class=label_name,
            explanation_data={
                "feature_attributions": attributions,
                "shap_values_raw": output_values.tolist(),
                "expected_value": expected_value,
            },
            feature_names=self.feature_names,
            metadata={
                "method": "KernelSHAP",
                "output_index": output_index,
                "model_class_label": model_label,
                "output_space": explanation_space,
                "model_output_space": self.output_space,
                "link": self.link,
                "nsamples": self.nsamples,
                "l1_reg": self.l1_reg,
                "random_state": self.random_state,
                "background_size": int(self.background_data.shape[0]),
                "model_reference_value": model_value,
                "explained_value": explained_value,
                "additivity_residual": explained_value - model_value,
            },
        )

    def explain_batch(
        self,
        X: np.ndarray,
        top_labels: int = 1,
        target_class: Optional[int] = None,
    ) -> List[Explanation]:
        """Explain each row independently under the same output contract."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        elif X.ndim != 2:
            raise ValueError("X must be a one- or two-dimensional array")
        if X.shape[0] == 0:
            return []
        if X.shape[1] != len(self.feature_names):
            raise ValueError("X feature count does not match feature_names")

        return [
            self.explain(
                X[index],
                top_labels=top_labels,
                target_class=target_class,
            )
            for index in range(X.shape[0])
        ]
