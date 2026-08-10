# src/explainiverse/explainers/attribution/lime_wrapper.py
"""
LIME Explainer - Local Interpretable Model-agnostic Explanations.

This wrapper fits LIME's weighted local linear surrogate to perturbed tabular
samples. Returned coefficients describe that fitted surrogate under the chosen
sampling/configuration; they are not causal feature effects.

Reference:
    Ribeiro, M.T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?":
    Explaining the Predictions of Any Classifier. KDD 2016.
    https://arxiv.org/abs/1602.04938
"""

from numbers import Integral
from typing import List, Optional

import numpy as np

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation
from explainiverse.explainers._validation import (
    as_real_array,
    validate_name_sequence,
    validate_single_tabular_instance,
)

# Lazy import check - don't import lime at module level
_LIME_AVAILABLE = None


def _check_lime_available():
    """Check if LIME is available and raise ImportError if not."""
    global _LIME_AVAILABLE

    if _LIME_AVAILABLE is None:
        try:
            import lime  # noqa: F401

            _LIME_AVAILABLE = True
        except ImportError:
            _LIME_AVAILABLE = False

    if not _LIME_AVAILABLE:
        raise ImportError(
            "LIME is required for LimeExplainer. " "Install it with: pip install lime"
        )


class LimeExplainer(BaseExplainer):
    """
    LIME explainer for local, model-agnostic explanations.

    LIME generates perturbed samples around an instance and fits a weighted
    local linear surrogate. The coefficients characterize that surrogate and
    do not by themselves establish fidelity or causal contribution.

    This implementation wraps the official LIME library for tabular data.

    Attributes:
        model: Model adapter with .predict() method
        feature_names: List of feature names
        class_names: List of class names
        mode: 'classification' or 'regression'
        explainer: The underlying LimeTabularExplainer

    Example:
        >>> from explainiverse.explainers.attribution import LimeExplainer
        >>> explainer = LimeExplainer(
        ...     model=adapter,
        ...     training_data=X_train,
        ...     feature_names=feature_names,
        ...     class_names=class_names
        ... )
        >>> explanation = explainer.explain(X_test[0])
    """

    def __init__(
        self,
        model,
        training_data: np.ndarray,
        feature_names: List[str],
        class_names: List[str],
        mode: Optional[str] = None,
        random_state: Optional[int] = 42,
    ):
        """
        Initialize the LIME explainer.

        Args:
            model: A model adapter (implements .predict()).
            training_data: The data used to initialize LIME (2D np.ndarray).
                          Used to compute statistics for perturbation generation.
            feature_names: List of feature names.
            class_names: List of class names.
            mode: ``'classification'`` or ``'regression'``. If omitted, the
                mode is inferred from adapter/estimator task semantics. Models
                without such semantics must specify it explicitly.
            random_state: Seed forwarded to the official LIME implementation.

        Raises:
            ImportError: If lime package is not installed.
        """
        # Check availability before importing
        _check_lime_available()

        # Import after check passes
        from lime.lime_tabular import LimeTabularExplainer

        super().__init__(model)
        validated_features = validate_name_sequence(feature_names, name="feature_names")
        validated_classes = validate_name_sequence(
            class_names,
            name="class_names",
            allow_empty=True,
        )
        assert validated_features is not None and validated_classes is not None
        self.feature_names = validated_features
        self.class_names = validated_classes
        self.mode = self._resolve_mode(mode)
        self.training_data = np.asarray(training_data)
        self.random_state = random_state

        self._validate_configuration()

        self.explainer = LimeTabularExplainer(
            training_data=self.training_data,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode=self.mode,
            random_state=random_state,
        )

    def _resolve_mode(self, mode: Optional[str]) -> str:
        """Resolve LIME mode from model semantics and reject contradictions."""
        if mode not in {None, "classification", "regression"}:
            raise ValueError("mode must be 'classification', 'regression', or None")

        semantic_mode = getattr(self.model, "task", None)
        if semantic_mode not in {None, "classification", "regression"}:
            raise ValueError("model.task must be 'classification' or 'regression'")

        raw_model = getattr(self.model, "model", self.model)
        if semantic_mode is None:
            estimator_type = getattr(raw_model, "_estimator_type", None)
            if estimator_type == "classifier":
                semantic_mode = "classification"
            elif estimator_type == "regressor":
                semantic_mode = "regression"
            elif hasattr(raw_model, "classes_") or hasattr(raw_model, "predict_proba"):
                semantic_mode = "classification"

        if mode is None:
            if semantic_mode is None:
                raise ValueError(
                    "mode could not be inferred from model semantics; pass "
                    "mode='classification' or mode='regression' explicitly"
                )
            return semantic_mode
        if semantic_mode is not None and mode != semantic_mode:
            raise ValueError(f"mode={mode!r} conflicts with the model's {semantic_mode} task")
        return mode

    def _validate_configuration(self) -> None:
        """Reject configurations that the wrapper cannot represent safely."""
        if self.mode not in {"classification", "regression"}:
            raise ValueError("mode must be 'classification' or 'regression'")
        if self.training_data.ndim != 2:
            raise ValueError("training_data must be a 2D array")
        if self.training_data.shape[0] == 0 or self.training_data.shape[1] == 0:
            raise ValueError("training_data must contain samples and features")
        if len(self.feature_names) != self.training_data.shape[1]:
            raise ValueError("feature_names length must match training_data columns")
        if self.mode == "classification" and len(self.class_names) < 2:
            raise ValueError("classification mode requires at least two class_names")
        if self.mode == "regression" and len(self.class_names) > 1:
            raise ValueError("LIME regression currently supports one named model output")

    def _predict_for_lime(self, data: np.ndarray) -> np.ndarray:
        """Normalize adapter output to the official LIME mode contract."""
        data = np.asarray(data)
        if data.ndim != 2 or data.shape[1] != len(self.feature_names):
            raise ValueError("LIME model inputs must be 2D with one column per feature")
        predictions = as_real_array(
            self.model.predict(data),
            name="model predictions",
        )

        if self.mode == "regression":
            if predictions.ndim == 1:
                normalized = predictions
            elif predictions.ndim == 2 and predictions.shape[1] == 1:
                normalized = predictions[:, 0]
            else:
                raise ValueError("LIME regression currently supports one model output")
            if normalized.shape[0] != data.shape[0]:
                raise ValueError("model returned the wrong number of regression predictions")
            try:
                normalized = normalized.astype(float, copy=False)
            except (TypeError, ValueError) as exc:
                raise ValueError("regression predictions must be numerical") from exc
            if not np.all(np.isfinite(normalized)):
                raise ValueError("regression predictions must be finite")
            return normalized

        if predictions.ndim != 2:
            raise ValueError(
                "LIME classification requires class probabilities with "
                "shape (n_samples, n_classes)"
            )
        if predictions.shape[0] != data.shape[0] or predictions.shape[1] == 0:
            raise ValueError("model returned an invalid classification output shape")
        try:
            predictions = predictions.astype(float, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("classification predictions must be numerical probabilities") from exc
        if not np.all(np.isfinite(predictions)):
            raise ValueError("classification probabilities must be finite")
        if predictions.shape[1] == 1:
            positive = predictions[:, 0]
            if np.any((positive < 0.0) | (positive > 1.0)):
                raise ValueError("one-column classification outputs must be probabilities")
            predictions = np.column_stack([1.0 - positive, positive])
        if predictions.shape[1] != len(self.class_names):
            raise ValueError("model output count must match class_names")
        if np.any((predictions < -1e-8) | (predictions > 1.0 + 1e-8)):
            raise ValueError("classification probabilities must lie in [0, 1]")
        if not np.allclose(predictions.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6):
            raise ValueError("classification probability rows must sum to 1")
        return predictions

    def explain(
        self,
        instance: np.ndarray,
        num_features: Optional[int] = None,
        top_labels: int = 1,
        target_class: Optional[int] = None,
    ) -> Explanation:
        """
        Generate a local explanation for the given instance.

        Args:
            instance: 1D numpy array (single row) to explain
            num_features: Number of top features to include in explanation.
                Defaults to None, which uses all features. This ensures
                evaluation metrics receive complete attribution vectors.
            top_labels: Compatibility parameter. This single-output API only
                accepts ``1``; requesting multiple labels would require a
                multi-explanation return type.
            target_class: Optional classification output-column index. If
                omitted, LIME explains the highest-probability output.

        Returns:
            Explanation object with feature attributions keyed by original
            feature names (not LIME's discretized feature strings).
        """
        instance = validate_single_tabular_instance(
            instance,
            len(self.feature_names),
            require_finite=True,
        )

        if num_features is None:
            num_features = len(self.feature_names)
        if not 1 <= num_features <= len(self.feature_names):
            raise ValueError("num_features must be between 1 and the feature count")
        if (
            not isinstance(top_labels, Integral)
            or isinstance(top_labels, bool)
            or int(top_labels) != 1
        ):
            raise ValueError(
                "LIME explain returns one output; top_labels must be 1. "
                "Use target_class to select a classification output."
            )

        if target_class is not None:
            if self.mode != "classification":
                raise ValueError("target_class is only supported in classification mode")
            if not isinstance(target_class, Integral) or isinstance(target_class, bool):
                raise TypeError("target_class must be an integer output index or None")
            target_class = int(target_class)
            if target_class < 0 or target_class >= len(self.class_names):
                raise ValueError("target_class is outside the configured class range")

        explain_kwargs = {
            "data_row": instance,
            "predict_fn": self._predict_for_lime,
            "num_features": num_features,
        }
        if self.mode == "classification":
            if target_class is None:
                explain_kwargs["top_labels"] = 1
            else:
                explain_kwargs["labels"] = (target_class,)

        lime_exp = self.explainer.explain_instance(**explain_kwargs)

        if self.mode == "regression":
            # The official LIME regression explanation stores the positive
            # local model under key 1 and its sign-reversed copy under key 0.
            label_index = 1
            label_name = self.class_names[0] if self.class_names else "output"
        else:
            label_index = int(lime_exp.top_labels[0]) if target_class is None else int(target_class)
            if label_index >= len(self.class_names):
                raise ValueError("model output has more classes than class_names")
            label_name = self.class_names[label_index]

        # Use as_map() to get (feature_index, weight) pairs.
        # This avoids the as_list() issue where LIME returns discretized
        # feature strings like "petal width (cm) <= 0.80" instead of
        # the original feature name "petal width (cm)".
        index_weight_pairs = lime_exp.as_map()[label_index]

        # Build attributions dict keyed by original feature names.
        # Initialize all features to 0.0 so we always return a
        # complete attribution vector even when num_features < total.
        attributions = {fname: 0.0 for fname in self.feature_names}
        for feat_idx, weight in index_weight_pairs:
            if 0 <= feat_idx < len(self.feature_names):
                attributions[self.feature_names[feat_idx]] = float(weight)

        model_prediction = self._predict_for_lime(instance.reshape(1, -1))
        if self.mode == "regression":
            serialized_prediction = float(model_prediction[0])
        else:
            serialized_prediction = model_prediction[0].astype(float).tolist()

        local_prediction = getattr(lime_exp, "local_pred", None)
        if local_prediction is not None:
            local_prediction = float(np.asarray(local_prediction).reshape(-1)[0])

        intercepts = getattr(lime_exp, "intercept", {})
        intercept = intercepts.get(label_index) if isinstance(intercepts, dict) else None

        return Explanation(
            explainer_name="LIME",
            target_class=label_name,
            explanation_data={
                "feature_attributions": attributions,
                "mode": self.mode,
                "model_prediction": serialized_prediction,
                "local_prediction": local_prediction,
                "local_model_score": float(getattr(lime_exp, "score", np.nan)),
                "intercept": float(intercept) if intercept is not None else None,
                "lime_feature_conditions": [
                    (str(condition), float(weight))
                    for condition, weight in lime_exp.as_list(label=label_index)
                ],
            },
            feature_names=self.feature_names,
        )

    def explain_batch(
        self,
        X: np.ndarray,
        num_features: Optional[int] = None,
        top_labels: int = 1,
        target_class: Optional[int] = None,
    ) -> List[Explanation]:
        """
        Generate explanations for multiple instances.

        Args:
            X: 2D numpy array of instances
            num_features: Number of features per explanation.
                Defaults to None, which uses all features.
            top_labels: Compatibility parameter; must be ``1``.
            target_class: Optional classification output-column index applied
                to every row.

        Returns:
            List of Explanation objects
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError("X must be a 2D array with one column per feature")

        return [
            self.explain(
                X[i],
                num_features=num_features,
                top_labels=top_labels,
                target_class=target_class,
            )
            for i in range(X.shape[0])
        ]
