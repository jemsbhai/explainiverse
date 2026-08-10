# src/explainiverse/explainers/attribution/treeshap_wrapper.py
"""
TreeSHAP adapter for tree-based models.

This module delegates to ``shap.TreeExplainer``. The meaning of the returned
values depends on SHAP's model-output and feature-perturbation modes; no
universal runtime advantage or mode-independent exactness claim is made here.

Reference:
    Lundberg, S.M., Erion, G.G., & Lee, S.I. (2018). Consistent Individualized
    Feature Attribution for Tree Ensembles. arXiv:1802.03888.
    
Accuracy-audited scope:
    - Selected scikit-learn and XGBoost classification/regression outputs covered
      by the repository's reference and additivity tests.
    - Other recognized ``TreeExplainer`` estimator families remain outside that
      verified claim scope even when their optional dependency is installed.
"""

from numbers import Integral
from typing import List, Optional

import numpy as np
import shap

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation
from explainiverse.explainers._validation import (
    as_real_array,
    validate_name_sequence,
    validate_single_tabular_instance,
)

# Tree-based model types that TreeSHAP supports
SUPPORTED_TREE_MODELS = (
    "RandomForestClassifier",
    "RandomForestRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "XGBClassifier",
    "XGBRegressor",
    "XGBRFClassifier",
    "XGBRFRegressor",
    "LGBMClassifier",
    "LGBMRegressor",
    "CatBoostClassifier",
    "CatBoostRegressor",
    "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor",
)


def _is_tree_model(model) -> bool:
    """Check a supported tree family without rejecting safe subclasses."""
    return any(cls.__name__ in SUPPORTED_TREE_MODELS for cls in type(model).mro())


def _is_xgboost_model(model) -> bool:
    """Return whether a model inherits from a recognized XGBoost estimator."""
    return any(cls.__name__.startswith("XGB") for cls in type(model).mro())


def _get_raw_model(model):
    """
    Extract the raw model from an adapter if necessary.

    TreeExplainer needs the actual sklearn/xgboost model, not an adapter.
    """
    if hasattr(model, "model"):
        return model.model
    return model


class TreeShapExplainer(BaseExplainer):
    """
    TreeSHAP explainer for tree-based models.

    Uses SHAP's ``TreeExplainer`` and records the effective output-space
    contract. Additivity is checked in the supported modes covered by tests.
    Interaction values and background-data requirements depend on the estimator
    and selected SHAP configuration.

    Attributes:
        model: The tree-based model (sklearn, XGBoost, LightGBM, or CatBoost)
        feature_names: List of feature names
        class_names: List of class names for classification
        explainer: The underlying SHAP TreeExplainer
        task: "classification" or "regression"
    """

    def __init__(
        self,
        model,
        feature_names: List[str],
        class_names: Optional[List[str]] = None,
        background_data: Optional[np.ndarray] = None,
        task: Optional[str] = None,
        model_output: str = "auto",
        feature_perturbation: str = "tree_path_dependent",
    ):
        """
        Initialize the TreeSHAP explainer.

        Args:
            model: A recognized tree-based model or adapter containing one.
                   See registry ``claim_scope`` for the accuracy-audited subset.
            feature_names: List of feature names.
            class_names: List of class names (for classification).
            background_data: Optional background dataset for interventional
                            feature perturbation. If None, uses tree_path_dependent.
            task: "classification" or "regression". If omitted, inferred
                  from the tree estimator's task semantics.
            model_output: How to transform model output. Options:
                         - "auto": Automatically detect
                         - "raw": Raw model output
                         - "probability": Probability output (classification)
            feature_perturbation: SHAP feature-dependence mode:
                                 - "tree_path_dependent": Uses tree path counts
                                 - "interventional": Requires background data
        """
        # Extract raw model if wrapped in adapter
        raw_model = _get_raw_model(model)

        # Validate that it's a supported tree model
        if not _is_tree_model(raw_model):
            model_type = type(raw_model).__name__
            raise ValueError(
                f"TreeSHAP requires a tree-based model. Got {model_type}. "
                f"Supported models: {', '.join(SUPPORTED_TREE_MODELS[:6])}..."
            )

        super().__init__(model)
        self.raw_model = raw_model
        validated_features = validate_name_sequence(feature_names, name="feature_names")
        assert validated_features is not None
        self.feature_names = validated_features
        self.class_names = (
            validate_name_sequence(class_names, name="class_names") if class_names else None
        )
        self.model_output = model_output
        self.feature_perturbation = feature_perturbation

        if task not in {None, "classification", "regression"}:
            raise ValueError("task must be 'classification', 'regression', or None")
        estimator_type = getattr(raw_model, "_estimator_type", None)
        if estimator_type == "classifier" or type(raw_model).__name__.endswith("Classifier"):
            inferred_task = "classification"
        elif estimator_type == "regressor" or type(raw_model).__name__.endswith("Regressor"):
            inferred_task = "regression"
        else:
            inferred_task = None
        if task is None:
            if inferred_task is None:
                raise ValueError("TreeSHAP could not infer model task; pass task explicitly")
            task = inferred_task
        elif inferred_task is not None and task != inferred_task:
            raise ValueError(f"task={task!r} conflicts with the tree estimator semantics")
        self.task = task
        if model_output == "log_loss":
            raise ValueError(
                "model_output='log_loss' is not supported because this wrapper "
                "does not accept the required per-sample labels"
            )
        if model_output not in {"auto", "raw", "probability"}:
            raise ValueError("model_output must be 'auto', 'raw', or 'probability'")
        if feature_perturbation not in {
            "tree_path_dependent",
            "interventional",
        }:
            raise ValueError(
                "feature_perturbation must be 'tree_path_dependent' or " "'interventional'"
            )
        if feature_perturbation == "interventional" and background_data is None:
            raise ValueError("background_data is required for interventional TreeSHAP")
        n_model_features = getattr(raw_model, "n_features_in_", None)
        if n_model_features is not None and len(self.feature_names) != n_model_features:
            raise ValueError("feature_names length must match the model input feature count")
        classes = getattr(raw_model, "classes_", None)
        if task == "classification" and isinstance(classes, (list, tuple)):
            raise ValueError("TreeSHAP does not support multi-output classification")
        if (
            task == "classification"
            and classes is not None
            and self.class_names is not None
            and len(self.class_names) != len(classes)
        ):
            raise ValueError("class_names length must match model.classes_")
        if task == "regression":
            if self.class_names is not None and len(self.class_names) != 1:
                raise ValueError("TreeSHAP currently supports single-output regression only")
            for attribute in ("n_outputs_", "n_targets_"):
                output_count = getattr(raw_model, attribute, None)
                if output_count is not None and int(output_count) > 1:
                    raise ValueError("TreeSHAP currently supports single-output regression only")

        # Create TreeExplainer
        explainer_kwargs: dict[str, object] = {}

        if feature_perturbation == "interventional":
            explainer_kwargs["data"] = background_data
            explainer_kwargs["feature_perturbation"] = "interventional"

        if model_output != "auto":
            explainer_kwargs["model_output"] = model_output

        # Do not mutate SHAP internals or collapse vector-valued XGBoost base
        # scores to a scalar. If an installed SHAP/XGBoost combination is
        # incompatible, fail explicitly rather than returning corrupted values.
        try:
            self.explainer = shap.TreeExplainer(raw_model, **explainer_kwargs)
        except ValueError as exc:
            if "could not convert string to float" in str(exc):
                raise RuntimeError(
                    "The installed SHAP version cannot parse this XGBoost "
                    "model's vector base_score. Upgrade SHAP to a compatible "
                    "version; Explainiverse will not apply a lossy scalar-mean "
                    "patch."
                ) from exc
            raise
        self.background_data = background_data
        self.effective_model_output = getattr(
            getattr(self.explainer, "model", None),
            "model_output",
            model_output,
        )

    def _number_of_classes(self) -> Optional[int]:
        """Return the classifier output count when it is knowable."""
        classes = getattr(self.raw_model, "classes_", None)
        if classes is not None:
            return len(classes)
        if self.class_names is not None:
            return len(self.class_names)
        return None

    def _resolve_class_index(
        self,
        X: np.ndarray,
        sample_index: int,
        target_class: Optional[int],
    ) -> int:
        """Resolve a target to a SHAP output-column index.

        ``raw_model.predict`` returns labels, which need not be zero-based
        integers. Automatic targets are therefore mapped through
        ``raw_model.classes_`` rather than cast to ``int``.
        """
        if self.task == "regression":
            if target_class not in {None, 0}:
                raise ValueError("Regression TreeSHAP has one output index: 0")
            return 0

        n_classes = self._number_of_classes()
        if target_class is not None:
            if not isinstance(target_class, Integral) or isinstance(target_class, bool):
                raise TypeError("target_class must be an integer output index")
            resolved = int(target_class)
        else:
            if not hasattr(self.raw_model, "predict"):
                raise ValueError("target_class is required when the model has no predict method")
            predicted_label = self.raw_model.predict(X[sample_index : sample_index + 1])[0]
            classes = getattr(self.raw_model, "classes_", None)
            if classes is not None:
                matches = np.flatnonzero(np.asarray(classes) == predicted_label)
                if len(matches) != 1:
                    raise ValueError(
                        f"Predicted label {predicted_label!r} was not found "
                        "uniquely in model.classes_"
                    )
                resolved = int(matches[0])
            elif isinstance(predicted_label, Integral):
                resolved = int(predicted_label)
            else:
                raise ValueError(
                    "Cannot map the predicted label to a class index without "
                    "model.classes_; pass target_class explicitly"
                )

        if resolved < 0 or (n_classes is not None and resolved >= n_classes):
            raise ValueError(f"target_class={resolved} is outside the model output range")
        return resolved

    def _label_name(self, class_index: int) -> str:
        """Return presentation metadata for a resolved output index."""
        if self.task == "regression":
            return self.class_names[0] if self.class_names else "output"
        if self.class_names is not None:
            if class_index >= len(self.class_names):
                raise ValueError("class_names does not cover target_class")
            return self.class_names[class_index]
        classes = getattr(self.raw_model, "classes_", None)
        if classes is not None and class_index < len(classes):
            return str(classes[class_index])
        return f"class_{class_index}"

    def _base_value(self, class_index: int, single_output_binary: bool) -> float:
        """Select/transform SHAP's expected value for the explained output."""
        values = np.asarray(self.explainer.expected_value).reshape(-1)
        if values.size == 0:
            raise ValueError("TreeExplainer returned an empty expected_value")
        if not single_output_binary:
            if values.size == 1:
                return float(values[0])
            if class_index >= values.size:
                raise ValueError("expected_value does not cover target_class")
            return float(values[class_index])

        positive_base = float(values[0])
        if class_index == 1:
            return positive_base
        if self.effective_model_output == "probability":
            return 1.0 - positive_base
        return -positive_base

    def _reference_output(
        self,
        X: np.ndarray,
        class_index: int,
        explained_value: float,
    ):
        """Identify the concrete model output represented by SHAP values."""
        row = X.reshape(1, -1)
        if self.task == "regression":
            prediction = np.asarray(self.raw_model.predict(row)).reshape(-1)
            if prediction.size != 1:
                raise ValueError("TreeSHAP currently supports single-output regression only")
            return "model_output", float(prediction[0])

        candidates = []
        if str(self.effective_model_output) == "raw" and _is_xgboost_model(self.raw_model):
            # XGBoost's default TreeSHAP output is the untransformed margin.
            # Its sklearn estimators expose that score through output_margin,
            # not decision_function.
            margin = np.asarray(self.raw_model.predict(row, output_margin=True))
            if margin.ndim == 1 or (margin.ndim == 2 and margin.shape[1] == 1):
                positive_margin = float(margin.reshape(-1)[0])
                candidates.append(
                    (
                        "raw_margin",
                        positive_margin if class_index == 1 else -positive_margin,
                    )
                )
            elif margin.ndim == 2 and class_index < margin.shape[1]:
                candidates.append(("raw_margin", float(margin[0, class_index])))

        if hasattr(self.raw_model, "predict_proba"):
            probabilities = np.asarray(self.raw_model.predict_proba(row))[0]
            if class_index < len(probabilities):
                candidates.append(("probability", float(probabilities[class_index])))

        if hasattr(self.raw_model, "decision_function"):
            margin = np.asarray(self.raw_model.decision_function(row))
            if margin.ndim == 1 or (margin.ndim == 2 and margin.shape[1] == 1):
                positive_margin = float(margin.reshape(-1)[0])
                candidates.append(
                    (
                        "raw_margin",
                        positive_margin if class_index == 1 else -positive_margin,
                    )
                )
            elif margin.ndim == 2 and class_index < margin.shape[1]:
                candidates.append(("raw_margin", float(margin[0, class_index])))

        if not candidates:
            return str(self.effective_model_output), None

        # SHAP's "raw" output is estimator-specific (e.g. sklearn forests
        # expose probabilities while gradient boosting exposes margins).
        # Match the additive value to the actual model API instead of guessing.
        output_space, reference = min(
            candidates,
            key=lambda item: abs(item[1] - explained_value),
        )
        return output_space, reference

    def _extract_class_shap_values(
        self,
        shap_values,
        X: np.ndarray,
        sample_index: int,
        target_class: Optional[int],
    ):
        """
        Extract per-feature SHAP values for the target class from a single
        sample within a (possibly batched) shap_values result.

        TreeExplainer.shap_values() can return:
          1. list of arrays -- one (n_samples, n_features) array per class.
          2. 3D ndarray (n_samples, n_features, n_classes).
          3. 2D ndarray (n_samples, n_features) -- binary/regression.
          4. 1D ndarray (n_features,) -- single sample, single output.

        Args:
            shap_values: Raw output from TreeExplainer.shap_values().
            X: The input instances as 2D array.
            sample_index: Which sample to extract (for batch calls).
            target_class: Which class to explain. If None, uses predicted class.

        Returns:
            ``(class_shap, label_index, label_name, single_output_binary)``
            where ``class_shap`` is a 1D feature vector.
        """
        resolved_class = self._resolve_class_index(X, sample_index, target_class)
        label_name = self._label_name(resolved_class)

        # Case 1: list of arrays -- one per class
        if isinstance(shap_values, list):
            n_classes = len(shap_values)
            if self.task == "regression" and n_classes != 1:
                raise ValueError("TreeSHAP currently supports single-output regression only")
            if resolved_class >= n_classes:
                raise ValueError("TreeExplainer output does not cover target_class")
            tc = resolved_class
            class_shap = np.asarray(shap_values[tc])
            if class_shap.ndim == 2:
                class_shap = class_shap[sample_index]
            # else already 1D
            return class_shap, tc, label_name, False

        # From here, shap_values is an ndarray
        shap_arr = np.asarray(shap_values)

        # Case 2: 3D -- (n_samples, n_features, n_classes)
        if shap_arr.ndim == 3:
            n_classes = shap_arr.shape[2]
            if self.task == "regression" and n_classes != 1:
                raise ValueError("TreeSHAP currently supports single-output regression only")
            if resolved_class >= n_classes:
                raise ValueError("TreeExplainer output does not cover target_class")
            tc = resolved_class
            class_shap = shap_arr[sample_index, :, tc]
            return class_shap, tc, label_name, False

        # Case 3: 2D -- (n_samples, n_features) -- binary/regression
        if shap_arr.ndim == 2:
            class_shap = shap_arr[sample_index]
            single_output_binary = self.task == "classification" and self._number_of_classes() == 2
            if single_output_binary and resolved_class == 0:
                class_shap = -class_shap
            elif self.task == "classification" and not single_output_binary:
                raise ValueError(
                    "A single TreeSHAP output cannot represent this " "multi-class model"
                )
            return (
                class_shap,
                resolved_class,
                label_name,
                single_output_binary,
            )

        # Case 4: 1D -- (n_features,) -- single sample
        if shap_arr.ndim == 1:
            single_output_binary = self.task == "classification" and self._number_of_classes() == 2
            class_shap = shap_arr
            if single_output_binary and resolved_class == 0:
                class_shap = -class_shap
            elif self.task == "classification" and not single_output_binary:
                raise ValueError(
                    "A single TreeSHAP output cannot represent this " "multi-class model"
                )
            return (
                class_shap,
                resolved_class,
                label_name,
                single_output_binary,
            )

        raise ValueError(
            f"Unexpected SHAP values shape: {shap_arr.shape}. " f"Expected list, 3D, 2D, or 1D."
        )

    def explain(
        self,
        instance: np.ndarray,
        target_class: Optional[int] = None,
        check_additivity: bool = False,
    ) -> Explanation:
        """
        Generate TreeSHAP explanation for a single instance.

        Args:
            instance: 1D numpy array of input features.
            target_class: For multi-class, which class to explain.
                         If None, uses the predicted class.
            check_additivity: Whether to verify SHAP values sum to
                             prediction - expected_value.

        Returns:
            Explanation object with feature attributions keyed by original
            feature names.
        """
        instance = validate_single_tabular_instance(
            instance,
            len(self.feature_names),
            require_finite=False,
        )
        instance_2d = instance.reshape(1, -1)

        shap_values = self.explainer.shap_values(instance_2d, check_additivity=check_additivity)

        (
            class_shap,
            label_index,
            label_name,
            single_output_binary,
        ) = self._extract_class_shap_values(
            shap_values,
            instance_2d,
            sample_index=0,
            target_class=target_class,
        )

        # Validate shape
        class_shap = np.asarray(class_shap).ravel()
        if len(class_shap) != len(self.feature_names):
            raise ValueError(
                f"SHAP values length ({len(class_shap)}) does not match "
                f"number of features ({len(self.feature_names)})."
            )

        # Build attributions dict keyed by original feature names
        attributions = {fname: float(class_shap[i]) for i, fname in enumerate(self.feature_names)}

        base_value = self._base_value(label_index, single_output_binary)
        explained_value = float(base_value + np.sum(class_shap))
        output_space, reference_value = self._reference_output(
            instance, label_index, explained_value
        )
        additivity_residual = (
            None if reference_value is None else float(explained_value - reference_value)
        )

        explanation_data = {
            "feature_attributions": attributions,
            "base_value": base_value,
            "shap_values_raw": class_shap.tolist(),
            "class_index": label_index,
            "shap_model_output": str(self.effective_model_output),
            "output_space": output_space,
            "explained_value": explained_value,
            "model_reference_value": reference_value,
            "additivity_residual": additivity_residual,
            "feature_perturbation": self.feature_perturbation,
        }

        # For multiclass list format, include all-class values
        if isinstance(shap_values, list) and len(shap_values) > 1:
            all_class_shap = {
                (
                    self.class_names[i]
                    if self.class_names and i < len(self.class_names)
                    else f"class_{i}"
                ): np.asarray(shap_values[i][0])
                .ravel()
                .tolist()
                for i in range(len(shap_values))
            }
            explanation_data["all_class_shap_values"] = all_class_shap

        return Explanation(
            explainer_name="TreeSHAP",
            target_class=label_name,
            explanation_data=explanation_data,
            feature_names=self.feature_names,
        )

    def explain_batch(
        self, X: np.ndarray, target_class: Optional[int] = None, check_additivity: bool = False
    ) -> List[Explanation]:
        """
        Generate TreeSHAP explanations for multiple instances.

        Args:
            X: 2D numpy array of instances (n_samples, n_features).
            target_class: For multi-class, which class to explain.
                         If None, uses the predicted class for each instance.
            check_additivity: Whether to verify SHAP value additivity.

        Returns:
            List of Explanation objects.
        """
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError("X must be 2D with one column per feature_name")

        shap_values = self.explainer.shap_values(X, check_additivity=check_additivity)

        explanations = []
        for i in range(X.shape[0]):
            (
                class_shap,
                label_index,
                label_name,
                single_output_binary,
            ) = self._extract_class_shap_values(
                shap_values,
                X,
                sample_index=i,
                target_class=target_class,
            )

            class_shap = np.asarray(class_shap).ravel()
            if len(class_shap) != len(self.feature_names):
                raise ValueError("TreeSHAP feature count does not match feature_names")

            attributions = {
                fname: float(class_shap[j]) for j, fname in enumerate(self.feature_names)
            }

            base_value = self._base_value(label_index, single_output_binary)
            explained_value = float(base_value + np.sum(class_shap))
            output_space, reference_value = self._reference_output(
                X[i], label_index, explained_value
            )
            additivity_residual = (
                None if reference_value is None else float(explained_value - reference_value)
            )

            explanations.append(
                Explanation(
                    explainer_name="TreeSHAP",
                    target_class=label_name,
                    explanation_data={
                        "feature_attributions": attributions,
                        "base_value": base_value,
                        "shap_values_raw": class_shap.tolist(),
                        "class_index": label_index,
                        "shap_model_output": str(self.effective_model_output),
                        "output_space": output_space,
                        "explained_value": explained_value,
                        "model_reference_value": reference_value,
                        "additivity_residual": additivity_residual,
                        "feature_perturbation": self.feature_perturbation,
                    },
                    feature_names=self.feature_names,
                )
            )

        return explanations

    def explain_interactions(
        self, instance: np.ndarray, target_class: Optional[int] = None
    ) -> Explanation:
        """
        Compute SHAP interaction values for an instance.

        Interaction values show how pairs of features jointly contribute
        to the prediction. The diagonal contains main effects.

        Args:
            instance: 1D numpy array of input features.
            target_class: For multi-class, which class to explain.

        Returns:
            Explanation object with interaction matrix.
        """
        if self.feature_perturbation != "tree_path_dependent":
            raise ValueError(
                "TreeSHAP interaction values are unsupported for interventional "
                "feature perturbation; use tree_path_dependent or call explain()"
            )
        instance = validate_single_tabular_instance(
            instance,
            len(self.feature_names),
            require_finite=False,
        )
        instance_2d = instance.reshape(1, -1)

        interaction_values = self.explainer.shap_interaction_values(instance_2d)

        resolved_class = self._resolve_class_index(
            instance_2d, sample_index=0, target_class=target_class
        )
        label_name = self._label_name(resolved_class)

        # Handle different return formats
        if isinstance(interaction_values, list):
            n_classes = len(interaction_values)
            if resolved_class >= n_classes:
                raise ValueError("TreeExplainer interaction output does not cover target_class")
            tc = resolved_class
            interactions = np.array(interaction_values[tc][0])
        elif interaction_values.ndim == 4:
            n_classes = interaction_values.shape[3]
            if resolved_class >= n_classes:
                raise ValueError("TreeExplainer interaction output does not cover target_class")
            tc = resolved_class
            interactions = interaction_values[0, :, :, tc]
        else:
            interactions = interaction_values[0]
            if (
                self.task == "classification"
                and self._number_of_classes() == 2
                and resolved_class == 0
            ):
                interactions = -interactions

        interactions = as_real_array(
            interactions,
            name="TreeExplainer interaction values",
            dtype=float,
            require_finite=True,
        )
        if interactions.ndim > 2:
            interactions = interactions[:, :, 0] if interactions.ndim == 3 else interactions

        n_features = len(self.feature_names)
        if interactions.shape != (n_features, n_features):
            raise ValueError(
                "TreeExplainer returned an interaction matrix with an unexpected shape"
            )
        interaction_dict = {}
        main_effects = {}

        for i in range(n_features):
            fname_i = self.feature_names[i]
            val = interactions[i, i]
            main_effects[fname_i] = (
                float(val) if np.isscalar(val) or val.size == 1 else float(val.flat[0])
            )

            for j in range(i + 1, n_features):
                fname_j = self.feature_names[j]
                val_ij = interactions[i, j]
                val_ji = interactions[j, i]
                ij = (
                    float(val_ij)
                    if np.isscalar(val_ij) or val_ij.size == 1
                    else float(val_ij.flat[0])
                )
                ji = (
                    float(val_ji)
                    if np.isscalar(val_ji) or val_ji.size == 1
                    else float(val_ji.flat[0])
                )
                interaction_dict[f"{fname_i} x {fname_j}"] = ij + ji

        sorted_interactions = dict(
            sorted(interaction_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        return Explanation(
            explainer_name="TreeSHAP_Interactions",
            target_class=label_name,
            explanation_data={
                "feature_attributions": main_effects,
                "interactions": sorted_interactions,
                "interaction_matrix": interactions.tolist(),
                "feature_names": self.feature_names,
            },
            feature_names=self.feature_names,
        )

    def get_expected_value(self, target_class: Optional[int] = None) -> float:
        """
        Get the expected (base) value of the model.

        Args:
            target_class: For multi-class, which class's expected value.

        Returns:
            The expected value as a float.
        """
        class_index = 0 if target_class is None else target_class
        if self.task == "classification":
            n_classes = self._number_of_classes()
            if not isinstance(class_index, Integral) or isinstance(class_index, bool):
                raise TypeError("target_class must be an integer output index")
            class_index = int(class_index)
            if class_index < 0 or (n_classes is not None and class_index >= n_classes):
                raise ValueError("target_class is outside the model output range")
        else:
            if target_class not in {None, 0}:
                raise ValueError("Regression TreeSHAP has one output index: 0")
            class_index = 0

        expected_values = np.asarray(self.explainer.expected_value).reshape(-1)
        if self.task == "regression" and expected_values.size != 1:
            raise ValueError("TreeSHAP currently supports single-output regression only")
        single_output_binary = (
            self.task == "classification"
            and self._number_of_classes() == 2
            and expected_values.size == 1
        )
        return self._base_value(class_index, single_output_binary)
