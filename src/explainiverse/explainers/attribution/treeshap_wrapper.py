# src/explainiverse/explainers/attribution/treeshap_wrapper.py
"""
TreeSHAP Explainer - Optimized SHAP for Tree-based Models.

TreeSHAP computes exact SHAP values in polynomial time for tree-based models,
making it significantly faster than KernelSHAP while providing exact (not
approximate) Shapley values.

Reference:
    Lundberg, S.M., Erion, G.G., & Lee, S.I. (2018). Consistent Individualized
    Feature Attribution for Tree Ensembles. arXiv:1802.03888.
    
Supported Models:
    - scikit-learn: RandomForest, GradientBoosting, DecisionTree, ExtraTrees
    - XGBoost: XGBClassifier, XGBRegressor
    - LightGBM: LGBMClassifier, LGBMRegressor (if installed)
    - CatBoost: CatBoostClassifier, CatBoostRegressor (if installed)
"""

import json
import logging

import numpy as np
import shap
from typing import List, Optional, Union

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation

logger = logging.getLogger(__name__)


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
    """Check if a model is a supported tree-based model."""
    model_name = type(model).__name__
    return model_name in SUPPORTED_TREE_MODELS


def _get_raw_model(model):
    """
    Extract the raw model from an adapter if necessary.
    
    TreeExplainer needs the actual sklearn/xgboost model, not an adapter.
    """
    if hasattr(model, 'model'):
        return model.model
    return model


def _patch_shap_xgboost3_compat():
    """
    Monkey-patch SHAP's ``XGBTreeModelLoader`` for xgboost >= 3.0 compat.

    In xgboost 3.x, multiclass models serialize ``base_score`` as a JSON
    array string (e.g. ``'[4.76E-1,4.76E-1,4.76E-2]'``), but SHAP <= 0.50
    calls ``float(base_score)`` in ``XGBTreeModelLoader.__init__`` and
    raises ``ValueError``.

    The patch is idempotent -- calling it multiple times is safe.

    See https://github.com/shap/shap/issues/3750
    """
    try:
        from shap.explainers._tree import XGBTreeModelLoader
    except ImportError:
        return

    if getattr(XGBTreeModelLoader, '_explainiverse_patched', False):
        return

    _original_init = XGBTreeModelLoader.__init__

    def _patched_init(self, xgb_model):
        """Wrap original init to handle xgboost 3.x array base_score."""
        try:
            _original_init(self, xgb_model)
            self.original_base_scores = None
        except ValueError as exc:
            if "could not convert string to float" not in str(exc):
                raise

            import shap.explainers._tree as _tree_mod

            _orig_decode = _tree_mod.decode_ubjson_buffer
            _captured_base_scores = [None]

            def _fixed_decode(fd):
                result = _orig_decode(fd)
                bs = result["learner"]["learner_model_param"]["base_score"]
                if isinstance(bs, str) and bs.startswith("["):
                    per_class = [float(v) for v in json.loads(bs)]
                    _captured_base_scores[0] = per_class
                    scalar_mean = float(np.mean(per_class))
                    result["learner"]["learner_model_param"][
                        "base_score"
                    ] = str(scalar_mean)
                return result

            _tree_mod.decode_ubjson_buffer = _fixed_decode
            try:
                _original_init(self, xgb_model)
            finally:
                _tree_mod.decode_ubjson_buffer = _orig_decode

            self.original_base_scores = _captured_base_scores[0]

            logger.info(
                "Applied xgboost 3.x base_score fix: parsed %d-class "
                "array to scalar %.6f",
                len(self.original_base_scores) if self.original_base_scores else 0,
                self.base_score,
            )

    XGBTreeModelLoader.__init__ = _patched_init
    XGBTreeModelLoader._explainiverse_patched = True
    logger.debug(
        "Patched SHAP XGBTreeModelLoader for xgboost 3.x compatibility"
    )


class TreeShapExplainer(BaseExplainer):
    """
    TreeSHAP explainer for tree-based models.
    
    Uses SHAP's TreeExplainer to compute exact SHAP values in polynomial time.
    This is significantly faster than KernelSHAP for supported tree models
    and provides exact Shapley values rather than approximations.
    
    Key advantages over KernelSHAP:
    - Exact SHAP values (not approximations)
    - O(TLD^2) complexity vs O(TL2^M) for KernelSHAP
    - Can compute interaction values
    - No background data sampling needed
    
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
        task: str = "classification",
        model_output: str = "auto",
        feature_perturbation: str = "tree_path_dependent"
    ):
        """
        Initialize the TreeSHAP explainer.
        
        Args:
            model: A tree-based model or adapter containing one.
                   Supported: RandomForest, GradientBoosting, XGBoost, 
                   LightGBM, CatBoost, DecisionTree, ExtraTrees.
            feature_names: List of feature names.
            class_names: List of class names (for classification).
            background_data: Optional background dataset for interventional
                            feature perturbation. If None, uses tree_path_dependent.
            task: "classification" or "regression".
            model_output: How to transform model output. Options:
                         - "auto": Automatically detect
                         - "raw": Raw model output
                         - "probability": Probability output (classification)
                         - "log_loss": Log loss output
            feature_perturbation: Method for handling feature perturbation:
                                 - "tree_path_dependent": Fast, uses tree structure
                                 - "interventional": Slower, requires background data
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
        self.feature_names = list(feature_names)
        self.class_names = list(class_names) if class_names else None
        self.task = task
        self.model_output = model_output
        self.feature_perturbation = feature_perturbation
        
        # Create TreeExplainer
        explainer_kwargs = {}
        
        if feature_perturbation == "interventional" and background_data is not None:
            explainer_kwargs["data"] = background_data
            explainer_kwargs["feature_perturbation"] = "interventional"
        
        if model_output != "auto":
            explainer_kwargs["model_output"] = model_output
        
        _patch_shap_xgboost3_compat()
        
        self.explainer = shap.TreeExplainer(raw_model, **explainer_kwargs)
        self.background_data = background_data

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
            (class_shap, label_index, label_name) where class_shap is
            a 1D array of shape (n_features,).
        """
        # Determine target class from prediction if not specified
        if target_class is None and hasattr(self.raw_model, 'predict'):
            pred = self.raw_model.predict(
                X[sample_index:sample_index + 1]
            )[0]
            resolved_class = int(pred)
        elif target_class is not None:
            resolved_class = target_class
        else:
            resolved_class = 0

        # Case 1: list of arrays -- one per class
        if isinstance(shap_values, list):
            n_classes = len(shap_values)
            tc = min(resolved_class, n_classes - 1)
            class_shap = np.asarray(shap_values[tc])
            if class_shap.ndim == 2:
                class_shap = class_shap[sample_index]
            # else already 1D

            if self.class_names and tc < len(self.class_names):
                label_name = self.class_names[tc]
            else:
                label_name = f"class_{tc}"

            return class_shap, tc, label_name

        # From here, shap_values is an ndarray
        shap_arr = np.asarray(shap_values)

        # Case 2: 3D -- (n_samples, n_features, n_classes)
        if shap_arr.ndim == 3:
            n_classes = shap_arr.shape[2]
            tc = min(resolved_class, n_classes - 1)
            class_shap = shap_arr[sample_index, :, tc]

            if self.class_names and tc < len(self.class_names):
                label_name = self.class_names[tc]
            else:
                label_name = f"class_{tc}"

            return class_shap, tc, label_name

        # Case 3: 2D -- (n_samples, n_features) -- binary/regression
        if shap_arr.ndim == 2:
            class_shap = shap_arr[sample_index]
            if self.class_names and len(self.class_names) > 1:
                label_name = self.class_names[1]
            elif self.class_names:
                label_name = self.class_names[0]
            else:
                label_name = "output"
            return class_shap, 0, label_name

        # Case 4: 1D -- (n_features,) -- single sample
        if shap_arr.ndim == 1:
            if self.class_names and len(self.class_names) > 1:
                label_name = self.class_names[1]
            elif self.class_names:
                label_name = self.class_names[0]
            else:
                label_name = "output"
            return shap_arr, 0, label_name

        raise ValueError(
            f"Unexpected SHAP values shape: {shap_arr.shape}. "
            f"Expected list, 3D, 2D, or 1D."
        )

    def explain(
        self,
        instance: np.ndarray,
        target_class: Optional[int] = None,
        check_additivity: bool = False
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
        instance = np.array(instance).flatten()
        instance_2d = instance.reshape(1, -1)
        
        shap_values = self.explainer.shap_values(
            instance_2d,
            check_additivity=check_additivity
        )

        class_shap, label_index, label_name = self._extract_class_shap_values(
            shap_values, instance_2d, sample_index=0,
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
        attributions = {
            fname: float(class_shap[i])
            for i, fname in enumerate(self.feature_names)
        }
        
        # Get expected value (base value)
        expected_value = self.explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            ev = np.asarray(expected_value)
            if label_index < len(ev):
                base_value = float(ev[label_index])
            else:
                base_value = float(ev[0])
        else:
            base_value = float(expected_value)
        
        explanation_data = {
            "feature_attributions": attributions,
            "base_value": base_value,
            "shap_values_raw": class_shap.tolist(),
        }
        
        # For multiclass list format, include all-class values
        if isinstance(shap_values, list) and len(shap_values) > 1:
            all_class_shap = {
                (self.class_names[i] if self.class_names and i < len(self.class_names)
                 else f"class_{i}"): np.asarray(shap_values[i][0]).ravel().tolist()
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
        self,
        X: np.ndarray,
        target_class: Optional[int] = None,
        check_additivity: bool = False
    ) -> List[Explanation]:
        """
        Generate TreeSHAP explanations for multiple instances efficiently.
        
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
        
        shap_values = self.explainer.shap_values(X, check_additivity=check_additivity)
        
        explanations = []
        for i in range(X.shape[0]):
            class_shap, label_index, label_name = self._extract_class_shap_values(
                shap_values, X, sample_index=i,
                target_class=target_class,
            )

            class_shap = np.asarray(class_shap).ravel()

            attributions = {
                fname: float(class_shap[j])
                for j, fname in enumerate(self.feature_names)
            }

            expected_value = self.explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                ev = np.asarray(expected_value)
                if label_index < len(ev):
                    base_value = float(ev[label_index])
                else:
                    base_value = float(ev[0])
            else:
                base_value = float(expected_value)

            explanations.append(Explanation(
                explainer_name="TreeSHAP",
                target_class=label_name,
                explanation_data={
                    "feature_attributions": attributions,
                    "base_value": base_value,
                    "shap_values_raw": class_shap.tolist(),
                },
                feature_names=self.feature_names,
            ))
        
        return explanations
    
    def explain_interactions(
        self,
        instance: np.ndarray,
        target_class: Optional[int] = None
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
        instance = np.array(instance).flatten()
        instance_2d = instance.reshape(1, -1)
        
        interaction_values = self.explainer.shap_interaction_values(instance_2d)
        
        # Determine target class for prediction
        if target_class is None and hasattr(self.raw_model, 'predict'):
            target_class = int(self.raw_model.predict(instance_2d)[0])
        elif target_class is None:
            target_class = 0
        
        # Handle different return formats
        if isinstance(interaction_values, list):
            n_classes = len(interaction_values)
            tc = min(target_class, n_classes - 1)
            interactions = np.array(interaction_values[tc][0])
            
            if self.class_names and tc < len(self.class_names):
                label_name = self.class_names[tc]
            else:
                label_name = f"class_{tc}"
        elif interaction_values.ndim == 4:
            n_classes = interaction_values.shape[3]
            tc = min(target_class, n_classes - 1)
            interactions = interaction_values[0, :, :, tc]
            
            if self.class_names and tc < len(self.class_names):
                label_name = self.class_names[tc]
            else:
                label_name = f"class_{tc}"
        else:
            interactions = interaction_values[0]
            label_name = self.class_names[1] if self.class_names and len(self.class_names) > 1 else "output"
        
        interactions = np.array(interactions)
        if interactions.ndim > 2:
            interactions = interactions[:, :, 0] if interactions.ndim == 3 else interactions
        
        n_features = len(self.feature_names)
        interaction_dict = {}
        main_effects = {}
        
        for i in range(n_features):
            fname_i = self.feature_names[i]
            val = interactions[i, i]
            main_effects[fname_i] = float(val) if np.isscalar(val) or val.size == 1 else float(val.flat[0])
            
            for j in range(i + 1, n_features):
                fname_j = self.feature_names[j]
                val_ij = interactions[i, j]
                val_ji = interactions[j, i]
                ij = float(val_ij) if np.isscalar(val_ij) or val_ij.size == 1 else float(val_ij.flat[0])
                ji = float(val_ji) if np.isscalar(val_ji) or val_ji.size == 1 else float(val_ji.flat[0])
                interaction_dict[f"{fname_i} x {fname_j}"] = ij + ji
        
        sorted_interactions = dict(sorted(
            interaction_dict.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ))
        
        return Explanation(
            explainer_name="TreeSHAP_Interactions",
            target_class=label_name,
            explanation_data={
                "feature_attributions": main_effects,
                "interactions": sorted_interactions,
                "interaction_matrix": interactions.tolist(),
                "feature_names": self.feature_names
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
        expected_value = self.explainer.expected_value
        
        if isinstance(expected_value, (list, np.ndarray)):
            tc = target_class if target_class is not None else 0
            return float(expected_value[min(tc, len(expected_value) - 1)])
        
        return float(expected_value)
