# src/explainiverse/explainers/attribution/shap_wrapper.py
"""
SHAP Explainer - SHapley Additive exPlanations.

SHAP values provide a unified measure of feature importance based on
game-theoretic Shapley values, offering both local and global interpretability.

Reference:
    Lundberg, S.M. & Lee, S.I. (2017). A Unified Approach to Interpreting 
    Model Predictions. NeurIPS 2017.
    https://arxiv.org/abs/1705.07874
"""

import numpy as np
from typing import List, Optional

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation

# Lazy import check - don't import shap at module level
_SHAP_AVAILABLE = None


def _check_shap_available():
    """Check if SHAP is available and raise ImportError if not."""
    global _SHAP_AVAILABLE
    
    if _SHAP_AVAILABLE is None:
        try:
            import shap
            _SHAP_AVAILABLE = True
        except ImportError:
            _SHAP_AVAILABLE = False
    
    if not _SHAP_AVAILABLE:
        raise ImportError(
            "SHAP is required for ShapExplainer. "
            "Install it with: pip install shap"
        )


class ShapExplainer(BaseExplainer):
    """
    SHAP explainer (KernelSHAP-based) for model-agnostic explanations.
    
    KernelSHAP is a model-agnostic method that approximates SHAP values
    using a weighted linear regression. It works with any model that
    provides predictions.
    
    Attributes:
        model: Model adapter with .predict() method
        feature_names: List of feature names
        class_names: List of class labels
        explainer: The underlying SHAP KernelExplainer
    
    Example:
        >>> from explainiverse.explainers.attribution import ShapExplainer
        >>> explainer = ShapExplainer(
        ...     model=adapter,
        ...     background_data=X_train[:100],
        ...     feature_names=feature_names,
        ...     class_names=class_names
        ... )
        >>> explanation = explainer.explain(X_test[0])
    """

    def __init__(
        self,
        model,
        background_data: np.ndarray,
        feature_names: List[str],
        class_names: List[str]
    ):
        """
        Initialize the SHAP explainer.
        
        Args:
            model: A model adapter with a .predict method.
            background_data: A 2D numpy array used as SHAP background distribution.
                            Typically a representative sample of training data.
            feature_names: List of feature names.
            class_names: List of class labels.
            
        Raises:
            ImportError: If shap package is not installed.
        """
        # Check availability before importing
        _check_shap_available()
        
        # Import after check passes
        import shap as shap_module
        
        super().__init__(model)
        self.feature_names = list(feature_names)
        self.class_names = list(class_names)
        self.background_data = np.asarray(background_data)
        
        self.explainer = shap_module.KernelExplainer(
            model.predict,
            self.background_data
        )

    def _extract_class_shap_values(
        self,
        shap_values,
        instance: np.ndarray,
        top_labels: int,
    ):
        """
        Extract per-feature SHAP values for the target class.

        KernelSHAP can return results in multiple formats depending on
        the model and SHAP version:

        1. list of arrays — one (n_samples, n_features) array per class.
           Common with older SHAP versions and multi-output models.
        2. 3D ndarray (n_samples, n_features, n_classes) — multi-class.
        3. 2D ndarray (n_samples, n_features) — binary or regression.
        4. 1D ndarray (n_features,) — single sample, single output.

        Args:
            shap_values: Raw output from KernelExplainer.shap_values()
            instance: The input instance (2D, shape (1, n_features))
            top_labels: Number of top labels to consider

        Returns:
            (class_shap, label_index, label_name) where class_shap is
            a 1D array of shape (n_features,).
        """
        n_features = len(self.feature_names)

        # Determine the predicted class
        predicted_probs = self.model.predict(instance)[0]
        top_indices = np.argsort(predicted_probs)[-top_labels:][::-1]
        label_index = int(top_indices[0])

        if label_index < len(self.class_names):
            label_name = self.class_names[label_index]
        else:
            label_name = f"class_{label_index}"

        # Case 1: list of arrays — one per class
        if isinstance(shap_values, list):
            class_shap = np.asarray(shap_values[label_index])
            # class_shap is (n_samples, n_features); take first sample
            if class_shap.ndim == 2:
                class_shap = class_shap[0]
            return class_shap, label_index, label_name

        # From here, shap_values is an ndarray
        shap_arr = np.asarray(shap_values)

        # Case 2: 3D — (n_samples, n_features, n_classes)
        if shap_arr.ndim == 3:
            class_shap = shap_arr[0, :, label_index]  # (n_features,)
            return class_shap, label_index, label_name

        # Case 3: 2D — (n_samples, n_features)
        if shap_arr.ndim == 2:
            class_shap = shap_arr[0]  # (n_features,)
            return class_shap, label_index, label_name

        # Case 4: 1D — (n_features,)
        if shap_arr.ndim == 1:
            return shap_arr, label_index, label_name

        raise ValueError(
            f"Unexpected SHAP values shape: {shap_arr.shape}. "
            f"Expected list, 3D (samples, features, classes), "
            f"2D (samples, features), or 1D (features,)."
        )

    def explain(
        self,
        instance: np.ndarray,
        top_labels: int = 1
    ) -> Explanation:
        """
        Generate SHAP explanation for a single instance.

        Args:
            instance: 1D numpy array of input features.
            top_labels: Number of top classes to explain (default: 1)

        Returns:
            Explanation object with feature attributions keyed by original
            feature names. The shap_values_raw list has exactly
            len(feature_names) entries — one per feature for the target class.
        """
        instance = np.asarray(instance)

        if instance.ndim == 1:
            instance = instance.reshape(1, -1)

        shap_values = self.explainer.shap_values(instance)

        class_shap, label_index, label_name = self._extract_class_shap_values(
            shap_values, instance, top_labels
        )

        # Validate shape before building attributions
        class_shap = np.asarray(class_shap).ravel()
        if len(class_shap) != len(self.feature_names):
            raise ValueError(
                f"SHAP values length ({len(class_shap)}) does not match "
                f"number of features ({len(self.feature_names)}). "
                f"Raw shap_values type={type(shap_values)}, "
                f"shape={getattr(shap_values, 'shape', 'list')}"
            )

        # Build attributions dict keyed by original feature names
        attributions = {
            fname: float(class_shap[i])
            for i, fname in enumerate(self.feature_names)
        }

        # Get expected value
        if isinstance(self.explainer.expected_value, (list, np.ndarray)):
            ev = np.asarray(self.explainer.expected_value)
            if label_index < len(ev):
                expected_val = float(ev[label_index])
            else:
                expected_val = float(ev[0])
        else:
            expected_val = float(self.explainer.expected_value)

        return Explanation(
            explainer_name="SHAP",
            target_class=label_name,
            explanation_data={
                "feature_attributions": attributions,
                "shap_values_raw": class_shap.tolist(),
                "expected_value": expected_val
            },
            feature_names=self.feature_names
        )

    def explain_batch(
        self,
        X: np.ndarray,
        top_labels: int = 1
    ) -> List[Explanation]:
        """
        Generate explanations for multiple instances.
        
        Args:
            X: 2D numpy array of instances
            top_labels: Number of top labels to explain
            
        Returns:
            List of Explanation objects
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        return [
            self.explain(X[i], top_labels=top_labels)
            for i in range(X.shape[0])
        ]
