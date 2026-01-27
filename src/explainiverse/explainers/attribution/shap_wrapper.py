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
            Explanation object with feature attributions
        """
        instance = np.asarray(instance)
        original_instance = instance.copy()
        
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)
        
        shap_values = self.explainer.shap_values(instance)

        if isinstance(shap_values, list):
            # Multi-class: list of arrays, one per class
            predicted_probs = self.model.predict(instance)[0]
            top_indices = np.argsort(predicted_probs)[-top_labels:][::-1]
            label_index = int(top_indices[0])
            label_name = self.class_names[label_index]
            class_shap = shap_values[label_index][0]
        else:
            # Single-class (regression or binary classification)
            label_index = 0
            label_name = self.class_names[0] if self.class_names else "class_0"
            class_shap = shap_values[0] if shap_values.ndim > 1 else shap_values

        # Build attributions dict
        flat_shap = np.array(class_shap).flatten()
        attributions = {
            fname: float(flat_shap[i])
            for i, fname in enumerate(self.feature_names)
        }
        
        # Get expected value
        if isinstance(self.explainer.expected_value, (list, np.ndarray)):
            expected_val = float(self.explainer.expected_value[label_index])
        else:
            expected_val = float(self.explainer.expected_value)

        return Explanation(
            explainer_name="SHAP",
            target_class=label_name,
            explanation_data={
                "feature_attributions": attributions,
                "shap_values_raw": flat_shap.tolist(),
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
