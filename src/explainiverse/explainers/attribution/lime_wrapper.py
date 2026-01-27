# src/explainiverse/explainers/attribution/lime_wrapper.py
"""
LIME Explainer - Local Interpretable Model-agnostic Explanations.

LIME explains individual predictions by fitting a simple interpretable
model (linear regression) to perturbed samples around the instance.

Reference:
    Ribeiro, M.T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?":
    Explaining the Predictions of Any Classifier. KDD 2016.
    https://arxiv.org/abs/1602.04938
"""

import numpy as np
from typing import List, Optional

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation

# Lazy import check - don't import lime at module level
_LIME_AVAILABLE = None


def _check_lime_available():
    """Check if LIME is available and raise ImportError if not."""
    global _LIME_AVAILABLE
    
    if _LIME_AVAILABLE is None:
        try:
            import lime
            _LIME_AVAILABLE = True
        except ImportError:
            _LIME_AVAILABLE = False
    
    if not _LIME_AVAILABLE:
        raise ImportError(
            "LIME is required for LimeExplainer. "
            "Install it with: pip install lime"
        )


class LimeExplainer(BaseExplainer):
    """
    LIME explainer for local, model-agnostic explanations.
    
    LIME (Local Interpretable Model-agnostic Explanations) explains individual
    predictions by approximating the model locally with an interpretable model.
    It generates perturbed samples around the instance and fits a weighted
    linear model to understand feature contributions.
    
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
        mode: str = "classification"
    ):
        """
        Initialize the LIME explainer.
        
        Args:
            model: A model adapter (implements .predict()).
            training_data: The data used to initialize LIME (2D np.ndarray).
                          Used to compute statistics for perturbation generation.
            feature_names: List of feature names.
            class_names: List of class names.
            mode: 'classification' or 'regression'.
            
        Raises:
            ImportError: If lime package is not installed.
        """
        # Check availability before importing
        _check_lime_available()
        
        # Import after check passes
        from lime.lime_tabular import LimeTabularExplainer
        
        super().__init__(model)
        self.feature_names = list(feature_names)
        self.class_names = list(class_names)
        self.mode = mode
        self.training_data = np.asarray(training_data)

        self.explainer = LimeTabularExplainer(
            training_data=self.training_data,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode=mode
        )

    def explain(
        self,
        instance: np.ndarray,
        num_features: int = 5,
        top_labels: int = 1
    ) -> Explanation:
        """
        Generate a local explanation for the given instance.

        Args:
            instance: 1D numpy array (single row) to explain
            num_features: Number of top features to include in explanation
            top_labels: Number of top predicted labels to explain

        Returns:
            Explanation object with feature attributions
        """
        instance = np.asarray(instance).flatten()
        
        lime_exp = self.explainer.explain_instance(
            data_row=instance,
            predict_fn=self.model.predict,
            num_features=num_features,
            top_labels=top_labels
        )

        label_index = lime_exp.top_labels[0]
        label_name = self.class_names[label_index]
        attributions = dict(lime_exp.as_list(label=label_index))

        return Explanation(
            explainer_name="LIME",
            target_class=label_name,
            explanation_data={"feature_attributions": attributions},
            feature_names=self.feature_names
        )
    
    def explain_batch(
        self,
        X: np.ndarray,
        num_features: int = 5,
        top_labels: int = 1
    ) -> List[Explanation]:
        """
        Generate explanations for multiple instances.
        
        Args:
            X: 2D numpy array of instances
            num_features: Number of features per explanation
            top_labels: Number of top labels to explain
            
        Returns:
            List of Explanation objects
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        return [
            self.explain(X[i], num_features=num_features, top_labels=top_labels)
            for i in range(X.shape[0])
        ]
