# src/explainiverse/explainers/attribution/lime_wrapper.py
"""
LIME Explainer - Local Interpretable Model-agnostic Explanations.

LIME explains individual predictions by fitting a simple interpretable
model (linear regression) to perturbed samples around the instance.

Reference:
    Ribeiro, M.T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?":
    Explaining the Predictions of Any Classifier. KDD 2016.
"""

import numpy as np
from lime.lime_tabular import LimeTabularExplainer

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation


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
    """

    def __init__(self, model, training_data, feature_names, class_names, mode="classification"):
        """
        Initialize the LIME explainer.
        
        Args:
            model: A model adapter (implements .predict()).
            training_data: The data used to initialize LIME (2D np.ndarray).
                          Used to compute statistics for perturbation generation.
            feature_names: List of feature names.
            class_names: List of class names.
            mode: 'classification' or 'regression'.
        """
        super().__init__(model)
        self.feature_names = list(feature_names)
        self.class_names = list(class_names)
        self.mode = mode

        self.explainer = LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            class_names=class_names,
            mode=mode
        )

    def explain(self, instance, num_features=5, top_labels=1):
        """
        Generate a local explanation for the given instance.

        Args:
            instance: 1D numpy array (single row) to explain
            num_features: Number of top features to include in explanation
            top_labels: Number of top predicted labels to explain

        Returns:
            Explanation object with feature attributions
        """
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
            explanation_data={"feature_attributions": attributions}
        )
