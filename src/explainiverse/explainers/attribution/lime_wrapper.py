# src/explainiverse/explainers/attribution/lime_wrapper.py

import numpy as np
from lime.lime_tabular import LimeTabularExplainer

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation


class LimeExplainer(BaseExplainer):
    """
    Wrapper for LIME that conforms to the BaseExplainer API.
    """

    def __init__(self, model, training_data, feature_names, class_names, mode="classification"):
        """
        Args:
            model: A model adapter (implements .predict()).
            training_data: The data used to initialize LIME (2D np.ndarray).
            feature_names: List of feature names.
            class_names: List of class names.
            mode: 'classification' or 'regression'.
        """
        super().__init__(model)
        self.feature_names = feature_names
        self.class_names = class_names
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
            instance: 1D numpy array (single row)
            num_features: Number of top features to include
            top_labels: Number of top labels to explain

        Returns:
            Explanation object
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
