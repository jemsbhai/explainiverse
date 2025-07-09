# src/explainiverse/adapters/sklearn_adapter.py

import numpy as np
from .base_adapter import BaseModelAdapter

class SklearnAdapter(BaseModelAdapter):
    """
    Adapter for Scikit-learn classifiers.
    """

    def __init__(self, model, feature_names=None, class_names=None):
        super().__init__(model, feature_names)
        self.class_names = class_names

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Returns prediction probabilities.

        Args:
            data: A 2D numpy array of inputs.

        Returns:
            Array of shape (n_samples, n_classes).
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(data)
        else:
            preds = self.model.predict(data)
            if self.class_names:
                return np.eye(len(self.class_names))[preds]
            else:
                return preds.reshape(-1, 1)  # regression: raw outputs
