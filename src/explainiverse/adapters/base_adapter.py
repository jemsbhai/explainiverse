# src/explainiverse/adapters/base_adapter.py

from abc import ABC, abstractmethod

class BaseModelAdapter(ABC):
    """
    Abstract base class for all model adapters.
    """

    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names

    @abstractmethod
    def predict(self, data):
        """
        Returns prediction probabilities or outputs in a standard format.

        Args:
            data: Input data (single instance or batch).

        Returns:
            List or NumPy array of prediction scores.
        """
        pass
