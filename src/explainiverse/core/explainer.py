# src/explainiverse/core/explainer.py

from abc import ABC, abstractmethod

class BaseExplainer(ABC):
    """
    Abstract base class for all explainers in Explainiverse.
    """

    def __init__(self, model):
        """
        Initialize with a model adapter or raw model.

        Args:
            model: A wrapped ML model with a standardized `predict` method.
        """
        self.model = model

    @abstractmethod
    def explain(self, instance, **kwargs):
        """
        Generate an explanation for a single input instance.

        Args:
            instance: The input to explain (e.g., feature vector, image, text).
            **kwargs: Optional method-specific parameters.

        Returns:
            An Explanation object or dict.
        """
        pass
