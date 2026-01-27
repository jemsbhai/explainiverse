# src/explainiverse/explainers/gradient/saliency.py
"""
Saliency Maps - Gradient-Based Feature Attribution.

Saliency Maps compute feature attributions using the gradient of the output
with respect to the input. This is one of the simplest and fastest gradient-based
attribution methods, requiring only a single forward and backward pass.

Key Properties:
- Simple: Just compute the gradient of output w.r.t. input
- Fast: Single forward + backward pass
- Foundation: Base method that other gradient methods build upon
- Variants: Absolute saliency, signed saliency, input × gradient

Variants:
- Saliency (absolute): |∂f(x)/∂x| - magnitude of sensitivity
- Saliency (signed): ∂f(x)/∂x - direction and magnitude
- Input × Gradient: x ⊙ ∂f(x)/∂x - scaled by input values

Reference:
    Simonyan, K., Vedaldi, A., & Zisserman, A. (2014).
    Deep Inside Convolutional Networks: Visualising Image Classification
    Models and Saliency Maps.
    ICLR Workshop 2014.
    https://arxiv.org/abs/1312.6034

Example:
    from explainiverse.explainers.gradient import SaliencyExplainer
    from explainiverse.adapters import PyTorchAdapter
    
    adapter = PyTorchAdapter(model, task="classification")
    
    explainer = SaliencyExplainer(
        model=adapter,
        feature_names=feature_names
    )
    
    explanation = explainer.explain(instance)
"""

import numpy as np
from typing import List, Optional

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation


class SaliencyExplainer(BaseExplainer):
    """
    Saliency Maps explainer for neural networks.
    
    Computes attributions using the gradient of the model output with respect
    to the input features. This is the simplest gradient-based attribution
    method and serves as the foundation for more sophisticated techniques.
    
    Algorithm:
        Saliency(x) = ∂f(x)/∂x  (signed)
        Saliency(x) = |∂f(x)/∂x|  (absolute, default)
        InputTimesGradient(x) = x ⊙ ∂f(x)/∂x
    
    Attributes:
        model: Model adapter with predict_with_gradients() method
        feature_names: List of feature names
        class_names: List of class names (for classification)
        absolute_value: Whether to take absolute value of gradients
    
    Example:
        >>> explainer = SaliencyExplainer(adapter, feature_names)
        >>> explanation = explainer.explain(instance)
        >>> print(explanation.explanation_data["feature_attributions"])
    """
    
    def __init__(
        self,
        model,
        feature_names: List[str],
        class_names: Optional[List[str]] = None,
        absolute_value: bool = True
    ):
        """
        Initialize the Saliency explainer.
        
        Args:
            model: A model adapter with predict_with_gradients() method.
                   Use PyTorchAdapter for PyTorch models.
            feature_names: List of input feature names.
            class_names: List of class names (for classification tasks).
            absolute_value: If True (default), return absolute value of
                          gradients. Set to False for signed saliency.
        
        Raises:
            TypeError: If model doesn't have predict_with_gradients method.
        """
        super().__init__(model)
        
        # Validate model has gradient capability
        if not hasattr(model, 'predict_with_gradients'):
            raise TypeError(
                "Model adapter must have predict_with_gradients() method. "
                "Use PyTorchAdapter for PyTorch models."
            )
        
        self.feature_names = list(feature_names)
        self.class_names = list(class_names) if class_names else None
        self.absolute_value = absolute_value
    
    def _compute_saliency(
        self,
        instance: np.ndarray,
        target_class: Optional[int] = None,
        method: str = "saliency"
    ) -> np.ndarray:
        """
        Compute saliency attributions for a single instance.
        
        Args:
            instance: Input instance (1D array).
            target_class: Target class for gradient computation.
            method: Attribution method:
                - "saliency": Raw gradient (default)
                - "input_times_gradient": Gradient multiplied by input
            
        Returns:
            Array of attribution scores for each input feature.
        """
        instance = instance.flatten().astype(np.float32)
        
        # Compute gradient
        _, gradients = self.model.predict_with_gradients(
            instance.reshape(1, -1),
            target_class=target_class
        )
        gradients = gradients.flatten()
        
        # Apply method
        if method == "saliency":
            attributions = gradients
        elif method == "input_times_gradient":
            attributions = instance * gradients
        else:
            raise ValueError(
                f"Unknown method: '{method}'. "
                f"Use 'saliency' or 'input_times_gradient'."
            )
        
        # Apply absolute value if configured
        if self.absolute_value and method == "saliency":
            attributions = np.abs(attributions)
        
        return attributions
    
    def explain(
        self,
        instance: np.ndarray,
        target_class: Optional[int] = None,
        method: str = "saliency"
    ) -> Explanation:
        """
        Generate Saliency explanation for an instance.
        
        Args:
            instance: 1D numpy array of input features.
            target_class: For classification, which class to explain.
                         If None, uses the predicted class.
            method: Attribution method:
                - "saliency": Gradient-based saliency (default)
                - "input_times_gradient": Gradient × input
        
        Returns:
            Explanation object with feature attributions.
        
        Example:
            >>> explanation = explainer.explain(instance)
            >>> print(explanation.explanation_data["feature_attributions"])
        """
        instance = np.array(instance).flatten().astype(np.float32)
        
        # Determine target class if not specified
        if target_class is None and self.class_names:
            predictions = self.model.predict(instance.reshape(1, -1))
            target_class = int(np.argmax(predictions))
        
        # Compute saliency
        attributions = self._compute_saliency(instance, target_class, method)
        
        # Build attributions dict
        attributions_dict = {
            fname: float(attributions[i])
            for i, fname in enumerate(self.feature_names)
        }
        
        # Determine explainer name based on method
        if method == "saliency":
            explainer_name = "Saliency"
        elif method == "input_times_gradient":
            explainer_name = "InputTimesGradient"
        else:
            explainer_name = f"Saliency_{method}"
        
        # Determine class name
        if self.class_names and target_class is not None:
            label_name = self.class_names[target_class]
        else:
            label_name = f"class_{target_class}" if target_class is not None else "output"
        
        explanation_data = {
            "feature_attributions": attributions_dict,
            "attributions_raw": attributions.tolist(),
            "method": method,
            "absolute_value": self.absolute_value if method == "saliency" else False
        }
        
        return Explanation(
            explainer_name=explainer_name,
            target_class=label_name,
            explanation_data=explanation_data
        )
    
    def explain_batch(
        self,
        X: np.ndarray,
        target_class: Optional[int] = None,
        method: str = "saliency"
    ) -> List[Explanation]:
        """
        Generate explanations for multiple instances.
        
        Args:
            X: 2D numpy array of instances (n_samples, n_features),
               or 1D array for single instance.
            target_class: Target class for all instances. If None,
                         uses predicted class for each instance.
            method: Attribution method (see explain()).
        
        Returns:
            List of Explanation objects.
        
        Example:
            >>> explanations = explainer.explain_batch(X_test[:10])
            >>> for exp in explanations:
            ...     print(exp.target_class)
        """
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        return [
            self.explain(X[i], target_class=target_class, method=method)
            for i in range(X.shape[0])
        ]
    
    def compute_all_variants(
        self,
        instance: np.ndarray,
        target_class: Optional[int] = None
    ) -> dict:
        """
        Compute all saliency variants for comparison.
        
        Useful for analyzing which variant provides the best explanation
        for a given instance or model architecture.
        
        Args:
            instance: Input instance.
            target_class: Target class for gradient computation.
        
        Returns:
            Dictionary containing:
                - saliency_absolute: |∂f/∂x|
                - saliency_signed: ∂f/∂x
                - input_times_gradient: x ⊙ ∂f/∂x
        """
        instance = np.array(instance).flatten().astype(np.float32)
        
        # Determine target class
        if target_class is None and self.class_names:
            predictions = self.model.predict(instance.reshape(1, -1))
            target_class = int(np.argmax(predictions))
        
        # Compute gradient (only once)
        _, gradients = self.model.predict_with_gradients(
            instance.reshape(1, -1),
            target_class=target_class
        )
        gradients = gradients.flatten()
        
        return {
            "saliency_absolute": np.abs(gradients).tolist(),
            "saliency_signed": gradients.tolist(),
            "input_times_gradient": (instance * gradients).tolist(),
            "feature_names": self.feature_names,
            "target_class": target_class
        }
