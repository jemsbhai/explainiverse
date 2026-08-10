# src/explainiverse/explainers/gradient/saliency.py
"""
Saliency Maps - Gradient-Based Feature Attribution.

This implementation computes the derivative of one selected output with
respect to one input. Each call uses one forward/backward gradient evaluation;
no comparative runtime or explanation-quality claim is made.

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

from numbers import Integral
from typing import List, Optional

import numpy as np

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation
from explainiverse.explainers._validation import validate_name_sequence
from explainiverse.explainers.gradient._model_state import preserve_adapter_model_eval


class SaliencyExplainer(BaseExplainer):
    """
    Saliency Maps explainer for neural networks.

    Computes attributions using the derivative of one selected model output
    with respect to the input features.

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
        absolute_value: bool = True,
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
        if not hasattr(model, "predict_with_gradients"):
            raise TypeError(
                "Model adapter must have predict_with_gradients() method. "
                "Use PyTorchAdapter for PyTorch models."
            )

        validated_features = validate_name_sequence(feature_names, name="feature_names")
        assert validated_features is not None
        self.feature_names = validated_features
        self.class_names = (
            validate_name_sequence(class_names, name="class_names") if class_names else None
        )
        self.absolute_value = absolute_value

    def _resolve_target_class(
        self, instance: np.ndarray, target_class: Optional[int]
    ) -> Optional[int]:
        """Resolve the predicted class independently of display metadata."""
        if target_class is not None:
            if isinstance(target_class, bool) or not isinstance(target_class, Integral):
                raise TypeError("target_class must be an integer output index or None")
            target_index = int(target_class)
            if target_index < 0:
                raise ValueError("target_class must be non-negative")
            if self.class_names is not None and target_index >= len(self.class_names):
                raise ValueError(f"target_class must be in [0, {len(self.class_names) - 1}]")
            return target_index

        is_classification = getattr(self.model, "task", None) == "classification"
        if not is_classification and self.class_names is None:
            return None

        with preserve_adapter_model_eval(self.model, preserve_gradients=False):
            predictions = np.asarray(self.model.predict(instance.reshape(1, -1)))
        if predictions.ndim != 2 or predictions.shape[0] != 1:
            raise ValueError(
                "Classification predictions must have shape (1, n_classes) "
                f"when resolving an attribution target; got {predictions.shape}"
            )
        resolved = int(np.argmax(predictions[0]))
        if self.class_names is not None and resolved >= len(self.class_names):
            raise ValueError("class_names length does not match the model prediction width")
        return resolved

    def _prepare_instance(self, instance: np.ndarray) -> np.ndarray:
        """Validate one complete flat feature vector without structural flattening."""
        raw = np.asarray(instance)
        if raw.ndim != 1:
            raise ValueError(
                "SaliencyExplainer supports one-dimensional flat feature vectors; "
                f"got shape {raw.shape}. Spatial image tensors are not supported."
            )
        if raw.size != len(self.feature_names):
            raise ValueError(
                "instance feature count must match feature_names exactly; "
                f"got {raw.size} values and {len(self.feature_names)} names"
            )
        if not np.isrealobj(raw):
            raise ValueError("instance must contain only finite real values")
        try:
            prepared = raw.astype(np.float32, copy=True)
        except (TypeError, ValueError) as error:
            raise TypeError("instance must contain real numeric values") from error
        if not np.isfinite(prepared).all():
            raise ValueError("instance must contain only finite real values")
        return prepared

    def _input_gradients(self, instance: np.ndarray, target_class: Optional[int]) -> np.ndarray:
        """Return one gradient per declared flat input feature."""
        with preserve_adapter_model_eval(self.model):
            _, gradients = self.model.predict_with_gradients(
                instance.reshape(1, -1), target_class=target_class
            )
        gradients = np.asarray(gradients)
        expected_shape = (1, instance.size)
        if gradients.shape != expected_shape:
            raise ValueError(
                "predict_with_gradients returned the wrong gradient shape; "
                f"expected {expected_shape}, got {gradients.shape}"
            )
        if not np.isrealobj(gradients) or not np.all(np.isfinite(gradients)):
            raise ValueError("input gradients must contain only finite real values")
        return gradients[0]

    def _score_space_metadata(self) -> dict:
        """Describe the score space used for the latest gradient call."""
        return {
            "score_space": getattr(self.model, "last_gradient_output_space", "unknown")
            or "unknown",
            "input_contract": "flat_feature_vector",
        }

    def _compute_saliency(
        self, instance: np.ndarray, target_class: Optional[int] = None, method: str = "saliency"
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
        gradients = self._input_gradients(instance, target_class)

        # Apply method
        if method == "saliency":
            attributions = gradients
        elif method == "input_times_gradient":
            attributions = instance * gradients
        else:
            raise ValueError(
                f"Unknown method: '{method}'. " f"Use 'saliency' or 'input_times_gradient'."
            )

        # Apply absolute value if configured
        if self.absolute_value and method == "saliency":
            attributions = np.abs(attributions)

        return attributions

    def explain(
        self, instance: np.ndarray, target_class: Optional[int] = None, method: str = "saliency"
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
        instance = self._prepare_instance(instance)

        target_class = self._resolve_target_class(instance, target_class)

        # Compute saliency
        attributions = self._compute_saliency(instance, target_class, method)

        # Build attributions dict
        attributions_dict = {
            fname: float(attributions[i]) for i, fname in enumerate(self.feature_names)
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
            "absolute_value": self.absolute_value if method == "saliency" else False,
        }

        return Explanation(
            explainer_name=explainer_name,
            target_class=label_name,
            explanation_data=explanation_data,
            feature_names=self.feature_names,
            metadata=self._score_space_metadata(),
        )

    def explain_batch(
        self, X: np.ndarray, target_class: Optional[int] = None, method: str = "saliency"
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
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        elif X.ndim != 2:
            raise ValueError(
                "X must have shape (n_samples, n_features); spatial image "
                f"tensors are not supported, got {X.shape}"
            )
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one instance")

        return [
            self.explain(X[i], target_class=target_class, method=method) for i in range(X.shape[0])
        ]

    def compute_all_variants(
        self, instance: np.ndarray, target_class: Optional[int] = None
    ) -> dict:
        """
        Compute all saliency variants for comparison.

        This returns several algebraic transformations for side-by-side
        inspection; it does not rank their explanation quality.

        Args:
            instance: Input instance.
            target_class: Target class for gradient computation.

        Returns:
            Dictionary containing:
                - saliency_absolute: |∂f/∂x|
                - saliency_signed: ∂f/∂x
                - input_times_gradient: x ⊙ ∂f/∂x
        """
        instance = self._prepare_instance(instance)

        target_class = self._resolve_target_class(instance, target_class)

        # Compute gradient (only once)
        gradients = self._input_gradients(instance, target_class)

        return {
            "saliency_absolute": np.abs(gradients).tolist(),
            "saliency_signed": gradients.tolist(),
            "input_times_gradient": (instance * gradients).tolist(),
            "feature_names": self.feature_names,
            "target_class": target_class,
        }
