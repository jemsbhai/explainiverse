# src/explainiverse/explainers/gradient/smoothgrad.py
"""
SmoothGrad - Removing Noise by Adding Noise.

SmoothGrad reduces noise in gradient-based saliency maps by averaging
gradients computed on noisy copies of the input. This produces smoother,
more visually coherent attributions that are often easier to interpret.

Key Properties:
- Simple: Just averages gradients over noisy inputs
- Effective: Significantly reduces noise in saliency maps
- Flexible: Works with any gradient-based method
- Fast: Only requires multiple forward/backward passes (parallelizable)

Variants:
- SmoothGrad: Average of gradients
- SmoothGrad-Squared: Average of squared gradients (sharper)
- VarGrad: Variance of gradients (uncertainty quantification)

Reference:
    Smilkov, D., Thorat, N., Kim, B., Viégas, F., & Wattenberg, M. (2017).
    SmoothGrad: removing noise by adding noise.
    ICML Workshop on Visualization for Deep Learning.
    https://arxiv.org/abs/1706.03825

Example:
    from explainiverse.explainers.gradient import SmoothGradExplainer
    from explainiverse.adapters import PyTorchAdapter
    
    adapter = PyTorchAdapter(model, task="classification")
    
    explainer = SmoothGradExplainer(
        model=adapter,
        feature_names=feature_names,
        n_samples=50,
        noise_scale=0.15
    )
    
    explanation = explainer.explain(instance)
"""

import numpy as np
from typing import List, Optional

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation


class SmoothGradExplainer(BaseExplainer):
    """
    SmoothGrad explainer for neural networks.
    
    Computes attributions by averaging gradients over noisy copies of the
    input. The noise helps smooth out local fluctuations in the gradient
    landscape, producing more interpretable saliency maps.
    
    Algorithm:
        SmoothGrad(x) = (1/n) * Σ_{i=1}^{n} ∂f(x + ε_i)/∂x
        where ε_i ~ N(0, σ²I) or U(-σ, σ)
    
    Attributes:
        model: Model adapter with predict_with_gradients() method
        feature_names: List of feature names
        class_names: List of class names (for classification)
        n_samples: Number of noisy samples to average
        noise_scale: Standard deviation (Gaussian) or half-range (Uniform)
        noise_type: Type of noise distribution ("gaussian" or "uniform")
    
    Example:
        >>> explainer = SmoothGradExplainer(adapter, feature_names, n_samples=50)
        >>> explanation = explainer.explain(instance)
        >>> print(explanation.explanation_data["feature_attributions"])
    """
    
    def __init__(
        self,
        model,
        feature_names: List[str],
        class_names: Optional[List[str]] = None,
        n_samples: int = 50,
        noise_scale: float = 0.15,
        noise_type: str = "gaussian"
    ):
        """
        Initialize the SmoothGrad explainer.
        
        Args:
            model: A model adapter with predict_with_gradients() method.
                   Use PyTorchAdapter for PyTorch models.
            feature_names: List of input feature names.
            class_names: List of class names (for classification tasks).
            n_samples: Number of noisy samples to average. More samples
                      reduce variance but increase computation. Default: 50.
            noise_scale: Scale of the noise to add:
                - For "gaussian": standard deviation (default: 0.15)
                - For "uniform": half-range, noise in [-scale, scale]
                Typically set to 10-20% of the input range.
            noise_type: Type of noise distribution:
                - "gaussian": Normal distribution N(0, σ²) (default)
                - "uniform": Uniform distribution U(-σ, σ)
        
        Raises:
            TypeError: If model doesn't have predict_with_gradients method.
            ValueError: If n_samples < 1, noise_scale < 0, or invalid noise_type.
        """
        super().__init__(model)
        
        # Validate model has gradient capability
        if not hasattr(model, 'predict_with_gradients'):
            raise TypeError(
                "Model adapter must have predict_with_gradients() method. "
                "Use PyTorchAdapter for PyTorch models."
            )
        
        # Validate parameters
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        
        if noise_scale < 0:
            raise ValueError(f"noise_scale must be >= 0, got {noise_scale}")
        
        if noise_type not in ["gaussian", "uniform"]:
            raise ValueError(
                f"noise_type must be 'gaussian' or 'uniform', got '{noise_type}'"
            )
        
        self.feature_names = list(feature_names)
        self.class_names = list(class_names) if class_names else None
        self.n_samples = n_samples
        self.noise_scale = noise_scale
        self.noise_type = noise_type
    
    def _generate_noise(self, shape: tuple) -> np.ndarray:
        """
        Generate noise samples based on the configured noise type.
        
        Args:
            shape: Shape of the noise array to generate.
            
        Returns:
            Numpy array of noise samples.
        """
        if self.noise_type == "gaussian":
            return np.random.normal(0, self.noise_scale, shape).astype(np.float32)
        else:  # uniform
            return np.random.uniform(
                -self.noise_scale, 
                self.noise_scale, 
                shape
            ).astype(np.float32)
    
    def _compute_smoothgrad(
        self,
        instance: np.ndarray,
        target_class: Optional[int] = None,
        method: str = "smoothgrad",
        absolute_value: bool = False
    ) -> tuple:
        """
        Compute SmoothGrad attributions for a single instance.
        
        Args:
            instance: Input instance (1D array).
            target_class: Target class for gradient computation.
            method: Aggregation method:
                - "smoothgrad": Average of gradients (default)
                - "smoothgrad_squared": Average of squared gradients
                - "vargrad": Variance of gradients
            absolute_value: If True, take absolute value of final attributions.
            
        Returns:
            Tuple of (attributions, std_attributions) arrays.
        """
        instance = instance.flatten().astype(np.float32)
        
        # Collect gradients for all noisy samples
        all_gradients = []
        
        for _ in range(self.n_samples):
            # Add noise to input
            if self.noise_scale > 0:
                noise = self._generate_noise(instance.shape)
                noisy_input = instance + noise
            else:
                noisy_input = instance.copy()
            
            # Compute gradient
            _, gradients = self.model.predict_with_gradients(
                noisy_input.reshape(1, -1),
                target_class=target_class
            )
            all_gradients.append(gradients.flatten())
        
        all_gradients = np.array(all_gradients)  # Shape: (n_samples, n_features)
        
        # Compute attributions based on method
        if method == "smoothgrad":
            attributions = np.mean(all_gradients, axis=0)
            std_attributions = np.std(all_gradients, axis=0)
        elif method == "smoothgrad_squared":
            # Average of squared gradients
            squared_gradients = all_gradients ** 2
            attributions = np.mean(squared_gradients, axis=0)
            std_attributions = np.std(squared_gradients, axis=0)
        elif method == "vargrad":
            # Variance of gradients
            attributions = np.var(all_gradients, axis=0)
            std_attributions = np.zeros_like(attributions)  # No std for variance
        else:
            raise ValueError(
                f"Unknown method: '{method}'. "
                f"Use 'smoothgrad', 'smoothgrad_squared', or 'vargrad'."
            )
        
        # Apply absolute value if requested
        if absolute_value:
            attributions = np.abs(attributions)
        
        return attributions, std_attributions
    
    def explain(
        self,
        instance: np.ndarray,
        target_class: Optional[int] = None,
        method: str = "smoothgrad",
        absolute_value: bool = False
    ) -> Explanation:
        """
        Generate SmoothGrad explanation for an instance.
        
        Args:
            instance: 1D numpy array of input features.
            target_class: For classification, which class to explain.
                         If None, uses the predicted class.
            method: Aggregation method:
                - "smoothgrad": Average of gradients (default)
                - "smoothgrad_squared": Average of squared gradients (sharper)
                - "vargrad": Variance of gradients (uncertainty)
            absolute_value: If True, return absolute values of attributions.
                           Useful for feature importance without direction.
        
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
        
        # Compute SmoothGrad
        attributions, std_attributions = self._compute_smoothgrad(
            instance, target_class, method, absolute_value
        )
        
        # Build attributions dict
        attributions_dict = {
            fname: float(attributions[i])
            for i, fname in enumerate(self.feature_names)
        }
        
        # Determine explainer name based on method
        if method == "smoothgrad":
            explainer_name = "SmoothGrad"
        elif method == "smoothgrad_squared":
            explainer_name = "SmoothGrad_Squared"
        elif method == "vargrad":
            explainer_name = "VarGrad"
        else:
            explainer_name = f"SmoothGrad_{method}"
        
        # Determine class name
        if self.class_names and target_class is not None:
            label_name = self.class_names[target_class]
        else:
            label_name = f"class_{target_class}" if target_class is not None else "output"
        
        explanation_data = {
            "feature_attributions": attributions_dict,
            "attributions_raw": attributions.tolist(),
            "attributions_std": std_attributions.tolist(),
            "n_samples": self.n_samples,
            "noise_scale": self.noise_scale,
            "noise_type": self.noise_type,
            "method": method,
            "absolute_value": absolute_value
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
        method: str = "smoothgrad",
        absolute_value: bool = False
    ) -> List[Explanation]:
        """
        Generate explanations for multiple instances.
        
        Args:
            X: 2D numpy array of instances (n_samples, n_features),
               or 1D array for single instance.
            target_class: Target class for all instances. If None,
                         uses predicted class for each instance.
            method: Aggregation method (see explain()).
            absolute_value: If True, return absolute values.
        
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
            self.explain(
                X[i], 
                target_class=target_class,
                method=method,
                absolute_value=absolute_value
            )
            for i in range(X.shape[0])
        ]
    
    def compute_with_baseline_comparison(
        self,
        instance: np.ndarray,
        target_class: Optional[int] = None
    ) -> dict:
        """
        Compare SmoothGrad with raw gradient for analysis.
        
        Useful for understanding the smoothing effect and validating
        that SmoothGrad is reducing noise appropriately.
        
        Args:
            instance: Input instance.
            target_class: Target class for gradient computation.
        
        Returns:
            Dictionary containing:
                - smoothgrad: SmoothGrad attributions
                - raw_gradient: Single gradient (no noise)
                - smoothgrad_squared: Squared variant
                - vargrad: Variance of gradients
                - correlation: Correlation between smoothgrad and raw
        """
        instance = np.array(instance).flatten().astype(np.float32)
        
        # Determine target class
        if target_class is None and self.class_names:
            predictions = self.model.predict(instance.reshape(1, -1))
            target_class = int(np.argmax(predictions))
        
        # Raw gradient (no noise)
        _, raw_gradient = self.model.predict_with_gradients(
            instance.reshape(1, -1),
            target_class=target_class
        )
        raw_gradient = raw_gradient.flatten()
        
        # SmoothGrad variants
        smoothgrad, _ = self._compute_smoothgrad(instance, target_class, "smoothgrad")
        smoothgrad_squared, _ = self._compute_smoothgrad(instance, target_class, "smoothgrad_squared")
        vargrad, _ = self._compute_smoothgrad(instance, target_class, "vargrad")
        
        # Compute correlation
        correlation = np.corrcoef(smoothgrad, raw_gradient)[0, 1]
        
        return {
            "smoothgrad": smoothgrad.tolist(),
            "raw_gradient": raw_gradient.tolist(),
            "smoothgrad_squared": smoothgrad_squared.tolist(),
            "vargrad": vargrad.tolist(),
            "correlation": float(correlation),
            "n_samples": self.n_samples,
            "noise_scale": self.noise_scale
        }
    
    def adaptive_noise_scale(
        self,
        instance: np.ndarray,
        percentile: float = 15.0
    ) -> float:
        """
        Compute adaptive noise scale based on input statistics.
        
        The original SmoothGrad paper suggests using noise scale
        proportional to the input range. This method computes an
        appropriate scale based on the instance.
        
        Args:
            instance: Input instance.
            percentile: Percentage of input range to use as noise scale.
                       Default: 15% (recommended in paper).
        
        Returns:
            Recommended noise scale for this instance.
        """
        instance = np.array(instance).flatten()
        input_range = instance.max() - instance.min()
        
        # Avoid zero scale for constant inputs
        if input_range == 0:
            input_range = np.abs(instance).max()
        if input_range == 0:
            input_range = 1.0
        
        return float(input_range * percentile / 100.0)
