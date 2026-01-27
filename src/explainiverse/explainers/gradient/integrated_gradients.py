# src/explainiverse/explainers/gradient/integrated_gradients.py
"""
Integrated Gradients - Axiomatic Attribution for Deep Networks.

Integrated Gradients computes feature attributions by accumulating gradients
along a straight-line path from a baseline to the input. It satisfies two
key axioms:
- Sensitivity: If a feature differs between input and baseline and changes
  the prediction, it receives non-zero attribution.
- Implementation Invariance: Attributions are identical for functionally
  equivalent networks.

Reference:
    Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for
    Deep Networks. ICML 2017. https://arxiv.org/abs/1703.01365

Example:
    from explainiverse.explainers.gradient import IntegratedGradientsExplainer
    from explainiverse.adapters import PyTorchAdapter
    
    adapter = PyTorchAdapter(model, task="classification")
    
    explainer = IntegratedGradientsExplainer(
        model=adapter,
        feature_names=feature_names,
        n_steps=50
    )
    
    explanation = explainer.explain(instance)
"""

import numpy as np
from typing import List, Optional, Union, Callable, Tuple

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation


class IntegratedGradientsExplainer(BaseExplainer):
    """
    Integrated Gradients explainer for neural networks.
    
    Computes attributions by integrating gradients along the path from
    a baseline (default: zero vector) to the input. The integral is
    approximated using the Riemann sum.
    
    Supports both tabular data (1D/2D) and image data (3D/4D), preserving
    the original input shape for proper gradient computation.
    
    Attributes:
        model: Model adapter with predict_with_gradients() method
        feature_names: List of feature names (for tabular data)
        class_names: List of class names (for classification)
        n_steps: Number of steps for integral approximation
        baseline: Baseline input (default: zeros)
        method: Integration method
        input_shape: Expected input shape (inferred or specified)
    """
    
    def __init__(
        self,
        model,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        n_steps: int = 50,
        baseline: Optional[Union[np.ndarray, str, Callable]] = None,
        method: str = "riemann_middle",
        input_shape: Optional[Tuple[int, ...]] = None
    ):
        """
        Initialize the Integrated Gradients explainer.
        
        Args:
            model: A model adapter with predict_with_gradients() method.
                   Use PyTorchAdapter for PyTorch models.
            feature_names: List of input feature names. Required for tabular
                          data to create named attributions.
            class_names: List of class names (for classification tasks).
            n_steps: Number of steps for approximating the integral.
                    More steps = more accurate but slower. Default: 50.
            baseline: Baseline input for comparison:
                     - None: uses zeros
                     - "random": random baseline (useful for images)
                     - np.ndarray: specific baseline values
                     - Callable: function(instance) -> baseline
            method: Integration method:
                   - "riemann_middle": Middle Riemann sum (default, most accurate)
                   - "riemann_left": Left Riemann sum
                   - "riemann_right": Right Riemann sum
                   - "riemann_trapezoid": Trapezoidal rule
            input_shape: Expected shape of a single input (excluding batch dim).
                        If None, inferred from first explain() call.
        """
        super().__init__(model)
        
        # Validate model has gradient capability
        if not hasattr(model, 'predict_with_gradients'):
            raise TypeError(
                "Model adapter must have predict_with_gradients() method. "
                "Use PyTorchAdapter for PyTorch models."
            )
        
        self.feature_names = list(feature_names) if feature_names else None
        self.class_names = list(class_names) if class_names else None
        self.n_steps = n_steps
        self.baseline = baseline
        self.method = method
        self.input_shape = input_shape
    
    def _infer_data_type(self, instance: np.ndarray) -> str:
        """
        Infer whether input is tabular or image data.
        
        Args:
            instance: Input instance (without batch dimension)
            
        Returns:
            "tabular" for 1D data, "image" for 2D+ data
        """
        if instance.ndim == 1:
            return "tabular"
        elif instance.ndim >= 2:
            return "image"
        else:
            return "tabular"
    
    def _get_baseline(self, instance: np.ndarray) -> np.ndarray:
        """
        Get the baseline for a given input shape.
        
        Args:
            instance: Input instance (preserves shape)
            
        Returns:
            Baseline array with same shape as instance
        """
        if self.baseline is None:
            # Default: zero baseline
            return np.zeros_like(instance)
        elif isinstance(self.baseline, str):
            if self.baseline == "random":
                # Random baseline (useful for images)
                return np.random.uniform(
                    low=float(instance.min()),
                    high=float(instance.max()),
                    size=instance.shape
                ).astype(instance.dtype)
            elif self.baseline == "mean":
                # Mean value baseline
                return np.full_like(instance, instance.mean())
            else:
                raise ValueError(f"Unknown baseline type: {self.baseline}")
        elif callable(self.baseline):
            result = self.baseline(instance)
            return np.asarray(result).reshape(instance.shape)
        else:
            return np.asarray(self.baseline).reshape(instance.shape)
    
    def _get_interpolation_alphas(self) -> np.ndarray:
        """Get interpolation points based on method."""
        if self.method == "riemann_left":
            return np.linspace(0, 1 - 1/self.n_steps, self.n_steps)
        elif self.method == "riemann_right":
            return np.linspace(1/self.n_steps, 1, self.n_steps)
        elif self.method == "riemann_middle":
            return np.linspace(0.5/self.n_steps, 1 - 0.5/self.n_steps, self.n_steps)
        elif self.method == "riemann_trapezoid":
            return np.linspace(0, 1, self.n_steps + 1)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _compute_integrated_gradients(
        self,
        instance: np.ndarray,
        baseline: np.ndarray,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute integrated gradients for a single instance.
        
        Preserves input shape throughout computation for proper gradient flow.
        
        The integral is approximated as:
        IG_i = (x_i - x'_i) * sum_{k=1}^{m} grad_i(x' + k/m * (x - x')) / m
        
        where x is the input, x' is the baseline, and m is n_steps.
        
        Args:
            instance: Input instance (any shape)
            baseline: Baseline with same shape as instance
            target_class: Target class for attribution
            
        Returns:
            Attributions with same shape as instance
        """
        # Store original shape
        original_shape = instance.shape
        
        # Get interpolation points
        alphas = self._get_interpolation_alphas()
        
        # Compute path from baseline to input
        delta = instance - baseline
        
        # Collect gradients at each interpolation point
        all_gradients = []
        
        for alpha in alphas:
            # Interpolated input: baseline + alpha * (input - baseline)
            interp_input = baseline + alpha * delta
            
            # Add batch dimension for model
            if interp_input.ndim == len(original_shape):
                interp_batch = interp_input[np.newaxis, ...]
            else:
                interp_batch = interp_input
            
            # Get gradients
            _, gradients = self.model.predict_with_gradients(
                interp_batch,
                target_class=target_class
            )
            
            # Remove batch dimension if present
            if gradients.shape[0] == 1 and len(gradients.shape) > len(original_shape):
                gradients = gradients[0]
            
            all_gradients.append(gradients.reshape(original_shape))
        
        all_gradients = np.array(all_gradients)  # Shape: (n_steps, *original_shape)
        
        # Approximate the integral
        if self.method == "riemann_trapezoid":
            # Trapezoidal rule
            weights = np.ones(self.n_steps + 1)
            weights[0] = 0.5
            weights[-1] = 0.5
            # Expand weights for broadcasting
            for _ in range(len(original_shape)):
                weights = weights[:, np.newaxis]
            avg_gradients = np.sum(all_gradients * weights, axis=0) / self.n_steps
        else:
            # Standard Riemann sum: average of gradients
            avg_gradients = np.mean(all_gradients, axis=0)
        
        # Scale by input - baseline difference
        integrated_gradients = delta * avg_gradients
        
        return integrated_gradients
    
    def explain(
        self,
        instance: np.ndarray,
        target_class: Optional[int] = None,
        baseline: Optional[np.ndarray] = None,
        return_convergence_delta: bool = False
    ) -> Explanation:
        """
        Generate Integrated Gradients explanation for an instance.
        
        Args:
            instance: Input instance. Can be:
                     - 1D array for tabular data
                     - 2D array for grayscale images
                     - 3D array for color images (C, H, W)
            target_class: For classification, which class to explain.
                         If None, uses the predicted class.
            baseline: Override the default baseline for this explanation.
            return_convergence_delta: If True, include the convergence delta
                                     (difference between sum of attributions
                                     and prediction difference).
        
        Returns:
            Explanation object with feature attributions.
        """
        instance = np.asarray(instance).astype(np.float32)
        original_shape = instance.shape
        
        # Infer data type
        data_type = self._infer_data_type(instance)
        
        # Get baseline (preserves shape)
        if baseline is not None:
            bl = np.asarray(baseline).astype(np.float32).reshape(original_shape)
        else:
            bl = self._get_baseline(instance)
        
        # Determine target class if not specified
        if target_class is None and self.class_names:
            # Add batch dim for prediction
            pred_input = instance[np.newaxis, ...] if instance.ndim == len(original_shape) else instance
            predictions = self.model.predict(pred_input)
            target_class = int(np.argmax(predictions[0]))
        
        # Compute integrated gradients (preserves shape)
        ig_attributions = self._compute_integrated_gradients(
            instance, bl, target_class
        )
        
        # Build explanation data
        explanation_data = {
            "attributions_raw": ig_attributions.tolist(),
            "baseline": bl.tolist(),
            "n_steps": self.n_steps,
            "method": self.method,
            "input_shape": list(original_shape),
            "data_type": data_type
        }
        
        # For tabular data, create named attributions
        if data_type == "tabular" and self.feature_names is not None:
            flat_ig = ig_attributions.flatten()
            if len(flat_ig) == len(self.feature_names):
                attributions = {
                    fname: float(flat_ig[i])
                    for i, fname in enumerate(self.feature_names)
                }
                explanation_data["feature_attributions"] = attributions
        elif data_type == "image":
            # For images, store aggregated feature importance
            explanation_data["attribution_map"] = ig_attributions
            # Also store channel-aggregated saliency for visualization
            if ig_attributions.ndim == 3:  # (C, H, W)
                explanation_data["saliency_map"] = np.abs(ig_attributions).sum(axis=0)
            else:
                explanation_data["saliency_map"] = np.abs(ig_attributions)
        
        # Determine class name
        if self.class_names and target_class is not None:
            label_name = self.class_names[target_class]
        else:
            label_name = f"class_{target_class}" if target_class is not None else "output"
        
        # Optionally compute convergence delta
        if return_convergence_delta:
            # The sum of attributions should equal F(x) - F(baseline)
            pred_input = instance[np.newaxis, ...]
            pred_baseline = bl[np.newaxis, ...]
            
            pred_input_val = self.model.predict(pred_input)
            pred_baseline_val = self.model.predict(pred_baseline)
            
            if target_class is not None and pred_input_val.shape[-1] > 1:
                pred_diff = pred_input_val[0, target_class] - pred_baseline_val[0, target_class]
            else:
                pred_diff = pred_input_val[0, 0] - pred_baseline_val[0, 0]
            
            attribution_sum = float(np.sum(ig_attributions))
            convergence_delta = abs(float(pred_diff) - attribution_sum)
            
            explanation_data["convergence_delta"] = convergence_delta
            explanation_data["prediction_difference"] = float(pred_diff)
            explanation_data["attribution_sum"] = attribution_sum
        
        return Explanation(
            explainer_name="IntegratedGradients",
            target_class=label_name,
            explanation_data=explanation_data,
            feature_names=self.feature_names
        )
    
    def explain_batch(
        self,
        X: np.ndarray,
        target_class: Optional[int] = None
    ) -> List[Explanation]:
        """
        Generate explanations for multiple instances.
        
        Note: This processes instances sequentially. For large batches,
        consider implementing batched gradient computation.
        
        Args:
            X: Array of instances. First dimension is batch.
            target_class: Target class for all instances.
        
        Returns:
            List of Explanation objects.
        """
        X = np.asarray(X)
        
        # Handle single instance passed as array
        if X.ndim == 1:
            return [self.explain(X, target_class=target_class)]
        
        return [
            self.explain(X[i], target_class=target_class)
            for i in range(X.shape[0])
        ]
    
    def compute_attributions_with_noise(
        self,
        instance: np.ndarray,
        target_class: Optional[int] = None,
        n_samples: int = 5,
        noise_scale: float = 0.1
    ) -> Explanation:
        """
        Compute attributions averaged over noisy baselines (SmoothGrad-style).
        
        This can help reduce noise in the attributions by averaging over
        multiple baselines sampled around the zero baseline.
        
        Args:
            instance: Input instance.
            target_class: Target class for attribution.
            n_samples: Number of noisy baselines to average.
            noise_scale: Standard deviation of Gaussian noise.
        
        Returns:
            Explanation with averaged attributions.
        """
        instance = np.asarray(instance).astype(np.float32)
        original_shape = instance.shape
        
        all_attributions = []
        for _ in range(n_samples):
            # Create noisy baseline
            noise = np.random.normal(0, noise_scale, original_shape).astype(np.float32)
            noisy_baseline = noise  # Noise around zero
            
            ig = self._compute_integrated_gradients(
                instance, noisy_baseline, target_class
            )
            all_attributions.append(ig)
        
        # Average attributions
        avg_attributions = np.mean(all_attributions, axis=0)
        std_attributions = np.std(all_attributions, axis=0)
        
        # Build explanation data
        data_type = self._infer_data_type(instance)
        explanation_data = {
            "attributions_raw": avg_attributions.tolist(),
            "attributions_std": std_attributions.tolist(),
            "n_samples": n_samples,
            "noise_scale": noise_scale,
            "data_type": data_type
        }
        
        # For tabular data, create named attributions
        if data_type == "tabular" and self.feature_names is not None:
            flat_avg = avg_attributions.flatten()
            if len(flat_avg) == len(self.feature_names):
                attributions = {
                    fname: float(flat_avg[i])
                    for i, fname in enumerate(self.feature_names)
                }
                explanation_data["feature_attributions"] = attributions
        
        if self.class_names and target_class is not None:
            label_name = self.class_names[target_class]
        else:
            label_name = f"class_{target_class}" if target_class is not None else "output"
        
        return Explanation(
            explainer_name="IntegratedGradients_Smooth",
            target_class=label_name,
            explanation_data=explanation_data,
            feature_names=self.feature_names
        )
