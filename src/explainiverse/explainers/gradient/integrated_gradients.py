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
from typing import List, Optional, Union, Callable

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation


class IntegratedGradientsExplainer(BaseExplainer):
    """
    Integrated Gradients explainer for neural networks.
    
    Computes attributions by integrating gradients along the path from
    a baseline (default: zero vector) to the input. The integral is
    approximated using the Riemann sum.
    
    Attributes:
        model: Model adapter with predict_with_gradients() method
        feature_names: List of feature names
        class_names: List of class names (for classification)
        n_steps: Number of steps for integral approximation
        baseline: Baseline input (default: zeros)
        method: Integration method ("riemann_middle", "riemann_left", "riemann_right", "riemann_trapezoid")
    """
    
    def __init__(
        self,
        model,
        feature_names: List[str],
        class_names: Optional[List[str]] = None,
        n_steps: int = 50,
        baseline: Optional[np.ndarray] = None,
        method: str = "riemann_middle"
    ):
        """
        Initialize the Integrated Gradients explainer.
        
        Args:
            model: A model adapter with predict_with_gradients() method.
                   Use PyTorchAdapter for PyTorch models.
            feature_names: List of input feature names.
            class_names: List of class names (for classification tasks).
            n_steps: Number of steps for approximating the integral.
                    More steps = more accurate but slower. Default: 50.
            baseline: Baseline input for comparison. If None, uses zeros.
                     Can also be "random" for random baseline or a callable.
            method: Integration method:
                   - "riemann_middle": Middle Riemann sum (default, most accurate)
                   - "riemann_left": Left Riemann sum
                   - "riemann_right": Right Riemann sum
                   - "riemann_trapezoid": Trapezoidal rule
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
        self.n_steps = n_steps
        self.baseline = baseline
        self.method = method
    
    def _get_baseline(self, instance: np.ndarray) -> np.ndarray:
        """Get the baseline for a given input shape."""
        if self.baseline is None:
            # Default: zero baseline
            return np.zeros_like(instance)
        elif isinstance(self.baseline, str) and self.baseline == "random":
            # Random baseline (useful for images)
            return np.random.uniform(
                low=instance.min(),
                high=instance.max(),
                size=instance.shape
            ).astype(instance.dtype)
        elif callable(self.baseline):
            return self.baseline(instance)
        else:
            return np.array(self.baseline).reshape(instance.shape)
    
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
        
        The integral is approximated as:
        IG_i = (x_i - x'_i) * sum_{k=1}^{m} grad_i(x' + k/m * (x - x')) / m
        
        where x is the input, x' is the baseline, and m is n_steps.
        """
        # Get interpolation points
        alphas = self._get_interpolation_alphas()
        
        # Compute path from baseline to input
        # Shape: (n_steps, n_features)
        delta = instance - baseline
        interpolated_inputs = baseline + alphas[:, np.newaxis] * delta
        
        # Compute gradients at each interpolation point
        all_gradients = []
        for interp_input in interpolated_inputs:
            _, gradients = self.model.predict_with_gradients(
                interp_input.reshape(1, -1),
                target_class=target_class
            )
            all_gradients.append(gradients.flatten())
        
        all_gradients = np.array(all_gradients)  # Shape: (n_steps, n_features)
        
        # Approximate the integral
        if self.method == "riemann_trapezoid":
            # Trapezoidal rule: (f(0) + 2*f(1) + ... + 2*f(n-1) + f(n)) / (2n)
            weights = np.ones(self.n_steps + 1)
            weights[0] = 0.5
            weights[-1] = 0.5
            avg_gradients = np.average(all_gradients, axis=0, weights=weights)
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
            instance: 1D numpy array of input features.
            target_class: For classification, which class to explain.
                         If None, uses the predicted class.
            baseline: Override the default baseline for this explanation.
            return_convergence_delta: If True, include the convergence delta
                                     (difference between sum of attributions
                                     and prediction difference).
        
        Returns:
            Explanation object with feature attributions.
        """
        instance = np.array(instance).flatten().astype(np.float32)
        
        # Get baseline
        if baseline is not None:
            bl = np.array(baseline).flatten().astype(np.float32)
        else:
            bl = self._get_baseline(instance)
        
        # Determine target class if not specified
        if target_class is None and self.class_names:
            predictions = self.model.predict(instance.reshape(1, -1))
            target_class = int(np.argmax(predictions))
        
        # Compute integrated gradients
        ig_attributions = self._compute_integrated_gradients(
            instance, bl, target_class
        )
        
        # Build attributions dict
        attributions = {
            fname: float(ig_attributions[i])
            for i, fname in enumerate(self.feature_names)
        }
        
        # Determine class name
        if self.class_names and target_class is not None:
            label_name = self.class_names[target_class]
        else:
            label_name = f"class_{target_class}" if target_class is not None else "output"
        
        explanation_data = {
            "feature_attributions": attributions,
            "attributions_raw": ig_attributions.tolist(),
            "baseline": bl.tolist(),
            "n_steps": self.n_steps,
            "method": self.method
        }
        
        # Optionally compute convergence delta
        if return_convergence_delta:
            # The sum of attributions should equal F(x) - F(baseline)
            pred_input = self.model.predict(instance.reshape(1, -1))
            pred_baseline = self.model.predict(bl.reshape(1, -1))
            
            if target_class is not None:
                pred_diff = pred_input[0, target_class] - pred_baseline[0, target_class]
            else:
                pred_diff = pred_input[0, 0] - pred_baseline[0, 0]
            
            attribution_sum = np.sum(ig_attributions)
            convergence_delta = abs(pred_diff - attribution_sum)
            
            explanation_data["convergence_delta"] = float(convergence_delta)
            explanation_data["prediction_difference"] = float(pred_diff)
            explanation_data["attribution_sum"] = float(attribution_sum)
        
        return Explanation(
            explainer_name="IntegratedGradients",
            target_class=label_name,
            explanation_data=explanation_data
        )
    
    def explain_batch(
        self,
        X: np.ndarray,
        target_class: Optional[int] = None
    ) -> List[Explanation]:
        """
        Generate explanations for multiple instances.
        
        Note: This is not optimized for batching - it processes
        instances sequentially. For large batches, consider using
        the batched gradient computation in a custom implementation.
        
        Args:
            X: 2D numpy array of instances (n_samples, n_features).
            target_class: Target class for all instances.
        
        Returns:
            List of Explanation objects.
        """
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
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
        instance = np.array(instance).flatten().astype(np.float32)
        
        all_attributions = []
        for _ in range(n_samples):
            # Create noisy baseline
            noise = np.random.normal(0, noise_scale, instance.shape).astype(np.float32)
            noisy_baseline = noise  # Noise around zero
            
            ig = self._compute_integrated_gradients(
                instance, noisy_baseline, target_class
            )
            all_attributions.append(ig)
        
        # Average attributions
        avg_attributions = np.mean(all_attributions, axis=0)
        std_attributions = np.std(all_attributions, axis=0)
        
        attributions = {
            fname: float(avg_attributions[i])
            for i, fname in enumerate(self.feature_names)
        }
        
        if self.class_names and target_class is not None:
            label_name = self.class_names[target_class]
        else:
            label_name = f"class_{target_class}" if target_class is not None else "output"
        
        return Explanation(
            explainer_name="IntegratedGradients_Smooth",
            target_class=label_name,
            explanation_data={
                "feature_attributions": attributions,
                "attributions_raw": avg_attributions.tolist(),
                "attributions_std": std_attributions.tolist(),
                "n_samples": n_samples,
                "noise_scale": noise_scale
            }
        )
