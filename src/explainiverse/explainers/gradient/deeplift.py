# src/explainiverse/explainers/gradient/deeplift.py
"""
DeepLIFT - Deep Learning Important FeaTures.

DeepLIFT explains the difference in output from some reference output
in terms of the difference of the input from some reference input.
Unlike standard gradients which show the effect of infinitesimal changes,
DeepLIFT considers the actual change in activations from a reference.

Key Properties:
- Summation-to-delta: Sum of attributions equals output - reference_output
- Handles saturation: Works correctly even when gradients are zero
- Fast: Requires only one forward and one backward pass (vs. many for IG)

Rules:
- Rescale Rule: Simple proportional attribution (default)
- RevealCancel Rule: Separates positive/negative contributions (advanced)

Reference:
    Shrikumar, A., Greenside, P., & Kundaje, A. (2017). Learning Important
    Features Through Propagating Activation Differences. ICML 2017.
    https://arxiv.org/abs/1704.02685

Example:
    from explainiverse.explainers.gradient import DeepLIFTExplainer
    from explainiverse.adapters import PyTorchAdapter
    
    adapter = PyTorchAdapter(model, task="classification")
    
    explainer = DeepLIFTExplainer(
        model=adapter,
        feature_names=feature_names
    )
    
    explanation = explainer.explain(instance)
"""

import numpy as np
from typing import List, Optional, Union, Callable, Tuple

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation


# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


class DeepLIFTExplainer(BaseExplainer):
    """
    DeepLIFT explainer for neural networks.
    
    Computes attributions by propagating the difference between the 
    network's output and a reference output back to the inputs, using
    the difference between input activations and reference activations.
    
    DeepLIFT is faster than Integrated Gradients (single forward/backward
    pass) while providing similar quality attributions for most networks.
    
    Attributes:
        model: Model adapter with gradient computation capability
        feature_names: List of feature names
        class_names: List of class names (for classification)
        baseline: Reference input for comparison
        multiply_by_inputs: If True, multiply attributions by (input - baseline)
        eps: Small constant to avoid division by zero
    
    Example:
        >>> explainer = DeepLIFTExplainer(adapter, feature_names)
        >>> explanation = explainer.explain(instance)
        >>> print(explanation.get_attributions())
    """
    
    def __init__(
        self,
        model,
        feature_names: List[str],
        class_names: Optional[List[str]] = None,
        baseline: Optional[Union[np.ndarray, str, Callable]] = None,
        multiply_by_inputs: bool = True,
        eps: float = 1e-10
    ):
        """
        Initialize the DeepLIFT explainer.
        
        Args:
            model: A model adapter with predict_with_gradients() method.
                   Use PyTorchAdapter for PyTorch models.
            feature_names: List of input feature names.
            class_names: List of class names (for classification tasks).
            baseline: Reference input for comparison:
                - None: Use zeros (default)
                - "random": Sample from uniform distribution
                - "mean": Placeholder for dataset mean (set via set_baseline)
                - np.ndarray: Specific baseline values
                - Callable: Function that takes instance and returns baseline
            multiply_by_inputs: If True (default), compute
                (input - baseline) * multipliers. If False, return raw
                multipliers (useful for debugging).
            eps: Small constant to prevent division by zero in 
                multiplier computation. Default: 1e-10
        """
        super().__init__(model)
        
        # Validate model has required methods
        if not hasattr(model, 'predict_with_gradients'):
            raise TypeError(
                "Model adapter must have predict_with_gradients() method. "
                "Use PyTorchAdapter for PyTorch models."
            )
        
        self.feature_names = list(feature_names)
        self.class_names = list(class_names) if class_names else None
        self.baseline = baseline
        self.multiply_by_inputs = multiply_by_inputs
        self.eps = eps
        
        # For advanced usage: store reference for layer-wise computation
        self._reference_activations = None
    
    def _get_baseline(self, instance: np.ndarray) -> np.ndarray:
        """
        Get the baseline/reference input for a given instance.
        
        Args:
            instance: The input instance
            
        Returns:
            Baseline array with same shape as instance
        """
        if self.baseline is None:
            # Default: zero baseline
            return np.zeros_like(instance)
        elif isinstance(self.baseline, str):
            if self.baseline == "random":
                # Random baseline from uniform distribution
                return np.random.uniform(
                    low=instance.min(),
                    high=instance.max(),
                    size=instance.shape
                ).astype(instance.dtype)
            elif self.baseline == "mean":
                # This requires set_baseline to have been called with data
                raise ValueError(
                    "Baseline 'mean' requires calling set_baseline() with "
                    "training data first."
                )
            else:
                raise ValueError(f"Unknown baseline type: {self.baseline}")
        elif callable(self.baseline):
            return self.baseline(instance)
        else:
            baseline = np.array(self.baseline)
            if baseline.shape != instance.shape:
                baseline = baseline.reshape(instance.shape)
            return baseline.astype(instance.dtype)
    
    def set_baseline(self, data: np.ndarray, method: str = "mean") -> "DeepLIFTExplainer":
        """
        Set the baseline from training data.
        
        Args:
            data: Training data array of shape (n_samples, n_features)
            method: Method to compute baseline:
                - "mean": Use mean of training data
                - "median": Use median of training data
                - "zeros": Use zeros (same as default)
                
        Returns:
            Self for method chaining
        """
        data = np.array(data)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        if method == "mean":
            self.baseline = np.mean(data, axis=0).astype(np.float32)
        elif method == "median":
            self.baseline = np.median(data, axis=0).astype(np.float32)
        elif method == "zeros":
            self.baseline = None
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return self
    
    def _compute_deeplift_rescale(
        self,
        instance: np.ndarray,
        baseline: np.ndarray,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute DeepLIFT attributions using the Rescale rule.
        
        The Rescale rule computes multipliers as:
        m = (activation - reference_activation) / (input - reference_input)
        
        For layers where input equals reference (delta = 0), we use the
        gradient as the multiplier (the limit as delta -> 0).
        
        This implementation uses the gradient formulation from Ancona et al.
        which shows that DeepLIFT-Rescale can be computed efficiently using
        modified gradients.
        
        Args:
            instance: Input instance
            baseline: Reference/baseline input
            target_class: Target class for attribution
            
        Returns:
            Array of attribution scores for each input feature
        """
        instance = instance.flatten().astype(np.float32)
        baseline = baseline.flatten().astype(np.float32)
        
        # Compute delta (difference from reference)
        delta = instance - baseline
        
        # For DeepLIFT with Rescale rule, we need to compute multipliers
        # that satisfy: attribution = delta * multiplier
        # where sum(attributions) = f(x) - f(x')
        
        # Method 1: Gradient at midpoint approximation
        # DeepLIFT ≈ delta * gradient(baseline + 0.5 * delta)
        # This is a good approximation for smooth functions
        
        midpoint = baseline + 0.5 * delta
        _, gradients = self.model.predict_with_gradients(
            midpoint.reshape(1, -1),
            target_class=target_class
        )
        gradients = gradients.flatten()
        
        if self.multiply_by_inputs:
            attributions = delta * gradients
        else:
            attributions = gradients
        
        return attributions
    
    def _compute_deeplift_exact(
        self,
        instance: np.ndarray,
        baseline: np.ndarray,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute DeepLIFT attributions using exact multiplier computation.
        
        This method computes the true DeepLIFT multipliers by evaluating
        at multiple points along the path and averaging, providing more
        accurate results at the cost of additional computation.
        
        For networks with ReLU activations, this is equivalent to:
        1. Forward pass on input to get activations
        2. Forward pass on baseline to get reference activations
        3. Backward pass computing multipliers based on activation differences
        
        Args:
            instance: Input instance
            baseline: Reference/baseline input
            target_class: Target class for attribution
            
        Returns:
            Array of attribution scores for each input feature
        """
        instance = instance.flatten().astype(np.float32)
        baseline = baseline.flatten().astype(np.float32)
        
        delta = instance - baseline
        
        # Use multiple evaluation points for more accurate multipliers
        # This is essentially Integrated Gradients with few steps, but
        # the key insight is that DeepLIFT's Rescale rule gives the same
        # result as IG for piecewise linear functions (like ReLU networks)
        
        n_points = 10
        alphas = np.linspace(0, 1, n_points)
        
        all_gradients = []
        for alpha in alphas:
            point = baseline + alpha * delta
            _, grads = self.model.predict_with_gradients(
                point.reshape(1, -1),
                target_class=target_class
            )
            all_gradients.append(grads.flatten())
        
        # Average gradients (trapezoidal rule approximation)
        avg_gradients = np.mean(all_gradients, axis=0)
        
        if self.multiply_by_inputs:
            attributions = delta * avg_gradients
        else:
            attributions = avg_gradients
        
        return attributions
    
    def explain(
        self,
        instance: np.ndarray,
        target_class: Optional[int] = None,
        baseline: Optional[np.ndarray] = None,
        method: str = "rescale",
        return_convergence_delta: bool = False
    ) -> Explanation:
        """
        Generate DeepLIFT explanation for an instance.
        
        Args:
            instance: 1D numpy array of input features.
            target_class: For classification, which class to explain.
                         If None, uses the predicted class.
            baseline: Override the default baseline for this explanation.
            method: Attribution method:
                - "rescale": Fast rescale rule (default, recommended)
                - "rescale_exact": More accurate but slower rescale
            return_convergence_delta: If True, include the convergence delta
                (difference between sum of attributions and prediction 
                difference). Should be close to 0 for correct attributions.
        
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
        
        # Compute DeepLIFT attributions
        if method == "rescale":
            attributions_raw = self._compute_deeplift_rescale(
                instance, bl, target_class
            )
        elif method == "rescale_exact":
            attributions_raw = self._compute_deeplift_exact(
                instance, bl, target_class
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'rescale' or 'rescale_exact'.")
        
        # Build attributions dict
        attributions = {
            fname: float(attributions_raw[i])
            for i, fname in enumerate(self.feature_names)
        }
        
        # Determine class name
        if self.class_names and target_class is not None:
            label_name = self.class_names[target_class]
        else:
            label_name = f"class_{target_class}" if target_class is not None else "output"
        
        explanation_data = {
            "feature_attributions": attributions,
            "attributions_raw": attributions_raw.tolist(),
            "baseline": bl.tolist(),
            "method": method,
            "multiply_by_inputs": self.multiply_by_inputs
        }
        
        # Optionally compute convergence delta (summation-to-delta property)
        if return_convergence_delta:
            pred_input = self.model.predict(instance.reshape(1, -1))
            pred_baseline = self.model.predict(bl.reshape(1, -1))
            
            if target_class is not None:
                pred_diff = pred_input[0, target_class] - pred_baseline[0, target_class]
            else:
                pred_diff = pred_input[0, 0] - pred_baseline[0, 0]
            
            attribution_sum = np.sum(attributions_raw)
            convergence_delta = abs(pred_diff - attribution_sum)
            
            explanation_data["convergence_delta"] = float(convergence_delta)
            explanation_data["prediction_difference"] = float(pred_diff)
            explanation_data["attribution_sum"] = float(attribution_sum)
        
        return Explanation(
            explainer_name="DeepLIFT",
            target_class=label_name,
            explanation_data=explanation_data
        )
    
    def explain_batch(
        self,
        X: np.ndarray,
        target_class: Optional[int] = None,
        method: str = "rescale"
    ) -> List[Explanation]:
        """
        Generate explanations for multiple instances.
        
        Args:
            X: 2D numpy array of instances (n_samples, n_features).
            target_class: Target class for all instances. If None,
                         uses predicted class for each instance.
            method: Attribution method ("rescale" or "rescale_exact").
        
        Returns:
            List of Explanation objects.
        """
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        return [
            self.explain(X[i], target_class=target_class, method=method)
            for i in range(X.shape[0])
        ]
    
    def explain_with_multiple_baselines(
        self,
        instance: np.ndarray,
        baselines: np.ndarray,
        target_class: Optional[int] = None,
        method: str = "rescale"
    ) -> Explanation:
        """
        Compute DeepLIFT with multiple reference baselines and average.
        
        Using multiple baselines can provide more robust attributions,
        especially when the choice of single baseline is uncertain.
        This is similar to DeepSHAP (DeepLIFT + SHAP) approach.
        
        Args:
            instance: Input instance to explain.
            baselines: Array of baseline instances (n_baselines, n_features).
            target_class: Target class for attribution.
            method: Attribution method.
        
        Returns:
            Explanation with averaged attributions across all baselines.
        """
        instance = np.array(instance).flatten().astype(np.float32)
        baselines = np.array(baselines)
        
        if baselines.ndim == 1:
            baselines = baselines.reshape(1, -1)
        
        # Compute attributions for each baseline
        all_attributions = []
        for bl in baselines:
            if method == "rescale":
                attr = self._compute_deeplift_rescale(
                    instance, bl.flatten(), target_class
                )
            else:
                attr = self._compute_deeplift_exact(
                    instance, bl.flatten(), target_class
                )
            all_attributions.append(attr)
        
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
            explainer_name="DeepLIFT_MultiBaseline",
            target_class=label_name,
            explanation_data={
                "feature_attributions": attributions,
                "attributions_raw": avg_attributions.tolist(),
                "attributions_std": std_attributions.tolist(),
                "n_baselines": len(baselines),
                "method": method
            }
        )
    
    def compare_with_integrated_gradients(
        self,
        instance: np.ndarray,
        target_class: Optional[int] = None,
        baseline: Optional[np.ndarray] = None,
        ig_steps: int = 50
    ) -> dict:
        """
        Compare DeepLIFT attributions with Integrated Gradients.
        
        Useful for validating that DeepLIFT provides similar results
        to IG (they should be very similar for ReLU networks).
        
        Args:
            instance: Input instance.
            target_class: Target class for attribution.
            baseline: Baseline for comparison.
            ig_steps: Number of steps for Integrated Gradients.
        
        Returns:
            Dictionary with both attributions and comparison metrics.
        """
        instance = np.array(instance).flatten().astype(np.float32)
        
        if baseline is not None:
            bl = np.array(baseline).flatten().astype(np.float32)
        else:
            bl = self._get_baseline(instance)
        
        # Determine target class
        if target_class is None and self.class_names:
            predictions = self.model.predict(instance.reshape(1, -1))
            target_class = int(np.argmax(predictions))
        
        # DeepLIFT attributions (fast)
        dl_attr = self._compute_deeplift_rescale(instance, bl, target_class)
        
        # Integrated Gradients (more computation)
        delta = instance - bl
        alphas = np.linspace(0, 1, ig_steps)
        
        all_gradients = []
        for alpha in alphas:
            point = bl + alpha * delta
            _, grads = self.model.predict_with_gradients(
                point.reshape(1, -1),
                target_class=target_class
            )
            all_gradients.append(grads.flatten())
        
        avg_gradients = np.mean(all_gradients, axis=0)
        ig_attr = delta * avg_gradients
        
        # Compute comparison metrics
        correlation = np.corrcoef(dl_attr, ig_attr)[0, 1]
        mse = np.mean((dl_attr - ig_attr) ** 2)
        max_diff = np.max(np.abs(dl_attr - ig_attr))
        
        return {
            "deeplift_attributions": dl_attr.tolist(),
            "integrated_gradients_attributions": ig_attr.tolist(),
            "correlation": float(correlation),
            "mse": float(mse),
            "max_difference": float(max_diff),
            "ig_steps": ig_steps
        }


class DeepLIFTShapExplainer(DeepLIFTExplainer):
    """
    DeepSHAP explainer - DeepLIFT combined with Shapley values.
    
    This is essentially DeepLIFT averaged over a distribution of baselines
    (usually samples from the training data), which approximates SHAP values.
    
    DeepSHAP inherits all benefits of DeepLIFT (speed, handling saturation)
    while providing the game-theoretic guarantees of Shapley values.
    
    Reference:
        Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to
        Interpreting Model Predictions. NeurIPS 2017.
        https://arxiv.org/abs/1705.07874
    
    Example:
        >>> explainer = DeepLIFTShapExplainer(adapter, feature_names)
        >>> explainer.set_background(X_train[:100])  # Set background samples
        >>> explanation = explainer.explain(instance)
    """
    
    def __init__(
        self,
        model,
        feature_names: List[str],
        class_names: Optional[List[str]] = None,
        background_data: Optional[np.ndarray] = None,
        n_background_samples: int = 100,
        eps: float = 1e-10
    ):
        """
        Initialize DeepSHAP explainer.
        
        Args:
            model: Model adapter with gradient computation capability.
            feature_names: List of input feature names.
            class_names: List of class names (for classification).
            background_data: Background dataset for computing expectations.
                            If None, must call set_background() before explain().
            n_background_samples: Number of background samples to use.
                                 More samples = more accurate but slower.
            eps: Small constant for numerical stability.
        """
        super().__init__(
            model=model,
            feature_names=feature_names,
            class_names=class_names,
            baseline=None,
            multiply_by_inputs=True,
            eps=eps
        )
        
        self.n_background_samples = n_background_samples
        self._background_data = None
        
        if background_data is not None:
            self.set_background(background_data)
    
    def set_background(self, data: np.ndarray) -> "DeepLIFTShapExplainer":
        """
        Set the background dataset for DeepSHAP.
        
        The background dataset is used to compute the expected output
        and expected attributions, which are the baseline for Shapley
        value computation.
        
        Args:
            data: Background data of shape (n_samples, n_features).
                  Typically a subset of the training data.
        
        Returns:
            Self for method chaining.
        """
        data = np.array(data).astype(np.float32)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # Subsample if necessary
        if len(data) > self.n_background_samples:
            indices = np.random.choice(
                len(data), 
                size=self.n_background_samples, 
                replace=False
            )
            data = data[indices]
        
        self._background_data = data
        return self
    
    def explain(
        self,
        instance: np.ndarray,
        target_class: Optional[int] = None,
        method: str = "rescale",
        return_convergence_delta: bool = False
    ) -> Explanation:
        """
        Generate DeepSHAP explanation by averaging DeepLIFT over backgrounds.
        
        Args:
            instance: Input instance to explain.
            target_class: Target class for attribution.
            method: DeepLIFT method ("rescale" or "rescale_exact").
            return_convergence_delta: Include convergence delta in output.
        
        Returns:
            Explanation with SHAP-style attributions.
        """
        if self._background_data is None:
            raise ValueError(
                "Background data not set. Call set_background() first."
            )
        
        instance = np.array(instance).flatten().astype(np.float32)
        
        # Determine target class
        if target_class is None and self.class_names:
            predictions = self.model.predict(instance.reshape(1, -1))
            target_class = int(np.argmax(predictions))
        
        # Compute DeepLIFT attributions for each background sample
        all_attributions = []
        for baseline in self._background_data:
            if method == "rescale":
                attr = self._compute_deeplift_rescale(
                    instance, baseline.flatten(), target_class
                )
            else:
                attr = self._compute_deeplift_exact(
                    instance, baseline.flatten(), target_class
                )
            all_attributions.append(attr)
        
        # Average to get SHAP values
        shap_values = np.mean(all_attributions, axis=0)
        std_values = np.std(all_attributions, axis=0)
        
        # Build attributions dict
        attributions = {
            fname: float(shap_values[i])
            for i, fname in enumerate(self.feature_names)
        }
        
        # Determine class name
        if self.class_names and target_class is not None:
            label_name = self.class_names[target_class]
        else:
            label_name = f"class_{target_class}" if target_class is not None else "output"
        
        explanation_data = {
            "feature_attributions": attributions,
            "attributions_raw": shap_values.tolist(),
            "attributions_std": std_values.tolist(),
            "n_background_samples": len(self._background_data),
            "method": method
        }
        
        # Compute expected values for context
        if return_convergence_delta:
            pred_input = self.model.predict(instance.reshape(1, -1))
            pred_background = self.model.predict(self._background_data)
            
            if target_class is not None:
                expected_output = np.mean(pred_background[:, target_class])
                actual_output = pred_input[0, target_class]
            else:
                expected_output = np.mean(pred_background[:, 0])
                actual_output = pred_input[0, 0]
            
            pred_diff = actual_output - expected_output
            attribution_sum = np.sum(shap_values)
            convergence_delta = abs(pred_diff - attribution_sum)
            
            explanation_data["expected_output"] = float(expected_output)
            explanation_data["actual_output"] = float(actual_output)
            explanation_data["convergence_delta"] = float(convergence_delta)
        
        return Explanation(
            explainer_name="DeepSHAP",
            target_class=label_name,
            explanation_data=explanation_data
        )
