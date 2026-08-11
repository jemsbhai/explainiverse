# src/explainiverse/explainers/gradient/integrated_gradients.py
"""
Integrated Gradients - Axiomatic Attribution for Deep Networks.

The canonical Integrated Gradients method accumulates gradients along a
straight-line path from a baseline to the input. Sundararajan et al. prove
Sensitivity and Implementation Invariance for the mathematical method under
their stated assumptions. This implementation numerically approximates that
integral for one fixed output; finite-step results do not independently certify
either axiom for an arbitrary supplied model.

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

from numbers import Integral, Real
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation
from explainiverse.explainers._validation import validate_name_sequence
from explainiverse.explainers.gradient._input import (
    as_floating_array,
    scale_safe_integrated_gradient,
    scale_safe_mean,
    scale_safe_mean_std,
    scale_safe_product_sum,
    scale_safe_sum,
)
from explainiverse.explainers.gradient._model_state import preserve_adapter_model_eval

BaselineCallable = Callable[[np.ndarray], np.ndarray]
BaselineSpec = Union[np.ndarray, str, BaselineCallable]


def _stable_convex_combination(start, end, fraction) -> np.ndarray:
    """Evaluate ``(1-fraction)*start + fraction*end`` without range overflow."""

    start_values, end_values, fractions = np.broadcast_arrays(start, end, fraction)
    with np.errstate(over="ignore", invalid="ignore"):
        direct = (1.0 - fractions) * start_values + fractions * end_values
    unstable = ~np.isfinite(direct)
    if np.any(unstable):
        scale = np.maximum(np.abs(start_values), np.abs(end_values))
        safe_scale = np.where(scale == 0, 1, scale)
        scaled = (
            (1.0 - fractions) * (start_values / safe_scale) + fractions * (end_values / safe_scale)
        ) * scale
        direct = np.where(unstable, scaled, direct)
    if not np.isfinite(direct).all():
        raise FloatingPointError("finite convex interpolation produced non-finite values")
    return direct


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
        baseline: Optional[BaselineSpec] = None,
        method: str = "riemann_middle",
        input_shape: Optional[Tuple[int, ...]] = None,
        random_state: Optional[int] = None,
    ):
        """
        Initialize the Integrated Gradients explainer.

        Args:
            model: A model adapter with predict_with_gradients() method.
                   Use PyTorchAdapter for PyTorch models.
            feature_names: List of input feature names. Required for tabular
                          data to create named attributions.
            class_names: List of class names (for classification tasks).
            n_steps: Number of integration samples. More samples require more
                    gradient evaluations, but approximation error is not
                    guaranteed to decrease monotonically. Default: 50.
            baseline: Baseline input for comparison:
                     - None: uses zeros
                     - "random": random baseline (useful for images)
                     - np.ndarray: specific baseline values
                     - Callable: function(instance) -> baseline
            method: Integration method:
                   - "riemann_middle": Middle Riemann sum (default)
                   - "riemann_left": Left Riemann sum
                   - "riemann_right": Right Riemann sum
                   - "riemann_trapezoid": Trapezoidal rule
            input_shape: Expected shape of a single input (excluding batch dim).
                        If None, inferred from first explain() call.
            random_state: Optional non-negative seed for random baselines and
                        noisy-baseline averaging. Every public call uses a
                        local NumPy Generator and never advances NumPy's
                        process-global random state.
        """
        super().__init__(model)

        # Validate model has gradient capability
        if not hasattr(model, "predict_with_gradients"):
            raise TypeError(
                "Model adapter must have predict_with_gradients() method. "
                "Use PyTorchAdapter for PyTorch models."
            )

        if isinstance(n_steps, bool) or not isinstance(n_steps, Integral):
            raise TypeError("n_steps must be an integer")
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1")

        if input_shape is not None:
            if (
                not isinstance(input_shape, tuple)
                or not input_shape
                or any(
                    isinstance(dimension, bool)
                    or not isinstance(dimension, Integral)
                    or int(dimension) <= 0
                    for dimension in input_shape
                )
            ):
                raise ValueError("input_shape must be a non-empty tuple of positive integers")
            input_shape = tuple(int(dimension) for dimension in input_shape)

        if random_state is not None:
            if isinstance(random_state, bool) or not isinstance(random_state, Integral):
                raise TypeError("random_state must be a non-negative integer or None")
            if int(random_state) < 0:
                raise ValueError("random_state must be non-negative")

        valid_methods = {
            "riemann_middle",
            "riemann_left",
            "riemann_right",
            "riemann_trapezoid",
        }
        if not isinstance(method, str):
            raise TypeError("method must be a supported Riemann integration method")
        if method not in valid_methods:
            raise ValueError(f"Unknown method: {method!r}. Use one of {sorted(valid_methods)}")

        self.feature_names: Optional[List[str]] = (
            validate_name_sequence(feature_names, name="feature_names")
            if feature_names is not None
            else None
        )
        self.class_names: Optional[List[str]] = (
            validate_name_sequence(class_names, name="class_names")
            if class_names is not None
            else None
        )
        self.n_steps: int = int(n_steps)
        self.baseline: Optional[BaselineSpec] = baseline
        self.method: str = method
        self.input_shape: Optional[Tuple[int, ...]] = input_shape
        self.random_state: Optional[int] = None if random_state is None else int(random_state)

    def _new_rng(self) -> np.random.Generator:
        """Create one operation-local generator without touching global state."""
        return np.random.default_rng(self.random_state)

    def _prepare_instance(self, instance: np.ndarray) -> np.ndarray:
        """Validate one finite real input and enforce the single-input shape."""

        raw = np.asarray(instance)
        if raw.ndim == 0 or raw.size == 0:
            raise ValueError("instance must be a non-empty array with at least one dimension")
        try:
            prepared = as_floating_array(raw, name="instance")
        except ValueError as exc:
            if "complex" in str(exc):
                raise ValueError("instance must contain only finite real values") from exc
            raise

        actual_shape = tuple(prepared.shape)
        if self.input_shape is not None and actual_shape != self.input_shape:
            raise ValueError(
                "instance shape must match input_shape exactly; "
                f"expected {self.input_shape}, got {actual_shape}"
            )
        return prepared

    @staticmethod
    def _prepare_baseline(value, expected_shape: Tuple[int, ...]) -> np.ndarray:
        """Validate a baseline without silently reshaping or discarding values."""

        raw = np.asarray(value)
        if tuple(raw.shape) != expected_shape:
            raise ValueError(
                "baseline shape must match the instance exactly; "
                f"expected {expected_shape}, got {tuple(raw.shape)}"
            )
        try:
            return as_floating_array(raw, name="baseline")
        except ValueError as exc:
            if "complex" in str(exc):
                raise ValueError("baseline must contain only finite real values") from exc
            raise

    @staticmethod
    def _as_model_batch(value: np.ndarray) -> np.ndarray:
        """Add the batch axis and the implicit channel for a 2-D grayscale image."""
        if value.ndim == 2:
            return value[np.newaxis, np.newaxis, ...]
        return value[np.newaxis, ...]

    @staticmethod
    def _gradient_for_instance(gradients: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        """Map a model-batch gradient back to the caller's single-input shape."""
        expected = (1, 1) + shape if len(shape) == 2 else (1,) + shape
        if gradients.shape != expected:
            raise ValueError(
                "predict_with_gradients returned the wrong gradient shape; "
                f"expected {expected}, got {gradients.shape}"
            )
        sample = gradients[0]
        if len(shape) == 2:
            sample = sample[0]
        return sample

    def _resolve_target_class(
        self, instance: np.ndarray, target_class: Optional[int]
    ) -> Optional[int]:
        """Resolve a classification target once at the original input.

        Integrated Gradients must integrate one scalar function along the
        entire path.  Letting the adapter choose an argmax independently at
        each interpolation point instead integrates a changing function and
        can make opposing class gradients cancel.
        """
        if target_class is not None:
            if isinstance(target_class, bool) or not isinstance(target_class, Integral):
                raise TypeError("target_class must be an integer output index or None")
            target_index = int(target_class)
            if target_index < 0:
                raise ValueError("target_class must be non-negative")
            task = getattr(self.model, "task", None)
            if (
                task != "regression"
                and self.class_names is not None
                and target_index >= len(self.class_names)
            ):
                raise ValueError(f"target_class must be in [0, {len(self.class_names) - 1}]")
            return target_index

        task = getattr(self.model, "task", None)
        if task == "regression":
            return None
        # Keep compatibility with gradient-capable third-party adapters that
        # predate the explicit ``task`` attribute but provide class metadata.
        if task != "classification" and self.class_names is None:
            return None

        with preserve_adapter_model_eval(self.model, preserve_gradients=False):
            predictions = np.asarray(self.model.predict(self._as_model_batch(instance)))
        if predictions.ndim != 2 or predictions.shape[0] != 1:
            raise ValueError(
                "Classification predictions must have shape (1, n_classes) "
                f"when resolving an attribution target; got {predictions.shape}"
            )
        resolved = int(np.argmax(predictions[0]))
        if self.class_names is not None and resolved >= len(self.class_names):
            raise ValueError("class_names length does not match the model prediction width")
        return resolved

    def _score_space_metadata(self) -> dict:
        """Describe the score space used for the latest gradient call."""
        return {
            "score_space": getattr(self.model, "last_gradient_output_space", "unknown") or "unknown"
        }

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

    def _validate_tabular_feature_names(self, instance: np.ndarray, data_type: str) -> None:
        """Require one declared feature identity per flat tabular input value."""
        if data_type != "tabular" or self.feature_names is None:
            return
        if instance.size != len(self.feature_names):
            raise ValueError(
                "instance feature count must match feature_names exactly; "
                f"got {instance.size} values and {len(self.feature_names)} names"
            )

    def _get_baseline(
        self,
        instance: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Get the baseline for a given input shape.

        Args:
            instance: Input instance (preserves shape)

        Returns:
            Baseline array with same shape as instance
        """
        if self.baseline is None:
            # Default: zero baseline
            baseline = np.zeros_like(instance)
        elif isinstance(self.baseline, str):
            if self.baseline == "random":
                # Random baseline (useful for images)
                local_rng = self._new_rng() if rng is None else rng
                fractions = local_rng.random(instance.shape)
                baseline = _stable_convex_combination(
                    float(instance.min()), float(instance.max()), fractions
                ).astype(instance.dtype)
            elif self.baseline == "mean":
                # Mean value baseline
                stable_mean = float(scale_safe_mean_std(instance.reshape(-1))[0])
                baseline = np.full_like(instance, stable_mean)
            else:
                raise ValueError(f"Unknown baseline type: {self.baseline}")
        elif callable(self.baseline):
            baseline = self.baseline(instance.copy())
        else:
            baseline = self.baseline
        return self._prepare_baseline(baseline, tuple(instance.shape))

    def _get_interpolation_alphas(self) -> np.ndarray:
        """Get interpolation points based on method."""
        if self.method == "riemann_left":
            return np.linspace(0, 1 - 1 / self.n_steps, self.n_steps)
        elif self.method == "riemann_right":
            return np.linspace(1 / self.n_steps, 1, self.n_steps)
        elif self.method == "riemann_middle":
            return np.linspace(0.5 / self.n_steps, 1 - 0.5 / self.n_steps, self.n_steps)
        elif self.method == "riemann_trapezoid":
            return np.linspace(0, 1, self.n_steps + 1)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _compute_integrated_gradients(
        self, instance: np.ndarray, baseline: np.ndarray, target_class: Optional[int] = None
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

        # Collect gradients at each interpolation point
        gradient_samples: List[np.ndarray] = []

        for alpha in alphas:
            # This convex form is algebraically identical to
            # ``baseline + alpha * (instance - baseline)`` but does not first
            # form an unrepresentable endpoint difference for opposite-sign
            # finite values near the dtype limit.
            interp_input = _stable_convex_combination(baseline, instance, alpha)

            interp_batch = self._as_model_batch(interp_input)

            # Get gradients
            with preserve_adapter_model_eval(self.model):
                _, gradients = self.model.predict_with_gradients(
                    interp_batch, target_class=target_class
                )

            gradients = np.asarray(gradients)
            if not np.isrealobj(gradients) or not np.isfinite(gradients).all():
                raise FloatingPointError(
                    "predict_with_gradients must return finite real input gradients"
                )
            gradient_samples.append(self._gradient_for_instance(gradients, original_shape))

        all_gradients = np.asarray(gradient_samples)  # Shape: (n_steps, *original_shape)

        # Approximate the integral
        if self.method == "riemann_trapezoid":
            # Trapezoidal rule
            quadrature_weights = np.ones(self.n_steps + 1)
            quadrature_weights[0] = 0.5
            quadrature_weights[-1] = 0.5
            broadcast_weights = quadrature_weights.reshape(
                (quadrature_weights.size,) + (1,) * len(original_shape)
            )
            try:
                avg_gradients = scale_safe_product_sum(
                    all_gradients,
                    broadcast_weights,
                    axis=0,
                    divisor=self.n_steps,
                )
            except FloatingPointError:
                return scale_safe_integrated_gradient(
                    all_gradients,
                    baseline,
                    instance,
                    weights=quadrature_weights,
                    divisor=self.n_steps,
                )
        else:
            # Standard Riemann sum: average of gradients
            quadrature_weights = np.ones(all_gradients.shape[0])
            try:
                avg_gradients = scale_safe_mean(all_gradients, axis=0)
            except FloatingPointError:
                return scale_safe_integrated_gradient(
                    all_gradients,
                    baseline,
                    instance,
                    weights=quadrature_weights,
                    divisor=self.n_steps,
                )

        # Scale by the endpoint difference. Form the direct difference where
        # representable; for an overflowing finite endpoint span, factor out
        # the endpoint magnitude and multiply it by the (typically small)
        # average gradient before applying the normalized difference.
        with np.errstate(over="ignore", invalid="ignore"):
            delta = instance - baseline
            integrated_gradients = delta * avg_gradients
        overflowing_delta = ~np.isfinite(delta)
        if np.any(overflowing_delta):
            endpoint_scale = np.maximum(np.abs(instance), np.abs(baseline))
            normalized_delta = instance / endpoint_scale - baseline / endpoint_scale
            with np.errstate(over="ignore", invalid="ignore"):
                stable_values = normalized_delta * (avg_gradients * endpoint_scale)
            integrated_gradients = np.where(overflowing_delta, stable_values, integrated_gradients)

        needs_exact_fusion = ~np.isfinite(integrated_gradients) | (
            (integrated_gradients == 0.0) & (instance != baseline) & (avg_gradients != 0.0)
        )
        if np.any(needs_exact_fusion):
            integrated_gradients = scale_safe_integrated_gradient(
                all_gradients,
                baseline,
                instance,
                weights=quadrature_weights,
                divisor=self.n_steps,
            )

        return integrated_gradients

    def explain(
        self,
        instance: np.ndarray,
        target_class: Optional[int] = None,
        baseline: Optional[np.ndarray] = None,
        return_convergence_delta: bool = False,
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
        if not isinstance(return_convergence_delta, (bool, np.bool_)):
            raise TypeError("return_convergence_delta must be a boolean")
        instance = self._prepare_instance(instance)
        original_shape = instance.shape

        # Infer data type
        data_type = self._infer_data_type(instance)
        self._validate_tabular_feature_names(instance, data_type)

        # Get baseline (preserves shape)
        if baseline is not None:
            bl = self._prepare_baseline(baseline, original_shape)
        else:
            bl = self._get_baseline(instance, rng=self._new_rng())

        # Resolve the target at the original input exactly once, then hold it
        # fixed over all interpolation points.
        target_class = self._resolve_target_class(instance, target_class)

        # Compute integrated gradients (preserves shape)
        ig_attributions = self._compute_integrated_gradients(instance, bl, target_class)

        # Build explanation data
        explanation_data = {
            "attributions_raw": ig_attributions.tolist(),
            "baseline": bl.tolist(),
            "n_steps": self.n_steps,
            "method": self.method,
            "input_shape": list(original_shape),
            "data_type": data_type,
            "random_state": self.random_state,
        }

        # For tabular data, create named attributions
        if data_type == "tabular" and self.feature_names is not None:
            flat_ig = ig_attributions.flatten()
            attributions = {fname: float(flat_ig[i]) for i, fname in enumerate(self.feature_names)}
            explanation_data["feature_attributions"] = attributions
        elif data_type == "image":
            # For images, retain the attribution tensor and a magnitude-pooled map.
            explanation_data["attribution_map"] = ig_attributions
            # Also store channel-aggregated saliency for visualization
            if ig_attributions.ndim == 3:  # (C, H, W)
                explanation_data["saliency_map"] = np.abs(ig_attributions).sum(axis=0)
            else:
                explanation_data["saliency_map"] = np.abs(ig_attributions)

        # Determine class name
        if (
            self.class_names is not None
            and target_class is not None
            and target_class < len(self.class_names)
        ):
            label_name = self.class_names[target_class]
        else:
            prefix = "output" if getattr(self.model, "task", None) == "regression" else "class"
            label_name = f"{prefix}_{target_class}" if target_class is not None else "output"

        # Optionally compute convergence delta
        if return_convergence_delta:
            # The sum of attributions should equal F(x) - F(baseline)
            pred_input = self._as_model_batch(instance)
            pred_baseline = self._as_model_batch(bl)

            # ``predict_with_gradients`` returns scores in the exact space
            # whose derivatives produced the attributions.  Using predict()
            # here could compare logit attributions with probability changes.
            with preserve_adapter_model_eval(self.model):
                pred_input_val, _ = self.model.predict_with_gradients(
                    pred_input, target_class=target_class
                )
            input_score_space = getattr(self.model, "last_gradient_output_space", None)
            with preserve_adapter_model_eval(self.model):
                pred_baseline_val, _ = self.model.predict_with_gradients(
                    pred_baseline, target_class=target_class
                )
            baseline_score_space = getattr(self.model, "last_gradient_output_space", None)
            if (
                input_score_space is not None
                and baseline_score_space is not None
                and input_score_space != baseline_score_space
            ):
                raise ValueError("Gradient score space changed between the input and baseline")

            score_index = target_class if target_class is not None else 0
            pred_diff = pred_input_val[0, score_index] - pred_baseline_val[0, score_index]

            attribution_sum = float(scale_safe_sum(ig_attributions))
            convergence_delta = abs(float(pred_diff) - attribution_sum)

            explanation_data["convergence_delta"] = convergence_delta
            explanation_data["prediction_difference"] = float(pred_diff)
            explanation_data["attribution_sum"] = attribution_sum

        explanation = Explanation(
            explainer_name="IntegratedGradients",
            target_class=label_name,
            explanation_data=explanation_data,
            feature_names=self.feature_names,
            metadata=self._score_space_metadata(),
        )
        if self.input_shape is None:
            self.input_shape = original_shape
        return explanation

    def explain_batch(self, X: np.ndarray, target_class: Optional[int] = None) -> List[Explanation]:
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
        if X.ndim == 0 or X.size == 0 or X.shape[0] == 0:
            raise ValueError("X must contain at least one non-empty instance")

        # Handle single instance passed as array
        if X.ndim == 1:
            return [self.explain(X, target_class=target_class)]

        return [self.explain(X[i], target_class=target_class) for i in range(X.shape[0])]

    def compute_attributions_with_noise(
        self,
        instance: np.ndarray,
        target_class: Optional[int] = None,
        n_samples: int = 5,
        noise_scale: float = 0.1,
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
        if isinstance(n_samples, bool) or not isinstance(n_samples, Integral):
            raise TypeError("n_samples must be an integer")
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1")
        if isinstance(noise_scale, bool) or not isinstance(noise_scale, Real):
            raise TypeError("noise_scale must be a finite non-negative real number")
        if not np.isfinite(float(noise_scale)) or noise_scale < 0:
            raise ValueError("noise_scale must be a finite non-negative real number")
        n_samples = int(n_samples)
        noise_scale = float(noise_scale)

        instance = self._prepare_instance(instance)
        original_shape = instance.shape
        data_type = self._infer_data_type(instance)
        self._validate_tabular_feature_names(instance, data_type)

        # As in standard IG, every noisy-baseline path must explain the same
        # output selected at the original (unperturbed) instance.
        target_class = self._resolve_target_class(instance, target_class)

        all_attributions = []
        rng = self._new_rng()
        for _ in range(n_samples):
            # Create noisy baseline
            noise = rng.normal(0, noise_scale, original_shape).astype(instance.dtype)
            noisy_baseline = self._prepare_baseline(noise, original_shape)

            ig = self._compute_integrated_gradients(instance, noisy_baseline, target_class)
            all_attributions.append(ig)

        # Average attributions
        avg_attributions, std_attributions = scale_safe_mean_std(np.asarray(all_attributions))

        # Build explanation data
        explanation_data = {
            "attributions_raw": avg_attributions.tolist(),
            "attributions_std": std_attributions.tolist(),
            "n_samples": n_samples,
            "noise_scale": noise_scale,
            "data_type": data_type,
            "random_state": self.random_state,
        }

        # For tabular data, create named attributions
        if data_type == "tabular" and self.feature_names is not None:
            flat_avg = avg_attributions.flatten()
            attributions = {fname: float(flat_avg[i]) for i, fname in enumerate(self.feature_names)}
            explanation_data["feature_attributions"] = attributions

        if (
            self.class_names is not None
            and target_class is not None
            and target_class < len(self.class_names)
        ):
            label_name = self.class_names[target_class]
        else:
            prefix = "output" if getattr(self.model, "task", None) == "regression" else "class"
            label_name = f"{prefix}_{target_class}" if target_class is not None else "output"

        explanation = Explanation(
            explainer_name="IntegratedGradients_Smooth",
            target_class=label_name,
            explanation_data=explanation_data,
            feature_names=self.feature_names,
            metadata=self._score_space_metadata(),
        )
        if self.input_shape is None:
            self.input_shape = original_shape
        return explanation
