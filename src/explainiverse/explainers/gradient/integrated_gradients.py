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
from explainiverse.explainers.gradient._model_state import preserve_adapter_model_eval

BaselineCallable = Callable[[np.ndarray], np.ndarray]
BaselineSpec = Union[np.ndarray, str, BaselineCallable]


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

        self.feature_names: Optional[List[str]] = (
            validate_name_sequence(feature_names, name="feature_names") if feature_names else None
        )
        self.class_names: Optional[List[str]] = (
            validate_name_sequence(class_names, name="class_names") if class_names else None
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
        if not np.isrealobj(raw):
            raise ValueError("instance must contain only finite real values")
        try:
            prepared = raw.astype(np.float32, copy=True)
        except (TypeError, ValueError) as error:
            raise TypeError("instance must contain real numeric values") from error
        if not np.isfinite(prepared).all():
            raise ValueError("instance must contain only finite real values")

        actual_shape = tuple(prepared.shape)
        if self.input_shape is None:
            self.input_shape = actual_shape
        elif actual_shape != self.input_shape:
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
        if not np.isrealobj(raw):
            raise ValueError("baseline must contain only finite real values")
        try:
            prepared = raw.astype(np.float32, copy=True)
        except (TypeError, ValueError) as error:
            raise TypeError("baseline must contain real numeric values") from error
        if prepared.size == 0 or not np.isfinite(prepared).all():
            raise ValueError("baseline must contain only finite real values")
        return prepared

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
            if self.class_names is not None and target_index >= len(self.class_names):
                raise ValueError(f"target_class must be in [0, {len(self.class_names) - 1}]")
            return target_index

        is_classification = getattr(self.model, "task", None) == "classification"
        # Keep compatibility with gradient-capable third-party adapters that
        # predate the explicit ``task`` attribute but provide class metadata.
        if not is_classification and self.class_names is None:
            return None

        with preserve_adapter_model_eval(self.model, preserve_gradients=False):
            predictions = np.asarray(self.model.predict(instance[np.newaxis, ...]))
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
                baseline = local_rng.uniform(
                    low=float(instance.min()), high=float(instance.max()), size=instance.shape
                ).astype(instance.dtype)
            elif self.baseline == "mean":
                # Mean value baseline
                baseline = np.full_like(instance, instance.mean())
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

        # Compute path from baseline to input
        delta = instance - baseline

        # Collect gradients at each interpolation point
        gradient_samples: List[np.ndarray] = []

        for alpha in alphas:
            # Interpolated input: baseline + alpha * (input - baseline)
            interp_input = baseline + alpha * delta

            # Add batch dimension for model
            if interp_input.ndim == len(original_shape):
                interp_batch = interp_input[np.newaxis, ...]
            else:
                interp_batch = interp_input

            # Get gradients
            with preserve_adapter_model_eval(self.model):
                _, gradients = self.model.predict_with_gradients(
                    interp_batch, target_class=target_class
                )

            gradients = np.asarray(gradients)
            expected_gradient_shape = (1,) + original_shape
            if gradients.shape != expected_gradient_shape:
                raise ValueError(
                    "predict_with_gradients returned the wrong gradient shape; "
                    f"expected {expected_gradient_shape}, got {gradients.shape}"
                )
            if not np.isrealobj(gradients) or not np.isfinite(gradients).all():
                raise FloatingPointError(
                    "predict_with_gradients must return finite real input gradients"
                )

            gradient_samples.append(gradients[0])

        all_gradients = np.asarray(gradient_samples)  # Shape: (n_steps, *original_shape)

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

        if not np.isfinite(integrated_gradients).all():
            raise FloatingPointError("Integrated Gradients produced non-finite attributions")

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
        if self.class_names and target_class is not None:
            label_name = self.class_names[target_class]
        else:
            label_name = f"class_{target_class}" if target_class is not None else "output"

        # Optionally compute convergence delta
        if return_convergence_delta:
            # The sum of attributions should equal F(x) - F(baseline)
            pred_input = instance[np.newaxis, ...]
            pred_baseline = bl[np.newaxis, ...]

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

            attribution_sum = float(np.sum(ig_attributions))
            convergence_delta = abs(float(pred_diff) - attribution_sum)

            explanation_data["convergence_delta"] = convergence_delta
            explanation_data["prediction_difference"] = float(pred_diff)
            explanation_data["attribution_sum"] = attribution_sum

        return Explanation(
            explainer_name="IntegratedGradients",
            target_class=label_name,
            explanation_data=explanation_data,
            feature_names=self.feature_names,
            metadata=self._score_space_metadata(),
        )

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
            noise = rng.normal(0, noise_scale, original_shape).astype(np.float32)
            noisy_baseline = self._prepare_baseline(noise, original_shape)

            ig = self._compute_integrated_gradients(instance, noisy_baseline, target_class)
            all_attributions.append(ig)

        # Average attributions
        avg_attributions = np.mean(all_attributions, axis=0)
        std_attributions = np.std(all_attributions, axis=0)

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

        if self.class_names and target_class is not None:
            label_name = self.class_names[target_class]
        else:
            label_name = f"class_{target_class}" if target_class is not None else "output"

        return Explanation(
            explainer_name="IntegratedGradients_Smooth",
            target_class=label_name,
            explanation_data=explanation_data,
            feature_names=self.feature_names,
            metadata=self._score_space_metadata(),
        )
