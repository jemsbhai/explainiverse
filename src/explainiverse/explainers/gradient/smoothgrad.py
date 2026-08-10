# src/explainiverse/explainers/gradient/smoothgrad.py
"""
SmoothGrad - Removing Noise by Adding Noise.

This implementation averages raw input gradients computed on noisy copies of
a one-dimensional tabular input. It does not preserve spatial image shapes.
The returned statistics do not by themselves establish perceptual smoothness,
explanation quality, uncertainty, or interpretability.

Variants:
- SmoothGrad: Average of gradients
- SmoothGrad-Squared: Average of squared gradients
- VarGrad: Element-wise variance of gradients

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

from numbers import Integral, Real
from typing import List, Optional

import numpy as np

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation
from explainiverse.explainers._validation import validate_name_sequence
from explainiverse.explainers.gradient._model_state import preserve_adapter_model_eval


class SmoothGradExplainer(BaseExplainer):
    """
    SmoothGrad explainer for neural networks.

    Computes attributions from raw input gradients over noisy copies of the
    input. This class is not a generic wrapper around arbitrary attribution
    methods and makes no interpretability guarantee.

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
        random_state: Optional seed for the per-call local NumPy generator

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
        noise_type: str = "gaussian",
        random_state: Optional[int] = None,
    ):
        """
        Initialize the SmoothGrad explainer.

        Args:
            model: A model adapter with predict_with_gradients() method.
                   Use PyTorchAdapter for PyTorch models.
            feature_names: List of input feature names.
            class_names: List of class names (for classification tasks).
            n_samples: Number of independently noised gradient samples to
                      average. More samples require more gradient evaluations;
                      no run-specific quality improvement is guaranteed.
                      Default: 50.
            noise_scale: Scale of the noise to add:
                - For "gaussian": standard deviation (default: 0.15)
                - For "uniform": half-range, noise in [-scale, scale]
                The caller chooses this scale in the units of the supplied input.
            noise_type: Type of noise distribution:
                - "gaussian": Normal distribution N(0, σ²) (default)
                - "uniform": Uniform distribution U(-σ, σ)
            random_state: Optional non-negative integer seed. Each public
                attribution call creates a fresh local NumPy Generator. With
                an integer seed, repeated calls use the same perturbation
                sequence and are reproducible. With None, each call obtains
                fresh entropy. Neither mode reads or advances NumPy's global
                random state.

        Raises:
            TypeError: If model doesn't have predict_with_gradients or if
                random_state is not None or an integer.
            ValueError: If n_samples < 1, noise_scale < 0, noise_type is
                invalid, or random_state is negative.
        """
        super().__init__(model)

        # Validate model has gradient capability
        if not hasattr(model, "predict_with_gradients"):
            raise TypeError(
                "Model adapter must have predict_with_gradients() method. "
                "Use PyTorchAdapter for PyTorch models."
            )

        # Validate parameters
        if isinstance(n_samples, bool) or not isinstance(n_samples, Integral):
            raise TypeError("n_samples must be an integer")
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")

        if isinstance(noise_scale, bool) or not isinstance(noise_scale, Real):
            raise TypeError("noise_scale must be a finite real number")
        if not np.isfinite(float(noise_scale)):
            raise ValueError("noise_scale must be finite")
        if noise_scale < 0:
            raise ValueError(f"noise_scale must be >= 0, got {noise_scale}")

        if noise_type not in ["gaussian", "uniform"]:
            raise ValueError(f"noise_type must be 'gaussian' or 'uniform', got '{noise_type}'")

        if random_state is not None:
            if isinstance(random_state, bool) or not isinstance(random_state, Integral):
                raise TypeError("random_state must be None or a non-negative integer")
            if random_state < 0:
                raise ValueError("random_state must be non-negative")

        validated_features = validate_name_sequence(feature_names, name="feature_names")
        assert validated_features is not None
        self.feature_names: List[str] = validated_features
        self.class_names: Optional[List[str]] = (
            validate_name_sequence(class_names, name="class_names") if class_names else None
        )
        self.n_samples: int = int(n_samples)
        self.noise_scale: float = float(noise_scale)
        self.noise_type: str = noise_type
        self.random_state: Optional[int] = int(random_state) if random_state is not None else None

    def _resolve_target_class(
        self, instance: np.ndarray, target_class: Optional[int]
    ) -> Optional[int]:
        """Resolve one output at the unperturbed input for all noise samples."""
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

    def _score_space_metadata(self) -> dict:
        """Describe the score space used for the latest gradient call."""
        return {
            "score_space": getattr(self.model, "last_gradient_output_space", "unknown") or "unknown"
        }

    def _prepare_instance(self, instance: np.ndarray) -> np.ndarray:
        """Validate one complete flat/tabular input without structural flattening."""
        raw = np.asarray(instance)
        if raw.ndim != 1:
            raise ValueError(
                "SmoothGradExplainer supports one-dimensional tabular instances; "
                f"got shape {raw.shape}. Image/spatial tensors are not supported."
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
        if not np.isrealobj(prepared) or not np.isfinite(prepared).all():
            raise ValueError("instance must contain only finite real values")
        return prepared

    def _new_rng(self) -> np.random.Generator:
        """Create the local generator for one public attribution call."""
        return np.random.default_rng(self.random_state)

    def _generate_noise(
        self, shape: tuple, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """
        Generate noise samples based on the configured noise type.

        Args:
            shape: Shape of the noise array to generate.
            rng: Local generator. If omitted, a generator is created using
                the configured ``random_state``.

        Returns:
            Numpy array of noise samples.
        """
        local_rng = self._new_rng() if rng is None else rng
        if self.noise_type == "gaussian":
            return local_rng.normal(0, self.noise_scale, shape).astype(np.float32)
        else:  # uniform
            return local_rng.uniform(-self.noise_scale, self.noise_scale, shape).astype(np.float32)

    def _compute_smoothgrad(
        self,
        instance: np.ndarray,
        target_class: Optional[int] = None,
        method: str = "smoothgrad",
        absolute_value: bool = False,
        rng: Optional[np.random.Generator] = None,
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
            rng: Local generator for this computation. If omitted, a fresh
                generator is created from ``random_state``.

        Returns:
            Tuple of (attributions, std_attributions) arrays.
        """
        instance = instance.flatten().astype(np.float32)
        target_class = self._resolve_target_class(instance, target_class)
        local_rng = self._new_rng() if rng is None else rng

        valid_methods = {"smoothgrad", "smoothgrad_squared", "vargrad"}
        if method not in valid_methods:
            raise ValueError(f"Unknown method: {method!r}. Use one of {sorted(valid_methods)}.")
        if not isinstance(absolute_value, bool):
            raise TypeError("absolute_value must be a boolean")

        all_gradients = self._collect_noisy_gradients(instance, target_class, local_rng)
        return self._aggregate_gradient_samples(all_gradients, method, absolute_value)

    def _collect_noisy_gradients(
        self,
        instance: np.ndarray,
        target_class: Optional[int],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Evaluate the configured noisy-input gradient sample set once."""
        gradient_samples: List[np.ndarray] = []

        for _ in range(self.n_samples):
            # Add noise to input
            if self.noise_scale > 0:
                noise = self._generate_noise(instance.shape, rng)
                noisy_input = instance + noise
            else:
                noisy_input = instance.copy()

            # Compute gradient
            with preserve_adapter_model_eval(self.model):
                _, gradients = self.model.predict_with_gradients(
                    noisy_input.reshape(1, -1), target_class=target_class
                )
            gradient_matrix = np.asarray(gradients)
            expected_shape = (1, instance.size)
            if gradient_matrix.shape != expected_shape:
                raise ValueError(
                    "predict_with_gradients returned the wrong feature count or gradient "
                    f"shape; expected {expected_shape}, got {gradient_matrix.shape}"
                )
            if not np.isrealobj(gradient_matrix) or not np.isfinite(gradient_matrix).all():
                raise FloatingPointError(
                    "predict_with_gradients must return finite real input gradients"
                )
            gradient_samples.append(gradient_matrix[0])

        return np.asarray(gradient_samples)

    @staticmethod
    def _aggregate_gradient_samples(
        all_gradients: np.ndarray,
        method: str,
        absolute_value: bool = False,
    ) -> tuple:
        """Apply one SmoothGrad-family aggregate to a shared sample matrix."""

        # Compute attributions based on method
        if method == "smoothgrad":
            attributions = np.mean(all_gradients, axis=0)
            std_attributions = np.std(all_gradients, axis=0)
        elif method == "smoothgrad_squared":
            # Average of squared gradients
            squared_gradients = all_gradients**2
            attributions = np.mean(squared_gradients, axis=0)
            std_attributions = np.std(squared_gradients, axis=0)
        elif method == "vargrad":
            # Variance of gradients
            attributions = np.var(all_gradients, axis=0)
            std_attributions = np.zeros_like(attributions)  # No std for variance
        # Apply absolute value if requested
        if absolute_value:
            attributions = np.abs(attributions)

        if not np.isfinite(attributions).all() or not np.isfinite(std_attributions).all():
            raise FloatingPointError("SmoothGrad aggregation produced non-finite values")

        return attributions, std_attributions

    def _explain_with_rng(
        self,
        instance: np.ndarray,
        target_class: Optional[int] = None,
        method: str = "smoothgrad",
        absolute_value: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> Explanation:
        """Build one explanation while consuming the supplied local RNG."""
        instance = self._prepare_instance(instance)

        # Resolve once before drawing noise so every sample differentiates the
        # same class selected at the original input.
        target_class = self._resolve_target_class(instance, target_class)

        # Compute SmoothGrad
        attributions, std_attributions = self._compute_smoothgrad(
            instance, target_class, method, absolute_value, rng=rng
        )

        # Build attributions dict
        attributions_dict = {
            fname: float(attributions[i]) for i, fname in enumerate(self.feature_names)
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
            "absolute_value": absolute_value,
            "random_state": self.random_state,
        }

        return Explanation(
            explainer_name=explainer_name,
            target_class=label_name,
            explanation_data=explanation_data,
            feature_names=self.feature_names,
            metadata=self._score_space_metadata(),
        )

    def explain(
        self,
        instance: np.ndarray,
        target_class: Optional[int] = None,
        method: str = "smoothgrad",
        absolute_value: bool = False,
    ) -> Explanation:
        """
        Generate a SmoothGrad explanation for one instance.

        A fresh local generator is created on every call. Consequently, an
        integer ``random_state`` makes repeated calls deterministic, while
        ``random_state=None`` uses fresh entropy without consuming NumPy's
        process-global random state.

        Args:
            instance: 1D numpy array of input features.
            target_class: For classification, which class to explain.
                         If None, uses the predicted class.
            method: Aggregation method:
                - "smoothgrad": Average of gradients (default)
                - "smoothgrad_squared": Average of squared gradients
                - "vargrad": Element-wise variance of gradients
            absolute_value: If True, return absolute values of attributions.
                           This discards attribution direction.

        Returns:
            Explanation object with feature attributions.

        Example:
            >>> explanation = explainer.explain(instance)
            >>> print(explanation.explanation_data["feature_attributions"])
        """
        return self._explain_with_rng(
            instance,
            target_class=target_class,
            method=method,
            absolute_value=absolute_value,
            rng=self._new_rng(),
        )

    def explain_batch(
        self,
        X: np.ndarray,
        target_class: Optional[int] = None,
        method: str = "smoothgrad",
        absolute_value: bool = False,
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
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        elif X.ndim != 2:
            raise ValueError(
                "X must be a one- or two-dimensional tabular array; "
                f"got shape {X.shape}. Image/spatial tensors are not supported."
            )

        rng = self._new_rng()
        return [
            self._explain_with_rng(
                X[i],
                target_class=target_class,
                method=method,
                absolute_value=absolute_value,
                rng=rng,
            )
            for i in range(X.shape[0])
        ]

    def compute_with_baseline_comparison(
        self, instance: np.ndarray, target_class: Optional[int] = None
    ) -> dict:
        """
        Compare SmoothGrad with raw gradient for analysis.

        This returns several configured gradient aggregates and their raw
        correlation; it does not validate noise reduction or explanation quality.

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
        instance = self._prepare_instance(instance)

        target_class = self._resolve_target_class(instance, target_class)

        # Raw gradient (no noise)
        with preserve_adapter_model_eval(self.model):
            _, raw_gradient = self.model.predict_with_gradients(
                instance.reshape(1, -1), target_class=target_class
            )
        raw_gradient = np.asarray(raw_gradient)
        expected_shape = (1, instance.size)
        if raw_gradient.shape != expected_shape:
            raise ValueError("predict_with_gradients returned an invalid raw input gradient")
        if not np.isrealobj(raw_gradient) or not np.all(np.isfinite(raw_gradient)):
            raise ValueError("predict_with_gradients returned an invalid raw input gradient")
        raw_gradient = raw_gradient[0]

        # All variants are algebraic transformations of one common noisy-
        # gradient sample matrix. This makes their comparison independent of
        # arbitrary differences between three noise draws.
        rng = self._new_rng()
        gradient_samples = self._collect_noisy_gradients(instance, target_class, rng)
        smoothgrad, _ = self._aggregate_gradient_samples(gradient_samples, "smoothgrad")
        smoothgrad_squared, _ = self._aggregate_gradient_samples(
            gradient_samples, "smoothgrad_squared"
        )
        vargrad, _ = self._aggregate_gradient_samples(gradient_samples, "vargrad")

        # Pearson correlation is undefined for a one-feature or constant map.
        # Return an explicit null/status contract instead of serializing NaN.
        correlation = None
        correlation_defined = False
        if smoothgrad.size < 2:
            correlation_reason = "requires_at_least_two_features"
        elif np.ptp(smoothgrad) == 0.0 or np.ptp(raw_gradient) == 0.0:
            correlation_reason = "constant_attribution_vector"
        else:
            candidate = float(np.corrcoef(smoothgrad, raw_gradient)[0, 1])
            if np.isfinite(candidate):
                correlation = candidate
                correlation_defined = True
                correlation_reason = None
            else:
                correlation_reason = "numerically_undefined"

        return {
            "smoothgrad": smoothgrad.tolist(),
            "raw_gradient": raw_gradient.tolist(),
            "smoothgrad_squared": smoothgrad_squared.tolist(),
            "vargrad": vargrad.tolist(),
            "correlation": correlation,
            "correlation_defined": correlation_defined,
            "correlation_reason": correlation_reason,
            "common_random_numbers": True,
            "n_samples": self.n_samples,
            "noise_scale": self.noise_scale,
            "random_state": self.random_state,
        }

    def adaptive_noise_scale(self, instance: np.ndarray, percentile: float = 15.0) -> float:
        """
        Compute a noise scale as a percentage of the input range.

        For a constant input, the fallback range is its maximum absolute value,
        or one when that value is also zero.

        Args:
            instance: Input instance.
            percentile: Percentage of input range to use as noise scale.
                       Default: 15%.

        Returns:
            Range-scaled noise value in the input's units.
        """
        instance = self._prepare_instance(instance)
        input_range = instance.max() - instance.min()

        # Avoid zero scale for constant inputs
        if input_range == 0:
            input_range = np.abs(instance).max()
        if input_range == 0:
            input_range = 1.0

        return float(input_range * percentile / 100.0)
