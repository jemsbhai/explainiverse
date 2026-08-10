# src/explainiverse/adapters/pytorch_adapter.py
"""
PyTorch Model Adapter for Explainiverse.

Provides the prediction and gradient contracts used by compatible PyTorch
explainers. Individual explainer registry metadata defines the supported model,
data, and task scope; this adapter does not imply universal compatibility.

Example:
    import torch.nn as nn
    from explainiverse.adapters import PyTorchAdapter
    
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 3)
    )
    
    adapter = PyTorchAdapter(
        model,
        task="classification",
        class_names=["cat", "dog", "bird"]
    )
    
    probs = adapter.predict(X)  # Returns numpy array
"""

from numbers import Integral
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union, cast

import numpy as np

from .base_adapter import BaseModelAdapter

# Check if PyTorch is available. Static analysis follows the import-only
# branch; runtime keeps the existing optional-dependency behavior.
if TYPE_CHECKING:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
else:
    try:
        import torch
        import torch.nn as nn

        TORCH_AVAILABLE = True
    except ImportError:
        TORCH_AVAILABLE = False
        torch = None
        nn = None


def _check_torch_available():
    """Raise ImportError if PyTorch is not installed."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for PyTorchAdapter. " "Install it with: pip install torch"
        )


class PyTorchAdapter(BaseModelAdapter):
    """
    Adapter for PyTorch neural network models.

    Implements the prediction and gradient contracts documented below for
    compatible explainers. It handles device management and tensor/NumPy
    conversion for the listed classification and regression output shapes.

    Supports:
        - Multi-class classification (output shape: [batch, n_classes])
        - Binary classification, including one-logit models normalized by
          ``predict`` to [P(class 0), P(class 1)]
        - Regression (output shape: [batch, n_outputs] or [batch])

    Attributes:
        model: The PyTorch model (nn.Module)
        task: "classification" or "regression"
        device: torch.device for computation
        class_names: List of class names (for classification)
        feature_names: List of feature names
        output_activation: Optional activation function for outputs
        raw_model_output_space: Declared semantic space of the wrapped
            module's untransformed output
        gradient_output: Requested score space for gradient computations
        last_gradient_output_space: Effective score space of the latest
            gradient computation

    Example:
        >>> model = MyNeuralNetwork()
        >>> adapter = PyTorchAdapter(model, task="classification")
        >>> probs = adapter.predict(X_numpy)  # Returns probabilities
    """

    model: "nn.Module"
    device: "torch.device"
    class_names: Optional[List[str]]
    input_dtype: Union[str, "torch.dtype"]

    def __init__(
        self,
        model,
        task: str = "classification",
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        device: Optional[str] = None,
        output_activation: Optional[str] = "auto",
        gradient_output: str = "model",
        batch_size: int = 32,
        input_dtype: Optional[Union[str, "torch.dtype"]] = "auto",
    ):
        """
        Initialize the PyTorch adapter.

        Args:
            model: A PyTorch nn.Module model.
            task: "classification" or "regression".
            feature_names: List of input feature names.
            class_names: List of output class names (classification only).
            device: Device to run on ("cpu", "cuda", "cuda:0", etc.).
                   If None, auto-detects based on model parameters.
            output_activation: Activation for output layer:
                - "auto": assumes logits; uses softmax for classification
                  (sigmoid for one binary logit) and none for regression
                - "softmax": Apply softmax (classification)
                - "sigmoid": Apply sigmoid (binary classification)
                - "none" or None: Do not transform model outputs. For a
                  single binary output, the value is interpreted as P(class 1)
                  and must be in [0, 1].
            gradient_output: Score space for ``predict_with_gradients`` and
                ``get_layer_gradients``. ``"model"`` (the backward-compatible
                default) differentiates raw model outputs; ``"prediction"``
                differentiates post-activation predictions. A one-logit binary
                classifier always resolves to prediction space because no
                separate raw class-0 score exists.
            batch_size: Batch size for large inputs (default: 32).
            input_dtype: Input conversion policy. ``"auto"`` aligns floating
                inputs with the first floating model parameter/buffer while
                preserving integer inputs (required by embedding/index models).
                ``"preserve"`` retains the caller's dtype. A ``torch.dtype``
                or supported dtype-name string forces that dtype.
        """
        _check_torch_available()

        if not isinstance(model, nn.Module):
            raise TypeError(
                f"Expected nn.Module, got {type(model).__name__}. "
                "For sklearn models, use SklearnAdapter instead."
            )

        if task not in {"classification", "regression"}:
            raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")

        valid_activations = {"auto", "softmax", "sigmoid", "none", None}
        if output_activation not in valid_activations:
            raise ValueError(
                "output_activation must be one of 'auto', 'softmax', "
                f"'sigmoid', 'none', or None; got {output_activation!r}"
            )
        if gradient_output not in {"model", "prediction"}:
            raise ValueError(
                "gradient_output must be 'model' or 'prediction', " f"got {gradient_output!r}"
            )
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
            raise TypeError("batch_size must be a positive integer")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be a positive integer")

        super().__init__(model, feature_names)

        self.task: str = task
        self.class_names = self._normalize_class_names(class_names, task)
        self.batch_size: int = int(batch_size)
        self.input_dtype = self._resolve_input_dtype_policy(input_dtype)
        self.gradient_output: str = gradient_output
        self.last_gradient_output_space: Optional[str] = None

        # Determine device
        if device is not None:
            self.device = torch.device(device)
        else:
            # Auto-detect from model parameters
            try:
                param = next(model.parameters())
                self.device = param.device
            except StopIteration:
                # Model has no parameters, use CPU
                self.device = torch.device("cpu")

        # Move model to device and set to eval mode
        self.model = model.to(self.device)
        self.model.eval()

        # Configure output activation
        effective_activation: Optional[str]
        if output_activation == "auto":
            if task == "classification":
                effective_activation = "softmax"
            else:
                effective_activation = None
        else:
            effective_activation = output_activation if output_activation != "none" else None
        self.output_activation: Optional[str] = effective_activation
        if self.task == "classification":
            # Applying a configured sigmoid/softmax declares that the wrapped
            # module returns logits. With no adapter activation, the raw model
            # output may already be probabilities or may be another score, so
            # its semantic space cannot be inferred safely.
            self.raw_model_output_space: str = (
                "logit" if effective_activation in {"softmax", "sigmoid"} else "unspecified"
            )
        else:
            self.raw_model_output_space = "regression_value"

    @staticmethod
    def _normalize_class_names(class_names: Optional[List[str]], task: str) -> Optional[List[str]]:
        """Validate detached display metadata for classification outputs."""
        if class_names is None:
            return None
        if task != "classification":
            raise ValueError("class_names is only valid for classification")
        if isinstance(class_names, (str, bytes)):
            raise TypeError("class_names must be an iterable of strings or None")
        try:
            names = list(class_names)
        except TypeError as exc:
            raise TypeError("class_names must be an iterable of strings or None") from exc
        if not names:
            raise ValueError("class_names must not be empty")
        if any(not isinstance(name, str) for name in names):
            raise TypeError("class_names must contain only strings")
        if any(not name.strip() for name in names):
            raise ValueError("class_names must contain non-empty strings")
        if len(names) != len(set(names)):
            raise ValueError("class_names must be unique")
        return names

    @staticmethod
    def _resolve_input_dtype_policy(
        input_dtype: Optional[Union[str, "torch.dtype"]],
    ) -> Union[str, "torch.dtype"]:
        """Return a validated dtype policy or concrete ``torch.dtype``."""
        if input_dtype is None:
            return "auto"
        if isinstance(input_dtype, torch.dtype):
            return input_dtype
        if not isinstance(input_dtype, str):
            raise TypeError("input_dtype must be 'auto', 'preserve', a dtype name, or torch.dtype")
        key = input_dtype.lower().strip()
        aliases: Dict[str, Union[str, "torch.dtype"]] = {
            "auto": "auto",
            "preserve": "preserve",
            "float16": torch.float16,
            "half": torch.float16,
            "float32": torch.float32,
            "float": torch.float32,
            "float64": torch.float64,
            "double": torch.float64,
            "bfloat16": torch.bfloat16,
            "int32": torch.int32,
            "int": torch.int32,
            "int64": torch.int64,
            "long": torch.int64,
        }
        if key not in aliases:
            raise ValueError(
                "input_dtype must be 'auto', 'preserve', a supported dtype name, or torch.dtype"
            )
        return aliases[key]

    def _model_floating_dtype(self) -> Optional["torch.dtype"]:
        """Return the first floating parameter/buffer dtype, if one exists."""
        for value in list(self.model.parameters()) + list(self.model.buffers()):
            if value.is_floating_point():
                return value.dtype
        return None

    def _prepare_input(self, data) -> Union[np.ndarray, "torch.Tensor"]:
        """Validate a non-empty, real, sample-first input without losing dtype."""
        if isinstance(data, torch.Tensor):
            tensor_values = data
            if tensor_values.dtype == torch.bool or tensor_values.is_complex():
                raise TypeError(
                    "data must contain real numeric values, not booleans or complex values"
                )
            if tensor_values.dim() == 1:
                tensor_values = tensor_values.reshape(1, -1)
            if (
                tensor_values.dim() < 2
                or tensor_values.numel() == 0
                or any(size == 0 for size in tensor_values.shape)
            ):
                raise ValueError("data must be a non-empty sample-first tensor")
            if tensor_values.is_floating_point() and not bool(
                torch.all(torch.isfinite(tensor_values))
            ):
                raise ValueError("data must contain only finite values")
            return tensor_values

        array_values = np.asarray(data)
        if array_values.dtype == np.bool_ or not np.issubdtype(array_values.dtype, np.number):
            raise TypeError("data must contain real numeric values")
        if np.issubdtype(array_values.dtype, np.complexfloating):
            raise TypeError("data must contain real numeric values, not complex values")
        if array_values.ndim == 1:
            array_values = array_values.reshape(1, -1)
        if (
            array_values.ndim < 2
            or array_values.size == 0
            or any(size == 0 for size in array_values.shape)
        ):
            raise ValueError("data must be a non-empty sample-first array")
        if not np.all(np.isfinite(array_values)):
            raise ValueError("data must contain only finite values")
        return array_values

    def _to_tensor(self, data) -> "torch.Tensor":
        """Convert input under the configured dtype and device policy."""
        tensor = (
            data.to(self.device)
            if isinstance(data, torch.Tensor)
            else torch.as_tensor(data, device=self.device)
        )
        if isinstance(self.input_dtype, torch.dtype):
            return tensor.to(dtype=self.input_dtype)
        if self.input_dtype == "preserve" or not tensor.is_floating_point():
            return tensor
        model_dtype = self._model_floating_dtype()
        return tensor.to(dtype=model_dtype) if model_dtype is not None else tensor

    def _to_numpy(self, tensor: "torch.Tensor") -> np.ndarray:
        """Convert tensor to numpy array."""
        return tensor.detach().cpu().numpy()

    def _apply_activation(self, output: "torch.Tensor") -> "torch.Tensor":
        """Apply output activation function."""
        if self.output_activation == "softmax":
            # Handle different output shapes
            if output.dim() == 1 or (output.dim() == 2 and output.shape[1] == 1):
                # Binary: apply sigmoid instead of softmax
                return torch.sigmoid(output)
            return torch.softmax(output, dim=-1)
        elif self.output_activation == "sigmoid":
            return torch.sigmoid(output)
        return output

    def _normalize_output_shape(
        self, output: "torch.Tensor", expected_batch_size: Optional[int] = None
    ) -> "torch.Tensor":
        """
        Normalize output to consistent 2D shape (batch, outputs).

        Handles:
            - (batch,) -> (batch, 1)
            - (batch, n) -> (batch, n)
        """
        if not isinstance(output, torch.Tensor):
            raise TypeError("model output must be one torch.Tensor")
        if output.dim() == 1:
            output = output.unsqueeze(-1)
        elif output.dim() != 2:
            raise ValueError(
                "model output must be a sample-first tensor with shape (batch,) or (batch, outputs)"
            )
        if output.shape[0] == 0 or output.shape[1] == 0:
            raise ValueError("model output must not be empty")
        if expected_batch_size is not None and output.shape[0] != expected_batch_size:
            raise ValueError(
                f"model returned {output.shape[0]} rows for a batch of {expected_batch_size} inputs"
            )
        if (
            output.dtype == torch.bool
            or output.is_complex()
            or not (
                output.is_floating_point()
                or output.dtype in {torch.int8, torch.int16, torch.int32, torch.int64}
            )
        ):
            raise TypeError("model output must contain real numeric values")
        if not bool(torch.all(torch.isfinite(output))):
            raise ValueError("model output must contain only finite values")
        return output

    def _prediction_output(
        self, output: "torch.Tensor", expected_batch_size: Optional[int] = None
    ) -> "torch.Tensor":
        """Return the exact tensor exposed by :meth:`predict`.

        A one-output binary classifier is normalized to two complementary
        probability columns. Keeping this operation in torch also lets
        prediction-space gradients differentiate those exact columns.
        """
        normalized = self._normalize_output_shape(output, expected_batch_size)
        if self.task == "classification" and not normalized.is_floating_point():
            raise TypeError("classification model output must contain floating-point scores")
        activated = self._apply_activation(normalized)

        if self.task != "classification" or normalized.shape[1] != 1:
            prediction = activated
        else:
            positive_probability = activated
            if self.output_activation is None:
                # Without an activation, a scalar classification output is only
                # meaningful under the documented P(class 1) contract.
                if bool(torch.any((positive_probability < 0) | (positive_probability > 1))):
                    raise ValueError(
                        "For one-output binary classification, output_activation='none' "
                        "requires model output P(class 1) in [0, 1]. Use "
                        "output_activation='auto' or 'sigmoid' for logits."
                    )
            prediction = torch.cat((1.0 - positive_probability, positive_probability), dim=1)

        if not bool(torch.all(torch.isfinite(prediction))):
            raise ValueError("activated model output must contain only finite values")
        if self.class_names is not None and prediction.shape[1] != len(self.class_names):
            raise ValueError(
                f"class_names has {len(self.class_names)} entries but the model exposes "
                f"{prediction.shape[1]} classification outputs"
            )
        return prediction

    def _gradient_score_output(
        self, output: "torch.Tensor", expected_batch_size: Optional[int] = None
    ) -> Tuple["torch.Tensor", str]:
        """Resolve the output tensor whose selected score is differentiated."""
        normalized = self._normalize_output_shape(output, expected_batch_size)
        if not normalized.is_floating_point():
            raise TypeError("gradient score output must be floating-point")

        # A single logit defines class-1 log odds, but the model has no raw
        # class-0 score. Use complementary probabilities instead of inventing
        # an arbitrary synthetic logit convention.
        if self.task == "classification" and normalized.shape[1] == 1:
            return self._prediction_output(output, expected_batch_size), "prediction"

        if self.gradient_output == "prediction":
            return self._prediction_output(output, expected_batch_size), "prediction"

        if self.class_names is not None and normalized.shape[1] != len(self.class_names):
            raise ValueError(
                f"class_names has {len(self.class_names)} entries but the model exposes "
                f"{normalized.shape[1]} classification outputs"
            )

        return normalized, "model"

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Generate predictions for input data.

        Args:
            data: Input data as numpy array. Shape: (n_samples, n_features)
                  or (n_samples, channels, height, width) for images.

        Returns:
            Predictions as numpy array:
            - Classification: probabilities of shape (n_samples, n_classes)
            - Regression: values of shape (n_samples, n_outputs)
        """
        prepared_data = self._prepare_input(data)

        n_samples = prepared_data.shape[0]
        outputs: List[np.ndarray] = []

        with torch.no_grad():
            for i in range(0, n_samples, self.batch_size):
                batch = prepared_data[i : i + self.batch_size]
                tensor_batch = self._to_tensor(batch)

                output = self.model(tensor_batch)
                output = self._prediction_output(output, expected_batch_size=len(batch))
                outputs.append(self._to_numpy(output))

        return np.concatenate(outputs, axis=0)

    def _normalize_target_indices(
        self,
        target_class,
        batch_size: int,
        n_outputs: int,
        default_indices: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """Normalize a scalar or per-sample target to validated indices."""
        if target_class is None:
            if default_indices is None:
                raise ValueError("An explicit output index is required")
            indices = default_indices.to(device=self.device, dtype=torch.long)
        elif isinstance(target_class, Integral) and not isinstance(target_class, bool):
            indices = torch.full(
                (batch_size,), int(target_class), device=self.device, dtype=torch.long
            )
        elif isinstance(target_class, torch.Tensor):
            if (
                target_class.dtype == torch.bool
                or target_class.dtype.is_floating_point
                or target_class.is_complex()
            ):
                raise TypeError("target_class tensor must contain integer indices")
            if target_class.dim() > 1:
                raise ValueError("target_class tensor must be scalar or one-dimensional")
            indices = target_class.to(device=self.device, dtype=torch.long).reshape(-1)
        elif isinstance(target_class, np.ndarray):
            if not np.issubdtype(target_class.dtype, np.integer):
                raise TypeError("target_class array must contain integer indices")
            if target_class.ndim > 1:
                raise ValueError("target_class array must be scalar or one-dimensional")
            indices = torch.as_tensor(target_class, device=self.device, dtype=torch.long).reshape(
                -1
            )
        else:
            raise TypeError(
                "target_class must be an integer, numpy integer, integer array, "
                "integer torch tensor, or None"
            )

        if indices.numel() == 1 and batch_size != 1:
            indices = indices.expand(batch_size)
        if indices.numel() != batch_size:
            raise ValueError(
                f"Expected one target per sample ({batch_size}), got {indices.numel()}"
            )
        if bool(torch.any((indices < 0) | (indices >= n_outputs))):
            raise ValueError(f"target_class indices must be in [0, {n_outputs - 1}]")
        return indices

    def _select_target_scores(
        self,
        score_output: "torch.Tensor",
        target_class: Optional[Union[int, np.integer, np.ndarray, "torch.Tensor"]],
    ) -> "torch.Tensor":
        """Select one explicitly defined class/output score per sample."""
        batch_size, n_outputs = score_output.shape

        if self.task == "classification":
            default_indices = score_output.argmax(dim=-1)
        elif n_outputs == 1:
            default_indices = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        else:
            default_indices = None
            if target_class is None:
                raise ValueError(
                    "An explicit target_class output index is required for "
                    "multi-output regression; outputs are not silently summed."
                )

        indices = self._normalize_target_indices(
            target_class,
            batch_size=batch_size,
            n_outputs=n_outputs,
            default_indices=default_indices,
        )
        return score_output.gather(1, indices.view(-1, 1)).squeeze(-1)

    def _get_target_scores(
        self,
        output: "torch.Tensor",
        target_class: Optional[Union[int, np.integer, np.ndarray, "torch.Tensor"]] = None,
    ) -> "torch.Tensor":
        """Extract scores in the adapter's effective gradient output space."""
        score_output, output_space = self._gradient_score_output(output)
        self.last_gradient_output_space = output_space
        return self._select_target_scores(score_output, target_class)

    def predict_with_gradients(
        self,
        data: np.ndarray,
        target_class: Optional[Union[int, np.integer, np.ndarray, "torch.Tensor"]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions and compute gradients w.r.t. inputs.

        This is essential for gradient-based attribution methods like
        Integrated Gradients, GradCAM, and Saliency Maps.

        Args:
            data: Input data as numpy array.
            target_class: Class or regression-output index for gradient
                computation. NumPy integer scalars and per-sample integer
                arrays/tensors are accepted. Classification defaults to each
                sample's highest score. Multi-output regression requires an
                explicit index.

        Returns:
            Tuple of (scores, gradients) as numpy arrays.
            - scores: Exact model- or prediction-space output selected by
              ``gradient_output`` and differentiated for ``gradients``.
              One-logit binary classification returns two complementary
              probabilities and records ``prediction`` in
              ``last_gradient_output_space``.
            - gradients: same shape as input data
        """
        prepared_data = self._prepare_input(data)

        # Convert to tensor with gradient tracking
        # Never toggle ``requires_grad`` or attach history to a caller-owned
        # tensor when conversion is a no-op on dtype/device.
        tensor_data = self._to_tensor(prepared_data).detach().clone()
        if not tensor_data.is_floating_point():
            raise TypeError("input gradients require a floating-point input dtype")
        tensor_data.requires_grad_(True)

        # Forward pass
        output = self.model(tensor_data)

        # Resolve once so returned scores and differentiated scores cannot
        # accidentally occupy different spaces.
        score_output, output_space = self._gradient_score_output(
            output, expected_batch_size=tensor_data.shape[0]
        )
        self.last_gradient_output_space = output_space
        target_scores = self._select_target_scores(score_output, target_class)
        gradients = torch.autograd.grad(target_scores.sum(), tensor_data)[0]

        return (self._to_numpy(score_output), self._to_numpy(gradients))

    def get_layer_output(self, data: np.ndarray, layer_name: str) -> np.ndarray:
        """
        Get intermediate layer activations.

        Useful for methods like GradCAM that need feature map activations.

        Args:
            data: Input data as numpy array.
            layer_name: Name of the layer to extract (as registered in model).

        Returns:
            Layer activations as numpy array.
        """
        prepared_data = self._prepare_input(data)

        activations: Dict[str, "torch.Tensor"] = {}

        def hook_fn(module, input, output):
            activations["output"] = output

        # Find and hook the layer
        layer = dict(self.model.named_modules()).get(layer_name)
        if layer is None:
            available = list(dict(self.model.named_modules()).keys())
            raise ValueError(f"Layer '{layer_name}' not found. Available layers: {available}")

        handle = layer.register_forward_hook(hook_fn)

        try:
            with torch.no_grad():
                tensor_data = self._to_tensor(prepared_data)
                _ = self.model(tensor_data)
        finally:
            handle.remove()

        return self._to_numpy(activations["output"])

    def get_layer_gradients(
        self,
        data: np.ndarray,
        layer_name: str,
        target_class: Optional[Union[int, np.integer, np.ndarray, "torch.Tensor"]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get gradients of output w.r.t. a specific layer's activations.

        Essential for GradCAM and similar visualization methods.

        Args:
            data: Input data as numpy array.
            layer_name: Name of the layer for gradient computation.
            target_class: Target class or regression-output index. Accepts the
                same scalar and per-sample integer forms as
                :meth:`predict_with_gradients`.

        Returns:
            Tuple of (layer_activations, layer_gradients) as numpy arrays.
        """
        prepared_data = self._prepare_input(data)

        activations: Dict[str, "torch.Tensor"] = {}

        def forward_hook(module, input, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    "get_layer_gradients requires the target layer to return " "one torch.Tensor"
                )
            activations["output"] = output

        # Find and hook the layer
        layer = dict(self.model.named_modules()).get(layer_name)
        if layer is None:
            available = list(dict(self.model.named_modules()).keys())
            raise ValueError(f"Layer '{layer_name}' not found. Available layers: {available}")

        # Module-level full backward hooks wrap outputs in an autograd view and
        # therefore fail for the common Conv -> ReLU(inplace=True) pattern.
        # Capture the forward tensor and differentiate the selected score with
        # respect to that tensor directly instead.  ``autograd.grad`` also
        # avoids accumulating gradients into model parameters.
        forward_handle = layer.register_forward_hook(forward_hook)
        modules = list(self.model.modules())
        training_states = [bool(module.training) for module in modules]
        parameters = list(self.model.parameters())
        original_gradients = [parameter.grad for parameter in parameters]
        saved_gradient_values = [
            None if gradient is None else gradient.detach().clone()
            for gradient in original_gradients
        ]
        buffers = list(self.model.buffers())
        saved_buffer_values = [buffer.detach().clone() for buffer in buffers]

        try:
            # Keep caller tensor state/history isolated from this gradient
            # graph even when dtype and device conversion would be a no-op.
            tensor_data = self._to_tensor(prepared_data).detach().clone()
            if tensor_data.is_floating_point():
                tensor_data.requires_grad_(True)

            output = self.model(tensor_data)
            if "output" not in activations:
                raise RuntimeError(
                    f"Layer '{layer_name}' did not run during the model forward pass"
                )

            score_output, output_space = self._gradient_score_output(
                output, expected_batch_size=tensor_data.shape[0]
            )
            self.last_gradient_output_space = output_space
            target_scores = self._select_target_scores(score_output, target_class)
            layer_output = activations["output"]
            layer_gradients = torch.autograd.grad(
                target_scores.sum(),
                layer_output,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]
            activation_values = self._to_numpy(layer_output)
            gradient_values = self._to_numpy(layer_gradients)
        finally:
            forward_handle.remove()
            # ``autograd.grad`` normally leaves parameter ``.grad`` untouched.
            # Restore both value and object identity defensively in case model
            # hooks or custom autograd functions wrote to them.
            with torch.no_grad():
                for parameter, original, saved in zip(
                    parameters, original_gradients, saved_gradient_values
                ):
                    if original is None:
                        parameter.grad = None
                    else:
                        original.copy_(cast("torch.Tensor", saved))
                        parameter.grad = original
                for buffer, saved in zip(buffers, saved_buffer_values):
                    buffer.copy_(saved)
            # Direct assignment preserves mixed parent/child training states.
            for module, training in zip(modules, training_states):
                module.training = training

        return activation_values, gradient_values

    def list_layers(self) -> List[str]:
        """
        List all named layers/modules in the model.

        Returns:
            List of layer names that can be used with get_layer_output/gradients.
        """
        return [name for name, _ in self.model.named_modules() if name]

    def to(self, device: str) -> "PyTorchAdapter":
        """
        Move the model to a different device.

        Args:
            device: Target device ("cpu", "cuda", "cuda:0", etc.)

        Returns:
            Self for chaining.
        """
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        return self

    def train_mode(self) -> "PyTorchAdapter":
        """Set model to training mode (enables dropout, batchnorm updates)."""
        self.model.train()
        return self

    def eval_mode(self) -> "PyTorchAdapter":
        """Set model to evaluation mode (disables dropout, freezes batchnorm)."""
        self.model.eval()
        return self
