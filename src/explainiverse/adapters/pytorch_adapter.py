# src/explainiverse/adapters/pytorch_adapter.py
"""
PyTorch Model Adapter for Explainiverse.

Provides a unified interface for PyTorch neural networks, enabling
compatibility with all explainers in the framework.

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

import numpy as np
from typing import List, Optional, Union, Callable

from .base_adapter import BaseModelAdapter

# Check if PyTorch is available
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
            "PyTorch is required for PyTorchAdapter. "
            "Install it with: pip install torch"
        )


class PyTorchAdapter(BaseModelAdapter):
    """
    Adapter for PyTorch neural network models.
    
    Wraps a PyTorch nn.Module to provide a consistent interface for
    explainability methods. Handles device management, tensor/numpy
    conversions, and supports both classification and regression tasks.
    
    Attributes:
        model: The PyTorch model (nn.Module)
        task: "classification" or "regression"
        device: torch.device for computation
        class_names: List of class names (for classification)
        feature_names: List of feature names
        output_activation: Optional activation function for outputs
    
    Example:
        >>> model = MyNeuralNetwork()
        >>> adapter = PyTorchAdapter(model, task="classification")
        >>> probs = adapter.predict(X_numpy)  # Returns probabilities
    """
    
    def __init__(
        self,
        model,
        task: str = "classification",
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        device: Optional[str] = None,
        output_activation: Optional[str] = "auto",
        batch_size: int = 32
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
                - "auto": softmax for classification, none for regression
                - "softmax": Apply softmax (classification)
                - "sigmoid": Apply sigmoid (binary classification)
                - "none" or None: No activation (raw logits/values)
            batch_size: Batch size for large inputs (default: 32).
        """
        _check_torch_available()
        
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"Expected nn.Module, got {type(model).__name__}. "
                "For sklearn models, use SklearnAdapter instead."
            )
        
        super().__init__(model, feature_names)
        
        self.task = task
        self.class_names = list(class_names) if class_names else None
        self.batch_size = batch_size
        
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
        if output_activation == "auto":
            if task == "classification":
                self.output_activation = "softmax"
            else:
                self.output_activation = None
        else:
            self.output_activation = output_activation if output_activation != "none" else None
    
    def _to_tensor(self, data: np.ndarray) -> "torch.Tensor":
        """Convert numpy array to tensor on the correct device."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device).float()
        return torch.tensor(data, dtype=torch.float32, device=self.device)
    
    def _to_numpy(self, tensor: "torch.Tensor") -> np.ndarray:
        """Convert tensor to numpy array."""
        return tensor.detach().cpu().numpy()
    
    def _apply_activation(self, output: "torch.Tensor") -> "torch.Tensor":
        """Apply output activation function."""
        if self.output_activation == "softmax":
            return torch.softmax(output, dim=-1)
        elif self.output_activation == "sigmoid":
            return torch.sigmoid(output)
        return output
    
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
        data = np.array(data)
        
        # Handle single instance
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        n_samples = data.shape[0]
        outputs = []
        
        with torch.no_grad():
            for i in range(0, n_samples, self.batch_size):
                batch = data[i:i + self.batch_size]
                tensor_batch = self._to_tensor(batch)
                
                output = self.model(tensor_batch)
                output = self._apply_activation(output)
                outputs.append(self._to_numpy(output))
        
        return np.vstack(outputs)
    
    def predict_with_gradients(
        self,
        data: np.ndarray,
        target_class: Optional[int] = None
    ) -> tuple:
        """
        Generate predictions and compute gradients w.r.t. inputs.
        
        This is essential for gradient-based attribution methods like
        Integrated Gradients, GradCAM, and Saliency Maps.
        
        Args:
            data: Input data as numpy array.
            target_class: Class index for gradient computation.
                         If None, uses the predicted class.
        
        Returns:
            Tuple of (predictions, gradients) as numpy arrays.
        """
        data = np.array(data)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # Convert to tensor with gradient tracking
        tensor_data = self._to_tensor(data)
        tensor_data.requires_grad_(True)
        
        # Forward pass
        output = self.model(tensor_data)
        activated_output = self._apply_activation(output)
        
        # Determine target for gradient
        if self.task == "classification":
            if target_class is None:
                target_class = output.argmax(dim=-1)
            elif isinstance(target_class, int):
                target_class = torch.tensor([target_class] * data.shape[0], device=self.device)
            
            # Select target class scores for gradient
            target_scores = output.gather(1, target_class.view(-1, 1)).squeeze()
        else:
            # Regression: gradient w.r.t. output
            target_scores = output.squeeze()
        
        # Backward pass
        if target_scores.dim() == 0:
            target_scores.backward()
        else:
            target_scores.sum().backward()
        
        gradients = tensor_data.grad
        
        return (
            self._to_numpy(activated_output),
            self._to_numpy(gradients)
        )
    
    def get_layer_output(
        self,
        data: np.ndarray,
        layer_name: str
    ) -> np.ndarray:
        """
        Get intermediate layer activations.
        
        Useful for methods like GradCAM that need feature map activations.
        
        Args:
            data: Input data as numpy array.
            layer_name: Name of the layer to extract (as registered in model).
        
        Returns:
            Layer activations as numpy array.
        """
        data = np.array(data)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        activations = {}
        
        def hook_fn(module, input, output):
            activations['output'] = output
        
        # Find and hook the layer
        layer = dict(self.model.named_modules()).get(layer_name)
        if layer is None:
            available = list(dict(self.model.named_modules()).keys())
            raise ValueError(
                f"Layer '{layer_name}' not found. Available layers: {available}"
            )
        
        handle = layer.register_forward_hook(hook_fn)
        
        try:
            with torch.no_grad():
                tensor_data = self._to_tensor(data)
                _ = self.model(tensor_data)
        finally:
            handle.remove()
        
        return self._to_numpy(activations['output'])
    
    def get_layer_gradients(
        self,
        data: np.ndarray,
        layer_name: str,
        target_class: Optional[int] = None
    ) -> tuple:
        """
        Get gradients of output w.r.t. a specific layer's activations.
        
        Essential for GradCAM and similar visualization methods.
        
        Args:
            data: Input data as numpy array.
            layer_name: Name of the layer for gradient computation.
            target_class: Target class for gradient (classification).
        
        Returns:
            Tuple of (layer_activations, layer_gradients) as numpy arrays.
        """
        data = np.array(data)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        activations = {}
        gradients = {}
        
        def forward_hook(module, input, output):
            activations['output'] = output
        
        def backward_hook(module, grad_input, grad_output):
            gradients['output'] = grad_output[0]
        
        # Find and hook the layer
        layer = dict(self.model.named_modules()).get(layer_name)
        if layer is None:
            available = list(dict(self.model.named_modules()).keys())
            raise ValueError(
                f"Layer '{layer_name}' not found. Available layers: {available}"
            )
        
        forward_handle = layer.register_forward_hook(forward_hook)
        backward_handle = layer.register_full_backward_hook(backward_hook)
        
        try:
            tensor_data = self._to_tensor(data)
            tensor_data.requires_grad_(True)
            
            output = self.model(tensor_data)
            
            if self.task == "classification":
                if target_class is None:
                    target_class = output.argmax(dim=-1)
                elif isinstance(target_class, int):
                    target_class = torch.tensor([target_class] * data.shape[0], device=self.device)
                
                target_scores = output.gather(1, target_class.view(-1, 1)).squeeze()
            else:
                target_scores = output.squeeze()
            
            if target_scores.dim() == 0:
                target_scores.backward()
            else:
                target_scores.sum().backward()
        finally:
            forward_handle.remove()
            backward_handle.remove()
        
        return (
            self._to_numpy(activations['output']),
            self._to_numpy(gradients['output'])
        )
    
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
