# src/explainiverse/explainers/gradient/lrp.py
"""
Layer-wise Relevance Propagation (LRP) - Decomposition-based Attribution.

LRP decomposes network predictions back to input features using a conservation
principle. Unlike gradient-based methods, LRP propagates relevance scores
layer-by-layer through the network using specific propagation rules.

Key Properties:
- Conservation: Sum of relevances at each layer equals the output
- Layer-wise decomposition: Relevance flows backward through layers
- Multiple rules: Different rules for different layer types and use cases

Supported Layer Types:
- Linear (fully connected)
- Conv2d (convolutional)
- BatchNorm1d, BatchNorm2d
- ReLU, LeakyReLU, ELU, Tanh, Sigmoid
- MaxPool2d, AvgPool2d
- Flatten, Dropout (passthrough)

Propagation Rules:
- LRP-0: Basic rule (no stabilization) - theoretical baseline
- LRP-ε (epsilon): Adds small constant for numerical stability (recommended)
- LRP-γ (gamma): Enhances positive contributions - good for image classification
- LRP-αβ (alpha-beta): Separates positive/negative contributions - fine control
- LRP-z⁺ (z-plus): Only considers positive weights - often used for input layers
- Composite: Different rules for different layers

Mathematical Formulation:
    For layer l with input a and output z = Wx + b:
    
    LRP-0:     R_j = Σ_k (a_j * w_jk / z_k) * R_k
    LRP-ε:     R_j = Σ_k (a_j * w_jk / (z_k + ε*sign(z_k))) * R_k
    LRP-γ:     R_j = Σ_k (a_j * (w_jk + γ*w_jk⁺) / (z_k + γ*z_k⁺)) * R_k
    LRP-αβ:    R_j = Σ_k (α * (a_j * w_jk)⁺ / z_k⁺ - β * (a_j * w_jk)⁻ / z_k⁻) * R_k
    LRP-z⁺:    R_j = Σ_k (a_j * w_jk⁺ / Σ_i a_i * w_ik⁺) * R_k

Reference:
    Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K. R., & Samek, W. (2015).
    On Pixel-wise Explanations for Non-Linear Classifier Decisions by Layer-wise
    Relevance Propagation. PLOS ONE.
    https://doi.org/10.1371/journal.pone.0130140

    Montavon, G., Binder, A., Lapuschkin, S., Samek, W., & Müller, K. R. (2019).
    Layer-wise Relevance Propagation: An Overview. Explainable AI: Interpreting,
    Explaining and Visualizing Deep Learning. Springer.

Example:
    from explainiverse.explainers.gradient import LRPExplainer
    from explainiverse.adapters import PyTorchAdapter
    
    adapter = PyTorchAdapter(model, task="classification")
    
    explainer = LRPExplainer(
        model=adapter,
        feature_names=feature_names,
        rule="epsilon",
        epsilon=1e-6
    )
    
    explanation = explainer.explain(instance)
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Union
from collections import OrderedDict

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation


# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


# Valid LRP rules
VALID_RULES = ["epsilon", "gamma", "alpha_beta", "z_plus", "composite"]

# Layer types that require special LRP handling
WEIGHTED_LAYERS = (nn.Linear, nn.Conv2d) if TORCH_AVAILABLE else ()
NORMALIZATION_LAYERS = (nn.BatchNorm1d, nn.BatchNorm2d) if TORCH_AVAILABLE else ()
ACTIVATION_LAYERS = (nn.ReLU, nn.LeakyReLU, nn.ELU, nn.Tanh, nn.Sigmoid, nn.GELU) if TORCH_AVAILABLE else ()
POOLING_LAYERS = (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d) if TORCH_AVAILABLE else ()
PASSTHROUGH_LAYERS = (nn.Dropout, nn.Dropout2d, nn.Flatten) if TORCH_AVAILABLE else ()


class LRPExplainer(BaseExplainer):
    """
    Layer-wise Relevance Propagation (LRP) explainer for neural networks.
    
    LRP decomposes the network output into relevance scores for each input
    feature by propagating relevance backward through the network layers.
    The key property is conservation: the sum of relevances at each layer
    equals the relevance at the layer above.
    
    Supports:
    - Fully connected networks (Linear + activations)
    - Convolutional networks (Conv2d + BatchNorm + pooling)
    - Mixed architectures
    
    Attributes:
        model: Model adapter (must be PyTorchAdapter)
        feature_names: List of feature names
        class_names: List of class names (for classification)
        rule: Propagation rule ("epsilon", "gamma", "alpha_beta", "z_plus", "composite")
        epsilon: Stabilization constant for epsilon rule
        gamma: Enhancement factor for gamma rule
        alpha: Positive contribution weight for alpha-beta rule
        beta: Negative contribution weight for alpha-beta rule
    
    Example:
        >>> explainer = LRPExplainer(adapter, feature_names, rule="epsilon")
        >>> explanation = explainer.explain(instance)
        >>> print(explanation.explanation_data["feature_attributions"])
    """
    
    def __init__(
        self,
        model,
        feature_names: List[str],
        class_names: Optional[List[str]] = None,
        rule: str = "epsilon",
        epsilon: float = 1e-6,
        gamma: float = 0.25,
        alpha: float = 2.0,
        beta: float = 1.0
    ):
        """
        Initialize the LRP explainer.
        
        Args:
            model: A PyTorchAdapter wrapping the model to explain.
            feature_names: List of input feature names.
            class_names: List of class names (for classification tasks).
            rule: Propagation rule to use:
                - "epsilon": LRP-ε with stabilization (default, recommended)
                - "gamma": LRP-γ enhancing positive contributions
                - "alpha_beta": LRP-αβ separating pos/neg contributions
                - "z_plus": LRP-z⁺ using only positive weights
                - "composite": Different rules for different layers
            epsilon: Small constant for numerical stability in epsilon rule.
                    Default: 1e-6
            gamma: Factor to enhance positive contributions in gamma rule.
                   Default: 0.25
            alpha: Weight for positive contributions in alpha-beta rule.
                   Must satisfy alpha - beta = 1. Default: 2.0
            beta: Weight for negative contributions in alpha-beta rule.
                  Must satisfy alpha - beta = 1. Default: 1.0
        
        Raises:
            TypeError: If model is not a PyTorchAdapter.
            ValueError: If rule is invalid or alpha-beta constraint violated.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for LRP. Install with: pip install torch"
            )
        
        super().__init__(model)
        
        # Validate model is PyTorchAdapter
        if not hasattr(model, 'model') or not isinstance(model.model, nn.Module):
            raise TypeError(
                "LRP requires a PyTorchAdapter wrapping a PyTorch model. "
                "Use: PyTorchAdapter(your_model, task='classification')"
            )
        
        # Validate rule
        if rule not in VALID_RULES:
            raise ValueError(
                f"Invalid rule: '{rule}'. Must be one of: {VALID_RULES}"
            )
        
        # Validate alpha-beta constraint
        if rule == "alpha_beta":
            if not np.isclose(alpha - beta, 1.0):
                raise ValueError(
                    f"For alpha-beta rule, alpha - beta must equal 1. "
                    f"Got alpha={alpha}, beta={beta}, difference={alpha - beta}"
                )
        
        self.feature_names = list(feature_names)
        self.class_names = list(class_names) if class_names else None
        self.rule = rule
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        
        # For composite rules
        self._layer_rules: Optional[Dict[int, str]] = None
        
        # Cache for layer information
        self._layers_info: Optional[List[Dict[str, Any]]] = None
    
    def _get_pytorch_model(self) -> nn.Module:
        """Get the underlying PyTorch model."""
        return self.model.model
    
    def _is_cnn_model(self) -> bool:
        """Check if the model's first weighted layer is Conv2d."""
        model = self._get_pytorch_model()
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                return True
            if isinstance(module, nn.Linear):
                return False
        return False
    
    def _prepare_input_tensor(self, instance: np.ndarray) -> torch.Tensor:
        """
        Prepare input tensor with correct shape for the model.
        
        For CNN models, preserves the spatial dimensions.
        For MLP models, flattens to 2D.
        
        Args:
            instance: Input array (1D for tabular, 3D for images)
        
        Returns:
            Tensor with batch dimension added and correct shape for model
        """
        instance = np.array(instance).astype(np.float32)
        original_shape = instance.shape
        
        model = self._get_pytorch_model()
        
        # Find first weighted layer to determine input type
        first_layer = None
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                first_layer = module
                break
        
        if isinstance(first_layer, nn.Conv2d):
            # CNN model - need 4D input (batch, channels, height, width)
            in_channels = first_layer.in_channels
            
            if len(original_shape) >= 3:
                # Already (C, H, W) format - just add batch dimension
                x = torch.tensor(instance).unsqueeze(0)
            elif len(original_shape) == 2:
                # (H, W) - assume single channel, add channel and batch dimensions
                x = torch.tensor(instance).unsqueeze(0).unsqueeze(0)
            else:
                # Flattened - try to infer spatial dimensions
                n_features = instance.size
                if n_features % in_channels == 0:
                    spatial_size = int(np.sqrt(n_features // in_channels))
                    if spatial_size * spatial_size * in_channels == n_features:
                        x = torch.tensor(instance.flatten()).reshape(
                            1, in_channels, spatial_size, spatial_size
                        )
                    else:
                        raise ValueError(
                            f"Cannot infer spatial dimensions for {n_features} features "
                            f"with {in_channels} channels"
                        )
                else:
                    raise ValueError(
                        f"Number of features ({n_features}) not divisible by "
                        f"input channels ({in_channels})"
                    )
        else:
            # MLP model - need 2D input (batch, features)
            x = torch.tensor(instance.flatten()).reshape(1, -1)
        
        return x.float()
    
    def _get_rule_for_layer(self, layer_idx: int, layer_type: str) -> str:
        """
        Get the propagation rule for a specific layer.
        
        Args:
            layer_idx: Index of the layer
            layer_type: Type of the layer (e.g., "Linear", "Conv2d")
        
        Returns:
            Rule name to use for this layer
        """
        if self.rule != "composite":
            return self.rule
        
        # Composite rule: check layer-specific rules
        if self._layer_rules and layer_idx in self._layer_rules:
            return self._layer_rules[layer_idx]
        
        # Default fallback for composite
        return "epsilon"
    
    def set_composite_rule(self, layer_rules: Dict[int, str]) -> "LRPExplainer":
        """
        Set layer-specific rules for composite LRP.
        
        This allows using different propagation rules for different layers,
        which is often beneficial. A common practice is:
        - z_plus for input/early layers (focuses on what's present)
        - epsilon for middle layers (balanced attribution)
        - epsilon or zero for final layers
        
        Args:
            layer_rules: Dictionary mapping layer indices to rule names.
                        Layers not in this dict use "epsilon" by default.
        
        Returns:
            Self for method chaining.
        
        Example:
            >>> explainer.set_composite_rule({
            ...     0: "z_plus",   # First layer
            ...     2: "epsilon",  # Middle layer
            ...     4: "epsilon"   # Final layer
            ... })
        """
        # Validate rules
        for idx, rule in layer_rules.items():
            if rule not in VALID_RULES and rule != "composite":
                raise ValueError(f"Invalid rule '{rule}' for layer {idx}")
        
        self._layer_rules = layer_rules
        return self
    
    # =========================================================================
    # Linear Layer LRP Rules
    # =========================================================================
    
    def _lrp_linear_epsilon(
        self,
        layer: nn.Linear,
        activation: torch.Tensor,
        relevance: torch.Tensor,
        epsilon: float
    ) -> torch.Tensor:
        """
        LRP-epsilon rule for linear layers.
        
        R_j = Σ_k (a_j * w_jk / (z_k + ε*sign(z_k))) * R_k
        """
        # Forward pass to get z
        z = torch.mm(activation, layer.weight.t())
        if layer.bias is not None:
            z = z + layer.bias
        
        # Stabilize: z + epsilon * sign(z)
        z_stabilized = z + epsilon * torch.sign(z)
        z_stabilized = torch.where(
            torch.abs(z_stabilized) < epsilon,
            torch.full_like(z_stabilized, epsilon),
            z_stabilized
        )
        
        # Compute relevance contribution: (R / z_stabilized) @ W
        s = relevance / z_stabilized
        c = torch.mm(s, layer.weight)
        
        return activation * c
    
    def _lrp_linear_gamma(
        self,
        layer: nn.Linear,
        activation: torch.Tensor,
        relevance: torch.Tensor,
        gamma: float
    ) -> torch.Tensor:
        """
        LRP-gamma rule for linear layers.
        Enhances positive contributions for sharper attributions.
        """
        w_plus = torch.clamp(layer.weight, min=0)
        w_modified = layer.weight + gamma * w_plus
        
        z = torch.mm(activation, w_modified.t())
        if layer.bias is not None:
            b_plus = torch.clamp(layer.bias, min=0)
            z = z + layer.bias + gamma * b_plus
        
        z_stabilized = z + self.epsilon * torch.sign(z)
        z_stabilized = torch.where(
            torch.abs(z_stabilized) < self.epsilon,
            torch.full_like(z_stabilized, self.epsilon),
            z_stabilized
        )
        
        s = relevance / z_stabilized
        c = torch.mm(s, w_modified)
        
        return activation * c
    
    def _lrp_linear_alpha_beta(
        self,
        layer: nn.Linear,
        activation: torch.Tensor,
        relevance: torch.Tensor,
        alpha: float,
        beta: float
    ) -> torch.Tensor:
        """
        LRP-alpha-beta rule for linear layers.
        Separates positive and negative contributions.
        """
        w_plus = torch.clamp(layer.weight, min=0)
        w_minus = torch.clamp(layer.weight, max=0)
        a_plus = torch.clamp(activation, min=0)
        
        z_plus = torch.mm(a_plus, w_plus.t())
        if layer.bias is not None:
            z_plus = z_plus + torch.clamp(layer.bias, min=0)
        
        z_minus = torch.mm(a_plus, w_minus.t())
        if layer.bias is not None:
            z_minus = z_minus + torch.clamp(layer.bias, max=0)
        
        z_plus_stable = z_plus + self.epsilon
        z_minus_stable = z_minus - self.epsilon
        z_minus_stable = torch.where(
            torch.abs(z_minus_stable) < self.epsilon,
            torch.full_like(z_minus_stable, -self.epsilon),
            z_minus_stable
        )
        
        s_plus = relevance / z_plus_stable
        s_minus = relevance / z_minus_stable
        
        c_plus = torch.mm(s_plus, w_plus)
        c_minus = torch.mm(s_minus, w_minus)
        
        return alpha * a_plus * c_plus - beta * a_plus * c_minus
    
    def _lrp_linear_z_plus(
        self,
        layer: nn.Linear,
        activation: torch.Tensor,
        relevance: torch.Tensor
    ) -> torch.Tensor:
        """
        LRP-z+ rule for linear layers.
        Only considers positive weights.
        """
        w_plus = torch.clamp(layer.weight, min=0)
        a_plus = torch.clamp(activation, min=0)
        
        z_plus = torch.mm(a_plus, w_plus.t())
        if layer.bias is not None:
            z_plus = z_plus + torch.clamp(layer.bias, min=0)
        
        z_plus_stable = z_plus + self.epsilon
        
        s = relevance / z_plus_stable
        c = torch.mm(s, w_plus)
        
        return a_plus * c
    
    def _propagate_linear(
        self,
        layer: nn.Linear,
        activation: torch.Tensor,
        relevance: torch.Tensor,
        rule: str
    ) -> torch.Tensor:
        """Propagate relevance through a linear layer."""
        if rule == "epsilon":
            return self._lrp_linear_epsilon(layer, activation, relevance, self.epsilon)
        elif rule == "gamma":
            return self._lrp_linear_gamma(layer, activation, relevance, self.gamma)
        elif rule == "alpha_beta":
            return self._lrp_linear_alpha_beta(layer, activation, relevance, self.alpha, self.beta)
        elif rule == "z_plus":
            return self._lrp_linear_z_plus(layer, activation, relevance)
        else:
            return self._lrp_linear_epsilon(layer, activation, relevance, self.epsilon)
    
    # =========================================================================
    # Conv2d Layer LRP Rules
    # =========================================================================
    
    def _lrp_conv2d_epsilon(
        self,
        layer: nn.Conv2d,
        activation: torch.Tensor,
        relevance: torch.Tensor,
        epsilon: float
    ) -> torch.Tensor:
        """
        LRP-epsilon rule for Conv2d layers.
        Uses convolution transpose for backward relevance propagation.
        """
        # Forward pass to get z
        z = F.conv2d(
            activation,
            layer.weight,
            bias=layer.bias,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups
        )
        
        # Stabilize
        z_stabilized = z + epsilon * torch.sign(z)
        z_stabilized = torch.where(
            torch.abs(z_stabilized) < epsilon,
            torch.full_like(z_stabilized, epsilon),
            z_stabilized
        )
        
        # Compute s = R / z
        s = relevance / z_stabilized
        
        # Backward pass using conv_transpose2d
        c = F.conv_transpose2d(
            s,
            layer.weight,
            bias=None,
            stride=layer.stride,
            padding=layer.padding,
            output_padding=0,
            groups=layer.groups,
            dilation=layer.dilation
        )
        
        # Handle output size mismatch
        if c.shape != activation.shape:
            # Pad or crop to match activation shape
            diff_h = activation.shape[2] - c.shape[2]
            diff_w = activation.shape[3] - c.shape[3]
            if diff_h > 0 or diff_w > 0:
                c = F.pad(c, [0, max(0, diff_w), 0, max(0, diff_h)])
            if diff_h < 0 or diff_w < 0:
                c = c[:, :, :activation.shape[2], :activation.shape[3]]
        
        return activation * c
    
    def _lrp_conv2d_gamma(
        self,
        layer: nn.Conv2d,
        activation: torch.Tensor,
        relevance: torch.Tensor,
        gamma: float
    ) -> torch.Tensor:
        """LRP-gamma rule for Conv2d layers."""
        w_plus = torch.clamp(layer.weight, min=0)
        w_modified = layer.weight + gamma * w_plus
        
        z = F.conv2d(
            activation,
            w_modified,
            bias=layer.bias,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups
        )
        
        if layer.bias is not None:
            b_plus = torch.clamp(layer.bias, min=0)
            # Bias is already added in conv2d, add gamma * b_plus
            z = z + gamma * b_plus.view(1, -1, 1, 1)
        
        z_stabilized = z + self.epsilon * torch.sign(z)
        z_stabilized = torch.where(
            torch.abs(z_stabilized) < self.epsilon,
            torch.full_like(z_stabilized, self.epsilon),
            z_stabilized
        )
        
        s = relevance / z_stabilized
        
        c = F.conv_transpose2d(
            s,
            w_modified,
            bias=None,
            stride=layer.stride,
            padding=layer.padding,
            output_padding=0,
            groups=layer.groups,
            dilation=layer.dilation
        )
        
        if c.shape != activation.shape:
            diff_h = activation.shape[2] - c.shape[2]
            diff_w = activation.shape[3] - c.shape[3]
            if diff_h > 0 or diff_w > 0:
                c = F.pad(c, [0, max(0, diff_w), 0, max(0, diff_h)])
            if diff_h < 0 or diff_w < 0:
                c = c[:, :, :activation.shape[2], :activation.shape[3]]
        
        return activation * c
    
    def _lrp_conv2d_z_plus(
        self,
        layer: nn.Conv2d,
        activation: torch.Tensor,
        relevance: torch.Tensor
    ) -> torch.Tensor:
        """LRP-z+ rule for Conv2d layers."""
        w_plus = torch.clamp(layer.weight, min=0)
        a_plus = torch.clamp(activation, min=0)
        
        z_plus = F.conv2d(
            a_plus,
            w_plus,
            bias=None,  # Ignore bias for z+
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups
        )
        
        if layer.bias is not None:
            z_plus = z_plus + torch.clamp(layer.bias, min=0).view(1, -1, 1, 1)
        
        z_plus_stable = z_plus + self.epsilon
        
        s = relevance / z_plus_stable
        
        c = F.conv_transpose2d(
            s,
            w_plus,
            bias=None,
            stride=layer.stride,
            padding=layer.padding,
            output_padding=0,
            groups=layer.groups,
            dilation=layer.dilation
        )
        
        if c.shape != a_plus.shape:
            diff_h = a_plus.shape[2] - c.shape[2]
            diff_w = a_plus.shape[3] - c.shape[3]
            if diff_h > 0 or diff_w > 0:
                c = F.pad(c, [0, max(0, diff_w), 0, max(0, diff_h)])
            if diff_h < 0 or diff_w < 0:
                c = c[:, :, :a_plus.shape[2], :a_plus.shape[3]]
        
        return a_plus * c
    
    def _propagate_conv2d(
        self,
        layer: nn.Conv2d,
        activation: torch.Tensor,
        relevance: torch.Tensor,
        rule: str
    ) -> torch.Tensor:
        """Propagate relevance through a Conv2d layer."""
        if rule == "epsilon":
            return self._lrp_conv2d_epsilon(layer, activation, relevance, self.epsilon)
        elif rule == "gamma":
            return self._lrp_conv2d_gamma(layer, activation, relevance, self.gamma)
        elif rule == "z_plus":
            return self._lrp_conv2d_z_plus(layer, activation, relevance)
        elif rule == "alpha_beta":
            # Alpha-beta for conv is complex, fall back to epsilon
            return self._lrp_conv2d_epsilon(layer, activation, relevance, self.epsilon)
        else:
            return self._lrp_conv2d_epsilon(layer, activation, relevance, self.epsilon)
    
    # =========================================================================
    # BatchNorm Layer LRP Rules
    # =========================================================================
    
    def _propagate_batchnorm(
        self,
        layer: Union[nn.BatchNorm1d, nn.BatchNorm2d],
        activation: torch.Tensor,
        relevance: torch.Tensor
    ) -> torch.Tensor:
        """
        Propagate relevance through BatchNorm layer.
        
        BatchNorm is an affine transformation: y = gamma * (x - mean) / std + beta
        We treat it as a linear scaling and propagate relevance proportionally.
        """
        # Get BatchNorm parameters
        if layer.running_mean is None or layer.running_var is None:
            # If no running stats, pass through
            return relevance
        
        mean = layer.running_mean
        var = layer.running_var
        eps = layer.eps
        
        # Compute the effective scale factor
        std = torch.sqrt(var + eps)
        
        if layer.weight is not None:
            scale = layer.weight / std
        else:
            scale = 1.0 / std
        
        # Reshape scale for broadcasting
        if isinstance(layer, nn.BatchNorm2d):
            scale = scale.view(1, -1, 1, 1)
        else:
            scale = scale.view(1, -1)
        
        # Relevance propagation: R_input = R_output (scaled back)
        # Since BN is essentially a rescaling, we redistribute proportionally
        return relevance / (scale + self.epsilon * torch.sign(scale))
    
    # =========================================================================
    # Activation Layer LRP Rules  
    # =========================================================================
    
    def _propagate_activation(
        self,
        layer: nn.Module,
        activation: torch.Tensor,
        relevance: torch.Tensor
    ) -> torch.Tensor:
        """
        Propagate relevance through activation layers (ReLU, etc.).
        
        For element-wise activations, relevance passes through unchanged
        to locations where the activation was positive.
        """
        if isinstance(layer, nn.ReLU):
            # ReLU: pass relevance where input was positive
            # Since we have post-activation values, we use them as mask
            mask = (activation > 0).float()
            return relevance * mask + relevance * (1 - mask)  # Actually just pass through
        elif isinstance(layer, (nn.LeakyReLU, nn.ELU)):
            # For leaky activations, relevance passes through
            return relevance
        elif isinstance(layer, (nn.Tanh, nn.Sigmoid)):
            # For bounded activations, relevance passes through
            return relevance
        else:
            # Default: pass through
            return relevance
    
    # =========================================================================
    # Pooling Layer LRP Rules
    # =========================================================================
    
    def _propagate_maxpool2d(
        self,
        layer: nn.MaxPool2d,
        activation: torch.Tensor,
        relevance: torch.Tensor
    ) -> torch.Tensor:
        """
        Propagate relevance through MaxPool2d.
        
        Relevance is distributed to the max locations (winner-take-all).
        """
        # Forward pass to get indices
        _, indices = F.max_pool2d(
            activation,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            return_indices=True,
            ceil_mode=layer.ceil_mode
        )
        
        # Unpool: place relevance at max locations
        unpooled = F.max_unpool2d(
            relevance,
            indices,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            output_size=activation.shape
        )
        
        return unpooled
    
    def _propagate_avgpool2d(
        self,
        layer: Union[nn.AvgPool2d, nn.AdaptiveAvgPool2d],
        activation: torch.Tensor,
        relevance: torch.Tensor
    ) -> torch.Tensor:
        """
        Propagate relevance through AvgPool2d.
        
        Relevance is distributed uniformly across pooling regions.
        """
        if isinstance(layer, nn.AdaptiveAvgPool2d):
            # For adaptive pooling, upsample relevance to input size
            return F.interpolate(
                relevance,
                size=activation.shape[2:],
                mode='nearest'
            )
        else:
            # For regular avg pooling, use nearest neighbor upsampling
            # and scale by pool area
            kernel_size = layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size, layer.kernel_size)
            
            upsampled = F.interpolate(
                relevance,
                size=activation.shape[2:],
                mode='nearest'
            )
            
            return upsampled
    
    # =========================================================================
    # Main LRP Computation
    # =========================================================================
    
    def _compute_lrp(
        self,
        instance: np.ndarray,
        target_class: Optional[int] = None,
        return_layer_relevances: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Compute LRP attributions for a single instance.
        
        Args:
            instance: Input instance (1D or multi-dimensional array)
            target_class: Target class for relevance initialization
            return_layer_relevances: If True, also return relevances at each layer
        
        Returns:
            If return_layer_relevances is False:
                Array of attribution scores for input features
            If return_layer_relevances is True:
                Tuple of (input_attributions, layer_relevances_dict)
        """
        model = self._get_pytorch_model()
        model.eval()
        
        # Prepare input with correct shape for model type (CNN vs MLP)
        x = self._prepare_input_tensor(instance)
        x.requires_grad_(False)
        
        # =====================================================================
        # Forward pass: collect activations at each layer
        # =====================================================================
        activations = OrderedDict()
        activations["input"] = x.clone()
        
        layer_list = []  # List of (idx, name, layer, input_activation)
        
        current = x
        
        # Handle Sequential models
        if isinstance(model, nn.Sequential):
            for idx, (name, layer) in enumerate(model.named_children()):
                layer_list.append((idx, name, layer, current.clone()))
                current = layer(current)
                activations[f"layer_{idx}_{name}"] = current.clone()
        else:
            # For non-Sequential models, use hooks
            hooks = []
            layer_data = OrderedDict()
            
            def make_hook(name):
                def hook(module, input, output):
                    inp = input[0] if isinstance(input, tuple) else input
                    layer_data[name] = {
                        "input": inp.clone().detach(),
                        "output": output.clone().detach() if isinstance(output, torch.Tensor) else output
                    }
                return hook
            
            # Register hooks on relevant layers
            idx = 0
            for name, module in model.named_modules():
                if isinstance(module, (*WEIGHTED_LAYERS, *NORMALIZATION_LAYERS, *ACTIVATION_LAYERS, *POOLING_LAYERS)):
                    hooks.append(module.register_forward_hook(make_hook(f"{idx}_{name}")))
                    idx += 1
            
            # Forward pass
            current = model(x)
            
            # Remove hooks
            for h in hooks:
                h.remove()
            
            # Build layer list from collected data
            for name, data in layer_data.items():
                idx_str, layer_name = name.split("_", 1)
                # Get the actual module
                module = dict(model.named_modules()).get(layer_name)
                if module is not None:
                    layer_list.append((int(idx_str), layer_name, module, data["input"]))
        
        output = current
        
        # =====================================================================
        # Initialize relevance at output layer
        # =====================================================================
        if target_class is not None:
            relevance = torch.zeros_like(output)
            relevance[0, target_class] = output[0, target_class]
        else:
            relevance = output.clone()
        
        # =====================================================================
        # Backward pass: propagate relevance through layers
        # =====================================================================
        layer_relevances = OrderedDict()
        layer_relevances["output"] = relevance.detach().cpu().numpy().flatten()
        
        # Reverse through layers
        for idx, name, layer, activation in reversed(layer_list):
            rule = self._get_rule_for_layer(idx, type(layer).__name__)
            
            # Propagate based on layer type
            if isinstance(layer, nn.Linear):
                # Flatten activation if needed
                if activation.dim() > 2:
                    activation = activation.flatten(1)
                if relevance.dim() > 2:
                    relevance = relevance.flatten(1)
                relevance = self._propagate_linear(layer, activation, relevance, rule)
            
            elif isinstance(layer, nn.Conv2d):
                relevance = self._propagate_conv2d(layer, activation, relevance, rule)
            
            elif isinstance(layer, NORMALIZATION_LAYERS):
                relevance = self._propagate_batchnorm(layer, activation, relevance)
            
            elif isinstance(layer, ACTIVATION_LAYERS):
                relevance = self._propagate_activation(layer, activation, relevance)
            
            elif isinstance(layer, nn.MaxPool2d):
                relevance = self._propagate_maxpool2d(layer, activation, relevance)
            
            elif isinstance(layer, (nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                relevance = self._propagate_avgpool2d(layer, activation, relevance)
            
            elif isinstance(layer, nn.Flatten):
                # Reshape relevance back to pre-flatten shape
                if activation.dim() > 2:
                    relevance = relevance.reshape(activation.shape)
            
            elif isinstance(layer, nn.Unflatten):
                # Unflatten in forward expands dimensions: (batch, features) -> (batch, *dims)
                # In backward, reshape relevance to match the flattened input activation
                relevance = relevance.view(activation.shape)
            
            elif isinstance(layer, PASSTHROUGH_LAYERS):
                # Dropout and other passthrough layers
                pass
            
            layer_relevances[f"layer_{idx}_{name}"] = relevance.detach().cpu().numpy().flatten()
        
        # Final relevance is the input attribution
        input_relevance = relevance.detach().cpu().numpy().flatten()
        layer_relevances["input"] = input_relevance
        
        if return_layer_relevances:
            return input_relevance, layer_relevances
        return input_relevance
    
    def explain(
        self,
        instance: np.ndarray,
        target_class: Optional[int] = None,
        return_convergence_delta: bool = False
    ) -> Explanation:
        """
        Generate LRP explanation for an instance.
        
        Args:
            instance: Numpy array of input features (1D for tabular, 
                     or multi-dimensional for images).
            target_class: For classification, which class to explain.
                         If None, uses the predicted class.
            return_convergence_delta: If True, include the convergence delta
                (difference between sum of attributions and target output).
                Should be close to 0 for correct LRP (conservation property).
        
        Returns:
            Explanation object with feature attributions.
        
        Example:
            >>> explanation = explainer.explain(instance)
            >>> print(explanation.explanation_data["feature_attributions"])
        """
        instance = np.array(instance).astype(np.float32)
        original_shape = instance.shape
        instance_flat = instance.flatten()
        
        # Determine target class if not specified
        if target_class is None and self.class_names:
            # Get prediction using properly shaped input
            model = self._get_pytorch_model()
            model.eval()
            with torch.no_grad():
                x = self._prepare_input_tensor(instance)
                output = model(x)
                target_class = int(torch.argmax(output, dim=1).item())
        
        # Compute LRP attributions
        attributions_raw = self._compute_lrp(instance, target_class)
        
        # Build attributions dict
        if len(self.feature_names) == len(attributions_raw):
            attributions = {
                fname: float(attributions_raw[i])
                for i, fname in enumerate(self.feature_names)
            }
        else:
            # For images or mismatched feature names, use indices
            attributions = {
                f"feature_{i}": float(attributions_raw[i])
                for i in range(len(attributions_raw))
            }
        
        # Determine class name
        if self.class_names and target_class is not None:
            label_name = self.class_names[target_class]
        else:
            label_name = f"class_{target_class}" if target_class is not None else "output"
        
        explanation_data = {
            "feature_attributions": attributions,
            "attributions_raw": [float(x) for x in attributions_raw],
            "rule": self.rule,
            "epsilon": self.epsilon if self.rule in ["epsilon", "composite"] else None,
            "gamma": self.gamma if self.rule == "gamma" else None,
            "alpha": self.alpha if self.rule == "alpha_beta" else None,
            "beta": self.beta if self.rule == "alpha_beta" else None,
            "input_shape": list(original_shape)
        }
        
        # Compute convergence delta (conservation check)
        if return_convergence_delta:
            model = self._get_pytorch_model()
            model.eval()
            
            with torch.no_grad():
                # Use the helper method to get properly shaped input
                x = self._prepare_input_tensor(instance)
                output = model(x)
                
                if target_class is not None:
                    target_output = output[0, target_class].item()
                else:
                    target_output = output.sum().item()
            
            attribution_sum = sum(attributions.values())
            convergence_delta = abs(target_output - attribution_sum)
            
            explanation_data["target_output"] = float(target_output)
            explanation_data["attribution_sum"] = float(attribution_sum)
            explanation_data["convergence_delta"] = float(convergence_delta)
        
        return Explanation(
            explainer_name="LRP",
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
        
        Args:
            X: Array of instances. For tabular: (n_samples, n_features).
               For images: (n_samples, channels, height, width) or similar.
            target_class: Target class for all instances. If None,
                         uses predicted class for each instance.
        
        Returns:
            List of Explanation objects.
        """
        X = np.array(X)
        
        # Handle single instance
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # For multi-dimensional data (images), first dim is batch
        n_samples = X.shape[0]
        
        return [
            self.explain(X[i], target_class=target_class)
            for i in range(n_samples)
        ]
    
    def explain_with_layer_relevances(
        self,
        instance: np.ndarray,
        target_class: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compute LRP with layer-wise relevance scores for detailed analysis.
        
        This method returns relevance scores at each layer, which is useful
        for understanding how relevance flows through the network and
        verifying the conservation property.
        
        Args:
            instance: Input instance.
            target_class: Target class for relevance computation.
        
        Returns:
            Dictionary containing:
                - input_relevances: Final attribution scores for input features
                - layer_relevances: Dict mapping layer names to relevance arrays
                - target_class: The target class used
                - rule: The rule used for computation
        """
        instance = np.array(instance).astype(np.float32)
        
        # Determine target class if not specified
        if target_class is None and self.class_names:
            # Get prediction using properly shaped input
            model = self._get_pytorch_model()
            model.eval()
            with torch.no_grad():
                x = self._prepare_input_tensor(instance)
                output = model(x)
                target_class = int(torch.argmax(output, dim=1).item())
        
        # Compute LRP with layer relevances
        input_relevances, layer_relevances = self._compute_lrp(
            instance, target_class, return_layer_relevances=True
        )
        
        return {
            "input_relevances": [float(x) for x in input_relevances],
            "layer_relevances": {
                name: [float(x) for x in rel] if isinstance(rel, np.ndarray) else float(rel)
                for name, rel in layer_relevances.items()
            },
            "target_class": target_class,
            "rule": self.rule,
            "feature_names": self.feature_names
        }
    
    def compare_rules(
        self,
        instance: np.ndarray,
        target_class: Optional[int] = None,
        rules: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare different LRP rules on the same instance.
        
        Useful for understanding how different rules affect attributions
        and for selecting the most appropriate rule for your use case.
        
        Args:
            instance: Input instance.
            target_class: Target class for comparison.
            rules: List of rules to compare. If None, compares all rules.
        
        Returns:
            Dictionary mapping rule names to their attribution results.
        """
        instance = np.array(instance).astype(np.float32)
        
        # Determine target class
        if target_class is None and self.class_names:
            # Get prediction using properly shaped input
            model = self._get_pytorch_model()
            model.eval()
            with torch.no_grad():
                x = self._prepare_input_tensor(instance)
                output = model(x)
                target_class = int(torch.argmax(output, dim=1).item())
        
        if rules is None:
            rules = ["epsilon", "gamma", "alpha_beta", "z_plus"]
        
        results = {}
        
        # Save original settings
        original_rule = self.rule
        
        for rule in rules:
            self.rule = rule
            
            try:
                attributions = self._compute_lrp(instance, target_class)
                
                # Find top feature
                top_idx = int(np.argmax(np.abs(attributions)))
                if top_idx < len(self.feature_names):
                    top_feature = self.feature_names[top_idx]
                else:
                    top_feature = f"feature_{top_idx}"
                
                results[rule] = {
                    "attributions": [float(x) for x in attributions],
                    "top_feature": top_feature,
                    "top_attribution": float(attributions[top_idx]),
                    "attribution_sum": float(np.sum(attributions)),
                    "attribution_range": (float(np.min(attributions)), float(np.max(attributions)))
                }
            except Exception as e:
                results[rule] = {"error": str(e)}
        
        # Restore original rule
        self.rule = original_rule
        
        return results
