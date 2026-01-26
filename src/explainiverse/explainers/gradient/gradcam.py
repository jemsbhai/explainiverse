# src/explainiverse/explainers/gradient/gradcam.py
"""
GradCAM and GradCAM++ - Visual Explanations for CNNs.

GradCAM produces visual explanations by highlighting important regions
in an image that contribute to the model's prediction. It uses gradients
flowing into the final convolutional layer to produce a coarse localization map.

GradCAM++ improves upon GradCAM by using a weighted combination of positive
partial derivatives, providing better localization for multiple instances
of the same class.

References:
    GradCAM: Selvaraju et al., 2017 - "Grad-CAM: Visual Explanations from
    Deep Networks via Gradient-based Localization"
    https://arxiv.org/abs/1610.02391
    
    GradCAM++: Chattopadhay et al., 2018 - "Grad-CAM++: Generalized Gradient-based
    Visual Explanations for Deep Convolutional Networks"
    https://arxiv.org/abs/1710.11063

Example:
    from explainiverse.explainers.gradient import GradCAMExplainer
    from explainiverse.adapters import PyTorchAdapter
    
    # For a CNN model
    adapter = PyTorchAdapter(cnn_model, task="classification")
    
    explainer = GradCAMExplainer(
        model=adapter,
        target_layer="layer4",  # Last conv layer
        class_names=class_names
    )
    
    explanation = explainer.explain(image)
    heatmap = explanation.explanation_data["heatmap"]
"""

import numpy as np
from typing import List, Optional, Tuple, Union

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation


class GradCAMExplainer(BaseExplainer):
    """
    GradCAM and GradCAM++ explainer for CNNs.
    
    Produces visual heatmaps showing which regions of an input image
    are most important for the model's prediction.
    
    Attributes:
        model: PyTorchAdapter wrapping a CNN model
        target_layer: Name of the convolutional layer to use
        class_names: List of class names
        method: "gradcam" or "gradcam++"
    """
    
    def __init__(
        self,
        model,
        target_layer: str,
        class_names: Optional[List[str]] = None,
        method: str = "gradcam"
    ):
        """
        Initialize the GradCAM explainer.
        
        Args:
            model: A PyTorchAdapter wrapping a CNN model.
            target_layer: Name of the target convolutional layer.
                         Usually the last conv layer before the classifier.
                         Use adapter.list_layers() to see available layers.
            class_names: List of class names for classification.
            method: "gradcam" for standard GradCAM, "gradcam++" for improved version.
        """
        super().__init__(model)
        
        # Validate model has layer access
        if not hasattr(model, 'get_layer_gradients'):
            raise TypeError(
                "Model adapter must have get_layer_gradients() method. "
                "Use PyTorchAdapter for PyTorch models."
            )
        
        self.target_layer = target_layer
        self.class_names = list(class_names) if class_names else None
        self.method = method.lower()
        
        if self.method not in ["gradcam", "gradcam++"]:
            raise ValueError(f"Method must be 'gradcam' or 'gradcam++', got '{method}'")
    
    def _compute_gradcam(
        self,
        activations: np.ndarray,
        gradients: np.ndarray
    ) -> np.ndarray:
        """
        Compute standard GradCAM heatmap.
        
        GradCAM = ReLU(sum_k(alpha_k * A^k))
        where alpha_k = global_avg_pool(gradients for channel k)
        """
        # Global average pooling of gradients to get weights
        # activations shape: (batch, channels, height, width)
        # gradients shape: (batch, channels, height, width)
        
        # For each channel, compute the average gradient (importance weight)
        weights = np.mean(gradients, axis=(2, 3), keepdims=True)  # (batch, channels, 1, 1)
        
        # Weighted combination of activation maps
        cam = np.sum(weights * activations, axis=1)  # (batch, height, width)
        
        # Apply ReLU (we only care about positive influence)
        cam = np.maximum(cam, 0)
        
        return cam
    
    def _compute_gradcam_plusplus(
        self,
        activations: np.ndarray,
        gradients: np.ndarray
    ) -> np.ndarray:
        """
        Compute GradCAM++ heatmap.
        
        GradCAM++ uses higher-order derivatives to weight the gradients,
        providing better localization especially for multiple instances.
        """
        # First derivative
        grad_2 = gradients ** 2
        grad_3 = gradients ** 3
        
        # Sum over spatial dimensions for denominator
        sum_activations = np.sum(activations, axis=(2, 3), keepdims=True)
        
        # Avoid division by zero
        eps = 1e-8
        
        # Alpha coefficients (pixel-wise weights)
        alpha_num = grad_2
        alpha_denom = 2 * grad_2 + sum_activations * grad_3 + eps
        alpha = alpha_num / alpha_denom
        
        # Set alpha to 0 where gradients are 0
        alpha = np.where(gradients != 0, alpha, 0)
        
        # Weights are sum of (alpha * ReLU(gradients))
        weights = np.sum(alpha * np.maximum(gradients, 0), axis=(2, 3), keepdims=True)
        
        # Weighted combination
        cam = np.sum(weights * activations, axis=1)
        
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        return cam
    
    def _normalize_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        """Normalize heatmap to [0, 1] range."""
        heatmap = heatmap.squeeze()
        
        min_val = heatmap.min()
        max_val = heatmap.max()
        
        if max_val - min_val > 1e-8:
            heatmap = (heatmap - min_val) / (max_val - min_val)
        else:
            heatmap = np.zeros_like(heatmap)
        
        return heatmap
    
    def _resize_heatmap(
        self,
        heatmap: np.ndarray,
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Resize heatmap to match input image size.
        
        Uses simple bilinear-like interpolation without requiring scipy/cv2.
        """
        h, w = heatmap.shape
        target_h, target_w = target_size
        
        # Create coordinate grids
        y_ratio = h / target_h
        x_ratio = w / target_w
        
        y_coords = np.arange(target_h) * y_ratio
        x_coords = np.arange(target_w) * x_ratio
        
        # Get integer indices and fractions
        y_floor = np.floor(y_coords).astype(int)
        x_floor = np.floor(x_coords).astype(int)
        
        y_ceil = np.minimum(y_floor + 1, h - 1)
        x_ceil = np.minimum(x_floor + 1, w - 1)
        
        y_frac = y_coords - y_floor
        x_frac = x_coords - x_floor
        
        # Bilinear interpolation
        resized = np.zeros((target_h, target_w))
        for i in range(target_h):
            for j in range(target_w):
                top_left = heatmap[y_floor[i], x_floor[j]]
                top_right = heatmap[y_floor[i], x_ceil[j]]
                bottom_left = heatmap[y_ceil[i], x_floor[j]]
                bottom_right = heatmap[y_ceil[i], x_ceil[j]]
                
                top = top_left * (1 - x_frac[j]) + top_right * x_frac[j]
                bottom = bottom_left * (1 - x_frac[j]) + bottom_right * x_frac[j]
                
                resized[i, j] = top * (1 - y_frac[i]) + bottom * y_frac[i]
        
        return resized
    
    def explain(
        self,
        image: np.ndarray,
        target_class: Optional[int] = None,
        resize_to_input: bool = True
    ) -> Explanation:
        """
        Generate GradCAM explanation for an image.
        
        Args:
            image: Input image as numpy array. Expected shapes:
                   - (C, H, W) for single image
                   - (1, C, H, W) for batched single image
                   - (H, W, C) will be transposed automatically
            target_class: Class to explain. If None, uses predicted class.
            resize_to_input: If True, resize heatmap to match input size.
        
        Returns:
            Explanation object with heatmap and metadata.
        """
        image = np.array(image, dtype=np.float32)
        
        # Handle different input shapes
        if image.ndim == 3:
            # Could be (C, H, W) or (H, W, C)
            if image.shape[0] in [1, 3, 4]:  # Likely (C, H, W)
                image = image[np.newaxis, ...]  # Add batch dim
            else:  # Likely (H, W, C)
                image = np.transpose(image, (2, 0, 1))[np.newaxis, ...]
        elif image.ndim == 4:
            pass  # Already (N, C, H, W)
        else:
            raise ValueError(f"Expected 3D or 4D input, got shape {image.shape}")
        
        input_size = (image.shape[2], image.shape[3])  # (H, W)
        
        # Get activations and gradients for target layer
        activations, gradients = self.model.get_layer_gradients(
            image,
            layer_name=self.target_layer,
            target_class=target_class
        )
        
        # Ensure 4D: (batch, channels, height, width)
        if activations.ndim == 2:
            # Fully connected layer output, reshape
            side = int(np.sqrt(activations.shape[1]))
            activations = activations.reshape(1, 1, side, side)
            gradients = gradients.reshape(1, 1, side, side)
        elif activations.ndim == 3:
            activations = activations[np.newaxis, ...]
            gradients = gradients[np.newaxis, ...]
        
        # Compute CAM based on method
        if self.method == "gradcam":
            cam = self._compute_gradcam(activations, gradients)
        else:  # gradcam++
            cam = self._compute_gradcam_plusplus(activations, gradients)
        
        # Normalize to [0, 1]
        heatmap = self._normalize_heatmap(cam)
        
        # Optionally resize to input size
        if resize_to_input and heatmap.shape != input_size:
            heatmap = self._resize_heatmap(heatmap, input_size)
        
        # Determine target class info
        if target_class is None:
            predictions = self.model.predict(image)
            target_class = int(np.argmax(predictions))
        
        if self.class_names and target_class < len(self.class_names):
            label_name = self.class_names[target_class]
        else:
            label_name = f"class_{target_class}"
        
        return Explanation(
            explainer_name=f"GradCAM" if self.method == "gradcam" else "GradCAM++",
            target_class=label_name,
            explanation_data={
                "heatmap": heatmap.tolist(),
                "heatmap_shape": list(heatmap.shape),
                "target_layer": self.target_layer,
                "method": self.method,
                "input_shape": list(image.shape)
            }
        )
    
    def explain_batch(
        self,
        images: np.ndarray,
        target_class: Optional[int] = None
    ) -> List[Explanation]:
        """
        Generate explanations for multiple images.
        
        Args:
            images: Batch of images (N, C, H, W).
            target_class: Target class for all images.
        
        Returns:
            List of Explanation objects.
        """
        images = np.array(images)
        
        return [
            self.explain(images[i], target_class=target_class)
            for i in range(images.shape[0])
        ]
    
    def get_overlay(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: str = "jet"
    ) -> np.ndarray:
        """
        Create an overlay of the heatmap on the original image.
        
        This is a simple implementation without matplotlib/cv2 dependencies.
        For better visualizations, use the heatmap with your preferred
        visualization library.
        
        Args:
            image: Original image (H, W, 3) in [0, 255] or [0, 1] range.
            heatmap: GradCAM heatmap (H, W) in [0, 1] range.
            alpha: Transparency of the heatmap overlay.
            colormap: Color scheme (currently only "jet" supported).
        
        Returns:
            Overlaid image as numpy array (H, W, 3) in [0, 1] range.
        """
        image = np.array(image)
        heatmap = np.array(heatmap)
        
        # Normalize image to [0, 1]
        if image.max() > 1:
            image = image / 255.0
        
        # Handle channel-first format
        if image.ndim == 3 and image.shape[0] in [1, 3]:
            image = np.transpose(image, (1, 2, 0))
        
        # Simple jet colormap approximation
        def jet_colormap(x):
            """Simple jet colormap: blue -> cyan -> green -> yellow -> red"""
            r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
            g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
            b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
            return np.stack([r, g, b], axis=-1)
        
        # Apply colormap to heatmap
        colored_heatmap = jet_colormap(heatmap)
        
        # Ensure same size
        if colored_heatmap.shape[:2] != image.shape[:2]:
            colored_heatmap = self._resize_heatmap(
                colored_heatmap.mean(axis=-1),
                image.shape[:2]
            )
            colored_heatmap = jet_colormap(colored_heatmap)
        
        # Blend
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        
        overlay = (1 - alpha) * image + alpha * colored_heatmap
        overlay = np.clip(overlay, 0, 1)
        
        return overlay
