"""Gradient-weighted class activation mapping for spatial CNN layers.

This module implements Grad-CAM as defined by Selvaraju et al. (ICCV 2017):
the gradient of one scalar target is globally averaged over each activation
channel, the channels are combined with those weights, and a final ReLU keeps
positive evidence.

The former ``method="gradcam++"`` option is intentionally unavailable.  The
old implementation substituted powers of first derivatives for the second and
third derivatives in the Grad-CAM++ definition without establishing the
piecewise-linear-tail assumptions required by the paper's closed form.  Calling
that path a general Grad-CAM++ implementation was therefore not justified.

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-Based Localization", ICCV 2017.
    https://arxiv.org/abs/1610.02391
"""

from __future__ import annotations

from contextlib import contextmanager
from numbers import Integral
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation
from explainiverse.explainers._validation import as_real_array, validate_name_sequence
from explainiverse.explainers.gradient._image_layout import (
    channel_axis_for_layout,
    image_to_nchw,
    validate_image_layout,
)
from explainiverse.explainers.gradient._input import (
    as_floating_array,
    scale_safe_mean_std,
    scale_safe_product_sum,
    scale_safe_spatial_mean_product_sum,
)
from explainiverse.explainers.gradient._model_state import preserve_adapter_model_eval

Target = Optional[Union[int, np.integer]]


def _model_has_unflatten(adapter) -> bool:
    """Return whether the wrapped torch model contains an ``Unflatten``."""
    try:
        import torch.nn as nn

        return any(isinstance(module, nn.Unflatten) for module in adapter.model.modules())
    except (AttributeError, ImportError):
        return False


def _unflatten_spatial_size(adapter) -> Optional[Tuple[int, int]]:
    """Return the last two dimensions of the first ``Unflatten`` layer."""
    try:
        import torch.nn as nn

        for module in adapter.model.modules():
            if isinstance(module, nn.Unflatten):
                dimensions = tuple(
                    int(value[1]) if isinstance(value, tuple) else int(value)
                    for value in module.unflattened_size
                )
                if len(dimensions) >= 2 and dimensions[-2] > 0 and dimensions[-1] > 0:
                    return dimensions[-2], dimensions[-1]
    except (AttributeError, ImportError, TypeError, ValueError):
        pass
    return None


def _validate_input_layout(input_layout: str) -> str:
    """Backward-compatible alias for the shared explicit layout validator."""

    return validate_image_layout(input_layout)


def _get_layer_output_with_trace(adapter, data, *, layer_name: str, **options):
    """Use atomic trace evidence when the adapter exposes it."""

    traced = getattr(adapter, "get_layer_output_with_trace", None)
    if callable(traced):
        return traced(data, layer_name=layer_name, **options)
    return adapter.get_layer_output(data, layer_name=layer_name, **options), None


def _get_layer_gradients_with_trace(
    adapter, data, *, layer_name: str, target_class: int, **options
):
    """Use atomic trace evidence while preserving third-party adapter support."""

    traced = getattr(adapter, "get_layer_gradients_with_trace", None)
    if callable(traced):
        return traced(
            data,
            layer_name=layer_name,
            target_class=target_class,
            **options,
        )
    activations, gradients = adapter.get_layer_gradients(
        data,
        layer_name=layer_name,
        target_class=target_class,
        **options,
    )
    return activations, gradients, None


def _prepare_single_input(
    adapter, image: np.ndarray, input_layout: str = "auto"
) -> Tuple[np.ndarray, Tuple[int, int], str]:
    """Normalize one declared image (or an explicit flat-unflatten input)."""
    input_layout = _validate_input_layout(input_layout)
    prepared = as_floating_array(image, name="image")

    if prepared.size == 0:
        raise ValueError("image must not be empty")

    if prepared.ndim in (1, 2) and _model_has_unflatten(adapter):
        if input_layout != "auto":
            raise ValueError("input_layout must be 'auto' for flat Unflatten inputs")
        if prepared.ndim == 1:
            prepared = prepared[np.newaxis, ...]
        elif prepared.shape[0] != 1:
            raise ValueError(
                "explain() accepts exactly one flat input; use explain_batch() "
                f"for a batch of size {prepared.shape[0]}"
            )
        resolved_layout = "flat"
    elif prepared.ndim in (2, 3, 4):
        prepared, resolved_layout = image_to_nchw(prepared, input_layout)
    else:
        raise ValueError(
            "Expected one declared HW/CHW/HWC/NCHW/NHWC image or a flat "
            "input for a model with an Unflatten layer; "
            f"got shape {prepared.shape}"
        )

    if prepared.ndim == 4:
        if any(dimension <= 0 for dimension in prepared.shape[1:]):
            raise ValueError(f"image dimensions must be positive; got {prepared.shape}")
        input_size = int(prepared.shape[-2]), int(prepared.shape[-1])
    else:
        resolved_size = _unflatten_spatial_size(adapter)
        if resolved_size is None:
            raise ValueError(
                "The model's Unflatten layer does not expose a two-dimensional "
                "spatial output size"
            )
        input_size = resolved_size

    return np.ascontiguousarray(prepared), input_size, resolved_layout


def _validate_target(target_class: Target) -> Optional[int]:
    if target_class is None:
        return None
    if isinstance(target_class, bool) or not isinstance(target_class, Integral):
        raise TypeError("target_class must be an integer output index or None")
    target = int(target_class)
    if target < 0:
        raise ValueError("target_class must be non-negative")
    return target


def _resolve_target(adapter, image: np.ndarray, target_class: Target) -> int:
    """Resolve one fixed target from the original, unperturbed input."""
    explicit = _validate_target(target_class)
    predictions = as_real_array(
        adapter.predict(image),
        name="model predictions",
        require_finite=True,
    )
    if predictions.ndim != 2 or predictions.shape[0] != 1 or predictions.shape[1] < 1:
        raise ValueError(
            "CAM expected model.predict() to return shape (1, n_outputs); "
            f"got {predictions.shape}"
        )
    n_outputs = int(predictions.shape[1])
    if explicit is not None:
        if explicit >= n_outputs:
            raise ValueError(f"target_class must be in [0, {n_outputs - 1}], got {explicit}")
        return explicit

    if getattr(adapter, "task", None) == "regression":
        if n_outputs != 1:
            raise ValueError(
                "An explicit target_class output index is required for " "multi-output regression"
            )
        return 0
    return int(np.argmax(predictions[0]))


def _validate_spatial_pair(
    activations: np.ndarray, gradients: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate one spatial activation tensor and its matching gradients."""
    activations = as_real_array(
        activations,
        name="target-layer activations",
        require_finite=True,
    )
    gradients = as_real_array(
        gradients,
        name="target-layer gradients",
        require_finite=True,
    )
    if activations.ndim != 4:
        raise ValueError(
            "The target layer must return a spatial tensor with shape "
            f"(1, channels, height, width); got {activations.shape}"
        )
    if activations.shape[0] != 1:
        raise ValueError(
            "A single-image CAM requires one target-layer activation batch; "
            f"got {activations.shape[0]}"
        )
    if activations.shape != gradients.shape:
        raise ValueError(
            "target-layer activations and gradients must have identical shapes; "
            f"got {activations.shape} and {gradients.shape}"
        )
    if any(dimension <= 0 for dimension in activations.shape[1:]):
        raise ValueError(f"target-layer dimensions must be positive; got {activations.shape}")
    return activations, gradients


def _validate_spatial_activations(activations: np.ndarray) -> np.ndarray:
    activations = as_real_array(
        activations,
        name="target-layer activations",
        require_finite=True,
    )
    if activations.ndim != 4 or activations.shape[0] != 1:
        raise ValueError(
            "The target layer must return shape (1, channels, height, width); "
            f"got {activations.shape}"
        )
    if any(dimension <= 0 for dimension in activations.shape[1:]):
        raise ValueError(f"target-layer dimensions must be positive; got {activations.shape}")
    return activations


def _normalize_cam(cam: np.ndarray) -> np.ndarray:
    """Min-max normalize one two-dimensional CAM using the common display rule."""
    cam = as_real_array(cam, name="CAM", dtype=np.float64, require_finite=True)
    if cam.ndim != 2 or cam.size == 0:
        raise ValueError(f"CAM must be a non-empty 2D array; got shape {cam.shape}")
    minimum = float(np.min(cam))
    maximum = float(np.max(cam))
    if minimum == maximum:
        return np.zeros(cam.shape, dtype=np.float64)

    # Scale before subtracting.  Direct ``maximum - minimum`` can overflow for
    # opposite-sign finite extremes, while an absolute epsilon threshold
    # incorrectly erases genuine structure at very small magnitudes.
    scale = max(abs(minimum), abs(maximum))
    scaled = cam / scale
    scaled_minimum = minimum / scale
    scaled_span = maximum / scale - scaled_minimum
    return (scaled - scaled_minimum) / scaled_span


def _cam_normalization_metadata(cam: np.ndarray) -> dict:
    """Describe the exact map consumed by display min-max normalization."""
    values = as_real_array(cam, name="CAM", dtype=np.float64, require_finite=True)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("CAM normalization metadata requires one finite non-empty 2D map")
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    degenerate = bool(maximum == minimum)
    return {
        "normalization_input_min": minimum,
        "normalization_input_max": maximum,
        "normalization_degenerate": degenerate,
        "constant_map_value": minimum if degenerate else None,
    }


def _resize_2d(heatmap: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize a 2D map with half-pixel bilinear interpolation.

    The coordinate rule matches ``torch.nn.functional.interpolate`` with
    ``align_corners=False``.  Resizing is post-processing and is not part of
    the mathematical Grad-CAM definition.
    """
    source = as_real_array(
        heatmap,
        name="heatmap",
        dtype=np.float64,
        require_finite=True,
    )
    if source.ndim != 2 or source.size == 0:
        raise ValueError(f"heatmap must be a non-empty 2D array; got {source.shape}")
    if len(target_size) != 2:
        raise ValueError("target_size must contain (height, width)")
    target_h, target_w = (int(target_size[0]), int(target_size[1]))
    if target_h <= 0 or target_w <= 0:
        raise ValueError("target_size dimensions must be positive")
    if source.shape == (target_h, target_w):
        return source.copy()

    height, width = source.shape
    y = (np.arange(target_h, dtype=np.float64) + 0.5) * height / target_h - 0.5
    x = (np.arange(target_w, dtype=np.float64) + 0.5) * width / target_w - 0.5
    y = np.clip(y, 0.0, height - 1.0)
    x = np.clip(x, 0.0, width - 1.0)
    y0 = np.floor(y).astype(np.intp)
    x0 = np.floor(x).astype(np.intp)
    y1 = np.minimum(y0 + 1, height - 1)
    x1 = np.minimum(x0 + 1, width - 1)
    y_fraction = y - y0
    x_fraction = x - x0

    top = (
        source[y0[:, None], x0[None, :]] * (1.0 - x_fraction[None, :])
        + source[y0[:, None], x1[None, :]] * x_fraction[None, :]
    )
    bottom = (
        source[y1[:, None], x0[None, :]] * (1.0 - x_fraction[None, :])
        + source[y1[:, None], x1[None, :]] * x_fraction[None, :]
    )
    return top * (1.0 - y_fraction[:, None]) + bottom * y_fraction[:, None]


@contextmanager
def _preserve_adapter_model_state(adapter, *, preserve_gradients: bool) -> Iterator[None]:
    """Run deterministic CAM forwards without leaking training/gradient state."""
    with preserve_adapter_model_eval(adapter, preserve_gradients=preserve_gradients):
        yield


def _targets_for_batch(
    target_class: Optional[Union[int, np.integer, Sequence[int], np.ndarray]],
    batch_size: int,
) -> List[Target]:
    if target_class is None:
        return [None] * batch_size
    if isinstance(target_class, Integral) and not isinstance(target_class, bool):
        target = _validate_target(int(target_class))
        return [target] * batch_size

    targets = np.asarray(target_class)
    if targets.ndim != 1 or targets.shape[0] != batch_size:
        raise ValueError(f"Expected one target per image ({batch_size}), got shape {targets.shape}")
    if not np.issubdtype(targets.dtype, np.integer) or targets.dtype == np.bool_:
        raise TypeError("batch targets must contain integer output indices")
    return [_validate_target(int(value)) for value in targets]


class GradCAMExplainer(BaseExplainer):
    """Grad-CAM for one spatial target layer and one fixed scalar target."""

    def __init__(
        self,
        model,
        target_layer: str,
        class_names: Optional[List[str]] = None,
        method: str = "gradcam",
        input_layout: str = "auto",
        target_occurrence: Optional[int] = None,
    ):
        super().__init__(model)
        if not callable(getattr(model, "get_layer_gradients", None)):
            raise TypeError(
                "model must provide get_layer_gradients(); use PyTorchAdapter " "for PyTorch CNNs"
            )
        if not isinstance(target_layer, str) or not target_layer.strip():
            raise ValueError("target_layer must be a non-empty layer name")
        if not isinstance(method, str):
            raise TypeError("method must be 'gradcam'")
        normalized_method = method.strip().lower().replace("-", "")
        if normalized_method in {"gradcam++", "gradcamplusplus"}:
            raise NotImplementedError(
                "Grad-CAM++ is not exposed: the adapter supplies first "
                "derivatives only, while the paper defines second- and "
                "third-order derivatives (or a conditional closed form)."
            )
        if normalized_method != "gradcam":
            raise ValueError(f"method must be 'gradcam', got {method!r}")

        self.target_layer = target_layer
        self.class_names = validate_name_sequence(
            class_names,
            name="class_names",
            allow_none=True,
        )
        self.method = "gradcam"
        self.input_layout = _validate_input_layout(input_layout)
        if target_occurrence is not None and (
            isinstance(target_occurrence, bool) or not isinstance(target_occurrence, Integral)
        ):
            raise TypeError("target_occurrence must be a non-negative integer or None")
        if target_occurrence is not None and int(target_occurrence) < 0:
            raise ValueError("target_occurrence must be a non-negative integer or None")
        self.target_occurrence = None if target_occurrence is None else int(target_occurrence)

    @staticmethod
    def _compute_gradcam(activations: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Apply Selvaraju et al. equations 1 and 2 before display scaling."""
        activations, gradients = _validate_spatial_pair(activations, gradients)
        try:
            weights = scale_safe_mean_std(gradients, axis=(2, 3))[0][:, :, None, None]
            raw_cam = scale_safe_product_sum(weights, activations, axis=1)[0]
        except FloatingPointError:
            raw_cam = scale_safe_spatial_mean_product_sum(activations, gradients)[0]
        return np.maximum(raw_cam, 0.0)

    def _label_for_target(self, target: int) -> str:
        if self.class_names is not None and target < len(self.class_names):
            return str(self.class_names[target])
        prefix = "output" if getattr(self.model, "task", None) == "regression" else "class"
        return f"{prefix}_{target}"

    def explain(
        self,
        instance: np.ndarray,
        target_class: Target = None,
        resize_to_input: bool = True,
        **kwargs: object,
    ) -> Explanation:
        """Explain one image (or one flat input reshaped by the model)."""
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected GradCAM option(s): {unexpected}")
        if not isinstance(resize_to_input, (bool, np.bool_)):
            raise TypeError("resize_to_input must be a boolean")
        prepared, input_size, resolved_layout = _prepare_single_input(
            self.model, instance, input_layout=self.input_layout
        )

        with _preserve_adapter_model_state(self.model, preserve_gradients=True):
            target = _resolve_target(self.model, prepared, target_class)
            layer_options: dict[str, int] = {}
            if self.target_occurrence is not None:
                layer_options["occurrence"] = self.target_occurrence
            activations, gradients, layer_trace = _get_layer_gradients_with_trace(
                self.model,
                prepared,
                layer_name=self.target_layer,
                target_class=target,
                **layer_options,
            )
            cam = self._compute_gradcam(activations, gradients)
            score_space = getattr(self.model, "last_gradient_output_space", None) or "unknown"

        normalization_metadata = _cam_normalization_metadata(cam)
        heatmap = _normalize_cam(cam)
        if resize_to_input and heatmap.shape != input_size:
            heatmap = _resize_2d(heatmap, input_size)

        return Explanation(
            explainer_name="GradCAM",
            target_class=self._label_for_target(target),
            explanation_data={
                "heatmap": heatmap.tolist(),
                "heatmap_shape": list(heatmap.shape),
                "target_layer": self.target_layer,
                "target_index": target,
                "method": "gradcam",
                "input_shape": list(prepared.shape),
                "input_layout": resolved_layout,
                "configured_input_layout": self.input_layout,
                "channel_axis": (
                    None if resolved_layout == "flat" else channel_axis_for_layout(resolved_layout)
                ),
                "target_occurrence": self.target_occurrence,
                "target_layer_call_count": (
                    None if layer_trace is None else layer_trace.call_count
                ),
            },
            metadata={
                "formula_verified": True,
                "score_space": score_space,
                "declared_raw_model_output_space": getattr(
                    self.model, "raw_model_output_space", "unspecified"
                ),
                "paper_score_space_match": (
                    score_space == "model"
                    and getattr(self.model, "raw_model_output_space", None) == "logit"
                ),
                "postprocessing": "relu_minmax_bilinear_align_corners_false",
                "reference": "Selvaraju et al. (ICCV 2017), equations 1-2",
                **normalization_metadata,
            },
        )

    def explain_batch(
        self,
        images: np.ndarray,
        target_class: Optional[Union[int, np.integer, Sequence[int], np.ndarray]] = None,
    ) -> List[Explanation]:
        """Explain each input independently with scalar or per-row targets."""
        batch = np.asarray(images)
        if batch.ndim < 2 or batch.shape[0] == 0:
            raise ValueError("images must be a non-empty batch")
        targets = _targets_for_batch(target_class, int(batch.shape[0]))
        return [
            self.explain(batch[index], target_class=targets[index])
            for index in range(batch.shape[0])
        ]

    def get_overlay(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: str = "jet",
    ) -> np.ndarray:
        """Blend a finite 2D heatmap over a grayscale or RGB image."""
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.number)):
            raise TypeError("alpha must be a number in [0, 1]")
        alpha = float(alpha)
        if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if colormap != "jet":
            raise ValueError("only the 'jet' colormap is supported")

        image_array = as_real_array(
            image,
            name="image",
            dtype=np.float64,
            require_finite=True,
        )
        heatmap_array = as_real_array(
            heatmap,
            name="heatmap",
            dtype=np.float64,
            require_finite=True,
        )
        if image_array.size == 0 or heatmap_array.size == 0:
            raise ValueError("image and heatmap must not be empty")
        if heatmap_array.ndim != 2:
            raise ValueError("heatmap must have shape (height, width)")

        if image_array.ndim == 3 and image_array.shape[0] in (1, 3):
            image_array = np.transpose(image_array, (1, 2, 0))
        if image_array.ndim == 2:
            image_array = np.repeat(image_array[..., None], 3, axis=-1)
        elif image_array.ndim == 3 and image_array.shape[-1] == 1:
            image_array = np.repeat(image_array, 3, axis=-1)
        elif image_array.ndim != 3 or image_array.shape[-1] != 3:
            raise ValueError("image must be grayscale or RGB in CHW/HWC layout")

        if np.min(image_array) < 0:
            raise ValueError("overlay image values must be non-negative")
        if np.max(image_array) > 1.0:
            if np.max(image_array) > 255.0:
                raise ValueError("overlay image values must be in [0, 1] or [0, 255]")
            image_array = image_array / 255.0
        if np.min(heatmap_array) < 0.0 or np.max(heatmap_array) > 1.0:
            raise ValueError("heatmap values must be in [0, 1]")
        if heatmap_array.shape != image_array.shape[:2]:
            image_height, image_width = image_array.shape[:2]
            heatmap_array = _resize_2d(heatmap_array, (image_height, image_width))

        red = np.clip(1.5 - np.abs(4.0 * heatmap_array - 3.0), 0.0, 1.0)
        green = np.clip(1.5 - np.abs(4.0 * heatmap_array - 2.0), 0.0, 1.0)
        blue = np.clip(1.5 - np.abs(4.0 * heatmap_array - 1.0), 0.0, 1.0)
        colored = np.stack((red, green, blue), axis=-1)
        return np.clip((1.0 - alpha) * image_array + alpha * colored, 0.0, 1.0)
