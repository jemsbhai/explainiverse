"""Verified CAM formulas for spatial convolutional activations.

Paper-defined methods in this module are implemented directly from their
published equations: HiResCAM, XGrad-CAM, LayerCAM, Eigen-CAM, Score-CAM, and
Ablation-CAM.  EigenGradCAM and GradCAMElementWise are retained as explicitly
identified pytorch-grad-cam library variants; neither is attributed to the
Eigen-CAM or Grad-CAM papers.

All methods explain exactly one input at a time internally. ``explain_batch``
iterates independently so each row can resolve and retain its own target.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation
from explainiverse.explainers._validation import as_real_array, validate_name_sequence
from explainiverse.explainers.gradient._image_layout import channel_axis_for_layout
from explainiverse.explainers.gradient._input import (
    scale_safe_multi_product_sum,
    scale_safe_product,
    scale_safe_product_sum,
    scale_safe_sum,
)
from explainiverse.explainers.gradient.gradcam import (
    Target,
    _cam_normalization_metadata,
    _get_layer_gradients_with_trace,
    _get_layer_output_with_trace,
    _normalize_cam,
    _prepare_single_input,
    _preserve_adapter_model_state,
    _resize_2d,
    _resolve_target,
    _targets_for_batch,
    _validate_input_layout,
    _validate_spatial_activations,
    _validate_spatial_pair,
    _validate_target,
)


def _validate_batch_size(batch_size: int) -> int:
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)):
        raise TypeError("batch_size must be a positive integer")
    value = int(batch_size)
    if value <= 0:
        raise ValueError("batch_size must be a positive integer")
    return value


def _wrapped_torch_module(adapter):
    module = getattr(adapter, "model", None)
    if module is None or not callable(getattr(module, "named_modules", None)):
        raise TypeError(
            "This CAM method requires a PyTorchAdapter exposing its wrapped " "torch module"
        )
    return module


def _adapter_forward(adapter, inputs: np.ndarray, *, prediction_space: bool) -> np.ndarray:
    """Run one direct torch forward in raw or adapter-prediction score space."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - adapter cannot exist without torch
        raise ImportError("PyTorch is required for CAM forward passes") from exc

    module = _wrapped_torch_module(adapter)
    to_tensor = getattr(adapter, "_to_tensor", None)
    if not callable(to_tensor):
        raise TypeError("The model adapter does not expose PyTorch tensor conversion")
    with torch.no_grad():
        raw_output = module(to_tensor(inputs))
        if not isinstance(raw_output, torch.Tensor):
            raise TypeError("CAM requires the wrapped model to return one tensor")
        output = raw_output
        if prediction_space:
            prediction_output = getattr(adapter, "_prediction_output", None)
            if not callable(prediction_output):
                raise TypeError("The model adapter does not expose its prediction-space transform")
            output = prediction_output(raw_output)
        elif output.ndim == 1:
            output = output.unsqueeze(-1)

    to_numpy = getattr(adapter, "_to_numpy", None)
    if not callable(to_numpy):
        raise TypeError("The model adapter does not expose PyTorch-to-NumPy conversion")
    scores = as_real_array(
        to_numpy(output),
        name="model scores",
        require_finite=True,
    )
    if scores.ndim != 2 or scores.shape[0] != inputs.shape[0] or scores.shape[1] < 1:
        raise ValueError(
            "The wrapped model must return shape (batch, n_outputs); "
            f"got {scores.shape} for batch {inputs.shape[0]}"
        )
    return scores


def _orient_principal_projection(projection: np.ndarray) -> np.ndarray:
    """Choose a deterministic representative of the SVD's arbitrary sign."""
    projection = as_real_array(
        projection,
        name="principal projection",
        dtype=np.float64,
        require_finite=True,
    )
    if projection.size:
        pivot = int(np.argmax(np.abs(projection)))
        if projection[pivot] < 0:
            projection = -projection
    return projection


def _principal_projection(activations: np.ndarray, *, center: bool) -> np.ndarray:
    """Project spatial feature vectors onto their first right singular vector."""
    activations = _validate_spatial_activations(activations)
    channels, height, width = activations.shape[1:]
    matrix: np.ndarray = (
        activations[0].reshape(channels, height * width).T.astype(np.float64, copy=False)
    )
    # A positive global scale prevents SVD/centering reductions from
    # overflowing on large finite values. Restore it before returning so raw
    # projection metadata remains truthful.
    matrix_scale = float(np.max(np.abs(matrix)))
    if matrix_scale > 0:
        matrix = matrix / matrix_scale
    if center:
        matrix = matrix - np.mean(matrix, axis=0, keepdims=True)
    _, _, right_vectors = np.linalg.svd(matrix, full_matrices=False)
    projection = matrix @ right_vectors[0]
    if matrix_scale > 0:
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            amplitude_projection = projection * matrix_scale
        lost_nonzero = (amplitude_projection == 0.0) & (projection != 0.0)
        if not np.isfinite(amplitude_projection).all() or np.any(lost_nonzero):
            raise FloatingPointError("principal projection is not representable")
        projection = amplitude_projection
    projection = _orient_principal_projection(projection)
    return projection.reshape(height, width)


def _principal_projection_of_products(
    gradients: np.ndarray,
    activations: np.ndarray,
) -> np.ndarray:
    """Project centered exact activation-gradient products without materializing them.

    EigenGradCAM's library formula first forms the element-wise product and
    then mean-centers each channel before its principal projection.  A direct
    binary64 product can overflow or underflow even when the centered,
    globally scaled matrix supplied to the SVD is finite.  The exceptional
    path below evaluates the products and centering from the exact binary64
    operands, applies one shared positive scale for the SVD, and restores that
    scale to the raw projection. A genuinely unrepresentable restored cell
    fails explicitly instead of returning a direction-only surrogate.
    """

    activation_values, gradient_values = _validate_spatial_pair(activations, gradients)
    channels, height, width = activation_values.shape[1:]
    activation_matrix = activation_values[0].reshape(channels, height * width).T
    gradient_matrix = gradient_values[0].reshape(channels, height * width).T

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        direct_products = gradient_matrix * activation_matrix
    product_underflow = (
        (direct_products == 0.0) & (gradient_matrix != 0.0) & (activation_matrix != 0.0)
    )
    with localcontext() as context:
        context.prec = 3500 + len(str(height * width))
        exact_columns: list[list[Decimal]] = []
        maximum = Decimal(0)
        for channel in range(channels):
            products = [
                Decimal.from_float(float(gradient_matrix[row, channel]))
                * Decimal.from_float(float(activation_matrix[row, channel]))
                for row in range(height * width)
            ]
            mean = sum(products, start=Decimal(0)) / Decimal(height * width)
            centered = [value - mean for value in products]
            exact_columns.append(centered)
            maximum = max(maximum, *(abs(value) for value in centered))

        # Preserve the bit-identical ordinary path only when both direct
        # centering and the global pre-SVD scale retain every exact non-zero
        # centered contribution. A finite product alone is not sufficient:
        # centering subnormals can round asymmetrically, and an unrelated huge
        # constant channel can scale a decisive tiny channel to zero.
        if np.all(np.isfinite(direct_products)) and not np.any(product_underflow):
            exact_nonzero = np.asarray(
                [
                    [exact_columns[channel][row] != 0 for channel in range(channels)]
                    for row in range(height * width)
                ],
                dtype=bool,
            )
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                rounded_exact_centered = np.asarray(
                    [
                        [float(exact_columns[channel][row]) for channel in range(channels)]
                        for row in range(height * width)
                    ],
                    dtype=np.float64,
                )
                direct_centered = direct_products - np.mean(direct_products, axis=0, keepdims=True)
                raw_scale = float(np.max(np.abs(direct_products)))
                if raw_scale > 0.0:
                    scaled_products = direct_products / raw_scale
                    scaled_centered = scaled_products - np.mean(
                        scaled_products, axis=0, keepdims=True
                    )
                else:
                    scaled_centered = direct_centered
            centered_value_lost = (rounded_exact_centered == 0.0) & exact_nonzero
            pre_svd_scale_lost = (scaled_centered == 0.0) & exact_nonzero
            if (
                np.all(np.isfinite(rounded_exact_centered))
                and np.all(np.isfinite(direct_centered))
                and np.array_equal(direct_centered, rounded_exact_centered)
                and not np.any(centered_value_lost)
                and not np.any(pre_svd_scale_lost)
            ):
                return _principal_projection(
                    direct_products.T.reshape(1, channels, height, width),
                    center=True,
                )

        if maximum == 0:
            return np.zeros((height, width), dtype=np.float64)
        matrix = np.asarray(
            [
                [float(exact_columns[channel][row] / maximum) for channel in range(channels)]
                for row in range(height * width)
            ],
            dtype=np.float64,
        )
        scaled_value_lost = np.asarray(
            [
                [
                    matrix[row, channel] == 0.0 and exact_columns[channel][row] != 0
                    for channel in range(channels)
                ]
                for row in range(height * width)
            ],
            dtype=bool,
        )
        if np.any(scaled_value_lost):
            raise FloatingPointError(
                "EigenGradCAM exact centered values exceed the binary64 dynamic range "
                "of one globally scaled SVD input"
            )

    _, _, right_vectors = np.linalg.svd(matrix, full_matrices=False)
    scaled_projection = _orient_principal_projection(matrix @ right_vectors[0])
    with localcontext() as context:
        context.prec = 3500 + len(str(height * width))
        restored_values = []
        for value in scaled_projection:
            exact_value = Decimal.from_float(float(value)) * maximum
            restored = float(exact_value)
            if not np.isfinite(restored) or (restored == 0.0 and exact_value != 0):
                raise FloatingPointError(
                    "EigenGradCAM centered principal projection is not representable"
                )
            restored_values.append(restored)
    return np.asarray(restored_values, dtype=np.float64).reshape(height, width)


class BaseCAMExplainer(BaseExplainer):
    """Shared validation, target resolution, state isolation, and post-processing."""

    _explainer_name = "BaseCAM"
    _method_key = "basecam"
    _uses_gradients = True
    _class_agnostic = False
    _canonical_paper_method = False
    _formula_verified = False
    _relu_postprocessing = True
    _reference = "none"

    def __init__(
        self,
        model,
        target_layer: str,
        class_names: Optional[List[str]] = None,
        input_layout: str = "auto",
        target_occurrence: Optional[int] = None,
    ):
        super().__init__(model)
        required_method = "get_layer_gradients" if self._uses_gradients else "get_layer_output"
        if not callable(getattr(model, required_method, None)):
            raise TypeError(
                f"model must provide {required_method}(); use PyTorchAdapter for " "PyTorch CNNs"
            )
        if not isinstance(target_layer, str) or not target_layer.strip():
            raise ValueError("target_layer must be a non-empty layer name")
        self.target_layer = target_layer
        self.class_names = validate_name_sequence(
            class_names,
            name="class_names",
            allow_none=True,
        )
        self.input_layout = _validate_input_layout(input_layout)
        if target_occurrence is not None and (
            isinstance(target_occurrence, bool)
            or not isinstance(target_occurrence, (int, np.integer))
        ):
            raise TypeError("target_occurrence must be a non-negative integer or None")
        if target_occurrence is not None and int(target_occurrence) < 0:
            raise ValueError("target_occurrence must be a non-negative integer or None")
        self.target_occurrence = None if target_occurrence is None else int(target_occurrence)

    def _compute_cam(
        self,
        activations: np.ndarray,
        gradients: Optional[np.ndarray],
        image: np.ndarray,
        target_class: Optional[int],
    ) -> np.ndarray:
        raise NotImplementedError("BaseCAMExplainer does not define a CAM formula")

    def _method_metadata(self) -> Dict[str, object]:
        return {}

    def _label_for_target(self, target: Optional[int]) -> str:
        if target is None:
            return "class_agnostic"
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
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected CAM option(s): {unexpected}")
        if not isinstance(resize_to_input, (bool, np.bool_)):
            raise TypeError("resize_to_input must be a boolean")
        prepared, input_size, resolved_layout = _prepare_single_input(
            self.model, instance, input_layout=self.input_layout
        )
        layer_options = {}
        if self.target_occurrence is not None:
            layer_options["occurrence"] = self.target_occurrence

        with _preserve_adapter_model_state(self.model, preserve_gradients=self._uses_gradients):
            if self._class_agnostic:
                if _validate_target(target_class) is not None:
                    raise ValueError(
                        f"{self._explainer_name} is class-agnostic; target_class " "must be None"
                    )
                target = None
                activations, layer_trace = _get_layer_output_with_trace(
                    self.model,
                    prepared,
                    layer_name=self.target_layer,
                    **layer_options,
                )
                gradients = None
                activations = _validate_spatial_activations(activations)
                score_space = "not_applicable"
            else:
                target = _resolve_target(self.model, prepared, target_class)
                if self._uses_gradients:
                    activations, gradients, layer_trace = _get_layer_gradients_with_trace(
                        self.model,
                        prepared,
                        layer_name=self.target_layer,
                        target_class=target,
                        **layer_options,
                    )
                    activations, gradients = _validate_spatial_pair(activations, gradients)
                    score_space = (
                        getattr(self.model, "last_gradient_output_space", None) or "unknown"
                    )
                else:
                    activations, layer_trace = _get_layer_output_with_trace(
                        self.model,
                        prepared,
                        layer_name=self.target_layer,
                        **layer_options,
                    )
                    activations = _validate_spatial_activations(activations)
                    gradients = None
                    score_space = "method_specific"

            cam = as_real_array(
                self._compute_cam(activations, gradients, prepared, target),
                name=f"{self._explainer_name} CAM",
                dtype=np.float64,
                require_finite=True,
            )

        if cam.ndim != 2:
            raise ValueError(
                f"{self._explainer_name} produced shape {cam.shape}; expected a 2D CAM"
            )
        normalization_input = np.maximum(cam, 0.0) if self._relu_postprocessing else cam
        normalization_metadata = _cam_normalization_metadata(normalization_input)
        heatmap = _normalize_cam(normalization_input)
        if resize_to_input and heatmap.shape != input_size:
            heatmap = _resize_2d(heatmap, input_size)

        metadata: Dict[str, object] = {
            "formula_verified": self._formula_verified,
            "canonical_paper_method": self._canonical_paper_method,
            "reference": self._reference,
            "score_space": score_space,
            "postprocessing": (
                "relu_minmax_bilinear_align_corners_false"
                if self._relu_postprocessing
                else "minmax_bilinear_align_corners_false"
            ),
            **normalization_metadata,
        }
        if self._uses_gradients and not self._class_agnostic:
            declared_space = getattr(self.model, "raw_model_output_space", "unspecified")
            paper_score_space_match = bool(score_space == "model" and declared_space == "logit")
            metadata.update(
                declared_raw_model_output_space=declared_space,
                paper_score_space_match=paper_score_space_match,
            )
            if self._canonical_paper_method:
                metadata["canonical_paper_method"] = paper_score_space_match
        metadata.update(self._method_metadata())
        return Explanation(
            explainer_name=self._explainer_name,
            target_class=self._label_for_target(target),
            explanation_data={
                "heatmap": heatmap.tolist(),
                "heatmap_shape": list(heatmap.shape),
                "target_layer": self.target_layer,
                "target_index": target,
                "method": self._method_key,
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
            metadata=metadata,
        )

    def explain_batch(
        self,
        images: np.ndarray,
        target_class: Optional[Union[int, np.integer, Sequence[int], np.ndarray]] = None,
    ) -> List[Explanation]:
        batch = np.asarray(images)
        if batch.ndim < 2 or batch.shape[0] == 0:
            raise ValueError("images must be a non-empty batch")
        if self._class_agnostic and target_class is not None:
            raise ValueError(f"{self._explainer_name} is class-agnostic; target_class must be None")
        targets = _targets_for_batch(target_class, int(batch.shape[0]))
        return [
            self.explain(batch[index], target_class=targets[index])
            for index in range(batch.shape[0])
        ]


class HiResCAMExplainer(BaseCAMExplainer):
    """HiResCAM: channel-summed elementwise activation-gradient products.

    The formula is general.  The paper's *faithfulness guarantee* is narrower:
    it applies to the specified CNN architecture ending in one fully connected
    layer.  This explainer does not infer an arbitrary model graph and therefore
    does not assert that the guarantee holds for a supplied model.
    """

    _explainer_name = "HiResCAM"
    _method_key = "hirescam"
    _canonical_paper_method = True
    _formula_verified = True
    _reference = "Draelos & Carin (2021), HiResCAM equation"

    def _compute_cam(self, activations, gradients, image, target_class):
        del image, target_class
        return scale_safe_product_sum(gradients, activations, axis=1)[0]

    def _method_metadata(self):
        return {
            "faithfulness_guarantee_asserted": False,
            "faithfulness_guarantee_scope": (
                "paper-specified CNN ending in one fully connected layer"
            ),
        }


class XGradCAMExplainer(BaseCAMExplainer):
    """XGrad-CAM using activation-normalized gradient channel weights."""

    _explainer_name = "XGradCAM"
    _method_key = "xgradcam"
    _canonical_paper_method = True
    _formula_verified = True
    _reference = "Fu et al. (BMVC 2020), equations 7-8"

    def _compute_cam(self, activations, gradients, image, target_class):
        del image, target_class
        activation_scale = np.max(np.abs(activations), axis=(2, 3))
        safe_activation_scale = np.where(activation_scale == 0, 1, activation_scale)
        normalized_activations = activations / safe_activation_scale[:, :, None, None]
        stable_denominator = scale_safe_sum(activations, axis=(2, 3))
        normalized_denominator = stable_denominator / safe_activation_scale
        zero_denominator = stable_denominator == 0.0
        nonzero_maps = activation_scale > 0
        if np.any(zero_denominator & nonzero_maps):
            raise ValueError(
                "XGrad-CAM is undefined for a nonzero activation channel whose "
                "spatial activation sum is zero"
            )

        gradient_scale = np.max(np.abs(gradients), axis=(2, 3))
        safe_gradient_scale = np.where(gradient_scale == 0, 1, gradient_scale)
        normalized_gradients = gradients / safe_gradient_scale[:, :, None, None]
        normalized_numerator = scale_safe_product_sum(
            normalized_gradients, normalized_activations, axis=(2, 3)
        )
        safe_denominator = np.where(zero_denominator, 1.0, normalized_denominator)
        normalized_ratio = normalized_numerator / safe_denominator
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            direct_denominator = np.sum(activations, axis=(2, 3))
            direct_numerator = np.sum(gradients * activations, axis=(2, 3))
            direct_weights = direct_numerator / direct_denominator
        lost_denominator = (direct_denominator == 0.0) & (stable_denominator != 0.0)
        lost_numerator = (
            (direct_numerator == 0.0)
            & (normalized_numerator != 0.0)
            & (gradient_scale != 0.0)
            & (activation_scale != 0.0)
        )
        direct_safe = (
            ~zero_denominator
            & ~lost_denominator
            & ~lost_numerator
            & np.isfinite(direct_denominator)
            & np.isfinite(direct_numerator)
            & np.isfinite(direct_weights)
        )
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            direct_weighted_activations = direct_weights[:, :, None, None] * activations
            activation_over_direct_denominator = activations / direct_denominator[:, :, None, None]
            factored_direct_activations = (
                direct_numerator[:, :, None, None] * activation_over_direct_denominator
            )
        fallback_weighted_activations = scale_safe_product(
            normalized_ratio[:, :, None, None],
            gradient_scale[:, :, None, None],
            activation_scale[:, :, None, None],
            normalized_activations,
        )
        direct_map_safe = direct_safe[:, :, None, None] & np.isfinite(direct_weighted_activations)
        factored_direct_safe = (
            (~zero_denominator & ~lost_denominator & ~lost_numerator)[:, :, None, None]
            & np.isfinite(direct_denominator)[:, :, None, None]
            & np.isfinite(direct_numerator)[:, :, None, None]
            & np.isfinite(activation_over_direct_denominator)
            & np.isfinite(factored_direct_activations)
        )
        # Preserve ordinary-range reference rounding; fully factored normalized
        # coordinates are used where the channel weight cannot itself be
        # represented even though its weighted activation map can.
        weighted_activations = np.where(
            direct_map_safe,
            direct_weighted_activations,
            np.where(
                factored_direct_safe,
                factored_direct_activations,
                fallback_weighted_activations,
            ),
        )
        weighted_activations = np.where(
            zero_denominator[:, :, None, None], 0.0, weighted_activations
        )
        fallback_nonzero_factors = (
            (normalized_ratio[:, :, None, None] != 0.0)
            & (gradient_scale[:, :, None, None] != 0.0)
            & (activation_scale[:, :, None, None] != 0.0)
            & (normalized_activations != 0.0)
        )
        fallback_underflow = (fallback_weighted_activations == 0.0) & fallback_nonzero_factors
        unstable_channel = (lost_denominator | lost_numerator)[:, :, None, None]
        needs_fused_fallback = np.any(
            fallback_underflow | (unstable_channel & (normalized_activations != 0.0)),
            axis=1,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            direct_cam = np.sum(weighted_activations, axis=1)
        stable_cam = scale_safe_sum(weighted_activations, axis=1)
        cam = np.where(np.isfinite(direct_cam), direct_cam, stable_cam)
        if np.any(needs_fused_fallback):
            fused_cam = scale_safe_multi_product_sum(
                normalized_ratio[:, :, None, None],
                gradient_scale[:, :, None, None],
                activation_scale[:, :, None, None],
                normalized_activations,
                axis=1,
            )
            cam = np.where(needs_fused_fallback, fused_cam, cam)
        return cam[0]

    def _method_metadata(self):
        return {
            "axiom_guarantee_asserted": False,
            "paper_scope": "approximate sensitivity/conservation for deep target layers",
        }


class LayerCAMExplainer(BaseCAMExplainer):
    """LayerCAM for a compatible spatial layer: ReLU(gradient) times activation."""

    _explainer_name = "LayerCAM"
    _method_key = "layercam"
    _canonical_paper_method = True
    _formula_verified = True
    _reference = "Jiang et al. (IEEE TIP 2021), LayerCAM equation"

    def _compute_cam(self, activations, gradients, image, target_class):
        del image, target_class
        return scale_safe_product_sum(np.maximum(gradients, 0.0), activations, axis=1)[0]


class EigenCAMExplainer(BaseCAMExplainer):
    """Class-agnostic Eigen-CAM projection of spatial activation vectors."""

    _explainer_name = "EigenCAM"
    _method_key = "eigencam"
    _uses_gradients = False
    _class_agnostic = True
    _canonical_paper_method = True
    _formula_verified = True
    _relu_postprocessing = False
    _reference = "Muhammad & Yeasin (IJCNN 2020), equations 2-3"

    def _compute_cam(self, activations, gradients, image, target_class):
        del gradients, image, target_class
        # The paper factorizes the raw activation matrix O and projects O onto
        # V1; it does not specify mean-centering.
        return _principal_projection(activations, center=False)

    def _method_metadata(self):
        return {
            "class_agnostic": True,
            "svd_centered": False,
            "svd_sign_convention": "largest_absolute_projection_is_positive",
        }


class ScoreCAMExplainer(BaseCAMExplainer):
    """Score-CAM Algorithm-1 raw-output/channel-softmax variant.

    This is a direct transcription of Algorithm 1's channel-softmax formula.
    The paper's experimental discussion and the authors' released code instead
    use post-softmax class probabilities without this channel softmax; that is
    a distinct variant and is not claimed here.
    """

    _explainer_name = "ScoreCAM"
    _method_key = "scorecam"
    _uses_gradients = False
    _canonical_paper_method = False
    _formula_verified = True
    _reference = "Wang et al. (CVPRW 2020), Algorithm 1 transcription"

    def __init__(
        self,
        model,
        target_layer: str,
        class_names: Optional[List[str]] = None,
        batch_size: int = 16,
        input_layout: str = "auto",
        target_occurrence: Optional[int] = None,
    ):
        super().__init__(
            model,
            target_layer,
            class_names,
            input_layout=input_layout,
            target_occurrence=target_occurrence,
        )
        self.batch_size = _validate_batch_size(batch_size)
        _wrapped_torch_module(model)

    def _compute_cam(self, activations, gradients, image, target_class):
        del gradients
        if image.ndim != 4:
            raise ValueError("Score-CAM requires a spatial image input")
        if target_class is None:  # guarded by BaseCAMExplainer
            raise RuntimeError("Score-CAM target resolution failed")

        raw_original = _adapter_forward(self.model, image, prediction_space=False)
        prediction_original = _adapter_forward(self.model, image, prediction_space=True)
        if (
            getattr(self.model, "task", None) == "classification"
            and raw_original.shape[1] != prediction_original.shape[1]
        ):
            raise ValueError(
                "Score-CAM Algorithm 1's raw-output target is unavailable for a "
                "one-logit classifier whose adapter exposes two complementary "
                "prediction classes"
            )
        if target_class >= raw_original.shape[1]:
            raise ValueError(f"Score-CAM raw model output does not contain target {target_class}")

        activation_maps = activations[0]
        channels, _, _ = activation_maps.shape
        input_height, input_width = image.shape[-2:]
        # Score-CAM's normalized activation masks participate directly in a
        # model forward pass. Preserve the prepared image/model-aligned dtype:
        # narrowing float64 masks can collapse channel-score differences that
        # are small in input space but material after model scaling.
        masks = np.empty((channels, input_height, input_width), dtype=image.dtype)
        for channel in range(channels):
            upsampled = _resize_2d(activation_maps[channel], (input_height, input_width))
            masks[channel] = _normalize_cam(upsampled).astype(image.dtype, copy=False)

        scores = np.empty(channels, dtype=np.float64)
        original_image = image[0]
        for start in range(0, channels, self.batch_size):
            stop = min(start + self.batch_size, channels)
            masked_inputs = masks[start:stop, None, :, :] * original_image[None, :, :, :]
            raw_scores = _adapter_forward(self.model, masked_inputs, prediction_space=False)
            if target_class >= raw_scores.shape[1]:
                raise ValueError(
                    "Score-CAM Algorithm 1 requires a raw model output for the selected "
                    f"target {target_class}; the model exposes {raw_scores.shape[1]} "
                    "raw output(s). One-logit binary class 0/1 targets are not "
                    "silently reinterpreted."
                )
            scores[start:stop] = raw_scores[:, target_class]

        # Algorithm 1 subtracts the baseline logit before this softmax.  The
        # baseline is constant across channels and cancels exactly by softmax
        # shift invariance, so it need not be evaluated.
        shifted = scores - np.max(scores)
        exponentials = np.exp(shifted)
        weights = exponentials / np.sum(exponentials)
        return scale_safe_product_sum(weights[:, None, None], activation_maps, axis=0)

    def _method_metadata(self):
        declared_space = getattr(self.model, "raw_model_output_space", "unspecified")
        paper_score_space_match = declared_space == "logit"
        return {
            "score_space": "raw_model_output",
            "declared_raw_model_output_space": declared_space,
            "scorecam_variant": "paper_algorithm_1_raw_output_channel_softmax",
            "paper_algorithm_1_score_space_match": paper_score_space_match,
            "paper_score_space_match": paper_score_space_match,
            "official_probability_weighting_match": False,
            "canonical_paper_method": False,
            "baseline_raw_output_omitted_by_softmax_shift_invariance": True,
            "baseline_logit_omitted_by_softmax_shift_invariance": paper_score_space_match,
        }


class EigenGradCAMExplainer(BaseCAMExplainer):
    """pytorch-grad-cam EigenGradCAM library variant (not an Eigen-CAM paper method)."""

    _explainer_name = "EigenGradCAM (library variant)"
    _method_key = "eigengradcam_library_variant"
    _canonical_paper_method = False
    _formula_verified = True
    _reference = "jacobgil/pytorch-grad-cam eigen_grad_cam.py"

    def _compute_cam(self, activations, gradients, image, target_class):
        del image, target_class
        return _principal_projection_of_products(gradients, activations)

    def _method_metadata(self):
        return {
            "variant_origin": "pytorch-grad-cam library",
            "paper_attribution": None,
            "svd_centered": True,
            "svd_sign_convention": "largest_absolute_projection_is_positive",
            "claim_status": "quarantined",
            "promotion_requires_primary_formula": True,
        }


class GradCAMElementWiseExplainer(BaseCAMExplainer):
    """pytorch-grad-cam element-wise library variant, not paper Grad-CAM."""

    _explainer_name = "GradCAMElementWise (library variant)"
    _method_key = "gradcam_elementwise_library_variant"
    _canonical_paper_method = False
    _formula_verified = True
    _reference = "jacobgil/pytorch-grad-cam grad_cam_elementwise.py"

    def _compute_cam(self, activations, gradients, image, target_class):
        del image, target_class
        same_sign = ((gradients > 0.0) & (activations > 0.0)) | (
            (gradients < 0.0) & (activations < 0.0)
        )
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            direct_products = gradients * activations
            direct_cam = np.sum(np.maximum(direct_products, 0.0), axis=1)
        lost_positive_product = (
            same_sign & (gradients != 0.0) & (activations != 0.0) & (direct_products == 0.0)
        )
        direct_safe = np.isfinite(direct_cam) & ~np.any(
            ~np.isfinite(direct_products) | lost_positive_product,
            axis=1,
        )
        if np.all(direct_safe):
            return direct_cam[0]

        stable_cam = scale_safe_product_sum(
            np.where(same_sign, np.abs(gradients), 0.0),
            np.abs(activations),
            axis=1,
        )
        return np.where(direct_safe, direct_cam, stable_cam)[0]

    def _method_metadata(self):
        return {
            "variant_origin": "pytorch-grad-cam library",
            "paper_attribution": None,
            "claim_status": "quarantined",
            "promotion_requires_primary_formula": True,
        }


class AblationCAMExplainer(BaseCAMExplainer):
    """Ablation-CAM using actual target-layer channel zeroing.

    For each channel, a forward hook replaces that channel of the selected
    layer's output with zero, exactly implementing the intervention in Desai &
    Ramaswamy.  This is not input masking.
    """

    _explainer_name = "AblationCAM"
    _method_key = "ablationcam"
    _uses_gradients = False
    _canonical_paper_method = True
    _formula_verified = True
    _reference = "Desai & Ramaswamy (WACV 2020), equations 3-4"

    def __init__(
        self,
        model,
        target_layer: str,
        class_names: Optional[List[str]] = None,
        batch_size: int = 16,
        input_layout: str = "auto",
        target_occurrence: Optional[int] = None,
    ):
        super().__init__(
            model,
            target_layer,
            class_names,
            input_layout=input_layout,
            target_occurrence=target_occurrence,
        )
        self.batch_size = _validate_batch_size(batch_size)
        module = _wrapped_torch_module(model)
        if target_layer not in dict(module.named_modules()):
            raise ValueError(f"Layer {target_layer!r} was not found in the wrapped model")

    def _scores_with_ablated_channels(
        self, image: np.ndarray, channel_indices: np.ndarray
    ) -> np.ndarray:
        import torch

        wrapped = _wrapped_torch_module(self.model)
        target_module = dict(wrapped.named_modules())[self.target_layer]
        expected_batch = int(channel_indices.shape[0])
        hook_calls = 0

        def zero_selected_channels(module, inputs, output):
            del module, inputs
            nonlocal hook_calls
            call_index = hook_calls
            hook_calls += 1
            selected = (
                call_index == 0
                if self.target_occurrence is None
                else call_index == self.target_occurrence
            )
            if not selected:
                return None
            if not isinstance(output, torch.Tensor) or output.ndim != 4:
                raise TypeError("Ablation-CAM target layer must return one 4D torch tensor")
            if output.shape[0] != expected_batch:
                raise ValueError("Unexpected target-layer batch size during channel ablation")
            replacement = output.clone()
            rows = torch.arange(expected_batch, device=output.device)
            channels = torch.as_tensor(channel_indices, dtype=torch.long, device=output.device)
            replacement[rows, channels, :, :] = 0
            return replacement

        repeated = np.repeat(image, expected_batch, axis=0)
        handle = target_module.register_forward_hook(zero_selected_channels)
        try:
            scores = _adapter_forward(self.model, repeated, prediction_space=False)
        finally:
            handle.remove()
        trace_validator = getattr(self.model, "_validate_layer_occurrence_trace", None)
        if callable(trace_validator):
            trace_validator(self.target_layer, hook_calls, self.target_occurrence)
        elif self.target_occurrence is None and hook_calls != 1:
            raise ValueError(
                "Ablation-CAM requires an explicit target_occurrence when the target "
                f"module runs {hook_calls} times"
            )
        elif self.target_occurrence is not None and self.target_occurrence >= hook_calls:
            raise ValueError(
                f"target_occurrence {self.target_occurrence} is out of range for "
                f"{hook_calls} traced target-layer call(s)"
            )
        return scores

    def _compute_cam(self, activations, gradients, image, target_class):
        del gradients
        if image.ndim != 4:
            raise ValueError("Ablation-CAM requires a spatial image input")
        if target_class is None:  # guarded by BaseCAMExplainer
            raise RuntimeError("Ablation-CAM target resolution failed")

        original_scores = _adapter_forward(self.model, image, prediction_space=False)
        prediction_scores = _adapter_forward(self.model, image, prediction_space=True)
        if (
            getattr(self.model, "task", None) == "classification"
            and original_scores.shape[1] != prediction_scores.shape[1]
        ):
            raise ValueError(
                "Ablation-CAM's paper-defined raw target score is unavailable "
                "for a one-logit classifier whose adapter exposes two "
                "complementary prediction classes"
            )
        if target_class >= original_scores.shape[1]:
            raise ValueError(f"target {target_class} is unavailable in prediction space")
        original_score = float(original_scores[0, target_class])
        if original_score == 0.0:
            raise ValueError(
                "Ablation-CAM relative channel weights are undefined when the "
                "original target score is zero"
            )

        activation_maps = activations[0]
        channels = activation_maps.shape[0]
        ablated_scores = np.empty(channels, dtype=np.float64)
        for start in range(0, channels, self.batch_size):
            stop = min(start + self.batch_size, channels)
            indices = np.arange(start, stop, dtype=np.int64)
            scores = self._scores_with_ablated_channels(image, indices)
            ablated_scores[start:stop] = scores[:, target_class]

        weights = (original_score - ablated_scores) / original_score
        return scale_safe_product_sum(weights[:, None, None], activation_maps, axis=0)

    def _method_metadata(self):
        declared_space = getattr(self.model, "raw_model_output_space", "unspecified")
        paper_score_space_match = declared_space == "logit"
        return {
            "score_space": "raw_model_output",
            "declared_raw_model_output_space": declared_space,
            "intervention": "zero_target_layer_channel",
            "paper_score_space_match": paper_score_space_match,
            "canonical_paper_method": paper_score_space_match,
        }
