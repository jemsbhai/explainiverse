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

from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation
from explainiverse.explainers._validation import as_real_array, validate_name_sequence
from explainiverse.explainers.gradient.gradcam import (
    Target,
    _cam_normalization_metadata,
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

    scores = as_real_array(
        output.detach().cpu().numpy(),
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
    if center:
        matrix = matrix - np.mean(matrix, axis=0, keepdims=True)
    _, _, right_vectors = np.linalg.svd(matrix, full_matrices=False)
    projection = matrix @ right_vectors[0]
    projection = _orient_principal_projection(projection)
    return projection.reshape(height, width)


class BaseCAMExplainer(BaseExplainer):
    """Shared validation, target resolution, state isolation, and post-processing."""

    _explainer_name = "BaseCAM"
    _method_key = "basecam"
    _uses_gradients = True
    _class_agnostic = False
    _canonical_paper_method = False
    _reference = "none"

    def __init__(
        self,
        model,
        target_layer: str,
        class_names: Optional[List[str]] = None,
        input_layout: str = "auto",
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
        prepared, input_size = _prepare_single_input(
            self.model, instance, input_layout=self.input_layout
        )

        with _preserve_adapter_model_state(self.model, preserve_gradients=self._uses_gradients):
            if self._class_agnostic:
                if _validate_target(target_class) is not None:
                    raise ValueError(
                        f"{self._explainer_name} is class-agnostic; target_class " "must be None"
                    )
                target = None
                activations = self.model.get_layer_output(prepared, layer_name=self.target_layer)
                gradients = None
                activations = _validate_spatial_activations(activations)
                score_space = "not_applicable"
            else:
                target = _resolve_target(self.model, prepared, target_class)
                if self._uses_gradients:
                    activations, gradients = self.model.get_layer_gradients(
                        prepared,
                        layer_name=self.target_layer,
                        target_class=target,
                    )
                    activations, gradients = _validate_spatial_pair(activations, gradients)
                    score_space = (
                        getattr(self.model, "last_gradient_output_space", None) or "unknown"
                    )
                else:
                    activations = self.model.get_layer_output(
                        prepared, layer_name=self.target_layer
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
        normalization_input = np.maximum(cam, 0.0)
        normalization_metadata = _cam_normalization_metadata(normalization_input)
        heatmap = _normalize_cam(normalization_input)
        if resize_to_input and heatmap.shape != input_size:
            heatmap = _resize_2d(heatmap, input_size)

        metadata: Dict[str, object] = {
            "formula_verified": True,
            "canonical_paper_method": self._canonical_paper_method,
            "reference": self._reference,
            "score_space": score_space,
            "postprocessing": "relu_minmax_bilinear_align_corners_false",
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
                "input_layout": self.input_layout,
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
    _reference = "Draelos & Carin (2021), HiResCAM equation"

    def _compute_cam(self, activations, gradients, image, target_class):
        del image, target_class
        return np.sum(gradients * activations, axis=1)[0]

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
    _reference = "Fu et al. (BMVC 2020), equations 7-8"

    def _compute_cam(self, activations, gradients, image, target_class):
        del image, target_class
        denominator = np.sum(activations, axis=(2, 3))
        absolute_sum = np.sum(np.abs(activations), axis=(2, 3))
        tolerance = np.finfo(np.float64).eps * np.maximum(1.0, absolute_sum) * 16.0
        near_zero = np.abs(denominator) <= tolerance
        nonzero_maps = absolute_sum > tolerance
        if np.any(near_zero & nonzero_maps):
            raise ValueError(
                "XGrad-CAM is undefined for a nonzero activation channel whose "
                "spatial activation sum is zero"
            )

        safe_denominator = np.where(near_zero, 1.0, denominator)
        numerator = np.sum(gradients * activations, axis=(2, 3))
        weights = numerator / safe_denominator
        weights = np.where(near_zero, 0.0, weights)
        return np.sum(weights[:, :, None, None] * activations, axis=1)[0]

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
    _reference = "Jiang et al. (IEEE TIP 2021), LayerCAM equation"

    def _compute_cam(self, activations, gradients, image, target_class):
        del image, target_class
        return np.sum(np.maximum(gradients, 0.0) * activations, axis=1)[0]


class EigenCAMExplainer(BaseCAMExplainer):
    """Class-agnostic Eigen-CAM projection of spatial activation vectors."""

    _explainer_name = "EigenCAM"
    _method_key = "eigencam"
    _uses_gradients = False
    _class_agnostic = True
    _canonical_paper_method = True
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
    _reference = "Wang et al. (CVPRW 2020), Algorithm 1 transcription"

    def __init__(
        self,
        model,
        target_layer: str,
        class_names: Optional[List[str]] = None,
        batch_size: int = 16,
        input_layout: str = "auto",
    ):
        super().__init__(model, target_layer, class_names, input_layout=input_layout)
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
        masks = np.empty((channels, input_height, input_width), dtype=np.float32)
        for channel in range(channels):
            upsampled = _resize_2d(activation_maps[channel], (input_height, input_width))
            masks[channel] = _normalize_cam(upsampled).astype(np.float32)

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
        return np.sum(weights[:, None, None] * activation_maps, axis=0)

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
    _reference = "jacobgil/pytorch-grad-cam eigen_grad_cam.py"

    def _compute_cam(self, activations, gradients, image, target_class):
        del image, target_class
        return _principal_projection(gradients * activations, center=True)

    def _method_metadata(self):
        return {
            "variant_origin": "pytorch-grad-cam library",
            "paper_attribution": None,
            "svd_centered": True,
            "svd_sign_convention": "largest_absolute_projection_is_positive",
        }


class GradCAMElementWiseExplainer(BaseCAMExplainer):
    """pytorch-grad-cam element-wise library variant, not paper Grad-CAM."""

    _explainer_name = "GradCAMElementWise (library variant)"
    _method_key = "gradcam_elementwise_library_variant"
    _canonical_paper_method = False
    _reference = "jacobgil/pytorch-grad-cam grad_cam_elementwise.py"

    def _compute_cam(self, activations, gradients, image, target_class):
        del image, target_class
        return np.sum(np.maximum(gradients * activations, 0.0), axis=1)[0]

    def _method_metadata(self):
        return {"variant_origin": "pytorch-grad-cam library", "paper_attribution": None}


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
    _reference = "Desai & Ramaswamy (WACV 2020), equations 3-4"

    def __init__(
        self,
        model,
        target_layer: str,
        class_names: Optional[List[str]] = None,
        batch_size: int = 16,
        input_layout: str = "auto",
    ):
        super().__init__(model, target_layer, class_names, input_layout=input_layout)
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
            hook_calls += 1
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
        if hook_calls != 1:
            raise ValueError(
                "Ablation-CAM requires the target module to be invoked exactly "
                f"once per model forward; observed {hook_calls} calls"
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
        return np.sum(weights[:, None, None] * activation_maps, axis=0)

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
