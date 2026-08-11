"""Adversarial regressions for the approved gradient remediation workstream."""

from __future__ import annotations

import importlib.util
import threading
from decimal import Decimal, localcontext

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from explainiverse.adapters import PyTorchAdapter
from explainiverse.explainers.gradient import (
    ConceptActivationVector,
    DeepLIFTExplainer,
    DeepLIFTShapExplainer,
    EigenCAMExplainer,
    EigenGradCAMExplainer,
    GradCAMExplainer,
    HiResCAMExplainer,
    IntegratedGradientsExplainer,
    LRPExplainer,
    SaliencyExplainer,
    ScoreCAMExplainer,
    SmoothGradExplainer,
    TCAVExplainer,
    XGradCAMExplainer,
)
from explainiverse.explainers.gradient._input import (
    scale_safe_mean,
    scale_safe_mean_std,
    scale_safe_product_sum,
    scale_safe_sum,
)
from explainiverse.explainers.gradient.cam_variants import BaseCAMExplainer
from explainiverse.explainers.gradient.deeplift import _stable_attribution_comparison
from explainiverse.explainers.gradient.gradcam import _normalize_cam


class _InplaceCAMModel(nn.Module):
    def __init__(self, *, dtype=torch.float64):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=1, bias=False, dtype=dtype)
        self.relu = nn.ReLU(inplace=True)
        with torch.no_grad():
            self.conv.weight.fill_(1.0)

    def forward(self, inputs):
        values = self.relu(self.conv(inputs))
        return values.flatten(1).sum(dim=1, keepdim=True)


def test_pre_inplace_layer_values_and_gradients_drive_public_gradcam_oracle():
    model = _InplaceCAMModel()
    adapter = PyTorchAdapter(model, task="regression")
    image = np.array([[[-1.0, 1.0], [2.0, -2.0]]], dtype=np.float64)

    activations, gradients = adapter.get_layer_gradients(
        image[np.newaxis, ...], "conv", target_class=0
    )
    np.testing.assert_array_equal(activations[0, 0], image[0])
    np.testing.assert_array_equal(gradients[0, 0], [[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_array_equal(
        adapter.get_layer_output(image[np.newaxis, ...], "conv")[0, 0], image[0]
    )

    explanation = GradCAMExplainer(adapter, "conv").explain(image, target_class=0)
    np.testing.assert_allclose(
        explanation.explanation_data["heatmap"], [[0.0, 0.5], [1.0, 0.0]], atol=1e-12
    )
    assert model.relu.inplace is True


class _StoreReturnedAlias(nn.Module):
    def forward(self, inputs):
        self.saved = inputs
        return inputs


class _AliasSensitiveLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.store = _StoreReturnedAlias()

    def forward(self, inputs):
        values = self.store(inputs)
        torch.relu_(values)
        return self.store.saved.sum(dim=1, keepdim=True)


def test_layer_taps_preserve_alias_sensitive_forward_semantics_and_caller_input():
    adapter = PyTorchAdapter(_AliasSensitiveLayerModel(), task="regression")
    caller_input = np.array([[-1.0, 2.0]], dtype=np.float64)
    original_input = caller_input.copy()

    np.testing.assert_array_equal(adapter.predict(caller_input), [[2.0]])
    np.testing.assert_array_equal(caller_input, original_input)
    np.testing.assert_array_equal(adapter.get_layer_output(caller_input, "store"), original_input)
    activations, gradients = adapter.get_layer_gradients(caller_input, "store", target_class=0)

    np.testing.assert_array_equal(activations, original_input)
    np.testing.assert_array_equal(gradients, [[0.0, 1.0]])
    np.testing.assert_array_equal(caller_input, original_input)
    np.testing.assert_array_equal(adapter.predict(caller_input), [[2.0]])


class _Float64NearTieScoreCAMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Identity()

    def forward(self, inputs):
        activations = self.features(inputs)
        score = 1e9 * activations[:, :, 0, 1].sum(dim=1)
        return torch.stack((score, torch.zeros_like(score)), dim=1)


def test_scorecam_preserves_float64_masks_for_near_tie_channel_scores():
    epsilon = 1e-9
    image = np.array(
        [
            [[0.0, 0.5 + epsilon], [1.0, 0.0]],
            [[0.0, 0.5], [0.0, 1.0]],
        ],
        dtype=np.float64,
    )
    adapter = PyTorchAdapter(
        _Float64NearTieScoreCAMModel().double(),
        task="classification",
        output_activation="auto",
    )

    actual = np.asarray(
        ScoreCAMExplainer(adapter, "features", input_layout="chw")
        .explain(image, target_class=0)
        .explanation_data["heatmap"]
    )
    expected = np.array([[0.0, 0.68393975], [1.0, 0.36787951]], dtype=np.float64)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-8)


def test_cam_minmax_preserves_tiny_structure_and_avoids_extreme_overflow():
    tiny = np.array([[0.0, 1e-300], [2e-300, 4e-300]], dtype=np.float64)
    np.testing.assert_array_equal(_normalize_cam(tiny), [[0.0, 0.25], [0.5, 1.0]])

    opposite_extremes = np.array([[-1e308, 0.0], [1e308, -5e307]], dtype=np.float64)
    normalized = _normalize_cam(opposite_extremes)
    assert np.isfinite(normalized).all()
    np.testing.assert_array_equal(normalized, [[0.0, 0.5], [1.0, 0.25]])


def test_public_gradcam_does_not_treat_tiny_nonconstant_cam_as_degenerate():
    adapter = PyTorchAdapter(_InplaceCAMModel(dtype=torch.float64), task="regression")
    image = np.array([[[0.0, 1e-300], [2e-300, 4e-300]]], dtype=np.float64)

    explanation = GradCAMExplainer(adapter, "conv").explain(image, target_class=0)

    np.testing.assert_array_equal(
        np.asarray(explanation.explanation_data["heatmap"]),
        [[0.0, 0.25], [0.5, 1.0]],
    )
    assert explanation.metadata["normalization_degenerate"] is False


class _ExtremeScaleCAMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.conv.weight.fill_(1.0)

    def forward(self, inputs):
        return self.conv(inputs).flatten(1).sum(dim=1, keepdim=True) * 1e308


@pytest.mark.parametrize("explainer_class", [GradCAMExplainer, XGradCAMExplainer])
def test_cam_reductions_are_scale_safe_for_tiny_activations_and_huge_gradients(
    explainer_class,
):
    adapter = PyTorchAdapter(_ExtremeScaleCAMModel(), task="regression")
    image = np.array([[[1e-308, 2e-308], [3e-308, 4e-308]]], dtype=np.float64)

    explanation = explainer_class(adapter, "conv").explain(image, target_class=0)

    np.testing.assert_allclose(
        np.asarray(explanation.explanation_data["heatmap"]),
        [[0.0, 1.0 / 3.0], [2.0 / 3.0, 1.0]],
        rtol=1e-15,
        atol=0.0,
    )


class _ExtremeCenteredEigenGradModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Identity()

    def forward(self, inputs):
        activations = self.features(inputs)
        return torch.sin(activations * 1e308).flatten(1).sum(dim=1, keepdim=True)


def test_eigengradcam_centering_is_scale_safe_for_constant_extreme_products():
    adapter = PyTorchAdapter(_ExtremeCenteredEigenGradModel().double(), task="regression")
    image = np.ones((1, 2, 2), dtype=np.float64)

    explanation = EigenGradCAMExplainer(adapter, "features").explain(image, target_class=0)

    np.testing.assert_array_equal(
        np.asarray(explanation.explanation_data["heatmap"]), np.zeros((2, 2))
    )
    assert explanation.metadata["normalization_degenerate"] is True


class _ExtremeCancellingChannelCAMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Identity()

    def forward(self, inputs):
        activations = self.features(inputs)
        grouped = (activations[:, 0] - activations[:, 2]) + (activations[:, 1] - activations[:, 3])
        return grouped.flatten(1).sum(dim=1, keepdim=True) * 1e308


@pytest.mark.parametrize(
    "explainer_class", [GradCAMExplainer, HiResCAMExplainer, XGradCAMExplainer]
)
def test_cam_channel_aggregation_preserves_representable_extreme_cancellation(
    explainer_class,
):
    adapter = PyTorchAdapter(_ExtremeCancellingChannelCAMModel().double(), task="regression")
    image = np.ones((4, 2, 2), dtype=np.float64)

    explanation = explainer_class(adapter, "features").explain(image, target_class=0)

    np.testing.assert_array_equal(
        np.asarray(explanation.explanation_data["heatmap"]), np.zeros((2, 2))
    )
    assert explanation.metadata["normalization_degenerate"] is True


def test_xgradcam_factored_map_survives_unrepresentable_intermediate_weight():
    scale = 1e-308
    activations = np.array([[[[scale, -scale * (1.0 - 1e-14)]]]], dtype=np.float64)
    gradients = np.array([[[[1e308, 0.0]]]], dtype=np.float64)

    actual = XGradCAMExplainer._compute_cam(None, activations, gradients, None, None)
    numerator = np.sum(gradients * activations)
    denominator = np.sum(activations)
    expected = numerator * (activations[0, 0] / denominator)

    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=0.0)


@pytest.mark.parametrize("method", ["gradcam", "hirescam"])
def test_cam_fused_product_sum_preserves_extreme_exact_cancellation(method):
    activations = np.array([[[[1e308]], [[-1e308]]]], dtype=np.float64)
    gradients = np.array([[[[1e308]], [[1e308]]]], dtype=np.float64)

    if method == "gradcam":
        actual = GradCAMExplainer._compute_gradcam(activations, gradients)
    else:
        actual = HiResCAMExplainer._compute_cam(None, activations, gradients, None, None)

    np.testing.assert_array_equal(actual, [[0.0]])


@pytest.mark.parametrize("method", ["gradcam", "hirescam"])
def test_cam_fused_product_sum_is_permutation_invariant_with_small_residual(method):
    activation_orders = [
        [1e308, 1.0, -1e308],
        [1e308, -1e308, 1.0],
        [1.0, -1e308, 1e308],
    ]
    results = []
    for order in activation_orders:
        activations = np.asarray(order, dtype=np.float64).reshape(1, 3, 1, 1)
        gradients = np.full_like(activations, 1e308)
        if method == "gradcam":
            result = GradCAMExplainer._compute_gradcam(activations, gradients)
        else:
            result = HiResCAMExplainer._compute_cam(None, activations, gradients, None, None)
        results.append(float(result[0, 0]))

    np.testing.assert_allclose(results, [1e308, 1e308, 1e308], rtol=1e-15)


@pytest.mark.parametrize("method", ["gradcam", "hirescam"])
def test_cam_fused_product_sum_preserves_cross_exponent_residual(method):
    activation_orders = [
        [1e308, 1e-308, -1e308],
        [1e308, -1e308, 1e-308],
        [1e-308, -1e308, 1e308],
    ]
    expected = 1e-308 * 1e308
    results = []
    for order in activation_orders:
        activations = np.asarray(order, dtype=np.float64).reshape(1, 3, 1, 1)
        gradients = np.full_like(activations, 1e308)
        if method == "gradcam":
            result = GradCAMExplainer._compute_gradcam(activations, gradients)
        else:
            result = HiResCAMExplainer._compute_cam(None, activations, gradients, None, None)
        results.append(float(result[0, 0]))

    np.testing.assert_allclose(results, [expected] * 3, rtol=1e-15, atol=0.0)


def test_shared_scale_safe_reducers_preserve_small_residual_across_permutations():
    orders = [
        np.array([1e308, 1.0, -1e308]),
        np.array([1e308, -1e308, 1.0]),
        np.array([1.0, -1e308, 1e308]),
    ]
    for values in orders:
        assert float(scale_safe_sum(values)) == pytest.approx(1.0, rel=1e-15)
        assert float(scale_safe_mean_std(values)[0]) == pytest.approx(1.0 / 3.0, rel=1e-15)

    matrix = np.column_stack((orders[0], 0.5 * orders[0]))
    np.testing.assert_allclose(scale_safe_mean_std(matrix)[0], [1.0 / 3.0, 1.0 / 6.0], rtol=1e-15)


def test_shared_scale_safe_reducers_preserve_subnormal_residual():
    orders = [
        np.array([1e308, 1e-308, -1e308]),
        np.array([1e308, -1e308, 1e-308]),
        np.array([1e-308, -1e308, 1e308]),
    ]
    expected_sum = np.float64(1e-308)
    expected_mean = expected_sum / 3.0
    for values in orders:
        np.testing.assert_allclose(scale_safe_sum(values), expected_sum, rtol=1e-15, atol=0.0)
        np.testing.assert_allclose(
            scale_safe_mean_std(values)[0], expected_mean, rtol=1e-15, atol=0.0
        )


def test_scale_safe_mean_uses_one_final_rounding_after_exact_division():
    values = np.asarray([0.0, 1.0, np.nextafter(1.0, np.inf)])
    with localcontext() as context:
        context.prec = 2000
        expected = float(
            sum(
                (Decimal.from_float(float(value)) for value in values),
                start=Decimal(0),
            )
            / Decimal(values.size)
        )

    assert float(scale_safe_mean(values)) == expected
    assert float(scale_safe_mean_std(values)[0]) == expected


@pytest.mark.parametrize(
    "values",
    [
        np.asarray([1.0, np.nextafter(1.0, 0.0)]),
        np.asarray([np.finfo(np.float64).max, np.nextafter(np.finfo(np.float64).max, 0.0)]),
    ],
)
def test_scale_safe_std_exactly_centers_adjacent_floats(values):
    expected = (values[0] - values[1]) / 2.0

    assert float(scale_safe_mean_std(values)[1]) == expected


def test_scale_safe_std_rejects_nonzero_unrepresentable_result():
    minimum_subnormal = np.nextafter(0.0, 1.0)
    values = np.zeros(9)
    values[:2] = [-minimum_subnormal, minimum_subnormal]

    with pytest.raises(FloatingPointError, match="standard deviation.*representable"):
        scale_safe_mean_std(values)


def test_product_sum_recovers_representable_sum_of_underflowed_products():
    minimum_subnormal = np.nextafter(0.0, 1.0)

    actual = scale_safe_product_sum(
        np.asarray([minimum_subnormal, minimum_subnormal]),
        np.asarray([0.5, 0.5]),
        axis=0,
    )

    assert actual == minimum_subnormal


class _ExactXGradModel(nn.Module):
    def __init__(self, gradient_scale=1.0):
        super().__init__()
        self.features = nn.Identity()
        self.gradient_scale = gradient_scale

    def forward(self, inputs):
        return self.features(inputs).flatten(1).sum(dim=1, keepdim=True) * self.gradient_scale


def test_xgradcam_fuses_half_subnormal_channel_contributions():
    minimum_subnormal = np.nextafter(0.0, 1.0)
    activations = np.asarray(
        [[[[minimum_subnormal, 0.0]], [[minimum_subnormal, 0.0]]]],
        dtype=np.float64,
    )
    gradients = np.full_like(activations, 0.5)

    with np.errstate(all="raise"):
        raw = XGradCAMExplainer._compute_cam(None, activations, gradients, None, None)
    np.testing.assert_array_equal(raw, [[minimum_subnormal, 0.0]])

    adapter = PyTorchAdapter(_ExactXGradModel(0.5).double(), task="regression")
    with np.errstate(all="raise"):
        heatmap = (
            XGradCAMExplainer(adapter, "features", input_layout="chw")
            .explain(activations[0], target_class=0)
            .explanation_data["heatmap"]
        )
    np.testing.assert_array_equal(heatmap, [[1.0, 0.0]])


def test_xgradcam_preserves_exact_nonzero_cancelling_denominator_without_warning():
    activations = np.asarray([[[[1.0, 1e-308, -1.0]]]], dtype=np.float64)
    gradients = np.ones_like(activations)

    with np.errstate(all="raise"):
        raw = XGradCAMExplainer._compute_cam(None, activations, gradients, None, None)
    np.testing.assert_array_equal(raw, activations[0, 0])

    adapter = PyTorchAdapter(_ExactXGradModel().double(), task="regression")
    with np.errstate(all="raise"):
        heatmap = (
            XGradCAMExplainer(adapter, "features", input_layout="chw")
            .explain(activations[0], target_class=0)
            .explanation_data["heatmap"]
        )
    np.testing.assert_array_equal(heatmap, [[1.0, 1e-308, 0.0]])


@pytest.mark.parametrize("method", ["gradcam", "hirescam"])
def test_cam_recovers_representable_subnormal_channel_sum(method):
    minimum_subnormal = np.nextafter(0.0, 1.0)
    activations = np.asarray(
        [[[[minimum_subnormal, 0.0]], [[minimum_subnormal, 0.0]]]], dtype=np.float64
    )
    gradients = np.full_like(activations, 0.5)

    if method == "gradcam":
        actual = GradCAMExplainer._compute_gradcam(activations, gradients)
    else:
        actual = HiResCAMExplainer._compute_cam(None, activations, gradients, None, None)

    np.testing.assert_array_equal(actual, [[minimum_subnormal, 0.0]])


def test_gradcam_fuses_unrepresentable_channel_means_before_rounding():
    minimum_subnormal = np.nextafter(0.0, 1.0)
    activations = np.asarray([[[[1.0, 0.0]], [[1.0, 0.0]]]], dtype=np.float64)
    gradients = np.asarray(
        [[[[minimum_subnormal, 0.0]], [[minimum_subnormal, 0.0]]]], dtype=np.float64
    )

    actual = GradCAMExplainer._compute_gradcam(activations, gradients)

    np.testing.assert_array_equal(actual, [[minimum_subnormal, 0.0]])


class _SharedLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(2, 2)

    def forward(self, inputs):
        return self.shared(self.shared(inputs))


@pytest.mark.parametrize("method", ["get_layer_output", "get_layer_gradients"])
def test_repeated_target_layer_execution_is_rejected_instead_of_taking_last(method):
    adapter = PyTorchAdapter(_SharedLayerModel(), task="regression")
    operation = getattr(adapter, method)
    kwargs = {"target_class": 0} if method == "get_layer_gradients" else {}
    with pytest.raises(RuntimeError, match="executed more than once|occurrence"):
        operation(np.ones((1, 2), dtype=np.float32), "shared", **kwargs)


class _RetainLayerOutputs(nn.Module):
    def __init__(self):
        super().__init__()
        self.history = []

    def forward(self, inputs):
        self.history.append(inputs)
        return inputs


class _RetainedOutputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.store = _RetainLayerOutputs()
        self.fail = False

    def forward(self, inputs):
        values = self.store(inputs)
        if self.fail:
            raise RuntimeError("injected post-layer failure")
        return values.sum(dim=1, keepdim=True)


def test_layer_tensor_gradient_hooks_are_removed_from_retained_outputs():
    model = _RetainedOutputModel()
    adapter = PyTorchAdapter(model, task="regression")
    inputs = np.ones((1, 2), dtype=np.float64)

    for _ in range(3):
        adapter.get_layer_gradients(inputs, "store", target_class=0)
    model.fail = True
    with pytest.raises(RuntimeError, match="post-layer failure"):
        adapter.get_layer_gradients(inputs, "store", target_class=0)

    assert len(model.store.history) == 4
    assert all(len(tensor._backward_hooks or {}) == 0 for tensor in model.store.history)


class _BFloatLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, inputs):
        return self.identity(inputs).sum(dim=1, keepdim=True)


def test_bfloat16_is_bridged_at_every_adapter_numpy_boundary():
    adapter = PyTorchAdapter(_BFloatLayerModel(), task="regression", input_dtype="bfloat16")
    inputs = np.arange(6, dtype=np.float32).reshape(2, 3)

    prediction = adapter.predict(inputs)
    scores, input_gradients = adapter.predict_with_gradients(inputs, target_class=0)
    layer_output = adapter.get_layer_output(inputs, "identity")
    layer_values, layer_gradients = adapter.get_layer_gradients(inputs, "identity", target_class=0)

    for value in (prediction, scores, input_gradients, layer_output, layer_values, layer_gradients):
        assert value.dtype == np.float32
        assert np.isfinite(value).all()
    np.testing.assert_array_equal(input_gradients, np.ones_like(inputs))
    np.testing.assert_array_equal(layer_gradients, np.ones_like(inputs))


def test_pytorch_prediction_output_kind_matches_activation_contract():
    classifier = nn.Linear(2, 2)
    assert PyTorchAdapter(classifier, task="classification").prediction_output_kind == (
        "probabilities"
    )
    assert (
        PyTorchAdapter(
            nn.Linear(2, 2), task="classification", output_activation="none"
        ).prediction_output_kind
        is None
    )
    assert (
        PyTorchAdapter(nn.Linear(2, 1), task="regression").prediction_output_kind
        == "regression_values"
    )


def test_rejected_meta_move_does_not_poison_device_state():
    model = nn.Linear(2, 1)
    adapter = PyTorchAdapter(model, task="regression")
    original_device = adapter.device

    with pytest.raises(ValueError, match="do not support the meta device"):
        adapter.to("meta")

    assert adapter.device == original_device
    assert next(model.parameters()).device == original_device
    assert adapter.predict(np.ones((1, 2), dtype=np.float32)).shape == (1, 1)


class _FailingAfterMove(nn.Linear):
    failing_moves = 0

    def to(self, *args, **kwargs):
        if self.failing_moves:
            self.failing_moves -= 1
            raise RuntimeError("injected move/rollback failure")
        return super().to(*args, **kwargs)


def test_partial_device_move_with_impossible_rollback_reports_inconsistent_state():
    model = _FailingAfterMove(2, 1)
    adapter = PyTorchAdapter(model, task="regression")
    model.failing_moves = 2

    with pytest.raises(RuntimeError, match="rollback failed.*may be inconsistent"):
        adapter.to("cpu")

    assert adapter.device == torch.device("cpu")
    assert next(model.parameters()).device == torch.device("cpu")
    with pytest.raises(RuntimeError, match="is poisoned.*Reconstruct"):
        adapter.predict(np.ones((1, 2), dtype=np.float32))


class _UnusedLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.used = nn.Linear(2, 1)
        self.unused = nn.Identity()

    def forward(self, inputs):
        return self.used(inputs)


class _Pair(nn.Module):
    def forward(self, inputs):
        return inputs, inputs


class _TupleLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pair = _Pair()

    def forward(self, inputs):
        first, _ = self.pair(inputs)
        return first.sum(dim=1, keepdim=True)


class _DisconnectedLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.disconnected = nn.Linear(2, 2)

    def forward(self, inputs):
        self.disconnected(inputs)
        return inputs.sum(dim=1, keepdim=True)


@pytest.mark.parametrize("method", ["get_layer_output", "get_layer_gradients"])
def test_layer_api_errors_are_actionable_for_unused_and_nontensor_layers(method):
    unused_adapter = PyTorchAdapter(_UnusedLayerModel(), task="regression")
    tuple_adapter = PyTorchAdapter(_TupleLayerModel(), task="regression")
    kwargs = {"target_class": 0} if method == "get_layer_gradients" else {}

    with pytest.raises(RuntimeError, match="did not run"):
        getattr(unused_adapter, method)(np.ones((1, 2)), "unused", **kwargs)
    with pytest.raises(TypeError, match=r"one.*[Tt]ensor"):
        getattr(tuple_adapter, method)(np.ones((1, 2)), "pair", **kwargs)


def test_layer_gradient_error_identifies_executed_but_disconnected_layer():
    adapter = PyTorchAdapter(_DisconnectedLayerModel(), task="regression")
    with pytest.raises(RuntimeError, match="does not depend on layer.*undefined"):
        adapter.get_layer_gradients(
            np.ones((1, 2), dtype=np.float32), "disconnected", target_class=0
        )


class _PrecisionModel(nn.Module):
    def forward(self, inputs):
        return 1e8 * inputs


def test_float64_gradient_explainers_do_not_collapse_small_input_differences():
    adapter = PyTorchAdapter(_PrecisionModel().double(), task="regression")
    instance = np.array([1.00000001], dtype=np.float64)

    integrated = IntegratedGradientsExplainer(
        adapter,
        feature_names=["x"],
        n_steps=8,
        baseline=np.array([1.0], dtype=np.float64),
    ).explain(instance, target_class=0, return_convergence_delta=True)
    saliency = SaliencyExplainer(adapter, ["x"], absolute_value=False).explain(
        instance, target_class=0
    )
    smooth = SmoothGradExplainer(
        adapter, ["x"], n_samples=2, noise_scale=0.0, random_state=7
    ).explain(instance, target_class=0)

    assert integrated.explanation_data["attributions_raw"][0] == pytest.approx(1.0, abs=1e-7)
    assert integrated.explanation_data["prediction_difference"] == pytest.approx(1.0)
    assert saliency.explanation_data["attributions_raw"][0] == pytest.approx(1e8)
    assert smooth.explanation_data["attributions_raw"][0] == pytest.approx(1e8)


class _HugeFiniteGradientModel(nn.Module):
    def forward(self, inputs):
        return inputs * 1e308


def test_smoothgrad_aggregation_does_not_overflow_a_representable_mean_and_std():
    adapter = PyTorchAdapter(_HugeFiniteGradientModel().double(), task="regression")
    explanation = SmoothGradExplainer(
        adapter,
        ["x"],
        n_samples=2,
        noise_scale=0.0,
    ).explain(np.array([1.0], dtype=np.float64), target_class=0)

    assert explanation.explanation_data["attributions_raw"] == [1e308]
    assert explanation.explanation_data["attributions_std"] == [0.0]


class _TinyGradientExtremeEndpointModel(nn.Module):
    def forward(self, inputs):
        return inputs * 1e-308


@pytest.mark.parametrize(
    "method",
    ["riemann_left", "riemann_right", "riemann_middle", "riemann_trapezoid"],
)
def test_integrated_gradients_handles_unrepresentable_endpoint_span(method):
    adapter = PyTorchAdapter(_TinyGradientExtremeEndpointModel().double(), task="regression")
    explanation = IntegratedGradientsExplainer(
        adapter,
        feature_names=["x"],
        n_steps=2,
        method=method,
        baseline=np.array([-1e308], dtype=np.float64),
    ).explain(
        np.array([1e308], dtype=np.float64),
        target_class=0,
        return_convergence_delta=True,
    )

    assert explanation.explanation_data["attributions_raw"] == pytest.approx([2.0])
    assert explanation.explanation_data["prediction_difference"] == pytest.approx(2.0)
    assert explanation.explanation_data["convergence_delta"] == pytest.approx(0.0)


class _FirstFeatureModel(nn.Module):
    def forward(self, inputs):
        return inputs[:, :1]


def test_integrated_gradients_mean_baseline_is_scale_safe_for_huge_equal_features():
    adapter = PyTorchAdapter(_FirstFeatureModel().double(), task="regression")
    explanation = IntegratedGradientsExplainer(
        adapter,
        feature_names=["a", "b"],
        baseline="mean",
        n_steps=2,
    ).explain(np.array([1e308, 1e308], dtype=np.float64), target_class=0)

    np.testing.assert_array_equal(explanation.explanation_data["baseline"], [1e308, 1e308])
    np.testing.assert_array_equal(explanation.explanation_data["attributions_raw"], [0.0, 0.0])


@pytest.mark.parametrize(("dtype", "magnitude"), [(np.float64, 1e308), (np.float32, 3e38)])
def test_integrated_gradients_random_baseline_handles_extreme_opposite_endpoints(dtype, magnitude):
    explainer = IntegratedGradientsExplainer(
        _NeverCalledGradientAdapter(), baseline="random", n_steps=1, random_state=123
    )
    instance = np.array([-magnitude, magnitude], dtype=dtype)

    first = explainer._get_baseline(instance, explainer._new_rng())
    second = explainer._get_baseline(instance, explainer._new_rng())

    assert first.dtype == instance.dtype
    assert np.isfinite(first).all()
    assert np.all(first >= instance.min()) and np.all(first <= instance.max())
    np.testing.assert_array_equal(first, second)


class _SequenceGradientAdapter:
    task = "regression"

    def __init__(self, gradients):
        self.gradients = list(gradients)
        self.index = 0

    def predict_with_gradients(self, inputs, target_class=None):
        del target_class
        gradient = np.full_like(inputs, self.gradients[self.index], dtype=np.float64)
        self.index += 1
        return np.zeros((len(inputs), 1), dtype=np.float64), gradient


@pytest.mark.parametrize(
    ("method", "n_steps", "expected"),
    [
        ("riemann_middle", 3, np.float64(1e-308) / 3.0),
        ("riemann_trapezoid", 2, np.float64(1e-308) / 2.0),
    ],
)
def test_integrated_gradients_quadrature_preserves_cancelling_residual(method, n_steps, expected):
    adapter = _SequenceGradientAdapter([1.0, 1e-308, -1.0])
    explanation = IntegratedGradientsExplainer(
        adapter,
        feature_names=["x"],
        n_steps=n_steps,
        method=method,
        baseline=np.asarray([0.0]),
    ).explain(np.asarray([1.0]), target_class=0)

    np.testing.assert_allclose(
        explanation.explanation_data["attributions_raw"],
        [expected],
        rtol=1e-15,
        atol=0.0,
    )


def test_integrated_gradients_fuses_unrepresentable_average_with_endpoint_delta():
    minimum_subnormal = np.nextafter(0.0, 1.0)
    adapter = _SequenceGradientAdapter([0.0, minimum_subnormal, 0.0])
    explanation = IntegratedGradientsExplainer(
        adapter,
        feature_names=["x"],
        n_steps=2,
        method="riemann_trapezoid",
        baseline=np.asarray([0.0]),
    ).explain(np.asarray([2.0]), target_class=0)

    np.testing.assert_array_equal(
        explanation.explanation_data["attributions_raw"], [minimum_subnormal]
    )


@pytest.mark.parametrize("method", ["riemann_middle", "riemann_trapezoid"])
def test_integrated_gradients_aggregation_does_not_overflow_finite_integral(method):
    adapter = PyTorchAdapter(_HugeFiniteGradientModel().double(), task="regression")
    explanation = IntegratedGradientsExplainer(
        adapter,
        feature_names=["x"],
        n_steps=2,
        method=method,
        baseline=np.array([0.0], dtype=np.float64),
    ).explain(np.array([1.0], dtype=np.float64), target_class=0)

    assert explanation.explanation_data["attributions_raw"] == [1e308]


@pytest.mark.parametrize("kind", ["saliency", "smoothgrad", "ig"])
def test_regression_task_wins_over_display_class_names(kind):
    model = nn.Linear(1, 2, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0], [2.0]], dtype=torch.float64))
    adapter = PyTorchAdapter(model, task="regression")
    if kind == "saliency":
        explainer = SaliencyExplainer(adapter, ["x"], class_names=["a", "b"])
    elif kind == "smoothgrad":
        explainer = SmoothGradExplainer(
            adapter,
            ["x"],
            class_names=["a", "b"],
            n_samples=1,
            noise_scale=0,
        )
    else:
        explainer = IntegratedGradientsExplainer(adapter, ["x"], class_names=["a", "b"], n_steps=1)

    with pytest.raises(ValueError, match="multi-output regression|explicit target_class"):
        explainer.explain(np.array([1.0], dtype=np.float64))


class _StochasticEvalModel(nn.Module):
    def forward(self, inputs):
        return inputs * torch.rand_like(inputs)


def test_gradient_explanation_restores_torch_rng_and_reuses_same_realization():
    adapter = PyTorchAdapter(_StochasticEvalModel(), task="regression")
    explainer = SaliencyExplainer(adapter, ["x"], absolute_value=False)
    torch.manual_seed(20260810)
    before = torch.random.get_rng_state().clone()

    first = explainer.explain(np.array([2.0], dtype=np.float32), target_class=0)
    after_first = torch.random.get_rng_state().clone()
    second = explainer.explain(np.array([2.0], dtype=np.float32), target_class=0)

    assert torch.equal(before, after_first)
    assert torch.equal(before, torch.random.get_rng_state())
    assert first.explanation_data["attributions_raw"] == second.explanation_data["attributions_raw"]


class _RandomThenFail(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        torch.rand(1)
        self.relu(inputs * 1.0)
        raise RuntimeError("injected forward failure")


def test_rng_and_inplace_flags_restore_when_gradient_model_raises():
    model = _RandomThenFail()
    adapter = PyTorchAdapter(model, task="regression")
    explainer = SaliencyExplainer(adapter, ["x"])
    torch.manual_seed(19)
    before = torch.random.get_rng_state().clone()

    with pytest.raises(RuntimeError, match="injected forward failure"):
        explainer.explain(np.array([-1.0], dtype=np.float32), target_class=0)

    assert torch.equal(before, torch.random.get_rng_state())
    assert model.relu.inplace is True


class _CoordinatedConcurrentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_entered = threading.Event()
        self.second_entered = threading.Event()
        self.release_first = threading.Event()
        self.first_done = threading.Event()
        self.observed_training = []

    def forward(self, inputs):
        if threading.current_thread().name == "gradient-first":
            self.observed_training.append(self.training)
            self.first_entered.set()
            if not self.release_first.wait(timeout=2.0):
                raise RuntimeError("concurrency test did not release first forward")
        else:
            self.second_entered.set()
            self.release_first.set()
            if not self.first_done.wait(timeout=2.0):
                raise RuntimeError("concurrency test did not finish first explanation")
            self.observed_training.append(self.training)
        return inputs.sum(dim=1, keepdim=True)


def test_same_model_gradient_contexts_serialize_state_snapshots_and_restores():
    model = _CoordinatedConcurrentModel()
    adapter = PyTorchAdapter(model, task="regression")
    model.train()
    explainer = SaliencyExplainer(adapter, ["x"], absolute_value=False)
    errors = []

    def run_first():
        try:
            explainer.explain(np.array([1.0]), target_class=0)
        except Exception as error:  # pragma: no cover - asserted below
            errors.append(error)
        finally:
            model.first_done.set()

    def run_second():
        try:
            explainer.explain(np.array([1.0]), target_class=0)
        except Exception as error:  # pragma: no cover - asserted below
            errors.append(error)

    first = threading.Thread(target=run_first, name="gradient-first")
    second = threading.Thread(target=run_second, name="gradient-second")
    first.start()
    assert model.first_entered.wait(timeout=2.0)
    second.start()
    # With serialization the second forward cannot enter yet; release the
    # first explicitly. Without it, the second forward releases the first and
    # then observes the prematurely restored training mode.
    if not model.second_entered.wait(timeout=0.1):
        model.release_first.set()
    first.join(timeout=2.0)
    second.join(timeout=2.0)

    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert model.observed_training == [False, False]
    assert model.training is True


class _BlockingTargetLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_entered = threading.Event()
        self.second_entered = threading.Event()
        self.release_first = threading.Event()

    def forward(self, inputs):
        if threading.current_thread().name == "layer-first":
            self.first_entered.set()
            if not self.release_first.wait(timeout=2.0):
                raise RuntimeError("layer concurrency test did not release first call")
        else:
            self.second_entered.set()
        return inputs


class _ConcurrentLayerAPIModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wait = _BlockingTargetLayer()

    def forward(self, inputs):
        return self.wait(inputs).sum(dim=1, keepdim=True)


def test_direct_layer_gradient_calls_serialize_forward_hooks_on_shared_model():
    model = _ConcurrentLayerAPIModel()
    adapter = PyTorchAdapter(model, task="regression")
    results = []
    errors = []

    def call_layer_api():
        try:
            results.append(
                adapter.get_layer_gradients(
                    np.ones((1, 2), dtype=np.float64), "wait", target_class=0
                )
            )
        except Exception as error:  # pragma: no cover - asserted below
            errors.append(error)

    first = threading.Thread(target=call_layer_api, name="layer-first")
    second = threading.Thread(target=call_layer_api, name="layer-second")
    first.start()
    assert model.wait.first_entered.wait(timeout=2.0)
    second.start()
    if not model.wait.second_entered.wait(timeout=0.1):
        model.wait.release_first.set()
    first.join(timeout=2.0)
    second.join(timeout=2.0)

    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert len(results) == 2
    for activations, gradients in results:
        np.testing.assert_array_equal(activations, [[1.0, 1.0]])
        np.testing.assert_array_equal(gradients, [[1.0, 1.0]])
    assert len(model.wait._forward_hooks) == 0


class _FailingEvalTransition(nn.Linear):
    fail_eval = False

    def __init__(self):
        super().__init__(1, 1, bias=False, dtype=torch.float64)
        self.register_buffer("cache", torch.tensor([7.0], dtype=torch.float64))

    def train(self, mode=True):
        result = super().train(mode)
        if not mode and self.fail_eval:
            self.cache.resize_(2).fill_(3.0)
            torch.rand(3)
            raise RuntimeError("injected eval transition failure")
        return result


def test_custom_eval_transition_is_rejected_before_rng_or_model_state_mutation():
    model = _FailingEvalTransition()
    with torch.no_grad():
        model.weight.fill_(1.0)
    adapter = PyTorchAdapter(model, task="regression")
    model.train(True)
    original_buffer = model.cache
    original_gradient = torch.tensor([[5.0]], dtype=torch.float64)
    model.weight.grad = original_gradient
    model.fail_eval = True
    torch.manual_seed(9876)
    rng_before = torch.random.get_rng_state().clone()

    with pytest.raises(RuntimeError, match="train overrides canonical"):
        SaliencyExplainer(adapter, ["x"]).explain(np.array([1.0], dtype=np.float64), target_class=0)

    assert model.training is True
    assert model.cache is original_buffer
    np.testing.assert_array_equal(model.cache.detach().numpy(), [7.0])
    assert tuple(model.cache.shape) == (1,)
    assert model.weight.grad is original_gradient
    np.testing.assert_array_equal(model.weight.grad.detach().numpy(), [[5.0]])
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    with pytest.raises(RuntimeError, match="is poisoned.*Reconstruct"):
        adapter.predict(np.ones((1, 1), dtype=np.float64))


class _ResizingBufferGradientModel(nn.Module):
    def __init__(self, *, fail=False):
        super().__init__()
        self.fail = fail
        self.register_buffer("cache", torch.tensor([7.0], dtype=torch.float64))

    def forward(self, inputs):
        self.cache.resize_(2).fill_(3.0)
        if self.fail:
            raise RuntimeError("injected buffer mutation failure")
        return inputs.sum(dim=1, keepdim=True)


@pytest.mark.parametrize("fail", [False, True])
def test_gradient_context_restores_resized_buffer_shape_and_identity(fail):
    model = _ResizingBufferGradientModel(fail=fail)
    adapter = PyTorchAdapter(model, task="regression")
    original_buffer = model.cache
    explainer = SaliencyExplainer(adapter, ["x"])

    if fail:
        with pytest.raises(RuntimeError, match="buffer mutation failure"):
            explainer.explain(np.array([1.0]), target_class=0)
    else:
        explainer.explain(np.array([1.0]), target_class=0)

    assert model.cache is original_buffer
    assert tuple(model.cache.shape) == (1,)
    assert tuple(model.cache.stride()) == (1,)
    np.testing.assert_array_equal(model.cache.detach().numpy(), [7.0])


class _AliasSensitiveInplaceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        hidden = inputs * 1.0
        alias = hidden
        self.relu(hidden)
        return (alias + hidden).sum(dim=1, keepdim=True)


def test_state_isolation_does_not_rewrite_alias_sensitive_inplace_semantics():
    model = _AliasSensitiveInplaceModel()
    adapter = PyTorchAdapter(model, task="regression")
    instance = np.array([-1.0, 2.0], dtype=np.float64)

    direct_scores, direct_gradients = adapter.predict_with_gradients(
        instance[np.newaxis, :], target_class=0
    )
    explanation = SaliencyExplainer(
        adapter, ["negative", "positive"], absolute_value=False
    ).explain(instance, target_class=0)

    np.testing.assert_allclose(direct_scores, [[4.0]])
    np.testing.assert_allclose(direct_gradients, [[0.0, 2.0]])
    np.testing.assert_allclose(explanation.explanation_data["attributions_raw"], [0.0, 2.0])
    assert model.relu.inplace is True


class _EigenAdapter:
    task = "classification"

    def __init__(self):
        self.model = nn.Identity()

    def get_layer_output(self, inputs, layer_name):
        del inputs, layer_name
        return np.array([[[[-1.0, 0.0], [4.0, 1.0]]]])


def test_eigencam_preserves_signed_principal_projection_before_minmax():
    explanation = EigenCAMExplainer(_EigenAdapter(), "layer").explain(
        np.zeros((1, 2, 2), dtype=np.float64)
    )
    np.testing.assert_allclose(
        explanation.explanation_data["heatmap"], [[0.0, 0.2], [1.0, 0.4]], atol=1e-12
    )
    assert explanation.metadata["postprocessing"] == "minmax_bilinear_align_corners_false"


class _BaseCAMAdapter:
    task = "classification"
    raw_model_output_space = "logit"
    last_gradient_output_space = "model"

    def __init__(self):
        self.model = nn.Identity()

    def predict(self, inputs):
        return np.tile([[0.1, 0.9]], (len(inputs), 1))

    def get_layer_gradients(self, inputs, layer_name, target_class):
        del inputs, layer_name, target_class
        values = np.ones((1, 1, 2, 2))
        return values, values


class _ThirdPartyCAM(BaseCAMExplainer):
    def _compute_cam(self, activations, gradients, image, target_class):
        del image, target_class
        return np.sum(activations * gradients, axis=1)[0]


def test_basecam_subclasses_are_conservative_until_their_formula_is_verified():
    explanation = _ThirdPartyCAM(_BaseCAMAdapter(), "layer").explain(
        np.zeros((1, 2, 2)), target_class=1
    )
    assert explanation.metadata["formula_verified"] is False
    assert explanation.metadata["canonical_paper_method"] is False


class _GrayModel(nn.Module):
    def __init__(self, *, fail_once=False):
        super().__init__()
        self.fail_once = fail_once

    def forward(self, inputs):
        assert inputs.ndim == 4 and inputs.shape[1] == 1
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("first call fails")
        return inputs.flatten(1).sum(dim=1, keepdim=True)


def test_ig_supports_2d_grayscale_and_commits_shape_only_after_success():
    adapter = PyTorchAdapter(_GrayModel(), task="regression")
    explainer = IntegratedGradientsExplainer(adapter, n_steps=4)
    image = np.arange(4.0, dtype=np.float64).reshape(2, 2)
    explanation = explainer.explain(image, target_class=0)
    np.testing.assert_allclose(explanation.explanation_data["attributions_raw"], image)
    assert explainer.input_shape == (2, 2)

    failing = IntegratedGradientsExplainer(
        PyTorchAdapter(_GrayModel(fail_once=True), task="regression"), n_steps=2
    )
    with pytest.raises(RuntimeError, match="first call fails"):
        failing.explain(np.zeros((2, 2)))
    assert failing.input_shape is None
    recovered = failing.explain(np.zeros((1, 2, 2)))
    assert recovered.explanation_data["input_shape"] == [1, 2, 2]
    assert failing.input_shape == (1, 2, 2)


class _NeverCalledGradientAdapter:
    task = "regression"

    def __init__(self):
        self.calls = 0

    def predict(self, inputs):
        self.calls += 1
        return np.zeros((len(inputs), 1))

    def predict_with_gradients(self, inputs, target_class=None):
        del target_class
        self.calls += 1
        return np.zeros((len(inputs), 1)), np.zeros_like(inputs)


def test_saliency_and_smoothgrad_reject_options_before_model_work():
    adapter = _NeverCalledGradientAdapter()
    with pytest.raises(TypeError, match="absolute_value"):
        SaliencyExplainer(adapter, ["x"], absolute_value="false")

    saliency = SaliencyExplainer(adapter, ["x"])
    with pytest.raises(ValueError, match="Unknown method"):
        saliency.explain(np.array([1.0]), method="unknown")
    smooth = SmoothGradExplainer(adapter, ["x"], n_samples=1)
    with pytest.raises(TypeError, match="absolute_value"):
        smooth.explain(np.array([1.0]), absolute_value="false")
    assert adapter.calls == 0


@pytest.mark.parametrize(
    "factory",
    [
        lambda adapter, names: SaliencyExplainer(adapter, ["x"], class_names=names),
        lambda adapter, names: SmoothGradExplainer(adapter, ["x"], class_names=names),
        lambda adapter, names: IntegratedGradientsExplainer(adapter, ["x"], class_names=names),
    ],
)
def test_gradient_class_name_sequences_do_not_use_array_truthiness(factory):
    adapter = _NeverCalledGradientAdapter()
    explainer = factory(adapter, np.array(["a", "b"]))
    assert explainer.class_names == ["a", "b"]
    with pytest.raises(ValueError, match="non-empty"):
        factory(adapter, [])


@pytest.mark.parametrize("percentile", [True, -1.0, np.nan, np.inf])
def test_smoothgrad_adaptive_percentile_is_strict(percentile):
    explainer = SmoothGradExplainer(_NeverCalledGradientAdapter(), ["x"], n_samples=1)
    with pytest.raises((TypeError, ValueError), match="percentile"):
        explainer.adaptive_noise_scale(np.array([1.0]), percentile=percentile)


def test_smoothgrad_adaptive_percentage_above_100_remains_a_valid_scale_factor():
    explainer = SmoothGradExplainer(_NeverCalledGradientAdapter(), ["low", "high"], n_samples=1)
    assert explainer.adaptive_noise_scale(np.array([2.0, 6.0]), percentile=150.0) == 6.0


def test_smoothgrad_adaptive_scale_avoids_overflow_for_extreme_finite_range():
    explainer = SmoothGradExplainer(_NeverCalledGradientAdapter(), ["low", "high"], n_samples=1)
    actual = explainer.adaptive_noise_scale(
        np.array([-1e308, 1e308], dtype=np.float64), percentile=15.0
    )
    assert actual == pytest.approx(3e307)


class _TCAVAdapter:
    task = "classification"
    raw_model_output_space = "logit"
    gradient_output = "model"

    def list_layers(self):
        return ["layer"]

    def get_layer_output(self, inputs, layer_name):
        del layer_name
        return np.asarray(inputs)

    def get_layer_gradients(self, inputs, layer_name, target_class=None):
        del layer_name, target_class
        values = np.asarray(inputs)
        return values, np.ones_like(values)

    def predict(self, inputs):
        return np.tile([[0.4, 0.6]], (len(inputs), 1))


def test_tcav_seed_boolean_and_empty_concept_contracts_are_fail_fast():
    maximum = int(np.iinfo(np.uint32).max)
    with pytest.raises(ValueError, match=r"\[0,"):
        TCAVExplainer(_TCAVAdapter(), "layer", random_seed=-1)
    with pytest.raises(ValueError, match=r"\[0,"):
        TCAVExplainer(_TCAVAdapter(), "layer", random_seed=maximum + 1)

    explainer = TCAVExplainer(_TCAVAdapter(), "layer", random_seed=maximum)
    assert explainer._derived_seed(1) == 0
    with pytest.raises(TypeError, match="return_derivatives"):
        explainer.compute_tcav_score(np.ones((2, 1)), 0, "missing", return_derivatives="false")
    with pytest.raises(TypeError, match="run_significance_test"):
        explainer.explain(np.ones((2, 1)), run_significance_test="false")
    with pytest.raises(ValueError, match="non-empty"):
        explainer.explain(np.ones((2, 1)), concept_names=[])


@pytest.mark.skipif(importlib.util.find_spec("captum") is None, reason="Captum is optional")
def test_double_precision_captum_explainers_and_deepshap_api_separation():
    deep_model = nn.Sequential(
        nn.Linear(2, 2, bias=False, dtype=torch.float64),
        nn.ReLU(),
        nn.Linear(2, 1, bias=False, dtype=torch.float64),
    )
    with torch.no_grad():
        deep_model[0].weight.copy_(torch.eye(2, dtype=torch.float64))
        deep_model[2].weight.fill_(1.0)
    adapter = PyTorchAdapter(deep_model, task="regression")
    instance = np.array([1.0, 2.0], dtype=np.float64)

    deep = DeepLIFTExplainer(adapter, ["a", "b"])
    deep_values = deep.explain(instance, target_class=0).explanation_data["attributions_raw"]
    np.testing.assert_allclose(deep_values, instance, atol=1e-10)

    shap = DeepLIFTShapExplainer(
        adapter,
        ["a", "b"],
        background_data=np.array([[0.0, 0.0], [0.5, 0.5]], dtype=np.float64),
    )
    shap_values = shap.explain(instance, target_class=0).explanation_data["attributions_raw"]
    assert np.isfinite(np.asarray(shap_values, dtype=float)).all()
    with pytest.raises(NotImplementedError, match="set_background"):
        shap.set_baseline(np.zeros(2))
    with pytest.raises(NotImplementedError, match="background"):
        shap.explain_with_multiple_baselines(instance, np.zeros((2, 2)))
    with pytest.raises(NotImplementedError, match="background expectation"):
        shap.compare_with_integrated_gradients(instance)

    lrp_model = nn.Sequential(nn.Linear(2, 1, bias=False, dtype=torch.float64))
    with torch.no_grad():
        lrp_model[0].weight.fill_(1.0)
    lrp = LRPExplainer(PyTorchAdapter(lrp_model, task="regression"), ["a", "b"])
    lrp_values = lrp.explain(instance, target_class=0).explanation_data["attributions_raw"]
    assert np.isfinite(np.asarray(lrp_values, dtype=float)).all()


def test_integrated_gradients_convergence_flag_is_strict_before_model_work():
    adapter = _NeverCalledGradientAdapter()
    explainer = IntegratedGradientsExplainer(adapter, ["x"], n_steps=2)

    with pytest.raises(TypeError, match="return_convergence_delta"):
        explainer.explain(np.array([1.0]), return_convergence_delta="false")

    assert adapter.calls == 0


@pytest.mark.skipif(importlib.util.find_spec("captum") is None, reason="Captum is optional")
def test_captum_gradient_boolean_flags_are_strict_and_deeplift_mean_is_scale_safe():
    model = nn.Sequential(nn.Linear(1, 1, bias=False, dtype=torch.float64))
    with torch.no_grad():
        model[0].weight.fill_(1.0)
    adapter = PyTorchAdapter(model, task="regression")

    with pytest.raises(TypeError, match="multiply_by_inputs"):
        DeepLIFTExplainer(adapter, ["x"], multiply_by_inputs="true")

    deep = DeepLIFTExplainer(adapter, ["x"])
    with pytest.raises(TypeError, match="return_convergence_delta"):
        deep.explain(np.array([1.0]), return_convergence_delta="false")
    deep.set_baseline(np.array([[1e308], [1e308]], dtype=np.float64), method="mean")
    np.testing.assert_array_equal(deep.baseline, [1e308])

    shap = DeepLIFTShapExplainer(
        adapter,
        ["x"],
        background_data=np.array([[0.0], [1.0]], dtype=np.float64),
    )
    with pytest.raises(TypeError, match="return_convergence_delta"):
        shap.explain(np.array([1.0]), return_convergence_delta="false")

    extreme_shap = DeepLIFTShapExplainer(
        adapter,
        ["x"],
        background_data=np.array([[0.0], [0.0]], dtype=np.float64),
    ).explain(np.array([1e308], dtype=np.float64), target_class=0)
    assert extreme_shap.explanation_data["attributions_raw"] == [1e308]
    assert extreme_shap.explanation_data["attributions_std"] == [0.0]

    lrp = LRPExplainer(adapter, ["x"])
    with pytest.raises(TypeError, match="return_convergence_delta"):
        lrp.explain(np.array([1.0]), return_convergence_delta="false")
    with pytest.raises(TypeError, match="return_layer_relevances"):
        lrp._compute_lrp(np.array([1.0]), return_layer_relevances="true")


@pytest.mark.parametrize(
    ("vector", "expected"),
    [
        (np.asarray([1e-300, 0.0]), np.asarray([1.0, 0.0])),
        (
            np.asarray([np.finfo(np.float64).max, np.finfo(np.float64).max]),
            np.asarray([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)]),
        ),
    ],
)
def test_tcav_cav_normalization_is_scale_safe(vector, expected):
    cav = ConceptActivationVector("concept", "layer", vector, object(), 1.0)

    np.testing.assert_allclose(cav.vector, expected, rtol=1e-15, atol=0.0)


class _ExtremeTCAVGradientAdapter:
    task = "classification"
    raw_model_output_space = "logit"
    gradient_output = "model"
    last_gradient_output_space = "model"

    def __init__(self, gradients):
        self.gradients = np.asarray(gradients, dtype=np.float64)

    def list_layers(self):
        return ["layer"]

    def get_layer_output(self, inputs, layer_name):
        del layer_name
        return np.zeros((len(inputs), self.gradients.shape[1]), dtype=np.float64)

    def get_layer_gradients(self, inputs, layer_name, target_class=None):
        del layer_name, target_class
        gradients = np.repeat(self.gradients, len(inputs), axis=0)
        return np.zeros_like(gradients), gradients


def test_tcav_directional_derivative_uses_exact_product_sum():
    maximum = np.finfo(np.float64).max
    adapter = _ExtremeTCAVGradientAdapter([[maximum, maximum, -maximum]])
    explainer = TCAVExplainer(adapter, "layer", class_names=["target"])
    cav = ConceptActivationVector("concept", "layer", np.ones(3), object(), 1.0)

    derivative = explainer.compute_directional_derivative(np.zeros((1, 1)), cav, target_class=0)
    expected = maximum * cav.vector[0]

    assert np.isfinite(derivative).all()
    np.testing.assert_array_equal(derivative, [expected])


def test_deeplift_ig_comparison_metrics_match_exact_extreme_oracle():
    left = np.asarray([1e154, 1.0, -1e154])
    right = np.asarray([-1.0, 2.0, -1.0])
    with localcontext() as context:
        context.prec = 3000
        left_decimal = [Decimal.from_float(float(value)) for value in left]
        right_decimal = [Decimal.from_float(float(value)) for value in right]
        count = Decimal(left.size)
        differences = [
            left_value - right_value for left_value, right_value in zip(left_decimal, right_decimal)
        ]
        expected_mse = float(
            sum((value * value for value in differences), start=Decimal(0)) / count
        )
        expected_max = float(max(abs(value) for value in differences))
        left_sum = sum(left_decimal, start=Decimal(0))
        right_sum = sum(right_decimal, start=Decimal(0))
        cross_sum = sum((a * b for a, b in zip(left_decimal, right_decimal)), start=Decimal(0))
        left_square_sum = sum((value * value for value in left_decimal), start=Decimal(0))
        right_square_sum = sum((value * value for value in right_decimal), start=Decimal(0))
        covariance = count * cross_sum - left_sum * right_sum
        left_variance = count * left_square_sum - left_sum * left_sum
        right_variance = count * right_square_sum - right_sum * right_sum
        expected_correlation = float(covariance / (left_variance * right_variance).sqrt())

    metrics = _stable_attribution_comparison(left, right)

    assert metrics["correlation_defined"] is True
    assert metrics["correlation"] == expected_correlation
    assert metrics["mse"] == expected_mse
    assert metrics["max_difference"] == expected_max


def test_deeplift_ig_comparison_handles_extreme_range_and_unrepresentable_mse():
    maximum = np.finfo(np.float64).max
    identical = np.asarray([maximum, -maximum])
    metrics = _stable_attribution_comparison(identical, identical)
    assert metrics == {
        "correlation": 1.0,
        "correlation_defined": True,
        "mse": 0.0,
        "max_difference": 0.0,
    }

    with pytest.raises(FloatingPointError, match="MSE.*representable"):
        _stable_attribution_comparison(np.asarray([np.nextafter(0.0, 1.0)]), np.asarray([0.0]))


@pytest.mark.skipif(importlib.util.find_spec("captum") is None, reason="Captum is optional")
def test_lrp_rejects_preexisting_mutating_hook_before_rng_or_buffer_change():
    layer = nn.Linear(1, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        layer.weight.fill_(1.0)
    layer.register_buffer("calls", torch.zeros((), dtype=torch.int64))
    original_buffer = layer.calls

    def mutating_hook(module, inputs):
        del inputs
        module.calls.add_(1)
        torch.rand(())

    layer.register_forward_pre_hook(mutating_hook)
    torch.manual_seed(123456)
    rng_before = torch.random.get_rng_state().clone()

    with pytest.raises(RuntimeError, match="pre-existing.*forward_pre_hooks"):
        LRPExplainer(PyTorchAdapter(nn.Sequential(layer), task="regression"), ["x"])

    assert layer.calls is original_buffer
    assert int(layer.calls.item()) == 0
    assert torch.equal(torch.random.get_rng_state(), rng_before)
