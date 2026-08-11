"""Real-kernel CUDA release gate.

This suite lives outside the ordinary CPU ``testpaths`` intentionally.  The
CUDA workflows invoke it directly and insufficient hardware is a test failure,
never a skip.  Both the supported Torch floor and latest resolver edge run the
same contracts.
"""

from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np
import pytest
import torch
from torch import nn

from explainiverse.adapters import PyTorchAdapter
from explainiverse.evaluation.randomisation import _randomise_layer_parameters
from explainiverse.explainers.gradient import (
    AblationCAMExplainer,
    ConceptActivationVector,
    DeepLIFTExplainer,
    DeepLIFTShapExplainer,
    EigenCAMExplainer,
    EigenGradCAMExplainer,
    GradCAMElementWiseExplainer,
    GradCAMExplainer,
    HiResCAMExplainer,
    IntegratedGradientsExplainer,
    LayerCAMExplainer,
    LRPExplainer,
    SaliencyExplainer,
    ScoreCAMExplainer,
    SmoothGradExplainer,
    TCAVExplainer,
    XGradCAMExplainer,
)

REQUIRED_CUDA_DEVICES = int(os.environ.get("EXPLAINIVERSE_REQUIRED_CUDA_DEVICES", "1"))


@pytest.fixture(scope="session", autouse=True)
def require_real_cuda_hardware():
    assert REQUIRED_CUDA_DEVICES in {1, 2}, "release gate supports exactly one or two GPUs"
    assert torch.cuda.is_available(), "CUDA release gate requires real CUDA kernels"
    assert torch.cuda.device_count() == REQUIRED_CUDA_DEVICES, (
        f"CUDA release gate requires exactly {REQUIRED_CUDA_DEVICES} visible device(s); "
        f"got {torch.cuda.device_count()}"
    )
    for index in range(REQUIRED_CUDA_DEVICES):
        value = torch.arange(8, dtype=torch.float32, device=f"cuda:{index}")
        assert torch.equal(value.square().cpu(), torch.arange(8, dtype=torch.float32).square())
        assert torch.cuda.get_device_properties(index).major > 0
    torch.cuda.synchronize()


def _vector_classifier(*, dtype=torch.float32):
    """Return an exact supported Sequential graph for the Captum CUDA edges."""
    model = nn.Sequential(
        OrderedDict(
            [
                ("bottleneck", nn.Linear(4, 4, dtype=dtype)),
                ("relu", nn.ReLU()),
                ("output", nn.Linear(4, 2, dtype=dtype)),
            ]
        )
    )
    with torch.no_grad():
        model.bottleneck.weight.copy_(torch.eye(4, dtype=dtype))
        model.bottleneck.bias.fill_(0.25)
        model.output.weight.copy_(
            torch.tensor(
                [[1.0, -0.5, 0.25, 0.75], [-0.25, 0.5, 1.0, -0.5]],
                dtype=dtype,
            )
        )
        model.output.bias.zero_()
    return model


class _ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU()
        self.output = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.tensor([[[[1.0]]], [[[0.5]]]]))
            self.conv.bias.copy_(torch.tensor([0.2, 0.1]))
            self.output.weight.copy_(torch.tensor([[1.0, -0.5], [-0.25, 1.0]]))

    def forward(self, inputs):
        features = self.relu(self.conv(inputs))
        return self.output(features.mean(dim=(2, 3)))


@pytest.mark.parametrize(
    ("dtype", "numpy_dtype"),
    [(torch.float32, np.float32), (torch.float64, np.float64)],
    ids=["float32", "float64"],
)
def test_adapter_prediction_gradients_dtype_and_device_placement(dtype, numpy_dtype):
    model = _vector_classifier(dtype=dtype).to("cuda:0")
    adapter = PyTorchAdapter(model, task="classification", device="cuda:0")
    inputs = np.asarray([[0.5, 1.0, 1.5, 2.0]], dtype=numpy_dtype)

    prediction = adapter.predict(inputs)
    scores, gradients = adapter.predict_with_gradients(inputs, target_class=1)
    activations, layer_gradients = adapter.get_layer_gradients(inputs, "bottleneck", target_class=1)

    assert next(model.parameters()).device == torch.device("cuda:0")
    assert next(model.parameters()).dtype == dtype
    assert prediction.dtype == numpy_dtype
    assert scores.dtype == numpy_dtype
    assert gradients.dtype == numpy_dtype
    assert activations.dtype == numpy_dtype
    assert layer_gradients.dtype == numpy_dtype
    assert np.isfinite(prediction).all()
    assert np.isfinite(gradients).all()
    assert np.linalg.norm(gradients) > 0


def _assert_finite_explanation(explanation):
    def visit(value):
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        elif isinstance(value, np.ndarray):
            if np.issubdtype(value.dtype, np.number):
                assert np.isfinite(value).all()
        elif isinstance(value, (float, np.floating)):
            assert np.isfinite(value)

    visit(explanation.explanation_data)


def test_every_vector_gradient_family_executes_real_cuda_kernels():
    model = _vector_classifier().to("cuda:0")
    adapter = PyTorchAdapter(model, task="classification", device="cuda:0")
    instance = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    features = ["a", "b", "c", "d"]
    explainers = [
        SaliencyExplainer(adapter, features, absolute_value=False),
        IntegratedGradientsExplainer(
            adapter, features, n_steps=4, baseline=np.zeros(4, dtype=np.float32)
        ),
        SmoothGradExplainer(adapter, features, n_samples=3, noise_scale=0.01, random_state=11),
        DeepLIFTExplainer(adapter, features, baseline=np.zeros(4, dtype=np.float32)),
        DeepLIFTShapExplainer(
            adapter,
            features,
            background_data=np.array(
                [[0.0, 0.0, 0.0, 0.0], [0.1, 0.2, 0.1, 0.2], [0.2, 0.1, 0.2, 0.1]],
                dtype=np.float32,
            ),
        ),
        LRPExplainer(adapter, features, rule="epsilon"),
    ]
    for explainer in explainers:
        explanation = explainer.explain(instance, target_class=1)
        _assert_finite_explanation(explanation)
        assert next(model.parameters()).device == torch.device("cuda:0")

    tcav = TCAVExplainer(adapter, "bottleneck", require_logit_scores=True)
    cav = ConceptActivationVector(
        concept_name="first-axis",
        layer_name="bottleneck",
        vector=np.array([1.0, 0.0, 0.0, 0.0]),
        classifier=None,
        accuracy=1.0,
    )
    derivatives = tcav.compute_directional_derivative(instance[None, :], cav, target_class=1)
    assert derivatives.shape == (1,)
    assert np.isfinite(derivatives).all()


_CAM_RELEASE_CASES = [
    (GradCAMExplainer, 1),
    (HiResCAMExplainer, 1),
    (XGradCAMExplainer, 1),
    (LayerCAMExplainer, 1),
    (EigenCAMExplainer, None),
    (ScoreCAMExplainer, 1),
    (EigenGradCAMExplainer, 1),
    (GradCAMElementWiseExplainer, 1),
    (AblationCAMExplainer, 1),
]


@pytest.mark.parametrize(
    ("explainer_type", "target_class"),
    _CAM_RELEASE_CASES,
    ids=[explainer.__name__ for explainer, _ in _CAM_RELEASE_CASES],
)
def test_every_cam_family_executes_real_cuda_kernels_and_cleans_hooks(explainer_type, target_class):
    model = _ImageClassifier().to("cuda:0")
    adapter = PyTorchAdapter(model, task="classification", device="cuda:0")
    image = np.array([[[0.2, 0.5], [1.0, 0.7]]], dtype=np.float32)
    before = (len(model.conv._forward_hooks), len(model.conv._backward_hooks))

    explanation = explainer_type(adapter, "conv").explain(image, target_class=target_class)

    _assert_finite_explanation(explanation)
    assert (len(model.conv._forward_hooks), len(model.conv._backward_hooks)) == before
    assert next(model.parameters()).device == torch.device("cuda:0")


def test_layer_gradient_hook_cleanup_on_success_and_failure():
    model = _ImageClassifier().to("cuda:0")
    adapter = PyTorchAdapter(model, task="classification", device="cuda:0")
    image = np.ones((1, 1, 2, 2), dtype=np.float32)
    before = (len(model.conv._forward_hooks), len(model.conv._backward_hooks))

    adapter.get_layer_gradients(image, "conv", target_class=0)
    assert (len(model.conv._forward_hooks), len(model.conv._backward_hooks)) == before
    with pytest.raises(ValueError, match="target|class|index|output"):
        adapter.get_layer_gradients(image, "conv", target_class=99)
    assert (len(model.conv._forward_hooks), len(model.conv._backward_hooks)) == before


class _FailingReset(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))

    def reset_parameters(self):
        torch.rand(4, device=self.weight.device)
        raise RuntimeError("injected CUDA reset failure")

    def forward(self, inputs):
        return inputs @ self.weight


def _cuda_rng_states(count):
    return [torch.cuda.get_rng_state(index).clone() for index in range(count)]


def test_randomisation_success_and_failure_restore_initialized_cuda_rng_bytes():
    for index in range(REQUIRED_CUDA_DEVICES):
        torch.empty(1, device=f"cuda:{index}")
    torch.cuda.manual_seed_all(20260810)
    before = _cuda_rng_states(REQUIRED_CUDA_DEVICES)

    model = nn.Sequential(nn.Linear(4, 4)).to("cuda:0")
    old_weight = model[0].weight.detach().clone()
    _randomise_layer_parameters(model, "0", rng=np.random.default_rng(7))
    assert not torch.equal(old_weight, model[0].weight)
    after_success = _cuda_rng_states(REQUIRED_CUDA_DEVICES)
    assert all(torch.equal(left, right) for left, right in zip(before, after_success))

    failing = nn.Sequential(_FailingReset()).to("cuda:0")
    with pytest.raises(RuntimeError, match="injected CUDA reset failure"):
        _randomise_layer_parameters(failing, "0", rng=np.random.default_rng(8))
    after_failure = _cuda_rng_states(REQUIRED_CUDA_DEVICES)
    assert all(torch.equal(left, right) for left, right in zip(before, after_failure))


class _CrossDeviceRandomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, device="cuda:0"))

    def forward(self, inputs):
        first = torch.rand((), device="cuda:0")
        second = torch.rand((), device="cuda:1")
        return inputs[:, :1] * self.weight + first + second.to("cuda:0")


def test_visible_gpu_topology_and_cross_device_rng_restoration():
    assert torch.cuda.device_count() == REQUIRED_CUDA_DEVICES
    if REQUIRED_CUDA_DEVICES == 1:
        return
    model = _CrossDeviceRandomModel()
    adapter = PyTorchAdapter(model, task="regression", device="cuda:0")
    explainer = SaliencyExplainer(adapter, ["x"], absolute_value=False)
    torch.cuda.manual_seed_all(314159)
    before = _cuda_rng_states(2)

    explanation = explainer.explain(np.array([2.0], dtype=np.float32), target_class=0)
    _assert_finite_explanation(explanation)
    assert all(torch.equal(left, right) for left, right in zip(before, _cuda_rng_states(2)))

    original_forward = model.forward

    def failing_forward(inputs):
        original_forward(inputs)
        raise RuntimeError("injected cross-device failure")

    model.forward = failing_forward
    with pytest.raises(RuntimeError, match="cross-device failure"):
        explainer.explain(np.array([2.0], dtype=np.float32), target_class=0)
    assert all(torch.equal(left, right) for left, right in zip(before, _cuda_rng_states(2)))
