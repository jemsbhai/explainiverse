"""Analytical gradient contracts under caller-owned Torch grad modes."""

import numpy as np
import pytest

from explainiverse import PyTorchAdapter

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")


def _linear_adapter():
    model = nn.Sequential(nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        model[1].weight.copy_(torch.tensor([[2.0, -1.0], [1.0, 1.0]]))
    return PyTorchAdapter(model, output_activation="none", classification_output_kind="scores")


def _gradient_call(adapter, operation, values, result_format="numpy"):
    if operation == "input":
        return adapter.predict_with_gradients(values, target_class=0, result_format=result_format)
    if operation == "layer":
        return adapter.get_layer_gradients(values, "0", target_class=0, result_format=result_format)
    return adapter.get_layer_gradients_with_trace(
        values, "0", target_class=0, result_format=result_format
    )


def _assert_values(result, expected, result_format):
    if result_format == "tensor":
        assert isinstance(result, torch.Tensor)
        assert not result.requires_grad
        assert result.grad_fn is None
        torch.testing.assert_close(result, torch.tensor(expected), rtol=0, atol=0)
    else:
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("operation", ["input", "layer", "layer_trace"])
@pytest.mark.parametrize("input_kind", ["numpy", "tensor"])
@pytest.mark.parametrize("result_format", ["numpy", "tensor"])
@pytest.mark.parametrize("caller_grad_enabled", [False, True], ids=["no_grad", "enabled"])
def test_gradient_api_restores_caller_mode_and_preserves_owned_state(
    operation, input_kind, result_format, caller_grad_enabled
):
    adapter = _linear_adapter()
    values = np.array([[1.0, 2.0]], dtype=np.float32)
    if input_kind == "tensor":
        values = torch.tensor(values, requires_grad=True)
    saved_parameters = []
    for parameter in adapter.model.parameters():
        parameter.grad = torch.ones_like(parameter)
        saved_parameters.append((parameter, parameter.grad, parameter.grad.clone()))
    outer_grad_mode = torch.is_grad_enabled()

    with torch.set_grad_enabled(caller_grad_enabled):
        try:
            result = _gradient_call(adapter, operation, values, result_format)
        finally:
            assert torch.is_grad_enabled() == caller_grad_enabled
        expected_values = [[-1.0, 16.0]] if operation == "input" else [[5.0, 11.0]]
        expected_gradients = [[-1.0, 0.0]] if operation == "input" else [[2.0, -1.0]]
        _assert_values(result[0], expected_values, result_format)
        _assert_values(result[1], expected_gradients, result_format)
        if operation == "layer_trace":
            assert result[2].call_count == 1
            assert result[2].layer_name == "0"
    assert torch.is_grad_enabled() == outer_grad_mode
    if input_kind == "tensor":
        assert values.requires_grad
        assert values.grad is None
    for parameter, gradient, before in saved_parameters:
        assert parameter.grad is gradient
        torch.testing.assert_close(parameter.grad, before)
    assert not adapter.model[0]._forward_hooks


@pytest.mark.parametrize("operation", ["input", "layer", "layer_trace"])
@pytest.mark.parametrize("caller_grad_enabled", [False, True], ids=["no_grad", "enabled"])
def test_gradient_api_restores_caller_mode_and_hooks_after_forward_failure(
    operation, caller_grad_enabled
):
    adapter = _linear_adapter()

    def fail_after_forward(_module, _args, _output):
        raise RuntimeError("deliberate model failure")

    handle = adapter.model[1].register_forward_hook(fail_after_forward)
    outer_grad_mode = torch.is_grad_enabled()
    with torch.set_grad_enabled(caller_grad_enabled):
        try:
            with pytest.raises(RuntimeError, match="deliberate model failure"):
                _gradient_call(adapter, operation, np.array([[1.0, 2.0]], dtype=np.float32))
        finally:
            assert torch.is_grad_enabled() == caller_grad_enabled
            assert not adapter.model[0]._forward_hooks
            handle.remove()
    assert torch.is_grad_enabled() == outer_grad_mode


@pytest.mark.parametrize("operation", ["input", "layer", "layer_trace"])
@pytest.mark.parametrize("caller_grad_enabled", [False, True], ids=["no_grad", "enabled"])
def test_gradient_api_supports_inplace_input_model_without_mutating_input(
    operation, caller_grad_enabled
):
    model = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(2, 2, bias=False))
    with torch.no_grad():
        model[1].weight.copy_(torch.tensor([[2.0, -1.0], [1.0, 1.0]]))
    adapter = PyTorchAdapter(model, output_activation="none", classification_output_kind="scores")
    values = torch.tensor([[-1.0, 2.0]], requires_grad=True)
    with torch.set_grad_enabled(caller_grad_enabled):
        result = _gradient_call(adapter, operation, values)
        assert torch.is_grad_enabled() == caller_grad_enabled
    expected_values = [[-2.0, 2.0]] if operation == "input" else [[0.0, 2.0]]
    expected_gradients = [[0.0, -1.0]] if operation == "input" else [[2.0, -1.0]]
    np.testing.assert_array_equal(result[0], expected_values)
    np.testing.assert_array_equal(result[1], expected_gradients)
    torch.testing.assert_close(values, torch.tensor([[-1.0, 2.0]]))
    assert values.requires_grad
    assert values.grad is None
    assert not adapter.model[0]._forward_hooks


@pytest.mark.parametrize("operation", ["input", "layer", "layer_trace"])
@pytest.mark.parametrize("caller_grad_enabled", [False, True], ids=["no_grad", "enabled"])
def test_gradient_api_rejects_inference_mode_before_calling_model(operation, caller_grad_enabled):
    adapter = _linear_adapter()
    forward_calls = []
    handle = adapter.model.register_forward_pre_hook(
        lambda _module, _args: forward_calls.append(True)
    )
    try:
        with torch.inference_mode(), torch.set_grad_enabled(caller_grad_enabled):
            with pytest.raises(RuntimeError, match="inference_mode"):
                _gradient_call(adapter, operation, np.array([[1.0, 2.0]], dtype=np.float32))
            assert torch.is_inference_mode_enabled()
            assert torch.is_grad_enabled() == caller_grad_enabled
    finally:
        handle.remove()
    assert not forward_calls
    assert not adapter.model[0]._forward_hooks


@pytest.mark.parametrize("operation", ["input", "layer", "layer_trace"])
def test_gradient_api_differentiates_frozen_model_through_input_under_no_grad(operation):
    adapter = _linear_adapter()
    adapter.model.requires_grad_(False)
    with torch.no_grad():
        result = _gradient_call(adapter, operation, np.array([[1.0, 2.0]], dtype=np.float32))
        assert not torch.is_grad_enabled()
    expected_values = [[-1.0, 16.0]] if operation == "input" else [[5.0, 11.0]]
    expected_gradients = [[-1.0, 0.0]] if operation == "input" else [[2.0, -1.0]]
    np.testing.assert_array_equal(result[0], expected_values)
    np.testing.assert_array_equal(result[1], expected_gradients)
    assert all(not parameter.requires_grad for parameter in adapter.model.parameters())
    assert all(parameter.grad is None for parameter in adapter.model.parameters())


@pytest.mark.parametrize("operation", ["layer", "layer_trace"])
def test_layer_gradients_differentiate_embedding_parameters_under_no_grad(operation):
    model = nn.Sequential(nn.Embedding(3, 2), nn.Flatten(), nn.Linear(4, 2, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        model[2].weight.copy_(torch.tensor([[2.0, -1.0, 3.0, -2.0], [1.0, 1.0, 1.0, 1.0]]))
    adapter = PyTorchAdapter(model, output_activation="none", classification_output_kind="scores")
    values = torch.tensor([[0, 1]])
    with torch.no_grad():
        result = _gradient_call(adapter, operation, values)
        assert not torch.is_grad_enabled()
    np.testing.assert_array_equal(result[0], [[[1.0, 2.0], [3.0, 4.0]]])
    np.testing.assert_array_equal(result[1], [[[2.0, -1.0], [3.0, -2.0]]])
    torch.testing.assert_close(values, torch.tensor([[0, 1]]))
    assert all(parameter.grad is None for parameter in adapter.model.parameters())
    assert not adapter.model[0]._forward_hooks
