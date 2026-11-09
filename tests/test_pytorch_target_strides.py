"""Analytical gradient contracts for reversed NumPy target arrays."""

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


def _adapter():
    model = nn.Sequential(nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        model[1].weight.copy_(torch.tensor([[2.0, -1.0], [1.0, 1.0]]))
    return PyTorchAdapter(model, output_activation="none", classification_output_kind="scores")


@pytest.mark.parametrize("operation", ["input", "layer", "layer_trace"])
@pytest.mark.parametrize("target_kind", ["per_row", "broadcast_singleton"])
@pytest.mark.parametrize("copy_target", [False, True], ids=["reversed_view", "owned_copy"])
def test_reversed_integer_targets_have_the_same_gradients_as_their_copy(
    operation, target_kind, copy_target
):
    adapter = _adapter()
    inputs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    source = np.array([0, 1] if target_kind == "per_row" else [1], dtype=np.int64)
    target = source[::-1]
    assert target.strides[0] < 0
    if target_kind == "broadcast_singleton":
        # A contiguous-array shortcut does not remove this negative stride.
        assert target.flags.c_contiguous
    if copy_target:
        target = target.copy()
    source_before = source.copy()
    target_before = target.copy()
    try:
        if operation == "input":
            result = adapter.predict_with_gradients(inputs, target_class=target)
        elif operation == "layer":
            result = adapter.get_layer_gradients(inputs, "0", target_class=target)
        else:
            result = adapter.get_layer_gradients_with_trace(inputs, "0", target_class=target)
    finally:
        np.testing.assert_array_equal(source, source_before)
        np.testing.assert_array_equal(target, target_before)
        assert not adapter.model[0]._forward_hooks

    expected_values = [[-1.0, 16.0], [-3.0, 36.0]]
    if operation == "input":
        expected_gradients = [[4.0, 6.0], [-1.0, 0.0]]
    else:
        expected_values = [[5.0, 11.0], [11.0, 25.0]]
        expected_gradients = [[1.0, 1.0], [2.0, -1.0]]
    if target_kind == "broadcast_singleton":
        expected_gradients[1] = expected_gradients[0]
    np.testing.assert_array_equal(result[0], expected_values)
    np.testing.assert_array_equal(result[1], expected_gradients)
    if operation == "layer_trace":
        assert result[2].layer_name == "0"
        assert result[2].call_count == 1


def _call(adapter, operation, inputs, target, result_format):
    if operation == "input":
        return adapter.predict_with_gradients(
            inputs, target_class=target, result_format=result_format
        )
    if operation == "layer":
        return adapter.get_layer_gradients(
            inputs, "0", target_class=target, result_format=result_format
        )
    return adapter.get_layer_gradients_with_trace(
        inputs, "0", target_class=target, result_format=result_format
    )


def _numpy_output(value, result_format):
    if result_format == "numpy":
        assert isinstance(value, np.ndarray)
        return value
    if result_format == "dlpack":
        value = torch.utils.dlpack.from_dlpack(value)
    assert isinstance(value, torch.Tensor)
    assert not value.requires_grad
    assert value.grad_fn is None
    return value.cpu().numpy()


@pytest.mark.parametrize("operation", ["input", "layer", "layer_trace"])
@pytest.mark.parametrize(
    "dtype,singleton,caller_grad_enabled,result_format",
    [
        pytest.param(np.uint8, False, False, "tensor", id="unsigned-no-grad-tensor"),
        pytest.param(np.int16, True, False, "numpy", id="int16-singleton-no-grad"),
        pytest.param(np.int32, False, True, "dlpack", id="int32-dlpack"),
        pytest.param(np.int64, True, True, "tensor", id="int64-singleton-tensor"),
    ],
)
def test_reversed_target_variants_preserve_caller_state_and_result_contracts(
    operation, dtype, singleton, caller_grad_enabled, result_format
):
    adapter = _adapter()
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    source = np.array([1] if singleton else [0, 1], dtype=dtype)
    target = source[::-1]
    source_before = source.copy()
    identity = (target.base, target.strides, target.dtype, target.__array_interface__["data"])
    caller_mode = torch.is_grad_enabled()

    try:
        with torch.set_grad_enabled(caller_grad_enabled):
            try:
                control = _call(adapter, operation, inputs, target.copy(), result_format)
                observed = _call(adapter, operation, inputs, target, result_format)
            finally:
                assert torch.is_grad_enabled() == caller_grad_enabled
    finally:
        assert torch.is_grad_enabled() == caller_mode
        assert target.base is identity[0]
        assert (target.strides, target.dtype, target.__array_interface__["data"]) == identity[1:]
        np.testing.assert_array_equal(source, source_before)
        torch.testing.assert_close(inputs, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        assert inputs.requires_grad
        assert inputs.grad is None
        assert not adapter.model[0]._forward_hooks

    expected_values = [[-1.0, 16.0], [-3.0, 36.0]]
    expected_gradients = [[4.0, 6.0], [-1.0, 0.0]]
    if operation != "input":
        expected_values = [[5.0, 11.0], [11.0, 25.0]]
        expected_gradients = [[1.0, 1.0], [2.0, -1.0]]
    if singleton:
        expected_gradients[1] = expected_gradients[0]
    for result in (control, observed):
        np.testing.assert_array_equal(_numpy_output(result[0], result_format), expected_values)
        np.testing.assert_array_equal(_numpy_output(result[1], result_format), expected_gradients)
    if operation == "layer_trace":
        assert observed[2] == control[2]
