"""Regression probes for retained callback output snapshots in invariance tests."""

import numpy as np
import pytest

from explainiverse.evaluation.axiomatic import (
    compute_batch_input_invariance,
    compute_batch_input_invariance_pytorch,
    compute_input_invariance,
    compute_input_invariance_pytorch,
)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


_TORCH_REQUIRED = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
_APIS = ["translation", pytest.param("compensated", marks=_TORCH_REQUIRED)]
_OUTPUT_STORAGE = [
    "fresh",
    "reused",
    "reused_view",
    pytest.param("shared_tensor", marks=_TORCH_REQUIRED),
]


def _call(api, explain_func, verify_determinism=True, *, batch=False):
    values = np.array([[2.0, 4.0], [-1.0, 3.0]]) if batch else np.array([2.0, 4.0])
    shift = np.array([1.0, -3.0])
    before_values = values.copy()
    before_shift = shift.copy()
    if api == "translation":
        operation = compute_batch_input_invariance if batch else compute_input_invariance
        try:
            with pytest.warns(RuntimeWarning, match="uncompensated"):
                return operation(explain_func, values, shift, verify_determinism=verify_determinism)
        finally:
            np.testing.assert_array_equal(values, before_values)
            np.testing.assert_array_equal(shift, before_shift)

    model = torch.nn.Sequential(torch.nn.Linear(2, 1, dtype=torch.float64))
    with torch.no_grad():
        model[0].weight.fill_(1.0)
        model[0].bias.zero_()
    model.eval()
    before_state = {name: value.clone() for name, value in model.state_dict().items()}
    source_gradients = []
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
        source_gradients.append((parameter, parameter.grad, parameter.grad.clone()))
    callback_models = {}

    def wrapped_callback(private_model, instance):
        assert private_model is not model
        identity = id(private_model)
        if identity not in callback_models:
            callback_models[identity] = (
                private_model,
                {name: value.clone() for name, value in private_model.state_dict().items()},
            )
        return explain_func(instance)

    operation = (
        compute_batch_input_invariance_pytorch if batch else compute_input_invariance_pytorch
    )
    try:
        return operation(
            model,
            wrapped_callback,
            values,
            shift,
            verify_determinism=verify_determinism,
        )
    finally:
        np.testing.assert_array_equal(values, before_values)
        np.testing.assert_array_equal(shift, before_shift)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before_state[name], rtol=0, atol=0)
        assert not model.training
        for parameter, gradient, before_gradient in source_gradients:
            assert parameter.grad is gradient
            torch.testing.assert_close(parameter.grad, before_gradient, rtol=0, atol=0)
        for private_model, before_private_state in callback_models.values():
            assert not private_model.training
            for name, value in private_model.state_dict().items():
                torch.testing.assert_close(value, before_private_state[name], rtol=0, atol=0)


def _attribution_callback(output_storage):
    buffer = np.empty(4 if output_storage == "reused_view" else 2, dtype=np.float64)
    selected = buffer[::2] if output_storage == "reused_view" else buffer
    calls = []

    def gradient_times_input(values):
        # E(x)=x is gradient-times-input for the unit-weight linear model above.
        calls.append(values.copy())
        if output_storage == "fresh":
            return values.copy()
        selected[:] = values
        if output_storage == "shared_tensor":
            # New Tensor objects can still share the same overwritten storage.
            return torch.from_numpy(selected)
        return selected

    return gradient_times_input, selected, calls


@pytest.mark.parametrize("api", _APIS)
@pytest.mark.parametrize("verify_determinism", [False, True])
@pytest.mark.parametrize("output_storage", _OUTPUT_STORAGE)
def test_invariance_retains_each_callback_result(api, verify_determinism, output_storage):
    callback, buffer, calls = _attribution_callback(output_storage)
    result = _call(api, callback, verify_determinism)
    # RMS([2,4] - [3,1]) = sqrt((1+9)/2).
    assert result == pytest.approx(np.sqrt(5.0))
    if output_storage != "fresh":
        np.testing.assert_array_equal(buffer, calls[-1])


@pytest.mark.parametrize("api", _APIS)
@pytest.mark.parametrize("verify_determinism", [False, True])
@pytest.mark.parametrize(
    "output_storage", ["fresh", "reused", pytest.param("shared_tensor", marks=_TORCH_REQUIRED)]
)
def test_batch_invariance_retains_each_callback_result(api, verify_determinism, output_storage):
    callback, buffer, calls = _attribution_callback(output_storage)
    result = _call(api, callback, verify_determinism, batch=True)
    assert result["scores"] == pytest.approx([np.sqrt(5.0), np.sqrt(5.0)])
    assert result["mean"] == pytest.approx(np.sqrt(5.0))
    assert result["std"] == 0.0
    assert result["n_evaluated"] == 2
    if output_storage != "fresh":
        np.testing.assert_array_equal(buffer, calls[-1])


@pytest.mark.parametrize("api", _APIS)
@pytest.mark.parametrize(
    "output_storage", ["fresh", "reused", pytest.param("shared_tensor", marks=_TORCH_REQUIRED)]
)
def test_reused_callback_result_cannot_hide_nondeterminism(api, output_storage):
    buffer = np.empty(2, dtype=np.float64)
    calls = 0

    def stochastic_attributions(_values):
        nonlocal calls
        calls += 1
        buffer.fill(calls)
        if output_storage == "fresh":
            return buffer.copy()
        if output_storage == "shared_tensor":
            return torch.from_numpy(buffer)
        return buffer

    with pytest.raises(RuntimeError, match="stochastic"):
        _call(api, stochastic_attributions, verify_determinism=True)
    # The metric must not alter callback-owned storage while snapshotting it.
    np.testing.assert_array_equal(buffer, np.full(2, calls, dtype=np.float64))
