"""Accuracy and failure-contract tests for standard MPRT."""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from explainiverse.evaluation.randomisation import (  # noqa: E402
    _extract_attribution_array,
    _get_named_layers,
    _randomise_layer_parameters,
    compute_mprt,
)


class SmallNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


def _gradient_explanation(model, x, y):
    tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    model(tensor)[0, int(y)].backward()
    return tensor.grad.detach().numpy()[0]


def test_seeded_reset_matches_modules_authoritative_initialiser():
    model = SmallNetwork()
    expected = copy.deepcopy(model)
    seed_source = np.random.default_rng(73)
    expected_seed = int(seed_source.integers(0, 2**31))

    with torch.random.fork_rng():
        torch.manual_seed(expected_seed)
        expected.linear.reset_parameters()

    _randomise_layer_parameters(model, "linear", rng=np.random.default_rng(73))

    assert torch.equal(model.linear.weight, expected.linear.weight)
    assert torch.equal(model.linear.bias, expected.linear.bias)


def test_seeded_reset_preserves_callers_torch_rng_stream():
    model = SmallNetwork()
    torch.manual_seed(9876)
    expected_next_values = torch.rand(8)

    torch.manual_seed(9876)
    _randomise_layer_parameters(model, "linear", rng=np.random.default_rng(3))
    observed_next_values = torch.rand(8)

    assert torch.equal(observed_next_values, expected_next_values)


def test_parameterless_requested_layer_is_rejected_instead_of_noop():
    with pytest.raises(ValueError, match="no direct learnable parameters"):
        _get_named_layers(SmallNetwork(), layer_names=["relu"])


class ParameterWithoutInitialisationContract(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))


def test_unknown_initialisation_contract_is_rejected_instead_of_fabricated():
    model = ParameterWithoutInitialisationContract()
    with pytest.raises(ValueError, match="does not expose reset_parameters"):
        _randomise_layer_parameters(model, "")


@pytest.mark.parametrize(
    "x_batch,y_batch,match",
    [
        (np.ones((0, 3), dtype=np.float32), np.array([], dtype=int), "must not be empty"),
        (np.ones((2, 3), dtype=np.float32), np.array([0]), "same batch size"),
        (np.ones((2, 3), dtype=np.float32), np.array([[0], [1]]), "one-dimensional"),
    ],
)
def test_high_level_mprt_rejects_invalid_batch_contract(x_batch, y_batch, match):
    with pytest.raises(ValueError, match=match):
        compute_mprt(
            SmallNetwork(),
            x_batch,
            y_batch,
            _gradient_explanation,
            similarity_func="mse",
            seed=1,
        )


@pytest.mark.parametrize("values", [np.array([]), np.array([1.0, np.nan])])
def test_attribution_extraction_rejects_noncomputable_values(values):
    with pytest.raises(ValueError):
        _extract_attribution_array(values)
