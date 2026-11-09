"""Relative stability preserves representation and logit callback snapshots."""

import numpy as np
import pytest

from explainiverse.core.explanation import Explanation
from explainiverse.evaluation.robustness import (
    compute_batch_relative_output_stability,
    compute_batch_relative_representation_stability,
    compute_batch_relative_stability,
    compute_relative_input_stability,
    compute_relative_output_stability,
    compute_relative_representation_stability,
    compute_relative_stability,
)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


_TORCH_REQUIRED = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
_STORAGE = [
    "fresh",
    "reused",
    "reused_row",
    "strided_view",
    pytest.param("shared_tensor", marks=_TORCH_REQUIRED),
]


class _LinearExplainer:
    def explain(self, instance):
        return Explanation(
            explainer_name="linear",
            target_class=0,
            feature_names=["x"],
            explanation_data={"feature_attributions": {"x": float(3.0 * instance[0])}},
        )


class _LinearLogitClassifier:
    def predict_proba(self, inputs):
        first_probability = 1.0 / (1.0 + np.exp(-3.0 * inputs[:, 0]))
        return np.column_stack((first_probability, 1.0 - first_probability))


def _callback(storage, *, two_logits=False):
    width = 2 if two_logits else 1
    scale = 3.0 if two_logits else 2.0
    backing = np.zeros(2 * width if storage == "strided_view" else width, dtype=np.float64)
    buffer = backing[::2] if storage == "strided_view" else backing
    observations = []

    def evaluate(instance):
        buffer[0] = scale * instance[0]
        observations.append(buffer.copy())
        if storage == "fresh":
            return buffer.copy()
        if storage == "reused_row":
            return buffer.reshape(1, -1)
        if storage == "shared_tensor":
            return torch.from_numpy(buffer)
        return buffer

    return evaluate, buffer, observations


def _assert_callback_storage_unchanged(callback_state):
    _callback_function, buffer, observations = callback_state
    if observations:
        np.testing.assert_array_equal(buffer, observations[-1])


def _assert_detail(result, expected):
    assert result["n_valid"] == 5
    assert result["n_total"] == 5
    assert result["score"] == pytest.approx(expected)
    assert result["max"] == pytest.approx(expected)
    assert result["mean"] == pytest.approx(expected)
    assert result["median"] == pytest.approx(expected)
    np.testing.assert_allclose(result["perturbation_scores"], np.full(5, expected))
    assert result["aggregation_is_canonical"] is True
    assert result["perturbation_space"] == "preprocessed_input_space"


@pytest.mark.parametrize("api", ["rrs", "ros", "combined", "ris_bound"])
@pytest.mark.parametrize("storage", _STORAGE)
@pytest.mark.parametrize("anchor", [1.0, 2.0])
def test_relative_stability_snapshots_callback_outputs(api, storage, anchor):
    instance = np.array([anchor])
    before = instance.copy()
    kwargs = {
        "explainer": _LinearExplainer(),
        "model": _LinearLogitClassifier(),
        "instance": instance,
        "n_perturbations": 5,
        "noise_scale": 0.1,
        "seed": 0,
        "return_details": True,
    }
    representation = _callback(storage)
    logits = _callback(storage, two_logits=True)
    bound_result = None
    if api == "rrs":
        results = [
            (
                compute_relative_representation_stability(
                    **kwargs, representation_fn=representation[0]
                ),
                1.0,
            )
        ]
    elif api == "ros":
        results = [
            (compute_relative_output_stability(**kwargs, logit_fn=logits[0]), 1.0 / (3.0 * anchor))
        ]
    elif api == "combined":
        combined = compute_relative_stability(
            **kwargs, representation_fn=representation[0], logit_fn=logits[0]
        )
        results = [
            (combined["ris"], 1.0),
            (combined["rrs"], 1.0),
            (combined["ros"], 1.0 / (3.0 * anchor)),
        ]
    else:
        bound_result = compute_relative_input_stability(
            **kwargs, representation_fn=representation[0]
        )
        results = [(bound_result, 1.0)]
    np.testing.assert_array_equal(instance, before)
    _assert_callback_storage_unchanged(representation)
    _assert_callback_storage_unchanged(logits)
    # E(x)=3x and L(x)=2x have identical percent changes. Logits [3x,0]
    # have absolute change 3*|delta|, so ROS at anchor x is 1/(3*x).
    for result, expected in results:
        _assert_detail(result, expected)
    if bound_result is not None:
        # lambda=||L(x)||/||x||=2, sampled representation Lipschitz=2,
        # and RRS=1. This nonzero Equation 4 diagnostic is therefore 4.
        assert bound_result["empirical_bound_estimate"] == pytest.approx(4.0)
        assert bound_result["theoretical_bound"] is None


@pytest.mark.parametrize("api", ["rrs", "ros", "combined"])
@pytest.mark.parametrize(
    "storage", ["fresh", "reused", pytest.param("shared_tensor", marks=_TORCH_REQUIRED)]
)
def test_batch_relative_stability_snapshots_callback_outputs(api, storage):
    inputs = np.array([[1.0], [2.0]])
    before = inputs.copy()
    kwargs = {
        "explainer": _LinearExplainer(),
        "model": _LinearLogitClassifier(),
        "X": inputs,
        "n_perturbations": 5,
        "noise_scale": 0.1,
        "seed": 0,
    }
    representation = _callback(storage)
    logits = _callback(storage, two_logits=True)
    if api == "rrs":
        results = [
            (
                compute_batch_relative_representation_stability(
                    **kwargs, representation_fn=representation[0]
                ),
                [1.0, 1.0],
            )
        ]
    elif api == "ros":
        results = [
            (
                compute_batch_relative_output_stability(**kwargs, logit_fn=logits[0]),
                [1.0 / 3.0, 1.0 / 6.0],
            )
        ]
    else:
        combined = compute_batch_relative_stability(
            **kwargs, representation_fn=representation[0], logit_fn=logits[0]
        )
        results = [
            (combined["ris"], [1.0, 1.0]),
            (combined["rrs"], [1.0, 1.0]),
            (combined["ros"], [1.0 / 3.0, 1.0 / 6.0]),
        ]
    np.testing.assert_array_equal(inputs, before)
    _assert_callback_storage_unchanged(representation)
    _assert_callback_storage_unchanged(logits)
    for result, expected in results:
        assert result["n_evaluated"] == 2
        assert result["n_attempted"] == 2
        assert result["n_undefined"] == 0
        np.testing.assert_allclose(result["scores"], expected)
        assert result["mean"] == pytest.approx(np.mean(expected))
        assert result["std"] == pytest.approx(np.std(expected), abs=1e-14)


@pytest.mark.parametrize("reuse_output", [False, True])
def test_combined_callbacks_can_share_one_buffer(reuse_output):
    shared = np.zeros(2, dtype=np.float64)
    instance = np.array([1.0])
    before = instance.copy()
    observations = []

    def representation(values):
        shared[0] = 2.0 * values[0]
        observations.append(shared.copy())
        return shared[:1] if reuse_output else shared[:1].copy()

    def logits(values):
        shared[0] = 3.0 * values[0]
        observations.append(shared.copy())
        return shared if reuse_output else shared.copy()

    result = compute_relative_stability(
        _LinearExplainer(),
        _LinearLogitClassifier(),
        instance,
        representation_fn=representation,
        logit_fn=logits,
        n_perturbations=5,
        noise_scale=0.1,
        seed=0,
        return_details=True,
    )
    np.testing.assert_array_equal(instance, before)
    np.testing.assert_array_equal(shared, observations[-1])
    _assert_detail(result["ris"], 1.0)
    _assert_detail(result["rrs"], 1.0)
    _assert_detail(result["ros"], 1.0 / 3.0)
