"""Primary-formula and Quantus-reference checks for Efficient MPRT."""

import numpy as np
import pytest

from explainiverse.evaluation.randomisation import (
    _add_noise_to_input,
    _discrete_entropy,
    compute_batch_efficient_mprt,
    compute_batch_smooth_mprt,
    compute_efficient_mprt,
    compute_smooth_mprt,
)


@pytest.mark.quantus_reference
def test_histogram_entropy_matches_quantus_reference():
    pytest.importorskip("quantus")
    from quantus.functions.complexity_func import discrete_entropy

    attributions = np.array([-2.0, -1.5, -1.0, 0.2, 0.3, 2.0, 2.0])
    expected = discrete_entropy(
        a=attributions,
        x=attributions,
        n_bins=4,
    )

    assert _discrete_entropy(attributions, n_bins=4) == pytest.approx(expected)


def test_signed_histogram_is_not_absolute_magnitude_entropy():
    signed = np.array([-1.0, -1.0, 1.0, 1.0])

    assert _discrete_entropy(signed, n_bins=2) == pytest.approx(np.log(2.0))


def test_efficient_mprt_uses_relative_rise_from_paper():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)

    original_attr = np.array([[0.0, 0.0, 0.0, 1.0]])
    randomised_attr = np.array([[-1.0, -0.25, 0.25, 1.0]])

    def explain_func(current_model, x, y):
        weights = current_model.weight.detach().cpu().numpy()
        if np.allclose(weights, 1.0):
            return original_attr.copy()
        return randomised_attr.copy()

    original_entropy = _discrete_entropy(original_attr, n_bins=2)
    randomised_entropy = _discrete_entropy(randomised_attr, n_bins=2)
    expected = (randomised_entropy - original_entropy) / original_entropy
    x = np.zeros((1, 4), dtype=np.float32)
    y = np.array([0])

    aggregate = compute_efficient_mprt(model, x, y, explain_func, seed=4, n_bins=2)
    per_sample = compute_batch_efficient_mprt(model, x, y, explain_func, seed=4, n_bins=2)

    assert aggregate == pytest.approx(expected)
    assert per_sample == pytest.approx([expected])


def test_zero_original_complexity_is_explicitly_undefined():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(3, 1, bias=False)

    def constant_explanation(current_model, x, y):
        return np.ones((1, 3))

    with pytest.raises(ValueError, match="undefined"):
        compute_efficient_mprt(
            model,
            np.zeros((1, 3), dtype=np.float32),
            np.array([0]),
            constant_explanation,
            seed=1,
            n_bins=3,
        )


@pytest.mark.parametrize("n_bins", [0, -1, 1.5, True])
def test_invalid_histogram_bin_count_is_rejected(n_bins):
    error = TypeError if n_bins in {1.5, True} else ValueError
    with pytest.raises(error):
        _discrete_entropy(np.array([0.0, 1.0]), n_bins=n_bins)


def test_smooth_mprt_one_sample_uses_the_unperturbed_input():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(3, 1, bias=False)
    query = np.array([[0.0, 1.0, 3.0]], dtype=np.float32)
    seen = []

    def recording_explanation(current_model, x, y):
        seen.append(np.asarray(x).copy())
        return np.asarray(x).copy()

    compute_smooth_mprt(
        model,
        query,
        np.array([0]),
        recording_explanation,
        nr_samples=1,
        noise_magnitude=10.0,
        seed=5,
    )

    assert seen
    for explained_input in seen:
        np.testing.assert_array_equal(explained_input, query)


def test_smooth_mprt_each_average_includes_original_as_final_sample():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(3, 1, bias=False)
    query = np.array([[0.0, 1.0, 3.0]], dtype=np.float32)
    seen = []

    def recording_explanation(current_model, x, y):
        seen.append(np.asarray(x).copy())
        return np.asarray(x).copy()

    compute_smooth_mprt(
        model,
        query,
        np.array([0]),
        recording_explanation,
        nr_samples=3,
        noise_magnitude=0.2,
        seed=5,
    )

    assert len(seen) % 3 == 0
    for start in range(0, len(seen), 3):
        assert not np.array_equal(seen[start], query)
        np.testing.assert_array_equal(seen[start + 2], query)


def test_constant_input_noise_scale_matches_zero_range_reference():
    query = np.ones((1, 4))
    noisy = _add_noise_to_input(query, 0.5, np.random.default_rng(8))
    np.testing.assert_array_equal(noisy, query)


@pytest.mark.parametrize("metric", [compute_smooth_mprt, compute_batch_smooth_mprt])
@pytest.mark.parametrize(
    "x_batch,y_batch,match",
    [
        (np.zeros((0, 3), dtype=np.float32), np.array([], dtype=int), "must not be empty"),
        (np.zeros((1, 3), dtype=np.float32), np.array([0, 1]), "same batch size"),
    ],
)
def test_smooth_mprt_variants_share_the_batch_contract(metric, x_batch, y_batch, match):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(3, 1, bias=False)

    with pytest.raises(ValueError, match=match):
        metric(model, x_batch, y_batch, lambda current_model, x, y: x, nr_samples=1)


@pytest.mark.parametrize("metric", [compute_efficient_mprt, compute_batch_efficient_mprt])
def test_efficient_mprt_rejects_models_without_randomisable_layers(metric):
    torch = pytest.importorskip("torch")
    model = torch.nn.Identity()

    with pytest.raises(ValueError, match="no layers with learnable parameters"):
        metric(
            model,
            np.zeros((1, 3), dtype=np.float32),
            np.array([0]),
            lambda current_model, x, y: np.array([[0.0, 1.0, 2.0]]),
            n_bins=3,
        )


@pytest.mark.parametrize("metric", [compute_efficient_mprt, compute_batch_efficient_mprt])
def test_efficient_mprt_variants_validate_model_callable_batch_and_seed(metric):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(3, 1, bias=False)
    x = np.zeros((1, 3), dtype=np.float32)
    y = np.array([0])

    def explanation(current_model, values, target):
        del current_model, target
        return values

    with pytest.raises(TypeError, match="torch.nn.Module"):
        metric(object(), x, y, explanation)
    with pytest.raises(TypeError, match="explain_func"):
        metric(model, x, y, None)
    with pytest.raises(TypeError, match="NumPy arrays"):
        metric(model, x.tolist(), y, explanation)
    with pytest.raises(ValueError, match="non-negative"):
        metric(model, x, y, explanation, seed=-1)
