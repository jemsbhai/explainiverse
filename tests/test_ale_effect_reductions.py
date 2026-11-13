"""Public ALE reductions preserve representable local effects and centered curves."""

import numpy as np
import pytest

from explainiverse.explainers import ALEExplainer

_LARGE = float.fromhex("0x1p+1023")
_TINY = float.fromhex("0x0.0000000000001p-1022")


def _explain(levels, responses):
    # Column two identifies the reference row, so its response surface can
    # differ from other rows even after the first column is perturbed.
    table = np.asarray(responses, dtype=float)
    rows = np.column_stack((levels, np.arange(len(levels))))

    class LookupRegressor:
        task = "regression"

        def predict(self, values):
            return table[values[:, 0], values[:, 1]]

    return (
        ALEExplainer(LookupRegressor(), rows, ["x", "row"], n_bins=len(rows))
        .explain("x")
        .explanation_data
    )


@pytest.mark.parametrize(
    ("lower", "upper", "mean"),
    [
        pytest.param([0] * 4, [2] * 4, 2.0, id="ordinary"),
        pytest.param([0] * 4, [_LARGE] * 4, _LARGE, id="sum-overflows"),
        pytest.param([0] * 4, [_LARGE, 1, -_LARGE, 3], 1.0, id="cancellation"),
        pytest.param([-1, 0, 0, 0], [_LARGE, -_LARGE, 0, 3], 1.0, id="rounded-difference"),
        pytest.param(
            [-_LARGE, _LARGE, 0, 0], [_LARGE, -_LARGE, 2, 2], 1.0, id="differences-overflow"
        ),
        pytest.param([0] * 4, [_LARGE, 8 * _TINY, -_LARGE, 0], 2 * _TINY, id="subnormal-residual"),
    ],
)
def test_one_bin_ale_is_the_mean_response_change(lower, upper, mean):
    data = _explain([0, 0, 1, 1], [lower, upper])
    assert data["bin_counts"] == [4]
    assert data["local_effects"] == [mean]
    assert data["ale_values"] == [-mean / 2, mean / 2]
    assert data["ale_values_at_bin_centers"] == [0.0]
    assert data["curve_range"] == mean


def test_ale_weights_trapezoids_with_large_responses():
    levels = [0, 0, 0, 1, 2, 3]
    responses = [[y] * len(levels) for y in [0.0, 0.0, 1.5 * _LARGE, 1.5 * _LARGE]]
    data = _explain(levels, responses)

    # Four rows see midpoint zero, one sees 0.75L and one sees 1.5L.
    # The empirical center is 0.375L. Both ends of the last bin are 1.125L
    # above this center, so their midpoint must also equal 1.125L.
    assert data["bin_counts"] == [4, 1, 1]
    assert data["local_effects"] == [0, 1.5 * _LARGE, 0]
    assert data["ale_values"] == [x * _LARGE for x in [-0.375, -0.375, 1.125, 1.125]]
    assert data["ale_values_at_bin_centers"] == [x * _LARGE for x in [-0.375, 0.375, 1.125]]
    assert data["curve_range"] == 1.5 * _LARGE


def test_ale_rounds_centered_subnormal_values_at_output():
    levels = [0, 1, 2]
    responses = [[y] * len(levels) for y in [0.0, _TINY, 10 * _TINY]]
    data = _explain(levels, responses)

    # In units of the smallest subnormal, the trapezoid midpoints are 1/2
    # and 11/2, with empirical weights 2/3 and 1/3. The center is 13/6.
    assert data["local_effects"] == [_TINY, 9 * _TINY]
    assert data["ale_values"] == [-2 * _TINY, -_TINY, 8 * _TINY]
    assert data["ale_values_at_bin_centers"] == [-2 * _TINY, 3 * _TINY]
    assert data["curve_range"] == 10 * _TINY


@pytest.mark.parametrize(
    "levels_and_responses",
    [
        pytest.param([(0, -_LARGE), (1, _LARGE)], id="unrepresentable-local-effect"),
        pytest.param([(0, -_LARGE), (1, 0), (2, _LARGE)], id="unrepresentable-curve-range"),
    ],
)
def test_ale_rejects_unrepresentable_effects(levels_and_responses):
    levels, values = zip(*levels_and_responses)
    with pytest.raises(FloatingPointError):
        _explain(levels, [[y] * len(levels) for y in values])


def test_curve_range_is_the_span_of_the_published_centered_edges():
    data = _explain([0, 1], [[0, 0], [3 * _TINY, 3 * _TINY]])
    # Half of 3 units rounds to 2, so the published span is 4 units.
    assert data["ale_values"] == [-2 * _TINY, 2 * _TINY]
    assert data["ale_values_at_bin_centers"] == [0]
    assert data["curve_range"] == 4 * _TINY


@pytest.mark.parametrize("upper", [[_TINY, 0, 0, 0], [_TINY] * 4], ids=["mean", "centered-edge"])
def test_ale_rejects_nonzero_effects_that_underflow(upper):
    with pytest.raises(FloatingPointError):
        _explain([0, 0, 1, 1], [[0] * 4, upper])
