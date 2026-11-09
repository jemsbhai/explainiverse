"""ALE bin coordinates retain finite, accurately rounded interval midpoints."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from explainiverse.explainers.global_explainers.ale import ALEExplainer

_TINY = float(np.nextafter(0.0, 1.0))
_MAX = float(np.finfo(np.float64).max)


class _ConstantRegressor:
    task = "regression"
    prediction_output_kind = "regression_values"

    def predict(self, values):
        return np.zeros((len(values), 1))


def _exact_midpoint(left, right):
    # Independent high-precision decimal oracle for the rational implementation.
    with localcontext() as context:
        context.prec = 3000
        return float((Decimal.from_float(float(left)) + Decimal.from_float(float(right))) / 2)


def _explain_and_check_midpoints(values, n_bins):
    inputs = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    before = inputs.copy()
    explanation = ALEExplainer(
        _ConstantRegressor(), inputs, feature_names=["x"], n_bins=n_bins
    ).explain("x")
    data = explanation.explanation_data
    edges = np.asarray(data["bin_edges"])
    centers = np.asarray(data["bin_centers"])
    expected = (
        edges.copy()
        if len(edges) == 1
        else np.asarray([_exact_midpoint(left, right) for left, right in zip(edges, edges[1:])])
    )
    assert np.all(np.isfinite(centers))
    np.testing.assert_array_equal(centers, expected)
    if len(edges) > 1:
        assert np.all(edges[:-1] <= centers)
        assert np.all(centers <= edges[1:])
        assert np.all(centers[:-1] <= centers[1:])
    np.testing.assert_array_equal(data["ale_values"], np.zeros(len(edges)))
    np.testing.assert_array_equal(data["ale_values_at_bin_centers"], np.zeros(len(centers)))
    np.testing.assert_array_equal(data["local_effects"], np.zeros(len(centers)))
    np.testing.assert_array_equal(inputs, before)
    return data


@pytest.mark.parametrize(
    "endpoints",
    [
        pytest.param([1.0, 1.5], id="ordinary-positive"),
        pytest.param([-1.5, -1.0], id="ordinary-negative"),
        pytest.param([1e308, 1.5e308], id="huge-positive"),
        pytest.param([-1.5e308, -1e308], id="huge-negative"),
        pytest.param([-_MAX, _MAX], id="opposite-extremes"),
        pytest.param([-1.5e308, 1e308], id="opposite-asymmetric"),
        pytest.param([_TINY, 2 * _TINY], id="neighboring-subnormals"),
        pytest.param([-2 * _TINY, -_TINY], id="negative-subnormals"),
        pytest.param([-_TINY, _TINY], id="opposite-subnormals"),
        pytest.param([0.0, _TINY], id="half-subnormal-rounds-zero"),
        pytest.param([0.0, 3 * _TINY], id="subnormal-tie-rounds-even"),
        pytest.param([_TINY, 1e308], id="tiny-and-large"),
        pytest.param([-1e308, -_TINY], id="negative-large-and-tiny"),
    ],
)
def test_ale_bin_center_matches_exact_float_midpoint(endpoints):
    data = _explain_and_check_midpoints(endpoints, n_bins=1)
    assert data["bin_counts"] == [2]


@pytest.mark.parametrize(
    "values",
    [
        pytest.param([1e308, 1.2e308, 1.4e308, 1.6e308], id="large-positive"),
        pytest.param([-_MAX, -1e308, 0.0, 1e308, _MAX], id="crossing-zero"),
        pytest.param([_TINY, 2 * _TINY, 3 * _TINY, 4 * _TINY], id="subnormals"),
    ],
)
def test_ale_multiple_bin_centers_are_ordered_finite_and_bounded(values):
    data = _explain_and_check_midpoints(values, n_bins=len(values) - 1)
    assert len(data["bin_centers"]) == len(values) - 1
    assert sum(data["bin_counts"]) == len(values)


@pytest.mark.parametrize("value", [0.0, _TINY, -_TINY, 1e308, -1e308])
def test_ale_constant_feature_keeps_its_coordinate(value):
    data = _explain_and_check_midpoints([value, value, value], n_bins=3)
    assert data["bin_centers"] == [value]
    assert data["bin_counts"] == [3]
