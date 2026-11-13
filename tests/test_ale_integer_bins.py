"""ALE uses observed integer edges and preserves other features during evaluation."""

from decimal import Decimal

import numpy as np
import pytest

from explainiverse.explainers import ALEExplainer


class CenteredIdentity:
    task = "regression"

    def __init__(self, origin):
        self.origin = origin

    def predict(self, rows):
        return np.array([int(x) - self.origin for x in rows[:, 0]], dtype=float)


@pytest.mark.parametrize(
    "dtype, origin",
    [
        (np.int64, 7),
        (np.int64, -(2**63)),
        (np.int64, 2**53 + 1),
        (np.uint64, 2**64 - 4),
        (object, 2**53 + 1),
    ],
)
def test_ale_empirical_edges_preserve_integer_levels(dtype, origin):
    inputs = np.array([[origin + offset] for offset in (2, 0, 3, 1)], dtype=dtype)
    data = (
        ALEExplainer(CenteredIdentity(origin), inputs, ["x"], n_bins=2)
        .explain("x")
        .explanation_data
    )

    # The empirical minimum, lower median, and maximum land at offsets 0, 1, 3.
    edges = [origin, origin + 1, origin + 3]
    assert data["bin_edges"] == data["grid_values"] == edges
    assert data["bin_counts"] == [2, 2]
    assert data["local_effects"] == [1.0, 2.0]
    assert data["ale_values"] == [-1.25, -0.25, 1.75]
    assert data["bin_centers"] == [
        float(Decimal(left + right) / 2) for left, right in zip(edges, edges[1:])
    ]


def test_ale_orders_mixed_float_and_integer_observations_exactly():
    origin = 2**53
    inputs = np.array(
        [[np.float64(origin)], [np.uint64(origin + 1)], [float(origin + 2)], [origin + 3]],
        dtype=object,
    )

    data = (
        ALEExplainer(CenteredIdentity(origin), inputs, ["x"], n_bins=2)
        .explain("x")
        .explanation_data
    )
    assert data["bin_edges"] == [origin, origin + 1, origin + 3]
    assert data["local_effects"] == [1.0, 2.0]


@pytest.mark.parametrize(
    "dtype, even_id",
    [(np.int64, 12), (np.int64, 2**53), (np.uint64, 2**64 - 2), (object, 2**53)],
)
def test_ale_preserves_untouched_integer_column(dtype, even_id):
    class Interaction:
        task = "regression"

        def predict(self, rows):
            return np.array([int(x) * (int(group) % 2) for x, group in rows], dtype=float)

    inputs = np.array([[0, even_id], [1, even_id + 1]], dtype=dtype)
    before = inputs.copy()
    data = ALEExplainer(Interaction(), inputs, ["x", "group"]).explain("x").explanation_data
    # Moving x from zero to one changes the response only for the odd group.
    assert data["local_effects"] == [0.5]
    assert data["ale_values"] == [-0.25, 0.25]
    np.testing.assert_array_equal(inputs, before)
