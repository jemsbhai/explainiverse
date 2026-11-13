"""ALE bin edges follow the inverse empirical CDF at exact rank boundaries."""

import numpy as np
import pytest

from explainiverse.explainers import ALEExplainer


class QuadraticRegressor:
    task = "regression"

    def predict(self, rows):
        return rows[:, 0] ** 2


@pytest.mark.parametrize(
    "observations, n_bins, edges, counts",
    [
        (range(6), 6, [0, 1, 2, 3, 4, 5], [2, 1, 1, 1, 1]),
        ([4, 2, 0, 6, 1, 8, 3, 7, 5], 3, [0, 2, 5, 8], [3, 3, 3]),
        ([0, 4, 2, 0, 1, 5, 3, 4, 0], 3, [0, 3, 5], [6, 3]),
        (range(8), 3, [0, 2, 5, 7], [3, 3, 2]),
        (range(6), 9, [0, 1, 2, 3, 4, 5], [2, 1, 1, 1, 1]),
    ],
    ids=["six-sixths", "nine-in-thirds", "tied-thirds", "inexact-ranks", "more-bins-than-rows"],
)
def test_ale_uses_exact_empirical_ranks(observations, n_bins, edges, counts):
    # At 1/3 and 2/3 of nine rows, the quantiles are sorted rows 3 and 6
    # (one-based). The tied sample has the same value at ranks 1 and 3.
    rows = np.asarray(observations, dtype=float).reshape(-1, 1)
    result = ALEExplainer(QuadraticRegressor(), rows, ["x"], n_bins=n_bins).explain("x")
    data = result.explanation_data
    assert data["grid_values"] == data["bin_edges"] == edges
    assert data["bin_counts"] == counts
    assert result.metadata["n_effective_bins"] == len(counts)

    # For f(x)=x², the local difference is b²-a² on each interval [a,b].
    levels = np.square(edges)
    np.testing.assert_array_equal(data["local_effects"], np.diff(levels))
    center = np.average((levels[1:] + levels[:-1]) / 2, weights=counts)
    np.testing.assert_allclose(data["ale_values"], levels - center)
