"""Permutation score-drop summaries stay accurate across binary64 scales."""

from statistics import mean, pstdev

import numpy as np
import pytest

from explainiverse.explainers import PermutationImportanceExplainer


class IdentityRegressor:
    task = "regression"

    def predict(self, rows):
        return rows[:, 0]


def _explain(scorer):
    values = np.arange(4)
    return (
        PermutationImportanceExplainer(
            IdentityRegressor(),
            values[:, None],
            values,
            ["x"],
            scoring_fn=scorer,
            n_repeats=3,
            random_state=3,
        )
        .explain()
        .explanation_data
    )


@pytest.mark.parametrize("drop", [2.0, 1e308])
def test_constant_permutation_drops_have_zero_spread(drop):
    def score(truth, predictions):
        return 0.0 if np.array_equal(truth, predictions) else -drop

    data = _explain(score)
    assert data["importances"]["x"] == [drop] * 3
    assert data["feature_attributions"]["x"] == drop
    assert data["std"]["x"] == 0.0


@pytest.mark.parametrize(
    "drops",
    [(1.0, 2.0, 3.0), (1e200, 2e200, 3e200), (1e-200, 2e-200, 3e-200), (1e308, 1.0, -1e308)],
)
def test_varied_permutation_drops_preserve_mean_and_population_spread(drops):
    # Score the first observation with a penalty depending on its prediction.
    # The seeded permutations place 1, 2, and 3 in this row, respectively.
    penalties = (0.0, *drops)

    def score(truth, predictions):
        return -penalties[abs(predictions[0] - truth[0])]

    data = _explain(score)
    assert data["importances"]["x"] == list(drops)
    assert data["feature_attributions"]["x"] == pytest.approx(mean(drops), rel=2e-15, abs=0)
    assert data["std"]["x"] == pytest.approx(pstdev(drops), rel=2e-15, abs=0)


def test_unrepresentable_raw_permutation_drop_raises():
    def score(truth, predictions):
        return 1e308 if np.array_equal(truth, predictions) else -1e308

    with pytest.raises(FloatingPointError, match="score drop"):
        _explain(score)
