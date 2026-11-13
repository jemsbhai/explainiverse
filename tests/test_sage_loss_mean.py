"""A SAGE loss callback can return scalar or per-observation losses."""

from statistics import mean

import numpy as np
import pytest

from explainiverse.explainers import SAGEExplainer


class IdentityRegressor:
    def predict(self, rows):
        return rows[:, 0]


@pytest.mark.parametrize(
    "losses",
    [
        np.array(2.0),
        np.array([1.0, 2.0, 3.0]),
        np.array([1e308, 1e308]),
        np.array([1e308, 1.0, -1e308]),
        np.array([[1e308, 1.0], [-1e308, 2.0]]),
    ],
    ids=["scalar", "ordinary", "overflow", "cancellation", "multioutput"],
)
def test_sage_custom_loss_mean_preserves_the_one_feature_game(losses):
    truth = np.arange(1, losses.shape[0] + 1) if losses.ndim else np.array([1])

    def loss(targets, predictions):
        return losses if np.all(predictions == 0) else np.zeros_like(losses)

    data = (
        SAGEExplainer(
            IdentityRegressor(),
            truth[:, None],
            truth,
            ["x"],
            task="regression",
            background_data=np.array([[0]]),
            loss_fn=loss,
            n_permutations=1,
        )
        .explain()
        .explanation_data
    )

    expected = mean(map(float, losses.flat))
    assert data["baseline_loss"] == expected
    assert data["full_loss"] == 0.0
    assert data["feature_attributions"] == {"x": expected}
    assert data["total_value"] == expected
    assert data["efficiency_error"] == 0.0
