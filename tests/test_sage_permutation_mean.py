"""Public SAGE values average scalar coalition-loss differences over orders."""

import numpy as np
import pytest

from explainiverse.explainers import SAGEExplainer


class CoalitionModel:
    def predict(self, rows):
        # With a single all-zero donor, this encodes the observed features.
        return rows @ (2 ** np.arange(rows.shape[1]))


def _game(losses, repetitions, seed=1):
    n_features = len(losses).bit_length() - 1
    rows = np.ones((1, n_features), dtype=int)
    return (
        SAGEExplainer(
            CoalitionModel(),
            rows,
            np.zeros(1),
            list("abc"[:n_features]),
            task="regression",
            loss_fn=lambda targets, predictions: losses[int(predictions[0])],
            background_data=np.zeros_like(rows),
            n_permutations=repetitions,
            random_state=seed,
        )
        .explain()
        .explanation_data
    )


@pytest.mark.parametrize("value", [2.0, 1e308, -1e308])
def test_repeated_identical_marginals_equal_the_one_feature_loss_drop(value):
    data = _game([value, 0.0], 4)
    assert data["feature_attributions"] == {"a": value}
    assert data["total_value"] == value
    assert data["efficiency_error"] == 0.0


@pytest.mark.parametrize("loss_after_a,expected_a", [(0.0, 0.0), (-1e308, 0.5e308)])
def test_opposing_marginals_cancel_before_rounding_the_mean(loss_after_a, expected_a):
    # Seed 1 samples each order twice. In the second game a single transition
    # spans 2e308 even though both published feature means are representable.
    data = _game([1e308, loss_after_a, 0.0, 1e308], 4)
    assert data["feature_attributions"] == {"a": expected_a, "b": -expected_a}
    assert data["efficiency_error"] == 0.0


def test_efficiency_retains_rounding_of_the_published_attributions():
    # Either feature reduces loss by 1 if it appears first (a twice, b once).
    data = _game([1.0, 0.0, 0.0, 0.0], 3)
    assert data["feature_attributions"] == {"a": 2 / 3, "b": 1 / 3}
    assert data["total_value"] == 1.0
    assert data["efficiency_error"] == -(2.0**-54)


def test_efficiency_allows_large_intermediate_sums():
    # Seed 5 visits a, b, c. The first two marginals sum to 2e308.
    data = _game([1e308, 0.0, 0.0, -1e308, 0.0, 0.0, 0.0, 0.0], 1, seed=5)
    assert data["feature_attributions"] == {"a": 1e308, "b": 1e308, "c": -1e308}
    assert data["total_value"] == 1e308
    assert data["efficiency_error"] == 0.0


@pytest.mark.parametrize(
    "losses,field",
    [
        ([1e308, -1e308], "attribution"),
        ([1e308, 0.0, 0.0, -1e308], "total"),
        ([np.nextafter(0.0, 1.0), 0.0, 0.0, 0.0], "attribution"),
    ],
)
def test_unrepresentable_game_summary_raises(losses, field):
    with pytest.raises(FloatingPointError, match=field):
        _game(losses, 4)
