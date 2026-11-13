"""Default SAGE regression loss averages squared residuals across rows and outputs."""

import numpy as np
import pytest

from explainiverse.explainers import SAGEExplainer


def _explain(targets, predictions):
    targets = np.asarray(targets, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    # Reference rows select their prediction by index. Donor row zero
    # supplies zero for every output in the empty coalition.
    lookup = np.concatenate((np.zeros_like(predictions[:1]), predictions))
    rows = np.arange(1, len(targets) + 1)[:, None]

    class LookupRegressor:
        def predict(self, values):
            return lookup[values[:, 0]]

    return (
        SAGEExplainer(
            LookupRegressor(),
            rows,
            targets,
            ["x"],
            task="regression",
            background_data=np.array([[0]]),
            n_permutations=1,
        )
        .explain()
        .explanation_data
    )


@pytest.mark.parametrize(
    ("targets", "predictions", "mse"),
    [
        pytest.param([2, -4], [[2], [-4]], 10.0, id="vector-target-column-prediction"),
        pytest.param([[2], [-4]], [2, -4], 10.0, id="column-target-vector-prediction"),
        pytest.param([2.0**511] * 4, [2.0**511] * 4, 2.0**1022, id="sum-of-squares-overflows"),
        pytest.param(
            [2.0**512, 0, 0, 0], [2.0**512, 0, 0, 0], 2.0**1022, id="one-square-overflows"
        ),
        pytest.param([[2.0**512, 0, 0, 0]], [[2.0**512, 0, 0, 0]], 2.0**1022, id="multioutput"),
        # The two squared errors are 1/4 and 1 minimum subnormal units.
        # Their exact mean (5/8 of a unit) rounds to one unit.
        pytest.param([2.0**-538, 2.0**-537], [2.0**-538, 2.0**-537], 2.0**-1074, id="subnormal"),
    ],
)
def test_sage_mse_with_perfect_full_predictions(targets, predictions, mse):
    data = _explain(targets, predictions)
    assert data["loss_name"] == "mean_squared_error"
    assert data["baseline_loss"] == mse
    assert data["full_loss"] == 0.0
    assert data["total_value"] == mse
    assert data["feature_attributions"] == {"x": mse}
    assert data["efficiency_error"] == 0.0


def test_sage_mse_retains_residual_precision_before_squaring():
    data = _explain([1.0], [-(2.0**-53)])
    # (1 + 2^-53)^2 = 1 + 2^-52 + 2^-106.
    assert data["full_loss"] == 1.0 + 2.0**-52
    assert data["baseline_loss"] == 1.0
    assert data["feature_attributions"] == {"x": -(2.0**-52)}


@pytest.mark.parametrize(
    ("targets", "predictions"),
    [
        ([[1, 2], [3, 4]], [[0], [0]]),
        ([[1], [2]], [[0, 0], [0, 0]]),
        ([[[1]], [[2]]], [[[0]], [[0]]]),
    ],
    ids=["missing-output", "extra-output", "rank-three"],
)
def test_sage_mse_rejects_incompatible_output_shapes(targets, predictions):
    with pytest.raises(ValueError):
        _explain(targets, predictions)


@pytest.mark.parametrize("target", [2.0**512, 2.0**-538], ids=["overflow", "underflow"])
def test_sage_rejects_unrepresentable_nonzero_mse(target):
    with pytest.raises(FloatingPointError):
        _explain([target], [target])
