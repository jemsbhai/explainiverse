"""Default SAGE regression loss keeps integer operands exact through subtraction."""

from fractions import Fraction

import numpy as np
import pytest

from explainiverse.explainers import SAGEExplainer


def _mse(targets, predictions):
    pairs = zip(targets.ravel().tolist(), predictions.ravel().tolist())
    return float(sum((Fraction(y) - Fraction(p)) ** 2 for y, p in pairs) / targets.size)


@pytest.mark.parametrize(
    "targets, predictions, donor",
    [
        pytest.param(
            np.array([2**62, 2**62], dtype=np.int64),
            np.array([[2**62 + 1], [2**62 - 3]], dtype=np.int64),
            2**62,
            id="signed-vector-target-column-prediction",
        ),
        pytest.param(
            np.array([2**64 - 1, 2**64 - 4], dtype=np.uint64),
            np.array([2**64 - 2, 2**64 - 1], dtype=np.uint64),
            2**64 - 2048,
            id="near-unsigned-maximum",
        ),
        pytest.param(
            np.array([[2**63 - 1, 2**63 - 3], [2**63 - 5, 2**63 - 7]], dtype=np.int64),
            np.array([[2**63 + 1, 2**63 + 2], [2**63 + 3, 2**63 + 4]], dtype=np.uint64),
            2**63,
            id="mixed-signedness-multioutput",
        ),
        pytest.param(
            np.array([-(2**62) + 1, -(2**62) + 3], dtype=np.int64),
            np.array([-(2.0**62), -(2.0**62) + 512]),
            -(2**62),
            id="integer-target-floating-prediction",
        ),
        pytest.param(
            np.array([2.0**53, 0.5]),
            np.array([2**53 + 1, 2], dtype=np.uint64),
            0,
            id="floating-target-integer-prediction",
        ),
        pytest.param(
            np.array([[1], [-3]], dtype=np.int16),
            np.array([2, 0], dtype=np.int8),
            1,
            id="small-column-target-vector-prediction",
        ),
        pytest.param(
            np.array(["1.25", "-2.5"]),
            np.array([2, -3], dtype=np.int64),
            1,
            id="numeric-string-targets",
        ),
    ],
)
def test_sage_regression_uses_exact_integer_residuals(targets, predictions, donor):
    donor_row = np.full_like(predictions[:1], donor)
    lookup = np.concatenate((donor_row, predictions))

    class LookupRegressor:
        def predict(self, rows):
            return lookup[rows[:, 0]]

    data = (
        SAGEExplainer(
            LookupRegressor(),
            np.arange(1, len(targets) + 1)[:, None],
            targets,
            ["x"],
            task="regression",
            background_data=np.array([[0]]),
            n_permutations=1,
        )
        .explain()
        .explanation_data
    )
    baseline = _mse(targets, np.repeat(donor_row, len(targets), axis=0))
    full = _mse(targets, predictions)
    assert data["baseline_loss"] == baseline
    assert data["full_loss"] == full
    assert data["total_value"] == baseline - full
    assert data["feature_attributions"] == {"x": baseline - full}
    assert data["efficiency_error"] == 0.0
