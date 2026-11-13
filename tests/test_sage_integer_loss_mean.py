"""Integer custom losses are averaged before SAGE rounds a coalition loss."""

from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from explainiverse.explainers import SAGEExplainer


def _explain(losses, task="regression", at_baseline=True):
    class Model:
        def predict(self, rows):
            values = rows[:, 0]
            return np.column_stack((1 - values, values)) if task == "classification" else values

    def custom_loss(targets, predictions):
        values = predictions[:, 1] if task == "classification" else predictions
        return losses if np.all(values == 0) == at_baseline else 0.0

    # Background predictions are zero and complete predictions are one.
    # Only one coalition has nonzero loss, so its mean determines the game.
    n_rows = max(1, losses.shape[0]) if losses.ndim else 1
    targets = np.ones(n_rows, dtype=int)
    return (
        SAGEExplainer(
            Model(),
            targets[:, None],
            targets,
            ["x"],
            task=task,
            background_data=np.array([[0]]),
            loss_fn=custom_loss,
            n_permutations=1,
        )
        .explain()
        .explanation_data
    )


@pytest.mark.parametrize(
    "losses, task, at_baseline",
    [
        (np.array([-(2**63) + 3, 2**63 - 1], dtype=np.int64), "regression", True),
        (np.array([[2**53 + 1, 1], [-(2**53), 0]], dtype=np.int64), "classification", False),
        (np.array([2**64 - 1, 2**64 - 2049], dtype=np.uint64), "regression", False),
        (np.array([[2**64 - 1, 2**64 - 2049]], dtype=np.uint64), "classification", True),
        (np.uint64(2**64 - 1), "regression", True),
        (np.array([[2, -1], [3, 5]], dtype=np.int16), "classification", False),
        (np.array([True, False, True]), "classification", True),
    ],
    ids=[
        "signed-extrema",
        "signed-matrix",
        "unsigned-vector",
        "unsigned-row",
        "scalar",
        "small",
        "bool",
    ],
)
def test_sage_scalarizes_integer_callback_losses_exactly(losses, task, at_baseline):
    data = _explain(losses, task, at_baseline)
    mean = float(Fraction(sum(map(int, losses.flat)), losses.size))
    baseline, full = (mean, 0.0) if at_baseline else (0.0, mean)
    assert data["baseline_loss"] == baseline
    assert data["full_loss"] == full
    assert data["total_value"] == baseline - full
    assert data["feature_attributions"] == {"x": baseline - full}
    assert data["efficiency_error"] == 0.0


@pytest.mark.parametrize(
    "losses",
    [np.array(["1.5", "-0.25"]), np.array([Decimal("1.5"), Decimal("-0.25")])],
    ids=["numeric-strings", "decimal-objects"],
)
def test_sage_converts_real_numeric_callback_losses(losses):
    assert _explain(losses)["total_value"] == 0.625


@pytest.mark.parametrize(
    "losses, message",
    [
        (np.array([], dtype=np.int64), "finite"),
        (np.array([np.inf]), "finite"),
        (np.array([0, 1j], dtype=object), "complex"),
    ],
    ids=["empty-integer", "nonfinite", "complex-object"],
)
def test_sage_rejects_invalid_callback_losses(losses, message):
    with pytest.raises(ValueError, match=message):
        _explain(losses)
