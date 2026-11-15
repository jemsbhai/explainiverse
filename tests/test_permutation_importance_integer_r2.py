"""Default permutation R² preserves the variation in native integer responses."""

from fractions import Fraction
from statistics import mean, pstdev

import numpy as np
import pytest

from explainiverse.explainers import PermutationImportanceExplainer


def _r2(targets, predictions):
    n = len(targets)
    scores = []
    for ys, ps in zip(targets.reshape(n, -1).T, predictions.reshape(n, -1).T):
        truth = [Fraction(value) for value in ys.tolist()]
        residual = sum((y - Fraction(p)) ** 2 for y, p in zip(truth, ps.tolist()))
        center = sum(truth) / n
        variation = sum((y - center) ** 2 for y in truth)
        scores.append(1 - residual / variation if variation else int(residual == 0))
    return float(sum(scores) / len(scores))


_SIGNED = 2**62
_LAST = 2**64 - 1


@pytest.mark.parametrize(
    "targets,predictions",
    [
        pytest.param(
            np.array([_SIGNED + i for i in range(4)], dtype=np.int64),
            np.array([_SIGNED + i for i in range(4)], dtype=np.int64),
            id="signed-adjacent",
        ),
        pytest.param(
            np.array([[_LAST - i] for i in (0, 4, 3, 8)], dtype=np.uint64),
            np.array([_LAST - i for i in (1, 6, 5, 4)], dtype=np.uint64),
            id="unsigned-column-target-vector-prediction",
        ),
        pytest.param(
            np.array([-_SIGNED + i for i in (-1, 1, 3, 5)], dtype=np.int64),
            np.full((4, 1), float(-_SIGNED)),
            id="integer-target-floating-prediction",
        ),
        pytest.param(
            np.array([float(_SIGNED) + i for i in (0, 1024, 2048, 3072)]),
            np.array([_SIGNED + i for i in (1, 1023, 2049, 3071)], dtype=np.uint64),
            id="floating-target-integer-prediction",
        ),
        pytest.param(
            np.array([2**24 + i for i in (0, 2, 4, 6)], dtype=np.float32),
            np.array([2**24 + i for i in (1, 3, 5, 7)], dtype=np.int32),
            id="integer-variation-below-float32-spacing",
        ),
        pytest.param(
            np.array([[2**63 - i, 11] for i in (1, 3, 4, 6)], dtype=np.int64),
            np.array([[2**63 + i, 11] for i in (2, 4, 0, 7)], dtype=np.uint64),
            id="mixed-signedness-and-constant-output",
        ),
        pytest.param(
            np.full(4, _SIGNED, dtype=np.int64),
            np.array([_SIGNED + i for i in (0, 1, 0, 0)], dtype=np.int64),
            id="constant-target-one-inexact-prediction",
        ),
        pytest.param(
            np.full(4, _LAST, dtype=np.uint64),
            np.full((4, 1), _LAST, dtype=np.uint64),
            id="constant-target-perfect-prediction",
        ),
        pytest.param(
            np.array([1, 2, 3, 4], dtype=np.int16),
            np.array([[1], [2], [3], [5]], dtype=np.int8),
            id="small-signed-vector-target-column-prediction",
        ),
    ],
)
def test_permutation_r2_preserves_integer_operands(targets, predictions):
    class LookupRegressor:
        task = "regression"

        def predict(self, rows):
            return predictions[rows[:, 0]]

    indices = np.arange(4)
    result = PermutationImportanceExplainer(
        LookupRegressor(),
        np.column_stack([indices, indices]),
        targets,
        ["signal", "ignored"],
        n_repeats=3,
        random_state=3,
    ).explain()
    # Independently score the three known row orders for seed 3.
    orders = ([1, 2, 3, 0], [2, 3, 1, 0], [3, 2, 1, 0])
    baseline = _r2(targets, predictions)
    drops = [baseline - _r2(targets, predictions[order]) for order in orders]
    data = result.explanation_data
    assert data["baseline_score"] == pytest.approx(baseline, rel=1e-15, abs=0)
    assert data["importances"]["signal"] == pytest.approx(drops, rel=1e-15, abs=0)
    assert data["feature_attributions"]["signal"] == pytest.approx(mean(drops), rel=1e-15, abs=0)
    assert data["std"]["signal"] == pytest.approx(pstdev(drops), rel=2e-15, abs=0)
    assert data["importances"]["ignored"] == [0.0] * 3
    assert data["feature_attributions"]["ignored"] == data["std"]["ignored"] == 0.0


def test_unrepresentable_integer_target_r2_preserves_finite_score_error():
    class ConstantRegressor:
        task = "regression"

        def predict(self, rows):
            return np.full(len(rows), 1e200)

    targets = np.arange(3)
    explainer = PermutationImportanceExplainer(
        ConstantRegressor(), targets[:, None], targets, ["x"]
    )
    with pytest.raises(ValueError, match="scoring function result must contain only finite values"):
        explainer.explain()
