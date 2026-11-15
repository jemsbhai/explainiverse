"""Default permutation R² scores the variation present in finite floating-point data."""

from fractions import Fraction
from statistics import mean, pstdev

import numpy as np
import pytest

from explainiverse.explainers import PermutationImportanceExplainer


def _exact_r2(targets, predictions):
    scores = []
    for truth, response in zip(targets.reshape(4, -1).T, predictions.reshape(4, -1).T):
        ys = [Fraction(float(value)) for value in truth]
        residual = sum((y - Fraction(float(p))) ** 2 for y, p in zip(ys, response))
        center = mean(ys)
        variation = sum((y - center) ** 2 for y in ys)
        scores.append(1 - residual / variation if variation else int(residual == 0))
    return mean(scores)


_TARGET = np.array([0.0, 1.0, 2.0, 3.0])
_RESPONSE = np.array([0.0, 1.0, 3.0, 2.0])
_NEXT = np.nextafter(1.0, 2.0)
_TINY = np.finfo(np.float64).smallest_subnormal
_HUGE = np.finfo(np.float64).max


@pytest.mark.parametrize(
    "targets,predictions",
    [
        pytest.param(_TARGET, _RESPONSE, id="ordinary"),
        pytest.param(_TARGET * 2.0**600, _RESPONSE * 2.0**600, id="overflowing-squares"),
        pytest.param(_TARGET * 2.0**-600, _RESPONSE * 2.0**-600, id="underflowing-squares"),
        pytest.param(_TARGET * _TINY, _RESPONSE * _TINY, id="subnormal-responses"),
        pytest.param(
            np.array([1.0, 1.0, _NEXT, _NEXT]),
            np.array([1.0, _NEXT, 1.0, _NEXT]),
            id="unrepresentable-target-mean",
        ),
        pytest.param(
            np.array([-1.0, 1.0, -1.0, 1.0]),
            np.array([-1.0, 1.0, -1.0, 1.0]) * 2.0**-60,
            id="almost-equal-residual-and-variation",
        ),
        pytest.param(
            np.array([0.1, 0.2, -0.1, -0.2]),
            np.zeros(4),
            id="exactly-equal-residual-and-variation",
        ),
        pytest.param(
            np.array([[-1.0, -1.0], [1.0, 1.0]] * 2),
            np.array([[-1.0, 1.0], [1.0, np.nextafter(1.0, 0.0)]] * 2),
            id="almost-cancelling-output-scores",
        ),
        pytest.param(
            (_TARGET * 2.0**70).astype(np.float32),
            (_RESPONSE * 2.0**70).astype(np.float32),
            id="float32-squares",
        ),
        pytest.param(
            (_TARGET * 2.0**-13).astype(np.float16),
            (_RESPONSE * 2.0**-13).astype(np.float16),
            id="float16-squares",
        ),
        pytest.param(
            np.full(4, _HUGE),
            np.full((4, 1), _HUGE),
            id="constant-perfect",
        ),
        pytest.param(
            np.zeros((4, 1)),
            np.array([0.0, 0.0, _TINY, 0.0]),
            id="constant-one-subnormal-residual",
        ),
        pytest.param(
            np.column_stack((_TARGET * 2.0**600, _TARGET * _TINY, np.ones(4))),
            np.column_stack((_RESPONSE * 2.0**600, _TARGET[::-1] * _TINY, np.ones(4))),
            id="uniform-average-of-mixed-scale-outputs",
        ),
    ],
)
def test_default_float_r2_matches_exact_scores(targets, predictions):
    class LookupRegressor:
        task = "regression"

        def predict(self, rows):
            return predictions[rows[:, 0]]

    baseline = float(_exact_r2(targets, predictions))
    # The three successive permutations for a single feature and seed 3.
    orders = ([1, 2, 3, 0], [2, 3, 1, 0], [3, 2, 1, 0])
    drops = [baseline - float(_exact_r2(targets, predictions[order])) for order in orders]

    with np.errstate(over="raise", under="raise", invalid="raise"):
        data = (
            PermutationImportanceExplainer(
                LookupRegressor(),
                np.arange(4)[:, None],
                targets,
                ["signal"],
                n_repeats=3,
                random_state=3,
            )
            .explain()
            .explanation_data
        )

    assert data["baseline_score"] == pytest.approx(baseline, rel=1e-15, abs=0)
    assert data["importances"]["signal"] == pytest.approx(drops, rel=1e-15, abs=0)
    assert data["feature_attributions"]["signal"] == pytest.approx(mean(drops), rel=1e-15, abs=0)
    assert data["std"]["signal"] == pytest.approx(pstdev(drops), rel=2e-15, abs=0)
