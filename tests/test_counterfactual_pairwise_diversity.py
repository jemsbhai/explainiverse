"""Counterfactual diversity averages squared distances across all unordered pairs."""

from fractions import Fraction
from itertools import combinations

import numpy as np
import pytest

from explainiverse.explainers import CounterfactualExplainer

_STEP = 2.0**-538
_TRIANGLE = [(0, 0), (1, 1), (2, 0)]


@pytest.mark.parametrize(
    "step,coordinates",
    [
        pytest.param(_STEP, _TRIANGLE, id="two-half-subnormal-pairs"),
        pytest.param(1.5 * _STEP, [(0, 0), (1, 0), (8, 0)], id="pairs-round-up"),
        pytest.param(_STEP, [(0, 0), (0, 1), (0, 2), (1, 0)], id="four-targets"),
        pytest.param(_STEP, [(0, 0), (0, 2), (1, 0), (3, 0)], id="halfway-to-even"),
        pytest.param(_STEP, [(0, 0), (0, 1), (0, 2)], id="halfway-to-zero"),
        pytest.param(_STEP, [(0, 0), (2, 0)], id="single-pair"),
        pytest.param(0.125, _TRIANGLE, id="ordinary-triangle"),
    ],
)
def test_diversity_rounds_the_mean_over_target_pairs(step, coordinates):
    query = [0.0, 1.0]
    targets = [(x * step, y * step) for x, y in coordinates]

    class TargetClassifier:
        prediction_output_kind = "probabilities"

        def predict(self, rows):
            target = rows[:, 1] < 1.0
            return np.column_stack((~target, target)).astype(float)

    explainer = CounterfactualExplainer(
        TargetClassifier(),
        [query, *targets],
        ["x", "y"],
        categorical_features=["x", "y"],
        feature_ranges={"x": (0.0, 1.0), "y": (0.0, 1.0)},
    )
    data = explainer.explain(
        query, desired_class=1, num_counterfactuals=len(targets), max_attempts=len(targets)
    ).explanation_data
    assert {tuple(row) for row in data["counterfactuals"]} == set(targets)

    # Both spans are one, so the coordinate differences are already normalized.
    distances = [
        sum((Fraction(a) - Fraction(b)) ** 2 for a, b in zip(left, right))
        for left, right in combinations(targets, 2)
    ]
    expected = -float(sum(distances) / len(distances))
    assert data["diversity_loss"] == expected
