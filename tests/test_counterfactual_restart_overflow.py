"""Gaussian restarts project finite outward steps near the limits of binary64."""

import sys
import warnings
from fractions import Fraction

import numpy as np
import pytest

from explainiverse.explainers import CounterfactualExplainer

_MAX = sys.float_info.max


@pytest.mark.parametrize(
    "scale,seed,policy",
    [
        pytest.param(_MAX, 3, "raise", id="positive-overflow-strict"),
        pytest.param(-_MAX, 8, "raise", id="negative-overflow-strict"),
        pytest.param(_MAX, 3, "warn", id="positive-overflow-warning"),
        pytest.param(-_MAX, 8, "warn", id="negative-overflow-warning"),
        pytest.param(1.0, 3, "raise", id="ordinary-outward-step"),
    ],
)
def test_gaussian_restart_returns_target_at_clipped_endpoint(scale, seed, policy):
    query, exemplar, endpoint = 0.5 * scale, 0.945 * scale, 0.99 * scale
    lower, upper = sorted((query, endpoint))
    rng = np.random.default_rng(seed)
    rng.integers(1)  # choose the only training target
    offset = Fraction(rng.standard_normal()) * Fraction(0.15) * (Fraction(upper) - Fraction(lower))
    proposed = Fraction(exemplar) + offset
    assert abs(offset) < _MAX
    assert (abs(proposed) > _MAX) == (abs(scale) == _MAX)
    assert min(upper, max(lower, proposed)) == endpoint

    class IsolatedTargets:
        prediction_output_kind = "probabilities"

        def predict(self, rows):
            x = rows[:, 0]
            target = (x == exemplar) | (x == endpoint)
            # Keep the target probability constant so the optimizer's objective is flat.
            return np.column_stack(
                (
                    np.where(target, 5 / 16, 1 / 2),
                    np.full(len(x), 3 / 8),
                    np.where(target, 5 / 16, 1 / 8),
                )
            )

    explainer = CounterfactualExplainer(
        IsolatedTargets(),
        [[query], [exemplar]],
        ["x"],
        feature_ranges={"x": (lower, upper)},
        random_state=seed,
        proximity_weight=0.0,
        diversity_weight=0.0,
    )
    with warnings.catch_warnings(), np.errstate(over=policy):
        warnings.simplefilter("error", RuntimeWarning)
        rows = explainer.explain(
            [query], desired_class=1, num_counterfactuals=2, max_attempts=2
        ).explanation_data["counterfactuals"]
    assert {x for (x,) in rows} == {exemplar, endpoint}
