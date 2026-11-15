"""Gaussian restarts can reach distinct target regions across narrow feature ranges."""

from fractions import Fraction

import numpy as np
import pytest

from explainiverse.explainers import CounterfactualExplainer

_TINY = float.fromhex("0x0.0000000000001p-1022")


@pytest.mark.parametrize(
    "unit,width,start,seed",
    [
        pytest.param(_TINY, 3, 1, 3, id="deviation-rounds-to-zero"),
        pytest.param(_TINY, 4, 2, 8, id="negative-draw-with-subnormal-deviation"),
        pytest.param(_TINY, 10, 7, 3, id="draw-clipped-at-upper-bound"),
        pytest.param(0.125, 3, 1, 3, id="ordinary-control"),
    ],
)
def test_gaussian_restart_finds_target_at_projected_draw(unit, width, start, seed):
    query = [0.0, 0.0]
    exemplar = [start * unit, 0.75]
    upper = width * unit
    rng = np.random.default_rng(seed)
    rng.integers(1)  # choose the sole target exemplar
    z_x, z_y = rng.standard_normal(2)
    proposed = Fraction(exemplar[0]) + Fraction(0.15) * Fraction(upper) * Fraction(z_x)
    expected_x = float(min(Fraction(upper), max(0, proposed)))
    expected_y = float(Fraction(exemplar[1]) + Fraction(0.15) * Fraction(z_y))

    def in_second_region(x, y):
        return (abs(x / unit - expected_x / unit) < 0.04) & (abs(y - expected_y) < 0.002)

    class TwoTargetRegions:
        prediction_output_kind = "probabilities"

        def predict(self, rows):
            x, y = rows.T
            target = ((x == exemplar[0]) & (y == exemplar[1])) | in_second_region(x, y)
            # The target probability is constant so optimization cannot move the restart.
            return np.column_stack(
                (
                    np.where(target, 5 / 16, 1 / 2),
                    np.full(len(x), 3 / 8),
                    np.where(target, 5 / 16, 1 / 8),
                )
            )

    explainer = CounterfactualExplainer(
        TwoTargetRegions(),
        [query, exemplar],
        ["x", "y"],
        feature_ranges={"x": (0.0, upper), "y": (0.0, 1.0)},
        random_state=seed,
        proximity_weight=0.0,
        diversity_weight=0.0,
    )
    rows = explainer.explain(
        query, desired_class=1, num_counterfactuals=2, max_attempts=2
    ).explanation_data["counterfactuals"]
    assert exemplar in rows
    assert any(in_second_region(x, y) for x, y in rows)
