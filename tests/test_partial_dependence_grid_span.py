"""PDP interventions remain finite across the range of binary64."""

from fractions import Fraction

import numpy as np
import pytest

from explainiverse.explainers import PartialDependenceExplainer

_UNIT = float.fromhex("0x1p+1023")


class SignRegressor:
    task = "regression"

    def predict(self, rows):
        return (rows[:, 0] > 0.0).astype(float)


@pytest.mark.parametrize("selection", ["x", ("z", "x")])
@pytest.mark.parametrize(
    "levels,bounds,resolution",
    [
        ([-2.0, -1.0, 0.0, 1.0, 2.0], (0, 100), 3),
        ([-1.5 * _UNIT, -_UNIT, 0.0, _UNIT, 1.5 * _UNIT], (0, 100), 4),
        ([-1.75 * _UNIT, -1.5 * _UNIT, 1.5 * _UNIT, 1.75 * _UNIT], (43.75, 56.25), 3),
        ([-1.75 * _UNIT, -1.5 * _UNIT, 1.5 * _UNIT, 1.75 * _UNIT], (37, 92), 3),
    ],
    ids=["ordinary", "grid-span", "percentile-gap", "asymmetric-percentiles"],
)
def test_grid_coordinates_and_responses_match_exact_interpolation(
    selection, levels, bounds, resolution
):
    ordered = list(map(Fraction.from_float, levels))
    endpoints = []
    for percentage in bounds:
        rank = Fraction(percentage) * (len(levels) - 1) / 100
        index = int(rank)
        left = ordered[index]
        right = ordered[min(index + 1, len(ordered) - 1)]
        endpoints.append(left + (rank - index) * (right - left))
    lower, upper = endpoints
    grid = [float(lower + (upper - lower) * i / (resolution - 1)) for i in range(resolution)]
    responses = [float(x > 0) for x in grid]

    rows = np.column_stack((levels, np.arange(len(levels)) % 2))
    with np.errstate(over="raise", invalid="raise"):
        data = (
            PartialDependenceExplainer(
                SignRegressor(),
                rows,
                ["x", "z"],
                grid_resolution=resolution,
                percentile_range=bounds,
            )
            .explain([selection])
            .explanation_data
        )
    if selection == "x":
        assert data["grid_values"]["x"] == grid
        assert data["pdp_values"]["x"] == responses
    else:
        assert data["grid_values"]["z_x_x"] == {"grid1": [0, 1], "grid2": grid}
        assert data["pdp_values"]["z_x_x"] == [responses, responses]
