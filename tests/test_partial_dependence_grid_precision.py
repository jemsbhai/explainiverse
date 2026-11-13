"""PDP must apply its reported grid without rounding observed integer values."""

from enum import IntEnum
from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from explainiverse.explainers import PartialDependenceExplainer

_WIDE = 2**53
_UINT_MAX = 2**64 - 1


def _exact(value):
    return Fraction(value.item() if isinstance(value, np.generic) else value)


class CenteredLinearModel:
    task = "regression"

    def __init__(self, origin):
        self.origin = _exact(origin)
        self.calls = []

    def predict(self, matrix):
        self.calls.append(matrix.tolist())
        return np.array([float(_exact(x) - self.origin + 2 * _exact(z)) for x, z in matrix])


def _assert_pdp(dtype, levels, reference_z, z_grid, resolution, percentiles, selection):
    a, b = levels
    rows = list(product(levels, reference_z))
    inputs = np.array(rows, dtype=dtype)
    before = inputs.copy()
    model = CenteredLinearModel(a)
    result = (
        PartialDependenceExplainer(
            model, inputs, ["x", "z"], grid_resolution=resolution, percentile_range=percentiles
        )
        .explain([selection])
        .explanation_data
    )
    np.testing.assert_array_equal(inputs, before)
    names = selection if isinstance(selection, tuple) else (selection,)
    key = "_x_".join(names)
    axes = [list(levels) if name == "x" else list(z_grid) for name in names]
    expected_grid = axes[0] if len(axes) == 1 else {"grid1": axes[0], "grid2": axes[1]}
    assert result["grid_values"][key] == expected_grid
    assert result["grid_types"][key] == ("numeric" if len(axes) == 1 else ["numeric", "numeric"])

    expected_values = []
    expected_inputs = []
    for coords in product(*axes):
        interventions = dict(zip(names, coords))
        grid_rows = [[interventions.get("x", x), interventions.get("z", z)] for x, z in rows]
        expected_inputs.append(grid_rows)
        responses = [_exact(x) - _exact(a) + 2 * _exact(z) for x, z in grid_rows]
        expected_values.append(float(sum(responses) / len(rows)))
    expected_shape = tuple(len(axis) for axis in axes)
    np.testing.assert_array_equal(
        result["pdp_values"][key], np.array(expected_values).reshape(expected_shape)
    )
    # Python scalar equality preserves the distinction between a large integer
    # and a float rounded to its neighbor. The first model call selects an output.
    assert model.calls[1:] == expected_inputs


@pytest.mark.parametrize("selection", ["x", "z", ("x", "z"), ("z", "x")])
@pytest.mark.parametrize(
    "dtype,levels",
    [
        (np.int64, (7, 8)),
        (np.uint64, (7, 8)),
        (np.int64, (_WIDE, _WIDE + 1)),
        (np.uint64, (_WIDE, _WIDE + 1)),
        (np.int64, (_WIDE + 1, _WIDE + 3)),
        (np.uint64, (_WIDE + 1, _WIDE + 3)),
        (np.uint64, (_UINT_MAX - 1, _UINT_MAX)),
        (object, (_WIDE, _WIDE + 1)),
        (object, (float(_WIDE), _WIDE + 1)),
    ],
)
def test_observed_integer_levels_reach_public_result_and_model(dtype, levels, selection):
    _assert_pdp(dtype, levels, [0, 1], [0, 1], 2, (5, 95), selection)


@pytest.mark.parametrize("selection", ["z", ("x", "z"), ("z", "x")])
@pytest.mark.parametrize(
    "dtype,levels",
    [
        (np.int64, (7, 8)),
        (np.int64, (_WIDE, _WIDE + 1)),
        (np.uint64, (_UINT_MAX - 1, _UINT_MAX)),
        (object, (float(_WIDE), _WIDE + 1)),
    ],
)
def test_fractional_interventions_preserve_wide_coordinates(dtype, levels, selection):
    # Quartiles of [0,0,1,1,2,2,3,3] are 0.75 and 2.25. The grid has three points.
    _assert_pdp(dtype, levels, range(4), [0.75, 1.5, 2.25], 3, (25, 75), selection)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_interpolated_grid_reaches_model_at_published_precision(dtype):
    model_coordinates = []

    class IdentityRegressor:
        task = "regression"

        def predict(self, rows):
            x = rows[:, 0]
            if len(x) == len(X):
                assert np.all(x == x[0])
                model_coordinates.append(float(x[0]))
            return x

    X = np.arange(11, dtype=dtype).reshape(-1, 1)
    data = (
        PartialDependenceExplainer(
            IdentityRegressor(), X, ["x"], grid_resolution=4, percentile_range=(1, 95)
        )
        .explain(["x"])
        .explanation_data
    )

    grid = data["grid_values"]["x"]
    np.testing.assert_array_equal(grid, np.linspace(0.1, 9.5, 4))
    np.testing.assert_array_equal([model_coordinates, data["pdp_values"]["x"]], [grid, grid])


class _Tier(IntEnum):
    FIRST = 1
    SECOND = 2


@pytest.mark.parametrize(
    "rows",
    [
        np.array([[1], [2]], dtype=np.longdouble),
        np.array([[_Tier.FIRST], [_Tier.SECOND]], dtype=object),
    ],
)
def test_grid_exports_plain_numeric_values(rows):
    class Model:
        task = "regression"

        def predict(self, data):
            return np.asarray(data[:, 0], dtype=float)

    grid = (
        PartialDependenceExplainer(Model(), rows, ["x"])
        .explain(["x"])
        .explanation_data["grid_values"]["x"]
    )
    assert grid == [1, 2]
    assert all(type(value) in (int, float) for value in grid)
