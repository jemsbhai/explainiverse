"""Analytical SAGE coalition contracts for mixed input/background dtypes."""

from fractions import Fraction
from itertools import permutations

import numpy as np
import pytest

from explainiverse.explainers.global_explainers.sage import SAGEExplainer


class _LinearRegressor:
    task = "regression"
    prediction_output_kind = "regression_values"

    def __init__(self):
        self.seen = []

    def predict(self, values):
        self.seen.append(values.copy())
        numeric = np.asarray(values, dtype=np.float64)
        return (numeric[:, 0] + 2.0 * numeric[:, 1]).reshape(-1, 1)


_FRACTIONAL = [[0.5, 1.5], [2.5, 3.5]]
_PRECISE = [[0.5 + 2**-25, 1.5 + 2**-24], [2.5 + 2**-23, 3.5 + 2**-23]]
_CASES = [
    pytest.param(np.int64, np.float64, _FRACTIONAL, id="int64-float64-fractional"),
    pytest.param(np.int32, np.float32, _FRACTIONAL, id="int32-float32-fractional"),
    pytest.param(np.uint8, np.float64, _FRACTIONAL, id="uint8-float64-fractional"),
    pytest.param(np.float32, np.float64, _PRECISE, id="float32-float64-precision"),
    pytest.param(np.float32, np.float64, _FRACTIONAL, id="float32-exact-control"),
    pytest.param(np.float64, np.float64, _FRACTIONAL, id="float64-control"),
    pytest.param(np.float64, np.float32, _FRACTIONAL, id="float64-float32-control"),
    pytest.param(np.int8, np.int16, [[-130, 128], [-132, 130]], id="int8-int16-range"),
]


def _fraction(value):
    return Fraction.from_float(float(value))


def _expected_predictions(values, background, mask):
    predictions = []
    for row in values:
        completed_predictions = [
            sum(
                (index + 1) * _fraction(row[index] if selected else donor[index])
                for index, selected in enumerate(mask)
            )
            for donor in background
        ]
        predictions.append(sum(completed_predictions) / len(completed_predictions))
    return predictions


def _expected_game(values, background):
    targets = [_fraction(row[0]) + 2 * _fraction(row[1]) for row in values]
    losses = {}
    for mask in ((False, False), (True, False), (False, True), (True, True)):
        predictions = _expected_predictions(values, background, mask)
        losses[mask] = sum(
            (target - pred) ** 2 for target, pred in zip(targets, predictions)
        ) / len(targets)
    marginal_sums = [Fraction(0), Fraction(0)]
    orders = list(permutations(range(2)))
    for order in orders:
        mask = [False, False]
        before = losses[tuple(mask)]
        for index in order:
            mask[index] = True
            after = losses[tuple(mask)]
            marginal_sums[index] += before - after
            before = after
    return losses, [float(value / len(orders)) for value in marginal_sums]


def _explainer(value_dtype, background_dtype, background_values):
    values = np.array([[1, 2], [3, 4]], dtype=value_dtype)
    background = np.array(background_values, dtype=background_dtype)
    model = _LinearRegressor()
    explainer = SAGEExplainer(
        model,
        values,
        np.array([5.0, 11.0]),
        ["x0", "x1"],
        task="regression",
        background_data=background,
        n_permutations=2,
        random_state=0,
    )
    return explainer, values, background, model


@pytest.mark.parametrize("value_dtype,background_dtype,background_values", _CASES)
def test_exact_two_order_game_preserves_reference_and_background_values(
    value_dtype, background_dtype, background_values
):
    explainer, values, background, _ = _explainer(value_dtype, background_dtype, background_values)
    before_values, before_background = values.copy(), background.copy()
    losses, expected = _expected_game(values, background)
    result = explainer.explain().explanation_data
    # RandomState(0) visits [1,0] and [0,1] exactly once for two features.
    np.testing.assert_allclose(
        list(result["feature_attributions"].values()), expected, rtol=2e-15, atol=0.0
    )
    assert result["baseline_loss"] == pytest.approx(float(losses[(False, False)]), rel=2e-15)
    assert result["full_loss"] == 0.0
    assert result["efficiency_error"] == pytest.approx(0.0, abs=1e-11)
    np.testing.assert_array_equal(values, before_values)
    np.testing.assert_array_equal(background, before_background)
    assert values.dtype == before_values.dtype
    assert background.dtype == before_background.dtype


class _IntegerParityRegressor:
    task = "regression"
    prediction_output_kind = "regression_values"

    def __init__(self, column):
        self.column = column
        self.seen = []

    def predict(self, values):
        self.seen.append(values.copy())
        # Converting directly to Python int preserves every integer-valued
        # longdouble; conversion through float would spoil the oracle itself.
        return np.array([int(row[self.column]) % 2 for row in values], dtype=float)


_WIDE_CASES = [
    pytest.param(np.int64, np.float64, "retained", 2**53 + 1, id="retained-int64-float64"),
    pytest.param(np.uint64, np.float64, "retained", 2**63 + 1, id="retained-uint64-float64"),
    pytest.param(np.float64, np.int64, "background", 2**53 + 1, id="background-int64-float64"),
    pytest.param(np.float64, np.uint64, "background", 2**63 + 1, id="background-uint64-float64"),
    pytest.param(np.int64, np.uint64, "retained", 2**53 + 1, id="retained-signed-unsigned"),
    pytest.param(np.int64, np.uint64, "background", 2**53 + 1, id="background-signed-unsigned"),
    pytest.param(np.int64, np.longdouble, "retained", 2**53 + 1, id="retained-longdouble"),
    pytest.param(np.longdouble, np.uint64, "background", 2**63 + 1, id="background-longdouble"),
]


@pytest.mark.parametrize("value_dtype,background_dtype,wide_location,wide_value", _WIDE_CASES)
def test_partial_coalition_preserves_wide_integer_values_exactly(
    value_dtype, background_dtype, wide_location, wide_value
):
    values = np.array(
        [[wide_value, 0], [wide_value + 2, 0]] if wide_location == "retained" else [[0, 1], [2, 1]],
        dtype=value_dtype,
    )
    background = np.array(
        (
            [[0, wide_value], [0, wide_value + 2]]
            if wide_location == "background"
            else [[0, 0], [0, 2]]
        ),
        dtype=background_dtype,
    )
    before_values, before_background = values.copy(), background.copy()
    model = _IntegerParityRegressor(0 if wide_location == "retained" else 1)
    explainer = SAGEExplainer(
        model,
        values,
        np.ones(len(values)),
        ["x0", "x1"],
        task="regression",
        background_data=background,
        n_permutations=1,
    )

    actual = explainer._predict_restricted(np.array([True, False]))

    completed = model.seen[-1]
    for row_index, row in enumerate(values):
        for donor_index, donor in enumerate(background):
            mixed = completed[row_index * len(background) + donor_index]
            # Object-array equality is insufficient on platforms where
            # astype(object) retains narrow np.longdouble scalars: comparison
            # may coerce the Python integer and falsely report exact equality.
            assert int(mixed[0]) == int(row[0])
            assert int(mixed[1]) == int(donor[1])
    np.testing.assert_array_equal(actual, np.ones(len(values)))
    np.testing.assert_array_equal(values, before_values)
    np.testing.assert_array_equal(background, before_background)
    assert values.dtype == before_values.dtype
    assert background.dtype == before_background.dtype


class _CategoricalRegressor:
    task = "regression"
    prediction_output_kind = "regression_values"

    def __init__(self):
        self.seen = []

    def predict(self, values):
        self.seen.append(values.copy())
        category_scores = {"longcategory": 1.0, "another": 3.0}
        return np.array([category_scores.get(row[1], -100.0) for row in values])


@pytest.mark.parametrize("mixed_object", [False, True], ids=["wider-unicode", "mixed-object"])
def test_partial_coalition_preserves_categorical_background_values(mixed_object):
    values = (
        np.array([[1, "b"], [3, "d"]], dtype=object)
        if mixed_object
        else np.array([["a", "b"], ["c", "d"]], dtype="U1")
    )
    background = np.array(
        [["unused", "longcategory"], ["unused", "another"]],
        dtype=object if mixed_object else "U12",
    )
    before_values, before_background = values.copy(), background.copy()
    model = _CategoricalRegressor()
    explainer = SAGEExplainer(
        model,
        values,
        np.zeros(len(values)),
        ["x0", "x1"],
        task="regression",
        background_data=background,
        n_permutations=1,
    )

    actual = explainer._predict_restricted(np.array([True, False]))

    for row_index, row in enumerate(values):
        for donor_index, donor in enumerate(background):
            mixed = model.seen[-1][row_index * len(background) + donor_index]
            assert mixed[0] == row[0]
            assert mixed[1] == donor[1]
            if mixed_object:
                assert isinstance(mixed[0], (int, np.integer))
    np.testing.assert_array_equal(actual, np.full(len(values), 2.0))
    np.testing.assert_array_equal(values, before_values)
    np.testing.assert_array_equal(background, before_background)


@pytest.mark.parametrize("value_dtype,background_dtype,background_values", _CASES)
@pytest.mark.parametrize("mask", [(False, False), (True, False), (False, True), (True, True)])
def test_coalition_predictions_match_exact_background_expectation(
    value_dtype, background_dtype, background_values, mask
):
    explainer, values, background, model = _explainer(
        value_dtype, background_dtype, background_values
    )
    before_values, before_background = values.copy(), background.copy()
    expected = np.array([float(value) for value in _expected_predictions(values, background, mask)])
    actual = explainer._predict_restricted(np.array(mask))
    np.testing.assert_allclose(actual[:, 0], expected, rtol=2e-15, atol=0.0)
    if any(mask) and not all(mask):
        completed = model.seen[-1]
        for row_index, row in enumerate(values):
            for donor_index, donor in enumerate(background):
                for index, selected in enumerate(mask):
                    assert completed[row_index * len(background) + donor_index, index] == (
                        row[index] if selected else donor[index]
                    )
    np.testing.assert_array_equal(values, before_values)
    np.testing.assert_array_equal(background, before_background)
    assert values.dtype == before_values.dtype
    assert background.dtype == before_background.dtype
