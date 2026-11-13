"""ALE finite differences preserve the units of integer model predictions."""

from fractions import Fraction

import numpy as np
import pytest

from explainiverse.explainers import ALEExplainer


@pytest.mark.parametrize(
    "dtype, intercept, task, target",
    [
        (np.int64, 2**63 - 16, "regression", None),
        (np.int64, -(2**63) + 16, "regression", 1),
        (np.uint64, 2**64 - 16, "regression", None),
        (np.uint64, 2**64 - 16, "classification", 1),
        (np.int32, 10, "regression", None),
    ],
    ids=["signed", "negative-selected", "unsigned", "class-score", "small-integers"],
)
def test_integer_intercept_does_not_change_ale(dtype, intercept, task, target):
    # Each row has a fixed slope. Construct its predictions with Python integers
    # so the model itself is exact, including when the final dtype is unsigned.
    slopes = [1, 1, -1, 3]
    rows = np.column_stack(([0, 1, 1, 2], np.arange(4)))

    class Model:
        def predict(self, data):
            scores = np.array([intercept + slopes[i] * int(x) for x, i in data], dtype=dtype)
            if target is None:
                return scores
            return np.column_stack((np.zeros_like(scores), scores))

    data = (
        ALEExplainer(Model(), rows, ["x", "row"], n_bins=4, task=task)
        .explain("x", target_class=target)
        .explanation_data
    )

    # Slopes average to 1/3 and 3 in the two unit-width bins. The uncentered
    # midpoints 1/6 and 11/6 carry weights 3/4 and 1/4, giving a center of 7/12.
    assert data["bin_counts"] == [3, 1]
    assert data["local_effects"] == [float(Fraction(1, 3)), 3]
    assert data["ale_values"] == [float(Fraction(-7, 12)), -0.25, 2.75]
    assert data["ale_values_at_bin_centers"] == [float(Fraction(-5, 12)), 1.25]


@pytest.mark.parametrize(
    "dtype, lower, upper",
    [(np.int64, -(2**63), 2**63 - 1), (np.uint64, 2**64 - 1, 0)],
    ids=["signed-increase", "unsigned-decrease"],
)
def test_ale_integer_difference_can_exceed_the_model_dtype(dtype, lower, upper):
    responses = np.array([lower, upper], dtype=dtype)

    class Model:
        def predict(self, data):
            return responses[data[:, 0]]

    data = (
        ALEExplainer(Model(), np.array([[0], [1]]), ["x"], task="regression")
        .explain("x")
        .explanation_data
    )
    change = upper - lower
    assert data["local_effects"] == [float(change)]
    assert data["ale_values"] == [float(Fraction(-change, 2)), float(Fraction(change, 2))]
    assert data["ale_values_at_bin_centers"] == [0]


@pytest.mark.parametrize("integer_at_upper", [True, False], ids=["integer-upper", "integer-lower"])
def test_ale_preserves_each_prediction_dtype(integer_at_upper):
    rows = np.array([[0, 0], [1, 1]])
    integers = np.array([2**53 + 1, 1], dtype=np.int64)
    floats = np.array([2**53, 0.5], dtype=float)

    class Model:
        def predict(self, data):
            # All rows in a one-bin edge call share the same feature value.
            table = integers if data[0, 0] == int(integer_at_upper) else floats
            return table[data[:, 1]]

    data = (
        ALEExplainer(Model(), rows, ["x", "row"], n_bins=1, task="regression")
        .explain("x")
        .explanation_data
    )
    # Taking integer minus float gives row changes of 1 and 1/2.
    mean = 0.75 if integer_at_upper else -0.75
    assert data["local_effects"] == [mean]
    assert data["ale_values"] == [-mean / 2, mean / 2]
    assert data["ale_values_at_bin_centers"] == [0]
