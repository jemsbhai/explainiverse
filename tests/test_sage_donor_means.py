"""SAGE averages donor predictions for each observed row and model output."""

from decimal import Decimal

import numpy as np
import pytest

from explainiverse.explainers import SAGEExplainer

_LARGE = 2.0**1023
_DONORS = [
    pytest.param([0.0, 1.0, 2.0], 1.0, id="ordinary"),
    pytest.param([_LARGE] * 3, _LARGE, id="sum-overflows"),
    pytest.param([_LARGE, 3.0, -_LARGE], 1.0, id="cancellation"),
    pytest.param([2**62 + 3, -(2**62 + 1), 1], 1.0, id="wide-int64"),
    pytest.param([_LARGE, 3.0 * 2.0**-1074, -_LARGE], 2.0**-1074, id="subnormal-residual"),
]


@pytest.mark.parametrize(
    ("donors", "expected"),
    [
        *_DONORS,
        pytest.param(np.array([0.0, 1.0, 2.0], dtype=object), 1.0, id="object-ordinary"),
        pytest.param(np.array([_LARGE, 3.0, -_LARGE], dtype=object), 1.0, id="object-floats"),
        pytest.param(
            np.array([2**62 + 3, float(-(2**62)), 0], dtype=object), 1.0, id="object-int-float"
        ),
    ],
)
def test_sage_empty_coalition_repeats_empirical_donor_mean(donors, expected):
    donors = np.asarray(donors)
    predictions = np.concatenate((donors, np.zeros(1, dtype=donors.dtype)))

    class Lookup:
        def predict(self, rows):
            return predictions[rows[:, 0]]

    seen = []

    def loss(targets, predicted):
        seen.append(predicted.copy())
        return predicted[0]

    data = (
        SAGEExplainer(
            Lookup(),
            np.array([[3], [3]]),
            np.zeros(2),
            ["donor"],
            task="regression",
            background_data=np.array([[0], [1], [2]]),
            loss_fn=loss,
            n_permutations=1,
        )
        .explain()
        .explanation_data
    )

    if donors.dtype == object:
        assert seen[0].dtype == object
    np.testing.assert_array_equal(seen[0], [expected, expected])
    assert data["baseline_loss"] == expected
    assert data["full_loss"] == 0.0
    assert data["feature_attributions"] == {"donor": expected}


@pytest.mark.parametrize(("donors", "expected"), _DONORS)
def test_sage_partial_coalition_averages_each_row_and_output(donors, expected):
    table = np.array([*donors, 0])

    class Lookup:
        def predict(self, rows):
            observed, donor = rows.T
            scores = observed * table[donor]
            return np.column_stack((scores, -scores))

    seen = []

    def loss(targets, predicted):
        seen.append(predicted.copy())
        return predicted[0, 0]

    data = (
        SAGEExplainer(
            Lookup(),
            np.array([[1, 3], [-1, 3]]),
            np.zeros(2),
            ["observed", "donor"],
            task="regression",
            background_data=np.array([[0, 0], [0, 1], [0, 2]]),
            loss_fn=loss,
            n_permutations=1,
            random_state=1,
        )
        .explain()
        .explanation_data
    )

    # Seed 1 selects the observed feature first. This reveals its sign
    # while retaining the empirical donor distribution in column two.
    partial = np.array([[expected, -expected], [-expected, expected]])
    assert any(np.array_equal(predicted, partial) for predicted in seen)
    assert data["baseline_loss"] == data["full_loss"] == 0.0
    assert data["feature_attributions"] == {"observed": -expected, "donor": expected}


def _tensor_output_game(donors):
    table = np.zeros((len(donors) + 1, 1, 1), dtype=donors.dtype)
    table[:-1, 0, 0] = donors

    class Lookup:
        def predict(self, rows):
            return table[rows[:, 0]]

    seen = []

    def loss(targets, predicted):
        seen.append(predicted.copy())
        return float(abs(predicted[0, 0, 0]))

    data = (
        SAGEExplainer(
            Lookup(),
            np.full((2, 1), len(donors)),
            np.zeros(2),
            ["x"],
            task="regression",
            background_data=np.arange(len(donors))[:, None],
            loss_fn=loss,
            n_permutations=1,
        )
        .explain()
        .explanation_data
    )
    return seen[0], data


@pytest.mark.parametrize(
    ("donors", "expected"),
    [
        pytest.param(
            np.array([4, 2.0**-22, 2.0**-100, 0], dtype=np.float32),
            1.0 + 2.0**-23,
            id="above-float32-midpoint",
        ),
        pytest.param(
            np.array([4, 2.0**-9, 2.0**-24, 0], dtype=np.float16),
            1.0 + 2.0**-10,
            id="above-float16-midpoint",
        ),
        pytest.param(np.array([1 + 4j, 3 + 2j]), 2 + 3j, id="complex"),
        pytest.param(
            np.array([complex(_LARGE, _LARGE), complex(_LARGE, 3), complex(-_LARGE, -_LARGE)]),
            complex(_LARGE / 3, 1),
            id="complex-real-overflow-imaginary-cancellation",
        ),
        pytest.param(
            np.array([complex(_LARGE, -_LARGE), complex(3, -_LARGE), complex(-_LARGE, _LARGE)]),
            complex(1, -_LARGE / 3),
            id="complex-imaginary-overflow-real-cancellation",
        ),
        pytest.param(np.array([Decimal("0.10"), Decimal("0.20")]), Decimal("0.15"), id="decimal"),
    ],
)
def test_sage_custom_loss_observes_native_donor_mean(donors, expected):
    # In each narrow-float case, dividing the first two donors' sum by
    # four gives the midpoint above 1; the third donor breaks the tie.
    observed, data = _tensor_output_game(donors)
    assert observed.dtype == donors.dtype
    np.testing.assert_array_equal(observed, np.full((2, 1, 1), expected, dtype=donors.dtype))
    assert data["baseline_loss"] == float(abs(expected))
    assert data["full_loss"] == 0.0


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_sage_detects_unrepresentable_nonzero_donor_mean(dtype):
    # The mean of zero and the least positive subnormal rounds to zero.
    donors = np.array([0, np.nextafter(dtype(0), dtype(1))], dtype=dtype)
    with pytest.raises(FloatingPointError):
        _tensor_output_game(donors)
