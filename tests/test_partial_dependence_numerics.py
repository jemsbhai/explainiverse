"""Exact mean oracles for finite partial-dependence predictions."""

from fractions import Fraction

import numpy as np
import pytest

from explainiverse.explainers import PartialDependenceExplainer


class _ReferenceOutputRegressor:
    task = "regression"

    def __init__(self, outputs):
        self.outputs = np.asarray(outputs)

    def predict(self, data):
        # The reference-row identity is not one of the features being varied.
        return self.outputs[data[:, 2].astype(int)]


_MAX_FLOAT = np.finfo(float).max


@pytest.mark.parametrize("feature", ["left", ("left", "right")], ids=["1d", "interaction"])
@pytest.mark.parametrize(
    "outputs",
    [
        [_MAX_FLOAT, _MAX_FLOAT],
        [-_MAX_FLOAT, -_MAX_FLOAT],
        [_MAX_FLOAT, _MAX_FLOAT, -_MAX_FLOAT],
        [_MAX_FLOAT, 1.0, -_MAX_FLOAT],
        [_MAX_FLOAT, _MAX_FLOAT, 1.0, -_MAX_FLOAT, -_MAX_FLOAT],
    ],
    ids=["positive-limit", "negative-limit", "overflowing-sum", "residual", "overflow-residual"],
)
def test_partial_dependence_preserves_representable_means_of_finite_predictions(feature, outputs):
    row_ids = np.arange(len(outputs))
    reference = np.column_stack((row_ids % 2, (row_ids + 1) % 2, row_ids))
    expected = float(
        sum((Fraction.from_float(value) for value in outputs), Fraction()) / len(outputs)
    )
    key = "left" if isinstance(feature, str) else "left_x_right"

    for rows in (reference, reference[::-1]):
        explainer = PartialDependenceExplainer(
            _ReferenceOutputRegressor(outputs), rows, ["left", "right", "row_id"]
        )
        with np.errstate(over="raise", invalid="raise"):
            explanation = explainer.explain([feature])
        observed = np.asarray(explanation.explanation_data["pdp_values"][key])
        assert np.all(np.isfinite(observed))
        np.testing.assert_array_equal(observed, np.full(observed.shape, expected))
