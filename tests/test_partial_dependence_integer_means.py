"""PDP averages integer model responses before rounding the result to float."""

from fractions import Fraction

import numpy as np
import pytest

from explainiverse.explainers import PartialDependenceExplainer


@pytest.mark.parametrize("feature", ["left", ("left", "right")], ids=["1d", "interaction"])
@pytest.mark.parametrize(
    "task, dtype, responses, output_index",
    [
        ("regression", np.int64, [-(2**63), 2**63 - 1], None),
        ("regression", np.uint64, [2**64 - 1, 2**64 - 2049], None),
        ("regression", np.int64, [2**53 + 1, -(2**53), 0], 1),
        ("classification", np.int64, [2**53 + 1, -(2**53), 0], 1),
        ("regression", np.int32, [-7, 2, 3], None),
        ("regression", np.uint64, [2**63, 2**63], None),
    ],
    ids=[
        "signed-extrema",
        "unsigned-midpoint",
        "selected-regression-output",
        "selected-class-score",
        "small-integers",
        "exact-wide-integers",
    ],
)
def test_partial_dependence_averages_exact_integer_predictions(
    feature, task, dtype, responses, output_index
):
    selected = np.array(responses, dtype=dtype)
    outputs = (
        selected if output_index is None else np.column_stack((np.zeros_like(selected), selected))
    )

    class Model:
        def predict(self, rows):
            return outputs[rows[:, 2]]

    row_ids = np.arange(len(responses))
    reference = np.column_stack((row_ids % 2, (row_ids + 1) % 2, row_ids))
    expected = float(Fraction(sum(responses), len(responses)))
    key = "left" if isinstance(feature, str) else "left_x_right"
    shape = (2,) if isinstance(feature, str) else (2, 2)

    for rows in (reference, reference[::-1]):
        result = PartialDependenceExplainer(Model(), rows, ["left", "right", "row"], task=task)
        data = result.explain([feature], target_class=output_index).explanation_data
        np.testing.assert_array_equal(data["pdp_values"][key], np.full(shape, expected))
