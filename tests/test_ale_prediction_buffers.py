"""ALE finite differences preserve predictions across model calls."""

import numpy as np
import pytest

from explainiverse.explainers.global_explainers.ale import ALEExplainer


@pytest.mark.parametrize("reuse_buffer", [False, True], ids=["fresh", "reused"])
@pytest.mark.parametrize(
    ("task", "shape", "target", "sign"),
    [
        ("regression", "vector", None, 1),
        ("regression", "matrix", 1, -1),
        ("classification", "column", 1, 1),
        ("classification", "column", 0, -1),
    ],
)
def test_ale_preserves_upper_and_lower_predictions(task, shape, target, sign, reuse_buffer):
    inputs = np.array([[0.0], [1.0], [2.0], [4.0]])
    buffer = np.empty((len(inputs), 2))

    class Model:
        def predict(self, values):
            result = buffer[: len(values)]
            result[:, 0] = values[:, 0] / 4.0
            result[:, 1] = 1.0 - result[:, 0]
            if shape == "vector":
                result = result[:, 0]
            elif shape == "column":
                result = result[:, :1]
            return result if reuse_buffer else result.copy()

    data = (
        ALEExplainer(Model(), inputs, ["x"], n_bins=2, task=task)
        .explain("x", target_class=target)
        .explanation_data
    )

    # Two rows in each interval [0, 1] and (1, 4] give equal weight to their
    # midpoint responses. For x/4 the centering constant is (1/8 + 5/8)/2.
    np.testing.assert_array_equal(data["bin_edges"], [0, 1, 4])
    np.testing.assert_array_equal(data["local_effects"], sign * np.array([0.25, 0.75]))
    np.testing.assert_array_equal(data["ale_values"], sign * np.array([-0.375, -0.125, 0.625]))
