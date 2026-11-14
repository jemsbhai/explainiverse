"""ProtoDash keeps the target's prediction when a model reuses its output buffer."""

from math import exp

import numpy as np
import pytest

from explainiverse.explainers.example_based import ProtoDashExplainer


class BufferedModel:
    def __init__(self, reuse, multioutput):
        self.reuse = reuse
        self.multioutput = multioutput
        self.buffer = np.empty((2, 2))

    def predict(self, rows):
        output = self.buffer[: len(rows)]
        output[:, 0] = 2 * rows[:, 0]
        output[:, 1] = -rows[:, 0]
        values = output if self.multioutput else output[:, 0]
        return values if self.reuse else values.copy()


@pytest.mark.parametrize("reuse", [False, True], ids=["fresh", "shared"])
@pytest.mark.parametrize("use_predictions", [False, True], ids=["input", "augmented"])
def test_single_prediction_buffers(reuse, use_predictions):
    explainer = ProtoDashExplainer(
        BufferedModel(reuse, multioutput=True), n_prototypes=1, kernel_width=1.0
    )
    data = explainer.explain(
        np.array([2.5]), np.array([[0.0], [3.0]]), use_predictions=use_predictions
    ).explanation_data

    # Row 3 is closest in both spaces. Its coordinate differences from the
    # target are 0.5 in input space and (0.5, 1, -0.5) in augmented space.
    squared_distance = 1.5 if use_predictions else 0.25
    assert data["prototype_indices"] == [1]
    assert data["similarity_scores"] == pytest.approx([exp(-squared_distance / 2)])
    assert data["instance_prediction"] == [[5.0, -2.5]]
    assert data["prototype_predictions"] == [[6.0, -3.0]]


@pytest.mark.parametrize("reuse", [False, True], ids=["fresh", "shared"])
def test_batch_prediction_buffers(reuse):
    explainer = ProtoDashExplainer(
        BufferedModel(reuse, multioutput=False), n_prototypes=1, kernel_width=1.0
    )
    results = explainer.explain_batch(np.array([[0.5], [2.5]]), np.array([[0.0], [3.0]]))
    # Each instance is half a unit away from its nearest reference in input space.
    assert len(results) == 2
    for result, index, target, prototype in zip(results, [0, 1], [1.0, 5.0], [0.0, 6.0]):
        data = result.explanation_data
        assert data["prototype_indices"] == [index]
        assert data["similarity_scores"] == pytest.approx([exp(-0.25 / 2)])
        assert data["instance_prediction"] == [target]
        assert data["prototype_predictions"] == [prototype]
