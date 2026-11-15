"""AnchorTabular records query scores before the model predicts any background rows."""

import numpy as np
import pytest

from explainiverse.explainers import AnchorTabularExplainer


class _LinearClassifier:
    def __init__(self, width, output_kind, vector, reuse):
        self.classes_ = np.arange(width)
        self.prediction_output_kind = output_kind
        self.vector = vector
        self.reuse = reuse
        self.buffer = np.empty((8, width))

    def predict(self, rows):
        x = rows[:, 0]
        scores = self.buffer[: len(rows)]
        if self.prediction_output_kind == "probabilities":
            scores[:, 0] = (2 + x) / 4
            scores[:, 1] = 1 - scores[:, 0]
        else:
            scores[:, 0] = x
            if scores.shape[1] >= 2:
                scores[:, 1] = -x
            if scores.shape[1] == 3:
                scores[:, 2] = -5
        if self.vector:
            scores = scores[:, 0]
        return scores if self.reuse else scores.copy()


@pytest.mark.parametrize("reuse", [False, True], ids=["fresh", "reused"])
@pytest.mark.parametrize(
    "kind,vector,query_scores,background_label",
    [
        pytest.param("scores", False, [2.0, -2.0], 1, id="binary-scores"),
        pytest.param(None, False, [2.0, -2.0, -5.0], 1, id="legacy-multiclass-scores"),
        pytest.param("scores", False, [2.0], 0, id="single-class-column"),
        pytest.param("scores", True, [2.0], 0, id="single-class-vector"),
        pytest.param("probabilities", False, [1.0, 0.0], 1, id="binary-probabilities"),
    ],
)
def test_anchor_records_query_prediction(kind, vector, query_scores, background_label, reuse):
    names = ["positive", "negative", "other"][: len(query_scores)]
    model = _LinearClassifier(len(names), kind, vector, reuse)
    explanation = AnchorTabularExplainer(
        model,
        np.full((4, 1), -1.0),
        ["x"],
        names,
        batch_size=8,
        max_samples=128,
        random_state=21,
    ).explain(np.array([2.0]))

    data = explanation.explanation_data
    assert explanation.target_class == names[0]
    assert data["target_output_index"] == 0
    assert data["rules"] == []
    assert data["coverage"] == 1.0
    assert data["precision"] == float(background_label == 0)
    np.testing.assert_array_equal(data["model_prediction"], query_scores)

    model.predict(np.array([[0.0]]))
    np.testing.assert_array_equal(data["model_prediction"], query_scores)
