"""A single class score cannot determine the winning class of a binary model."""

import numpy as np
import pytest

from explainiverse.explainers import PermutationImportanceExplainer

ROWS = np.arange(4)[:, None]
LABELS = np.array(["cat", "dog", "cat", "dog"])
POSITIVE = np.array([0.25, 0.75, 0.375, 0.625], dtype=np.float32)


class LookupClassifier:
    task = "classification"

    def __init__(self, outputs, kind, classes=("cat", "dog")):
        self.outputs = outputs
        self.prediction_output_kind = kind
        self.classes_ = classes

    def predict(self, rows):
        return self.outputs[rows[:, 0].astype(int)]


def explain(model, scoring_fn=None):
    return (
        PermutationImportanceExplainer(
            model, ROWS, LABELS, ["x"], n_repeats=3, random_state=3, scoring_fn=scoring_fn
        )
        .explain()
        .explanation_data
    )


@pytest.mark.parametrize("outputs", [POSITIVE, POSITIVE[:, None]], ids=["vector", "column"])
@pytest.mark.parametrize("classes", [("cat", "dog"), None], ids=["fitted", "from-targets"])
def test_default_accuracy_rejects_incomplete_class_scores(outputs, classes):
    with pytest.raises(ValueError, match="one column per class"):
        explain(LookupClassifier(outputs, "scores", classes))


def test_full_scores_with_the_same_second_column_can_predict_opposite_labels():
    for first_column, accuracy, drops in (
        (1 - POSITIVE, 1.0, [1.0, 0.5, 1.0]),
        (POSITIVE > 0.5, 0.0, [-1.0, -0.5, -1.0]),
    ):
        scores = np.column_stack((first_column, POSITIVE))
        data = explain(LookupClassifier(scores, "scores"))
        assert data["baseline_score"] == accuracy
        assert data["importances"]["x"] == drops


@pytest.mark.parametrize("kind", ["probabilities", None], ids=["explicit", "legacy"])
@pytest.mark.parametrize("outputs", [POSITIVE, POSITIVE[:, None]], ids=["vector", "column"])
def test_compact_binary_probabilities_still_identify_the_winning_class(kind, outputs):
    data = explain(LookupClassifier(outputs, kind))
    assert data["baseline_score"] == 1.0
    assert data["importances"]["x"] == [1.0, 0.5, 1.0]


def test_single_fitted_class_can_be_scored_with_a_vector():
    data = explain(LookupClassifier(-POSITIVE, "scores", ("cat",)))
    assert data["baseline_score"] == 0.5
    assert data["importances"]["x"] == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("outputs", [POSITIVE, POSITIVE[:, None]], ids=["vector", "column"])
def test_custom_scorer_receives_compact_scores_in_model_format(outputs):
    def scorer(targets, predictions):
        np.testing.assert_array_equal(targets, LABELS)
        assert predictions.shape == outputs.shape
        assert predictions.dtype == outputs.dtype
        np.testing.assert_array_equal(np.sort(predictions, axis=0), np.sort(outputs, axis=0))
        labels = np.where(predictions.reshape(-1) > 0.5, "dog", "cat")
        return np.mean(labels == targets)

    data = explain(LookupClassifier(outputs, "scores"), scoring_fn=scorer)
    assert data["baseline_score"] == 1.0
    assert data["importances"]["x"] == [1.0, 0.5, 1.0]
