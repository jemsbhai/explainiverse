"""Default permutation accuracy with single-class probability columns."""

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from explainiverse.adapters.sklearn_adapter import SklearnAdapter
from explainiverse.explainers import PermutationImportanceExplainer


def explain(model, labels):
    rows = np.arange(len(labels) * 2).reshape(len(labels), 2)
    return (
        PermutationImportanceExplainer(model, rows, np.array(labels), ["a", "b"], n_repeats=2)
        .explain()
        .explanation_data
    )


@pytest.mark.parametrize(
    ("label", "evaluation"),
    [
        ("yes", ["yes", "yes", "yes", "yes"]),
        ("yes", ["yes", "no", "no", "yes"]),
        (17, [9, 12, 12, 9]),
    ],
    ids=["all-correct", "partly-correct", "unseen-targets"],
)
def test_single_class_estimator_accuracy(label, evaluation):
    estimator = DummyClassifier().fit(np.zeros((3, 2)), [label] * 3)
    data = explain(SklearnAdapter(estimator), evaluation)
    # The fitted estimator always predicts its sole training label.
    assert data["baseline_score"] == evaluation.count(label) / len(evaluation)
    assert data["importances"] == {"a": [0.0, 0.0], "b": [0.0, 0.0]}
    assert data["feature_attributions"] == data["std"] == {"a": 0.0, "b": 0.0}


class ConstantColumn:
    task = "classification"

    def __init__(self, classes, value, output_kind="probabilities"):
        self.classes_ = classes
        self.value = value
        self.prediction_output_kind = output_kind

    def predict(self, rows):
        return np.full((len(rows), 1), self.value)


@pytest.mark.parametrize(("kind", "value"), [("probabilities", 1.0), ("scores", -5.0), (None, 1.0)])
def test_direct_single_class_metadata(kind, value):
    data = explain(ConstantColumn(["dog"], value, kind), ["cat", "dog", "cat", "cat"])
    assert data["baseline_score"] == 0.25
    assert data["importances"] == {"a": [0.0, 0.0], "b": [0.0, 0.0]}


@pytest.mark.parametrize(
    ("kind", "probability"), [("probabilities", 0.75), ("probabilities", 1.2), (None, 0.75)]
)
def test_single_class_requires_normalized_probabilities(kind, probability):
    with pytest.raises(ValueError):
        explain(ConstantColumn(["yes"], probability, kind), ["yes", "yes"])


@pytest.mark.parametrize(("probability", "accuracy"), [(0.5, 1.0), (1.0, 0.0)])
def test_declared_binary_classes_with_only_first_label_in_evaluation(probability, accuracy):
    data = explain(ConstantColumn(["yes", "no"], probability), ["yes", "yes"])
    assert data["baseline_score"] == accuracy


def test_evaluation_subset_does_not_establish_single_class_model():
    with pytest.raises(ValueError):
        explain(ConstantColumn(None, 1.0), ["yes", "yes"])
