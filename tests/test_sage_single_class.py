"""The SAGE zero-one loss game for classifiers trained on a single class."""

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from explainiverse.adapters.sklearn_adapter import SklearnAdapter
from explainiverse.explainers import SAGEExplainer


def _explain(model, targets):
    rows = np.arange(2 * len(targets)).reshape(-1, 2)
    return (
        SAGEExplainer(
            model,
            rows,
            np.asarray(targets),
            ["a", "b"],
            n_permutations=2,
            background_data=rows[:2],
        )
        .explain()
        .explanation_data
    )


def _check_constant_label_game(model, targets, predicted):
    data = _explain(model, targets)
    # Every coalition predicts the same label. Count its errors directly.
    expected_loss = sum(target != predicted for target in targets) / len(targets)
    assert data["baseline_loss"] == data["full_loss"] == expected_loss
    assert data["feature_attributions"] == {"a": 0.0, "b": 0.0}
    assert data["total_value"] == data["efficiency_error"] == 0.0


@pytest.mark.parametrize(
    ("label", "targets"),
    [
        ("dog", ["dog", "dog", "dog"]),
        ("dog", ["cat", "dog", "cat"]),
        (17, [9, 12, 9]),
    ],
    ids=["all-correct", "partly-correct", "all-unseen"],
)
def test_sage_fitted_single_class_estimator(label, targets):
    fitted = DummyClassifier().fit(np.zeros((3, 2)), [label] * 3)
    _check_constant_label_game(SklearnAdapter(fitted), targets, label)


class ColumnClassifier:
    def __init__(self, classes, kind, value):
        self.classes_ = None if classes is None else np.asarray(classes)
        self.prediction_output_kind = kind
        self.value = value

    def predict(self, rows):
        return np.full((len(rows), 1), self.value)


@pytest.mark.parametrize(("kind", "value"), [("probabilities", 1.0), ("scores", -5.0), (None, 1.0)])
def test_sage_direct_single_class_column(kind, value):
    model = ColumnClassifier(["cat"], kind, value)
    _check_constant_label_game(model, ["cat", "fox", "owl", "cat"], "cat")


@pytest.mark.parametrize(
    ("kind", "value"), [("probabilities", 0.75), ("probabilities", 1.2), (None, 0.75)]
)
def test_sage_rejects_invalid_single_class_probability(kind, value):
    with pytest.raises(ValueError):
        _explain(ColumnClassifier(["cat"], kind, value), ["cat", "cat"])


@pytest.mark.parametrize(("probability", "label"), [(0.5, "cat"), (0.75, "dog")])
def test_sage_one_evaluation_class_preserves_compact_binary_column(probability, label):
    model = ColumnClassifier(["cat", "dog"], "probabilities", probability)
    _check_constant_label_game(model, ["cat", "cat", "cat"], label)


def test_sage_single_numeric_evaluation_label_without_metadata_remains_binary():
    model = ColumnClassifier(None, "probabilities", 1.0)
    _check_constant_label_game(model, [0, 0, 0], 1)
