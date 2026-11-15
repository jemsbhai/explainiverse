"""Tabular rule and counterfactual explainers respect fitted single-class output widths."""

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from explainiverse.adapters import SklearnAdapter
from explainiverse.explainers import (
    AnchorsExplainer,
    AnchorTabularExplainer,
    CounterfactualExplainer,
)

_ROWS = np.arange(12).reshape(6, 2)
_FEATURES = ["x", "y"]
_DISPLAY = ["only class"]


def _anchor(explainer, model, names):
    options = (
        {"batch_size": 32, "max_samples": 256}
        if explainer is AnchorTabularExplainer
        else {"n_samples": 24}
    )
    return explainer(model, _ROWS, _FEATURES, names, threshold=0.5, **options).explain(_ROWS[1])


def _check_empty_rule(result, names, expected_scores):
    index = int(np.argmax(expected_scores))
    assert result.target_class == names[index]
    data = result.explanation_data
    assert data["rules"] == []
    assert data["precision"] == 1.0
    assert data["coverage"] == 1.0
    if result.explainer_name == "AnchorTabular":
        assert data["target_output_index"] == index
        np.testing.assert_array_equal(data["model_prediction"], expected_scores)


@pytest.mark.parametrize("explainer", [AnchorTabularExplainer, AnchorsExplainer])
def test_anchor_for_classifier_fitted_on_one_class(explainer):
    fitted = DummyClassifier().fit(_ROWS, ["sole"] * len(_ROWS))
    np.testing.assert_array_equal(fitted.predict(_ROWS), ["sole"] * len(_ROWS))
    result = _anchor(explainer, SklearnAdapter(fitted), _DISPLAY)
    _check_empty_rule(result, _DISPLAY, [1.0])


class _ConstantOutput:
    def __init__(self, classes, kind, value, vector=False):
        self.classes_ = np.asarray(classes)
        self.prediction_output_kind = kind
        self.value = value
        self.vector = vector

    def predict(self, rows):
        values = np.full(len(rows), self.value)
        return values if self.vector else values[:, None]


@pytest.mark.parametrize(
    "classes,kind,value,vector,names,expected",
    [
        ([17], "probabilities", 1.0, True, _DISPLAY, [1.0]),
        ([17], "scores", -3.75, False, _DISPLAY, [-3.75]),
        ([17], "scores", -3.75, True, _DISPLAY, [-3.75]),
        ([17], None, 1.0, False, _DISPLAY, [1.0]),
        ([0], "class_labels", 0, True, _DISPLAY, [1.0]),
        ([17, 49], "probabilities", 0.75, True, ["low", "high"], [0.25, 0.75]),
        ([17, 49], None, 0.5, False, ["low", "high"], [0.5, 0.5]),
    ],
)
def test_anchor_respects_output_kind_and_fitted_width(
    classes, kind, value, vector, names, expected
):
    model = _ConstantOutput(classes, kind, value, vector)
    _check_empty_rule(_anchor(AnchorTabularExplainer, model, names), names, expected)


@pytest.mark.parametrize(
    "model,match",
    [
        (_ConstantOutput([17], "probabilities", 0.25), "sum to 1"),
        (_ConstantOutput([17], "class_labels", 17), "one-dimensional"),
        (_ConstantOutput([17, 49], "scores", -3.75), "one column per class"),
    ],
)
def test_anchor_rejects_invalid_single_column_outputs(model, match):
    names = [str(label) for label in model.classes_]
    with pytest.raises(ValueError, match=match):
        _anchor(AnchorsExplainer, model, names)


@pytest.mark.parametrize("use_adapter", [False, True])
def test_counterfactual_requires_an_alternative_fitted_class(use_adapter):
    model = DummyClassifier().fit(_ROWS, ["sole"] * len(_ROWS))
    with pytest.raises(ValueError, match="at least two classes"):
        CounterfactualExplainer(SklearnAdapter(model) if use_adapter else model, _ROWS, _FEATURES)
