"""PDP and ALE for the single output of a declared one-class classifier."""

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from explainiverse.adapters.sklearn_adapter import SklearnAdapter
from explainiverse.explainers import ALEExplainer, PartialDependenceExplainer

ROWS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
NAMES = ["x", "z"]


def _check_target(result, index):
    assert result.target_class == f"class_{index}"
    assert result.explanation_data["output_index"] == result.metadata["output_index"] == index


def _fitted_classifier():
    return SklearnAdapter(DummyClassifier().fit(ROWS, ["dog"] * len(ROWS)))


@pytest.mark.parametrize("target", [None, 0])
def test_pdp_fitted_single_class_probability(target):
    result = PartialDependenceExplainer(_fitted_classifier(), ROWS, NAMES).explain(
        ["x", ("x", "z")], target_class=target
    )
    data = result.explanation_data
    assert data["grid_values"] == {"x": [0, 1], "x_x_z": {"grid1": [0, 1], "grid2": [0, 1]}}
    assert data["pdp_values"] == {"x": [1, 1], "x_x_z": [[1, 1], [1, 1]]}
    _check_target(result, 0)


@pytest.mark.parametrize("target", [None, 0])
def test_ale_fitted_single_class_probability(target):
    for result in ALEExplainer(_fitted_classifier(), ROWS, NAMES).explain_all(target_class=target):
        data = result.explanation_data
        assert data["grid_values"] == [0, 1]
        assert data["ale_values"] == [0, 0]
        _check_target(result, 0)


@pytest.mark.parametrize("method", ["pdp", "ale"])
def test_single_class_classifier_rejects_a_second_class(method):
    model = _fitted_classifier()
    with pytest.raises(ValueError, match="target_class"):
        if method == "pdp":
            PartialDependenceExplainer(model, ROWS, NAMES).explain(["x"], target_class=1)
        else:
            ALEExplainer(model, ROWS, NAMES).explain("x", target_class=1)


class SingleClassScore:
    task = "classification"
    classes_ = np.array(["dog"])
    prediction_output_kind = "scores"

    def predict(self, rows):
        return 2 * rows[:, [0]] - 4 * rows[:, [1]] - 1


def test_pdp_uses_single_class_score_column():
    result = PartialDependenceExplainer(SingleClassScore(), ROWS, NAMES).explain(
        ["x", ("x", "z")], target_class=0
    )
    # Holding x fixed leaves z uniformly distributed across its two levels.
    assert result.explanation_data["pdp_values"] == {
        "x": [-3, -1],
        "x_x_z": [[-1, -5], [1, -3]],
    }
    _check_target(result, 0)


@pytest.mark.parametrize("target", [None, 0])
def test_ale_uses_single_class_score_differences(target):
    results = ALEExplainer(SingleClassScore(), ROWS, NAMES).explain_all(target_class=target)
    for result, slope in zip(results, (2, -4)):
        data = result.explanation_data
        assert data["local_effects"] == [slope]
        assert data["ale_values"] == [-slope / 2, slope / 2]
        _check_target(result, 0)


def test_ale_does_not_complement_a_bounded_single_class_score():
    class BoundedScore(SingleClassScore):
        def predict(self, rows):
            return (1 + rows[:, [0]]) / 4

    result = ALEExplainer(BoundedScore(), ROWS, NAMES).explain("x", target_class=0)
    # The sole score increases from 1/4 to 1/2 across the bin.
    assert result.explanation_data["local_effects"] == [0.25]
    assert result.explanation_data["ale_values"] == [-0.125, 0.125]


@pytest.mark.parametrize("classes", [np.array(["cat", "dog"]), None], ids=["binary", "unspecified"])
@pytest.mark.parametrize("target", [None, 0])
def test_compact_binary_curves_remain_class_complements(classes, target):
    class CompactBinary:
        task = "classification"
        classes_ = classes
        prediction_output_kind = "probabilities"

        def predict(self, rows):
            return (1 + rows[:, [0]]) / 4

    # P(class 1) increases from 1/4 to 1/2. P(class 0) decreases by 1/4.
    model = CompactBinary()
    pdp = PartialDependenceExplainer(model, ROWS, NAMES).explain(["x"], target_class=target)
    ale = ALEExplainer(model, ROWS, NAMES).explain("x", target_class=target)
    output = 1 if target is None else target
    _check_target(pdp, output)
    _check_target(ale, output)
    assert pdp.explanation_data["pdp_values"]["x"] == ([0.25, 0.5] if output else [0.75, 0.5])
    delta = 0.25 if output else -0.25
    assert ale.explanation_data["ale_values"] == [-delta / 2, delta / 2]
