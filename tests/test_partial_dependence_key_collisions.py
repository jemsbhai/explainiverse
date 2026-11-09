"""Regression contracts for unambiguous PDP result keys."""

import numpy as np
import pytest

from explainiverse.explainers.global_explainers.partial_dependence import PartialDependenceExplainer


class _CountingLinearModel:
    task = "regression"

    def __init__(self):
        self.calls = 0

    def predict(self, X):
        self.calls += 1
        return np.asarray(X) @ np.arange(1, X.shape[1] + 1)


def _explainer(names):
    model = _CountingLinearModel()
    X = np.array([np.zeros(len(names)), np.ones(len(names))])
    return PartialDependenceExplainer(model, X, names, grid_resolution=2), model


@pytest.mark.parametrize(
    "names,features",
    [
        (["a", "b", "a_x_b"], [("a", "b"), "a_x_b"]),
        (["a", "b", "a_x_b"], ["a_x_b", ("a", "b")]),
        (["a", "b", "a_x_b"], [(0, 1), 2]),
        (["a", "b", "a_x_b"], [2, (0, 1)]),
        (["a_x_b", "c", "a", "b_x_c"], [("a_x_b", "c"), ("a", "b_x_c")]),
        (["a_x_b", "c", "a", "b_x_c"], [("a", "b_x_c"), ("a_x_b", "c")]),
        (["a", "a_x_a"], [("a", "a_x_a"), ("a_x_a", "a")]),
    ],
)
def test_different_pdp_selections_cannot_overwrite_the_same_result_key(names, features):
    explainer, model = _explainer(names)

    with pytest.raises(ValueError, match="result key|collid|collision"):
        explainer.explain(features)

    assert model.calls == 0


@pytest.mark.parametrize(
    "features,unique_features",
    [(["a", 0, "a"], ["a"]), ([("a", "b"), (0, 1)], [("a", "b")])],
)
def test_duplicate_selections_of_the_same_features_remain_compatible(features, unique_features):
    explainer, _ = _explainer(["a", "b"])
    expected = explainer.explain(unique_features).explanation_data
    observed = explainer.explain(features).explanation_data

    for field in ["pdp_values", "grid_values", "grid_types"]:
        assert observed[field] == expected[field]
    assert observed["features_analyzed"] == [str(feature) for feature in features]


@pytest.mark.parametrize("features", [["a_x_b"], [("a", "b")]])
def test_delimiter_in_feature_names_is_allowed_when_requested_result_keys_are_unique(features):
    explainer, model = _explainer(["a", "b", "a_x_b"])

    result = explainer.explain(features).explanation_data

    assert list(result["pdp_values"]) == ["a_x_b"]
    assert model.calls > 0


def test_noncolliding_mixed_selections_preserve_existing_result_keys_and_grids():
    explainer, _ = _explainer(["a", "b", "c"])

    result = explainer.explain(["a", ("b", "c")]).explanation_data

    assert list(result["pdp_values"]) == ["a", "b_x_c"]
    assert result["grid_values"] == {
        "a": [0.0, 1.0],
        "b_x_c": {"grid1": [0.0, 1.0], "grid2": [0.0, 1.0]},
    }
    assert result["grid_types"] == {"a": "numeric", "b_x_c": ["numeric", "numeric"]}
    np.testing.assert_allclose(result["pdp_values"]["a"], [2.5, 3.5])
    np.testing.assert_allclose(result["pdp_values"]["b_x_c"], [[0.5, 3.5], [2.5, 5.5]])
