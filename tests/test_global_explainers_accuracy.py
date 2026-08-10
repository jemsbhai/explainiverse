"""Accuracy oracles for the released global explainers.

These tests assert analytical identities and independent scikit-learn
references. They intentionally go beyond shape/range smoke tests.
"""

import numpy as np
import pytest
from sklearn.inspection import partial_dependence as sklearn_partial_dependence
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression

from explainiverse.adapters.sklearn_adapter import SklearnAdapter
from explainiverse.explainers.global_explainers.ale import ALEExplainer
from explainiverse.explainers.global_explainers.partial_dependence import PartialDependenceExplainer
from explainiverse.explainers.global_explainers.permutation_importance import (
    PermutationImportanceExplainer,
)


class FunctionAdapter:
    """Minimal adapter with an explicit task contract for analytical models."""

    def __init__(self, task, prediction_fn, classes=None):
        self.task = task
        self.prediction_fn = prediction_fn
        if classes is not None:
            self.classes_ = np.asarray(classes)

    def predict(self, X):
        return np.asarray(self.prediction_fn(np.asarray(X)))


class AmbiguousShapeModel:
    """A 2D prediction shape alone must not be treated as a task declaration."""

    def predict(self, X):
        return np.zeros((len(X), 1))


def test_permutation_importance_matches_sklearn_regression_reference():
    X = np.column_stack((np.linspace(-2.0, 2.0, 30), np.tile([-1.0, 0.0, 1.0], 10)))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1]
    estimator = LinearRegression().fit(X, y)

    result = PermutationImportanceExplainer(
        SklearnAdapter(estimator),
        X,
        y,
        ["x0", "x1"],
        n_repeats=7,
        random_state=13,
    ).explain()
    reference = permutation_importance(estimator, X, y, n_repeats=7, random_state=13)

    observed = result.explanation_data
    np.testing.assert_allclose(
        [observed["feature_attributions"]["x0"], observed["feature_attributions"]["x1"]],
        reference.importances_mean,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        [observed["std"]["x0"], observed["std"]["x1"]],
        reference.importances_std,
        atol=1e-12,
    )
    assert observed["baseline_score"] == pytest.approx(1.0)
    assert observed["task"] == "regression"
    assert observed["scoring"] == "r2"
    assert observed["score_direction"] == "higher_is_better"
    assert observed["score_input"] == "regression_response"


def test_permutation_importance_matches_sklearn_with_string_class_labels():
    X = np.column_stack((np.r_[np.zeros(10), np.ones(10)], np.arange(20) % 2))
    y = np.where(X[:, 0] > 0, "yes", "no")
    estimator = LogisticRegression(C=1e4).fit(X, y)

    result = PermutationImportanceExplainer(
        SklearnAdapter(estimator),
        X,
        y,
        ["signal", "noise"],
        n_repeats=5,
        random_state=2,
    ).explain()
    reference = permutation_importance(estimator, X, y, n_repeats=5, random_state=2)

    observed = result.explanation_data["feature_attributions"]
    np.testing.assert_allclose(
        [observed["signal"], observed["noise"]],
        reference.importances_mean,
        atol=1e-12,
    )
    assert result.explanation_data["score_input"] == "class_labels"


def test_permutation_importance_handles_one_column_binary_probability():
    X = np.array([[0.0], [0.0], [1.0], [1.0]])
    y = np.array(["negative", "negative", "positive", "positive"])
    adapter = FunctionAdapter(
        "classification",
        lambda data: (0.1 + 0.8 * data[:, [0]]),
        classes=["negative", "positive"],
    )

    result = PermutationImportanceExplainer(
        adapter, X, y, ["signal"], n_repeats=3, random_state=0
    ).explain()

    assert result.explanation_data["baseline_score"] == pytest.approx(1.0)
    assert result.explanation_data["feature_attributions"]["signal"] > 0.0


def test_permutation_importance_custom_scorer_receives_raw_model_prediction():
    X = np.arange(6.0).reshape(-1, 1)
    y = 2.0 * X[:, 0]
    adapter = FunctionAdapter("regression", lambda data: 2.0 * data[:, [0]])
    observed_shapes = []

    def negative_mse(y_true, model_prediction):
        observed_shapes.append(model_prediction.shape)
        return -np.mean((y_true - model_prediction[:, 0]) ** 2)

    result = PermutationImportanceExplainer(
        adapter,
        X,
        y,
        ["x"],
        n_repeats=2,
        random_state=0,
        scoring_fn=negative_mse,
    ).explain()

    assert observed_shapes == [(6, 1), (6, 1), (6, 1)]
    assert result.explanation_data["score_input"] == "model_prediction"
    assert result.explanation_data["scoring"] == "negative_mse"


@pytest.mark.parametrize(
    "constructor,args",
    [
        (
            PermutationImportanceExplainer,
            (np.ones((3, 1)), np.ones(3), ["x"]),
        ),
        (PartialDependenceExplainer, (np.ones((3, 1)), ["x"])),
        (ALEExplainer, (np.ones((3, 1)), ["x"])),
    ],
)
def test_global_explainers_do_not_infer_task_from_2d_output(constructor, args):
    with pytest.raises(ValueError, match="task is ambiguous"):
        constructor(AmbiguousShapeModel(), *args)


def test_partial_dependence_single_output_regression_is_analytically_exact():
    X = np.column_stack((np.linspace(-3.0, 3.0, 31), np.linspace(2.0, 5.0, 31) ** 2))
    adapter = FunctionAdapter(
        "regression",
        lambda data: (2.0 * data[:, 0] + 3.0 * data[:, 1]).reshape(-1, 1),
    )
    result = PartialDependenceExplainer(adapter, X, ["x0", "x1"], grid_resolution=7).explain(["x0"])

    grid = np.asarray(result.explanation_data["grid_values"]["x0"])
    observed = np.asarray(result.explanation_data["pdp_values"]["x0"])
    expected = 2.0 * grid + 3.0 * np.mean(X[:, 1])
    np.testing.assert_allclose(observed, expected, atol=1e-12)
    assert result.target_class == "output_0"
    assert result.explanation_data["output_index"] == 0
    assert result.explanation_data["output_space"] == "regression_response"


def test_partial_dependence_matches_sklearn_brute_reference():
    X = np.column_stack((np.tile([-2.0, -1.0, 0.0, 1.0, 2.0], 6), np.linspace(-3.0, 4.0, 30)))
    y = 1.5 * X[:, 0] - 0.75 * X[:, 1]
    estimator = LinearRegression().fit(X, y)

    result = PartialDependenceExplainer(
        SklearnAdapter(estimator), X, ["x0", "x1"], grid_resolution=10
    ).explain([0])
    reference = sklearn_partial_dependence(
        estimator,
        X,
        [0],
        method="brute",
        grid_resolution=10,
        percentiles=(0.05, 0.95),
    )

    np.testing.assert_allclose(
        result.explanation_data["grid_values"]["x0"], reference["grid_values"][0]
    )
    np.testing.assert_allclose(result.explanation_data["pdp_values"]["x0"], reference["average"][0])


def test_partial_dependence_preserves_fractional_grid_for_integer_reference_data():
    X = np.arange(11).reshape(-1, 1)
    adapter = FunctionAdapter("regression", lambda data: data[:, [0]])

    result = PartialDependenceExplainer(adapter, X, ["x"], grid_resolution=4).explain([0])

    grid = np.asarray(result.explanation_data["grid_values"]["x"])
    np.testing.assert_allclose(grid, [0.5, 3.5, 6.5, 9.5])
    np.testing.assert_allclose(result.explanation_data["pdp_values"]["x"], grid)


def test_partial_dependence_one_column_binary_targets_are_complements():
    X = np.linspace(0.0, 1.0, 11).reshape(-1, 1)
    adapter = FunctionAdapter("classification", lambda data: 0.2 + 0.6 * data[:, [0]])
    explainer = PartialDependenceExplainer(adapter, X, ["x"], grid_resolution=5)

    positive = explainer.explain([0], target_class=1)
    negative = explainer.explain([0], target_class=0)
    p1 = np.asarray(positive.explanation_data["pdp_values"]["x"])
    p0 = np.asarray(negative.explanation_data["pdp_values"]["x"])

    np.testing.assert_allclose(p0 + p1, np.ones_like(p1), atol=1e-12)
    assert positive.target_class == "class_1"
    assert negative.target_class == "class_0"
    with pytest.raises(ValueError, match="only supports target_class 0 or 1"):
        explainer.explain([0], target_class=2)


def test_partial_dependence_two_column_binary_validates_target_index():
    X = np.linspace(0.0, 1.0, 11).reshape(-1, 1)
    adapter = FunctionAdapter(
        "classification",
        lambda data: np.column_stack((1.0 - (0.2 + 0.6 * data[:, 0]), 0.2 + 0.6 * data[:, 0])),
    )
    explainer = PartialDependenceExplainer(adapter, X, ["x"])

    with pytest.raises(ValueError, match=r"target_class must be in \[0, 1\]"):
        explainer.explain([0], target_class=2)


def test_partial_dependence_uses_observed_categorical_values_only():
    X = np.array(
        [["bronze", 0.0], ["gold", 1.0], ["bronze", 2.0], ["silver", 3.0]],
        dtype=object,
    )
    values = {"bronze": 1.0, "silver": 4.0, "gold": 9.0}
    adapter = FunctionAdapter(
        "regression",
        lambda data: np.array([values[value] for value in data[:, 0]]).reshape(-1, 1),
    )
    explainer = PartialDependenceExplainer(
        adapter,
        X,
        ["tier", "numeric"],
        categorical_features=["tier"],
    )

    result = explainer.explain(["tier"])
    grid = result.explanation_data["grid_values"]["tier"]
    observed = result.explanation_data["pdp_values"]["tier"]

    assert grid == ["bronze", "gold", "silver"]
    np.testing.assert_allclose(observed, [1.0, 9.0, 4.0])
    assert result.explanation_data["grid_types"]["tier"] == "categorical"

    with pytest.raises(TypeError, match="Declare it in categorical_features"):
        PartialDependenceExplainer(adapter, X, ["tier", "numeric"]).explain(["tier"])


def test_partial_dependence_multioutput_regression_requires_output_index():
    X = np.linspace(-1.0, 1.0, 9).reshape(-1, 1)
    adapter = FunctionAdapter(
        "regression", lambda data: np.column_stack((data[:, 0], data[:, 0] ** 2))
    )
    explainer = PartialDependenceExplainer(adapter, X, ["x"])

    with pytest.raises(ValueError, match="explicit output index"):
        explainer.explain([0])
    result = explainer.explain([0], target_class=1)
    grid = np.asarray(result.explanation_data["grid_values"]["x"])
    np.testing.assert_allclose(result.explanation_data["pdp_values"]["x"], grid**2, atol=1e-12)
    assert result.target_class == "output_1"


def test_ale_linear_model_uses_empirical_weighted_centering_on_skewed_data():
    # Quantiles collapse to edges [0, 1, 10], with interval counts [80, 20].
    # Apley's ALEPlot centering constant for f(x)=x is therefore
    # 0.8 * (0 + 1)/2 + 0.2 * (1 + 10)/2 = 1.5.
    values = np.array([0.0] * 60 + [1.0] * 20 + [2.0] * 10 + [10.0] * 10)
    X = values.reshape(-1, 1)
    adapter = FunctionAdapter("regression", lambda data: data[:, [0]])

    result = ALEExplainer(adapter, X, ["x"], n_bins=4).explain(0)
    data = result.explanation_data

    np.testing.assert_allclose(data["grid_values"], [0.0, 1.0, 10.0])
    np.testing.assert_allclose(data["ale_values"], [-1.5, -0.5, 8.5])
    np.testing.assert_array_equal(data["bin_counts"], [80, 20])
    midpoint_effects = np.asarray(data["ale_values_at_bin_centers"])
    assert np.average(midpoint_effects, weights=data["bin_counts"]) == pytest.approx(0.0, abs=1e-12)
    assert len(data["grid_values"]) == len(data["ale_values"])
    assert result.metadata["centering"] == "empirical_bin_count_weighted_trapezoid"


def test_ale_nonlinear_model_matches_reference_accumulation_formula():
    X = np.arange(5.0).reshape(-1, 1)
    adapter = FunctionAdapter("regression", lambda data: data[:, [0]] ** 2)

    result = ALEExplainer(adapter, X, ["x"], n_bins=4).explain("x")
    data = result.explanation_data

    # Edge effects are [0, 1, 4, 9, 16]. Right-closed bin counts are
    # [2, 1, 1, 1], so the weighted trapezoid centering constant is 4.5.
    np.testing.assert_allclose(data["grid_values"], [0, 1, 2, 3, 4])
    np.testing.assert_allclose(data["local_effects"], [1, 3, 5, 7])
    np.testing.assert_allclose(data["ale_values"], [-4.5, -3.5, -0.5, 4.5, 11.5])
    np.testing.assert_array_equal(data["bin_counts"], [2, 1, 1, 1])


def test_ale_matches_reference_type1_quantiles_for_integer_data():
    X = np.array([0, 1, 2, 10]).reshape(-1, 1)
    adapter = FunctionAdapter("regression", lambda data: data[:, [0]] ** 2)

    result = ALEExplainer(adapter, X, ["x"], n_bins=2).explain(0)
    data = result.explanation_data

    np.testing.assert_allclose(data["grid_values"], [0.0, 1.0, 10.0])
    np.testing.assert_allclose(data["local_effects"], [1.0, 99.0])
    np.testing.assert_allclose(data["ale_values"], [-25.5, -24.5, 74.5])
    assert data["curve_range"] == pytest.approx(100.0)
    assert data["curve_range_semantics"] == "max_minus_min_of_centered_ale_curve"
    assert "feature_attributions" not in data
    assert result.metadata["quantile_method"] == "inverse_empirical_cdf (R type=1)"


def test_ale_one_column_binary_targets_are_sign_reversals():
    X = np.linspace(0.0, 1.0, 9).reshape(-1, 1)
    adapter = FunctionAdapter("classification", lambda data: 0.15 + 0.7 * data[:, [0]])
    explainer = ALEExplainer(adapter, X, ["x"], n_bins=4)

    positive = explainer.explain(0, target_class=1)
    negative = explainer.explain(0, target_class=0)
    np.testing.assert_allclose(
        negative.explanation_data["ale_values"],
        -np.asarray(positive.explanation_data["ale_values"]),
        atol=1e-12,
    )
    assert positive.explanation_data["output_space"] == "classification_score"


def test_ale_rejects_nominal_features_instead_of_imposing_an_order():
    X = np.array([["small"], ["large"]], dtype=object)
    adapter = FunctionAdapter(
        "regression", lambda data: np.arange(len(data), dtype=float).reshape(-1, 1)
    )

    with pytest.raises(TypeError, match="requires an ordered numeric feature"):
        ALEExplainer(adapter, X, ["size"]).explain("size")
