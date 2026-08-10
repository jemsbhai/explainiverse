"""Accuracy and contract tests for the model-agnostic KernelSHAP wrapper."""

from __future__ import annotations

import numpy as np
import pytest
import shap
from sklearn.linear_model import LogisticRegression

from explainiverse.explainers.attribution.shap_wrapper import ShapExplainer


class LinearModel:
    """Minimal regression model with a batched ``predict`` contract."""

    task = "regression"

    def __init__(self, coefficients):
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.n_features_in_ = self.coefficients.size

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coefficients


class OneColumnBinaryModel:
    """Binary adapter exposing only the positive-class probability column."""

    task = "classification"

    def predict(self, X):
        logits = np.asarray(X, dtype=float)[:, 0]
        return (1.0 / (1.0 + np.exp(-logits)))[:, None]


class NonlinearModel:
    """Non-additive model whose sampled KernelSHAP fit depends on coalitions."""

    task = "regression"

    def __init__(self, n_features):
        self.coefficients = np.linspace(0.5, 2.0, n_features)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.sin(X @ self.coefficients) + np.prod(X[:, :4] + 0.2, axis=1)


class TwoOutputRegressionModel:
    task = "regression"

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.column_stack((X[:, 0] + X[:, 1], X[:, 0] - 2 * X[:, 1]))


class AmbiguousTwoOutputModel:
    """Two numerical outputs with no task semantics."""

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.column_stack((X[:, 0], -X[:, 0]))


def test_default_disables_shap_version_dependent_top_ten_sparsification():
    n_features = 12
    model = LinearModel(np.arange(1, n_features + 1, dtype=float))
    feature_names = [f"x{i}" for i in range(n_features)]
    explainer = ShapExplainer(
        model,
        background_data=np.zeros((1, n_features)),
        feature_names=feature_names,
        class_names=["output"],
        nsamples=500,
        random_state=7,
    )

    explanation = explainer.explain(np.ones(n_features))
    actual = np.asarray(explanation.explanation_data["shap_values_raw"])

    np.testing.assert_allclose(actual, model.coefficients, rtol=0, atol=1e-8)
    assert np.count_nonzero(actual) == n_features
    assert explanation.metadata["l1_reg"] == 0.0


def test_matches_official_kernel_explainer_with_explicit_unregularized_fit():
    coefficients = np.array([1.5, -2.0, 0.25, 3.0, -0.75, 2.5])
    model = LinearModel(coefficients)
    background = np.array(
        [
            [-1.0, 0.0, 1.0, 0.5, 2.0, -0.5],
            [1.0, 2.0, -1.0, 1.5, 0.0, 0.5],
        ]
    )
    instance = np.array([0.5, -1.0, 2.0, 1.0, -2.0, 3.0])
    feature_names = [f"x{i}" for i in range(coefficients.size)]

    wrapper = ShapExplainer(
        model,
        background,
        feature_names,
        ["output"],
        nsamples="auto",
        l1_reg=0.0,
        random_state=19,
    )
    wrapped = wrapper.explain(instance)

    official = shap.KernelExplainer(model.predict, background)
    state = np.random.get_state()
    try:
        np.random.seed(19)
        expected = official.shap_values(
            instance[None, :], nsamples="auto", l1_reg=0.0, silent=True
        )[0]
    finally:
        np.random.set_state(state)

    np.testing.assert_allclose(wrapped.explanation_data["shap_values_raw"], expected, atol=1e-12)


@pytest.mark.parametrize(
    ("labels", "display_names"),
    [
        (np.array(["ant", "zebra"]), None),
        (np.array([2, 7]), ["two", "seven"]),
    ],
)
def test_raw_classifier_predictions_map_through_model_classes(labels, display_names):
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([labels[0], labels[0], labels[1], labels[1]])
    model = LogisticRegression(random_state=0).fit(X, y)
    expected_index = int(np.argmax(model.predict_proba([[2.0]])[0]))

    wrapper = ShapExplainer(
        model,
        X,
        ["x"],
        class_names=display_names,
        random_state=11,
    )
    explanation = wrapper.explain(np.array([2.0]))

    expected_display = (
        display_names[expected_index]
        if display_names is not None
        else str(model.classes_[expected_index])
    )
    assert explanation.target_class == expected_display
    assert explanation.metadata["output_index"] == expected_index
    assert explanation.metadata["model_class_label"] == model.classes_[expected_index]
    assert abs(explanation.metadata["additivity_residual"]) < 1e-10


def test_one_column_binary_outputs_expand_and_classes_are_complements():
    model = OneColumnBinaryModel()
    background = np.array([[-2.0], [0.0], [2.0]])
    wrapper = ShapExplainer(
        model,
        background,
        ["x"],
        ["negative", "positive"],
        random_state=5,
    )

    negative = wrapper.explain(np.array([1.0]), target_class=0)
    positive = wrapper.explain(np.array([1.0]), target_class=1)

    np.testing.assert_allclose(
        negative.explanation_data["shap_values_raw"],
        -np.asarray(positive.explanation_data["shap_values_raw"]),
        atol=1e-12,
    )
    assert negative.explanation_data["expected_value"] == pytest.approx(
        1.0 - positive.explanation_data["expected_value"]
    )
    assert negative.metadata["output_index"] == 0
    assert positive.metadata["output_index"] == 1
    assert negative.metadata["output_space"] == "probability"
    assert abs(negative.metadata["additivity_residual"]) < 1e-12
    assert abs(positive.metadata["additivity_residual"]) < 1e-12


def test_binary_one_column_matches_official_two_column_reference():
    model = OneColumnBinaryModel()
    background = np.array([[-1.5], [0.0], [1.5]])
    instance = np.array([[0.75]])
    wrapper = ShapExplainer(
        model,
        background,
        ["x"],
        ["negative", "positive"],
        random_state=31,
    )

    def two_column_predict(X):
        positive = model.predict(X)[:, 0]
        return np.column_stack((1.0 - positive, positive))

    official = shap.KernelExplainer(two_column_predict, background)
    reference = official.shap_values(instance, nsamples="auto", l1_reg=0.0, silent=True)

    for output_index in (0, 1):
        result = wrapper.explain(instance[0], target_class=output_index)
        np.testing.assert_allclose(
            result.explanation_data["shap_values_raw"],
            reference[0, :, output_index],
            atol=1e-12,
        )


def test_additivity_metadata_identifies_regression_output_space():
    model = LinearModel([2.0, -3.0])
    wrapper = ShapExplainer(
        model,
        np.array([[0.0, 0.0], [1.0, -1.0]]),
        ["x0", "x1"],
        ["output"],
    )
    result = wrapper.explain(np.array([2.0, 4.0]))

    assert result.metadata["output_space"] == "model_output"
    assert result.metadata["model_reference_value"] == pytest.approx(-8.0)
    assert result.metadata["explained_value"] == pytest.approx(-8.0)
    assert abs(result.metadata["additivity_residual"]) < 1e-10


def test_logit_link_reports_and_satisfies_log_odds_additivity():
    wrapper = ShapExplainer(
        OneColumnBinaryModel(),
        np.array([[-2.0], [0.0], [2.0]]),
        ["x"],
        ["negative", "positive"],
        link="logit",
    )

    result = wrapper.explain(np.array([0.75]), target_class=1)

    assert result.metadata["output_space"] == "log_odds"
    assert result.metadata["model_output_space"] == "probability"
    assert result.metadata["model_reference_value"] == pytest.approx(0.75)
    assert result.metadata["explained_value"] == pytest.approx(0.75)
    assert abs(result.metadata["additivity_residual"]) < 1e-10


def test_multioutput_regression_requires_and_honors_an_output_index():
    model = TwoOutputRegressionModel()
    wrapper = ShapExplainer(
        model,
        np.zeros((1, 2)),
        ["x0", "x1"],
        ["sum", "contrast"],
    )

    with pytest.raises(ValueError, match="multi-output regression"):
        wrapper.explain(np.array([3.0, 4.0]))

    result = wrapper.explain(np.array([3.0, 4.0]), target_class=1)
    assert result.target_class == "contrast"
    assert result.metadata["model_reference_value"] == pytest.approx(-5.0)
    assert result.metadata["explained_value"] == pytest.approx(-5.0)
    np.testing.assert_allclose(result.explanation_data["shap_values_raw"], [3, -8])


def test_class_names_do_not_infer_task_for_ambiguous_multioutput_model():
    with pytest.raises(ValueError, match="task is required"):
        ShapExplainer(
            AmbiguousTwoOutputModel(),
            np.zeros((1, 1)),
            ["x"],
            class_names=["first", "second"],
        )

    explicit = ShapExplainer(
        AmbiguousTwoOutputModel(),
        np.zeros((1, 1)),
        ["x"],
        class_names=["first", "second"],
        task="regression",
    )
    result = explicit.explain(np.array([2.0]), target_class=1)
    assert result.target_class == "second"
    assert result.metadata["model_reference_value"] == pytest.approx(-2.0)


def test_seeded_sampling_is_repeatable_and_does_not_mutate_global_rng():
    n_features = 14
    model = NonlinearModel(n_features)
    instance = np.linspace(0.2, 1.2, n_features)
    wrapper = ShapExplainer(
        model,
        np.zeros((1, n_features)),
        [f"x{i}" for i in range(n_features)],
        ["output"],
        nsamples=80,
        random_state=123,
    )

    np.random.seed(991)
    expected_next = np.random.random(3)
    np.random.seed(991)
    first = wrapper.explain(instance)
    actual_next = np.random.random(3)
    second = wrapper.explain(instance)

    other_seed = ShapExplainer(
        model,
        np.zeros((1, n_features)),
        [f"x{i}" for i in range(n_features)],
        ["output"],
        nsamples=80,
        random_state=124,
    ).explain(instance)

    official = shap.KernelExplainer(model.predict, np.zeros((1, n_features)))
    state = np.random.get_state()
    try:
        np.random.seed(123)
        official_values = official.shap_values(
            instance[None, :], nsamples=80, l1_reg=0.0, silent=True
        )[0]
    finally:
        np.random.set_state(state)

    np.testing.assert_array_equal(actual_next, expected_next)
    np.testing.assert_array_equal(
        first.explanation_data["shap_values_raw"],
        second.explanation_data["shap_values_raw"],
    )
    np.testing.assert_array_equal(first.explanation_data["shap_values_raw"], official_values)
    assert not np.allclose(
        first.explanation_data["shap_values_raw"],
        other_seed.explanation_data["shap_values_raw"],
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"background_data": np.zeros(2)}, "two-dimensional"),
        ({"feature_names": ["only_one"]}, "feature_names"),
        ({"feature_names": ["duplicate", "duplicate"]}, "unique"),
        ({"nsamples": 0}, "nsamples"),
        ({"l1_reg": "not-a-rule"}, "l1_reg"),
        ({"random_state": -1}, "random_state"),
    ],
)
def test_constructor_rejects_invalid_contracts(kwargs, match):
    arguments = {
        "model": LinearModel([1.0, 2.0]),
        "background_data": np.zeros((2, 2)),
        "feature_names": ["x0", "x1"],
        "class_names": ["output"],
    }
    arguments.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=match):
        ShapExplainer(**arguments)


def test_explain_rejects_invalid_instance_target_and_top_labels():
    wrapper = ShapExplainer(
        OneColumnBinaryModel(),
        np.zeros((2, 1)),
        ["x"],
        ["negative", "positive"],
    )

    with pytest.raises(ValueError, match="exactly one instance"):
        wrapper.explain(np.zeros((2, 1)))
    with pytest.raises(ValueError, match="feature count"):
        wrapper.explain(np.zeros(2))
    with pytest.raises(ValueError, match="outside"):
        wrapper.explain(np.zeros(1), target_class=2)
    with pytest.raises(TypeError, match="integer output index"):
        wrapper.explain(np.zeros(1), target_class="positive")
    with pytest.raises(ValueError, match="top_labels"):
        wrapper.explain(np.zeros(1), top_labels=0)
    with pytest.raises(ValueError, match="top_labels must be 1"):
        wrapper.explain(np.zeros(1), top_labels=2)
