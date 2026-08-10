"""Regression tests for the non-gradient explainer correctness remediation."""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from explainiverse.explainers._validation import (
    normalize_classifier_outputs,
    validate_single_tabular_instance,
)
from explainiverse.explainers.attribution.lime_wrapper import LimeExplainer
from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer
from explainiverse.explainers.counterfactual.dice_wrapper import CounterfactualExplainer
from explainiverse.explainers.example_based.protodash import ProtoDashExplainer
from explainiverse.explainers.global_explainers.sage import SAGEExplainer
from explainiverse.explainers.rule_based.anchors_wrapper import AnchorsExplainer


class _AlwaysPositiveOneColumn:
    task = "classification"

    def predict(self, X):
        return np.ones((len(X), 1), dtype=float)


class _BoundedRegressor:
    task = "regression"

    def predict(self, X):
        X = np.asarray(X)
        return 0.25 + 0.5 * X[:, 0]


class _TwoFeatureProbabilityModel:
    task = "classification"

    def predict(self, X):
        X = np.asarray(X)
        positive = 1.0 / (1.0 + np.exp(-(X[:, 0] + X[:, 1])))
        return np.column_stack((1.0 - positive, positive))


class _FlatSequenceLabelModel:
    task = "classification"

    def __init__(self, classes):
        self.classes_ = classes

    def predict(self, X):
        return np.asarray([self.classes_[index % 2] for index in range(len(X))])


def test_classifier_normalizer_accepts_flat_python_class_sequences():
    X = np.zeros((2, 1))

    for classes in (["negative", "positive"], ("negative", "positive")):
        observed = normalize_classifier_outputs(
            _FlatSequenceLabelModel(classes),
            X,
            context="test classifier",
            class_names=["negative", "positive"],
            require_probabilities=False,
            allow_label_predictions=True,
        )
        np.testing.assert_array_equal(observed, np.eye(2))


def test_classifier_normalizer_rejects_nested_multioutput_classes():
    model = _FlatSequenceLabelModel([np.array([0, 1]), np.array([0, 1])])

    with pytest.raises(ValueError, match="multi-output classification"):
        normalize_classifier_outputs(
            model,
            np.zeros((2, 1)),
            context="test classifier",
            class_names=["negative", "positive"],
            require_probabilities=False,
        )


@pytest.mark.parametrize(
    "class_names, message",
    [
        ([], "non-empty"),
        (["same", "same"], "unique"),
        (["negative", 1], "non-empty strings"),
        (["negative", ""], "non-empty strings"),
        (["negative", "   "], "non-empty strings"),
    ],
)
def test_classifier_normalizer_validates_display_class_names(class_names, message):
    with pytest.raises(ValueError, match=message):
        normalize_classifier_outputs(
            _TwoFeatureProbabilityModel(),
            np.zeros((1, 2)),
            context="test classifier",
            class_names=class_names,
            require_probabilities=True,
        )


def test_classifier_normalizer_rejects_a_string_as_the_class_names_container():
    with pytest.raises(TypeError, match="sequence of non-empty strings"):
        normalize_classifier_outputs(
            _TwoFeatureProbabilityModel(),
            np.zeros((1, 2)),
            context="test classifier",
            class_names="negative",
            require_probabilities=True,
        )


def test_classifier_normalizer_rejects_duplicate_model_class_labels():
    model = _FlatSequenceLabelModel(["same", "same"])

    with pytest.raises(ValueError, match="model.classes_ must contain unique labels"):
        normalize_classifier_outputs(
            model,
            np.zeros((2, 1)),
            context="test classifier",
            class_names=["negative", "positive"],
            require_probabilities=False,
            allow_label_predictions=True,
        )


@pytest.mark.parametrize(
    "instance",
    [
        np.array([1.0 + 2.0j]),
        np.array([1.0 + 2.0j], dtype=object),
    ],
)
def test_single_instance_validation_rejects_complex_values_before_cast(instance):
    with pytest.raises(ValueError, match="complex"):
        validate_single_tabular_instance(instance, 1, dtype=float)


def test_anchors_normalizes_one_column_binary_probabilities():
    training = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    explainer = AnchorsExplainer(
        _AlwaysPositiveOneColumn(),
        training,
        ["x0", "x1"],
        ["negative", "positive"],
        n_samples=32,
        random_state=3,
    )

    explanation = explainer.explain(np.array([0.5, 0.5]))

    assert explanation.target_class == "positive"
    assert explanation.explanation_data["precision"] == 1.0
    assert explanation.explanation_data["anchor_indices"] == []


def test_anchors_rejects_regressors_and_multirow_instances():
    training = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="requires a classification model"):
        AnchorsExplainer(
            _BoundedRegressor(),
            training,
            ["x0", "x1"],
            ["negative", "positive"],
        )

    explainer = AnchorsExplainer(
        _TwoFeatureProbabilityModel(),
        training,
        ["x0", "x1"],
        ["negative", "positive"],
        n_samples=8,
    )
    with pytest.raises(ValueError, match="single-row 2D array"):
        explainer.explain(np.array([[0.0], [1.0]]))


def test_counterfactual_rejects_regressors_and_multirow_instances():
    with pytest.raises(ValueError, match="requires a classification model"):
        CounterfactualExplainer(
            _BoundedRegressor(),
            np.array([[0.0], [0.5], [1.0]]),
            ["x"],
        )

    training = np.array([[-1.0, -1.0], [-0.5, 0.0], [0.5, 0.0], [1.0, 1.0]])
    explainer = CounterfactualExplainer(
        _TwoFeatureProbabilityModel(),
        training,
        ["x0", "x1"],
    )
    with pytest.raises(ValueError, match="single-row 2D array"):
        explainer.explain(np.array([[-0.5], [0.0]]))


def test_protodash_rejects_empty_support_and_multirow_instance():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    explainer = ProtoDashExplainer(n_prototypes=1, kernel="linear")

    with pytest.raises(ValueError, match="at least one prototype"):
        explainer.find_criticisms(X, [], n_criticisms=1)
    with pytest.raises(ValueError, match="single-row 2D array"):
        explainer.explain(np.array([[0.0], [1.0]]), X)


def test_lime_rejects_multirow_instance_that_has_the_right_element_count():
    pytest.importorskip("lime")
    iris = load_iris()
    model = LogisticRegression(max_iter=1000, random_state=0).fit(iris.data, iris.target)

    class _Adapter:
        task = "classification"

        def __init__(self, estimator):
            self.model = estimator

        def predict(self, X):
            return self.model.predict_proba(X)

    explainer = LimeExplainer(
        _Adapter(model),
        iris.data,
        list(iris.feature_names),
        list(iris.target_names),
        random_state=0,
    )

    with pytest.raises(ValueError, match="single-row 2D array"):
        explainer.explain(iris.data[0].reshape(2, 2))


def test_treeshap_accepts_supported_subclasses_and_rejects_multirow_instance():
    iris = load_iris()

    class _CustomForest(RandomForestClassifier):
        # Scikit-learn 1.9 removed this private marker from estimator mixins.
        _estimator_type = None

    model = _CustomForest(n_estimators=3, random_state=0).fit(iris.data, iris.target)
    explainer = TreeShapExplainer(
        model,
        list(iris.feature_names),
        list(iris.target_names),
    )

    assert explainer.task == "classification"
    explanation = explainer.explain(iris.data[0])
    assert len(explanation.explanation_data["feature_attributions"]) == iris.data.shape[1]
    with pytest.raises(ValueError, match="single-row 2D array"):
        explainer.explain(iris.data[0].reshape(2, 2))


def test_treeshap_preserves_supported_regressor_subclass_semantics():
    iris = load_iris()

    class _CustomRegressor(RandomForestRegressor):
        _estimator_type = None

    model = _CustomRegressor(n_estimators=3, random_state=0).fit(
        iris.data,
        iris.data[:, 0],
    )
    explainer = TreeShapExplainer(model, list(iris.feature_names))

    assert explainer.task == "regression"
    with pytest.raises(ValueError, match="conflicts with the tree estimator semantics"):
        TreeShapExplainer(
            model,
            list(iris.feature_names),
            task="classification",
        )


def test_sage_reports_feature_and_custom_loss_semantics():
    class _LinearRegressor:
        task = "regression"

        def predict(self, X):
            return np.asarray(X)[:, [0]]

    def absolute_loss(y_true, predictions):
        return np.mean(np.abs(np.asarray(y_true) - np.asarray(predictions).reshape(-1)))

    X = np.array([[-1.0], [0.0], [1.0]])
    explanation = SAGEExplainer(
        _LinearRegressor(),
        X,
        X[:, 0],
        ["signal"],
        n_permutations=1,
        loss_fn=absolute_loss,
        task="regression",
        random_state=0,
    ).explain()

    assert explanation.feature_names == ["signal"]
    assert explanation.explanation_data["loss_name"] == "absolute_loss"
    assert explanation.explanation_data["loss_direction"] == "lower_is_better"
    assert explanation.explanation_data["loss_is_custom"] is True
    assert explanation.metadata["loss_name"] == "absolute_loss"
