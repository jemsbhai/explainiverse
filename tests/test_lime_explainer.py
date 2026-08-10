# tests/test_lime_explainer.py
"""
Tests for LIME explainer wrapper.

Reference:
    Ribeiro et al., 2016 — "Why Should I Trust You?": Explaining the
    Predictions of Any Classifier. KDD 2016.
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from explainiverse.adapters.sklearn_adapter import SklearnAdapter
from explainiverse.core.explanation import Explanation
from explainiverse.explainers.attribution.lime_wrapper import LimeExplainer


@pytest.fixture
def iris_setup():
    """Standard Iris dataset setup for LIME tests."""
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = list(iris.feature_names)
    class_names = iris.target_names.tolist()

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    adapter = SklearnAdapter(model, class_names=class_names)

    explainer = LimeExplainer(
        model=adapter,
        training_data=X,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
    )
    return X, y, feature_names, class_names, adapter, explainer


def test_lime_explainer_target_and_shape(iris_setup):
    """LIME produces valid explanation with correct structure."""
    X, y, feature_names, class_names, adapter, explainer = iris_setup

    explanation = explainer.explain(X[0], num_features=3)

    # Explanation object structure
    assert isinstance(explanation, Explanation)
    assert "feature_attributions" in explanation.explanation_data
    attributions = explanation.explanation_data["feature_attributions"]
    assert isinstance(attributions, dict)

    # Attribution keys must be original feature names (not LIME discretized strings)
    assert set(attributions.keys()) == set(
        feature_names
    ), f"Attribution keys {set(attributions.keys())} != feature names {set(feature_names)}"

    # All features present; at most num_features have non-zero values
    non_zero = {k: v for k, v in attributions.items() if abs(v) > 1e-10}
    assert len(non_zero) <= 3, f"Expected at most 3 non-zero attributions, got {len(non_zero)}"

    # Attribution value types
    for value in attributions.values():
        assert isinstance(value, float)

    # target_class matches model prediction
    preds = adapter.predict(np.array([X[0]]))
    predicted_index = np.argmax(preds[0])
    predicted_label = class_names[predicted_index]
    assert explanation.target_class == predicted_label


def test_lime_explainer_num_features_range(iris_setup):
    """num_features controls how many features get non-zero attributions."""
    X, y, feature_names, class_names, adapter, explainer = iris_setup

    explanation = explainer.explain(X[1], num_features=2)
    attributions = explanation.explanation_data["feature_attributions"]

    # All features present in dict (complete attribution vector)
    assert len(attributions) == len(feature_names)

    # But at most 2 have non-zero values
    non_zero = {k: v for k, v in attributions.items() if abs(v) > 1e-10}
    assert len(non_zero) <= 2, f"Expected at most 2 non-zero attributions, got {len(non_zero)}"


def test_lime_explainer_requires_one_output_and_supports_explicit_target(iris_setup):
    """The singular return type never silently discards requested outputs."""
    X, y, feature_names, class_names, adapter, explainer = iris_setup

    with pytest.raises(ValueError, match="top_labels must be 1"):
        explainer.explain(X[2], top_labels=2)

    explanation = explainer.explain(X[2], target_class=0)
    assert explanation.target_class == class_names[0]


def test_lime_explainer_plot_is_explicitly_unimplemented(iris_setup):
    """The generic container must not pretend that a plot was rendered."""
    X, y, feature_names, class_names, adapter, explainer = iris_setup

    explanation = explainer.explain(X[3])
    with pytest.raises(NotImplementedError, match="no implemented plotting backend"):
        explanation.plot()


def test_lime_all_features_default(iris_setup):
    """When num_features=None (default), all features get attributions."""
    X, y, feature_names, class_names, adapter, explainer = iris_setup

    explanation = explainer.explain(X[0])
    attributions = explanation.explanation_data["feature_attributions"]
    assert set(attributions.keys()) == set(feature_names)
    # On this fitted fixture at least one returned coefficient is non-zero.
    non_zero = sum(1 for v in attributions.values() if abs(v) > 1e-10)
    assert non_zero >= 1


def test_lime_attribution_keys_match_feature_names(iris_setup):
    """Critical: attribution dict keys must be original feature names,
    NOT LIME's discretized strings like 'petal width (cm) <= 0.80'."""
    X, y, feature_names, class_names, adapter, explainer = iris_setup

    for i in [0, 50, 100]:  # One from each class
        explanation = explainer.explain(X[i])
        keys = set(explanation.explanation_data["feature_attributions"].keys())
        assert keys == set(
            feature_names
        ), f"Instance {i}: keys {keys} != feature names {set(feature_names)}"


def test_lime_batch_explain(iris_setup):
    """explain_batch produces correct number of explanations."""
    X, y, feature_names, class_names, adapter, explainer = iris_setup

    explanations = explainer.explain_batch(X[:3])
    assert len(explanations) == 3
    for exp in explanations:
        assert isinstance(exp, Explanation)
        assert set(exp.explanation_data["feature_attributions"].keys()) == set(feature_names)
