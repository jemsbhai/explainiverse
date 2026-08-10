"""Contract and analytical tests for the tabular LIME wrapper."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from explainiverse.adapters.sklearn_adapter import SklearnAdapter
from explainiverse.explainers.attribution.lime_wrapper import LimeExplainer


def test_lime_infers_regression_from_adapter_and_supports_standard_output_shape():
    """Default mode follows model.task and never invents class probabilities."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(300, 2))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1]
    raw_model = LinearRegression().fit(X, y)
    adapter = SklearnAdapter(
        raw_model,
        feature_names=["positive", "negative"],
    )
    instance = np.array([1.0, 1.0])

    explanation = LimeExplainer(
        model=adapter,
        training_data=X,
        feature_names=["positive", "negative"],
        class_names=["output"],
    ).explain(instance)

    attributions = explanation.explanation_data["feature_attributions"]
    assert explanation.target_class == "output"
    assert attributions["positive"] > 0.0
    assert attributions["negative"] < 0.0
    assert explanation.explanation_data["mode"] == "regression"
    assert explanation.explanation_data["model_prediction"] == pytest.approx(
        float(raw_model.predict(instance.reshape(1, -1))[0]), rel=1e-12
    )


def test_lime_rejects_mode_that_conflicts_with_model_task():
    X = np.array([[-1.0], [0.0], [1.0]])
    model = SklearnAdapter(LinearRegression().fit(X, X[:, 0]))

    with pytest.raises(ValueError, match="conflicts"):
        LimeExplainer(
            model=model,
            training_data=X,
            feature_names=["x"],
            class_names=["negative", "positive"],
            mode="classification",
        )


def test_lime_rejects_unknown_mode():
    X = np.array([[0.0], [1.0]])
    model = LinearRegression().fit(X, np.array([0.0, 1.0]))

    with pytest.raises(ValueError, match="mode"):
        LimeExplainer(
            model=SklearnAdapter(model, class_names=["output"]),
            training_data=X,
            feature_names=["x"],
            class_names=["output"],
            mode="unsupported",
        )


def test_lime_rejects_feature_name_shape_mismatch():
    X = np.array([[0.0, 1.0], [1.0, 2.0]])
    model = LinearRegression().fit(X, np.array([0.0, 1.0]))

    with pytest.raises(ValueError, match="feature_names"):
        LimeExplainer(
            model=SklearnAdapter(model, class_names=["output"]),
            training_data=X,
            feature_names=["only_one"],
            class_names=["output"],
            mode="regression",
        )
