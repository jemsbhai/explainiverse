"""Accuracy and output-contract tests for the TreeSHAP wrapper."""

import numpy as np
import pytest
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)

from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer


@pytest.fixture
def binary_data():
    X = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 1.0],
            [-0.5, 0.0],
            [0.5, 1.0],
            [1.0, 0.0],
            [2.0, 1.0],
        ]
    )
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_default_target_maps_string_labels_through_model_classes(binary_data):
    X, y = binary_data
    labels = np.where(y == 1, "positive", "negative")
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, labels)
    explainer = TreeShapExplainer(
        model,
        feature_names=["signal", "noise"],
        class_names=["negative-name", "positive-name"],
    )

    explanation = explainer.explain(X[-1])

    predicted_label = model.predict(X[-1:].copy())[0]
    predicted_index = int(np.flatnonzero(model.classes_ == predicted_label)[0])
    assert explanation.target_class == ["negative-name", "positive-name"][predicted_index]
    assert explanation.explanation_data["class_index"] == predicted_index


def test_default_target_maps_noncontiguous_numeric_labels(binary_data):
    X, y = binary_data
    labels = np.where(y == 1, 20, 10)
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, labels)
    explainer = TreeShapExplainer(
        model,
        feature_names=["signal", "noise"],
        class_names=["ten", "twenty"],
    )

    negative = explainer.explain(X[0])
    positive = explainer.explain(X[-1])

    assert negative.target_class == "ten"
    assert negative.explanation_data["class_index"] == 0
    assert positive.target_class == "twenty"
    assert positive.explanation_data["class_index"] == 1


@pytest.mark.parametrize("target_class, sign", [(0, -1.0), (1, 1.0)])
def test_single_margin_binary_outputs_are_class_specific_and_additive(
    binary_data, target_class, sign
):
    X, y = binary_data
    model = GradientBoostingClassifier(random_state=0).fit(X, y)
    explainer = TreeShapExplainer(
        model,
        feature_names=["signal", "noise"],
        class_names=["negative", "positive"],
    )

    explanation = explainer.explain(X[-1], target_class=target_class)
    data = explanation.explanation_data
    expected_margin = sign * float(model.decision_function(X[-1:])[0])

    assert explanation.target_class == ["negative", "positive"][target_class]
    assert data["class_index"] == target_class
    assert data["output_space"] == "raw_margin"
    assert data["explained_value"] == pytest.approx(expected_margin, abs=1e-6)
    assert data["model_reference_value"] == pytest.approx(expected_margin, abs=1e-6)
    assert data["additivity_residual"] == pytest.approx(0.0, abs=1e-6)


def test_interventional_mode_requires_background_data(binary_data):
    X, y = binary_data
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)

    with pytest.raises(ValueError, match="background_data"):
        TreeShapExplainer(
            model,
            feature_names=["signal", "noise"],
            class_names=["negative", "positive"],
            feature_perturbation="interventional",
        )


def test_interventional_interactions_are_rejected_instead_of_returning_zeros(binary_data):
    X, y = binary_data
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    explainer = TreeShapExplainer(
        model,
        feature_names=["signal", "noise"],
        class_names=["negative", "positive"],
        background_data=X,
        feature_perturbation="interventional",
    )

    ordinary = explainer.explain(X[-1], target_class=1)
    assert ordinary.explanation_data["explained_value"] == pytest.approx(
        model.predict_proba(X[-1:])[0, 1], abs=1e-6
    )
    with pytest.raises(ValueError, match="unsupported.*interventional"):
        explainer.explain_interactions(X[-1], target_class=1)


def test_multioutput_regression_is_rejected_consistently(binary_data):
    X, _ = binary_data
    y = np.column_stack((X[:, 0] + X[:, 1], X[:, 0] - X[:, 1]))
    model = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)

    with pytest.raises(ValueError, match="single-output regression"):
        TreeShapExplainer(
            model,
            feature_names=["signal", "noise"],
            class_names=["first", "second"],
            task="regression",
        )


def test_log_loss_output_is_rejected_until_labels_are_part_of_api(binary_data):
    X, y = binary_data
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)

    with pytest.raises(ValueError, match="log_loss"):
        TreeShapExplainer(
            model,
            feature_names=["signal", "noise"],
            class_names=["negative", "positive"],
            background_data=X,
            feature_perturbation="interventional",
            model_output="log_loss",
        )


def test_xgboost_31_vector_intercept_is_preserved_and_additive():
    """SHAP 0.50+ must retain each XGBoost output's learned intercept."""
    xgboost = pytest.importorskip("xgboost")
    shap = pytest.importorskip("shap")
    from packaging.version import Version

    if Version(xgboost.__version__) < Version("3.1"):
        pytest.skip("XGBoost vector-valued base_score was introduced in 3.1")
    if Version(shap.__version__) < Version("0.50"):
        pytest.fail(
            "XGBoost >=3.1 requires SHAP >=0.50; older SHAP cannot parse "
            "the vector-valued base_score"
        )

    rng = np.random.default_rng(12)
    X = rng.normal(size=(90, 3)).astype(np.float32)
    # Deliberately unbalanced so a scalar-mean intercept patch cannot pass.
    y = np.repeat(np.array([0, 1, 2]), [60, 20, 10])
    rng.shuffle(y)
    model = xgboost.XGBClassifier(
        n_estimators=8,
        max_depth=2,
        learning_rate=0.2,
        random_state=4,
    ).fit(X, y)
    explainer = TreeShapExplainer(
        model,
        feature_names=["a", "b", "c"],
        class_names=["zero", "one", "two"],
    )

    raw_margins = model.predict(X[:3], output_margin=True)
    for sample_index in range(3):
        for class_index in range(3):
            explanation = explainer.explain(X[sample_index], target_class=class_index)
            data = explanation.explanation_data
            assert data["explained_value"] == pytest.approx(
                raw_margins[sample_index, class_index], abs=2e-5
            )
            assert data["additivity_residual"] == pytest.approx(0.0, abs=2e-5)
