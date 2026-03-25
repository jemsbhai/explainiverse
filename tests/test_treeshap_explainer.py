# tests/test_treeshap_explainer.py
"""
Tests for TreeSHAP explainer wrapper.

Thorough tests covering value correctness, key matching, class selection,
batch consistency, and edge cases.

Reference:
    Lundberg et al., 2018 — "Consistent Individualized Feature Attribution
    for Tree Ensembles." arXiv:1802.03888.
"""

import pytest
import numpy as np
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from explainiverse.core.explanation import Explanation


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def iris_rf_setup():
    """Iris dataset with RandomForestClassifier (3-class)."""
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = list(iris.feature_names)
    class_names = iris.target_names.tolist()

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)

    return X, y, feature_names, class_names, model


@pytest.fixture
def binary_rf_setup():
    """Binary classification with RandomForestClassifier."""
    X, y = make_classification(
        n_samples=150, n_features=6, n_classes=2,
        n_informative=4, random_state=42
    )
    feature_names = [f"feature_{i}" for i in range(6)]
    class_names = ["negative", "positive"]

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)

    return X, y, feature_names, class_names, model


@pytest.fixture
def regression_rf_setup():
    """Regression with RandomForestRegressor."""
    X, y = make_regression(
        n_samples=100, n_features=5, noise=0.3, random_state=42
    )
    feature_names = [f"reg_f{i}" for i in range(5)]

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    return X, y, feature_names, model


# ──────────────────────────────────────────────
# Core Functionality Tests
# ──────────────────────────────────────────────

class TestTreeShapBasic:
    """Basic TreeSHAP functionality."""

    def test_explain_returns_explanation(self, iris_rf_setup):
        """explain() returns a valid Explanation object."""
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer

        X, y, feature_names, class_names, model = iris_rf_setup
        explainer = TreeShapExplainer(model=model, feature_names=feature_names,
                                      class_names=class_names)

        explanation = explainer.explain(X[0])
        assert isinstance(explanation, Explanation)
        assert explanation.explainer_name == "TreeSHAP"

    def test_attribution_keys_match_feature_names(self, iris_rf_setup):
        """Critical: attribution dict keys must be original feature names."""
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer

        X, y, feature_names, class_names, model = iris_rf_setup
        explainer = TreeShapExplainer(model=model, feature_names=feature_names,
                                      class_names=class_names)

        # Test across all three Iris classes
        for i in [0, 50, 100]:
            explanation = explainer.explain(X[i])
            keys = set(explanation.explanation_data["feature_attributions"].keys())
            assert keys == set(feature_names), \
                f"Instance {i}: keys {keys} != feature names {set(feature_names)}"

    def test_attribution_count_matches_features(self, iris_rf_setup):
        """Number of attributions equals number of features."""
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer

        X, y, feature_names, class_names, model = iris_rf_setup
        explainer = TreeShapExplainer(model=model, feature_names=feature_names,
                                      class_names=class_names)

        explanation = explainer.explain(X[0])
        assert len(explanation.explanation_data["feature_attributions"]) == len(feature_names)

    def test_attribution_values_are_float(self, iris_rf_setup):
        """All attribution values are floats."""
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer

        X, y, feature_names, class_names, model = iris_rf_setup
        explainer = TreeShapExplainer(model=model, feature_names=feature_names,
                                      class_names=class_names)

        explanation = explainer.explain(X[0])
        for k, v in explanation.explanation_data["feature_attributions"].items():
            assert isinstance(v, float), f"Attribution for '{k}' is {type(v)}"

    def test_base_value_present(self, iris_rf_setup):
        """Explanation includes base_value."""
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer

        X, y, feature_names, class_names, model = iris_rf_setup
        explainer = TreeShapExplainer(model=model, feature_names=feature_names,
                                      class_names=class_names)

        explanation = explainer.explain(X[0])
        assert "base_value" in explanation.explanation_data
        assert isinstance(explanation.explanation_data["base_value"], float)

    def test_shap_values_raw_length(self, iris_rf_setup):
        """shap_values_raw has exactly n_features entries."""
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer

        X, y, feature_names, class_names, model = iris_rf_setup
        explainer = TreeShapExplainer(model=model, feature_names=feature_names,
                                      class_names=class_names)

        explanation = explainer.explain(X[0])
        raw = explanation.explanation_data["shap_values_raw"]
        assert isinstance(raw, list)
        assert len(raw) == len(feature_names), \
            f"shap_values_raw has {len(raw)} entries, expected {len(feature_names)}"

    def test_shap_values_raw_matches_attributions(self, iris_rf_setup):
        """shap_values_raw values must match feature_attributions values."""
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer

        X, y, feature_names, class_names, model = iris_rf_setup
        explainer = TreeShapExplainer(model=model, feature_names=feature_names,
                                      class_names=class_names)

        explanation = explainer.explain(X[0])
        raw = explanation.explanation_data["shap_values_raw"]
        attributions = explanation.explanation_data["feature_attributions"]

        for i, fname in enumerate(feature_names):
            assert abs(raw[i] - attributions[fname]) < 1e-10, \
                f"raw[{i}]={raw[i]} != attributions['{fname}']={attributions[fname]}"

    def test_feature_names_stored_on_explanation(self, iris_rf_setup):
        """Explanation must have feature_names attribute for evaluation metrics."""
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer

        X, y, feature_names, class_names, model = iris_rf_setup
        explainer = TreeShapExplainer(model=model, feature_names=feature_names,
                                      class_names=class_names)

        explanation = explainer.explain(X[0])
        assert hasattr(explanation, "feature_names"), \
            "Explanation missing feature_names — evaluation metrics will fail"
        assert explanation.feature_names == feature_names

    def test_deterministic(self, iris_rf_setup):
        """Same input produces same output."""
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer

        X, y, feature_names, class_names, model = iris_rf_setup
        explainer = TreeShapExplainer(model=model, feature_names=feature_names,
                                      class_names=class_names)

        v1 = list(explainer.explain(X[0]).explanation_data["feature_attributions"].values())
        v2 = list(explainer.explain(X[0]).explanation_data["feature_attributions"].values())
        np.testing.assert_array_almost_equal(v1, v2, decimal=10)

    def test_rejects_non_tree_model(self):
        """TreeSHAP raises error for non-tree models."""
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer

        iris = load_iris()
        model = LogisticRegression(max_iter=200).fit(iris.data, iris.target)

        with pytest.raises(ValueError, match="tree-based model"):
            TreeShapExplainer(model=model, feature_names=list(iris.feature_names),
                              class_names=iris.target_names.tolist())


# ──────────────────────────────────────────────
# Target Class Selection Tests
# ──────────────────────────────────────────────

class TestTreeShapClassSelection:
    """Tests verifying correct target class selection."""

    def test_target_class_matches_prediction(self, iris_rf_setup):
        """target_class must match model's prediction when target_class=None."""
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer

        X, y, feature_names, class_names, model = iris_rf_setup
        explainer = TreeShapExplainer(model=model, feature_names=feature_names,
                                      class_names=class_names)

        for i in [0, 50, 100]:
            explanation = explainer.explain(X[i])
            predicted_class = int(model.predict(X[i:i+1])[0])
            expected_label = class_names[predicted_class]
            assert explanation.target_class == expected_label, \
                f"Instance {i}: got '{explanation.target_class}', expected '{expected_label}'"

    def test_explicit_target_class(self, iris_rf_setup):
        """Explicit target_class parameter is respected."""
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer

        X, y, feature_names, class_names, model = iris_rf_setup
        explainer = TreeShapExplainer(model=model, feature_names=feature_names,
                                      class_names=class_names)

        for tc in [0, 1, 2]:
            explanation = explainer.explain(X[0], target_class=tc)
            assert explanation.target_class == class_names[tc]

    def test_different_classes_give_different_values(self, iris_rf_setup):
        """Different target classes should produce different SHAP values."""
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer

        X, y, feature_names, class_names, model = iris_rf_setup
        explainer = TreeShapExplainer(model=model, feature_names=feature_names,
                                      class_names=class_names)

        vals_0 = list(explainer.explain(X[0], target_class=0)
                       .explanation_data["feature_attributions"].values())
        vals_1 = list(explainer.explain(X[0], target_class=1)
                       .explanation_data["feature_attributions"].values())

        assert not np.allclose(vals_0, vals_1, atol=1e-10), \
            "SHAP values identical for different classes — class selection broken"


# ──────────────────────────────────────────────
# Value Correctness Tests
# ──────────────────────────────────────────────

class TestTreeShapValueCorrectness:
    """Tests verifying SHAP values are numerically correct."""

    def test_values_match_direct_shap_library(self, iris_rf_setup):
        """Wrapper attributions must match direct SHAP library access."""
        import shap
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer

        X, y, feature_names, class_names, model = iris_rf_setup
        explainer = TreeShapExplainer(model=model, feature_names=feature_names,
                                      class_names=class_names)

        # Pick instance where predicted class is not 0
        for i in range(len(X)):
            pred = int(model.predict(X[i:i+1])[0])
            if pred != 0:
                break

        explanation = explainer.explain(X[i])

        # Get values directly from SHAP
        direct_explainer = shap.TreeExplainer(model)
        raw_sv = direct_explainer.shap_values(X[i:i+1])

        if isinstance(raw_sv, list):
            expected_vals = raw_sv[pred][0]
        else:
            raw_arr = np.asarray(raw_sv)
            if raw_arr.ndim == 3:
                expected_vals = raw_arr[0, :, pred]
            else:
                expected_vals = raw_arr[0]

        for j, fname in enumerate(feature_names):
            wrapper_val = explanation.explanation_data["feature_attributions"][fname]
            assert abs(wrapper_val - float(expected_vals[j])) < 1e-10, \
                f"Feature '{fname}': wrapper={wrapper_val}, expected={float(expected_vals[j])}"

    def test_additivity(self, iris_rf_setup):
        """SHAP values + base_value should approximately equal prediction."""
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer

        X, y, feature_names, class_names, model = iris_rf_setup
        explainer = TreeShapExplainer(model=model, feature_names=feature_names,
                                      class_names=class_names)

        explanation = explainer.explain(X[0])
        pred_class = int(model.predict(X[0:1])[0])

        shap_sum = sum(explanation.explanation_data["feature_attributions"].values())
        base_val = explanation.explanation_data["base_value"]

        # For tree models, base_value + sum(shap_values) ≈ model output for that class
        model_output = model.predict_proba(X[0:1])[0][pred_class]

        assert abs((base_val + shap_sum) - model_output) < 0.05, \
            f"Additivity check failed: base({base_val:.4f}) + shap_sum({shap_sum:.4f}) " \
            f"= {base_val + shap_sum:.4f} != model_output({model_output:.4f})"

    def test_values_are_finite(self, iris_rf_setup):
        """All SHAP values must be finite."""
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer

        X, y, feature_names, class_names, model = iris_rf_setup
        explainer = TreeShapExplainer(model=model, feature_names=feature_names,
                                      class_names=class_names)

        for i in [0, 50, 100]:
            explanation = explainer.explain(X[i])
            for fname, val in explanation.explanation_data["feature_attributions"].items():
                assert np.isfinite(val), f"Non-finite value for '{fname}': {val}"


# ──────────────────────────────────────────────
# Batch Tests
# ──────────────────────────────────────────────

class TestTreeShapBatch:
    """Tests for explain_batch correctness."""

    def test_batch_count(self, iris_rf_setup):
        """explain_batch returns correct number of explanations."""
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer

        X, y, feature_names, class_names, model = iris_rf_setup
        explainer = TreeShapExplainer(model=model, feature_names=feature_names,
                                      class_names=class_names)

        explanations = explainer.explain_batch(X[:5])
        assert len(explanations) == 5

    def test_batch_keys_match_feature_names(self, iris_rf_setup):
        """Each batch explanation must have correct attribution keys."""
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer

        X, y, feature_names, class_names, model = iris_rf_setup
        explainer = TreeShapExplainer(model=model, feature_names=feature_names,
                                      class_names=class_names)

        explanations = explainer.explain_batch(X[:5])
        for i, exp in enumerate(explanations):
            keys = set(exp.explanation_data["feature_attributions"].keys())
            assert keys == set(feature_names), \
                f"Batch instance {i}: keys {keys} != feature names"

    def test_batch_shap_values_raw_length(self, iris_rf_setup):
        """Each batch explanation must have correct shap_values_raw length."""
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer

        X, y, feature_names, class_names, model = iris_rf_setup
        explainer = TreeShapExplainer(model=model, feature_names=feature_names,
                                      class_names=class_names)

        explanations = explainer.explain_batch(X[:5])
        for i, exp in enumerate(explanations):
            raw = exp.explanation_data["shap_values_raw"]
            assert len(raw) == len(feature_names), \
                f"Batch instance {i}: raw length {len(raw)} != {len(feature_names)}"

    def test_batch_matches_individual(self, iris_rf_setup):
        """Batch results must match individual explain() results."""
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer

        X, y, feature_names, class_names, model = iris_rf_setup
        explainer = TreeShapExplainer(model=model, feature_names=feature_names,
                                      class_names=class_names)

        batch_explanations = explainer.explain_batch(X[:3], target_class=0)

        for i in range(3):
            individual = explainer.explain(X[i], target_class=0)
            batch_vals = list(batch_explanations[i].explanation_data["feature_attributions"].values())
            indiv_vals = list(individual.explanation_data["feature_attributions"].values())
            np.testing.assert_array_almost_equal(batch_vals, indiv_vals, decimal=10,
                err_msg=f"Batch instance {i} differs from individual explain()")

    def test_batch_target_class_not_hardcoded(self, iris_rf_setup):
        """Batch explain must use predicted class when target_class=None,
        NOT hardcode class 0. This is a regression test for the bug where
        explain_batch always used tc=0."""
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer

        X, y, feature_names, class_names, model = iris_rf_setup
        explainer = TreeShapExplainer(model=model, feature_names=feature_names,
                                      class_names=class_names)

        # Find an instance where the predicted class is NOT 0
        for i in range(len(X)):
            pred = int(model.predict(X[i:i+1])[0])
            if pred != 0:
                break

        # explain_batch with target_class=None should use predicted class
        explanations = explainer.explain_batch(X[i:i+1])
        expected_label = class_names[pred]
        assert explanations[0].target_class == expected_label, \
            f"Batch used '{explanations[0].target_class}' instead of " \
            f"predicted '{expected_label}' — hardcoded class 0 bug"

    def test_batch_feature_names_stored(self, iris_rf_setup):
        """Each batch explanation must have feature_names attribute."""
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer

        X, y, feature_names, class_names, model = iris_rf_setup
        explainer = TreeShapExplainer(model=model, feature_names=feature_names,
                                      class_names=class_names)

        explanations = explainer.explain_batch(X[:3])
        for i, exp in enumerate(explanations):
            assert hasattr(exp, "feature_names"), \
                f"Batch instance {i}: missing feature_names"
            assert exp.feature_names == feature_names


# ──────────────────────────────────────────────
# Binary Classification Tests
# ──────────────────────────────────────────────

class TestTreeShapBinary:
    """TreeSHAP with binary classifiers."""

    def test_binary_rf(self, binary_rf_setup):
        """TreeSHAP works with binary RandomForest."""
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer

        X, y, feature_names, class_names, model = binary_rf_setup
        explainer = TreeShapExplainer(model=model, feature_names=feature_names,
                                      class_names=class_names)

        explanation = explainer.explain(X[0])
        assert set(explanation.explanation_data["feature_attributions"].keys()) == set(feature_names)
        assert len(explanation.explanation_data["shap_values_raw"]) == len(feature_names)

    def test_binary_gradient_boosting(self):
        """TreeSHAP works with GradientBoostingClassifier."""
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer

        X, y = make_classification(n_samples=150, n_features=5,
                                    n_classes=2, random_state=42)
        feature_names = [f"f{i}" for i in range(5)]
        model = GradientBoostingClassifier(n_estimators=30, random_state=42)
        model.fit(X, y)

        explainer = TreeShapExplainer(model=model, feature_names=feature_names,
                                      class_names=["neg", "pos"])
        explanation = explainer.explain(X[0])

        assert set(explanation.explanation_data["feature_attributions"].keys()) == set(feature_names)
        assert len(explanation.explanation_data["shap_values_raw"]) == len(feature_names)


# ──────────────────────────────────────────────
# Regression Tests
# ──────────────────────────────────────────────

class TestTreeShapRegression:
    """TreeSHAP with regression models."""

    def test_regression_rf(self, regression_rf_setup):
        """TreeSHAP works with RandomForestRegressor."""
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer

        X, y, feature_names, model = regression_rf_setup
        explainer = TreeShapExplainer(model=model, feature_names=feature_names,
                                      class_names=["output"], task="regression")

        explanation = explainer.explain(X[0])
        assert set(explanation.explanation_data["feature_attributions"].keys()) == set(feature_names)
        assert len(explanation.explanation_data["shap_values_raw"]) == len(feature_names)

        for val in explanation.explanation_data["feature_attributions"].values():
            assert np.isfinite(val)


# ──────────────────────────────────────────────
# Adapter Tests
# ──────────────────────────────────────────────

class TestTreeShapAdapter:
    """TreeSHAP with model adapters."""

    def test_accepts_sklearn_adapter(self, iris_rf_setup):
        """TreeSHAP extracts raw model from SklearnAdapter."""
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer
        from explainiverse.adapters.sklearn_adapter import SklearnAdapter

        X, y, feature_names, class_names, model = iris_rf_setup
        adapter = SklearnAdapter(model, class_names=class_names)

        explainer = TreeShapExplainer(model=adapter, feature_names=feature_names,
                                      class_names=class_names)
        explanation = explainer.explain(X[0])

        assert isinstance(explanation, Explanation)
        assert set(explanation.explanation_data["feature_attributions"].keys()) == set(feature_names)


# ──────────────────────────────────────────────
# XGBoost Tests
# ──────────────────────────────────────────────

class TestTreeShapXGBoost:
    """TreeSHAP with XGBoost models."""

    def test_xgboost_multiclass(self):
        """TreeSHAP works with XGBoost multiclass."""
        from xgboost import XGBClassifier
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer

        iris = load_iris()
        X, y = iris.data, iris.target
        feature_names = list(iris.feature_names)
        class_names = iris.target_names.tolist()

        model = XGBClassifier(n_estimators=30, random_state=42,
                              eval_metric='mlogloss')
        model.fit(X, y)

        explainer = TreeShapExplainer(model=model, feature_names=feature_names,
                                      class_names=class_names)
        explanation = explainer.explain(X[0])

        assert set(explanation.explanation_data["feature_attributions"].keys()) == set(feature_names)
        assert len(explanation.explanation_data["shap_values_raw"]) == len(feature_names)
        assert explanation.target_class in class_names


# ──────────────────────────────────────────────
# Interaction Tests
# ──────────────────────────────────────────────

class TestTreeShapInteractions:
    """TreeSHAP interaction value tests."""

    def test_interaction_matrix_shape(self, iris_rf_setup):
        """Interaction matrix is n_features x n_features."""
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer

        X, y, feature_names, class_names, model = iris_rf_setup
        explainer = TreeShapExplainer(model=model, feature_names=feature_names,
                                      class_names=class_names)

        explanation = explainer.explain_interactions(X[0])
        matrix = explanation.explanation_data["interaction_matrix"]
        n = len(feature_names)
        assert len(matrix) == n
        assert len(matrix[0]) == n

    def test_interaction_main_effects_keys(self, iris_rf_setup):
        """Interaction main effects use original feature names."""
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer

        X, y, feature_names, class_names, model = iris_rf_setup
        explainer = TreeShapExplainer(model=model, feature_names=feature_names,
                                      class_names=class_names)

        explanation = explainer.explain_interactions(X[0])
        main_effects = explanation.explanation_data["feature_attributions"]
        assert set(main_effects.keys()) == set(feature_names)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
