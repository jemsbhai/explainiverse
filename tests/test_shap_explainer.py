# tests/test_shap_explainer.py
"""
Tests for SHAP (KernelSHAP) explainer wrapper.

Reference:
    Lundberg & Lee, 2017 — "A Unified Approach to Interpreting Model
    Predictions." NeurIPS 2017.
"""

import pytest
import numpy as np
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier

from explainiverse.adapters.sklearn_adapter import SklearnAdapter
from explainiverse.explainers.attribution.shap_wrapper import ShapExplainer
from explainiverse.core.explanation import Explanation


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def iris_setup():
    """Standard Iris multiclass setup for SHAP tests."""
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = list(iris.feature_names)
    class_names = iris.target_names.tolist()

    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X, y)
    adapter = SklearnAdapter(model, class_names=class_names)

    explainer = ShapExplainer(
        model=adapter,
        background_data=X[:30],
        feature_names=feature_names,
        class_names=class_names
    )
    return X, y, feature_names, class_names, adapter, explainer


@pytest.fixture
def binary_setup():
    """Binary classification setup."""
    X, y = make_classification(
        n_samples=100, n_features=5, n_classes=2,
        n_informative=3, random_state=42
    )
    feature_names = [f"feature_{i}" for i in range(5)]
    class_names = ["class_0", "class_1"]

    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X, y)
    adapter = SklearnAdapter(model, class_names=class_names)

    explainer = ShapExplainer(
        model=adapter,
        background_data=X[:20],
        feature_names=feature_names,
        class_names=class_names
    )
    return X, y, feature_names, class_names, adapter, explainer


@pytest.fixture
def regression_setup():
    """Linear regression setup."""
    X, y = make_regression(
        n_samples=100, n_features=4, noise=0.3, random_state=99
    )
    feature_names = [f"reg_f{i}" for i in range(4)]

    model = LinearRegression()
    model.fit(X, y)
    adapter = SklearnAdapter(model, class_names=None)

    explainer = ShapExplainer(
        model=adapter,
        background_data=X[:30],
        feature_names=feature_names,
        class_names=["target"]
    )
    return X, y, feature_names, adapter, explainer


@pytest.fixture
def multiclass_5_setup():
    """5-class RandomForest setup."""
    X, y = make_classification(
        n_samples=120, n_features=6, n_classes=5,
        n_informative=4, n_redundant=0,
        n_clusters_per_class=1, random_state=123
    )
    feature_names = [f"f{i}" for i in range(6)]
    class_names = [f"class_{i}" for i in range(5)]

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    adapter = SklearnAdapter(model, class_names=class_names)

    explainer = ShapExplainer(
        model=adapter,
        background_data=X[:20],
        feature_names=feature_names,
        class_names=class_names
    )
    return X, y, feature_names, class_names, adapter, explainer


# ──────────────────────────────────────────────
# Core Functionality Tests
# ──────────────────────────────────────────────

class TestShapBasic:
    """Basic SHAP explainer functionality."""

    def test_explain_returns_explanation(self, iris_setup):
        """explain() returns a valid Explanation object."""
        X, y, feature_names, class_names, adapter, explainer = iris_setup

        explanation = explainer.explain(X[0])
        assert isinstance(explanation, Explanation)
        assert explanation.explainer_name == "SHAP"

    def test_attribution_keys_match_feature_names(self, iris_setup):
        """Critical: attribution dict keys must be original feature names."""
        X, y, feature_names, class_names, adapter, explainer = iris_setup

        for i in [0, 50, 100]:  # One from each Iris class
            explanation = explainer.explain(X[i])
            keys = set(explanation.explanation_data["feature_attributions"].keys())
            assert keys == set(feature_names), \
                f"Instance {i}: keys {keys} != feature names {set(feature_names)}"

    def test_attribution_count_matches_features(self, iris_setup):
        """Number of attributions equals number of features."""
        X, y, feature_names, class_names, adapter, explainer = iris_setup

        explanation = explainer.explain(X[0])
        attributions = explanation.explanation_data["feature_attributions"]
        assert len(attributions) == len(feature_names)

    def test_attribution_values_are_float(self, iris_setup):
        """All attribution values are floats."""
        X, y, feature_names, class_names, adapter, explainer = iris_setup

        explanation = explainer.explain(X[0])
        for k, v in explanation.explanation_data["feature_attributions"].items():
            assert isinstance(v, float), \
                f"Attribution for '{k}' is {type(v)}, expected float"

    def test_target_class_is_valid(self, iris_setup):
        """target_class is a valid class name matching model prediction."""
        X, y, feature_names, class_names, adapter, explainer = iris_setup

        explanation = explainer.explain(X[0])
        assert explanation.target_class in class_names

        # Should match model's top prediction
        preds = adapter.predict(X[0:1])
        predicted_label = class_names[np.argmax(preds[0])]
        assert explanation.target_class == predicted_label

    def test_expected_value_present(self, iris_setup):
        """Explanation includes expected_value from SHAP."""
        X, y, feature_names, class_names, adapter, explainer = iris_setup

        explanation = explainer.explain(X[0])
        assert "expected_value" in explanation.explanation_data
        assert isinstance(explanation.explanation_data["expected_value"], float)

    def test_shap_values_raw_length(self, iris_setup):
        """shap_values_raw has exactly n_features entries (not flattened 3D)."""
        X, y, feature_names, class_names, adapter, explainer = iris_setup

        explanation = explainer.explain(X[0])
        raw = explanation.explanation_data["shap_values_raw"]
        assert isinstance(raw, list)
        assert len(raw) == len(feature_names), \
            f"shap_values_raw has {len(raw)} entries, expected {len(feature_names)}"

    def test_shap_values_raw_matches_attributions(self, iris_setup):
        """shap_values_raw values must match feature_attributions values."""
        X, y, feature_names, class_names, adapter, explainer = iris_setup

        explanation = explainer.explain(X[0])
        raw = explanation.explanation_data["shap_values_raw"]
        attributions = explanation.explanation_data["feature_attributions"]

        for i, fname in enumerate(feature_names):
            assert abs(raw[i] - attributions[fname]) < 1e-10, \
                f"raw[{i}]={raw[i]} != attributions['{fname}']={attributions[fname]}"

    def test_feature_names_stored(self, iris_setup):
        """Explanation stores feature_names attribute."""
        X, y, feature_names, class_names, adapter, explainer = iris_setup

        explanation = explainer.explain(X[0])
        assert hasattr(explanation, "feature_names")
        assert explanation.feature_names == feature_names

    def test_deterministic(self, iris_setup):
        """Same input produces same output."""
        X, y, feature_names, class_names, adapter, explainer = iris_setup

        exp1 = explainer.explain(X[0])
        exp2 = explainer.explain(X[0])
        v1 = list(exp1.explanation_data["feature_attributions"].values())
        v2 = list(exp2.explanation_data["feature_attributions"].values())
        np.testing.assert_array_almost_equal(v1, v2, decimal=8)


# ──────────────────────────────────────────────
# Value Correctness Tests
# ──────────────────────────────────────────────

class TestShapValueCorrectness:
    """Tests that verify SHAP values are numerically correct, not just
    structurally valid. These catch bugs like the 3D array flattening
    issue where keys looked right but values were wrong."""

    def test_attributions_correspond_to_correct_class(self, iris_setup):
        """SHAP values must be for the predicted class, not hardcoded class 0.

        Verify by comparing wrapper output to direct SHAP library access.
        """
        X, y, feature_names, class_names, adapter, explainer = iris_setup

        # Pick an instance where predicted class is NOT 0
        for i in range(len(X)):
            preds = adapter.predict(X[i:i+1])
            pred_class = int(np.argmax(preds[0]))
            if pred_class != 0:
                break

        explanation = explainer.explain(X[i])

        # Verify target class matches prediction
        assert explanation.target_class == class_names[pred_class]

        # Get raw SHAP values directly from the library
        raw_sv = explainer.explainer.shap_values(X[i:i+1])

        # Extract the correct class values directly
        if isinstance(raw_sv, list):
            expected_vals = raw_sv[pred_class][0]
        else:
            raw_arr = np.asarray(raw_sv)
            if raw_arr.ndim == 3:
                expected_vals = raw_arr[0, :, pred_class]
            else:
                expected_vals = raw_arr[0]

        # Compare wrapper attributions to direct extraction
        for j, fname in enumerate(feature_names):
            wrapper_val = explanation.explanation_data["feature_attributions"][fname]
            assert abs(wrapper_val - float(expected_vals[j])) < 1e-10, \
                f"Feature '{fname}': wrapper={wrapper_val}, " \
                f"expected={float(expected_vals[j])} (class {pred_class})"

    def test_3d_ndarray_not_flattened_across_classes(self, iris_setup):
        """Verify that 3D SHAP output (samples, features, classes) is sliced
        correctly — not flattened, which would mix features and classes.

        This is the specific regression test for the bug where flatten()
        on a (4, 3) array produced 12 values and the first 4 mixed
        feature 0's class values with feature 1's values.
        """
        X, y, feature_names, class_names, adapter, explainer = iris_setup

        raw_sv = explainer.explainer.shap_values(X[0:1])
        raw_arr = np.asarray(raw_sv)

        # This test is only meaningful if SHAP returns a 3D array
        if isinstance(raw_sv, list) or raw_arr.ndim != 3:
            pytest.skip("SHAP returned list or non-3D array; 3D test not applicable")

        n_features = raw_arr.shape[1]
        n_classes = raw_arr.shape[2]

        # Get wrapper output
        explanation = explainer.explain(X[0])
        raw_output = explanation.explanation_data["shap_values_raw"]

        # Must have exactly n_features values, NOT n_features * n_classes
        assert len(raw_output) == n_features, \
            f"shap_values_raw has {len(raw_output)} values, expected {n_features}. " \
            f"Possible flatten bug: {n_features} * {n_classes} = {n_features * n_classes}"

    def test_each_class_gets_different_attributions(self, iris_setup):
        """Different target classes should generally produce different
        SHAP values for the same instance, confirming class selection works."""
        X, y, feature_names, class_names, adapter, explainer = iris_setup

        raw_sv = explainer.explainer.shap_values(X[0:1])
        raw_arr = np.asarray(raw_sv)

        if isinstance(raw_sv, list) and len(raw_sv) >= 2:
            vals_class0 = raw_sv[0][0]
            vals_class1 = raw_sv[1][0]
        elif raw_arr.ndim == 3 and raw_arr.shape[2] >= 2:
            vals_class0 = raw_arr[0, :, 0]
            vals_class1 = raw_arr[0, :, 1]
        else:
            pytest.skip("Cannot test class differentiation with this SHAP format")

        # At least one feature should differ between classes
        assert not np.allclose(vals_class0, vals_class1, atol=1e-10), \
            "SHAP values identical across classes — class selection may be broken"


# ──────────────────────────────────────────────
# Multiclass Tests
# ──────────────────────────────────────────────

class TestShapMulticlass:
    """SHAP with multiclass classifiers."""

    def test_logistic_regression_iris(self, iris_setup):
        """SHAP works with LogisticRegression on Iris (3-class)."""
        X, y, feature_names, class_names, adapter, explainer = iris_setup

        explanation = explainer.explain(X[10], top_labels=2)
        assert explanation.target_class in class_names
        assert len(explanation.explanation_data["feature_attributions"]) == len(feature_names)
        assert set(explanation.explanation_data["feature_attributions"].keys()) == set(feature_names)

    def test_random_forest_5class(self, multiclass_5_setup):
        """SHAP works with RandomForest on a 5-class problem."""
        X, y, feature_names, class_names, adapter, explainer = multiclass_5_setup

        explanation = explainer.explain(X[5])
        assert explanation.target_class in class_names
        assert set(explanation.explanation_data["feature_attributions"].keys()) == set(feature_names)
        assert len(explanation.explanation_data["feature_attributions"]) == len(feature_names)
        assert len(explanation.explanation_data["shap_values_raw"]) == len(feature_names)

    def test_multiclass_target_matches_prediction(self, multiclass_5_setup):
        """For multiclass, target_class must match model's top prediction."""
        X, y, feature_names, class_names, adapter, explainer = multiclass_5_setup

        for i in [0, 20, 40, 60, 80]:
            explanation = explainer.explain(X[i])
            preds = adapter.predict(X[i:i+1])
            predicted_label = class_names[np.argmax(preds[0])]
            assert explanation.target_class == predicted_label, \
                f"Instance {i}: got '{explanation.target_class}', " \
                f"expected '{predicted_label}'"


# ──────────────────────────────────────────────
# Binary Classification Tests
# ──────────────────────────────────────────────

class TestShapBinary:
    """SHAP with binary classifiers."""

    def test_binary_classification(self, binary_setup):
        """SHAP works with binary classification."""
        X, y, feature_names, class_names, adapter, explainer = binary_setup

        explanation = explainer.explain(X[0])
        assert explanation.target_class in class_names
        assert set(explanation.explanation_data["feature_attributions"].keys()) == set(feature_names)
        assert len(explanation.explanation_data["feature_attributions"]) == len(feature_names)
        assert len(explanation.explanation_data["shap_values_raw"]) == len(feature_names)

    def test_binary_target_matches_prediction(self, binary_setup):
        """Binary: target_class must match model's top prediction."""
        X, y, feature_names, class_names, adapter, explainer = binary_setup

        for i in [0, 25, 50, 75]:
            explanation = explainer.explain(X[i])
            preds = adapter.predict(X[i:i+1])
            predicted_label = class_names[np.argmax(preds[0])]
            assert explanation.target_class == predicted_label


# ──────────────────────────────────────────────
# Regression Tests
# ──────────────────────────────────────────────

class TestShapRegression:
    """SHAP with regression models."""

    def test_linear_regression(self, regression_setup):
        """SHAP works with LinearRegression."""
        X, y, feature_names, adapter, explainer = regression_setup

        explanation = explainer.explain(X[0])
        attributions = explanation.explanation_data["feature_attributions"]
        assert isinstance(attributions, dict)
        assert set(attributions.keys()) == set(feature_names)
        assert len(attributions) == len(feature_names)
        assert len(explanation.explanation_data["shap_values_raw"]) == len(feature_names)

    def test_regression_values_are_finite(self, regression_setup):
        """Regression SHAP values must be finite (no NaN/Inf)."""
        X, y, feature_names, adapter, explainer = regression_setup

        explanation = explainer.explain(X[0])
        for fname, val in explanation.explanation_data["feature_attributions"].items():
            assert np.isfinite(val), f"Non-finite SHAP value for '{fname}': {val}"


# ──────────────────────────────────────────────
# Batch Tests
# ──────────────────────────────────────────────

class TestShapBatch:
    """SHAP batch explanation tests."""

    def test_batch_explain(self, iris_setup):
        """explain_batch produces correct number of valid explanations."""
        X, y, feature_names, class_names, adapter, explainer = iris_setup

        explanations = explainer.explain_batch(X[:3])
        assert len(explanations) == 3
        for exp in explanations:
            assert isinstance(exp, Explanation)
            assert set(exp.explanation_data["feature_attributions"].keys()) == set(feature_names)
            assert exp.target_class in class_names
            assert len(exp.explanation_data["shap_values_raw"]) == len(feature_names)

    def test_batch_1d_input(self, iris_setup):
        """explain_batch handles 1D input (single instance)."""
        X, y, feature_names, class_names, adapter, explainer = iris_setup

        explanations = explainer.explain_batch(X[0])
        assert len(explanations) == 1
        assert isinstance(explanations[0], Explanation)
        assert len(explanations[0].explanation_data["shap_values_raw"]) == len(feature_names)

    def test_batch_each_instance_correct_class(self, iris_setup):
        """Each instance in a batch should have the correct target class."""
        X, y, feature_names, class_names, adapter, explainer = iris_setup

        # Pick one from each class
        instances = np.array([X[0], X[50], X[100]])
        explanations = explainer.explain_batch(instances)

        for i, exp in enumerate(explanations):
            preds = adapter.predict(instances[i:i+1])
            predicted_label = class_names[np.argmax(preds[0])]
            assert exp.target_class == predicted_label, \
                f"Batch instance {i}: got '{exp.target_class}', expected '{predicted_label}'"


# ──────────────────────────────────────────────
# Global / Cohort SHAP Tests
# ──────────────────────────────────────────────

class TestShapGlobal:
    """Tests for global and cohort-level SHAP analysis."""

    def test_global_feature_importance(self, iris_setup):
        """Global SHAP importance can be computed by aggregating shap_values."""
        X, y, feature_names, class_names, adapter, explainer = iris_setup

        shap_values = explainer.explainer.shap_values(X[:10])

        if isinstance(shap_values, list):
            shap_matrix = np.abs(shap_values[0])
        elif np.asarray(shap_values).ndim == 3:
            shap_matrix = np.abs(np.asarray(shap_values)).mean(axis=-1)
        else:
            shap_matrix = np.abs(np.asarray(shap_values))

        global_importance = np.mean(shap_matrix, axis=0)
        assert len(global_importance) == len(feature_names)
        assert np.all(global_importance >= 0)

    def test_cohort_explanation(self, iris_setup):
        """Cohort-level SHAP works for a subset of data."""
        X, y, feature_names, class_names, adapter, explainer = iris_setup

        cohort_mask = (y == 0)
        X_cohort = X[cohort_mask][:5]

        shap_values = explainer.explainer.shap_values(X_cohort)

        if isinstance(shap_values, list):
            shap_matrix = np.abs(shap_values[0])
        elif np.asarray(shap_values).ndim == 3:
            shap_matrix = np.abs(np.asarray(shap_values)).mean(axis=-1)
        else:
            shap_matrix = np.abs(np.asarray(shap_values))

        cohort_importance = np.mean(shap_matrix, axis=0)
        assert len(cohort_importance) == len(feature_names)
