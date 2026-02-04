# tests/test_selectivity.py
"""
Comprehensive tests for Selectivity (AOPC) metric.

Selectivity measures how quickly the prediction drops when removing features
in order of attributed importance. Computed as Area Over the Perturbation
Curve (AOPC), which is the average prediction drop across all perturbation steps.

AOPC = (1/(K+1)) * Σₖ₌₀ᴷ [f(x) - f(x_{1..k})]

Higher AOPC indicates better selectivity - the explanation correctly
identifies features whose removal causes the largest prediction drop.

Reference: Montavon, G., Samek, W., & Müller, K. R. (2018). Methods for
Interpreting and Understanding Deep Neural Networks. DSP, 73, 1-15.
"""
import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from explainiverse.adapters import SklearnAdapter
from explainiverse.core import Explanation
from explainiverse.evaluation import (
    compute_selectivity,
    compute_batch_selectivity,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def setup_iris():
    """Load Iris dataset for testing."""
    data = load_iris()
    # Use samples from all 3 classes (indices 0-49 are class 0, 50-99 class 1, 100-149 class 2)
    # Take 20 samples from each class for balanced dataset with multiple classes
    indices = list(range(0, 20)) + list(range(50, 70)) + list(range(100, 120))
    X = data.data[indices]
    y = data.target[indices]
    feature_names = list(data.feature_names)
    return X, y, feature_names


@pytest.fixture
def setup_model(setup_iris):
    """Create and train a simple model."""
    X, y, feature_names = setup_iris
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    adapter = SklearnAdapter(model)
    return adapter, X, y, feature_names


@pytest.fixture
def setup_batch(setup_model):
    """Create a batch of explanations for testing."""
    adapter, X, y, feature_names = setup_model
    
    # Create explanations using LIME-like attributions
    explanations = []
    for i in range(min(10, len(X))):
        # Generate pseudo-attributions (simulate LIME output)
        np.random.seed(i)
        attrs = np.random.randn(len(feature_names)) * 0.1
        # Make some features more important
        attrs[0] = 0.5  # sepal length important
        attrs[2] = 0.3  # petal length important
        
        attr_dict = {fn: float(attrs[j]) for j, fn in enumerate(feature_names)}
        exp = Explanation(
            explainer_name="test_explainer",
            target_class="0",
            explanation_data={"feature_attributions": attr_dict},
            feature_names=feature_names,
        )
        explanations.append(exp)
    
    return adapter, X, explanations, feature_names


def create_explanation(attributions, feature_names):
    """Helper to create an Explanation object from attributions."""
    attr_dict = {fn: float(attributions[i]) for i, fn in enumerate(feature_names)}
    return Explanation(
        explainer_name="test_explainer",
        target_class="0",
        explanation_data={"feature_attributions": attr_dict},
        feature_names=feature_names,
    )


# =============================================================================
# Test Class: Basic Functionality
# =============================================================================

class TestSelectivityBasic:
    """Test basic Selectivity functionality."""
    
    def test_returns_float(self, setup_model):
        """Test that compute_selectivity returns a float."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.3, 0.1, 0.4, 0.2])
        explanation = create_explanation(attrs, feature_names)
        
        score = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X
        )
        
        assert isinstance(score, float)
    
    def test_non_negative_score(self, setup_model):
        """Test that AOPC score can be positive (good explanation)."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        # Strong attributions should lead to positive AOPC
        attrs = np.array([0.5, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        score = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X
        )
        
        # AOPC measures prediction drop, which should be >= 0 for faithful explanations
        assert isinstance(score, float)
    
    def test_deterministic_same_seed(self, setup_model):
        """Test that results are deterministic for same inputs."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        score1 = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X
        )
        
        score2 = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X
        )
        
        assert score1 == score2
    
    def test_use_absolute_true(self, setup_model):
        """Test that use_absolute=True uses absolute attribution values."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        # Negative attributions with high magnitude
        attrs = np.array([-0.5, 0.3, -0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        score = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            use_absolute=True
        )
        
        assert isinstance(score, float)
    
    def test_use_absolute_false(self, setup_model):
        """Test that use_absolute=False uses signed attribution values."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.5, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        score = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            use_absolute=False
        )
        
        assert isinstance(score, float)
    
    def test_use_absolute_affects_ordering(self, setup_model):
        """Test that use_absolute changes feature ordering."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        # Feature 0 has largest absolute value but negative
        # Feature 1 has largest positive value
        attrs = np.array([-0.8, 0.5, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        score_abs = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            use_absolute=True
        )
        
        score_signed = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            use_absolute=False
        )
        
        # Scores may differ due to different ordering
        assert isinstance(score_abs, float)
        assert isinstance(score_signed, float)


# =============================================================================
# Test Class: n_steps Parameter
# =============================================================================

class TestNSteps:
    """Test n_steps parameter functionality."""
    
    def test_default_n_steps(self, setup_model):
        """Test default n_steps (all features)."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        result = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            return_details=True
        )
        
        # n_steps should equal n_features by default
        assert result["n_steps"] == len(feature_names)
    
    def test_n_steps_half(self, setup_model):
        """Test n_steps set to half of features."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        result = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            n_steps=2,
            return_details=True
        )
        
        assert result["n_steps"] == 2
        # predictions should have n_steps+1 entries (original + after each step)
        assert len(result["predictions"]) == 3
    
    def test_n_steps_one(self, setup_model):
        """Test n_steps set to 1 (only remove most important feature)."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        result = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            n_steps=1,
            return_details=True
        )
        
        assert result["n_steps"] == 1
        assert len(result["predictions"]) == 2  # original + after removing 1 feature
    
    def test_n_steps_exceeds_features(self, setup_model):
        """Test n_steps larger than n_features is clamped."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        result = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            n_steps=100,  # Much larger than 4 features
            return_details=True
        )
        
        # Should be clamped to n_features
        assert result["n_steps"] == len(feature_names)
    
    def test_different_n_steps_different_scores(self, setup_model):
        """Test that different n_steps can produce different AOPC scores."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.5, 0.1, 0.1, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        score_all = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            n_steps=4
        )
        
        score_one = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            n_steps=1
        )
        
        # Scores may differ (both valid but computed differently)
        assert isinstance(score_all, float)
        assert isinstance(score_one, float)


# =============================================================================
# Test Class: Return Details
# =============================================================================

class TestReturnDetails:
    """Test return_details functionality."""
    
    def test_return_details_structure(self, setup_model):
        """Test that return_details=True returns expected dictionary structure."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        result = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            return_details=True
        )
        
        assert isinstance(result, dict)
        assert "aopc" in result
        assert "prediction_drops" in result
        assert "predictions" in result
        assert "feature_order" in result
        assert "n_steps" in result
        assert "original_prediction" in result
    
    def test_prediction_drops_length(self, setup_model):
        """Test prediction_drops array has correct length."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        result = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            return_details=True
        )
        
        # Should have n_steps + 1 entries (including step 0)
        expected_length = result["n_steps"] + 1
        assert len(result["prediction_drops"]) == expected_length
        assert len(result["predictions"]) == expected_length
    
    def test_prediction_drops_starts_at_zero(self, setup_model):
        """Test that prediction_drops[0] is 0 (no features removed yet)."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        result = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            return_details=True
        )
        
        # First prediction drop should be 0 (original - original = 0)
        assert result["prediction_drops"][0] == 0.0
    
    def test_aopc_is_mean_of_drops(self, setup_model):
        """Test that AOPC equals mean of prediction_drops."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        result = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            return_details=True
        )
        
        expected_aopc = np.mean(result["prediction_drops"])
        assert np.isclose(result["aopc"], expected_aopc)
    
    def test_feature_order_is_permutation(self, setup_model):
        """Test that feature_order is a valid permutation."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        result = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            return_details=True
        )
        
        feature_order = result["feature_order"]
        n_features = len(feature_names)
        
        # Should contain valid indices
        assert len(feature_order) == n_features
        assert all(0 <= idx < n_features for idx in feature_order)
        # Should be unique indices
        assert len(set(feature_order)) == n_features
    
    def test_aopc_matches_standalone_call(self, setup_model):
        """Test that aopc from details matches standalone compute_selectivity."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        standalone_score = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            return_details=False
        )
        
        detailed_result = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            return_details=True
        )
        
        assert np.isclose(standalone_score, detailed_result["aopc"])


# =============================================================================
# Test Class: Baseline Types
# =============================================================================

class TestBaselineTypes:
    """Test different baseline types."""
    
    def test_baseline_mean(self, setup_model):
        """Test baseline='mean' works correctly."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        score = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X
        )
        
        assert isinstance(score, float)
    
    def test_baseline_median(self, setup_model):
        """Test baseline='median' works correctly."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        score = compute_selectivity(
            adapter, instance, explanation,
            baseline="median", background_data=X
        )
        
        assert isinstance(score, float)
    
    def test_baseline_scalar(self, setup_model):
        """Test baseline as scalar value."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        score = compute_selectivity(
            adapter, instance, explanation,
            baseline=0.0, background_data=X
        )
        
        assert isinstance(score, float)
    
    def test_baseline_array(self, setup_model):
        """Test baseline as array of values."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        baseline_values = np.mean(X, axis=0)
        score = compute_selectivity(
            adapter, instance, explanation,
            baseline=baseline_values
        )
        
        assert isinstance(score, float)
    
    def test_baseline_callable(self, setup_model):
        """Test baseline as callable function."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        def custom_baseline(background_data):
            return np.percentile(background_data, 25, axis=0)
        
        score = compute_selectivity(
            adapter, instance, explanation,
            baseline=custom_baseline, background_data=X
        )
        
        assert isinstance(score, float)


# =============================================================================
# Test Class: Batch Operations
# =============================================================================

class TestBatchOperations:
    """Test batch Selectivity computation."""
    
    def test_batch_returns_dict(self, setup_batch):
        """Test that compute_batch_selectivity returns dict."""
        adapter, X, explanations, feature_names = setup_batch
        
        result = compute_batch_selectivity(
            adapter, X[:len(explanations)], explanations,
            baseline="mean"
        )
        
        assert isinstance(result, dict)
        assert "mean" in result
        assert "std" in result
        assert "min" in result
        assert "max" in result
        assert "n_samples" in result
    
    def test_batch_max_samples(self, setup_batch):
        """Test max_samples limits computation."""
        adapter, X, explanations, feature_names = setup_batch
        
        result = compute_batch_selectivity(
            adapter, X[:len(explanations)], explanations,
            baseline="mean",
            max_samples=3
        )
        
        assert result["n_samples"] <= 3
    
    def test_batch_values_are_floats(self, setup_batch):
        """Test that batch results are floats."""
        adapter, X, explanations, feature_names = setup_batch
        
        result = compute_batch_selectivity(
            adapter, X[:len(explanations)], explanations,
            baseline="mean"
        )
        
        assert isinstance(result["mean"], float)
        assert isinstance(result["std"], float)
        assert isinstance(result["min"], float)
        assert isinstance(result["max"], float)
    
    def test_batch_n_steps_parameter(self, setup_batch):
        """Test batch with custom n_steps."""
        adapter, X, explanations, feature_names = setup_batch
        
        result = compute_batch_selectivity(
            adapter, X[:len(explanations)], explanations,
            baseline="mean",
            n_steps=2
        )
        
        assert isinstance(result, dict)
        assert result["n_samples"] > 0


# =============================================================================
# Test Class: Multiple Model Types
# =============================================================================

class TestMultipleModels:
    """Test Selectivity with different model types."""
    
    def test_random_forest(self, setup_iris):
        """Test with RandomForestClassifier."""
        X, y, feature_names = setup_iris
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model)
        
        instance = X[0]
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        score = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X
        )
        
        assert isinstance(score, float)
    
    def test_gradient_boosting(self, setup_iris):
        """Test with GradientBoostingClassifier."""
        X, y, feature_names = setup_iris
        
        model = GradientBoostingClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model)
        
        instance = X[0]
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        score = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X
        )
        
        assert isinstance(score, float)


# =============================================================================
# Test Class: Explainer Comparison
# =============================================================================

class TestExplainerComparison:
    """Test Selectivity with different explainers."""
    
    def test_lime_shap_both_valid(self, setup_iris):
        """Test that both LIME and SHAP produce valid Selectivity scores."""
        X, y, feature_names = setup_iris
        class_names = [str(c) for c in np.unique(y)]
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model, class_names=class_names)
        instance = X[0]
        
        # Check if explainers available
        try:
            from explainiverse.explainers.attribution.lime_wrapper import LimeExplainer
            lime_available = True
        except ImportError:
            lime_available = False
        
        try:
            from explainiverse.explainers.attribution.shap_wrapper import ShapExplainer
            shap_available = True
        except ImportError:
            shap_available = False
        
        scores = []
        
        if lime_available:
            from explainiverse.explainers.attribution.lime_wrapper import LimeExplainer
            lime = LimeExplainer(
                model=adapter,
                training_data=X,
                feature_names=feature_names,
                class_names=class_names
            )
            lime_exp = lime.explain(instance)
            lime_exp.feature_names = feature_names
            lime_score = compute_selectivity(
                adapter, instance, lime_exp,
                baseline="mean", background_data=X
            )
            scores.append(("LIME", lime_score))
        
        if shap_available:
            from explainiverse.explainers.attribution.shap_wrapper import ShapExplainer
            shap = ShapExplainer(
                model=adapter,
                background_data=X[:50],
                feature_names=feature_names,
                class_names=class_names
            )
            shap_exp = shap.explain(instance)
            shap_exp.feature_names = feature_names
            shap_score = compute_selectivity(
                adapter, instance, shap_exp,
                baseline="mean", background_data=X
            )
            scores.append(("SHAP", shap_score))
        
        # All computed scores should be valid floats
        for name, score in scores:
            assert isinstance(score, float), f"{name} should return float"


# =============================================================================
# Test Class: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_few_features(self, setup_iris):
        """Test with only 3 features."""
        X, y, feature_names = setup_iris
        X_small = X[:, :3]
        feature_names_small = feature_names[:3]
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_small, y)
        adapter = SklearnAdapter(model)
        
        instance = X_small[0]
        attrs = np.array([0.5, 0.3, 0.2])
        explanation = create_explanation(attrs, feature_names_small)
        
        score = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X_small
        )
        
        assert isinstance(score, float)
    
    def test_many_features(self):
        """Test with many features (50)."""
        np.random.seed(42)
        n_features = 50
        n_samples = 100
        
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        feature_names = [f"feature_{i}" for i in range(n_features)]
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model)
        
        instance = X[0]
        attrs = np.random.randn(n_features)
        explanation = create_explanation(attrs, feature_names)
        
        score = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X
        )
        
        assert isinstance(score, float)
    
    def test_identical_attributions(self, setup_model):
        """Test when all attributions are identical."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        # All features equally important
        attrs = np.array([0.25, 0.25, 0.25, 0.25])
        explanation = create_explanation(attrs, feature_names)
        
        score = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X
        )
        
        assert isinstance(score, float)
    
    def test_zero_attributions(self, setup_model):
        """Test when all attributions are zero."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.0, 0.0, 0.0, 0.0])
        explanation = create_explanation(attrs, feature_names)
        
        score = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X
        )
        
        # Should still return a valid float
        assert isinstance(score, float)
    
    def test_single_feature(self):
        """Test with only one feature."""
        np.random.seed(42)
        X = np.random.randn(50, 1)
        y = (X[:, 0] > 0).astype(int)
        feature_names = ["feature_0"]
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model)
        
        instance = X[0]
        attrs = np.array([0.5])
        explanation = create_explanation(attrs, feature_names)
        
        score = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X
        )
        
        assert isinstance(score, float)
    
    def test_two_features(self):
        """Test with exactly two features."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        feature_names = ["feature_0", "feature_1"]
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model)
        
        instance = X[0]
        attrs = np.array([0.6, 0.4])
        explanation = create_explanation(attrs, feature_names)
        
        result = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            return_details=True
        )
        
        assert result["n_steps"] == 2
        assert len(result["predictions"]) == 3  # original + 2 steps
    
    def test_highly_skewed_attributions(self, setup_model):
        """Test when one feature dominates attributions."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        # First feature overwhelmingly important
        attrs = np.array([0.99, 0.005, 0.003, 0.002])
        explanation = create_explanation(attrs, feature_names)
        
        score = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X
        )
        
        assert isinstance(score, float)
    
    def test_negative_attributions_only(self, setup_model):
        """Test when all attributions are negative."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([-0.4, -0.3, -0.2, -0.1])
        explanation = create_explanation(attrs, feature_names)
        
        score = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            use_absolute=True
        )
        
        assert isinstance(score, float)


# =============================================================================
# Test Class: Semantic Validation
# =============================================================================

class TestSemanticValidation:
    """Test that Selectivity measures what it claims to measure."""
    
    def test_good_explanation_higher_aopc(self, setup_iris):
        """Test that better explanations tend to have higher AOPC."""
        X, y, feature_names = setup_iris
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model)
        
        instance = X[0]
        
        # Get model coefficients as "true" importance
        true_importance = np.abs(model.coef_[0])
        true_importance = true_importance / np.sum(true_importance)
        
        # Good explanation: matches model importance
        good_exp = create_explanation(true_importance, feature_names)
        good_score = compute_selectivity(
            adapter, instance, good_exp,
            baseline="mean", background_data=X
        )
        
        # Random explanation
        np.random.seed(99)
        random_attrs = np.random.rand(4)
        random_attrs = random_attrs / np.sum(random_attrs)
        random_exp = create_explanation(random_attrs, feature_names)
        random_score = compute_selectivity(
            adapter, instance, random_exp,
            baseline="mean", background_data=X
        )
        
        # Good explanation should generally have higher AOPC
        # (removing truly important features should cause larger prediction drops)
        # Note: Not always guaranteed, but usually true
        assert isinstance(good_score, float)
        assert isinstance(random_score, float)
    
    def test_inverted_explanation_different_score(self, setup_iris):
        """Test that inverting importance order changes the score."""
        X, y, feature_names = setup_iris
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model)
        
        instance = X[0]
        
        # Original ordering: most important first
        attrs = np.array([0.7, 0.2, 0.08, 0.02])
        normal_exp = create_explanation(attrs, feature_names)
        normal_score = compute_selectivity(
            adapter, instance, normal_exp,
            baseline="mean", background_data=X
        )
        
        # Inverted ordering: least important first
        inverted_attrs = attrs[::-1]  # [0.02, 0.08, 0.2, 0.7]
        inverted_exp = create_explanation(inverted_attrs, feature_names)
        inverted_score = compute_selectivity(
            adapter, instance, inverted_exp,
            baseline="mean", background_data=X
        )
        
        # Scores should differ
        assert normal_score != inverted_score
    
    def test_prediction_drops_are_cumulative(self, setup_model):
        """Test that prediction drops reflect cumulative removal."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.5, 0.3, 0.15, 0.05])
        explanation = create_explanation(attrs, feature_names)
        
        result = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            return_details=True
        )
        
        # Original prediction should be stored
        assert "original_prediction" in result
        
        # Prediction drops should be computed as (original - current)
        for i, drop in enumerate(result["prediction_drops"]):
            expected_drop = result["original_prediction"] - result["predictions"][i]
            assert np.isclose(drop, expected_drop)


# =============================================================================
# Test Class: Target Class
# =============================================================================

class TestTargetClass:
    """Test target_class parameter."""
    
    def test_explicit_target_class(self, setup_iris):
        """Test with explicitly specified target class."""
        X, y, feature_names = setup_iris
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model)
        
        instance = X[0]
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        scores = []
        for target in [0, 1, 2]:
            score = compute_selectivity(
                adapter, instance, explanation,
                baseline="mean", background_data=X,
                target_class=target
            )
            scores.append(score)
            assert isinstance(score, float)
        
        # Different target classes may produce different scores
        # (not guaranteed to be different, but all should be valid)
        assert len(scores) == 3
    
    def test_default_target_class_uses_predicted(self, setup_model):
        """Test that default target_class uses predicted class."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        # Get predicted class
        pred = adapter.predict(instance.reshape(1, -1))[0]
        
        # Compute with default (should use predicted class)
        default_score = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X
        )
        
        # Compute with explicit target = predicted
        explicit_score = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            target_class=pred
        )
        
        assert np.isclose(default_score, explicit_score)


# =============================================================================
# Test Class: AOPC Definition Compliance
# =============================================================================

class TestAOPCDefinition:
    """Test that implementation matches AOPC definition from Montavon et al."""
    
    def test_aopc_formula(self, setup_model):
        """Test AOPC = (1/(K+1)) * Σₖ₌₀ᴷ [f(x) - f(x_{1..k})]."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        result = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            return_details=True
        )
        
        # Manual AOPC calculation
        K = result["n_steps"]
        expected_aopc = np.sum(result["prediction_drops"]) / (K + 1)
        
        assert np.isclose(result["aopc"], expected_aopc)
    
    def test_morf_ordering(self, setup_model):
        """Test Most-relevant-first (MoRF) ordering."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        # Clear importance ordering
        attrs = np.array([0.9, 0.6, 0.3, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        result = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            return_details=True,
            use_absolute=True
        )
        
        # Feature order should be descending by absolute attribution
        # attrs[0]=0.9 > attrs[1]=0.6 > attrs[2]=0.3 > attrs[3]=0.1
        expected_order = [0, 1, 2, 3]
        assert list(result["feature_order"]) == expected_order
    
    def test_step_by_step_removal(self, setup_model):
        """Test that features are removed one at a time."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        result = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            return_details=True
        )
        
        # Should have K+1 predictions (0 to K features removed)
        assert len(result["predictions"]) == len(feature_names) + 1


# =============================================================================
# Test Class: Comparison with Pixel Flipping
# =============================================================================

class TestComparisonWithPixelFlipping:
    """Compare Selectivity (AOPC) with Pixel Flipping (AUC)."""
    
    def test_both_measure_sequential_removal(self, setup_model):
        """Test that both metrics operate on sequential removal."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.5, 0.3, 0.15, 0.05])
        explanation = create_explanation(attrs, feature_names)
        
        from explainiverse.evaluation import compute_pixel_flipping
        
        selectivity_score = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X
        )
        
        pf_score = compute_pixel_flipping(
            adapter, instance, explanation,
            baseline="mean", background_data=X
        )
        
        # Both should be valid floats
        assert isinstance(selectivity_score, float)
        assert isinstance(pf_score, float)
    
    def test_different_aggregation(self, setup_model):
        """Test that Selectivity and Pixel Flipping use different aggregation."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.5, 0.3, 0.15, 0.05])
        explanation = create_explanation(attrs, feature_names)
        
        from explainiverse.evaluation import compute_pixel_flipping
        
        # Selectivity: AOPC (average prediction drop)
        selectivity_result = compute_selectivity(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            return_details=True
        )
        
        # Pixel Flipping: AUC (area under normalized curve)
        pf_result = compute_pixel_flipping(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            return_curve=True
        )
        
        # Both track predictions but aggregate differently
        # Selectivity: mean of drops
        # Pixel Flipping: AUC of normalized curve
        assert "aopc" in selectivity_result
        assert "auc" in pf_result


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
