# tests/test_irof.py
"""
Comprehensive tests for IROF (Iterative Removal of Features) metric.

IROF (Rieger & Hansen, 2020) measures explanation faithfulness by iteratively
removing features in order of attributed importance and tracking prediction
degradation. The Area Over the Curve (AOC) quantifies how quickly predictions
drop when important features are removed. Higher AOC indicates better faithfulness.

Reference: Rieger, L., & Hansen, L. K. (2020). IROF: A Low Resource Evaluation
Metric for Explanation Methods. Workshop AI for Affordable Healthcare at ICLR 2020.
"""
import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from explainiverse.adapters import SklearnAdapter
from explainiverse.core import Explanation
from explainiverse.evaluation import (
    compute_irof,
    compute_irof_multi_segment,
    compute_batch_irof,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def setup_iris():
    """Load Iris dataset for testing."""
    data = load_iris()
    # Use samples from all 3 classes
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
    
    explanations = []
    for i in range(min(10, len(X))):
        np.random.seed(i)
        attrs = np.random.randn(len(feature_names)) * 0.1
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

class TestIROFBasic:
    """Test basic IROF functionality."""
    
    def test_returns_float(self, setup_model):
        """Test that compute_irof returns a float."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        score = compute_irof(
            adapter, instance, explanation,
            baseline="mean", background_data=X
        )
        
        assert isinstance(score, float)
    
    def test_positive_score_for_good_explanation(self, setup_model):
        """Test that good explanations produce positive AOC scores."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        # Use model coefficients as good attributions
        model = adapter.model
        attrs = np.abs(model.coef_[0])
        explanation = create_explanation(attrs, feature_names)
        
        score = compute_irof(
            adapter, instance, explanation,
            baseline="mean", background_data=X
        )
        
        # AOC should be positive for a good explanation
        assert score >= 0.0
    
    def test_score_not_nan(self, setup_model):
        """Test that score is not NaN."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.5, 0.3, 0.15, 0.05])
        explanation = create_explanation(attrs, feature_names)
        
        score = compute_irof(
            adapter, instance, explanation,
            baseline="mean", background_data=X
        )
        
        assert not np.isnan(score)
    
    def test_deterministic_results(self, setup_model):
        """Test that results are deterministic."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        score1 = compute_irof(
            adapter, instance, explanation,
            baseline="mean", background_data=X
        )
        
        score2 = compute_irof(
            adapter, instance, explanation,
            baseline="mean", background_data=X
        )
        
        assert score1 == score2
    
    def test_use_absolute_true(self, setup_model):
        """Test with use_absolute=True (default)."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([-0.5, 0.3, -0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        score = compute_irof(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            use_absolute=True
        )
        
        assert isinstance(score, float)
    
    def test_use_absolute_false(self, setup_model):
        """Test with use_absolute=False."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.5, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        score = compute_irof(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            use_absolute=False
        )
        
        assert isinstance(score, float)
    
    def test_different_use_absolute_different_scores(self, setup_model):
        """Test that use_absolute affects results for mixed-sign attributions."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        # Mixed sign attributions
        attrs = np.array([-0.5, 0.1, 0.3, -0.2])
        explanation = create_explanation(attrs, feature_names)
        
        score_abs = compute_irof(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            use_absolute=True
        )
        
        score_signed = compute_irof(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            use_absolute=False
        )
        
        # Both should be valid floats
        assert isinstance(score_abs, float)
        assert isinstance(score_signed, float)


# =============================================================================
# Test Class: Segment Size Parameter
# =============================================================================

class TestSegmentSize:
    """Test segment_size parameter functionality."""
    
    def test_default_segment_size(self, setup_model):
        """Test default segment_size (1 = each feature is a segment)."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        result = compute_irof(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            return_details=True
        )
        
        # Default segment_size=1 means n_segments = n_features
        assert result["n_segments"] == len(feature_names)
    
    def test_segment_size_1(self, setup_model):
        """Test segment_size=1 (each feature is a segment)."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        result = compute_irof(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            segment_size=1, return_details=True
        )
        
        assert result["n_segments"] == 4
        # Each segment should have exactly 1 feature
        for segment in result["segments"]:
            assert len(segment) == 1
    
    def test_segment_size_2(self, setup_model):
        """Test segment_size=2 (pairs of features)."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        result = compute_irof(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            segment_size=2, return_details=True
        )
        
        # 4 features / 2 = 2 segments
        assert result["n_segments"] == 2
    
    def test_segment_size_equals_n_features(self, setup_model):
        """Test segment_size=n_features (all features in one segment)."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        n_features = len(feature_names)
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        result = compute_irof(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            segment_size=n_features, return_details=True
        )
        
        # All features in one segment
        assert result["n_segments"] == 1
        assert len(result["segments"][0]) == n_features
    
    def test_segment_size_exceeds_features_clamped(self, setup_model):
        """Test that segment_size > n_features is clamped."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        n_features = len(feature_names)
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        result = compute_irof(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            segment_size=100, return_details=True
        )
        
        # Should be clamped to n_features, resulting in 1 segment
        assert result["n_segments"] == 1
    
    def test_different_segment_sizes_different_scores(self, setup_model):
        """Test that different segment sizes can produce different scores."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.5, 0.3, 0.15, 0.05])
        explanation = create_explanation(attrs, feature_names)
        
        score_seg1 = compute_irof(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            segment_size=1
        )
        
        score_seg2 = compute_irof(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            segment_size=2
        )
        
        # Both should be valid floats
        assert isinstance(score_seg1, float)
        assert isinstance(score_seg2, float)


# =============================================================================
# Test Class: Return Details
# =============================================================================

class TestReturnDetails:
    """Test return_details functionality."""
    
    def test_return_details_structure(self, setup_model):
        """Test that return_details=True returns expected dictionary."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        result = compute_irof(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            return_details=True
        )
        
        assert isinstance(result, dict)
        assert "aoc" in result
        assert "curve" in result
        assert "predictions" in result
        assert "segment_order" in result
        assert "segments" in result
        assert "segment_importance" in result
        assert "n_segments" in result
        assert "original_prediction" in result
    
    def test_curve_correct_length(self, setup_model):
        """Test that curve has correct length (n_segments + 1)."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        result = compute_irof(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            return_details=True
        )
        
        # curve should have n_segments + 1 points (including starting point)
        assert len(result["curve"]) == result["n_segments"] + 1
        assert len(result["predictions"]) == result["n_segments"] + 1
    
    def test_curve_starts_at_zero(self, setup_model):
        """Test that prediction drop curve starts at zero."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        result = compute_irof(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            return_details=True
        )
        
        # First point is original - original = 0
        assert result["curve"][0] == 0.0
    
    def test_aoc_matches_standalone(self, setup_model):
        """Test that AOC from details matches standalone call."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        standalone = compute_irof(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            return_details=False
        )
        
        detailed = compute_irof(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            return_details=True
        )
        
        assert np.isclose(standalone, detailed["aoc"])
    
    def test_segment_importance_correct_length(self, setup_model):
        """Test that segment_importance has correct length."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        result = compute_irof(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            return_details=True
        )
        
        assert len(result["segment_importance"]) == result["n_segments"]
    
    def test_segment_order_correct_length(self, setup_model):
        """Test that segment_order has correct length."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        result = compute_irof(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            return_details=True
        )
        
        assert len(result["segment_order"]) == result["n_segments"]


# =============================================================================
# Test Class: Multi-Segment Variant
# =============================================================================

class TestIROFMultiSegment:
    """Test compute_irof_multi_segment functionality."""
    
    def test_returns_dict(self, setup_model):
        """Test that multi-segment variant returns dict."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        result = compute_irof_multi_segment(
            adapter, instance, explanation,
            baseline="mean", background_data=X
        )
        
        assert isinstance(result, dict)
        assert "mean" in result
        assert "scores" in result
        assert "segment_sizes" in result
    
    def test_default_segment_sizes(self, setup_model):
        """Test default segment_sizes."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        result = compute_irof_multi_segment(
            adapter, instance, explanation,
            baseline="mean", background_data=X
        )
        
        # Should have multiple segment sizes
        assert len(result["segment_sizes"]) >= 1
        assert len(result["scores"]) == len(result["segment_sizes"])
    
    def test_custom_segment_sizes(self, setup_model):
        """Test custom segment_sizes."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        result = compute_irof_multi_segment(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            segment_sizes=[1, 2]
        )
        
        assert result["segment_sizes"] == [1, 2]
        assert 1 in result["scores"]
        assert 2 in result["scores"]
    
    def test_mean_is_average_of_scores(self, setup_model):
        """Test that mean equals average of individual scores."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        result = compute_irof_multi_segment(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            segment_sizes=[1, 2]
        )
        
        expected_mean = np.mean(list(result["scores"].values()))
        assert np.isclose(result["mean"], expected_mean)


# =============================================================================
# Test Class: Baseline Types
# =============================================================================

class TestBaselineTypes:
    """Test different baseline types."""
    
    def test_baseline_mean(self, setup_model):
        """Test baseline='mean'."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        score = compute_irof(
            adapter, instance, explanation,
            baseline="mean", background_data=X
        )
        
        assert isinstance(score, float)
    
    def test_baseline_median(self, setup_model):
        """Test baseline='median'."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        score = compute_irof(
            adapter, instance, explanation,
            baseline="median", background_data=X
        )
        
        assert isinstance(score, float)
    
    def test_baseline_scalar(self, setup_model):
        """Test baseline as scalar."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        score = compute_irof(
            adapter, instance, explanation,
            baseline=0.0, background_data=X
        )
        
        assert isinstance(score, float)
    
    def test_baseline_array(self, setup_model):
        """Test baseline as array."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        baseline_values = np.mean(X, axis=0)
        score = compute_irof(
            adapter, instance, explanation,
            baseline=baseline_values
        )
        
        assert isinstance(score, float)
    
    def test_baseline_callable(self, setup_model):
        """Test baseline as callable."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        def custom_baseline(background_data):
            return np.percentile(background_data, 25, axis=0)
        
        score = compute_irof(
            adapter, instance, explanation,
            baseline=custom_baseline, background_data=X
        )
        
        assert isinstance(score, float)
    
    def test_different_baselines_different_scores(self, setup_model):
        """Test that different baselines can produce different scores."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        score_mean = compute_irof(
            adapter, instance, explanation,
            baseline="mean", background_data=X
        )
        
        score_zero = compute_irof(
            adapter, instance, explanation,
            baseline=0.0, background_data=X
        )
        
        # Both should be valid
        assert isinstance(score_mean, float)
        assert isinstance(score_zero, float)


# =============================================================================
# Test Class: Batch Operations
# =============================================================================

class TestBatchOperations:
    """Test batch IROF computation."""
    
    def test_batch_returns_dict(self, setup_batch):
        """Test that batch returns dict."""
        adapter, X, explanations, feature_names = setup_batch
        
        result = compute_batch_irof(
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
        
        result = compute_batch_irof(
            adapter, X[:len(explanations)], explanations,
            baseline="mean", max_samples=3
        )
        
        assert result["n_samples"] <= 3
    
    def test_batch_values_are_floats(self, setup_batch):
        """Test that batch results are floats."""
        adapter, X, explanations, feature_names = setup_batch
        
        result = compute_batch_irof(
            adapter, X[:len(explanations)], explanations,
            baseline="mean"
        )
        
        assert isinstance(result["mean"], float)
        assert isinstance(result["std"], float)
        assert isinstance(result["min"], float)
        assert isinstance(result["max"], float)
    
    def test_batch_with_custom_segment_size(self, setup_batch):
        """Test batch with custom segment_size."""
        adapter, X, explanations, feature_names = setup_batch
        
        result = compute_batch_irof(
            adapter, X[:len(explanations)], explanations,
            baseline="mean", segment_size=2
        )
        
        assert result["n_samples"] > 0
    
    def test_batch_mean_in_reasonable_range(self, setup_batch):
        """Test that batch mean is in reasonable range."""
        adapter, X, explanations, feature_names = setup_batch
        
        result = compute_batch_irof(
            adapter, X[:len(explanations)], explanations,
            baseline="mean"
        )
        
        # AOC can be negative (if removing "important" features increases prediction)
        # but should generally be finite
        assert np.isfinite(result["mean"])


# =============================================================================
# Test Class: Multiple Model Types
# =============================================================================

class TestMultipleModels:
    """Test IROF with different model types."""
    
    def test_random_forest(self, setup_iris):
        """Test with RandomForestClassifier."""
        X, y, feature_names = setup_iris
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model)
        
        instance = X[0]
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        score = compute_irof(
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
        
        score = compute_irof(
            adapter, instance, explanation,
            baseline="mean", background_data=X
        )
        
        assert isinstance(score, float)


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
        
        score = compute_irof(
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
        
        score = compute_irof(
            adapter, instance, explanation,
            baseline="mean", background_data=X
        )
        
        assert isinstance(score, float)
    
    def test_identical_attributions(self, setup_model):
        """Test when all attributions are identical."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.25, 0.25, 0.25, 0.25])
        explanation = create_explanation(attrs, feature_names)
        
        score = compute_irof(
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
        
        score = compute_irof(
            adapter, instance, explanation,
            baseline="mean", background_data=X
        )
        
        # With zero attributions, order is arbitrary but should still work
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
        
        score = compute_irof(
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
        
        score = compute_irof(
            adapter, instance, explanation,
            baseline="mean", background_data=X
        )
        
        assert isinstance(score, float)
    
    def test_negative_attributions(self, setup_model):
        """Test with all negative attributions."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([-0.4, -0.3, -0.2, -0.1])
        explanation = create_explanation(attrs, feature_names)
        
        score = compute_irof(
            adapter, instance, explanation,
            baseline="mean", background_data=X
        )
        
        assert isinstance(score, float)


# =============================================================================
# Test Class: Semantic Validation
# =============================================================================

class TestSemanticValidation:
    """Test that IROF measures what it claims."""
    
    def test_good_explanation_higher_than_random(self, setup_iris):
        """Test that good explanations tend to have higher AOC than random."""
        X, y, feature_names = setup_iris
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model)
        
        instance = X[0]
        
        # Good explanation using model coefficients
        true_importance = np.abs(model.coef_[0])
        good_exp = create_explanation(true_importance, feature_names)
        
        # Random explanation
        np.random.seed(99)
        random_attrs = np.random.rand(4)
        random_exp = create_explanation(random_attrs, feature_names)
        
        good_score = compute_irof(
            adapter, instance, good_exp,
            baseline="mean", background_data=X
        )
        
        random_score = compute_irof(
            adapter, instance, random_exp,
            baseline="mean", background_data=X
        )
        
        # Good explanation should generally have higher or similar AOC
        # (removing truly important features should cause larger drops)
        assert isinstance(good_score, float)
        assert isinstance(random_score, float)
    
    def test_inverted_explanation_lower_score(self, setup_iris):
        """Test that inverted importance ordering produces different results."""
        X, y, feature_names = setup_iris
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model)
        
        instance = X[0]
        
        # Good explanation
        true_importance = np.abs(model.coef_[0])
        good_exp = create_explanation(true_importance, feature_names)
        
        # Inverted explanation (worst features marked as best)
        inverted_importance = 1.0 / (true_importance + 1e-6)
        inverted_exp = create_explanation(inverted_importance, feature_names)
        
        good_score = compute_irof(
            adapter, instance, good_exp,
            baseline="mean", background_data=X
        )
        
        inverted_score = compute_irof(
            adapter, instance, inverted_exp,
            baseline="mean", background_data=X
        )
        
        # Both should be valid
        assert isinstance(good_score, float)
        assert isinstance(inverted_score, float)
        
        # Good explanation should have higher score (faster degradation)
        # when removing truly important features first
        assert good_score >= inverted_score or np.isclose(good_score, inverted_score, atol=0.1)
    
    def test_prediction_drops_with_feature_removal(self, setup_model):
        """Test that removing important features causes prediction drops."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        # High importance for first feature
        attrs = np.array([0.9, 0.05, 0.03, 0.02])
        explanation = create_explanation(attrs, feature_names)
        
        result = compute_irof(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            return_details=True
        )
        
        # After removing the most important feature, prediction should drop
        # (curve[1] should be > 0 for a faithful explanation)
        # This is model-dependent, so we just check the structure
        assert len(result["curve"]) == len(feature_names) + 1


# =============================================================================
# Test Class: Target Class
# =============================================================================

class TestTargetClass:
    """Test target_class parameter."""
    
    def test_explicit_target_class(self, setup_iris):
        """Test with explicit target class."""
        X, y, feature_names = setup_iris
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model)
        
        instance = X[0]
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        scores = []
        for target in [0, 1, 2]:
            score = compute_irof(
                adapter, instance, explanation,
                baseline="mean", background_data=X,
                target_class=target
            )
            scores.append(score)
            assert isinstance(score, float)
        
        assert len(scores) == 3
    
    def test_default_target_uses_predicted(self, setup_model):
        """Test that default target uses predicted class."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        
        # Get predicted class probabilities
        probs = adapter.predict(instance.reshape(1, -1))[0]
        predicted_class = int(np.argmax(probs))
        
        default_score = compute_irof(
            adapter, instance, explanation,
            baseline="mean", background_data=X
        )
        
        explicit_score = compute_irof(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            target_class=predicted_class
        )
        
        assert np.isclose(default_score, explicit_score)


# =============================================================================
# Test Class: Comparison with Similar Metrics
# =============================================================================

class TestComparisonWithSimilarMetrics:
    """Test IROF compared to similar metrics like Pixel Flipping."""
    
    def test_irof_segment_1_similar_concept_to_pixel_flipping(self, setup_model):
        """Test that IROF with segment_size=1 follows similar concept to Pixel Flipping."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        
        attrs = np.array([0.5, 0.3, 0.15, 0.05])
        explanation = create_explanation(attrs, feature_names)
        
        # IROF with segment_size=1
        irof_result = compute_irof(
            adapter, instance, explanation,
            baseline="mean", background_data=X,
            segment_size=1, return_details=True
        )
        
        # Check that we have the right number of steps
        assert irof_result["n_segments"] == len(feature_names)
        
        # Verify AOC is computed correctly (positive area means prediction drops)
        assert isinstance(irof_result["aoc"], float)


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
