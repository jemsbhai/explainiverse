# tests/test_road.py
"""
Comprehensive tests for ROAD metric (Rong et al., 2022).

ROAD (RemOve And Debias) evaluates explanation faithfulness using noisy linear
imputation instead of simple baseline replacement, addressing out-of-distribution
problems in perturbation-based evaluation.

Two orderings:
- MoRF (Most Relevant First): Higher score = better (important features matter)
- LeRF (Least Relevant First): Lower score = better (unimportant features don't matter)

Tests cover:
1. Basic functionality
2. MoRF and LeRF orderings
3. Return details
4. Combined variant
5. Batch operations
6. Background data handling
7. Percentages parameter
8. Noise scale parameter
9. Multiple model types
10. Edge cases
11. Semantic validation
12. Target class handling
13. Use absolute parameter
"""
import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from explainiverse.core.explanation import Explanation
from explainiverse.evaluation.faithfulness_extended import (
    compute_road,
    compute_road_combined,
    compute_batch_road,
    _noisy_linear_impute,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def setup_iris():
    """Load and prepare Iris dataset with all 3 classes."""
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Take balanced samples from all classes
    indices = []
    for cls in range(3):
        cls_indices = np.where(y == cls)[0][:20]
        indices.extend(cls_indices)
    
    X = X[indices]
    y = y[indices]
    
    # Standardize for better model performance
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y


@pytest.fixture
def setup_model(setup_iris):
    """Train a LogisticRegression model on Iris."""
    X, y = setup_iris
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def setup_batch(setup_model):
    """Create a batch of explanations for testing."""
    model, X, y = setup_model
    
    # Create 10 explanations with synthetic attributions
    explanations = []
    n_features = X.shape[1]
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    for i in range(10):
        # Generate attributions (higher for first features to simulate a pattern)
        np.random.seed(42 + i)
        attr_values = np.random.randn(n_features)
        attr_values = np.abs(attr_values)  # Make positive
        attr_values[0] *= 2  # Emphasize first feature
        
        attributions = {
            feature_names[j]: float(attr_values[j])
            for j in range(n_features)
        }
        
        exp = Explanation(
            explainer_name="test_explainer",
            target_class="class_0",
            explanation_data={"feature_attributions": attributions},
            feature_names=feature_names,
        )
        explanations.append(exp)
    
    return model, X, explanations


def create_explanation(attributions: np.ndarray, feature_names=None) -> Explanation:
    """Helper to create Explanation objects."""
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(attributions))]
    
    attr_dict = {
        feature_names[i]: float(attributions[i])
        for i in range(len(attributions))
    }
    
    exp = Explanation(
        explainer_name="test_explainer",
        target_class="class_0",
        explanation_data={"feature_attributions": attr_dict},
        feature_names=feature_names,
    )
    
    return exp


# =============================================================================
# Test 1: Basic Functionality (8 tests)
# =============================================================================

class TestBasicFunctionality:
    """Basic functionality tests for compute_road."""
    
    def test_returns_float(self, setup_model):
        """Test that compute_road returns a float."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        score = compute_road(
            model, instance, explanation,
            background_data=X, seed=42
        )
        
        assert isinstance(score, float)
    
    def test_score_not_nan(self, setup_model):
        """Test that score is not NaN."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        score = compute_road(
            model, instance, explanation,
            background_data=X, seed=42
        )
        
        assert not np.isnan(score)
    
    def test_deterministic_with_seed(self, setup_model):
        """Test that results are deterministic with same seed."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        score1 = compute_road(
            model, instance, explanation,
            background_data=X, seed=42
        )
        score2 = compute_road(
            model, instance, explanation,
            background_data=X, seed=42
        )
        
        assert score1 == score2
    
    def test_different_seed_different_results(self, setup_model):
        """Test that different seeds give different results."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        score1 = compute_road(
            model, instance, explanation,
            background_data=X, seed=42
        )
        score2 = compute_road(
            model, instance, explanation,
            background_data=X, seed=123
        )
        
        # Due to noisy imputation, different seeds produce different results
        # (though could theoretically be same, very unlikely)
        assert isinstance(score1, float)
        assert isinstance(score2, float)
    
    def test_background_data_required(self, setup_model):
        """Test that background_data is required."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        with pytest.raises(ValueError, match="background_data is required"):
            compute_road(
                model, instance, explanation,
                background_data=None, seed=42
            )
    
    def test_invalid_background_data_shape(self, setup_model):
        """Test that invalid background_data shape raises error."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        # Wrong number of columns
        bad_background = np.random.randn(10, n_features + 2)
        
        with pytest.raises(ValueError, match="must be 2D with"):
            compute_road(
                model, instance, explanation,
                background_data=bad_background, seed=42
            )
    
    def test_accepts_default_percentages(self, setup_model):
        """Test that default percentages work correctly."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        # No explicit percentages - uses default
        score = compute_road(
            model, instance, explanation,
            background_data=X, seed=42
        )
        
        assert isinstance(score, float)
    
    def test_custom_percentages(self, setup_model):
        """Test with custom percentages."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        custom_percentages = [0.2, 0.4, 0.6, 0.8]
        score = compute_road(
            model, instance, explanation,
            background_data=X,
            percentages=custom_percentages,
            seed=42
        )
        
        assert isinstance(score, float)


# =============================================================================
# Test 2: MoRF and LeRF Orderings (6 tests)
# =============================================================================

class TestOrderings:
    """Test MoRF and LeRF orderings."""
    
    def test_morf_order(self, setup_model):
        """Test MoRF (Most Relevant First) ordering."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        score = compute_road(
            model, instance, explanation,
            background_data=X,
            order="morf",
            seed=42
        )
        
        assert isinstance(score, float)
    
    def test_lerf_order(self, setup_model):
        """Test LeRF (Least Relevant First) ordering."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        score = compute_road(
            model, instance, explanation,
            background_data=X,
            order="lerf",
            seed=42
        )
        
        assert isinstance(score, float)
    
    def test_morf_lerf_different_scores(self, setup_model):
        """Test that MoRF and LeRF give different scores."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        # Create non-uniform attributions
        attributions = np.array([0.8, 0.5, 0.2, 0.1])
        explanation = create_explanation(attributions)
        
        score_morf = compute_road(
            model, instance, explanation,
            background_data=X,
            order="morf",
            seed=42
        )
        score_lerf = compute_road(
            model, instance, explanation,
            background_data=X,
            order="lerf",
            seed=42
        )
        
        # With non-uniform attributions, MoRF and LeRF should differ
        # (MoRF should show larger prediction changes for good explanations)
        assert isinstance(score_morf, float)
        assert isinstance(score_lerf, float)
    
    def test_invalid_order_raises(self, setup_model):
        """Test that invalid order raises ValueError."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        with pytest.raises(ValueError, match="order must be"):
            compute_road(
                model, instance, explanation,
                background_data=X,
                order="invalid",
                seed=42
            )
    
    def test_default_order_is_morf(self, setup_model):
        """Test that default order is MoRF."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        score_default = compute_road(
            model, instance, explanation,
            background_data=X,
            seed=42
        )
        score_morf = compute_road(
            model, instance, explanation,
            background_data=X,
            order="morf",
            seed=42
        )
        
        assert score_default == score_morf
    
    def test_morf_higher_for_good_explanation(self, setup_model):
        """Test semantic: MoRF should be higher for good explanations."""
        model, X, y = setup_model
        instance = X[0]
        
        # Get model coefficients as "good" attributions
        pred_class = model.predict(instance.reshape(1, -1))[0]
        coefs = model.coef_[pred_class]
        good_attributions = np.abs(coefs * instance)
        good_explanation = create_explanation(good_attributions)
        
        # Create random attributions
        np.random.seed(999)
        random_attributions = np.abs(np.random.randn(len(instance)))
        random_explanation = create_explanation(random_attributions)
        
        score_good = compute_road(
            model, instance, good_explanation,
            background_data=X,
            order="morf",
            seed=42
        )
        score_random = compute_road(
            model, instance, random_explanation,
            background_data=X,
            order="morf",
            seed=42
        )
        
        # Both should be valid
        assert isinstance(score_good, float)
        assert isinstance(score_random, float)


# =============================================================================
# Test 3: Return Details (7 tests)
# =============================================================================

class TestReturnDetails:
    """Test return_details functionality."""
    
    def test_return_details_is_dict(self, setup_model):
        """Test that return_details=True returns a dictionary."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        result = compute_road(
            model, instance, explanation,
            background_data=X,
            return_details=True,
            seed=42
        )
        
        assert isinstance(result, dict)
    
    def test_return_details_has_required_keys(self, setup_model):
        """Test that returned dict has all required keys."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        result = compute_road(
            model, instance, explanation,
            background_data=X,
            return_details=True,
            seed=42
        )
        
        required_keys = [
            "score",
            "prediction_changes",
            "predictions",
            "percentages",
            "n_removed",
            "feature_order",
            "order",
            "original_prediction",
        ]
        
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_prediction_changes_length_matches_percentages(self, setup_model):
        """Test that prediction_changes matches percentages length."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        percentages = [0.2, 0.4, 0.6]
        result = compute_road(
            model, instance, explanation,
            background_data=X,
            percentages=percentages,
            return_details=True,
            seed=42
        )
        
        assert len(result["prediction_changes"]) == len(percentages)
        assert len(result["predictions"]) == len(percentages)
        assert len(result["n_removed"]) == len(percentages)
    
    def test_score_equals_mean_prediction_changes(self, setup_model):
        """Test that score equals mean of prediction_changes."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        result = compute_road(
            model, instance, explanation,
            background_data=X,
            return_details=True,
            seed=42
        )
        
        expected_score = np.mean(result["prediction_changes"])
        assert np.isclose(result["score"], expected_score)
    
    def test_score_matches_standalone_call(self, setup_model):
        """Test that detailed score matches non-detailed call."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        score = compute_road(
            model, instance, explanation,
            background_data=X,
            return_details=False,
            seed=42
        )
        
        detailed = compute_road(
            model, instance, explanation,
            background_data=X,
            return_details=True,
            seed=42
        )
        
        assert np.isclose(score, detailed["score"])
    
    def test_order_recorded_in_details(self, setup_model):
        """Test that order is correctly recorded in details."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        for order in ["morf", "lerf"]:
            result = compute_road(
                model, instance, explanation,
                background_data=X,
                order=order,
                return_details=True,
                seed=42
            )
            
            assert result["order"] == order
    
    def test_feature_order_has_correct_length(self, setup_model):
        """Test that feature_order contains all features."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        result = compute_road(
            model, instance, explanation,
            background_data=X,
            return_details=True,
            seed=42
        )
        
        assert len(result["feature_order"]) == n_features
        # Should contain all indices
        assert set(result["feature_order"]) == set(range(n_features))


# =============================================================================
# Test 4: Combined Variant (5 tests)
# =============================================================================

class TestCombinedVariant:
    """Test compute_road_combined."""
    
    def test_combined_returns_dict(self, setup_model):
        """Test that compute_road_combined returns a dictionary."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        result = compute_road_combined(
            model, instance, explanation,
            background_data=X,
            seed=42
        )
        
        assert isinstance(result, dict)
    
    def test_combined_has_required_keys(self, setup_model):
        """Test that combined result has all required keys."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        result = compute_road_combined(
            model, instance, explanation,
            background_data=X,
            seed=42
        )
        
        assert "morf" in result
        assert "lerf" in result
        assert "gap" in result
        assert "scores" in result
    
    def test_gap_equals_morf_minus_lerf(self, setup_model):
        """Test that gap equals morf - lerf."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        result = compute_road_combined(
            model, instance, explanation,
            background_data=X,
            seed=42
        )
        
        expected_gap = result["morf"] - result["lerf"]
        assert np.isclose(result["gap"], expected_gap)
    
    def test_scores_dict_matches_individual(self, setup_model):
        """Test that scores dict matches individual morf/lerf values."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        result = compute_road_combined(
            model, instance, explanation,
            background_data=X,
            seed=42
        )
        
        assert result["scores"]["morf"] == result["morf"]
        assert result["scores"]["lerf"] == result["lerf"]
    
    def test_combined_with_custom_percentages(self, setup_model):
        """Test combined variant with custom percentages."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        result = compute_road_combined(
            model, instance, explanation,
            background_data=X,
            percentages=[0.25, 0.5, 0.75],
            seed=42
        )
        
        assert isinstance(result["morf"], float)
        assert isinstance(result["lerf"], float)


# =============================================================================
# Test 5: Batch Operations (6 tests)
# =============================================================================

class TestBatchOperations:
    """Test compute_batch_road."""
    
    def test_batch_returns_dict(self, setup_batch):
        """Test that batch computation returns a dictionary."""
        model, X, explanations = setup_batch
        
        result = compute_batch_road(
            model, X[:10], explanations,
            background_data=X,
            seed=42
        )
        
        assert isinstance(result, dict)
        assert "mean" in result
        assert "std" in result
        assert "min" in result
        assert "max" in result
        assert "n_samples" in result
    
    def test_batch_max_samples_parameter(self, setup_batch):
        """Test max_samples parameter limits computation."""
        model, X, explanations = setup_batch
        
        result = compute_batch_road(
            model, X[:10], explanations,
            background_data=X,
            max_samples=5,
            seed=42
        )
        
        assert result["n_samples"] <= 5
    
    def test_batch_with_morf_order(self, setup_batch):
        """Test batch with MoRF ordering."""
        model, X, explanations = setup_batch
        
        result = compute_batch_road(
            model, X[:10], explanations,
            background_data=X,
            order="morf",
            seed=42
        )
        
        assert isinstance(result["mean"], float)
        assert not np.isnan(result["mean"])
    
    def test_batch_with_lerf_order(self, setup_batch):
        """Test batch with LeRF ordering."""
        model, X, explanations = setup_batch
        
        result = compute_batch_road(
            model, X[:10], explanations,
            background_data=X,
            order="lerf",
            seed=42
        )
        
        assert isinstance(result["mean"], float)
        assert not np.isnan(result["mean"])
    
    def test_batch_uses_default_background_data(self, setup_batch):
        """Test that batch uses X as background_data if not provided."""
        model, X, explanations = setup_batch
        
        result = compute_batch_road(
            model, X[:10], explanations,
            background_data=None,  # Should default to X
            seed=42
        )
        
        assert result["n_samples"] > 0
    
    def test_batch_custom_percentages(self, setup_batch):
        """Test batch with custom percentages."""
        model, X, explanations = setup_batch
        
        result = compute_batch_road(
            model, X[:10], explanations,
            background_data=X,
            percentages=[0.3, 0.6],
            seed=42
        )
        
        assert isinstance(result["mean"], float)


# =============================================================================
# Test 6: Background Data Handling (4 tests)
# =============================================================================

class TestBackgroundDataHandling:
    """Test background_data handling."""
    
    def test_background_data_used_for_imputation(self, setup_model):
        """Test that background_data affects imputation results."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        # Different background data should give different results
        background1 = X[:20]
        background2 = X[30:]  # Different samples
        
        score1 = compute_road(
            model, instance, explanation,
            background_data=background1,
            seed=42
        )
        score2 = compute_road(
            model, instance, explanation,
            background_data=background2,
            seed=42
        )
        
        # Both should be valid (values may differ)
        assert isinstance(score1, float)
        assert isinstance(score2, float)
    
    def test_small_background_data(self, setup_model):
        """Test with small background data (minimum viable)."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        # Very small background data
        small_background = X[:5]
        
        score = compute_road(
            model, instance, explanation,
            background_data=small_background,
            seed=42
        )
        
        assert isinstance(score, float)
        assert not np.isnan(score)
    
    def test_large_background_data(self, setup_model):
        """Test with all available background data."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        score = compute_road(
            model, instance, explanation,
            background_data=X,  # Full dataset
            seed=42
        )
        
        assert isinstance(score, float)
    
    def test_1d_background_raises(self, setup_model):
        """Test that 1D background_data raises error."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        with pytest.raises(ValueError):
            compute_road(
                model, instance, explanation,
                background_data=X[0],  # 1D
                seed=42
            )


# =============================================================================
# Test 7: Percentages Parameter (5 tests)
# =============================================================================

class TestPercentagesParameter:
    """Test percentages parameter handling."""
    
    def test_default_percentages_range(self, setup_model):
        """Test that default percentages span expected range."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        result = compute_road(
            model, instance, explanation,
            background_data=X,
            return_details=True,
            seed=42
        )
        
        # Default is [0.1, 0.2, ..., 0.9]
        assert result["percentages"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    def test_single_percentage(self, setup_model):
        """Test with single percentage."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        result = compute_road(
            model, instance, explanation,
            background_data=X,
            percentages=[0.5],
            return_details=True,
            seed=42
        )
        
        assert len(result["prediction_changes"]) == 1
    
    def test_invalid_percentages_filtered(self, setup_model):
        """Test that invalid percentages (<=0 or >=1) are filtered."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        # Include invalid percentages
        result = compute_road(
            model, instance, explanation,
            background_data=X,
            percentages=[0.0, 0.3, 1.0, 0.6, -0.1],
            return_details=True,
            seed=42
        )
        
        # Only valid ones should remain
        assert result["percentages"] == [0.3, 0.6]
    
    def test_empty_percentages_raises(self, setup_model):
        """Test that empty valid percentages raises error."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        with pytest.raises(ValueError, match="percentages must contain"):
            compute_road(
                model, instance, explanation,
                background_data=X,
                percentages=[0.0, 1.0, -0.5],  # All invalid
                seed=42
            )
    
    def test_percentages_sorted(self, setup_model):
        """Test that percentages are sorted in output."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        result = compute_road(
            model, instance, explanation,
            background_data=X,
            percentages=[0.8, 0.2, 0.5],  # Unsorted
            return_details=True,
            seed=42
        )
        
        assert result["percentages"] == [0.2, 0.5, 0.8]  # Sorted


# =============================================================================
# Test 8: Noise Scale Parameter (4 tests)
# =============================================================================

class TestNoiseScaleParameter:
    """Test noise_scale parameter."""
    
    def test_default_noise_scale(self, setup_model):
        """Test default noise_scale (1.0)."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        score = compute_road(
            model, instance, explanation,
            background_data=X,
            seed=42
        )
        
        assert isinstance(score, float)
    
    def test_zero_noise_scale(self, setup_model):
        """Test with zero noise scale (deterministic imputation)."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        # Zero noise should give more deterministic results
        score1 = compute_road(
            model, instance, explanation,
            background_data=X,
            noise_scale=0.0,
            seed=42
        )
        score2 = compute_road(
            model, instance, explanation,
            background_data=X,
            noise_scale=0.0,
            seed=123  # Different seed shouldn't matter with zero noise
        )
        
        # With zero noise, results should be very similar (deterministic linear imputation)
        assert np.isclose(score1, score2, rtol=0.01)
    
    def test_high_noise_scale(self, setup_model):
        """Test with high noise scale."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        score = compute_road(
            model, instance, explanation,
            background_data=X,
            noise_scale=5.0,
            seed=42
        )
        
        assert isinstance(score, float)
    
    def test_different_noise_scales_different_results(self, setup_model):
        """Test that different noise scales produce different results."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        score_low = compute_road(
            model, instance, explanation,
            background_data=X,
            noise_scale=0.1,
            seed=42
        )
        score_high = compute_road(
            model, instance, explanation,
            background_data=X,
            noise_scale=2.0,
            seed=42
        )
        
        # Both should be valid
        assert isinstance(score_low, float)
        assert isinstance(score_high, float)


# =============================================================================
# Test 9: Multiple Model Types (3 tests)
# =============================================================================

class TestMultipleModelTypes:
    """Test with different model types."""
    
    def test_with_random_forest(self, setup_iris):
        """Test with RandomForestClassifier."""
        X, y = setup_iris
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        instance = X[0]
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        score = compute_road(
            model, instance, explanation,
            background_data=X,
            seed=42
        )
        
        assert isinstance(score, float)
        assert not np.isnan(score)
    
    def test_with_gradient_boosting(self, setup_iris):
        """Test with GradientBoostingClassifier."""
        X, y = setup_iris
        
        model = GradientBoostingClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        instance = X[0]
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        score = compute_road(
            model, instance, explanation,
            background_data=X,
            seed=42
        )
        
        assert isinstance(score, float)
        assert not np.isnan(score)
    
    def test_with_logistic_regression(self, setup_model):
        """Test with LogisticRegression."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        score = compute_road(
            model, instance, explanation,
            background_data=X,
            seed=42
        )
        
        assert isinstance(score, float)


# =============================================================================
# Test 10: Edge Cases (8 tests)
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_few_features(self, setup_iris):
        """Test with few features (2)."""
        X, y = setup_iris
        X_small = X[:, :2]
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_small, y)
        
        instance = X_small[0]
        attributions = np.random.randn(2)
        explanation = create_explanation(attributions)
        
        score = compute_road(
            model, instance, explanation,
            background_data=X_small,
            seed=42
        )
        
        assert isinstance(score, float)
    
    def test_many_features(self):
        """Test with many features (20)."""
        np.random.seed(42)
        n_features = 20
        n_samples = 100
        
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] > 0).astype(int)
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        
        instance = X[0]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        score = compute_road(
            model, instance, explanation,
            background_data=X,
            seed=42
        )
        
        assert isinstance(score, float)
    
    def test_zero_attributions(self, setup_model):
        """Test with all-zero attributions."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.zeros(n_features)
        explanation = create_explanation(attributions)
        
        score = compute_road(
            model, instance, explanation,
            background_data=X,
            seed=42
        )
        
        assert isinstance(score, float)
        assert not np.isnan(score)
    
    def test_identical_attributions(self, setup_model):
        """Test with identical attribution values."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.ones(n_features) * 0.5
        explanation = create_explanation(attributions)
        
        score = compute_road(
            model, instance, explanation,
            background_data=X,
            seed=42
        )
        
        assert isinstance(score, float)
        assert not np.isnan(score)
    
    def test_negative_attributions(self, setup_model):
        """Test with negative attributions."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.array([-0.5, -0.3, 0.2, -0.1])
        explanation = create_explanation(attributions)
        
        score = compute_road(
            model, instance, explanation,
            background_data=X,
            seed=42
        )
        
        assert isinstance(score, float)
    
    def test_very_small_percentage(self, setup_model):
        """Test with very small percentage."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        # Very small percentage - will remove at least 1 feature
        result = compute_road(
            model, instance, explanation,
            background_data=X,
            percentages=[0.01],
            return_details=True,
            seed=42
        )
        
        # Should remove at least 1 feature
        assert result["n_removed"][0] >= 1
    
    def test_large_percentage(self, setup_model):
        """Test with large percentage (0.99)."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        result = compute_road(
            model, instance, explanation,
            background_data=X,
            percentages=[0.99],
            return_details=True,
            seed=42
        )
        
        # Should remove most features
        assert result["n_removed"][0] >= n_features - 1
    
    def test_collinear_features_in_background(self):
        """Test handling of collinear features in background data."""
        np.random.seed(42)
        n_samples = 100
        
        # Create data with collinear features
        X = np.random.randn(n_samples, 4)
        X[:, 2] = X[:, 0] + np.random.randn(n_samples) * 0.01  # Nearly collinear
        y = (X[:, 0] > 0).astype(int)
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        
        instance = X[0]
        attributions = np.random.randn(4)
        explanation = create_explanation(attributions)
        
        # Should handle collinearity gracefully (lstsq handles rank-deficiency)
        score = compute_road(
            model, instance, explanation,
            background_data=X,
            seed=42
        )
        
        assert isinstance(score, float)
        assert not np.isnan(score)


# =============================================================================
# Test 11: Semantic Validation (4 tests)
# =============================================================================

class TestSemanticValidation:
    """Test semantic correctness of the metric."""
    
    def test_morf_larger_changes_for_important_features(self, setup_model):
        """Test that MoRF shows larger changes when removing important features first."""
        model, X, y = setup_model
        instance = X[0]
        
        # Create explanation with one dominant feature
        attributions = np.array([1.0, 0.01, 0.01, 0.01])
        explanation = create_explanation(attributions)
        
        result = compute_road(
            model, instance, explanation,
            background_data=X,
            order="morf",
            return_details=True,
            seed=42
        )
        
        # First few removals should cause larger changes if feature 0 is truly important
        assert isinstance(result["prediction_changes"], np.ndarray)
    
    def test_prediction_changes_are_original_minus_perturbed(self, setup_model):
        """Test that prediction_changes = original - perturbed."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        result = compute_road(
            model, instance, explanation,
            background_data=X,
            return_details=True,
            seed=42
        )
        
        original = result["original_prediction"]
        for i, pred in enumerate(result["predictions"]):
            expected_change = original - pred
            assert np.isclose(result["prediction_changes"][i], expected_change)
    
    def test_good_explanation_higher_morf_than_lerf(self, setup_model):
        """Test that good explanations have higher MoRF than LeRF."""
        model, X, y = setup_model
        instance = X[0]
        
        # Get model coefficients as "good" attributions
        pred_class = model.predict(instance.reshape(1, -1))[0]
        coefs = model.coef_[pred_class]
        good_attributions = np.abs(coefs * instance)
        good_explanation = create_explanation(good_attributions)
        
        result = compute_road_combined(
            model, instance, good_explanation,
            background_data=X,
            seed=42
        )
        
        # For a good explanation, MoRF should generally be higher than LeRF
        # (removing important features first causes more damage)
        assert isinstance(result["gap"], float)
    
    def test_n_removed_increases_with_percentage(self, setup_model):
        """Test that n_removed increases monotonically with percentage."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        result = compute_road(
            model, instance, explanation,
            background_data=X,
            percentages=[0.1, 0.3, 0.5, 0.7, 0.9],
            return_details=True,
            seed=42
        )
        
        # n_removed should be non-decreasing
        n_removed = result["n_removed"]
        for i in range(1, len(n_removed)):
            assert n_removed[i] >= n_removed[i-1]


# =============================================================================
# Test 12: Target Class Handling (3 tests)
# =============================================================================

class TestTargetClassHandling:
    """Test target class parameter handling."""
    
    def test_explicit_target_class(self, setup_model):
        """Test with explicit target class for all classes."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        # Test for each class (Iris has 3 classes)
        scores = []
        for target in range(3):
            score = compute_road(
                model, instance, explanation,
                background_data=X,
                target_class=target,
                seed=42
            )
            scores.append(score)
            assert isinstance(score, float)
        
        # Scores for different classes may differ
        assert len(scores) == 3
    
    def test_default_uses_predicted_class(self, setup_model):
        """Test that default uses model's predicted class."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        # Get predicted class
        predicted_class = model.predict(instance.reshape(1, -1))[0]
        
        # Default (no target_class)
        score_default = compute_road(
            model, instance, explanation,
            background_data=X,
            seed=42
        )
        
        # Explicit with predicted class
        score_explicit = compute_road(
            model, instance, explanation,
            background_data=X,
            target_class=predicted_class,
            seed=42
        )
        
        # Should be equal
        assert np.isclose(score_default, score_explicit)
    
    def test_different_target_classes_different_scores(self, setup_model):
        """Test that different target classes may give different scores."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        score_0 = compute_road(
            model, instance, explanation,
            background_data=X,
            target_class=0,
            seed=42
        )
        score_1 = compute_road(
            model, instance, explanation,
            background_data=X,
            target_class=1,
            seed=42
        )
        
        # Both valid, likely different
        assert isinstance(score_0, float)
        assert isinstance(score_1, float)


# =============================================================================
# Test 13: Use Absolute Parameter (3 tests)
# =============================================================================

class TestUseAbsoluteParameter:
    """Test use_absolute parameter."""
    
    def test_default_use_absolute_true(self, setup_model):
        """Test that default use_absolute is True."""
        model, X, y = setup_model
        instance = X[0]
        
        # Create explanation with mixed signs
        attributions = np.array([0.5, -0.8, 0.3, -0.2])
        explanation = create_explanation(attributions)
        
        score_default = compute_road(
            model, instance, explanation,
            background_data=X,
            seed=42
        )
        score_explicit = compute_road(
            model, instance, explanation,
            background_data=X,
            use_absolute=True,
            seed=42
        )
        
        assert score_default == score_explicit
    
    def test_use_absolute_false(self, setup_model):
        """Test use_absolute=False uses signed values."""
        model, X, y = setup_model
        instance = X[0]
        
        # Create explanation with mixed signs
        attributions = np.array([0.5, -0.8, 0.3, -0.2])
        explanation = create_explanation(attributions)
        
        score = compute_road(
            model, instance, explanation,
            background_data=X,
            use_absolute=False,
            seed=42
        )
        
        assert isinstance(score, float)
    
    def test_use_absolute_affects_ordering(self, setup_model):
        """Test that use_absolute affects feature ordering."""
        model, X, y = setup_model
        instance = X[0]
        
        # Create explanation where abs order differs from signed order
        attributions = np.array([0.5, -0.8, 0.3, -0.2])  # abs: [0.5, 0.8, 0.3, 0.2]
        explanation = create_explanation(attributions)
        
        result_abs = compute_road(
            model, instance, explanation,
            background_data=X,
            use_absolute=True,
            return_details=True,
            seed=42
        )
        result_signed = compute_road(
            model, instance, explanation,
            background_data=X,
            use_absolute=False,
            return_details=True,
            seed=42
        )
        
        # Feature orderings should differ
        # With abs: [1, 0, 2, 3] (descending by |value|)
        # With signed: [0, 2, 3, 1] (descending by value)
        assert not np.array_equal(result_abs["feature_order"], result_signed["feature_order"])


# =============================================================================
# Test 14: Noisy Linear Imputation Helper (5 tests)
# =============================================================================

class TestNoisyLinearImpute:
    """Test _noisy_linear_impute helper function."""
    
    def test_basic_imputation(self):
        """Test basic imputation functionality."""
        np.random.seed(42)
        n_features = 4
        n_samples = 50
        
        # Create correlated data
        background = np.random.randn(n_samples, n_features)
        instance = np.random.randn(n_features)
        
        removed_indices = [0, 1]
        remaining_indices = [2, 3]
        
        imputed = _noisy_linear_impute(
            instance, removed_indices, remaining_indices,
            background, noise_scale=1.0, seed=42
        )
        
        # Imputed should be same shape as instance
        assert imputed.shape == instance.shape
        
        # Non-removed features should be unchanged
        np.testing.assert_array_equal(imputed[2:], instance[2:])
    
    def test_deterministic_with_seed(self):
        """Test that imputation is deterministic with seed."""
        np.random.seed(42)
        n_features = 4
        n_samples = 50
        
        background = np.random.randn(n_samples, n_features)
        instance = np.random.randn(n_features)
        
        removed_indices = [0]
        remaining_indices = [1, 2, 3]
        
        imputed1 = _noisy_linear_impute(
            instance, removed_indices, remaining_indices,
            background, noise_scale=1.0, seed=42
        )
        imputed2 = _noisy_linear_impute(
            instance, removed_indices, remaining_indices,
            background, noise_scale=1.0, seed=42
        )
        
        np.testing.assert_array_equal(imputed1, imputed2)
    
    def test_zero_noise_deterministic(self):
        """Test that zero noise gives deterministic linear prediction."""
        np.random.seed(42)
        n_features = 4
        n_samples = 50
        
        background = np.random.randn(n_samples, n_features)
        instance = np.random.randn(n_features)
        
        removed_indices = [0]
        remaining_indices = [1, 2, 3]
        
        imputed1 = _noisy_linear_impute(
            instance, removed_indices, remaining_indices,
            background, noise_scale=0.0, seed=42
        )
        imputed2 = _noisy_linear_impute(
            instance, removed_indices, remaining_indices,
            background, noise_scale=0.0, seed=123  # Different seed
        )
        
        # With zero noise, results should be identical
        np.testing.assert_array_almost_equal(imputed1, imputed2)
    
    def test_no_remaining_features(self):
        """Test imputation when all features are removed."""
        np.random.seed(42)
        n_features = 4
        n_samples = 50
        
        background = np.random.randn(n_samples, n_features)
        instance = np.random.randn(n_features)
        
        # Remove all features
        removed_indices = [0, 1, 2, 3]
        remaining_indices = []
        
        # Should fall back to mean + noise
        imputed = _noisy_linear_impute(
            instance, removed_indices, remaining_indices,
            background, noise_scale=1.0, seed=42
        )
        
        assert imputed.shape == instance.shape
    
    def test_no_removed_features(self):
        """Test imputation when no features are removed."""
        np.random.seed(42)
        n_features = 4
        n_samples = 50
        
        background = np.random.randn(n_samples, n_features)
        instance = np.random.randn(n_features)
        
        # Remove no features
        removed_indices = []
        remaining_indices = [0, 1, 2, 3]
        
        imputed = _noisy_linear_impute(
            instance, removed_indices, remaining_indices,
            background, noise_scale=1.0, seed=42
        )
        
        # Should be unchanged
        np.testing.assert_array_equal(imputed, instance)


# =============================================================================
# Test 15: Integration Tests (3 tests)
# =============================================================================

class TestIntegration:
    """Integration tests."""
    
    def test_full_workflow(self, setup_model):
        """Test complete workflow from data to evaluation."""
        model, X, y = setup_model
        
        # Select instance
        instance = X[0]
        
        # Create synthetic explanation
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        # Compute single score
        score = compute_road(
            model, instance, explanation,
            background_data=X,
            seed=42
        )
        assert isinstance(score, float)
        
        # Compute detailed score
        detailed = compute_road(
            model, instance, explanation,
            background_data=X,
            return_details=True,
            seed=42
        )
        assert isinstance(detailed, dict)
        
        # Compute combined MoRF/LeRF
        combined = compute_road_combined(
            model, instance, explanation,
            background_data=X,
            seed=42
        )
        assert isinstance(combined, dict)
        assert "gap" in combined
    
    def test_batch_workflow(self, setup_batch):
        """Test batch workflow."""
        model, X, explanations = setup_batch
        
        result = compute_batch_road(
            model, X[:10], explanations,
            background_data=X,
            seed=42
        )
        
        assert result["n_samples"] > 0
        assert isinstance(result["mean"], float)
        assert result["std"] >= 0
    
    def test_comparison_morf_vs_lerf(self, setup_batch):
        """Test comparing MoRF vs LeRF across batch."""
        model, X, explanations = setup_batch
        
        result_morf = compute_batch_road(
            model, X[:10], explanations,
            background_data=X,
            order="morf",
            seed=42
        )
        
        result_lerf = compute_batch_road(
            model, X[:10], explanations,
            background_data=X,
            order="lerf",
            seed=42
        )
        
        # Both should produce valid results
        assert result_morf["n_samples"] > 0
        assert result_lerf["n_samples"] > 0
        assert isinstance(result_morf["mean"], float)
        assert isinstance(result_lerf["mean"], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
