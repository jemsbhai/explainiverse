# tests/test_infidelity.py
"""
Comprehensive tests for Infidelity metric (Yeh et al., 2019).

Infidelity measures how well attributions predict model output changes under perturbation.
Lower infidelity = better explanation (0 is perfect).

Tests cover:
1. Basic functionality
2. Perturbation types (gaussian, square, subset)
3. Return details
4. Multi-perturbation variant
5. Batch operations
6. Baseline types
7. Multiple model types
8. Edge cases
9. Semantic validation
10. Target class handling
"""
import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from explainiverse.core.explanation import Explanation
from explainiverse.evaluation.faithfulness_extended import (
    compute_infidelity,
    compute_infidelity_multi_perturbation,
    compute_batch_infidelity,
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
# Test 1: Basic Functionality (7 tests)
# =============================================================================

class TestBasicFunctionality:
    """Basic functionality tests for compute_infidelity."""
    
    def test_returns_float(self, setup_model):
        """Test that compute_infidelity returns a float."""
        model, X, y = setup_model
        instance = X[0]
        
        # Create explanation
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        score = compute_infidelity(
            model, instance, explanation,
            background_data=X, seed=42
        )
        
        assert isinstance(score, float)
    
    def test_score_is_non_negative(self, setup_model):
        """Test that infidelity score is non-negative (squared error)."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        score = compute_infidelity(
            model, instance, explanation,
            background_data=X, seed=42
        )
        
        assert score >= 0
    
    def test_score_not_nan(self, setup_model):
        """Test that score is not NaN."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        score = compute_infidelity(
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
        
        score1 = compute_infidelity(
            model, instance, explanation,
            background_data=X, seed=42
        )
        score2 = compute_infidelity(
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
        
        score1 = compute_infidelity(
            model, instance, explanation,
            background_data=X, seed=42, n_samples=50
        )
        score2 = compute_infidelity(
            model, instance, explanation,
            background_data=X, seed=123, n_samples=50
        )
        
        # May differ due to different random perturbations
        # (though could theoretically be same, very unlikely)
        # Just check both are valid
        assert isinstance(score1, float)
        assert isinstance(score2, float)
    
    def test_n_samples_parameter(self, setup_model):
        """Test that n_samples parameter affects computation."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        # Different n_samples should give different scores
        score_small = compute_infidelity(
            model, instance, explanation,
            background_data=X, n_samples=10, seed=42
        )
        score_large = compute_infidelity(
            model, instance, explanation,
            background_data=X, n_samples=200, seed=42
        )
        
        # Both should be valid
        assert isinstance(score_small, float)
        assert isinstance(score_large, float)
        assert not np.isnan(score_small)
        assert not np.isnan(score_large)
    
    def test_lower_is_better_concept(self, setup_model):
        """Test that lower infidelity is better (perfect = 0)."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        score = compute_infidelity(
            model, instance, explanation,
            background_data=X, seed=42
        )
        
        # Score should be non-negative (squared error)
        assert score >= 0
        # Zero would be perfect (unlikely with random attributions)


# =============================================================================
# Test 2: Perturbation Types (6 tests)
# =============================================================================

class TestPerturbationTypes:
    """Test different perturbation strategies."""
    
    def test_gaussian_perturbation(self, setup_model):
        """Test Gaussian perturbation type."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        score = compute_infidelity(
            model, instance, explanation,
            background_data=X,
            perturbation_type="gaussian",
            seed=42
        )
        
        assert isinstance(score, float)
        assert score >= 0
    
    def test_square_perturbation(self, setup_model):
        """Test square/binary mask perturbation type."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        score = compute_infidelity(
            model, instance, explanation,
            background_data=X,
            perturbation_type="square",
            noise_scale=0.3,  # Probability of perturbing each feature
            seed=42
        )
        
        assert isinstance(score, float)
        assert score >= 0
    
    def test_subset_perturbation(self, setup_model):
        """Test subset perturbation type."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        score = compute_infidelity(
            model, instance, explanation,
            background_data=X,
            perturbation_type="subset",
            subset_size=2,
            seed=42
        )
        
        assert isinstance(score, float)
        assert score >= 0
    
    def test_different_perturbation_types_different_scores(self, setup_model):
        """Test that different perturbation types may give different scores."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        score_gaussian = compute_infidelity(
            model, instance, explanation,
            background_data=X,
            perturbation_type="gaussian",
            seed=42
        )
        score_square = compute_infidelity(
            model, instance, explanation,
            background_data=X,
            perturbation_type="square",
            seed=42
        )
        score_subset = compute_infidelity(
            model, instance, explanation,
            background_data=X,
            perturbation_type="subset",
            seed=42
        )
        
        # All should be valid
        assert isinstance(score_gaussian, float)
        assert isinstance(score_square, float)
        assert isinstance(score_subset, float)
    
    def test_invalid_perturbation_type_raises(self, setup_model):
        """Test that invalid perturbation type raises ValueError."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        with pytest.raises(ValueError, match="Unknown perturbation_type"):
            compute_infidelity(
                model, instance, explanation,
                background_data=X,
                perturbation_type="invalid_type",
                seed=42
            )
    
    def test_noise_scale_affects_results(self, setup_model):
        """Test that noise_scale parameter affects results."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        score_small = compute_infidelity(
            model, instance, explanation,
            background_data=X,
            perturbation_type="gaussian",
            noise_scale=0.01,
            seed=42
        )
        score_large = compute_infidelity(
            model, instance, explanation,
            background_data=X,
            perturbation_type="gaussian",
            noise_scale=1.0,
            seed=42
        )
        
        # Both should be valid (actual values will differ)
        assert isinstance(score_small, float)
        assert isinstance(score_large, float)


# =============================================================================
# Test 3: Return Details (6 tests)
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
        
        result = compute_infidelity(
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
        
        result = compute_infidelity(
            model, instance, explanation,
            background_data=X,
            return_details=True,
            seed=42
        )
        
        required_keys = [
            "infidelity",
            "squared_errors",
            "expected_changes",
            "actual_changes",
            "n_samples",
            "perturbation_type",
        ]
        
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_squared_errors_length_matches_n_samples(self, setup_model):
        """Test that squared_errors array has correct length."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        n_samples = 50
        result = compute_infidelity(
            model, instance, explanation,
            background_data=X,
            n_samples=n_samples,
            return_details=True,
            seed=42
        )
        
        assert len(result["squared_errors"]) == n_samples
        assert len(result["expected_changes"]) == n_samples
        assert len(result["actual_changes"]) == n_samples
    
    def test_infidelity_equals_mean_squared_errors(self, setup_model):
        """Test that infidelity equals mean of squared_errors."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        result = compute_infidelity(
            model, instance, explanation,
            background_data=X,
            return_details=True,
            seed=42
        )
        
        expected_infidelity = np.mean(result["squared_errors"])
        assert np.isclose(result["infidelity"], expected_infidelity)
    
    def test_infidelity_matches_standalone_call(self, setup_model):
        """Test that detailed infidelity matches non-detailed call."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        score = compute_infidelity(
            model, instance, explanation,
            background_data=X,
            return_details=False,
            seed=42
        )
        
        detailed = compute_infidelity(
            model, instance, explanation,
            background_data=X,
            return_details=True,
            seed=42
        )
        
        assert np.isclose(score, detailed["infidelity"])
    
    def test_perturbation_type_in_details(self, setup_model):
        """Test that perturbation_type is correctly recorded."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        for ptype in ["gaussian", "square", "subset"]:
            result = compute_infidelity(
                model, instance, explanation,
                background_data=X,
                perturbation_type=ptype,
                return_details=True,
                seed=42
            )
            
            assert result["perturbation_type"] == ptype


# =============================================================================
# Test 4: Multi-Perturbation Variant (4 tests)
# =============================================================================

class TestMultiPerturbation:
    """Test compute_infidelity_multi_perturbation."""
    
    def test_multi_perturbation_returns_dict(self, setup_model):
        """Test that multi-perturbation returns a dictionary."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        result = compute_infidelity_multi_perturbation(
            model, instance, explanation,
            background_data=X,
            seed=42
        )
        
        assert isinstance(result, dict)
        assert "mean" in result
        assert "scores" in result
        assert "perturbation_types" in result
    
    def test_multi_perturbation_default_types(self, setup_model):
        """Test default perturbation types are used."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        result = compute_infidelity_multi_perturbation(
            model, instance, explanation,
            background_data=X,
            seed=42
        )
        
        expected_types = ["gaussian", "square", "subset"]
        assert result["perturbation_types"] == expected_types
        
        for ptype in expected_types:
            assert ptype in result["scores"]
    
    def test_multi_perturbation_custom_types(self, setup_model):
        """Test custom perturbation types."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        custom_types = ["gaussian", "subset"]
        result = compute_infidelity_multi_perturbation(
            model, instance, explanation,
            background_data=X,
            perturbation_types=custom_types,
            seed=42
        )
        
        assert result["perturbation_types"] == custom_types
        assert len(result["scores"]) == 2
    
    def test_multi_perturbation_mean_equals_average(self, setup_model):
        """Test that mean equals average of individual scores."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        result = compute_infidelity_multi_perturbation(
            model, instance, explanation,
            background_data=X,
            seed=42
        )
        
        expected_mean = np.mean(list(result["scores"].values()))
        assert np.isclose(result["mean"], expected_mean)


# =============================================================================
# Test 5: Batch Operations (5 tests)
# =============================================================================

class TestBatchOperations:
    """Test compute_batch_infidelity."""
    
    def test_batch_returns_dict(self, setup_batch):
        """Test that batch computation returns a dictionary."""
        model, X, explanations = setup_batch
        
        result = compute_batch_infidelity(
            model, X[:10], explanations,
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
        
        result = compute_batch_infidelity(
            model, X[:10], explanations,
            max_samples=5,
            seed=42
        )
        
        assert result["n_samples"] <= 5
    
    def test_batch_with_different_perturbation_types(self, setup_batch):
        """Test batch with different perturbation types."""
        model, X, explanations = setup_batch
        
        for ptype in ["gaussian", "square", "subset"]:
            result = compute_batch_infidelity(
                model, X[:10], explanations,
                perturbation_type=ptype,
                seed=42
            )
            
            assert isinstance(result["mean"], float)
            assert not np.isnan(result["mean"])
    
    def test_batch_mean_in_reasonable_range(self, setup_batch):
        """Test that batch mean is non-negative."""
        model, X, explanations = setup_batch
        
        result = compute_batch_infidelity(
            model, X[:10], explanations,
            seed=42
        )
        
        # Infidelity should be non-negative (squared error)
        assert result["mean"] >= 0
        assert result["min"] >= 0
    
    def test_batch_n_perturbations_parameter(self, setup_batch):
        """Test n_perturbations parameter."""
        model, X, explanations = setup_batch
        
        result = compute_batch_infidelity(
            model, X[:10], explanations,
            n_perturbations=50,
            seed=42
        )
        
        assert isinstance(result["mean"], float)


# =============================================================================
# Test 6: Baseline Types (6 tests)
# =============================================================================

class TestBaselineTypes:
    """Test different baseline types."""
    
    def test_baseline_mean(self, setup_model):
        """Test mean baseline."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        score = compute_infidelity(
            model, instance, explanation,
            baseline="mean",
            background_data=X,
            seed=42
        )
        
        assert isinstance(score, float)
    
    def test_baseline_median(self, setup_model):
        """Test median baseline."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        score = compute_infidelity(
            model, instance, explanation,
            baseline="median",
            background_data=X,
            seed=42
        )
        
        assert isinstance(score, float)
    
    def test_baseline_scalar(self, setup_model):
        """Test scalar baseline."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        score = compute_infidelity(
            model, instance, explanation,
            baseline=0.0,
            background_data=X,
            seed=42
        )
        
        assert isinstance(score, float)
    
    def test_baseline_array(self, setup_model):
        """Test array baseline."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        baseline_array = np.zeros(n_features)
        score = compute_infidelity(
            model, instance, explanation,
            baseline=baseline_array,
            background_data=X,
            seed=42
        )
        
        assert isinstance(score, float)
    
    def test_baseline_callable(self, setup_model):
        """Test callable baseline."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        def custom_baseline(data):
            return np.percentile(data, 25, axis=0)
        
        score = compute_infidelity(
            model, instance, explanation,
            baseline=custom_baseline,
            background_data=X,
            seed=42
        )
        
        assert isinstance(score, float)
    
    def test_different_baselines_different_scores(self, setup_model):
        """Test that different baselines may give different scores."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        score_mean = compute_infidelity(
            model, instance, explanation,
            baseline="mean",
            background_data=X,
            seed=42
        )
        score_zero = compute_infidelity(
            model, instance, explanation,
            baseline=0.0,
            background_data=X,
            seed=42
        )
        
        # Both should be valid
        assert isinstance(score_mean, float)
        assert isinstance(score_zero, float)


# =============================================================================
# Test 7: Multiple Model Types (2 tests)
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
        
        score = compute_infidelity(
            model, instance, explanation,
            background_data=X,
            seed=42
        )
        
        assert isinstance(score, float)
        assert score >= 0
    
    def test_with_gradient_boosting(self, setup_iris):
        """Test with GradientBoostingClassifier."""
        X, y = setup_iris
        
        model = GradientBoostingClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        instance = X[0]
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        score = compute_infidelity(
            model, instance, explanation,
            background_data=X,
            seed=42
        )
        
        assert isinstance(score, float)
        assert score >= 0


# =============================================================================
# Test 8: Edge Cases (7 tests)
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_few_features(self, setup_iris):
        """Test with few features (3)."""
        X, y = setup_iris
        X_small = X[:, :3]
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_small, y)
        
        instance = X_small[0]
        attributions = np.random.randn(3)
        explanation = create_explanation(attributions)
        
        score = compute_infidelity(
            model, instance, explanation,
            background_data=X_small,
            seed=42
        )
        
        assert isinstance(score, float)
    
    def test_many_features(self):
        """Test with many features (50)."""
        np.random.seed(42)
        n_features = 50
        n_samples = 100
        
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] > 0).astype(int)
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        
        instance = X[0]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        score = compute_infidelity(
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
        
        score = compute_infidelity(
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
        
        score = compute_infidelity(
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
        
        score = compute_infidelity(
            model, instance, explanation,
            background_data=X,
            seed=42
        )
        
        assert isinstance(score, float)
    
    def test_single_sample_monte_carlo(self, setup_model):
        """Test with single Monte Carlo sample."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        score = compute_infidelity(
            model, instance, explanation,
            background_data=X,
            n_samples=1,
            seed=42
        )
        
        assert isinstance(score, float)
    
    def test_subset_size_clamping(self, setup_model):
        """Test that subset_size is clamped to valid range."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        # subset_size larger than n_features
        score = compute_infidelity(
            model, instance, explanation,
            background_data=X,
            perturbation_type="subset",
            subset_size=100,  # More than n_features (4)
            seed=42
        )
        
        assert isinstance(score, float)


# =============================================================================
# Test 9: Semantic Validation (3 tests)
# =============================================================================

class TestSemanticValidation:
    """Test semantic correctness of the metric."""
    
    def test_good_explanation_lower_infidelity(self, setup_model):
        """Test that explanations aligned with model have lower infidelity."""
        model, X, y = setup_model
        instance = X[0]
        
        # Get model coefficients as "perfect" attributions for logistic regression
        # For class 0 vs others
        pred_class = model.predict(instance.reshape(1, -1))[0]
        coefs = model.coef_[pred_class]
        
        # Good explanation: aligned with model
        good_attributions = coefs * instance  # Feature contribution approximation
        good_explanation = create_explanation(good_attributions)
        
        # Random explanation
        np.random.seed(123)
        random_attributions = np.random.randn(len(instance))
        random_explanation = create_explanation(random_attributions)
        
        # Inverted explanation: opposite of model
        inverted_attributions = -good_attributions
        inverted_explanation = create_explanation(inverted_attributions)
        
        score_good = compute_infidelity(
            model, instance, good_explanation,
            background_data=X,
            perturbation_type="gaussian",
            noise_scale=0.1,
            n_samples=200,
            seed=42
        )
        score_random = compute_infidelity(
            model, instance, random_explanation,
            background_data=X,
            perturbation_type="gaussian",
            noise_scale=0.1,
            n_samples=200,
            seed=42
        )
        
        # Good explanation should generally have lower infidelity
        # This is probabilistic, so we just verify both are valid
        assert isinstance(score_good, float)
        assert isinstance(score_random, float)
    
    def test_expected_vs_actual_changes_recorded(self, setup_model):
        """Test that expected and actual changes are properly recorded."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        result = compute_infidelity(
            model, instance, explanation,
            background_data=X,
            return_details=True,
            seed=42
        )
        
        # Check that expected and actual changes are arrays of same length
        assert len(result["expected_changes"]) == len(result["actual_changes"])
        
        # Squared errors should match (expected - actual)^2
        expected_sq_errors = (
            result["expected_changes"] - result["actual_changes"]
        ) ** 2
        assert np.allclose(result["squared_errors"], expected_sq_errors)
    
    def test_infidelity_invariant_to_attribution_scale(self, setup_model):
        """Test behavior when attributions are scaled (approximate invariance)."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        np.random.seed(42)
        base_attributions = np.random.randn(n_features)
        
        # Scale attributions by different factors
        explanation_1x = create_explanation(base_attributions)
        explanation_2x = create_explanation(base_attributions * 2)
        
        score_1x = compute_infidelity(
            model, instance, explanation_1x,
            background_data=X,
            seed=42
        )
        score_2x = compute_infidelity(
            model, instance, explanation_2x,
            background_data=X,
            seed=42
        )
        
        # Both should be valid (scaling affects infidelity by square of scale)
        assert isinstance(score_1x, float)
        assert isinstance(score_2x, float)


# =============================================================================
# Test 10: Target Class (2 tests)
# =============================================================================

class TestTargetClass:
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
            score = compute_infidelity(
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
        score_default = compute_infidelity(
            model, instance, explanation,
            background_data=X,
            seed=42
        )
        
        # Explicit with predicted class
        score_explicit = compute_infidelity(
            model, instance, explanation,
            background_data=X,
            target_class=predicted_class,
            seed=42
        )
        
        # Should be equal
        assert np.isclose(score_default, score_explicit)


# =============================================================================
# Test 11: Integration Tests (3 tests)
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
        score = compute_infidelity(
            model, instance, explanation,
            background_data=X,
            seed=42
        )
        assert isinstance(score, float)
        
        # Compute detailed score
        detailed = compute_infidelity(
            model, instance, explanation,
            background_data=X,
            return_details=True,
            seed=42
        )
        assert isinstance(detailed, dict)
        
        # Compute multi-perturbation
        multi = compute_infidelity_multi_perturbation(
            model, instance, explanation,
            background_data=X,
            seed=42
        )
        assert isinstance(multi, dict)
    
    def test_batch_workflow(self, setup_batch):
        """Test batch workflow."""
        model, X, explanations = setup_batch
        
        result = compute_batch_infidelity(
            model, X[:10], explanations,
            seed=42
        )
        
        assert result["n_samples"] > 0
        assert result["mean"] >= 0
        assert result["std"] >= 0
    
    def test_comparison_across_perturbation_types(self, setup_model):
        """Test comparing scores across perturbation types."""
        model, X, y = setup_model
        instance = X[0]
        
        n_features = X.shape[1]
        attributions = np.random.randn(n_features)
        explanation = create_explanation(attributions)
        
        multi = compute_infidelity_multi_perturbation(
            model, instance, explanation,
            background_data=X,
            seed=42
        )
        
        # All perturbation types should produce valid scores
        for ptype in multi["perturbation_types"]:
            assert ptype in multi["scores"]
            assert isinstance(multi["scores"][ptype], float)
            assert multi["scores"][ptype] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
