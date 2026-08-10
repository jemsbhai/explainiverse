# tests/test_sensitivity_n.py
"""Contract tests for the Sensitivity-n estimator (Ancona et al., 2018).

Sensitivity-n measures the correlation between the sum of attributions for
random subsets of n features and the prediction change when those features
are removed. The returned correlation is conditional on the subset sampling,
baseline, and fixed output; it is not a universal explanation-quality score.

Reference: Ancona, M., Ceolini, E., Öztireli, C., & Gross, M. (2018).
Towards Better Understanding of Gradient-based Attribution Methods for
Deep Neural Networks. ICLR 2018.
"""
import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from explainiverse.adapters import SklearnAdapter
from explainiverse.core import Explanation
from explainiverse.evaluation import (
    compute_batch_sensitivity_n,
    compute_sensitivity_n,
    compute_sensitivity_n_multi,
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


def create_explanation(attributions, feature_names, target_class="0"):
    """Helper to create an Explanation object from attributions."""
    attr_dict = {fn: float(attributions[i]) for i, fn in enumerate(feature_names)}
    return Explanation(
        explainer_name="test_explainer",
        target_class=target_class,
        explanation_data={"feature_attributions": attr_dict},
        feature_names=feature_names,
    )


# =============================================================================
# Test Class: Basic Functionality
# =============================================================================


class TestSensitivityNBasic:
    """Test basic Sensitivity-n functionality."""

    def test_returns_float(self, setup_model):
        """Test that compute_sensitivity_n returns a float."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]

        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)

        score = compute_sensitivity_n(
            adapter, instance, explanation, baseline="mean", background_data=X, seed=42
        )

        assert isinstance(score, float)

    def test_valid_range(self, setup_model):
        """Test that score is in valid range [-1, 1] (correlation)."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]

        attrs = np.array([0.5, 0.3, 0.15, 0.05])
        explanation = create_explanation(attrs, feature_names)

        score = compute_sensitivity_n(
            adapter, instance, explanation, baseline="mean", background_data=X, seed=42
        )

        assert -1.0 <= score <= 1.0

    def test_deterministic_with_seed(self, setup_model):
        """Test that results are deterministic with same seed."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]

        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)

        score1 = compute_sensitivity_n(
            adapter, instance, explanation, baseline="mean", background_data=X, seed=42
        )

        score2 = compute_sensitivity_n(
            adapter, instance, explanation, baseline="mean", background_data=X, seed=42
        )

        assert score1 == score2

    @pytest.mark.parametrize("seed", [42, 123])
    def test_seed_selects_the_exact_rng_subset_sequence(self, setup_model, seed):
        """Subset sampling is exactly the local NumPy generator sequence."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]

        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)
        n_subsets = 8

        result = compute_sensitivity_n(
            adapter,
            instance,
            explanation,
            baseline=0.0,
            n=1,
            n_subsets=n_subsets,
            seed=seed,
            return_details=True,
        )

        rng = np.random.default_rng(seed)
        expected_subsets = [
            rng.choice(len(attrs), size=1, replace=False).tolist() for _ in range(n_subsets)
        ]
        expected_sums = np.array([attrs[subset].sum() for subset in expected_subsets])
        original = adapter.predict(instance.reshape(1, -1))[0, 0]
        perturbed = np.repeat(instance.reshape(1, -1), n_subsets, axis=0)
        for row, subset in zip(perturbed, expected_subsets):
            row[subset] = 0.0
        expected_drops = original - adapter.predict(perturbed)[:, 0]

        assert result["subsets"] == expected_subsets
        np.testing.assert_allclose(result["attribution_sums"], expected_sums)
        np.testing.assert_allclose(result["prediction_drops"], expected_drops)
        assert result["correlation"] == pytest.approx(
            np.corrcoef(expected_sums, expected_drops)[0, 1]
        )

    def test_use_absolute_true(self, setup_model):
        """Test with use_absolute=True (default)."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]

        attrs = np.array([-0.5, 0.3, -0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)

        score = compute_sensitivity_n(
            adapter,
            instance,
            explanation,
            baseline="mean",
            background_data=X,
            use_absolute=True,
            seed=42,
        )

        assert isinstance(score, float)

    def test_use_absolute_false(self, setup_model):
        """Test with use_absolute=False."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]

        attrs = np.array([0.5, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)

        score = compute_sensitivity_n(
            adapter,
            instance,
            explanation,
            baseline="mean",
            background_data=X,
            use_absolute=False,
            seed=42,
        )

        assert isinstance(score, float)


# =============================================================================
# Test Class: Subset Size (n) Parameter
# =============================================================================


class TestSubsetSize:
    """Test n (subset size) parameter functionality."""

    def test_default_n(self, setup_model):
        """Test default n (n_features // 4)."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]

        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)

        result = compute_sensitivity_n(
            adapter,
            instance,
            explanation,
            baseline="mean",
            background_data=X,
            seed=42,
            return_details=True,
        )

        # Default n = max(1, 4 // 4) = 1
        assert result["n"] == 1

    def test_n_equals_1(self, setup_model):
        """Test n=1 (single feature subsets)."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]

        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)

        result = compute_sensitivity_n(
            adapter,
            instance,
            explanation,
            baseline="mean",
            background_data=X,
            n=1,
            seed=42,
            return_details=True,
        )

        assert result["n"] == 1
        # Each subset should have exactly 1 feature
        for subset in result["subsets"]:
            assert len(subset) == 1

    def test_n_equals_2(self, setup_model):
        """Test n=2 (pairs of features)."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]

        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)

        result = compute_sensitivity_n(
            adapter,
            instance,
            explanation,
            baseline="mean",
            background_data=X,
            n=2,
            seed=42,
            return_details=True,
        )

        assert result["n"] == 2
        for subset in result["subsets"]:
            assert len(subset) == 2

    def test_n_equals_n_features(self, setup_model):
        """n=n_features has one unique subset, so PCC is undefined."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]
        n_features = len(feature_names)
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)

        with pytest.raises(ValueError, match="undefined for a constant input"):
            compute_sensitivity_n(
                adapter,
                instance,
                explanation,
                baseline="mean",
                background_data=X,
                n=n_features,
                seed=42,
                return_details=True,
            )

    def test_n_exceeds_features_rejected(self, setup_model):
        """An impossible subset cardinality is rejected."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]

        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)

        with pytest.raises(ValueError, match="n cannot exceed"):
            compute_sensitivity_n(
                adapter,
                instance,
                explanation,
                baseline="mean",
                background_data=X,
                n=100,
                seed=42,
                return_details=True,
            )

    @pytest.mark.parametrize("subset_size", [1, 2])
    def test_n_fixes_exact_subset_cardinality_and_equation(self, setup_model, subset_size):
        """The returned arrays obey the requested subset size and PCC equation."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]

        attrs = np.array([0.5, 0.3, 0.15, 0.05])
        explanation = create_explanation(attrs, feature_names)

        result = compute_sensitivity_n(
            adapter,
            instance,
            explanation,
            baseline="mean",
            background_data=X,
            n=subset_size,
            n_subsets=20,
            seed=42,
            return_details=True,
        )

        assert result["n"] == subset_size
        assert all(len(subset) == subset_size for subset in result["subsets"])
        expected_sums = np.array([attrs[subset].sum() for subset in result["subsets"]])
        np.testing.assert_allclose(result["attribution_sums"], expected_sums)
        assert result["correlation"] == pytest.approx(
            np.corrcoef(expected_sums, result["prediction_drops"])[0, 1]
        )


# =============================================================================
# Test Class: Number of Subsets Parameter
# =============================================================================


class TestNSubsets:
    """Test n_subsets parameter functionality."""

    def test_default_n_subsets(self, setup_model):
        """Test default n_subsets (100)."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]

        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)

        result = compute_sensitivity_n(
            adapter,
            instance,
            explanation,
            baseline="mean",
            background_data=X,
            seed=42,
            return_details=True,
        )

        assert result["n_subsets"] == 100
        assert len(result["subsets"]) == 100

    def test_custom_n_subsets(self, setup_model):
        """Test custom n_subsets."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]

        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)

        result = compute_sensitivity_n(
            adapter,
            instance,
            explanation,
            baseline="mean",
            background_data=X,
            n_subsets=50,
            seed=42,
            return_details=True,
        )

        assert result["n_subsets"] == 50
        assert len(result["subsets"]) == 50

    def test_small_n_subsets(self, setup_model):
        """Test with small n_subsets."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]

        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)

        result = compute_sensitivity_n(
            adapter,
            instance,
            explanation,
            baseline="mean",
            background_data=X,
            n_subsets=10,
            seed=42,
            return_details=True,
        )

        assert result["n_subsets"] == 10


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

        result = compute_sensitivity_n(
            adapter,
            instance,
            explanation,
            baseline="mean",
            background_data=X,
            seed=42,
            return_details=True,
        )

        assert isinstance(result, dict)
        assert "correlation" in result
        assert "p_value" in result
        assert "attribution_sums" in result
        assert "prediction_drops" in result
        assert "subsets" in result
        assert "n" in result
        assert "n_subsets" in result

    def test_arrays_correct_length(self, setup_model):
        """Test that arrays have correct length."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]

        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)

        result = compute_sensitivity_n(
            adapter,
            instance,
            explanation,
            baseline="mean",
            background_data=X,
            n_subsets=50,
            seed=42,
            return_details=True,
        )

        assert len(result["attribution_sums"]) == 50
        assert len(result["prediction_drops"]) == 50
        assert len(result["subsets"]) == 50

    def test_correlation_matches_standalone(self, setup_model):
        """Test that correlation from details matches standalone call."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]

        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)

        standalone = compute_sensitivity_n(
            adapter,
            instance,
            explanation,
            baseline="mean",
            background_data=X,
            seed=42,
            return_details=False,
        )

        detailed = compute_sensitivity_n(
            adapter,
            instance,
            explanation,
            baseline="mean",
            background_data=X,
            seed=42,
            return_details=True,
        )

        assert np.isclose(standalone, detailed["correlation"])

    def test_p_value_valid(self, setup_model):
        """Test that p-value is in valid range [0, 1]."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]

        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)

        result = compute_sensitivity_n(
            adapter,
            instance,
            explanation,
            baseline="mean",
            background_data=X,
            seed=42,
            return_details=True,
        )

        assert 0.0 <= result["p_value"] <= 1.0


# =============================================================================
# Test Class: Multi-n Variant
# =============================================================================


class TestSensitivityNMulti:
    """Test compute_sensitivity_n_multi functionality."""

    def test_returns_dict(self, setup_model):
        """Test that multi variant returns dict."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]

        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)

        result = compute_sensitivity_n_multi(
            adapter, instance, explanation, baseline="mean", background_data=X, seed=42
        )

        assert isinstance(result, dict)
        assert "mean" in result
        assert "scores" in result
        assert "n_values" in result

    def test_default_n_values(self, setup_model):
        """Test default n_values."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]

        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)

        result = compute_sensitivity_n_multi(
            adapter, instance, explanation, baseline="mean", background_data=X, seed=42
        )

        # Should have multiple n values
        assert len(result["n_values"]) >= 1
        assert len(result["scores"]) == len(result["n_values"])

    def test_custom_n_values(self, setup_model):
        """Test custom n_values."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]

        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)

        result = compute_sensitivity_n_multi(
            adapter,
            instance,
            explanation,
            baseline="mean",
            background_data=X,
            n_values=[1, 2, 3],
            seed=42,
        )

        assert result["n_values"] == [1, 2, 3]
        assert 1 in result["scores"]
        assert 2 in result["scores"]
        assert 3 in result["scores"]

    def test_mean_is_average_of_scores(self, setup_model):
        """Test that mean equals average of individual scores."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]

        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)

        result = compute_sensitivity_n_multi(
            adapter,
            instance,
            explanation,
            baseline="mean",
            background_data=X,
            n_values=[1, 2],
            seed=42,
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

        score = compute_sensitivity_n(
            adapter, instance, explanation, baseline="mean", background_data=X, seed=42
        )

        assert isinstance(score, float)

    def test_baseline_median(self, setup_model):
        """Test baseline='median'."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]

        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)

        score = compute_sensitivity_n(
            adapter, instance, explanation, baseline="median", background_data=X, seed=42
        )

        assert isinstance(score, float)

    def test_baseline_scalar(self, setup_model):
        """Test baseline as scalar."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]

        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)

        score = compute_sensitivity_n(
            adapter, instance, explanation, baseline=0.0, background_data=X, seed=42
        )

        assert isinstance(score, float)

    def test_baseline_array(self, setup_model):
        """Test baseline as array."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]

        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)

        baseline_values = np.mean(X, axis=0)
        score = compute_sensitivity_n(
            adapter, instance, explanation, baseline=baseline_values, seed=42
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

        score = compute_sensitivity_n(
            adapter, instance, explanation, baseline=custom_baseline, background_data=X, seed=42
        )

        assert isinstance(score, float)


# =============================================================================
# Test Class: Batch Operations
# =============================================================================


class TestBatchOperations:
    """Test batch Sensitivity-n computation."""

    def test_batch_returns_dict(self, setup_batch):
        """Test that batch returns dict."""
        adapter, X, explanations, feature_names = setup_batch

        result = compute_batch_sensitivity_n(
            adapter,
            X[: len(explanations)],
            explanations,
            baseline="mean",
            seed=42,
            background_data=X[len(explanations) :],
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

        result = compute_batch_sensitivity_n(
            adapter,
            X[: len(explanations)],
            explanations,
            baseline="mean",
            max_samples=3,
            seed=42,
            background_data=X[len(explanations) :],
        )

        assert result["n_samples"] <= 3

    def test_batch_values_are_floats(self, setup_batch):
        """Test that batch results are floats."""
        adapter, X, explanations, feature_names = setup_batch

        result = compute_batch_sensitivity_n(
            adapter,
            X[: len(explanations)],
            explanations,
            baseline="mean",
            seed=42,
            background_data=X[len(explanations) :],
        )

        assert isinstance(result["mean"], float)
        assert isinstance(result["std"], float)
        assert isinstance(result["min"], float)
        assert isinstance(result["max"], float)

    def test_batch_with_custom_n(self, setup_batch):
        """Test batch with custom n."""
        adapter, X, explanations, feature_names = setup_batch

        result = compute_batch_sensitivity_n(
            adapter,
            X[: len(explanations)],
            explanations,
            baseline="mean",
            n=2,
            seed=42,
            background_data=X[len(explanations) :],
        )

        assert result["n_samples"] > 0


# =============================================================================
# Test Class: Multiple Model Types
# =============================================================================


class TestMultipleModels:
    """Test Sensitivity-n with different model types."""

    def test_random_forest(self, setup_iris):
        """Test with RandomForestClassifier."""
        X, y, feature_names = setup_iris

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model)

        instance = X[0]
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)

        score = compute_sensitivity_n(
            adapter, instance, explanation, baseline="mean", background_data=X, seed=42
        )

        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    def test_gradient_boosting(self, setup_iris):
        """Test with GradientBoostingClassifier."""
        X, y, feature_names = setup_iris

        model = GradientBoostingClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model)

        instance = X[0]
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)

        score = compute_sensitivity_n(
            adapter, instance, explanation, baseline="mean", background_data=X, seed=42
        )

        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0


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

        score = compute_sensitivity_n(
            adapter, instance, explanation, baseline="mean", background_data=X_small, seed=42
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

        score = compute_sensitivity_n(
            adapter, instance, explanation, baseline="mean", background_data=X, seed=42
        )

        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    def test_identical_attributions(self, setup_model):
        """Test when all attributions are identical."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]

        attrs = np.array([0.25, 0.25, 0.25, 0.25])
        explanation = create_explanation(attrs, feature_names)

        with pytest.raises(ValueError, match="undefined for a constant input"):
            compute_sensitivity_n(
                adapter, instance, explanation, baseline="mean", background_data=X, seed=42
            )

    def test_zero_attributions(self, setup_model):
        """Test when all attributions are zero."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]

        attrs = np.array([0.0, 0.0, 0.0, 0.0])
        explanation = create_explanation(attrs, feature_names)

        with pytest.raises(ValueError, match="undefined for a constant input"):
            compute_sensitivity_n(
                adapter, instance, explanation, baseline="mean", background_data=X, seed=42
            )

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

        with pytest.raises(ValueError, match="undefined for a constant input"):
            compute_sensitivity_n(
                adapter, instance, explanation, baseline="mean", background_data=X, seed=42
            )

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

        score = compute_sensitivity_n(
            adapter, instance, explanation, baseline="mean", background_data=X, n=1, seed=42
        )

        assert isinstance(score, float)


# =============================================================================
# Test Class: Semantic Validation
# =============================================================================


class TestSemanticValidation:
    """Integration smoke tests for arbitrary attribution orderings."""

    def test_model_coefficient_ordering_returns_float(self, setup_iris):
        """A coefficient-derived ordering satisfies the numeric API contract."""
        X, y, feature_names = setup_iris

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model)

        instance = X[0]

        # Use model coefficients as one deterministic attribution vector.
        true_importance = np.abs(model.coef_[0])
        good_exp = create_explanation(true_importance, feature_names)

        good_score = compute_sensitivity_n(
            adapter, instance, good_exp, baseline="mean", background_data=X, n_subsets=200, seed=42
        )

        assert isinstance(good_score, float)

    def test_two_orderings_return_float_scores(self, setup_iris):
        """Two attribution vectors both satisfy the numeric API contract."""
        X, y, feature_names = setup_iris

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model)

        instance = X[0]

        # First deterministic attribution vector.
        true_importance = np.abs(model.coef_[0])
        good_exp = create_explanation(true_importance, feature_names)

        # Second seeded attribution vector; no quality comparison is asserted.
        np.random.seed(99)
        random_attrs = np.random.rand(4)
        random_exp = create_explanation(random_attrs, feature_names)

        good_score = compute_sensitivity_n(
            adapter, instance, good_exp, baseline="mean", background_data=X, n_subsets=200, seed=42
        )

        random_score = compute_sensitivity_n(
            adapter,
            instance,
            random_exp,
            baseline="mean",
            background_data=X,
            n_subsets=200,
            seed=42,
        )

        # Both should be valid floats
        assert isinstance(good_score, float)
        assert isinstance(random_score, float)


# =============================================================================
# Test Class: Target Class
# =============================================================================


class TestTargetClass:
    """Test target_class parameter."""

    def test_explicit_target_columns_match_exact_probability_drops(self, setup_iris):
        """Every sampled drop uses the exact explicitly selected output column."""
        X, y, feature_names = setup_iris

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model)

        instance = X[0]
        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        seed = 42
        n_subsets = 12
        rng = np.random.default_rng(seed)
        expected_subsets = [
            rng.choice(len(attrs), size=1, replace=False).tolist() for _ in range(n_subsets)
        ]
        perturbed = np.repeat(instance.reshape(1, -1), n_subsets, axis=0)
        for row, subset in zip(perturbed, expected_subsets):
            row[subset] = 0.0
        original_probabilities = adapter.predict(instance.reshape(1, -1))[0]
        perturbed_probabilities = adapter.predict(perturbed)

        for target in [0, 1, 2]:
            explanation = create_explanation(attrs, feature_names, target_class=target)
            result = compute_sensitivity_n(
                adapter,
                instance,
                explanation,
                baseline=0.0,
                target_class=target,
                n=1,
                n_subsets=n_subsets,
                seed=seed,
                return_details=True,
            )

            expected_drops = original_probabilities[target] - perturbed_probabilities[:, target]
            assert result["subsets"] == expected_subsets
            np.testing.assert_allclose(result["prediction_drops"], expected_drops)
            assert result["correlation"] == pytest.approx(
                np.corrcoef(result["attribution_sums"], expected_drops)[0, 1]
            )

    def test_default_target_uses_explanation_identity(self, setup_model):
        """The default uses the output identified by the explanation."""
        adapter, X, y, feature_names = setup_model
        instance = X[0]

        attrs = np.array([0.4, 0.3, 0.2, 0.1])
        explanation = create_explanation(attrs, feature_names)

        default_score = compute_sensitivity_n(
            adapter, instance, explanation, baseline="mean", background_data=X, seed=42
        )

        explicit_score = compute_sensitivity_n(
            adapter,
            instance,
            explanation,
            baseline="mean",
            background_data=X,
            target_class=0,
            seed=42,
        )

        assert np.isclose(default_score, explicit_score)


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
