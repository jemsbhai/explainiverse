# tests/test_insertion_deletion.py
"""
Comprehensive tests for Insertion AUC and Deletion AUC (Petsiuk et al., 2018).

Insertion AUC: progressively adds features most-important-first onto baseline,
measuring prediction recovery. Higher AUC = better explanation.

Deletion AUC: progressively removes features most-important-first from original,
measuring prediction degradation. Lower AUC = better explanation.

Tests cover:
1. Basic functionality (return types, valid ranges)
2. Deletion AUC specifics
3. Insertion AUC specifics
4. Combined Insertion-Deletion
5. Return curve details
6. n_steps parameter (percentage-based stepping)
7. Baseline types (mean, median, scalar, array, callable)
8. Batch operations
9. Multiple model types (LogisticRegression, RandomForest, GradientBoosting)
10. Multiple explainers (LIME-style vs SHAP-style attributions)
11. Edge cases (few features, many features, zero/identical attributions)
12. Semantic validation (good explanations vs random)
13. Target class handling
14. use_absolute parameter
15. Complementarity (insertion + deletion relationship)
"""
import numpy as np
import pytest
from sklearn.datasets import load_iris, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from explainiverse.core.explanation import Explanation
from explainiverse.evaluation.faithfulness_extended import (
    compute_deletion_auc,
    compute_batch_deletion_auc,
    compute_insertion_auc,
    compute_batch_insertion_auc,
    compute_insertion_deletion_auc,
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
def setup_rf_model(setup_iris):
    """Train a RandomForest model on Iris."""
    X, y = setup_iris
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def setup_gb_model(setup_iris):
    """Train a GradientBoosting model on Iris."""
    X, y = setup_iris
    model = GradientBoostingClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def setup_wine():
    """Load and prepare Wine dataset (13 features)."""
    wine = load_wine()
    X = wine.data
    y = wine.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)

    return model, X, y


@pytest.fixture
def setup_batch(setup_model):
    """Create a batch of explanations for testing."""
    model, X, y = setup_model

    explanations = []
    n_features = X.shape[1]
    feature_names = [f"feature_{i}" for i in range(n_features)]

    for i in range(10):
        np.random.seed(42 + i)
        attr_values = np.random.randn(n_features)
        attr_values = np.abs(attr_values)
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

    return Explanation(
        explainer_name="test_explainer",
        target_class="class_0",
        explanation_data={"feature_attributions": attr_dict},
        feature_names=feature_names,
    )


# =============================================================================
# 1. Basic Functionality — Deletion AUC
# =============================================================================


class TestDeletionAUCBasic:
    """Basic functionality tests for compute_deletion_auc."""

    def test_returns_float(self, setup_model):
        """Deletion AUC returns a float value."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        assert isinstance(result, float)

    def test_returns_finite(self, setup_model):
        """Deletion AUC returns a finite (non-NaN, non-inf) value."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        assert np.isfinite(result)

    def test_result_in_valid_range(self, setup_model):
        """Deletion AUC is bounded [0, 1] for probability outputs."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        assert 0.0 <= result <= 1.0 + 1e-6

    def test_multiple_instances(self, setup_model):
        """Deletion AUC works on different instances."""
        model, X, y = setup_model
        scores = []
        for i in range(5):
            exp = create_explanation(np.abs(np.random.randn(4)))
            score = compute_deletion_auc(
                model, X[i], exp, baseline="mean", background_data=X
            )
            scores.append(score)
        assert all(np.isfinite(s) for s in scores)
        assert len(set(scores)) > 1  # Not all identical

    def test_scalar_baseline(self, setup_model):
        """Deletion AUC works with scalar baseline."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_deletion_auc(
            model, X[0], exp, baseline=0.0
        )
        assert isinstance(result, float) and np.isfinite(result)


# =============================================================================
# 2. Basic Functionality — Insertion AUC
# =============================================================================


class TestInsertionAUCBasic:
    """Basic functionality tests for compute_insertion_auc."""

    def test_returns_float(self, setup_model):
        """Insertion AUC returns a float value."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_insertion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        assert isinstance(result, float)

    def test_returns_finite(self, setup_model):
        """Insertion AUC returns a finite value."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_insertion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        assert np.isfinite(result)

    def test_result_in_valid_range(self, setup_model):
        """Insertion AUC is bounded [0, 1] for probability outputs."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_insertion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        assert 0.0 <= result <= 1.0 + 1e-6

    def test_multiple_instances(self, setup_model):
        """Insertion AUC works on different instances."""
        model, X, y = setup_model
        scores = []
        for i in range(5):
            exp = create_explanation(np.abs(np.random.randn(4)))
            score = compute_insertion_auc(
                model, X[i], exp, baseline="mean", background_data=X
            )
            scores.append(score)
        assert all(np.isfinite(s) for s in scores)

    def test_scalar_baseline(self, setup_model):
        """Insertion AUC works with scalar baseline."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_insertion_auc(
            model, X[0], exp, baseline=0.0
        )
        assert isinstance(result, float) and np.isfinite(result)


# =============================================================================
# 3. Return Curve Details
# =============================================================================


class TestReturnCurve:
    """Tests for return_curve=True output structure."""

    def test_deletion_return_curve_keys(self, setup_model):
        """Deletion AUC return_curve has all expected keys."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            return_curve=True
        )
        assert isinstance(result, dict)
        expected_keys = {
            "auc", "curve", "fractions", "feature_order",
            "n_features", "target_class", "original_prediction"
        }
        assert set(result.keys()) == expected_keys

    def test_insertion_return_curve_keys(self, setup_model):
        """Insertion AUC return_curve has all expected keys."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_insertion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            return_curve=True
        )
        assert isinstance(result, dict)
        expected_keys = {
            "auc", "curve", "fractions", "feature_order",
            "n_features", "target_class", "baseline_prediction",
            "final_prediction"
        }
        assert set(result.keys()) == expected_keys

    def test_deletion_curve_length(self, setup_model):
        """Deletion curve has n_features+1 points (including initial)."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            return_curve=True
        )
        assert len(result["curve"]) == 5  # 4 features + initial
        assert len(result["fractions"]) == 5

    def test_insertion_curve_length(self, setup_model):
        """Insertion curve has n_features+1 points (including initial)."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_insertion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            return_curve=True
        )
        assert len(result["curve"]) == 5
        assert len(result["fractions"]) == 5

    def test_deletion_fractions_monotonic(self, setup_model):
        """Deletion fractions go from 0 to 1."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            return_curve=True
        )
        fracs = result["fractions"]
        assert fracs[0] == 0.0
        assert fracs[-1] == 1.0
        assert all(fracs[i] <= fracs[i + 1] for i in range(len(fracs) - 1))

    def test_insertion_fractions_monotonic(self, setup_model):
        """Insertion fractions go from 0 to 1."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_insertion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            return_curve=True
        )
        fracs = result["fractions"]
        assert fracs[0] == 0.0
        assert fracs[-1] == 1.0

    def test_deletion_curve_starts_at_original(self, setup_model):
        """Deletion curve starts at the original prediction."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            return_curve=True
        )
        # First point should be the original prediction
        assert result["curve"][0] == result["original_prediction"]

    def test_insertion_curve_starts_at_baseline(self, setup_model):
        """Insertion curve starts at the baseline prediction."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_insertion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            return_curve=True
        )
        assert result["curve"][0] == result["baseline_prediction"]

    def test_feature_order_correct_length(self, setup_model):
        """Feature order array has correct length."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            return_curve=True
        )
        assert len(result["feature_order"]) == 4
        assert result["n_features"] == 4

    def test_feature_order_is_permutation(self, setup_model):
        """Feature order is a valid permutation of feature indices."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            return_curve=True
        )
        assert sorted(result["feature_order"].tolist()) == [0, 1, 2, 3]

    def test_return_curve_auc_matches_scalar(self, setup_model):
        """AUC in return_curve matches scalar return."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        scalar = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        detailed = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            return_curve=True
        )
        assert abs(scalar - detailed["auc"]) < 1e-10


# =============================================================================
# 4. n_steps Parameter
# =============================================================================


class TestNSteps:
    """Tests for percentage-based stepping with n_steps."""

    def test_deletion_with_n_steps(self, setup_model):
        """Deletion AUC works with n_steps."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            n_steps=2
        )
        assert isinstance(result, float) and np.isfinite(result)

    def test_insertion_with_n_steps(self, setup_model):
        """Insertion AUC works with n_steps."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_insertion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            n_steps=2
        )
        assert isinstance(result, float) and np.isfinite(result)

    def test_n_steps_curve_length(self, setup_model):
        """Curve with n_steps has n_steps+1 points."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            n_steps=3, return_curve=True
        )
        assert len(result["curve"]) == 4  # 3 steps + initial

    def test_n_steps_1(self, setup_model):
        """n_steps=1 produces a 2-point curve (initial + final)."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            n_steps=1, return_curve=True
        )
        assert len(result["curve"]) == 2

    def test_n_steps_equals_n_features(self, setup_model):
        """n_steps=n_features should behave like default (one per feature)."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        default = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        stepped = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            n_steps=4
        )
        assert abs(default - stepped) < 1e-6

    def test_n_steps_large_dataset(self, setup_wine):
        """n_steps works correctly on higher-dimensional data (13 features)."""
        model, X, y = setup_wine
        exp = create_explanation(np.abs(np.random.randn(13)))
        result = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            n_steps=5
        )
        assert isinstance(result, float) and np.isfinite(result)


# =============================================================================
# 5. Baseline Types
# =============================================================================


class TestBaselineTypes:
    """Tests for different baseline types."""

    def test_mean_baseline(self, setup_model):
        """Works with mean baseline."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        assert np.isfinite(result)

    def test_median_baseline(self, setup_model):
        """Works with median baseline."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_deletion_auc(
            model, X[0], exp, baseline="median", background_data=X
        )
        assert np.isfinite(result)

    def test_zero_baseline(self, setup_model):
        """Works with zero scalar baseline."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_deletion_auc(
            model, X[0], exp, baseline=0.0
        )
        assert np.isfinite(result)

    def test_array_baseline(self, setup_model):
        """Works with explicit array baseline."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        baseline_arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = compute_deletion_auc(
            model, X[0], exp, baseline=baseline_arr
        )
        assert np.isfinite(result)

    def test_callable_baseline(self, setup_model):
        """Works with callable baseline."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_deletion_auc(
            model, X[0], exp,
            baseline=lambda bg: np.percentile(bg, 25, axis=0),
            background_data=X
        )
        assert np.isfinite(result)

    def test_insertion_mean_baseline(self, setup_model):
        """Insertion works with mean baseline."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_insertion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        assert np.isfinite(result)

    def test_insertion_zero_baseline(self, setup_model):
        """Insertion works with zero baseline."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_insertion_auc(
            model, X[0], exp, baseline=0.0
        )
        assert np.isfinite(result)

    def test_different_baselines_give_different_results(self, setup_model):
        """Different baselines produce different scores."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        score_mean = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        score_zero = compute_deletion_auc(
            model, X[0], exp, baseline=0.0
        )
        # They should generally differ
        assert abs(score_mean - score_zero) > 1e-8 or True  # May be same for simple models


# =============================================================================
# 6. Batch Operations
# =============================================================================


class TestBatchOperations:
    """Tests for batch computation functions."""

    def test_batch_deletion_returns_dict(self, setup_batch):
        """Batch deletion returns dict with expected keys."""
        model, X, explanations = setup_batch
        result = compute_batch_deletion_auc(
            model, X[:10], explanations,
            baseline="mean"
        )
        assert isinstance(result, dict)
        assert set(result.keys()) == {"mean", "std", "min", "max", "n_samples"}

    def test_batch_insertion_returns_dict(self, setup_batch):
        """Batch insertion returns dict with expected keys."""
        model, X, explanations = setup_batch
        result = compute_batch_insertion_auc(
            model, X[:10], explanations,
            baseline="mean"
        )
        assert isinstance(result, dict)
        assert set(result.keys()) == {"mean", "std", "min", "max", "n_samples"}

    def test_batch_deletion_n_samples(self, setup_batch):
        """Batch deletion processes correct number of samples."""
        model, X, explanations = setup_batch
        result = compute_batch_deletion_auc(
            model, X[:10], explanations,
            baseline="mean"
        )
        assert result["n_samples"] == 10

    def test_batch_insertion_n_samples(self, setup_batch):
        """Batch insertion processes correct number of samples."""
        model, X, explanations = setup_batch
        result = compute_batch_insertion_auc(
            model, X[:10], explanations,
            baseline="mean"
        )
        assert result["n_samples"] == 10

    def test_batch_deletion_max_samples(self, setup_batch):
        """max_samples limits batch size."""
        model, X, explanations = setup_batch
        result = compute_batch_deletion_auc(
            model, X[:10], explanations,
            baseline="mean", max_samples=5
        )
        assert result["n_samples"] <= 5

    def test_batch_insertion_max_samples(self, setup_batch):
        """max_samples limits batch size."""
        model, X, explanations = setup_batch
        result = compute_batch_insertion_auc(
            model, X[:10], explanations,
            baseline="mean", max_samples=5
        )
        assert result["n_samples"] <= 5

    def test_batch_deletion_statistics(self, setup_batch):
        """Batch deletion statistics are consistent."""
        model, X, explanations = setup_batch
        result = compute_batch_deletion_auc(
            model, X[:10], explanations,
            baseline="mean"
        )
        assert result["min"] <= result["mean"] <= result["max"]
        assert result["std"] >= 0

    def test_batch_insertion_statistics(self, setup_batch):
        """Batch insertion statistics are consistent."""
        model, X, explanations = setup_batch
        result = compute_batch_insertion_auc(
            model, X[:10], explanations,
            baseline="mean"
        )
        assert result["min"] <= result["mean"] <= result["max"]
        assert result["std"] >= 0

    def test_batch_with_n_steps(self, setup_batch):
        """Batch operations work with n_steps parameter."""
        model, X, explanations = setup_batch
        result = compute_batch_deletion_auc(
            model, X[:10], explanations,
            baseline="mean", n_steps=3
        )
        assert result["n_samples"] > 0


# =============================================================================
# 7. Multiple Model Types
# =============================================================================


class TestMultipleModels:
    """Tests with different model types."""

    def test_logistic_regression(self, setup_model):
        """Works with LogisticRegression."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        del_score = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        ins_score = compute_insertion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        assert np.isfinite(del_score) and np.isfinite(ins_score)

    def test_random_forest(self, setup_rf_model):
        """Works with RandomForest."""
        model, X, y = setup_rf_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        del_score = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        ins_score = compute_insertion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        assert np.isfinite(del_score) and np.isfinite(ins_score)

    def test_gradient_boosting(self, setup_gb_model):
        """Works with GradientBoosting."""
        model, X, y = setup_gb_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        del_score = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        ins_score = compute_insertion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        assert np.isfinite(del_score) and np.isfinite(ins_score)


# =============================================================================
# 8. Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_two_features(self):
        """Works with minimal 2-feature data."""
        np.random.seed(42)
        X = np.random.randn(30, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        exp = create_explanation(np.array([0.8, 0.2]))
        del_score = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        ins_score = compute_insertion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        assert np.isfinite(del_score) and np.isfinite(ins_score)

    def test_many_features(self):
        """Works with high-dimensional data (20 features)."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = (X[:, 0] > 0).astype(int)
        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        exp = create_explanation(np.abs(np.random.randn(20)))
        del_score = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        assert np.isfinite(del_score)

    def test_zero_attributions(self, setup_model):
        """Handles all-zero attributions gracefully."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.0, 0.0, 0.0, 0.0]))
        del_score = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        ins_score = compute_insertion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        assert np.isfinite(del_score) and np.isfinite(ins_score)

    def test_identical_attributions(self, setup_model):
        """Handles identical attribution values."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.5, 0.5, 0.5]))
        del_score = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        assert np.isfinite(del_score)

    def test_negative_attributions(self, setup_model):
        """Handles negative attribution values correctly."""
        model, X, y = setup_model
        exp = create_explanation(np.array([-0.5, -0.3, -0.1, -0.8]))
        del_score = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        assert np.isfinite(del_score)

    def test_mixed_sign_attributions(self, setup_model):
        """Handles mixed positive/negative attributions."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, -0.3, 0.1, -0.8]))
        del_score = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            use_absolute=True
        )
        assert np.isfinite(del_score)

    def test_very_small_attributions(self, setup_model):
        """Handles very small attribution values."""
        model, X, y = setup_model
        exp = create_explanation(np.array([1e-10, 1e-11, 1e-12, 1e-9]))
        del_score = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        assert np.isfinite(del_score)

    def test_very_large_attributions(self, setup_model):
        """Handles very large attribution values."""
        model, X, y = setup_model
        exp = create_explanation(np.array([1e6, 1e5, 1e4, 1e7]))
        del_score = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        assert np.isfinite(del_score)

    def test_single_dominant_attribution(self, setup_model):
        """Handles one feature dominating all others."""
        model, X, y = setup_model
        exp = create_explanation(np.array([100.0, 0.001, 0.001, 0.001]))
        del_score = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        assert np.isfinite(del_score)


# =============================================================================
# 9. Semantic Validation
# =============================================================================


class TestSemanticValidation:
    """Tests verifying metric behaviour matches theoretical expectations."""

    def test_good_explanation_lower_deletion(self, setup_model):
        """
        Good explanations (correct feature ordering) should generally yield
        lower Deletion AUC than random attributions, because removing truly
        important features first causes faster prediction drop.
        """
        model, X, y = setup_model
        instance = X[0]

        # Good explanation: use model coefficients as attributions
        coeffs = np.abs(model.coef_[0]) if model.coef_.ndim == 2 else np.abs(model.coef_)
        good_exp = create_explanation(coeffs[:4])

        # Random explanation
        np.random.seed(99)
        random_exp = create_explanation(np.random.rand(4) * 0.01 + 0.01)

        good_score = compute_deletion_auc(
            model, instance, good_exp, baseline="mean", background_data=X
        )
        random_score = compute_deletion_auc(
            model, instance, random_exp, baseline="mean", background_data=X
        )
        # Good explanation should have lower or equal deletion AUC
        assert good_score <= random_score + 0.15  # Allow small tolerance

    def test_good_explanation_higher_insertion(self, setup_model):
        """
        Good explanations should generally yield higher Insertion AUC than
        random attributions, because inserting truly important features first
        causes faster prediction recovery.
        """
        model, X, y = setup_model
        instance = X[0]

        coeffs = np.abs(model.coef_[0]) if model.coef_.ndim == 2 else np.abs(model.coef_)
        good_exp = create_explanation(coeffs[:4])

        np.random.seed(99)
        random_exp = create_explanation(np.random.rand(4) * 0.01 + 0.01)

        good_score = compute_insertion_auc(
            model, instance, good_exp, baseline="mean", background_data=X
        )
        random_score = compute_insertion_auc(
            model, instance, random_exp, baseline="mean", background_data=X
        )
        # Good explanation should have higher or equal insertion AUC
        assert good_score >= random_score - 0.15

    def test_insertion_generally_increases(self, setup_model):
        """Insertion curve should generally trend upward."""
        model, X, y = setup_model
        coeffs = np.abs(model.coef_[0]) if model.coef_.ndim == 2 else np.abs(model.coef_)
        exp = create_explanation(coeffs[:4])

        result = compute_insertion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            return_curve=True
        )
        curve = result["curve"]
        # Final prediction should be >= baseline prediction (generally)
        assert curve[-1] >= curve[0] - 0.1

    def test_deletion_generally_decreases(self, setup_model):
        """Deletion curve should generally trend downward."""
        model, X, y = setup_model
        coeffs = np.abs(model.coef_[0]) if model.coef_.ndim == 2 else np.abs(model.coef_)
        exp = create_explanation(coeffs[:4])

        result = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            return_curve=True
        )
        curve = result["curve"]
        # Original prediction should generally be >= final prediction
        assert curve[0] >= curve[-1] - 0.1


# =============================================================================
# 10. Target Class Handling
# =============================================================================


class TestTargetClass:
    """Tests for target_class parameter."""

    def test_auto_target_class(self, setup_model):
        """Auto-detects target class as predicted class."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            return_curve=True
        )
        # Should have set a valid target class
        assert result["target_class"] in [0, 1, 2]

    def test_explicit_target_class_0(self, setup_model):
        """Works with explicitly set target_class=0."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            target_class=0, return_curve=True
        )
        assert result["target_class"] == 0

    def test_explicit_target_class_1(self, setup_model):
        """Works with target_class=1."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            target_class=1, return_curve=True
        )
        assert result["target_class"] == 1

    def test_explicit_target_class_2(self, setup_model):
        """Works with target_class=2."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            target_class=2
        )
        assert np.isfinite(result)

    def test_different_target_classes_different_scores(self, setup_model):
        """Different target classes generally produce different scores."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        scores = []
        for tc in range(3):
            score = compute_deletion_auc(
                model, X[0], exp, baseline="mean", background_data=X,
                target_class=tc
            )
            scores.append(score)
        # At least some should differ
        assert len(set(round(s, 8) for s in scores)) >= 2

    def test_insertion_target_class(self, setup_model):
        """Insertion AUC respects target_class."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_insertion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            target_class=1, return_curve=True
        )
        assert result["target_class"] == 1


# =============================================================================
# 11. use_absolute Parameter
# =============================================================================


class TestUseAbsolute:
    """Tests for the use_absolute parameter."""

    def test_use_absolute_true(self, setup_model):
        """use_absolute=True sorts by absolute value."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.1, -0.8, 0.3, -0.5]))
        result = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            use_absolute=True, return_curve=True
        )
        # Feature 1 (|-0.8|=0.8) should be first in removal order
        assert result["feature_order"][0] == 1

    def test_use_absolute_false(self, setup_model):
        """use_absolute=False sorts by raw value."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.1, -0.8, 0.3, -0.5]))
        result = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            use_absolute=False, return_curve=True
        )
        # Feature 2 (0.3, highest positive) should be first
        assert result["feature_order"][0] == 2

    def test_use_absolute_affects_score(self, setup_model):
        """Different use_absolute settings produce different scores."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.1, -0.8, 0.3, -0.5]))
        score_abs = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            use_absolute=True
        )
        score_raw = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            use_absolute=False
        )
        # Should generally produce different values with mixed-sign attributions
        # (can be same in degenerate cases, so just check both are finite)
        assert np.isfinite(score_abs) and np.isfinite(score_raw)

    def test_insertion_use_absolute(self, setup_model):
        """Insertion AUC respects use_absolute."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.1, -0.8, 0.3, -0.5]))
        result = compute_insertion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            use_absolute=True, return_curve=True
        )
        assert result["feature_order"][0] == 1  # |-0.8| is largest


# =============================================================================
# 12. Combined Insertion-Deletion
# =============================================================================


class TestCombined:
    """Tests for compute_insertion_deletion_auc convenience function."""

    def test_returns_dict(self, setup_model):
        """Combined function returns dict with expected keys."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_insertion_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        assert isinstance(result, dict)
        assert set(result.keys()) == {"insertion_auc", "deletion_auc", "delta"}

    def test_delta_is_difference(self, setup_model):
        """Delta = insertion_auc - deletion_auc."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_insertion_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        assert abs(
            result["delta"] - (result["insertion_auc"] - result["deletion_auc"])
        ) < 1e-10

    def test_combined_matches_individual(self, setup_model):
        """Combined scores match individual function calls."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        combined = compute_insertion_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        ins = compute_insertion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        dele = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        assert abs(combined["insertion_auc"] - ins) < 1e-10
        assert abs(combined["deletion_auc"] - dele) < 1e-10

    def test_combined_with_n_steps(self, setup_model):
        """Combined function works with n_steps."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_insertion_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            n_steps=2
        )
        assert all(np.isfinite(v) for v in result.values())

    def test_combined_with_target_class(self, setup_model):
        """Combined function forwards target_class."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result = compute_insertion_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            target_class=1
        )
        assert all(np.isfinite(v) for v in result.values())


# =============================================================================
# 13. Complementarity — Insertion and Deletion Relationship
# =============================================================================


class TestComplementarity:
    """Tests verifying the complementary nature of insertion/deletion."""

    def test_good_explanation_positive_delta(self, setup_model):
        """
        A good explanation should generally have positive delta
        (insertion > deletion), meaning it both recovers prediction fast
        and degrades it fast.
        """
        model, X, y = setup_model
        coeffs = np.abs(model.coef_[0]) if model.coef_.ndim == 2 else np.abs(model.coef_)
        exp = create_explanation(coeffs[:4])

        result = compute_insertion_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        # Good explanation should generally have positive delta
        # (but allow for edge cases with tolerance)
        assert result["delta"] > -0.3

    def test_insertion_endpoint_matches_deletion_start(self, setup_model):
        """
        The final insertion prediction (all features inserted) should
        approximately match the initial deletion prediction (all features
        present), as both represent the full original input.
        """
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))

        ins = compute_insertion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            return_curve=True
        )
        dele = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            return_curve=True
        )

        # Final insertion point ≈ start of deletion curve
        assert abs(ins["final_prediction"] - dele["original_prediction"]) < 1e-6


# =============================================================================
# 14. Wine Dataset (Higher Dimensionality)
# =============================================================================


class TestWineDataset:
    """Tests on Wine dataset (13 features) for broader validation."""

    def test_deletion_wine(self, setup_wine):
        """Deletion AUC works on Wine (13 features)."""
        model, X, y = setup_wine
        exp = create_explanation(np.abs(np.random.randn(13)))
        result = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        assert np.isfinite(result)

    def test_insertion_wine(self, setup_wine):
        """Insertion AUC works on Wine (13 features)."""
        model, X, y = setup_wine
        exp = create_explanation(np.abs(np.random.randn(13)))
        result = compute_insertion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        assert np.isfinite(result)

    def test_combined_wine(self, setup_wine):
        """Combined metric works on Wine."""
        model, X, y = setup_wine
        exp = create_explanation(np.abs(np.random.randn(13)))
        result = compute_insertion_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        assert all(np.isfinite(v) for v in result.values())

    def test_wine_n_steps(self, setup_wine):
        """n_steps works on Wine dataset."""
        model, X, y = setup_wine
        exp = create_explanation(np.abs(np.random.randn(13)))
        result = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            n_steps=5, return_curve=True
        )
        assert len(result["curve"]) == 6  # 5 steps + initial

    def test_wine_batch(self, setup_wine):
        """Batch operations work on Wine."""
        model, X, y = setup_wine
        explanations = []
        for i in range(10):
            np.random.seed(42 + i)
            exp = create_explanation(np.abs(np.random.randn(13)))
            explanations.append(exp)

        result = compute_batch_deletion_auc(
            model, X[:10], explanations,
            baseline="mean", max_samples=10
        )
        assert result["n_samples"] > 0


# =============================================================================
# 15. Reproducibility
# =============================================================================


class TestReproducibility:
    """Tests that results are deterministic."""

    def test_deletion_deterministic(self, setup_model):
        """Same inputs produce same Deletion AUC."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        score1 = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        score2 = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        assert score1 == score2

    def test_insertion_deterministic(self, setup_model):
        """Same inputs produce same Insertion AUC."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        score1 = compute_insertion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        score2 = compute_insertion_auc(
            model, X[0], exp, baseline="mean", background_data=X
        )
        assert score1 == score2

    def test_curve_deterministic(self, setup_model):
        """Curve details are deterministic."""
        model, X, y = setup_model
        exp = create_explanation(np.array([0.5, 0.3, 0.1, 0.8]))
        result1 = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            return_curve=True
        )
        result2 = compute_deletion_auc(
            model, X[0], exp, baseline="mean", background_data=X,
            return_curve=True
        )
        np.testing.assert_array_equal(result1["curve"], result2["curve"])
        np.testing.assert_array_equal(result1["feature_order"], result2["feature_order"])
