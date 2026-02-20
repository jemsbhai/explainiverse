# tests/test_robustness_metrics.py
"""
Tests for Phase 2 robustness evaluation metrics.

- Max-Sensitivity (Yeh et al., 2019)
- Avg-Sensitivity (Yeh et al., 2019)
- Continuity (Montavon et al., 2018; Alvarez-Melis & Jaakkola, 2018)

Reference:
    Yeh, C. K., Hsieh, C. Y., Suggala, A. S., Inber, D. I., & Ravikumar, P.
    (2019). On the (In)fidelity and Sensitivity of Explanations. NeurIPS.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from explainiverse.core.explanation import Explanation


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def feature_names():
    return ["f0", "f1", "f2", "f3"]


@pytest.fixture
def sample_data():
    """Deterministic sample data."""
    np.random.seed(42)
    return np.random.randn(20, 4).astype(np.float32)


@pytest.fixture
def single_instance():
    np.random.seed(42)
    return np.random.randn(4).astype(np.float32)


@pytest.fixture
def trained_model_and_explainer(feature_names):
    """
    Train a real sklearn model and create a LIME explainer.
    Returns (model_adapter, explainer, X_train).
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.datasets import make_classification
    from explainiverse.adapters import SklearnAdapter
    from explainiverse.explainers.attribution.lime_wrapper import LimeExplainer

    X, y = make_classification(
        n_samples=100, n_features=4, n_informative=3,
        n_redundant=0, n_classes=2, random_state=42
    )
    X = X.astype(np.float32)

    clf = GradientBoostingClassifier(n_estimators=20, random_state=42)
    clf.fit(X, y)

    adapter = SklearnAdapter(clf, feature_names=feature_names, class_names=["class_0", "class_1"])
    explainer = LimeExplainer(
        model=adapter,
        training_data=X,
        feature_names=feature_names,
        class_names=["class_0", "class_1"],
    )
    return adapter, explainer, X


class _DeterministicExplainer:
    """
    Mock explainer that returns attributions proportional to the input.
    Perfectly continuous and smooth — useful for testing metric properties.
    """
    def __init__(self, feature_names, scale=1.0):
        self.feature_names = feature_names
        self.scale = scale

    def explain(self, instance):
        instance = np.asarray(instance).flatten()
        attrs = {fn: float(self.scale * instance[i])
                 for i, fn in enumerate(self.feature_names)}
        exp = Explanation(
            explainer_name="deterministic",
            target_class="class_0",
            explanation_data={"feature_attributions": attrs},
        )
        exp.feature_names = self.feature_names
        return exp


class _RandomExplainer:
    """
    Mock explainer that returns random attributions regardless of input.
    High sensitivity, low continuity — useful for testing metric properties.
    """
    def __init__(self, feature_names, seed=None):
        self.feature_names = feature_names
        self.rng = np.random.default_rng(seed)

    def explain(self, instance):
        attrs = {fn: float(self.rng.standard_normal())
                 for fn in self.feature_names}
        exp = Explanation(
            explainer_name="random",
            target_class="class_0",
            explanation_data={"feature_attributions": attrs},
        )
        exp.feature_names = self.feature_names
        return exp


class _ConstantExplainer:
    """
    Mock explainer that always returns the same attributions.
    Zero sensitivity, perfect continuity.
    """
    def __init__(self, feature_names, values=None):
        self.feature_names = feature_names
        self.values = values or [1.0] * len(feature_names)

    def explain(self, instance):
        attrs = {fn: float(self.values[i])
                 for i, fn in enumerate(self.feature_names)}
        exp = Explanation(
            explainer_name="constant",
            target_class="class_0",
            explanation_data={"feature_attributions": attrs},
        )
        exp.feature_names = self.feature_names
        return exp


# =============================================================================
# Max-Sensitivity Tests
# =============================================================================

class TestMaxSensitivity:
    """Tests for compute_max_sensitivity (Yeh et al., 2019)."""

    def test_returns_float(self, feature_names, single_instance):
        """Max-Sensitivity returns a float."""
        from explainiverse.evaluation import compute_max_sensitivity

        explainer = _DeterministicExplainer(feature_names)
        score = compute_max_sensitivity(
            explainer, single_instance, radius=0.1, n_samples=10, seed=42
        )
        assert isinstance(score, float)

    def test_non_negative(self, feature_names, single_instance):
        """Max-Sensitivity is always non-negative."""
        from explainiverse.evaluation import compute_max_sensitivity

        explainer = _DeterministicExplainer(feature_names)
        score = compute_max_sensitivity(
            explainer, single_instance, radius=0.1, n_samples=20, seed=42
        )
        assert score >= 0.0

    def test_constant_explainer_zero_sensitivity(self, feature_names, single_instance):
        """A constant explainer has zero Max-Sensitivity."""
        from explainiverse.evaluation import compute_max_sensitivity

        explainer = _ConstantExplainer(feature_names)
        score = compute_max_sensitivity(
            explainer, single_instance, radius=0.1, n_samples=30, seed=42
        )
        assert score == pytest.approx(0.0, abs=1e-10)

    def test_random_explainer_high_sensitivity(self, feature_names, single_instance):
        """A random explainer has higher sensitivity than a deterministic one."""
        from explainiverse.evaluation import compute_max_sensitivity

        det_explainer = _DeterministicExplainer(feature_names)
        rng_explainer = _RandomExplainer(feature_names, seed=99)

        det_score = compute_max_sensitivity(
            det_explainer, single_instance, radius=0.1, n_samples=30, seed=42
        )
        rng_score = compute_max_sensitivity(
            rng_explainer, single_instance, radius=0.1, n_samples=30, seed=42
        )
        assert rng_score > det_score

    def test_larger_radius_higher_sensitivity(self, feature_names, single_instance):
        """Larger perturbation radius generally yields higher sensitivity."""
        from explainiverse.evaluation import compute_max_sensitivity

        explainer = _DeterministicExplainer(feature_names)
        score_small = compute_max_sensitivity(
            explainer, single_instance, radius=0.01, n_samples=30, seed=42
        )
        score_large = compute_max_sensitivity(
            explainer, single_instance, radius=1.0, n_samples=30, seed=42
        )
        assert score_large >= score_small

    def test_reproducible_with_seed(self, feature_names, single_instance):
        """Same seed produces same result."""
        from explainiverse.evaluation import compute_max_sensitivity

        explainer = _DeterministicExplainer(feature_names)
        s1 = compute_max_sensitivity(
            explainer, single_instance, radius=0.1, n_samples=20, seed=42
        )
        s2 = compute_max_sensitivity(
            explainer, single_instance, radius=0.1, n_samples=20, seed=42
        )
        assert s1 == pytest.approx(s2, rel=1e-10)

    def test_l2_perturbation_norm(self, feature_names, single_instance):
        """L2 perturbation norm works."""
        from explainiverse.evaluation import compute_max_sensitivity

        explainer = _DeterministicExplainer(feature_names)
        score = compute_max_sensitivity(
            explainer, single_instance, radius=0.1, n_samples=10,
            perturb_norm="l2", seed=42
        )
        assert np.isfinite(score)

    def test_linf_perturbation_norm(self, feature_names, single_instance):
        """L-inf perturbation norm works."""
        from explainiverse.evaluation import compute_max_sensitivity

        explainer = _DeterministicExplainer(feature_names)
        score = compute_max_sensitivity(
            explainer, single_instance, radius=0.1, n_samples=10,
            perturb_norm="linf", seed=42
        )
        assert np.isfinite(score)

    def test_invalid_perturb_norm_raises(self, feature_names, single_instance):
        """Invalid perturbation norm raises ValueError."""
        from explainiverse.evaluation import compute_max_sensitivity

        explainer = _DeterministicExplainer(feature_names)
        with pytest.raises(ValueError, match="perturb_norm"):
            compute_max_sensitivity(
                explainer, single_instance, perturb_norm="l3", seed=42
            )

    def test_l1_norm_order(self, feature_names, single_instance):
        """L1 norm order for explanation distances works."""
        from explainiverse.evaluation import compute_max_sensitivity

        explainer = _DeterministicExplainer(feature_names)
        score = compute_max_sensitivity(
            explainer, single_instance, norm_ord=1, n_samples=10, seed=42
        )
        assert np.isfinite(score)

    def test_linf_norm_order(self, feature_names, single_instance):
        """L-inf norm order for explanation distances works."""
        from explainiverse.evaluation import compute_max_sensitivity

        explainer = _DeterministicExplainer(feature_names)
        score = compute_max_sensitivity(
            explainer, single_instance, norm_ord=np.inf, n_samples=10, seed=42
        )
        assert np.isfinite(score)

    def test_normalize_false(self, feature_names, single_instance):
        """Without normalization, returns absolute difference."""
        from explainiverse.evaluation import compute_max_sensitivity

        explainer = _DeterministicExplainer(feature_names)
        score_norm = compute_max_sensitivity(
            explainer, single_instance, normalize=True, n_samples=20, seed=42
        )
        score_abs = compute_max_sensitivity(
            explainer, single_instance, normalize=False, n_samples=20, seed=42
        )
        # Both should be finite, but different
        assert np.isfinite(score_norm)
        assert np.isfinite(score_abs)

    def test_zero_explanation_normalized_returns_zero(self, feature_names, single_instance):
        """Zero explanations with normalize=True returns 0."""
        from explainiverse.evaluation import compute_max_sensitivity

        explainer = _ConstantExplainer(feature_names, values=[0.0, 0.0, 0.0, 0.0])
        score = compute_max_sensitivity(
            explainer, single_instance, normalize=True, n_samples=10, seed=42
        )
        assert score == 0.0

    def test_more_samples_tighter_bound(self, feature_names, single_instance):
        """More samples should produce a max >= the fewer-samples max."""
        from explainiverse.evaluation import compute_max_sensitivity

        explainer = _DeterministicExplainer(feature_names)
        score_few = compute_max_sensitivity(
            explainer, single_instance, n_samples=5, seed=42
        )
        score_many = compute_max_sensitivity(
            explainer, single_instance, n_samples=100, seed=42
        )
        # More samples includes the first 5 (same seed), so max is >=
        assert score_many >= score_few - 1e-10

    def test_with_real_explainer(self, trained_model_and_explainer):
        """Max-Sensitivity works with a real LIME explainer."""
        from explainiverse.evaluation import compute_max_sensitivity

        _, explainer, X = trained_model_and_explainer
        score = compute_max_sensitivity(
            explainer, X[0], radius=0.1, n_samples=5, seed=42
        )
        assert isinstance(score, float)
        assert np.isfinite(score)
        assert score >= 0.0


class TestBatchMaxSensitivity:
    """Tests for compute_batch_max_sensitivity."""

    def test_returns_dict(self, feature_names, sample_data):
        """Batch Max-Sensitivity returns dict with expected keys."""
        from explainiverse.evaluation import compute_batch_max_sensitivity

        explainer = _DeterministicExplainer(feature_names)
        result = compute_batch_max_sensitivity(
            explainer, sample_data, n_samples=5, max_instances=5, seed=42
        )
        assert isinstance(result, dict)
        assert "mean" in result
        assert "std" in result
        assert "max" in result
        assert "min" in result
        assert "scores" in result
        assert "n_evaluated" in result

    def test_max_instances_limits(self, feature_names, sample_data):
        """max_instances limits the number of evaluations."""
        from explainiverse.evaluation import compute_batch_max_sensitivity

        explainer = _DeterministicExplainer(feature_names)
        result = compute_batch_max_sensitivity(
            explainer, sample_data, n_samples=5, max_instances=3, seed=42
        )
        assert result["n_evaluated"] == 3

    def test_mean_between_min_max(self, feature_names, sample_data):
        """Mean is between min and max."""
        from explainiverse.evaluation import compute_batch_max_sensitivity

        explainer = _DeterministicExplainer(feature_names)
        result = compute_batch_max_sensitivity(
            explainer, sample_data, n_samples=5, max_instances=5, seed=42
        )
        assert result["min"] <= result["mean"] <= result["max"]

    def test_with_real_explainer(self, trained_model_and_explainer):
        """Batch works with a real LIME explainer."""
        from explainiverse.evaluation import compute_batch_max_sensitivity

        _, explainer, X = trained_model_and_explainer
        result = compute_batch_max_sensitivity(
            explainer, X, n_samples=3, max_instances=3, seed=42
        )
        assert result["n_evaluated"] == 3
        assert np.isfinite(result["mean"])


# =============================================================================
# Avg-Sensitivity Tests
# =============================================================================

class TestAvgSensitivity:
    """Tests for compute_avg_sensitivity (Yeh et al., 2019)."""

    def test_returns_float(self, feature_names, single_instance):
        """Avg-Sensitivity returns a float."""
        from explainiverse.evaluation import compute_avg_sensitivity

        explainer = _DeterministicExplainer(feature_names)
        score = compute_avg_sensitivity(
            explainer, single_instance, radius=0.1, n_samples=10, seed=42
        )
        assert isinstance(score, float)

    def test_non_negative(self, feature_names, single_instance):
        """Avg-Sensitivity is always non-negative."""
        from explainiverse.evaluation import compute_avg_sensitivity

        explainer = _DeterministicExplainer(feature_names)
        score = compute_avg_sensitivity(
            explainer, single_instance, radius=0.1, n_samples=20, seed=42
        )
        assert score >= 0.0

    def test_constant_explainer_zero(self, feature_names, single_instance):
        """A constant explainer has zero Avg-Sensitivity."""
        from explainiverse.evaluation import compute_avg_sensitivity

        explainer = _ConstantExplainer(feature_names)
        score = compute_avg_sensitivity(
            explainer, single_instance, radius=0.1, n_samples=30, seed=42
        )
        assert score == pytest.approx(0.0, abs=1e-10)

    def test_avg_leq_max(self, feature_names, single_instance):
        """Avg-Sensitivity ≤ Max-Sensitivity (mean ≤ max)."""
        from explainiverse.evaluation import (
            compute_max_sensitivity, compute_avg_sensitivity
        )

        explainer = _DeterministicExplainer(feature_names)
        avg = compute_avg_sensitivity(
            explainer, single_instance, radius=0.1, n_samples=30, seed=42
        )
        mx = compute_max_sensitivity(
            explainer, single_instance, radius=0.1, n_samples=30, seed=42
        )
        assert avg <= mx + 1e-10

    def test_random_explainer_higher_than_deterministic(self, feature_names, single_instance):
        """Random explainer has higher Avg-Sensitivity than deterministic."""
        from explainiverse.evaluation import compute_avg_sensitivity

        det = _DeterministicExplainer(feature_names)
        rng = _RandomExplainer(feature_names, seed=99)

        det_score = compute_avg_sensitivity(
            det, single_instance, radius=0.1, n_samples=30, seed=42
        )
        rng_score = compute_avg_sensitivity(
            rng, single_instance, radius=0.1, n_samples=30, seed=42
        )
        assert rng_score > det_score

    def test_reproducible_with_seed(self, feature_names, single_instance):
        """Same seed produces same result."""
        from explainiverse.evaluation import compute_avg_sensitivity

        explainer = _DeterministicExplainer(feature_names)
        s1 = compute_avg_sensitivity(
            explainer, single_instance, radius=0.1, n_samples=20, seed=42
        )
        s2 = compute_avg_sensitivity(
            explainer, single_instance, radius=0.1, n_samples=20, seed=42
        )
        assert s1 == pytest.approx(s2, rel=1e-10)

    def test_linf_perturbation(self, feature_names, single_instance):
        """L-inf perturbation works."""
        from explainiverse.evaluation import compute_avg_sensitivity

        explainer = _DeterministicExplainer(feature_names)
        score = compute_avg_sensitivity(
            explainer, single_instance, perturb_norm="linf", n_samples=10, seed=42
        )
        assert np.isfinite(score)

    def test_normalize_false(self, feature_names, single_instance):
        """normalize=False returns absolute scores."""
        from explainiverse.evaluation import compute_avg_sensitivity

        explainer = _DeterministicExplainer(feature_names)
        score = compute_avg_sensitivity(
            explainer, single_instance, normalize=False, n_samples=10, seed=42
        )
        assert np.isfinite(score)
        assert score >= 0.0

    def test_invalid_perturb_norm(self, feature_names, single_instance):
        """Invalid perturbation norm raises ValueError."""
        from explainiverse.evaluation import compute_avg_sensitivity

        explainer = _DeterministicExplainer(feature_names)
        with pytest.raises(ValueError, match="perturb_norm"):
            compute_avg_sensitivity(
                explainer, single_instance, perturb_norm="invalid", seed=42
            )

    def test_with_real_explainer(self, trained_model_and_explainer):
        """Works with real LIME explainer."""
        from explainiverse.evaluation import compute_avg_sensitivity

        _, explainer, X = trained_model_and_explainer
        score = compute_avg_sensitivity(
            explainer, X[0], radius=0.1, n_samples=5, seed=42
        )
        assert isinstance(score, float)
        assert np.isfinite(score)
        assert score >= 0.0


class TestBatchAvgSensitivity:
    """Tests for compute_batch_avg_sensitivity."""

    def test_returns_dict(self, feature_names, sample_data):
        """Returns dict with expected keys."""
        from explainiverse.evaluation import compute_batch_avg_sensitivity

        explainer = _DeterministicExplainer(feature_names)
        result = compute_batch_avg_sensitivity(
            explainer, sample_data, n_samples=5, max_instances=5, seed=42
        )
        assert "mean" in result
        assert "std" in result
        assert "scores" in result
        assert "n_evaluated" in result

    def test_max_instances_limits(self, feature_names, sample_data):
        """max_instances limits evaluations."""
        from explainiverse.evaluation import compute_batch_avg_sensitivity

        explainer = _DeterministicExplainer(feature_names)
        result = compute_batch_avg_sensitivity(
            explainer, sample_data, n_samples=5, max_instances=4, seed=42
        )
        assert result["n_evaluated"] == 4

    def test_with_real_explainer(self, trained_model_and_explainer):
        """Batch works with real LIME explainer."""
        from explainiverse.evaluation import compute_batch_avg_sensitivity

        _, explainer, X = trained_model_and_explainer
        result = compute_batch_avg_sensitivity(
            explainer, X, n_samples=3, max_instances=3, seed=42
        )
        assert result["n_evaluated"] == 3
        assert np.isfinite(result["mean"])


# =============================================================================
# Continuity Tests
# =============================================================================

class TestContinuity:
    """Tests for compute_continuity (Montavon et al., 2018)."""

    def test_returns_float(self, feature_names, single_instance, sample_data):
        """Continuity returns a float."""
        from explainiverse.evaluation import compute_continuity

        explainer = _DeterministicExplainer(feature_names)
        score = compute_continuity(
            explainer, single_instance, sample_data, k_neighbors=5
        )
        assert isinstance(score, float)

    def test_range_minus_one_to_one(self, feature_names, single_instance, sample_data):
        """Continuity is a Spearman correlation in [-1, +1]."""
        from explainiverse.evaluation import compute_continuity

        explainer = _DeterministicExplainer(feature_names)
        score = compute_continuity(
            explainer, single_instance, sample_data, k_neighbors=5
        )
        assert -1.0 <= score <= 1.0

    def test_deterministic_explainer_high_continuity(self, feature_names, sample_data):
        """
        A deterministic explainer (E(x) = scale * x) should have high
        continuity because nearby inputs produce nearby explanations.
        """
        from explainiverse.evaluation import compute_continuity

        explainer = _DeterministicExplainer(feature_names, scale=2.0)
        score = compute_continuity(
            explainer, sample_data[0], sample_data[1:], k_neighbors=10
        )
        # Linear transformation: input distance and explanation distance
        # are perfectly correlated
        assert score > 0.9

    def test_constant_explainer_returns_zero_or_nan(self, feature_names, sample_data):
        """
        A constant explainer has zero explanation distance for all neighbours.
        Spearman correlation is undefined (all tied) → should return 0.0 or NaN.
        """
        from explainiverse.evaluation import compute_continuity

        explainer = _ConstantExplainer(feature_names)
        score = compute_continuity(
            explainer, sample_data[0], sample_data[1:], k_neighbors=5
        )
        # All explanation distances are 0 → tied ranks → correlation = 0
        assert score == pytest.approx(0.0, abs=1e-10) or np.isnan(score)

    def test_fewer_than_3_neighbors_returns_nan(self, feature_names, single_instance):
        """Returns NaN if fewer than 3 neighbours available."""
        from explainiverse.evaluation import compute_continuity

        explainer = _DeterministicExplainer(feature_names)
        tiny_ref = np.random.randn(2, 4).astype(np.float32)
        score = compute_continuity(
            explainer, single_instance, tiny_ref, k_neighbors=5
        )
        assert np.isnan(score)

    def test_k_neighbors_clamped(self, feature_names, single_instance):
        """k_neighbors larger than reference set is clamped."""
        from explainiverse.evaluation import compute_continuity

        explainer = _DeterministicExplainer(feature_names)
        small_ref = np.random.randn(5, 4).astype(np.float32)
        # Ask for 100 neighbours but only 5 are available
        score = compute_continuity(
            explainer, single_instance, small_ref, k_neighbors=100
        )
        assert isinstance(score, float)

    def test_excludes_self_from_neighbors(self, feature_names, sample_data):
        """Instance itself is excluded from neighbours (distance ≈ 0)."""
        from explainiverse.evaluation import compute_continuity

        explainer = _DeterministicExplainer(feature_names)
        # Include the instance in the reference set
        instance = sample_data[0]
        reference_with_self = np.vstack([instance.reshape(1, -1), sample_data])
        score = compute_continuity(
            explainer, instance, reference_with_self, k_neighbors=5
        )
        # Should still work (self excluded)
        assert isinstance(score, float)
        assert not np.isnan(score)

    def test_different_input_distances(self, feature_names, single_instance, sample_data):
        """Different input distance metrics work."""
        from explainiverse.evaluation import compute_continuity

        explainer = _DeterministicExplainer(feature_names)
        for metric in ["euclidean", "cityblock", "cosine"]:
            score = compute_continuity(
                explainer, single_instance, sample_data,
                k_neighbors=5, input_distance=metric
            )
            assert isinstance(score, float)

    def test_different_norm_orders(self, feature_names, single_instance, sample_data):
        """Different norm orders for explanation distances work."""
        from explainiverse.evaluation import compute_continuity

        explainer = _DeterministicExplainer(feature_names)
        for norm in [1, 2, np.inf]:
            score = compute_continuity(
                explainer, single_instance, sample_data,
                k_neighbors=5, norm_ord=norm
            )
            assert isinstance(score, float)

    def test_with_real_explainer(self, trained_model_and_explainer):
        """Works with real LIME explainer."""
        from explainiverse.evaluation import compute_continuity

        _, explainer, X = trained_model_and_explainer
        score = compute_continuity(
            explainer, X[0], X[1:20], k_neighbors=5
        )
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0 or np.isnan(score)


class TestBatchContinuity:
    """Tests for compute_batch_continuity."""

    def test_returns_dict(self, feature_names, sample_data):
        """Returns dict with expected keys."""
        from explainiverse.evaluation import compute_batch_continuity

        explainer = _DeterministicExplainer(feature_names)
        result = compute_batch_continuity(
            explainer, sample_data, k_neighbors=5, max_instances=5
        )
        assert "mean" in result
        assert "std" in result
        assert "scores" in result
        assert "n_evaluated" in result

    def test_max_instances_limits(self, feature_names, sample_data):
        """max_instances limits evaluations."""
        from explainiverse.evaluation import compute_batch_continuity

        explainer = _DeterministicExplainer(feature_names)
        result = compute_batch_continuity(
            explainer, sample_data, k_neighbors=5, max_instances=4
        )
        assert result["n_evaluated"] == 4

    def test_deterministic_high_mean_continuity(self, feature_names, sample_data):
        """Deterministic explainer has high mean continuity across batch."""
        from explainiverse.evaluation import compute_batch_continuity

        explainer = _DeterministicExplainer(feature_names, scale=2.0)
        result = compute_batch_continuity(
            explainer, sample_data, k_neighbors=5, max_instances=10
        )
        assert result["mean"] > 0.8

    def test_with_real_explainer(self, trained_model_and_explainer):
        """Batch works with real LIME explainer."""
        from explainiverse.evaluation import compute_batch_continuity

        _, explainer, X = trained_model_and_explainer
        result = compute_batch_continuity(
            explainer, X[:15], k_neighbors=5, max_instances=3
        )
        assert result["n_evaluated"] == 3


# =============================================================================
# Cross-Metric Relationship Tests
# =============================================================================

class TestCrossMetricRelationships:
    """Tests verifying expected relationships between metrics."""

    def test_avg_leq_max_sensitivity(self, feature_names, sample_data):
        """Avg-Sensitivity ≤ Max-Sensitivity for every instance in a batch."""
        from explainiverse.evaluation import (
            compute_max_sensitivity, compute_avg_sensitivity
        )

        explainer = _DeterministicExplainer(feature_names)
        for i in range(5):
            avg = compute_avg_sensitivity(
                explainer, sample_data[i], radius=0.1, n_samples=20, seed=42
            )
            mx = compute_max_sensitivity(
                explainer, sample_data[i], radius=0.1, n_samples=20, seed=42
            )
            assert avg <= mx + 1e-10, f"Instance {i}: avg={avg} > max={mx}"

    def test_stable_explainer_low_sensitivity_high_continuity(self, feature_names, sample_data):
        """
        A stable (deterministic linear) explainer should have:
        - Low sensitivity scores
        - High continuity scores
        """
        from explainiverse.evaluation import (
            compute_max_sensitivity, compute_continuity
        )

        explainer = _DeterministicExplainer(feature_names)
        sens = compute_max_sensitivity(
            explainer, sample_data[0], radius=0.05, n_samples=20, seed=42
        )
        cont = compute_continuity(
            explainer, sample_data[0], sample_data[1:], k_neighbors=10
        )
        # Deterministic linear: low sensitivity, high continuity
        assert sens < 1.0  # Relative sensitivity should be modest
        assert cont > 0.9  # Near-perfect rank correlation


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests for robustness metrics."""

    def test_single_feature(self):
        """Metrics work with a single feature."""
        from explainiverse.evaluation import (
            compute_max_sensitivity, compute_avg_sensitivity
        )

        fnames = ["f0"]
        explainer = _DeterministicExplainer(fnames)
        instance = np.array([1.0])

        ms = compute_max_sensitivity(explainer, instance, n_samples=10, seed=42)
        assert np.isfinite(ms)

        avs = compute_avg_sensitivity(explainer, instance, n_samples=10, seed=42)
        assert np.isfinite(avs)

    def test_high_dimensional(self):
        """Metrics work with many features."""
        from explainiverse.evaluation import compute_max_sensitivity

        fnames = [f"f{i}" for i in range(100)]
        explainer = _DeterministicExplainer(fnames)
        instance = np.random.randn(100).astype(np.float32)

        score = compute_max_sensitivity(
            explainer, instance, n_samples=10, seed=42
        )
        assert np.isfinite(score)

    def test_zero_radius(self, feature_names, single_instance):
        """Zero radius produces zero sensitivity (no perturbation)."""
        from explainiverse.evaluation import compute_max_sensitivity

        explainer = _DeterministicExplainer(feature_names)
        score = compute_max_sensitivity(
            explainer, single_instance, radius=0.0, n_samples=10, seed=42
        )
        assert score == pytest.approx(0.0, abs=1e-10)

    def test_very_large_radius(self, feature_names, single_instance):
        """Very large radius still produces finite results."""
        from explainiverse.evaluation import compute_max_sensitivity

        explainer = _DeterministicExplainer(feature_names)
        score = compute_max_sensitivity(
            explainer, single_instance, radius=100.0, n_samples=10, seed=42
        )
        assert np.isfinite(score)

    def test_continuity_all_identical_reference(self, feature_names, single_instance):
        """Continuity handles all identical reference points gracefully."""
        from explainiverse.evaluation import compute_continuity

        explainer = _DeterministicExplainer(feature_names)
        # All reference points are the same
        identical_ref = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (10, 1))
        score = compute_continuity(
            explainer, single_instance, identical_ref, k_neighbors=5
        )
        # Should handle gracefully (NaN or a valid float)
        assert isinstance(score, float)


# =============================================================================
# Import Tests
# =============================================================================

class TestImports:
    """Verify metrics are importable from the evaluation package."""

    def test_import_max_sensitivity(self):
        from explainiverse.evaluation import compute_max_sensitivity
        assert callable(compute_max_sensitivity)

    def test_import_batch_max_sensitivity(self):
        from explainiverse.evaluation import compute_batch_max_sensitivity
        assert callable(compute_batch_max_sensitivity)

    def test_import_avg_sensitivity(self):
        from explainiverse.evaluation import compute_avg_sensitivity
        assert callable(compute_avg_sensitivity)

    def test_import_batch_avg_sensitivity(self):
        from explainiverse.evaluation import compute_batch_avg_sensitivity
        assert callable(compute_batch_avg_sensitivity)

    def test_import_continuity(self):
        from explainiverse.evaluation import compute_continuity
        assert callable(compute_continuity)

    def test_import_batch_continuity(self):
        from explainiverse.evaluation import compute_batch_continuity
        assert callable(compute_batch_continuity)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
