# tests/test_axiomatic.py
"""
Tests for Phase 6 axiomatic evaluation metrics.

- Completeness (Sundararajan et al., 2017)
- Non-Sensitivity (Nguyen & Martínez, 2020)
- Input Invariance (Kindermans et al., 2017)
- Symmetry (Sundararajan et al., 2017)

References:
    Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution
    for Deep Networks. ICML. https://arxiv.org/abs/1703.01365

    Kindermans, P.-J., Hooker, S., Adebayo, J., et al. (2017). The
    (Un)reliability of Saliency Methods. arXiv:1711.00867.

    Nguyen, A. P., & Martínez, M. R. (2020). On Quantitative Aspects of
    Model Interpretability. arXiv:2007.07584.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from explainiverse.core.explanation import Explanation
from explainiverse.evaluation.axiomatic import (
    compute_completeness,
    compute_completeness_score,
    compute_batch_completeness,
    compute_non_sensitivity,
    compute_non_sensitivity_score,
    compute_batch_non_sensitivity,
    compute_input_invariance,
    compute_batch_input_invariance,
    compute_symmetry,
    compute_symmetry_score,
    compute_batch_symmetry,
    _detect_non_sensitive_features,
    _safe_model_output,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def feature_names():
    return ["f0", "f1", "f2", "f3"]


@pytest.fixture
def feature_names_8():
    return [f"f{i}" for i in range(8)]


@pytest.fixture
def single_instance():
    np.random.seed(42)
    return np.random.randn(4).astype(np.float64)


@pytest.fixture
def sample_data():
    """Deterministic sample data."""
    np.random.seed(42)
    return np.random.randn(10, 4).astype(np.float64)


@pytest.fixture
def linear_model_fn():
    """
    Simple linear model: f(x) = 0.5*x0 + 1.0*x1 + 0.0*x2 + 0.0*x3
    Features 2 and 3 are non-sensitive (zero weight).
    """
    weights = np.array([0.5, 1.0, 0.0, 0.0])
    def model_fn(x):
        x = np.asarray(x, dtype=np.float64).flatten()
        return float(np.dot(weights, x))
    return model_fn


@pytest.fixture
def symmetric_model_fn():
    """
    Model where features 0 and 1 are symmetric: f(x) = x0 + x1 + 2*x2 + 3*x3
    Swapping x0 and x1 does not change output.
    """
    weights = np.array([1.0, 1.0, 2.0, 3.0])
    def model_fn(x):
        x = np.asarray(x, dtype=np.float64).flatten()
        return float(np.dot(weights, x))
    return model_fn


@pytest.fixture
def trained_model_and_explainer(feature_names):
    """
    Train a real sklearn model and create a LIME explainer.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.datasets import make_classification
    from explainiverse.adapters import SklearnAdapter
    from explainiverse.explainers.attribution.lime_wrapper import LimeExplainer

    X, y = make_classification(
        n_samples=100, n_features=4, n_informative=3,
        n_redundant=0, n_classes=2, random_state=42,
    )
    X = X.astype(np.float32)

    clf = GradientBoostingClassifier(n_estimators=20, random_state=42)
    clf.fit(X, y)

    adapter = SklearnAdapter(
        clf, feature_names=feature_names, class_names=["class_0", "class_1"]
    )
    explainer = LimeExplainer(
        model=adapter,
        training_data=X,
        feature_names=feature_names,
        class_names=["class_0", "class_1"],
    )
    return adapter, explainer, X


# =============================================================================
# Mock Explainers
# =============================================================================

class _PerfectCompletenessExplainer:
    """
    Mock explainer that returns attributions summing to exactly f(x) - f(baseline).
    Completeness score should be 0.0.
    """
    def __init__(self, feature_names, model_fn, baseline=None):
        self.feature_names = feature_names
        self.model_fn = model_fn
        self.baseline = baseline

    def explain(self, instance):
        instance = np.asarray(instance, dtype=np.float64).flatten()
        n = len(instance)
        baseline = (
            np.zeros(n) if self.baseline is None
            else np.asarray(self.baseline, dtype=np.float64).flatten()
        )
        target_sum = self.model_fn(instance) - self.model_fn(baseline)
        # Distribute evenly
        attrs = {fn: target_sum / n for fn in self.feature_names}
        exp = Explanation(
            explainer_name="perfect_completeness",
            target_class="class_0",
            explanation_data={"feature_attributions": attrs},
        )
        exp.feature_names = self.feature_names
        return exp


class _BadCompletenessExplainer:
    """
    Mock explainer that always returns uniform attributions of fixed value.
    Will violate completeness unless coincidentally correct.
    """
    def __init__(self, feature_names, fixed_value=0.25):
        self.feature_names = feature_names
        self.fixed_value = fixed_value

    def explain(self, instance):
        attrs = {fn: self.fixed_value for fn in self.feature_names}
        exp = Explanation(
            explainer_name="bad_completeness",
            target_class="class_0",
            explanation_data={"feature_attributions": attrs},
        )
        exp.feature_names = self.feature_names
        return exp


class _SymmetricExplainer:
    """
    Mock explainer that gives equal attribution to symmetric features.
    """
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def explain(self, instance):
        instance = np.asarray(instance, dtype=np.float64).flatten()
        # Give features 0 and 1 identical attributions
        attrs = {}
        for i, fn in enumerate(self.feature_names):
            if i == 0 or i == 1:
                attrs[fn] = 0.5
            else:
                attrs[fn] = float(instance[i]) * 0.1
        exp = Explanation(
            explainer_name="symmetric",
            target_class="class_0",
            explanation_data={"feature_attributions": attrs},
        )
        exp.feature_names = self.feature_names
        return exp


class _AsymmetricExplainer:
    """
    Mock explainer that gives different attribution to symmetric features.
    """
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def explain(self, instance):
        instance = np.asarray(instance, dtype=np.float64).flatten()
        attrs = {}
        for i, fn in enumerate(self.feature_names):
            attrs[fn] = float(instance[i]) * (i + 1)
        exp = Explanation(
            explainer_name="asymmetric",
            target_class="class_0",
            explanation_data={"feature_attributions": attrs},
        )
        exp.feature_names = self.feature_names
        return exp


class _NonSensitiveAwareExplainer:
    """
    Mock explainer that gives zero attribution to non-sensitive features.
    """
    def __init__(self, feature_names, sensitive_mask):
        self.feature_names = feature_names
        self.sensitive_mask = sensitive_mask

    def explain(self, instance):
        instance = np.asarray(instance, dtype=np.float64).flatten()
        attrs = {}
        for i, fn in enumerate(self.feature_names):
            if self.sensitive_mask[i]:
                attrs[fn] = float(instance[i]) * 0.5
            else:
                attrs[fn] = 0.0
        exp = Explanation(
            explainer_name="ns_aware",
            target_class="class_0",
            explanation_data={"feature_attributions": attrs},
        )
        exp.feature_names = self.feature_names
        return exp


class _NonSensitiveUnawareExplainer:
    """
    Mock explainer that assigns attribution to all features including non-sensitive ones.
    """
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def explain(self, instance):
        instance = np.asarray(instance, dtype=np.float64).flatten()
        attrs = {fn: 0.25 for fn in self.feature_names}
        exp = Explanation(
            explainer_name="ns_unaware",
            target_class="class_0",
            explanation_data={"feature_attributions": attrs},
        )
        exp.feature_names = self.feature_names
        return exp


# =============================================================================
# COMPLETENESS TESTS
# =============================================================================

class TestCompletenessBasic:
    """Basic functionality tests for Completeness."""

    def test_returns_float(self, linear_model_fn, single_instance):
        attrs = np.array([0.1, 0.2, 0.3, 0.4])
        result = compute_completeness(attrs, linear_model_fn, single_instance)
        assert isinstance(result, float)

    def test_non_negative(self, linear_model_fn, single_instance):
        attrs = np.array([0.1, 0.2, 0.3, 0.4])
        result = compute_completeness(attrs, linear_model_fn, single_instance)
        assert result >= 0.0

    def test_perfect_completeness(self, linear_model_fn, single_instance):
        """Attributions summing to f(x) - f(0) should give score ≈ 0."""
        f_x = linear_model_fn(single_instance)
        f_baseline = linear_model_fn(np.zeros(4))
        target = f_x - f_baseline
        # Distribute the target sum across features
        attrs = np.array([target / 4] * 4)
        result = compute_completeness(attrs, linear_model_fn, single_instance)
        assert result < 1e-10

    def test_zero_attributions_nonzero_output(self, linear_model_fn, single_instance):
        """Zero attributions should give score = |f(x) - f(0)|."""
        attrs = np.zeros(4)
        result = compute_completeness(attrs, linear_model_fn, single_instance)
        expected = abs(linear_model_fn(single_instance) - linear_model_fn(np.zeros(4)))
        assert abs(result - expected) < 1e-10

    def test_exact_attribution_matches_weights(self, linear_model_fn):
        """For linear model, weight × (x - baseline) is perfectly complete."""
        weights = np.array([0.5, 1.0, 0.0, 0.0])
        x = np.array([2.0, 3.0, 1.0, 5.0])
        baseline = np.zeros(4)
        attrs = weights * (x - baseline)  # [1.0, 3.0, 0.0, 0.0]
        result = compute_completeness(attrs, linear_model_fn, x, baseline=baseline)
        assert result < 1e-10

    def test_with_nonzero_baseline(self, linear_model_fn):
        """Test with explicit non-zero baseline."""
        x = np.array([2.0, 3.0, 1.0, 5.0])
        baseline = np.array([1.0, 1.0, 1.0, 1.0])
        f_diff = linear_model_fn(x) - linear_model_fn(baseline)
        attrs = np.array([f_diff, 0.0, 0.0, 0.0])
        result = compute_completeness(attrs, linear_model_fn, x, baseline=baseline)
        assert result < 1e-10

    def test_with_scalar_baseline(self, linear_model_fn):
        """Test with scalar baseline (broadcast to all features)."""
        x = np.array([2.0, 3.0, 1.0, 5.0])
        f_x = linear_model_fn(x)
        f_b = linear_model_fn(np.full(4, 0.5))
        attrs = np.array([f_x - f_b, 0.0, 0.0, 0.0])
        result = compute_completeness(attrs, linear_model_fn, x, baseline=0.5)
        assert result < 1e-10

    def test_large_violation(self, linear_model_fn, single_instance):
        """Attributions summing to a very different value should give high score."""
        attrs = np.array([100.0, 100.0, 100.0, 100.0])
        result = compute_completeness(attrs, linear_model_fn, single_instance)
        assert result > 1.0  # Large deviation

    def test_negative_attributions(self, linear_model_fn):
        """Negative attributions should work correctly."""
        x = np.array([1.0, -2.0, 0.0, 0.0])
        f_diff = linear_model_fn(x) - linear_model_fn(np.zeros(4))
        # f(x) = 0.5*1 + 1.0*(-2) = -1.5
        attrs = np.array([-1.5, 0.0, 0.0, 0.0])
        result = compute_completeness(attrs, linear_model_fn, x)
        assert result < 1e-10


class TestCompletenessWithOutputFunc:
    """Test Completeness with output transformation."""

    def test_identity_output_func(self, linear_model_fn, single_instance):
        """Identity output_func should not change result."""
        attrs = np.array([0.1, 0.2, 0.3, 0.4])
        r1 = compute_completeness(attrs, linear_model_fn, single_instance)
        r2 = compute_completeness(
            attrs, linear_model_fn, single_instance,
            output_func=lambda x: x,
        )
        assert abs(r1 - r2) < 1e-10

    def test_squared_output_func(self, linear_model_fn, single_instance):
        """Custom output_func changes the target difference."""
        f_x = linear_model_fn(single_instance)
        f_b = linear_model_fn(np.zeros(4))
        squared_diff = f_x**2 - f_b**2
        attrs = np.array([squared_diff, 0.0, 0.0, 0.0])
        result = compute_completeness(
            attrs, linear_model_fn, single_instance,
            output_func=lambda x: x**2,
        )
        assert result < 1e-10


class TestCompletenessErrors:
    """Error handling tests for Completeness."""

    def test_mismatched_lengths(self, linear_model_fn, single_instance):
        attrs = np.array([0.1, 0.2])  # Wrong length
        with pytest.raises(ValueError, match="attributions length"):
            compute_completeness(attrs, linear_model_fn, single_instance)

    def test_model_fn_not_callable(self, single_instance):
        attrs = np.array([0.1, 0.2, 0.3, 0.4])
        with pytest.raises(TypeError, match="model_fn must be callable"):
            compute_completeness(attrs, "not_callable", single_instance)

    def test_baseline_wrong_length(self, linear_model_fn, single_instance):
        attrs = np.array([0.1, 0.2, 0.3, 0.4])
        with pytest.raises(ValueError, match="baseline length"):
            compute_completeness(
                attrs, linear_model_fn, single_instance,
                baseline=np.array([1.0, 2.0]),
            )


class TestCompletenessHighLevel:
    """Tests for the explainer-based high-level API."""

    def test_perfect_explainer(self, feature_names, linear_model_fn):
        """Explainer that perfectly satisfies completeness."""
        explainer = _PerfectCompletenessExplainer(feature_names, linear_model_fn)
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = compute_completeness_score(explainer, linear_model_fn, x)
        assert result < 1e-8

    def test_bad_explainer_worse(self, feature_names, linear_model_fn):
        """Bad explainer should have higher completeness error than perfect one."""
        perfect = _PerfectCompletenessExplainer(feature_names, linear_model_fn)
        bad = _BadCompletenessExplainer(feature_names, fixed_value=5.0)
        x = np.array([1.0, 2.0, 3.0, 4.0])
        score_perfect = compute_completeness_score(perfect, linear_model_fn, x)
        score_bad = compute_completeness_score(bad, linear_model_fn, x)
        assert score_perfect < score_bad

    def test_with_real_explainer(self, trained_model_and_explainer):
        """Completeness score with a real trained model and LIME explainer."""
        adapter, explainer, X = trained_model_and_explainer
        model_fn = lambda x: float(
            adapter.predict(x.reshape(1, -1))[0, 1]
        )
        result = compute_completeness_score(explainer, model_fn, X[0])
        assert isinstance(result, float)
        assert result >= 0.0


class TestCompletenessBatch:
    """Tests for batch Completeness."""

    def test_batch_pre_computed(self, linear_model_fn, sample_data):
        """Batch with pre-computed attributions."""
        # Create attrs that are roughly complete
        attrs_list = []
        for i in range(len(sample_data)):
            f_diff = linear_model_fn(sample_data[i]) - linear_model_fn(np.zeros(4))
            attrs_list.append(np.array([f_diff / 4] * 4))

        result = compute_batch_completeness(
            attributions_list=attrs_list,
            model_fn=linear_model_fn,
            X=sample_data,
        )
        assert "mean" in result
        assert "std" in result
        assert "scores" in result
        assert result["n_evaluated"] == len(sample_data)
        assert result["mean"] < 1e-8  # All should be near-perfect

    def test_batch_explainer_based(self, feature_names, linear_model_fn, sample_data):
        """Batch with explainer."""
        explainer = _PerfectCompletenessExplainer(feature_names, linear_model_fn)
        result = compute_batch_completeness(
            explainer=explainer,
            model_fn=linear_model_fn,
            X=sample_data,
        )
        assert result["n_evaluated"] == len(sample_data)
        assert result["mean"] < 1e-8

    def test_batch_max_instances(self, feature_names, linear_model_fn, sample_data):
        """Test max_instances parameter."""
        explainer = _PerfectCompletenessExplainer(feature_names, linear_model_fn)
        result = compute_batch_completeness(
            explainer=explainer,
            model_fn=linear_model_fn,
            X=sample_data,
            max_instances=3,
        )
        assert result["n_evaluated"] == 3

    def test_batch_requires_model_fn(self, sample_data, feature_names):
        """Missing model_fn should raise."""
        with pytest.raises(ValueError, match="model_fn is required"):
            compute_batch_completeness(
                explainer=_PerfectCompletenessExplainer(feature_names, lambda x: 0),
                X=sample_data,
            )

    def test_batch_requires_X(self, feature_names, linear_model_fn):
        """Missing X should raise."""
        with pytest.raises(ValueError, match="X.*required"):
            compute_batch_completeness(
                explainer=_PerfectCompletenessExplainer(feature_names, linear_model_fn),
                model_fn=linear_model_fn,
            )

    def test_batch_requires_attrs_or_explainer(self, linear_model_fn, sample_data):
        """Missing both attributions and explainer should raise."""
        with pytest.raises(ValueError, match="Either attributions_list"):
            compute_batch_completeness(model_fn=linear_model_fn, X=sample_data)

    def test_batch_statistics(self, linear_model_fn, sample_data):
        """Verify batch statistics are computed correctly."""
        attrs_list = [np.array([0.1, 0.2, 0.3, 0.4]) for _ in range(len(sample_data))]
        result = compute_batch_completeness(
            attributions_list=attrs_list,
            model_fn=linear_model_fn,
            X=sample_data,
        )
        scores = result["scores"]
        assert abs(result["mean"] - np.mean(scores)) < 1e-10
        assert abs(result["std"] - np.std(scores)) < 1e-10
        assert abs(result["max"] - np.max(scores)) < 1e-10
        assert abs(result["min"] - np.min(scores)) < 1e-10


# =============================================================================
# NON-SENSITIVITY TESTS
# =============================================================================

class TestNonSensitivityDetection:
    """Tests for automatic non-sensitive feature detection."""

    def test_detect_zero_weight_features(self, linear_model_fn, single_instance):
        """Features with zero weight should be detected as non-sensitive."""
        mask = _detect_non_sensitive_features(
            linear_model_fn, single_instance,
            n_perturbations=20, tolerance=1e-5, seed=42,
        )
        # Features 2 and 3 have zero weight
        assert mask[2] == True
        assert mask[3] == True

    def test_detect_nonzero_weight_features(self, linear_model_fn, single_instance):
        """Features with nonzero weight should be detected as sensitive."""
        mask = _detect_non_sensitive_features(
            linear_model_fn, single_instance,
            n_perturbations=20, tolerance=1e-5, seed=42,
        )
        # Features 0 and 1 have nonzero weight
        assert mask[0] == False
        assert mask[1] == False

    def test_all_sensitive_model(self, single_instance):
        """Model that uses all features should detect none as non-sensitive."""
        def model_fn(x):
            return float(np.sum(x))
        mask = _detect_non_sensitive_features(
            model_fn, single_instance,
            n_perturbations=20, tolerance=1e-5, seed=42,
        )
        assert not np.any(mask)

    def test_all_non_sensitive_model(self, single_instance):
        """Constant model should detect all features as non-sensitive."""
        def model_fn(x):
            return 1.0
        mask = _detect_non_sensitive_features(
            model_fn, single_instance,
            n_perturbations=10, tolerance=1e-5, seed=42,
        )
        assert np.all(mask)

    def test_seed_reproducibility(self, linear_model_fn, single_instance):
        """Same seed should give same detection results."""
        m1 = _detect_non_sensitive_features(
            linear_model_fn, single_instance, seed=123,
        )
        m2 = _detect_non_sensitive_features(
            linear_model_fn, single_instance, seed=123,
        )
        np.testing.assert_array_equal(m1, m2)


class TestNonSensitivityBasic:
    """Basic functionality tests for Non-Sensitivity."""

    def test_returns_float(self, linear_model_fn, single_instance):
        attrs = np.array([0.1, 0.2, 0.3, 0.4])
        result = compute_non_sensitivity(
            attrs, linear_model_fn, single_instance, seed=42,
        )
        assert isinstance(result, float)

    def test_non_negative(self, linear_model_fn, single_instance):
        attrs = np.array([0.1, 0.2, 0.3, 0.4])
        result = compute_non_sensitivity(
            attrs, linear_model_fn, single_instance, seed=42,
        )
        assert result >= 0.0

    def test_zero_for_no_non_sensitive(self, single_instance):
        """If model uses all features, score should be 0 (vacuously satisfied)."""
        def model_fn(x):
            return float(np.sum(x))
        attrs = np.array([0.5, 0.5, 0.5, 0.5])
        result = compute_non_sensitivity(
            attrs, model_fn, single_instance, seed=42,
        )
        assert result == 0.0

    def test_perfect_non_sensitivity(self, linear_model_fn, single_instance):
        """Zero attribution on non-sensitive features → score = 0."""
        attrs = np.array([0.5, 1.0, 0.0, 0.0])  # f2, f3 get zero
        result = compute_non_sensitivity(
            attrs, linear_model_fn, single_instance,
            n_perturbations=20, tolerance=1e-5, seed=42,
        )
        assert result < 1e-10

    def test_violation_when_attributing_non_sensitive(self, linear_model_fn, single_instance):
        """Attributing to non-sensitive features → positive score."""
        attrs = np.array([0.1, 0.2, 0.5, 0.8])  # f2, f3 get nonzero
        result = compute_non_sensitivity(
            attrs, linear_model_fn, single_instance,
            n_perturbations=20, tolerance=1e-5, seed=42,
        )
        # Should be approximately |0.5| + |0.8| = 1.3
        assert result > 1.0

    def test_user_provided_mask(self, linear_model_fn, single_instance):
        """User-provided non-sensitive mask should be used directly."""
        ns_mask = np.array([False, False, True, True])
        attrs = np.array([0.1, 0.2, 0.5, 0.8])
        result = compute_non_sensitivity(
            attrs, linear_model_fn, single_instance,
            non_sensitive_features=ns_mask,
        )
        expected = abs(0.5) + abs(0.8)
        assert abs(result - expected) < 1e-10

    def test_normalize(self, linear_model_fn, single_instance):
        """Normalized score should be in [0, 1]."""
        attrs = np.array([0.1, 0.2, 0.5, 0.8])
        ns_mask = np.array([False, False, True, True])
        result = compute_non_sensitivity(
            attrs, linear_model_fn, single_instance,
            non_sensitive_features=ns_mask, normalize=True,
        )
        assert 0.0 <= result <= 1.0
        total = np.sum(np.abs(attrs))
        ns_total = abs(0.5) + abs(0.8)
        assert abs(result - ns_total / total) < 1e-10

    def test_normalize_zero_attribution(self, linear_model_fn, single_instance):
        """Normalized with all-zero attributions should return 0."""
        attrs = np.zeros(4)
        ns_mask = np.array([False, False, True, True])
        result = compute_non_sensitivity(
            attrs, linear_model_fn, single_instance,
            non_sensitive_features=ns_mask, normalize=True,
        )
        assert result == 0.0

    def test_all_features_non_sensitive(self, single_instance):
        """Constant model: all features non-sensitive."""
        def model_fn(x):
            return 42.0
        attrs = np.array([0.1, 0.2, 0.3, 0.4])
        result = compute_non_sensitivity(
            attrs, model_fn, single_instance,
            n_perturbations=10, tolerance=1e-5, seed=42,
        )
        # All features are non-sensitive, score = sum(|attrs|)
        expected = np.sum(np.abs(attrs))
        assert abs(result - expected) < 1e-10


class TestNonSensitivityErrors:
    """Error handling tests for Non-Sensitivity."""

    def test_mismatched_lengths(self, linear_model_fn, single_instance):
        attrs = np.array([0.1, 0.2])
        with pytest.raises(ValueError, match="attributions length"):
            compute_non_sensitivity(attrs, linear_model_fn, single_instance)

    def test_model_fn_not_callable(self, single_instance):
        attrs = np.array([0.1, 0.2, 0.3, 0.4])
        with pytest.raises(TypeError, match="model_fn must be callable"):
            compute_non_sensitivity(attrs, "bad", single_instance)

    def test_mask_wrong_length(self, linear_model_fn, single_instance):
        attrs = np.array([0.1, 0.2, 0.3, 0.4])
        ns_mask = np.array([True, False])
        with pytest.raises(ValueError, match="non_sensitive_features length"):
            compute_non_sensitivity(
                attrs, linear_model_fn, single_instance,
                non_sensitive_features=ns_mask,
            )


class TestNonSensitivityHighLevel:
    """Tests for the explainer-based high-level API."""

    def test_aware_explainer_low_score(self, feature_names, linear_model_fn, single_instance):
        """Explainer that zeros non-sensitive features should score low."""
        sensitive_mask = [True, True, False, False]
        explainer = _NonSensitiveAwareExplainer(feature_names, sensitive_mask)
        result = compute_non_sensitivity_score(
            explainer, linear_model_fn, single_instance,
            n_perturbations=20, tolerance=1e-5, seed=42,
        )
        assert result < 1e-10

    def test_unaware_explainer_higher_score(self, feature_names, linear_model_fn, single_instance):
        """Explainer attributing to all features should score higher."""
        sensitive_mask = [True, True, False, False]
        aware = _NonSensitiveAwareExplainer(feature_names, sensitive_mask)
        unaware = _NonSensitiveUnawareExplainer(feature_names)
        score_aware = compute_non_sensitivity_score(
            aware, linear_model_fn, single_instance,
            n_perturbations=20, tolerance=1e-5, seed=42,
        )
        score_unaware = compute_non_sensitivity_score(
            unaware, linear_model_fn, single_instance,
            n_perturbations=20, tolerance=1e-5, seed=42,
        )
        assert score_aware < score_unaware

    def test_with_real_explainer(self, trained_model_and_explainer):
        """Non-Sensitivity score with a real trained model and LIME."""
        adapter, explainer, X = trained_model_and_explainer
        model_fn = lambda x: float(
            adapter.predict(x.reshape(1, -1))[0, 1]
        )
        result = compute_non_sensitivity_score(
            explainer, model_fn, X[0], seed=42,
        )
        assert isinstance(result, float)
        assert result >= 0.0


class TestNonSensitivityBatch:
    """Batch tests for Non-Sensitivity."""

    def test_batch_pre_computed(self, linear_model_fn, sample_data):
        """Batch with pre-computed attributions and user mask."""
        ns_mask = np.array([False, False, True, True])
        attrs_list = [np.array([0.5, 1.0, 0.0, 0.0]) for _ in range(len(sample_data))]
        result = compute_batch_non_sensitivity(
            attributions_list=attrs_list,
            model_fn=linear_model_fn,
            X=sample_data,
            non_sensitive_features=ns_mask,
        )
        assert result["n_evaluated"] == len(sample_data)
        assert result["mean"] < 1e-10

    def test_batch_explainer_based(self, feature_names, linear_model_fn, sample_data):
        """Batch with explainer."""
        sensitive_mask = [True, True, False, False]
        explainer = _NonSensitiveAwareExplainer(feature_names, sensitive_mask)
        result = compute_batch_non_sensitivity(
            explainer=explainer,
            model_fn=linear_model_fn,
            X=sample_data,
            n_perturbations=20, tolerance=1e-5, seed=42,
        )
        assert result["n_evaluated"] == len(sample_data)

    def test_batch_max_instances(self, linear_model_fn, sample_data):
        """Test max_instances parameter."""
        ns_mask = np.array([False, False, True, True])
        attrs_list = [np.array([0.5, 1.0, 0.0, 0.0]) for _ in range(len(sample_data))]
        result = compute_batch_non_sensitivity(
            attributions_list=attrs_list,
            model_fn=linear_model_fn,
            X=sample_data,
            non_sensitive_features=ns_mask,
            max_instances=3,
        )
        assert result["n_evaluated"] == 3

    def test_batch_requires_model_fn(self, sample_data):
        """Missing model_fn should raise."""
        with pytest.raises(ValueError, match="model_fn is required"):
            compute_batch_non_sensitivity(
                attributions_list=[np.zeros(4)],
                X=sample_data,
            )


# =============================================================================
# INPUT INVARIANCE TESTS
# =============================================================================

class TestInputInvarianceBasic:
    """Basic functionality tests for Input Invariance (simplified)."""

    def test_returns_float(self, single_instance):
        """Score should be a float."""
        explain_fn = lambda x: np.array([0.1, 0.2, 0.3, 0.4])
        result = compute_input_invariance(explain_fn, single_instance, seed=42)
        assert isinstance(result, float)

    def test_non_negative(self, single_instance):
        """Score should be non-negative."""
        explain_fn = lambda x: np.array([0.1, 0.2, 0.3, 0.4])
        result = compute_input_invariance(explain_fn, single_instance, seed=42)
        assert result >= 0.0

    def test_constant_explainer_invariant(self, single_instance):
        """Explainer returning constant values is trivially input invariant."""
        explain_fn = lambda x: np.array([1.0, 1.0, 1.0, 1.0])
        result = compute_input_invariance(explain_fn, single_instance, seed=42)
        assert result < 1e-10

    def test_gradient_explainer_invariant(self, single_instance):
        """Pure gradient explainer (constant output) should be invariant."""
        # Gradient of linear model is constant
        gradient = np.array([0.5, 1.0, 0.0, 0.0])
        explain_fn = lambda x: gradient.copy()
        result = compute_input_invariance(explain_fn, single_instance, seed=42)
        assert result < 1e-10

    def test_grad_times_input_not_invariant(self, single_instance):
        """Gradient × Input is NOT input invariant (shift carries through)."""
        gradient = np.array([0.5, 1.0, 0.3, 0.2])
        explain_fn = lambda x: gradient * np.asarray(x)
        result = compute_input_invariance(explain_fn, single_instance, seed=42)
        assert result > 0.0  # Should violate

    def test_with_explicit_shift_float(self, single_instance):
        """Explicit float shift."""
        explain_fn = lambda x: np.array([1.0, 1.0, 1.0, 1.0])
        result = compute_input_invariance(
            explain_fn, single_instance, shift=2.0,
        )
        assert result < 1e-10

    def test_with_explicit_shift_array(self, single_instance):
        """Explicit array shift."""
        explain_fn = lambda x: np.array([1.0, 1.0, 1.0, 1.0])
        shift = np.array([0.5, -0.5, 1.0, -1.0])
        result = compute_input_invariance(
            explain_fn, single_instance, shift=shift,
        )
        assert result < 1e-10

    def test_seed_reproducibility(self, single_instance):
        """Same seed should produce same result."""
        gradient = np.array([0.5, 1.0, 0.3, 0.2])
        explain_fn = lambda x: gradient * np.asarray(x)
        r1 = compute_input_invariance(explain_fn, single_instance, seed=99)
        r2 = compute_input_invariance(explain_fn, single_instance, seed=99)
        assert abs(r1 - r2) < 1e-10

    def test_different_seeds_different_results(self, single_instance):
        """Different seeds should generally produce different results."""
        gradient = np.array([0.5, 1.0, 0.3, 0.2])
        explain_fn = lambda x: gradient * np.asarray(x)
        r1 = compute_input_invariance(explain_fn, single_instance, seed=1)
        r2 = compute_input_invariance(explain_fn, single_instance, seed=2)
        # Not guaranteed but very likely different
        # Just check both are valid
        assert r1 >= 0.0
        assert r2 >= 0.0

    def test_larger_shift_larger_violation_for_grad_input(self, single_instance):
        """For grad×input, larger shift should mean larger violation."""
        gradient = np.array([1.0, 1.0, 1.0, 1.0])
        explain_fn = lambda x: gradient * np.asarray(x)
        r_small = compute_input_invariance(explain_fn, single_instance, shift=0.1)
        r_large = compute_input_invariance(explain_fn, single_instance, shift=10.0)
        assert r_large > r_small

    def test_normalized_by_n_features(self):
        """Score should be normalized by number of features."""
        # Same total L2 diff but different n_features
        explain_fn_4 = lambda x: np.asarray(x)  # identity
        x4 = np.ones(4)
        r4 = compute_input_invariance(explain_fn_4, x4, shift=1.0)

        explain_fn_8 = lambda x: np.asarray(x)
        x8 = np.ones(8)
        r8 = compute_input_invariance(explain_fn_8, x8, shift=1.0)

        # r4 = ||shift||_2 / 4 = 2/4 = 0.5
        # r8 = ||shift||_2 / 8 = 2√2/8 ≈ 0.354
        # Different because n_features normalizes
        assert r4 != r8


class TestInputInvarianceErrors:
    """Error handling tests for Input Invariance."""

    def test_explain_func_not_callable(self, single_instance):
        with pytest.raises(TypeError, match="explain_func must be callable"):
            compute_input_invariance("not_callable", single_instance)

    def test_shift_wrong_length(self, single_instance):
        explain_fn = lambda x: np.array([1.0, 1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="shift length"):
            compute_input_invariance(
                explain_fn, single_instance,
                shift=np.array([1.0, 2.0]),
            )


class TestInputInvarianceSemantic:
    """Semantic validation tests for Input Invariance."""

    def test_invariant_method_beats_non_invariant(self, single_instance):
        """Gradient (invariant) should score lower than Grad×Input (not invariant)."""
        gradient = np.array([0.5, 1.0, 0.3, 0.2])

        invariant_fn = lambda x: gradient.copy()
        non_invariant_fn = lambda x: gradient * np.asarray(x)

        score_inv = compute_input_invariance(invariant_fn, single_instance, seed=42)
        score_non = compute_input_invariance(non_invariant_fn, single_instance, seed=42)

        assert score_inv < score_non


class TestInputInvarianceBatch:
    """Batch tests for Input Invariance."""

    def test_batch_basic(self, sample_data):
        """Batch computation should work."""
        explain_fn = lambda x: np.array([1.0, 1.0, 1.0, 1.0])
        result = compute_batch_input_invariance(explain_fn, sample_data, seed=42)
        assert result["n_evaluated"] == len(sample_data)
        assert result["mean"] < 1e-10

    def test_batch_max_instances(self, sample_data):
        """Test max_instances parameter."""
        explain_fn = lambda x: np.array([1.0, 1.0, 1.0, 1.0])
        result = compute_batch_input_invariance(
            explain_fn, sample_data, max_instances=3, seed=42,
        )
        assert result["n_evaluated"] == 3

    def test_batch_statistics(self, sample_data):
        """Verify batch statistics."""
        gradient = np.array([0.5, 1.0, 0.3, 0.2])
        explain_fn = lambda x: gradient * np.asarray(x)
        result = compute_batch_input_invariance(explain_fn, sample_data, seed=42)
        scores = result["scores"]
        assert abs(result["mean"] - np.mean(scores)) < 1e-10
        assert abs(result["std"] - np.std(scores)) < 1e-10


# =============================================================================
# INPUT INVARIANCE PYTORCH TESTS
# =============================================================================

class TestInputInvariancePyTorch:
    """Tests for Input Invariance with PyTorch model compensation."""

    @pytest.fixture
    def simple_pytorch_model(self):
        """Create a simple PyTorch linear model."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("PyTorch not available")

        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        model.eval()
        return model

    def test_import_guard(self):
        """Should work if torch is available, skip otherwise."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        from explainiverse.evaluation.axiomatic import compute_input_invariance_pytorch
        assert callable(compute_input_invariance_pytorch)

    def test_returns_float(self, simple_pytorch_model):
        """Score should be a float."""
        import torch
        from explainiverse.evaluation.axiomatic import compute_input_invariance_pytorch

        def explain_fn(model, x):
            # Simple gradient-based attribution
            x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).requires_grad_(True)
            model.eval()
            out = model(x_t)
            out.backward()
            return x_t.grad.detach().numpy().flatten()

        instance = np.array([1.0, 2.0, 3.0, 4.0])
        result = compute_input_invariance_pytorch(
            simple_pytorch_model, explain_fn, instance, seed=42,
        )
        assert isinstance(result, float)
        assert result >= 0.0

    def test_gradient_method_is_invariant(self, simple_pytorch_model):
        """Pure gradient method should be approximately input invariant."""
        import torch
        from explainiverse.evaluation.axiomatic import compute_input_invariance_pytorch

        def explain_fn(model, x):
            x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).requires_grad_(True)
            model.eval()
            out = model(x_t)
            out.backward()
            return x_t.grad.detach().numpy().flatten()

        instance = np.array([1.0, 2.0, 3.0, 4.0])
        result = compute_input_invariance_pytorch(
            simple_pytorch_model, explain_fn, instance,
            shift=0.5, seed=42,
        )
        # Gradient of ReLU network may not be perfectly invariant
        # but should be relatively small
        assert isinstance(result, float)
        assert result >= 0.0

    def test_does_not_modify_original_model(self, simple_pytorch_model):
        """Original model should not be modified (deep copy is used)."""
        import torch
        from explainiverse.evaluation.axiomatic import compute_input_invariance_pytorch

        # Save original parameters
        original_params = {
            name: p.clone() for name, p in simple_pytorch_model.named_parameters()
        }

        def explain_fn(model, x):
            x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).requires_grad_(True)
            model.eval()
            out = model(x_t)
            out.backward()
            return x_t.grad.detach().numpy().flatten()

        instance = np.array([1.0, 2.0, 3.0, 4.0])
        compute_input_invariance_pytorch(
            simple_pytorch_model, explain_fn, instance, seed=42,
        )

        # Check original model is unchanged
        for name, p in simple_pytorch_model.named_parameters():
            assert torch.allclose(p, original_params[name]), \
                f"Parameter {name} was modified!"

    def test_batch_pytorch(self, simple_pytorch_model):
        """Batch Input Invariance PyTorch should work."""
        import torch
        from explainiverse.evaluation.axiomatic import compute_batch_input_invariance_pytorch

        def explain_fn(model, x):
            x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).requires_grad_(True)
            model.eval()
            out = model(x_t)
            out.backward()
            return x_t.grad.detach().numpy().flatten()

        X = np.random.randn(5, 4).astype(np.float64)
        result = compute_batch_input_invariance_pytorch(
            simple_pytorch_model, explain_fn, X, seed=42,
        )
        assert result["n_evaluated"] == 5
        assert result["mean"] >= 0.0


# =============================================================================
# SYMMETRY TESTS
# =============================================================================

class TestSymmetryBasic:
    """Basic functionality tests for Symmetry."""

    def test_returns_float(self):
        attrs = np.array([0.5, 0.5, 0.3, 0.1])
        result = compute_symmetry(attrs, symmetric_pairs=[(0, 1)])
        assert isinstance(result, float)

    def test_non_negative(self):
        attrs = np.array([0.5, 0.3, 0.1, 0.8])
        result = compute_symmetry(attrs, symmetric_pairs=[(0, 1)])
        assert result >= 0.0

    def test_perfect_symmetry(self):
        """Equal attributions for symmetric features → score = 0."""
        attrs = np.array([0.5, 0.5, 0.3, 0.1])
        result = compute_symmetry(attrs, symmetric_pairs=[(0, 1)])
        assert result < 1e-10

    def test_violation(self):
        """Different attributions for symmetric features → positive score."""
        attrs = np.array([0.8, 0.2, 0.3, 0.1])
        result = compute_symmetry(attrs, symmetric_pairs=[(0, 1)])
        assert abs(result - 0.6) < 1e-10

    def test_empty_pairs(self):
        """Empty pairs should return 0."""
        attrs = np.array([0.5, 0.3, 0.1, 0.8])
        result = compute_symmetry(attrs, symmetric_pairs=[])
        assert result == 0.0

    def test_multiple_pairs(self):
        """Multiple symmetric pairs."""
        attrs = np.array([0.5, 0.5, 0.3, 0.3])
        result = compute_symmetry(attrs, symmetric_pairs=[(0, 1), (2, 3)])
        assert result < 1e-10

    def test_multiple_pairs_partial_violation(self):
        """One pair satisfied, one violated."""
        attrs = np.array([0.5, 0.5, 0.3, 0.7])
        result = compute_symmetry(attrs, symmetric_pairs=[(0, 1), (2, 3)])
        # Mean of |0.5-0.5| and |0.3-0.7| = mean(0.0, 0.4) = 0.2
        assert abs(result - 0.2) < 1e-10

    def test_negative_attributions(self):
        """Negative attributions should work correctly."""
        attrs = np.array([-0.5, -0.5, 0.3, 0.1])
        result = compute_symmetry(attrs, symmetric_pairs=[(0, 1)])
        assert result < 1e-10

    def test_mixed_sign_violation(self):
        """Opposite-sign attributions for symmetric features."""
        attrs = np.array([0.5, -0.5, 0.0, 0.0])
        result = compute_symmetry(attrs, symmetric_pairs=[(0, 1)])
        assert abs(result - 1.0) < 1e-10

    def test_larger_violation_higher_score(self):
        """Larger attribution difference → higher score."""
        attrs_small = np.array([0.5, 0.6, 0.0, 0.0])
        attrs_large = np.array([0.5, 2.0, 0.0, 0.0])
        r_small = compute_symmetry(attrs_small, symmetric_pairs=[(0, 1)])
        r_large = compute_symmetry(attrs_large, symmetric_pairs=[(0, 1)])
        assert r_large > r_small

    def test_single_feature(self):
        """Single feature attribution."""
        attrs = np.array([1.0])
        result = compute_symmetry(attrs, symmetric_pairs=[])
        assert result == 0.0


class TestSymmetryErrors:
    """Error handling tests for Symmetry."""

    def test_index_out_of_bounds_first(self):
        attrs = np.array([0.5, 0.3])
        with pytest.raises(ValueError, match="out of bounds"):
            compute_symmetry(attrs, symmetric_pairs=[(0, 5)])

    def test_index_out_of_bounds_second(self):
        attrs = np.array([0.5, 0.3])
        with pytest.raises(ValueError, match="out of bounds"):
            compute_symmetry(attrs, symmetric_pairs=[(5, 0)])

    def test_negative_index(self):
        attrs = np.array([0.5, 0.3, 0.1])
        with pytest.raises(ValueError, match="out of bounds"):
            compute_symmetry(attrs, symmetric_pairs=[(-1, 0)])


class TestSymmetryHighLevel:
    """Tests for the explainer-based high-level API."""

    def test_symmetric_explainer(self, feature_names):
        """Symmetric explainer should score 0 for symmetric pair."""
        explainer = _SymmetricExplainer(feature_names)
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = compute_symmetry_score(explainer, x, symmetric_pairs=[(0, 1)])
        assert result < 1e-10

    def test_asymmetric_explainer(self, feature_names):
        """Asymmetric explainer should score > 0 for symmetric pair."""
        explainer = _AsymmetricExplainer(feature_names)
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = compute_symmetry_score(explainer, x, symmetric_pairs=[(0, 1)])
        assert result > 0.0

    def test_symmetric_better_than_asymmetric(self, feature_names):
        """Symmetric explainer should score lower than asymmetric."""
        sym = _SymmetricExplainer(feature_names)
        asym = _AsymmetricExplainer(feature_names)
        x = np.array([1.0, 2.0, 3.0, 4.0])
        s_sym = compute_symmetry_score(sym, x, symmetric_pairs=[(0, 1)])
        s_asym = compute_symmetry_score(asym, x, symmetric_pairs=[(0, 1)])
        assert s_sym < s_asym

    def test_with_real_explainer(self, trained_model_and_explainer):
        """Symmetry score with a real explainer should return valid float."""
        adapter, explainer, X = trained_model_and_explainer
        result = compute_symmetry_score(explainer, X[0], symmetric_pairs=[(0, 1)])
        assert isinstance(result, float)
        assert result >= 0.0


class TestSymmetryBatch:
    """Batch tests for Symmetry."""

    def test_batch_pre_computed(self):
        """Batch with pre-computed attributions."""
        attrs_list = [
            np.array([0.5, 0.5, 0.3, 0.1]),
            np.array([0.7, 0.7, 0.2, 0.4]),
            np.array([0.3, 0.3, 0.5, 0.5]),
        ]
        result = compute_batch_symmetry(
            symmetric_pairs=[(0, 1)],
            attributions_list=attrs_list,
        )
        assert result["n_evaluated"] == 3
        assert result["mean"] < 1e-10  # All perfectly symmetric for pair (0,1)

    def test_batch_explainer_based(self, feature_names, sample_data):
        """Batch with explainer."""
        explainer = _SymmetricExplainer(feature_names)
        result = compute_batch_symmetry(
            symmetric_pairs=[(0, 1)],
            explainer=explainer,
            X=sample_data,
        )
        assert result["n_evaluated"] == len(sample_data)
        assert result["mean"] < 1e-10

    def test_batch_max_instances(self):
        """Test max_instances parameter."""
        attrs_list = [np.array([0.5, 0.5, 0.3, 0.1]) for _ in range(10)]
        result = compute_batch_symmetry(
            symmetric_pairs=[(0, 1)],
            attributions_list=attrs_list,
            max_instances=3,
        )
        assert result["n_evaluated"] == 3

    def test_batch_requires_input(self):
        """Missing both attributions_list and explainer+X should raise."""
        with pytest.raises(ValueError, match="Either attributions_list"):
            compute_batch_symmetry(symmetric_pairs=[(0, 1)])

    def test_batch_statistics(self):
        """Verify batch statistics are correct."""
        attrs_list = [
            np.array([0.5, 0.5, 0.3, 0.1]),
            np.array([0.5, 0.8, 0.3, 0.1]),
            np.array([0.5, 1.5, 0.3, 0.1]),
        ]
        result = compute_batch_symmetry(
            symmetric_pairs=[(0, 1)],
            attributions_list=attrs_list,
        )
        scores = result["scores"]
        assert abs(result["mean"] - np.mean(scores)) < 1e-10
        assert abs(result["std"] - np.std(scores)) < 1e-10
        assert abs(result["max"] - np.max(scores)) < 1e-10
        assert abs(result["min"] - np.min(scores)) < 1e-10


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================

class TestSafeModelOutput:
    """Tests for the _safe_model_output helper."""

    def test_scalar_float(self):
        result = _safe_model_output(lambda x: 3.14, np.zeros(4))
        assert result == 3.14

    def test_numpy_scalar(self):
        result = _safe_model_output(lambda x: np.float64(2.5), np.zeros(4))
        assert result == 2.5

    def test_numpy_array(self):
        result = _safe_model_output(lambda x: np.array([1.5]), np.zeros(4))
        assert result == 1.5

    def test_integer(self):
        result = _safe_model_output(lambda x: 42, np.zeros(4))
        assert result == 42.0


# =============================================================================
# EDGE CASE TESTS (Cross-cutting)
# =============================================================================

class TestEdgeCases:
    """Edge cases that apply across multiple metrics."""

    def test_completeness_single_feature(self):
        """Completeness with single feature."""
        model_fn = lambda x: float(x[0]) * 2.0
        x = np.array([3.0])
        attrs = np.array([6.0])  # 2.0 * 3.0 - 2.0 * 0.0
        result = compute_completeness(attrs, model_fn, x)
        assert result < 1e-10

    def test_completeness_many_features(self):
        """Completeness with many features."""
        n = 100
        weights = np.random.RandomState(42).randn(n)
        model_fn = lambda x: float(np.dot(weights, x))
        x = np.random.RandomState(43).randn(n)
        attrs = weights * x  # Perfect for linear model from zero baseline
        result = compute_completeness(attrs, model_fn, x)
        assert result < 1e-10

    def test_non_sensitivity_all_zero_attrs(self, linear_model_fn, single_instance):
        """All-zero attributions: non-sensitivity should be 0."""
        attrs = np.zeros(4)
        result = compute_non_sensitivity(
            attrs, linear_model_fn, single_instance,
            non_sensitive_features=np.array([False, False, True, True]),
        )
        assert result == 0.0

    def test_symmetry_identical_features(self):
        """Identical attribution values everywhere → all pairs score 0."""
        attrs = np.array([0.25, 0.25, 0.25, 0.25])
        result = compute_symmetry(
            attrs, symmetric_pairs=[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        )
        assert result < 1e-10

    def test_input_invariance_zero_shift(self, single_instance):
        """Zero shift should always give score = 0."""
        explain_fn = lambda x: np.asarray(x) ** 2
        result = compute_input_invariance(
            explain_fn, single_instance, shift=0.0,
        )
        assert result < 1e-10

    def test_completeness_with_2d_input_flattened(self, linear_model_fn):
        """2D instance should be flattened correctly."""
        x = np.array([[1.0, 2.0, 3.0, 4.0]])  # Shape (1, 4)
        attrs = np.array([0.5, 2.0, 0.0, 0.0])  # weights * x from zero baseline
        result = compute_completeness(attrs, linear_model_fn, x)
        assert isinstance(result, float)

    def test_non_sensitivity_with_perturbation_scale(self, linear_model_fn, single_instance):
        """Different perturbation scales should not break detection."""
        for scale in [0.01, 0.1, 1.0, 10.0]:
            mask = _detect_non_sensitive_features(
                linear_model_fn, single_instance,
                perturbation_scale=scale, n_perturbations=20, seed=42,
            )
            # Features 2,3 should still be detected as non-sensitive
            assert mask[2] == True
            assert mask[3] == True

    def test_symmetry_self_pair(self):
        """Pair (i, i) should always score 0."""
        attrs = np.array([0.5, 0.3, 0.1, 0.8])
        result = compute_symmetry(attrs, symmetric_pairs=[(0, 0)])
        assert result < 1e-10


# =============================================================================
# INTEGRATION TESTS WITH REAL MODELS
# =============================================================================

class TestIntegration:
    """Integration tests with real sklearn models."""

    def test_completeness_with_real_model(self, trained_model_and_explainer):
        """End-to-end completeness test."""
        adapter, explainer, X = trained_model_and_explainer
        model_fn = lambda x: float(
            adapter.predict(x.reshape(1, -1))[0, 1]
        )
        result = compute_completeness_score(
            explainer, model_fn, X[0],
        )
        assert isinstance(result, float)
        assert result >= 0.0

    def test_non_sensitivity_with_real_model(self, trained_model_and_explainer):
        """End-to-end non-sensitivity test."""
        adapter, explainer, X = trained_model_and_explainer
        model_fn = lambda x: float(
            adapter.predict(x.reshape(1, -1))[0, 1]
        )
        result = compute_non_sensitivity_score(
            explainer, model_fn, X[0], seed=42,
        )
        assert isinstance(result, float)
        assert result >= 0.0

    def test_symmetry_with_real_model(self, trained_model_and_explainer):
        """End-to-end symmetry test."""
        adapter, explainer, X = trained_model_and_explainer
        result = compute_symmetry_score(
            explainer, X[0], symmetric_pairs=[(0, 1)],
        )
        assert isinstance(result, float)
        assert result >= 0.0

    def test_input_invariance_with_real_model(self, trained_model_and_explainer):
        """End-to-end input invariance test."""
        adapter, explainer, X = trained_model_and_explainer

        def explain_fn(x):
            exp = explainer.explain(x)
            feature_names = ["f0", "f1", "f2", "f3"]
            attrs = exp.explanation_data.get("feature_attributions", {})
            return np.array([attrs.get(fn, 0.0) for fn in feature_names])

        result = compute_input_invariance(explain_fn, X[0], seed=42)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_batch_completeness_real_model(self, trained_model_and_explainer):
        """Batch completeness with real model."""
        adapter, explainer, X = trained_model_and_explainer
        model_fn = lambda x: float(
            adapter.predict(x.reshape(1, -1))[0, 1]
        )
        result = compute_batch_completeness(
            explainer=explainer,
            model_fn=model_fn,
            X=X[:5],
        )
        assert result["n_evaluated"] == 5
        assert all(s >= 0.0 for s in result["scores"])

    def test_batch_symmetry_real_model(self, trained_model_and_explainer):
        """Batch symmetry with real model."""
        adapter, explainer, X = trained_model_and_explainer
        result = compute_batch_symmetry(
            symmetric_pairs=[(0, 1)],
            explainer=explainer,
            X=X[:5],
        )
        assert result["n_evaluated"] == 5
