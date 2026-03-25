# tests/test_fairness.py
"""
Tests for Phase 7 fairness evaluation metrics.

Metrics:
1. Group Fairness (Dai et al., 2022) — composable disparity across demographic groups
2. Individual Fairness (Dwork et al., 2012; adapted for XAI) — similar individuals get
   similar explanations regardless of protected attribute
3. Counterfactual Explanation Fairness (Kusner et al., 2017; adapted) — explanations
   should not change when only the protected attribute is flipped
4. Fidelity Disparity (Balagopalan et al., 2022) — max/mean explanation quality gaps
   across subgroup pairs
5. Attribution Parity (novel synthesis from Dai et al. + Aïvodji et al., 2019) —
   whether the protected feature itself receives disproportionate attribution
6. Conditional Fairness (Hardt et al., 2016; adapted) — explanation quality equality
   conditioned on model prediction

Also tests the FairnessMetricRegistry for extensibility.

References:
    Dai, J., Upadhyay, S., Aïvodji, U., Bach, S. H., & Lakkaraju, H. (2022).
    Fairness via Explanation Quality: Evaluating Disparities in the Quality
    of Post hoc Explanations. AIES. https://doi.org/10.1145/3514094.3534159

    Balagopalan, A., Zhang, H., Hamidieh, K., Hartvigsen, T., Rudzicz, F., &
    Ghassemi, M. (2022). The Road to Explainability is Paved with Bias:
    Measuring the Fairness of Explanations. FAccT.
    https://doi.org/10.1145/3531146.3533179

    Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R. (2012).
    Fairness Through Awareness. ITCS.

    Kusner, M. J., Loftus, J., Russell, C., & Silva, R. (2017).
    Counterfactual Fairness. NeurIPS.

    Hardt, M., Price, E., & Srebro, N. (2016). Equality of Opportunity in
    Supervised Learning. NeurIPS.

    Aïvodji, U., Arai, H., Fortineau, O., Gambs, S., Hara, S., & Tapp, A.
    (2019). Fairwashing: the risk of rationalization. ICML.
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
    return ["age", "income", "gender", "education", "credit_score"]


@pytest.fixture
def feature_names_with_sensitive():
    """Feature names where 'gender' (index 2) is the sensitive attribute."""
    return ["age", "income", "gender", "education", "credit_score"]


@pytest.fixture
def sample_data():
    """20 samples, 5 features. Feature index 2 is binary (sensitive)."""
    np.random.seed(42)
    X = np.random.randn(20, 5).astype(np.float64)
    # Make feature 2 binary (simulating a sensitive attribute like gender)
    X[:, 2] = np.random.choice([0, 1], size=20)
    return X


@pytest.fixture
def sensitive_features_binary(sample_data):
    """Binary sensitive feature array derived from sample_data column 2."""
    return sample_data[:, 2].astype(int)


@pytest.fixture
def sensitive_features_multigroup():
    """Multi-valued sensitive feature (3 groups)."""
    np.random.seed(42)
    return np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                     0, 1, 2, 0, 1])


@pytest.fixture
def fair_attributions(sample_data):
    """
    Attributions that are roughly equal across groups.
    All instances get identical attributions -> no disparity.
    """
    n = sample_data.shape[0]
    # All instances get the same attribution vector
    return np.tile([0.3, 0.5, 0.0, 0.15, 0.05], (n, 1))


@pytest.fixture
def unfair_attributions(sample_data, sensitive_features_binary):
    """
    Attributions with intentional disparity across groups.
    Group 0 gets high-quality (sparse, focused) attributions.
    Group 1 gets low-quality (spread, noisy) attributions.
    """
    n = sample_data.shape[0]
    attrs = np.zeros((n, 5))
    for i in range(n):
        if sensitive_features_binary[i] == 0:
            # Group 0: sparse, focused attribution
            attrs[i] = [0.7, 0.2, 0.0, 0.1, 0.0]
        else:
            # Group 1: spread, noisy attribution
            attrs[i] = [0.2, 0.2, 0.1, 0.25, 0.25]
    return attrs


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
        n_samples=100, n_features=5, n_informative=3,
        n_redundant=0, n_classes=2, random_state=42,
    )
    X = X.astype(np.float32)
    # Make feature index 2 binary (sensitive attribute)
    X[:, 2] = (X[:, 2] > 0).astype(np.float32)

    clf = GradientBoostingClassifier(n_estimators=20, random_state=42)
    clf.fit(X, y)

    adapter = SklearnAdapter(
        clf, feature_names=feature_names, class_names=["denied", "approved"]
    )
    explainer = LimeExplainer(
        model=adapter,
        training_data=X,
        feature_names=feature_names,
        class_names=["denied", "approved"],
    )
    return adapter, explainer, X, y


# =============================================================================
# Helper: Build attribution arrays from Explanation objects
# =============================================================================

def _make_explanation(feature_names, attribution_values):
    """Create an Explanation with given attribution values."""
    return Explanation(
        explainer_name="test",
        target_class="class_0",
        explanation_data={
            "feature_attributions": dict(zip(feature_names, attribution_values))
        },
        feature_names=feature_names,
    )


# =============================================================================
# 1. FairnessMetricRegistry Tests
# =============================================================================

class TestFairnessMetricRegistry:
    """Tests for the extensible fairness metric registry."""

    def test_import_registry(self):
        """FairnessMetricRegistry can be imported."""
        from explainiverse.evaluation.fairness import (
            FairnessMetricRegistry,
            FairnessMetricMeta,
        )
        assert FairnessMetricRegistry is not None
        assert FairnessMetricMeta is not None

    def test_create_empty_registry(self):
        """Empty registry can be created."""
        from explainiverse.evaluation.fairness import FairnessMetricRegistry
        registry = FairnessMetricRegistry()
        assert registry.list_metrics() == []

    def test_register_metric(self):
        """A fairness metric can be registered."""
        from explainiverse.evaluation.fairness import (
            FairnessMetricRegistry,
            FairnessMetricMeta,
        )
        registry = FairnessMetricRegistry()

        def dummy_metric(attributions, sensitive_features, **kwargs):
            return {"score": 0.0}

        registry.register(
            name="dummy",
            metric_fn=dummy_metric,
            meta=FairnessMetricMeta(
                level="group",
                description="A dummy fairness metric",
            ),
        )
        assert "dummy" in registry.list_metrics()

    def test_register_duplicate_raises(self):
        """Registering the same name twice raises ValueError."""
        from explainiverse.evaluation.fairness import (
            FairnessMetricRegistry,
            FairnessMetricMeta,
        )
        registry = FairnessMetricRegistry()

        def dummy(a, s, **kw):
            return {"score": 0.0}

        registry.register("dummy", dummy, FairnessMetricMeta(level="group"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register("dummy", dummy, FairnessMetricMeta(level="group"))

    def test_register_override(self):
        """Override flag allows re-registration."""
        from explainiverse.evaluation.fairness import (
            FairnessMetricRegistry,
            FairnessMetricMeta,
        )
        registry = FairnessMetricRegistry()

        def v1(a, s, **kw):
            return {"score": 1.0}

        def v2(a, s, **kw):
            return {"score": 2.0}

        registry.register("dummy", v1, FairnessMetricMeta(level="group"))
        registry.register(
            "dummy", v2, FairnessMetricMeta(level="group"), override=True
        )
        result = registry.get("dummy")
        assert result["fn"] is v2

    def test_unregister_metric(self):
        """Metric can be unregistered."""
        from explainiverse.evaluation.fairness import (
            FairnessMetricRegistry,
            FairnessMetricMeta,
        )
        registry = FairnessMetricRegistry()

        def dummy(a, s, **kw):
            return {"score": 0.0}

        registry.register("dummy", dummy, FairnessMetricMeta(level="group"))
        registry.unregister("dummy")
        assert "dummy" not in registry.list_metrics()

    def test_unregister_unknown_raises(self):
        """Unregistering unknown metric raises KeyError."""
        from explainiverse.evaluation.fairness import FairnessMetricRegistry
        registry = FairnessMetricRegistry()
        with pytest.raises(KeyError, match="not registered"):
            registry.unregister("nonexistent")

    def test_get_unknown_raises(self):
        """Getting unknown metric raises KeyError."""
        from explainiverse.evaluation.fairness import FairnessMetricRegistry
        registry = FairnessMetricRegistry()
        with pytest.raises(KeyError, match="not registered"):
            registry.get("nonexistent")

    def test_decorator_registration(self):
        """Decorator-based registration works."""
        from explainiverse.evaluation.fairness import (
            FairnessMetricRegistry,
            FairnessMetricMeta,
        )
        registry = FairnessMetricRegistry()

        @registry.register_decorator(
            name="decorated",
            meta=FairnessMetricMeta(level="individual"),
        )
        def my_metric(attributions, sensitive_features, **kwargs):
            return {"score": 42.0}

        assert "decorated" in registry.list_metrics()
        result = registry.evaluate("decorated", np.ones((5, 3)), np.array([0, 0, 1, 1, 1]))
        assert result["score"] == 42.0

    def test_evaluate_calls_registered_function(self):
        """registry.evaluate() calls the registered function with correct args."""
        from explainiverse.evaluation.fairness import (
            FairnessMetricRegistry,
            FairnessMetricMeta,
        )
        registry = FairnessMetricRegistry()
        call_log = []

        def tracking_metric(attributions, sensitive_features, **kwargs):
            call_log.append((attributions.shape, sensitive_features.shape))
            return {"score": 0.5}

        registry.register("tracker", tracking_metric, FairnessMetricMeta(level="group"))
        attrs = np.ones((10, 4))
        sf = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        result = registry.evaluate("tracker", attrs, sf)
        assert len(call_log) == 1
        assert call_log[0] == ((10, 4), (10,))
        assert result["score"] == 0.5

    def test_list_metrics_with_meta(self):
        """list_metrics(with_meta=True) returns metadata."""
        from explainiverse.evaluation.fairness import (
            FairnessMetricRegistry,
            FairnessMetricMeta,
        )
        registry = FairnessMetricRegistry()

        def d(a, s, **kw):
            return {"score": 0.0}

        registry.register(
            "test_metric", d,
            FairnessMetricMeta(level="group", composable=True, description="Test"),
        )
        result = registry.list_metrics(with_meta=True)
        assert isinstance(result, dict)
        assert "test_metric" in result
        assert result["test_metric"]["meta"].level == "group"
        assert result["test_metric"]["meta"].composable is True

    def test_filter_by_level(self):
        """Metrics can be filtered by level (group, individual, conditional)."""
        from explainiverse.evaluation.fairness import (
            FairnessMetricRegistry,
            FairnessMetricMeta,
        )
        registry = FairnessMetricRegistry()

        def d(a, s, **kw):
            return {"score": 0.0}

        registry.register("g1", d, FairnessMetricMeta(level="group"))
        registry.register("g2", d, FairnessMetricMeta(level="group"))
        registry.register("i1", d, FairnessMetricMeta(level="individual"))

        group_metrics = registry.filter(level="group")
        assert set(group_metrics) == {"g1", "g2"}
        individual_metrics = registry.filter(level="individual")
        assert individual_metrics == ["i1"]

    def test_default_registry_has_builtin_metrics(self):
        """The default fairness registry includes all 6 built-in metrics."""
        from explainiverse.evaluation.fairness import get_default_fairness_registry
        registry = get_default_fairness_registry()
        metrics = registry.list_metrics()
        expected = [
            "group_fairness",
            "individual_fairness",
            "counterfactual_fairness",
            "fidelity_disparity",
            "attribution_parity",
            "conditional_fairness",
        ]
        for name in expected:
            assert name in metrics, f"Built-in metric '{name}' missing from default registry"

    def test_summary_output(self):
        """summary() returns a human-readable string."""
        from explainiverse.evaluation.fairness import (
            FairnessMetricRegistry,
            FairnessMetricMeta,
        )
        registry = FairnessMetricRegistry()

        def d(a, s, **kw):
            return {"score": 0.0}

        registry.register("test_m", d, FairnessMetricMeta(level="group", description="Test metric"))
        s = registry.summary()
        assert isinstance(s, str)
        assert "test_m" in s


# =============================================================================
# 2. Group Fairness Tests — compute_group_fairness()
# =============================================================================

class TestGroupFairness:
    """Tests for the Group Fairness metric (Dai et al., 2022)."""

    def test_import(self):
        """compute_group_fairness can be imported."""
        from explainiverse.evaluation.fairness import compute_group_fairness
        assert callable(compute_group_fairness)

    def test_basic_fair_attributions(
        self, fair_attributions, sensitive_features_binary
    ):
        """
        Identical attributions across groups should produce zero disparity.
        """
        from explainiverse.evaluation.fairness import compute_group_fairness

        result = compute_group_fairness(
            attributions=fair_attributions,
            sensitive_features=sensitive_features_binary,
        )
        assert isinstance(result, dict)
        assert "disparity" in result
        assert result["disparity"] == pytest.approx(0.0, abs=1e-10)

    def test_basic_unfair_attributions(
        self, unfair_attributions, sensitive_features_binary
    ):
        """
        Intentionally disparate attributions should produce non-zero disparity.
        """
        from explainiverse.evaluation.fairness import compute_group_fairness

        result = compute_group_fairness(
            attributions=unfair_attributions,
            sensitive_features=sensitive_features_binary,
        )
        assert result["disparity"] > 0.0

    def test_returns_statistical_test(
        self, unfair_attributions, sensitive_features_binary
    ):
        """
        Result must contain Mann-Whitney U p-value and Cohen's d effect size.
        """
        from explainiverse.evaluation.fairness import compute_group_fairness

        result = compute_group_fairness(
            attributions=unfair_attributions,
            sensitive_features=sensitive_features_binary,
        )
        assert "p_value" in result
        assert "effect_size" in result
        assert 0.0 <= result["p_value"] <= 1.0
        assert isinstance(result["effect_size"], float)

    def test_returns_per_group_means(
        self, unfair_attributions, sensitive_features_binary
    ):
        """
        Result must contain per-group metric means.
        """
        from explainiverse.evaluation.fairness import compute_group_fairness

        result = compute_group_fairness(
            attributions=unfair_attributions,
            sensitive_features=sensitive_features_binary,
        )
        assert "group_means" in result
        assert isinstance(result["group_means"], dict)
        # Should have entries for group 0 and group 1
        assert len(result["group_means"]) == 2

    def test_default_inner_metric(self, sample_data, sensitive_features_binary):
        """
        Default inner metric (L1 norm / sparseness) should work without
        user specifying one.
        """
        from explainiverse.evaluation.fairness import compute_group_fairness

        np.random.seed(99)
        attrs = np.random.randn(sample_data.shape[0], sample_data.shape[1])
        result = compute_group_fairness(
            attributions=attrs,
            sensitive_features=sensitive_features_binary,
        )
        assert "disparity" in result
        assert isinstance(result["disparity"], float)

    def test_custom_inner_metric(
        self, unfair_attributions, sensitive_features_binary
    ):
        """
        User-supplied inner metric function should be called correctly.
        """
        from explainiverse.evaluation.fairness import compute_group_fairness

        def custom_sparsity(attr_vector):
            """Count non-zero attributions (higher = less sparse)."""
            return float(np.count_nonzero(np.abs(attr_vector) > 0.05))

        result = compute_group_fairness(
            attributions=unfair_attributions,
            sensitive_features=sensitive_features_binary,
            inner_metric=custom_sparsity,
        )
        assert "disparity" in result
        # Group 1 has more non-zero entries, so there should be a gap
        assert result["disparity"] > 0.0

    def test_multigroup_sensitive_features(
        self, fair_attributions, sensitive_features_multigroup
    ):
        """
        Should work with 3+ groups, not just binary.
        """
        from explainiverse.evaluation.fairness import compute_group_fairness

        result = compute_group_fairness(
            attributions=fair_attributions,
            sensitive_features=sensitive_features_multigroup,
        )
        assert "disparity" in result
        # Fair attributions: disparity should be ~0
        assert result["disparity"] == pytest.approx(0.0, abs=1e-10)
        assert len(result["group_means"]) == 3

    def test_multigroup_unfair(self, sensitive_features_multigroup):
        """
        Multi-group scenario with intentional disparity.
        """
        from explainiverse.evaluation.fairness import compute_group_fairness

        n = len(sensitive_features_multigroup)
        attrs = np.zeros((n, 4))
        for i in range(n):
            g = sensitive_features_multigroup[i]
            if g == 0:
                attrs[i] = [0.9, 0.3, 0.05, 0.05]
            elif g == 1:
                attrs[i] = [0.3, 0.3, 0.2, 0.2]
            else:
                attrs[i] = [0.05, 0.05, 0.05, 0.15]
        result = compute_group_fairness(
            attributions=attrs,
            sensitive_features=sensitive_features_multigroup,
        )
        assert result["disparity"] > 0.0

    def test_single_group_returns_zero_disparity(self):
        """
        If all instances belong to the same group, disparity is 0.
        """
        from explainiverse.evaluation.fairness import compute_group_fairness

        attrs = np.random.randn(10, 4)
        sf = np.zeros(10, dtype=int)
        result = compute_group_fairness(attributions=attrs, sensitive_features=sf)
        assert result["disparity"] == pytest.approx(0.0, abs=1e-10)

    def test_invalid_attributions_type_raises(self):
        """Non-array attributions should raise TypeError."""
        from explainiverse.evaluation.fairness import compute_group_fairness

        with pytest.raises((TypeError, ValueError)):
            compute_group_fairness(
                attributions="not_an_array",
                sensitive_features=np.array([0, 1]),
            )

    def test_mismatched_lengths_raises(self):
        """Attributions and sensitive_features must have same number of rows."""
        from explainiverse.evaluation.fairness import compute_group_fairness

        with pytest.raises(ValueError, match="length|shape|mismatch"):
            compute_group_fairness(
                attributions=np.ones((10, 4)),
                sensitive_features=np.array([0, 1, 0]),
            )

    def test_1d_attributions_raises(self):
        """1D attributions should raise ValueError (need 2D: n_samples x n_features)."""
        from explainiverse.evaluation.fairness import compute_group_fairness

        with pytest.raises(ValueError):
            compute_group_fairness(
                attributions=np.ones(10),
                sensitive_features=np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            )

    def test_empty_attributions_raises(self):
        """Empty arrays should raise ValueError."""
        from explainiverse.evaluation.fairness import compute_group_fairness

        with pytest.raises(ValueError):
            compute_group_fairness(
                attributions=np.array([]).reshape(0, 4),
                sensitive_features=np.array([]),
            )

    def test_semantic_validation_unfair_detected(self):
        """
        Semantic validation: deliberately biased attributions should have
        higher disparity than fair ones.
        """
        from explainiverse.evaluation.fairness import compute_group_fairness

        np.random.seed(42)
        n = 50
        sf = np.array([0] * 25 + [1] * 25)

        # Fair: identical distributions
        fair_attrs = np.random.randn(n, 4)
        # Unfair: group 1 gets all-zero attributions
        unfair_attrs = fair_attrs.copy()
        unfair_attrs[25:] = 0.0

        fair_result = compute_group_fairness(fair_attrs, sf)
        unfair_result = compute_group_fairness(unfair_attrs, sf)

        assert unfair_result["disparity"] > fair_result["disparity"]


# =============================================================================
# 3. Group Fairness — Score-based API (with explainer)
# =============================================================================

class TestGroupFairnessScore:
    """Tests for compute_group_fairness_score() — explainer-based API."""

    def test_import(self):
        """compute_group_fairness_score can be imported."""
        from explainiverse.evaluation.fairness import compute_group_fairness_score
        assert callable(compute_group_fairness_score)

    def test_with_real_model(self, trained_model_and_explainer):
        """Score-based API works with a real model and LIME explainer."""
        from explainiverse.evaluation.fairness import compute_group_fairness_score

        adapter, explainer, X, y = trained_model_and_explainer
        sensitive_features = X[:20, 2].astype(int)

        result = compute_group_fairness_score(
            explainer=explainer,
            inputs=X[:20],
            sensitive_features=sensitive_features,
        )
        assert isinstance(result, dict)
        assert "disparity" in result
        assert isinstance(result["disparity"], float)


# =============================================================================
# 4. Batch Group Fairness
# =============================================================================

class TestBatchGroupFairness:
    """Tests for compute_batch_group_fairness()."""

    def test_import(self):
        """compute_batch_group_fairness can be imported."""
        from explainiverse.evaluation.fairness import compute_batch_group_fairness
        assert callable(compute_batch_group_fairness)

    def test_batch_returns_list(
        self, unfair_attributions, sensitive_features_binary
    ):
        """Batch computation returns a list of result dicts."""
        from explainiverse.evaluation.fairness import compute_batch_group_fairness

        # Simulate 3 batches (different explainer outputs for same data)
        batch_attributions = [
            unfair_attributions,
            unfair_attributions * 0.5,
            unfair_attributions * 2.0,
        ]
        batch_sensitive = [sensitive_features_binary] * 3
        results = compute_batch_group_fairness(
            batch_attributions=batch_attributions,
            batch_sensitive_features=batch_sensitive,
        )
        assert isinstance(results, list)
        assert len(results) == 3
        for r in results:
            assert "disparity" in r


# =============================================================================
# 5. Individual Fairness Tests
# =============================================================================

class TestIndividualFairness:
    """Tests for individual fairness metric (Dwork et al., 2012; adapted)."""

    def test_import(self):
        """compute_individual_fairness can be imported."""
        from explainiverse.evaluation.fairness import compute_individual_fairness
        assert callable(compute_individual_fairness)

    def test_identical_attributions_across_groups(self):
        """
        If all instances get identical attributions, individual fairness
        should be perfect (score near 0).
        """
        from explainiverse.evaluation.fairness import compute_individual_fairness

        np.random.seed(42)
        n = 20
        inputs = np.random.randn(n, 4)
        # All same attributions
        attrs = np.tile([0.5, 0.3, 0.1, 0.1], (n, 1))
        sf = np.array([0] * 10 + [1] * 10)

        result = compute_individual_fairness(
            inputs=inputs,
            attributions=attrs,
            sensitive_features=sf,
        )
        assert isinstance(result, dict)
        assert "score" in result
        assert result["score"] == pytest.approx(0.0, abs=1e-10)

    def test_disparate_attributions_detected(self):
        """
        Similar individuals from different groups receiving different
        explanations should yield a high score.
        """
        from explainiverse.evaluation.fairness import compute_individual_fairness

        np.random.seed(42)
        n = 20
        # All instances are similar (small feature distance)
        inputs = np.ones((n, 4)) + np.random.randn(n, 4) * 0.01
        sf = np.array([0] * 10 + [1] * 10)
        attrs = np.zeros((n, 4))
        attrs[:10] = [0.9, 0.05, 0.025, 0.025]  # group 0
        attrs[10:] = [0.1, 0.4, 0.3, 0.2]        # group 1

        result = compute_individual_fairness(
            inputs=inputs,
            attributions=attrs,
            sensitive_features=sf,
        )
        assert result["score"] > 0.0

    def test_mismatched_shapes_raises(self):
        """inputs and attributions must have same shape."""
        from explainiverse.evaluation.fairness import compute_individual_fairness

        with pytest.raises(ValueError):
            compute_individual_fairness(
                inputs=np.ones((10, 4)),
                attributions=np.ones((5, 4)),
                sensitive_features=np.array([0] * 10),
            )


# =============================================================================
# 6. Counterfactual Explanation Fairness Tests
# =============================================================================

class TestCounterfactualFairness:
    """Tests for counterfactual explanation fairness (Kusner et al., 2017; adapted)."""

    def test_import(self):
        """compute_counterfactual_fairness can be imported."""
        from explainiverse.evaluation.fairness import compute_counterfactual_fairness
        assert callable(compute_counterfactual_fairness)

    def test_insensitive_attributions_score_zero(self):
        """
        If attributions don't change when the sensitive feature is flipped,
        counterfactual fairness should be 0.
        """
        from explainiverse.evaluation.fairness import compute_counterfactual_fairness

        np.random.seed(42)
        n = 10
        inputs = np.random.randn(n, 5)
        inputs[:, 2] = np.random.choice([0, 1], size=n)  # sensitive feature

        # Attributions that are identical regardless of input
        attrs = np.tile([0.3, 0.5, 0.0, 0.15, 0.05], (n, 1))

        result = compute_counterfactual_fairness(
            inputs=inputs,
            attributions=attrs,
            sensitive_feature_idx=2,
        )
        assert isinstance(result, dict)
        assert "score" in result
        # Score should be zero since attributions don't depend on sensitive feature
        assert result["score"] == pytest.approx(0.0, abs=1e-10)

    def test_sensitive_attributions_detected(self):
        """
        If attributions change based on sensitive feature value, score > 0.
        """
        from explainiverse.evaluation.fairness import compute_counterfactual_fairness

        np.random.seed(42)
        n = 10
        inputs = np.random.randn(n, 5)
        inputs[:, 2] = np.random.choice([0, 1], size=n)

        # Attributions that depend on the sensitive feature
        attrs = np.zeros((n, 5))
        for i in range(n):
            if inputs[i, 2] == 0:
                attrs[i] = [0.5, 0.3, 0.0, 0.1, 0.1]
            else:
                attrs[i] = [0.1, 0.1, 0.5, 0.2, 0.1]

        # Also need a way to compute counterfactual attributions
        def counterfactual_explainer(instance):
            """Returns attributions for the given instance (already flipped)."""
            if instance[2] == 0:
                return np.array([0.5, 0.3, 0.0, 0.1, 0.1])
            else:
                return np.array([0.1, 0.1, 0.5, 0.2, 0.1])
                return np.array([0.1, 0.1, 0.5, 0.2, 0.1])

        result = compute_counterfactual_fairness(
            inputs=inputs,
            attributions=attrs,
            sensitive_feature_idx=2,
            counterfactual_explainer=counterfactual_explainer,
        )
        assert result["score"] > 0.0

    def test_invalid_sensitive_feature_idx_raises(self):
        """Out-of-bounds sensitive_feature_idx should raise ValueError."""
        from explainiverse.evaluation.fairness import compute_counterfactual_fairness

        with pytest.raises((ValueError, IndexError)):
            compute_counterfactual_fairness(
                inputs=np.ones((5, 3)),
                attributions=np.ones((5, 3)),
                sensitive_feature_idx=10,
            )


# =============================================================================
# 7. Fidelity Disparity Tests
# =============================================================================

class TestFidelityDisparity:
    """Tests for fidelity disparity metric (Balagopalan et al., 2022)."""

    def test_import(self):
        """compute_fidelity_disparity can be imported."""
        from explainiverse.evaluation.fairness import compute_fidelity_disparity
        assert callable(compute_fidelity_disparity)

    def test_equal_groups_zero_disparity(self):
        """Identical quality across groups -> zero max_gap and mean_gap."""
        from explainiverse.evaluation.fairness import compute_fidelity_disparity

        attrs = np.tile([0.3, 0.5, 0.1, 0.1], (20, 1))
        sf = np.array([0] * 10 + [1] * 10)

        result = compute_fidelity_disparity(
            attributions=attrs,
            sensitive_features=sf,
        )
        assert isinstance(result, dict)
        assert "max_gap" in result
        assert "mean_gap" in result
        assert result["max_gap"] == pytest.approx(0.0, abs=1e-10)
        assert result["mean_gap"] == pytest.approx(0.0, abs=1e-10)

    def test_disparate_groups_positive_gap(self):
        """Different quality across groups -> positive gaps."""
        from explainiverse.evaluation.fairness import compute_fidelity_disparity

        n = 20
        attrs = np.zeros((n, 4))
        attrs[:10] = [0.9, 0.3, 0.05, 0.05]
        attrs[10:] = [0.2, 0.1, 0.1, 0.1]
        sf = np.array([0] * 10 + [1] * 10)

        result = compute_fidelity_disparity(
            attributions=attrs,
            sensitive_features=sf,
        )
        assert result["max_gap"] > 0.0
        assert result["mean_gap"] > 0.0

    def test_multigroup_worst_case_pair(self, sensitive_features_multigroup):
        """
        With 3 groups, max_gap should reflect the worst-case pair.
        """
        from explainiverse.evaluation.fairness import compute_fidelity_disparity

        n = len(sensitive_features_multigroup)
        attrs = np.zeros((n, 4))
        for i in range(n):
            g = sensitive_features_multigroup[i]
            if g == 0:
                attrs[i] = [0.9, 0.05, 0.025, 0.025]  # very sparse
            elif g == 1:
                attrs[i] = [0.5, 0.3, 0.1, 0.1]         # moderate
            else:
                attrs[i] = [0.25, 0.25, 0.25, 0.25]     # uniform

        result = compute_fidelity_disparity(
            attributions=attrs,
            sensitive_features=sensitive_features_multigroup,
        )
        # max_gap should be the gap between group 0 (sparsest) and group 2 (most uniform)
        assert result["max_gap"] >= result["mean_gap"]
        assert "pairwise_gaps" in result
        # With 3 groups, there should be 3 pairs: (0,1), (0,2), (1,2)
        assert len(result["pairwise_gaps"]) == 3

    def test_custom_inner_metric_with_fidelity(self):
        """Fidelity disparity accepts a user-supplied quality metric."""
        from explainiverse.evaluation.fairness import compute_fidelity_disparity

        def entropy_metric(attr_vector):
            """Explanation entropy (higher = more complex)."""
            p = np.abs(attr_vector)
            total = p.sum()
            if total < 1e-10:
                return 0.0
            p = p / total
            p = p[p > 0]
            return float(-np.sum(p * np.log2(p)))

        attrs = np.zeros((20, 4))
        attrs[:10] = [0.9, 0.05, 0.025, 0.025]  # low entropy
        attrs[10:] = [0.25, 0.25, 0.25, 0.25]   # high entropy
        sf = np.array([0] * 10 + [1] * 10)

        result = compute_fidelity_disparity(
            attributions=attrs,
            sensitive_features=sf,
            inner_metric=entropy_metric,
        )
        assert result["max_gap"] > 0.0


# =============================================================================
# 8. Attribution Parity Tests
# =============================================================================

class TestAttributionParity:
    """Tests for Attribution Parity (novel synthesis)."""

    def test_import(self):
        """compute_attribution_parity can be imported."""
        from explainiverse.evaluation.fairness import compute_attribution_parity
        assert callable(compute_attribution_parity)

    def test_no_sensitive_attribution_is_fair(self):
        """
        If the sensitive feature gets zero attribution in all groups,
        attribution parity should be high (low divergence).
        """
        from explainiverse.evaluation.fairness import compute_attribution_parity

        n = 20
        attrs = np.zeros((n, 5))
        # Sensitive feature (idx 2) gets zero attribution for everyone
        attrs[:10] = [0.4, 0.5, 0.0, 0.05, 0.05]
        attrs[10:] = [0.3, 0.6, 0.0, 0.05, 0.05]
        sf = np.array([0] * 10 + [1] * 10)

        result = compute_attribution_parity(
            attributions=attrs,
            sensitive_features=sf,
            sensitive_feature_idx=2,
        )
        assert isinstance(result, dict)
        assert "divergence" in result
        assert result["divergence"] == pytest.approx(0.0, abs=1e-10)

    def test_disparate_sensitive_attribution_detected(self):
        """
        If the sensitive feature gets different attribution by group,
        divergence should be positive.
        """
        from explainiverse.evaluation.fairness import compute_attribution_parity

        n = 20
        attrs = np.zeros((n, 5))
        # Group 0: sensitive feature gets 0 attribution
        attrs[:10] = [0.5, 0.4, 0.0, 0.05, 0.05]
        # Group 1: sensitive feature gets high attribution
        attrs[10:] = [0.2, 0.2, 0.5, 0.05, 0.05]
        sf = np.array([0] * 10 + [1] * 10)

        result = compute_attribution_parity(
            attributions=attrs,
            sensitive_features=sf,
            sensitive_feature_idx=2,
        )
        assert result["divergence"] > 0.0

    def test_returns_per_group_stats(self):
        """Result should include per-group mean attribution of the sensitive feature."""
        from explainiverse.evaluation.fairness import compute_attribution_parity

        n = 20
        attrs = np.random.randn(n, 5)
        sf = np.array([0] * 10 + [1] * 10)

        result = compute_attribution_parity(
            attributions=attrs,
            sensitive_features=sf,
            sensitive_feature_idx=2,
        )
        assert "group_sensitive_means" in result
        assert len(result["group_sensitive_means"]) == 2


# =============================================================================
# 9. Conditional Fairness Tests
# =============================================================================

class TestConditionalFairness:
    """Tests for Conditional Fairness / Equalized Explanation Quality (Hardt et al., 2016; adapted)."""

    def test_import(self):
        """compute_conditional_fairness can be imported."""
        from explainiverse.evaluation.fairness import compute_conditional_fairness
        assert callable(compute_conditional_fairness)

    def test_equal_quality_conditioned_on_prediction(self):
        """
        If explanation quality is equal within each prediction class,
        conditional disparity should be 0.
        """
        from explainiverse.evaluation.fairness import compute_conditional_fairness

        n = 20
        attrs = np.tile([0.3, 0.4, 0.2, 0.1], (n, 1))
        sf = np.array([0] * 10 + [1] * 10)
        predictions = np.array([0] * 10 + [1] * 10)

        result = compute_conditional_fairness(
            attributions=attrs,
            sensitive_features=sf,
            predictions=predictions,
        )
        assert isinstance(result, dict)
        assert "disparity" in result
        assert result["disparity"] == pytest.approx(0.0, abs=1e-10)

    def test_disparity_conditioned_on_prediction(self):
        """
        If within a prediction class, one group gets worse explanations,
        conditional disparity should be positive.
        """
        from explainiverse.evaluation.fairness import compute_conditional_fairness

        n = 40
        sf = np.array([0] * 20 + [1] * 20)
        # Both groups get predicted class 1 equally
        predictions = np.tile([0, 1], 20)

        attrs = np.zeros((n, 4))
        for i in range(n):
            if sf[i] == 0:
                attrs[i] = [0.9, 0.4, 0.05, 0.05]  # good quality for both preds
            else:
                attrs[i] = [0.1, 0.1, 0.1, 0.1]  # poor quality for both preds

        result = compute_conditional_fairness(
            attributions=attrs,
            sensitive_features=sf,
            predictions=predictions,
        )
        assert result["disparity"] > 0.0

    def test_returns_per_class_disparity(self):
        """Result should include disparity broken down by prediction class."""
        from explainiverse.evaluation.fairness import compute_conditional_fairness

        n = 20
        attrs = np.random.randn(n, 4)
        sf = np.array([0] * 10 + [1] * 10)
        predictions = np.array([0] * 5 + [1] * 5 + [0] * 5 + [1] * 5)

        result = compute_conditional_fairness(
            attributions=attrs,
            sensitive_features=sf,
            predictions=predictions,
        )
        assert "per_class_disparity" in result
        assert isinstance(result["per_class_disparity"], dict)

    def test_predictions_mismatch_raises(self):
        """predictions must match length of attributions."""
        from explainiverse.evaluation.fairness import compute_conditional_fairness

        with pytest.raises(ValueError):
            compute_conditional_fairness(
                attributions=np.ones((10, 4)),
                sensitive_features=np.zeros(10, dtype=int),
                predictions=np.zeros(5, dtype=int),
            )


# =============================================================================
# 10. Top-level API: compute_X / compute_X_score / compute_batch_X
# =============================================================================

class TestThreeTierAPI:
    """Verify the 3-tier API pattern is consistent across all fairness metrics."""

    def test_all_metrics_have_compute_function(self):
        """Each metric has a compute_<name>() function."""
        from explainiverse.evaluation import fairness
        expected = [
            "compute_group_fairness",
            "compute_individual_fairness",
            "compute_counterfactual_fairness",
            "compute_fidelity_disparity",
            "compute_attribution_parity",
            "compute_conditional_fairness",
        ]
        for fn_name in expected:
            assert hasattr(fairness, fn_name), f"Missing: {fn_name}"
            assert callable(getattr(fairness, fn_name))

    def test_group_fairness_has_score_variant(self):
        """Group fairness has an explainer-based score variant."""
        from explainiverse.evaluation.fairness import compute_group_fairness_score
        assert callable(compute_group_fairness_score)

    def test_group_fairness_has_batch_variant(self):
        """Group fairness has a batch variant."""
        from explainiverse.evaluation.fairness import compute_batch_group_fairness
        assert callable(compute_batch_group_fairness)


# =============================================================================
# 11. Integration with evaluation/__init__.py exports
# =============================================================================

class TestEvaluationExports:
    """Fairness metrics are properly exported from evaluation package."""

    def test_group_fairness_importable_from_evaluation(self):
        """compute_group_fairness is importable from evaluation package."""
        from explainiverse.evaluation import compute_group_fairness
        assert callable(compute_group_fairness)

    def test_individual_fairness_importable_from_evaluation(self):
        from explainiverse.evaluation import compute_individual_fairness
        assert callable(compute_individual_fairness)

    def test_counterfactual_fairness_importable_from_evaluation(self):
        from explainiverse.evaluation import compute_counterfactual_fairness
        assert callable(compute_counterfactual_fairness)

    def test_fidelity_disparity_importable_from_evaluation(self):
        from explainiverse.evaluation import compute_fidelity_disparity
        assert callable(compute_fidelity_disparity)

    def test_attribution_parity_importable_from_evaluation(self):
        from explainiverse.evaluation import compute_attribution_parity
        assert callable(compute_attribution_parity)

    def test_conditional_fairness_importable_from_evaluation(self):
        from explainiverse.evaluation import compute_conditional_fairness
        assert callable(compute_conditional_fairness)

    def test_registry_importable_from_evaluation(self):
        from explainiverse.evaluation import get_default_fairness_registry
        assert callable(get_default_fairness_registry)

    def test_batch_group_fairness_importable(self):
        from explainiverse.evaluation import compute_batch_group_fairness
        assert callable(compute_batch_group_fairness)

    def test_group_fairness_score_importable(self):
        from explainiverse.evaluation import compute_group_fairness_score
        assert callable(compute_group_fairness_score)


# =============================================================================
# 12. Edge Cases & Robustness
# =============================================================================

class TestEdgeCases:
    """Edge cases and robustness tests across all fairness metrics."""

    def test_group_fairness_with_single_sample_per_group(self):
        """Should handle groups with only 1 sample each."""
        from explainiverse.evaluation.fairness import compute_group_fairness

        attrs = np.array([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]])
        sf = np.array([0, 1])
        result = compute_group_fairness(attributions=attrs, sensitive_features=sf)
        assert "disparity" in result
        assert isinstance(result["disparity"], float)

    def test_group_fairness_with_all_zero_attributions(self):
        """All-zero attributions should produce 0 disparity."""
        from explainiverse.evaluation.fairness import compute_group_fairness

        attrs = np.zeros((10, 4))
        sf = np.array([0] * 5 + [1] * 5)
        result = compute_group_fairness(attributions=attrs, sensitive_features=sf)
        assert result["disparity"] == pytest.approx(0.0, abs=1e-10)

    def test_group_fairness_with_nan_handling(self):
        """NaN in attributions should be handled gracefully."""
        from explainiverse.evaluation.fairness import compute_group_fairness

        attrs = np.ones((10, 4))
        attrs[0, 0] = np.nan
        sf = np.array([0] * 5 + [1] * 5)
        # Should either handle NaN or raise a clear error
        try:
            result = compute_group_fairness(attributions=attrs, sensitive_features=sf)
            assert isinstance(result["disparity"], float)
        except ValueError:
            pass  # Acceptable to raise ValueError for NaN

    def test_large_number_of_groups(self):
        """Should handle many groups (e.g., 10)."""
        from explainiverse.evaluation.fairness import compute_group_fairness

        np.random.seed(42)
        n = 100
        attrs = np.random.randn(n, 5)
        sf = np.arange(n) % 10  # 10 groups
        result = compute_group_fairness(attributions=attrs, sensitive_features=sf)
        assert "disparity" in result
        assert len(result["group_means"]) == 10

    def test_fidelity_disparity_single_group(self):
        """Single group should produce zero gaps."""
        from explainiverse.evaluation.fairness import compute_fidelity_disparity

        attrs = np.random.randn(10, 4)
        sf = np.zeros(10, dtype=int)
        result = compute_fidelity_disparity(attributions=attrs, sensitive_features=sf)
        assert result["max_gap"] == pytest.approx(0.0, abs=1e-10)

    def test_attribution_parity_all_groups_same_sensitive_attr(self):
        """If sensitive feature gets same attribution in all groups, divergence=0."""
        from explainiverse.evaluation.fairness import compute_attribution_parity

        n = 20
        attrs = np.zeros((n, 5))
        attrs[:, 0] = 0.5
        attrs[:, 1] = 0.3
        attrs[:, 2] = 0.1  # same for all
        attrs[:, 3] = 0.05
        attrs[:, 4] = 0.05
        sf = np.array([0] * 10 + [1] * 10)

        result = compute_attribution_parity(
            attributions=attrs,
            sensitive_features=sf,
            sensitive_feature_idx=2,
        )
        assert result["divergence"] == pytest.approx(0.0, abs=1e-10)

    def test_conditional_fairness_single_prediction_class(self):
        """If all predictions are the same class, should still work."""
        from explainiverse.evaluation.fairness import compute_conditional_fairness

        attrs = np.random.randn(10, 4)
        sf = np.array([0] * 5 + [1] * 5)
        preds = np.ones(10, dtype=int)  # all predict class 1

        result = compute_conditional_fairness(
            attributions=attrs,
            sensitive_features=sf,
            predictions=preds,
        )
        assert "disparity" in result

    def test_float_sensitive_features_cast_to_int(self):
        """Float sensitive features should be silently cast to int groups."""
        from explainiverse.evaluation.fairness import compute_group_fairness

        attrs = np.random.randn(10, 4)
        sf = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        result = compute_group_fairness(attributions=attrs, sensitive_features=sf)
        assert "disparity" in result

    def test_string_sensitive_features_work(self):
        """String sensitive features (e.g., 'male'/'female') should work."""
        from explainiverse.evaluation.fairness import compute_group_fairness

        attrs = np.random.randn(10, 4)
        sf = np.array(["male"] * 5 + ["female"] * 5)
        result = compute_group_fairness(attributions=attrs, sensitive_features=sf)
        assert "disparity" in result
        assert len(result["group_means"]) == 2
