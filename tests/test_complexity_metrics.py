# tests/test_complexity_metrics.py
"""
Tests for Phase 4 complexity evaluation metrics.

- Sparseness (Chalasani et al., 2020) — Gini Index of absolute attributions
- Complexity (Bhatt et al., 2020) — Shannon entropy of fractional contributions
- Effective Complexity (Nguyen & Martínez, 2020) — threshold-based feature count

References:
    Chalasani, P., Chen, J., Chowdhury, A. R., Wu, X., & Jha, S. (2020).
    Concise Explanations of Neural Networks using Adversarial Training.
    ICML.

    Bhatt, U., Weller, A., & Moura, J. M. F. (2020). Evaluating and
    Aggregating Feature-based Model Explanations. IJCAI.

    Nguyen, A. P., & Martínez, M. R. (2020). On Quantitative Aspects
    of Model Interpretability. arXiv:2007.07584.
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
def feature_names_8():
    return [f"f{i}" for i in range(8)]


@pytest.fixture
def single_instance():
    np.random.seed(42)
    return np.random.randn(4).astype(np.float32)


@pytest.fixture
def sample_data():
    """Deterministic sample data."""
    np.random.seed(42)
    return np.random.randn(20, 4).astype(np.float32)


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
# Mock Explainers for Controlled Testing
# =============================================================================

class _SparseExplainer:
    """
    Mock explainer that concentrates all attribution on a single feature.
    Should yield high sparseness (Gini ≈ (n-1)/n).
    """
    def __init__(self, feature_names, dominant_index=0):
        self.feature_names = feature_names
        self.dominant_index = dominant_index

    def explain(self, instance):
        attrs = {fn: 0.0 for fn in self.feature_names}
        attrs[self.feature_names[self.dominant_index]] = 1.0
        exp = Explanation(
            explainer_name="sparse",
            target_class="class_0",
            explanation_data={"feature_attributions": attrs},
        )
        exp.feature_names = self.feature_names
        return exp


class _UniformExplainer:
    """
    Mock explainer that distributes attribution equally across all features.
    Should yield sparseness = 0.0 (Gini = 0).
    """
    def __init__(self, feature_names, value=1.0):
        self.feature_names = feature_names
        self.value = value

    def explain(self, instance):
        attrs = {fn: self.value for fn in self.feature_names}
        exp = Explanation(
            explainer_name="uniform",
            target_class="class_0",
            explanation_data={"feature_attributions": attrs},
        )
        exp.feature_names = self.feature_names
        return exp


class _GradientProportionalExplainer:
    """
    Mock explainer that returns attributions proportional to the absolute
    input values. Sparseness depends on the input distribution.
    """
    def __init__(self, feature_names, scale=1.0):
        self.feature_names = feature_names
        self.scale = scale

    def explain(self, instance):
        instance = np.asarray(instance).flatten()
        attrs = {fn: float(self.scale * abs(instance[i]))
                 for i, fn in enumerate(self.feature_names)}
        exp = Explanation(
            explainer_name="gradient_proportional",
            target_class="class_0",
            explanation_data={"feature_attributions": attrs},
        )
        exp.feature_names = self.feature_names
        return exp


class _ZeroExplainer:
    """
    Mock explainer that returns all-zero attributions.
    Edge case — degenerate explanation.
    """
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def explain(self, instance):
        attrs = {fn: 0.0 for fn in self.feature_names}
        exp = Explanation(
            explainer_name="zero",
            target_class="class_0",
            explanation_data={"feature_attributions": attrs},
        )
        exp.feature_names = self.feature_names
        return exp


class _TwoFeatureExplainer:
    """
    Mock explainer that places attribution on exactly 2 features.
    Intermediate sparseness.
    """
    def __init__(self, feature_names, weight_a=0.7, weight_b=0.3):
        self.feature_names = feature_names
        self.weight_a = weight_a
        self.weight_b = weight_b

    def explain(self, instance):
        attrs = {fn: 0.0 for fn in self.feature_names}
        attrs[self.feature_names[0]] = self.weight_a
        attrs[self.feature_names[1]] = self.weight_b
        exp = Explanation(
            explainer_name="two_feature",
            target_class="class_0",
            explanation_data={"feature_attributions": attrs},
        )
        exp.feature_names = self.feature_names
        return exp


# =============================================================================
# Sparseness Tests — compute_sparseness()
# =============================================================================

class TestComputeSparseness:
    """Tests for compute_sparseness (Chalasani et al., 2020)."""

    def test_import(self):
        """Sparseness function is importable from evaluation module."""
        from explainiverse.evaluation import compute_sparseness
        assert callable(compute_sparseness)

    def test_returns_float(self, feature_names, single_instance):
        """compute_sparseness returns a float."""
        from explainiverse.evaluation import compute_sparseness
        explainer = _UniformExplainer(feature_names)
        result = compute_sparseness(explainer, single_instance)
        assert isinstance(result, float)

    def test_uniform_attribution_is_zero(self, feature_names, single_instance):
        """Perfectly uniform attributions should have Gini = 0.0."""
        from explainiverse.evaluation import compute_sparseness
        explainer = _UniformExplainer(feature_names, value=1.0)
        result = compute_sparseness(explainer, single_instance)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_perfectly_sparse_attribution(self, feature_names, single_instance):
        """
        All weight on one feature: Gini = (n-1)/n.
        For n=4, expected = 0.75.
        """
        from explainiverse.evaluation import compute_sparseness
        explainer = _SparseExplainer(feature_names, dominant_index=0)
        result = compute_sparseness(explainer, single_instance)
        n = len(feature_names)
        expected = (n - 1) / n  # 0.75
        assert result == pytest.approx(expected, abs=1e-10)

    def test_perfectly_sparse_8_features(self, feature_names_8, single_instance):
        """
        All weight on one of 8 features: Gini = 7/8 = 0.875.
        Verifies formula scales correctly with n.
        """
        from explainiverse.evaluation import compute_sparseness
        instance_8 = np.random.default_rng(42).standard_normal(8).astype(np.float32)
        explainer = _SparseExplainer(feature_names_8, dominant_index=3)
        result = compute_sparseness(explainer, instance_8)
        expected = 7.0 / 8.0
        assert result == pytest.approx(expected, abs=1e-10)

    def test_range_zero_to_one(self, feature_names, single_instance):
        """Sparseness should always be in [0, 1]."""
        from explainiverse.evaluation import compute_sparseness
        for Explainer in [_SparseExplainer, _UniformExplainer,
                          _GradientProportionalExplainer, _TwoFeatureExplainer]:
            explainer = Explainer(feature_names)
            result = compute_sparseness(explainer, single_instance)
            assert 0.0 <= result <= 1.0, (
                f"{Explainer.__name__} gave sparseness {result} outside [0, 1]"
            )

    def test_zero_attributions_returns_zero(self, feature_names, single_instance):
        """All-zero attributions: degenerate case, should return 0.0."""
        from explainiverse.evaluation import compute_sparseness
        explainer = _ZeroExplainer(feature_names)
        result = compute_sparseness(explainer, single_instance)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_negative_attributions_uses_absolute(self, feature_names, single_instance):
        """
        Sparseness uses |a_i|. Negative attributions should be treated
        identically to positive ones of the same magnitude.
        """
        from explainiverse.evaluation import compute_sparseness

        class _NegativeSparseExplainer:
            def __init__(self, fnames):
                self.feature_names = fnames

            def explain(self, instance):
                attrs = {fn: 0.0 for fn in self.feature_names}
                attrs[self.feature_names[0]] = -1.0
                exp = Explanation(
                    explainer_name="neg_sparse",
                    target_class="class_0",
                    explanation_data={"feature_attributions": attrs},
                )
                exp.feature_names = self.feature_names
                return exp

        positive = _SparseExplainer(feature_names, dominant_index=0)
        negative = _NegativeSparseExplainer(feature_names)

        score_pos = compute_sparseness(positive, single_instance)
        score_neg = compute_sparseness(negative, single_instance)
        assert score_pos == pytest.approx(score_neg, abs=1e-10)

    def test_two_feature_intermediate_sparseness(self, feature_names, single_instance):
        """
        Two non-zero features out of 4 should give intermediate Gini.
        For [0.7, 0.3, 0.0, 0.0] sorted ascending = [0, 0, 0.3, 0.7]:
        G = (2*sum((i+1)*a_sorted[i])) / (n*sum_a) - (n+1)/n
          = 2*(1*0 + 2*0 + 3*0.3 + 4*0.7) / (4*1.0) - 5/4
          = 2*(0.9 + 2.8) / 4 - 1.25
          = 2*3.7/4 - 1.25
          = 1.85 - 1.25
          = 0.60
        """
        from explainiverse.evaluation import compute_sparseness
        explainer = _TwoFeatureExplainer(feature_names, weight_a=0.7, weight_b=0.3)
        result = compute_sparseness(explainer, single_instance)
        assert result == pytest.approx(0.6, abs=1e-10)

    def test_sparse_greater_than_uniform(self, feature_names, single_instance):
        """Sparse explanation should have higher sparseness than uniform."""
        from explainiverse.evaluation import compute_sparseness
        sparse_exp = _SparseExplainer(feature_names)
        uniform_exp = _UniformExplainer(feature_names)

        sparse_score = compute_sparseness(sparse_exp, single_instance)
        uniform_score = compute_sparseness(uniform_exp, single_instance)
        assert sparse_score > uniform_score

    def test_ordering_sparse_gt_partial_gt_uniform(
        self, feature_names, single_instance
    ):
        """
        Semantic ordering: perfectly sparse > two-feature > uniform.
        """
        from explainiverse.evaluation import compute_sparseness
        sparse_score = compute_sparseness(
            _SparseExplainer(feature_names), single_instance
        )
        partial_score = compute_sparseness(
            _TwoFeatureExplainer(feature_names), single_instance
        )
        uniform_score = compute_sparseness(
            _UniformExplainer(feature_names), single_instance
        )
        assert sparse_score > partial_score > uniform_score

    def test_known_gini_three_values(self):
        """
        Verify against hand-computed Gini for [1, 2, 3].
        sorted ascending = [1, 2, 3], sum = 6, n = 3
        G = 2*(1*1 + 2*2 + 3*3) / (3*6) - (3+1)/3
          = 2*(1+4+9)/18 - 4/3
          = 2*14/18 - 4/3
          = 28/18 - 4/3
          = 28/18 - 24/18
          = 4/18 = 2/9 ≈ 0.2222
        """
        from explainiverse.evaluation import compute_sparseness
        fnames = ["a", "b", "c"]

        class _FixedExplainer:
            def __init__(self):
                self.feature_names = fnames

            def explain(self, instance):
                exp = Explanation(
                    explainer_name="fixed",
                    target_class="class_0",
                    explanation_data={
                        "feature_attributions": {"a": 1.0, "b": 2.0, "c": 3.0}
                    },
                )
                exp.feature_names = fnames
                return exp

        result = compute_sparseness(_FixedExplainer(), np.array([0.0, 0.0, 0.0]))
        assert result == pytest.approx(2.0 / 9.0, abs=1e-10)

    def test_single_feature_returns_zero(self):
        """A single feature has no inequality — Gini = 0.0."""
        from explainiverse.evaluation import compute_sparseness
        fnames = ["f0"]

        class _SingleExplainer:
            def __init__(self):
                self.feature_names = fnames

            def explain(self, instance):
                exp = Explanation(
                    explainer_name="single",
                    target_class="class_0",
                    explanation_data={"feature_attributions": {"f0": 5.0}},
                )
                exp.feature_names = fnames
                return exp

        result = compute_sparseness(_SingleExplainer(), np.array([1.0]))
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_scale_invariance(self, feature_names, single_instance):
        """
        Gini index is scale-invariant: scaling all attributions by a
        constant c > 0 should not change the result.
        """
        from explainiverse.evaluation import compute_sparseness
        explainer_1x = _GradientProportionalExplainer(feature_names, scale=1.0)
        explainer_100x = _GradientProportionalExplainer(feature_names, scale=100.0)

        score_1x = compute_sparseness(explainer_1x, single_instance)
        score_100x = compute_sparseness(explainer_100x, single_instance)
        assert score_1x == pytest.approx(score_100x, abs=1e-8)

    def test_symmetric_to_feature_permutation(self, feature_names, single_instance):
        """
        Gini index is permutation-invariant: the order of features
        should not affect the result.
        """
        from explainiverse.evaluation import compute_sparseness
        explainer_f0 = _SparseExplainer(feature_names, dominant_index=0)
        explainer_f3 = _SparseExplainer(feature_names, dominant_index=3)

        score_f0 = compute_sparseness(explainer_f0, single_instance)
        score_f3 = compute_sparseness(explainer_f3, single_instance)
        assert score_f0 == pytest.approx(score_f3, abs=1e-10)

    def test_with_real_explainer(self, trained_model_and_explainer):
        """Sparseness works with a real LIME explainer and trained model."""
        from explainiverse.evaluation import compute_sparseness
        _, explainer, X = trained_model_and_explainer
        result = compute_sparseness(explainer, X[0])
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_near_zero_attributions_treated_as_zero(self, feature_names, single_instance):
        """
        Very small (near-machine-epsilon) attributions should not cause
        numerical instability.
        """
        from explainiverse.evaluation import compute_sparseness

        class _NearZeroExplainer:
            def __init__(self):
                self.feature_names = feature_names

            def explain(self, instance):
                attrs = {fn: 1e-300 for fn in feature_names}
                attrs[feature_names[0]] = 1.0
                exp = Explanation(
                    explainer_name="near_zero",
                    target_class="class_0",
                    explanation_data={"feature_attributions": attrs},
                )
                exp.feature_names = feature_names
                return exp

        result = compute_sparseness(_NearZeroExplainer(), single_instance)
        assert isinstance(result, float)
        assert not np.isnan(result)
        assert 0.0 <= result <= 1.0


# =============================================================================
# Sparseness Tests — compute_batch_sparseness()
# =============================================================================

class TestComputeBatchSparseness:
    """Tests for compute_batch_sparseness."""

    def test_import(self):
        """Batch function is importable."""
        from explainiverse.evaluation import compute_batch_sparseness
        assert callable(compute_batch_sparseness)

    def test_returns_dict_with_expected_keys(self, feature_names, sample_data):
        """Batch result contains all expected summary statistics."""
        from explainiverse.evaluation import compute_batch_sparseness
        explainer = _UniformExplainer(feature_names)
        result = compute_batch_sparseness(explainer, sample_data)
        assert isinstance(result, dict)
        for key in ["mean", "std", "max", "min", "scores", "n_evaluated"]:
            assert key in result, f"Missing key: {key}"

    def test_scores_list_length(self, feature_names, sample_data):
        """scores list has one entry per evaluated instance."""
        from explainiverse.evaluation import compute_batch_sparseness
        explainer = _UniformExplainer(feature_names)
        result = compute_batch_sparseness(explainer, sample_data)
        assert len(result["scores"]) == len(sample_data)
        assert result["n_evaluated"] == len(sample_data)

    def test_max_instances_limits_evaluation(self, feature_names, sample_data):
        """max_instances caps the number of instances evaluated."""
        from explainiverse.evaluation import compute_batch_sparseness
        explainer = _UniformExplainer(feature_names)
        result = compute_batch_sparseness(
            explainer, sample_data, max_instances=5
        )
        assert result["n_evaluated"] == 5
        assert len(result["scores"]) == 5

    def test_uniform_batch_all_zeros(self, feature_names, sample_data):
        """Uniform explainer should yield all zero sparseness scores."""
        from explainiverse.evaluation import compute_batch_sparseness
        explainer = _UniformExplainer(feature_names)
        result = compute_batch_sparseness(explainer, sample_data)
        assert result["mean"] == pytest.approx(0.0, abs=1e-10)
        assert all(s == pytest.approx(0.0, abs=1e-10) for s in result["scores"])

    def test_sparse_batch_all_high(self, feature_names, sample_data):
        """Sparse explainer should yield consistent high sparseness."""
        from explainiverse.evaluation import compute_batch_sparseness
        explainer = _SparseExplainer(feature_names)
        result = compute_batch_sparseness(explainer, sample_data)
        expected = (len(feature_names) - 1) / len(feature_names)
        assert result["mean"] == pytest.approx(expected, abs=1e-10)

    def test_mean_within_min_max(self, feature_names, sample_data):
        """Mean should always be between min and max."""
        from explainiverse.evaluation import compute_batch_sparseness
        explainer = _GradientProportionalExplainer(feature_names)
        result = compute_batch_sparseness(explainer, sample_data)
        assert result["min"] <= result["mean"] <= result["max"]

    def test_with_real_explainer(self, trained_model_and_explainer):
        """Batch sparseness works with a real LIME explainer."""
        from explainiverse.evaluation import compute_batch_sparseness
        _, explainer, X = trained_model_and_explainer
        result = compute_batch_sparseness(
            explainer, X[:10], max_instances=10
        )
        assert result["n_evaluated"] == 10
        assert all(0.0 <= s <= 1.0 for s in result["scores"])

    def test_std_is_zero_for_constant_explainer(self, feature_names, sample_data):
        """Constant explainer (uniform) → zero std across batch."""
        from explainiverse.evaluation import compute_batch_sparseness
        explainer = _UniformExplainer(feature_names)
        result = compute_batch_sparseness(explainer, sample_data)
        assert result["std"] == pytest.approx(0.0, abs=1e-10)


# =============================================================================
# Complexity Tests — compute_complexity()
# =============================================================================

class TestComputeComplexity:
    """Tests for compute_complexity (Bhatt et al., 2020)."""

    def test_import(self):
        """Complexity function is importable from evaluation module."""
        from explainiverse.evaluation import compute_complexity
        assert callable(compute_complexity)

    def test_returns_float(self, feature_names, single_instance):
        """compute_complexity returns a float."""
        from explainiverse.evaluation import compute_complexity
        explainer = _UniformExplainer(feature_names)
        result = compute_complexity(explainer, single_instance)
        assert isinstance(result, float)

    def test_uniform_attribution_is_log2_n(self, feature_names, single_instance):
        """
        Perfectly uniform attributions over n features:
        H = log2(n). For n=4, expected = 2.0.
        """
        from explainiverse.evaluation import compute_complexity
        explainer = _UniformExplainer(feature_names, value=1.0)
        result = compute_complexity(explainer, single_instance)
        expected = np.log2(len(feature_names))  # log2(4) = 2.0
        assert result == pytest.approx(expected, abs=1e-10)

    def test_uniform_8_features(self, feature_names_8):
        """
        Uniform over 8 features: H = log2(8) = 3.0.
        """
        from explainiverse.evaluation import compute_complexity
        instance_8 = np.ones(8, dtype=np.float32)
        explainer = _UniformExplainer(feature_names_8, value=1.0)
        result = compute_complexity(explainer, instance_8)
        assert result == pytest.approx(3.0, abs=1e-10)

    def test_perfectly_sparse_is_zero(self, feature_names, single_instance):
        """
        All weight on one feature: p = [1, 0, 0, 0].
        H = -1*log2(1) = 0.0.
        """
        from explainiverse.evaluation import compute_complexity
        explainer = _SparseExplainer(feature_names, dominant_index=0)
        result = compute_complexity(explainer, single_instance)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_range_zero_to_log2n(self, feature_names, single_instance):
        """Complexity should always be in [0, log2(n)]."""
        from explainiverse.evaluation import compute_complexity
        max_entropy = np.log2(len(feature_names))
        for Explainer in [_SparseExplainer, _UniformExplainer,
                          _GradientProportionalExplainer, _TwoFeatureExplainer]:
            explainer = Explainer(feature_names)
            result = compute_complexity(explainer, single_instance)
            assert 0.0 <= result <= max_entropy + 1e-10, (
                f"{Explainer.__name__} gave complexity {result} outside [0, {max_entropy}]"
            )

    def test_zero_attributions_returns_zero(self, feature_names, single_instance):
        """All-zero attributions: degenerate case, should return 0.0."""
        from explainiverse.evaluation import compute_complexity
        explainer = _ZeroExplainer(feature_names)
        result = compute_complexity(explainer, single_instance)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_negative_attributions_uses_absolute(self, feature_names, single_instance):
        """
        Complexity uses |a_i|. Negative attributions should be treated
        identically to positive ones of the same magnitude.
        """
        from explainiverse.evaluation import compute_complexity

        class _NegativeUniformExplainer:
            def __init__(self, fnames):
                self.feature_names = fnames

            def explain(self, instance):
                attrs = {fn: -1.0 for fn in self.feature_names}
                exp = Explanation(
                    explainer_name="neg_uniform",
                    target_class="class_0",
                    explanation_data={"feature_attributions": attrs},
                )
                exp.feature_names = self.feature_names
                return exp

        pos = _UniformExplainer(feature_names, value=1.0)
        neg = _NegativeUniformExplainer(feature_names)

        score_pos = compute_complexity(pos, single_instance)
        score_neg = compute_complexity(neg, single_instance)
        assert score_pos == pytest.approx(score_neg, abs=1e-10)

    def test_two_feature_known_entropy(self, feature_names, single_instance):
        """
        Two features [0.7, 0.3, 0.0, 0.0]: p = [0.7, 0.3]
        H = -(0.7*log2(0.7) + 0.3*log2(0.3))
          = -(0.7*(-0.51457) + 0.3*(-1.73697))
          = -(-.36020 + -.52109)
          = 0.88129 bits
        """
        from explainiverse.evaluation import compute_complexity
        explainer = _TwoFeatureExplainer(feature_names, weight_a=0.7, weight_b=0.3)
        result = compute_complexity(explainer, single_instance)
        expected = -(0.7 * np.log2(0.7) + 0.3 * np.log2(0.3))
        assert result == pytest.approx(expected, abs=1e-8)

    def test_known_entropy_three_values(self):
        """
        Verify against hand-computed entropy for |[1, 2, 3]|.
        sum = 6, p = [1/6, 2/6, 3/6] = [1/6, 1/3, 1/2]
        H = -(1/6*log2(1/6) + 1/3*log2(1/3) + 1/2*log2(1/2))
        """
        from explainiverse.evaluation import compute_complexity
        fnames = ["a", "b", "c"]

        class _FixedExplainer:
            def __init__(self):
                self.feature_names = fnames

            def explain(self, instance):
                exp = Explanation(
                    explainer_name="fixed",
                    target_class="class_0",
                    explanation_data={
                        "feature_attributions": {"a": 1.0, "b": 2.0, "c": 3.0}
                    },
                )
                exp.feature_names = fnames
                return exp

        result = compute_complexity(_FixedExplainer(), np.array([0.0, 0.0, 0.0]))
        p = np.array([1.0/6, 2.0/6, 3.0/6])
        expected = -np.sum(p * np.log2(p))
        assert result == pytest.approx(expected, abs=1e-10)

    def test_scale_invariance(self, feature_names, single_instance):
        """
        Entropy of fractional contributions is scale-invariant:
        H(c*|a|) = H(|a|) for c > 0.
        """
        from explainiverse.evaluation import compute_complexity
        explainer_1x = _GradientProportionalExplainer(feature_names, scale=1.0)
        explainer_100x = _GradientProportionalExplainer(feature_names, scale=100.0)

        score_1x = compute_complexity(explainer_1x, single_instance)
        score_100x = compute_complexity(explainer_100x, single_instance)
        assert score_1x == pytest.approx(score_100x, abs=1e-8)

    def test_single_feature_returns_zero(self):
        """A single feature always has H = 0.0 (no uncertainty)."""
        from explainiverse.evaluation import compute_complexity
        fnames = ["f0"]

        class _SingleExplainer:
            def __init__(self):
                self.feature_names = fnames

            def explain(self, instance):
                exp = Explanation(
                    explainer_name="single",
                    target_class="class_0",
                    explanation_data={"feature_attributions": {"f0": 5.0}},
                )
                exp.feature_names = fnames
                return exp

        result = compute_complexity(_SingleExplainer(), np.array([1.0]))
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_inverse_relationship_to_sparseness(self, feature_names, single_instance):
        """
        Higher sparseness (Gini) generally implies lower complexity (entropy).
        Sparse explanation: high Gini, low entropy.
        Uniform explanation: low Gini, high entropy.
        """
        from explainiverse.evaluation import compute_sparseness, compute_complexity

        sparse_exp = _SparseExplainer(feature_names)
        uniform_exp = _UniformExplainer(feature_names)

        sparse_gini = compute_sparseness(sparse_exp, single_instance)
        uniform_gini = compute_sparseness(uniform_exp, single_instance)
        sparse_entropy = compute_complexity(sparse_exp, single_instance)
        uniform_entropy = compute_complexity(uniform_exp, single_instance)

        assert sparse_gini > uniform_gini
        assert sparse_entropy < uniform_entropy

    def test_ordering_sparse_lt_partial_lt_uniform(
        self, feature_names, single_instance
    ):
        """
        Semantic ordering (lower complexity = better):
        sparse < two-feature < uniform.
        """
        from explainiverse.evaluation import compute_complexity
        sparse_score = compute_complexity(
            _SparseExplainer(feature_names), single_instance
        )
        partial_score = compute_complexity(
            _TwoFeatureExplainer(feature_names), single_instance
        )
        uniform_score = compute_complexity(
            _UniformExplainer(feature_names), single_instance
        )
        assert sparse_score < partial_score < uniform_score

    def test_with_real_explainer(self, trained_model_and_explainer):
        """Complexity works with a real LIME explainer and trained model."""
        from explainiverse.evaluation import compute_complexity
        _, explainer, X = trained_model_and_explainer
        result = compute_complexity(explainer, X[0])
        assert isinstance(result, float)
        max_entropy = np.log2(4)  # 4 features
        assert 0.0 <= result <= max_entropy + 1e-10

    def test_near_zero_attributions_stable(self, feature_names, single_instance):
        """
        Very small attributions should not cause NaN or inf.
        """
        from explainiverse.evaluation import compute_complexity

        class _NearZeroExplainer:
            def __init__(self):
                self.feature_names = feature_names

            def explain(self, instance):
                attrs = {fn: 1e-300 for fn in feature_names}
                attrs[feature_names[0]] = 1.0
                exp = Explanation(
                    explainer_name="near_zero",
                    target_class="class_0",
                    explanation_data={"feature_attributions": attrs},
                )
                exp.feature_names = feature_names
                return exp

        result = compute_complexity(_NearZeroExplainer(), single_instance)
        assert isinstance(result, float)
        assert not np.isnan(result)
        assert not np.isinf(result)


# =============================================================================
# Complexity Tests — compute_batch_complexity()
# =============================================================================

class TestComputeBatchComplexity:
    """Tests for compute_batch_complexity."""

    def test_import(self):
        """Batch function is importable."""
        from explainiverse.evaluation import compute_batch_complexity
        assert callable(compute_batch_complexity)

    def test_returns_dict_with_expected_keys(self, feature_names, sample_data):
        """Batch result contains all expected summary statistics."""
        from explainiverse.evaluation import compute_batch_complexity
        explainer = _UniformExplainer(feature_names)
        result = compute_batch_complexity(explainer, sample_data)
        assert isinstance(result, dict)
        for key in ["mean", "std", "max", "min", "scores", "n_evaluated"]:
            assert key in result, f"Missing key: {key}"

    def test_scores_list_length(self, feature_names, sample_data):
        """scores list has one entry per evaluated instance."""
        from explainiverse.evaluation import compute_batch_complexity
        explainer = _UniformExplainer(feature_names)
        result = compute_batch_complexity(explainer, sample_data)
        assert len(result["scores"]) == len(sample_data)
        assert result["n_evaluated"] == len(sample_data)

    def test_max_instances_limits_evaluation(self, feature_names, sample_data):
        """max_instances caps the number of instances evaluated."""
        from explainiverse.evaluation import compute_batch_complexity
        explainer = _UniformExplainer(feature_names)
        result = compute_batch_complexity(
            explainer, sample_data, max_instances=5
        )
        assert result["n_evaluated"] == 5
        assert len(result["scores"]) == 5

    def test_uniform_batch_all_log2n(self, feature_names, sample_data):
        """Uniform explainer should yield all log2(n) complexity scores."""
        from explainiverse.evaluation import compute_batch_complexity
        explainer = _UniformExplainer(feature_names)
        result = compute_batch_complexity(explainer, sample_data)
        expected = np.log2(len(feature_names))
        assert result["mean"] == pytest.approx(expected, abs=1e-10)
        assert all(
            s == pytest.approx(expected, abs=1e-10) for s in result["scores"]
        )

    def test_sparse_batch_all_zero(self, feature_names, sample_data):
        """Sparse explainer should yield all zero complexity scores."""
        from explainiverse.evaluation import compute_batch_complexity
        explainer = _SparseExplainer(feature_names)
        result = compute_batch_complexity(explainer, sample_data)
        assert result["mean"] == pytest.approx(0.0, abs=1e-10)

    def test_mean_within_min_max(self, feature_names, sample_data):
        """Mean should always be between min and max."""
        from explainiverse.evaluation import compute_batch_complexity
        explainer = _GradientProportionalExplainer(feature_names)
        result = compute_batch_complexity(explainer, sample_data)
        assert result["min"] <= result["mean"] <= result["max"]

    def test_with_real_explainer(self, trained_model_and_explainer):
        """Batch complexity works with a real LIME explainer."""
        from explainiverse.evaluation import compute_batch_complexity
        _, explainer, X = trained_model_and_explainer
        result = compute_batch_complexity(
            explainer, X[:10], max_instances=10
        )
        assert result["n_evaluated"] == 10
        max_entropy = np.log2(4)
        assert all(0.0 <= s <= max_entropy + 1e-10 for s in result["scores"])


# =============================================================================
# Effective Complexity Tests — compute_effective_complexity()
# =============================================================================

class TestComputeEffectiveComplexity:
    """Tests for compute_effective_complexity (Nguyen & Martínez, 2020)."""

    def test_import(self):
        """Effective Complexity function is importable from evaluation module."""
        from explainiverse.evaluation import compute_effective_complexity
        assert callable(compute_effective_complexity)

    def test_returns_float(self, feature_names, single_instance):
        """compute_effective_complexity returns a float (int-valued or normalized)."""
        from explainiverse.evaluation import compute_effective_complexity
        explainer = _UniformExplainer(feature_names)
        result = compute_effective_complexity(explainer, single_instance)
        assert isinstance(result, float)

    # --- Absolute threshold tests ---

    def test_uniform_absolute_all_above(self, feature_names, single_instance):
        """
        Uniform attributions of 1.0, absolute threshold 1e-5:
        all 4 features exceed threshold → EC = 4.
        """
        from explainiverse.evaluation import compute_effective_complexity
        explainer = _UniformExplainer(feature_names, value=1.0)
        result = compute_effective_complexity(
            explainer, single_instance,
            threshold=1e-5, threshold_type="absolute",
        )
        assert result == pytest.approx(4.0)

    def test_sparse_absolute_one_above(self, feature_names, single_instance):
        """
        Sparse attributions [1, 0, 0, 0], absolute threshold 1e-5:
        only 1 feature exceeds threshold → EC = 1.
        """
        from explainiverse.evaluation import compute_effective_complexity
        explainer = _SparseExplainer(feature_names)
        result = compute_effective_complexity(
            explainer, single_instance,
            threshold=1e-5, threshold_type="absolute",
        )
        assert result == pytest.approx(1.0)

    def test_zero_absolute_none_above(self, feature_names, single_instance):
        """
        All-zero attributions: no features exceed any positive threshold.
        EC = 0.
        """
        from explainiverse.evaluation import compute_effective_complexity
        explainer = _ZeroExplainer(feature_names)
        result = compute_effective_complexity(
            explainer, single_instance,
            threshold=1e-5, threshold_type="absolute",
        )
        assert result == pytest.approx(0.0)

    def test_two_feature_absolute(self, feature_names, single_instance):
        """
        [0.7, 0.3, 0.0, 0.0] with threshold 1e-5: 2 features above.
        """
        from explainiverse.evaluation import compute_effective_complexity
        explainer = _TwoFeatureExplainer(feature_names, weight_a=0.7, weight_b=0.3)
        result = compute_effective_complexity(
            explainer, single_instance,
            threshold=1e-5, threshold_type="absolute",
        )
        assert result == pytest.approx(2.0)

    def test_absolute_threshold_filters_small(self, feature_names, single_instance):
        """
        With threshold=0.5, [0.7, 0.3, 0.0, 0.0] → only 1 feature above.
        """
        from explainiverse.evaluation import compute_effective_complexity
        explainer = _TwoFeatureExplainer(feature_names, weight_a=0.7, weight_b=0.3)
        result = compute_effective_complexity(
            explainer, single_instance,
            threshold=0.5, threshold_type="absolute",
        )
        assert result == pytest.approx(1.0)

    def test_absolute_high_threshold_yields_zero(self, feature_names, single_instance):
        """
        Threshold higher than all attributions → EC = 0.
        """
        from explainiverse.evaluation import compute_effective_complexity
        explainer = _UniformExplainer(feature_names, value=1.0)
        result = compute_effective_complexity(
            explainer, single_instance,
            threshold=100.0, threshold_type="absolute",
        )
        assert result == pytest.approx(0.0)

    # --- Relative threshold tests ---

    def test_uniform_relative_all_above(self, feature_names, single_instance):
        """
        Uniform [1,1,1,1], relative threshold 0.5:
        max(|a|) = 1.0, threshold = 0.5 * 1.0 = 0.5.
        All features = 1.0 > 0.5 → EC = 4.
        """
        from explainiverse.evaluation import compute_effective_complexity
        explainer = _UniformExplainer(feature_names, value=1.0)
        result = compute_effective_complexity(
            explainer, single_instance,
            threshold=0.5, threshold_type="relative",
        )
        assert result == pytest.approx(4.0)

    def test_sparse_relative_one_above(self, feature_names, single_instance):
        """
        Sparse [1,0,0,0], relative threshold 0.01:
        max(|a|) = 1.0, threshold = 0.01. Only feature 0 > 0.01 → EC = 1.
        """
        from explainiverse.evaluation import compute_effective_complexity
        explainer = _SparseExplainer(feature_names)
        result = compute_effective_complexity(
            explainer, single_instance,
            threshold=0.01, threshold_type="relative",
        )
        assert result == pytest.approx(1.0)

    def test_relative_threshold_known_case(self, feature_names, single_instance):
        """
        [0.7, 0.3, 0.0, 0.0], relative threshold 0.5:
        max(|a|) = 0.7, effective threshold = 0.5 * 0.7 = 0.35.
        0.7 > 0.35 ✓, 0.3 < 0.35 ✗ → EC = 1.
        """
        from explainiverse.evaluation import compute_effective_complexity
        explainer = _TwoFeatureExplainer(feature_names, weight_a=0.7, weight_b=0.3)
        result = compute_effective_complexity(
            explainer, single_instance,
            threshold=0.5, threshold_type="relative",
        )
        assert result == pytest.approx(1.0)

    def test_relative_low_threshold_captures_more(self, feature_names, single_instance):
        """
        [0.7, 0.3, 0.0, 0.0], relative threshold 0.1:
        max(|a|) = 0.7, effective threshold = 0.1 * 0.7 = 0.07.
        0.7 > 0.07 ✓, 0.3 > 0.07 ✓ → EC = 2.
        """
        from explainiverse.evaluation import compute_effective_complexity
        explainer = _TwoFeatureExplainer(feature_names, weight_a=0.7, weight_b=0.3)
        result = compute_effective_complexity(
            explainer, single_instance,
            threshold=0.1, threshold_type="relative",
        )
        assert result == pytest.approx(2.0)

    def test_zero_attributions_relative_returns_zero(self, feature_names, single_instance):
        """
        All-zero attributions with relative threshold: max = 0,
        so effective threshold = 0. By convention EC = 0 (no signal).
        """
        from explainiverse.evaluation import compute_effective_complexity
        explainer = _ZeroExplainer(feature_names)
        result = compute_effective_complexity(
            explainer, single_instance,
            threshold=0.01, threshold_type="relative",
        )
        assert result == pytest.approx(0.0)

    # --- Normalize tests ---

    def test_normalize_returns_fraction(self, feature_names, single_instance):
        """
        With normalize=True, EC is divided by n.
        Uniform over 4 features: 4/4 = 1.0.
        """
        from explainiverse.evaluation import compute_effective_complexity
        explainer = _UniformExplainer(feature_names, value=1.0)
        result = compute_effective_complexity(
            explainer, single_instance,
            threshold=1e-5, threshold_type="absolute", normalize=True,
        )
        assert result == pytest.approx(1.0)

    def test_normalize_sparse(self, feature_names, single_instance):
        """
        Sparse 1/4: normalized = 0.25.
        """
        from explainiverse.evaluation import compute_effective_complexity
        explainer = _SparseExplainer(feature_names)
        result = compute_effective_complexity(
            explainer, single_instance,
            threshold=1e-5, threshold_type="absolute", normalize=True,
        )
        assert result == pytest.approx(0.25)

    def test_normalize_range_zero_to_one(self, feature_names, single_instance):
        """Normalized EC should always be in [0, 1]."""
        from explainiverse.evaluation import compute_effective_complexity
        for Explainer in [_SparseExplainer, _UniformExplainer,
                          _GradientProportionalExplainer, _TwoFeatureExplainer,
                          _ZeroExplainer]:
            explainer = Explainer(feature_names)
            result = compute_effective_complexity(
                explainer, single_instance, normalize=True,
            )
            assert 0.0 <= result <= 1.0, (
                f"{Explainer.__name__} gave normalized EC {result} outside [0, 1]"
            )

    def test_unnormalized_range_zero_to_n(self, feature_names, single_instance):
        """Unnormalized EC should always be in [0, n]."""
        from explainiverse.evaluation import compute_effective_complexity
        n = len(feature_names)
        for Explainer in [_SparseExplainer, _UniformExplainer,
                          _GradientProportionalExplainer, _TwoFeatureExplainer,
                          _ZeroExplainer]:
            explainer = Explainer(feature_names)
            result = compute_effective_complexity(
                explainer, single_instance, normalize=False,
            )
            assert 0.0 <= result <= n, (
                f"{Explainer.__name__} gave EC {result} outside [0, {n}]"
            )

    # --- Property tests ---

    def test_negative_attributions_uses_absolute(self, feature_names, single_instance):
        """
        Effective Complexity uses |a_i|. Negative attributions above
        threshold in absolute value should be counted.
        """
        from explainiverse.evaluation import compute_effective_complexity

        class _NegativeSparseExplainer:
            def __init__(self, fnames):
                self.feature_names = fnames

            def explain(self, instance):
                attrs = {fn: 0.0 for fn in self.feature_names}
                attrs[self.feature_names[0]] = -1.0
                exp = Explanation(
                    explainer_name="neg_sparse",
                    target_class="class_0",
                    explanation_data={"feature_attributions": attrs},
                )
                exp.feature_names = self.feature_names
                return exp

        pos = _SparseExplainer(feature_names)
        neg = _NegativeSparseExplainer(feature_names)

        score_pos = compute_effective_complexity(pos, single_instance)
        score_neg = compute_effective_complexity(neg, single_instance)
        assert score_pos == pytest.approx(score_neg)

    def test_monotonic_in_threshold_absolute(self, feature_names, single_instance):
        """
        Increasing absolute threshold should never increase EC.
        EC(threshold=0.01) >= EC(threshold=0.5) >= EC(threshold=10).
        """
        from explainiverse.evaluation import compute_effective_complexity
        explainer = _GradientProportionalExplainer(feature_names)
        ec_low = compute_effective_complexity(
            explainer, single_instance,
            threshold=0.01, threshold_type="absolute",
        )
        ec_mid = compute_effective_complexity(
            explainer, single_instance,
            threshold=0.5, threshold_type="absolute",
        )
        ec_high = compute_effective_complexity(
            explainer, single_instance,
            threshold=10.0, threshold_type="absolute",
        )
        assert ec_low >= ec_mid >= ec_high

    def test_monotonic_in_threshold_relative(self, feature_names, single_instance):
        """
        Increasing relative threshold should never increase EC.
        """
        from explainiverse.evaluation import compute_effective_complexity
        explainer = _GradientProportionalExplainer(feature_names)
        ec_low = compute_effective_complexity(
            explainer, single_instance,
            threshold=0.01, threshold_type="relative",
        )
        ec_mid = compute_effective_complexity(
            explainer, single_instance,
            threshold=0.5, threshold_type="relative",
        )
        ec_high = compute_effective_complexity(
            explainer, single_instance,
            threshold=0.99, threshold_type="relative",
        )
        assert ec_low >= ec_mid >= ec_high

    def test_ordering_sparse_lt_partial_lt_uniform(
        self, feature_names, single_instance
    ):
        """
        Semantic ordering (lower EC = simpler):
        sparse < two-feature < uniform (with low threshold).
        """
        from explainiverse.evaluation import compute_effective_complexity
        sparse_ec = compute_effective_complexity(
            _SparseExplainer(feature_names), single_instance,
            threshold=1e-5, threshold_type="absolute",
        )
        partial_ec = compute_effective_complexity(
            _TwoFeatureExplainer(feature_names), single_instance,
            threshold=1e-5, threshold_type="absolute",
        )
        uniform_ec = compute_effective_complexity(
            _UniformExplainer(feature_names), single_instance,
            threshold=1e-5, threshold_type="absolute",
        )
        assert sparse_ec < partial_ec < uniform_ec

    def test_invalid_threshold_type_raises(self, feature_names, single_instance):
        """Invalid threshold_type should raise ValueError."""
        from explainiverse.evaluation import compute_effective_complexity
        explainer = _UniformExplainer(feature_names)
        with pytest.raises(ValueError, match="threshold_type"):
            compute_effective_complexity(
                explainer, single_instance, threshold_type="invalid",
            )

    def test_single_feature(self):
        """Single feature: EC = 1 if above threshold, 0 otherwise."""
        from explainiverse.evaluation import compute_effective_complexity
        fnames = ["f0"]

        class _SingleExplainer:
            def __init__(self):
                self.feature_names = fnames

            def explain(self, instance):
                exp = Explanation(
                    explainer_name="single",
                    target_class="class_0",
                    explanation_data={"feature_attributions": {"f0": 5.0}},
                )
                exp.feature_names = fnames
                return exp

        result = compute_effective_complexity(
            _SingleExplainer(), np.array([1.0]),
            threshold=1e-5, threshold_type="absolute",
        )
        assert result == pytest.approx(1.0)

    def test_with_real_explainer(self, trained_model_and_explainer):
        """Effective Complexity works with a real LIME explainer."""
        from explainiverse.evaluation import compute_effective_complexity
        _, explainer, X = trained_model_and_explainer
        result = compute_effective_complexity(explainer, X[0])
        assert isinstance(result, float)
        assert 0.0 <= result <= 4.0  # 4 features

    def test_with_real_explainer_normalized(self, trained_model_and_explainer):
        """Normalized EC with real explainer is in [0, 1]."""
        from explainiverse.evaluation import compute_effective_complexity
        _, explainer, X = trained_model_and_explainer
        result = compute_effective_complexity(explainer, X[0], normalize=True)
        assert 0.0 <= result <= 1.0


# =============================================================================
# Effective Complexity Tests — compute_batch_effective_complexity()
# =============================================================================

class TestComputeBatchEffectiveComplexity:
    """Tests for compute_batch_effective_complexity."""

    def test_import(self):
        """Batch function is importable."""
        from explainiverse.evaluation import compute_batch_effective_complexity
        assert callable(compute_batch_effective_complexity)

    def test_returns_dict_with_expected_keys(self, feature_names, sample_data):
        """Batch result contains all expected summary statistics."""
        from explainiverse.evaluation import compute_batch_effective_complexity
        explainer = _UniformExplainer(feature_names)
        result = compute_batch_effective_complexity(explainer, sample_data)
        assert isinstance(result, dict)
        for key in ["mean", "std", "max", "min", "scores", "n_evaluated"]:
            assert key in result, f"Missing key: {key}"

    def test_scores_list_length(self, feature_names, sample_data):
        """scores list has one entry per evaluated instance."""
        from explainiverse.evaluation import compute_batch_effective_complexity
        explainer = _UniformExplainer(feature_names)
        result = compute_batch_effective_complexity(explainer, sample_data)
        assert len(result["scores"]) == len(sample_data)
        assert result["n_evaluated"] == len(sample_data)

    def test_max_instances_limits_evaluation(self, feature_names, sample_data):
        """max_instances caps the number of instances evaluated."""
        from explainiverse.evaluation import compute_batch_effective_complexity
        explainer = _UniformExplainer(feature_names)
        result = compute_batch_effective_complexity(
            explainer, sample_data, max_instances=5,
        )
        assert result["n_evaluated"] == 5
        assert len(result["scores"]) == 5

    def test_uniform_batch_all_n(self, feature_names, sample_data):
        """Uniform explainer: all instances should yield EC = n."""
        from explainiverse.evaluation import compute_batch_effective_complexity
        explainer = _UniformExplainer(feature_names)
        result = compute_batch_effective_complexity(
            explainer, sample_data,
            threshold=1e-5, threshold_type="absolute",
        )
        n = len(feature_names)
        assert result["mean"] == pytest.approx(float(n))
        assert all(s == pytest.approx(float(n)) for s in result["scores"])

    def test_sparse_batch_all_one(self, feature_names, sample_data):
        """Sparse explainer: all instances should yield EC = 1."""
        from explainiverse.evaluation import compute_batch_effective_complexity
        explainer = _SparseExplainer(feature_names)
        result = compute_batch_effective_complexity(
            explainer, sample_data,
            threshold=1e-5, threshold_type="absolute",
        )
        assert result["mean"] == pytest.approx(1.0)

    def test_mean_within_min_max(self, feature_names, sample_data):
        """Mean should always be between min and max."""
        from explainiverse.evaluation import compute_batch_effective_complexity
        explainer = _GradientProportionalExplainer(feature_names)
        result = compute_batch_effective_complexity(explainer, sample_data)
        assert result["min"] <= result["mean"] <= result["max"]

    def test_batch_with_normalize(self, feature_names, sample_data):
        """Batch normalized EC: all scores in [0, 1]."""
        from explainiverse.evaluation import compute_batch_effective_complexity
        explainer = _GradientProportionalExplainer(feature_names)
        result = compute_batch_effective_complexity(
            explainer, sample_data, normalize=True,
        )
        assert all(0.0 <= s <= 1.0 for s in result["scores"])

    def test_with_real_explainer(self, trained_model_and_explainer):
        """Batch EC works with a real LIME explainer."""
        from explainiverse.evaluation import compute_batch_effective_complexity
        _, explainer, X = trained_model_and_explainer
        result = compute_batch_effective_complexity(
            explainer, X[:10], max_instances=10,
        )
        assert result["n_evaluated"] == 10
        assert all(0.0 <= s <= 4.0 for s in result["scores"])
