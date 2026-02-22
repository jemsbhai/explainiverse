# tests/test_agreement.py
"""
Tests for pairwise agreement metrics (Feature Agreement, Rank Agreement).

Reference:
    Krishna, S., Han, T., Gu, A., Pombra, J., Jabbari, S., Wu, S.,
    & Lakkaraju, H. (2022). The Disagreement Problem in Explainable
    Machine Learning: A Practitioner's Perspective. TMLR.
"""
import pytest
import numpy as np

from explainiverse.evaluation.agreement import (
    compute_feature_agreement,
    compute_batch_feature_agreement,
    compute_rank_agreement,
    compute_batch_rank_agreement,
    _extract_attribution_array,
    _validate_pair,
    _top_k_indices,
)
from explainiverse.core.explanation import Explanation


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_pair():
    """Two attributions with known top-k features."""
    # a: top-3 by abs = [3](0.8), [0](0.5), [2](0.3)
    # b: top-3 by abs = [3](0.7), [2](0.6), [0](0.4)
    a = np.array([0.5, 0.1, 0.3, 0.8, 0.2])
    b = np.array([0.4, 0.05, 0.6, 0.7, 0.15])
    return a, b


@pytest.fixture
def identical_pair():
    """Two identical attribution vectors."""
    a = np.array([0.3, 0.7, 0.1, 0.5, 0.9])
    return a, a.copy()


@pytest.fixture
def disjoint_pair():
    """Two attributions with completely disjoint top-k (for small k)."""
    # a: top-2 by abs = [0](0.9), [1](0.8)
    # b: top-2 by abs = [3](0.9), [4](0.8)
    a = np.array([0.9, 0.8, 0.01, 0.02, 0.03])
    b = np.array([0.01, 0.02, 0.03, 0.9, 0.8])
    return a, b


@pytest.fixture
def explanation_pair():
    """Pair of Explanation objects."""
    exp_a = Explanation(
        explainer_name="method_a",
        target_class=0,
        explanation_data={"feature_attributions": {"f0": 0.5, "f1": 0.1, "f2": 0.9}},
        feature_names=["f0", "f1", "f2"],
    )
    exp_b = Explanation(
        explainer_name="method_b",
        target_class=0,
        explanation_data={"feature_attributions": {"f0": 0.4, "f1": 0.8, "f2": 0.05}},
        feature_names=["f0", "f1", "f2"],
    )
    return exp_a, exp_b


# =============================================================================
# Tests: _top_k_indices
# =============================================================================

class TestTopKIndices:
    """Tests for the _top_k_indices helper."""

    def test_basic_ordering(self):
        attr = np.array([0.1, 0.5, 0.3, 0.9, 0.7])
        result = _top_k_indices(attr, 3)
        assert list(result) == [3, 4, 1]  # 0.9, 0.7, 0.5

    def test_k_equals_1(self):
        attr = np.array([0.2, 0.8, 0.5])
        result = _top_k_indices(attr, 1)
        assert list(result) == [1]

    def test_k_equals_n(self):
        attr = np.array([0.3, 0.1, 0.5])
        result = _top_k_indices(attr, 3)
        assert set(result) == {0, 1, 2}
        assert result[0] == 2  # highest

    def test_uses_absolute_values(self):
        attr = np.array([-0.9, 0.1, 0.5])
        result = _top_k_indices(attr, 1)
        assert result[0] == 0  # |-0.9| > 0.5

    def test_negative_attributions(self):
        attr = np.array([-0.8, -0.2, -0.5])
        result = _top_k_indices(attr, 2)
        assert list(result) == [0, 2]  # |-0.8|, |-0.5|


# =============================================================================
# Tests: Feature Agreement — Basic
# =============================================================================

class TestFeatureAgreementBasic:
    """Basic functionality tests for Feature Agreement."""

    def test_identical_attributions(self, identical_pair):
        a, b = identical_pair
        assert compute_feature_agreement(a, b, k=3) == 1.0

    def test_disjoint_top_k(self, disjoint_pair):
        a, b = disjoint_pair
        assert compute_feature_agreement(a, b, k=2) == 0.0

    def test_partial_overlap(self, simple_pair):
        a, b = simple_pair
        # a top-2: {3, 0}, b top-2: {3, 2}  => overlap {3} => 1/2
        result = compute_feature_agreement(a, b, k=2)
        assert result == pytest.approx(0.5)

    def test_full_overlap_at_k1(self, simple_pair):
        a, b = simple_pair
        # Both have feature 3 as top-1
        assert compute_feature_agreement(a, b, k=1) == 1.0

    def test_returns_float(self, simple_pair):
        a, b = simple_pair
        result = compute_feature_agreement(a, b, k=2)
        assert isinstance(result, float)

    def test_range_zero_to_one(self, simple_pair):
        a, b = simple_pair
        for k in range(1, 6):
            result = compute_feature_agreement(a, b, k=k)
            assert 0.0 <= result <= 1.0

    def test_symmetric(self, simple_pair):
        a, b = simple_pair
        assert compute_feature_agreement(a, b, k=3) == compute_feature_agreement(b, a, k=3)


# =============================================================================
# Tests: Feature Agreement — Parameter Variations
# =============================================================================

class TestFeatureAgreementParameters:
    """Test Feature Agreement across different k values."""

    def test_increasing_k_disjoint(self, disjoint_pair):
        """With disjoint top-2, k=5 must yield overlap since all features
        are in top-5 when n=5."""
        a, b = disjoint_pair
        assert compute_feature_agreement(a, b, k=2) == 0.0
        # k=5 means all features, overlap = 5/5 = 1.0
        assert compute_feature_agreement(a, b, k=5) == 1.0

    def test_k_equals_n_always_perfect(self):
        """When k equals the number of features, FA must be 1.0."""
        a = np.array([0.1, 0.2, 0.3])
        b = np.array([0.9, 0.8, 0.7])
        assert compute_feature_agreement(a, b, k=3) == 1.0

    def test_monotonicity_not_guaranteed(self):
        """FA(k) is not necessarily monotone in k, but FA(n)=1 always."""
        a = np.array([0.9, 0.01, 0.02, 0.03, 0.8])
        b = np.array([0.01, 0.9, 0.02, 0.8, 0.03])
        # Compute across k values — final one must be 1
        scores = [compute_feature_agreement(a, b, k=k) for k in range(1, 6)]
        assert scores[-1] == 1.0

    def test_k1_single_feature(self):
        """k=1 checks if the top feature agrees."""
        a = np.array([0.1, 0.9])
        b = np.array([0.8, 0.2])
        # a top-1: {1}, b top-1: {0} => 0.0
        assert compute_feature_agreement(a, b, k=1) == 0.0


# =============================================================================
# Tests: Feature Agreement — Edge Cases
# =============================================================================

class TestFeatureAgreementEdgeCases:
    """Edge case tests for Feature Agreement."""

    def test_two_features(self):
        a = np.array([0.6, 0.4])
        b = np.array([0.3, 0.7])
        # top-1: a={0}, b={1} => 0.0
        assert compute_feature_agreement(a, b, k=1) == 0.0
        # top-2: both sets are {0,1} => 1.0
        assert compute_feature_agreement(a, b, k=2) == 1.0

    def test_all_zeros_a(self):
        """When one attribution is all zeros, top-k is deterministic but
        arbitrary. No crash expected."""
        a = np.zeros(5)
        b = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        result = compute_feature_agreement(a, b, k=3)
        assert 0.0 <= result <= 1.0

    def test_all_zeros_both(self):
        """Two all-zero attributions should have perfect agreement
        (same arbitrary ordering)."""
        a = np.zeros(5)
        b = np.zeros(5)
        assert compute_feature_agreement(a, b, k=3) == 1.0

    def test_negative_attributions(self):
        """Absolute values are used for ranking."""
        a = np.array([-0.9, 0.1, 0.5])
        b = np.array([0.9, 0.1, 0.5])
        # top-1 for both: feature 0 (|0.9|)
        assert compute_feature_agreement(a, b, k=1) == 1.0

    def test_mixed_signs(self):
        a = np.array([-0.9, 0.8, -0.7])
        b = np.array([0.8, -0.9, 0.7])
        # a top-2: {0, 1}, b top-2: {1, 0} => overlap 2/2 = 1.0
        assert compute_feature_agreement(a, b, k=2) == 1.0

    def test_single_feature_vector(self):
        """n=1, k=1 must yield 1.0."""
        a = np.array([0.5])
        b = np.array([0.3])
        assert compute_feature_agreement(a, b, k=1) == 1.0

    def test_large_feature_space(self):
        """Test with many features."""
        rng = np.random.RandomState(42)
        a = rng.randn(100)
        b = a.copy()
        assert compute_feature_agreement(a, b, k=50) == 1.0

    def test_ties_in_attributions(self):
        """When there are ties, the function should not crash."""
        a = np.array([0.5, 0.5, 0.5, 0.1, 0.1])
        b = np.array([0.5, 0.5, 0.5, 0.1, 0.1])
        result = compute_feature_agreement(a, b, k=3)
        # Same arrays => same tie-breaking => 1.0
        assert result == 1.0


# =============================================================================
# Tests: Rank Agreement — Basic
# =============================================================================

class TestRankAgreementBasic:
    """Basic functionality tests for Rank Agreement."""

    def test_identical_attributions(self, identical_pair):
        a, b = identical_pair
        assert compute_rank_agreement(a, b, k=3) == 1.0

    def test_disjoint_top_k(self, disjoint_pair):
        a, b = disjoint_pair
        assert compute_rank_agreement(a, b, k=2) == 0.0

    def test_same_features_different_order(self):
        """Features in top-k are the same but ranks differ."""
        a = np.array([0.9, 0.8, 0.1])
        b = np.array([0.8, 0.9, 0.1])
        # a: rank1=0, rank2=1; b: rank1=1, rank2=0
        assert compute_rank_agreement(a, b, k=2) == 0.0
        # Feature sets overlap perfectly though
        assert compute_feature_agreement(a, b, k=2) == 1.0

    def test_partial_rank_match(self, simple_pair):
        a, b = simple_pair
        # a top-2: [3, 0], b top-2: [3, 2]
        # rank 1: both=3 ✓, rank 2: a=0, b=2 ✗ => 1/2
        result = compute_rank_agreement(a, b, k=2)
        assert result == pytest.approx(0.5)

    def test_returns_float(self, simple_pair):
        a, b = simple_pair
        assert isinstance(compute_rank_agreement(a, b, k=2), float)

    def test_range_zero_to_one(self, simple_pair):
        a, b = simple_pair
        for k in range(1, 6):
            result = compute_rank_agreement(a, b, k=k)
            assert 0.0 <= result <= 1.0

    def test_symmetric(self, simple_pair):
        a, b = simple_pair
        assert compute_rank_agreement(a, b, k=3) == compute_rank_agreement(b, a, k=3)


# =============================================================================
# Tests: Rank Agreement — Parameter Variations
# =============================================================================

class TestRankAgreementParameters:
    """Test Rank Agreement across different k values."""

    def test_k1_top_feature_same(self, simple_pair):
        a, b = simple_pair
        # Both rank feature 3 first
        assert compute_rank_agreement(a, b, k=1) == 1.0

    def test_k1_top_feature_different(self):
        a = np.array([0.9, 0.1])
        b = np.array([0.1, 0.9])
        assert compute_rank_agreement(a, b, k=1) == 0.0

    def test_k_equals_n_identical(self, identical_pair):
        a, b = identical_pair
        assert compute_rank_agreement(a, b, k=5) == 1.0

    def test_reversed_ranking(self):
        """Completely reversed ranking should have low RA."""
        a = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        b = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
        # a: [4,3,2,1,0], b: [0,1,2,3,4]
        # rank3: position match only at middle (feature 2) => 1/5
        result = compute_rank_agreement(a, b, k=5)
        assert result == pytest.approx(1.0 / 5.0)


# =============================================================================
# Tests: Rank Agreement — Edge Cases
# =============================================================================

class TestRankAgreementEdgeCases:
    """Edge case tests for Rank Agreement."""

    def test_single_feature(self):
        a = np.array([0.5])
        b = np.array([0.3])
        assert compute_rank_agreement(a, b, k=1) == 1.0

    def test_all_zeros_both(self):
        a = np.zeros(5)
        b = np.zeros(5)
        assert compute_rank_agreement(a, b, k=3) == 1.0

    def test_negative_attributions(self):
        a = np.array([-0.9, 0.1, 0.5])
        b = np.array([-0.9, 0.1, 0.5])
        assert compute_rank_agreement(a, b, k=2) == 1.0

    def test_large_feature_space_identical(self):
        rng = np.random.RandomState(123)
        a = rng.randn(200)
        assert compute_rank_agreement(a, a.copy(), k=50) == 1.0

    def test_large_feature_space_random(self):
        """Two random attributions should have low RA for small k."""
        rng = np.random.RandomState(99)
        a = rng.randn(100)
        b = rng.randn(100)
        result = compute_rank_agreement(a, b, k=5)
        # Very unlikely to have high agreement by chance
        assert result < 1.0


# =============================================================================
# Tests: Semantic Invariant — RA ≤ FA
# =============================================================================

class TestRAleqFA:
    """Rank Agreement should never exceed Feature Agreement."""

    def test_ra_leq_fa_simple(self, simple_pair):
        a, b = simple_pair
        for k in range(1, 6):
            fa = compute_feature_agreement(a, b, k=k)
            ra = compute_rank_agreement(a, b, k=k)
            assert ra <= fa + 1e-12, f"RA > FA at k={k}: RA={ra}, FA={fa}"

    def test_ra_leq_fa_random(self):
        rng = np.random.RandomState(42)
        for _ in range(20):
            n = rng.randint(3, 30)
            a = rng.randn(n)
            b = rng.randn(n)
            k = rng.randint(1, n + 1)
            fa = compute_feature_agreement(a, b, k=k)
            ra = compute_rank_agreement(a, b, k=k)
            assert ra <= fa + 1e-12

    def test_ra_equals_fa_when_identical(self, identical_pair):
        a, b = identical_pair
        for k in range(1, 6):
            fa = compute_feature_agreement(a, b, k=k)
            ra = compute_rank_agreement(a, b, k=k)
            assert fa == 1.0
            assert ra == 1.0


# =============================================================================
# Tests: Explanation Objects
# =============================================================================

class TestExplanationObjects:
    """Test that Explanation objects work as inputs."""

    def test_feature_agreement_with_explanations(self, explanation_pair):
        exp_a, exp_b = explanation_pair
        # exp_a: f2=0.9, f0=0.5, f1=0.1
        # exp_b: f1=0.8, f0=0.4, f2=0.05
        # top-1: a={f2}, b={f1} => 0.0
        assert compute_feature_agreement(exp_a, exp_b, k=1) == 0.0

    def test_rank_agreement_with_explanations(self, explanation_pair):
        exp_a, exp_b = explanation_pair
        assert compute_rank_agreement(exp_a, exp_b, k=1) == 0.0

    def test_mixed_array_and_explanation(self, explanation_pair):
        """One Explanation, one np.ndarray."""
        exp_a, _ = explanation_pair
        b = np.array([0.4, 0.8, 0.05])  # same values as exp_b
        result = compute_feature_agreement(exp_a, b, k=1)
        assert isinstance(result, float)

    def test_explanation_preserves_feature_order(self):
        """Feature names should control ordering."""
        exp = Explanation(
            explainer_name="test",
            target_class=0,
            explanation_data={"feature_attributions": {"z": 0.1, "a": 0.9, "m": 0.5}},
            feature_names=["z", "a", "m"],
        )
        arr = _extract_attribution_array(exp)
        assert list(arr) == [0.1, 0.9, 0.5]


# =============================================================================
# Tests: Batch Operations
# =============================================================================

class TestBatchOperations:
    """Test batch versions of both metrics."""

    def test_batch_feature_agreement_basic(self):
        batch_a = [np.array([0.9, 0.1, 0.5]), np.array([0.1, 0.9, 0.5])]
        batch_b = [np.array([0.9, 0.1, 0.5]), np.array([0.5, 0.1, 0.9])]
        results = compute_batch_feature_agreement(batch_a, batch_b, k=2)
        assert len(results) == 2
        assert results[0] == 1.0  # identical
        assert isinstance(results[1], float)

    def test_batch_rank_agreement_basic(self):
        batch_a = [np.array([0.9, 0.1, 0.5]), np.array([0.1, 0.9, 0.5])]
        batch_b = [np.array([0.9, 0.1, 0.5]), np.array([0.5, 0.1, 0.9])]
        results = compute_batch_rank_agreement(batch_a, batch_b, k=2)
        assert len(results) == 2
        assert results[0] == 1.0

    def test_batch_single_element(self):
        results = compute_batch_feature_agreement(
            [np.array([0.5, 0.3])],
            [np.array([0.3, 0.5])],
            k=1,
        )
        assert len(results) == 1

    def test_batch_empty(self):
        results = compute_batch_feature_agreement([], [], k=1)
        assert results == []

    def test_batch_with_explanations(self, explanation_pair):
        exp_a, exp_b = explanation_pair
        results = compute_batch_feature_agreement([exp_a], [exp_b], k=2)
        assert len(results) == 1

    def test_batch_size_mismatch_fa(self):
        with pytest.raises(ValueError, match="Batch sizes must match"):
            compute_batch_feature_agreement(
                [np.array([0.1, 0.2])],
                [np.array([0.1, 0.2]), np.array([0.3, 0.4])],
                k=1,
            )

    def test_batch_size_mismatch_ra(self):
        with pytest.raises(ValueError, match="Batch sizes must match"):
            compute_batch_rank_agreement(
                [np.array([0.1, 0.2])],
                [np.array([0.1, 0.2]), np.array([0.3, 0.4])],
                k=1,
            )


# =============================================================================
# Tests: Error Handling
# =============================================================================

class TestErrorHandling:
    """Test validation and error handling."""

    def test_shape_mismatch(self):
        a = np.array([0.1, 0.2, 0.3])
        b = np.array([0.1, 0.2])
        with pytest.raises(ValueError, match="same length"):
            compute_feature_agreement(a, b, k=1)

    def test_k_zero(self):
        a = np.array([0.1, 0.2])
        with pytest.raises(ValueError, match="positive integer"):
            compute_feature_agreement(a, a, k=0)

    def test_k_negative(self):
        a = np.array([0.1, 0.2])
        with pytest.raises(ValueError, match="positive integer"):
            compute_feature_agreement(a, a, k=-1)

    def test_k_exceeds_n(self):
        a = np.array([0.1, 0.2])
        with pytest.raises(ValueError, match="exceeds"):
            compute_feature_agreement(a, a, k=3)

    def test_empty_arrays(self):
        a = np.array([])
        b = np.array([])
        with pytest.raises(ValueError, match="empty"):
            compute_feature_agreement(a, b, k=1)

    def test_2d_arrays(self):
        a = np.array([[0.1, 0.2]])
        b = np.array([[0.3, 0.4]])
        # Should work because _extract_attribution_array flattens
        result = compute_feature_agreement(a, b, k=1)
        assert isinstance(result, float)

    def test_invalid_type_string(self):
        with pytest.raises(TypeError, match="Expected"):
            compute_feature_agreement("not an array", np.array([0.1]), k=1)

    def test_invalid_type_list(self):
        with pytest.raises(TypeError, match="Expected"):
            compute_feature_agreement([0.1, 0.2], np.array([0.1, 0.2]), k=1)

    def test_explanation_no_attributions(self):
        exp = Explanation(
            explainer_name="test",
            target_class=0,
            explanation_data={},
        )
        with pytest.raises(ValueError, match="No feature attributions"):
            compute_feature_agreement(exp, exp, k=1)

    def test_k_float_raises(self):
        a = np.array([0.1, 0.2])
        with pytest.raises(ValueError, match="positive integer"):
            compute_feature_agreement(a, a, k=1.5)

    def test_rank_agreement_same_errors(self):
        """Rank Agreement should raise the same errors as Feature Agreement."""
        a = np.array([0.1, 0.2])
        b = np.array([0.1])
        with pytest.raises(ValueError, match="same length"):
            compute_rank_agreement(a, b, k=1)

        with pytest.raises(ValueError, match="exceeds"):
            compute_rank_agreement(a, a, k=5)


# =============================================================================
# Tests: Specific Scenarios
# =============================================================================

class TestSpecificScenarios:
    """Manually computed expected values for verification."""

    def test_known_fa_3features_k2(self):
        """
        a = [0.1, 0.9, 0.5] => top-2: {1, 2}
        b = [0.8, 0.2, 0.5] => top-2: {0, 2}
        intersection = {2} => FA = 1/2 = 0.5
        """
        a = np.array([0.1, 0.9, 0.5])
        b = np.array([0.8, 0.2, 0.5])
        assert compute_feature_agreement(a, b, k=2) == pytest.approx(0.5)

    def test_known_ra_3features_k2(self):
        """
        a = [0.1, 0.9, 0.5] => ranking: [1, 2] (rank1=1, rank2=2)
        b = [0.8, 0.2, 0.5] => ranking: [0, 2] (rank1=0, rank2=2)
        Position 1: 1 vs 0 ✗, Position 2: 2 vs 2 ✓ => RA = 1/2 = 0.5
        """
        a = np.array([0.1, 0.9, 0.5])
        b = np.array([0.8, 0.2, 0.5])
        assert compute_rank_agreement(a, b, k=2) == pytest.approx(0.5)

    def test_known_fa_perfect(self):
        """Scaled version should have perfect FA."""
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([10.0, 20.0, 30.0, 40.0])
        assert compute_feature_agreement(a, b, k=3) == 1.0

    def test_known_ra_perfect_scaled(self):
        """Positive scaling preserves ranking => perfect RA."""
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = a * 5.0
        assert compute_rank_agreement(a, b, k=4) == 1.0

    def test_known_fa_with_negatives(self):
        """
        a = [-0.9, 0.5, 0.1] => abs: [0.9, 0.5, 0.1] => top-2: {0, 1}
        b = [0.1, -0.8, 0.6] => abs: [0.1, 0.8, 0.6] => top-2: {1, 2}
        intersection = {1} => FA = 1/2 = 0.5
        """
        a = np.array([-0.9, 0.5, 0.1])
        b = np.array([0.1, -0.8, 0.6])
        assert compute_feature_agreement(a, b, k=2) == pytest.approx(0.5)

    def test_swapped_top2(self):
        """
        a = [0.9, 0.8, 0.1] => top-2: [0, 1]
        b = [0.8, 0.9, 0.1] => top-2: [1, 0]
        FA(k=2) = 2/2 = 1.0 (same set)
        RA(k=2) = 0/2 = 0.0 (positions swapped)
        """
        a = np.array([0.9, 0.8, 0.1])
        b = np.array([0.8, 0.9, 0.1])
        assert compute_feature_agreement(a, b, k=2) == 1.0
        assert compute_rank_agreement(a, b, k=2) == 0.0

    def test_one_match_out_of_three(self):
        """
        a = [0.9, 0.7, 0.5, 0.1] => top-3: [0, 1, 2]
        b = [0.1, 0.5, 0.7, 0.9] => top-3: [3, 2, 1]
        FA: {0,1,2} ∩ {3,2,1} = {1,2} => 2/3
        RA: pos1: 0 vs 3 ✗, pos2: 1 vs 2 ✗, pos3: 2 vs 1 ✗ => 0/3
        """
        a = np.array([0.9, 0.7, 0.5, 0.1])
        b = np.array([0.1, 0.5, 0.7, 0.9])
        assert compute_feature_agreement(a, b, k=3) == pytest.approx(2.0 / 3.0)
        assert compute_rank_agreement(a, b, k=3) == pytest.approx(0.0)


# =============================================================================
# Tests: Consistency across runs
# =============================================================================

class TestDeterminism:
    """Results should be deterministic."""

    def test_feature_agreement_deterministic(self, simple_pair):
        a, b = simple_pair
        r1 = compute_feature_agreement(a, b, k=3)
        r2 = compute_feature_agreement(a, b, k=3)
        assert r1 == r2

    def test_rank_agreement_deterministic(self, simple_pair):
        a, b = simple_pair
        r1 = compute_rank_agreement(a, b, k=3)
        r2 = compute_rank_agreement(a, b, k=3)
        assert r1 == r2

    def test_batch_deterministic(self):
        batch_a = [np.array([0.5, 0.3, 0.8])] * 5
        batch_b = [np.array([0.3, 0.5, 0.7])] * 5
        r1 = compute_batch_feature_agreement(batch_a, batch_b, k=2)
        r2 = compute_batch_feature_agreement(batch_a, batch_b, k=2)
        assert r1 == r2
