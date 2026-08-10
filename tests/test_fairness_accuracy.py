"""Accuracy and claim-scope tests for fairness-related diagnostics."""

import numpy as np
import pytest

from explainiverse.core.explanation import Explanation
from explainiverse.evaluation.fairness import (
    compute_attribution_parity,
    compute_batch_group_fairness,
    compute_conditional_fairness,
    compute_counterfactual_fairness,
    compute_cross_group_lipschitz_diagnostic,
    compute_fidelity_disparity,
    compute_fidelity_gap,
    compute_group_fairness_score,
    compute_group_metric_disparity,
    compute_prediction_conditioned_metric_disparity,
    compute_sensitive_attribution_change,
    get_default_fairness_registry,
)


def test_group_default_is_explicitly_magnitude_not_quality_or_fairness():
    attrs = np.array([[1.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
    result = compute_group_metric_disparity(attrs, [0, 0, 1, 1])

    assert result["disparity"] == pytest.approx(1.0)
    assert result["metric_name"] == "attribution_l1_magnitude"
    assert result["canonical_explanation_quality"] is False
    assert result["canonical_fairness_metric"] is False


@pytest.mark.parametrize(
    "attributions",
    [np.array([[1.0, np.nan], [2.0, 3.0]]), np.empty((2, 0))],
)
def test_group_rejects_invalid_attribution_matrices(attributions):
    with pytest.raises(ValueError):
        compute_group_metric_disparity(attributions, [0, 1])


@pytest.mark.parametrize("labels", [[0, np.nan], [0, None]])
def test_group_rejects_missing_sensitive_labels(labels):
    with pytest.raises(ValueError, match="missing|NaN"):
        compute_group_metric_disparity(np.ones((2, 1)), labels)


def test_group_rejects_unhashable_sensitive_labels():
    labels = np.empty(2, dtype=object)
    labels[:] = [["a"], ["b"]]
    with pytest.raises(TypeError, match="hashable"):
        compute_group_metric_disparity(np.ones((2, 1)), labels)


def test_group_rejects_one_group_instead_of_fabricating_zero():
    with pytest.raises(ValueError, match="at least two observed groups"):
        compute_group_metric_disparity(np.ones((3, 2)), ["a", "a", "a"])


@pytest.mark.parametrize(
    "metric, error_type",
    [
        (lambda row: np.array([row[0], row[0]]), TypeError),
        (lambda row: np.nan, ValueError),
        (lambda row: True, TypeError),
    ],
)
def test_group_rejects_non_scalar_or_nonfinite_metric_outputs(metric, error_type):
    with pytest.raises(error_type, match="inner_metric"):
        compute_group_metric_disparity(np.ones((2, 1)), [0, 1], metric)


def test_group_does_not_hide_inner_metric_errors():
    def broken_metric(_row):
        raise RuntimeError("deliberate failure")

    with pytest.raises(ValueError, match="deliberate failure"):
        compute_group_metric_disparity(np.ones((2, 1)), [0, 1], broken_metric)


def test_multigroup_p_value_is_labeled_uncorrected_and_pairwise_values_are_retained():
    attrs = np.arange(12, dtype=float).reshape(6, 2)
    result = compute_group_metric_disparity(attrs, [0, 0, 1, 1, 2, 2])

    assert len(result["pairwise_p_values"]) == 3
    assert result["p_value"] == min(result["pairwise_p_values"].values())
    assert result["p_value_adjustment"] == "none"
    assert "uncorrected" in result["p_value_summary"]


def test_completely_tied_groups_report_mann_whitney_p_value_as_undefined():
    result = compute_group_metric_disparity(np.ones((4, 2)), [0, 0, 1, 1])

    assert result["disparity"] == 0.0
    assert result["pairwise_p_values"] == {(0, 1): None}
    assert result["p_value"] is None
    assert result["p_value_unavailable_pairs"] == [(0, 1)]
    assert "completely tied" in result["p_value_summary"]


def test_completely_tied_fidelity_scores_report_pairwise_p_value_as_undefined():
    result = compute_fidelity_gap([1.0, 1.0, 1.0, 1.0], [0, 0, 1, 1])

    assert result["pairwise_p_values"] == {(0, 1): None}
    assert result["p_value_unavailable_pairs"] == [(0, 1)]


def test_constant_but_different_groups_have_infinite_standardised_effect_not_zero():
    attrs = np.array([[1.0], [1.0], [2.0], [2.0]])
    result = compute_group_metric_disparity(attrs, [0, 0, 1, 1])

    assert np.isinf(result["effect_size"])


def test_singleton_groups_report_effect_size_unavailable():
    result = compute_group_metric_disparity(np.array([[1.0], [2.0]]), [0, 1])
    assert result["effect_size"] is None


def test_batch_lengths_must_match_and_empty_batches_are_rejected():
    with pytest.raises(ValueError, match="equal lengths"):
        compute_batch_group_fairness([np.ones((2, 1))], [])
    with pytest.raises(ValueError, match="must not be empty"):
        compute_batch_group_fairness([], [])


class _StaticExplainer:
    def __init__(self, explanations):
        self.explanations = iter(explanations)

    def explain(self, _row):
        return next(self.explanations)


def _explanation(names, values):
    return Explanation(
        explainer_name="test",
        target_class="target",
        explanation_data={"feature_attributions": dict(zip(names, values))},
        feature_names=names,
    )


def test_score_api_uses_declared_feature_order_without_mutating_explanations():
    first = _explanation(["a", "b"], [1.0, 0.0])
    second = _explanation(["a", "b"], [3.0, 0.0])
    result = compute_group_fairness_score(
        _StaticExplainer([first, second]),
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        [0, 1],
        inner_metric=lambda row: row[0],
    )

    assert result["disparity"] == pytest.approx(2.0)
    assert first.feature_names == ["a", "b"]
    assert second.feature_names == ["a", "b"]


def test_score_api_rejects_missing_or_changing_feature_alignment():
    missing_names = Explanation(
        "test",
        "target",
        {"feature_attributions": {"a": 1.0, "b": 2.0}},
    )
    with pytest.raises(ValueError, match="feature_names is required"):
        compute_group_fairness_score(
            _StaticExplainer([missing_names, missing_names]),
            np.ones((2, 2)),
            [0, 1],
        )

    with pytest.raises(ValueError, match="order changed"):
        compute_group_fairness_score(
            _StaticExplainer([_explanation(["a", "b"], [1, 2]), _explanation(["b", "a"], [2, 1])]),
            np.ones((2, 2)),
            [0, 1],
        )


def test_lipschitz_threshold_does_not_silently_relax_to_distant_pairs():
    with pytest.raises(ValueError, match="No cross-group pairs"):
        compute_cross_group_lipschitz_diagnostic(
            [[0.0], [10.0]], [[0.0], [1.0]], [0, 1], distance_threshold=1.0
        )


def test_lipschitz_zero_input_distance_with_different_explanations_is_infinite():
    result = compute_cross_group_lipschitz_diagnostic(
        [[0.0], [0.0]], [[0.0], [1.0]], [0, 1], distance_threshold=0.0
    )
    assert np.isinf(result["score"])
    assert result["canonical_individual_fairness"] is False


@pytest.mark.parametrize("n_pairs", [0, -1, True, 1.5])
def test_lipschitz_validates_pair_budget(n_pairs):
    with pytest.raises((TypeError, ValueError), match="n_pairs"):
        compute_cross_group_lipschitz_diagnostic(
            [[0.0], [1.0]], [[0.0], [1.0]], [0, 1], n_pairs=n_pairs
        )


def test_lipschitz_uses_local_rng_state():
    data = np.arange(20, dtype=float).reshape(10, 2)
    attrs = data / 2
    labels = np.tile([0, 1], 5)
    np.random.seed(123)
    expected = np.random.random()
    np.random.seed(123)
    compute_cross_group_lipschitz_diagnostic(data, attrs, labels, n_pairs=2)
    observed = np.random.random()
    assert observed == expected


def test_sensitive_change_rejects_one_group_and_ambiguous_multigroup_data():
    with pytest.raises(ValueError, match="exactly two"):
        compute_sensitive_attribution_change([[0, 1], [0, 2]], [[1], [1]], 0)
    with pytest.raises(ValueError, match="exactly two"):
        compute_sensitive_attribution_change([[0, 1], [1, 2], [2, 3]], [[1], [1], [1]], 0)


def test_sensitive_intervention_requires_zero_one_encoding_and_valid_output_shape():
    with pytest.raises(ValueError, match="encoded as 0 and 1"):
        compute_sensitive_attribution_change(
            [[-1, 0], [1, 0]], [[0], [0]], 0, lambda row: np.array([0.0])
        )
    with pytest.raises(ValueError, match="shape"):
        compute_sensitive_attribution_change(
            [[0, 0], [1, 0]], [[0], [0]], 0, lambda row: np.array([0.0, 1.0])
        )


def test_counterfactual_compatibility_name_quarantines_causal_claim():
    result = compute_counterfactual_fairness(
        [[0, 0], [1, 0]], [[0.0], [1.0]], 0, lambda row: np.array([row[0]])
    )
    assert result["method"] == "one_feature_intervention"
    assert result["canonical_counterfactual_fairness"] is False
    assert result["requires_structural_causal_model_for_counterfactual_claim"] is True


def test_matching_mode_is_identified_as_observational_proxy():
    result = compute_sensitive_attribution_change([[0, 0], [1, 0]], [[0.0], [1.0]], 0)
    assert result["method"] == "nearest_opposite_group_matching"
    assert result["match_distances"] == [0.0, 0.0]


def test_balagopalan_definitions_for_higher_is_better_scores():
    result = compute_fidelity_gap([1.0, 1.0, 0.2, 0.2], ["a", "a", "b", "b"])

    assert result["overall_mean"] == pytest.approx(0.6)
    assert result["max_gap_from_average"] == pytest.approx(0.4)
    assert result["mean_group_gap"] == pytest.approx(0.8)
    assert result["max_gap"] == result["max_gap_from_average"]
    assert result["mean_gap"] == result["mean_group_gap"]


def test_balagopalan_definition_reverses_direction_for_lower_is_better_loss():
    result = compute_fidelity_gap([0.1, 0.1, 0.5, 0.5], [0, 0, 1, 1], higher_is_better=False)

    assert result["overall_mean"] == pytest.approx(0.3)
    assert result["max_gap_from_average"] == pytest.approx(0.2)
    assert result["group_deficits_from_average"][1] == pytest.approx(0.2)


def test_fidelity_disparity_refuses_to_infer_fidelity_from_attributions():
    with pytest.raises(ValueError, match="attribution magnitude is not fidelity"):
        compute_fidelity_disparity(np.ones((4, 2)), [0, 0, 1, 1])


def test_sensitive_attribution_zero_gap_is_not_labeled_fair():
    result = compute_attribution_parity(np.zeros((4, 2)), [0, 0, 1, 1], 0)
    assert result["divergence"] == 0.0
    assert result["canonical_fairness_metric"] is False
    assert "not proof" in result["interpretation"]


def test_sensitive_attribution_requires_multiple_groups():
    with pytest.raises(ValueError, match="at least two observed groups"):
        compute_attribution_parity(np.zeros((3, 2)), [0, 0, 0], 0)


def test_prediction_conditioned_diagnostic_does_not_fabricate_zero_for_no_overlap():
    with pytest.raises(ValueError, match="not estimable"):
        compute_prediction_conditioned_metric_disparity(np.ones((4, 2)), [0, 0, 1, 1], [0, 0, 1, 1])


def test_prediction_conditioned_diagnostic_marks_noncomparable_strata():
    attrs = np.array([[1.0], [1.0], [2.0], [3.0], [2.0], [4.0]])
    result = compute_conditional_fairness(
        attrs,
        [0, 0, 0, 0, 1, 1],
        ["isolated", "isolated", "shared", "shared", "shared", "shared"],
    )

    assert result["per_class_disparity"]["isolated"] is None
    assert result["per_class_disparity"]["shared"] == pytest.approx(0.5)
    assert result["non_comparable_classes"] == ["isolated"]
    assert result["canonical_equal_opportunity"] is False


def test_prediction_labels_must_be_observed_and_finite():
    with pytest.raises(ValueError, match="NaN"):
        compute_conditional_fairness(np.ones((2, 1)), [0, 1], [0, np.nan])


def test_registry_does_not_synthesize_required_context_or_overstate_claims():
    registry = get_default_fairness_registry()
    attrs = np.ones((2, 1))
    labels = np.array([0, 1])

    with pytest.raises(ValueError, match="requires inputs"):
        registry.evaluate("individual_fairness", attrs, labels)
    with pytest.raises(ValueError, match="requires predictions"):
        registry.evaluate("conditional_fairness", attrs, labels)
    assert registry.get_meta("group_fairness").canonical_claim is False
    assert registry.get_meta("fidelity_disparity").canonical_claim is True
