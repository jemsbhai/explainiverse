"""Analytical regressions for empty matching spaces in sensitive diagnostics."""

import numpy as np
import pytest

from explainiverse.evaluation.fairness import (
    compute_counterfactual_fairness,
    compute_cross_group_lipschitz_diagnostic,
    compute_sensitive_attribution_change,
)


@pytest.mark.parametrize(
    "function", [compute_sensitive_attribution_change, compute_counterfactual_fairness]
)
@pytest.mark.parametrize("with_constant_covariate", [False, True])
def test_matching_a_single_sensitive_feature(function, with_constant_covariate):
    inputs = np.array([[0.0], [1.0]])
    if with_constant_covariate:
        inputs = np.column_stack((inputs, np.zeros(2)))
    attrs = np.array([[1.0], [3.0]])
    before_inputs = inputs.copy()
    before_attrs = attrs.copy()
    result = function(inputs, attrs, sensitive_feature_idx=0)
    assert result["score"] == 2.0
    assert result["per_instance_scores"] == [2.0, 2.0]
    assert result["match_distances"] == [0.0, 0.0]
    assert result["method"] == "nearest_opposite_group_matching"
    assert result["canonical_counterfactual_fairness"] is False
    assert result["fairness_conclusion_defined"] is False
    np.testing.assert_array_equal(inputs, before_inputs)
    np.testing.assert_array_equal(attrs, before_attrs)


@pytest.mark.parametrize(
    "function", [compute_sensitive_attribution_change, compute_counterfactual_fairness]
)
@pytest.mark.parametrize("observed_values", [(0.0, 1.0), (-7.0, 4.0)])
@pytest.mark.parametrize(
    "groups, attrs, expected_scores",
    [
        ([0, 1, 1], [1.0, 3.0, 8.0], [2.0, 2.0, 7.0]),
        ([0, 1, 0, 1], [1.0, 3.0, 8.0, 12.0], [2.0, 2.0, 5.0, 11.0]),
    ],
)
def test_empty_matching_space_uses_first_opposite_row_on_distance_ties(
    function, observed_values, groups, attrs, expected_scores
):
    inputs = np.asarray([[observed_values[group]] for group in groups])
    attributions = np.asarray(attrs)[:, None]
    result = function(inputs, attributions, sensitive_feature_idx=0)
    assert result["per_instance_scores"] == expected_scores
    assert result["score"] == pytest.approx(sum(expected_scores) / len(groups))
    assert result["match_distances"] == [0.0] * len(groups)
    # Empty matching coordinates are equivalent to a constant extra coordinate.
    with_constant = function(
        np.column_stack((inputs, np.full(len(groups), 5.0))),
        attributions,
        sensitive_feature_idx=0,
    )
    assert result == with_constant


@pytest.mark.parametrize(
    "function", [compute_sensitive_attribution_change, compute_counterfactual_fairness]
)
def test_single_sensitive_feature_with_equal_attributions_has_zero_change(function):
    result = function([[0.0], [1.0]], [[2.0, 5.0], [2.0, 5.0]], sensitive_feature_idx=0)
    assert result["score"] == 0.0
    assert result["per_instance_scores"] == [0.0, 0.0]
    assert result["match_distances"] == [0.0, 0.0]


@pytest.mark.parametrize(
    "function", [compute_sensitive_attribution_change, compute_counterfactual_fairness]
)
def test_nonconstant_matching_coordinates_still_choose_nearest_opposite_row(function):
    result = function(
        [[0.0, 0.0], [1.0, 8.0], [1.0, 2.0], [0.0, 10.0]],
        [[1.0], [3.0], [8.0], [12.0]],
        sensitive_feature_idx=0,
    )
    assert result["score"] == 8.0
    assert result["per_instance_scores"] == [7.0, 9.0, 7.0, 9.0]
    assert result["match_distances"] == [2.0, 2.0, 2.0, 2.0]


@pytest.mark.parametrize(
    "function", [compute_sensitive_attribution_change, compute_counterfactual_fairness]
)
@pytest.mark.parametrize("empty_argument", ["inputs", "attributions"])
def test_sensitive_change_still_rejects_truly_empty_public_feature_matrices(
    function, empty_argument
):
    kwargs = {"inputs": np.array([[0.0], [1.0]]), "attributions": np.array([[1.0], [3.0]])}
    kwargs[empty_argument] = np.empty((2, 0))
    with pytest.raises(ValueError, match=f"{empty_argument} must contain at least one feature"):
        function(**kwargs, sensitive_feature_idx=0)


@pytest.mark.parametrize("empty_argument", ["inputs", "attributions"])
def test_cross_group_diagnostic_still_rejects_truly_empty_public_feature_matrices(
    empty_argument,
):
    kwargs = {"inputs": np.array([[0.0], [1.0]]), "attributions": np.array([[1.0], [3.0]])}
    kwargs[empty_argument] = np.empty((2, 0))
    with pytest.raises(ValueError, match=f"{empty_argument} must contain at least one feature"):
        compute_cross_group_lipschitz_diagnostic(**kwargs, sensitive_features=[0, 1])


@pytest.mark.parametrize(
    "function", [compute_sensitive_attribution_change, compute_counterfactual_fairness]
)
def test_single_sensitive_feature_explicit_intervention_is_unchanged(function):
    result = function(
        [[0.0], [1.0]],
        [[1.0], [3.0]],
        sensitive_feature_idx=0,
        counterfactual_explainer=lambda row: np.array([1.0 + 2.0 * row[0]]),
    )
    assert result["score"] == 2.0
    assert result["per_instance_scores"] == [2.0, 2.0]
    assert result["match_distances"] is None
    assert result["method"] == "one_feature_intervention"
