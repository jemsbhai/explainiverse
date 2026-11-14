"""Public ProtoDash Gaussian similarities and median widths across feature scales."""

from math import exp

import numpy as np
import pytest

from explainiverse.explainers.example_based import ProtoDashExplainer


@pytest.mark.parametrize(
    "width",
    [1.0, np.finfo(float).max, np.nextafter(0.0, 1.0)],
    ids=["unit", "maximum", "minimum-subnormal"],
)
def test_rbf_similarity_across_two_kernel_widths(width):
    explainer = ProtoDashExplainer(n_prototypes=1, kernel_width=width)
    # An identical coordinate makes no contribution, even when large compared
    # with the bandwidth. The other coordinates are exactly two widths apart.
    common = np.finfo(float).max
    data = explainer.explain(
        np.array([common, width]), np.array([[common, -width]])
    ).explanation_data
    assert data["similarity_scores"] == pytest.approx([exp(-2.0)], rel=1e-14)
    assert data["objective_weights"] == pytest.approx([exp(-2.0)], rel=1e-14)


@pytest.mark.parametrize(
    "unit",
    [1.0, 2.0**600, 2.0**-600],
    ids=["unit", "large", "small"],
)
def test_rbf_median_distance_and_prototype_gain(unit):
    rows = np.array([[0.0], [unit], [3 * unit], [7 * unit]])
    # The six unordered distances are 1, 2, 3, 4, 6 and 7 units.
    width = 3.5 * unit
    data = (
        ProtoDashExplainer(n_prototypes=1, epsilon=np.nextafter(0.0, 1.0))
        .find_prototypes(rows)
        .explanation_data
    )
    assert data["kernel_width"] == width
    # The central row at three units has the largest mean similarity.
    expected_weight = (1.0 + sum(exp(-0.5 * (d / 3.5) ** 2) for d in (3, 2, 4))) / 4
    assert data["prototype_indices"] == [2]
    assert data["objective_weights"] == pytest.approx([expected_weight], rel=1e-14)


def test_rbf_median_width_floor_after_small_nonzero_distance():
    unit = 2.0**-600
    rows = np.array([[0.0], [unit]])
    data = (
        ProtoDashExplainer(n_prototypes=1, epsilon=2 * unit)
        .explain(np.array([2 * unit]), rows)
        .explanation_data
    )
    assert data["kernel_width"] == 2 * unit
    assert data["prototype_indices"] == [1]
    assert data["similarity_scores"] == pytest.approx([exp(-0.125)], rel=1e-14)


def test_rbf_criticism_score_at_large_scale():
    unit = 2.0**600
    rows = np.array([[0.0], [unit], [3 * unit]])
    data = (
        ProtoDashExplainer(kernel_width=2 * unit)
        .find_criticisms(rows, [0], n_criticisms=1)
        .explanation_data
    )
    # At three units the mean over all three rows includes the row itself.
    to_prototype = exp(-9 / 8)
    expected = (to_prototype + exp(-0.5) + 1.0) / 3 - to_prototype
    assert data["criticism_indices"] == [2]
    assert data["unusualness_scores"] == pytest.approx([expected], rel=1e-14)


def test_rbf_finite_median_with_unrepresentable_upper_middle_distance():
    unit = 2.0**1023
    rows = np.array([[-1.75], [-1.5], [0.25], [0.625]]) * unit
    # Sorted distances in units are 0.25, 0.375, 1.75, 2, 2.125 and 2.375.
    data = ProtoDashExplainer(n_prototypes=1).find_prototypes(rows).explanation_data
    assert data["kernel_width"] == 1.875 * unit


def test_rbf_rounds_subnormal_median_after_averaging_distances():
    unit = np.nextafter(0.0, 1.0)
    rows = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 2], [2, 2, 0]]) * unit
    # The two middle distances are sqrt(5) and sqrt(8) units. Their mean
    # rounds to three units, while rounding both first would give two.
    data = ProtoDashExplainer(n_prototypes=1, epsilon=unit).find_prototypes(rows).explanation_data
    assert data["kernel_width"] == 3 * unit


def test_rbf_rejects_unrepresentable_automatic_width():
    largest = np.finfo(float).max
    rows = np.array([[-largest], [largest]])
    with pytest.raises(FloatingPointError, match="kernel width"):
        ProtoDashExplainer().find_prototypes(rows)


def test_rbf_median_counts_nonzero_distances_at_mixed_scales():
    unit = 2.0**800
    tiny = 2.0**-800
    rows = np.array([[0.0], [tiny], [unit], [3 * unit]])
    # The middle distances are unit and 2 * unit. The nonzero distance
    # between the first two rows counts even though tiny / unit rounds to zero.
    data = ProtoDashExplainer(n_prototypes=1).find_prototypes(rows).explanation_data
    assert data["kernel_width"] == 1.5 * unit


def test_rbf_median_uses_both_middle_distances_at_mixed_scales():
    small = 2.0**-550
    large = 2.0**-508
    rows = np.array([[0.0], [small], [3 * small], [large]])
    # The middle distances are 3 * small and large - 3 * small.
    data = ProtoDashExplainer(n_prototypes=1, epsilon=small).find_prototypes(rows).explanation_data
    assert data["kernel_width"] == large / 2
