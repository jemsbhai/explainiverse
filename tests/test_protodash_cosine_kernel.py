"""Public ProtoDash cosine results depend on direction for every nonzero row."""

from math import sqrt

import numpy as np
import pytest

from explainiverse.explainers.example_based import ProtoDashExplainer


@pytest.mark.parametrize(
    "scale",
    [1.0, np.finfo(float).max, np.nextafter(0.0, 1.0)],
    ids=["unit", "maximum", "minimum-subnormal"],
)
def test_cosine_similarity_at_different_magnitudes(scale):
    explainer = ProtoDashExplainer(n_prototypes=1, kernel="cosine")
    # The reference lies at 135 degrees to the positive first axis.
    data = explainer.explain(np.array([1.0, 0.0]), np.array([[-scale, scale]])).explanation_data
    assert data["similarity_scores"] == pytest.approx([-1 / sqrt(2)], rel=1e-14)


def test_cosine_selects_aligned_reference_across_magnitudes():
    explainer = ProtoDashExplainer(n_prototypes=1, kernel="cosine")
    # The first row has cosine 3/5. The second is parallel to the instance.
    data = explainer.explain(
        np.array([2.0, 0.0]), np.array([[3.0, 4.0], [1e-14, 0.0]])
    ).explanation_data
    assert data["prototype_indices"] == [1]
    assert data["similarity_scores"] == pytest.approx([1.0], rel=1e-14)
    assert data["objective_weights"] == pytest.approx([1.0], rel=1e-14)


def test_cosine_distribution_selects_direction_with_highest_mean_similarity():
    explainer = ProtoDashExplainer(n_prototypes=1, kernel="cosine")
    rows = np.array([[3e-15, 4e-15], [1.0, 0.0], [0.0, 1.0]])
    data = explainer.find_prototypes(rows, return_mmd=True).explanation_data
    # The row means of the cosine matrix are 4/5, 8/15, and 3/5.
    # The unit-vector centroid is (8/15, 3/5), so its distance from the
    # selected unit vector (3/5, 4/5) is sqrt(10)/15.
    assert data["prototype_indices"] == [0]
    assert data["objective_weights"] == pytest.approx([4 / 5], rel=1e-14)
    assert data["mmd_score"] == pytest.approx(sqrt(10) / 15, rel=1e-14)


@pytest.mark.parametrize(
    ("target", "reference"),
    [
        ([0.0, 0.0], [3.0, 4.0]),
        ([3.0, 4.0], [0.0, 0.0]),
        ([1.0, 0.0], [0.0, 1.0]),
    ],
    ids=["zero-target", "zero-reference", "orthogonal"],
)
def test_cosine_zero_similarity(target, reference):
    explainer = ProtoDashExplainer(n_prototypes=1, kernel="cosine")
    data = explainer.explain(np.array(target), np.array([reference])).explanation_data
    assert data["similarity_scores"] == [0.0]
