"""Refinement bisects the segment from the query to a valid continuous candidate."""

from fractions import Fraction

import numpy as np
import pytest

from explainiverse.explainers import CounterfactualExplainer

_MAX = float(np.finfo(float).max)


class _ThresholdClassifier:
    prediction_output_kind = "probabilities"

    def __init__(self, threshold, increasing):
        self.threshold = threshold
        self.increasing = increasing

    def predict(self, rows):
        x = rows[:, 0]
        target = x >= self.threshold if self.increasing else x <= self.threshold
        return np.column_stack((~target, target)).astype(float)


@pytest.mark.parametrize("limit", [32.0, _MAX], ids=["ordinary", "large"])
@pytest.mark.parametrize("sign", [1, -1], ids=["positive", "negative"])
@pytest.mark.parametrize("inward", [False, True], ids=["outward", "inward"])
def test_continuous_threshold_search_refines_within_the_initial_segment(limit, sign, inward):
    query, exemplar = sign * limit * 0.5, sign * limit * 0.875
    if inward:
        query, exemplar = exemplar, query
    threshold = sign * limit * 0.75
    model = _ThresholdClassifier(threshold, increasing=exemplar > query)
    explainer = CounterfactualExplainer(
        model,
        [[query], [exemplar]],
        ["x"],
        feature_ranges={"x": (0.0, limit) if sign > 0 else (-limit, 0.0)},
        proximity_weight=0.0,
    )
    data = explainer.explain(
        [query], desired_class=1, num_counterfactuals=1, max_attempts=1
    ).explanation_data
    (candidate,) = data["counterfactuals"]

    # Binary search halves the query-to-exemplar segment at each step. Compare
    # against midpoints rounded once from exact binary64 endpoints.
    low, high = query, exemplar
    for _ in range(30):
        middle = float((Fraction(low) + Fraction(high)) / 2)
        is_target = middle >= threshold if exemplar > query else middle <= threshold
        if is_target:
            high = middle
        else:
            low = middle
    assert candidate == [high]


def test_large_and_subnormal_coordinates_share_the_same_midpoint():
    tiny = float(np.nextafter(0.0, 1.0))
    query = [_MAX * 0.5, tiny]
    exemplar = [_MAX * 0.875, 5 * tiny]
    # The first midpoint is on the boundary. Both coordinates must stop there.
    midpoint = [float((Fraction(a) + Fraction(b)) / 2) for a, b in zip(query, exemplar)]
    model = _ThresholdClassifier(midpoint[0], increasing=True)
    explainer = CounterfactualExplainer(
        model, [query, exemplar], ["large", "small"], proximity_weight=0.0
    )
    data = explainer.explain(
        query, desired_class=1, num_counterfactuals=1, max_attempts=1
    ).explanation_data

    assert data["counterfactuals"] == [midpoint]
