"""Continuous alternatives are compared in units of each feature's range."""

from fractions import Fraction
from math import nextafter

import numpy as np
import pytest

from explainiverse.explainers import CounterfactualExplainer


def _probabilities(target):
    # Keep the target probability constant to give the optimizer a flat objective.
    return np.column_stack(
        (
            np.where(target, 5 / 16, 1 / 2),
            np.full(len(target), 3 / 8),
            np.where(target, 5 / 16, 1 / 8),
        )
    )


@pytest.mark.parametrize(
    "limit,declared_limit,expected_count",
    [
        pytest.param(1e-8, None, 2, id="narrow-inferred-range"),
        pytest.param(1e-8, 2e-8, 2, id="narrow-declared-range"),
        pytest.param(1.0, 1e8, 1, id="nearby-in-large-declared-range"),
        pytest.param(1.0, None, 2, id="distinct-in-ordinary-units"),
        pytest.param(1e-8, 1.0, 1, id="nearby-in-ordinary-units"),
    ],
)
def test_continuous_duplicates_use_the_feature_range(limit, declared_limit, expected_count):
    threshold = limit / 2

    class TwoDirections:
        prediction_output_kind = "probabilities"

        def __init__(self):
            self.visited = set()

        def predict(self, rows):
            x = rows[:, 0]
            self.visited.update(map(float, x))
            return _probabilities(abs(x) >= threshold)

    model = TwoDirections()
    query = [0.0, 0.0]
    ranges = {"y": (-1.0, 1.0)}
    if declared_limit is not None:
        ranges["x"] = (-declared_limit, declared_limit)
    explainer = CounterfactualExplainer(
        model,
        [query, [-limit, 0.0], [limit, 0.0]],
        ["x", "y"],
        continuous_features=["y", "x"],
        feature_ranges=ranges,
        proximity_weight=0.0,
        diversity_weight=0.0,
    )
    rows = explainer.explain(
        query, desired_class=1, num_counterfactuals=2, max_attempts=2
    ).explanation_data["counterfactuals"]
    boundaries = {-threshold, threshold}
    assert boundaries <= model.visited
    assert len(rows) == expected_count
    assert all(x in boundaries and y == 0.0 for x, y in rows)
    assert len({x for x, _ in rows}) == len(rows)


@pytest.mark.parametrize(
    "span,gap",
    [
        pytest.param(19.0, nextafter(1.9e-6, 0.0), id="just-below"),
        pytest.param(10_000_000.0, 1.0, id="inclusive-boundary"),
        pytest.param(19.0, 1.9e-6, id="just-above-rounds-to-cutoff"),
    ],
)
def test_normalized_duplicate_cutoff_is_inclusive(span, gap):
    class Endpoints:
        prediction_output_kind = "probabilities"

        def predict(self, rows):
            x = rows[:, 0]
            return _probabilities((x == 0.0) | (x == gap))

    explainer = CounterfactualExplainer(
        Endpoints(),
        [[span], [0.0], [gap]],
        ["x"],
        proximity_weight=0.0,
        diversity_weight=0.0,
    )
    rows = explainer.explain(
        [span], desired_class=1, num_counterfactuals=2, max_attempts=2
    ).explanation_data["counterfactuals"]
    if Fraction(gap) / Fraction(span) > Fraction(1, 10_000_000):
        assert sorted(rows) == [[0.0], [gap]]
    else:
        assert rows in ([[0.0]], [[gap]])
