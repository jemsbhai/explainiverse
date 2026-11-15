"""Gaussian restarts perturb target exemplars within each feature's domain."""

from fractions import Fraction

import numpy as np
import pytest

from explainiverse.explainers import CounterfactualExplainer

_MAX = float(np.finfo(float).max)


@pytest.mark.parametrize("limit", [32.0, _MAX], ids=["ordinary", "wide"])
def test_gaussian_restart_reaches_a_second_disjoint_target_interval(limit):
    exemplar = limit / 4
    rng = np.random.default_rng(0)
    rng.integers(1)  # draw the only target exemplar
    z = Fraction(rng.standard_normal())
    center = float(Fraction(exemplar) / Fraction(limit) + 2 * Fraction(0.15) * z)
    radius = 0.0005

    class TwoIslands:
        prediction_output_kind = "probabilities"

        def predict(self, rows):
            x = rows[:, 0]
            target = (x == exemplar) | (abs(x / limit - center) <= radius)
            return np.column_stack((~target, target)).astype(float)

    explainer = CounterfactualExplainer(
        TwoIslands(),
        [[0.0], [exemplar]],
        ["x"],
        feature_ranges={"x": (-limit, limit)},
        random_state=0,
        proximity_weight=0.0,
        diversity_weight=0.0,
    )
    data = explainer.explain(
        [0.0], desired_class=1, num_counterfactuals=2, max_attempts=2
    ).explanation_data
    targets = [candidate[0] for candidate in data["counterfactuals"]]
    assert exemplar in targets
    assert any(abs(x / limit - center) <= radius for x in targets)


@pytest.mark.parametrize(
    "bounds,exemplar,seed",
    [
        pytest.param((-_MAX, _MAX), _MAX / 4, 4, id="negative-draw"),
        pytest.param((-_MAX, _MAX / 2), _MAX / 4, 0, id="asymmetric"),
        pytest.param((-_MAX, _MAX), -_MAX * 0.625, 2429, id="overflowing-draw-cancels"),
        pytest.param((-_MAX, _MAX), _MAX * 0.875, 3, id="clipped-to-upper-bound"),
        pytest.param((-32.0, 64.0), 16.0, 0, id="ordinary-control"),
    ],
)
def test_mixed_scale_restart_visits_the_projected_gaussian_draw(bounds, exemplar, seed):
    class RecordingClassifier:
        prediction_output_kind = "probabilities"

        def __init__(self):
            self.visited = []

        def predict(self, rows):
            self.visited.extend(map(tuple, rows))
            target = rows[:, 1] == exemplar
            return np.column_stack((~target, target)).astype(float)

    model = RecordingClassifier()
    query = [32.0, 0.0, 7.0]
    explainer = CounterfactualExplainer(
        model,
        [query, [32.0, exemplar, 7.0]],
        ["ordinary", "x", "fixed"],
        continuous_features=["x", "ordinary"],
        feature_ranges={"x": bounds, "ordinary": (0.0, 64.0)},
        random_state=seed,
        proximity_weight=0.0,
        diversity_weight=0.0,
    )
    explainer.explain(query, desired_class=1, num_counterfactuals=2, max_attempts=2)

    rng = np.random.default_rng(seed)
    rng.integers(1)
    z_x, z_ordinary = rng.standard_normal(2)
    lower, upper = map(Fraction, bounds)
    proposed = Fraction(exemplar) + Fraction(z_x) * Fraction(0.15) * (upper - lower)
    expected_x = float(min(upper, max(lower, proposed)))
    expected_ordinary = np.clip(32.0 + z_ordinary * (0.15 * 64.0), 0.0, 64.0)
    scale = max(map(abs, bounds))
    assert any(
        abs(x / scale - expected_x / scale) <= 1e-15
        and ordinary == expected_ordinary
        and fixed == 7.0
        for ordinary, x, fixed in model.visited
    )
