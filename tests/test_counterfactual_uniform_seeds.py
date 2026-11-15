"""Public counterfactual searches sample finite domains without target exemplars."""

from fractions import Fraction

import numpy as np
import pytest

from explainiverse.explainers import CounterfactualExplainer

_MAX = float(np.finfo(float).max)


class _RecordingNegativeClassifier:
    prediction_output_kind = "probabilities"

    def __init__(self):
        self.visited = []

    def predict(self, rows):
        self.visited.extend(map(tuple, rows))
        return np.tile([1.0, 0.0], (len(rows), 1))


@pytest.mark.parametrize(
    "bounds",
    [
        pytest.param((-32.0, 32.0), id="ordinary"),
        pytest.param((-_MAX, _MAX), id="symmetric"),
        pytest.param((-_MAX, _MAX / 2), id="longer-negative-side"),
        pytest.param((-_MAX / 2, _MAX), id="longer-positive-side"),
    ],
)
def test_exhausted_search_visits_each_uniform_draw_with_its_other_coordinates(bounds):
    model = _RecordingNegativeClassifier()
    training = np.array([[0.0, 4.0, 0.0, 7.0], [0.0, 4.0, 2.0, 7.0]])
    explainer = CounterfactualExplainer(
        model,
        training,
        ["x", "y", "category", "fixed"],
        continuous_features=["x", "y"],
        categorical_features=["category"],
        feature_ranges={"x": bounds, "y": (4.0, 8.0)},
        proximity_weight=0.0,
        random_state=42,
    )
    attempts = 4
    data = explainer.explain(training[0], desired_class=1, max_attempts=attempts).explanation_data
    assert data["search_succeeded"] is False
    assert data["counterfactuals"] == []

    # Recover the independent draws for each attempt. The public model must
    # have evaluated each corresponding row, including the categorical draw.
    lower, upper = map(Fraction, bounds)
    scale = max(abs(value) for value in bounds)
    rng = np.random.default_rng(42)
    for _ in range(attempts):
        u = Fraction(rng.random())
        expected_x = float((1 - u) * lower + u * upper)
        expected_y = rng.uniform(4.0, 8.0)
        expected_category = [0.0, 2.0][rng.integers(2)]
        assert any(
            abs(x / scale - expected_x / scale) <= 1e-15
            and (y, category, fixed) == (expected_y, expected_category, 7.0)
            for x, y, category, fixed in model.visited
        )
    assert all(bounds[0] <= x <= bounds[1] for x, _, _, _ in model.visited)


@pytest.mark.parametrize("limit", [32.0, _MAX], ids=["ordinary", "wide"])
def test_uniform_seed_can_find_target_absent_from_training(limit):
    class ThresholdClassifier:
        prediction_output_kind = "probabilities"

        def predict(self, rows):
            target = rows[:, 0] > limit / 8
            return np.column_stack((~target, target)).astype(float)

    explainer = CounterfactualExplainer(
        ThresholdClassifier(),
        [[0.0]],
        ["x"],
        feature_ranges={"x": (-limit, limit)},
        random_state=0,
        proximity_weight=0.0,
    )
    # The first uniform quantile for seed zero exceeds 9/16, so its
    # coordinate in [-limit, limit] exceeds the target threshold.
    data = explainer.explain(
        [0.0], desired_class=1, num_counterfactuals=1, max_attempts=1
    ).explanation_data
    assert data["search_succeeded"] is True
    (candidate,) = data["counterfactuals"]
    assert limit / 8 < candidate[0] <= limit
