"""Public counterfactual searches at the finite limits of binary64."""

import sys
import warnings

import numpy as np
import pytest

from explainiverse.explainers import CounterfactualExplainer


@pytest.mark.parametrize("policy", ["raise", "warn"])
@pytest.mark.parametrize(
    "endpoint",
    [sys.float_info.max, -sys.float_info.max, 1.0],
    ids=["upper-limit", "lower-limit", "ordinary"],
)
def test_optimizer_returns_sole_target_endpoint(endpoint, policy):
    query = endpoint / 2

    class EndpointClassifier:
        prediction_output_kind = "probabilities"

        def predict(self, rows):
            assert np.geterr()["over"] == policy
            target = rows[:, 0] == endpoint
            # A constant target probability leaves the objective flat.
            return np.column_stack(
                (
                    np.where(target, 5 / 16, 1 / 2),
                    np.full(len(rows), 3 / 8),
                    np.where(target, 5 / 16, 1 / 8),
                )
            )

    with warnings.catch_warnings(), np.errstate(over=policy):
        warnings.simplefilter("error", RuntimeWarning)
        explainer = CounterfactualExplainer(
            EndpointClassifier(),
            [[query], [endpoint]],
            ["x"],
            feature_ranges={"x": sorted((0.0, endpoint))},
            proximity_weight=0.0,
            diversity_weight=0.0,
        )
        data = explainer.explain(
            [query], desired_class=1, num_counterfactuals=1, max_attempts=1
        ).explanation_data

    assert data["counterfactuals"] == [[endpoint]]
    assert data["changes"] == [
        {"x": {"original": query, "counterfactual": endpoint, "change": query}}
    ]
    assert data["feature_attributions"] == {"x": 0.5}
    assert data["normalized_squared_distances"] == [0.25]
    assert data["diversity_loss"] == 0.0


@pytest.mark.parametrize("endpoint", [sys.float_info.max, 1.0], ids=["upper-limit", "ordinary"])
def test_optimizer_can_still_move_another_continuous_feature(endpoint):
    class FixedOrAdjustable:
        prediction_output_kind = "probabilities"

        def predict(self, rows):
            adjustable, fixed = rows[:, 1], rows[:, 2]
            return np.where(fixed == 1, 0.75, 0.25 + 0.5 * adjustable)

    query = [endpoint / 2, 0.0, 0.0]
    explainer = CounterfactualExplainer(
        FixedOrAdjustable(),
        [query, [endpoint, 0.0, 1.0]],
        ["x", "adjustable", "fixed"],
        continuous_features=["x", "adjustable"],
        feature_ranges={"x": (0.0, endpoint), "adjustable": (0.0, 1.0)},
        proximity_weight=0.0,
        diversity_weight=0.0,
    )
    with np.errstate(over="raise"):
        data = explainer.explain(
            query, desired_class=1, num_counterfactuals=1, max_attempts=1
        ).explanation_data

    # The projected training target loses its fixed flag. With one attempt,
    # advancing the adjustable coordinate past one half is the only solution.
    (row,) = data["counterfactuals"]
    assert 0.0 <= row[0] <= endpoint
    assert 0.5 < row[1] <= 1.0
    assert row[2] == 0.0
