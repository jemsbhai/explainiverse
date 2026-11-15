"""Projecting observed categorical codes keeps the search free of overflow."""

import warnings

import numpy as np
import pytest

from explainiverse.explainers import CounterfactualExplainer

_MAX = float(np.finfo(float).max)


class _CategoryClassifier:
    prediction_output_kind = "probabilities"

    def __init__(self, targets):
        self.targets = targets
        self.visited = set()

    def predict(self, rows):
        codes = rows[:, 0]
        if len(codes) == 1:
            self.visited.add(float(codes[0]))
        target = np.isin(codes, self.targets)
        return np.column_stack((~target, target)).astype(float)


@pytest.mark.parametrize("overflow", ["warn", "raise"])
@pytest.mark.parametrize(
    "codes",
    [
        pytest.param((-32.0, 32.0), id="ordinary"),
        pytest.param((-_MAX, _MAX), id="opposite-extremes"),
        pytest.param((-_MAX / 2, _MAX), id="asymmetric"),
    ],
)
def test_projects_target_exemplars_without_overflow(codes, overflow):
    model = _CategoryClassifier(codes)
    explainer = CounterfactualExplainer(
        model, [[0.0], [codes[0]], [codes[1]]], ["code"], categorical_features=["code"]
    )
    with warnings.catch_warnings(), np.errstate(over=overflow):
        warnings.simplefilter("error", RuntimeWarning)
        data = explainer.explain(
            [0.0], desired_class=1, num_counterfactuals=2, max_attempts=2
        ).explanation_data

    assert {candidate[0] for candidate in data["counterfactuals"]} == set(codes)
    assert {changes["code"]["change"] for changes in data["changes"]} == set(codes)
    assert data["all_counterfactuals_valid"] is True


@pytest.mark.parametrize("limit", [32.0, _MAX], ids=["ordinary", "opposite-extremes"])
def test_projects_randomly_sampled_categories_without_overflow(limit):
    model = _CategoryClassifier([])
    explainer = CounterfactualExplainer(
        model,
        [[0.0], [-limit], [limit]],
        ["code"],
        categorical_features=["code"],
        random_state=0,
    )
    with np.errstate(over="raise"):
        data = explainer.explain([0.0], desired_class=1, max_attempts=1).explanation_data

    # With no target exemplar, seed zero draws the last of the three sorted codes.
    assert limit in model.visited
    assert data["counterfactuals"] == []
    assert data["search_succeeded"] is False
