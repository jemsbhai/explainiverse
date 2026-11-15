"""Distinct observed category codes are distinct counterfactual alternatives."""

import numpy as np
import pytest

from explainiverse.explainers import CounterfactualExplainer


class _CategoryClassifier:
    prediction_output_kind = "probabilities"

    def __init__(self):
        self.visited = set()

    def predict(self, rows):
        codes = rows[:, 1]
        if len(codes) == 1:
            self.visited.add(float(codes[0]))
        target = codes != -1.0
        return np.column_stack((~target, target)).astype(float)


@pytest.mark.parametrize(
    "codes",
    [
        pytest.param([1.0, 1.0, np.nextafter(1.0, 2.0)], id="adjacent-distinct-codes"),
        pytest.param([0.0, 1e-7, 0.0], id="separated-by-tolerance"),
        pytest.param([0.0, -0.0, 0.0], id="repeated-code"),
        pytest.param([0.0, 0.25, 0.25], id="separated-codes"),
    ],
)
@pytest.mark.parametrize("mixed", [False, True], ids=["categorical", "mixed"])
def test_returns_each_distinct_target_category(codes, mixed):
    # Every feasible category other than -1 is a target. Request more than exist
    # and give the search enough attempts to visit every target exemplar.
    training = np.array([[0.0, code] for code in [-1.0, *codes]])
    model = _CategoryClassifier()
    explainer = CounterfactualExplainer(
        model,
        training,
        ["other", "code"],
        continuous_features=["other"] if mixed else [],
        categorical_features=["code"] if mixed else ["other", "code"],
    )
    result = explainer.explain(
        training[0], desired_class=1, num_counterfactuals=len(codes) + 1, max_attempts=len(codes)
    ).explanation_data

    expected = {(0.0, code) for code in codes}
    assert set(codes).issubset(model.visited)
    assert {tuple(row) for row in result["counterfactuals"]} == expected
    assert result["num_generated"] == len(expected)


@pytest.mark.parametrize("categorical", [False, True], ids=["continuous", "same-category"])
def test_close_continuous_solutions_remain_duplicates(categorical):
    endpoints = np.array([1.0, np.nextafter(1.0, 2.0)])

    class EndpointClassifier:
        prediction_output_kind = "probabilities"

        def predict(self, rows):
            target = np.isin(rows[:, 0], endpoints)
            return np.column_stack((~target, target)).astype(float)

    # The only two target rows differ by one continuous-coordinate ULP.
    training = np.array([[0.0, 0.0], [endpoints[0], 0.0], [endpoints[1], 0.0]])
    explainer = CounterfactualExplainer(
        EndpointClassifier(),
        training,
        ["value", "code"],
        continuous_features=["value"],
        categorical_features=["code"] if categorical else [],
    )
    result = explainer.explain(
        training[0], desired_class=1, num_counterfactuals=2, max_attempts=2
    ).explanation_data

    assert result["num_generated"] == 1
    assert result["counterfactuals"][0][0] in endpoints
