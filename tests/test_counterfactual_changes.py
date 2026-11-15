"""A counterfactual edit list must reproduce the returned target-class candidate."""

import numpy as np
import pytest

from explainiverse.explainers import CounterfactualExplainer


class _EndpointClassifier:
    prediction_output_kind = "probabilities"

    def __init__(self, endpoint):
        self.endpoint = endpoint

    def predict(self, rows):
        target = rows[:, 0] == self.endpoint
        return np.column_stack((~target, target)).astype(float)


@pytest.mark.parametrize("feature_type", ["categorical", "continuous"])
@pytest.mark.parametrize(
    "endpoint",
    [
        pytest.param(-1e-9, id="negative-subthreshold"),
        pytest.param(1e-7, id="old-cutoff"),
        pytest.param(np.nextafter(1e-7, np.inf), id="above-old-cutoff"),
    ],
)
def test_changes_reconstruct_target_candidate(feature_type, endpoint):
    # The endpoint is the only possible counterfactual, for either domain.
    query = np.array([0.0, 5.0])
    candidate = np.array([endpoint, 5.0])
    model = _EndpointClassifier(endpoint)
    explainer = CounterfactualExplainer(
        model,
        np.stack((query, candidate)),
        ["x", "fixed"],
        continuous_features=["x"] if feature_type == "continuous" else [],
        categorical_features=["x"] if feature_type == "categorical" else [],
    )

    result = explainer.explain(
        query, desired_class=1, num_counterfactuals=1, max_attempts=1
    ).explanation_data

    assert result["original_class"] == 0
    assert result["counterfactuals"] == [candidate.tolist()]
    assert result["counterfactual_predictions"] == [[0.0, 1.0]]
    assert result["changes"] == [
        {"x": {"original": 0.0, "counterfactual": endpoint, "change": endpoint}}
    ]
