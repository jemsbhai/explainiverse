"""Counterfactual categorical domains consist of the codes observed in training."""

import numpy as np
import pytest

from explainiverse.explainers import CounterfactualExplainer


class _Classifier:
    prediction_output_kind = "probabilities"

    def predict(self, rows):
        positive = rows[:, 0] == 1.0
        return np.column_stack((~positive, positive)).astype(float)


def _explainer(categorical):
    return CounterfactualExplainer(
        _Classifier(),
        np.array([[0.0], [2.0**-46], [1.0]]),
        ["x"],
        categorical_features=["x"] if categorical else [],
    )


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(np.nextafter(0.0, 1.0), id="adjacent-to-zero"),
        pytest.param(2.0**-47, id="between-close-codes"),
        pytest.param(np.nextafter(1.0, 0.0), id="adjacent-to-one"),
        pytest.param(0.5, id="far-from-observations"),
    ],
)
def test_unobserved_categorical_query_is_rejected(value):
    with pytest.raises(ValueError, match="categorical feature 'x' was not observed"):
        _explainer(categorical=True).explain(np.array([value]), num_counterfactuals=1)


@pytest.mark.parametrize(
    "categorical,value",
    [
        pytest.param(True, -0.0, id="signed-zero"),
        pytest.param(True, 2.0**-46, id="observed-close-code"),
        pytest.param(True, 1.0, id="observed-endpoint"),
        pytest.param(False, 2.0**-47, id="intermediate-continuous-value"),
    ],
)
def test_valid_query_keeps_its_value(categorical, value):
    result = (
        _explainer(categorical)
        .explain(np.array([value]), num_counterfactuals=1, max_attempts=1)
        .explanation_data
    )
    assert result["original_class"] == int(value == 1.0)
    assert result["num_generated"] == 1
    assert result["changes"][0]["x"]["original"] == value
