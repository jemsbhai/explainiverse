"""Accuracy and feasibility contracts for constrained counterfactual search."""

import numpy as np
import pytest

from explainiverse.explainers.counterfactual.dice_wrapper import CounterfactualExplainer


class MixedProbabilityModel:
    class_names = ["negative", "positive"]

    def predict(self, X):
        X = np.asarray(X)
        logit = 2.0 * X[:, 0] + 3.0 * (X[:, 1] == 1.0) - 1.0
        positive = 1.0 / (1.0 + np.exp(-logit))
        return np.column_stack((1.0 - positive, positive))


class OneColumnProbabilityModel:
    def predict(self, X):
        X = np.asarray(X)
        positive = 1.0 / (1.0 + np.exp(-5.0 * X[:, 0]))
        return positive[:, None]


class ConstantProbabilityModel:
    def predict(self, X):
        return np.tile(np.array([[0.9, 0.1]]), (len(X), 1))


@pytest.fixture
def mixed_explainer():
    training = np.array(
        [
            [-1.0, 0.0, 5.0],
            [-0.4, 0.0, 6.0],
            [0.2, 0.0, 5.0],
            [-0.5, 1.0, 6.0],
            [0.5, 1.0, 5.0],
            [1.0, 1.0, 6.0],
        ]
    )
    return CounterfactualExplainer(
        MixedProbabilityModel(),
        training_data=training,
        feature_names=["continuous", "category", "fixed"],
        continuous_features=["continuous"],
        categorical_features=["category"],
        random_state=9,
    )


def test_mixed_domain_candidates_are_valid_categorical_and_fixed(mixed_explainer):
    query = np.array([-0.8, 0.0, 5.0])
    explanation = mixed_explainer.explain(query, num_counterfactuals=3, desired_class=1)
    data = explanation.explanation_data

    assert data["algorithm"] == "constrained_multistart_search"
    assert data["is_dice_implementation"] is False
    assert data["all_counterfactuals_valid"] is True
    assert data["num_generated"] >= 1
    assert data["fixed_features"] == ["fixed"]
    assert explanation.target_class == "positive"

    candidates = np.asarray(data["counterfactuals"])
    assert set(candidates[:, 1]).issubset({0.0, 1.0})
    np.testing.assert_allclose(candidates[:, 2], query[2])
    predicted = np.argmax(MixedProbabilityModel().predict(candidates), axis=1)
    np.testing.assert_array_equal(predicted, np.ones(len(candidates), dtype=int))


def test_repeated_calls_are_deterministic(mixed_explainer):
    query = np.array([-0.8, 0.0, 5.0])
    first = mixed_explainer.explain(query, num_counterfactuals=3)
    second = mixed_explainer.explain(query, num_counterfactuals=3)

    assert first.explanation_data["counterfactuals"] == second.explanation_data["counterfactuals"]
    assert (
        first.explanation_data["counterfactual_predictions"]
        == second.explanation_data["counterfactual_predictions"]
    )


def test_one_column_binary_probability_contract_generates_target():
    training = np.linspace(-1.0, 1.0, 11).reshape(-1, 1)
    explainer = CounterfactualExplainer(
        OneColumnProbabilityModel(),
        training,
        ["x"],
        random_state=2,
    )

    explanation = explainer.explain(np.array([-0.75]), num_counterfactuals=1, desired_class=1)
    candidate = np.asarray(explanation.explanation_data["counterfactuals"][0])

    assert candidate[0] > 0.0
    assert explanation.explanation_data["target_class"] == 1
    assert explanation.explanation_data["all_counterfactuals_valid"] is True


def test_impossible_target_reports_empty_result_not_invalid_candidate():
    training = np.linspace(-1.0, 1.0, 7).reshape(-1, 1)
    explainer = CounterfactualExplainer(ConstantProbabilityModel(), training, ["x"], random_state=1)

    explanation = explainer.explain(
        np.array([0.0]),
        num_counterfactuals=2,
        desired_class=1,
        max_attempts=5,
    )
    data = explanation.explanation_data

    assert data["counterfactuals"] == []
    assert data["num_generated"] == 0
    assert data["search_succeeded"] is False
    assert data["all_counterfactuals_valid"] is False
    assert data["failure_reason"] is not None
    assert "feature_attributions" not in data
    assert "feature_attribution_semantics" not in data


@pytest.mark.parametrize("desired", [-1, 2, 0, 1.5, True])
def test_invalid_or_unchanged_desired_class_is_rejected(mixed_explainer, desired):
    query = np.array([-0.8, 0.0, 5.0])
    error = TypeError if desired in {1.5, True} else ValueError
    with pytest.raises(error):
        mixed_explainer.explain(query, desired_class=desired)


def test_non_probability_model_output_is_rejected():
    class LabelModel:
        def predict(self, X):
            return np.full(len(X), 3.0)

    with pytest.raises(ValueError, match="probabilities"):
        CounterfactualExplainer(LabelModel(), np.array([[0.0], [1.0]]), ["x"])


def test_action_values_have_explicit_non_attribution_semantics(mixed_explainer):
    explanation = mixed_explainer.explain(np.array([-0.8, 0.0, 5.0]), num_counterfactuals=2)
    data = explanation.explanation_data

    assert data["feature_attribution_semantics"] == (
        "mean_absolute_normalized_counterfactual_action"
    )
    assert data["feature_attributions"]["fixed"] == 0.0


@pytest.mark.parametrize(
    "query, message",
    [
        (np.array([-2.0, 0.0, 5.0]), "feature range"),
        (np.array([-0.8, 0.5, 5.0]), "categorical feature"),
    ],
)
def test_query_outside_declared_domain_is_rejected(mixed_explainer, query, message):
    with pytest.raises(ValueError, match=message):
        mixed_explainer.explain(query)
