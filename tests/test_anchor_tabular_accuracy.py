"""Accuracy and contract tests for confidence-certified tabular Anchors."""

import math

import numpy as np
import pytest

from explainiverse.explainers.rule_based.anchor_tabular import (
    AnchorTabularExplainer,
    _CandidateStatistics,
    _PredictionBudget,
)


class _BinaryRuleModel:
    task = "classification"
    classes_ = np.array([10, 20])

    def __init__(self, rule, *, hard_labels=False):
        self.rule = rule
        self.hard_labels = hard_labels

    def predict(self, data):
        values = np.asarray(data)
        positive = np.asarray(self.rule(values), dtype=bool)
        if self.hard_labels:
            return np.where(positive, 20, 10)
        probability = positive.astype(float)
        return np.column_stack((1.0 - probability, probability))


class _RegressionModel:
    task = "regression"

    def predict(self, data):
        return np.asarray(data)[:, 0]


class _InputMutatingRuleModel(_BinaryRuleModel):
    def predict(self, data):
        values = np.asarray(data)
        positive = np.asarray(self.rule(values), dtype=bool)
        values[...] = 0.0
        probability = positive.astype(float)
        return np.column_stack((1.0 - probability, probability))


def _binary_background(repeats=32):
    return np.tile(
        np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        ),
        (repeats, 1),
    )


def _make_explainer(model, **kwargs):
    return AnchorTabularExplainer(
        model=model,
        background_data=kwargs.pop("background_data", _binary_background()),
        feature_names=kwargs.pop("feature_names", ["signal", "noise"]),
        class_names=kwargs.pop("class_names", ["negative", "positive"]),
        threshold=kwargs.pop("threshold", 0.75),
        delta=kwargs.pop("delta", 0.1),
        epsilon=kwargs.pop("epsilon", 0.1),
        beam_size=kwargs.pop("beam_size", 2),
        batch_size=kwargs.pop("batch_size", 8),
        max_samples=kwargs.pop("max_samples", 10_000),
        random_state=kwargs.pop("random_state", 7),
        **kwargs,
    )


def test_certified_singleton_recovers_the_sufficient_feature():
    model = _BinaryRuleModel(lambda data: data[:, 0] >= 0.5)

    explanation = _make_explainer(model).explain(np.array([1.0, 1.0]))
    result = explanation.explanation_data

    assert explanation.explainer_name == "AnchorTabular"
    assert explanation.target_class == "positive"
    assert result["rules"] == ["signal > 0"]
    assert result["anchor_feature_indices"] == [0]
    assert result["anchor_features"] == ["signal"]
    assert result["feature_attributions"] == {"signal": 1.0, "noise": 0.0}
    assert result["precision"] == 1.0
    assert result["precision_lower_bound"] > result["precision_threshold"]
    assert result["coverage"] == 0.5
    assert result["is_certified_anchor"] is True
    assert result["provides_high_probability_precision_guarantee"] is True
    assert result["globally_maximum_coverage_claim"] is False
    assert result["causal_claim"] is False


def test_constant_classifier_certifies_the_empty_maximum_coverage_anchor():
    model = _BinaryRuleModel(lambda data: np.ones(len(data), dtype=bool))

    result = _make_explainer(model).explain(np.array([1.0, 0.0])).explanation_data

    assert result["rules"] == []
    assert result["anchor_feature_indices"] == []
    assert result["feature_attributions"] == {"signal": 0.0, "noise": 0.0}
    assert result["precision"] == 1.0
    assert result["coverage"] == 1.0
    assert result["is_certified_anchor"] is True
    assert result["termination_reason"] == "certified_empty_anchor"


def test_one_row_budget_never_turns_a_lucky_draw_into_a_certificate():
    model = _BinaryRuleModel(lambda data: data[:, 0] >= 0.5)

    result = (
        _make_explainer(model, max_samples=1, batch_size=1)
        .explain(np.array([1.0, 1.0]))
        .explanation_data
    )

    assert result["is_certified_anchor"] is False
    assert result["provides_high_probability_precision_guarantee"] is False
    assert result["budget_exhausted"] is True
    assert result["termination_reason"] == "sample_budget_exhausted"
    assert result["perturbation_prediction_rows"] == 1
    assert result["query_prediction_rows"] == 1
    assert result["total_prediction_rows"] == 2
    assert result["precision_sample_count"] == 1


def test_certification_is_strict_when_lower_bound_equals_threshold():
    model = _BinaryRuleModel(lambda data: np.ones(len(data), dtype=bool))
    preliminary = _make_explainer(model)
    certification_delta, _, _ = preliminary._confidence_allocations(2, 2)
    statistics = _CandidateStatistics(successes=10, samples=10)
    exact_lower, _ = preliminary._candidate_bounds(statistics, certification_delta)
    explainer = _make_explainer(model, threshold=exact_lower)
    exhausted_budget = _PredictionBudget(max_samples=1, samples_used=1)

    status, lower, _ = explainer._certify_candidate(
        (),
        statistics,
        [],
        target_index=1,
        certification_delta=certification_delta,
        rng=np.random.default_rng(0),
        budget=exhausted_budget,
    )

    assert lower == explainer.threshold
    assert status == "budget_exhausted"


def test_conditional_sampler_satisfies_every_selected_cut_predicate():
    background = np.column_stack((np.arange(10.0), np.arange(10.0)[::-1]))
    model = _BinaryRuleModel(lambda data: data[:, 0] >= 4.0)
    explainer = _make_explainer(
        model,
        background_data=background,
        feature_names=["ascending", "descending"],
    )
    predicates = explainer._generate_predicates(np.array([4.0, 5.0]))
    ascending = [predicate.index for predicate in predicates if predicate.feature_index == 0]
    rule = tuple(ascending)

    samples = explainer._conditional_sample(rule, predicates, 2_000, np.random.default_rng(11))

    assert len(rule) >= 2
    for predicate_index in rule:
        predicate = predicates[predicate_index]
        assert np.all(predicate.applies(samples[:, predicate.feature_index]))


def test_coverage_is_the_exact_empirical_support_fraction():
    background = np.column_stack((np.arange(10.0), np.zeros(10)))
    model = _BinaryRuleModel(lambda data: data[:, 0] >= 4.0)
    explainer = _make_explainer(
        model,
        background_data=background,
        feature_names=["signal", "constant"],
    )
    predicates = explainer._generate_predicates(np.array([4.0, 0.0]))
    predicate = next(item for item in predicates if item.label == "signal <= 4.5")

    coverage = explainer._coverage((predicate.index,), predicates)

    assert coverage == 0.5
    assert coverage == np.mean(background[:, 0] <= 4.5)


def test_conditional_sampling_preserves_joint_rows_and_feature_dependence():
    background = np.vstack(
        (
            np.tile([0.0, 0.0], (90, 1)),
            np.tile([1.0, 0.0], (1, 1)),
            np.tile([1.0, 1.0], (9, 1)),
        )
    )
    model = _BinaryRuleModel(lambda data: data[:, 1] == 0.0)
    explainer = _make_explainer(model, background_data=background)
    predicates = explainer._generate_predicates(np.array([1.0, 0.0]))
    signal_predicate = next(item for item in predicates if item.feature_index == 0)

    samples = explainer._conditional_sample(
        (signal_predicate.index,), predicates, 5_000, np.random.default_rng(19)
    )

    assert set(map(tuple, samples)) <= set(map(tuple, background))
    assert np.mean(samples[:, 1] == 0.0) == pytest.approx(0.1, abs=0.02)
    assert explainer._coverage((signal_predicate.index,), predicates) == 0.1


def test_seed_is_repeatable_without_mutating_global_numpy_rng():
    model = _BinaryRuleModel(lambda data: data[:, 0] >= 0.5)
    explainer = _make_explainer(model)
    np.random.seed(1234)
    state_before = np.random.get_state()

    first = explainer.explain(np.array([1.0, 0.0])).explanation_data
    state_after = np.random.get_state()
    second = explainer.explain(np.array([1.0, 0.0])).explanation_data

    first_prediction = first.pop("model_prediction")
    second_prediction = second.pop("model_prediction")
    assert np.array_equal(first_prediction, second_prediction)
    assert first == second
    assert state_before[0] == state_after[0]
    assert np.array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]


def test_noncontiguous_hard_labels_map_to_display_names_and_fixed_output():
    model = _BinaryRuleModel(lambda data: data[:, 0] >= 0.5, hard_labels=True)

    explanation = _make_explainer(model).explain(np.array([1.0, 0.0]))

    assert explanation.target_class == "positive"
    assert explanation.explanation_data["target_output_index"] == 1
    assert np.array_equal(explanation.explanation_data["model_prediction"], [0.0, 1.0])


def test_mutating_model_cannot_change_the_caller_query_or_rule_predicates():
    model = _InputMutatingRuleModel(lambda data: data[:, 0] >= 0.5)
    instance = np.array([1.0, 1.0])

    result = _make_explainer(model).explain(instance).explanation_data

    assert np.array_equal(instance, [1.0, 1.0])
    assert result["rules"] == ["signal > 0"]
    assert result["is_certified_anchor"] is True


def test_budget_accounting_is_exact_and_never_overshoots():
    model = _BinaryRuleModel(lambda data: data[:, 0] >= 0.5)
    max_samples = 17

    result = (
        _make_explainer(
            model,
            max_samples=max_samples,
            batch_size=8,
        )
        .explain(np.array([1.0, 1.0]))
        .explanation_data
    )

    assert 0 < result["perturbation_prediction_rows"] <= max_samples
    assert result["total_prediction_rows"] == result["perturbation_prediction_rows"] + 1
    assert result["prediction_calls"] >= 2
    if result["budget_exhausted"]:
        assert max_samples - result["perturbation_prediction_rows"] <= 1


def test_unconverged_lucb_reports_the_atomic_pair_budget_limit():
    model = _BinaryRuleModel(lambda data: np.all(data > 1.5, axis=1))
    explainer = _make_explainer(
        model,
        threshold=0.75,
        epsilon=0.0,
        beam_size=1,
        max_anchor_size=1,
        batch_size=1,
        max_samples=70,
    )

    result = explainer.explain(np.array([2.0, 2.0])).explanation_data

    assert result["all_lucb_stages_converged"] is False
    assert result["budget_exhausted"] is True
    assert result["bounded_beam_search_completed"] is False
    assert result["termination_reason"] == "sample_budget_exhausted"
    assert result["perturbation_prediction_rows"] <= result["max_samples"]


@pytest.mark.parametrize(
    "kwargs, error_type, match",
    [
        ({"background_data": np.array([[0.0 + 1.0j]])}, ValueError, "complex"),
        ({"background_data": np.array([[math.nan]])}, ValueError, "finite"),
        ({"feature_names": ["signal", " "]}, ValueError, "feature_names"),
        ({"threshold": 1.0}, ValueError, "threshold"),
        ({"delta": 0.0}, ValueError, "delta"),
        ({"epsilon": 1.0}, ValueError, "epsilon"),
        ({"batch_size": 0}, ValueError, "batch_size"),
        ({"random_state": -1}, ValueError, "random_state"),
    ],
)
def test_constructor_rejects_invalid_contracts(kwargs, error_type, match):
    model = _BinaryRuleModel(lambda data: data[:, 0] >= 0.5)

    with pytest.raises(error_type, match=match):
        _make_explainer(model, **kwargs)


def test_classifier_only_and_single_instance_contracts_are_enforced():
    with pytest.raises(ValueError, match="classification"):
        _make_explainer(_RegressionModel())

    model = _BinaryRuleModel(lambda data: data[:, 0] >= 0.5)
    explainer = _make_explainer(model)
    with pytest.raises(ValueError, match="single-row"):
        explainer.explain(np.ones((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        explainer.explain(np.array([np.inf, 1.0]))
    with pytest.raises(TypeError, match="Unexpected keyword"):
        explainer.explain(np.array([1.0, 1.0]), target_class=1)


def test_unrepresentable_global_confidence_split_fails_with_actionable_guidance():
    model = _BinaryRuleModel(lambda data: data[:, 0] >= 0.5)
    explainer = _make_explainer(model)

    with pytest.raises(ValueError, match="smaller max_anchor_size"):
        explainer._confidence_allocations(1_100, 1_100)
