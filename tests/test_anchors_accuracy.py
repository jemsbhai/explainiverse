"""Accuracy-oracle tests for the approximate Anchors-style explainer."""

import numpy as np

from explainiverse.explainers.rule_based.anchors_wrapper import AnchorsExplainer


class _BinaryRuleModel:
    """Small probability adapter whose positive class is defined by a rule."""

    def __init__(self, rule):
        self.rule = rule

    def predict(self, data):
        data = np.asarray(data)
        positive = np.asarray(self.rule(data), dtype=float)
        return np.column_stack((1.0 - positive, positive))


def _binary_training_data(repeats=64):
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
    return AnchorsExplainer(
        model=model,
        training_data=_binary_training_data(),
        feature_names=["signal", "noise"],
        class_names=["negative", "positive"],
        threshold=kwargs.pop("threshold", 0.95),
        n_samples=kwargs.pop("n_samples", 256),
        beam_size=kwargs.pop("beam_size", 2),
        random_state=kwargs.pop("random_state", 7),
        **kwargs,
    )


def test_relevant_singleton_beats_irrelevant_superset():
    """A sufficient singleton must not be replaced by a lower-coverage superset."""
    model = _BinaryRuleModel(lambda data: data[:, 0] >= 0.5)
    explainer = _make_explainer(model)

    result = explainer.explain(np.array([1.0, 1.0])).explanation_data

    assert result["anchor_indices"] == [0]
    assert result["anchor_features"] == ["signal"]
    assert result["precision"] == 1.0
    assert result["coverage"] == 0.5
    assert result["meets_empirical_precision_threshold"] is True
    assert result["feature_attributions"] == {"signal": 1.0}
    assert result["feature_attribution_semantics"] == "anchor_membership_indicator"


def test_constant_classifier_returns_empty_anchor():
    """The empty rule is the maximum-coverage sufficient condition."""
    model = _BinaryRuleModel(lambda data: np.ones(len(data), dtype=bool))
    explainer = _make_explainer(model)

    explanation = explainer.explain(np.array([1.0, 1.0]))
    result = explanation.explanation_data

    assert explanation.explainer_name == "ApproximateAnchors"
    assert result["anchor_indices"] == []
    assert result["anchor_features"] == []
    assert result["rules"] == []
    assert result["precision"] == 1.0
    assert result["coverage"] == 1.0
    assert result["meets_empirical_precision_threshold"] is True


def test_valid_candidates_are_ranked_by_coverage_before_precision(monkeypatch):
    """Among threshold-meeting candidates, optimize empirical coverage."""
    model = _BinaryRuleModel(lambda data: data[:, 0] >= 0.5)
    explainer = _make_explainer(model)

    precision = {
        (): 0.50,
        (0,): 0.96,
        (1,): 1.00,
        (0, 1): 1.00,
    }
    coverage = {
        (): 1.00,
        (0,): 0.75,
        (1,): 0.25,
        (0, 1): 0.20,
    }

    monkeypatch.setattr(
        explainer,
        "_compute_precision",
        lambda instance, anchor, target_class: (
            precision[tuple(sorted(anchor))],
            int(256 * precision[tuple(sorted(anchor))]),
        ),
    )
    monkeypatch.setattr(
        explainer,
        "_compute_coverage",
        lambda anchor, instance: coverage[tuple(sorted(anchor))],
    )

    anchor, empirical_precision, empirical_coverage = explainer._beam_search(
        np.array([1.0, 1.0]), target_class=1
    )

    assert anchor == [0]
    assert empirical_precision == 0.96
    assert empirical_coverage == 0.75


def test_result_discloses_fixed_sample_heuristic_without_guarantee():
    model = _BinaryRuleModel(lambda data: data[:, 0] >= 0.5)
    explainer = _make_explainer(model)

    explanation = explainer.explain(np.array([1.0, 0.0]))
    result = explanation.explanation_data

    assert explanation.explainer_name == "ApproximateAnchors"
    assert result["search_method"] == "fixed_sample_beam_search"
    assert result["provides_high_probability_guarantee"] is False
    assert result["claim_status"] == "quarantined"
    assert result["promotion_requires_sequential_confidence_certificate"] is True
    assert result["budget_exhaustion_is_certified"] is False
    assert result["precision_sample_size"] == 256


def test_seeded_explanations_are_repeatable():
    model = _BinaryRuleModel(lambda data: data[:, 0] >= 0.5)
    explainer = _make_explainer(model, n_samples=101)

    first = explainer.explain(np.array([1.0, 0.0])).explanation_data
    second = explainer.explain(np.array([1.0, 0.0])).explanation_data

    assert first == second


def test_small_thresholds_remain_exact_and_human_readable():
    training = np.tile(np.array([[0.0], [0.001], [0.002], [0.003]]), (64, 1))
    model = _BinaryRuleModel(lambda data: data[:, 0] <= 0.0015)
    explainer = AnchorsExplainer(
        model=model,
        training_data=training,
        feature_names=["tiny"],
        class_names=["negative", "positive"],
        threshold=0.95,
        n_samples=128,
        beam_size=1,
        random_state=4,
    )

    data = explainer.explain(np.array([0.001])).explanation_data
    condition = data["rule_conditions"][0]

    assert data["rules"] == ["0.00075 < tiny <= 0.0015"]
    assert condition["label"] == data["rules"][0]
    assert condition["lower_bound"] == 0.00075
    assert condition["upper_bound"] == 0.0015
    assert condition["lower_bound"] < condition["upper_bound"]
    assert condition["lower_inclusive"] is False
    assert condition["upper_inclusive"] is True
