"""A budget boundary can leave selected AnchorTabular predicates entirely unsampled."""

import numpy as np
import pytest

from explainiverse.explainers import AnchorTabularExplainer


class _SignClassifier:
    prediction_output_kind = "scores"

    def __init__(self):
        self.calls = []

    def predict(self, rows):
        self.calls.append(rows.copy())
        x = rows[:, 0]
        return np.column_stack((x, -x))


def _explain(background, query, budget, **kwargs):
    model = _SignClassifier()
    result = (
        AnchorTabularExplainer(
            model,
            background,
            ["x", "z"][: len(query)],
            ["positive", "negative"],
            max_samples=budget,
            batch_size=8,
            random_state=21,
            **kwargs,
        )
        .explain(query)
        .explanation_data
    )

    np.testing.assert_array_equal(model.calls[0], [query])
    perturbation_rows = sum(len(batch) for batch in model.calls[1:])
    assert result["query_prediction_rows"] == 1
    assert result["perturbation_prediction_rows"] == perturbation_rows <= budget
    assert result["total_prediction_rows"] == perturbation_rows + 1
    assert result["prediction_calls"] == len(model.calls)
    return result, model.calls[1:]


@pytest.mark.parametrize(
    "budget,max_depth,exhausted",
    [
        (1, None, True),
        (8, None, True),
        (16, None, True),
        (31, None, True),
        (32, None, True),
        (33, None, True),
        (32, 1, False),
    ],
)
def test_zero_precision_tie_keeps_the_full_coverage_empty_rule(budget, max_depth, exhausted):
    background = np.array([[-4.0], [-3.0], [-2.0], [-1.0]])
    query = np.array([2.0])
    # The sign of x determines the class. Every supported rule has precision zero,
    # so the empty rule wins the tie with all four background rows in its support.
    assert np.all(background[:, 0] < 0) and query[0] > 0
    result, batches = _explain(background, query, budget, max_anchor_size=max_depth)

    assert result["rules"] == []
    assert result["coverage"] == 1.0
    assert result["precision"] == result["precision_lower_bound"] == 0.0
    assert result["precision_success_count"] == 0
    assert result["precision_sample_count"] == len(batches[0])
    assert result["is_certified_anchor"] is False
    assert result["provides_high_probability_precision_guarantee"] is False
    assert result["perturbation_prediction_rows"] == budget
    assert result["budget_exhausted"] is exhausted
    assert result["bounded_beam_search_completed"] is (not exhausted)
    assert result["all_lucb_stages_converged"] is True
    expected_reason = (
        "sample_budget_exhausted"
        if exhausted
        else "bounded_beam_search_completed_without_certified_anchor"
    )
    assert result["termination_reason"] == expected_reason


def test_budget_boundary_preserves_a_better_sampled_nonempty_fallback():
    background = np.array([[-4.0], [-3.0], [-2.0], [1.0]])
    result, batches = _explain(background, np.array([2.0]), 40)

    # The first batch is unconditional. The remaining draws condition on the
    # first quartile predicate. Count the observed positive scores independently.
    selected_draws = np.concatenate(batches[1:])[:, 0]
    cut = -3.25
    assert np.all(selected_draws > cut)
    successes = np.count_nonzero(selected_draws > 0)
    observed_precision = successes / len(selected_draws)
    assert observed_precision > np.mean(batches[0][:, 0] > 0)
    assert result["rules"] == ["x > -3.25"]
    assert result["coverage"] == np.mean(background[:, 0] > cut)
    assert result["precision"] == observed_precision
    assert result["precision_success_count"] == successes
    assert result["precision_sample_count"] == len(selected_draws)
    assert result["precision_upper_bound"] < result["precision_threshold"]
    assert result["is_certified_anchor"] is False
    assert result["budget_exhausted"] is True
    assert result["termination_reason"] == "sample_budget_exhausted"
    assert result["perturbation_prediction_rows"] == 40


def test_budget_boundary_preserves_an_already_certified_nonempty_rule():
    background = np.array([[-1.0, 0.0], [-1.0, 1.0], [-1.0, 2.0], [1.0, 3.0]])
    result, _ = _explain(background, np.array([2.0, 3.0]), 120, threshold=0.75, max_anchor_size=1)

    # The sign predicate keeps the single positive row. The other feature
    # supplies additional beam candidates still awaiting evaluation.
    assert result["rules"] == ["x > -1"]
    assert result["coverage"] == 1 / 4
    assert result["precision"] == 1.0
    assert result["precision_success_count"] == result["precision_sample_count"] > 0
    assert result["precision_lower_bound"] > result["precision_threshold"]
    assert result["is_certified_anchor"] is True
    assert result["provides_high_probability_precision_guarantee"] is True
    assert result["budget_exhausted"] is True
    assert result["bounded_beam_search_completed"] is False
    assert result["termination_reason"] == "sample_budget_exhausted"
    assert result["perturbation_prediction_rows"] == 120
