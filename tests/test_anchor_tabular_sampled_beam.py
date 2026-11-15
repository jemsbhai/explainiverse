"""Use all observed selected-arm evidence when the Anchors sampling budget runs out."""

import math

import numpy as np
import pytest

from explainiverse.explainers import AnchorTabularExplainer


class _RecordingClassifier:
    def __init__(self, positive):
        self.positive = positive
        self.calls = []

    def predict(self, rows):
        self.calls.append(rows.copy())
        positive = self.positive(rows).astype(float)
        return np.column_stack((1 - positive, positive))


def _explain(model, background, query, names, budget, **options):
    result = (
        AnchorTabularExplainer(
            model,
            background,
            names,
            ["negative", "positive"],
            beam_size=2,
            max_anchor_size=1,
            max_samples=budget,
            epsilon=0,
            **options,
        )
        .explain(query)
        .explanation_data
    )
    np.testing.assert_array_equal(model.calls[0], [query])
    rows = sum(len(batch) for batch in model.calls[1:])
    assert result["perturbation_prediction_rows"] == rows == budget
    assert result["query_prediction_rows"] == 1
    assert result["total_prediction_rows"] == rows + 1
    assert result["prediction_calls"] == len(model.calls)
    assert result["budget_exhausted"] is True
    assert result["bounded_beam_search_completed"] is False
    assert result["all_lucb_stages_converged"] is False
    assert result["termination_reason"] == "sample_budget_exhausted"
    return result


@pytest.mark.parametrize("signal", [0, 1, 2])
@pytest.mark.parametrize("budget", [11, 12])
def test_sampled_beam_fallback_is_independent_of_signal_position(signal, budget):
    names = ["a", "b", "c"]
    # Preserve the label sequence as we move the informative column.
    background = np.vstack((np.zeros((9, 3)), np.roll(np.eye(3), signal - 1, axis=1)))
    model = _RecordingClassifier(lambda rows: rows[:, signal] == 1)
    result = _explain(model, background, np.ones(3), names, budget, batch_size=8, random_state=1)

    # Every singleton condition has one support row. Only the signal's row
    # shares the query class, and none of the unconditional draws do.
    assert not model.positive(model.calls[1]).any()
    for index, initial_pull in enumerate(model.calls[2:5]):
        np.testing.assert_array_equal(initial_pull, np.eye(3)[[index]])
    signal_draws = np.count_nonzero(model.positive(np.concatenate(model.calls[2:])))
    assert signal_draws > 0
    assert result["rules"] == [f"{names[signal]} > 0"]
    assert result["coverage"] == 1 / len(background)
    assert result["precision"] == 1.0
    assert result["precision_sample_count"] == result["precision_success_count"] == signal_draws
    assert result["precision_lower_bound"] < result["precision_threshold"]
    assert result["is_certified_anchor"] is False
    assert result["provides_high_probability_precision_guarantee"] is False


@pytest.mark.parametrize("budget", [259, 260])
def test_already_sampled_beam_certificate_beats_higher_empirical_mean(budget):
    background = np.arange(64.0)[:, None]
    model = _RecordingClassifier(lambda rows: rows[:, 0] >= 49)
    threshold = 0.55
    result = _explain(
        model,
        background,
        np.array([65.0]),
        ["x"],
        budget,
        threshold=threshold,
        batch_size=64,
        random_state=214,
    )

    # Two unconditional batches precede initialization (one draw per cut),
    # then a paired pull samples the middle and upper quartile arms.
    initial = model.calls[3:6]
    np.testing.assert_array_equal(np.concatenate(initial)[:, 0], [54, 35, 48])
    upper_draws = np.concatenate([initial[2], model.calls[7]])
    assert np.all(upper_draws[:, 0] > 47.25)
    successes = int(np.sum(model.positive(upper_draws)))
    n = len(upper_draws)
    assert (successes, n) == (63, 65)
    p = successes / n
    # Independently test whether the threshold lies below the KL interval.
    # Four hypotheses share half of delta: the empty rule and three cuts.
    log_argument = math.log(405.5 * n**1.1 / (0.05 / 8))
    beta = log_argument + math.log(log_argument)
    kl = p * math.log(p / threshold) + (1 - p) * math.log((1 - p) / (1 - threshold))
    assert p > threshold and n * kl > beta
    assert result["rules"] == ["x > 47.25"]
    assert result["coverage"] == np.mean(background[:, 0] > 47.25)
    assert result["precision_sample_count"] == n
    assert result["precision_success_count"] == successes
    assert result["precision"] == p
    assert result["precision_lower_bound"] > threshold
    assert result["is_certified_anchor"] is True
    assert result["provides_high_probability_precision_guarantee"] is True
