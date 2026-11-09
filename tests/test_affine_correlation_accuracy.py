"""Regression tests for exact output-change correlation contracts."""

from decimal import Decimal

import numpy as np
import pytest
from scipy.stats import pearsonr

from explainiverse.core.explanation import Explanation
from explainiverse.evaluation import (
    compute_faithfulness_correlation,
    compute_faithfulness_estimate,
    compute_monotonicity_nguyen,
)
from explainiverse.evaluation._utils import (
    _stable_pearson_affine,
    _stable_pearson_decimal_affine,
    _stable_spearman_affine,
)


class _ConfidentClassifier:
    task = "classification"
    classes_ = np.array([0, 1])

    def __init__(self, removal_probabilities):
        self.removal_probabilities = np.asarray(removal_probabilities)

    def predict_proba(self, X):
        probabilities = []
        for row in X:
            removed = np.flatnonzero(row == 0.0)
            assert removed.size <= 1
            probabilities.append(self.removal_probabilities[removed[0]] if removed.size else 0.9)
        probabilities = np.asarray(probabilities)
        return np.column_stack((1.0 - probabilities, probabilities))


@pytest.mark.parametrize(
    "metric",
    [compute_faithfulness_estimate, compute_faithfulness_correlation, compute_monotonicity_nguyen],
)
@pytest.mark.parametrize("probabilities", [[2e-30, 3e-30, 4e-30], [2e-28, 3e-28, 3.1e-28]])
def test_confident_classifier_keeps_distinct_removal_effects(metric, probabilities):
    attributions = np.array([1.0, 2.0, 3.0])
    probabilities = np.asarray(probabilities)
    explanation = Explanation(
        explainer_name="fixed",
        target_class=1,
        explanation_data={"feature_attributions": dict(zip(["f0", "f1", "f2"], attributions))},
        feature_names=["f0", "f1", "f2"],
    )
    # A common offset cannot change correlation. Evaluate an independent
    # reference on scaled post-removal probabilities, avoiding subtraction.
    expected = (
        -1.0
        if metric is compute_monotonicity_nguyen
        else -pearsonr(attributions, probabilities / probabilities.max()).statistic
    )

    assert metric(
        _ConfidentClassifier(probabilities), np.ones(3), explanation, baseline=0.0
    ) == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("anchor", [-0.9, 0.9])
@pytest.mark.parametrize("absolute", [False, True])
def test_affine_correlations_preserve_close_effects_on_both_sides_of_anchor(anchor, absolute):
    attributions = np.array([1.0, 2.0, 3.0])
    predictions = np.array([2e-28, 3e-28, 3.1e-28])
    direction = 1.0 if absolute and anchor < 0.0 else -1.0
    pearson = pearsonr(attributions, predictions / predictions.max()).statistic

    assert _stable_pearson_affine(
        attributions, anchor, predictions, absolute=absolute
    ) == pytest.approx(direction * pearson, abs=1e-14)
    assert _stable_spearman_affine(
        attributions, anchor, predictions, absolute=absolute
    ) == pytest.approx(direction, abs=1e-14)


@pytest.mark.parametrize("anchor", [-0.9, 0.9])
def test_decimal_aggregate_correlation_preserves_close_effects(anchor):
    attributions = [Decimal(1), Decimal(2), Decimal(3)]
    predictions = np.array([2e-28, 3e-28, 3.1e-28])
    expected = -pearsonr([1.0, 2.0, 3.0], predictions / predictions.max()).statistic

    assert _stable_pearson_decimal_affine(attributions, anchor, predictions) == pytest.approx(
        expected, abs=1e-14
    )


@pytest.mark.parametrize("anchor_sign", [-1.0, 1.0])
def test_affine_correlations_preserve_full_binary64_exponent_range(anchor_sign):
    anchor = anchor_sign * np.finfo(np.float64).max
    minimum_subnormal = np.nextafter(0.0, 1.0)
    predictions = np.array([0.0, minimum_subnormal, 2.0 * minimum_subnormal])
    attributions = np.array([1.0, 2.0, 3.0])

    assert _stable_pearson_affine(attributions, anchor, predictions) == -1.0
    assert (
        _stable_pearson_decimal_affine([Decimal(1), Decimal(2), Decimal(3)], anchor, predictions)
        == -1.0
    )
    for helper in [_stable_pearson_affine, _stable_spearman_affine]:
        assert helper(attributions, anchor, predictions, absolute=True) == -anchor_sign


@pytest.mark.parametrize(
    "helper", [_stable_pearson_affine, _stable_pearson_decimal_affine, _stable_spearman_affine]
)
def test_affine_correlations_still_reject_genuinely_constant_effects(helper):
    attributions = (
        [Decimal(1), Decimal(2), Decimal(3)]
        if helper is _stable_pearson_decimal_affine
        else np.array([1.0, 2.0, 3.0])
    )
    with pytest.raises(ValueError, match="constant"):
        helper(attributions, 0.9, np.full(3, 2e-30))


@pytest.mark.parametrize("helper", [_stable_pearson_affine, _stable_spearman_affine])
def test_absolute_affine_correlations_reject_equal_effect_magnitudes(helper):
    with pytest.raises(ValueError, match="constant"):
        helper(np.array([1.0, 2.0]), 0.5, np.array([0.25, 0.75]), absolute=True)
