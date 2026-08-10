"""Primary-formula and failure-contract tests for complexity metrics."""

import numpy as np
import pytest
from scipy.stats import entropy as scipy_entropy

from explainiverse.core.explanation import Explanation
from explainiverse.evaluation import (
    compute_attribution_threshold_count,
    compute_batch_complexity,
    compute_complexity,
    compute_effective_complexity,
    compute_sparseness,
)


class FixedExplainer:
    def __init__(self, values, names=None):
        self.values = list(values)
        self.names = names or [f"f{i}" for i in range(len(self.values))]

    def explain(self, instance):
        return Explanation(
            explainer_name="fixed",
            target_class="0",
            explanation_data={"feature_attributions": dict(zip(self.names, self.values))},
            feature_names=list(self.names),
        )


def test_bhatt_complexity_uses_primary_natural_log_formula():
    magnitudes = np.array([1.0, 2.0, 3.0])
    probabilities = magnitudes / magnitudes.sum()
    result = compute_complexity(FixedExplainer(magnitudes), np.zeros(3))

    assert result == pytest.approx(-np.sum(probabilities * np.log(probabilities)))
    assert result == pytest.approx(scipy_entropy(probabilities))
    assert result != pytest.approx(-np.sum(probabilities * np.log2(probabilities)))


def test_complexity_and_gini_remain_scale_invariant_for_subnormal_magnitudes():
    values = np.array([1.0, 2.0, 3.0])
    tiny_values = values * 1e-310
    x = np.zeros(3)

    assert compute_complexity(FixedExplainer(tiny_values), x) == pytest.approx(
        compute_complexity(FixedExplainer(values), x)
    )
    assert compute_sparseness(FixedExplainer(tiny_values), x) == pytest.approx(
        compute_sparseness(FixedExplainer(values), x)
    )


def test_all_zero_complexity_is_explicitly_undefined():
    with pytest.raises(ValueError, match="fractional contribution distribution"):
        compute_complexity(FixedExplainer([0.0, 0.0]), np.zeros(2))


class MissingFeatureExplainer:
    def explain(self, instance):
        return Explanation(
            explainer_name="missing",
            target_class="0",
            explanation_data={"feature_attributions": {"f0": 1.0}},
            feature_names=["f0", "f1"],
        )


def test_missing_attribution_is_not_silently_replaced_with_zero():
    with pytest.raises(ValueError, match="match feature_names exactly"):
        compute_sparseness(MissingFeatureExplainer(), np.zeros(2))


@pytest.mark.parametrize(
    "explainer,match",
    [
        (FixedExplainer([1.0]), "returned 1 attributions"),
        (FixedExplainer([1.0, np.inf]), "only finite"),
    ],
)
def test_invalid_attribution_contracts_fail(explainer, match):
    with pytest.raises(ValueError, match=match):
        compute_sparseness(explainer, np.zeros(2))


def test_matrix_is_not_flattened_into_a_fake_single_instance():
    with pytest.raises(ValueError, match="one-dimensional"):
        compute_sparseness(FixedExplainer([1.0, 2.0]), np.zeros((1, 2)))


class FailsOnSecondRow:
    def __init__(self):
        self.calls = 0

    def explain(self, instance):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("deliberate explanation failure")
        return FixedExplainer([1.0, 2.0]).explain(instance)


def test_batch_does_not_hide_failed_explanations_or_bias_aggregate():
    with pytest.raises(RuntimeError, match="deliberate explanation failure"):
        compute_batch_complexity(FailsOnSecondRow(), np.zeros((3, 2)))


@pytest.mark.parametrize("max_instances", [0, -1, 1.5, True])
def test_batch_rejects_invalid_max_instances(max_instances):
    with pytest.raises(ValueError, match="positive integer"):
        compute_batch_complexity(
            FixedExplainer([1.0, 2.0]),
            np.zeros((2, 2)),
            max_instances=max_instances,
        )


def test_threshold_count_has_honest_name_and_legacy_alias_warns():
    explainer = FixedExplainer([0.0, 0.2, 0.8])
    x = np.zeros(3)
    expected = compute_attribution_threshold_count(explainer, x, threshold=0.1)
    assert expected == 2.0

    with pytest.warns(FutureWarning, match="not Nguyen"):
        observed = compute_effective_complexity(explainer, x, threshold=0.1)
    assert observed == expected


@pytest.mark.parametrize("threshold", [-1.0, np.nan, np.inf])
def test_threshold_count_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="finite non-negative"):
        compute_attribution_threshold_count(
            FixedExplainer([1.0, 2.0]), np.zeros(2), threshold=threshold
        )
