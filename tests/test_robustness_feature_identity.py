"""Robustness comparisons require stable ordered feature identities."""

import numpy as np
import pytest

from explainiverse.core.explanation import Explanation
from explainiverse.evaluation import (
    compute_avg_sensitivity,
    compute_consistency,
    compute_continuity,
    compute_lipschitz_estimate,
    compute_max_sensitivity,
    compute_relative_input_stability,
    compute_relative_output_stability,
    compute_relative_representation_stability,
    compute_relative_stability,
)

_METRICS = [
    "max",
    "average",
    "lipschitz",
    "continuity",
    "ris",
    "rrs",
    "ros",
    "relative",
    "consistency",
]


class _ConstantClass:
    def predict_proba(self, X):
        return np.tile([0.8, 0.2], (len(X), 1))


class _FixedAttributions:
    def __init__(self, mode):
        self.mode = mode
        self.calls = 0

    def explain(self, instance):
        self.calls += 1
        is_anchor = np.array_equal(instance, np.ones(2))
        names = ["a", "b"]
        attributions = {"a": 1.0, "b": 2.0}
        if self.mode == "reorder" and not is_anchor:
            names = ["b", "a"]
        elif self.mode == "rename" and not is_anchor:
            names = ["c", "d"]
            attributions = {"c": 1.0, "d": 2.0}
        elif self.mode == "dictionary_order" and not is_anchor:
            attributions = {"b": 2.0, "a": 1.0}
        elif self.mode in {"canonical_unnamed", "canonical_transition"}:
            names = (
                ["feature_0", "feature_1"]
                if self.mode == "canonical_transition" and is_anchor
                else None
            )
            attributions = (
                {"feature_0": 1.0, "feature_1": 2.0}
                if is_anchor
                else {"feature_1": 2.0, "feature_0": 1.0}
            )
        return Explanation(
            explainer_name="fixed",
            target_class=1 if self.mode == "target_change" and not is_anchor else 0,
            explanation_data={"feature_attributions": attributions},
            feature_names=names,
        )


def _evaluate(metric, explainer):
    instance = np.ones(2)
    model = _ConstantClass()
    sampling = {"n_samples": 5, "seed": 0}
    relative_sampling = {"n_perturbations": 5, "seed": 0}
    if metric == "max":
        return compute_max_sensitivity(explainer, instance, **sampling)
    if metric == "average":
        return compute_avg_sensitivity(explainer, instance, **sampling)
    if metric == "lipschitz":
        return compute_lipschitz_estimate(explainer, instance, **sampling)
    if metric == "continuity":
        return compute_continuity(
            explainer, instance, np.array([[1.1, 1.0], [1.0, 1.1]]), k_neighbors=2
        )
    if metric == "ris":
        return compute_relative_input_stability(explainer, model, instance, **relative_sampling)
    if metric == "rrs":
        return compute_relative_representation_stability(
            explainer, model, instance, representation_fn=np.asarray, **relative_sampling
        )
    if metric == "ros":
        return compute_relative_output_stability(
            explainer, model, instance, logit_fn=np.asarray, **relative_sampling
        )
    if metric == "relative":
        return compute_relative_stability(
            explainer,
            model,
            instance,
            representation_fn=np.asarray,
            logit_fn=np.asarray,
            **relative_sampling,
        )
    assert metric == "consistency"
    return compute_consistency(explainer, model, np.array([[1.0, 1.0], [2.0, 2.0]]), top_k=1)


@pytest.mark.parametrize("metric", _METRICS)
@pytest.mark.parametrize("mode", ["reorder", "rename"])
def test_robustness_rejects_changed_feature_identity(metric, mode):
    with pytest.raises(ValueError, match="feature identity"):
        _evaluate(metric, _FixedAttributions(mode))


@pytest.mark.parametrize("metric", _METRICS)
@pytest.mark.parametrize("mode", ["dictionary_order", "canonical_unnamed", "canonical_transition"])
def test_robustness_accepts_equivalent_feature_identity_without_extra_explanations(metric, mode):
    explainer = _FixedAttributions(mode)

    result = _evaluate(metric, explainer)

    expected = (
        {"ris": 0.0, "rrs": 0.0, "ros": 0.0}
        if metric == "relative"
        else float(metric == "consistency")
    )
    assert result == expected
    assert explainer.calls == {"continuity": 3, "consistency": 2}.get(metric, 6)


def test_consistency_keeps_its_existing_cross_target_comparison_contract():
    assert _evaluate("consistency", _FixedAttributions("target_change")) == 1.0
