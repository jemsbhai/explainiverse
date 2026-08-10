"""Analytical and counterexample tests for faithfulness diagnostics.

These tests encode the defining equations rather than merely checking output
types.  Primary references are listed in ``evaluation/faithfulness.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from explainiverse.core.explanation import Explanation
from explainiverse.evaluation.faithfulness import (
    compare_explainer_faithfulness,
    compute_batch_faithfulness,
    compute_comprehensiveness,
    compute_faithfulness_correlation,
    compute_faithfulness_score,
    compute_pgi,
    compute_pgu,
    compute_sufficiency,
)
from explainiverse.evaluation.faithfulness_extended import compute_batch_deletion_auc


class LinearRegressor:
    _estimator_type = "regressor"

    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=float)

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.weights


class ThreeOutputClassifier:
    _estimator_type = "classifier"
    class_names = ["a", "b", "c"]

    def predict(self, X):
        values = np.asarray(X, dtype=float)[:, 0]
        return np.column_stack([0.1 * values, 0.05 * values + 0.1, 0.9 - 0.15 * values])


class NoncontiguousNumericLabelClassifier:
    _estimator_type = "classifier"
    classes_ = np.array([10, 20])

    def predict(self, X):
        positive = 0.2 + 0.1 * np.asarray(X, dtype=float)[:, 0]
        return np.column_stack([1.0 - positive, positive])


class TwoOutputRegressor:
    _estimator_type = "regressor"

    def predict(self, X):
        values = np.asarray(X, dtype=float)
        return np.column_stack([values[:, 0], 2.0 * values[:, 0]])


def explanation(values, *, target="output_0", metadata=None, names=None):
    values = list(values)
    feature_names = names or [f"f{index}" for index in range(len(values))]
    return Explanation(
        explainer_name="analytical",
        target_class=target,
        explanation_data={"feature_attributions": dict(zip(feature_names, values, strict=True))},
        feature_names=feature_names,
        metadata=metadata,
    )


def test_pgu_replaces_the_complete_non_top_k_complement():
    """OpenXAI PGU perturbs non-top-k, not merely bottom-k."""
    model = LinearRegressor([1.0, 1.0, 1.0, 1.0])
    instance = np.array([10.0, 3.0, 2.0, 1.0])
    exp = explanation(instance)

    # f(x)=16. PGI removes x0, leaving 6. PGU holds x0 fixed and
    # replaces x1,x2,x3, leaving 10.
    assert compute_pgi(model, instance, exp, k=1, baseline=0.0) == pytest.approx(10.0)
    assert compute_pgu(model, instance, exp, k=1, baseline=0.0) == pytest.approx(6.0)
    assert compute_pgu(model, instance, exp, k=4, baseline=0.0) == pytest.approx(0.0)


def test_pgi_tracks_the_output_identified_by_explanation_metadata():
    model = ThreeOutputClassifier()
    instance = np.array([4.0, 1.0])
    exp = explanation([2.0, 1.0], target="b", metadata={"output_index": 1})

    # Output b changes from .30 to .20 when x0 is replaced by 2.
    assert compute_pgi(model, instance, exp, k=1, baseline=np.array([2.0, 0.0])) == pytest.approx(
        0.1
    )


def test_conflicting_explicit_and_explanation_targets_are_rejected():
    model = ThreeOutputClassifier()
    exp = explanation([2.0, 1.0], target="b", metadata={"output_index": 1})
    with pytest.raises(ValueError, match="conflicting target"):
        compute_pgi(
            model,
            np.array([4.0, 1.0]),
            exp,
            k=1,
            baseline=0.0,
            target_class=0,
        )


def test_unmappable_explanation_target_is_not_replaced_by_model_argmax():
    model = ThreeOutputClassifier()
    exp = explanation([2.0, 1.0], target="free-form-unmapped-label")
    with pytest.raises(ValueError, match="cannot establish which model output"):
        compute_pgi(model, np.array([4.0, 1.0]), exp, k=1, baseline=0.0)


def test_numeric_explanation_label_maps_through_noncontiguous_model_classes():
    model = NoncontiguousNumericLabelClassifier()
    exp = explanation([1.0, 0.0], target=20)

    result = compute_comprehensiveness(
        model,
        np.array([2.0, 0.0]),
        exp,
        k_values=[1],
        baseline=0.0,
    )

    assert result == {
        "comp_k1": pytest.approx(0.2),
        "comprehensiveness": pytest.approx(0.2),
    }


def test_boolean_explanation_target_is_not_treated_as_output_one():
    model = ThreeOutputClassifier()
    exp = explanation([1.0, 0.0], target=True)

    with pytest.raises(TypeError, match="explanation.target_class"):
        compute_pgi(model, np.array([2.0, 0.0]), exp, k=1, baseline=0.0)


def test_multioutput_regression_requires_and_honors_an_output_index():
    model = TwoOutputRegressor()
    instance = np.array([4.0, 1.0])
    ambiguous = explanation([2.0, 1.0], target="regression")

    with pytest.raises(ValueError, match="multi-output regression"):
        compute_pgi(model, instance, ambiguous, k=1, baseline=0.0)

    selected = Explanation(
        "analytical",
        "output_1",
        {
            "feature_attributions": {"f0": 2.0, "f1": 1.0},
            "output_index": 1,
        },
        ["f0", "f1"],
    )
    assert compute_pgi(model, instance, selected, k=1, baseline=0.0) == pytest.approx(8.0)


def test_eraser_adaptations_use_signed_scores_and_signed_output_differences():
    model = LinearRegressor([1.0, 1.0, 1.0])
    instance = np.array([-10.0, 3.0, 1.0])
    exp = explanation(instance)

    # ERASER selects the largest score (+3), not largest magnitude (-10).
    # Original output is -6. Removing +3 gives -9, while keeping only +3
    # gives +3. The equations retain both signs.
    comp = compute_comprehensiveness(model, instance, exp, k_values=[1], baseline=0.0)
    suff = compute_sufficiency(model, instance, exp, k_values=[1], baseline=0.0)

    assert comp == {"comp_k1": pytest.approx(3.0), "comprehensiveness": pytest.approx(3.0)}
    assert suff == {"suff_k1": pytest.approx(-9.0), "sufficiency": pytest.approx(-9.0)}


def test_eraser_default_aopc_thresholds_match_released_scorer():
    model = LinearRegressor([1.0, 1.0, 1.0, 1.0])
    instance = np.array([4.0, 3.0, 2.0, 1.0])
    result = compute_comprehensiveness(model, instance, explanation(instance), baseline=0.0)
    assert set(result) == {
        "comp_k0.01",
        "comp_k0.05",
        "comp_k0.1",
        "comp_k0.2",
        "comp_k0.5",
        "comprehensiveness",
    }
    assert result["comprehensiveness"] == pytest.approx(
        np.mean([result[key] for key in result if key.startswith("comp_k")])
    )


def test_bhatt_correlation_detects_an_explanation_with_reversed_signs():
    model = LinearRegressor([1.0, 1.0, 1.0])
    instance = np.array([-3.0, 1.0, 2.0])
    anti_faithful = explanation([3.0, -1.0, -2.0])

    # For singleton subsets the true signed output drops are [-3, 1, 2].
    # The reversed explanation is exactly anticorrelated. Magnitude-only code
    # would incorrectly report +1.
    result = compute_faithfulness_correlation(
        model, instance, anti_faithful, baseline=0.0, subset_size=1
    )
    assert result == pytest.approx(-1.0)


def test_bhatt_correlation_sums_fixed_size_subsets():
    model = LinearRegressor([1.0, 1.0, 1.0, 1.0])
    instance = np.array([-3.0, 1.0, 2.0, 5.0])
    result = compute_faithfulness_correlation(
        model,
        instance,
        explanation(instance),
        baseline=0.0,
        subset_size=2,
    )
    assert result == pytest.approx(1.0)


def test_subset_sampling_is_reproducible_and_does_not_mutate_numpy_global_rng():
    model = LinearRegressor([1.0, 1.0, 1.0, 1.0])
    instance = np.array([-3.0, 1.0, 2.0, 5.0])
    exp = explanation([2.0, -1.0, 4.0, 3.0])

    np.random.seed(123)
    expected_next = np.random.random()
    np.random.seed(123)
    first = compute_faithfulness_correlation(
        model,
        instance,
        exp,
        baseline=0.0,
        subset_size=2,
        n_steps=4,
        random_state=7,
    )
    actual_next = np.random.random()
    second = compute_faithfulness_correlation(
        model,
        instance,
        exp,
        baseline=0.0,
        subset_size=2,
        n_steps=4,
        random_state=7,
    )

    assert actual_next == expected_next
    assert first == second


def test_subset_sampling_rejects_impossible_counts_and_invalid_seed():
    model = LinearRegressor([1.0, 1.0, 1.0])
    instance = np.array([1.0, 2.0, 3.0])
    exp = explanation(instance)

    with pytest.raises(ValueError, match="exceeds"):
        compute_faithfulness_correlation(
            model, instance, exp, baseline=0.0, subset_size=1, n_steps=4
        )
    with pytest.raises(TypeError, match="random_state"):
        compute_faithfulness_correlation(model, instance, exp, baseline=0.0, random_state=True)


@pytest.mark.parametrize(
    "attributions",
    ([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]),
)
def test_undefined_pearson_correlation_is_not_fabricated_as_zero(attributions):
    model = LinearRegressor([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="undefined"):
        compute_faithfulness_correlation(
            model,
            np.array([1.0, 2.0, 3.0]),
            explanation(attributions),
            baseline=0.0,
        )


def test_top_k_tie_crossing_cutoff_is_rejected():
    model = LinearRegressor([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="tie"):
        compute_pgi(
            model,
            np.array([3.0, 2.0, 1.0]),
            explanation([2.0, 1.0, 1.0]),
            k=2,
            baseline=0.0,
        )


@pytest.mark.parametrize(
    ("bad_instance", "match"),
    [
        (np.array([[1.0, 2.0]]), "one-dimensional"),
        (np.array([1.0, np.nan]), "finite"),
        (np.array([], dtype=float), "at least one"),
    ],
)
def test_instance_validation_is_strict(bad_instance, match):
    with pytest.raises(ValueError, match=match):
        compute_pgi(
            LinearRegressor([1.0, 1.0]),
            bad_instance,
            explanation([1.0, 2.0]),
            k=1,
            baseline=0.0,
        )


def test_attributions_must_cover_every_feature_exactly_and_be_finite():
    model = LinearRegressor([1.0, 1.0])
    missing = Explanation("bad", "output_0", {"feature_attributions": {"f0": 1.0}}, ["f0", "f1"])
    non_finite = explanation([1.0, np.inf])

    with pytest.raises(ValueError, match="cover feature_names exactly"):
        compute_pgi(model, np.array([1.0, 2.0]), missing, k=1, baseline=0.0)
    with pytest.raises(ValueError, match="finite"):
        compute_pgi(model, np.array([1.0, 2.0]), non_finite, k=1, baseline=0.0)


def test_baseline_shape_and_k_are_not_silently_repaired():
    model = LinearRegressor([1.0, 1.0])
    instance = np.array([1.0, 2.0])
    exp = explanation(instance)

    with pytest.raises(ValueError, match="baseline must resolve to shape"):
        compute_pgi(model, instance, exp, k=1, baseline=np.array([0.0]))
    with pytest.raises(ValueError, match="integer k"):
        compute_pgi(model, instance, exp, k=3, baseline=0.0)
    with pytest.raises(TypeError, match="not boolean"):
        compute_pgi(model, instance, exp, k=True, baseline=0.0)


def test_exact_ratio_fails_when_zero_denominator_makes_it_undefined():
    instance = np.array([2.0, 1.0])
    with pytest.raises(ValueError, match="undefined because PGU is zero"):
        compute_faithfulness_score(
            LinearRegressor([1.0, 1.0]),
            instance,
            explanation(instance),
            k=2,
            baseline=0.0,
        )


def test_batch_and_comparison_reject_misaligned_explanations():
    model = LinearRegressor([1.0, 1.0])
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    explanations = [explanation(X[0])]

    with pytest.raises(ValueError, match="one explanation per row"):
        compute_batch_faithfulness(model, X, explanations, baseline=0.0)
    with pytest.raises(ValueError, match="one explanation per row"):
        compare_explainer_faithfulness(
            model, X, {"method": explanations}, baseline=0.0, max_samples=1
        )


def test_batch_propagates_invalid_metric_input_instead_of_skipping_it():
    model = LinearRegressor([1.0, 1.0])
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    invalid = Explanation("bad", "output_0", {"feature_attributions": {"f0": 1.0}}, ["f0", "f1"])
    with pytest.raises(ValueError, match="cover feature_names exactly"):
        compute_batch_faithfulness(
            model,
            X,
            [explanation(X[0]), invalid],
            baseline=0.0,
        )


def test_batch_statistical_baseline_requires_explicit_background_data():
    model = LinearRegressor([1.0, 1.0])
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    explanations = [explanation(row) for row in X]

    with pytest.raises(ValueError, match="background_data is required"):
        compute_batch_faithfulness(model, X, explanations, baseline="mean")

    result = compute_batch_faithfulness(
        model,
        X,
        explanations,
        baseline="mean",
        background_data=np.array([[-10.0, -20.0], [10.0, 20.0]]),
    )
    assert result["n_samples"] == 2


def test_batch_reports_ratio_of_means_separately_from_mean_sample_ratio():
    model = LinearRegressor([1.0, 1.0])
    X = np.array([[2.0, 1.0], [2.0, 8.0]])
    explanations = [explanation(row) for row in X]

    result = compute_batch_faithfulness(model, X, explanations, k=1, baseline=0.0)
    assert result["ratio_of_means"] == pytest.approx(10.0 / 3.0)
    assert result["mean_of_sample_ratios"] == pytest.approx(3.0)
    assert result["mean_ratio"] == result["ratio_of_means"]


def test_batch_never_emits_nonfinite_values_for_an_undefined_ratio():
    model = LinearRegressor([1.0, 1.0])
    X = np.array([[2.0, 1.0], [3.0, 4.0]])
    explanations = [explanation(row) for row in X]
    with pytest.raises(ValueError, match="undefined because PGU is zero"):
        compute_batch_faithfulness(model, X, explanations, k=2, baseline=0.0)


def test_extended_batch_wrappers_require_exact_explanation_pairing_even_with_max_samples():
    model = LinearRegressor([1.0, 1.0])
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="exactly one explanation per row"):
        compute_batch_deletion_auc(
            model,
            X,
            [explanation(X[0])],
            baseline=0.0,
            max_samples=1,
        )
