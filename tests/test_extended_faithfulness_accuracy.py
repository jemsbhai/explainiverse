"""Analytical oracles for extended faithfulness diagnostics.

The equations are checked against:

* Petsiuk et al. (2018), Appendix A, raw insertion/deletion AUC;
* Yeh et al. (2019), Definition 2.1, Infidelity;
* Ancona et al. (2018), Section 4.1, Sensitivity-n PCC;
* Rieger & Hansen (2020), Equation 1, normalized IROF curve;
* Arya et al. (2019) and the AIX360 ``local_metrics`` implementation.
"""

import numpy as np
import pytest

from explainiverse.core.explanation import Explanation
from explainiverse.evaluation.faithfulness_extended import (
    _extract_attribution_array,
    compute_batch_pixel_flipping,
    compute_deletion_auc,
    compute_faithfulness_estimate,
    compute_infidelity,
    compute_insertion_auc,
    compute_irof,
    compute_monotonicity,
    compute_monotonicity_nguyen,
    compute_pixel_flipping,
    compute_region_perturbation,
    compute_road,
    compute_selectivity,
    compute_sensitivity_n,
)


class LinearScalarModel:
    """Finite scalar oracle f(x) = intercept + x dot weights."""

    task = "regression"

    def __init__(self, weights, intercept=0.1):
        self.weights = np.asarray(weights, dtype=float)
        self.intercept = float(intercept)

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return (self.intercept + X @ self.weights).reshape(-1, 1)


def make_explanation(values):
    values = np.asarray(values, dtype=float)
    names = [f"feature_{index}" for index in range(values.size)]
    return Explanation(
        explainer_name="linear-oracle",
        target_class="scalar-output",
        explanation_data={"feature_attributions": dict(zip(names, values.tolist()))},
        feature_names=names,
    )


def test_petsiuk_vector_adaptations_integrate_raw_output_curves():
    weights = np.array([0.4, 0.2, 0.1])
    model = LinearScalarModel(weights)
    instance = np.ones(3)
    explanation = make_explanation(weights)

    deletion = compute_deletion_auc(model, instance, explanation, baseline=0.0, return_curve=True)
    insertion = compute_insertion_auc(model, instance, explanation, baseline=0.0, return_curve=True)

    deletion_curve = np.array([0.8, 0.4, 0.2, 0.1])
    insertion_curve = np.array([0.1, 0.5, 0.7, 0.8])
    fractions = np.linspace(0.0, 1.0, 4)
    assert deletion["curve"] == pytest.approx(deletion_curve)
    assert deletion["auc"] == pytest.approx(np.trapezoid(deletion_curve, fractions))
    assert insertion["curve"] == pytest.approx(insertion_curve)
    assert insertion["auc"] == pytest.approx(np.trapezoid(insertion_curve, fractions))


def test_unspecified_ranking_ties_use_stable_feature_index_order():
    explanation = make_explanation([1.0, 1.0])
    details = compute_deletion_auc(
        LinearScalarModel([0.5, 0.2]),
        np.ones(2),
        explanation,
        baseline=0.0,
        return_curve=True,
    )
    assert details["feature_order"].tolist() == [0, 1]


def test_pixel_flipping_uses_raw_recorded_function_values():
    weights = np.array([0.4, 0.2, 0.1])
    result = compute_pixel_flipping(
        LinearScalarModel(weights),
        np.ones(3),
        make_explanation(weights),
        baseline=0.0,
        return_curve=True,
    )

    expected = np.array([0.8, 0.4, 0.2, 0.1])
    assert result["curve"] == pytest.approx(expected)
    assert result["auc"] == pytest.approx(np.trapezoid(expected, np.linspace(0.0, 1.0, 4)))


def test_aix360_faithfulness_proxy_uses_signed_drops():
    weights = np.array([-0.2, 0.1, 0.3, 0.4])
    score = compute_faithfulness_estimate(
        LinearScalarModel(weights, intercept=1.0),
        np.ones(4),
        make_explanation(weights),
        baseline=0.0,
    )
    assert score == pytest.approx(1.0)


@pytest.mark.parametrize("perturbation_type", ["gaussian", "square", "subset"])
def test_yeh_infidelity_definition_is_exact_for_a_linear_model(
    perturbation_type,
):
    weights = np.array([-0.2, 0.1, 0.3, 0.4])
    kwargs = {
        "perturbation_type": perturbation_type,
        "n_samples": 40,
        "seed": 9,
    }
    if perturbation_type == "square":
        kwargs["noise_scale"] = 0.5
        kwargs["baseline"] = 0.0
    elif perturbation_type == "subset":
        kwargs["subset_size"] = 2
        kwargs["baseline"] = 0.0

    score = compute_infidelity(
        LinearScalarModel(weights, intercept=1.0),
        np.ones(4),
        make_explanation(weights),
        **kwargs,
    )
    assert score == pytest.approx(0.0, abs=1e-28)


def test_ancona_sensitivity_n_signed_equation_on_linear_model():
    weights = np.array([-0.2, 0.1, 0.3, 0.4])
    details = compute_sensitivity_n(
        LinearScalarModel(weights, intercept=1.0),
        np.ones(4),
        make_explanation(weights),
        baseline=0.0,
        n=2,
        n_subsets=80,
        seed=4,
        return_details=True,
    )
    assert details["prediction_drops"] == pytest.approx(details["attribution_sums"])
    assert details["correlation"] == pytest.approx(1.0)


def test_irof_uses_original_score_normalized_aoc():
    weights = np.array([0.4, 0.2, 0.1])
    details = compute_irof(
        LinearScalarModel(weights),
        np.ones(3),
        make_explanation(weights),
        baseline=0.0,
        segment_size=1,
        return_details=True,
    )
    predictions = np.array([0.8, 0.4, 0.2, 0.1])
    expected_curve = 1.0 - predictions / predictions[0]
    assert details["normalised_predictions"] == pytest.approx(predictions / predictions[0])
    assert details["curve"] == pytest.approx(expected_curve)
    assert details["aoc"] == pytest.approx(np.trapezoid(expected_curve, np.linspace(0.0, 1.0, 4)))


def test_irof_default_segment_importance_is_mean_l1_relevance():
    explanation = make_explanation([-5.0, 4.0, 2.0, 2.0])
    model = LinearScalarModel([0.4, 0.2, 0.1, 0.05])

    l1_details = compute_irof(
        model,
        np.ones(4),
        explanation,
        baseline=0.0,
        segment_size=2,
        return_details=True,
    )
    signed_details = compute_irof(
        model,
        np.ones(4),
        explanation,
        baseline=0.0,
        segment_size=2,
        use_absolute=False,
        return_details=True,
    )

    assert l1_details["segment_importance"] == pytest.approx([4.5, 2.0])
    assert l1_details["segment_order"] == [0, 1]
    assert signed_details["segment_importance"] == pytest.approx([-0.5, 2.0])
    assert signed_details["segment_order"] == [1, 0]


def test_selectivity_is_the_documented_local_aopc():
    weights = np.array([0.4, 0.2, 0.1])
    details = compute_selectivity(
        LinearScalarModel(weights),
        np.ones(3),
        make_explanation(weights),
        baseline=0.0,
        return_details=True,
    )
    expected_drops = np.array([0.0, 0.4, 0.6, 0.7])
    assert details["prediction_drops"] == pytest.approx(expected_drops)
    assert details["aopc"] == pytest.approx(np.mean(expected_drops))


def test_aix360_monotonicity_is_binary_and_uses_increasing_signed_order():
    weights = np.array([-0.2, 0.1, 0.3, 0.4])
    score = compute_monotonicity(
        LinearScalarModel(weights, intercept=1.0),
        np.ones(4),
        make_explanation(weights),
        baseline=0.0,
    )
    assert score == 1.0


def test_nguyen_single_baseline_proxy_matches_its_documented_formula():
    weights = np.array([0.4, 0.2, 0.1])
    score = compute_monotonicity_nguyen(
        LinearScalarModel(weights),
        np.ones(3),
        make_explanation(weights),
        baseline=0.0,
    )
    assert score == pytest.approx(1.0)


def test_local_road_inspired_score_is_not_dataset_accuracy():
    weights = np.array([0.4, 0.2, 0.1])
    details = compute_road(
        LinearScalarModel(weights),
        np.ones(3),
        make_explanation(weights),
        background_data=np.zeros((8, 3)),
        percentages=[0.34],
        noise_scale=0.0,
        seed=2,
        return_details=True,
    )
    assert details["predictions"] == pytest.approx([0.4])
    assert details["prediction_changes"] == pytest.approx([0.4])
    assert details["score"] == pytest.approx(0.4)
    assert "not implement ROAD's spatial imputer" in compute_road.__doc__


def test_region_perturbation_reports_its_relative_local_curve():
    weights = np.array([0.4, 0.2, 0.1])
    details = compute_region_perturbation(
        LinearScalarModel(weights),
        np.ones(3),
        make_explanation(weights),
        baseline=0.0,
        region_size=1,
        return_curve=True,
    )
    assert details["curve"] == pytest.approx(np.array([0.8, 0.4, 0.2, 0.1]) / 0.8)


def test_undefined_correlations_fail_instead_of_fabricating_zero_or_one():
    model = LinearScalarModel([0.4, 0.2, 0.1])
    constant = make_explanation([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="undefined for a constant input"):
        compute_faithfulness_estimate(model, np.ones(3), constant, baseline=0.0)
    with pytest.raises(ValueError, match="undefined for a constant input"):
        compute_monotonicity_nguyen(model, np.ones(3), constant, baseline=0.0)
    with pytest.raises(ValueError, match="undefined for a constant input"):
        compute_sensitivity_n(
            model,
            np.ones(3),
            constant,
            baseline=0.0,
            n=3,
            n_subsets=5,
            seed=1,
        )


def test_attribution_mapping_rejects_ambiguous_or_unmapped_payloads():
    explanation = Explanation(
        explainer_name="test",
        target_class="output",
        explanation_data={
            "feature_attributions": {
                "0.0 < long feature <= 1.0": 2.0,
                "other": 1.0,
            }
        },
        feature_names=["long feature", "other"],
    )
    assert _extract_attribution_array(explanation, 2) == pytest.approx([2.0, 1.0])

    unknown = Explanation(
        explainer_name="test",
        target_class="output",
        explanation_data={"feature_attributions": {"unknown": 1.0}},
        feature_names=["a", "b"],
    )
    with pytest.raises(ValueError, match="cannot map attribution feature"):
        _extract_attribution_array(unknown, 2)

    substring_collision = Explanation(
        explainer_name="test",
        target_class="output",
        explanation_data={"feature_attributions": {"mortgage": 1.0, "income": 2.0}},
        feature_names=["age", "income"],
    )
    with pytest.raises(ValueError, match="cannot map attribution feature"):
        _extract_attribution_array(substring_collision, 2)

    incomplete = Explanation(
        explainer_name="test",
        target_class="output",
        explanation_data={"feature_attributions": {"age": 1.0}},
        feature_names=["age", "income"],
    )
    with pytest.raises(ValueError, match="cover every input feature"):
        _extract_attribution_array(incomplete, 2)


def test_seeded_monte_carlo_does_not_mutate_numpy_global_rng():
    weights = np.array([0.4, 0.2, 0.1])
    np.random.seed(1234)
    expected_next = np.random.random()
    np.random.seed(1234)
    compute_infidelity(
        LinearScalarModel(weights),
        np.ones(3),
        make_explanation(weights),
        perturbation_type="gaussian",
        n_samples=5,
        seed=8,
    )
    assert np.random.random() == pytest.approx(expected_next)


def test_batch_contract_propagates_invalid_explanations():
    good = make_explanation([0.4, 0.2, 0.1])
    bad = Explanation(
        explainer_name="bad",
        target_class="output",
        explanation_data={"feature_attributions": {"unknown": 1.0}},
        feature_names=["feature_0", "feature_1", "feature_2"],
    )
    with pytest.raises(ValueError, match="cannot map attribution feature"):
        compute_batch_pixel_flipping(
            LinearScalarModel([0.4, 0.2, 0.1]),
            np.ones((2, 3)),
            [good, bad],
            baseline=0.0,
        )


@pytest.mark.parametrize(
    "bad_instance",
    [np.ones((1, 3)), np.array([1.0, np.nan, 2.0])],
)
def test_instance_shape_and_finiteness_are_enforced(bad_instance):
    with pytest.raises(ValueError):
        compute_pixel_flipping(
            LinearScalarModel([0.4, 0.2, 0.1]),
            bad_instance,
            make_explanation([0.4, 0.2, 0.1]),
            baseline=0.0,
        )
