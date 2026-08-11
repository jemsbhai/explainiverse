"""Falsifiable gates for residual numerical and result-schema limitations."""

from __future__ import annotations

import json
import math
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest
from scipy import stats

from explainiverse import DetailRepresentationError, decode_scaled_detail, encode_scaled_detail
from explainiverse.core.explanation import Explanation
from explainiverse.evaluation import (
    compare_consistency_results,
    compare_intervention_sensitivity_reports,
    compute_consistency,
    evaluate_intervention_sensitivity,
    run_seeded_replicates,
    summarize_replicate_estimates,
)
from explainiverse.evaluation.fairness import compute_group_metric_disparity
from explainiverse.evaluation.faithfulness_extended import (
    compute_infidelity,
    compute_irof,
    compute_region_perturbation,
    compute_road,
    compute_selectivity,
    compute_sensitivity_n,
)
from explainiverse.evaluation.metrics import compute_aopc
from explainiverse.evaluation.randomisation import (
    _efficient_mprt_relative_entropy_change,
    _entropy_from_counts,
    _ssim_similarity,
)
from explainiverse.explainers.example_based.protodash import ProtoDashExplainer
from explainiverse.explainers.gradient.cam_variants import (
    EigenGradCAMExplainer,
    GradCAMElementWiseExplainer,
    LayerCAMExplainer,
    _principal_projection,
)
from explainiverse.explainers.gradient.gradcam import _cam_normalization_metadata


def _explanation(values) -> Explanation:
    names = [f"f{index}" for index in range(len(values))]
    return Explanation(
        "test",
        "output_0",
        {"feature_attributions": dict(zip(names, values))},
        feature_names=names,
        metadata={"output_index": 0},
    )


def test_layercam_extreme_oracles_and_explicit_out_of_range_failure():
    maximum = np.finfo(np.float64).max
    activations = np.array([[[[maximum]], [[-maximum]]]], dtype=np.float64)
    gradients = np.full_like(activations, maximum)

    np.testing.assert_array_equal(
        LayerCAMExplainer._compute_cam(None, activations, gradients, None, None),
        [[0.0]],
    )

    minimum_subnormal = np.nextafter(0.0, 1.0)
    tiny_activations = np.full((1, 2, 1, 1), np.ldexp(1.0, -600))
    tiny_gradients = np.full((1, 2, 1, 1), np.ldexp(1.0, -474))
    np.testing.assert_array_equal(
        LayerCAMExplainer._compute_cam(None, tiny_activations, tiny_gradients, None, None),
        [[2.0 * minimum_subnormal]],
    )

    with pytest.raises(FloatingPointError, match="not representable"):
        LayerCAMExplainer._compute_cam(
            None,
            np.array([[[[maximum]]]]),
            np.array([[[[2.0]]]]),
            None,
            None,
        )


def test_gradcam_elementwise_recovers_summed_underflow_but_not_true_overflow():
    minimum_subnormal = np.nextafter(0.0, 1.0)
    activations = np.full((1, 2, 1, 1), np.ldexp(1.0, -600))
    gradients = np.full((1, 2, 1, 1), np.ldexp(1.0, -475))

    np.testing.assert_array_equal(
        GradCAMElementWiseExplainer._compute_cam(None, activations, gradients, None, None),
        [[minimum_subnormal]],
    )
    with pytest.raises(FloatingPointError, match="not representable"):
        GradCAMElementWiseExplainer._compute_cam(
            None,
            np.array([[[[np.finfo(np.float64).max]]]]),
            np.array([[[[2.0]]]]),
            None,
            None,
        )


def test_eigengradcam_scaled_projection_restores_representable_extreme_amplitude():
    ordinary_activations = np.array([[[[1.0, 2.0], [3.0, 4.0]], [[4.0, 3.0], [2.0, 1.0]]]])
    ordinary_gradients = np.array([[[[2.0, -1.0], [0.0, 1.0]], [[-2.0, 1.0], [3.0, 0.0]]]])
    expected = _principal_projection(ordinary_gradients * ordinary_activations, center=True)
    actual = EigenGradCAMExplainer._compute_cam(
        None, ordinary_activations, ordinary_gradients, None, None
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

    maximum = np.finfo(np.float64).max
    below_maximum = np.nextafter(maximum, 0.0)
    extreme_activations = np.array([[[[maximum, below_maximum]]]])
    extreme_gradients = np.full_like(extreme_activations, 1.5)
    exact_amplitude = (
        Fraction.from_float(maximum) - Fraction.from_float(below_maximum)
    ) * Fraction(3, 4)
    expected_extreme = np.array([[float(exact_amplitude), -float(exact_amplitude)]])
    extreme = EigenGradCAMExplainer._compute_cam(
        None, extreme_activations, extreme_gradients, None, None
    )
    np.testing.assert_array_equal(extreme, expected_extreme)
    assert _cam_normalization_metadata(extreme) == {
        "normalization_input_min": -float(exact_amplitude),
        "normalization_input_max": float(exact_amplitude),
        "normalization_degenerate": False,
        "constant_map_value": None,
    }

    minimum_subnormal = float(np.nextafter(0.0, 1.0))
    mixed_scale = EigenGradCAMExplainer._compute_cam(
        None,
        np.array(
            [[[[1e308, 1e308]], [[minimum_subnormal, 3.0 * minimum_subnormal]]]],
            dtype=np.float64,
        ),
        np.ones((1, 2, 1, 2), dtype=np.float64),
        None,
        None,
    )
    np.testing.assert_array_equal(
        mixed_scale,
        np.array([[minimum_subnormal, -minimum_subnormal]], dtype=np.float64),
    )

    subnormal_centering = EigenGradCAMExplainer._compute_cam(
        None,
        np.array(
            [[[[minimum_subnormal, 6.0 * minimum_subnormal]]]],
            dtype=np.float64,
        ),
        np.ones((1, 1, 1, 2), dtype=np.float64),
        None,
        None,
    )
    np.testing.assert_array_equal(
        subnormal_centering,
        np.array([[2.0 * minimum_subnormal, -2.0 * minimum_subnormal]]),
    )

    with pytest.raises(FloatingPointError, match="not representable"):
        EigenGradCAMExplainer._compute_cam(
            None,
            np.array([[[[maximum, -maximum]]]]),
            np.full((1, 1, 1, 2), 2.0),
            None,
            None,
        )
    with pytest.raises(FloatingPointError, match="not representable"):
        _principal_projection(
            np.array([[[[maximum, -maximum]], [[maximum, -maximum]]]]),
            center=True,
        )
    with pytest.raises(FloatingPointError, match="globally scaled SVD input"):
        EigenGradCAMExplainer._compute_cam(
            None,
            np.array(
                [
                    [
                        [[maximum, -maximum]],
                        [[minimum_subnormal, -minimum_subnormal]],
                    ]
                ]
            ),
            np.ones((1, 2, 1, 2)),
            None,
            None,
        )


def test_eigengradcam_scaled_projection_retains_extreme_direction_when_representable():
    maximum = np.finfo(np.float64).max
    exceptional_gradient = np.nextafter(1.0, 2.0)
    extreme_activations = np.array(
        [[[[maximum, 0.0], [maximum, 0.0]], [[0.0, maximum], [0.0, maximum]]]]
    )
    extreme_gradients = np.full_like(extreme_activations, exceptional_gradient)
    extreme = EigenGradCAMExplainer._compute_cam(
        None, extreme_activations, extreme_gradients, None, None
    )
    assert np.isfinite(extreme).all()
    assert (extreme.reshape(-1) > 0.0).tolist() == [True, False, True, False]


def test_quarantined_cam_variants_disclose_their_promotion_boundary():
    for explainer in (
        object.__new__(EigenGradCAMExplainer),
        object.__new__(GradCAMElementWiseExplainer),
    ):
        metadata = explainer._method_metadata()
        assert metadata["claim_status"] == "quarantined"
        assert metadata["promotion_requires_primary_formula"] is True


def test_count_domain_entropy_distinguishes_exact_zero_from_positive_below_epsilon():
    sample_count = 10**18
    tiny_positive = _entropy_from_counts([sample_count - 1, 1])

    assert _entropy_from_counts([sample_count, 0]) == 0.0
    assert 0.0 < tiny_positive < np.finfo(float).eps
    assert _efficient_mprt_relative_entropy_change(tiny_positive, 2.0 * tiny_positive) == 1.0
    with pytest.raises(ValueError, match="exactly zero"):
        _efficient_mprt_relative_entropy_change(0.0, tiny_positive)


def test_ssim_small_map_error_names_owned_alternatives():
    pytest.importorskip("skimage.metrics")
    with pytest.raises(ValueError, match="pearson.*cosine.*aggregate"):
        _ssim_similarity(np.zeros((2, 2)), np.ones((2, 2)))


class _TiedExplainer:
    def explain(self, instance):
        del instance
        return _explanation([3.0, 2.0, -2.0, 0.0])


class _ConstantClassifier:
    def predict(self, values):
        return np.zeros(len(np.asarray(values)), dtype=int)


def test_consistency_tie_policies_record_incidence_and_mixed_comparisons_fail():
    X = np.arange(12.0).reshape(3, 4)
    stable = compute_consistency(
        _TiedExplainer(),
        _ConstantClassifier(),
        X,
        top_k=2,
        tie_policy="stable_order",
        return_details=True,
    )
    include_all = compute_consistency(
        _TiedExplainer(),
        _ConstantClassifier(),
        X,
        top_k=2,
        tie_policy="include_all",
        return_details=True,
    )

    assert stable["score"] == include_all["score"] == 1.0
    assert stable["cutoff_tie_count"] == include_all["cutoff_tie_count"] == len(X)
    assert stable["selected_feature_counts"] == [2, 2, 2]
    assert include_all["selected_feature_counts"] == [3, 3, 3]
    with pytest.raises(ValueError, match="tie spanning the cutoff"):
        compute_consistency(
            _TiedExplainer(),
            _ConstantClassifier(),
            X,
            top_k=2,
            tie_policy="reject",
        )
    with pytest.raises(ValueError, match="mixed tie_policy"):
        compare_consistency_results({"stable": stable, "inclusive": include_all})
    comparison = compare_consistency_results({"first": stable, "second": stable})
    assert comparison["tie_policy"] == "stable_order"


def test_fairness_boundary_states_are_machine_readable_and_not_certificates():
    equal = compute_group_metric_disparity(np.ones((4, 1)), [0, 0, 1, 1])
    unequal = compute_group_metric_disparity(np.array([[1.0], [1.0], [2.0], [2.0]]), [0, 0, 1, 1])
    finite = compute_group_metric_disparity(np.array([[1.0], [2.0], [3.0], [5.0]]), [0, 0, 1, 1])

    pair = (0, 1)
    assert equal["pairwise_p_value_defined"][pair] is False
    assert "completely_tied" in equal["pairwise_p_value_reasons"][pair]
    assert equal["pairwise_effect_size_defined"][pair] is False
    assert "equal_group_means" in equal["pairwise_effect_size_reasons"][pair]
    assert unequal["pairwise_effect_size_defined"][pair] is False
    assert "signed_infinity" in unequal["pairwise_effect_size_reasons"][pair]
    assert finite["pairwise_effect_size_defined"][pair] is True
    assert finite["pairwise_effect_size_reasons"][pair] is None
    for result in (equal, unequal, finite):
        assert result["fairness_conclusion_defined"] is False
        assert "external_domain" in result["fairness_conclusion_reason"]


@pytest.mark.parametrize("epsilon", [1e-20, 1e-10, 1e10])
def test_protodash_reports_exact_normalization_threshold_relations(epsilon):
    explainer = ProtoDashExplainer(n_prototypes=1, epsilon=epsilon)
    cases = [
        (np.nextafter(epsilon, 0.0), "below", False),
        (epsilon, "equal", False),
        (np.nextafter(epsilon, np.inf), "above", True),
    ]
    for mass, relation, defined in cases:
        values = np.array([mass])
        metadata = explainer._weight_normalization_metadata(values)
        assert metadata["normalization_threshold"] == epsilon
        assert metadata["objective_weight_mass_threshold_relation"] == relation
        assert explainer._normalized_weights_defined(values) is defined
        if defined:
            assert metadata["normalized_weights_undefined_reason"] is None
        else:
            assert relation in metadata["normalized_weights_undefined_reason"]


def test_scaled_detail_v1_preserves_exceptional_exact_values_and_is_wire_safe():
    below_binary64 = Fraction.from_float(np.nextafter(0.0, 1.0)) / 2
    repeating = Fraction(1, 3)
    values = [1.5, Decimal("1e10000"), below_binary64, repeating]
    payload = encode_scaled_detail(values)
    transported = json.loads(json.dumps(payload, allow_nan=False))
    decoded = decode_scaled_detail(transported)

    assert transported["values"][0] == 1.5
    assert decoded[0] == 1.5
    assert decoded[1] == Decimal("1e10000")
    assert decoded[2] == values[2]
    assert decoded[3] == repeating
    assert transported["values"][2]["exact_fraction"]["denominator"].isdigit()
    assert transported["values"][3] == {"exact_fraction": {"numerator": "1", "denominator": "3"}}
    explanation = Explanation("detail", "target", {"details": payload})
    assert Explanation.from_wire_dict(explanation.to_wire_dict()).to_wire_dict() == (
        explanation.to_wire_dict()
    )

    with pytest.raises(ValueError, match="negative zero"):
        encode_scaled_detail([-0.0])
    with pytest.raises(ValueError, match="safe-integer"):
        decode_scaled_detail(
            {
                "schema_version": "explainiverse.scaled-detail.v1",
                "source_dtype": "float64",
                "values": [2**53],
            }
        )


@pytest.mark.parametrize("sign", [1, -1])
def test_scaled_detail_v1_uses_wire_safe_numbers_at_and_beyond_integer_boundary(sign):
    safe = sign * (2**53 - 1)
    unsafe = sign * 2**53
    non_integral_large = sign * (float(2**51) + 0.5)
    payload = encode_scaled_detail(
        [
            Fraction(safe, 1),
            Decimal(safe),
            float(safe),
            Fraction(unsafe, 1),
            Decimal(unsafe),
            float(unsafe),
            non_integral_large,
        ]
    )

    assert payload["values"][:3] == [float(safe)] * 3
    assert payload["values"][3] == {
        "exact_fraction": {"numerator": str(unsafe), "denominator": "1"}
    }
    assert payload["values"][4] == {"exact_decimal": str(unsafe)}
    assert payload["values"][5] == {
        "exact_fraction": {"numerator": str(unsafe), "denominator": "1"}
    }
    assert payload["values"][6] == non_integral_large
    assert decode_scaled_detail(payload) == [
        float(safe),
        float(safe),
        float(safe),
        Fraction(unsafe, 1),
        Decimal(unsafe),
        Fraction(unsafe, 1),
        non_integral_large,
    ]

    explanation = Explanation("detail", "target", {"details": payload})
    transported = json.loads(json.dumps(explanation.to_wire_dict(), allow_nan=False))
    assert Explanation.from_wire_dict(transported).to_wire_dict() == transported

    malicious = dict(payload)
    malicious["values"] = [float(unsafe)]
    with pytest.raises(ValueError, match="safe-integer"):
        decode_scaled_detail(malicious)


def test_scaled_detail_numpy_scalar_width_policy_never_silently_narrows():
    supported = [np.float16(0.5), np.float32(0.25), np.float64(0.125)]
    payload = encode_scaled_detail(supported)
    assert decode_scaled_detail(payload) == [0.5, 0.25, 0.125]

    class UndeclaredFloatSubclass(float):
        pass

    with pytest.raises(TypeError, match="must be Python int/float"):
        encode_scaled_detail([UndeclaredFloatSubclass(1.0)])

    if np.dtype(np.longdouble).itemsize > np.dtype(np.float64).itemsize:
        first = np.longdouble("1")
        second = np.nextafter(first, np.longdouble("2"))
        assert first != second
        assert float(first) == float(second)
        for value in (first, second):
            with pytest.raises(TypeError, match="wider than binary64"):
                encode_scaled_detail([value])


class _StepwiseExtremeRegressor:
    _estimator_type = "regressor"
    task = "regression"

    def __init__(self, values):
        self.values = list(values)

    def predict(self, X):
        result = []
        for row in np.asarray(X):
            removed = int(np.sum(row == 0.0))
            result.append(self.values[removed])
        return np.asarray(result)


class _ZeroRegressor:
    _estimator_type = "regressor"
    task = "regression"

    def predict(self, X):
        return np.zeros(len(np.asarray(X)))


class _LinearRegressor:
    _estimator_type = "regressor"
    task = "regression"

    def __init__(self, coefficients):
        self.coefficients = np.asarray(coefficients, dtype=np.float64)

    def predict(self, X):
        return np.asarray(X) @ self.coefficients


def test_opt_in_scaled_details_retire_each_owned_scalar_detail_counterexample():
    maximum = np.finfo(np.float64).max
    original = 1e308
    path = [original, -original, maximum, maximum, maximum]
    model = _StepwiseExtremeRegressor(path)
    explanation = _explanation([4.0, 3.0, 2.0, 1.0])
    expected_extreme_drop = Fraction.from_float(original) - Fraction.from_float(-original)

    aopc = compute_aopc(
        model,
        np.ones(4),
        explanation,
        num_steps=4,
        baseline_value=0.0,
        return_details=True,
        detail_format="scaled_decimal_v1",
    )
    selectivity = compute_selectivity(
        model,
        np.ones(4),
        explanation,
        baseline=0.0,
        n_steps=4,
        return_details=True,
        detail_format="scaled_decimal_v1",
    )
    road = compute_road(
        model,
        np.ones(4),
        explanation,
        background_data=np.zeros((8, 4)),
        percentages=[0.13, 0.38, 0.63, 0.88],
        noise_scale=0.0,
        return_details=True,
        detail_format="scaled_decimal_v1",
    )
    for result, key in (
        (aopc, "prediction_drops"),
        (selectivity, "prediction_drops"),
        (road, "prediction_changes"),
    ):
        decoded = decode_scaled_detail(result[key])
        assert expected_extreme_drop in decoded
        json.dumps(result[key], allow_nan=False)

    residual = 2e154
    infidelity = compute_infidelity(
        _ZeroRegressor(),
        np.ones(1),
        _explanation([residual]),
        baseline=0.0,
        perturbation_type="square",
        noise_scale=0.5,
        n_samples=4,
        seed=1,
        return_details=True,
        detail_format="scaled_decimal_v1",
    )
    assert Fraction.from_float(residual) ** 2 in decode_scaled_detail(infidelity["squared_errors"])

    tiny_original = 1e-308
    ratio_model = _StepwiseExtremeRegressor([tiny_original, maximum, -maximum, 0.0])
    ratio_explanation = _explanation([3.0, 2.0, 1.0])
    irof = compute_irof(
        ratio_model,
        np.ones(3),
        ratio_explanation,
        baseline=0.0,
        segment_size=1,
        return_details=True,
        detail_format="scaled_decimal_v1",
    )
    region = compute_region_perturbation(
        ratio_model,
        np.ones(3),
        ratio_explanation,
        baseline=0.0,
        region_size=1,
        return_curve=True,
        detail_format="scaled_decimal_v1",
    )
    extreme_ratio = Fraction.from_float(maximum) / Fraction.from_float(tiny_original)
    assert extreme_ratio in decode_scaled_detail(irof["normalised_predictions"])
    assert extreme_ratio in decode_scaled_detail(region["curve"])
    assert all(isinstance(value, str) for value in irof["segment_importance_exact_decimal"])
    for result, key in ((irof, "normalised_predictions"), (region, "curve")):
        assert json.loads(json.dumps(result[key], allow_nan=False)) == result[key]
        assert any(
            "exact_fraction" in item for item in result[key]["values"] if isinstance(item, dict)
        )

    coefficients = np.array([1.0, 1.0, 0.5, 0.25])
    extreme_attributions = maximum * coefficients
    sensitivity_kwargs = {
        "baseline": 0.0,
        "n": 2,
        "n_subsets": 64,
        "seed": 3,
        "return_details": True,
    }
    with pytest.raises(DetailRepresentationError, match="attribution_sum"):
        compute_sensitivity_n(
            _LinearRegressor(coefficients),
            np.ones(4),
            _explanation(extreme_attributions),
            **sensitivity_kwargs,
        )
    sensitivity = compute_sensitivity_n(
        _LinearRegressor(coefficients),
        np.ones(4),
        _explanation(extreme_attributions),
        detail_format="scaled_decimal_v1",
        **sensitivity_kwargs,
    )
    assert sensitivity["correlation"] == pytest.approx(1.0)
    assert any(
        isinstance(value, Fraction) and value > Fraction.from_float(maximum)
        for value in decode_scaled_detail(sensitivity["attribution_sums"])
    )


def test_seeded_replicate_summary_reports_uncertainty_and_declared_convergence_only():
    values_by_seed = {11: 1.0, 12: 2.0, 13: 3.0, 14: 4.0}
    result = run_seeded_replicates(
        lambda *, seed: values_by_seed[seed],
        seeds=list(values_by_seed),
        sample_count=100,
        convergence_tolerance=1.0,
    )

    assert result["estimate"] == 2.5
    assert result["replicate_estimates"] == [1.0, 2.0, 3.0, 4.0]
    assert result["seeds"] == [11, 12, 13, 14]
    assert result["sample_count_per_replicate"] == 100
    assert result["confidence_interval_defined"] is True
    assert result["confidence_interval"][0] < result["estimate"]
    assert result["confidence_interval"][1] > result["estimate"]
    expected_standard_error = np.std(list(values_by_seed.values()), ddof=1) / np.sqrt(4)
    critical = float(stats.t.ppf(0.975, 3))
    assert result["standard_error"] == expected_standard_error
    assert result["confidence_interval"] == [
        result["estimate"] - critical * expected_standard_error,
        result["estimate"] + critical * expected_standard_error,
    ]
    assert result["confidence_interval_computation"] == "direct_float64_student_t"
    assert result["converged_under_declared_tolerance"] is True
    assert result["convergence_diagnostic_only"] is True
    assert result["finite_estimate_is_global_proof"] is False

    singleton = summarize_replicate_estimates([1.0], seeds=[7], sample_count=5)
    assert singleton["confidence_interval_defined"] is False
    assert singleton["confidence_interval"] is None
    assert "at_least_two" in singleton["confidence_interval_reason"]

    with pytest.raises(ValueError, match="unique"):
        summarize_replicate_estimates([1.0, 2.0], seeds=[7, 7], sample_count=5)

    calls: list[int] = []
    with pytest.raises(TypeError, match="seeds must be a sequence"):
        run_seeded_replicates(
            lambda *, seed: calls.append(seed) or float(seed),
            seeds=(seed for seed in [1, 2]),
            sample_count=1,
        )
    assert calls == []


def test_convergence_uses_exact_cumulative_means_before_display_rounding():
    next_float = float(np.nextafter(1.0, 2.0))
    values = [1.0, next_float, 1.0]
    result = summarize_replicate_estimates(
        values,
        seeds=[1, 2, 3],
        sample_count=1,
        convergence_tolerance=0.0,
    )

    exact_values = [Fraction.from_float(value) for value in values]
    penultimate_mean = sum(exact_values[:2], start=Fraction(0)) / 2
    terminal_mean = sum(exact_values, start=Fraction(0)) / 3
    exact_change = abs(terminal_mean - penultimate_mean)

    assert exact_change == Fraction(1, 27021597764222976)
    assert result["cumulative_means"] == [1.0, 1.0, 1.0]
    assert result["terminal_cumulative_mean_change"] == float(exact_change)
    assert result["terminal_cumulative_mean_change"] > 0.0
    assert result["terminal_cumulative_mean_change_defined"] is True
    assert result["terminal_cumulative_mean_change_reason"] is None
    assert (
        result["terminal_cumulative_mean_change_computation"]
        == "exact_binary64_rational_cumulative_means"
    )
    assert result["converged_under_declared_tolerance"] is False


def test_student_t_interval_fuses_subnormal_standard_error_before_rounding():
    minimum_subnormal = float(np.nextafter(0.0, 1.0))
    values = [
        minimum_subnormal,
        minimum_subnormal,
        2.0 * minimum_subnormal,
        2.0 * minimum_subnormal,
    ]
    result = summarize_replicate_estimates(
        values,
        seeds=[1, 2, 3, 4],
        sample_count=1,
        convergence_tolerance=0.0,
    )

    exact_values = [Fraction.from_float(value) for value in values]
    exact_mean = sum(exact_values, start=Fraction(0)) / len(exact_values)
    exact_variance = sum(
        ((value - exact_mean) ** 2 for value in exact_values),
        start=Fraction(0),
    ) / (len(exact_values) - 1)
    with localcontext() as context:
        context.prec = 4000
        mean_decimal = Decimal(exact_mean.numerator) / Decimal(exact_mean.denominator)
        variance_decimal = Decimal(exact_variance.numerator) / Decimal(exact_variance.denominator)
        exact_standard_error = (variance_decimal / Decimal(len(values))).sqrt()
        exact_margin = Decimal.from_float(float(stats.t.ppf(0.975, 3))) * exact_standard_error
        expected_interval = [
            float(mean_decimal - exact_margin),
            float(mean_decimal + exact_margin),
        ]

    assert float(exact_standard_error) == 0.0
    assert result["standard_error"] is None
    assert result["standard_error_defined"] is False
    assert "not_representable" in result["standard_error_reason"]
    assert result["confidence_interval"] == expected_interval
    assert result["confidence_interval"][0] < result["confidence_interval"][1]
    assert result["confidence_interval_computation"] == (
        "high_precision_fused_student_t_from_exact_binary64_replicates"
    )
    assert result["terminal_cumulative_mean_change"] is None
    assert result["terminal_cumulative_mean_change_defined"] is False
    assert (
        result["terminal_cumulative_mean_change_reason"]
        == "positive_terminal_change_is_not_representable_as_float64"
    )
    assert (
        result["terminal_cumulative_mean_change_computation"]
        == "exact_binary64_rational_cumulative_means"
    )
    assert result["converged_under_declared_tolerance"] is False


def test_student_t_interval_uses_finite_upper_tail_at_largest_confidence_level():
    confidence_level = np.nextafter(1.0, 0.0)
    result = summarize_replicate_estimates(
        [1.0, 2.0],
        seeds=[31, 37],
        sample_count=100,
        confidence_level=confidence_level,
    )

    # With two replicates df=1, the Student-t inverse survival function has
    # the independent closed form cot(pi * upper_tail_probability).
    upper_tail_probability = (1.0 - confidence_level) / 2.0
    expected_critical = 1.0 / math.tan(math.pi * upper_tail_probability)
    expected_margin = expected_critical * 0.5
    observed_lower, observed_upper = result["confidence_interval"]

    assert np.isfinite([observed_lower, observed_upper]).all()
    assert observed_lower == pytest.approx(1.5 - expected_margin, rel=2e-15)
    assert observed_upper == pytest.approx(1.5 + expected_margin, rel=2e-15)
    assert result["confidence_interval_computation"] == "direct_float64_student_t"


def test_uncertainty_entry_points_reject_nonzero_values_that_float_cast_to_zero():
    tiny = [Fraction(1, 10**400), Fraction(2, 10**400)]

    with pytest.raises(ValueError, match="not losslessly representable"):
        summarize_replicate_estimates(tiny, seeds=[41, 43], sample_count=10)
    with pytest.raises(ValueError, match="not losslessly representable"):
        run_seeded_replicates(
            lambda seed: tiny[seed],
            seeds=[0, 1],
            sample_count=10,
        )
    with pytest.raises(ValueError, match="not losslessly representable"):
        evaluate_intervention_sensitivity(
            {"first": 0, "second": 1},
            lambda reference: tiny[reference],
            intervention_contract="declared_tiny_fraction_score",
        )

    ordinary = evaluate_intervention_sensitivity(
        {"first": 0, "second": 1},
        float,
        intervention_contract="declared_tiny_fraction_score",
    )
    lossy_report = dict(ordinary)
    lossy_report["scores"] = {"first": tiny[0], "second": tiny[1]}
    with pytest.raises(ValueError, match="not losslessly representable"):
        compare_intervention_sensitivity_reports({"ordinary": ordinary, "lossy": lossy_report})


def test_uncertainty_lossless_binary64_gate_preserves_exact_supported_scalar_boundaries():
    estimates = [
        Fraction(1, 2),
        Decimal("0.75"),
        np.float16(1.0),
        np.float32(1.25),
        np.float64(1.5),
        np.int8(2),
        3,
    ]
    result = summarize_replicate_estimates(
        estimates,
        seeds=[2, 3, 5, 7, 11, 13, 17],
        sample_count=10,
        confidence_level=Fraction(1, 2),
        convergence_tolerance=Decimal("0.25"),
    )
    assert result["replicate_estimates"] == [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    assert result["confidence_level"] == 0.5
    assert result["convergence_tolerance"] == 0.25

    exact_outputs = [Fraction(1, 2), Decimal("0.75")]
    seeded = run_seeded_replicates(
        lambda seed: exact_outputs[seed],
        seeds=[0, 1],
        sample_count=10,
    )
    assert seeded["replicate_estimates"] == [0.5, 0.75]
    sensitivity = evaluate_intervention_sensitivity(
        {"fraction": 0, "decimal": 1},
        lambda reference: exact_outputs[reference],
        intervention_contract="declared_exact_binary64_score",
    )
    assert sensitivity["scores"] == {"fraction": 0.5, "decimal": 0.75}
    external_exact_scores = dict(sensitivity)
    external_exact_scores["scores"] = {
        "fraction": Fraction(1, 2),
        "decimal": Decimal("0.75"),
    }
    comparison = compare_intervention_sensitivity_reports(
        {"generated": sensitivity, "external": external_exact_scores}
    )
    assert comparison["scores_by_report"]["external"] == {
        "fraction": 0.5,
        "decimal": 0.75,
    }

    class UndeclaredFloatSubclass(float):
        pass

    rejected = [
        Fraction(1, 10),
        Decimal("0.1"),
        2**53 + 1,
        np.int64(2**53 + 1),
        Fraction(10**400, 1),
        UndeclaredFloatSubclass(0.5),
    ]
    if np.dtype(np.longdouble).itemsize > np.dtype(np.float64).itemsize:
        rejected.append(np.longdouble("0.5"))
    for value in rejected:
        with pytest.raises((TypeError, ValueError), match="lossless|wider than binary64"):
            summarize_replicate_estimates([value], seeds=[19], sample_count=10)

    with pytest.raises(ValueError, match="not losslessly representable"):
        summarize_replicate_estimates(
            [0.5, 1.0],
            seeds=[23, 29],
            sample_count=10,
            confidence_level=Decimal("0.95"),
        )


def test_named_intervention_sensitivity_requires_one_shared_comparison_contract():
    report_a = evaluate_intervention_sensitivity(
        {"zero": 0.0, "one": 1.0, "two": 2.0},
        lambda baseline: 2.0 * baseline,
        intervention_contract="replace_each_feature_with_declared_scalar",
    )
    report_b = evaluate_intervention_sensitivity(
        {"zero": 0.0, "one": 1.0, "two": 2.0},
        lambda baseline: baseline + 1.0,
        intervention_contract="replace_each_feature_with_declared_scalar",
    )

    assert report_a["scores"] == {"zero": 0.0, "one": 2.0, "two": 4.0}
    assert report_a["sensitivity_range"] == 4.0
    assert report_a["universal_default_claimed"] is False
    comparison = compare_intervention_sensitivity_reports(
        {"estimator_a": report_a, "estimator_b": report_b}
    )
    assert comparison["automatic_best_estimator_selected"] is False
    assert comparison["intervention_names"] == ["zero", "one", "two"]
    assert (
        comparison["intervention_reference_fingerprints"]
        == report_a["intervention_reference_fingerprints"]
    )
    assert json.loads(json.dumps(report_a, allow_nan=False)) == report_a

    repeated = evaluate_intervention_sensitivity(
        {"zero": 0.0, "one": 1.0, "two": 2.0},
        lambda baseline: baseline,
        intervention_contract="replace_each_feature_with_declared_scalar",
    )
    assert (
        repeated["intervention_reference_fingerprints"]
        == report_a["intervention_reference_fingerprints"]
    )

    different_reference = evaluate_intervention_sensitivity(
        {"zero": 0.0, "one": 1.0, "two": 3.0},
        lambda baseline: baseline,
        intervention_contract="replace_each_feature_with_declared_scalar",
    )
    with pytest.raises(ValueError, match="different intervention reference"):
        compare_intervention_sensitivity_reports({"a": report_a, "b": different_reference})

    mixed = dict(report_b)
    mixed["intervention_contract"] = "conditional_background_resampling"
    with pytest.raises(ValueError, match="mixed intervention_contract"):
        compare_intervention_sensitivity_reports({"a": report_a, "b": mixed})


def test_intervention_array_fingerprints_are_exact_deterministic_and_value_sensitive():
    references = {
        "left": np.array([0.0, 1.0], dtype=np.float64),
        "right": np.array([1.0, 0.0], dtype=np.float64),
    }
    report = evaluate_intervention_sensitivity(
        references,
        lambda reference: float(np.mean(reference)),
        intervention_contract="replace_with_declared_float64_background",
    )
    repeated = evaluate_intervention_sensitivity(
        {name: value.copy() for name, value in references.items()},
        lambda reference: float(np.mean(reference)),
        intervention_contract="replace_with_declared_float64_background",
    )
    assert (
        report["intervention_reference_fingerprints"]
        == repeated["intervention_reference_fingerprints"]
    )

    changed_references = {name: value.copy() for name, value in references.items()}
    changed_references["right"][0] = np.nextafter(1.0, 0.0)
    changed = evaluate_intervention_sensitivity(
        changed_references,
        lambda reference: float(np.mean(reference)),
        intervention_contract="replace_with_declared_float64_background",
    )
    with pytest.raises(ValueError, match="different intervention reference"):
        compare_intervention_sensitivity_reports({"original": report, "changed": changed})


def test_intervention_scalar_fingerprints_reject_any_precision_narrowing():
    float32_report = evaluate_intervention_sensitivity(
        {"left": np.float32(0.25), "right": np.float32(0.75)},
        float,
        intervention_contract="declared_numpy_scalar_reference",
    )
    repeated = evaluate_intervention_sensitivity(
        {"left": np.float32(0.25), "right": np.float32(0.75)},
        float,
        intervention_contract="declared_numpy_scalar_reference",
    )
    float64_report = evaluate_intervention_sensitivity(
        {"left": np.float64(0.25), "right": np.float64(0.75)},
        float,
        intervention_contract="declared_numpy_scalar_reference",
    )
    assert (
        float32_report["intervention_reference_fingerprints"]
        == repeated["intervention_reference_fingerprints"]
    )
    with pytest.raises(ValueError, match="different intervention reference"):
        compare_intervention_sensitivity_reports(
            {"float32": float32_report, "float64": float64_report}
        )

    class UndeclaredFloatSubclass(float):
        pass

    with pytest.raises(TypeError, match="no supported exact fingerprint"):
        evaluate_intervention_sensitivity(
            {"left": UndeclaredFloatSubclass(0.25), "right": 0.75},
            float,
            intervention_contract="declared_numpy_scalar_reference",
        )

    if np.dtype(np.longdouble).itemsize > np.dtype(np.float64).itemsize:
        first = np.longdouble("1")
        second = np.nextafter(first, np.longdouble("2"))
        assert first != second
        assert float(first) == float(second)
        with pytest.raises(TypeError, match="wider than binary64"):
            evaluate_intervention_sensitivity(
                {"first": first, "second": second},
                float,
                intervention_contract="declared_numpy_scalar_reference",
            )


def test_intervention_integer_and_boolean_fingerprints_preserve_scalar_type_identity():
    def report(left, right):
        return evaluate_intervention_sensitivity(
            {"left": left, "right": right},
            float,
            intervention_contract="declared_scalar_type_sensitive_reference",
        )

    integer_reports = {
        "python": report(1, 2),
        "int8": report(np.int8(1), np.int8(2)),
        "int64": report(np.int64(1), np.int64(2)),
    }
    fingerprints = {
        name: tuple(value["intervention_reference_fingerprints"].values())
        for name, value in integer_reports.items()
    }
    assert len(set(fingerprints.values())) == 3
    for first_name, second_name in (("python", "int8"), ("python", "int64"), ("int8", "int64")):
        with pytest.raises(ValueError, match="different intervention reference"):
            compare_intervention_sensitivity_reports(
                {
                    first_name: integer_reports[first_name],
                    second_name: integer_reports[second_name],
                }
            )

    python_bool = report(True, False)
    numpy_bool = report(np.bool_(True), np.bool_(False))
    assert (
        python_bool["intervention_reference_fingerprints"]
        != numpy_bool["intervention_reference_fingerprints"]
    )
    with pytest.raises(ValueError, match="different intervention reference"):
        compare_intervention_sensitivity_reports(
            {"python_bool": python_bool, "numpy_bool": numpy_bool}
        )
