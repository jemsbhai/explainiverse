"""Adversarial regressions for the v0.14 evaluation remediation."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext
from threading import Event, Thread
from time import perf_counter

import numpy as np
import pytest

from explainiverse.core.explanation import Explanation
from explainiverse.engine.suite import ExplanationSuite
from explainiverse.evaluation import (
    LocalisationMask,
    compare_explainer_faithfulness,
    compute_attribution_iou,
    compute_attribution_localisation,
    compute_attribution_threshold_count,
    compute_avg_sensitivity,
    compute_batch_completeness,
    compute_batch_consistency,
    compute_batch_faithfulness,
    compute_batch_max_sensitivity,
    compute_completeness,
    compute_complexity,
    compute_consistency,
    compute_continuity,
    compute_cross_group_lipschitz_diagnostic,
    compute_deletion_auc,
    compute_energy_based_pointing_game,
    compute_faithfulness_correlation,
    compute_faithfulness_estimate,
    compute_fidelity_gap,
    compute_focus,
    compute_group_metric_disparity,
    compute_infidelity,
    compute_input_invariance,
    compute_insertion_auc,
    compute_irof,
    compute_max_sensitivity,
    compute_prediction_conditioned_metric_disparity,
    compute_region_perturbation,
    compute_relevance_mass_accuracy,
    compute_road,
    compute_selectivity,
    compute_sensitive_attribution_change,
    compute_sensitive_attribution_gap,
    compute_sensitivity_n,
    compute_sparseness,
    compute_symmetry,
)
from explainiverse.evaluation._utils import (
    _stable_dot,
    _stable_mean,
    _stable_mean_square,
    _stable_pearson,
    _stable_std,
    compute_baseline_values,
    compute_prediction_change,
)
from explainiverse.evaluation.faithfulness import compute_comprehensiveness
from explainiverse.evaluation.metrics import _score_predictions, compute_aopc
from explainiverse.evaluation.randomisation import (
    _add_noise_to_input,
    _classification_output_width,
    _randomise_layer_parameters,
    _ssim_similarity,
    compute_data_randomisation,
    compute_random_logit_score,
    compute_smooth_mprt,
)
from explainiverse.evaluation.stability import compute_lipschitz_estimate


class _AttributionExplainer:
    def __init__(self, values):
        self.values = list(values)

    def explain(self, _instance):
        names = [f"f{i}" for i in range(len(self.values))]
        return Explanation(
            "fixed",
            "output_0",
            {"feature_attributions": dict(zip(names, self.values, strict=True))},
            names,
        )


class _LinearRegressor:
    _estimator_type = "regressor"

    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=np.float64)

    def predict(self, X):
        return np.asarray(X, dtype=np.float64) @ self.weights


class _ZeroRegressor:
    _estimator_type = "regressor"
    task = "regression"

    def predict(self, X):
        return np.zeros(len(X), dtype=np.float64)


class _StepwiseExtremeRegressor:
    """Return a prescribed value for each cumulative zero-baseline step."""

    _estimator_type = "regressor"
    task = "regression"

    def __init__(self, outputs_by_removed):
        self.outputs_by_removed = np.asarray(outputs_by_removed, dtype=np.float64)

    def predict(self, X):
        rows = np.asarray(X, dtype=np.float64)
        removed = np.count_nonzero(rows == 0.0, axis=1)
        return self.outputs_by_removed[removed]


def _decimal_dot_oracle(left, right):
    with localcontext() as context:
        context.prec = 3000
        exact = sum(
            (
                Decimal.from_float(float(left_value)) * Decimal.from_float(float(right_value))
                for left_value, right_value in zip(left, right, strict=True)
            ),
            start=Decimal(0),
        )
    return float(exact)


def _decimal_mean_difference_oracle(anchor, values):
    with localcontext() as context:
        context.prec = 3000
        exact = Decimal.from_float(float(anchor)) - sum(
            (Decimal.from_float(float(value)) for value in values),
            start=Decimal(0),
        ) / Decimal(len(values))
    return float(exact)


def _decimal_pearson_oracle(left, right):
    with localcontext() as context:
        context.prec = 3000
        left_decimal = [Decimal.from_float(float(value)) for value in left]
        right_decimal = [Decimal.from_float(float(value)) for value in right]
        count = Decimal(len(left_decimal))
        left_sum = sum(left_decimal, start=Decimal(0))
        right_sum = sum(right_decimal, start=Decimal(0))
        cross_sum = sum(
            (
                left_value * right_value
                for left_value, right_value in zip(left_decimal, right_decimal, strict=True)
            ),
            start=Decimal(0),
        )
        left_square_sum = sum((value * value for value in left_decimal), start=Decimal(0))
        right_square_sum = sum((value * value for value in right_decimal), start=Decimal(0))
        covariance = count * cross_sum - left_sum * right_sum
        left_variance = count * left_square_sum - left_sum * left_sum
        right_variance = count * right_square_sum - right_sum * right_sum
        exact = covariance / (left_variance * right_variance).sqrt()
    return float(exact)


def _explanation(values, target="output_0", contract="same"):
    names = [f"f{i}" for i in range(len(values))]
    return Explanation(
        "fixed",
        target,
        {"feature_attributions": dict(zip(names, values, strict=True))},
        names,
        {"comparison_contract": contract},
    )


def test_stable_dot_preserves_exact_product_rounding_cancellation():
    small = 1e-100
    large = 1e100
    next_lower_large = np.nextafter(large, 0.0)
    left = np.array([small, -small])
    right = np.array([large, next_lower_large])
    expected = _decimal_dot_oracle(left, right)

    # Summing the already-rounded products loses almost half the residual.
    rounded_product_result = float(np.sum(left * right))
    assert expected == 1.942668892225729e-16
    assert rounded_product_result == 1.1102230246251565e-16
    assert _stable_dot(left, right) == expected


def test_stable_dot_recovers_a_min_subnormal_after_product_underflow():
    minimum_subnormal = np.nextafter(0.0, 1.0)
    left = np.array([minimum_subnormal, minimum_subnormal])
    right = np.array([0.5, 0.5])

    assert np.all(left * right == 0.0)
    assert _decimal_dot_oracle(left, right) == minimum_subnormal
    assert _stable_dot(left, right) == minimum_subnormal


def test_extreme_pearson_matches_an_exact_decimal_oracle_through_public_api():
    left = np.array([1e308, 1.0, -1e308])
    right = np.array([-1.0, 2.0, -1.0])
    expected = _decimal_pearson_oracle(left, right)

    class SingleFeatureDropRegressor:
        _estimator_type = "regressor"
        task = "regression"

        def predict(self, X):
            predictions = []
            for row in np.asarray(X, dtype=np.float64):
                removed = np.flatnonzero(row == 0.0)
                predictions.append(0.0 if removed.size == 0 else -right[removed[0]])
            return np.asarray(predictions)

    assert expected == 5.773502691896257e-309
    assert _stable_pearson(left, right) == expected
    assert compute_random_logit_score(left, right, "pearson") == expected
    assert (
        compute_faithfulness_correlation(
            SingleFeatureDropRegressor(),
            np.ones(3),
            _explanation(left),
            baseline=0.0,
            subset_size=1,
        )
        == expected
    )
    assert (
        compute_faithfulness_estimate(
            SingleFeatureDropRegressor(),
            np.ones(3),
            _explanation(left),
            baseline=0.0,
            subset_size=1,
        )
        == expected
    )


def test_extreme_spearman_completes_without_range_overflow_warnings():
    maximum = np.finfo(np.float64).max
    left = np.array([-maximum, 0.0, maximum])
    right = np.array([maximum, 0.0, -maximum])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        score = compute_random_logit_score(left, right, "spearman")

    assert score == pytest.approx(-1.0)
    assert caught == []


def test_sparse_mean_square_is_representable_even_when_one_square_is_not():
    residuals = np.array([2e154, 0.0, 0.0, 0.0])
    with localcontext() as context:
        context.prec = 3000
        expected = float(Decimal.from_float(float(residuals[0])) ** 2 / Decimal(4))

    assert expected == 1e308
    assert _stable_mean_square(residuals) == expected
    assert compute_random_logit_score(residuals, np.zeros(4), "mse") == -expected


@pytest.mark.parametrize(
    ("residuals", "expected"),
    [
        (
            np.array(
                [
                    float.fromhex("0x1.6a044b143c9ddp+512"),
                    float.fromhex("0x1.fdb8388f2e6f9p+505"),
                ]
            ),
            np.finfo(np.float64).max,
        ),
        (np.array([2e154, 1e154, 0.0]), None),
    ],
)
def test_mean_square_has_one_exact_rounding_at_the_float_boundary(residuals, expected):
    with localcontext() as context:
        context.prec = 3000
        exact = sum(
            (Decimal.from_float(float(value)) ** 2 for value in residuals),
            start=Decimal(0),
        ) / Decimal(residuals.size)
        decimal_expected = float(exact)

    if expected is not None:
        assert decimal_expected == expected
    assert _stable_mean_square(residuals) == decimal_expected


def test_mean_square_rejects_a_true_result_above_float_max():
    residuals = np.array(
        [
            float.fromhex("0x1.6a033a471b1a2p+512"),
            float.fromhex("0x1.1606932845f76p+506"),
        ]
    )

    with pytest.raises(FloatingPointError, match="mean square is not representable"):
        _stable_mean_square(residuals)


def test_standard_deviation_centers_adjacent_floats_exactly():
    adjacent = np.array([1.0, np.nextafter(1.0, 0.0)])

    assert _stable_std(adjacent) == 2.0**-54


def test_stable_mean_divides_before_its_only_binary64_rounding():
    values = np.array([0.0, 1.0, np.nextafter(1.0, np.inf)])
    with localcontext() as context:
        context.prec = 3000
        expected = float(
            sum(
                (Decimal.from_float(float(value)) for value in values),
                start=Decimal(0),
            )
            / Decimal(values.size)
        )

    assert expected.hex() == "0x1.5555555555556p-1"
    assert _stable_mean(values) == expected


def test_stable_mean_rejects_unrepresentable_underflow_but_keeps_exact_cancellation():
    minimum_subnormal = np.nextafter(0.0, 1.0)

    with pytest.raises(FloatingPointError, match="reduction result is not representable"):
        _stable_mean(np.array([minimum_subnormal, 0.0]))
    assert _stable_mean(np.array([minimum_subnormal, -minimum_subnormal])) == 0.0
    with pytest.raises(FloatingPointError, match="reduction result is not representable"):
        compute_baseline_values("mean", np.array([[minimum_subnormal], [0.0]]), n_features=1)


def test_sparseness_preserves_adjacent_float_inequality():
    values = np.array([1.0, np.nextafter(1.0, 0.0)])
    with localcontext() as context:
        context.prec = 3000
        decimal_values = [Decimal.from_float(float(value)) for value in values]
        expected = float(
            abs(decimal_values[0] - decimal_values[1])
            / (Decimal(2) * sum(decimal_values, start=Decimal(0)))
        )

    assert expected == 2.7755575615628914e-17
    assert compute_sparseness(_AttributionExplainer(values), np.ones(2)) == expected


def test_complexity_entropy_preserves_small_probability_mass():
    small = 2.0**-53
    with localcontext() as context:
        context.prec = 3000
        total = Decimal(1) + Decimal.from_float(small)
        probabilities = [Decimal(1) / total, Decimal.from_float(small) / total]
        expected = float(
            -sum(
                (probability * probability.ln() for probability in probabilities),
                start=Decimal(0),
            )
        )

    assert expected == 4.189626486814324e-15
    assert compute_complexity(_AttributionExplainer([1.0, small]), np.ones(2)) == expected


def test_complexity_entropy_remains_linear_time_for_large_vectors():
    values = np.ones(10_000)
    start = perf_counter()
    result = compute_complexity(_AttributionExplainer(values), values)
    elapsed = perf_counter() - start

    assert result == pytest.approx(np.log(values.size), rel=1e-15)
    assert elapsed < 2.0


def test_complexity_entropy_recovers_a_representable_underflowed_probability_term():
    minimum_subnormal = np.nextafter(0.0, 1.0)
    expected = float.fromhex("0x0.0000000000175p-1022")

    assert (
        compute_complexity(_AttributionExplainer([2.0, minimum_subnormal]), np.ones(2)) == expected
    )


def test_relative_threshold_compares_against_the_exact_binary64_product():
    maximum = 1.1
    threshold = 0.3
    rounded_product = threshold * maximum

    assert (
        compute_attribution_threshold_count(
            _AttributionExplainer([maximum, rounded_product]),
            np.ones(2),
            threshold=threshold,
            threshold_type="relative",
        )
        == 2.0
    )


def test_fairness_gap_and_cohens_d_use_unrounded_group_statistics():
    maximum = np.finfo(np.float64).max
    previous = np.nextafter(maximum, 0.0)
    previous_previous = np.nextafter(previous, 0.0)
    attributions = np.array([[maximum], [previous], [previous], [previous_previous]])

    result = compute_group_metric_disparity(
        attributions,
        np.array(["A", "A", "B", "B"]),
        inner_metric=lambda row: row[0],
    )

    expected_gap = maximum - previous
    assert result["disparity"] == expected_gap
    assert result["pairwise_gaps"][("A", "B")] == expected_gap
    assert result["effect_size"] == np.sqrt(2.0)


def test_batch_faithfulness_combines_score_arrays_before_rounding_aggregates():
    maximum = np.finfo(np.float64).max
    previous = np.nextafter(maximum, 0.0)
    previous_previous = np.nextafter(previous, 0.0)
    unit = maximum - previous

    class MappingRegressor:
        _estimator_type = "regressor"
        task = "regression"

        def predict(self, X):
            mapping = {
                (1.0, 1.0): maximum,
                (2.0, 2.0): maximum,
                (0.0, 1.0): 0.0,
                (1.0, 0.0): unit,
                (0.0, 2.0): unit,
                (2.0, 0.0): 2.0 * unit,
            }
            return np.asarray([mapping[tuple(row)] for row in np.asarray(X)])

    result = compute_batch_faithfulness(
        MappingRegressor(),
        np.array([[1.0, 1.0], [2.0, 2.0]]),
        [_explanation([2.0, 1.0]), _explanation([2.0, 1.0])],
        k=1,
        baseline=0.0,
    )

    with localcontext() as context:
        context.prec = 3000
        exact_ratio = (Decimal.from_float(maximum) + Decimal.from_float(previous)) / (
            Decimal.from_float(previous) + Decimal.from_float(previous_previous)
        )
    assert result["ratio_of_means"] == float(exact_ratio) == np.nextafter(1.0, np.inf)
    assert result["mean_diff"] == unit


def test_subset_correlations_keep_exact_attribution_aggregate_keys():
    adjacent_upper = np.nextafter(1.0, np.inf)

    class SubsetRegressor:
        _estimator_type = "regressor"
        task = "regression"

        def predict(self, X):
            mapping = {
                (1.0, 1.0, 1.0): 0.0,
                (0.0, 0.0, 1.0): -1.0,
                (0.0, 1.0, 0.0): -2.0,
            }
            return np.asarray([mapping.get(tuple(row), -3.0) for row in np.asarray(X)])

    explanation = _explanation([1.0, 1.0, adjacent_upper])
    model = SubsetRegressor()
    estimate = compute_faithfulness_estimate(
        model,
        np.ones(3),
        explanation,
        baseline=0.0,
        subset_size=2,
        n_subsets=2,
        seed=3,
    )
    sensitivity = compute_sensitivity_n(
        model,
        np.ones(3),
        explanation,
        baseline=0.0,
        n=2,
        n_subsets=2,
        seed=3,
    )

    assert estimate == 1.0
    assert sensitivity == 1.0


def test_localisation_mass_ratio_has_one_exact_rounding_and_rejects_underflow():
    small = 2.0**-53
    with localcontext() as context:
        context.prec = 3000
        exact = Decimal.from_float(small) / (Decimal(1) + Decimal.from_float(small))
        expected = float(exact)

    assert expected.hex() == "0x1.fffffffffffffp-54"
    assert compute_attribution_localisation(np.array([1.0, small]), np.array([0, 1])) == expected

    maximum = np.finfo(np.float64).max
    minimum_subnormal = np.nextafter(0.0, 1.0)
    with pytest.raises(FloatingPointError, match="mass ratio is not representable"):
        compute_attribution_localisation(np.array([maximum, minimum_subnormal]), np.array([0, 1]))


def test_localisation_percentile_interpolation_avoids_extreme_subtraction_overflow():
    maximum = np.finfo(np.float64).max
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_attribution_iou(
            np.array([-maximum, maximum]),
            np.array([1, 0]),
            percentile=50.0,
            use_abs=False,
        )

    assert result == 0.0
    assert caught == []


@pytest.mark.parametrize(
    ("values", "percentile"),
    [
        (np.array([1.0, np.nextafter(1.0, np.inf)]), 75.0),
        (np.array([0.0, np.nextafter(0.0, 1.0)]), 25.0),
        (np.array([0.0, np.nextafter(0.0, 1.0)]), 50.0),
        (np.array([0.0, np.nextafter(0.0, 1.0)]), 75.0),
    ],
)
def test_localisation_percentile_keeps_exact_cutoff_for_strict_selection(values, percentile):
    assert (
        compute_attribution_iou(
            values,
            np.array([0, 1]),
            percentile=percentile,
            use_abs=False,
        )
        == 1.0
    )


def test_standard_deviation_rounds_a_representable_subnormal_once():
    minimum_subnormal = np.nextafter(0.0, 1.0)

    assert _stable_std(np.array([2 * minimum_subnormal, 5 * minimum_subnormal])) == (
        2 * minimum_subnormal
    )
    with pytest.raises(FloatingPointError, match="standard deviation is not representable"):
        _stable_std(np.array([2 * minimum_subnormal, 3 * minimum_subnormal]))


def test_median_baseline_avoids_overflow_at_float_max():
    maximum = np.finfo(np.float64).max
    background = np.array([[maximum], [maximum]])
    result = compute_baseline_values("median", background, n_features=1)
    np.testing.assert_array_equal(result, np.array([maximum]))


@pytest.mark.parametrize("curve_kind", ["deletion", "insertion"])
def test_feature_count_curve_integrates_exact_rational_coordinates(curve_kind):
    maximum = np.finfo(np.float64).max
    desired_curve = np.array([1e-308, maximum, -maximum, 0.0, 0.0, 0.0, 0.0])

    class CountRegressor:
        _estimator_type = "regressor"
        task = "regression"

        def predict(self, X):
            results = []
            for row in np.asarray(X):
                count = int(np.count_nonzero(row))
                index = 6 - count if curve_kind == "deletion" else count
                results.append(desired_curve[index])
            return np.asarray(results)

    metric = compute_deletion_auc if curve_kind == "deletion" else compute_insertion_auc
    result = metric(
        CountRegressor(),
        np.ones(6),
        _explanation(np.arange(6.0, 0.0, -1.0)),
        baseline=0.0,
    )

    with localcontext() as context:
        context.prec = 3000
        expected = float(Decimal.from_float(1e-308) / Decimal(12))
    assert result == expected


def test_segment_and_region_ranking_use_exact_aggregate_keys():
    adjacent_upper = np.nextafter(1.0, np.inf)

    class RegionPathRegressor:
        _estimator_type = "regressor"
        task = "regression"

        def predict(self, X):
            results = []
            for row in np.asarray(X):
                first_removed = bool(np.all(row[:2] == 0.0))
                second_removed = bool(np.all(row[2:] == 0.0))
                if first_removed and second_removed:
                    results.append(0.0)
                elif second_removed:
                    results.append(0.2)
                elif first_removed:
                    results.append(0.8)
                else:
                    results.append(1.0)
            return np.asarray(results)

    model = RegionPathRegressor()
    instance = np.ones(4)
    explanation = _explanation([1.0, 1.0, 1.0, adjacent_upper])
    irof = compute_irof(
        model,
        instance,
        explanation,
        baseline=0.0,
        segment_size=2,
        return_details=True,
    )
    region = compute_region_perturbation(
        model,
        instance,
        explanation,
        baseline=0.0,
        region_size=2,
        return_curve=True,
    )

    assert irof["segment_order"] == [1, 0]
    assert irof["aoc"] == pytest.approx(0.65)
    assert region["region_order"] == [1, 0]
    assert region["auc"] == pytest.approx(0.35)


def test_relative_prediction_change_fuses_an_unrepresentable_difference_with_division():
    maximum = np.finfo(np.float64).max

    class SignedMaximumRegressor:
        _estimator_type = "regressor"

        def predict(self, X):
            return np.where(np.asarray(X)[:, 0] > 0.0, maximum, -maximum)

    model = SignedMaximumRegressor()
    assert (
        compute_prediction_change(
            model,
            np.array([1.0]),
            np.array([-1.0]),
            metric="relative",
        )
        == 2.0
    )
    with pytest.raises(FloatingPointError, match="absolute prediction change"):
        compute_prediction_change(
            model,
            np.array([1.0]),
            np.array([-1.0]),
            metric="absolute",
        )


def test_roar_builtin_r2_uses_exact_residual_and_total_sums():
    maximum = np.finfo(np.float64).max
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        score, name, greater_is_better = _score_predictions(
            np.array([-maximum, maximum]),
            np.array([maximum, -maximum]),
            "regression",
            "r2",
            None,
        )

    assert score == -3.0
    assert name == "r2"
    assert greater_is_better is True
    assert caught == []


@pytest.mark.parametrize("radius", [8.9e307, 1e308])
def test_identity_lipschitz_ratio_survives_high_dimensional_extreme_linf_radius(radius):
    class IdentityExplainer:
        def explain(self, instance):
            return _explanation(np.asarray(instance, dtype=np.float64))

    with np.errstate(over="raise", invalid="raise"):
        result = compute_lipschitz_estimate(
            IdentityExplainer(),
            np.zeros(10),
            radius=radius,
            n_samples=1,
            perturb_norm="linf",
            norm_ord=2,
            seed=0,
        )
    assert result == 1.0


def test_infidelity_uses_exact_dot_cancellation_before_squaring():
    small = 1e-100
    large = 1e100
    next_lower_large = np.nextafter(large, 0.0)
    attributions = np.array([small, -small])
    instance = np.array([large, next_lower_large])
    expected_change = _decimal_dot_oracle(attributions, instance)
    with localcontext() as context:
        context.prec = 3000
        expected_infidelity = float(Decimal.from_float(expected_change) ** 2)

    details = compute_infidelity(
        _ZeroRegressor(),
        instance,
        _explanation(attributions),
        baseline=0.0,
        perturbation_type="square",
        noise_scale=1.0,
        n_samples=1,
        seed=0,
        return_details=True,
    )
    assert details["expected_changes"][0] == expected_change
    assert details["actual_changes"][0] == 0.0
    assert details["infidelity"] == expected_infidelity


def test_infidelity_sparse_residual_aggregate_and_detail_only_limit():
    # NumPy's documented generator sequence makes only the third of four
    # one-feature Bernoulli masks true for this seed.
    rng = np.random.default_rng(1)
    assert (rng.random(4) < 0.5).tolist() == [False, False, True, False]
    residual = 2e154
    with localcontext() as context:
        context.prec = 3000
        expected = float(Decimal.from_float(residual) ** 2 / Decimal(4))

    kwargs = {
        "baseline": 0.0,
        "perturbation_type": "square",
        "noise_scale": 0.5,
        "n_samples": 4,
        "seed": 1,
    }
    score = compute_infidelity(_ZeroRegressor(), np.ones(1), _explanation([residual]), **kwargs)
    assert score == expected == 1e308

    with pytest.raises(FloatingPointError, match="individual infidelity squared_errors"):
        compute_infidelity(
            _ZeroRegressor(),
            np.ones(1),
            _explanation([residual]),
            return_details=True,
            **kwargs,
        )


def test_public_aopc_preserves_exact_affine_cancellation_and_owns_detail_limit():
    maximum = np.finfo(np.float64).max
    original = 1e308
    predictions = [original, -original, maximum, maximum, maximum]
    model = _StepwiseExtremeRegressor(predictions)
    explanation = _explanation([4.0, 3.0, 2.0, 1.0])
    expected = _decimal_mean_difference_oracle(original, predictions)

    assert expected == -7.861588091738942e306
    assert compute_aopc(model, np.ones(4), explanation, num_steps=4, baseline_value=0.0) == expected
    with pytest.raises(FloatingPointError, match="individual AOPC prediction_drop"):
        compute_aopc(
            model,
            np.ones(4),
            explanation,
            num_steps=4,
            baseline_value=0.0,
            return_details=True,
        )


def test_selectivity_preserves_exact_affine_cancellation_and_owns_detail_limit():
    maximum = np.finfo(np.float64).max
    original = 1e308
    predictions = [original, -original, maximum, maximum, maximum]
    model = _StepwiseExtremeRegressor(predictions)
    explanation = _explanation([4.0, 3.0, 2.0, 1.0])
    expected = _decimal_mean_difference_oracle(original, predictions)

    assert compute_selectivity(model, np.ones(4), explanation, baseline=0.0, n_steps=4) == expected
    with pytest.raises(FloatingPointError, match="individual selectivity prediction_drop"):
        compute_selectivity(
            model,
            np.ones(4),
            explanation,
            baseline=0.0,
            n_steps=4,
            return_details=True,
        )


def test_road_preserves_exact_affine_cancellation_and_owns_detail_limit():
    maximum = np.finfo(np.float64).max
    original = 1e308
    outputs_by_removed = [original, -original, maximum, maximum, maximum]
    predictions = outputs_by_removed[1:]
    model = _StepwiseExtremeRegressor(outputs_by_removed)
    explanation = _explanation([4.0, 3.0, 2.0, 1.0])
    percentages = [0.13, 0.38, 0.63, 0.88]
    expected = _decimal_mean_difference_oracle(original, predictions)
    kwargs = {
        "background_data": np.zeros((8, 4)),
        "percentages": percentages,
        "noise_scale": 0.0,
        "seed": 3,
    }

    assert compute_road(model, np.ones(4), explanation, **kwargs) == expected
    with pytest.raises(FloatingPointError, match="individual ROAD prediction_change"):
        compute_road(
            model,
            np.ones(4),
            explanation,
            return_details=True,
            **kwargs,
        )


def test_normalized_irof_and_region_aggregate_before_unrepresentable_ratios():
    maximum = np.finfo(np.float64).max
    original = 1e-308
    model = _StepwiseExtremeRegressor([original, maximum, -maximum, 0.0])
    instance = np.ones(3)
    explanation = _explanation([3.0, 2.0, 1.0])

    # Uniform three-interval trapezoidal weights are [1, 2, 2, 1] / 6.
    # The two maximum terms cancel exactly, leaving original / 6.
    assert compute_region_perturbation(
        model, instance, explanation, baseline=0.0, region_size=1
    ) == float(Decimal(1) / Decimal(6))
    assert compute_irof(model, instance, explanation, baseline=0.0, segment_size=1) == float(
        Decimal(5) / Decimal(6)
    )

    with pytest.raises(FloatingPointError, match="per-step normalized curve"):
        compute_region_perturbation(
            model,
            instance,
            explanation,
            baseline=0.0,
            region_size=1,
            return_curve=True,
        )
    with pytest.raises(FloatingPointError, match="per-step normalized details"):
        compute_irof(
            model,
            instance,
            explanation,
            baseline=0.0,
            segment_size=1,
            return_details=True,
        )


def test_scale_invariant_metrics_remain_finite_near_float_max():
    explainer = _AttributionExplainer([1e308, 1e308])
    assert compute_sparseness(explainer, np.ones(2)) == pytest.approx(0.0)
    assert compute_complexity(explainer, np.ones(2)) == pytest.approx(np.log(2.0))

    values = np.array([1e308, 1e308])
    mask = np.array([1.0, 0.0])
    assert compute_attribution_localisation(values, mask) == pytest.approx(0.5)
    assert compute_relevance_mass_accuracy(values, mask) == pytest.approx(0.5)
    assert compute_energy_based_pointing_game(values, mask) == pytest.approx(0.5)
    focus_values = np.full((2, 2), 1e308)
    focus_mask = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert compute_focus(focus_values, focus_mask) == pytest.approx(0.5)


def test_mean_baseline_avoids_overflowing_a_representable_result():
    background = np.array([[1e308, 1e308], [1e308, -1e308]])
    result = compute_baseline_values("mean", background, n_features=2)
    np.testing.assert_array_equal(result, np.array([1e308, 0.0]))


@pytest.mark.parametrize(
    "ordered_values",
    [
        [1e308, 1.0, -1e308],
        [1e308, -1e308, 1.0],
        [1e308, 1e308, -1e308, -1e308, 1.0],
    ],
)
def test_mean_baseline_preserves_compensated_residuals_independent_of_order(ordered_values):
    background = np.asarray(ordered_values, dtype=np.float64).reshape(-1, 1)
    result = compute_baseline_values("mean", background, n_features=1)
    assert result[0] == pytest.approx(1.0 / len(ordered_values))


def test_smooth_mprt_noise_scale_avoids_representable_range_overflow():
    class RecordingRng:
        scale = None

        def normal(self, _loc, scale, size):
            self.scale = scale
            return np.zeros(size, dtype=np.float64)

    rng = RecordingRng()
    source = np.array([[-1e308, 1e308]], dtype=np.float64)
    noisy = _add_noise_to_input(source, 1e-308, rng)

    assert rng.scale == pytest.approx(2.0)
    np.testing.assert_array_equal(noisy, source)


def test_public_smooth_mprt_never_passes_nonfinite_overflow_noise_to_callback():
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False, dtype=torch.float64))
    inputs_seen = []

    def explain_func(_model, x, _target):
        inputs_seen.append(np.asarray(x).copy())
        return np.array([1.0, 2.0])

    result = compute_smooth_mprt(
        model,
        np.array([[-1e308, 1e308]], dtype=np.float64),
        np.array([0]),
        explain_func,
        nr_samples=2,
        noise_magnitude=1e-308,
        seed=7,
    )

    assert result["mean_score"] == pytest.approx(1.0)
    assert inputs_seen
    assert all(np.all(np.isfinite(values)) for values in inputs_seen)


def test_public_smooth_mprt_uses_a_scale_safe_attribution_mean():
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False, dtype=torch.float64))

    def explain_func(_model, _x, _target):
        return np.array([1e308, 1e308])

    result = compute_smooth_mprt(
        model,
        np.array([[1.0, 2.0]], dtype=np.float64),
        np.array([0]),
        explain_func,
        similarity_func="cosine",
        nr_samples=2,
        noise_magnitude=0.0,
        seed=11,
    )

    assert result["mean_score"] == pytest.approx(1.0)


def test_randomisation_attribution_extraction_rejects_complex_values_before_cast():
    with pytest.raises(TypeError, match="complex"):
        compute_random_logit_score(
            np.array([1.0 + 2.0j, 3.0]),
            np.array([1.0, 3.0]),
            similarity_func="cosine",
        )


@pytest.mark.parametrize("similarity", ["cosine", "pearson"])
def test_tiny_nonzero_similarity_vectors_are_not_treated_as_zero(similarity):
    a = np.array([1e-300, 2e-300, 4e-300])
    b = 2.0 * a
    assert compute_random_logit_score(a, b, similarity) == pytest.approx(1.0)


def test_tiny_faithfulness_correlation_uses_a_scaled_pearson_oracle():
    model = _LinearRegressor([1e-300, 2e-300])
    result = compute_faithfulness_correlation(
        model,
        np.ones(2),
        _explanation([1e-300, 2e-300]),
        baseline=0.0,
        subset_size=1,
    )
    assert result == pytest.approx(1.0)


@pytest.mark.parametrize(
    "values",
    [
        [1e308, 1e308, -1e308, -1e308],
        [1e308, -1e308, 1e308, -1e308],
    ],
)
def test_faithfulness_subset_sums_preserve_extreme_signed_cancellation(values):
    class DecimalSumRegressor:
        _estimator_type = "regressor"

        def predict(self, X):
            predictions = []
            for row in np.asarray(X, dtype=np.float64):
                with localcontext() as context:
                    context.prec = 1500
                    total = sum(
                        (Decimal.from_float(float(value)) for value in row),
                        start=Decimal(0),
                    )
                predictions.append(float(total))
            return np.asarray(predictions)

    array = np.asarray(values, dtype=np.float64)
    with np.errstate(over="raise", invalid="raise"):
        result = compute_faithfulness_correlation(
            DecimalSumRegressor(),
            array,
            _explanation(values),
            baseline=0.0,
            subset_size=3,
        )
    assert result == pytest.approx(1.0)


@pytest.mark.parametrize(
    "values",
    [
        [1e308, 1e308, -1e308, -1e308],
        [1e308, -1e308, 1e308, -1e308],
    ],
)
def test_completeness_fuses_extreme_signed_terms_before_rounding(values):
    array = np.asarray(values, dtype=np.float64)
    with np.errstate(over="raise", invalid="raise"):
        result = compute_completeness(array, lambda _x: 0.0, np.zeros(array.size))
    assert result == 0.0


def test_extended_signed_subset_metrics_preserve_extreme_cancellation():
    class DecimalSumRegressor:
        _estimator_type = "regressor"

        def predict(self, X):
            predictions = []
            for row in np.asarray(X, dtype=np.float64):
                with localcontext() as context:
                    context.prec = 1500
                    total = sum(
                        (Decimal.from_float(float(value)) for value in row),
                        start=Decimal(0),
                    )
                predictions.append(float(total))
            return np.asarray(predictions)

    values = np.array([1e308, 1e308, -1e308, -1e308])
    explanation = _explanation(values)
    model = DecimalSumRegressor()
    with np.errstate(over="raise", invalid="raise"):
        faithfulness = compute_faithfulness_estimate(
            model,
            values,
            explanation,
            baseline=0.0,
            subset_size=3,
            n_subsets=24,
            seed=5,
        )
        sensitivity = compute_sensitivity_n(
            model,
            values,
            explanation,
            baseline=0.0,
            n=3,
            n_subsets=24,
            use_absolute=False,
            seed=5,
        )
    assert faithfulness == pytest.approx(1.0)
    assert sensitivity == pytest.approx(1.0)


def test_region_perturbation_signed_importance_never_serializes_overflow():
    class ConstantRegressor:
        _estimator_type = "regressor"

        def predict(self, X):
            return np.ones(len(X), dtype=np.float64)

    values = np.array([1e308, 1e308, -1e308, -1e308])
    with np.errstate(over="raise", invalid="raise"):
        result = compute_region_perturbation(
            ConstantRegressor(),
            np.ones(values.size),
            _explanation(values),
            baseline=0.0,
            region_size=values.size,
            use_absolute=False,
        )
    assert np.isfinite(result)


def test_fairness_group_means_avoid_representable_overflow():
    maximum = np.finfo(np.float64).max
    attrs = np.array([[maximum, 0.0], [maximum, 0.0], [0.0, 0.0], [0.0, 0.0]])
    groups = np.array(["A", "A", "B", "B"])

    with np.errstate(over="raise", invalid="raise"):
        result = compute_group_metric_disparity(attrs, groups)
        sensitive = compute_sensitive_attribution_gap(attrs, groups, 0)
        conditioned = compute_prediction_conditioned_metric_disparity(
            attrs, groups, np.zeros(4, dtype=int)
        )
        fidelity = compute_fidelity_gap(np.array([maximum, maximum, 0.0, 0.0]), groups)

    assert result["group_means"] == {"A": maximum, "B": 0.0}
    assert result["disparity"] == maximum
    assert sensitive["divergence"] == maximum
    assert conditioned["disparity"] == maximum
    assert fidelity["overall_mean"] == maximum / 2.0
    assert fidelity["group_means"] == {"A": maximum, "B": 0.0}
    assert fidelity["max_gap"] == maximum / 2.0
    assert fidelity["mean_gap"] == maximum


def test_fairness_means_preserve_cancellation_and_cohens_d_uses_scaled_variance():
    maximum = np.finfo(np.float64).max
    groups = np.array(["A", "A", "A", "B", "B", "B"])
    scores = np.array([maximum, 1.0, -maximum, 0.0, 0.0, 0.0])
    attrs = scores.reshape(-1, 1)
    result = compute_group_metric_disparity(attrs, groups, inner_metric=lambda row: row[0])
    fidelity = compute_fidelity_gap(scores, groups)
    assert result["group_means"]["A"] == pytest.approx(1.0 / 3.0)
    assert result["disparity"] == pytest.approx(1.0 / 3.0)
    assert fidelity["group_means"]["A"] == pytest.approx(1.0 / 3.0)

    d_result = compute_group_metric_disparity(
        np.array([[maximum], [0.0], [maximum / 2.0], [0.0]]),
        np.array(["A", "A", "B", "B"]),
        inner_metric=lambda row: row[0],
    )
    assert d_result["effect_size"] == pytest.approx(0.4472135954999579)


def test_fairness_rejects_an_unrepresentable_group_gap():
    maximum = np.finfo(np.float64).max
    with pytest.raises(FloatingPointError, match="not representable"):
        compute_group_metric_disparity(
            np.array([[maximum], [-maximum]]),
            np.array(["A", "B"]),
            inner_metric=lambda row: row[0],
        )


@pytest.mark.parametrize("scale", [1e-300, np.finfo(np.float64).max])
def test_cross_group_l2_ratio_is_stable_at_tiny_and_max_scales(scale):
    values = np.array([[scale, 0.0], [0.0, 0.0]])
    with np.errstate(over="raise", under="ignore", invalid="raise"):
        result = compute_cross_group_lipschitz_diagnostic(
            values,
            values.copy(),
            np.array(["A", "B"]),
            distance_threshold=scale,
        )
    assert result["score"] == pytest.approx(1.0)
    assert result["max_ratio"] == pytest.approx(1.0)


def test_sensitive_attribution_change_uses_scale_safe_norms_and_mean():
    maximum = np.finfo(np.float64).max
    inputs = np.array([[0.0, 0.0], [1.0, 0.0]])
    attrs = np.array([[maximum, 0.0], [0.0, 0.0]])

    with np.errstate(over="raise", invalid="raise"):
        intervened = compute_sensitive_attribution_change(
            inputs,
            attrs,
            0,
            counterfactual_explainer=lambda _row: np.zeros(2),
        )
        matched = compute_sensitive_attribution_change(inputs, attrs, 0)

    np.testing.assert_array_equal(intervened["per_instance_scores"], [maximum, 0.0])
    assert intervened["score"] == maximum / 2.0
    np.testing.assert_array_equal(matched["per_instance_scores"], [maximum, maximum])
    assert matched["score"] == maximum


@pytest.mark.parametrize("scale", [1e-300, np.finfo(np.float64).max])
def test_robustness_norm_metrics_are_stable_at_tiny_and_max_scales(scale):
    class DiscontinuousExplainer:
        def explain(self, instance):
            values = [scale, 0.0] if np.all(np.asarray(instance) == 0.0) else [0.0, 0.0]
            return _explanation(values)

    class IdentityExplainer:
        def explain(self, instance):
            return _explanation(np.asarray(instance, dtype=np.float64))

    with np.errstate(over="raise", under="ignore", invalid="raise"):
        maximum = compute_max_sensitivity(
            DiscontinuousExplainer(), np.zeros(2), radius=0.1, n_samples=1, seed=1
        )
        average = compute_avg_sensitivity(
            DiscontinuousExplainer(), np.zeros(2), radius=0.1, n_samples=1, seed=1
        )
        continuity = compute_continuity(
            IdentityExplainer(),
            np.zeros(2),
            np.array([[scale, 0.0]]),
            k_neighbors=1,
        )
    assert maximum == scale
    assert average == scale
    assert continuity == pytest.approx(1.0)


def test_robustness_batch_summary_avoids_representable_overflow():
    maximum = np.finfo(np.float64).max

    class DiscontinuousExplainer:
        def explain(self, instance):
            values = [maximum, 0.0] if np.all(np.asarray(instance) == 0.0) else [0.0, 0.0]
            return _explanation(values)

    result = compute_batch_max_sensitivity(
        DiscontinuousExplainer(),
        np.zeros((2, 2)),
        radius=0.1,
        n_samples=1,
        seed=2,
    )
    assert result["scores"] == [maximum, maximum]
    assert result["mean"] == maximum
    assert result["std"] == 0.0


def test_axiomatic_rms_summaries_and_pair_means_preserve_extreme_scale():
    maximum = np.finfo(np.float64).max

    def discontinuous(instance):
        return np.array([maximum, 0.0]) if np.all(np.asarray(instance) == 0.0) else np.zeros(2)

    rms = compute_input_invariance(discontinuous, np.zeros(2), shift=1.0)
    assert rms == pytest.approx(maximum / np.sqrt(2.0))

    summary = compute_batch_completeness(
        attributions_list=[np.array([maximum]), np.array([maximum])],
        model_fn=lambda _x: 0.0,
        X=np.zeros((2, 1)),
    )
    assert summary["mean"] == maximum
    assert summary["std"] == 0.0

    pair_mean = compute_symmetry(
        np.array([maximum, -maximum, 0.0, 0.0]),
        symmetric_pairs=[(0, 1), (2, 3)],
    )
    assert pair_mean == maximum


def test_stability_lipschitz_uses_scale_safe_tiny_norms():
    class IdentityExplainer:
        def explain(self, instance):
            return _explanation(np.asarray(instance, dtype=np.float64))

    result = compute_lipschitz_estimate(
        IdentityExplainer(),
        np.zeros(2),
        radius=1e-300,
        n_samples=5,
        seed=3,
    )
    assert result == pytest.approx(1.0)


def test_batch_faithfulness_summaries_preserve_float_max_scores():
    maximum = np.finfo(np.float64).max

    class MaxIfAllNonzero:
        _estimator_type = "regressor"

        def predict(self, X):
            values = np.asarray(X, dtype=np.float64)
            return np.where(np.all(values != 0.0, axis=1), maximum, 0.0)

    values = np.ones((2, 2), dtype=np.float64)
    explanations = [_explanation([1.0, 0.0]), _explanation([1.0, 0.0])]
    result = compute_batch_faithfulness(MaxIfAllNonzero(), values, explanations, k=1, baseline=0.0)
    assert result["mean_pgi"] == maximum
    assert result["std_pgi"] == 0.0
    assert result["mean_pgu"] == maximum
    assert result["std_pgu"] == 0.0
    assert result["ratio_of_means"] == pytest.approx(1.0)
    assert result["mean_diff"] == 0.0

    comparison = compare_explainer_faithfulness(
        MaxIfAllNonzero(), values, {"fixed": explanations}, k=1, baseline=0.0
    ).iloc[0]
    assert comparison["mean_pgi"] == maximum
    assert comparison["std_pgi"] == 0.0
    assert comparison["mean_pgu"] == maximum
    assert comparison["std_pgu"] == 0.0
    assert comparison["ratio_of_means"] == pytest.approx(1.0)
    assert comparison["mean_diff"] == 0.0


def test_ssim_owns_small_spatial_window_behavior():
    with pytest.raises(ValueError, match="at least 3"):
        _ssim_similarity(np.zeros((2, 2)), np.ones((2, 2)))
    with pytest.raises(ValueError, match="at least 3"):
        _ssim_similarity(np.zeros((2, 2)), np.zeros((2, 2)))
    a = np.arange(9.0).reshape(3, 3)
    assert _ssim_similarity(a, a.copy()) == pytest.approx(1.0)
    high_dynamic_range = np.array(
        [[-1e308, 0.0, 1e308], [1e308, -1e308, 0.0], [0.0, 1e308, -1e308]]
    )
    assert _ssim_similarity(high_dynamic_range, high_dynamic_range.copy()) == pytest.approx(1.0)


@pytest.mark.parametrize("size", [3, 4, 5, 6, 7])
def test_ssim_supports_each_declared_adaptive_window_size(size):
    values = np.arange(float(size * size)).reshape(size, size)
    assert _ssim_similarity(values, values.copy()) == pytest.approx(1.0)


def test_ssim_adaptive_window_uses_only_chw_spatial_axes():
    values = np.arange(24.0).reshape(2, 3, 4)
    assert _ssim_similarity(values, values.copy()) == pytest.approx(1.0)


def test_localisation_mask_owns_a_read_only_copy_and_cannot_be_reassigned():
    source = np.array([1, 0], dtype=np.int64)
    mask = LocalisationMask(source)
    source[:] = 0

    np.testing.assert_array_equal(mask.mask, np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="read-only"):
        mask.mask[0] = 0.0
    with pytest.raises(ValueError, match="WRITEABLE"):
        mask.mask.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        mask.mask = np.array([0.0, 1.0])
    assert compute_attribution_localisation(np.array([2.0, 1.0]), mask) == pytest.approx(2 / 3)

    # Defensive extraction also covers unusual deserialisers that bypass a
    # frozen dataclass's constructor.
    object.__setattr__(mask, "mask", np.array([2.0, 2.0]))
    with pytest.raises(ValueError, match="binary"):
        compute_attribution_localisation(np.array([2.0, 1.0]), mask)


@pytest.mark.parametrize(
    "metric",
    [compute_attribution_localisation, compute_energy_based_pointing_game],
)
def test_localisation_metrics_reject_string_or_object_arrays_before_numeric_cast(metric):
    valid_values = np.array([1.0, 0.0])
    valid_mask = np.array([1, 0])
    for invalid_values in (
        np.array(["1", "0"]),
        np.array(["1", "0"], dtype=object),
    ):
        with pytest.raises(TypeError, match="numeric dtype"):
            metric(invalid_values, valid_mask)
        with pytest.raises(TypeError, match="numeric or boolean dtype"):
            metric(valid_values, invalid_values)


def test_randomisation_callbacks_receive_fresh_input_copies():
    torch = pytest.importorskip("torch")
    model_a = torch.nn.Sequential(torch.nn.Linear(2, 2))
    model_b = torch.nn.Sequential(torch.nn.Linear(2, 2))
    X = np.array([[1.0, 2.0]], dtype=np.float32)
    original = X.copy()
    seen = []

    def mutating_callback(_model, x, _target):
        seen.append(x.copy())
        x += 10.0
        return np.array([1.0, 2.0])

    assert compute_data_randomisation(
        model_a, model_b, X, np.array([0]), mutating_callback, similarity_func="cosine"
    ) == pytest.approx(1.0)
    np.testing.assert_array_equal(X, original)
    assert len(seen) == 2
    for callback_input in seen:
        np.testing.assert_array_equal(callback_input, original)


def test_randomisation_model_validation_cannot_mutate_caller_input():
    torch = pytest.importorskip("torch")

    class InPlaceInputModel(torch.nn.Module):
        def forward(self, inputs):
            inputs.add_(10.0)
            return torch.column_stack((inputs[:, 0], -inputs[:, 0]))

    X = np.array([[1.0, 2.0]], dtype=np.float32)
    original = X.copy()
    assert _classification_output_width(InPlaceInputModel(), X, "model") == 2
    np.testing.assert_array_equal(X, original)


def test_torch_rng_fork_covers_all_devices_even_for_a_cpu_layer(monkeypatch):
    torch = pytest.importorskip("torch")
    captured = []

    @contextmanager
    def recording_fork_rng(*, devices, enabled):
        captured.append((devices, enabled))
        yield

    class Resettable(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def reset_parameters(self):
            with torch.no_grad():
                self.weight.fill_(2.0)

        def forward(self, x):
            return x * self.weight

    monkeypatch.setattr(torch.random, "fork_rng", recording_fork_rng)
    monkeypatch.setattr(torch, "manual_seed", lambda _seed: None)
    _randomise_layer_parameters(
        torch.nn.Sequential(Resettable()), "0", rng=np.random.default_rng(7)
    )
    assert captured == [(None, True)]


def test_torch_rng_is_restored_when_parameter_reset_raises():
    torch = pytest.importorskip("torch")

    class RaisingReset(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def reset_parameters(self):
            torch.rand(5)
            raise RuntimeError("reset failed")

        def forward(self, x):
            return x * self.weight

    torch.manual_seed(9123)
    state = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="reset failed"):
        _randomise_layer_parameters(
            torch.nn.Sequential(RaisingReset()), "0", rng=np.random.default_rng(8)
        )
    assert torch.equal(torch.random.get_rng_state(), state)


def test_overlapping_parameter_randomisations_serialize_global_torch_rng_forks():
    torch = pytest.importorskip("torch")

    class BlockingReset(torch.nn.Module):
        def __init__(self, entered, release):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.entered = entered
            self.release = release

        def reset_parameters(self):
            torch.rand(5)
            self.entered.set()
            if not self.release.wait(timeout=5):
                raise RuntimeError("test reset release timed out")

        def forward(self, x):
            return x * self.weight

    a_entered, a_release = Event(), Event()
    b_entered, b_release = Event(), Event()
    b_started = Event()
    errors = []
    model_a = torch.nn.Sequential(BlockingReset(a_entered, a_release))
    model_b = torch.nn.Sequential(BlockingReset(b_entered, b_release))

    def run(model, seed, started=None):
        if started is not None:
            started.set()
        try:
            _randomise_layer_parameters(model, "0", rng=np.random.default_rng(seed))
        except Exception as exc:  # pragma: no cover - asserted after joining.
            errors.append(exc)

    torch.manual_seed(3819)
    state = torch.random.get_rng_state().clone()
    thread_a = Thread(target=run, args=(model_a, 1))
    thread_b = Thread(target=run, args=(model_b, 2, b_started))
    thread_a.start()
    assert a_entered.wait(timeout=5)
    thread_b.start()
    assert b_started.wait(timeout=5)
    try:
        assert not b_entered.wait(timeout=0.25)
    finally:
        a_release.set()
        b_release.set()
        thread_a.join(timeout=5)
        thread_b.join(timeout=5)

    assert not thread_a.is_alive() and not thread_b.is_alive()
    assert errors == []
    assert torch.equal(torch.random.get_rng_state(), state)


class _TiedExplainer:
    def explain(self, _instance):
        return _explanation([1.0, 1.0, 1.0], target=0)


class _ConstantModel:
    def predict(self, X):
        return np.zeros(len(X), dtype=int)


def test_consistency_declares_feature_order_as_its_cutoff_tie_policy():
    X = np.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
    assert compute_consistency(_TiedExplainer(), _ConstantModel(), X, top_k=2) == 1.0


def test_batch_consistency_rejects_duplicate_discretisations():
    X = np.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="duplicate"):
        compute_batch_consistency(
            _TiedExplainer(), _ConstantModel(), X, top_k_values=[1, np.int64(1)]
        )


@pytest.mark.parametrize("numpy_value", [np.float16(0.1), np.float32(0.1), np.float64(0.1)])
def test_k_values_reject_numpy_python_semantic_duplicates_before_key_overwrite(numpy_value):
    with pytest.raises(ValueError, match="semantically duplicate"):
        compute_comprehensiveness(
            _LinearRegressor([1.0, 2.0]),
            np.ones(2),
            _explanation([1.0, 2.0]),
            k_values=[0.1, numpy_value],
            baseline=0.0,
        )


def test_suite_compares_semantic_targets_not_repr(capsys):
    suite = ExplanationSuite(None, [("fixed", {})])
    suite.explanations = {
        "python": _explanation([1.0], target=1),
        "numpy": _explanation([1.0], target=np.int64(1)),
    }
    suite.compare()
    assert "Side-by-Side" in capsys.readouterr().out

    class ReprCollision:
        def __repr__(self):
            return "same"

        def __eq__(self, other):
            return self is other

    suite.explanations = {
        "left": _explanation([1.0], target=ReprCollision()),
        "right": _explanation([1.0], target=ReprCollision()),
    }
    with pytest.raises(ValueError, match="different explained targets"):
        suite.compare()


def test_public_boolean_modes_are_strict():
    from explainiverse.evaluation.axiomatic import (
        compute_non_sensitivity,
        compute_non_sensitivity_score,
    )
    from explainiverse.evaluation.metrics import compute_aopc, compute_roar
    from explainiverse.evaluation.robustness import compute_relative_input_stability

    with pytest.raises(TypeError, match="normalize"):
        compute_non_sensitivity(
            np.array([0.0]),
            lambda x: x[0],
            np.array([0.0]),
            non_sensitive_features=np.array([True]),
            normalize="false",
        )

    class MustNotRun:
        def explain(self, _instance):
            raise AssertionError("invalid options must fail before explainer execution")

    with pytest.raises(TypeError, match="normalize"):
        compute_non_sensitivity_score(
            MustNotRun(),
            lambda x: x[0],
            np.array([0.0]),
            non_sensitive_features=np.array([True]),
            normalize="false",
        )
    with pytest.raises(TypeError, match="verify_determinism"):
        compute_non_sensitivity_score(
            MustNotRun(),
            lambda x: x[0],
            np.array([0.0]),
            non_sensitive_features=np.array([True]),
            verify_determinism="false",
        )
    with pytest.raises(TypeError, match="return_details"):
        compute_aopc(None, None, None, return_details="false")
    with pytest.raises(TypeError, match="return_details"):
        compute_roar(None, None, None, None, None, [], return_details="false")
    with pytest.raises(TypeError, match="return_details"):
        compute_relative_input_stability(None, None, None, return_details="false")


def test_registry_manifest_exposes_reviewed_level_and_stochasticity():
    from explainiverse.evaluation import default_metric_registry

    expected = {
        "compute_mprt_score": ("instance", "deterministic"),
        "compute_mprt": ("dataset", "stochastic"),
        "compute_batch_mprt": ("batch", "stochastic"),
        "compute_consistency": ("dataset", "conditional"),
        "compute_batch_consistency": ("dataset", "conditional"),
        "compute_cross_group_lipschitz_diagnostic": ("dataset", "conditional"),
    }
    for name, behavior in expected.items():
        meta = default_metric_registry.get_meta(name)
        assert (meta.level, meta.stochasticity) == behavior
        assert meta.stochastic is (behavior[1] != "deterministic")

    for name in default_metric_registry.list_metrics():
        meta = default_metric_registry.get_meta(name)
        assert meta.stochasticity in {"deterministic", "conditional", "stochastic"}
        assert meta.stochastic is (meta.stochasticity != "deterministic")

    assert "compute_mprt" in default_metric_registry.filter(stochasticity="stochastic")
    assert "compute_consistency" in default_metric_registry.filter(stochasticity="conditional")
    assert "compute_mprt_score" in default_metric_registry.filter(stochasticity="deterministic")
    with pytest.raises(ValueError, match="stochasticity"):
        default_metric_registry.filter(stochasticity="sometimes")
