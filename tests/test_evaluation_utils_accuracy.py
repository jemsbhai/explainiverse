"""Analytical and counterexample tests for shared evaluation helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from explainiverse.core.explanation import Explanation
from explainiverse.evaluation._utils import (
    _get_prediction_proba_vector,
    apply_feature_mask,
    compute_baseline_values,
    compute_prediction_change,
    get_prediction_value,
    get_sorted_feature_indices,
    resolve_k,
    resolve_target_class,
)


def _explanation(attributions, feature_names=None):
    return Explanation(
        explainer_name="analytical",
        target_class="output_0",
        explanation_data={"feature_attributions": attributions},
        feature_names=feature_names,
    )


def test_feature_ranking_resolves_exact_names_lime_intervals_and_index_keys():
    explanation = _explanation(
        {
            "20 < age <= 30": 3.0,
            "feature_1 > 1e-4": -2.0,
            "height": 1.0,
        },
        ["age", "income", "height"],
    )

    assert get_sorted_feature_indices(explanation) == [0, 1, 2]


def test_feature_ranking_ascending_is_stable_across_equal_magnitudes():
    explanation = _explanation({"a": 1.0, "b": -1.0, "c": 2.0}, ["a", "b", "c"])

    assert get_sorted_feature_indices(explanation, descending=False) == [0, 1, 2]


def test_feature_ranking_without_names_requires_complete_explicit_indices():
    assert get_sorted_feature_indices(_explanation({"f2": 4.0, "x0": 1.0})) == [2, 0]
    with pytest.raises(ValueError, match="has no explicit index"):
        get_sorted_feature_indices(_explanation({"contains_f2_text": 1.0}))


def test_feature_ranking_validates_container_and_sort_contracts():
    with pytest.raises(TypeError, match="Explanation"):
        get_sorted_feature_indices({"feature_attributions": {"f0": 1.0}})
    with pytest.raises(TypeError, match="descending"):
        get_sorted_feature_indices(_explanation({"f0": 1.0}), descending=1)
    with pytest.raises(ValueError, match="No feature attributions"):
        get_sorted_feature_indices(_explanation({}))


def test_feature_ranking_rejects_unknown_and_duplicate_resolutions():
    with pytest.raises(ValueError, match="does not resolve"):
        get_sorted_feature_indices(_explanation({"not-age": 1.0}, ["age"]))
    with pytest.raises(ValueError, match="one-to-one"):
        get_sorted_feature_indices(
            _explanation({"feature_0": 2.0, "f0": 1.0}, ["feature_0", "other"])
        )
    with pytest.raises(ValueError, match="non-empty strings"):
        get_sorted_feature_indices(_explanation({"f0": 1.0}, [" "]))


def test_baseline_statistics_are_analytical_and_results_are_defensive_copies():
    background = np.array([[1.0, 8.0], [3.0, 4.0], [11.0, 0.0]])
    source = np.array([7.0, 9.0])

    assert compute_baseline_values("mean", background, 2) == pytest.approx([5.0, 4.0])
    assert compute_baseline_values("median", background, 2) == pytest.approx([3.0, 4.0])
    result = compute_baseline_values(source, n_features=2)
    result[0] = -100.0
    assert source.tolist() == [7.0, 9.0]


def test_callable_baseline_receives_a_copy_and_must_match_background_width():
    background = np.array([[1.0, 2.0], [3.0, 4.0]])

    def mutating_baseline(data):
        data[:] = 0.0
        return np.max(data, axis=0)

    assert compute_baseline_values(mutating_baseline, background) == pytest.approx([0.0, 0.0])
    np.testing.assert_array_equal(background, [[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="background_data columns"):
        compute_baseline_values(lambda data: np.array([0.0]), background)
    with pytest.raises(ValueError, match="background_data is required"):
        compute_baseline_values(lambda data: np.mean(data, axis=0))


@pytest.mark.parametrize(
    ("baseline", "background", "n_features", "error", "match"),
    [
        (np.array([0.0]), None, 2, ValueError, "baseline must resolve to shape"),
        (True, None, 2, TypeError, "baseline must"),
        (0.0, None, None, ValueError, "n_features is required"),
        ("zero", np.ones((2, 2)), 2, ValueError, "'mean' or 'median'"),
        ("mean", np.array([1.0, 2.0]), 2, ValueError, "2-dimensional"),
        ("mean", np.empty((0, 2)), 2, ValueError, "must not be empty"),
        ("mean", np.array([[1.0, np.inf]]), 2, ValueError, "finite"),
        (np.array([0.0, np.nan]), None, 2, ValueError, "finite"),
    ],
)
def test_baseline_contract_rejects_silent_shape_type_and_finite_repairs(
    baseline, background, n_features, error, match
):
    with pytest.raises(error, match=match):
        compute_baseline_values(baseline, background, n_features)


def test_feature_mask_replaces_exact_positions_without_mutating_inputs():
    instance = np.array([10.0, 20.0, 30.0])
    baseline = np.array([-1.0, -2.0, -3.0])

    result = apply_feature_mask(instance, (index for index in [0, 2]), baseline)

    assert result == pytest.approx([-1.0, 20.0, -3.0])
    assert instance == pytest.approx([10.0, 20.0, 30.0])
    assert baseline == pytest.approx([-1.0, -2.0, -3.0])


@pytest.mark.parametrize(
    ("indices", "error", "match"),
    [
        ([0, 0], ValueError, "duplicates"),
        ([-1], ValueError, "outside"),
        ([3], ValueError, "outside"),
        ([True], TypeError, "integers"),
        ([1.0], TypeError, "integers"),
        ("1", TypeError, "iterable"),
    ],
)
def test_feature_mask_rejects_invalid_indices(indices, error, match):
    with pytest.raises(error, match=match):
        apply_feature_mask(np.ones(3), indices, np.zeros(3))


def test_feature_mask_rejects_shape_and_nonfinite_inputs_and_empty_mask_still_copies():
    with pytest.raises(ValueError, match="same shape"):
        apply_feature_mask(np.ones(3), [0], np.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        apply_feature_mask(np.array([1.0, np.nan]), [0], np.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        apply_feature_mask(np.ones(2), [0], np.array([0.0, np.inf]))

    instance = np.array([1.0, 2.0])
    result = apply_feature_mask(instance, [], np.zeros(2))
    assert result == pytest.approx(instance)
    assert result is not instance


def test_resolve_k_distinguishes_integer_counts_from_fractional_floor_counts():
    assert resolve_k(np.int64(20), 5) == 5
    assert resolve_k(np.float64(0.49), 5) == 2
    assert resolve_k(0.01, 5) == 1
    assert resolve_k(1.0, 5) == 5


@pytest.mark.parametrize(
    ("k", "n_features", "error"),
    [
        (True, 3, TypeError),
        (0, 3, ValueError),
        (1.1, 3, ValueError),
        (np.nan, 3, ValueError),
        (1, True, TypeError),
        (1, 0, ValueError),
    ],
)
def test_resolve_k_rejects_ambiguous_or_nonfinite_values(k, n_features, error):
    with pytest.raises(error):
        resolve_k(k, n_features)


class _BinaryProbabilityModel:
    task = "classification"

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        positive = np.where(X[:, 0] > 0.0, 0.9, 0.1)
        return np.column_stack([1.0 - positive, positive])


class _OneColumnBinaryModel(_BinaryProbabilityModel):
    def predict(self, X):
        return super().predict(X)[:, 1:2]


class _ScalarRegression:
    task = "regression"

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, 0]


def test_binary_probability_vectors_and_fixed_class_change_are_exact():
    for model in (_BinaryProbabilityModel(), _OneColumnBinaryModel()):
        assert _get_prediction_proba_vector(model, np.array([1.0])) == pytest.approx([0.1, 0.9])
        assert resolve_target_class(model, np.array([1.0])) == 1
        assert get_prediction_value(model, np.array([1.0])) == pytest.approx(0.9)
        assert get_prediction_value(model, np.array([1.0]), target_class=0) == pytest.approx(0.1)
        assert compute_prediction_change(model, np.array([1.0]), np.array([-1.0])) == pytest.approx(
            0.8
        )


def test_bounded_regression_is_not_reinterpreted_as_binary_probability():
    model = _ScalarRegression()

    assert _get_prediction_proba_vector(model, np.array([0.25])) == pytest.approx([0.25])
    assert resolve_target_class(model, np.array([0.25])) == 0
    assert get_prediction_value(model, np.array([0.25])) == pytest.approx(0.25)
    with pytest.raises(ValueError, match="undefined for regression"):
        get_prediction_value(model, np.array([0.25]), output_type="class")


def test_regression_prefers_predict_even_if_a_misleading_predict_proba_exists():
    class Model(_ScalarRegression):
        def predict_proba(self, X):
            raise AssertionError("regression must not call predict_proba")

    assert get_prediction_value(Model(), np.array([2.5])) == pytest.approx(2.5)


def test_raw_classifier_hard_labels_are_mapped_through_classes_order():
    class HardLabelClassifier:
        _estimator_type = "classifier"
        classes_ = np.array(["zebra", "ant"])

        def predict(self, X):
            return np.array(["ant"])

    model = HardLabelClassifier()
    assert _get_prediction_proba_vector(model, np.array([1.0])) == pytest.approx([0.0, 1.0])
    assert get_prediction_value(model, np.array([1.0]), output_type="class") == 1.0


def test_unknown_predict_vector_is_rejected_instead_of_guessing_its_semantics():
    class AmbiguousModel:
        def predict(self, X):
            return np.array([[0.2, 0.8]])

    with pytest.raises(ValueError, match="ambiguous"):
        _get_prediction_proba_vector(AmbiguousModel(), np.array([1.0]))


@pytest.mark.parametrize(
    ("values", "match"),
    [
        ([0.2, 0.2], "sum to 1"),
        ([-0.1, 1.1], "lie in"),
        ([np.nan, np.nan], "finite"),
    ],
)
def test_classification_probability_contract_rejects_invalid_vectors(values, match):
    class Model:
        task = "classification"

        def predict(self, X):
            return np.array([values], dtype=float)

    with pytest.raises(ValueError, match=match):
        _get_prediction_proba_vector(Model(), np.array([1.0]))


def test_class_metadata_width_shape_and_uniqueness_are_fail_fast():
    class WrongWidth:
        task = "classification"
        classes_ = np.array([0, 1, 2])

        def predict(self, X):
            return np.array([[0.4]])

    class MalformedClasses:
        task = "classification"
        classes_ = np.array([0, 0])

        def predict(self, X):
            return np.array([[0.5, 0.5]])

    with pytest.raises(ValueError, match="exactly two classes"):
        _get_prediction_proba_vector(WrongWidth(), np.array([1.0]))
    with pytest.raises(ValueError, match="unique"):
        _get_prediction_proba_vector(MalformedClasses(), np.array([1.0]))


def test_conflicting_task_and_estimator_metadata_is_rejected():
    class ConflictingModel:
        task = "regression"
        _estimator_type = "classifier"

        def predict(self, X):
            return np.zeros(len(X))

    with pytest.raises(ValueError, match="conflicts"):
        _get_prediction_proba_vector(ConflictingModel(), np.array([1.0]))


def test_one_class_predict_proba_must_still_be_a_complete_probability_vector():
    class InvalidOneClassModel:
        _estimator_type = "classifier"
        classes_ = np.array(["only"])

        def predict_proba(self, X):
            return np.array([[0.4]])

    with pytest.raises(ValueError, match="sum to 1"):
        _get_prediction_proba_vector(InvalidOneClassModel(), np.array([1.0]))


def test_one_class_adapter_indicator_remains_one_complete_output():
    class RawOneClassModel:
        classes_ = np.array(["only"])

    class OneClassAdapter:
        task = "classification"
        model = RawOneClassModel()

        def predict(self, X):
            return np.ones((len(X), 1))

    assert _get_prediction_proba_vector(OneClassAdapter(), np.array([1.0])) == pytest.approx([1.0])


def test_multioutput_regression_requires_selection_except_for_explicit_value_lookup():
    class MultioutputRegression:
        task = "regression"

        def predict(self, X):
            X = np.asarray(X, dtype=float)
            return np.column_stack([X[:, 0], 2.0 * X[:, 0]])

    model = MultioutputRegression()
    assert get_prediction_value(model, np.array([3.0]), target_class=1) == pytest.approx(6.0)
    with pytest.raises(ValueError, match="multi-output regression"):
        resolve_target_class(model, np.array([3.0]))
    with pytest.raises(ValueError, match="multi-output regression"):
        get_prediction_value(model, np.array([3.0]))


def test_prediction_input_and_output_shapes_are_not_flattened_or_truncated():
    model = _BinaryProbabilityModel()
    with pytest.raises(ValueError, match="exactly one row"):
        get_prediction_value(model, np.ones((2, 1)))

    class TooManyRows(_BinaryProbabilityModel):
        def predict(self, X):
            return np.array([[0.1, 0.9], [0.8, 0.2]])

    with pytest.raises(ValueError, match="one scalar/vector"):
        get_prediction_value(TooManyRows(), np.array([1.0]))


@pytest.mark.parametrize(
    ("instance", "error", "match"),
    [
        (np.array([]), ValueError, "must not be empty"),
        (np.array([np.nan]), ValueError, "finite"),
        (np.array([True]), TypeError, "real numeric"),
        (np.array([1.0 + 2.0j]), TypeError, "real numeric"),
    ],
)
def test_prediction_instance_contract_is_finite_real_and_nonempty(instance, error, match):
    with pytest.raises(error, match=match):
        get_prediction_value(_BinaryProbabilityModel(), instance)


def test_prediction_target_index_must_exist():
    with pytest.raises(ValueError, match="target_class=2"):
        get_prediction_value(_BinaryProbabilityModel(), np.array([1.0]), target_class=2)


def test_model_cannot_mutate_the_callers_instance_through_prediction_helper():
    class MutatingRegressor:
        task = "regression"

        def predict(self, X):
            X[0, 0] = 99.0
            return X[:, 0]

    instance = np.array([2.0])
    assert get_prediction_value(MutatingRegressor(), instance) == pytest.approx(99.0)
    assert instance == pytest.approx([2.0])


def test_prediction_change_rejects_feature_and_output_width_changes():
    class VariableWidthModel:
        task = "classification"

        def predict(self, X):
            if X[0, 0] > 0:
                return np.array([[0.2, 0.8]])
            return np.array([[0.2, 0.3, 0.5]])

    with pytest.raises(ValueError, match="same feature shape"):
        compute_prediction_change(_ScalarRegression(), np.ones(2), np.ones(3))
    with pytest.raises(ValueError, match="output width changed"):
        compute_prediction_change(VariableWidthModel(), np.array([1.0]), np.array([-1.0]))


def test_relative_prediction_change_exposes_exact_zero_denominators():
    model = _ScalarRegression()
    assert math.isnan(
        compute_prediction_change(model, np.array([0.0]), np.array([0.0]), metric="relative")
    )
    assert math.isinf(
        compute_prediction_change(model, np.array([0.0]), np.array([1.0]), metric="relative")
    )
    assert compute_prediction_change(
        model, np.array([2.0]), np.array([1.0]), metric="relative"
    ) == pytest.approx(0.5)


def test_invalid_output_arguments_fail_before_invoking_the_model():
    class Bomb:
        task = "classification"

        def predict(self, X):
            raise AssertionError("should not be called")

    with pytest.raises(ValueError, match="not used"):
        get_prediction_value(Bomb(), np.array([1.0]), output_type="class", target_class=0)
    with pytest.raises(ValueError, match="output_type"):
        get_prediction_value(Bomb(), np.array([1.0]), output_type="score")
    with pytest.raises(TypeError, match="target_class"):
        get_prediction_value(Bomb(), np.array([1.0]), target_class=True)
