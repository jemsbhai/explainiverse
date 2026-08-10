"""Deterministic oracles for fixed-class faithfulness evaluation.

The model deliberately flips from class 1 to class 0 when its most important
feature is removed. Every metric must continue measuring P(class 1), rather
than switching to the newly predicted class's confidence.
"""

import numpy as np
import pytest

from explainiverse.core.explanation import Explanation
from explainiverse.evaluation._utils import (
    compute_prediction_change,
    get_prediction_value,
    resolve_target_class,
)
from explainiverse.evaluation.faithfulness_extended import (
    compute_deletion_auc,
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

WEIGHTS = np.array([0.70, 0.14, 0.05])
INSTANCE = np.ones(3)
BASELINE = np.zeros(3)
EXPECTED_REMOVAL_CURVE = np.array([0.90, 0.20, 0.06, 0.01])
EXPECTED_INSERTION_CURVE = np.array([0.01, 0.71, 0.85, 0.90])


class AdditiveClassFlipModel:
    """P(class 1) = 0.01 + x @ WEIGHTS, returned as two columns."""

    task = "classification"

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        positive = np.clip(0.01 + X @ WEIGHTS, 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])


class OneColumnClassFlipModel(AdditiveClassFlipModel):
    """The same binary model exposing only P(class 1)."""

    def predict(self, X):
        return super().predict(X)[:, 1:2]


class BoundedRegressionModel:
    """A bounded scalar regression output must not be treated as binary."""

    task = "regression"

    def predict(self, X):
        return np.full((len(np.atleast_2d(X)), 1), 0.25)


@pytest.fixture(params=[AdditiveClassFlipModel, OneColumnClassFlipModel])
def model(request):
    """Exercise every metric with two-column and sigmoid one-column output."""
    return request.param()


@pytest.fixture
def explanation():
    names = ["feature_0", "feature_1", "feature_2"]
    return Explanation(
        explainer_name="additive-oracle",
        target_class="class_1",
        explanation_data={"feature_attributions": dict(zip(names, WEIGHTS))},
        feature_names=names,
    )


def test_one_column_binary_probability_contract():
    model = OneColumnClassFlipModel()

    assert resolve_target_class(model, INSTANCE) == 1
    assert get_prediction_value(model, INSTANCE) == pytest.approx(0.90)
    assert get_prediction_value(model, INSTANCE, target_class=0) == pytest.approx(0.10)
    assert get_prediction_value(model, INSTANCE, target_class=1) == pytest.approx(0.90)
    assert get_prediction_value(model, INSTANCE, output_type="class") == 1.0
    assert compute_prediction_change(model, INSTANCE, BASELINE) == pytest.approx(0.89)


def test_known_regression_output_remains_scalar():
    model = BoundedRegressionModel()

    assert resolve_target_class(model, INSTANCE) == 0
    assert get_prediction_value(model, INSTANCE) == pytest.approx(0.25)
    assert get_prediction_value(model, INSTANCE, target_class=0) == pytest.approx(0.25)
    with pytest.raises(ValueError, match="target_class=1"):
        get_prediction_value(model, INSTANCE, target_class=1)


def test_road_tracks_original_class_after_class_flip(model, explanation):
    details = compute_road(
        model,
        INSTANCE,
        explanation,
        background_data=np.zeros((8, 3)),
        percentages=[0.34],
        noise_scale=0.0,
        return_details=True,
    )

    assert details["original_prediction"] == pytest.approx(0.90)
    assert details["predictions"] == pytest.approx([0.20])
    assert details["prediction_changes"] == pytest.approx([0.70])
    assert details["score"] == pytest.approx(0.70)


def test_irof_tracks_original_class_after_class_flip(model, explanation):
    details = compute_irof(
        model,
        INSTANCE,
        explanation,
        baseline=BASELINE,
        segment_size=1,
        return_details=True,
    )

    assert details["predictions"] == pytest.approx(EXPECTED_REMOVAL_CURVE)
    assert details["curve"] == pytest.approx(1.0 - EXPECTED_REMOVAL_CURVE / 0.90)


def test_infidelity_tracks_original_class_after_class_flip(model, explanation):
    details = compute_infidelity(
        model,
        INSTANCE,
        explanation,
        baseline=BASELINE,
        perturbation_type="square",
        noise_scale=1.0,
        n_samples=4,
        seed=7,
        return_details=True,
    )

    assert details["original_prediction"] == pytest.approx(0.90)
    assert details["expected_changes"] == pytest.approx(np.full(4, 0.89))
    assert details["actual_changes"] == pytest.approx(np.full(4, 0.89))
    assert details["infidelity"] == pytest.approx(0.0, abs=1e-15)


def test_selectivity_tracks_original_class_after_class_flip(model, explanation):
    details = compute_selectivity(
        model,
        INSTANCE,
        explanation,
        baseline=BASELINE,
        return_details=True,
    )

    assert details["predictions"] == pytest.approx(EXPECTED_REMOVAL_CURVE)
    assert details["prediction_drops"] == pytest.approx(0.90 - EXPECTED_REMOVAL_CURVE)


def test_sensitivity_n_tracks_original_class_after_class_flip(model, explanation):
    details = compute_sensitivity_n(
        model,
        INSTANCE,
        explanation,
        baseline=BASELINE,
        n=1,
        n_subsets=30,
        seed=5,
        return_details=True,
    )

    expected_drops = np.array([WEIGHTS[subset[0]] for subset in details["subsets"]])
    assert set(subset[0] for subset in details["subsets"]) == {0, 1, 2}
    assert details["prediction_drops"] == pytest.approx(expected_drops)
    assert details["correlation"] == pytest.approx(1.0)


def test_region_perturbation_tracks_original_class_after_class_flip(model, explanation):
    details = compute_region_perturbation(
        model,
        INSTANCE,
        explanation,
        baseline=BASELINE,
        region_size=1,
        return_curve=True,
    )

    assert details["predictions"] == pytest.approx(EXPECTED_REMOVAL_CURVE)
    assert details["curve"] == pytest.approx(EXPECTED_REMOVAL_CURVE / 0.90)


def test_pixel_flipping_tracks_original_class_after_class_flip(model, explanation):
    details = compute_pixel_flipping(
        model,
        INSTANCE,
        explanation,
        baseline=BASELINE,
        return_curve=True,
    )

    assert details["predictions"] == pytest.approx(EXPECTED_REMOVAL_CURVE)
    assert details["curve"] == pytest.approx(EXPECTED_REMOVAL_CURVE)


def test_explicit_target_class_agrees_with_explanation_identity(model, explanation):
    class_zero_explanation = Explanation(
        explainer_name=explanation.explainer_name,
        target_class="class_0",
        explanation_data=explanation.explanation_data,
        feature_names=explanation.feature_names,
    )
    details = compute_pixel_flipping(
        model,
        INSTANCE,
        class_zero_explanation,
        baseline=BASELINE,
        target_class=0,
        return_curve=True,
    )

    assert details["predictions"] == pytest.approx(1.0 - EXPECTED_REMOVAL_CURVE)


def test_conflicting_or_unmappable_explanation_target_is_rejected(model, explanation):
    with pytest.raises(ValueError, match="conflicting target"):
        compute_pixel_flipping(
            model,
            INSTANCE,
            explanation,
            baseline=BASELINE,
            target_class=0,
        )

    unmappable = Explanation(
        explainer_name=explanation.explainer_name,
        target_class="free-form-unmapped-label",
        explanation_data=explanation.explanation_data,
        feature_names=explanation.feature_names,
    )
    with pytest.raises(ValueError, match="cannot establish which model output"):
        compute_pixel_flipping(model, INSTANCE, unmappable, baseline=BASELINE)


def test_monotonicity_nguyen_tracks_original_class(model, explanation):
    score = compute_monotonicity_nguyen(
        model,
        INSTANCE,
        explanation,
        baseline=BASELINE,
    )

    assert score == pytest.approx(1.0)


def test_monotonicity_tracks_original_class(model, explanation):
    score = compute_monotonicity(
        model,
        INSTANCE,
        explanation,
        baseline=BASELINE,
    )

    assert score == pytest.approx(1.0)


def test_insertion_and_deletion_remain_sound_for_one_column_binary(explanation):
    model = OneColumnClassFlipModel()

    deletion = compute_deletion_auc(
        model,
        INSTANCE,
        explanation,
        baseline=BASELINE,
        return_curve=True,
    )
    insertion = compute_insertion_auc(
        model,
        INSTANCE,
        explanation,
        baseline=BASELINE,
        return_curve=True,
    )

    assert deletion["target_class"] == 1
    assert deletion["curve"] == pytest.approx(EXPECTED_REMOVAL_CURVE)
    assert insertion["target_class"] == 1
    assert insertion["curve"] == pytest.approx(EXPECTED_INSERTION_CURVE)
