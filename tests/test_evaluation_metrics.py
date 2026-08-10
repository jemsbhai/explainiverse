"""Accuracy contracts for the legacy AOPC and ROAR entry points.

The oracles in this module are deliberately analytical.  In particular, ROAR
must mask every row according to that row's explanation in *both* splits; a
global vote over a subset of training explanations is not ROAR.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from explainiverse.core.explanation import Explanation
from explainiverse.evaluation.metrics import (
    compute_aopc,
    compute_batch_aopc,
    compute_roar,
    compute_roar_curve,
)


def _explanation(
    attributions: dict[str, float],
    *,
    feature_names: list[str],
    target_class: str = "positive",
    class_index: int | None = None,
) -> Explanation:
    metadata = {} if class_index is None else {"class_index": class_index}
    return Explanation(
        explainer_name="analytical",
        target_class=target_class,
        explanation_data={"feature_attributions": attributions},
        feature_names=feature_names,
        metadata=metadata,
    )


def _row_explanations(top_features: list[int], n_features: int = 2) -> list[Explanation]:
    names = [f"f{i}" for i in range(n_features)]
    explanations = []
    for top in top_features:
        attrs = {name: 0.0 for name in names}
        attrs[names[top]] = 1.0
        explanations.append(_explanation(attrs, feature_names=names))
    return explanations


class _SwitchingProbabilityModel:
    """The argmax flips after f0 is masked while max confidence stays 0.9."""

    task = "classification"
    class_names = ["negative", "positive"]

    def predict(self, X):
        X = np.asarray(X)
        positive = np.where(X[:, 0] > 0.5, 0.1, 0.9)
        return np.column_stack([1.0 - positive, positive])


class _AdditiveProbabilityModel:
    task = "classification"
    class_names = ["negative", "positive"]

    def predict(self, X):
        X = np.asarray(X)
        positive = 0.1 + 0.4 * X[:, 0] + 0.3 * X[:, 1]
        return np.column_stack([1.0 - positive, positive])


class _OneColumnBinaryModel:
    task = "classification"

    def predict(self, X):
        X = np.asarray(X)
        positive = np.where(X[:, 0] > 0.5, 0.2, 0.8)
        return positive[:, None]


class _LinearRegressionModel:
    task = "regression"

    def predict(self, X):
        X = np.asarray(X)
        return X[:, 0] + 2.0 * X[:, 1]


class _NonContiguousLabelAdapter:
    task = "classification"
    class_names = ["low", "high"]

    class _RawModel:
        classes_ = np.array([2, 4])

    model = _RawModel()

    def predict(self, X):
        X = np.asarray(X)
        positive = np.where(X[:, 0] > 0.5, 0.2, 0.8)
        return np.column_stack([1.0 - positive, positive])


def test_aopc_tracks_one_fixed_original_class_through_a_class_flip():
    explanation = _explanation({"f0": 1.0}, feature_names=["f0"], target_class="negative")

    # P(original class 0) drops 0.8.  Samek's denominator is L + 1, so
    # the zero-change k=0 term makes the one-step AOPC 0.8 / 2.
    result = compute_aopc(
        _SwitchingProbabilityModel(),
        np.array([1.0]),
        explanation,
        num_steps=1,
        return_details=True,
    )

    assert result["aopc"] == pytest.approx(0.4)
    assert result["target_index"] == 0
    assert result["prediction_drops"] == pytest.approx([0.0, 0.8])


def test_aopc_treats_numeric_explanation_targets_as_labels_before_indices():
    explanation = _explanation({"f0": 1.0}, feature_names=["f0"], target_class=2)
    result = compute_aopc(
        _NonContiguousLabelAdapter(),
        np.array([1.0]),
        explanation,
        num_steps=1,
        return_details=True,
    )

    assert result["target_index"] == 0
    assert result["aopc"] == pytest.approx(0.3)


def test_aopc_uses_signed_drops_and_explicit_explained_output():
    explanation = _explanation({"f0": 1.0}, feature_names=["f0"], target_class="positive")

    # P(class 1) rises from .1 to .9, so the signed AOPC is negative.
    score = compute_aopc(_SwitchingProbabilityModel(), np.array([1.0]), explanation, num_steps=1)

    assert score == pytest.approx(-0.4)


def test_aopc_maps_raw_noncontiguous_labels_without_conflating_display_names():
    explanation = _explanation({"f0": 1.0}, feature_names=["f0"], target_class="4")

    result = compute_aopc(
        _NonContiguousLabelAdapter(),
        np.array([1.0]),
        explanation,
        num_steps=1,
        return_details=True,
    )

    assert result["target_index"] == 1
    assert result["prediction_drops"] == pytest.approx([0.0, -0.6])


def test_aopc_rejects_unmappable_or_conflicting_output_identity():
    unmappable = _explanation(
        {"f0": 1.0},
        feature_names=["f0"],
        target_class="free-form-unmapped-label",
    )
    with pytest.raises(ValueError, match="Cannot map target_class"):
        compute_aopc(_SwitchingProbabilityModel(), np.array([1.0]), unmappable)

    conflicting = _explanation(
        {"f0": 1.0},
        feature_names=["f0"],
        target_class="positive",
        class_index=0,
    )
    with pytest.raises(ValueError, match="conflicting target output"):
        compute_aopc(_SwitchingProbabilityModel(), np.array([1.0]), conflicting)


def test_aopc_expands_one_column_binary_probability_for_both_fixed_classes():
    class_zero = _explanation(
        {"f0": 1.0}, feature_names=["f0"], target_class="ignored", class_index=0
    )
    class_one = _explanation(
        {"f0": 1.0}, feature_names=["f0"], target_class="ignored", class_index=1
    )

    score_zero = compute_aopc(_OneColumnBinaryModel(), np.array([1.0]), class_zero, num_steps=1)
    score_one = compute_aopc(_OneColumnBinaryModel(), np.array([1.0]), class_one, num_steps=1)

    assert score_zero == pytest.approx(0.3)
    assert score_one == pytest.approx(-0.3)


def test_aopc_matches_primary_formula_including_k_zero_term():
    explanation = _explanation({"f0": 2.0, "f1": 1.0}, feature_names=["f0", "f1"])

    # Original P(positive)=.8.  Cumulative masking gives .4 then .1,
    # hence AOPC=(0 + .4 + .7)/(2 + 1).
    score = compute_aopc(
        _AdditiveProbabilityModel(),
        np.array([1.0, 1.0]),
        explanation,
        num_steps=2,
    )

    assert score == pytest.approx(1.1 / 3.0)


def test_aopc_preserves_estimator_order_and_magnitude_ranking_is_explicit():
    explanation = _explanation({"f0": -100.0, "f1": 1.0}, feature_names=["f0", "f1"])

    descending = compute_aopc(
        _AdditiveProbabilityModel(),
        np.array([1.0, 1.0]),
        explanation,
        num_steps=1,
    )
    magnitude = compute_aopc(
        _AdditiveProbabilityModel(),
        np.array([1.0, 1.0]),
        explanation,
        num_steps=1,
        ranking="absolute",
    )

    assert descending == pytest.approx(0.15)  # f1: (0 + .3) / 2
    assert magnitude == pytest.approx(0.2)  # f0: (0 + .4) / 2


def test_aopc_regression_tracks_the_single_output_without_probability_logic():
    explanation = _explanation(
        {"f1": 2.0, "f0": 1.0},
        feature_names=["f0", "f1"],
        target_class="output",
    )

    # f([1,1])=3; cumulative predictions are 1 and 0.
    result = compute_aopc(
        _LinearRegressionModel(),
        np.array([1.0, 1.0]),
        explanation,
        num_steps=2,
        return_details=True,
    )

    assert result["aopc"] == pytest.approx(5.0 / 3.0)
    assert result["task"] == "regression"
    assert result["output_space"] == "regression_output"


def test_aopc_supports_training_only_statistical_baseline_and_lime_condition_names():
    explanation = _explanation({"f0 <= 10": 1.0}, feature_names=["f0"], target_class="positive")
    background = np.array([[0.0], [0.4]])

    result = compute_aopc(
        _SwitchingProbabilityModel(),
        np.array([1.0]),
        explanation,
        num_steps=1,
        baseline_value="mean",
        background_data=background,
        return_details=True,
    )

    assert result["baseline_values"] == pytest.approx([0.2])
    assert result["feature_order"] == [0]


def test_aopc_rejects_unmappable_or_duplicate_features_instead_of_guessing():
    unknown = _explanation({"mystery": 1.0}, feature_names=["f0"])
    duplicate = _explanation({"f0": 2.0, "feature_0": 1.0}, feature_names=["f0"])
    partial = _explanation({"f0": 1.0}, feature_names=["f0", "f1"])

    with pytest.raises(ValueError, match="Cannot map"):
        compute_aopc(_SwitchingProbabilityModel(), np.array([1.0]), unknown)
    with pytest.raises(ValueError, match="same feature index"):
        compute_aopc(_SwitchingProbabilityModel(), np.array([1.0]), duplicate)
    with pytest.raises(ValueError, match="AOPC requires 2"):
        compute_aopc(
            _AdditiveProbabilityModel(),
            np.array([1.0, 1.0]),
            partial,
            num_steps=2,
        )


def test_batch_aopc_requires_one_valid_explanation_per_row_and_does_not_swallow_errors():
    X = np.array([[1.0], [1.0]])
    valid = _explanation({"f0": 1.0}, feature_names=["f0"], target_class="negative")
    invalid = _explanation({"unknown": 1.0}, feature_names=["f0"])

    with pytest.raises(ValueError, match="same number"):
        compute_batch_aopc(_SwitchingProbabilityModel(), X, {"method": [valid]})
    with pytest.raises(ValueError, match="Cannot map"):
        compute_batch_aopc(_SwitchingProbabilityModel(), X, {"method": [valid, invalid]})

    scores = compute_batch_aopc(
        _SwitchingProbabilityModel(), X, {"method": [valid, valid]}, num_steps=1
    )
    assert scores == {"method": pytest.approx(0.4)}


class _RecordingClassifier:
    """Small estimator oracle that records the arrays and seeds it receives."""

    fit_inputs: list[np.ndarray] = []
    predict_inputs: list[np.ndarray] = []
    fit_seeds: list[int | None] = []

    def __init__(self, random_state=None):
        self.random_state = random_state

    def get_params(self, deep=True):
        return {"random_state": self.random_state}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        type(self).fit_inputs.append(np.asarray(X).copy())
        type(self).fit_seeds.append(self.random_state)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        type(self).predict_inputs.append(np.asarray(X).copy())
        return np.full(len(X), self.classes_[0])


def test_roar_masks_each_train_and_test_row_by_its_own_ranking_using_train_baseline():
    _RecordingClassifier.fit_inputs = []
    _RecordingClassifier.predict_inputs = []
    _RecordingClassifier.fit_seeds = []
    X_train = np.array([[1, 10], [2, 20], [3, 30], [4, 40]])
    y_train = np.array([0, 1, 0, 1])
    X_test = np.array([[5, 50], [6, 60]])
    y_test = np.array([0, 1])

    details = compute_roar(
        _RecordingClassifier,
        X_train,
        y_train,
        X_test,
        y_test,
        _row_explanations([0, 1, 0, 1]),
        test_explanations=_row_explanations([1, 0]),
        top_k=1,
        baseline_value="mean",
        n_repeats=1,
        random_state=11,
        task="classification",
        return_details=True,
    )

    expected_train = np.array([[2.5, 10.0], [2.0, 25.0], [2.5, 30.0], [4.0, 25.0]])
    expected_test = np.array([[5.0, 25.0], [2.5, 60.0]])
    assert np.array_equal(_RecordingClassifier.fit_inputs[0], X_train)
    assert np.array_equal(_RecordingClassifier.fit_inputs[1], expected_train)
    assert np.array_equal(_RecordingClassifier.predict_inputs[0], X_test)
    assert np.array_equal(_RecordingClassifier.predict_inputs[1], expected_test)
    assert details["baseline_values"] == pytest.approx([2.5, 25.0])
    assert details["protocol"] == "per_sample_remove_and_retrain"
    assert details["canonical_core_contract"] is True


def test_roar_pairs_clean_and_masked_retraining_seeds_and_repeats_independently():
    _RecordingClassifier.fit_inputs = []
    _RecordingClassifier.predict_inputs = []
    _RecordingClassifier.fit_seeds = []
    X_train = np.array([[-1.0], [1.0]])
    y_train = np.array([0, 1])
    X_test = np.array([[-2.0], [2.0]])
    y_test = np.array([0, 1])
    train_explanations = _row_explanations([0, 0], n_features=1)
    test_explanations = _row_explanations([0, 0], n_features=1)

    compute_roar(
        _RecordingClassifier,
        X_train,
        y_train,
        X_test,
        y_test,
        train_explanations,
        test_explanations=test_explanations,
        top_k=1,
        n_repeats=3,
        random_state=7,
        task="classification",
    )

    assert _RecordingClassifier.fit_seeds == [7, 7, 8, 8, 9, 9]


def test_roar_preserves_estimator_ranking_unless_absolute_is_requested():
    X_train = np.array([[1.0, 10.0], [3.0, 30.0]])
    y_train = np.array([0, 1])
    X_test = np.array([[5.0, 50.0]])
    y_test = np.array([0])
    explanation = _explanation({"f0": -100.0, "f1": 1.0}, feature_names=["f0", "f1"])

    _RecordingClassifier.fit_inputs = []
    _RecordingClassifier.predict_inputs = []
    _RecordingClassifier.fit_seeds = []
    compute_roar(
        _RecordingClassifier,
        X_train,
        y_train,
        X_test,
        y_test,
        [explanation, explanation],
        test_explanations=[explanation],
        top_k=1,
        n_repeats=1,
        task="classification",
    )
    assert np.array_equal(_RecordingClassifier.fit_inputs[1], np.array([[1.0, 20.0], [3.0, 20.0]]))

    _RecordingClassifier.fit_inputs = []
    _RecordingClassifier.predict_inputs = []
    _RecordingClassifier.fit_seeds = []
    details = compute_roar(
        _RecordingClassifier,
        X_train,
        y_train,
        X_test,
        y_test,
        [explanation, explanation],
        test_explanations=[explanation],
        top_k=1,
        n_repeats=1,
        task="classification",
        ranking="absolute",
        return_details=True,
    )
    assert np.array_equal(_RecordingClassifier.fit_inputs[1], np.array([[2.0, 10.0], [2.0, 30.0]]))
    assert details["ranking_transformation_applied"] is True


def test_roar_classification_matches_a_sklearn_retraining_oracle():
    X_train = np.array(
        [
            [-4.0, 0.0],
            [-3.0, 0.0],
            [-2.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ]
    )
    y_train = (X_train[:, 0] > 0).astype(int)
    X_test = np.array([[-3.5, 0.0], [-1.5, 0.0], [1.5, 0.0], [3.5, 0.0]])
    y_test = (X_test[:, 0] > 0).astype(int)
    train_explanations = _row_explanations([0] * len(X_train))
    test_explanations = _row_explanations([0] * len(X_test))

    result = compute_roar(
        DecisionTreeClassifier(max_depth=1),
        X_train,
        y_train,
        X_test,
        y_test,
        train_explanations,
        test_explanations=test_explanations,
        top_k=1,
        baseline_value="mean",
        n_repeats=1,
        random_state=3,
        return_details=True,
    )

    assert result["baseline_score"] == pytest.approx(1.0)
    assert result["retrained_score"] == pytest.approx(0.5)
    assert result["score_drop"] == pytest.approx(0.5)
    assert result["scoring"] == "accuracy"


def test_roar_regression_uses_r2_and_matches_sklearn_retraining_oracle():
    X_train = np.column_stack([np.array([-4, -3, -2, -1, 1, 2, 3, 4]), np.zeros(8)])
    y_train = 3.0 * X_train[:, 0]
    X_test = np.column_stack([np.array([-3.5, -1.5, 1.5, 3.5]), np.zeros(4)])
    y_test = 3.0 * X_test[:, 0]
    train_explanations = _row_explanations([0] * len(X_train))
    test_explanations = _row_explanations([0] * len(X_test))

    result = compute_roar(
        LinearRegression,
        X_train,
        y_train,
        X_test,
        y_test,
        train_explanations,
        test_explanations=test_explanations,
        top_k=1,
        baseline_value="mean",
        n_repeats=1,
        task="regression",
        return_details=True,
    )

    assert result["baseline_score"] == pytest.approx(1.0)
    assert result["retrained_score"] == pytest.approx(0.0)
    assert result["score_drop"] == pytest.approx(1.0)
    assert result["scoring"] == "r2"


def test_roar_requires_and_honours_callable_scorer_direction():
    X_train = np.array([[-4.0, 0.0], [-2.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    y_train = (X_train[:, 0] > 0).astype(int)
    X_test = np.array([[-3.0, 0.0], [3.0, 0.0]])
    y_test = (X_test[:, 0] > 0).astype(int)
    train_explanations = _row_explanations([0] * len(X_train))
    test_explanations = _row_explanations([0] * len(X_test))

    def error_rate(truth, prediction):
        return np.mean(truth != prediction)

    with pytest.raises(ValueError, match="scoring_greater_is_better"):
        compute_roar(
            DecisionTreeClassifier,
            X_train,
            y_train,
            X_test,
            y_test,
            train_explanations,
            test_explanations=test_explanations,
            top_k=1,
            n_repeats=1,
            scoring=error_rate,
        )

    details = compute_roar(
        DecisionTreeClassifier,
        X_train,
        y_train,
        X_test,
        y_test,
        train_explanations,
        test_explanations=test_explanations,
        top_k=1,
        n_repeats=1,
        scoring=error_rate,
        scoring_greater_is_better=False,
        return_details=True,
    )
    assert details["score_drop"] == pytest.approx(0.5)
    assert details["scoring_greater_is_better"] is False


def test_roar_controls_and_reports_nested_estimator_random_states():
    X_train = np.array([[-4.0], [-2.0], [2.0], [4.0]])
    y_train = (X_train[:, 0] > 0).astype(int)
    X_test = np.array([[-3.0], [3.0]])
    y_test = (X_test[:, 0] > 0).astype(int)
    explanations = _row_explanations([0] * len(X_train), n_features=1)
    test_explanations = _row_explanations([0] * len(X_test), n_features=1)
    estimator = make_pipeline(StandardScaler(), DecisionTreeClassifier())

    details = compute_roar(
        estimator,
        X_train,
        y_train,
        X_test,
        y_test,
        explanations,
        test_explanations=test_explanations,
        top_k=1,
        n_repeats=2,
        random_state=17,
        return_details=True,
    )

    assert "decisiontreeclassifier__random_state" in details["random_state_parameters"]
    assert details["nested_random_state_parameters_controlled"] is True
    assert details["repeat_seeds"] == [17, 18]


def test_roar_requires_aligned_test_explanations_and_a_held_out_split():
    X_train = np.array([[-1.0], [1.0]])
    y_train = np.array([0, 1])
    X_test = np.array([[-2.0], [2.0]])
    y_test = np.array([0, 1])
    explanations = _row_explanations([0, 0], n_features=1)

    with pytest.raises(ValueError, match="test_explanations is required"):
        compute_roar(
            DecisionTreeClassifier,
            X_train,
            y_train,
            X_test,
            y_test,
            explanations,
        )
    with pytest.raises(ValueError, match="held-out"):
        compute_roar(
            DecisionTreeClassifier,
            X_train,
            y_train,
            X_train.copy(),
            y_train.copy(),
            explanations,
            test_explanations=explanations,
        )


def test_roar_rejects_partial_explanation_sets_and_invalid_baselines():
    X_train = np.array([[-1.0, 0.0], [1.0, 0.0]])
    y_train = np.array([0, 1])
    X_test = np.array([[-2.0, 0.0], [2.0, 0.0]])
    y_test = np.array([0, 1])
    full = _row_explanations([0, 0])

    with pytest.raises(ValueError, match="training rows"):
        compute_roar(
            DecisionTreeClassifier,
            X_train,
            y_train,
            X_test,
            y_test,
            full[:1],
            test_explanations=full,
        )
    with pytest.raises(ValueError, match="one value per feature"):
        compute_roar(
            DecisionTreeClassifier,
            X_train,
            y_train,
            X_test,
            y_test,
            full,
            test_explanations=full,
            top_k=1,
            baseline_value=np.array([0.0]),
        )


def test_roar_curve_is_a_sequence_of_verified_single_threshold_runs():
    X_train = np.array([[-4.0, -1.0], [-2.0, 1.0], [2.0, -1.0], [4.0, 1.0]])
    y_train = (X_train[:, 0] > 0).astype(int)
    X_test = np.array([[-3.0, 1.0], [3.0, -1.0]])
    y_test = (X_test[:, 0] > 0).astype(int)
    train_explanations = _row_explanations([0] * len(X_train))
    test_explanations = _row_explanations([0] * len(X_test))

    curve = compute_roar_curve(
        DecisionTreeClassifier,
        X_train,
        y_train,
        X_test,
        y_test,
        train_explanations,
        test_explanations=test_explanations,
        max_k=2,
        n_repeats=1,
        random_state=5,
    )

    assert set(curve) == {1, 2}
    assert all(isinstance(value, float) for value in curve.values())
