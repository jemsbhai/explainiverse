"""Adversarial regression tests for the approved non-gradient remediation."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from explainiverse.adapters.base_adapter import BaseModelAdapter
from explainiverse.adapters.sklearn_adapter import SklearnAdapter
from explainiverse.core.explanation import Explanation
from explainiverse.explainers._validation import normalize_classifier_outputs
from explainiverse.explainers.attribution.lime_wrapper import LimeExplainer
from explainiverse.explainers.counterfactual.dice_wrapper import CounterfactualExplainer
from explainiverse.explainers.example_based.protodash import ProtoDashExplainer
from explainiverse.explainers.global_explainers.ale import ALEExplainer
from explainiverse.explainers.global_explainers.partial_dependence import PartialDependenceExplainer
from explainiverse.explainers.global_explainers.permutation_importance import (
    PermutationImportanceExplainer,
)
from explainiverse.explainers.global_explainers.sage import SAGEExplainer


class _ConcreteAdapter(BaseModelAdapter):
    def predict(self, data):
        return np.asarray(data)


class _EndpointBinaryModel:
    task = "classification"
    classes_ = np.array([0, 1])

    def __init__(self, output_kind):
        self.prediction_output_kind = output_kind

    def predict(self, X):
        return (np.asarray(X)[:, 0] >= 0.5).astype(float)


class _ProbabilityModel:
    task = "classification"
    classes_ = np.array([0, 1])
    prediction_output_kind = "probabilities"

    def __init__(self, two_columns=False):
        self.two_columns = two_columns

    def predict(self, X):
        positive = np.full(len(X), 0.5)
        if self.two_columns:
            return np.column_stack((1.0 - positive, positive))
        return positive.reshape(-1, 1)


class _ChangingWidthClassifier:
    task = "classification"
    prediction_output_kind = "probabilities"

    def __init__(self):
        self.calls = 0

    def predict(self, X):
        self.calls += 1
        if self.calls == 1:
            return np.tile(np.array([[0.2, 0.5, 0.3]]), (len(X), 1))
        return np.full((len(X), 1), 0.5)


def test_base_adapter_rejects_whitespace_names_and_validates_output_marker():
    with pytest.raises(ValueError, match="non-whitespace"):
        _ConcreteAdapter(object(), feature_names=["   "])
    with pytest.raises(ValueError, match="prediction_output_kind"):
        _ConcreteAdapter(object(), prediction_output_kind="maybe_probabilities")

    adapter = _ConcreteAdapter(object(), prediction_output_kind="probabilities")
    assert adapter.prediction_output_kind == "probabilities"


def test_explanation_rejects_whitespace_only_feature_names():
    with pytest.raises(ValueError, match="non-empty"):
        Explanation("test", "target", {}, feature_names=["   "])


def test_explicit_output_kind_disambiguates_endpoint_probabilities_and_labels():
    X = np.array([[0.0], [1.0]])
    expected = np.eye(2)

    probabilities = normalize_classifier_outputs(
        _EndpointBinaryModel("probabilities"),
        X,
        context="endpoint oracle",
        require_probabilities=True,
        allow_label_predictions=False,
    )
    np.testing.assert_array_equal(probabilities, expected)

    labels = normalize_classifier_outputs(
        _EndpointBinaryModel("class_labels"),
        X,
        context="endpoint oracle",
        require_probabilities=False,
        allow_label_predictions=True,
    )
    np.testing.assert_array_equal(labels, expected)
    with pytest.raises(ValueError, match="hard class-label"):
        normalize_classifier_outputs(
            _EndpointBinaryModel("class_labels"),
            X,
            context="endpoint oracle",
            require_probabilities=True,
            allow_label_predictions=False,
        )


def test_counterfactual_accepts_declared_endpoint_probability_adapter():
    model = _EndpointBinaryModel("probabilities")
    explainer = CounterfactualExplainer(
        model,
        np.array([[0.0], [1.0]]),
        ["x"],
        random_state=0,
    )

    np.testing.assert_array_equal(
        explainer._predict_probabilities(np.array([[0.0], [1.0]])),
        np.eye(2),
    )


def test_one_column_probability_tie_matches_two_column_argmax_oracle():
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 0])

    one_column = PermutationImportanceExplainer(
        _ProbabilityModel(two_columns=False),
        X,
        y,
        ["x"],
        n_repeats=1,
        random_state=0,
    ).explain()
    two_columns = PermutationImportanceExplainer(
        _ProbabilityModel(two_columns=True),
        X,
        y,
        ["x"],
        n_repeats=1,
        random_state=0,
    ).explain()

    assert one_column.explanation_data["baseline_score"] == 1.0
    assert (
        one_column.explanation_data["baseline_score"]
        == two_columns.explanation_data["baseline_score"]
    )

    sage = SAGEExplainer(
        _ProbabilityModel(two_columns=False),
        X,
        y,
        ["x"],
        n_permutations=1,
        random_state=0,
    )
    one_column_loss = sage._zero_one_loss(np.array([0]), np.array([[0.5]]))
    two_column_loss = sage._zero_one_loss(np.array([0]), np.array([[0.5, 0.5]]))
    assert one_column_loss == two_column_loss == 0.0


@pytest.mark.parametrize("wrapped", [1.0 + 7.0j, np.complex128(1.0 + 7.0j)])
def test_sklearn_adapter_rejects_complex_before_real_cast(wrapped):
    class ComplexRegressor(RegressorMixin, BaseEstimator):
        def predict(self, X):
            return np.full(len(X), wrapped, dtype=object)

    adapter = SklearnAdapter(ComplexRegressor(), task="regression")
    with pytest.raises(ValueError, match="complex"):
        adapter.predict(np.zeros((2, 1)))


def test_sklearn_adapter_rejects_complex_predict_proba_before_real_cast():
    class ComplexClassifier(ClassifierMixin, BaseEstimator):
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            return np.tile(np.array([[0.5 + 1j, 0.5 - 1j]]), (len(X), 1))

    adapter = SklearnAdapter(ComplexClassifier())
    with pytest.raises(ValueError, match="complex"):
        adapter.predict(np.zeros((2, 1)))


@pytest.mark.parametrize(
    "class_names, error",
    [
        ("negative", TypeError),
        ([], ValueError),
        (["negative", 1], TypeError),
        (["negative", "  "], ValueError),
        (["same", "same"], ValueError),
    ],
)
def test_sklearn_adapter_validates_class_names_container_and_entries(class_names, error):
    class Classifier:
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            return np.tile(np.array([[0.5, 0.5]]), (len(X), 1))

    with pytest.raises(error, match="class_names"):
        SklearnAdapter(Classifier(), class_names=class_names)


def test_protodash_preserves_zero_objective_mass_and_suppresses_undefined_mmd():
    # For X={-1, +1} under a linear kernel, each candidate's mean similarity
    # to the empirical target is zero. The one-support QP oracle is therefore
    # max(mu / K_jj, 0) = 0, so no normalized prototype measure exists.
    X = np.array([[-1.0], [1.0]])
    result = ProtoDashExplainer(n_prototypes=1, kernel="linear").find_prototypes(
        X,
        return_mmd=True,
    )
    data = result.explanation_data

    assert data["objective_weights"] == [0.0]
    assert data["weights"] == [0.0]
    assert data["normalized_weights_defined"] is False
    assert data["mmd_defined"] is False
    assert "mmd_score" not in data
    assert data["mmd_undefined_reason"] == "objective_weights_have_zero_normalizable_mass"

    local = ProtoDashExplainer(n_prototypes=1, kernel="linear").explain(
        np.array([0.0]),
        np.array([[1.0]]),
    )
    assert local.explanation_data["objective_weights"] == [0.0]
    assert local.explanation_data["weights"] == [0.0]
    assert local.explanation_data["normalized_weights_defined"] is False


def test_protodash_normalizes_large_weights_without_overflow_and_preserves_zero_mass():
    explainer = ProtoDashExplainer(n_prototypes=2)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        normalized = explainer._normalized_display_weights(np.array([1e308, 1e308]))
        is_defined = explainer._normalized_weights_defined(np.array([1e308, 1e308]))

    np.testing.assert_array_equal(normalized, np.array([0.5, 0.5]))
    assert is_defined is True

    for zero_mass in (np.array([]), np.zeros(2), np.array([explainer.epsilon])):
        np.testing.assert_array_equal(
            explainer._normalized_display_weights(zero_mass),
            np.zeros_like(zero_mass),
        )
        assert explainer._normalized_weights_defined(zero_mass) is False


def test_protodash_objective_is_scale_safe_or_fails_before_serializing_nonfinite_data():
    explainer = ProtoDashExplainer(n_prototypes=1, kernel="linear")

    assert explainer._evaluate_objective(
        np.array([2.0]),
        np.array([[4.0]]),
        np.array([0.5]),
    ) == pytest.approx(0.5)

    # Both terms equal 2**1199 and overflow float64 separately, but their exact
    # difference is representable zero. The fallback must retain that result.
    cancelling_weight = np.ldexp(1.0, 800)
    with np.errstate(over="raise", invalid="raise"):
        cancelled = explainer._evaluate_objective(
            np.array([np.ldexp(1.0, 399)]),
            np.array([[np.ldexp(1.0, -400)]]),
            np.array([cancelling_weight]),
        )
    assert cancelled == 0.0

    with pytest.raises(ValueError, match="not representable as a nonzero float64"):
        explainer._evaluate_objective(
            np.array([np.nextafter(0.0, 1.0)]),
            np.array([[0.0]]),
            np.array([0.25]),
        )

    # This public path has a finite optimal weight and normalized display mass,
    # but its positive objective is genuinely larger than float64 can encode.
    with pytest.raises(ValueError, match="not representable as a finite float64"):
        explainer.explain(np.array([1e303]), np.array([[2e-5]]))


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"kernel": 1}, TypeError),
        ({"kernel_width": "1"}, TypeError),
        ({"epsilon": "1e-10"}, TypeError),
        ({"optimize_weights": "false"}, TypeError),
        ({"force_n_prototypes": 1}, TypeError),
        ({"random_state": -1}, ValueError),
        ({"random_state": 2**32}, ValueError),
    ],
)
def test_protodash_rejects_coercive_constructor_values(kwargs, error):
    with pytest.raises(error):
        ProtoDashExplainer(**kwargs)


def test_protodash_rejects_nonboolean_method_flags_and_scalar_batches():
    explainer = ProtoDashExplainer(n_prototypes=1, kernel="linear")
    with pytest.raises(TypeError, match="return_mmd"):
        explainer.find_prototypes(np.array([[1.0]]), return_mmd="false")
    with pytest.raises(TypeError, match="use_predictions"):
        explainer.explain(np.array([1.0]), np.array([[1.0]]), use_predictions="false")
    with pytest.raises(TypeError, match="include_criticisms"):
        explainer.get_prototype_summary(np.array([[1.0]]), include_criticisms="false")
    with pytest.raises(ValueError, match="non-empty 2D"):
        explainer.explain_batch(np.array(1.0), np.array([[1.0]]))


@pytest.mark.parametrize("explainer_type", [PartialDependenceExplainer, ALEExplainer])
def test_global_effect_explainers_pin_prediction_output_width(explainer_type):
    model = _ChangingWidthClassifier()
    X = np.array([[0.0], [1.0]])
    if explainer_type is PartialDependenceExplainer:
        explainer = explainer_type(model, X, ["x"], grid_resolution=2)
        operation = lambda: explainer.explain([0], target_class=1)
    else:
        explainer = explainer_type(model, X, ["x"], n_bins=1)
        operation = lambda: explainer.explain(0, target_class=1)

    with pytest.raises(ValueError, match="changed its number of output columns.*expected 3"):
        operation()


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_permutation_seed_range_fails_at_construction(seed):
    with pytest.raises(ValueError, match="random_state"):
        PermutationImportanceExplainer(
            _ProbabilityModel(),
            np.array([[0.0], [1.0]]),
            np.array([0, 0]),
            ["x"],
            random_state=seed,
        )


def test_sage_rejects_noncallable_custom_loss_at_construction():
    with pytest.raises(TypeError, match="loss_fn"):
        SAGEExplainer(
            _ProbabilityModel(),
            np.array([[0.0], [1.0]]),
            np.array([0, 0]),
            ["x"],
            n_permutations=1,
            loss_fn="mean_absolute_error",
            random_state=0,
        )


@pytest.mark.parametrize("num_features", [True, 1.0, np.float64(1.0)])
def test_lime_num_features_requires_integer_semantics(num_features):
    # Boundary validation occurs before the third-party LIME object is used,
    # so this focused test does not require optional backend execution.
    explainer = object.__new__(LimeExplainer)
    explainer.feature_names = ["x"]

    with pytest.raises(TypeError, match="num_features"):
        explainer.explain(np.array([0.0]), num_features=num_features)
