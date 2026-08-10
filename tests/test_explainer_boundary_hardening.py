"""Regression tests for strict explainer API-boundary validation."""

import numpy as np
import pytest

from explainiverse.explainers.example_based.protodash import ProtoDashExplainer
from explainiverse.explainers.global_explainers.ale import ALEExplainer
from explainiverse.explainers.global_explainers.partial_dependence import PartialDependenceExplainer
from explainiverse.explainers.global_explainers.permutation_importance import (
    PermutationImportanceExplainer,
)
from explainiverse.explainers.global_explainers.sage import SAGEExplainer
from explainiverse.explainers.gradient.gradcam import GradCAMExplainer, _prepare_single_input
from explainiverse.explainers.rule_based.anchors_wrapper import AnchorsExplainer


class _RegressionModel:
    task = "regression"

    def predict(self, X):
        matrix = np.asarray(X)
        return np.sum(matrix, axis=1, keepdims=True)


class _ComplexRegressionModel(_RegressionModel):
    def predict(self, X):
        return super().predict(X).astype(complex) + 1j


class _BinaryModel:
    task = "classification"
    classes_ = np.array([0, 1])

    def predict(self, X):
        positive = (np.asarray(X)[:, 0] > 0).astype(float)
        return np.column_stack((1.0 - positive, positive))


class _ComplexBinaryModel(_BinaryModel):
    def predict(self, X):
        return super().predict(X).astype(complex) + 1j


class _CAMAdapter:
    task = "classification"

    def get_layer_gradients(self, *args, **kwargs):
        raise AssertionError("constructor validation must run before gradients")

    def predict(self, X):
        return np.ones((len(X), 2)) / 2.0


def _sage(model=None):
    X = np.array([[-1.0], [1.0]])
    return SAGEExplainer(
        model=model or _RegressionModel(),
        X=X,
        y=X[:, 0],
        feature_names=["x"],
        n_permutations=1,
        task="regression",
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ALEExplainer(_RegressionModel(), np.ones((2, 1)), ["   "]),
        lambda: PartialDependenceExplainer(
            _RegressionModel(), np.ones((2, 1)), ["\t"], grid_resolution=2
        ),
        lambda: PermutationImportanceExplainer(
            _RegressionModel(), np.ones((2, 1)), np.ones(2), ["\n"]
        ),
        lambda: SAGEExplainer(
            _RegressionModel(),
            np.ones((2, 1)),
            np.ones(2),
            ["  "],
            n_permutations=1,
            task="regression",
        ),
        lambda: AnchorsExplainer(
            _BinaryModel(),
            np.array([[-1.0], [1.0]]),
            [" "],
            ["negative", "positive"],
            n_samples=2,
        ),
        lambda: GradCAMExplainer(_CAMAdapter(), "conv", class_names=["negative", " "]),
    ],
)
def test_public_name_boundaries_reject_whitespace_only_names(factory):
    with pytest.raises(ValueError, match="non-empty strings"):
        factory()


def test_cam_rejects_complex_input_before_float_conversion():
    image = np.ones((1, 2, 2), dtype=complex) + 1j
    with pytest.raises(ValueError, match="complex"):
        _prepare_single_input(_CAMAdapter(), image, input_layout="chw")


def test_global_explainers_reject_complex_numeric_values_and_outputs():
    complex_X = np.array([[1.0 + 1j], [2.0 + 1j]])
    with pytest.raises(ValueError, match="complex"):
        ALEExplainer(_RegressionModel(), complex_X, ["x"], n_bins=2).explain(0)
    with pytest.raises(ValueError, match="complex"):
        PartialDependenceExplainer(_RegressionModel(), complex_X, ["x"], grid_resolution=2).explain(
            [0]
        )

    real_X = np.array([[0.0], [1.0]])
    with pytest.raises(ValueError, match="complex"):
        PartialDependenceExplainer(
            _ComplexRegressionModel(), real_X, ["x"], grid_resolution=2
        ).explain([0])
    with pytest.raises(ValueError, match="complex"):
        PermutationImportanceExplainer(
            _ComplexBinaryModel(), real_X, np.array([0, 1]), ["x"], n_repeats=1
        ).explain()
    with pytest.raises(ValueError, match="complex"):
        _sage(_ComplexRegressionModel()).explain()


def test_protodash_rejects_complex_data_and_whitespace_names():
    explainer = ProtoDashExplainer(n_prototypes=1)
    with pytest.raises(ValueError, match="complex"):
        explainer.find_prototypes(np.array([[1.0 + 1j]]))
    with pytest.raises(ValueError, match="non-empty strings"):
        explainer.find_prototypes(np.array([[1.0]]), feature_names=["  "])


def test_anchors_and_sage_reject_unknown_explain_options():
    anchors = AnchorsExplainer(
        _BinaryModel(),
        np.array([[-1.0], [1.0]]),
        ["x"],
        ["negative", "positive"],
        n_samples=2,
    )
    with pytest.raises(TypeError, match="Unexpected keyword.*unknown"):
        anchors.explain(np.array([1.0]), unknown=True)
    with pytest.raises(TypeError, match="Unexpected keyword.*unknown"):
        _sage().explain(unknown=True)
