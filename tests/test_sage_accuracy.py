"""Analytical accuracy tests for the marginal SAGE implementation."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from explainiverse.adapters.sklearn_adapter import SklearnAdapter
from explainiverse.explainers.global_explainers.sage import SAGEExplainer


class _AdditiveRegressionModel:
    """f(x) = x0 + 2*x1 with the standard adapter output shape."""

    def predict(self, X):
        X = np.asarray(X)
        return (X[:, 0] + 2.0 * X[:, 1]).reshape(-1, 1)


class _BinaryClassificationModel:
    """A deterministic two-class model whose label depends only on x0."""

    def predict(self, X):
        X = np.asarray(X)
        positive = (X[:, 0] > 0).astype(float)
        return np.column_stack([1.0 - positive, positive])


@pytest.fixture
def factorial_regression_data():
    # Full factorial support makes the marginal restricted predictors exact:
    # f_empty=0, f_{0}=x0, f_{1}=2*x1, f_{0,1}=f.
    X = np.array(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ]
    )
    y = X[:, 0] + 2.0 * X[:, 1]
    return X, y


def test_sage_matches_analytical_values_for_additive_regression(
    factorial_regression_data,
):
    """Marginal SAGE must recover the exact MSE game values [1, 4]."""
    X, y = factorial_regression_data
    explainer = SAGEExplainer(
        model=_AdditiveRegressionModel(),
        X=X,
        y=y,
        feature_names=["x0", "x1"],
        n_permutations=8,
        task="regression",
        random_state=7,
    )

    explanation = explainer.explain()
    values = explanation.explanation_data["feature_attributions"]

    assert values["x0"] == pytest.approx(1.0, abs=1e-12)
    assert values["x1"] == pytest.approx(4.0, abs=1e-12)


def test_sage_efficiency_equals_null_loss_minus_full_loss(
    factorial_regression_data,
):
    """The sampled Shapley values must telescope to the total game value."""
    X, y = factorial_regression_data
    explanation = SAGEExplainer(
        model=_AdditiveRegressionModel(),
        X=X,
        y=y,
        feature_names=["x0", "x1"],
        n_permutations=5,
        task="regression",
        random_state=11,
    ).explain()

    data = explanation.explanation_data
    total_attribution = sum(data["feature_attributions"].values())

    assert data["baseline_loss"] == pytest.approx(5.0, abs=1e-12)
    assert data["full_loss"] == pytest.approx(0.0, abs=1e-12)
    assert total_attribution == pytest.approx(data["baseline_loss"] - data["full_loss"], abs=1e-12)
    assert data["efficiency_error"] == pytest.approx(0.0, abs=1e-12)
    assert data["imputer"] == "marginal"


def test_sage_zero_one_game_matches_analytical_classification_values(
    factorial_regression_data,
):
    """The documented default classification game is zero-one loss."""
    X, _ = factorial_regression_data
    y = (X[:, 0] > 0).astype(int)

    explanation = SAGEExplainer(
        model=_BinaryClassificationModel(),
        X=X,
        y=y,
        feature_names=["signal", "noise"],
        n_permutations=8,
        task="classification",
        random_state=3,
    ).explain()
    data = explanation.explanation_data
    values = data["feature_attributions"]

    assert values["signal"] == pytest.approx(0.5, abs=1e-12)
    assert values["noise"] == pytest.approx(0.0, abs=1e-12)
    assert data["baseline_loss"] == pytest.approx(0.5, abs=1e-12)
    assert data["full_loss"] == pytest.approx(0.0, abs=1e-12)
    assert data["efficiency_error"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "negative_label, positive_label",
    [(10, 20), ("negative-label", "positive-label")],
)
def test_sage_maps_probability_columns_through_model_classes(negative_label, positive_label):
    X = np.array(
        [
            [-2.0, -1.0],
            [-2.0, 1.0],
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [2.0, -1.0],
            [2.0, 1.0],
        ]
    )
    y = np.where(X[:, 0] > 0, positive_label, negative_label)
    raw_model = LogisticRegression(C=1e6, random_state=0).fit(X, y)
    model = SklearnAdapter(raw_model)

    data = (
        SAGEExplainer(
            model=model,
            X=X,
            y=y,
            feature_names=["signal", "noise"],
            n_permutations=4,
            task="classification",
            random_state=3,
        )
        .explain()
        .explanation_data
    )

    assert data["full_loss"] == pytest.approx(0.0)
    assert data["baseline_loss"] == pytest.approx(0.5)
    assert data["feature_attributions"]["signal"] == pytest.approx(0.5)
    assert data["feature_attributions"]["noise"] == pytest.approx(0.0)


def test_sage_rejects_probability_width_that_disagrees_with_model_classes():
    class RawModel:
        classes_ = np.array(["a", "b", "c"])

    class BadAdapter:
        task = "classification"
        model = RawModel()

        def predict(self, X):
            return np.tile(np.array([[0.25, 0.75]]), (len(X), 1))

    with pytest.raises(ValueError, match="3 labels"):
        SAGEExplainer(
            model=BadAdapter(),
            X=np.array([[0.0], [1.0]]),
            y=np.array(["a", "b"]),
            feature_names=["x"],
            n_permutations=2,
            task="classification",
        )


@pytest.mark.parametrize("n_permutations", [0, -1])
def test_sage_rejects_nonpositive_permutation_counts(factorial_regression_data, n_permutations):
    X, y = factorial_regression_data

    with pytest.raises(ValueError, match="n_permutations"):
        SAGEExplainer(
            model=_AdditiveRegressionModel(),
            X=X,
            y=y,
            feature_names=["x0", "x1"],
            n_permutations=n_permutations,
            task="regression",
        )


def test_sage_rejects_mismatched_feature_names(factorial_regression_data):
    X, y = factorial_regression_data

    with pytest.raises(ValueError, match="feature_names"):
        SAGEExplainer(
            model=_AdditiveRegressionModel(),
            X=X,
            y=y,
            feature_names=["only_one"],
            task="regression",
        )
