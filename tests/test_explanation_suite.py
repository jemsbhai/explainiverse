import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from explainiverse.adapters.sklearn_adapter import SklearnAdapter
from explainiverse.core.explanation import Explanation
from explainiverse.engine.suite import ExplanationSuite


def test_explanation_suite_lime_vs_shap():
    iris = load_iris()
    X, y = iris.data, iris.target
    class_names = iris.target_names.tolist()
    feature_names = iris.feature_names

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    adapter = SklearnAdapter(model, class_names=class_names)

    suite = ExplanationSuite(
        model=adapter,
        explainer_configs=[
            (
                "lime",
                {
                    "training_data": X,
                    "feature_names": feature_names,
                    "class_names": class_names,
                    "mode": "classification",
                },
            ),
            (
                "shap",
                {
                    "background_data": X[:30],
                    "feature_names": feature_names,
                    "class_names": class_names,
                },
            ),
        ],
        data_meta={"task": "classification"},
    )

    explanations = suite.run(X[0])
    assert "lime" in explanations
    assert "shap" in explanations

    print("\n[Test] LIME vs SHAP Comparison:")
    with pytest.raises(ValueError, match="mathematical comparability"):
        suite.compare()
    with pytest.warns(RuntimeWarning, match="descriptively only"):
        suite.compare(allow_incommensurate=True)

    with pytest.warns(FutureWarning, match="cannot establish explainer quality"):
        suggestion = suite.suggest_best()
    print(f"\nSuggested explainer based on model/task analysis: {suggestion}")


class _CountingFeatureExplainer:
    def __init__(self):
        self.calls = 0

    def explain(self, row):
        self.calls += 1
        return Explanation(
            explainer_name="fixed",
            target_class="1",
            explanation_data={"feature_attributions": {"f0": 2.0, "f1": 1.0}},
            feature_names=["f0", "f1"],
        )


class _FixedRegistry:
    def __init__(self, explainer):
        self.explainer = explainer

    def create(self, name, **kwargs):
        return self.explainer


def test_suite_roar_generates_aligned_train_and_test_explanations():
    X_train = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [2.0, 0.0], [2.0, 1.0]])
    y_train = np.array([0, 0, 1, 1, 1, 1])
    X_test = np.array([[0.1, 0.25], [1.1, 0.75]])
    y_test = np.array([0, 1])
    estimator = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
    explainer = _CountingFeatureExplainer()
    suite = ExplanationSuite(estimator, [("fixed", {})], data_meta={"task": "classification"})
    suite._registry = _FixedRegistry(explainer)

    result = suite.evaluate_roar(
        X_train,
        y_train,
        X_test,
        y_test,
        top_k=1,
        n_repeats=1,
        random_state=7,
    )

    assert set(result) == {"fixed"}
    assert np.isfinite(result["fixed"])
    assert explainer.calls == len(X_train) + len(X_test)


class _FailingExplainer:
    def explain(self, row):
        raise RuntimeError("explanation generation failed")


def test_suite_roar_surfaces_explanation_failures_instead_of_returning_zero():
    X_train = np.array([[0.0, 0.0], [1.0, 1.0]])
    y_train = np.array([0, 1])
    X_test = np.array([[0.25, 0.5]])
    y_test = np.array([0])
    estimator = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
    suite = ExplanationSuite(estimator, [("failing", {})])
    suite._registry = _FixedRegistry(_FailingExplainer())

    with pytest.raises(RuntimeError, match="explanation generation failed"):
        suite.evaluate_roar(X_train, y_train, X_test, y_test, n_repeats=1)


if __name__ == "__main__":
    test_explanation_suite_lime_vs_shap()
