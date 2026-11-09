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


class _RecordingFeatureExplainer(_CountingFeatureExplainer):
    def __init__(self, *, mutate=False, fail_on_call=None):
        super().__init__()
        self.mutate = mutate
        self.fail_on_call = fail_on_call
        self.seen = []

    def explain(self, row):
        self.seen.append(row.copy())
        if self.mutate:
            row[:] = 0.0
        if len(self.seen) == self.fail_on_call:
            raise RuntimeError("explanation generation failed after modifying input")
        return super().explain(row)


class _NamedRegistry:
    def __init__(self, explainers):
        self.explainers = explainers

    def create(self, name, **kwargs):
        return self.explainers[name]


def _roar_data():
    X_train = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[0.0, 0.5], [1.0, 0.5]])
    y_test = np.array([0, 1])
    return X_train, y_train, X_test, y_test


def test_suite_roar_isolates_data_from_explainers_before_retraining():
    X_train, y_train, X_test, y_test = _roar_data()
    original_train, original_test = X_train.copy(), X_test.copy()
    mutating = _RecordingFeatureExplainer(mutate=True)
    observing = _RecordingFeatureExplainer()
    estimator = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
    suite = ExplanationSuite(estimator, [("mutating", {}), ("observing", {})])
    suite._registry = _NamedRegistry({"mutating": mutating, "observing": observing})

    scores = suite.evaluate_roar(X_train, y_train, X_test, y_test, top_k=1, n_repeats=1)

    np.testing.assert_array_equal(X_train, original_train)
    np.testing.assert_array_equal(X_test, original_test)
    expected_rows = np.concatenate((original_train, original_test))
    np.testing.assert_array_equal(mutating.seen, expected_rows)
    np.testing.assert_array_equal(observing.seen, expected_rows)
    # Removing f0 drops accuracy from 1.0 to 0.5. If explanation generation
    # corrupts the clean datasets, both fits instead see zeros and report 0.0.
    assert scores == {"mutating": 0.5, "observing": 0.5}


@pytest.mark.parametrize("fail_on_call", [2, 5], ids=["training-row", "test-row"])
def test_suite_roar_preserves_inputs_when_mutating_explainer_fails(fail_on_call):
    X_train, y_train, X_test, y_test = _roar_data()
    original_train, original_test = X_train.copy(), X_test.copy()
    explainer = _RecordingFeatureExplainer(mutate=True, fail_on_call=fail_on_call)
    estimator = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
    suite = ExplanationSuite(estimator, [("mutating", {})])
    suite._registry = _FixedRegistry(explainer)

    with pytest.raises(RuntimeError, match="failed after modifying input"):
        suite.evaluate_roar(X_train, y_train, X_test, y_test, top_k=1, n_repeats=1)

    np.testing.assert_array_equal(X_train, original_train)
    np.testing.assert_array_equal(X_test, original_test)


if __name__ == "__main__":
    test_explanation_suite_lime_vs_shap()
