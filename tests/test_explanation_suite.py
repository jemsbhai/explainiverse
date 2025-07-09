from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from explainiverse.adapters.sklearn_adapter import SklearnAdapter
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
            ("lime", {
                "training_data": X,
                "feature_names": feature_names,
                "class_names": class_names,
                "mode": "classification"
            }),
            ("shap", {
                "background_data": X[:30],
                "feature_names": feature_names,
                "class_names": class_names
            })
        ],
        data_meta={"task": "classification"}
    )

    explanations = suite.run(X[0])
    assert "lime" in explanations
    assert "shap" in explanations

    print("\n[Test] LIME vs SHAP Comparison:")
    suite.compare()

    suggestion = suite.suggest_best()
    print(f"\nSuggested explainer based on model/task analysis: {suggestion}")
    


if __name__ == "__main__":
    test_explanation_suite_lime_vs_shap()