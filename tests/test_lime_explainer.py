# tests/test_lime_explainer.py

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from explainiverse.adapters.sklearn_adapter import SklearnAdapter
from explainiverse.explainers.attribution.lime_wrapper import LimeExplainer
from explainiverse.core.explanation import Explanation

import numpy as np


def test_lime_explainer_target_and_shape():
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names.tolist()

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    adapter = SklearnAdapter(model, class_names=class_names)
    explainer = LimeExplainer(
        model=adapter,
        training_data=X,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification"
    )

    explanation = explainer.explain(X[0], num_features=3)

    #  Explanation object structure
    assert isinstance(explanation, Explanation)
    assert "feature_attributions" in explanation.explanation_data
    attributions = explanation.explanation_data["feature_attributions"]
    assert isinstance(attributions, dict)
    assert 1 <= len(attributions) <= 3

    #  Attribution value range
    for value in attributions.values():
        assert isinstance(value, float)
        assert -1.0 <= value <= 1.0

    #  target_class matches model prediction
    preds = adapter.predict(np.array([X[0]]))  # shape (1, 3)
    predicted_index = np.argmax(preds[0])
    predicted_label = class_names[predicted_index]
    assert explanation.target_class == predicted_label, \
        f"Expected target class '{predicted_label}', got '{explanation.target_class}'"
    
    print(f"\nExplained class: {explanation.target_class}")
    print("Top feature attributions:")
    for feat, val in explanation.explanation_data["feature_attributions"].items():
        print(f"  {feat}: {val:+.4f}")


    print(" Robust LIME test passed.")


def test_lime_explainer_num_features_range():
    iris = load_iris()
    X, y = iris.data, iris.target
    model = LogisticRegression(max_iter=200).fit(X, y)
    adapter = SklearnAdapter(model, class_names=iris.target_names.tolist())
    explainer = LimeExplainer(adapter, X, iris.feature_names, iris.target_names.tolist())

    explanation = explainer.explain(X[1], num_features=2)
    attributions = explanation.explanation_data["feature_attributions"]
    assert len(attributions) <= 2, "Should not exceed requested num_features"


def test_lime_explainer_top_labels():
    iris = load_iris()
    X, y = iris.data, iris.target
    model = LogisticRegression(max_iter=200).fit(X, y)
    adapter = SklearnAdapter(model, class_names=iris.target_names.tolist())
    explainer = LimeExplainer(adapter, X, iris.feature_names, iris.target_names.tolist())

    explanation = explainer.explain(X[2], top_labels=2)
    assert explanation.target_class in iris.target_names, "Label must be human-readable class name"


def test_lime_explainer_plot_does_not_crash():
    iris = load_iris()
    X, y = iris.data, iris.target
    model = LogisticRegression(max_iter=200).fit(X, y)
    adapter = SklearnAdapter(model, class_names=iris.target_names.tolist())
    explainer = LimeExplainer(adapter, X, iris.feature_names, iris.target_names.tolist())

    explanation = explainer.explain(X[3])
    try:
        explanation.plot()  # Basic print for now
    except Exception as e:
        assert False, f"Plot method should not raise an error. Got: {e}"




# if __name__ == "__main__":
#     test_lime_explainer_target_and_shape()
#     test_lime_explainer_num_features_range()
#     test_lime_explainer_top_labels()
#     test_lime_explainer_plot_does_not_crash()
#     print(" All LIME tests passed.")
