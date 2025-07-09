from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np

from explainiverse.adapters.sklearn_adapter import SklearnAdapter
from explainiverse.explainers.attribution.lime_wrapper import LimeExplainer
from explainiverse.explainers.attribution.shap_wrapper import ShapExplainer
from explainiverse.evaluation.metrics import compute_aopc

from explainiverse.evaluation.metrics import compute_batch_aopc

def test_batch_aopc_lime_vs_shap():
    iris = load_iris()
    X, y = iris.data, iris.target
    class_names = iris.target_names.tolist()
    feature_names = iris.feature_names

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    adapter = SklearnAdapter(model, class_names=class_names)

    lime = LimeExplainer(
        model=adapter,
        training_data=X,
        feature_names=feature_names,
        class_names=class_names
    )
    shap = ShapExplainer(
        model=adapter,
        background_data=X[:30],
        feature_names=feature_names,
        class_names=class_names
    )

    lime_explanations = [lime.explain(X[i]) for i in range(10)]
    shap_explanations = [shap.explain(X[i]) for i in range(10)]

    for exp in lime_explanations + shap_explanations:
        exp.feature_names = feature_names

    scores = compute_batch_aopc(
        model=adapter,
        X=X[:10],
        explanations={
            "lime": lime_explanations,
            "shap": shap_explanations
        },
        num_steps=4
    )

    print("\nBatch AOPC Scores:")
    for method, score in scores.items():
        print(f"  {method}: {score:.4f}")



def test_aopc_lime_vs_shap():
    iris = load_iris()
    X, y = iris.data, iris.target
    class_names = iris.target_names.tolist()
    feature_names = iris.feature_names

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    adapter = SklearnAdapter(model, class_names=class_names)

    instance = X[0]

    lime = LimeExplainer(
        model=adapter,
        training_data=X,
        feature_names=feature_names,
        class_names=class_names
    )

    shap = ShapExplainer(
        model=adapter,
        background_data=X[:30],
        feature_names=feature_names,
        class_names=class_names
    )

    lime_exp = lime.explain(instance)
    shap_exp = shap.explain(instance)

    # Inject feature names (for robust index matching)
    lime_exp.feature_names = feature_names
    shap_exp.feature_names = feature_names

    lime_score = compute_aopc(adapter, instance, lime_exp, num_steps=4)
    shap_score = compute_aopc(adapter, instance, shap_exp, num_steps=4)

    print(f"\nAOPC (LIME): {lime_score:.4f}")
    print(f"AOPC (SHAP): {shap_score:.4f}")
    print(" AOPC test passed.")


if __name__ == "__main__":
    test_aopc_lime_vs_shap()
    test_batch_aopc_lime_vs_shap()
    
