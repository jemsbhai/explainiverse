# tests/test_shap_explainer.py

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np

from explainiverse.adapters.sklearn_adapter import SklearnAdapter
from explainiverse.explainers.attribution.shap_wrapper import ShapExplainer
from explainiverse.core.explanation import Explanation


from sklearn.datasets import make_classification

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

def test_shap_linear_regression():
    X, y = make_regression(
        n_samples=100,
        n_features=4,
        noise=0.3,
        random_state=99
    )
    feature_names = [f"reg_f{i}" for i in range(X.shape[1])]

    model = LinearRegression()
    model.fit(X, y)

    # No class names for regression
    adapter = SklearnAdapter(model, class_names=None)

    explainer = ShapExplainer(
        model=adapter,
        background_data=X[:30],
        feature_names=feature_names,
        class_names=["target"]  # Dummy value to avoid NoneType errors
    )

    explanation = explainer.explain(X[0])

    print(f"\n[LinearRegression] pseudo-class: {explanation.target_class}")
    for k, v in explanation.explanation_data["feature_attributions"].items():
        print(f"  {k}: {v:+.4f}")

    assert isinstance(explanation.explanation_data["feature_attributions"], dict)
    assert len(explanation.explanation_data["feature_attributions"]) == len(feature_names)

def test_shap_logistic_regression_multiclass():
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_iris

    data = load_iris()
    X, y = data.data, data.target
    class_names = data.target_names.tolist()
    feature_names = data.feature_names

    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X, y)
    adapter = SklearnAdapter(model, class_names=class_names)

    explainer = ShapExplainer(
        model=adapter,
        background_data=X[:25],
        feature_names=feature_names,
        class_names=class_names
    )

    explanation = explainer.explain(X[10], top_labels=2)

    print(f"\n[LogReg] class: {explanation.target_class}")
    for k, v in explanation.explanation_data["feature_attributions"].items():
        print(f"  {k}: {v:+.4f}")

    assert explanation.target_class in class_names
    assert len(explanation.explanation_data["feature_attributions"]) == len(feature_names)



def test_shap_multiclass_classifier_explainer():
    X, y = make_classification(
        n_samples=120,
        n_features=6,
        n_classes=5,
        n_informative=4,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=123
    )
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    class_names = [f"class_{i}" for i in range(5)]

    model = RandomForestClassifier()
    model.fit(X, y)
    adapter = SklearnAdapter(model, class_names=class_names)

    explainer = ShapExplainer(
        model=adapter,
        background_data=X[:20],
        feature_names=feature_names,
        class_names=class_names
    )

    explanation = explainer.explain(X[5])
    print(f"\n[Multi-class] class: {explanation.target_class}")
    for k, v in explanation.explanation_data["feature_attributions"].items():
        print(f"  {k}: {v:+.4f}")

    assert explanation.target_class in class_names
    assert isinstance(explanation.explanation_data["feature_attributions"], dict)
    assert len(explanation.explanation_data["feature_attributions"]) == len(feature_names)



def test_shap_binary_classifier_explainer():
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_classes=2,
        n_informative=3,
        random_state=42
    )
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    class_names = ["class_0", "class_1"]

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    adapter = SklearnAdapter(model, class_names=class_names)

    explainer = ShapExplainer(
        model=adapter,
        background_data=X[:20],
        feature_names=feature_names,
        class_names=class_names
    )

    explanation = explainer.explain(X[0])
    print(f"\n[Binary] class: {explanation.target_class}")
    for k, v in explanation.explanation_data["feature_attributions"].items():
        print(f"  {k}: {v:+.4f}")

    assert explanation.target_class in class_names
    assert isinstance(explanation.explanation_data["feature_attributions"], dict)
    assert len(explanation.explanation_data["feature_attributions"]) == len(feature_names)


def test_shap_explainer_single_instance():
    iris = load_iris()
    X, y = iris.data, iris.target
    class_names = iris.target_names.tolist()
    feature_names = iris.feature_names

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    adapter = SklearnAdapter(model, class_names=class_names)

    explainer = ShapExplainer(
        model=adapter,
        background_data=X[:50],  # smaller background set for speed
        feature_names=feature_names,
        class_names=class_names
    )

    explanation = explainer.explain(X[0])

    assert isinstance(explanation, Explanation)
    assert explanation.target_class in class_names
    attributions = explanation.explanation_data["feature_attributions"]
    assert isinstance(attributions, dict)
    assert len(attributions) == len(feature_names)

    print("Basic SHAP test passed.")
    print(f"Explained class: {explanation.target_class}")
    for k, v in attributions.items():
        print(f"  {k}: {v:+.4f}")
        

def test_shap_global_feature_importance():
    iris = load_iris()
    X, y = iris.data, iris.target
    class_names = iris.target_names.tolist()
    feature_names = iris.feature_names

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    adapter = SklearnAdapter(model, class_names=class_names)

    explainer = ShapExplainer(
        model=adapter,
        background_data=X[:30],
        feature_names=feature_names,
        class_names=class_names
    )

    # Global SHAP: aggregate over first 10 instances
    shap_values = explainer.explainer.shap_values(X[:10])
    # Handle list (multi-class) or array (single-class or collapsed)
    if isinstance(shap_values, list):
        shap_matrix = np.abs(shap_values[0])
    else:
        shap_matrix = np.abs(shap_values)

    global_attributions = np.mean(shap_matrix, axis=0)

    # Aggregate across all 3 classes
    # global_attributions = np.mean(np.abs(shap_values[0]), axis=0)
    print("\n[Global SHAP importance]")
    
    print("\n[Global SHAP importance by class]")
    for i, fname in enumerate(feature_names):
        val = global_attributions[i]
        print(f"  {fname}:")
        if isinstance(val, np.ndarray):
            for j, v in enumerate(val):
                print(f"    class_{j}: {v:.4f}")
        else:
            print(f"    value: {val:.4f}")
            
    # for fname, value in zip(feature_names, global_attributions):
        
    #     print(f"  {fname}: value={value}, type={type(value)}, shape={getattr(value, 'shape', 'N/A')}")

    assert len(global_attributions) == len(feature_names)


def test_shap_cohort_explanation():
    iris = load_iris()
    X, y = iris.data, iris.target
    class_names = iris.target_names.tolist()
    feature_names = iris.feature_names

    cohort_mask = (y == 0)  # Filter only setosa
    X_cohort = X[cohort_mask]

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    adapter = SklearnAdapter(model, class_names=class_names)

    explainer = ShapExplainer(
        model=adapter,
        background_data=X[:30],
        feature_names=feature_names,
        class_names=class_names
    )

    shap_values = explainer.explainer.shap_values(X_cohort[:5])

    # Handle list (multi-class) or array (single-class or collapsed)
    if isinstance(shap_values, list):
        shap_matrix = np.abs(shap_values[0])
    else:
        shap_matrix = np.abs(shap_values)

    cohort_importance = np.mean(shap_matrix, axis=0)

    # print("\n[Cohort SHAP: class=0 (setosa)]")
    
    print("\n[Cohort SHAP: class=0 (setosa), per-class feature attributions]")
    for i, fname in enumerate(feature_names):
        val = cohort_importance[i]
        print(f"  {fname}:")
        if isinstance(val, np.ndarray):
            for j, v in enumerate(val):
                print(f"    class_{j}: {v:.4f}")
        else:
            print(f"    value: {val:.4f}")
    
    # for fname, value in zip(feature_names, cohort_importance):
    #     print(f"  {fname}: {float(value):.4f}")

    assert len(cohort_importance) == len(feature_names)


def test_shap_feature_range_cohort():
    
    X, y = make_classification(
        n_samples=150,
        n_features=6,
        n_classes=3,
        n_informative=3,
        n_redundant=1,
        n_repeated=0,
        n_clusters_per_class=1,
        random_state=42
    )
    feature_names = [f"feat_{i}" for i in range(6)]
    class_names = [f"class_{i}" for i in range(3)]

    # Slice: pick samples where feature 0 > 0.5
    feature_index = 0
    X_cohort = X[X[:, feature_index] > 0.5]

    model = LogisticRegression(max_iter=300)
    model.fit(X, y)
    adapter = SklearnAdapter(model, class_names=class_names)

    explainer = ShapExplainer(
        model=adapter,
        background_data=X[:40],
        feature_names=feature_names,
        class_names=class_names
    )

    shap_values = explainer.explainer.shap_values(X_cohort[:10])

    # Normalize for printing
    shap_matrix = shap_values[0] if isinstance(shap_values, list) else shap_values
    
    print("\n[Cohort SHAP: feat_0 > 0.5 — full per-class attributions]")
    n_features = shap_matrix.shape[1]
    n_classes = shap_matrix.shape[2]

    mean_attributions = np.mean(shap_matrix, axis=0)  # shape: (features, classes)

    for i, fname in enumerate(feature_names):
        print(f"  {fname}:")
        for c in range(n_classes):
            print(f"    class_{c}: {mean_attributions[i][c]:+.4f}")
    
    
    if shap_matrix.ndim == 3:
        shap_matrix = shap_matrix.mean(axis=-1)
    cohort_attributions = np.mean(shap_matrix, axis=0)
    
    print("Model predictions for sliced cohort:")
    print(adapter.predict(X_cohort[:10]))

    print("\nRaw SHAP values shape/type:")
    print(type(shap_values))
    print(np.array(shap_values).shape)

    print("\n[Cohort SHAP: feat_0 > 0.5]")
    for i, fname in enumerate(feature_names):
        val = cohort_attributions[i]
        print(f"  {fname}:")
        if isinstance(val, np.ndarray):
            for j, v in enumerate(val):
                print(f"    class_{j}: {v:.4f}")
        else:
            print(f"    value: {val:.4f}")

    assert len(cohort_attributions) == len(feature_names)



if __name__ == "__main__":
    test_shap_explainer_single_instance()
    test_shap_binary_classifier_explainer()
    test_shap_multiclass_classifier_explainer()
    test_shap_logistic_regression_multiclass()
    test_shap_linear_regression()
    test_shap_global_feature_importance()
    test_shap_cohort_explanation()
    test_shap_feature_range_cohort()