from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np

from explainiverse.adapters.sklearn_adapter import SklearnAdapter
from explainiverse.explainers.attribution.lime_wrapper import LimeExplainer
from explainiverse.explainers.attribution.shap_wrapper import ShapExplainer
from explainiverse.evaluation.metrics import compute_aopc

from explainiverse.evaluation.metrics import compute_batch_aopc
from explainiverse.evaluation.metrics import compute_roar
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier


from explainiverse.evaluation.metrics import compute_roar_curve

def test_roar_curve_shap():
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=100,
        n_features=6,
        n_classes=3,
        n_informative=4,
        random_state=42
    )
    class_names = [f"class_{i}" for i in range(3)]
    feature_names = [f"feat_{i}" for i in range(X.shape[1])]

    model_class = LogisticRegression
    model_args = {"max_iter": 200}
    model = model_class(**model_args)
    model.fit(X, y)
    adapter = SklearnAdapter(model, class_names=class_names)

    shap_exp = ShapExplainer(
        model=adapter,
        background_data=X[:30],
        feature_names=feature_names,
        class_names=class_names
    )

    shap_exps = []
    for i in range(10):
        exp = shap_exp.explain(X[i])
        exp.feature_names = feature_names
        shap_exps.append(exp)

    curve = compute_roar_curve(
        model_class=model_class,
        X_train=X,
        y_train=y,
        X_test=X,
        y_test=y,
        explanations=shap_exps,
        max_k=5,
        baseline_value="mean",
        model_kwargs=model_args
    )

    print("\nROAR Curve (SHAP):")
    for k, drop in curve.items():
        print(f"  top-{k} drop: {drop:.4f}")



def test_roar_per_model(model_class, model_name):
    iris = load_iris()
    X, y = iris.data, iris.target
    class_names = iris.target_names.tolist()
    feature_names = iris.feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\n[ROAR Test: {model_name}]")
    try:
        model = model_class()
        model.fit(X_train, y_train)
        adapter = SklearnAdapter(model, class_names=class_names)

        lime = LimeExplainer(
            model=adapter,
            training_data=X_train,
            feature_names=feature_names,
            class_names=class_names
        )
        shap = ShapExplainer(
            model=adapter,
            background_data=X_train[:30],
            feature_names=feature_names,
            class_names=class_names
        )

        lime_exps = [lime.explain(X_train[i]) for i in range(10)]
        shap_exps = [shap.explain(X_train[i]) for i in range(10)]
        for e in lime_exps + shap_exps:
            e.feature_names = feature_names

        roar_lime = compute_roar(model_class, X_train, y_train, X_test, y_test, lime_exps, top_k=2)
        roar_shap = compute_roar(model_class, X_train, y_train, X_test, y_test, shap_exps, top_k=2)

        print(f"  ROAR Drop - LIME: {roar_lime:.4f}")
        print(f"  ROAR Drop - SHAP: {roar_shap:.4f}")

    except Exception as e:
        print(f"  [ERROR] Failed on {model_name}: {e}")


def test_roar_all_supported_models():
    model_classes = {
        "logreg": LogisticRegression,
        "rf": RandomForestClassifier,
        "gb": GradientBoostingClassifier,
        "svc": SVC,
        "knn": KNeighborsClassifier,
        "nb": GaussianNB,
        "xgb": XGBClassifier,
    }

    for name, cls in model_classes.items():
        test_roar_per_model(cls, name)



def test_roar_multiple_models():
    iris = load_iris()
    X, y = iris.data, iris.target
    class_names = iris.target_names.tolist()
    feature_names = iris.feature_names

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model_variants = {
        "logreg": LogisticRegression,
        "rf": RandomForestClassifier
    }

    for model_name, model_class in model_variants.items():
        print(f"\n[ROAR Test: {model_name}]")

        model = model_class()
        model.fit(X_train, y_train)
        adapter = SklearnAdapter(model, class_names=class_names)

        lime = LimeExplainer(
            model=adapter,
            training_data=X_train,
            feature_names=feature_names,
            class_names=class_names
        )
        shap = ShapExplainer(
            model=adapter,
            background_data=X_train[:30],
            feature_names=feature_names,
            class_names=class_names
        )

        lime_exps = [lime.explain(X_train[i]) for i in range(10)]
        shap_exps = [shap.explain(X_train[i]) for i in range(10)]
        for e in lime_exps + shap_exps:
            e.feature_names = feature_names

        roar_lime = compute_roar(model_class, X_train, y_train, X_test, y_test, lime_exps, top_k=2)
        roar_shap = compute_roar(model_class, X_train, y_train, X_test, y_test, shap_exps, top_k=2)

        print(f"  ROAR Drop - LIME: {roar_lime:.4f}")
        print(f"  ROAR Drop - SHAP: {roar_shap:.4f}")

def test_roar_lime_vs_shap():
    iris = load_iris()
    X, y = iris.data, iris.target
    class_names = iris.target_names.tolist()
    feature_names = iris.feature_names
    
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y
)

    # X_train, X_test = X[:100], X[100:]
    # y_train, y_test = y[:100], y[100:]

    model_class = LogisticRegression
    model_args = {"max_iter": 200}

    base_model = model_class(**model_args)
    base_model.fit(X_train, y_train)
    adapter = SklearnAdapter(base_model, class_names=class_names)

    lime = LimeExplainer(
        model=adapter,
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names
    )
    shap = ShapExplainer(
        model=adapter,
        background_data=X_train[:30],
        feature_names=feature_names,
        class_names=class_names
    )

    lime_exps = [lime.explain(X_train[i]) for i in range(20)]
    shap_exps = [shap.explain(X_train[i]) for i in range(20)]
    for e in lime_exps + shap_exps:
        e.feature_names = feature_names

    print("Baseline accuracy before ROAR:", accuracy_score(y_test, base_model.predict(X_test)))
    
    roar_lime = compute_roar(
        model_class=model_class,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        explanations=lime_exps,
        top_k=2,
        model_kwargs=model_args
    )

    roar_shap = compute_roar(
        model_class=model_class,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        explanations=shap_exps,
        top_k=2,
        model_kwargs=model_args
    )

    print("\nROAR Accuracy Drop:")
    print(f"  LIME: {roar_lime:.4f}")
    print(f"  SHAP: {roar_shap:.4f}")



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


def test_roar_baseline_variants():
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=100,
        n_features=6,
        n_classes=3,
        n_informative=4,
        random_state=42
    )

    model_class = LogisticRegression
    model_args = {"max_iter": 200}

    lime_exps = []
    class_names = [f"class_{i}" for i in range(3)]
    feature_names = [f"feat_{i}" for i in range(X.shape[1])]

    model = model_class(**model_args)
    model.fit(X, y)
    adapter = SklearnAdapter(model, class_names=class_names)

    lime = LimeExplainer(
        model=adapter,
        training_data=X,
        feature_names=feature_names,
        class_names=class_names
    )

    for i in range(10):
        exp = lime.explain(X[i])
        exp.feature_names = feature_names
        lime_exps.append(exp)

    print("\n[ROAR Baseline Variants]")

    # String baseline
    print("  mean baseline:", compute_roar(model_class, X, y, X, y, lime_exps, baseline_value="mean", top_k=2))
    print("  median baseline:", compute_roar(model_class, X, y, X, y, lime_exps, baseline_value="median", top_k=2))

    # Callable
    print("  callable baseline (mean):", compute_roar(
        model_class, X, y, X, y, lime_exps,
        baseline_value=lambda X: np.mean(X, axis=0),
        top_k=2
    ))

    # Array
    print("  array baseline:", compute_roar(
        model_class, X, y, X, y, lime_exps,
        baseline_value=np.median(X, axis=0),
        top_k=2
    ))


def test_roar_baseline_variants_shap():
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=100,
        n_features=6,
        n_classes=3,
        n_informative=4,
        random_state=42
    )

    model_class = LogisticRegression
    model_args = {"max_iter": 200}
    model = model_class(**model_args)
    model.fit(X, y)
    adapter = SklearnAdapter(model, class_names=[f"class_{i}" for i in range(3)])

    shap_exp = ShapExplainer(
        model=adapter,
        background_data=X[:30],
        feature_names=[f"feat_{i}" for i in range(X.shape[1])],
        class_names=[f"class_{i}" for i in range(3)]
    )

    shap_exps = []
    for i in range(10):
        exp = shap_exp.explain(X[i])
        exp.feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        shap_exps.append(exp)

    print("\n[ROAR Baseline Variants - SHAP]")
    print("  mean baseline:", compute_roar(model_class, X, y, X, y, shap_exps, baseline_value="mean", top_k=2))
    print("  median baseline:", compute_roar(model_class, X, y, X, y, shap_exps, baseline_value="median", top_k=2))
    print("  callable baseline (mean):", compute_roar(model_class, X, y, X, y, shap_exps, baseline_value=lambda X: np.mean(X, axis=0), top_k=2))
    print("  array baseline:", compute_roar(model_class, X, y, X, y, shap_exps, baseline_value=np.median(X, axis=0), top_k=2))




if __name__ == "__main__":
    test_aopc_lime_vs_shap()
    test_batch_aopc_lime_vs_shap()
    test_roar_lime_vs_shap()
    test_roar_multiple_models()
    test_roar_all_supported_models()
    test_roar_baseline_variants()
    test_roar_baseline_variants_shap()
    test_roar_curve_shap()

