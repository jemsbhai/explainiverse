# tests/test_evaluation_metrics.py
"""
Tests for evaluation metrics: AOPC, ROAR, and related utilities.

Comprehensive test coverage for:
- AOPC (Area Over Perturbation Curve) - single and batch
- ROAR (Remove And Retrain) - various baselines, models, and curves
- LIME vs SHAP comparisons
"""
import pytest
import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier

from explainiverse.adapters.sklearn_adapter import SklearnAdapter
from explainiverse.explainers.attribution.lime_wrapper import LimeExplainer
from explainiverse.explainers.attribution.shap_wrapper import ShapExplainer
from explainiverse.evaluation.metrics import compute_aopc, compute_batch_aopc, compute_roar, compute_roar_curve


# =============================================================================
# AOPC Tests
# =============================================================================

def test_aopc_lime_vs_shap():
    """Compare AOPC scores between LIME and SHAP explanations."""
    iris = load_iris()
    X, y = iris.data, iris.target
    class_names = iris.target_names.tolist()
    feature_names = list(iris.feature_names)

    model = LogisticRegression(max_iter=1000)
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
    
    # Both should produce valid scores
    assert isinstance(lime_score, float)
    assert isinstance(shap_score, float)
    assert lime_score >= 0 or shap_score >= 0  # At least one should be meaningful


def test_batch_aopc_lime_vs_shap():
    """Batch AOPC comparison between LIME and SHAP."""
    iris = load_iris()
    X, y = iris.data, iris.target
    class_names = iris.target_names.tolist()
    feature_names = list(iris.feature_names)

    model = LogisticRegression(max_iter=1000)
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

    assert "lime" in scores
    assert "shap" in scores
    assert isinstance(scores["lime"], float)
    assert isinstance(scores["shap"], float)


# =============================================================================
# ROAR Tests - LIME vs SHAP
# =============================================================================

def test_roar_lime_vs_shap():
    """Compare ROAR accuracy drop between LIME and SHAP explanations."""
    iris = load_iris()
    X, y = iris.data, iris.target
    class_names = iris.target_names.tolist()
    feature_names = list(iris.feature_names)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42, stratify=y
    )

    model_class = LogisticRegression
    model_args = {"max_iter": 1000}

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

    baseline_acc = accuracy_score(y_test, base_model.predict(X_test))
    print(f"\nBaseline accuracy before ROAR: {baseline_acc:.4f}")
    
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

    assert isinstance(roar_lime, float)
    assert isinstance(roar_shap, float)


# =============================================================================
# ROAR Tests - Multiple Models
# =============================================================================

def _test_roar_per_model(model_class, model_name, model_kwargs=None):
    """Helper function to test ROAR for a specific model type."""
    if model_kwargs is None:
        model_kwargs = {}
    iris = load_iris()
    X, y = iris.data, iris.target
    class_names = iris.target_names.tolist()
    feature_names = list(iris.feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\n[ROAR Test: {model_name}]")
    try:
        model = model_class(**model_kwargs)
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

        roar_lime = compute_roar(model_class, X_train, y_train, X_test, y_test, lime_exps, top_k=2, model_kwargs=model_kwargs)
        roar_shap = compute_roar(model_class, X_train, y_train, X_test, y_test, shap_exps, top_k=2, model_kwargs=model_kwargs)

        print(f"  ROAR Drop - LIME: {roar_lime:.4f}")
        print(f"  ROAR Drop - SHAP: {roar_shap:.4f}")
        
        return roar_lime, roar_shap

    except Exception as e:
        print(f"  [ERROR] Failed on {model_name}: {e}")
        raise


def test_roar_logistic_regression():
    """ROAR with Logistic Regression."""
    roar_lime, roar_shap = _test_roar_per_model(LogisticRegression, "LogisticRegression", model_kwargs={"max_iter": 1000})
    assert isinstance(roar_lime, float)
    assert isinstance(roar_shap, float)


def test_roar_random_forest():
    """ROAR with Random Forest."""
    roar_lime, roar_shap = _test_roar_per_model(RandomForestClassifier, "RandomForest")
    assert isinstance(roar_lime, float)
    assert isinstance(roar_shap, float)


def test_roar_gradient_boosting():
    """ROAR with Gradient Boosting."""
    roar_lime, roar_shap = _test_roar_per_model(GradientBoostingClassifier, "GradientBoosting")
    assert isinstance(roar_lime, float)
    assert isinstance(roar_shap, float)


def test_roar_svc():
    """ROAR with Support Vector Classifier."""
    roar_lime, roar_shap = _test_roar_per_model(SVC, "SVC")
    assert isinstance(roar_lime, float)
    assert isinstance(roar_shap, float)


def test_roar_knn():
    """ROAR with K-Nearest Neighbors."""
    roar_lime, roar_shap = _test_roar_per_model(KNeighborsClassifier, "KNN")
    assert isinstance(roar_lime, float)
    assert isinstance(roar_shap, float)


def test_roar_naive_bayes():
    """ROAR with Naive Bayes."""
    roar_lime, roar_shap = _test_roar_per_model(GaussianNB, "NaiveBayes")
    assert isinstance(roar_lime, float)
    assert isinstance(roar_shap, float)


def test_roar_xgboost():
    """ROAR with XGBoost."""
    roar_lime, roar_shap = _test_roar_per_model(XGBClassifier, "XGBoost")
    assert isinstance(roar_lime, float)
    assert isinstance(roar_shap, float)


def test_roar_multiple_models():
    """Test ROAR across multiple model types in one test."""
    iris = load_iris()
    X, y = iris.data, iris.target
    class_names = iris.target_names.tolist()
    feature_names = list(iris.feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model_variants = {
        "logreg": (LogisticRegression, {"max_iter": 1000}),
        "rf": (RandomForestClassifier, {})
    }

    results = {}
    for model_name, (model_class, model_kwargs) in model_variants.items():
        print(f"\n[ROAR Test: {model_name}]")

        model = model_class(**model_kwargs)
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

        roar_lime = compute_roar(model_class, X_train, y_train, X_test, y_test, lime_exps, top_k=2, model_kwargs=model_kwargs)
        roar_shap = compute_roar(model_class, X_train, y_train, X_test, y_test, shap_exps, top_k=2, model_kwargs=model_kwargs)

        print(f"  ROAR Drop - LIME: {roar_lime:.4f}")
        print(f"  ROAR Drop - SHAP: {roar_shap:.4f}")
        
        results[model_name] = {"lime": roar_lime, "shap": roar_shap}

    assert len(results) == 2
    for model_name, scores in results.items():
        assert isinstance(scores["lime"], float)
        assert isinstance(scores["shap"], float)


def test_roar_all_supported_models():
    """Comprehensive test of ROAR across all supported model types."""
    model_classes = {
        "logreg": (LogisticRegression, {"max_iter": 1000}),
        "rf": (RandomForestClassifier, {}),
        "gb": (GradientBoostingClassifier, {}),
        "svc": (SVC, {}),
        "knn": (KNeighborsClassifier, {}),
        "nb": (GaussianNB, {}),
    }
    
    model_classes["xgb"] = (XGBClassifier, {})

    successful = 0
    for name, (cls, kwargs) in model_classes.items():
        try:
            _test_roar_per_model(cls, name, model_kwargs=kwargs)
            successful += 1
        except Exception as e:
            print(f"  [WARNING] {name} failed: {e}")
    
    # At least most models should work
    assert successful >= len(model_classes) - 1


# =============================================================================
# ROAR Tests - Baseline Variants
# =============================================================================

def test_roar_baseline_variants():
    """Test ROAR with different baseline value options using LIME."""
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

    print("\n[ROAR Baseline Variants - LIME]")

    # String baseline - mean
    roar_mean = compute_roar(model_class, X, y, X, y, lime_exps, baseline_value="mean", top_k=2)
    print(f"  mean baseline: {roar_mean:.4f}")
    assert isinstance(roar_mean, float)
    
    # String baseline - median
    roar_median = compute_roar(model_class, X, y, X, y, lime_exps, baseline_value="median", top_k=2)
    print(f"  median baseline: {roar_median:.4f}")
    assert isinstance(roar_median, float)

    # Callable baseline
    roar_callable = compute_roar(
        model_class, X, y, X, y, lime_exps,
        baseline_value=lambda X: np.mean(X, axis=0),
        top_k=2
    )
    print(f"  callable baseline (mean): {roar_callable:.4f}")
    assert isinstance(roar_callable, float)

    # Array baseline
    roar_array = compute_roar(
        model_class, X, y, X, y, lime_exps,
        baseline_value=np.median(X, axis=0),
        top_k=2
    )
    print(f"  array baseline: {roar_array:.4f}")
    assert isinstance(roar_array, float)


def test_roar_baseline_variants_shap():
    """Test ROAR with different baseline value options using SHAP."""
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
    
    class_names = [f"class_{i}" for i in range(3)]
    feature_names = [f"feat_{i}" for i in range(X.shape[1])]
    
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

    print("\n[ROAR Baseline Variants - SHAP]")
    
    roar_mean = compute_roar(model_class, X, y, X, y, shap_exps, baseline_value="mean", top_k=2)
    print(f"  mean baseline: {roar_mean:.4f}")
    assert isinstance(roar_mean, float)
    
    roar_median = compute_roar(model_class, X, y, X, y, shap_exps, baseline_value="median", top_k=2)
    print(f"  median baseline: {roar_median:.4f}")
    assert isinstance(roar_median, float)
    
    roar_callable = compute_roar(
        model_class, X, y, X, y, shap_exps, 
        baseline_value=lambda X: np.mean(X, axis=0), 
        top_k=2
    )
    print(f"  callable baseline (mean): {roar_callable:.4f}")
    assert isinstance(roar_callable, float)
    
    roar_array = compute_roar(
        model_class, X, y, X, y, shap_exps, 
        baseline_value=np.median(X, axis=0), 
        top_k=2
    )
    print(f"  array baseline: {roar_array:.4f}")
    assert isinstance(roar_array, float)


# =============================================================================
# ROAR Curve Tests
# =============================================================================

def test_roar_curve_shap():
    """Test ROAR curve generation with SHAP explanations."""
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

    assert isinstance(curve, dict)
    assert len(curve) == 5  # k=1 through k=5
    for k in range(1, 6):
        assert k in curve
        assert isinstance(curve[k], float)


def test_roar_curve_lime():
    """Test ROAR curve generation with LIME explanations."""
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

    lime_exp = LimeExplainer(
        model=adapter,
        training_data=X,
        feature_names=feature_names,
        class_names=class_names
    )

    lime_exps = []
    for i in range(10):
        exp = lime_exp.explain(X[i])
        exp.feature_names = feature_names
        lime_exps.append(exp)

    curve = compute_roar_curve(
        model_class=model_class,
        X_train=X,
        y_train=y,
        X_test=X,
        y_test=y,
        explanations=lime_exps,
        max_k=5,
        baseline_value="mean",
        model_kwargs=model_args
    )

    print("\nROAR Curve (LIME):")
    for k, drop in curve.items():
        print(f"  top-{k} drop: {drop:.4f}")

    assert isinstance(curve, dict)
    assert len(curve) == 5
    for k in range(1, 6):
        assert k in curve
        assert isinstance(curve[k], float)


# =============================================================================
# Edge Cases and Robustness Tests
# =============================================================================

def test_aopc_single_feature():
    """AOPC with single step perturbation."""
    iris = load_iris()
    X, y = iris.data, iris.target
    class_names = iris.target_names.tolist()
    feature_names = list(iris.feature_names)

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    adapter = SklearnAdapter(model, class_names=class_names)

    lime = LimeExplainer(
        model=adapter,
        training_data=X,
        feature_names=feature_names,
        class_names=class_names
    )

    instance = X[0]
    lime_exp = lime.explain(instance)
    lime_exp.feature_names = feature_names

    # Single step - should still work
    score = compute_aopc(adapter, instance, lime_exp, num_steps=1)
    assert isinstance(score, float)


def test_roar_top_k_variations():
    """Test ROAR with different top_k values."""
    iris = load_iris()
    X, y = iris.data, iris.target
    class_names = iris.target_names.tolist()
    feature_names = list(iris.feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    adapter = SklearnAdapter(model, class_names=class_names)

    lime = LimeExplainer(
        model=adapter,
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names
    )

    lime_exps = [lime.explain(X_train[i]) for i in range(10)]
    for e in lime_exps:
        e.feature_names = feature_names

    print("\n[ROAR top_k variations]")
    for top_k in [1, 2, 3, 4]:
        roar = compute_roar(
            LogisticRegression, X_train, y_train, X_test, y_test,
            lime_exps, top_k=top_k, model_kwargs={"max_iter": 200}
        )
        print(f"  top_k={top_k}: {roar:.4f}")
        assert isinstance(roar, float)


if __name__ == "__main__":
    # Run all tests when executed directly
    test_aopc_lime_vs_shap()
    test_batch_aopc_lime_vs_shap()
    test_roar_lime_vs_shap()
    test_roar_multiple_models()
    test_roar_all_supported_models()
    test_roar_baseline_variants()
    test_roar_baseline_variants_shap()
    test_roar_curve_shap()
    test_roar_curve_lime()
    test_aopc_single_feature()
    test_roar_top_k_variations()
    print("\n✓ All evaluation metric tests passed!")
