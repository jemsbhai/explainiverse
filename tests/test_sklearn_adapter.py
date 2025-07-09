from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from explainiverse.adapters.sklearn_adapter import SklearnAdapter
import numpy as np


def test_sklearn_adapter_prediction():
    data = load_iris()
    X, y = data.data, data.target
    clf = LogisticRegression(max_iter=200)
    clf.fit(X, y)
    adapter = SklearnAdapter(model=clf, class_names=data.target_names.tolist())
    preds = adapter.predict(X[:5])
    print("Adapter predictions:\n", preds)
    assert preds.shape == (5, 3)


def test_predict_proba_shape_and_range():
    data = load_iris()
    X, y = data.data, data.target
    clf = LogisticRegression(max_iter=200)
    clf.fit(X, y)
    adapter = SklearnAdapter(model=clf, class_names=data.target_names.tolist())
    preds = adapter.predict(X[:10])
    assert preds.shape == (10, 3)
    assert np.all(preds >= 0) and np.all(preds <= 1)
    assert np.allclose(preds.sum(axis=1), 1.0)


def test_predict_fallback_without_proba():
    data = load_iris()
    X, y = data.data, data.target
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    adapter = SklearnAdapter(model=clf, class_names=data.target_names.tolist())
    preds = adapter.predict(X[:3])
    assert preds.shape == (3, 3)
    assert set(preds.flatten()).issubset({0, 1})
    assert np.all(preds.sum(axis=1) == 1)



# if __name__ == "__main__":
#     test_predict_proba_shape_and_range()
#     test_predict_fallback_without_proba()
#     print(" SklearnAdapter tests passed.")
