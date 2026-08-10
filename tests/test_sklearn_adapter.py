import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from explainiverse.adapters.sklearn_adapter import SklearnAdapter


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


def test_regression_task_is_not_inferred_from_class_names():
    """Display names must not turn continuous predictions into class indices."""
    X = np.array([[-1.0], [0.0], [1.0], [2.0]])
    y = 3.0 * X[:, 0] - 2.0
    model = LinearRegression().fit(X, y)

    adapter = SklearnAdapter(model=model, class_names=["output"])
    predictions = adapter.predict(X)

    assert adapter.task == "regression"
    assert predictions.shape == (4, 1)
    np.testing.assert_allclose(predictions[:, 0], model.predict(X))


def test_classifier_without_predict_proba_maps_noncontiguous_labels():
    """Fallback indicators follow classes_, not integer label values."""
    data = load_iris()
    labels = np.array([10, 20, 40])[data.target]
    model = SVC(kernel="linear", probability=False).fit(data.data, labels)
    adapter = SklearnAdapter(model=model, class_names=["a", "b", "c"])

    predictions = adapter.predict(data.data[:10])
    predicted_labels = model.predict(data.data[:10])

    assert predictions.shape == (10, 3)
    np.testing.assert_array_equal(model.classes_[predictions.argmax(axis=1)], predicted_labels)
    np.testing.assert_allclose(predictions.sum(axis=1), 1.0)


def test_classifier_without_predict_proba_maps_string_labels():
    """Fallback indicators support arbitrary hashable sklearn class labels."""
    data = load_iris()
    labels = np.array(["alpha", "beta", "gamma"])[data.target]
    model = SVC(kernel="linear", probability=False).fit(data.data, labels)
    adapter = SklearnAdapter(model=model)

    predictions = adapter.predict(data.data[:10])
    predicted_labels = model.predict(data.data[:10])

    np.testing.assert_array_equal(model.classes_[predictions.argmax(axis=1)], predicted_labels)


def test_multioutput_classifier_is_rejected_instead_of_returning_three_dimensions():
    X = np.array(
        [
            [-2.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [2.0, 1.0],
        ]
    )
    y = np.column_stack((X[:, 0] > 0, X[:, 1] > 0)).astype(int)
    model = MultiOutputClassifier(LogisticRegression()).fit(X, y)

    with pytest.raises(ValueError, match="multi-output classification"):
        SklearnAdapter(model)


def test_predict_proba_width_must_match_class_metadata():
    data = load_iris()
    model = LogisticRegression(max_iter=200).fit(data.data, data.target)

    with pytest.raises(ValueError, match="class_names"):
        SklearnAdapter(model, class_names=["only", "two"])


# if __name__ == "__main__":
#     test_predict_proba_shape_and_range()
#     test_predict_fallback_without_proba()
#     print(" SklearnAdapter tests passed.")
