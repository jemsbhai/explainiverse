"""ProtoDash can return an empty selected set with a fitted prediction model."""

import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor

from explainiverse.adapters.sklearn_adapter import SklearnAdapter
from explainiverse.explainers.example_based import ProtoDashExplainer


@pytest.fixture
def explainer():
    rows = np.array([[-1.0], [1.0], [2.0]])
    tree = DecisionTreeRegressor(random_state=0).fit(rows, 2 * rows[:, 0])
    return ProtoDashExplainer(
        SklearnAdapter(tree), n_prototypes=1, kernel="linear", force_n_prototypes=False
    )


@pytest.mark.parametrize("use_predictions", [False, True], ids=["input", "augmented"])
@pytest.mark.parametrize("target", [-1.0, 1.0], ids=["empty", "nonempty"])
def test_predictions_for_selected_rows(explainer, use_predictions, target):
    references = np.array([[1.0], [2.0]])
    data = explainer.explain(
        np.array([target]), references, use_predictions=use_predictions
    ).explanation_data

    # At training points the tree predicts 2*x. Linear similarities are x*r
    # in input space and 5*x*r after appending the prediction coordinate.
    expected_indices = [] if target < 0 else [1]
    assert data["prototype_indices"] == expected_indices
    assert data["n_prototypes"] == len(expected_indices)
    assert data["instance_prediction"] == [[2 * target]]
    assert data["prototype_predictions"] == ([] if target < 0 else [[4.0]])
    assert data["similarity_scores"] == ([] if target < 0 else [10.0 if use_predictions else 2.0])


def test_batch_with_both_empty_and_nonempty_selections(explainer):
    results = explainer.explain_batch(np.array([[1.0], [-1.0], [2.0]]), np.array([[1.0], [2.0]]))
    data = [result.explanation_data for result in results]
    assert [row["prototype_indices"] for row in data] == [[1], [], [1]]
    assert [row["instance_prediction"] for row in data] == [[[2.0]], [[-2.0]], [[4.0]]]
    assert [row["prototype_predictions"] for row in data] == [[[4.0]], [], [[4.0]]]
