"""SAGE zero-one loss rounds the observed fraction of classification errors once."""

import numpy as np
import pytest

from explainiverse.explainers import SAGEExplainer


def _explain(targets, full_labels, classes, one_column):
    # A separate donor at row zero always predicts class zero.
    indices = np.asarray([0, *full_labels])
    probabilities = np.eye(len(classes))[indices]

    class LookupClassifier:
        classes_ = np.asarray(classes)
        prediction_output_kind = "probabilities"

        def predict(self, rows):
            outputs = probabilities[rows[:, 0]]
            return outputs[:, 1:] if one_column else outputs

    return (
        SAGEExplainer(
            LookupClassifier(),
            np.arange(1, len(targets) + 1)[:, None],
            np.asarray(targets),
            ["x"],
            background_data=np.array([[0]]),
            n_permutations=1,
        )
        .explain()
        .explanation_data
    )


@pytest.mark.parametrize(
    "target_indices, full_indices, classes, one_column",
    [
        pytest.param([0, 1, 0], [0, 1, 0], [0, 1], True, id="one-in-three-baseline"),
        pytest.param(
            [0] * 10, [0] * 9 + [2], ["red", "green", "blue"], False, id="one-in-ten-full"
        ),
        pytest.param([0, 1, 1], [0, 1, 0], [5, 2], False, id="two-nondyadic-losses"),
        pytest.param([0, 1, 0, 1], [0, 1, 1, 1], [0, 1], True, id="dyadic-losses"),
        pytest.param([1, 1, 1], [1, 1, 1], [0, 1], False, id="all-wrong-to-perfect"),
    ],
)
def test_sage_zero_one_loss_from_error_counts(target_indices, full_indices, classes, one_column):
    targets = [classes[index] for index in target_indices]
    data = _explain(targets, full_indices, classes, one_column)
    baseline = sum(index != 0 for index in target_indices) / len(targets)
    full = sum(a != b for a, b in zip(target_indices, full_indices)) / len(targets)
    assert data["baseline_loss"] == baseline
    assert data["full_loss"] == full
    assert data["total_value"] == baseline - full
    assert data["feature_attributions"] == {"x": baseline - full}
    assert data["efficiency_error"] == 0.0


def test_sage_zero_one_loss_rejects_continuous_targets():
    with pytest.raises(ValueError):
        _explain([0, 0.25, 1], [0, 0, 1], [0, 1], True)
