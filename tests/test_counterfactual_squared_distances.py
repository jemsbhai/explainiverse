"""Public proximity and diversity retain the sum of tiny coordinate squares."""

from fractions import Fraction

import numpy as np
import pytest

from explainiverse.explainers import CounterfactualExplainer

_STEP = 2.0**-538


def _squared_distance(left, right):
    # All feature ranges are [0, 1], so the coordinates are already normalized.
    return sum((Fraction(a) - Fraction(b)) ** 2 for a, b in zip(left, right))


def _explain(steps, attempts):
    # Moving every coordinate is farther than moving the first coordinate.
    # Supply the farther exemplar first so a spurious tie affects the search.
    query = [step if i % 2 else 0.0 for i, step in enumerate(steps)]
    far = [step - x for step, x in zip(steps, query)]
    near = [far[0], *query[1:]]

    class ChangedClassifier:
        prediction_output_kind = "probabilities"

        def predict(self, rows):
            target = np.any(rows != query, axis=1)
            return np.column_stack((~target, target)).astype(float)

    names = [f"x{i}" for i in range(len(steps))]
    explainer = CounterfactualExplainer(
        ChangedClassifier(),
        [query, far, near],
        names,
        categorical_features=names,
        feature_ranges={name: (0.0, 1.0) for name in names},
    )
    data = explainer.explain(
        query, desired_class=1, max_attempts=attempts, num_counterfactuals=attempts
    ).explanation_data
    return query, far, near, data


@pytest.mark.parametrize(
    "steps",
    [
        pytest.param([_STEP] * 2, id="halfway-rounds-to-zero"),
        pytest.param([np.nextafter(_STEP, 1.0)] * 2, id="just-above-halfway"),
        pytest.param([_STEP] * 4, id="four-quarter-subnormals"),
        pytest.param([_STEP] * 6, id="halfway-rounds-to-even"),
        pytest.param([1.5 * _STEP] * 3, id="individual-squares-round-up"),
        pytest.param([2 * _STEP, _STEP, _STEP, _STEP], id="mixed-size-changes"),
        pytest.param([0.25, 0.125, 0.375], id="ordinary-control"),
    ],
)
def test_proximity_and_pairwise_diversity_round_the_sum_of_squares(steps):
    query, far, near, data = _explain(steps, attempts=2)
    candidates = data["counterfactuals"]
    assert {tuple(row) for row in candidates} == {tuple(far), tuple(near)}
    assert data["normalized_squared_distances"] == [
        float(_squared_distance(row, query)) for row in candidates
    ]
    assert data["diversity_loss"] == -float(_squared_distance(far, near))


@pytest.mark.parametrize("step", [_STEP, 0.25], ids=["subnormal-squares", "ordinary-control"])
def test_one_attempt_prefers_the_nearer_exemplar(step):
    query, far, near, data = _explain([step] * 4, attempts=1)
    assert float(_squared_distance(near, query)) < float(_squared_distance(far, query))
    assert data["counterfactuals"] == [near]
