"""Counterfactual proximity, diversity and actions use the declared feature span."""

from fractions import Fraction

import numpy as np
import pytest

from explainiverse.explainers import CounterfactualExplainer


class _NonzeroClassifier:
    prediction_output_kind = "probabilities"

    def predict(self, rows):
        target = np.any(rows != 0.0, axis=1)
        return np.column_stack((~target, target)).astype(float)


_MAX = float(np.finfo(float).max)


@pytest.mark.parametrize(
    "targets,bounds",
    [
        pytest.param((-16.0, 16.0), (-32.0, 32.0), id="ordinary-control"),
        pytest.param((-_MAX / 2, _MAX / 2), (-_MAX, _MAX), id="wide-declared-span"),
        pytest.param((-_MAX / 2, _MAX / 4), (-_MAX, _MAX / 2), id="asymmetric-span"),
        pytest.param((-_MAX, _MAX), None, id="inferred-span-and-pairwise-difference"),
    ],
)
def test_normalized_geometry_for_observed_target_categories(targets, bounds):
    training = np.array([[0.0], [targets[0]], [targets[1]]])
    explainer = CounterfactualExplainer(
        _NonzeroClassifier(),
        training,
        ["code"],
        categorical_features=["code"],
        feature_ranges={"code": bounds} if bounds is not None else None,
    )
    data = explainer.explain(
        [0.0], desired_class=1, num_counterfactuals=2, max_attempts=2
    ).explanation_data
    values = [row[0] for row in data["counterfactuals"]]
    assert set(values) == set(targets)

    lower, upper = bounds or targets
    span = Fraction(upper) - Fraction(lower)
    offsets = [Fraction(value) / span for value in values]
    np.testing.assert_allclose(
        data["normalized_squared_distances"],
        [float(offset**2) for offset in offsets],
        rtol=1e-15,
        atol=0.0,
    )
    assert data["feature_attributions"]["code"] == pytest.approx(
        float(sum(map(abs, offsets)) / len(offsets)), rel=1e-15, abs=0.0
    )
    assert data["diversity_loss"] == pytest.approx(
        -float((offsets[0] - offsets[1]) ** 2), rel=1e-15, abs=0.0
    )


@pytest.mark.parametrize("limit", [32.0, _MAX], ids=["ordinary-control", "wide-span"])
def test_single_attempt_selects_nearest_normalized_target_across_features(limit):
    # The first target moves a quarter of the first feature's span. The
    # second moves only an eighth of the second feature's span.
    training = np.array([[0.0, 0.0], [limit / 2, 0.0], [0.0, 1.0]])
    explainer = CounterfactualExplainer(
        _NonzeroClassifier(),
        training,
        ["first", "second"],
        categorical_features=["first", "second"],
        feature_ranges={"first": (-limit, limit), "second": (0.0, 8.0)},
    )
    data = explainer.explain(
        training[0], desired_class=1, num_counterfactuals=1, max_attempts=1
    ).explanation_data

    assert data["counterfactuals"] == [[0.0, 1.0]]
    assert data["normalized_squared_distances"] == [1.0 / 64]
    assert data["feature_attributions"] == {"first": 0.0, "second": 1.0 / 8}
