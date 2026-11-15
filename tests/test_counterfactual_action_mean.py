"""Mean absolute normalized actions retain contributions below binary64 precision."""

from fractions import Fraction

import numpy as np
import pytest

from explainiverse.explainers import CounterfactualExplainer

_TINY = float.fromhex("0x0.0000000000001p-1022")


@pytest.mark.parametrize(
    "step,origin,offsets,bounds",
    [
        pytest.param(_TINY, 0, [1, 2, 4], (0, 4), id="round-up-from-zero"),
        pytest.param(_TINY, 2, [-1, 2, -4], (-2, 2), id="bidirectional-actions"),
        pytest.param(_TINY, 0, [1, 4, 13], (0, 4), id="halfway-rounds-up-to-even"),
        pytest.param(_TINY, 0, [3, 7, 20], (0, 4), id="halfway-rounds-down-to-even"),
        pytest.param(_TINY, 0, [1, 2, 3], (0, 4), id="halfway-to-zero"),
        pytest.param(2.0**-52, 0, [1, 2, 4], (-(2.0**1023), 2.0**1023), id="wide-range"),
        pytest.param(0.125, 0, [1, 2, 4], (0, 4), id="ordinary-control"),
    ],
)
def test_feature_attributions_round_the_mean_normalized_action(step, origin, offsets, bounds):
    query = [origin * step, 0.0]
    targets = [((origin + offset) * step, float(i % 2)) for i, offset in enumerate(offsets)]

    class ChangedCategory:
        prediction_output_kind = "probabilities"

        def predict(self, rows):
            target = rows[:, 0] != query[0]
            return np.column_stack((~target, target)).astype(float)

    explainer = CounterfactualExplainer(
        ChangedCategory(),
        [query, *targets],
        ["code", "flag"],
        categorical_features=["code", "flag"],
        feature_ranges={"code": bounds, "flag": (0, 1)},
    )
    data = explainer.explain(
        query, desired_class=1, num_counterfactuals=len(targets), max_attempts=len(targets)
    ).explanation_data
    assert sorted(tuple(row) for row in data["counterfactuals"]) == sorted(targets)

    span = Fraction(bounds[1]) - Fraction(bounds[0])
    total_action = sum(abs(Fraction(code) - Fraction(query[0])) for code, _ in targets)
    assert data["feature_attributions"] == {
        "code": float(total_action / (span * len(targets))),
        "flag": sum(flag for _, flag in targets) / len(targets),
    }
