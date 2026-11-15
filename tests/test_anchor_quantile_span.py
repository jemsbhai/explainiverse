"""Both tabular Anchors discretizers preserve finite linear quantile boundaries."""

from fractions import Fraction

import numpy as np
import pytest

from explainiverse.explainers import AnchorsExplainer, AnchorTabularExplainer

_MAX = float(np.finfo(float).max)


class _SignClassifier:
    def predict(self, rows):
        positive = (rows[:, 0] > 0).astype(float)
        return np.column_stack((1 - positive, positive))


def _exact_cuts(levels, discretizer):
    ordered = sorted(Fraction(value) for value in levels)
    percentages = [25, 50, 75] if discretizer == "quartile" else range(10, 100, 10)
    cuts = set()
    for percentage in percentages:
        rank = Fraction((len(ordered) - 1) * percentage, 100)
        index = int(rank)
        weight = rank - index
        lower = ordered[index]
        upper = ordered[min(index + 1, len(ordered) - 1)]
        cuts.add(float((1 - weight) * lower + weight * upper))
    return sorted(cuts)


def _expected_bounds(cuts, values, query, canonical):
    if not canonical:
        # The historical explainer fixes the feature to the query's bin,
        # which includes its right endpoint.
        return (
            max((cut for cut in cuts if cut < query), default=None),
            min((cut for cut in cuts if query <= cut), default=None),
        )

    # The canonical explainer keeps the first predicate for each support.
    # Among pure predicates the one covering the most rows is preferred.
    best_count = 0
    best_bounds = None
    for cut in cuts:
        bounds = (None, cut) if query <= cut else (cut, None)
        support = values <= cut if query <= cut else values > cut
        count = np.count_nonzero(support)
        if count > best_count and np.all((values[support] > 0) == (query > 0)):
            best_count = count
            best_bounds = bounds
    assert best_bounds is not None
    return best_bounds


@pytest.mark.parametrize("explainer_class", [AnchorTabularExplainer, AnchorsExplainer])
@pytest.mark.parametrize(
    "levels,query,discretizer",
    [
        ([-_MAX, _MAX], _MAX, "quartile"),
        ([_MAX, -_MAX], -_MAX, "quartile"),
        ([-_MAX, _MAX], _MAX, "decile"),
        ([_MAX / 2, -_MAX], -_MAX, "decile"),
        ([_MAX, -_MAX, _MAX, -_MAX], _MAX, "quartile"),
        ([-8.0, 8.0], 8.0, "quartile"),
        ([1.0, -1.0, 0.0, -1.0, 1.0], 0.0, "quartile"),
    ],
    ids=[
        "positive-extreme",
        "negative-extreme",
        "extreme-deciles",
        "asymmetric-deciles",
        "repeated-extremes",
        "ordinary",
        "query-at-interior-cut",
    ],
)
def test_anchor_rules_use_finite_empirical_quantiles(explainer_class, levels, query, discretizer):
    values = np.asarray(levels)
    canonical = explainer_class is AnchorTabularExplainer
    options = {"batch_size": 64, "max_samples": 2048} if canonical else {"n_samples": 128}
    explainer = explainer_class(
        _SignClassifier(),
        values[:, None],
        ["x"],
        ["nonpositive", "positive"],
        discretizer=discretizer,
        threshold=0.85,
        max_anchor_size=1,
        random_state=1,
        **options,
    )
    explanation = explainer.explain(np.array([query]))
    result = explanation.explanation_data

    cuts = _exact_cuts(levels, discretizer)
    lower, upper = _expected_bounds(cuts, values, query, canonical)
    support = np.ones(len(values), dtype=bool)
    if lower is not None:
        support &= values > lower
    if upper is not None:
        support &= values <= upper
    assert support.any() and np.all((values[support] > 0) == (query > 0))
    assert explanation.target_class == ("positive" if query > 0 else "nonpositive")
    assert result["coverage"] == np.mean(support)
    assert result["precision"] == 1.0
    (condition,) = result["rule_conditions"]
    assert result["rules"] == [condition["label"]]
    assert condition["feature"] == "x"
    assert condition["lower_bound"] == lower
    assert condition["upper_bound"] == upper
    assert condition["lower_inclusive"] is (False if lower is not None else None)
    assert condition["upper_inclusive"] is (True if upper is not None else None)
    if canonical:
        assert result["is_certified_anchor"] is True
        assert condition["threshold"] == (upper if lower is None else lower)
        np.testing.assert_array_equal(explainer.cuts[0], cuts)
    else:
        assert result["meets_empirical_precision_threshold"] is True
        np.testing.assert_array_equal(explainer.bins[0], cuts)
