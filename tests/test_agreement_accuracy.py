"""Primary-formula and identity-alignment tests for agreement metrics."""

import numpy as np
import pytest

from explainiverse.core.explanation import Explanation
from explainiverse.evaluation.agreement import (
    compute_batch_feature_agreement,
    compute_feature_agreement,
    compute_rank_agreement,
)


def _explanation(values, names):
    return Explanation(
        explainer_name="test",
        target_class="0",
        explanation_data={
            "feature_attributions": {name: value for name, value in zip(names, values)}
        },
        feature_names=list(names),
    )


def test_feature_and_rank_agreement_match_paper_formulas():
    a = np.array([0.9, 0.8, 0.1, 0.05])
    b = np.array([0.8, 0.05, 0.9, 0.1])

    # top_2(a)=[0,1], top_2(b)=[2,0]: one common feature, at rank 1 vs 2.
    assert compute_feature_agreement(a, b, k=2) == pytest.approx(1 / 2)
    assert compute_rank_agreement(a, b, k=2) == 0.0


def test_explanations_align_by_feature_identity_not_storage_position():
    a = _explanation([0.9, 0.5, 0.1], ["age", "income", "debt"])
    b = _explanation([0.1, 0.9, 0.5], ["debt", "age", "income"])

    assert compute_feature_agreement(a, b, k=2) == 1.0
    assert compute_rank_agreement(a, b, k=2) == 1.0


def test_different_feature_sets_fail_instead_of_comparing_positions():
    a = _explanation([0.9, 0.1], ["age", "income"])
    b = _explanation([0.9, 0.1], ["age", "debt"])

    with pytest.raises(ValueError, match="same feature-name set"):
        compute_feature_agreement(a, b, k=1)


def test_missing_named_attribution_is_not_silently_filled_with_zero():
    invalid = Explanation(
        explainer_name="invalid",
        target_class="0",
        explanation_data={"feature_attributions": {"age": 1.0}},
        feature_names=["age", "income"],
    )
    with pytest.raises(ValueError, match="match feature_names exactly"):
        compute_feature_agreement(invalid, np.array([1.0, 0.0]), k=1)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_attributions_fail(bad_value):
    with pytest.raises(ValueError, match="only finite"):
        compute_feature_agreement(np.array([1.0, bad_value]), np.array([1.0, 0.5]), k=1)


def test_feature_set_tie_at_cutoff_is_explicitly_undefined():
    with pytest.raises(ValueError, match="tie spans the cutoff"):
        compute_feature_agreement(np.array([1.0, 0.5, 0.5]), np.array([1.0, 0.6, 0.4]), k=2)


def test_rank_tie_within_selected_features_is_explicitly_undefined():
    with pytest.raises(ValueError, match="tied magnitudes"):
        compute_rank_agreement(np.array([1.0, 1.0, 0.0]), np.array([1.0, 0.5, 0.0]), k=2)


def test_boolean_k_is_not_accepted_as_integer_one():
    with pytest.raises(ValueError, match="positive integer"):
        compute_feature_agreement(np.array([1.0]), np.array([1.0]), k=True)


def test_empty_batch_is_rejected_as_an_undefined_aggregate():
    with pytest.raises(ValueError, match="must not be empty"):
        compute_batch_feature_agreement([], [], k=1)
