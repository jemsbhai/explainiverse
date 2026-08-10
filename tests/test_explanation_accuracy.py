"""Failure-contract tests for the generic :class:`Explanation` container."""

import math

import pytest

from explainiverse.core.explanation import Explanation


def _explanation(**overrides):
    values = {
        "explainer_name": "test",
        "target_class": 1,
        "explanation_data": {"feature_attributions": {"a": 2.0, "b": -3.0}},
        "feature_names": ["a", "b"],
        "metadata": {"score_space": "raw"},
    }
    values.update(overrides)
    return Explanation(**values)


@pytest.mark.parametrize("name", ["", "  ", None, 1])
def test_explainer_name_must_be_nonempty(name):
    with pytest.raises((TypeError, ValueError)):
        _explanation(explainer_name=name)


def test_constructor_copies_mutable_inputs():
    payload = {"feature_attributions": {"a": 1.0}}
    names = ["a"]
    metadata = {"nested": {"value": 1}}
    explanation = _explanation(explanation_data=payload, feature_names=names, metadata=metadata)

    payload["feature_attributions"]["a"] = 9.0
    names[0] = "changed"
    metadata["nested"]["value"] = 9

    assert explanation.explanation_data["feature_attributions"] == {"a": 1.0}
    assert explanation.feature_names == ["a"]
    assert explanation.metadata == {"nested": {"value": 1}}


def test_constructor_rejects_duplicate_feature_names():
    with pytest.raises(ValueError, match="unique"):
        _explanation(feature_names=["duplicate", "duplicate"])


def test_to_dict_returns_a_defensive_copy():
    explanation = _explanation()
    serialized = explanation.to_dict()
    serialized["explanation_data"]["feature_attributions"]["a"] = 99.0
    serialized["metadata"]["score_space"] = "changed"

    assert explanation.explanation_data["feature_attributions"]["a"] == 2.0
    assert explanation.metadata["score_space"] == "raw"


def test_get_attributions_rejects_a_false_mapping_contract():
    explanation = _explanation(explanation_data={"feature_attributions": [1.0]})
    with pytest.raises(TypeError, match="must be a mapping"):
        explanation.get_attributions()


@pytest.mark.parametrize("k", [True, False, 0, -1, 1.5, "1"])
def test_top_features_requires_positive_integer_k(k):
    with pytest.raises((TypeError, ValueError)):
        _explanation().get_top_features(k=k)


@pytest.mark.parametrize("value", [True, "1", None, math.inf, -math.inf, math.nan])
def test_top_features_rejects_invalid_attribution_values(value):
    explanation = _explanation(explanation_data={"feature_attributions": {"a": value}})
    with pytest.raises((TypeError, ValueError)):
        explanation.get_top_features()


def test_top_features_preserves_signed_value_while_ranking_by_magnitude():
    assert _explanation().get_top_features(k=1) == [("b", -3.0)]


def test_plot_fails_instead_of_claiming_to_render():
    with pytest.raises(NotImplementedError, match="no implemented plotting backend"):
        _explanation().plot()


def test_from_dict_validates_required_fields_and_input_type():
    with pytest.raises(TypeError, match="must be a mapping"):
        Explanation.from_dict([])
    with pytest.raises(ValueError, match="target_class"):
        Explanation.from_dict({"explainer_name": "test", "explanation_data": {}})
