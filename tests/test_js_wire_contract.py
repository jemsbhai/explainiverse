"""Cross-language fixture for the experimental JavaScript Explanation wire contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from explainiverse.core.explanation import Explanation

FIXTURE = (
    Path(__file__).parents[1] / "packages" / "js" / "tests" / "fixtures" / "explanation-wire.json"
)
REJECTION_FIXTURE = (
    Path(__file__).parents[1]
    / "packages"
    / "js"
    / "tests"
    / "fixtures"
    / "explanation-wire-rejections.json"
)


def test_javascript_wire_fixture_is_accepted_and_round_trips_in_python():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    restored = Explanation.from_wire_dict(payload)

    assert restored.to_wire_dict() == payload
    assert restored.get_top_features(k=2) == [("alpha", 0.75), ("beta", -0.25)]


def test_python_rejects_duplicate_feature_names_from_wire_payload():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["feature_names"] = ["alpha", "alpha"]

    with pytest.raises(ValueError, match="unique"):
        Explanation.from_wire_dict(payload)


def test_python_rejects_every_shared_cross_language_rejection_fixture():
    fixtures = json.loads(REJECTION_FIXTURE.read_text(encoding="utf-8"))

    for fixture in fixtures["cases"]:
        with pytest.raises((TypeError, ValueError), match=".+"):
            Explanation.from_wire_dict(fixture["payload"])


def test_python_wire_is_json_exact_and_does_not_change_broad_to_dict_contract():
    explanation = Explanation(
        "wire",
        "target",
        {"diagnostics": [True, None, "finite", 3.5]},
        metadata={"nested": {"count": 2}},
    )

    payload = explanation.to_wire_dict()
    transported = json.loads(json.dumps(payload, allow_nan=False))

    assert Explanation.from_wire_dict(transported).to_wire_dict() == payload
    assert "schema_version" not in explanation.to_dict()


@pytest.mark.parametrize("target", [0, None, [], ""])
def test_python_wire_producer_rejects_target_values_that_v1_consumers_cannot_read(target):
    explanation = Explanation("wire", target, {"value": 1.0})

    assert explanation.to_dict()["target_class"] == target
    with pytest.raises(ValueError, match="target_class must be a non-empty string"):
        explanation.to_wire_dict()


@pytest.mark.parametrize(
    "value,match",
    [
        (float("nan"), "finite"),
        (float("inf"), "finite"),
        (-0.0, "negative zero"),
        (2**53, "safe-integer"),
        (float(2**53), "safe-integer"),
    ],
)
def test_python_wire_rejects_lossy_numbers(value, match):
    explanation = Explanation("wire", "target", {"value": value})

    with pytest.raises(ValueError, match=match):
        explanation.to_wire_dict()


def test_python_wire_rejects_numpy_and_cyclic_payloads_without_coercion():
    import numpy as np

    with pytest.raises(TypeError, match="JSON"):
        Explanation("wire", "target", {"value": np.float64(1.0)}).to_wire_dict()

    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(TypeError, match="cyclic"):
        Explanation("wire", "target", {"value": cyclic}).to_wire_dict()


def test_python_wire_matches_javascript_safe_number_boundaries():
    values = [
        float(2**53 - 1),
        float(-(2**53 - 1)),
        float(2**51) + 0.5,
    ]
    explanation = Explanation("wire", "target", {"values": values})
    payload = explanation.to_wire_dict()

    assert values[2].is_integer() is False
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
