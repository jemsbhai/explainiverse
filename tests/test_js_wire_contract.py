"""Cross-language fixture for the experimental JavaScript Explanation wire contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from explainiverse.core.explanation import Explanation

FIXTURE = (
    Path(__file__).parents[1] / "packages" / "js" / "tests" / "fixtures" / "explanation-wire.json"
)


def test_javascript_wire_fixture_is_accepted_and_round_trips_in_python():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    restored = Explanation.from_dict(payload)

    assert restored.to_dict() == payload
    assert restored.get_top_features(k=2) == [("alpha", 0.75), ("beta", -0.25)]


def test_python_rejects_duplicate_feature_names_from_wire_payload():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["feature_names"] = ["alpha", "alpha"]

    with pytest.raises(ValueError, match="unique"):
        Explanation.from_dict(payload)
