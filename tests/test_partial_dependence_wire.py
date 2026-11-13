"""Small, finite PDP results should survive the public wire JSON round trip."""

import json

import numpy as np
import pytest

from explainiverse.core.explanation import EXPLANATION_WIRE_SCHEMA_VERSION, Explanation
from explainiverse.explainers import PartialDependenceExplainer


class LinearModel:
    task = "regression"

    def predict(self, matrix):
        return matrix[:, 0] + 2 * matrix[:, 1]


@pytest.mark.parametrize("selection", ["x", ("x", "z")])
@pytest.mark.parametrize("percentiles", [None, (12.5, 87.5)])
def test_pdp_returns_wire_ready_percentile_metadata(selection, percentiles):
    kwargs = {} if percentiles is None else {"percentile_range": percentiles}
    explainer = PartialDependenceExplainer(
        LinearModel(), np.array([[1, 1], [2, 3]], dtype=float), ["x", "z"], **kwargs
    )
    result = explainer.explain([selection])
    expected_range = [5.0, 95.0] if percentiles is None else list(percentiles)
    expected_scores = {"x": [5.0, 6.0]} if selection == "x" else {"x_x_z": [[3.0, 7.0], [4.0, 8.0]]}
    assert result.explanation_data["pdp_values"] == expected_scores
    wire = result.to_wire_dict()
    assert wire["schema_version"] == EXPLANATION_WIRE_SCHEMA_VERSION
    assert wire["metadata"]["percentile_range"] == expected_range
    assert (
        Explanation.from_wire_dict(json.loads(json.dumps(wire, allow_nan=False))).to_wire_dict()
        == wire
    )


def test_generic_wire_rejects_tuple_and_broad_native_payload_retains_it():
    result = Explanation("generic", "target", {}, metadata={"bounds": (1.0, 2.0)})
    native = result.to_dict()
    assert type(native["metadata"]["bounds"]) is tuple
    with pytest.raises(TypeError, match="payload.metadata.bounds"):
        result.to_wire_dict()


def test_exact_wide_pdp_grid_still_exceeds_wire_v1_safe_integer_contract():
    result = PartialDependenceExplainer(
        LinearModel(), np.array([[2**53, 0], [2**53 + 1, 0]], dtype=np.int64), ["x", "z"]
    ).explain(["x"])
    assert result.explanation_data["grid_values"]["x"] == [2**53, 2**53 + 1]
    with pytest.raises(ValueError, match="safe-integer"):
        result.to_wire_dict()
