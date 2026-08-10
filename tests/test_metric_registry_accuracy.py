"""Accuracy and trust-boundary tests for evaluation metric metadata."""

from dataclasses import FrozenInstanceError

import pytest

import explainiverse.evaluation as evaluation
from explainiverse.evaluation.registry import MetricMeta, MetricRegistry, build_metric_registry


def _metric_names():
    return {name for name in evaluation.__all__ if name.startswith("compute_")}


def test_default_metric_registry_exactly_covers_public_compute_endpoints():
    registry = evaluation.default_metric_registry

    assert set(registry.list_metrics()) == _metric_names()
    registry.validate_inventory(evaluation.__all__)
    assert len(registry.list_metrics()) == 131


def test_every_public_metric_has_falsifiable_audit_metadata():
    for name in _metric_names():
        meta = evaluation.default_metric_registry.get_meta(name)
        assert meta.claim_status in {"verified", "quarantined", "unverified"}
        assert meta.claim_scope.strip()
        assert meta.family.strip()
        assert meta.level in {"instance", "batch", "dataset"}
        assert meta.score_direction in {"higher", "lower", "zero", "contextual", "none"}
        if meta.claim_status == "unverified":
            assert meta.canonical_claim is False


def test_known_adaptations_do_not_claim_an_undeclared_canonical_method():
    registry = evaluation.default_metric_registry

    assert registry.get_meta("compute_pgi").canonical_claim is False
    assert "baseline-replacement" in registry.get_meta("compute_pgi").claim_scope
    assert registry.get_meta("compute_avg_sensitivity").canonical_claim is False
    assert "mean sensitivity" in registry.get_meta("compute_avg_sensitivity").claim_scope
    assert registry.get_meta("compute_attribution_iou").canonical_claim is False
    assert registry.get_meta("compute_effective_complexity").claim_status == "quarantined"


def test_metric_metadata_is_frozen_and_registry_reads_are_detached():
    registry = MetricRegistry()

    def metric(value):
        return value

    meta = MetricMeta(
        family="test",
        level="instance",
        claim_status="verified",
        claim_scope="Exact identity test metric.",
        canonical_claim=False,
        score_direction="none",
    )
    registry.register("compute_identity", metric, meta)
    entry = registry.get("compute_identity")
    entry["fn"] = lambda value: "mutated"

    assert registry.evaluate("compute_identity", 3) == 3
    with pytest.raises(FrozenInstanceError):
        entry["meta"].claim_scope = "mutated"


@pytest.mark.parametrize(
    "kwargs,error",
    [
        (
            {"family": "", "level": "instance", "claim_status": "verified", "claim_scope": "x"},
            TypeError,
        ),
        (
            {"family": "x", "level": "row", "claim_status": "verified", "claim_scope": "x"},
            ValueError,
        ),
        (
            {"family": "x", "level": "instance", "claim_status": "unknown", "claim_scope": "x"},
            ValueError,
        ),
        (
            {
                "family": "x",
                "level": "instance",
                "claim_status": "unverified",
                "claim_scope": "x",
                "canonical_claim": True,
            },
            ValueError,
        ),
    ],
)
def test_metric_meta_rejects_invalid_claims(kwargs, error):
    with pytest.raises(error):
        MetricMeta(**kwargs)


def test_registry_validates_override_and_inventory():
    registry = MetricRegistry()
    meta = MetricMeta(
        family="test",
        level="instance",
        claim_status="unverified",
        claim_scope="Not yet audited.",
    )
    registry.register("compute_one", lambda: 1, meta)

    with pytest.raises(ValueError, match="already registered"):
        registry.register("compute_one", lambda: 2, meta)
    with pytest.raises(TypeError, match="override"):
        registry.register("compute_one", lambda: 2, meta, override=1)
    with pytest.raises(ValueError, match="missing"):
        registry.validate_inventory(["compute_one", "compute_two"])


def test_builder_rejects_public_names_missing_from_namespace():
    with pytest.raises(ValueError, match="missing from the namespace"):
        build_metric_registry({}, ["compute_missing"])
