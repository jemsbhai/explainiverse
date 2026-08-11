"""Tests for the fail-closed manual accessibility evidence contract."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_accessibility_evidence.py"
SPEC = importlib.util.spec_from_file_location("validate_accessibility_evidence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
accessibility = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = accessibility
SPEC.loader.exec_module(accessibility)
POLICY = ROOT / ".github" / "accessibility-certification-policy.json"
NOW = datetime(2026, 8, 10, 18, 0, tzinfo=timezone.utc)


def _valid_evidence():
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    scenarios = [
        {"id": scenario, "result": "pass", "notes": f"Observed {scenario}."}
        for scenario in policy["required_scenarios"]
    ]
    artifacts = [
        {
            "kind": kind,
            "uri": f"https://evidence.example.invalid/{kind}",
            "sha256": "a" * 64,
        }
        for kind in policy["required_artifact_kinds"]
    ]
    return {
        "schema_version": 1,
        "commit_sha": "1" * 40,
        "demo_url": "https://example.invalid/explainiverse/",
        "deployment_provenance_uri": "https://example.invalid/builds/immutable-run-123",
        "demo_build_sha256": "c" * 64,
        "completed_at": "2026-08-10T17:00:00Z",
        "independent_reviewer": "Accessibility Reviewer",
        "reviewer_independent_from_implementation": True,
        "runs": [
            {
                "profile_id": profile["id"],
                "os_version": "exact OS build 1",
                "browser_version": "exact browser build 2",
                "assistive_technology_version": "exact AT build 3",
                "result": "pass",
                "scenarios": copy.deepcopy(scenarios),
                "artifacts": copy.deepcopy(artifacts),
            }
            for profile in policy["required_profiles"]
        ],
    }


def _validate(tmp_path, evidence):
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    return accessibility.validate_evidence(POLICY, path, expected_commit="1" * 40, now=NOW)


def test_complete_independent_manual_evidence_is_accepted(tmp_path):
    result = _validate(tmp_path, _valid_evidence())
    assert result["validated_profiles"] == [
        "macos-safari-voiceover",
        "windows-edge-nvda",
    ]


def test_missing_profile_or_scenario_is_rejected(tmp_path):
    evidence = _valid_evidence()
    evidence["runs"].pop()
    with pytest.raises(accessibility.AccessibilityEvidenceError, match="profile set"):
        _validate(tmp_path, evidence)

    evidence = _valid_evidence()
    evidence["runs"][0]["scenarios"].pop()
    with pytest.raises(accessibility.AccessibilityEvidenceError, match="scenario set"):
        _validate(tmp_path, evidence)


def test_failures_and_non_independent_review_cannot_be_certified(tmp_path):
    evidence = _valid_evidence()
    evidence["runs"][0]["scenarios"][0]["result"] = "fail"
    with pytest.raises(accessibility.AccessibilityEvidenceError, match="must have result 'pass'"):
        _validate(tmp_path, evidence)

    evidence = _valid_evidence()
    evidence["reviewer_independent_from_implementation"] = False
    with pytest.raises(accessibility.AccessibilityEvidenceError, match="independent"):
        _validate(tmp_path, evidence)


def test_artifacts_must_be_https_content_addressed_and_complete(tmp_path):
    evidence = _valid_evidence()
    evidence["runs"][0]["artifacts"].pop()
    with pytest.raises(accessibility.AccessibilityEvidenceError, match="missing artifacts"):
        _validate(tmp_path, evidence)

    evidence = _valid_evidence()
    evidence["runs"][0]["artifacts"][0]["uri"] = "http://example.invalid/transcript"
    with pytest.raises(accessibility.AccessibilityEvidenceError, match="HTTPS URI"):
        _validate(tmp_path, evidence)

    evidence = _valid_evidence()
    evidence["deployment_provenance_uri"] = "https://user:secret@example.invalid/build"
    with pytest.raises(accessibility.AccessibilityEvidenceError, match="HTTPS URI"):
        _validate(tmp_path, evidence)

    evidence = _valid_evidence()
    evidence["demo_build_sha256"] = "not-a-digest"
    with pytest.raises(accessibility.AccessibilityEvidenceError, match="lowercase SHA-256"):
        _validate(tmp_path, evidence)


def test_evidence_is_bound_to_the_checked_out_commit(tmp_path):
    evidence = _valid_evidence()
    evidence["commit_sha"] = "2" * 40
    with pytest.raises(accessibility.AccessibilityEvidenceError, match="checked-out commit"):
        _validate(tmp_path, evidence)


def test_stale_or_future_evidence_is_rejected(tmp_path):
    evidence = _valid_evidence()
    evidence["completed_at"] = "2025-01-01T00:00:00Z"
    with pytest.raises(accessibility.AccessibilityEvidenceError, match="stale"):
        _validate(tmp_path, evidence)

    evidence = _valid_evidence()
    evidence["completed_at"] = "2026-08-11T00:00:00Z"
    with pytest.raises(accessibility.AccessibilityEvidenceError, match="future"):
        _validate(tmp_path, evidence)


@pytest.mark.parametrize(
    "mutation",
    ("age", "profiles", "scenarios", "artifacts"),
)
def test_manual_evidence_policy_cannot_redirect_canonical_boundaries(tmp_path, mutation):
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    if mutation == "age":
        policy["max_evidence_age_days"] = 9999
    elif mutation == "profiles":
        policy["required_profiles"][0]["platform"] = "Emulated macOS"
    elif mutation == "scenarios":
        policy["required_scenarios"] = ["disclosure-and-landmarks"]
    else:
        policy["required_artifact_kinds"] = ["interaction-transcript"]

    policy_path = tmp_path / "accessibility-policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(_valid_evidence()), encoding="utf-8")

    with pytest.raises(accessibility.AccessibilityEvidenceError, match="canonical"):
        accessibility.validate_evidence(
            policy_path,
            evidence_path,
            expected_commit="1" * 40,
            now=NOW,
        )
