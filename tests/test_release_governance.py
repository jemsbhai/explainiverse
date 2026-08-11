"""The single-operator fallback must be explicit and evidence-bound."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "create_release_governance_record.py"
SPEC = importlib.util.spec_from_file_location("create_release_governance_record", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
governance = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = governance
SPEC.loader.exec_module(governance)

SHA = "a" * 40


def _evidence():
    policy_bytes = (ROOT / ".github" / "release-control-policy.json").read_bytes()
    policy = json.loads(policy_bytes)
    snapshot = {
        "schema_version": 1,
        "observed_at": "2026-08-11T07:00:00+00:00",
        "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "repository_controls_accepted": True,
        "violations": [],
        "observation": {
            "repository": policy["repository"],
            "release_tag": "v0.15.0",
            "release_commit": SHA,
            "capture_principal": "jemsbhai",
            "pypi_environment": {
                "name": "pypi",
                "can_admins_bypass": False,
                "protection_rules": [
                    {
                        "type": "required_reviewers",
                        "prevent_self_review": False,
                        "reviewers": [{"type": "User", "reviewer": {"login": "jemsbhai"}}],
                    }
                ],
            },
        },
        "workflow_run": {"id": "123", "actor": "jemsbhai"},
        "cuda_evidence": {"run": {"id": 456, "head_sha": SHA}},
    }
    return policy_bytes, snapshot


def _record(policy_bytes, snapshot):
    return governance.build_record(
        policy_bytes=policy_bytes,
        snapshot_bytes=(json.dumps(snapshot, sort_keys=True) + "\n").encode(),
        repository="jemsbhai/explainiverse",
        release_tag="v0.15.0",
        release_commit=SHA,
        preflight_run_id="123",
        cuda_run_id="456",
        release_run_id="789",
        release_run_attempt="1",
        release_actor="jemsbhai",
        release_triggering_actor="jemsbhai",
    )


def test_single_operator_record_discloses_non_independence_and_direct_evidence():
    policy_bytes, snapshot = _evidence()
    record = _record(policy_bytes, snapshot)
    assert record["governance"] == {
        "mode": "single_operator_disclosed",
        "segregation_of_duties": False,
        "capture_principal": "jemsbhai",
        "release_dispatch_actor": "jemsbhai",
        "release_triggering_actor": "jemsbhai",
        "release_run_attempt": "1",
        "environment_reviewer_logins": ["jemsbhai"],
        "prevent_self_review": False,
    }
    markdown = governance.render_markdown(record)
    assert markdown.startswith("<!-- explainiverse-release-governance-v1 -->")
    assert "does not claim segregation of duties" in markdown
    assert record["evidence"]["external_controls_sha256"] in markdown
    assert "/actions/runs/123" in markdown and "/actions/runs/456" in markdown


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda _policy, snapshot: snapshot.update(repository_controls_accepted=False), "accepted"),
        (lambda _policy, snapshot: snapshot.update(violations=["ignored"]), "accepted"),
        (
            lambda _policy, snapshot: snapshot["observation"].update(release_commit="b" * 40),
            "commit mismatch",
        ),
        (lambda _policy, snapshot: snapshot["workflow_run"].update(id="999"), "preflight run id"),
        (
            lambda _policy, snapshot: snapshot["workflow_run"].update(actor="other"),
            "preflight actor",
        ),
        (lambda _policy, snapshot: snapshot["cuda_evidence"]["run"].update(id=999), "CUDA run id"),
        (
            lambda policy, _snapshot: policy["pypi_environment"].update(
                required_reviewer_logins=["jemsbhai", "reviewer"]
            ),
            "sole reviewed",
        ),
        (
            lambda policy, _snapshot: policy["pypi_environment"].update(prevent_self_review=True),
            "single-operator mode",
        ),
        (
            lambda _policy, snapshot: snapshot["observation"]["pypi_environment"][
                "protection_rules"
            ][0].update(prevent_self_review=True),
            "self-review setting",
        ),
        (
            lambda _policy, snapshot: snapshot["observation"]["pypi_environment"][
                "protection_rules"
            ][0]["reviewers"][0]["reviewer"].update(login="attacker"),
            "reviewers differ",
        ),
    ],
)
def test_governance_record_rejects_false_or_mismatched_evidence(mutation, match):
    policy_bytes, snapshot = _evidence()
    policy = json.loads(policy_bytes)
    mutation(policy, snapshot)
    policy_bytes = (json.dumps(policy, indent=2) + "\n").encode()
    snapshot["policy_sha256"] = hashlib.sha256(policy_bytes).hexdigest()
    with pytest.raises(ValueError, match=match):
        _record(policy_bytes, snapshot)


def test_governance_record_rejects_release_actor_independence_fiction():
    policy_bytes, snapshot = _evidence()
    with pytest.raises(ValueError, match="release actor"):
        governance.build_record(
            policy_bytes=policy_bytes,
            snapshot_bytes=json.dumps(snapshot).encode(),
            repository="jemsbhai/explainiverse",
            release_tag="v0.15.0",
            release_commit=SHA,
            preflight_run_id="123",
            cuda_run_id="456",
            release_run_id="789",
            release_run_attempt="1",
            release_actor="someone-else",
            release_triggering_actor="someone-else",
        )


def test_governance_record_rejects_a_different_rerun_triggering_actor():
    policy_bytes, snapshot = _evidence()
    with pytest.raises(ValueError, match="triggering actor"):
        governance.build_record(
            policy_bytes=policy_bytes,
            snapshot_bytes=json.dumps(snapshot).encode(),
            repository="jemsbhai/explainiverse",
            release_tag="v0.15.0",
            release_commit=SHA,
            preflight_run_id="123",
            cuda_run_id="456",
            release_run_id="789",
            release_run_attempt="2",
            release_actor="jemsbhai",
            release_triggering_actor="collaborator",
        )
