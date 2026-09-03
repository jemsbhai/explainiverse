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


def _evidence(*, cpu_only=False):
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
    }
    if cpu_only:
        exception = policy["cuda_release_exception"]
        snapshot["cuda_release_exception"] = exception
        snapshot["cuda_release_gate"] = {
            "schema_version": 1,
            "mode": "cpu_only_exception",
            "status": "not_run",
            "exception_id": exception["id"],
            "release_tag": exception["release_tag"],
            "release_commit": SHA,
            "package_version": exception["package_version"],
            "merge_pull_request": exception["merge_pull_request"],
            "hardware_evidence_collected": False,
            "cuda_release_verified": False,
            "omitted_required_checks": exception["omitted_required_checks"],
            "omitted_cuda_jobs": exception["omitted_cuda_jobs"],
            "authorized_by": exception["authorized_by"],
            "approved_at": exception["approved_at"],
            "reason": exception["reason"],
            "disclosure": exception["disclosure"],
        }
    else:
        snapshot["cuda_release_gate"] = {
            "schema_version": 1,
            "mode": "hardware_evidence",
            "status": "verified",
            "exception_id": None,
            "release_tag": "v0.15.0",
            "release_commit": SHA,
            "hardware_evidence_collected": True,
            "cuda_release_verified": True,
            "cuda_run_id": "456",
        }
        snapshot["cuda_evidence"] = {"run": {"id": 456, "head_sha": SHA}}
    return policy_bytes, snapshot


def _record(policy_bytes, snapshot):
    gate = snapshot["cuda_release_gate"]
    cuda_args = (
        {"cuda_run_id": "456", "cuda_exception_id": None}
        if gate["mode"] == "hardware_evidence"
        else {"cuda_run_id": None, "cuda_exception_id": gate["exception_id"]}
    )
    return governance.build_record(
        policy_bytes=policy_bytes,
        snapshot_bytes=(json.dumps(snapshot, sort_keys=True) + "\n").encode(),
        repository="jemsbhai/explainiverse",
        release_tag="v0.15.0",
        release_commit=SHA,
        preflight_run_id="123",
        **cuda_args,
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
        "cuda_release_mode": "hardware_evidence",
    }
    assert record["cuda_release_gate"] == snapshot["cuda_release_gate"]
    markdown = governance.render_markdown(record)
    assert markdown.startswith("<!-- explainiverse-release-governance-v1 -->")
    assert "does not claim segregation of duties" in markdown
    assert record["evidence"]["external_controls_sha256"] in markdown
    assert "/actions/runs/123" in markdown and "/actions/runs/456" in markdown
    assert "verified CUDA hardware evidence" in markdown


def test_cpu_only_record_discloses_the_exact_exception_without_cuda_evidence():
    policy_bytes, snapshot = _evidence(cpu_only=True)
    record = _record(policy_bytes, snapshot)

    assert record["governance"]["cuda_release_mode"] == "cpu_only_exception"
    assert record["cuda_release_gate"] == snapshot["cuda_release_gate"]
    assert "cuda_run_id" not in record["evidence"]
    assert "cuda_run_url" not in record["evidence"]
    markdown = governance.render_markdown(record)
    assert "CPU-only CUDA release exception" in markdown
    assert "EXPLAINIVERSE-v0.15.0-CPU-ONLY" in markdown
    assert "No CUDA hardware evidence was collected" in markdown
    assert "CUDA release verification: `false`" in markdown
    assert snapshot["cuda_release_gate"]["disclosure"] in markdown
    assert "CUDA evidence run:" not in markdown


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
            lambda _policy, snapshot: snapshot["cuda_release_gate"].update(
                cuda_release_verified=False
            ),
            "cuda_release_verified mismatch",
        ),
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
            cuda_exception_id=None,
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
            cuda_exception_id=None,
            release_run_id="789",
            release_run_attempt="2",
            release_actor="jemsbhai",
            release_triggering_actor="collaborator",
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda policy, snapshot: snapshot["cuda_release_gate"].update(
                exception_id=policy["cuda_release_exception"]["id"] + "-forged"
            ),
            "differs from reviewed policy",
        ),
        (
            lambda _policy, snapshot: snapshot["cuda_release_gate"].update(
                hardware_evidence_collected=True
            ),
            "differs from reviewed policy",
        ),
        (
            lambda _policy, snapshot: snapshot.update(
                cuda_evidence={"run": {"id": 456, "head_sha": SHA}}
            ),
            "must not contain CUDA hardware evidence",
        ),
        (
            lambda policy, _snapshot: policy["cuda_release_exception"].update(
                cuda_release_verified=True
            ),
            "differs from reviewed policy",
        ),
    ],
)
def test_cpu_only_governance_record_rejects_forged_or_mixed_gate(mutation, match):
    policy_bytes, snapshot = _evidence(cpu_only=True)
    policy = json.loads(policy_bytes)
    mutation(policy, snapshot)
    policy_bytes = (json.dumps(policy, indent=2) + "\n").encode()
    snapshot["policy_sha256"] = hashlib.sha256(policy_bytes).hexdigest()
    with pytest.raises(ValueError, match=match):
        _record(policy_bytes, snapshot)


def test_governance_record_requires_exactly_one_cuda_gate_credential():
    policy_bytes, snapshot = _evidence(cpu_only=True)
    common = {
        "policy_bytes": policy_bytes,
        "snapshot_bytes": (json.dumps(snapshot, sort_keys=True) + "\n").encode(),
        "repository": "jemsbhai/explainiverse",
        "release_tag": "v0.15.0",
        "release_commit": SHA,
        "preflight_run_id": "123",
        "release_run_id": "789",
        "release_run_attempt": "1",
        "release_actor": "jemsbhai",
        "release_triggering_actor": "jemsbhai",
    }
    with pytest.raises(ValueError, match="exactly one"):
        governance.build_record(**common, cuda_run_id=None, cuda_exception_id=None)
    with pytest.raises(ValueError, match="exactly one"):
        governance.build_record(
            **common,
            cuda_run_id="456",
            cuda_exception_id="EXPLAINIVERSE-v0.15.0-CPU-ONLY",
        )
