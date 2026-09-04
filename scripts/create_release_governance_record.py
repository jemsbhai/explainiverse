"""Create a fail-closed single-operator release governance disclosure.

The record does not claim segregation of duties.  It binds the current policy,
the attested external-control snapshot, the CUDA release gate, and the release
workflow to the public disclosure required while Explainiverse has one release
operator.  The gate is either verified hardware evidence or the exact reviewed
CPU-only exception for ``v0.15.2``; those states are never interchangeable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_SHA = re.compile(r"[0-9a-f]{40}")
_TAG = re.compile(r"v\d+\.\d+\.\d+")
_RUN_ID = re.compile(r"[1-9][0-9]*")
_SENTINEL = "<!-- explainiverse-release-governance-v1 -->"
_HARDWARE_MODE = "hardware_evidence"
_EXCEPTION_MODE = "cpu_only_exception"
_CUDA_EXCEPTION_ID = "EXPLAINIVERSE-v0.15.2-CPU-ONLY"
_CUDA_EXCEPTION_TAG = "v0.15.2"
_CUDA_EXCEPTION_VERSION = "0.15.2"
_CUDA_EXCEPTION_PULL_REQUEST = 7
_CUDA_EXCEPTION_APPROVED_AT = "2026-09-04"
_CUDA_EXCEPTION_AUTHORIZED_BY = ["jemsbhai"]
_CUDA_EXCEPTION_OMITTED_CHECKS = [
    "CUDA single-GPU (Torch latest)",
    "CUDA single-GPU (Torch minimum)",
]
_CUDA_EXCEPTION_OMITTED_JOBS = [
    "CUDA single-GPU (Torch latest)",
    "CUDA single-GPU (Torch minimum)",
    "CUDA two-GPU scheduled (Torch latest)",
    "CUDA two-GPU scheduled (Torch minimum)",
]
_CUDA_EXCEPTION_REASON = (
    "Approved one-release CPU-only roll-forward because isolated one- and two-GPU "
    "release runners remain unavailable and the immutable v0.15.0 and v0.15.1 release "
    "attempts both stopped before publication."
)
_CUDA_EXCEPTION_DISCLOSURE = (
    "Explainiverse 0.15.2 is CPU-verified; CUDA hardware validation was not performed "
    "and this release makes no CUDA release-verification claim. The signed v0.15.0 and "
    "v0.15.1 Git tags remain immutable; neither version is on PyPI or has a GitHub "
    "Release. Workflow run 33891048942 for v0.15.0 failed during SBOM generation before "
    "artifact upload, attestation, PyPI publication, or GitHub Release creation. Workflow "
    "run 33901507340 for v0.15.1 successfully built and retained workflow artifacts, "
    "including the repaired SBOM, but GitHub skipped distribution attestation, PyPI "
    "publication, and GitHub Release creation because a skipped ancestor condition "
    "propagated to those jobs."
)
_HARDWARE_GATE_FIELDS = {
    "schema_version",
    "mode",
    "status",
    "exception_id",
    "release_tag",
    "release_commit",
    "hardware_evidence_collected",
    "cuda_release_verified",
    "cuda_run_id",
}
_EXCEPTION_POLICY_FIELDS = {
    "id",
    "release_tag",
    "package_version",
    "merge_pull_request",
    "omitted_required_checks",
    "omitted_cuda_jobs",
    "hardware_evidence_collected",
    "cuda_release_verified",
    "authorized_by",
    "approved_at",
    "reason",
    "disclosure",
}
_EXCEPTION_GATE_FIELDS = {
    "schema_version",
    "mode",
    "status",
    "exception_id",
    "release_tag",
    "release_commit",
    "package_version",
    "merge_pull_request",
    "merge_commit_sha",
    "hardware_evidence_collected",
    "cuda_release_verified",
    "omitted_required_checks",
    "omitted_cuda_jobs",
    "authorized_by",
    "approved_at",
    "reason",
    "disclosure",
}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _strict_equal(actual: Any, expected: Any) -> bool:
    """Compare JSON-shaped values without Python's bool/int or int/float coercion."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _strict_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_equal(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected)
        )
    return bool(actual == expected)


def _run_id(value: str, name: str) -> str:
    if _RUN_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validated_cuda_release_gate(
    *,
    policy: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    release_tag: str,
    release_commit: str,
    cuda_run_id: str | None,
    cuda_exception_id: str | None,
) -> Mapping[str, Any]:
    """Return the exact attested hardware/exception union after strict validation."""
    if (cuda_run_id is None) == (cuda_exception_id is None):
        raise ValueError("exactly one of CUDA run id or CUDA exception id is required")

    gate = _mapping(snapshot.get("cuda_release_gate"), "snapshot CUDA release gate")
    schema_version = gate.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError("snapshot CUDA release gate schema_version must be the integer 1")
    if gate.get("release_tag") != release_tag:
        raise ValueError("snapshot CUDA release gate tag mismatch")
    if gate.get("release_commit") != release_commit:
        raise ValueError("snapshot CUDA release gate commit mismatch")

    mode = gate.get("mode")
    if mode == _HARDWARE_MODE:
        if set(gate) != _HARDWARE_GATE_FIELDS:
            raise ValueError("snapshot hardware CUDA release gate fields differ from schema")
        if cuda_run_id is None or cuda_exception_id is not None:
            raise ValueError("hardware CUDA release gate requires only a CUDA run id")
        normalized_run_id = _run_id(cuda_run_id, "CUDA run id")
        expected = {
            "status": "verified",
            "exception_id": None,
            "hardware_evidence_collected": True,
            "cuda_release_verified": True,
            "cuda_run_id": normalized_run_id,
        }
        for field, expected_value in expected.items():
            if not _strict_equal(gate.get(field), expected_value):
                raise ValueError(
                    f"snapshot hardware CUDA release gate {field} mismatch: "
                    f"expected {expected_value!r}, got {gate.get(field)!r}"
                )
        cuda_evidence = _mapping(snapshot.get("cuda_evidence"), "snapshot CUDA evidence")
        cuda_run = _mapping(cuda_evidence.get("run"), "snapshot CUDA run")
        if str(cuda_run.get("id")) != normalized_run_id:
            raise ValueError("snapshot CUDA run id mismatch")
        if cuda_run.get("head_sha") != release_commit:
            raise ValueError("snapshot CUDA run commit mismatch")
        return dict(gate)

    if mode != _EXCEPTION_MODE:
        raise ValueError(f"unsupported snapshot CUDA release gate mode: {mode!r}")
    if set(gate) != _EXCEPTION_GATE_FIELDS:
        raise ValueError("snapshot CPU-only CUDA release gate fields differ from schema")
    if cuda_exception_id is None or cuda_run_id is not None:
        raise ValueError("CPU-only CUDA release gate requires only a CUDA exception id")
    if "cuda_evidence" in snapshot:
        raise ValueError("CPU-only CUDA release gate must not contain CUDA hardware evidence")

    exception = _mapping(policy.get("cuda_release_exception"), "CUDA release exception policy")
    expected_exception = {
        "id": _CUDA_EXCEPTION_ID,
        "release_tag": _CUDA_EXCEPTION_TAG,
        "package_version": _CUDA_EXCEPTION_VERSION,
        "merge_pull_request": _CUDA_EXCEPTION_PULL_REQUEST,
        "omitted_required_checks": _CUDA_EXCEPTION_OMITTED_CHECKS,
        "omitted_cuda_jobs": _CUDA_EXCEPTION_OMITTED_JOBS,
        "hardware_evidence_collected": False,
        "cuda_release_verified": False,
        "authorized_by": _CUDA_EXCEPTION_AUTHORIZED_BY,
        "approved_at": _CUDA_EXCEPTION_APPROVED_AT,
        "reason": _CUDA_EXCEPTION_REASON,
        "disclosure": _CUDA_EXCEPTION_DISCLOSURE,
    }
    if set(exception) != _EXCEPTION_POLICY_FIELDS or not _strict_equal(
        dict(exception), expected_exception
    ):
        raise ValueError("CUDA release exception policy differs from the reviewed exception")
    snapshot_exception = _mapping(
        snapshot.get("cuda_release_exception"), "snapshot CUDA release exception"
    )
    if not _strict_equal(dict(snapshot_exception), expected_exception):
        raise ValueError("snapshot CUDA release exception differs from reviewed policy")
    if cuda_exception_id != _CUDA_EXCEPTION_ID:
        raise ValueError("CUDA exception id differs from reviewed policy")
    if release_tag != _CUDA_EXCEPTION_TAG:
        raise ValueError("CUDA release exception policy tag mismatch")
    if release_tag.removeprefix("v") != _CUDA_EXCEPTION_VERSION:
        raise ValueError("CUDA release exception package version mismatch")

    observation = _mapping(snapshot.get("observation"), "snapshot observation")
    merge_pull_request = _mapping(
        observation.get("cuda_exception_merge_pull_request"),
        "snapshot CUDA exception merge pull request",
    )
    expected_pull_request_fields = {
        "number": _CUDA_EXCEPTION_PULL_REQUEST,
        "state": "closed",
        "merged": True,
        "merge_commit_sha": release_commit,
        "base_ref": "main",
        "base_repository": policy.get("repository"),
        "head_repository": policy.get("repository"),
        "merged_by": _CUDA_EXCEPTION_AUTHORIZED_BY[0],
    }
    for field, expected_pull_value in expected_pull_request_fields.items():
        if not _strict_equal(merge_pull_request.get(field), expected_pull_value):
            raise ValueError(
                f"snapshot CUDA exception merge pull request {field} mismatch: "
                f"expected {expected_pull_value!r}, got {merge_pull_request.get(field)!r}"
            )

    expected_gate = {
        "schema_version": 1,
        "mode": _EXCEPTION_MODE,
        "status": "not_run",
        "exception_id": _CUDA_EXCEPTION_ID,
        "release_tag": release_tag,
        "release_commit": release_commit,
        "package_version": _CUDA_EXCEPTION_VERSION,
        "merge_pull_request": _CUDA_EXCEPTION_PULL_REQUEST,
        "merge_commit_sha": release_commit,
        "hardware_evidence_collected": False,
        "cuda_release_verified": False,
        "omitted_required_checks": _CUDA_EXCEPTION_OMITTED_CHECKS,
        "omitted_cuda_jobs": _CUDA_EXCEPTION_OMITTED_JOBS,
        "authorized_by": _CUDA_EXCEPTION_AUTHORIZED_BY,
        "approved_at": _CUDA_EXCEPTION_APPROVED_AT,
        "reason": _CUDA_EXCEPTION_REASON,
        "disclosure": _CUDA_EXCEPTION_DISCLOSURE,
    }
    if not _strict_equal(dict(gate), expected_gate):
        raise ValueError("snapshot CPU-only CUDA release gate differs from reviewed policy")
    return expected_gate


def build_record(
    *,
    policy_bytes: bytes,
    snapshot_bytes: bytes,
    repository: str,
    release_tag: str,
    release_commit: str,
    preflight_run_id: str,
    cuda_run_id: str | None = None,
    cuda_exception_id: str | None = None,
    release_run_id: str,
    release_run_attempt: str,
    release_actor: str,
    release_triggering_actor: str,
) -> Mapping[str, Any]:
    """Validate the bound evidence and return the truthful governance record."""
    if _TAG.fullmatch(release_tag) is None:
        raise ValueError("release tag must have the form vMAJOR.MINOR.PATCH")
    if _SHA.fullmatch(release_commit) is None:
        raise ValueError("release commit must be a complete lowercase SHA")
    preflight_run_id = _run_id(preflight_run_id, "preflight run id")
    release_run_id = _run_id(release_run_id, "release run id")
    release_run_attempt = _run_id(release_run_attempt, "release run attempt")
    if not isinstance(release_actor, str) or not release_actor:
        raise ValueError("release actor must be non-empty")
    if release_triggering_actor != release_actor:
        raise ValueError(
            "single-operator disclosure requires release triggering actor to match release actor"
        )

    policy = _mapping(json.loads(policy_bytes), "release policy")
    snapshot = _mapping(json.loads(snapshot_bytes), "external-control snapshot")
    if (
        type(policy.get("schema_version")) is not int
        or policy.get("schema_version") != 1
        or type(snapshot.get("schema_version")) is not int
        or snapshot.get("schema_version") != 1
    ):
        raise ValueError("policy and external-control snapshot schema_version must be 1")
    policy_sha256 = hashlib.sha256(policy_bytes).hexdigest()
    if snapshot.get("policy_sha256") != policy_sha256:
        raise ValueError("external-control snapshot policy digest mismatch")
    if snapshot.get("repository_controls_accepted") is not True or snapshot.get("violations") != []:
        raise ValueError("external-control snapshot is not an accepted zero-violation capture")

    observation = _mapping(snapshot.get("observation"), "snapshot observation")
    expected_observation = {
        "repository": repository,
        "release_tag": release_tag,
        "release_commit": release_commit,
    }
    for field, expected in expected_observation.items():
        if observation.get(field) != expected:
            raise ValueError(
                f"snapshot {field} mismatch: expected {expected!r}, "
                f"got {observation.get(field)!r}"
            )
    if policy.get("repository") != repository:
        raise ValueError("release repository differs from reviewed policy")

    capture_principal = observation.get("capture_principal")
    workflow_run = _mapping(snapshot.get("workflow_run"), "preflight workflow run")
    if not _strict_equal(workflow_run.get("id"), preflight_run_id):
        raise ValueError("snapshot preflight run id mismatch")
    if workflow_run.get("actor") != capture_principal:
        raise ValueError("preflight actor differs from authenticated capture principal")
    if release_actor != capture_principal:
        raise ValueError("single-operator disclosure requires release actor to match capture actor")

    cuda_release_gate = _validated_cuda_release_gate(
        policy=policy,
        snapshot=snapshot,
        release_tag=release_tag,
        release_commit=release_commit,
        cuda_run_id=cuda_run_id,
        cuda_exception_id=cuda_exception_id,
    )

    environment = _mapping(policy.get("pypi_environment"), "PyPI environment policy")
    reviewers = environment.get("required_reviewer_logins")
    if reviewers != [capture_principal]:
        raise ValueError(
            "single-operator record requires the sole reviewed environment reviewer to "
            "match the capture principal"
        )
    if environment.get("prevent_self_review") is not False:
        raise ValueError("policy no longer describes the reviewed single-operator mode")
    observed_environment = _mapping(
        observation.get("pypi_environment"), "snapshot PyPI environment"
    )
    if observed_environment.get("name") != environment.get("name"):
        raise ValueError("snapshot PyPI environment name differs from policy")
    if observed_environment.get("can_admins_bypass") is not environment.get("can_admins_bypass"):
        raise ValueError("snapshot PyPI environment administrator bypass differs from policy")
    protection_rules = observed_environment.get("protection_rules")
    if not isinstance(protection_rules, list):
        raise ValueError("snapshot PyPI environment protection rules must be an array")
    reviewer_rules = [
        _mapping(value, "snapshot PyPI reviewer rule")
        for value in protection_rules
        if isinstance(value, Mapping) and value.get("type") == "required_reviewers"
    ]
    if len(reviewer_rules) != 1:
        raise ValueError("snapshot must contain exactly one PyPI required-reviewers rule")
    reviewer_rule = reviewer_rules[0]
    observed_reviewers = []
    for value in reviewer_rule.get("reviewers", []):
        reviewer = _mapping(value, "snapshot PyPI reviewer")
        identity = _mapping(reviewer.get("reviewer"), "snapshot PyPI reviewer identity")
        observed_reviewers.append(identity.get("login"))
    if observed_reviewers != reviewers:
        raise ValueError("snapshot PyPI environment reviewers differ from policy")
    if reviewer_rule.get("prevent_self_review") is not False:
        raise ValueError("snapshot PyPI environment self-review setting differs from policy")

    controls_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
    base_url = f"https://github.com/{repository}/actions/runs"
    evidence = {
        "observed_at": snapshot.get("observed_at"),
        "policy_sha256": policy_sha256,
        "external_controls_sha256": controls_sha256,
        "preflight_run_id": preflight_run_id,
        "preflight_run_url": f"{base_url}/{preflight_run_id}",
        "release_workflow_run_id": release_run_id,
        "release_workflow_run_url": f"{base_url}/{release_run_id}",
    }
    if cuda_release_gate["mode"] == _HARDWARE_MODE:
        if cuda_run_id is None:  # Defensive narrowing after the validated union.
            raise ValueError("hardware CUDA release gate has no CUDA run id")
        normalized_cuda_run_id = _run_id(cuda_run_id, "CUDA run id")
        evidence.update(
            {
                "cuda_run_id": normalized_cuda_run_id,
                "cuda_run_url": f"{base_url}/{normalized_cuda_run_id}",
            }
        )

    return {
        "schema_version": 1,
        "release": {
            "repository": repository,
            "tag": release_tag,
            "commit": release_commit,
        },
        "governance": {
            "mode": "single_operator_disclosed",
            "segregation_of_duties": False,
            "capture_principal": capture_principal,
            "release_dispatch_actor": release_actor,
            "release_triggering_actor": release_triggering_actor,
            "release_run_attempt": release_run_attempt,
            "environment_reviewer_logins": observed_reviewers,
            "prevent_self_review": reviewer_rule.get("prevent_self_review"),
            "cuda_release_mode": cuda_release_gate["mode"],
        },
        "cuda_release_gate": dict(cuda_release_gate),
        "evidence": evidence,
    }


def render_markdown(record: Mapping[str, Any]) -> str:
    release = _mapping(record.get("release"), "record release")
    governance = _mapping(record.get("governance"), "record governance")
    evidence = _mapping(record.get("evidence"), "record evidence")
    gate = _mapping(record.get("cuda_release_gate"), "record CUDA release gate")
    if gate.get("mode") == _EXCEPTION_MODE:
        expected_exception_gate = {
            "schema_version": 1,
            "mode": _EXCEPTION_MODE,
            "status": "not_run",
            "exception_id": _CUDA_EXCEPTION_ID,
            "release_tag": _CUDA_EXCEPTION_TAG,
            "release_commit": release.get("commit"),
            "package_version": _CUDA_EXCEPTION_VERSION,
            "merge_pull_request": _CUDA_EXCEPTION_PULL_REQUEST,
            "merge_commit_sha": release.get("commit"),
            "hardware_evidence_collected": False,
            "cuda_release_verified": False,
            "omitted_required_checks": _CUDA_EXCEPTION_OMITTED_CHECKS,
            "omitted_cuda_jobs": _CUDA_EXCEPTION_OMITTED_JOBS,
            "authorized_by": _CUDA_EXCEPTION_AUTHORIZED_BY,
            "approved_at": _CUDA_EXCEPTION_APPROVED_AT,
            "reason": _CUDA_EXCEPTION_REASON,
            "disclosure": _CUDA_EXCEPTION_DISCLOSURE,
        }
        if release.get("tag") != _CUDA_EXCEPTION_TAG or not _strict_equal(
            dict(gate), expected_exception_gate
        ):
            raise ValueError(
                "record CPU-only CUDA release gate differs from the reviewed exception"
            )
    lines = [
        _SENTINEL,
        "# Release governance disclosure",
        "",
        "This release used the documented single-operator approval path. It does not "
        "claim segregation of duties or independent approval.",
        "",
        f"- Repository: `{release['repository']}`",
        f"- Tag: `{release['tag']}`",
        f"- Commit: `{release['commit']}`",
        f"- Operator: `{governance['capture_principal']}`",
        f"- Release triggering actor: `{governance['release_triggering_actor']}`",
        f"- Release run attempt: `{governance['release_run_attempt']}`",
        f"- External-control snapshot SHA-256: `{evidence['external_controls_sha256']}`",
        f"- External controls observed at: `{evidence['observed_at']}`",
        f"- Preflight run: {evidence['preflight_run_url']}",
        f"- Release workflow run: {evidence['release_workflow_run_url']}",
        "",
    ]
    if gate.get("mode") == _HARDWARE_MODE:
        lines.extend(
            [
                "## CUDA release gate",
                "",
                "The release used verified CUDA hardware evidence.",
                "",
                f"- CUDA evidence run: {evidence['cuda_run_url']}",
                "- CUDA hardware evidence collected: `true`",
                "- CUDA release verification: `true`",
                "",
            ]
        )
    elif gate.get("mode") == _EXCEPTION_MODE:
        omitted_checks = ", ".join(f"`{name}`" for name in gate["omitted_required_checks"])
        omitted_jobs = ", ".join(f"`{name}`" for name in gate["omitted_cuda_jobs"])
        authorized_by = ", ".join(f"`{name}`" for name in gate["authorized_by"])
        lines.extend(
            [
                "## CPU-only CUDA release exception",
                "",
                _CUDA_EXCEPTION_DISCLOSURE,
                "",
                "No CUDA hardware evidence was collected, the CUDA release jobs were not run, "
                "and this release does not claim CUDA release verification.",
                "",
                f"- Exception ID: `{gate['exception_id']}`",
                f"- Approved at: `{gate['approved_at']}`",
                f"- Authorized by: {authorized_by}",
                f"- Merge pull request: `#{gate['merge_pull_request']}`",
                f"- Verified PR merge commit: `{gate['merge_commit_sha']}`",
                f"- Omitted required checks: {omitted_checks}",
                f"- Omitted CUDA jobs: {omitted_jobs}",
                f"- Reason: {gate['reason']}",
                "- CUDA hardware evidence collected: `false`",
                "- CUDA release verification: `false`",
                "",
            ]
        )
    else:
        raise ValueError(f"unsupported record CUDA release gate mode: {gate.get('mode')!r}")
    lines.extend(
        [
            "The attached `external-controls.json` and related preflight files are the "
            "content-addressed control record for this disclosure.",
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--preflight-run-id", required=True)
    cuda_gate = parser.add_mutually_exclusive_group(required=True)
    cuda_gate.add_argument("--cuda-run-id")
    cuda_gate.add_argument("--cuda-exception-id")
    parser.add_argument("--release-run-id", required=True)
    parser.add_argument("--release-run-attempt", required=True)
    parser.add_argument("--release-actor", required=True)
    parser.add_argument("--release-triggering-actor", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        record = build_record(
            policy_bytes=args.policy.read_bytes(),
            snapshot_bytes=args.snapshot.read_bytes(),
            repository=args.repository,
            release_tag=args.tag,
            release_commit=args.commit,
            preflight_run_id=args.preflight_run_id,
            cuda_run_id=args.cuda_run_id,
            cuda_exception_id=args.cuda_exception_id,
            release_run_id=args.release_run_id,
            release_run_attempt=args.release_run_attempt,
            release_actor=args.release_actor,
            release_triggering_actor=args.release_triggering_actor,
        )
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        args.output_markdown.write_text(render_markdown(record), encoding="utf-8")
        return 0
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
