"""Create a fail-closed single-operator release governance disclosure.

The record does not claim segregation of duties.  It binds the current policy,
the attested external-control snapshot, the CUDA run, and the release workflow
to the public disclosure required while Explainiverse has one release operator.
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


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _run_id(value: str, name: str) -> str:
    if _RUN_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must be a positive integer")
    return value


def build_record(
    *,
    policy_bytes: bytes,
    snapshot_bytes: bytes,
    repository: str,
    release_tag: str,
    release_commit: str,
    preflight_run_id: str,
    cuda_run_id: str,
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
    cuda_run_id = _run_id(cuda_run_id, "CUDA run id")
    release_run_id = _run_id(release_run_id, "release run id")
    release_run_attempt = _run_id(release_run_attempt, "release run attempt")
    if not release_actor:
        raise ValueError("release actor must be non-empty")
    if release_triggering_actor != release_actor:
        raise ValueError(
            "single-operator disclosure requires release triggering actor to match release actor"
        )

    policy = _mapping(json.loads(policy_bytes), "release policy")
    snapshot = _mapping(json.loads(snapshot_bytes), "external-control snapshot")
    if policy.get("schema_version") != 1 or snapshot.get("schema_version") != 1:
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
    if str(workflow_run.get("id")) != preflight_run_id:
        raise ValueError("snapshot preflight run id mismatch")
    if workflow_run.get("actor") != capture_principal:
        raise ValueError("preflight actor differs from authenticated capture principal")
    if release_actor != capture_principal:
        raise ValueError("single-operator disclosure requires release actor to match capture actor")

    cuda_evidence = _mapping(snapshot.get("cuda_evidence"), "snapshot CUDA evidence")
    cuda_run = _mapping(cuda_evidence.get("run"), "snapshot CUDA run")
    if str(cuda_run.get("id")) != cuda_run_id:
        raise ValueError("snapshot CUDA run id mismatch")
    if cuda_run.get("head_sha") != release_commit:
        raise ValueError("snapshot CUDA run commit mismatch")

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
        },
        "evidence": {
            "observed_at": snapshot.get("observed_at"),
            "policy_sha256": policy_sha256,
            "external_controls_sha256": controls_sha256,
            "preflight_run_id": preflight_run_id,
            "preflight_run_url": f"{base_url}/{preflight_run_id}",
            "cuda_run_id": cuda_run_id,
            "cuda_run_url": f"{base_url}/{cuda_run_id}",
            "release_workflow_run_id": release_run_id,
            "release_workflow_run_url": f"{base_url}/{release_run_id}",
        },
    }


def render_markdown(record: Mapping[str, Any]) -> str:
    release = _mapping(record.get("release"), "record release")
    governance = _mapping(record.get("governance"), "record governance")
    evidence = _mapping(record.get("evidence"), "record evidence")
    return "\n".join(
        [
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
            f"- CUDA evidence run: {evidence['cuda_run_url']}",
            f"- Release workflow run: {evidence['release_workflow_run_url']}",
            "",
            "The attached `external-controls.json` and related preflight files are the "
            "content-addressed control record for this disclosure.",
            "",
        ]
    )


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
    parser.add_argument("--cuda-run-id", required=True)
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
