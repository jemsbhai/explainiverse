"""Validate manual assistive-technology evidence against the checked-in policy."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse


class AccessibilityEvidenceError(ValueError):
    """Raised when manual certification evidence is incomplete or ambiguous."""


def _load_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        if path.stat().st_size > 1024 * 1024:
            raise AccessibilityEvidenceError(f"{label} exceeds the 1 MiB manifest limit")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AccessibilityEvidenceError(f"cannot load {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AccessibilityEvidenceError(f"{label} must be a JSON object")
    return value


def _required_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AccessibilityEvidenceError(f"{field} must be a non-empty string")
    return value.strip()


def _https_uri(value: object, *, field: str) -> str:
    uri = _required_string(value, field=field)
    parsed = urlparse(uri)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise AccessibilityEvidenceError(f"{field} must be an HTTPS URI without credentials")
    return uri


def _timestamp(value: object, *, field: str) -> datetime:
    raw = _required_string(value, field=field)
    try:
        parsed = datetime.fromisoformat(raw[:-1] + "+00:00" if raw.endswith("Z") else raw)
    except ValueError as exc:
        raise AccessibilityEvidenceError(f"{field} must be an RFC 3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise AccessibilityEvidenceError(f"{field} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _string_list(policy: dict[str, object], field: str) -> list[str]:
    values = policy.get(field)
    if not isinstance(values, list) or not values:
        raise AccessibilityEvidenceError(f"policy {field} must be a non-empty array")
    strings = [_required_string(value, field=f"policy.{field}") for value in values]
    if len(strings) != len(set(strings)):
        raise AccessibilityEvidenceError(f"policy {field} must be unique")
    return strings


def validate_evidence(
    policy_path: Path,
    evidence_path: Path,
    *,
    expected_commit: str,
    now: datetime | None = None,
) -> dict[str, object]:
    """Return a summary only when every required profile and artifact passes."""
    policy = _load_object(policy_path, label="policy")
    evidence = _load_object(evidence_path, label="evidence")
    if policy.get("schema_version") != 1 or evidence.get("schema_version") != 1:
        raise AccessibilityEvidenceError("policy and evidence must use schema version 1")
    if policy.get("claim_status") != "blocked_pending_manual_evidence":
        raise AccessibilityEvidenceError("policy must remain blocked until reviewed evidence lands")

    revision = _required_string(evidence.get("commit_sha"), field="commit_sha")
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise AccessibilityEvidenceError("commit_sha must be a full lowercase Git commit SHA")
    expected_revision = _required_string(expected_commit, field="expected_commit")
    if re.fullmatch(r"[0-9a-f]{40}", expected_revision) is None:
        raise AccessibilityEvidenceError("expected_commit must be a full lowercase Git commit SHA")
    if revision != expected_revision:
        raise AccessibilityEvidenceError(
            f"evidence commit_sha {revision} does not match checked-out commit {expected_revision}"
        )
    demo_url = _https_uri(evidence.get("demo_url"), field="demo_url")
    deployment_provenance_uri = _https_uri(
        evidence.get("deployment_provenance_uri"),
        field="deployment_provenance_uri",
    )
    demo_build_sha256 = _required_string(
        evidence.get("demo_build_sha256"),
        field="demo_build_sha256",
    )
    if re.fullmatch(r"[0-9a-f]{64}", demo_build_sha256) is None:
        raise AccessibilityEvidenceError("demo_build_sha256 must be lowercase SHA-256")
    reviewer = _required_string(evidence.get("independent_reviewer"), field="independent_reviewer")
    if evidence.get("reviewer_independent_from_implementation") is not True:
        raise AccessibilityEvidenceError(
            "the named reviewer must be independent from implementation"
        )

    age_days = policy.get("max_evidence_age_days")
    if not isinstance(age_days, int) or age_days <= 0:
        raise AccessibilityEvidenceError("policy max_evidence_age_days must be a positive integer")
    reference_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    evidence_time = _timestamp(evidence.get("completed_at"), field="completed_at")
    if evidence_time > reference_time + timedelta(minutes=5):
        raise AccessibilityEvidenceError("completed_at cannot be in the future")
    if reference_time - evidence_time > timedelta(days=age_days):
        raise AccessibilityEvidenceError("manual accessibility evidence is stale")

    profiles_value = policy.get("required_profiles")
    if not isinstance(profiles_value, list) or not profiles_value:
        raise AccessibilityEvidenceError("policy required_profiles must be a non-empty array")
    required_profiles: set[str] = set()
    for profile in profiles_value:
        if not isinstance(profile, dict):
            raise AccessibilityEvidenceError("each required profile must be an object")
        required_profiles.add(_required_string(profile.get("id"), field="profile.id"))
    if len(required_profiles) != len(profiles_value):
        raise AccessibilityEvidenceError("required profile IDs must be unique")

    required_scenarios = set(_string_list(policy, "required_scenarios"))
    required_artifacts = set(_string_list(policy, "required_artifact_kinds"))
    runs = evidence.get("runs")
    if not isinstance(runs, list):
        raise AccessibilityEvidenceError("runs must be an array")
    seen_profiles: set[str] = set()
    for index, run in enumerate(runs):
        prefix = f"runs[{index}]"
        if not isinstance(run, dict):
            raise AccessibilityEvidenceError(f"{prefix} must be an object")
        profile_id = _required_string(run.get("profile_id"), field=f"{prefix}.profile_id")
        if profile_id in seen_profiles:
            raise AccessibilityEvidenceError(f"duplicate run for profile {profile_id}")
        seen_profiles.add(profile_id)
        for field in ("os_version", "browser_version", "assistive_technology_version"):
            _required_string(run.get(field), field=f"{prefix}.{field}")
        if run.get("result") != "pass":
            raise AccessibilityEvidenceError(f"{prefix}.result must be 'pass'")

        scenarios = run.get("scenarios")
        if not isinstance(scenarios, list):
            raise AccessibilityEvidenceError(f"{prefix}.scenarios must be an array")
        seen_scenarios: set[str] = set()
        for scenario in scenarios:
            if not isinstance(scenario, dict):
                raise AccessibilityEvidenceError(f"{prefix}.scenarios entries must be objects")
            scenario_id = _required_string(scenario.get("id"), field=f"{prefix}.scenarios.id")
            if scenario.get("result") != "pass":
                raise AccessibilityEvidenceError(
                    f"{prefix} scenario {scenario_id} must have result 'pass'"
                )
            _required_string(scenario.get("notes"), field=f"{prefix}.{scenario_id}.notes")
            if scenario_id in seen_scenarios:
                raise AccessibilityEvidenceError(f"duplicate scenario {scenario_id} in {prefix}")
            seen_scenarios.add(scenario_id)
        if seen_scenarios != required_scenarios:
            raise AccessibilityEvidenceError(
                f"{prefix} scenario set differs from policy: "
                f"missing={sorted(required_scenarios - seen_scenarios)!r}, "
                f"unknown={sorted(seen_scenarios - required_scenarios)!r}"
            )

        artifacts = run.get("artifacts")
        if not isinstance(artifacts, list):
            raise AccessibilityEvidenceError(f"{prefix}.artifacts must be an array")
        seen_artifacts: set[str] = set()
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                raise AccessibilityEvidenceError(f"{prefix}.artifacts entries must be objects")
            kind = _required_string(artifact.get("kind"), field=f"{prefix}.artifacts.kind")
            _https_uri(artifact.get("uri"), field=f"{prefix}.{kind}.uri")
            digest = _required_string(artifact.get("sha256"), field=f"{prefix}.{kind}.sha256")
            if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise AccessibilityEvidenceError(
                    f"{prefix}.{kind}.sha256 must be lowercase SHA-256"
                )
            if kind in seen_artifacts:
                raise AccessibilityEvidenceError(f"duplicate artifact kind {kind} in {prefix}")
            seen_artifacts.add(kind)
        if not required_artifacts <= seen_artifacts:
            raise AccessibilityEvidenceError(
                f"{prefix} is missing artifacts {sorted(required_artifacts - seen_artifacts)!r}"
            )

    if seen_profiles != required_profiles:
        raise AccessibilityEvidenceError(
            "run profile set differs from policy: "
            f"missing={sorted(required_profiles - seen_profiles)!r}, "
            f"unknown={sorted(seen_profiles - required_profiles)!r}"
        )

    return {
        "schema_version": 1,
        "commit_sha": revision,
        "demo_url": demo_url,
        "deployment_provenance_uri": deployment_provenance_uri,
        "demo_build_sha256": demo_build_sha256,
        "independent_reviewer": reviewer,
        "validated_profiles": sorted(seen_profiles),
        "validated_at": reference_time.isoformat(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument(
        "--expected-commit",
        required=True,
        help="Full commit SHA checked out by the validating workflow",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path(".github/accessibility-certification-policy.json"),
    )
    parser.add_argument("--normalized-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = validate_evidence(
            args.policy,
            args.evidence,
            expected_commit=args.expected_commit,
        )
        if args.normalized_output is not None:
            args.normalized_output.parent.mkdir(parents=True, exist_ok=True)
            args.normalized_output.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    except (OSError, AccessibilityEvidenceError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
