"""Verify a recovery-only GitHub Release against the original PyPI artifacts.

This script never uploads anything.  It validates the source workflow run, the
retained distribution artifact, PyPI's published SHA-256 records, and (after a
draft release is populated) the downloaded GitHub assets.  The corresponding
workflow therefore has no path that can invoke the PyPI upload job a second
time.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_TAG = re.compile(r"v(\d+)\.(\d+)\.(\d+)")
_DISTRIBUTION_SUFFIXES = (".whl", ".tar.gz")
_POST_PYPI_RELEASE_STEPS = (
    "Check out the immutable release tag for final verification",
    "Set up Python 3.12 for provenance verification",
    "Pin the provenance verifier installer",
    "Install the hash-locked provenance verifier",
    "Download attested distributions",
    "Download hashes and SBOM",
    "Verify release assets against reviewed hashes",
    "Create and verify a draft from the already-published signed tag",
    "Require and reverify the finalized immutable release",
    "Archive normal-path release verification evidence",
)
_HARDWARE_MODE = "hardware_evidence"
_EXCEPTION_MODE = "cpu_only_exception"
_CUDA_EXCEPTION_ID = "EXPLAINIVERSE-v0.15.0-CPU-ONLY"
_CUDA_EXCEPTION_TAG = "v0.15.0"
_CUDA_EXCEPTION_VERSION = "0.15.0"
_CUDA_EXCEPTION_PULL_REQUEST = 5
_CUDA_EXCEPTION_APPROVED_AT = "2026-09-03"
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
    "Approved one-release CPU-only exception because isolated one- and two-GPU "
    "release runners are unavailable."
)
_CUDA_EXCEPTION_DISCLOSURE = (
    "Explainiverse 0.15.0 is CPU-verified; CUDA hardware validation was not performed "
    "and this release makes no CUDA release-verification claim."
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


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return value


def parse_sha256sums(path: Path) -> dict[str, str]:
    """Parse the exact GNU sha256sum subset emitted by the build workflow."""
    hashes: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line or len(line) < 67 or line[64:66] not in {"  ", " *"}:
            raise ValueError(f"SHA256SUMS line {line_number} is malformed")
        digest = line[:64].lower()
        filename = line[66:]
        if _SHA256.fullmatch(digest) is None:
            raise ValueError(f"SHA256SUMS line {line_number} has an invalid digest")
        if (
            not filename
            or filename in {".", ".."}
            or Path(filename).name != filename
            or "/" in filename
            or "\\" in filename
        ):
            raise ValueError(f"SHA256SUMS line {line_number} has an unsafe filename")
        if not filename.endswith(_DISTRIBUTION_SUFFIXES):
            raise ValueError(f"SHA256SUMS contains non-distribution file {filename!r}")
        if filename in hashes:
            raise ValueError(f"SHA256SUMS contains duplicate filename {filename!r}")
        hashes[filename] = digest
    if not hashes:
        raise ValueError("SHA256SUMS must contain at least one distribution")
    if not any(name.endswith(".whl") for name in hashes):
        raise ValueError("SHA256SUMS must contain a wheel")
    if not any(name.endswith(".tar.gz") for name in hashes):
        raise ValueError("SHA256SUMS must contain a source distribution")
    return hashes


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_distribution_directory(directory: Path, expected: Mapping[str, str]) -> None:
    """Require the directory to contain exactly the inventoried distributions."""
    actual_names = sorted(
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.name.endswith(_DISTRIBUTION_SUFFIXES)
    )
    expected_names = sorted(expected)
    if actual_names != expected_names:
        raise ValueError(
            f"distribution inventory mismatch: expected {expected_names!r}, got {actual_names!r}"
        )
    for filename, expected_digest in expected.items():
        actual_digest = _file_sha256(directory / filename)
        if actual_digest != expected_digest:
            raise ValueError(
                f"SHA-256 mismatch for {filename!r}: expected {expected_digest}, "
                f"got {actual_digest}"
            )


def verify_release_asset_directory(
    directory: Path, *, expected_directories: Sequence[Path]
) -> None:
    """Require every GitHub asset to match the retained downstream inventory."""
    expected: dict[str, str] = {}
    for source_directory in expected_directories:
        for path in source_directory.iterdir():
            if not path.is_file():
                continue
            if path.name in expected:
                raise ValueError(f"retained artifacts contain duplicate asset name {path.name!r}")
            expected[path.name] = _file_sha256(path)
    actual_paths = [path for path in directory.iterdir() if path.is_file()]
    actual_names = sorted(path.name for path in actual_paths)
    if actual_names != sorted(expected):
        raise ValueError(
            f"GitHub release asset inventory mismatch: expected {sorted(expected)!r}, "
            f"got {actual_names!r}"
        )
    for path in actual_paths:
        digest = _file_sha256(path)
        if digest != expected[path.name]:
            raise ValueError(
                f"GitHub release asset SHA-256 mismatch for {path.name!r}: "
                f"expected {expected[path.name]}, got {digest}"
            )


def verify_pypi_json(
    metadata: Mapping[str, Any],
    *,
    project: str,
    version: str,
    expected: Mapping[str, str],
) -> None:
    """Require PyPI to expose precisely the retained release distributions."""
    info = _mapping(metadata.get("info"), "PyPI info")
    if str(info.get("name", "")).lower().replace("-", "_") != project.lower().replace("-", "_"):
        raise ValueError(f"PyPI project mismatch: expected {project!r}, got {info.get('name')!r}")
    if info.get("version") != version:
        raise ValueError(
            f"PyPI version mismatch: expected {version!r}, got {info.get('version')!r}"
        )
    published: dict[str, str] = {}
    for raw_file in _sequence(metadata.get("urls"), "PyPI urls"):
        file = _mapping(raw_file, "PyPI release file")
        filename = file.get("filename")
        digest = _mapping(file.get("digests"), "PyPI file digests").get("sha256")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("PyPI returned an unsafe or missing release filename")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest.lower()) is None:
            raise ValueError(f"PyPI returned an invalid SHA-256 for {filename!r}")
        if filename in published:
            raise ValueError(f"PyPI returned duplicate filename {filename!r}")
        published[filename] = digest.lower()
    if published != dict(expected):
        raise ValueError(
            f"PyPI artifact inventory/hash mismatch: expected {dict(expected)!r}, "
            f"got {published!r}"
        )


def verify_release_governance_disclosure(release: Mapping[str, Any], disclosure: str) -> None:
    """Require the REST release body to contain the exact retained disclosure."""
    if not disclosure:
        raise ValueError("governance disclosure must not be empty")
    body = release.get("body")
    if not isinstance(body, str):
        raise ValueError("GitHub release body must be a string")
    if disclosure not in body:
        raise ValueError("GitHub release omitted the exact governance disclosure")


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _verify_recovery_cuda_release_gate(
    record: Mapping[str, Any],
    *,
    repository: str,
    release_tag: str,
    release_commit: str,
    governance: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> None:
    """Preserve the exact hardware/CPU-only disclosure during recovery."""
    gate = _mapping(record.get("cuda_release_gate"), "governance record CUDA release gate")
    schema_version = gate.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError("governance record CUDA release gate schema_version must be the integer 1")
    mode = gate.get("mode")
    if governance.get("cuda_release_mode") != mode:
        raise ValueError("governance record CUDA release mode differs from its gate")
    if gate.get("release_tag") != release_tag:
        raise ValueError("governance record CUDA release gate tag mismatch")
    if gate.get("release_commit") != release_commit:
        raise ValueError("governance record CUDA release gate commit mismatch")

    if mode == _HARDWARE_MODE:
        if set(gate) != _HARDWARE_GATE_FIELDS:
            raise ValueError("governance record hardware CUDA release gate fields differ")
        if gate.get("status") != "verified":
            raise ValueError("governance record hardware CUDA release gate is not verified")
        if gate.get("exception_id") is not None:
            raise ValueError("governance record hardware CUDA release gate has an exception id")
        if gate.get("hardware_evidence_collected") is not True:
            raise ValueError("governance record falsely denies CUDA hardware evidence")
        if gate.get("cuda_release_verified") is not True:
            raise ValueError("governance record hardware CUDA release gate is unverified")
        cuda_run_id = gate.get("cuda_run_id")
        if not isinstance(cuda_run_id, str) or re.fullmatch(r"[1-9][0-9]*", cuda_run_id) is None:
            raise ValueError("CUDA run id must be a positive integer string")
        expected_url = f"https://github.com/{repository}/actions/runs/{cuda_run_id}"
        if str(evidence.get("cuda_run_id")) != cuda_run_id:
            raise ValueError("governance record CUDA evidence run id mismatch")
        if evidence.get("cuda_run_url") != expected_url:
            raise ValueError("governance record CUDA evidence run URL mismatch")
        return

    if mode != _EXCEPTION_MODE:
        raise ValueError(f"unsupported governance record CUDA release mode: {mode!r}")
    if set(gate) != _EXCEPTION_GATE_FIELDS:
        raise ValueError("governance record CPU-only CUDA release gate fields differ")
    expected_gate = {
        "schema_version": 1,
        "mode": _EXCEPTION_MODE,
        "status": "not_run",
        "exception_id": _CUDA_EXCEPTION_ID,
        "release_tag": _CUDA_EXCEPTION_TAG,
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
    if release_tag != _CUDA_EXCEPTION_TAG or not _strict_equal(dict(gate), expected_gate):
        raise ValueError(
            "governance record CPU-only CUDA release gate differs from the reviewed exception"
        )
    if "cuda_run_id" in evidence or "cuda_run_url" in evidence:
        raise ValueError("CPU-only governance record must not claim a CUDA evidence run")


def _rebuild_canonical_governance(
    record: Mapping[str, Any],
    *,
    policy_bytes: bytes,
    snapshot_bytes: bytes,
    governance_markdown: str,
    repository: str,
    release_tag: str,
    release_commit: str,
    source_run_id: str,
    source_run_attempt: int,
    source_actor: str,
    source_triggering_actor: str,
) -> None:
    """Rebuild the record from retained policy/snapshot bytes and exact source identity."""
    try:
        generator = importlib.import_module("scripts.create_release_governance_record")
    except ModuleNotFoundError:
        generator = importlib.import_module("create_release_governance_record")

    evidence = _mapping(record.get("evidence"), "governance record evidence")
    gate = _mapping(record.get("cuda_release_gate"), "governance record CUDA release gate")
    mode = gate.get("mode")
    cuda_run_id = None
    cuda_exception_id = None
    if mode == _HARDWARE_MODE:
        value = gate.get("cuda_run_id")
        cuda_run_id = value if isinstance(value, str) else None
    elif mode == _EXCEPTION_MODE:
        value = gate.get("exception_id")
        cuda_exception_id = value if isinstance(value, str) else None

    expected_record = generator.build_record(
        policy_bytes=policy_bytes,
        snapshot_bytes=snapshot_bytes,
        repository=repository,
        release_tag=release_tag,
        release_commit=release_commit,
        preflight_run_id=str(evidence.get("preflight_run_id")),
        cuda_run_id=cuda_run_id,
        cuda_exception_id=cuda_exception_id,
        release_run_id=source_run_id,
        release_run_attempt=str(source_run_attempt),
        release_actor=source_actor,
        release_triggering_actor=source_triggering_actor,
    )
    if not _strict_equal(dict(record), dict(expected_record)):
        raise ValueError(
            "governance record differs from the exact retained policy and external-control snapshot"
        )
    expected_markdown = generator.render_markdown(expected_record)
    if governance_markdown != expected_markdown:
        raise ValueError("retained governance Markdown is not the canonical record rendering")


def verify_recovery_governance_record(
    record: Mapping[str, Any],
    run: Mapping[str, Any],
    *,
    policy_bytes: bytes,
    snapshot_bytes: bytes,
    governance_markdown: str,
    repository: str,
    release_tag: str,
    release_commit: str,
    source_run_id: str,
) -> None:
    """Bind the retained governance asset to the exact original publish run."""
    if _TAG.fullmatch(release_tag) is None:
        raise ValueError("release tag must have the form vMAJOR.MINOR.PATCH")
    if _COMMIT.fullmatch(release_commit) is None:
        raise ValueError("release commit must be a complete lowercase commit SHA")
    if re.fullmatch(r"[1-9][0-9]*", source_run_id) is None:
        raise ValueError("source run id must be a positive integer")
    schema_version = record.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError("release governance record schema_version must be the integer 1")

    source_repository = _mapping(run.get("repository"), "source run repository").get("full_name")
    source_id = _positive_integer(run.get("id"), "source run id")
    source_attempt = _positive_integer(run.get("run_attempt"), "source run attempt")
    source_actor = _mapping(run.get("actor"), "source run actor").get("login")
    source_triggering_actor = _mapping(
        run.get("triggering_actor"), "source run triggering actor"
    ).get("login")
    if not isinstance(source_actor, str) or not source_actor:
        raise ValueError("source run actor login must be a non-empty string")
    if not isinstance(source_triggering_actor, str) or not source_triggering_actor:
        raise ValueError("source run triggering actor login must be a non-empty string")

    source_fields = {
        "id": (str(source_id), source_run_id),
        "repository": (source_repository, repository),
        "tag": (run.get("head_branch"), release_tag),
        "commit": (run.get("head_sha"), release_commit),
    }
    for field, (actual, expected) in source_fields.items():
        if actual != expected:
            raise ValueError(
                f"source run {field} mismatch for governance binding: "
                f"expected {expected!r}, got {actual!r}"
            )

    release = _mapping(record.get("release"), "governance record release")
    governance = _mapping(record.get("governance"), "governance record governance")
    evidence = _mapping(record.get("evidence"), "governance record evidence")
    _verify_recovery_cuda_release_gate(
        record,
        repository=repository,
        release_tag=release_tag,
        release_commit=release_commit,
        governance=governance,
        evidence=evidence,
    )
    expected_url = f"https://github.com/{repository}/actions/runs/{source_run_id}"
    record_fields = {
        "repository": (release.get("repository"), repository),
        "tag": (release.get("tag"), release_tag),
        "commit": (release.get("commit"), release_commit),
        "source run id": (str(evidence.get("release_workflow_run_id")), source_run_id),
        "source run URL": (evidence.get("release_workflow_run_url"), expected_url),
        "source run attempt": (str(governance.get("release_run_attempt")), str(source_attempt)),
        "source actor": (governance.get("release_dispatch_actor"), source_actor),
        "source triggering actor": (
            governance.get("release_triggering_actor"),
            source_triggering_actor,
        ),
    }
    for field, (actual, expected) in record_fields.items():
        if actual != expected:
            raise ValueError(
                f"governance record {field} mismatch: expected {expected!r}, got {actual!r}"
            )

    _rebuild_canonical_governance(
        record,
        policy_bytes=policy_bytes,
        snapshot_bytes=snapshot_bytes,
        governance_markdown=governance_markdown,
        repository=repository,
        release_tag=release_tag,
        release_commit=release_commit,
        source_run_id=source_run_id,
        source_run_attempt=source_attempt,
        source_actor=source_actor,
        source_triggering_actor=source_triggering_actor,
    )


def verify_source_run(
    run: Mapping[str, Any],
    jobs_response: Mapping[str, Any],
    *,
    repository: str,
    workflow_path: str,
    release_tag: str,
    release_commit: str,
) -> str:
    """Prove recovery consumes a completed original publish run, not a rebuild."""
    if _TAG.fullmatch(release_tag) is None:
        raise ValueError("release tag must have the form vMAJOR.MINOR.PATCH")
    if _COMMIT.fullmatch(release_commit) is None:
        raise ValueError("release commit must be a complete lowercase commit SHA")
    actual_repository = _mapping(run.get("repository"), "source run repository").get("full_name")
    expected_fields = {
        "repository": (actual_repository, repository),
        "workflow path": (run.get("path"), workflow_path),
        "event": (run.get("event"), "workflow_dispatch"),
        "head SHA": (run.get("head_sha"), release_commit),
        "head branch/tag": (run.get("head_branch"), release_tag),
        "status": (run.get("status"), "completed"),
        "conclusion": (run.get("conclusion"), "failure"),
    }
    for label, (actual, expected) in expected_fields.items():
        if actual != expected:
            raise ValueError(f"source run {label} mismatch: expected {expected!r}, got {actual!r}")

    if jobs_response.get("query_filter") != "all":
        raise ValueError("source jobs must be queried with filter=all")
    if jobs_response.get("pagination_complete") is not True:
        raise ValueError("source jobs response does not prove complete pagination")
    jobs = [
        _mapping(value, "source workflow job")
        for value in _sequence(jobs_response.get("jobs"), "source jobs")
    ]
    required_jobs = {
        "Verify, build once, and inventory",
        "Attest the immutable distributions",
        "Publish through PyPI Trusted Publishing",
    }
    for job_name in sorted(required_jobs):
        matches = [job for job in jobs if job.get("name") == job_name]
        if len(matches) != 1:
            raise ValueError(
                f"source run must contain exactly one all-attempt {job_name!r} job; "
                f"got {len(matches)}"
            )
        job = matches[0]
        if job.get("status") != "completed" or job.get("conclusion") != "success":
            raise ValueError(
                f"source job {job_name!r} did not complete successfully: "
                f"{job.get('status')!r}/{job.get('conclusion')!r}"
            )

    release_job_name = "Create the immutable GitHub release"
    release_matches = [job for job in jobs if job.get("name") == release_job_name]
    if len(release_matches) != 1:
        raise ValueError(
            f"source run must contain exactly one all-attempt {release_job_name!r} job; "
            f"got {len(release_matches)}"
        )
    release_job = release_matches[0]
    if release_job.get("status") != "completed" or release_job.get("conclusion") != "failure":
        raise ValueError(
            f"source job {release_job_name!r} must demonstrate a downstream failure: "
            f"got {release_job.get('status')!r}/{release_job.get('conclusion')!r}"
        )

    steps = [
        _mapping(value, "GitHub Release job step")
        for value in _sequence(release_job.get("steps"), "GitHub Release job steps")
    ]
    stage_name = "Stage an explicitly requested post-PyPI recovery drill"
    stage_matches = [
        (index, step) for index, step in enumerate(steps) if step.get("name") == stage_name
    ]
    if len(stage_matches) != 1:
        raise ValueError(f"GitHub Release job must contain exactly one {stage_name!r} step")
    stage_index, stage_step = stage_matches[0]
    later_steps = steps[stage_index + 1 :]
    if stage_step.get("status") != "completed":
        raise ValueError("recovery-drill staging step did not complete")
    if stage_step.get("conclusion") == "failure":
        for expected_name in _POST_PYPI_RELEASE_STEPS:
            matches = [step for step in later_steps if step.get("name") == expected_name]
            if len(matches) != 1:
                raise ValueError(
                    f"staged recovery drill must contain exactly one {expected_name!r} step"
                )
            step = matches[0]
            if step.get("status") != "completed" or step.get("conclusion") != "skipped":
                raise ValueError(
                    f"staged recovery drill executed downstream step {expected_name!r}: "
                    f"{step.get('status')!r}/{step.get('conclusion')!r}"
                )
        complete_steps = [step for step in later_steps if step.get("name") == "Complete job"]
        if len(complete_steps) != 1 or complete_steps[0].get("conclusion") != "success":
            raise ValueError("staged recovery drill has no successful runner Complete job step")
        allowed_names = {*_POST_PYPI_RELEASE_STEPS, "Complete job"}
        unexpected = [
            step.get("name") for step in later_steps if step.get("name") not in allowed_names
        ]
        if unexpected:
            raise ValueError(
                f"staged recovery drill contains unexpected later steps: {unexpected!r}"
            )
        return "staged_drill"
    if stage_step.get("conclusion") != "skipped":
        raise ValueError(
            "recovery-drill staging step must be failed for a drill or skipped for an "
            "unplanned downstream failure"
        )
    if not any(step.get("conclusion") == "failure" for step in later_steps):
        raise ValueError("source job has no failed post-PyPI GitHub Release step")
    return "unplanned_downstream_failure"


def _version_from_tag(tag: str) -> str:
    match = _TAG.fullmatch(tag)
    if match is None:
        raise ValueError("release tag must have the form vMAJOR.MINOR.PATCH")
    return ".".join(match.groups())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    source = subparsers.add_parser("source-run")
    source.add_argument("--run-json", type=Path, required=True)
    source.add_argument("--jobs-json", type=Path, required=True)
    source.add_argument("--repository", required=True)
    source.add_argument("--workflow-path", default=".github/workflows/publish-pypi.yml")
    source.add_argument("--tag", required=True)
    source.add_argument("--commit", required=True)
    source.add_argument("--require-staged-drill", action="store_true")
    artifacts = subparsers.add_parser("artifacts")
    artifacts.add_argument("--sums", type=Path, required=True)
    artifacts.add_argument("--dist", type=Path, required=True)
    artifacts.add_argument("--pypi-json", type=Path, required=True)
    artifacts.add_argument("--project", default="explainiverse")
    artifacts.add_argument("--tag", required=True)
    artifacts.add_argument("--github-assets", type=Path)
    artifacts.add_argument("--provenance", type=Path)
    governance = subparsers.add_parser("governance-record")
    governance.add_argument("--record-json", type=Path, required=True)
    governance.add_argument("--record-markdown", type=Path, required=True)
    governance.add_argument("--policy", type=Path, required=True)
    governance.add_argument("--snapshot", type=Path, required=True)
    governance.add_argument("--run-json", type=Path, required=True)
    governance.add_argument("--repository", required=True)
    governance.add_argument("--tag", required=True)
    governance.add_argument("--commit", required=True)
    governance.add_argument("--source-run-id", required=True)
    release_body = subparsers.add_parser("release-body")
    release_body.add_argument("--release-json", type=Path, required=True)
    release_body.add_argument("--disclosure", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "source-run":
            run = _mapping(json.loads(args.run_json.read_text(encoding="utf-8")), "run JSON")
            jobs = _mapping(json.loads(args.jobs_json.read_text(encoding="utf-8")), "jobs JSON")
            source_kind = verify_source_run(
                run,
                jobs,
                repository=args.repository,
                workflow_path=args.workflow_path,
                release_tag=args.tag,
                release_commit=args.commit.strip().lower(),
            )
            if args.require_staged_drill and source_kind != "staged_drill":
                raise ValueError(
                    "source run is an unplanned downstream failure, not the required staged drill"
                )
            print(f"verified recovery source kind: {source_kind}")
        elif args.command == "governance-record":
            record = _mapping(
                json.loads(args.record_json.read_text(encoding="utf-8")),
                "release governance record",
            )
            run = _mapping(
                json.loads(args.run_json.read_text(encoding="utf-8")),
                "source run JSON",
            )
            verify_recovery_governance_record(
                record,
                run,
                policy_bytes=args.policy.read_bytes(),
                snapshot_bytes=args.snapshot.read_bytes(),
                governance_markdown=args.record_markdown.read_text(encoding="utf-8"),
                repository=args.repository,
                release_tag=args.tag,
                release_commit=args.commit.strip().lower(),
                source_run_id=args.source_run_id,
            )
            print("verified governance record binding to the original publish run")
        elif args.command == "artifacts":
            expected = parse_sha256sums(args.sums)
            verify_distribution_directory(args.dist, expected)
            metadata = _mapping(json.loads(args.pypi_json.read_text(encoding="utf-8")), "PyPI JSON")
            verify_pypi_json(
                metadata,
                project=args.project,
                version=_version_from_tag(args.tag),
                expected=expected,
            )
            if args.github_assets is not None:
                if args.provenance is None:
                    raise ValueError("--provenance is required with --github-assets")
                verify_distribution_directory(args.github_assets, expected)
                verify_release_asset_directory(
                    args.github_assets,
                    expected_directories=[args.dist, args.provenance],
                )
        else:
            release = _mapping(
                json.loads(args.release_json.read_text(encoding="utf-8")),
                "GitHub release JSON",
            )
            disclosure = args.disclosure.read_text(encoding="utf-8").strip()
            verify_release_governance_disclosure(release, disclosure)
        return 0
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
