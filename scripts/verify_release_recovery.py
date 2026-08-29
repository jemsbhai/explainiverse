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
    "Verify signed immutable release source before candidate code",
    "Set up Python 3.12 for provenance verification",
    "Pin the provenance verifier installer",
    "Install the hash-locked provenance verifier",
    "Download and hard-verify the exact release artifacts",
    "Verify release assets against reviewed hashes and PyPI provenance",
    "Prepare the exact fixed-command GitHub Release plan",
    "Archive the fixed normal-release plan",
)
_RELEASE_PREP_JOB = "Create the immutable GitHub release"
_RELEASE_FINALIZE_JOB = "Finalize the immutable GitHub release with fixed commands"


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


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


def _first_attempt(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != 1:
        raise ValueError(f"{name} must be the integer 1")
    return value


def verify_recovery_governance_record(
    record: Mapping[str, Any],
    run: Mapping[str, Any],
    *,
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
    if isinstance(schema_version, bool) or schema_version != 1:
        raise ValueError("release governance record schema_version must be the integer 1")

    source_repository = _mapping(run.get("repository"), "source run repository").get("full_name")
    source_id = _positive_integer(run.get("id"), "source run id")
    source_attempt = _first_attempt(run.get("run_attempt"), "source run attempt")
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
        "capture principal": (governance.get("capture_principal"), source_actor),
    }
    for field, (actual, expected) in record_fields.items():
        if actual != expected:
            raise ValueError(
                f"governance record {field} mismatch: expected {expected!r}, got {actual!r}"
            )


def verify_source_run_evidence(
    run: Mapping[str, Any],
    jobs_response: Mapping[str, Any],
    *,
    repository: str,
    workflow_path: str,
    release_tag: str,
    release_commit: str,
) -> Mapping[str, Any]:
    """Verify and normalize the exact trusted jobs from the original publish run."""
    if _TAG.fullmatch(release_tag) is None:
        raise ValueError("release tag must have the form vMAJOR.MINOR.PATCH")
    if _COMMIT.fullmatch(release_commit) is None:
        raise ValueError("release commit must be a complete lowercase commit SHA")
    actual_repository = _mapping(run.get("repository"), "source run repository").get("full_name")
    source_id = _positive_integer(run.get("id"), "source run id")
    source_attempt = _first_attempt(run.get("run_attempt"), "source run attempt")
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
    trusted_jobs: list[Mapping[str, Any]] = []
    trusted_job_ids: dict[int, str] = {}

    def verify_trusted_job(job: Mapping[str, Any], job_name: str) -> Mapping[str, Any]:
        job_id = _positive_integer(job.get("id"), f"source job {job_name!r} id")
        previous_job = trusted_job_ids.get(job_id)
        if previous_job is not None:
            raise ValueError(
                f"source job id {job_id!r} is reused by trusted jobs "
                f"{previous_job!r} and {job_name!r}"
            )
        trusted_job_ids[job_id] = job_name
        job_run_id = _positive_integer(job.get("run_id"), f"source job {job_name!r} run id")
        if job_run_id != source_id:
            raise ValueError(
                f"source job {job_name!r} run id mismatch: "
                f"expected {source_id!r}, got {job_run_id!r}"
            )
        job_head_sha = job.get("head_sha")
        if job_head_sha != release_commit:
            raise ValueError(
                f"source job {job_name!r} head SHA mismatch: "
                f"expected {release_commit!r}, got {job_head_sha!r}"
            )
        job_attempt = _first_attempt(job.get("run_attempt"), f"source job {job_name!r} attempt")
        return {
            "id": job_id,
            "run_id": job_run_id,
            "name": job_name,
            "head_sha": job_head_sha,
            "status": job.get("status"),
            "conclusion": job.get("conclusion"),
            "run_attempt": job_attempt,
        }

    for job_name in sorted(required_jobs):
        matches = [job for job in jobs if job.get("name") == job_name]
        if len(matches) != 1:
            raise ValueError(
                f"source run must contain exactly one all-attempt {job_name!r} job; "
                f"got {len(matches)}"
            )
        job = matches[0]
        normalized_job = verify_trusted_job(job, job_name)
        if job.get("status") != "completed" or job.get("conclusion") != "success":
            raise ValueError(
                f"source job {job_name!r} did not complete successfully: "
                f"{job.get('status')!r}/{job.get('conclusion')!r}"
            )
        trusted_jobs.append(normalized_job)

    downstream_jobs: dict[str, Mapping[str, Any]] = {}
    normalized_downstream_jobs: dict[str, Mapping[str, Any]] = {}
    for job_name in (_RELEASE_PREP_JOB, _RELEASE_FINALIZE_JOB):
        matches = [job for job in jobs if job.get("name") == job_name]
        if len(matches) != 1:
            raise ValueError(
                f"source run must contain exactly one all-attempt {job_name!r} job; "
                f"got {len(matches)}"
            )
        downstream_jobs[job_name] = matches[0]
        normalized_downstream_jobs[job_name] = verify_trusted_job(matches[0], job_name)

    release_job = downstream_jobs[_RELEASE_PREP_JOB]
    finalizer_job = downstream_jobs[_RELEASE_FINALIZE_JOB]

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
        if release_job.get("status") != "completed" or release_job.get("conclusion") != "failure":
            raise ValueError(
                f"source job {_RELEASE_PREP_JOB!r} must fail for a staged drill: "
                f"got {release_job.get('status')!r}/{release_job.get('conclusion')!r}"
            )
        if (
            finalizer_job.get("status") != "completed"
            or finalizer_job.get("conclusion") != "skipped"
        ):
            raise ValueError(
                f"source job {_RELEASE_FINALIZE_JOB!r} must be skipped for a staged drill: "
                f"got {finalizer_job.get('status')!r}/{finalizer_job.get('conclusion')!r}"
            )
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
        source_kind = "staged_drill"
    else:
        if stage_step.get("conclusion") != "skipped":
            raise ValueError(
                "recovery-drill staging step must be failed for a drill or skipped for an "
                "unplanned downstream failure"
            )
        if release_job.get("status") != "completed" or release_job.get("conclusion") != "success":
            raise ValueError(
                f"source job {_RELEASE_PREP_JOB!r} must succeed before a finalizer failure: "
                f"got {release_job.get('status')!r}/{release_job.get('conclusion')!r}"
            )
        if (
            finalizer_job.get("status") != "completed"
            or finalizer_job.get("conclusion") != "failure"
        ):
            raise ValueError(
                f"source job {_RELEASE_FINALIZE_JOB!r} must demonstrate the downstream failure: "
                f"got {finalizer_job.get('status')!r}/{finalizer_job.get('conclusion')!r}"
            )
        finalizer_steps = [
            _mapping(value, "GitHub Release finalizer job step")
            for value in _sequence(finalizer_job.get("steps"), "GitHub Release finalizer job steps")
        ]
        if not any(
            step.get("status") == "completed" and step.get("conclusion") == "failure"
            for step in finalizer_steps
        ):
            raise ValueError("source finalizer job has no completed failed step")
        source_kind = "unplanned_downstream_failure"

    trusted_jobs.extend(
        [
            normalized_downstream_jobs[_RELEASE_PREP_JOB],
            normalized_downstream_jobs[_RELEASE_FINALIZE_JOB],
        ]
    )
    return {
        "schema_version": 1,
        "source_kind": source_kind,
        "query_filter": "all",
        "pagination_complete": True,
        "run": {
            "id": source_id,
            "repository": actual_repository,
            "path": run.get("path"),
            "event": run.get("event"),
            "head_branch": run.get("head_branch"),
            "head_sha": run.get("head_sha"),
            "status": run.get("status"),
            "conclusion": run.get("conclusion"),
            "run_attempt": source_attempt,
        },
        "trusted_jobs": trusted_jobs,
    }


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
    evidence = verify_source_run_evidence(
        run,
        jobs_response,
        repository=repository,
        workflow_path=workflow_path,
        release_tag=release_tag,
        release_commit=release_commit,
    )
    return str(evidence["source_kind"])


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
    source.add_argument("--normalized-output", type=Path)
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
            source_evidence = verify_source_run_evidence(
                run,
                jobs,
                repository=args.repository,
                workflow_path=args.workflow_path,
                release_tag=args.tag,
                release_commit=args.commit.strip().lower(),
            )
            source_kind = str(source_evidence["source_kind"])
            if args.require_staged_drill and source_kind != "staged_drill":
                raise ValueError(
                    "source run is an unplanned downstream failure, not the required staged drill"
                )
            if args.normalized_output is not None:
                args.normalized_output.parent.mkdir(parents=True, exist_ok=True)
                args.normalized_output.write_text(
                    json.dumps(source_evidence, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
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
