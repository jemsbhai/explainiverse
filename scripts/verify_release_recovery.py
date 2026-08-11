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


def verify_source_run(
    run: Mapping[str, Any],
    jobs_response: Mapping[str, Any],
    *,
    repository: str,
    workflow_path: str,
    release_tag: str,
    release_commit: str,
) -> None:
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
    artifacts = subparsers.add_parser("artifacts")
    artifacts.add_argument("--sums", type=Path, required=True)
    artifacts.add_argument("--dist", type=Path, required=True)
    artifacts.add_argument("--pypi-json", type=Path, required=True)
    artifacts.add_argument("--project", default="explainiverse")
    artifacts.add_argument("--tag", required=True)
    artifacts.add_argument("--github-assets", type=Path)
    artifacts.add_argument("--provenance", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "source-run":
            run = _mapping(json.loads(args.run_json.read_text(encoding="utf-8")), "run JSON")
            jobs = _mapping(json.loads(args.jobs_json.read_text(encoding="utf-8")), "jobs JSON")
            verify_source_run(
                run,
                jobs,
                repository=args.repository,
                workflow_path=args.workflow_path,
                release_tag=args.tag,
                release_commit=args.commit.strip().lower(),
            )
        else:
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
        return 0
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
