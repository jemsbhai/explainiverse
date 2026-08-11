"""Record and compare hosted-runner identities used for release builds."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_RELEASE_TOOLS = ("poetry", "twine", "cyclonedx-bom")
_RUNNER_ENVIRONMENT_KEYS = (
    "GITHUB_ACTION",
    "GITHUB_ACTIONS",
    "GITHUB_JOB",
    "GITHUB_REF",
    "GITHUB_REPOSITORY",
    "GITHUB_RUN_ATTEMPT",
    "GITHUB_RUN_ID",
    "GITHUB_SHA",
    "GITHUB_WORKFLOW",
    "ImageOS",
    "ImageVersion",
    "RUNNER_ARCH",
    "RUNNER_NAME",
    "RUNNER_OS",
    "REPRODUCIBILITY_BUILD_ID",
    "REPRODUCIBILITY_JOB_INDEX",
    "REPRODUCIBILITY_JOB_TOTAL",
    "REPRODUCIBILITY_RUNNER_ENVIRONMENT",
)
_STABLE_RUNNER_KEYS = (
    "GITHUB_JOB",
    "GITHUB_ACTIONS",
    "GITHUB_REF",
    "GITHUB_REPOSITORY",
    "GITHUB_RUN_ATTEMPT",
    "GITHUB_RUN_ID",
    "GITHUB_SHA",
    "GITHUB_WORKFLOW",
    "ImageOS",
    "ImageVersion",
    "RUNNER_ARCH",
    "RUNNER_OS",
)
_BUILD_IDENTITY = re.compile(
    r"(?P<run_id>[1-9][0-9]*):(?P<run_attempt>[1-9][0-9]*):(?P<slot>one|two)"
)
_EXPECTED_JOB_INDEX = {"one": "0", "two": "1"}
_STABLE_ENVIRONMENT_PATHS = (
    ("schema_version",),
    ("python", "implementation"),
    ("python", "version"),
    ("bootstrap_pip_version",),
    ("platform_identity", "system"),
    ("platform_identity", "machine"),
    ("source",),
    ("release_tools",),
    ("requirements",),
) + tuple(("runner", key) for key in _STABLE_RUNNER_KEYS)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _command_stdout(command: Sequence[str]) -> str:
    return subprocess.run(
        list(command),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _pip_version(rendered: str) -> str:
    fields = rendered.split()
    if len(fields) < 2 or fields[0].lower() != "pip":
        raise ValueError(f"could not parse bootstrap pip identity: {rendered!r}")
    return fields[1]


def release_environment(requirements: Path, *, build_identity: str | None = None) -> dict[str, Any]:
    """Build a versioned, JSON-safe environment identity payload."""
    if not requirements.is_file():
        raise ValueError(f"hashed release requirements do not exist: {requirements}")
    if build_identity is not None and not build_identity.strip():
        raise ValueError("build identity must be non-empty when provided")

    pip_identity = _command_stdout([sys.executable, "-m", "pip", "--version"])
    payload: dict[str, Any] = {
        "schema_version": 1,
        "build_identity": build_identity,
        "python": {
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "bootstrap_pip": pip_identity,
        "bootstrap_pip_version": _pip_version(pip_identity),
        "platform": platform.platform(),
        "platform_identity": {
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "source": {
            "commit": _command_stdout(("git", "rev-parse", "HEAD")),
            "commit_timestamp": _command_stdout(("git", "log", "-1", "--format=%ct")),
        },
        "release_tools": {
            package: importlib.metadata.version(package) for package in _RELEASE_TOOLS
        },
        "requirements": {
            "path": requirements.as_posix(),
            "sha256": _file_sha256(requirements),
        },
        "runner": {key: os.environ.get(key) for key in _RUNNER_ENVIRONMENT_KEYS},
    }
    if build_identity is not None:
        expected_slot = _parse_build_identity(build_identity, "recorded")["slot"]
        _validate_hosted_build_identity(payload, "recorded", expected_slot)
    return payload


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _nested(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = payload
    for field in path:
        current = _mapping(current, ".".join(path))[field]
    return current


def _parse_build_identity(value: Any, label: str) -> Mapping[str, str]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} release environment has no build identity")
    match = _BUILD_IDENTITY.fullmatch(value)
    if match is None:
        raise ValueError(
            f"{label} build identity must be RUN_ID:RUN_ATTEMPT:one|two; got {value!r}"
        )
    return match.groupdict()


def _required_runner_field(payload: Mapping[str, Any], label: str, field: str) -> Any:
    try:
        return _nested(payload, ("runner", field))
    except KeyError as exc:
        raise ValueError(f"{label} release environment is missing runner field {field!r}") from exc


def _validate_hosted_build_identity(
    payload: Mapping[str, Any], label: str, expected_slot: str
) -> tuple[str, str]:
    identity = _parse_build_identity(payload.get("build_identity"), label)
    if identity["slot"] != expected_slot:
        raise ValueError(
            f"{label} build identity must use slot {expected_slot!r}; got {identity['slot']!r}"
        )

    bindings = {
        "run_id": (
            "runner.GITHUB_RUN_ID",
            _required_runner_field(payload, label, "GITHUB_RUN_ID"),
        ),
        "run_attempt": (
            "runner.GITHUB_RUN_ATTEMPT",
            _required_runner_field(payload, label, "GITHUB_RUN_ATTEMPT"),
        ),
        "slot": (
            "runner.REPRODUCIBILITY_BUILD_ID",
            _required_runner_field(payload, label, "REPRODUCIBILITY_BUILD_ID"),
        ),
    }
    for field, (runner_field, observed) in bindings.items():
        if identity[field] != observed:
            raise ValueError(
                f"{label} build identity {field} does not match {runner_field}: "
                f"{identity[field]!r} != {observed!r}"
            )

    job_index = _required_runner_field(payload, label, "REPRODUCIBILITY_JOB_INDEX")
    if job_index != _EXPECTED_JOB_INDEX[expected_slot]:
        raise ValueError(
            f"{label} slot {expected_slot!r} must use matrix job index "
            f"{_EXPECTED_JOB_INDEX[expected_slot]!r}; got {job_index!r}"
        )
    if _required_runner_field(payload, label, "REPRODUCIBILITY_JOB_TOTAL") != "2":
        raise ValueError(f"{label} reproducibility matrix must contain exactly two jobs")
    if (
        _required_runner_field(payload, label, "REPRODUCIBILITY_RUNNER_ENVIRONMENT")
        != "github-hosted"
    ):
        raise ValueError(f"{label} release build did not run on a GitHub-hosted runner")
    if _required_runner_field(payload, label, "GITHUB_ACTIONS") != "true":
        raise ValueError(f"{label} release build is not bound to GitHub Actions")

    runner_name = _required_runner_field(payload, label, "RUNNER_NAME")
    if not isinstance(runner_name, str) or not runner_name:
        raise ValueError(f"{label} release environment has no runner name")
    return str(payload["build_identity"]), runner_name


def compare_release_environments(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, Any]:
    """Require stable build inputs to match and logical build identities to differ."""
    first_identity, first_runner = _validate_hosted_build_identity(first, "first", "one")
    second_identity, second_runner = _validate_hosted_build_identity(second, "second", "two")

    matching_identity: dict[str, Any] = {}
    for path in _STABLE_ENVIRONMENT_PATHS:
        dotted = ".".join(path)
        try:
            first_value = _nested(first, path)
            second_value = _nested(second, path)
        except KeyError as exc:
            raise ValueError(f"release environment is missing stable field {dotted!r}") from exc
        if first_value in (None, "") or second_value in (None, ""):
            raise ValueError(f"stable release environment field {dotted!r} must be non-empty")
        if first_value != second_value:
            raise ValueError(
                f"stable release environment field {dotted!r} differs: "
                f"first={first_value!r}, second={second_value!r}"
            )
        matching_identity[dotted] = first_value

    for label, payload in (("first", first), ("second", second)):
        source_commit = _nested(payload, ("source", "commit"))
        github_sha = _nested(payload, ("runner", "GITHUB_SHA"))
        if source_commit != github_sha:
            raise ValueError(
                f"{label} source commit does not match runner GITHUB_SHA: "
                f"{source_commit!r} != {github_sha!r}"
            )
    return {
        "schema_version": 1,
        "comparison": "stable-environment-identity",
        "compatible": True,
        "distinct_build_identities": {
            "first": first_identity,
            "second": second_identity,
        },
        "hosted_runner_jobs": {
            "first": {
                "runner_name": first_runner,
                "matrix_slot": "one",
                "matrix_job_index": "0",
            },
            "second": {
                "runner_name": second_runner,
                "matrix_slot": "two",
                "matrix_job_index": "1",
            },
        },
        "matching_identity": matching_identity,
        "first": dict(first),
        "second": dict(second),
    }


def _load_manifest(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError(f"release environment manifest does not exist: {path}")
    return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _write_payload(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--requirements", type=Path)
    mode.add_argument("--compare", nargs=2, type=Path, metavar=("FIRST", "SECOND"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--build-identity")
    args = parser.parse_args()

    first_payload: Mapping[str, Any] | None = None
    second_payload: Mapping[str, Any] | None = None
    try:
        if args.compare is not None:
            if args.build_identity is not None:
                raise ValueError("--build-identity is valid only with --requirements")
            first, second = args.compare
            first_payload = _load_manifest(first)
            second_payload = _load_manifest(second)
            payload = compare_release_environments(first_payload, second_payload)
        else:
            payload = release_environment(args.requirements, build_identity=args.build_identity)
    except (
        ValueError,
        KeyError,
        OSError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
        importlib.metadata.PackageNotFoundError,
    ) as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "comparison": (
                "stable-environment-identity"
                if args.compare is not None
                else "release-environment-recording"
            ),
            "compatible": False,
            "error": str(exc),
        }
        if first_payload is not None:
            failure["first"] = dict(first_payload)
        if second_payload is not None:
            failure["second"] = dict(second_payload)
        _write_payload(args.output, failure)
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    _write_payload(args.output, payload)


if __name__ == "__main__":
    main()
