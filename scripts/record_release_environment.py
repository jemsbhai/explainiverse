"""Record the exact hosted-runner and bootstrap identities used for a release build."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

_RELEASE_TOOLS = ("poetry", "twine", "cyclonedx-bom")
_RUNNER_ENVIRONMENT_KEYS = (
    "GITHUB_ACTION",
    "GITHUB_RUN_ATTEMPT",
    "GITHUB_RUN_ID",
    "GITHUB_SHA",
    "ImageOS",
    "ImageVersion",
    "RUNNER_ARCH",
    "RUNNER_NAME",
    "RUNNER_OS",
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def release_environment(requirements: Path) -> dict[str, Any]:
    """Build a versioned, JSON-safe environment identity payload."""
    if not requirements.is_file():
        raise ValueError(f"hashed release requirements do not exist: {requirements}")

    pip_version = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "schema_version": 1,
        "python": {
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "bootstrap_pip": pip_version,
        "platform": platform.platform(),
        "release_tools": {
            package: importlib.metadata.version(package) for package in _RELEASE_TOOLS
        },
        "requirements": {
            "path": requirements.as_posix(),
            "sha256": _file_sha256(requirements),
        },
        "runner": {key: os.environ.get(key) for key in _RUNNER_ENVIRONMENT_KEYS},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirements", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    try:
        payload = release_environment(args.requirements)
    except (ValueError, importlib.metadata.PackageNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
