"""Record the boundary of a source-only dependency candidate probe.

This verifier is intentionally not a full installed-distribution compatibility
gate.  The checked-in Explainiverse metadata still excludes the candidate, so
the candidate lane removes the Explainiverse distribution, imports the checkout
through ``PYTHONPATH``, and records that its successful ``pip check`` covers only
the remaining dependency environment.  When supplied, a built wheel is checked
to prove that it still carries the reviewed bound rather than being presented as
candidate-compatible metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from email.parser import BytesParser
from email.policy import default
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as installed_version
from pathlib import Path
from typing import Any, Mapping, Sequence

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised by the Python 3.10 CI lane
    import tomli as tomllib


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a table")
    return value


def _project_metadata(project_file: Path) -> tuple[str, str, list[str]]:
    project = _mapping(
        tomllib.loads(project_file.read_text(encoding="utf-8")).get("project"),
        "project metadata",
    )
    name = project.get("name")
    version = project.get("version")
    dependencies = project.get("dependencies")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("project metadata name must be a non-empty string")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("project metadata version must be a non-empty string")
    if not isinstance(dependencies, list) or not all(
        isinstance(value, str) for value in dependencies
    ):
        raise ValueError("project metadata dependencies must be an array of strings")
    return name, version, dependencies


def _requirement_for(dependencies: Sequence[str], package: str, label: str) -> Requirement:
    package_name = canonicalize_name(package)
    matches: list[Requirement] = []
    for raw in dependencies:
        try:
            requirement = Requirement(raw)
        except InvalidRequirement as exc:
            raise ValueError(f"{label} contains an invalid requirement {raw!r}") from exc
        if canonicalize_name(requirement.name) == package_name:
            matches.append(requirement)
    if len(matches) != 1:
        raise ValueError(
            f"{label} must contain exactly one {package!r} requirement; got {len(matches)}"
        )
    return matches[0]


def _requirement_identity(requirement: Requirement) -> tuple[object, ...]:
    return (
        canonicalize_name(requirement.name),
        tuple(sorted(requirement.extras)),
        str(requirement.specifier),
        str(requirement.marker) if requirement.marker is not None else None,
        requirement.url,
    )


def _wheel_metadata(wheel: Path) -> Mapping[str, Any]:
    if wheel.suffix != ".whl" or not wheel.is_file():
        raise ValueError(f"candidate boundary wheel does not exist or is not a wheel: {wheel}")
    with zipfile.ZipFile(wheel) as archive:
        metadata_names = [
            name
            for name in archive.namelist()
            if name.endswith(".dist-info/METADATA") and not name.endswith("/")
        ]
        if len(metadata_names) != 1:
            raise ValueError(
                "wheel must contain exactly one .dist-info/METADATA file; "
                f"got {metadata_names!r}"
            )
        metadata = BytesParser(policy=default).parsebytes(archive.read(metadata_names[0]))
    return {
        "name": metadata.get("Name"),
        "version": metadata.get("Version"),
        "requirements": metadata.get_all("Requires-Dist", []),
    }


def build_probe_record(
    *,
    project_file: Path,
    package: str,
    candidate: str,
    wheel: Path | None = None,
) -> Mapping[str, Any]:
    """Validate and describe a source-only, dependency-only candidate probe."""
    project_name, project_version, project_dependencies = _project_metadata(project_file)
    project_requirement = _requirement_for(project_dependencies, package, "project dependencies")
    try:
        candidate_version = Version(candidate)
    except InvalidVersion as exc:
        raise ValueError(f"candidate version is invalid: {candidate!r}") from exc
    if not candidate_version.is_prerelease:
        raise ValueError("candidate must be a prerelease")
    if project_requirement.specifier.contains(candidate_version, prereleases=True):
        raise ValueError(
            "candidate unexpectedly satisfies the checked-in dependency bound; this probe is "
            "reserved for an outside-bound source compatibility check"
        )

    try:
        installed_project_version = installed_version(project_name)
    except PackageNotFoundError:
        installed_project_version = None
    if installed_project_version is not None:
        raise ValueError(
            "source-only candidate probe requires the Explainiverse distribution to be absent; "
            f"found {project_name} {installed_project_version}"
        )
    try:
        observed_candidate = installed_version(package)
    except PackageNotFoundError as exc:
        raise ValueError(f"candidate dependency {package!r} is not installed") from exc
    if Version(observed_candidate) != candidate_version:
        raise ValueError(
            f"installed {package!r} version {observed_candidate!r} does not match "
            f"candidate {candidate!r}"
        )

    record: dict[str, Any] = {
        "schema_version": 1,
        "probe_scope": "source-compatibility-only",
        "post_candidate_graph_scope": "dependencies-only",
        "project_distribution_installed": False,
        "full_distribution_graph_verified": False,
        "project": {"name": project_name, "version": project_version},
        "candidate_dependency": {"name": package, "version": str(candidate_version)},
        "checked_in_requirement": str(project_requirement),
        "candidate_satisfies_checked_in_requirement": False,
        "wheel_metadata": None,
    }

    if wheel is not None:
        metadata = _wheel_metadata(wheel)
        if canonicalize_name(str(metadata["name"])) != canonicalize_name(project_name):
            raise ValueError("wheel project name differs from checked-in project metadata")
        if metadata["version"] != project_version:
            raise ValueError("wheel version differs from checked-in project metadata")
        wheel_requirement = _requirement_for(
            list(metadata["requirements"]), package, "wheel requirements"
        )
        if _requirement_identity(wheel_requirement) != _requirement_identity(project_requirement):
            raise ValueError(
                "wheel dependency requirement differs from checked-in project metadata"
            )
        if wheel_requirement.specifier.contains(candidate_version, prereleases=True):
            raise ValueError("wheel unexpectedly claims the outside-bound candidate is supported")
        record["wheel_metadata"] = {
            "path": str(wheel),
            "requirement": str(wheel_requirement),
            "candidate_satisfies_requirement": False,
        }
    return record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-file", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--package", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        record = build_probe_record(
            project_file=args.project_file,
            package=args.package,
            candidate=args.candidate,
            wheel=args.wheel,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, TypeError, ValueError, zipfile.BadZipFile) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
