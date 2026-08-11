"""Fail closed while Explainiverse's public Python typing claim remains blocked."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import zipfile
from email.parser import BytesParser
from email.policy import default
from pathlib import Path, PurePosixPath
from typing import Sequence

_CANONICAL_PACKAGE = "explainiverse"
_CANONICAL_MARKER = "src/explainiverse/py.typed"
_CANONICAL_CLASSIFIER = "Typing :: Typed"
_CANONICAL_ACCEPTANCE = {
    "mypy": "poetry run mypy --strict src/explainiverse",
    "pyright": "pyright --verifytypes explainiverse --ignoreexternal",
    "required_result": (
        "Both commands exit zero, Pyright reports 100% type completeness, and an annotated "
        "built wheel passes strict external consumer fixtures before this guard is replaced."
    ),
}


class TypingReadinessError(ValueError):
    """Raised when the blocked/untyped distribution boundary is violated."""


def _is_typing_marker(member_name: str, package: str) -> bool:
    parts = PurePosixPath(member_name.replace("\\", "/")).parts
    return len(parts) >= 2 and parts[-2:] == (package, "py.typed")


def _archive_members(path: Path) -> list[str]:
    if path.suffix == ".whl" or path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    if path.name.endswith(".tar.gz") or path.suffix in {".tar", ".tgz"}:
        with tarfile.open(path) as archive:
            return archive.getnames()
    raise TypingReadinessError(f"unsupported distribution archive: {path}")


def _archive_metadata(path: Path) -> dict[str, bytes]:
    def is_metadata(name: str) -> bool:
        normalized = PurePosixPath(name.replace("\\", "/"))
        return normalized.name in {"METADATA", "PKG-INFO"}

    if path.suffix in {".whl", ".zip"}:
        with zipfile.ZipFile(path) as zip_archive:
            return {
                name: zip_archive.read(name)
                for name in zip_archive.namelist()
                if is_metadata(name) and not name.endswith("/")
            }
    if path.name.endswith(".tar.gz") or path.suffix in {".tar", ".tgz"}:
        metadata: dict[str, bytes] = {}
        with tarfile.open(path) as tar_archive:
            for member in tar_archive.getmembers():
                if not member.isfile() or not is_metadata(member.name):
                    continue
                stream = tar_archive.extractfile(member)
                if stream is None:
                    raise TypingReadinessError(
                        f"cannot read distribution metadata {member.name!r} from {path}"
                    )
                metadata[member.name] = stream.read()
        return metadata
    raise TypingReadinessError(f"unsupported distribution archive: {path}")


def load_blocked_policy(path: Path) -> dict[str, object]:
    try:
        policy = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TypingReadinessError(f"cannot load typing policy {path}: {exc}") from exc
    if not isinstance(policy, dict) or policy.get("schema_version") != 1:
        raise TypingReadinessError("typing policy must be a schema-version-1 object")
    if policy.get("claim_status") != "blocked":
        raise TypingReadinessError(
            "this guard only permits the audited blocked state; replace it with green strict "
            "consumer certification before declaring typed support"
        )
    if policy.get("package") != _CANONICAL_PACKAGE:
        raise TypingReadinessError("typing policy package must remain 'explainiverse'")
    expected_boundaries = {
        "forbidden_marker": _CANONICAL_MARKER,
        "forbidden_classifier": _CANONICAL_CLASSIFIER,
    }
    for field, expected in expected_boundaries.items():
        if policy.get(field) != expected:
            raise TypingReadinessError(
                f"typing policy {field} must remain the canonical boundary {expected!r}"
            )
    if policy.get("acceptance") != _CANONICAL_ACCEPTANCE:
        raise TypingReadinessError(
            "typing policy acceptance must remain the canonical strict consumer boundary"
        )
    return policy


def audit_blocked_distribution(
    *,
    policy_path: Path,
    project_file: Path,
    repository_root: Path,
    distributions: Sequence[Path] = (),
) -> dict[str, object]:
    """Assert that neither source metadata nor built archives claim PEP 561 support."""
    policy = load_blocked_policy(policy_path)
    marker = repository_root / _CANONICAL_MARKER
    if marker.exists():
        raise TypingReadinessError(f"blocked typing marker exists: {_CANONICAL_MARKER}")
    try:
        project_text = project_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise TypingReadinessError(f"cannot read project metadata {project_file}: {exc}") from exc
    if _CANONICAL_CLASSIFIER in project_text:
        raise TypingReadinessError(f"blocked typing classifier exists: {_CANONICAL_CLASSIFIER}")

    package = str(policy["package"])
    checked_archives: list[str] = []
    for distribution in distributions:
        if not distribution.is_file():
            raise TypingReadinessError(f"distribution does not exist: {distribution}")
        marker_members = [
            name for name in _archive_members(distribution) if _is_typing_marker(name, package)
        ]
        if marker_members:
            raise TypingReadinessError(
                f"blocked typing marker shipped in {distribution}: {marker_members!r}"
            )
        metadata = _archive_metadata(distribution)
        if not metadata:
            raise TypingReadinessError(
                f"distribution contains no auditable METADATA or PKG-INFO: {distribution}"
            )
        typed_metadata = []
        for name, payload in metadata.items():
            parsed = BytesParser(policy=default).parsebytes(payload)
            classifiers = parsed.get_all("Classifier", [])
            if any(str(value).strip() == _CANONICAL_CLASSIFIER for value in classifiers):
                typed_metadata.append(name)
        if typed_metadata:
            raise TypingReadinessError(
                f"blocked typing classifier shipped in {distribution}: {typed_metadata!r}"
            )
        checked_archives.append(str(distribution))

    return {
        "schema_version": 1,
        "claim_status": "blocked",
        "source_marker_absent": True,
        "typed_classifier_absent": True,
        "checked_distributions": checked_archives,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path(".github/typing-readiness-policy.json"),
    )
    parser.add_argument("--project-file", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--distribution", action="append", type=Path, default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = audit_blocked_distribution(
            policy_path=args.policy,
            project_file=args.project_file,
            repository_root=args.repository_root,
            distributions=args.distribution,
        )
    except (OSError, TypingReadinessError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
