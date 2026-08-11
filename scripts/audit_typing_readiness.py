"""Fail closed while Explainiverse's public Python typing claim remains blocked."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Sequence


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
    if policy.get("package") != "explainiverse":
        raise TypingReadinessError("typing policy package must remain 'explainiverse'")
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
    marker_value = policy.get("forbidden_marker")
    classifier_value = policy.get("forbidden_classifier")
    if not isinstance(marker_value, str) or not isinstance(classifier_value, str):
        raise TypingReadinessError("typing policy is missing its forbidden claim boundaries")

    marker = repository_root / marker_value
    if marker.exists():
        raise TypingReadinessError(f"blocked typing marker exists: {marker_value}")
    try:
        project_text = project_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise TypingReadinessError(f"cannot read project metadata {project_file}: {exc}") from exc
    if classifier_value in project_text:
        raise TypingReadinessError(f"blocked typing classifier exists: {classifier_value}")

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
