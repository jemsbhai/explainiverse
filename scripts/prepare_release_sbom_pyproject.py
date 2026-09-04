"""Create a PEP 621-only project view for the release SBOM generator.

CycloneDX Python treats any ``tool.poetry`` table as legacy package metadata,
even when Poetry 2 uses PEP 621's ``project`` table and ``tool.poetry`` contains
only build or dependency-group configuration.  The release SBOM must retain
the reviewed PEP 621 metadata while hiding that incompatible tool namespace.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised by the Python 3.10 CI lane
    import tomli as tomllib

_TABLE_HEADER = re.compile(
    r"^\s*(?P<open>\[\[|\[)(?P<path>[A-Za-z0-9_.-]+)(?P<close>\]\]|\])\s*(?:#.*)?$"
)


def _table_path(line: str) -> str | None:
    match = _TABLE_HEADER.fullmatch(line.rstrip("\r\n"))
    if match is None:
        return None
    if len(match.group("open")) != len(match.group("close")):
        return None
    return match.group("path")


def prepare_sbom_pyproject(source: str) -> str:
    """Remove every ``tool.poetry`` table while preserving PEP 621 metadata."""
    parsed_source = tomllib.loads(source)
    project = parsed_source.get("project")
    if not isinstance(project, Mapping) or not project.get("name"):
        raise ValueError("release pyproject must contain non-empty PEP 621 project metadata")

    rendered: list[str] = []
    omit_table = False
    for line in source.splitlines(keepends=True):
        table_path = _table_path(line)
        if table_path is not None:
            omit_table = table_path == "tool.poetry" or table_path.startswith("tool.poetry.")
        if not omit_table:
            rendered.append(line)

    prepared = "".join(rendered)
    parsed_prepared = tomllib.loads(prepared)
    if parsed_prepared.get("project") != project:
        raise ValueError("prepared SBOM manifest changed the reviewed PEP 621 project metadata")
    tool = parsed_prepared.get("tool", {})
    if isinstance(tool, Mapping) and "poetry" in tool:
        raise ValueError("prepared SBOM manifest still contains tool.poetry metadata")
    return prepared


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.source.resolve() == args.output.resolve():
        raise ValueError("SBOM manifest output must differ from its source")
    prepared = prepare_sbom_pyproject(args.source.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(prepared, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
