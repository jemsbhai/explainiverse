"""Create a PEP 621-only project view for the release SBOM generator.

CycloneDX Python treats any ``tool.poetry`` table as legacy package metadata,
even when Poetry 2 uses PEP 621's ``project`` table and ``tool.poetry`` contains
only build or dependency-group configuration.  The release SBOM must retain
the reviewed PEP 621 metadata while hiding that incompatible tool namespace.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised by the Python 3.10 CI lane
    import tomli as tomllib

_TABLE_SENTINEL = "__explainiverse_sbom_table_sentinel_6f6c45f1__"


def _table_path(line: str) -> tuple[str, ...] | None:
    """Parse one TOML table header with the standard library's full grammar."""
    if not line.lstrip().startswith("["):
        return None
    header = line.rstrip("\r\n")
    candidate = f"{header}\n{_TABLE_SENTINEL} = true\n"
    try:
        parsed = tomllib.loads(candidate)
    except tomllib.TOMLDecodeError:
        return None

    path: list[str] = []
    node: Any = parsed
    while isinstance(node, Mapping):
        if node.get(_TABLE_SENTINEL) is True:
            return tuple(path)
        if len(node) != 1:
            return None
        key, node = next(iter(node.items()))
        if not isinstance(key, str):  # pragma: no cover - TOML keys are strings
            return None
        path.append(key)
        if isinstance(node, list):
            if len(node) != 1:
                return None
            node = node[0]
    return None


def _without_poetry(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return parsed TOML data with only the ``tool.poetry`` subtree removed."""
    normalized = dict(data)
    tool = data.get("tool")
    if isinstance(tool, Mapping):
        remaining_tool = {key: value for key, value in tool.items() if key != "poetry"}
        if remaining_tool:
            normalized["tool"] = remaining_tool
        else:
            normalized.pop("tool", None)
    return normalized


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
            omit_table = table_path[:2] == ("tool", "poetry")
        if not omit_table:
            rendered.append(line)

    prepared = "".join(rendered)
    parsed_prepared = tomllib.loads(prepared)
    if _without_poetry(parsed_prepared) != _without_poetry(parsed_source):
        raise ValueError("prepared SBOM manifest changed reviewed non-Poetry metadata")
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
