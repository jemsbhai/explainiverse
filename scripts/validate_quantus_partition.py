"""Audit the exact pytest partition owned by the Quantus reference lane."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Sequence


def _is_quantus_marker(decorator: ast.expr) -> bool:
    return isinstance(decorator, ast.Attribute) and decorator.attr == "quantus_reference"


def _contains_quantus_import(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Import) and any(
            alias.name == "quantus" or alias.name.startswith("quantus.") for alias in child.names
        ):
            return True
        if (
            isinstance(child, ast.ImportFrom)
            and child.module is not None
            and (child.module == "quantus" or child.module.startswith("quantus."))
        ):
            return True
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "importorskip"
            and child.args
            and isinstance(child.args[0], ast.Constant)
            and child.args[0].value == "quantus"
        ):
            return True
    return False


def _called_names(node: ast.AST) -> set[str]:
    return {
        child.func.id
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
    }


def discover_partition(tests_root: Path) -> tuple[list[str], list[str]]:
    """Return marked node IDs and static import/marker contract violations."""
    marked: list[str] = []
    violations: list[str] = []
    for path in sorted(tests_root.rglob("test_*.py")):
        relative = path.relative_to(tests_root.parent).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except SyntaxError as exc:
            violations.append(f"{relative}: cannot parse: {exc}")
            continue
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        helper_imports = {
            name
            for name, node in functions.items()
            if name.startswith("_") and _contains_quantus_import(node)
        }
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)) and _contains_quantus_import(node):
                violations.append(f"{relative}: Quantus must not be imported at module scope")
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            is_marked = any(_is_quantus_marker(value) for value in node.decorator_list)
            node_id = f"{relative}::{node.name}"
            if is_marked:
                marked.append(node_id)
                if not (_contains_quantus_import(node) or (_called_names(node) & helper_imports)):
                    violations.append(
                        f"{node_id}: quantus_reference test has no direct or audited helper import"
                    )
            elif _contains_quantus_import(node) and node.name not in helper_imports:
                violations.append(f"{node_id}: Quantus use is missing quantus_reference marker")
            elif _called_names(node) & helper_imports:
                violations.append(
                    f"{node_id}: calls a Quantus-loading helper without quantus_reference marker"
                )
    return sorted(marked), sorted(violations)


def load_manifest(path: Path) -> list[str]:
    entries = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if entries != sorted(set(entries)):
        raise ValueError("Quantus reference manifest must be sorted and contain unique node IDs")
    for entry in entries:
        if not entry.startswith("tests/") or "::test_" not in entry:
            raise ValueError(f"invalid Quantus reference node ID {entry!r}")
    return entries


def validate_partition(tests_root: Path, manifest_path: Path) -> None:
    marked, violations = discover_partition(tests_root)
    manifest = load_manifest(manifest_path)
    if marked != manifest:
        missing = sorted(set(marked) - set(manifest))
        stale = sorted(set(manifest) - set(marked))
        if missing:
            violations.append(f"manifest is missing marked tests: {missing!r}")
        if stale:
            violations.append(f"manifest contains unmarked/stale tests: {stale!r}")
    if violations:
        raise ValueError("; ".join(violations))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-root", type=Path, default=Path("tests"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(".github/constraints/quantus-reference-tests.txt"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        validate_partition(args.tests_root, args.manifest)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
