"""Compare two independently built Python distribution inventories byte-for-byte."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

_DISTRIBUTION_SUFFIXES = (".whl", ".tar.gz")


class DistributionComparisonError(ValueError):
    """A failed comparison that retains both artifact inventories."""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution_inventory(directory: Path) -> dict[str, dict[str, Any]]:
    """Return the release artifacts in *directory* with size and SHA-256."""
    if not directory.is_dir():
        raise ValueError(f"distribution directory does not exist: {directory}")

    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.name.endswith(_DISTRIBUTION_SUFFIXES)
    )
    if not paths:
        raise ValueError(f"no wheel or source distribution found in {directory}")

    wheels = [path for path in paths if path.name.endswith(".whl")]
    source_distributions = [path for path in paths if path.name.endswith(".tar.gz")]
    if len(wheels) != 1 or len(source_distributions) != 1:
        raise ValueError(
            "distribution directory must contain exactly one wheel and one source distribution: "
            f"{directory} has wheels={[path.name for path in wheels]!r}, "
            f"source_distributions={[path.name for path in source_distributions]!r}"
        )

    inventory: dict[str, dict[str, Any]] = {}
    for path in paths:
        if path.name in inventory:
            raise ValueError(f"duplicate distribution filename: {path.name}")
        inventory[path.name] = {"sha256": _sha256(path), "size": path.stat().st_size}
    return inventory


def compare_distribution_directories(first: Path, second: Path) -> dict[str, Any]:
    """Require equal filenames and byte-identical artifacts from two builds."""
    first_inventory = distribution_inventory(first)
    second_inventory = distribution_inventory(second)
    report: dict[str, Any] = {
        "schema_version": 1,
        "comparison": "byte-identical",
        "first": first_inventory,
        "second": second_inventory,
        "reproducible": first_inventory == second_inventory,
    }

    if set(first_inventory) != set(second_inventory):
        first_only = sorted(set(first_inventory) - set(second_inventory))
        second_only = sorted(set(second_inventory) - set(first_inventory))
        raise DistributionComparisonError(
            "distribution filename sets differ: "
            f"first_only={first_only}, second_only={second_only}; "
            f"report={json.dumps(report, sort_keys=True)}",
            report,
        )

    differing = [
        filename
        for filename in first_inventory
        if first_inventory[filename] != second_inventory[filename]
    ]
    if differing:
        raise DistributionComparisonError(
            "independent builds are not byte-identical for "
            f"{differing}; report={json.dumps(report, sort_keys=True)}",
            report,
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    failure: ValueError | None = None
    try:
        report = compare_distribution_directories(args.first, args.second)
    except DistributionComparisonError as exc:
        report = dict(exc.report)
        report["error"] = str(exc)
        failure = exc
    except ValueError as exc:
        report = {
            "schema_version": 1,
            "comparison": "byte-identical",
            "reproducible": False,
            "error": str(exc),
        }
        failure = exc

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if failure is not None:
        raise SystemExit(2) from failure


if __name__ == "__main__":
    main()
