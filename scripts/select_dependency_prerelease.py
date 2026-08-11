"""Select the latest published prerelease in the next dependency major."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

from packaging.version import InvalidVersion, Version


def select_next_major_prerelease(metadata: Mapping[str, Any], *, current_major: int) -> str | None:
    """Return the latest non-yanked prerelease in ``current_major + 1``."""
    releases = metadata.get("releases")
    if not isinstance(releases, Mapping):
        raise ValueError("PyPI metadata has no releases mapping")
    candidates: list[Version] = []
    for raw_version, raw_files in releases.items():
        try:
            version = Version(str(raw_version))
        except InvalidVersion:
            continue
        if version.major != current_major + 1 or not version.is_prerelease:
            continue
        if not isinstance(raw_files, Sequence) or isinstance(raw_files, (str, bytes)):
            raise ValueError(f"PyPI files for {raw_version!r} are malformed")
        if not any(
            isinstance(file, Mapping) and not file.get("yanked", False) for file in raw_files
        ):
            continue
        candidates.append(version)
    if not candidates:
        return None
    return str(max(candidates))


def _load_metadata(package: str, metadata_path: Path | None) -> Mapping[str, Any]:
    if metadata_path is not None:
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        request = urllib.request.Request(
            f"https://pypi.org/pypi/{package}/json",
            headers={"User-Agent": "explainiverse-dependency-prerelease-probe"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                value = json.load(response)
        except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"could not read PyPI metadata for {package!r}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("PyPI metadata must be a JSON object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--current-major", type=int, required=True)
    parser.add_argument("--metadata", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.current_major < 0:
            raise ValueError("current-major must be non-negative")
        metadata = _load_metadata(args.package, args.metadata)
        selected = select_next_major_prerelease(metadata, current_major=args.current_major)
        if selected is None:
            print(
                f"no non-yanked {args.current_major + 1}.x prerelease exists for {args.package}",
                file=sys.stderr,
            )
            return 3
        print(selected)
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
