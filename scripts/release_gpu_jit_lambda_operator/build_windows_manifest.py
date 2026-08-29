"""Reproduce the sealed CPython 3.13 Windows operator site manifest.

This build-time helper accepts only the four exact retained runtime wheels. It
derives the importable site-packages bytes directly from those archives; it
never trusts an installed environment or its mutable ``RECORD`` files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, Sequence

ARCHIVES = {
    "cffi-2.1.1-cp313-cp313-win_amd64.whl": (
        "cffi",
        "2.1.1",
        "1aa5645c30469b09530c4ebca77ebf8f17618293c58f8549cb1a543a50236e7d",
    ),
    "cryptography-50.0.0-cp311-abi3-win_amd64.whl": (
        "cryptography",
        "50.0.0",
        "bd1c592e4d5974f0d08d4888e432157adba757c66da0246918e43677fafa2d30",
    ),
    "pycparser-3.0-py3-none-any.whl": (
        "pycparser",
        "3.0",
        "b727414169a36b7d524c1c3e31839a521725078d7b2ff038656844266160a992",
    ),
    "pywin32-311-cp313-cp313-win_amd64.whl": (
        "pywin32",
        "311",
        "718a38f7e5b058e76aee1c56ddd06908116d35147e133427e59a3983f703a20d",
    ),
}


def _fail(code: str) -> NoReturn:
    raise ValueError(code)


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("ascii")


def _safe_member(value: str) -> PurePosixPath:
    if "\\" in value or value.startswith("/") or "\x00" in value:
        _fail("wheel_member_path_rejected")
    path = PurePosixPath(value)
    if not value or any(part in {"", ".", ".."} for part in path.parts):
        _fail("wheel_member_path_rejected")
    return path


def build_manifest(
    wheelhouse: Path,
    *,
    requirements_path: Path,
    bootstrap_requirements_path: Path,
) -> dict[str, Any]:
    wheelhouse = wheelhouse.resolve(strict=True)
    if not wheelhouse.is_dir() or wheelhouse.is_symlink():
        _fail("wheelhouse_rejected")
    observed = {path.name: path for path in wheelhouse.glob("*.whl")}
    if set(observed) != set(ARCHIVES):
        _fail("wheel_archive_set_rejected")
    files: dict[str, dict[str, Any]] = {}
    archives: list[dict[str, Any]] = []
    for filename, (distribution, version, expected_sha256) in sorted(ARCHIVES.items()):
        wheel = observed[filename]
        wheel_raw = wheel.read_bytes()
        if _digest(wheel_raw) != expected_sha256:
            _fail("wheel_archive_digest_rejected")
        archives.append(
            {
                "distribution": distribution,
                "filename": filename,
                "sha256": expected_sha256,
                "version": version,
            }
        )
        with zipfile.ZipFile(wheel) as archive:
            names: set[str] = set()
            record_count = 0
            for info in archive.infolist():
                path = _safe_member(info.filename.rstrip("/"))
                if info.filename in names:
                    _fail("wheel_member_duplicate")
                names.add(info.filename)
                mode = (info.external_attr >> 16) & 0xFFFF
                if mode and stat.S_ISLNK(mode):
                    _fail("wheel_member_symlink_rejected")
                if info.is_dir():
                    continue
                raw = archive.read(info)
                path_text = path.as_posix()
                if path_text.endswith(".dist-info/RECORD"):
                    record_count += 1
                    continue
                data_index = next(
                    (index for index, part in enumerate(path.parts) if part.endswith(".data")),
                    None,
                )
                if data_index is not None:
                    suffix = path.parts[data_index + 1 :]
                    if len(suffix) < 2 or suffix[0] != "scripts":
                        _fail("wheel_data_scheme_rejected")
                    continue
                if path_text in files:
                    _fail("site_manifest_target_duplicate")
                files[path_text] = {
                    "archive": filename,
                    "bytes": len(raw),
                    "sha256": _digest(raw),
                }
            if record_count != 1:
                _fail("wheel_record_cardinality_rejected")
    directories = sorted(
        {
            PurePosixPath(path).parents[index].as_posix()
            for path in files
            for index in range(len(PurePosixPath(path).parents) - 1)
        }
    )
    archive_set_sha256 = _digest(_canonical(archives))
    return {
        "schema_version": 1,
        "kind": "explainiverse-operator-windows-cp313-site-manifest",
        "target": {
            "implementation": "CPython",
            "python_major_minor": "3.13",
            "platform": "win_amd64",
            "site_processing_disabled_at_startup": True,
        },
        "archives": archives,
        "archive_set_sha256": archive_set_sha256,
        "requirements": {
            "runtime_sha256": _digest(requirements_path.resolve(strict=True).read_bytes()),
            "bootstrap_sha256": _digest(
                bootstrap_requirements_path.resolve(strict=True).read_bytes()
            ),
        },
        "files": {name: files[name] for name in sorted(files)},
        "directories": directories,
        "bytecode_allowed": False,
        "untracked_files_or_directories_allowed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheelhouse", required=True, type=Path)
    parser.add_argument("--requirements", required=True, type=Path)
    parser.add_argument("--bootstrap-requirements", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve(strict=False)
    if output.exists() or not output.is_absolute():
        _fail("manifest_output_must_be_new_absolute_path")
    output.write_bytes(
        _canonical(
            build_manifest(
                args.wheelhouse,
                requirements_path=args.requirements,
                bootstrap_requirements_path=args.bootstrap_requirements,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
