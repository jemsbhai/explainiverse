"""Reproduce the sealed CPython 3.13.15 Windows runtime manifest.

This build-time helper accepts only the official python.org embeddable AMD64
archive.  The resulting manifest binds every extracted file directly to the
reviewed archive bytes; it is never derived from a previously installed tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, Sequence

ARCHIVE_FILENAME = "python-3.13.15-embed-amd64.zip"
ARCHIVE_BYTES = 11_009_825
ARCHIVE_SHA256 = "d1f04d990aee1253d8569e8e5104e30fa9f5fa830899f14843448872d936a2cf"
ARCHIVE_URL = "https://www.python.org/ftp/python/3.13.15/python-3.13.15-embed-amd64.zip"
PYTHON_VERSION = "3.13.15"
PTH_FILENAME = "python313._pth"
EXPECTED_PTH = (
    b"python313.zip\r\n.\r\n\r\n# Uncomment to run site.main() automatically\r\n#import site\r\n"
)


def _fail(code: str) -> NoReturn:
    raise ValueError(code)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("ascii")


def _safe_root_file(name: str) -> str:
    if "\\" in name or "\x00" in name or name.startswith("/"):
        _fail("python_archive_member_path_rejected")
    path = PurePosixPath(name)
    if name != path.as_posix() or len(path.parts) != 1 or path.name in {"", ".", ".."}:
        _fail("python_archive_member_path_rejected")
    return name


def build_manifest(archive_path: Path) -> dict[str, Any]:
    archive_path = archive_path.resolve(strict=True)
    if (
        archive_path.name != ARCHIVE_FILENAME
        or not archive_path.is_file()
        or archive_path.is_symlink()
    ):
        _fail("python_archive_path_rejected")
    archive_raw = archive_path.read_bytes()
    if len(archive_raw) != ARCHIVE_BYTES or _sha256(archive_raw) != ARCHIVE_SHA256:
        _fail("python_archive_digest_rejected")
    files: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(archive_path) as archive:
        names: set[str] = set()
        for info in archive.infolist():
            name = _safe_root_file(info.filename.rstrip("/"))
            if info.filename in names:
                _fail("python_archive_member_duplicate")
            names.add(info.filename)
            mode = (info.external_attr >> 16) & 0xFFFF
            if info.is_dir() or (mode and stat.S_ISLNK(mode)):
                _fail("python_archive_member_rejected")
            raw = archive.read(info)
            files[name] = {"bytes": len(raw), "sha256": _sha256(raw)}
    if len(files) != 34 or files.get(PTH_FILENAME, {}).get("sha256") != _sha256(EXPECTED_PTH):
        _fail("python_archive_inventory_rejected")
    return {
        "schema_version": 1,
        "kind": "explainiverse-operator-python-3.13.15-embed-amd64-manifest",
        "archive": {
            "bytes": ARCHIVE_BYTES,
            "filename": ARCHIVE_FILENAME,
            "sha256": ARCHIVE_SHA256,
            "source_url": ARCHIVE_URL,
        },
        "target": {
            "implementation": "CPython",
            "platform": "win_amd64",
            "python_version": PYTHON_VERSION,
        },
        "files": {name: files[name] for name in sorted(files)},
        "directories": [],
        "startup": {
            "pth_filename": PTH_FILENAME,
            "pth_sha256": _sha256(EXPECTED_PTH),
            "site_import_enabled": False,
        },
        "untracked_files_or_directories_allowed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve(strict=False)
    if output.exists() or not output.is_absolute():
        _fail("python_manifest_output_must_be_new_absolute_path")
    output.write_bytes(_canonical(build_manifest(args.archive)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
