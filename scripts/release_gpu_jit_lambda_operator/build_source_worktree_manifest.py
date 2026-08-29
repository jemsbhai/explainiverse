"""Build the operator's exact worktree manifest from the staged Git index.

This credential-free review helper is run only after the exact positive
release-candidate allowlist has been staged.  Runtime code does not invoke Git;
it consumes the canonical manifest produced here and a digest frozen into the
reviewed byte-sealed preloader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, Sequence

GIT_PATH = Path(r"C:\Program Files\Git\cmd\git.exe")
GIT_SHA256 = "d90e36cafd656d52984f7546bfcb5b065d73e2e66957c952b7a4a1cd260e8f36"
GIT_RUNTIME_PATH = Path(r"C:\Program Files\Git\mingw64\bin\git.exe")
GIT_RUNTIME_SHA256 = "3591764e521c340b8cca2ca300b3ce265df271ac41d2b338113c9a76fb32bcaa"
MANIFEST_RELATIVE = "scripts/release_gpu_jit_lambda_operator/source-worktree-manifest.json"
PRELOADER_RELATIVE = "scripts/release_gpu_jit_lambda_operator/preloader.py"
EXCLUDED_PATHS = (MANIFEST_RELATIVE, PRELOADER_RELATIVE)
MAX_BLOB_BYTES = 512 * 1024 * 1024


def _fail(code: str) -> NoReturn:
    raise ValueError(code)


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("ascii")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git_blob_sha1(raw: bytes) -> str:
    return hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()


def _bound_git() -> Path:
    try:
        path = GIT_PATH.resolve(strict=True)
    except OSError:
        _fail("source_manifest_git_unavailable")
    if path != GIT_PATH or not path.is_file() or path.is_symlink():
        _fail("source_manifest_git_path_rejected")
    raw = path.read_bytes()
    if _sha256(raw) != GIT_SHA256:
        _fail("source_manifest_git_digest_rejected")
    try:
        runtime_path = GIT_RUNTIME_PATH.resolve(strict=True)
    except OSError:
        _fail("source_manifest_git_runtime_unavailable")
    if (
        runtime_path != GIT_RUNTIME_PATH
        or not runtime_path.is_file()
        or runtime_path.is_symlink()
        or _sha256(runtime_path.read_bytes()) != GIT_RUNTIME_SHA256
    ):
        _fail("source_manifest_git_runtime_identity_rejected")
    return path


def _git(root: Path, arguments: Sequence[str], *, maximum: int) -> bytes:
    executable = _bound_git()
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in {"COMSPEC", "SYSTEMROOT", "TEMP", "TMP", "WINDIR"}
    }
    environment.update(
        {
            "GIT_CONFIG_COUNT": "2",
            "GIT_CONFIG_KEY_0": "core.fsmonitor",
            "GIT_CONFIG_VALUE_0": "false",
            "GIT_CONFIG_KEY_1": "core.untrackedCache",
            "GIT_CONFIG_VALUE_1": "false",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PAGER": "",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    completed = subprocess.run(
        [str(executable), "--no-pager", *arguments],
        cwd=root,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        shell=False,
        check=False,
    )
    if completed.returncode != 0 or completed.stderr or len(completed.stdout) > maximum:
        _fail("source_manifest_git_command_rejected")
    return completed.stdout


def _index(root: Path) -> dict[str, dict[str, Any]]:
    raw = _git(root, ("ls-files", "--stage", "-z"), maximum=32 * 1024 * 1024)
    result: dict[str, dict[str, Any]] = {}
    for row in raw.split(b"\0"):
        if not row:
            continue
        try:
            metadata, path_raw = row.split(b"\t", 1)
            mode, object_sha, stage = metadata.decode("ascii").split(" ")
            relative = path_raw.decode("utf-8", errors="strict")
        except (UnicodeDecodeError, ValueError):
            _fail("source_manifest_index_parse_rejected")
        pure = PurePosixPath(relative)
        if (
            stage != "0"
            or mode not in {"100644", "100755"}
            or relative != pure.as_posix()
            or relative.startswith("/")
            or any(part in {"", ".", ".."} for part in pure.parts)
            or relative in result
        ):
            _fail("source_manifest_index_entry_rejected")
        result[relative] = {"mode": mode, "git_blob_sha": object_sha}
    if not result or PRELOADER_RELATIVE not in result:
        _fail("source_manifest_index_required_path_missing")
    return result


def build(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        _fail("source_manifest_repository_root_rejected")
    indexed = _index(root)
    files: dict[str, dict[str, Any]] = {}
    directories: set[str] = set()
    for relative, metadata in sorted(indexed.items()):
        if relative in EXCLUDED_PATHS:
            continue
        raw = _git(root, ("show", f":{relative}"), maximum=MAX_BLOB_BYTES)
        if _git_blob_sha1(raw) != metadata["git_blob_sha"]:
            _fail("source_manifest_index_blob_mismatch")
        files[relative] = {
            "mode": metadata["mode"],
            "bytes": len(raw),
            "sha256": _sha256(raw),
            "git_blob_sha": metadata["git_blob_sha"],
        }
        parent = PurePosixPath(relative).parent
        while parent != PurePosixPath("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    rows = [
        f"{name}\t{item['mode']}\t{item['bytes']}\t{item['sha256']}\t{item['git_blob_sha']}\n".encode(
            "utf-8"
        )
        for name, item in sorted(files.items())
    ]
    return {
        "schema_version": 1,
        "kind": "explainiverse-operator-source-worktree-manifest",
        "excluded_paths": list(EXCLUDED_PATHS),
        "files": files,
        "directories": sorted(directories),
        "file_count": len(files),
        "directory_count": len(directories),
        "file_inventory_sha256": _sha256(b"".join(rows)),
        "source": "exact-staged-index-blobs",
        "runtime_git_dependency": False,
    }


def _write_no_replace(path: Path, raw: bytes) -> None:
    path = path.resolve(strict=False)
    if not path.is_absolute() or path.exists():
        _fail("source_manifest_output_rejected")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o600,
    )
    try:
        view = memoryview(raw)
        try:
            offset = 0
            while offset < len(view):
                written = os.write(descriptor, view[offset:])
                if written <= 0:
                    _fail("source_manifest_short_write")
                offset += written
            os.fsync(descriptor)
        finally:
            view.release()
    finally:
        os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sealed-preloader-output", required=True, type=Path)
    args = parser.parse_args(argv)
    value = build(args.repository_root)
    raw = _canonical(value)
    _write_no_replace(args.output, raw)
    preloader_raw = _git(
        args.repository_root.resolve(strict=True),
        ("show", f":{PRELOADER_RELATIVE}"),
        maximum=MAX_BLOB_BYTES,
    )
    pattern = rb'SOURCE_MANIFEST_SHA256 = "[0-9a-f]{64}"\n'
    replacement = f'SOURCE_MANIFEST_SHA256 = "{_sha256(raw)}"\n'.encode("ascii")
    sealed_preloader, count = re.subn(pattern, replacement, preloader_raw)
    if count != 1:
        _fail("source_manifest_preloader_digest_slot_rejected")
    _write_no_replace(args.sealed_preloader_output, sealed_preloader)
    receipt = {
        "schema_version": 1,
        "kind": "explainiverse-operator-source-worktree-manifest-published",
        "manifest_path": str(args.output.resolve(strict=True)),
        "manifest_sha256": _sha256(raw),
        "manifest_bytes": len(raw),
        "manifest_no_replace": True,
        "sealed_preloader_path": str(args.sealed_preloader_output.resolve(strict=True)),
        "sealed_preloader_sha256": _sha256(sealed_preloader),
        "preloader_manifest_digest_replacement_count": 1,
        "preloader_no_replace": True,
    }
    os.write(1, _canonical(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
