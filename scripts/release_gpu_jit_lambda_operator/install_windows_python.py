"""Extract the exact pinned CPython embeddable runtime into a new directory.

This is a pre-secret setup helper.  It accepts only the reviewed official
archive, writes only the archive-derived manifest members with no-replace
semantics, and protects the new directory before creating its children.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import stat
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, Sequence

ARCHIVE_FILENAME = "python-3.13.15-embed-amd64.zip"
ARCHIVE_BYTES = 11_009_825
ARCHIVE_SHA256 = "d1f04d990aee1253d8569e8e5104e30fa9f5fa830899f14843448872d936a2cf"
MANIFEST_SHA256 = "e2d965a1f8b09d1e5f0349133dfd869eceb92cf730f54a456a4f79bb22d5a519"
MAX_FILE_BYTES = 512 * 1024 * 1024


def _fail(code: str) -> NoReturn:
    raise ValueError(code)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("ascii")


def _safe_root_file(value: str) -> Path:
    pure = PurePosixPath(value)
    if (
        not value
        or value != pure.as_posix()
        or value.startswith("/")
        or "\\" in value
        or len(pure.parts) != 1
        or pure.name in {"", ".", ".."}
    ):
        _fail("python_manifest_path_rejected")
    return Path(pure.name)


def _current_user_sid() -> str:
    from ctypes import wintypes

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_token = advapi32.OpenProcessToken
    open_token.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    open_token.restype = wintypes.BOOL
    get_information = advapi32.GetTokenInformation
    get_information.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    get_information.restype = wintypes.BOOL
    convert_sid = advapi32.ConvertSidToStringSidW
    convert_sid.argtypes = [wintypes.LPVOID, ctypes.POINTER(wintypes.LPWSTR)]
    convert_sid.restype = wintypes.BOOL
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.argtypes = []
    get_current_process.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    local_free = kernel32.LocalFree
    local_free.argtypes = [wintypes.HLOCAL]
    local_free.restype = wintypes.HLOCAL
    token = wintypes.HANDLE()
    if not open_token(get_current_process(), 0x0008, ctypes.byref(token)):
        _fail("python_owner_token_open_failed")
    try:
        needed = wintypes.DWORD()
        get_information(token, 1, None, 0, ctypes.byref(needed))
        if needed.value <= 0 or needed.value > 65536:
            _fail("python_owner_token_size_rejected")
        buffer = ctypes.create_string_buffer(needed.value)
        if not get_information(token, 1, buffer, needed, ctypes.byref(needed)):
            _fail("python_owner_token_query_failed")
        sid_pointer = ctypes.cast(buffer, ctypes.POINTER(wintypes.LPVOID))[0]
        sid_text = wintypes.LPWSTR()
        if not convert_sid(sid_pointer, ctypes.byref(sid_text)):
            _fail("python_owner_sid_conversion_failed")
        try:
            value = sid_text.value
        finally:
            local_free(sid_text)
        if not value or not value.startswith("S-1-"):
            _fail("python_owner_sid_rejected")
        return value
    finally:
        close_handle(token)


def _harden_windows_directory(path: Path) -> None:
    if os.name != "nt":
        _fail("python_installer_windows_only")
    system_root = os.environ.get("SYSTEMROOT")
    if not system_root:
        _fail("python_system_root_missing")
    icacls = Path(system_root) / "System32" / "icacls.exe"
    try:
        icacls = icacls.resolve(strict=True)
    except OSError:
        _fail("python_icacls_unavailable")
    if not icacls.is_file() or icacls.is_symlink():
        _fail("python_icacls_rejected")
    owner_sid = _current_user_sid()
    owner = subprocess.run(
        [str(icacls), str(path), "/setowner", f"*{owner_sid}"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        shell=False,
        check=False,
    )
    if owner.returncode != 0 or owner.stderr:
        _fail("python_owner_private_owner_create_failed")
    completed = subprocess.run(
        [
            str(icacls),
            str(path),
            "/inheritance:r",
            "/grant:r",
            f"*{owner_sid}:(OI)(CI)F",
            "*S-1-5-18:(OI)(CI)F",
            "*S-1-5-32-544:(OI)(CI)F",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        shell=False,
        check=False,
    )
    if completed.returncode != 0 or completed.stderr:
        _fail("python_owner_private_acl_create_failed")


def install(archive_path: Path, manifest_path: Path, output: Path) -> dict[str, Any]:
    archive_path = archive_path.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    output = output.resolve(strict=False)
    if (
        archive_path.name != ARCHIVE_FILENAME
        or not archive_path.is_file()
        or archive_path.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.is_symlink()
        or not output.is_absolute()
        or output.exists()
    ):
        _fail("python_install_path_rejected")
    archive_raw = archive_path.read_bytes()
    if len(archive_raw) != ARCHIVE_BYTES or _sha256(archive_raw) != ARCHIVE_SHA256:
        _fail("python_archive_digest_rejected")
    manifest_raw = manifest_path.read_bytes()
    if _sha256(manifest_raw) != MANIFEST_SHA256:
        _fail("python_manifest_digest_rejected")
    try:
        manifest = json.loads(manifest_raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail("python_manifest_json_rejected")
    files = manifest.get("files") if type(manifest) is dict else None
    if (
        type(manifest) is not dict
        or manifest_raw != _canonical(manifest)
        or type(files) is not dict
        or manifest.get("archive")
        != {
            "bytes": ARCHIVE_BYTES,
            "filename": ARCHIVE_FILENAME,
            "sha256": ARCHIVE_SHA256,
            "source_url": (
                "https://www.python.org/ftp/python/3.13.15/" "python-3.13.15-embed-amd64.zip"
            ),
        }
        or manifest.get("directories") != []
        or manifest.get("untracked_files_or_directories_allowed") is not False
    ):
        _fail("python_manifest_schema_rejected")
    opened = zipfile.ZipFile(archive_path)
    created = False
    try:
        archive_names = {item.filename for item in opened.infolist()}
        if archive_names != set(files):
            _fail("python_archive_member_set_rejected")
        os.mkdir(output, 0o700)
        created = True
        _harden_windows_directory(output)
        rows: list[bytes] = []
        for relative, expected in sorted(files.items()):
            destination = output / _safe_root_file(relative)
            try:
                info = opened.getinfo(relative)
            except KeyError:
                _fail("python_archive_member_missing")
            mode = (info.external_attr >> 16) & 0xFFFF
            if info.is_dir() or (mode and stat.S_ISLNK(mode)) or info.file_size > MAX_FILE_BYTES:
                _fail("python_archive_member_rejected")
            raw = opened.read(info)
            if len(raw) != expected.get("bytes") or _sha256(raw) != expected.get("sha256"):
                _fail("python_archive_member_digest_rejected")
            descriptor = os.open(
                destination,
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
                            _fail("python_install_short_write")
                        offset += written
                    os.fsync(descriptor)
                finally:
                    view.release()
            finally:
                os.close(descriptor)
            rows.append(f"{relative}\t{len(raw)}\t{_sha256(raw)}\n".encode("utf-8"))
        return {
            "schema_version": 1,
            "kind": "explainiverse-operator-python-runtime-installed",
            "python_runtime_root": str(output),
            "archive_sha256": ARCHIVE_SHA256,
            "manifest_sha256": MANIFEST_SHA256,
            "file_count": len(files),
            "directory_count": 0,
            "file_inventory_sha256": _sha256(b"".join(rows)),
            "owner_private_acl_applied_before_children": True,
            "site_processing_disabled_by_embeddable_pth": True,
            "untracked_files_or_directories_present": False,
            "crash_recovery": "discard-partial-directory-and-create-a-new-path",
        }
    except BaseException:
        # A partial path is never deleted, resumed, or reused after ambiguity.
        if created:
            pass
        raise
    finally:
        opened.close()


def _write_install_receipt(directory: Path, value: dict[str, Any]) -> dict[str, Any]:
    directory = directory.resolve(strict=False)
    runtime_root = Path(str(value["python_runtime_root"]))
    if (
        not directory.is_absolute()
        or directory.exists()
        or directory == runtime_root
        or directory in runtime_root.parents
        or runtime_root in directory.parents
    ):
        _fail("python_install_receipt_directory_rejected")
    os.mkdir(directory, 0o700)
    _harden_windows_directory(directory)
    raw = _canonical(value)
    path = directory / "python-install-receipt.json"
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
                    _fail("python_install_receipt_short_write")
                offset += written
            os.fsync(descriptor)
        finally:
            view.release()
    finally:
        os.close(descriptor)
    return {
        "schema_version": 1,
        "kind": "explainiverse-operator-python-install-receipt-published",
        "receipt_directory": str(directory),
        "receipt_path": str(path),
        "receipt_sha256": _sha256(raw),
        "receipt_bytes": len(raw),
        "receipt_no_replace": True,
        "receipt_directory_owner_private_before_write": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt-directory", required=True, type=Path)
    args = parser.parse_args(argv)
    installed = install(args.archive, args.manifest, args.output)
    os.write(1, _canonical(_write_install_receipt(args.receipt_directory, installed)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
