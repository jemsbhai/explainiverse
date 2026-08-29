"""Install the exact wheel-derived operator runtime into a new directory.

This setup helper intentionally does not use pip for the live runtime tree. It
extracts only files listed in the reviewed manifest, excluding wheel RECORDs,
console scripts, pip, bytecode, and every generated installer artifact.
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

MAX_FILE_BYTES = 512 * 1024 * 1024
MANIFEST_SHA256 = "5a6282da0fd87317986b97da1725480c0877686f0e559a83520acf95f46d945f"
EXPECTED_ARCHIVES = {
    "cffi-2.1.1-cp313-cp313-win_amd64.whl": "1aa5645c30469b09530c4ebca77ebf8f17618293c58f8549cb1a543a50236e7d",
    "cryptography-50.0.0-cp311-abi3-win_amd64.whl": "bd1c592e4d5974f0d08d4888e432157adba757c66da0246918e43677fafa2d30",
    "pycparser-3.0-py3-none-any.whl": "b727414169a36b7d524c1c3e31839a521725078d7b2ff038656844266160a992",
    "pywin32-311-cp313-cp313-win_amd64.whl": "718a38f7e5b058e76aee1c56ddd06908116d35147e133427e59a3983f703a20d",
}


def _fail(code: str) -> NoReturn:
    raise ValueError(code)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("ascii")


def _safe_relative(value: str) -> Path:
    pure = PurePosixPath(value)
    if (
        not value
        or value != pure.as_posix()
        or value.startswith("/")
        or "\\" in value
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        _fail("runtime_manifest_path_rejected")
    return Path(*pure.parts)


def _current_user_sid() -> str:
    from ctypes import wintypes

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)  # type: ignore[attr-defined]
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
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
        _fail("runtime_owner_token_open_failed")
    try:
        needed = wintypes.DWORD()
        get_information(token, 1, None, 0, ctypes.byref(needed))
        if needed.value <= 0 or needed.value > 65536:
            _fail("runtime_owner_token_size_rejected")
        buffer = ctypes.create_string_buffer(needed.value)
        if not get_information(token, 1, buffer, needed, ctypes.byref(needed)):
            _fail("runtime_owner_token_query_failed")
        sid_pointer = ctypes.cast(buffer, ctypes.POINTER(wintypes.LPVOID))[0]
        sid_text = wintypes.LPWSTR()
        if not convert_sid(sid_pointer, ctypes.byref(sid_text)):
            _fail("runtime_owner_sid_conversion_failed")
        try:
            value = sid_text.value
        finally:
            local_free(sid_text)
        if not value or not value.startswith("S-1-"):
            _fail("runtime_owner_sid_rejected")
        return value
    finally:
        close_handle(token)


def _harden_windows_directory(path: Path) -> None:
    if os.name != "nt":
        _fail("runtime_installer_windows_only")
    system_root = os.environ.get("SYSTEMROOT")
    if not system_root:
        _fail("runtime_system_root_missing")
    icacls = Path(system_root) / "System32" / "icacls.exe"
    try:
        icacls = icacls.resolve(strict=True)
    except OSError:
        _fail("runtime_icacls_unavailable")
    if not icacls.is_file() or icacls.is_symlink():
        _fail("runtime_icacls_rejected")
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
        _fail("runtime_owner_private_owner_create_failed")
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
        _fail("runtime_owner_private_acl_create_failed")


def install(wheelhouse: Path, manifest_path: Path, output: Path) -> dict[str, Any]:
    wheelhouse = wheelhouse.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    output = output.resolve(strict=False)
    if (
        not wheelhouse.is_dir()
        or wheelhouse.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.is_symlink()
        or not output.is_absolute()
        or output.exists()
    ):
        _fail("runtime_install_path_rejected")
    manifest_raw = manifest_path.read_bytes()
    if _sha256(manifest_raw) != MANIFEST_SHA256:
        _fail("runtime_manifest_digest_rejected")
    try:
        manifest = json.loads(manifest_raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail("runtime_manifest_json_rejected")
    if (
        type(manifest) is not dict
        or manifest_raw != _canonical(manifest)
        or manifest.get("schema_version") != 1
        or manifest.get("kind") != "explainiverse-operator-windows-cp313-site-manifest"
        or manifest.get("target")
        != {
            "implementation": "CPython",
            "platform": "win_amd64",
            "python_major_minor": "3.13",
            "site_processing_disabled_at_startup": True,
        }
        or manifest.get("bytecode_allowed") is not False
        or manifest.get("untracked_files_or_directories_allowed") is not False
    ):
        _fail("runtime_manifest_not_canonical")
    archives = manifest.get("archives")
    files = manifest.get("files")
    directories = manifest.get("directories")
    if type(archives) is not list or type(files) is not dict or type(directories) is not list:
        _fail("runtime_manifest_schema_rejected")
    if {item.get("filename"): item.get("sha256") for item in archives} != EXPECTED_ARCHIVES:
        _fail("runtime_manifest_archive_set_rejected")
    archive_paths = {path.name: path for path in wheelhouse.glob("*.whl")}
    expected_archive_names = {item["filename"] for item in archives}
    if set(archive_paths) != expected_archive_names:
        _fail("runtime_wheel_archive_set_rejected")
    opened: dict[str, zipfile.ZipFile] = {}
    created = False
    try:
        for item in archives:
            path = archive_paths[item["filename"]]
            raw = path.read_bytes()
            if _sha256(raw) != item["sha256"]:
                _fail("runtime_wheel_archive_digest_rejected")
            opened[item["filename"]] = zipfile.ZipFile(path)
        os.mkdir(output, 0o700)
        created = True
        _harden_windows_directory(output)
        for relative in sorted(directories, key=lambda value: (value.count("/"), value)):
            destination = output / _safe_relative(relative)
            os.mkdir(destination, 0o700)
        rows: list[bytes] = []
        for relative, expected in sorted(files.items()):
            destination = output / _safe_relative(relative)
            archive = opened[expected["archive"]]
            try:
                info = archive.getinfo(relative)
            except KeyError:
                _fail("runtime_wheel_member_missing")
            mode = (info.external_attr >> 16) & 0xFFFF
            if info.is_dir() or (mode and stat.S_ISLNK(mode)) or info.file_size > MAX_FILE_BYTES:
                _fail("runtime_wheel_member_rejected")
            raw = archive.read(info)
            if len(raw) != expected["bytes"] or _sha256(raw) != expected["sha256"]:
                _fail("runtime_wheel_member_digest_rejected")
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
            descriptor = os.open(destination, flags, 0o600)
            try:
                view = memoryview(raw)
                try:
                    offset = 0
                    while offset < len(view):
                        written = os.write(descriptor, view[offset:])
                        if written <= 0:
                            _fail("runtime_install_short_write")
                        offset += written
                    os.fsync(descriptor)
                finally:
                    view.release()
            finally:
                os.close(descriptor)
            rows.append(f"{relative}\t{len(raw)}\t{_sha256(raw)}\n".encode("utf-8"))
        return {
            "schema_version": 1,
            "kind": "explainiverse-operator-runtime-installed",
            "runtime_root": str(output),
            "manifest_sha256": _sha256(manifest_raw),
            "file_count": len(files),
            "directory_count": len(directories),
            "file_inventory_sha256": _sha256(b"".join(rows)),
            "owner_private_acl_applied_before_children": True,
            "pip_present_in_runtime": False,
            "record_files_present": False,
            "generated_scripts_present": False,
            "bytecode_present": False,
            "crash_recovery": "discard-partial-directory-and-create-a-new-path",
        }
    except BaseException:
        # A partial directory is intentionally left fail-closed for evidence.
        # It cannot be resumed or reused and the next attempt needs a new path.
        if created:
            pass
        raise
    finally:
        for archive in opened.values():
            archive.close()


def _write_install_receipt(directory: Path, value: dict[str, Any]) -> dict[str, Any]:
    directory = directory.resolve(strict=False)
    runtime_root = Path(str(value["runtime_root"]))
    if (
        not directory.is_absolute()
        or directory.exists()
        or directory == runtime_root
        or directory in runtime_root.parents
        or runtime_root in directory.parents
    ):
        _fail("runtime_install_receipt_directory_rejected")
    os.mkdir(directory, 0o700)
    _harden_windows_directory(directory)
    raw = _canonical(value)
    path = directory / "site-install-receipt.json"
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
                    _fail("runtime_install_receipt_short_write")
                offset += written
            os.fsync(descriptor)
        finally:
            view.release()
    finally:
        os.close(descriptor)
    return {
        "schema_version": 1,
        "kind": "explainiverse-operator-site-install-receipt-published",
        "receipt_directory": str(directory),
        "receipt_path": str(path),
        "receipt_sha256": _sha256(raw),
        "receipt_bytes": len(raw),
        "receipt_no_replace": True,
        "receipt_directory_owner_private_before_write": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheelhouse", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt-directory", required=True, type=Path)
    args = parser.parse_args(argv)
    installed = install(args.wheelhouse, args.manifest, args.output)
    sys_output = _canonical(_write_install_receipt(args.receipt_directory, installed))
    os.write(1, sys_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
