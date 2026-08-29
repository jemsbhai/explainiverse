"""Isolated, byte-sealed preloader for every production operator process.

The supported caller is the pinned embeddable CPython runtime under
``-I -S -B -c <reviewed shim>``.  That tiny shim reads this file once through
one stable descriptor, checks the frozen digest, and executes those same
bytes.  This preloader then captures every release-stack Python module from a
clean exact Git tree into memory before importing project code.
"""

from __future__ import annotations

import builtins
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import platform
import runpy
import stat
import struct
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, NoReturn, Sequence, cast

EXPECTED_ORIGIN = "https://github.com/jemsbhai/explainiverse.git"
MARKER_NAME = "_EXPLAINIVERSE_OPERATOR_PRELOADER_RECEIPT"
SHIM_MARKER_NAME = "_EXPLAINIVERSE_OPERATOR_SHIM_RECEIPT"
RESOURCE_MARKER_NAME = "_EXPLAINIVERSE_OPERATOR_CAPTURED_RESOURCES"
PRELOADER_SHIM_SHA256 = "22bb14f6e5fed4e7c5456f62e11569c9e5a0846ad1428854e545b2e4c1c979aa"
TARGETS = {
    "operator": "scripts.release_gpu_jit_lambda_operator",
    "windows-launcher": "scripts.release_gpu_jit_lambda_operator.windows_launcher",
}
SOURCE_PREFIXES = (
    "scripts/release_gpu_jit_lambda_controller/",
    "scripts/release_gpu_jit_lambda_live/",
    "scripts/release_gpu_jit_lambda_operator/",
    "scripts/release_gpu_jit_lambda_runtime/",
)
SOURCE_EXACT = {
    ".github/release-control-policy.json",
    ".github/workflows/cuda-ci.yml",
    ".github/workflows/publish-pypi.yml",
    ".github/workflows/recover-github-release.yml",
    "poetry.lock",
    "pyproject.toml",
    "scripts/release_external_controls.py",
    "scripts/verify_release_recovery.py",
}
MAX_SOURCE_BYTES = 4 * 1024 * 1024
RUNTIME_BUNDLE_NAMES = ("__init__.py", "bootstrap.py", "executor.py", "runtime_contract.py")
PYTHON_ARCHIVE_SHA256 = "d1f04d990aee1253d8569e8e5104e30fa9f5fa830899f14843448872d936a2cf"
PYTHON_MANIFEST_SHA256 = "e2d965a1f8b09d1e5f0349133dfd869eceb92cf730f54a456a4f79bb22d5a519"
PYTHON_FILE_INVENTORY_SHA256 = "ea028b8d42b0231c116581c4184297900bd4c0152a54017127b822f10b9742d9"
SITE_MANIFEST_SHA256 = "5a6282da0fd87317986b97da1725480c0877686f0e559a83520acf95f46d945f"
SITE_FILE_INVENTORY_SHA256 = "2cf1cf52ad8d284fcc2e7790acaaa32f3e77a9f39fa717f8bc2a67bc83ba31fe"
SOURCE_MANIFEST_RELATIVE = "scripts/release_gpu_jit_lambda_operator/source-worktree-manifest.json"
PRELOADER_RELATIVE = "scripts/release_gpu_jit_lambda_operator/preloader.py"
# Replaced from the staged-index builder receipt before the candidate commit.
SOURCE_MANIFEST_SHA256 = "85ad3e3a15e6473f50d4e37a0bb7b5ae710ea8e8a0ab0ed2d4808c307151e813"


def _fail(code: str) -> NoReturn:
    payload = {
        "schema_version": 1,
        "kind": "explainiverse-operator-preloader-error",
        "stable_code": code,
        "secret_values_logged": False,
    }
    sys.stderr.buffer.write(
        (
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        ).encode("ascii")
    )
    sys.stderr.buffer.flush()
    raise SystemExit(2)


def _require(condition: bool, code: str) -> None:
    if not condition:
        _fail(code)


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("ascii")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha1_git_blob(raw: bytes) -> str:
    return hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()


def _runtime_bundle_sha256(files: Mapping[str, bytes]) -> str:
    _require(set(files) == set(RUNTIME_BUNDLE_NAMES), "preloader_runtime_bundle_set_rejected")
    digest = hashlib.sha256()
    for name in RUNTIME_BUNDLE_NAMES:
        content = files[name]
        encoded = name.encode("ascii")
        digest.update(struct.pack(">H", len(encoded)))
        digest.update(encoded)
        digest.update(struct.pack(">Q", len(content)))
        digest.update(content)
    return digest.hexdigest()


def _is_reparse(path: Path) -> bool:
    return bool(
        getattr(path.lstat(), "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _bound_file(path: Path, *, context: str, maximum: int) -> bytes:
    _require(path.is_absolute(), f"{context}_not_absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        _fail(f"{context}_unavailable")
    _require(
        path == resolved and path.is_file() and not path.is_symlink() and not _is_reparse(path),
        f"{context}_rejected",
    )
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        current = os.lstat(path)
        _require(
            stat.S_ISREG(before.st_mode)
            and before.st_nlink == 1
            and current.st_nlink == 1
            and (before.st_dev, before.st_ino) == (current.st_dev, current.st_ino),
            f"{context}_identity_rejected",
        )
        _require(0 <= before.st_size <= maximum, f"{context}_size_rejected")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            _require(bool(chunk), f"{context}_short_read")
            chunks.append(chunk)
            remaining -= len(chunk)
        _require(os.read(descriptor, 1) == b"", f"{context}_grew")
        after = os.fstat(descriptor)
        _require(
            (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            == (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ),
            f"{context}_changed",
        )
        return b"".join(chunks)
    except OSError:
        _fail(f"{context}_read_failed")
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _option(arguments: Sequence[str], name: str) -> str:
    values: list[str] = []
    for index, item in enumerate(arguments):
        if item == name:
            _require(index + 1 < len(arguments), "preloader_option_value_missing")
            values.append(arguments[index + 1])
        elif item.startswith(name + "="):
            _fail("preloader_option_equals_form_rejected")
    _require(len(values) == 1 and bool(values[0]), "preloader_option_cardinality_rejected")
    return values[0]


def _without_option(arguments: Sequence[str], name: str) -> list[str]:
    result: list[str] = []
    skip = False
    for item in arguments:
        if skip:
            skip = False
            continue
        if item == name:
            skip = True
            continue
        result.append(item)
    _require(not skip, "preloader_option_value_missing")
    return result


def _canonical_directory(value: str, *, context: str) -> Path:
    path = Path(value)
    _require(path.is_absolute(), f"{context}_not_absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        _fail(f"{context}_unavailable")
    _require(
        path == resolved and path.is_dir() and not path.is_symlink() and not _is_reparse(path),
        f"{context}_rejected",
    )
    return path


def _require_disjoint(paths: Mapping[str, Path]) -> None:
    items = list(paths.items())
    for index, (left_name, left) in enumerate(items):
        for right_name, right in items[index + 1 :]:
            _require(
                left != right and left not in right.parents and right not in left.parents,
                f"preloader_{left_name}_{right_name}_not_disjoint",
            )


def _strict_canonical_object(raw: bytes, *, context: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            _require(key not in result, f"{context}_duplicate_key")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail(f"{context}_json_rejected")
    _require(type(value) is dict and raw == _canonical(value), f"{context}_not_canonical")
    return value


def _install_receipt(
    path_text: str,
    expected_sha256: str,
    *,
    root: Path,
    context: str,
) -> tuple[dict[str, Any], Path]:
    _require(
        len(expected_sha256) == 64
        and all(character in "0123456789abcdef" for character in expected_sha256),
        f"{context}_expected_sha256_rejected",
    )
    path = Path(path_text)
    raw = _bound_file(path, context=context, maximum=65_536)
    _require(_sha256(raw) == expected_sha256, f"{context}_digest_mismatch")
    value = _strict_canonical_object(raw, context=context)
    try:
        receipt_entries = list(path.parent.iterdir())
    except OSError:
        _fail(f"{context}_directory_inventory_failed")
    _require(
        receipt_entries == [path] and not path.parent.is_symlink() and not _is_reparse(path.parent),
        f"{context}_directory_residue_rejected",
    )
    if context == "preloader_python_install_receipt":
        _require(
            set(value)
            == {
                "schema_version",
                "kind",
                "python_runtime_root",
                "archive_sha256",
                "manifest_sha256",
                "file_count",
                "directory_count",
                "file_inventory_sha256",
                "owner_private_acl_applied_before_children",
                "site_processing_disabled_by_embeddable_pth",
                "untracked_files_or_directories_present",
                "crash_recovery",
            }
            and value.get("schema_version") == 1
            and value.get("kind") == "explainiverse-operator-python-runtime-installed"
            and value.get("python_runtime_root") == str(root)
            and value.get("archive_sha256") == PYTHON_ARCHIVE_SHA256
            and value.get("manifest_sha256") == PYTHON_MANIFEST_SHA256
            and value.get("file_count") == 34
            and value.get("directory_count") == 0
            and value.get("file_inventory_sha256") == PYTHON_FILE_INVENTORY_SHA256
            and value.get("owner_private_acl_applied_before_children") is True
            and value.get("site_processing_disabled_by_embeddable_pth") is True
            and value.get("untracked_files_or_directories_present") is False
            and value.get("crash_recovery") == "discard-partial-directory-and-create-a-new-path",
            "preloader_python_install_receipt_binding_rejected",
        )
    else:
        _require(context == "preloader_site_install_receipt", "preloader_receipt_context_rejected")
        _require(
            set(value)
            == {
                "schema_version",
                "kind",
                "runtime_root",
                "manifest_sha256",
                "file_count",
                "directory_count",
                "file_inventory_sha256",
                "owner_private_acl_applied_before_children",
                "pip_present_in_runtime",
                "record_files_present",
                "generated_scripts_present",
                "bytecode_present",
                "crash_recovery",
            }
            and value.get("schema_version") == 1
            and value.get("kind") == "explainiverse-operator-runtime-installed"
            and value.get("runtime_root") == str(root)
            and value.get("manifest_sha256") == SITE_MANIFEST_SHA256
            and value.get("file_count") == 756
            and value.get("directory_count") == 113
            and value.get("file_inventory_sha256") == SITE_FILE_INVENTORY_SHA256
            and value.get("owner_private_acl_applied_before_children") is True
            and value.get("pip_present_in_runtime") is False
            and value.get("record_files_present") is False
            and value.get("generated_scripts_present") is False
            and value.get("bytecode_present") is False
            and value.get("crash_recovery") == "discard-partial-directory-and-create-a-new-path",
            "preloader_site_install_receipt_binding_rejected",
        )
    return value, path.parent


def _windows_current_user_sid() -> str:
    import ctypes
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
    _require(
        bool(open_token(get_current_process(), 0x0008, ctypes.byref(token))),
        "preloader_owner_token_open_failed",
    )
    try:
        needed = wintypes.DWORD()
        get_information(token, 1, None, 0, ctypes.byref(needed))
        _require(0 < needed.value <= 65_536, "preloader_owner_token_size_rejected")
        buffer = ctypes.create_string_buffer(needed.value)
        _require(
            bool(get_information(token, 1, buffer, needed, ctypes.byref(needed))),
            "preloader_owner_token_query_failed",
        )
        sid_pointer = ctypes.cast(buffer, ctypes.POINTER(wintypes.LPVOID))[0]
        sid_text = wintypes.LPWSTR()
        _require(
            bool(convert_sid(sid_pointer, ctypes.byref(sid_text))),
            "preloader_owner_sid_conversion_failed",
        )
        try:
            value = sid_text.value
        finally:
            local_free(sid_text)
        _require(
            type(value) is str and value.startswith("S-1-"),
            "preloader_owner_sid_rejected",
        )
        return cast(str, value)
    finally:
        close_handle(token)


def _windows_owner_private_acl(path: Path, *, context: str) -> dict[str, Any]:
    """Validate the exact protected three-principal directory DACL pre-site."""

    import ctypes
    from ctypes import wintypes

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_security = advapi32.GetNamedSecurityInfoW
    get_security.argtypes = [
        wintypes.LPWSTR,
        ctypes.c_int,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.LPVOID),
    ]
    get_security.restype = wintypes.DWORD
    convert_sid = advapi32.ConvertSidToStringSidW
    convert_sid.argtypes = [wintypes.LPVOID, ctypes.POINTER(wintypes.LPWSTR)]
    convert_sid.restype = wintypes.BOOL
    get_acl_information = advapi32.GetAclInformation
    get_acl_information.argtypes = [
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.c_int,
    ]
    get_acl_information.restype = wintypes.BOOL
    get_ace = advapi32.GetAce
    get_ace.argtypes = [wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.LPVOID)]
    get_ace.restype = wintypes.BOOL
    descriptor_control = advapi32.GetSecurityDescriptorControl
    descriptor_control.argtypes = [
        wintypes.LPVOID,
        ctypes.POINTER(wintypes.WORD),
        ctypes.POINTER(wintypes.DWORD),
    ]
    descriptor_control.restype = wintypes.BOOL
    descriptor_length = advapi32.GetSecurityDescriptorLength
    descriptor_length.argtypes = [wintypes.LPVOID]
    descriptor_length.restype = wintypes.DWORD
    local_free = kernel32.LocalFree
    local_free.argtypes = [wintypes.HLOCAL]
    local_free.restype = wintypes.HLOCAL

    class AclSizeInformation(ctypes.Structure):
        _fields_ = [
            ("ace_count", wintypes.DWORD),
            ("acl_bytes_in_use", wintypes.DWORD),
            ("acl_bytes_free", wintypes.DWORD),
        ]

    owner = wintypes.LPVOID()
    dacl = wintypes.LPVOID()
    descriptor = wintypes.LPVOID()
    result = get_security(
        str(path),
        1,
        0x00000001 | 0x00000004,
        ctypes.byref(owner),
        None,
        ctypes.byref(dacl),
        None,
        ctypes.byref(descriptor),
    )
    _require(
        result == 0 and bool(descriptor) and bool(owner) and bool(dacl),
        f"{context}_acl_query_failed",
    )
    try:
        control = wintypes.WORD()
        revision = wintypes.DWORD()
        _require(
            bool(descriptor_control(descriptor, ctypes.byref(control), ctypes.byref(revision)))
            and bool(control.value & 0x1000),
            f"{context}_dacl_not_protected",
        )

        def sid_text(pointer: Any) -> str:
            text_pointer = wintypes.LPWSTR()
            _require(
                bool(convert_sid(pointer, ctypes.byref(text_pointer))),
                f"{context}_sid_conversion_failed",
            )
            try:
                return str(text_pointer.value)
            finally:
                local_free(text_pointer)

        current_user = _windows_current_user_sid()
        _require(sid_text(owner) == current_user, f"{context}_owner_rejected")
        info = AclSizeInformation()
        _require(
            bool(get_acl_information(dacl, ctypes.byref(info), ctypes.sizeof(info), 2))
            and info.ace_count == 3,
            f"{context}_ace_count_rejected",
        )
        observed: set[str] = set()
        for index in range(info.ace_count):
            ace = wintypes.LPVOID()
            _require(bool(get_ace(dacl, index, ctypes.byref(ace))), f"{context}_ace_query_failed")
            address = int(cast(int, ace.value))
            ace_type = ctypes.c_ubyte.from_address(address).value
            ace_flags = ctypes.c_ubyte.from_address(address + 1).value
            mask = wintypes.DWORD.from_address(address + 4).value
            trustee = sid_text(wintypes.LPVOID(address + 8))
            _require(
                ace_type == 0
                and ace_flags == 0x03
                and mask == 0x001F01FF
                and trustee not in observed,
                f"{context}_ace_rejected",
            )
            observed.add(trustee)
        allowed = {current_user, "S-1-5-18", "S-1-5-32-544"}
        _require(observed == allowed, f"{context}_trustees_rejected")
        length = int(descriptor_length(descriptor))
        _require(0 < length <= 65_536, f"{context}_descriptor_size_rejected")
        raw = ctypes.string_at(descriptor, length)
        return {
            "owner_sid": current_user,
            "inheritance_protected": True,
            "child_inheritance_enabled": True,
            "allowed_sids": sorted(allowed),
            "ace_count": 3,
            "rights": "full-control",
            "security_descriptor_sha256": _sha256(raw),
            "security_descriptor_bytes": length,
            "validated_before_third_party_site_or_third_party_native_import": True,
            "pinned_stdlib_native_modules_loaded_before_hold": True,
        }
    finally:
        local_free(descriptor)


class _HeldWindowsTrees:
    """Keep every verified runtime path open without write/delete sharing."""

    def __init__(self, roots: Sequence[Path]) -> None:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._close_handle = kernel32.CloseHandle
        self._close_handle.argtypes = [wintypes.HANDLE]
        self._close_handle.restype = wintypes.BOOL
        create_file = kernel32.CreateFileW
        create_file.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        create_file.restype = wintypes.HANDLE
        self._handles: list[Any] = []
        self._closed = False
        try:
            paths: list[tuple[Path, bool]] = []
            for root in roots:
                paths.append((root, True))
                for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
                    state = path.lstat()
                    _require(
                        not stat.S_ISLNK(state.st_mode) and not _is_reparse(path),
                        "preloader_held_tree_reparse_rejected",
                    )
                    _require(
                        stat.S_ISDIR(state.st_mode) or stat.S_ISREG(state.st_mode),
                        "preloader_held_tree_entry_rejected",
                    )
                    paths.append((path, stat.S_ISDIR(state.st_mode)))
            _require(
                len({os.path.normcase(str(path)) for path, _ in paths}) == len(paths),
                "preloader_held_tree_path_alias_rejected",
            )
            invalid = ctypes.c_void_p(-1).value
            for path, directory in paths:
                desired = 0x00000081 if directory else 0x80000000
                flags = 0x00200000 | (0x02000000 if directory else 0x00000080)
                handle = create_file(str(path), desired, 0x00000001, None, 3, flags, None)
                _require(
                    handle not in {None, 0, invalid},
                    "preloader_held_tree_open_failed",
                )
                self._handles.append(handle)
            self.mapping = {
                "root_count": len(roots),
                "held_handle_count": len(self._handles),
                "write_share_allowed": False,
                "delete_share_allowed": False,
                "read_share_allowed": True,
                "held_before_third_party_site_or_third_party_native_import": True,
            }
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        first_error = False
        for handle in reversed(self._handles):
            if not self._close_handle(handle):
                first_error = True
        self._handles.clear()
        self._closed = True
        _require(not first_error, "preloader_held_tree_close_failed")


def _forbidden_environment_name(name: str) -> bool:
    normalized = name.upper()
    exact = {
        "ALL_PROXY",
        "GH_ENTERPRISE_TOKEN",
        "GH_HOST",
        "GH_REPO",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "LAMBDA_API_KEY",
        "NO_PROXY",
        "PYTHONHOME",
        "PYTHONPATH",
        "REQUESTS_CA_BUNDLE",
        "SSH_AGENT_PID",
        "SSH_AUTH_SOCK",
        "SSL_CERT_FILE",
    }
    prefixes = ("ANTHROPIC_", "AWS_", "AZURE_", "GCP_", "GOOGLE_", "OPENAI_")
    fragments = (
        "API_KEY",
        "CREDENTIAL",
        "JITCONFIG",
        "PASSWORD",
        "PASSWD",
        "PRIVATE_KEY",
        "SECRET",
        "TOKEN",
    )
    return (
        normalized in exact
        or normalized.startswith(prefixes)
        or any(fragment in normalized for fragment in fragments)
    )


def _scrub_environment() -> dict[str, Any]:
    removed: list[str] = []
    for name in tuple(os.environ):
        if _forbidden_environment_name(name):
            del os.environ[name]
            removed.append(name.upper())
    names = "\n".join(sorted(set(removed))).encode("ascii", errors="backslashreplace")
    return {
        "schema_version": 1,
        "kind": "operator-environment-scrub",
        "removed_name_count": len(set(removed)),
        "removed_names_sha256": _sha256(names),
        "removed_values_observed": False,
        "ambient_credentials_retained": False,
        "ambient_proxies_retained": False,
    }


def _source_snapshot(
    arguments: Sequence[str],
    *,
    root: Path,
    expected_head_sha: str,
    shim_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    _require(
        len(expected_head_sha) == 40
        and all(character in "0123456789abcdef" for character in expected_head_sha),
        "preloader_head_rejected",
    )
    manifest_raw = _bound_file(
        root / SOURCE_MANIFEST_RELATIVE,
        context="preloader_source_manifest",
        maximum=16 * 1024 * 1024,
    )
    _require(
        _sha256(manifest_raw) == SOURCE_MANIFEST_SHA256,
        "preloader_source_manifest_digest_rejected",
    )
    manifest = _strict_canonical_object(manifest_raw, context="preloader_source_manifest")
    _require(
        set(manifest)
        == {
            "schema_version",
            "kind",
            "excluded_paths",
            "files",
            "directories",
            "file_count",
            "directory_count",
            "file_inventory_sha256",
            "source",
            "runtime_git_dependency",
        }
        and manifest.get("schema_version") == 1
        and manifest.get("kind") == "explainiverse-operator-source-worktree-manifest"
        and manifest.get("excluded_paths") == [SOURCE_MANIFEST_RELATIVE, PRELOADER_RELATIVE]
        and manifest.get("source") == "exact-staged-index-blobs"
        and manifest.get("runtime_git_dependency") is False
        and type(manifest.get("files")) is dict
        and type(manifest.get("directories")) is list,
        "preloader_source_manifest_schema_rejected",
    )
    expected_files = manifest["files"]
    expected_directories = manifest["directories"]
    _require(
        manifest.get("file_count") == len(expected_files)
        and manifest.get("directory_count") == len(expected_directories)
        and len(expected_directories) == len(set(expected_directories)),
        "preloader_source_manifest_count_rejected",
    )
    observed_files: dict[str, bytes] = {}
    observed_directories: set[str] = set()
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError:
            _fail("preloader_source_inventory_failed")
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            if relative == ".git":
                continue
            try:
                state = path.lstat()
            except OSError:
                _fail("preloader_source_entry_unavailable")
            _require(
                not stat.S_ISLNK(state.st_mode) and not _is_reparse(path),
                "preloader_source_reparse_rejected",
            )
            if stat.S_ISDIR(state.st_mode):
                observed_directories.add(relative)
                pending.append(path)
            elif stat.S_ISREG(state.st_mode):
                if relative in {SOURCE_MANIFEST_RELATIVE, PRELOADER_RELATIVE}:
                    continue
                _require(relative not in observed_files, "preloader_source_path_duplicate")
                observed_files[relative] = _bound_file(
                    path, context="preloader_source", maximum=MAX_SOURCE_BYTES
                )
            else:
                _fail("preloader_source_entry_type_rejected")
    _require(
        observed_directories == set(expected_directories)
        and set(observed_files) == set(expected_files),
        "preloader_source_file_or_directory_set_rejected",
    )
    captured: dict[str, bytes] = {}
    rows: list[bytes] = []
    for relative, expected in sorted(expected_files.items()):
        raw = observed_files[relative]
        _require(
            type(expected) is dict
            and set(expected) == {"mode", "bytes", "sha256", "git_blob_sha"}
            and expected.get("mode") in {"100644", "100755"}
            and expected.get("bytes") == len(raw)
            and expected.get("sha256") == _sha256(raw)
            and expected.get("git_blob_sha") == _sha1_git_blob(raw),
            "preloader_source_bytes_rejected",
        )
        rows.append(
            f"{relative}\t{expected['mode']}\t{len(raw)}\t{_sha256(raw)}\t{_sha1_git_blob(raw)}\n".encode(
                "utf-8"
            )
        )
        if relative.startswith(SOURCE_PREFIXES) or relative in SOURCE_EXACT:
            captured[relative] = raw
    _require(
        _sha256(b"".join(rows)) == manifest.get("file_inventory_sha256"),
        "preloader_source_inventory_digest_rejected",
    )
    required = {
        "scripts/release_gpu_jit_lambda_operator/bootstrap.py",
        "scripts/release_gpu_jit_lambda_operator/__main__.py",
        "scripts/release_gpu_jit_lambda_operator/windows_launcher.py",
        "scripts/release_gpu_jit_lambda_operator/receipt_contract.py",
        "scripts/release_gpu_jit_lambda_controller/controller.py",
        "scripts/release_gpu_jit_lambda_controller/driver.py",
        "scripts/release_gpu_jit_lambda_live/adapter.py",
        "scripts/release_gpu_jit_lambda_runtime/runtime_contract.py",
        "scripts/release_external_controls.py",
        "scripts/verify_release_recovery.py",
        ".github/release-control-policy.json",
        ".github/workflows/cuda-ci.yml",
        ".github/workflows/publish-pypi.yml",
        ".github/workflows/recover-github-release.yml",
        "poetry.lock",
        "pyproject.toml",
    }
    _require(required.issubset(captured), "preloader_required_source_missing")
    shim_relative = "scripts/release_gpu_jit_lambda_operator/preloader_shim.py"
    _require(
        shim_relative in captured and _sha256(captured[shim_relative]) == PRELOADER_SHIM_SHA256,
        "preloader_shim_source_digest_rejected",
    )
    self_raw = _bound_file(
        root / PRELOADER_RELATIVE, context="preloader_self", maximum=MAX_SOURCE_BYTES
    )
    _require(
        shim_receipt.get("preloader_sha256") == _sha256(self_raw),
        "preloader_self_digest_drift",
    )
    capture_rows = [
        f"{path}\t{len(raw)}\t{_sha256(raw)}\n".encode("utf-8")
        for path, raw in sorted(captured.items())
    ]
    material = {
        "schema_version": 1,
        "kind": "explainiverse-operator-clean-source-preload",
        "repository_root": str(root),
        "origin_url": EXPECTED_ORIGIN,
        "head_sha": expected_head_sha,
        "head_and_origin_verified_during_credential_free_inventory": False,
        "source_manifest": manifest,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_manifest_inventory_sha256": manifest["file_inventory_sha256"],
        "tracked_and_untracked_clean": True,
        "runtime_git_dependency": False,
        "preloader_path": str(Path(__file__).resolve(strict=True)),
        "preloader_sha256": shim_receipt["preloader_sha256"],
        "captured_module_count": len(captured),
        "captured_module_inventory_sha256": _sha256(b"".join(capture_rows)),
        "project_modules_execute_from_captured_bytes": True,
        "arguments_sha256": _sha256("\0".join(arguments).encode("utf-8")),
    }
    return {**material, "evidence_sha256": _sha256(_canonical(material))}, captured


class _CapturedSourceLoader(importlib.abc.InspectLoader):
    def __init__(self, fullname: str, path: str, raw: bytes, is_package: bool) -> None:
        self._fullname = fullname
        self._path = path
        self._raw = raw
        self._is_package = is_package

    def get_filename(self, fullname: str) -> str:
        _require(fullname == self._fullname, "preloader_loader_module_drift")
        return self._path

    def get_source(self, fullname: str) -> str:
        _require(fullname == self._fullname, "preloader_loader_module_drift")
        try:
            return self._raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            _fail("preloader_source_encoding_rejected")

    def is_package(self, fullname: str) -> bool:
        _require(fullname == self._fullname, "preloader_loader_module_drift")
        return self._is_package

    def get_code(self, fullname: str) -> Any:
        return compile(self.get_source(fullname), self._path, "exec", dont_inherit=True)


class _CapturedSourceFinder(importlib.abc.MetaPathFinder):
    def __init__(self, root: Path, sources: Mapping[str, bytes]) -> None:
        modules: dict[str, tuple[str, bytes, bool]] = {}
        for relative, raw in sources.items():
            path = Path(relative)
            if path.suffix != ".py":
                continue
            if path.name == "__init__.py":
                fullname = ".".join(path.parent.parts)
                is_package = True
            else:
                fullname = ".".join(path.with_suffix("").parts)
                is_package = False
            modules[fullname] = (str(root / path), raw, is_package)
        self._modules = modules

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        del path, target
        if fullname == "scripts":
            spec = importlib.machinery.ModuleSpec(fullname, loader=None, is_package=True)
            spec.submodule_search_locations = []
            return spec
        item = self._modules.get(fullname)
        if item is not None:
            filename, raw, is_package = item
            loader = _CapturedSourceLoader(fullname, filename, raw, is_package)
            return importlib.util.spec_from_loader(fullname, loader, is_package=is_package)
        if fullname.startswith("scripts."):
            _fail("preloader_uncaptured_project_module_rejected")
        return None


def _shim_receipt() -> dict[str, Any]:
    receipt_value = getattr(builtins, SHIM_MARKER_NAME, None)
    _require(type(receipt_value) is dict, "preloader_verified_shim_receipt_missing")
    receipt = cast(dict[str, Any], receipt_value)
    required = {
        "schema_version",
        "kind",
        "preloader_path",
        "preloader_bytes",
        "preloader_sha256",
        "shim_sha256",
        "stable_descriptor_read",
        "compiled_verified_bytes_without_reopen",
    }
    _require(
        set(receipt) == required
        and receipt.get("schema_version") == 1
        and receipt.get("kind") == "explainiverse-operator-preloader-shim"
        and receipt.get("shim_sha256") == PRELOADER_SHIM_SHA256
        and type(receipt.get("preloader_bytes")) is int
        and 1 <= receipt["preloader_bytes"] <= MAX_SOURCE_BYTES
        and type(receipt.get("preloader_sha256")) is str
        and len(receipt["preloader_sha256"]) == 64
        and all(character in "0123456789abcdef" for character in receipt["preloader_sha256"])
        and receipt.get("stable_descriptor_read") is True
        and receipt.get("compiled_verified_bytes_without_reopen") is True,
        "preloader_verified_shim_receipt_rejected",
    )
    path = Path(str(receipt["preloader_path"]))
    _require(
        path.resolve(strict=True) == Path(__file__).resolve(strict=True),
        "preloader_shim_path_drift",
    )
    delattr(builtins, SHIM_MARKER_NAME)
    return dict(receipt)


def _main_with_held_roots(arguments: Sequence[str], *, early_boundary: Mapping[str, Any]) -> int:
    argv = list(arguments)
    _require(
        os.name == "nt"
        and platform.python_implementation() == "CPython"
        and sys.version_info[:3] == (3, 13, 15)
        and sys.flags.isolated == 1
        and sys.flags.safe_path
        and sys.flags.no_site == 1
        and sys.dont_write_bytecode,
        "preloader_requires_pinned_python_I_S_B",
    )
    shim = _shim_receipt()
    environment = _scrub_environment()
    root = _canonical_directory(_option(argv, "--repository-root"), context="preloader_root")
    python_root = _canonical_directory(
        _option(argv, "--operator-python-root"), context="preloader_python_root"
    )
    site_root = _canonical_directory(
        _option(argv, "--operator-site-root"), context="preloader_site_root"
    )
    python_install, python_install_directory = _install_receipt(
        _option(argv, "--operator-python-install-receipt"),
        _option(argv, "--operator-python-install-receipt-sha256"),
        root=python_root,
        context="preloader_python_install_receipt",
    )
    site_install, site_install_directory = _install_receipt(
        _option(argv, "--operator-site-install-receipt"),
        _option(argv, "--operator-site-install-receipt-sha256"),
        root=site_root,
        context="preloader_site_install_receipt",
    )
    _require_disjoint(
        {
            "repository_root": root,
            "python_root": python_root,
            "site_root": site_root,
            "python_receipt_root": python_install_directory,
            "site_receipt_root": site_install_directory,
        }
    )
    expected_head_sha = _option(argv, "--expected-head-sha")
    target_name = _option(argv, "--operator-target")
    _require(target_name in TARGETS, "preloader_target_rejected")
    _require(
        Path.cwd().resolve(strict=True) == python_install_directory,
        "preloader_working_directory_mismatch",
    )
    _require(
        Path(sys.executable).resolve(strict=True) == python_root / "python.exe",
        "preloader_python_identity_drift",
    )
    initial_paths = [Path(value).resolve(strict=False) for value in sys.path]
    _require(
        all(
            (path == python_root or python_root in path.parents)
            and path != root
            and root not in path.parents
            and path != site_root
            and site_root not in path.parents
            for path in initial_paths
        ),
        "preloader_untrusted_root_in_initial_sys_path",
    )
    source, captured = _source_snapshot(
        argv,
        root=root,
        expected_head_sha=expected_head_sha,
        shim_receipt=shim,
    )
    bootstrap_path = "scripts/release_gpu_jit_lambda_operator/bootstrap.py"
    bootstrap_namespace: dict[str, Any] = {
        "__name__": "_explainiverse_verified_operator_bootstrap",
        "__file__": str(root / Path(bootstrap_path)),
        "__builtins__": builtins.__dict__,
    }
    try:
        exec(
            compile(captured[bootstrap_path], bootstrap_namespace["__file__"], "exec"),
            bootstrap_namespace,
        )
        bootstrap_receipt = bootstrap_namespace["verify_and_enable"](
            root,
            python_root,
            site_root,
            python_install_receipt=python_install,
            site_install_receipt=site_install,
        )
    except Exception as exc:
        _fail(str(exc))
    finder = _CapturedSourceFinder(root, captured)
    sys.meta_path.insert(0, finder)
    _require(str(root) not in sys.path, "preloader_repository_root_enabled")
    runtime_receipt = None
    site_receipt = None
    python_install_directory_receipt = None
    site_install_directory_receipt = None
    try:
        import importlib

        adapter = importlib.import_module("scripts.release_gpu_jit_lambda_live.adapter")
        runtime_receipt = adapter._capture_evidence_directory(python_root)
        site_receipt = adapter._capture_evidence_directory(site_root)
        python_install_directory_receipt = adapter._capture_evidence_directory(
            python_install_directory
        )
        site_install_directory_receipt = adapter._capture_evidence_directory(site_install_directory)
        runtime_validation = runtime_receipt.validate()
        site_validation = site_receipt.validate()
        python_install_directory_validation = python_install_directory_receipt.validate()
        site_install_directory_validation = site_install_directory_receipt.validate()
        _require(
            bootstrap_receipt["python_tree"]["file_inventory_sha256"]
            == python_install["file_inventory_sha256"]
            and bootstrap_receipt["site_tree"]["file_inventory_sha256"]
            == site_install["file_inventory_sha256"],
            "preloader_install_receipt_tree_binding_rejected",
        )
        policy_bytes = captured[".github/release-control-policy.json"]
        controller_source_bytes = captured[
            "scripts/release_gpu_jit_lambda_controller/controller.py"
        ]
        runtime_files = {
            name: captured[f"scripts/release_gpu_jit_lambda_runtime/{name}"]
            for name in RUNTIME_BUNDLE_NAMES
        }
        resource_material = {
            "schema_version": 1,
            "kind": "explainiverse-operator-sealed-resource-binding",
            "policy_sha256": _sha256(policy_bytes),
            "controller_source_sha256": _sha256(controller_source_bytes),
            "runtime_bundle_sha256": _runtime_bundle_sha256(runtime_files),
            "runtime_file_sha256": {
                name: _sha256(value) for name, value in sorted(runtime_files.items())
            },
            "captured_before_project_import": True,
            "live_repository_reopen_permitted": False,
        }
        material = {
            "schema_version": 1,
            "kind": "explainiverse-operator-isolated-preloader",
            "shim": shim,
            "source": source,
            "bootstrap": bootstrap_receipt,
            "python_runtime_directory_receipt": runtime_receipt.to_public_mapping(),
            "python_runtime_validation": runtime_validation,
            "runtime_site_directory_receipt": site_receipt.to_public_mapping(),
            "runtime_site_validation": site_validation,
            "python_install_receipt": python_install,
            "python_install_receipt_sha256": _sha256(_canonical(python_install)),
            "python_install_directory_receipt": (
                python_install_directory_receipt.to_public_mapping()
            ),
            "python_install_directory_validation": python_install_directory_validation,
            "site_install_receipt": site_install,
            "site_install_receipt_sha256": _sha256(_canonical(site_install)),
            "site_install_directory_receipt": (site_install_directory_receipt.to_public_mapping()),
            "site_install_directory_validation": site_install_directory_validation,
            "environment": environment,
            "early_runtime_boundary": dict(early_boundary),
            "sealed_resources": resource_material,
            "working_directory": str(python_install_directory),
            "working_directory_is_python_install_receipt_directory": True,
            "isolated": True,
            "safe_path": True,
            "site_disabled": True,
            "bytecode_disabled": True,
            "repository_absent_from_sys_path": True,
            "project_imports_from_captured_bytes": True,
        }
        receipt = {**material, "evidence_sha256": _sha256(_canonical(material))}
        _require(
            not hasattr(builtins, MARKER_NAME) and not hasattr(builtins, RESOURCE_MARKER_NAME),
            "preloader_marker_already_present",
        )
        setattr(builtins, MARKER_NAME, receipt)
        setattr(
            builtins,
            RESOURCE_MARKER_NAME,
            {
                "schema_version": 1,
                "kind": "explainiverse-operator-captured-resources",
                "preloader_evidence_sha256": receipt["evidence_sha256"],
                "policy_bytes": policy_bytes,
                "controller_source_bytes": controller_source_bytes,
                "runtime_files": runtime_files,
            },
        )
        forwarded = _without_option(argv, "--operator-target")
        previous_argv = sys.argv
        try:
            sys.argv = [TARGETS[target_name], *forwarded]
            runpy.run_module(TARGETS[target_name], run_name="__main__", alter_sys=True)
        finally:
            sys.argv = previous_argv
            if hasattr(builtins, MARKER_NAME):
                delattr(builtins, MARKER_NAME)
            if hasattr(builtins, RESOURCE_MARKER_NAME):
                delattr(builtins, RESOURCE_MARKER_NAME)
    finally:
        if site_install_directory_receipt is not None:
            site_install_directory_receipt.close()
        if python_install_directory_receipt is not None:
            python_install_directory_receipt.close()
        if site_receipt is not None:
            site_receipt.close()
        if runtime_receipt is not None:
            runtime_receipt.close()
        try:
            sys.meta_path.remove(finder)
        except ValueError:
            pass
    return 0


def main(arguments: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if arguments is None else arguments)
    _require(
        os.name == "nt"
        and platform.python_implementation() == "CPython"
        and sys.version_info[:3] == (3, 13, 15)
        and sys.flags.isolated == 1
        and sys.flags.safe_path
        and sys.flags.no_site == 1
        and sys.dont_write_bytecode,
        "preloader_requires_pinned_python_I_S_B",
    )
    root = _canonical_directory(_option(argv, "--repository-root"), context="preloader_root")
    python_root = _canonical_directory(
        _option(argv, "--operator-python-root"), context="preloader_python_root"
    )
    site_root = _canonical_directory(
        _option(argv, "--operator-site-root"), context="preloader_site_root"
    )
    _, python_receipt_root = _install_receipt(
        _option(argv, "--operator-python-install-receipt"),
        _option(argv, "--operator-python-install-receipt-sha256"),
        root=python_root,
        context="preloader_python_install_receipt",
    )
    _, site_receipt_root = _install_receipt(
        _option(argv, "--operator-site-install-receipt"),
        _option(argv, "--operator-site-install-receipt-sha256"),
        root=site_root,
        context="preloader_site_install_receipt",
    )
    roots = {
        "repository_root": root,
        "python_root": python_root,
        "site_root": site_root,
        "python_receipt_root": python_receipt_root,
        "site_receipt_root": site_receipt_root,
    }
    _require_disjoint(roots)
    _require(
        Path.cwd().resolve(strict=True) == python_receipt_root,
        "preloader_working_directory_mismatch",
    )
    held = _HeldWindowsTrees((python_root, site_root, python_receipt_root, site_receipt_root))
    try:
        acl = {
            name: _windows_owner_private_acl(path, context=f"preloader_{name}")
            for name, path in roots.items()
            if name != "repository_root"
        }
        early_material = {
            "schema_version": 1,
            "kind": "explainiverse-operator-early-runtime-boundary",
            "acl": acl,
            "held_trees": held.mapping,
            "all_runtime_and_receipt_roots_owner_private": True,
            "all_runtime_and_receipt_paths_held_without_write_or_delete_share": True,
            "validated_before_third_party_site_or_third_party_native_import": True,
            "pinned_official_python_runtime_is_the_pre_hold_trust_boundary": True,
            "working_directory": str(python_receipt_root),
            "working_directory_repository_disjoint": True,
        }
        early_boundary = {
            **early_material,
            "evidence_sha256": _sha256(_canonical(early_material)),
        }
        return _main_with_held_roots(argv, early_boundary=early_boundary)
    finally:
        active_exception = sys.exc_info()[0] is not None
        try:
            held.close()
        except BaseException:
            if not active_exception:
                raise


if __name__ == "__main__":
    raise SystemExit(main())
