"""Security boundary helpers for the production Lambda CUDA operator.

This module is intentionally free of provider and GitHub mutations.  It binds
the local process, exact source checkout, executable/dependency inventory, and
owner-authenticated App evidence before :mod:`.cli` can construct live gates.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import secrets
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, Sequence

from scripts.release_gpu_jit_lambda_controller import (
    ControllerError,
    GhCliTransport,
    GitHubResponse,
    TrustedAppCapture,
)
from scripts.release_gpu_jit_lambda_controller.controller import SealedControllerResources
from scripts.release_gpu_jit_lambda_live import (
    EvidenceDirectoryReceipt,
    ImmutablePlan,
    build_immutable_plan,
)
from scripts.release_gpu_jit_lambda_runtime import runtime_contract as runtime

from . import bootstrap as operator_bootstrap

REPOSITORY = "jemsbhai/explainiverse"
EXPECTED_ORIGIN_URL = "https://github.com/jemsbhai/explainiverse.git"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
SAFE_CAPTURE_FILENAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}\Z")
MAX_PUBLIC_JSON_BYTES = 4 * 1024 * 1024
MAX_CAPTURE_PAGE_BYTES = 16 * 1024 * 1024
MAX_COMMAND_BYTES = 4 * 1024 * 1024
INVENTORY_KIND = "explainiverse-lambda-operator-inventory"
INSPECTION_KIND = "explainiverse-lambda-operator-inspection"
CAPTURE_READY_KIND = "explainiverse-installed-app-capture-ready"

TRUSTED_INSTALLER_SID = "S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


PINNED_WINDOWS_POWERSHELL = {
    "absolute_path": r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
    "sha256": "7600ffe12da441fe89d035b13801e8e91d064bc544a27b19a5cf49f6ab8b18f5",
    "owner_sid": TRUSTED_INSTALLER_SID,
}
PINNED_WINDOWS_EXECUTABLES = {
    "git": {
        "absolute_path": r"C:\Program Files\Git\cmd\git.exe",
        "sha256": "d90e36cafd656d52984f7546bfcb5b065d73e2e66957c952b7a4a1cd260e8f36",
        "version": "git version 2.46.1.windows.1",
        "owner_sid": "S-1-5-32-544",
        "authenticode_subject": (
            "CN=Johannes Schindelin, O=Johannes Schindelin, " "S=Nordrhein-Westfalen, C=DE"
        ),
        "authenticode_thumbprint": "3EB14A3AEF84B7153E139397F0A49E2FAC662B0E",
        "runtime_absolute_path": r"C:\Program Files\Git\mingw64\bin\git.exe",
        "runtime_sha256": "3591764e521c340b8cca2ca300b3ce265df271ac41d2b338113c9a76fb32bcaa",
    },
    "gh": {
        "absolute_path": r"C:\Program Files\HP\AIStudio\bin\gh.exe",
        "sha256": "383bc207db46f000ca6fce3dfad1459c06665f6dfc88741b711137e31eb5eddf",
        "version": (
            "gh version 2.64.0 (2024-12-20)\n" "https://github.com/cli/cli/releases/tag/v2.64.0"
        ),
        "owner_sid": "S-1-5-18",
        "authenticode_subject": (
            'CN="GitHub, Inc.", O="GitHub, Inc.", ' "L=San Francisco, S=California, C=US"
        ),
        "authenticode_thumbprint": "9C7CE6D3ED2CD2D8A0C5F2B3F687298B81298E68",
    },
    "ssh": {
        "absolute_path": r"C:\Windows\System32\OpenSSH\ssh.exe",
        "sha256": "6250fd52163fe99a0dc49403ed1b4bbef9b764bdb7bada017a93d057d9376a42",
        "version": "OpenSSH_for_Windows_9.5p2, LibreSSL 3.8.2",
        "owner_sid": TRUSTED_INSTALLER_SID,
        "authenticode_subject": (
            "CN=Microsoft Windows, O=Microsoft Corporation, " "L=Redmond, S=Washington, C=US"
        ),
        "authenticode_thumbprint": "BAC13DF18B37E808208A39D3A54CCE975FAC8C1D",
    },
}

OPERATOR_LOCK_RELATIVE = "scripts/release_gpu_jit_lambda_operator/requirements-windows-cp313.txt"
OPERATOR_BOOTSTRAP_LOCK_RELATIVE = (
    "scripts/release_gpu_jit_lambda_operator/requirements-windows-cp313-bootstrap.txt"
)
OPERATOR_MANIFEST_RELATIVE = (
    "scripts/release_gpu_jit_lambda_operator/site-packages-windows-cp313.json"
)
OPERATOR_PYTHON_MANIFEST_RELATIVE = (
    "scripts/release_gpu_jit_lambda_operator/python-runtime-windows-cp313.json"
)
OPERATOR_SOURCE_MANIFEST_RELATIVE = (
    "scripts/release_gpu_jit_lambda_operator/source-worktree-manifest.json"
)
LOCKED_DISTRIBUTIONS = {
    "cffi": {
        "version": "2.1.1",
        "wheel_sha256": "1aa5645c30469b09530c4ebca77ebf8f17618293c58f8549cb1a543a50236e7d",
    },
    "cryptography": {
        "version": "50.0.0",
        "wheel_sha256": "bd1c592e4d5974f0d08d4888e432157adba757c66da0246918e43677fafa2d30",
    },
    "pycparser": {
        "version": "3.0",
        "wheel_sha256": "b727414169a36b7d524c1c3e31839a521725078d7b2ff038656844266160a992",
    },
    "pywin32": {
        "version": "311",
        "wheel_sha256": "718a38f7e5b058e76aee1c56ddd06908116d35147e133427e59a3983f703a20d",
    },
}
ALLOWED_PTH_FILES = {
    "pywin32.pth": {
        "bytes": 185,
        "sha256": "d902584a2a0a5216ce12c712d1378fe07541d32c383d0cc5abcd68412144fe4d",
    }
}
LOCK_REQUIREMENT_RE = re.compile(
    r"(?P<name>[a-z0-9][a-z0-9._-]*)==(?P<version>[A-Za-z0-9][A-Za-z0-9.!+_-]*) "
    r"--hash=sha256:(?P<sha256>[0-9a-f]{64})\Z"
)

CRITICAL_SOURCE_PATHS = (
    ".github/release-control-policy.json",
    ".github/workflows/cuda-ci.yml",
    ".github/workflows/publish-pypi.yml",
    ".github/workflows/recover-github-release.yml",
    "poetry.lock",
    "pyproject.toml",
    "scripts/release_external_controls.py",
    "scripts/verify_release_recovery.py",
    "scripts/release_gpu_jit_lambda_controller/README.md",
    "scripts/release_gpu_jit_lambda_controller/__init__.py",
    "scripts/release_gpu_jit_lambda_controller/controller.py",
    "scripts/release_gpu_jit_lambda_controller/driver.py",
    "scripts/release_gpu_jit_lambda_live/README.md",
    "scripts/release_gpu_jit_lambda_live/__init__.py",
    "scripts/release_gpu_jit_lambda_live/adapter.py",
    "scripts/release_gpu_jit_lambda_operator/__init__.py",
    "scripts/release_gpu_jit_lambda_operator/__main__.py",
    "scripts/release_gpu_jit_lambda_operator/README.md",
    "scripts/release_gpu_jit_lambda_operator/bootstrap.py",
    "scripts/release_gpu_jit_lambda_operator/boundary.py",
    "scripts/release_gpu_jit_lambda_operator/build_windows_manifest.py",
    "scripts/release_gpu_jit_lambda_operator/build_windows_python_manifest.py",
    "scripts/release_gpu_jit_lambda_operator/build_source_worktree_manifest.py",
    "scripts/release_gpu_jit_lambda_operator/cli.py",
    "scripts/release_gpu_jit_lambda_operator/install_windows_python.py",
    "scripts/release_gpu_jit_lambda_operator/install_windows_runtime.py",
    "scripts/release_gpu_jit_lambda_operator/preloader.py",
    "scripts/release_gpu_jit_lambda_operator/preloader_shim.py",
    "scripts/release_gpu_jit_lambda_operator/receipt_contract.py",
    OPERATOR_BOOTSTRAP_LOCK_RELATIVE,
    OPERATOR_LOCK_RELATIVE,
    OPERATOR_MANIFEST_RELATIVE,
    OPERATOR_PYTHON_MANIFEST_RELATIVE,
    OPERATOR_SOURCE_MANIFEST_RELATIVE,
    "scripts/release_gpu_jit_lambda_operator/windows_launcher.py",
    "scripts/release_gpu_jit_lambda_runtime/README.md",
    "scripts/release_gpu_jit_lambda_runtime/__init__.py",
    "scripts/release_gpu_jit_lambda_runtime/bootstrap.py",
    "scripts/release_gpu_jit_lambda_runtime/executor.py",
    "scripts/release_gpu_jit_lambda_runtime/runtime_contract.py",
)

PHASE_REFS = {
    "pull-request": runtime.PULL_REQUEST_REF,
    "final-main": runtime.FINAL_MAIN_REF,
    "publication": runtime.PUBLICATION_REF,
}
PHASE_CAPTURE_COUNTS = {"pull-request": 2, "final-main": 4, "publication": 2}


class OperatorError(RuntimeError):
    """A stable, secret-free operator rejection code."""


def _fail(code: str) -> NoReturn:
    raise OperatorError(code)


def _require(condition: bool, code: str) -> None:
    if not condition:
        _fail(code)


def canonical_json(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        ).encode("ascii")
    except (TypeError, ValueError):
        _fail("public_value_not_canonical_json")


def sha256_bytes(value: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(value).hexdigest()


def _pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, "json_duplicate_key_rejected")
        result[key] = value
    return result


def strict_json(raw: bytes, *, context: str, maximum: int = MAX_PUBLIC_JSON_BYTES) -> Any:
    _require(0 < len(raw) <= maximum, f"{context}_size_rejected")
    try:
        return json.loads(raw, object_pairs_hook=_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail(f"{context}_json_rejected")


def _is_reparse(path: Path) -> bool:
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & flag)


def canonical_existing_file(path: str | os.PathLike[str], *, context: str) -> Path:
    candidate = Path(path)
    _require(candidate.is_absolute(), f"{context}_not_absolute")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        _fail(f"{context}_unavailable")
    _require(candidate == resolved, f"{context}_not_canonical")
    _require(
        candidate.is_file() and not candidate.is_symlink() and not _is_reparse(candidate),
        f"{context}_file_rejected",
    )
    return candidate


def canonical_existing_directory(path: str | os.PathLike[str], *, context: str) -> Path:
    candidate = Path(path)
    _require(candidate.is_absolute(), f"{context}_not_absolute")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        _fail(f"{context}_unavailable")
    _require(candidate == resolved, f"{context}_not_canonical")
    _require(
        candidate.is_dir() and not candidate.is_symlink() and not _is_reparse(candidate),
        f"{context}_directory_rejected",
    )
    return candidate


def _windows_handle_path(fd: int) -> str:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    handle = msvcrt.get_osfhandle(fd)  # type: ignore[attr-defined]
    get_final_path = ctypes.WinDLL(  # type: ignore[attr-defined]
        "kernel32", use_last_error=True
    ).GetFinalPathNameByHandleW
    get_final_path.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
    get_final_path.restype = wintypes.DWORD
    needed = get_final_path(handle, None, 0, 0)
    _require(0 < needed <= 32768, "bound_file_handle_path_rejected")
    buffer = ctypes.create_unicode_buffer(needed + 1)
    written = get_final_path(handle, buffer, len(buffer), 0)
    _require(0 < written < len(buffer), "bound_file_handle_path_rejected")
    value = buffer.value
    if value.startswith("\\\\?\\"):
        value = value[4:]
    return os.path.normcase(os.path.abspath(value))


def read_bound_file(
    path: str | os.PathLike[str],
    *,
    context: str,
    maximum: int = MAX_PUBLIC_JSON_BYTES,
    require_single_link: bool = True,
) -> bytes:
    """Read one canonical, single-link regular file through a held descriptor."""

    candidate = canonical_existing_file(path, context=context)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError:
        _fail(f"{context}_open_failed")
    try:
        before = os.fstat(descriptor)
        current = os.lstat(candidate)
        _require(stat.S_ISREG(before.st_mode), f"{context}_not_regular")
        _require(
            before.st_nlink == current.st_nlink
            and before.st_nlink >= 1
            and (not require_single_link or before.st_nlink == 1),
            f"{context}_link_count_rejected",
        )
        _require(
            (before.st_dev, before.st_ino) == (current.st_dev, current.st_ino),
            f"{context}_identity_mismatch",
        )
        if os.name == "nt":
            _require(
                _windows_handle_path(descriptor) == os.path.normcase(str(candidate)),
                f"{context}_handle_path_mismatch",
            )
        _require(0 < before.st_size <= maximum, f"{context}_size_rejected")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            _require(bool(chunk), f"{context}_short_read")
            chunks.append(chunk)
            remaining -= len(chunk)
        _require(os.read(descriptor, 1) == b"", f"{context}_grew_during_read")
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
            f"{context}_changed_during_read",
        )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def read_canonical_json_file(
    path: str | os.PathLike[str],
    *,
    context: str,
    maximum: int = MAX_PUBLIC_JSON_BYTES,
) -> tuple[dict[str, Any], bytes]:
    raw = read_bound_file(path, context=context, maximum=maximum)
    value = strict_json(raw, context=context, maximum=maximum)
    _require(type(value) is dict, f"{context}_not_object")
    _require(raw == canonical_json(value), f"{context}_not_canonical")
    return value, raw


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
    prefixes = (
        "ANTHROPIC_",
        "AWS_",
        "AZURE_",
        "GCP_",
        "GOOGLE_",
        "OPENAI_",
    )
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


def scrub_process_environment() -> dict[str, Any]:
    """Delete ambient credential/proxy inputs without inspecting their values."""

    removed: list[str] = []
    for name in tuple(os.environ):
        if _forbidden_environment_name(name):
            del os.environ[name]
            removed.append(name.upper())
    material = "\n".join(sorted(set(removed))).encode("ascii", errors="backslashreplace")
    return {
        "schema_version": 1,
        "kind": "operator-environment-scrub",
        "removed_name_count": len(set(removed)),
        "removed_names_sha256": sha256_bytes(material),
        "removed_values_observed": False,
        "ambient_credentials_retained": False,
        "ambient_proxies_retained": False,
    }


def child_environment() -> dict[str, str]:
    allowed = {
        "APPDATA",
        "COMSPEC",
        "LANG",
        "LC_ALL",
        "LOCALAPPDATA",
        "NO_COLOR",
        "PROGRAMDATA",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "WINDIR",
    }
    return {name: value for name, value in os.environ.items() if name.upper() in allowed}


def _run_bound(
    executable: Path,
    arguments: Sequence[str],
    *,
    context: str,
    cwd: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> tuple[bytes, bytes]:
    try:
        completed = subprocess.run(
            [str(executable), *arguments],
            cwd=str(cwd) if cwd is not None else None,
            env=dict(environment) if environment is not None else child_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            shell=False,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        _fail(f"{context}_execution_failed")
    _require(completed.returncode == 0, f"{context}_nonzero")
    _require(
        len(completed.stdout) <= MAX_COMMAND_BYTES and len(completed.stderr) <= MAX_COMMAND_BYTES,
        f"{context}_output_too_large",
    )
    return completed.stdout, completed.stderr


def _command_version(executable: Path, arguments: Sequence[str], *, context: str) -> str:
    stdout, stderr = _run_bound(executable, arguments, context=context)
    raw = stdout + stderr
    _require(b"\x00" not in raw, f"{context}_version_nul")
    try:
        value = raw.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError:
        _fail(f"{context}_version_encoding")
    _require(0 < len(value) <= 4096, f"{context}_version_rejected")
    return value


def _windows_acl_identity(path: Path, *, context: str, expected_owner_sid: str) -> dict[str, Any]:
    """Prove the pinned executable is not writable by an unprivileged SID."""

    _require(os.name == "nt", f"{context}_requires_windows")
    try:
        import win32security  # type: ignore[import-not-found,import-untyped]

        descriptor = win32security.GetFileSecurity(
            str(path),
            win32security.OWNER_SECURITY_INFORMATION | win32security.DACL_SECURITY_INFORMATION,
        )
        owner_sid = win32security.ConvertSidToStringSid(descriptor.GetSecurityDescriptorOwner())
        dacl = descriptor.GetSecurityDescriptorDacl()
    except (ImportError, OSError, AttributeError):
        _fail(f"{context}_acl_unavailable")
    _require(owner_sid == expected_owner_sid, f"{context}_owner_drift")
    _require(dacl is not None, f"{context}_null_dacl_rejected")
    privileged = {"S-1-5-18", "S-1-5-32-544", TRUSTED_INSTALLER_SID}
    write_mask = (
        0x00000002
        | 0x00000004
        | 0x00000010
        | 0x00000040
        | 0x00000100
        | 0x00010000
        | 0x00040000
        | 0x00080000
        | 0x10000000
        | 0x40000000
    )
    rows: list[bytes] = []
    try:
        ace_count = dacl.GetAceCount()
        for index in range(ace_count):
            ace = dacl.GetAce(index)
            header, mask, sid = ace[0], int(ace[1]), ace[2]
            ace_type, ace_flags = int(header[0]), int(header[1])
            sid_text = win32security.ConvertSidToStringSid(sid)
            _require(
                ace_type != win32security.ACCESS_ALLOWED_ACE_TYPE
                or not (mask & write_mask)
                or sid_text in privileged,
                f"{context}_unprivileged_write_ace_rejected",
            )
            rows.append(f"{ace_type}\t{ace_flags}\t{mask}\t{sid_text}\n".encode("ascii"))
    except (OSError, AttributeError, IndexError, TypeError):
        _fail(f"{context}_acl_parse_rejected")
    _require(bool(rows), f"{context}_empty_dacl_rejected")
    return {
        "owner_sid": owner_sid,
        "expected_owner_sid": expected_owner_sid,
        "unprivileged_write_ace_present": False,
        "dacl_ace_count": len(rows),
        "dacl_inventory_sha256": sha256_bytes(b"".join(rows)),
    }


def _authenticode_identity(path: Path, *, context: str) -> dict[str, str]:
    """Read Authenticode identity through the exact reviewed Windows helper."""

    helper_expected = PINNED_WINDOWS_POWERSHELL
    helper = canonical_existing_file(
        str(helper_expected["absolute_path"]), context="authenticode_powershell"
    )
    helper_raw = read_bound_file(
        helper,
        context="authenticode_powershell_binary",
        maximum=512 * 1024 * 1024,
        require_single_link=False,
    )
    _require(
        sha256_bytes(helper_raw) == helper_expected["sha256"],
        "authenticode_powershell_digest_drift",
    )
    _windows_acl_identity(
        helper,
        context="authenticode_powershell",
        expected_owner_sid=str(helper_expected["owner_sid"]),
    )
    path_literal = str(path).replace("'", "''")
    command = (
        "$ErrorActionPreference='Stop';"
        "$s=Microsoft.PowerShell.Security\\Get-AuthenticodeSignature "
        f"-LiteralPath '{path_literal}';"
        "[ordered]@{status=$s.Status.ToString();"
        "subject=$s.SignerCertificate.Subject;"
        "thumbprint=$s.SignerCertificate.Thumbprint}|ConvertTo-Json -Compress"
    )
    stdout, stderr = _run_bound(
        helper,
        (
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "AllSigned",
            "-Command",
            command,
        ),
        context=f"{context}_authenticode",
    )
    _require(not stderr.strip(), f"{context}_authenticode_stderr_rejected")
    value = strict_json(stdout.strip(), context=f"{context}_authenticode")
    _require(
        type(value) is dict
        and set(value) == {"status", "subject", "thumbprint"}
        and all(type(value[key]) is str for key in value),
        f"{context}_authenticode_payload_rejected",
    )
    return {
        "status": value["status"],
        "subject": value["subject"],
        "thumbprint": value["thumbprint"],
    }


def executable_inventory(
    *, git_executable: str, gh_executable: str, ssh_executable: str
) -> dict[str, Any]:
    _require(os.name == "nt", "pinned_executable_inventory_requires_windows")
    paths = {
        "git": canonical_existing_file(git_executable, context="git_executable"),
        "gh": canonical_existing_file(gh_executable, context="gh_executable"),
        "ssh": canonical_existing_file(ssh_executable, context="ssh_executable"),
        "python": canonical_existing_file(sys.executable, context="python_executable"),
    }
    arguments = {
        "git": ("--version",),
        "gh": ("--version",),
        "ssh": ("-V",),
        "python": ("--version",),
    }
    result: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        raw = read_bound_file(
            path,
            context=f"{name}_binary",
            maximum=512 * 1024 * 1024,
            require_single_link=name != "git" and name != "ssh",
        )
        version = _command_version(path, arguments[name], context=name)
        item: dict[str, Any] = {
            "absolute_path": str(path),
            "sha256": sha256_bytes(raw),
            "version": version,
            "regular_file": True,
            "symlink_or_reparse": False,
            "path_lookup_used": False,
            "hardlink_count": os.stat(path).st_nlink,
        }
        if name in PINNED_WINDOWS_EXECUTABLES:
            expected = PINNED_WINDOWS_EXECUTABLES[name]
            _require(
                path == Path(str(expected["absolute_path"]))
                and item["sha256"] == expected["sha256"]
                and version == expected["version"],
                f"{name}_pinned_identity_drift",
            )
            acl = _windows_acl_identity(
                path,
                context=f"{name}_executable",
                expected_owner_sid=str(expected["owner_sid"]),
            )
            signature = _authenticode_identity(path, context=name)
            _require(
                signature["status"] == "Valid"
                and signature["subject"] == expected["authenticode_subject"]
                and signature["thumbprint"] == expected["authenticode_thumbprint"],
                f"{name}_authenticode_identity_drift",
            )
            item.update(
                {
                    "pinned_reviewed_identity": True,
                    "acl": acl,
                    "authenticode": signature,
                    "authenticode_validated_by_pinned_helper": True,
                }
            )
            if name == "git":
                runtime_path = canonical_existing_file(
                    str(expected["runtime_absolute_path"]),
                    context="git_runtime_executable",
                )
                runtime_raw = read_bound_file(
                    runtime_path,
                    context="git_runtime_binary",
                    maximum=512 * 1024 * 1024,
                    require_single_link=False,
                )
                runtime_version = _command_version(
                    runtime_path, arguments[name], context="git_runtime"
                )
                _require(
                    sha256_bytes(runtime_raw) == expected["runtime_sha256"]
                    and runtime_version == expected["version"],
                    "git_runtime_pinned_identity_drift",
                )
                runtime_acl = _windows_acl_identity(
                    runtime_path,
                    context="git_runtime_executable",
                    expected_owner_sid=str(expected["owner_sid"]),
                )
                runtime_signature = _authenticode_identity(runtime_path, context="git_runtime")
                _require(
                    runtime_signature["status"] == "Valid"
                    and runtime_signature["subject"] == expected["authenticode_subject"]
                    and runtime_signature["thumbprint"] == expected["authenticode_thumbprint"],
                    "git_runtime_authenticode_identity_drift",
                )
                item["resolved_runtime"] = {
                    "absolute_path": str(runtime_path),
                    "sha256": sha256_bytes(runtime_raw),
                    "version": runtime_version,
                    "hardlink_count": os.stat(runtime_path).st_nlink,
                    "acl": runtime_acl,
                    "authenticode": runtime_signature,
                }
        else:
            item["pinned_runtime_manifest_authority"] = True
        result[name] = item
    return result


def _distribution_inventory(
    name: str,
    *,
    manifest: Mapping[str, Any],
    manifest_revalidation_sha256: str,
) -> dict[str, Any]:
    archives = manifest.get("archives")
    files = manifest.get("files")
    _require(type(archives) is list and type(files) is dict, "dependency_manifest_rejected")
    assert isinstance(archives, list) and isinstance(files, dict)
    _require(
        all(type(item) is dict for item in archives)
        and all(type(path) is str and type(item) is dict for path, item in files.items()),
        "dependency_manifest_inventory_schema_rejected",
    )
    matching = [item for item in archives if item.get("distribution") == name]
    _require(len(matching) == 1, f"dependency_{name}_archive_binding_rejected")
    archive = matching[0]
    rows: list[bytes] = []
    total = 0
    for relative, item in sorted(files.items()):
        if item.get("archive") != archive.get("filename"):
            continue
        _require(
            type(item.get("bytes")) is int
            and item["bytes"] >= 0
            and type(item.get("sha256")) is str
            and SHA256_RE.fullmatch(item["sha256"]) is not None,
            f"dependency_{name}_manifest_file_rejected",
        )
        total += item["bytes"]
        rows.append(f"{relative}\t{item['bytes']}\t{item['sha256']}\n".encode("utf-8"))
    _require(bool(rows), f"dependency_{name}_inventory_empty")
    return {
        "distribution": name,
        "version": archive["version"],
        "archive_filename": archive["filename"],
        "archive_sha256": archive["sha256"],
        "file_count": len(rows),
        "total_bytes": total,
        "inventory_sha256": sha256_bytes(b"".join(rows)),
        "actual_files_hashed": True,
        "actual_tree_revalidation_sha256": manifest_revalidation_sha256,
        "record_metadata_trusted": False,
        "wheel_archive_manifest_authoritative": True,
        "bytecode_excluded": True,
    }


def _normalized_distribution_name(value: str) -> str:
    normalized = re.sub(r"[-_.]+", "-", value).lower()
    _require(bool(normalized), "dependency_distribution_name_rejected")
    return normalized


def _parse_operator_lock(raw: bytes) -> dict[str, dict[str, str]]:
    """Accept only the one hash-locked, wheel-only CPython 3.13 environment."""

    try:
        text = raw.decode("ascii", errors="strict")
    except UnicodeDecodeError:
        _fail("operator_dependency_lock_encoding_rejected")
    _require("\r" not in text and text.endswith("\n"), "operator_dependency_lock_format_rejected")
    options: list[str] = []
    requirements: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("--"):
            _require(
                line in {"--require-hashes", "--only-binary=:all:"},
                "operator_dependency_lock_option_rejected",
            )
            options.append(line)
            continue
        match = LOCK_REQUIREMENT_RE.fullmatch(line)
        _require(match is not None, "operator_dependency_lock_requirement_rejected")
        assert match is not None
        name = _normalized_distribution_name(match.group("name"))
        _require(name not in requirements, "operator_dependency_lock_duplicate")
        requirements[name] = {
            "version": match.group("version"),
            "wheel_sha256": match.group("sha256"),
        }
    _require(
        options == ["--require-hashes", "--only-binary=:all:"],
        "operator_dependency_lock_options_rejected",
    )
    _require(
        requirements == LOCKED_DISTRIBUTIONS,
        "operator_dependency_lock_set_rejected",
    )
    return requirements


def _installed_distributions(site_root: Path) -> dict[str, importlib.metadata.Distribution]:
    installed: dict[str, importlib.metadata.Distribution] = {}
    for distribution in importlib.metadata.distributions(path=[str(site_root)]):
        raw_name = distribution.metadata.get("Name")
        _require(type(raw_name) is str and bool(raw_name), "dependency_distribution_name_rejected")
        assert isinstance(raw_name, str)
        name = _normalized_distribution_name(raw_name)
        _require(name not in installed, "dependency_distribution_duplicate")
        installed[name] = distribution
    return installed


def _startup_pth_inventory(site_roots: Sequence[Path]) -> dict[str, Any]:
    observed: dict[str, dict[str, Any]] = {}
    for root in site_roots:
        canonical_root = canonical_existing_directory(root, context="dependency_site_root")
        for path in sorted(canonical_root.rglob("*.pth")):
            relative = path.relative_to(canonical_root).as_posix()
            _require(relative not in observed, "dependency_pth_duplicate_path")
            raw = read_bound_file(
                path.resolve(strict=True), context="dependency_pth", maximum=65536
            )
            observed[relative] = {"bytes": len(raw), "sha256": sha256_bytes(raw)}
    _require(observed == ALLOWED_PTH_FILES, "dependency_pth_set_or_bytes_rejected")
    return {
        "allowed_files": observed,
        "unexpected_files_present": False,
        "all_startup_files_hashed": True,
    }


def dependency_inventory(
    repository_root: str, operator_python_root: str, operator_site_root: str
) -> dict[str, Any]:
    _require(os.name == "nt", "operator_dependency_platform_rejected")
    _require(
        platform.python_implementation() == "CPython"
        and sys.version_info[:3] == (3, 13, 15)
        and platform.machine().lower() in {"amd64", "x86_64"},
        "operator_dependency_interpreter_rejected",
    )
    root = canonical_existing_directory(repository_root, context="repository_root")
    python_root = canonical_existing_directory(operator_python_root, context="operator_python_root")
    site_root = canonical_existing_directory(operator_site_root, context="operator_site_root")
    manifest_revalidation = operator_bootstrap.revalidate_enabled_environment(
        root, python_root, site_root
    )
    manifest, manifest_raw = operator_bootstrap._load_manifest(root)
    _require(
        manifest_revalidation["site_tree"]["site_root"] == str(site_root),
        "operator_dependency_site_root_binding_rejected",
    )
    lock_path = canonical_existing_file(
        root / Path(OPERATOR_LOCK_RELATIVE), context="operator_dependency_lock"
    )
    lock_raw = read_bound_file(lock_path, context="operator_dependency_lock", maximum=128 * 1024)
    parsed = _parse_operator_lock(lock_raw)
    installed = _installed_distributions(site_root)
    _require(
        set(installed) == set(LOCKED_DISTRIBUTIONS),
        "operator_dependency_installed_set_rejected",
    )
    distributions: dict[str, Any] = {}
    tree_revalidation_sha256 = sha256_bytes(canonical_json(manifest_revalidation["site_tree"]))
    for name, expected in sorted(parsed.items()):
        distribution = installed[name]
        _require(
            distribution.version == expected["version"],
            f"dependency_{name}_version_rejected",
        )
        distribution_root = Path(str(distribution.locate_file(""))).resolve(strict=True)
        _require(distribution_root == site_root, f"dependency_{name}_site_root_rejected")
        distributions[name] = {
            **_distribution_inventory(
                name,
                manifest=manifest,
                manifest_revalidation_sha256=tree_revalidation_sha256,
            ),
            "locked_version": expected["version"],
            "source_wheel_sha256": expected["wheel_sha256"],
        }
    return {
        "schema_version": 1,
        "kind": "explainiverse-operator-dependency-inventory",
        "target": "CPython 3.13.15 Windows AMD64",
        "lock": {
            "relative_path": OPERATOR_LOCK_RELATIVE,
            "absolute_path": str(lock_path),
            "bytes": len(lock_raw),
            "sha256": sha256_bytes(lock_raw),
            "require_hashes": True,
            "wheels_only": True,
        },
        "distributions": distributions,
        "installed_distribution_set_exact": True,
        "startup_pth": _startup_pth_inventory((site_root,)),
        "site_manifest": {
            "relative_path": OPERATOR_MANIFEST_RELATIVE,
            "bytes": len(manifest_raw),
            "sha256": sha256_bytes(manifest_raw),
        },
        "wheel_derived_site_manifest": manifest_revalidation,
    }


def _module_resolution(
    module: str,
    distribution_name: str,
    *,
    site_root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    spec = importlib.util.find_spec(module)
    _require(spec is not None and spec.origin is not None, f"module_{module}_unresolved")
    assert spec is not None and spec.origin is not None
    origin = canonical_existing_file(spec.origin, context=f"module_{module}_origin")
    try:
        relative = origin.relative_to(site_root).as_posix()
    except ValueError:
        _fail(f"module_{module}_outside_distribution")
    archives = manifest.get("archives")
    files = manifest.get("files")
    _require(type(archives) is list and type(files) is dict, "module_manifest_rejected")
    assert isinstance(archives, list) and isinstance(files, dict)
    _require(
        all(type(item) is dict for item in archives)
        and all(type(path) is str and type(item) is dict for path, item in files.items()),
        "module_manifest_inventory_schema_rejected",
    )
    matching = [item for item in archives if item.get("distribution") == distribution_name]
    _require(len(matching) == 1, f"module_{module}_archive_binding_rejected")
    expected = files.get(relative)
    _require(
        type(expected) is dict and expected.get("archive") == matching[0]["filename"],
        f"module_{module}_outside_distribution",
    )
    assert isinstance(expected, dict)
    raw_origin = read_bound_file(
        origin, context=f"module_{module}_origin", maximum=64 * 1024 * 1024
    )
    _require(
        len(raw_origin) == expected.get("bytes")
        and sha256_bytes(raw_origin) == expected.get("sha256"),
        f"module_{module}_origin_digest_rejected",
    )
    locations = tuple(spec.submodule_search_locations or ())
    search_roots: list[str] = []
    for raw_location in locations:
        location = canonical_existing_directory(
            raw_location, context=f"module_{module}_search_root"
        )
        try:
            location.relative_to(site_root)
        except ValueError:
            _fail(f"module_{module}_search_root_outside_distribution")
        relative_location = location.relative_to(site_root).as_posix()
        _require(
            any(
                path == relative_location or path.startswith(relative_location + "/")
                for path in files
            ),
            f"module_{module}_search_root_unbound",
        )
        search_roots.append(str(location))
    return {
        "module": module,
        "distribution": distribution_name,
        "origin": str(origin),
        "origin_sha256": sha256_bytes(raw_origin),
        "search_roots": search_roots,
        "distribution_root": str(site_root),
        "distribution_root_sha256": sha256_bytes(str(site_root).encode("utf-8")),
        "origin_present_in_hashed_distribution_inventory": True,
    }


def interpreter_runtime_inventory(
    repository_root: str, operator_python_root: str, operator_site_root: str
) -> dict[str, Any]:
    _require(
        sys.flags.isolated == 1
        and sys.flags.safe_path
        and sys.flags.ignore_environment == 1
        and sys.flags.no_user_site == 1
        and sys.flags.no_site == 1
        and sys.dont_write_bytecode,
        "interpreter_secure_launch_flags_missing",
    )
    root = canonical_existing_directory(repository_root, context="repository_root")
    python_root = canonical_existing_directory(operator_python_root, context="operator_python_root")
    site_root = canonical_existing_directory(operator_site_root, context="operator_site_root")
    manifest, _ = operator_bootstrap._load_manifest(root)
    revalidation = operator_bootstrap.revalidate_enabled_environment(root, python_root, site_root)
    try:
        working = Path.cwd().resolve(strict=True)
        first = Path(sys.path[0] or os.curdir).resolve(strict=True)
    except OSError:
        _fail("interpreter_import_root_unavailable")
    _require(
        working != root
        and root not in working.parents
        and working != python_root
        and python_root not in working.parents
        and working != site_root
        and site_root not in working.parents,
        "interpreter_import_root_mismatch",
    )
    activation_roots = tuple(
        canonical_existing_directory(value, context="operator_activation_root")
        for value in (
            site_root,
            site_root / "win32",
            site_root / "win32" / "lib",
            site_root / "pythonwin",
        )
    )
    allowed_paths = {
        python_root,
        python_root / "python313.zip",
        *activation_roots,
    }
    path_entries: list[dict[str, Any]] = []
    for raw in sys.path:
        candidate = Path(raw or os.curdir)
        _require(candidate.is_absolute(), "interpreter_sys_path_relative")
        resolved = candidate.resolve(strict=False)
        _require(
            resolved in allowed_paths and resolved != root and root not in resolved.parents,
            "interpreter_sys_path_outside_bound_roots",
        )
        entry: dict[str, Any] = {
            "absolute_path": str(resolved),
            "path_sha256": sha256_bytes(str(resolved).encode("utf-8")),
        }
        if resolved.exists():
            _require(
                (resolved.is_dir() or resolved.is_file())
                and not resolved.is_symlink()
                and not _is_reparse(resolved),
                "interpreter_sys_path_entry_rejected",
            )
            entry["kind"] = "directory" if resolved.is_dir() else "file"
            if resolved.is_file():
                entry["content_sha256"] = sha256_bytes(
                    read_bound_file(
                        resolved,
                        context="interpreter_sys_path_file",
                        maximum=512 * 1024 * 1024,
                    )
                )
        else:
            _fail("interpreter_sys_path_missing")
        path_entries.append(entry)
    _require(
        len({item["absolute_path"] for item in path_entries}) == len(path_entries),
        "interpreter_sys_path_duplicate",
    )
    prefixes: dict[str, dict[str, Any]] = {}
    for name, raw in (("prefix", sys.prefix), ("base_prefix", sys.base_prefix)):
        prefix = canonical_existing_directory(raw, context=f"python_{name}")
        _require(prefix == python_root, f"python_{name}_root_rejected")
        prefixes[name] = {
            "absolute_path": str(prefix),
            "path_sha256": sha256_bytes(str(prefix).encode("utf-8")),
        }
    module_bindings = {
        "_cffi_backend": "cffi",
        "cffi": "cffi",
        "cryptography": "cryptography",
        "pycparser": "pycparser",
    }
    if os.name == "nt":
        module_bindings.update({"win32api": "pywin32", "win32security": "pywin32"})
    resolutions = {
        module: _module_resolution(
            module,
            distribution,
            site_root=site_root,
            manifest=manifest,
        )
        for module, distribution in sorted(module_bindings.items())
    }
    _require(
        {value["distribution_root"] for value in resolutions.values()} == {str(site_root)},
        "interpreter_module_site_root_drift",
    )
    return {
        "secure_flags": {
            "isolated": True,
            "safe_path": True,
            "ignore_environment": True,
            "no_user_site": True,
            "no_site": True,
            "dont_write_bytecode": True,
        },
        "working_directory": str(working),
        "sys_path_first": str(first),
        "sys_path": path_entries,
        "sys_path_sha256": sha256_bytes(canonical_json(path_entries)),
        "repository_present_in_sys_path": False,
        "prefixes": prefixes,
        "site_package_roots": [
            {
                "absolute_path": str(site_root),
                "path_sha256": sha256_bytes(str(site_root).encode("utf-8")),
            }
        ],
        "module_resolutions": resolutions,
        "pinned_runtime_and_site_revalidation": revalidation,
    }


def _git(
    git_executable: Path,
    root: Path,
    arguments: Sequence[str],
    *,
    context: str,
) -> bytes:
    environment = child_environment()
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
    stdout, stderr = _run_bound(
        git_executable,
        ("--no-pager", *arguments),
        context=context,
        cwd=root,
        environment=environment,
    )
    _require(not stderr.strip(), f"{context}_stderr_rejected")
    return stdout


def _github_json(
    transport: GhCliTransport, path: str, *, context: str
) -> tuple[dict[str, Any], str]:
    response: GitHubResponse | None = None
    try:
        response = transport.request("GET", path)
        _require(
            response.method == "GET" and response.path == path and response.status_code == 200,
            f"{context}_response_rejected",
        )
        raw = bytes(response.body)
        value = strict_json(raw, context=context)
        _require(type(value) is dict, f"{context}_not_object")
        return value, sha256_bytes(raw)
    finally:
        if response is not None:
            response.destroy()


def repository_inventory(
    *,
    repository_root: str,
    git_executable: str,
    github: GhCliTransport,
    expected_head_sha: str,
    supplied_ref: str,
) -> dict[str, Any]:
    _require(COMMIT_RE.fullmatch(expected_head_sha) is not None, "expected_head_sha_rejected")
    _require(supplied_ref in set(PHASE_REFS.values()), "supplied_ref_rejected")
    root = canonical_existing_directory(repository_root, context="repository_root")
    git_path = canonical_existing_file(git_executable, context="git_executable")
    top = _git(git_path, root, ("rev-parse", "--show-toplevel"), context="git_root")
    try:
        observed_root = Path(top.decode("utf-8", errors="strict").strip()).resolve(strict=True)
    except (UnicodeDecodeError, OSError):
        _fail("git_root_encoding_rejected")
    _require(observed_root == root, "repository_root_mismatch")
    head = _git(git_path, root, ("rev-parse", "HEAD"), context="git_head").decode("ascii").strip()
    _require(head == expected_head_sha, "local_head_sha_drift")
    origin_url = (
        _git(
            git_path,
            root,
            ("remote", "get-url", "origin"),
            context="git_origin",
        )
        .decode("utf-8", errors="strict")
        .strip()
    )
    _require(origin_url == EXPECTED_ORIGIN_URL, "repository_origin_url_drift")
    status = _git(
        git_path,
        root,
        ("status", "--porcelain=v1", "-z", "--untracked-files=all"),
        context="git_status",
    )
    _require(status == b"", "repository_worktree_not_clean")
    tree_sha = (
        _git(git_path, root, ("rev-parse", "HEAD^{tree}"), context="git_tree")
        .decode("ascii")
        .strip()
    )
    _require(COMMIT_RE.fullmatch(tree_sha) is not None, "git_tree_sha_rejected")
    tree = _git(
        git_path,
        root,
        ("ls-tree", "-r", "--full-tree", "-z", "HEAD"),
        context="git_tree_inventory",
    )
    tree_objects: dict[str, str] = {}
    for row in tree.split(b"\0"):
        if not row:
            continue
        try:
            metadata, path_raw = row.split(b"\t", 1)
            mode, kind, object_sha = metadata.decode("ascii").split(" ")
            relative = path_raw.decode("utf-8", errors="strict")
        except (UnicodeDecodeError, ValueError):
            _fail("git_tree_inventory_parse_rejected")
        _require(
            mode in {"100644", "100755"}
            and kind == "blob"
            and COMMIT_RE.fullmatch(object_sha) is not None
            and relative not in tree_objects,
            "git_tree_inventory_entry_rejected",
        )
        tree_objects[relative] = object_sha
    index_flags = _git(
        git_path,
        root,
        ("ls-files", "-v", "-z"),
        context="git_index_flags",
    )
    indexed: set[str] = set()
    for row in index_flags.split(b"\0"):
        if not row:
            continue
        try:
            flag = row[:1].decode("ascii")
            relative = row[2:].decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            _fail("git_index_flags_encoding_rejected")
        _require(
            len(row) >= 3 and row[1:2] == b" " and flag == "H",
            "git_index_nonordinary_flag_rejected",
        )
        indexed.add(relative)
    _require(indexed == set(tree_objects), "git_index_inventory_drift")
    names_raw = _git(
        git_path,
        root,
        ("ls-tree", "-r", "--full-tree", "--name-only", "HEAD"),
        context="git_tree_names",
    )
    try:
        tracked = set(names_raw.decode("utf-8", errors="strict").splitlines())
    except UnicodeDecodeError:
        _fail("git_tree_names_encoding_rejected")
    _require(set(CRITICAL_SOURCE_PATHS).issubset(tracked), "critical_source_not_tracked")
    sources: dict[str, dict[str, Any]] = {}
    for relative in CRITICAL_SOURCE_PATHS:
        source = canonical_existing_file(root / Path(relative), context="critical_source")
        raw = read_bound_file(source, context="critical_source", maximum=16 * 1024 * 1024)
        git_blob_sha = hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()
        _require(git_blob_sha == tree_objects[relative], "critical_source_git_blob_drift")
        sources[relative] = {
            "bytes": len(raw),
            "sha256": sha256_bytes(raw),
            "git_blob_sha": git_blob_sha,
        }

    ref_path = f"/repos/{REPOSITORY}/git/ref/{supplied_ref.removeprefix('refs/')}"
    ref, ref_response_sha256 = _github_json(github, ref_path, context="github_ref")
    obj = ref.get("object")
    _require(
        ref.get("ref") == supplied_ref and type(obj) is dict,
        "github_ref_binding_rejected",
    )
    assert isinstance(obj, dict)
    remote_type = obj.get("type")
    remote_object_sha = obj.get("sha")
    _require(
        remote_type in {"commit", "tag"}
        and type(remote_object_sha) is str
        and COMMIT_RE.fullmatch(remote_object_sha) is not None,
        "github_ref_object_rejected",
    )
    assert isinstance(remote_object_sha, str)
    remote_target_sha = remote_object_sha
    annotated_tag_response_sha256: str | None = None
    if remote_type == "tag":
        _require(supplied_ref == runtime.PUBLICATION_REF, "unexpected_annotated_tag_ref")
        tag, annotated_tag_response_sha256 = _github_json(
            github,
            f"/repos/{REPOSITORY}/git/tags/{remote_object_sha}",
            context="github_tag",
        )
        target = tag.get("object")
        verification = tag.get("verification")
        _require(
            tag.get("tag") == runtime.PUBLICATION_TAG
            and type(target) is dict
            and target.get("type") == "commit"
            and type(target.get("sha")) is str
            and COMMIT_RE.fullmatch(target["sha"]) is not None
            and type(verification) is dict
            and verification.get("verified") is True
            and verification.get("reason") == "valid",
            "github_tag_binding_rejected",
        )
        assert isinstance(target, dict)
        remote_target_sha = target["sha"]
    _require(remote_target_sha == expected_head_sha, "remote_head_sha_drift")
    return {
        "repository": REPOSITORY,
        "absolute_root": str(root),
        "origin_url": origin_url,
        "head_sha": head,
        "tree_object_sha": tree_sha,
        "tree_inventory_sha256": sha256_bytes(tree),
        "clean_tracked_and_untracked": True,
        "supplied_ref": supplied_ref,
        "remote_object_type": remote_type,
        "remote_object_sha": remote_object_sha,
        "remote_target_sha": remote_target_sha,
        "remote_ref_response_sha256": ref_response_sha256,
        "annotated_tag_response_sha256": annotated_tag_response_sha256,
        "critical_sources": sources,
        "git_configuration": {
            "system_config_disabled": True,
            "system_config_path": os.devnull,
            "global_config_path": os.devnull,
            "system_attributes_disabled": True,
            "repository_fsmonitor_overridden_false": True,
            "repository_untracked_cache_overridden_false": True,
            "pager_disabled": True,
            "terminal_prompt_disabled": True,
            "optional_locks_disabled": True,
        },
    }


def capture_inventory(
    *,
    repository_root: str,
    operator_python_root: str,
    operator_site_root: str,
    git_executable: str,
    gh_executable: str,
    ssh_executable: str,
    expected_head_sha: str,
    supplied_ref: str,
) -> dict[str, Any]:
    executables = executable_inventory(
        git_executable=git_executable,
        gh_executable=gh_executable,
        ssh_executable=ssh_executable,
    )
    gh_receipt = executables["gh"]
    github = GhCliTransport(
        executable_path=str(gh_receipt["absolute_path"]),
        executable_sha256=str(gh_receipt["sha256"]),
    )
    repository = repository_inventory(
        repository_root=repository_root,
        git_executable=str(executables["git"]["absolute_path"]),
        github=github,
        expected_head_sha=expected_head_sha,
        supplied_ref=supplied_ref,
    )
    return {
        "schema_version": 1,
        "kind": INVENTORY_KIND,
        "python_implementation": platform.python_implementation(),
        "interpreter_runtime": interpreter_runtime_inventory(
            repository_root, operator_python_root, operator_site_root
        ),
        "executables": executables,
        "dependencies": dependency_inventory(
            repository_root, operator_python_root, operator_site_root
        ),
        "repository": repository,
    }


def inspection_receipt(
    inventory: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    phase: str,
    environment: Mapping[str, Any],
    secure_launch: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_inventory = strict_json(canonical_json(inventory), context="inventory")
    normalized_contract = strict_json(canonical_json(contract), context="contract")
    return {
        "schema_version": 1,
        "kind": INSPECTION_KIND,
        "inventory": normalized_inventory,
        "inventory_sha256": sha256_bytes(canonical_json(normalized_inventory)),
        "credentialed_provider_contact": False,
        "provider_mutation": False,
        "contract": normalized_contract,
        "phase": phase,
        "environment": strict_json(canonical_json(environment), context="environment"),
        "secure_launch": strict_json(canonical_json(secure_launch), context="secure_launch"),
    }


def load_inspection_receipt(
    path: str,
    *,
    expected_file_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(
        SHA256_RE.fullmatch(expected_file_sha256) is not None, "inspection_file_sha256_rejected"
    )
    value, raw = read_canonical_json_file(path, context="inspection_receipt")
    _require(sha256_bytes(raw) == expected_file_sha256, "inspection_file_digest_mismatch")
    _require(
        set(value)
        == {
            "schema_version",
            "kind",
            "inventory",
            "inventory_sha256",
            "credentialed_provider_contact",
            "provider_mutation",
            "contract",
            "phase",
            "environment",
            "secure_launch",
        }
        and value["schema_version"] == 1
        and value["kind"] == INSPECTION_KIND
        and value["credentialed_provider_contact"] is False
        and value["provider_mutation"] is False
        and type(value["inventory"]) is dict
        and type(value["contract"]) is dict,
        "inspection_receipt_schema_rejected",
    )
    _require(
        value["phase"] in PHASE_REFS
        and type(value["environment"]) is dict
        and type(value["secure_launch"]) is dict
        and value["secure_launch"].get("ignore_environment") is True
        and value["secure_launch"].get("no_user_site") is True
        and value["secure_launch"].get("dont_write_bytecode") is True,
        "inspection_launch_binding_rejected",
    )
    inventory = value["inventory"]
    _require(
        inventory.get("schema_version") == 1
        and inventory.get("kind") == INVENTORY_KIND
        and SHA256_RE.fullmatch(str(value["inventory_sha256"])) is not None
        and sha256_bytes(canonical_json(inventory)) == value["inventory_sha256"],
        "inspection_inventory_binding_rejected",
    )
    return value, inventory


def validate_inventory_matches(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> None:
    _require(canonical_json(expected) == canonical_json(observed), "operator_inventory_drift")


def validate_phase_ref(phase: str, supplied_ref: str) -> None:
    _require(phase in PHASE_REFS, "operator_phase_rejected")
    _require(PHASE_REFS[phase] == supplied_ref, "operator_phase_ref_mismatch")


def ensure_path_outside_repository(path: str, repository_root: str, *, context: str) -> Path:
    candidate = Path(path)
    _require(candidate.is_absolute(), f"{context}_not_absolute")
    root = canonical_existing_directory(repository_root, context="repository_root")
    normalized = Path(os.path.abspath(candidate))
    try:
        resolved = normalized.resolve(strict=False)
    except OSError:
        _fail(f"{context}_not_canonical")
    _require(normalized == resolved, f"{context}_not_canonical")
    try:
        common = Path(os.path.commonpath((str(root), str(resolved))))
    except ValueError:
        return resolved
    _require(
        common not in {root, resolved},
        f"{context}_not_disjoint_from_repository",
    )
    return resolved


def require_pairwise_disjoint_paths(paths: Mapping[str, str | os.PathLike[str]]) -> None:
    """Reject equality or ancestor relationships across security roots."""

    normalized: dict[str, Path] = {}
    for name, raw_path in paths.items():
        _require(bool(name) and name not in normalized, "disjoint_path_name_rejected")
        candidate = Path(raw_path)
        _require(candidate.is_absolute(), f"{name}_not_absolute")
        try:
            resolved = candidate.resolve(strict=False)
        except OSError:
            _fail(f"{name}_not_canonical")
        _require(Path(os.path.abspath(candidate)) == resolved, f"{name}_not_canonical")
        normalized[name] = resolved
    items = tuple(normalized.items())
    for index, (left_name, left) in enumerate(items):
        for right_name, right in items[index + 1 :]:
            try:
                common = Path(os.path.commonpath((str(left), str(right))))
            except ValueError:
                continue
            _require(
                common not in {left, right},
                f"{left_name}_and_{right_name}_not_disjoint",
            )


def validate_anonymous_fd(fd: int, *, context: str) -> dict[str, Any]:
    """Require a caller-owned anonymous pipe/socket, never a file or terminal."""

    _require(type(fd) is int and fd >= 3, f"{context}_fd_rejected")
    try:
        descriptor = os.fstat(fd)
    except OSError:
        _fail(f"{context}_fd_unavailable")
    _require(not os.isatty(fd), f"{context}_terminal_rejected")
    _require(
        not stat.S_ISREG(descriptor.st_mode)
        and not stat.S_ISDIR(descriptor.st_mode)
        and not stat.S_ISCHR(descriptor.st_mode)
        and not stat.S_ISBLK(descriptor.st_mode),
        f"{context}_anonymous_transport_rejected",
    )
    kind: str
    owner_verified = True
    if os.name == "nt":
        import ctypes
        import msvcrt

        handle = msvcrt.get_osfhandle(fd)  # type: ignore[attr-defined]
        file_type = ctypes.WinDLL(  # type: ignore[attr-defined]
            "kernel32", use_last_error=True
        ).GetFileType(handle)
        _require(file_type == 3, f"{context}_not_pipe")
        kind = "anonymous-pipe"
    else:
        _require(
            stat.S_ISFIFO(descriptor.st_mode) or stat.S_ISSOCK(descriptor.st_mode),
            f"{context}_not_pipe_or_socket",
        )
        if hasattr(os, "geteuid"):
            _require(descriptor.st_uid == os.geteuid(), f"{context}_owner_mismatch")
        proc_link = Path(f"/proc/self/fd/{fd}")
        if proc_link.exists():
            try:
                target = os.readlink(proc_link)
            except OSError:
                _fail(f"{context}_fd_link_unavailable")
            _require(
                target.startswith("pipe:[") or target.startswith("socket:["),
                f"{context}_named_transport_rejected",
            )
        kind = "anonymous-pipe" if stat.S_ISFIFO(descriptor.st_mode) else "anonymous-socket"
    return {
        "kind": kind,
        "descriptor_owned_by_operator": True,
        "current_user_owner_verified": owner_verified,
        "regular_file": False,
        "terminal": False,
        "value_archived": False,
    }


def read_plan_confirmation(
    fd: int, *, expected_sha256: str, timeout_seconds: int = 30
) -> dict[str, Any]:
    _require(SHA256_RE.fullmatch(expected_sha256) is not None, "plan_sha256_rejected")
    receipt = validate_anonymous_fd(fd, context="plan_confirmation")
    try:
        os.set_blocking(fd, False)
    except OSError:
        _fail("plan_confirmation_nonblocking_unavailable")
    value = bytearray()
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            chunk = os.read(fd, 66 - len(value))
        except BlockingIOError:
            if time.monotonic() >= deadline:
                for index in range(len(value)):
                    value[index] = 0
                _fail("plan_confirmation_timeout")
            time.sleep(0.01)
            continue
        if not chunk:
            break
        value.extend(chunk)
        _require(len(value) <= 65, "plan_confirmation_too_large")
    try:
        supplied = bytes(value).decode("ascii", errors="strict")
    except UnicodeDecodeError:
        _fail("plan_confirmation_encoding_rejected")
    finally:
        for index in range(len(value)):
            value[index] = 0
        value.clear()
    _require(supplied == expected_sha256 + "\n", "plan_confirmation_digest_mismatch")
    return {
        **receipt,
        "confirmed_plan_sha256": expected_sha256,
        "confirmation_exact_line": True,
        "confirmation_read_after_plan": True,
    }


def close_owned_fd(fd: int | None) -> None:
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        _require(written > 0, "app_capture_publish_short_write")
        offset += written


def _sync_directory(directory: Path) -> None:
    if os.name == "nt":
        # Final ready-marker publication uses MOVEFILE_WRITE_THROUGH.  Bundle
        # files are flushed individually before that marker can become visible.
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive_file(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0),
        0o600,
    )
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    state = path.stat()
    _require(
        path.is_file()
        and not path.is_symlink()
        and not _is_reparse(path)
        and state.st_nlink == 1
        and state.st_size == len(payload),
        "app_capture_published_file_rejected",
    )


def _publish_no_replace(temporary: Path, destination: Path, payload: bytes) -> None:
    _require(not destination.exists(), "app_capture_ready_already_exists")
    _write_exclusive_file(temporary, payload)
    published = False
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            move_file = ctypes.WinDLL(  # type: ignore[attr-defined]
                "kernel32", use_last_error=True
            ).MoveFileExW
            move_file.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
            move_file.restype = wintypes.BOOL
            # MOVEFILE_WRITE_THROUGH without REPLACE_EXISTING is atomic and
            # cannot overwrite a raced ready marker.
            moved = move_file(str(temporary), str(destination), 0x8)
            if not moved:
                _fail("app_capture_ready_publish_failed")
        else:
            os.link(temporary, destination)
            temporary.unlink()
            _sync_directory(destination.parent)
        published = True
    finally:
        if not published:
            try:
                temporary.unlink()
            except OSError:
                pass
    _require(
        destination.is_file()
        and not destination.is_symlink()
        and not _is_reparse(destination)
        and destination.stat().st_nlink == 1
        and destination.read_bytes() == payload,
        "app_capture_ready_publish_ambiguous",
    )


def publish_app_capture_generation(
    receipt: EvidenceDirectoryReceipt,
    *,
    controller_resources: SealedControllerResources,
    staging_receipt: EvidenceDirectoryReceipt,
    phase: str,
    ordinal: int,
    generation: int,
    publication_nonce: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate and durably publish one immutable App capture generation."""

    _require(phase in PHASE_CAPTURE_COUNTS, "app_capture_publish_phase_rejected")
    _require(
        type(ordinal) is int and 1 <= ordinal <= PHASE_CAPTURE_COUNTS[phase],
        "app_capture_publish_ordinal_rejected",
    )
    _require(
        type(generation) is int and 1 <= generation <= 999999,
        "app_capture_publish_generation_rejected",
    )
    _require(
        type(publication_nonce) is str
        and re.fullmatch(r"[0-9a-f]{32}", publication_nonce) is not None,
        "app_capture_publication_nonce_rejected",
    )
    receipt.validate()
    staging_receipt.validate()
    root = Path(receipt.absolute_path)
    staging_root = Path(staging_receipt.absolute_path)
    require_pairwise_disjoint_paths(
        {
            "app_capture_inbox": root,
            "app_capture_staging": staging_root,
        }
    )
    _require(
        root != staging_root and receipt.receipt_sha256 != staging_receipt.receipt_sha256,
        "app_capture_staging_not_separate",
    )
    staging_children = tuple(staging_root.iterdir())
    _require(
        {child.name for child in staging_children} == {"capture.json", "pages"}
        and len(staging_children) == 2,
        "app_capture_staging_inventory_drift",
    )
    capture_json_path = staging_root / "capture.json"
    pages_directory = staging_root / "pages"
    staging_receipt.validate()
    source_capture, source_capture_raw = read_canonical_json_file(
        capture_json_path, context="app_capture_source_json"
    )
    source_pages = canonical_existing_directory(pages_directory, context="app_capture_source_pages")
    _require(source_pages.parent == staging_root, "app_capture_source_pages_escape")
    evidence = source_capture.get("evidence")
    _require(type(evidence) is list and bool(evidence), "app_capture_source_evidence_rejected")
    assert isinstance(evidence, list)
    pages: dict[str, bytes] = {}
    rows: list[dict[str, Any]] = []
    names: set[str] = set()
    digests: set[str] = set()
    for raw_item in evidence:
        staging_receipt.validate()
        _require(type(raw_item) is dict, "app_capture_source_evidence_item_rejected")
        assert isinstance(raw_item, dict)
        filename = raw_item.get("filename")
        _require(
            type(filename) is str
            and SAFE_CAPTURE_FILENAME_RE.fullmatch(filename) is not None
            and filename not in names,
            "app_capture_source_filename_rejected",
        )
        assert isinstance(filename, str)
        names.add(filename)
        page = read_bound_file(
            source_pages / filename,
            context="app_capture_source_page",
            maximum=MAX_CAPTURE_PAGE_BYTES,
        )
        digest = sha256_bytes(page)
        _require(digest not in digests, "app_capture_source_page_digest_repeated")
        digests.add(digest)
        _require(
            raw_item.get("bytes") == len(page) and raw_item.get("sha256") == digest,
            "app_capture_source_page_binding_rejected",
        )
        pages[filename] = page
        rows.append({"filename": filename, "bytes": len(page), "sha256": digest})
    source_children = tuple(source_pages.iterdir())
    _require(
        len(source_children) == len(names)
        and {
            child.name
            for child in source_children
            if child.is_file() and not child.is_symlink() and not _is_reparse(child)
        }
        == names,
        "app_capture_source_pages_inventory_drift",
    )
    staging_receipt.validate()

    def reader(filename: str) -> bytes:
        _require(filename in pages, "app_capture_source_page_not_declared")
        return pages[filename]

    accepted = TrustedAppCapture.from_mapping(
        source_capture,
        resources=controller_resources,
        evidence_reader=reader,
        now=now or datetime.now(timezone.utc),
    )
    _require(
        source_capture == accepted.normalized_capture
        and source_capture_raw == canonical_json(accepted.normalized_capture),
        "app_capture_source_not_normalized",
    )
    suffix = f"{ordinal:02d}-{generation:06d}"
    bundle = root / f"capture-{suffix}"
    ready = root / f"ready-{suffix}.json"
    for predecessor in range(1, generation):
        predecessor_suffix = f"{ordinal:02d}-{predecessor:06d}"
        predecessor_bundle = root / f"capture-{predecessor_suffix}"
        predecessor_ready = root / f"ready-{predecessor_suffix}.json"
        _require(
            predecessor_bundle.is_dir()
            and not predecessor_bundle.is_symlink()
            and not _is_reparse(predecessor_bundle)
            and predecessor_ready.is_file()
            and not predecessor_ready.is_symlink()
            and not _is_reparse(predecessor_ready),
            "app_capture_prior_generation_incomplete_requires_fresh_inbox",
        )
    _require(not bundle.exists() and not ready.exists(), "app_capture_generation_already_exists")
    os.mkdir(bundle, 0o700)
    pages_destination = bundle / "pages"
    os.mkdir(pages_destination, 0o700)
    try:
        for filename, page in pages.items():
            receipt.validate()
            _write_exclusive_file(pages_destination / filename, page)
        _sync_directory(pages_destination)
        receipt.validate()
        _write_exclusive_file(bundle / "capture.json", source_capture_raw)
        _sync_directory(bundle)
        pages_inventory_sha256 = sha256_bytes(canonical_json(rows))
        ready_value = {
            "schema_version": 1,
            "kind": CAPTURE_READY_KIND,
            "phase": phase,
            "ordinal": ordinal,
            "generation": generation,
            "publication_nonce": publication_nonce,
            "capture_directory": bundle.name,
            "capture_json_sha256": sha256_bytes(source_capture_raw),
            "pages_inventory_sha256": pages_inventory_sha256,
        }
        ready_raw = canonical_json(ready_value)
        temporary = bundle / f".ready-{secrets.token_hex(16)}.tmp"
        receipt.validate()
        staging_receipt.validate()
        _publish_no_replace(temporary, ready, ready_raw)
        receipt.validate()
        staging_receipt.validate()
    except BaseException:
        # No ready marker means the generation is permanently unconsumable.
        # Never delete or reuse possibly published bytes after an ambiguous
        # crash. The phase must abort and restart with a fresh inbox; skipping
        # an unannounced generation would deadlock the sequential consumer.
        raise
    return {
        "schema_version": 1,
        "kind": "explainiverse-installed-app-capture-publication",
        "phase": phase,
        "ordinal": ordinal,
        "generation": generation,
        "publication_nonce": publication_nonce,
        "capture_evidence_sha256": accepted.evidence_sha256,
        "capture_json_sha256": sha256_bytes(source_capture_raw),
        "pages_inventory_sha256": pages_inventory_sha256,
        "ready_marker": ready.name,
        "ready_marker_sha256": sha256_bytes(ready_raw),
        "ready_marker_published_last": True,
        "ready_marker_no_replace": True,
        "owner_private_directory_receipt_sha256": receipt.receipt_sha256,
        "owner_private_staging_receipt_sha256": staging_receipt.receipt_sha256,
        "staging_directory_separate": True,
        "raw_page_values_logged": False,
    }


@dataclass
class AppCaptureInbox:
    """Consume one fresh, atomically announced App capture per JIT job.

    A browser-side producer first writes an immutable generation directory and
    all raw pages, then exclusively publishes the canonical ``ready`` marker.
    Stale generations are retained as evidence but skipped; malformed or
    replayed generations fail closed.
    """

    receipt: EvidenceDirectoryReceipt
    controller_resources: SealedControllerResources
    phase: str
    poll_limit: int
    sleep: Callable[[float], None] = time.sleep
    poll_seconds: float = 1.0
    clock: Callable[[], datetime] = _utc_now
    _ordinal: int = 1
    _next_generation: int = 1
    _seen_ready_sha256: set[str] | None = None
    _seen_publication_nonces: set[str] | None = None
    _seen_capture_json_sha256: set[str] | None = None
    _seen_capture_evidence_sha256: set[str] | None = None
    _seen_page_sha256: set[str] | None = None
    _stale_sha256: list[str] | None = None
    _consumed_generations: list[dict[str, Any]] | None = None
    _final_inventory: dict[str, Any] | None = None
    _last_classified_at: datetime | None = None
    _stale_archive_sink: (
        Callable[[str, Mapping[str, Any], Mapping[str, bytes]], Mapping[str, Any]] | None
    ) = None

    def __post_init__(self) -> None:
        _require(self.phase in PHASE_CAPTURE_COUNTS, "app_capture_phase_rejected")
        _require(
            type(self.poll_limit) is int and self.poll_limit > 0, "app_capture_poll_limit_rejected"
        )
        _require(
            type(self.poll_seconds) in {int, float} and 0 <= self.poll_seconds <= 60,
            "app_capture_poll_seconds_rejected",
        )
        _require(callable(self.sleep), "app_capture_sleep_rejected")
        _require(callable(self.clock), "app_capture_clock_rejected")
        self.receipt.validate()
        _require(
            type(self.controller_resources) is SealedControllerResources,
            "app_capture_controller_resources_required",
        )
        object.__setattr__(self, "_seen_ready_sha256", set())
        object.__setattr__(self, "_seen_publication_nonces", set())
        object.__setattr__(self, "_seen_capture_json_sha256", set())
        object.__setattr__(self, "_seen_capture_evidence_sha256", set())
        object.__setattr__(self, "_seen_page_sha256", set())
        object.__setattr__(self, "_stale_sha256", [])
        object.__setattr__(self, "_consumed_generations", [])

    @property
    def expected_count(self) -> int:
        return PHASE_CAPTURE_COUNTS[self.phase]

    def bind_stale_archive_sink(
        self,
        sink: Callable[[str, Mapping[str, Any], Mapping[str, bytes]], Mapping[str, Any]],
    ) -> None:
        """Bind the evidence-journal archive before any stale generation is consumed."""

        _require(callable(sink), "app_capture_stale_archive_sink_rejected")
        _require(
            self._stale_archive_sink is None
            and self._ordinal == 1
            and self._next_generation == 1
            and not self._consumed_generations,
            "app_capture_stale_archive_sink_late_or_rebound",
        )
        object.__setattr__(self, "_stale_archive_sink", sink)

    def _archive_stale_generation(
        self,
        *,
        classified_at: str,
        generation_receipt: Mapping[str, Any],
        pages: Mapping[str, bytes],
    ) -> dict[str, Any]:
        sink = self._stale_archive_sink
        _require(sink is not None, "app_capture_stale_archive_sink_missing")
        assert sink is not None
        archive = strict_json(
            canonical_json(sink(classified_at, generation_receipt, pages)),
            context="app_capture_stale_archive",
        )
        keys = {
            "schema_version",
            "kind",
            "phase",
            "ordinal",
            "generation",
            "publication_nonce",
            "ready_marker_sha256",
            "capture_json_sha256",
            "classified_at",
            "archive_identity_sha256",
            "archive_directory",
            "files",
            "all_pages_exclusive_single_link",
            "archive_evidence_sha256",
        }
        _require(type(archive) is dict and set(archive) == keys, "app_capture_stale_archive_schema")
        identity_material = {
            "phase": self.phase,
            "ordinal": generation_receipt["ordinal"],
            "generation": generation_receipt["generation"],
            "publication_nonce": generation_receipt["publication_nonce"],
            "ready_marker_sha256": generation_receipt["ready_marker_sha256"],
            "capture_json_sha256": generation_receipt["capture_json_sha256"],
            "classified_at": classified_at,
        }
        identity_sha256 = sha256_bytes(canonical_json(identity_material))
        material = dict(archive)
        evidence_sha256 = material.pop("archive_evidence_sha256", None)
        _require(
            archive["schema_version"] == 1
            and archive["kind"] == "explainiverse-installed-app-stale-raw-archive"
            and all(archive[key] == value for key, value in identity_material.items())
            and archive["archive_identity_sha256"] == identity_sha256
            and archive["archive_directory"] == f"installed-app-pages/{identity_sha256}"
            and archive["files"] == generation_receipt["pages"]
            and archive["all_pages_exclusive_single_link"] is True
            and type(evidence_sha256) is str
            and SHA256_RE.fullmatch(evidence_sha256) is not None
            and sha256_bytes(canonical_json(material)) == evidence_sha256,
            "app_capture_stale_archive_binding_rejected",
        )
        return archive

    def _validate_root_inventory(self) -> None:
        self.receipt.validate()
        root = Path(self.receipt.absolute_path)
        ready_re = re.compile(r"ready-[0-9]{2}-[0-9]{6}\.json\Z")
        bundle_re = re.compile(r"capture-[0-9]{2}-[0-9]{6}\Z")
        for child in root.iterdir():
            accepted = (
                child.is_file()
                and not child.is_symlink()
                and not _is_reparse(child)
                and ready_re.fullmatch(child.name) is not None
            ) or (
                child.is_dir()
                and not child.is_symlink()
                and not _is_reparse(child)
                and bundle_re.fullmatch(child.name) is not None
            )
            _require(accepted, "app_capture_inbox_residue")

    def _generation(
        self, ordinal: int, generation: int
    ) -> tuple[dict[str, Any], Mapping[str, bytes], str, str, dict[str, Any]]:
        self._validate_root_inventory()
        root = Path(self.receipt.absolute_path)
        suffix = f"{ordinal:02d}-{generation:06d}"
        ready_path = root / f"ready-{suffix}.json"
        ready, ready_raw = read_canonical_json_file(ready_path, context="app_capture_ready_marker")
        ready_sha256 = sha256_bytes(ready_raw)
        assert self._seen_ready_sha256 is not None
        assert self._seen_publication_nonces is not None
        _require(ready_sha256 not in self._seen_ready_sha256, "app_capture_ready_replayed")
        nonce = ready.get("publication_nonce")
        _require(
            set(ready)
            == {
                "schema_version",
                "kind",
                "phase",
                "ordinal",
                "generation",
                "publication_nonce",
                "capture_directory",
                "capture_json_sha256",
                "pages_inventory_sha256",
            }
            and ready["schema_version"] == 1
            and ready["kind"] == CAPTURE_READY_KIND
            and ready["phase"] == self.phase
            and ready["ordinal"] == ordinal
            and ready["generation"] == generation
            and type(nonce) is str
            and re.fullmatch(r"[0-9a-f]{32}", nonce) is not None
            and nonce not in self._seen_publication_nonces
            and ready["capture_directory"] == f"capture-{suffix}"
            and SHA256_RE.fullmatch(str(ready["capture_json_sha256"])) is not None
            and SHA256_RE.fullmatch(str(ready["pages_inventory_sha256"])) is not None,
            "app_capture_ready_binding_rejected",
        )
        bundle = canonical_existing_directory(
            root / f"capture-{suffix}", context="app_capture_bundle"
        )
        _require(bundle.parent == root, "app_capture_bundle_escape")
        pages_dir = canonical_existing_directory(
            bundle / "pages", context="app_capture_pages_directory"
        )
        _require(pages_dir.parent == bundle, "app_capture_pages_escape")
        capture_path = bundle / "capture.json"
        capture, capture_raw = read_canonical_json_file(capture_path, context="app_capture")
        capture_json_sha256 = sha256_bytes(capture_raw)
        _require(
            capture_json_sha256 == ready["capture_json_sha256"],
            "app_capture_json_digest_mismatch",
        )
        assert self._seen_capture_json_sha256 is not None
        _require(
            capture_json_sha256 not in self._seen_capture_json_sha256,
            "app_capture_json_replayed",
        )
        evidence = capture.get("evidence")
        _require(type(evidence) is list and bool(evidence), "app_capture_evidence_rejected")
        assert isinstance(evidence, list)
        pages: dict[str, bytes] = {}
        expected_names: set[str] = set()
        generation_page_sha256: set[str] = set()
        rows: list[dict[str, Any]] = []
        newest_input_mtime = capture_path.stat().st_mtime_ns
        for evidence_item in evidence:
            _require(type(evidence_item) is dict, "app_capture_evidence_item_rejected")
            assert isinstance(evidence_item, dict)
            filename = evidence_item.get("filename")
            _require(
                type(filename) is str
                and SAFE_CAPTURE_FILENAME_RE.fullmatch(filename) is not None
                and filename not in expected_names,
                "app_capture_filename_rejected",
            )
            assert isinstance(filename, str)
            expected_names.add(filename)
            page_path = pages_dir / filename
            page = read_bound_file(
                page_path,
                context="app_capture_page",
                maximum=MAX_CAPTURE_PAGE_BYTES,
            )
            newest_input_mtime = max(newest_input_mtime, page_path.stat().st_mtime_ns)
            digest = sha256_bytes(page)
            assert self._seen_page_sha256 is not None
            _require(
                digest not in self._seen_page_sha256 and digest not in generation_page_sha256,
                "app_capture_page_replayed",
            )
            generation_page_sha256.add(digest)
            _require(
                evidence_item.get("bytes") == len(page) and evidence_item.get("sha256") == digest,
                "app_capture_page_binding_rejected",
            )
            pages[filename] = page
            rows.append({"filename": filename, "bytes": len(page), "sha256": digest})
        children = tuple(pages_dir.iterdir())
        actual_names = {
            child.name
            for child in children
            if child.is_file() and not child.is_symlink() and not _is_reparse(child)
        }
        _require(
            actual_names == expected_names and len(children) == len(expected_names),
            "app_capture_pages_inventory_drift",
        )
        _require(
            set(child.name for child in bundle.iterdir()) == {"capture.json", "pages"},
            "app_capture_bundle_residue",
        )
        pages_sha256 = sha256_bytes(canonical_json(rows))
        _require(
            pages_sha256 == ready["pages_inventory_sha256"],
            "app_capture_pages_digest_mismatch",
        )
        _require(
            ready_path.stat().st_mtime_ns >= newest_input_mtime,
            "app_capture_ready_not_published_last",
        )
        ready_after = read_bound_file(ready_path, context="app_capture_ready_marker_recheck")
        _require(
            ready_after == ready_raw and sha256_bytes(ready_after) == ready_sha256,
            "app_capture_ready_changed_during_bundle_read",
        )
        self.receipt.validate()
        self._seen_ready_sha256.add(ready_sha256)
        assert isinstance(nonce, str)
        self._seen_publication_nonces.add(nonce)
        self._seen_capture_json_sha256.add(capture_json_sha256)
        assert self._seen_page_sha256 is not None
        self._seen_page_sha256.update(generation_page_sha256)
        return (
            capture,
            pages,
            ready_sha256,
            nonce,
            {
                "ordinal": ordinal,
                "generation": generation,
                "publication_nonce": nonce,
                "ready_marker": ready_path.name,
                "ready_marker_bytes": len(ready_raw),
                "ready_marker_sha256": ready_sha256,
                "capture_directory": bundle.name,
                "capture_json_bytes": len(capture_raw),
                "capture_json_sha256": capture_json_sha256,
                "capture": capture,
                "pages": rows,
                "pages_inventory_sha256": pages_sha256,
            },
        )

    def __call__(self) -> tuple[Mapping[str, Any], Callable[[str], bytes]]:
        _require(self._ordinal <= self.expected_count, "app_capture_inbox_exhausted")
        for poll in range(self.poll_limit):
            root = Path(self.receipt.absolute_path)
            ready_path = root / f"ready-{self._ordinal:02d}-{self._next_generation:06d}.json"
            if not ready_path.exists():
                if poll + 1 < self.poll_limit:
                    self.sleep(self.poll_seconds)
                continue
            capture, pages, ready_sha256, _, generation_receipt = self._generation(
                self._ordinal, self._next_generation
            )
            self._next_generation += 1

            def reader(filename: str) -> bytes:
                _require(
                    type(filename) is str and filename in pages,
                    "app_capture_page_not_declared",
                )
                return pages[filename]

            classified_at = self.clock()
            _require(
                type(classified_at) is datetime
                and classified_at.tzinfo is not None
                and classified_at.utcoffset() == timezone.utc.utcoffset(None),
                "app_capture_classification_time_rejected",
            )
            _require(
                self._last_classified_at is None or classified_at >= self._last_classified_at,
                "app_capture_classification_time_regressed",
            )
            captured_at_value = capture.get("captured_at")
            _require(type(captured_at_value) is str, "app_capture_captured_at_rejected")
            assert isinstance(captured_at_value, str)
            try:
                captured_at = datetime.fromisoformat(captured_at_value.replace("Z", "+00:00"))
            except ValueError:
                _fail("app_capture_captured_at_rejected")
            _require(
                captured_at.tzinfo is not None
                and captured_at.astimezone(timezone.utc) <= classified_at,
                "app_capture_classified_before_capture",
            )
            classified_at_text = classified_at.isoformat()
            try:
                accepted = TrustedAppCapture.from_mapping(
                    capture,
                    resources=self.controller_resources,
                    evidence_reader=reader,
                    now=classified_at,
                )
            except ControllerError as exc:
                if str(exc) == "app_capture_stale":
                    assert self._stale_sha256 is not None
                    assert self._consumed_generations is not None
                    stale_archive = self._archive_stale_generation(
                        classified_at=classified_at_text,
                        generation_receipt=generation_receipt,
                        pages=pages,
                    )
                    object.__setattr__(self, "_last_classified_at", classified_at)
                    self._stale_sha256.append(ready_sha256)
                    self._consumed_generations.append(
                        {
                            **generation_receipt,
                            "classified_at": classified_at_text,
                            "classification": "stale",
                            "stale_archive": stale_archive,
                        }
                    )
                    continue
                raise
            assert self._seen_capture_evidence_sha256 is not None
            _require(
                accepted.evidence_sha256 not in self._seen_capture_evidence_sha256,
                "app_capture_evidence_replayed",
            )
            self._seen_capture_evidence_sha256.add(accepted.evidence_sha256)
            object.__setattr__(self, "_last_classified_at", classified_at)
            assert self._consumed_generations is not None
            self._consumed_generations.append(
                {
                    **generation_receipt,
                    "classified_at": classified_at_text,
                    "classification": "accepted",
                    "capture_evidence_sha256": accepted.evidence_sha256,
                }
            )
            self._ordinal += 1
            self._next_generation = 1
            return capture, reader
        _fail("fresh_app_capture_timeout")

    def validate_consumed(self) -> dict[str, Any]:
        _require(
            self._ordinal == self.expected_count + 1,
            "app_capture_inbox_not_consumed",
        )
        self.receipt.validate()
        assert self._consumed_generations is not None
        root = Path(self.receipt.absolute_path)
        expected_children: set[str] = set()
        files: list[dict[str, Any]] = []
        directories: list[str] = []
        for generation in self._consumed_generations:
            bundle_name = str(generation["capture_directory"])
            ready_name = str(generation["ready_marker"])
            expected_children.update((bundle_name, ready_name))
            bundle = canonical_existing_directory(
                root / bundle_name, context="app_capture_final_bundle"
            )
            _require(bundle.parent == root, "app_capture_final_bundle_escape")
            pages_root = canonical_existing_directory(
                bundle / "pages", context="app_capture_final_pages"
            )
            _require(pages_root.parent == bundle, "app_capture_final_pages_escape")
            directories.extend((bundle_name, f"{bundle_name}/pages"))
            capture_value, capture_raw = read_canonical_json_file(
                bundle / "capture.json", context="app_capture_final_json"
            )
            _require(
                capture_value == generation["capture"]
                and capture_raw == canonical_json(generation["capture"])
                and len(capture_raw) == generation["capture_json_bytes"]
                and sha256_bytes(capture_raw) == generation["capture_json_sha256"],
                "app_capture_final_json_drift",
            )
            files.append(
                {
                    "path": f"{bundle_name}/capture.json",
                    "bytes": len(capture_raw),
                    "sha256": sha256_bytes(capture_raw),
                }
            )
            pages = generation["pages"]
            _require(type(pages) is list and bool(pages), "app_capture_final_pages_rejected")
            expected_page_names: set[str] = set()
            for page in pages:
                _require(
                    type(page) is dict
                    and set(page) == {"filename", "bytes", "sha256"}
                    and type(page["filename"]) is str
                    and SAFE_CAPTURE_FILENAME_RE.fullmatch(page["filename"]) is not None
                    and page["filename"] not in expected_page_names,
                    "app_capture_final_page_receipt_rejected",
                )
                expected_page_names.add(page["filename"])
                raw = read_bound_file(
                    pages_root / page["filename"],
                    context="app_capture_final_page",
                    maximum=MAX_CAPTURE_PAGE_BYTES,
                )
                _require(
                    len(raw) == page["bytes"] and sha256_bytes(raw) == page["sha256"],
                    "app_capture_final_page_drift",
                )
                files.append(
                    {
                        "path": f"{bundle_name}/pages/{page['filename']}",
                        "bytes": len(raw),
                        "sha256": sha256_bytes(raw),
                    }
                )
            _require(
                {child.name for child in pages_root.iterdir()} == expected_page_names
                and {child.name for child in bundle.iterdir()} == {"capture.json", "pages"},
                "app_capture_final_bundle_inventory_drift",
            )
            ready_value, ready_raw = read_canonical_json_file(
                root / ready_name, context="app_capture_final_ready"
            )
            expected_ready = {
                "schema_version": 1,
                "kind": CAPTURE_READY_KIND,
                "phase": self.phase,
                "ordinal": generation["ordinal"],
                "generation": generation["generation"],
                "publication_nonce": generation["publication_nonce"],
                "capture_directory": generation["capture_directory"],
                "capture_json_sha256": generation["capture_json_sha256"],
                "pages_inventory_sha256": generation["pages_inventory_sha256"],
            }
            _require(
                ready_value == expected_ready
                and ready_raw == canonical_json(expected_ready)
                and len(ready_raw) == generation["ready_marker_bytes"]
                and sha256_bytes(ready_raw) == generation["ready_marker_sha256"],
                "app_capture_final_ready_drift",
            )
            files.append(
                {
                    "path": ready_name,
                    "bytes": len(ready_raw),
                    "sha256": sha256_bytes(ready_raw),
                }
            )
        _require(
            {child.name for child in root.iterdir()} == expected_children,
            "app_capture_final_inbox_inventory_drift",
        )
        files.sort(key=lambda item: item["path"])
        directories.sort()
        material = {
            "schema_version": 1,
            "kind": "explainiverse-installed-app-inbox-final-inventory",
            "phase": self.phase,
            "accepted_generation_count": self.expected_count,
            "stale_generation_count": len(self._stale_sha256 or ()),
            "generation_count": len(self._consumed_generations),
            "consumed_generations": self._consumed_generations,
            "files": files,
            "directories": directories,
            "file_count": len(files),
            "directory_count": len(directories),
            "owner_private_directory_receipt_sha256": self.receipt.receipt_sha256,
            "accepted_source_generations_retained": True,
            "unobserved_residue_present": False,
        }
        final_inventory = {
            **material,
            "evidence_sha256": sha256_bytes(canonical_json(material)),
        }
        self.receipt.validate()
        object.__setattr__(self, "_final_inventory", final_inventory)
        return strict_json(canonical_json(final_inventory), context="app_capture_final_inventory")

    def close(self) -> None:
        self.receipt.close()

    def to_public_mapping(self) -> dict[str, Any]:
        assert self._seen_capture_evidence_sha256 is not None
        assert self._stale_sha256 is not None
        return {
            "phase": self.phase,
            "expected_capture_count": self.expected_count,
            "accepted_capture_count": len(self._seen_capture_evidence_sha256),
            "stale_generation_count": len(self._stale_sha256),
            "stale_generations_sha256": sha256_bytes(canonical_json(self._stale_sha256)),
            "owner_private_directory_receipt_sha256": self.receipt.receipt_sha256,
            "on_demand_before_each_jit": True,
            "ready_marker_no_replace_required": True,
            "raw_pages_archived_by_driver": self._stale_archive_sink is not None,
        }


def immutable_plan_from_mapping(value: Any, *, expected_sha256: str) -> ImmutablePlan:
    _require(type(value) is dict, "recovery_plan_not_object")
    _require(SHA256_RE.fullmatch(expected_sha256) is not None, "recovery_plan_sha256_rejected")
    try:
        target = value["target"]
        image = target["image"]
        ssh = value["ssh_access"]
        remote = value["remote_runtime"]
        plan = build_immutable_plan(
            head_sha=value["head_sha"],
            lifecycle_nonce=value["lifecycle_nonce"],
            created_at_unix=value["created_at_unix"],
            expires_at_unix=value["expires_at_unix"],
            current_public_ipv4_cidr=value["controller_source"],
            region_description=target["region_description"],
            image_id=image["id"],
            image_created_time=image["created_time"],
            image_description=image["description"],
            image_name=image["name"],
            image_family=image["family"],
            image_version=image["version"],
            image_updated_time=image["updated_time"],
            instance_type_description=target["instance_type_description"],
            gpu_description=target["gpu_description"],
            price_cents_per_hour=target["price_cents_per_hour"],
            vcpus=target["vcpus"],
            memory_gib=target["memory_gib"],
            storage_gib=target["storage_gib"],
            ssh_key_name=ssh["key_name"],
            ssh_public_key_sha256=ssh["public_key_sha256"],
            baseline_file_systems_sha256=value["baseline_file_systems_sha256"],
            original_global_rules=value["original_global_rules"],
            host_key_fingerprint=ssh["ephemeral_host_key_fingerprint"],
            runtime_bundle_sha256=remote["bundle_sha256"],
        )
    except (KeyError, TypeError):
        _fail("recovery_plan_schema_rejected")
    _require(plan.to_mapping() == value, "recovery_plan_mapping_drift")
    _require(plan.sha256 == expected_sha256, "recovery_plan_digest_mismatch")
    return plan


def read_recovery_plan(
    evidence_directory: EvidenceDirectoryReceipt,
    *,
    expected_plan_sha256: str,
) -> ImmutablePlan:
    evidence_directory.validate()
    root = Path(evidence_directory.absolute_path)
    candidates = sorted(root.glob("002-immutable-plan.json"))
    _require(len(candidates) == 1, "recovery_immutable_plan_entry_missing")
    envelope, raw = read_canonical_json_file(candidates[0], context="recovery_immutable_plan_entry")
    _require(
        set(envelope)
        == {
            "schema_version",
            "kind",
            "sequence",
            "label",
            "control_plane_plan_sha256",
            "evidence_directory_acl_receipt_sha256",
            "previous_evidence_sha256",
            "payload",
        }
        and envelope["schema_version"] == 1
        and envelope["kind"] == "explainiverse-lambda-live-driver-evidence"
        and envelope["sequence"] == 2
        and envelope["label"] == "immutable-plan"
        and envelope["control_plane_plan_sha256"] == expected_plan_sha256
        and envelope["evidence_directory_acl_receipt_sha256"] == evidence_directory.receipt_sha256
        and type(envelope["previous_evidence_sha256"]) is str
        and SHA256_RE.fullmatch(envelope["previous_evidence_sha256"]) is not None
        and type(envelope["payload"]) is dict
        and sha256_bytes(raw) != envelope["previous_evidence_sha256"],
        "recovery_immutable_plan_envelope_rejected",
    )
    return immutable_plan_from_mapping(envelope["payload"], expected_sha256=expected_plan_sha256)
