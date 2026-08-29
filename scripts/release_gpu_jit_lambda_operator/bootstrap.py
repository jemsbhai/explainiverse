"""Stdlib-only verifier for the pinned interpreter and dependency roots.

The absolute preloader executes this code under base CPython ``-I -S -B`` only
after proving the repository is the clean expected commit. No project or
third-party import is permitted until :func:`verify_and_enable` returns.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import stat
import sys
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence, cast

MANIFEST_RELATIVE = "scripts/release_gpu_jit_lambda_operator/site-packages-windows-cp313.json"
PYTHON_MANIFEST_RELATIVE = (
    "scripts/release_gpu_jit_lambda_operator/python-runtime-windows-cp313.json"
)
RUNTIME_LOCK_RELATIVE = "scripts/release_gpu_jit_lambda_operator/requirements-windows-cp313.txt"
BOOTSTRAP_LOCK_RELATIVE = (
    "scripts/release_gpu_jit_lambda_operator/requirements-windows-cp313-bootstrap.txt"
)
PYTHON_MANIFEST_SHA256 = "e2d965a1f8b09d1e5f0349133dfd869eceb92cf730f54a456a4f79bb22d5a519"
SITE_MANIFEST_SHA256 = "5a6282da0fd87317986b97da1725480c0877686f0e559a83520acf95f46d945f"
PYTHON_ARCHIVE_SHA256 = "d1f04d990aee1253d8569e8e5104e30fa9f5fa830899f14843448872d936a2cf"
PYTHON_VERSION = (3, 13, 15)
EXPECTED_ARCHIVES = {
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
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_RUNTIME_FILE_BYTES = 512 * 1024 * 1024


class BootstrapError(RuntimeError):
    """Stable, secret-free pre-import rejection."""


def _fail(code: str) -> NoReturn:
    raise BootstrapError(code)


def _require(condition: bool, code: str) -> None:
    if not condition:
        _fail(code)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        ).encode("ascii")
    except (TypeError, ValueError):
        _fail("bootstrap_value_not_canonical_json")


def _pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, "bootstrap_json_duplicate_key")
        result[key] = value
    return result


def _is_reparse(path: Path) -> bool:
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def bound_file(path: Path, *, context: str, maximum: int) -> bytes:
    """Read a canonical single-link file through one held descriptor."""

    _require(path.is_absolute(), f"{context}_not_absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        _fail(f"{context}_unavailable")
    _require(path == resolved, f"{context}_not_canonical")
    _require(
        path.is_file() and not path.is_symlink() and not _is_reparse(path),
        f"{context}_file_rejected",
    )
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        _fail(f"{context}_open_failed")
    try:
        before = os.fstat(descriptor)
        current = os.lstat(path)
        _require(
            stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and current.st_nlink == 1,
            f"{context}_identity_rejected",
        )
        _require(
            (before.st_dev, before.st_ino) == (current.st_dev, current.st_ino),
            f"{context}_identity_drift",
        )
        _require(0 <= before.st_size <= maximum, f"{context}_size_rejected")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
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
    finally:
        os.close(descriptor)


def canonical_directory(path: Path, *, context: str) -> Path:
    _require(path.is_absolute(), f"{context}_not_absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        _fail(f"{context}_unavailable")
    _require(
        path == resolved and path.is_dir() and not path.is_symlink() and not _is_reparse(path),
        f"{context}_directory_rejected",
    )
    return path


def _load_manifest(repository_root: Path) -> tuple[dict[str, Any], bytes]:
    raw = bound_file(
        repository_root / MANIFEST_RELATIVE,
        context="bootstrap_manifest",
        maximum=MAX_MANIFEST_BYTES,
    )
    _require(_sha256(raw) == SITE_MANIFEST_SHA256, "bootstrap_manifest_digest_rejected")
    try:
        value = json.loads(raw, object_pairs_hook=_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail("bootstrap_manifest_json_rejected")
    required = {
        "schema_version",
        "kind",
        "target",
        "archives",
        "archive_set_sha256",
        "requirements",
        "files",
        "directories",
        "bytecode_allowed",
        "untracked_files_or_directories_allowed",
    }
    _require(
        type(value) is dict
        and raw == _canonical(value)
        and set(value) == required
        and value.get("schema_version") == 1
        and value.get("kind") == "explainiverse-operator-windows-cp313-site-manifest"
        and value.get("bytecode_allowed") is False
        and value.get("untracked_files_or_directories_allowed") is False,
        "bootstrap_manifest_schema_rejected",
    )
    archives = value.get("archives")
    _require(type(archives) is list, "bootstrap_manifest_archives_rejected")
    observed: dict[str, tuple[str, str, str]] = {}
    for item in archives:
        _require(
            type(item) is dict and set(item) == {"distribution", "filename", "sha256", "version"},
            "bootstrap_manifest_archive_rejected",
        )
        observed[item["filename"]] = (
            item["distribution"],
            item["version"],
            item["sha256"],
        )
    _require(observed == EXPECTED_ARCHIVES, "bootstrap_manifest_archive_set_rejected")
    _require(
        value.get("archive_set_sha256") == _sha256(_canonical(archives)),
        "bootstrap_manifest_archive_binding_rejected",
    )
    requirements = value.get("requirements")
    _require(type(requirements) is dict, "bootstrap_manifest_requirements_rejected")
    runtime_lock = bound_file(
        repository_root / RUNTIME_LOCK_RELATIVE,
        context="bootstrap_runtime_lock",
        maximum=131072,
    )
    bootstrap_lock = bound_file(
        repository_root / BOOTSTRAP_LOCK_RELATIVE,
        context="bootstrap_pip_lock",
        maximum=131072,
    )
    _require(
        requirements
        == {
            "runtime_sha256": _sha256(runtime_lock),
            "bootstrap_sha256": _sha256(bootstrap_lock),
        },
        "bootstrap_manifest_lock_binding_rejected",
    )
    return value, raw


def _load_python_manifest(repository_root: Path) -> tuple[dict[str, Any], bytes]:
    raw = bound_file(
        repository_root / PYTHON_MANIFEST_RELATIVE,
        context="bootstrap_python_manifest",
        maximum=MAX_MANIFEST_BYTES,
    )
    _require(_sha256(raw) == PYTHON_MANIFEST_SHA256, "bootstrap_python_manifest_digest_rejected")
    try:
        value = json.loads(raw, object_pairs_hook=_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail("bootstrap_python_manifest_json_rejected")
    required = {
        "schema_version",
        "kind",
        "archive",
        "target",
        "files",
        "directories",
        "startup",
        "untracked_files_or_directories_allowed",
    }
    _require(
        type(value) is dict
        and raw == _canonical(value)
        and set(value) == required
        and value.get("schema_version") == 1
        and value.get("kind") == "explainiverse-operator-python-3.13.15-embed-amd64-manifest"
        and value.get("archive")
        == {
            "bytes": 11_009_825,
            "filename": "python-3.13.15-embed-amd64.zip",
            "sha256": PYTHON_ARCHIVE_SHA256,
            "source_url": (
                "https://www.python.org/ftp/python/3.13.15/" "python-3.13.15-embed-amd64.zip"
            ),
        }
        and value.get("target")
        == {
            "implementation": "CPython",
            "platform": "win_amd64",
            "python_version": "3.13.15",
        }
        and value.get("directories") == []
        and value.get("untracked_files_or_directories_allowed") is False,
        "bootstrap_python_manifest_schema_rejected",
    )
    files = value.get("files")
    _require(type(files) is dict and len(files) == 34, "bootstrap_python_manifest_files_rejected")
    startup = value.get("startup")
    _require(
        startup
        == {
            "pth_filename": "python313._pth",
            "pth_sha256": "35ddf94682ff9aa713a8d63557242ad00f3f28fdd39337f02c3bda4c0f791577",
            "site_import_enabled": False,
        },
        "bootstrap_python_startup_rejected",
    )
    return value, raw


def verify_site_tree(site_root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Compare every file and directory without importing or executing it."""

    site_root = canonical_directory(site_root, context="bootstrap_site_root")
    expected_files = manifest.get("files")
    expected_directories = manifest.get("directories")
    _require(
        type(expected_files) is dict and type(expected_directories) is list,
        "bootstrap_manifest_tree_schema_rejected",
    )
    expected_files = cast(dict[str, Any], expected_files)
    expected_directories = cast(list[Any], expected_directories)
    observed_files: dict[str, dict[str, Any]] = {}
    observed_directories: set[str] = set()
    for path in sorted(
        site_root.rglob("*"), key=lambda item: item.relative_to(site_root).as_posix()
    ):
        relative = path.relative_to(site_root).as_posix()
        _require(not path.is_symlink() and not _is_reparse(path), "bootstrap_site_reparse_rejected")
        if path.is_dir():
            _require(path.name != "__pycache__", "bootstrap_bytecode_directory_rejected")
            observed_directories.add(relative)
            continue
        _require(path.is_file(), "bootstrap_site_entry_rejected")
        _require(path.suffix.lower() not in {".pyc", ".pyo"}, "bootstrap_bytecode_file_rejected")
        raw = bound_file(path, context="bootstrap_site_file", maximum=MAX_RUNTIME_FILE_BYTES)
        observed_files[relative] = {"bytes": len(raw), "sha256": _sha256(raw)}
    expected_bytes = {
        name: {"bytes": item["bytes"], "sha256": item["sha256"]}
        for name, item in expected_files.items()
    }
    _require(observed_files == expected_bytes, "bootstrap_site_file_set_or_bytes_rejected")
    _require(
        observed_directories == set(expected_directories),
        "bootstrap_site_directory_set_rejected",
    )
    rows = [
        f"{name}\t{item['bytes']}\t{item['sha256']}\n".encode("utf-8")
        for name, item in sorted(observed_files.items())
    ]
    return {
        "site_root": str(site_root),
        "file_count": len(observed_files),
        "directory_count": len(observed_directories),
        "file_inventory_sha256": _sha256(b"".join(rows)),
        "untracked_files_or_directories_present": False,
        "bytecode_present": False,
        "all_importable_bytes_match_verified_wheels": True,
    }


def verify_python_tree(python_root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Compare the complete embeddable runtime to the official archive."""

    python_root = canonical_directory(python_root, context="bootstrap_python_root")
    expected_files = manifest.get("files")
    _require(type(expected_files) is dict, "bootstrap_python_tree_schema_rejected")
    expected_files = cast(dict[str, Any], expected_files)
    observed_files: dict[str, dict[str, Any]] = {}
    observed_directories: set[str] = set()
    for path in sorted(
        python_root.rglob("*"), key=lambda item: item.relative_to(python_root).as_posix()
    ):
        relative = path.relative_to(python_root).as_posix()
        _require(
            not path.is_symlink() and not _is_reparse(path),
            "bootstrap_python_reparse_rejected",
        )
        if path.is_dir():
            observed_directories.add(relative)
            continue
        _require(path.is_file(), "bootstrap_python_entry_rejected")
        raw = bound_file(path, context="bootstrap_python_file", maximum=MAX_RUNTIME_FILE_BYTES)
        observed_files[relative] = {"bytes": len(raw), "sha256": _sha256(raw)}
    expected_bytes = {
        name: {"bytes": item["bytes"], "sha256": item["sha256"]}
        for name, item in expected_files.items()
    }
    _require(observed_files == expected_bytes, "bootstrap_python_file_set_or_bytes_rejected")
    _require(not observed_directories, "bootstrap_python_directory_set_rejected")
    rows = [
        f"{name}\t{item['bytes']}\t{item['sha256']}\n".encode("utf-8")
        for name, item in sorted(observed_files.items())
    ]
    return {
        "python_root": str(python_root),
        "file_count": len(observed_files),
        "directory_count": 0,
        "file_inventory_sha256": _sha256(b"".join(rows)),
        "official_archive_sha256": PYTHON_ARCHIVE_SHA256,
        "untracked_files_or_directories_present": False,
        "all_runtime_bytes_match_official_archive": True,
    }


def verify_and_enable(
    repository_root: Path,
    python_root: Path,
    site_root: Path,
    *,
    python_install_receipt: Mapping[str, Any],
    site_install_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify isolated base Python and the runtime tree, then enable four roots."""

    _require(
        os.name == "nt"
        and platform.python_implementation() == "CPython"
        and sys.version_info[:3] == PYTHON_VERSION
        and platform.machine().lower() in {"amd64", "x86_64"},
        "bootstrap_interpreter_target_rejected",
    )
    _require(
        sys.flags.isolated == 1
        and sys.flags.ignore_environment == 1
        and sys.flags.no_user_site == 1
        and sys.flags.safe_path
        and sys.flags.no_site == 1
        and sys.dont_write_bytecode,
        "bootstrap_requires_python_I_S_B",
    )
    repository_root = canonical_directory(repository_root, context="bootstrap_repository_root")
    python_root = canonical_directory(python_root, context="bootstrap_python_root")
    site_root = canonical_directory(site_root, context="bootstrap_site_root")
    try:
        working = Path.cwd().resolve(strict=True)
        base_root = Path(sys.executable).resolve(strict=True).parent
    except OSError:
        _fail("bootstrap_import_root_unavailable")
    _require(
        working != repository_root
        and repository_root not in working.parents
        and working != python_root
        and python_root not in working.parents
        and working != site_root
        and site_root not in working.parents,
        "bootstrap_working_directory_mismatch",
    )
    _require(base_root == python_root, "bootstrap_python_root_identity_rejected")
    preactivation_paths: list[str] = []
    for raw_path in sys.path:
        candidate = Path(raw_path or os.curdir)
        _require(candidate.is_absolute(), "bootstrap_sys_path_relative")
        resolved = candidate.resolve(strict=False)
        _require(
            resolved == base_root or base_root in resolved.parents,
            "bootstrap_sys_path_outside_base_stdlib",
        )
        _require(
            resolved != repository_root
            and repository_root not in resolved.parents
            and resolved != site_root
            and site_root not in resolved.parents,
            "bootstrap_untrusted_root_present_before_verification",
        )
        preactivation_paths.append(str(resolved))
    _require(
        len(preactivation_paths) == len(set(preactivation_paths)), "bootstrap_sys_path_duplicate"
    )
    python_manifest, python_manifest_raw = _load_python_manifest(repository_root)
    python_tree = verify_python_tree(python_root, python_manifest)
    manifest, manifest_raw = _load_manifest(repository_root)
    tree = verify_site_tree(site_root, manifest)
    _require(
        python_install_receipt.get("python_runtime_root") == str(python_root)
        and python_install_receipt.get("archive_sha256") == PYTHON_ARCHIVE_SHA256
        and python_install_receipt.get("manifest_sha256") == _sha256(python_manifest_raw)
        and python_install_receipt.get("file_count") == python_tree["file_count"]
        and python_install_receipt.get("directory_count") == python_tree["directory_count"]
        and python_install_receipt.get("file_inventory_sha256")
        == python_tree["file_inventory_sha256"],
        "bootstrap_python_install_receipt_rejected",
    )
    _require(
        site_install_receipt.get("runtime_root") == str(site_root)
        and site_install_receipt.get("manifest_sha256") == _sha256(manifest_raw)
        and site_install_receipt.get("file_count") == tree["file_count"]
        and site_install_receipt.get("directory_count") == tree["directory_count"]
        and site_install_receipt.get("file_inventory_sha256") == tree["file_inventory_sha256"],
        "bootstrap_site_install_receipt_rejected",
    )
    activation_paths = (
        site_root,
        site_root / "win32",
        site_root / "win32" / "lib",
        site_root / "pythonwin",
    )
    for path in activation_paths:
        canonical_directory(path, context="bootstrap_activation_root")
        _require(str(path) not in sys.path, "bootstrap_activation_root_already_present")
        sys.path.append(str(path))
    importlib.import_module("pywin32_bootstrap")
    return {
        "schema_version": 1,
        "kind": "explainiverse-operator-pre-site-bootstrap",
        "python_manifest_sha256": _sha256(python_manifest_raw),
        "python_archive_sha256": PYTHON_ARCHIVE_SHA256,
        "python_tree": python_tree,
        "manifest_sha256": _sha256(manifest_raw),
        "archive_set_sha256": manifest["archive_set_sha256"],
        "runtime_requirements_sha256": manifest["requirements"]["runtime_sha256"],
        "bootstrap_requirements_sha256": manifest["requirements"]["bootstrap_sha256"],
        "base_python_executable": str(Path(sys.executable).resolve(strict=True)),
        "base_python_executable_sha256": _sha256(
            bound_file(
                Path(sys.executable).resolve(strict=True),
                context="bootstrap_python_executable",
                maximum=16 * 1024 * 1024,
            )
        ),
        "preactivation": {
            "working_directory": str(working),
            "sys_path_sha256": _sha256("\n".join(preactivation_paths).encode("utf-8")),
            "only_base_stdlib_roots": True,
        },
        "site_tree": tree,
        "activation_paths": [str(path) for path in activation_paths],
        "site_processing_disabled": True,
        "pth_executed_by_cpython": False,
        "verified_pywin32_bootstrap_imported_after_verification": True,
    }


def revalidate_enabled_environment(
    repository_root: Path, python_root: Path, site_root: Path
) -> dict[str, Any]:
    repository_root = canonical_directory(repository_root, context="bootstrap_repository_root")
    python_root = canonical_directory(python_root, context="bootstrap_python_root")
    site_root = canonical_directory(site_root, context="bootstrap_site_root")
    _require(
        Path(sys.executable).resolve(strict=True).parent == python_root,
        "bootstrap_python_root_identity_rejected",
    )
    python_manifest, python_manifest_raw = _load_python_manifest(repository_root)
    python_tree = verify_python_tree(python_root, python_manifest)
    manifest, manifest_raw = _load_manifest(repository_root)
    tree = verify_site_tree(site_root, manifest)
    activation_paths = (
        site_root,
        site_root / "win32",
        site_root / "win32" / "lib",
        site_root / "pythonwin",
    )
    normalized = [str(Path(value).resolve(strict=False)) for value in sys.path]
    _require(
        all(normalized.count(str(path)) == 1 for path in activation_paths),
        "bootstrap_activation_root_cardinality",
    )
    return {
        "schema_version": 1,
        "kind": "explainiverse-operator-enabled-environment-revalidation",
        "python_manifest_sha256": _sha256(python_manifest_raw),
        "python_archive_sha256": PYTHON_ARCHIVE_SHA256,
        "python_tree": python_tree,
        "manifest_sha256": _sha256(manifest_raw),
        "archive_set_sha256": manifest["archive_set_sha256"],
        "runtime_requirements_sha256": manifest["requirements"]["runtime_sha256"],
        "bootstrap_requirements_sha256": manifest["requirements"]["bootstrap_sha256"],
        "site_tree": tree,
        "activation_paths": [str(path) for path in activation_paths],
        "site_processing_disabled": sys.flags.no_site == 1,
        "pth_executed_by_cpython": False,
    }
