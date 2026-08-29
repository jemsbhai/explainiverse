"""Pure validation for archived production-operator preflight evidence.

This module is deliberately stdlib-only and performs no filesystem, process,
environment, clock, network, controller, provider, or Git access.  Both the
operator producer and historical evidence loaders use the same exact schema
and cross-binding rules.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Mapping, NoReturn

SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
GIT_SHA_RE = re.compile(r"[0-9a-f]{40}\Z")
NONCE_RE = re.compile(r"[0-9a-f]{32}\Z")
SID_RE = re.compile(r"S-1-[0-9-]+\Z")
SSH_FINGERPRINT_RE = re.compile(r"SHA256:[A-Za-z0-9+/]{43}\Z")
REPOSITORY = "jemsbhai/explainiverse"
ORIGIN_URL = "https://github.com/jemsbhai/explainiverse.git"
PUBLICATION_TAG = "v0.15.0"
TARGET_REGION = "us-midwest-1"
TARGET_REGION_DESCRIPTION = "Illinois, USA"
PHASE_REFS = {
    "pull-request": "refs/heads/codex/harden-cuda-runner-routing",
    "final-main": "refs/heads/main",
    "publication": "refs/tags/v0.15.0",
}
PHASE_CAPTURE_COUNTS = {"pull-request": 2, "final-main": 4, "publication": 2}
PRELOADER_SHIM_SHA256 = "22bb14f6e5fed4e7c5456f62e11569c9e5a0846ad1428854e545b2e4c1c979aa"
PYTHON_ARCHIVE_SHA256 = "d1f04d990aee1253d8569e8e5104e30fa9f5fa830899f14843448872d936a2cf"
PYTHON_MANIFEST_SHA256 = "e2d965a1f8b09d1e5f0349133dfd869eceb92cf730f54a456a4f79bb22d5a519"
PYTHON_FILE_INVENTORY_SHA256 = "ea028b8d42b0231c116581c4184297900bd4c0152a54017127b822f10b9742d9"
PYTHON_EXECUTABLE_SHA256 = "85b71d8c6ec1905935f74be0c9869aae198d00e98f39df699ec66f9c5a84cecd"
PYTHON_ZIP_SHA256 = "1916abd946d2044ec8c04c3319f96c8415d5b6fce01e125622827f2b7756cbab"
SITE_MANIFEST_SHA256 = "5a6282da0fd87317986b97da1725480c0877686f0e559a83520acf95f46d945f"
SITE_FILE_INVENTORY_SHA256 = "2cf1cf52ad8d284fcc2e7790acaaa32f3e77a9f39fa717f8bc2a67bc83ba31fe"
RUNTIME_REQUIREMENTS_BYTES = 541
RUNTIME_REQUIREMENTS_SHA256 = "123845e69b0bcd47e4477ade6f8a1c8d75db6873d61361f84bac4f49e9564eda"
BOOTSTRAP_REQUIREMENTS_SHA256 = "739975124f37e4e5245c750a9413d1269583964aa10eb6a6fbafa9bce9ebc08d"
SITE_MANIFEST_BYTES = 138_807
ALLOWED_PTH_FILES = {
    "pywin32.pth": {
        "bytes": 185,
        "sha256": "d902584a2a0a5216ce12c712d1378fe07541d32c383d0cc5abcd68412144fe4d",
    }
}
SITE_ARCHIVE_SET_SHA256 = "4e3c35ff45fc3c5897e5b7f1c8e80e7b027e7571701a37d3b9df5cf9c6bc515c"
RUNTIME_BUNDLE_NAMES = ("__init__.py", "bootstrap.py", "executor.py", "runtime_contract.py")
REMOTE_RUNTIME_ROOT = "/opt/explainiverse/bin/release_gpu_jit_lambda_runtime"
FIXED_REMOTE_COMMAND = (
    "/usr/bin/sudo",
    "-n",
    "--",
    "/usr/bin/python3",
    "-B",
    f"{REMOTE_RUNTIME_ROOT}/bootstrap.py",
)
FIXED_CLOUD_INIT_WAIT_COMMAND = (
    "/usr/bin/sudo",
    "-n",
    "--",
    "/usr/bin/cloud-init",
    "status",
    "--wait",
)
FIXED_PREFLIGHT_COMMAND = (
    "/usr/bin/sudo",
    "-n",
    "--",
    "/usr/bin/python3",
    "-B",
    f"{REMOTE_RUNTIME_ROOT}/executor.py",
    "probe-host",
)
MUTATION_ORDER = (
    "restrict_global",
    "create_ruleset",
    "launch",
    "terminate",
    "delete_ruleset",
    "restore_global",
)
SOURCE_MANIFEST_RELATIVE = "scripts/release_gpu_jit_lambda_operator/source-worktree-manifest.json"
PRELOADER_RELATIVE = "scripts/release_gpu_jit_lambda_operator/preloader.py"
SHIM_RELATIVE = "scripts/release_gpu_jit_lambda_operator/preloader_shim.py"
RECEIPT_CONTRACT_RELATIVE = "scripts/release_gpu_jit_lambda_operator/receipt_contract.py"
SITE_MANIFEST_RELATIVE = "scripts/release_gpu_jit_lambda_operator/site-packages-windows-cp313.json"
PYTHON_MANIFEST_RELATIVE = (
    "scripts/release_gpu_jit_lambda_operator/python-runtime-windows-cp313.json"
)
RUNTIME_LOCK_RELATIVE = "scripts/release_gpu_jit_lambda_operator/requirements-windows-cp313.txt"
BOOTSTRAP_LOCK_RELATIVE = (
    "scripts/release_gpu_jit_lambda_operator/requirements-windows-cp313-bootstrap.txt"
)

CRITICAL_SOURCE_PATHS = frozenset(
    {
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
        PRELOADER_RELATIVE,
        SHIM_RELATIVE,
        RECEIPT_CONTRACT_RELATIVE,
        BOOTSTRAP_LOCK_RELATIVE,
        RUNTIME_LOCK_RELATIVE,
        SITE_MANIFEST_RELATIVE,
        PYTHON_MANIFEST_RELATIVE,
        SOURCE_MANIFEST_RELATIVE,
        "scripts/release_gpu_jit_lambda_operator/windows_launcher.py",
        "scripts/release_gpu_jit_lambda_runtime/README.md",
        "scripts/release_gpu_jit_lambda_runtime/__init__.py",
        "scripts/release_gpu_jit_lambda_runtime/bootstrap.py",
        "scripts/release_gpu_jit_lambda_runtime/executor.py",
        "scripts/release_gpu_jit_lambda_runtime/runtime_contract.py",
    }
)

PINNED_EXECUTABLES: Mapping[str, Mapping[str, str]] = {
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
        "owner_sid": "S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464",
        "authenticode_subject": (
            "CN=Microsoft Windows, O=Microsoft Corporation, " "L=Redmond, S=Washington, C=US"
        ),
        "authenticode_thumbprint": "BAC13DF18B37E808208A39D3A54CCE975FAC8C1D",
    },
}

LOCKED_DISTRIBUTIONS: Mapping[str, Mapping[str, Any]] = {
    "cffi": {
        "version": "2.1.1",
        "archive_filename": "cffi-2.1.1-cp313-cp313-win_amd64.whl",
        "archive_sha256": "1aa5645c30469b09530c4ebca77ebf8f17618293c58f8549cb1a543a50236e7d",
        "file_count": 30,
        "total_bytes": 564_653,
        "inventory_sha256": "d608d7e2097bd6b3af5e3f9f37e9ac8362072b1e900dc6e529e8350ed73a9951",
    },
    "cryptography": {
        "version": "50.0.0",
        "archive_filename": "cryptography-50.0.0-cp311-abi3-win_amd64.whl",
        "archive_sha256": "bd1c592e4d5974f0d08d4888e432157adba757c66da0246918e43677fafa2d30",
        "file_count": 119,
        "total_bytes": 10_488_299,
        "inventory_sha256": "5b474b27a215c9c2d639b9c5139a92ab6684bc2fca106393dbb5a90c12fe5ae9",
    },
    "pycparser": {
        "version": "3.0",
        "archive_filename": "pycparser-3.0-py3-none-any.whl",
        "archive_sha256": "b727414169a36b7d524c1c3e31839a521725078d7b2ff038656844266160a992",
        "file_count": 12,
        "total_bytes": 202_717,
        "inventory_sha256": "6b59745856ab1576f328afda83b81349250acc34b9c5caa472bcb903e64ca95d",
    },
    "pywin32": {
        "version": "311",
        "archive_filename": "pywin32-311-cp313-cp313-win_amd64.whl",
        "archive_sha256": "718a38f7e5b058e76aee1c56ddd06908116d35147e133427e59a3983f703a20d",
        "file_count": 595,
        "total_bytes": 20_327_774,
        "inventory_sha256": "90a677a2540e20b99e1fc7afda16154c9a5fe50d6ebc5da7eb8d5081d1070118",
    },
}


class OperatorReceiptContractError(ValueError):
    """Stable, secret-free pure receipt rejection."""


def _fail(code: str) -> NoReturn:
    raise OperatorReceiptContractError(code)


def _require(condition: bool, code: str) -> None:
    if not condition:
        _fail(code)


def _canonical(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        ).encode("ascii")
    except (TypeError, ValueError):
        _fail("operator_receipt_non_json_value")


def _live_canonical(value: Any) -> bytes:
    """Mirror the Lambda live adapter's newline-free canonical JSON domain."""

    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "ascii"
        )
    except (TypeError, ValueError):
        _fail("operator_receipt_non_json_value")


def _normalized(value: Any, *, context: str) -> Any:
    try:
        return json.loads(_canonical(value))
    except (UnicodeError, json.JSONDecodeError):
        _fail(f"{context}_normalization_rejected")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git_blob(raw: bytes) -> str:
    return hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()


def _exact(value: Any, keys: set[str] | frozenset[str], context: str) -> dict[str, Any]:
    _require(type(value) is dict and set(value) == set(keys), f"{context}_schema_rejected")
    return value


def _digest(value: Any, context: str) -> str:
    _require(type(value) is str and SHA256_RE.fullmatch(value) is not None, context)
    return value


def _git_sha(value: Any, context: str) -> str:
    _require(type(value) is str and GIT_SHA_RE.fullmatch(value) is not None, context)
    return value


def _positive_int(value: Any, context: str, *, allow_zero: bool = False) -> int:
    _require(
        type(value) is int and (value >= 0 if allow_zero else value > 0),
        context,
    )
    return value


def _text(value: Any, context: str, *, allow_empty: bool = False) -> str:
    _require(type(value) is str and (allow_empty or bool(value)), context)
    return value


def _windows_absolute(value: Any, context: str) -> str:
    text = _text(value, context)
    path = PureWindowsPath(text)
    _require(path.is_absolute() and ".." not in path.parts, context)
    return text


def _posix_relative(value: Any, context: str) -> str:
    text = _text(value, context)
    path = PurePosixPath(text)
    _require(
        not path.is_absolute()
        and path.as_posix() == text
        and all(part not in {"", ".", ".."} for part in path.parts),
        context,
    )
    return text


def _time(value: Any, context: str) -> datetime:
    text = _text(value, context)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        _fail(context)
    _require(
        parsed.tzinfo is not None
        and parsed.utcoffset() == timedelta(0)
        and text == parsed.isoformat(),
        context,
    )
    return parsed


def _live_time(value: Any, context: str) -> datetime:
    """Mirror the provider timestamp parser while retaining the exact source text."""

    text = _text(value, context)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        _fail(context)
    _require(parsed.tzinfo is not None, context)
    return parsed.astimezone(timezone.utc)


def _git_tree_object_sha(files: Mapping[str, Mapping[str, Any]]) -> str:
    tree: dict[str, Any] = {}
    for relative, item in files.items():
        cursor = tree
        parts = PurePosixPath(relative).parts
        for part in parts[:-1]:
            child = cursor.setdefault(part, {})
            _require(type(child) is dict, "source_manifest_tree_collision")
            cursor = child
        _require(parts[-1] not in cursor, "source_manifest_tree_collision")
        cursor[parts[-1]] = (item["mode"], item["git_blob_sha"])

    def digest(node: Mapping[str, Any]) -> str:
        entries: list[tuple[bytes, bytes]] = []
        for name, value in node.items():
            encoded = name.encode("utf-8")
            if type(value) is dict:
                object_sha = digest(value)
                sort_key = encoded + b"/"
                row = b"40000 " + encoded + b"\0" + bytes.fromhex(object_sha)
            else:
                mode, object_sha = value
                sort_key = encoded
                row = mode.encode("ascii") + b" " + encoded + b"\0" + bytes.fromhex(object_sha)
            entries.append((sort_key, row))
        payload = b"".join(row for _, row in sorted(entries, key=lambda item: item[0]))
        framed = f"tree {len(payload)}\0".encode("ascii") + payload
        return hashlib.sha1(framed).hexdigest()

    return digest(tree)


def _validate_source_manifest(
    value: Any,
    *,
    expected_sha256: str,
    context: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = _exact(
        value,
        {
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
        },
        context,
    )
    _require(
        manifest["schema_version"] == 1
        and manifest["kind"] == "explainiverse-operator-source-worktree-manifest"
        and manifest["excluded_paths"] == [SOURCE_MANIFEST_RELATIVE, PRELOADER_RELATIVE]
        and manifest["source"] == "exact-staged-index-blobs"
        and manifest["runtime_git_dependency"] is False
        and _sha(_canonical(manifest)) == expected_sha256,
        f"{context}_binding_rejected",
    )
    raw_files = manifest["files"]
    raw_directories = manifest["directories"]
    _require(type(raw_files) is dict and bool(raw_files), f"{context}_files_rejected")
    _require(
        type(raw_directories) is list
        and all(type(item) is str for item in raw_directories)
        and raw_directories == sorted(set(raw_directories)),
        f"{context}_directories_rejected",
    )
    files: dict[str, dict[str, Any]] = {}
    expected_directories: set[str] = set()
    rows: list[bytes] = []
    for raw_name, raw_item in sorted(raw_files.items()):
        name = _posix_relative(raw_name, f"{context}_path_rejected")
        _require(
            name not in {SOURCE_MANIFEST_RELATIVE, PRELOADER_RELATIVE},
            f"{context}_excluded_file_present",
        )
        item = _exact(raw_item, {"mode", "bytes", "sha256", "git_blob_sha"}, context)
        _require(item["mode"] in {"100644", "100755"}, f"{context}_mode_rejected")
        size = _positive_int(item["bytes"], f"{context}_size_rejected", allow_zero=True)
        digest = _digest(item["sha256"], f"{context}_digest_rejected")
        blob = _git_sha(item["git_blob_sha"], f"{context}_blob_rejected")
        files[name] = {
            "mode": item["mode"],
            "bytes": size,
            "sha256": digest,
            "git_blob_sha": blob,
        }
        parent = PurePosixPath(name).parent
        while parent != PurePosixPath("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
        rows.append(f"{name}\t{item['mode']}\t{size}\t{digest}\t{blob}\n".encode("utf-8"))
    _require(
        manifest["file_count"] == len(files)
        and type(manifest["file_count"]) is int
        and manifest["directory_count"] == len(expected_directories)
        and type(manifest["directory_count"]) is int
        and raw_directories == sorted(expected_directories)
        and manifest["file_inventory_sha256"] == _sha(b"".join(rows)),
        f"{context}_inventory_rejected",
    )
    return manifest, files


def _critical_row(value: Any, context: str) -> dict[str, Any]:
    row = _exact(value, {"bytes", "sha256", "git_blob_sha"}, context)
    _positive_int(row["bytes"], f"{context}_bytes", allow_zero=True)
    _digest(row["sha256"], f"{context}_sha256")
    _git_sha(row["git_blob_sha"], f"{context}_git_blob")
    return row


def _validate_repository(
    value: Any,
    *,
    expected_phase: str,
    expected_head_sha: str,
    expected_ref: str,
    source: Mapping[str, Any],
    shim: Mapping[str, Any],
    context: str,
) -> dict[str, Any]:
    repository = _exact(
        value,
        {
            "repository",
            "absolute_root",
            "origin_url",
            "head_sha",
            "tree_object_sha",
            "tree_inventory_sha256",
            "clean_tracked_and_untracked",
            "supplied_ref",
            "remote_object_type",
            "remote_object_sha",
            "remote_target_sha",
            "remote_ref_response_sha256",
            "annotated_tag_response_sha256",
            "critical_sources",
            "git_configuration",
        },
        context,
    )
    root = _windows_absolute(repository["absolute_root"], f"{context}_root_rejected")
    _require(
        repository["repository"] == REPOSITORY
        and repository["origin_url"] == ORIGIN_URL
        and repository["head_sha"] == expected_head_sha
        and repository["clean_tracked_and_untracked"] is True
        and repository["supplied_ref"] == expected_ref
        and source["repository_root"] == root
        and source["origin_url"] == ORIGIN_URL
        and source["head_sha"] == expected_head_sha,
        f"{context}_identity_rejected",
    )
    _git_sha(repository["tree_object_sha"], f"{context}_tree_object_rejected")
    _digest(repository["tree_inventory_sha256"], f"{context}_tree_inventory_rejected")
    _digest(repository["remote_ref_response_sha256"], f"{context}_remote_response_rejected")
    if expected_phase == "publication":
        _require(
            repository["remote_object_type"] == "tag"
            and repository["remote_target_sha"] == expected_head_sha
            and repository["remote_object_sha"] != expected_head_sha
            and type(repository["annotated_tag_response_sha256"]) is str,
            f"{context}_publication_ref_rejected",
        )
        _git_sha(repository["remote_object_sha"], f"{context}_tag_object_rejected")
        _digest(
            repository["annotated_tag_response_sha256"],
            f"{context}_tag_response_rejected",
        )
    else:
        _require(
            repository["remote_object_type"] == "commit"
            and repository["remote_object_sha"] == expected_head_sha
            and repository["remote_target_sha"] == expected_head_sha
            and repository["annotated_tag_response_sha256"] is None,
            f"{context}_branch_ref_rejected",
        )
    git_configuration = _exact(
        repository["git_configuration"],
        {
            "system_config_disabled",
            "system_config_path",
            "global_config_path",
            "system_attributes_disabled",
            "repository_fsmonitor_overridden_false",
            "repository_untracked_cache_overridden_false",
            "pager_disabled",
            "terminal_prompt_disabled",
            "optional_locks_disabled",
        },
        f"{context}_git_configuration",
    )
    _require(
        git_configuration
        == {
            "system_config_disabled": True,
            "system_config_path": "nul",
            "global_config_path": "nul",
            "system_attributes_disabled": True,
            "repository_fsmonitor_overridden_false": True,
            "repository_untracked_cache_overridden_false": True,
            "pager_disabled": True,
            "terminal_prompt_disabled": True,
            "optional_locks_disabled": True,
        },
        f"{context}_git_configuration_rejected",
    )
    raw_critical = repository["critical_sources"]
    _require(
        type(raw_critical) is dict and set(raw_critical) == CRITICAL_SOURCE_PATHS,
        f"{context}_critical_source_set_rejected",
    )
    critical = {
        name: _critical_row(raw_critical[name], f"{context}_critical_source")
        for name in sorted(raw_critical)
    }
    manifest, manifest_files = _validate_source_manifest(
        source["source_manifest"],
        expected_sha256=source["source_manifest_sha256"],
        context=f"{context}_source_manifest",
    )
    _require(
        source["source_manifest_inventory_sha256"] == manifest["file_inventory_sha256"],
        f"{context}_source_manifest_inventory_binding_rejected",
    )
    for name in CRITICAL_SOURCE_PATHS - {SOURCE_MANIFEST_RELATIVE, PRELOADER_RELATIVE}:
        _require(
            name in manifest_files
            and all(
                critical[name][field] == manifest_files[name][field]
                for field in ("bytes", "sha256", "git_blob_sha")
            ),
            f"{context}_critical_source_manifest_binding_rejected",
        )
    manifest_raw = _canonical(manifest)
    _require(
        critical[SOURCE_MANIFEST_RELATIVE]
        == {
            "bytes": len(manifest_raw),
            "sha256": _sha(manifest_raw),
            "git_blob_sha": _git_blob(manifest_raw),
        },
        f"{context}_manifest_exclusion_binding_rejected",
    )
    _require(
        critical[PRELOADER_RELATIVE]["bytes"] == shim["preloader_bytes"]
        and critical[PRELOADER_RELATIVE]["sha256"] == shim["preloader_sha256"]
        and source["preloader_sha256"] == shim["preloader_sha256"]
        and critical[SHIM_RELATIVE]["sha256"] == PRELOADER_SHIM_SHA256,
        f"{context}_preloader_exclusion_binding_rejected",
    )
    complete_files = dict(manifest_files)
    complete_files[SOURCE_MANIFEST_RELATIVE] = {
        "mode": "100644",
        **critical[SOURCE_MANIFEST_RELATIVE],
    }
    complete_files[PRELOADER_RELATIVE] = {
        "mode": "100644",
        **critical[PRELOADER_RELATIVE],
    }
    tree_rows = b"".join(
        (f"{item['mode']} blob {item['git_blob_sha']}\t{name}\0").encode("utf-8")
        for name, item in sorted(complete_files.items())
    )
    _require(
        _sha(tree_rows) == repository["tree_inventory_sha256"]
        and _git_tree_object_sha(complete_files) == repository["tree_object_sha"],
        f"{context}_tree_reconstruction_rejected",
    )
    capture_names = sorted(
        name
        for name in manifest_files
        if name.startswith(
            (
                "scripts/release_gpu_jit_lambda_controller/",
                "scripts/release_gpu_jit_lambda_live/",
                "scripts/release_gpu_jit_lambda_operator/",
                "scripts/release_gpu_jit_lambda_runtime/",
            )
        )
        or name
        in {
            ".github/release-control-policy.json",
            ".github/workflows/cuda-ci.yml",
            ".github/workflows/publish-pypi.yml",
            ".github/workflows/recover-github-release.yml",
            "poetry.lock",
            "pyproject.toml",
            "scripts/release_external_controls.py",
            "scripts/verify_release_recovery.py",
        }
    )
    capture_rows = b"".join(
        f"{name}\t{manifest_files[name]['bytes']}\t{manifest_files[name]['sha256']}\n".encode(
            "utf-8"
        )
        for name in capture_names
    )
    _require(
        source["captured_module_count"] == len(capture_names)
        and source["captured_module_inventory_sha256"] == _sha(capture_rows),
        f"{context}_captured_source_inventory_rejected",
    )
    return repository


def _validate_environment(value: Any, context: str) -> dict[str, Any]:
    environment = _exact(
        value,
        {
            "schema_version",
            "kind",
            "removed_name_count",
            "removed_names_sha256",
            "removed_values_observed",
            "ambient_credentials_retained",
            "ambient_proxies_retained",
        },
        context,
    )
    _require(
        environment["schema_version"] == 1
        and environment["kind"] == "operator-environment-scrub"
        and type(environment["removed_name_count"]) is int
        and environment["removed_name_count"] >= 0
        and SHA256_RE.fullmatch(str(environment["removed_names_sha256"])) is not None
        and environment["removed_values_observed"] is False
        and environment["ambient_credentials_retained"] is False
        and environment["ambient_proxies_retained"] is False,
        f"{context}_rejected",
    )
    return environment


def _validate_anonymous_transport(
    value: Any,
    *,
    expected_plan_sha256: str | None,
    context: str,
) -> dict[str, Any]:
    base = {
        "kind",
        "descriptor_owned_by_operator",
        "current_user_owner_verified",
        "regular_file",
        "terminal",
        "value_archived",
    }
    keys = set(base)
    if expected_plan_sha256 is not None:
        keys.update(
            {
                "confirmed_plan_sha256",
                "confirmation_exact_line",
                "confirmation_read_after_plan",
            }
        )
    receipt = _exact(value, keys, context)
    _require(
        receipt["kind"] == "anonymous-pipe"
        and receipt["descriptor_owned_by_operator"] is True
        and receipt["current_user_owner_verified"] is True
        and receipt["regular_file"] is False
        and receipt["terminal"] is False
        and receipt["value_archived"] is False,
        f"{context}_rejected",
    )
    if expected_plan_sha256 is not None:
        _require(
            receipt["confirmed_plan_sha256"] == expected_plan_sha256
            and receipt["confirmation_exact_line"] is True
            and receipt["confirmation_read_after_plan"] is True,
            f"{context}_confirmation_rejected",
        )
    return receipt


def _validate_executable_acl(value: Any, *, expected_owner: str, context: str) -> dict[str, Any]:
    acl = _exact(
        value,
        {
            "owner_sid",
            "expected_owner_sid",
            "unprivileged_write_ace_present",
            "dacl_ace_count",
            "dacl_inventory_sha256",
        },
        context,
    )
    _require(
        acl["owner_sid"] == expected_owner
        and acl["expected_owner_sid"] == expected_owner
        and acl["unprivileged_write_ace_present"] is False
        and type(acl["dacl_ace_count"]) is int
        and acl["dacl_ace_count"] > 0
        and SHA256_RE.fullmatch(str(acl["dacl_inventory_sha256"])) is not None,
        f"{context}_rejected",
    )
    return acl


def _validate_signature(value: Any, *, expected: Mapping[str, str], context: str) -> dict[str, Any]:
    signature = _exact(value, {"status", "subject", "thumbprint"}, context)
    _require(
        signature
        == {
            "status": "Valid",
            "subject": expected["authenticode_subject"],
            "thumbprint": expected["authenticode_thumbprint"],
        },
        f"{context}_rejected",
    )
    return signature


def _validate_executables(value: Any, context: str) -> dict[str, Any]:
    executables = _exact(value, {"git", "gh", "ssh", "python"}, context)
    common = {
        "absolute_path",
        "sha256",
        "version",
        "regular_file",
        "symlink_or_reparse",
        "path_lookup_used",
        "hardlink_count",
    }
    pinned = common | {
        "pinned_reviewed_identity",
        "acl",
        "authenticode",
        "authenticode_validated_by_pinned_helper",
    }
    for name in ("git", "gh", "ssh"):
        expected = PINNED_EXECUTABLES[name]
        keys = pinned | ({"resolved_runtime"} if name == "git" else set())
        row = _exact(executables[name], keys, f"{context}_{name}")
        _require(
            row["absolute_path"] == expected["absolute_path"]
            and row["sha256"] == expected["sha256"]
            and row["version"] == expected["version"]
            and row["regular_file"] is True
            and row["symlink_or_reparse"] is False
            and row["path_lookup_used"] is False
            and type(row["hardlink_count"]) is int
            and row["hardlink_count"] >= 1
            and (name != "gh" or row["hardlink_count"] == 1)
            and row["pinned_reviewed_identity"] is True
            and row["authenticode_validated_by_pinned_helper"] is True,
            f"{context}_{name}_rejected",
        )
        acl = _validate_executable_acl(
            row["acl"], expected_owner=expected["owner_sid"], context=f"{context}_{name}_acl"
        )
        signature = _validate_signature(
            row["authenticode"], expected=expected, context=f"{context}_{name}_signature"
        )
        if name == "git":
            runtime = _exact(
                row["resolved_runtime"],
                {"absolute_path", "sha256", "version", "hardlink_count", "acl", "authenticode"},
                f"{context}_git_runtime",
            )
            _require(
                runtime["absolute_path"] == expected["runtime_absolute_path"]
                and runtime["sha256"] == expected["runtime_sha256"]
                and runtime["version"] == expected["version"]
                and type(runtime["hardlink_count"]) is int
                and runtime["hardlink_count"] >= 1
                and runtime["acl"] == acl
                and runtime["authenticode"] == signature,
                f"{context}_git_runtime_rejected",
            )
    python = _exact(executables["python"], common | {"pinned_runtime_manifest_authority"}, context)
    _require(
        bool(_windows_absolute(python["absolute_path"], f"{context}_python_path"))
        and python["sha256"] == PYTHON_EXECUTABLE_SHA256
        and python["version"] == "Python 3.13.15"
        and python["regular_file"] is True
        and python["symlink_or_reparse"] is False
        and python["path_lookup_used"] is False
        and python["hardlink_count"] == 1
        and type(python["hardlink_count"]) is int
        and python["pinned_runtime_manifest_authority"] is True,
        f"{context}_python_rejected",
    )
    return executables


def _validate_install_receipts(
    preloader: Mapping[str, Any], context: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    python = _exact(
        preloader["python_install_receipt"],
        {
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
        },
        f"{context}_python_install",
    )
    site = _exact(
        preloader["site_install_receipt"],
        {
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
        },
        f"{context}_site_install",
    )
    _windows_absolute(python["python_runtime_root"], f"{context}_python_root")
    _windows_absolute(site["runtime_root"], f"{context}_site_root")
    _require(
        python["schema_version"] == 1
        and python["kind"] == "explainiverse-operator-python-runtime-installed"
        and python["archive_sha256"] == PYTHON_ARCHIVE_SHA256
        and python["manifest_sha256"] == PYTHON_MANIFEST_SHA256
        and python["file_count"] == 34
        and type(python["file_count"]) is int
        and python["directory_count"] == 0
        and type(python["directory_count"]) is int
        and python["file_inventory_sha256"] == PYTHON_FILE_INVENTORY_SHA256
        and python["owner_private_acl_applied_before_children"] is True
        and python["site_processing_disabled_by_embeddable_pth"] is True
        and python["untracked_files_or_directories_present"] is False
        and python["crash_recovery"] == "discard-partial-directory-and-create-a-new-path"
        and preloader["python_install_receipt_sha256"] == _sha(_canonical(python)),
        f"{context}_python_install_rejected",
    )
    _require(
        site["schema_version"] == 1
        and site["kind"] == "explainiverse-operator-runtime-installed"
        and site["manifest_sha256"] == SITE_MANIFEST_SHA256
        and site["file_count"] == 756
        and type(site["file_count"]) is int
        and site["directory_count"] == 113
        and type(site["directory_count"]) is int
        and site["file_inventory_sha256"] == SITE_FILE_INVENTORY_SHA256
        and site["owner_private_acl_applied_before_children"] is True
        and site["pip_present_in_runtime"] is False
        and site["record_files_present"] is False
        and site["generated_scripts_present"] is False
        and site["bytecode_present"] is False
        and site["crash_recovery"] == "discard-partial-directory-and-create-a-new-path"
        and preloader["site_install_receipt_sha256"] == _sha(_canonical(site)),
        f"{context}_site_install_rejected",
    )
    return python, site


def _validate_directory_receipt(
    public_value: Any, validation_value: Any, *, context: str
) -> tuple[str, str, str]:
    public = _exact(
        public_value,
        {
            "captured_at",
            "receipt_sha256",
            "absolute_path_redacted",
            "directory_identity_recorded",
            "no_reparse_or_symlink",
            "owner_private",
            "acl",
        },
        f"{context}_public",
    )
    validation = _exact(
        validation_value,
        {
            "validated_at",
            "receipt_sha256",
            "absolute_path_redacted",
            "directory_identity_recorded",
            "no_reparse_or_symlink",
            "owner_private",
            "acl_evidence_sha256",
        },
        f"{context}_validation",
    )
    acl = _exact(
        public["acl"],
        {
            "owner_sid",
            "current_user_sid",
            "inheritance_protected",
            "child_inheritance_enabled",
            "aces",
            "security_descriptor_sha256",
            "security_descriptor_bytes",
            "captured_at",
            "evidence_sha256",
        },
        f"{context}_acl",
    )
    owner = _text(acl["owner_sid"], f"{context}_owner")
    _require(SID_RE.fullmatch(owner) is not None and acl["current_user_sid"] == owner, context)
    aces = acl["aces"]
    _require(
        type(aces) is list
        and len(aces) == 3
        and aces == sorted(aces, key=lambda item: str(item.get("sid")))
        and {item.get("sid") for item in aces if type(item) is dict}
        == {owner, "S-1-5-18", "S-1-5-32-544"}
        and all(
            type(item) is dict
            and set(item) == {"sid", "access", "rights", "mask", "ace_flags"}
            and item["access"] == "allow"
            and item["rights"] == "full-control"
            and item["mask"] == 2_032_127
            and type(item["mask"]) is int
            and item["ace_flags"] == 3
            and type(item["ace_flags"]) is int
            for item in aces
        )
        and acl["inheritance_protected"] is True
        and acl["child_inheritance_enabled"] is True,
        f"{context}_acl_rejected",
    )
    descriptor_sha = _digest(
        acl["security_descriptor_sha256"], f"{context}_descriptor_sha_rejected"
    )
    _positive_int(acl["security_descriptor_bytes"], f"{context}_descriptor_size_rejected")
    acl_material = {
        key: item for key, item in acl.items() if key not in {"captured_at", "evidence_sha256"}
    }
    _require(
        acl["evidence_sha256"] == _sha(_canonical(acl_material)),
        f"{context}_acl_digest_rejected",
    )
    captured = _time(acl["captured_at"], f"{context}_captured_at_rejected")
    _require(
        public["captured_at"] == acl["captured_at"]
        and all(
            public[name] is True
            for name in (
                "absolute_path_redacted",
                "directory_identity_recorded",
                "no_reparse_or_symlink",
                "owner_private",
            )
        )
        and _time(validation["validated_at"], f"{context}_validated_at_rejected") >= captured
        and validation["receipt_sha256"] == public["receipt_sha256"]
        and all(
            validation[name] is True
            for name in (
                "absolute_path_redacted",
                "directory_identity_recorded",
                "no_reparse_or_symlink",
                "owner_private",
            )
        )
        and validation["acl_evidence_sha256"] == acl["evidence_sha256"],
        f"{context}_binding_rejected",
    )
    receipt_sha = _digest(public["receipt_sha256"], f"{context}_receipt_sha_rejected")
    return receipt_sha, owner, descriptor_sha


def _validate_runtime_trees(
    *,
    python_tree_value: Any,
    site_tree_value: Any,
    python_install: Mapping[str, Any],
    site_install: Mapping[str, Any],
    context: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    python_tree = _exact(
        python_tree_value,
        {
            "python_root",
            "file_count",
            "directory_count",
            "file_inventory_sha256",
            "official_archive_sha256",
            "untracked_files_or_directories_present",
            "all_runtime_bytes_match_official_archive",
        },
        f"{context}_python_tree",
    )
    site_tree = _exact(
        site_tree_value,
        {
            "site_root",
            "file_count",
            "directory_count",
            "file_inventory_sha256",
            "untracked_files_or_directories_present",
            "bytecode_present",
            "all_importable_bytes_match_verified_wheels",
        },
        f"{context}_site_tree",
    )
    _require(
        python_tree
        == {
            "python_root": python_install["python_runtime_root"],
            "file_count": 34,
            "directory_count": 0,
            "file_inventory_sha256": PYTHON_FILE_INVENTORY_SHA256,
            "official_archive_sha256": PYTHON_ARCHIVE_SHA256,
            "untracked_files_or_directories_present": False,
            "all_runtime_bytes_match_official_archive": True,
        },
        f"{context}_python_tree_rejected",
    )
    _require(
        site_tree
        == {
            "site_root": site_install["runtime_root"],
            "file_count": 756,
            "directory_count": 113,
            "file_inventory_sha256": SITE_FILE_INVENTORY_SHA256,
            "untracked_files_or_directories_present": False,
            "bytecode_present": False,
            "all_importable_bytes_match_verified_wheels": True,
        },
        f"{context}_site_tree_rejected",
    )
    return python_tree, site_tree


def _validate_bootstrap(
    value: Any,
    *,
    python_install: Mapping[str, Any],
    site_install: Mapping[str, Any],
    python_executable: Mapping[str, Any],
    working_directory: str,
    context: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    bootstrap = _exact(
        value,
        {
            "schema_version",
            "kind",
            "python_manifest_sha256",
            "python_archive_sha256",
            "python_tree",
            "manifest_sha256",
            "archive_set_sha256",
            "runtime_requirements_sha256",
            "bootstrap_requirements_sha256",
            "base_python_executable",
            "base_python_executable_sha256",
            "preactivation",
            "site_tree",
            "activation_paths",
            "site_processing_disabled",
            "pth_executed_by_cpython",
            "verified_pywin32_bootstrap_imported_after_verification",
        },
        context,
    )
    python_tree, site_tree = _validate_runtime_trees(
        python_tree_value=bootstrap["python_tree"],
        site_tree_value=bootstrap["site_tree"],
        python_install=python_install,
        site_install=site_install,
        context=context,
    )
    preactivation = _exact(
        bootstrap["preactivation"],
        {"working_directory", "sys_path_sha256", "only_base_stdlib_roots"},
        f"{context}_preactivation",
    )
    site_root = PureWindowsPath(site_install["runtime_root"])
    _require(
        preactivation["working_directory"] == working_directory
        and SHA256_RE.fullmatch(str(preactivation["sys_path_sha256"])) is not None
        and preactivation["only_base_stdlib_roots"] is True
        and bootstrap["schema_version"] == 1
        and bootstrap["kind"] == "explainiverse-operator-pre-site-bootstrap"
        and bootstrap["python_manifest_sha256"] == PYTHON_MANIFEST_SHA256
        and bootstrap["python_archive_sha256"] == PYTHON_ARCHIVE_SHA256
        and bootstrap["manifest_sha256"] == SITE_MANIFEST_SHA256
        and bootstrap["archive_set_sha256"] == SITE_ARCHIVE_SET_SHA256
        and bootstrap["runtime_requirements_sha256"] == RUNTIME_REQUIREMENTS_SHA256
        and bootstrap["bootstrap_requirements_sha256"] == BOOTSTRAP_REQUIREMENTS_SHA256
        and bootstrap["base_python_executable"] == python_executable["absolute_path"]
        and bootstrap["base_python_executable_sha256"] == PYTHON_EXECUTABLE_SHA256
        and bootstrap["activation_paths"]
        == [
            str(site_root),
            str(site_root / "win32"),
            str(site_root / "win32" / "lib"),
            str(site_root / "pythonwin"),
        ]
        and bootstrap["site_processing_disabled"] is True
        and bootstrap["pth_executed_by_cpython"] is False
        and bootstrap["verified_pywin32_bootstrap_imported_after_verification"] is True,
        f"{context}_rejected",
    )
    return bootstrap, python_tree, site_tree


def _validate_early_runtime_boundary(
    value: Any,
    *,
    working_directory: str,
    directory_bindings: Mapping[str, tuple[str, str]],
    expected_held_handle_count: int,
    context: str,
) -> dict[str, Any]:
    early = _exact(
        value,
        {
            "schema_version",
            "kind",
            "acl",
            "held_trees",
            "all_runtime_and_receipt_roots_owner_private",
            "all_runtime_and_receipt_paths_held_without_write_or_delete_share",
            "validated_before_third_party_site_or_third_party_native_import",
            "pinned_official_python_runtime_is_the_pre_hold_trust_boundary",
            "working_directory",
            "working_directory_repository_disjoint",
            "evidence_sha256",
        },
        context,
    )
    acl_inventory = _exact(
        early["acl"],
        {"python_root", "site_root", "python_receipt_root", "site_receipt_root"},
        f"{context}_acl_inventory",
    )
    held = _exact(
        early["held_trees"],
        {
            "root_count",
            "held_handle_count",
            "write_share_allowed",
            "delete_share_allowed",
            "read_share_allowed",
            "held_before_third_party_site_or_third_party_native_import",
        },
        f"{context}_held",
    )
    _require(
        held["root_count"] == 4
        and type(held["root_count"]) is int
        and type(held["held_handle_count"]) is int
        and held["held_handle_count"] == expected_held_handle_count
        and held["write_share_allowed"] is False
        and held["delete_share_allowed"] is False
        and held["read_share_allowed"] is True
        and held["held_before_third_party_site_or_third_party_native_import"] is True,
        f"{context}_held_rejected",
    )
    owner: str | None = None
    for name in sorted(acl_inventory):
        acl = _exact(
            acl_inventory[name],
            {
                "owner_sid",
                "inheritance_protected",
                "child_inheritance_enabled",
                "allowed_sids",
                "ace_count",
                "rights",
                "security_descriptor_sha256",
                "security_descriptor_bytes",
                "validated_before_third_party_site_or_third_party_native_import",
                "pinned_stdlib_native_modules_loaded_before_hold",
            },
            f"{context}_{name}",
        )
        observed_owner = _text(acl["owner_sid"], f"{context}_{name}_owner")
        _require(
            SID_RE.fullmatch(observed_owner) is not None
            and (owner is None or owner == observed_owner)
            and acl["inheritance_protected"] is True
            and acl["child_inheritance_enabled"] is True
            and acl["allowed_sids"] == sorted([observed_owner, "S-1-5-18", "S-1-5-32-544"])
            and acl["ace_count"] == 3
            and type(acl["ace_count"]) is int
            and acl["rights"] == "full-control"
            and type(acl["security_descriptor_bytes"]) is int
            and acl["security_descriptor_bytes"] > 0
            and acl["validated_before_third_party_site_or_third_party_native_import"] is True
            and acl["pinned_stdlib_native_modules_loaded_before_hold"] is True,
            f"{context}_{name}_rejected",
        )
        descriptor = _digest(acl["security_descriptor_sha256"], f"{context}_{name}_descriptor")
        receipt_owner, receipt_descriptor = directory_bindings[name]
        _require(
            observed_owner == receipt_owner and descriptor == receipt_descriptor,
            f"{context}_{name}_directory_binding_rejected",
        )
        owner = observed_owner
    material = {key: item for key, item in early.items() if key != "evidence_sha256"}
    _require(
        early["schema_version"] == 1
        and early["kind"] == "explainiverse-operator-early-runtime-boundary"
        and early["all_runtime_and_receipt_roots_owner_private"] is True
        and early["all_runtime_and_receipt_paths_held_without_write_or_delete_share"] is True
        and early["validated_before_third_party_site_or_third_party_native_import"] is True
        and early["pinned_official_python_runtime_is_the_pre_hold_trust_boundary"] is True
        and early["working_directory"] == working_directory
        and early["working_directory_repository_disjoint"] is True
        and early["evidence_sha256"] == _sha(_canonical(material)),
        f"{context}_rejected",
    )
    return early


def _validate_sealed_resources(
    value: Any,
    *,
    expected_policy_sha256: str,
    expected_controller_source_sha256: str,
    expected_runtime_bundle_sha256: str,
    repository: Mapping[str, Any],
    context: str,
) -> dict[str, Any]:
    sealed = _exact(
        value,
        {
            "schema_version",
            "kind",
            "policy_sha256",
            "controller_source_sha256",
            "runtime_bundle_sha256",
            "runtime_file_sha256",
            "captured_before_project_import",
            "live_repository_reopen_permitted",
        },
        context,
    )
    runtime_files = _exact(
        sealed["runtime_file_sha256"], set(RUNTIME_BUNDLE_NAMES), f"{context}_files"
    )
    for value_digest in runtime_files.values():
        _digest(value_digest, f"{context}_file_digest")
    critical = repository["critical_sources"]
    _require(
        sealed["schema_version"] == 1
        and sealed["kind"] == "explainiverse-operator-sealed-resource-binding"
        and sealed["policy_sha256"] == expected_policy_sha256
        and sealed["controller_source_sha256"] == expected_controller_source_sha256
        and sealed["runtime_bundle_sha256"] == expected_runtime_bundle_sha256
        and critical[".github/release-control-policy.json"]["sha256"] == expected_policy_sha256
        and critical["scripts/release_gpu_jit_lambda_controller/controller.py"]["sha256"]
        == expected_controller_source_sha256
        and all(
            runtime_files[name]
            == critical[f"scripts/release_gpu_jit_lambda_runtime/{name}"]["sha256"]
            for name in RUNTIME_BUNDLE_NAMES
        )
        and sealed["captured_before_project_import"] is True
        and sealed["live_repository_reopen_permitted"] is False,
        f"{context}_rejected",
    )
    return sealed


def _validate_preloader(
    value: Any,
    *,
    expected_phase: str,
    expected_head_sha: str,
    expected_ref: str,
    expected_policy_sha256: str,
    expected_controller_source_sha256: str,
    expected_runtime_bundle_sha256: str,
    environment: Mapping[str, Any],
    inventory_executables: Mapping[str, Any],
    inventory_repository: Mapping[str, Any],
    context: str,
) -> dict[str, Any]:
    preloader = _exact(
        value,
        {
            "schema_version",
            "kind",
            "shim",
            "source",
            "bootstrap",
            "python_runtime_directory_receipt",
            "python_runtime_validation",
            "runtime_site_directory_receipt",
            "runtime_site_validation",
            "python_install_receipt",
            "python_install_receipt_sha256",
            "python_install_directory_receipt",
            "python_install_directory_validation",
            "site_install_receipt",
            "site_install_receipt_sha256",
            "site_install_directory_receipt",
            "site_install_directory_validation",
            "environment",
            "early_runtime_boundary",
            "sealed_resources",
            "working_directory",
            "working_directory_is_python_install_receipt_directory",
            "isolated",
            "safe_path",
            "site_disabled",
            "bytecode_disabled",
            "repository_absent_from_sys_path",
            "project_imports_from_captured_bytes",
            "evidence_sha256",
        },
        context,
    )
    _require(
        preloader["schema_version"] == 1
        and preloader["kind"] == "explainiverse-operator-isolated-preloader"
        and preloader["environment"] == environment
        and preloader["working_directory_is_python_install_receipt_directory"] is True
        and preloader["isolated"] is True
        and preloader["safe_path"] is True
        and preloader["site_disabled"] is True
        and preloader["bytecode_disabled"] is True
        and preloader["repository_absent_from_sys_path"] is True
        and preloader["project_imports_from_captured_bytes"] is True,
        f"{context}_rejected",
    )
    working = _windows_absolute(preloader["working_directory"], f"{context}_working")
    shim = _exact(
        preloader["shim"],
        {
            "schema_version",
            "kind",
            "preloader_path",
            "preloader_bytes",
            "preloader_sha256",
            "shim_sha256",
            "stable_descriptor_read",
            "compiled_verified_bytes_without_reopen",
        },
        f"{context}_shim",
    )
    _require(
        shim["schema_version"] == 1
        and shim["kind"] == "explainiverse-operator-preloader-shim"
        and bool(_windows_absolute(shim["preloader_path"], f"{context}_shim_path"))
        and type(shim["preloader_bytes"]) is int
        and 1 <= shim["preloader_bytes"] <= 4 * 1024 * 1024
        and SHA256_RE.fullmatch(str(shim["preloader_sha256"])) is not None
        and shim["shim_sha256"] == PRELOADER_SHIM_SHA256
        and shim["stable_descriptor_read"] is True
        and shim["compiled_verified_bytes_without_reopen"] is True,
        f"{context}_shim_rejected",
    )
    source = _exact(
        preloader["source"],
        {
            "schema_version",
            "kind",
            "repository_root",
            "origin_url",
            "head_sha",
            "head_and_origin_verified_during_credential_free_inventory",
            "source_manifest",
            "source_manifest_sha256",
            "source_manifest_inventory_sha256",
            "tracked_and_untracked_clean",
            "runtime_git_dependency",
            "preloader_path",
            "preloader_sha256",
            "captured_module_count",
            "captured_module_inventory_sha256",
            "project_modules_execute_from_captured_bytes",
            "arguments_sha256",
            "evidence_sha256",
        },
        f"{context}_source",
    )
    source_material = {key: item for key, item in source.items() if key != "evidence_sha256"}
    _require(
        source["schema_version"] == 1
        and source["kind"] == "explainiverse-operator-clean-source-preload"
        and source["origin_url"] == ORIGIN_URL
        and source["head_sha"] == expected_head_sha
        and source["head_and_origin_verified_during_credential_free_inventory"] is False
        and source["tracked_and_untracked_clean"] is True
        and source["runtime_git_dependency"] is False
        and source["preloader_path"] == shim["preloader_path"]
        and source["preloader_sha256"] == shim["preloader_sha256"]
        and source["project_modules_execute_from_captured_bytes"] is True
        and SHA256_RE.fullmatch(str(source["arguments_sha256"])) is not None
        and source["evidence_sha256"] == _sha(_canonical(source_material)),
        f"{context}_source_rejected",
    )
    repository = _validate_repository(
        inventory_repository,
        expected_phase=expected_phase,
        expected_head_sha=expected_head_sha,
        expected_ref=expected_ref,
        source=source,
        shim=shim,
        context=f"{context}_repository",
    )
    root = PureWindowsPath(repository["absolute_root"])
    _require(
        PureWindowsPath(shim["preloader_path"]) == root / PurePosixPath(PRELOADER_RELATIVE)
        and source["repository_root"] == repository["absolute_root"],
        f"{context}_source_path_rejected",
    )
    python_install, site_install = _validate_install_receipts(preloader, context)
    roots = {
        "repository": root,
        "python": PureWindowsPath(python_install["python_runtime_root"]),
        "site": PureWindowsPath(site_install["runtime_root"]),
        "working": PureWindowsPath(working),
    }
    for left_name, left in roots.items():
        for right_name, right in roots.items():
            if left_name >= right_name:
                continue
            _require(
                left != right and left not in right.parents and right not in left.parents,
                f"{context}_{left_name}_{right_name}_not_disjoint",
            )
    pairs = (
        (
            "python_root",
            "python_runtime_directory_receipt",
            "python_runtime_validation",
        ),
        ("site_root", "runtime_site_directory_receipt", "runtime_site_validation"),
        (
            "python_receipt_root",
            "python_install_directory_receipt",
            "python_install_directory_validation",
        ),
        (
            "site_receipt_root",
            "site_install_directory_receipt",
            "site_install_directory_validation",
        ),
    )
    receipt_shas: list[str] = []
    directory_bindings: dict[str, tuple[str, str]] = {}
    for logical_name, public_name, validation_name in pairs:
        receipt_sha, owner, descriptor_sha = _validate_directory_receipt(
            preloader[public_name],
            preloader[validation_name],
            context=f"{context}_{public_name}",
        )
        receipt_shas.append(receipt_sha)
        directory_bindings[logical_name] = (owner, descriptor_sha)
    _require(len(set(receipt_shas)) == 4, f"{context}_directory_receipt_reused")
    bootstrap, _, _ = _validate_bootstrap(
        preloader["bootstrap"],
        python_install=python_install,
        site_install=site_install,
        python_executable=inventory_executables["python"],
        working_directory=working,
        context=f"{context}_bootstrap",
    )
    _validate_early_runtime_boundary(
        preloader["early_runtime_boundary"],
        working_directory=working,
        directory_bindings=directory_bindings,
        expected_held_handle_count=(
            1
            + python_install["file_count"]
            + python_install["directory_count"]
            + 1
            + site_install["file_count"]
            + site_install["directory_count"]
            + 2
            + 2
        ),
        context=f"{context}_early",
    )
    _validate_sealed_resources(
        preloader["sealed_resources"],
        expected_policy_sha256=expected_policy_sha256,
        expected_controller_source_sha256=expected_controller_source_sha256,
        expected_runtime_bundle_sha256=expected_runtime_bundle_sha256,
        repository=repository,
        context=f"{context}_sealed",
    )
    _require(
        bootstrap["base_python_executable"]
        == str(PureWindowsPath(python_install["python_runtime_root"]) / "python.exe"),
        f"{context}_python_executable_root_binding_rejected",
    )
    material = {key: item for key, item in preloader.items() if key != "evidence_sha256"}
    _require(
        preloader["evidence_sha256"] == _sha(_canonical(material)),
        f"{context}_digest_rejected",
    )
    return preloader


def _validate_secure_launch(
    value: Any,
    *,
    expected_phase: str,
    expected_head_sha: str,
    expected_ref: str,
    expected_policy_sha256: str,
    expected_controller_source_sha256: str,
    expected_runtime_bundle_sha256: str,
    environment: Mapping[str, Any],
    inventory_executables: Mapping[str, Any],
    inventory_repository: Mapping[str, Any],
    context: str,
) -> dict[str, Any]:
    launch = _exact(
        value,
        {
            "schema_version",
            "kind",
            "isolated",
            "safe_path",
            "ignore_environment",
            "no_user_site",
            "no_site",
            "dont_write_bytecode",
            "invocation",
            "working_directory",
            "repository_absent_from_sys_path",
            "sys_path_sha256",
            "site_processing_disabled",
            "preloader",
            "controller_imported_after_launch_validation",
            "windows_handle_transport",
            "inherited_handle_count",
            "handles_distinct",
            "file_type_pipe",
            "child_handles_made_noninheritable",
            "raw_handle_values_archived",
            "secret_values_in_argv",
            "secret_values_in_environment",
            "windows_launcher_parent_declaration",
        },
        context,
    )
    _require(
        launch["schema_version"] == 1
        and launch["kind"] == "operator-secure-interpreter-launch"
        and all(
            launch[name] is True
            for name in (
                "isolated",
                "safe_path",
                "ignore_environment",
                "no_user_site",
                "no_site",
                "dont_write_bytecode",
                "repository_absent_from_sys_path",
                "site_processing_disabled",
                "controller_imported_after_launch_validation",
                "windows_handle_transport",
                "handles_distinct",
                "file_type_pipe",
                "child_handles_made_noninheritable",
            )
        )
        and launch["invocation"] == "pinned-python -I -S -B -c <byte-sealing-shim>"
        and launch["inherited_handle_count"] == 2
        and type(launch["inherited_handle_count"]) is int
        and launch["raw_handle_values_archived"] is False
        and launch["secret_values_in_argv"] is False
        and launch["secret_values_in_environment"] is False
        and SHA256_RE.fullmatch(str(launch["sys_path_sha256"])) is not None,
        f"{context}_rejected",
    )
    preloader = _validate_preloader(
        launch["preloader"],
        expected_phase=expected_phase,
        expected_head_sha=expected_head_sha,
        expected_ref=expected_ref,
        expected_policy_sha256=expected_policy_sha256,
        expected_controller_source_sha256=expected_controller_source_sha256,
        expected_runtime_bundle_sha256=expected_runtime_bundle_sha256,
        environment=environment,
        inventory_executables=inventory_executables,
        inventory_repository=inventory_repository,
        context=f"{context}_preloader",
    )
    _require(
        launch["working_directory"] == preloader["working_directory"],
        f"{context}_working_directory_binding_rejected",
    )
    parent = _exact(
        launch["windows_launcher_parent_declaration"],
        {
            "receipt_sha256",
            "preloader_metadata_matched",
            "parent_provenance_authenticated",
            "security_authority_derived_from_declaration",
            "child_revalidated_handle_transport_and_sealed_resources",
        },
        f"{context}_parent_declaration",
    )
    _require(
        SHA256_RE.fullmatch(str(parent["receipt_sha256"])) is not None
        and parent["preloader_metadata_matched"] is True
        and parent["parent_provenance_authenticated"] is False
        and parent["security_authority_derived_from_declaration"] is False
        and parent["child_revalidated_handle_transport_and_sealed_resources"] is True,
        f"{context}_parent_declaration_rejected",
    )
    return launch


def _validate_environment_revalidation(
    value: Any,
    *,
    python_install: Mapping[str, Any],
    site_install: Mapping[str, Any],
    context: str,
) -> dict[str, Any]:
    receipt = _exact(
        value,
        {
            "schema_version",
            "kind",
            "python_manifest_sha256",
            "python_archive_sha256",
            "python_tree",
            "manifest_sha256",
            "archive_set_sha256",
            "runtime_requirements_sha256",
            "bootstrap_requirements_sha256",
            "site_tree",
            "activation_paths",
            "site_processing_disabled",
            "pth_executed_by_cpython",
        },
        context,
    )
    _validate_runtime_trees(
        python_tree_value=receipt["python_tree"],
        site_tree_value=receipt["site_tree"],
        python_install=python_install,
        site_install=site_install,
        context=context,
    )
    site_root = PureWindowsPath(site_install["runtime_root"])
    _require(
        receipt["schema_version"] == 1
        and receipt["kind"] == "explainiverse-operator-enabled-environment-revalidation"
        and receipt["python_manifest_sha256"] == PYTHON_MANIFEST_SHA256
        and receipt["python_archive_sha256"] == PYTHON_ARCHIVE_SHA256
        and receipt["manifest_sha256"] == SITE_MANIFEST_SHA256
        and receipt["archive_set_sha256"] == SITE_ARCHIVE_SET_SHA256
        and receipt["runtime_requirements_sha256"] == RUNTIME_REQUIREMENTS_SHA256
        and receipt["bootstrap_requirements_sha256"] == BOOTSTRAP_REQUIREMENTS_SHA256
        and receipt["activation_paths"]
        == [
            str(site_root),
            str(site_root / "win32"),
            str(site_root / "win32" / "lib"),
            str(site_root / "pythonwin"),
        ]
        and receipt["site_processing_disabled"] is True
        and receipt["pth_executed_by_cpython"] is False,
        f"{context}_rejected",
    )
    return receipt


def _validate_dependency_inventory(
    value: Any,
    *,
    repository_root: str,
    python_install: Mapping[str, Any],
    site_install: Mapping[str, Any],
    context: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dependencies = _exact(
        value,
        {
            "schema_version",
            "kind",
            "target",
            "lock",
            "distributions",
            "installed_distribution_set_exact",
            "startup_pth",
            "site_manifest",
            "wheel_derived_site_manifest",
        },
        context,
    )
    lock = _exact(
        dependencies["lock"],
        {"relative_path", "absolute_path", "bytes", "sha256", "require_hashes", "wheels_only"},
        f"{context}_lock",
    )
    site_manifest = _exact(
        dependencies["site_manifest"],
        {"relative_path", "bytes", "sha256"},
        f"{context}_site_manifest",
    )
    root = PureWindowsPath(repository_root)
    _require(
        dependencies["schema_version"] == 1
        and dependencies["kind"] == "explainiverse-operator-dependency-inventory"
        and dependencies["target"] == "CPython 3.13.15 Windows AMD64"
        and lock
        == {
            "relative_path": RUNTIME_LOCK_RELATIVE,
            "absolute_path": str(root / PurePosixPath(RUNTIME_LOCK_RELATIVE)),
            "bytes": RUNTIME_REQUIREMENTS_BYTES,
            "sha256": RUNTIME_REQUIREMENTS_SHA256,
            "require_hashes": True,
            "wheels_only": True,
        }
        and site_manifest
        == {
            "relative_path": SITE_MANIFEST_RELATIVE,
            "bytes": SITE_MANIFEST_BYTES,
            "sha256": SITE_MANIFEST_SHA256,
        }
        and dependencies["installed_distribution_set_exact"] is True,
        f"{context}_rejected",
    )
    startup = _exact(
        dependencies["startup_pth"],
        {"allowed_files", "unexpected_files_present", "all_startup_files_hashed"},
        f"{context}_startup",
    )
    _require(
        startup["allowed_files"] == ALLOWED_PTH_FILES
        and startup["unexpected_files_present"] is False
        and startup["all_startup_files_hashed"] is True,
        f"{context}_startup_rejected",
    )
    revalidation = _validate_environment_revalidation(
        dependencies["wheel_derived_site_manifest"],
        python_install=python_install,
        site_install=site_install,
        context=f"{context}_revalidation",
    )
    revalidation_sha = _sha(_canonical(revalidation))
    distributions = _exact(
        dependencies["distributions"], set(LOCKED_DISTRIBUTIONS), f"{context}_distributions"
    )
    for name in sorted(distributions):
        expected = LOCKED_DISTRIBUTIONS[name]
        row = _exact(
            distributions[name],
            {
                "distribution",
                "version",
                "archive_filename",
                "archive_sha256",
                "file_count",
                "total_bytes",
                "inventory_sha256",
                "actual_files_hashed",
                "actual_tree_revalidation_sha256",
                "record_metadata_trusted",
                "wheel_archive_manifest_authoritative",
                "bytecode_excluded",
                "locked_version",
                "source_wheel_sha256",
            },
            f"{context}_{name}",
        )
        _require(
            row["distribution"] == name
            and row["version"] == expected["version"]
            and row["archive_filename"] == expected["archive_filename"]
            and row["archive_sha256"] == expected["archive_sha256"]
            and row["file_count"] == expected["file_count"]
            and type(row["file_count"]) is int
            and row["total_bytes"] == expected["total_bytes"]
            and type(row["total_bytes"]) is int
            and row["inventory_sha256"] == expected["inventory_sha256"]
            and row["actual_files_hashed"] is True
            and row["actual_tree_revalidation_sha256"] == revalidation_sha
            and row["record_metadata_trusted"] is False
            and row["wheel_archive_manifest_authoritative"] is True
            and row["bytecode_excluded"] is True
            and row["locked_version"] == expected["version"]
            and row["source_wheel_sha256"] == expected["archive_sha256"],
            f"{context}_{name}_rejected",
        )
    return dependencies, revalidation


def _validate_interpreter_runtime(
    value: Any,
    *,
    repository_root: str,
    python_install: Mapping[str, Any],
    site_install: Mapping[str, Any],
    expected_revalidation: Mapping[str, Any],
    context: str,
) -> dict[str, Any]:
    runtime = _exact(
        value,
        {
            "secure_flags",
            "working_directory",
            "sys_path_first",
            "sys_path",
            "sys_path_sha256",
            "repository_present_in_sys_path",
            "prefixes",
            "site_package_roots",
            "module_resolutions",
            "pinned_runtime_and_site_revalidation",
        },
        context,
    )
    _require(
        runtime["secure_flags"]
        == {
            "isolated": True,
            "safe_path": True,
            "ignore_environment": True,
            "no_user_site": True,
            "no_site": True,
            "dont_write_bytecode": True,
        }
        and runtime["repository_present_in_sys_path"] is False
        and runtime["pinned_runtime_and_site_revalidation"] == expected_revalidation,
        f"{context}_rejected",
    )
    working = PureWindowsPath(_windows_absolute(runtime["working_directory"], f"{context}_working"))
    repository = PureWindowsPath(repository_root)
    python_root = PureWindowsPath(python_install["python_runtime_root"])
    site_root = PureWindowsPath(site_install["runtime_root"])
    _require(
        all(
            working != candidate
            and working not in candidate.parents
            and candidate not in working.parents
            for candidate in (repository, python_root, site_root)
        ),
        f"{context}_working_directory_rejected",
    )
    expected_paths = {
        str(python_root),
        str(python_root / "python313.zip"),
        str(site_root),
        str(site_root / "win32"),
        str(site_root / "win32" / "lib"),
        str(site_root / "pythonwin"),
    }
    path_rows = runtime["sys_path"]
    _require(type(path_rows) is list and len(path_rows) == 6, f"{context}_path_rejected")
    observed_paths: set[str] = set()
    for item in path_rows:
        _require(type(item) is dict, f"{context}_path_row_rejected")
        absolute = item.get("absolute_path")
        keys = {"absolute_path", "path_sha256", "kind"}
        if absolute == str(python_root / "python313.zip"):
            keys.add("content_sha256")
        row = _exact(item, keys, f"{context}_path_row")
        _require(
            absolute in expected_paths
            and absolute not in observed_paths
            and row["path_sha256"] == _sha(str(absolute).encode("utf-8"))
            and row["kind"] == ("file" if "content_sha256" in row else "directory")
            and ("content_sha256" not in row or row["content_sha256"] == PYTHON_ZIP_SHA256),
            f"{context}_path_row_rejected",
        )
        observed_paths.add(absolute)
    _require(
        observed_paths == expected_paths
        and runtime["sys_path_sha256"] == _sha(_canonical(path_rows))
        and runtime["sys_path_first"] in expected_paths,
        f"{context}_path_inventory_rejected",
    )
    prefixes = _exact(runtime["prefixes"], {"prefix", "base_prefix"}, f"{context}_prefixes")
    for item in prefixes.values():
        row = _exact(item, {"absolute_path", "path_sha256"}, f"{context}_prefix")
        _require(
            row["absolute_path"] == str(python_root)
            and row["path_sha256"] == _sha(str(python_root).encode("utf-8")),
            f"{context}_prefix_rejected",
        )
    site_roots = runtime["site_package_roots"]
    expected_site_row = {
        "absolute_path": str(site_root),
        "path_sha256": _sha(str(site_root).encode("utf-8")),
    }
    _require(site_roots == [expected_site_row], f"{context}_site_roots_rejected")
    expected_modules = {
        "_cffi_backend": "cffi",
        "cffi": "cffi",
        "cryptography": "cryptography",
        "pycparser": "pycparser",
        "win32api": "pywin32",
        "win32security": "pywin32",
    }
    resolutions = _exact(runtime["module_resolutions"], set(expected_modules), f"{context}_modules")
    for module, distribution in expected_modules.items():
        row = _exact(
            resolutions[module],
            {
                "module",
                "distribution",
                "origin",
                "origin_sha256",
                "search_roots",
                "distribution_root",
                "distribution_root_sha256",
                "origin_present_in_hashed_distribution_inventory",
            },
            f"{context}_{module}",
        )
        origin = PureWindowsPath(_windows_absolute(row["origin"], f"{context}_{module}_origin"))
        _require(
            row["module"] == module
            and row["distribution"] == distribution
            and site_root in origin.parents
            and SHA256_RE.fullmatch(str(row["origin_sha256"])) is not None
            and type(row["search_roots"]) is list
            and all(
                site_root == PureWindowsPath(item) or site_root in PureWindowsPath(item).parents
                for item in row["search_roots"]
            )
            and row["distribution_root"] == str(site_root)
            and row["distribution_root_sha256"] == _sha(str(site_root).encode("utf-8"))
            and row["origin_present_in_hashed_distribution_inventory"] is True,
            f"{context}_{module}_rejected",
        )
    return runtime


def _validate_inventory(
    value: Any,
    *,
    expected_phase: str,
    expected_head_sha: str,
    expected_ref: str,
    secure_launch: Mapping[str, Any],
    context: str,
) -> dict[str, Any]:
    inventory = _exact(
        value,
        {
            "schema_version",
            "kind",
            "python_implementation",
            "interpreter_runtime",
            "executables",
            "dependencies",
            "repository",
        },
        context,
    )
    _require(
        inventory["schema_version"] == 1
        and inventory["kind"] == "explainiverse-lambda-operator-inventory"
        and inventory["python_implementation"] == "CPython",
        f"{context}_rejected",
    )
    preloader = secure_launch["preloader"]
    python_install = preloader["python_install_receipt"]
    site_install = preloader["site_install_receipt"]
    dependencies, revalidation = _validate_dependency_inventory(
        inventory["dependencies"],
        repository_root=inventory["repository"]["absolute_root"],
        python_install=python_install,
        site_install=site_install,
        context=f"{context}_dependencies",
    )
    _validate_interpreter_runtime(
        inventory["interpreter_runtime"],
        repository_root=inventory["repository"]["absolute_root"],
        python_install=python_install,
        site_install=site_install,
        expected_revalidation=revalidation,
        context=f"{context}_interpreter",
    )
    _require(
        dependencies["wheel_derived_site_manifest"]["python_tree"]
        == preloader["bootstrap"]["python_tree"]
        and dependencies["wheel_derived_site_manifest"]["site_tree"]
        == preloader["bootstrap"]["site_tree"]
        and inventory["repository"]["head_sha"] == expected_head_sha
        and inventory["repository"]["supplied_ref"] == expected_ref
        and expected_ref == PHASE_REFS[expected_phase],
        f"{context}_cross_binding_rejected",
    )
    return inventory


def _validate_expected_immutable_plan(
    value: Any,
    *,
    expected_head_sha: str,
    expected_lifecycle_nonce: str,
    expected_plan_sha256: str,
    expected_runtime_bundle_sha256: str,
    context: str,
) -> dict[str, Any]:
    plan = _exact(
        value,
        {
            "schema_version",
            "kind",
            "openapi",
            "repository",
            "head_sha",
            "lifecycle_nonce",
            "created_at_unix",
            "expires_at_unix",
            "controller_source",
            "target",
            "ssh_access",
            "remote_runtime",
            "baseline_file_systems_sha256",
            "original_global_rules",
            "desired_global_and_instance_rules",
            "ownership_tags",
            "mutation_order",
            "secret_transport",
            "production_authorized",
            "provider_mutation_authorized",
            "live_go",
        },
        context,
    )
    _require(
        _sha(_live_canonical(plan)) == expected_plan_sha256
        and plan["schema_version"] == 1
        and type(plan["schema_version"]) is int
        and plan["kind"] == "explainiverse-lambda-live-plan"
        and plan["openapi"]
        == {
            "openapi_version": "3.1.0",
            "api_version": "1.10.0",
            "document_sha256": "2e00f2884d043fa2377a1a6f898eba4b81d8b0c4546d5d98079c7faa4451ba8f",
            "production_origin": "https://cloud.lambda.ai",
        }
        and plan["repository"] == REPOSITORY
        and plan["head_sha"] == expected_head_sha
        and plan["lifecycle_nonce"] == expected_lifecycle_nonce
        and type(plan["created_at_unix"]) is int
        and type(plan["expires_at_unix"]) is int
        and 0 < plan["expires_at_unix"] - plan["created_at_unix"] <= 4 * 60 * 60
        and plan["production_authorized"] is False
        and plan["provider_mutation_authorized"] is False
        and plan["live_go"] is False,
        f"{context}_rejected",
    )
    try:
        controller_network = ipaddress.ip_network(plan["controller_source"], strict=True)
    except (TypeError, ValueError):
        _fail(f"{context}_controller_source_rejected")
    _require(
        type(controller_network) is ipaddress.IPv4Network
        and controller_network.prefixlen == 32
        and controller_network.network_address.is_global,
        f"{context}_controller_source_rejected",
    )
    target = _exact(
        plan["target"],
        {
            "instance_type_name",
            "instance_type_description",
            "gpu_description",
            "physical_gpu_count",
            "architecture",
            "price_cents_per_hour",
            "vcpus",
            "memory_gib",
            "storage_gib",
            "region_name",
            "region_description",
            "image",
        },
        f"{context}_target",
    )
    image = _exact(
        target["image"],
        {
            "id",
            "created_time",
            "description",
            "name",
            "family",
            "version",
            "updated_time",
            "architecture",
            "region_name",
        },
        f"{context}_image",
    )
    normalized_gpu = re.sub(r"[^A-Z0-9]+", " ", str(target["gpu_description"]).upper()).split()
    _require(
        target["instance_type_name"] == "gpu_8x_a100_80gb_sxm4"
        and bool(_text(target["instance_type_description"], f"{context}_description"))
        and "A100" in normalized_gpu
        and "80" in normalized_gpu
        and "GB" in normalized_gpu
        and "SXM4" in normalized_gpu
        and "H100" not in normalized_gpu
        and target["physical_gpu_count"] == 8
        and type(target["physical_gpu_count"]) is int
        and target["architecture"] == "x86_64"
        and all(
            type(target[name]) is int and target[name] > 0
            for name in (
                "price_cents_per_hour",
                "vcpus",
                "memory_gib",
                "storage_gib",
            )
        )
        and target["region_name"] == TARGET_REGION
        and target["region_description"] == TARGET_REGION_DESCRIPTION
        and bool(_text(image["id"], f"{context}_image_id"))
        and _live_time(image["created_time"], f"{context}_image_created")
        <= _live_time(image["updated_time"], f"{context}_image_updated")
        and type(image["description"]) is str
        and bool(_text(image["name"], f"{context}_image_name"))
        and image["family"] == "lambda-stack-22-04"
        and bool(_text(image["version"], f"{context}_image_version"))
        and image["architecture"] == "x86_64"
        and image["region_name"] == TARGET_REGION,
        f"{context}_target_rejected",
    )
    ssh = _exact(
        plan["ssh_access"],
        {"key_name", "public_key_sha256", "ephemeral_host_key_fingerprint"},
        f"{context}_ssh",
    )
    _require(
        bool(_text(ssh["key_name"], f"{context}_ssh_key"))
        and SHA256_RE.fullmatch(str(ssh["public_key_sha256"])) is not None
        and SSH_FINGERPRINT_RE.fullmatch(str(ssh["ephemeral_host_key_fingerprint"])) is not None,
        f"{context}_ssh_rejected",
    )
    remote = _exact(
        plan["remote_runtime"],
        {
            "bundle_sha256",
            "bundle_files",
            "install_root",
            "fixed_cloud_init_wait_command",
            "fixed_preflight_command",
            "fixed_command",
            "fixed_command_contains_dynamic_or_plan_values",
        },
        f"{context}_remote_runtime",
    )
    _require(
        remote["bundle_sha256"] == expected_runtime_bundle_sha256
        and remote["bundle_files"] == list(RUNTIME_BUNDLE_NAMES)
        and remote["install_root"] == REMOTE_RUNTIME_ROOT
        and remote["fixed_cloud_init_wait_command"] == list(FIXED_CLOUD_INIT_WAIT_COMMAND)
        and remote["fixed_preflight_command"] == list(FIXED_PREFLIGHT_COMMAND)
        and remote["fixed_command"] == list(FIXED_REMOTE_COMMAND)
        and remote["fixed_command_contains_dynamic_or_plan_values"] is False,
        f"{context}_remote_runtime_rejected",
    )
    _digest(plan["baseline_file_systems_sha256"], f"{context}_file_systems_rejected")
    _require(
        type(plan["original_global_rules"]) is list
        and bool(plan["original_global_rules"])
        and plan["desired_global_and_instance_rules"]
        == [
            {
                "protocol": "tcp",
                "port_range": [22, 22],
                "source_network": plan["controller_source"],
                "description": f"Explainiverse {expected_lifecycle_nonce} controller SSH",
            }
        ]
        and plan["ownership_tags"]
        == [
            {"key": "explainiverse-lifecycle-nonce", "value": expected_lifecycle_nonce},
            {"key": "explainiverse-owner", "value": REPOSITORY},
            {"key": "explainiverse-purpose", "value": "stable-release-cuda"},
            {"key": "explainiverse-source-sha", "value": expected_head_sha},
        ]
        and plan["mutation_order"] == list(MUTATION_ORDER)
        and plan["secret_transport"]
        == {
            "lambda_api_key": "anonymous-fd-or-stdin-only",
            "github_jit_config": "anonymous-fd-or-stdin-only",
            "host_private_key": "in-memory-cloud-init-only",
        },
        f"{context}_static_contract_rejected",
    )
    return plan


def _validate_discovery(
    value: Any, *, expected_plan: Mapping[str, Any], context: str
) -> dict[str, Any]:
    discovery = _exact(
        value,
        {
            "snapshot_sha256",
            "binding_sha256",
            "payload_digests",
            "response_bindings",
            "zero_instances",
            "zero_firewall_rulesets",
            "target",
            "image_candidates",
            "ssh_access",
            "baseline_file_systems_sha256",
            "original_global_rules",
        },
        context,
    )
    _digest(discovery["snapshot_sha256"], f"{context}_snapshot")
    _digest(discovery["binding_sha256"], f"{context}_binding")
    _digest(discovery["baseline_file_systems_sha256"], f"{context}_file_systems")
    _require(
        discovery["zero_instances"] is True and discovery["zero_firewall_rulesets"] is True,
        f"{context}_zero_inventory_rejected",
    )
    operations = (
        ("instances", "/api/v1/instances"),
        ("file_systems", "/api/v1/file-systems"),
        ("ssh_keys", "/api/v1/ssh-keys"),
        ("instance_types", "/api/v1/instance-types"),
        ("images", "/api/v1/images"),
        ("regions", "/api/v1/regions"),
        ("global_firewall", "/api/v1/firewall-rulesets/global"),
        ("firewall_rulesets", "/api/v1/firewall-rulesets"),
    )
    payload_digests = _exact(
        discovery["payload_digests"],
        {operation for operation, _ in operations},
        f"{context}_payload_digests",
    )
    for digest in payload_digests.values():
        _digest(digest, f"{context}_payload_digest_rejected")
    bindings = discovery["response_bindings"]
    _require(
        type(bindings) is list and len(bindings) == len(operations),
        f"{context}_response_bindings_rejected",
    )
    for item, (operation, path) in zip(bindings, operations, strict=True):
        binding = _exact(
            item,
            {
                "operation",
                "method",
                "path",
                "request_sha256",
                "request_body_sha256",
                "response_body_sha256",
                "status_code",
                "content_type",
            },
            f"{context}_response_binding",
        )
        _require(
            binding["operation"] == operation
            and binding["method"] == "GET"
            and binding["path"] == path
            and binding["request_body_sha256"] is None
            and binding["status_code"] == 200
            and type(binding["status_code"]) is int
            and binding["content_type"] == "application/json",
            f"{context}_response_binding_rejected",
        )
        _digest(binding["request_sha256"], f"{context}_request_sha")
        _digest(binding["response_body_sha256"], f"{context}_response_sha")
    _require(
        discovery["snapshot_sha256"]
        == _sha(
            _live_canonical(
                {
                    "payload_digests": payload_digests,
                    "bindings": bindings,
                }
            )
        ),
        f"{context}_snapshot_binding_rejected",
    )
    target = _exact(
        discovery["target"],
        {
            "instance_type_name",
            "instance_type_description",
            "gpu_description",
            "price_cents_per_hour",
            "vcpus",
            "memory_gib",
            "storage_gib",
            "gpus",
            "architecture",
            "capacity_region",
            "region_description",
        },
        f"{context}_target",
    )
    normalized_gpu = re.sub(r"[^A-Z0-9]+", " ", str(target["gpu_description"]).upper()).split()
    _require(
        target["instance_type_name"] == "gpu_8x_a100_80gb_sxm4"
        and bool(_text(target["instance_type_description"], f"{context}_description"))
        and "A100" in normalized_gpu
        and "80" in normalized_gpu
        and "GB" in normalized_gpu
        and "SXM4" in normalized_gpu
        and "H100" not in normalized_gpu
        and type(target["price_cents_per_hour"]) is int
        and target["price_cents_per_hour"] > 0
        and all(
            type(target[name]) is int and target[name] > 0
            for name in ("vcpus", "memory_gib", "storage_gib")
        )
        and target["gpus"] == 8
        and type(target["gpus"]) is int
        and target["architecture"] == "x86_64"
        and target["capacity_region"] == TARGET_REGION
        and target["region_description"] == TARGET_REGION_DESCRIPTION,
        f"{context}_target_rejected",
    )
    images = discovery["image_candidates"]
    _require(type(images) is list and bool(images), f"{context}_images_rejected")
    image_ids: set[str] = set()
    for value_image in images:
        image = _exact(
            value_image,
            {
                "id",
                "created_time",
                "updated_time",
                "name",
                "description",
                "family",
                "version",
                "architecture",
                "region",
            },
            f"{context}_image",
        )
        image_id = _text(image["id"], f"{context}_image_id")
        created = _live_time(image["created_time"], f"{context}_image_created")
        updated = _live_time(image["updated_time"], f"{context}_image_updated")
        region = _exact(image["region"], {"name", "description"}, f"{context}_image_region")
        _require(
            image_id not in image_ids
            and created <= updated
            and bool(_text(image["name"], f"{context}_image_name"))
            and type(image["description"]) is str
            and image["family"] == "lambda-stack-22-04"
            and bool(_text(image["version"], f"{context}_image_version"))
            and image["architecture"] == "x86_64"
            and region == {"name": TARGET_REGION, "description": TARGET_REGION_DESCRIPTION},
            f"{context}_image_rejected",
        )
        image_ids.add(image_id)
    ssh = _exact(discovery["ssh_access"], {"key_name", "public_key_sha256"}, f"{context}_ssh")
    _text(ssh["key_name"], f"{context}_ssh_key")
    _digest(ssh["public_key_sha256"], f"{context}_ssh_digest")
    rules = discovery["original_global_rules"]
    _require(type(rules) is list, f"{context}_rules_rejected")
    for value_rule in rules:
        _require(type(value_rule) is dict, f"{context}_rule_rejected")
        protocol = value_rule.get("protocol")
        expected_keys = {"protocol", "source_network", "description"}
        if protocol != "icmp":
            expected_keys.add("port_range")
        rule = _exact(value_rule, expected_keys, f"{context}_rule")
        try:
            network = ipaddress.ip_network(rule["source_network"], strict=True)
        except (TypeError, ValueError):
            _fail(f"{context}_rule_network_rejected")
        _require(
            protocol in {"tcp", "udp", "icmp"}
            and type(network) is ipaddress.IPv4Network
            and type(rule["description"]) is str
            and len(rule["description"]) <= 128,
            f"{context}_rule_rejected",
        )
        if protocol != "icmp":
            ports = rule["port_range"]
            _require(
                type(ports) is list
                and len(ports) == 2
                and all(type(item) is int for item in ports)
                and 1 <= ports[0] <= ports[1] <= 65535,
                f"{context}_rule_ports_rejected",
            )
    binding_material = {
        "snapshot_sha256": discovery["snapshot_sha256"],
        "region_description": target["region_description"],
        "instance_type_description": target["instance_type_description"],
        "gpu_description": target["gpu_description"],
        "price_cents_per_hour": target["price_cents_per_hour"],
        "vcpus": target["vcpus"],
        "memory_gib": target["memory_gib"],
        "storage_gib": target["storage_gib"],
        "images": images,
        "ssh_key_name": ssh["key_name"],
        "ssh_public_key_sha256": ssh["public_key_sha256"],
        "baseline_file_systems_sha256": discovery["baseline_file_systems_sha256"],
        "original_global_rules": rules,
    }
    plan_target = expected_plan["target"]
    plan_image = plan_target["image"]
    expected_selected_image = {
        "id": plan_image["id"],
        "created_time": plan_image["created_time"],
        "updated_time": plan_image["updated_time"],
        "name": plan_image["name"],
        "description": plan_image["description"],
        "family": plan_image["family"],
        "version": plan_image["version"],
        "architecture": plan_image["architecture"],
        "region": {
            "name": plan_image["region_name"],
            "description": plan_target["region_description"],
        },
    }
    _require(
        discovery["binding_sha256"] == _sha(_live_canonical(binding_material))
        and target["instance_type_name"] == plan_target["instance_type_name"]
        and target["instance_type_description"] == plan_target["instance_type_description"]
        and target["gpu_description"] == plan_target["gpu_description"]
        and target["price_cents_per_hour"] == plan_target["price_cents_per_hour"]
        and target["vcpus"] == plan_target["vcpus"]
        and target["memory_gib"] == plan_target["memory_gib"]
        and target["storage_gib"] == plan_target["storage_gib"]
        and target["gpus"] == plan_target["physical_gpu_count"]
        and target["architecture"] == plan_target["architecture"]
        and target["capacity_region"] == plan_target["region_name"]
        and target["region_description"] == plan_target["region_description"]
        and expected_selected_image in images
        and ssh["key_name"] == expected_plan["ssh_access"]["key_name"]
        and ssh["public_key_sha256"] == expected_plan["ssh_access"]["public_key_sha256"]
        and discovery["baseline_file_systems_sha256"]
        == expected_plan["baseline_file_systems_sha256"]
        and rules == expected_plan["original_global_rules"],
        f"{context}_plan_binding_rejected",
    )
    return discovery


def _validate_app_capture_inbox(value: Any, *, expected_phase: str, context: str) -> dict[str, Any]:
    inbox = _exact(
        value,
        {
            "phase",
            "expected_capture_count",
            "accepted_capture_count",
            "stale_generation_count",
            "stale_generations_sha256",
            "owner_private_directory_receipt_sha256",
            "on_demand_before_each_jit",
            "ready_marker_no_replace_required",
            "raw_pages_archived_by_driver",
        },
        context,
    )
    _require(
        inbox["phase"] == expected_phase
        and inbox["expected_capture_count"] == PHASE_CAPTURE_COUNTS[expected_phase]
        and type(inbox["expected_capture_count"]) is int
        and inbox["accepted_capture_count"] == 0
        and type(inbox["accepted_capture_count"]) is int
        and inbox["stale_generation_count"] == 0
        and type(inbox["stale_generation_count"]) is int
        and inbox["stale_generations_sha256"] == _sha(_canonical([]))
        and inbox["on_demand_before_each_jit"] is True
        and inbox["ready_marker_no_replace_required"] is True
        and inbox["raw_pages_archived_by_driver"] is True,
        f"{context}_rejected",
    )
    _digest(inbox["owner_private_directory_receipt_sha256"], f"{context}_receipt_sha")
    return inbox


def _validate_final_main_acceptance(
    value: Any, *, expected_phase: str, expected_head_sha: str, context: str
) -> dict[str, Any] | None:
    if expected_phase != "publication":
        _require(value is None, f"{context}_unexpected")
        return None
    acceptance = _exact(
        value,
        {"loader_verified", "evidence_sha256", "head_sha", "run_id"},
        context,
    )
    _require(
        acceptance["loader_verified"] is True
        and acceptance["head_sha"] == expected_head_sha
        and type(acceptance["run_id"]) is int
        and acceptance["run_id"] > 0,
        f"{context}_rejected",
    )
    _digest(acceptance["evidence_sha256"], f"{context}_digest")
    return acceptance


def validate_operator_preflight(
    value: Mapping[str, Any],
    *,
    expected_immutable_plan: Mapping[str, Any],
    expected_phase: str,
    expected_head_sha: str,
    expected_ref: str,
    expected_plan_sha256: str,
    expected_lifecycle_nonce: str,
    expected_inspection_receipt_sha256: str,
    expected_inventory_sha256: str,
    expected_policy_sha256: str,
    expected_controller_source_sha256: str,
    expected_runtime_bundle_sha256: str,
) -> dict[str, Any]:
    """Validate one archived operator preflight without consulting live state.

    The returned identity is a normalized, non-authoritative summary of the
    already verified receipt.  All authority remains in the exact archived
    bytes and the explicit expected values supplied by the caller.
    """

    _require(expected_phase in PHASE_REFS, "operator_preflight_phase_rejected")
    _require(
        expected_ref == PHASE_REFS[expected_phase],
        "operator_preflight_ref_rejected",
    )
    _git_sha(expected_head_sha, "operator_preflight_head_rejected")
    _digest(expected_plan_sha256, "operator_preflight_plan_rejected")
    _require(
        type(expected_lifecycle_nonce) is str
        and NONCE_RE.fullmatch(expected_lifecycle_nonce) is not None,
        "operator_preflight_lifecycle_nonce_rejected",
    )
    for name, digest in (
        ("inspection", expected_inspection_receipt_sha256),
        ("inventory", expected_inventory_sha256),
        ("policy", expected_policy_sha256),
        ("controller", expected_controller_source_sha256),
        ("runtime_bundle", expected_runtime_bundle_sha256),
    ):
        _digest(digest, f"operator_preflight_expected_{name}_sha256_rejected")
    immutable_plan = _validate_expected_immutable_plan(
        _normalized(expected_immutable_plan, context="operator_preflight_expected_plan"),
        expected_head_sha=expected_head_sha,
        expected_lifecycle_nonce=expected_lifecycle_nonce,
        expected_plan_sha256=expected_plan_sha256,
        expected_runtime_bundle_sha256=expected_runtime_bundle_sha256,
        context="operator_preflight_expected_plan",
    )
    normalized = _normalized(value, context="operator_preflight")
    _require(normalized == value, "operator_preflight_noncanonical_value_rejected")
    preflight = _exact(
        normalized,
        {
            "schema_version",
            "kind",
            "plan_sha256",
            "head_sha",
            "lifecycle_nonce",
            "discovery",
            "inspection_receipt_sha256",
            "inventory",
            "inventory_sha256",
            "executables",
            "repository",
            "environment",
            "secure_launch",
            "lambda_secret_transport",
            "plan_confirmation",
            "app_capture_inbox",
            "final_main_acceptance",
            "live_gates_not_constructed_before_confirmation",
            "direct_publication_dispatch_exposed",
        },
        "operator_preflight",
    )
    _require(
        preflight["schema_version"] == 1
        and preflight["kind"] == "explainiverse-lambda-operator-preflight"
        and preflight["plan_sha256"] == expected_plan_sha256
        and preflight["head_sha"] == expected_head_sha
        and preflight["lifecycle_nonce"] == expected_lifecycle_nonce
        and preflight["inspection_receipt_sha256"] == expected_inspection_receipt_sha256
        and preflight["live_gates_not_constructed_before_confirmation"] is True
        and preflight["direct_publication_dispatch_exposed"] is False,
        "operator_preflight_binding_rejected",
    )
    environment = _validate_environment(preflight["environment"], "operator_preflight_environment")
    inventory = _exact(
        preflight["inventory"],
        {
            "schema_version",
            "kind",
            "python_implementation",
            "interpreter_runtime",
            "executables",
            "dependencies",
            "repository",
        },
        "operator_preflight_inventory",
    )
    _require(
        preflight["inventory_sha256"] == expected_inventory_sha256
        and expected_inventory_sha256 == _sha(_canonical(inventory)),
        "operator_preflight_inventory_digest_rejected",
    )
    executables = _validate_executables(inventory["executables"], "operator_preflight_executables")
    _require(
        preflight["executables"] == executables
        and preflight["repository"] == inventory["repository"],
        "operator_preflight_inventory_copy_rejected",
    )
    secure_launch = _validate_secure_launch(
        preflight["secure_launch"],
        expected_phase=expected_phase,
        expected_head_sha=expected_head_sha,
        expected_ref=expected_ref,
        expected_policy_sha256=expected_policy_sha256,
        expected_controller_source_sha256=expected_controller_source_sha256,
        expected_runtime_bundle_sha256=expected_runtime_bundle_sha256,
        environment=environment,
        inventory_executables=executables,
        inventory_repository=inventory["repository"],
        context="operator_preflight_secure_launch",
    )
    _validate_inventory(
        inventory,
        expected_phase=expected_phase,
        expected_head_sha=expected_head_sha,
        expected_ref=expected_ref,
        secure_launch=secure_launch,
        context="operator_preflight_inventory",
    )
    _validate_discovery(
        preflight["discovery"],
        expected_plan=immutable_plan,
        context="operator_preflight_discovery",
    )
    _validate_anonymous_transport(
        preflight["lambda_secret_transport"],
        expected_plan_sha256=None,
        context="operator_preflight_lambda_secret_transport",
    )
    _validate_anonymous_transport(
        preflight["plan_confirmation"],
        expected_plan_sha256=expected_plan_sha256,
        context="operator_preflight_plan_confirmation",
    )
    inbox = _validate_app_capture_inbox(
        preflight["app_capture_inbox"],
        expected_phase=expected_phase,
        context="operator_preflight_app_capture_inbox",
    )
    acceptance = _validate_final_main_acceptance(
        preflight["final_main_acceptance"],
        expected_phase=expected_phase,
        expected_head_sha=expected_head_sha,
        context="operator_preflight_final_main_acceptance",
    )
    source = secure_launch["preloader"]["source"]
    sealed = secure_launch["preloader"]["sealed_resources"]
    return {
        "schema_version": 1,
        "kind": "explainiverse-operator-preflight-validated-identity",
        "preflight_sha256": _sha(_canonical(preflight)),
        "phase": expected_phase,
        "head_sha": expected_head_sha,
        "supplied_ref": expected_ref,
        "plan_sha256": expected_plan_sha256,
        "lifecycle_nonce": expected_lifecycle_nonce,
        "inspection_receipt_sha256": expected_inspection_receipt_sha256,
        "inventory_sha256": expected_inventory_sha256,
        "repository_tree_object_sha": inventory["repository"]["tree_object_sha"],
        "repository_tree_inventory_sha256": inventory["repository"]["tree_inventory_sha256"],
        "source_manifest_sha256": source["source_manifest_sha256"],
        "source_manifest_inventory_sha256": source["source_manifest_inventory_sha256"],
        "preloader_sha256": source["preloader_sha256"],
        "shim_sha256": PRELOADER_SHIM_SHA256,
        "policy_sha256": sealed["policy_sha256"],
        "controller_source_sha256": sealed["controller_source_sha256"],
        "runtime_bundle_sha256": sealed["runtime_bundle_sha256"],
        "app_capture_inbox_receipt_sha256": inbox["owner_private_directory_receipt_sha256"],
        "final_main_acceptance_evidence_sha256": (
            acceptance["evidence_sha256"] if acceptance is not None else None
        ),
        "parent_provenance_authenticated": False,
        "security_authority_derived_from_parent_declaration": False,
    }


__all__ = [
    "OperatorReceiptContractError",
    "validate_operator_preflight",
]
