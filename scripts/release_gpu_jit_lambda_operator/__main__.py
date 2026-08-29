"""Minimal pre-import guard for the production operator entrypoint."""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence

PARENT_RECEIPT_KIND = "explainiverse-windows-launcher-parent-boundary"
PRELOADER_MARKER_NAME = "_EXPLAINIVERSE_OPERATOR_PRELOADER_RECEIPT"
RESOURCE_MARKER_NAME = "_EXPLAINIVERSE_OPERATOR_CAPTURED_RESOURCES"
RUNTIME_BUNDLE_NAMES = ("__init__.py", "bootstrap.py", "executor.py", "runtime_contract.py")


def _die(code: str) -> NoReturn:
    payload = {
        "schema_version": 1,
        "kind": "explainiverse-lambda-operator-error",
        "exception_type": "SecureLaunchError",
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


def _repository_argument(argv: Sequence[str]) -> Path:
    values: list[str] = []
    for index, item in enumerate(argv):
        if item == "--repository-root":
            if index + 1 >= len(argv):
                _die("secure_launch_repository_root_missing")
            values.append(argv[index + 1])
        elif item.startswith("--repository-root="):
            values.append(item.split("=", 1)[1])
    if len(values) != 1:
        _die("secure_launch_repository_root_cardinality")
    candidate = Path(values[0])
    if not candidate.is_absolute():
        _die("secure_launch_repository_root_not_absolute")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        _die("secure_launch_repository_root_unavailable")
    if candidate != resolved or not candidate.is_dir() or candidate.is_symlink():
        _die("secure_launch_repository_root_rejected")
    return candidate


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


def _require_environment_scrubbed() -> None:
    if any(_forbidden_environment_name(name) for name in os.environ):
        _die("secure_launch_environment_not_scrubbed")


def _consume_preloader_receipt(root: Path) -> dict[str, Any]:
    value = getattr(builtins, PRELOADER_MARKER_NAME, None)
    if type(value) is not dict:
        _die("secure_launch_preloader_receipt_missing")
    material = dict(value)
    evidence_sha256 = material.pop("evidence_sha256", None)
    if not (
        type(evidence_sha256) is str
        and len(evidence_sha256) == 64
        and hashlib.sha256(
            (
                json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
                + "\n"
            ).encode("ascii")
        ).hexdigest()
        == evidence_sha256
        and value.get("schema_version") == 1
        and value.get("kind") == "explainiverse-operator-isolated-preloader"
        and value.get("isolated") is True
        and value.get("safe_path") is True
        and value.get("site_disabled") is True
        and value.get("bytecode_disabled") is True
        and value.get("repository_absent_from_sys_path") is True
        and value.get("project_imports_from_captured_bytes") is True
        and type(value.get("source")) is dict
        and value["source"].get("repository_root") == str(root)
        and value["source"].get("tracked_and_untracked_clean") is True
        and value["source"].get("project_modules_execute_from_captured_bytes") is True
        and type(value.get("bootstrap")) is dict
        and value["bootstrap"].get("python_archive_sha256")
        == "d1f04d990aee1253d8569e8e5104e30fa9f5fa830899f14843448872d936a2cf"
        and type(value.get("environment")) is dict
        and value["environment"].get("ambient_credentials_retained") is False
        and value["environment"].get("ambient_proxies_retained") is False
    ):
        _die("secure_launch_preloader_receipt_rejected")
    delattr(builtins, PRELOADER_MARKER_NAME)
    return dict(value)


def _consume_captured_resources(preloader_receipt: Mapping[str, Any]) -> dict[str, Any]:
    value = getattr(builtins, RESOURCE_MARKER_NAME, None)
    if type(value) is not dict:
        _die("secure_launch_captured_resources_missing")
    binding = preloader_receipt.get("sealed_resources")
    runtime_files = value.get("runtime_files")
    valid = (
        set(value)
        == {
            "schema_version",
            "kind",
            "preloader_evidence_sha256",
            "policy_bytes",
            "controller_source_bytes",
            "runtime_files",
        }
        and value.get("schema_version") == 1
        and value.get("kind") == "explainiverse-operator-captured-resources"
        and value.get("preloader_evidence_sha256") == preloader_receipt.get("evidence_sha256")
        and type(binding) is dict
        and type(value.get("policy_bytes")) is bytes
        and hashlib.sha256(value["policy_bytes"]).hexdigest() == binding.get("policy_sha256")
        and type(value.get("controller_source_bytes")) is bytes
        and hashlib.sha256(value["controller_source_bytes"]).hexdigest()
        == binding.get("controller_source_sha256")
        and type(runtime_files) is dict
        and set(runtime_files) == set(RUNTIME_BUNDLE_NAMES)
        and all(type(runtime_files[name]) is bytes for name in RUNTIME_BUNDLE_NAMES)
        and {
            name: hashlib.sha256(runtime_files[name]).hexdigest() for name in sorted(runtime_files)
        }
        == binding.get("runtime_file_sha256")
        and binding.get("captured_before_project_import") is True
        and binding.get("live_repository_reopen_permitted") is False
    )
    if not valid:
        _die("secure_launch_captured_resources_rejected")
    delattr(builtins, RESOURCE_MARKER_NAME)
    return dict(value)


def _secure_launch(
    *,
    root: Path,
    preloader_receipt: dict[str, Any],
    environment_receipt: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not (
        sys.flags.isolated == 1
        and sys.flags.safe_path
        and sys.flags.ignore_environment == 1
        and sys.flags.no_user_site == 1
        and sys.flags.no_site == 1
        and sys.dont_write_bytecode
    ):
        _die("secure_launch_requires_pinned_python_I_S_B")
    try:
        working = Path.cwd().resolve(strict=True)
    except OSError:
        _die("secure_launch_import_root_unavailable")
    if (
        preloader_receipt.get("working_directory") != str(working)
        or preloader_receipt.get("working_directory_is_python_install_receipt_directory")
        is not True
        or working == root
        or root in working.parents
    ):
        _die("secure_launch_import_root_mismatch")
    normalized_path: list[str] = []
    for item in sys.path:
        try:
            resolved = Path(item or os.curdir).resolve(strict=False)
        except OSError:
            _die("secure_launch_sys_path_rejected")
        normalized_path.append(str(resolved))
    if any(Path(item) == root or root in Path(item).parents for item in normalized_path):
        _die("secure_launch_repository_present_in_sys_path")
    path_payload = "\n".join(normalized_path).encode("utf-8")
    launch = {
        "schema_version": 1,
        "kind": "operator-secure-interpreter-launch",
        "isolated": True,
        "safe_path": True,
        "ignore_environment": True,
        "no_user_site": True,
        "no_site": True,
        "dont_write_bytecode": True,
        "invocation": "pinned-python -I -S -B -c <byte-sealing-shim>",
        "working_directory": str(working),
        "repository_absent_from_sys_path": True,
        "sys_path_sha256": hashlib.sha256(path_payload).hexdigest(),
        "site_processing_disabled": True,
        "preloader": preloader_receipt,
        "controller_imported_after_launch_validation": True,
    }
    return launch, environment_receipt


def _option_values(argv: Sequence[str], option: str) -> list[str]:
    values: list[str] = []
    for index, item in enumerate(argv):
        if item == option:
            if index + 1 >= len(argv):
                _die("windows_handle_option_missing_value")
            values.append(argv[index + 1])
        elif item.startswith(option + "="):
            _die("windows_handle_option_form_rejected")
    return values


def _action(argv: Sequence[str]) -> str:
    values = _option_values(argv, "--action")
    return values[0] if len(values) == 1 else "inspect"


def _extract_windows_launcher_parent_receipt(
    argv: Sequence[str],
) -> tuple[list[str], dict[str, Any] | None]:
    values = _option_values(argv, "--windows-launcher-parent-receipt")
    if not values:
        return list(argv), None
    if len(values) != 1:
        _die("windows_launcher_parent_receipt_cardinality")
    raw = values[0]
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError):
        _die("windows_launcher_parent_receipt_json_rejected")
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    if type(value) is not dict or canonical != raw:
        _die("windows_launcher_parent_receipt_not_canonical")
    required = {
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
        "preloader_binding",
        "environment",
        "parent_declares_secret_read_after_boundary_validation",
        "parent_provenance_authenticated",
        "child_must_revalidate_all_security_boundaries",
        "evidence_sha256",
    }
    material = dict(value)
    evidence_sha256 = material.pop("evidence_sha256", None)
    if not (
        set(value) == required
        and value.get("schema_version") == 1
        and value.get("kind") == PARENT_RECEIPT_KIND
        and value.get("isolated") is True
        and value.get("safe_path") is True
        and value.get("ignore_environment") is True
        and value.get("no_user_site") is True
        and value.get("no_site") is True
        and value.get("dont_write_bytecode") is True
        and value.get("repository_absent_from_sys_path") is True
        and value.get("site_processing_disabled") is True
        and type(value.get("preloader_binding")) is dict
        and value.get("parent_declares_secret_read_after_boundary_validation") is True
        and value.get("parent_provenance_authenticated") is False
        and value.get("child_must_revalidate_all_security_boundaries") is True
        and type(value.get("environment")) is dict
        and type(evidence_sha256) is str
        and hashlib.sha256(
            (
                json.dumps(
                    material,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
                + "\n"
            ).encode("ascii")
        ).hexdigest()
        == evidence_sha256
    ):
        _die("windows_launcher_parent_receipt_binding_rejected")
    rewritten: list[str] = []
    skip = 0
    for item in argv:
        if skip:
            skip -= 1
            continue
        if item == "--windows-launcher-parent-receipt":
            skip = 1
            continue
        rewritten.append(item)
    return rewritten, value


def _convert_windows_handles(argv: Sequence[str]) -> tuple[list[str], dict[str, Any]]:
    lambda_values = _option_values(argv, "--lambda-api-key-handle")
    confirmation_values = _option_values(argv, "--plan-confirmation-handle")
    action = _action(argv)
    expected = (
        (1, 1)
        if action in {"execute", "transport-self-test"}
        else ((1, 0) if action == "resume-abort" else (0, 0))
    )
    if (len(lambda_values), len(confirmation_values)) != expected:
        _die("windows_handle_cardinality_rejected")
    if expected == (0, 0):
        return list(argv), {
            "windows_handle_transport": False,
            "inherited_handle_count": 0,
        }
    if os.name != "nt":
        _die("windows_handle_transport_nonwindows")
    import ctypes
    import msvcrt
    from ctypes import wintypes

    raw_values = lambda_values + confirmation_values
    try:
        handles = [int(value, 10) for value in raw_values]
    except ValueError:
        _die("windows_handle_not_decimal")
    if any(value <= 0 for value in handles) or len(set(handles)) != len(handles):
        _die("windows_handle_identity_rejected")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_file_type = kernel32.GetFileType
    get_file_type.argtypes = [wintypes.HANDLE]
    get_file_type.restype = wintypes.DWORD
    opened: list[int] = []
    try:
        for handle in handles:
            if get_file_type(handle) != 3:
                _die("windows_handle_not_pipe")
            os.set_handle_inheritable(handle, False)
            descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | getattr(os, "O_BINARY", 0))
            opened.append(descriptor)
    except BaseException:
        for descriptor in opened:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise
    rewritten: list[str] = []
    skip = 0
    for item in argv:
        if skip:
            skip -= 1
            continue
        if item in {"--lambda-api-key-handle", "--plan-confirmation-handle"}:
            skip = 1
            continue
        rewritten.append(item)
    rewritten.extend(("--lambda-api-key-fd", str(opened[0])))
    if confirmation_values:
        rewritten.extend(("--plan-confirmation-fd", str(opened[1])))
    return rewritten, {
        "windows_handle_transport": True,
        "inherited_handle_count": len(opened),
        "handles_distinct": True,
        "file_type_pipe": True,
        "child_handles_made_noninheritable": True,
        "raw_handle_values_archived": False,
        "secret_values_in_argv": False,
        "secret_values_in_environment": False,
    }


_original_argv = sys.argv[1:]
_receipt_free_argv, _windows_launcher_parent_receipt = _extract_windows_launcher_parent_receipt(
    _original_argv
)
_repository_root = _repository_argument(_receipt_free_argv)
_preloader_receipt = _consume_preloader_receipt(_repository_root)
_captured_resources = _consume_captured_resources(_preloader_receipt)
_require_environment_scrubbed()
_pre_bootstrap_environment_receipt = dict(_preloader_receipt["environment"])
_launch_receipt, _environment_receipt = _secure_launch(
    root=_repository_root,
    preloader_receipt=_preloader_receipt,
    environment_receipt=_pre_bootstrap_environment_receipt,
)
_operator_argv, _handle_receipt = _convert_windows_handles(_receipt_free_argv)
if _windows_launcher_parent_receipt is not None and not _handle_receipt["windows_handle_transport"]:
    _die("windows_launcher_parent_receipt_without_handle_transport")
if _handle_receipt["windows_handle_transport"] and _windows_launcher_parent_receipt is None:
    _die("windows_launcher_parent_receipt_missing")
_launch_receipt.update(_handle_receipt)
if _windows_launcher_parent_receipt is not None:
    if (
        _windows_launcher_parent_receipt["preloader_binding"].get("manifest_sha256")
        != _preloader_receipt["bootstrap"]["manifest_sha256"]
        or _windows_launcher_parent_receipt["preloader_binding"].get("head_sha")
        != _preloader_receipt["source"]["head_sha"]
        or _windows_launcher_parent_receipt["preloader_binding"].get("preloader_sha256")
        != _preloader_receipt["source"]["preloader_sha256"]
    ):
        _die("windows_launcher_parent_bootstrap_binding_rejected")
    declaration_sha256 = hashlib.sha256(
        (
            json.dumps(
                _windows_launcher_parent_receipt,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            + "\n"
        ).encode("ascii")
    ).hexdigest()
    _launch_receipt["windows_launcher_parent_declaration"] = {
        "receipt_sha256": declaration_sha256,
        "preloader_metadata_matched": True,
        "parent_provenance_authenticated": False,
        "security_authority_derived_from_declaration": False,
        "child_revalidated_handle_transport_and_sealed_resources": True,
    }

# This import must remain below the launch and environment boundary.
from .cli import main  # noqa: E402

raise SystemExit(
    main(
        _operator_argv,
        environment_receipt=_environment_receipt,
        launch_receipt=_launch_receipt,
        captured_resources=_captured_resources,
    )
)
