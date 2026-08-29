"""Production trusted-local controller for Explainiverse release GPU evidence.

Only this module talks to GitHub.  The disposable Lambda host receives one
public canonical runtime plan and one short-lived encoded JIT configuration;
it never receives a GitHub API credential or a Lambda API key.  All transports
are injected and the default transports use shell-free subprocess argument
vectors.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import secrets
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, Protocol, Sequence

from scripts import release_external_controls as external_controls
from scripts.release_gpu_jit_lambda_live import adapter as live
from scripts.release_gpu_jit_lambda_runtime import runtime_contract as runtime
from scripts.verify_release_recovery import verify_source_run_evidence

REPOSITORY = "jemsbhai/explainiverse"
OWNER = "jemsbhai"
API_VERSION = runtime.GITHUB_API_VERSION
API_ACCEPT = "application/vnd.github+json"
RUNNER_GROUP_ID = 1
CHECKS_APP_ID = 15368
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_LOG_BYTES = 32 * 1024 * 1024
AUTHORITY_CAPTURE_MAX_AGE = timedelta(minutes=10)
AUTHORITY_WINDOW = timedelta(minutes=30)
OBSERVATION_MAX_AGE = timedelta(minutes=15)
RECOVERY_WORKFLOW = "recover-github-release.yml"
RECOVERY_WORKFLOW_PATH = ".github/workflows/recover-github-release.yml"
RECOVERY_RUN_TITLE_PREFIX = "explainiverse-recovery"
RECOVERY_TERMINAL_CONCLUSIONS = frozenset(
    {
        "success",
        "failure",
        "cancelled",
        "timed_out",
        "action_required",
        "neutral",
        "skipped",
        "stale",
        "startup_failure",
    }
)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
NONCE_RE = re.compile(r"[0-9a-f]{16}\Z")
RUNNER_NAME_RE = re.compile(r"explainiverse-cuda-(?:single|two)-jit-[0-9a-f]{16}\Z")
JIT_RE = re.compile(rb"[A-Za-z0-9+/]+={0,2}\Z")
GPU_RE = re.compile(
    r"GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-" r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\Z"
)

RELEASE_CONTROL_POLICY_PATH = (
    Path(__file__).resolve().parents[2] / ".github" / ("release-control-policy.json")
)

PHASES: dict[str, dict[str, Any]] = {
    "pull-request": {
        "workflow": "cuda-ci.yml",
        "workflow_path": runtime.CUDA_WORKFLOW_PATH,
        "dispatch_ref": runtime.PULL_REQUEST_REF.removeprefix("refs/heads/"),
        "run_ref": runtime.PULL_REQUEST_REF,
        "job_keys": ("single_minimum", "single_latest"),
        "queued_job_keys": ("single_minimum", "single_latest", "two_minimum", "two_latest"),
        "all_nonce_keys": runtime.CUDA_NONCE_INPUT_KEYS,
    },
    "final-main": {
        "workflow": "cuda-ci.yml",
        "workflow_path": runtime.CUDA_WORKFLOW_PATH,
        "dispatch_ref": "main",
        "run_ref": runtime.FINAL_MAIN_REF,
        "job_keys": ("single_minimum", "single_latest", "two_minimum", "two_latest"),
        "queued_job_keys": ("single_minimum", "single_latest", "two_minimum", "two_latest"),
        "all_nonce_keys": runtime.CUDA_NONCE_INPUT_KEYS,
    },
    "publication": {
        "workflow": "publish-pypi.yml",
        "workflow_path": runtime.PUBLISH_WORKFLOW_PATH,
        "dispatch_ref": runtime.PUBLICATION_TAG,
        "run_ref": runtime.PUBLICATION_REF,
        "job_keys": ("publication_single_minimum", "publication_single_latest"),
        "queued_job_keys": ("publication_single_minimum", "publication_single_latest"),
        "all_nonce_keys": runtime.PUBLICATION_NONCE_INPUT_KEYS,
    },
}

CUDA_ROUTING_JOB_NAME = "Require approved CUDA runner routing"
PUBLICATION_PREFLIGHT_JOB_NAME = "Verify the attested pre-tag external-control snapshot"
CUDA_TARGET_JOB_NAMES = frozenset(
    str(runtime.JOB_SPECS[key]["name"])
    for key in ("single_minimum", "single_latest", "two_minimum", "two_latest")
)
PUBLICATION_TARGET_JOB_NAMES = frozenset(
    str(runtime.JOB_SPECS[key]["name"])
    for key in ("publication_single_minimum", "publication_single_latest")
)
REVIEWED_HOSTED_COMPANION_JOB_NAMES = frozenset(
    {
        CUDA_ROUTING_JOB_NAME,
        PUBLICATION_PREFLIGHT_JOB_NAME,
        "Verify, build once, and inventory",
        "Attest the immutable distributions",
        "Publish through PyPI Trusted Publishing",
        "Create the immutable GitHub release",
        "Finalize the immutable GitHub release with fixed commands",
    }
)
REVIEWED_HOSTED_COMPANION_LABELS = {
    CUDA_ROUTING_JOB_NAME: "ubuntu-latest",
    PUBLICATION_PREFLIGHT_JOB_NAME: "ubuntu-latest",
    "Verify, build once, and inventory": "ubuntu-24.04",
    "Attest the immutable distributions": "ubuntu-latest",
    "Publish through PyPI Trusted Publishing": "ubuntu-latest",
    "Create the immutable GitHub release": "ubuntu-latest",
    "Finalize the immutable GitHub release with fixed commands": "ubuntu-latest",
}


@dataclass(frozen=True, init=False)
class SealedControllerResources:
    """Immutable policy/source bytes captured before untrusted repo execution.

    Production callers construct this object from the preloader's held bytes
    and independently recorded digests.  Controller/JIT authority code never
    reopens either repository path after this boundary.
    """

    policy_sha256: str
    controller_source_sha256: str
    _policy_bytes: bytes = field(repr=False)

    def __new__(cls, *_: object, **__: object) -> SealedControllerResources:
        raise TypeError("SealedControllerResources must be created from captured bytes")

    @classmethod
    def from_captured_bytes(
        cls,
        *,
        policy_bytes: bytes,
        controller_source_bytes: bytes,
        expected_policy_sha256: str,
        expected_controller_source_sha256: str,
    ) -> SealedControllerResources:
        _require(
            type(policy_bytes) is bytes
            and 1 <= len(policy_bytes) <= 8 * 1024 * 1024
            and type(controller_source_bytes) is bytes
            and 1 <= len(controller_source_bytes) <= 8 * 1024 * 1024
            and type(expected_policy_sha256) is str
            and SHA256_RE.fullmatch(expected_policy_sha256) is not None
            and type(expected_controller_source_sha256) is str
            and SHA256_RE.fullmatch(expected_controller_source_sha256) is not None
            and _sha(policy_bytes) == expected_policy_sha256
            and _sha(controller_source_bytes) == expected_controller_source_sha256,
            "sealed_controller_resource_digest_rejected",
        )
        try:
            policy = json.loads(policy_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError):
            _fail("sealed_controller_policy_json_rejected")
        _require(
            type(policy) is dict
            and type(policy.get("schema_version")) is int
            and policy.get("schema_version") == 1
            and policy.get("repository") == REPOSITORY,
            "sealed_controller_policy_binding_rejected",
        )
        instance = object.__new__(cls)
        object.__setattr__(instance, "policy_sha256", expected_policy_sha256)
        object.__setattr__(
            instance,
            "controller_source_sha256",
            expected_controller_source_sha256,
        )
        object.__setattr__(instance, "_policy_bytes", bytes(policy_bytes))
        return instance

    @classmethod
    def from_files_for_tests(
        cls,
        *,
        policy_path: Path = RELEASE_CONTROL_POLICY_PATH,
        controller_source_path: Path | None = None,
    ) -> SealedControllerResources:
        """Explicit test-only filesystem adapter; production must preload bytes."""

        source_path = controller_source_path or Path(__file__).resolve()
        _require(
            policy_path.is_file()
            and not policy_path.is_symlink()
            and source_path.is_file()
            and not source_path.is_symlink(),
            "test_controller_resource_file_rejected",
        )
        policy_bytes = policy_path.read_bytes()
        source_bytes = source_path.read_bytes()
        return cls.from_captured_bytes(
            policy_bytes=policy_bytes,
            controller_source_bytes=source_bytes,
            expected_policy_sha256=_sha(policy_bytes),
            expected_controller_source_sha256=_sha(source_bytes),
        )

    def policy_mapping(self) -> dict[str, Any]:
        try:
            value = json.loads(self._policy_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError):
            _fail("sealed_controller_policy_memory_drift")
        _require(
            type(value) is dict and _sha(self._policy_bytes) == self.policy_sha256,
            "sealed_controller_policy_memory_drift",
        )
        return _json_mapping_copy(value, "sealed_controller_policy")

    def to_public_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "explainiverse-sealed-controller-resources",
            "policy_sha256": self.policy_sha256,
            "controller_source_sha256": self.controller_source_sha256,
            "policy_bytes": len(self._policy_bytes),
            "source_bytes_retained": False,
            "filesystem_reopened_after_capture": False,
        }


SENSITIVE_RUNNER_AUTHORITY_PERMISSIONS = {"actions", "administration", "workflows"}

REMOTE_RECEIPT_KEYS = {
    "schema_version",
    "kind",
    "status",
    "policy_sha256",
    "control_plane_plan_sha256",
    "runtime_plan_sha256",
    "phase",
    "repository",
    "workflow_path",
    "ref",
    "run_id",
    "run_attempt",
    "head_sha",
    "job_id",
    "job_name",
    "runner_id",
    "runner_name",
    "labels",
    "host_physical_gpu_uuids",
    "host_physical_gpu_products",
    "assigned_physical_gpu_uuids",
    "unrequested_physical_gpu_uuids",
    "nvidia_visible_devices",
    "cuda_visible_devices",
    "runner_version",
    "runner_archive_sha256",
    "runner_image_reference",
    "runner_image_manifest_digest",
    "jit_config_sha256",
    "jit_config_persisted",
    "jit_config_destroyed",
    "jit_config_sent_at",
    "one_job_jit_configuration_supplied",
    "claimed_job_count_verified_by_runtime",
    "runner_exit_code",
    "started_at",
    "stopped_at",
    "cleanup_verified_at",
    "authority_expires_at",
    "workload_stopped_before_authority_expiry",
    "cleanup_grace_seconds",
    "cleanup_deadline_at",
    "cleanup_completed_within_grace",
    "descendants_remaining",
    "container_present",
    "network_present",
    "firewall_chain_present",
    "pre_jit_registration_absence_evidence_sha256",
    "post_exit_registration_absence_verified_by_runtime",
    "post_exit_registration_state",
    "github_contacted_by_runtime",
    "test_counts_verified_by_runtime",
    "job_success_verified_by_runtime",
    "accepted_actions_evidence",
}


class ControllerError(RuntimeError):
    """Stable fail-closed error that never includes secret values."""


class AmbiguousRemoteExecution(ControllerError):
    """Remote start was attempted; sanitized reconciliation is attached."""

    def __init__(self, receipt: Mapping[str, Any]) -> None:
        super().__init__("ambiguous_remote_execution")
        self.receipt = dict(receipt)


class AmbiguousGitHubMutation(ControllerError):
    """A GitHub mutation response was lost or unusable and must not be replayed."""

    def __init__(
        self,
        method: str,
        path: str,
        request_sha256: str,
        reason_code: str,
        *,
        reconciliation: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(f"ambiguous_github_{reason_code}")
        self.method = method
        self.path = path
        self.request_sha256 = request_sha256
        self.reason_code = reason_code
        self.reconciliation = dict(reconciliation or {})

    def to_public_mapping(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "path": self.path,
            "request_sha256": self.request_sha256,
            "reason_code": self.reason_code,
            "mutation_retried": False,
            "reconciliation": dict(self.reconciliation),
        }


def _fail(code: str) -> NoReturn:
    raise ControllerError(code)


def _require(condition: bool, code: str) -> None:
    if not condition:
        _fail(code)


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("ascii")


def _json_mapping_copy(value: Any, context: str) -> dict[str, Any]:
    try:
        copied = json.loads(_canonical(value))
    except (TypeError, ValueError):
        _fail(f"{context}_not_canonical_json")
    _require(type(copied) is dict, f"{context}_not_object")
    return copied


def _sha(value: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(value).hexdigest()


def _iso(value: datetime) -> str:
    current = value.astimezone(timezone.utc)
    return current.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_time(value: Any, context: str) -> datetime:
    _require(type(value) is str and value.endswith("Z"), f"{context}_timestamp_rejected")
    try:
        result = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        _fail(f"{context}_timestamp_rejected")
    return result.astimezone(timezone.utc)


def _pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, "json_duplicate_key_rejected")
        result[key] = value
    return result


def _json(raw: bytes | bytearray, context: str, maximum: int = MAX_RESPONSE_BYTES) -> Any:
    _require(0 < len(raw) <= maximum, f"{context}_size_rejected")
    try:
        return json.loads(bytes(raw), object_pairs_hook=_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail(f"{context}_json_rejected")


def _object(value: Any, keys: set[str], context: str) -> Mapping[str, Any]:
    _require(type(value) is dict, f"{context}_not_object")
    _require(set(value) == keys, f"{context}_keys_rejected")
    return value


def _required(value: Any, keys: set[str], context: str) -> Mapping[str, Any]:
    _require(type(value) is dict, f"{context}_not_object")
    _require(keys.issubset(value), f"{context}_keys_missing")
    return value


def _positive(value: Any, context: str) -> int:
    _require(type(value) is int and 0 < value < 2**63, f"{context}_rejected")
    return value


def _commit(value: Any, context: str) -> str:
    _require(type(value) is str and COMMIT_RE.fullmatch(value) is not None, f"{context}_rejected")
    return value


def _validate_strict_ssh_binding_shape(
    binding: live.StrictSshBinding,
    *,
    expected_mode: str,
    expected_public_ipv4: str,
    expected_host_fingerprint: str,
    expected_known_hosts_path: str,
    expected_known_hosts_sha256: str,
    expected_acl_receipt_sha256: str,
) -> None:
    fixed_command = {
        "cloud-init": live.FIXED_CLOUD_INIT_WAIT_COMMAND,
        "preflight": live.FIXED_PREFLIGHT_COMMAND,
        "run": live.FIXED_REMOTE_COMMAND,
    }.get(expected_mode)
    _require(fixed_command is not None, "ssh_expected_mode_rejected")
    assert fixed_command is not None
    _require(
        type(binding) is live.StrictSshBinding
        and binding.remote_mode == expected_mode
        and binding.remote_command == fixed_command
        and binding.host_fingerprint == expected_host_fingerprint
        and binding.known_hosts_path == expected_known_hosts_path
        and binding.known_hosts_sha256 == expected_known_hosts_sha256
        and binding.evidence_directory_acl_receipt_sha256 == expected_acl_receipt_sha256,
        "ssh_binding_metadata_drift",
    )
    known_path = Path(expected_known_hosts_path)
    _require(
        known_path.is_absolute() and known_path.is_file() and not known_path.is_symlink(),
        "ssh_known_hosts_file_rejected",
    )
    known_bytes = known_path.read_bytes()
    _require(
        _sha(known_bytes) == expected_known_hosts_sha256
        and binding.known_hosts.encode("ascii") == known_bytes,
        "ssh_known_hosts_content_drift",
    )
    argv = tuple(binding.argv_prefix)
    _require(len(argv) >= 7 + len(fixed_command), "ssh_argv_too_short")
    _require(argv[0] == "ssh" and argv[4] == "-i", "ssh_argv_prefix_rejected")
    identity_path = Path(argv[5])
    _require(
        identity_path.is_absolute() and identity_path.is_file(),
        "ssh_access_identity_file_rejected",
    )
    expected_prefix = (
        "ssh",
        "-T",
        "-F",
        os.devnull,
        "-i",
        str(identity_path),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "IdentityAgent=none",
        "-o",
        "RequestTTY=no",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "HostKeyAlgorithms=ssh-ed25519",
        "-o",
        f"UserKnownHostsFile={expected_known_hosts_path}",
        "-o",
        f"GlobalKnownHostsFile={os.devnull}",
        "-o",
        "UpdateHostKeys=no",
        "-o",
        "CheckHostIP=yes",
        "-o",
        "PasswordAuthentication=no",
        "-o",
        "KbdInteractiveAuthentication=no",
        "-o",
        "ForwardAgent=no",
        "-o",
        "ClearAllForwardings=yes",
        "-o",
        "ConnectTimeout=20",
        "-p",
        "22",
        f"ubuntu@{expected_public_ipv4}",
        *fixed_command,
    )
    _require(argv == expected_prefix, "ssh_argv_not_canonical")


def _response_envelope_digest(response: GitHubResponse) -> str:
    return _sha(
        _canonical(
            {
                "method": response.method,
                "path": response.path,
                "status_code": response.status_code,
                "headers_sha256": response.headers_sha256,
                "body_sha256": _sha(response.body),
            }
        )
    )


@dataclass
class GitHubResponse:
    method: str
    path: str
    status_code: int
    body: bytearray = field(repr=False)
    headers_sha256: str

    def destroy(self) -> None:
        for index in range(len(self.body)):
            self.body[index] = 0
        self.body.clear()


class GitHubTransport(Protocol):
    def request(
        self, method: str, path: str, body: Mapping[str, Any] | None = None
    ) -> GitHubResponse: ...


ProgressSink = Callable[[str, Mapping[str, Any]], None]


def _progress(sink: ProgressSink | None, label: str, payload: Mapping[str, Any]) -> None:
    if sink is not None:
        _require(callable(sink), "progress_sink_not_callable")
        sink(label, _json_mapping_copy(payload, "progress_payload"))


class GhCliTransport:
    """Use the locally authenticated ``gh api`` without a command shell.

    The process inherits the user's local authentication mechanism.  This
    class never queries ``gh auth token`` and never reads, copies, or logs a
    token.  ``--include`` makes the HTTP status directly enforceable.
    """

    def __init__(
        self,
        *,
        executable_path: str,
        executable_sha256: str,
        runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    ) -> None:
        path = Path(executable_path)
        _require(path.is_absolute(), "gh_executable_path_not_absolute")
        _require(path == path.resolve(strict=True), "gh_executable_path_not_canonical")
        _require(path.is_file() and not path.is_symlink(), "gh_executable_file_rejected")
        _require(
            SHA256_RE.fullmatch(executable_sha256) is not None,
            "gh_executable_digest_rejected",
        )
        _require(
            _sha(path.read_bytes()) == executable_sha256,
            "gh_executable_digest_mismatch",
        )
        self._executable_path = path
        self._executable_sha256 = executable_sha256
        self._runner = runner

    def executable_receipt(self) -> dict[str, Any]:
        _require(
            self._executable_path.is_file()
            and not self._executable_path.is_symlink()
            and _sha(self._executable_path.read_bytes()) == self._executable_sha256,
            "gh_executable_posture_drift",
        )
        return {
            "absolute_path": str(self._executable_path),
            "sha256": self._executable_sha256,
            "regular_file": True,
            "symlink": False,
            "path_lookup_used": False,
            "hostname_pinned": "github.com",
            "child_environment_names": sorted(self._environment()),
            "ambient_token_environment_forwarded": False,
        }

    @staticmethod
    def _environment() -> dict[str, str]:
        # The absolute gh executable needs only OS/profile/config discovery and
        # temporary-directory values.  Authentication remains in gh's local
        # owner profile; ambient tokens, provider keys, PATH, host overrides,
        # proxies, and debug/pager knobs are deliberately not inherited.
        allowed = {
            "APPDATA",
            "LOCALAPPDATA",
            "USERPROFILE",
            "SYSTEMROOT",
            "WINDIR",
            "COMSPEC",
            "TEMP",
            "TMP",
            "LANG",
            "LC_ALL",
            "NO_COLOR",
        }
        return {key: value for key, value in os.environ.items() if key.upper() in allowed}

    @staticmethod
    def _decode_included(raw: bytes) -> tuple[int, bytes, bytes]:
        remaining = raw
        final_status: int | None = None
        final_headers = b""
        while remaining.startswith(b"HTTP/"):
            separator = b"\r\n\r\n" if b"\r\n\r\n" in remaining else b"\n\n"
            _require(separator in remaining, "gh_response_headers_rejected")
            header, remaining = remaining.split(separator, 1)
            first = header.splitlines()[0]
            match = re.fullmatch(rb"HTTP/\S+ ([0-9]{3})(?: .*)?", first)
            _require(match is not None, "gh_response_status_rejected")
            assert match is not None
            final_status = int(match.group(1))
            final_headers = header
            if remaining.startswith(b"HTTP/") and (final_status < 200 or 300 <= final_status < 400):
                continue
            break
        _require(final_status is not None, "gh_response_status_missing")
        assert final_status is not None
        return final_status, final_headers, remaining

    def request(
        self, method: str, path: str, body: Mapping[str, Any] | None = None
    ) -> GitHubResponse:
        _require(method in {"GET", "POST", "DELETE"}, "github_method_rejected")
        _require(path.startswith("/repos/jemsbhai/explainiverse/"), "github_path_rejected")
        self.executable_receipt()
        argv = [
            str(self._executable_path),
            "api",
            "--hostname",
            "github.com",
            "--include",
            "--method",
            method,
            "-H",
            f"Accept: {API_ACCEPT}",
            "-H",
            f"X-GitHub-Api-Version: {API_VERSION}",
            path,
        ]
        stdin = None
        if body is not None:
            argv.extend(("--input", "-"))
            stdin = _canonical(body)
        request_sha256 = _sha(_canonical({"method": method, "path": path, "body": body}))
        mutating = method in {"POST", "DELETE"}
        try:
            completed = self._runner(
                argv,
                input=stdin,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
                shell=False,
                check=False,
                env=self._environment(),
            )
        except (OSError, subprocess.SubprocessError):
            if mutating:
                raise AmbiguousGitHubMutation(
                    method, path, request_sha256, "transport_failure"
                ) from None
            _fail("gh_transport_failure")
        if completed.returncode != 0:
            if mutating:
                raise AmbiguousGitHubMutation(method, path, request_sha256, "cli_nonzero")
            _fail("gh_api_failed")
        try:
            _require(len(completed.stdout) <= MAX_LOG_BYTES, "gh_response_too_large")
            status, headers, response_body = self._decode_included(completed.stdout)
        except ControllerError as exc:
            if mutating:
                raise AmbiguousGitHubMutation(
                    method,
                    path,
                    request_sha256,
                    "successful_response_unusable",
                    reconciliation={
                        "response_validation_code": str(exc),
                        "mutation_retried": False,
                    },
                ) from None
            raise
        return GitHubResponse(
            method=method,
            path=path,
            status_code=status,
            body=bytearray(response_body),
            headers_sha256=_sha(headers),
        )


@dataclass(frozen=True, init=False)
class TrustedAppCapture:
    captured_at: str
    evidence_sha256: str
    policy_sha256: str
    installations: tuple[dict[str, Any], ...]
    normalized_capture: dict[str, Any]

    def __new__(cls, *_: object, **__: object) -> TrustedAppCapture:
        raise TypeError("TrustedAppCapture must be created with from_mapping")

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        resources: SealedControllerResources,
        evidence_reader: Callable[[str], bytes],
        now: datetime | None = None,
    ) -> TrustedAppCapture:
        """Bind the complete owner-authenticated App capture and every raw page.

        The accepted schema and allow/deny policy are deliberately shared with
        ``release_external_controls``. A caller cannot substitute a reduced
        permission map or an unbound digest for the retained browser pages.
        """

        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        _require(
            type(resources) is SealedControllerResources,
            "sealed_controller_resources_required",
        )
        _require(callable(evidence_reader), "app_evidence_reader_not_callable")
        try:
            policy = resources.policy_mapping()
            policy_sha256 = resources.policy_sha256
            _require(policy.get("repository") == REPOSITORY, "app_policy_repository_drift")
            authority_policy = external_controls._mapping(
                policy.get("release_runner_authority"), "release runner authority policy"
            )
            apps_policy = external_controls._mapping(
                authority_policy.get("installed_apps"), "release runner installed Apps policy"
            )
            _require(
                set(apps_policy) == {"source_url", "expected_installations"},
                "app_policy_schema_drift",
            )
            expected = [
                external_controls._canonical_app_installation(
                    item, "expected installed App installation"
                )
                for item in external_controls._sequence(
                    apps_policy.get("expected_installations"),
                    "expected installed App installations",
                )
            ]
            expected.sort(key=lambda item: item["id"])
            normalized = external_controls._normalize_installed_app_authority(
                value,
                repository=REPOSITORY,
                capture_principal=OWNER,
                evidence_reader=evidence_reader,
            )
        except ControllerError:
            raise
        except (KeyError, OSError, TypeError, ValueError):
            _fail("app_capture_invalid_or_unbound")

        _require(
            normalized.get("source_url") == apps_policy.get("source_url"),
            "app_capture_source_drift",
        )
        _require(
            normalized.get("installations") == expected,
            "installed_app_policy_drift",
        )
        for installation in expected:
            effective_sensitive = SENSITIVE_RUNNER_AUTHORITY_PERMISSIONS.intersection(
                installation["permissions"]["write"]
            )
            requested_sensitive = SENSITIVE_RUNNER_AUTHORITY_PERMISSIONS.intersection(
                installation["requested_additional_permissions"]["write"]
            )
            _require(
                not (
                    installation["repository_access"]
                    and not installation["suspended"]
                    and (effective_sensitive or requested_sensitive)
                ),
                "active_app_runner_authority_present",
            )
            _require(
                not (installation["permission_update_requested"] and not installation["suspended"]),
                "active_app_permission_update_unresolved",
            )

        try:
            captured_at = external_controls._aware_utc_timestamp(
                normalized["captured_at"], "installed App authority captured_at"
            )
        except ValueError:
            _fail("app_capture_timestamp_rejected")
        _require(
            captured_at <= current + timedelta(seconds=5)
            and current - captured_at <= AUTHORITY_CAPTURE_MAX_AGE,
            "app_capture_stale",
        )
        normalized_copy = json.loads(_canonical(normalized))
        _require(type(normalized_copy) is dict, "app_capture_normalization_failed")
        digest = _sha(
            _canonical(
                {
                    "policy_sha256": policy_sha256,
                    "normalized_capture": normalized_copy,
                }
            )
        )
        installations = normalized_copy["installations"]
        _require(type(installations) is list, "app_capture_normalization_failed")
        instance = object.__new__(cls)
        object.__setattr__(instance, "captured_at", _iso(captured_at))
        object.__setattr__(instance, "evidence_sha256", digest)
        object.__setattr__(instance, "policy_sha256", policy_sha256)
        object.__setattr__(instance, "installations", tuple(dict(item) for item in installations))
        object.__setattr__(instance, "normalized_capture", normalized_copy)
        return instance

    def to_mapping(self) -> dict[str, Any]:
        return {
            "captured_at": self.captured_at,
            "evidence_sha256": self.evidence_sha256,
            "policy_sha256": self.policy_sha256,
            "installations": [dict(item) for item in self.installations],
            "normalized_capture": json.loads(_canonical(self.normalized_capture)),
        }


@dataclass(frozen=True)
class AuthorityReceipt:
    observed_at: str
    expires_at: str
    evidence_sha256: str
    app_capture_sha256: str
    collaborators_response_sha256: str
    invitations_response_sha256: str
    runners_response_sha256: str
    variables_response_sha256: str
    queue_evidence_sha256: str
    _evidence_material: Mapping[str, Any] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def runtime_mapping(self) -> dict[str, Any]:
        return {
            "observed_at": self.observed_at,
            "expires_at": self.expires_at,
            "evidence_sha256": self.evidence_sha256,
            "owner_login": OWNER,
            "collaborators": [{"login": OWNER, "permission": "admin"}],
            "pending_invitation_count": 0,
            "enabled_nonowner_authorities": [],
            "unexpected_target_job_count": 0,
        }

    def evidence_mapping(self) -> dict[str, Any]:
        """Return the full public material needed to revalidate this receipt."""

        _require(
            self._evidence_material is not None,
            "authority_evidence_material_missing",
        )
        assert self._evidence_material is not None
        material = _json_mapping_copy(
            self._evidence_material,
            "authority_evidence_material",
        )
        mapping = {
            "schema_version": 1,
            "kind": "explainiverse-github-authority-window",
            "observed_at": self.observed_at,
            "expires_at": self.expires_at,
            "evidence_sha256": self.evidence_sha256,
            "app_capture_sha256": self.app_capture_sha256,
            "collaborators_response_sha256": self.collaborators_response_sha256,
            "invitations_response_sha256": self.invitations_response_sha256,
            "runners_response_sha256": self.runners_response_sha256,
            "variables_response_sha256": self.variables_response_sha256,
            "queue_evidence_sha256": self.queue_evidence_sha256,
            "evidence_material": material,
        }
        AuthorityReceipt.from_evidence_mapping(mapping)
        return mapping

    @classmethod
    def from_evidence_mapping(cls, value: Mapping[str, Any]) -> AuthorityReceipt:
        mapping = _object(
            value,
            {
                "schema_version",
                "kind",
                "observed_at",
                "expires_at",
                "evidence_sha256",
                "app_capture_sha256",
                "collaborators_response_sha256",
                "invitations_response_sha256",
                "runners_response_sha256",
                "variables_response_sha256",
                "queue_evidence_sha256",
                "evidence_material",
            },
            "authority_evidence",
        )
        material = _object(
            mapping["evidence_material"],
            {
                "observed_at",
                "app_capture",
                "collaborators",
                "invitations",
                "runners",
                "variables",
                "queue",
            },
            "authority_evidence_material",
        )
        observed_at = _iso(_parse_time(mapping["observed_at"], "authority_observed"))
        expires_at = _iso(_parse_time(mapping["expires_at"], "authority_expires"))
        app_capture = _required(
            material["app_capture"],
            {"evidence_sha256"},
            "authority_app_capture",
        )
        queue = _required(
            material["queue"],
            {"response_sha256"},
            "authority_queue",
        )
        collaborators = material["collaborators"]
        invitations = material["invitations"]
        variables = material["variables"]
        digest_fields = (
            "evidence_sha256",
            "app_capture_sha256",
            "collaborators_response_sha256",
            "invitations_response_sha256",
            "runners_response_sha256",
            "variables_response_sha256",
            "queue_evidence_sha256",
        )
        _require(
            type(mapping["schema_version"]) is int
            and mapping["schema_version"] == 1
            and mapping["kind"] == "explainiverse-github-authority-window"
            and mapping["observed_at"] == observed_at
            and mapping["expires_at"] == expires_at
            and _parse_time(expires_at, "authority_expires")
            == _parse_time(observed_at, "authority_observed") + AUTHORITY_WINDOW
            and material["observed_at"] == observed_at
            and all(
                type(mapping[field_name]) is str
                and SHA256_RE.fullmatch(mapping[field_name]) is not None
                for field_name in digest_fields
            )
            and type(collaborators) is list
            and bool(collaborators)
            and type(invitations) is list
            and bool(invitations)
            and type(variables) is list
            and bool(variables)
            and all(
                type(digest) is str and SHA256_RE.fullmatch(digest) is not None
                for digests in (collaborators, invitations, variables)
                for digest in digests
            )
            and type(material["runners"]) is str
            and SHA256_RE.fullmatch(material["runners"]) is not None
            and type(app_capture["evidence_sha256"]) is str
            and app_capture["evidence_sha256"] == mapping["app_capture_sha256"]
            and type(queue["response_sha256"]) is str
            and queue["response_sha256"] == mapping["queue_evidence_sha256"]
            and _sha(_canonical(material)) == mapping["evidence_sha256"]
            and _sha(_canonical(collaborators)) == mapping["collaborators_response_sha256"]
            and _sha(_canonical(invitations)) == mapping["invitations_response_sha256"]
            and material["runners"] == mapping["runners_response_sha256"]
            and _sha(_canonical(variables)) == mapping["variables_response_sha256"],
            "authority_evidence_binding_rejected",
        )
        return cls(
            observed_at=observed_at,
            expires_at=expires_at,
            evidence_sha256=str(mapping["evidence_sha256"]),
            app_capture_sha256=str(mapping["app_capture_sha256"]),
            collaborators_response_sha256=str(mapping["collaborators_response_sha256"]),
            invitations_response_sha256=str(mapping["invitations_response_sha256"]),
            runners_response_sha256=str(mapping["runners_response_sha256"]),
            variables_response_sha256=str(mapping["variables_response_sha256"]),
            queue_evidence_sha256=str(mapping["queue_evidence_sha256"]),
            _evidence_material=_json_mapping_copy(
                material,
                "authority_evidence_material",
            ),
        )


def _validated_authority_evidence_identity(
    value: Mapping[str, Any],
    *,
    context: str,
    expected_phase: str | None = None,
    expected_head_sha: str | None = None,
    expected_run_id: int | None = None,
    expected_job_key: str | None = None,
) -> dict[str, Any]:
    """Validate one public, digest-bound per-JIT authority identity."""

    mapping = _object(
        value,
        {
            "schema_version",
            "kind",
            "phase",
            "head_sha",
            "run_id",
            "job_key",
            "capture_evidence_sha256",
            "authority_evidence_sha256",
            "archive_evidence_sha256",
            "raw_page_sha256",
            "dispatch_observed_at",
            "captured_at",
            "authority_observed_at",
            "runtime_created_at",
            "evidence_sha256",
        },
        context,
    )
    phase = mapping["phase"]
    head_sha = _commit(mapping["head_sha"], f"{context}_head_sha")
    run_id = _positive(mapping["run_id"], f"{context}_run_id")
    job_key = mapping["job_key"]
    raw_pages = mapping["raw_page_sha256"]
    dispatch_at = _parse_time(mapping["dispatch_observed_at"], f"{context}_dispatch_observed")
    captured_at = _parse_time(mapping["captured_at"], f"{context}_captured")
    authority_at = _parse_time(mapping["authority_observed_at"], f"{context}_authority_observed")
    runtime_at = _parse_time(mapping["runtime_created_at"], f"{context}_runtime_created")
    digest_fields = (
        "capture_evidence_sha256",
        "authority_evidence_sha256",
        "archive_evidence_sha256",
        "evidence_sha256",
    )
    _require(
        type(mapping["schema_version"]) is int
        and mapping["schema_version"] == 1
        and mapping["kind"] == "explainiverse-jit-authority-evidence-identity"
        and type(phase) is str
        and phase in PHASES
        and type(job_key) is str
        and job_key in PHASES[phase]["job_keys"]
        and all(
            type(mapping[field_name]) is str
            and SHA256_RE.fullmatch(mapping[field_name]) is not None
            for field_name in digest_fields
        )
        and type(raw_pages) is list
        and bool(raw_pages)
        and len(set(raw_pages)) == len(raw_pages)
        and all(type(item) is str and SHA256_RE.fullmatch(item) is not None for item in raw_pages)
        and mapping["dispatch_observed_at"] == _iso(dispatch_at)
        and mapping["captured_at"] == _iso(captured_at)
        and mapping["authority_observed_at"] == _iso(authority_at)
        and mapping["runtime_created_at"] == _iso(runtime_at)
        and dispatch_at < captured_at < authority_at < runtime_at
        and authority_at - captured_at <= AUTHORITY_CAPTURE_MAX_AGE,
        f"{context}_binding_rejected",
    )
    material = dict(mapping)
    evidence_sha256 = material.pop("evidence_sha256")
    _require(
        _sha(_canonical(material)) == evidence_sha256
        and (expected_phase is None or phase == expected_phase)
        and (expected_head_sha is None or head_sha == expected_head_sha)
        and (expected_run_id is None or run_id == expected_run_id)
        and (expected_job_key is None or job_key == expected_job_key),
        f"{context}_evidence_rejected",
    )
    normalized = _json_mapping_copy(mapping, context)
    normalized["head_sha"] = head_sha
    normalized["run_id"] = run_id
    return normalized


@dataclass(frozen=True)
class JobBinding:
    key: str
    ordinal: int
    job_id: int
    name: str
    nonce: str
    runner_name: str


@dataclass(frozen=True)
class DispatchReceipt:
    observed_at: str
    request_sha256: str
    response_sha256: str
    workflow_response_sha256: str
    run_response_sha256: str
    nonce_history_observed_at: str
    nonce_history_response_sha256: str
    mutation_response_received: bool
    mutation_reconciliation_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DispatchReceipt:
        mapping = _object(value, set(cls.__dataclass_fields__), "dispatch_receipt")
        observed_at = _iso(_parse_time(mapping["observed_at"], "dispatch_receipt_observed"))
        history_at = _iso(
            _parse_time(
                mapping["nonce_history_observed_at"],
                "dispatch_receipt_nonce_history_observed",
            )
        )
        digest_fields = (
            "request_sha256",
            "response_sha256",
            "workflow_response_sha256",
            "run_response_sha256",
            "nonce_history_response_sha256",
            "mutation_reconciliation_sha256",
        )
        _require(
            mapping["observed_at"] == observed_at
            and mapping["nonce_history_observed_at"] == history_at
            and _parse_time(history_at, "dispatch_receipt_nonce_history_order")
            < _parse_time(observed_at, "dispatch_receipt_observed_order")
            and type(mapping["mutation_response_received"]) is bool
            and all(
                type(mapping[field_name]) is str
                and SHA256_RE.fullmatch(mapping[field_name]) is not None
                for field_name in digest_fields
            ),
            "dispatch_receipt_binding_rejected",
        )
        return cls(
            observed_at=observed_at,
            request_sha256=str(mapping["request_sha256"]),
            response_sha256=str(mapping["response_sha256"]),
            workflow_response_sha256=str(mapping["workflow_response_sha256"]),
            run_response_sha256=str(mapping["run_response_sha256"]),
            nonce_history_observed_at=history_at,
            nonce_history_response_sha256=str(mapping["nonce_history_response_sha256"]),
            mutation_response_received=bool(mapping["mutation_response_received"]),
            mutation_reconciliation_sha256=str(mapping["mutation_reconciliation_sha256"]),
        )


@dataclass(frozen=True)
class RecoveryDispatchReceipt:
    observed_at: str
    tag: str
    head_sha: str
    source_run_id: int
    require_staged_drill: bool
    recovery_request_nonce: str
    display_title: str
    run_id: int
    run_attempt: int
    status: str
    conclusion: str | None
    request_sha256: str
    mutation_response_received: bool
    mutation_response_sha256: str | None
    workflow_response_sha256: str
    immutable_source_evidence_sha256: str
    source_run_evidence_sha256: str
    pre_dispatch_history_sha256: str
    reconciliation_sha256: str
    evidence_sha256: str

    def to_mapping(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RecoveryDispatchReceipt:
        mapping = _object(
            value,
            set(cls.__dataclass_fields__),
            "recovery_dispatch_receipt",
        )
        observed_at = _iso(_parse_time(mapping["observed_at"], "recovery_dispatch_observed"))
        head_sha = _commit(mapping["head_sha"], "recovery_dispatch_receipt_head")
        source_run_id = _positive(mapping["source_run_id"], "recovery_dispatch_receipt_source_run")
        run_id = _positive(mapping["run_id"], "recovery_dispatch_receipt_run")
        nonce = mapping["recovery_request_nonce"]
        status = mapping["status"]
        conclusion = mapping["conclusion"]
        response_received = mapping["mutation_response_received"]
        response_sha256 = mapping["mutation_response_sha256"]
        _require(
            mapping["observed_at"] == observed_at
            and mapping["tag"] == runtime.PUBLICATION_TAG
            and mapping["require_staged_drill"] is True
            and type(nonce) is str
            and NONCE_RE.fullmatch(nonce) is not None
            and mapping["display_title"]
            == ReleaseGpuController._recovery_display_title(
                runtime.PUBLICATION_TAG, source_run_id, nonce
            )
            and type(mapping["run_attempt"]) is int
            and mapping["run_attempt"] == 1
            and type(status) is str
            and status in {"queued", "in_progress", "completed"}
            and (
                (status != "completed" and conclusion is None)
                or (
                    status == "completed"
                    and type(conclusion) is str
                    and conclusion
                    in {
                        "success",
                        "failure",
                        "cancelled",
                        "timed_out",
                        "action_required",
                        "neutral",
                        "skipped",
                        "stale",
                        "startup_failure",
                    }
                )
            )
            and type(response_received) is bool
            and (
                (
                    response_received
                    and type(response_sha256) is str
                    and SHA256_RE.fullmatch(response_sha256) is not None
                )
                or (not response_received and response_sha256 is None)
            ),
            "recovery_dispatch_receipt_binding_rejected",
        )
        for field_name in (
            "request_sha256",
            "workflow_response_sha256",
            "immutable_source_evidence_sha256",
            "source_run_evidence_sha256",
            "pre_dispatch_history_sha256",
            "reconciliation_sha256",
            "evidence_sha256",
        ):
            _require(
                type(mapping[field_name]) is str
                and SHA256_RE.fullmatch(mapping[field_name]) is not None,
                "recovery_dispatch_receipt_digest_rejected",
            )
        material = dict(mapping)
        evidence_sha256 = material.pop("evidence_sha256")
        _require(
            _sha(_canonical(material)) == evidence_sha256,
            "recovery_dispatch_receipt_evidence_mismatch",
        )
        return cls(
            observed_at=observed_at,
            tag=runtime.PUBLICATION_TAG,
            head_sha=head_sha,
            source_run_id=source_run_id,
            require_staged_drill=True,
            recovery_request_nonce=str(nonce),
            display_title=str(mapping["display_title"]),
            run_id=run_id,
            run_attempt=1,
            status=str(status),
            conclusion=str(conclusion) if conclusion is not None else None,
            request_sha256=str(mapping["request_sha256"]),
            mutation_response_received=bool(response_received),
            mutation_response_sha256=(
                str(response_sha256) if response_sha256 is not None else None
            ),
            workflow_response_sha256=str(mapping["workflow_response_sha256"]),
            immutable_source_evidence_sha256=str(mapping["immutable_source_evidence_sha256"]),
            source_run_evidence_sha256=str(mapping["source_run_evidence_sha256"]),
            pre_dispatch_history_sha256=str(mapping["pre_dispatch_history_sha256"]),
            reconciliation_sha256=str(mapping["reconciliation_sha256"]),
            evidence_sha256=str(evidence_sha256),
        )


@dataclass
class PhaseSession:
    phase: str
    workflow: str
    workflow_path: str
    dispatch_ref: str
    run_ref: str
    head_sha: str
    inputs: dict[str, Any]
    prior_accepted_cuda_runner_nonces: tuple[str, ...]
    run: dict[str, Any]
    jobs: tuple[JobBinding, ...]
    queued_jobs: tuple[JobBinding, ...]
    dispatch_receipt: DispatchReceipt
    accepted: dict[str, AcceptedJobReceipt] = field(default_factory=dict)
    prior_authority_evidence_identities: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, init=False)
class HostReadinessReceipt:
    cloud_init: live.CloudInitWaitReceipt
    preflight: live.HostPreflightReceipt
    cloud_init_sha256: str
    preflight_sha256: str
    cloud_binding: dict[str, Any]
    preflight_binding: dict[str, Any]
    ssh_attempts: dict[str, Any]
    evidence_sha256: str

    def __new__(cls, *_: object, **__: object) -> HostReadinessReceipt:
        raise TypeError("HostReadinessReceipt is created only after live validation")

    def _material(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "explainiverse-lambda-host-readiness-binding",
            "control_plane_plan_sha256": self.cloud_init.plan_sha256,
            "instance_id": self.cloud_init.instance_id,
            "instance_public_ipv4": self.cloud_init.instance_public_ipv4,
            "host_fingerprint": self.cloud_init.host_fingerprint,
            "known_hosts_sha256": self.cloud_init.known_hosts_sha256,
            "cloud_init": self.cloud_init.to_public_mapping(),
            "preflight": self.preflight.to_public_mapping(),
            "cloud_init_sha256": self.cloud_init_sha256,
            "preflight_sha256": self.preflight_sha256,
            "cloud_binding": _json_mapping_copy(self.cloud_binding, "cloud_binding"),
            "preflight_binding": _json_mapping_copy(self.preflight_binding, "preflight_binding"),
            "ssh_attempts": _json_mapping_copy(self.ssh_attempts, "ssh_attempts"),
        }

    @classmethod
    def _from_validated(
        cls,
        cloud_init: live.CloudInitWaitReceipt,
        preflight: live.HostPreflightReceipt,
        cloud_binding: live.StrictSshBinding | Mapping[str, Any],
        preflight_binding: live.StrictSshBinding,
        ssh_attempts: Mapping[str, Any],
    ) -> HostReadinessReceipt:
        instance = object.__new__(cls)
        object.__setattr__(instance, "cloud_init", cloud_init)
        object.__setattr__(instance, "preflight", preflight)
        object.__setattr__(instance, "cloud_init_sha256", cloud_init.sha256)
        object.__setattr__(
            instance,
            "preflight_sha256",
            _sha(_canonical(preflight.to_public_mapping())),
        )
        cloud_binding_mapping = (
            cloud_binding.to_public_mapping()
            if type(cloud_binding) is live.StrictSshBinding
            else cloud_binding
        )
        object.__setattr__(
            instance,
            "cloud_binding",
            _json_mapping_copy(cloud_binding_mapping, "cloud"),
        )
        object.__setattr__(
            instance,
            "preflight_binding",
            _json_mapping_copy(preflight_binding.to_public_mapping(), "preflight"),
        )
        object.__setattr__(
            instance, "ssh_attempts", _json_mapping_copy(ssh_attempts, "ssh_attempts")
        )
        object.__setattr__(instance, "evidence_sha256", "0" * 64)
        object.__setattr__(instance, "evidence_sha256", _sha(_canonical(instance._material())))
        instance.validate()
        return instance

    def validate(self) -> None:
        self.cloud_init.validate_binding()
        _require(
            self.cloud_init_sha256 == self.cloud_init.sha256
            and self.preflight_sha256 == _sha(_canonical(self.preflight.to_public_mapping()))
            and self.cloud_init.plan_sha256 == self.preflight.plan_sha256
            and self.cloud_init.instance_id == self.preflight.instance_id
            and self.cloud_init.instance_public_ipv4 == self.preflight.instance_public_ipv4
            and self.cloud_init.host_fingerprint == self.preflight.host_fingerprint
            and self.cloud_init.known_hosts_sha256 == self.preflight.known_hosts_sha256
            and self.cloud_init.provider_receipt_nonce != self.preflight.provider_receipt_nonce
            and self.preflight.cloud_init_wait_binding_sha256 == self.cloud_init.binding_sha256,
            "host_readiness_component_drift",
        )
        material = self._material()
        _require(
            _sha(_canonical(material)) == self.evidence_sha256,
            "host_readiness_evidence_digest_mismatch",
        )


@dataclass(frozen=True)
class RemoteExecution:
    receipt: dict[str, Any]
    stdout_sha256: str
    stderr_sha256: str
    frame_receipt: dict[str, Any]


@dataclass(frozen=True)
class PublicSshResult:
    stdout: bytes
    stderr: bytes
    exit_code: int


@dataclass(frozen=True)
class AcceptedJobReceipt:
    phase: str
    run_id: int
    job_key: str
    job_id: int
    runner_id: int
    runner_name: str
    runtime_plan_sha256: str
    remote_receipt_sha256: str
    actions_job_response_sha256: str
    check_response_sha256: str
    log_sha256: str
    pytest_passed: int
    pytest_skipped: int
    runner_inventory_response_sha256: str
    post_execution_observation_sha256: str
    evidence_sha256: str

    def to_mapping(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True, init=False)
class FinalMainAcceptance:
    """Proof of four accepted jobs, optionally sealed by the durable journal loader.

    The controller creates an unsealed value solely so the final-main driver can
    archive it.  Publication accepts only the same value reconstructed and
    provenance-sealed by ``EvidenceJournal.load_final_main_acceptance``.
    """

    head_sha: str
    run_id: int
    accepted_cuda_runner_nonces: tuple[str, ...]
    jobs: tuple[dict[str, Any], ...]
    settlement: dict[str, Any]
    dispatch_nonce_history_observed_at: str
    dispatch_nonce_history_response_sha256: str
    evidence_sha256: str
    _final_journal_sha256: str | None
    _evidence_directory_receipt_sha256: str | None
    _journal_provenance_sha256: str | None
    _authority_evidence_identities: tuple[dict[str, Any], ...] | None

    def __new__(cls, *_: object, **__: object) -> FinalMainAcceptance:
        raise TypeError("FinalMainAcceptance must be created from a completed session")

    @classmethod
    def _from_completed_session(
        cls, session: PhaseSession, settlement: Mapping[str, Any]
    ) -> FinalMainAcceptance:
        _require(session.phase == "final-main", "final_acceptance_phase_rejected")
        expected_keys = tuple(PHASES["final-main"]["job_keys"])
        _require(tuple(session.accepted) == expected_keys, "final_acceptance_jobs_missing")
        jobs: list[dict[str, Any]] = []
        nonces: list[str] = []
        for binding in session.jobs:
            accepted = session.accepted[binding.key]
            mapping = accepted.to_mapping()
            _require(
                accepted.phase == "final-main"
                and accepted.run_id == session.run["id"]
                and accepted.job_key == binding.key
                and accepted.job_id == binding.job_id
                and accepted.runner_name == binding.runner_name
                and accepted.pytest_passed == 15
                and accepted.pytest_skipped == 0
                and all(
                    SHA256_RE.fullmatch(str(mapping[field])) is not None
                    for field in (
                        "runtime_plan_sha256",
                        "remote_receipt_sha256",
                        "actions_job_response_sha256",
                        "check_response_sha256",
                        "log_sha256",
                        "runner_inventory_response_sha256",
                        "post_execution_observation_sha256",
                        "evidence_sha256",
                    )
                ),
                "final_acceptance_job_receipt_rejected",
            )
            job_material = dict(mapping)
            job_evidence = job_material.pop("evidence_sha256")
            _require(
                _sha(_canonical(job_material)) == job_evidence,
                "final_acceptance_job_digest_mismatch",
            )
            jobs.append(mapping)
            nonces.append(binding.nonce)
        _require(len(set(nonces)) == 4, "final_acceptance_nonces_not_distinct")
        _require(
            len({item["job_id"] for item in jobs}) == 4
            and len({item["runner_id"] for item in jobs}) == 4,
            "final_acceptance_job_or_runner_ids_not_distinct",
        )
        settlement_copy = _json_mapping_copy(settlement, "final_settlement")
        settlement_material = dict(settlement_copy)
        settlement_evidence = settlement_material.pop("evidence_sha256", None)
        _require(
            type(settlement_evidence) is str
            and _sha(_canonical(settlement_material)) == settlement_evidence,
            "final_acceptance_settlement_digest_mismatch",
        )
        _require(
            settlement_copy.get("phase") == "final-main"
            and settlement_copy.get("run_id") == session.run["id"]
            and type(settlement_copy.get("run_attempt")) is int
            and settlement_copy.get("run_attempt") == 1
            and settlement_copy.get("head_sha") == session.head_sha
            and settlement_copy.get("accepted_cuda_runner_nonces") == nonces
            and settlement_copy.get("job_evidence_sha256")
            == [item["evidence_sha256"] for item in jobs]
            and settlement_copy.get("all_four_jobs_15_of_15_zero_skips") is True
            and settlement_copy.get("rerun_performed") is False,
            "final_acceptance_settlement_binding_rejected",
        )
        material = {
            "schema_version": 1,
            "kind": "explainiverse-final-main-cuda-acceptance",
            "head_sha": session.head_sha,
            "run_id": session.run["id"],
            "run_attempt": 1,
            "accepted_cuda_runner_nonces": nonces,
            "jobs": jobs,
            "settlement": settlement_copy,
            "dispatch_nonce_history_observed_at": (
                session.dispatch_receipt.nonce_history_observed_at
            ),
            "dispatch_nonce_history_response_sha256": (
                session.dispatch_receipt.nonce_history_response_sha256
            ),
        }
        _parse_time(material["dispatch_nonce_history_observed_at"], "final_nonce_history")
        _require(
            SHA256_RE.fullmatch(material["dispatch_nonce_history_response_sha256"]) is not None,
            "final_nonce_history_digest_rejected",
        )
        instance = object.__new__(cls)
        object.__setattr__(instance, "head_sha", session.head_sha)
        object.__setattr__(instance, "run_id", int(session.run["id"]))
        object.__setattr__(instance, "accepted_cuda_runner_nonces", tuple(nonces))
        object.__setattr__(
            instance, "jobs", tuple(_json_mapping_copy(item, "job") for item in jobs)
        )
        object.__setattr__(instance, "settlement", settlement_copy)
        object.__setattr__(
            instance,
            "dispatch_nonce_history_observed_at",
            session.dispatch_receipt.nonce_history_observed_at,
        )
        object.__setattr__(
            instance,
            "dispatch_nonce_history_response_sha256",
            session.dispatch_receipt.nonce_history_response_sha256,
        )
        object.__setattr__(instance, "evidence_sha256", _sha(_canonical(material)))
        object.__setattr__(instance, "_final_journal_sha256", None)
        object.__setattr__(instance, "_evidence_directory_receipt_sha256", None)
        object.__setattr__(instance, "_journal_provenance_sha256", None)
        object.__setattr__(instance, "_authority_evidence_identities", None)
        return instance

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "explainiverse-final-main-cuda-acceptance",
            "head_sha": self.head_sha,
            "run_id": self.run_id,
            "run_attempt": 1,
            "accepted_cuda_runner_nonces": list(self.accepted_cuda_runner_nonces),
            "jobs": [_json_mapping_copy(item, "job") for item in self.jobs],
            "settlement": _json_mapping_copy(self.settlement, "settlement"),
            "dispatch_nonce_history_observed_at": self.dispatch_nonce_history_observed_at,
            "dispatch_nonce_history_response_sha256": (self.dispatch_nonce_history_response_sha256),
            "evidence_sha256": self.evidence_sha256,
        }

    @classmethod
    def _from_verified_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        final_journal_sha256: str,
        evidence_directory_receipt_sha256: str,
        authority_evidence_identities: Sequence[Mapping[str, Any]] | None = None,
    ) -> FinalMainAcceptance:
        """Reconstruct only after the driver has verified the complete journal chain."""

        _require(
            SHA256_RE.fullmatch(final_journal_sha256) is not None
            and SHA256_RE.fullmatch(evidence_directory_receipt_sha256) is not None,
            "final_acceptance_journal_provenance_rejected",
        )

        mapping = _object(
            _json_mapping_copy(value, "final_acceptance"),
            {
                "schema_version",
                "kind",
                "head_sha",
                "run_id",
                "run_attempt",
                "accepted_cuda_runner_nonces",
                "jobs",
                "settlement",
                "dispatch_nonce_history_observed_at",
                "dispatch_nonce_history_response_sha256",
                "evidence_sha256",
            },
            "final_acceptance",
        )
        _require(
            type(mapping["schema_version"]) is int
            and mapping["schema_version"] == 1
            and mapping["kind"] == "explainiverse-final-main-cuda-acceptance"
            and type(mapping["run_attempt"]) is int
            and mapping["run_attempt"] == 1,
            "final_acceptance_identity_rejected",
        )
        head_sha = _commit(mapping["head_sha"], "final_acceptance_head")
        run_id = _positive(mapping["run_id"], "final_acceptance_run")
        raw_nonces = mapping["accepted_cuda_runner_nonces"]
        raw_jobs = mapping["jobs"]
        _require(
            type(raw_nonces) is list
            and len(raw_nonces) == 4
            and len(set(raw_nonces)) == 4
            and all(type(item) is str and NONCE_RE.fullmatch(item) for item in raw_nonces),
            "final_acceptance_nonces_rejected",
        )
        _require(type(raw_jobs) is list and len(raw_jobs) == 4, "final_acceptance_jobs_rejected")
        bindings: list[JobBinding] = []
        accepted: dict[str, AcceptedJobReceipt] = {}
        accepted_keys = set(AcceptedJobReceipt.__dataclass_fields__)
        for ordinal, (key, nonce, raw_job) in enumerate(
            zip(
                ("single_minimum", "single_latest", "two_minimum", "two_latest"),
                raw_nonces,
                raw_jobs,
            ),
            start=1,
        ):
            job = _object(raw_job, accepted_keys, f"final_acceptance_job_{ordinal}")
            try:
                receipt = AcceptedJobReceipt(**job)
            except TypeError:
                _fail("final_acceptance_job_schema_rejected")
            runner_name = f"{runtime.JOB_SPECS[key]['prefix']}{nonce}"
            _require(
                receipt.phase == "final-main"
                and receipt.run_id == run_id
                and receipt.job_key == key
                and receipt.runner_name == runner_name,
                "final_acceptance_job_binding_rejected",
            )
            bindings.append(
                JobBinding(
                    key,
                    ordinal,
                    receipt.job_id,
                    str(runtime.JOB_SPECS[key]["name"]),
                    nonce,
                    runner_name,
                )
            )
            accepted[key] = receipt
        dispatch = DispatchReceipt(
            observed_at=str(mapping["dispatch_nonce_history_observed_at"]),
            request_sha256="0" * 64,
            response_sha256="0" * 64,
            workflow_response_sha256="0" * 64,
            run_response_sha256="0" * 64,
            nonce_history_observed_at=str(mapping["dispatch_nonce_history_observed_at"]),
            nonce_history_response_sha256=str(mapping["dispatch_nonce_history_response_sha256"]),
            mutation_response_received=True,
            mutation_reconciliation_sha256="0" * 64,
        )
        session = PhaseSession(
            phase="final-main",
            workflow="cuda-ci.yml",
            workflow_path=runtime.CUDA_WORKFLOW_PATH,
            dispatch_ref="main",
            run_ref=runtime.FINAL_MAIN_REF,
            head_sha=head_sha,
            inputs=dict(zip(runtime.CUDA_NONCE_INPUT_KEYS, raw_nonces)),
            prior_accepted_cuda_runner_nonces=(),
            run={"id": run_id, "run_attempt": 1},
            jobs=tuple(bindings),
            queued_jobs=tuple(bindings),
            dispatch_receipt=dispatch,
            accepted=accepted,
        )
        rebuilt = cls._from_completed_session(session, mapping["settlement"])
        _require(
            rebuilt.to_mapping() == mapping,
            "final_acceptance_verified_mapping_drift",
        )
        if authority_evidence_identities is None:
            return rebuilt
        _require(
            type(authority_evidence_identities) in {list, tuple}
            and len(authority_evidence_identities) == 4,
            "final_acceptance_authority_identity_cardinality_rejected",
        )
        normalized_identities: list[dict[str, Any]] = [
            _validated_authority_evidence_identity(
                item,
                context=f"final_main_authority_identity_{ordinal}",
                expected_phase="final-main",
                expected_head_sha=head_sha,
                expected_run_id=run_id,
                expected_job_key=key,
            )
            for ordinal, (key, item) in enumerate(
                zip(
                    ("single_minimum", "single_latest", "two_minimum", "two_latest"),
                    authority_evidence_identities,
                ),
                start=1,
            )
        ]
        provenance = {
            "schema_version": 1,
            "kind": "explainiverse-final-main-journal-provenance",
            "acceptance_evidence_sha256": rebuilt.evidence_sha256,
            "final_journal_sha256": final_journal_sha256,
            "evidence_directory_receipt_sha256": (evidence_directory_receipt_sha256),
            "authority_evidence_identities": normalized_identities,
        }
        _require(
            len(normalized_identities) == 4
            and len({item["evidence_sha256"] for item in normalized_identities}) == 4,
            "final_acceptance_authority_identity_cardinality_rejected",
        )
        object.__setattr__(rebuilt, "_final_journal_sha256", final_journal_sha256)
        object.__setattr__(
            rebuilt,
            "_evidence_directory_receipt_sha256",
            evidence_directory_receipt_sha256,
        )
        object.__setattr__(rebuilt, "_journal_provenance_sha256", _sha(_canonical(provenance)))
        object.__setattr__(
            rebuilt,
            "_authority_evidence_identities",
            tuple(
                _json_mapping_copy(item, "final_main_authority_identity")
                for item in normalized_identities
            ),
        )
        return rebuilt

    def _journal_provenance_mapping(self) -> dict[str, Any]:
        _require(
            type(self._final_journal_sha256) is str
            and SHA256_RE.fullmatch(self._final_journal_sha256) is not None
            and type(self._evidence_directory_receipt_sha256) is str
            and SHA256_RE.fullmatch(self._evidence_directory_receipt_sha256) is not None
            and type(self._journal_provenance_sha256) is str
            and SHA256_RE.fullmatch(self._journal_provenance_sha256) is not None,
            "final_main_acceptance_not_loaded_from_journal",
        )
        _require(
            type(self._authority_evidence_identities) is tuple
            and len(self._authority_evidence_identities) == 4,
            "final_main_acceptance_authority_provenance_missing",
        )
        assert self._authority_evidence_identities is not None
        normalized_identities = [
            _validated_authority_evidence_identity(
                item,
                context=f"final_main_authority_provenance_{ordinal}",
                expected_phase="final-main",
                expected_head_sha=self.head_sha,
                expected_run_id=self.run_id,
                expected_job_key=key,
            )
            for ordinal, (key, item) in enumerate(
                zip(
                    ("single_minimum", "single_latest", "two_minimum", "two_latest"),
                    self._authority_evidence_identities,
                ),
                start=1,
            )
        ]
        material = {
            "schema_version": 1,
            "kind": "explainiverse-final-main-journal-provenance",
            "acceptance_evidence_sha256": self.evidence_sha256,
            "final_journal_sha256": self._final_journal_sha256,
            "evidence_directory_receipt_sha256": (self._evidence_directory_receipt_sha256),
            "authority_evidence_identities": normalized_identities,
        }
        _require(
            _sha(_canonical(material)) == self._journal_provenance_sha256,
            "final_main_acceptance_journal_provenance_drift",
        )
        return {
            **material,
            "journal_provenance_sha256": self._journal_provenance_sha256,
        }


class RemoteExecutor(Protocol):
    def wait_cloud_init(self, binding: live.StrictSshBinding) -> PublicSshResult: ...

    def probe_host(self, binding: live.StrictSshBinding) -> PublicSshResult: ...

    def run_job(
        self,
        binding: live.StrictSshBinding,
        canonical_plan: bytes,
        jit_config: live.SecretBuffer,
    ) -> RemoteExecution: ...


class SshRemoteExecutor:
    """Execute only the live adapter's fixed SSH argv, never a shell string."""

    def __init__(
        self,
        *,
        executable_path: str,
        executable_sha256: str,
        access_identity: live.AccessIdentityReceipt,
    ) -> None:
        path = Path(executable_path)
        _require(path.is_absolute(), "ssh_executable_path_not_absolute")
        _require(path == path.resolve(strict=True), "ssh_executable_path_not_canonical")
        _require(path.is_file() and not path.is_symlink(), "ssh_executable_file_rejected")
        _require(SHA256_RE.fullmatch(executable_sha256) is not None, "ssh_executable_digest")
        _require(_sha(path.read_bytes()) == executable_sha256, "ssh_executable_digest_mismatch")
        self._executable_path = path
        self._executable_sha256 = executable_sha256
        self._access_identity = access_identity

    def access_identity_receipt(self, *, expected_public_key_sha256: str) -> dict[str, Any]:
        validation = self._access_identity.validate(
            expected_public_key_sha256=expected_public_key_sha256
        )
        return {
            "capture": self._access_identity.to_public_mapping(),
            "validation": validation,
            "private_path_archived": False,
            "private_digest_archived": False,
        }

    def owns_access_identity(self, receipt: live.AccessIdentityReceipt) -> bool:
        return self._access_identity is receipt

    def close_access_identity(self) -> None:
        self._access_identity.close()

    def executable_receipt(self) -> dict[str, Any]:
        _require(
            self._executable_path.is_file()
            and not self._executable_path.is_symlink()
            and _sha(self._executable_path.read_bytes()) == self._executable_sha256,
            "ssh_executable_posture_drift",
        )
        return {
            "absolute_path": str(self._executable_path),
            "sha256": self._executable_sha256,
            "regular_file": True,
            "symlink": False,
            "path_lookup_used": False,
        }

    @staticmethod
    def _environment() -> dict[str, str]:
        allowed = {
            "SystemRoot",
            "WINDIR",
            "TEMP",
            "TMP",
            "LANG",
            "LC_ALL",
        }
        return {key: value for key, value in os.environ.items() if key in allowed}

    def _argv(self, binding: live.StrictSshBinding, expected_mode: str) -> list[str]:
        _require(binding.known_hosts_path is not None, "ssh_known_hosts_path_missing")
        known_hosts_path = binding.known_hosts_path
        assert known_hosts_path is not None
        try:
            known_host = binding.known_hosts.split(" ", 1)[0]
        except (AttributeError, IndexError):
            _fail("ssh_known_hosts_format_rejected")
        _validate_strict_ssh_binding_shape(
            binding,
            expected_mode=expected_mode,
            expected_public_ipv4=known_host,
            expected_host_fingerprint=binding.host_fingerprint,
            expected_known_hosts_path=known_hosts_path,
            expected_known_hosts_sha256=binding.known_hosts_sha256,
            expected_acl_receipt_sha256=str(binding.evidence_directory_acl_receipt_sha256),
        )
        argv = list(binding.argv_prefix)
        self.executable_receipt()
        self._access_identity.validate(
            expected_public_key_sha256=self._access_identity.public_key_sha256
        )
        identity_indexes = [index for index, item in enumerate(argv) if item == "-i"]
        _require(len(identity_indexes) == 1, "ssh_access_identity_option_rejected")
        identity_index = identity_indexes[0] + 1
        _require(
            identity_index < len(argv)
            and argv[identity_index] == self._access_identity.absolute_path,
            "ssh_access_identity_binding_drift",
        )
        argv[0] = str(self._executable_path)
        _require(
            tuple(argv[-len(binding.remote_command) :]) == binding.remote_command,
            "ssh_command_drift",
        )
        return argv

    @staticmethod
    def _validate_remote_argv(argv: Sequence[str], plan: Mapping[str, Any]) -> None:
        forbidden = [
            plan["job"]["runner_nonce"],
            plan["job"]["jit_config_sha256"],
            *plan["hardware"]["host_physical_gpu_uuids"],
        ]
        rendered_argv = "\n".join(argv)
        _require(
            not any(value in rendered_argv for value in forbidden),
            "secret_or_plan_value_in_ssh_argv",
        )

    def _run_public(
        self, binding: live.StrictSshBinding, expected_mode: str, timeout: int
    ) -> PublicSshResult:
        argv = self._argv(binding, expected_mode)
        try:
            completed = subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=SshRemoteExecutor._environment(),
                timeout=timeout,
                shell=False,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            _fail("ssh_public_transport_failure")
        _require(len(completed.stdout) <= MAX_RESPONSE_BYTES, "ssh_public_output_too_large")
        _require(len(completed.stderr) <= MAX_RESPONSE_BYTES, "ssh_public_error_too_large")
        return PublicSshResult(completed.stdout, completed.stderr, completed.returncode)

    def wait_cloud_init(self, binding: live.StrictSshBinding) -> PublicSshResult:
        return self._run_public(binding, "cloud-init", 900)

    def probe_host(self, binding: live.StrictSshBinding) -> PublicSshResult:
        return self._run_public(binding, "preflight", 1200)

    def run_job(
        self,
        binding: live.StrictSshBinding,
        canonical_plan: bytes,
        jit_config: live.SecretBuffer,
    ) -> RemoteExecution:
        argv = self._argv(binding, "run")
        plan = runtime.parse_plan_document(canonical_plan)
        self._validate_remote_argv(argv, plan)
        read_fd, write_fd = os.pipe()
        process: subprocess.Popen[bytes] | None = None
        frame: live.RuntimeFrameReceipt | None = None
        try:
            process = subprocess.Popen(
                argv,
                stdin=read_fd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self._environment(),
                shell=False,
                close_fds=True,
            )
            os.close(read_fd)
            read_fd = -1
            frame_fd = write_fd
            write_fd = -1
            frame = live.write_runtime_frame_and_close(
                frame_fd, canonical_plan=canonical_plan, jit_config=jit_config
            )
            try:
                stdout, stderr = process.communicate(timeout=runtime.HARD_WALL_SECONDS + 180)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                _fail("ssh_runtime_timeout")
        except BaseException as error:
            if process is not None and process.poll() is None:
                process.kill()
                try:
                    process.communicate(timeout=30)
                except subprocess.SubprocessError:
                    pass
            if isinstance(error, ControllerError):
                raise
            if isinstance(error, live.ContractError):
                _fail("ssh_runtime_frame_failure")
            if isinstance(error, (OSError, subprocess.SubprocessError)):
                _fail("ssh_runtime_transport_failure")
            raise
        finally:
            jit_config.destroy()
            if read_fd >= 0:
                os.close(read_fd)
            if write_fd >= 0:
                os.close(write_fd)
        _require(process is not None and process.returncode == 0, "ssh_runtime_failed")
        _require(len(stdout) <= MAX_RESPONSE_BYTES, "remote_receipt_too_large")
        value = _json(bytearray(stdout), "remote_receipt")
        _require(stdout == runtime.canonical_json(value), "remote_receipt_not_canonical")
        assert frame is not None
        return RemoteExecution(value, _sha(stdout), _sha(stderr), frame.to_public_mapping())


class ReleaseGpuController:
    """Stateful, sequential controller for PR, final-main, and publication phases."""

    def __init__(
        self,
        github: GitHubTransport,
        remote: RemoteExecutor,
        *,
        resources: SealedControllerResources,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        nonce_source: Callable[[], str] | None = None,
    ) -> None:
        _require(
            type(resources) is SealedControllerResources,
            "sealed_controller_resources_required",
        )
        self._resources = resources
        self._github = github
        self._remote = remote
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._sleep = sleep
        self._nonce_source = nonce_source or (lambda: secrets.token_hex(8))
        self._seen_nonces: set[str] = set()
        self._seen_job_ids: set[int] = set()
        self._seen_runner_ids: set[int] = set()
        self._seen_app_capture_sha256: set[str] = set()
        self._seen_app_page_sha256: set[str] = set()

    def _now(self) -> datetime:
        value = self._clock().astimezone(timezone.utc)
        _require(value.tzinfo is not None, "controller_clock_naive")
        return value

    @property
    def sealed_resources(self) -> SealedControllerResources:
        return self._resources

    def ssh_executable_receipt(self) -> dict[str, Any]:
        _require(
            type(self._remote) is SshRemoteExecutor,
            "production_ssh_executor_not_bound",
        )
        assert isinstance(self._remote, SshRemoteExecutor)
        return self._remote.executable_receipt()

    def github_executable_receipt(self) -> dict[str, Any]:
        _require(
            type(self._github) is GhCliTransport,
            "production_github_transport_not_bound",
        )
        assert isinstance(self._github, GhCliTransport)
        return self._github.executable_receipt()

    def bind_access_identity(
        self,
        receipt: live.AccessIdentityReceipt,
        *,
        expected_public_key_sha256: str,
    ) -> dict[str, Any]:
        _require(
            type(self._remote) is SshRemoteExecutor
            and type(receipt) is live.AccessIdentityReceipt
            and self._remote.owns_access_identity(receipt),
            "production_access_identity_not_bound",
        )
        assert isinstance(self._remote, SshRemoteExecutor)
        return self._remote.access_identity_receipt(
            expected_public_key_sha256=expected_public_key_sha256
        )

    def close_access_identity(self) -> None:
        _require(
            type(self._remote) is SshRemoteExecutor,
            "production_ssh_executor_not_bound",
        )
        assert isinstance(self._remote, SshRemoteExecutor)
        self._remote.close_access_identity()

    def policy_sha256(self) -> str:
        return self._resources.controller_source_sha256

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        expected: int,
    ) -> GitHubResponse:
        request_sha256 = _sha(_canonical({"method": method, "path": path, "body": body}))
        try:
            response = self._github.request(method, path, body)
        except AmbiguousGitHubMutation:
            raise
        except Exception:
            if method in {"POST", "DELETE"}:
                raise AmbiguousGitHubMutation(
                    method, path, request_sha256, "injected_transport_failure"
                ) from None
            raise
        if response.method != method or response.path != path:
            response.destroy()
            if method in {"POST", "DELETE"}:
                raise AmbiguousGitHubMutation(
                    method, path, request_sha256, "response_binding_drift"
                )
            _fail("github_response_binding_drift")
        if response.status_code != expected and method in {"POST", "DELETE"}:
            response.destroy()
            raise AmbiguousGitHubMutation(method, path, request_sha256, "unexpected_http_status")
        _require(response.status_code == expected, "github_status_rejected")
        return response

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        expected: int = 200,
    ) -> tuple[Any, str]:
        response = self._request(method, path, body=body, expected=expected)
        try:
            value = _json(response.body, "github_response")
            return value, _response_envelope_digest(response)
        finally:
            response.destroy()

    def _request_empty(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        expected: int,
    ) -> str:
        response = self._request(method, path, body=body, expected=expected)
        try:
            if bytes(response.body).strip() not in {b"", b"{}"}:
                if method in {"POST", "DELETE"}:
                    raise AmbiguousGitHubMutation(
                        method,
                        path,
                        _sha(_canonical({"method": method, "path": path, "body": body})),
                        "successful_response_body_unusable",
                        reconciliation={
                            "status_code": response.status_code,
                            "response_envelope_sha256": _response_envelope_digest(response),
                            "mutation_retried": False,
                        },
                    )
                _fail("github_empty_body_rejected")
            return _response_envelope_digest(response)
        finally:
            response.destroy()

    def _paginate(self, path: str, key: str | None) -> tuple[list[Any], list[str]]:
        result: list[Any] = []
        digests: list[str] = []
        declared_total: int | None = None
        page = 1
        while True:
            separator = "&" if "?" in path else "?"
            value, digest = self._request_json("GET", f"{path}{separator}per_page=100&page={page}")
            digests.append(digest)
            if key is None:
                items = value
            else:
                mapping = _required(value, {"total_count", key}, "paginated_response")
                _require(type(mapping["total_count"]) is int, "paginated_total_count_rejected")
                if declared_total is None:
                    declared_total = mapping["total_count"]
                _require(
                    mapping["total_count"] == declared_total,
                    "pagination_total_count_changed",
                )
                items = mapping[key]
            _require(type(items) is list, "paginated_items_not_list")
            result.extend(items)
            if len(items) < 100:
                break
            page += 1
            _require(page <= 10_000, "pagination_unbounded")
        if declared_total is not None:
            _require(len(result) == declared_total, "pagination_incomplete")
        return result, digests

    def _workflow(self, filename: str, expected_path: str) -> str:
        path = f"/repos/{REPOSITORY}/actions/workflows/{filename}"
        value, digest = self._request_json("GET", path)
        workflow = _required(value, {"id", "path", "state"}, "workflow")
        _positive(workflow["id"], "workflow_id")
        _require(
            workflow["path"] == expected_path and workflow["state"] == "active", "workflow_drift"
        )
        return digest

    def _all_runs(self, workflow: str) -> list[Mapping[str, Any]]:
        path = f"/repos/{REPOSITORY}/actions/workflows/{workflow}/runs?exclude_pull_requests=true"
        raw, _ = self._paginate(path, "workflow_runs")
        return [_required(item, {"id", "run_attempt"}, "historical_run") for item in raw]

    @staticmethod
    def _project_reviewed_attempt_jobs(
        raw_jobs: Sequence[Mapping[str, Any]], *, attempt: int
    ) -> list[Mapping[str, Any]]:
        """Remove only reviewed hosted companion jobs from an attempt page.

        The CUDA and publication workflows contain hosted control/downstream
        jobs in addition to the nonce-bound JIT matrix.  Those jobs are part of
        the immutable workflow, but they must not weaken exact JIT cardinality.
        This projection validates their complete live shape, requires the
        phase's control job, and leaves every unknown job in the result so the
        exact-set validator rejects it.
        """

        _positive(attempt, "attempt_job_projection_attempt")
        _require(
            type(raw_jobs) in {list, tuple},
            "attempt_job_projection_shape_rejected",
        )
        projected: list[Mapping[str, Any]] = []
        companions: dict[str, Mapping[str, Any]] = {}
        seen_ids: set[int] = set()
        known_target_names: set[str] = set()
        known_target_head_shas: set[str] = set()
        custom_target_present = False
        all_target_names = CUDA_TARGET_JOB_NAMES | PUBLICATION_TARGET_JOB_NAMES

        for raw in raw_jobs:
            job = _required(
                raw,
                {
                    "id",
                    "name",
                    "head_sha",
                    "run_attempt",
                    "status",
                    "conclusion",
                    "labels",
                    "runner_id",
                    "runner_name",
                },
                "attempt_projection_job",
            )
            job_id = _positive(job["id"], "attempt_projection_job_id")
            _require(job_id not in seen_ids, "attempt_projection_duplicate_job_id")
            seen_ids.add(job_id)
            name = job["name"]
            _require(type(name) is str and bool(name), "attempt_projection_job_name_rejected")

            if name not in REVIEWED_HOSTED_COMPANION_JOB_NAMES:
                projected.append(job)
                if name in all_target_names:
                    _commit(job["head_sha"], "attempt_projection_target_head_sha")
                    _require(
                        type(job["run_attempt"]) is int
                        and job["run_attempt"] == attempt
                        and type(job["labels"]) is list
                        and all(type(label) is str for label in job["labels"]),
                        "attempt_projection_target_binding_rejected",
                    )
                    known_target_names.add(name)
                    known_target_head_shas.add(job["head_sha"])
                    custom_target_present = custom_target_present or any(
                        RUNNER_NAME_RE.fullmatch(label) is not None for label in job["labels"]
                    )
                continue

            labels = job["labels"]
            runner_id = job["runner_id"]
            runner_name = job["runner_name"]
            _commit(job["head_sha"], "attempt_projection_companion_head_sha")
            _require(
                name not in companions
                and type(job["run_attempt"]) is int
                and job["run_attempt"] == attempt
                and type(job["status"]) is str
                and job["status"] in {"queued", "in_progress", "completed"}
                and (
                    job["conclusion"] is None
                    or (
                        type(job["conclusion"]) is str
                        and job["conclusion"] in RECOVERY_TERMINAL_CONCLUSIONS
                    )
                )
                and labels == [REVIEWED_HOSTED_COMPANION_LABELS[name]]
                and (runner_id is None or (type(runner_id) is int and 0 < runner_id < 2**63))
                and (
                    runner_name in {None, ""}
                    or (type(runner_name) is str and runner_name.startswith("GitHub Actions "))
                ),
                "attempt_projection_hosted_companion_rejected",
            )
            companions[name] = job

        cuda_targets = known_target_names & CUDA_TARGET_JOB_NAMES
        publication_targets = known_target_names & PUBLICATION_TARGET_JOB_NAMES
        _require(
            not (cuda_targets and publication_targets),
            "attempt_projection_mixed_workflow_targets_rejected",
        )
        allowed_companions: frozenset[str]
        if cuda_targets:
            allowed_companions = frozenset({CUDA_ROUTING_JOB_NAME})
            required_control_name = CUDA_ROUTING_JOB_NAME
        elif publication_targets:
            allowed_companions = REVIEWED_HOSTED_COMPANION_JOB_NAMES - {CUDA_ROUTING_JOB_NAME}
            required_control_name = PUBLICATION_PREFLIGHT_JOB_NAME
        else:
            allowed_companions = REVIEWED_HOSTED_COMPANION_JOB_NAMES
            required_control_name = None
        _require(
            set(companions).issubset(allowed_companions),
            "attempt_projection_cross_workflow_companion_rejected",
        )
        if required_control_name is not None:
            _require(
                required_control_name in companions
                and len(known_target_head_shas) == 1
                and all(item["head_sha"] in known_target_head_shas for item in companions.values()),
                "attempt_projection_control_companion_rejected",
            )
            if custom_target_present:
                control = companions[required_control_name]
                _require(
                    control["status"] == "completed"
                    and control["conclusion"] == "success"
                    and type(control["runner_id"]) is int
                    and control["runner_id"] > 0
                    and type(control["runner_name"]) is str
                    and control["runner_name"].startswith("GitHub Actions "),
                    "attempt_projection_control_not_accepted",
                )
        return projected

    def _attempt_jobs_with_digests(
        self, run_id: int, attempt: int
    ) -> tuple[list[Mapping[str, Any]], list[str]]:
        path = f"/repos/{REPOSITORY}/actions/runs/{run_id}/attempts/{attempt}/jobs" "?filter=all"
        raw, digests = self._paginate(path, "jobs")
        jobs = [
            _required(item, {"id", "name", "labels", "status"}, "historical_job") for item in raw
        ]
        return self._project_reviewed_attempt_jobs(jobs, attempt=attempt), digests

    def _attempt_jobs(self, run_id: int, attempt: int) -> list[Mapping[str, Any]]:
        jobs, _ = self._attempt_jobs_with_digests(run_id, attempt)
        return jobs

    @staticmethod
    def _validate_exact_attempt_job_set(
        jobs: Sequence[Mapping[str, Any]],
        *,
        expected_bindings: Sequence[tuple[str, str, int | None]],
        head_sha: str,
        context: str,
    ) -> tuple[Mapping[str, Any], ...]:
        """Validate the complete job set for one immutable attempt.

        GitHub's attempt jobs endpoint is evidence-bearing throughout this
        controller.  Matching only the expected names would let an injected
        fifth job run beside the protected matrix.  Every caller therefore
        supplies the exact expected name, requested runner label, and (once
        known) job id; this helper rejects any extra, missing, duplicate, or
        cross-attempt record before returning jobs in policy order.
        """

        _commit(head_sha, f"{context}_head_sha")
        _require(
            type(jobs) in {list, tuple}
            and type(expected_bindings) in {list, tuple}
            and bool(expected_bindings),
            f"{context}_shape_rejected",
        )
        expected_names = [item[0] for item in expected_bindings]
        expected_labels = [item[1] for item in expected_bindings]
        expected_ids = [item[2] for item in expected_bindings]
        _require(
            len(expected_names) == len(set(expected_names))
            and len(expected_labels) == len(set(expected_labels))
            and all(type(item) is str and bool(item) for item in expected_names)
            and all(
                type(item) is str and RUNNER_NAME_RE.fullmatch(item) is not None
                for item in expected_labels
            )
            and all(
                item is None or (type(item) is int and 0 < item < 2**63) for item in expected_ids
            )
            and len([item for item in expected_ids if item is not None])
            == len(set(item for item in expected_ids if item is not None)),
            f"{context}_expected_binding_rejected",
        )
        _require(len(jobs) == len(expected_bindings), f"{context}_cardinality_rejected")
        normalized: dict[str, Mapping[str, Any]] = {}
        seen_ids: set[int] = set()
        for raw in jobs:
            job = _required(
                raw,
                {
                    "id",
                    "name",
                    "head_sha",
                    "run_attempt",
                    "status",
                    "conclusion",
                    "labels",
                    "runner_id",
                    "runner_name",
                },
                f"{context}_job",
            )
            job_id = _positive(job["id"], f"{context}_job_id")
            name = job["name"]
            labels = job["labels"]
            runner_id = job["runner_id"]
            runner_name = job["runner_name"]
            _require(
                type(name) is str
                and name in expected_names
                and name not in normalized
                and job_id not in seen_ids
                and job["head_sha"] == head_sha
                and type(job["run_attempt"]) is int
                and job["run_attempt"] == 1
                and type(job["status"]) is str
                and job["status"] in {"queued", "in_progress", "completed"}
                and (
                    job["conclusion"] is None
                    or (
                        type(job["conclusion"]) is str
                        and job["conclusion"] in RECOVERY_TERMINAL_CONCLUSIONS
                    )
                )
                and type(labels) is list
                and len(labels) == 1
                and type(labels[0]) is str
                and labels[0] == expected_labels[expected_names.index(name)]
                and (runner_id is None or (type(runner_id) is int and 0 < runner_id < 2**63))
                and (
                    runner_name in {None, ""}
                    or (
                        type(runner_name) is str
                        and runner_name == expected_labels[expected_names.index(name)]
                    )
                ),
                f"{context}_job_binding_rejected",
            )
            normalized[name] = job
            seen_ids.add(job_id)
        ordered = tuple(normalized[name] for name in expected_names)
        for index, expected_id in enumerate(expected_ids):
            if expected_id is not None:
                _require(
                    ordered[index]["id"] == expected_id,
                    f"{context}_job_id_drift",
                )
        return ordered

    @staticmethod
    def _session_expected_job_bindings(
        session: PhaseSession,
    ) -> tuple[tuple[str, str, int | None], ...]:
        return tuple(
            (binding.name, binding.runner_name, binding.job_id) for binding in session.queued_jobs
        )

    def _nonce_history(
        self,
        nonces: Sequence[str],
        *,
        exclude: tuple[int, int, int] | None = None,
        allowed_active_job_ids: set[int] | None = None,
    ) -> dict[str, Any]:
        _require(
            bool(nonces) and all(NONCE_RE.fullmatch(value) for value in nonces),
            "nonce_history_input",
        )
        nonce_set = set(nonces)
        matches: list[dict[str, int]] = []
        unexpected_active: list[int] = []
        seen_job_ids: set[int] = set()
        page_material: list[dict[str, Any]] = []
        allowed = allowed_active_job_ids or set()
        for workflow in ("cuda-ci.yml", "publish-pypi.yml"):
            for run in self._all_runs(workflow):
                run_id = _positive(run["id"], "historical_run_id")
                attempts = _positive(run["run_attempt"], "historical_run_attempt")
                for attempt in range(1, attempts + 1):
                    for job in self._attempt_jobs(run_id, attempt):
                        job_id = _positive(job["id"], "historical_job_id")
                        _require(job_id not in seen_job_ids, "duplicate_historical_job_id")
                        seen_job_ids.add(job_id)
                        labels = job["labels"]
                        _require(
                            type(labels) is list and all(type(x) is str for x in labels),
                            "historical_labels",
                        )
                        runner_name = job.get("runner_name")
                        searchable = set(labels)
                        if type(runner_name) is str:
                            searchable.add(runner_name)
                        for nonce in nonce_set:
                            if any(nonce in value for value in searchable):
                                if exclude != (run_id, attempt, job_id):
                                    matches.append(
                                        {"run_id": run_id, "attempt": attempt, "job_id": job_id}
                                    )
                        if job["status"] in {"queued", "in_progress"}:
                            target = any(RUNNER_NAME_RE.fullmatch(value) for value in searchable)
                            if target and job_id not in allowed:
                                unexpected_active.append(job_id)
                        page_material.append(
                            {
                                "workflow": workflow,
                                "run_id": run_id,
                                "attempt": attempt,
                                "job_id": job_id,
                                "status": job["status"],
                                "labels": labels,
                                "runner_name": runner_name,
                            }
                        )
        return {
            "observed_at": _iso(self._now()),
            "response_sha256": _sha(_canonical(page_material)),
            "historical_match_count": len(matches),
            "unexpected_queued_or_in_progress_count": len(set(unexpected_active)),
        }

    def _fresh_nonces(self, count: int, prior: Sequence[str]) -> tuple[str, ...]:
        result: list[str] = []
        for _ in range(count):
            nonce = self._nonce_source()
            _require(
                type(nonce) is str and NONCE_RE.fullmatch(nonce) is not None,
                "nonce_source_rejected",
            )
            _require(
                nonce not in self._seen_nonces and nonce not in prior and nonce not in result,
                "nonce_reuse",
            )
            result.append(nonce)
        return tuple(result)

    def _main_sha(self) -> tuple[str, str]:
        value, digest = self._request_json("GET", f"/repos/{REPOSITORY}/git/ref/heads/main")
        ref = _required(value, {"ref", "object"}, "main_ref")
        obj = _required(ref["object"], {"type", "sha"}, "main_ref_object")
        _require(ref["ref"] == "refs/heads/main" and obj["type"] == "commit", "main_ref_drift")
        return _commit(obj["sha"], "main_sha"), digest

    def _validate_pull_request_source(self, head_sha: str) -> dict[str, str]:
        """Bind the dispatch to the one open same-repository PR #4."""

        main_sha, main_digest = self._main_sha()
        value, pull_digest = self._request_json("GET", f"/repos/{REPOSITORY}/pulls/4")
        pull = _required(
            value,
            {
                "number",
                "state",
                "draft",
                "base",
                "head",
                "mergeable",
                "mergeable_state",
            },
            "pull_request_4",
        )
        base = _required(pull["base"], {"ref", "sha", "repo"}, "pull_request_base")
        head = _required(pull["head"], {"ref", "sha", "repo"}, "pull_request_head")
        base_repo = _required(base["repo"], {"full_name"}, "pull_request_base_repo")
        head_repo = _required(head["repo"], {"full_name"}, "pull_request_head_repo")
        _require(
            pull["number"] == 4
            and pull["state"] == "open"
            and pull["draft"] is False
            and base["ref"] == "main"
            and base["sha"] == main_sha
            and base_repo["full_name"] == REPOSITORY
            and head["ref"] == runtime.PULL_REQUEST_REF.removeprefix("refs/heads/")
            and head["sha"] == head_sha
            and head_repo["full_name"] == REPOSITORY
            and pull["mergeable"] is True
            and pull["mergeable_state"] == "blocked",
            "pull_request_4_source_drift",
        )
        return {"pull_request": pull_digest, "main": main_digest}

    def _validate_publication_source(self, head_sha: str) -> dict[str, str]:
        ref_value, ref_digest = self._request_json(
            "GET", f"/repos/{REPOSITORY}/git/ref/tags/{runtime.PUBLICATION_TAG}"
        )
        ref = _required(ref_value, {"ref", "object"}, "publication_ref")
        obj = _required(ref["object"], {"type", "sha"}, "publication_ref_object")
        _require(
            ref["ref"] == runtime.PUBLICATION_REF and obj["type"] == "tag",
            "publication_tag_not_annotated",
        )
        tag_sha = _commit(obj["sha"], "publication_tag_object_sha")
        tag_value, tag_digest = self._request_json("GET", f"/repos/{REPOSITORY}/git/tags/{tag_sha}")
        tag = _required(
            tag_value, {"tag", "object", "verification", "tagger", "message"}, "tag_object"
        )
        target = _required(tag["object"], {"type", "sha"}, "tag_target")
        verification = _required(
            tag["verification"], {"verified", "reason", "signature"}, "tag_verification"
        )
        _require(
            tag["tag"] == runtime.PUBLICATION_TAG
            and target["type"] == "commit"
            and target["sha"] == head_sha
            and verification["verified"] is True
            and verification["reason"] == "valid"
            and type(verification["signature"]) is str
            and bool(verification["signature"].strip()),
            "publication_tag_signature_rejected",
        )
        main_sha, main_digest = self._main_sha()
        _require(main_sha == head_sha, "publication_tag_not_final_main")
        return {"ref": ref_digest, "tag": tag_digest, "main": main_digest}

    @staticmethod
    def _recovery_display_title(tag: str, source_run_id: int, recovery_request_nonce: str) -> str:
        return f"{RECOVERY_RUN_TITLE_PREFIX}-{tag}-{source_run_id}-" f"{recovery_request_nonce}"

    def _validate_recovery_source_run(
        self, source_run_id: int, head_sha: str
    ) -> tuple[dict[str, Any], str]:
        value, _ = self._request_json("GET", f"/repos/{REPOSITORY}/actions/runs/{source_run_id}")
        run = _required(
            value,
            {
                "id",
                "repository",
                "path",
                "event",
                "head_sha",
                "head_branch",
                "status",
                "conclusion",
                "run_attempt",
                "actor",
                "triggering_actor",
            },
            "recovery_source_run",
        )
        actor = _required(run["actor"], {"login"}, "recovery_source_actor")
        triggering_actor = _required(
            run["triggering_actor"], {"login"}, "recovery_source_triggering_actor"
        )
        _require(
            run["id"] == source_run_id
            and actor["login"] == OWNER
            and triggering_actor["login"] == OWNER,
            "recovery_source_run_authority_rejected",
        )
        jobs, _ = self._paginate(
            f"/repos/{REPOSITORY}/actions/runs/{source_run_id}/jobs?filter=all",
            "jobs",
        )
        try:
            verified = verify_source_run_evidence(
                run,
                {
                    "query_filter": "all",
                    "pagination_complete": True,
                    "jobs": jobs,
                },
                repository=REPOSITORY,
                workflow_path=runtime.PUBLISH_WORKFLOW_PATH,
                release_tag=runtime.PUBLICATION_TAG,
                release_commit=head_sha,
            )
        except (TypeError, ValueError):
            _fail("recovery_source_run_evidence_rejected")
        _require(
            verified.get("source_kind") == "staged_drill",
            "recovery_source_not_staged_drill",
        )
        material = {
            "source_run_id": source_run_id,
            "actor": actor["login"],
            "triggering_actor": triggering_actor["login"],
            "verified": verified,
        }
        return material, _sha(_canonical(material))

    def _recovery_run_history(
        self, *, tag: str, head_sha: str, source_run_id: int
    ) -> tuple[list[dict[str, Any]], str]:
        raw, page_digests = self._paginate(
            f"/repos/{REPOSITORY}/actions/workflows/{RECOVERY_WORKFLOW}/runs"
            "?event=workflow_dispatch",
            "workflow_runs",
        )
        title_pattern = re.compile(
            rf"{re.escape(RECOVERY_RUN_TITLE_PREFIX)}-{re.escape(tag)}-"
            rf"{source_run_id}-([0-9a-f]{{16}})\Z"
        )
        result: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        for value in raw:
            run = _required(
                value,
                {
                    "id",
                    "display_title",
                    "path",
                    "head_sha",
                    "head_branch",
                    "event",
                    "run_attempt",
                    "status",
                    "conclusion",
                    "actor",
                    "triggering_actor",
                },
                "recovery_history_run",
            )
            if not (
                run["path"] == RECOVERY_WORKFLOW_PATH
                and run["head_sha"] == head_sha
                and run["head_branch"] == tag
                and run["event"] == "workflow_dispatch"
            ):
                continue
            match = title_pattern.fullmatch(str(run["display_title"]))
            _require(match is not None, "recovery_history_foreign_run_present")
            assert match is not None
            actor = _required(run["actor"], {"login"}, "recovery_history_actor")
            triggering_actor = _required(
                run["triggering_actor"],
                {"login"},
                "recovery_history_triggering_actor",
            )
            run_id = _positive(run["id"], "recovery_history_run_id")
            _require(
                run_id not in seen_ids
                and type(run["run_attempt"]) is int
                and run["run_attempt"] == 1
                and actor["login"] == OWNER
                and triggering_actor["login"] == OWNER
                and type(run["status"]) is str
                and run["status"] in {"queued", "in_progress", "completed"}
                and (
                    (
                        run["status"] == "completed"
                        and type(run["conclusion"]) is str
                        and run["conclusion"] in RECOVERY_TERMINAL_CONCLUSIONS
                    )
                    or (run["status"] != "completed" and run["conclusion"] is None)
                ),
                "recovery_history_run_drift",
            )
            seen_ids.add(run_id)
            result.append(
                {
                    "id": run_id,
                    "display_title": run["display_title"],
                    "head_sha": run["head_sha"],
                    "head_branch": run["head_branch"],
                    "run_attempt": 1,
                    "status": run["status"],
                    "conclusion": run["conclusion"],
                    "actor": actor["login"],
                    "triggering_actor": triggering_actor["login"],
                    "recovery_request_nonce": match.group(1),
                }
            )
        result.sort(key=lambda item: int(item["id"]))
        return result, _sha(_canonical({"page_digests": page_digests, "matching_runs": result}))

    @staticmethod
    def _validate_recovery_dispatch_intent(value: Mapping[str, Any]) -> dict[str, Any]:
        intent = _object(
            value,
            {
                "schema_version",
                "kind",
                "repository",
                "workflow",
                "workflow_path",
                "ref",
                "head_sha",
                "tag",
                "source_run_id",
                "require_staged_drill",
                "recovery_request_nonce",
                "display_title",
                "request_path",
                "request_body",
                "request_sha256",
                "workflow_response_sha256",
                "immutable_source_evidence_sha256",
                "source_run_evidence_sha256",
                "pre_dispatch_run_ids",
                "pre_dispatch_runs",
                "pre_dispatch_history_sha256",
                "mutation_retried",
            },
            "recovery_dispatch_intent",
        )
        source_run_id = _positive(intent["source_run_id"], "recovery_dispatch_source_run_id")
        nonce = intent["recovery_request_nonce"]
        _require(
            type(intent["schema_version"]) is int
            and intent["schema_version"] == 1
            and intent["kind"] == "explainiverse-recovery-dispatch-intent"
            and intent["repository"] == REPOSITORY
            and intent["workflow"] == RECOVERY_WORKFLOW
            and intent["workflow_path"] == RECOVERY_WORKFLOW_PATH
            and intent["ref"] == runtime.PUBLICATION_TAG
            and intent["tag"] == runtime.PUBLICATION_TAG
            and type(nonce) is str
            and NONCE_RE.fullmatch(nonce) is not None
            and intent["require_staged_drill"] is True
            and intent["mutation_retried"] is False,
            "recovery_dispatch_intent_binding_rejected",
        )
        head_sha = _commit(intent["head_sha"], "recovery_dispatch_head_sha")
        display_title = ReleaseGpuController._recovery_display_title(
            runtime.PUBLICATION_TAG, source_run_id, nonce
        )
        request_path = f"/repos/{REPOSITORY}/actions/workflows/{RECOVERY_WORKFLOW}/dispatches"
        request_body = {
            "ref": runtime.PUBLICATION_TAG,
            "inputs": {
                "tag": runtime.PUBLICATION_TAG,
                "source_run_id": str(source_run_id),
                "recovery_request_nonce": nonce,
                "require_staged_drill": True,
            },
        }
        raw_pre_ids = intent["pre_dispatch_run_ids"]
        raw_pre_runs = intent["pre_dispatch_runs"]
        _require(
            intent["display_title"] == display_title
            and intent["request_path"] == request_path
            and intent["request_body"] == request_body
            and intent["request_sha256"]
            == _sha(_canonical({"method": "POST", "path": request_path, "body": request_body}))
            and type(raw_pre_ids) is list
            and all(type(item) is int and item > 0 for item in raw_pre_ids)
            and raw_pre_ids == sorted(raw_pre_ids)
            and len(set(raw_pre_ids)) == len(raw_pre_ids)
            and type(raw_pre_runs) is list
            and len(raw_pre_runs) == len(raw_pre_ids)
            and all(
                type(intent[field]) is str and SHA256_RE.fullmatch(intent[field]) is not None
                for field in (
                    "request_sha256",
                    "workflow_response_sha256",
                    "immutable_source_evidence_sha256",
                    "source_run_evidence_sha256",
                    "pre_dispatch_history_sha256",
                )
            ),
            "recovery_dispatch_intent_integrity_rejected",
        )
        assert isinstance(raw_pre_runs, list)
        normalized_pre_runs: list[dict[str, Any]] = []
        for index, raw_run in enumerate(raw_pre_runs, start=1):
            run = _object(
                raw_run,
                {
                    "id",
                    "display_title",
                    "head_sha",
                    "head_branch",
                    "run_attempt",
                    "status",
                    "conclusion",
                    "actor",
                    "triggering_actor",
                    "recovery_request_nonce",
                },
                f"recovery_dispatch_prior_run_{index}",
            )
            _positive(
                run["id"],
                f"recovery_dispatch_prior_run_{index}_id",
            )
            prior_nonce = run["recovery_request_nonce"]
            _require(
                type(prior_nonce) is str
                and NONCE_RE.fullmatch(prior_nonce) is not None
                and run["display_title"]
                == ReleaseGpuController._recovery_display_title(
                    runtime.PUBLICATION_TAG,
                    source_run_id,
                    prior_nonce,
                )
                and run["head_sha"] == head_sha
                and run["head_branch"] == runtime.PUBLICATION_TAG
                and type(run["run_attempt"]) is int
                and run["run_attempt"] == 1
                and run["status"] == "completed"
                and run["conclusion"] == "failure"
                and run["actor"] == OWNER
                and run["triggering_actor"] == OWNER,
                "recovery_dispatch_prior_run_not_exact_failure",
            )
            normalized_pre_runs.append(dict(run))
        _require(
            [item["id"] for item in normalized_pre_runs] == raw_pre_ids
            and len({item["recovery_request_nonce"] for item in normalized_pre_runs})
            == len(normalized_pre_runs),
            "recovery_dispatch_prior_runs_binding_rejected",
        )
        normalized = dict(intent)
        normalized["head_sha"] = head_sha
        normalized["pre_dispatch_runs"] = normalized_pre_runs
        return normalized

    @staticmethod
    def _build_recovery_dispatch_receipt(
        intent: Mapping[str, Any],
        run: Mapping[str, Any],
        *,
        observed_at: str,
        mutation_response_received: bool,
        mutation_response_sha256: str | None,
        reconciliation_history_sha256: str,
    ) -> RecoveryDispatchReceipt:
        material = {
            "observed_at": observed_at,
            "tag": intent["tag"],
            "head_sha": intent["head_sha"],
            "source_run_id": intent["source_run_id"],
            "require_staged_drill": True,
            "recovery_request_nonce": intent["recovery_request_nonce"],
            "display_title": intent["display_title"],
            "run_id": run["id"],
            "run_attempt": 1,
            "status": run["status"],
            "conclusion": run["conclusion"],
            "request_sha256": intent["request_sha256"],
            "mutation_response_received": mutation_response_received,
            "mutation_response_sha256": mutation_response_sha256,
            "workflow_response_sha256": intent["workflow_response_sha256"],
            "immutable_source_evidence_sha256": intent["immutable_source_evidence_sha256"],
            "source_run_evidence_sha256": intent["source_run_evidence_sha256"],
            "pre_dispatch_history_sha256": intent["pre_dispatch_history_sha256"],
            "reconciliation_sha256": _sha(
                _canonical(
                    {
                        "run": run,
                        "reconciliation_history_sha256": (reconciliation_history_sha256),
                        "mutation_response_received": mutation_response_received,
                        "mutation_response_sha256": mutation_response_sha256,
                    }
                )
            ),
        }
        return RecoveryDispatchReceipt.from_mapping(
            {**material, "evidence_sha256": _sha(_canonical(material))}
        )

    @staticmethod
    def _expected_recovery_history_from_source(
        recovery_source: object,
        *,
        head_sha: str,
        source_run_id: int,
        required_tail_state: str,
        pending_intent: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Recover the exact prior-run set sealed by the publication journal.

        The local import avoids a module import cycle: the durable journal
        driver imports this controller, while this point-of-use check needs the
        driver's loader-only source type.  A fresh dispatch is never authorized
        from live GitHub history alone.
        """

        from .driver import PublicationRecoverySource

        _require(
            required_tail_state in {"complete", "pending-intent"}
            and (required_tail_state == "pending-intent") == (pending_intent is not None),
            "recovery_history_tail_requirement_rejected",
        )
        _require(
            type(recovery_source) is PublicationRecoverySource,
            "recovery_history_source_loader_proof_missing",
        )
        assert isinstance(recovery_source, PublicationRecoverySource)
        source_mapping = recovery_source.to_mapping()
        source_material = dict(source_mapping)
        source_evidence_sha256 = source_material.pop("evidence_sha256", None)
        _require(
            type(source_evidence_sha256) is str
            and _sha(_canonical(source_material)) == source_evidence_sha256
            and recovery_source.head_sha == head_sha
            and recovery_source.run_id == source_run_id,
            "recovery_history_source_binding_rejected",
        )
        tail = recovery_source.recovery_tail
        tail_mapping = tail.to_mapping()
        tail_material = dict(tail_mapping)
        tail_evidence_sha256 = tail_material.pop("evidence_sha256", None)
        _require(
            type(tail_evidence_sha256) is str
            and _sha(_canonical(tail_material)) == tail_evidence_sha256
            and tail.source_evidence_sha256 == source_evidence_sha256
            and tail.state == required_tail_state
            and tail.pending_intent == pending_intent
            and tail.pending_operator_settlement is None
            and len(tail.completed_run_ids) == len(tail.completed_request_nonces)
            and list(tail.completed_run_ids) == sorted(tail.completed_run_ids),
            "recovery_history_tail_binding_rejected",
        )
        expected: list[dict[str, Any]] = []
        for run_id, nonce in zip(
            tail.completed_run_ids,
            tail.completed_request_nonces,
        ):
            expected.append(
                {
                    "id": run_id,
                    "display_title": ReleaseGpuController._recovery_display_title(
                        runtime.PUBLICATION_TAG,
                        source_run_id,
                        nonce,
                    ),
                    "head_sha": head_sha,
                    "head_branch": runtime.PUBLICATION_TAG,
                    "run_attempt": 1,
                    "status": "completed",
                    "conclusion": "failure",
                    "actor": OWNER,
                    "triggering_actor": OWNER,
                    "recovery_request_nonce": nonce,
                }
            )
        return expected

    def dispatch_release_recovery(
        self,
        *,
        head_sha: str,
        source_run_id: int,
        recovery_request_nonce: str,
        recovery_source: object,
        progress: ProgressSink,
        poll_limit: int = 60,
    ) -> RecoveryDispatchReceipt:
        """Dispatch the sole staged recovery with durable, nonce-bound evidence."""

        _require(callable(progress), "recovery_dispatch_progress_sink_required")
        _require(type(poll_limit) is int and poll_limit > 0, "recovery_poll_limit_rejected")
        head_sha = _commit(head_sha, "recovery_head_sha")
        source_run_id = _positive(source_run_id, "recovery_source_run_id")
        expected_history = self._expected_recovery_history_from_source(
            recovery_source,
            head_sha=head_sha,
            source_run_id=source_run_id,
            required_tail_state="complete",
        )
        _require(
            type(recovery_request_nonce) is str
            and NONCE_RE.fullmatch(recovery_request_nonce) is not None,
            "recovery_request_nonce_rejected",
        )
        immutable_source = self._validate_publication_source(head_sha)
        immutable_source_evidence_sha256 = _sha(_canonical(immutable_source))
        _, source_run_evidence_sha256 = self._validate_recovery_source_run(source_run_id, head_sha)
        workflow_response_sha256 = self._workflow(RECOVERY_WORKFLOW, RECOVERY_WORKFLOW_PATH)
        history, pre_dispatch_history_sha256 = self._recovery_run_history(
            tag=runtime.PUBLICATION_TAG,
            head_sha=head_sha,
            source_run_id=source_run_id,
        )
        _require(
            history == expected_history,
            "recovery_live_history_not_exact_journal_tail",
        )
        _require(
            all(item["recovery_request_nonce"] != recovery_request_nonce for item in history),
            "recovery_request_nonce_reused",
        )
        display_title = self._recovery_display_title(
            runtime.PUBLICATION_TAG, source_run_id, recovery_request_nonce
        )
        request_path = f"/repos/{REPOSITORY}/actions/workflows/{RECOVERY_WORKFLOW}/dispatches"
        request_body = {
            "ref": runtime.PUBLICATION_TAG,
            "inputs": {
                "tag": runtime.PUBLICATION_TAG,
                "source_run_id": str(source_run_id),
                "recovery_request_nonce": recovery_request_nonce,
                "require_staged_drill": True,
            },
        }
        request_sha256 = _sha(
            _canonical({"method": "POST", "path": request_path, "body": request_body})
        )
        intent = self._validate_recovery_dispatch_intent(
            {
                "schema_version": 1,
                "kind": "explainiverse-recovery-dispatch-intent",
                "repository": REPOSITORY,
                "workflow": RECOVERY_WORKFLOW,
                "workflow_path": RECOVERY_WORKFLOW_PATH,
                "ref": runtime.PUBLICATION_TAG,
                "head_sha": head_sha,
                "tag": runtime.PUBLICATION_TAG,
                "source_run_id": source_run_id,
                "require_staged_drill": True,
                "recovery_request_nonce": recovery_request_nonce,
                "display_title": display_title,
                "request_path": request_path,
                "request_body": request_body,
                "request_sha256": request_sha256,
                "workflow_response_sha256": workflow_response_sha256,
                "immutable_source_evidence_sha256": (immutable_source_evidence_sha256),
                "source_run_evidence_sha256": source_run_evidence_sha256,
                "pre_dispatch_run_ids": [item["id"] for item in history],
                "pre_dispatch_runs": history,
                "pre_dispatch_history_sha256": pre_dispatch_history_sha256,
                "mutation_retried": False,
            }
        )
        _progress(progress, "github-recovery-dispatch-intent", intent)
        response_sha256: str | None = None
        response_received = True
        initial_ambiguity: AmbiguousGitHubMutation | None = None
        try:
            response_sha256 = self._request_empty(
                "POST", request_path, body=request_body, expected=204
            )
        except AmbiguousGitHubMutation as exc:
            _require(
                exc.method == "POST"
                and exc.path == request_path
                and exc.request_sha256 == request_sha256,
                "recovery_dispatch_ambiguity_binding_rejected",
            )
            response_received = False
            initial_ambiguity = exc
        pre_ids = set(intent["pre_dispatch_run_ids"])
        expected_prior_runs = intent["pre_dispatch_runs"]
        for _ in range(poll_limit):
            current, history_sha256 = self._recovery_run_history(
                tag=runtime.PUBLICATION_TAG,
                head_sha=head_sha,
                source_run_id=source_run_id,
            )
            candidates = [
                item
                for item in current
                if item["id"] not in pre_ids
                and item["display_title"] == display_title
                and item["recovery_request_nonce"] == recovery_request_nonce
            ]
            new_runs = [item for item in current if item["id"] not in pre_ids]
            current_prior_runs = [item for item in current if item["id"] in pre_ids]
            if (
                current_prior_runs == expected_prior_runs
                and len(candidates) == 1
                and len(new_runs) == 1
            ):
                receipt = self._build_recovery_dispatch_receipt(
                    intent,
                    candidates[0],
                    observed_at=_iso(self._now()),
                    mutation_response_received=response_received,
                    mutation_response_sha256=response_sha256,
                    reconciliation_history_sha256=history_sha256,
                )
                _progress(
                    progress,
                    "github-recovery-dispatch-settled",
                    receipt.to_mapping(),
                )
                return receipt
            if len(candidates) > 1:
                break
            self._sleep(2)
        raise AmbiguousGitHubMutation(
            "POST",
            request_path,
            request_sha256,
            "recovery_dispatch_visibility_unresolved",
            reconciliation={
                "display_title": display_title,
                "head_sha": head_sha,
                "source_run_id": source_run_id,
                "recovery_request_nonce": recovery_request_nonce,
                "pre_dispatch_run_ids": sorted(pre_ids),
                "response_received": response_received,
                "response_sha256": response_sha256,
                "initial_ambiguity": (
                    initial_ambiguity.to_public_mapping() if initial_ambiguity is not None else None
                ),
                "mutation_retried": False,
            },
        )

    def reconcile_release_recovery_dispatch(
        self,
        intent_mapping: Mapping[str, Any],
        *,
        recovery_source: object,
        poll_limit: int = 60,
        progress: ProgressSink | None = None,
    ) -> RecoveryDispatchReceipt:
        """Observe an unresolved dispatch intent without ever replaying its POST."""

        _require(type(poll_limit) is int and poll_limit > 0, "recovery_poll_limit_rejected")
        intent = self._validate_recovery_dispatch_intent(intent_mapping)
        head_sha = intent["head_sha"]
        source_run_id = intent["source_run_id"]
        assert isinstance(head_sha, str)
        assert type(source_run_id) is int
        expected_history = self._expected_recovery_history_from_source(
            recovery_source,
            head_sha=head_sha,
            source_run_id=source_run_id,
            required_tail_state="pending-intent",
            pending_intent=intent,
        )
        _require(
            intent["pre_dispatch_runs"] == expected_history
            and intent["pre_dispatch_run_ids"] == [item["id"] for item in expected_history],
            "recovery_dispatch_intent_not_exact_journal_tail",
        )
        immutable_source = self._validate_publication_source(head_sha)
        _require(
            _sha(_canonical(immutable_source)) == intent["immutable_source_evidence_sha256"],
            "recovery_dispatch_immutable_source_drift",
        )
        _, source_evidence_sha256 = self._validate_recovery_source_run(source_run_id, head_sha)
        _require(
            source_evidence_sha256 == intent["source_run_evidence_sha256"]
            and self._workflow(RECOVERY_WORKFLOW, RECOVERY_WORKFLOW_PATH)
            == intent["workflow_response_sha256"],
            "recovery_dispatch_source_evidence_drift",
        )
        pre_ids = set(intent["pre_dispatch_run_ids"])
        expected_prior_runs = intent["pre_dispatch_runs"]
        for _ in range(poll_limit):
            current, history_sha256 = self._recovery_run_history(
                tag=runtime.PUBLICATION_TAG,
                head_sha=head_sha,
                source_run_id=source_run_id,
            )
            candidates = [
                item
                for item in current
                if item["id"] not in pre_ids
                and item["display_title"] == intent["display_title"]
                and item["recovery_request_nonce"] == intent["recovery_request_nonce"]
            ]
            new_runs = [item for item in current if item["id"] not in pre_ids]
            current_prior_runs = [item for item in current if item["id"] in pre_ids]
            if (
                current_prior_runs == expected_prior_runs
                and len(candidates) == 1
                and len(new_runs) == 1
            ):
                receipt = self._build_recovery_dispatch_receipt(
                    intent,
                    candidates[0],
                    observed_at=_iso(self._now()),
                    mutation_response_received=False,
                    mutation_response_sha256=None,
                    reconciliation_history_sha256=history_sha256,
                )
                _progress(
                    progress,
                    "github-recovery-dispatch-settled",
                    receipt.to_mapping(),
                )
                return receipt
            _require(
                len(candidates) == 0 and len(new_runs) == 0,
                "recovery_dispatch_reconciliation_ambiguous",
            )
            self._sleep(2)
        raise AmbiguousGitHubMutation(
            "POST",
            str(intent["request_path"]),
            str(intent["request_sha256"]),
            "recovery_dispatch_absence_not_proof",
            reconciliation={
                "display_title": intent["display_title"],
                "head_sha": head_sha,
                "source_run_id": source_run_id,
                "recovery_request_nonce": intent["recovery_request_nonce"],
                "pre_dispatch_run_ids": sorted(pre_ids),
                "mutation_retried": False,
            },
        )

    def _validate_positive_run(self, run_id: int, path: str, head_sha: str, context: str) -> str:
        value, digest = self._request_json("GET", f"/repos/{REPOSITORY}/actions/runs/{run_id}")
        run = _required(
            value,
            {
                "id",
                "path",
                "head_sha",
                "event",
                "run_attempt",
                "status",
                "conclusion",
                "actor",
                "triggering_actor",
            },
            context,
        )
        actor = _required(run["actor"], {"login"}, f"{context}_actor")
        trigger = _required(run["triggering_actor"], {"login"}, f"{context}_trigger")
        _require(
            run["id"] == run_id
            and run["path"] == path
            and run["head_sha"] == head_sha
            and run["event"] == "workflow_dispatch"
            and type(run["run_attempt"]) is int
            and run["run_attempt"] == 1
            and run["status"] == "completed"
            and run["conclusion"] == "success"
            and actor["login"] == OWNER
            and trigger["login"] == OWNER,
            f"{context}_not_accepted",
        )
        return digest

    def _accepted_final_cuda_nonces(
        self, run_id: int, head_sha: str
    ) -> tuple[tuple[str, ...], str]:
        jobs = self._attempt_jobs(run_id, 1)
        _require(len(jobs) == 4, "final_cuda_job_cardinality")
        preliminaries: list[tuple[str, str, int | None]] = []
        for key in ("single_minimum", "single_latest", "two_minimum", "two_latest"):
            spec = runtime.JOB_SPECS[key]
            selected = [item for item in jobs if item.get("name") == spec["name"]]
            _require(len(selected) == 1, "final_cuda_job_cardinality")
            raw_labels = selected[0].get("labels")
            _require(
                type(raw_labels) is list and len(raw_labels) == 1 and type(raw_labels[0]) is str,
                "final_cuda_job_labels",
            )
            assert isinstance(raw_labels, list)
            preliminaries.append((str(spec["name"]), raw_labels[0], None))
        jobs = list(
            self._validate_exact_attempt_job_set(
                jobs,
                expected_bindings=preliminaries,
                head_sha=head_sha,
                context="final_cuda_attempt",
            )
        )
        nonces: list[str] = []
        evidence: list[dict[str, Any]] = []
        for key in ("single_minimum", "single_latest", "two_minimum", "two_latest"):
            spec = runtime.JOB_SPECS[key]
            selected = [item for item in jobs if item.get("name") == spec["name"]]
            _require(len(selected) == 1, "final_cuda_job_cardinality")
            job = selected[0]
            labels = job.get("labels")
            _require(type(labels) is list and len(labels) == 1, "final_cuda_job_labels")
            assert isinstance(labels, list)
            runner_name = labels[0]
            _require(
                type(runner_name) is str
                and runner_name.startswith(str(spec["prefix"]))
                and RUNNER_NAME_RE.fullmatch(runner_name) is not None
                and job.get("head_sha") == head_sha
                and type(job.get("run_attempt")) is int
                and job.get("run_attempt") == 1
                and job.get("status") == "completed"
                and job.get("conclusion") == "success"
                and type(job.get("runner_id")) is int
                and job["runner_id"] > 0
                and job.get("runner_name") == runner_name,
                "final_cuda_job_not_accepted",
            )
            nonce = runner_name[-16:]
            _require(NONCE_RE.fullmatch(nonce) is not None, "final_cuda_nonce_rejected")
            nonces.append(nonce)
            evidence.append(
                {
                    "key": key,
                    "job_id": job["id"],
                    "runner_id": job["runner_id"],
                    "runner_name": runner_name,
                    "nonce": nonce,
                }
            )
        _require(len(set(nonces)) == 4, "final_cuda_nonces_not_distinct")
        _require(
            len({item["job_id"] for item in evidence}) == 4
            and len({item["runner_id"] for item in evidence}) == 4,
            "final_cuda_job_or_runner_id_reuse",
        )
        return tuple(nonces), _sha(_canonical(evidence))

    def seal_final_main_acceptance(
        self, session: PhaseSession, settlement: Mapping[str, Any]
    ) -> FinalMainAcceptance:
        """Seal the in-process evidence required by a later publication dispatch."""

        return FinalMainAcceptance._from_completed_session(session, settlement)

    def _revalidate_final_main_acceptance(
        self,
        acceptance: FinalMainAcceptance,
        *,
        run_id: int,
        head_sha: str,
    ) -> tuple[tuple[str, ...], str]:
        _require(
            type(acceptance) is FinalMainAcceptance and hasattr(acceptance, "evidence_sha256"),
            "final_main_acceptance_factory_proof_missing",
        )
        journal_provenance = acceptance._journal_provenance_mapping()
        raw_authority_identities = journal_provenance.get("authority_evidence_identities")
        _require(
            type(raw_authority_identities) is list and len(raw_authority_identities) == 4,
            "final_main_acceptance_authority_provenance_missing",
        )
        assert isinstance(raw_authority_identities, list)
        authority_identities = tuple(
            _validated_authority_evidence_identity(
                item,
                context=f"final_main_acceptance_authority_{ordinal}",
                expected_phase="final-main",
                expected_head_sha=head_sha,
                expected_run_id=run_id,
                expected_job_key=key,
            )
            for ordinal, (key, item) in enumerate(
                zip(
                    ("single_minimum", "single_latest", "two_minimum", "two_latest"),
                    raw_authority_identities,
                ),
                start=1,
            )
        )
        mapping = acceptance.to_mapping()
        evidence_sha256 = mapping.pop("evidence_sha256")
        _require(
            _sha(_canonical(mapping)) == evidence_sha256,
            "final_main_acceptance_digest_mismatch",
        )
        _require(
            acceptance.head_sha == head_sha
            and acceptance.run_id == run_id
            and type(mapping["run_attempt"]) is int
            and mapping["run_attempt"] == 1,
            "final_main_acceptance_run_binding_rejected",
        )
        current_nonces, current_jobs_sha256 = self._accepted_final_cuda_nonces(run_id, head_sha)
        _require(
            current_nonces == acceptance.accepted_cuda_runner_nonces,
            "final_main_acceptance_nonce_drift",
        )
        jobs = list(acceptance.jobs)
        _require(len(jobs) == 4, "final_main_acceptance_job_count_rejected")
        current_jobs = self._attempt_jobs(run_id, 1)
        current_jobs = list(
            self._validate_exact_attempt_job_set(
                current_jobs,
                expected_bindings=tuple(
                    (
                        str(runtime.JOB_SPECS[key]["name"]),
                        f"{runtime.JOB_SPECS[key]['prefix']}{nonce}",
                        int(accepted["job_id"]),
                    )
                    for key, nonce, accepted in zip(
                        ("single_minimum", "single_latest", "two_minimum", "two_latest"),
                        current_nonces,
                        jobs,
                    )
                ),
                head_sha=head_sha,
                context="final_main_acceptance_live_attempt",
            )
        )
        live_binding: list[dict[str, Any]] = []
        history_bindings: list[dict[str, Any]] = []
        for ordinal, (key, nonce, accepted) in enumerate(
            zip(
                ("single_minimum", "single_latest", "two_minimum", "two_latest"),
                current_nonces,
                jobs,
            ),
            start=1,
        ):
            binding_material = dict(accepted)
            accepted_evidence_sha256 = binding_material.pop("evidence_sha256", None)
            _require(
                type(accepted_evidence_sha256) is str
                and _sha(_canonical(binding_material)) == accepted_evidence_sha256,
                "final_main_acceptance_job_digest_mismatch",
            )
            runner_name = f"{runtime.JOB_SPECS[key]['prefix']}{nonce}"
            selected = [
                item
                for item in current_jobs
                if type(item) is dict and item.get("id") == accepted.get("job_id")
            ]
            _require(len(selected) == 1, "final_main_acceptance_live_job_cardinality")
            live_job = selected[0]
            _require(
                accepted.get("phase") == "final-main"
                and accepted.get("run_id") == run_id
                and accepted.get("job_key") == key
                and accepted.get("runner_name") == runner_name
                and accepted.get("pytest_passed") == 15
                and accepted.get("pytest_skipped") == 0
                and live_job.get("name") == runtime.JOB_SPECS[key]["name"]
                and live_job.get("head_sha") == head_sha
                and type(live_job.get("run_attempt")) is int
                and live_job.get("run_attempt") == 1
                and live_job.get("status") == "completed"
                and live_job.get("conclusion") == "success"
                and live_job.get("labels") == [runner_name]
                and live_job.get("runner_id") == accepted.get("runner_id")
                and live_job.get("runner_name") == runner_name,
                "final_main_acceptance_live_job_drift",
            )
            checks, check_digest = self._request_json(
                "GET",
                f"/repos/{REPOSITORY}/commits/{head_sha}/check-runs"
                f"?check_name={job_spec_name_for_check(str(runtime.JOB_SPECS[key]['name']))}"
                "&filter=all",
            )
            check_mapping = _required(checks, {"total_count", "check_runs"}, "check_runs")
            matching_checks = [
                item
                for item in check_mapping["check_runs"]
                if type(item) is dict
                and item.get("name") == runtime.JOB_SPECS[key]["name"]
                and item.get("head_sha") == head_sha
                and item.get("status") == "completed"
                and item.get("conclusion") == "success"
                and type(item.get("app")) is dict
                and item["app"].get("id") == CHECKS_APP_ID
                and str(item.get("details_url", ""))
                .rstrip("/")
                .endswith(f"/job/{accepted['job_id']}")
            ]
            _require(len(matching_checks) == 1, "final_main_acceptance_check_drift")
            log_response = self._request(
                "GET",
                f"/repos/{REPOSITORY}/actions/jobs/{accepted['job_id']}/logs",
                expected=200,
            )
            try:
                passed, skipped, log_digest = self._validate_pytest_log(bytes(log_response.body))
            finally:
                log_response.destroy()
            _require(
                passed == 15 and skipped == 0 and log_digest == accepted.get("log_sha256"),
                "final_main_acceptance_log_drift",
            )
            history = self._nonce_history(
                [nonce],
                exclude=(run_id, 1, int(accepted["job_id"])),
            )
            _require(
                history["historical_match_count"] == 0
                and history["unexpected_queued_or_in_progress_count"] == 0,
                "final_main_acceptance_nonce_reused",
            )
            live_binding.append(
                {
                    "ordinal": ordinal,
                    "key": key,
                    "job_id": accepted["job_id"],
                    "runner_id": accepted["runner_id"],
                    "runner_name": runner_name,
                    "check_response_sha256": check_digest,
                    "log_sha256": log_digest,
                }
            )
            history_bindings.append(history)
        _require(
            len({item["job_id"] for item in live_binding}) == 4
            and len({item["runner_id"] for item in live_binding}) == 4,
            "final_main_acceptance_ids_not_distinct",
        )
        self._seen_nonces.update(current_nonces)
        self._seen_job_ids.update(int(item["job_id"]) for item in jobs)
        self._seen_runner_ids.update(int(item["runner_id"]) for item in jobs)
        self._seen_app_capture_sha256.update(
            item["capture_evidence_sha256"] for item in authority_identities
        )
        self._seen_app_page_sha256.update(
            page for item in authority_identities for page in item["raw_page_sha256"]
        )
        material = {
            "acceptance_evidence_sha256": evidence_sha256,
            "journal_provenance": journal_provenance,
            "current_jobs_sha256": current_jobs_sha256,
            "live_binding": live_binding,
            "fresh_nonce_history": history_bindings,
        }
        return current_nonces, _sha(_canonical(material))

    def dispatch_phase(
        self,
        phase: str,
        *,
        head_sha: str,
        supplied_ref: str,
        prior_accepted_cuda_runner_nonces: Sequence[str] | None = None,
        preflight_run_id: int | None = None,
        cuda_run_id: int | None = None,
        final_main_acceptance: FinalMainAcceptance | None = None,
        poll_limit: int | None = None,
        progress: ProgressSink | None = None,
    ) -> PhaseSession:
        _require(phase in PHASES, "phase_rejected")
        if poll_limit is None:
            poll_limit = 420 if phase == "publication" else 60
        _require(type(poll_limit) is int and poll_limit > 0, "dispatch_poll_limit_rejected")
        spec = PHASES[phase]
        head_sha = _commit(head_sha, "dispatch_head_sha")
        _require(supplied_ref == spec["run_ref"], "dispatch_supplied_ref_rejected")
        supplied_prior = tuple(prior_accepted_cuda_runner_nonces or ())
        expected_prior = {0, 4} if phase == "publication" else {0}
        _require(
            len(supplied_prior) in expected_prior
            and len(set(supplied_prior)) == len(supplied_prior)
            and all(NONCE_RE.fullmatch(item) for item in supplied_prior),
            "prior_cuda_nonces_rejected",
        )
        prior: tuple[str, ...] = ()
        prior_authority_identities: tuple[dict[str, Any], ...] = ()
        source_bindings: dict[str, str] = {}
        if phase == "pull-request":
            source_bindings.update(self._validate_pull_request_source(head_sha))
        elif phase == "final-main":
            observed_main, digest = self._main_sha()
            _require(observed_main == head_sha, "final_main_sha_drift")
            source_bindings["main"] = digest
        elif phase == "publication":
            _require(
                preflight_run_id is not None
                and cuda_run_id is not None
                and final_main_acceptance is not None,
                "publication_run_ids_missing",
            )
            validated_preflight_run_id = _positive(preflight_run_id, "preflight_run_id")
            validated_cuda_run_id = _positive(cuda_run_id, "cuda_run_id")
            assert final_main_acceptance is not None
            source_bindings.update(self._validate_publication_source(head_sha))
            source_bindings["preflight"] = self._validate_positive_run(
                validated_preflight_run_id,
                ".github/workflows/release-preflight.yml",
                head_sha,
                "preflight_run",
            )
            source_bindings["cuda"] = self._validate_positive_run(
                validated_cuda_run_id,
                runtime.CUDA_WORKFLOW_PATH,
                head_sha,
                "final_cuda_run",
            )
            accepted_prior, accepted_prior_digest = self._revalidate_final_main_acceptance(
                final_main_acceptance,
                run_id=validated_cuda_run_id,
                head_sha=head_sha,
            )
            if supplied_prior:
                _require(
                    supplied_prior == accepted_prior,
                    "supplied_prior_cuda_nonces_not_accepted_run",
                )
            prior = accepted_prior
            provenance = final_main_acceptance._journal_provenance_mapping()
            raw_prior_authority = provenance["authority_evidence_identities"]
            assert isinstance(raw_prior_authority, list)
            prior_authority_identities = tuple(
                _json_mapping_copy(item, "prior_authority_identity") for item in raw_prior_authority
            )
            source_bindings["accepted_cuda_nonces"] = accepted_prior_digest

        nonce_keys = tuple(spec["all_nonce_keys"])
        nonces = self._fresh_nonces(len(nonce_keys), prior)
        history = self._nonce_history(nonces)
        _require(history["historical_match_count"] == 0, "nonce_seen_in_history")
        _require(
            history["unexpected_queued_or_in_progress_count"] == 0, "unexpected_active_cuda_queue"
        )
        inputs: dict[str, Any] = dict(zip(nonce_keys, nonces))
        if phase == "publication":
            assert preflight_run_id is not None and cuda_run_id is not None
            inputs = {
                "tag": runtime.PUBLICATION_TAG,
                "preflight_run_id": validated_preflight_run_id,
                "cuda_run_id": validated_cuda_run_id,
                **inputs,
                "stage_recovery_drill": True,
            }
        workflow_digest = self._workflow(spec["workflow"], spec["workflow_path"])
        existing_runs = {
            _positive(run["id"], "pre_dispatch_run_id") for run in self._all_runs(spec["workflow"])
        }
        request_body = {"ref": spec["dispatch_ref"], "inputs": inputs}
        dispatch_path = f"/repos/{REPOSITORY}/actions/workflows/{spec['workflow']}/dispatches"
        request_sha = _sha(
            _canonical({"method": "POST", "path": dispatch_path, "body": request_body})
        )
        _progress(
            progress,
            "github-dispatch-intent",
            {
                "phase": phase,
                "workflow": spec["workflow"],
                "workflow_path": spec["workflow_path"],
                "dispatch_path": dispatch_path,
                "dispatch_ref": spec["dispatch_ref"],
                "run_ref": spec["run_ref"],
                "head_sha": head_sha,
                "inputs": inputs,
                "expected_runner_nonces": list(nonces),
                "pre_dispatch_run_ids": sorted(existing_runs),
                "request_sha256": request_sha,
                "mutation_retried": False,
            },
        )
        dispatch_ambiguity: dict[str, Any] | None = None
        try:
            response_sha = self._request_empty(
                "POST",
                dispatch_path,
                body=request_body,
                expected=204,
            )
        except AmbiguousGitHubMutation as ambiguity:
            response_sha = None
            dispatch_ambiguity = ambiguity.to_public_mapping()
        observed_run: Mapping[str, Any] | None = None
        run_digest = ""
        expected_keys = tuple(spec["queued_job_keys"])
        for _ in range(poll_limit):
            runs, digests = self._paginate(
                f"/repos/{REPOSITORY}/actions/workflows/{spec['workflow']}/runs"
                f"?event=workflow_dispatch&branch={spec['dispatch_ref']}",
                "workflow_runs",
            )
            candidates = []
            for raw in runs:
                run = _required(
                    raw,
                    {
                        "id",
                        "path",
                        "head_sha",
                        "head_branch",
                        "event",
                        "run_attempt",
                        "status",
                        "conclusion",
                        "actor",
                        "triggering_actor",
                    },
                    "dispatched_run",
                )
                actor = _required(run["actor"], {"login"}, "dispatched_actor")
                trigger = _required(run["triggering_actor"], {"login"}, "dispatched_trigger")
                if (
                    run["id"] not in existing_runs
                    and run["path"] == spec["workflow_path"]
                    and run["head_sha"] == head_sha
                    and run["head_branch"] == spec["dispatch_ref"]
                    and run["event"] == "workflow_dispatch"
                    and type(run["run_attempt"]) is int
                    and run["run_attempt"] == 1
                    and run["status"] in {"queued", "in_progress"}
                    and run["conclusion"] is None
                    and actor["login"] == OWNER
                    and trigger["login"] == OWNER
                ):
                    candidates.append(run)
            exact_candidates: list[Mapping[str, Any]] = []
            for candidate in candidates:
                candidate_run_id = _positive(candidate["id"], "dispatch_candidate_run_id")
                candidate_jobs = self._attempt_jobs(candidate_run_id, 1)
                expected_candidate_bindings: list[tuple[str, str, int | None]] = []
                all_expected_labels_present = True
                for ordinal, key in enumerate(expected_keys):
                    expected_name = runtime.JOB_SPECS[key]["name"]
                    expected_label = f"{runtime.JOB_SPECS[key]['prefix']}{nonces[ordinal]}"
                    matching = [
                        item for item in candidate_jobs if item.get("name") == expected_name
                    ]
                    if len(matching) != 1 or matching[0].get("labels") != [expected_label]:
                        all_expected_labels_present = False
                        break
                    expected_candidate_bindings.append((str(expected_name), expected_label, None))
                if all_expected_labels_present:
                    ordered_candidate_jobs = self._validate_exact_attempt_job_set(
                        candidate_jobs,
                        expected_bindings=expected_candidate_bindings,
                        head_sha=head_sha,
                        context="dispatch_candidate_attempt",
                    )
                    _require(
                        all(
                            item.get("status") == "queued"
                            and item.get("conclusion") is None
                            and item.get("runner_id") is None
                            and item.get("runner_name") in {None, ""}
                            for item in ordered_candidate_jobs
                        ),
                        "dispatch_candidate_job_state_rejected",
                    )
                    exact_candidates.append(candidate)
            if len(exact_candidates) == 1:
                observed_run = exact_candidates[0]
                run_digest = _sha(_canonical(digests))
                break
            _require(len(exact_candidates) == 0, "dispatch_run_ambiguous")
            self._sleep(2)
        if observed_run is None:
            raise AmbiguousGitHubMutation(
                "POST",
                dispatch_path,
                request_sha,
                (
                    "dispatch_not_observed"
                    if dispatch_ambiguity is not None
                    else "dispatch_visibility_unresolved"
                ),
                reconciliation={
                    "workflow": spec["workflow"],
                    "head_sha": head_sha,
                    "expected_runner_nonces": list(nonces),
                    "pre_dispatch_run_ids": sorted(existing_runs),
                    "response_received": dispatch_ambiguity is None,
                    "response_sha256": response_sha,
                    "initial_ambiguity": dispatch_ambiguity,
                    "dispatch_retried": False,
                },
            )
        assert observed_run is not None
        run_id = _positive(observed_run["id"], "dispatch_run_id")
        bindings: list[JobBinding] = []
        for _ in range(poll_limit):
            jobs = self._attempt_jobs(run_id, 1)
            bindings = []
            if len(jobs) > len(expected_keys):
                _fail("queued_job_set_cardinality_rejected")
            if len(jobs) < len(expected_keys):
                self._sleep(2)
                continue
            expected_attempt_bindings = tuple(
                (
                    str(runtime.JOB_SPECS[key]["name"]),
                    f"{runtime.JOB_SPECS[key]['prefix']}{inputs[nonce_keys[index]]}",
                    None,
                )
                for index, key in enumerate(expected_keys)
            )
            ordered_jobs = self._validate_exact_attempt_job_set(
                jobs,
                expected_bindings=expected_attempt_bindings,
                head_sha=head_sha,
                context="queued_job_set",
            )
            for ordinal, key in enumerate(expected_keys, start=1):
                job_spec = runtime.JOB_SPECS[key]
                _require(type(job_spec["name"]) is str, "job_spec_name_rejected")
                nonce = inputs[nonce_keys[ordinal - 1]]
                runner_name = f"{job_spec['prefix']}{nonce}"
                job = ordered_jobs[ordinal - 1]
                labels = job["labels"]
                _require(
                    job["status"] == "queued"
                    and job.get("conclusion") is None
                    and job.get("head_sha") == head_sha
                    and type(job.get("run_attempt")) is int
                    and job.get("run_attempt") == 1
                    and labels == [runner_name]
                    and job.get("runner_id") is None
                    and job.get("runner_name") in {None, ""},
                    "queued_job_binding_rejected",
                )
                job_id = _positive(job["id"], "queued_job_id")
                _require(job_id not in self._seen_job_ids, "job_id_reuse")
                bindings.append(
                    JobBinding(
                        key,
                        ordinal,
                        job_id,
                        str(job_spec["name"]),
                        nonce,
                        runner_name,
                    )
                )
            if len(bindings) == len(expected_keys):
                break
            self._sleep(2)
        if len(bindings) != len(expected_keys):
            runners, before_cancel_inventory = self._runner_inventory()
            _require(runners == [], "dispatch_timeout_runner_inventory_not_zero")
            cancel_path = f"/repos/{REPOSITORY}/actions/runs/{run_id}/cancel"
            cancel_request_sha256 = _sha(
                _canonical({"method": "POST", "path": cancel_path, "body": None})
            )
            _progress(
                progress,
                "github-cancel-intent",
                {
                    "phase": phase,
                    "run_id": run_id,
                    "run_attempt": 1,
                    "head_sha": head_sha,
                    "cancel_path": cancel_path,
                    "request_sha256": cancel_request_sha256,
                    "reason": "dispatch-job-materialization-timeout",
                    "accepted_job_ids": [],
                    "serviced_job_ids": [],
                    "mutation_retried": False,
                },
            )
            cancel_ambiguity: AmbiguousGitHubMutation | None = None
            cancel_digest: str | None = None
            try:
                cancel_digest = self._request_empty("POST", cancel_path, expected=202)
            except AmbiguousGitHubMutation as exc:
                cancel_ambiguity = exc
            terminal = False
            terminal_digest = ""
            for _ in range(90):
                value, digest = self._request_json(
                    "GET", f"/repos/{REPOSITORY}/actions/runs/{run_id}"
                )
                run = _required(
                    value,
                    {"id", "run_attempt", "head_sha", "status", "conclusion"},
                    "dispatch_timeout_run",
                )
                if run["status"] == "completed":
                    _require(
                        run["id"] == run_id
                        and type(run["run_attempt"]) is int
                        and run["run_attempt"] == 1
                        and run["head_sha"] == head_sha
                        and run["conclusion"] in {"cancelled", "failure"},
                        "dispatch_timeout_terminal_drift",
                    )
                    terminal = True
                    terminal_digest = digest
                    break
                self._sleep(2)
            runners, after_cancel_inventory = self._runner_inventory()
            _require(runners == [], "dispatch_timeout_post_cancel_runner_inventory_not_zero")
            if not terminal:
                raise AmbiguousGitHubMutation(
                    "POST",
                    dispatch_path,
                    request_sha,
                    "dispatch_timeout_cancel_unresolved",
                    reconciliation={
                        "workflow": spec["workflow"],
                        "head_sha": head_sha,
                        "expected_runner_nonces": list(nonces),
                        "pre_dispatch_run_ids": sorted(existing_runs),
                        "run_id": run_id,
                        "cancel_already_attempted": True,
                        "cancel_request_sha256": cancel_request_sha256,
                        "cancel_response_received": cancel_ambiguity is None,
                        "cancel_response_sha256": cancel_digest,
                        "cancel_ambiguity": (
                            cancel_ambiguity.to_public_mapping()
                            if cancel_ambiguity is not None
                            else None
                        ),
                        "runner_inventory_before_sha256": before_cancel_inventory,
                        "runner_inventory_after_sha256": after_cancel_inventory,
                        "dispatch_retried": False,
                        "cancel_retried": False,
                    },
                ) from None
            _require(bool(terminal_digest), "dispatch_timeout_terminal_digest_missing")
            _progress(
                progress,
                "github-cancel-settled",
                {
                    "run_id": run_id,
                    "cancel_request_sha256": cancel_request_sha256,
                    "cancel_response_sha256": cancel_digest,
                    "run_response_sha256": terminal_digest,
                    "cancel_retried": False,
                },
            )
            raise ControllerError("expected_jobs_not_queued_run_cancelled")
        _require(len({item.job_id for item in bindings}) == len(bindings), "duplicate_phase_job_id")
        service_keys = tuple(spec["job_keys"])
        service_bindings = tuple(item for item in bindings if item.key in service_keys)
        _require(
            tuple(item.key for item in service_bindings) == service_keys,
            "service_job_binding_rejected",
        )
        self._seen_nonces.update(nonces)
        self._seen_job_ids.update(item.job_id for item in bindings)
        dispatch_reconciliation = {
            "response_received": dispatch_ambiguity is None,
            "response_sha256": response_sha,
            "ambiguity": dispatch_ambiguity,
            "run_id": run_id,
            "run_attempt": 1,
            "head_sha": head_sha,
            "run_response_sha256": run_digest,
            "queued_jobs": [
                {
                    "job_id": item.job_id,
                    "job_name": item.name,
                    "runner_name": item.runner_name,
                    "nonce": item.nonce,
                }
                for item in bindings
            ],
            "mutation_retried": False,
        }
        reconciliation_sha256 = _sha(_canonical(dispatch_reconciliation))
        receipt = DispatchReceipt(
            observed_at=_iso(self._now()),
            request_sha256=request_sha,
            response_sha256=_sha(
                _canonical(
                    {
                        "dispatch_reconciliation_sha256": reconciliation_sha256,
                        "source": source_bindings,
                    }
                )
            ),
            workflow_response_sha256=workflow_digest,
            run_response_sha256=run_digest,
            nonce_history_observed_at=str(history["observed_at"]),
            nonce_history_response_sha256=str(history["response_sha256"]),
            mutation_response_received=dispatch_ambiguity is None,
            mutation_reconciliation_sha256=reconciliation_sha256,
        )
        _progress(
            progress,
            "github-dispatch-settled",
            {
                "schema_version": 1,
                "kind": "explainiverse-github-dispatch-settlement",
                "phase": phase,
                "head_sha": head_sha,
                "run_id": run_id,
                "run_attempt": 1,
                "dispatch_reconciliation": dispatch_reconciliation,
                "source_bindings": source_bindings,
                "dispatch_receipt": dict(receipt.__dict__),
            },
        )
        return PhaseSession(
            phase=phase,
            workflow=spec["workflow"],
            workflow_path=spec["workflow_path"],
            dispatch_ref=spec["dispatch_ref"],
            run_ref=spec["run_ref"],
            head_sha=head_sha,
            inputs=inputs,
            prior_accepted_cuda_runner_nonces=prior,
            run=dict(observed_run),
            jobs=service_bindings,
            queued_jobs=tuple(bindings),
            dispatch_receipt=receipt,
            prior_authority_evidence_identities=prior_authority_identities,
        )

    def reconcile_ambiguous_dispatch_for_abort(
        self,
        phase: str,
        head_sha: str,
        ambiguity: AmbiguousGitHubMutation,
        *,
        poll_limit: int = 30,
        progress: ProgressSink | None = None,
    ) -> dict[str, Any]:
        """Find and cancel only the exact nonce-bound run after a lost dispatch response."""

        _require(phase in PHASES, "abort_dispatch_phase_rejected")
        spec = PHASES[phase]
        reconciliation = ambiguity.reconciliation
        raw_nonces = reconciliation.get("expected_runner_nonces")
        raw_existing = reconciliation.get("pre_dispatch_run_ids")
        _require(
            type(poll_limit) is int
            and poll_limit > 0
            and ambiguity.method == "POST"
            and ambiguity.path
            == f"/repos/{REPOSITORY}/actions/workflows/{spec['workflow']}/dispatches"
            and reconciliation.get("workflow") == spec["workflow"]
            and reconciliation.get("head_sha") == head_sha
            and type(raw_nonces) is list
            and len(raw_nonces) == len(spec["queued_job_keys"])
            and all(type(item) is str and NONCE_RE.fullmatch(item) for item in raw_nonces)
            and type(raw_existing) is list
            and all(type(item) is int and item > 0 for item in raw_existing),
            "abort_dispatch_ambiguity_binding_rejected",
        )
        assert isinstance(raw_nonces, list)
        assert isinstance(raw_existing, list)
        existing = set(raw_existing)
        exact: tuple[Mapping[str, Any], list[Mapping[str, Any]]] | None = None
        observations: list[dict[str, Any]] = []
        for _ in range(poll_limit):
            candidates: list[tuple[Mapping[str, Any], list[Mapping[str, Any]]]] = []
            for run in self._all_runs(str(spec["workflow"])):
                if (
                    type(run.get("id")) is not int
                    or run["id"] in existing
                    or run.get("head_sha") != head_sha
                    or run.get("head_branch") != spec["dispatch_ref"]
                    or run.get("event") != "workflow_dispatch"
                    or type(run.get("run_attempt")) is not int
                    or run.get("run_attempt") != 1
                ):
                    continue
                candidate_run_id = _positive(run["id"], "abort_dispatch_candidate_run_id")
                jobs = self._attempt_jobs(candidate_run_id, 1)
                expected_attempt_bindings: list[tuple[str, str, int | None]] = []
                all_expected_labels_present = True
                for ordinal, key in enumerate(spec["queued_job_keys"]):
                    expected_name = runtime.JOB_SPECS[key]["name"]
                    expected_runner = f"{runtime.JOB_SPECS[key]['prefix']}{raw_nonces[ordinal]}"
                    matches = [job for job in jobs if job.get("name") == expected_name]
                    if len(matches) != 1 or matches[0].get("labels") != [expected_runner]:
                        all_expected_labels_present = False
                        break
                    expected_attempt_bindings.append((str(expected_name), expected_runner, None))
                if all_expected_labels_present:
                    selected = list(
                        self._validate_exact_attempt_job_set(
                            jobs,
                            expected_bindings=expected_attempt_bindings,
                            head_sha=head_sha,
                            context="abort_dispatch_candidate_attempt",
                        )
                    )
                    candidates.append((run, selected))
            _require(len(candidates) <= 1, "abort_dispatch_exact_run_ambiguous")
            observations.append(
                {
                    "observed_at": _iso(self._now()),
                    "exact_candidate_count": len(candidates),
                    "exact_run_id": candidates[0][0]["id"] if candidates else None,
                }
            )
            if candidates:
                exact = candidates[0]
                break
            self._sleep(2)
        if exact is None:
            raise AmbiguousGitHubMutation(
                ambiguity.method,
                ambiguity.path,
                ambiguity.request_sha256,
                "dispatch_late_materialization_unresolved",
                reconciliation={
                    "workflow": spec["workflow"],
                    "head_sha": head_sha,
                    "expected_runner_nonces": list(raw_nonces),
                    "pre_dispatch_run_ids": list(raw_existing),
                    "observation_count": len(observations),
                    "observations_sha256": _sha(_canonical(observations)),
                    "dispatch_retried": False,
                },
            ) from None
        assert exact is not None
        run, bound_jobs = exact
        runners, before_digest = self._runner_inventory()
        _require(runners == [], "abort_dispatch_runner_inventory_not_zero")
        run_id = _positive(run["id"], "abort_dispatch_exact_run_id")
        cancel_path = f"/repos/{REPOSITORY}/actions/runs/{run_id}/cancel"
        cancel_request_sha256 = _sha(
            _canonical({"method": "POST", "path": cancel_path, "body": None})
        )
        cancel_ambiguity: AmbiguousGitHubMutation | None = None
        cancel_already_attempted = reconciliation.get("cancel_already_attempted") is True
        if cancel_already_attempted:
            supplied_cancel_digest = reconciliation.get("cancel_request_sha256")
            _require(
                supplied_cancel_digest == cancel_request_sha256,
                "abort_dispatch_prior_cancel_digest_rejected",
            )
            cancel_digest = reconciliation.get("cancel_response_sha256")
            _require(
                cancel_digest is None
                or (type(cancel_digest) is str and SHA256_RE.fullmatch(cancel_digest) is not None),
                "abort_dispatch_prior_cancel_response_digest_rejected",
            )
        elif run.get("status") == "completed":
            cancel_digest = None
        else:
            _progress(
                progress,
                "github-cancel-intent",
                {
                    "phase": phase,
                    "run_id": run_id,
                    "run_attempt": 1,
                    "head_sha": head_sha,
                    "cancel_path": cancel_path,
                    "request_sha256": cancel_request_sha256,
                    "reason": "ambiguous-dispatch-exact-run",
                    "accepted_job_ids": [],
                    "serviced_job_ids": [],
                    "mutation_retried": False,
                },
            )
            try:
                cancel_digest = self._request_empty("POST", cancel_path, expected=202)
            except AmbiguousGitHubMutation as exc:
                cancel_ambiguity = exc
                cancel_digest = exc.request_sha256
        terminal = False
        run_digest = ""
        for _ in range(90):
            value, digest = self._request_json("GET", f"/repos/{REPOSITORY}/actions/runs/{run_id}")
            observed = _required(
                value, {"id", "head_sha", "run_attempt", "status", "conclusion"}, "abort_run"
            )
            if observed["status"] == "completed":
                _require(
                    observed["id"] == run_id
                    and observed["head_sha"] == head_sha
                    and type(observed["run_attempt"]) is int
                    and observed["run_attempt"] == 1
                    and observed["conclusion"] in {"cancelled", "failure"},
                    "abort_dispatch_terminal_drift",
                )
                terminal = True
                run_digest = digest
                break
            self._sleep(2)
        if not terminal:
            raise AmbiguousGitHubMutation(
                "POST",
                cancel_path,
                cancel_request_sha256,
                "abort_dispatch_cancel_unresolved",
                reconciliation={
                    "run_id": run_id,
                    "cancel_retried": False,
                    "response_received": cancel_ambiguity is None,
                    "response_sha256": cancel_digest,
                },
            ) from None
        settled_jobs = self._validate_exact_attempt_job_set(
            self._attempt_jobs(run_id, 1),
            expected_bindings=tuple(
                (
                    str(item["name"]),
                    str(item["labels"][0]),
                    _positive(item["id"], "abort_dispatch_bound_job_id"),
                )
                for item in bound_jobs
            ),
            head_sha=head_sha,
            context="abort_dispatch_settled_attempt",
        )
        settled = {item["id"]: item for item in settled_jobs}
        for job in bound_jobs:
            current = settled.get(job["id"])
            _require(
                type(current) is dict
                and current.get("status") == "completed"
                and current.get("conclusion") in {"cancelled", "skipped"}
                and current.get("runner_id") is None
                and current.get("runner_name") in {None, ""},
                "abort_dispatch_job_was_serviced",
            )
        runners, after_digest = self._runner_inventory()
        _require(runners == [], "abort_dispatch_post_cancel_runner_inventory_not_zero")
        material = {
            "outcome": "exact-run-cancelled",
            "run_id": run_id,
            "dispatch_request_sha256": ambiguity.request_sha256,
            "cancel_request_sha256": cancel_request_sha256,
            "cancel_response_sha256": cancel_digest,
            "cancel_response_ambiguous": cancel_ambiguity is not None,
            "cancel_already_attempted_before_reconciliation": cancel_already_attempted,
            "run_response_sha256": run_digest,
            "runner_inventory_before_sha256": before_digest,
            "runner_inventory_after_sha256": after_digest,
            "observations_sha256": _sha(_canonical(observations)),
            "dispatch_retried": False,
        }
        result = {**material, "evidence_sha256": _sha(_canonical(material))}
        _progress(progress, "github-cancel-settled", result)
        return result

    def _runner_inventory(self) -> tuple[list[Any], str]:
        value, digest = self._request_json("GET", f"/repos/{REPOSITORY}/actions/runners")
        mapping = _object(value, {"total_count", "runners"}, "runner_inventory")
        _require(type(mapping["runners"]) is list, "runner_inventory_not_list")
        _require(mapping["total_count"] == len(mapping["runners"]), "runner_inventory_count_drift")
        return mapping["runners"], digest

    def _stable_zero_runner_inventory(
        self, *, observations: int = 3
    ) -> tuple[str, tuple[str, ...]]:
        _require(
            type(observations) is int and observations >= 3,
            "runner_zero_observation_count_rejected",
        )
        digests: list[str] = []
        for index in range(observations):
            runners, digest = self._runner_inventory()
            _require(runners == [], "runner_inventory_not_stably_zero")
            digests.append(digest)
            if index + 1 < observations:
                self._sleep(2)
        return _sha(_canonical(digests)), tuple(digests)

    def prove_zero_runner_inventory(self) -> dict[str, Any]:
        """Return a fresh public proof that no repository runner remains."""

        response_sha256, observations = self._stable_zero_runner_inventory()
        material = {
            "schema_version": 1,
            "kind": "explainiverse-repository-runner-zero-inventory",
            "repository": REPOSITORY,
            "observed_at": _iso(self._now()),
            "runner_count": 0,
            "response_sha256": response_sha256,
            "observation_count": len(observations),
            "observation_response_sha256": list(observations),
        }
        return {**material, "evidence_sha256": _sha(_canonical(material))}

    @staticmethod
    def _normalize_live_runner(raw: Any) -> dict[str, Any]:
        runner = _object(raw, {"id", "name", "os", "status", "busy", "labels"}, "live_runner")
        labels = runner["labels"]
        _require(type(labels) is list, "live_runner_labels_not_list")
        names: list[str] = []
        for index, raw_label in enumerate(labels):
            label = _object(raw_label, {"id", "name", "type"}, f"live_runner_label_{index}")
            _positive(label["id"], f"live_runner_label_{index}_id")
            _require(type(label["name"]) is str, f"live_runner_label_{index}_name")
            _require(label["type"] in {"custom", "read-only"}, f"live_runner_label_{index}_type")
            names.append(label["name"])
        return {
            "id": _positive(runner["id"], "live_runner_id"),
            "name": runner["name"],
            "os": runner["os"],
            "status": runner["status"],
            "busy": runner["busy"],
            "labels": names,
        }

    def _live_observation(self, session: PhaseSession) -> dict[str, Any]:
        run_value, run_digest = self._request_json(
            "GET", f"/repos/{REPOSITORY}/actions/runs/{session.run['id']}"
        )
        run = _required(
            run_value,
            {
                "id",
                "event",
                "path",
                "head_sha",
                "head_branch",
                "run_attempt",
                "actor",
                "triggering_actor",
                "status",
                "conclusion",
            },
            "live_run",
        )
        actor = _required(run["actor"], {"login"}, "live_run_actor")
        trigger = _required(run["triggering_actor"], {"login"}, "live_run_trigger")
        raw_jobs, job_digests = self._attempt_jobs_with_digests(session.run["id"], 1)
        validated_jobs = self._validate_exact_attempt_job_set(
            raw_jobs,
            expected_bindings=self._session_expected_job_bindings(session),
            head_sha=session.head_sha,
            context="live_attempt",
        )
        jobs: list[dict[str, Any]] = []
        for index, raw in enumerate(validated_jobs):
            job = _required(
                raw,
                {
                    "id",
                    "name",
                    "head_sha",
                    "run_attempt",
                    "status",
                    "conclusion",
                    "labels",
                    "runner_id",
                    "runner_name",
                },
                f"live_job_{index}",
            )
            jobs.append({key: job[key] for key in runtime.LIVE_JOB_KEYS})
        downloads_value, downloads_digest = self._request_json(
            "GET", f"/repos/{REPOSITORY}/actions/runners/downloads"
        )
        _require(type(downloads_value) is list, "live_downloads_not_list")
        downloads: list[dict[str, Any]] = []
        for index, raw in enumerate(downloads_value):
            item = _object(raw, runtime.DOWNLOAD_KEYS, f"live_download_{index}")
            downloads.append(dict(item))
        runners_raw, runners_digest = self._runner_inventory()
        runners = [self._normalize_live_runner(raw) for raw in runners_raw]
        return {
            "captured_at": _iso(self._now()),
            "run_response_sha256": run_digest,
            "jobs_response_sha256": _sha(_canonical(job_digests)),
            "downloads_response_sha256": downloads_digest,
            "runners_response_sha256": runners_digest,
            "run": {
                "id": run["id"],
                "event": run["event"],
                "path": run["path"],
                "ref": session.run_ref,
                "head_sha": run["head_sha"],
                "run_attempt": run["run_attempt"],
                "actor": actor["login"],
                "triggering_actor": trigger["login"],
                "status": run["status"],
                "conclusion": run["conclusion"],
            },
            "jobs": jobs,
            "downloads": downloads,
            "runners": runners,
        }

    def capture_authority(
        self,
        session: PhaseSession,
        app_capture: TrustedAppCapture,
        *,
        installed_app_evidence_reader: Callable[[str], bytes],
    ) -> AuthorityReceipt:
        now = self._now()
        _require(type(app_capture) is TrustedAppCapture, "app_capture_type_rejected")
        _require(hasattr(app_capture, "normalized_capture"), "app_capture_factory_proof_missing")
        validated_capture = TrustedAppCapture.from_mapping(
            app_capture.normalized_capture,
            resources=self._resources,
            evidence_reader=installed_app_evidence_reader,
            now=now,
        )
        _require(
            validated_capture.to_mapping() == app_capture.to_mapping(),
            "app_capture_point_of_use_revalidation_failed",
        )
        app_capture = validated_capture
        captured = _parse_time(app_capture.captured_at, "app_capture")
        dispatch_observed = _parse_time(
            session.dispatch_receipt.observed_at,
            "app_capture_dispatch_observed",
        )
        _require(
            dispatch_observed < captured < now and now - captured <= AUTHORITY_CAPTURE_MAX_AGE,
            "app_capture_phase_freshness_rejected",
        )
        raw_manifest = app_capture.normalized_capture.get("evidence")
        _require(type(raw_manifest) is list, "app_capture_manifest_rejected")
        assert isinstance(raw_manifest, list)
        page_sha256 = tuple(str(item.get("sha256")) for item in raw_manifest if type(item) is dict)
        _require(
            len(page_sha256) == len(raw_manifest)
            and len(set(page_sha256)) == len(page_sha256)
            and all(SHA256_RE.fullmatch(item) is not None for item in page_sha256),
            "app_capture_page_digests_rejected",
        )
        _require(
            app_capture.evidence_sha256 not in self._seen_app_capture_sha256
            and self._seen_app_page_sha256.isdisjoint(page_sha256),
            "app_capture_replayed",
        )
        collaborators, collaborator_digest = self._paginate(
            f"/repos/{REPOSITORY}/collaborators?affiliation=all", None
        )
        _require(len(collaborators) == 1, "authority_not_sole_collaborator")
        collaborator = _required(collaborators[0], {"login", "permissions"}, "collaborator")
        permissions = _required(collaborator["permissions"], {"admin"}, "collaborator_permissions")
        _require(
            collaborator["login"] == OWNER and permissions["admin"] is True,
            "owner_admin_rejected",
        )
        invitations, invitation_digest = self._paginate(f"/repos/{REPOSITORY}/invitations", None)
        _require(invitations == [], "pending_invitations_present")
        runners, runner_digest = self._runner_inventory()
        _require(runners == [], "pre_jit_runner_inventory_not_zero")
        variables, variable_digest = self._paginate(
            f"/repos/{REPOSITORY}/actions/variables", "variables"
        )
        _require(variables == [], "repository_variables_present")
        allowed = {job.job_id for job in session.queued_jobs}
        history = self._nonce_history(
            [job.nonce for job in session.queued_jobs], allowed_active_job_ids=allowed
        )
        _require(
            history["unexpected_queued_or_in_progress_count"] == 0, "unexpected_active_cuda_queue"
        )
        observed_at = _iso(now)
        material = {
            "observed_at": observed_at,
            "app_capture": app_capture.to_mapping(),
            "collaborators": collaborator_digest,
            "invitations": invitation_digest,
            "runners": runner_digest,
            "variables": variable_digest,
            "queue": history,
        }
        self._seen_app_capture_sha256.add(app_capture.evidence_sha256)
        self._seen_app_page_sha256.update(page_sha256)
        return AuthorityReceipt(
            observed_at=observed_at,
            expires_at=_iso(now + AUTHORITY_WINDOW),
            evidence_sha256=_sha(_canonical(material)),
            app_capture_sha256=app_capture.evidence_sha256,
            collaborators_response_sha256=_sha(_canonical(collaborator_digest)),
            invitations_response_sha256=_sha(_canonical(invitation_digest)),
            runners_response_sha256=runner_digest,
            variables_response_sha256=_sha(_canonical(variable_digest)),
            queue_evidence_sha256=history["response_sha256"],
            _evidence_material=_json_mapping_copy(
                material,
                "authority_evidence_material",
            ),
        )

    def establish_host_readiness(
        self,
        cloud_init_binding: live.StrictSshBinding,
        preflight_binding: live.StrictSshBinding,
        *,
        provider_plan: live.ImmutablePlan,
        known_hosts: live.KnownHostsFileReceipt,
        observe_provider_instance: Callable[[], live.SnapshotReceipt],
        ssh_poll_limit: int = 24,
    ) -> HostReadinessReceipt:
        """Run cloud-init first, then probe-host, with a fresh provider read after each."""

        _require(type(ssh_poll_limit) is int and ssh_poll_limit > 0, "ssh_poll_limit_rejected")
        cloud_attempts: list[dict[str, Any]] = []
        cloud: live.CloudInitWaitReceipt | None = None
        bound_ip: str | None = None
        for attempt in range(ssh_poll_limit):
            before_cloud = observe_provider_instance()
            _, before_cloud_ip = live._validate_fresh_bound_instance_receipt(
                before_cloud, provider_plan
            )
            known_hosts_sha256 = live._validate_known_hosts_file_receipt(
                known_hosts,
                plan=provider_plan,
                instance_public_ipv4=before_cloud_ip,
            )
            _validate_strict_ssh_binding_shape(
                cloud_init_binding,
                expected_mode="cloud-init",
                expected_public_ipv4=before_cloud_ip,
                expected_host_fingerprint=provider_plan.host_key_fingerprint,
                expected_known_hosts_path=known_hosts.absolute_path,
                expected_known_hosts_sha256=known_hosts_sha256,
                expected_acl_receipt_sha256=known_hosts.evidence_directory_acl_receipt_sha256,
            )
            try:
                cloud_raw = self._remote.wait_cloud_init(cloud_init_binding)
            except ControllerError as exc:
                cloud_attempts.append({"attempt": attempt + 1, "transport_error": str(exc)})
            else:
                cloud_instance = observe_provider_instance()
                attempt_material = {
                    "attempt": attempt + 1,
                    "stdout_sha256": _sha(cloud_raw.stdout),
                    "stderr_sha256": _sha(cloud_raw.stderr),
                    "exit_code": cloud_raw.exit_code,
                    "provider_snapshot_sha256": cloud_instance.snapshot_sha256,
                }
                try:
                    cloud = live.validate_cloud_init_wait_receipt(
                        cloud_raw.stdout,
                        cloud_raw.stderr,
                        cloud_raw.exit_code,
                        plan=provider_plan,
                        provider_instance=cloud_instance,
                        known_hosts=known_hosts,
                        now=self._now(),
                    )
                except live.ContractError as exc:
                    attempt_material["validation_error"] = str(exc)
                else:
                    attempt_material["accepted"] = True
                    bound_ip = before_cloud_ip
                cloud_attempts.append(attempt_material)
                if cloud is not None:
                    break
            if attempt + 1 < ssh_poll_limit:
                self._sleep(5)
        _require(cloud is not None and bound_ip is not None, "cloud_init_ssh_not_ready")
        assert cloud is not None and bound_ip is not None
        preflight_attempts: list[dict[str, Any]] = []
        preflight: live.HostPreflightReceipt | None = None
        for attempt in range(ssh_poll_limit):
            before_preflight = observe_provider_instance()
            _, before_preflight_ip = live._validate_fresh_bound_instance_receipt(
                before_preflight, provider_plan
            )
            _require(before_preflight_ip == bound_ip, "preflight_instance_ip_drift")
            _validate_strict_ssh_binding_shape(
                preflight_binding,
                expected_mode="preflight",
                expected_public_ipv4=before_preflight_ip,
                expected_host_fingerprint=provider_plan.host_key_fingerprint,
                expected_known_hosts_path=known_hosts.absolute_path,
                expected_known_hosts_sha256=known_hosts.content_sha256,
                expected_acl_receipt_sha256=known_hosts.evidence_directory_acl_receipt_sha256,
            )
            try:
                preflight_raw = self._remote.probe_host(preflight_binding)
            except ControllerError as exc:
                preflight_attempts.append({"attempt": attempt + 1, "transport_error": str(exc)})
            else:
                preflight_instance = observe_provider_instance()
                attempt_material = {
                    "attempt": attempt + 1,
                    "stdout_sha256": _sha(preflight_raw.stdout),
                    "stderr_sha256": _sha(preflight_raw.stderr),
                    "exit_code": preflight_raw.exit_code,
                    "provider_snapshot_sha256": preflight_instance.snapshot_sha256,
                }
                if preflight_raw.exit_code != 0 or preflight_raw.stderr != b"":
                    attempt_material["validation_error"] = "preflight_exit_or_stderr_rejected"
                else:
                    try:
                        preflight = live.validate_host_preflight_receipt(
                            preflight_raw.stdout,
                            plan=provider_plan,
                            provider_instance=preflight_instance,
                            known_hosts=known_hosts,
                            cloud_init_wait=cloud,
                            now=self._now(),
                        )
                    except live.ContractError as exc:
                        attempt_material["validation_error"] = str(exc)
                    else:
                        attempt_material["accepted"] = True
                preflight_attempts.append(attempt_material)
                if preflight is not None:
                    break
            if attempt + 1 < ssh_poll_limit:
                self._sleep(5)
        _require(preflight is not None, "host_preflight_ssh_not_ready")
        assert preflight is not None
        return HostReadinessReceipt._from_validated(
            cloud,
            preflight,
            cloud_init_binding,
            preflight_binding,
            {"cloud_init": cloud_attempts, "preflight": preflight_attempts},
        )

    def refresh_host_preflight(
        self,
        readiness: HostReadinessReceipt,
        preflight_binding: live.StrictSshBinding,
        *,
        provider_plan: live.ImmutablePlan,
        known_hosts: live.KnownHostsFileReceipt,
        observe_provider_instance: Callable[[], live.SnapshotReceipt],
        ssh_poll_limit: int = 12,
    ) -> HostReadinessReceipt:
        readiness.validate()
        _require(type(ssh_poll_limit) is int and ssh_poll_limit > 0, "ssh_poll_limit_rejected")
        attempts: list[dict[str, Any]] = []
        preflight: live.HostPreflightReceipt | None = None
        for attempt in range(ssh_poll_limit):
            before_preflight = observe_provider_instance()
            _, before_preflight_ip = live._validate_fresh_bound_instance_receipt(
                before_preflight, provider_plan
            )
            known_hosts_sha256 = live._validate_known_hosts_file_receipt(
                known_hosts,
                plan=provider_plan,
                instance_public_ipv4=before_preflight_ip,
            )
            _validate_strict_ssh_binding_shape(
                preflight_binding,
                expected_mode="preflight",
                expected_public_ipv4=before_preflight_ip,
                expected_host_fingerprint=provider_plan.host_key_fingerprint,
                expected_known_hosts_path=known_hosts.absolute_path,
                expected_known_hosts_sha256=known_hosts_sha256,
                expected_acl_receipt_sha256=known_hosts.evidence_directory_acl_receipt_sha256,
            )
            try:
                raw = self._remote.probe_host(preflight_binding)
            except ControllerError as exc:
                attempts.append({"attempt": attempt + 1, "transport_error": str(exc)})
            else:
                provider_instance = observe_provider_instance()
                material = {
                    "attempt": attempt + 1,
                    "stdout_sha256": _sha(raw.stdout),
                    "stderr_sha256": _sha(raw.stderr),
                    "exit_code": raw.exit_code,
                    "provider_snapshot_sha256": provider_instance.snapshot_sha256,
                }
                if raw.exit_code != 0 or raw.stderr != b"":
                    material["validation_error"] = "preflight_exit_or_stderr_rejected"
                else:
                    try:
                        preflight = live.validate_host_preflight_receipt(
                            raw.stdout,
                            plan=provider_plan,
                            provider_instance=provider_instance,
                            known_hosts=known_hosts,
                            cloud_init_wait=readiness.cloud_init,
                            now=self._now(),
                        )
                    except live.ContractError as exc:
                        material["validation_error"] = str(exc)
                    else:
                        material["accepted"] = True
                attempts.append(material)
                if preflight is not None:
                    break
            if attempt + 1 < ssh_poll_limit:
                self._sleep(5)
        _require(preflight is not None, "host_preflight_refresh_not_ready")
        assert preflight is not None
        return HostReadinessReceipt._from_validated(
            readiness.cloud_init,
            preflight,
            readiness.cloud_binding,
            preflight_binding,
            {"cloud_init": readiness.ssh_attempts["cloud_init"], "preflight": attempts},
        )

    def _downloads(self) -> dict[str, Any]:
        value, digest = self._request_json("GET", f"/repos/{REPOSITORY}/actions/runners/downloads")
        _require(type(value) is list, "runner_downloads_not_list")
        matches = [
            item
            for item in value
            if type(item) is dict
            and item.get("os") == "linux"
            and item.get("architecture") == "x64"
        ]
        _require(len(matches) == 1, "runner_download_cardinality")
        item = _object(
            matches[0],
            {"os", "architecture", "download_url", "filename", "sha256_checksum"},
            "runner_download",
        )
        _require(
            item
            == {
                "os": "linux",
                "architecture": "x64",
                "download_url": runtime.RUNNER_DOWNLOAD_URL,
                "filename": runtime.RUNNER_FILENAME,
                "sha256_checksum": runtime.RUNNER_ARCHIVE_SHA256,
            },
            "runner_download_drift",
        )
        return {
            "observed_at": _iso(self._now()),
            "response_sha256": digest,
            **item,
            "api_version": API_VERSION,
            "version": runtime.RUNNER_VERSION,
        }

    def _validate_job_still_queued(self, session: PhaseSession, job: JobBinding) -> str:
        jobs, digests = self._attempt_jobs_with_digests(session.run["id"], 1)
        normalized_jobs = self._validate_exact_attempt_job_set(
            jobs,
            expected_bindings=self._session_expected_job_bindings(session),
            head_sha=session.head_sha,
            context="pre_jit_attempt",
        )
        selected = [item for item in normalized_jobs if item.get("id") == job.job_id]
        _require(len(selected) == 1, "pre_jit_job_cardinality")
        current = selected[0]
        _require(
            current.get("name") == job.name
            and current.get("head_sha") == session.head_sha
            and type(current.get("run_attempt")) is int
            and current.get("run_attempt") == 1
            and current.get("status") == "queued"
            and current.get("conclusion") is None
            and current.get("labels") == [job.runner_name]
            and current.get("runner_id") is None
            and current.get("runner_name") in {None, ""},
            "pre_jit_job_binding_rejected",
        )
        return _sha(_canonical(digests))

    @staticmethod
    def _normalize_jit_runner(value: Any, expected_name: str) -> dict[str, Any]:
        runner = _object(value, {"id", "name", "os", "status", "busy", "labels"}, "jit_runner")
        runner_id = _positive(runner["id"], "jit_runner_id")
        labels = runner["labels"]
        _require(type(labels) is list and len(labels) == 1, "jit_runner_labels_not_sole")
        label = _object(labels[0], {"id", "name", "type"}, "jit_runner_label")
        _positive(label["id"], "jit_runner_label_id")
        _require(
            runner["name"] == expected_name
            and runner["os"] == "unknown"
            and runner["status"] == "offline"
            and runner["busy"] is False
            and label["name"] == expected_name
            and label["type"] == "custom",
            "jit_runner_binding_rejected",
        )
        return {
            "id": runner_id,
            "name": expected_name,
            "os": "unknown",
            "status": "offline",
            "busy": False,
            "labels": [expected_name],
        }

    def _delete_unused_runner(
        self,
        runner_id: int,
        *,
        session: PhaseSession | None = None,
        job: JobBinding | None = None,
        progress: ProgressSink | None = None,
    ) -> dict[str, Any]:
        _require(
            (session is None and job is None)
            or (
                session is not None
                and job is not None
                and any(item == job for item in session.jobs)
            ),
            "runner_delete_context_rejected",
        )
        path = f"/repos/{REPOSITORY}/actions/runners/{runner_id}"
        request_sha256 = _sha(_canonical({"method": "DELETE", "path": path, "body": None}))
        _progress(
            progress,
            "github-runner-delete-intent",
            {
                "runner_id": runner_id,
                "phase": session.phase if session is not None else None,
                "run_id": session.run["id"] if session is not None else None,
                "head_sha": session.head_sha if session is not None else None,
                "job_id": job.job_id if job is not None else None,
                "runner_name": job.runner_name if job is not None else None,
                "path": path,
                "request_sha256": request_sha256,
                "mutation_retried": False,
            },
        )
        try:
            delete_digest = self._request_empty("DELETE", path, expected=204)
        except AmbiguousGitHubMutation as ambiguity:
            absence_digests: list[str] = []
            exact: list[Any] = []
            runners: list[Any] = []
            for index in range(3):
                runners, inventory_digest = self._runner_inventory()
                if runners:
                    exact = [
                        item
                        for item in runners
                        if type(item) is dict and item.get("id") == runner_id
                    ]
                    break
                absence_digests.append(inventory_digest)
                if index < 2:
                    self._sleep(2)
            if len(absence_digests) == 3:
                result = {
                    "runner_id": runner_id,
                    "delete_request_sha256": request_sha256,
                    "delete_response_sha256": ambiguity.request_sha256,
                    "delete_response_ambiguous_but_absence_proven": True,
                    "inventory_response_sha256": _sha(_canonical(absence_digests)),
                    "inventory_observation_count": 3,
                    "inventory_observation_response_sha256": absence_digests,
                }
                _progress(progress, "github-runner-delete-settled", result)
                return result
            reconciliation = {
                "runner_id": runner_id,
                "inventory_response_sha256": inventory_digest,
                "runner_count": len(runners),
                "exact_runner_still_present": len(exact) == 1,
                "exact_runner_status": exact[0].get("status") if len(exact) == 1 else None,
                "exact_runner_busy": exact[0].get("busy") if len(exact) == 1 else None,
                "delete_retried": False,
            }
            raise AmbiguousGitHubMutation(
                ambiguity.method,
                ambiguity.path,
                ambiguity.request_sha256,
                "runner_delete_unresolved",
                reconciliation=reconciliation,
            ) from None
        inventory_digest, observation_digests = self._stable_zero_runner_inventory()
        result = {
            "runner_id": runner_id,
            "delete_request_sha256": request_sha256,
            "delete_response_sha256": delete_digest,
            "inventory_response_sha256": inventory_digest,
            "inventory_observation_count": len(observation_digests),
            "inventory_observation_response_sha256": list(observation_digests),
        }
        _progress(progress, "github-runner-delete-settled", result)
        return result

    def _retire_runner_if_present(
        self,
        runner_id: int,
        *,
        session: PhaseSession | None = None,
        job: JobBinding | None = None,
        progress: ProgressSink | None = None,
    ) -> dict[str, Any]:
        runners, before_digest = self._runner_inventory()
        if runners == []:
            return {
                "runner_already_absent": True,
                "inventory_before_sha256": before_digest,
                "delete_response_sha256": None,
                "inventory_after_sha256": before_digest,
            }
        _require(len(runners) == 1, "runner_cleanup_foreign_inventory_present")
        runner = _required(runners[0], {"id"}, "runner_cleanup_candidate")
        _require(runner["id"] == runner_id, "runner_cleanup_identity_mismatch")
        cleanup = self._delete_unused_runner(runner_id, session=session, job=job, progress=progress)
        return {
            "runner_already_absent": False,
            "inventory_before_sha256": before_digest,
            **cleanup,
            "inventory_after_sha256": cleanup["inventory_response_sha256"],
        }

    def reconcile_runner_delete_for_abort(
        self,
        session: PhaseSession,
        job: JobBinding,
        intent: Mapping[str, Any],
        *,
        observations: int = 3,
    ) -> dict[str, Any]:
        """Observe a previously attempted DELETE; never transmit it again."""

        runner_id = _positive(intent.get("runner_id"), "delete_intent_runner_id")
        path = f"/repos/{REPOSITORY}/actions/runners/{runner_id}"
        request_sha256 = _sha(_canonical({"method": "DELETE", "path": path, "body": None}))
        _require(
            type(observations) is int
            and observations >= 3
            and any(item == job for item in session.jobs)
            and intent.get("phase") == session.phase
            and intent.get("run_id") == session.run["id"]
            and intent.get("head_sha") == session.head_sha
            and intent.get("job_id") == job.job_id
            and intent.get("runner_name") == job.runner_name
            and intent.get("path") == path
            and intent.get("request_sha256") == request_sha256
            and intent.get("mutation_retried") is False,
            "abort_runner_delete_intent_binding_rejected",
        )
        digests: list[str] = []
        for index in range(observations):
            runners, digest = self._runner_inventory()
            digests.append(digest)
            exact = [
                self._normalize_live_runner(item)
                for item in runners
                if type(item) is dict and item.get("id") == runner_id
            ]
            _require(
                len(runners) == len(exact) and len(exact) <= 1,
                "abort_runner_delete_foreign_inventory",
            )
            if exact:
                runner = exact[0]
                _require(
                    runner["name"] == job.runner_name and runner["labels"] == [job.runner_name],
                    "abort_runner_delete_identity_drift",
                )
                raise AmbiguousGitHubMutation(
                    "DELETE",
                    path,
                    request_sha256,
                    "runner_delete_still_present_no_replay",
                    reconciliation={
                        "runner_id": runner_id,
                        "runner_status": runner["status"],
                        "runner_busy": runner["busy"],
                        "inventory_response_sha256": digest,
                        "delete_retried": False,
                    },
                ) from None
            if index + 1 < observations:
                self._sleep(2)
        material = {
            "runner_id": runner_id,
            "runner_name": job.runner_name,
            "delete_request_sha256": request_sha256,
            "stable_absence_observation_count": len(digests),
            "inventory_observation_response_sha256": digests,
            "inventory_response_sha256": _sha(_canonical(digests)),
            "delete_retried": False,
        }
        return {**material, "evidence_sha256": _sha(_canonical(material))}

    def _reconcile_ambiguous_remote_start(
        self,
        session: PhaseSession,
        job: JobBinding,
        runner_id: int,
        *,
        poll_limit: int = 90,
        progress: ProgressSink | None = None,
    ) -> dict[str, Any]:
        """Observe before cleanup; never delete an online/busy or assigned runner."""

        observations: list[dict[str, Any]] = []
        for _ in range(poll_limit):
            jobs, job_digests = self._attempt_jobs_with_digests(session.run["id"], 1)
            validated_jobs = self._validate_exact_attempt_job_set(
                jobs,
                expected_bindings=self._session_expected_job_bindings(session),
                head_sha=session.head_sha,
                context="ambiguous_remote_attempt",
            )
            selected = [item for item in validated_jobs if item.get("id") == job.job_id]
            _require(len(selected) == 1, "ambiguous_job_cardinality")
            current_job = selected[0]
            runners, inventory_digest = self._runner_inventory()
            _require(
                len(runners) <= 1
                and all(type(item) is dict and item.get("id") == runner_id for item in runners),
                "ambiguous_foreign_runner_inventory",
            )
            current_runner = runners[0] if runners else None
            observations.append(
                {
                    "observed_at": _iso(self._now()),
                    "job_response_sha256": _sha(_canonical(job_digests)),
                    "runner_inventory_response_sha256": inventory_digest,
                    "job_status": current_job.get("status"),
                    "job_conclusion": current_job.get("conclusion"),
                    "job_runner_id": current_job.get("runner_id"),
                    "runner_status": (
                        current_runner.get("status") if type(current_runner) is dict else None
                    ),
                    "runner_busy": (
                        current_runner.get("busy") if type(current_runner) is dict else None
                    ),
                }
            )
            status = current_job.get("status")
            if status == "completed":
                _require(runners == [], "ambiguous_completed_runner_residue")
                material = {
                    "resolution": "job-terminal-runner-absent",
                    "job_id": job.job_id,
                    "runner_id": runner_id,
                    "job_conclusion": current_job.get("conclusion"),
                    "accepted_actions_evidence": False,
                    "remote_receipt_available": False,
                    "runner_deleted_by_reconciliation": False,
                    "observations_sha256": _sha(_canonical(observations)),
                }
                return {**material, "evidence_sha256": _sha(_canonical(material))}
            if status == "queued" and runners == []:
                material = {
                    "resolution": "registration-already-absent-job-unclaimed",
                    "job_id": job.job_id,
                    "runner_id": runner_id,
                    "accepted_actions_evidence": False,
                    "remote_receipt_available": False,
                    "runner_deleted_by_reconciliation": False,
                    "observations_sha256": _sha(_canonical(observations)),
                }
                return {**material, "evidence_sha256": _sha(_canonical(material))}
            if status == "queued" and type(current_runner) is dict:
                if (
                    current_runner.get("status") == "offline"
                    and current_runner.get("busy") is False
                ):
                    cleanup = self._delete_unused_runner(
                        runner_id,
                        session=session,
                        job=job,
                        progress=progress,
                    )
                    material = {
                        "resolution": "offline-unclaimed-runner-deleted",
                        "job_id": job.job_id,
                        "runner_id": runner_id,
                        "accepted_actions_evidence": False,
                        "remote_receipt_available": False,
                        "runner_deleted_by_reconciliation": True,
                        "cleanup": cleanup,
                        "observations_sha256": _sha(_canonical(observations)),
                    }
                    return {**material, "evidence_sha256": _sha(_canonical(material))}
                _require(
                    current_runner.get("status") == "online" or current_runner.get("busy") is True,
                    "ambiguous_runner_state_rejected",
                )
            elif status == "in_progress":
                _require(
                    current_job.get("runner_id") == runner_id
                    and current_job.get("runner_name") == job.runner_name,
                    "ambiguous_in_progress_binding_rejected",
                )
            else:
                _fail("ambiguous_job_state_rejected")
            self._sleep(5)
        material = {
            "resolution": "still-running-no-deletion",
            "job_id": job.job_id,
            "runner_id": runner_id,
            "accepted_actions_evidence": False,
            "remote_receipt_available": False,
            "runner_deleted_by_reconciliation": False,
            "observations_sha256": _sha(_canonical(observations)),
        }
        return {**material, "evidence_sha256": _sha(_canonical(material))}

    def reconcile_runner_after_host_stop(
        self,
        session: PhaseSession,
        ambiguity_receipt: Mapping[str, Any],
        *,
        poll_limit: int = 60,
        progress: ProgressSink | None = None,
    ) -> dict[str, Any]:
        """After exact host termination, retire only the ambiguity's exact runner."""

        job_id = _positive(ambiguity_receipt.get("job_id"), "stopped_host_job_id")
        runner_id = _positive(ambiguity_receipt.get("runner_id"), "stopped_host_runner_id")
        bindings = [item for item in session.jobs if item.job_id == job_id]
        _require(len(bindings) == 1, "stopped_host_job_not_in_session")
        binding = bindings[0]
        observations: list[dict[str, Any]] = []
        cleanup: Mapping[str, Any] | None = None
        for _ in range(poll_limit):
            jobs = self._validate_exact_attempt_job_set(
                self._attempt_jobs(session.run["id"], 1),
                expected_bindings=self._session_expected_job_bindings(session),
                head_sha=session.head_sha,
                context="stopped_host_attempt",
            )
            selected = [item for item in jobs if item.get("id") == job_id]
            _require(len(selected) == 1, "stopped_host_job_cardinality")
            job = selected[0]
            runners, inventory_digest = self._runner_inventory()
            _require(
                len(runners) <= 1
                and all(type(item) is dict and item.get("id") == runner_id for item in runners),
                "stopped_host_foreign_runner_inventory",
            )
            runner = self._normalize_live_runner(runners[0]) if runners else None
            observations.append(
                {
                    "observed_at": _iso(self._now()),
                    "job_status": job.get("status"),
                    "job_conclusion": job.get("conclusion"),
                    "runner_inventory_response_sha256": inventory_digest,
                    "runner_status": runner.get("status") if runner else None,
                    "runner_busy": runner.get("busy") if runner else None,
                }
            )
            if runner is None and job.get("status") in {"queued", "completed"}:
                break
            if runner is not None:
                _require(
                    runner["name"] == binding.runner_name
                    and runner["labels"] == [binding.runner_name],
                    "stopped_host_runner_binding_drift",
                )
                if (
                    runner["status"] == "offline"
                    and runner["busy"] is False
                    and job.get("status") != "in_progress"
                ):
                    cleanup = self._delete_unused_runner(
                        runner_id,
                        session=session,
                        job=binding,
                        progress=progress,
                    )
                    break
            self._sleep(2)
        runners, final_inventory_digest = self._runner_inventory()
        _require(runners == [], "stopped_host_runner_did_not_retire")
        material = {
            "run_id": session.run["id"],
            "job_id": job_id,
            "runner_id": runner_id,
            "runner_name": binding.runner_name,
            "cleanup": cleanup,
            "observations_sha256": _sha(_canonical(observations)),
            "final_inventory_response_sha256": final_inventory_digest,
            "host_was_stopped_before_runner_deletion": True,
        }
        return {**material, "evidence_sha256": _sha(_canonical(material))}

    def reconcile_crash_jit_after_host_stop(
        self,
        session: PhaseSession,
        job: JobBinding,
        intent: Mapping[str, Any],
        *,
        poll_limit: int = 60,
        progress: ProgressSink | None = None,
    ) -> dict[str, Any]:
        """Retire only the intent-bound registration after a controller crash."""

        body = {
            "name": job.runner_name,
            "runner_group_id": RUNNER_GROUP_ID,
            "labels": [job.runner_name],
            "work_folder": f"_work-{job.nonce}",
        }
        expected_path = f"/repos/{REPOSITORY}/actions/runners/generate-jitconfig"
        expected_request = _sha(_canonical({"method": "POST", "path": expected_path, "body": body}))
        supplied_runner_id = intent.get("runner_id")
        _require(
            type(poll_limit) is int
            and poll_limit > 0
            and intent.get("phase") == session.phase
            and intent.get("run_id") == session.run["id"]
            and intent.get("head_sha") == session.head_sha
            and intent.get("job_key") == job.key
            and intent.get("job_id") == job.job_id
            and intent.get("runner_name") == job.runner_name
            and intent.get("runner_nonce") == job.nonce
            and intent.get("path") == expected_path
            and intent.get("request_sha256") == expected_request
            and intent.get("runner_group_id") == RUNNER_GROUP_ID
            and intent.get("mutation_retried") is False
            and (
                supplied_runner_id is None
                or (type(supplied_runner_id) is int and supplied_runner_id > 0)
            ),
            "crash_jit_intent_binding_rejected",
        )
        observations: list[dict[str, Any]] = []
        cleanup: Mapping[str, Any] | None = None
        terminal_conclusion: str | None = None
        for _ in range(poll_limit):
            jobs = self._validate_exact_attempt_job_set(
                self._attempt_jobs(session.run["id"], 1),
                expected_bindings=self._session_expected_job_bindings(session),
                head_sha=session.head_sha,
                context="crash_jit_attempt",
            )
            selected = [item for item in jobs if item.get("id") == job.job_id]
            _require(len(selected) == 1, "crash_jit_job_cardinality")
            current_job = selected[0]
            runners, inventory_digest = self._runner_inventory()
            _require(len(runners) <= 1, "crash_jit_foreign_runner_inventory")
            runner = self._normalize_live_runner(runners[0]) if runners else None
            if runner is not None:
                _require(
                    runner["name"] == job.runner_name
                    and runner["labels"] == [job.runner_name]
                    and (supplied_runner_id is None or runner["id"] == supplied_runner_id),
                    "crash_jit_runner_binding_drift",
                )
            observations.append(
                {
                    "observed_at": _iso(self._now()),
                    "job_status": current_job.get("status"),
                    "job_conclusion": current_job.get("conclusion"),
                    "runner_inventory_response_sha256": inventory_digest,
                    "runner_id": runner.get("id") if runner else None,
                    "runner_status": runner.get("status") if runner else None,
                    "runner_busy": runner.get("busy") if runner else None,
                }
            )
            if current_job.get("status") == "completed":
                terminal_conclusion = current_job.get("conclusion")
                if runner is None:
                    break
            elif current_job.get("status") == "queued" and runner is None:
                break
            if (
                runner is not None
                and runner["status"] == "offline"
                and runner["busy"] is False
                and current_job.get("status") != "in_progress"
            ):
                cleanup = self._delete_unused_runner(
                    runner["id"],
                    session=session,
                    job=job,
                    progress=progress,
                )
                break
            self._sleep(2)
        runners, final_inventory_digest = self._runner_inventory()
        _require(runners == [], "crash_jit_runner_did_not_retire")
        material = {
            "run_id": session.run["id"],
            "job_id": job.job_id,
            "runner_name": job.runner_name,
            "supplied_runner_id": supplied_runner_id,
            "cleanup": cleanup,
            "job_terminal_conclusion": terminal_conclusion,
            "observations_sha256": _sha(_canonical(observations)),
            "final_inventory_response_sha256": final_inventory_digest,
            "host_was_stopped_before_runner_deletion": True,
            "mutation_retried": False,
        }
        return {**material, "evidence_sha256": _sha(_canonical(material))}

    def _reconcile_jit_ambiguity(
        self,
        ambiguity: AmbiguousGitHubMutation,
        session: PhaseSession,
        job: JobBinding,
        *,
        poll_limit: int = 6,
        progress: ProgressSink | None = None,
    ) -> NoReturn:
        observations: list[dict[str, Any]] = []
        stable_absence_count = 0
        for attempt in range(poll_limit):
            queued_digest = self._validate_job_still_queued(session, job)
            runners, inventory_digest = self._runner_inventory()
            observations.append(
                {
                    "observed_at": _iso(self._now()),
                    "queued_job_response_sha256": queued_digest,
                    "runner_inventory_response_sha256": inventory_digest,
                    "runner_count": len(runners),
                }
            )
            if runners == []:
                stable_absence_count += 1
                if stable_absence_count >= 3:
                    reconciliation = {
                        "runner_name": job.runner_name,
                        "outcome": "stable-absence-run-requires-cancellation",
                        "observations_sha256": _sha(_canonical(observations)),
                        "stable_absence_observations": stable_absence_count,
                        "generate_jit_retried": False,
                    }
                    raise AmbiguousGitHubMutation(
                        ambiguity.method,
                        ambiguity.path,
                        ambiguity.request_sha256,
                        "jit_secret_response_lost_runner_absent",
                        reconciliation=reconciliation,
                    ) from None
            else:
                stable_absence_count = 0
                _require(len(runners) == 1, "jit_ambiguity_foreign_runner_inventory")
                live_runner = self._normalize_live_runner(runners[0])
                _require(
                    live_runner["name"] == job.runner_name
                    and live_runner["id"] > 0
                    and live_runner["os"] == "unknown"
                    and live_runner["status"] == "offline"
                    and live_runner["busy"] is False
                    and live_runner["labels"] == [job.runner_name],
                    "jit_ambiguity_runner_not_safely_unused",
                )
                cleanup = self._delete_unused_runner(
                    live_runner["id"],
                    session=session,
                    job=job,
                    progress=progress,
                )
                reconciliation = {
                    "runner_name": job.runner_name,
                    "runner_id": live_runner["id"],
                    "outcome": "secret-lost-offline-unclaimed-runner-deleted",
                    "cleanup": cleanup,
                    "observations_sha256": _sha(_canonical(observations)),
                    "generate_jit_retried": False,
                }
                raise AmbiguousGitHubMutation(
                    ambiguity.method,
                    ambiguity.path,
                    ambiguity.request_sha256,
                    "jit_secret_response_lost",
                    reconciliation=reconciliation,
                ) from None
            if attempt + 1 < poll_limit:
                self._sleep(2)
        raise AmbiguousGitHubMutation(
            ambiguity.method,
            ambiguity.path,
            ambiguity.request_sha256,
            "jit_registration_reconciliation_unresolved",
            reconciliation={
                "runner_name": job.runner_name,
                "observations_sha256": _sha(_canonical(observations)),
                "generate_jit_retried": False,
            },
        ) from None

    def _generate_jit(
        self,
        job: JobBinding,
        session: PhaseSession | None = None,
        *,
        progress: ProgressSink | None = None,
    ) -> tuple[dict[str, Any], live.SecretBuffer]:
        body = {
            "name": job.runner_name,
            "runner_group_id": RUNNER_GROUP_ID,
            "labels": [job.runner_name],
            "work_folder": f"_work-{job.nonce}",
        }
        path = f"/repos/{REPOSITORY}/actions/runners/generate-jitconfig"
        request_sha256 = _sha(_canonical({"method": "POST", "path": path, "body": body}))
        _progress(
            progress,
            "github-jit-intent",
            {
                "phase": session.phase if session is not None else None,
                "run_id": session.run["id"] if session is not None else None,
                "head_sha": session.head_sha if session is not None else None,
                "job_key": job.key,
                "job_id": job.job_id,
                "job_name": job.name,
                "runner_name": job.runner_name,
                "runner_nonce": job.nonce,
                "path": path,
                "request_sha256": request_sha256,
                "runner_group_id": RUNNER_GROUP_ID,
                "mutation_retried": False,
            },
        )
        try:
            response = self._request("POST", path, body=body, expected=201)
        except AmbiguousGitHubMutation as transport_ambiguity:
            _require(session is not None, "jit_ambiguity_session_missing")
            assert session is not None
            self._reconcile_jit_ambiguity(transport_ambiguity, session, job, progress=progress)
        response_digest = _response_envelope_digest(response)
        raw_body_digest = _sha(response.body)
        candidate_runner_id: int | None = None
        secret: live.SecretBuffer | None = None
        try:
            payload = _json(response.body, "jit_response")
            mapping = _object(payload, {"runner", "encoded_jit_config"}, "jit_response")
            raw_runner = mapping["runner"]
            if (
                type(raw_runner) is dict
                and type(raw_runner.get("id")) is int
                and raw_runner["id"] > 0
            ):
                candidate_runner_id = raw_runner["id"]
            runner = self._normalize_jit_runner(raw_runner, job.runner_name)
            _require(runner["id"] not in self._seen_runner_ids, "runner_id_reuse")
            encoded = mapping["encoded_jit_config"]
            _require(type(encoded) is str, "jit_config_not_text")
            jit_bytes = bytearray(encoded.encode("ascii"))
            _require(100 <= len(jit_bytes) <= 1_048_576, "jit_config_size_rejected")
            _require(JIT_RE.fullmatch(jit_bytes) is not None, "jit_config_shape_rejected")
            try:
                base64.b64decode(jit_bytes, validate=True)
            except (ValueError, binascii.Error):
                _fail("jit_config_encoding_rejected")
            secret = live.SecretBuffer(jit_bytes, label="github_jit_config")
            for index in range(len(jit_bytes)):
                jit_bytes[index] = 0
            receipt = {
                "observed_at": _iso(self._now()),
                "request_sha256": _sha(_canonical(body)),
                "response_sha256": response_digest,
                "response_body_sha256": raw_body_digest,
                "runner": runner,
                "jit_config_sha256": _sha(secret.view()),
                "encoded_jit_config_persisted": False,
                "runner_group_id": RUNNER_GROUP_ID,
                "runner_group_get_performed": False,
            }
            self._seen_runner_ids.add(runner["id"])
            try:
                _progress(
                    progress,
                    "github-jit-created",
                    {
                        "phase": session.phase if session is not None else None,
                        "run_id": session.run["id"] if session is not None else None,
                        "head_sha": session.head_sha if session is not None else None,
                        "job_key": job.key,
                        "job_id": job.job_id,
                        "runner_name": job.runner_name,
                        "jit_receipt": receipt,
                    },
                )
            except BaseException:
                secret.destroy()
                self._retire_runner_if_present(
                    runner["id"],
                    session=session,
                    job=job if session is not None else None,
                    progress=progress,
                )
                raise
            return receipt, secret
        except (ControllerError, UnicodeEncodeError) as exc:
            if secret is not None:
                secret.destroy()
            if session is not None:
                response.destroy()
                parse_ambiguity = AmbiguousGitHubMutation(
                    "POST",
                    path,
                    request_sha256,
                    "jit_success_response_unusable",
                    reconciliation={
                        "response_received": True,
                        "response_sha256": response_digest,
                        "response_body_sha256": raw_body_digest,
                        "candidate_runner_id": candidate_runner_id,
                        "parse_error_code": (
                            str(exc)
                            if isinstance(exc, ControllerError)
                            else "jit_response_not_ascii"
                        ),
                        "generate_jit_retried": False,
                    },
                )
                self._reconcile_jit_ambiguity(parse_ambiguity, session, job, progress=progress)
            if candidate_runner_id is not None:
                self._retire_runner_if_present(
                    candidate_runner_id,
                    session=session,
                    job=job if session is not None else None,
                    progress=progress,
                )
            raise
        finally:
            response.destroy()

    def _build_plan(
        self,
        session: PhaseSession,
        job: JobBinding,
        authority: AuthorityReceipt,
        readiness: HostReadinessReceipt,
        downloads: Mapping[str, Any],
        history: Mapping[str, Any],
        absence_digest: str,
        jit_receipt: Mapping[str, Any],
        *,
        control_plane_plan_sha256: str,
        previous_cleanup_receipt_sha256: str | None,
    ) -> tuple[dict[str, Any], bytes]:
        created_at = self._now()
        preflight = readiness.preflight
        host_uuids = list(preflight.host_physical_gpu_uuids)
        host_products = list(preflight.host_physical_gpu_products)
        spec = runtime.JOB_SPECS[job.key]
        _require(type(spec["gpu_count"]) is int, "job_spec_gpu_count_rejected")
        required_gpu_count = spec["gpu_count"]
        assert isinstance(required_gpu_count, int)
        assigned = host_uuids[:required_gpu_count]
        unrequested = host_uuids[required_gpu_count:]
        runner = jit_receipt["runner"]
        dispatch_inputs = dict(session.inputs)
        plan = {
            "schema_version": runtime.SCHEMA_VERSION,
            "kind": runtime.PLAN_KIND,
            "execution_authorized": True,
            "created_at": _iso(created_at),
            "policy_sha256": self.policy_sha256(),
            "control_plane_plan_sha256": control_plane_plan_sha256,
            "runtime_bundle_sha256": preflight.runtime_bundle_sha256,
            "phase": session.phase,
            "repository": REPOSITORY,
            "workflow_path": session.workflow_path,
            "authority_window": authority.runtime_mapping(),
            "dispatch": {
                "observed_at": session.dispatch_receipt.observed_at,
                "request_sha256": session.dispatch_receipt.request_sha256,
                "response_sha256": session.dispatch_receipt.response_sha256,
                "event": "workflow_dispatch",
                "ref": session.run_ref,
                "inputs": dispatch_inputs,
                "prior_accepted_cuda_runner_nonces": list(
                    session.prior_accepted_cuda_runner_nonces
                ),
                "run_id": session.run["id"],
                "run_attempt": 1,
                "head_sha": session.head_sha,
                "actor": OWNER,
                "triggering_actor": OWNER,
                "status": session.run["status"],
                "conclusion": None,
            },
            "job": {
                "ordinal": job.ordinal,
                "key": job.key,
                "job_id": job.job_id,
                "name": job.name,
                "runner_nonce": job.nonce,
                "runner_id": runner["id"],
                "runner_name": job.runner_name,
                "labels": [job.runner_name],
                "status": "queued",
                "conclusion": None,
                "work_folder": f"_work-{job.nonce}",
                "jit_config_sha256": jit_receipt["jit_config_sha256"],
            },
            "sequencing": {
                "sequential_only": True,
                "previous_cleanup_receipt_sha256": previous_cleanup_receipt_sha256,
            },
            "hardware": {
                "host_physical_gpu_count": 8,
                "host_physical_gpu_uuids": host_uuids,
                "host_physical_gpu_products": host_products,
                "assigned_physical_gpu_uuids": assigned,
                "unrequested_physical_gpu_uuids": unrequested,
                "device_request": ",".join(assigned),
                "nvidia_visible_devices": ",".join(assigned),
                "cuda_visible_devices": spec["cuda_visible_devices"],
                "required_cuda_devices": required_gpu_count,
                "exclusive_device_scope_required": True,
            },
            "runner_source": dict(downloads),
            "runner_image": {
                "tag_reference": runtime.IMAGE_TAG_REFERENCE,
                "image_reference": runtime.IMAGE_REFERENCE,
                "platform": runtime.IMAGE_PLATFORM,
                "manifest_digest": runtime.IMAGE_MANIFEST_DIGEST,
                "manifest_media_type": runtime.IMAGE_MANIFEST_MEDIA_TYPE,
                "manifest_size": runtime.IMAGE_MANIFEST_SIZE,
                "config_digest": runtime.IMAGE_CONFIG_DIGEST,
                "config_media_type": runtime.IMAGE_CONFIG_MEDIA_TYPE,
                "config_size": runtime.IMAGE_CONFIG_SIZE,
                "manifest_source": runtime.IMAGE_MANIFEST_SOURCE,
                "manifest_observed_at": runtime.IMAGE_MANIFEST_OBSERVED_AT,
                "probe_observed_at": preflight.observed_at,
                "probe_receipt_sha256": preflight.image_probe_sha256,
                "container_uid": runtime.CONTAINER_UID,
                "container_gid": runtime.CONTAINER_GID,
                "runner_listener_present": True,
                "runner_listener_version": runtime.RUNNER_VERSION,
                "runner_commit": runtime.IMAGE_RUNNER_COMMIT,
                "node20_present": True,
                "node20_version": runtime.IMAGE_NODE20_VERSION,
                "node20_sha256": runtime.IMAGE_NODE20_SHA256,
            },
            "github_evidence": {
                "pre_jit_registration_absence": {
                    "observed_at": jit_receipt["absence_observed_at"],
                    "response_sha256": absence_digest,
                    "total_count": 0,
                    "runners": [],
                },
                "nonce_history": dict(history),
                "jit_response": {
                    "observed_at": jit_receipt["observed_at"],
                    "response_sha256": jit_receipt["response_sha256"],
                    "runner": dict(runner),
                },
            },
            "limits": {
                "hard_wall_seconds": runtime.HARD_WALL_SECONDS,
                "fd_read_seconds": runtime.FD_READ_SECONDS,
                "post_github_settle_seconds": runtime.POST_GITHUB_SETTLE_SECONDS,
                "external_watchdog_required": True,
                "cleanup_grace_seconds": runtime.CLEANUP_GRACE_SECONDS,
            },
        }
        normalized = runtime.validate_runtime_plan(plan, now=created_at)
        return normalized, runtime.canonical_json(normalized)

    @staticmethod
    def _validate_remote_receipt(plan: Mapping[str, Any], execution: RemoteExecution) -> str:
        receipt = _object(execution.receipt, REMOTE_RECEIPT_KEYS, "remote_receipt")
        expected = {
            "schema_version": runtime.SCHEMA_VERSION,
            "kind": runtime.RECEIPT_KIND,
            "status": "runner-container-stopped-and-host-cleaned",
            "policy_sha256": plan["policy_sha256"],
            "control_plane_plan_sha256": plan["control_plane_plan_sha256"],
            "runtime_plan_sha256": runtime.runtime_plan_sha256(plan),
            "phase": plan["phase"],
            "repository": REPOSITORY,
            "workflow_path": plan["workflow_path"],
            "ref": plan["dispatch"]["ref"],
            "run_id": plan["dispatch"]["run_id"],
            "run_attempt": 1,
            "head_sha": plan["dispatch"]["head_sha"],
            "job_id": plan["job"]["job_id"],
            "job_name": plan["job"]["name"],
            "runner_id": plan["job"]["runner_id"],
            "runner_name": plan["job"]["runner_name"],
            "labels": plan["job"]["labels"],
            "host_physical_gpu_uuids": plan["hardware"]["host_physical_gpu_uuids"],
            "host_physical_gpu_products": plan["hardware"]["host_physical_gpu_products"],
            "assigned_physical_gpu_uuids": plan["hardware"]["assigned_physical_gpu_uuids"],
            "unrequested_physical_gpu_uuids": plan["hardware"]["unrequested_physical_gpu_uuids"],
            "nvidia_visible_devices": plan["hardware"]["nvidia_visible_devices"],
            "cuda_visible_devices": plan["hardware"]["cuda_visible_devices"],
            "runner_version": runtime.RUNNER_VERSION,
            "runner_archive_sha256": runtime.RUNNER_ARCHIVE_SHA256,
            "runner_image_reference": runtime.IMAGE_REFERENCE,
            "runner_image_manifest_digest": runtime.IMAGE_MANIFEST_DIGEST,
            "jit_config_sha256": plan["job"]["jit_config_sha256"],
            "jit_config_persisted": False,
            "jit_config_destroyed": True,
            "one_job_jit_configuration_supplied": True,
            "claimed_job_count_verified_by_runtime": False,
            "runner_exit_code": 0,
            "authority_expires_at": plan["authority_window"]["expires_at"],
            "workload_stopped_before_authority_expiry": True,
            "cleanup_grace_seconds": runtime.CLEANUP_GRACE_SECONDS,
            "cleanup_completed_within_grace": True,
            "descendants_remaining": 0,
            "container_present": False,
            "network_present": False,
            "firewall_chain_present": False,
            "pre_jit_registration_absence_evidence_sha256": plan["github_evidence"][
                "pre_jit_registration_absence"
            ]["response_sha256"],
            "post_exit_registration_absence_verified_by_runtime": False,
            "post_exit_registration_state": "not-observed-on-remote-host",
            "github_contacted_by_runtime": False,
            "test_counts_verified_by_runtime": False,
            "job_success_verified_by_runtime": False,
            "accepted_actions_evidence": False,
        }
        _require(
            all(receipt[key] == value for key, value in expected.items()),
            "remote_receipt_overclaim_or_drift",
        )
        expected_deadline = _parse_time(
            plan["authority_window"]["expires_at"], "authority_expires_at"
        ) + timedelta(seconds=runtime.CLEANUP_GRACE_SECONDS)
        _require(
            _parse_time(receipt["cleanup_deadline_at"], "remote_cleanup_deadline")
            == expected_deadline,
            "remote_cleanup_deadline_drift",
        )
        for timestamp_field in (
            "jit_config_sent_at",
            "started_at",
            "stopped_at",
            "cleanup_verified_at",
            "authority_expires_at",
        ):
            _parse_time(receipt[timestamp_field], f"remote_{timestamp_field}")
        _require(
            set(execution.frame_receipt)
            == {
                "magic",
                "version",
                "flags",
                "header_bytes",
                "plan_bytes",
                "jit_config_bytes",
                "plan_sha256",
                "jit_config_sha256",
                "header_sha256",
                "trailing_bytes_permitted",
                "remote_argv_contains_plan_or_jit_values",
            }
            and execution.frame_receipt.get("magic") == "EXJIT01"
            and execution.frame_receipt.get("version") == 1
            and execution.frame_receipt.get("flags") == 0
            and execution.frame_receipt.get("header_bytes") == 84
            and type(execution.frame_receipt.get("plan_bytes")) is int
            and execution.frame_receipt["plan_bytes"] == len(runtime.canonical_json(plan))
            and type(execution.frame_receipt.get("jit_config_bytes")) is int
            and execution.frame_receipt["jit_config_bytes"] >= 100
            and execution.frame_receipt.get("plan_sha256") == runtime.runtime_plan_sha256(plan)
            and execution.frame_receipt.get("jit_config_sha256") == plan["job"]["jit_config_sha256"]
            and type(execution.frame_receipt.get("header_sha256")) is str
            and SHA256_RE.fullmatch(execution.frame_receipt["header_sha256"]) is not None
            and execution.frame_receipt.get("trailing_bytes_permitted") is False
            and execution.frame_receipt.get("remote_argv_contains_plan_or_jit_values") is False,
            "runtime_frame_receipt_rejected",
        )
        receipt_sha256 = _sha(_canonical(receipt))
        _require(
            execution.stdout_sha256 == receipt_sha256
            and SHA256_RE.fullmatch(execution.stderr_sha256) is not None,
            "remote_execution_output_digest_rejected",
        )
        return receipt_sha256

    def execute_job(
        self,
        session: PhaseSession,
        job_key: str,
        *,
        app_capture: TrustedAppCapture,
        installed_app_evidence_reader: Callable[[str], bytes],
        readiness: HostReadinessReceipt,
        run_binding: live.StrictSshBinding,
        control_plane_plan_sha256: str,
        progress: ProgressSink | None = None,
    ) -> tuple[dict[str, Any], RemoteExecution]:
        _require(
            SHA256_RE.fullmatch(control_plane_plan_sha256) is not None, "control_plane_plan_digest"
        )
        cloud = readiness.cloud_init
        preflight = readiness.preflight
        readiness.validate()
        _validate_strict_ssh_binding_shape(
            run_binding,
            expected_mode="run",
            expected_public_ipv4=preflight.instance_public_ipv4,
            expected_host_fingerprint=preflight.host_fingerprint,
            expected_known_hosts_path=str(readiness.preflight_binding["known_hosts_path"]),
            expected_known_hosts_sha256=preflight.known_hosts_sha256,
            expected_acl_receipt_sha256=str(
                readiness.preflight_binding["evidence_directory_acl_receipt_sha256"]
            ),
        )
        _require(
            control_plane_plan_sha256 == cloud.plan_sha256 == preflight.plan_sha256
            and cloud.instance_id == preflight.instance_id
            and cloud.instance_public_ipv4 == preflight.instance_public_ipv4
            and cloud.host_fingerprint == preflight.host_fingerprint
            and cloud.known_hosts_sha256 == preflight.known_hosts_sha256
            and cloud.provider_receipt_nonce != preflight.provider_receipt_nonce
            and preflight.cloud_init_wait_binding_sha256 == cloud.binding_sha256,
            "live_control_plane_host_binding_rejected",
        )
        preflight_age = self._now() - _parse_time(preflight.observed_at, "host_preflight")
        _require(
            timedelta(seconds=-30) <= preflight_age <= OBSERVATION_MAX_AGE,
            "live_host_preflight_stale",
        )
        candidates = [job for job in session.jobs if job.key == job_key]
        _require(len(candidates) == 1, "job_key_not_in_phase")
        job = candidates[0]
        expected_prior = [item.key for item in session.jobs[: job.ordinal - 1]]
        _require(list(session.accepted) == expected_prior, "job_sequence_rejected")
        authority = self.capture_authority(
            session,
            app_capture,
            installed_app_evidence_reader=installed_app_evidence_reader,
        )
        _progress(progress, "authority-window", authority.evidence_mapping())
        downloads = self._downloads()
        history = self._nonce_history(
            [job.nonce],
            exclude=(session.run["id"], 1, job.job_id),
            allowed_active_job_ids={item.job_id for item in session.queued_jobs},
        )
        _require(history["historical_match_count"] == 0, "nonce_seen_in_history")
        _require(
            history["unexpected_queued_or_in_progress_count"] == 0, "unexpected_active_cuda_queue"
        )
        self._validate_job_still_queued(session, job)
        runners, absence_digest = self._runner_inventory()
        _require(runners == [], "pre_jit_runner_inventory_not_zero")
        absence_observed_at = _iso(self._now())
        absence_material = {
            "phase": session.phase,
            "run_id": session.run["id"],
            "run_attempt": 1,
            "head_sha": session.head_sha,
            "job_key": job.key,
            "job_id": job.job_id,
            "runner_name": job.runner_name,
            "observed_at": absence_observed_at,
            "response_sha256": absence_digest,
            "total_count": 0,
            "runners": [],
        }
        _progress(
            progress,
            "github-pre-jit-runner-absence",
            {
                **absence_material,
                "evidence_sha256": _sha(_canonical(absence_material)),
            },
        )
        jit_receipt: dict[str, Any]
        secret: live.SecretBuffer
        jit_receipt, secret = self._generate_jit(job, session, progress=progress)
        jit_receipt["absence_observed_at"] = absence_observed_at
        previous_digest = None
        if expected_prior:
            previous_digest = session.accepted[expected_prior[-1]].remote_receipt_sha256
        try:
            plan, plan_bytes = self._build_plan(
                session,
                job,
                authority,
                readiness,
                downloads,
                history,
                absence_digest,
                jit_receipt,
                control_plane_plan_sha256=control_plane_plan_sha256,
                previous_cleanup_receipt_sha256=previous_digest,
            )
            pre_execution = self._live_observation(session)
            runtime.validate_pre_execution_observation(plan, pre_execution, now=self._now())
            _progress(progress, "runtime-plan", plan)
            _progress(
                progress,
                "remote-start-intent",
                {
                    "phase": session.phase,
                    "run_id": session.run["id"],
                    "head_sha": session.head_sha,
                    "job_key": job.key,
                    "job_id": job.job_id,
                    "runner_id": plan["job"]["runner_id"],
                    "runner_name": job.runner_name,
                    "runtime_plan_sha256": runtime.runtime_plan_sha256(plan),
                    "remote_start_retried": False,
                },
            )
        except BaseException:
            secret.destroy()
            self._retire_runner_if_present(
                jit_receipt["runner"]["id"],
                session=session,
                job=job,
                progress=progress,
            )
            raise
        try:
            execution = self._remote.run_job(run_binding, plan_bytes, secret)
        except BaseException:
            secret.destroy()
            reconciliation = self._reconcile_ambiguous_remote_start(
                session,
                job,
                jit_receipt["runner"]["id"],
                progress=progress,
            )
            raise AmbiguousRemoteExecution(reconciliation) from None
        finally:
            secret.destroy()
        try:
            _progress(
                progress,
                "remote-cleanup",
                {
                    "receipt": execution.receipt,
                    "stdout_sha256": execution.stdout_sha256,
                    "stderr_sha256": execution.stderr_sha256,
                    "frame_receipt": execution.frame_receipt,
                },
            )
            self._validate_remote_receipt(plan, execution)
        except BaseException:
            reconciliation = self._reconcile_ambiguous_remote_start(
                session,
                job,
                jit_receipt["runner"]["id"],
                progress=progress,
            )
            raise AmbiguousRemoteExecution(reconciliation) from None
        return plan, execution

    @staticmethod
    def _normalize_completed_job(raw: Any, plan: Mapping[str, Any]) -> dict[str, Any]:
        job = _required(
            raw,
            {
                "id",
                "name",
                "head_sha",
                "run_attempt",
                "status",
                "conclusion",
                "labels",
                "runner_id",
                "runner_name",
                "steps",
            },
            "completed_job",
        )
        _require(
            job["id"] == plan["job"]["job_id"]
            and job["name"] == plan["job"]["name"]
            and job["head_sha"] == plan["dispatch"]["head_sha"]
            and type(job["run_attempt"]) is int
            and job["run_attempt"] == 1
            and job["status"] == "completed"
            and job["conclusion"] == "success"
            and job["labels"] == [plan["job"]["runner_name"]]
            and job["runner_id"] == plan["job"]["runner_id"]
            and job["runner_name"] == plan["job"]["runner_name"],
            "completed_job_binding_rejected",
        )
        steps = job["steps"]
        _require(type(steps) is list, "completed_job_steps_not_list")
        expected_test = (
            "Run the CUDA release suite with zero skips"
            if plan["phase"] == "publication"
            else "Run every CUDA contract with zero skips"
        )
        expected_gpu = (
            "Prove real CUDA hardware"
            if plan["phase"] == "publication"
            else (
                "Prove two real visible CUDA devices"
                if plan["hardware"]["required_cuda_devices"] == 2
                else "Prove real driver, runtime, and device visibility"
            )
        )
        for expected in ("Require Linux GPU runner OS", expected_gpu, expected_test):
            selected = [
                item for item in steps if type(item) is dict and item.get("name") == expected
            ]
            _require(
                len(selected) == 1
                and selected[0].get("status") == "completed"
                and selected[0].get("conclusion") == "success",
                "completed_job_required_step_rejected",
            )
        return dict(job)

    @staticmethod
    def _validate_pytest_log(raw_log: bytes) -> tuple[int, int, str]:
        _require(0 < len(raw_log) <= MAX_LOG_BYTES, "job_log_size_rejected")
        try:
            text = raw_log.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            _fail("job_log_encoding_rejected")
        summaries = re.findall(r"(?<![0-9])15 passed(?: in|,)", text)
        skipped = re.findall(r"(?<![A-Za-z])([0-9]+) skipped(?![A-Za-z])", text)
        _require(len(summaries) == 1 and skipped == [], "pytest_15_zero_skip_evidence_missing")
        return 15, 0, _sha(raw_log)

    def settle_job(
        self,
        session: PhaseSession,
        plan: Mapping[str, Any],
        execution: RemoteExecution,
        *,
        poll_limit: int = 90,
        progress: ProgressSink | None = None,
    ) -> AcceptedJobReceipt:
        job_key = plan["job"]["key"]
        _require(job_key not in session.accepted, "job_already_settled")
        completed: dict[str, Any] | None = None
        job_digest = ""
        for _ in range(poll_limit):
            jobs, digests = self._attempt_jobs_with_digests(session.run["id"], 1)
            validated_jobs = self._validate_exact_attempt_job_set(
                jobs,
                expected_bindings=self._session_expected_job_bindings(session),
                head_sha=session.head_sha,
                context="job_settlement_attempt",
            )
            selected = [item for item in validated_jobs if item.get("id") == plan["job"]["job_id"]]
            _require(len(selected) == 1, "settlement_job_cardinality")
            if selected[0].get("status") == "completed":
                completed = self._normalize_completed_job(selected[0], plan)
                job_digest = _sha(_canonical(digests))
                break
            self._sleep(5)
        _require(completed is not None, "job_did_not_settle")
        checks, check_digest = self._request_json(
            "GET",
            f"/repos/{REPOSITORY}/commits/{session.head_sha}/check-runs"
            f"?check_name={job_spec_name_for_check(plan['job']['name'])}&filter=all",
        )
        check_mapping = _required(checks, {"total_count", "check_runs"}, "check_runs")
        selected_checks = [
            item
            for item in check_mapping["check_runs"]
            if type(item) is dict
            and item.get("name") == plan["job"]["name"]
            and item.get("head_sha") == session.head_sha
            and item.get("status") == "completed"
            and item.get("conclusion") == "success"
            and type(item.get("app")) is dict
            and item["app"].get("id") == CHECKS_APP_ID
            and str(item.get("details_url", ""))
            .rstrip("/")
            .endswith(f"/job/{plan['job']['job_id']}")
        ]
        _require(len(selected_checks) == 1, "accepted_check_binding_rejected")
        log_response = self._request(
            "GET",
            f"/repos/{REPOSITORY}/actions/jobs/{plan['job']['job_id']}/logs",
            expected=200,
        )
        try:
            raw_log = bytes(log_response.body)
            _, _, log_digest = self._validate_pytest_log(raw_log)
        finally:
            log_response.destroy()
        runners, inventory_digest = self._runner_inventory()
        _require(runners == [], "post_job_runner_inventory_not_zero")
        post_execution = self._live_observation(session)
        normalized_post = runtime.validate_post_execution_observation(
            plan, post_execution, now=self._now()
        )
        post_execution_digest = _sha(runtime.canonical_json(normalized_post))
        remote_digest = self._validate_remote_receipt(plan, execution)
        material = {
            "phase": session.phase,
            "run_id": session.run["id"],
            "job_key": job_key,
            "job_id": plan["job"]["job_id"],
            "runner_id": plan["job"]["runner_id"],
            "runner_name": plan["job"]["runner_name"],
            "runtime_plan_sha256": runtime.runtime_plan_sha256(plan),
            "remote_receipt_sha256": remote_digest,
            "actions_job_response_sha256": job_digest,
            "check_response_sha256": check_digest,
            "log_sha256": log_digest,
            "pytest_passed": 15,
            "pytest_skipped": 0,
            "runner_inventory_response_sha256": inventory_digest,
            "post_execution_observation_sha256": post_execution_digest,
        }
        accepted = AcceptedJobReceipt(**material, evidence_sha256=_sha(_canonical(material)))
        session.accepted[job_key] = accepted
        _progress(progress, "accepted-actions-job", accepted.to_mapping())
        return accepted

    def cancel_pull_request_after_singles(
        self,
        session: PhaseSession,
        *,
        poll_limit: int = 60,
        progress: ProgressSink | None = None,
    ) -> dict[str, Any]:
        _require(session.phase == "pull-request", "cancel_phase_rejected")
        _require(
            tuple(session.accepted) == tuple(PHASES["pull-request"]["job_keys"]),
            "pr_singles_not_accepted",
        )
        runners, before_digest = self._runner_inventory()
        _require(runners == [], "pr_cancel_runner_inventory_not_zero")
        cancel_path = f"/repos/{REPOSITORY}/actions/runs/{session.run['id']}/cancel"
        cancel_request_sha256 = _sha(
            _canonical({"method": "POST", "path": cancel_path, "body": None})
        )
        cancel_intent = {
            "phase": session.phase,
            "run_id": session.run["id"],
            "run_attempt": 1,
            "head_sha": session.head_sha,
            "cancel_path": cancel_path,
            "request_sha256": cancel_request_sha256,
            "reason": "pull-request-singles-accepted",
            "accepted_job_ids": [item.job_id for item in session.accepted.values()],
            "serviced_job_ids": [],
            "mutation_retried": False,
        }
        _progress(progress, "github-cancel-intent", cancel_intent)
        cancel_ambiguity: AmbiguousGitHubMutation | None = None
        cancel_digest: str | None = None
        try:
            cancel_digest = self._request_empty("POST", cancel_path, expected=202)
        except AmbiguousGitHubMutation as ambiguity:
            cancel_ambiguity = ambiguity
        final_run: Mapping[str, Any] | None = None
        run_digest = ""
        for _ in range(poll_limit):
            value, digest = self._request_json(
                "GET", f"/repos/{REPOSITORY}/actions/runs/{session.run['id']}"
            )
            run = _required(value, {"id", "run_attempt", "status", "conclusion"}, "cancelled_run")
            if run["status"] == "completed":
                _require(
                    run["id"] == session.run["id"]
                    and type(run["run_attempt"]) is int
                    and run["run_attempt"] == 1
                    and run["conclusion"] == "cancelled",
                    "pr_cancel_settlement_rejected",
                )
                final_run = run
                run_digest = digest
                break
            self._sleep(2)
        if final_run is None:
            raise AmbiguousGitHubMutation(
                "POST",
                cancel_path,
                cancel_request_sha256,
                "run_cancel_unresolved",
                reconciliation={
                    "run_id": session.run["id"],
                    "run_attempt": 1,
                    "reason": "pull-request-singles-accepted",
                    "accepted_job_ids": [item.job_id for item in session.accepted.values()],
                    "serviced_job_ids": [],
                    "response_received": cancel_ambiguity is None,
                    "response_sha256": cancel_digest,
                    "cancel_retried": False,
                },
            )
        assert final_run is not None
        settled_jobs = self._validate_exact_attempt_job_set(
            self._attempt_jobs(session.run["id"], 1),
            expected_bindings=self._session_expected_job_bindings(session),
            head_sha=session.head_sha,
            context="pr_cancel_settled_attempt",
        )
        by_id = {job["id"]: job for job in settled_jobs}
        service_ids = {job.job_id for job in session.jobs}
        withheld = [job for job in session.queued_jobs if job.job_id not in service_ids]
        _require(len(withheld) == 2, "pr_withheld_job_count_rejected")
        for job in session.jobs:
            settled = by_id.get(job.job_id)
            _require(
                type(settled) is dict
                and settled.get("status") == "completed"
                and settled.get("conclusion") == "success",
                "pr_protected_single_settlement_rejected",
            )
        for job in withheld:
            settled = by_id.get(job.job_id)
            _require(
                type(settled) is dict
                and settled.get("status") == "completed"
                and settled.get("conclusion") == "cancelled"
                and settled.get("runner_id") is None
                and settled.get("runner_name") in {None, ""},
                "pr_withheld_job_not_cancelled",
            )
        runners, after_digest = self._runner_inventory()
        _require(runners == [], "pr_cancel_post_inventory_not_zero")
        material = {
            "phase": "pull-request",
            "run_id": session.run["id"],
            "attempt": 1,
            "protected_single_job_ids": [receipt.job_id for receipt in session.accepted.values()],
            "protected_single_conclusions": ["success", "success"],
            "withheld_two_gpu_job_ids": [job.job_id for job in withheld],
            "withheld_two_gpu_conclusions": ["cancelled", "cancelled"],
            "withheld_two_gpu_runner_ids": [None, None],
            "cancel_request_sha256": cancel_request_sha256,
            "cancel_response_sha256": cancel_digest,
            "cancel_response_ambiguous": cancel_ambiguity is not None,
            "settled_run_response_sha256": run_digest,
            "runner_inventory_before_sha256": before_digest,
            "runner_inventory_after_sha256": after_digest,
            "rerun_performed": False,
        }
        result = {**material, "evidence_sha256": _sha(_canonical(material))}
        _progress(progress, "github-cancel-settled", result)
        return result

    def settle_final_main(self, session: PhaseSession, *, poll_limit: int = 90) -> dict[str, Any]:
        _require(session.phase == "final-main", "final_settlement_phase_rejected")
        _require(
            tuple(session.accepted) == tuple(PHASES["final-main"]["job_keys"]),
            "final_jobs_not_accepted",
        )
        run_digest = ""
        for _ in range(poll_limit):
            value, digest = self._request_json(
                "GET", f"/repos/{REPOSITORY}/actions/runs/{session.run['id']}"
            )
            run = _required(
                value, {"id", "run_attempt", "status", "conclusion", "head_sha"}, "final_run"
            )
            if run["status"] == "completed":
                _require(
                    run["id"] == session.run["id"]
                    and type(run["run_attempt"]) is int
                    and run["run_attempt"] == 1
                    and run["conclusion"] == "success"
                    and run["head_sha"] == session.head_sha,
                    "final_run_not_accepted",
                )
                run_digest = digest
                break
            self._sleep(5)
        _require(bool(run_digest), "final_run_not_settled")
        settled_jobs = self._validate_exact_attempt_job_set(
            self._attempt_jobs(session.run["id"], 1),
            expected_bindings=self._session_expected_job_bindings(session),
            head_sha=session.head_sha,
            context="final_main_settled_attempt",
        )
        _require(
            all(
                item.get("status") == "completed" and item.get("conclusion") == "success"
                for item in settled_jobs
            ),
            "final_main_job_set_not_accepted",
        )
        runners, inventory_digest = self._runner_inventory()
        _require(runners == [], "final_run_runner_inventory_not_zero")
        material = {
            "phase": "final-main",
            "run_id": session.run["id"],
            "run_attempt": 1,
            "head_sha": session.head_sha,
            "accepted_cuda_runner_nonces": [job.nonce for job in session.jobs],
            "job_evidence_sha256": [item.evidence_sha256 for item in session.accepted.values()],
            "run_response_sha256": run_digest,
            "runner_inventory_response_sha256": inventory_digest,
            "all_four_jobs_15_of_15_zero_skips": True,
            "rerun_performed": False,
        }
        return {**material, "evidence_sha256": _sha(_canonical(material))}

    def cancel_failed_phase(
        self,
        session: PhaseSession,
        *,
        poll_limit: int = 60,
        progress: ProgressSink | None = None,
    ) -> dict[str, Any]:
        """Cancel an incomplete phase once fresh inventory proves no runner exists."""

        _require(
            tuple(session.accepted) != tuple(PHASES[session.phase]["job_keys"]),
            "failed_phase_already_complete",
        )
        runners, before_digest = self._runner_inventory()
        _require(runners == [], "failed_phase_runner_inventory_not_zero")
        cancel_path = f"/repos/{REPOSITORY}/actions/runs/{session.run['id']}/cancel"
        cancel_request_sha256 = _sha(
            _canonical({"method": "POST", "path": cancel_path, "body": None})
        )
        cancel_intent = {
            "phase": session.phase,
            "run_id": session.run["id"],
            "run_attempt": 1,
            "head_sha": session.head_sha,
            "cancel_path": cancel_path,
            "request_sha256": cancel_request_sha256,
            "reason": "incomplete-phase",
            "accepted_job_ids": [item.job_id for item in session.accepted.values()],
            "serviced_job_ids": [],
            "mutation_retried": False,
        }
        _progress(progress, "github-cancel-intent", cancel_intent)
        cancel_ambiguity: AmbiguousGitHubMutation | None = None
        cancel_digest: str | None = None
        try:
            cancel_digest = self._request_empty("POST", cancel_path, expected=202)
        except AmbiguousGitHubMutation as exc:
            cancel_ambiguity = exc
        terminal: Mapping[str, Any] | None = None
        run_digest = ""
        for _ in range(poll_limit):
            value, digest = self._request_json(
                "GET", f"/repos/{REPOSITORY}/actions/runs/{session.run['id']}"
            )
            run = _required(
                value, {"id", "run_attempt", "status", "conclusion", "head_sha"}, "abort_run"
            )
            if run["status"] == "completed":
                _require(
                    run["id"] == session.run["id"]
                    and type(run["run_attempt"]) is int
                    and run["run_attempt"] == 1
                    and run["head_sha"] == session.head_sha
                    and run["conclusion"] in {"cancelled", "failure"},
                    "failed_phase_terminal_state_rejected",
                )
                terminal = run
                run_digest = digest
                break
            self._sleep(2)
        if terminal is None:
            raise AmbiguousGitHubMutation(
                "POST",
                cancel_path,
                cancel_request_sha256,
                "failed_phase_cancel_unresolved",
                reconciliation={
                    "run_id": session.run["id"],
                    "reason": "incomplete-phase",
                    "accepted_job_ids": [item.job_id for item in session.accepted.values()],
                    "serviced_job_ids": [],
                    "response_received": cancel_ambiguity is None,
                    "response_sha256": cancel_digest,
                    "cancel_retried": False,
                },
            ) from None
        assert terminal is not None
        jobs = self._validate_exact_attempt_job_set(
            self._attempt_jobs(session.run["id"], 1),
            expected_bindings=self._session_expected_job_bindings(session),
            head_sha=session.head_sha,
            context="failed_phase_settled_attempt",
        )
        by_id = {item["id"]: item for item in jobs}
        accepted_ids = {item.job_id for item in session.accepted.values()}
        for binding in session.queued_jobs:
            job = by_id.get(binding.job_id)
            _require(
                type(job) is dict and job.get("status") == "completed", "abort_job_not_terminal"
            )
            assert isinstance(job, dict)
            if binding.job_id in accepted_ids:
                _require(job.get("conclusion") == "success", "abort_accepted_job_drift")
            else:
                _require(
                    job.get("conclusion") in {"cancelled", "skipped"}
                    and job.get("runner_id") is None
                    and job.get("runner_name") in {None, ""},
                    "abort_unaccepted_job_was_serviced",
                )
        runners, after_digest = self._runner_inventory()
        _require(runners == [], "failed_phase_post_cancel_runner_inventory_not_zero")
        material = {
            "phase": session.phase,
            "run_id": session.run["id"],
            "run_attempt": 1,
            "head_sha": session.head_sha,
            "conclusion": terminal["conclusion"],
            "cancel_request_sha256": cancel_request_sha256,
            "cancel_response_sha256": cancel_digest,
            "cancel_response_ambiguous": cancel_ambiguity is not None,
            "run_response_sha256": run_digest,
            "runner_inventory_before_sha256": before_digest,
            "runner_inventory_after_sha256": after_digest,
            "rerun_performed": False,
        }
        result = {**material, "evidence_sha256": _sha(_canonical(material))}
        _progress(progress, "github-cancel-settled", result)
        return result

    def reconcile_cancel_for_abort(
        self,
        session: PhaseSession,
        ambiguity: AmbiguousGitHubMutation,
        *,
        poll_limit: int = 60,
    ) -> dict[str, Any]:
        """Settle an already-attempted run cancellation without replaying it."""

        cancel_path = f"/repos/{REPOSITORY}/actions/runs/{session.run['id']}/cancel"
        _require(
            type(poll_limit) is int
            and poll_limit > 0
            and ambiguity.method == "POST"
            and ambiguity.path == cancel_path
            and ambiguity.request_sha256
            == _sha(_canonical({"method": "POST", "path": cancel_path, "body": None})),
            "abort_cancel_ambiguity_binding_rejected",
        )
        supplied_run_id = ambiguity.reconciliation.get("run_id")
        _require(
            supplied_run_id in {None, session.run["id"]},
            "abort_cancel_ambiguity_run_drift",
        )
        runners, before_digest = self._runner_inventory()
        _require(runners == [], "abort_cancel_runner_inventory_not_zero")
        terminal: Mapping[str, Any] | None = None
        run_digest = ""
        for _ in range(poll_limit):
            value, digest = self._request_json(
                "GET", f"/repos/{REPOSITORY}/actions/runs/{session.run['id']}"
            )
            run = _required(
                value,
                {"id", "run_attempt", "status", "conclusion", "head_sha"},
                "abort_cancel_run",
            )
            if run["status"] == "completed":
                _require(
                    run["id"] == session.run["id"]
                    and type(run["run_attempt"]) is int
                    and run["run_attempt"] == 1
                    and run["head_sha"] == session.head_sha,
                    "abort_cancel_terminal_binding_drift",
                )
                terminal = run
                run_digest = digest
                break
            self._sleep(2)
        _require(terminal is not None, "abort_cancel_still_unresolved")
        assert terminal is not None
        complete = tuple(session.accepted) == tuple(item.key for item in session.jobs)
        raw_serviced = ambiguity.reconciliation.get("serviced_job_ids", [])
        _require(
            type(raw_serviced) is list
            and len(raw_serviced) == len(set(raw_serviced))
            and all(type(item) is int and item > 0 for item in raw_serviced),
            "abort_cancel_serviced_jobs_rejected",
        )
        serviced_ids = set(raw_serviced)
        _require(
            serviced_ids <= {item.job_id for item in session.queued_jobs},
            "abort_cancel_serviced_job_not_in_session",
        )
        if complete and session.phase == "pull-request":
            _require(
                terminal["conclusion"] == "cancelled",
                "abort_pr_cancel_conclusion_drift",
            )
        else:
            _require(not complete, "abort_cancel_complete_non_pr_rejected")
            _require(
                terminal["conclusion"]
                in (
                    {"success", "cancelled", "failure"}
                    if serviced_ids
                    else {"cancelled", "failure"}
                ),
                "abort_failed_cancel_conclusion_drift",
            )
        jobs = self._validate_exact_attempt_job_set(
            self._attempt_jobs(session.run["id"], 1),
            expected_bindings=self._session_expected_job_bindings(session),
            head_sha=session.head_sha,
            context="abort_cancel_settled_attempt",
        )
        by_id = {item["id"]: item for item in jobs}
        accepted_ids = {item.job_id for item in session.accepted.values()}
        for binding in session.queued_jobs:
            job = by_id.get(binding.job_id)
            _require(
                type(job) is dict and job.get("status") == "completed",
                "abort_cancel_job_not_terminal",
            )
            assert isinstance(job, dict)
            if binding.job_id in accepted_ids:
                _require(
                    job.get("conclusion") == "success",
                    "abort_cancel_accepted_job_drift",
                )
            elif binding.job_id in serviced_ids:
                _require(
                    job.get("conclusion") in {"success", "failure", "cancelled", "skipped"},
                    "abort_cancel_serviced_job_conclusion_rejected",
                )
            else:
                _require(
                    job.get("conclusion") in {"cancelled", "skipped"}
                    and job.get("runner_id") is None
                    and job.get("runner_name") in {None, ""},
                    "abort_cancel_unaccepted_job_was_serviced",
                )
        runners, after_digest = self._runner_inventory()
        _require(runners == [], "abort_cancel_post_inventory_not_zero")
        material = {
            "phase": session.phase,
            "run_id": session.run["id"],
            "run_attempt": 1,
            "head_sha": session.head_sha,
            "conclusion": terminal["conclusion"],
            "serviced_job_ids": sorted(serviced_ids),
            "ambiguous_cancel_request_sha256": ambiguity.request_sha256,
            "run_response_sha256": run_digest,
            "runner_inventory_before_sha256": before_digest,
            "runner_inventory_after_sha256": after_digest,
            "cancel_retried": False,
        }
        return {**material, "evidence_sha256": _sha(_canonical(material))}

    def cancel_crashed_phase(
        self,
        session: PhaseSession,
        *,
        serviced_job_ids: Sequence[int],
        poll_limit: int = 60,
        progress: ProgressSink | None = None,
    ) -> dict[str, Any]:
        """Settle a crash-tainted run without accepting any unjournaled job evidence."""

        serviced = tuple(serviced_job_ids)
        _require(
            type(poll_limit) is int
            and poll_limit > 0
            and len(serviced) == len(set(serviced))
            and all(type(item) is int and item > 0 for item in serviced),
            "crash_serviced_job_ids_rejected",
        )
        queued_ids = {item.job_id for item in session.queued_jobs}
        _require(set(serviced) <= queued_ids, "crash_serviced_job_not_in_session")
        runners, before_digest = self._runner_inventory()
        _require(runners == [], "crash_cancel_runner_inventory_not_zero")
        run_path = f"/repos/{REPOSITORY}/actions/runs/{session.run['id']}"
        value, run_digest = self._request_json("GET", run_path)
        run = _required(
            value, {"id", "run_attempt", "status", "conclusion", "head_sha"}, "crash_run"
        )
        _require(
            run["id"] == session.run["id"]
            and type(run["run_attempt"]) is int
            and run["run_attempt"] == 1
            and run["head_sha"] == session.head_sha,
            "crash_run_binding_drift",
        )
        cancel_digest: str | None = None
        cancel_ambiguity: AmbiguousGitHubMutation | None = None
        if run["status"] != "completed":
            cancel_path = f"{run_path}/cancel"
            cancel_request_sha256 = _sha(
                _canonical({"method": "POST", "path": cancel_path, "body": None})
            )
            _progress(
                progress,
                "github-cancel-intent",
                {
                    "phase": session.phase,
                    "run_id": session.run["id"],
                    "run_attempt": 1,
                    "head_sha": session.head_sha,
                    "cancel_path": cancel_path,
                    "request_sha256": cancel_request_sha256,
                    "reason": "controller-process-crash",
                    "accepted_job_ids": [item.job_id for item in session.accepted.values()],
                    "serviced_job_ids": list(serviced),
                    "mutation_retried": False,
                },
            )
            try:
                cancel_digest = self._request_empty("POST", cancel_path, expected=202)
            except AmbiguousGitHubMutation as exc:
                cancel_ambiguity = exc
            for _ in range(poll_limit):
                value, run_digest = self._request_json("GET", run_path)
                run = _required(
                    value,
                    {"id", "run_attempt", "status", "conclusion", "head_sha"},
                    "crash_cancel_run",
                )
                _require(
                    run["id"] == session.run["id"]
                    and type(run["run_attempt"]) is int
                    and run["run_attempt"] == 1
                    and run["head_sha"] == session.head_sha,
                    "crash_cancel_run_binding_drift",
                )
                if run["status"] == "completed":
                    break
                self._sleep(2)
        if run["status"] != "completed":
            cancel_path = f"{run_path}/cancel"
            cancel_request_sha256 = _sha(
                _canonical({"method": "POST", "path": cancel_path, "body": None})
            )
            raise AmbiguousGitHubMutation(
                "POST",
                cancel_path,
                cancel_request_sha256,
                "crash_cancel_unresolved",
                reconciliation={
                    "run_id": session.run["id"],
                    "reason": "controller-process-crash",
                    "accepted_job_ids": [item.job_id for item in session.accepted.values()],
                    "serviced_job_ids": list(serviced),
                    "response_received": cancel_ambiguity is None,
                    "response_sha256": cancel_digest,
                    "cancel_retried": False,
                },
            ) from None
        _require(
            run["status"] == "completed"
            and run["conclusion"] in {"success", "failure", "cancelled"},
            "crash_run_not_terminal",
        )
        jobs = self._validate_exact_attempt_job_set(
            self._attempt_jobs(session.run["id"], 1),
            expected_bindings=self._session_expected_job_bindings(session),
            head_sha=session.head_sha,
            context="crash_cancel_settled_attempt",
        )
        by_id = {item["id"]: item for item in jobs}
        accepted_ids = {item.job_id for item in session.accepted.values()}
        for binding in session.queued_jobs:
            job = by_id.get(binding.job_id)
            _require(
                type(job) is dict and job.get("status") == "completed",
                "crash_cancel_job_not_terminal",
            )
            assert isinstance(job, dict)
            if binding.job_id in accepted_ids:
                _require(
                    job.get("conclusion") == "success",
                    "crash_cancel_accepted_job_drift",
                )
            elif binding.job_id in serviced:
                _require(
                    job.get("conclusion") in {"success", "failure", "cancelled", "skipped"},
                    "crash_serviced_job_conclusion_rejected",
                )
            else:
                _require(
                    job.get("conclusion") in {"cancelled", "skipped"}
                    and job.get("runner_id") is None
                    and job.get("runner_name") in {None, ""},
                    "crash_unserviced_job_was_claimed",
                )
        runners, after_digest = self._runner_inventory()
        _require(runners == [], "crash_cancel_post_inventory_not_zero")
        material = {
            "phase": session.phase,
            "run_id": session.run["id"],
            "run_attempt": 1,
            "head_sha": session.head_sha,
            "conclusion": run["conclusion"],
            "serviced_job_ids": list(serviced),
            "accepted_job_ids": sorted(accepted_ids),
            "cancel_response_sha256": cancel_digest,
            "cancel_response_ambiguous": cancel_ambiguity is not None,
            "run_response_sha256": run_digest,
            "runner_inventory_before_sha256": before_digest,
            "runner_inventory_after_sha256": after_digest,
            "unjournaled_job_evidence_accepted": False,
            "rerun_performed": False,
        }
        result = {**material, "evidence_sha256": _sha(_canonical(material))}
        _progress(progress, "github-cancel-settled", result)
        return result

    def publication_gpu_gate(self, session: PhaseSession) -> dict[str, Any]:
        _require(session.phase == "publication", "publication_gate_phase_rejected")
        _require(
            tuple(session.accepted) == tuple(PHASES["publication"]["job_keys"]),
            "publication_gpu_jobs_not_accepted",
        )
        settled_jobs = self._validate_exact_attempt_job_set(
            self._attempt_jobs(session.run["id"], 1),
            expected_bindings=self._session_expected_job_bindings(session),
            head_sha=session.head_sha,
            context="publication_gate_attempt",
        )
        _require(
            all(
                item.get("status") == "completed" and item.get("conclusion") == "success"
                for item in settled_jobs
            ),
            "publication_gate_job_set_not_accepted",
        )
        runners, inventory_digest = self._runner_inventory()
        _require(runners == [], "publication_runner_inventory_not_zero")
        material = {
            "phase": "publication",
            "run_id": session.run["id"],
            "run_attempt": 1,
            "head_sha": session.head_sha,
            "tag": runtime.PUBLICATION_TAG,
            "stage_recovery_drill": True,
            "job_evidence_sha256": [item.evidence_sha256 for item in session.accepted.values()],
            "runner_inventory_response_sha256": inventory_digest,
            "both_release_jobs_15_of_15_zero_skips": True,
            "workflow_publication_success_not_claimed": True,
            "rerun_performed": False,
        }
        return {**material, "evidence_sha256": _sha(_canonical(material))}


def job_spec_name_for_check(name: str) -> str:
    """Percent-encode the one query value without importing a URL client."""

    from urllib.parse import quote

    return quote(name, safe="")
