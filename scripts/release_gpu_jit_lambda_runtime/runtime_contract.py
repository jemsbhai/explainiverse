"""Fail-closed contracts for the disposable Lambda GPU runner runtime.

This module contains no network or subprocess calls.  It validates the exact
runtime envelope produced by an independently reviewed control plane, models
the Docker/network argv, and validates normalized action-time GitHub reads.
The encoded JIT configuration and GitHub credential are deliberately absent
from every mapping handled here.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, NoReturn, Sequence, cast

SCHEMA_VERSION = 1
PLAN_KIND = "explainiverse-lambda-jit-runtime-plan"
RECEIPT_KIND = "explainiverse-lambda-jit-runtime-receipt"
HOST_PREFLIGHT_KIND = "explainiverse-lambda-jit-host-preflight"
REPOSITORY = "jemsbhai/explainiverse"
CUDA_WORKFLOW_PATH = ".github/workflows/cuda-ci.yml"
PUBLISH_WORKFLOW_PATH = ".github/workflows/publish-pypi.yml"
# Kept as the CUDA workflow alias for callers which predate the publication lane.
WORKFLOW_PATH = CUDA_WORKFLOW_PATH
PULL_REQUEST_REF = "refs/heads/codex/harden-cuda-runner-routing"
FINAL_MAIN_REF = "refs/heads/main"
PUBLICATION_TAG = "v0.15.0"
PUBLICATION_REF = f"refs/tags/{PUBLICATION_TAG}"
OWNER = "jemsbhai"
GITHUB_API_VERSION = "2026-03-10"

RUNNER_VERSION = "2.336.0"
RUNNER_FILENAME = f"actions-runner-linux-x64-{RUNNER_VERSION}.tar.gz"
RUNNER_DOWNLOAD_URL = (
    "https://github.com/actions/runner/releases/download/" f"v{RUNNER_VERSION}/{RUNNER_FILENAME}"
)
RUNNER_ARCHIVE_SHA256 = "04cf0be1aff4c3ec3554466c39124ca250e3effd8873bb7e8d68535aa9505d5d"

IMAGE_TAG_REFERENCE = f"ghcr.io/actions/actions-runner:{RUNNER_VERSION}"
IMAGE_MANIFEST_DIGEST = "sha256:a1919047b038c38871d667c58cfdc7a878452711ab1212fb6036188f27a7ab16"
IMAGE_REFERENCE = f"ghcr.io/actions/actions-runner@{IMAGE_MANIFEST_DIGEST}"
IMAGE_MANIFEST_MEDIA_TYPE = "application/vnd.oci.image.manifest.v1+json"
IMAGE_MANIFEST_SIZE = 2332
IMAGE_CONFIG_DIGEST = "sha256:bd6fe162bb4ab4821daa8d694e20d779865618825d30c94342a0228b89947305"
IMAGE_CONFIG_MEDIA_TYPE = "application/vnd.oci.image.config.v1+json"
IMAGE_CONFIG_SIZE = 5405
IMAGE_PLATFORM = "linux/amd64"
IMAGE_MANIFEST_SOURCE = "docker manifest inspect --verbose ghcr.io/actions/actions-runner:2.336.0"
IMAGE_MANIFEST_OBSERVED_AT = "2026-08-28T21:05:51.032Z"
IMAGE_RUNNER_COMMIT = "98aabcd429c4e8402406c56ce2d26387fed3b9ce"
IMAGE_NODE20_VERSION = "v20.20.2"
IMAGE_NODE20_SHA256 = "6295488653f0d93b0a157841746fef7e72cc4328cfb60c4bbe0ca2668a836ffd"

DOCKER_PATH = "/usr/bin/docker"
IPTABLES_PATH = "/usr/sbin/iptables"
NVIDIA_SMI_PATH = "/usr/bin/nvidia-smi"
PYTHON_PATH = "/usr/bin/python3"
CLOUD_INIT_PATH = "/usr/bin/cloud-init"
CONTAINER_UID = 1001
CONTAINER_GID = 1001
HARD_WALL_SECONDS = 3000
CLEANUP_GRACE_SECONDS = 60
FD_READ_SECONDS = 15
POST_GITHUB_SETTLE_SECONDS = 120
AUTHORITY_MAX_SECONDS = 1800
PLAN_MAX_AGE_SECONDS = 300
OBSERVATION_MAX_AGE_SECONDS = 900
NETWORK_SUBNET = "172.31.240.0/28"
PROBE_CONTAINER_NAME = "explainiverse-jit-image-probe"
GLOBAL_RUNTIME_LABEL = "org.opencontainers.image.vendor=explainiverse-release-control"
HOST_PHYSICAL_GPU_COUNT = 8
HOST_GPU_PRODUCT = "NVIDIA A100-SXM4-80GB"

SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
NONCE_RE = re.compile(r"[0-9a-f]{16}\Z")
GPU_UUID_RE = re.compile(
    r"GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-" r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\Z"
)
RUNNER_NAME_RE = re.compile(r"explainiverse-cuda-(single|two)-jit-[0-9a-f]{16}\Z")
WORK_FOLDER_RE = re.compile(r"_work-[0-9a-f]{16}\Z")

JOB_SPECS = {
    "single_minimum": {
        "name": "CUDA single-GPU (Torch minimum)",
        "prefix": "explainiverse-cuda-single-jit-",
        "gpu_count": 1,
        "cuda_visible_devices": "0",
    },
    "single_latest": {
        "name": "CUDA single-GPU (Torch latest)",
        "prefix": "explainiverse-cuda-single-jit-",
        "gpu_count": 1,
        "cuda_visible_devices": "0",
    },
    "two_minimum": {
        "name": "CUDA two-GPU scheduled (Torch minimum)",
        "prefix": "explainiverse-cuda-two-jit-",
        "gpu_count": 2,
        "cuda_visible_devices": "0,1",
    },
    "two_latest": {
        "name": "CUDA two-GPU scheduled (Torch latest)",
        "prefix": "explainiverse-cuda-two-jit-",
        "gpu_count": 2,
        "cuda_visible_devices": "0,1",
    },
    "publication_single_minimum": {
        "name": "Release CUDA single-GPU (Torch minimum, zero skips)",
        "prefix": "explainiverse-cuda-single-jit-",
        "gpu_count": 1,
        "cuda_visible_devices": "0",
    },
    "publication_single_latest": {
        "name": "Release CUDA single-GPU (Torch latest, zero skips)",
        "prefix": "explainiverse-cuda-single-jit-",
        "gpu_count": 1,
        "cuda_visible_devices": "0",
    },
}

CUDA_NONCE_INPUT_KEYS = (
    "single_minimum_runner_nonce",
    "single_latest_runner_nonce",
    "two_minimum_runner_nonce",
    "two_latest_runner_nonce",
)
PUBLICATION_NONCE_INPUT_KEYS = (
    "single_minimum_runner_nonce",
    "single_latest_runner_nonce",
)
PHASE_SPECS = {
    "pull-request": {
        "workflow_path": CUDA_WORKFLOW_PATH,
        "ref": PULL_REQUEST_REF,
        "job_keys": ("single_minimum", "single_latest"),
        "nonce_input_keys": CUDA_NONCE_INPUT_KEYS,
        "prior_accepted_nonce_count": 0,
    },
    "final-main": {
        "workflow_path": CUDA_WORKFLOW_PATH,
        "ref": FINAL_MAIN_REF,
        "job_keys": ("single_minimum", "single_latest", "two_minimum", "two_latest"),
        "nonce_input_keys": CUDA_NONCE_INPUT_KEYS,
        "prior_accepted_nonce_count": 0,
    },
    "publication": {
        "workflow_path": PUBLISH_WORKFLOW_PATH,
        "ref": PUBLICATION_REF,
        "job_keys": ("publication_single_minimum", "publication_single_latest"),
        "nonce_input_keys": PUBLICATION_NONCE_INPUT_KEYS,
        "prior_accepted_nonce_count": 4,
    },
}

PLAN_KEYS = {
    "schema_version",
    "kind",
    "execution_authorized",
    "created_at",
    "policy_sha256",
    "control_plane_plan_sha256",
    "runtime_bundle_sha256",
    "phase",
    "repository",
    "workflow_path",
    "authority_window",
    "dispatch",
    "job",
    "sequencing",
    "hardware",
    "runner_source",
    "runner_image",
    "github_evidence",
    "limits",
}
AUTHORITY_KEYS = {
    "observed_at",
    "expires_at",
    "evidence_sha256",
    "owner_login",
    "collaborators",
    "pending_invitation_count",
    "enabled_nonowner_authorities",
    "unexpected_target_job_count",
}
COLLABORATOR_KEYS = {"login", "permission"}
DISPATCH_KEYS = {
    "observed_at",
    "request_sha256",
    "response_sha256",
    "event",
    "ref",
    "inputs",
    "prior_accepted_cuda_runner_nonces",
    "run_id",
    "run_attempt",
    "head_sha",
    "actor",
    "triggering_actor",
    "status",
    "conclusion",
}
CUDA_DISPATCH_INPUT_KEYS = set(CUDA_NONCE_INPUT_KEYS)
PUBLICATION_DISPATCH_INPUT_KEYS = {
    "tag",
    "preflight_run_id",
    "cuda_run_id",
    "single_minimum_runner_nonce",
    "single_latest_runner_nonce",
    "stage_recovery_drill",
}
JOB_KEYS = {
    "ordinal",
    "key",
    "job_id",
    "name",
    "runner_nonce",
    "runner_id",
    "runner_name",
    "labels",
    "status",
    "conclusion",
    "work_folder",
    "jit_config_sha256",
}
SEQUENCING_KEYS = {"sequential_only", "previous_cleanup_receipt_sha256"}
HARDWARE_KEYS = {
    "host_physical_gpu_count",
    "host_physical_gpu_uuids",
    "host_physical_gpu_products",
    "assigned_physical_gpu_uuids",
    "unrequested_physical_gpu_uuids",
    "device_request",
    "nvidia_visible_devices",
    "cuda_visible_devices",
    "required_cuda_devices",
    "exclusive_device_scope_required",
}
RUNNER_SOURCE_KEYS = {
    "observed_at",
    "response_sha256",
    "api_version",
    "os",
    "architecture",
    "filename",
    "download_url",
    "sha256_checksum",
    "version",
}
RUNNER_IMAGE_KEYS = {
    "tag_reference",
    "image_reference",
    "platform",
    "manifest_digest",
    "manifest_media_type",
    "manifest_size",
    "config_digest",
    "config_media_type",
    "config_size",
    "manifest_source",
    "manifest_observed_at",
    "probe_observed_at",
    "probe_receipt_sha256",
    "container_uid",
    "container_gid",
    "runner_listener_present",
    "runner_listener_version",
    "runner_commit",
    "node20_present",
    "node20_version",
    "node20_sha256",
}
GITHUB_EVIDENCE_KEYS = {
    "pre_jit_registration_absence",
    "nonce_history",
    "jit_response",
}
ABSENCE_KEYS = {"observed_at", "response_sha256", "total_count", "runners"}
NONCE_HISTORY_KEYS = {
    "observed_at",
    "response_sha256",
    "historical_match_count",
    "unexpected_queued_or_in_progress_count",
}
JIT_RESPONSE_KEYS = {"observed_at", "response_sha256", "runner"}
RUNNER_KEYS = {"id", "name", "os", "status", "busy", "labels"}
LIMIT_KEYS = {
    "hard_wall_seconds",
    "fd_read_seconds",
    "post_github_settle_seconds",
    "external_watchdog_required",
    "cleanup_grace_seconds",
}

LIVE_OBSERVATION_KEYS = {
    "captured_at",
    "run_response_sha256",
    "jobs_response_sha256",
    "downloads_response_sha256",
    "runners_response_sha256",
    "run",
    "jobs",
    "downloads",
    "runners",
}
LIVE_RUN_KEYS = {
    "id",
    "event",
    "path",
    "ref",
    "head_sha",
    "run_attempt",
    "actor",
    "triggering_actor",
    "status",
    "conclusion",
}
LIVE_JOB_KEYS = {
    "id",
    "name",
    "head_sha",
    "run_attempt",
    "status",
    "conclusion",
    "labels",
    "runner_id",
    "runner_name",
}
DOWNLOAD_KEYS = {"os", "architecture", "filename", "download_url", "sha256_checksum"}

BLOCKED_IPV4_DESTINATIONS = (
    "0.0.0.0/8",
    "10.0.0.0/8",
    "100.64.0.0/10",
    "127.0.0.0/8",
    "169.254.0.0/16",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "224.0.0.0/4",
    "240.0.0.0/4",
)

CONTAINER_LAUNCHER = r"""
set -Eeuo pipefail
set +x
umask 077
ulimit -c 0
test "$(id -u)" = "1001"
test "$(id -g)" = "1001"
test -x /home/runner/run.sh
test -x /home/runner/bin/Runner.Listener
test -x /home/runner/externals/node20/bin/node
cp -r --no-preserve=ownership /home/runner/. /runner/
test -x /runner/run.sh
mapfile -t observed_gpu_uuids < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
IFS=',' read -r -a expected_gpu_uuids <<< "${EXPLAINIVERSE_ASSIGNED_GPU_UUIDS}"
test "${#observed_gpu_uuids[@]}" -eq "${#expected_gpu_uuids[@]}"
for index in "${!expected_gpu_uuids[@]}"; do
  test "${observed_gpu_uuids[$index]}" = "${expected_gpu_uuids[$index]}"
done
test "${NVIDIA_VISIBLE_DEVICES}" = "${EXPLAINIVERSE_ASSIGNED_GPU_UUIDS}"
test "${CUDA_VISIBLE_DEVICES}" = "${EXPLAINIVERSE_LOGICAL_CUDA_VISIBLE_DEVICES}"
IFS= read -r -t 15 ACTIONS_RUNNER_INPUT_JITCONFIG
test -n "${ACTIONS_RUNNER_INPUT_JITCONFIG}"
export ACTIONS_RUNNER_INPUT_JITCONFIG
cd /runner
exec /runner/run.sh
""".strip()

IMAGE_PROBE_SCRIPT = r"""
set -Eeuo pipefail
test "$(id -u)" = "1001"
test "$(id -g)" = "1001"
test -x /home/runner/run.sh
test -x /home/runner/bin/Runner.Listener
test -x /home/runner/externals/node20/bin/node
cp -r --no-preserve=ownership /home/runner/. /runner/
test -x /runner/bin/Runner.Listener
version=$(/runner/externals/node20/bin/node --version)
test "$version" = "v20.20.2"
node_sha=$(sha256sum /runner/externals/node20/bin/node | cut -d' ' -f1)
test "$node_sha" = "6295488653f0d93b0a157841746fef7e72cc4328cfb60c4bbe0ca2668a836ffd"
cd /runner
runner_output=$(/runner/bin/Runner.Listener --version)
test "$(printf '%s\n' "$runner_output" | grep -c '^2\.336\.0$')" = "1"
test "$(printf '%s\n' "$runner_output" | grep -Fc ' INFO Listener] Version: 2.336.0')" = "1"
test "$(printf '%s\n' "$runner_output" | grep -Fc ' INFO Listener] Commit: 98aabcd429c4e8402406c56ce2d26387fed3b9ce')" = "1"
unset runner_output
printf '%s\n' \
  'uid=1001' \
  'gid=1001' \
  'runner_listener=present' \
  'runner_version=2.336.0' \
  'runner_commit=98aabcd429c4e8402406c56ce2d26387fed3b9ce' \
  'node20=v20.20.2' \
  'node20_sha256=6295488653f0d93b0a157841746fef7e72cc4328cfb60c4bbe0ca2668a836ffd'
""".strip()

GPU_INJECTION_PROBE_SCRIPT = r"""
set -Eeuo pipefail
test "$(id -u)" = "1001"
test "$(id -g)" = "1001"
IFS=',' read -r -a expected_gpu_uuids <<< "${EXPLAINIVERSE_EXPECTED_GPU_UUIDS}"
mapfile -t observed_gpu_rows < <(nvidia-smi --query-gpu=uuid,name --format=csv,noheader)
test "${#observed_gpu_rows[@]}" -eq 8
for index in "${!observed_gpu_rows[@]}"; do
  test "${observed_gpu_rows[$index]}" = "${expected_gpu_uuids[$index]}, NVIDIA A100-SXM4-80GB"
done
test "${NVIDIA_VISIBLE_DEVICES}" = "${EXPLAINIVERSE_EXPECTED_GPU_UUIDS}"
printf '%s\n' 'gpu_injection=verified' 'gpu_count=8' 'gpu_product=NVIDIA A100-SXM4-80GB'
""".strip()


class ContractError(RuntimeError):
    """Stable fail-closed contract violation without sensitive values."""


def _fail(code: str) -> NoReturn:
    raise ContractError(code)


def _require(condition: bool, code: str) -> None:
    if not condition:
        _fail(code)


def _object(value: Any, keys: set[str], context: str) -> Mapping[str, Any]:
    _require(type(value) is dict, f"{context}_not_object")
    _require(set(value) == keys, f"{context}_keys_rejected")
    return value


def _list(value: Any, context: str) -> list[Any]:
    _require(type(value) is list, f"{context}_not_list")
    return value


def _text(
    value: Any,
    context: str,
    *,
    pattern: re.Pattern[str] | None = None,
    maximum: int = 512,
) -> str:
    _require(type(value) is str and 0 < len(value) <= maximum, f"{context}_text_rejected")
    _require("\x00" not in value and "\r" not in value and "\n" not in value, f"{context}_unsafe")
    if pattern is not None:
        _require(pattern.fullmatch(value) is not None, f"{context}_shape_rejected")
    return value


def _integer(value: Any, context: str, minimum: int, maximum: int) -> int:
    _require(type(value) is int and minimum <= value <= maximum, f"{context}_integer_rejected")
    return value


def _bool(value: Any, context: str) -> bool:
    _require(type(value) is bool, f"{context}_boolean_rejected")
    return value


def _timestamp(value: Any, context: str) -> datetime:
    text = _text(value, context, maximum=40)
    _require(text.endswith("Z"), f"{context}_utc_rejected")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError:
        _fail(f"{context}_timestamp_rejected")
    _require(parsed.tzinfo is not None, f"{context}_timezone_rejected")
    return parsed.astimezone(timezone.utc)


def _sha(value: Any, context: str) -> str:
    return _text(value, context, pattern=SHA256_RE, maximum=64)


def _gpu_list(value: Any, context: str, *, minimum: int, maximum: int) -> list[str]:
    items = _list(value, context)
    _require(minimum <= len(items) <= maximum, f"{context}_cardinality_rejected")
    result = [
        _text(item, f"{context}_{index}", pattern=GPU_UUID_RE, maximum=40)
        for index, item in enumerate(items)
    ]
    _require(len(result) == len(set(result)), f"{context}_duplicate_rejected")
    return result


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("ascii")


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("json_duplicate_key_rejected")
        result[key] = value
    return result


def parse_plan_document(raw: bytes, *, now: datetime | None = None) -> dict[str, Any]:
    _require(type(raw) is bytes and 0 < len(raw) <= 1_048_576, "plan_bytes_rejected")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail("plan_json_rejected")
    normalized = validate_runtime_plan(value, now=now)
    _require(raw == canonical_json(normalized), "plan_not_canonical")
    return normalized


def validate_runtime_plan(value: Any, *, now: datetime | None = None) -> dict[str, Any]:
    """Validate the complete, short-lived per-job execution authorization."""

    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    plan = _object(value, PLAN_KEYS, "runtime_plan")
    _require(plan["schema_version"] == SCHEMA_VERSION, "plan_schema_rejected")
    _require(plan["kind"] == PLAN_KIND, "plan_kind_rejected")
    _require(plan["execution_authorized"] is True, "execution_not_authorized")
    created = _timestamp(plan["created_at"], "plan_created_at")
    _require(created <= current + timedelta(seconds=5), "plan_created_in_future")
    _require(current - created <= timedelta(seconds=PLAN_MAX_AGE_SECONDS), "plan_stale")
    policy_sha = _sha(plan["policy_sha256"], "policy_sha256")
    _sha(plan["control_plane_plan_sha256"], "control_plane_plan_sha256")
    _sha(plan["runtime_bundle_sha256"], "runtime_bundle_sha256")
    phase = _text(plan["phase"], "phase", maximum=20)
    _require(phase in PHASE_SPECS, "phase_rejected")
    phase_spec = PHASE_SPECS[phase]
    _require(plan["repository"] == REPOSITORY, "repository_rejected")
    _require(plan["workflow_path"] == phase_spec["workflow_path"], "workflow_path_rejected")

    authority = _object(plan["authority_window"], AUTHORITY_KEYS, "authority")
    authority_at = _timestamp(authority["observed_at"], "authority_observed_at")
    authority_expires = _timestamp(authority["expires_at"], "authority_expires_at")
    _require(authority_at <= created, "authority_after_plan")
    _require(authority_expires > current, "authority_window_expired")
    _require(
        authority_expires - authority_at <= timedelta(seconds=AUTHORITY_MAX_SECONDS),
        "authority_window_too_long",
    )
    _sha(authority["evidence_sha256"], "authority_evidence_sha256")
    _require(authority["owner_login"] == OWNER, "authority_owner_rejected")
    collaborators = _list(authority["collaborators"], "authority_collaborators")
    _require(len(collaborators) == 1, "authority_not_sole_collaborator")
    collaborator = _object(collaborators[0], COLLABORATOR_KEYS, "authority_collaborator")
    _require(
        dict(collaborator) == {"login": OWNER, "permission": "admin"},
        "authority_collaborator_rejected",
    )
    _integer(authority["pending_invitation_count"], "pending_invitation_count", 0, 0)
    _integer(authority["unexpected_target_job_count"], "unexpected_target_job_count", 0, 0)
    _require(authority["enabled_nonowner_authorities"] == [], "authority_boundary_open")

    dispatch = _object(plan["dispatch"], DISPATCH_KEYS, "dispatch")
    dispatch_at = _timestamp(dispatch["observed_at"], "dispatch_observed_at")
    _require(dispatch_at < authority_at < created, "dispatch_observation_order_rejected")
    _sha(dispatch["request_sha256"], "dispatch_request_sha256")
    _sha(dispatch["response_sha256"], "dispatch_response_sha256")
    _integer(dispatch["run_id"], "dispatch_run_id", 1, 2**63 - 1)
    _text(dispatch["head_sha"], "dispatch_head_sha", pattern=COMMIT_RE, maximum=40)
    _require(
        dispatch["event"] == "workflow_dispatch"
        and dispatch["run_attempt"] == 1
        and dispatch["actor"] == OWNER
        and dispatch["triggering_actor"] == OWNER
        and dispatch["status"] in {"queued", "in_progress"}
        and dispatch["conclusion"] is None,
        "dispatch_rejected",
    )
    _require(dispatch["ref"] == phase_spec["ref"], "dispatch_ref_rejected")
    dispatch_inputs = dispatch["inputs"]
    expected_nonce_input_keys = cast(tuple[str, ...], phase_spec["nonce_input_keys"])
    if phase == "publication":
        inputs = _object(
            dispatch_inputs,
            PUBLICATION_DISPATCH_INPUT_KEYS,
            "publication_dispatch_inputs",
        )
        _require(
            inputs["tag"] == PUBLICATION_TAG and inputs["stage_recovery_drill"] is True,
            "publication_dispatch_inputs_rejected",
        )
        _integer(inputs["preflight_run_id"], "publication_preflight_run_id", 1, 2**63 - 1)
        _integer(inputs["cuda_run_id"], "publication_cuda_run_id", 1, 2**63 - 1)
    else:
        inputs = _object(dispatch_inputs, CUDA_DISPATCH_INPUT_KEYS, "cuda_dispatch_inputs")
    current_nonces = [
        _text(inputs[key], f"dispatch_{key}", pattern=NONCE_RE, maximum=16)
        for key in expected_nonce_input_keys
    ]
    _require(
        len(current_nonces) == len(set(current_nonces)),
        "dispatch_runner_nonces_not_distinct",
    )
    prior_raw = _list(
        dispatch["prior_accepted_cuda_runner_nonces"],
        "prior_accepted_cuda_runner_nonces",
    )
    expected_prior_count = cast(int, phase_spec["prior_accepted_nonce_count"])
    _require(
        len(prior_raw) == expected_prior_count,
        "prior_accepted_cuda_runner_nonce_count_rejected",
    )
    prior_nonces = [
        _text(item, f"prior_accepted_cuda_runner_nonce_{index}", pattern=NONCE_RE, maximum=16)
        for index, item in enumerate(prior_raw)
    ]
    _require(
        len(prior_nonces) == len(set(prior_nonces))
        and set(prior_nonces).isdisjoint(current_nonces),
        "cuda_runner_nonce_reuse_rejected",
    )

    job = _object(plan["job"], JOB_KEYS, "job")
    allowed_job_keys = cast(tuple[str, ...], phase_spec["job_keys"])
    ordinal = _integer(job["ordinal"], "job_ordinal", 1, len(allowed_job_keys))
    key = _text(job["key"], "job_key", maximum=32)
    _require(key == allowed_job_keys[ordinal - 1], "job_key_rejected")
    spec = JOB_SPECS[key]
    _integer(job["job_id"], "job_id", 1, 2**63 - 1)
    nonce = _text(job["runner_nonce"], "runner_nonce", pattern=NONCE_RE, maximum=16)
    runner_name = _text(job["runner_name"], "runner_name", pattern=RUNNER_NAME_RE)
    expected_runner_name = f"{spec['prefix']}{nonce}"
    expected_selected_input_key = expected_nonce_input_keys[ordinal - 1]
    _require(
        job["name"] == spec["name"]
        and nonce == inputs[expected_selected_input_key]
        and runner_name == expected_runner_name
        and job["labels"] == [runner_name]
        and job["status"] == "queued"
        and job["conclusion"] is None
        and job["work_folder"] == f"_work-{nonce}",
        "job_binding_rejected",
    )
    _integer(job["runner_id"], "runner_id", 1, 2**63 - 1)
    _sha(job["jit_config_sha256"], "jit_config_sha256")

    sequencing = _object(plan["sequencing"], SEQUENCING_KEYS, "sequencing")
    _require(sequencing["sequential_only"] is True, "sequential_execution_required")
    previous = sequencing["previous_cleanup_receipt_sha256"]
    if ordinal == 1:
        _require(previous is None, "first_job_previous_receipt_rejected")
    else:
        _sha(previous, "previous_cleanup_receipt_sha256")

    hardware = _object(plan["hardware"], HARDWARE_KEYS, "hardware")
    host_gpu_count = _integer(
        hardware["host_physical_gpu_count"],
        "host_physical_gpu_count",
        HOST_PHYSICAL_GPU_COUNT,
        HOST_PHYSICAL_GPU_COUNT,
    )
    host_gpus = _gpu_list(
        hardware["host_physical_gpu_uuids"],
        "host_physical_gpu_uuids",
        minimum=host_gpu_count,
        maximum=host_gpu_count,
    )
    products = _list(hardware["host_physical_gpu_products"], "host_physical_gpu_products")
    _require(
        products == [HOST_GPU_PRODUCT] * HOST_PHYSICAL_GPU_COUNT,
        "host_gpu_products_rejected",
    )
    required = _integer(hardware["required_cuda_devices"], "required_cuda_devices", 1, 2)
    _require(required == spec["gpu_count"], "job_gpu_count_rejected")
    assigned = _gpu_list(
        hardware["assigned_physical_gpu_uuids"],
        "assigned_physical_gpu_uuids",
        minimum=required,
        maximum=required,
    )
    unrequested = _gpu_list(
        hardware["unrequested_physical_gpu_uuids"],
        "unrequested_physical_gpu_uuids",
        minimum=0,
        maximum=15,
    )
    _require(set(assigned).issubset(set(host_gpus)), "assigned_gpu_not_on_host")
    _require(set(assigned).isdisjoint(unrequested), "gpu_scope_overlap")
    _require(
        unrequested == [uuid for uuid in host_gpus if uuid not in assigned],
        "unrequested_gpu_partition_rejected",
    )
    device_request = ",".join(assigned)
    _require(
        hardware["device_request"] == device_request
        and hardware["nvidia_visible_devices"] == device_request
        and hardware["cuda_visible_devices"] == spec["cuda_visible_devices"]
        and hardware["exclusive_device_scope_required"] is True,
        "gpu_scope_rejected",
    )

    source = _object(plan["runner_source"], RUNNER_SOURCE_KEYS, "runner_source")
    source_at = _timestamp(source["observed_at"], "runner_source_observed_at")
    _require(
        source_at <= current + timedelta(seconds=5)
        and current - source_at <= timedelta(seconds=OBSERVATION_MAX_AGE_SECONDS),
        "runner_source_observation_stale",
    )
    _sha(source["response_sha256"], "runner_source_response_sha256")
    expected_source = {
        "api_version": GITHUB_API_VERSION,
        "os": "linux",
        "architecture": "x64",
        "filename": RUNNER_FILENAME,
        "download_url": RUNNER_DOWNLOAD_URL,
        "sha256_checksum": RUNNER_ARCHIVE_SHA256,
        "version": RUNNER_VERSION,
    }
    _require(
        {key: source[key] for key in expected_source} == expected_source,
        "runner_source_rejected",
    )

    image = _object(plan["runner_image"], RUNNER_IMAGE_KEYS, "runner_image")
    _timestamp(image["manifest_observed_at"], "image_manifest_observed_at")
    probe_at = _timestamp(image["probe_observed_at"], "image_probe_observed_at")
    _require(
        probe_at <= current + timedelta(seconds=5)
        and current - probe_at <= timedelta(seconds=OBSERVATION_MAX_AGE_SECONDS),
        "image_probe_stale",
    )
    _sha(image["probe_receipt_sha256"], "image_probe_receipt_sha256")
    expected_image = {
        "tag_reference": IMAGE_TAG_REFERENCE,
        "image_reference": IMAGE_REFERENCE,
        "platform": IMAGE_PLATFORM,
        "manifest_digest": IMAGE_MANIFEST_DIGEST,
        "manifest_media_type": IMAGE_MANIFEST_MEDIA_TYPE,
        "manifest_size": IMAGE_MANIFEST_SIZE,
        "config_digest": IMAGE_CONFIG_DIGEST,
        "config_media_type": IMAGE_CONFIG_MEDIA_TYPE,
        "config_size": IMAGE_CONFIG_SIZE,
        "manifest_source": IMAGE_MANIFEST_SOURCE,
        "manifest_observed_at": IMAGE_MANIFEST_OBSERVED_AT,
        "container_uid": CONTAINER_UID,
        "container_gid": CONTAINER_GID,
        "runner_listener_present": True,
        "runner_listener_version": RUNNER_VERSION,
        "runner_commit": IMAGE_RUNNER_COMMIT,
        "node20_present": True,
        "node20_version": IMAGE_NODE20_VERSION,
        "node20_sha256": IMAGE_NODE20_SHA256,
    }
    _require(
        {key: image[key] for key in expected_image} == expected_image,
        "runner_image_rejected",
    )

    github = _object(plan["github_evidence"], GITHUB_EVIDENCE_KEYS, "github_evidence")
    absence = _object(github["pre_jit_registration_absence"], ABSENCE_KEYS, "pre_jit_absence")
    absence_at = _timestamp(absence["observed_at"], "pre_jit_absence_observed_at")
    _sha(absence["response_sha256"], "pre_jit_absence_response_sha256")
    _integer(absence["total_count"], "pre_jit_total_count", 0, 0)
    _require(absence["runners"] == [], "pre_jit_residue")
    history = _object(github["nonce_history"], NONCE_HISTORY_KEYS, "nonce_history")
    history_at = _timestamp(history["observed_at"], "nonce_history_observed_at")
    _sha(history["response_sha256"], "nonce_history_response_sha256")
    _integer(history["historical_match_count"], "historical_match_count", 0, 0)
    _integer(
        history["unexpected_queued_or_in_progress_count"],
        "unexpected_queued_or_in_progress_count",
        0,
        0,
    )
    jit = _object(github["jit_response"], JIT_RESPONSE_KEYS, "jit_response")
    jit_at = _timestamp(jit["observed_at"], "jit_response_observed_at")
    _sha(jit["response_sha256"], "jit_response_sha256")
    jit_runner = _object(jit["runner"], RUNNER_KEYS, "jit_runner")
    _require(
        jit_runner["id"] == job["runner_id"]
        and jit_runner["name"] == runner_name
        and jit_runner["os"] == "unknown"
        and jit_runner["status"] == "offline"
        and jit_runner["busy"] is False
        and jit_runner["labels"] == [runner_name],
        "jit_runner_rejected",
    )
    _require(
        dispatch_at < authority_at < history_at < absence_at < jit_at < created,
        "github_evidence_order_rejected",
    )

    limits = _object(plan["limits"], LIMIT_KEYS, "limits")
    _require(
        limits
        == {
            "hard_wall_seconds": HARD_WALL_SECONDS,
            "fd_read_seconds": FD_READ_SECONDS,
            "post_github_settle_seconds": POST_GITHUB_SETTLE_SECONDS,
            "external_watchdog_required": True,
            "cleanup_grace_seconds": CLEANUP_GRACE_SECONDS,
        },
        "limits_rejected",
    )
    _require(policy_sha == plan["policy_sha256"], "policy_binding_rejected")
    return json.loads(canonical_json(plan))


def runtime_plan_sha256(plan: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(plan)).hexdigest()


def runtime_names(plan: Mapping[str, Any]) -> dict[str, str]:
    nonce = str(plan["job"]["runner_nonce"])
    return {
        "container": f"explainiverse-jit-{nonce}",
        "network": f"exjit-{nonce}",
        "bridge": f"xjit{nonce[:10]}",
        "chain": f"EXJIT_{nonce[:12].upper()}",
    }


def render_image_pull_argv() -> tuple[str, ...]:
    return (
        DOCKER_PATH,
        "pull",
        "--platform",
        IMAGE_PLATFORM,
        IMAGE_REFERENCE,
    )


def render_image_probe_argv() -> tuple[str, ...]:
    return (
        DOCKER_PATH,
        "run",
        "--rm",
        "--name",
        PROBE_CONTAINER_NAME,
        "--label",
        GLOBAL_RUNTIME_LABEL,
        "--pull=never",
        "--network=none",
        "--platform",
        IMAGE_PLATFORM,
        "--read-only",
        "--user",
        f"{CONTAINER_UID}:{CONTAINER_GID}",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges:true",
        "--pids-limit=128",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,noexec,size=64m,uid=1001,gid=1001,mode=0700",
        "--tmpfs",
        "/runner:rw,nosuid,nodev,exec,size=2g,uid=1001,gid=1001,mode=0700",
        "--entrypoint",
        "/bin/bash",
        IMAGE_REFERENCE,
        "-ceu",
        IMAGE_PROBE_SCRIPT,
    )


def render_gpu_injection_probe_argv(gpu_uuids: Sequence[str]) -> tuple[str, ...]:
    uuids = _gpu_list(
        list(gpu_uuids),
        "probe_gpu_uuids",
        minimum=HOST_PHYSICAL_GPU_COUNT,
        maximum=HOST_PHYSICAL_GPU_COUNT,
    )
    device_request = ",".join(uuids)
    gpu_selector = _docker_gpu_selector(device_request)
    return (
        DOCKER_PATH,
        "run",
        "--rm",
        "--name",
        PROBE_CONTAINER_NAME,
        "--label",
        GLOBAL_RUNTIME_LABEL,
        "--pull=never",
        "--network=none",
        "--platform",
        IMAGE_PLATFORM,
        "--read-only",
        "--user",
        f"{CONTAINER_UID}:{CONTAINER_GID}",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges:true",
        "--pids-limit=128",
        "--gpus",
        gpu_selector,
        "--env",
        f"NVIDIA_VISIBLE_DEVICES={device_request}",
        "--env",
        f"EXPLAINIVERSE_EXPECTED_GPU_UUIDS={device_request}",
        "--entrypoint",
        "/bin/bash",
        IMAGE_REFERENCE,
        "-ceu",
        GPU_INJECTION_PROBE_SCRIPT,
    )


def _docker_gpu_selector(device_request: str) -> str:
    """Render Docker's CSV-valued GPU selector for direct subprocess argv.

    Docker's CLI requires the value to retain literal double quotes when the
    ``device`` value contains commas.  A shell example uses outer single
    quotes to preserve those double quotes; a direct subprocess call must pass
    the double quotes as part of the argv element itself.
    """

    values = device_request.split(",")
    _gpu_list(values, "docker_gpu_selector", minimum=1, maximum=HOST_PHYSICAL_GPU_COUNT)
    _require(",".join(values) == device_request, "docker_gpu_selector_rejected")
    selector = f"device={device_request}"
    return f'"{selector}"' if len(values) > 1 else selector


def expected_network_setup_argv(plan: Mapping[str, Any]) -> tuple[tuple[str, ...], ...]:
    names = runtime_names(plan)
    chain = names["chain"]
    bridge = names["bridge"]
    commands: list[tuple[str, ...]] = [
        (
            DOCKER_PATH,
            "network",
            "create",
            "--driver=bridge",
            f"--subnet={NETWORK_SUBNET}",
            f"--opt=com.docker.network.bridge.name={bridge}",
            "--opt=com.docker.network.bridge.enable_icc=false",
            "--opt=com.docker.network.bridge.enable_ip_masquerade=true",
            f"--label={GLOBAL_RUNTIME_LABEL}",
            f"--label=explainiverse.runner={plan['job']['runner_name']}",
            names["network"],
        ),
        (IPTABLES_PATH, "-N", chain),
    ]
    for destination in BLOCKED_IPV4_DESTINATIONS:
        commands.append((IPTABLES_PATH, "-A", chain, "-d", destination, "-j", "REJECT"))
    commands.extend(
        [
            (
                IPTABLES_PATH,
                "-A",
                chain,
                "-p",
                "tcp",
                "--dport",
                "443",
                "-m",
                "conntrack",
                "--ctstate",
                "NEW,ESTABLISHED",
                "-j",
                "ACCEPT",
            ),
            (IPTABLES_PATH, "-A", chain, "-p", "udp", "--dport", "53", "-j", "ACCEPT"),
            (IPTABLES_PATH, "-A", chain, "-p", "tcp", "--dport", "53", "-j", "ACCEPT"),
            (IPTABLES_PATH, "-A", chain, "-j", "REJECT"),
            (IPTABLES_PATH, "-I", "DOCKER-USER", "1", "-i", bridge, "-j", chain),
        ]
    )
    return tuple(commands)


def expected_network_cleanup_argv(plan: Mapping[str, Any]) -> tuple[tuple[str, ...], ...]:
    names = runtime_names(plan)
    return (
        (
            IPTABLES_PATH,
            "-D",
            "DOCKER-USER",
            "-i",
            names["bridge"],
            "-j",
            names["chain"],
        ),
        (IPTABLES_PATH, "-F", names["chain"]),
        (IPTABLES_PATH, "-X", names["chain"]),
        (DOCKER_PATH, "network", "rm", names["network"]),
    )


def render_docker_run_argv(plan: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the fixed runner argv.  No secret value is accepted by this function."""

    names = runtime_names(plan)
    job = plan["job"]
    hardware = plan["hardware"]
    work_folder = _text(job["work_folder"], "work_folder", pattern=WORK_FOLDER_RE)
    device_request = _text(hardware["device_request"], "device_request", maximum=81)
    gpu_selector = _docker_gpu_selector(device_request)
    argv = (
        DOCKER_PATH,
        "run",
        "--rm",
        "--interactive",
        "--pull=never",
        "--name",
        names["container"],
        "--hostname",
        names["container"],
        "--network",
        names["network"],
        "--platform",
        IMAGE_PLATFORM,
        "--read-only",
        "--user",
        f"{CONTAINER_UID}:{CONTAINER_GID}",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges:true",
        "--pids-limit=2048",
        "--ipc=private",
        "--uts=private",
        "--init",
        "--stop-timeout=10",
        "--shm-size=2g",
        "--ulimit=core=0:0",
        "--ulimit=nofile=4096:4096",
        "--sysctl=net.ipv6.conf.all.disable_ipv6=1",
        "--sysctl=net.ipv4.conf.all.route_localnet=0",
        "--log-driver=none",
        "--tmpfs",
        "/runner:rw,nosuid,nodev,exec,size=2g,uid=1001,gid=1001,mode=0700",
        "--tmpfs",
        "/runner-home:rw,nosuid,nodev,noexec,size=8g,uid=1001,gid=1001,mode=0700",
        "--tmpfs",
        "/runner-tmp:rw,nosuid,nodev,exec,size=8g,uid=1001,gid=1001,mode=0700",
        "--tmpfs",
        "/runner-toolcache:rw,nosuid,nodev,exec,size=16g,uid=1001,gid=1001,mode=0700",
        "--tmpfs",
        f"/runner/{work_folder}:rw,nosuid,nodev,exec,size=16g,uid=1001,gid=1001,mode=0700",
        "--tmpfs",
        "/runner/_diag:rw,nosuid,nodev,noexec,size=256m,uid=1001,gid=1001,mode=0700",
        "--env",
        "HOME=/runner-home",
        "--env",
        "TMPDIR=/runner-tmp",
        "--env",
        "RUNNER_TOOL_CACHE=/runner-toolcache",
        "--env",
        "AGENT_TOOLSDIRECTORY=/runner-toolcache",
        "--env",
        f"NVIDIA_VISIBLE_DEVICES={hardware['nvidia_visible_devices']}",
        "--env",
        f"CUDA_VISIBLE_DEVICES={hardware['cuda_visible_devices']}",
        "--env",
        f"EXPLAINIVERSE_ASSIGNED_GPU_UUIDS={device_request}",
        "--env",
        ("EXPLAINIVERSE_LOGICAL_CUDA_VISIBLE_DEVICES=" f"{hardware['cuda_visible_devices']}"),
        "--gpus",
        gpu_selector,
        "--label",
        GLOBAL_RUNTIME_LABEL,
        "--label",
        f"explainiverse.run_id={plan['dispatch']['run_id']}",
        "--label",
        f"explainiverse.job_id={job['job_id']}",
        "--label",
        f"explainiverse.runner_name={job['runner_name']}",
        "--entrypoint",
        "/bin/bash",
        IMAGE_REFERENCE,
        "-ceu",
        CONTAINER_LAUNCHER,
    )
    forbidden = ("--volume", "-v", "--mount", "/var/run/docker.sock", "--privileged")
    _require(not any(item in forbidden for item in argv), "docker_argv_mount_or_privilege")
    _require(
        not any("--jitconfig" in item.lower() for item in argv)
        and not any(item.startswith("ACTIONS_RUNNER_INPUT_JITCONFIG=") for item in argv),
        "jit_secret_channel_rejected",
    )
    return argv


def validate_host_gpu_inventory(plan: Mapping[str, Any], output: bytes) -> list[str]:
    observed, products = parse_host_gpu_inventory(output)
    _require(
        observed == plan["hardware"]["host_physical_gpu_uuids"],
        "host_gpu_inventory_mismatch",
    )
    _require(
        products == plan["hardware"]["host_physical_gpu_products"],
        "host_gpu_product_inventory_mismatch",
    )
    return observed


def parse_host_gpu_inventory(output: bytes) -> tuple[list[str], list[str]]:
    """Parse the complete ordered UUID/product inventory from fixed nvidia-smi output."""

    _require(type(output) is bytes and len(output) <= 4096, "gpu_inventory_bytes_rejected")
    try:
        lines = [line for line in output.decode("ascii").splitlines() if line]
    except UnicodeDecodeError:
        _fail("gpu_inventory_encoding_rejected")
    _require(len(lines) == HOST_PHYSICAL_GPU_COUNT, "gpu_inventory_cardinality_rejected")
    uuids: list[str] = []
    products: list[str] = []
    for index, line in enumerate(lines):
        fields = line.split(", ")
        _require(len(fields) == 2, "gpu_inventory_line_rejected")
        uuids.append(
            _text(fields[0], f"observed_gpu_uuid_{index}", pattern=GPU_UUID_RE, maximum=40)
        )
        products.append(_text(fields[1], f"observed_gpu_product_{index}", maximum=64))
    _require(len(set(uuids)) == len(uuids), "gpu_inventory_duplicate_rejected")
    _require(products == [HOST_GPU_PRODUCT] * HOST_PHYSICAL_GPU_COUNT, "gpu_product_rejected")
    return uuids, products


def validate_image_inspect(plan: Mapping[str, Any], value: Any) -> dict[str, Any]:
    items = _list(value, "image_inspect")
    _require(len(items) == 1 and type(items[0]) is dict, "image_inspect_cardinality")
    item = items[0]
    _require(item.get("Id") == IMAGE_CONFIG_DIGEST, "image_config_digest_mismatch")
    _require(item.get("Architecture") == "amd64" and item.get("Os") == "linux", "image_platform")
    config = item.get("Config")
    _require(type(config) is dict, "image_config_missing")
    _require(config.get("User") in {"runner", "1001", "1001:1001"}, "image_user_rejected")
    repo_digests = item.get("RepoDigests")
    _require(type(repo_digests) is list, "image_repo_digests_missing")
    _require(
        f"ghcr.io/actions/actions-runner@{IMAGE_MANIFEST_DIGEST}" in repo_digests,
        "image_manifest_digest_mismatch",
    )
    return json.loads(canonical_json(item))


def validate_image_probe_output(output: bytes) -> dict[str, Any]:
    _require(type(output) is bytes and 0 < len(output) <= 512, "image_probe_bytes_rejected")
    try:
        lines = output.decode("ascii").splitlines()
    except UnicodeDecodeError:
        _fail("image_probe_encoding_rejected")
    _require(len(lines) == 7, "image_probe_line_count_rejected")
    _require(
        lines[0] == "uid=1001"
        and lines[1] == "gid=1001"
        and lines[2] == "runner_listener=present"
        and lines[3] == f"runner_version={RUNNER_VERSION}"
        and lines[4] == f"runner_commit={IMAGE_RUNNER_COMMIT}"
        and lines[5] == f"node20={IMAGE_NODE20_VERSION}"
        and lines[6] == f"node20_sha256={IMAGE_NODE20_SHA256}",
        "image_probe_output_rejected",
    )
    return {
        "container_uid": 1001,
        "container_gid": 1001,
        "runner_listener_present": True,
        "runner_listener_version": RUNNER_VERSION,
        "runner_commit": IMAGE_RUNNER_COMMIT,
        "node20_present": True,
        "node20_version": IMAGE_NODE20_VERSION,
        "node20_sha256": IMAGE_NODE20_SHA256,
        "output_sha256": hashlib.sha256(output).hexdigest(),
    }


def validate_gpu_injection_probe_output(output: bytes) -> dict[str, Any]:
    expected = b"gpu_injection=verified\n" b"gpu_count=8\n" b"gpu_product=NVIDIA A100-SXM4-80GB\n"
    _require(output == expected, "gpu_injection_probe_output_rejected")
    return {
        "gpu_injection_verified": True,
        "gpu_count": HOST_PHYSICAL_GPU_COUNT,
        "gpu_product": HOST_GPU_PRODUCT,
        "output_sha256": hashlib.sha256(output).hexdigest(),
    }


def _normalize_runner(value: Any, context: str) -> dict[str, Any]:
    runner = _object(value, RUNNER_KEYS, context)
    _integer(runner["id"], f"{context}_id", 1, 2**63 - 1)
    _text(runner["name"], f"{context}_name", pattern=RUNNER_NAME_RE)
    _require(runner["os"] in {"unknown", "linux"}, f"{context}_os_rejected")
    _require(runner["status"] in {"offline", "online"}, f"{context}_status_rejected")
    _bool(runner["busy"], f"{context}_busy")
    labels = _list(runner["labels"], f"{context}_labels")
    _require(all(type(label) is str for label in labels), f"{context}_label_type")
    return json.loads(canonical_json(runner))


def _validate_live_common(
    plan: Mapping[str, Any], value: Any, *, now: datetime, context: str
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    observation = _object(value, LIVE_OBSERVATION_KEYS, context)
    captured = _timestamp(observation["captured_at"], f"{context}_captured_at")
    _require(captured <= now + timedelta(seconds=5), f"{context}_future")
    _require(now - captured <= timedelta(seconds=120), f"{context}_stale")
    for field in (
        "run_response_sha256",
        "jobs_response_sha256",
        "downloads_response_sha256",
        "runners_response_sha256",
    ):
        _sha(observation[field], f"{context}_{field}")
    run = _object(observation["run"], LIVE_RUN_KEYS, f"{context}_run")
    expected_run = {
        "id": plan["dispatch"]["run_id"],
        "event": "workflow_dispatch",
        "path": plan["workflow_path"],
        "ref": plan["dispatch"]["ref"],
        "head_sha": plan["dispatch"]["head_sha"],
        "run_attempt": 1,
        "actor": OWNER,
        "triggering_actor": OWNER,
    }
    _require(
        all(run[key] == expected for key, expected in expected_run.items()),
        f"{context}_run_binding_rejected",
    )
    jobs_raw = _list(observation["jobs"], f"{context}_jobs")
    jobs = []
    for index, raw_job in enumerate(jobs_raw):
        job = _object(raw_job, LIVE_JOB_KEYS, f"{context}_job_{index}")
        _integer(job["id"], f"{context}_job_{index}_id", 1, 2**63 - 1)
        jobs.append(json.loads(canonical_json(job)))
    downloads_raw = _list(observation["downloads"], f"{context}_downloads")
    downloads = []
    for index, raw_download in enumerate(downloads_raw):
        item = _object(raw_download, DOWNLOAD_KEYS, f"{context}_download_{index}")
        downloads.append(json.loads(canonical_json(item)))
    matching_downloads = [
        item
        for item in downloads
        if item.get("os") == "linux" and item.get("architecture") == "x64"
    ]
    _require(len(matching_downloads) == 1, f"{context}_linux_x64_download_cardinality")
    expected_download = {
        "os": "linux",
        "architecture": "x64",
        "filename": RUNNER_FILENAME,
        "download_url": RUNNER_DOWNLOAD_URL,
        "sha256_checksum": RUNNER_ARCHIVE_SHA256,
    }
    _require(matching_downloads[0] == expected_download, f"{context}_runner_download_changed")
    runners = [
        _normalize_runner(item, f"{context}_runner_{index}")
        for index, item in enumerate(_list(observation["runners"], f"{context}_runners"))
    ]
    normalized = json.loads(canonical_json(observation))
    return normalized, dict(run), jobs, runners


def validate_pre_execution_observation(
    plan: Mapping[str, Any], value: Any, *, now: datetime | None = None
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    normalized, run, jobs, runners = _validate_live_common(
        plan, value, now=current, context="pre_execution"
    )
    _require(
        run["status"] in {"queued", "in_progress"} and run["conclusion"] is None,
        "pre_execution_run_not_active",
    )
    selected = [job for job in jobs if job["id"] == plan["job"]["job_id"]]
    _require(len(selected) == 1, "pre_execution_job_cardinality")
    expected_job = {
        "id": plan["job"]["job_id"],
        "name": plan["job"]["name"],
        "head_sha": plan["dispatch"]["head_sha"],
        "run_attempt": 1,
        "status": "queued",
        "conclusion": None,
        "labels": [plan["job"]["runner_name"]],
        "runner_id": None,
        "runner_name": None,
    }
    _require(selected[0] == expected_job, "pre_execution_job_binding_rejected")
    _require(len(runners) == 1, "pre_execution_runner_inventory_cardinality")
    expected_runner = plan["github_evidence"]["jit_response"]["runner"]
    _require(runners[0] == expected_runner, "pre_execution_runner_binding_rejected")
    return normalized


def validate_post_execution_observation(
    plan: Mapping[str, Any], value: Any, *, now: datetime | None = None
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    normalized, run, jobs, runners = _validate_live_common(
        plan, value, now=current, context="post_execution"
    )
    _require(
        run["status"] in {"in_progress", "completed"} and run["conclusion"] in {None, "success"},
        "post_execution_run_rejected",
    )
    selected = [job for job in jobs if job["id"] == plan["job"]["job_id"]]
    _require(len(selected) == 1, "post_execution_job_cardinality")
    expected_job = {
        "id": plan["job"]["job_id"],
        "name": plan["job"]["name"],
        "head_sha": plan["dispatch"]["head_sha"],
        "run_attempt": 1,
        "status": "completed",
        "conclusion": "success",
        "labels": [plan["job"]["runner_name"]],
        "runner_id": plan["job"]["runner_id"],
        "runner_name": plan["job"]["runner_name"],
    }
    _require(selected[0] == expected_job, "post_execution_job_binding_rejected")
    _require(runners == [], "post_execution_runner_registration_residue")
    return normalized


def build_runtime_receipt(
    plan: Mapping[str, Any],
    *,
    host_gpu_uuids: Sequence[str],
    started_at: str,
    jit_config_sent_at: str,
    stopped_at: str,
    cleanup_verified_at: str,
    runner_exit_code: int,
) -> dict[str, Any]:
    _require(runner_exit_code == 0, "runner_exit_nonzero")
    _require(list(host_gpu_uuids) == plan["hardware"]["host_physical_gpu_uuids"], "gpu_receipt")
    start = _timestamp(started_at, "receipt_started_at")
    sent = _timestamp(jit_config_sent_at, "receipt_jit_config_sent_at")
    stop = _timestamp(stopped_at, "receipt_stopped_at")
    cleanup = _timestamp(cleanup_verified_at, "receipt_cleanup_at")
    authority_expires = _timestamp(
        plan["authority_window"]["expires_at"], "receipt_authority_expires_at"
    )
    cleanup_deadline = authority_expires + timedelta(seconds=CLEANUP_GRACE_SECONDS)
    _require(start <= sent < stop <= cleanup, "receipt_time_order_rejected")
    _require(stop - start <= timedelta(seconds=HARD_WALL_SECONDS), "receipt_wall_exceeded")
    _require(stop <= authority_expires, "receipt_stopped_after_authority_expiry")
    _require(cleanup <= cleanup_deadline, "receipt_cleanup_after_grace")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": RECEIPT_KIND,
        "status": "runner-container-stopped-and-host-cleaned",
        "policy_sha256": plan["policy_sha256"],
        "control_plane_plan_sha256": plan["control_plane_plan_sha256"],
        "runtime_plan_sha256": runtime_plan_sha256(plan),
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
        "host_physical_gpu_uuids": list(host_gpu_uuids),
        "host_physical_gpu_products": plan["hardware"]["host_physical_gpu_products"],
        "assigned_physical_gpu_uuids": plan["hardware"]["assigned_physical_gpu_uuids"],
        "unrequested_physical_gpu_uuids": plan["hardware"]["unrequested_physical_gpu_uuids"],
        "nvidia_visible_devices": plan["hardware"]["nvidia_visible_devices"],
        "cuda_visible_devices": plan["hardware"]["cuda_visible_devices"],
        "runner_version": RUNNER_VERSION,
        "runner_archive_sha256": RUNNER_ARCHIVE_SHA256,
        "runner_image_reference": IMAGE_REFERENCE,
        "runner_image_manifest_digest": IMAGE_MANIFEST_DIGEST,
        "jit_config_sha256": plan["job"]["jit_config_sha256"],
        "jit_config_persisted": False,
        "jit_config_destroyed": True,
        "jit_config_sent_at": jit_config_sent_at,
        "one_job_jit_configuration_supplied": True,
        "claimed_job_count_verified_by_runtime": False,
        "runner_exit_code": 0,
        "started_at": started_at,
        "stopped_at": stopped_at,
        "cleanup_verified_at": cleanup_verified_at,
        "authority_expires_at": plan["authority_window"]["expires_at"],
        "workload_stopped_before_authority_expiry": True,
        "cleanup_grace_seconds": CLEANUP_GRACE_SECONDS,
        "cleanup_deadline_at": cleanup_deadline.isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        ),
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
    return json.loads(canonical_json(receipt))
