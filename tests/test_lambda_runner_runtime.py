from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "scripts" / "release_gpu_jit_lambda_runtime"


def _load(name: str, path: Path, *, package_locations=None):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
        submodule_search_locations=package_locations,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_load(
    "release_gpu_jit_lambda_runtime",
    PACKAGE_ROOT / "__init__.py",
    package_locations=[str(PACKAGE_ROOT)],
)
contract = _load(
    "release_gpu_jit_lambda_runtime.runtime_contract",
    PACKAGE_ROOT / "runtime_contract.py",
)
executor = _load(
    "release_gpu_jit_lambda_runtime.executor",
    PACKAGE_ROOT / "executor.py",
)
bootstrap = _load(
    "release_gpu_jit_lambda_runtime.bootstrap",
    PACKAGE_ROOT / "bootstrap.py",
)

NOW = datetime(2026, 8, 28, 21, 8, 0, tzinfo=timezone.utc)
HEAD_SHA = "a" * 40
POLICY_SHA = "b" * 64
CONTROL_SHA = "c" * 64
JIT_CONFIG = b"A" * 256
JIT_SHA = hashlib.sha256(JIT_CONFIG).hexdigest()
GPU_ONE = "GPU-11111111-1111-1111-1111-111111111111"
GPU_TWO = "GPU-22222222-2222-2222-2222-222222222222"
GPU_THREE = "GPU-33333333-3333-3333-3333-333333333333"
GPU_FOUR = "GPU-44444444-4444-4444-4444-444444444444"
GPU_FIVE = "GPU-55555555-5555-5555-5555-555555555555"
GPU_SIX = "GPU-66666666-6666-6666-6666-666666666666"
GPU_SEVEN = "GPU-77777777-7777-7777-7777-777777777777"
GPU_EIGHT = "GPU-88888888-8888-8888-8888-888888888888"
HOST_GPUS = [
    GPU_ONE,
    GPU_TWO,
    GPU_THREE,
    GPU_FOUR,
    GPU_FIVE,
    GPU_SIX,
    GPU_SEVEN,
    GPU_EIGHT,
]
HOST_GPU_PRODUCTS = [contract.HOST_GPU_PRODUCT] * 8
NONCE = "0123456789abcdef"
RUNNER_NAME = f"explainiverse-cuda-single-jit-{NONCE}"


def valid_plan() -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": contract.PLAN_KIND,
        "execution_authorized": True,
        "created_at": "2026-08-28T21:08:00Z",
        "policy_sha256": POLICY_SHA,
        "control_plane_plan_sha256": CONTROL_SHA,
        "runtime_bundle_sha256": "9" * 64,
        "phase": "final-main",
        "repository": contract.REPOSITORY,
        "workflow_path": contract.WORKFLOW_PATH,
        "authority_window": {
            "observed_at": "2026-08-28T21:02:30Z",
            "expires_at": "2026-08-28T21:20:00Z",
            "evidence_sha256": "d" * 64,
            "owner_login": "jemsbhai",
            "collaborators": [{"login": "jemsbhai", "permission": "admin"}],
            "pending_invitation_count": 0,
            "enabled_nonowner_authorities": [],
            "unexpected_target_job_count": 0,
        },
        "dispatch": {
            "observed_at": "2026-08-28T21:02:00Z",
            "request_sha256": "0" * 64,
            "response_sha256": "e" * 64,
            "event": "workflow_dispatch",
            "ref": contract.FINAL_MAIN_REF,
            "inputs": {
                "single_minimum_runner_nonce": NONCE,
                "single_latest_runner_nonce": "1111111111111111",
                "two_minimum_runner_nonce": "2222222222222222",
                "two_latest_runner_nonce": "3333333333333333",
            },
            "prior_accepted_cuda_runner_nonces": [],
            "run_id": 123456789,
            "run_attempt": 1,
            "head_sha": HEAD_SHA,
            "actor": "jemsbhai",
            "triggering_actor": "jemsbhai",
            "status": "in_progress",
            "conclusion": None,
        },
        "job": {
            "ordinal": 1,
            "key": "single_minimum",
            "job_id": 987654321,
            "name": "CUDA single-GPU (Torch minimum)",
            "runner_nonce": NONCE,
            "runner_id": 42,
            "runner_name": RUNNER_NAME,
            "labels": [RUNNER_NAME],
            "status": "queued",
            "conclusion": None,
            "work_folder": f"_work-{NONCE}",
            "jit_config_sha256": JIT_SHA,
        },
        "sequencing": {
            "sequential_only": True,
            "previous_cleanup_receipt_sha256": None,
        },
        "hardware": {
            "host_physical_gpu_count": 8,
            "host_physical_gpu_uuids": HOST_GPUS,
            "host_physical_gpu_products": HOST_GPU_PRODUCTS,
            "assigned_physical_gpu_uuids": [GPU_ONE],
            "unrequested_physical_gpu_uuids": HOST_GPUS[1:],
            "device_request": GPU_ONE,
            "nvidia_visible_devices": GPU_ONE,
            "cuda_visible_devices": "0",
            "required_cuda_devices": 1,
            "exclusive_device_scope_required": True,
        },
        "runner_source": {
            "observed_at": "2026-08-28T21:06:00Z",
            "response_sha256": "f" * 64,
            "api_version": contract.GITHUB_API_VERSION,
            "os": "linux",
            "architecture": "x64",
            "filename": contract.RUNNER_FILENAME,
            "download_url": contract.RUNNER_DOWNLOAD_URL,
            "sha256_checksum": contract.RUNNER_ARCHIVE_SHA256,
            "version": contract.RUNNER_VERSION,
        },
        "runner_image": {
            "tag_reference": contract.IMAGE_TAG_REFERENCE,
            "image_reference": contract.IMAGE_REFERENCE,
            "platform": contract.IMAGE_PLATFORM,
            "manifest_digest": contract.IMAGE_MANIFEST_DIGEST,
            "manifest_media_type": contract.IMAGE_MANIFEST_MEDIA_TYPE,
            "manifest_size": contract.IMAGE_MANIFEST_SIZE,
            "config_digest": contract.IMAGE_CONFIG_DIGEST,
            "config_media_type": contract.IMAGE_CONFIG_MEDIA_TYPE,
            "config_size": contract.IMAGE_CONFIG_SIZE,
            "manifest_source": contract.IMAGE_MANIFEST_SOURCE,
            "manifest_observed_at": contract.IMAGE_MANIFEST_OBSERVED_AT,
            "probe_observed_at": "2026-08-28T21:07:00Z",
            "probe_receipt_sha256": "1" * 64,
            "container_uid": 1001,
            "container_gid": 1001,
            "runner_listener_present": True,
            "runner_listener_version": contract.RUNNER_VERSION,
            "runner_commit": contract.IMAGE_RUNNER_COMMIT,
            "node20_present": True,
            "node20_version": contract.IMAGE_NODE20_VERSION,
            "node20_sha256": contract.IMAGE_NODE20_SHA256,
        },
        "github_evidence": {
            "pre_jit_registration_absence": {
                "observed_at": "2026-08-28T21:04:00Z",
                "response_sha256": "2" * 64,
                "total_count": 0,
                "runners": [],
            },
            "nonce_history": {
                "observed_at": "2026-08-28T21:03:00Z",
                "response_sha256": "3" * 64,
                "historical_match_count": 0,
                "unexpected_queued_or_in_progress_count": 0,
            },
            "jit_response": {
                "observed_at": "2026-08-28T21:05:00Z",
                "response_sha256": "4" * 64,
                "runner": {
                    "id": 42,
                    "name": RUNNER_NAME,
                    "os": "unknown",
                    "status": "offline",
                    "busy": False,
                    "labels": [RUNNER_NAME],
                },
            },
        },
        "limits": {
            "hard_wall_seconds": 3000,
            "fd_read_seconds": 15,
            "post_github_settle_seconds": 120,
            "external_watchdog_required": True,
            "cleanup_grace_seconds": 60,
        },
    }


def valid_publication_plan() -> dict[str, object]:
    plan = valid_plan()
    plan["phase"] = "publication"
    plan["workflow_path"] = contract.PUBLISH_WORKFLOW_PATH
    plan["dispatch"].update(
        {
            "ref": contract.PUBLICATION_REF,
            "inputs": {
                "tag": contract.PUBLICATION_TAG,
                "preflight_run_id": 223456789,
                "cuda_run_id": 323456789,
                "single_minimum_runner_nonce": NONCE,
                "single_latest_runner_nonce": "1111111111111111",
                "stage_recovery_drill": True,
            },
            "prior_accepted_cuda_runner_nonces": [
                "aaaaaaaaaaaaaaaa",
                "bbbbbbbbbbbbbbbb",
                "cccccccccccccccc",
                "dddddddddddddddd",
            ],
        }
    )
    plan["job"].update(
        {
            "key": "publication_single_minimum",
            "name": "Release CUDA single-GPU (Torch minimum, zero skips)",
        }
    )
    return plan


def valid_image_inspect() -> list[dict[str, object]]:
    return [
        {
            "Id": contract.IMAGE_CONFIG_DIGEST,
            "Architecture": "amd64",
            "Os": "linux",
            "Config": {"User": "runner"},
            "RepoDigests": [f"ghcr.io/actions/actions-runner@{contract.IMAGE_MANIFEST_DIGEST}"],
        }
    ]


def live_observation(*, completed: bool) -> dict[str, object]:
    plan = valid_plan()
    job = plan["job"]
    assert isinstance(job, dict)
    runner = plan["github_evidence"]["jit_response"]["runner"]
    return {
        "captured_at": "2026-08-28T21:08:00Z",
        "run_response_sha256": "5" * 64,
        "jobs_response_sha256": "6" * 64,
        "downloads_response_sha256": "7" * 64,
        "runners_response_sha256": "8" * 64,
        "run": {
            "id": 123456789,
            "event": "workflow_dispatch",
            "path": contract.WORKFLOW_PATH,
            "ref": contract.FINAL_MAIN_REF,
            "head_sha": HEAD_SHA,
            "run_attempt": 1,
            "actor": "jemsbhai",
            "triggering_actor": "jemsbhai",
            "status": "in_progress",
            "conclusion": None,
        },
        "jobs": [
            {
                "id": 987654321,
                "name": job["name"],
                "head_sha": HEAD_SHA,
                "run_attempt": 1,
                "status": "completed" if completed else "queued",
                "conclusion": "success" if completed else None,
                "labels": [RUNNER_NAME],
                "runner_id": 42 if completed else None,
                "runner_name": RUNNER_NAME if completed else None,
            }
        ],
        "downloads": [
            {
                "os": "linux",
                "architecture": "x64",
                "filename": contract.RUNNER_FILENAME,
                "download_url": contract.RUNNER_DOWNLOAD_URL,
                "sha256_checksum": contract.RUNNER_ARCHIVE_SHA256,
            }
        ],
        "runners": [] if completed else [runner],
    }


def test_exact_runtime_plan_is_canonical_and_short_lived() -> None:
    plan = contract.validate_runtime_plan(valid_plan(), now=NOW)
    raw = contract.canonical_json(plan)
    assert contract.parse_plan_document(raw, now=NOW) == plan
    assert contract.runtime_plan_sha256(plan) == hashlib.sha256(raw).hexdigest()
    assert plan["execution_authorized"] is True


@pytest.mark.parametrize("permission", ["read", "write"])
def test_every_second_collaborator_is_rejected(permission: str) -> None:
    plan = valid_plan()
    plan["authority_window"]["collaborators"] = [
        {"login": "b-urge", "permission": permission},
        {"login": contract.OWNER, "permission": "admin"},
    ]
    with pytest.raises(contract.ContractError, match="authority_not_sole_collaborator"):
        contract.validate_runtime_plan(plan, now=NOW)


def test_plan_document_rejects_noncanonical_and_duplicate_json() -> None:
    plan = valid_plan()
    with pytest.raises(contract.ContractError, match="plan_not_canonical"):
        contract.parse_plan_document(json.dumps(plan, indent=2).encode(), now=NOW)
    raw = contract.canonical_json(plan)
    duplicate = raw.replace(b'"kind":', b'"kind":"wrong","kind":', 1)
    with pytest.raises(contract.ContractError, match="duplicate"):
        contract.parse_plan_document(duplicate, now=NOW)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("execution_authorized", False, "execution_not_authorized"),
        ("repository", "attacker/repo", "repository_rejected"),
        ("workflow_path", ".github/workflows/evil.yml", "workflow_path_rejected"),
        ("phase", "arbitrary", "phase_rejected"),
    ],
)
def test_plan_rejects_broad_or_wrong_execution_authority(
    field: str, value: object, error: str
) -> None:
    plan = valid_plan()
    plan[field] = value
    with pytest.raises(contract.ContractError, match=error):
        contract.validate_runtime_plan(plan, now=NOW)


def test_publication_phase_is_narrowly_bound_to_v0150_and_two_single_gpu_jobs() -> None:
    plan = valid_publication_plan()
    normalized = contract.validate_runtime_plan(plan, now=NOW)
    assert normalized["phase"] == "publication"
    assert normalized["workflow_path"] == ".github/workflows/publish-pypi.yml"
    assert normalized["dispatch"]["ref"] == "refs/tags/v0.15.0"
    assert normalized["dispatch"]["inputs"]["stage_recovery_drill"] is True

    for mutation, error in (
        (lambda value: value["dispatch"].update({"ref": "refs/heads/main"}), "dispatch_ref"),
        (
            lambda value: value["dispatch"]["inputs"].update({"tag": "v0.15.1"}),
            "publication_dispatch_inputs",
        ),
        (
            lambda value: value["dispatch"]["inputs"].update({"stage_recovery_drill": False}),
            "publication_dispatch_inputs",
        ),
        (lambda value: value["job"].update({"key": "two_minimum"}), "job_key"),
        (
            lambda value: value["dispatch"].update(
                {
                    "prior_accepted_cuda_runner_nonces": [
                        NONCE,
                        "bbbbbbbbbbbbbbbb",
                        "cccccccccccccccc",
                        "dddddddddddddddd",
                    ]
                }
            ),
            "cuda_runner_nonce_reuse",
        ),
    ):
        tampered = valid_publication_plan()
        mutation(tampered)
        with pytest.raises(contract.ContractError, match=error):
            contract.validate_runtime_plan(tampered, now=NOW)


def test_publication_latest_job_is_ordinal_two_and_one_gpu_only() -> None:
    plan = valid_publication_plan()
    latest_name = f"explainiverse-cuda-single-jit-{'1' * 16}"
    plan["job"].update(
        {
            "ordinal": 2,
            "key": "publication_single_latest",
            "name": "Release CUDA single-GPU (Torch latest, zero skips)",
            "runner_nonce": "1" * 16,
            "runner_name": latest_name,
            "labels": [latest_name],
            "work_folder": f"_work-{'1' * 16}",
        }
    )
    plan["sequencing"]["previous_cleanup_receipt_sha256"] = "5" * 64
    plan["github_evidence"]["jit_response"]["runner"].update(
        {"name": latest_name, "labels": [latest_name]}
    )
    assert contract.validate_runtime_plan(plan, now=NOW)["hardware"]["required_cuda_devices"] == 1


def test_plan_rejects_stale_or_overlong_authority_window() -> None:
    stale = valid_plan()
    stale["created_at"] = "2026-08-28T21:00:00Z"
    with pytest.raises(contract.ContractError, match="plan_stale"):
        contract.validate_runtime_plan(stale, now=NOW)
    broad = valid_plan()
    broad["authority_window"]["expires_at"] = "2026-08-28T21:40:01Z"
    with pytest.raises(contract.ContractError, match="authority_window_too_long"):
        contract.validate_runtime_plan(broad, now=NOW)


def test_fresh_authority_capture_must_follow_queued_dispatch_and_precede_jit() -> None:
    plan = valid_plan()
    plan["authority_window"]["observed_at"] = "2026-08-28T21:01:59Z"
    with pytest.raises(contract.ContractError, match="dispatch_observation_order_rejected"):
        contract.validate_runtime_plan(plan, now=NOW)
    plan = valid_plan()
    plan["authority_window"]["observed_at"] = "2026-08-28T21:05:01Z"
    with pytest.raises(contract.ContractError, match="github_evidence_order_rejected"):
        contract.validate_runtime_plan(plan, now=NOW)
    plan = valid_plan()
    plan["authority_window"]["observed_at"] = plan["dispatch"]["observed_at"]
    with pytest.raises(contract.ContractError, match="dispatch_observation_order_rejected"):
        contract.validate_runtime_plan(plan, now=NOW)
    plan = valid_plan()
    plan["github_evidence"]["nonce_history"]["observed_at"] = plan["authority_window"][
        "observed_at"
    ]
    with pytest.raises(contract.ContractError, match="github_evidence_order_rejected"):
        contract.validate_runtime_plan(plan, now=NOW)


@pytest.mark.parametrize(
    ("path", "field"),
    [
        (("authority_window",), "pending_invitation_count"),
        (("authority_window",), "unexpected_target_job_count"),
        (("github_evidence", "pre_jit_registration_absence"), "total_count"),
        (("github_evidence", "nonce_history"), "historical_match_count"),
        (("github_evidence", "nonce_history"), "unexpected_queued_or_in_progress_count"),
    ],
)
def test_zero_count_attestations_reject_boolean_false(path: tuple[str, ...], field: str) -> None:
    plan = valid_plan()
    target = plan
    for component in path:
        target = target[component]
    target[field] = False
    with pytest.raises(contract.ContractError, match="integer_rejected"):
        contract.validate_runtime_plan(plan, now=NOW)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ({"actor": "b-urge"}, "dispatch_rejected"),
        ({"triggering_actor": "b-urge"}, "dispatch_rejected"),
        ({"run_attempt": 2}, "dispatch_rejected"),
        ({"event": "pull_request"}, "dispatch_rejected"),
        ({"head_sha": "f" * 40}, "job_binding_rejected"),
    ],
)
def test_dispatch_and_job_are_bound_to_owner_attempt_one(
    mutation: dict[str, object], error: str
) -> None:
    plan = valid_plan()
    if "head_sha" in mutation:
        plan["job"]["labels"] = ["wrong"]
    else:
        plan["dispatch"].update(mutation)
    with pytest.raises(contract.ContractError, match=error):
        contract.validate_runtime_plan(plan, now=NOW)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("runner_name", "explainiverse-cuda-single-jit-ffffffffffffffff"),
        ("labels", [RUNNER_NAME, "self-hosted"]),
        ("status", "in_progress"),
        ("work_folder", "_work-shared"),
        ("jit_config_sha256", "9" * 64),
    ],
)
def test_job_identity_and_jit_binding_tampering_is_rejected(field: str, value: object) -> None:
    plan = valid_plan()
    plan["job"][field] = value
    if field == "jit_config_sha256":
        contract.validate_runtime_plan(plan, now=NOW)
        with pytest.raises(executor.RuntimeErrorClosed, match="jit_config_digest_mismatch"):
            executor.validate_jit_config(bytearray(JIT_CONFIG), str(value))
    else:
        with pytest.raises(contract.ContractError):
            contract.validate_runtime_plan(plan, now=NOW)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("device_request", GPU_TWO, "gpu_scope_rejected"),
        ("nvidia_visible_devices", f"{GPU_ONE},{GPU_TWO}", "gpu_scope_rejected"),
        ("cuda_visible_devices", "0,1", "gpu_scope_rejected"),
        ("exclusive_device_scope_required", False, "gpu_scope_rejected"),
        ("unrequested_physical_gpu_uuids", [GPU_TWO], "unrequested_gpu_partition"),
    ],
)
def test_gpu_scope_cannot_be_widened_or_partially_described(
    field: str, value: object, error: str
) -> None:
    plan = valid_plan()
    plan["hardware"][field] = value
    with pytest.raises(contract.ContractError, match=error):
        contract.validate_runtime_plan(plan, now=NOW)


def test_two_gpu_job_requires_exact_two_assigned_physical_uuids() -> None:
    plan = valid_plan()
    plan["dispatch"]["inputs"].update(
        {
            "single_minimum_runner_nonce": "4444444444444444",
            "two_minimum_runner_nonce": NONCE,
        }
    )
    plan["job"].update(
        {
            "ordinal": 3,
            "key": "two_minimum",
            "name": "CUDA two-GPU scheduled (Torch minimum)",
            "runner_name": f"explainiverse-cuda-two-jit-{NONCE}",
            "labels": [f"explainiverse-cuda-two-jit-{NONCE}"],
        }
    )
    plan["sequencing"]["previous_cleanup_receipt_sha256"] = "5" * 64
    plan["github_evidence"]["jit_response"]["runner"].update(
        {
            "name": f"explainiverse-cuda-two-jit-{NONCE}",
            "labels": [f"explainiverse-cuda-two-jit-{NONCE}"],
        }
    )
    plan["hardware"].update(
        {
            "assigned_physical_gpu_uuids": [GPU_ONE, GPU_TWO],
            "unrequested_physical_gpu_uuids": HOST_GPUS[2:],
            "device_request": f"{GPU_ONE},{GPU_TWO}",
            "nvidia_visible_devices": f"{GPU_ONE},{GPU_TWO}",
            "cuda_visible_devices": "0,1",
            "required_cuda_devices": 2,
        }
    )
    assert contract.validate_runtime_plan(plan, now=NOW)["hardware"][
        "assigned_physical_gpu_uuids"
    ] == [GPU_ONE, GPU_TWO]
    plan["hardware"]["assigned_physical_gpu_uuids"] = [GPU_ONE]
    with pytest.raises(contract.ContractError, match="cardinality"):
        contract.validate_runtime_plan(plan, now=NOW)


@pytest.mark.parametrize(
    ("section", "field", "value", "error"),
    [
        ("runner_source", "version", "2.337.0", "runner_source_rejected"),
        ("runner_source", "sha256_checksum", "0" * 64, "runner_source_rejected"),
        ("runner_image", "manifest_digest", "sha256:" + "0" * 64, "runner_image_rejected"),
        ("runner_image", "node20_present", False, "runner_image_rejected"),
        ("runner_image", "node20_sha256", "0" * 64, "runner_image_rejected"),
    ],
)
def test_runner_download_and_immutable_image_must_match_action_time_record(
    section: str, field: str, value: object, error: str
) -> None:
    plan = valid_plan()
    plan[section][field] = value
    with pytest.raises(contract.ContractError, match=error):
        contract.validate_runtime_plan(plan, now=NOW)


def test_pre_jit_absence_nonce_history_and_exact_jit_runner_are_required() -> None:
    for mutate, error in (
        (
            lambda plan: plan["github_evidence"]["pre_jit_registration_absence"].update(
                {"total_count": 1}
            ),
            "pre_jit_total_count_integer_rejected",
        ),
        (
            lambda plan: plan["github_evidence"]["nonce_history"].update(
                {"historical_match_count": 1}
            ),
            "historical_match_count_integer_rejected",
        ),
        (
            lambda plan: plan["github_evidence"]["jit_response"]["runner"].update(
                {"labels": [RUNNER_NAME, "self-hosted"]}
            ),
            "jit_runner_rejected",
        ),
    ):
        plan = valid_plan()
        mutate(plan)
        with pytest.raises(contract.ContractError, match=error):
            contract.validate_runtime_plan(plan, now=NOW)


def test_generate_jitconfig_runner_os_must_be_unknown_before_start() -> None:
    plan = valid_plan()
    assert plan["github_evidence"]["jit_response"]["runner"]["os"] == "unknown"
    plan["github_evidence"]["jit_response"]["runner"]["os"] = "linux"
    with pytest.raises(contract.ContractError, match="jit_runner_rejected"):
        contract.validate_runtime_plan(plan, now=NOW)


def test_docker_argv_is_nonroot_read_only_tmpfs_and_contains_no_secret_channel() -> None:
    plan = contract.validate_runtime_plan(valid_plan(), now=NOW)
    argv = contract.render_docker_run_argv(plan)
    joined = "\n".join(argv)
    assert argv[:2] == ("/usr/bin/docker", "run")
    assert contract.IMAGE_REFERENCE in argv
    assert "--pull=never" in argv
    assert "--read-only" in argv
    assert "1001:1001" in argv
    assert "--cap-drop=ALL" in argv
    assert "--security-opt=no-new-privileges:true" in argv
    assert "--log-driver=none" in argv
    assert contract.GLOBAL_RUNTIME_LABEL in argv
    assert argv[argv.index("--gpus") + 1] == f"device={GPU_ONE}"
    assert f"NVIDIA_VISIBLE_DEVICES={GPU_ONE}" in argv
    assert "CUDA_VISIBLE_DEVICES=0" in argv
    assert "ACTIONS_RUNNER_INPUT_JITCONFIG" in contract.CONTAINER_LAUNCHER
    assert "cp -r --no-preserve=ownership" in contract.CONTAINER_LAUNCHER
    assert JIT_CONFIG.decode() not in joined
    assert "--volume" not in argv
    assert "--mount" not in argv
    assert "/var/run/docker.sock" not in joined
    assert "--privileged" not in argv
    assert "self-hosted" not in joined
    assert not any("encoded_jit" in item.lower() for item in argv)

    two_gpu_plan = deepcopy(plan)
    two_gpu_plan["hardware"].update(
        {
            "assigned_physical_gpu_uuids": [GPU_ONE, GPU_TWO],
            "unrequested_physical_gpu_uuids": HOST_GPUS[2:],
            "device_request": f"{GPU_ONE},{GPU_TWO}",
            "nvidia_visible_devices": f"{GPU_ONE},{GPU_TWO}",
            "cuda_visible_devices": "0,1",
            "required_cuda_devices": 2,
        }
    )
    two_gpu_argv = contract.render_docker_run_argv(two_gpu_plan)
    assert two_gpu_argv[two_gpu_argv.index("--gpus") + 1] == (f'"device={GPU_ONE},{GPU_TWO}"')


def test_dedicated_network_has_no_ports_and_blocks_nonpublic_destinations() -> None:
    plan = contract.validate_runtime_plan(valid_plan(), now=NOW)
    setup = contract.expected_network_setup_argv(plan)
    flattened = "\n".join(" ".join(argv) for argv in setup)
    assert "enable_icc=false" in flattened
    assert "enable_ip_masquerade=true" in flattened
    for destination in contract.BLOCKED_IPV4_DESTINATIONS:
        assert f"-d {destination} -j REJECT" in flattened
    assert "--dport 443" in flattened
    assert "--dport 53" in flattened
    docker_argv = contract.render_docker_run_argv(plan)
    assert "--publish" not in docker_argv
    assert "-p" not in docker_argv
    cleanup = contract.expected_network_cleanup_argv(plan)
    assert cleanup[-1][:3] == ("/usr/bin/docker", "network", "rm")


def test_image_probe_has_no_network_and_requires_exact_node20_runtime() -> None:
    argv = contract.render_image_probe_argv()
    assert "--network=none" in argv
    assert contract.IMAGE_REFERENCE in argv
    assert contract.PROBE_CONTAINER_NAME in argv
    assert contract.GLOBAL_RUNTIME_LABEL in argv
    assert contract.IMAGE_NODE20_SHA256 in contract.IMAGE_PROBE_SCRIPT
    output = (
        b"uid=1001\n"
        b"gid=1001\n"
        b"runner_listener=present\n"
        b"runner_version=2.336.0\n"
        b"runner_commit=98aabcd429c4e8402406c56ce2d26387fed3b9ce\n"
        b"node20=v20.20.2\n"
        b"node20_sha256=6295488653f0d93b0a157841746fef7e72cc4328cfb60c4bbe0ca2668a836ffd\n"
    )
    assert contract.validate_image_probe_output(output)["node20_present"] is True
    with pytest.raises(contract.ContractError, match="image_probe_output_rejected"):
        contract.validate_image_probe_output(output.replace(b"v20.20.2", b"v24.18.0"))

    gpu_argv = contract.render_gpu_injection_probe_argv(HOST_GPUS)
    assert "--network=none" in gpu_argv
    assert gpu_argv[gpu_argv.index("--gpus") + 1] == f'"device={",".join(HOST_GPUS)}"'
    assert contract.GLOBAL_RUNTIME_LABEL in gpu_argv
    gpu_output = b"gpu_injection=verified\n" b"gpu_count=8\n" b"gpu_product=NVIDIA A100-SXM4-80GB\n"
    assert (
        contract.validate_gpu_injection_probe_output(gpu_output)["gpu_injection_verified"] is True
    )


def test_fixed_probe_host_waits_for_cloud_init_before_full_posture_and_binds_products(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_probe_output = (
        b"uid=1001\n"
        b"gid=1001\n"
        b"runner_listener=present\n"
        b"runner_version=2.336.0\n"
        b"runner_commit=98aabcd429c4e8402406c56ce2d26387fed3b9ce\n"
        b"node20=v20.20.2\n"
        b"node20_sha256=6295488653f0d93b0a157841746fef7e72cc4328cfb60c4bbe0ca2668a836ffd\n"
    )
    inventory = "".join(f"{uuid}, {contract.HOST_GPU_PRODUCT}\n" for uuid in HOST_GPUS).encode()
    events: list[str] = []

    class FakeCommands:
        def run(self, argv, **kwargs):
            del kwargs
            command = tuple(argv)
            events.append(" ".join(command))
            exit_code = 0
            if command == (contract.CLOUD_INIT_PATH, "status", "--wait"):
                stdout = b"status: done\n"
            elif command == (
                contract.DOCKER_PATH,
                "container",
                "inspect",
                contract.PROBE_CONTAINER_NAME,
            ):
                stdout = b""
                exit_code = 1
            elif command == (
                contract.NVIDIA_SMI_PATH,
                "--query-gpu=uuid,name",
                "--format=csv,noheader",
            ):
                stdout = inventory
            elif command[:3] == (contract.DOCKER_PATH, "image", "inspect"):
                stdout = json.dumps(valid_image_inspect()).encode()
            elif command == contract.render_image_probe_argv():
                stdout = image_probe_output
            elif command == contract.render_gpu_injection_probe_argv(HOST_GPUS):
                stdout = (
                    b"gpu_injection=verified\n"
                    b"gpu_count=8\n"
                    b"gpu_product=NVIDIA A100-SXM4-80GB\n"
                )
            else:
                stdout = b""
            return executor.CommandResult(command, exit_code, stdout, b"")

    monkeypatch.setattr(executor, "harden_secret_process", lambda: events.append("harden"))
    monkeypatch.setattr(executor, "verify_no_sensitive_environment", lambda: None)
    monkeypatch.setattr(executor, "require_probe_stdin_eof", lambda: None)
    monkeypatch.setattr(executor.os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(executor, "_verify_host_binary", lambda path: events.append(path))
    monkeypatch.setattr(executor, "verify_host_posture", lambda: events.append("posture"))
    monkeypatch.setattr(executor, "runtime_bundle_sha256", lambda: "9" * 64)
    monkeypatch.setattr(executor, "ExclusiveRuntimeLock", nullcontext)

    receipt = executor.probe_host(FakeCommands())
    cloud_index = events.index(f"{contract.CLOUD_INIT_PATH} status --wait")
    assert cloud_index < events.index("posture")
    assert receipt["kind"] == contract.HOST_PREFLIGHT_KIND
    assert receipt["host_physical_gpu_count"] == 8
    assert receipt["host_physical_gpu_uuids"] == HOST_GPUS
    assert receipt["host_physical_gpu_products"] == HOST_GPU_PRODUCTS
    assert receipt["jit_config_received"] is False
    assert receipt["github_api_credential_received"] is False


def test_image_inspect_requires_exact_platform_config_and_manifest_digests() -> None:
    assert contract.validate_image_inspect(valid_plan(), valid_image_inspect())["Os"] == "linux"
    for field, value in (
        ("Id", "sha256:" + "0" * 64),
        ("Architecture", "arm64"),
        ("RepoDigests", ["ghcr.io/actions/actions-runner@sha256:" + "0" * 64]),
    ):
        observed = valid_image_inspect()
        observed[0][field] = value
        with pytest.raises(contract.ContractError):
            contract.validate_image_inspect(valid_plan(), observed)


def test_host_gpu_inventory_is_exact_and_ordered() -> None:
    plan = valid_plan()
    output = "".join(f"{uuid}, {contract.HOST_GPU_PRODUCT}\n" for uuid in HOST_GPUS).encode()
    assert contract.validate_host_gpu_inventory(plan, output) == HOST_GPUS
    with pytest.raises(contract.ContractError, match="host_gpu_inventory_mismatch"):
        swapped = [GPU_TWO, GPU_ONE, *HOST_GPUS[2:]]
        contract.validate_host_gpu_inventory(
            plan,
            "".join(f"{uuid}, {contract.HOST_GPU_PRODUCT}\n" for uuid in swapped).encode(),
        )
    with pytest.raises(contract.ContractError, match="gpu_product_rejected"):
        contract.validate_host_gpu_inventory(
            plan,
            output.replace(contract.HOST_GPU_PRODUCT.encode(), b"NVIDIA H100 80GB", 1),
        )


def test_local_controller_observation_verifiers_bind_pre_and_post_state() -> None:
    plan = contract.validate_runtime_plan(valid_plan(), now=NOW)
    pre = live_observation(completed=False)
    post = live_observation(completed=True)
    assert contract.validate_pre_execution_observation(plan, pre, now=NOW) == pre
    assert contract.validate_post_execution_observation(plan, post, now=NOW) == post
    post["jobs"][0]["conclusion"] = "failure"
    with pytest.raises(contract.ContractError, match="post_execution_job_binding_rejected"):
        contract.validate_post_execution_observation(plan, post, now=NOW)


def test_remote_receipt_never_claims_job_tests_or_registration_absence() -> None:
    plan = contract.validate_runtime_plan(valid_plan(), now=NOW)
    receipt = contract.build_runtime_receipt(
        plan,
        host_gpu_uuids=HOST_GPUS,
        started_at="2026-08-28T21:08:01Z",
        jit_config_sent_at="2026-08-28T21:08:02Z",
        stopped_at="2026-08-28T21:08:10Z",
        cleanup_verified_at="2026-08-28T21:08:11Z",
        runner_exit_code=0,
    )
    assert receipt["status"] == "runner-container-stopped-and-host-cleaned"
    assert receipt["github_contacted_by_runtime"] is False
    assert receipt["job_success_verified_by_runtime"] is False
    assert receipt["test_counts_verified_by_runtime"] is False
    assert receipt["post_exit_registration_absence_verified_by_runtime"] is False
    assert receipt["post_exit_registration_state"] == "not-observed-on-remote-host"
    assert receipt["accepted_actions_evidence"] is False
    assert "runner_registration_present" not in receipt
    assert "claimed_job_count" not in receipt
    assert JIT_CONFIG.decode() not in contract.canonical_json(receipt).decode()


def test_receipt_enforces_workload_expiry_and_bounded_cleanup_grace() -> None:
    plan = contract.validate_runtime_plan(valid_plan(), now=NOW)
    common = {
        "host_gpu_uuids": HOST_GPUS,
        "started_at": "2026-08-28T21:19:00Z",
        "jit_config_sent_at": "2026-08-28T21:19:01Z",
        "runner_exit_code": 0,
    }
    with pytest.raises(contract.ContractError, match="stopped_after_authority_expiry"):
        contract.build_runtime_receipt(
            plan,
            **common,
            stopped_at="2026-08-28T21:20:00.001Z",
            cleanup_verified_at="2026-08-28T21:20:01Z",
        )
    with pytest.raises(contract.ContractError, match="cleanup_after_grace"):
        contract.build_runtime_receipt(
            plan,
            **common,
            stopped_at="2026-08-28T21:19:59Z",
            cleanup_verified_at="2026-08-28T21:21:00.001Z",
        )


def test_shared_deadline_rejects_command_that_finishes_after_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Inner:
        def run(self, argv, **kwargs):
            assert kwargs["timeout"] == 1
            return executor.CommandResult(tuple(argv), 0, b"", b"")

    ticks = iter((100.2, 101.1))
    monkeypatch.setattr(executor.time, "monotonic", lambda: next(ticks))
    bounded = executor.DeadlineCommands(Inner(), 101.0, "authority_expired_during_setup")
    with pytest.raises(executor.RuntimeErrorClosed, match="authority_expired_during_setup"):
        bounded.run(("/fixed",), timeout=30)


def test_global_residue_detects_previous_nonce_and_detached_firewall_chain() -> None:
    class ResidueCommands:
        def __init__(self, *, container_ids: bytes = b"", rules: bytes = b"") -> None:
            self.container_ids = container_ids
            self.rules = rules

        def run(self, argv, **kwargs):
            del kwargs
            command = tuple(argv)
            if command[1:3] == ("container", "inspect"):
                return executor.CommandResult(command, 1, b"", b"")
            if command[1:3] == ("container", "ls"):
                return executor.CommandResult(command, 0, self.container_ids, b"")
            if command[1:3] == ("network", "ls"):
                return executor.CommandResult(command, 0, b"", b"")
            if command == (contract.IPTABLES_PATH, "-S"):
                return executor.CommandResult(command, 0, self.rules, b"")
            raise AssertionError(command)

    with pytest.raises(executor.RuntimeErrorClosed, match="runtime_residue"):
        executor._ensure_no_global_runtime_residue(ResidueCommands(container_ids=b"old\n"))
    with pytest.raises(executor.RuntimeErrorClosed, match="runtime_firewall_residue"):
        executor._ensure_no_global_runtime_residue(ResidueCommands(rules=b"-N EXJIT_DETACHED\n"))


def test_partial_network_cleanup_continues_after_one_cleanup_error() -> None:
    plan = contract.validate_runtime_plan(valid_plan(), now=NOW)

    class CleanupCommands:
        def __init__(self) -> None:
            self.commands: list[tuple[str, ...]] = []
            self.failed_once = False

        def run(self, argv, **kwargs):
            del kwargs
            command = tuple(argv)
            self.commands.append(command)
            if command[1:4] == ("container", "rm", "--force") and not self.failed_once:
                self.failed_once = True
                raise executor.RuntimeErrorClosed("injected_cleanup_failure")
            if command[1:3] in {("container", "inspect"), ("network", "inspect")}:
                return executor.CommandResult(command, 1, b"", b"")
            if command[0] == contract.IPTABLES_PATH and command[1] in {"-C", "-S"}:
                if len(command) > 2:
                    return executor.CommandResult(command, 1, b"", b"")
                return executor.CommandResult(command, 0, b"", b"")
            return executor.CommandResult(command, 0, b"", b"")

    commands = CleanupCommands()
    executor._cleanup_runtime(commands, plan)
    expected_cleanup = {tuple(item) for item in contract.expected_network_cleanup_argv(plan)}
    assert expected_cleanup.issubset(set(commands.commands))


def test_jit_config_validation_uses_only_digest_and_stable_error() -> None:
    secret = bytearray(JIT_CONFIG)
    executor.validate_jit_config(secret, JIT_SHA)
    with pytest.raises(executor.RuntimeErrorClosed) as caught:
        executor.validate_jit_config(secret, "0" * 64)
    assert str(caught.value) == "jit_config_digest_mismatch"
    assert JIT_CONFIG.decode() not in str(caught.value)


def test_remote_executor_contains_no_http_or_github_credential_path() -> None:
    source = (PACKAGE_ROOT / "executor.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    lowered = source.lower()
    assert "http.client" not in imported
    assert "urllib" not in imported
    assert "requests" not in imported
    assert "github-token" not in lowered
    assert "authorization" not in lowered
    assert "api.github.com" not in lowered
    args = executor._parse_cli(["run"])
    assert vars(args) == {"command": "run"}
    assert vars(executor._parse_cli(["probe-host"])) == {"command": "probe-host"}
    with pytest.raises(SystemExit):
        executor._parse_cli(["probe-host", "--host-gpus", GPU_ONE])


def test_remote_rejects_inherited_credential_environment_without_reading_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(executor.os, "environ", {"PATH": "/usr/bin"})
    executor.verify_no_sensitive_environment()
    monkeypatch.setattr(
        executor.os,
        "environ",
        {"PATH": "/usr/bin", "GITHUB_TOKEN": "must-not-be-inspected-or-logged"},
    )
    with pytest.raises(executor.RuntimeErrorClosed) as caught:
        executor.verify_no_sensitive_environment()
    assert str(caught.value) == "sensitive_environment_rejected"
    assert "must-not-be-inspected-or-logged" not in str(caught.value)


def test_fixed_bootstrap_frame_has_exact_header_and_no_payload_copy() -> None:
    plan_bytes = contract.canonical_json(valid_plan())
    jit = bytearray(JIT_CONFIG)
    header = bootstrap.frame_header(plan_bytes, jit)
    assert len(header) == 84
    magic, version, flags, plan_length, jit_length, plan_sha, jit_sha = bootstrap.HEADER.unpack(
        header
    )
    assert magic == b"EXJIT01\n"
    assert version == 1
    assert flags == 0
    assert plan_length == len(plan_bytes)
    assert jit_length == len(jit)
    assert plan_sha == hashlib.sha256(plan_bytes).digest()
    assert jit_sha == hashlib.sha256(jit).digest()
    assert plan_bytes not in header
    assert bytes(jit) not in header


def test_bootstrap_and_executor_remote_argv_are_fixed_and_value_free() -> None:
    bootstrap_source = (PACKAGE_ROOT / "bootstrap.py").read_text(encoding="utf-8")
    executor_source = (PACKAGE_ROOT / "executor.py").read_text(encoding="utf-8")
    assert '(contract.PYTHON_PATH, "-B", str(executor_path), "run")' in bootstrap_source
    assert "--policy-sha256" not in executor_source
    assert "--runner-ordinal" not in executor_source
    assert "--device-request" not in executor_source
    assert "--hard-wall-seconds" not in executor_source
    assert 'context="jit_config"' in executor_source
    assert "absolute_deadline=authority_deadline" in executor_source
    assert 'context="runner_plan"' in executor_source
    assert JIT_CONFIG.decode() not in bootstrap_source
    readme = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
    assert "fixed no-argument bootstrap" in readme
    assert "with the plan-bound policy SHA, ordinal, deadline" not in readme
    assert (
        "/usr/bin/sudo -n -- /usr/bin/python3 -B "
        "/opt/explainiverse/bin/release_gpu_jit_lambda_runtime/bootstrap.py"
    ) in readme
    assert "sudo_noninteractive=true" in readme


def test_fd_contract_rejects_generic_network_sockets() -> None:
    pattern = executor.ANONYMOUS_FD_RE
    assert pattern.fullmatch("pipe:[123]") is not None
    assert pattern.fullmatch("/memfd:jit (deleted)") is not None
    assert pattern.fullmatch("socket:[123]") is None
    assert pattern.fullmatch("/tmp/named-fifo") is None


def test_runtime_names_are_nonce_bound_and_do_not_accept_shell_text() -> None:
    plan = contract.validate_runtime_plan(valid_plan(), now=NOW)
    names = contract.runtime_names(plan)
    assert names == {
        "container": f"explainiverse-jit-{NONCE}",
        "network": f"exjit-{NONCE}",
        "bridge": f"xjit{NONCE[:10]}",
        "chain": f"EXJIT_{NONCE[:12].upper()}",
    }
    tampered = deepcopy(plan)
    tampered["job"]["runner_nonce"] = "$(id)"
    with pytest.raises(contract.ContractError):
        contract.validate_runtime_plan(tampered, now=NOW)
