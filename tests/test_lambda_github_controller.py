from __future__ import annotations

import base64
import errno
import hashlib
import json
import os
import subprocess
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

import pytest
from lambda_operator_receipt_fixtures import operator_preflight_fixture

from scripts import release_external_controls as controls
from scripts.release_gpu_jit_lambda_controller import controller, driver
from scripts.release_gpu_jit_lambda_live import adapter as live
from scripts.release_gpu_jit_lambda_runtime import runtime_contract as runtime

NOW = datetime(2026, 8, 28, 22, 0, 0, tzinfo=timezone.utc)
HEAD = "a" * 40
CONTROL_SHA = "c" * 64
NONCES = (
    "0000000000000001",
    "0000000000000002",
    "0000000000000003",
    "0000000000000004",
)
PUBLICATION_NONCES = ("1000000000000001", "1000000000000002")
GPU_UUIDS = tuple(
    f"GPU-{index:08x}-{index:04x}-{index:04x}-{index:04x}-{index:012x}" for index in range(1, 9)
)
TEST_RESOURCES = controller.SealedControllerResources.from_files_for_tests()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("ascii")


def _immutable_plan_mapping(*, lifecycle_nonce: str) -> dict[str, Any]:
    plan = live.build_immutable_plan(
        head_sha=HEAD,
        lifecycle_nonce=lifecycle_nonce,
        created_at_unix=int((NOW - timedelta(minutes=15)).timestamp()),
        expires_at_unix=int((NOW + timedelta(minutes=45)).timestamp()),
        current_public_ipv4_cidr="8.8.8.8/32",
        region_description="Illinois, USA",
        image_id="image-ubuntu-2204-a100",
        image_created_time="2026-07-01T00:00:00Z",
        image_description="Ubuntu 22.04 Lambda Stack",
        image_name="Lambda Stack 22.04",
        image_family=live.TARGET_IMAGE_FAMILY,
        image_version="2026.08",
        image_updated_time="2026-08-01T00:00:00Z",
        instance_type_description="8x A100 (80 GB SXM4)",
        gpu_description="A100 (80 GB SXM4)",
        price_cents_per_hour=2232,
        vcpus=240,
        memory_gib=1800,
        storage_gib=20480,
        ssh_key_name="preexisting-fixture-key",
        ssh_public_key_sha256="3" * 64,
        baseline_file_systems_sha256="7" * 64,
        original_global_rules=(
            {
                "protocol": "tcp",
                "port_range": [22, 22],
                "source_network": "0.0.0.0/0",
                "description": "Original SSH",
            },
            {
                "protocol": "icmp",
                "source_network": "0.0.0.0/0",
                "description": "Original ICMP",
            },
        ),
        host_key_fingerprint="SHA256:" + "A" * 43,
        runtime_bundle_sha256="8" * 64,
    )
    return plan.to_mapping()


def _operator_immutable_plan(phase: str) -> dict[str, Any]:
    _, expected = operator_preflight_fixture(
        phase,
        policy_sha256=TEST_RESOURCES.policy_sha256,
        controller_source_sha256=TEST_RESOURCES.controller_source_sha256,
        runtime_bundle_sha256="8" * 64,
    )
    return deepcopy(expected["expected_immutable_plan"])


def _rewrite_journal_chain(
    root: Path,
    mutation: Callable[[int, dict[str, Any]], None],
) -> str:
    previous_sha256: str | None = None
    paths = sorted(root.glob("[0-9][0-9][0-9]-*.json"))
    for sequence, path in enumerate(paths, start=1):
        envelope = json.loads(path.read_text(encoding="ascii"))
        mutation(sequence, envelope)
        envelope["previous_evidence_sha256"] = previous_sha256
        raw = _canonical(envelope)
        path.write_bytes(raw)
        previous_sha256 = hashlib.sha256(raw).hexdigest()
    assert previous_sha256 is not None
    return previous_sha256


class QueueTransport:
    def __init__(self) -> None:
        self.responses: dict[tuple[str, str], deque[controller.GitHubResponse]] = defaultdict(deque)
        self.calls: list[tuple[str, str, Mapping[str, Any] | None]] = []

    def add(self, method: str, path: str, status: int, value: Any = None) -> None:
        if value is None:
            body = b""
        elif isinstance(value, bytes):
            body = value
        else:
            body = _canonical(value)
        self.responses[(method, path)].append(
            controller.GitHubResponse(method, path, status, bytearray(body), "f" * 64)
        )

    def request(
        self, method: str, path: str, body: Mapping[str, Any] | None = None
    ) -> controller.GitHubResponse:
        self.calls.append((method, path, body))
        assert self.responses[(method, path)], (method, path)
        return self.responses[(method, path)].popleft()


class NoRemote:
    def wait_cloud_init(self, binding: live.StrictSshBinding) -> controller.PublicSshResult:
        raise AssertionError("unexpected remote call")

    def probe_host(self, binding: live.StrictSshBinding) -> controller.PublicSshResult:
        raise AssertionError("unexpected remote call")

    def run_job(
        self,
        binding: live.StrictSshBinding,
        canonical_plan: bytes,
        jit_config: live.SecretBuffer,
    ) -> controller.RemoteExecution:
        raise AssertionError("unexpected remote call")


def _response_runner(
    name: str,
    *,
    runner_id: int = 41,
    os_name: str = "unknown",
    labels: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": runner_id,
        "name": name,
        "os": os_name,
        "status": "offline",
        "busy": False,
        "labels": labels if labels is not None else [{"id": 91, "name": name, "type": "custom"}],
    }


def _app_capture(
    *,
    captured_at: datetime = NOW,
) -> tuple[dict[str, Any], Mapping[str, bytes]]:
    policy, _ = controls.load_policy(controller.RELEASE_CONTROL_POLICY_PATH)
    apps = policy["release_runner_authority"]["installed_apps"]
    captured_at_text = captured_at.isoformat()
    installations = sorted(deepcopy(apps["expected_installations"]), key=lambda item: item["id"])
    roles: list[tuple[str, int | None]] = [("installation-list", None)]
    roles.extend(("installation-configure", item["id"]) for item in installations)
    roles.extend(
        ("permission-update", item["id"])
        for item in installations
        if item["permission_update_requested"]
    )
    evidence: list[dict[str, Any]] = []
    raw_pages: dict[str, bytes] = {}
    for kind, installation_id in roles:
        suffix = "list" if installation_id is None else str(installation_id)
        if kind == "installation-list":
            source_url = "https://github.com/settings/installations"
        elif kind == "installation-configure":
            source_url = f"https://github.com/settings/installations/{installation_id}"
        else:
            source_url = (
                f"https://github.com/settings/installations/{installation_id}/permissions/update"
            )
        item = {
            "filename": f"capture-{kind}-{suffix}.txt",
            "kind": kind,
            "installation_id": installation_id,
            "source_url": source_url,
            "captured_at": captured_at_text,
            "media_type": "text/plain; charset=utf-8",
            "full_page": True,
        }
        raw = (
            controls._app_evidence_header(item)
            + f"full owner-authenticated page capture: {item['filename']}\n"
        ).encode("utf-8")
        item.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        evidence.append(item)
        raw_pages[item["filename"]] = raw
    capture = {
        "schema_version": 1,
        "repository": controller.REPOSITORY,
        "captured_at": captured_at_text,
        "capture_principal": controller.OWNER,
        "source_url": apps["source_url"],
        "coverage_complete": True,
        "installations": installations,
        "evidence": evidence,
    }
    return capture, raw_pages


def _session() -> controller.PhaseSession:
    jobs = tuple(
        controller.JobBinding(
            key,
            ordinal,
            100 + ordinal,
            str(runtime.JOB_SPECS[key]["name"]),
            NONCES[ordinal - 1],
            f"{runtime.JOB_SPECS[key]['prefix']}{NONCES[ordinal - 1]}",
        )
        for ordinal, key in enumerate(
            ("single_minimum", "single_latest", "two_minimum", "two_latest"), start=1
        )
    )
    return controller.PhaseSession(
        phase="final-main",
        workflow="cuda-ci.yml",
        workflow_path=runtime.CUDA_WORKFLOW_PATH,
        dispatch_ref="main",
        run_ref=runtime.FINAL_MAIN_REF,
        head_sha=HEAD,
        inputs=dict(zip(runtime.CUDA_NONCE_INPUT_KEYS, NONCES)),
        prior_accepted_cuda_runner_nonces=(),
        run={"id": 500, "run_attempt": 1, "status": "in_progress"},
        jobs=jobs,
        queued_jobs=jobs,
        dispatch_receipt=controller.DispatchReceipt(
            observed_at=controller._iso(NOW - timedelta(minutes=5)),
            request_sha256="1" * 64,
            response_sha256="2" * 64,
            workflow_response_sha256="3" * 64,
            run_response_sha256="4" * 64,
            nonce_history_observed_at=controller._iso(NOW - timedelta(minutes=6)),
            nonce_history_response_sha256="5" * 64,
            mutation_response_received=True,
            mutation_reconciliation_sha256="6" * 64,
        ),
    )


def _publication_session() -> controller.PhaseSession:
    jobs = tuple(
        controller.JobBinding(
            key,
            ordinal,
            700 + ordinal,
            str(runtime.JOB_SPECS[key]["name"]),
            PUBLICATION_NONCES[ordinal - 1],
            f"{runtime.JOB_SPECS[key]['prefix']}{PUBLICATION_NONCES[ordinal - 1]}",
        )
        for ordinal, key in enumerate(
            ("publication_single_minimum", "publication_single_latest"), start=1
        )
    )
    return controller.PhaseSession(
        phase="publication",
        workflow="publish-pypi.yml",
        workflow_path=runtime.PUBLISH_WORKFLOW_PATH,
        dispatch_ref=runtime.PUBLICATION_TAG,
        run_ref=runtime.PUBLICATION_REF,
        head_sha=HEAD,
        inputs={
            "tag": runtime.PUBLICATION_TAG,
            "preflight_run_id": 600,
            "cuda_run_id": 500,
            "single_minimum_runner_nonce": PUBLICATION_NONCES[0],
            "single_latest_runner_nonce": PUBLICATION_NONCES[1],
            "stage_recovery_drill": True,
        },
        prior_accepted_cuda_runner_nonces=NONCES,
        run={"id": 700, "run_attempt": 1, "status": "in_progress"},
        jobs=jobs,
        queued_jobs=jobs,
        dispatch_receipt=controller.DispatchReceipt(
            observed_at=controller._iso(NOW - timedelta(minutes=5)),
            request_sha256="1" * 64,
            response_sha256="2" * 64,
            workflow_response_sha256="3" * 64,
            run_response_sha256="4" * 64,
            nonce_history_observed_at=controller._iso(NOW - timedelta(minutes=6)),
            nonce_history_response_sha256="5" * 64,
            mutation_response_received=True,
            mutation_reconciliation_sha256="6" * 64,
        ),
        prior_authority_evidence_identities=_synthetic_final_authority_identities(),
    )


def _attempt_jobs_for_session(
    session: controller.PhaseSession,
    *,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    supplied = overrides or {}
    for binding in session.queued_jobs:
        item: dict[str, Any] = {
            "id": binding.job_id,
            "name": binding.name,
            "head_sha": session.head_sha,
            "run_attempt": 1,
            "labels": [binding.runner_name],
            "status": "queued",
            "conclusion": None,
            "runner_id": None,
            "runner_name": None,
        }
        item.update(supplied.get(binding.key, {}))
        result.append(item)
    return result


def _hosted_attempt_job(
    *,
    name: str,
    head_sha: str,
    job_id: int,
    label: str,
    status: str = "completed",
    conclusion: str | None = "success",
) -> dict[str, Any]:
    active = status in {"in_progress", "completed"} and conclusion != "skipped"
    return {
        "id": job_id,
        "name": name,
        "head_sha": head_sha,
        "run_attempt": 1,
        "labels": [label],
        "status": status,
        "conclusion": conclusion,
        "runner_id": job_id + 10_000 if active else None,
        "runner_name": f"GitHub Actions {job_id + 10_000}" if active else None,
    }


def _live_attempt_api_jobs(
    session: controller.PhaseSession,
    *,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    jobs = _attempt_jobs_for_session(session, overrides=overrides)
    if session.phase in {"pull-request", "final-main"}:
        return [
            _hosted_attempt_job(
                name=controller.CUDA_ROUTING_JOB_NAME,
                head_sha=session.head_sha,
                job_id=90,
                label="ubuntu-latest",
            ),
            *jobs,
        ]
    assert session.phase == "publication"
    companions = [
        _hosted_attempt_job(
            name=controller.PUBLICATION_PREFLIGHT_JOB_NAME,
            head_sha=session.head_sha,
            job_id=800,
            label="ubuntu-latest",
        ),
        _hosted_attempt_job(
            name="Verify, build once, and inventory",
            head_sha=session.head_sha,
            job_id=801,
            label="ubuntu-24.04",
            status="queued",
            conclusion=None,
        ),
        _hosted_attempt_job(
            name="Attest the immutable distributions",
            head_sha=session.head_sha,
            job_id=802,
            label="ubuntu-latest",
            status="queued",
            conclusion=None,
        ),
        _hosted_attempt_job(
            name="Publish through PyPI Trusted Publishing",
            head_sha=session.head_sha,
            job_id=803,
            label="ubuntu-latest",
            status="queued",
            conclusion=None,
        ),
        _hosted_attempt_job(
            name="Create the immutable GitHub release",
            head_sha=session.head_sha,
            job_id=804,
            label="ubuntu-latest",
            status="queued",
            conclusion=None,
        ),
        _hosted_attempt_job(
            name="Finalize the immutable GitHub release with fixed commands",
            head_sha=session.head_sha,
            job_id=805,
            label="ubuntu-latest",
            status="queued",
            conclusion=None,
        ),
    ]
    return [companions[0], *jobs, *companions[1:]]


def _phase_dispatch_evidence(
    session: controller.PhaseSession,
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = controller.PHASES[session.phase]
    dispatch_path = (
        f"/repos/{controller.REPOSITORY}/actions/workflows/" f"{spec['workflow']}/dispatches"
    )
    request_body = {"ref": spec["dispatch_ref"], "inputs": session.inputs}
    request_sha256 = hashlib.sha256(
        _canonical({"method": "POST", "path": dispatch_path, "body": request_body})
    ).hexdigest()
    intent = {
        "phase": session.phase,
        "workflow": spec["workflow"],
        "workflow_path": spec["workflow_path"],
        "dispatch_path": dispatch_path,
        "dispatch_ref": spec["dispatch_ref"],
        "run_ref": spec["run_ref"],
        "head_sha": session.head_sha,
        "inputs": dict(session.inputs),
        "expected_runner_nonces": [session.inputs[key] for key in spec["all_nonce_keys"]],
        "pre_dispatch_run_ids": [10, 20],
        "request_sha256": request_sha256,
        "mutation_retried": False,
    }
    reconciliation = {
        "response_received": True,
        "response_sha256": "7" * 64,
        "ambiguity": None,
        "run_id": session.run["id"],
        "run_attempt": 1,
        "head_sha": session.head_sha,
        "run_response_sha256": "4" * 64,
        "queued_jobs": [
            {
                "job_id": item.job_id,
                "job_name": item.name,
                "runner_name": item.runner_name,
                "nonce": item.nonce,
            }
            for item in session.queued_jobs
        ],
        "mutation_retried": False,
    }
    reconciliation_sha256 = hashlib.sha256(_canonical(reconciliation)).hexdigest()
    source_keys = (
        ("main",)
        if session.phase == "final-main"
        else ("ref", "tag", "main", "preflight", "cuda", "accepted_cuda_nonces")
    )
    source_bindings = {
        key: hashlib.sha256(f"dispatch-source:{key}".encode("ascii")).hexdigest()
        for key in source_keys
    }
    response_sha256 = hashlib.sha256(
        _canonical(
            {
                "dispatch_reconciliation_sha256": reconciliation_sha256,
                "source": source_bindings,
            }
        )
    ).hexdigest()
    session.dispatch_receipt = controller.DispatchReceipt(
        observed_at=session.dispatch_receipt.observed_at,
        request_sha256=request_sha256,
        response_sha256=response_sha256,
        workflow_response_sha256=session.dispatch_receipt.workflow_response_sha256,
        run_response_sha256="4" * 64,
        nonce_history_observed_at=session.dispatch_receipt.nonce_history_observed_at,
        nonce_history_response_sha256=(session.dispatch_receipt.nonce_history_response_sha256),
        mutation_response_received=True,
        mutation_reconciliation_sha256=reconciliation_sha256,
    )
    settlement = {
        "schema_version": 1,
        "kind": "explainiverse-github-dispatch-settlement",
        "phase": session.phase,
        "head_sha": session.head_sha,
        "run_id": session.run["id"],
        "run_attempt": 1,
        "dispatch_reconciliation": reconciliation,
        "source_bindings": source_bindings,
        "dispatch_receipt": dict(session.dispatch_receipt.__dict__),
    }
    return intent, settlement


def test_attempt_jobs_projects_live_final_cuda_api_shape() -> None:
    session = _session()
    api_jobs = _live_attempt_api_jobs(session)
    transport = QueueTransport()
    path = (
        f"/repos/{controller.REPOSITORY}/actions/runs/{session.run['id']}"
        "/attempts/1/jobs?filter=all&per_page=100&page=1"
    )
    transport.add(
        "GET",
        path,
        200,
        {"total_count": len(api_jobs), "jobs": api_jobs},
    )
    service = controller.ReleaseGpuController(
        transport,
        NoRemote(),
        resources=TEST_RESOURCES,
        clock=lambda: NOW,
    )
    projected = service._attempt_jobs(session.run["id"], 1)
    assert [item["id"] for item in projected] == [item.job_id for item in session.queued_jobs]
    assert all(item["name"] != controller.CUDA_ROUTING_JOB_NAME for item in projected)


def test_attempt_jobs_projects_live_publication_api_shape() -> None:
    session = _publication_session()
    api_jobs = _live_attempt_api_jobs(session)
    assert len(api_jobs) == 8
    transport = QueueTransport()
    path = (
        f"/repos/{controller.REPOSITORY}/actions/runs/{session.run['id']}"
        "/attempts/1/jobs?filter=all&per_page=100&page=1"
    )
    transport.add(
        "GET",
        path,
        200,
        {"total_count": len(api_jobs), "jobs": api_jobs},
    )
    service = controller.ReleaseGpuController(
        transport,
        NoRemote(),
        resources=TEST_RESOURCES,
        clock=lambda: NOW,
    )
    projected = service._attempt_jobs(session.run["id"], 1)
    assert [item["id"] for item in projected] == [item.job_id for item in session.queued_jobs]
    assert all(
        item["name"] not in controller.REVIEWED_HOSTED_COMPANION_JOB_NAMES for item in projected
    )


def test_attempt_jobs_keeps_unknown_extra_job_for_exact_set_rejection() -> None:
    session = _session()
    api_jobs = [
        *_live_attempt_api_jobs(session),
        _hosted_attempt_job(
            name="Unexpected hosted side job",
            head_sha=session.head_sha,
            job_id=899,
            label="ubuntu-latest",
            status="queued",
            conclusion=None,
        ),
    ]
    projected = controller.ReleaseGpuController._project_reviewed_attempt_jobs(
        api_jobs,
        attempt=1,
    )
    assert projected[-1]["name"] == "Unexpected hosted side job"
    with pytest.raises(controller.ControllerError, match="live_shape_cardinality_rejected"):
        controller.ReleaseGpuController._validate_exact_attempt_job_set(
            projected,
            expected_bindings=controller.ReleaseGpuController._session_expected_job_bindings(
                session
            ),
            head_sha=session.head_sha,
            context="live_shape",
        )


def test_attempt_jobs_requires_accepted_control_before_custom_runner_jobs() -> None:
    session = _session()
    api_jobs = _live_attempt_api_jobs(session)
    api_jobs[0]["conclusion"] = "failure"
    with pytest.raises(controller.ControllerError, match="attempt_projection_control_not_accepted"):
        controller.ReleaseGpuController._project_reviewed_attempt_jobs(api_jobs, attempt=1)


def test_attempt_jobs_rejects_cross_workflow_reviewed_companion() -> None:
    session = _session()
    api_jobs = [
        *_live_attempt_api_jobs(session),
        _hosted_attempt_job(
            name=controller.PUBLICATION_PREFLIGHT_JOB_NAME,
            head_sha=session.head_sha,
            job_id=898,
            label="ubuntu-latest",
        ),
    ]
    with pytest.raises(
        controller.ControllerError,
        match="attempt_projection_cross_workflow_companion_rejected",
    ):
        controller.ReleaseGpuController._project_reviewed_attempt_jobs(api_jobs, attempt=1)


def test_archived_windows_paths_validate_independently_of_verifier_os() -> None:
    assert driver._is_windows_absolute_path(r"C:\fixture\operator\ssh.exe")
    assert not driver._is_windows_absolute_path("/fixture/operator/ssh.exe")
    assert not driver._is_windows_absolute_path(r"fixture\operator\ssh.exe")
    assert not driver._is_windows_absolute_path(r"C:fixture\operator\ssh.exe")


def _provider_restored_payload(control_sha256: str) -> dict[str, Any]:
    return _provider_snapshot_payload(control_sha256, "restored", seed=90)


def _provider_snapshot_payload(
    control_sha256: str,
    phase: str,
    *,
    seed: int,
) -> dict[str, Any]:
    ruleset_id: str | None = None
    instance_id: str | None = None
    instance_public_ipv4: str | None = None
    if phase in {"ruleset_ready", "instance_bound", "instance_absent"}:
        ruleset_id = "ruleset-0123456789abcdef"
    if phase == "instance_bound":
        instance_id = "instance-0123456789abcdef"
        instance_public_ipv4 = "8.8.8.8"
    return {
        "plan_sha256": control_sha256,
        "phase": phase,
        "snapshot_sha256": hashlib.sha256(
            f"provider-snapshot:{phase}:{seed}".encode("ascii")
        ).hexdigest(),
        "receipt_nonce": f"{seed:032x}"[-32:],
        "ruleset_id": ruleset_id,
        "instance_id": instance_id,
        "instance_public_ipv4": instance_public_ipv4,
        "response_bindings": [
            {
                "operation": operation,
                "method": "GET",
                "path": path,
                "request_sha256": live.ProviderRequest(
                    operation, "GET", path, False
                ).request_sha256,
                "request_body_sha256": None,
                "response_body_sha256": hashlib.sha256(
                    f"{phase}:{seed}:{operation}".encode("ascii")
                ).hexdigest(),
                "status_code": 200,
                "content_type": "application/json",
            }
            for operation, path in live.READ_OPERATIONS
        ],
    }


def _record_provider_transition(
    journal: driver.EvidenceJournal,
    *,
    control_sha256: str,
    callback_binding_sha256: str,
    operation: str,
    prestate: Mapping[str, Any],
    next_phase: str,
    seed: int,
) -> dict[str, Any]:
    label = operation.replace("_", "-")
    journal.record(
        f"provider-{label}-intent",
        {
            "plan_sha256": control_sha256,
            "operation": operation,
            "prestate": dict(prestate),
            "mutation_retried": False,
        },
    )
    method, path = live.MUTATION_PATHS[operation]
    if operation == "delete_ruleset":
        path = path.replace("{id}", "ruleset-0123456789abcdef")
    request_body_sha256 = (
        None
        if method == "DELETE"
        else hashlib.sha256(f"provider-body:{operation}:{seed}".encode("ascii")).hexdigest()
    )
    request_sha256 = hashlib.sha256(
        live._canonical_json(
            {
                "operation": operation,
                "method": method,
                "path": path,
                "body_sha256": request_body_sha256,
                "timeout_seconds": live.PROVIDER_TIMEOUT_SECONDS,
            }
        )
    ).hexdigest()
    intent = live.MutationIntent(
        plan_sha256=control_sha256,
        operation=operation,
        prestate_phase=str(prestate["phase"]),
        prestate_snapshot_sha256=str(prestate["snapshot_sha256"]),
        prestate_receipt_nonce=str(prestate["receipt_nonce"]),
        callback_binding_sha256=callback_binding_sha256,
        method=method,
        path=path,
        request_sha256=request_sha256,
        request_body_sha256=request_body_sha256,
        sensitive_body=operation == "launch",
        timeout_seconds=live.PROVIDER_TIMEOUT_SECONDS,
    )
    journal.record("provider-mutation-intent", intent.to_public_mapping())
    journal.record(
        f"provider-{label}",
        {
            "plan_sha256": control_sha256,
            "operation": operation,
            "prestate_snapshot_sha256": prestate["snapshot_sha256"],
            "request_sha256": request_sha256,
            "request_body_sha256": request_body_sha256,
            "response_body_sha256": hashlib.sha256(
                f"provider-response:{operation}:{seed}".encode("ascii")
            ).hexdigest(),
            "response_body_redacted": True,
            "ruleset_id": (
                "ruleset-0123456789abcdef"
                if operation in {"create_ruleset", "launch", "terminate", "delete_ruleset"}
                else None
            ),
            "instance_id": (
                "instance-0123456789abcdef" if operation in {"launch", "terminate"} else None
            ),
            "host_fingerprint": ("SHA256:" + "A" * 43 if operation == "launch" else None),
        },
    )
    poststate = _provider_snapshot_payload(control_sha256, next_phase, seed=seed + 1)
    journal.record(f"provider-{next_phase.replace('_', '-')}", poststate)
    return poststate


def _test_app_inbox_mapping(
    phase: str,
    *,
    accepted: int,
) -> dict[str, Any]:
    expected = 4 if phase == "final-main" else 2
    return {
        "phase": phase,
        "expected_capture_count": expected,
        "accepted_capture_count": accepted,
        "stale_generation_count": 0,
        "stale_generations_sha256": hashlib.sha256(_canonical([])).hexdigest(),
        "owner_private_directory_receipt_sha256": "9" * 64,
        "on_demand_before_each_jit": True,
        "ready_marker_no_replace_required": True,
        "raw_pages_archived_by_driver": True,
    }


def _test_environment_mapping() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "operator-environment-scrub",
        "removed_name_count": 0,
        "removed_names_sha256": hashlib.sha256(b"").hexdigest(),
        "removed_values_observed": False,
        "ambient_credentials_retained": False,
        "ambient_proxies_retained": False,
    }


def _test_operator_executables() -> dict[str, Any]:
    def pinned(name: str) -> dict[str, Any]:
        expected = driver.PINNED_OPERATOR_EXECUTABLES[name]
        acl = {
            "owner_sid": expected["owner_sid"],
            "expected_owner_sid": expected["owner_sid"],
            "unprivileged_write_ace_present": False,
            "dacl_ace_count": 3,
            "dacl_inventory_sha256": hashlib.sha256(f"{name}:acl".encode("ascii")).hexdigest(),
        }
        signature = {
            "status": "Valid",
            "subject": expected["authenticode_subject"],
            "thumbprint": expected["authenticode_thumbprint"],
        }
        row: dict[str, Any] = {
            "absolute_path": expected["absolute_path"],
            "sha256": expected["sha256"],
            "version": expected["version"],
            "regular_file": True,
            "symlink_or_reparse": False,
            "path_lookup_used": False,
            "hardlink_count": 1,
            "pinned_reviewed_identity": True,
            "acl": acl,
            "authenticode": signature,
            "authenticode_validated_by_pinned_helper": True,
        }
        if name == "git":
            row["resolved_runtime"] = {
                "absolute_path": expected["runtime_absolute_path"],
                "sha256": expected["runtime_sha256"],
                "version": expected["version"],
                "hardlink_count": 1,
                "acl": dict(acl),
                "authenticode": dict(signature),
            }
        return row

    return {
        "git": pinned("git"),
        "gh": pinned("gh"),
        "ssh": pinned("ssh"),
        "python": {
            "absolute_path": r"C:\fixture\python\python.exe",
            "sha256": "a" * 64,
            "version": "Python 3.13.15",
            "regular_file": True,
            "symlink_or_reparse": False,
            "path_lookup_used": False,
            "hardlink_count": 1,
            "pinned_runtime_manifest_authority": True,
        },
    }


def _test_operator_repository_and_source(
    *, immutable_plan: Mapping[str, Any], phase: str, preloader_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    excluded = {
        driver.OPERATOR_SOURCE_MANIFEST_RELATIVE,
        driver.OPERATOR_PRELOADER_RELATIVE,
    }
    files: dict[str, dict[str, Any]] = {}
    for index, relative in enumerate(
        sorted(driver.OPERATOR_CRITICAL_SOURCE_PATHS - excluded), start=1
    ):
        raw = f"fixture-source:{relative}:{index}\n".encode("utf-8")
        sha256 = hashlib.sha256(raw).hexdigest()
        if relative == ".github/release-control-policy.json":
            sha256 = TEST_RESOURCES.policy_sha256
        elif relative == "scripts/release_gpu_jit_lambda_controller/controller.py":
            sha256 = TEST_RESOURCES.controller_source_sha256
        files[relative] = {
            "mode": "100644",
            "bytes": len(raw),
            "sha256": sha256,
            "git_blob_sha": hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest(),
        }
    files[driver.OPERATOR_PRELOADER_SHIM_RELATIVE]["sha256"] = driver.OPERATOR_PRELOADER_SHIM_SHA256
    directories: set[str] = set()
    for relative in files:
        parent = Path(relative).parent
        while parent != Path("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    rows = [
        (
            f"{relative}\t{item['mode']}\t{item['bytes']}\t"
            f"{item['sha256']}\t{item['git_blob_sha']}\n"
        ).encode("utf-8")
        for relative, item in sorted(files.items())
    ]
    manifest = {
        "schema_version": 1,
        "kind": "explainiverse-operator-source-worktree-manifest",
        "excluded_paths": [
            driver.OPERATOR_SOURCE_MANIFEST_RELATIVE,
            driver.OPERATOR_PRELOADER_RELATIVE,
        ],
        "files": files,
        "directories": sorted(directories),
        "file_count": len(files),
        "directory_count": len(directories),
        "file_inventory_sha256": hashlib.sha256(b"".join(rows)).hexdigest(),
        "source": "exact-staged-index-blobs",
        "runtime_git_dependency": False,
    }
    manifest_raw = _canonical(manifest)
    manifest_blob = hashlib.sha1(
        f"blob {len(manifest_raw)}\0".encode("ascii") + manifest_raw
    ).hexdigest()
    critical_sources = {
        relative: {
            "bytes": item["bytes"],
            "sha256": item["sha256"],
            "git_blob_sha": item["git_blob_sha"],
        }
        for relative, item in files.items()
    }
    critical_sources[driver.OPERATOR_SOURCE_MANIFEST_RELATIVE] = {
        "bytes": len(manifest_raw),
        "sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "git_blob_sha": manifest_blob,
    }
    critical_sources[driver.OPERATOR_PRELOADER_RELATIVE] = {
        "bytes": 4096,
        "sha256": preloader_sha256,
        "git_blob_sha": "b" * 40,
    }
    tree_items = {
        relative: (item["mode"], item["git_blob_sha"]) for relative, item in files.items()
    }
    tree_items[driver.OPERATOR_SOURCE_MANIFEST_RELATIVE] = ("100644", manifest_blob)
    tree_items[driver.OPERATOR_PRELOADER_RELATIVE] = ("100644", "b" * 40)
    tree_raw = b"".join(
        f"{mode} blob {blob}\t{relative}\0".encode("utf-8")
        for relative, (mode, blob) in sorted(tree_items.items())
    )
    repository = {
        "repository": controller.REPOSITORY,
        "absolute_root": r"C:\fixture\repository",
        "origin_url": "https://github.com/jemsbhai/explainiverse.git",
        "head_sha": immutable_plan["head_sha"],
        "tree_object_sha": "b" * 40,
        "tree_inventory_sha256": hashlib.sha256(tree_raw).hexdigest(),
        "clean_tracked_and_untracked": True,
        "supplied_ref": (
            runtime.FINAL_MAIN_REF if phase == "final-main" else runtime.PUBLICATION_REF
        ),
        "remote_object_type": "commit" if phase == "final-main" else "tag",
        "remote_object_sha": (immutable_plan["head_sha"] if phase == "final-main" else "c" * 40),
        "remote_target_sha": immutable_plan["head_sha"],
        "remote_ref_response_sha256": "d" * 64,
        "annotated_tag_response_sha256": None if phase == "final-main" else "e" * 64,
        "critical_sources": critical_sources,
        "git_configuration": {
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
    }
    captured = {
        relative: item
        for relative, item in files.items()
        if relative.startswith(driver.OPERATOR_CAPTURE_PREFIXES)
        or relative in driver.OPERATOR_CAPTURE_EXACT
    }
    capture_rows = [
        f"{relative}\t{item['bytes']}\t{item['sha256']}\n".encode("utf-8")
        for relative, item in sorted(captured.items())
    ]
    source_material = {
        "schema_version": 1,
        "kind": "explainiverse-operator-clean-source-preload",
        "repository_root": repository["absolute_root"],
        "origin_url": repository["origin_url"],
        "head_sha": immutable_plan["head_sha"],
        "head_and_origin_verified_during_credential_free_inventory": False,
        "source_manifest": manifest,
        "source_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "source_manifest_inventory_sha256": manifest["file_inventory_sha256"],
        "tracked_and_untracked_clean": True,
        "runtime_git_dependency": False,
        "preloader_path": r"C:\fixture\sealed\preloader.py",
        "preloader_sha256": preloader_sha256,
        "captured_module_count": len(captured),
        "captured_module_inventory_sha256": hashlib.sha256(b"".join(capture_rows)).hexdigest(),
        "project_modules_execute_from_captured_bytes": True,
        "arguments_sha256": "f" * 64,
    }
    source = {
        **source_material,
        "evidence_sha256": hashlib.sha256(_canonical(source_material)).hexdigest(),
    }
    return repository, source


def _test_directory_receipt(seed: str) -> tuple[dict[str, Any], dict[str, Any]]:
    owner_sid = "S-1-5-21-1000"
    captured_at = (NOW - timedelta(minutes=20)).isoformat()
    acl_material = {
        "owner_sid": owner_sid,
        "current_user_sid": owner_sid,
        "inheritance_protected": True,
        "child_inheritance_enabled": True,
        "aces": sorted(
            [
                {
                    "sid": sid,
                    "access": "allow",
                    "rights": "full-control",
                    "mask": 2_032_127,
                    "ace_flags": 3,
                }
                for sid in (owner_sid, "S-1-5-18", "S-1-5-32-544")
            ],
            key=lambda item: item["sid"],
        ),
        "security_descriptor_sha256": seed * 64,
        "security_descriptor_bytes": 128,
    }
    acl = {
        **acl_material,
        "captured_at": captured_at,
        "evidence_sha256": hashlib.sha256(_canonical(acl_material)).hexdigest(),
    }
    public = {
        "captured_at": captured_at,
        "receipt_sha256": hashlib.sha256(f"directory:{seed}".encode("ascii")).hexdigest(),
        "absolute_path_redacted": True,
        "directory_identity_recorded": True,
        "no_reparse_or_symlink": True,
        "owner_private": True,
        "acl": acl,
    }
    validation = {
        "validated_at": (NOW - timedelta(minutes=19)).isoformat(),
        "receipt_sha256": public["receipt_sha256"],
        "absolute_path_redacted": True,
        "directory_identity_recorded": True,
        "no_reparse_or_symlink": True,
        "owner_private": True,
        "acl_evidence_sha256": acl["evidence_sha256"],
    }
    return public, validation


def _test_secure_launch_mapping(
    *,
    immutable_plan: Mapping[str, Any],
    working_directory: Path,
    phase: str,
    executables: Mapping[str, Any],
) -> dict[str, Any]:
    environment = _test_environment_mapping()
    preloader_sha256 = "4" * 64
    repository, source = _test_operator_repository_and_source(
        immutable_plan=immutable_plan,
        phase=phase,
        preloader_sha256=preloader_sha256,
    )
    runtime_public, runtime_validation = _test_directory_receipt("1")
    site_public, site_validation = _test_directory_receipt("2")
    python_receipt_public, python_receipt_validation = _test_directory_receipt("3")
    site_receipt_public, site_receipt_validation = _test_directory_receipt("4")
    python_root = r"C:\fixture\python"
    site_root = r"C:\fixture\site"
    python_install = {
        "schema_version": 1,
        "kind": "explainiverse-operator-python-runtime-installed",
        "python_runtime_root": python_root,
        "archive_sha256": driver.OPERATOR_PYTHON_ARCHIVE_SHA256,
        "manifest_sha256": driver.OPERATOR_PYTHON_MANIFEST_SHA256,
        "file_count": 34,
        "directory_count": 0,
        "file_inventory_sha256": driver.OPERATOR_PYTHON_FILE_INVENTORY_SHA256,
        "owner_private_acl_applied_before_children": True,
        "site_processing_disabled_by_embeddable_pth": True,
        "untracked_files_or_directories_present": False,
        "crash_recovery": "discard-partial-directory-and-create-a-new-path",
    }
    site_install = {
        "schema_version": 1,
        "kind": "explainiverse-operator-runtime-installed",
        "runtime_root": site_root,
        "manifest_sha256": driver.OPERATOR_SITE_MANIFEST_SHA256,
        "file_count": 756,
        "directory_count": 113,
        "file_inventory_sha256": driver.OPERATOR_SITE_FILE_INVENTORY_SHA256,
        "owner_private_acl_applied_before_children": True,
        "pip_present_in_runtime": False,
        "record_files_present": False,
        "generated_scripts_present": False,
        "bytecode_present": False,
        "crash_recovery": "discard-partial-directory-and-create-a-new-path",
    }
    bootstrap = {
        "schema_version": 1,
        "kind": "explainiverse-operator-pre-site-bootstrap",
        "python_manifest_sha256": driver.OPERATOR_PYTHON_MANIFEST_SHA256,
        "python_archive_sha256": driver.OPERATOR_PYTHON_ARCHIVE_SHA256,
        "python_tree": {
            "python_root": python_root,
            "file_count": 34,
            "directory_count": 0,
            "file_inventory_sha256": driver.OPERATOR_PYTHON_FILE_INVENTORY_SHA256,
            "official_archive_sha256": driver.OPERATOR_PYTHON_ARCHIVE_SHA256,
            "untracked_files_or_directories_present": False,
            "all_runtime_bytes_match_official_archive": True,
        },
        "manifest_sha256": driver.OPERATOR_SITE_MANIFEST_SHA256,
        "archive_set_sha256": "5" * 64,
        "runtime_requirements_sha256": "6" * 64,
        "bootstrap_requirements_sha256": "7" * 64,
        "base_python_executable": executables["python"]["absolute_path"],
        "base_python_executable_sha256": executables["python"]["sha256"],
        "preactivation": {
            "working_directory": str(working_directory),
            "sys_path_sha256": "8" * 64,
            "only_base_stdlib_roots": True,
        },
        "site_tree": {
            "site_root": site_root,
            "file_count": 756,
            "directory_count": 113,
            "file_inventory_sha256": driver.OPERATOR_SITE_FILE_INVENTORY_SHA256,
            "untracked_files_or_directories_present": False,
            "bytecode_present": False,
            "all_importable_bytes_match_verified_wheels": True,
        },
        "activation_paths": [
            site_root,
            site_root + r"\win32",
            site_root + r"\win32\lib",
            site_root + r"\pythonwin",
        ],
        "site_processing_disabled": True,
        "pth_executed_by_cpython": False,
        "verified_pywin32_bootstrap_imported_after_verification": True,
    }
    public_acl = {
        "python_root": runtime_public["acl"],
        "site_root": site_public["acl"],
        "python_receipt_root": python_receipt_public["acl"],
        "site_receipt_root": site_receipt_public["acl"],
    }
    early_acl = {
        name: {
            "owner_sid": value["owner_sid"],
            "inheritance_protected": True,
            "child_inheritance_enabled": True,
            "allowed_sids": sorted([value["owner_sid"], "S-1-5-18", "S-1-5-32-544"]),
            "ace_count": 3,
            "rights": "full-control",
            "security_descriptor_sha256": value["security_descriptor_sha256"],
            "security_descriptor_bytes": value["security_descriptor_bytes"],
            "validated_before_third_party_site_or_native_import": True,
            "pinned_stdlib_native_modules_loaded_before_hold": True,
        }
        for name, value in public_acl.items()
    }
    early_material = {
        "schema_version": 1,
        "kind": "explainiverse-operator-early-runtime-boundary",
        "acl": early_acl,
        "held_trees": {
            "root_count": 4,
            "held_handle_count": 909,
            "write_share_allowed": False,
            "delete_share_allowed": False,
            "read_share_allowed": True,
            "held_before_third_party_site_or_native_import": True,
        },
        "all_runtime_and_receipt_roots_owner_private": True,
        "all_runtime_and_receipt_paths_held_without_write_or_delete_share": True,
        "validated_before_third_party_site_or_native_import": True,
        "pinned_official_python_runtime_is_the_pre_hold_trust_boundary": True,
        "working_directory": str(working_directory),
        "working_directory_repository_disjoint": True,
    }
    early = {
        **early_material,
        "evidence_sha256": hashlib.sha256(_canonical(early_material)).hexdigest(),
    }
    preloader_material = {
        "schema_version": 1,
        "kind": "explainiverse-operator-isolated-preloader",
        "shim": {
            "schema_version": 1,
            "kind": "explainiverse-operator-preloader-shim",
            "preloader_path": source["preloader_path"],
            "preloader_bytes": 4096,
            "preloader_sha256": preloader_sha256,
            "shim_sha256": driver.OPERATOR_PRELOADER_SHIM_SHA256,
            "stable_descriptor_read": True,
            "compiled_verified_bytes_without_reopen": True,
        },
        "source": source,
        "bootstrap": bootstrap,
        "python_runtime_directory_receipt": runtime_public,
        "python_runtime_validation": runtime_validation,
        "runtime_site_directory_receipt": site_public,
        "runtime_site_validation": site_validation,
        "python_install_receipt": python_install,
        "python_install_receipt_sha256": hashlib.sha256(_canonical(python_install)).hexdigest(),
        "python_install_directory_receipt": python_receipt_public,
        "python_install_directory_validation": python_receipt_validation,
        "site_install_receipt": site_install,
        "site_install_receipt_sha256": hashlib.sha256(_canonical(site_install)).hexdigest(),
        "site_install_directory_receipt": site_receipt_public,
        "site_install_directory_validation": site_receipt_validation,
        "environment": environment,
        "early_runtime_boundary": early,
        "sealed_resources": {
            "schema_version": 1,
            "kind": "explainiverse-operator-sealed-resource-binding",
            "policy_sha256": TEST_RESOURCES.policy_sha256,
            "controller_source_sha256": TEST_RESOURCES.controller_source_sha256,
            "runtime_bundle_sha256": immutable_plan["remote_runtime"]["bundle_sha256"],
            "runtime_file_sha256": {
                name: hashlib.sha256(name.encode("ascii")).hexdigest()
                for name in live.RUNTIME_BUNDLE_NAMES
            },
            "captured_before_project_import": True,
            "live_repository_reopen_permitted": False,
        },
        "working_directory": str(working_directory),
        "working_directory_is_python_install_receipt_directory": True,
        "isolated": True,
        "safe_path": True,
        "site_disabled": True,
        "bytecode_disabled": True,
        "repository_absent_from_sys_path": True,
        "project_imports_from_captured_bytes": True,
    }
    preloader = {
        **preloader_material,
        "evidence_sha256": hashlib.sha256(_canonical(preloader_material)).hexdigest(),
    }
    return {
        "schema_version": 1,
        "kind": "operator-secure-interpreter-launch",
        "isolated": True,
        "safe_path": True,
        "ignore_environment": True,
        "no_user_site": True,
        "no_site": True,
        "dont_write_bytecode": True,
        "invocation": "pinned-python -I -S -B -c <byte-sealing-shim>",
        "working_directory": str(working_directory),
        "repository_absent_from_sys_path": True,
        "sys_path_sha256": "f" * 64,
        "site_processing_disabled": True,
        "preloader": preloader,
        "controller_imported_after_launch_validation": True,
        "windows_handle_transport": True,
        "inherited_handle_count": 2,
        "handles_distinct": True,
        "file_type_pipe": True,
        "child_handles_made_noninheritable": True,
        "raw_handle_values_archived": False,
        "secret_values_in_argv": False,
        "secret_values_in_environment": False,
        "windows_launcher_parent_declaration": {
            "receipt_sha256": "0" * 64,
            "preloader_metadata_matched": True,
            "parent_provenance_authenticated": False,
            "security_authority_derived_from_declaration": False,
            "child_revalidated_handle_transport_and_sealed_resources": True,
        },
        "_test_repository": repository,
    }


def _test_readiness_mapping(
    *,
    control_sha256: str,
    immutable_plan: Mapping[str, Any],
    binding_mappings: Mapping[str, Mapping[str, Any]],
    known_hosts_sha256: str,
    cloud_provider_seed: int,
    preflight_provider_seed: int,
) -> dict[str, Any]:
    cloud = {
        "schema_version": 1,
        "kind": "explainiverse-lambda-live-cloud-init-wait-binding",
        "plan_sha256": control_sha256,
        "provider_snapshot_sha256": _provider_snapshot_payload(
            control_sha256, "instance_bound", seed=cloud_provider_seed
        )["snapshot_sha256"],
        "provider_receipt_nonce": f"{cloud_provider_seed:032x}"[-32:],
        "instance_id": "instance-0123456789abcdef",
        "instance_public_ipv4": "8.8.8.8",
        "host_fingerprint": immutable_plan["ssh_access"]["ephemeral_host_key_fingerprint"],
        "known_hosts_sha256": known_hosts_sha256,
        "observed_at": controller._iso(NOW - timedelta(minutes=8)),
        "exit_code": 0,
        "stdout_sha256": "4" * 64,
        "stderr_sha256": "5" * 64,
        "fixed_command": immutable_plan["remote_runtime"]["fixed_cloud_init_wait_command"],
        "credential_received": False,
        "jit_config_received": False,
    }
    cloud["binding_sha256"] = hashlib.sha256(_canonical(cloud)).hexdigest()
    preflight = {
        "schema_version": 1,
        "kind": "explainiverse-lambda-live-host-preflight-binding",
        "plan_sha256": control_sha256,
        "provider_snapshot_sha256": _provider_snapshot_payload(
            control_sha256, "instance_bound", seed=preflight_provider_seed
        )["snapshot_sha256"],
        "provider_receipt_nonce": f"{preflight_provider_seed:032x}"[-32:],
        "instance_id": "instance-0123456789abcdef",
        "instance_public_ipv4": "8.8.8.8",
        "host_fingerprint": immutable_plan["ssh_access"]["ephemeral_host_key_fingerprint"],
        "known_hosts_sha256": known_hosts_sha256,
        "cloud_init_wait_binding_sha256": cloud["binding_sha256"],
        "remote_response_sha256": "6" * 64,
        "observed_at": controller._iso(NOW - timedelta(minutes=7)),
        "runtime_bundle_sha256": immutable_plan["remote_runtime"]["bundle_sha256"],
        "host_physical_gpu_count": 8,
        "host_physical_gpu_uuids": list(GPU_UUIDS),
        "host_physical_gpu_products": ["NVIDIA A100-SXM4-80GB"] * 8,
        "image_probe_sha256": "7" * 64,
        "gpu_injection": {
            "verified": True,
            "gpu_count": 8,
            "gpu_product": "NVIDIA A100-SXM4-80GB",
            "output_sha256": "8" * 64,
            "physical_gpu_uuids": list(GPU_UUIDS),
            "device_request_sha256": "9" * 64,
            "network_mode": "none",
            "published_ports": False,
        },
        "fixed_preflight_command": immutable_plan["remote_runtime"]["fixed_preflight_command"],
        "jit_config_received": False,
        "github_api_credential_received": False,
        "accepted_actions_evidence": False,
    }
    ssh_attempts = {
        "cloud_init": [
            {
                "attempt": 1,
                "stdout_sha256": cloud["stdout_sha256"],
                "stderr_sha256": cloud["stderr_sha256"],
                "exit_code": 0,
                "provider_snapshot_sha256": cloud["provider_snapshot_sha256"],
                "accepted": True,
            }
        ],
        "preflight": [
            {
                "attempt": 1,
                "stdout_sha256": preflight["remote_response_sha256"],
                "stderr_sha256": hashlib.sha256(b"").hexdigest(),
                "exit_code": 0,
                "provider_snapshot_sha256": preflight["provider_snapshot_sha256"],
                "accepted": True,
            }
        ],
    }
    material = {
        "schema_version": 1,
        "kind": "explainiverse-lambda-host-readiness-binding",
        "control_plane_plan_sha256": control_sha256,
        "instance_id": cloud["instance_id"],
        "instance_public_ipv4": cloud["instance_public_ipv4"],
        "host_fingerprint": cloud["host_fingerprint"],
        "known_hosts_sha256": cloud["known_hosts_sha256"],
        "cloud_init": cloud,
        "preflight": preflight,
        "cloud_init_sha256": hashlib.sha256(_canonical(cloud)).hexdigest(),
        "preflight_sha256": hashlib.sha256(_canonical(preflight)).hexdigest(),
        "cloud_binding": dict(binding_mappings["cloud-init"]),
        "preflight_binding": dict(binding_mappings["preflight"]),
        "ssh_attempts": ssh_attempts,
    }
    return {
        "cloud_init": cloud,
        "host_preflight": preflight,
        "cloud_init_sha256": material["cloud_init_sha256"],
        "preflight_sha256": material["preflight_sha256"],
        "cloud_binding": material["cloud_binding"],
        "preflight_binding": material["preflight_binding"],
        "ssh_attempts": ssh_attempts,
        "readiness_evidence_sha256": hashlib.sha256(_canonical(material)).hexdigest(),
    }


def _record_successful_lifecycle_prefix(
    journal: driver.EvidenceJournal,
    *,
    control_sha256: str,
    phase: str,
    immutable_plan: Mapping[str, Any],
) -> str:
    callback_binding_sha256 = hashlib.sha256(b"provider-intent-sink").hexdigest()
    journal.record(
        "provider-intent-sink-bound",
        {
            "plan_sha256": control_sha256,
            "binding_sha256": callback_binding_sha256,
            "sink": "evidence-journal",
            "bound_before_observation": True,
            "recovery_process": False,
        },
    )
    preflight_document, preflight_expected = operator_preflight_fixture(
        phase,
        policy_sha256=TEST_RESOURCES.policy_sha256,
        controller_source_sha256=TEST_RESOURCES.controller_source_sha256,
        runtime_bundle_sha256=str(immutable_plan["remote_runtime"]["bundle_sha256"]),
    )
    assert preflight_expected["expected_immutable_plan"] == immutable_plan
    assert preflight_expected["expected_plan_sha256"] == control_sha256
    operator_executables = preflight_document["executables"]
    executable = {
        "absolute_path": operator_executables["ssh"]["absolute_path"],
        "sha256": operator_executables["ssh"]["sha256"],
        "regular_file": True,
        "symlink": False,
        "path_lookup_used": False,
    }
    journal.record("ssh-executable", executable)
    journal.record(
        "github-executable",
        {
            **executable,
            "absolute_path": operator_executables["gh"]["absolute_path"],
            "sha256": operator_executables["gh"]["sha256"],
            "hostname_pinned": "github.com",
            "child_environment_names": ["APPDATA", "NO_COLOR", "SYSTEMROOT"],
            "ambient_token_environment_forwarded": False,
        },
    )
    owner_sid = "S-1-5-21-1000"
    acl_material = {
        "owner_sid": owner_sid,
        "current_user_sid": owner_sid,
        "inheritance_protected": True,
        "aces": sorted(
            [
                {
                    "sid": sid,
                    "access": "allow",
                    "rights": "full-control",
                    "mask": 2_032_127,
                    "ace_flags": 0,
                }
                for sid in (owner_sid, "S-1-5-18", "S-1-5-32-544")
            ],
            key=lambda item: item["sid"],
        ),
        "security_descriptor_sha256": "d" * 64,
        "security_descriptor_bytes": 128,
    }
    acl = {
        **acl_material,
        "captured_at": (NOW - timedelta(minutes=10)).isoformat(),
        "evidence_sha256": hashlib.sha256(_canonical(acl_material)).hexdigest(),
    }
    access_capture = {
        "captured_at": acl["captured_at"],
        "public_key_sha256": immutable_plan["ssh_access"]["public_key_sha256"],
        "public_key_fingerprint": "SHA256:" + "B" * 43,
        "key_type": "ssh-ed25519",
        "private_file_bytes": 411,
        "private_digest_recorded": True,
        "absolute_path_redacted": True,
        "file_identity_recorded": True,
        "single_link": True,
        "no_reparse_or_symlink": True,
        "acl": acl,
    }
    journal.record(
        "ssh-access-identity",
        {
            "capture": access_capture,
            "validation": {
                "validated_at": (NOW - timedelta(minutes=9)).isoformat(),
                "public_key_sha256": access_capture["public_key_sha256"],
                "public_key_fingerprint": access_capture["public_key_fingerprint"],
                "private_file_bytes": access_capture["private_file_bytes"],
                "private_digest_recorded": True,
                "absolute_path_redacted": True,
                "file_identity_recorded": True,
                "single_link": True,
                "no_reparse_or_symlink": True,
                "acl_evidence_sha256": acl["evidence_sha256"],
            },
            "private_path_archived": False,
            "private_digest_archived": False,
        },
    )
    current = _provider_snapshot_payload(control_sha256, "baseline", seed=1)
    journal.record("provider-baseline", current)
    for seed, (operation, next_phase) in enumerate(
        (
            ("restrict_global", "global_restricted"),
            ("create_ruleset", "ruleset_ready"),
            ("launch", "instance_bound"),
        ),
        start=10,
    ):
        current = _record_provider_transition(
            journal,
            control_sha256=control_sha256,
            callback_binding_sha256=callback_binding_sha256,
            operation=operation,
            prestate=current,
            next_phase=next_phase,
            seed=seed,
        )
    evidence_receipt = journal.evidence_directory_receipt
    root = Path(evidence_receipt.absolute_path)
    known_hosts_path = root / "known_hosts"
    known_hosts_content = b"8.8.8.8 ssh-ed25519 AAAA\n"
    known_hosts_path.write_bytes(known_hosts_content)
    known_hosts_sha256 = hashlib.sha256(known_hosts_content).hexdigest()
    known_hosts = {
        "absolute_path": str(known_hosts_path),
        "content_sha256": known_hosts_sha256,
        "evidence_directory_acl_receipt_sha256": evidence_receipt.receipt_sha256,
        "public_ipv4": "8.8.8.8",
        "host_fingerprint": immutable_plan["ssh_access"]["ephemeral_host_key_fingerprint"],
        "content_is_public": True,
    }
    journal.record("known-hosts", known_hosts)
    binding_mappings: dict[str, dict[str, Any]] = {}
    command_fields = {
        "cloud-init": "fixed_cloud_init_wait_command",
        "preflight": "fixed_preflight_command",
        "run": "fixed_command",
    }
    for mode, command_field in command_fields.items():
        command = immutable_plan["remote_runtime"][command_field]
        argv = [
            "ssh",
            "-T",
            "-F",
            os.devnull,
            "-i",
            "<redacted-existing-access-identity-file>",
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
            f"UserKnownHostsFile={known_hosts_path}",
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
            "ubuntu@8.8.8.8",
            *command,
        ]
        binding_mappings[mode] = {
            "argv_prefix": argv,
            "access_identity_file_redacted": True,
            "known_hosts": known_hosts_content.decode("ascii"),
            "known_hosts_path": str(known_hosts_path),
            "known_hosts_sha256": known_hosts_sha256,
            "evidence_directory_acl_receipt_sha256": evidence_receipt.receipt_sha256,
            "host_fingerprint": known_hosts["host_fingerprint"],
            "trust_on_first_use": False,
            "remote_mode": mode,
            "fixed_remote_command": command,
        }
        journal.record(f"ssh-{mode}", binding_mappings[mode])
    for seed in range(30, 34):
        journal.record(
            "provider-instance-bound",
            _provider_snapshot_payload(control_sha256, "instance_bound", seed=seed),
        )
    journal.record(
        "host-readiness",
        _test_readiness_mapping(
            control_sha256=control_sha256,
            immutable_plan=immutable_plan,
            binding_mappings=binding_mappings,
            known_hosts_sha256=known_hosts_sha256,
            cloud_provider_seed=31,
            preflight_provider_seed=33,
        ),
    )
    app_inbox = preflight_document["app_capture_inbox"]
    inventory_sha256 = preflight_document["inventory_sha256"]
    preflight_sha256 = hashlib.sha256(_canonical(preflight_document)).hexdigest()
    preflight_name = f"operator-preflight-{preflight_sha256}.json"
    live.write_public_evidence(
        root / preflight_name,
        preflight_document,
        evidence_directory_receipt=evidence_receipt,
    )
    journal.record(
        "operator-preflight-binding",
        {
            "plan_sha256": control_sha256,
            "operator_preflight_filename": preflight_name,
            "operator_preflight_sha256": preflight_sha256,
            "inspection_receipt_sha256": "d" * 64,
            "inventory_sha256": inventory_sha256,
            "app_capture_inbox": app_inbox,
            "bound_before_first_jit": True,
        },
    )
    return callback_binding_sha256


def _record_host_refresh(
    journal: driver.EvidenceJournal,
    *,
    control_sha256: str,
    ordinal: int,
    immutable_plan: Mapping[str, Any],
) -> None:
    first_seed = 40 + ordinal * 3
    for offset in range(2):
        journal.record(
            "provider-instance-bound",
            _provider_snapshot_payload(
                control_sha256,
                "instance_bound",
                seed=first_seed + offset,
            ),
        )
    setup = dict(journal.verified_entries())
    journal.record(
        "host-preflight-refresh",
        _test_readiness_mapping(
            control_sha256=control_sha256,
            immutable_plan=immutable_plan,
            binding_mappings={
                "cloud-init": setup["ssh-cloud-init"],
                "preflight": setup["ssh-preflight"],
            },
            known_hosts_sha256=setup["known-hosts"]["content_sha256"],
            cloud_provider_seed=31,
            preflight_provider_seed=first_seed + 1,
        ),
    )


def _record_job_mutation_wal(
    journal: driver.EvidenceJournal,
    *,
    session: controller.PhaseSession,
    binding: controller.JobBinding,
    runtime_plan: Mapping[str, Any],
) -> None:
    path = f"/repos/{controller.REPOSITORY}/actions/runners/generate-jitconfig"
    body = {
        "name": binding.runner_name,
        "runner_group_id": controller.RUNNER_GROUP_ID,
        "labels": [binding.runner_name],
        "work_folder": f"_work-{binding.nonce}",
    }
    plan_absence = runtime_plan["github_evidence"]["pre_jit_registration_absence"]
    absence_material = {
        "phase": session.phase,
        "run_id": session.run["id"],
        "run_attempt": 1,
        "head_sha": session.head_sha,
        "job_key": binding.key,
        "job_id": binding.job_id,
        "runner_name": binding.runner_name,
        "observed_at": plan_absence["observed_at"],
        "response_sha256": plan_absence["response_sha256"],
        "total_count": 0,
        "runners": [],
    }
    journal.record(
        "github-pre-jit-runner-absence",
        {
            **absence_material,
            "evidence_sha256": hashlib.sha256(_canonical(absence_material)).hexdigest(),
        },
    )
    journal.record(
        "github-jit-intent",
        {
            "phase": session.phase,
            "run_id": session.run["id"],
            "head_sha": session.head_sha,
            "job_key": binding.key,
            "job_id": binding.job_id,
            "job_name": binding.name,
            "runner_name": binding.runner_name,
            "runner_nonce": binding.nonce,
            "path": path,
            "request_sha256": hashlib.sha256(
                _canonical({"method": "POST", "path": path, "body": body})
            ).hexdigest(),
            "runner_group_id": controller.RUNNER_GROUP_ID,
            "mutation_retried": False,
        },
    )
    plan_jit = runtime_plan["github_evidence"]["jit_response"]
    jit_receipt = {
        "observed_at": plan_jit["observed_at"],
        "request_sha256": hashlib.sha256(_canonical(body)).hexdigest(),
        "response_sha256": plan_jit["response_sha256"],
        "response_body_sha256": hashlib.sha256(
            f"jit-body:{binding.key}".encode("ascii")
        ).hexdigest(),
        "runner": plan_jit["runner"],
        "jit_config_sha256": runtime_plan["job"]["jit_config_sha256"],
        "encoded_jit_config_persisted": False,
        "runner_group_id": controller.RUNNER_GROUP_ID,
        "runner_group_get_performed": False,
    }
    journal.record(
        "github-jit-created",
        {
            "phase": session.phase,
            "run_id": session.run["id"],
            "head_sha": session.head_sha,
            "job_key": binding.key,
            "job_id": binding.job_id,
            "runner_name": binding.runner_name,
            "jit_receipt": jit_receipt,
        },
    )
    journal.record("runtime-plan", runtime_plan)
    journal.record(
        "remote-start-intent",
        {
            "phase": session.phase,
            "run_id": session.run["id"],
            "head_sha": session.head_sha,
            "job_key": binding.key,
            "job_id": binding.job_id,
            "runner_id": runtime_plan["job"]["runner_id"],
            "runner_name": binding.runner_name,
            "runtime_plan_sha256": runtime.runtime_plan_sha256(runtime_plan),
            "remote_start_retried": False,
        },
    )


def _record_successful_lifecycle_teardown(
    journal: driver.EvidenceJournal,
    *,
    phase: str,
    control_sha256: str,
    callback_binding_sha256: str,
    known_hosts_sha256: str,
    lifecycle_overrides: Mapping[str, Any] | None = None,
) -> str:
    immutable_plan = next(
        payload for label, payload in journal.verified_entries() if label == "immutable-plan"
    )
    operator_inbox = next(
        payload["app_capture_inbox"]
        for label, payload in journal.verified_entries()
        if label == "operator-preflight-binding"
    )
    accepted_captures = [
        payload
        for label, payload in journal.verified_entries()
        if label == "installed-app-authority"
    ]
    consumed_generations: list[dict[str, Any]] = []
    final_files: list[dict[str, Any]] = []
    final_directories: list[str] = []
    for ordinal, app_capture in enumerate(accepted_captures, start=1):
        normalized = app_capture["normalized_capture"]
        capture_raw = _canonical(normalized)
        bundle = f"capture-{ordinal:02d}-000001"
        ready = f"ready-{ordinal:02d}-000001.json"
        pages = [
            {
                "filename": item["filename"],
                "bytes": item["bytes"],
                "sha256": item["sha256"],
            }
            for item in normalized["evidence"]
        ]
        ready_raw = _canonical(
            {
                "schema_version": 1,
                "kind": "explainiverse-installed-app-capture-ready",
                "phase": phase,
                "ordinal": ordinal,
                "generation": 1,
                "publication_nonce": f"{ordinal:032x}",
                "capture_directory": bundle,
                "capture_json_sha256": hashlib.sha256(capture_raw).hexdigest(),
                "pages_inventory_sha256": hashlib.sha256(_canonical(pages)).hexdigest(),
            }
        )
        consumed_generations.append(
            {
                "ordinal": ordinal,
                "generation": 1,
                "publication_nonce": f"{ordinal:032x}",
                "ready_marker": ready,
                "ready_marker_bytes": len(ready_raw),
                "ready_marker_sha256": hashlib.sha256(ready_raw).hexdigest(),
                "capture_directory": bundle,
                "capture_json_bytes": len(capture_raw),
                "capture_json_sha256": hashlib.sha256(capture_raw).hexdigest(),
                "capture": normalized,
                "classified_at": normalized["captured_at"],
                "pages": pages,
                "pages_inventory_sha256": hashlib.sha256(_canonical(pages)).hexdigest(),
                "classification": "accepted",
                "capture_evidence_sha256": app_capture["evidence_sha256"],
            }
        )
        final_directories.extend((bundle, f"{bundle}/pages"))
        final_files.append(
            {
                "path": f"{bundle}/capture.json",
                "bytes": len(capture_raw),
                "sha256": hashlib.sha256(capture_raw).hexdigest(),
            }
        )
        final_files.extend(
            {
                "path": f"{bundle}/pages/{item['filename']}",
                "bytes": item["bytes"],
                "sha256": item["sha256"],
            }
            for item in pages
        )
        final_files.append(
            {
                "path": ready,
                "bytes": len(ready_raw),
                "sha256": hashlib.sha256(ready_raw).hexdigest(),
            }
        )
    final_files.sort(key=lambda item: item["path"])
    final_directories.sort()
    final_inventory_material = {
        "schema_version": 1,
        "kind": "explainiverse-installed-app-inbox-final-inventory",
        "phase": phase,
        "accepted_generation_count": len(accepted_captures),
        "stale_generation_count": 0,
        "generation_count": len(accepted_captures),
        "consumed_generations": consumed_generations,
        "files": final_files,
        "directories": final_directories,
        "file_count": len(final_files),
        "directory_count": len(final_directories),
        "owner_private_directory_receipt_sha256": operator_inbox[
            "owner_private_directory_receipt_sha256"
        ],
        "accepted_source_generations_retained": True,
        "unobserved_residue_present": False,
    }
    final_inventory = {
        **final_inventory_material,
        "evidence_sha256": hashlib.sha256(_canonical(final_inventory_material)).hexdigest(),
    }
    journal.record(
        "operator-app-inbox-settlement",
        {
            **operator_inbox,
            "accepted_capture_count": (4 if phase == "final-main" else 2),
            "all_expected_captures_consumed": True,
            "capture_bytes_retained_only_as_driver_archive": False,
            "all_consumed_raw_pages_archived_in_evidence_root": True,
            "accepted_source_generations_retained_in_owner_private_inbox": True,
            "final_inbox_inventory": final_inventory,
        },
    )
    journal.record(
        "ssh-access-identity-closed",
        {
            "public_key_sha256": immutable_plan["ssh_access"]["public_key_sha256"],
            "closed": True,
            "private_path_archived": False,
            "private_digest_archived": False,
        },
    )
    current = _provider_snapshot_payload(control_sha256, "instance_bound", seed=70)
    journal.record("provider-instance-bound", current)
    current = _record_provider_transition(
        journal,
        control_sha256=control_sha256,
        callback_binding_sha256=callback_binding_sha256,
        operation="terminate",
        prestate=current,
        next_phase="instance_absent",
        seed=71,
    )
    zero_material = {
        "schema_version": 1,
        "kind": "explainiverse-repository-runner-zero-inventory",
        "repository": controller.REPOSITORY,
        "observed_at": controller._iso(NOW),
        "runner_count": 0,
        "response_sha256": "8" * 64,
        "observation_count": 3,
        "observation_response_sha256": ["8" * 64, "9" * 64, "a" * 64],
    }
    journal.record(
        "github-zero-runners-before-abort",
        {
            **zero_material,
            "evidence_sha256": hashlib.sha256(_canonical(zero_material)).hexdigest(),
        },
    )
    current = _record_provider_transition(
        journal,
        control_sha256=control_sha256,
        callback_binding_sha256=callback_binding_sha256,
        operation="delete_ruleset",
        prestate=current,
        next_phase="ruleset_absent",
        seed=73,
    )
    _record_provider_transition(
        journal,
        control_sha256=control_sha256,
        callback_binding_sha256=callback_binding_sha256,
        operation="restore_global",
        prestate=current,
        next_phase="restored",
        seed=75,
    )
    lifecycle = {
        "plan_sha256": control_sha256,
        "provider_instances": 0,
        "provider_firewall_rulesets": 0,
        "global_firewall_restored": True,
        "repository_runners": 0,
        "known_hosts_sha256": known_hosts_sha256,
    }
    lifecycle.update(lifecycle_overrides or {})
    return journal.record("lifecycle-restored", lifecycle)


def _completed_final_acceptance(
    session: controller.PhaseSession | None = None,
) -> tuple[
    controller.PhaseSession,
    Mapping[str, Any],
    controller.FinalMainAcceptance,
]:
    current = session or _session()
    for ordinal, binding in enumerate(current.jobs, start=1):

        def digest(label: str) -> str:
            return hashlib.sha256(f"{ordinal}:{label}".encode("ascii")).hexdigest()

        material = {
            "phase": "final-main",
            "run_id": current.run["id"],
            "job_key": binding.key,
            "job_id": binding.job_id,
            "runner_id": 200 + ordinal,
            "runner_name": binding.runner_name,
            "runtime_plan_sha256": digest("runtime-plan"),
            "remote_receipt_sha256": digest("remote-receipt"),
            "actions_job_response_sha256": digest("actions-job"),
            "check_response_sha256": digest("check"),
            "log_sha256": digest("log"),
            "pytest_passed": 15,
            "pytest_skipped": 0,
            "runner_inventory_response_sha256": digest("runner-inventory"),
            "post_execution_observation_sha256": digest("post-observation"),
        }
        current.accepted[binding.key] = controller.AcceptedJobReceipt(
            **material,
            evidence_sha256=hashlib.sha256(_canonical(material)).hexdigest(),
        )
    settlement_material = {
        "phase": "final-main",
        "run_id": current.run["id"],
        "run_attempt": 1,
        "head_sha": current.head_sha,
        "accepted_cuda_runner_nonces": [item.nonce for item in current.jobs],
        "job_evidence_sha256": [
            current.accepted[item.key].evidence_sha256 for item in current.jobs
        ],
        "all_four_jobs_15_of_15_zero_skips": True,
        "rerun_performed": False,
    }
    settlement = {
        **settlement_material,
        "evidence_sha256": hashlib.sha256(_canonical(settlement_material)).hexdigest(),
    }
    service = controller.ReleaseGpuController(
        QueueTransport(), NoRemote(), resources=TEST_RESOURCES, clock=lambda: NOW
    )
    acceptance = service.seal_final_main_acceptance(current, settlement)
    return current, settlement, acceptance


def _readiness() -> controller.HostReadinessReceipt:
    cloud = live.CloudInitWaitReceipt(
        plan_sha256="1" * 64,
        provider_snapshot_sha256="2" * 64,
        provider_receipt_nonce="0" * 32,
        instance_id="instance-0123456789abcdef",
        instance_public_ipv4="8.8.8.8",
        host_fingerprint="SHA256:" + "A" * 43,
        known_hosts_sha256="3" * 64,
        observed_at=controller._iso(NOW - timedelta(minutes=3)),
        stdout_sha256="4" * 64,
        stderr_sha256="5" * 64,
        binding_sha256="6" * 64,
    )
    preflight = live.HostPreflightReceipt(
        plan_sha256="1" * 64,
        provider_snapshot_sha256="2" * 64,
        provider_receipt_nonce="1" * 32,
        instance_id="instance-0123456789abcdef",
        instance_public_ipv4="8.8.8.8",
        host_fingerprint="SHA256:" + "A" * 43,
        known_hosts_sha256="3" * 64,
        cloud_init_wait_binding_sha256="6" * 64,
        remote_response_sha256="7" * 64,
        observed_at=controller._iso(NOW - timedelta(minutes=2)),
        runtime_bundle_sha256="8" * 64,
        host_physical_gpu_uuids=GPU_UUIDS,
        host_physical_gpu_products=("NVIDIA A100-SXM4-80GB",) * 8,
        image_probe_sha256="9" * 64,
        gpu_injection_output_sha256="a" * 64,
        gpu_injection_device_request_sha256="b" * 64,
    )
    readiness = object.__new__(controller.HostReadinessReceipt)
    object.__setattr__(readiness, "cloud_init", cloud)
    object.__setattr__(readiness, "preflight", preflight)
    object.__setattr__(readiness, "cloud_init_sha256", "a" * 64)
    object.__setattr__(readiness, "preflight_sha256", "b" * 64)
    object.__setattr__(readiness, "cloud_binding", {})
    object.__setattr__(readiness, "preflight_binding", {})
    object.__setattr__(readiness, "ssh_attempts", {})
    object.__setattr__(readiness, "evidence_sha256", "c" * 64)
    return readiness


def _authority_receipt_for_capture(
    capture: controller.TrustedAppCapture,
    *,
    job_index: int,
    observed_at_override: datetime | None = None,
) -> controller.AuthorityReceipt:
    observed_at = controller._iso(
        observed_at_override or (NOW - timedelta(minutes=4) + timedelta(seconds=job_index))
    )

    def digest(label: str) -> str:
        return hashlib.sha256(f"authority:{job_index}:{label}".encode("ascii")).hexdigest()

    collaborators = [digest("collaborators-page")]
    invitations = [digest("invitations-page")]
    variables = [digest("variables-page")]
    runners = digest("runners")
    queue = {"response_sha256": digest("queue")}
    material = {
        "observed_at": observed_at,
        "app_capture": capture.to_mapping(),
        "collaborators": collaborators,
        "invitations": invitations,
        "runners": runners,
        "variables": variables,
        "queue": queue,
    }
    return controller.AuthorityReceipt(
        observed_at=observed_at,
        expires_at=controller._iso(
            controller._parse_time(observed_at, "test_authority_observed")
            + controller.AUTHORITY_WINDOW
        ),
        evidence_sha256=hashlib.sha256(_canonical(material)).hexdigest(),
        app_capture_sha256=capture.evidence_sha256,
        collaborators_response_sha256=hashlib.sha256(_canonical(collaborators)).hexdigest(),
        invitations_response_sha256=hashlib.sha256(_canonical(invitations)).hexdigest(),
        runners_response_sha256=runners,
        variables_response_sha256=hashlib.sha256(_canonical(variables)).hexdigest(),
        queue_evidence_sha256=queue["response_sha256"],
        _evidence_material=material,
    )


@pytest.mark.parametrize(
    "fault",
    (None, "max-age", "dispatch-capture", "capture-authority", "authority-created"),
)
def test_historical_job_authority_freshness_checks_reach_each_strict_boundary(
    fault: str | None,
) -> None:
    capture_at = NOW - timedelta(minutes=4, seconds=30)
    authority_at = NOW - timedelta(minutes=4)
    dispatch_at = NOW - timedelta(minutes=5)
    created_at = NOW
    if fault == "max-age":
        capture_at = NOW - timedelta(minutes=15)
        dispatch_at = NOW - timedelta(minutes=16)
    elif fault == "dispatch-capture":
        dispatch_at = capture_at
    elif fault == "capture-authority":
        authority_at = capture_at
    elif fault == "authority-created":
        created_at = authority_at

    capture_mapping, pages = _app_capture(captured_at=capture_at)
    capture = controller.TrustedAppCapture.from_mapping(
        capture_mapping,
        resources=TEST_RESOURCES,
        evidence_reader=pages.__getitem__,
        now=capture_at,
    )
    authority = _authority_receipt_for_capture(
        capture,
        job_index=0,
        observed_at_override=authority_at,
    )
    runtime_plan = _valid_plan_for(authority_receipt=authority)
    runtime_plan["dispatch"]["observed_at"] = controller._iso(dispatch_at)
    runtime_plan["created_at"] = controller._iso(created_at)
    kwargs = {
        "capture": capture,
        "app_payload": capture.to_mapping(),
        "archive_payload": {"archive_evidence_sha256": "a" * 64},
        "authority_payload": authority.evidence_mapping(),
        "runtime_plan": runtime_plan,
    }
    if fault is None:
        identity = driver.EvidenceJournal._validate_job_authority_evidence(**kwargs)
        assert identity["captured_at"] == capture.captured_at
        return
    with pytest.raises(
        controller.ControllerError,
        match="journal_authority_capture_freshness_rejected",
    ):
        driver.EvidenceJournal._validate_job_authority_evidence(**kwargs)


def _synthetic_final_authority_identities() -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    for ordinal, key in enumerate(
        ("single_minimum", "single_latest", "two_minimum", "two_latest"),
        start=1,
    ):
        dispatch_at = NOW - timedelta(minutes=8)
        captured_at = dispatch_at + timedelta(seconds=ordinal)
        authority_at = captured_at + timedelta(seconds=1)
        runtime_at = authority_at + timedelta(seconds=1)

        def digest(label: str) -> str:
            return hashlib.sha256(
                f"prior-final-authority:{ordinal}:{label}".encode("ascii")
            ).hexdigest()

        material = {
            "schema_version": 1,
            "kind": "explainiverse-jit-authority-evidence-identity",
            "phase": "final-main",
            "head_sha": HEAD,
            "run_id": 500,
            "job_key": key,
            "capture_evidence_sha256": digest("capture"),
            "authority_evidence_sha256": digest("authority"),
            "archive_evidence_sha256": digest("archive"),
            "raw_page_sha256": [digest("page")],
            "dispatch_observed_at": controller._iso(dispatch_at),
            "captured_at": controller._iso(captured_at),
            "authority_observed_at": controller._iso(authority_at),
            "runtime_created_at": controller._iso(runtime_at),
        }
        result.append(
            controller._validated_authority_evidence_identity(
                {
                    **material,
                    "evidence_sha256": hashlib.sha256(_canonical(material)).hexdigest(),
                },
                context="test_prior_final_authority_identity",
                expected_phase="final-main",
                expected_head_sha=HEAD,
                expected_run_id=500,
                expected_job_key=key,
            )
        )
    return tuple(result)


def _valid_plan_for(
    *,
    control_plane_plan_sha256: str = CONTROL_SHA,
    job_index: int = 0,
    previous_cleanup_receipt_sha256: str | None = None,
    session: controller.PhaseSession | None = None,
    authority_receipt: controller.AuthorityReceipt | None = None,
) -> dict[str, Any]:
    service = controller.ReleaseGpuController(
        QueueTransport(), NoRemote(), resources=TEST_RESOURCES, clock=lambda: NOW
    )
    current_session = session or _session()
    job = current_session.jobs[job_index]
    if job_index and previous_cleanup_receipt_sha256 is None:
        previous_cleanup_receipt_sha256 = hashlib.sha256(
            f"previous-cleanup:{job.key}".encode("ascii")
        ).hexdigest()
    authority = authority_receipt or controller.AuthorityReceipt(
        observed_at=controller._iso(NOW - timedelta(minutes=4)),
        expires_at=controller._iso(NOW + timedelta(minutes=10)),
        evidence_sha256="d" * 64,
        app_capture_sha256="e" * 64,
        collaborators_response_sha256="f" * 64,
        invitations_response_sha256="0" * 64,
        runners_response_sha256="1" * 64,
        variables_response_sha256="2" * 64,
        queue_evidence_sha256="3" * 64,
    )
    downloads = {
        "observed_at": controller._iso(NOW - timedelta(minutes=2)),
        "response_sha256": "4" * 64,
        "os": "linux",
        "architecture": "x64",
        "filename": runtime.RUNNER_FILENAME,
        "download_url": runtime.RUNNER_DOWNLOAD_URL,
        "sha256_checksum": runtime.RUNNER_ARCHIVE_SHA256,
        "api_version": runtime.GITHUB_API_VERSION,
        "version": runtime.RUNNER_VERSION,
    }
    history = {
        "observed_at": controller._iso(NOW - timedelta(minutes=3)),
        "response_sha256": hashlib.sha256(f"history:{job.key}".encode("ascii")).hexdigest(),
        "historical_match_count": 0,
        "unexpected_queued_or_in_progress_count": 0,
    }
    jit = {
        "observed_at": controller._iso(NOW - timedelta(minutes=1)),
        "response_sha256": hashlib.sha256(f"jit-response:{job.key}".encode("ascii")).hexdigest(),
        "jit_config_sha256": hashlib.sha256(bytes([65 + job_index]) * 256).hexdigest(),
        "absence_observed_at": controller._iso(NOW - timedelta(minutes=2)),
        "runner": {
            "id": 41 + job_index,
            "name": job.runner_name,
            "os": "unknown",
            "status": "offline",
            "busy": False,
            "labels": [job.runner_name],
        },
    }
    plan, _ = service._build_plan(
        current_session,
        job,
        authority,
        _readiness(),
        downloads,
        history,
        hashlib.sha256(f"absence:{job.key}".encode("ascii")).hexdigest(),
        jit,
        control_plane_plan_sha256=control_plane_plan_sha256,
        previous_cleanup_receipt_sha256=previous_cleanup_receipt_sha256,
    )
    return plan


def _valid_plan() -> dict[str, Any]:
    return _valid_plan_for()


def _remote_execution(plan: Mapping[str, Any]) -> controller.RemoteExecution:
    receipt = runtime.build_runtime_receipt(
        plan,
        host_gpu_uuids=GPU_UUIDS,
        started_at=controller._iso(NOW + timedelta(seconds=1)),
        jit_config_sent_at=controller._iso(NOW + timedelta(seconds=2)),
        stopped_at=controller._iso(NOW + timedelta(seconds=3)),
        cleanup_verified_at=controller._iso(NOW + timedelta(seconds=4)),
        runner_exit_code=0,
    )
    frame = {
        "magic": "EXJIT01",
        "version": 1,
        "flags": 0,
        "header_bytes": 84,
        "plan_bytes": len(runtime.canonical_json(plan)),
        "jit_config_bytes": 256,
        "plan_sha256": runtime.runtime_plan_sha256(plan),
        "jit_config_sha256": plan["job"]["jit_config_sha256"],
        "header_sha256": "3" * 64,
        "trailing_bytes_permitted": False,
        "remote_argv_contains_plan_or_jit_values": False,
    }
    return controller.RemoteExecution(
        receipt, hashlib.sha256(runtime.canonical_json(receipt)).hexdigest(), "2" * 64, frame
    )


def test_gh_cli_is_shell_free_and_never_requests_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, Any] = {}
    executable = tmp_path / "gh.exe"
    executable.write_bytes(b"audited gh fixture")
    executable_sha256 = hashlib.sha256(executable.read_bytes()).hexdigest()
    monkeypatch.setenv("GH_HOST", "hostile.example")

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        observed.update({"argv": argv, **kwargs})
        return subprocess.CompletedProcess(
            argv, 0, b"HTTP/2 200 OK\r\ncontent-type: application/json\r\n\r\n{}\n", b""
        )

    transport = controller.GhCliTransport(
        executable_path=str(executable.resolve()),
        executable_sha256=executable_sha256,
        runner=run,
    )
    response = transport.request("GET", f"/repos/{controller.REPOSITORY}/actions/runners")
    assert response.status_code == 200
    assert observed["shell"] is False
    assert "auth" not in observed["argv"]
    assert "token" not in observed["argv"]
    assert observed["argv"][0] == str(executable.resolve())
    assert observed["argv"][2:4] == ["--hostname", "github.com"]
    assert "GH_HOST" not in observed["env"]
    assert f"X-GitHub-Api-Version: {runtime.GITHUB_API_VERSION}" in observed["argv"]


def test_remote_environment_strips_github_and_provider_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GH_TOKEN", "forbidden")
    monkeypatch.setenv("GITHUB_TOKEN", "forbidden")
    monkeypatch.setenv("LAMBDA_API_KEY", "forbidden")
    environment = controller.SshRemoteExecutor._environment()
    assert not {"GH_TOKEN", "GITHUB_TOKEN", "LAMBDA_API_KEY"}.intersection(environment)
    gh_environment = controller.GhCliTransport._environment()
    assert not {"GH_TOKEN", "GITHUB_TOKEN", "LAMBDA_API_KEY", "PATH"}.intersection(gh_environment)


def test_trusted_app_capture_binds_permissions_repository_and_raw_evidence() -> None:
    value, raw_pages = _app_capture()
    capture = controller.TrustedAppCapture.from_mapping(
        value, resources=TEST_RESOURCES, evidence_reader=raw_pages.__getitem__, now=NOW
    )
    assert len(capture.evidence_sha256) == 64
    assert all(item["repository_selection"] == "all" for item in capture.installations)
    socket = next(item for item in capture.installations if item["name"] == "Socket Security")
    assert socket["permissions"]["write"] == ["checks", "code", "pull requests"]

    dangerous, dangerous_pages = _app_capture()
    socket = next(item for item in dangerous["installations"] if item["name"] == "Socket Security")
    socket["permissions"]["write"].append("workflows")
    with pytest.raises(controller.ControllerError, match="installed_app_policy_drift"):
        controller.TrustedAppCapture.from_mapping(
            dangerous,
            resources=TEST_RESOURCES,
            evidence_reader=dangerous_pages.__getitem__,
            now=NOW,
        )

    tampered, tampered_pages = _app_capture()
    first_name = tampered["evidence"][0]["filename"]
    tampered_pages[first_name] += b"tampered\n"
    with pytest.raises(controller.ControllerError, match="app_capture_invalid_or_unbound"):
        controller.TrustedAppCapture.from_mapping(
            tampered,
            resources=TEST_RESOURCES,
            evidence_reader=tampered_pages.__getitem__,
            now=NOW,
        )

    with pytest.raises(controller.ControllerError, match="app_capture_invalid_or_unbound"):
        controller.TrustedAppCapture.from_mapping(
            {"schema_version": 1},
            resources=TEST_RESOURCES,
            evidence_reader=raw_pages.__getitem__,
            now=NOW,
        )

    with pytest.raises(TypeError, match="from_mapping"):
        controller.TrustedAppCapture()


def test_capture_authority_revalidates_raw_app_evidence_and_rejects_forgery() -> None:
    value, raw_pages = _app_capture()
    value["installations"] = [
        item for item in value["installations"] if item["name"] != "Socket Security"
    ]
    value["evidence"] = [item for item in value["evidence"] if item["installation_id"] != 109872254]
    forged = object.__new__(controller.TrustedAppCapture)
    object.__setattr__(forged, "captured_at", controller._iso(NOW))
    object.__setattr__(forged, "evidence_sha256", "1" * 64)
    object.__setattr__(forged, "policy_sha256", "2" * 64)
    object.__setattr__(forged, "installations", ())
    object.__setattr__(forged, "normalized_capture", value)
    service = controller.ReleaseGpuController(
        QueueTransport(), NoRemote(), resources=TEST_RESOURCES, clock=lambda: NOW
    )
    with pytest.raises(controller.ControllerError, match="installed_app_policy_drift"):
        service.capture_authority(
            _session(),
            forged,
            installed_app_evidence_reader=raw_pages.__getitem__,
        )


def test_capture_authority_rejects_replayed_capture_and_raw_pages() -> None:
    class AuthorityController(controller.ReleaseGpuController):
        def _paginate(self, path: str, key: str | None) -> tuple[list[Any], list[str]]:
            if path.endswith("/collaborators?affiliation=all"):
                return [{"login": controller.OWNER, "permissions": {"admin": True}}], ["1" * 64]
            if path.endswith("/invitations"):
                return [], ["2" * 64]
            if path.endswith("/actions/variables"):
                return [], ["3" * 64]
            raise AssertionError(path)

        def _runner_inventory(self) -> tuple[list[Mapping[str, Any]], str]:
            return [], "4" * 64

        def _nonce_history(
            self,
            nonces: list[str],
            *,
            exclude: tuple[int, int, int] | None = None,
            allowed_active_job_ids: set[int] | None = None,
        ) -> dict[str, Any]:
            return {
                "observed_at": controller._iso(NOW),
                "response_sha256": "5" * 64,
                "historical_match_count": 0,
                "unexpected_queued_or_in_progress_count": 0,
            }

    capture_mapping, raw_pages = _app_capture(captured_at=NOW - timedelta(minutes=1))
    capture = controller.TrustedAppCapture.from_mapping(
        capture_mapping,
        resources=TEST_RESOURCES,
        evidence_reader=raw_pages.__getitem__,
        now=NOW,
    )
    service = AuthorityController(
        QueueTransport(), NoRemote(), resources=TEST_RESOURCES, clock=lambda: NOW
    )
    service.capture_authority(
        _session(),
        capture,
        installed_app_evidence_reader=raw_pages.__getitem__,
    )
    with pytest.raises(controller.ControllerError, match="app_capture_replayed"):
        service.capture_authority(
            _session(),
            capture,
            installed_app_evidence_reader=raw_pages.__getitem__,
        )


@pytest.mark.parametrize("nonowner_permission", ["read", "write"])
def test_capture_authority_rejects_every_second_collaborator(
    nonowner_permission: str,
) -> None:
    class AuthorityController(controller.ReleaseGpuController):
        def _paginate(self, path: str, key: str | None) -> tuple[list[Any], list[str]]:
            if path.endswith("/collaborators?affiliation=all"):
                return [
                    {
                        "login": controller.OWNER,
                        "permissions": {"admin": True},
                    },
                    {
                        "login": "b-urge",
                        "permissions": {
                            "admin": False,
                            "maintain": False,
                            "push": nonowner_permission == "write",
                            "triage": False,
                            "pull": True,
                        },
                    },
                ], ["1" * 64]
            if path.endswith("/invitations"):
                return [], ["2" * 64]
            if path.endswith("/actions/variables"):
                return [], ["3" * 64]
            raise AssertionError(path)

        def _runner_inventory(self) -> tuple[list[Mapping[str, Any]], str]:
            return [], "4" * 64

        def _nonce_history(
            self,
            nonces: list[str],
            *,
            exclude: tuple[int, int, int] | None = None,
            allowed_active_job_ids: set[int] | None = None,
        ) -> dict[str, Any]:
            return {
                "observed_at": controller._iso(NOW),
                "response_sha256": "5" * 64,
                "historical_match_count": 0,
                "unexpected_queued_or_in_progress_count": 0,
            }

    mapping, raw_pages = _app_capture(captured_at=NOW - timedelta(minutes=1))
    capture = controller.TrustedAppCapture.from_mapping(
        mapping,
        resources=TEST_RESOURCES,
        evidence_reader=raw_pages.__getitem__,
        now=NOW,
    )
    service = AuthorityController(
        QueueTransport(),
        NoRemote(),
        resources=TEST_RESOURCES,
        clock=lambda: NOW,
    )
    with pytest.raises(controller.ControllerError, match="authority_not_sole_collaborator"):
        service.capture_authority(
            _session(),
            capture,
            installed_app_evidence_reader=raw_pages.__getitem__,
        )


def test_capture_authority_phase_dispatch_boundary_blocks_cross_phase_replay() -> None:
    class BoundaryController(controller.ReleaseGpuController):
        def _paginate(self, path: str, key: str | None) -> tuple[list[Any], list[str]]:
            if path.endswith("/collaborators?affiliation=all"):
                return [{"login": controller.OWNER, "permissions": {"admin": True}}], ["1" * 64]
            if path.endswith("/invitations"):
                return [], ["2" * 64]
            if path.endswith("/actions/variables"):
                return [], ["3" * 64]
            raise AssertionError(path)

        def _runner_inventory(self) -> tuple[list[Mapping[str, Any]], str]:
            return [], "4" * 64

        def _nonce_history(
            self,
            nonces: list[str],
            *,
            exclude: tuple[int, int, int] | None = None,
            allowed_active_job_ids: set[int] | None = None,
        ) -> dict[str, Any]:
            return {
                "observed_at": controller._iso(self._now()),
                "response_sha256": "5" * 64,
                "historical_match_count": 0,
                "unexpected_queued_or_in_progress_count": 0,
            }

    captured_at = NOW - timedelta(minutes=7)
    capture_mapping, raw_pages = _app_capture(captured_at=captured_at)
    capture = controller.TrustedAppCapture.from_mapping(
        capture_mapping,
        resources=TEST_RESOURCES,
        evidence_reader=raw_pages.__getitem__,
        now=NOW,
    )
    pull_request_session = _session()
    pull_request_session.dispatch_receipt = controller.DispatchReceipt(
        **{
            **pull_request_session.dispatch_receipt.__dict__,
            "observed_at": controller._iso(NOW - timedelta(minutes=10)),
        }
    )
    BoundaryController(
        QueueTransport(),
        NoRemote(),
        resources=TEST_RESOURCES,
        clock=lambda: NOW - timedelta(minutes=6),
    ).capture_authority(
        pull_request_session,
        capture,
        installed_app_evidence_reader=raw_pages.__getitem__,
    )

    final_session = _session()
    with pytest.raises(controller.ControllerError, match="phase_freshness"):
        BoundaryController(
            QueueTransport(),
            NoRemote(),
            resources=TEST_RESOURCES,
            clock=lambda: NOW,
        ).capture_authority(
            final_session,
            capture,
            installed_app_evidence_reader=raw_pages.__getitem__,
        )


@pytest.mark.parametrize(
    "captured_at",
    [
        NOW - timedelta(minutes=5),
        NOW - timedelta(minutes=5, microseconds=1),
        NOW,
    ],
)
def test_capture_authority_rejects_dispatch_or_authority_time_equality(
    captured_at: datetime,
) -> None:
    mapping, raw_pages = _app_capture(captured_at=captured_at)
    capture = controller.TrustedAppCapture.from_mapping(
        mapping,
        resources=TEST_RESOURCES,
        evidence_reader=raw_pages.__getitem__,
        now=NOW,
    )
    with pytest.raises(controller.ControllerError, match="phase_freshness"):
        controller.ReleaseGpuController(
            QueueTransport(),
            NoRemote(),
            resources=TEST_RESOURCES,
            clock=lambda: NOW,
        ).capture_authority(
            _session(),
            capture,
            installed_app_evidence_reader=raw_pages.__getitem__,
        )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ({"os": "linux"}, "jit_runner_binding_rejected"),
        ({"status": "online"}, "jit_runner_binding_rejected"),
        ({"busy": True}, "jit_runner_binding_rejected"),
        ({"name": "wrong"}, "jit_runner_binding_rejected"),
    ],
)
def test_jit_runner_requires_unknown_offline_exact_binding(
    mutation: Mapping[str, Any], error: str
) -> None:
    name = f"explainiverse-cuda-single-jit-{NONCES[0]}"
    value = _response_runner(name)
    value.update(mutation)
    with pytest.raises(controller.ControllerError, match=error):
        controller.ReleaseGpuController._normalize_jit_runner(value, name)


def test_jit_runner_rejects_any_default_or_extra_label() -> None:
    name = f"explainiverse-cuda-single-jit-{NONCES[0]}"
    labels = [
        {"id": 1, "name": "self-hosted", "type": "read-only"},
        {"id": 2, "name": name, "type": "custom"},
    ]
    with pytest.raises(controller.ControllerError, match="jit_runner_labels_not_sole"):
        controller.ReleaseGpuController._normalize_jit_runner(
            _response_runner(name, labels=labels), name
        )


def test_generate_jit_uses_group_one_redacts_and_destroys_response() -> None:
    transport = QueueTransport()
    job = _session().jobs[0]
    encoded = base64.b64encode(b"secret-jit" * 20).decode("ascii")
    transport.add(
        "POST",
        f"/repos/{controller.REPOSITORY}/actions/runners/generate-jitconfig",
        201,
        {"runner": _response_runner(job.runner_name), "encoded_jit_config": encoded},
    )
    response = transport.responses[
        ("POST", f"/repos/{controller.REPOSITORY}/actions/runners/generate-jitconfig")
    ][0]
    service = controller.ReleaseGpuController(
        transport, NoRemote(), resources=TEST_RESOURCES, clock=lambda: NOW
    )
    receipt, secret = service._generate_jit(job)
    assert receipt["runner_group_id"] == 1
    assert receipt["runner_group_get_performed"] is False
    assert "secret-jit" not in repr(receipt)
    assert "secret-jit" not in repr(secret)
    assert response.body == bytearray()
    assert transport.calls[0][2] == {
        "name": job.runner_name,
        "runner_group_id": 1,
        "labels": [job.runner_name],
        "work_folder": f"_work-{job.nonce}",
    }
    secret.destroy()
    assert secret.destroyed


def test_rejected_jit_runner_is_deleted_and_zero_inventory_is_proved() -> None:
    transport = QueueTransport()
    job = _session().jobs[0]
    encoded = base64.b64encode(b"secret-jit" * 20).decode("ascii")
    labels = [
        {"id": 1, "name": "self-hosted", "type": "read-only"},
        {"id": 2, "name": job.runner_name, "type": "custom"},
    ]
    transport.add(
        "POST",
        f"/repos/{controller.REPOSITORY}/actions/runners/generate-jitconfig",
        201,
        {
            "runner": _response_runner(job.runner_name, labels=labels),
            "encoded_jit_config": encoded,
        },
    )
    transport.add(
        "GET",
        f"/repos/{controller.REPOSITORY}/actions/runners",
        200,
        {
            "total_count": 1,
            "runners": [
                {
                    "id": 41,
                    "name": job.runner_name,
                    "os": "unknown",
                    "status": "offline",
                    "busy": False,
                    "labels": labels,
                }
            ],
        },
    )
    transport.add("DELETE", f"/repos/{controller.REPOSITORY}/actions/runners/41", 204)
    for _ in range(3):
        transport.add(
            "GET",
            f"/repos/{controller.REPOSITORY}/actions/runners",
            200,
            {"total_count": 0, "runners": []},
        )
    service = controller.ReleaseGpuController(
        transport, NoRemote(), resources=TEST_RESOURCES, clock=lambda: NOW
    )
    with pytest.raises(controller.ControllerError, match="jit_runner_labels_not_sole"):
        service._generate_jit(job)
    assert [call[:2] for call in transport.calls] == [
        (
            "POST",
            f"/repos/{controller.REPOSITORY}/actions/runners/generate-jitconfig",
        ),
        ("GET", f"/repos/{controller.REPOSITORY}/actions/runners"),
        ("DELETE", f"/repos/{controller.REPOSITORY}/actions/runners/41"),
        ("GET", f"/repos/{controller.REPOSITORY}/actions/runners"),
        ("GET", f"/repos/{controller.REPOSITORY}/actions/runners"),
        ("GET", f"/repos/{controller.REPOSITORY}/actions/runners"),
    ]
    assert not any("groups" in path for _, path, _ in transport.calls)


def test_ambiguous_online_runner_is_never_deleted() -> None:
    transport = QueueTransport()
    session = _session()
    job = session.jobs[0]
    jobs_path = (
        f"/repos/{controller.REPOSITORY}/actions/runs/{session.run['id']}"
        "/attempts/1/jobs?filter=all&per_page=100&page=1"
    )
    transport.add(
        "GET",
        jobs_path,
        200,
        {
            "total_count": 5,
            "jobs": _live_attempt_api_jobs(
                session,
                overrides={
                    job.key: {
                        "status": "in_progress",
                        "runner_id": 41,
                        "runner_name": job.runner_name,
                    }
                },
            ),
        },
    )
    transport.add(
        "GET",
        f"/repos/{controller.REPOSITORY}/actions/runners",
        200,
        {
            "total_count": 1,
            "runners": [
                {
                    "id": 41,
                    "name": job.runner_name,
                    "os": "linux",
                    "status": "online",
                    "busy": True,
                    "labels": [{"id": 91, "name": job.runner_name, "type": "custom"}],
                }
            ],
        },
    )
    service = controller.ReleaseGpuController(
        transport,
        NoRemote(),
        resources=TEST_RESOURCES,
        clock=lambda: NOW,
        sleep=lambda _: None,
    )
    receipt = service._reconcile_ambiguous_remote_start(session, job, 41, poll_limit=1)
    assert receipt["resolution"] == "still-running-no-deletion"
    assert receipt["runner_deleted_by_reconciliation"] is False
    assert not any(method == "DELETE" for method, _, _ in transport.calls)


def test_ssh_frame_failure_reaps_process_and_destroys_jit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _valid_plan()
    killed: list[bool] = []

    class FakeProcess:
        returncode = None

        def poll(self) -> None:
            return None

        def kill(self) -> None:
            killed.append(True)
            self.returncode = -9

        def communicate(self, timeout: int | None = None) -> tuple[bytes, bytes]:
            return b"", b""

    monkeypatch.setattr(controller.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(runtime, "parse_plan_document", lambda _: plan)
    monkeypatch.setattr(
        controller, "_validate_strict_ssh_binding_shape", lambda *args, **kwargs: None
    )

    def fail_frame(*args: Any, **kwargs: Any) -> None:
        raise live.ContractError("test_frame_failure")

    monkeypatch.setattr(live, "write_runtime_frame_and_close", fail_frame)
    access_identity_file = (tmp_path / "id_ed25519").resolve()
    access_identity_file.write_bytes(b"test identity fixture")

    class FakeAccessIdentity:
        public_key_sha256 = "9" * 64
        absolute_path = str(access_identity_file)
        closed = False

        def validate(self, *, expected_public_key_sha256: str) -> dict[str, Any]:
            assert expected_public_key_sha256 == self.public_key_sha256
            return {}

        def to_public_mapping(self) -> dict[str, Any]:
            return {}

        def close(self) -> None:
            self.closed = True

    binding = live.StrictSshBinding(
        argv_prefix=(
            "ssh",
            "-i",
            str(access_identity_file),
            "ubuntu@8.8.8.8",
            *live.FIXED_REMOTE_COMMAND,
        ),
        known_hosts="8.8.8.8 ssh-ed25519 public\n",
        known_hosts_path="C:/evidence/known_hosts",
        known_hosts_sha256="1" * 64,
        evidence_directory_acl_receipt_sha256="2" * 64,
        host_fingerprint="SHA256:" + "A" * 43,
        remote_mode="run",
        remote_command=live.FIXED_REMOTE_COMMAND,
    )
    secret = live.SecretBuffer(b"A" * 256, label="test_jit")
    executable = (tmp_path / "ssh.exe").resolve()
    executable.write_bytes(b"audited-test-ssh")
    executor = controller.SshRemoteExecutor(
        executable_path=str(executable),
        executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
        access_identity=FakeAccessIdentity(),  # type: ignore[arg-type]
    )
    with pytest.raises(controller.ControllerError, match="ssh_runtime_frame_failure"):
        executor.run_job(binding, runtime.canonical_json(plan), secret)
    assert killed == [True]
    assert secret.destroyed


class HistoryController(controller.ReleaseGpuController):
    def __init__(self, jobs: list[Mapping[str, Any]]) -> None:
        super().__init__(QueueTransport(), NoRemote(), resources=TEST_RESOURCES, clock=lambda: NOW)
        self._jobs = jobs

    def _all_runs(self, workflow: str) -> list[Mapping[str, Any]]:
        return [{"id": 50, "run_attempt": 1}] if workflow == "cuda-ci.yml" else []

    def _attempt_jobs(self, run_id: int, attempt: int) -> list[Mapping[str, Any]]:
        return self._jobs


def test_nonce_history_excludes_only_the_exact_current_job() -> None:
    name = f"explainiverse-cuda-single-jit-{NONCES[0]}"
    job = {"id": 51, "name": "job", "labels": [name], "status": "queued"}
    service = HistoryController([job])
    clean = service._nonce_history([NONCES[0]], exclude=(50, 1, 51), allowed_active_job_ids={51})
    assert clean["historical_match_count"] == 0
    assert clean["unexpected_queued_or_in_progress_count"] == 0
    reused = service._nonce_history([NONCES[0]], allowed_active_job_ids={51})
    assert reused["historical_match_count"] == 1


def test_nonce_history_rejects_duplicate_job_ids_across_complete_scan() -> None:
    jobs = [
        {"id": 51, "name": "one", "labels": [], "status": "completed"},
        {"id": 51, "name": "two", "labels": [], "status": "completed"},
    ]
    with pytest.raises(controller.ControllerError, match="duplicate_historical_job_id"):
        HistoryController(jobs)._nonce_history([NONCES[0]])


def test_publication_prior_nonces_are_derived_from_the_accepted_final_run() -> None:
    jobs = []
    for ordinal, key in enumerate(
        ("single_minimum", "single_latest", "two_minimum", "two_latest"), start=1
    ):
        spec = runtime.JOB_SPECS[key]
        name = f"{spec['prefix']}{NONCES[ordinal - 1]}"
        jobs.append(
            {
                "id": 60 + ordinal,
                "name": spec["name"],
                "head_sha": HEAD,
                "run_attempt": 1,
                "status": "completed",
                "conclusion": "success",
                "labels": [name],
                "runner_id": 70 + ordinal,
                "runner_name": name,
            }
        )
    service = HistoryController(jobs)
    nonces, digest = service._accepted_final_cuda_nonces(50, HEAD)
    assert nonces == NONCES
    assert controller.SHA256_RE.fullmatch(digest)

    jobs[3]["runner_id"] = jobs[2]["runner_id"]
    with pytest.raises(controller.ControllerError, match="final_cuda_job_or_runner_id_reuse"):
        service._accepted_final_cuda_nonces(50, HEAD)


def test_publication_rejects_in_memory_final_acceptance_without_journal_provenance() -> None:
    session, _, acceptance = _completed_final_acceptance()
    service = controller.ReleaseGpuController(
        QueueTransport(), NoRemote(), resources=TEST_RESOURCES, clock=lambda: NOW
    )
    with pytest.raises(
        controller.ControllerError,
        match="final_main_acceptance_not_loaded_from_journal",
    ):
        service._revalidate_final_main_acceptance(
            acceptance,
            run_id=int(session.run["id"]),
            head_sha=session.head_sha,
        )


class RecoveryDispatchController(controller.ReleaseGpuController):
    def __init__(
        self,
        transport: QueueTransport,
        histories: list[list[dict[str, Any]]],
        *,
        expected_prior_history: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(
            transport,
            NoRemote(),
            resources=TEST_RESOURCES,
            clock=lambda: NOW,
            sleep=lambda _: None,
        )
        self.histories = deque(histories)
        self.recovery_source = _recovery_source(expected_prior_history or [])

    def dispatch_release_recovery(self, **kwargs: Any) -> controller.RecoveryDispatchReceipt:
        kwargs.setdefault("recovery_source", self.recovery_source)
        return super().dispatch_release_recovery(**kwargs)

    def reconcile_release_recovery_dispatch(
        self,
        intent_mapping: Mapping[str, Any],
        **kwargs: Any,
    ) -> controller.RecoveryDispatchReceipt:
        tail = self.recovery_source.recovery_tail
        pending_tail = driver.PublicationRecoveryTail._from_verified(
            state="pending-intent",
            source_evidence_sha256=self.recovery_source.evidence_sha256,
            completed_run_ids=tail.completed_run_ids,
            completed_request_nonces=tail.completed_request_nonces,
            pending_intent=intent_mapping,
            pending_operator_settlement=None,
            last_operator_settlement=tail.last_operator_settlement,
        )
        object.__setattr__(self.recovery_source, "_recovery_tail", pending_tail)
        kwargs.setdefault("recovery_source", self.recovery_source)
        return super().reconcile_release_recovery_dispatch(intent_mapping, **kwargs)

    def _validate_publication_source(self, head_sha: str) -> dict[str, str]:
        assert head_sha == HEAD
        return {"ref": "1" * 64, "tag": "2" * 64, "main": "3" * 64}

    def _validate_recovery_source_run(
        self, source_run_id: int, head_sha: str
    ) -> tuple[dict[str, Any], str]:
        assert source_run_id == 700 and head_sha == HEAD
        material = {"source_run_id": source_run_id, "source_kind": "staged_drill"}
        return material, hashlib.sha256(_canonical(material)).hexdigest()

    def _workflow(self, filename: str, expected_path: str) -> str:
        assert filename == controller.RECOVERY_WORKFLOW
        assert expected_path == controller.RECOVERY_WORKFLOW_PATH
        return "4" * 64

    def _recovery_run_history(
        self, *, tag: str, head_sha: str, source_run_id: int
    ) -> tuple[list[dict[str, Any]], str]:
        assert tag == runtime.PUBLICATION_TAG
        assert head_sha == HEAD and source_run_id == 700
        current = self.histories.popleft()
        return current, hashlib.sha256(_canonical(current)).hexdigest()


def _recovery_run(nonce: str, *, status: str = "queued") -> dict[str, Any]:
    return {
        "id": 800,
        "display_title": controller.ReleaseGpuController._recovery_display_title(
            runtime.PUBLICATION_TAG, 700, nonce
        ),
        "head_sha": HEAD,
        "head_branch": runtime.PUBLICATION_TAG,
        "run_attempt": 1,
        "status": status,
        "conclusion": None if status != "completed" else "failure",
        "actor": controller.OWNER,
        "triggering_actor": controller.OWNER,
        "recovery_request_nonce": nonce,
    }


def _recovery_source(
    prior_history: list[dict[str, Any]],
) -> driver.PublicationRecoverySource:
    source = driver.PublicationRecoverySource._from_verified(
        head_sha=HEAD,
        run_id=700,
        control_plane_plan_sha256="1" * 64,
        publication_journal_sha256="2" * 64,
        evidence_directory_receipt_sha256="3" * 64,
        job_evidence_sha256=("4" * 64, "5" * 64),
        phase_settlement_evidence_sha256="6" * 64,
    )
    completed_run_ids = [int(item["id"]) for item in prior_history]
    completed_nonces = [str(item["recovery_request_nonce"]) for item in prior_history]
    tail = driver.PublicationRecoveryTail._from_verified(
        state="complete",
        source_evidence_sha256=source.evidence_sha256,
        completed_run_ids=completed_run_ids,
        completed_request_nonces=completed_nonces,
        pending_intent=None,
        pending_operator_settlement=None,
        last_operator_settlement=(
            None if not prior_history else {"test_only_completed_tail": True}
        ),
    )
    object.__setattr__(source, "_recovery_tail", tail)
    return source


def _raw_recovery_run(nonce: str, *, status: str = "queued") -> dict[str, Any]:
    normalized = _recovery_run(nonce, status=status)
    return {
        **normalized,
        "path": controller.RECOVERY_WORKFLOW_PATH,
        "event": "workflow_dispatch",
        "actor": {"login": normalized["actor"]},
        "triggering_actor": {"login": normalized["triggering_actor"]},
    }


def test_recovery_history_paginates_all_dispatches_without_branch_filter() -> None:
    nonce = "9123456789abcdef"
    title = controller.ReleaseGpuController._recovery_display_title(
        runtime.PUBLICATION_TAG, 700, nonce
    )
    transport = QueueTransport()
    path = (
        f"/repos/{controller.REPOSITORY}/actions/workflows/"
        f"{controller.RECOVERY_WORKFLOW}/runs?event=workflow_dispatch"
        "&per_page=100&page=1"
    )
    transport.add(
        "GET",
        path,
        200,
        {
            "total_count": 1,
            "workflow_runs": [
                {
                    "id": 800,
                    "display_title": title,
                    "path": controller.RECOVERY_WORKFLOW_PATH,
                    "head_sha": HEAD,
                    "head_branch": runtime.PUBLICATION_TAG,
                    "event": "workflow_dispatch",
                    "run_attempt": 1,
                    "status": "queued",
                    "conclusion": None,
                    "actor": {"login": controller.OWNER},
                    "triggering_actor": {"login": controller.OWNER},
                }
            ],
        },
    )
    service = controller.ReleaseGpuController(
        transport, NoRemote(), resources=TEST_RESOURCES, clock=lambda: NOW
    )
    history, digest = service._recovery_run_history(
        tag=runtime.PUBLICATION_TAG,
        head_sha=HEAD,
        source_run_id=700,
    )
    assert history[0]["recovery_request_nonce"] == nonce
    assert controller.SHA256_RE.fullmatch(digest)
    assert "branch=" not in transport.calls[0][1]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_attempt", True),
        ("status", []),
        ("conclusion", []),
        ("conclusion", "unknown"),
    ],
)
def test_recovery_history_rejects_noncanonical_run_state(field: str, value: Any) -> None:
    nonce = "f123456789abcdef"
    malformed = _raw_recovery_run(nonce, status="completed")
    malformed[field] = value
    transport = QueueTransport()
    path = (
        f"/repos/{controller.REPOSITORY}/actions/workflows/"
        f"{controller.RECOVERY_WORKFLOW}/runs?event=workflow_dispatch"
        "&per_page=100&page=1"
    )
    transport.add(
        "GET",
        path,
        200,
        {"total_count": 1, "workflow_runs": [malformed]},
    )
    service = controller.ReleaseGpuController(
        transport, NoRemote(), resources=TEST_RESOURCES, clock=lambda: NOW
    )
    with pytest.raises(controller.ControllerError, match="recovery_history_run_drift"):
        service._recovery_run_history(
            tag=runtime.PUBLICATION_TAG,
            head_sha=HEAD,
            source_run_id=700,
        )


def test_recovery_dispatch_is_nonce_bound_journaled_and_reconciled() -> None:
    transport = QueueTransport()
    nonce = "0123456789abcdef"
    path = (
        f"/repos/{controller.REPOSITORY}/actions/workflows/"
        f"{controller.RECOVERY_WORKFLOW}/dispatches"
    )
    transport.add("POST", path, 204)
    progress: list[tuple[str, Mapping[str, Any]]] = []
    service = RecoveryDispatchController(transport, [[], [_recovery_run(nonce)]])
    receipt = service.dispatch_release_recovery(
        head_sha=HEAD,
        source_run_id=700,
        recovery_request_nonce=nonce,
        progress=lambda label, payload: progress.append((label, payload)),
        poll_limit=1,
    )
    assert receipt.run_id == 800
    assert receipt.mutation_response_received is True
    assert [label for label, _ in progress] == [
        "github-recovery-dispatch-intent",
        "github-recovery-dispatch-settled",
    ]
    assert transport.calls[-1] == (
        "POST",
        path,
        {
            "ref": runtime.PUBLICATION_TAG,
            "inputs": {
                "tag": runtime.PUBLICATION_TAG,
                "source_run_id": "700",
                "recovery_request_nonce": nonce,
                "require_staged_drill": True,
            },
        },
    )


def test_recovery_dispatch_lost_response_is_observed_without_replay() -> None:
    class AmbiguousTransport(QueueTransport):
        def request(
            self, method: str, path: str, body: Mapping[str, Any] | None = None
        ) -> controller.GitHubResponse:
            self.calls.append((method, path, body))
            if method == "POST":
                raise controller.AmbiguousGitHubMutation(
                    method,
                    path,
                    hashlib.sha256(
                        _canonical({"method": method, "path": path, "body": body})
                    ).hexdigest(),
                    "test-response-lost",
                )
            return super().request(method, path, body)

    nonce = "1123456789abcdef"
    transport = AmbiguousTransport()
    progress: list[tuple[str, Mapping[str, Any]]] = []
    service = RecoveryDispatchController(transport, [[], [_recovery_run(nonce)]])
    receipt = service.dispatch_release_recovery(
        head_sha=HEAD,
        source_run_id=700,
        recovery_request_nonce=nonce,
        progress=lambda label, payload: progress.append((label, payload)),
        poll_limit=1,
    )
    assert receipt.mutation_response_received is False
    assert len([item for item in transport.calls if item[0] == "POST"]) == 1

    intent = progress[0][1]
    observation_only_transport = QueueTransport()
    reconciler = RecoveryDispatchController(observation_only_transport, [[_recovery_run(nonce)]])
    reconciled = reconciler.reconcile_release_recovery_dispatch(intent, poll_limit=1)
    assert reconciled.run_id == 800
    assert not any(item[0] == "POST" for item in observation_only_transport.calls)


def test_recovery_dispatch_absence_never_authorizes_a_retry() -> None:
    nonce = "2123456789abcdef"
    progress: list[tuple[str, Mapping[str, Any]]] = []
    transport = QueueTransport()
    path = (
        f"/repos/{controller.REPOSITORY}/actions/workflows/"
        f"{controller.RECOVERY_WORKFLOW}/dispatches"
    )
    transport.add("POST", path, 204)
    service = RecoveryDispatchController(transport, [[], []])
    with pytest.raises(
        controller.AmbiguousGitHubMutation,
        match="recovery_dispatch_visibility_unresolved",
    ):
        service.dispatch_release_recovery(
            head_sha=HEAD,
            source_run_id=700,
            recovery_request_nonce=nonce,
            progress=lambda label, payload: progress.append((label, payload)),
            poll_limit=1,
        )
    intent = progress[0][1]
    reconciler = RecoveryDispatchController(QueueTransport(), [[]])
    with pytest.raises(
        controller.AmbiguousGitHubMutation,
        match="recovery_dispatch_absence_not_proof",
    ):
        reconciler.reconcile_release_recovery_dispatch(intent, poll_limit=1)
    assert not any(item[0] == "POST" for item in reconciler._github.calls)


def test_recovery_dispatch_journal_reopens_pending_intent_and_settles_read_only(
    tmp_path: Path,
) -> None:
    nonce = "6123456789abcdef"
    evidence = (tmp_path / "recovery-evidence").resolve()
    directory_receipt = live.create_evidence_directory(evidence)
    receipt_sha256 = directory_receipt.receipt_sha256
    journal = driver.EvidenceJournal(directory_receipt, plan_sha256=CONTROL_SHA)
    transport = QueueTransport()
    path = (
        f"/repos/{controller.REPOSITORY}/actions/workflows/"
        f"{controller.RECOVERY_WORKFLOW}/dispatches"
    )
    transport.add("POST", path, 204)
    service = RecoveryDispatchController(transport, [[], []])
    with pytest.raises(controller.AmbiguousGitHubMutation):
        service.dispatch_release_recovery(
            head_sha=HEAD,
            source_run_id=700,
            recovery_request_nonce=nonce,
            progress=journal.record,
            poll_limit=1,
        )
    expected_intent = journal.pending_recovery_dispatch_intent()
    assert expected_intent is not None
    journal.close()

    reopened_receipt = live.reopen_evidence_directory(
        evidence, expected_receipt_sha256=receipt_sha256
    )
    reopened = driver.EvidenceJournal.reopen_for_recovery(reopened_receipt, plan_sha256=CONTROL_SHA)
    assert reopened.pending_recovery_dispatch_intent() == expected_intent
    reconciler_transport = QueueTransport()
    reconciler = RecoveryDispatchController(reconciler_transport, [[_recovery_run(nonce)]])
    receipt = reconciler.reconcile_release_recovery_dispatch(
        expected_intent,
        poll_limit=1,
        progress=reopened.record,
    )
    assert receipt.run_id == 800
    assert reopened.pending_recovery_dispatch_intent() is None
    assert not any(item[0] == "POST" for item in reconciler_transport.calls)
    reopened.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("workflow_response_sha256", "a" * 64),
        ("immutable_source_evidence_sha256", "b" * 64),
        ("source_run_evidence_sha256", "c" * 64),
        ("pre_dispatch_history_sha256", "d" * 64),
        ("run_id", "800"),
        ("run_attempt", 2),
        ("run_attempt", True),
        ("status", "waiting"),
        ("status", []),
        ("conclusion", "success"),
        ("observed_at", "not-a-timestamp"),
        ("observed_at", "2026-08-28T22:00:00Z"),
        ("reconciliation_sha256", "short"),
        ("mutation_response_received", False),
    ],
)
def test_recovery_dispatch_journal_rejects_settlement_drift(
    tmp_path: Path, field: str, value: Any
) -> None:
    nonce = "a123456789abcdef"
    transport = QueueTransport()
    path = (
        f"/repos/{controller.REPOSITORY}/actions/workflows/"
        f"{controller.RECOVERY_WORKFLOW}/dispatches"
    )
    transport.add("POST", path, 204)
    progress: list[tuple[str, Mapping[str, Any]]] = []
    service = RecoveryDispatchController(transport, [[], [_recovery_run(nonce)]])
    service.dispatch_release_recovery(
        head_sha=HEAD,
        source_run_id=700,
        recovery_request_nonce=nonce,
        progress=lambda label, payload: progress.append((label, payload)),
        poll_limit=1,
    )
    intent = progress[0][1]
    settlement = deepcopy(progress[1][1])
    settlement[field] = value
    settlement_material = dict(settlement)
    settlement_material.pop("evidence_sha256")
    settlement["evidence_sha256"] = hashlib.sha256(_canonical(settlement_material)).hexdigest()

    evidence = (tmp_path / "evidence").resolve()
    directory_receipt = live.create_evidence_directory(evidence)
    journal = driver.EvidenceJournal(directory_receipt, plan_sha256=CONTROL_SHA)
    journal.record("github-recovery-dispatch-intent", intent)
    journal.record("github-recovery-dispatch-settled", settlement)
    with pytest.raises(controller.ControllerError, match="recovery_dispatch_"):
        journal.pending_recovery_dispatch_intent()
    journal.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("pre_dispatch_run_ids", [2, 1]),
        ("pre_dispatch_run_ids", [1, "2"]),
    ],
)
def test_recovery_dispatch_intent_rejects_noncanonical_schema(field: str, value: Any) -> None:
    nonce = "b123456789abcdef"
    transport = QueueTransport()
    path = (
        f"/repos/{controller.REPOSITORY}/actions/workflows/"
        f"{controller.RECOVERY_WORKFLOW}/dispatches"
    )
    transport.add("POST", path, 204)
    progress: list[tuple[str, Mapping[str, Any]]] = []
    service = RecoveryDispatchController(transport, [[], [_recovery_run(nonce)]])
    service.dispatch_release_recovery(
        head_sha=HEAD,
        source_run_id=700,
        recovery_request_nonce=nonce,
        progress=lambda label, payload: progress.append((label, payload)),
        poll_limit=1,
    )
    malformed = deepcopy(progress[0][1])
    malformed[field] = value
    with pytest.raises(controller.ControllerError, match="recovery_dispatch_intent_"):
        controller.ReleaseGpuController._validate_recovery_dispatch_intent(malformed)


@pytest.mark.parametrize("nonce", ["ABCDEF0123456789", "0" * 15, "0" * 17])
def test_recovery_dispatch_rejects_invalid_nonce_before_contact(nonce: str) -> None:
    transport = QueueTransport()
    service = RecoveryDispatchController(transport, [])
    with pytest.raises(controller.ControllerError, match="recovery_request_nonce_rejected"):
        service.dispatch_release_recovery(
            head_sha=HEAD,
            source_run_id=700,
            recovery_request_nonce=nonce,
            progress=lambda *_: None,
        )
    assert transport.calls == []


def test_recovery_dispatch_rejects_active_success_and_nonce_reuse() -> None:
    nonce = "3123456789abcdef"
    cases = [
        (
            [_recovery_run("4123456789abcdef")],
            "recovery_live_history_not_exact_journal_tail",
        ),
        (
            [
                {
                    **_recovery_run("5123456789abcdef", status="completed"),
                    "conclusion": "success",
                }
            ],
            "recovery_live_history_not_exact_journal_tail",
        ),
        (
            [
                {
                    **_recovery_run("6123456789abcdef", status="completed"),
                    "conclusion": "neutral",
                }
            ],
            "recovery_live_history_not_exact_journal_tail",
        ),
        (
            [_recovery_run(nonce, status="completed")],
            "recovery_live_history_not_exact_journal_tail",
        ),
    ]
    for history, error in cases:
        transport = QueueTransport()
        service = RecoveryDispatchController(transport, [history])
        with pytest.raises(controller.ControllerError, match=error):
            service.dispatch_release_recovery(
                head_sha=HEAD,
                source_run_id=700,
                recovery_request_nonce=nonce,
                progress=lambda *_: None,
            )
        assert transport.calls == []


def test_recovery_dispatch_allows_fresh_nonce_only_after_exact_failed_history() -> None:
    prior = _recovery_run("7123456789abcdef", status="completed")
    nonce = "8123456789abcdef"
    current = [prior, {**_recovery_run(nonce), "id": 801}]
    transport = QueueTransport()
    path = (
        f"/repos/{controller.REPOSITORY}/actions/workflows/"
        f"{controller.RECOVERY_WORKFLOW}/dispatches"
    )
    transport.add("POST", path, 204)
    service = RecoveryDispatchController(
        transport,
        [[prior], current],
        expected_prior_history=[prior],
    )
    progress: list[tuple[str, Mapping[str, Any]]] = []
    receipt = service.dispatch_release_recovery(
        head_sha=HEAD,
        source_run_id=700,
        recovery_request_nonce=nonce,
        progress=lambda label, payload: progress.append((label, payload)),
        poll_limit=1,
    )
    assert receipt.run_id == 801
    assert progress[0][1]["pre_dispatch_run_ids"] == [800]
    assert progress[0][1]["pre_dispatch_runs"] == [prior]


@pytest.mark.parametrize(
    "live_history",
    [
        [],
        [_recovery_run("7123456789abcdef", status="completed")],
        [
            {**_recovery_run("7123456789abcdef", status="completed"), "id": 801},
            {**_recovery_run("6123456789abcdef", status="completed"), "id": 800},
        ],
        [
            {
                **_recovery_run("6123456789abcdef", status="completed"),
                "actor": "not-the-owner",
            },
            {**_recovery_run("7123456789abcdef", status="completed"), "id": 801},
        ],
    ],
    ids=("omitted", "subset", "superset", "identity-drift"),
)
def test_recovery_dispatch_rejects_live_history_not_equal_to_loader_tail(
    live_history: list[dict[str, Any]],
) -> None:
    prior = [
        _recovery_run("6123456789abcdef", status="completed"),
        {**_recovery_run("7123456789abcdef", status="completed"), "id": 801},
    ]
    transport = QueueTransport()
    service = RecoveryDispatchController(
        transport,
        [live_history],
        expected_prior_history=prior,
    )
    with pytest.raises(
        controller.ControllerError,
        match="recovery_live_history_not_exact_journal_tail",
    ):
        service.dispatch_release_recovery(
            head_sha=HEAD,
            source_run_id=700,
            recovery_request_nonce="8123456789abcdef",
            progress=lambda *_: None,
        )
    assert not any(method == "POST" for method, _, _ in transport.calls)


def test_recovery_reconciliation_rejects_missing_or_drifted_prior_history() -> None:
    prior = _recovery_run("7123456789abcdef", status="completed")
    nonce = "8123456789abcdef"
    current = [prior, {**_recovery_run(nonce), "id": 801}]
    dispatch_transport = QueueTransport()
    dispatch_transport.add(
        "POST",
        f"/repos/{controller.REPOSITORY}/actions/workflows/"
        f"{controller.RECOVERY_WORKFLOW}/dispatches",
        204,
    )
    progress: list[tuple[str, Mapping[str, Any]]] = []
    RecoveryDispatchController(
        dispatch_transport,
        [[prior], current],
        expected_prior_history=[prior],
    ).dispatch_release_recovery(
        head_sha=HEAD,
        source_run_id=700,
        recovery_request_nonce=nonce,
        progress=lambda label, payload: progress.append((label, payload)),
        poll_limit=1,
    )
    intent = progress[0][1]
    reconciler_transport = QueueTransport()
    reconciler = RecoveryDispatchController(
        reconciler_transport,
        [[{**_recovery_run(nonce), "id": 801}]],
        expected_prior_history=[prior],
    )
    with pytest.raises(
        controller.ControllerError,
        match="recovery_dispatch_reconciliation_ambiguous",
    ):
        reconciler.reconcile_release_recovery_dispatch(intent, poll_limit=1)
    assert not any(method == "POST" for method, _, _ in reconciler_transport.calls)


@pytest.mark.parametrize(
    ("status", "conclusion"),
    [
        ("queued", None),
        ("in_progress", None),
        ("completed", "success"),
        ("completed", "cancelled"),
        ("completed", "timed_out"),
        ("completed", "action_required"),
        ("completed", "neutral"),
        ("completed", "skipped"),
        ("completed", "stale"),
        ("completed", "startup_failure"),
        ("completed", "unknown"),
    ],
)
def test_recovery_dispatch_intent_rejects_nonfailure_prior_run_evidence(
    status: str,
    conclusion: str | None,
) -> None:
    prior = _recovery_run("7123456789abcdef", status="completed")
    nonce = "8123456789abcdef"
    current = [prior, {**_recovery_run(nonce), "id": 801}]
    transport = QueueTransport()
    path = (
        f"/repos/{controller.REPOSITORY}/actions/workflows/"
        f"{controller.RECOVERY_WORKFLOW}/dispatches"
    )
    transport.add("POST", path, 204)
    progress: list[tuple[str, Mapping[str, Any]]] = []
    RecoveryDispatchController(
        transport,
        [[prior], current],
        expected_prior_history=[prior],
    ).dispatch_release_recovery(
        head_sha=HEAD,
        source_run_id=700,
        recovery_request_nonce=nonce,
        progress=lambda label, payload: progress.append((label, payload)),
        poll_limit=1,
    )
    malformed = deepcopy(progress[0][1])
    malformed["pre_dispatch_runs"][0]["status"] = status
    malformed["pre_dispatch_runs"][0]["conclusion"] = conclusion
    with pytest.raises(controller.ControllerError, match="prior_run_not_exact_failure"):
        controller.ReleaseGpuController._validate_recovery_dispatch_intent(malformed)


def test_recovery_workflow_binds_nonce_into_title_and_request_evidence() -> None:
    workflow = Path(".github/workflows/recover-github-release.yml").read_text(encoding="utf-8")
    assert (
        "run-name: explainiverse-recovery-${{ inputs.tag }}-"
        "${{ inputs.source_run_id }}-${{ inputs.recovery_request_nonce }}"
    ) in workflow
    assert "recovery_request_nonce:" in workflow
    assert "^[0-9a-f]{16}$" in workflow
    assert ".recovery_request_nonce == $nonce" in workflow


def _pr_payload(*, state: str = "open", head_sha: str = HEAD) -> dict[str, Any]:
    return {
        "number": 4,
        "state": state,
        "draft": False,
        "base": {
            "ref": "main",
            "sha": "b" * 40,
            "repo": {"full_name": controller.REPOSITORY},
        },
        "head": {
            "ref": runtime.PULL_REQUEST_REF.removeprefix("refs/heads/"),
            "sha": head_sha,
            "repo": {"full_name": controller.REPOSITORY},
        },
        "mergeable": True,
        "mergeable_state": "blocked",
    }


def test_pr_four_source_binds_open_head_base_and_same_repository() -> None:
    transport = QueueTransport()
    transport.add(
        "GET",
        f"/repos/{controller.REPOSITORY}/git/ref/heads/main",
        200,
        {"ref": "refs/heads/main", "object": {"type": "commit", "sha": "b" * 40}},
    )
    transport.add("GET", f"/repos/{controller.REPOSITORY}/pulls/4", 200, _pr_payload())
    service = controller.ReleaseGpuController(
        transport, NoRemote(), resources=TEST_RESOURCES, clock=lambda: NOW
    )
    assert set(service._validate_pull_request_source(HEAD)) == {"pull_request", "main"}


@pytest.mark.parametrize(
    "change",
    [
        {"state": "closed"},
        {"head_sha": "c" * 40},
    ],
)
def test_pr_four_source_rejects_state_or_head_drift(change: Mapping[str, str]) -> None:
    transport = QueueTransport()
    transport.add(
        "GET",
        f"/repos/{controller.REPOSITORY}/git/ref/heads/main",
        200,
        {"ref": "refs/heads/main", "object": {"type": "commit", "sha": "b" * 40}},
    )
    transport.add(
        "GET",
        f"/repos/{controller.REPOSITORY}/pulls/4",
        200,
        _pr_payload(**change),
    )
    service = controller.ReleaseGpuController(
        transport, NoRemote(), resources=TEST_RESOURCES, clock=lambda: NOW
    )
    with pytest.raises(controller.ControllerError, match="pull_request_4_source_drift"):
        service._validate_pull_request_source(HEAD)


def test_dispatch_rejects_wrong_ref_before_any_api_call() -> None:
    transport = QueueTransport()
    service = controller.ReleaseGpuController(
        transport, NoRemote(), resources=TEST_RESOURCES, clock=lambda: NOW
    )
    with pytest.raises(controller.ControllerError, match="dispatch_supplied_ref_rejected"):
        service.dispatch_phase("final-main", head_sha=HEAD, supplied_ref="main")
    assert transport.calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (("workflow_path",), ".github/workflows/wrong.yml"),
        (("dispatch", "ref"), "refs/heads/wrong"),
        (("dispatch", "actor"), "attacker"),
        (("dispatch", "run_attempt"), 2),
        (("job", "labels"), ["self-hosted"]),
    ],
)
def test_runtime_plan_rejects_workflow_ref_actor_attempt_and_label_drift(
    field: tuple[str, ...], value: Any
) -> None:
    plan = deepcopy(_valid_plan())
    target = plan
    for key in field[:-1]:
        target = target[key]
    target[field[-1]] = value
    with pytest.raises(runtime.ContractError):
        runtime.validate_runtime_plan(plan, now=NOW)


def test_publication_plan_rejects_tag_signature_and_input_drift() -> None:
    plan = _valid_plan()
    plan["phase"] = "publication"
    plan["workflow_path"] = runtime.PUBLISH_WORKFLOW_PATH
    plan["dispatch"].update(
        {
            "ref": runtime.PUBLICATION_REF,
            "inputs": {
                "tag": "v0.15.1",
                "preflight_run_id": 1,
                "cuda_run_id": 2,
                "single_minimum_runner_nonce": NONCES[0],
                "single_latest_runner_nonce": NONCES[1],
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
    with pytest.raises(runtime.ContractError, match="publication_dispatch_inputs_rejected"):
        runtime.validate_runtime_plan(plan, now=NOW)


def test_remote_receipt_cannot_overclaim_actions_or_test_evidence() -> None:
    plan = _valid_plan()
    execution = _remote_execution(plan)
    assert controller.ReleaseGpuController._validate_remote_receipt(plan, execution)
    overclaim = deepcopy(execution.receipt)
    overclaim["accepted_actions_evidence"] = True
    with pytest.raises(controller.ControllerError, match="remote_receipt_overclaim_or_drift"):
        controller.ReleaseGpuController._validate_remote_receipt(
            plan,
            controller.RemoteExecution(
                overclaim,
                execution.stdout_sha256,
                execution.stderr_sha256,
                execution.frame_receipt,
            ),
        )


@pytest.mark.parametrize(
    "log",
    [
        b"14 passed in 1.00s\n",
        b"15 passed, 1 skipped in 1.00s\n",
        b"15 passed in 1.00s\n15 passed in 1.01s\n",
    ],
)
def test_actions_log_requires_exactly_one_15_passed_and_zero_skips(log: bytes) -> None:
    with pytest.raises(controller.ControllerError, match="pytest_15_zero_skip_evidence_missing"):
        controller.ReleaseGpuController._validate_pytest_log(log)
    assert controller.ReleaseGpuController._validate_pytest_log(b"15 passed in 2.34s\n")[:2] == (
        15,
        0,
    )


def test_remote_argv_rejects_nonce_gpu_or_jit_values() -> None:
    plan = _valid_plan()
    safe = ["ssh", "host", *live.FIXED_REMOTE_COMMAND]
    controller.SshRemoteExecutor._validate_remote_argv(safe, plan)
    for forbidden in (
        plan["job"]["runner_nonce"],
        plan["job"]["jit_config_sha256"],
        plan["hardware"]["host_physical_gpu_uuids"][0],
    ):
        with pytest.raises(controller.ControllerError, match="secret_or_plan_value_in_ssh_argv"):
            controller.SshRemoteExecutor._validate_remote_argv([*safe, forbidden], plan)


def test_no_controller_source_mentions_remote_github_token_or_shell_true() -> None:
    source = Path(controller.__file__).read_text(encoding="utf-8")
    assert "shell=True" not in source
    assert "--input" in source


class _DriverReceipt:
    def __init__(self, phase: str) -> None:
        self.phase = phase
        self.instance_public_ipv4 = "8.8.8.8" if phase == "instance_bound" else None
        self.ruleset_id = "ruleset-0123456789abcdef"
        self.instance_id = "instance-0123456789abcdef"

    def to_public_mapping(self) -> dict[str, Any]:
        return {
            "plan_sha256": CONTROL_SHA,
            "phase": self.phase,
            "snapshot_sha256": hashlib.sha256(self.phase.encode()).hexdigest(),
            "receipt_nonce": "0" * 32,
            "ruleset_id": self.ruleset_id,
            "instance_id": self.instance_id,
            "instance_public_ipv4": self.instance_public_ipv4,
            "response_bindings": [],
        }


class _DriverMutation:
    def __init__(self, operation: str) -> None:
        self.operation = operation

    def to_public_mapping(self) -> dict[str, Any]:
        return {
            "plan_sha256": CONTROL_SHA,
            "operation": self.operation,
            "request_sha256": hashlib.sha256(self.operation.encode()).hexdigest(),
        }


class _DriverProvider:
    def __init__(self, state: str, events: list[str]) -> None:
        self.state = state
        self.events = events
        self.plan_sha256 = CONTROL_SHA
        self.mutation_intent_binding_sha256: str | None = None
        self._mutation_intent_callback: Any = None

    def bind_mutation_intent_callback(self, callback: Any) -> str:
        assert self.mutation_intent_binding_sha256 is None
        self._mutation_intent_callback = callback
        self.mutation_intent_binding_sha256 = "d" * 64
        return self.mutation_intent_binding_sha256

    def mutation_intent_callback_matches(self, callback: Any) -> bool:
        if self._mutation_intent_callback is None:
            return False
        return getattr(self._mutation_intent_callback, "__self__", None) is getattr(
            callback, "__self__", None
        ) and getattr(self._mutation_intent_callback, "__func__", None) is getattr(
            callback, "__func__", None
        )

    def ambiguity_from_persisted_intent(self, value: Mapping[str, Any]) -> live.AmbiguousMutation:
        intent = live.MutationIntent.from_public_mapping(value)
        assert intent.plan_sha256 == self.plan_sha256
        return live.AmbiguousMutation(intent.operation, intent.request_sha256, "test-process-crash")

    def observe(self, phase: str) -> _DriverReceipt:
        aliases = {
            "ruleset_ready": {"ruleset_ready", "instance_absent"},
            "global_restricted": {"global_restricted", "ruleset_absent"},
            "baseline": {"baseline", "restored"},
        }
        if phase not in aliases.get(self.state, {self.state}):
            raise live.ContractError("test_phase_mismatch")
        return _DriverReceipt(phase)

    def terminate(self, receipt: _DriverReceipt) -> _DriverMutation:
        self.events.append("provider:terminate")
        self.state = "instance_absent"
        return _DriverMutation("terminate")

    def delete_ruleset(self, receipt: _DriverReceipt) -> _DriverMutation:
        self.events.append("provider:delete-ruleset")
        self.state = "ruleset_absent"
        return _DriverMutation("delete_ruleset")

    def restore_global(self, receipt: _DriverReceipt) -> _DriverMutation:
        self.events.append("provider:restore-global")
        self.state = "restored"
        return _DriverMutation("restore_global")

    def recover_ambiguous(self, ambiguity: live.AmbiguousMutation, receipt: _DriverReceipt) -> Any:
        raise AssertionError("unexpected recovery")


class _DriverJournal:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.records: list[tuple[str, Mapping[str, Any]]] = []
        self.acl_receipt_sha256 = "a" * 64

    def record(self, label: str, payload: Mapping[str, Any]) -> str:
        self.events.append(f"journal:{label}")
        self.records.append((label, payload))
        return hashlib.sha256(_canonical({"label": label, "payload": payload})).hexdigest()

    def verified_entries(self) -> tuple[tuple[str, dict[str, Any]], ...]:
        return tuple((label, dict(payload)) for label, payload in self.records)

    def close(self) -> None:
        self.events.append("journal:close")


class _DriverController:
    def __init__(self, provider: _DriverProvider, events: list[str]) -> None:
        self.provider = provider
        self.events = events

    def reconcile_runner_after_host_stop(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        assert self.provider.state != "instance_bound"
        self.events.append("controller:runner-reconcile")
        return {"evidence_sha256": "1" * 64}

    def prove_zero_runner_inventory(self) -> dict[str, Any]:
        self.events.append("controller:zero")
        return {"evidence_sha256": "2" * 64}

    def cancel_failed_phase(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.events.append("controller:cancel")
        return {"evidence_sha256": "3" * 64}

    def reconcile_ambiguous_dispatch_for_abort(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.events.append("controller:dispatch-reconcile")
        return {"evidence_sha256": "4" * 64}

    def reconcile_cancel_for_abort(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.events.append("controller:cancel-reconcile")
        return {"evidence_sha256": "5" * 64}


class _DriverIdentity:
    def __init__(self) -> None:
        self.destroyed = False

    def destroy(self) -> None:
        self.destroyed = True


@pytest.mark.parametrize("close_fails", [False, True])
def test_driver_constructor_failure_destroys_owned_identities_and_closes_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    close_fails: bool,
) -> None:
    events: list[str] = []

    class AccessIdentity:
        absolute_path = str((tmp_path / "access.key").resolve())
        closed = False

    monkeypatch.setattr(driver.live, "AccessIdentityReceipt", AccessIdentity)

    class Identity:
        fingerprint = "SHA256:" + "A" * 43
        destroyed = False

        def destroy(self) -> None:
            self.destroyed = True
            events.append("identity:destroy")

    class Journal:
        _plan_sha256 = CONTROL_SHA
        directory = tmp_path.resolve()

        def record_provider_mutation_intent(self, intent: Any) -> None:
            raise AssertionError("provider must not mutate during construction")

        def close(self) -> None:
            events.append("journal:close")

    class Service:
        def bind_access_identity(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
            raise controller.ControllerError("simulated_bind_access_failure")

        def close_access_identity(self) -> None:
            access.closed = True
            events.append("access:close")
            if close_fails:
                raise OSError("simulated access close failure")

    access = AccessIdentity()
    identity = Identity()
    provider = _DriverProvider("baseline", events)
    plan = SimpleNamespace(
        sha256=CONTROL_SHA,
        host_key_fingerprint=identity.fingerprint,
        runtime_bundle_sha256="8" * 64,
        ssh_public_key_sha256="9" * 64,
    )
    runtime_bundle = SimpleNamespace(sha256="8" * 64)
    with pytest.raises(controller.ControllerError, match="simulated_bind_access_failure"):
        driver.LiveReleaseDriver(
            Service(),  # type: ignore[arg-type]
            provider,  # type: ignore[arg-type]
            plan,  # type: ignore[arg-type]
            identity,  # type: ignore[arg-type]
            runtime_bundle,  # type: ignore[arg-type]
            Journal(),  # type: ignore[arg-type]
            access_identity=access,  # type: ignore[arg-type]
            known_hosts_path=(tmp_path / "known_hosts").resolve(),
        )
    assert identity.destroyed is True
    assert access.closed is True
    assert "journal:close" in events


def _bare_driver(
    provider_state: str,
    *,
    session: controller.PhaseSession | None = None,
) -> tuple[driver.LiveReleaseDriver, _DriverProvider, list[str]]:
    events: list[str] = []
    provider = _DriverProvider(provider_state, events)
    journal = _DriverJournal(events)
    service = _DriverController(provider, events)
    value = object.__new__(driver.LiveReleaseDriver)
    value._controller = service  # type: ignore[assignment]
    value._provider = provider  # type: ignore[assignment]
    value._plan = SimpleNamespace(
        sha256=CONTROL_SHA,
        head_sha=HEAD,
        ssh_public_key_sha256="9" * 64,
        host_key_fingerprint="SHA256:" + "A" * 43,
    )
    value._identity = _DriverIdentity()
    value._journal = journal  # type: ignore[assignment]
    value._sleep = lambda _: None
    value._observation_poll_limit = 2
    value._state = "phase-failed"
    value._known_hosts = None
    value._access_identity = None
    value._phase = session.phase if session is not None else None
    value._session = session
    value._active_job = session.jobs[0] if session is not None else None
    value._dispatch_ambiguity = None
    value._github_ambiguity = None
    value._remote_ambiguity_receipt = None
    value._crash_jit_intent = None
    value._provider_crash_ambiguity = None
    value._runner_delete_intent = None
    value._cleanup_evidence_errors = None
    return value, provider, events


def test_driver_remote_ambiguity_stops_host_before_runner_and_run_cleanup() -> None:
    value, provider, events = _bare_driver("instance_bound", session=_session())
    value._state = "remote-start-ambiguous"
    value._remote_ambiguity_receipt = {"job_id": 101, "runner_id": 41}
    value.abort()
    assert provider.state == "restored"
    assert events.index("provider:terminate") < events.index("controller:runner-reconcile")
    assert events.index("controller:runner-reconcile") < events.index("controller:zero")
    assert events.index("controller:zero") < events.index("controller:cancel")
    assert value._identity.destroyed


@pytest.mark.parametrize(
    ("initial", "mutations"),
    [
        ("baseline", []),
        ("global_restricted", ["provider:restore-global"]),
        ("ruleset_ready", ["provider:delete-ruleset", "provider:restore-global"]),
    ],
)
def test_driver_unwinds_every_partial_provider_state(initial: str, mutations: list[str]) -> None:
    value, provider, events = _bare_driver(initial)
    value.abort()
    assert provider.state in {"baseline", "restored"}
    assert [event for event in events if event.startswith("provider:")] == mutations


def test_driver_dispatch_ambiguity_is_reconciled_before_provider_restore() -> None:
    value, provider, events = _bare_driver("instance_bound")
    value._phase = "final-main"
    value._dispatch_ambiguity = controller.AmbiguousGitHubMutation(
        "POST",
        f"/repos/{controller.REPOSITORY}/actions/workflows/cuda-ci.yml/dispatches",
        "8" * 64,
        "test",
        reconciliation={},
    )
    value.abort()
    assert provider.state == "restored"
    assert events.index("controller:dispatch-reconcile") < events.index("provider:terminate")


def test_driver_cancel_intent_uses_read_only_reconciliation_same_process() -> None:
    session = _session()
    value, provider, events = _bare_driver("baseline", session=session)
    cancel_path = f"/repos/{controller.REPOSITORY}/actions/runs/{session.run['id']}/cancel"
    value._record_progress(
        "github-cancel-intent",
        {
            "phase": session.phase,
            "run_id": session.run["id"],
            "run_attempt": 1,
            "head_sha": session.head_sha,
            "cancel_path": cancel_path,
            "request_sha256": hashlib.sha256(
                _canonical({"method": "POST", "path": cancel_path, "body": None})
            ).hexdigest(),
            "reason": "incomplete-phase",
            "accepted_job_ids": [],
            "serviced_job_ids": [],
            "mutation_retried": False,
        },
    )
    value.abort()
    assert provider.state in {"baseline", "restored"}
    assert "controller:cancel-reconcile" in events
    assert "controller:cancel" not in events


def test_driver_provider_cleanup_survives_regular_journal_failures() -> None:
    value, provider, events = _bare_driver("instance_bound")

    class FailingJournal(_DriverJournal):
        def record(self, label: str, payload: Mapping[str, Any]) -> str:
            if label.startswith("provider-") or label.startswith("lifecycle-"):
                raise OSError("simulated evidence failure")
            return super().record(label, payload)

    value._journal = FailingJournal(events)  # type: ignore[assignment]
    with pytest.raises(controller.ControllerError, match="abort_evidence_archival_incomplete"):
        value.abort()
    assert provider.state == "restored"
    assert [event for event in events if event.startswith("provider:")] == [
        "provider:terminate",
        "provider:delete-ruleset",
        "provider:restore-global",
    ]


def test_driver_provider_cleanup_survives_access_identity_close_failure() -> None:
    value, provider, events = _bare_driver("instance_bound")

    class Access:
        closed = False

    value._access_identity = Access()  # type: ignore[assignment]

    def fail_close() -> None:
        raise OSError("simulated CloseHandle failure")

    value._controller.close_access_identity = fail_close  # type: ignore[attr-defined,method-assign]
    with pytest.raises(OSError, match="CloseHandle"):
        value.abort()
    assert provider.state == "restored"
    assert [event for event in events if event.startswith("provider:")] == [
        "provider:terminate",
        "provider:delete-ruleset",
        "provider:restore-global",
    ]


def test_evidence_journal_reopens_and_resume_recovers_dispatch_intent(
    tmp_path: Path,
) -> None:
    evidence = (tmp_path / "evidence").resolve()
    directory_receipt = live.create_evidence_directory(evidence)
    plan = SimpleNamespace(
        sha256=CONTROL_SHA,
        head_sha=HEAD,
        to_mapping=lambda: {"head_sha": HEAD, "kind": "test-plan"},
    )
    journal = driver.EvidenceJournal(
        directory_receipt,
        plan_sha256=CONTROL_SHA,
    )
    journal.record("immutable-plan", plan.to_mapping())
    journal.record(
        "github-dispatch-intent",
        {
            "phase": "final-main",
            "workflow": "cuda-ci.yml",
            "workflow_path": runtime.CUDA_WORKFLOW_PATH,
            "dispatch_path": (
                f"/repos/{controller.REPOSITORY}/actions/workflows/cuda-ci.yml/dispatches"
            ),
            "dispatch_ref": "main",
            "run_ref": runtime.FINAL_MAIN_REF,
            "head_sha": HEAD,
            "inputs": dict(zip(runtime.CUDA_NONCE_INPUT_KEYS, NONCES)),
            "expected_runner_nonces": list(NONCES),
            "pre_dispatch_run_ids": [10],
            "request_sha256": "7" * 64,
            "mutation_retried": False,
        },
    )
    directory_receipt_sha256 = directory_receipt.receipt_sha256
    journal.close()
    reopened_directory = live.reopen_evidence_directory(
        evidence, expected_receipt_sha256=directory_receipt_sha256
    )
    reopened = driver.EvidenceJournal.reopen_for_recovery(
        reopened_directory,
        plan_sha256=CONTROL_SHA,
    )
    events: list[str] = []
    provider = _DriverProvider("baseline", events)
    recovered = driver.LiveReleaseDriver.resume_for_abort(
        _DriverController(provider, events),  # type: ignore[arg-type]
        provider,  # type: ignore[arg-type]
        plan,  # type: ignore[arg-type]
        reopened,
        sleep=lambda _: None,
        observation_poll_limit=1,
    )
    assert recovered._dispatch_ambiguity is not None
    assert recovered._session is None
    assert reopened.last_evidence_sha256 is not None
    reopened.close()


def test_final_acceptance_loader_rejects_minimal_forged_journal(tmp_path: Path) -> None:
    evidence = (tmp_path / "evidence").resolve()
    directory_receipt = live.create_evidence_directory(evidence)
    journal = driver.EvidenceJournal(
        directory_receipt,
        plan_sha256=CONTROL_SHA,
    )
    journal.record("immutable-plan", {"head_sha": HEAD})
    journal.record("final-main-acceptance", {})
    anchor = journal.record(
        "lifecycle-restored",
        {
            "plan_sha256": CONTROL_SHA,
            "provider_instances": 0,
            "provider_firewall_rulesets": 0,
            "global_firewall_restored": True,
            "repository_runners": 0,
            "known_hosts_sha256": "b" * 64,
        },
    )
    with pytest.raises(controller.ControllerError, match="immutable_plan_shape_rejected"):
        driver.EvidenceJournal.load_final_main_acceptance(
            directory_receipt,
            controller_resources=TEST_RESOURCES,
            final_control_plane_plan_sha256=CONTROL_SHA,
            final_journal_sha256=anchor,
        )
    journal.close()


@pytest.mark.parametrize(
    ("authority_fault", "error"),
    [
        (None, None),
        ("missing", "journal_event_order_rejected"),
        ("replay", "operator_app_inbox_generation_rejected"),
        ("stale", "operator_app_inbox_classification_order_rejected"),
        ("future", "operator_app_inbox_accepted_generation_binding_rejected"),
        ("dispatch-equal", "operator_app_inbox_classification_order_rejected"),
        ("authority-equal", "operator_app_inbox_classification_order_rejected"),
        ("plan-drift", "authority_capture_or_plan_binding"),
        ("acceptance-schema-bool", "acceptance_identity_rejected"),
        ("acceptance-attempt-bool", "acceptance_identity_rejected"),
        ("dispatch-attempt-bool", "recovery_session_binding_rejected"),
        ("lifecycle-int-bool", "final-main_lifecycle_restoration_rejected"),
        ("envelope-schema-bool", "acceptance_journal_chain_rejected"),
        ("envelope-sequence-bool", "acceptance_journal_chain_rejected"),
    ],
)
def test_final_acceptance_loader_verifies_full_runtime_and_restoration_chain(
    tmp_path: Path,
    authority_fault: str | None,
    error: str | None,
) -> None:
    immutable_plan = _operator_immutable_plan("final-main")
    control_sha256 = hashlib.sha256(
        json.dumps(
            immutable_plan,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    evidence = (tmp_path / "evidence").resolve()
    directory_receipt = live.create_evidence_directory(evidence)
    journal = driver.EvidenceJournal(
        directory_receipt,
        plan_sha256=control_sha256,
    )
    journal.record("immutable-plan", immutable_plan)
    provider_binding_sha256 = _record_successful_lifecycle_prefix(
        journal,
        control_sha256=control_sha256,
        phase="final-main",
        immutable_plan=immutable_plan,
    )
    session = _session()
    dispatch_intent, dispatch_settlement = _phase_dispatch_evidence(session)
    journal.record("github-dispatch-intent", dispatch_intent)
    journal.record("github-dispatch-settled", dispatch_settlement)
    dispatch_mapping = driver.LiveReleaseDriver._session_mapping(session)
    if authority_fault == "dispatch-attempt-bool":
        dispatch_mapping["run_attempt"] = True
    journal.record("github-dispatch", dispatch_mapping)

    previous_cleanup_sha256: str | None = None
    first_capture: controller.TrustedAppCapture | None = None
    first_archive: Mapping[str, Any] | None = None
    first_authority: controller.AuthorityReceipt | None = None
    for job_index, binding in enumerate(session.jobs):
        if job_index:
            _record_host_refresh(
                journal,
                control_sha256=control_sha256,
                ordinal=job_index,
                immutable_plan=immutable_plan,
            )
        if authority_fault == "replay" and job_index == 1:
            assert (
                first_capture is not None
                and first_archive is not None
                and first_authority is not None
            )
            app_capture = first_capture
            archive = first_archive
            authority = first_authority
        else:
            capture_time = NOW - timedelta(minutes=4, seconds=30) + timedelta(seconds=job_index)
            if authority_fault == "stale" and job_index == 1:
                capture_time = NOW - timedelta(minutes=20)
            elif authority_fault == "future" and job_index == 1:
                capture_time = NOW - timedelta(minutes=3)
            elif authority_fault == "dispatch-equal" and job_index == 1:
                capture_time = NOW - timedelta(minutes=5)
            elif authority_fault == "authority-equal" and job_index == 1:
                capture_time = NOW - timedelta(minutes=4) + timedelta(seconds=1)
            capture_mapping, source_pages = _app_capture(captured_at=capture_time)
            pages = dict(source_pages)
            evidence_item = capture_mapping["evidence"][0]
            filename = evidence_item["filename"]
            page = pages[filename] + f"fresh-job={job_index}\n".encode("ascii")
            pages[filename] = page
            evidence_item["bytes"] = len(page)
            evidence_item["sha256"] = hashlib.sha256(page).hexdigest()
            app_capture = controller.TrustedAppCapture.from_mapping(
                capture_mapping,
                resources=TEST_RESOURCES,
                evidence_reader=pages.__getitem__,
                now=(capture_time if authority_fault == "stale" else NOW),
            )
            archive = journal.archive_installed_app_capture(
                app_capture,
                pages.__getitem__,
            )
            authority = _authority_receipt_for_capture(
                app_capture,
                job_index=job_index,
            )
            if job_index == 0:
                first_capture = app_capture
                first_archive = archive
                first_authority = authority
        journal.record("installed-app-raw-archive", archive)
        journal.record("installed-app-authority", app_capture.to_mapping())
        if not (authority_fault == "missing" and job_index == 1):
            journal.record("authority-window", authority.evidence_mapping())

        runtime_plan = _valid_plan_for(
            control_plane_plan_sha256=control_sha256,
            job_index=job_index,
            previous_cleanup_receipt_sha256=previous_cleanup_sha256,
            authority_receipt=authority,
        )
        if authority_fault == "plan-drift" and job_index == 1:
            runtime_plan["authority_window"]["evidence_sha256"] = "f" * 64
        _record_job_mutation_wal(
            journal,
            session=session,
            binding=binding,
            runtime_plan=runtime_plan,
        )
        execution = _remote_execution(runtime_plan)
        remote_receipt_sha256 = controller.ReleaseGpuController._validate_remote_receipt(
            runtime_plan, execution
        )
        journal.record(
            "remote-cleanup",
            {
                "receipt": execution.receipt,
                "stdout_sha256": execution.stdout_sha256,
                "stderr_sha256": execution.stderr_sha256,
                "frame_receipt": execution.frame_receipt,
            },
        )
        previous_cleanup_sha256 = remote_receipt_sha256

        def digest(label: str) -> str:
            return hashlib.sha256(f"accepted:{job_index}:{label}".encode("ascii")).hexdigest()

        accepted_material = {
            "phase": "final-main",
            "run_id": session.run["id"],
            "job_key": binding.key,
            "job_id": binding.job_id,
            "runner_id": runtime_plan["job"]["runner_id"],
            "runner_name": binding.runner_name,
            "runtime_plan_sha256": runtime.runtime_plan_sha256(runtime_plan),
            "remote_receipt_sha256": remote_receipt_sha256,
            "actions_job_response_sha256": digest("actions-job"),
            "check_response_sha256": digest("check"),
            "log_sha256": hashlib.sha256(b"15 passed in 1.00s\n").hexdigest(),
            "pytest_passed": 15,
            "pytest_skipped": 0,
            "runner_inventory_response_sha256": digest("runner-inventory"),
            "post_execution_observation_sha256": digest("post-observation"),
        }
        accepted = controller.AcceptedJobReceipt(
            **accepted_material,
            evidence_sha256=hashlib.sha256(_canonical(accepted_material)).hexdigest(),
        )
        session.accepted[binding.key] = accepted
        journal.record("accepted-actions-job", accepted.to_mapping())

    settlement_material = {
        "phase": "final-main",
        "run_id": session.run["id"],
        "run_attempt": 1,
        "head_sha": session.head_sha,
        "accepted_cuda_runner_nonces": [item.nonce for item in session.jobs],
        "job_evidence_sha256": [
            session.accepted[item.key].evidence_sha256 for item in session.jobs
        ],
        "all_four_jobs_15_of_15_zero_skips": True,
        "rerun_performed": False,
    }
    settlement = {
        **settlement_material,
        "evidence_sha256": hashlib.sha256(_canonical(settlement_material)).hexdigest(),
    }
    service = controller.ReleaseGpuController(
        QueueTransport(), NoRemote(), resources=TEST_RESOURCES, clock=lambda: NOW
    )
    acceptance = service.seal_final_main_acceptance(session, settlement)
    acceptance_mapping = acceptance.to_mapping()
    if authority_fault == "acceptance-schema-bool":
        acceptance_mapping["schema_version"] = True
    elif authority_fault == "acceptance-attempt-bool":
        acceptance_mapping["run_attempt"] = True
    if authority_fault in {"acceptance-schema-bool", "acceptance-attempt-bool"}:
        acceptance_material = dict(acceptance_mapping)
        acceptance_material.pop("evidence_sha256")
        acceptance_mapping["evidence_sha256"] = hashlib.sha256(
            _canonical(acceptance_material)
        ).hexdigest()
    journal.record("final-main-acceptance", acceptance_mapping)
    journal.record("phase-settlement", settlement)
    anchor = _record_successful_lifecycle_teardown(
        journal,
        phase="final-main",
        control_sha256=control_sha256,
        callback_binding_sha256=provider_binding_sha256,
        known_hosts_sha256="b" * 64,
        lifecycle_overrides=(
            {"provider_instances": False} if authority_fault == "lifecycle-int-bool" else None
        ),
    )
    if authority_fault in {"envelope-schema-bool", "envelope-sequence-bool"}:

        def mutate_envelope(sequence: int, envelope: dict[str, Any]) -> None:
            if sequence == 1:
                envelope[
                    "schema_version" if authority_fault == "envelope-schema-bool" else "sequence"
                ] = True

        anchor = _rewrite_journal_chain(evidence, mutate_envelope)
    if error is not None:
        with pytest.raises(controller.ControllerError, match=error):
            driver.EvidenceJournal.load_final_main_acceptance(
                directory_receipt,
                controller_resources=TEST_RESOURCES,
                final_control_plane_plan_sha256=control_sha256,
                final_journal_sha256=anchor,
            )
        journal.close()
        return
    loaded = driver.EvidenceJournal.load_final_main_acceptance(
        directory_receipt,
        controller_resources=TEST_RESOURCES,
        final_control_plane_plan_sha256=control_sha256,
        final_journal_sha256=anchor,
    )
    assert loaded.to_mapping() == acceptance.to_mapping()
    provenance = loaded._journal_provenance_mapping()
    assert provenance["final_journal_sha256"] == anchor
    assert provenance["evidence_directory_receipt_sha256"] == directory_receipt.receipt_sha256
    assert len(provenance["authority_evidence_identities"]) == 4

    class LoadedAcceptanceController(controller.ReleaseGpuController):
        def _accepted_final_cuda_nonces(
            self,
            run_id: int,
            head_sha: str,
        ) -> tuple[tuple[str, ...], str]:
            assert run_id == 500 and head_sha == HEAD
            return NONCES, "1" * 64

        def _attempt_jobs(self, run_id: int, attempt: int) -> list[Mapping[str, Any]]:
            assert run_id == 500 and attempt == 1
            return [
                {
                    "id": item["job_id"],
                    "name": runtime.JOB_SPECS[item["job_key"]]["name"],
                    "head_sha": HEAD,
                    "run_attempt": 1,
                    "status": "completed",
                    "conclusion": "success",
                    "labels": [item["runner_name"]],
                    "runner_id": item["runner_id"],
                    "runner_name": item["runner_name"],
                }
                for item in loaded.jobs
            ]

        def _request_json(
            self,
            method: str,
            path: str,
            *,
            body: Mapping[str, Any] | None = None,
            expected: int = 200,
        ) -> tuple[Any, str]:
            assert method == "GET" and body is None and expected == 200
            selected = next(
                item
                for item in loaded.jobs
                if controller.job_spec_name_for_check(
                    str(runtime.JOB_SPECS[item["job_key"]]["name"])
                )
                in path
            )
            return (
                {
                    "total_count": 1,
                    "check_runs": [
                        {
                            "name": runtime.JOB_SPECS[selected["job_key"]]["name"],
                            "head_sha": HEAD,
                            "status": "completed",
                            "conclusion": "success",
                            "app": {"id": controller.CHECKS_APP_ID},
                            "details_url": f"https://github.com/job/{selected['job_id']}",
                        }
                    ],
                },
                "2" * 64,
            )

        def _request(
            self,
            method: str,
            path: str,
            *,
            body: Mapping[str, Any] | None = None,
            expected: int,
        ) -> controller.GitHubResponse:
            assert method == "GET" and path.endswith("/logs") and body is None and expected == 200
            return controller.GitHubResponse(
                method,
                path,
                200,
                bytearray(b"15 passed in 1.00s\n"),
                "3" * 64,
            )

        def _nonce_history(
            self,
            nonces: list[str],
            *,
            exclude: tuple[int, int, int] | None = None,
            allowed_active_job_ids: set[int] | None = None,
        ) -> dict[str, Any]:
            return {
                "observed_at": controller._iso(NOW),
                "response_sha256": hashlib.sha256(_canonical(nonces)).hexdigest(),
                "historical_match_count": 0,
                "unexpected_queued_or_in_progress_count": 0,
            }

    live_service = LoadedAcceptanceController(
        QueueTransport(),
        NoRemote(),
        resources=TEST_RESOURCES,
        clock=lambda: NOW,
    )
    live_service._revalidate_final_main_acceptance(
        loaded,
        run_id=500,
        head_sha=HEAD,
    )
    final_capture_payload = next(
        payload
        for label, payload in journal.verified_entries()
        if label == "installed-app-authority"
    )
    final_capture_sha = final_capture_payload["evidence_sha256"]
    final_capture = controller.TrustedAppCapture.from_mapping(
        final_capture_payload["normalized_capture"],
        resources=TEST_RESOURCES,
        evidence_reader=lambda filename: driver.EvidenceJournal._read_archived_app_page(
            evidence,
            final_capture_sha,
            filename,
        ),
        now=NOW,
    )
    with pytest.raises(controller.ControllerError, match="app_capture_replayed"):
        live_service.capture_authority(
            _publication_session(),
            final_capture,
            installed_app_evidence_reader=lambda filename: (
                driver.EvidenceJournal._read_archived_app_page(
                    evidence,
                    final_capture_sha,
                    filename,
                )
            ),
        )
    journal.close()


def _build_publication_recovery_journal(
    tmp_path: Path,
    *,
    job_order: tuple[int, ...] = (0, 1),
    settlement_drift: bool = False,
    lifecycle_drift: bool = False,
    lifecycle_boolean_drift: bool = False,
    post_anchor_label: str | None = None,
    authority_fault: str | None = None,
) -> tuple[
    live.EvidenceDirectoryReceipt,
    driver.EvidenceJournal,
    str,
    str,
    str,
]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    immutable_plan = _operator_immutable_plan("publication")
    control_sha256 = hashlib.sha256(
        json.dumps(
            immutable_plan,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    evidence = (tmp_path / "publication-evidence").resolve()
    directory_receipt = live.create_evidence_directory(evidence)
    journal = driver.EvidenceJournal(
        directory_receipt,
        plan_sha256=control_sha256,
    )
    journal.record("immutable-plan", immutable_plan)
    provider_binding_sha256 = _record_successful_lifecycle_prefix(
        journal,
        control_sha256=control_sha256,
        phase="publication",
        immutable_plan=immutable_plan,
    )
    session = _publication_session()
    if authority_fault in {
        "prior-capture-replay",
        "prior-authority-replay",
        "prior-archive-replay",
        "prior-page-replay",
    }:
        preview_mapping, preview_pages_source = _app_capture(
            captured_at=NOW - timedelta(minutes=4, seconds=30)
        )
        preview_pages = dict(preview_pages_source)
        preview_item = preview_mapping["evidence"][0]
        preview_filename = preview_item["filename"]
        preview_page = preview_pages[preview_filename] + b"publication-job=0\n"
        preview_pages[preview_filename] = preview_page
        preview_item["bytes"] = len(preview_page)
        preview_item["sha256"] = hashlib.sha256(preview_page).hexdigest()
        preview_capture = controller.TrustedAppCapture.from_mapping(
            preview_mapping,
            resources=TEST_RESOURCES,
            evidence_reader=preview_pages.__getitem__,
            now=NOW,
        )
        preview_authority = _authority_receipt_for_capture(
            preview_capture,
            job_index=0,
        )
        preview_archive_material = {
            "capture_evidence_sha256": preview_capture.evidence_sha256,
            "archive_directory": (f"installed-app-pages/{preview_capture.evidence_sha256}"),
            "files": [
                {
                    "filename": item["filename"],
                    "bytes": item["bytes"],
                    "sha256": item["sha256"],
                }
                for item in preview_capture.normalized_capture["evidence"]
            ],
            "all_pages_exclusive_single_link": True,
        }
        prior = [dict(item) for item in session.prior_authority_evidence_identities]
        replayed = dict(prior[0])
        if authority_fault == "prior-capture-replay":
            replayed["capture_evidence_sha256"] = preview_capture.evidence_sha256
        elif authority_fault == "prior-authority-replay":
            replayed["authority_evidence_sha256"] = preview_authority.evidence_sha256
        elif authority_fault == "prior-archive-replay":
            replayed["archive_evidence_sha256"] = hashlib.sha256(
                _canonical(preview_archive_material)
            ).hexdigest()
        else:
            replayed["raw_page_sha256"] = [
                item["sha256"] for item in preview_capture.normalized_capture["evidence"]
            ]
        replayed_material = dict(replayed)
        replayed_material.pop("evidence_sha256")
        replayed["evidence_sha256"] = hashlib.sha256(_canonical(replayed_material)).hexdigest()
        prior[0] = replayed
        session.prior_authority_evidence_identities = tuple(prior)
    dispatch_intent, dispatch_settlement = _phase_dispatch_evidence(session)
    journal.record("github-dispatch-intent", dispatch_intent)
    journal.record("github-dispatch-settled", dispatch_settlement)
    journal.record("github-dispatch", driver.LiveReleaseDriver._session_mapping(session))

    accepted_receipts: list[controller.AcceptedJobReceipt] = []
    previous_cleanup_sha256: str | None = None
    first_capture: controller.TrustedAppCapture | None = None
    first_archive: Mapping[str, Any] | None = None
    first_authority: controller.AuthorityReceipt | None = None
    for group_ordinal, job_index in enumerate(job_order):
        binding = session.jobs[job_index]
        if group_ordinal:
            _record_host_refresh(
                journal,
                control_sha256=control_sha256,
                ordinal=group_ordinal,
                immutable_plan=immutable_plan,
            )
        if authority_fault == "replay" and group_ordinal == 1:
            assert (
                first_capture is not None
                and first_archive is not None
                and first_authority is not None
            )
            app_capture = first_capture
            archive = first_archive
            authority = first_authority
        else:
            capture_time = NOW - timedelta(minutes=4, seconds=30) + timedelta(seconds=group_ordinal)
            if authority_fault == "stale" and group_ordinal == 1:
                capture_time = NOW - timedelta(minutes=20)
            elif authority_fault == "future" and group_ordinal == 1:
                capture_time = NOW - timedelta(minutes=3)
            elif authority_fault == "dispatch-equal" and group_ordinal == 1:
                capture_time = NOW - timedelta(minutes=5)
            elif authority_fault == "dispatch-old" and group_ordinal == 1:
                capture_time = NOW - timedelta(minutes=5, seconds=1)
            elif authority_fault == "authority-equal" and group_ordinal == 1:
                capture_time = NOW - timedelta(minutes=4) + timedelta(seconds=1)
            capture_mapping, source_pages = _app_capture(captured_at=capture_time)
            pages = dict(source_pages)
            evidence_item = capture_mapping["evidence"][0]
            filename = evidence_item["filename"]
            page = pages[filename] + f"publication-job={group_ordinal}\n".encode("ascii")
            pages[filename] = page
            evidence_item["bytes"] = len(page)
            evidence_item["sha256"] = hashlib.sha256(page).hexdigest()
            app_capture = controller.TrustedAppCapture.from_mapping(
                capture_mapping,
                resources=TEST_RESOURCES,
                evidence_reader=pages.__getitem__,
                now=(capture_time if authority_fault == "stale" else NOW),
            )
            archive = journal.archive_installed_app_capture(app_capture, pages.__getitem__)
            authority = _authority_receipt_for_capture(
                app_capture,
                job_index=group_ordinal,
            )
            if group_ordinal == 0:
                first_capture = app_capture
                first_archive = archive
                first_authority = authority
        journal.record("installed-app-raw-archive", archive)
        journal.record("installed-app-authority", app_capture.to_mapping())
        if not (authority_fault == "missing" and group_ordinal == 1):
            journal.record("authority-window", authority.evidence_mapping())

        runtime_plan = _valid_plan_for(
            control_plane_plan_sha256=control_sha256,
            job_index=job_index,
            previous_cleanup_receipt_sha256=(None if job_index == 0 else previous_cleanup_sha256),
            session=session,
            authority_receipt=authority,
        )
        if authority_fault == "plan-drift" and group_ordinal == 1:
            runtime_plan["authority_window"]["evidence_sha256"] = "f" * 64
        _record_job_mutation_wal(
            journal,
            session=session,
            binding=binding,
            runtime_plan=runtime_plan,
        )
        execution = _remote_execution(runtime_plan)
        remote_receipt_sha256 = controller.ReleaseGpuController._validate_remote_receipt(
            runtime_plan, execution
        )
        journal.record(
            "remote-cleanup",
            {
                "receipt": execution.receipt,
                "stdout_sha256": execution.stdout_sha256,
                "stderr_sha256": execution.stderr_sha256,
                "frame_receipt": execution.frame_receipt,
            },
        )
        previous_cleanup_sha256 = remote_receipt_sha256

        def digest(label: str) -> str:
            return hashlib.sha256(
                f"publication:{group_ordinal}:{label}".encode("ascii")
            ).hexdigest()

        accepted_material = {
            "phase": "publication",
            "run_id": session.run["id"],
            "job_key": binding.key,
            "job_id": binding.job_id,
            "runner_id": runtime_plan["job"]["runner_id"],
            "runner_name": binding.runner_name,
            "runtime_plan_sha256": runtime.runtime_plan_sha256(runtime_plan),
            "remote_receipt_sha256": remote_receipt_sha256,
            "actions_job_response_sha256": digest("actions-job"),
            "check_response_sha256": digest("check"),
            "log_sha256": digest("log"),
            "pytest_passed": 15,
            "pytest_skipped": 0,
            "runner_inventory_response_sha256": digest("runner-inventory"),
            "post_execution_observation_sha256": digest("post-observation"),
        }
        accepted = controller.AcceptedJobReceipt(
            **accepted_material,
            evidence_sha256=hashlib.sha256(_canonical(accepted_material)).hexdigest(),
        )
        accepted_receipts.append(accepted)
        journal.record("accepted-actions-job", accepted.to_mapping())

    settlement_material = {
        "phase": "publication",
        "run_id": session.run["id"],
        "run_attempt": 1,
        "head_sha": session.head_sha,
        "tag": runtime.PUBLICATION_TAG,
        "stage_recovery_drill": True,
        "job_evidence_sha256": [item.evidence_sha256 for item in accepted_receipts],
        "runner_inventory_response_sha256": "9" * 64,
        "both_release_jobs_15_of_15_zero_skips": True,
        "workflow_publication_success_not_claimed": True,
        "rerun_performed": False,
    }
    if settlement_drift:
        settlement_material["stage_recovery_drill"] = False
    settlement = {
        **settlement_material,
        "evidence_sha256": hashlib.sha256(_canonical(settlement_material)).hexdigest(),
    }
    settlement_anchor = journal.record("phase-settlement", settlement)
    lifecycle_overrides: dict[str, Any] = {}
    if lifecycle_drift:
        lifecycle_overrides["repository_runners"] = 1
    if lifecycle_boolean_drift:
        lifecycle_overrides["provider_instances"] = False
    anchor = _record_successful_lifecycle_teardown(
        journal,
        phase="publication",
        control_sha256=control_sha256,
        callback_binding_sha256=provider_binding_sha256,
        known_hosts_sha256="e" * 64,
        lifecycle_overrides=lifecycle_overrides,
    )
    if post_anchor_label is not None:
        journal.record(post_anchor_label, {"unexpected": True})
    return directory_receipt, journal, control_sha256, anchor, settlement_anchor


def _recovery_dispatch_intent(
    source: driver.PublicationRecoverySource,
    *,
    nonce: str = "2" * 16,
    pre_dispatch_run_ids: tuple[int, ...] = (),
    pre_dispatch_request_nonces: tuple[str, ...] = (),
) -> dict[str, Any]:
    if pre_dispatch_run_ids and not pre_dispatch_request_nonces:
        pre_dispatch_request_nonces = tuple(
            f"{run_id:016x}"[-16:] for run_id in pre_dispatch_run_ids
        )
    assert len(pre_dispatch_run_ids) == len(pre_dispatch_request_nonces)
    request_path = (
        f"/repos/{controller.REPOSITORY}/actions/workflows/"
        f"{controller.RECOVERY_WORKFLOW}/dispatches"
    )
    request_body = {
        "ref": runtime.PUBLICATION_TAG,
        "inputs": {
            "tag": runtime.PUBLICATION_TAG,
            "source_run_id": str(source.run_id),
            "recovery_request_nonce": nonce,
            "require_staged_drill": True,
        },
    }
    return controller.ReleaseGpuController._validate_recovery_dispatch_intent(
        {
            "schema_version": 1,
            "kind": "explainiverse-recovery-dispatch-intent",
            "repository": controller.REPOSITORY,
            "workflow": controller.RECOVERY_WORKFLOW,
            "workflow_path": controller.RECOVERY_WORKFLOW_PATH,
            "ref": runtime.PUBLICATION_TAG,
            "head_sha": source.head_sha,
            "tag": runtime.PUBLICATION_TAG,
            "source_run_id": source.run_id,
            "require_staged_drill": True,
            "recovery_request_nonce": nonce,
            "display_title": controller.ReleaseGpuController._recovery_display_title(
                runtime.PUBLICATION_TAG, source.run_id, nonce
            ),
            "request_path": request_path,
            "request_body": request_body,
            "request_sha256": hashlib.sha256(
                _canonical(
                    {
                        "method": "POST",
                        "path": request_path,
                        "body": request_body,
                    }
                )
            ).hexdigest(),
            "workflow_response_sha256": "1" * 64,
            "immutable_source_evidence_sha256": "2" * 64,
            "source_run_evidence_sha256": "3" * 64,
            "pre_dispatch_run_ids": list(pre_dispatch_run_ids),
            "pre_dispatch_runs": [
                {
                    "id": run_id,
                    "display_title": controller.ReleaseGpuController._recovery_display_title(
                        runtime.PUBLICATION_TAG,
                        source.run_id,
                        prior_nonce,
                    ),
                    "head_sha": source.head_sha,
                    "head_branch": runtime.PUBLICATION_TAG,
                    "run_attempt": 1,
                    "status": "completed",
                    "conclusion": "failure",
                    "actor": controller.OWNER,
                    "triggering_actor": controller.OWNER,
                    "recovery_request_nonce": prior_nonce,
                }
                for run_id, prior_nonce in zip(
                    pre_dispatch_run_ids,
                    pre_dispatch_request_nonces,
                )
            ],
            "pre_dispatch_history_sha256": "4" * 64,
            "mutation_retried": False,
        }
    )


def _recovery_dispatch_receipt(
    intent: Mapping[str, Any],
    *,
    run_id: int = 900,
    mutation_response_received: bool = True,
) -> controller.RecoveryDispatchReceipt:
    material = {
        "observed_at": controller._iso(NOW),
        "tag": intent["tag"],
        "head_sha": intent["head_sha"],
        "source_run_id": intent["source_run_id"],
        "require_staged_drill": True,
        "recovery_request_nonce": intent["recovery_request_nonce"],
        "display_title": intent["display_title"],
        "run_id": run_id,
        "run_attempt": 1,
        "status": "queued",
        "conclusion": None,
        "request_sha256": intent["request_sha256"],
        "mutation_response_received": mutation_response_received,
        "mutation_response_sha256": ("5" * 64 if mutation_response_received else None),
        "workflow_response_sha256": intent["workflow_response_sha256"],
        "immutable_source_evidence_sha256": intent["immutable_source_evidence_sha256"],
        "source_run_evidence_sha256": intent["source_run_evidence_sha256"],
        "pre_dispatch_history_sha256": intent["pre_dispatch_history_sha256"],
        "reconciliation_sha256": "6" * 64,
    }
    return controller.RecoveryDispatchReceipt.from_mapping(
        {
            **material,
            "evidence_sha256": hashlib.sha256(_canonical(material)).hexdigest(),
        }
    )


def test_publication_recovery_source_loader_verifies_full_chain_and_marker_resume(
    tmp_path: Path,
) -> None:
    receipt, journal, control_sha256, anchor, _ = _build_publication_recovery_journal(tmp_path)
    source = driver.EvidenceJournal.load_publication_recovery_source(
        receipt,
        controller_resources=TEST_RESOURCES,
        publication_control_plane_plan_sha256=control_sha256,
        publication_journal_sha256=anchor,
        source_run_id=700,
    )
    assert source.head_sha == HEAD
    assert source.run_id == 700
    assert source.run_attempt == 1
    assert source.tag == runtime.PUBLICATION_TAG
    assert source.publication_journal_sha256 == anchor
    assert source.evidence_directory_receipt_sha256 == receipt.receipt_sha256
    assert len(source.job_evidence_sha256) == 2
    assert len(source.evidence_sha256) == 64
    assert source.recovery_tail.state == "source-unrecorded"
    with pytest.raises(TypeError, match="verified journal"):
        driver.PublicationRecoverySource()
    with pytest.raises(TypeError, match="verified journal"):
        driver.PublicationRecoveryTail()

    journal.record("operator-publication-recovery-source", source.to_mapping())
    resumed = driver.EvidenceJournal.load_publication_recovery_source(
        receipt,
        controller_resources=TEST_RESOURCES,
        publication_control_plane_plan_sha256=control_sha256,
        publication_journal_sha256=anchor,
        source_run_id=700,
    )
    assert resumed.to_mapping() == source.to_mapping()
    assert resumed.recovery_tail.state == "complete"
    assert resumed.recovery_tail.completed_run_ids == ()

    intent = _recovery_dispatch_intent(resumed)
    journal.record("github-recovery-dispatch-intent", intent)
    pending = driver.EvidenceJournal.load_publication_recovery_source(
        receipt,
        controller_resources=TEST_RESOURCES,
        publication_control_plane_plan_sha256=control_sha256,
        publication_journal_sha256=anchor,
        source_run_id=700,
    )
    assert pending.recovery_tail.state == "pending-intent"
    assert pending.recovery_tail.pending_intent == intent

    recovery_receipt = _recovery_dispatch_receipt(intent)
    journal.record("github-recovery-dispatch-settled", recovery_receipt.to_mapping())
    pending_operator = driver.EvidenceJournal.load_publication_recovery_source(
        receipt,
        controller_resources=TEST_RESOURCES,
        publication_control_plane_plan_sha256=control_sha256,
        publication_journal_sha256=anchor,
        source_run_id=700,
    )
    assert pending_operator.recovery_tail.state == "pending-operator-settlement"
    expected_operator_settlement = (
        driver.EvidenceJournal.build_publication_recovery_operator_settlement(
            pending_operator, recovery_receipt
        )
    )
    assert (
        pending_operator.recovery_tail.pending_operator_settlement == expected_operator_settlement
    )
    assert expected_operator_settlement["mode"] == "mutation-response-observed"

    journal.record(
        "operator-release-recovery-dispatch-settled",
        expected_operator_settlement,
    )
    completed = driver.EvidenceJournal.load_publication_recovery_source(
        receipt,
        controller_resources=TEST_RESOURCES,
        publication_control_plane_plan_sha256=control_sha256,
        publication_journal_sha256=anchor,
        source_run_id=700,
    )
    assert completed.recovery_tail.state == "complete"
    assert completed.recovery_tail.completed_run_ids == (900,)
    assert completed.recovery_tail.completed_request_nonces == ("2" * 16,)
    assert completed.recovery_tail.last_operator_settlement == expected_operator_settlement

    retry_intent = _recovery_dispatch_intent(
        completed,
        nonce="7" * 16,
        pre_dispatch_run_ids=(900,),
        pre_dispatch_request_nonces=("2" * 16,),
    )
    journal.record("github-recovery-dispatch-intent", retry_intent)
    retry_pending = driver.EvidenceJournal.load_publication_recovery_source(
        receipt,
        controller_resources=TEST_RESOURCES,
        publication_control_plane_plan_sha256=control_sha256,
        publication_journal_sha256=anchor,
        source_run_id=700,
    )
    assert retry_pending.recovery_tail.state == "pending-intent"
    assert retry_pending.recovery_tail.pending_intent == retry_intent
    journal.close()


def test_publication_recovery_source_loader_rejects_wrong_source_or_anchor(
    tmp_path: Path,
) -> None:
    receipt, journal, control_sha256, anchor, settlement_anchor = (
        _build_publication_recovery_journal(tmp_path)
    )
    with pytest.raises(controller.ControllerError, match="dispatch_binding"):
        driver.EvidenceJournal.load_publication_recovery_source(
            receipt,
            controller_resources=TEST_RESOURCES,
            publication_control_plane_plan_sha256=control_sha256,
            publication_journal_sha256=anchor,
            source_run_id=701,
        )
    with pytest.raises(controller.ControllerError, match="anchor_not_lifecycle"):
        driver.EvidenceJournal.load_publication_recovery_source(
            receipt,
            controller_resources=TEST_RESOURCES,
            publication_control_plane_plan_sha256=control_sha256,
            publication_journal_sha256=settlement_anchor,
            source_run_id=700,
        )
    journal.close()


@pytest.mark.parametrize(
    ("job_order", "error"),
    [
        ((0,), "host_refresh_observation_count"),
        ((0, 1, 1), "journal_event_order"),
        ((1, 0), "pre_jit_runner_absence_rejected"),
    ],
)
def test_publication_recovery_source_loader_rejects_missing_extra_or_reordered_jobs(
    tmp_path: Path,
    job_order: tuple[int, ...],
    error: str,
) -> None:
    receipt, journal, control_sha256, anchor, _ = _build_publication_recovery_journal(
        tmp_path,
        job_order=job_order,
    )
    with pytest.raises(controller.ControllerError, match=error):
        driver.EvidenceJournal.load_publication_recovery_source(
            receipt,
            controller_resources=TEST_RESOURCES,
            publication_control_plane_plan_sha256=control_sha256,
            publication_journal_sha256=anchor,
            source_run_id=700,
        )
    journal.close()


@pytest.mark.parametrize(
    (
        "settlement_drift",
        "lifecycle_drift",
        "lifecycle_boolean_drift",
        "error",
    ),
    [
        (True, False, False, "settlement_binding"),
        (False, True, False, "lifecycle_restoration"),
        (False, False, True, "lifecycle_restoration"),
    ],
)
def test_publication_recovery_source_loader_rejects_settlement_or_lifecycle_drift(
    tmp_path: Path,
    settlement_drift: bool,
    lifecycle_drift: bool,
    lifecycle_boolean_drift: bool,
    error: str,
) -> None:
    receipt, journal, control_sha256, anchor, _ = _build_publication_recovery_journal(
        tmp_path,
        settlement_drift=settlement_drift,
        lifecycle_drift=lifecycle_drift,
        lifecycle_boolean_drift=lifecycle_boolean_drift,
    )
    with pytest.raises(controller.ControllerError, match=error):
        driver.EvidenceJournal.load_publication_recovery_source(
            receipt,
            controller_resources=TEST_RESOURCES,
            publication_control_plane_plan_sha256=control_sha256,
            publication_journal_sha256=anchor,
            source_run_id=700,
        )
    journal.close()


@pytest.mark.parametrize(
    ("authority_fault", "error"),
    [
        ("missing", "journal_event_order_rejected"),
        ("replay", "operator_app_inbox_generation_rejected"),
        ("stale", "operator_app_inbox_classification_order_rejected"),
        ("future", "operator_app_inbox_accepted_generation_binding_rejected"),
        ("dispatch-equal", "operator_app_inbox_classification_order_rejected"),
        ("dispatch-old", "operator_app_inbox_classification_order_rejected"),
        ("authority-equal", "authority_capture_freshness"),
        ("plan-drift", "authority_capture_or_plan_binding"),
        ("prior-capture-replay", "authority_evidence_replayed"),
        ("prior-authority-replay", "authority_evidence_replayed"),
        ("prior-archive-replay", "authority_evidence_replayed"),
        ("prior-page-replay", "authority_evidence_replayed"),
    ],
)
def test_publication_recovery_source_loader_rejects_authority_evidence_drift(
    tmp_path: Path,
    authority_fault: str,
    error: str,
) -> None:
    receipt, journal, control_sha256, anchor, _ = _build_publication_recovery_journal(
        tmp_path,
        authority_fault=authority_fault,
    )
    with pytest.raises(controller.ControllerError, match=error):
        driver.EvidenceJournal.load_publication_recovery_source(
            receipt,
            controller_resources=TEST_RESOURCES,
            publication_control_plane_plan_sha256=control_sha256,
            publication_journal_sha256=anchor,
            source_run_id=700,
        )
    journal.close()


@pytest.mark.parametrize(
    ("fault", "error"),
    [
        ("envelope-schema-bool", "journal_chain_rejected"),
        ("envelope-sequence-bool", "journal_chain_rejected"),
        ("dispatch-attempt-bool", "recovery_session_binding_rejected"),
        ("lifecycle-int-bool", "lifecycle_restoration_rejected"),
    ],
)
def test_publication_recovery_source_loader_rejects_json_boolean_integer_aliases(
    tmp_path: Path,
    fault: str,
    error: str,
) -> None:
    receipt, journal, control_sha256, _, _ = _build_publication_recovery_journal(tmp_path)

    def mutate(sequence: int, envelope: dict[str, Any]) -> None:
        if fault == "envelope-schema-bool" and sequence == 1:
            envelope["schema_version"] = True
        elif fault == "envelope-sequence-bool" and sequence == 1:
            envelope["sequence"] = True
        elif fault == "dispatch-attempt-bool" and envelope["label"] == "github-dispatch":
            envelope["payload"]["run_attempt"] = True
        elif fault == "lifecycle-int-bool" and envelope["label"] == "lifecycle-restored":
            envelope["payload"]["provider_instances"] = False

    rewritten_anchor = _rewrite_journal_chain(Path(receipt.absolute_path), mutate)
    with pytest.raises(controller.ControllerError, match=error):
        driver.EvidenceJournal.load_publication_recovery_source(
            receipt,
            controller_resources=TEST_RESOURCES,
            publication_control_plane_plan_sha256=control_sha256,
            publication_journal_sha256=rewritten_anchor,
            source_run_id=700,
        )
    journal.close()


def test_publication_recovery_source_loader_rejects_post_anchor_label_or_marker_drift(
    tmp_path: Path,
) -> None:
    receipt, journal, control_sha256, anchor, _ = _build_publication_recovery_journal(
        tmp_path,
        post_anchor_label="unexpected-post-restoration",
    )
    with pytest.raises(controller.ControllerError, match="post_restoration_event"):
        driver.EvidenceJournal.load_publication_recovery_source(
            receipt,
            controller_resources=TEST_RESOURCES,
            publication_control_plane_plan_sha256=control_sha256,
            publication_journal_sha256=anchor,
            source_run_id=700,
        )
    journal.close()

    receipt, journal, control_sha256, anchor, _ = _build_publication_recovery_journal(
        tmp_path / "marker"
    )
    journal.record(
        "operator-publication-recovery-source",
        {
            "schema_version": 1,
            "kind": "explainiverse-publication-recovery-source",
            "evidence_sha256": "f" * 64,
        },
    )
    with pytest.raises(controller.ControllerError, match="source_marker"):
        driver.EvidenceJournal.load_publication_recovery_source(
            receipt,
            controller_resources=TEST_RESOURCES,
            publication_control_plane_plan_sha256=control_sha256,
            publication_journal_sha256=anchor,
            source_run_id=700,
        )
    journal.close()


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("marker-after-intent", "source_marker"),
        ("orphan-controller-settlement", "suffix_order"),
        ("duplicate-marker", "suffix_order"),
        ("operator-before-intent", "suffix_order"),
        ("operator-settlement-drift", "operator_settlement_binding"),
        ("nonce-reuse", "suffix_intent_binding"),
        ("run-id-reuse", "receipt_intent_binding"),
        ("prehistory-drift", "suffix_intent_binding"),
        ("duplicate-operator-settlement", "suffix_order"),
    ],
)
def test_publication_recovery_source_loader_rejects_malformed_suffix_state(
    tmp_path: Path,
    case: str,
    error: str,
) -> None:
    receipt, journal, control_sha256, anchor, _ = _build_publication_recovery_journal(tmp_path)
    source = driver.EvidenceJournal.load_publication_recovery_source(
        receipt,
        controller_resources=TEST_RESOURCES,
        publication_control_plane_plan_sha256=control_sha256,
        publication_journal_sha256=anchor,
        source_run_id=700,
    )
    intent = _recovery_dispatch_intent(source)
    recovery_receipt = _recovery_dispatch_receipt(intent)
    operator_settlement = driver.EvidenceJournal.build_publication_recovery_operator_settlement(
        source, recovery_receipt
    )
    if case == "marker-after-intent":
        journal.record("github-recovery-dispatch-intent", intent)
        journal.record("operator-publication-recovery-source", source.to_mapping())
    else:
        journal.record("operator-publication-recovery-source", source.to_mapping())
        if case == "orphan-controller-settlement":
            journal.record("github-recovery-dispatch-settled", recovery_receipt.to_mapping())
        elif case == "duplicate-marker":
            journal.record("operator-publication-recovery-source", source.to_mapping())
        elif case == "operator-before-intent":
            journal.record(
                "operator-release-recovery-dispatch-settled",
                operator_settlement,
            )
        elif case == "operator-settlement-drift":
            journal.record("github-recovery-dispatch-intent", intent)
            journal.record("github-recovery-dispatch-settled", recovery_receipt.to_mapping())
            drifted = dict(operator_settlement)
            drifted["recovery_run_id"] = 901
            journal.record("operator-release-recovery-dispatch-settled", drifted)
        elif case in {"nonce-reuse", "run-id-reuse"}:
            journal.record("github-recovery-dispatch-intent", intent)
            journal.record("github-recovery-dispatch-settled", recovery_receipt.to_mapping())
            journal.record(
                "operator-release-recovery-dispatch-settled",
                operator_settlement,
            )
            second_nonce = "2" * 16 if case == "nonce-reuse" else "7" * 16
            second_intent = _recovery_dispatch_intent(
                source,
                nonce=second_nonce,
                pre_dispatch_run_ids=(900,),
                pre_dispatch_request_nonces=("2" * 16,),
            )
            journal.record("github-recovery-dispatch-intent", second_intent)
            if case == "run-id-reuse":
                journal.record(
                    "github-recovery-dispatch-settled",
                    _recovery_dispatch_receipt(
                        second_intent,
                        run_id=900,
                        mutation_response_received=False,
                    ).to_mapping(),
                )
        elif case == "prehistory-drift":
            journal.record(
                "github-recovery-dispatch-intent",
                _recovery_dispatch_intent(
                    source,
                    pre_dispatch_run_ids=(899,),
                ),
            )
        elif case == "duplicate-operator-settlement":
            journal.record("github-recovery-dispatch-intent", intent)
            journal.record("github-recovery-dispatch-settled", recovery_receipt.to_mapping())
            journal.record(
                "operator-release-recovery-dispatch-settled",
                operator_settlement,
            )
            journal.record(
                "operator-release-recovery-dispatch-settled",
                operator_settlement,
            )
        else:
            raise AssertionError(case)
    with pytest.raises(controller.ControllerError, match=error):
        driver.EvidenceJournal.load_publication_recovery_source(
            receipt,
            controller_resources=TEST_RESOURCES,
            publication_control_plane_plan_sha256=control_sha256,
            publication_journal_sha256=anchor,
            source_run_id=700,
        )
    journal.close()


def test_publication_recovery_operator_settlement_derives_response_loss_mode(
    tmp_path: Path,
) -> None:
    receipt, journal, control_sha256, anchor, _ = _build_publication_recovery_journal(tmp_path)
    source = driver.EvidenceJournal.load_publication_recovery_source(
        receipt,
        controller_resources=TEST_RESOURCES,
        publication_control_plane_plan_sha256=control_sha256,
        publication_journal_sha256=anchor,
        source_run_id=700,
    )
    intent = _recovery_dispatch_intent(source)
    recovery_receipt = _recovery_dispatch_receipt(
        intent,
        mutation_response_received=False,
    )
    settlement = driver.EvidenceJournal.build_publication_recovery_operator_settlement(
        source,
        recovery_receipt,
    )
    assert settlement["mode"] == "response-loss-reconciled"
    assert settlement["pending_intent_replayed"] is False
    journal.close()


def _interrupted_journal_fixture(
    tmp_path: Path,
    name: str,
) -> tuple[Path, str, Path, dict[str, Any], bytes, dict[str, Any], bytes]:
    evidence = (tmp_path / name).resolve()
    receipt = live.create_evidence_directory(evidence)
    journal = driver.EvidenceJournal(receipt, plan_sha256=CONTROL_SHA)
    journal.record("immutable-plan", {"head_sha": HEAD})
    previous_sha256 = journal.last_evidence_sha256
    receipt_sha256 = receipt.receipt_sha256
    journal.close()
    temporary = evidence / (".evidence-" + "1" * 32 + ".tmp")
    envelope = {
        "schema_version": 1,
        "kind": "explainiverse-lambda-live-driver-evidence",
        "sequence": 3,
        "label": "recovery-probe",
        "control_plane_plan_sha256": CONTROL_SHA,
        "evidence_directory_acl_receipt_sha256": receipt_sha256,
        "previous_evidence_sha256": previous_sha256,
        "payload": {"preserved": True},
    }
    raw = _canonical(envelope)
    temporary.write_bytes(raw)
    sidecar = {
        "schema_version": 1,
        "kind": "explainiverse-local-evidence-recovery-classification",
        "control_plane_plan_sha256": CONTROL_SHA,
        "evidence_directory_acl_receipt_sha256": receipt_sha256,
        "journal_temporaries": [
            {
                "temporary_filename": temporary.name,
                "temporary_bytes": len(raw),
                "temporary_sha256": hashlib.sha256(raw).hexdigest(),
                "initial_link_count": 1,
                "envelope": envelope,
            }
        ],
        "emergency_uncommitted_slot": None,
    }
    sidecar_raw = _canonical(sidecar)
    return (
        evidence,
        receipt_sha256,
        temporary,
        envelope,
        raw,
        sidecar,
        sidecar_raw,
    )


@pytest.mark.parametrize("prefix_selector", (0, 1, -1, None))
def test_journal_reopen_resumes_exact_partial_sidecar_prefix(
    tmp_path: Path,
    prefix_selector: int | None,
) -> None:
    (
        evidence,
        receipt_sha256,
        temporary,
        _,
        _,
        _,
        sidecar_raw,
    ) = _interrupted_journal_fixture(tmp_path, f"sidecar-prefix-{prefix_selector}")
    prefix_length = len(sidecar_raw) if prefix_selector is None else prefix_selector
    if prefix_length == -1:
        prefix_length = len(sidecar_raw) - 1
    sidecar_digest = hashlib.sha256(sidecar_raw).hexdigest()
    partial = evidence / f".local-evidence-recovery-{sidecar_digest}.tmp"
    partial.write_bytes(sidecar_raw[:prefix_length])
    reopened_receipt = live.reopen_evidence_directory(
        evidence, expected_receipt_sha256=receipt_sha256
    )
    reopened = driver.EvidenceJournal.reopen_for_recovery(reopened_receipt, plan_sha256=CONTROL_SHA)
    assert reopened.interrupted_publish_recovery is not None
    assert not temporary.exists()
    assert not partial.exists()
    assert (evidence / f".local-evidence-recovery-{sidecar_digest}.json").is_file()
    reopened.close()


def test_journal_reopen_rejects_foreign_sidecar_prefix_without_mutation(
    tmp_path: Path,
) -> None:
    (
        evidence,
        receipt_sha256,
        temporary,
        _,
        raw,
        _,
        _,
    ) = _interrupted_journal_fixture(tmp_path, "foreign-sidecar-prefix")
    foreign = evidence / f".local-evidence-recovery-{'f' * 64}.tmp"
    foreign.write_bytes(b"")
    before = {
        path.name: (path.read_bytes(), path.stat().st_nlink, path.stat().st_mtime_ns)
        for path in evidence.iterdir()
        if path.is_file()
    }
    reopened_receipt = live.reopen_evidence_directory(
        evidence, expected_receipt_sha256=receipt_sha256
    )
    with pytest.raises(
        controller.ControllerError,
        match="local_recovery_sidecar_foreign_temp_rejected",
    ):
        driver.EvidenceJournal.reopen_for_recovery(reopened_receipt, plan_sha256=CONTROL_SHA)
    after = {
        path.name: (path.read_bytes(), path.stat().st_nlink, path.stat().st_mtime_ns)
        for path in evidence.iterdir()
        if path.is_file()
    }
    assert after == before
    assert temporary.read_bytes() == raw
    reopened_receipt.close()


def test_journal_reopen_preserves_complete_unpublished_atomic_temp(tmp_path: Path) -> None:
    evidence = (tmp_path / "evidence").resolve()
    receipt = live.create_evidence_directory(evidence)
    journal = driver.EvidenceJournal(receipt, plan_sha256=CONTROL_SHA)
    journal.record("immutable-plan", {"head_sha": HEAD})
    previous_sha256 = journal.last_evidence_sha256
    receipt_sha256 = receipt.receipt_sha256
    journal.close()
    temporary = evidence / (".evidence-" + "1" * 32 + ".tmp")
    envelope = {
        "schema_version": 1,
        "kind": "explainiverse-lambda-live-driver-evidence",
        "sequence": 3,
        "label": "recovery-probe",
        "control_plane_plan_sha256": CONTROL_SHA,
        "evidence_directory_acl_receipt_sha256": receipt_sha256,
        "previous_evidence_sha256": previous_sha256,
        "payload": {"preserved": True},
    }
    raw = _canonical(envelope)
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    reopened_receipt = live.reopen_evidence_directory(
        evidence, expected_receipt_sha256=receipt_sha256
    )
    reopened = driver.EvidenceJournal.reopen_for_recovery(reopened_receipt, plan_sha256=CONTROL_SHA)
    assert not temporary.exists()
    assert reopened.verified_entries()[-1] == (
        "recovery-probe",
        {"preserved": True},
    )
    recovery = reopened.interrupted_publish_recovery
    assert recovery is not None
    assert recovery["recovered_entries"] == [
        {
            "classification": "complete-unpublished-envelope",
            "temporary_filename": temporary.name,
            "temporary_bytes": len(raw),
            "temporary_sha256": hashlib.sha256(raw).hexdigest(),
            "sequence": 3,
            "label": "recovery-probe",
            "previous_evidence_sha256": previous_sha256,
            "final_filename": "003-recovery-probe.json",
        }
    ]
    sidecar = evidence / recovery["sidecar_filename"]
    assert sidecar.read_bytes()
    reopened.close()

    second_receipt = live.reopen_evidence_directory(
        evidence, expected_receipt_sha256=receipt_sha256
    )
    second = driver.EvidenceJournal.reopen_for_recovery(second_receipt, plan_sha256=CONTROL_SHA)
    assert second.interrupted_publish_recovery == recovery
    assert second.record_interrupted_publish_recovery() is not None
    _, recorded_payload = second.verified_entries()[-1]
    assert recorded_payload == recovery
    driver.EvidenceJournal._validate_journal_publish_recovery_payload(
        recorded_payload,
        record_sequence=4,
        root=evidence,
        plan_sha256=CONTROL_SHA,
        acl_sha256=receipt_sha256,
        seen_temporary_names=set(),
        seen_sidecar_names=set(),
    )
    second.close()
    third_receipt = live.reopen_evidence_directory(evidence, expected_receipt_sha256=receipt_sha256)
    third = driver.EvidenceJournal.reopen_for_recovery(third_receipt, plan_sha256=CONTROL_SHA)
    assert third.interrupted_publish_recovery is None
    assert sidecar.is_file()
    third.close()


@pytest.mark.parametrize("prefix_selector", (0, 1, -1, None))
def test_journal_reopen_completes_recovery_record_from_same_retained_sidecar(
    tmp_path: Path,
    prefix_selector: int | None,
) -> None:
    (
        evidence,
        receipt_sha256,
        _,
        _,
        _,
        _,
        _,
    ) = _interrupted_journal_fixture(tmp_path, f"recursive-record-{prefix_selector}")
    first_receipt = live.reopen_evidence_directory(
        evidence,
        expected_receipt_sha256=receipt_sha256,
    )
    first = driver.EvidenceJournal.reopen_for_recovery(
        first_receipt,
        plan_sha256=CONTROL_SHA,
    )
    recovery = first.interrupted_publish_recovery
    assert recovery is not None
    first.close()

    previous_raw = (evidence / "003-recovery-probe.json").read_bytes()
    record = {
        "schema_version": 1,
        "kind": "explainiverse-lambda-live-driver-evidence",
        "sequence": 4,
        "label": "journal-publish-recovery",
        "control_plane_plan_sha256": CONTROL_SHA,
        "evidence_directory_acl_receipt_sha256": receipt_sha256,
        "previous_evidence_sha256": hashlib.sha256(previous_raw).hexdigest(),
        "payload": recovery,
    }
    record_raw = _canonical(record)
    prefix_length = len(record_raw) if prefix_selector is None else prefix_selector
    if prefix_length == -1:
        prefix_length = len(record_raw) - 1
    temporary = evidence / f".evidence-{'9' * 32}.tmp"
    temporary.write_bytes(record_raw[:prefix_length])

    second_receipt = live.reopen_evidence_directory(
        evidence,
        expected_receipt_sha256=receipt_sha256,
    )
    second = driver.EvidenceJournal.reopen_for_recovery(
        second_receipt,
        plan_sha256=CONTROL_SHA,
    )
    assert second.interrupted_publish_recovery is None
    assert not temporary.exists()
    assert (evidence / "004-journal-publish-recovery.json").read_bytes() == record_raw
    assert len(list(evidence.glob(".local-evidence-recovery-*.json"))) == 1
    assert second.verified_entries()[-1] == ("journal-publish-recovery", recovery)
    second.close()

    third_receipt = live.reopen_evidence_directory(
        evidence,
        expected_receipt_sha256=receipt_sha256,
    )
    third = driver.EvidenceJournal.reopen_for_recovery(
        third_receipt,
        plan_sha256=CONTROL_SHA,
    )
    assert third.interrupted_publish_recovery is None
    assert len(list(evidence.glob(".local-evidence-recovery-*.json"))) == 1
    third.close()


def test_journal_reopen_rejects_and_preserves_partial_atomic_temp(
    tmp_path: Path,
) -> None:
    evidence = (tmp_path / "partial-evidence").resolve()
    receipt = live.create_evidence_directory(evidence)
    journal = driver.EvidenceJournal(receipt, plan_sha256=CONTROL_SHA)
    receipt_sha256 = receipt.receipt_sha256
    journal.close()
    temporary = evidence / (".evidence-" + "2" * 32 + ".tmp")
    partial = b'{"kind":"torn-journal-envelope"'
    with temporary.open("xb") as stream:
        stream.write(partial)
        stream.flush()
        os.fsync(stream.fileno())
    reopened_receipt = live.reopen_evidence_directory(
        evidence,
        expected_receipt_sha256=receipt_sha256,
    )
    with pytest.raises(controller.ControllerError, match="evidence_temporary_not_canonical"):
        driver.EvidenceJournal.reopen_for_recovery(
            reopened_receipt,
            plan_sha256=CONTROL_SHA,
        )
    assert temporary.read_bytes() == partial
    reopened_receipt.close()


def test_journal_reopen_does_not_mutate_valid_temp_after_malformed_prefix(
    tmp_path: Path,
) -> None:
    evidence = (tmp_path / "malformed-prefix").resolve()
    receipt = live.create_evidence_directory(evidence)
    journal = driver.EvidenceJournal(receipt, plan_sha256=CONTROL_SHA)
    journal.record("immutable-plan", {"head_sha": HEAD})
    receipt_sha256 = receipt.receipt_sha256
    journal.close()

    prefix_path = evidence / "002-immutable-plan.json"
    prefix = json.loads(prefix_path.read_bytes())
    prefix["unreviewed"] = True
    prefix_raw = _canonical(prefix)
    prefix_path.write_bytes(prefix_raw)
    temporary = evidence / (".evidence-" + "3" * 32 + ".tmp")
    temporary_raw = _canonical(
        {
            "schema_version": 1,
            "kind": "explainiverse-lambda-live-driver-evidence",
            "sequence": 3,
            "label": "recovery-probe",
            "control_plane_plan_sha256": CONTROL_SHA,
            "evidence_directory_acl_receipt_sha256": receipt_sha256,
            "previous_evidence_sha256": hashlib.sha256(prefix_raw).hexdigest(),
            "payload": {"preserved": True},
        }
    )
    with temporary.open("xb") as stream:
        stream.write(temporary_raw)
        stream.flush()
        os.fsync(stream.fileno())

    def snapshot() -> dict[str, tuple[bytes, int, int, int]]:
        return {
            path.name: (
                path.read_bytes(),
                path.stat().st_size,
                path.stat().st_mtime_ns,
                path.stat().st_nlink,
            )
            for path in evidence.iterdir()
            if path.is_file()
        }

    before = snapshot()
    reopened_receipt = live.reopen_evidence_directory(
        evidence,
        expected_receipt_sha256=receipt_sha256,
    )
    with pytest.raises(
        controller.ControllerError,
        match="local_recovery_precondition_journal_chain_rejected",
    ):
        driver.EvidenceJournal.reopen_for_recovery(
            reopened_receipt,
            plan_sha256=CONTROL_SHA,
        )
    assert snapshot() == before
    assert not (evidence / "003-recovery-probe.json").exists()
    reopened_receipt.close()


def test_evidence_journal_rejects_sequence_1000_before_write(tmp_path: Path) -> None:
    evidence = (tmp_path / "sequence-cap").resolve()
    receipt = live.create_evidence_directory(evidence)
    journal = driver.EvidenceJournal(receipt, plan_sha256=CONTROL_SHA)
    journal._sequence = driver.MAX_JOURNAL_SEQUENCE
    with pytest.raises(
        controller.ControllerError,
        match="journal_sequence_capacity_exhausted",
    ):
        journal.record("overflow", {"not_written": True})
    assert not (evidence / "1000-overflow.json").exists()
    journal.close()


def test_all_journal_openers_reject_four_digit_journal_residue(tmp_path: Path) -> None:
    evidence = (tmp_path / "sequence-residue").resolve()
    receipt = live.create_evidence_directory(evidence)
    receipt_sha256 = receipt.receipt_sha256
    journal = driver.EvidenceJournal(receipt, plan_sha256=CONTROL_SHA)
    journal.record("evidence-directory", receipt.to_public_mapping())
    journal.close()
    (evidence / "1000-overflow.json").write_bytes(b"{}")

    reopened_receipt = live.reopen_evidence_directory(
        evidence,
        expected_receipt_sha256=receipt_sha256,
    )
    with pytest.raises(
        controller.ControllerError,
        match="journal_filename_namespace_rejected",
    ):
        driver.EvidenceJournal.reopen_for_recovery(
            reopened_receipt,
            plan_sha256=CONTROL_SHA,
        )
    with pytest.raises(
        controller.ControllerError,
        match="journal_filename_namespace_rejected",
    ):
        driver.EvidenceJournal.load_final_main_acceptance(
            reopened_receipt,
            controller_resources=TEST_RESOURCES,
            final_control_plane_plan_sha256=CONTROL_SHA,
            final_journal_sha256="f" * 64,
        )
    with pytest.raises(
        controller.ControllerError,
        match="journal_filename_namespace_rejected",
    ):
        driver.EvidenceJournal.load_publication_recovery_source(
            reopened_receipt,
            controller_resources=TEST_RESOURCES,
            publication_control_plane_plan_sha256=CONTROL_SHA,
            publication_journal_sha256="f" * 64,
            source_run_id=1,
        )
    reopened_receipt.close()


def test_app_capture_archive_namespaces_identical_basenames(tmp_path: Path) -> None:
    evidence = (tmp_path / "evidence").resolve()
    receipt = live.create_evidence_directory(evidence)
    journal = driver.EvidenceJournal(receipt, plan_sha256=CONTROL_SHA)
    archived: list[tuple[str, str, bytes]] = []
    for ordinal in range(4):
        value, source_pages = _app_capture()
        pages = dict(source_pages)
        item = value["evidence"][0]
        filename = item["filename"]
        page = pages[filename] + f"capture-variant={ordinal}\n".encode("ascii")
        pages[filename] = page
        item["bytes"] = len(page)
        item["sha256"] = hashlib.sha256(page).hexdigest()
        capture = controller.TrustedAppCapture.from_mapping(
            value,
            resources=TEST_RESOURCES,
            evidence_reader=pages.__getitem__,
            now=NOW,
        )
        archive = journal.archive_installed_app_capture(capture, pages.__getitem__)
        journal.record("installed-app-raw-archive", archive)
        archived.append((capture.evidence_sha256, filename, page))
    assert len({capture_sha for capture_sha, _, _ in archived}) == 4
    for capture_sha, filename, expected in archived:
        assert (
            driver.EvidenceJournal._read_archived_app_page(evidence, capture_sha, filename)
            == expected
        )
    journal.close()


def _provider_intent_fixture() -> live.MutationIntent:
    operation = "terminate"
    method, path = live.MUTATION_PATHS[operation]
    body_sha256 = "b" * 64
    request_sha256 = hashlib.sha256(
        live._canonical_json(
            {
                "operation": operation,
                "method": method,
                "path": path,
                "body_sha256": body_sha256,
                "timeout_seconds": live.PROVIDER_TIMEOUT_SECONDS,
            }
        )
    ).hexdigest()
    return live.MutationIntent(
        plan_sha256=CONTROL_SHA,
        operation=operation,
        prestate_phase="instance_bound",
        prestate_snapshot_sha256="1" * 64,
        prestate_receipt_nonce="2" * 32,
        callback_binding_sha256="3" * 64,
        method=method,
        path=path,
        request_sha256=request_sha256,
        request_body_sha256=body_sha256,
        sensitive_body=False,
        timeout_seconds=live.PROVIDER_TIMEOUT_SECONDS,
    )


def test_provider_intent_uses_preallocated_reserve_after_journal_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = (tmp_path / "evidence").resolve()
    receipt = live.create_evidence_directory(evidence)
    journal = driver.EvidenceJournal(receipt, plan_sha256=CONTROL_SHA)
    intent = _provider_intent_fixture()
    reserve_descriptor = journal._emergency_fd
    real_fsync = driver.os.fsync
    failed = False

    def fail_atomic_fsync_once(descriptor: int) -> None:
        nonlocal failed
        if descriptor != reserve_descriptor and not failed:
            failed = True
            raise OSError(errno.ENOSPC, "simulated journal ENOSPC")
        real_fsync(descriptor)

    monkeypatch.setattr(driver.os, "fsync", fail_atomic_fsync_once)
    journal.record_provider_mutation_intent(intent)
    assert failed is True
    assert journal.emergency_provider_intents() == (intent.to_public_mapping(),)
    assert not list(evidence.glob(".evidence-*.tmp"))
    assert not list(evidence.glob("*-provider-mutation-intent.json"))
    journal.close()


def test_provider_intent_rejects_nonstorage_journal_os_error_without_fallback(
    tmp_path: Path,
) -> None:
    evidence = (tmp_path / "nonstorage-intent-failure").resolve()
    receipt = live.create_evidence_directory(evidence)
    journal = driver.EvidenceJournal(receipt, plan_sha256=CONTROL_SHA)
    original_record = journal.record
    journal.record = lambda *_: (_ for _ in ()).throw(  # type: ignore[method-assign]
        FileExistsError(errno.EEXIST, "injected namespace collision")
    )
    with pytest.raises(
        controller.ControllerError,
        match="provider_mutation_journal_os_error_rejected",
    ):
        journal.record_provider_mutation_intent(_provider_intent_fixture())
    assert journal.emergency_provider_intents() == ()
    journal.record = original_record  # type: ignore[method-assign]
    journal.close()


def test_provider_intent_late_destination_race_preserves_source_and_never_falls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "nt":
        pytest.skip("Windows no-replace publication race")
    evidence = (tmp_path / "late-destination-race").resolve()
    receipt = live.create_evidence_directory(evidence)
    journal = driver.EvidenceJournal(receipt, plan_sha256=CONTROL_SHA)
    previous_sha256 = journal.last_evidence_sha256
    raced_destination = evidence / "002-provider-mutation-intent.json"
    import ctypes
    from ctypes import wintypes

    real_windll = ctypes.WinDLL
    observed_errors: list[tuple[int, int]] = []

    class Kernel32Proxy:
        def __init__(self, library: Any) -> None:
            self._library = library

        def __getattr__(self, name: str) -> Any:
            if name != "SetFileInformationByHandle":
                return getattr(self._library, name)
            real_set_information = self._library.SetFileInformationByHandle
            real_set_information.argtypes = [
                wintypes.HANDLE,
                wintypes.INT,
                wintypes.LPVOID,
                wintypes.DWORD,
            ]
            real_set_information.restype = wintypes.BOOL

            def race_after_source_validation(*args: Any) -> int:
                raced_destination.write_bytes(b"foreign destination")
                result = int(real_set_information(*args))
                observed_errors.append((result, ctypes.get_last_error()))
                return result

            return race_after_source_validation

    def interposed_windll(name: str, *args: Any, **kwargs: Any) -> Any:
        library = real_windll(name, *args, **kwargs)
        if name.lower() == "kernel32":
            return Kernel32Proxy(library)
        return library

    monkeypatch.setattr(ctypes, "WinDLL", interposed_windll)
    with pytest.raises(
        controller.ControllerError,
        match="evidence_atomic_publication_requires_recovery",
    ):
        journal.record_provider_mutation_intent(_provider_intent_fixture())
    assert journal.emergency_provider_intents() == ()
    assert journal._sequence == 1
    assert journal.last_evidence_sha256 == previous_sha256
    assert observed_errors == [(0, 183)]
    assert raced_destination.read_bytes() == b"foreign destination"
    sources = list(evidence.glob(".evidence-*.tmp"))
    assert len(sources) == 1
    source_envelope = json.loads(sources[0].read_bytes())
    assert source_envelope["label"] == "provider-mutation-intent"
    journal.close()


def test_journal_reopen_rejects_and_preserves_partial_next_emergency_slot(
    tmp_path: Path,
) -> None:
    evidence = (tmp_path / "evidence").resolve()
    receipt = live.create_evidence_directory(evidence)
    receipt_sha256 = receipt.receipt_sha256
    journal = driver.EvidenceJournal(receipt, plan_sha256=CONTROL_SHA)
    operation = "terminate"
    method, path = live.MUTATION_PATHS[operation]
    body_sha256 = "b" * 64
    request_sha256 = hashlib.sha256(
        live._canonical_json(
            {
                "operation": operation,
                "method": method,
                "path": path,
                "body_sha256": body_sha256,
                "timeout_seconds": live.PROVIDER_TIMEOUT_SECONDS,
            }
        )
    ).hexdigest()
    intent = live.MutationIntent(
        plan_sha256=CONTROL_SHA,
        operation=operation,
        prestate_phase="instance_bound",
        prestate_snapshot_sha256="1" * 64,
        prestate_receipt_nonce="2" * 32,
        callback_binding_sha256="3" * 64,
        method=method,
        path=path,
        request_sha256=request_sha256,
        request_body_sha256=body_sha256,
        sensitive_body=False,
        timeout_seconds=live.PROVIDER_TIMEOUT_SECONDS,
    )
    original_record = journal.record
    journal.record = lambda *_: (_ for _ in ()).throw(  # type: ignore[method-assign]
        OSError(errno.ENOSPC, "ENOSPC")
    )
    journal.record_provider_mutation_intent(intent)
    journal.record = original_record  # type: ignore[method-assign]
    journal.close()

    reserve = evidence / driver.EMERGENCY_EVIDENCE_FILENAME
    with reserve.open("r+b", buffering=0) as stream:
        stream.seek(driver.EMERGENCY_EVIDENCE_SLOT_SIZE)
        stream.write(driver.EMERGENCY_EVIDENCE_MAGIC[:5])
        os.fsync(stream.fileno())

    reopened_receipt = live.reopen_evidence_directory(
        evidence, expected_receipt_sha256=receipt_sha256
    )
    before = reserve.read_bytes()
    with pytest.raises(
        controller.ControllerError,
        match="emergency_evidence_uncommitted_slot_rejected",
    ):
        driver.EvidenceJournal.reopen_for_recovery(reopened_receipt, plan_sha256=CONTROL_SHA)
    assert reserve.read_bytes() == before
    reopened_receipt.close()


def test_journal_reopen_sidecar_binds_complete_uncommitted_emergency_slot(
    tmp_path: Path,
) -> None:
    evidence = (tmp_path / "emergency-sidecar").resolve()
    receipt = live.create_evidence_directory(evidence)
    receipt_sha256 = receipt.receipt_sha256
    journal = driver.EvidenceJournal(receipt, plan_sha256=CONTROL_SHA)
    operation = "terminate"
    method, path = live.MUTATION_PATHS[operation]
    body_sha256 = "b" * 64
    request_sha256 = hashlib.sha256(
        live._canonical_json(
            {
                "operation": operation,
                "method": method,
                "path": path,
                "body_sha256": body_sha256,
                "timeout_seconds": live.PROVIDER_TIMEOUT_SECONDS,
            }
        )
    ).hexdigest()
    intent = live.MutationIntent(
        plan_sha256=CONTROL_SHA,
        operation=operation,
        prestate_phase="instance_bound",
        prestate_snapshot_sha256="1" * 64,
        prestate_receipt_nonce="2" * 32,
        callback_binding_sha256="3" * 64,
        method=method,
        path=path,
        request_sha256=request_sha256,
        request_body_sha256=body_sha256,
        sensitive_body=False,
        timeout_seconds=live.PROVIDER_TIMEOUT_SECONDS,
    )
    original_record = journal.record
    journal.record = lambda *_: (_ for _ in ()).throw(  # type: ignore[method-assign]
        OSError(errno.ENOSPC, "ENOSPC")
    )
    journal.record_provider_mutation_intent(intent)
    journal.record = original_record  # type: ignore[method-assign]
    previous_sha256 = journal._emergency_previous_sha256
    journal.close()

    second_material = {
        "schema_version": 1,
        "kind": "explainiverse-provider-mutation-emergency-evidence",
        "sequence": 2,
        "control_plane_plan_sha256": CONTROL_SHA,
        "evidence_directory_acl_receipt_sha256": receipt_sha256,
        "previous_evidence_sha256": previous_sha256,
        "intent": intent.to_public_mapping(),
    }
    payload = _canonical(second_material)
    slot = (
        driver.EMERGENCY_EVIDENCE_MAGIC
        + len(payload).to_bytes(4, "big")
        + bytes.fromhex(hashlib.sha256(payload).hexdigest())
        + payload
    )
    slot += b"\0" * (driver.EMERGENCY_EVIDENCE_SLOT_SIZE - len(slot))
    reserve = evidence / driver.EMERGENCY_EVIDENCE_FILENAME
    with reserve.open("r+b", buffering=0) as stream:
        stream.seek(driver.EMERGENCY_EVIDENCE_SLOT_SIZE)
        stream.write(slot)
        os.fsync(stream.fileno())

    reopened_receipt = live.reopen_evidence_directory(
        evidence, expected_receipt_sha256=receipt_sha256
    )
    reopened = driver.EvidenceJournal.reopen_for_recovery(reopened_receipt, plan_sha256=CONTROL_SHA)
    recovery = reopened.interrupted_publish_recovery
    assert recovery is not None
    assert recovery["recovered_entries"] == []
    assert recovery["emergency_uncommitted_slot"] == {
        "classification": "complete-uncommitted-provider-intent-slot",
        "reserve_filename": driver.EMERGENCY_EVIDENCE_FILENAME,
        "slot_index": 1,
        "slot_bytes": driver.EMERGENCY_EVIDENCE_SLOT_SIZE,
        "slot_sha256": hashlib.sha256(slot).hexdigest(),
    }
    assert reopened.emergency_provider_intents() == (intent.to_public_mapping(),)
    with reserve.open("rb") as stream:
        stream.seek(driver.EMERGENCY_EVIDENCE_SLOT_SIZE)
        assert stream.read(driver.EMERGENCY_EVIDENCE_SLOT_SIZE) == (
            b"\0" * driver.EMERGENCY_EVIDENCE_SLOT_SIZE
        )
    assert reopened.record_interrupted_publish_recovery() is not None
    reopened.close()

    second_receipt = live.reopen_evidence_directory(
        evidence, expected_receipt_sha256=receipt_sha256
    )
    second = driver.EvidenceJournal.reopen_for_recovery(second_receipt, plan_sha256=CONTROL_SHA)
    assert second.interrupted_publish_recovery is None
    assert second.emergency_provider_intents() == (intent.to_public_mapping(),)
    second.close()


def _uncommitted_emergency_slot(
    *,
    receipt_sha256: str,
    sequence: int = 1,
    previous_sha256: str | None = None,
) -> bytes:
    material = {
        "schema_version": 1,
        "kind": "explainiverse-provider-mutation-emergency-evidence",
        "sequence": sequence,
        "control_plane_plan_sha256": CONTROL_SHA,
        "evidence_directory_acl_receipt_sha256": receipt_sha256,
        "previous_evidence_sha256": previous_sha256,
        "intent": _provider_intent_fixture().to_public_mapping(),
    }
    payload = _canonical(material)
    slot = (
        driver.EMERGENCY_EVIDENCE_MAGIC
        + len(payload).to_bytes(4, "big")
        + bytes.fromhex(hashlib.sha256(payload).hexdigest())
        + payload
    )
    return slot + b"\0" * (driver.EMERGENCY_EVIDENCE_SLOT_SIZE - len(slot))


def _append_test_journal_prefix(
    evidence: Path,
    *,
    receipt_sha256: str,
    target_sequence: int,
) -> str:
    first = evidence / "001-evidence-directory.json"
    previous_sha256 = hashlib.sha256(first.read_bytes()).hexdigest()
    entries: list[tuple[Path, bytes]] = []
    for sequence in range(2, target_sequence + 1):
        envelope = {
            "schema_version": 1,
            "kind": "explainiverse-lambda-live-driver-evidence",
            "sequence": sequence,
            "label": "capacity-probe",
            "control_plane_plan_sha256": CONTROL_SHA,
            "evidence_directory_acl_receipt_sha256": receipt_sha256,
            "previous_evidence_sha256": previous_sha256,
            "payload": {"sequence": sequence},
        }
        raw = _canonical(envelope)
        entries.append((evidence / f"{sequence:03d}-capacity-probe.json", raw))
        previous_sha256 = hashlib.sha256(raw).hexdigest()

    def write_entry(entry: tuple[Path, bytes]) -> None:
        path, raw = entry
        path.write_bytes(raw)

    with ThreadPoolExecutor(max_workers=32) as executor:
        list(executor.map(write_entry, entries))
    return previous_sha256


def _file_tree_snapshot(root: Path) -> dict[str, tuple[bytes, int, int, int]]:
    paths = [path for path in root.iterdir() if path.is_file()]

    def snapshot(path: Path) -> tuple[str, tuple[bytes, int, int, int]]:
        return path.name, (
            path.read_bytes(),
            path.stat().st_size,
            path.stat().st_mtime_ns,
            path.stat().st_nlink,
        )

    with ThreadPoolExecutor(max_workers=32) as executor:
        return dict(executor.map(snapshot, paths))


@pytest.mark.parametrize("target_sequence", (996, 997, 998, 999))
@pytest.mark.parametrize("recovery_kind", ("temporary", "emergency", "combined"))
def test_journal_recovery_capacity_boundaries_are_checked_before_repair(
    tmp_path: Path,
    target_sequence: int,
    recovery_kind: str,
) -> None:
    evidence = (tmp_path / f"capacity-{recovery_kind}-{target_sequence}").resolve()
    receipt = live.create_evidence_directory(evidence)
    receipt_sha256 = receipt.receipt_sha256
    journal = driver.EvidenceJournal(receipt, plan_sha256=CONTROL_SHA)
    journal.close()
    previous_sha256 = _append_test_journal_prefix(
        evidence,
        receipt_sha256=receipt_sha256,
        target_sequence=target_sequence,
    )

    has_temporary = recovery_kind in {"temporary", "combined"}
    has_emergency = recovery_kind in {"emergency", "combined"}
    temporary: Path | None = None
    if has_temporary:
        temporary = evidence / f".evidence-{target_sequence:032x}.tmp"
        envelope = {
            "schema_version": 1,
            "kind": "explainiverse-lambda-live-driver-evidence",
            "sequence": target_sequence + 1,
            "label": "capacity-tail",
            "control_plane_plan_sha256": CONTROL_SHA,
            "evidence_directory_acl_receipt_sha256": receipt_sha256,
            "previous_evidence_sha256": previous_sha256,
            "payload": {"tail": True},
        }
        temporary.write_bytes(_canonical(envelope))
    if has_emergency:
        reserve = evidence / driver.EMERGENCY_EVIDENCE_FILENAME
        slot = _uncommitted_emergency_slot(receipt_sha256=receipt_sha256)
        with reserve.open("r+b", buffering=0) as stream:
            stream.write(slot)
            os.fsync(stream.fileno())

    entries_needed = (1 if has_temporary else 0) + 1
    allowed = target_sequence + entries_needed <= driver.MAX_JOURNAL_SEQUENCE
    before = _file_tree_snapshot(evidence)
    reopened_receipt = live.reopen_evidence_directory(
        evidence,
        expected_receipt_sha256=receipt_sha256,
    )
    if not allowed:
        with pytest.raises(
            controller.ControllerError,
            match=(
                "local_recovery_journal_capacity_rejected" "|evidence_temporary_envelope_rejected"
            ),
        ):
            driver.EvidenceJournal.reopen_for_recovery(
                reopened_receipt,
                plan_sha256=CONTROL_SHA,
            )
        assert _file_tree_snapshot(evidence) == before
        reopened_receipt.close()
        return

    reopened = driver.EvidenceJournal.reopen_for_recovery(
        reopened_receipt,
        plan_sha256=CONTROL_SHA,
    )
    recovery = reopened.interrupted_publish_recovery
    assert recovery is not None
    assert bool(recovery["recovered_entries"]) is has_temporary
    assert (recovery["emergency_uncommitted_slot"] is not None) is has_emergency
    if temporary is not None:
        assert not temporary.exists()
        assert (evidence / f"{target_sequence + 1:03d}-capacity-tail.json").is_file()
    assert reopened.record_interrupted_publish_recovery() is not None
    assert reopened._sequence == target_sequence + entries_needed
    reopened.close()


def test_journal_reopen_resumes_exact_emergency_zero_prefix_from_same_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = (tmp_path / "emergency-zero-prefix").resolve()
    receipt = live.create_evidence_directory(evidence)
    receipt_sha256 = receipt.receipt_sha256
    journal = driver.EvidenceJournal(receipt, plan_sha256=CONTROL_SHA)
    journal.close()
    reserve = evidence / driver.EMERGENCY_EVIDENCE_FILENAME
    original_slot = _uncommitted_emergency_slot(receipt_sha256=receipt_sha256)
    with reserve.open("r+b", buffering=0) as stream:
        stream.write(original_slot)
        os.fsync(stream.fileno())

    real_write_all = driver.EvidenceJournal._write_all
    crash_prefix = 4096
    crashed = False

    def crash_mid_zero(descriptor: int, payload: bytes) -> None:
        nonlocal crashed
        if payload == b"\0" * driver.EMERGENCY_EVIDENCE_SLOT_SIZE and not crashed:
            crashed = True
            written = os.write(descriptor, payload[:crash_prefix])
            assert written == crash_prefix
            os.fsync(descriptor)
            raise OSError(errno.EIO, "injected crash during reserve zero")
        real_write_all(descriptor, payload)

    monkeypatch.setattr(
        driver.EvidenceJournal,
        "_write_all",
        staticmethod(crash_mid_zero),
    )
    first_receipt = live.reopen_evidence_directory(
        evidence,
        expected_receipt_sha256=receipt_sha256,
    )
    with pytest.raises(OSError, match="injected crash during reserve zero"):
        driver.EvidenceJournal.reopen_for_recovery(
            first_receipt,
            plan_sha256=CONTROL_SHA,
        )
    first_receipt.close()
    sidecars = list(evidence.glob(".local-evidence-recovery-*.json"))
    assert len(sidecars) == 1
    current_slot = reserve.read_bytes()[: driver.EMERGENCY_EVIDENCE_SLOT_SIZE]
    assert current_slot[:crash_prefix] == b"\0" * crash_prefix
    assert current_slot[crash_prefix:] == original_slot[crash_prefix:]

    second_receipt = live.reopen_evidence_directory(
        evidence,
        expected_receipt_sha256=receipt_sha256,
    )
    second = driver.EvidenceJournal.reopen_for_recovery(
        second_receipt,
        plan_sha256=CONTROL_SHA,
    )
    recovery = second.interrupted_publish_recovery
    assert recovery is not None
    assert (
        recovery["emergency_uncommitted_slot"]["slot_sha256"]
        == hashlib.sha256(original_slot).hexdigest()
    )
    assert list(evidence.glob(".local-evidence-recovery-*.json")) == sidecars
    assert reserve.read_bytes()[: driver.EMERGENCY_EVIDENCE_SLOT_SIZE] == (
        b"\0" * driver.EMERGENCY_EVIDENCE_SLOT_SIZE
    )
    assert second.record_interrupted_publish_recovery() is not None
    second.close()

    third_receipt = live.reopen_evidence_directory(
        evidence,
        expected_receipt_sha256=receipt_sha256,
    )
    third = driver.EvidenceJournal.reopen_for_recovery(
        third_receipt,
        plan_sha256=CONTROL_SHA,
    )
    assert third.interrupted_publish_recovery is None
    assert third.emergency_provider_intents() == ()
    third.close()


def test_journal_reserve_handle_denies_second_writer_for_full_lifetime_on_windows(
    tmp_path: Path,
) -> None:
    if os.name != "nt":
        pytest.skip("Windows share-mode contract")
    evidence = (tmp_path / "exclusive-reserve").resolve()
    receipt = live.create_evidence_directory(evidence)
    journal = driver.EvidenceJournal(receipt, plan_sha256=CONTROL_SHA)
    reserve = evidence / driver.EMERGENCY_EVIDENCE_FILENAME
    with pytest.raises(OSError):
        competing = os.open(
            reserve,
            os.O_RDWR | getattr(os, "O_BINARY", 0),
        )
        os.close(competing)
    with pytest.raises(OSError):
        reserve.unlink()
    journal.close()
    competing = os.open(
        reserve,
        os.O_RDWR | getattr(os, "O_BINARY", 0),
    )
    os.close(competing)
    descriptor = driver.EvidenceJournal._open_recovery_reserve_exclusive(reserve)
    try:
        with pytest.raises(OSError):
            competing = os.open(
                reserve,
                os.O_RDWR | getattr(os, "O_BINARY", 0),
            )
            os.close(competing)
    finally:
        os.close(descriptor)


def test_journal_close_releases_handles_when_reserve_flush_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = (tmp_path / "close-flush-failure").resolve()
    receipt = live.create_evidence_directory(evidence)
    journal = driver.EvidenceJournal(receipt, plan_sha256=CONTROL_SHA)
    reserve = evidence / driver.EMERGENCY_EVIDENCE_FILENAME
    reserve_descriptor = journal._emergency_fd
    real_fsync = driver.os.fsync

    def fail_reserve_fsync(descriptor: int) -> None:
        if descriptor == reserve_descriptor:
            raise OSError("injected reserve flush failure")
        real_fsync(descriptor)

    monkeypatch.setattr(driver.os, "fsync", fail_reserve_fsync)
    with pytest.raises(OSError, match="injected reserve flush failure"):
        journal.close()
    assert journal._emergency_fd == -1
    assert receipt.closed is True
    competing = os.open(
        reserve,
        os.O_RDWR | getattr(os, "O_BINARY", 0),
    )
    os.close(competing)


def test_journal_constructor_failure_releases_owned_handles_and_preserves_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = (tmp_path / "constructor-failure").resolve()
    receipt = live.create_evidence_directory(evidence)
    preserved = b"durable partial evidence"

    def fail_first_publish(
        self: driver.EvidenceJournal,
        destination: Path,
        payload: bytes,
    ) -> None:
        del payload
        (destination.parent / f".evidence-{'a' * 32}.tmp").write_bytes(preserved)
        raise OSError("injected first journal publication failure")

    monkeypatch.setattr(driver.EvidenceJournal, "_publish_atomic", fail_first_publish)
    with pytest.raises(OSError, match="injected first journal publication failure"):
        driver.EvidenceJournal(receipt, plan_sha256=CONTROL_SHA)
    assert receipt.closed is True
    reserve = evidence / driver.EMERGENCY_EVIDENCE_FILENAME
    assert reserve.stat().st_size == (
        driver.EMERGENCY_EVIDENCE_SLOT_SIZE * driver.EMERGENCY_EVIDENCE_SLOT_COUNT
    )
    assert (evidence / f".evidence-{'a' * 32}.tmp").read_bytes() == preserved
    competing = os.open(
        reserve,
        os.O_RDWR | getattr(os, "O_BINARY", 0),
    )
    os.close(competing)


def test_journal_reserve_initialization_failure_closes_owned_directory_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = (tmp_path / "reserve-initialization-failure").resolve()
    receipt = live.create_evidence_directory(evidence)

    def fail_reserve_fsync(_: int) -> None:
        raise OSError("injected initial reserve flush failure")

    monkeypatch.setattr(driver.os, "fsync", fail_reserve_fsync)
    with pytest.raises(OSError, match="injected initial reserve flush failure"):
        driver.EvidenceJournal(receipt, plan_sha256=CONTROL_SHA)
    assert receipt.closed is True
    assert not (evidence / driver.EMERGENCY_EVIDENCE_FILENAME).exists()


def test_journal_reserve_open_failure_closes_owned_directory_receipt(
    tmp_path: Path,
) -> None:
    evidence = (tmp_path / "reserve-open-failure").resolve()
    receipt = live.create_evidence_directory(evidence)
    reserve = evidence / driver.EMERGENCY_EVIDENCE_FILENAME
    reserve.write_bytes(b"preexisting reserve collision")
    with pytest.raises(OSError):
        driver.EvidenceJournal(receipt, plan_sha256=CONTROL_SHA)
    assert receipt.closed is True
    assert reserve.read_bytes() == b"preexisting reserve collision"


def test_journal_constructor_validation_failure_closes_owned_directory_receipt(
    tmp_path: Path,
) -> None:
    evidence = (tmp_path / "constructor-validation-failure").resolve()
    receipt = live.create_evidence_directory(evidence)
    with pytest.raises(controller.ControllerError, match="journal_plan_sha256_rejected"):
        driver.EvidenceJournal(receipt, plan_sha256="not-a-digest")
    assert receipt.closed is True


def test_windows_atomic_publication_denies_source_write_delete_and_rebind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "nt":
        pytest.skip("Windows held-handle publication contract")
    evidence = (tmp_path / "held-publication").resolve()
    receipt = live.create_evidence_directory(evidence)
    journal = driver.EvidenceJournal(receipt, plan_sha256=CONTROL_SHA)
    original = driver.EvidenceJournal._publish_windows_held_file.__func__
    attempts: list[str] = []

    def attack_then_publish(
        cls: type[driver.EvidenceJournal],
        descriptor: int,
        *,
        temporary: Path,
        destination: Path,
        payload: bytes,
        context: str,
    ) -> None:
        replacement = temporary.parent / "replacement-source.bin"
        replacement.write_bytes(b"attacker replacement")
        with pytest.raises(OSError):
            competing = os.open(
                temporary,
                os.O_RDWR | getattr(os, "O_BINARY", 0),
            )
            os.close(competing)
        attempts.append("write-denied")
        with pytest.raises(OSError):
            temporary.unlink()
        attempts.append("delete-denied")
        with pytest.raises(OSError):
            os.replace(replacement, temporary)
        attempts.append("replace-denied")
        aside = temporary.with_name(temporary.name + ".aside")
        with pytest.raises(OSError):
            temporary.rename(aside)
        attempts.append("rename-denied")
        original(
            cls,
            descriptor,
            temporary=temporary,
            destination=destination,
            payload=payload,
            context=context,
        )

    monkeypatch.setattr(
        driver.EvidenceJournal,
        "_publish_windows_held_file",
        classmethod(attack_then_publish),
    )
    digest = journal.record("held-windows-publication", {"expected": True})
    destination = evidence / "002-held-windows-publication.json"
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == digest
    assert json.loads(destination.read_bytes())["payload"] == {"expected": True}
    assert attempts == [
        "write-denied",
        "delete-denied",
        "replace-denied",
        "rename-denied",
    ]
    journal.close()


def test_windows_recovery_sidecar_and_journal_publication_hold_source_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "nt":
        pytest.skip("Windows held-handle recovery publication contract")
    (
        evidence,
        receipt_sha256,
        _,
        _,
        _,
        _,
        _,
    ) = _interrupted_journal_fixture(tmp_path, "held-recovery-publication")
    original = driver.EvidenceJournal._publish_windows_held_file.__func__
    contexts: list[str] = []

    def attack_then_publish(
        cls: type[driver.EvidenceJournal],
        descriptor: int,
        *,
        temporary: Path,
        destination: Path,
        payload: bytes,
        context: str,
    ) -> None:
        replacement = tmp_path / f"{context}-replacement.bin"
        replacement.write_bytes(b"attacker replacement")
        with pytest.raises(OSError):
            competing = os.open(
                temporary,
                os.O_RDWR | getattr(os, "O_BINARY", 0),
            )
            os.close(competing)
        with pytest.raises(OSError):
            temporary.unlink()
        with pytest.raises(OSError):
            os.replace(replacement, temporary)
        with pytest.raises(OSError):
            temporary.rename(temporary.with_name(temporary.name + ".aside"))
        contexts.append(context)
        original(
            cls,
            descriptor,
            temporary=temporary,
            destination=destination,
            payload=payload,
            context=context,
        )

    monkeypatch.setattr(
        driver.EvidenceJournal,
        "_publish_windows_held_file",
        classmethod(attack_then_publish),
    )
    reopened_receipt = live.reopen_evidence_directory(
        evidence,
        expected_receipt_sha256=receipt_sha256,
    )
    reopened = driver.EvidenceJournal.reopen_for_recovery(
        reopened_receipt,
        plan_sha256=CONTROL_SHA,
    )
    assert contexts == ["local_recovery_sidecar", "local_recovery_journal"]
    assert reopened.verified_entries()[-1] == (
        "recovery-probe",
        {"preserved": True},
    )
    reopened.close()


def test_windows_atomic_open_failure_closes_full_width_native_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "nt":
        pytest.skip("Windows native handle-width contract")
    import ctypes
    import msvcrt
    from ctypes import wintypes

    sentinel = 0x1_0000_1234
    closed: list[int] = []

    class FakeFunction:
        def __init__(self, callback: Callable[..., Any]) -> None:
            self.callback = callback
            self.argtypes: Any = None
            self.restype: Any = None

        def __call__(self, *args: Any) -> Any:
            return self.callback(*args)

    close_functions: list[FakeFunction] = []

    class FakeKernel32:
        def __init__(self) -> None:
            self.CreateFileW = FakeFunction(lambda *_: sentinel)
            self.CloseHandle = FakeFunction(lambda handle: closed.append(int(handle)) or 1)
            close_functions.append(self.CloseHandle)

    monkeypatch.setattr(ctypes, "WinDLL", lambda *_args, **_kwargs: FakeKernel32())
    monkeypatch.setattr(
        msvcrt,
        "open_osfhandle",
        lambda *_: (_ for _ in ()).throw(OSError("injected CRT adoption failure")),
    )
    atomic_path = (tmp_path / "atomic-handle.bin").resolve()
    reserve_path = (tmp_path / "reserve-handle.bin").resolve()
    with pytest.raises(OSError, match="injected CRT adoption failure"):
        driver.EvidenceJournal._open_windows_atomic_file(
            atomic_path,
            create_new=True,
        )
    with pytest.raises(OSError, match="injected CRT adoption failure"):
        driver.EvidenceJournal._open_recovery_reserve_exclusive(
            reserve_path,
            create_new=True,
        )
    assert closed == [sentinel, sentinel]
    typed_close_functions = [item for item in close_functions if item.argtypes is not None]
    assert len(typed_close_functions) == 2
    assert all(item.argtypes == [wintypes.HANDLE] for item in typed_close_functions)
    assert all(item.restype is wintypes.BOOL for item in typed_close_functions)


def test_local_recovery_sidecar_cardinality_fails_before_body_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = (tmp_path / "sidecar-cardinality").resolve()
    evidence.mkdir()
    for index in range(driver.MAX_JOURNAL_SEQUENCE + 1):
        (evidence / f".local-evidence-recovery-{index:064x}.json").write_bytes(b"x")

    def reject_read(_: Path) -> bytes:
        raise AssertionError("sidecar body read before cardinality rejection")

    monkeypatch.setattr(Path, "read_bytes", reject_read)
    with pytest.raises(
        controller.ControllerError,
        match="local_recovery_sidecar_cardinality_rejected",
    ):
        driver.EvidenceJournal._load_local_recovery_sidecars(evidence)


def test_journal_entry_cardinality_fails_before_body_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = (tmp_path / "journal-cardinality").resolve()
    evidence.mkdir()
    for index in range(driver.MAX_JOURNAL_SEQUENCE + 1):
        (evidence / f"001-residue-{index}.json").write_bytes(b"x")

    def reject_read(_: Path) -> bytes:
        raise AssertionError("journal body read before cardinality rejection")

    monkeypatch.setattr(Path, "read_bytes", reject_read)
    with pytest.raises(
        controller.ControllerError,
        match="journal_entry_cardinality_rejected",
    ):
        driver.EvidenceJournal._validate_journal_chain(
            evidence,
            plan_sha256=CONTROL_SHA,
            acl_sha256="d" * 64,
            context="cardinality",
        )


def test_journal_recovery_rejects_two_atomic_temps_before_body_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = (tmp_path / "atomic-temp-cardinality").resolve()
    receipt = live.create_evidence_directory(evidence)
    receipt_sha256 = receipt.receipt_sha256
    journal = driver.EvidenceJournal(receipt, plan_sha256=CONTROL_SHA)
    journal.close()
    temporaries = [evidence / f".evidence-{index:032x}.tmp" for index in (1, 2)]
    for temporary in temporaries:
        temporary.write_bytes(b"unread impossible residue")
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_nlink)
        for path in evidence.iterdir()
        if path.is_file()
    }
    real_read_bytes = Path.read_bytes

    def reject_atomic_temp_read(path: Path) -> bytes:
        if path.name.startswith(".evidence-"):
            raise AssertionError("atomic temp body read before cardinality rejection")
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_atomic_temp_read)
    reopened_receipt = live.reopen_evidence_directory(
        evidence,
        expected_receipt_sha256=receipt_sha256,
    )
    with pytest.raises(
        controller.ControllerError,
        match="evidence_temporary_cardinality_rejected",
    ):
        driver.EvidenceJournal.reopen_for_recovery(
            reopened_receipt,
            plan_sha256=CONTROL_SHA,
        )
    monkeypatch.setattr(Path, "read_bytes", real_read_bytes)
    after = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_nlink)
        for path in evidence.iterdir()
        if path.is_file()
    }
    assert after == before
    reopened_receipt.close()


def test_gh_mutation_unusable_success_response_is_ambiguous(tmp_path: Path) -> None:
    executable = (tmp_path / "gh.exe").resolve()
    executable.write_bytes(b"audited gh fixture")

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(argv, 0, b"not-an-included-response", b"")

    transport = controller.GhCliTransport(
        executable_path=str(executable),
        executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
        runner=run,
    )
    with pytest.raises(controller.AmbiguousGitHubMutation, match="successful_response_unusable"):
        transport.request("POST", f"/repos/{controller.REPOSITORY}/actions/runs/1/cancel")


def test_ambiguous_runner_delete_rejects_empty_then_reappearing_inventory() -> None:
    class Transport:
        reads = 0

        def request(
            self, method: str, path: str, body: Mapping[str, Any] | None = None
        ) -> controller.GitHubResponse:
            if method == "DELETE":
                raise controller.AmbiguousGitHubMutation(method, path, "1" * 64, "test")
            self.reads += 1
            runners = [] if self.reads == 1 else [_response_runner("expected", runner_id=41)]
            payload = {"total_count": len(runners), "runners": runners}
            return controller.GitHubResponse(
                method, path, 200, bytearray(_canonical(payload)), "2" * 64
            )

    service = controller.ReleaseGpuController(
        Transport(),
        NoRemote(),
        resources=TEST_RESOURCES,
        clock=lambda: NOW,
        sleep=lambda _: None,
    )
    with pytest.raises(controller.AmbiguousGitHubMutation, match="runner_delete_unresolved"):
        service._delete_unused_runner(41)


def test_cancel_success_visibility_timeout_is_reconciled_without_replay() -> None:
    transport = QueueTransport()
    session = _session()
    runners_path = f"/repos/{controller.REPOSITORY}/actions/runners"
    run_path = f"/repos/{controller.REPOSITORY}/actions/runs/{session.run['id']}"
    cancel_path = f"{run_path}/cancel"
    jobs_path = f"{run_path}/attempts/1/jobs?filter=all&per_page=100&page=1"
    transport.add("GET", runners_path, 200, {"total_count": 0, "runners": []})
    transport.add("POST", cancel_path, 202)
    transport.add(
        "GET",
        run_path,
        200,
        {
            "id": session.run["id"],
            "run_attempt": 1,
            "status": "in_progress",
            "conclusion": None,
            "head_sha": session.head_sha,
        },
    )
    intents: list[tuple[str, Mapping[str, Any]]] = []
    service = controller.ReleaseGpuController(
        transport,
        NoRemote(),
        resources=TEST_RESOURCES,
        clock=lambda: NOW,
        sleep=lambda _: None,
    )
    with pytest.raises(
        controller.AmbiguousGitHubMutation,
        match="failed_phase_cancel_unresolved",
    ) as caught:
        service.cancel_failed_phase(
            session,
            poll_limit=1,
            progress=lambda label, payload: intents.append((label, payload)),
        )
    transport.add("GET", runners_path, 200, {"total_count": 0, "runners": []})
    transport.add(
        "GET",
        run_path,
        200,
        {
            "id": session.run["id"],
            "run_attempt": 1,
            "status": "completed",
            "conclusion": "cancelled",
            "head_sha": session.head_sha,
        },
    )
    jobs = _live_attempt_api_jobs(
        session,
        overrides={
            item.key: {"status": "completed", "conclusion": "cancelled"}
            for item in session.queued_jobs
        },
    )
    transport.add("GET", jobs_path, 200, {"total_count": len(jobs), "jobs": jobs})
    transport.add("GET", runners_path, 200, {"total_count": 0, "runners": []})
    receipt = service.reconcile_cancel_for_abort(session, caught.value, poll_limit=1)
    assert receipt["cancel_retried"] is False
    assert [method for method, path, _ in transport.calls if path == cancel_path] == ["POST"]
    assert [label for label, _ in intents] == ["github-cancel-intent"]


def test_runner_delete_intent_reconciles_by_observation_without_replay() -> None:
    transport = QueueTransport()
    session = _session()
    job = session.jobs[0]
    runner_path = f"/repos/{controller.REPOSITORY}/actions/runners/41"
    inventory_path = f"/repos/{controller.REPOSITORY}/actions/runners"
    transport.add("DELETE", runner_path, 204)
    transport.add(
        "GET",
        inventory_path,
        200,
        {"total_count": 1, "runners": [_response_runner(job.runner_name)]},
    )
    progress: list[tuple[str, Mapping[str, Any]]] = []
    service = controller.ReleaseGpuController(
        transport,
        NoRemote(),
        resources=TEST_RESOURCES,
        clock=lambda: NOW,
        sleep=lambda _: None,
    )
    with pytest.raises(controller.ControllerError, match="runner_inventory_not_stably_zero"):
        service._delete_unused_runner(
            41,
            session=session,
            job=job,
            progress=lambda label, payload: progress.append((label, payload)),
        )
    intent = progress[0][1]
    for _ in range(3):
        transport.add(
            "GET",
            inventory_path,
            200,
            {"total_count": 0, "runners": []},
        )
    receipt = service.reconcile_runner_delete_for_abort(session, job, intent, observations=3)
    assert receipt["delete_retried"] is False
    assert [method for method, path, _ in transport.calls if path == runner_path] == ["DELETE"]
