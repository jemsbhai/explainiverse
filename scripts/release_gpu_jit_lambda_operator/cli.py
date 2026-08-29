"""Production operator entrypoint for one Explainiverse Lambda GPU phase.

The default action is credential-free inspection.  Provider or GitHub writes
are reachable only through the explicit ``execute`` or ``resume-abort``
actions, after exact immutable-plan confirmation and local/live binding checks.
"""

from __future__ import annotations

import argparse
import os
import re
import secrets
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence

from scripts.release_gpu_jit_lambda_controller import (
    ControllerError,
    EvidenceJournal,
    FinalMainAcceptance,
    GhCliTransport,
    LiveReleaseDriver,
    ReleaseGpuController,
    SshRemoteExecutor,
)
from scripts.release_gpu_jit_lambda_controller.controller import SealedControllerResources
from scripts.release_gpu_jit_lambda_live import (
    AccessIdentityReceipt,
    ContractError,
    EvidenceDirectoryReceipt,
    HostIdentity,
    LambdaHttpClient,
    LambdaLiveAdapter,
    LiveGates,
    RuntimeBundle,
    build_plan_from_discovery,
    capture_access_identity,
    capture_action_time_discovery,
    create_evidence_directory,
    dry_run_contract,
    generate_ephemeral_host_identity,
    reopen_evidence_directory,
    runtime_bundle_from_captured_files,
    write_public_evidence,
)

from .boundary import (
    PHASE_REFS,
    SHA256_RE,
    AppCaptureInbox,
    OperatorError,
    canonical_existing_directory,
    canonical_existing_file,
    canonical_json,
    capture_inventory,
    close_owned_fd,
    ensure_path_outside_repository,
    executable_inventory,
    inspection_receipt,
    load_inspection_receipt,
    publish_app_capture_generation,
    read_canonical_json_file,
    read_plan_confirmation,
    read_recovery_plan,
    repository_inventory,
    require_pairwise_disjoint_paths,
    sha256_bytes,
    strict_json,
    validate_anonymous_fd,
    validate_inventory_matches,
    validate_phase_ref,
)
from .receipt_contract import validate_operator_preflight

NONCE_RE = re.compile(r"[0-9a-f]{32}\Z")
CUDA_RUNNER_NONCE_RE = re.compile(r"[0-9a-f]{16}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
RUNTIME_RELATIVE = Path("scripts/release_gpu_jit_lambda_runtime")
PREFLIGHT_KIND = "explainiverse-lambda-operator-preflight"
PREFLIGHT_KEYS = frozenset(
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
    }
)
REPOSITORY_PREFLIGHT_KEYS = frozenset(
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
    }
)
GIT_CONFIGURATION_PREFLIGHT_KEYS = frozenset(
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
    }
)


@dataclass(frozen=True)
class OperatorSealedResources:
    controller: SealedControllerResources
    runtime_bundle: RuntimeBundle
    binding: Mapping[str, Any]


def _sealed_resources(
    captured: Mapping[str, Any], launch_receipt: Mapping[str, Any]
) -> OperatorSealedResources:
    try:
        binding = launch_receipt["preloader"]["sealed_resources"]
        policy_bytes = captured["policy_bytes"]
        controller_source_bytes = captured["controller_source_bytes"]
        runtime_files = captured["runtime_files"]
        controller = SealedControllerResources.from_captured_bytes(
            policy_bytes=policy_bytes,
            controller_source_bytes=controller_source_bytes,
            expected_policy_sha256=binding["policy_sha256"],
            expected_controller_source_sha256=binding["controller_source_sha256"],
        )
        runtime_bundle = runtime_bundle_from_captured_files(
            runtime_files,
            expected_bundle_sha256=binding["runtime_bundle_sha256"],
        )
    except (KeyError, TypeError):
        _fail("operator_captured_resources_rejected")
    return OperatorSealedResources(
        controller=controller,
        runtime_bundle=runtime_bundle,
        binding=dict(binding),
    )


def _fail(code: str) -> NoReturn:
    raise OperatorError(code)


def _require(condition: bool, code: str) -> None:
    if not condition:
        _fail(code)


def _required_text(args: argparse.Namespace, name: str) -> str:
    value = getattr(args, name, None)
    _require(type(value) is str and bool(value), f"operator_{name}_required")
    assert isinstance(value, str)
    return value


def _required_integer(args: argparse.Namespace, name: str, *, minimum: int = 1) -> int:
    value = getattr(args, name, None)
    _require(type(value) is int and value >= minimum, f"operator_{name}_required")
    assert isinstance(value, int)
    return value


def _emit(value: Mapping[str, Any], *, stream: Any = None) -> None:
    destination = stream if stream is not None else sys.stdout.buffer
    destination.write(canonical_json(value))
    destination.flush()


def _cleanup_actions(
    actions: Sequence[tuple[str, Any]],
    *,
    failure_code: str,
) -> None:
    """Run every idempotent close while preserving an active exception."""

    active_exception = sys.exc_info()[0] is not None
    first_error: BaseException | None = None
    for _, action in actions:
        try:
            action()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None and not active_exception:
        raise OperatorError(failure_code) from first_error


def _phase_inputs(args: argparse.Namespace) -> tuple[str, str, str]:
    phase = _required_text(args, "phase")
    supplied_ref = _required_text(args, "supplied_ref")
    expected_head_sha = _required_text(args, "expected_head_sha")
    validate_phase_ref(phase, supplied_ref)
    _require(COMMIT_RE.fullmatch(expected_head_sha) is not None, "expected_head_sha_rejected")
    return phase, supplied_ref, expected_head_sha


def _common_paths(args: argparse.Namespace) -> tuple[str, str, str, str]:
    return (
        _required_text(args, "repository_root"),
        _required_text(args, "git_executable"),
        _required_text(args, "gh_executable"),
        _required_text(args, "ssh_executable"),
    )


def _operator_roots(args: argparse.Namespace) -> tuple[str, str]:
    return (
        _required_text(args, "operator_python_root"),
        _required_text(args, "operator_site_root"),
    )


def _load_and_revalidate_inventory(
    args: argparse.Namespace,
    *,
    phase: str,
    supplied_ref: str,
    expected_head_sha: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    receipt_path = _required_text(args, "inspection_receipt")
    receipt_sha256 = _required_text(args, "inspection_receipt_sha256")
    directory_receipt_sha256 = _required_text(args, "inspection_evidence_directory_receipt_sha256")
    _require(
        SHA256_RE.fullmatch(directory_receipt_sha256) is not None,
        "inspection_evidence_directory_receipt_sha256_rejected",
    )
    receipt_file = canonical_existing_file(receipt_path, context="inspection_receipt")
    inspection_directory: EvidenceDirectoryReceipt | None = None
    try:
        inspection_directory = reopen_evidence_directory(
            receipt_file.parent,
            expected_receipt_sha256=directory_receipt_sha256,
        )
        _require(
            receipt_file.parent == Path(inspection_directory.absolute_path),
            "inspection_receipt_outside_bound_directory",
        )
        inspection, expected = load_inspection_receipt(
            str(receipt_file), expected_file_sha256=receipt_sha256
        )
        inspection_directory.validate()
    finally:
        if inspection_directory is not None:
            inspection_directory.close()
    _require(
        inspection["contract"] == dry_run_contract(),
        "inspection_contract_drift",
    )
    _require(inspection["phase"] == phase, "inspection_phase_drift")
    repository_root, git_executable, gh_executable, ssh_executable = _common_paths(args)
    operator_python_root, operator_site_root = _operator_roots(args)
    observed = capture_inventory(
        repository_root=repository_root,
        operator_python_root=operator_python_root,
        operator_site_root=operator_site_root,
        git_executable=git_executable,
        gh_executable=gh_executable,
        ssh_executable=ssh_executable,
        expected_head_sha=expected_head_sha,
        supplied_ref=supplied_ref,
    )
    validate_inventory_matches(expected, observed)
    _require(expected["repository"]["supplied_ref"] == PHASE_REFS[phase], "inspection_phase_drift")
    return inspection, expected, receipt_sha256


def _locked_github(inventory: Mapping[str, Any]) -> GhCliTransport:
    try:
        gh = inventory["executables"]["gh"]
        return GhCliTransport(
            executable_path=gh["absolute_path"],
            executable_sha256=gh["sha256"],
        )
    except (KeyError, TypeError):
        _fail("inspection_gh_inventory_rejected")


def _locked_ssh(
    inventory: Mapping[str, Any], access_identity: AccessIdentityReceipt
) -> SshRemoteExecutor:
    try:
        ssh = inventory["executables"]["ssh"]
        return SshRemoteExecutor(
            executable_path=ssh["absolute_path"],
            executable_sha256=ssh["sha256"],
            access_identity=access_identity,
        )
    except (KeyError, TypeError):
        _fail("inspection_ssh_inventory_rejected")


def inspect(
    args: argparse.Namespace,
    *,
    environment_receipt: Mapping[str, Any],
    launch_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    phase, supplied_ref, expected_head_sha = _phase_inputs(args)
    repository_root, git_executable, gh_executable, ssh_executable = _common_paths(args)
    operator_python_root, operator_site_root = _operator_roots(args)
    inventory = capture_inventory(
        repository_root=repository_root,
        operator_python_root=operator_python_root,
        operator_site_root=operator_site_root,
        git_executable=git_executable,
        gh_executable=gh_executable,
        ssh_executable=ssh_executable,
        expected_head_sha=expected_head_sha,
        supplied_ref=supplied_ref,
    )
    value = inspection_receipt(
        inventory,
        dry_run_contract(),
        phase=phase,
        environment=environment_receipt,
        secure_launch=launch_receipt,
    )
    repository_root = _required_text(args, "repository_root")
    evidence_path = ensure_path_outside_repository(
        _required_text(args, "inspection_evidence_directory"),
        repository_root,
        context="inspection_evidence_directory",
    )
    _require(not evidence_path.exists(), "inspection_evidence_directory_already_exists")
    receipt: EvidenceDirectoryReceipt | None = None
    try:
        receipt = create_evidence_directory(evidence_path)
        expected_sha256 = sha256_bytes(canonical_json(value))
        filename = f"operator-inspection-{expected_sha256}.json"
        destination = evidence_path / filename
        observed_sha256 = write_public_evidence(
            destination,
            value,
            evidence_directory_receipt=receipt,
        )
        _require(observed_sha256 == expected_sha256, "inspection_evidence_write_digest_drift")
        return {
            "schema_version": 1,
            "kind": "explainiverse-lambda-operator-inspection-published",
            "inspection_receipt": str(destination),
            "inspection_receipt_sha256": observed_sha256,
            "inspection_evidence_directory": str(evidence_path),
            "inspection_evidence_directory_receipt": receipt.to_public_mapping(),
            "crash_safe_no_replace": True,
            "provider_contacted": False,
            "provider_mutation": False,
            "github_mutation": False,
        }
    finally:
        if receipt is not None:
            receipt.close()


def create_app_inbox(
    args: argparse.Namespace,
    *,
    environment_receipt: Mapping[str, Any],
    launch_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    repository_root = _required_text(args, "repository_root")
    path = ensure_path_outside_repository(
        _required_text(args, "app_capture_inbox"),
        repository_root,
        context="app_capture_inbox",
    )
    _require(not path.exists(), "app_capture_inbox_already_exists")
    receipt: EvidenceDirectoryReceipt | None = None
    try:
        receipt = create_evidence_directory(path)
        return {
            "schema_version": 1,
            "kind": "explainiverse-app-capture-inbox-created",
            "absolute_path": str(path),
            "receipt": receipt.to_public_mapping(),
            "ready_protocol": {
                "bundle": "capture-<job ordinal:02d>-<generation:06d>",
                "ready_marker": "ready-<job ordinal:02d>-<generation:06d>.json",
                "ready_marker_published_exclusively_after_bundle": True,
            },
            "environment": dict(environment_receipt),
            "secure_launch": dict(launch_receipt),
        }
    finally:
        if receipt is not None:
            receipt.close()


def create_app_staging(
    args: argparse.Namespace,
    *,
    environment_receipt: Mapping[str, Any],
    launch_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the separate owner-private input directory for one App capture."""

    repository_root = _required_text(args, "repository_root")
    path = ensure_path_outside_repository(
        _required_text(args, "app_capture_staging"),
        repository_root,
        context="app_capture_staging",
    )
    _require(not path.exists(), "app_capture_staging_already_exists")
    receipt: EvidenceDirectoryReceipt | None = None
    try:
        receipt = create_evidence_directory(path)
        return {
            "schema_version": 1,
            "kind": "explainiverse-app-capture-staging-created",
            "absolute_path": str(path),
            "receipt": receipt.to_public_mapping(),
            "required_inventory": {
                "capture_json": "capture.json",
                "raw_pages_directory": "pages",
                "additional_entries_allowed": False,
            },
            "environment": dict(environment_receipt),
            "secure_launch": dict(launch_receipt),
        }
    finally:
        if receipt is not None:
            receipt.close()


def publish_app_capture(
    args: argparse.Namespace,
    *,
    environment_receipt: Mapping[str, Any],
    launch_receipt: Mapping[str, Any],
    resources: OperatorSealedResources,
) -> dict[str, Any]:
    phase = _required_text(args, "phase")
    _require(phase in PHASE_REFS, "operator_phase_rejected")
    repository_root = _required_text(args, "repository_root")
    inbox_path = ensure_path_outside_repository(
        _required_text(args, "app_capture_inbox"),
        repository_root,
        context="app_capture_inbox",
    )
    receipt_sha256 = _required_text(args, "app_capture_inbox_receipt_sha256")
    staging_path = ensure_path_outside_repository(
        _required_text(args, "app_capture_staging"),
        repository_root,
        context="app_capture_staging",
    )
    staging_receipt_sha256 = _required_text(args, "app_capture_staging_receipt_sha256")
    receipt: EvidenceDirectoryReceipt | None = None
    staging_receipt: EvidenceDirectoryReceipt | None = None
    try:
        receipt = reopen_evidence_directory(
            inbox_path,
            expected_receipt_sha256=receipt_sha256,
        )
        staging_receipt = reopen_evidence_directory(
            staging_path,
            expected_receipt_sha256=staging_receipt_sha256,
        )
        result = publish_app_capture_generation(
            receipt,
            controller_resources=resources.controller,
            staging_receipt=staging_receipt,
            phase=phase,
            ordinal=_required_integer(args, "capture_ordinal"),
            generation=_required_integer(args, "capture_generation"),
            publication_nonce=_required_text(args, "capture_publication_nonce"),
        )
        result["environment"] = dict(environment_receipt)
        result["secure_launch"] = dict(launch_receipt)
        return result
    finally:
        if staging_receipt is not None:
            staging_receipt.close()
        if receipt is not None:
            receipt.close()


def _validate_runtime_root(repository_root: str, runtime_root: str) -> Path:
    root = canonical_existing_directory(repository_root, context="repository_root")
    expected = (root / RUNTIME_RELATIVE).resolve(strict=True)
    observed = canonical_existing_directory(runtime_root, context="runtime_root")
    _require(observed == expected, "runtime_root_not_candidate_source")
    return observed


def _publication_acceptance(
    args: argparse.Namespace,
    *,
    repository_root: str,
    expected_head_sha: str,
    resources: OperatorSealedResources,
) -> FinalMainAcceptance:
    directory_path = ensure_path_outside_repository(
        _required_text(args, "final_main_evidence_directory"),
        repository_root,
        context="final_main_evidence_directory",
    )
    receipt_sha256 = _required_text(args, "final_main_evidence_receipt_sha256")
    plan_sha256 = _required_text(args, "final_main_plan_sha256")
    journal_sha256 = _required_text(args, "final_main_journal_sha256")
    receipt: EvidenceDirectoryReceipt | None = None
    try:
        receipt = reopen_evidence_directory(directory_path, expected_receipt_sha256=receipt_sha256)
        acceptance = EvidenceJournal.load_final_main_acceptance(
            receipt,
            controller_resources=resources.controller,
            final_control_plane_plan_sha256=plan_sha256,
            final_journal_sha256=journal_sha256,
        )
        _require(acceptance.head_sha == expected_head_sha, "final_main_acceptance_head_drift")
        return acceptance
    finally:
        if receipt is not None:
            receipt.close()


def _validate_phase_specific_inputs(
    args: argparse.Namespace,
    phase: str,
    acceptance: FinalMainAcceptance | None,
) -> tuple[tuple[str, ...], int | None, int | None]:
    prior = tuple(args.prior_accepted_cuda_runner_nonce or ())
    _require(
        len(set(prior)) == len(prior)
        and all(CUDA_RUNNER_NONCE_RE.fullmatch(item) for item in prior),
        "prior_cuda_nonce_rejected",
    )
    preflight_run_id = args.preflight_run_id
    cuda_run_id = args.cuda_run_id
    if phase == "publication":
        _require(acceptance is not None, "publication_final_acceptance_missing")
        assert acceptance is not None
        _require(
            prior == acceptance.accepted_cuda_runner_nonces,
            "publication_prior_cuda_nonce_mismatch",
        )
        _require(
            type(preflight_run_id) is int
            and preflight_run_id > 0
            and type(cuda_run_id) is int
            and cuda_run_id > 0,
            "publication_run_ids_missing",
        )
    else:
        _require(not prior, "nonpublication_prior_cuda_nonce_rejected")
        _require(
            preflight_run_id is None and cuda_run_id is None,
            "nonpublication_run_ids_rejected",
        )
    return prior, preflight_run_id, cuda_run_id


def _write_operator_preflight(
    receipt: EvidenceDirectoryReceipt,
    *,
    value: Mapping[str, Any],
) -> tuple[str, str]:
    expected_sha256 = sha256_bytes(canonical_json(value))
    name = f"operator-preflight-{expected_sha256}.json"
    path = str(Path(receipt.absolute_path) / name)
    observed_sha256 = write_public_evidence(
        path,
        value,
        evidence_directory_receipt=receipt,
    )
    _require(observed_sha256 == expected_sha256, "operator_preflight_digest_mismatch")
    return name, observed_sha256


def _preflight_mapping(
    *,
    plan: Any,
    discovery: Any,
    inspection_receipt_sha256: str,
    inventory: Mapping[str, Any],
    environment_receipt: Mapping[str, Any],
    launch_receipt: Mapping[str, Any],
    lambda_fd_receipt: Mapping[str, Any],
    confirmation_receipt: Mapping[str, Any],
    app_inbox: AppCaptureInbox,
    final_acceptance: FinalMainAcceptance | None,
    resources: OperatorSealedResources,
) -> dict[str, Any]:
    normalized_inventory = strict_json(
        canonical_json(inventory), context="operator_preflight_inventory"
    )
    executables = inventory.get("executables")
    _require(
        type(executables) is dict and set(executables) == {"git", "gh", "ssh", "python"},
        "operator_preflight_executable_inventory_rejected",
    )
    normalized_executables = strict_json(
        canonical_json(executables), context="operator_preflight_executables"
    )
    repository = inventory.get("repository")
    _require(
        type(repository) is dict and set(repository) == REPOSITORY_PREFLIGHT_KEYS,
        "operator_preflight_repository_inventory_rejected",
    )
    assert isinstance(repository, dict)
    git_configuration = repository.get("git_configuration")
    _require(
        type(git_configuration) is dict
        and set(git_configuration) == GIT_CONFIGURATION_PREFLIGHT_KEYS
        and git_configuration
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
        "operator_preflight_git_configuration_rejected",
    )
    _require(
        repository.get("head_sha") == plan.head_sha
        and repository.get("remote_target_sha") == plan.head_sha
        and repository.get("clean_tracked_and_untracked") is True,
        "operator_preflight_repository_binding_rejected",
    )
    normalized_repository = strict_json(
        canonical_json(repository), context="operator_preflight_repository"
    )
    value = {
        "schema_version": 1,
        "kind": PREFLIGHT_KIND,
        "plan_sha256": plan.sha256,
        "head_sha": plan.head_sha,
        "lifecycle_nonce": plan.lifecycle_nonce,
        "discovery": discovery.to_public_mapping(),
        "inspection_receipt_sha256": inspection_receipt_sha256,
        "inventory": normalized_inventory,
        "inventory_sha256": sha256_bytes(canonical_json(normalized_inventory)),
        "executables": normalized_executables,
        "repository": normalized_repository,
        "environment": dict(environment_receipt),
        "secure_launch": dict(launch_receipt),
        "lambda_secret_transport": dict(lambda_fd_receipt),
        "plan_confirmation": dict(confirmation_receipt),
        "app_capture_inbox": app_inbox.to_public_mapping(),
        "final_main_acceptance": (
            {
                "loader_verified": True,
                "evidence_sha256": final_acceptance.evidence_sha256,
                "head_sha": final_acceptance.head_sha,
                "run_id": final_acceptance.run_id,
            }
            if final_acceptance is not None
            else None
        ),
        "live_gates_not_constructed_before_confirmation": True,
        "direct_publication_dispatch_exposed": False,
    }
    validate_operator_preflight(
        value,
        expected_immutable_plan=plan.to_mapping(),
        expected_phase=app_inbox.phase,
        expected_head_sha=plan.head_sha,
        expected_ref=repository["supplied_ref"],
        expected_plan_sha256=plan.sha256,
        expected_lifecycle_nonce=plan.lifecycle_nonce,
        expected_inspection_receipt_sha256=inspection_receipt_sha256,
        expected_inventory_sha256=sha256_bytes(canonical_json(normalized_inventory)),
        expected_policy_sha256=resources.binding["policy_sha256"],
        expected_controller_source_sha256=resources.binding["controller_source_sha256"],
        expected_runtime_bundle_sha256=resources.binding["runtime_bundle_sha256"],
    )
    return value


def _revalidate_locked_posture(
    args: argparse.Namespace,
    *,
    inventory: Mapping[str, Any],
    expected_head_sha: str,
    supplied_ref: str,
) -> None:
    repository_root, git_executable, _, _ = _common_paths(args)
    github = _locked_github(inventory)
    observed_repository = repository_inventory(
        repository_root=repository_root,
        git_executable=git_executable,
        github=github,
        expected_head_sha=expected_head_sha,
        supplied_ref=supplied_ref,
    )
    _require(
        canonical_json(observed_repository) == canonical_json(inventory["repository"]),
        "action_time_repository_drift",
    )
    observed_executables = executable_inventory(
        git_executable=inventory["executables"]["git"]["absolute_path"],
        gh_executable=inventory["executables"]["gh"]["absolute_path"],
        ssh_executable=inventory["executables"]["ssh"]["absolute_path"],
    )
    _require(
        canonical_json(observed_executables) == canonical_json(inventory["executables"]),
        "action_time_executable_identity_drift",
    )


def execute(
    args: argparse.Namespace,
    *,
    environment_receipt: Mapping[str, Any],
    launch_receipt: Mapping[str, Any],
    resources: OperatorSealedResources,
) -> dict[str, Any]:
    phase, supplied_ref, expected_head_sha = _phase_inputs(args)
    if os.name == "nt":
        _require(
            launch_receipt.get("windows_handle_transport") is True
            and launch_receipt.get("inherited_handle_count") == 2,
            "windows_launcher_required",
        )
    repository_root, _, _, _ = _common_paths(args)
    _, inventory, inspection_sha256 = _load_and_revalidate_inventory(
        args,
        phase=phase,
        supplied_ref=supplied_ref,
        expected_head_sha=expected_head_sha,
    )
    _validate_runtime_root(repository_root, _required_text(args, "runtime_root"))
    evidence_path = ensure_path_outside_repository(
        _required_text(args, "evidence_directory"),
        repository_root,
        context="evidence_directory",
    )
    _require(not evidence_path.exists(), "evidence_directory_already_exists")
    inbox_path = ensure_path_outside_repository(
        _required_text(args, "app_capture_inbox"),
        repository_root,
        context="app_capture_inbox",
    )
    security_roots: dict[str, Path] = {
        "repository_root": Path(repository_root),
        "evidence_directory": evidence_path,
        "app_capture_inbox": inbox_path,
    }
    if phase == "publication":
        security_roots["final_main_evidence_directory"] = ensure_path_outside_repository(
            _required_text(args, "final_main_evidence_directory"),
            repository_root,
            context="final_main_evidence_directory",
        )
    require_pairwise_disjoint_paths(security_roots)
    inbox_receipt_sha256 = _required_text(args, "app_capture_inbox_receipt_sha256")
    ssh_access_key = str(
        canonical_existing_file(_required_text(args, "ssh_access_key"), context="ssh_access_key")
    )
    ssh_key_name = _required_text(args, "ssh_key_name")
    image_id = _required_text(args, "image_id")
    source_cidr = _required_text(args, "controller_public_ipv4_cidr")
    lifecycle_nonce = _required_text(args, "lifecycle_nonce")
    _require(NONCE_RE.fullmatch(lifecycle_nonce) is not None, "lifecycle_nonce_rejected")
    lifetime = _required_integer(args, "plan_lifetime_seconds")
    _require(lifetime <= 4 * 60 * 60, "plan_lifetime_rejected")
    lambda_fd = _required_integer(args, "lambda_api_key_fd", minimum=3)
    confirmation_fd = _required_integer(args, "plan_confirmation_fd", minimum=3)
    _require(lambda_fd != confirmation_fd, "secret_and_confirmation_fd_alias")
    lambda_fd_receipt = validate_anonymous_fd(lambda_fd, context="lambda_api_key")
    confirmation_fd_preflight = validate_anonymous_fd(confirmation_fd, context="plan_confirmation")
    _require(
        confirmation_fd_preflight["regular_file"] is False,
        "plan_confirmation_transport_rejected",
    )

    final_acceptance = (
        _publication_acceptance(
            args,
            repository_root=repository_root,
            expected_head_sha=expected_head_sha,
            resources=resources,
        )
        if phase == "publication"
        else None
    )
    prior, preflight_run_id, cuda_run_id = _validate_phase_specific_inputs(
        args, phase, final_acceptance
    )

    inbox_receipt: EvidenceDirectoryReceipt | None = None
    inbox: AppCaptureInbox | None = None
    client: LambdaHttpClient | None = None
    identity: HostIdentity | None = None
    runtime_bundle: RuntimeBundle | None = None
    evidence_receipt: EvidenceDirectoryReceipt | None = None
    journal: EvidenceJournal | None = None
    access_identity: AccessIdentityReceipt | None = None
    driver: LiveReleaseDriver | None = None
    try:
        inbox_receipt = reopen_evidence_directory(
            inbox_path,
            expected_receipt_sha256=inbox_receipt_sha256,
        )
        inbox = AppCaptureInbox(
            inbox_receipt,
            resources.controller,
            phase=phase,
            poll_limit=_required_integer(args, "app_capture_poll_limit"),
            poll_seconds=args.app_capture_poll_seconds,
        )
        try:
            client = LambdaHttpClient.from_secret_fd(lambda_fd)
        finally:
            close_owned_fd(lambda_fd)
            lambda_fd = -1
        discovery = capture_action_time_discovery(client, ssh_key_name=ssh_key_name)
        runtime_bundle = resources.runtime_bundle
        identity = generate_ephemeral_host_identity()
        created_at = int(time.time())
        plan = build_plan_from_discovery(
            discovery,
            head_sha=expected_head_sha,
            lifecycle_nonce=lifecycle_nonce,
            created_at_unix=created_at,
            expires_at_unix=created_at + lifetime,
            current_public_ipv4_cidr=source_cidr,
            image_id=image_id,
            host_identity=identity,
            runtime_bundle=runtime_bundle,
        )
        _emit(
            {
                "schema_version": 1,
                "kind": "explainiverse-lambda-plan-awaiting-confirmation",
                "phase": phase,
                "plan_sha256": plan.sha256,
                "plan": plan.to_mapping(),
                "price_and_capacity_from_fresh_discovery": True,
                "live_mutation_gates_constructed": False,
                "confirmation_protocol": "write the exact lowercase plan SHA plus LF to the anonymous confirmation FD",
            }
        )
        try:
            confirmation_receipt = read_plan_confirmation(
                confirmation_fd, expected_sha256=plan.sha256
            )
        finally:
            close_owned_fd(confirmation_fd)
            confirmation_fd = -1
        _revalidate_locked_posture(
            args,
            inventory=inventory,
            expected_head_sha=expected_head_sha,
            supplied_ref=supplied_ref,
        )
        # This second construction is intentionally immediately before the
        # gates: it re-enforces the discovery freshness window after the
        # operator's action-time digest confirmation.
        confirmed_plan = build_plan_from_discovery(
            discovery,
            head_sha=expected_head_sha,
            lifecycle_nonce=lifecycle_nonce,
            created_at_unix=created_at,
            expires_at_unix=created_at + lifetime,
            current_public_ipv4_cidr=source_cidr,
            image_id=image_id,
            host_identity=identity,
            runtime_bundle=runtime_bundle,
        )
        _require(confirmed_plan.sha256 == plan.sha256, "confirmed_plan_drift")
        inbox_receipt.validate()

        # This is the sole gate construction in a fresh execution.  Every
        # read-only source and the post-discovery confirmation are complete.
        gates = LiveGates(True, True, confirmed_plan.sha256)
        provider = LambdaLiveAdapter(client, confirmed_plan, gates)
        evidence_receipt = create_evidence_directory(evidence_path)
        evidence_receipt_sha256 = evidence_receipt.receipt_sha256
        journal = EvidenceJournal(
            evidence_receipt,
            plan_sha256=confirmed_plan.sha256,
        )

        def archive_stale_app_capture(
            classified_at: str,
            generation_receipt: Mapping[str, Any],
            evidence_pages: Mapping[str, bytes],
        ) -> Mapping[str, Any]:
            assert journal is not None
            return journal.archive_stale_installed_app_capture(
                phase=phase,
                classified_at=classified_at,
                generation_receipt=generation_receipt,
                evidence_pages=evidence_pages,
                controller_resources=resources.controller,
            )

        inbox.bind_stale_archive_sink(archive_stale_app_capture)
        preflight = _preflight_mapping(
            plan=confirmed_plan,
            discovery=discovery,
            inspection_receipt_sha256=inspection_sha256,
            inventory=inventory,
            environment_receipt=environment_receipt,
            launch_receipt=launch_receipt,
            lambda_fd_receipt=lambda_fd_receipt,
            confirmation_receipt=confirmation_receipt,
            app_inbox=inbox,
            final_acceptance=final_acceptance,
            resources=resources,
        )
        preflight_name, preflight_sha256 = _write_operator_preflight(
            evidence_receipt, value=preflight
        )
        access_identity = capture_access_identity(
            ssh_access_key,
            expected_public_key_sha256=confirmed_plan.ssh_public_key_sha256,
        )
        github = _locked_github(inventory)
        remote = _locked_ssh(inventory, access_identity)
        controller = ReleaseGpuController(github, remote, resources=resources.controller)
        driver = LiveReleaseDriver(
            controller,
            provider,
            confirmed_plan,
            identity,
            runtime_bundle,
            journal,
            access_identity=access_identity,
            known_hosts_path=evidence_path / "known_hosts",
            observation_poll_limit=args.observation_poll_limit,
        )
        driver.provision()
        journal.record(
            "operator-preflight-binding",
            {
                "plan_sha256": confirmed_plan.sha256,
                "operator_preflight_filename": preflight_name,
                "operator_preflight_sha256": preflight_sha256,
                "inspection_receipt_sha256": inspection_sha256,
                "inventory_sha256": sha256_bytes(canonical_json(inventory)),
                "app_capture_inbox": inbox.to_public_mapping(),
                "bound_before_first_jit": True,
            },
        )
        completion = driver.run_phase(
            phase,
            supplied_ref=supplied_ref,
            app_capture_supplier=inbox,
            prior_accepted_cuda_runner_nonces=prior,
            preflight_run_id=preflight_run_id,
            cuda_run_id=cuda_run_id,
            final_main_acceptance=final_acceptance,
            dispatch_poll_limit=args.dispatch_poll_limit,
        )
        final_inbox_inventory = inbox.validate_consumed()
        journal.record(
            "operator-app-inbox-settlement",
            {
                **inbox.to_public_mapping(),
                "all_expected_captures_consumed": True,
                "capture_bytes_retained_only_as_driver_archive": False,
                "all_consumed_raw_pages_archived_in_evidence_root": True,
                "accepted_source_generations_retained_in_owner_private_inbox": True,
                "final_inbox_inventory": final_inbox_inventory,
            },
        )
        final_journal_sha256 = driver.teardown()
        return {
            "schema_version": 1,
            "kind": "explainiverse-lambda-operator-completion",
            "phase": phase,
            "head_sha": completion.head_sha,
            "run_id": completion.run_id,
            "plan_sha256": confirmed_plan.sha256,
            "phase_evidence_sha256": completion.final_evidence_sha256,
            "final_journal_sha256": final_journal_sha256,
            "evidence_directory_receipt_sha256": evidence_receipt_sha256,
            "operator_preflight_sha256": preflight_sha256,
            "provider_and_github_restored": True,
            "final_main_acceptance_requires_reopen_loader": phase == "final-main",
            "publication_acceptance_loaded_from_closed_final_journal": phase == "publication",
        }
    except BaseException:
        if driver is not None and driver.state != "restored":
            try:
                driver.abort()
            except BaseException as cleanup_error:
                raise OperatorError("operator_abort_failed") from cleanup_error
        raise
    finally:
        cleanup: list[tuple[str, Any]] = [
            (
                "lambda-fd",
                lambda: close_owned_fd(lambda_fd if lambda_fd >= 3 else None),
            ),
            (
                "confirmation-fd",
                lambda: close_owned_fd(confirmation_fd if confirmation_fd >= 3 else None),
            ),
        ]
        if client is not None:
            cleanup.append(("lambda-client", client.close))
        if identity is not None and not identity.destroyed:
            cleanup.append(("host-identity", identity.destroy))
        if access_identity is not None and not access_identity.closed:
            cleanup.append(("access-identity", access_identity.close))
        if journal is not None:
            cleanup.append(("evidence-journal", journal.close))
        elif evidence_receipt is not None:
            cleanup.append(("evidence-directory", evidence_receipt.close))
        if inbox is not None:
            cleanup.append(("app-inbox", inbox.close))
        elif inbox_receipt is not None:
            cleanup.append(("app-inbox-receipt", inbox_receipt.close))
        _cleanup_actions(cleanup, failure_code="operator_local_cleanup_failed")


class _AbortOnlyRemote:
    """Recovery cannot execute SSH; any attempted use is a contract defect."""

    @staticmethod
    def wait_cloud_init(*_: Any, **__: Any) -> NoReturn:
        _fail("recovery_remote_execution_rejected")

    @staticmethod
    def probe_host(*_: Any, **__: Any) -> NoReturn:
        _fail("recovery_remote_execution_rejected")

    @staticmethod
    def run_job(*_: Any, **__: Any) -> NoReturn:
        _fail("recovery_remote_execution_rejected")


def _validate_recovery_preflight(
    evidence_directory: EvidenceDirectoryReceipt,
    *,
    plan_sha256: str,
    inventory: Mapping[str, Any],
    inspection_receipt_sha256: str,
) -> dict[str, Any]:
    root = Path(evidence_directory.absolute_path)
    candidates = tuple(root.glob("operator-preflight-*.json"))
    _require(len(candidates) == 1, "recovery_operator_preflight_cardinality")
    value, raw = read_canonical_json_file(candidates[0], context="recovery_operator_preflight")
    digest = sha256_bytes(raw)
    _require(
        set(value) == PREFLIGHT_KEYS
        and candidates[0].name == f"operator-preflight-{digest}.json"
        and value.get("schema_version") == 1
        and value.get("kind") == PREFLIGHT_KIND
        and value.get("plan_sha256") == plan_sha256
        and value.get("inspection_receipt_sha256") == inspection_receipt_sha256
        and type(value.get("inventory")) is dict
        and canonical_json(value["inventory"]) == canonical_json(inventory)
        and value.get("inventory_sha256") == sha256_bytes(canonical_json(inventory)),
        "recovery_operator_preflight_binding_rejected",
    )
    _require(
        value.get("executables") == inventory.get("executables")
        and value.get("repository") == inventory.get("repository"),
        "recovery_operator_preflight_security_binding_rejected",
    )
    return {"filename": candidates[0].name, "sha256": digest}


def _reopen_recovery_journal_barrier(
    evidence_path: Path,
    *,
    receipt_sha256: str,
    plan_sha256: str,
) -> tuple[
    EvidenceDirectoryReceipt,
    EvidenceJournal,
    dict[str, Any] | None,
    str | None,
    str,
]:
    """Archive an exact interrupted publish and strictly reopen before live work."""

    receipt: EvidenceDirectoryReceipt | None = None
    journal: EvidenceJournal | None = None
    interrupted: dict[str, Any] | None = None
    recovery_record_sha256: str | None = None
    try:
        receipt = reopen_evidence_directory(
            evidence_path,
            expected_receipt_sha256=receipt_sha256,
        )
        journal = EvidenceJournal.reopen_for_recovery(
            receipt,
            plan_sha256=plan_sha256,
        )
        interrupted = journal.interrupted_publish_recovery
        if interrupted is not None:
            recovery_record_sha256 = journal.record_interrupted_publish_recovery()
            _require(
                type(recovery_record_sha256) is str
                and SHA256_RE.fullmatch(recovery_record_sha256) is not None,
                "interrupted_publish_recovery_record_digest_rejected",
            )
        barrier_tail_sha256 = journal.last_evidence_sha256
        _require(
            type(barrier_tail_sha256) is str
            and SHA256_RE.fullmatch(barrier_tail_sha256) is not None
            and (recovery_record_sha256 is None or recovery_record_sha256 == barrier_tail_sha256),
            "recovery_barrier_tail_rejected",
        )
        assert isinstance(barrier_tail_sha256, str)

        # Journal.close() closes the held directory receipt last.  A new held
        # receipt and journal are therefore mandatory for the strict second
        # validation; no closed handle is reused across this barrier.
        journal.close()
        journal = None
        receipt = None
        receipt = reopen_evidence_directory(
            evidence_path,
            expected_receipt_sha256=receipt_sha256,
        )
        journal = EvidenceJournal.reopen_for_recovery(
            receipt,
            plan_sha256=plan_sha256,
        )
        _require(
            journal.interrupted_publish_recovery is None
            and journal.last_evidence_sha256 == barrier_tail_sha256,
            "interrupted_publish_recovery_strict_reopen_rejected",
        )
        return (
            receipt,
            journal,
            interrupted,
            recovery_record_sha256,
            barrier_tail_sha256,
        )
    except BaseException:
        cleanup: list[tuple[str, Any]] = []
        if journal is not None:
            cleanup.append(("recovery-barrier-journal", journal.close))
        elif receipt is not None:
            cleanup.append(("recovery-barrier-directory", receipt.close))
        _cleanup_actions(cleanup, failure_code="operator_recovery_barrier_cleanup_failed")
        raise


def resume_abort(
    args: argparse.Namespace,
    *,
    environment_receipt: Mapping[str, Any],
    launch_receipt: Mapping[str, Any],
    resources: OperatorSealedResources,
) -> dict[str, Any]:
    phase, supplied_ref, expected_head_sha = _phase_inputs(args)
    if os.name == "nt":
        _require(
            launch_receipt.get("windows_handle_transport") is True
            and launch_receipt.get("inherited_handle_count") == 1,
            "windows_launcher_required",
        )
    repository_root, _, _, _ = _common_paths(args)
    plan_sha256 = _required_text(args, "confirm_plan_sha256")
    _require(SHA256_RE.fullmatch(plan_sha256) is not None, "recovery_plan_sha256_rejected")
    evidence_path = ensure_path_outside_repository(
        _required_text(args, "evidence_directory"),
        repository_root,
        context="evidence_directory",
    )
    receipt_sha256 = _required_text(args, "evidence_directory_receipt_sha256")
    lambda_fd = _required_integer(args, "lambda_api_key_fd", minimum=3)
    lambda_fd_receipt = validate_anonymous_fd(lambda_fd, context="lambda_api_key")
    receipt: EvidenceDirectoryReceipt | None = None
    journal: EvidenceJournal | None = None
    client: LambdaHttpClient | None = None
    driver: LiveReleaseDriver | None = None
    interrupted_publish_recovery: dict[str, Any] | None = None
    interrupted_publish_recovery_record_sha256: str | None = None
    recovery_reopen_tail_sha256: str | None = None
    try:
        (
            receipt,
            journal,
            interrupted_publish_recovery,
            interrupted_publish_recovery_record_sha256,
            recovery_reopen_tail_sha256,
        ) = _reopen_recovery_journal_barrier(
            evidence_path,
            receipt_sha256=receipt_sha256,
            plan_sha256=plan_sha256,
        )
        plan = read_recovery_plan(receipt, expected_plan_sha256=plan_sha256)
        _require(plan.head_sha == expected_head_sha, "recovery_plan_head_drift")
        _, inventory, inspection_sha256 = _load_and_revalidate_inventory(
            args,
            phase=phase,
            supplied_ref=supplied_ref,
            expected_head_sha=expected_head_sha,
        )
        preflight = _validate_recovery_preflight(
            receipt,
            plan_sha256=plan_sha256,
            inventory=inventory,
            inspection_receipt_sha256=inspection_sha256,
        )
        _revalidate_locked_posture(
            args,
            inventory=inventory,
            expected_head_sha=expected_head_sha,
            supplied_ref=supplied_ref,
        )
        try:
            client = LambdaHttpClient.from_secret_fd(lambda_fd)
        finally:
            close_owned_fd(lambda_fd)
            lambda_fd = -1
        gates = LiveGates(True, True, plan_sha256)
        provider = LambdaLiveAdapter(client, plan, gates)
        controller = ReleaseGpuController(
            _locked_github(inventory),
            _AbortOnlyRemote(),
            resources=resources.controller,
        )
        driver = LiveReleaseDriver.resume_for_abort(
            controller,
            provider,
            plan,
            journal,
            observation_poll_limit=args.observation_poll_limit,
        )
        final_journal_sha256 = driver.abort()
        return {
            "schema_version": 1,
            "kind": "explainiverse-lambda-operator-recovery-completion",
            "phase": phase,
            "head_sha": plan.head_sha,
            "plan_sha256": plan_sha256,
            "final_journal_sha256": final_journal_sha256,
            "evidence_directory_receipt_sha256": receipt_sha256,
            "operator_preflight": preflight,
            "lambda_secret_transport": lambda_fd_receipt,
            "environment": dict(environment_receipt),
            "secure_launch": dict(launch_receipt),
            "interrupted_publish_recovery": interrupted_publish_recovery,
            "interrupted_publish_recovery_record_sha256": (
                interrupted_publish_recovery_record_sha256
            ),
            "recovery_reopen_tail_sha256": recovery_reopen_tail_sha256,
            "cleanup_only": True,
            "remote_execution_available": False,
            "provider_and_github_restored": True,
        }
    finally:
        cleanup = [
            (
                "lambda-fd",
                lambda: close_owned_fd(lambda_fd if lambda_fd >= 3 else None),
            )
        ]
        if client is not None:
            cleanup.append(("lambda-client", client.close))
        if journal is not None:
            cleanup.append(("evidence-journal", journal.close))
        elif receipt is not None:
            cleanup.append(("evidence-directory", receipt.close))
        _cleanup_actions(cleanup, failure_code="operator_recovery_cleanup_failed")


def transport_self_test(
    args: argparse.Namespace,
    *,
    environment_receipt: Mapping[str, Any],
    launch_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove the Windows inherited-HANDLE bridge without external contact."""

    _require(os.name == "nt", "transport_self_test_windows_only")
    _require(
        launch_receipt.get("windows_handle_transport") is True,
        "windows_handle_transport_required",
    )
    nonce = _required_text(args, "transport_self_test_nonce")
    _require(NONCE_RE.fullmatch(nonce) is not None, "transport_self_test_nonce_rejected")
    lambda_fd = _required_integer(args, "lambda_api_key_fd", minimum=3)
    confirmation_fd = _required_integer(args, "plan_confirmation_fd", minimum=3)
    _require(lambda_fd != confirmation_fd, "secret_and_confirmation_fd_alias")
    lambda_receipt = validate_anonymous_fd(lambda_fd, context="lambda_api_key")
    client: LambdaHttpClient | None = None
    try:
        try:
            client = LambdaHttpClient.from_secret_fd(lambda_fd)
        finally:
            close_owned_fd(lambda_fd)
            lambda_fd = -1
        material = {
            "schema_version": 1,
            "kind": "explainiverse-windows-handle-transport-self-test-plan",
            "nonce": nonce,
        }
        plan_sha256 = sha256_bytes(canonical_json(material))
        _emit(
            {
                "schema_version": 1,
                "kind": "explainiverse-lambda-plan-awaiting-confirmation",
                "phase": "transport-self-test",
                "plan_sha256": plan_sha256,
                "plan": material,
                "live_mutation_gates_constructed": False,
                "external_contact": False,
            }
        )
        try:
            confirmation = read_plan_confirmation(confirmation_fd, expected_sha256=plan_sha256)
        finally:
            close_owned_fd(confirmation_fd)
            confirmation_fd = -1
        return {
            "schema_version": 1,
            "kind": "explainiverse-windows-handle-transport-self-test",
            "nonce": nonce,
            "lambda_secret_received": True,
            "lambda_secret_value_logged": False,
            "plan_confirmation": confirmation,
            "lambda_transport": lambda_receipt,
            "secure_launch": dict(launch_receipt),
            "environment": dict(environment_receipt),
            "external_contact": False,
            "provider_mutation": False,
            "github_mutation": False,
        }
    finally:
        close_owned_fd(lambda_fd if lambda_fd >= 3 else None)
        close_owned_fd(confirmation_fd if confirmation_fd >= 3 else None)
        if client is not None:
            client.close()


def dispatch_release_recovery(
    args: argparse.Namespace,
    *,
    environment_receipt: Mapping[str, Any],
    launch_receipt: Mapping[str, Any],
    resources: OperatorSealedResources,
) -> dict[str, Any]:
    """Dispatch or observation-only reconcile the no-republish recovery run."""

    phase, supplied_ref, expected_head_sha = _phase_inputs(args)
    _require(phase == "publication", "recovery_dispatch_phase_rejected")
    repository_root, _, _, _ = _common_paths(args)
    plan_sha256 = _required_text(args, "confirm_plan_sha256")
    _require(SHA256_RE.fullmatch(plan_sha256) is not None, "recovery_plan_sha256_rejected")
    publication_journal_sha256 = _required_text(args, "publication_journal_sha256")
    _require(
        SHA256_RE.fullmatch(publication_journal_sha256) is not None,
        "publication_journal_sha256_rejected",
    )
    source_run_id = _required_integer(args, "source_run_id")
    poll_limit = _required_integer(args, "recovery_poll_limit")
    evidence_path = ensure_path_outside_repository(
        _required_text(args, "evidence_directory"),
        repository_root,
        context="evidence_directory",
    )
    receipt_sha256 = _required_text(args, "evidence_directory_receipt_sha256")
    receipt: EvidenceDirectoryReceipt | None = None
    journal: EvidenceJournal | None = None
    interrupted_publish_recovery: dict[str, Any] | None = None
    interrupted_publish_recovery_record_sha256: str | None = None
    recovery_reopen_tail_sha256: str | None = None
    try:
        (
            receipt,
            journal,
            interrupted_publish_recovery,
            interrupted_publish_recovery_record_sha256,
            recovery_reopen_tail_sha256,
        ) = _reopen_recovery_journal_barrier(
            evidence_path,
            receipt_sha256=receipt_sha256,
            plan_sha256=plan_sha256,
        )
        plan = read_recovery_plan(receipt, expected_plan_sha256=plan_sha256)
        _require(plan.head_sha == expected_head_sha, "recovery_plan_head_drift")

        def load_publication_source() -> Any:
            source = EvidenceJournal.load_publication_recovery_source(
                receipt,
                controller_resources=resources.controller,
                publication_control_plane_plan_sha256=plan_sha256,
                publication_journal_sha256=publication_journal_sha256,
                source_run_id=source_run_id,
            )
            _require(
                source.head_sha == expected_head_sha
                and source.run_id == source_run_id
                and source.control_plane_plan_sha256 == plan_sha256
                and source.publication_journal_sha256 == publication_journal_sha256
                and source.evidence_directory_receipt_sha256 == receipt_sha256,
                "publication_recovery_source_cli_binding_drift",
            )
            return source

        publication_source = load_publication_source()
        _, inventory, inspection_sha256 = _load_and_revalidate_inventory(
            args,
            phase=phase,
            supplied_ref=supplied_ref,
            expected_head_sha=expected_head_sha,
        )
        preflight = _validate_recovery_preflight(
            receipt,
            plan_sha256=plan_sha256,
            inventory=inventory,
            inspection_receipt_sha256=inspection_sha256,
        )
        _revalidate_locked_posture(
            args,
            inventory=inventory,
            expected_head_sha=expected_head_sha,
            supplied_ref=supplied_ref,
        )
        tail = publication_source.recovery_tail
        if tail.state == "source-unrecorded":
            # The marker plus one complete fresh dispatch consumes four
            # journal entries.  Reserve the full crash-safe suffix before the
            # first local append, not incrementally after a GitHub mutation.
            journal.require_capacity(4)
            journal.record(
                "operator-publication-recovery-source",
                publication_source.to_mapping(),
            )
            # The marker is part of the core journal grammar.  Reload it from
            # the held receipt before constructing any controller or making a
            # recovery decision; the unsealed pre-marker object cannot
            # authorize a dispatch.
            publication_source = load_publication_source()
            tail = publication_source.recovery_tail
            _require(
                tail.state == "complete",
                "publication_recovery_source_marker_reload_rejected",
            )

        def record_progress(label: str, payload: Mapping[str, Any]) -> None:
            journal.record(label, payload)

        receipt_value: Any | None = None
        settlement: Mapping[str, Any]
        if tail.state == "pending-operator-settlement":
            # A controller receipt is already durable.  Repair only the
            # deterministic local summary and make zero GitHub/controller
            # calls, including live history observations.
            pending_settlement = tail.pending_operator_settlement
            _require(
                type(pending_settlement) is dict,
                "pending_operator_settlement_missing",
            )
            journal.require_capacity(1)
            settlement = pending_settlement
        else:
            _require(
                tail.state in {"complete", "pending-intent"},
                "publication_recovery_tail_state_rejected",
            )
            journal.require_capacity(2 if tail.state == "pending-intent" else 3)
            controller = ReleaseGpuController(
                _locked_github(inventory),
                _AbortOnlyRemote(),
                resources=resources.controller,
            )
            if tail.state == "pending-intent":
                pending = tail.pending_intent
                journal_pending = journal.pending_recovery_dispatch_intent()
                _require(
                    type(pending) is dict
                    and type(journal_pending) is dict
                    and canonical_json(journal_pending) == canonical_json(pending)
                    and pending.get("head_sha") == expected_head_sha
                    and pending.get("source_run_id") == source_run_id,
                    "pending_recovery_dispatch_cli_binding_drift",
                )
                receipt_value = controller.reconcile_release_recovery_dispatch(
                    pending,
                    recovery_source=publication_source,
                    poll_limit=poll_limit,
                    progress=record_progress,
                )
            else:
                _require(
                    tail.pending_intent is None
                    and journal.pending_recovery_dispatch_intent() is None,
                    "complete_recovery_tail_pending_intent_drift",
                )
                recovery_request_nonce = secrets.token_hex(8)
                receipt_value = controller.dispatch_release_recovery(
                    head_sha=expected_head_sha,
                    source_run_id=source_run_id,
                    recovery_request_nonce=recovery_request_nonce,
                    recovery_source=publication_source,
                    poll_limit=poll_limit,
                    progress=record_progress,
                )

            # The progress sink must have durably archived the controller
            # settlement.  Reload that exact suffix, then compare the core
            # builder output before appending the local-only settlement.
            publication_source = load_publication_source()
            tail = publication_source.recovery_tail
            _require(
                tail.state == "pending-operator-settlement"
                and type(tail.pending_operator_settlement) is dict,
                "recovery_controller_settlement_reload_rejected",
            )
            built_settlement = EvidenceJournal.build_publication_recovery_operator_settlement(
                publication_source,
                receipt_value,
            )
            _require(
                canonical_json(built_settlement)
                == canonical_json(tail.pending_operator_settlement),
                "recovery_operator_settlement_builder_drift",
            )
            settlement = built_settlement

        final_journal_sha256 = journal.record(
            "operator-release-recovery-dispatch-settled",
            settlement,
        )
        publication_source = load_publication_source()
        completed_tail = publication_source.recovery_tail
        _require(
            completed_tail.state == "complete"
            and type(completed_tail.last_operator_settlement) is dict
            and canonical_json(completed_tail.last_operator_settlement)
            == canonical_json(settlement),
            "recovery_operator_settlement_reload_rejected",
        )
        _require(
            settlement.get("plan_sha256") == plan_sha256
            and settlement.get("head_sha") == expected_head_sha
            and settlement.get("source_run_id") == source_run_id
            and settlement.get("publication_journal_sha256") == publication_journal_sha256
            and settlement.get("publication_recovery_source_evidence_sha256")
            == publication_source.evidence_sha256
            and type(settlement.get("recovery_run_id")) is int
            and type(settlement.get("recovery_dispatch_evidence_sha256")) is str
            and type(settlement.get("mode")) is str,
            "recovery_operator_settlement_cli_binding_drift",
        )
        return {
            "schema_version": 1,
            "kind": "explainiverse-release-recovery-dispatch-settled",
            "mode": settlement["mode"],
            "head_sha": expected_head_sha,
            "source_run_id": source_run_id,
            "recovery_run_id": settlement["recovery_run_id"],
            "recovery_run_attempt": 1,
            "recovery_dispatch_evidence_sha256": settlement["recovery_dispatch_evidence_sha256"],
            "publication_recovery_source_evidence_sha256": (publication_source.evidence_sha256),
            "publication_journal_sha256": publication_journal_sha256,
            "final_journal_sha256": final_journal_sha256,
            "evidence_directory_receipt_sha256": receipt_sha256,
            "operator_preflight": preflight,
            "interrupted_publish_recovery": interrupted_publish_recovery,
            "interrupted_publish_recovery_record_sha256": (
                interrupted_publish_recovery_record_sha256
            ),
            "recovery_reopen_tail_sha256": recovery_reopen_tail_sha256,
            "secure_launch": dict(launch_receipt),
            "environment": dict(environment_receipt),
            "raw_dispatch_bypass_used": False,
            "workflow_completion_verified": False,
            "no_republish_verified": False,
            "next_required_evidence": "read-only terminal run/job/artifact and PyPI exactly-once verification",
        }
    finally:
        if journal is not None:
            journal.close()
        elif receipt is not None:
            receipt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed Explainiverse Lambda CUDA release operator"
    )
    parser.add_argument(
        "--action",
        choices=(
            "inspect",
            "create-app-inbox",
            "create-app-staging",
            "publish-app-capture",
            "execute",
            "resume-abort",
            "dispatch-release-recovery",
            "transport-self-test",
        ),
        default="inspect",
        help="inspect is the default and cannot mutate provider or GitHub state",
    )
    parser.add_argument("--phase", choices=tuple(PHASE_REFS))
    parser.add_argument("--repository-root")
    parser.add_argument("--operator-python-root")
    parser.add_argument("--operator-site-root")
    parser.add_argument("--operator-python-install-receipt")
    parser.add_argument("--operator-python-install-receipt-sha256")
    parser.add_argument("--operator-site-install-receipt")
    parser.add_argument("--operator-site-install-receipt-sha256")
    parser.add_argument("--expected-head-sha")
    parser.add_argument("--supplied-ref", choices=tuple(PHASE_REFS.values()))
    parser.add_argument("--git-executable")
    parser.add_argument("--gh-executable")
    parser.add_argument("--ssh-executable")
    parser.add_argument("--inspection-receipt")
    parser.add_argument("--inspection-receipt-sha256")
    parser.add_argument("--inspection-evidence-directory")
    parser.add_argument("--inspection-evidence-directory-receipt-sha256")
    parser.add_argument("--runtime-root")
    parser.add_argument("--lambda-api-key-fd", type=int)
    parser.add_argument("--plan-confirmation-fd", type=int)
    parser.add_argument("--confirm-plan-sha256")
    parser.add_argument("--evidence-directory")
    parser.add_argument("--evidence-directory-receipt-sha256")
    parser.add_argument("--app-capture-inbox")
    parser.add_argument("--app-capture-inbox-receipt-sha256")
    parser.add_argument("--app-capture-staging")
    parser.add_argument("--app-capture-staging-receipt-sha256")
    parser.add_argument("--app-capture-poll-limit", type=int, default=3600)
    parser.add_argument("--app-capture-poll-seconds", type=float, default=1.0)
    parser.add_argument("--capture-ordinal", type=int)
    parser.add_argument("--capture-generation", type=int)
    parser.add_argument("--capture-publication-nonce")
    parser.add_argument("--ssh-access-key")
    parser.add_argument("--ssh-key-name")
    parser.add_argument("--image-id")
    parser.add_argument("--controller-public-ipv4-cidr")
    parser.add_argument("--lifecycle-nonce")
    parser.add_argument("--plan-lifetime-seconds", type=int)
    parser.add_argument("--prior-accepted-cuda-runner-nonce", action="append")
    parser.add_argument("--preflight-run-id", type=int)
    parser.add_argument("--cuda-run-id", type=int)
    parser.add_argument("--final-main-evidence-directory")
    parser.add_argument("--final-main-evidence-receipt-sha256")
    parser.add_argument("--final-main-plan-sha256")
    parser.add_argument("--final-main-journal-sha256")
    parser.add_argument("--dispatch-poll-limit", type=int)
    parser.add_argument("--observation-poll-limit", type=int, default=24)
    parser.add_argument("--transport-self-test-nonce")
    parser.add_argument("--source-run-id", type=int)
    parser.add_argument("--publication-journal-sha256")
    parser.add_argument("--recovery-poll-limit", type=int, default=60)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environment_receipt: Mapping[str, Any] | None = None,
    launch_receipt: Mapping[str, Any] | None = None,
    captured_resources: Mapping[str, Any] | None = None,
) -> int:
    if environment_receipt is None or launch_receipt is None or captured_resources is None:
        _emit(
            {
                "schema_version": 1,
                "kind": "explainiverse-lambda-operator-error",
                "exception_type": "SecureLaunchError",
                "stable_code": "secure_entrypoint_required",
                "secret_values_logged": False,
            },
            stream=sys.stderr.buffer,
        )
        return 2
    args = build_parser().parse_args(argv)
    try:
        resources = _sealed_resources(captured_resources, launch_receipt)
        if args.action == "inspect":
            result = inspect(
                args,
                environment_receipt=environment_receipt,
                launch_receipt=launch_receipt,
            )
        elif args.action == "create-app-inbox":
            result = create_app_inbox(
                args,
                environment_receipt=environment_receipt,
                launch_receipt=launch_receipt,
            )
        elif args.action == "create-app-staging":
            result = create_app_staging(
                args,
                environment_receipt=environment_receipt,
                launch_receipt=launch_receipt,
            )
        elif args.action == "publish-app-capture":
            result = publish_app_capture(
                args,
                environment_receipt=environment_receipt,
                launch_receipt=launch_receipt,
                resources=resources,
            )
        elif args.action == "execute":
            result = execute(
                args,
                environment_receipt=environment_receipt,
                launch_receipt=launch_receipt,
                resources=resources,
            )
        elif args.action == "resume-abort":
            result = resume_abort(
                args,
                environment_receipt=environment_receipt,
                launch_receipt=launch_receipt,
                resources=resources,
            )
        elif args.action == "transport-self-test":
            result = transport_self_test(
                args,
                environment_receipt=environment_receipt,
                launch_receipt=launch_receipt,
            )
        elif args.action == "dispatch-release-recovery":
            result = dispatch_release_recovery(
                args,
                environment_receipt=environment_receipt,
                launch_receipt=launch_receipt,
                resources=resources,
            )
        else:  # pragma: no cover - argparse owns this branch.
            _fail("operator_action_rejected")
    except (OperatorError, ContractError, ControllerError) as exc:
        _emit(
            {
                "schema_version": 1,
                "kind": "explainiverse-lambda-operator-error",
                "exception_type": type(exc).__name__,
                "stable_code": str(exc),
                "secret_values_logged": False,
            },
            stream=sys.stderr.buffer,
        )
        return 2
    except Exception as exc:
        _emit(
            {
                "schema_version": 1,
                "kind": "explainiverse-lambda-operator-error",
                "exception_type": type(exc).__name__,
                "stable_code": "unclassified-local-operator-failure",
                "secret_values_logged": False,
            },
            stream=sys.stderr.buffer,
        )
        return 3
    _emit(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
