from __future__ import annotations

import builtins
import hashlib
import json
import socket
import subprocess
import urllib.request
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, cast

import pytest
from lambda_operator_receipt_fixtures import (  # type: ignore[import-not-found]
    HEAD,
    INSPECTION_SHA256,
    LIFECYCLE_NONCE,
    operator_preflight_fixture,
)

from scripts.release_gpu_jit_lambda_live import adapter as live
from scripts.release_gpu_jit_lambda_operator import cli as operator
from scripts.release_gpu_jit_lambda_operator import receipt_contract as contract


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("ascii")


def _live_canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _rebind_dynamic_receipts(value: dict[str, Any], expected: dict[str, Any]) -> None:
    discovery = value["discovery"]
    binding_material = {
        "snapshot_sha256": discovery["snapshot_sha256"],
        "region_description": discovery["target"]["region_description"],
        "instance_type_description": discovery["target"]["instance_type_description"],
        "gpu_description": discovery["target"]["gpu_description"],
        "price_cents_per_hour": discovery["target"]["price_cents_per_hour"],
        "vcpus": discovery["target"]["vcpus"],
        "memory_gib": discovery["target"]["memory_gib"],
        "storage_gib": discovery["target"]["storage_gib"],
        "images": discovery["image_candidates"],
        "ssh_key_name": discovery["ssh_access"]["key_name"],
        "ssh_public_key_sha256": discovery["ssh_access"]["public_key_sha256"],
        "baseline_file_systems_sha256": discovery["baseline_file_systems_sha256"],
        "original_global_rules": discovery["original_global_rules"],
    }
    discovery["binding_sha256"] = _sha(_live_canonical(binding_material))
    plan_sha256 = _sha(_live_canonical(expected["expected_immutable_plan"]))
    expected["expected_plan_sha256"] = plan_sha256
    value["plan_sha256"] = plan_sha256
    value["plan_confirmation"]["confirmed_plan_sha256"] = plan_sha256


@pytest.mark.parametrize("phase", ("pull-request", "final-main", "publication"))
def test_operator_preflight_producer_canonical_round_trip(phase: str) -> None:
    expected_value, expected = operator_preflight_fixture(phase)
    canonical_value = json.loads(_canonical(expected_value))
    identity = contract.validate_operator_preflight(canonical_value, **expected)
    assert identity["phase"] == phase
    assert identity["preflight_sha256"] == _sha(_canonical(expected_value))
    acceptance = expected_value["final_main_acceptance"]
    produced = operator._preflight_mapping(
        plan=SimpleNamespace(
            sha256=expected["expected_plan_sha256"],
            head_sha=HEAD,
            lifecycle_nonce=LIFECYCLE_NONCE,
            to_mapping=lambda: deepcopy(expected["expected_immutable_plan"]),
        ),
        discovery=SimpleNamespace(to_public_mapping=lambda: deepcopy(expected_value["discovery"])),
        inspection_receipt_sha256=INSPECTION_SHA256,
        inventory=deepcopy(expected_value["inventory"]),
        environment_receipt=deepcopy(expected_value["environment"]),
        launch_receipt=deepcopy(expected_value["secure_launch"]),
        lambda_fd_receipt=deepcopy(expected_value["lambda_secret_transport"]),
        confirmation_receipt=deepcopy(expected_value["plan_confirmation"]),
        app_inbox=cast(
            Any,
            SimpleNamespace(
                phase=phase,
                to_public_mapping=lambda: deepcopy(expected_value["app_capture_inbox"]),
            ),
        ),
        final_acceptance=(SimpleNamespace(**acceptance) if acceptance is not None else None),
        resources=cast(
            Any,
            SimpleNamespace(
                binding=expected_value["secure_launch"]["preloader"]["sealed_resources"]
            ),
        ),
    )
    assert produced == expected_value


def test_shared_fixture_accepts_exact_core_resource_identities() -> None:
    policy = "4" * 64
    controller = "5" * 64
    runtime = "6" * 64
    value, expected = operator_preflight_fixture(
        "final-main",
        policy_sha256=policy,
        controller_source_sha256=controller,
        runtime_bundle_sha256=runtime,
    )
    assert expected["expected_policy_sha256"] == policy
    assert expected["expected_controller_source_sha256"] == controller
    assert expected["expected_runtime_bundle_sha256"] == runtime
    assert contract.validate_operator_preflight(value, **expected)["phase"] == "final-main"


def test_live_plan_digest_matches_real_immutable_plan_without_file_newline() -> None:
    assert contract.TARGET_REGION == live.TARGET_REGION == "us-midwest-1"
    assert contract.TARGET_REGION_DESCRIPTION == live.TARGET_REGION_DESCRIPTION == "Illinois, USA"
    _, expected = operator_preflight_fixture("final-main")
    plan_mapping = expected["expected_immutable_plan"]
    target = plan_mapping["target"]
    image = target["image"]
    ssh = plan_mapping["ssh_access"]
    plan = live.build_immutable_plan(
        head_sha=plan_mapping["head_sha"],
        lifecycle_nonce=plan_mapping["lifecycle_nonce"],
        created_at_unix=plan_mapping["created_at_unix"],
        expires_at_unix=plan_mapping["expires_at_unix"],
        current_public_ipv4_cidr=plan_mapping["controller_source"],
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
        baseline_file_systems_sha256=plan_mapping["baseline_file_systems_sha256"],
        original_global_rules=plan_mapping["original_global_rules"],
        host_key_fingerprint=ssh["ephemeral_host_key_fingerprint"],
        runtime_bundle_sha256=plan_mapping["remote_runtime"]["bundle_sha256"],
    )
    assert plan.to_mapping() == plan_mapping
    assert plan.sha256 == expected["expected_plan_sha256"]
    assert plan.sha256 == _sha(_live_canonical(plan_mapping))
    assert plan.sha256 != _sha(_canonical(plan_mapping))


def test_live_discovery_public_receipt_hash_domains_have_no_file_newline() -> None:
    payloads = MappingProxyType(
        {
            operation: {"data": [], "fixture_operation": operation}
            for operation, _ in live.READ_OPERATIONS
        }
    )
    bindings = tuple(
        live.ResponseBinding(
            operation=operation,
            method="GET",
            path=path,
            request_sha256=_sha(f"request:{operation}".encode("ascii")),
            request_body_sha256=None,
            response_body_sha256=_sha(f"response:{operation}".encode("ascii")),
            status_code=200,
            content_type="application/json",
        )
        for operation, path in live.READ_OPERATIONS
    )
    snapshot = live._Snapshot(
        payloads=payloads,
        bindings=bindings,
        observed_started_monotonic_ns=1,
        observed_finished_monotonic_ns=2,
    )
    images = (
        {
            "id": "image-fixture",
            "created_time": "2026-01-01T00:00:00Z",
            "updated_time": "2026-01-02T01:00:00+01:00",
            "name": "lambda-stack-fixture",
            "description": "fixture",
            "family": "lambda-stack-22-04",
            "version": "1",
            "architecture": "x86_64",
            "region": {
                "name": live.TARGET_REGION,
                "description": live.TARGET_REGION_DESCRIPTION,
            },
        },
    )
    rules = (live.FirewallRule("icmp", None, "0.0.0.0/0", "fixture"),)
    binding_material = {
        "snapshot_sha256": snapshot.sha256,
        "region_description": live.TARGET_REGION_DESCRIPTION,
        "instance_type_description": "NVIDIA A100 80 GB SXM4",
        "gpu_description": "NVIDIA A100 80 GB SXM4",
        "price_cents_per_hour": 200,
        "vcpus": 30,
        "memory_gib": 200,
        "storage_gib": 1400,
        "images": list(images),
        "ssh_key_name": "fixture-key",
        "ssh_public_key_sha256": "7" * 64,
        "baseline_file_systems_sha256": "8" * 64,
        "original_global_rules": [rule.to_mapping() for rule in rules],
    }
    receipt = live.DiscoveryReceipt(
        snapshot_sha256=snapshot.sha256,
        observed_monotonic_ns=2,
        region_description=live.TARGET_REGION_DESCRIPTION,
        instance_type_description="NVIDIA A100 80 GB SXM4",
        gpu_description="NVIDIA A100 80 GB SXM4",
        price_cents_per_hour=200,
        vcpus=30,
        memory_gib=200,
        storage_gib=1400,
        images=images,
        ssh_key_name="fixture-key",
        ssh_public_key_sha256="7" * 64,
        baseline_file_systems_sha256="8" * 64,
        original_global_rules=rules,
        binding_sha256=_sha(_live_canonical(binding_material)),
        _snapshot=snapshot,
    )
    receipt.validate_binding()
    public = receipt.to_public_mapping()
    snapshot_material = {
        "payload_digests": public["payload_digests"],
        "bindings": public["response_bindings"],
    }
    assert public["snapshot_sha256"] == _sha(_live_canonical(snapshot_material))
    assert public["snapshot_sha256"] != _sha(_canonical(snapshot_material))
    assert public["binding_sha256"] == _sha(_live_canonical(binding_material))
    assert public["binding_sha256"] != _sha(_canonical(binding_material))


def test_live_rfc3339_image_timestamps_round_trip_without_text_normalization() -> None:
    value, expected = operator_preflight_fixture("final-main")
    plan = expected["expected_immutable_plan"]
    plan["target"]["image"]["created_time"] = "2026-01-01T00:00:00Z"
    plan["target"]["image"]["updated_time"] = "2026-01-02T01:00:00+01:00"
    image = value["discovery"]["image_candidates"][0]
    image["created_time"] = plan["target"]["image"]["created_time"]
    image["updated_time"] = plan["target"]["image"]["updated_time"]
    _rebind_dynamic_receipts(value, expected)
    assert contract.validate_operator_preflight(value, **expected)["phase"] == "final-main"


@pytest.mark.parametrize("phase", ("pull-request", "final-main", "publication"))
def test_legacy_east_region_is_rejected_even_with_coherently_rebound_hashes(
    phase: str,
) -> None:
    value, expected = operator_preflight_fixture(phase)
    plan_target = expected["expected_immutable_plan"]["target"]
    plan_target["region_name"] = "us-east-1"
    plan_target["region_description"] = "US East"
    plan_target["image"]["region_name"] = "us-east-1"
    discovery_target = value["discovery"]["target"]
    discovery_target["capacity_region"] = "us-east-1"
    discovery_target["region_description"] = "US East"
    value["discovery"]["image_candidates"][0]["region"] = {
        "name": "us-east-1",
        "description": "US East",
    }
    _rebind_dynamic_receipts(value, expected)
    with pytest.raises(contract.OperatorReceiptContractError, match="target_rejected"):
        contract.validate_operator_preflight(value, **expected)


TAMPERS: tuple[tuple[str, Callable[[dict[str, Any]], None]], ...] = (
    ("top-extra", lambda value: value.update(unreviewed=True)),
    ("top-missing", lambda value: value.pop("direct_publication_dispatch_exposed")),
    (
        "bool-int",
        lambda value: value.update(live_gates_not_constructed_before_confirmation=1),
    ),
    ("inventory-digest", lambda value: value.update(inventory_sha256="0" * 64)),
    (
        "git-config",
        lambda value: value["repository"]["git_configuration"].update(
            repository_fsmonitor_overridden_false=False
        ),
    ),
    ("tree", lambda value: value["repository"].update(tree_object_sha="0" * 40)),
    (
        "critical-source",
        lambda value: value["repository"]["critical_sources"]["pyproject.toml"].update(
            sha256="0" * 64
        ),
    ),
    (
        "manifest-extra",
        lambda value: value["secure_launch"]["preloader"]["source"]["source_manifest"].update(
            extra=True
        ),
    ),
    (
        "source-digest",
        lambda value: value["secure_launch"]["preloader"]["source"].update(
            evidence_sha256="0" * 64
        ),
    ),
    (
        "shim",
        lambda value: value["secure_launch"]["preloader"]["shim"].update(
            stable_descriptor_read=False
        ),
    ),
    (
        "bootstrap",
        lambda value: value["secure_launch"]["preloader"]["bootstrap"].update(
            pth_executed_by_cpython=True
        ),
    ),
    (
        "install",
        lambda value: value["secure_launch"]["preloader"]["site_install_receipt"].update(
            pip_present_in_runtime=True
        ),
    ),
    (
        "directory-acl",
        lambda value: value["secure_launch"]["preloader"]["runtime_site_directory_receipt"][
            "acl"
        ].update(inheritance_protected=False),
    ),
    (
        "directory-acl-owner",
        lambda value: value["secure_launch"]["preloader"]["runtime_site_directory_receipt"][
            "acl"
        ].update(owner_sid="S-1-5-21-9999"),
    ),
    (
        "duplicate-directory-receipt",
        lambda value: value["secure_launch"]["preloader"]["runtime_site_directory_receipt"].update(
            receipt_sha256=value["secure_launch"]["preloader"]["python_runtime_directory_receipt"][
                "receipt_sha256"
            ]
        ),
    ),
    (
        "early-hold",
        lambda value: value["secure_launch"]["preloader"]["early_runtime_boundary"][
            "held_trees"
        ].update(delete_share_allowed=True),
    ),
    (
        "early-hold-missing",
        lambda value: value["secure_launch"]["preloader"]["early_runtime_boundary"][
            "held_trees"
        ].update(held_handle_count=908),
    ),
    (
        "early-hold-extra",
        lambda value: value["secure_launch"]["preloader"]["early_runtime_boundary"][
            "held_trees"
        ].update(held_handle_count=910),
    ),
    (
        "resource",
        lambda value: value["secure_launch"]["preloader"]["sealed_resources"].update(
            live_repository_reopen_permitted=True
        ),
    ),
    (
        "parent-claim",
        lambda value: value["secure_launch"]["windows_launcher_parent_declaration"].update(
            parent_provenance_authenticated=True
        ),
    ),
    ("executable", lambda value: value["executables"]["gh"].update(sha256="0" * 64)),
    (
        "dependency",
        lambda value: value["inventory"]["dependencies"]["startup_pth"].update(
            unexpected_files_present=True
        ),
    ),
    (
        "interpreter",
        lambda value: value["inventory"]["interpreter_runtime"].update(
            repository_present_in_sys_path=True
        ),
    ),
    ("discovery-gpu", lambda value: value["discovery"]["target"].update(gpus=2)),
    (
        "discovery-price-bool",
        lambda value: value["discovery"]["target"].update(price_cents_per_hour=True),
    ),
    (
        "discovery-image",
        lambda value: value["discovery"]["image_candidates"][0].update(id="image-unreviewed"),
    ),
    (
        "discovery-image-z-time",
        lambda value: value["discovery"]["image_candidates"][0].update(
            created_time="2026-08-28T12:00:00Z"
        ),
    ),
    (
        "discovery-image-offset-time",
        lambda value: value["discovery"]["image_candidates"][0].update(
            created_time="2026-08-28T13:00:00+01:00"
        ),
    ),
    (
        "discovery-ssh-key",
        lambda value: value["discovery"]["ssh_access"].update(key_name="wrong-key"),
    ),
    (
        "discovery-ssh-digest",
        lambda value: value["discovery"]["ssh_access"].update(public_key_sha256="0" * 64),
    ),
    (
        "discovery-baseline",
        lambda value: value["discovery"].update(baseline_file_systems_sha256="0" * 64),
    ),
    (
        "discovery-firewall",
        lambda value: value["discovery"]["original_global_rules"][0].update(
            source_network="10.0.0.0/8"
        ),
    ),
    (
        "discovery-payload-digest",
        lambda value: value["discovery"]["payload_digests"].update(images="0" * 64),
    ),
    (
        "discovery-snapshot",
        lambda value: value["discovery"].update(snapshot_sha256="0" * 64),
    ),
    (
        "discovery-binding",
        lambda value: value["discovery"].update(binding_sha256="0" * 64),
    ),
    (
        "plan-image",
        lambda value: value["secure_launch"]["preloader"]["sealed_resources"].update(
            runtime_bundle_sha256="0" * 64
        ),
    ),
    (
        "lambda-secret",
        lambda value: value["lambda_secret_transport"].update(value_archived=True),
    ),
    (
        "confirmation",
        lambda value: value["plan_confirmation"].update(confirmation_exact_line=False),
    ),
    (
        "app-inbox",
        lambda value: value["app_capture_inbox"].update(accepted_capture_count=1),
    ),
)


@pytest.mark.parametrize(("label", "mutate"), TAMPERS, ids=[item[0] for item in TAMPERS])
def test_operator_preflight_contract_rejects_nested_leaf_tamper(
    label: str, mutate: Callable[[dict[str, Any]], None]
) -> None:
    del label
    value, expected = operator_preflight_fixture("final-main")
    mutate(value)
    with pytest.raises(contract.OperatorReceiptContractError):
        contract.validate_operator_preflight(value, **expected)


def test_operator_preflight_contract_rejects_expected_immutable_plan_drift() -> None:
    _, expected = operator_preflight_fixture("final-main")
    drift_cases = (
        ("target", "price_cents_per_hour", 201),
        ("target", "gpu_description", "unreviewed GPU"),
        ("ssh_access", "key_name", "wrong-key"),
        ("remote_runtime", "bundle_sha256", "0" * 64),
    )
    for section, key, replacement in drift_cases:
        value, case_expected = operator_preflight_fixture("final-main")
        case_expected["expected_immutable_plan"][section][key] = replacement
        with pytest.raises(contract.OperatorReceiptContractError):
            contract.validate_operator_preflight(value, **case_expected)
    assert expected["expected_immutable_plan"]["baseline_file_systems_sha256"] != "0" * 64


def test_operator_preflight_contract_is_pure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, expected = operator_preflight_fixture("final-main")

    def rejected(*_: Any, **__: Any) -> Any:
        raise AssertionError("pure receipt validation attempted external access")

    monkeypatch.setattr(builtins, "open", rejected)
    monkeypatch.setattr(Path, "open", rejected)
    monkeypatch.setattr(Path, "read_bytes", rejected)
    monkeypatch.setattr(subprocess, "run", rejected)
    monkeypatch.setattr(socket, "socket", rejected)
    monkeypatch.setattr(urllib.request, "urlopen", rejected)
    assert contract.validate_operator_preflight(value, **expected)["head_sha"] == HEAD
