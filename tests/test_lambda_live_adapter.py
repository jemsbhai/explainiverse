from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import time
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.release_gpu_jit_lambda_live import adapter as live


def _test_ed25519_public_key(fill: int) -> str:
    blob = (
        len(b"ssh-ed25519").to_bytes(4, "big")
        + b"ssh-ed25519"
        + (32).to_bytes(4, "big")
        + bytes([fill]) * 32
    )
    return "ssh-ed25519 " + base64.b64encode(blob).decode("ascii")


HEAD_SHA = "c5068f3acfe7531d189e8717d1f8c86cb0fe9ef4"
NONCE = "0123456789abcdef0123456789abcdef"
SSH_PUBLIC = _test_ed25519_public_key(1)
RULESET_ID = "ruleset-0123456789abcdef"
INSTANCE_ID = "instance-0123456789abcdef"
INSTANCE_IP = "8.8.4.4"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def _pipe(value: bytes) -> int:
    read_fd, write_fd = os.pipe()
    os.write(write_fd, value)
    os.close(write_fd)
    return read_fd


def _region() -> dict[str, str]:
    return {"name": live.TARGET_REGION, "description": "Illinois, USA"}


def _instance_type() -> dict[str, Any]:
    return {
        "name": live.TARGET_INSTANCE_TYPE,
        "description": "8x A100 (80 GB SXM4)",
        "gpu_description": "A100 (80 GB SXM4)",
        "price_cents_per_hour": 2232,
        "specs": {
            "vcpus": 240,
            "memory_gib": 1800,
            "storage_gib": 20480,
            "gpus": 8,
        },
        "architecture": "x86_64",
    }


def _image() -> dict[str, Any]:
    return {
        "id": "image-ubuntu-2204-a100",
        "created_time": "2026-07-01T00:00:00Z",
        "updated_time": "2026-08-01T00:00:00Z",
        "name": "Lambda Stack 22.04",
        "description": "Ubuntu 22.04 Lambda Stack",
        "family": "lambda-stack-22-04",
        "version": "2026.08",
        "architecture": "x86_64",
        "region": _region(),
    }


def _original_rules() -> list[dict[str, Any]]:
    return [
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
    ]


def _file_systems() -> list[dict[str, Any]]:
    return [
        {
            "id": "fs-0123456789abcdef",
            "name": "preexisting-fixture-storage",
            "mount_point": "/lambda/nfs/preexisting-fixture-storage",
            "created": "2025-01-02T03:04:05Z",
            "created_by": {
                "id": "user-0123456789abcdef",
                "email": "jemsbhai@example.com",
                "status": "active",
            },
            "is_in_use": False,
            "region": _region(),
            "bytes_used": 7,
        }
    ]


def _base_payloads() -> dict[str, dict[str, Any]]:
    return {
        "instances": {"data": []},
        "file_systems": {"data": _file_systems()},
        "ssh_keys": {
            "data": [
                {
                    "id": "key-existing",
                    "name": "preexisting-fixture-key",
                    "public_key": SSH_PUBLIC,
                }
            ]
        },
        "instance_types": {
            "data": {
                live.TARGET_INSTANCE_TYPE: {
                    "instance_type": _instance_type(),
                    "regions_with_capacity_available": [_region()],
                }
            }
        },
        "images": {"data": [_image()]},
        "regions": {"data": [_region()]},
        "global_firewall": {"data": {"id": "global", "name": "Global", "rules": _original_rules()}},
        "firewall_rulesets": {"data": []},
    }


class FakeTransport:
    def __init__(self, payloads: dict[str, dict[str, Any]]) -> None:
        self.payloads = payloads
        self.mutations: dict[str, dict[str, Any]] = {}
        self.requests: list[dict[str, Any]] = []
        self.force_response: live.ProviderResponse | None = None

    def __call__(
        self, request: live.ProviderRequest, api_key: live.SecretBuffer
    ) -> live.ProviderResponse:
        assert api_key.copy_bytes() == b"lambda-test-key"
        body_copy = bytes(request.body) if request.body is not None else None
        self.requests.append(
            {
                "operation": request.operation,
                "method": request.method,
                "path": request.path,
                "body": body_copy,
                "request_sha256": request.request_sha256,
            }
        )
        if self.force_response is not None:
            response = self.force_response
            self.force_response = None
            return response
        payload = (
            self.mutations[request.operation]
            if request.mutating
            else self.payloads[request.operation]
        )
        return live.ProviderResponse(
            request_sha256=request.request_sha256,
            status_code=200,
            content_type="application/json; charset=utf-8",
            body=_canonical(payload),
        )


class FakePacingClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _client(transport: FakeTransport) -> live.LambdaHttpClient:
    clock = FakePacingClock()
    fd = _pipe(b"lambda-test-key")
    try:
        return live.LambdaHttpClient.from_secret_fd(
            fd,
            transport=transport,
            monotonic_clock=clock.monotonic,
            sleep=clock.sleep,
        )
    finally:
        os.close(fd)


def _fake_identity() -> live.HostIdentity:
    public = _test_ed25519_public_key(2)
    key_blob = base64.b64decode(public.split()[1], validate=True)
    fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(key_blob).digest()).decode(
        "ascii"
    ).rstrip("=")
    return live.HostIdentity(
        live.SecretBuffer(
            b"-----BEGIN OPENSSH PRIVATE KEY-----\nfake\n-----END OPENSSH PRIVATE KEY-----\n",
            label="test_host_key",
        ),
        public,
        fingerprint,
    )


def _secure_access_identity(tmp_path: Path) -> tuple[Path, str, str]:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private_key = Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.OpenSSH,
        encryption_algorithm=serialization.NoEncryption(),
    )
    canonical_public = (
        private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.OpenSSH,
            format=serialization.PublicFormat.OpenSSH,
        )
        .decode("ascii")
    )
    path = (tmp_path / "lambda-access-key").resolve()
    path.write_bytes(private_bytes)
    if os.name == "nt":
        import ntsecuritycon
        import win32api
        import win32con
        import win32security

        token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
        current_user = win32security.GetTokenInformation(token, win32security.TokenUser)[0]
        dacl = win32security.ACL()
        for sid in (
            current_user,
            win32security.ConvertStringSidToSid("S-1-5-18"),
            win32security.ConvertStringSidToSid("S-1-5-32-544"),
        ):
            dacl.AddAccessAllowedAceEx(
                win32security.ACL_REVISION_DS,
                0,
                ntsecuritycon.FILE_ALL_ACCESS,
                sid,
            )
        descriptor = win32security.SECURITY_DESCRIPTOR()
        descriptor.SetSecurityDescriptorOwner(current_user, False)
        descriptor.SetSecurityDescriptorDacl(True, dacl, False)
        descriptor.SetSecurityDescriptorControl(
            win32security.SE_DACL_PROTECTED,
            win32security.SE_DACL_PROTECTED,
        )
        win32security.SetFileSecurity(
            str(path),
            win32security.OWNER_SECURITY_INFORMATION | win32security.DACL_SECURITY_INFORMATION,
            descriptor,
        )
    else:
        path.chmod(0o600)
    public_digest = hashlib.sha256(canonical_public.encode("ascii")).hexdigest()
    key_blob = base64.b64decode(canonical_public.split()[1], validate=True)
    fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(key_blob).digest()).decode(
        "ascii"
    ).rstrip("=")
    return path, public_digest, fingerprint


def test_access_identity_is_sealed_without_leaking_private_metadata(
    tmp_path: Path,
) -> None:
    path, public_digest, fingerprint = _secure_access_identity(tmp_path)
    receipt = live.capture_access_identity(path, expected_public_key_sha256=public_digest)
    try:
        public = receipt.to_public_mapping()
        assert public["public_key_sha256"] == public_digest
        assert public["public_key_fingerprint"] == fingerprint
        assert public["private_digest_recorded"] is True
        assert public["absolute_path_redacted"] is True
        encoded_public = json.dumps(public, sort_keys=True)
        assert str(path) not in encoded_public
        assert receipt._private_file_sha256 not in encoded_public

        validation = receipt.validate(expected_public_key_sha256=public_digest)
        assert validation["public_key_sha256"] == public_digest
        assert validation["public_key_fingerprint"] == fingerprint
        if os.name == "nt":
            with pytest.raises(PermissionError):
                path.write_bytes(b"attacker replacement")
        else:
            held_path = path.with_name("lambda-access-key-held")
            path.replace(held_path)
            path.write_bytes(held_path.read_bytes())
            path.chmod(0o600)
            with pytest.raises(live.ContractError, match="ssh_access_identity_path_identity_drift"):
                receipt.validate(expected_public_key_sha256=public_digest)
    finally:
        receipt.close()

    assert receipt.closed is True
    with pytest.raises(live.ContractError, match="ssh_access_identity_closed"):
        receipt.validate(expected_public_key_sha256=public_digest)


def test_access_identity_rejects_wrong_provider_key_digest(tmp_path: Path) -> None:
    path, _, _ = _secure_access_identity(tmp_path)
    with pytest.raises(live.ContractError, match="ssh_access_identity_public_key_mismatch"):
        live.capture_access_identity(path, expected_public_key_sha256="f" * 64)


def test_access_identity_rejects_insecure_permissions(tmp_path: Path) -> None:
    path, public_digest, _ = _secure_access_identity(tmp_path)
    if os.name == "nt":
        import ntsecuritycon
        import win32security

        descriptor = win32security.GetFileSecurity(
            str(path), win32security.DACL_SECURITY_INFORMATION
        )
        dacl = descriptor.GetSecurityDescriptorDacl()
        assert dacl is not None
        dacl.AddAccessAllowedAceEx(
            win32security.ACL_REVISION_DS,
            0,
            ntsecuritycon.FILE_GENERIC_READ,
            win32security.ConvertStringSidToSid("S-1-1-0"),
        )
        descriptor.SetSecurityDescriptorDacl(True, dacl, False)
        win32security.SetFileSecurity(
            str(path), win32security.DACL_SECURITY_INFORMATION, descriptor
        )
        expected = "ssh_access_identity_trustee_rejected"
    else:
        path.chmod(0o644)
        expected = "ssh_access_identity_permissions_open"
    with pytest.raises(live.ContractError, match=expected):
        live.capture_access_identity(path, expected_public_key_sha256=public_digest)


def test_access_identity_rejects_hard_linked_private_key(tmp_path: Path) -> None:
    path, public_digest, _ = _secure_access_identity(tmp_path)
    alias = tmp_path / "lambda-access-key-alias"
    os.link(path, alias)
    with pytest.raises(live.ContractError, match="ssh_access_identity_link_count_rejected"):
        live.capture_access_identity(path, expected_public_key_sha256=public_digest)


def test_access_identity_receipt_cannot_redirect_held_handle_to_another_path(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_path, first_public_digest, _ = _secure_access_identity(first_root)
    second_path, _, _ = _secure_access_identity(second_root)
    receipt = live.capture_access_identity(
        first_path,
        expected_public_key_sha256=first_public_digest,
    )
    try:
        redirected = replace(receipt, _absolute_path=str(second_path))
        expected = (
            "ssh_access_identity_handle_path_mismatch"
            if os.name == "nt"
            else "ssh_access_identity_path_identity_drift"
        )
        with pytest.raises(live.ContractError, match=expected):
            redirected.validate(expected_public_key_sha256=first_public_digest)
    finally:
        receipt.close()


def test_evidence_directory_is_created_owner_private_and_held(tmp_path: Path) -> None:
    path = (tmp_path / "release-evidence").resolve()
    receipt = live.create_evidence_directory(path)
    try:
        public = receipt.to_public_mapping()
        assert public["receipt_sha256"] == receipt.receipt_sha256
        assert public["absolute_path_redacted"] is True
        assert public["directory_identity_recorded"] is True
        assert public["owner_private"] is True
        assert str(path) not in json.dumps(public, sort_keys=True)
        assert receipt.validate()["receipt_sha256"] == receipt.receipt_sha256

        forged = replace(receipt, receipt_sha256="f" * 64)
        with pytest.raises(live.ContractError, match="evidence_directory_receipt_binding_rejected"):
            forged.validate()

        reopened = live.reopen_evidence_directory(
            path,
            expected_receipt_sha256=receipt.receipt_sha256,
        )
        try:
            assert reopened.receipt_sha256 == receipt.receipt_sha256
        finally:
            reopened.close()

        if os.name == "nt":
            with pytest.raises(PermissionError):
                path.rmdir()
        else:
            assert stat.S_IMODE(path.stat().st_mode) == 0o700
    finally:
        receipt.close()

    assert receipt.closed is True
    with pytest.raises(live.ContractError, match="evidence_directory_receipt_closed"):
        receipt.validate()


def test_evidence_directory_rejects_acl_drift(tmp_path: Path) -> None:
    path = (tmp_path / "release-evidence").resolve()
    receipt = live.create_evidence_directory(path)
    try:
        if os.name == "nt":
            import ntsecuritycon
            import win32security

            descriptor = win32security.GetFileSecurity(
                str(path), win32security.DACL_SECURITY_INFORMATION
            )
            dacl = descriptor.GetSecurityDescriptorDacl()
            assert dacl is not None
            dacl.AddAccessAllowedAceEx(
                win32security.ACL_REVISION_DS,
                win32security.OBJECT_INHERIT_ACE | win32security.CONTAINER_INHERIT_ACE,
                ntsecuritycon.FILE_GENERIC_READ,
                win32security.ConvertStringSidToSid("S-1-1-0"),
            )
            descriptor.SetSecurityDescriptorDacl(True, dacl, False)
            win32security.SetFileSecurity(
                str(path), win32security.DACL_SECURITY_INFORMATION, descriptor
            )
            expected = "evidence_directory_trustee_rejected"
        else:
            path.chmod(0o755)
            expected = "evidence_directory_acl_drift"
        with pytest.raises(live.ContractError, match=expected):
            receipt.validate()
    finally:
        receipt.close()


def test_evidence_directory_reopen_rejects_wrong_binding(tmp_path: Path) -> None:
    path = (tmp_path / "release-evidence").resolve()
    receipt = live.create_evidence_directory(path)
    receipt.close()
    with pytest.raises(live.ContractError, match="evidence_directory_receipt_mismatch"):
        live.reopen_evidence_directory(path, expected_receipt_sha256="f" * 64)


def test_public_writers_reject_paths_outside_held_evidence_directory(
    tmp_path: Path,
) -> None:
    receipt = live.create_evidence_directory((tmp_path / "release-evidence").resolve())
    try:
        outside = (tmp_path / "outside-evidence").resolve()
        with pytest.raises(live.ContractError, match="public_evidence_outside_evidence_directory"):
            live.write_public_evidence(
                outside,
                {"public": True},
                evidence_directory_receipt=receipt,
            )
        with pytest.raises(live.ContractError, match="known_hosts_outside_evidence_directory"):
            live.write_public_known_hosts(
                outside,
                identity=_fake_identity(),
                public_ipv4=INSTANCE_IP,
                evidence_directory_receipt=receipt,
            )
        assert not outside.exists()
    finally:
        receipt.close()


def test_public_file_write_never_exposes_partial_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = (tmp_path / "public-evidence").resolve()
    real_write = live.os.write
    calls = 0

    def interrupted_write(fd: int, payload: bytes) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(fd, payload[:1])
        raise OSError("simulated interruption")

    monkeypatch.setattr(live.os, "write", interrupted_write)
    with pytest.raises(live.ContractError, match="public_file_atomic_publish_failed"):
        live._write_exclusive_bytes(destination, b"complete-public-evidence")
    assert not destination.exists()
    assert not list(tmp_path.glob(".public-evidence.pending-*"))


def test_public_file_write_never_overwrites_raced_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = (tmp_path / "public-evidence").resolve()
    if os.name == "nt":
        import win32file

        real_publish = win32file.MoveFileEx

        def raced_publish(source: str, target: str, flags: int) -> None:
            destination.write_bytes(b"raced-owner-bytes")
            real_publish(source, target, flags)

        monkeypatch.setattr(win32file, "MoveFileEx", raced_publish)
    else:
        real_publish = live.os.link

        def raced_publish(
            source: str | os.PathLike[str],
            target: str | os.PathLike[str],
            *,
            follow_symlinks: bool,
        ) -> None:
            destination.write_bytes(b"raced-owner-bytes")
            real_publish(source, target, follow_symlinks=follow_symlinks)

        monkeypatch.setattr(live.os, "link", raced_publish)

    with pytest.raises(live.ContractError, match="public_file_atomic_publish_failed"):
        live._write_exclusive_bytes(destination, b"controller-bytes")
    assert destination.read_bytes() == b"raced-owner-bytes"
    assert not list(tmp_path.glob(".public-evidence.pending-*"))


def _runtime_bundle() -> live.RuntimeBundle:
    return live.RuntimeBundle(
        tuple((name, f"# {name}\n".encode("ascii")) for name in live.RUNTIME_BUNDLE_NAMES)
    )


def _plan(
    identity: live.HostIdentity | None = None,
    runtime_bundle: live.RuntimeBundle | None = None,
) -> live.ImmutablePlan:
    identity = identity or _fake_identity()
    runtime_bundle = runtime_bundle or _runtime_bundle()
    now = int(time.time())
    return live.build_immutable_plan(
        head_sha=HEAD_SHA,
        lifecycle_nonce=NONCE,
        created_at_unix=now - 1,
        expires_at_unix=now + 3600,
        current_public_ipv4_cidr="8.8.8.8/32",
        region_description="Illinois, USA",
        image_id=_image()["id"],
        image_created_time=_image()["created_time"],
        image_description=_image()["description"],
        image_name=_image()["name"],
        image_family=_image()["family"],
        image_version=_image()["version"],
        image_updated_time=_image()["updated_time"],
        instance_type_description=_instance_type()["description"],
        gpu_description=_instance_type()["gpu_description"],
        price_cents_per_hour=_instance_type()["price_cents_per_hour"],
        vcpus=_instance_type()["specs"]["vcpus"],
        memory_gib=_instance_type()["specs"]["memory_gib"],
        storage_gib=_instance_type()["specs"]["storage_gib"],
        ssh_key_name="preexisting-fixture-key",
        ssh_public_key_sha256=hashlib.sha256(SSH_PUBLIC.encode()).hexdigest(),
        baseline_file_systems_sha256=hashlib.sha256(_canonical(_file_systems())).hexdigest(),
        original_global_rules=_original_rules(),
        host_key_fingerprint=identity.fingerprint,
        runtime_bundle_sha256=runtime_bundle.sha256,
    )


def _adapter(
    transport: FakeTransport,
    identity: live.HostIdentity | None = None,
    runtime_bundle: live.RuntimeBundle | None = None,
    mutation_intent_callback: Any | None = None,
) -> live.LambdaLiveAdapter:
    plan = _plan(identity, runtime_bundle)
    if mutation_intent_callback is None:

        def ignore_mutation_intent(_: Any) -> None:
            return None

        mutation_intent_callback = ignore_mutation_intent
    return live.LambdaLiveAdapter(
        _client(transport),
        plan,
        live.LiveGates(True, True, plan.sha256),
        mutation_intent_callback=mutation_intent_callback,
    )


def _ruleset(plan: live.ImmutablePlan, *, instance_ids: list[str] | None = None) -> dict[str, Any]:
    return {
        "id": RULESET_ID,
        "name": plan.ruleset_name,
        "region": _region(),
        "rules": [rule.to_mapping() for rule in plan.desired_firewall_rules],
        "created": "2026-08-28T21:00:00Z",
        "instance_ids": instance_ids or [],
    }


def _instance(plan: live.ImmutablePlan, *, status: str = "active") -> dict[str, Any]:
    return {
        "id": INSTANCE_ID,
        "name": plan.instance_name,
        "ip": INSTANCE_IP,
        "status": status,
        "ssh_key_names": [plan.ssh_key_name],
        "file_system_names": [],
        "region": _region(),
        "instance_type": _instance_type(),
        "image": {"id": plan.image_id, "family": plan.image_family},
        "actions": {},
        "tags": [{"key": key, "value": value} for key, value in plan.ownership_tags],
        "firewall_rulesets": [{"id": RULESET_ID}],
    }


def _set_global(transport: FakeTransport, rules: tuple[live.FirewallRule, ...]) -> None:
    transport.payloads["global_firewall"] = {
        "data": {"id": "global", "name": "Global", "rules": [r.to_mapping() for r in rules]}
    }


def _host_preflight_payload(plan: live.ImmutablePlan) -> dict[str, Any]:
    observed = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    uuids = [f"GPU-0000000{i}-0000-0000-0000-00000000000{i}" for i in range(1, 9)]
    return {
        "schema_version": 1,
        "kind": "explainiverse-lambda-jit-host-preflight",
        "observed_at": observed,
        "cloud_init_status": "done",
        "cloud_init_output_sha256": "1" * 64,
        "effective_uid": 0,
        "root_owned_nonwritable_runtime_bundle": True,
        "runtime_bundle_sha256": plan.runtime_bundle_sha256,
        "host_physical_gpu_count": 8,
        "host_physical_gpu_uuids": uuids,
        "host_physical_gpu_products": [live.EXPECTED_HOST_GPU_PRODUCT] * 8,
        "gpu_inventory_output_sha256": "2" * 64,
        "image": {
            "schema_version": 1,
            "kind": "explainiverse-runner-image-probe",
            "observed_at": observed,
            "image_reference": live.EXPECTED_RUNNER_IMAGE_REFERENCE,
            "manifest_digest": live.EXPECTED_RUNNER_IMAGE_MANIFEST,
            "config_digest": live.EXPECTED_RUNNER_IMAGE_CONFIG,
            "platform": live.EXPECTED_RUNNER_IMAGE_PLATFORM,
            "pull_output_sha256": "3" * 64,
            "inspect_response_sha256": "4" * 64,
            "probe": {
                "container_uid": 1001,
                "container_gid": 1001,
                "runner_listener_present": True,
                "runner_listener_version": live.EXPECTED_RUNNER_VERSION,
                "runner_commit": live.EXPECTED_RUNNER_COMMIT,
                "node20_present": True,
                "node20_version": live.EXPECTED_NODE20_VERSION,
                "node20_sha256": live.EXPECTED_NODE20_SHA256,
                "output_sha256": "5" * 64,
            },
            "network_contact_during_probe_container": False,
            "registry_contacted_for_digest_pull": True,
            "github_api_contacted": False,
        },
        "gpu_injection": {
            "gpu_injection_verified": True,
            "gpu_count": 8,
            "gpu_product": live.EXPECTED_HOST_GPU_PRODUCT,
            "output_sha256": hashlib.sha256(
                b"gpu_injection=verified\n" b"gpu_count=8\n" b"gpu_product=NVIDIA A100-SXM4-80GB\n"
            ).hexdigest(),
        },
        "local_runtime_residue_absent": True,
        "jit_config_received": False,
        "github_api_credential_received": False,
        "github_api_contacted": False,
        "accepted_actions_evidence": False,
    }


def test_dry_run_is_inert_and_pins_exact_official_paths() -> None:
    dry_run = live.dry_run_contract()
    assert dry_run["production_authorized"] is False
    assert dry_run["provider_mutation_authorized"] is False
    assert dry_run["live_go"] is False
    assert dry_run["provider_contacted"] is False
    assert dry_run["mutations_performed"] is False
    assert dry_run["api_key_lifecycle_supported"] is False
    assert dry_run["provider_mutation_write_ahead_intent_required"] is True
    assert dry_run["persisted_mutation_intent_exact_schema_revalidated"] is True
    assert dry_run["access_identity_private_metadata_persisted"] is False
    assert dry_run["access_identity_provider_public_key_binding_required"] is True
    assert dry_run["allowed_target"]["image_family"] == "lambda-stack-22-04"
    assert dry_run["fixed_cloud_init_wait_command"] == list(live.FIXED_CLOUD_INIT_WAIT_COMMAND)
    pacing = dry_run["provider_request_pacing"]
    assert pacing["minimum_seconds_between_starts"] == 1.0
    assert pacing["full_snapshot_minimum_pacing_seconds"] == 7.0
    assert (
        pacing["full_snapshot_minimum_pacing_seconds"]
        < pacing["observation_window_seconds"]
        < pacing["prestate_freshness_seconds"]
    )
    assert pacing["automatic_retry"] is False
    assert {entry["path"] for entry in dry_run["exact_read_operations"]} == {
        "/api/v1/instances",
        "/api/v1/file-systems",
        "/api/v1/ssh-keys",
        "/api/v1/instance-types",
        "/api/v1/images",
        "/api/v1/regions",
        "/api/v1/firewall-rulesets/global",
        "/api/v1/firewall-rulesets",
    }
    mutations = {entry["operation"]: entry for entry in dry_run["exact_mutation_operations"]}
    assert mutations["launch"] == {
        "operation": "launch",
        "method": "POST",
        "path": "/api/v1/instance-operations/launch",
    }
    assert mutations["terminate"]["path"] == "/api/v1/instance-operations/terminate"
    assert "H100" not in json.dumps(dry_run)


def test_secret_inputs_require_pipe_or_stdin_and_have_redacted_repr(tmp_path: Path) -> None:
    fd = _pipe(base64.b64encode(b"opaque-jit" * 16))
    try:
        jit = live.read_jit_config_from_fd(fd)
    finally:
        os.close(fd)
    assert "opaque" not in repr(jit)
    jit.destroy()
    assert jit.destroyed is True
    with pytest.raises(live.ContractError, match="destroyed"):
        jit.copy_bytes()

    regular = tmp_path / "forbidden-secret.txt"
    regular.write_bytes(b"lambda-test-key")
    with regular.open("rb") as handle:
        with pytest.raises(live.ContractError, match="regular_file_rejected"):
            live.LambdaHttpClient.from_secret_fd(handle.fileno())


def test_secret_pipe_read_timeout_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    read_fd, write_fd = os.pipe()
    ticks = iter((0.0, float(live.SECRET_FD_READ_SECONDS + 1)))

    def blocked_read(_: int, __: int) -> bytes:
        raise BlockingIOError

    monkeypatch.setattr(live.os, "read", blocked_read)
    monkeypatch.setattr(live.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(live.time, "sleep", lambda _: None)
    try:
        with pytest.raises(live.ContractError, match="jit_config_read_timeout"):
            live.read_jit_config_from_fd(read_fd)
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_lambda_client_cannot_be_constructed_with_value_or_environment() -> None:
    with pytest.raises(live.ContractError, match="requires_secret_fd"):
        live.LambdaHttpClient("secret")
    assert "LAMBDA_API_KEY" not in live.LambdaHttpClient.__dict__


@pytest.mark.parametrize(
    ("operation", "method", "path", "body"),
    [
        ("create_api_key", "POST", "/api/v1/api-keys", bytearray(b"{}")),
        (
            "create_filesystem",
            "POST",
            "/api/v1/filesystems",
            bytearray(b'{"name":"forbidden","region":"us-east-1"}'),
        ),
        ("instances", "GET", "/api/v1/ssh-keys", None),
        (
            "delete_ruleset",
            "DELETE",
            "/api/v1/firewall-rulesets/%2E%2E%2Fglobal",
            None,
        ),
    ],
)
def test_provider_request_surface_is_exactly_allowlisted(
    operation: str,
    method: str,
    path: str,
    body: bytearray | None,
) -> None:
    with pytest.raises(live.ContractError, match="allowlisted"):
        live.ProviderRequest(operation, method, path, method != "GET", body=body)


def test_lambda_client_paces_every_read_and_mutation_start_without_retry() -> None:
    clock = FakePacingClock()
    starts: list[float] = []

    def transport(request: live.ProviderRequest, _: live.SecretBuffer) -> live.ProviderResponse:
        starts.append(clock.monotonic())
        return live.ProviderResponse(
            request_sha256=request.request_sha256,
            status_code=200,
            content_type="application/json",
            body=b'{"data":[]}',
        )

    fd = _pipe(b"lambda-test-key")
    try:
        client = live.LambdaHttpClient.from_secret_fd(
            fd,
            transport=transport,
            monotonic_clock=clock.monotonic,
            sleep=clock.sleep,
        )
    finally:
        os.close(fd)
    requests = [
        live.ProviderRequest("instances", "GET", "/api/v1/instances", False),
        live.ProviderRequest("images", "GET", "/api/v1/images", False),
        live.ProviderRequest(
            "terminate",
            "POST",
            "/api/v1/instance-operations/terminate",
            True,
            body=bytearray(b'{"instance_ids":["owned"]}'),
        ),
    ]
    for request in requests:
        if request.mutating:
            client._request_mutation(request)
        else:
            client.request(request)
    assert starts == [0.0, 1.0, 2.0]
    assert clock.sleeps == [1.0, 1.0]
    assert len(starts) == len(requests)

    direct = live.ProviderRequest(
        "terminate",
        "POST",
        "/api/v1/instance-operations/terminate",
        True,
        body=bytearray(b'{"instance_ids":["owned"]}'),
    )
    with pytest.raises(live.ContractError, match="direct_provider_mutation_rejected"):
        client.request(direct)
    client.close()


def test_lambda_client_rejects_a_stalled_injected_pacing_clock() -> None:
    contacted = False

    def transport(request: live.ProviderRequest, _: live.SecretBuffer) -> live.ProviderResponse:
        nonlocal contacted
        contacted = True
        return live.ProviderResponse(
            request.request_sha256,
            200,
            "application/json",
            b'{"data":[]}',
        )

    fd = _pipe(b"lambda-test-key")
    try:
        client = live.LambdaHttpClient.from_secret_fd(
            fd,
            transport=transport,
            monotonic_clock=lambda: 0.0,
            sleep=lambda _: None,
        )
    finally:
        os.close(fd)
    client.request(live.ProviderRequest("instances", "GET", "/api/v1/instances", False))
    contacted = False
    with pytest.raises(live.ContractError, match="pacing_clock_stalled"):
        client.request(live.ProviderRequest("images", "GET", "/api/v1/images", False))
    assert contacted is False
    client.close()


def test_read_only_discovery_binds_exact_live_target_and_builds_plan() -> None:
    transport = FakeTransport(_base_payloads())
    client = _client(transport)
    identity = _fake_identity()
    discovery = live.capture_action_time_discovery(client, ssh_key_name="preexisting-fixture-key")
    public = discovery.to_public_mapping()
    assert public["zero_instances"] is True
    assert public["zero_firewall_rulesets"] is True
    assert public["target"]["price_cents_per_hour"] == 2232
    assert public["target"]["storage_gib"] == 20480
    assert public["image_candidates"] == [_image()]
    assert set(public["payload_digests"]) == {operation for operation, _ in live.READ_OPERATIONS}
    assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in public["payload_digests"].values())
    assert len(public["response_bindings"]) == 8
    assert (
        public["snapshot_sha256"]
        == hashlib.sha256(
            live._canonical_json(
                {
                    "payload_digests": public["payload_digests"],
                    "bindings": public["response_bindings"],
                }
            )
        ).hexdigest()
    )
    now = int(time.time())
    plan = live.build_plan_from_discovery(
        discovery,
        head_sha=HEAD_SHA,
        lifecycle_nonce=NONCE,
        created_at_unix=now - 1,
        expires_at_unix=now + 3600,
        current_public_ipv4_cidr="8.8.8.8/32",
        image_id=_image()["id"],
        host_identity=identity,
        runtime_bundle=_runtime_bundle(),
    )
    assert len(plan.sha256) == 64
    assert plan.to_mapping()["target"]["gpu_description"] == "A100 (80 GB SXM4)"
    assert plan.to_mapping()["production_authorized"] is False
    assert all(request["method"] == "GET" for request in transport.requests)
    client.close()


def test_discovery_canonicalizes_provider_public_key_comments() -> None:
    payloads = _base_payloads()
    payloads["ssh_keys"]["data"][0]["public_key"] = f"{SSH_PUBLIC} preexisting-fixture-key"
    client = _client(FakeTransport(payloads))
    discovery = live.capture_action_time_discovery(client, ssh_key_name="preexisting-fixture-key")
    assert discovery.ssh_public_key_sha256 == hashlib.sha256(SSH_PUBLIC.encode("ascii")).hexdigest()
    client.close()


def test_action_time_revalidation_canonicalizes_provider_public_key_comments() -> None:
    payloads = _base_payloads()
    payloads["ssh_keys"]["data"][0]["public_key"] = f"{SSH_PUBLIC} preexisting-fixture-key"
    adapter = _adapter(FakeTransport(payloads))
    baseline = adapter.observe("baseline")
    assert baseline.phase == "baseline"


def test_full_eight_read_snapshot_is_paced_within_declared_budgets() -> None:
    fake = FakeTransport(_base_payloads())
    clock = FakePacingClock()
    starts: list[float] = []

    def transport(
        request: live.ProviderRequest, api_key: live.SecretBuffer
    ) -> live.ProviderResponse:
        starts.append(clock.monotonic())
        return fake(request, api_key)

    fd = _pipe(b"lambda-test-key")
    try:
        client = live.LambdaHttpClient.from_secret_fd(
            fd,
            transport=transport,
            monotonic_clock=clock.monotonic,
            sleep=clock.sleep,
        )
    finally:
        os.close(fd)
    discovery = live.capture_action_time_discovery(client, ssh_key_name="preexisting-fixture-key")
    assert starts == [float(index) for index in range(8)]
    assert sum(clock.sleeps) == 7.0
    assert sum(clock.sleeps) < live.MAX_OBSERVATION_WINDOW_SECONDS
    assert sum(clock.sleeps) < live.PRESTATE_FRESHNESS_SECONDS
    assert len(discovery.to_public_mapping()["response_bindings"]) == 8
    client.close()


def test_discovery_admits_only_exact_lambda_stack_22_04_family() -> None:
    payloads = _base_payloads()
    unrelated = dict(_image())
    unrelated.update(
        {
            "id": "image-gpu-base-2204",
            "name": "GPU Base 22.04",
            "family": "gpu-base-22-04",
            "version": "2026.08-other",
        }
    )
    payloads["images"]["data"].append(unrelated)
    client = _client(FakeTransport(payloads))
    discovery = live.capture_action_time_discovery(client, ssh_key_name="preexisting-fixture-key")
    assert [image["id"] for image in discovery.images] == [_image()["id"]]
    assert discovery.images[0]["family"] == live.TARGET_IMAGE_FAMILY
    with pytest.raises(live.ContractError, match="selected_image_not_in_discovery"):
        live.build_plan_from_discovery(
            discovery,
            head_sha=HEAD_SHA,
            lifecycle_nonce=NONCE,
            created_at_unix=int(time.time()) - 1,
            expires_at_unix=int(time.time()) + 60,
            current_public_ipv4_cidr="8.8.8.8/32",
            image_id=unrelated["id"],
            host_identity=_fake_identity(),
            runtime_bundle=_runtime_bundle(),
        )
    client.close()


def test_discovery_public_field_mutation_cannot_change_the_sealed_plan_source() -> None:
    transport = FakeTransport(_base_payloads())
    client = _client(transport)
    discovery = live.capture_action_time_discovery(client, ssh_key_name="preexisting-fixture-key")
    discovery.images[0]["id"] = "image-attacker-substitution"
    with pytest.raises(live.ContractError, match="discovery_binding_digest_mismatch"):
        live.build_plan_from_discovery(
            discovery,
            head_sha=HEAD_SHA,
            lifecycle_nonce=NONCE,
            created_at_unix=int(time.time()) - 1,
            expires_at_unix=int(time.time()) + 60,
            current_public_ipv4_cidr="8.8.8.8/32",
            image_id="image-attacker-substitution",
            host_identity=_fake_identity(),
            runtime_bundle=_runtime_bundle(),
        )
    client.close()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda p: p["instance_types"]["data"][live.TARGET_INSTANCE_TYPE][
                "instance_type"
            ].update({"gpu_description": "H100 (80 GB SXM5)"}),
            "not_a100",
        ),
        (
            lambda p: p["instance_types"]["data"][live.TARGET_INSTANCE_TYPE].update(
                {"regions_with_capacity_available": []}
            ),
            "capacity",
        ),
        (
            lambda p: p["instance_types"]["data"][live.TARGET_INSTANCE_TYPE]["instance_type"][
                "specs"
            ].update({"gpus": 2}),
            "gpu_count",
        ),
        (
            lambda p: p["images"]["data"][0].update({"architecture": "arm64"}),
            "eligible_lambda_stack_22_04",
        ),
        (
            lambda p: p["images"]["data"][0].update({"family": "ubuntu-24-04"}),
            "eligible_lambda_stack_22_04",
        ),
        (
            lambda p: p["images"]["data"][0].update({"updated_time": "2099-01-01T00:00:00Z"}),
            "timestamp_in_future",
        ),
        (
            lambda p: p["ssh_keys"]["data"][0].update(
                {"public_key": "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCTest"}
            ),
            "not_ed25519",
        ),
        (
            lambda p: p["file_systems"]["data"][0].pop("created_by"),
            "required_keys_missing",
        ),
        (lambda p: p["instances"].update({"data": [{"id": "foreign"}]}), "instance"),
    ],
)
def test_discovery_fails_closed_on_capacity_hardware_or_inventory_drift(
    change: Any, message: str
) -> None:
    payloads = _base_payloads()
    change(payloads)
    client = _client(FakeTransport(payloads))
    with pytest.raises(live.ContractError, match=message):
        live.capture_action_time_discovery(client, ssh_key_name="preexisting-fixture-key")
    client.close()


def test_two_independent_gates_and_exact_plan_digest_are_required() -> None:
    plan = _plan()
    client = _client(FakeTransport(_base_payloads()))
    with pytest.raises(live.ContractError, match="production_gate_closed"):
        live.LambdaLiveAdapter(client, plan, live.LiveGates(False, True, plan.sha256))
    with pytest.raises(live.ContractError, match="provider_mutation_gate_closed"):
        live.LambdaLiveAdapter(client, plan, live.LiveGates(True, False, plan.sha256))
    with pytest.raises(live.ContractError, match="digest_mismatch"):
        live.LambdaLiveAdapter(client, plan, live.LiveGates(True, True, "f" * 64))
    client.close()


def test_mutation_intent_sink_is_bound_once_before_any_observation_receipt() -> None:
    transport = FakeTransport(_base_payloads())
    plan = _plan()
    adapter = live.LambdaLiveAdapter(
        _client(transport), plan, live.LiveGates(True, True, plan.sha256)
    )
    captured: list[live.MutationIntent] = []
    binding = adapter.bind_mutation_intent_callback(captured.append)
    assert adapter.mutation_intent_callback_bound is True
    assert adapter.mutation_intent_binding_sha256 == binding
    assert adapter.mutation_intent_callback_matches(captured.append) is True
    assert len(binding) == 64
    with pytest.raises(live.ContractError, match="mutation_intent_callback_already_bound"):
        adapter.bind_mutation_intent_callback(captured.append)


def test_mutation_intent_sink_cannot_be_bound_after_observation() -> None:
    transport = FakeTransport(_base_payloads())
    plan = _plan()
    adapter = live.LambdaLiveAdapter(
        _client(transport), plan, live.LiveGates(True, True, plan.sha256)
    )
    adapter.observe("baseline")
    with pytest.raises(live.ContractError, match="mutation_intent_callback_binding_too_late"):
        adapter.bind_mutation_intent_callback(lambda _: None)


def test_provider_mutation_requires_bound_write_ahead_sink() -> None:
    transport = FakeTransport(_base_payloads())
    plan = _plan()
    adapter = live.LambdaLiveAdapter(
        _client(transport), plan, live.LiveGates(True, True, plan.sha256)
    )
    baseline = adapter.observe("baseline")
    with pytest.raises(live.ContractError, match="mutation_intent_callback_not_bound"):
        adapter.restrict_global(baseline)
    assert not any(request["operation"] == "restrict_global" for request in transport.requests)


@pytest.mark.parametrize("source", ["0.0.0.0/0", "8.8.8.0/24", "10.0.0.1/32", "::1/128"])
def test_plan_rejects_nonpublic_or_nonhost_controller_sources(source: str) -> None:
    identity = _fake_identity()
    now = int(time.time())
    with pytest.raises(live.ContractError, match="controller_source"):
        live.build_immutable_plan(
            head_sha=HEAD_SHA,
            lifecycle_nonce=NONCE,
            created_at_unix=now,
            expires_at_unix=now + 60,
            current_public_ipv4_cidr=source,
            region_description="Illinois, USA",
            image_id=_image()["id"],
            image_created_time=_image()["created_time"],
            image_description=_image()["description"],
            image_name=_image()["name"],
            image_family=_image()["family"],
            image_version=_image()["version"],
            image_updated_time=_image()["updated_time"],
            instance_type_description=_instance_type()["description"],
            gpu_description=_instance_type()["gpu_description"],
            price_cents_per_hour=2232,
            vcpus=240,
            memory_gib=1800,
            storage_gib=20480,
            ssh_key_name="preexisting-fixture-key",
            ssh_public_key_sha256=hashlib.sha256(SSH_PUBLIC.encode()).hexdigest(),
            baseline_file_systems_sha256=hashlib.sha256(_canonical(_file_systems())).hexdigest(),
            original_global_rules=_original_rules(),
            host_key_fingerprint=identity.fingerprint,
            runtime_bundle_sha256=_runtime_bundle().sha256,
        )


def test_full_lifecycle_uses_exact_requests_ownership_and_fresh_receipts() -> None:
    identity = _fake_identity()
    transport = FakeTransport(_base_payloads())
    intents: list[live.MutationIntent] = []
    adapter = _adapter(transport, identity, mutation_intent_callback=intents.append)
    plan = adapter._plan

    baseline = adapter.observe("baseline")
    desired_global = {
        "data": {
            "id": "global",
            "name": "Global",
            "rules": [rule.to_mapping() for rule in plan.desired_firewall_rules],
        }
    }
    transport.mutations["restrict_global"] = desired_global
    restrict_receipt = adapter.restrict_global(baseline)
    assert restrict_receipt.request_body_sha256
    assert intents[-1].request_sha256 == restrict_receipt.request_sha256
    assert intents[-1].prestate_snapshot_sha256 == baseline.snapshot_sha256
    with pytest.raises(live.ContractError, match="reused"):
        adapter.restrict_global(baseline)

    _set_global(transport, plan.desired_firewall_rules)
    restricted = adapter.observe("global_restricted")
    created_ruleset = _ruleset(plan)
    transport.mutations["create_ruleset"] = {"data": created_ruleset}
    ruleset_receipt = adapter.create_ruleset(restricted)
    assert ruleset_receipt.ruleset_id == RULESET_ID

    transport.payloads["firewall_rulesets"] = {"data": [created_ruleset]}
    ready = adapter.observe("ruleset_ready")
    transport.mutations["launch"] = {"data": {"instance_ids": [INSTANCE_ID]}}
    launch_receipt = adapter.launch(ready, identity, _runtime_bundle())
    assert launch_receipt.instance_id == INSTANCE_ID
    assert identity.destroyed is True
    launch_request = [
        request for request in transport.requests if request["operation"] == "launch"
    ][0]
    launch_body = json.loads(launch_request["body"])
    assert set(launch_body) == {
        "region_name",
        "instance_type_name",
        "ssh_key_names",
        "file_system_names",
        "name",
        "image",
        "user_data",
        "tags",
        "firewall_rulesets",
    }
    assert launch_body["file_system_names"] == []
    assert "quantity" not in launch_body
    assert "ed25519_private" in launch_body["user_data"]
    assert f"path: {live.REMOTE_RUNTIME_ROOT}/bootstrap.py" in launch_body["user_data"]
    assert "permissions: '0444'" in launch_body["user_data"]
    assert launch_body["tags"] == [
        {"key": key, "value": value} for key, value in plan.ownership_tags
    ]
    assert "user_data" not in json.dumps(launch_receipt.to_public_mapping())
    launch_intent = [intent for intent in intents if intent.operation == "launch"][0]
    assert launch_intent.request_sha256 == launch_receipt.request_sha256
    assert launch_intent.sensitive_body is True
    assert "ed25519_private" not in json.dumps(launch_intent.to_public_mapping())

    active_instance = _instance(plan)
    created_ruleset["instance_ids"] = [INSTANCE_ID]
    transport.payloads["instances"] = {"data": [active_instance]}
    transport.payloads["firewall_rulesets"] = {"data": [created_ruleset]}
    bound = adapter.observe("instance_bound")
    assert bound.instance_id == INSTANCE_ID
    assert bound.instance_public_ipv4 == INSTANCE_IP

    transport.mutations["terminate"] = {
        "data": {"terminated_instances": [_instance(plan, status="terminating")]}
    }
    terminate_receipt = adapter.terminate(bound)
    assert terminate_receipt.instance_id == INSTANCE_ID

    created_ruleset["instance_ids"] = []
    transport.payloads["instances"] = {"data": []}
    transport.payloads["firewall_rulesets"] = {"data": [created_ruleset]}
    absent = adapter.observe("instance_absent")
    transport.mutations["delete_ruleset"] = {"data": {}}
    delete_receipt = adapter.delete_ruleset(absent)
    assert delete_receipt.ruleset_id == RULESET_ID
    delete_request = [
        request for request in transport.requests if request["operation"] == "delete_ruleset"
    ][0]
    assert delete_request["method"] == "DELETE"
    assert delete_request["path"] == f"/api/v1/firewall-rulesets/{RULESET_ID}"

    transport.payloads["firewall_rulesets"] = {"data": []}
    no_ruleset = adapter.observe("ruleset_absent")
    original_global = {"data": {"id": "global", "name": "Global", "rules": _original_rules()}}
    transport.mutations["restore_global"] = original_global
    adapter.restore_global(no_ruleset)
    _set_global(transport, plan.original_global_rules)
    restored = adapter.observe("restored")
    assert restored.instance_id is None
    assert restored.ruleset_id is None
    assert transport.payloads["file_systems"]["data"] == _file_systems()
    assert [intent.operation for intent in intents] == list(live.MUTATION_PATHS)


def test_mutation_intent_is_durable_before_provider_contact() -> None:
    transport = FakeTransport(_base_payloads())
    captured: list[live.MutationIntent] = []

    def persist(intent: live.MutationIntent) -> None:
        assert not any(request["operation"] == "restrict_global" for request in transport.requests)
        captured.append(intent)

    adapter = _adapter(transport, mutation_intent_callback=persist)
    baseline = adapter.observe("baseline")
    transport.mutations["restrict_global"] = {
        "data": {
            "id": "global",
            "name": "Global",
            "rules": [rule.to_mapping() for rule in adapter._plan.desired_firewall_rules],
        }
    }
    receipt = adapter.restrict_global(baseline)
    assert len(captured) == 1
    public = captured[0].to_public_mapping()
    assert public["request_sha256"] == receipt.request_sha256
    assert public["prestate_receipt_nonce"] == baseline.receipt_nonce
    assert public["callback_binding_sha256"] == adapter.mutation_intent_binding_sha256
    assert public["request_body_redacted"] is True
    assert public["path"] == "/api/v1/firewall-rulesets/global"
    reloaded = live.MutationIntent.from_public_mapping(public)
    assert reloaded == captured[0]
    ambiguity = adapter.ambiguity_from_persisted_intent(public)
    assert ambiguity.operation == "restrict_global"
    assert ambiguity.request_sha256 == receipt.request_sha256

    for field, replacement, expected in (
        ("method", "POST", "mutation_intent_method_mismatch"),
        ("request_sha256", "f" * 64, "mutation_intent_request_sha256_mismatch"),
        ("request_body_redacted", False, "mutation_intent_request_body_redaction_rejected"),
        ("sensitive_body", True, "mutation_intent_sensitive_body_mismatch"),
    ):
        tampered = dict(public)
        tampered[field] = replacement
        with pytest.raises(live.ContractError, match=expected):
            live.MutationIntent.from_public_mapping(tampered)

    wrong_plan = dict(public)
    wrong_plan["plan_sha256"] = "f" * 64
    with pytest.raises(live.ContractError, match="mutation_intent_plan_mismatch"):
        adapter.ambiguity_from_persisted_intent(wrong_plan)


def test_failed_mutation_intent_persistence_prevents_provider_contact() -> None:
    transport = FakeTransport(_base_payloads())

    def fail_to_persist(_: live.MutationIntent) -> None:
        raise RuntimeError("simulated_durable_journal_failure")

    adapter = _adapter(transport, mutation_intent_callback=fail_to_persist)
    baseline = adapter.observe("baseline")
    with pytest.raises(RuntimeError, match="simulated_durable_journal_failure"):
        adapter.restrict_global(baseline)
    assert not any(request["operation"] == "restrict_global" for request in transport.requests)
    with pytest.raises(live.ContractError, match="prestate_receipt_reused"):
        adapter.restrict_global(baseline)


def test_mutation_timeout_is_ambiguous_never_retried_and_inventory_classifies() -> None:
    transport = FakeTransport(_base_payloads())
    adapter = _adapter(transport)
    baseline = adapter.observe("baseline")
    transport.force_response = live.ProviderResponse(
        request_sha256="will-be-rebound-below",
        status_code=None,
        content_type=None,
        body=b"",
        timed_out=True,
    )

    def timeout_response(
        outgoing: live.ProviderRequest, api_key: live.SecretBuffer
    ) -> live.ProviderResponse:
        assert api_key.copy_bytes() == b"lambda-test-key"
        transport.requests.append({"operation": outgoing.operation})
        return live.ProviderResponse(
            request_sha256=outgoing.request_sha256,
            status_code=None,
            content_type=None,
            body=b"",
            timed_out=True,
        )

    adapter._client._transport = timeout_response
    with pytest.raises(live.AmbiguousMutation, match="ambiguous_restrict_global_timeout") as caught:
        adapter.restrict_global(baseline)
    mutation_calls = [r for r in transport.requests if r["operation"] == "restrict_global"]
    assert len(mutation_calls) == 1
    with pytest.raises(live.ContractError, match="reused"):
        adapter.restrict_global(baseline)

    adapter._client._transport = FakeTransport(transport.payloads)
    _set_global(adapter._client._transport, adapter._plan.desired_firewall_rules)
    recovery = adapter.observe("recovery")
    result = adapter.recover_ambiguous(caught.value, recovery)
    assert result.outcome == "applied_exactly_once"


@pytest.mark.parametrize(
    "response",
    [
        live.ProviderResponse("wrong", 200, "application/json", b'{"data":{}}'),
        live.ProviderResponse("placeholder", 500, "application/json", b"{}"),
        live.ProviderResponse("placeholder", 200, "text/html", b"{}"),
        live.ProviderResponse("placeholder", 200, "application/json", b"not-json"),
        live.ProviderResponse("placeholder", 200, "application/json", b"[]"),
        live.ProviderResponse("placeholder", 200, "application/json", b"null"),
        live.ProviderResponse("placeholder", 200, "application/json", b"true"),
        live.ProviderResponse("placeholder", 200, "application/json", b"1"),
    ],
)
def test_every_uncertain_mutation_response_fails_as_ambiguous(
    response: live.ProviderResponse,
) -> None:
    transport = FakeTransport(_base_payloads())
    adapter = _adapter(transport)
    baseline = adapter.observe("baseline")

    def uncertain(
        request: live.ProviderRequest, api_key: live.SecretBuffer
    ) -> live.ProviderResponse:
        assert api_key.copy_bytes() == b"lambda-test-key"
        return live.ProviderResponse(
            request_sha256=(
                request.request_sha256
                if response.request_sha256 == "placeholder"
                else response.request_sha256
            ),
            status_code=response.status_code,
            content_type=response.content_type,
            body=response.body,
        )

    adapter._client._transport = uncertain
    with pytest.raises(live.AmbiguousMutation):
        adapter.restrict_global(baseline)


def test_success_status_with_mutation_schema_drift_is_still_ambiguous() -> None:
    transport = FakeTransport(_base_payloads())
    adapter = _adapter(transport)
    baseline = adapter.observe("baseline")
    transport.mutations["restrict_global"] = {"data": {}}
    with pytest.raises(live.AmbiguousMutation, match="response_schema"):
        adapter.restrict_global(baseline)


def test_duplicate_json_key_in_mutation_response_is_ambiguous() -> None:
    transport = FakeTransport(_base_payloads())
    adapter = _adapter(transport)
    baseline = adapter.observe("baseline")

    def duplicate_json(
        request: live.ProviderRequest, api_key: live.SecretBuffer
    ) -> live.ProviderResponse:
        assert api_key.copy_bytes() == b"lambda-test-key"
        return live.ProviderResponse(
            request.request_sha256,
            200,
            "application/json",
            b'{"data":{},"data":{}}',
        )

    adapter._client._transport = duplicate_json
    with pytest.raises(live.AmbiguousMutation, match="_json"):
        adapter.restrict_global(baseline)


def test_ambiguous_termination_recognizes_exact_owned_in_progress_state() -> None:
    transport = FakeTransport(_base_payloads())
    adapter = _adapter(transport)
    plan = adapter._plan
    _set_global(transport, plan.desired_firewall_rules)
    ruleset = _ruleset(plan, instance_ids=[INSTANCE_ID])
    transport.payloads["firewall_rulesets"] = {"data": [ruleset]}
    transport.payloads["instances"] = {"data": [_instance(plan, status="terminating")]}
    recovery = adapter.observe("recovery")
    ambiguity = live.AmbiguousMutation("terminate", "d" * 64, "timeout")
    result = adapter.recover_ambiguous(ambiguity, recovery)
    assert result.outcome == "applied_in_progress"
    assert result.instance_id == INSTANCE_ID
    with pytest.raises(live.ContractError, match="reused"):
        adapter.recover_ambiguous(ambiguity, recovery)


def test_plan_expiry_blocks_creation_but_never_blocks_exact_owned_teardown() -> None:
    transport = FakeTransport(_base_payloads())
    current = _plan()
    now = int(time.time())
    expired = replace(current, created_at_unix=now - 3600, expires_at_unix=now - 1)
    gates = live.LiveGates(True, True, expired.sha256)
    adapter = live.LambdaLiveAdapter(
        _client(transport),
        expired,
        gates,
        mutation_intent_callback=lambda _: None,
    )

    baseline = adapter.observe("baseline")
    with pytest.raises(live.ContractError, match="plan_not_current"):
        adapter.restrict_global(baseline)

    _set_global(transport, expired.desired_firewall_rules)
    ruleset = _ruleset(expired, instance_ids=[INSTANCE_ID])
    transport.payloads["firewall_rulesets"] = {"data": [ruleset]}
    transport.payloads["instances"] = {"data": [_instance(expired)]}
    bound = adapter.observe("instance_bound")
    transport.mutations["terminate"] = {
        "data": {"terminated_instances": [_instance(expired, status="terminating")]}
    }
    receipt = adapter.terminate(bound)
    assert receipt.instance_id == INSTANCE_ID


def test_action_time_price_drift_is_rechecked_before_each_mutation() -> None:
    transport = FakeTransport(_base_payloads())
    adapter = _adapter(transport)
    transport.payloads["instance_types"]["data"][live.TARGET_INSTANCE_TYPE]["instance_type"][
        "price_cents_per_hour"
    ] = 2233
    with pytest.raises(live.ContractError, match="price_drift"):
        adapter.observe("baseline")


def test_foreign_ruleset_or_instance_is_never_adopted() -> None:
    transport = FakeTransport(_base_payloads())
    adapter = _adapter(transport)
    plan = adapter._plan
    _set_global(transport, plan.desired_firewall_rules)
    foreign = _ruleset(plan)
    foreign["name"] = "someone-else"
    transport.payloads["firewall_rulesets"] = {"data": [foreign]}
    with pytest.raises(live.ContractError, match="owned_ruleset_not_sole"):
        adapter.observe("ruleset_ready")


def test_strict_ssh_binding_never_uses_tofu(tmp_path: Path) -> None:
    identity = _fake_identity()
    assert "PRIVATE" not in repr(identity)
    known_hosts = identity.known_hosts(INSTANCE_IP)
    assert known_hosts.startswith(f"{INSTANCE_IP} ssh-ed25519 ")
    identity_file = tmp_path / "lambda_key.pem"
    identity_file.write_text("test access key", encoding="ascii")
    if os.name != "nt":
        identity_file.chmod(0o600)
    evidence_directory = live.create_evidence_directory((tmp_path / "release-evidence").resolve())
    try:
        known_hosts_receipt = live.write_public_known_hosts(
            (Path(evidence_directory.absolute_path) / "known_hosts").resolve(),
            identity=identity,
            public_ipv4=INSTANCE_IP,
            evidence_directory_receipt=evidence_directory,
        )
    finally:
        evidence_directory.close()
    binding = live.build_strict_ssh_binding(
        identity=identity,
        public_ipv4=INSTANCE_IP,
        access_identity_file=str(identity_file.resolve()),
        known_hosts_file=known_hosts_receipt,
        remote_mode="run",
    )
    argv = list(binding.argv_prefix)
    assert "-T" in argv
    assert "RequestTTY=no" in argv
    assert "StrictHostKeyChecking=yes" in argv
    assert "HostKeyAlgorithms=ssh-ed25519" in argv
    assert "IdentityAgent=none" in argv
    assert f"UserKnownHostsFile={known_hosts_receipt.absolute_path}" in argv
    assert "ForwardAgent=no" in argv
    assert f"GlobalKnownHostsFile={os.devnull}" in argv
    assert not any(
        "accept-new" in argument or "StrictHostKeyChecking=no" in argument for argument in argv
    )
    assert tuple(argv[-len(live.FIXED_REMOTE_COMMAND) :]) == live.FIXED_REMOTE_COMMAND
    assert binding.known_hosts_sha256 == known_hosts_receipt.content_sha256
    preflight = live.build_strict_ssh_binding(
        identity=identity,
        public_ipv4=INSTANCE_IP,
        access_identity_file=str(identity_file.resolve()),
        known_hosts_file=known_hosts_receipt,
        remote_mode="preflight",
    )
    assert tuple(preflight.argv_prefix[-len(live.FIXED_PREFLIGHT_COMMAND) :]) == (
        live.FIXED_PREFLIGHT_COMMAND
    )
    assert preflight.remote_mode == "preflight"
    cloud_init = live.build_strict_ssh_binding(
        identity=identity,
        public_ipv4=INSTANCE_IP,
        access_identity_file=str(identity_file.resolve()),
        known_hosts_file=known_hosts_receipt,
        remote_mode="cloud-init",
    )
    assert tuple(cloud_init.argv_prefix[-len(live.FIXED_CLOUD_INIT_WAIT_COMMAND) :]) == (
        live.FIXED_CLOUD_INIT_WAIT_COMMAND
    )
    assert cloud_init.remote_mode == "cloud-init"
    ssh_keygen = shutil.which("ssh-keygen")
    if ssh_keygen is not None:
        lookup = subprocess.run(
            [ssh_keygen, "-F", INSTANCE_IP, "-f", known_hosts_receipt.absolute_path],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert lookup.returncode == 0
        assert INSTANCE_IP in lookup.stdout
    public = json.dumps(binding.to_public_mapping())
    assert "PRIVATE" not in public
    assert str(identity_file.resolve()) not in public
    assert "<redacted-existing-access-identity-file>" in public
    identity.destroy()
    assert identity.destroyed is True


def _bound_host_preflight_inputs(
    tmp_path: Path,
    runtime_bundle: live.RuntimeBundle | None = None,
) -> tuple[
    live.ImmutablePlan,
    live.SnapshotReceipt,
    live.KnownHostsFileReceipt,
    live.CloudInitWaitReceipt,
]:
    identity = _fake_identity()
    transport = FakeTransport(_base_payloads())
    adapter = _adapter(transport, identity, runtime_bundle)
    plan = adapter._plan
    _set_global(transport, plan.desired_firewall_rules)
    transport.payloads["firewall_rulesets"] = {"data": [_ruleset(plan, instance_ids=[INSTANCE_ID])]}
    transport.payloads["instances"] = {"data": [_instance(plan)]}
    cloud_init_provider_instance = adapter.observe("instance_bound")
    evidence_directory = live.create_evidence_directory((tmp_path / "release-evidence").resolve())
    try:
        known_hosts = live.write_public_known_hosts(
            (Path(evidence_directory.absolute_path) / "known_hosts").resolve(),
            identity=identity,
            public_ipv4=INSTANCE_IP,
            evidence_directory_receipt=evidence_directory,
        )
    finally:
        evidence_directory.close()
    cloud_init_wait = live.validate_cloud_init_wait_receipt(
        b"status: done\n",
        b"",
        0,
        plan=plan,
        provider_instance=cloud_init_provider_instance,
        known_hosts=known_hosts,
    )
    provider_instance = adapter.observe("instance_bound")
    return plan, provider_instance, known_hosts, cloud_init_wait


def test_cloud_init_wait_and_host_preflight_bind_exact_instance_and_runtime(
    tmp_path: Path,
) -> None:
    plan, provider_instance, known_hosts, cloud_init_wait = _bound_host_preflight_inputs(tmp_path)
    assert cloud_init_wait.instance_id == INSTANCE_ID
    assert cloud_init_wait.instance_public_ipv4 == INSTANCE_IP
    assert cloud_init_wait.provider_receipt_nonce != provider_instance.receipt_nonce
    assert cloud_init_wait.to_public_mapping()["fixed_command"] == list(
        live.FIXED_CLOUD_INIT_WAIT_COMMAND
    )
    assert len(cloud_init_wait.sha256) == 64

    payload = _host_preflight_payload(plan)
    receipt = live.validate_host_preflight_receipt(
        _canonical(payload),
        plan=plan,
        provider_instance=provider_instance,
        known_hosts=known_hosts,
        cloud_init_wait=cloud_init_wait,
    )
    public = receipt.to_public_mapping()
    assert receipt.instance_id == INSTANCE_ID
    assert receipt.instance_public_ipv4 == INSTANCE_IP
    assert receipt.host_physical_gpu_uuids == tuple(payload["host_physical_gpu_uuids"])
    assert receipt.host_physical_gpu_products == (live.EXPECTED_HOST_GPU_PRODUCT,) * 8
    assert public["cloud_init_wait_binding_sha256"] == cloud_init_wait.sha256
    assert public["fixed_preflight_command"] == list(live.FIXED_PREFLIGHT_COMMAND)
    assert public["gpu_injection"]["verified"] is True
    assert public["gpu_injection"]["physical_gpu_uuids"] == payload["host_physical_gpu_uuids"]
    assert public["gpu_injection"]["network_mode"] == "none"
    assert public["gpu_injection"]["published_ports"] is False
    assert public["jit_config_received"] is False
    assert public["github_api_credential_received"] is False
    assert public["accepted_actions_evidence"] is False


def test_live_validator_accepts_actual_runtime_probe_host_gpu_injection_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.release_gpu_jit_lambda_runtime import executor
    from scripts.release_gpu_jit_lambda_runtime import runtime_contract as contract

    runtime_root = Path(live.__file__).resolve().parent.parent / "release_gpu_jit_lambda_runtime"
    runtime_bundle = live.load_runtime_bundle(runtime_root)
    plan, provider_instance, known_hosts, cloud_init_wait = _bound_host_preflight_inputs(
        tmp_path,
        runtime_bundle,
    )
    uuids = _host_preflight_payload(plan)["host_physical_gpu_uuids"]
    inventory = "".join(f"{uuid}, {contract.HOST_GPU_PRODUCT}\n" for uuid in uuids).encode("ascii")
    image_probe_output = (
        b"uid=1001\n"
        b"gid=1001\n"
        b"runner_listener=present\n"
        b"runner_version=2.336.0\n"
        b"runner_commit=98aabcd429c4e8402406c56ce2d26387fed3b9ce\n"
        b"node20=v20.20.2\n"
        b"node20_sha256=6295488653f0d93b0a157841746fef7e72cc4328cfb60c4bbe0ca2668a836ffd\n"
    )
    injection_output = (
        b"gpu_injection=verified\n" b"gpu_count=8\n" b"gpu_product=NVIDIA A100-SXM4-80GB\n"
    )
    image_inspect = [
        {
            "Id": contract.IMAGE_CONFIG_DIGEST,
            "Architecture": "amd64",
            "Os": "linux",
            "Config": {"User": "runner"},
            "RepoDigests": [contract.IMAGE_REFERENCE],
        }
    ]

    class RuntimeProbeCommands:
        def run(self, argv: Any, **_: Any) -> executor.CommandResult:
            command = tuple(argv)
            if command == (contract.CLOUD_INIT_PATH, "status", "--wait"):
                stdout = b"status: done\n"
            elif command == (
                contract.NVIDIA_SMI_PATH,
                "--query-gpu=uuid,name",
                "--format=csv,noheader",
            ):
                stdout = inventory
            elif command == (
                contract.DOCKER_PATH,
                "image",
                "inspect",
                contract.IMAGE_REFERENCE,
            ):
                stdout = json.dumps(image_inspect).encode("ascii")
            elif command == contract.render_image_probe_argv():
                stdout = image_probe_output
            elif command == contract.render_gpu_injection_probe_argv(uuids):
                stdout = injection_output
            else:
                stdout = b""
            return executor.CommandResult(command, 0, stdout, b"")

    monkeypatch.setattr(executor, "require_probe_stdin_eof", lambda: None)
    monkeypatch.setattr(executor, "verify_no_sensitive_environment", lambda: None)
    monkeypatch.setattr(executor, "harden_secret_process", lambda: None)
    monkeypatch.setattr(executor.os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(executor, "_verify_host_binary", lambda _: None)
    monkeypatch.setattr(executor, "verify_host_posture", lambda: None)
    monkeypatch.setattr(executor, "_ensure_no_global_runtime_residue", lambda _: None)
    monkeypatch.setattr(executor, "ExclusiveRuntimeLock", nullcontext)
    monkeypatch.setattr(executor, "runtime_bundle_sha256", lambda: runtime_bundle.sha256)

    runtime_receipt = executor.probe_host(RuntimeProbeCommands())
    gpu_argv = contract.render_gpu_injection_probe_argv(uuids)
    device_request = ",".join(uuids)
    assert "--network=none" in gpu_argv
    assert gpu_argv[gpu_argv.index("--gpus") + 1] == f'"device={device_request}"'
    assert "--publish" not in gpu_argv and "-p" not in gpu_argv
    validated = live.validate_host_preflight_receipt(
        _canonical(runtime_receipt),
        plan=plan,
        provider_instance=provider_instance,
        known_hosts=known_hosts,
        cloud_init_wait=cloud_init_wait,
    )
    public = validated.to_public_mapping()["gpu_injection"]
    assert public["verified"] is True
    assert public["physical_gpu_uuids"] == uuids
    assert (
        public["device_request_sha256"]
        == hashlib.sha256(device_request.encode("ascii")).hexdigest()
    )
    assert public["network_mode"] == "none"
    assert public["published_ports"] is False


@pytest.mark.parametrize(
    ("stdout", "stderr", "exit_code", "message"),
    [
        (b"status: running\n", b"", 0, "not_done"),
        (b"status: done\n", b"cloud-init error\n", 0, "failure_marker"),
        (b"status: done\n", b"", 1, "exit_code"),
        (b"status: done\x00\n", b"", 0, "output_nul"),
    ],
)
def test_cloud_init_wait_receipt_fails_closed(
    tmp_path: Path,
    stdout: bytes,
    stderr: bytes,
    exit_code: int,
    message: str,
) -> None:
    plan, provider_instance, known_hosts, _ = _bound_host_preflight_inputs(tmp_path)
    with pytest.raises(live.ContractError, match=message):
        live.validate_cloud_init_wait_receipt(
            stdout,
            stderr,
            exit_code,
            plan=plan,
            provider_instance=provider_instance,
            known_hosts=known_hosts,
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda p: p.update({"schema_version": True}), "schema_version"),
        (
            lambda p: p["host_physical_gpu_products"].__setitem__(0, "NVIDIA H100 80GB HBM3"),
            "gpu_products",
        ),
        (
            lambda p: p.update({"runtime_bundle_sha256": "f" * 64}),
            "runtime_bundle_mismatch",
        ),
        (
            lambda p: p["host_physical_gpu_uuids"].__setitem__(1, p["host_physical_gpu_uuids"][0]),
            "not_distinct",
        ),
        (
            lambda p: p["image"].update({"manifest_digest": "sha256:" + "f" * 64}),
            "manifest_mismatch",
        ),
        (
            lambda p: p["gpu_injection"].update({"gpu_injection_verified": False}),
            "injection_not_verified",
        ),
        (
            lambda p: p["gpu_injection"].update({"gpu_count": 7}),
            "injection_count",
        ),
        (
            lambda p: p["gpu_injection"].update({"gpu_product": "NVIDIA H100 80GB HBM3"}),
            "injection_product",
        ),
        (
            lambda p: p["gpu_injection"].update({"output_sha256": "f" * 64}),
            "injection_output_digest",
        ),
        (lambda p: p.update({"jit_config_received": True}), "jit_config_received"),
        (lambda p: p.update({"accepted_actions_evidence": True}), "accepted_actions_evidence"),
        (
            lambda p: p.update({"observed_at": "2026-01-01T00:00:00Z"}),
            "not_fresh",
        ),
        (
            lambda p: p["image"].update({"observed_at": "2026-01-01T00:00:00Z"}),
            "image_not_fresh",
        ),
        (lambda p: p.update({"unexpected": False}), "keys_rejected"),
    ],
)
def test_host_preflight_receipt_rejects_adversarial_drift(
    tmp_path: Path, change: Any, message: str
) -> None:
    plan, provider_instance, known_hosts, cloud_init_wait = _bound_host_preflight_inputs(tmp_path)
    payload = _host_preflight_payload(plan)
    change(payload)
    with pytest.raises(live.ContractError, match=message):
        live.validate_host_preflight_receipt(
            _canonical(payload),
            plan=plan,
            provider_instance=provider_instance,
            known_hosts=known_hosts,
            cloud_init_wait=cloud_init_wait,
        )


def test_host_preflight_requires_fresh_post_command_provider_inventory(tmp_path: Path) -> None:
    plan, provider_instance, known_hosts, cloud_init_wait = _bound_host_preflight_inputs(tmp_path)
    stale = replace(provider_instance, expires_monotonic_ns=time.monotonic_ns() - 1)
    with pytest.raises(live.ContractError, match="provider_receipt_stale"):
        live.validate_host_preflight_receipt(
            _canonical(_host_preflight_payload(plan)),
            plan=plan,
            provider_instance=stale,
            known_hosts=known_hosts,
            cloud_init_wait=cloud_init_wait,
        )


def test_host_preflight_requires_sealed_cloud_init_receipt_and_distinct_inventory(
    tmp_path: Path,
) -> None:
    plan, provider_instance, known_hosts, cloud_init_wait = _bound_host_preflight_inputs(tmp_path)
    tampered = replace(cloud_init_wait, stdout_sha256="f" * 64)
    with pytest.raises(live.ContractError, match="binding_digest_mismatch"):
        live.validate_host_preflight_receipt(
            _canonical(_host_preflight_payload(plan)),
            plan=plan,
            provider_instance=provider_instance,
            known_hosts=known_hosts,
            cloud_init_wait=tampered,
        )

    reused_inventory = replace(
        provider_instance,
        receipt_nonce=cloud_init_wait.provider_receipt_nonce,
    )
    with pytest.raises(live.ContractError, match="inventory_not_refreshed"):
        live.validate_host_preflight_receipt(
            _canonical(_host_preflight_payload(plan)),
            plan=plan,
            provider_instance=reused_inventory,
            known_hosts=known_hosts,
            cloud_init_wait=cloud_init_wait,
        )


def test_host_preflight_rereads_and_rejects_tampered_known_hosts(tmp_path: Path) -> None:
    plan, provider_instance, known_hosts, cloud_init_wait = _bound_host_preflight_inputs(tmp_path)
    Path(known_hosts.absolute_path).write_text(
        f"{INSTANCE_IP} ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAttacker\n",
        encoding="ascii",
    )
    with pytest.raises(live.ContractError, match="known_hosts_digest"):
        live.validate_host_preflight_receipt(
            _canonical(_host_preflight_payload(plan)),
            plan=plan,
            provider_instance=provider_instance,
            known_hosts=known_hosts,
            cloud_init_wait=cloud_init_wait,
        )


def test_missing_live_cryptography_dependency_fails_closed_without_import_breakage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    original_import = builtins.__import__

    def rejecting_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("cryptography"):
            raise ImportError("simulated absent live prerequisite")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", rejecting_import)
    with pytest.raises(live.ContractError, match="cryptography_dependency_unavailable"):
        live.generate_ephemeral_host_identity()


def test_public_evidence_is_exclusive_sanitized_and_owner_only_on_posix(
    tmp_path: Path,
) -> None:
    evidence_directory = live.create_evidence_directory((tmp_path / "release-evidence").resolve())
    try:
        output = (Path(evidence_directory.absolute_path) / "evidence.json").resolve()
        digest = live.write_public_evidence(
            output,
            {"plan_sha256": "a" * 64},
            evidence_directory_receipt=evidence_directory,
        )
        assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
        assert json.loads(output.read_text(encoding="ascii"))["plan_sha256"] == "a" * 64
        if os.name != "nt":
            assert stat.S_IMODE(output.stat().st_mode) == 0o600
        with pytest.raises(live.ContractError, match="exists"):
            live.write_public_evidence(
                output,
                {"overwrite": True},
                evidence_directory_receipt=evidence_directory,
            )
    finally:
        evidence_directory.close()


def test_plan_and_receipts_never_expose_secret_material() -> None:
    identity = _fake_identity()
    plan = _plan(identity)
    serialized = json.dumps(plan.to_mapping())
    assert "BEGIN OPENSSH PRIVATE KEY" not in serialized
    assert "lambda-test-key" not in serialized
    assert "encoded_jit_config" not in serialized
    assert plan.to_mapping()["secret_transport"] == {
        "lambda_api_key": "anonymous-fd-or-stdin-only",
        "github_jit_config": "anonymous-fd-or-stdin-only",
        "host_private_key": "in-memory-cloud-init-only",
    }
    assert plan.to_mapping()["remote_runtime"]["fixed_command"] == list(live.FIXED_REMOTE_COMMAND)
    assert plan.to_mapping()["remote_runtime"]["fixed_cloud_init_wait_command"] == list(
        live.FIXED_CLOUD_INIT_WAIT_COMMAND
    )
    assert plan.to_mapping()["remote_runtime"]["fixed_preflight_command"] == list(
        live.FIXED_PREFLIGHT_COMMAND
    )


def test_runtime_bundle_hash_matches_remote_executor_framing() -> None:
    from scripts.release_gpu_jit_lambda_runtime import runtime_contract

    root = Path(live.__file__).resolve().parent.parent / "release_gpu_jit_lambda_runtime"
    bundle = live.load_runtime_bundle(root)
    digest = hashlib.sha256()
    for name, content in bundle.files:
        encoded_name = name.encode("ascii")
        digest.update(len(encoded_name).to_bytes(2, "big"))
        digest.update(encoded_name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    assert bundle.sha256 == digest.hexdigest()
    assert live.EXPECTED_HOST_GPU_PRODUCT == runtime_contract.HOST_GPU_PRODUCT
    assert _host_preflight_payload(_plan())["kind"] == runtime_contract.HOST_PREFLIGHT_KIND
    assert live.EXPECTED_RUNNER_IMAGE_REFERENCE == runtime_contract.IMAGE_REFERENCE
    assert live.EXPECTED_RUNNER_IMAGE_MANIFEST == runtime_contract.IMAGE_MANIFEST_DIGEST
    assert live.EXPECTED_RUNNER_IMAGE_CONFIG == runtime_contract.IMAGE_CONFIG_DIGEST
    assert live.EXPECTED_RUNNER_IMAGE_PLATFORM == runtime_contract.IMAGE_PLATFORM
    assert live.EXPECTED_RUNNER_VERSION == runtime_contract.RUNNER_VERSION
    assert live.EXPECTED_RUNNER_COMMIT == runtime_contract.IMAGE_RUNNER_COMMIT
    assert live.EXPECTED_NODE20_VERSION == runtime_contract.IMAGE_NODE20_VERSION
    assert live.EXPECTED_NODE20_SHA256 == runtime_contract.IMAGE_NODE20_SHA256


def test_runtime_frame_matches_fixed_bootstrap_contract_and_destroys_jit() -> None:
    from scripts.release_gpu_jit_lambda_runtime import bootstrap

    canonical_plan = _canonical({"kind": "test-runtime-plan", "ordinal": 1})
    raw_jit = base64.b64encode(b"opaque-jit-material" * 16)
    jit = live.SecretBuffer(raw_jit, label="jit_config")
    read_fd, write_fd = os.pipe()
    receipt = live.write_runtime_frame_and_close(
        write_fd, canonical_plan=canonical_plan, jit_config=jit
    )
    frame = b""
    while True:
        chunk = os.read(read_fd, 4096)
        if not chunk:
            break
        frame += chunk
    os.close(read_fd)
    assert jit.destroyed is True
    assert live.RUNTIME_FRAME_HEADER.size == 84
    unpacked = live.RUNTIME_FRAME_HEADER.unpack(frame[:84])
    assert unpacked[0] == b"EXJIT01\n"
    assert unpacked[1:5] == (1, 0, len(canonical_plan), len(raw_jit))
    assert unpacked[5] == hashlib.sha256(canonical_plan).digest()
    assert unpacked[6] == hashlib.sha256(raw_jit).digest()
    assert frame[:84] == bootstrap.frame_header(canonical_plan, bytearray(raw_jit))
    assert frame[84 : 84 + len(canonical_plan)] == canonical_plan
    assert frame[84 + len(canonical_plan) :] == raw_jit
    assert receipt.to_public_mapping()["remote_argv_contains_plan_or_jit_values"] is False


def test_runtime_frame_rejects_regular_destination_and_noncanonical_plan(
    tmp_path: Path,
) -> None:
    jit = live.SecretBuffer(base64.b64encode(b"x" * 100), label="jit_config")
    destination = tmp_path / "forbidden-frame"
    output_fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with pytest.raises(live.ContractError, match="regular_file_rejected"):
        live.write_runtime_frame_and_close(output_fd, canonical_plan=b'{"z":1}', jit_config=jit)
    assert jit.destroyed is True

    read_fd, write_fd = os.pipe()
    second = live.SecretBuffer(base64.b64encode(b"y" * 100), label="jit_config")
    with pytest.raises(live.ContractError, match="not_canonical"):
        live.write_runtime_frame_and_close(write_fd, canonical_plan=b'{"z": 1}', jit_config=second)
    os.close(read_fd)
    assert second.destroyed is True


def test_runtime_frame_write_timeout_closes_transport_and_destroys_jit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_fd, write_fd = os.pipe()
    jit = live.SecretBuffer(base64.b64encode(b"z" * 100), label="jit_config")
    ticks = iter((0.0, float(live.RUNTIME_FRAME_WRITE_SECONDS + 1)))

    def blocked_write(_: int, __: Any) -> int:
        raise BlockingIOError

    monkeypatch.setattr(live.os, "write", blocked_write)
    monkeypatch.setattr(live.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(live.time, "sleep", lambda _: None)
    with pytest.raises(live.ContractError, match="write_timeout"):
        live.write_runtime_frame_and_close(
            write_fd,
            canonical_plan=_canonical({"kind": "test-runtime-plan"}),
            jit_config=jit,
        )
    os.close(read_fd)
    assert jit.destroyed is True
