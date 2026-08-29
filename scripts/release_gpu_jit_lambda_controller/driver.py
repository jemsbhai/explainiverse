"""Fail-closed live lifecycle driver for one immutable CUDA evidence phase.

The low-level live adapter deliberately exposes one-shot mutations.  This
module supplies the small production sequence around those primitives: it
archives every fresh prestate and public receipt, provisions one exact host,
runs one controller phase, proves repository-runner absence, and restores the
provider firewall state.  It never handles or archives an encoded JIT config.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Mapping, Protocol, Sequence

from scripts.release_gpu_jit_lambda_live import adapter as live
from scripts.release_gpu_jit_lambda_operator import receipt_contract as operator_receipts
from scripts.release_gpu_jit_lambda_runtime import runtime_contract as runtime

from .controller import (
    AUTHORITY_CAPTURE_MAX_AGE,
    OWNER,
    PHASES,
    REPOSITORY,
    AcceptedJobReceipt,
    AmbiguousGitHubMutation,
    AmbiguousRemoteExecution,
    AuthorityReceipt,
    ControllerError,
    DispatchReceipt,
    FinalMainAcceptance,
    HostReadinessReceipt,
    JobBinding,
    PhaseSession,
    RecoveryDispatchReceipt,
    ReleaseGpuController,
    RemoteExecution,
    SealedControllerResources,
    TrustedAppCapture,
    _canonical,
    _iso,
    _json,
    _parse_time,
    _require,
    _sha,
    _validated_authority_evidence_identity,
)


def _is_windows_absolute_path(value: Any) -> bool:
    """Validate an archived Windows path without depending on the verifier OS."""

    return type(value) is str and PureWindowsPath(value).is_absolute()


SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
EVIDENCE_LABEL_RE = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")
JOURNAL_FILENAME_RE = re.compile(r"[0-9]{3}-[a-z][a-z0-9-]{0,63}\.json\Z")
JOURNAL_SHAPED_FILENAME_RE = re.compile(r"[0-9]+-.*\.json\Z")
MAX_JOURNAL_SEQUENCE = 999
OPERATOR_PRELOADER_SHIM_SHA256 = "22bb14f6e5fed4e7c5456f62e11569c9e5a0846ad1428854e545b2e4c1c979aa"
OPERATOR_PYTHON_ARCHIVE_SHA256 = "d1f04d990aee1253d8569e8e5104e30fa9f5fa830899f14843448872d936a2cf"
OPERATOR_PYTHON_MANIFEST_SHA256 = "e2d965a1f8b09d1e5f0349133dfd869eceb92cf730f54a456a4f79bb22d5a519"
OPERATOR_PYTHON_FILE_INVENTORY_SHA256 = (
    "ea028b8d42b0231c116581c4184297900bd4c0152a54017127b822f10b9742d9"
)
OPERATOR_SITE_MANIFEST_SHA256 = "5a6282da0fd87317986b97da1725480c0877686f0e559a83520acf95f46d945f"
OPERATOR_SITE_FILE_INVENTORY_SHA256 = (
    "2cf1cf52ad8d284fcc2e7790acaaa32f3e77a9f39fa717f8bc2a67bc83ba31fe"
)
PINNED_OPERATOR_EXECUTABLES: Mapping[str, Mapping[str, str]] = {
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
        "owner_sid": ("S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464"),
        "authenticode_subject": (
            "CN=Microsoft Windows, O=Microsoft Corporation, " "L=Redmond, S=Washington, C=US"
        ),
        "authenticode_thumbprint": "BAC13DF18B37E808208A39D3A54CCE975FAC8C1D",
    },
}
OPERATOR_SOURCE_MANIFEST_RELATIVE = (
    "scripts/release_gpu_jit_lambda_operator/source-worktree-manifest.json"
)
OPERATOR_PRELOADER_RELATIVE = "scripts/release_gpu_jit_lambda_operator/preloader.py"
OPERATOR_PRELOADER_SHIM_RELATIVE = "scripts/release_gpu_jit_lambda_operator/preloader_shim.py"
OPERATOR_CAPTURE_PREFIXES = (
    "scripts/release_gpu_jit_lambda_controller/",
    "scripts/release_gpu_jit_lambda_live/",
    "scripts/release_gpu_jit_lambda_operator/",
    "scripts/release_gpu_jit_lambda_runtime/",
)
OPERATOR_CAPTURE_EXACT = {
    ".github/release-control-policy.json",
    ".github/workflows/cuda-ci.yml",
    ".github/workflows/publish-pypi.yml",
    ".github/workflows/recover-github-release.yml",
    "poetry.lock",
    "pyproject.toml",
    "scripts/release_external_controls.py",
    "scripts/verify_release_recovery.py",
}
OPERATOR_CRITICAL_SOURCE_PATHS = {
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
    OPERATOR_PRELOADER_RELATIVE,
    OPERATOR_PRELOADER_SHIM_RELATIVE,
    "scripts/release_gpu_jit_lambda_operator/requirements-windows-cp313-bootstrap.txt",
    "scripts/release_gpu_jit_lambda_operator/requirements-windows-cp313.txt",
    "scripts/release_gpu_jit_lambda_operator/site-packages-windows-cp313.json",
    "scripts/release_gpu_jit_lambda_operator/python-runtime-windows-cp313.json",
    OPERATOR_SOURCE_MANIFEST_RELATIVE,
    "scripts/release_gpu_jit_lambda_operator/windows_launcher.py",
    "scripts/release_gpu_jit_lambda_runtime/README.md",
    "scripts/release_gpu_jit_lambda_runtime/__init__.py",
    "scripts/release_gpu_jit_lambda_runtime/bootstrap.py",
    "scripts/release_gpu_jit_lambda_runtime/executor.py",
    "scripts/release_gpu_jit_lambda_runtime/runtime_contract.py",
}
EMERGENCY_EVIDENCE_MAGIC = b"EXEVI01\n"
EMERGENCY_EVIDENCE_COMMIT = b"EXEVCM01"
EMERGENCY_EVIDENCE_SLOT_SIZE = 8192
EMERGENCY_EVIDENCE_SLOT_COUNT = 24
EMERGENCY_EVIDENCE_FILENAME = ".provider-mutation-intents.reserve"
LOCAL_RECOVERY_SIDECAR_RE = re.compile(r"\.local-evidence-recovery-([0-9a-f]{64})\.json\Z")
LOCAL_RECOVERY_SIDECAR_TEMP_RE = re.compile(r"\.local-evidence-recovery-([0-9a-f]{64})\.tmp\Z")
MAX_EVIDENCE_ATOMIC_BYTES = 16 * 1024 * 1024
# At most one journal publish can remain unresolved: `_publish_atomic` never
# returns while its POSIX source hardlink remains.  This cap therefore covers
# one maximum-size journal envelope, one emergency slot, and bounded metadata.
MAX_LOCAL_RECOVERY_SIDECAR_BYTES = 32 * 1024 * 1024


class InstalledAppCaptureSupplier(Protocol):
    """Obtain one action-time raw-page-bound installed-App capture."""

    def __call__(
        self,
    ) -> tuple[Mapping[str, Any], Callable[[str], bytes]]: ...


class ProviderLifecycle(Protocol):
    @property
    def plan_sha256(self) -> str: ...

    @property
    def mutation_intent_binding_sha256(self) -> str | None: ...

    def bind_mutation_intent_callback(self, callback: live.MutationIntentCallback) -> str: ...

    def mutation_intent_callback_matches(self, callback: live.MutationIntentCallback) -> bool: ...

    def ambiguity_from_persisted_intent(self, value: Any) -> live.AmbiguousMutation: ...

    def observe(self, phase: str) -> live.SnapshotReceipt: ...

    def restrict_global(self, receipt: live.SnapshotReceipt) -> live.MutationReceipt: ...

    def create_ruleset(self, receipt: live.SnapshotReceipt) -> live.MutationReceipt: ...

    def launch(
        self,
        receipt: live.SnapshotReceipt,
        identity: live.HostIdentity,
        runtime_bundle: live.RuntimeBundle,
    ) -> live.MutationReceipt: ...

    def terminate(self, receipt: live.SnapshotReceipt) -> live.MutationReceipt: ...

    def delete_ruleset(self, receipt: live.SnapshotReceipt) -> live.MutationReceipt: ...

    def restore_global(self, receipt: live.SnapshotReceipt) -> live.MutationReceipt: ...

    def recover_ambiguous(
        self, ambiguity: live.AmbiguousMutation, receipt: live.SnapshotReceipt
    ) -> live.RecoveryReceipt: ...


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical(value))


def _public_receipt_time(value: Any, context: str) -> datetime:
    """Parse the live adapter's canonical ``+00:00`` public timestamps."""

    _require(type(value) is str, f"{context}_timestamp_rejected")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ControllerError(f"{context}_timestamp_rejected") from None
    _require(
        parsed.tzinfo is not None
        and parsed.utcoffset() == timezone.utc.utcoffset(parsed)
        and value == parsed.isoformat(),
        f"{context}_timestamp_rejected",
    )
    return parsed.astimezone(timezone.utc)


def _reject_secret_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _require(key != "encoded_jit_config", "evidence_contains_encoded_jit_config")
            _reject_secret_keys(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_secret_keys(nested)


@dataclass(frozen=True, init=False)
class PublicationRecoverySource:
    """Loader-sealed publication lifecycle authorized as a recovery source."""

    head_sha: str
    run_id: int
    run_attempt: int
    tag: str
    control_plane_plan_sha256: str
    publication_journal_sha256: str
    evidence_directory_receipt_sha256: str
    job_evidence_sha256: tuple[str, ...]
    phase_settlement_evidence_sha256: str
    evidence_sha256: str
    _recovery_tail: PublicationRecoveryTail | None

    def __new__(cls, *_: object, **__: object) -> PublicationRecoverySource:
        raise TypeError("PublicationRecoverySource must be loaded from a verified journal")

    @classmethod
    def _from_verified(
        cls,
        *,
        head_sha: str,
        run_id: int,
        control_plane_plan_sha256: str,
        publication_journal_sha256: str,
        evidence_directory_receipt_sha256: str,
        job_evidence_sha256: Sequence[str],
        phase_settlement_evidence_sha256: str,
    ) -> PublicationRecoverySource:
        _require(
            re.fullmatch(r"[0-9a-f]{40}", head_sha) is not None
            and type(run_id) is int
            and run_id > 0
            and SHA256_RE.fullmatch(control_plane_plan_sha256) is not None
            and SHA256_RE.fullmatch(publication_journal_sha256) is not None
            and SHA256_RE.fullmatch(evidence_directory_receipt_sha256) is not None
            and len(job_evidence_sha256) == 2
            and all(
                type(value) is str and SHA256_RE.fullmatch(value) is not None
                for value in job_evidence_sha256
            )
            and SHA256_RE.fullmatch(phase_settlement_evidence_sha256) is not None,
            "publication_recovery_source_fields_rejected",
        )
        material = {
            "schema_version": 1,
            "kind": "explainiverse-publication-recovery-source",
            "head_sha": head_sha,
            "run_id": run_id,
            "run_attempt": 1,
            "tag": runtime.PUBLICATION_TAG,
            "control_plane_plan_sha256": control_plane_plan_sha256,
            "publication_journal_sha256": publication_journal_sha256,
            "evidence_directory_receipt_sha256": evidence_directory_receipt_sha256,
            "job_evidence_sha256": list(job_evidence_sha256),
            "phase_settlement_evidence_sha256": phase_settlement_evidence_sha256,
        }
        instance = object.__new__(cls)
        object.__setattr__(instance, "head_sha", head_sha)
        object.__setattr__(instance, "run_id", run_id)
        object.__setattr__(instance, "run_attempt", 1)
        object.__setattr__(instance, "tag", runtime.PUBLICATION_TAG)
        object.__setattr__(
            instance,
            "control_plane_plan_sha256",
            control_plane_plan_sha256,
        )
        object.__setattr__(
            instance,
            "publication_journal_sha256",
            publication_journal_sha256,
        )
        object.__setattr__(
            instance,
            "evidence_directory_receipt_sha256",
            evidence_directory_receipt_sha256,
        )
        object.__setattr__(
            instance,
            "job_evidence_sha256",
            tuple(job_evidence_sha256),
        )
        object.__setattr__(
            instance,
            "phase_settlement_evidence_sha256",
            phase_settlement_evidence_sha256,
        )
        object.__setattr__(instance, "evidence_sha256", _sha(_canonical(material)))
        object.__setattr__(instance, "_recovery_tail", None)
        return instance

    @property
    def recovery_tail(self) -> PublicationRecoveryTail:
        """Return the suffix state verified in the same journal load."""

        _require(
            self._recovery_tail is not None,
            "publication_recovery_tail_not_loader_sealed",
        )
        assert self._recovery_tail is not None
        return self._recovery_tail

    def to_mapping(self) -> dict[str, Any]:
        material = {
            "schema_version": 1,
            "kind": "explainiverse-publication-recovery-source",
            "head_sha": self.head_sha,
            "run_id": self.run_id,
            "run_attempt": self.run_attempt,
            "tag": self.tag,
            "control_plane_plan_sha256": self.control_plane_plan_sha256,
            "publication_journal_sha256": self.publication_journal_sha256,
            "evidence_directory_receipt_sha256": (self.evidence_directory_receipt_sha256),
            "job_evidence_sha256": list(self.job_evidence_sha256),
            "phase_settlement_evidence_sha256": self.phase_settlement_evidence_sha256,
        }
        return {**material, "evidence_sha256": self.evidence_sha256}


@dataclass(frozen=True, init=False)
class PublicationRecoveryTail:
    """Loader-sealed recovery suffix state for no-replay operator decisions."""

    state: str
    source_evidence_sha256: str
    completed_run_ids: tuple[int, ...]
    completed_request_nonces: tuple[str, ...]
    evidence_sha256: str
    _pending_intent_json: bytes | None
    _pending_operator_settlement_json: bytes | None
    _last_operator_settlement_json: bytes | None

    def __new__(cls, *_: object, **__: object) -> PublicationRecoveryTail:
        raise TypeError("PublicationRecoveryTail must be loaded from a verified journal")

    @classmethod
    def _from_verified(
        cls,
        *,
        state: str,
        source_evidence_sha256: str,
        completed_run_ids: Sequence[int],
        completed_request_nonces: Sequence[str],
        pending_intent: Mapping[str, Any] | None,
        pending_operator_settlement: Mapping[str, Any] | None,
        last_operator_settlement: Mapping[str, Any] | None,
    ) -> PublicationRecoveryTail:
        _require(
            state
            in {
                "source-unrecorded",
                "complete",
                "pending-intent",
                "pending-operator-settlement",
            }
            and SHA256_RE.fullmatch(source_evidence_sha256) is not None
            and all(type(item) is int and item > 0 for item in completed_run_ids)
            and len(set(completed_run_ids)) == len(completed_run_ids)
            and all(
                type(item) is str and re.fullmatch(r"[0-9a-f]{16}", item) is not None
                for item in completed_request_nonces
            )
            and len(completed_run_ids) == len(completed_request_nonces)
            and len(set(completed_request_nonces)) == len(completed_request_nonces),
            "publication_recovery_tail_fields_rejected",
        )
        pending_intent_json = _canonical(pending_intent) if pending_intent is not None else None
        pending_operator_json = (
            _canonical(pending_operator_settlement)
            if pending_operator_settlement is not None
            else None
        )
        last_operator_json = (
            _canonical(last_operator_settlement) if last_operator_settlement is not None else None
        )
        _require(
            (state == "pending-intent") == (pending_intent_json is not None)
            and (state == "pending-operator-settlement") == (pending_operator_json is not None)
            and (
                state != "source-unrecorded"
                or (
                    not completed_run_ids
                    and not completed_request_nonces
                    and last_operator_json is None
                )
            ),
            "publication_recovery_tail_state_rejected",
        )
        material = {
            "schema_version": 1,
            "kind": "explainiverse-publication-recovery-tail",
            "state": state,
            "source_evidence_sha256": source_evidence_sha256,
            "completed_run_ids": list(completed_run_ids),
            "completed_request_nonces": list(completed_request_nonces),
            "pending_intent_sha256": (
                _sha(pending_intent_json) if pending_intent_json is not None else None
            ),
            "pending_operator_settlement_sha256": (
                _sha(pending_operator_json) if pending_operator_json is not None else None
            ),
            "last_operator_settlement_sha256": (
                _sha(last_operator_json) if last_operator_json is not None else None
            ),
        }
        instance = object.__new__(cls)
        object.__setattr__(instance, "state", state)
        object.__setattr__(instance, "source_evidence_sha256", source_evidence_sha256)
        object.__setattr__(instance, "completed_run_ids", tuple(completed_run_ids))
        object.__setattr__(
            instance,
            "completed_request_nonces",
            tuple(completed_request_nonces),
        )
        object.__setattr__(instance, "_pending_intent_json", pending_intent_json)
        object.__setattr__(
            instance,
            "_pending_operator_settlement_json",
            pending_operator_json,
        )
        object.__setattr__(
            instance,
            "_last_operator_settlement_json",
            last_operator_json,
        )
        object.__setattr__(instance, "evidence_sha256", _sha(_canonical(material)))
        return instance

    @staticmethod
    def _mapping(raw: bytes | None) -> dict[str, Any] | None:
        if raw is None:
            return None
        value = _json(raw, "publication_recovery_tail_mapping")
        _require(type(value) is dict, "publication_recovery_tail_mapping_not_object")
        assert isinstance(value, dict)
        return value

    @property
    def pending_intent(self) -> dict[str, Any] | None:
        return self._mapping(self._pending_intent_json)

    @property
    def pending_operator_settlement(self) -> dict[str, Any] | None:
        return self._mapping(self._pending_operator_settlement_json)

    @property
    def last_operator_settlement(self) -> dict[str, Any] | None:
        return self._mapping(self._last_operator_settlement_json)

    def to_mapping(self) -> dict[str, Any]:
        material = {
            "schema_version": 1,
            "kind": "explainiverse-publication-recovery-tail",
            "state": self.state,
            "source_evidence_sha256": self.source_evidence_sha256,
            "completed_run_ids": list(self.completed_run_ids),
            "completed_request_nonces": list(self.completed_request_nonces),
            "pending_intent_sha256": (
                _sha(self._pending_intent_json) if self._pending_intent_json is not None else None
            ),
            "pending_operator_settlement_sha256": (
                _sha(self._pending_operator_settlement_json)
                if self._pending_operator_settlement_json is not None
                else None
            ),
            "last_operator_settlement_sha256": (
                _sha(self._last_operator_settlement_json)
                if self._last_operator_settlement_json is not None
                else None
            ),
        }
        return {**material, "evidence_sha256": self.evidence_sha256}


class EvidenceJournal:
    """Exclusive, hash-chained public evidence files in one presecured directory."""

    def __init__(
        self,
        evidence_directory: live.EvidenceDirectoryReceipt,
        *,
        plan_sha256: str,
    ) -> None:
        _require(
            type(evidence_directory) is live.EvidenceDirectoryReceipt,
            "evidence_directory_receipt_type_rejected",
        )
        try:
            evidence_directory.validate()
            root = Path(evidence_directory.absolute_path)
            _require(root.is_absolute(), "evidence_directory_not_absolute")
            _require(
                root == root.resolve(strict=True),
                "evidence_directory_not_canonical",
            )
            _require(
                root.is_dir() and not root.is_symlink(),
                "evidence_directory_rejected",
            )
            _require(
                SHA256_RE.fullmatch(plan_sha256) is not None,
                "journal_plan_sha256_rejected",
            )
            _require(
                SHA256_RE.fullmatch(evidence_directory.receipt_sha256) is not None,
                "journal_acl_receipt_rejected",
            )
            _require(not self._journal_paths(root), "evidence_journal_not_empty")
            _require(
                not any(root.glob(".evidence-*.tmp")),
                "evidence_journal_temporary_present",
            )
        except BaseException:
            try:
                evidence_directory.close()
            except BaseException:
                pass
            raise
        self._directory = root
        self._plan_sha256 = plan_sha256
        self._acl_sha256 = evidence_directory.receipt_sha256
        self._evidence_directory = evidence_directory
        self._sequence = 0
        self._previous_sha256: str | None = None
        self._emergency_path = root / EMERGENCY_EVIDENCE_FILENAME
        self._emergency_fd = -1
        try:
            self._emergency_fd = self._open_recovery_reserve_exclusive(
                self._emergency_path,
                create_new=True,
            )
        except BaseException:
            try:
                self._evidence_directory.close()
            except BaseException:
                pass
            raise
        reserve_size = EMERGENCY_EVIDENCE_SLOT_SIZE * EMERGENCY_EVIDENCE_SLOT_COUNT
        try:
            remaining = reserve_size
            zero_block = b"\0" * min(65536, reserve_size)
            while remaining:
                written = os.write(self._emergency_fd, zero_block[:remaining])
                _require(written > 0, "emergency_evidence_reserve_short_write")
                remaining -= written
            os.fsync(self._emergency_fd)
        except BaseException:
            descriptor = self._emergency_fd
            self._emergency_fd = -1
            try:
                os.close(descriptor)
            except BaseException:
                pass
            try:
                self._emergency_path.unlink()
            except BaseException:
                pass
            try:
                self._evidence_directory.close()
            except BaseException:
                pass
            raise
        self._emergency_count = 0
        self._emergency_previous_sha256: str | None = None
        self._interrupted_publish_recovery: dict[str, Any] | None = None
        try:
            self.record("evidence-directory", evidence_directory.to_public_mapping())
        except BaseException:
            descriptor = self._emergency_fd
            self._emergency_fd = -1
            try:
                os.close(descriptor)
            except BaseException:
                pass
            try:
                self._evidence_directory.close()
            except BaseException:
                pass
            # Preserve the reserve and any atomic-publish source/final exactly
            # as observed. Explicit recovery or abandonment, never constructor
            # cleanup, determines their disposition.
            raise

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def acl_receipt_sha256(self) -> str:
        self._evidence_directory.validate()
        return self._acl_sha256

    @property
    def evidence_directory_receipt(self) -> live.EvidenceDirectoryReceipt:
        self._evidence_directory.validate()
        return self._evidence_directory

    @property
    def last_evidence_sha256(self) -> str | None:
        return self._previous_sha256

    @staticmethod
    def _validate_directory_evidence_mapping(
        value: Mapping[str, Any], receipt: live.EvidenceDirectoryReceipt
    ) -> None:
        current = receipt.to_public_mapping()
        supplied_acl = value.get("acl")
        current_acl = current.get("acl")
        _require(
            set(value) == set(current)
            and value.get("receipt_sha256") == receipt.receipt_sha256
            and value.get("absolute_path_redacted") is True
            and value.get("directory_identity_recorded") is True
            and value.get("no_reparse_or_symlink") is True
            and value.get("owner_private") is True
            and type(supplied_acl) is dict
            and type(current_acl) is dict
            and {key: item for key, item in supplied_acl.items() if key != "captured_at"}
            == {key: item for key, item in current_acl.items() if key != "captured_at"},
            "evidence_directory_journal_binding_rejected",
        )

    @classmethod
    def reopen_for_recovery(
        cls,
        evidence_directory: live.EvidenceDirectoryReceipt,
        *,
        plan_sha256: str,
    ) -> EvidenceJournal:
        """Verify and append to an interrupted owner-protected journal."""

        _require(
            type(evidence_directory) is live.EvidenceDirectoryReceipt,
            "recovery_evidence_directory_receipt_type_rejected",
        )
        evidence_directory.validate()
        root = Path(evidence_directory.absolute_path)
        _require(root.is_absolute(), "recovery_journal_not_absolute")
        _require(root == root.resolve(strict=True), "recovery_journal_not_canonical")
        _require(root.is_dir() and not root.is_symlink(), "recovery_journal_rejected")
        _require(SHA256_RE.fullmatch(plan_sha256) is not None, "recovery_plan_sha_rejected")
        _require(
            SHA256_RE.fullmatch(evidence_directory.receipt_sha256) is not None,
            "recovery_acl_sha_rejected",
        )
        # Every independent recovery precondition is checked before the
        # interrupted journal directory entry is linked, renamed, or unlinked.
        # A corrupt reserve or mismatched evidence-directory receipt must leave
        # the complete tree byte-for-byte untouched for external inspection.
        reserve_path = root / EMERGENCY_EVIDENCE_FILENAME
        expected_reserve_size = EMERGENCY_EVIDENCE_SLOT_SIZE * EMERGENCY_EVIDENCE_SLOT_COUNT
        _require(
            reserve_path.is_file()
            and not reserve_path.is_symlink()
            and reserve_path.stat().st_nlink == 1
            and reserve_path.stat().st_size == expected_reserve_size,
            "recovery_emergency_evidence_reserve_rejected",
        )
        paths_before_recovery = cls._journal_paths(root)
        if paths_before_recovery:
            first_raw = paths_before_recovery[0].read_bytes()
            first_envelope_before = _json(
                first_raw,
                "recovery_directory_evidence_precondition",
            )
            _require(
                type(first_envelope_before) is dict
                and first_raw == _canonical(first_envelope_before)
                and first_envelope_before.get("label") == "evidence-directory"
                and type(first_envelope_before.get("payload")) is dict,
                "recovery_directory_evidence_missing",
            )
            cls._validate_directory_evidence_mapping(
                first_envelope_before["payload"],
                evidence_directory,
            )
        else:
            initial_temporaries = sorted(root.glob(".evidence-*.tmp"))
            _require(
                len(initial_temporaries) == 1,
                "recovery_directory_evidence_missing",
            )
            initial_raw = initial_temporaries[0].read_bytes()
            initial_envelope = _json(
                initial_raw,
                "recovery_initial_directory_evidence_precondition",
            )
            _require(
                type(initial_envelope) is dict
                and initial_raw == _canonical(initial_envelope)
                and type(initial_envelope.get("sequence")) is int
                and initial_envelope.get("sequence") == 1
                and initial_envelope.get("label") == "evidence-directory"
                and type(initial_envelope.get("payload")) is dict,
                "recovery_directory_evidence_missing",
            )
            cls._validate_directory_evidence_mapping(
                initial_envelope["payload"],
                evidence_directory,
            )
        recovery_reserve_fd = cls._open_recovery_reserve_exclusive(reserve_path)
        try:
            _require(
                os.path.samestat(os.fstat(recovery_reserve_fd), reserve_path.stat()),
                "recovery_emergency_evidence_reserve_identity_rejected",
            )
            os.lseek(recovery_reserve_fd, 0, os.SEEK_SET)
            reserve_raw = bytearray()
            while len(reserve_raw) < expected_reserve_size:
                chunk = os.read(
                    recovery_reserve_fd,
                    expected_reserve_size - len(reserve_raw),
                )
                _require(bool(chunk), "recovery_emergency_evidence_reserve_short_read")
                reserve_raw.extend(chunk)
            interrupted_publish_recovery = cls._recover_interrupted_journal_publish(
                root,
                plan_sha256=plan_sha256,
                acl_sha256=evidence_directory.receipt_sha256,
                reserve_path=reserve_path,
                reserve_fd=recovery_reserve_fd,
                reserve_raw=bytes(reserve_raw),
            )
            chain, previous_sha256 = cls._validate_journal_chain(
                root,
                plan_sha256=plan_sha256,
                acl_sha256=evidence_directory.receipt_sha256,
                context="recovery",
            )
            _require(bool(chain), "recovery_journal_empty")
            paths = [path for path, _, _ in chain]
            instance = object.__new__(cls)
            instance._directory = root
            instance._plan_sha256 = plan_sha256
            instance._acl_sha256 = evidence_directory.receipt_sha256
            instance._evidence_directory = evidence_directory
            instance._sequence = len(paths)
            instance._previous_sha256 = previous_sha256
            instance._interrupted_publish_recovery = interrupted_publish_recovery
            instance._emergency_path = reserve_path
            _require(
                instance._emergency_path.is_file()
                and not instance._emergency_path.is_symlink()
                and instance._emergency_path.stat().st_nlink == 1
                and instance._emergency_path.stat().st_size
                == EMERGENCY_EVIDENCE_SLOT_SIZE * EMERGENCY_EVIDENCE_SLOT_COUNT
                and os.path.samestat(
                    os.fstat(recovery_reserve_fd), instance._emergency_path.stat()
                ),
                "recovery_emergency_evidence_reserve_rejected",
            )
            instance._emergency_fd = recovery_reserve_fd
            emergency = instance._read_emergency_provider_intents()
            instance._emergency_count = len(emergency)
            if emergency:
                digest_offset = (
                    (len(emergency) - 1) * EMERGENCY_EVIDENCE_SLOT_SIZE
                    + len(EMERGENCY_EVIDENCE_MAGIC)
                    + 4
                )
                os.lseek(instance._emergency_fd, digest_offset, os.SEEK_SET)
                digest_bytes = os.read(instance._emergency_fd, 32)
                _require(
                    len(digest_bytes) == 32,
                    "recovery_emergency_evidence_tail_rejected",
                )
                instance._emergency_previous_sha256 = digest_bytes.hex()
            else:
                instance._emergency_previous_sha256 = None
            first_envelope = chain[0][2]
            _require(
                first_envelope.get("label") == "evidence-directory"
                and type(first_envelope.get("payload")) is dict,
                "recovery_directory_evidence_missing",
            )
            cls._validate_directory_evidence_mapping(first_envelope["payload"], evidence_directory)
            return instance
        except BaseException:
            os.close(recovery_reserve_fd)
            raise

    @property
    def interrupted_publish_recovery(self) -> dict[str, Any] | None:
        """Return the exact interrupted publish preserved during explicit reopen."""

        if self._interrupted_publish_recovery is None:
            return None
        return _json_copy(self._interrupted_publish_recovery)

    def record_interrupted_publish_recovery(self) -> str | None:
        """Durably archive one explicit-reopen classification exactly once.

        The retained private sidecar already makes the destructive repair
        crash-resumable.  This public numbered record binds that sidecar into
        the journal before a caller may perform any GitHub observation or
        mutation.
        """

        mapping = self._interrupted_publish_recovery
        if mapping is None:
            return None
        rows = mapping.get("recovered_entries")
        _require(
            type(rows) is list
            and (bool(rows) or type(mapping.get("emergency_uncommitted_slot")) is dict),
            "interrupted_publish_recovery_mapping_rejected",
        )
        sidecar_path = self._directory / str(mapping.get("sidecar_filename"))
        _require(
            sidecar_path.is_file()
            and not sidecar_path.is_symlink()
            and sidecar_path.stat().st_nlink == 1,
            "interrupted_publish_recovery_sidecar_rejected",
        )
        sidecar_raw = sidecar_path.read_bytes()
        sidecar = _json(sidecar_raw, "interrupted_publish_recovery_sidecar")
        _require(
            type(sidecar) is dict
            and self._local_recovery_public_mapping(
                sidecar_path,
                sidecar_raw,
                sidecar,
            )
            == mapping,
            "interrupted_publish_recovery_sidecar_binding_rejected",
        )
        self.require_capacity(1)
        digest = self.record("journal-publish-recovery", mapping)
        self._interrupted_publish_recovery = None
        return digest

    @staticmethod
    def _write_all(fd: int, payload: bytes) -> None:
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            _require(written > 0, "evidence_short_write")
            offset += written

    @staticmethod
    def _is_storage_durability_failure(error: OSError) -> bool:
        """Recognize only failures safe for the preallocated intent reserve."""

        return error.errno in {
            errno.EIO,
            errno.ENOSPC,
            getattr(errno, "EDQUOT", -1),
        } or getattr(error, "winerror", None) in {
            23,  # ERROR_CRC
            29,  # ERROR_WRITE_FAULT
            39,  # ERROR_HANDLE_DISK_FULL
            112,  # ERROR_DISK_FULL
            1117,  # ERROR_IO_DEVICE
        }

    @staticmethod
    def _read_exact_descriptor(fd: int, expected_bytes: int) -> bytes:
        _require(
            type(fd) is int and fd >= 0 and type(expected_bytes) is int and expected_bytes >= 0,
            "evidence_descriptor_read_input_rejected",
        )
        os.lseek(fd, 0, os.SEEK_SET)
        result = bytearray()
        while len(result) < expected_bytes:
            chunk = os.read(fd, expected_bytes - len(result))
            _require(bool(chunk), "evidence_descriptor_short_read")
            result.extend(chunk)
        _require(not os.read(fd, 1), "evidence_descriptor_trailing_bytes")
        return bytes(result)

    @staticmethod
    def _open_windows_atomic_file(path: Path, *, create_new: bool) -> int:
        """Open one file with write/delete sharing denied until publication."""

        _require(os.name == "nt" and path.is_absolute(), "windows_atomic_open_rejected")
        import ctypes
        import msvcrt
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        create_file.restype = wintypes.HANDLE
        raw_handle = create_file(
            str(path),
            0x80000000 | 0x40000000 | 0x00010000,
            # GENERIC_READ | GENERIC_WRITE | DELETE
            0x00000001,  # FILE_SHARE_READ; write and delete are denied.
            None,
            1 if create_new else 3,  # CREATE_NEW | OPEN_EXISTING
            0x00200000 | 0x80000000,  # OPEN_REPARSE_POINT | WRITE_THROUGH
            None,
        )
        invalid_handle = ctypes.c_void_p(-1).value
        if raw_handle in {None, invalid_handle}:
            raise OSError(
                ctypes.get_last_error(),
                "exclusive Windows atomic file open failed",
            )
        try:
            return msvcrt.open_osfhandle(
                int(raw_handle),
                os.O_RDWR | getattr(os, "O_BINARY", 0),
            )
        except BaseException:
            close_handle = kernel32.CloseHandle
            close_handle.argtypes = [wintypes.HANDLE]
            close_handle.restype = wintypes.BOOL
            close_handle(raw_handle)
            raise

    @classmethod
    def _publish_windows_held_file(
        cls,
        descriptor: int,
        *,
        temporary: Path,
        destination: Path,
        payload: bytes,
        context: str,
    ) -> None:
        """Rename an identity-sealed handle and verify it before releasing it."""

        _require(
            os.name == "nt"
            and type(descriptor) is int
            and descriptor >= 0
            and temporary.parent == destination.parent
            and temporary.parent.is_dir()
            and not destination.exists(),
            f"{context}_windows_publish_input_rejected",
        )
        import ctypes
        import msvcrt
        from ctypes import wintypes

        before_descriptor = os.fstat(descriptor)
        before_path = temporary.stat()
        _require(
            temporary.is_file()
            and not temporary.is_symlink()
            and temporary.resolve(strict=True) == temporary
            and temporary.parent == destination.parent
            and before_descriptor.st_nlink == 1
            and before_path.st_nlink == 1
            and before_descriptor.st_size == len(payload)
            and os.path.samestat(before_descriptor, before_path)
            and cls._read_exact_descriptor(descriptor, len(payload)) == payload,
            f"{context}_windows_source_rejected",
        )

        destination_text = str(destination)

        class FileRenameInfo(ctypes.Structure):
            _fields_ = [
                ("ReplaceIfExists", wintypes.BOOLEAN),
                ("RootDirectory", wintypes.HANDLE),
                ("FileNameLength", wintypes.DWORD),
                ("FileName", wintypes.WCHAR * (len(destination_text) + 1)),
            ]

        rename = FileRenameInfo()
        rename.ReplaceIfExists = 0
        rename.RootDirectory = None
        rename.FileNameLength = len(destination_text.encode("utf-16-le"))
        rename.FileName = destination_text
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        set_information = kernel32.SetFileInformationByHandle
        set_information.argtypes = [
            wintypes.HANDLE,
            wintypes.INT,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        set_information.restype = wintypes.BOOL
        moved = set_information(
            msvcrt.get_osfhandle(descriptor),
            3,  # FileRenameInfo; ReplaceIfExists is false.
            ctypes.byref(rename),
            ctypes.sizeof(rename),
        )
        if not moved:
            raise OSError(
                ctypes.get_last_error(),
                "SetFileInformationByHandle(FileRenameInfo) failed",
            )
        os.fsync(descriptor)
        after_descriptor = os.fstat(descriptor)
        after_path = destination.stat()
        _require(
            not temporary.exists()
            and destination.is_file()
            and not destination.is_symlink()
            and destination.resolve(strict=True) == destination
            and after_descriptor.st_nlink == 1
            and after_path.st_nlink == 1
            and after_descriptor.st_size == len(payload)
            and os.path.samestat(before_descriptor, after_descriptor)
            and os.path.samestat(after_descriptor, after_path)
            and cls._read_exact_descriptor(descriptor, len(payload)) == payload,
            f"{context}_windows_destination_rejected",
        )

    @staticmethod
    def _open_recovery_reserve_exclusive(
        path: Path,
        *,
        create_new: bool = False,
    ) -> int:
        """Hold the reserve against concurrent write/delete through recovery."""

        if os.name != "nt":
            descriptor = os.open(
                path,
                os.O_RDWR
                | getattr(os, "O_BINARY", 0)
                | (os.O_CREAT | os.O_EXCL if create_new else 0),
                0o600,
            )
            try:
                import fcntl

                flock = getattr(fcntl, "flock")
                lock_ex = int(getattr(fcntl, "LOCK_EX"))
                lock_nb = int(getattr(fcntl, "LOCK_NB"))
                flock(descriptor, lock_ex | lock_nb)
            except BaseException:
                os.close(descriptor)
                raise
            return descriptor

        import ctypes
        import msvcrt
        from ctypes import wintypes

        create_file = ctypes.WinDLL("kernel32", use_last_error=True).CreateFileW
        create_file.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        create_file.restype = wintypes.HANDLE
        raw_handle = create_file(
            str(path),
            0x80000000 | 0x40000000,  # GENERIC_READ | GENERIC_WRITE
            0x00000001,  # FILE_SHARE_READ; write and delete are denied.
            None,
            1 if create_new else 3,  # CREATE_NEW | OPEN_EXISTING
            0x00200000 | 0x80000000,  # OPEN_REPARSE_POINT | WRITE_THROUGH
            None,
        )
        invalid_handle = ctypes.c_void_p(-1).value
        if raw_handle in {None, invalid_handle}:
            raise OSError(ctypes.get_last_error(), "exclusive recovery reserve open failed")
        try:
            return msvcrt.open_osfhandle(
                int(raw_handle),
                os.O_RDWR | getattr(os, "O_BINARY", 0),
            )
        except BaseException:
            close_handle = ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle
            close_handle.argtypes = [wintypes.HANDLE]
            close_handle.restype = wintypes.BOOL
            close_handle(raw_handle)
            raise

    @staticmethod
    def _journal_paths(root: Path) -> list[Path]:
        """Return the complete bounded journal namespace or reject residue."""

        paths: list[Path] = []
        malformed: list[str] = []
        for item in root.iterdir():
            name = item.name
            if JOURNAL_FILENAME_RE.fullmatch(name) is not None:
                paths.append(item)
            elif JOURNAL_SHAPED_FILENAME_RE.fullmatch(name) is not None:
                malformed.append(name)
        _require(not malformed, "journal_filename_namespace_rejected")
        _require(
            len(paths) <= MAX_JOURNAL_SEQUENCE,
            "journal_entry_cardinality_rejected",
        )
        return sorted(paths)

    @classmethod
    def _validate_journal_chain(
        cls,
        root: Path,
        *,
        plan_sha256: str,
        acl_sha256: str,
        context: str,
        allowed_hardlinks: Mapping[Path, Path] | None = None,
    ) -> tuple[list[tuple[Path, bytes, dict[str, Any]]], str | None]:
        """Strictly validate the complete current numbered chain without mutation."""

        hardlinks = dict(allowed_hardlinks or {})
        paths = cls._journal_paths(root)
        result: list[tuple[Path, bytes, dict[str, Any]]] = []
        previous_sha256: str | None = None
        for sequence, path in enumerate(paths, start=1):
            source = hardlinks.get(path)
            expected_links = 2 if source is not None else 1
            _require(
                path.is_file()
                and not path.is_symlink()
                and path.resolve(strict=True) == path
                and path.parent == root
                and path.stat().st_nlink == expected_links,
                f"{context}_journal_entry_rejected",
            )
            if source is not None:
                _require(
                    source.is_file()
                    and not source.is_symlink()
                    and source.resolve(strict=True) == source
                    and source.parent == root
                    and source.stat().st_nlink == 2
                    and os.path.samestat(path.stat(), source.stat()),
                    f"{context}_journal_hardlink_rejected",
                )
            raw = path.read_bytes()
            envelope = _json(raw, f"{context}_journal_entry")
            _require(
                type(envelope) is dict
                and raw == _canonical(envelope)
                and set(envelope)
                == {
                    "schema_version",
                    "kind",
                    "sequence",
                    "label",
                    "control_plane_plan_sha256",
                    "evidence_directory_acl_receipt_sha256",
                    "previous_evidence_sha256",
                    "payload",
                }
                and type(envelope["schema_version"]) is int
                and envelope["schema_version"] == 1
                and envelope["kind"] == "explainiverse-lambda-live-driver-evidence"
                and type(envelope["sequence"]) is int
                and envelope["sequence"] == sequence
                and type(envelope["label"]) is str
                and EVIDENCE_LABEL_RE.fullmatch(envelope["label"]) is not None
                and envelope["control_plane_plan_sha256"] == plan_sha256
                and envelope["evidence_directory_acl_receipt_sha256"] == acl_sha256
                and envelope["previous_evidence_sha256"] == previous_sha256
                and type(envelope["payload"]) is dict
                and path.name == f"{sequence:03d}-{envelope['label']}.json",
                f"{context}_journal_chain_rejected",
            )
            _reject_secret_keys(envelope["payload"])
            previous_sha256 = _sha(raw)
            result.append((path, raw, envelope))
        _require(
            set(hardlinks).issubset({path for path, _, _ in result}),
            f"{context}_journal_hardlink_destination_rejected",
        )
        return result, previous_sha256

    @staticmethod
    def _sync_directory(directory: Path) -> None:
        if os.name == "nt":
            # Windows publication holds a WRITE_THROUGH file handle through
            # SetFileInformationByHandle and flushes that same handle after
            # the rename; opening a directory for fsync is not portable there.
            return
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @classmethod
    def _load_local_recovery_sidecars(
        cls,
        directory: Path,
    ) -> list[tuple[Path, bytes, dict[str, Any]]]:
        """Read every durable local-recovery classification without mutation."""

        result: list[tuple[Path, bytes, dict[str, Any]]] = []
        malformed = [
            path.name
            for path in directory.iterdir()
            if path.name.startswith(".local-evidence-recovery-")
            and LOCAL_RECOVERY_SIDECAR_RE.fullmatch(path.name) is None
            and LOCAL_RECOVERY_SIDECAR_TEMP_RE.fullmatch(path.name) is None
        ]
        _require(not malformed, "local_recovery_sidecar_namespace_rejected")
        sidecar_paths = [
            path
            for path in sorted(directory.iterdir())
            if LOCAL_RECOVERY_SIDECAR_RE.fullmatch(path.name) is not None
        ]
        _require(
            len(sidecar_paths) <= MAX_JOURNAL_SEQUENCE,
            "local_recovery_sidecar_cardinality_rejected",
        )
        for path in sidecar_paths:
            match = LOCAL_RECOVERY_SIDECAR_RE.fullmatch(path.name)
            assert match is not None
            temporary = directory / f".local-evidence-recovery-{match.group(1)}.tmp"
            link_count = path.stat().st_nlink
            _require(
                path.is_file()
                and not path.is_symlink()
                and path.resolve(strict=True) == path
                and path.parent == directory
                and link_count in {1, 2}
                and 0 < path.stat().st_size <= MAX_LOCAL_RECOVERY_SIDECAR_BYTES,
                "local_recovery_sidecar_file_rejected",
            )
            raw = path.read_bytes()
            value = _json(raw, "local_recovery_sidecar")
            _require(
                type(value) is dict and raw == _canonical(value) and _sha(raw) == match.group(1),
                "local_recovery_sidecar_digest_rejected",
            )
            if link_count == 2:
                _require(
                    os.name != "nt"
                    and temporary.is_file()
                    and not temporary.is_symlink()
                    and temporary.resolve(strict=True) == temporary
                    and temporary.parent == directory
                    and temporary.stat().st_nlink == 2
                    and os.path.samestat(path.stat(), temporary.stat())
                    and temporary.read_bytes() == raw,
                    "local_recovery_sidecar_published_temp_rejected",
                )
            result.append((path, raw, value))
        return result

    @staticmethod
    def _local_recovery_sidecar_temporaries(directory: Path) -> list[Path]:
        """Return the complete exact sidecar-temp namespace without mutation."""

        result = [
            path
            for path in sorted(directory.iterdir())
            if LOCAL_RECOVERY_SIDECAR_TEMP_RE.fullmatch(path.name) is not None
        ]
        _require(
            len(result) <= 1,
            "local_recovery_sidecar_temp_cardinality_rejected",
        )
        for path in result:
            _require(
                path.is_file()
                and not path.is_symlink()
                and path.resolve(strict=True) == path
                and path.parent == directory
                and path.stat().st_nlink in {1, 2}
                and path.stat().st_size <= MAX_LOCAL_RECOVERY_SIDECAR_BYTES,
                "local_recovery_sidecar_temp_file_rejected",
            )
        return result

    @classmethod
    def _publish_local_recovery_sidecar(
        cls,
        directory: Path,
        value: Mapping[str, Any],
    ) -> tuple[Path, bytes]:
        """Durably publish a deterministic sidecar before source repair.

        A crash while writing this sidecar cannot have changed the journal or
        reserve.  Therefore a retained partial sidecar is retryable only when
        it is an exact byte prefix of the mapping reconstructed from those
        still-untouched sources.
        """

        raw = _canonical(value)
        _require(
            0 < len(raw) <= MAX_LOCAL_RECOVERY_SIDECAR_BYTES,
            "local_recovery_sidecar_size_rejected",
        )
        digest = _sha(raw)
        final = directory / f".local-evidence-recovery-{digest}.json"
        temporary = directory / f".local-evidence-recovery-{digest}.tmp"
        temporary_namespace = cls._local_recovery_sidecar_temporaries(directory)
        _require(
            not temporary_namespace or temporary_namespace == [temporary],
            "local_recovery_sidecar_foreign_temp_rejected",
        )
        if final.exists():
            _require(
                final.is_file()
                and not final.is_symlink()
                and final.resolve(strict=True) == final
                and final.parent == directory
                and final.read_bytes() == raw,
                "local_recovery_sidecar_existing_drift",
            )
            if temporary.exists():
                _require(
                    os.name != "nt"
                    and temporary.is_file()
                    and not temporary.is_symlink()
                    and temporary.stat().st_nlink == 2
                    and final.stat().st_nlink == 2
                    and os.path.samestat(temporary.stat(), final.stat())
                    and temporary.read_bytes() == raw,
                    "local_recovery_sidecar_published_temp_rejected",
                )
                temporary.unlink()
                cls._sync_directory(directory)
            _require(final.stat().st_nlink == 1, "local_recovery_sidecar_link_rejected")
            return final, raw
        if os.name == "nt":
            descriptor = cls._open_windows_atomic_file(
                temporary,
                create_new=not temporary.exists(),
            )
            try:
                current_size = os.fstat(descriptor).st_size
                current = cls._read_exact_descriptor(descriptor, current_size)
                _require(
                    temporary.is_file()
                    and not temporary.is_symlink()
                    and temporary.resolve(strict=True) == temporary
                    and temporary.parent == directory
                    and temporary.stat().st_nlink == 1
                    and os.path.samestat(os.fstat(descriptor), temporary.stat())
                    and 0 <= current_size <= len(raw)
                    and raw.startswith(current),
                    "local_recovery_sidecar_partial_rejected",
                )
                os.lseek(descriptor, current_size, os.SEEK_SET)
                cls._write_all(descriptor, raw[current_size:])
                os.fsync(descriptor)
                cls._publish_windows_held_file(
                    descriptor,
                    temporary=temporary,
                    destination=final,
                    payload=raw,
                    context="local_recovery_sidecar",
                )
            finally:
                os.close(descriptor)
        else:
            if temporary.exists():
                _require(
                    temporary.is_file()
                    and not temporary.is_symlink()
                    and temporary.resolve(strict=True) == temporary
                    and temporary.parent == directory
                    and temporary.stat().st_nlink == 1
                    and 0 <= temporary.stat().st_size <= len(raw)
                    and raw.startswith(temporary.read_bytes()),
                    "local_recovery_sidecar_partial_rejected",
                )
                temporary.unlink()
                cls._sync_directory(directory)
            descriptor = os.open(
                temporary,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0),
                0o600,
            )
            try:
                cls._write_all(descriptor, raw)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.link(temporary, final)
            cls._sync_directory(directory)
            temporary.unlink()
            cls._sync_directory(directory)
        _require(
            final.is_file()
            and not final.is_symlink()
            and final.stat().st_nlink == 1
            and final.read_bytes() == raw,
            "local_recovery_sidecar_publish_rejected",
        )
        return final, raw

    @classmethod
    def _local_recovery_public_mapping(
        cls,
        sidecar_path: Path,
        sidecar_raw: bytes,
        sidecar: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Exact-validate a private sidecar and derive its public binding."""

        _require(
            set(sidecar)
            == {
                "schema_version",
                "kind",
                "control_plane_plan_sha256",
                "evidence_directory_acl_receipt_sha256",
                "journal_temporaries",
                "emergency_uncommitted_slot",
            }
            and type(sidecar["schema_version"]) is int
            and sidecar["schema_version"] == 1
            and sidecar["kind"] == "explainiverse-local-evidence-recovery-classification"
            and type(sidecar["control_plane_plan_sha256"]) is str
            and SHA256_RE.fullmatch(sidecar["control_plane_plan_sha256"]) is not None
            and type(sidecar["evidence_directory_acl_receipt_sha256"]) is str
            and SHA256_RE.fullmatch(sidecar["evidence_directory_acl_receipt_sha256"]) is not None
            and type(sidecar["journal_temporaries"]) is list
            and type(sidecar["emergency_uncommitted_slot"]) in {dict, type(None)}
            and (
                bool(sidecar["journal_temporaries"])
                or sidecar["emergency_uncommitted_slot"] is not None
            ),
            "local_recovery_sidecar_schema_rejected",
        )
        recovered_entries: list[dict[str, Any]] = []
        private_names: set[str] = set()
        final_names: set[str] = set()
        previous_private_sequence = 0
        for row in sidecar["journal_temporaries"]:
            _require(
                type(row) is dict
                and set(row)
                == {
                    "temporary_filename",
                    "temporary_bytes",
                    "temporary_sha256",
                    "initial_link_count",
                    "envelope",
                }
                and type(row["temporary_filename"]) is str
                and re.fullmatch(r"\.evidence-[0-9a-f]{32}\.tmp", row["temporary_filename"])
                is not None
                and row["temporary_filename"] not in private_names
                and type(row["temporary_bytes"]) is int
                and 0 < row["temporary_bytes"] <= MAX_EVIDENCE_ATOMIC_BYTES
                and type(row["temporary_sha256"]) is str
                and SHA256_RE.fullmatch(row["temporary_sha256"]) is not None
                and type(row["initial_link_count"]) is int
                and row["initial_link_count"] in {1, 2}
                and type(row["envelope"]) is dict
                and type(row["envelope"].get("sequence")) is int
                and row["envelope"]["sequence"] > previous_private_sequence,
                "local_recovery_sidecar_journal_row_rejected",
            )
            envelope = row["envelope"]
            envelope_raw = _canonical(envelope)
            _require(
                len(envelope_raw) == row["temporary_bytes"]
                and _sha(envelope_raw) == row["temporary_sha256"]
                and set(envelope)
                == {
                    "schema_version",
                    "kind",
                    "sequence",
                    "label",
                    "control_plane_plan_sha256",
                    "evidence_directory_acl_receipt_sha256",
                    "previous_evidence_sha256",
                    "payload",
                }
                and type(envelope["schema_version"]) is int
                and envelope["schema_version"] == 1
                and envelope["kind"] == "explainiverse-lambda-live-driver-evidence"
                and type(envelope["sequence"]) is int
                and 1 <= envelope["sequence"] <= MAX_JOURNAL_SEQUENCE
                and type(envelope["label"]) is str
                and EVIDENCE_LABEL_RE.fullmatch(envelope["label"]) is not None
                and envelope["control_plane_plan_sha256"] == sidecar["control_plane_plan_sha256"]
                and envelope["evidence_directory_acl_receipt_sha256"]
                == sidecar["evidence_directory_acl_receipt_sha256"]
                and type(envelope["payload"]) is dict,
                "local_recovery_sidecar_envelope_rejected",
            )
            _reject_secret_keys(envelope["payload"])
            final_filename = f"{envelope['sequence']:03d}-{envelope['label']}.json"
            _require(
                final_filename not in final_names,
                "local_recovery_sidecar_final_reused",
            )
            private_names.add(str(row["temporary_filename"]))
            final_names.add(final_filename)
            previous_private_sequence = int(envelope["sequence"])
            recovered_entries.append(
                {
                    "classification": (
                        "complete-unpublished-envelope"
                        if row["initial_link_count"] == 1
                        else "published-hardlink-envelope"
                    ),
                    "temporary_filename": row["temporary_filename"],
                    "temporary_bytes": row["temporary_bytes"],
                    "temporary_sha256": row["temporary_sha256"],
                    "sequence": envelope["sequence"],
                    "label": envelope["label"],
                    "previous_evidence_sha256": envelope["previous_evidence_sha256"],
                    "final_filename": final_filename,
                }
            )
        recovered_entries.sort(
            key=lambda item: (int(item["sequence"]), str(item["temporary_filename"]))
        )
        sidecar_sha256 = _sha(sidecar_raw)
        _require(
            sidecar_path.name == f".local-evidence-recovery-{sidecar_sha256}.json",
            "local_recovery_sidecar_filename_rejected",
        )
        emergency = sidecar["emergency_uncommitted_slot"]
        emergency_public: dict[str, Any] | None = None
        if emergency is not None:
            _require(
                type(emergency) is dict
                and set(emergency)
                == {
                    "slot_index",
                    "slot_bytes",
                    "slot_sha256",
                    "slot_hex",
                }
                and type(emergency["slot_index"]) is int
                and 0 <= emergency["slot_index"] < EMERGENCY_EVIDENCE_SLOT_COUNT
                and type(emergency["slot_bytes"]) is int
                and emergency["slot_bytes"] == EMERGENCY_EVIDENCE_SLOT_SIZE
                and type(emergency["slot_sha256"]) is str
                and SHA256_RE.fullmatch(emergency["slot_sha256"]) is not None
                and type(emergency["slot_hex"]) is str
                and len(emergency["slot_hex"]) == EMERGENCY_EVIDENCE_SLOT_SIZE * 2,
                "local_recovery_sidecar_emergency_rejected",
            )
            try:
                emergency_raw = bytes.fromhex(emergency["slot_hex"])
            except ValueError:
                raise ControllerError("local_recovery_sidecar_emergency_rejected") from None
            _require(
                _sha(emergency_raw) == emergency["slot_sha256"],
                "local_recovery_sidecar_emergency_rejected",
            )
            emergency_public = {
                "classification": "complete-uncommitted-provider-intent-slot",
                "reserve_filename": EMERGENCY_EVIDENCE_FILENAME,
                "slot_index": emergency["slot_index"],
                "slot_bytes": emergency["slot_bytes"],
                "slot_sha256": emergency["slot_sha256"],
            }
        public_material = {
            "schema_version": 1,
            "kind": "explainiverse-interrupted-local-evidence-recovery",
            "control_plane_plan_sha256": sidecar["control_plane_plan_sha256"],
            "evidence_directory_acl_receipt_sha256": sidecar[
                "evidence_directory_acl_receipt_sha256"
            ],
            "sidecar_filename": sidecar_path.name,
            "sidecar_bytes": len(sidecar_raw),
            "sidecar_sha256": sidecar_sha256,
            "recovered_entries": recovered_entries,
            "emergency_uncommitted_slot": emergency_public,
        }
        return {
            **public_material,
            "recovery_evidence_sha256": _sha(_canonical(public_material)),
        }

    @classmethod
    def _recover_interrupted_journal_publish(
        cls,
        directory: Path,
        *,
        plan_sha256: str,
        acl_sha256: str,
        reserve_path: Path,
        reserve_fd: int,
        reserve_raw: bytes,
    ) -> dict[str, Any] | None:
        """Durably classify, then idempotently normalize interrupted local writes."""

        _require(
            reserve_path.parent == directory
            and reserve_path.name == EMERGENCY_EVIDENCE_FILENAME
            and type(reserve_fd) is int
            and reserve_fd >= 0
            and reserve_path.is_file()
            and not reserve_path.is_symlink()
            and reserve_path.stat().st_nlink == 1
            and os.path.samestat(os.fstat(reserve_fd), reserve_path.stat())
            and os.fstat(reserve_fd).st_size
            == EMERGENCY_EVIDENCE_SLOT_SIZE * EMERGENCY_EVIDENCE_SLOT_COUNT
            and len(reserve_raw) == EMERGENCY_EVIDENCE_SLOT_SIZE * EMERGENCY_EVIDENCE_SLOT_COUNT,
            "local_recovery_reserve_input_rejected",
        )

        def envelope_destination(envelope: Mapping[str, Any]) -> Path:
            return directory / f"{envelope['sequence']:03d}-{envelope['label']}.json"

        def publish_source(
            temporary: Path,
            destination: Path,
            payload: bytes,
        ) -> None:
            _require(not destination.exists(), "local_recovery_destination_exists")
            if os.name == "nt":
                descriptor = cls._open_windows_atomic_file(
                    temporary,
                    create_new=False,
                )
                try:
                    cls._publish_windows_held_file(
                        descriptor,
                        temporary=temporary,
                        destination=destination,
                        payload=payload,
                        context="local_recovery_journal",
                    )
                finally:
                    os.close(descriptor)
            else:
                os.link(temporary, destination)
                cls._sync_directory(directory)
                temporary.unlink()
                cls._sync_directory(directory)

        def emergency_original(
            sidecar: Mapping[str, Any], current: bytes
        ) -> tuple[bytes, int | None]:
            emergency = sidecar["emergency_uncommitted_slot"]
            if emergency is None:
                _, index = cls._parse_emergency_provider_intents(
                    current,
                    plan_sha256=plan_sha256,
                    acl_sha256=acl_sha256,
                )
                _require(index is None, "local_recovery_unclassified_emergency_slot")
                return current, None
            assert isinstance(emergency, dict)
            original = bytes.fromhex(str(emergency["slot_hex"]))
            index = int(emergency["slot_index"])
            start = index * EMERGENCY_EVIDENCE_SLOT_SIZE
            current_slot = current[start : start + EMERGENCY_EVIDENCE_SLOT_SIZE]
            differences = [
                offset
                for offset, (left, right) in enumerate(zip(current_slot, original))
                if left != right
            ]
            boundary = differences[-1] + 1 if differences else 0
            _require(
                current_slot[:boundary] == b"\0" * boundary
                and current_slot[boundary:] == original[boundary:],
                "local_recovery_emergency_zero_progress_rejected",
            )
            reconstructed = bytearray(current)
            reconstructed[start : start + EMERGENCY_EVIDENCE_SLOT_SIZE] = original
            _, reconstructed_index = cls._parse_emergency_provider_intents(
                bytes(reconstructed),
                plan_sha256=plan_sha256,
                acl_sha256=acl_sha256,
            )
            _require(
                reconstructed_index == index,
                "local_recovery_emergency_sidecar_binding_rejected",
            )
            return bytes(reconstructed), index

        def read_reserve_descriptor() -> bytes:
            current_path_stat = reserve_path.stat()
            descriptor_stat = os.fstat(reserve_fd)
            expected_size = EMERGENCY_EVIDENCE_SLOT_SIZE * EMERGENCY_EVIDENCE_SLOT_COUNT
            _require(
                reserve_path.is_file()
                and not reserve_path.is_symlink()
                and current_path_stat.st_nlink == 1
                and descriptor_stat.st_nlink == 1
                and descriptor_stat.st_size == expected_size
                and os.path.samestat(descriptor_stat, current_path_stat),
                "local_recovery_reserve_identity_rejected",
            )
            os.lseek(reserve_fd, 0, os.SEEK_SET)
            current = bytearray()
            while len(current) < expected_size:
                chunk = os.read(reserve_fd, expected_size - len(current))
                _require(bool(chunk), "local_recovery_reserve_short_read")
                current.extend(chunk)
            return bytes(current)

        sidecars = cls._load_local_recovery_sidecars(directory)
        sidecar_temporaries = cls._local_recovery_sidecar_temporaries(directory)
        classified_sidecars: list[tuple[Path, bytes, dict[str, Any], dict[str, Any]]] = []
        for sidecar_path, sidecar_raw, sidecar in sidecars:
            _require(
                sidecar.get("control_plane_plan_sha256") == plan_sha256
                and sidecar.get("evidence_directory_acl_receipt_sha256") == acl_sha256,
                "local_recovery_sidecar_context_rejected",
            )
            classified_sidecars.append(
                (
                    sidecar_path,
                    sidecar_raw,
                    sidecar,
                    cls._local_recovery_public_mapping(sidecar_path, sidecar_raw, sidecar),
                )
            )

        evidence_temporaries = sorted(directory.glob(".evidence-*.tmp"))
        _require(
            len(evidence_temporaries) <= 1,
            "evidence_temporary_cardinality_rejected",
        )
        temporary_rows: dict[str, tuple[Path, bytes, dict[str, Any], Path, int]] = {}
        allowed_hardlinks: dict[Path, Path] = {}
        for temporary in evidence_temporaries:
            _require(
                re.fullmatch(r"\.evidence-[0-9a-f]{32}\.tmp", temporary.name) is not None
                and temporary.is_file()
                and not temporary.is_symlink()
                and temporary.resolve(strict=True) == temporary
                and temporary.parent == directory
                and 0 <= temporary.stat().st_size <= MAX_EVIDENCE_ATOMIC_BYTES
                and temporary.stat().st_nlink in {1, 2},
                "evidence_temporary_repair_rejected",
            )
            raw = temporary.read_bytes()
            try:
                envelope_value = _json(raw, "interrupted_journal_publish")
            except ControllerError:
                # A partial recovery-record temp is validated below from its
                # already-durable sidecar.  Every other partial temp is fatal.
                continue
            if type(envelope_value) is not dict or raw != _canonical(envelope_value):
                # A syntactically complete prefix (for example, exact JSON
                # missing only the final canonical LF) is still a partial
                # recovery-record candidate. The pending sidecar must bind it
                # byte-for-byte below; without that sidecar it remains fatal.
                continue
            envelope = envelope_value
            _require(
                set(envelope)
                == {
                    "schema_version",
                    "kind",
                    "sequence",
                    "label",
                    "control_plane_plan_sha256",
                    "evidence_directory_acl_receipt_sha256",
                    "previous_evidence_sha256",
                    "payload",
                }
                and type(envelope["schema_version"]) is int
                and envelope["schema_version"] == 1
                and envelope["kind"] == "explainiverse-lambda-live-driver-evidence"
                and type(envelope["sequence"]) is int
                and 1 <= envelope["sequence"] <= MAX_JOURNAL_SEQUENCE
                and type(envelope["label"]) is str
                and EVIDENCE_LABEL_RE.fullmatch(envelope["label"]) is not None
                and envelope["control_plane_plan_sha256"] == plan_sha256
                and envelope["evidence_directory_acl_receipt_sha256"] == acl_sha256
                and type(envelope["payload"]) is dict,
                "evidence_temporary_envelope_rejected",
            )
            _reject_secret_keys(envelope["payload"])
            destination = envelope_destination(envelope)
            link_count = temporary.stat().st_nlink
            if link_count == 2:
                _require(
                    os.name != "nt"
                    and destination.is_file()
                    and not destination.is_symlink()
                    and destination.stat().st_nlink == 2
                    and os.path.samestat(temporary.stat(), destination.stat())
                    and destination.read_bytes() == raw
                    and destination not in allowed_hardlinks,
                    "evidence_temporary_final_binding_rejected",
                )
                allowed_hardlinks[destination] = temporary
            temporary_rows[temporary.name] = (
                temporary,
                raw,
                envelope,
                destination,
                link_count,
            )

        chain, chain_sha256 = cls._validate_journal_chain(
            directory,
            plan_sha256=plan_sha256,
            acl_sha256=acl_sha256,
            context="local_recovery_precondition",
            allowed_hardlinks=allowed_hardlinks,
        )
        journal_recovery_payloads = [
            envelope["payload"]
            for _, _, envelope in chain
            if envelope["label"] == "journal-publish-recovery"
        ]
        pending = [
            item
            for item in classified_sidecars
            if not any(item[3] == payload for payload in journal_recovery_payloads)
        ]
        _require(
            len(pending) <= 1,
            "local_recovery_pending_sidecar_cardinality_rejected",
        )
        for _, _, _, public in classified_sidecars:
            _require(
                sum(public == payload for payload in journal_recovery_payloads) <= 1,
                "local_recovery_sidecar_receipt_cardinality_rejected",
            )

        completed_source_names: set[str] = set()
        for _, _, sidecar, public in classified_sidecars:
            if not any(public == payload for payload in journal_recovery_payloads):
                continue
            for row in sidecar["journal_temporaries"]:
                temporary = directory / str(row["temporary_filename"])
                final = envelope_destination(row["envelope"])
                _require(
                    final.is_file()
                    and not final.is_symlink()
                    and final.read_bytes() == _canonical(row["envelope"]),
                    "local_recovery_completed_final_rejected",
                )
                if temporary.exists():
                    _require(
                        os.name != "nt"
                        and temporary.stat().st_nlink == 2
                        and final.stat().st_nlink == 2
                        and os.path.samestat(temporary.stat(), final.stat()),
                        "local_recovery_completed_temp_rejected",
                    )
                    completed_source_names.add(temporary.name)
        for temporary_name, (
            temporary,
            raw,
            envelope,
            destination,
            link_count,
        ) in temporary_rows.items():
            if not (
                link_count == 2
                and envelope["label"] == "journal-publish-recovery"
                and any(
                    envelope["payload"] == public
                    and any(public == payload for payload in journal_recovery_payloads)
                    for _, _, _, public in classified_sidecars
                )
            ):
                continue
            _require(
                os.name != "nt"
                and destination.is_file()
                and destination.read_bytes() == raw
                and destination in {path for path, _, _ in chain}
                and temporary.stat().st_nlink == 2
                and destination.stat().st_nlink == 2
                and os.path.samestat(temporary.stat(), destination.stat()),
                "local_recovery_completed_record_temp_rejected",
            )
            completed_source_names.add(temporary_name)

        if pending:
            pending_path, _, pending_sidecar, pending_public = pending[0]
            pending_digest = pending_path.name.removeprefix(
                ".local-evidence-recovery-"
            ).removesuffix(".json")
            _require(
                not sidecar_temporaries
                or (
                    len(sidecar_temporaries) == 1
                    and sidecar_temporaries[0].name
                    == f".local-evidence-recovery-{pending_digest}.tmp"
                    and os.name != "nt"
                    and sidecar_temporaries[0].stat().st_nlink == 2
                    and pending_path.stat().st_nlink == 2
                    and os.path.samestat(sidecar_temporaries[0].stat(), pending_path.stat())
                ),
                "local_recovery_pending_sidecar_temp_rejected",
            )
            original_names = {
                str(row["temporary_filename"]) for row in pending_sidecar["journal_temporaries"]
            }
            extras = [
                path
                for path in evidence_temporaries
                if path.name not in original_names and path.name not in completed_source_names
            ]
            _require(
                len(extras) <= 1,
                "local_recovery_recursive_temp_cardinality_rejected",
            )
            missing_rows: list[Mapping[str, Any]] = []
            for row in pending_sidecar["journal_temporaries"]:
                envelope = row["envelope"]
                raw = _canonical(envelope)
                temporary = directory / str(row["temporary_filename"])
                final = envelope_destination(envelope)
                if final.exists():
                    _require(
                        final.read_bytes() == raw and final in {path for path, _, _ in chain},
                        "local_recovery_source_final_rejected",
                    )
                    if temporary.exists():
                        _require(
                            temporary.read_bytes() == raw
                            and temporary.stat().st_nlink == 2
                            and final.stat().st_nlink == 2
                            and os.path.samestat(temporary.stat(), final.stat()),
                            "local_recovery_source_hardlink_rejected",
                        )
                else:
                    _require(
                        row["initial_link_count"] == 1
                        and temporary.is_file()
                        and not temporary.is_symlink()
                        and temporary.stat().st_nlink == 1
                        and temporary.read_bytes() == raw,
                        "local_recovery_source_temp_rejected",
                    )
                    missing_rows.append(row)
            _require(
                len(missing_rows) <= 1,
                "local_recovery_source_tail_cardinality_rejected",
            )
            if missing_rows:
                missing_envelope = missing_rows[0]["envelope"]
                _require(
                    missing_envelope["sequence"] == len(chain) + 1
                    and missing_envelope["previous_evidence_sha256"] == chain_sha256,
                    "local_recovery_source_tail_binding_rejected",
                )
            final_count = len(chain) + len(missing_rows)
            _require(
                final_count + 1 <= MAX_JOURNAL_SEQUENCE,
                "local_recovery_journal_capacity_rejected",
            )
            emergency_original(pending_sidecar, reserve_raw)
            expected_previous = (
                _sha(_canonical(missing_rows[0]["envelope"])) if missing_rows else chain_sha256
            )
            expected_record = {
                "schema_version": 1,
                "kind": "explainiverse-lambda-live-driver-evidence",
                "sequence": final_count + 1,
                "label": "journal-publish-recovery",
                "control_plane_plan_sha256": plan_sha256,
                "evidence_directory_acl_receipt_sha256": acl_sha256,
                "previous_evidence_sha256": expected_previous,
                "payload": pending_public,
            }
            expected_record_raw = _canonical(expected_record)
            if extras:
                extra = extras[0]
                extra_raw = extra.read_bytes()
                _require(
                    extra.stat().st_nlink == 1
                    and len(extra_raw) <= len(expected_record_raw)
                    and expected_record_raw.startswith(extra_raw),
                    "local_recovery_recursive_temp_rejected",
                )
            active_sidecar = pending_sidecar
            active_public = pending_public
        else:
            parsed_reserve, uncommitted_index = cls._parse_emergency_provider_intents(
                reserve_raw,
                plan_sha256=plan_sha256,
                acl_sha256=acl_sha256,
            )
            del parsed_reserve
            fresh_rows = [
                item for name, item in temporary_rows.items() if name not in completed_source_names
            ]
            _require(
                all(
                    path.name in temporary_rows or path.name in completed_source_names
                    for path in evidence_temporaries
                ),
                "evidence_temporary_not_canonical",
            )
            missing = [item for item in fresh_rows if item[4] == 1]
            _require(
                len(missing) <= 1,
                "evidence_temporary_sequence_rejected",
            )
            for _, raw, envelope, destination, link_count in fresh_rows:
                if link_count == 1:
                    _require(
                        not destination.exists()
                        and envelope["sequence"] == len(chain) + 1
                        and envelope["previous_evidence_sha256"] == chain_sha256
                        and envelope["label"] != "journal-publish-recovery",
                        "evidence_temporary_sequence_rejected",
                    )
                else:
                    _require(
                        destination in {path for path, _, _ in chain}
                        and destination.read_bytes() == raw,
                        "evidence_temporary_final_binding_rejected",
                    )
            has_recovery = bool(fresh_rows) or uncommitted_index is not None
            if not has_recovery:
                _require(
                    not sidecar_temporaries,
                    "local_recovery_orphan_sidecar_temp_rejected",
                )
                _require(
                    read_reserve_descriptor() == reserve_raw,
                    "local_recovery_reserve_changed_before_cleanup",
                )
                # All validation is complete; only now remove exact hardlinks
                # left from already journal-bound recovery transactions.
                for name in completed_source_names:
                    (directory / name).unlink()
                    cls._sync_directory(directory)
                for sidecar_path, _, _, public in classified_sidecars:
                    sidecar_temp = directory / sidecar_path.name.replace(".json", ".tmp")
                    if sidecar_temp.exists():
                        _require(
                            any(public == payload for payload in journal_recovery_payloads),
                            "local_recovery_sidecar_temp_unbound",
                        )
                        sidecar_temp.unlink()
                        cls._sync_directory(directory)
                return None
            _require(
                len(chain) + len(missing) + 1 <= MAX_JOURNAL_SEQUENCE,
                "local_recovery_journal_capacity_rejected",
            )
            emergency_private: dict[str, Any] | None = None
            if uncommitted_index is not None:
                start = uncommitted_index * EMERGENCY_EVIDENCE_SLOT_SIZE
                slot = reserve_raw[start : start + EMERGENCY_EVIDENCE_SLOT_SIZE]
                emergency_private = {
                    "slot_index": uncommitted_index,
                    "slot_bytes": len(slot),
                    "slot_sha256": _sha(slot),
                    "slot_hex": slot.hex(),
                }
            sidecar_value = {
                "schema_version": 1,
                "kind": "explainiverse-local-evidence-recovery-classification",
                "control_plane_plan_sha256": plan_sha256,
                "evidence_directory_acl_receipt_sha256": acl_sha256,
                "journal_temporaries": [
                    {
                        "temporary_filename": temporary.name,
                        "temporary_bytes": len(raw),
                        "temporary_sha256": _sha(raw),
                        "initial_link_count": link_count,
                        "envelope": envelope,
                    }
                    for temporary, raw, envelope, _, link_count in fresh_rows
                ],
                "emergency_uncommitted_slot": emergency_private,
            }
            sidecar_value_raw = _canonical(sidecar_value)
            _require(
                0 < len(sidecar_value_raw) <= MAX_LOCAL_RECOVERY_SIDECAR_BYTES,
                "local_recovery_sidecar_size_rejected",
            )
            expected_sidecar_temp = directory / (
                ".local-evidence-recovery-" + _sha(sidecar_value_raw) + ".tmp"
            )
            _require(
                not sidecar_temporaries
                or (
                    sidecar_temporaries == [expected_sidecar_temp]
                    and expected_sidecar_temp.stat().st_nlink == 1
                    and expected_sidecar_temp.stat().st_size <= len(sidecar_value_raw)
                    and sidecar_value_raw.startswith(expected_sidecar_temp.read_bytes())
                )
                or any(
                    sidecar_temporaries == [directory / path.name.replace(".json", ".tmp")]
                    and path.stat().st_nlink == 2
                    and sidecar_temporaries[0].stat().st_nlink == 2
                    and os.path.samestat(path.stat(), sidecar_temporaries[0].stat())
                    for path, _, _, public in classified_sidecars
                    if any(public == payload for payload in journal_recovery_payloads)
                ),
                "local_recovery_sidecar_foreign_temp_rejected",
            )
            # A completed sidecar receipt already authorizes cleanup of its
            # exact POSIX publication links.  All current sources, chain,
            # reserve, capacity, and the new classification were proved above.
            for name in completed_source_names:
                path = directory / name
                if path.exists():
                    path.unlink()
                    cls._sync_directory(directory)
            for sidecar_path, _, _, public in classified_sidecars:
                if not any(public == payload for payload in journal_recovery_payloads):
                    continue
                sidecar_temp = directory / sidecar_path.name.replace(".json", ".tmp")
                if sidecar_temp.exists():
                    sidecar_temp.unlink()
                    cls._sync_directory(directory)
            sidecar_path, sidecar_raw = cls._publish_local_recovery_sidecar(
                directory,
                sidecar_value,
            )
            active_sidecar = sidecar_value
            active_public = cls._local_recovery_public_mapping(
                sidecar_path,
                sidecar_raw,
                sidecar_value,
            )
            extras = []

        # From this point onward the exact private classification is durable.
        # Every mutation is monotonic and can be resumed from the same sidecar.
        current_reserve = read_reserve_descriptor()
        if active_sidecar["emergency_uncommitted_slot"] is None:
            _require(
                current_reserve == reserve_raw,
                "local_recovery_reserve_changed_before_repair",
            )
            emergency_original(active_sidecar, current_reserve)
        else:
            emergency_original(active_sidecar, current_reserve)
        for row in active_sidecar["journal_temporaries"]:
            envelope = row["envelope"]
            raw = _canonical(envelope)
            temporary = directory / str(row["temporary_filename"])
            destination = envelope_destination(envelope)
            if not destination.exists():
                _require(temporary.exists(), "local_recovery_source_temp_missing")
                publish_source(temporary, destination, raw)
            _require(
                destination.is_file()
                and not destination.is_symlink()
                and destination.read_bytes() == raw,
                "local_recovery_final_publish_rejected",
            )
            if temporary.exists():
                _require(
                    os.name != "nt"
                    and temporary.stat().st_nlink == 2
                    and destination.stat().st_nlink == 2
                    and os.path.samestat(temporary.stat(), destination.stat()),
                    "local_recovery_source_cleanup_rejected",
                )
                temporary.unlink()
                cls._sync_directory(directory)
            _require(
                destination.stat().st_nlink == 1,
                "local_recovery_final_link_rejected",
            )
        emergency = active_sidecar["emergency_uncommitted_slot"]
        if emergency is not None:
            assert isinstance(emergency, dict)
            # Re-read through the same identity-bound descriptor immediately
            # before the destructive zero. A concurrent byte change or path
            # replacement is preserved and fails closed.
            emergency_original(active_sidecar, read_reserve_descriptor())
            os.lseek(
                reserve_fd,
                int(emergency["slot_index"]) * EMERGENCY_EVIDENCE_SLOT_SIZE,
                os.SEEK_SET,
            )
            cls._write_all(reserve_fd, b"\0" * EMERGENCY_EVIDENCE_SLOT_SIZE)
            os.fsync(reserve_fd)
        recovery_record_completed = False
        for extra in extras:
            record_destination = envelope_destination(expected_record)
            if os.name == "nt":
                descriptor = cls._open_windows_atomic_file(
                    extra,
                    create_new=False,
                )
                try:
                    current_size = os.fstat(descriptor).st_size
                    current = cls._read_exact_descriptor(descriptor, current_size)
                    _require(
                        extra.is_file()
                        and not extra.is_symlink()
                        and extra.stat().st_nlink == 1
                        and os.path.samestat(os.fstat(descriptor), extra.stat())
                        and current_size <= len(expected_record_raw)
                        and expected_record_raw.startswith(current),
                        "local_recovery_recursive_temp_rejected",
                    )
                    os.lseek(descriptor, current_size, os.SEEK_SET)
                    cls._write_all(descriptor, expected_record_raw[current_size:])
                    os.fsync(descriptor)
                    cls._publish_windows_held_file(
                        descriptor,
                        temporary=extra,
                        destination=record_destination,
                        payload=expected_record_raw,
                        context="local_recovery_record",
                    )
                finally:
                    os.close(descriptor)
            else:
                descriptor = os.open(
                    extra,
                    os.O_RDWR | getattr(os, "O_BINARY", 0),
                )
                try:
                    current_size = os.fstat(descriptor).st_size
                    current = cls._read_exact_descriptor(descriptor, current_size)
                    _require(
                        extra.is_file()
                        and not extra.is_symlink()
                        and extra.stat().st_nlink == 1
                        and os.path.samestat(os.fstat(descriptor), extra.stat())
                        and current_size <= len(expected_record_raw)
                        and expected_record_raw.startswith(current),
                        "local_recovery_recursive_temp_rejected",
                    )
                    os.lseek(descriptor, current_size, os.SEEK_SET)
                    cls._write_all(descriptor, expected_record_raw[current_size:])
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                os.link(extra, record_destination)
                cls._sync_directory(directory)
                extra.unlink()
                cls._sync_directory(directory)
            recovery_record_completed = True
        for name in completed_source_names:
            path = directory / name
            if path.exists():
                path.unlink()
                cls._sync_directory(directory)
        for sidecar_path, _, _, public in classified_sidecars:
            sidecar_temp = directory / sidecar_path.name.replace(".json", ".tmp")
            if sidecar_temp.exists():
                _require(
                    public == active_public
                    or any(public == payload for payload in journal_recovery_payloads),
                    "local_recovery_sidecar_temp_unbound",
                )
                sidecar_temp.unlink()
                cls._sync_directory(directory)
        final_chain, _ = cls._validate_journal_chain(
            directory,
            plan_sha256=plan_sha256,
            acl_sha256=acl_sha256,
            context="local_recovery_postcondition",
        )
        _require(bool(final_chain), "local_recovery_postcondition_empty")
        final_reserve = read_reserve_descriptor()
        _, final_uncommitted = cls._parse_emergency_provider_intents(
            final_reserve,
            plan_sha256=plan_sha256,
            acl_sha256=acl_sha256,
        )
        _require(
            final_uncommitted is None,
            "local_recovery_emergency_postcondition_rejected",
        )
        return None if recovery_record_completed else active_public

    def _publish_atomic(self, destination: Path, payload: bytes) -> None:
        _require(not destination.exists(), "evidence_destination_exists")
        _require(
            type(payload) is bytes and len(payload) <= MAX_EVIDENCE_ATOMIC_BYTES,
            "evidence_atomic_payload_too_large",
        )
        _require(
            destination.parent.is_dir()
            and not destination.parent.is_symlink()
            and destination.parent.resolve(strict=True) == destination.parent,
            "evidence_destination_parent_rejected",
        )
        _require(
            not any(destination.parent.glob(".evidence-*.tmp")),
            "evidence_unresolved_atomic_publish_present",
        )
        temporary = destination.parent / f".evidence-{secrets.token_hex(16)}.tmp"
        descriptor = (
            self._open_windows_atomic_file(temporary, create_new=True)
            if os.name == "nt"
            else os.open(
                temporary,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0),
                0o600,
            )
        )
        published = False
        linked = False
        source_durable = False
        try:
            self._write_all(descriptor, payload)
            os.fsync(descriptor)
            source_durable = True
            _require(not destination.exists(), "evidence_destination_raced")
            if os.name == "nt":
                self._publish_windows_held_file(
                    descriptor,
                    temporary=temporary,
                    destination=destination,
                    payload=payload,
                    context="evidence_atomic",
                )
            else:
                os.close(descriptor)
                descriptor = -1
                os.link(temporary, destination)
                linked = True
                temporary.unlink()
            published = True
            self._sync_directory(destination.parent)
            if descriptor >= 0:
                os.close(descriptor)
                descriptor = -1
        except BaseException as error:
            if descriptor >= 0:
                os.close(descriptor)
            if not source_durable:
                try:
                    temporary.unlink()
                except OSError:
                    pass
            if published:
                _require(
                    destination.is_file() and destination.read_bytes() == payload,
                    "evidence_atomic_publish_ambiguous",
                )
                # The bytes exist but directory durability was not proven.
                # A second emergency copy would leave two authoritative WAL
                # channels for one request, so explicit recovery is required.
                raise ControllerError("evidence_atomic_publish_durability_ambiguous") from error
            if linked:
                _require(
                    destination.is_file() and destination.read_bytes() == payload,
                    "evidence_hardlink_publish_ambiguous",
                )
                try:
                    temporary.unlink()
                except OSError:
                    pass
                self._sync_directory(destination.parent)
                if not temporary.exists() and destination.stat().st_nlink == 1:
                    return
                # The numbered entry is durable, but its source link remains.
                # Do not let the caller proceed to an external mutation and
                # accumulate another unresolved publish. Explicit recovery
                # classifies and removes this one bound link.
                raise ControllerError("evidence_hardlink_cleanup_incomplete")
            if source_durable:
                _require(
                    temporary.is_file()
                    and not temporary.is_symlink()
                    and temporary.stat().st_nlink == 1
                    and temporary.read_bytes() == payload,
                    "evidence_atomic_durable_source_drift",
                )
                raise ControllerError("evidence_atomic_publication_requires_recovery") from error
            if isinstance(error, OSError):
                _require(
                    not destination.exists() and not temporary.exists(),
                    "evidence_atomic_storage_failure_left_residue",
                )
                if self._is_storage_durability_failure(error):
                    raise
                raise ControllerError("evidence_atomic_os_error_rejected") from error
            raise

    def record(self, label: str, payload: Mapping[str, Any]) -> str:
        self._evidence_directory.validate()
        _require(EVIDENCE_LABEL_RE.fullmatch(label) is not None, "evidence_label_rejected")
        _reject_secret_keys(payload)
        self.require_capacity(6 if label == "github-recovery-dispatch-intent" else 1)
        next_sequence = self._sequence + 1
        material = {
            "schema_version": 1,
            "kind": "explainiverse-lambda-live-driver-evidence",
            "sequence": next_sequence,
            "label": label,
            "control_plane_plan_sha256": self._plan_sha256,
            "evidence_directory_acl_receipt_sha256": self._acl_sha256,
            "previous_evidence_sha256": self._previous_sha256,
            "payload": _json_copy(payload),
        }
        destination = self._directory / f"{next_sequence:03d}-{label}.json"
        raw = _canonical(material)
        _require(
            len(raw) <= MAX_EVIDENCE_ATOMIC_BYTES,
            "evidence_atomic_payload_too_large",
        )
        self._publish_atomic(destination, raw)
        digest = _sha(raw)
        self._sequence = next_sequence
        self._previous_sha256 = digest
        return digest

    def require_capacity(self, entries: int) -> None:
        """Prove an entire upcoming write-ahead transaction fits the journal."""

        self._evidence_directory.validate()
        _require(
            type(entries) is int
            and entries > 0
            and self._sequence + entries <= MAX_JOURNAL_SEQUENCE,
            "journal_sequence_capacity_exhausted",
        )

    def record_provider_mutation_intent(self, intent: live.MutationIntent) -> None:
        _require(
            type(intent) is live.MutationIntent,
            "provider_mutation_intent_type_rejected",
        )
        mapping = intent.to_public_mapping()
        # Capacity and schema errors are policy failures, not durability
        # failures. They must stop the provider callback before contact.
        self.require_capacity(1)
        try:
            self.record("provider-mutation-intent", mapping)
        except OSError as error:
            _require(
                self._is_storage_durability_failure(error),
                "provider_mutation_journal_os_error_rejected",
            )
            # The preallocated, already-open reserve remains writable through
            # storage I/O failure and does not need new blocks during ENOSPC.
            # The provider contacts its API only after this method returns.
            self._record_emergency_provider_intent(mapping)

    def _record_emergency_provider_intent(self, intent: Mapping[str, Any]) -> None:
        self._evidence_directory.validate()
        _require(
            self._emergency_count < EMERGENCY_EVIDENCE_SLOT_COUNT,
            "emergency_evidence_reserve_exhausted",
        )
        normalized = live.MutationIntent.from_public_mapping(intent)
        material = {
            "schema_version": 1,
            "kind": "explainiverse-provider-mutation-emergency-evidence",
            "sequence": self._emergency_count + 1,
            "control_plane_plan_sha256": self._plan_sha256,
            "evidence_directory_acl_receipt_sha256": self._acl_sha256,
            "previous_evidence_sha256": self._emergency_previous_sha256,
            "intent": normalized.to_public_mapping(),
        }
        payload = _canonical(material)
        digest = _sha(payload)
        header_size = len(EMERGENCY_EVIDENCE_MAGIC) + 4 + 32
        usable_size = EMERGENCY_EVIDENCE_SLOT_SIZE - len(EMERGENCY_EVIDENCE_COMMIT)
        _require(
            len(payload) <= usable_size - header_size,
            "emergency_evidence_payload_too_large",
        )
        uncommitted_slot = (
            EMERGENCY_EVIDENCE_MAGIC
            + len(payload).to_bytes(4, "big")
            + bytes.fromhex(digest)
            + payload
        )
        uncommitted_slot += b"\0" * (EMERGENCY_EVIDENCE_SLOT_SIZE - len(uncommitted_slot))
        slot_offset = self._emergency_count * EMERGENCY_EVIDENCE_SLOT_SIZE
        os.lseek(
            self._emergency_fd,
            slot_offset,
            os.SEEK_SET,
        )
        # Commit the slot only after every byte of its content is durable.  A
        # process death before the final marker therefore leaves an
        # unambiguously unpublished slot that recovery may zero without
        # discarding an intent whose callback could have returned.
        self._write_all(self._emergency_fd, uncommitted_slot)
        os.fsync(self._emergency_fd)
        os.lseek(
            self._emergency_fd,
            slot_offset + EMERGENCY_EVIDENCE_SLOT_SIZE - len(EMERGENCY_EVIDENCE_COMMIT),
            os.SEEK_SET,
        )
        self._write_all(self._emergency_fd, EMERGENCY_EVIDENCE_COMMIT)
        os.fsync(self._emergency_fd)
        self._emergency_count += 1
        self._emergency_previous_sha256 = digest

    @staticmethod
    def _parse_emergency_provider_intents(
        raw: bytes,
        *,
        plan_sha256: str,
        acl_sha256: str,
    ) -> tuple[tuple[dict[str, Any], ...], int | None]:
        """Validate the entire preallocated reserve without mutating it."""

        expected_size = EMERGENCY_EVIDENCE_SLOT_SIZE * EMERGENCY_EVIDENCE_SLOT_COUNT
        _require(len(raw) == expected_size, "emergency_evidence_reserve_short_read")
        result: list[dict[str, Any]] = []
        previous_sha256: str | None = None
        unused_seen = False
        uncommitted_index: int | None = None
        header_size = len(EMERGENCY_EVIDENCE_MAGIC) + 4 + 32
        usable_size = EMERGENCY_EVIDENCE_SLOT_SIZE - len(EMERGENCY_EVIDENCE_COMMIT)
        for index in range(EMERGENCY_EVIDENCE_SLOT_COUNT):
            start = index * EMERGENCY_EVIDENCE_SLOT_SIZE
            slot = bytes(raw[start : start + EMERGENCY_EVIDENCE_SLOT_SIZE])
            if slot == b"\0" * EMERGENCY_EVIDENCE_SLOT_SIZE:
                unused_seen = True
                continue
            _require(not unused_seen, "emergency_evidence_slot_gap")
            if not slot.endswith(EMERGENCY_EVIDENCE_COMMIT):
                _require(
                    all(byte == 0 for byte in raw[start + EMERGENCY_EVIDENCE_SLOT_SIZE :]),
                    "emergency_evidence_uncommitted_slot_not_tail",
                )
                # A recoverable pre-callback crash has completed and fsynced
                # the entire staged slot; its sole missing byte transition is
                # the final commit marker.  Partial prefixes and arbitrary
                # nonzero residue remain untouched and fail closed.
                _require(
                    slot.startswith(EMERGENCY_EVIDENCE_MAGIC)
                    and slot[usable_size:] == b"\0" * len(EMERGENCY_EVIDENCE_COMMIT),
                    "emergency_evidence_uncommitted_slot_rejected",
                )
                payload_length = int.from_bytes(
                    slot[len(EMERGENCY_EVIDENCE_MAGIC) : len(EMERGENCY_EVIDENCE_MAGIC) + 4],
                    "big",
                )
                _require(
                    0 < payload_length <= usable_size - header_size,
                    "emergency_evidence_uncommitted_slot_rejected",
                )
                digest_start = len(EMERGENCY_EVIDENCE_MAGIC) + 4
                digest = slot[digest_start : digest_start + 32].hex()
                payload = slot[header_size : header_size + payload_length]
                _require(
                    _sha(payload) == digest
                    and slot[header_size + payload_length : usable_size]
                    == b"\0" * (usable_size - header_size - payload_length),
                    "emergency_evidence_uncommitted_slot_rejected",
                )
                value = _json(payload, "emergency_uncommitted_evidence")
                _require(
                    type(value) is dict
                    and payload == _canonical(value)
                    and set(value)
                    == {
                        "schema_version",
                        "kind",
                        "sequence",
                        "control_plane_plan_sha256",
                        "evidence_directory_acl_receipt_sha256",
                        "previous_evidence_sha256",
                        "intent",
                    }
                    and type(value["schema_version"]) is int
                    and value["schema_version"] == 1
                    and value["kind"] == "explainiverse-provider-mutation-emergency-evidence"
                    and type(value["sequence"]) is int
                    and value["sequence"] == index + 1
                    and value["control_plane_plan_sha256"] == plan_sha256
                    and value["evidence_directory_acl_receipt_sha256"] == acl_sha256
                    and value["previous_evidence_sha256"] == previous_sha256,
                    "emergency_evidence_uncommitted_slot_rejected",
                )
                live.MutationIntent.from_public_mapping(value["intent"])
                uncommitted_index = index
                unused_seen = True
                continue
            _require(
                slot.startswith(EMERGENCY_EVIDENCE_MAGIC),
                "emergency_evidence_magic_rejected",
            )
            payload_length = int.from_bytes(
                slot[len(EMERGENCY_EVIDENCE_MAGIC) : len(EMERGENCY_EVIDENCE_MAGIC) + 4],
                "big",
            )
            _require(
                0 < payload_length <= usable_size - header_size,
                "emergency_evidence_length_rejected",
            )
            digest_start = len(EMERGENCY_EVIDENCE_MAGIC) + 4
            digest = slot[digest_start : digest_start + 32].hex()
            payload = slot[header_size : header_size + payload_length]
            _require(
                _sha(payload) == digest
                and slot[header_size + payload_length : usable_size]
                == b"\0" * (usable_size - header_size - payload_length),
                "emergency_evidence_slot_digest_rejected",
            )
            value = _json(payload, "emergency_evidence")
            _require(
                type(value) is dict
                and payload == _canonical(value)
                and set(value)
                == {
                    "schema_version",
                    "kind",
                    "sequence",
                    "control_plane_plan_sha256",
                    "evidence_directory_acl_receipt_sha256",
                    "previous_evidence_sha256",
                    "intent",
                }
                and value["schema_version"] == 1
                and type(value["schema_version"]) is int
                and value["kind"] == "explainiverse-provider-mutation-emergency-evidence"
                and value["sequence"] == index + 1
                and type(value["sequence"]) is int
                and value["control_plane_plan_sha256"] == plan_sha256
                and value["evidence_directory_acl_receipt_sha256"] == acl_sha256
                and value["previous_evidence_sha256"] == previous_sha256,
                "emergency_evidence_binding_rejected",
            )
            normalized = live.MutationIntent.from_public_mapping(value["intent"])
            result.append(normalized.to_public_mapping())
            previous_sha256 = digest
        return tuple(result), uncommitted_index

    def _read_emergency_provider_intents(self) -> tuple[dict[str, Any], ...]:
        self._evidence_directory.validate()
        os.lseek(self._emergency_fd, 0, os.SEEK_SET)
        raw = bytearray()
        expected_size = EMERGENCY_EVIDENCE_SLOT_SIZE * EMERGENCY_EVIDENCE_SLOT_COUNT
        while len(raw) < expected_size:
            chunk = os.read(self._emergency_fd, expected_size - len(raw))
            _require(bool(chunk), "emergency_evidence_reserve_short_read")
            raw.extend(chunk)
        result, uncommitted_index = self._parse_emergency_provider_intents(
            bytes(raw),
            plan_sha256=self._plan_sha256,
            acl_sha256=self._acl_sha256,
        )
        _require(
            uncommitted_index is None,
            "emergency_evidence_requires_explicit_recovery",
        )
        return result

    def emergency_provider_intents(self) -> tuple[dict[str, Any], ...]:
        return self._read_emergency_provider_intents()

    def close(self) -> None:
        """Close the emergency reserve and held directory only after cleanup."""

        failure: BaseException | None = None
        descriptor = getattr(self, "_emergency_fd", -1)
        if descriptor >= 0:
            try:
                os.fsync(descriptor)
            except BaseException as error:
                failure = error
            try:
                os.close(descriptor)
            except BaseException as error:
                if failure is None:
                    failure = error
            self._emergency_fd = -1
        try:
            self._evidence_directory.close()
        except BaseException as error:
            if failure is None:
                failure = error
        if failure is not None:
            raise failure

    def archive_installed_app_capture(
        self,
        capture: TrustedAppCapture,
        evidence_reader: Callable[[str], bytes],
    ) -> dict[str, Any]:
        """Durably copy every authenticated App page into a digest namespace."""

        self._evidence_directory.validate()
        _require(
            type(capture) is TrustedAppCapture and callable(evidence_reader),
            "app_archive_inputs_rejected",
        )
        pages_root = self._directory / "installed-app-pages"
        if not pages_root.exists():
            os.mkdir(pages_root, 0o700)
            self._sync_directory(self._directory)
        _require(
            pages_root.is_dir()
            and not pages_root.is_symlink()
            and pages_root.resolve(strict=True) == pages_root,
            "app_archive_root_rejected",
        )
        capture_root = pages_root / capture.evidence_sha256
        if not capture_root.exists():
            os.mkdir(capture_root, 0o700)
            self._sync_directory(pages_root)
        _require(
            capture_root.is_dir()
            and not capture_root.is_symlink()
            and capture_root.resolve(strict=True) == capture_root,
            "app_archive_capture_directory_rejected",
        )
        manifest = capture.normalized_capture.get("evidence")
        _require(type(manifest) is list and bool(manifest), "app_archive_manifest_rejected")
        assert isinstance(manifest, list)
        archived: list[dict[str, Any]] = []
        for raw_item in manifest:
            _require(type(raw_item) is dict, "app_archive_manifest_item_rejected")
            item = raw_item
            filename = item.get("filename")
            expected_sha256 = item.get("sha256")
            expected_size = item.get("bytes")
            _require(
                type(filename) is str
                and Path(filename).name == filename
                and filename not in {".", ".."}
                and type(expected_sha256) is str
                and SHA256_RE.fullmatch(expected_sha256) is not None
                and type(expected_size) is int
                and expected_size > 0,
                "app_archive_manifest_binding_rejected",
            )
            page = evidence_reader(filename)
            _require(type(page) is bytes, "app_archive_reader_not_bytes")
            _require(
                len(page) == expected_size and _sha(page) == expected_sha256,
                "app_archive_page_drift",
            )
            destination = capture_root / filename
            if destination.exists():
                _require(
                    destination.is_file()
                    and not destination.is_symlink()
                    and destination.stat().st_nlink == 1
                    and destination.read_bytes() == page,
                    "app_archive_existing_page_drift",
                )
            else:
                self._publish_atomic(destination, page)
            archived.append(
                {"filename": filename, "bytes": expected_size, "sha256": expected_sha256}
            )
        material = {
            "capture_evidence_sha256": capture.evidence_sha256,
            "archive_directory": f"installed-app-pages/{capture.evidence_sha256}",
            "files": archived,
            "all_pages_exclusive_single_link": True,
        }
        return {**material, "archive_evidence_sha256": _sha(_canonical(material))}

    def archive_stale_installed_app_capture(
        self,
        *,
        phase: str,
        classified_at: str,
        generation_receipt: Mapping[str, Any],
        evidence_pages: Mapping[str, bytes],
        controller_resources: SealedControllerResources,
    ) -> dict[str, Any]:
        """Archive one stale inbox generation before it is advanced.

        The inbox has already classified this capture as stale, but that
        classification is not historically reproducible from page digests
        alone.  This method durably copies every raw page into the protected
        evidence directory and journals the exact source-generation binding.
        Accepted-evidence loaders later rerun ``TrustedAppCapture`` at the
        archived ``classified_at`` and require the sole failure to be
        ``app_capture_stale``.
        """

        self._evidence_directory.validate()
        _require(
            phase in PHASES
            and type(classified_at) is str
            and type(generation_receipt) is dict
            and type(evidence_pages) is dict,
            "stale_app_archive_inputs_rejected",
        )
        _require(
            type(controller_resources) is SealedControllerResources,
            "stale_app_archive_resources_rejected",
        )
        classified = _public_receipt_time(
            classified_at,
            "stale_app_archive_classified_at",
        )
        _require(
            classified.isoformat() == classified_at,
            "stale_app_archive_classified_at_rejected",
        )
        generation_keys = {
            "ordinal",
            "generation",
            "publication_nonce",
            "ready_marker",
            "ready_marker_bytes",
            "ready_marker_sha256",
            "capture_directory",
            "capture_json_bytes",
            "capture_json_sha256",
            "capture",
            "pages",
            "pages_inventory_sha256",
        }
        _require(
            set(generation_receipt) == generation_keys
            and type(generation_receipt["ordinal"]) is int
            and generation_receipt["ordinal"] > 0
            and type(generation_receipt["generation"]) is int
            and generation_receipt["generation"] > 0
            and type(generation_receipt["publication_nonce"]) is str
            and re.fullmatch(r"[0-9a-f]{32}", generation_receipt["publication_nonce"]) is not None
            and type(generation_receipt["capture"]) is dict
            and type(generation_receipt["pages"]) is list
            and bool(generation_receipt["pages"]),
            "stale_app_archive_generation_rejected",
        )
        capture = _json_copy(generation_receipt["capture"])
        capture_raw = _canonical(capture)
        _require(
            generation_receipt["capture_json_bytes"] == len(capture_raw)
            and generation_receipt["capture_json_sha256"] == _sha(capture_raw),
            "stale_app_archive_capture_binding_rejected",
        )
        manifest = capture.get("evidence")
        _require(
            type(manifest) is list and bool(manifest),
            "stale_app_archive_manifest_rejected",
        )
        declared_rows: list[dict[str, Any]] = []
        declared_names: set[str] = set()
        evidence_keys = {
            "filename",
            "kind",
            "installation_id",
            "source_url",
            "captured_at",
            "media_type",
            "full_page",
            "bytes",
            "sha256",
        }
        for raw_item in manifest:
            _require(
                type(raw_item) is dict
                and set(raw_item) == evidence_keys
                and type(raw_item["filename"]) is str
                and Path(raw_item["filename"]).name == raw_item["filename"]
                and raw_item["filename"] not in {".", ".."}
                and raw_item["filename"] not in declared_names
                and type(raw_item["bytes"]) is int
                and raw_item["bytes"] > 0
                and type(raw_item["sha256"]) is str
                and SHA256_RE.fullmatch(raw_item["sha256"]) is not None,
                "stale_app_archive_manifest_item_rejected",
            )
            declared_names.add(str(raw_item["filename"]))
            declared_rows.append(
                {
                    "filename": raw_item["filename"],
                    "bytes": raw_item["bytes"],
                    "sha256": raw_item["sha256"],
                }
            )
        _require(
            generation_receipt["pages"] == declared_rows
            and generation_receipt["pages_inventory_sha256"] == _sha(_canonical(declared_rows))
            and set(evidence_pages) == declared_names,
            "stale_app_archive_page_inventory_rejected",
        )
        try:
            TrustedAppCapture.from_mapping(
                capture,
                resources=controller_resources,
                evidence_reader=evidence_pages.__getitem__,
                now=classified,
            )
        except ControllerError as exc:
            _require(
                str(exc) == "app_capture_stale",
                "stale_app_archive_classification_rejected",
            )
        else:
            raise ControllerError("stale_app_archive_classification_rejected")
        identity_material = {
            "phase": phase,
            "ordinal": generation_receipt["ordinal"],
            "generation": generation_receipt["generation"],
            "publication_nonce": generation_receipt["publication_nonce"],
            "ready_marker_sha256": generation_receipt["ready_marker_sha256"],
            "capture_json_sha256": generation_receipt["capture_json_sha256"],
            "classified_at": classified_at,
        }
        archive_identity_sha256 = _sha(_canonical(identity_material))
        pages_root = self._directory / "installed-app-pages"
        if not pages_root.exists():
            os.mkdir(pages_root, 0o700)
            self._sync_directory(self._directory)
        _require(
            pages_root.is_dir()
            and not pages_root.is_symlink()
            and pages_root.resolve(strict=True) == pages_root,
            "stale_app_archive_root_rejected",
        )
        capture_root = pages_root / archive_identity_sha256
        if not capture_root.exists():
            os.mkdir(capture_root, 0o700)
            self._sync_directory(pages_root)
        _require(
            capture_root.is_dir()
            and not capture_root.is_symlink()
            and capture_root.resolve(strict=True) == capture_root,
            "stale_app_archive_capture_directory_rejected",
        )
        archived: list[dict[str, Any]] = []
        for item in declared_rows:
            filename = str(item["filename"])
            page = evidence_pages[filename]
            _require(
                type(page) is bytes and len(page) == item["bytes"] and _sha(page) == item["sha256"],
                "stale_app_archive_page_drift",
            )
            destination = capture_root / filename
            if destination.exists():
                _require(
                    destination.is_file()
                    and not destination.is_symlink()
                    and destination.stat().st_nlink == 1
                    and destination.read_bytes() == page,
                    "stale_app_archive_existing_page_drift",
                )
            else:
                self._publish_atomic(destination, page)
            archived.append(dict(item))
        material = {
            "schema_version": 1,
            "kind": "explainiverse-installed-app-stale-raw-archive",
            **identity_material,
            "archive_identity_sha256": archive_identity_sha256,
            "archive_directory": f"installed-app-pages/{archive_identity_sha256}",
            "files": archived,
            "all_pages_exclusive_single_link": True,
        }
        receipt = {
            **material,
            "archive_evidence_sha256": _sha(_canonical(material)),
        }
        self.record("installed-app-stale-raw-archive", receipt)
        return _json_copy(receipt)

    @staticmethod
    def _read_archived_app_page(root: Path, capture_evidence_sha256: str, filename: str) -> bytes:
        _require(
            SHA256_RE.fullmatch(capture_evidence_sha256) is not None
            and Path(filename).name == filename
            and filename not in {".", ".."},
            "archived_app_page_path_rejected",
        )
        capture_root = root / "installed-app-pages" / capture_evidence_sha256
        path = capture_root / filename
        _require(
            capture_root.is_dir()
            and not capture_root.is_symlink()
            and capture_root.resolve(strict=True) == capture_root
            and path.is_file()
            and not path.is_symlink()
            and path.resolve(strict=True) == path
            and path.parent == capture_root
            and path.stat().st_nlink == 1,
            "archived_app_page_file_rejected",
        )
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
        _require(
            before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns,
            "archived_app_page_changed_while_reading",
        )
        return raw

    @staticmethod
    def _validate_job_authority_evidence(
        *,
        capture: TrustedAppCapture,
        app_payload: Mapping[str, Any],
        archive_payload: Mapping[str, Any],
        authority_payload: Mapping[str, Any],
        runtime_plan: Mapping[str, Any],
    ) -> dict[str, Any]:
        authority = AuthorityReceipt.from_evidence_mapping(authority_payload)
        authority_mapping = authority.evidence_mapping()
        evidence_material = authority_mapping["evidence_material"]
        _require(
            type(evidence_material) is dict
            and evidence_material.get("app_capture") == app_payload
            and authority.app_capture_sha256 == capture.evidence_sha256
            and runtime_plan.get("authority_window") == authority.runtime_mapping(),
            "journal_authority_capture_or_plan_binding_rejected",
        )
        captured_at = _parse_time(capture.captured_at, "journal_app_capture_time")
        dispatch_at = _parse_time(
            runtime_plan.get("dispatch", {}).get("observed_at"),
            "journal_dispatch_observed",
        )
        observed_at = _parse_time(authority.observed_at, "journal_authority_observed")
        created_at = _parse_time(runtime_plan.get("created_at"), "journal_runtime_created")
        _require(
            dispatch_at < captured_at < observed_at < created_at
            and observed_at - captured_at <= AUTHORITY_CAPTURE_MAX_AGE,
            "journal_authority_capture_freshness_rejected",
        )
        archive_sha256 = archive_payload.get("archive_evidence_sha256")
        raw_manifest = capture.normalized_capture.get("evidence")
        _require(
            type(archive_sha256) is str
            and SHA256_RE.fullmatch(archive_sha256) is not None
            and type(raw_manifest) is list,
            "journal_authority_identity_rejected",
        )
        assert isinstance(raw_manifest, list)
        page_sha256 = tuple(str(item.get("sha256")) for item in raw_manifest if type(item) is dict)
        _require(
            len(page_sha256) == len(raw_manifest)
            and len(set(page_sha256)) == len(page_sha256)
            and all(SHA256_RE.fullmatch(item) is not None for item in page_sha256),
            "journal_authority_page_manifest_rejected",
        )
        material = {
            "schema_version": 1,
            "kind": "explainiverse-jit-authority-evidence-identity",
            "phase": runtime_plan["phase"],
            "head_sha": runtime_plan["dispatch"]["head_sha"],
            "run_id": runtime_plan["dispatch"]["run_id"],
            "job_key": runtime_plan["job"]["key"],
            "capture_evidence_sha256": capture.evidence_sha256,
            "authority_evidence_sha256": authority.evidence_sha256,
            "archive_evidence_sha256": archive_sha256,
            "raw_page_sha256": list(page_sha256),
            "dispatch_observed_at": runtime_plan["dispatch"]["observed_at"],
            "captured_at": capture.captured_at,
            "authority_observed_at": authority.observed_at,
            "runtime_created_at": runtime_plan["created_at"],
        }
        return _validated_authority_evidence_identity(
            {**material, "evidence_sha256": _sha(_canonical(material))},
            context="journal_authority_identity",
            expected_phase=str(runtime_plan["phase"]),
            expected_head_sha=str(runtime_plan["dispatch"]["head_sha"]),
            expected_run_id=int(runtime_plan["dispatch"]["run_id"]),
            expected_job_key=str(runtime_plan["job"]["key"]),
        )

    @staticmethod
    def _validate_provider_snapshot(
        payload: Mapping[str, Any],
        *,
        plan_sha256: str,
        expected_phase: str,
        context: str,
    ) -> None:
        """Validate one complete provider inventory, including every bound GET."""

        response_bindings = payload.get("response_bindings")
        _require(
            type(payload) is dict
            and set(payload)
            == {
                "plan_sha256",
                "phase",
                "snapshot_sha256",
                "receipt_nonce",
                "ruleset_id",
                "instance_id",
                "instance_public_ipv4",
                "response_bindings",
            }
            and payload.get("plan_sha256") == plan_sha256
            and payload.get("phase") == expected_phase
            and type(payload.get("snapshot_sha256")) is str
            and SHA256_RE.fullmatch(payload["snapshot_sha256"]) is not None
            and type(payload.get("receipt_nonce")) is str
            and re.fullmatch(r"[0-9a-f]{32}", payload["receipt_nonce"]) is not None
            and (
                (
                    expected_phase
                    in {"baseline", "global_restricted", "ruleset_absent", "restored"}
                    and payload.get("ruleset_id") is None
                    and payload.get("instance_id") is None
                    and payload.get("instance_public_ipv4") is None
                )
                or (
                    expected_phase == "ruleset_ready"
                    and type(payload.get("ruleset_id")) is str
                    and bool(payload.get("ruleset_id"))
                    and payload.get("instance_id") is None
                    and payload.get("instance_public_ipv4") is None
                )
                or (
                    expected_phase == "instance_bound"
                    and type(payload.get("ruleset_id")) is str
                    and bool(payload.get("ruleset_id"))
                    and type(payload.get("instance_id")) is str
                    and bool(payload.get("instance_id"))
                    and type(payload.get("instance_public_ipv4")) is str
                    and bool(payload.get("instance_public_ipv4"))
                )
                or (
                    expected_phase == "instance_absent"
                    and type(payload.get("ruleset_id")) is str
                    and bool(payload.get("ruleset_id"))
                    and payload.get("instance_id") is None
                    and payload.get("instance_public_ipv4") is None
                )
                or expected_phase == "recovery"
            )
            and type(response_bindings) is list
            and len(response_bindings) == len(live.READ_OPERATIONS)
            and all(
                type(binding) is dict
                and set(binding)
                == {
                    "operation",
                    "method",
                    "path",
                    "request_sha256",
                    "request_body_sha256",
                    "response_body_sha256",
                    "status_code",
                    "content_type",
                }
                and binding["operation"] == operation
                and binding["method"] == "GET"
                and binding["path"] == path
                and type(binding["request_sha256"]) is str
                and binding["request_sha256"]
                == live.ProviderRequest(operation, "GET", path, False).request_sha256
                and binding["request_body_sha256"] is None
                and type(binding["response_body_sha256"]) is str
                and SHA256_RE.fullmatch(binding["response_body_sha256"]) is not None
                and type(binding["status_code"]) is int
                and binding["status_code"] == 200
                and binding["content_type"] == "application/json"
                for binding, (operation, path) in zip(
                    response_bindings,
                    live.READ_OPERATIONS,
                )
            ),
            context,
        )

    @staticmethod
    def _validate_provider_restored_snapshot(
        payload: Mapping[str, Any],
        *,
        plan_sha256: str,
        context: str,
    ) -> None:
        """Validate the complete final provider inventory, not a semantic subset."""

        EvidenceJournal._validate_provider_snapshot(
            payload,
            plan_sha256=plan_sha256,
            expected_phase="restored",
            context=context,
        )

    @staticmethod
    def _validate_accepted_job_receipt(
        payload: Mapping[str, Any],
        *,
        phase: str,
        run_id: int,
        job_key: str,
        job_id: int,
        runner_id: int,
        runner_name: str,
        runtime_plan_sha256: str,
        remote_receipt_sha256: str,
        context: str,
    ) -> AcceptedJobReceipt:
        """Strictly reconstruct one accepted job; JSON bools never alias integers."""

        _require(
            type(payload) is dict and set(payload) == set(AcceptedJobReceipt.__dataclass_fields__),
            f"{context}_schema_rejected",
        )
        try:
            accepted = AcceptedJobReceipt(**payload)
        except TypeError:
            raise ControllerError(f"{context}_schema_rejected") from None
        digest_fields = (
            "runtime_plan_sha256",
            "remote_receipt_sha256",
            "actions_job_response_sha256",
            "check_response_sha256",
            "log_sha256",
            "runner_inventory_response_sha256",
            "post_execution_observation_sha256",
            "evidence_sha256",
        )
        material = accepted.to_mapping()
        evidence_sha256 = material.pop("evidence_sha256")
        _require(
            type(accepted.phase) is str
            and accepted.phase == phase
            and type(accepted.run_id) is int
            and accepted.run_id == run_id
            and type(accepted.job_key) is str
            and accepted.job_key == job_key
            and type(accepted.job_id) is int
            and accepted.job_id == job_id
            and type(accepted.runner_id) is int
            and accepted.runner_id == runner_id
            and type(accepted.runner_name) is str
            and accepted.runner_name == runner_name
            and accepted.runtime_plan_sha256 == runtime_plan_sha256
            and accepted.remote_receipt_sha256 == remote_receipt_sha256
            and type(accepted.pytest_passed) is int
            and accepted.pytest_passed == 15
            and type(accepted.pytest_skipped) is int
            and accepted.pytest_skipped == 0
            and all(
                type(getattr(accepted, field_name)) is str
                and SHA256_RE.fullmatch(getattr(accepted, field_name)) is not None
                for field_name in digest_fields
            )
            and _sha(_canonical(material)) == evidence_sha256,
            f"{context}_binding_drift",
        )
        return accepted

    @staticmethod
    def _validate_phase_settlement(
        payload: Mapping[str, Any],
        *,
        phase: str,
        run_id: int,
        head_sha: str,
        expected_job_evidence_sha256: Sequence[str],
        expected_nonces: Sequence[str] = (),
        context: str,
    ) -> str:
        """Validate the exact terminal GPU settlement archived for one phase."""

        if phase == "final-main":
            expected_keys = {
                "phase",
                "run_id",
                "run_attempt",
                "head_sha",
                "accepted_cuda_runner_nonces",
                "job_evidence_sha256",
                "all_four_jobs_15_of_15_zero_skips",
                "rerun_performed",
                "evidence_sha256",
            }
        elif phase == "publication":
            expected_keys = {
                "phase",
                "run_id",
                "run_attempt",
                "head_sha",
                "tag",
                "stage_recovery_drill",
                "job_evidence_sha256",
                "runner_inventory_response_sha256",
                "both_release_jobs_15_of_15_zero_skips",
                "workflow_publication_success_not_claimed",
                "rerun_performed",
                "evidence_sha256",
            }
        else:
            raise ControllerError(f"{context}_phase_rejected")
        _require(type(payload) is dict and set(payload) == expected_keys, f"{context}_rejected")
        material = dict(payload)
        evidence_sha256 = material.pop("evidence_sha256", None)
        common_valid = (
            payload.get("phase") == phase
            and type(payload.get("run_id")) is int
            and payload.get("run_id") == run_id
            and type(payload.get("run_attempt")) is int
            and payload.get("run_attempt") == 1
            and payload.get("head_sha") == head_sha
            and type(payload.get("job_evidence_sha256")) is list
            and payload.get("job_evidence_sha256") == list(expected_job_evidence_sha256)
            and all(
                type(value) is str and SHA256_RE.fullmatch(value) is not None
                for value in expected_job_evidence_sha256
            )
            and payload.get("rerun_performed") is False
            and type(evidence_sha256) is str
            and SHA256_RE.fullmatch(evidence_sha256) is not None
            and _sha(_canonical(material)) == evidence_sha256
        )
        if phase == "final-main":
            phase_valid = (
                type(payload.get("accepted_cuda_runner_nonces")) is list
                and payload.get("accepted_cuda_runner_nonces") == list(expected_nonces)
                and len(expected_nonces) == 4
                and len(set(expected_nonces)) == 4
                and all(
                    type(value) is str and re.fullmatch(r"[0-9a-f]{16}", value) is not None
                    for value in expected_nonces
                )
                and payload.get("all_four_jobs_15_of_15_zero_skips") is True
            )
        else:
            inventory_sha256 = payload.get("runner_inventory_response_sha256")
            phase_valid = (
                payload.get("tag") == runtime.PUBLICATION_TAG
                and payload.get("stage_recovery_drill") is True
                and type(inventory_sha256) is str
                and SHA256_RE.fullmatch(inventory_sha256) is not None
                and payload.get("both_release_jobs_15_of_15_zero_skips") is True
                and payload.get("workflow_publication_success_not_claimed") is True
            )
        _require(common_valid and phase_valid, f"{context}_rejected")
        assert isinstance(evidence_sha256, str)
        return evidence_sha256

    @staticmethod
    def _validate_operator_executables(
        value: Mapping[str, Any],
        *,
        ssh_receipt: Mapping[str, Any],
        github_receipt: Mapping[str, Any],
        context: str,
    ) -> Mapping[str, Mapping[str, Any]]:
        """Validate and bind the complete pinned action-time binary inventory."""

        _require(type(value) is dict and set(value) == {"git", "gh", "ssh", "python"}, context)
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

        def pinned_row(name: str, *, resolved_runtime: bool = False) -> Mapping[str, Any]:
            row = value[name]
            expected = PINNED_OPERATOR_EXECUTABLES[name]
            expected_keys = pinned | ({"resolved_runtime"} if resolved_runtime else set())
            _require(type(row) is dict and set(row) == expected_keys, context)
            acl = row["acl"]
            signature = row["authenticode"]
            _require(
                row["absolute_path"] == expected["absolute_path"]
                and row["sha256"] == expected["sha256"]
                and row["version"] == expected["version"]
                and row["regular_file"] is True
                and row["symlink_or_reparse"] is False
                and row["path_lookup_used"] is False
                and type(row["hardlink_count"]) is int
                and row["hardlink_count"] >= 1
                and (name not in {"gh"} or row["hardlink_count"] == 1)
                and row["pinned_reviewed_identity"] is True
                and row["authenticode_validated_by_pinned_helper"] is True
                and type(acl) is dict
                and set(acl)
                == {
                    "owner_sid",
                    "expected_owner_sid",
                    "unprivileged_write_ace_present",
                    "dacl_ace_count",
                    "dacl_inventory_sha256",
                }
                and acl["owner_sid"] == expected["owner_sid"]
                and acl["expected_owner_sid"] == expected["owner_sid"]
                and acl["unprivileged_write_ace_present"] is False
                and type(acl["dacl_ace_count"]) is int
                and acl["dacl_ace_count"] > 0
                and type(acl["dacl_inventory_sha256"]) is str
                and SHA256_RE.fullmatch(acl["dacl_inventory_sha256"]) is not None
                and type(signature) is dict
                and set(signature) == {"status", "subject", "thumbprint"}
                and signature["status"] == "Valid"
                and signature["subject"] == expected["authenticode_subject"]
                and signature["thumbprint"] == expected["authenticode_thumbprint"],
                context,
            )
            return row

        git = pinned_row("git", resolved_runtime=True)
        gh = pinned_row("gh")
        ssh = pinned_row("ssh")
        runtime_row = git["resolved_runtime"]
        _require(
            type(runtime_row) is dict
            and set(runtime_row)
            == common
            - {
                "regular_file",
                "symlink_or_reparse",
                "path_lookup_used",
            }
            | {"acl", "authenticode"},
            context,
        )
        runtime_acl = runtime_row["acl"]
        runtime_signature = runtime_row["authenticode"]
        git_expected = PINNED_OPERATOR_EXECUTABLES["git"]
        _require(
            runtime_row["absolute_path"] == git_expected["runtime_absolute_path"]
            and runtime_row["sha256"] == git_expected["runtime_sha256"]
            and runtime_row["version"] == git_expected["version"]
            and type(runtime_row["hardlink_count"]) is int
            and runtime_row["hardlink_count"] >= 1
            and runtime_acl == git["acl"]
            and runtime_signature == git["authenticode"],
            context,
        )
        python_row = value["python"]
        _require(
            type(python_row) is dict
            and set(python_row) == common | {"pinned_runtime_manifest_authority"}
            and type(python_row["absolute_path"]) is str
            and _is_windows_absolute_path(python_row["absolute_path"])
            and type(python_row["sha256"]) is str
            and SHA256_RE.fullmatch(python_row["sha256"]) is not None
            and python_row["version"] == "Python 3.13.15"
            and python_row["regular_file"] is True
            and python_row["symlink_or_reparse"] is False
            and python_row["path_lookup_used"] is False
            and type(python_row["hardlink_count"]) is int
            and python_row["hardlink_count"] == 1
            and python_row["pinned_runtime_manifest_authority"] is True
            and ssh["absolute_path"] == ssh_receipt["absolute_path"]
            and ssh["sha256"] == ssh_receipt["sha256"]
            and gh["absolute_path"] == github_receipt["absolute_path"]
            and gh["sha256"] == github_receipt["sha256"],
            context,
        )
        return value  # type: ignore[return-value]

    @staticmethod
    def _validate_preloader_directory_receipt(
        public: Mapping[str, Any],
        validation: Mapping[str, Any],
        *,
        context: str,
    ) -> str:
        public_keys = {
            "captured_at",
            "receipt_sha256",
            "absolute_path_redacted",
            "directory_identity_recorded",
            "no_reparse_or_symlink",
            "owner_private",
            "acl",
        }
        validation_keys = {
            "validated_at",
            "receipt_sha256",
            "absolute_path_redacted",
            "directory_identity_recorded",
            "no_reparse_or_symlink",
            "owner_private",
            "acl_evidence_sha256",
        }
        _require(
            type(public) is dict
            and set(public) == public_keys
            and type(validation) is dict
            and set(validation) == validation_keys,
            context,
        )
        acl = public["acl"]
        acl_keys = {
            "owner_sid",
            "current_user_sid",
            "inheritance_protected",
            "child_inheritance_enabled",
            "aces",
            "security_descriptor_sha256",
            "security_descriptor_bytes",
            "captured_at",
            "evidence_sha256",
        }
        _require(type(acl) is dict and set(acl) == acl_keys, context)
        aces = acl["aces"]
        _require(
            type(aces) is list
            and len(aces) == 3
            and aces == sorted(aces, key=lambda item: str(item.get("sid")))
            and type(acl["owner_sid"]) is str
            and re.fullmatch(r"S-1-[0-9-]+", acl["owner_sid"]) is not None
            and acl["current_user_sid"] == acl["owner_sid"]
            and acl["inheritance_protected"] is True
            and acl["child_inheritance_enabled"] is True
            and {item["sid"] for item in aces if type(item) is dict and "sid" in item}
            == {acl["owner_sid"], "S-1-5-18", "S-1-5-32-544"}
            and all(
                type(item) is dict
                and set(item) == {"sid", "access", "rights", "mask", "ace_flags"}
                and item["access"] == "allow"
                and item["rights"] == "full-control"
                and type(item["mask"]) is int
                and item["mask"] == 2_032_127
                and type(item["ace_flags"]) is int
                and item["ace_flags"] == 3
                for item in aces
            )
            and type(acl["security_descriptor_sha256"]) is str
            and SHA256_RE.fullmatch(acl["security_descriptor_sha256"]) is not None
            and type(acl["security_descriptor_bytes"]) is int
            and acl["security_descriptor_bytes"] > 0
            and type(acl["evidence_sha256"]) is str
            and _sha(
                _canonical(
                    {
                        key: item
                        for key, item in acl.items()
                        if key not in {"captured_at", "evidence_sha256"}
                    }
                )
            )
            == acl["evidence_sha256"],
            context,
        )
        captured_at = _public_receipt_time(acl["captured_at"], f"{context}_acl")
        _require(
            public["captured_at"] == acl["captured_at"]
            and type(public["receipt_sha256"]) is str
            and SHA256_RE.fullmatch(public["receipt_sha256"]) is not None
            and all(
                public[field_name] is True
                for field_name in (
                    "absolute_path_redacted",
                    "directory_identity_recorded",
                    "no_reparse_or_symlink",
                    "owner_private",
                )
            )
            and _public_receipt_time(validation["validated_at"], f"{context}_validation")
            >= captured_at
            and validation["receipt_sha256"] == public["receipt_sha256"]
            and all(
                validation[field_name] is True
                for field_name in (
                    "absolute_path_redacted",
                    "directory_identity_recorded",
                    "no_reparse_or_symlink",
                    "owner_private",
                )
            )
            and validation["acl_evidence_sha256"] == acl["evidence_sha256"],
            context,
        )
        return str(public["receipt_sha256"])

    @staticmethod
    def _validate_operator_preloader(
        value: Mapping[str, Any],
        *,
        phase: str,
        immutable_plan: Mapping[str, Any],
        repository: Mapping[str, Any],
        environment: Mapping[str, Any],
        executables: Mapping[str, Mapping[str, Any]],
        controller_resources: SealedControllerResources,
        working_directory: str,
        context: str,
    ) -> None:
        keys = {
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
        }
        _require(
            type(value) is dict
            and set(value) == keys
            and type(value["schema_version"]) is int
            and value["schema_version"] == 1
            and value["kind"] == "explainiverse-operator-isolated-preloader"
            and value["environment"] == environment
            and value["working_directory"] == working_directory
            and value["working_directory_is_python_install_receipt_directory"] is True
            and all(
                value[field_name] is True
                for field_name in (
                    "isolated",
                    "safe_path",
                    "site_disabled",
                    "bytecode_disabled",
                    "repository_absent_from_sys_path",
                    "project_imports_from_captured_bytes",
                )
            ),
            context,
        )
        shim = value["shim"]
        shim_keys = {
            "schema_version",
            "kind",
            "preloader_path",
            "preloader_bytes",
            "preloader_sha256",
            "shim_sha256",
            "stable_descriptor_read",
            "compiled_verified_bytes_without_reopen",
        }
        _require(
            type(shim) is dict
            and set(shim) == shim_keys
            and type(shim["schema_version"]) is int
            and shim["schema_version"] == 1
            and shim["kind"] == "explainiverse-operator-preloader-shim"
            and type(shim["preloader_path"]) is str
            and _is_windows_absolute_path(shim["preloader_path"])
            and type(shim["preloader_bytes"]) is int
            and 1 <= shim["preloader_bytes"] <= 4 * 1024 * 1024
            and type(shim["preloader_sha256"]) is str
            and SHA256_RE.fullmatch(shim["preloader_sha256"]) is not None
            and shim["shim_sha256"] == OPERATOR_PRELOADER_SHIM_SHA256
            and shim["stable_descriptor_read"] is True
            and shim["compiled_verified_bytes_without_reopen"] is True,
            f"{context}_shim_rejected",
        )
        source = value["source"]
        source_keys = {
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
        }
        _require(type(source) is dict and set(source) == source_keys, f"{context}_source_rejected")
        manifest = source["source_manifest"]
        manifest_keys = {
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
        }
        _require(
            type(manifest) is dict
            and set(manifest) == manifest_keys
            and type(manifest["schema_version"]) is int
            and manifest["schema_version"] == 1
            and manifest["kind"] == "explainiverse-operator-source-worktree-manifest"
            and manifest["excluded_paths"]
            == [OPERATOR_SOURCE_MANIFEST_RELATIVE, OPERATOR_PRELOADER_RELATIVE]
            and manifest["source"] == "exact-staged-index-blobs"
            and manifest["runtime_git_dependency"] is False
            and type(manifest["files"]) is dict
            and type(manifest["directories"]) is list,
            f"{context}_source_manifest_rejected",
        )
        manifest_files = manifest["files"]
        manifest_directories = manifest["directories"]
        assert isinstance(manifest_files, dict)
        assert isinstance(manifest_directories, list)
        file_rows: list[bytes] = []
        expected_directories: set[str] = set()
        for relative, item in sorted(manifest_files.items()):
            pure = PurePosixPath(relative) if type(relative) is str else PurePosixPath(".")
            _require(
                type(relative) is str
                and relative == pure.as_posix()
                and not pure.is_absolute()
                and all(part not in {"", ".", ".."} for part in pure.parts)
                and relative not in {OPERATOR_SOURCE_MANIFEST_RELATIVE, OPERATOR_PRELOADER_RELATIVE}
                and type(item) is dict
                and set(item) == {"mode", "bytes", "sha256", "git_blob_sha"}
                and item["mode"] in {"100644", "100755"}
                and type(item["bytes"]) is int
                and item["bytes"] >= 0
                and type(item["sha256"]) is str
                and SHA256_RE.fullmatch(item["sha256"]) is not None
                and type(item["git_blob_sha"]) is str
                and re.fullmatch(r"[0-9a-f]{40}", item["git_blob_sha"]) is not None,
                f"{context}_source_manifest_file_rejected",
            )
            file_rows.append(
                (
                    f"{relative}\t{item['mode']}\t{item['bytes']}\t"
                    f"{item['sha256']}\t{item['git_blob_sha']}\n"
                ).encode("utf-8")
            )
            parent = pure.parent
            while parent != PurePosixPath("."):
                expected_directories.add(parent.as_posix())
                parent = parent.parent
        _require(
            type(manifest["file_count"]) is int
            and manifest["file_count"] == len(manifest_files)
            and type(manifest["directory_count"]) is int
            and manifest["directory_count"] == len(manifest_directories)
            and manifest_directories == sorted(expected_directories)
            and manifest["file_inventory_sha256"] == _sha(b"".join(file_rows)),
            f"{context}_source_manifest_inventory_rejected",
        )
        manifest_raw = _canonical(manifest)
        manifest_sha256 = _sha(manifest_raw)
        manifest_blob_sha = hashlib.sha1(
            f"blob {len(manifest_raw)}\0".encode("ascii") + manifest_raw
        ).hexdigest()

        repository_keys = {
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
        _require(
            type(repository) is dict
            and set(repository) == repository_keys
            and repository["repository"] == REPOSITORY
            and type(repository["absolute_root"]) is str
            and _is_windows_absolute_path(repository["absolute_root"])
            and repository["origin_url"] == "https://github.com/jemsbhai/explainiverse.git"
            and repository["head_sha"] == immutable_plan["head_sha"]
            and type(repository["tree_object_sha"]) is str
            and re.fullmatch(r"[0-9a-f]{40}", repository["tree_object_sha"]) is not None
            and type(repository["tree_inventory_sha256"]) is str
            and SHA256_RE.fullmatch(repository["tree_inventory_sha256"]) is not None
            and repository["clean_tracked_and_untracked"] is True
            and repository["remote_target_sha"] == immutable_plan["head_sha"]
            and type(repository["remote_object_sha"]) is str
            and re.fullmatch(r"[0-9a-f]{40}", repository["remote_object_sha"]) is not None
            and type(repository["remote_ref_response_sha256"]) is str
            and SHA256_RE.fullmatch(repository["remote_ref_response_sha256"]) is not None,
            f"{context}_repository_rejected",
        )
        _require(
            repository["git_configuration"]
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
            f"{context}_repository_git_configuration_rejected",
        )
        if phase == "final-main":
            _require(
                repository["supplied_ref"] == runtime.FINAL_MAIN_REF
                and repository["remote_object_type"] == "commit"
                and repository["remote_object_sha"] == immutable_plan["head_sha"]
                and repository["annotated_tag_response_sha256"] is None,
                f"{context}_repository_ref_rejected",
            )
        elif phase == "publication":
            _require(
                repository["supplied_ref"] == runtime.PUBLICATION_REF
                and repository["remote_object_type"] == "tag"
                and type(repository["annotated_tag_response_sha256"]) is str
                and SHA256_RE.fullmatch(repository["annotated_tag_response_sha256"]) is not None,
                f"{context}_repository_ref_rejected",
            )
        else:
            raise ControllerError(f"{context}_phase_rejected")
        critical_sources = repository["critical_sources"]
        _require(
            type(critical_sources) is dict
            and set(critical_sources) == OPERATOR_CRITICAL_SOURCE_PATHS,
            f"{context}_critical_source_set_rejected",
        )
        for relative, item in critical_sources.items():
            _require(
                type(item) is dict
                and set(item) == {"bytes", "sha256", "git_blob_sha"}
                and type(item["bytes"]) is int
                and item["bytes"] >= 0
                and type(item["sha256"]) is str
                and SHA256_RE.fullmatch(item["sha256"]) is not None
                and type(item["git_blob_sha"]) is str
                and re.fullmatch(r"[0-9a-f]{40}", item["git_blob_sha"]) is not None
                and (
                    relative in {OPERATOR_SOURCE_MANIFEST_RELATIVE, OPERATOR_PRELOADER_RELATIVE}
                    or (
                        relative in manifest_files
                        and item["bytes"] == manifest_files[relative]["bytes"]
                        and item["sha256"] == manifest_files[relative]["sha256"]
                        and item["git_blob_sha"] == manifest_files[relative]["git_blob_sha"]
                    )
                ),
                f"{context}_critical_source_binding_rejected",
            )
        _require(
            critical_sources[OPERATOR_SOURCE_MANIFEST_RELATIVE]
            == {
                "bytes": len(manifest_raw),
                "sha256": manifest_sha256,
                "git_blob_sha": manifest_blob_sha,
            },
            f"{context}_source_manifest_repository_binding_rejected",
        )
        _require(
            critical_sources[".github/release-control-policy.json"]["sha256"]
            == controller_resources.policy_sha256
            and critical_sources["scripts/release_gpu_jit_lambda_controller/controller.py"][
                "sha256"
            ]
            == controller_resources.controller_source_sha256,
            f"{context}_sealed_controller_source_binding_rejected",
        )
        tree_items: dict[str, tuple[str, str]] = {
            relative: (item["mode"], item["git_blob_sha"])
            for relative, item in manifest_files.items()
        }
        tree_items[OPERATOR_SOURCE_MANIFEST_RELATIVE] = ("100644", manifest_blob_sha)
        tree_items[OPERATOR_PRELOADER_RELATIVE] = (
            "100644",
            critical_sources[OPERATOR_PRELOADER_RELATIVE]["git_blob_sha"],
        )
        tree_raw = b"".join(
            f"{mode} blob {blob}\t{relative}\0".encode("utf-8")
            for relative, (mode, blob) in sorted(tree_items.items())
        )
        _require(
            _sha(tree_raw) == repository["tree_inventory_sha256"],
            f"{context}_repository_tree_inventory_rejected",
        )
        source_material = dict(source)
        source_evidence = source_material.pop("evidence_sha256", None)
        captured_entries = {
            relative: item
            for relative, item in manifest_files.items()
            if relative.startswith(OPERATOR_CAPTURE_PREFIXES) or relative in OPERATOR_CAPTURE_EXACT
        }
        capture_rows = [
            f"{relative}\t{item['bytes']}\t{item['sha256']}\n".encode("utf-8")
            for relative, item in sorted(captured_entries.items())
        ]
        _require(
            type(source["schema_version"]) is int
            and source["schema_version"] == 1
            and source["kind"] == "explainiverse-operator-clean-source-preload"
            and type(source["repository_root"]) is str
            and _is_windows_absolute_path(source["repository_root"])
            and source["repository_root"] == repository["absolute_root"]
            and source["origin_url"] == "https://github.com/jemsbhai/explainiverse.git"
            and source["head_sha"] == immutable_plan["head_sha"]
            and source["head_and_origin_verified_during_credential_free_inventory"] is False
            and source["source_manifest"] == manifest
            and source["source_manifest_sha256"] == manifest_sha256
            and source["source_manifest_inventory_sha256"] == manifest["file_inventory_sha256"]
            and source["tracked_and_untracked_clean"] is True
            and source["runtime_git_dependency"] is False
            and source["preloader_path"] == shim["preloader_path"]
            and source["preloader_sha256"] == shim["preloader_sha256"]
            and source["preloader_sha256"]
            == critical_sources[OPERATOR_PRELOADER_RELATIVE]["sha256"]
            and shim["shim_sha256"] == critical_sources[OPERATOR_PRELOADER_SHIM_RELATIVE]["sha256"]
            and type(source["captured_module_count"]) is int
            and source["captured_module_count"] == len(captured_entries)
            and source["captured_module_inventory_sha256"] == _sha(b"".join(capture_rows))
            and source["project_modules_execute_from_captured_bytes"] is True
            and type(source["arguments_sha256"]) is str
            and SHA256_RE.fullmatch(source["arguments_sha256"]) is not None
            and type(source_evidence) is str
            and _sha(_canonical(source_material)) == source_evidence,
            f"{context}_source_rejected",
        )

        python_install = value["python_install_receipt"]
        site_install = value["site_install_receipt"]
        python_install_keys = {
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
        }
        site_install_keys = {
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
        }
        _require(
            type(python_install) is dict
            and set(python_install) == python_install_keys
            and type(python_install["schema_version"]) is int
            and python_install["schema_version"] == 1
            and python_install["kind"] == "explainiverse-operator-python-runtime-installed"
            and type(python_install["python_runtime_root"]) is str
            and _is_windows_absolute_path(python_install["python_runtime_root"])
            and python_install["archive_sha256"] == OPERATOR_PYTHON_ARCHIVE_SHA256
            and python_install["manifest_sha256"] == OPERATOR_PYTHON_MANIFEST_SHA256
            and type(python_install["file_count"]) is int
            and python_install["file_count"] == 34
            and type(python_install["directory_count"]) is int
            and python_install["directory_count"] == 0
            and python_install["file_inventory_sha256"] == OPERATOR_PYTHON_FILE_INVENTORY_SHA256
            and python_install["owner_private_acl_applied_before_children"] is True
            and python_install["site_processing_disabled_by_embeddable_pth"] is True
            and python_install["untracked_files_or_directories_present"] is False
            and python_install["crash_recovery"]
            == "discard-partial-directory-and-create-a-new-path"
            and value["python_install_receipt_sha256"] == _sha(_canonical(python_install)),
            f"{context}_python_install_rejected",
        )
        _require(
            type(site_install) is dict
            and set(site_install) == site_install_keys
            and type(site_install["schema_version"]) is int
            and site_install["schema_version"] == 1
            and site_install["kind"] == "explainiverse-operator-runtime-installed"
            and type(site_install["runtime_root"]) is str
            and _is_windows_absolute_path(site_install["runtime_root"])
            and site_install["manifest_sha256"] == OPERATOR_SITE_MANIFEST_SHA256
            and type(site_install["file_count"]) is int
            and site_install["file_count"] == 756
            and type(site_install["directory_count"]) is int
            and site_install["directory_count"] == 113
            and site_install["file_inventory_sha256"] == OPERATOR_SITE_FILE_INVENTORY_SHA256
            and site_install["owner_private_acl_applied_before_children"] is True
            and site_install["pip_present_in_runtime"] is False
            and site_install["record_files_present"] is False
            and site_install["generated_scripts_present"] is False
            and site_install["bytecode_present"] is False
            and site_install["crash_recovery"] == "discard-partial-directory-and-create-a-new-path"
            and value["site_install_receipt_sha256"] == _sha(_canonical(site_install)),
            f"{context}_site_install_rejected",
        )

        directory_pairs = (
            ("python_runtime_directory_receipt", "python_runtime_validation"),
            ("runtime_site_directory_receipt", "runtime_site_validation"),
            ("python_install_directory_receipt", "python_install_directory_validation"),
            ("site_install_directory_receipt", "site_install_directory_validation"),
        )
        receipt_sha256 = [
            EvidenceJournal._validate_preloader_directory_receipt(
                value[public_name],
                value[validation_name],
                context=f"{context}_{public_name}",
            )
            for public_name, validation_name in directory_pairs
        ]
        _require(len(set(receipt_sha256)) == 4, f"{context}_directory_receipt_reused")

        bootstrap = value["bootstrap"]
        bootstrap_keys = {
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
        }
        _require(
            type(bootstrap) is dict and set(bootstrap) == bootstrap_keys,
            f"{context}_bootstrap_rejected",
        )
        python_tree = bootstrap["python_tree"]
        site_tree = bootstrap["site_tree"]
        preactivation = bootstrap["preactivation"]
        _require(
            type(python_tree) is dict
            and set(python_tree)
            == {
                "python_root",
                "file_count",
                "directory_count",
                "file_inventory_sha256",
                "official_archive_sha256",
                "untracked_files_or_directories_present",
                "all_runtime_bytes_match_official_archive",
            }
            and python_tree["python_root"] == python_install["python_runtime_root"]
            and python_tree["file_count"] == python_install["file_count"]
            and type(python_tree["file_count"]) is int
            and python_tree["directory_count"] == python_install["directory_count"]
            and type(python_tree["directory_count"]) is int
            and python_tree["file_inventory_sha256"] == python_install["file_inventory_sha256"]
            and python_tree["official_archive_sha256"] == OPERATOR_PYTHON_ARCHIVE_SHA256
            and python_tree["untracked_files_or_directories_present"] is False
            and python_tree["all_runtime_bytes_match_official_archive"] is True
            and type(site_tree) is dict
            and set(site_tree)
            == {
                "site_root",
                "file_count",
                "directory_count",
                "file_inventory_sha256",
                "untracked_files_or_directories_present",
                "bytecode_present",
                "all_importable_bytes_match_verified_wheels",
            }
            and site_tree["site_root"] == site_install["runtime_root"]
            and type(site_tree["file_count"]) is int
            and site_tree["file_count"] == site_install["file_count"]
            and type(site_tree["directory_count"]) is int
            and site_tree["directory_count"] == site_install["directory_count"]
            and site_tree["file_inventory_sha256"] == site_install["file_inventory_sha256"]
            and site_tree["untracked_files_or_directories_present"] is False
            and site_tree["bytecode_present"] is False
            and site_tree["all_importable_bytes_match_verified_wheels"] is True
            and type(preactivation) is dict
            and set(preactivation)
            == {"working_directory", "sys_path_sha256", "only_base_stdlib_roots"}
            and preactivation["working_directory"] == working_directory
            and type(preactivation["sys_path_sha256"]) is str
            and SHA256_RE.fullmatch(preactivation["sys_path_sha256"]) is not None
            and preactivation["only_base_stdlib_roots"] is True,
            f"{context}_bootstrap_tree_rejected",
        )
        activation_root = Path(str(site_tree["site_root"]))
        _require(
            type(bootstrap["schema_version"]) is int
            and bootstrap["schema_version"] == 1
            and bootstrap["kind"] == "explainiverse-operator-pre-site-bootstrap"
            and bootstrap["python_manifest_sha256"] == OPERATOR_PYTHON_MANIFEST_SHA256
            and bootstrap["python_archive_sha256"] == OPERATOR_PYTHON_ARCHIVE_SHA256
            and bootstrap["manifest_sha256"] == OPERATOR_SITE_MANIFEST_SHA256
            and all(
                type(bootstrap[field_name]) is str
                and SHA256_RE.fullmatch(bootstrap[field_name]) is not None
                for field_name in (
                    "archive_set_sha256",
                    "runtime_requirements_sha256",
                    "bootstrap_requirements_sha256",
                )
            )
            and bootstrap["base_python_executable"] == executables["python"]["absolute_path"]
            and bootstrap["base_python_executable_sha256"] == executables["python"]["sha256"]
            and bootstrap["activation_paths"]
            == [
                str(activation_root),
                str(activation_root / "win32"),
                str(activation_root / "win32" / "lib"),
                str(activation_root / "pythonwin"),
            ]
            and bootstrap["site_processing_disabled"] is True
            and bootstrap["pth_executed_by_cpython"] is False
            and bootstrap["verified_pywin32_bootstrap_imported_after_verification"] is True,
            f"{context}_bootstrap_rejected",
        )

        early = value["early_runtime_boundary"]
        early_keys = {
            "schema_version",
            "kind",
            "acl",
            "held_trees",
            "all_runtime_and_receipt_roots_owner_private",
            "all_runtime_and_receipt_paths_held_without_write_or_delete_share",
            "validated_before_third_party_site_or_native_import",
            "pinned_official_python_runtime_is_the_pre_hold_trust_boundary",
            "working_directory",
            "working_directory_repository_disjoint",
            "evidence_sha256",
        }
        _require(
            type(early) is dict and set(early) == early_keys, f"{context}_early_boundary_rejected"
        )
        early_material = dict(early)
        early_evidence = early_material.pop("evidence_sha256", None)
        acl_inventory = early["acl"]
        held_trees = early["held_trees"]
        _require(
            type(acl_inventory) is dict
            and set(acl_inventory)
            == {"python_root", "site_root", "python_receipt_root", "site_receipt_root"}
            and type(held_trees) is dict
            and set(held_trees)
            == {
                "root_count",
                "held_handle_count",
                "write_share_allowed",
                "delete_share_allowed",
                "read_share_allowed",
                "held_before_third_party_site_or_native_import",
            }
            and type(held_trees["root_count"]) is int
            and held_trees["root_count"] == 4
            and type(held_trees["held_handle_count"]) is int
            and held_trees["held_handle_count"] >= 4
            and held_trees["write_share_allowed"] is False
            and held_trees["delete_share_allowed"] is False
            and held_trees["read_share_allowed"] is True
            and held_trees["held_before_third_party_site_or_native_import"] is True,
            f"{context}_early_boundary_rejected",
        )
        early_owner: str | None = None
        for acl_name in sorted(acl_inventory):
            acl = acl_inventory[acl_name]
            _require(
                type(acl) is dict
                and set(acl)
                == {
                    "owner_sid",
                    "inheritance_protected",
                    "child_inheritance_enabled",
                    "allowed_sids",
                    "ace_count",
                    "rights",
                    "security_descriptor_sha256",
                    "security_descriptor_bytes",
                    "validated_before_third_party_site_or_native_import",
                    "pinned_stdlib_native_modules_loaded_before_hold",
                }
                and type(acl["owner_sid"]) is str
                and re.fullmatch(r"S-1-[0-9-]+", acl["owner_sid"]) is not None
                and (early_owner is None or acl["owner_sid"] == early_owner)
                and acl["inheritance_protected"] is True
                and acl["child_inheritance_enabled"] is True
                and acl["allowed_sids"] == sorted([acl["owner_sid"], "S-1-5-18", "S-1-5-32-544"])
                and type(acl["ace_count"]) is int
                and acl["ace_count"] == 3
                and acl["rights"] == "full-control"
                and type(acl["security_descriptor_sha256"]) is str
                and SHA256_RE.fullmatch(acl["security_descriptor_sha256"]) is not None
                and type(acl["security_descriptor_bytes"]) is int
                and acl["security_descriptor_bytes"] > 0
                and acl["validated_before_third_party_site_or_native_import"] is True
                and acl["pinned_stdlib_native_modules_loaded_before_hold"] is True,
                f"{context}_early_acl_rejected",
            )
            early_owner = str(acl["owner_sid"])
        _require(
            type(early["schema_version"]) is int
            and early["schema_version"] == 1
            and early["kind"] == "explainiverse-operator-early-runtime-boundary"
            and all(
                early[field_name] is True
                for field_name in (
                    "all_runtime_and_receipt_roots_owner_private",
                    "all_runtime_and_receipt_paths_held_without_write_or_delete_share",
                    "validated_before_third_party_site_or_native_import",
                    "pinned_official_python_runtime_is_the_pre_hold_trust_boundary",
                    "working_directory_repository_disjoint",
                )
            )
            and early["working_directory"] == working_directory
            and type(early_evidence) is str
            and _sha(_canonical(early_material)) == early_evidence,
            f"{context}_early_boundary_rejected",
        )

        sealed = value["sealed_resources"]
        sealed_keys = {
            "schema_version",
            "kind",
            "policy_sha256",
            "controller_source_sha256",
            "runtime_bundle_sha256",
            "runtime_file_sha256",
            "captured_before_project_import",
            "live_repository_reopen_permitted",
        }
        runtime_file_sha256 = sealed.get("runtime_file_sha256") if type(sealed) is dict else None
        _require(
            type(sealed) is dict
            and set(sealed) == sealed_keys
            and type(sealed["schema_version"]) is int
            and sealed["schema_version"] == 1
            and sealed["kind"] == "explainiverse-operator-sealed-resource-binding"
            and sealed["policy_sha256"] == controller_resources.policy_sha256
            and sealed["controller_source_sha256"] == controller_resources.controller_source_sha256
            and sealed["runtime_bundle_sha256"] == immutable_plan["remote_runtime"]["bundle_sha256"]
            and type(runtime_file_sha256) is dict
            and set(runtime_file_sha256) == set(live.RUNTIME_BUNDLE_NAMES)
            and all(
                type(item) is str and SHA256_RE.fullmatch(item) is not None
                for item in runtime_file_sha256.values()
            )
            and sealed["captured_before_project_import"] is True
            and sealed["live_repository_reopen_permitted"] is False,
            f"{context}_sealed_resources_rejected",
        )
        material = dict(value)
        evidence_sha256 = material.pop("evidence_sha256", None)
        _require(
            type(evidence_sha256) is str and _sha(_canonical(material)) == evidence_sha256,
            f"{context}_digest_rejected",
        )

    @staticmethod
    def _validate_immutable_plan_mapping(
        payload: Mapping[str, Any], *, context: str
    ) -> live.ImmutablePlan:
        """Rebuild and exact-compare the complete historical provider plan."""

        _require(type(payload) is dict, f"{context}_not_object")
        target = payload.get("target")
        ssh_access = payload.get("ssh_access")
        remote_runtime = payload.get("remote_runtime")
        _require(
            type(target) is dict
            and type(target.get("image")) is dict
            and type(ssh_access) is dict
            and type(remote_runtime) is dict
            and type(payload.get("original_global_rules")) is list,
            f"{context}_shape_rejected",
        )
        assert isinstance(target, dict)
        image = target["image"]
        assert isinstance(image, dict)
        assert isinstance(ssh_access, dict)
        assert isinstance(remote_runtime, dict)
        try:
            plan = live.build_immutable_plan(
                head_sha=payload["head_sha"],
                lifecycle_nonce=payload["lifecycle_nonce"],
                created_at_unix=payload["created_at_unix"],
                expires_at_unix=payload["expires_at_unix"],
                current_public_ipv4_cidr=payload["controller_source"],
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
                ssh_key_name=ssh_access["key_name"],
                ssh_public_key_sha256=ssh_access["public_key_sha256"],
                baseline_file_systems_sha256=payload["baseline_file_systems_sha256"],
                original_global_rules=payload["original_global_rules"],
                host_key_fingerprint=ssh_access["ephemeral_host_key_fingerprint"],
                runtime_bundle_sha256=remote_runtime["bundle_sha256"],
            )
        except (KeyError, TypeError):
            raise ControllerError(f"{context}_shape_rejected") from None
        _require(plan.to_mapping() == payload, f"{context}_exact_mapping_rejected")
        return plan

    @staticmethod
    def _validate_accepted_evidence_root_inventory(
        root: Path,
        *,
        operator_preflight_filename: str,
        app_archives: Sequence[Mapping[str, Any]],
        journal_recovery_payloads: Sequence[Mapping[str, Any]] = (),
        context: str,
    ) -> None:
        """Reject every unbound file, directory, temporary, or reserve byte.

        The accepted lifecycle is the point at which no emergency provider
        intent may remain outside the numbered journal.  Each archived App
        capture is therefore also treated as an exact directory inventory,
        rather than merely validating the files later selected by a manifest.
        """

        _require(
            root.is_absolute()
            and root == root.resolve(strict=True)
            and root.is_dir()
            and not root.is_symlink(),
            f"{context}_root_rejected",
        )
        journal_paths = EvidenceJournal._journal_paths(root)
        journal_names = {path.name for path in journal_paths}
        _require(
            len(journal_names) == len(journal_paths),
            f"{context}_journal_name_reused",
        )
        sidecar_names: set[str] = set()
        for payload in journal_recovery_payloads:
            sidecar_name = payload.get("sidecar_filename")
            _require(
                type(sidecar_name) is str
                and LOCAL_RECOVERY_SIDECAR_RE.fullmatch(sidecar_name) is not None
                and sidecar_name not in sidecar_names,
                f"{context}_recovery_sidecar_manifest_rejected",
            )
            assert isinstance(sidecar_name, str)
            sidecar_names.add(sidecar_name)
        expected_root_files = {
            *journal_names,
            *sidecar_names,
            EMERGENCY_EVIDENCE_FILENAME,
            "known_hosts",
            operator_preflight_filename,
        }
        expected_root_directories = {"installed-app-pages"}
        actual_root_files: set[str] = set()
        actual_root_directories: set[str] = set()
        for path in root.iterdir():
            _require(
                path.parent == root and not path.is_symlink() and path.resolve(strict=True) == path,
                f"{context}_root_entry_rejected",
            )
            if path.is_file():
                _require(path.stat().st_nlink == 1, f"{context}_root_file_link_rejected")
                actual_root_files.add(path.name)
            elif path.is_dir():
                actual_root_directories.add(path.name)
            else:
                raise ControllerError(f"{context}_root_entry_type_rejected")
        _require(
            actual_root_files == expected_root_files
            and actual_root_directories == expected_root_directories,
            f"{context}_root_inventory_rejected",
        )

        reserve = root / EMERGENCY_EVIDENCE_FILENAME
        expected_reserve_size = EMERGENCY_EVIDENCE_SLOT_SIZE * EMERGENCY_EVIDENCE_SLOT_COUNT
        reserve_raw = reserve.read_bytes()
        _require(
            reserve.is_file()
            and not reserve.is_symlink()
            and reserve.resolve(strict=True) == reserve
            and reserve.stat().st_nlink == 1
            and len(reserve_raw) == expected_reserve_size
            and reserve_raw == b"\0" * expected_reserve_size,
            f"{context}_emergency_reserve_not_empty",
        )

        pages_root = root / "installed-app-pages"
        _require(
            pages_root.is_dir()
            and not pages_root.is_symlink()
            and pages_root.resolve(strict=True) == pages_root,
            f"{context}_app_archive_root_rejected",
        )
        expected_capture_directories: dict[str, set[str]] = {}
        for archive in app_archives:
            capture_sha256 = archive.get("capture_evidence_sha256")
            stale_identity_sha256 = archive.get("archive_identity_sha256")
            archive_identity_sha256 = (
                capture_sha256 if type(capture_sha256) is str else stale_identity_sha256
            )
            archive_directory = archive.get("archive_directory")
            files = archive.get("files")
            _require(
                type(archive_identity_sha256) is str
                and SHA256_RE.fullmatch(archive_identity_sha256) is not None
                and (type(capture_sha256) is str) != (type(stale_identity_sha256) is str)
                and archive_directory == f"installed-app-pages/{archive_identity_sha256}"
                and type(files) is list
                and bool(files)
                and archive_identity_sha256 not in expected_capture_directories,
                f"{context}_app_archive_manifest_rejected",
            )
            assert isinstance(archive_identity_sha256, str)
            assert isinstance(files, list)
            filenames: set[str] = set()
            for item in files:
                _require(
                    type(item) is dict
                    and set(item) == {"filename", "bytes", "sha256"}
                    and type(item["filename"]) is str
                    and Path(item["filename"]).name == item["filename"]
                    and item["filename"] not in {".", ".."}
                    and item["filename"] not in filenames
                    and type(item["bytes"]) is int
                    and item["bytes"] > 0
                    and type(item["sha256"]) is str
                    and SHA256_RE.fullmatch(item["sha256"]) is not None,
                    f"{context}_app_archive_file_manifest_rejected",
                )
                filenames.add(str(item["filename"]))
            expected_capture_directories[archive_identity_sha256] = filenames
        _require(
            {path.name for path in pages_root.iterdir()} == set(expected_capture_directories),
            f"{context}_app_archive_directory_inventory_rejected",
        )
        for capture_sha256, filenames in expected_capture_directories.items():
            capture_root = pages_root / capture_sha256
            _require(
                capture_root.is_dir()
                and not capture_root.is_symlink()
                and capture_root.resolve(strict=True) == capture_root,
                f"{context}_app_archive_capture_root_rejected",
            )
            actual_files: set[str] = set()
            for path in capture_root.iterdir():
                _require(
                    path.is_file()
                    and not path.is_symlink()
                    and path.resolve(strict=True) == path
                    and path.parent == capture_root
                    and path.stat().st_nlink == 1,
                    f"{context}_app_archive_entry_rejected",
                )
                actual_files.add(path.name)
            _require(
                actual_files == filenames,
                f"{context}_app_archive_file_inventory_rejected",
            )

    @staticmethod
    def _validate_accepted_phase_event_grammar(
        entries: Sequence[tuple[str, Mapping[str, Any]]],
        *,
        phase: str,
        evidence_root: Path,
        controller_resources: SealedControllerResources,
        journal_recovery_payloads: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        """Validate the one truthful, ordered success trace accepted for release.

        This is deliberately a state machine, not an allowlist.  In
        particular, a successful ephemeral runner removes itself, so a runner
        DELETE WAL is evidence of a recovery branch and is forbidden here.
        Provider ambiguity is accepted only when the exact ambiguity is
        followed by its inventory observation and exactly-once recovery.
        """

        _require(phase in {"final-main", "publication"}, "accepted_phase_rejected")
        _require(type(entries) in {list, tuple} and bool(entries), f"{phase}_journal_empty")
        cursor = 0

        def take(expected: str) -> Mapping[str, Any]:
            nonlocal cursor
            _require(cursor < len(entries), f"{phase}_journal_event_missing_{expected}")
            label, payload = entries[cursor]
            _require(
                label == expected and type(payload) is dict,
                f"{phase}_journal_event_order_rejected",
            )
            cursor += 1
            return payload

        _require(
            isinstance(evidence_root, Path)
            and evidence_root.is_absolute()
            and evidence_root == evidence_root.resolve(strict=True),
            f"{phase}_evidence_root_rejected",
        )
        directory_receipt = take("evidence-directory")
        _require(
            type(directory_receipt.get("receipt_sha256")) is str
            and SHA256_RE.fullmatch(directory_receipt["receipt_sha256"]) is not None,
            f"{phase}_evidence_directory_receipt_rejected",
        )
        evidence_directory_receipt_sha256 = str(directory_receipt["receipt_sha256"])
        immutable_plan = take("immutable-plan")
        validated_plan = EvidenceJournal._validate_immutable_plan_mapping(
            immutable_plan,
            context=f"{phase}_immutable_plan",
        )
        plan_sha256 = validated_plan.sha256
        plan_ssh = immutable_plan["ssh_access"]
        plan_remote = immutable_plan["remote_runtime"]
        assert isinstance(plan_ssh, dict)
        assert isinstance(plan_remote, dict)
        sink = take("provider-intent-sink-bound")
        _require(
            set(sink)
            == {
                "plan_sha256",
                "binding_sha256",
                "sink",
                "bound_before_observation",
                "recovery_process",
            }
            and sink["plan_sha256"] == plan_sha256
            and type(sink["binding_sha256"]) is str
            and SHA256_RE.fullmatch(sink["binding_sha256"]) is not None
            and sink["sink"] == "evidence-journal"
            and sink["bound_before_observation"] is True
            and sink["recovery_process"] is False,
            f"{phase}_provider_sink_binding_rejected",
        )
        binding_sha256 = str(sink["binding_sha256"])
        ssh_executable = take("ssh-executable")
        executable_keys = {
            "absolute_path",
            "sha256",
            "regular_file",
            "symlink",
            "path_lookup_used",
        }
        _require(
            set(ssh_executable) == executable_keys
            and type(ssh_executable["absolute_path"]) is str
            and _is_windows_absolute_path(ssh_executable["absolute_path"])
            and type(ssh_executable["sha256"]) is str
            and SHA256_RE.fullmatch(ssh_executable["sha256"]) is not None
            and ssh_executable["regular_file"] is True
            and ssh_executable["symlink"] is False
            and ssh_executable["path_lookup_used"] is False,
            f"{phase}_ssh_executable_receipt_rejected",
        )
        github_executable = take("github-executable")
        _require(
            set(github_executable)
            == executable_keys
            | {
                "hostname_pinned",
                "child_environment_names",
                "ambient_token_environment_forwarded",
            }
            and type(github_executable["absolute_path"]) is str
            and _is_windows_absolute_path(github_executable["absolute_path"])
            and type(github_executable["sha256"]) is str
            and SHA256_RE.fullmatch(github_executable["sha256"]) is not None
            and github_executable["regular_file"] is True
            and github_executable["symlink"] is False
            and github_executable["path_lookup_used"] is False
            and github_executable["hostname_pinned"] == "github.com"
            and type(github_executable["child_environment_names"]) is list
            and all(type(name) is str for name in github_executable["child_environment_names"])
            and github_executable["child_environment_names"]
            == sorted(set(github_executable["child_environment_names"]))
            and github_executable["ambient_token_environment_forwarded"] is False,
            f"{phase}_github_executable_receipt_rejected",
        )
        access_identity = take("ssh-access-identity")
        access_capture = access_identity.get("capture")
        access_validation = access_identity.get("validation")
        _require(
            set(access_identity)
            == {
                "capture",
                "validation",
                "private_path_archived",
                "private_digest_archived",
            }
            and type(access_capture) is dict
            and set(access_capture)
            == {
                "captured_at",
                "public_key_sha256",
                "public_key_fingerprint",
                "key_type",
                "private_file_bytes",
                "private_digest_recorded",
                "absolute_path_redacted",
                "file_identity_recorded",
                "single_link",
                "no_reparse_or_symlink",
                "acl",
            }
            and type(access_validation) is dict
            and set(access_validation)
            == {
                "validated_at",
                "public_key_sha256",
                "public_key_fingerprint",
                "private_file_bytes",
                "private_digest_recorded",
                "absolute_path_redacted",
                "file_identity_recorded",
                "single_link",
                "no_reparse_or_symlink",
                "acl_evidence_sha256",
            }
            and access_capture["public_key_sha256"] == plan_ssh["public_key_sha256"]
            and access_validation["public_key_sha256"] == plan_ssh["public_key_sha256"]
            and access_capture["public_key_fingerprint"]
            == access_validation["public_key_fingerprint"]
            and access_capture["key_type"] == "ssh-ed25519"
            and type(access_capture["private_file_bytes"]) is int
            and access_capture["private_file_bytes"] > 0
            and access_validation["private_file_bytes"] == access_capture["private_file_bytes"]
            and all(
                access_capture[field_name] is True
                for field_name in (
                    "private_digest_recorded",
                    "absolute_path_redacted",
                    "file_identity_recorded",
                    "single_link",
                    "no_reparse_or_symlink",
                )
            )
            and all(
                access_validation[field_name] is True
                for field_name in (
                    "private_digest_recorded",
                    "absolute_path_redacted",
                    "file_identity_recorded",
                    "single_link",
                    "no_reparse_or_symlink",
                )
            )
            and access_identity["private_path_archived"] is False
            and access_identity["private_digest_archived"] is False,
            f"{phase}_ssh_access_identity_receipt_rejected",
        )
        assert isinstance(access_capture, dict)
        assert isinstance(access_validation, dict)
        access_acl = access_capture["acl"]
        access_acl_keys = {
            "owner_sid",
            "current_user_sid",
            "inheritance_protected",
            "aces",
            "security_descriptor_sha256",
            "security_descriptor_bytes",
            "captured_at",
            "evidence_sha256",
        }
        access_aces = access_acl.get("aces") if type(access_acl) is dict else None
        _require(
            type(access_acl) is dict
            and set(access_acl) == access_acl_keys
            and type(access_acl["owner_sid"]) is str
            and re.fullmatch(r"S-1-[0-9-]+", access_acl["owner_sid"]) is not None
            and access_acl["current_user_sid"] == access_acl["owner_sid"]
            and access_acl["inheritance_protected"] is True
            and type(access_aces) is list
            and len(access_aces) == 3
            and access_aces == sorted(access_aces, key=lambda item: str(item.get("sid")))
            and {item["sid"] for item in access_aces if type(item) is dict and "sid" in item}
            == {
                access_acl["owner_sid"],
                "S-1-5-18",
                "S-1-5-32-544",
            }
            and all(
                type(item) is dict
                and set(item) == {"sid", "access", "rights", "mask", "ace_flags"}
                and item["access"] == "allow"
                and item["rights"] == "full-control"
                and type(item["mask"]) is int
                and item["mask"] == 2_032_127
                and type(item["ace_flags"]) is int
                and item["ace_flags"] == 0
                for item in access_aces
            )
            and type(access_acl.get("captured_at")) is str
            and _public_receipt_time(access_acl["captured_at"], "ssh_access_acl_captured")
            <= _public_receipt_time(access_validation["validated_at"], "ssh_access_acl_validated")
            and type(access_acl["security_descriptor_sha256"]) is str
            and SHA256_RE.fullmatch(access_acl["security_descriptor_sha256"]) is not None
            and type(access_acl["security_descriptor_bytes"]) is int
            and access_acl["security_descriptor_bytes"] > 0
            and type(access_acl.get("evidence_sha256")) is str
            and _sha(
                _canonical(
                    {
                        key: value
                        for key, value in access_acl.items()
                        if key not in {"captured_at", "evidence_sha256"}
                    }
                )
            )
            == access_acl["evidence_sha256"]
            and access_validation["acl_evidence_sha256"] == access_acl["evidence_sha256"],
            f"{phase}_ssh_access_identity_acl_rejected",
        )
        assert isinstance(access_acl, dict)
        assert isinstance(access_aces, list)

        used_provider_requests: set[str] = set()
        used_provider_receipt_nonces: set[str] = set()

        def snapshot(expected_phase: str, expected_label: str | None = None) -> Mapping[str, Any]:
            payload = take(expected_label or f"provider-{expected_phase.replace('_', '-')}")
            EvidenceJournal._validate_provider_snapshot(
                payload,
                plan_sha256=plan_sha256,
                expected_phase=expected_phase,
                context=f"{phase}_provider_{expected_phase}_snapshot_rejected",
            )
            nonce = str(payload["receipt_nonce"])
            _require(
                nonce not in used_provider_receipt_nonces,
                f"{phase}_provider_snapshot_receipt_replayed",
            )
            used_provider_receipt_nonces.add(nonce)
            return payload

        def transition(
            operation: str,
            prestate: Mapping[str, Any],
            next_phase: str,
        ) -> Mapping[str, Any]:
            operation_label = operation.replace("_", "-")
            local = take(f"provider-{operation_label}-intent")
            _require(
                set(local) == {"plan_sha256", "operation", "prestate", "mutation_retried"}
                and local["plan_sha256"] == plan_sha256
                and local["operation"] == operation
                and local["prestate"] == prestate
                and local["mutation_retried"] is False,
                f"{phase}_provider_{operation}_local_intent_rejected",
            )
            intent = live.MutationIntent.from_public_mapping(take("provider-mutation-intent"))
            if operation == "delete_ruleset":
                _require(
                    type(prestate.get("ruleset_id")) is str
                    and intent.path
                    == live.MUTATION_PATHS[operation][1].replace(
                        "{id}", str(prestate["ruleset_id"])
                    ),
                    f"{phase}_provider_delete_ruleset_target_rejected",
                )
            _require(
                intent.plan_sha256 == plan_sha256
                and intent.operation == operation
                and intent.prestate_phase == prestate["phase"]
                and intent.prestate_snapshot_sha256 == prestate["snapshot_sha256"]
                and intent.prestate_receipt_nonce == prestate["receipt_nonce"]
                and intent.callback_binding_sha256 == binding_sha256
                and intent.request_sha256 not in used_provider_requests,
                f"{phase}_provider_{operation}_intent_binding_rejected",
            )
            used_provider_requests.add(intent.request_sha256)
            next_label = entries[cursor][0] if cursor < len(entries) else ""
            direct_receipt: Mapping[str, Any] | None = None
            recovery_receipt: Mapping[str, Any] | None = None
            recovery_snapshot: Mapping[str, Any] | None = None
            if next_label == f"provider-{operation_label}":
                receipt = take(f"provider-{operation_label}")
                _require(
                    set(receipt)
                    == {
                        "plan_sha256",
                        "operation",
                        "prestate_snapshot_sha256",
                        "request_sha256",
                        "request_body_sha256",
                        "response_body_sha256",
                        "response_body_redacted",
                        "ruleset_id",
                        "instance_id",
                        "host_fingerprint",
                    }
                    and receipt["plan_sha256"] == plan_sha256
                    and receipt["operation"] == operation
                    and receipt["prestate_snapshot_sha256"] == prestate["snapshot_sha256"]
                    and receipt["request_sha256"] == intent.request_sha256
                    and receipt["request_body_sha256"] == intent.request_body_sha256
                    and type(receipt["response_body_sha256"]) is str
                    and SHA256_RE.fullmatch(receipt["response_body_sha256"]) is not None
                    and receipt["response_body_redacted"] is True,
                    f"{phase}_provider_{operation}_receipt_rejected",
                )
                direct_receipt = receipt
            else:
                ambiguity = take(f"provider-{operation_label}-ambiguous")
                _require(
                    set(ambiguity)
                    == {"operation", "request_sha256", "reason_code", "mutation_retried"}
                    and ambiguity["operation"] == operation
                    and ambiguity["request_sha256"] == intent.request_sha256
                    and type(ambiguity["reason_code"]) is str
                    and bool(ambiguity["reason_code"])
                    and ambiguity["mutation_retried"] is False,
                    f"{phase}_provider_{operation}_ambiguity_rejected",
                )
                recovery_snapshot = snapshot("recovery", "provider-recovery")
                recovery = take(f"provider-{operation_label}-recovery")
                _require(
                    set(recovery)
                    == {
                        "plan_sha256",
                        "operation",
                        "ambiguous_request_sha256",
                        "inventory_snapshot_sha256",
                        "outcome",
                        "ruleset_id",
                        "instance_id",
                    }
                    and recovery["plan_sha256"] == plan_sha256
                    and recovery["operation"] == operation
                    and recovery["ambiguous_request_sha256"] == intent.request_sha256
                    and recovery["inventory_snapshot_sha256"]
                    == recovery_snapshot["snapshot_sha256"]
                    and recovery["outcome"] in {"applied_exactly_once", "applied_in_progress"}
                    and (
                        recovery["outcome"] != "applied_in_progress"
                        or operation in {"launch", "terminate"}
                    ),
                    f"{phase}_provider_{operation}_recovery_rejected",
                )
                recovery_receipt = recovery
            poststate = snapshot(next_phase)
            plan_fingerprint = plan_ssh["ephemeral_host_key_fingerprint"]
            expected_resources: dict[str, tuple[Any, Any, Any]] = {
                "restrict_global": (None, None, None),
                "create_ruleset": (poststate["ruleset_id"], None, None),
                "launch": (
                    prestate["ruleset_id"],
                    poststate["instance_id"],
                    plan_fingerprint,
                ),
                "terminate": (
                    prestate["ruleset_id"],
                    prestate["instance_id"],
                    None,
                ),
                "delete_ruleset": (prestate["ruleset_id"], None, None),
                "restore_global": (None, None, None),
            }
            if direct_receipt is not None:
                _require(
                    (
                        direct_receipt["ruleset_id"],
                        direct_receipt["instance_id"],
                        direct_receipt["host_fingerprint"],
                    )
                    == expected_resources[operation]
                    and (operation != "launch" or prestate["ruleset_id"] == poststate["ruleset_id"])
                    and (
                        operation != "terminate"
                        or prestate["ruleset_id"] == poststate["ruleset_id"]
                    ),
                    f"{phase}_provider_{operation}_resource_binding_rejected",
                )
            else:
                assert recovery_receipt is not None and recovery_snapshot is not None
                outcome = recovery_receipt["outcome"]
                if outcome == "applied_exactly_once":
                    expected_recovery_resources = (
                        poststate["ruleset_id"],
                        poststate["instance_id"],
                    )
                    _require(
                        recovery_snapshot["ruleset_id"] == poststate["ruleset_id"]
                        and recovery_snapshot["instance_id"] == poststate["instance_id"]
                        and recovery_snapshot["instance_public_ipv4"]
                        == poststate["instance_public_ipv4"],
                        f"{phase}_provider_{operation}_recovery_snapshot_drift",
                    )
                elif operation == "launch":
                    expected_recovery_resources = (
                        poststate["ruleset_id"],
                        poststate["instance_id"],
                    )
                else:
                    expected_recovery_resources = (
                        prestate["ruleset_id"],
                        prestate["instance_id"],
                    )
                _require(
                    (
                        recovery_receipt["ruleset_id"],
                        recovery_receipt["instance_id"],
                    )
                    == expected_recovery_resources
                    and (operation != "launch" or prestate["ruleset_id"] == poststate["ruleset_id"])
                    and (
                        operation != "terminate"
                        or prestate["ruleset_id"] == poststate["ruleset_id"]
                    ),
                    f"{phase}_provider_{operation}_recovery_resource_binding_rejected",
                )
            return poststate

        current = snapshot("baseline")
        current = transition("restrict_global", current, "global_restricted")
        current = transition("create_ruleset", current, "ruleset_ready")
        current = transition("launch", current, "instance_bound")
        launched_instance = dict(current)
        known_hosts = take("known-hosts")
        _require(
            set(known_hosts)
            == {
                "absolute_path",
                "content_sha256",
                "evidence_directory_acl_receipt_sha256",
                "public_ipv4",
                "host_fingerprint",
                "content_is_public",
            }
            and type(known_hosts["absolute_path"]) is str
            and Path(known_hosts["absolute_path"]) == evidence_root / "known_hosts"
            and type(known_hosts["content_sha256"]) is str
            and SHA256_RE.fullmatch(known_hosts["content_sha256"]) is not None
            and known_hosts["evidence_directory_acl_receipt_sha256"]
            == evidence_directory_receipt_sha256
            and known_hosts["public_ipv4"] == launched_instance["instance_public_ipv4"]
            and known_hosts["host_fingerprint"] == plan_ssh["ephemeral_host_key_fingerprint"]
            and known_hosts["content_is_public"] is True,
            f"{phase}_known_hosts_receipt_rejected",
        )
        known_hosts_path = Path(str(known_hosts["absolute_path"]))
        _require(
            known_hosts_path.is_file()
            and not known_hosts_path.is_symlink()
            and known_hosts_path.stat().st_nlink == 1,
            f"{phase}_known_hosts_file_rejected",
        )
        known_hosts_bytes = known_hosts_path.read_bytes()
        _require(
            _sha(known_hosts_bytes) == known_hosts["content_sha256"]
            and known_hosts_bytes.endswith(b"\n")
            and known_hosts_bytes.count(b"\n") == 1,
            f"{phase}_known_hosts_content_rejected",
        )
        try:
            known_hosts_text = known_hosts_bytes.decode("ascii").rstrip("\n")
        except UnicodeDecodeError:
            raise ControllerError(f"{phase}_known_hosts_encoding_rejected") from None
        _require(
            known_hosts_text.startswith(
                f"{launched_instance['instance_public_ipv4']} ssh-ed25519 "
            ),
            f"{phase}_known_hosts_target_rejected",
        )

        ssh_bindings: dict[str, Mapping[str, Any]] = {}
        command_fields = {
            "cloud-init": "fixed_cloud_init_wait_command",
            "preflight": "fixed_preflight_command",
            "run": "fixed_command",
        }
        for mode, command_field in command_fields.items():
            binding = take(f"ssh-{mode}")
            command = plan_remote[command_field]
            expected_argv = [
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
                f"ubuntu@{launched_instance['instance_public_ipv4']}",
                *command,
            ]
            _require(
                set(binding)
                == {
                    "argv_prefix",
                    "access_identity_file_redacted",
                    "known_hosts",
                    "known_hosts_path",
                    "known_hosts_sha256",
                    "evidence_directory_acl_receipt_sha256",
                    "host_fingerprint",
                    "trust_on_first_use",
                    "remote_mode",
                    "fixed_remote_command",
                }
                and binding["argv_prefix"] == expected_argv
                and binding["access_identity_file_redacted"] is True
                and binding["known_hosts"] == known_hosts_text + "\n"
                and binding["known_hosts_path"] == str(known_hosts_path)
                and binding["known_hosts_sha256"] == known_hosts["content_sha256"]
                and binding["evidence_directory_acl_receipt_sha256"]
                == evidence_directory_receipt_sha256
                and binding["host_fingerprint"] == plan_ssh["ephemeral_host_key_fingerprint"]
                and binding["trust_on_first_use"] is False
                and binding["remote_mode"] == mode
                and binding["fixed_remote_command"] == command,
                f"{phase}_ssh_{mode}_receipt_rejected",
            )
            ssh_bindings[mode] = binding

        def require_bound_instance(payload: Mapping[str, Any], *, context: str) -> None:
            _require(
                payload.get("ruleset_id") == launched_instance["ruleset_id"]
                and payload.get("instance_id") == launched_instance["instance_id"]
                and payload.get("instance_public_ipv4")
                == launched_instance["instance_public_ipv4"],
                f"{phase}_{context}_instance_binding_rejected",
            )

        def validate_readiness(
            payload: Mapping[str, Any], *, context: str, observations: Sequence[Mapping[str, Any]]
        ) -> None:
            expected_keys = {
                "cloud_init",
                "host_preflight",
                "cloud_init_sha256",
                "preflight_sha256",
                "cloud_binding",
                "preflight_binding",
                "ssh_attempts",
                "readiness_evidence_sha256",
            }
            cloud = payload.get("cloud_init")
            preflight = payload.get("host_preflight")
            ssh_attempts = payload.get("ssh_attempts")
            cloud_keys = {
                "schema_version",
                "kind",
                "plan_sha256",
                "provider_snapshot_sha256",
                "provider_receipt_nonce",
                "instance_id",
                "instance_public_ipv4",
                "host_fingerprint",
                "known_hosts_sha256",
                "observed_at",
                "exit_code",
                "stdout_sha256",
                "stderr_sha256",
                "fixed_command",
                "credential_received",
                "jit_config_received",
                "binding_sha256",
            }
            preflight_keys = {
                "schema_version",
                "kind",
                "plan_sha256",
                "provider_snapshot_sha256",
                "provider_receipt_nonce",
                "instance_id",
                "instance_public_ipv4",
                "host_fingerprint",
                "known_hosts_sha256",
                "cloud_init_wait_binding_sha256",
                "remote_response_sha256",
                "observed_at",
                "runtime_bundle_sha256",
                "host_physical_gpu_count",
                "host_physical_gpu_uuids",
                "host_physical_gpu_products",
                "image_probe_sha256",
                "gpu_injection",
                "fixed_preflight_command",
                "jit_config_received",
                "github_api_credential_received",
                "accepted_actions_evidence",
            }
            _require(
                set(payload) == expected_keys
                and type(cloud) is dict
                and set(cloud) == cloud_keys
                and type(preflight) is dict
                and set(preflight) == preflight_keys
                and cloud["schema_version"] == 1
                and type(cloud["schema_version"]) is int
                and cloud["kind"] == "explainiverse-lambda-live-cloud-init-wait-binding"
                and cloud["plan_sha256"] == plan_sha256
                and cloud["instance_id"] == launched_instance["instance_id"]
                and cloud["instance_public_ipv4"] == launched_instance["instance_public_ipv4"]
                and cloud["host_fingerprint"] == plan_ssh["ephemeral_host_key_fingerprint"]
                and cloud["known_hosts_sha256"] == known_hosts["content_sha256"]
                and cloud["exit_code"] == 0
                and type(cloud["exit_code"]) is int
                and cloud["fixed_command"] == plan_remote["fixed_cloud_init_wait_command"]
                and cloud["credential_received"] is False
                and cloud["jit_config_received"] is False
                and type(cloud["binding_sha256"]) is str
                and _sha(
                    _canonical(
                        {key: value for key, value in cloud.items() if key != "binding_sha256"}
                    )
                )
                == cloud["binding_sha256"]
                and preflight["schema_version"] == 1
                and type(preflight["schema_version"]) is int
                and preflight["kind"] == "explainiverse-lambda-live-host-preflight-binding"
                and preflight["plan_sha256"] == plan_sha256
                and preflight["instance_id"] == launched_instance["instance_id"]
                and preflight["instance_public_ipv4"] == launched_instance["instance_public_ipv4"]
                and preflight["host_fingerprint"] == plan_ssh["ephemeral_host_key_fingerprint"]
                and preflight["known_hosts_sha256"] == known_hosts["content_sha256"]
                and preflight["cloud_init_wait_binding_sha256"] == cloud["binding_sha256"]
                and preflight["runtime_bundle_sha256"] == plan_remote["bundle_sha256"]
                and type(preflight["host_physical_gpu_count"]) is int
                and preflight["host_physical_gpu_count"] == 8
                and type(preflight["host_physical_gpu_uuids"]) is list
                and len(preflight["host_physical_gpu_uuids"]) == 8
                and len(set(preflight["host_physical_gpu_uuids"])) == 8
                and type(preflight["host_physical_gpu_products"]) is list
                and preflight["host_physical_gpu_products"] == ["NVIDIA A100-SXM4-80GB"] * 8
                and preflight["fixed_preflight_command"] == plan_remote["fixed_preflight_command"]
                and preflight["jit_config_received"] is False
                and preflight["github_api_credential_received"] is False
                and preflight["accepted_actions_evidence"] is False
                and payload["cloud_init_sha256"] == _sha(_canonical(cloud))
                and payload["preflight_sha256"] == _sha(_canonical(preflight))
                and payload["cloud_binding"] == ssh_bindings["cloud-init"]
                and payload["preflight_binding"] == ssh_bindings["preflight"]
                and type(ssh_attempts) is dict
                and set(ssh_attempts) == {"cloud_init", "preflight"}
                and type(ssh_attempts["cloud_init"]) is list
                and bool(ssh_attempts["cloud_init"])
                and type(ssh_attempts["preflight"]) is list
                and bool(ssh_attempts["preflight"]),
                f"{phase}_{context}_receipt_rejected",
            )
            assert isinstance(cloud, dict)
            assert isinstance(preflight, dict)
            assert isinstance(ssh_attempts, dict)
            readiness_material = {
                "schema_version": 1,
                "kind": "explainiverse-lambda-host-readiness-binding",
                "control_plane_plan_sha256": plan_sha256,
                "instance_id": cloud["instance_id"],
                "instance_public_ipv4": cloud["instance_public_ipv4"],
                "host_fingerprint": cloud["host_fingerprint"],
                "known_hosts_sha256": cloud["known_hosts_sha256"],
                "cloud_init": cloud,
                "preflight": preflight,
                "cloud_init_sha256": payload["cloud_init_sha256"],
                "preflight_sha256": payload["preflight_sha256"],
                "cloud_binding": payload["cloud_binding"],
                "preflight_binding": payload["preflight_binding"],
                "ssh_attempts": ssh_attempts,
            }
            _require(
                type(payload["readiness_evidence_sha256"]) is str
                and payload["readiness_evidence_sha256"] == _sha(_canonical(readiness_material)),
                f"{phase}_{context}_digest_rejected",
            )
            observation_cursor = 0
            accepted_observations: dict[str, Mapping[str, Any]] = {}
            for group_name in ("cloud_init", "preflight"):
                attempts = ssh_attempts[group_name]
                for attempt_index, attempt in enumerate(attempts, start=1):
                    _require(
                        type(attempt) is dict
                        and type(attempt.get("attempt")) is int
                        and attempt["attempt"] == attempt_index,
                        f"{phase}_{context}_{group_name}_attempt_rejected",
                    )
                    if "transport_error" in attempt:
                        _require(
                            set(attempt) == {"attempt", "transport_error"}
                            and type(attempt["transport_error"]) is str
                            and bool(attempt["transport_error"])
                            and observation_cursor < len(observations),
                            f"{phase}_{context}_{group_name}_transport_attempt_rejected",
                        )
                        observation_cursor += 1
                        continue
                    common_attempt_keys = {
                        "attempt",
                        "stdout_sha256",
                        "stderr_sha256",
                        "exit_code",
                        "provider_snapshot_sha256",
                    }
                    outcome_keys = set(attempt) - common_attempt_keys
                    _require(
                        outcome_keys in ({"accepted"}, {"validation_error"})
                        and all(
                            type(attempt[field_name]) is str
                            and SHA256_RE.fullmatch(attempt[field_name]) is not None
                            for field_name in ("stdout_sha256", "stderr_sha256")
                        )
                        and type(attempt["exit_code"]) is int
                        and type(attempt["provider_snapshot_sha256"]) is str
                        and SHA256_RE.fullmatch(attempt["provider_snapshot_sha256"]) is not None
                        and observation_cursor + 1 < len(observations)
                        and attempt["provider_snapshot_sha256"]
                        == observations[observation_cursor + 1]["snapshot_sha256"]
                        and (
                            (attempt.get("accepted") is True and attempt_index == len(attempts))
                            or (
                                type(attempt.get("validation_error")) is str
                                and bool(attempt["validation_error"])
                            )
                        ),
                        f"{phase}_{context}_{group_name}_result_attempt_rejected",
                    )
                    if attempt.get("accepted") is True:
                        accepted_observations[group_name] = observations[observation_cursor + 1]
                    observation_cursor += 2
            _require(
                observation_cursor == len(observations),
                f"{phase}_{context}_attempt_observation_count_rejected",
            )
            _require(
                set(accepted_observations) == {"cloud_init", "preflight"}
                and accepted_observations["cloud_init"]["snapshot_sha256"]
                == cloud["provider_snapshot_sha256"]
                and accepted_observations["cloud_init"]["receipt_nonce"]
                == cloud["provider_receipt_nonce"]
                and accepted_observations["preflight"]["snapshot_sha256"]
                == preflight["provider_snapshot_sha256"]
                and accepted_observations["preflight"]["receipt_nonce"]
                == preflight["provider_receipt_nonce"]
                and ssh_attempts["cloud_init"][-1].get("accepted") is True
                and ssh_attempts["cloud_init"][-1].get("provider_snapshot_sha256")
                == cloud["provider_snapshot_sha256"]
                and ssh_attempts["preflight"][-1].get("accepted") is True
                and ssh_attempts["preflight"][-1].get("provider_snapshot_sha256")
                == preflight["provider_snapshot_sha256"],
                f"{phase}_{context}_provider_observation_binding_rejected",
            )

        readiness_observations = 0
        readiness_provider_observations: list[Mapping[str, Any]] = []
        while cursor < len(entries) and entries[cursor][0] == "provider-instance-bound":
            observation = snapshot("instance_bound")
            require_bound_instance(observation, context="host_readiness")
            readiness_provider_observations.append(observation)
            readiness_observations += 1
        _require(
            4 <= readiness_observations <= 96,
            f"{phase}_host_readiness_observation_count_rejected",
        )
        host_readiness = take("host-readiness")
        validate_readiness(
            host_readiness,
            context="host_readiness",
            observations=readiness_provider_observations,
        )
        initial_cloud_attempts = host_readiness["ssh_attempts"]["cloud_init"]
        initial_cloud_observation_count = sum(
            1 if "transport_error" in attempt else 2 for attempt in initial_cloud_attempts
        )

        operator_binding = take("operator-preflight-binding")
        app_inbox = operator_binding.get("app_capture_inbox")
        expected_capture_count = 4 if phase == "final-main" else 2
        app_inbox_keys = {
            "phase",
            "expected_capture_count",
            "accepted_capture_count",
            "stale_generation_count",
            "stale_generations_sha256",
            "owner_private_directory_receipt_sha256",
            "on_demand_before_each_jit",
            "ready_marker_no_replace_required",
            "raw_pages_archived_by_driver",
        }
        _require(
            set(operator_binding)
            == {
                "plan_sha256",
                "operator_preflight_filename",
                "operator_preflight_sha256",
                "inspection_receipt_sha256",
                "inventory_sha256",
                "app_capture_inbox",
                "bound_before_first_jit",
            }
            and operator_binding["plan_sha256"] == plan_sha256
            and type(operator_binding["operator_preflight_filename"]) is str
            and re.fullmatch(
                r"operator-preflight-[0-9a-f]{64}\.json",
                operator_binding["operator_preflight_filename"],
            )
            is not None
            and type(operator_binding["operator_preflight_sha256"]) is str
            and SHA256_RE.fullmatch(operator_binding["operator_preflight_sha256"]) is not None
            and operator_binding["operator_preflight_filename"]
            == f"operator-preflight-{operator_binding['operator_preflight_sha256']}.json"
            and all(
                type(operator_binding[field_name]) is str
                and SHA256_RE.fullmatch(operator_binding[field_name]) is not None
                for field_name in ("inspection_receipt_sha256", "inventory_sha256")
            )
            and type(app_inbox) is dict
            and set(app_inbox) == app_inbox_keys
            and app_inbox["phase"] == phase
            and type(app_inbox["expected_capture_count"]) is int
            and app_inbox["expected_capture_count"] == expected_capture_count
            and type(app_inbox["accepted_capture_count"]) is int
            and app_inbox["accepted_capture_count"] == 0
            and type(app_inbox["stale_generation_count"]) is int
            and app_inbox["stale_generation_count"] == 0
            and type(app_inbox["stale_generations_sha256"]) is str
            and SHA256_RE.fullmatch(app_inbox["stale_generations_sha256"]) is not None
            and type(app_inbox["owner_private_directory_receipt_sha256"]) is str
            and SHA256_RE.fullmatch(app_inbox["owner_private_directory_receipt_sha256"]) is not None
            and app_inbox["on_demand_before_each_jit"] is True
            and app_inbox["ready_marker_no_replace_required"] is True
            and app_inbox["raw_pages_archived_by_driver"] is True
            and operator_binding["bound_before_first_jit"] is True,
            f"{phase}_operator_preflight_binding_rejected",
        )
        assert isinstance(app_inbox, dict)
        preflight_path = evidence_root / str(operator_binding["operator_preflight_filename"])
        _require(
            preflight_path.parent == evidence_root
            and preflight_path.is_file()
            and not preflight_path.is_symlink()
            and preflight_path.stat().st_nlink == 1,
            f"{phase}_operator_preflight_file_rejected",
        )
        preflight_raw = preflight_path.read_bytes()
        preflight_document = _json(bytearray(preflight_raw), "operator_preflight")
        preflight_keys = {
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
        _require(
            type(preflight_document) is dict
            and preflight_raw == _canonical(preflight_document)
            and _sha(preflight_raw) == operator_binding["operator_preflight_sha256"]
            and set(preflight_document) == preflight_keys
            and type(preflight_document["schema_version"]) is int
            and preflight_document["schema_version"] == 1
            and preflight_document["kind"] == "explainiverse-lambda-operator-preflight"
            and preflight_document["plan_sha256"] == plan_sha256
            and preflight_document["head_sha"] == immutable_plan["head_sha"]
            and preflight_document["lifecycle_nonce"] == immutable_plan["lifecycle_nonce"]
            and type(preflight_document["discovery"]) is dict
            and preflight_document["inspection_receipt_sha256"]
            == operator_binding["inspection_receipt_sha256"]
            and type(preflight_document["inventory"]) is dict
            and preflight_document["inventory_sha256"] == operator_binding["inventory_sha256"]
            and preflight_document["inventory_sha256"]
            == _sha(_canonical(preflight_document["inventory"]))
            and type(preflight_document["executables"]) is dict
            and type(preflight_document["repository"]) is dict
            and preflight_document["inventory"].get("executables")
            == preflight_document["executables"]
            and preflight_document["inventory"].get("repository")
            == preflight_document["repository"]
            and preflight_document["app_capture_inbox"] == app_inbox
            and (
                (phase == "final-main" and preflight_document["final_main_acceptance"] is None)
                or (
                    phase == "publication"
                    and type(preflight_document["final_main_acceptance"]) is dict
                )
            )
            and preflight_document["live_gates_not_constructed_before_confirmation"] is True
            and preflight_document["direct_publication_dispatch_exposed"] is False,
            f"{phase}_operator_preflight_file_binding_rejected",
        )
        try:
            operator_preflight_identity = operator_receipts.validate_operator_preflight(
                preflight_document,
                expected_immutable_plan=immutable_plan,
                expected_phase=phase,
                expected_head_sha=str(immutable_plan["head_sha"]),
                expected_ref=str(PHASES[phase]["run_ref"]),
                expected_plan_sha256=plan_sha256,
                expected_lifecycle_nonce=str(immutable_plan["lifecycle_nonce"]),
                expected_inspection_receipt_sha256=str(
                    operator_binding["inspection_receipt_sha256"]
                ),
                expected_inventory_sha256=str(operator_binding["inventory_sha256"]),
                expected_policy_sha256=controller_resources.policy_sha256,
                expected_controller_source_sha256=(controller_resources.controller_source_sha256),
                expected_runtime_bundle_sha256=str(plan_remote["bundle_sha256"]),
            )
        except operator_receipts.OperatorReceiptContractError as exc:
            raise ControllerError(f"{phase}_operator_preflight_contract_rejected:{exc}") from None
        _require(
            operator_preflight_identity["preflight_sha256"]
            == operator_binding["operator_preflight_sha256"]
            and operator_preflight_identity["plan_sha256"] == plan_sha256
            and operator_preflight_identity["phase"] == phase,
            f"{phase}_operator_preflight_contract_identity_rejected",
        )
        executables = preflight_document["executables"]
        _require(
            type(executables) is dict
            and type(executables.get("ssh")) is dict
            and type(executables.get("gh")) is dict
            and executables["ssh"].get("absolute_path") == ssh_executable["absolute_path"]
            and executables["ssh"].get("sha256") == ssh_executable["sha256"]
            and executables["gh"].get("absolute_path") == github_executable["absolute_path"]
            and executables["gh"].get("sha256") == github_executable["sha256"],
            f"{phase}_operator_executable_transport_binding_rejected",
        )

        dispatch_intent = take("github-dispatch-intent")
        dispatch_settlement = take("github-dispatch-settled")
        dispatch_payload = take("github-dispatch")
        session = EvidenceJournal._validate_phase_dispatch_chain(
            dispatch_intent,
            dispatch_settlement,
            dispatch_payload,
            expected_phase=phase,
            expected_head_sha=str(immutable_plan["head_sha"]),
        )

        seen_pre_jit_absence_evidence: set[str] = set()
        app_archives: list[Mapping[str, Any]] = []
        app_captures: list[Mapping[str, Any]] = []
        stale_app_archives: list[Mapping[str, Any]] = []
        authority_windows: list[Mapping[str, Any]] = []
        for ordinal, job in enumerate(session.jobs):
            if ordinal:
                refresh_observations = 0
                refresh_provider_observations: list[Mapping[str, Any]] = []
                while cursor < len(entries) and entries[cursor][0] == "provider-instance-bound":
                    observation = snapshot("instance_bound")
                    require_bound_instance(observation, context="host_refresh")
                    refresh_provider_observations.append(observation)
                    refresh_observations += 1
                _require(
                    2 <= refresh_observations <= 24,
                    f"{phase}_host_refresh_observation_count_rejected",
                )
                refresh = take("host-preflight-refresh")
                validate_readiness(
                    refresh,
                    context="host_refresh",
                    observations=[
                        *readiness_provider_observations[:initial_cloud_observation_count],
                        *refresh_provider_observations,
                    ],
                )
            while cursor < len(entries) and entries[cursor][0] == "installed-app-stale-raw-archive":
                stale_app_archives.append(take("installed-app-stale-raw-archive"))
            app_archives.append(take("installed-app-raw-archive"))
            app_captures.append(take("installed-app-authority"))
            authority_windows.append(take("authority-window"))
            absence = take("github-pre-jit-runner-absence")
            absence_material = dict(absence)
            absence_evidence_sha256 = absence_material.pop("evidence_sha256", None)
            _require(
                set(absence)
                == {
                    "phase",
                    "run_id",
                    "run_attempt",
                    "head_sha",
                    "job_key",
                    "job_id",
                    "runner_name",
                    "observed_at",
                    "response_sha256",
                    "total_count",
                    "runners",
                    "evidence_sha256",
                }
                and absence["phase"] == phase
                and type(absence["run_id"]) is int
                and absence["run_id"] == session.run["id"]
                and type(absence["run_attempt"]) is int
                and absence["run_attempt"] == 1
                and absence["head_sha"] == session.head_sha
                and absence["job_key"] == job.key
                and type(absence["job_id"]) is int
                and absence["job_id"] == job.job_id
                and absence["runner_name"] == job.runner_name
                and type(absence["observed_at"]) is str
                and type(absence["response_sha256"]) is str
                and SHA256_RE.fullmatch(absence["response_sha256"]) is not None
                and type(absence["total_count"]) is int
                and absence["total_count"] == 0
                and absence["runners"] == []
                and type(absence_evidence_sha256) is str
                and SHA256_RE.fullmatch(absence_evidence_sha256) is not None
                and absence_evidence_sha256 not in seen_pre_jit_absence_evidence
                and _sha(_canonical(absence_material)) == absence_evidence_sha256,
                f"{phase}_pre_jit_runner_absence_rejected",
            )
            seen_pre_jit_absence_evidence.add(str(absence_evidence_sha256))
            jit_intent = take("github-jit-intent")
            expected_jit_path = f"/repos/{REPOSITORY}/actions/runners/generate-jitconfig"
            jit_body = {
                "name": job.runner_name,
                "runner_group_id": 1,
                "labels": [job.runner_name],
                "work_folder": f"_work-{job.nonce}",
            }
            _require(
                set(jit_intent)
                == {
                    "phase",
                    "run_id",
                    "head_sha",
                    "job_key",
                    "job_id",
                    "job_name",
                    "runner_name",
                    "runner_nonce",
                    "path",
                    "request_sha256",
                    "runner_group_id",
                    "mutation_retried",
                }
                and jit_intent["phase"] == phase
                and type(jit_intent["run_id"]) is int
                and jit_intent["run_id"] == session.run["id"]
                and jit_intent["head_sha"] == session.head_sha
                and jit_intent["job_key"] == job.key
                and type(jit_intent["job_id"]) is int
                and jit_intent["job_id"] == job.job_id
                and jit_intent["job_name"] == job.name
                and jit_intent["runner_name"] == job.runner_name
                and jit_intent["runner_nonce"] == job.nonce
                and jit_intent["path"] == expected_jit_path
                and jit_intent["request_sha256"]
                == _sha(_canonical({"method": "POST", "path": expected_jit_path, "body": jit_body}))
                and type(jit_intent["runner_group_id"]) is int
                and jit_intent["runner_group_id"] == 1
                and jit_intent["mutation_retried"] is False,
                f"{phase}_github_jit_intent_rejected",
            )
            jit_created = take("github-jit-created")
            _require(
                set(jit_created)
                == {
                    "phase",
                    "run_id",
                    "head_sha",
                    "job_key",
                    "job_id",
                    "runner_name",
                    "jit_receipt",
                }
                and jit_created["phase"] == phase
                and type(jit_created["run_id"]) is int
                and jit_created["run_id"] == session.run["id"]
                and jit_created["head_sha"] == session.head_sha
                and jit_created["job_key"] == job.key
                and type(jit_created["job_id"]) is int
                and jit_created["job_id"] == job.job_id
                and jit_created["runner_name"] == job.runner_name
                and type(jit_created["jit_receipt"]) is dict,
                f"{phase}_github_jit_created_rejected",
            )
            runtime_plan = take("runtime-plan")
            plan_jit = runtime_plan.get("github_evidence", {}).get("jit_response")
            plan_absence = runtime_plan.get("github_evidence", {}).get(
                "pre_jit_registration_absence"
            )
            created_jit = jit_created["jit_receipt"]
            created_runner = created_jit.get("runner") if type(created_jit) is dict else None
            _require(
                type(created_jit) is dict
                and set(created_jit)
                == {
                    "observed_at",
                    "request_sha256",
                    "response_sha256",
                    "response_body_sha256",
                    "runner",
                    "jit_config_sha256",
                    "encoded_jit_config_persisted",
                    "runner_group_id",
                    "runner_group_get_performed",
                }
                and created_jit["request_sha256"] == _sha(_canonical(jit_body))
                and type(created_jit["observed_at"]) is str
                and _iso(_parse_time(created_jit["observed_at"], "journal_jit_observed_at"))
                == created_jit["observed_at"]
                and all(
                    type(created_jit[field_name]) is str
                    and SHA256_RE.fullmatch(created_jit[field_name]) is not None
                    for field_name in (
                        "request_sha256",
                        "response_sha256",
                        "response_body_sha256",
                        "jit_config_sha256",
                    )
                )
                and created_jit["response_sha256"] != created_jit["response_body_sha256"]
                and type(created_runner) is dict
                and set(created_runner) == {"id", "name", "os", "status", "busy", "labels"}
                and type(created_runner["id"]) is int
                and created_runner["id"] > 0
                and created_runner["name"] == job.runner_name
                and created_runner["os"] == "unknown"
                and created_runner["status"] == "offline"
                and created_runner["busy"] is False
                and created_runner["labels"] == [job.runner_name]
                and created_jit["encoded_jit_config_persisted"] is False
                and type(created_jit["runner_group_id"]) is int
                and created_jit["runner_group_id"] == 1
                and created_jit["runner_group_get_performed"] is False
                and runtime_plan.get("phase") == phase
                and runtime_plan.get("control_plane_plan_sha256") == plan_sha256
                and runtime_plan.get("dispatch", {}).get("run_id") == session.run["id"]
                and runtime_plan.get("job", {}).get("key") == job.key
                and runtime_plan.get("job", {}).get("job_id") == job.job_id
                and runtime_plan.get("job", {}).get("runner_name") == job.runner_name
                and plan_absence
                == {
                    "observed_at": absence["observed_at"],
                    "response_sha256": absence["response_sha256"],
                    "total_count": 0,
                    "runners": [],
                }
                and plan_jit
                == {
                    "observed_at": created_jit["observed_at"],
                    "response_sha256": created_jit["response_sha256"],
                    "runner": created_jit["runner"],
                }
                and runtime_plan.get("job", {}).get("jit_config_sha256")
                == created_jit["jit_config_sha256"],
                f"{phase}_runtime_plan_jit_binding_rejected",
            )
            assert isinstance(created_jit, dict)
            assert isinstance(created_runner, dict)
            _require(
                _parse_time(absence["observed_at"], "journal_pre_jit_absence_time")
                < _parse_time(created_jit["observed_at"], "journal_jit_observed_time")
                < _parse_time(runtime_plan.get("created_at"), "journal_runtime_created_time"),
                f"{phase}_jit_observation_order_rejected",
            )
            start_intent = take("remote-start-intent")
            _require(
                set(start_intent)
                == {
                    "phase",
                    "run_id",
                    "head_sha",
                    "job_key",
                    "job_id",
                    "runner_id",
                    "runner_name",
                    "runtime_plan_sha256",
                    "remote_start_retried",
                }
                and start_intent["phase"] == phase
                and start_intent["run_id"] == session.run["id"]
                and start_intent["head_sha"] == session.head_sha
                and start_intent["job_key"] == job.key
                and start_intent["job_id"] == job.job_id
                and type(start_intent["runner_id"]) is int
                and start_intent["runner_id"] == runtime_plan["job"]["runner_id"]
                and start_intent["runner_name"] == job.runner_name
                and start_intent["runtime_plan_sha256"] == runtime.runtime_plan_sha256(runtime_plan)
                and start_intent["remote_start_retried"] is False,
                f"{phase}_remote_start_intent_rejected",
            )
            take("remote-cleanup")
            take("accepted-actions-job")

        if phase == "final-main":
            take("final-main-acceptance")
        take("phase-settlement")
        inbox_settlement = take("operator-app-inbox-settlement")
        _require(
            set(inbox_settlement)
            == app_inbox_keys
            | {
                "all_expected_captures_consumed",
                "capture_bytes_retained_only_as_driver_archive",
                "all_consumed_raw_pages_archived_in_evidence_root",
                "accepted_source_generations_retained_in_owner_private_inbox",
                "final_inbox_inventory",
            }
            and inbox_settlement["phase"] == app_inbox["phase"]
            and inbox_settlement["expected_capture_count"] == app_inbox["expected_capture_count"]
            and type(inbox_settlement["accepted_capture_count"]) is int
            and inbox_settlement["accepted_capture_count"] == expected_capture_count
            and type(inbox_settlement["stale_generation_count"]) is int
            and inbox_settlement["stale_generation_count"] >= 0
            and type(inbox_settlement["stale_generations_sha256"]) is str
            and SHA256_RE.fullmatch(inbox_settlement["stale_generations_sha256"]) is not None
            and inbox_settlement["owner_private_directory_receipt_sha256"]
            == app_inbox["owner_private_directory_receipt_sha256"]
            and inbox_settlement["on_demand_before_each_jit"] is True
            and inbox_settlement["ready_marker_no_replace_required"] is True
            and inbox_settlement["raw_pages_archived_by_driver"] is True
            and inbox_settlement["all_expected_captures_consumed"] is True
            and inbox_settlement["capture_bytes_retained_only_as_driver_archive"] is False
            and inbox_settlement["all_consumed_raw_pages_archived_in_evidence_root"] is True
            and inbox_settlement["accepted_source_generations_retained_in_owner_private_inbox"]
            is True,
            f"{phase}_operator_app_inbox_settlement_rejected",
        )
        final_inbox_inventory = inbox_settlement["final_inbox_inventory"]
        final_inventory_keys = {
            "schema_version",
            "kind",
            "phase",
            "accepted_generation_count",
            "stale_generation_count",
            "generation_count",
            "consumed_generations",
            "files",
            "directories",
            "file_count",
            "directory_count",
            "owner_private_directory_receipt_sha256",
            "accepted_source_generations_retained",
            "unobserved_residue_present",
            "evidence_sha256",
        }
        _require(
            type(final_inbox_inventory) is dict
            and set(final_inbox_inventory) == final_inventory_keys,
            f"{phase}_operator_app_inbox_final_inventory_rejected",
        )
        assert isinstance(final_inbox_inventory, dict)
        final_inventory_material = dict(final_inbox_inventory)
        final_inventory_evidence = final_inventory_material.pop("evidence_sha256", None)
        consumed_generations = final_inbox_inventory["consumed_generations"]
        final_files = final_inbox_inventory["files"]
        final_directories = final_inbox_inventory["directories"]
        _require(
            type(final_inbox_inventory["schema_version"]) is int
            and final_inbox_inventory["schema_version"] == 1
            and final_inbox_inventory["kind"] == "explainiverse-installed-app-inbox-final-inventory"
            and final_inbox_inventory["phase"] == phase
            and type(final_inbox_inventory["accepted_generation_count"]) is int
            and final_inbox_inventory["accepted_generation_count"] == expected_capture_count
            and type(final_inbox_inventory["stale_generation_count"]) is int
            and final_inbox_inventory["stale_generation_count"]
            == inbox_settlement["stale_generation_count"]
            and type(final_inbox_inventory["generation_count"]) is int
            and final_inbox_inventory["generation_count"]
            == expected_capture_count + inbox_settlement["stale_generation_count"]
            and type(consumed_generations) is list
            and len(consumed_generations) == final_inbox_inventory["generation_count"]
            and type(final_files) is list
            and type(final_directories) is list
            and type(final_inbox_inventory["file_count"]) is int
            and final_inbox_inventory["file_count"] == len(final_files)
            and type(final_inbox_inventory["directory_count"]) is int
            and final_inbox_inventory["directory_count"] == len(final_directories)
            and final_inbox_inventory["owner_private_directory_receipt_sha256"]
            == app_inbox["owner_private_directory_receipt_sha256"]
            and final_inbox_inventory["accepted_source_generations_retained"] is True
            and final_inbox_inventory["unobserved_residue_present"] is False
            and type(final_inventory_evidence) is str
            and SHA256_RE.fullmatch(final_inventory_evidence) is not None
            and _sha(_canonical(final_inventory_material)) == final_inventory_evidence,
            f"{phase}_operator_app_inbox_final_inventory_rejected",
        )
        assert isinstance(consumed_generations, list)
        assert isinstance(final_files, list)
        assert isinstance(final_directories, list)
        expected_flat_files: list[dict[str, Any]] = []
        expected_flat_directories: list[str] = []
        stale_ready_sha256: list[str] = []
        seen_publication_nonces: set[str] = set()
        seen_ready_sha256: set[str] = set()
        seen_capture_json_sha256: set[str] = set()
        seen_capture_evidence_sha256: set[str] = set()
        seen_page_sha256: set[str] = set()
        accepted_index = 0
        stale_archive_index = 0
        expected_ordinal = 1
        expected_generation = 1
        previous_classified_at: datetime | None = None
        for generation in consumed_generations:
            base_generation_keys = {
                "ordinal",
                "generation",
                "publication_nonce",
                "ready_marker",
                "ready_marker_bytes",
                "ready_marker_sha256",
                "capture_directory",
                "capture_json_bytes",
                "capture_json_sha256",
                "capture",
                "classified_at",
                "pages",
                "pages_inventory_sha256",
                "classification",
            }
            classification = generation.get("classification") if type(generation) is dict else None
            _require(
                type(generation) is dict
                and set(generation)
                == base_generation_keys
                | (
                    {"capture_evidence_sha256"}
                    if classification == "accepted"
                    else {"stale_archive"}
                )
                and classification in {"accepted", "stale"}
                and type(generation["ordinal"]) is int
                and generation["ordinal"] == expected_ordinal
                and type(generation["generation"]) is int
                and generation["generation"] == expected_generation
                and type(generation["publication_nonce"]) is str
                and re.fullmatch(r"[0-9a-f]{32}", generation["publication_nonce"]) is not None
                and generation["publication_nonce"] not in seen_publication_nonces
                and generation["capture_directory"]
                == f"capture-{expected_ordinal:02d}-{expected_generation:06d}"
                and generation["ready_marker"]
                == f"ready-{expected_ordinal:02d}-{expected_generation:06d}.json"
                and type(generation["ready_marker_bytes"]) is int
                and generation["ready_marker_bytes"] > 0
                and type(generation["ready_marker_sha256"]) is str
                and SHA256_RE.fullmatch(generation["ready_marker_sha256"]) is not None
                and generation["ready_marker_sha256"] not in seen_ready_sha256
                and type(generation["capture_json_bytes"]) is int
                and generation["capture_json_bytes"] > 0
                and type(generation["capture_json_sha256"]) is str
                and SHA256_RE.fullmatch(generation["capture_json_sha256"]) is not None
                and generation["capture_json_sha256"] not in seen_capture_json_sha256
                and type(generation["capture"]) is dict
                and type(generation["classified_at"]) is str
                and type(generation["pages"]) is list
                and bool(generation["pages"]),
                f"{phase}_operator_app_inbox_generation_rejected",
            )
            classified_at = _public_receipt_time(
                generation["classified_at"],
                f"{phase}_operator_app_inbox_classified_at",
            )
            _require(
                previous_classified_at is None or classified_at >= previous_classified_at,
                f"{phase}_operator_app_inbox_classification_order_rejected",
            )
            previous_classified_at = classified_at
            capture_mapping = _json_copy(generation["capture"])
            capture_raw = _canonical(capture_mapping)
            _require(
                generation["capture_json_bytes"] == len(capture_raw)
                and generation["capture_json_sha256"] == _sha(capture_raw),
                f"{phase}_operator_app_inbox_capture_json_binding_rejected",
            )
            page_names: set[str] = set()
            generation_page_sha256: set[str] = set()
            normalized_pages: list[dict[str, Any]] = []
            for page in generation["pages"]:
                _require(
                    type(page) is dict
                    and set(page) == {"filename", "bytes", "sha256"}
                    and type(page["filename"]) is str
                    and Path(page["filename"]).name == page["filename"]
                    and page["filename"] not in {".", ".."}
                    and page["filename"] not in page_names
                    and type(page["bytes"]) is int
                    and page["bytes"] > 0
                    and type(page["sha256"]) is str
                    and SHA256_RE.fullmatch(page["sha256"]) is not None
                    and page["sha256"] not in generation_page_sha256
                    and page["sha256"] not in seen_page_sha256,
                    f"{phase}_operator_app_inbox_generation_page_rejected",
                )
                page_names.add(str(page["filename"]))
                generation_page_sha256.add(str(page["sha256"]))
                normalized_pages.append(dict(page))
            _require(
                generation["pages_inventory_sha256"] == _sha(_canonical(normalized_pages)),
                f"{phase}_operator_app_inbox_generation_pages_digest_rejected",
            )
            ready_material = {
                "schema_version": 1,
                "kind": "explainiverse-installed-app-capture-ready",
                "phase": phase,
                "ordinal": expected_ordinal,
                "generation": expected_generation,
                "publication_nonce": generation["publication_nonce"],
                "capture_directory": generation["capture_directory"],
                "capture_json_sha256": generation["capture_json_sha256"],
                "pages_inventory_sha256": generation["pages_inventory_sha256"],
            }
            ready_raw = _canonical(ready_material)
            _require(
                generation["ready_marker_bytes"] == len(ready_raw)
                and generation["ready_marker_sha256"] == _sha(ready_raw),
                f"{phase}_operator_app_inbox_ready_marker_binding_rejected",
            )
            seen_publication_nonces.add(str(generation["publication_nonce"]))
            seen_ready_sha256.add(str(generation["ready_marker_sha256"]))
            seen_capture_json_sha256.add(str(generation["capture_json_sha256"]))
            seen_page_sha256.update(generation_page_sha256)
            bundle = str(generation["capture_directory"])
            expected_flat_directories.extend((bundle, f"{bundle}/pages"))
            expected_flat_files.append(
                {
                    "path": f"{bundle}/capture.json",
                    "bytes": generation["capture_json_bytes"],
                    "sha256": generation["capture_json_sha256"],
                }
            )
            expected_flat_files.extend(
                {
                    "path": f"{bundle}/pages/{page['filename']}",
                    "bytes": page["bytes"],
                    "sha256": page["sha256"],
                }
                for page in normalized_pages
            )
            expected_flat_files.append(
                {
                    "path": generation["ready_marker"],
                    "bytes": generation["ready_marker_bytes"],
                    "sha256": generation["ready_marker_sha256"],
                }
            )
            if classification == "stale":
                _require(
                    stale_archive_index < len(stale_app_archives)
                    and generation["stale_archive"] == stale_app_archives[stale_archive_index],
                    f"{phase}_operator_app_inbox_stale_archive_order_rejected",
                )
                stale_archive = generation["stale_archive"]
                _require(
                    type(stale_archive) is dict,
                    f"{phase}_operator_app_inbox_stale_archive_rejected",
                )
                stale_material = dict(stale_archive)
                stale_evidence_sha256 = stale_material.pop("archive_evidence_sha256", None)
                stale_identity_material = {
                    "phase": phase,
                    "ordinal": generation["ordinal"],
                    "generation": generation["generation"],
                    "publication_nonce": generation["publication_nonce"],
                    "ready_marker_sha256": generation["ready_marker_sha256"],
                    "capture_json_sha256": generation["capture_json_sha256"],
                    "classified_at": generation["classified_at"],
                }
                stale_archive_identity = _sha(_canonical(stale_identity_material))
                _require(
                    set(stale_archive)
                    == {
                        "schema_version",
                        "kind",
                        *stale_identity_material,
                        "archive_identity_sha256",
                        "archive_directory",
                        "files",
                        "all_pages_exclusive_single_link",
                        "archive_evidence_sha256",
                    }
                    and type(stale_archive["schema_version"]) is int
                    and stale_archive["schema_version"] == 1
                    and stale_archive["kind"] == "explainiverse-installed-app-stale-raw-archive"
                    and all(
                        stale_archive[key] == value
                        for key, value in stale_identity_material.items()
                    )
                    and stale_archive["archive_identity_sha256"] == stale_archive_identity
                    and stale_archive["archive_directory"]
                    == f"installed-app-pages/{stale_archive_identity}"
                    and stale_archive["files"] == normalized_pages
                    and stale_archive["all_pages_exclusive_single_link"] is True
                    and type(stale_evidence_sha256) is str
                    and SHA256_RE.fullmatch(stale_evidence_sha256) is not None
                    and _sha(_canonical(stale_material)) == stale_evidence_sha256,
                    f"{phase}_operator_app_inbox_stale_archive_rejected",
                )

                def read_stale_page(filename: str) -> bytes:
                    return EvidenceJournal._read_archived_app_page(
                        evidence_root,
                        stale_archive_identity,
                        filename,
                    )

                try:
                    TrustedAppCapture.from_mapping(
                        capture_mapping,
                        resources=controller_resources,
                        evidence_reader=read_stale_page,
                        now=classified_at,
                    )
                except ControllerError as exc:
                    _require(
                        str(exc) == "app_capture_stale",
                        f"{phase}_operator_app_inbox_stale_classification_rejected",
                    )
                else:
                    raise ControllerError(
                        f"{phase}_operator_app_inbox_stale_classification_rejected"
                    )
                stale_ready_sha256.append(str(generation["ready_marker_sha256"]))
                stale_archive_index += 1
                expected_generation += 1
                continue
            _require(
                accepted_index < len(app_archives) and accepted_index < len(app_captures),
                f"{phase}_operator_app_inbox_accepted_generation_count_rejected",
            )
            accepted_archive_identity = app_archives[accepted_index].get("capture_evidence_sha256")
            _require(
                type(accepted_archive_identity) is str
                and SHA256_RE.fullmatch(accepted_archive_identity) is not None,
                f"{phase}_operator_app_inbox_accepted_archive_identity_rejected",
            )
            assert isinstance(accepted_archive_identity, str)

            def read_accepted_page(filename: str) -> bytes:
                return EvidenceJournal._read_archived_app_page(
                    evidence_root,
                    accepted_archive_identity,
                    filename,
                )

            accepted_capture = TrustedAppCapture.from_mapping(
                capture_mapping,
                resources=controller_resources,
                evidence_reader=read_accepted_page,
                now=classified_at,
            )
            authority_observed_at = _parse_time(
                authority_windows[accepted_index].get("observed_at"),
                f"{phase}_operator_app_inbox_authority_observed_at",
            )
            _require(
                accepted_index < len(app_captures)
                and generation["capture_evidence_sha256"] not in seen_capture_evidence_sha256
                and generation["capture_evidence_sha256"]
                == app_captures[accepted_index].get("evidence_sha256")
                and generation["capture_json_sha256"]
                == _sha(_canonical(app_captures[accepted_index].get("normalized_capture")))
                and capture_mapping == app_captures[accepted_index].get("normalized_capture")
                and accepted_capture.to_mapping() == app_captures[accepted_index]
                and accepted_capture.evidence_sha256 == generation["capture_evidence_sha256"]
                and classified_at <= authority_observed_at
                and normalized_pages == app_archives[accepted_index].get("files"),
                f"{phase}_operator_app_inbox_accepted_generation_binding_rejected",
            )
            seen_capture_evidence_sha256.add(str(generation["capture_evidence_sha256"]))
            accepted_index += 1
            expected_ordinal += 1
            expected_generation = 1
        _require(
            accepted_index == expected_capture_count
            and stale_archive_index == len(stale_app_archives)
            and expected_ordinal == expected_capture_count + 1
            and _sha(_canonical(stale_ready_sha256)) == inbox_settlement["stale_generations_sha256"]
            and final_files == sorted(expected_flat_files, key=lambda item: str(item["path"]))
            and final_directories == sorted(expected_flat_directories),
            f"{phase}_operator_app_inbox_generation_sequence_rejected",
        )
        normalized_final_files: dict[str, Mapping[str, Any]] = {}
        for item in final_files:
            _require(
                type(item) is dict
                and set(item) == {"path", "bytes", "sha256"}
                and type(item["path"]) is str
                and PurePosixPath(item["path"]).as_posix() == item["path"]
                and not PurePosixPath(item["path"]).is_absolute()
                and all(part not in {"", ".", ".."} for part in PurePosixPath(item["path"]).parts)
                and item["path"] not in normalized_final_files
                and type(item["bytes"]) is int
                and item["bytes"] > 0
                and type(item["sha256"]) is str
                and SHA256_RE.fullmatch(item["sha256"]) is not None,
                f"{phase}_operator_app_inbox_final_file_rejected",
            )
            normalized_final_files[str(item["path"])] = item
        _require(
            final_files == [normalized_final_files[path] for path in sorted(normalized_final_files)]
            and all(type(item) is str for item in final_directories)
            and final_directories == sorted(set(final_directories))
            and len(final_directories) == 2 * final_inbox_inventory["generation_count"],
            f"{phase}_operator_app_inbox_final_order_rejected",
        )
        accepted_bundles: set[str] = set()
        for app_capture in app_captures:
            normalized_capture = app_capture.get("normalized_capture")
            _require(
                type(normalized_capture) is dict,
                f"{phase}_operator_app_inbox_capture_rejected",
            )
            assert isinstance(normalized_capture, dict)
            capture_sha256 = _sha(_canonical(normalized_capture))
            matching_capture_files = [
                path
                for path, item in normalized_final_files.items()
                if path.endswith("/capture.json") and item["sha256"] == capture_sha256
            ]
            _require(
                len(matching_capture_files) == 1,
                f"{phase}_operator_app_inbox_capture_file_binding_rejected",
            )
            bundle = matching_capture_files[0].removesuffix("/capture.json")
            _require(
                re.fullmatch(r"capture-[0-9]{2}-[0-9]{6}", bundle) is not None
                and bundle not in accepted_bundles
                and bundle in final_directories
                and f"{bundle}/pages" in final_directories
                and re.sub(r"^capture-", "ready-", bundle) + ".json" in normalized_final_files,
                f"{phase}_operator_app_inbox_capture_directory_binding_rejected",
            )
            accepted_bundles.add(bundle)
            evidence_items = normalized_capture.get("evidence")
            _require(
                type(evidence_items) is list and bool(evidence_items),
                f"{phase}_operator_app_inbox_capture_manifest_rejected",
            )
            assert isinstance(evidence_items, list)
            for evidence_item in evidence_items:
                _require(
                    type(evidence_item) is dict
                    and type(evidence_item.get("filename")) is str
                    and f"{bundle}/pages/{evidence_item['filename']}" in normalized_final_files
                    and normalized_final_files[f"{bundle}/pages/{evidence_item['filename']}"][
                        "bytes"
                    ]
                    == evidence_item.get("bytes")
                    and normalized_final_files[f"{bundle}/pages/{evidence_item['filename']}"][
                        "sha256"
                    ]
                    == evidence_item.get("sha256"),
                    f"{phase}_operator_app_inbox_page_binding_rejected",
                )
        closed = take("ssh-access-identity-closed")
        _require(
            set(closed)
            == {
                "public_key_sha256",
                "closed",
                "private_path_archived",
                "private_digest_archived",
            }
            and closed.get("public_key_sha256") == plan_ssh["public_key_sha256"]
            and closed.get("closed") is True
            and closed.get("private_path_archived") is False
            and closed.get("private_digest_archived") is False,
            f"{phase}_ssh_access_identity_close_rejected",
        )
        current = snapshot("instance_bound")
        require_bound_instance(current, context="teardown")
        current = transition("terminate", current, "instance_absent")
        zero_runners = take("github-zero-runners-before-abort")
        zero_runner_observation_count = zero_runners.get("observation_count")
        _require(
            zero_runners.get("repository") == REPOSITORY
            and type(zero_runners.get("runner_count")) is int
            and zero_runners.get("runner_count") == 0
            and type(zero_runner_observation_count) is int
            and zero_runner_observation_count >= 3,
            f"{phase}_zero_runner_receipt_rejected",
        )
        current = transition("delete_ruleset", current, "ruleset_absent")
        current = transition("restore_global", current, "restored")
        lifecycle = take("lifecycle-restored")
        _require(
            lifecycle.get("plan_sha256") == plan_sha256
            and type(lifecycle.get("provider_instances")) is int
            and lifecycle.get("provider_instances") == 0
            and type(lifecycle.get("provider_firewall_rulesets")) is int
            and lifecycle.get("provider_firewall_rulesets") == 0
            and lifecycle.get("global_firewall_restored") is True
            and type(lifecycle.get("repository_runners")) is int
            and lifecycle.get("repository_runners") == 0,
            f"{phase}_lifecycle_restoration_rejected",
        )
        _require(cursor == len(entries), f"{phase}_journal_event_trailing_rejected")
        EvidenceJournal._validate_accepted_evidence_root_inventory(
            evidence_root,
            operator_preflight_filename=str(operator_binding["operator_preflight_filename"]),
            app_archives=[*app_archives, *stale_app_archives],
            journal_recovery_payloads=journal_recovery_payloads,
            context=f"{phase}_evidence_root",
        )

    def verified_entries(self) -> tuple[tuple[str, dict[str, Any]], ...]:
        """Re-read the current chain; recovery never trusts cached state alone."""

        _require(
            not any(self._directory.glob(".evidence-*.tmp")),
            "journal_temporary_requires_explicit_reopen",
        )
        paths = self._journal_paths(self._directory)
        _require(len(paths) == self._sequence, "journal_entry_count_drift")
        previous_sha256: str | None = None
        result: list[tuple[str, dict[str, Any]]] = []
        for sequence, path in enumerate(paths, start=1):
            _require(
                path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1,
                "journal_entry_file_rejected",
            )
            raw = path.read_bytes()
            envelope = _json(raw, "journal_revalidation_entry")
            _require(
                type(envelope) is dict
                and raw == _canonical(envelope)
                and envelope.get("sequence") == sequence
                and envelope.get("control_plane_plan_sha256") == self._plan_sha256
                and envelope.get("evidence_directory_acl_receipt_sha256") == self._acl_sha256
                and envelope.get("previous_evidence_sha256") == previous_sha256
                and type(envelope.get("label")) is str
                and type(envelope.get("payload")) is dict,
                "journal_revalidation_failed",
            )
            _require(
                path.name == f"{sequence:03d}-{envelope['label']}.json",
                "journal_filename_binding_rejected",
            )
            payload = envelope["payload"]
            assert isinstance(payload, dict)
            _reject_secret_keys(payload)
            result.append((str(envelope["label"]), _json_copy(payload)))
            previous_sha256 = _sha(raw)
        _require(
            previous_sha256 == self._previous_sha256,
            "journal_tail_digest_drift",
        )
        return tuple(result)

    @staticmethod
    def _validate_phase_dispatch_chain(
        intent_payload: Mapping[str, Any],
        settled_payload: Mapping[str, Any],
        session_payload: Mapping[str, Any],
        *,
        expected_phase: str,
        expected_head_sha: str,
    ) -> PhaseSession:
        """Validate the dispatch WAL, reconciliation material, and session as one unit."""

        _require(expected_phase in {"final-main", "publication"}, "dispatch_chain_phase_rejected")
        spec = PHASES[expected_phase]
        intent_keys = {
            "phase",
            "workflow",
            "workflow_path",
            "dispatch_path",
            "dispatch_ref",
            "run_ref",
            "head_sha",
            "inputs",
            "expected_runner_nonces",
            "pre_dispatch_run_ids",
            "request_sha256",
            "mutation_retried",
        }
        _require(
            type(intent_payload) is dict
            and set(intent_payload) == intent_keys
            and intent_payload["phase"] == expected_phase
            and intent_payload["workflow"] == spec["workflow"]
            and intent_payload["workflow_path"] == spec["workflow_path"]
            and intent_payload["dispatch_ref"] == spec["dispatch_ref"]
            and intent_payload["run_ref"] == spec["run_ref"]
            and intent_payload["head_sha"] == expected_head_sha
            and type(intent_payload["inputs"]) is dict
            and type(intent_payload["expected_runner_nonces"]) is list
            and type(intent_payload["pre_dispatch_run_ids"]) is list
            and all(
                type(item) is int and item > 0 for item in intent_payload["pre_dispatch_run_ids"]
            )
            and intent_payload["pre_dispatch_run_ids"]
            == sorted(intent_payload["pre_dispatch_run_ids"])
            and len(set(intent_payload["pre_dispatch_run_ids"]))
            == len(intent_payload["pre_dispatch_run_ids"])
            and intent_payload["mutation_retried"] is False,
            "accepted_dispatch_intent_schema_rejected",
        )
        dispatch_path = (
            f"/repos/jemsbhai/explainiverse/actions/workflows/" f"{spec['workflow']}/dispatches"
        )
        request_body = {
            "ref": spec["dispatch_ref"],
            "inputs": intent_payload["inputs"],
        }
        request_sha256 = _sha(
            _canonical({"method": "POST", "path": dispatch_path, "body": request_body})
        )
        nonce_keys = tuple(spec["all_nonce_keys"])
        _require(
            intent_payload["dispatch_path"] == dispatch_path
            and intent_payload["request_sha256"] == request_sha256
            and intent_payload["expected_runner_nonces"]
            == [intent_payload["inputs"].get(key) for key in nonce_keys],
            "accepted_dispatch_intent_binding_rejected",
        )

        session = LiveReleaseDriver._session_from_journal(session_payload, ())
        _require(
            session.phase == expected_phase
            and session.head_sha == expected_head_sha
            and session.inputs == intent_payload["inputs"],
            "accepted_dispatch_session_binding_rejected",
        )
        settled_keys = {
            "schema_version",
            "kind",
            "phase",
            "head_sha",
            "run_id",
            "run_attempt",
            "dispatch_reconciliation",
            "source_bindings",
            "dispatch_receipt",
        }
        reconciliation = settled_payload.get("dispatch_reconciliation")
        source_bindings = settled_payload.get("source_bindings")
        raw_receipt = settled_payload.get("dispatch_receipt")
        expected_source_keys = (
            {"main"}
            if expected_phase == "final-main"
            else {"ref", "tag", "main", "preflight", "cuda", "accepted_cuda_nonces"}
        )
        _require(
            type(settled_payload) is dict
            and set(settled_payload) == settled_keys
            and type(settled_payload["schema_version"]) is int
            and settled_payload["schema_version"] == 1
            and settled_payload["kind"] == "explainiverse-github-dispatch-settlement"
            and settled_payload["phase"] == expected_phase
            and settled_payload["head_sha"] == expected_head_sha
            and type(settled_payload["run_id"]) is int
            and settled_payload["run_id"] == session.run["id"]
            and type(settled_payload["run_attempt"]) is int
            and settled_payload["run_attempt"] == 1
            and type(reconciliation) is dict
            and type(source_bindings) is dict
            and set(source_bindings) == expected_source_keys
            and all(
                type(item) is str and SHA256_RE.fullmatch(item) is not None
                for item in source_bindings.values()
            )
            and type(raw_receipt) is dict,
            "accepted_dispatch_settlement_schema_rejected",
        )
        assert isinstance(reconciliation, dict)
        assert isinstance(source_bindings, dict)
        assert isinstance(raw_receipt, dict)
        receipt = DispatchReceipt.from_mapping(raw_receipt)
        expected_queued = [
            {
                "job_id": item.job_id,
                "job_name": item.name,
                "runner_name": item.runner_name,
                "nonce": item.nonce,
            }
            for item in session.queued_jobs
        ]
        reconciliation_keys = {
            "response_received",
            "response_sha256",
            "ambiguity",
            "run_id",
            "run_attempt",
            "head_sha",
            "run_response_sha256",
            "queued_jobs",
            "mutation_retried",
        }
        response_received = reconciliation.get("response_received")
        response_sha256 = reconciliation.get("response_sha256")
        _require(
            set(reconciliation) == reconciliation_keys
            and type(response_received) is bool
            and (
                (
                    response_received
                    and type(response_sha256) is str
                    and SHA256_RE.fullmatch(response_sha256) is not None
                    and reconciliation["ambiguity"] is None
                )
                or (
                    not response_received
                    and response_sha256 is None
                    and type(reconciliation["ambiguity"]) is dict
                )
            )
            and reconciliation["run_id"] == session.run["id"]
            and type(reconciliation["run_attempt"]) is int
            and reconciliation["run_attempt"] == 1
            and reconciliation["head_sha"] == expected_head_sha
            and type(reconciliation["run_response_sha256"]) is str
            and SHA256_RE.fullmatch(reconciliation["run_response_sha256"]) is not None
            and reconciliation["queued_jobs"] == expected_queued
            and reconciliation["mutation_retried"] is False,
            "accepted_dispatch_reconciliation_rejected",
        )
        reconciliation_sha256 = _sha(_canonical(reconciliation))
        expected_response_sha256 = _sha(
            _canonical(
                {
                    "dispatch_reconciliation_sha256": reconciliation_sha256,
                    "source": source_bindings,
                }
            )
        )
        _require(
            receipt == session.dispatch_receipt
            and receipt.request_sha256 == request_sha256
            and receipt.mutation_response_received is response_received
            and receipt.mutation_reconciliation_sha256 == reconciliation_sha256
            and receipt.response_sha256 == expected_response_sha256
            and receipt.run_response_sha256 == reconciliation["run_response_sha256"],
            "accepted_dispatch_receipt_binding_rejected",
        )
        return session

    def pending_recovery_dispatch_intent(self) -> dict[str, Any] | None:
        """Return the sole unresolved recovery POST intent, if one exists."""

        pending: dict[str, Any] | None = None
        for label, payload in self.verified_entries():
            if label == "github-recovery-dispatch-intent":
                _require(
                    pending is None,
                    "recovery_dispatch_intent_overlap",
                )
                pending = ReleaseGpuController._validate_recovery_dispatch_intent(payload)
            elif label == "github-recovery-dispatch-settled":
                _require(
                    pending is not None
                    and set(payload) == set(RecoveryDispatchReceipt.__dataclass_fields__),
                    "recovery_dispatch_settlement_without_intent",
                )
                assert pending is not None
                receipt = RecoveryDispatchReceipt.from_mapping(payload)
                _require(
                    receipt.request_sha256 == pending["request_sha256"]
                    and receipt.tag == pending["tag"]
                    and receipt.head_sha == pending["head_sha"]
                    and receipt.source_run_id == pending["source_run_id"]
                    and receipt.require_staged_drill is True
                    and receipt.recovery_request_nonce == pending["recovery_request_nonce"]
                    and receipt.display_title == pending["display_title"]
                    and receipt.workflow_response_sha256 == pending["workflow_response_sha256"]
                    and receipt.immutable_source_evidence_sha256
                    == pending["immutable_source_evidence_sha256"]
                    and receipt.source_run_evidence_sha256 == pending["source_run_evidence_sha256"]
                    and receipt.pre_dispatch_history_sha256
                    == pending["pre_dispatch_history_sha256"],
                    "recovery_dispatch_settlement_binding_rejected",
                )
                pending = None
        return _json_copy(pending) if pending is not None else None

    @staticmethod
    def _validate_recovery_receipt_intent_binding(
        intent: Mapping[str, Any], receipt: RecoveryDispatchReceipt
    ) -> None:
        _require(
            receipt.tag == intent["tag"]
            and receipt.head_sha == intent["head_sha"]
            and receipt.source_run_id == intent["source_run_id"]
            and receipt.require_staged_drill is True
            and receipt.recovery_request_nonce == intent["recovery_request_nonce"]
            and receipt.display_title == intent["display_title"]
            and receipt.request_sha256 == intent["request_sha256"]
            and receipt.workflow_response_sha256 == intent["workflow_response_sha256"]
            and receipt.immutable_source_evidence_sha256
            == intent["immutable_source_evidence_sha256"]
            and receipt.source_run_evidence_sha256 == intent["source_run_evidence_sha256"]
            and receipt.pre_dispatch_history_sha256 == intent["pre_dispatch_history_sha256"]
            and receipt.run_id not in intent["pre_dispatch_run_ids"],
            "publication_recovery_receipt_intent_binding_rejected",
        )

    @staticmethod
    def build_publication_recovery_operator_settlement(
        source: PublicationRecoverySource,
        receipt: RecoveryDispatchReceipt,
    ) -> dict[str, Any]:
        """Build the exact local-only summary for one observed recovery run."""

        _require(
            type(source) is PublicationRecoverySource and type(receipt) is RecoveryDispatchReceipt,
            "publication_recovery_operator_settlement_type_rejected",
        )
        source_mapping = source.to_mapping()
        source_evidence_sha256 = source_mapping.pop("evidence_sha256", None)
        _require(
            type(source_evidence_sha256) is str
            and _sha(_canonical(source_mapping)) == source_evidence_sha256
            and receipt.head_sha == source.head_sha
            and receipt.source_run_id == source.run_id,
            "publication_recovery_operator_settlement_source_rejected",
        )
        receipt = RecoveryDispatchReceipt.from_mapping(receipt.to_mapping())
        return {
            "plan_sha256": source.control_plane_plan_sha256,
            "head_sha": source.head_sha,
            "source_run_id": source.run_id,
            "mode": (
                "mutation-response-observed"
                if receipt.mutation_response_received
                else "response-loss-reconciled"
            ),
            "recovery_dispatch_evidence_sha256": receipt.evidence_sha256,
            "publication_recovery_source_evidence_sha256": source.evidence_sha256,
            "publication_journal_sha256": source.publication_journal_sha256,
            "recovery_run_id": receipt.run_id,
            "request_nonce_archived_in_controller_receipt": True,
            "raw_dispatch_bypass_used": False,
            "pending_intent_replayed": False,
            "workflow_completion_verified": False,
            "no_republish_verified": False,
        }

    @staticmethod
    def _validate_journal_publish_recovery_payload(
        payload: Mapping[str, Any],
        *,
        record_sequence: int,
        root: Path,
        plan_sha256: str,
        acl_sha256: str,
        seen_temporary_names: set[str],
        seen_sidecar_names: set[str],
    ) -> None:
        material = dict(payload)
        evidence_sha256 = material.pop("recovery_evidence_sha256", None)
        rows = payload.get("recovered_entries")
        emergency = payload.get("emergency_uncommitted_slot")
        sidecar_filename = payload.get("sidecar_filename")
        _require(
            set(payload)
            == {
                "schema_version",
                "kind",
                "control_plane_plan_sha256",
                "evidence_directory_acl_receipt_sha256",
                "sidecar_filename",
                "sidecar_bytes",
                "sidecar_sha256",
                "recovered_entries",
                "emergency_uncommitted_slot",
                "recovery_evidence_sha256",
            }
            and type(payload.get("schema_version")) is int
            and payload.get("schema_version") == 1
            and payload.get("kind") == "explainiverse-interrupted-local-evidence-recovery"
            and payload.get("control_plane_plan_sha256") == plan_sha256
            and payload.get("evidence_directory_acl_receipt_sha256") == acl_sha256
            and type(sidecar_filename) is str
            and LOCAL_RECOVERY_SIDECAR_RE.fullmatch(sidecar_filename) is not None
            and sidecar_filename not in seen_sidecar_names
            and type(payload.get("sidecar_bytes")) is int
            and 0 < payload["sidecar_bytes"] <= MAX_LOCAL_RECOVERY_SIDECAR_BYTES
            and type(payload.get("sidecar_sha256")) is str
            and SHA256_RE.fullmatch(payload["sidecar_sha256"]) is not None
            and sidecar_filename == f".local-evidence-recovery-{payload['sidecar_sha256']}.json"
            and type(rows) is list
            and type(emergency) in {dict, type(None)}
            and (bool(rows) or emergency is not None)
            and type(evidence_sha256) is str
            and SHA256_RE.fullmatch(evidence_sha256) is not None
            and _sha(_canonical(material)) == evidence_sha256,
            "publication_journal_publish_recovery_rejected",
        )
        assert isinstance(rows, list)
        sidecar_path = root / str(sidecar_filename)
        _require(
            sidecar_path.is_file()
            and not sidecar_path.is_symlink()
            and sidecar_path.resolve(strict=True) == sidecar_path
            and sidecar_path.parent == root
            and sidecar_path.stat().st_nlink == 1,
            "publication_journal_publish_recovery_sidecar_rejected",
        )
        sidecar_raw = sidecar_path.read_bytes()
        sidecar = _json(sidecar_raw, "publication_journal_recovery_sidecar")
        _require(
            type(sidecar) is dict
            and sidecar_raw == _canonical(sidecar)
            and len(sidecar_raw) == payload["sidecar_bytes"]
            and _sha(sidecar_raw) == payload["sidecar_sha256"]
            and EvidenceJournal._local_recovery_public_mapping(
                sidecar_path,
                sidecar_raw,
                sidecar,
            )
            == payload,
            "publication_journal_publish_recovery_sidecar_binding_rejected",
        )
        complete_sequences: list[int] = []
        previous_row_sequence = 0
        for row in rows:
            _require(
                type(row) is dict
                and set(row)
                == {
                    "classification",
                    "temporary_filename",
                    "temporary_bytes",
                    "temporary_sha256",
                    "sequence",
                    "label",
                    "previous_evidence_sha256",
                    "final_filename",
                }
                and row["classification"]
                in {
                    "complete-unpublished-envelope",
                    "published-hardlink-envelope",
                }
                and type(row["temporary_filename"]) is str
                and re.fullmatch(
                    r"\.evidence-[0-9a-f]{32}\.tmp",
                    row["temporary_filename"],
                )
                is not None
                and row["temporary_filename"] not in seen_temporary_names
                and type(row["temporary_bytes"]) is int
                and 0 < row["temporary_bytes"] <= MAX_EVIDENCE_ATOMIC_BYTES
                and type(row["temporary_sha256"]) is str
                and SHA256_RE.fullmatch(row["temporary_sha256"]) is not None
                and type(row["sequence"]) is int
                and previous_row_sequence < row["sequence"] < record_sequence
                and type(row["label"]) is str
                and EVIDENCE_LABEL_RE.fullmatch(row["label"]) is not None
                and row["label"] != "journal-publish-recovery"
                and (
                    row["previous_evidence_sha256"] is None
                    or (
                        type(row["previous_evidence_sha256"]) is str
                        and SHA256_RE.fullmatch(row["previous_evidence_sha256"]) is not None
                    )
                )
                and row["final_filename"] == f"{row['sequence']:03d}-{row['label']}.json",
                "publication_journal_publish_recovery_row_rejected",
            )
            final_path = root / str(row["final_filename"])
            _require(
                final_path.is_file()
                and not final_path.is_symlink()
                and final_path.stat().st_nlink == 1,
                "publication_journal_publish_recovery_final_rejected",
            )
            final_raw = final_path.read_bytes()
            final_envelope = _json(final_raw, "publication_recovered_journal_entry")
            _require(
                final_raw == _canonical(final_envelope)
                and _sha(final_raw) == row["temporary_sha256"]
                and len(final_raw) == row["temporary_bytes"]
                and type(final_envelope) is dict
                and final_envelope.get("sequence") == row["sequence"]
                and final_envelope.get("label") == row["label"]
                and final_envelope.get("previous_evidence_sha256")
                == row["previous_evidence_sha256"]
                and final_envelope.get("control_plane_plan_sha256") == plan_sha256
                and final_envelope.get("evidence_directory_acl_receipt_sha256") == acl_sha256,
                "publication_journal_publish_recovery_final_binding_rejected",
            )
            seen_temporary_names.add(str(row["temporary_filename"]))
            previous_row_sequence = int(row["sequence"])
            if row["classification"] == "complete-unpublished-envelope":
                complete_sequences.append(int(row["sequence"]))
        _require(
            len(complete_sequences) <= 1
            and (not complete_sequences or complete_sequences == [record_sequence - 1]),
            "publication_journal_publish_recovery_tail_rejected",
        )
        if emergency is not None:
            _require(
                type(emergency) is dict
                and set(emergency)
                == {
                    "classification",
                    "reserve_filename",
                    "slot_index",
                    "slot_bytes",
                    "slot_sha256",
                }
                and emergency["classification"] == "complete-uncommitted-provider-intent-slot"
                and emergency["reserve_filename"] == EMERGENCY_EVIDENCE_FILENAME
                and type(emergency["slot_index"]) is int
                and 0 <= emergency["slot_index"] < EMERGENCY_EVIDENCE_SLOT_COUNT
                and type(emergency["slot_bytes"]) is int
                and emergency["slot_bytes"] == EMERGENCY_EVIDENCE_SLOT_SIZE
                and type(emergency["slot_sha256"]) is str
                and SHA256_RE.fullmatch(emergency["slot_sha256"]) is not None,
                "publication_journal_publish_recovery_emergency_rejected",
            )
        seen_sidecar_names.add(str(sidecar_filename))

    @staticmethod
    def _validate_publication_recovery_suffix(
        entries: Sequence[tuple[str, Mapping[str, Any], str]],
        *,
        anchor_index: int,
        source: PublicationRecoverySource,
    ) -> PublicationRecoveryTail:
        suffix = entries[anchor_index + 1 :]
        if not suffix:
            return PublicationRecoveryTail._from_verified(
                state="source-unrecorded",
                source_evidence_sha256=source.evidence_sha256,
                completed_run_ids=(),
                completed_request_nonces=(),
                pending_intent=None,
                pending_operator_settlement=None,
                last_operator_settlement=None,
            )
        _require(
            suffix[0][0] == "operator-publication-recovery-source"
            and suffix[0][1] == source.to_mapping(),
            "publication_recovery_source_marker_rejected",
        )
        completed_run_ids: list[int] = []
        completed_nonces: list[str] = []
        completed_intents: list[Mapping[str, Any]] = []
        completed_receipts: list[RecoveryDispatchReceipt] = []
        last_operator_settlement: Mapping[str, Any] | None = None
        cursor = 1
        while cursor < len(suffix):
            label, raw_intent, _ = suffix[cursor]
            _require(
                label == "github-recovery-dispatch-intent",
                "publication_recovery_suffix_order_rejected",
            )
            intent = ReleaseGpuController._validate_recovery_dispatch_intent(raw_intent)
            nonce = intent["recovery_request_nonce"]
            expected_prior_runs = [
                {
                    "id": prior_receipt.run_id,
                    "display_title": prior_receipt.display_title,
                    "head_sha": prior_receipt.head_sha,
                    "head_branch": runtime.PUBLICATION_TAG,
                    "run_attempt": 1,
                    "status": "completed",
                    "conclusion": "failure",
                    "actor": OWNER,
                    "triggering_actor": OWNER,
                    "recovery_request_nonce": prior_intent["recovery_request_nonce"],
                }
                for prior_intent, prior_receipt in zip(
                    completed_intents,
                    completed_receipts,
                )
            ]
            _require(
                intent["head_sha"] == source.head_sha
                and intent["source_run_id"] == source.run_id
                and intent["pre_dispatch_run_ids"] == sorted(completed_run_ids)
                and intent["pre_dispatch_runs"] == expected_prior_runs
                and nonce not in completed_nonces,
                "publication_recovery_suffix_intent_binding_rejected",
            )
            if cursor + 1 == len(suffix):
                return PublicationRecoveryTail._from_verified(
                    state="pending-intent",
                    source_evidence_sha256=source.evidence_sha256,
                    completed_run_ids=completed_run_ids,
                    completed_request_nonces=completed_nonces,
                    pending_intent=intent,
                    pending_operator_settlement=None,
                    last_operator_settlement=last_operator_settlement,
                )
            receipt_label, raw_receipt, _ = suffix[cursor + 1]
            _require(
                receipt_label == "github-recovery-dispatch-settled",
                "publication_recovery_suffix_order_rejected",
            )
            receipt = RecoveryDispatchReceipt.from_mapping(raw_receipt)
            EvidenceJournal._validate_recovery_receipt_intent_binding(intent, receipt)
            _require(
                receipt.run_id not in completed_run_ids
                and (not completed_run_ids or receipt.run_id > max(completed_run_ids)),
                "publication_recovery_run_id_reuse_rejected",
            )
            pending_operator_settlement = (
                EvidenceJournal.build_publication_recovery_operator_settlement(source, receipt)
            )
            if cursor + 2 == len(suffix):
                return PublicationRecoveryTail._from_verified(
                    state="pending-operator-settlement",
                    source_evidence_sha256=source.evidence_sha256,
                    completed_run_ids=completed_run_ids,
                    completed_request_nonces=completed_nonces,
                    pending_intent=None,
                    pending_operator_settlement=pending_operator_settlement,
                    last_operator_settlement=last_operator_settlement,
                )
            operator_label, operator_settlement, _ = suffix[cursor + 2]
            _require(
                operator_label == "operator-release-recovery-dispatch-settled"
                and operator_settlement == pending_operator_settlement,
                "publication_recovery_operator_settlement_binding_rejected",
            )
            completed_run_ids.append(receipt.run_id)
            completed_nonces.append(str(nonce))
            completed_intents.append(intent)
            completed_receipts.append(receipt)
            last_operator_settlement = operator_settlement
            cursor += 3
        return PublicationRecoveryTail._from_verified(
            state="complete",
            source_evidence_sha256=source.evidence_sha256,
            completed_run_ids=completed_run_ids,
            completed_request_nonces=completed_nonces,
            pending_intent=None,
            pending_operator_settlement=None,
            last_operator_settlement=last_operator_settlement,
        )

    @staticmethod
    def load_final_main_acceptance(
        evidence_directory: live.EvidenceDirectoryReceipt,
        *,
        controller_resources: SealedControllerResources,
        final_control_plane_plan_sha256: str,
        final_journal_sha256: str,
    ) -> FinalMainAcceptance:
        """Verify the complete final-main journal before restoring acceptance state."""

        _require(
            type(evidence_directory) is live.EvidenceDirectoryReceipt,
            "acceptance_evidence_directory_receipt_type_rejected",
        )
        _require(
            type(controller_resources) is SealedControllerResources,
            "acceptance_sealed_controller_resources_required",
        )
        evidence_directory.validate()
        root = Path(evidence_directory.absolute_path)
        _require(root.is_absolute(), "acceptance_journal_not_absolute")
        _require(root == root.resolve(strict=True), "acceptance_journal_not_canonical")
        _require(root.is_dir() and not root.is_symlink(), "acceptance_journal_rejected")
        _require(
            SHA256_RE.fullmatch(final_control_plane_plan_sha256) is not None,
            "acceptance_plan_sha256_rejected",
        )
        _require(
            SHA256_RE.fullmatch(evidence_directory.receipt_sha256) is not None,
            "acceptance_acl_sha256_rejected",
        )
        _require(
            SHA256_RE.fullmatch(final_journal_sha256) is not None,
            "acceptance_journal_anchor_rejected",
        )
        paths = EvidenceJournal._journal_paths(root)
        _require(bool(paths), "acceptance_journal_empty")
        previous_sha256: str | None = None
        entries: list[tuple[str, Mapping[str, Any]]] = []
        for sequence, path in enumerate(paths, start=1):
            _require(
                path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1,
                "acceptance_journal_entry_rejected",
            )
            raw = path.read_bytes()
            envelope = _json(raw, "acceptance_journal_entry")
            _require(type(envelope) is dict, "acceptance_journal_entry_not_object")
            _require(raw == _canonical(envelope), "acceptance_journal_entry_not_canonical")
            _require(
                set(envelope)
                == {
                    "schema_version",
                    "kind",
                    "sequence",
                    "label",
                    "control_plane_plan_sha256",
                    "evidence_directory_acl_receipt_sha256",
                    "previous_evidence_sha256",
                    "payload",
                }
                and type(envelope["schema_version"]) is int
                and envelope["schema_version"] == 1
                and envelope["kind"] == "explainiverse-lambda-live-driver-evidence"
                and type(envelope["sequence"]) is int
                and envelope["sequence"] == sequence
                and envelope["control_plane_plan_sha256"] == final_control_plane_plan_sha256
                and envelope["evidence_directory_acl_receipt_sha256"]
                == evidence_directory.receipt_sha256
                and envelope["previous_evidence_sha256"] == previous_sha256
                and type(envelope["label"]) is str
                and type(envelope["payload"]) is dict,
                "acceptance_journal_chain_rejected",
            )
            label = str(envelope["label"])
            _require(
                EVIDENCE_LABEL_RE.fullmatch(label) is not None,
                "acceptance_journal_label_rejected",
            )
            _require(
                path.name == f"{sequence:03d}-{label}.json",
                "acceptance_journal_filename_binding_rejected",
            )
            _reject_secret_keys(envelope["payload"])
            entries.append((label, envelope["payload"]))
            previous_sha256 = _sha(raw)
        _require(
            previous_sha256 == final_journal_sha256,
            "acceptance_journal_anchor_mismatch",
        )
        _require(
            entries[-1][0] == "lifecycle-restored",
            "acceptance_journal_lifecycle_not_restored",
        )
        EvidenceJournal._validate_accepted_phase_event_grammar(
            entries,
            phase="final-main",
            evidence_root=root,
            controller_resources=controller_resources,
        )

        def selected(label: str) -> list[tuple[int, Mapping[str, Any]]]:
            return [
                (index, payload)
                for index, (current, payload) in enumerate(entries)
                if current == label
            ]

        directory_evidence = selected("evidence-directory")
        immutable_plans = selected("immutable-plan")
        dispatch_intents = selected("github-dispatch-intent")
        dispatch_settlements = selected("github-dispatch-settled")
        dispatches = selected("github-dispatch")
        app_archives = selected("installed-app-raw-archive")
        app_captures = selected("installed-app-authority")
        authority_windows = selected("authority-window")
        runtime_plans = selected("runtime-plan")
        remote_cleanups = selected("remote-cleanup")
        accepted_jobs = selected("accepted-actions-job")
        acceptance_payloads = selected("final-main-acceptance")
        settlements = selected("phase-settlement")
        restored_snapshots = selected("provider-restored")
        _require(
            len(directory_evidence) == 1
            and directory_evidence[0][0] == 0
            and len(immutable_plans) == 1
            and immutable_plans[0][0] == 1
            and len(dispatch_intents) == 1
            and len(dispatch_settlements) == 1
            and len(dispatches) == 1
            and len(app_archives) == 4
            and len(app_captures) == 4
            and len(authority_windows) == 4
            and len(runtime_plans) == 4
            and len(remote_cleanups) == 4
            and len(accepted_jobs) == 4
            and len(acceptance_payloads) == 1
            and len(settlements) == 1
            and len(restored_snapshots) == 1,
            "acceptance_journal_event_cardinality",
        )
        EvidenceJournal._validate_directory_evidence_mapping(
            directory_evidence[0][1], evidence_directory
        )
        acceptance = FinalMainAcceptance._from_verified_mapping(
            acceptance_payloads[0][1],
            final_journal_sha256=final_journal_sha256,
            evidence_directory_receipt_sha256=(evidence_directory.receipt_sha256),
        )

        plan_payload = immutable_plans[0][1]
        plan_sha256 = _sha(
            json.dumps(
                plan_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
        )
        _require(
            plan_sha256 == final_control_plane_plan_sha256
            and plan_payload.get("head_sha") == acceptance.head_sha
            and type(plan_payload.get("remote_runtime")) is dict,
            "acceptance_control_plane_plan_binding_rejected",
        )
        runtime_bundle_sha256 = plan_payload["remote_runtime"].get("bundle_sha256")
        _require(
            type(runtime_bundle_sha256) is str
            and SHA256_RE.fullmatch(runtime_bundle_sha256) is not None,
            "acceptance_runtime_bundle_binding_rejected",
        )

        dispatch_intent_index, dispatch_intent = dispatch_intents[0]
        dispatch_settlement_index, dispatch_settlement = dispatch_settlements[0]
        dispatch_index, dispatch = dispatches[0]
        dispatch_session = EvidenceJournal._validate_phase_dispatch_chain(
            dispatch_intent,
            dispatch_settlement,
            dispatch,
            expected_phase="final-main",
            expected_head_sha=acceptance.head_sha,
        )
        dispatch_receipt = dict(dispatch_session.dispatch_receipt.__dict__)
        _require(
            dispatch_intent_index < dispatch_settlement_index < dispatch_index
            and dispatch_session.run["id"] == acceptance.run_id
            and dispatch_session.dispatch_receipt.nonce_history_observed_at
            == acceptance.dispatch_nonce_history_observed_at
            and dispatch_session.dispatch_receipt.nonce_history_response_sha256
            == acceptance.dispatch_nonce_history_response_sha256,
            "acceptance_dispatch_binding_rejected",
        )
        dispatch_jobs = [dict(item.__dict__) for item in dispatch_session.jobs]
        _require(
            len(dispatch_jobs) == 4 and type(dispatch_receipt.get("observed_at")) is str,
            "acceptance_dispatch_jobs_rejected",
        )

        previous_group_index = dispatch_index
        expected_keys = ("single_minimum", "single_latest", "two_minimum", "two_latest")
        seen_capture_sha256: set[str] = set()
        seen_authority_sha256: set[str] = set()
        seen_archive_sha256: set[str] = set()
        seen_page_sha256: set[str] = set()
        authority_identities: list[dict[str, Any]] = []
        previous_remote_receipt_sha256: str | None = None
        for ordinal, key in enumerate(expected_keys):
            archive_index, archive_payload = app_archives[ordinal]
            app_index, app_payload = app_captures[ordinal]
            authority_index, authority_payload = authority_windows[ordinal]
            plan_index, raw_runtime_plan = runtime_plans[ordinal]
            remote_index, remote_payload = remote_cleanups[ordinal]
            accepted_index, accepted_payload = accepted_jobs[ordinal]
            _require(
                previous_group_index
                < archive_index
                < app_index
                < authority_index
                < plan_index
                < remote_index
                < accepted_index,
                "acceptance_journal_job_order_rejected",
            )
            previous_group_index = accepted_index

            normalized_capture = app_payload.get("normalized_capture")
            _require(
                type(normalized_capture) is dict,
                "acceptance_app_capture_payload_rejected",
            )
            assert isinstance(normalized_capture, dict)
            capture_now = datetime.fromisoformat(
                str(normalized_capture.get("captured_at")).replace("Z", "+00:00")
            ).astimezone(timezone.utc)
            capture_sha = str(app_payload.get("evidence_sha256"))
            raw_manifest = normalized_capture.get("evidence")
            _require(
                type(raw_manifest) is list,
                "acceptance_app_capture_manifest_rejected",
            )
            assert isinstance(raw_manifest, list)
            expected_archive_files = [
                {
                    "filename": item.get("filename"),
                    "bytes": item.get("bytes"),
                    "sha256": item.get("sha256"),
                }
                for item in raw_manifest
                if type(item) is dict
            ]
            archive_material = {
                "capture_evidence_sha256": capture_sha,
                "archive_directory": f"installed-app-pages/{capture_sha}",
                "files": expected_archive_files,
                "all_pages_exclusive_single_link": True,
            }
            _require(
                archive_payload
                == {
                    **archive_material,
                    "archive_evidence_sha256": _sha(_canonical(archive_material)),
                },
                "acceptance_app_archive_binding_rejected",
            )

            def read_capture_evidence(filename: str) -> bytes:
                return EvidenceJournal._read_archived_app_page(root, capture_sha, filename)

            rebuilt_capture = TrustedAppCapture.from_mapping(
                normalized_capture,
                resources=controller_resources,
                evidence_reader=read_capture_evidence,
                now=capture_now,
            )
            _require(
                rebuilt_capture.to_mapping() == app_payload,
                "acceptance_app_capture_binding_drift",
            )

            _require(type(raw_runtime_plan) is dict, "acceptance_runtime_plan_not_object")
            assert isinstance(raw_runtime_plan, dict)
            created_at = _parse_time(
                raw_runtime_plan.get("created_at"), "acceptance_runtime_plan_created"
            )
            normalized_plan = runtime.validate_runtime_plan(raw_runtime_plan, now=created_at)
            _require(
                normalized_plan == raw_runtime_plan
                and raw_runtime_plan.get("phase") == "final-main"
                and raw_runtime_plan.get("control_plane_plan_sha256")
                == final_control_plane_plan_sha256
                and raw_runtime_plan.get("runtime_bundle_sha256") == runtime_bundle_sha256
                and raw_runtime_plan.get("dispatch", {}).get("run_id") == acceptance.run_id
                and raw_runtime_plan.get("dispatch", {}).get("head_sha") == acceptance.head_sha
                and raw_runtime_plan.get("dispatch", {}).get("observed_at")
                == dispatch_receipt.get("observed_at")
                and raw_runtime_plan.get("job", {}).get("key") == key
                and raw_runtime_plan.get("sequencing", {}).get("previous_cleanup_receipt_sha256")
                == previous_remote_receipt_sha256,
                "acceptance_runtime_plan_binding_drift",
            )
            authority_identity = EvidenceJournal._validate_job_authority_evidence(
                capture=rebuilt_capture,
                app_payload=app_payload,
                archive_payload=archive_payload,
                authority_payload=authority_payload,
                runtime_plan=normalized_plan,
            )
            capture_sha256 = authority_identity["capture_evidence_sha256"]
            authority_sha256 = authority_identity["authority_evidence_sha256"]
            archive_sha256 = authority_identity["archive_evidence_sha256"]
            page_sha256 = authority_identity["raw_page_sha256"]
            _require(
                capture_sha256 not in seen_capture_sha256
                and authority_sha256 not in seen_authority_sha256
                and archive_sha256 not in seen_archive_sha256
                and seen_page_sha256.isdisjoint(page_sha256),
                "acceptance_authority_evidence_replayed",
            )
            seen_capture_sha256.add(capture_sha256)
            seen_authority_sha256.add(authority_sha256)
            seen_archive_sha256.add(archive_sha256)
            seen_page_sha256.update(page_sha256)
            authority_identities.append(authority_identity)
            runtime_plan_sha256 = runtime.runtime_plan_sha256(normalized_plan)

            _require(
                set(remote_payload)
                == {"receipt", "stdout_sha256", "stderr_sha256", "frame_receipt"}
                and type(remote_payload.get("receipt")) is dict
                and type(remote_payload.get("frame_receipt")) is dict
                and all(
                    type(remote_payload.get(field)) is str
                    and SHA256_RE.fullmatch(str(remote_payload[field])) is not None
                    for field in ("stdout_sha256", "stderr_sha256")
                ),
                "acceptance_remote_cleanup_schema_rejected",
            )
            execution = RemoteExecution(
                dict(remote_payload["receipt"]),
                str(remote_payload["stdout_sha256"]),
                str(remote_payload["stderr_sha256"]),
                dict(remote_payload["frame_receipt"]),
            )
            remote_receipt_sha256 = ReleaseGpuController._validate_remote_receipt(
                normalized_plan, execution
            )
            previous_remote_receipt_sha256 = remote_receipt_sha256

            accepted = EvidenceJournal._validate_accepted_job_receipt(
                accepted_payload,
                phase="final-main",
                run_id=acceptance.run_id,
                job_key=key,
                job_id=raw_runtime_plan["job"]["job_id"],
                runner_id=raw_runtime_plan["job"]["runner_id"],
                runner_name=raw_runtime_plan["job"]["runner_name"],
                runtime_plan_sha256=runtime_plan_sha256,
                remote_receipt_sha256=remote_receipt_sha256,
                context="acceptance_job_receipt",
            )
            _require(
                accepted.to_mapping() == acceptance.jobs[ordinal],
                "acceptance_job_receipt_binding_drift",
            )
            _require(
                type(dispatch_jobs[ordinal]) is dict,
                "acceptance_dispatch_job_not_object",
            )
            dispatch_job = dispatch_jobs[ordinal]
            assert isinstance(dispatch_job, dict)
            _require(
                dispatch_job.get("key") == key
                and dispatch_job.get("job_id") == accepted.job_id
                and dispatch_job.get("runner_name") == accepted.runner_name,
                "acceptance_dispatch_job_binding_drift",
            )

        acceptance = FinalMainAcceptance._from_verified_mapping(
            acceptance_payloads[0][1],
            final_journal_sha256=final_journal_sha256,
            evidence_directory_receipt_sha256=evidence_directory.receipt_sha256,
            authority_evidence_identities=authority_identities,
        )

        acceptance_index = acceptance_payloads[0][0]
        settlement_index, settlement_payload = settlements[0]
        restored_index, restored_payload = restored_snapshots[0]
        _require(
            previous_group_index < acceptance_index < settlement_index < restored_index,
            "acceptance_journal_final_order_rejected",
        )
        EvidenceJournal._validate_phase_settlement(
            settlement_payload,
            phase="final-main",
            run_id=acceptance.run_id,
            head_sha=acceptance.head_sha,
            expected_job_evidence_sha256=[item["evidence_sha256"] for item in acceptance.jobs],
            expected_nonces=acceptance.accepted_cuda_runner_nonces,
            context="acceptance_settlement_payload",
        )
        _require(
            settlement_payload == acceptance.settlement,
            "acceptance_settlement_payload_drift",
        )
        EvidenceJournal._validate_provider_restored_snapshot(
            restored_payload,
            plan_sha256=final_control_plane_plan_sha256,
            context="acceptance_provider_restored_snapshot_rejected",
        )
        lifecycle_payload = entries[-1][1]
        _require(
            set(lifecycle_payload)
            == {
                "plan_sha256",
                "provider_instances",
                "provider_firewall_rulesets",
                "global_firewall_restored",
                "repository_runners",
                "known_hosts_sha256",
            }
            and lifecycle_payload["plan_sha256"] == final_control_plane_plan_sha256
            and type(lifecycle_payload["provider_instances"]) is int
            and lifecycle_payload["provider_instances"] == 0
            and type(lifecycle_payload["provider_firewall_rulesets"]) is int
            and lifecycle_payload["provider_firewall_rulesets"] == 0
            and lifecycle_payload["global_firewall_restored"] is True
            and type(lifecycle_payload["repository_runners"]) is int
            and lifecycle_payload["repository_runners"] == 0
            and type(lifecycle_payload["known_hosts_sha256"]) is str
            and SHA256_RE.fullmatch(lifecycle_payload["known_hosts_sha256"]) is not None,
            "acceptance_lifecycle_restoration_rejected",
        )
        return acceptance

    @staticmethod
    def load_publication_recovery_source(
        evidence_directory: live.EvidenceDirectoryReceipt,
        *,
        controller_resources: SealedControllerResources,
        publication_control_plane_plan_sha256: str,
        publication_journal_sha256: str,
        source_run_id: int,
    ) -> PublicationRecoverySource:
        """Verify the restored publication GPU journal before recovery dispatch.

        ``publication_journal_sha256`` names the clean ``lifecycle-restored``
        boundary.  A crash-safe recovery dispatch may already have appended
        only its narrowly defined intent/settlement records after that anchor.
        """

        _require(
            type(evidence_directory) is live.EvidenceDirectoryReceipt,
            "publication_source_evidence_directory_receipt_type_rejected",
        )
        _require(
            type(controller_resources) is SealedControllerResources,
            "publication_source_sealed_controller_resources_required",
        )
        evidence_directory.validate()
        root = Path(evidence_directory.absolute_path)
        _require(
            root.is_absolute()
            and root == root.resolve(strict=True)
            and root.is_dir()
            and not root.is_symlink(),
            "publication_source_journal_rejected",
        )
        _require(
            SHA256_RE.fullmatch(publication_control_plane_plan_sha256) is not None
            and SHA256_RE.fullmatch(publication_journal_sha256) is not None
            and type(source_run_id) is int
            and source_run_id > 0,
            "publication_source_arguments_rejected",
        )
        paths = EvidenceJournal._journal_paths(root)
        _require(bool(paths), "publication_source_journal_empty")
        previous_sha256: str | None = None
        entries: list[tuple[str, Mapping[str, Any], str]] = []
        for sequence, path in enumerate(paths, start=1):
            _require(
                path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1,
                "publication_source_journal_entry_rejected",
            )
            raw = path.read_bytes()
            envelope = _json(raw, "publication_source_journal_entry")
            _require(
                type(envelope) is dict
                and raw == _canonical(envelope)
                and set(envelope)
                == {
                    "schema_version",
                    "kind",
                    "sequence",
                    "label",
                    "control_plane_plan_sha256",
                    "evidence_directory_acl_receipt_sha256",
                    "previous_evidence_sha256",
                    "payload",
                }
                and type(envelope["schema_version"]) is int
                and envelope["schema_version"] == 1
                and envelope["kind"] == "explainiverse-lambda-live-driver-evidence"
                and type(envelope["sequence"]) is int
                and envelope["sequence"] == sequence
                and envelope["control_plane_plan_sha256"] == publication_control_plane_plan_sha256
                and envelope["evidence_directory_acl_receipt_sha256"]
                == evidence_directory.receipt_sha256
                and envelope["previous_evidence_sha256"] == previous_sha256
                and type(envelope["label"]) is str
                and EVIDENCE_LABEL_RE.fullmatch(envelope["label"]) is not None
                and type(envelope["payload"]) is dict,
                "publication_source_journal_chain_rejected",
            )
            _require(
                path.name == f"{sequence:03d}-{envelope['label']}.json",
                "publication_source_journal_filename_binding_rejected",
            )
            _reject_secret_keys(envelope["payload"])
            previous_sha256 = _sha(raw)
            entries.append((envelope["label"], envelope["payload"], previous_sha256))
        anchor_matches = [
            index
            for index, (_, _, digest) in enumerate(entries)
            if digest == publication_journal_sha256
        ]
        _require(
            len(anchor_matches) == 1,
            "publication_source_journal_anchor_missing",
        )
        anchor_index = anchor_matches[0]
        _require(
            entries[anchor_index][0] == "lifecycle-restored",
            "publication_source_anchor_not_lifecycle_restored",
        )
        allowed_suffix = {
            "github-recovery-dispatch-intent",
            "github-recovery-dispatch-settled",
            "journal-publish-recovery",
            "operator-publication-recovery-source",
            "operator-release-recovery-dispatch-settled",
        }
        _require(
            all(label in allowed_suffix for label, _, _ in entries[anchor_index + 1 :]),
            "publication_source_post_restoration_event_rejected",
        )
        seen_recovered_temporary_names: set[str] = set()
        seen_recovery_sidecar_names: set[str] = set()
        journal_recovery_payloads: list[Mapping[str, Any]] = []
        semantic_suffix: list[tuple[str, Mapping[str, Any], str]] = []
        for absolute_index, entry in enumerate(
            entries[anchor_index + 1 :],
            start=anchor_index + 1,
        ):
            label, payload, _ = entry
            if label == "journal-publish-recovery":
                EvidenceJournal._validate_journal_publish_recovery_payload(
                    payload,
                    record_sequence=absolute_index + 1,
                    root=root,
                    plan_sha256=publication_control_plane_plan_sha256,
                    acl_sha256=evidence_directory.receipt_sha256,
                    seen_temporary_names=seen_recovered_temporary_names,
                    seen_sidecar_names=seen_recovery_sidecar_names,
                )
                journal_recovery_payloads.append(payload)
            else:
                semantic_suffix.append(entry)
        semantic_entries = [*entries[: anchor_index + 1], *semantic_suffix]
        lifecycle_entries = entries[: anchor_index + 1]
        EvidenceJournal._validate_accepted_phase_event_grammar(
            [(label, payload) for label, payload, _ in lifecycle_entries],
            phase="publication",
            evidence_root=root,
            controller_resources=controller_resources,
            journal_recovery_payloads=journal_recovery_payloads,
        )

        def selected(label: str) -> list[tuple[int, Mapping[str, Any]]]:
            return [
                (index, payload)
                for index, (current, payload, _) in enumerate(lifecycle_entries)
                if current == label
            ]

        directory_evidence = selected("evidence-directory")
        immutable_plans = selected("immutable-plan")
        dispatch_intents = selected("github-dispatch-intent")
        dispatch_settlements = selected("github-dispatch-settled")
        dispatches = selected("github-dispatch")
        app_archives = selected("installed-app-raw-archive")
        app_captures = selected("installed-app-authority")
        authority_windows = selected("authority-window")
        runtime_plans = selected("runtime-plan")
        remote_cleanups = selected("remote-cleanup")
        accepted_jobs = selected("accepted-actions-job")
        settlements = selected("phase-settlement")
        restored_snapshots = selected("provider-restored")
        lifecycle_boundaries = selected("lifecycle-restored")
        _require(
            len(directory_evidence) == 1
            and directory_evidence[0][0] == 0
            and len(immutable_plans) == 1
            and immutable_plans[0][0] == 1
            and len(dispatch_intents) == 1
            and len(dispatch_settlements) == 1
            and len(dispatches) == 1
            and len(app_archives) == 2
            and len(app_captures) == 2
            and len(authority_windows) == 2
            and len(runtime_plans) == 2
            and len(remote_cleanups) == 2
            and len(accepted_jobs) == 2
            and len(settlements) == 1
            and len(restored_snapshots) == 1
            and len(lifecycle_boundaries) == 1
            and lifecycle_boundaries[0][0] == anchor_index
            and not selected("final-main-acceptance"),
            "publication_source_journal_event_cardinality",
        )
        EvidenceJournal._validate_directory_evidence_mapping(
            directory_evidence[0][1], evidence_directory
        )
        plan_payload = immutable_plans[0][1]
        plan_sha256 = _sha(
            json.dumps(
                plan_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
        )
        head_sha = plan_payload.get("head_sha")
        remote_runtime = plan_payload.get("remote_runtime")
        _require(
            plan_sha256 == publication_control_plane_plan_sha256
            and type(head_sha) is str
            and re.fullmatch(r"[0-9a-f]{40}", head_sha) is not None
            and type(remote_runtime) is dict,
            "publication_source_control_plane_plan_binding_rejected",
        )
        assert isinstance(remote_runtime, dict) and isinstance(head_sha, str)
        runtime_bundle_sha256 = remote_runtime.get("bundle_sha256")
        _require(
            type(runtime_bundle_sha256) is str
            and SHA256_RE.fullmatch(runtime_bundle_sha256) is not None,
            "publication_source_runtime_bundle_rejected",
        )

        dispatch_intent_index, dispatch_intent = dispatch_intents[0]
        dispatch_settlement_index, dispatch_settlement = dispatch_settlements[0]
        dispatch_index, dispatch = dispatches[0]
        dispatch_session = EvidenceJournal._validate_phase_dispatch_chain(
            dispatch_intent,
            dispatch_settlement,
            dispatch,
            expected_phase="publication",
            expected_head_sha=head_sha,
        )
        dispatch_jobs = [dict(item.__dict__) for item in dispatch_session.jobs]
        dispatch_receipt = dict(dispatch_session.dispatch_receipt.__dict__)
        dispatch_inputs = dispatch_session.inputs
        cuda_run_id = (
            dispatch_inputs.get("cuda_run_id") if isinstance(dispatch_inputs, dict) else None
        )
        raw_prior_authority = dispatch.get("prior_authority_evidence_identities")
        _require(
            dispatch_intent_index < dispatch_settlement_index < dispatch_index
            and dispatch_session.run["id"] == source_run_id
            and len(dispatch_jobs) == 2
            and type(dispatch_receipt.get("observed_at")) is str
            and type(cuda_run_id) is int
            and cuda_run_id > 0
            and type(raw_prior_authority) is list
            and len(raw_prior_authority) == 4,
            "publication_source_dispatch_binding_rejected",
        )
        assert isinstance(raw_prior_authority, list)
        assert isinstance(cuda_run_id, int)
        prior_authority_identities = tuple(
            _validated_authority_evidence_identity(
                item,
                context=f"publication_source_prior_authority_{ordinal}",
                expected_phase="final-main",
                expected_head_sha=head_sha,
                expected_run_id=cuda_run_id,
                expected_job_key=key,
            )
            for ordinal, (key, item) in enumerate(
                zip(
                    ("single_minimum", "single_latest", "two_minimum", "two_latest"),
                    raw_prior_authority,
                ),
                start=1,
            )
        )
        _require(
            len({item["capture_evidence_sha256"] for item in prior_authority_identities}) == 4
            and len({item["authority_evidence_sha256"] for item in prior_authority_identities}) == 4
            and len({item["archive_evidence_sha256"] for item in prior_authority_identities}) == 4
            and len(
                {page for item in prior_authority_identities for page in item["raw_page_sha256"]}
            )
            == sum(len(item["raw_page_sha256"]) for item in prior_authority_identities),
            "publication_source_prior_authority_replayed",
        )

        expected_keys = ("publication_single_minimum", "publication_single_latest")
        accepted_receipts: list[AcceptedJobReceipt] = []
        previous_group_index = dispatch_index
        previous_remote_receipt_sha256: str | None = None
        seen_capture_sha256 = {
            item["capture_evidence_sha256"] for item in prior_authority_identities
        }
        seen_authority_sha256 = {
            item["authority_evidence_sha256"] for item in prior_authority_identities
        }
        seen_archive_sha256 = {
            item["archive_evidence_sha256"] for item in prior_authority_identities
        }
        seen_page_sha256 = {
            page for item in prior_authority_identities for page in item["raw_page_sha256"]
        }
        for ordinal, key in enumerate(expected_keys):
            archive_index, archive_payload = app_archives[ordinal]
            app_index, app_payload = app_captures[ordinal]
            authority_index, authority_payload = authority_windows[ordinal]
            plan_index, raw_runtime_plan = runtime_plans[ordinal]
            remote_index, remote_payload = remote_cleanups[ordinal]
            accepted_index, accepted_payload = accepted_jobs[ordinal]
            _require(
                previous_group_index
                < archive_index
                < app_index
                < authority_index
                < plan_index
                < remote_index
                < accepted_index,
                "publication_source_job_order_rejected",
            )
            previous_group_index = accepted_index

            normalized_capture = app_payload.get("normalized_capture")
            _require(
                type(normalized_capture) is dict,
                "publication_source_app_capture_payload_rejected",
            )
            assert isinstance(normalized_capture, dict)
            try:
                capture_now = datetime.fromisoformat(
                    str(normalized_capture.get("captured_at")).replace("Z", "+00:00")
                ).astimezone(timezone.utc)
            except ValueError:
                raise ControllerError("publication_source_app_capture_timestamp_rejected") from None
            capture_sha256 = app_payload.get("evidence_sha256")
            raw_manifest = normalized_capture.get("evidence")
            _require(
                type(capture_sha256) is str
                and SHA256_RE.fullmatch(capture_sha256) is not None
                and type(raw_manifest) is list,
                "publication_source_app_capture_manifest_rejected",
            )
            assert isinstance(raw_manifest, list)
            assert isinstance(capture_sha256, str)
            bound_capture_sha256 = capture_sha256
            expected_archive_files = [
                {
                    "filename": item.get("filename"),
                    "bytes": item.get("bytes"),
                    "sha256": item.get("sha256"),
                }
                for item in raw_manifest
                if type(item) is dict
            ]
            archive_material = {
                "capture_evidence_sha256": capture_sha256,
                "archive_directory": f"installed-app-pages/{capture_sha256}",
                "files": expected_archive_files,
                "all_pages_exclusive_single_link": True,
            }
            _require(
                archive_payload
                == {
                    **archive_material,
                    "archive_evidence_sha256": _sha(_canonical(archive_material)),
                },
                "publication_source_app_archive_binding_rejected",
            )

            def read_capture_evidence(filename: str) -> bytes:
                return EvidenceJournal._read_archived_app_page(
                    root,
                    bound_capture_sha256,
                    filename,
                )

            rebuilt_capture = TrustedAppCapture.from_mapping(
                normalized_capture,
                resources=controller_resources,
                evidence_reader=read_capture_evidence,
                now=capture_now,
            )
            _require(
                rebuilt_capture.to_mapping() == app_payload,
                "publication_source_app_capture_binding_drift",
            )

            _require(
                type(raw_runtime_plan) is dict,
                "publication_source_runtime_plan_not_object",
            )
            assert isinstance(raw_runtime_plan, dict)
            created_at = _parse_time(
                raw_runtime_plan.get("created_at"),
                "publication_source_runtime_plan_created",
            )
            normalized_plan = runtime.validate_runtime_plan(raw_runtime_plan, now=created_at)
            _require(
                normalized_plan == raw_runtime_plan
                and raw_runtime_plan.get("phase") == "publication"
                and raw_runtime_plan.get("control_plane_plan_sha256")
                == publication_control_plane_plan_sha256
                and raw_runtime_plan.get("runtime_bundle_sha256") == runtime_bundle_sha256
                and raw_runtime_plan.get("dispatch", {}).get("run_id") == source_run_id
                and raw_runtime_plan.get("dispatch", {}).get("head_sha") == head_sha
                and raw_runtime_plan.get("dispatch", {}).get("observed_at")
                == dispatch_receipt.get("observed_at")
                and raw_runtime_plan.get("job", {}).get("key") == key
                and raw_runtime_plan.get("sequencing", {}).get("previous_cleanup_receipt_sha256")
                == previous_remote_receipt_sha256,
                "publication_source_runtime_plan_binding_drift",
            )
            authority_identity = EvidenceJournal._validate_job_authority_evidence(
                capture=rebuilt_capture,
                app_payload=app_payload,
                archive_payload=archive_payload,
                authority_payload=authority_payload,
                runtime_plan=normalized_plan,
            )
            capture_sha256 = authority_identity["capture_evidence_sha256"]
            authority_sha256 = authority_identity["authority_evidence_sha256"]
            archive_sha256 = authority_identity["archive_evidence_sha256"]
            page_sha256 = authority_identity["raw_page_sha256"]
            _require(
                capture_sha256 not in seen_capture_sha256
                and authority_sha256 not in seen_authority_sha256
                and archive_sha256 not in seen_archive_sha256
                and seen_page_sha256.isdisjoint(page_sha256),
                "publication_source_authority_evidence_replayed",
            )
            seen_capture_sha256.add(capture_sha256)
            seen_authority_sha256.add(authority_sha256)
            seen_archive_sha256.add(archive_sha256)
            seen_page_sha256.update(page_sha256)
            runtime_plan_sha256 = runtime.runtime_plan_sha256(normalized_plan)

            _require(
                set(remote_payload)
                == {"receipt", "stdout_sha256", "stderr_sha256", "frame_receipt"}
                and type(remote_payload.get("receipt")) is dict
                and type(remote_payload.get("frame_receipt")) is dict
                and all(
                    type(remote_payload.get(field)) is str
                    and SHA256_RE.fullmatch(str(remote_payload[field])) is not None
                    for field in ("stdout_sha256", "stderr_sha256")
                ),
                "publication_source_remote_cleanup_schema_rejected",
            )
            execution = RemoteExecution(
                dict(remote_payload["receipt"]),
                str(remote_payload["stdout_sha256"]),
                str(remote_payload["stderr_sha256"]),
                dict(remote_payload["frame_receipt"]),
            )
            remote_receipt_sha256 = ReleaseGpuController._validate_remote_receipt(
                normalized_plan, execution
            )
            previous_remote_receipt_sha256 = remote_receipt_sha256

            accepted = EvidenceJournal._validate_accepted_job_receipt(
                accepted_payload,
                phase="publication",
                run_id=source_run_id,
                job_key=key,
                job_id=raw_runtime_plan["job"]["job_id"],
                runner_id=raw_runtime_plan["job"]["runner_id"],
                runner_name=raw_runtime_plan["job"]["runner_name"],
                runtime_plan_sha256=runtime_plan_sha256,
                remote_receipt_sha256=remote_receipt_sha256,
                context="publication_source_job_receipt",
            )
            dispatch_job = dispatch_jobs[ordinal]
            _require(
                type(dispatch_job) is dict
                and dispatch_job.get("key") == key
                and dispatch_job.get("job_id") == accepted.job_id
                and dispatch_job.get("runner_name") == accepted.runner_name,
                "publication_source_dispatch_job_binding_drift",
            )
            accepted_receipts.append(accepted)

        _require(
            len({item.job_id for item in accepted_receipts}) == 2
            and len({item.runner_id for item in accepted_receipts}) == 2
            and len({item.evidence_sha256 for item in accepted_receipts}) == 2,
            "publication_source_job_identity_reuse_rejected",
        )

        settlement_index, settlement_payload = settlements[0]
        restored_index, restored_payload = restored_snapshots[0]
        expected_job_evidence = [item.evidence_sha256 for item in accepted_receipts]
        _require(
            previous_group_index < settlement_index < restored_index < anchor_index,
            "publication_source_journal_final_order_rejected",
        )
        settlement_evidence_sha256 = EvidenceJournal._validate_phase_settlement(
            settlement_payload,
            phase="publication",
            run_id=source_run_id,
            head_sha=head_sha,
            expected_job_evidence_sha256=expected_job_evidence,
            context="publication_source_settlement_binding",
        )
        EvidenceJournal._validate_provider_restored_snapshot(
            restored_payload,
            plan_sha256=publication_control_plane_plan_sha256,
            context="publication_source_provider_restored_snapshot_rejected",
        )
        lifecycle_payload = lifecycle_entries[anchor_index][1]
        _require(
            set(lifecycle_payload)
            == {
                "plan_sha256",
                "provider_instances",
                "provider_firewall_rulesets",
                "global_firewall_restored",
                "repository_runners",
                "known_hosts_sha256",
            }
            and lifecycle_payload.get("plan_sha256") == publication_control_plane_plan_sha256
            and type(lifecycle_payload.get("provider_instances")) is int
            and lifecycle_payload.get("provider_instances") == 0
            and type(lifecycle_payload.get("provider_firewall_rulesets")) is int
            and lifecycle_payload.get("provider_firewall_rulesets") == 0
            and lifecycle_payload.get("global_firewall_restored") is True
            and type(lifecycle_payload.get("repository_runners")) is int
            and lifecycle_payload.get("repository_runners") == 0
            and type(lifecycle_payload.get("known_hosts_sha256")) is str
            and SHA256_RE.fullmatch(lifecycle_payload["known_hosts_sha256"]) is not None,
            "publication_source_lifecycle_restoration_rejected",
        )
        source = PublicationRecoverySource._from_verified(
            head_sha=head_sha,
            run_id=source_run_id,
            control_plane_plan_sha256=publication_control_plane_plan_sha256,
            publication_journal_sha256=publication_journal_sha256,
            evidence_directory_receipt_sha256=evidence_directory.receipt_sha256,
            job_evidence_sha256=expected_job_evidence,
            phase_settlement_evidence_sha256=settlement_evidence_sha256,
        )
        tail = EvidenceJournal._validate_publication_recovery_suffix(
            semantic_entries,
            anchor_index=anchor_index,
            source=source,
        )
        object.__setattr__(source, "_recovery_tail", tail)
        return source


@dataclass(frozen=True)
class PhaseCompletion:
    phase: str
    head_sha: str
    run_id: int
    settlement: Mapping[str, Any]
    final_evidence_sha256: str
    final_main_acceptance: FinalMainAcceptance | None


class LiveReleaseDriver:
    """Sequence one provider lifecycle and exactly one GitHub evidence phase.

    A PR, final-main, and publication run each use a fresh instance of this
    class and a fresh immutable provider plan.  Reusing a host across a merge
    boundary is rejected because the provider plan is pinned to one source SHA.
    """

    _NEXT_PHASE = {
        "restrict_global": "global_restricted",
        "create_ruleset": "ruleset_ready",
        "launch": "instance_bound",
        "terminate": "instance_absent",
        "delete_ruleset": "ruleset_absent",
        "restore_global": "restored",
    }

    @staticmethod
    def _bind_provider_intent_sink(provider: ProviderLifecycle, journal: EvidenceJournal) -> str:
        """Bind the exact durable journal before the provider can be observed."""

        callback = journal.record_provider_mutation_intent
        existing = provider.mutation_intent_binding_sha256
        if existing is None:
            binding_sha256 = provider.bind_mutation_intent_callback(callback)
        else:
            _require(
                provider.mutation_intent_callback_matches(callback),
                "driver_provider_intent_sink_mismatch",
            )
            binding_sha256 = existing
        _require(
            SHA256_RE.fullmatch(binding_sha256) is not None
            and provider.mutation_intent_binding_sha256 == binding_sha256
            and provider.mutation_intent_callback_matches(callback),
            "driver_provider_intent_sink_binding_rejected",
        )
        return binding_sha256

    def __init__(
        self,
        controller: ReleaseGpuController,
        provider: ProviderLifecycle,
        plan: live.ImmutablePlan,
        identity: live.HostIdentity,
        runtime_bundle: live.RuntimeBundle,
        journal: EvidenceJournal,
        *,
        access_identity: live.AccessIdentityReceipt,
        known_hosts_path: str | Path,
        sleep: Callable[[float], None] = time.sleep,
        observation_poll_limit: int = 24,
    ) -> None:
        _require(provider.plan_sha256 == plan.sha256, "driver_provider_plan_mismatch")
        _require(journal._plan_sha256 == plan.sha256, "driver_journal_plan_mismatch")
        _require(identity.fingerprint == plan.host_key_fingerprint, "driver_host_key_mismatch")
        _require(
            runtime_bundle.sha256 == plan.runtime_bundle_sha256,
            "driver_runtime_bundle_mismatch",
        )
        known_path = Path(known_hosts_path)
        _require(
            type(access_identity) is live.AccessIdentityReceipt,
            "driver_access_identity_receipt_rejected",
        )
        _require(known_path.is_absolute(), "driver_known_hosts_not_absolute")
        _require(
            known_path.parent.resolve(strict=True) == journal.directory,
            "driver_known_hosts_outside_evidence_directory",
        )
        _require(not known_path.exists(), "driver_known_hosts_already_exists")
        _require(callable(sleep), "driver_sleep_not_callable")
        _require(
            type(observation_poll_limit) is int and observation_poll_limit > 0,
            "driver_observation_poll_limit_rejected",
        )
        try:
            provider_intent_binding_sha256 = self._bind_provider_intent_sink(provider, journal)
            access_evidence = controller.bind_access_identity(
                access_identity,
                expected_public_key_sha256=plan.ssh_public_key_sha256,
            )
            access_identity_file = access_identity.absolute_path
        except BaseException:
            if not identity.destroyed:
                identity.destroy()
            if not access_identity.closed:
                try:
                    controller.close_access_identity()
                except BaseException:
                    pass
            try:
                journal.close()
            except BaseException:
                pass
            raise
        self._controller = controller
        self._provider = provider
        self._plan = plan
        self._identity = identity
        self._runtime_bundle = runtime_bundle
        self._journal = journal
        self._access_identity: live.AccessIdentityReceipt | None = access_identity
        self._access_identity_file = access_identity_file
        self._access_identity_evidence: Mapping[str, Any] | None = access_evidence
        self._known_hosts_path = str(known_path)
        self._sleep = sleep
        self._observation_poll_limit = observation_poll_limit
        self._state = "planned"
        self._readiness: HostReadinessReceipt | None = None
        self._known_hosts: live.KnownHostsFileReceipt | None = None
        self._bindings: dict[str, live.StrictSshBinding] = {}
        self._phase: str | None = None
        self._session: PhaseSession | None = None
        self._active_job: Any | None = None
        self._dispatch_ambiguity: AmbiguousGitHubMutation | None = None
        self._github_ambiguity: AmbiguousGitHubMutation | None = None
        self._remote_ambiguity_receipt: dict[str, Any] | None = None
        self._crash_jit_intent: dict[str, Any] | None = None
        self._runner_delete_intent: dict[str, Any] | None = None
        self._provider_crash_ambiguity: live.AmbiguousMutation | None = None
        self._provider_intent_binding_sha256 = provider_intent_binding_sha256
        self._cleanup_evidence_errors: list[BaseException] | None = None

    @property
    def state(self) -> str:
        return self._state

    @staticmethod
    def _session_from_journal(
        value: Mapping[str, Any], accepted_payloads: Sequence[Mapping[str, Any]]
    ) -> PhaseSession:
        required = {
            "phase",
            "workflow",
            "workflow_path",
            "dispatch_ref",
            "run_ref",
            "head_sha",
            "inputs",
            "prior_accepted_cuda_runner_nonces",
            "prior_authority_evidence_identities",
            "run_id",
            "run_attempt",
            "jobs",
            "queued_jobs",
            "dispatch_receipt",
        }
        _require(set(value) == required, "recovery_session_schema_rejected")
        phase = value["phase"]
        _require(type(phase) is str and phase in PHASES, "recovery_session_phase_rejected")
        spec = PHASES[phase]
        _require(
            value["workflow"] == spec["workflow"]
            and value["workflow_path"] == spec["workflow_path"]
            and value["dispatch_ref"] == spec["dispatch_ref"]
            and value["run_ref"] == spec["run_ref"]
            and type(value["head_sha"]) is str
            and re.fullmatch(r"[0-9a-f]{40}", value["head_sha"]) is not None
            and type(value["run_id"]) is int
            and value["run_id"] > 0
            and type(value["run_attempt"]) is int
            and value["run_attempt"] == 1
            and type(value["inputs"]) is dict
            and type(value["prior_accepted_cuda_runner_nonces"]) is list
            and type(value["prior_authority_evidence_identities"]) is list
            and type(value["jobs"]) is list
            and type(value["queued_jobs"]) is list
            and type(value["dispatch_receipt"]) is dict,
            "recovery_session_binding_rejected",
        )
        inputs = value["inputs"]
        raw_prior_nonces = value["prior_accepted_cuda_runner_nonces"]
        assert isinstance(inputs, dict) and isinstance(raw_prior_nonces, list)
        nonce_keys = tuple(spec["all_nonce_keys"])
        if phase == "publication":
            expected_input_keys = {
                "tag",
                "preflight_run_id",
                "cuda_run_id",
                "stage_recovery_drill",
                *nonce_keys,
            }
            _require(
                set(inputs) == expected_input_keys
                and inputs["tag"] == runtime.PUBLICATION_TAG
                and type(inputs["preflight_run_id"]) is int
                and inputs["preflight_run_id"] > 0
                and type(inputs["cuda_run_id"]) is int
                and inputs["cuda_run_id"] > 0
                and inputs["stage_recovery_drill"] is True
                and len(raw_prior_nonces) == 4
                and len(set(raw_prior_nonces)) == 4
                and all(
                    type(item) is str and re.fullmatch(r"[0-9a-f]{16}", item) is not None
                    for item in raw_prior_nonces
                ),
                "recovery_session_publication_inputs_rejected",
            )
        else:
            _require(
                set(inputs) == set(nonce_keys) and raw_prior_nonces == [],
                "recovery_session_cuda_inputs_rejected",
            )
        nonce_values = [inputs[key] for key in nonce_keys]
        _require(
            len(set(nonce_values)) == len(nonce_values)
            and all(
                type(item) is str and re.fullmatch(r"[0-9a-f]{16}", item) is not None
                for item in nonce_values
            ),
            "recovery_session_nonce_inputs_rejected",
        )

        job_fields = set(JobBinding.__dataclass_fields__)

        def validated_jobs(raw_jobs: Any, expected_keys: Sequence[str]) -> tuple[JobBinding, ...]:
            _require(
                type(raw_jobs) is list and len(raw_jobs) == len(expected_keys),
                "recovery_session_job_cardinality_rejected",
            )
            result: list[JobBinding] = []
            for ordinal, (key, raw_job) in enumerate(zip(expected_keys, raw_jobs), start=1):
                _require(
                    type(raw_job) is dict and set(raw_job) == job_fields,
                    "recovery_session_nested_job_schema_rejected",
                )
                nonce = inputs[nonce_keys[ordinal - 1]]
                runner_name = f"{runtime.JOB_SPECS[key]['prefix']}{nonce}"
                _require(
                    raw_job["key"] == key
                    and type(raw_job["ordinal"]) is int
                    and raw_job["ordinal"] == ordinal
                    and type(raw_job["job_id"]) is int
                    and raw_job["job_id"] > 0
                    and raw_job["name"] == runtime.JOB_SPECS[key]["name"]
                    and raw_job["nonce"] == nonce
                    and raw_job["runner_name"] == runner_name,
                    "recovery_session_job_binding_rejected",
                )
                result.append(
                    JobBinding(
                        key=key,
                        ordinal=ordinal,
                        job_id=raw_job["job_id"],
                        name=str(raw_job["name"]),
                        nonce=str(nonce),
                        runner_name=runner_name,
                    )
                )
            _require(
                len({item.job_id for item in result}) == len(result),
                "recovery_session_job_id_reuse",
            )
            return tuple(result)

        queued_jobs = validated_jobs(value["queued_jobs"], tuple(spec["queued_job_keys"]))
        jobs = validated_jobs(value["jobs"], tuple(spec["job_keys"]))
        queued_by_key = {item.key: item for item in queued_jobs}
        _require(
            all(item == queued_by_key.get(item.key) for item in jobs),
            "recovery_session_service_job_binding_rejected",
        )
        dispatch_receipt = DispatchReceipt.from_mapping(value["dispatch_receipt"])
        _require(
            tuple(item.key for item in jobs) == tuple(spec["job_keys"])
            and tuple(item.key for item in queued_jobs) == tuple(spec["queued_job_keys"])
            and len({item.job_id for item in queued_jobs}) == len(queued_jobs),
            "recovery_session_job_binding_rejected",
        )
        raw_prior_authority = value["prior_authority_evidence_identities"]
        assert isinstance(raw_prior_authority, list)
        expected_prior_keys = (
            ("single_minimum", "single_latest", "two_minimum", "two_latest")
            if phase == "publication"
            else ()
        )
        expected_prior_run = (
            value["inputs"].get("cuda_run_id")
            if phase == "publication" and isinstance(value["inputs"], dict)
            else None
        )
        _require(
            len(raw_prior_authority) == len(expected_prior_keys)
            and (
                phase != "publication"
                or (type(expected_prior_run) is int and expected_prior_run > 0)
            ),
            "recovery_prior_authority_identity_cardinality_rejected",
        )
        prior_authority = tuple(
            _validated_authority_evidence_identity(
                item,
                context=f"recovery_prior_authority_identity_{ordinal}",
                expected_phase="final-main",
                expected_head_sha=str(value["head_sha"]),
                expected_run_id=(
                    int(expected_prior_run) if type(expected_prior_run) is int else None
                ),
                expected_job_key=key,
            )
            for ordinal, (key, item) in enumerate(
                zip(expected_prior_keys, raw_prior_authority),
                start=1,
            )
        )
        session = PhaseSession(
            phase=phase,
            workflow=str(value["workflow"]),
            workflow_path=str(value["workflow_path"]),
            dispatch_ref=str(value["dispatch_ref"]),
            run_ref=str(value["run_ref"]),
            head_sha=str(value["head_sha"]),
            inputs=_json_copy(value["inputs"]),
            prior_accepted_cuda_runner_nonces=tuple(raw_prior_nonces),
            run={"id": value["run_id"], "run_attempt": 1},
            jobs=jobs,
            queued_jobs=queued_jobs,
            dispatch_receipt=dispatch_receipt,
            prior_authority_evidence_identities=prior_authority,
        )
        accepted_fields = set(AcceptedJobReceipt.__dataclass_fields__)
        _require(
            len(accepted_payloads) <= len(jobs),
            "recovery_accepted_cardinality_rejected",
        )
        for raw in accepted_payloads:
            _require(set(raw) == accepted_fields, "recovery_accepted_schema_rejected")
            try:
                receipt = AcceptedJobReceipt(**raw)
            except TypeError:
                raise ControllerError("recovery_accepted_schema_rejected") from None
            material = receipt.to_mapping()
            evidence_sha256 = material.pop("evidence_sha256")
            expected_key = jobs[len(session.accepted)].key
            binding = jobs[len(session.accepted)]
            _require(
                _sha(_canonical(material)) == evidence_sha256
                and receipt.phase == phase
                and receipt.run_id == value["run_id"]
                and receipt.job_key == expected_key
                and receipt.job_id == binding.job_id
                and receipt.runner_name == binding.runner_name,
                "recovery_accepted_binding_rejected",
            )
            session.accepted[expected_key] = receipt
        return session

    @classmethod
    def resume_for_abort(
        cls,
        controller: ReleaseGpuController,
        provider: ProviderLifecycle,
        plan: live.ImmutablePlan,
        journal: EvidenceJournal,
        *,
        sleep: Callable[[float], None] = time.sleep,
        observation_poll_limit: int = 24,
    ) -> LiveReleaseDriver:
        """Reopen a crash-interrupted lifecycle only for exact cleanup, never replay."""

        _require(provider.plan_sha256 == plan.sha256, "recovery_provider_plan_mismatch")
        _require(journal._plan_sha256 == plan.sha256, "recovery_journal_plan_mismatch")
        _require(
            callable(sleep) and type(observation_poll_limit) is int and observation_poll_limit > 0,
            "recovery_driver_options_rejected",
        )
        entries = journal.verified_entries()
        _require(
            len(entries) >= 2
            and entries[0][0] == "evidence-directory"
            and entries[1] == ("immutable-plan", plan.to_mapping()),
            "recovery_immutable_plan_drift",
        )
        EvidenceJournal._validate_directory_evidence_mapping(
            entries[0][1], journal._evidence_directory
        )
        _require(
            entries[-1][0] != "lifecycle-restored",
            "recovery_lifecycle_already_restored",
        )
        prior_provider_sink_bindings = [
            payload for label, payload in entries if label == "provider-intent-sink-bound"
        ]
        for payload in prior_provider_sink_bindings:
            _require(
                set(payload)
                == {
                    "plan_sha256",
                    "binding_sha256",
                    "sink",
                    "bound_before_observation",
                    "recovery_process",
                }
                and payload["plan_sha256"] == plan.sha256
                and type(payload["binding_sha256"]) is str
                and SHA256_RE.fullmatch(payload["binding_sha256"]) is not None
                and payload["sink"] == "evidence-journal"
                and payload["bound_before_observation"] is True
                and type(payload["recovery_process"]) is bool,
                "recovery_provider_sink_binding_rejected",
            )
        provider_intent_binding_sha256 = cls._bind_provider_intent_sink(provider, journal)

        instance = object.__new__(cls)
        instance._controller = controller
        instance._provider = provider
        instance._plan = plan
        instance._journal = journal
        instance._access_identity_file = ""
        instance._access_identity = None
        instance._access_identity_evidence = None
        instance._known_hosts_path = ""
        instance._sleep = sleep
        instance._observation_poll_limit = observation_poll_limit
        instance._state = "crash-recovery"
        instance._readiness = None
        instance._bindings = {}
        instance._phase = None
        instance._session = None
        instance._active_job = None
        instance._dispatch_ambiguity = None
        instance._github_ambiguity = None
        instance._remote_ambiguity_receipt = None
        instance._crash_jit_intent = None
        instance._runner_delete_intent = None
        instance._provider_crash_ambiguity = None
        instance._provider_intent_binding_sha256 = provider_intent_binding_sha256
        instance._cleanup_evidence_errors = None

        known_entries = [payload for label, payload in entries if label == "known-hosts"]
        _require(len(known_entries) <= 1, "recovery_known_hosts_cardinality")
        instance._known_hosts = None
        if known_entries:
            known = known_entries[0]
            known_path = Path(str(known.get("absolute_path")))
            _require(
                set(known)
                == {
                    "absolute_path",
                    "content_sha256",
                    "evidence_directory_acl_receipt_sha256",
                    "public_ipv4",
                    "host_fingerprint",
                    "content_is_public",
                }
                and known_path.is_absolute()
                and known_path == known_path.resolve(strict=True)
                and known_path.parent == journal.directory
                and known["evidence_directory_acl_receipt_sha256"] == journal.acl_receipt_sha256
                and known["host_fingerprint"] == plan.host_key_fingerprint
                and known["content_is_public"] is True
                and known_path.is_file()
                and not known_path.is_symlink()
                and _sha(known_path.read_bytes()) == known["content_sha256"],
                "recovery_known_hosts_binding_rejected",
            )
            instance._known_hosts_path = str(known_path)
            instance._known_hosts = live.KnownHostsFileReceipt(
                absolute_path=str(known["absolute_path"]),
                content_sha256=str(known["content_sha256"]),
                evidence_directory_acl_receipt_sha256=str(
                    known["evidence_directory_acl_receipt_sha256"]
                ),
                public_ipv4=str(known["public_ipv4"]),
                host_fingerprint=str(known["host_fingerprint"]),
            )

        dispatch_entries = [payload for label, payload in entries if label == "github-dispatch"]
        _require(len(dispatch_entries) <= 1, "recovery_dispatch_cardinality")
        if dispatch_entries:
            accepted_payloads = [
                payload for label, payload in entries if label == "accepted-actions-job"
            ]
            session = cls._session_from_journal(dispatch_entries[0], accepted_payloads)
            _require(session.head_sha == plan.head_sha, "recovery_session_head_drift")
            instance._session = session
            instance._phase = session.phase
        else:
            intents = [payload for label, payload in entries if label == "github-dispatch-intent"]
            _require(len(intents) <= 1, "recovery_dispatch_intent_cardinality")
            if intents:
                intent = intents[0]
                phase = intent.get("phase")
                _require(
                    type(phase) is str
                    and phase in PHASES
                    and intent.get("head_sha") == plan.head_sha
                    and type(intent.get("dispatch_path")) is str
                    and type(intent.get("request_sha256")) is str,
                    "recovery_dispatch_intent_rejected",
                )
                instance._phase = phase
                instance._dispatch_ambiguity = AmbiguousGitHubMutation(
                    "POST",
                    str(intent["dispatch_path"]),
                    str(intent["request_sha256"]),
                    "process_crash_after_dispatch_intent",
                    reconciliation={
                        "workflow": intent.get("workflow"),
                        "head_sha": plan.head_sha,
                        "expected_runner_nonces": intent.get("expected_runner_nonces"),
                        "pre_dispatch_run_ids": intent.get("pre_dispatch_run_ids"),
                        "dispatch_retried": False,
                    },
                )

        cancel_intents = [payload for label, payload in entries if label == "github-cancel-intent"]
        _require(len(cancel_intents) <= 1, "recovery_cancel_intent_cardinality")
        if instance._session is None and cancel_intents:
            _require(
                instance._dispatch_ambiguity is not None,
                "recovery_orphan_cancel_intent",
            )
            cancel_intent = cancel_intents[0]
            run_id = cancel_intent.get("run_id")
            cancel_path = f"/repos/jemsbhai/explainiverse/actions/runs/{run_id}/cancel"
            cancel_request_sha256 = _sha(
                _canonical({"method": "POST", "path": cancel_path, "body": None})
            )
            _require(
                type(run_id) is int
                and run_id > 0
                and cancel_intent.get("phase") == instance._phase
                and cancel_intent.get("run_attempt") == 1
                and cancel_intent.get("head_sha") == plan.head_sha
                and cancel_intent.get("cancel_path") == cancel_path
                and cancel_intent.get("request_sha256") == cancel_request_sha256
                and cancel_intent.get("reason")
                in {
                    "dispatch-job-materialization-timeout",
                    "ambiguous-dispatch-exact-run",
                }
                and cancel_intent.get("accepted_job_ids") == []
                and cancel_intent.get("serviced_job_ids") == []
                and cancel_intent.get("mutation_retried") is False,
                "recovery_dispatch_cancel_intent_rejected",
            )
            original = instance._dispatch_ambiguity
            assert original is not None
            reconciliation = dict(original.reconciliation)
            reconciliation.update(
                {
                    "run_id": run_id,
                    "cancel_already_attempted": True,
                    "cancel_request_sha256": cancel_request_sha256,
                    "cancel_retried": False,
                }
            )
            instance._dispatch_ambiguity = AmbiguousGitHubMutation(
                original.method,
                original.path,
                original.request_sha256,
                original.reason_code,
                reconciliation=reconciliation,
            )

        if instance._session is not None:
            jit_intents = [payload for label, payload in entries if label == "github-jit-intent"]
            created_entries = [
                payload for label, payload in entries if label == "github-jit-created"
            ]
            accepted_ids = {item.job_id for item in instance._session.accepted.values()}
            pending = [item for item in jit_intents if item.get("job_id") not in accepted_ids]
            _require(len(pending) <= 1, "recovery_pending_jit_intent_cardinality")
            if pending:
                intent = dict(pending[0])
                matching_created = [
                    item for item in created_entries if item.get("job_id") == intent.get("job_id")
                ]
                _require(
                    len(matching_created) <= 1,
                    "recovery_pending_jit_created_cardinality",
                )
                if matching_created:
                    jit_receipt = matching_created[0].get("jit_receipt")
                    _require(
                        type(jit_receipt) is dict
                        and type(jit_receipt.get("runner")) is dict
                        and type(jit_receipt["runner"].get("id")) is int,
                        "recovery_jit_created_binding_rejected",
                    )
                    assert isinstance(jit_receipt, dict)
                    runner_mapping = jit_receipt["runner"]
                    assert isinstance(runner_mapping, dict)
                    intent["runner_id"] = runner_mapping["id"]
                bound = [
                    item for item in instance._session.jobs if item.job_id == intent.get("job_id")
                ]
                _require(len(bound) == 1, "recovery_pending_job_not_in_session")
                instance._active_job = bound[0]
                instance._crash_jit_intent = intent

            runner_delete_intents = [
                payload for label, payload in entries if label == "github-runner-delete-intent"
            ]
            runner_delete_settled = {
                payload.get("delete_request_sha256")
                for label, payload in entries
                if label == "github-runner-delete-settled"
            }
            pending_runner_deletes = [
                payload
                for payload in runner_delete_intents
                if payload.get("request_sha256") not in runner_delete_settled
                and payload.get("job_id") not in accepted_ids
            ]
            _require(
                len(pending_runner_deletes) <= 1,
                "recovery_runner_delete_intent_cardinality",
            )
            if pending_runner_deletes:
                delete_intent = pending_runner_deletes[0]
                raw_runner_id = delete_intent.get("runner_id")
                expected_path = f"/repos/jemsbhai/explainiverse/actions/runners/" f"{raw_runner_id}"
                expected_request = _sha(
                    _canonical({"method": "DELETE", "path": expected_path, "body": None})
                )
                bound = [
                    item
                    for item in instance._session.jobs
                    if item.job_id == delete_intent.get("job_id")
                ]
                _require(
                    set(delete_intent)
                    == {
                        "runner_id",
                        "phase",
                        "run_id",
                        "head_sha",
                        "job_id",
                        "runner_name",
                        "path",
                        "request_sha256",
                        "mutation_retried",
                    }
                    and type(raw_runner_id) is int
                    and raw_runner_id > 0
                    and len(bound) == 1
                    and delete_intent["phase"] == instance._session.phase
                    and delete_intent["run_id"] == instance._session.run["id"]
                    and delete_intent["head_sha"] == instance._session.head_sha
                    and delete_intent["runner_name"] == bound[0].runner_name
                    and delete_intent["path"] == expected_path
                    and delete_intent["request_sha256"] == expected_request
                    and delete_intent["mutation_retried"] is False,
                    "recovery_runner_delete_intent_rejected",
                )
                instance._active_job = bound[0]
                instance._runner_delete_intent = dict(delete_intent)

            phase_was_settled = any(label == "phase-settlement" for label, _ in entries)
            if cancel_intents and not phase_was_settled:
                cancel_intent = cancel_intents[0]
                cancel_path = (
                    f"/repos/jemsbhai/explainiverse/actions/runs/"
                    f"{instance._session.run['id']}/cancel"
                )
                expected_request_sha256 = _sha(
                    _canonical({"method": "POST", "path": cancel_path, "body": None})
                )
                _require(
                    set(cancel_intent)
                    == {
                        "phase",
                        "run_id",
                        "run_attempt",
                        "head_sha",
                        "cancel_path",
                        "request_sha256",
                        "reason",
                        "accepted_job_ids",
                        "serviced_job_ids",
                        "mutation_retried",
                    }
                    and cancel_intent["phase"] == instance._session.phase
                    and cancel_intent["run_id"] == instance._session.run["id"]
                    and cancel_intent["run_attempt"] == 1
                    and cancel_intent["head_sha"] == instance._session.head_sha
                    and cancel_intent["cancel_path"] == cancel_path
                    and cancel_intent["request_sha256"] == expected_request_sha256
                    and cancel_intent["mutation_retried"] is False
                    and type(cancel_intent["accepted_job_ids"]) is list
                    and type(cancel_intent["serviced_job_ids"]) is list,
                    "recovery_cancel_intent_rejected",
                )
                instance._github_ambiguity = AmbiguousGitHubMutation(
                    "POST",
                    cancel_path,
                    expected_request_sha256,
                    "controller_process_crash_after_cancel_intent",
                    reconciliation={
                        "run_id": instance._session.run["id"],
                        "reason": cancel_intent["reason"],
                        "accepted_job_ids": cancel_intent["accepted_job_ids"],
                        "serviced_job_ids": cancel_intent["serviced_job_ids"],
                        "response_received": None,
                        "response_sha256": None,
                        "cancel_retried": False,
                    },
                )

        provider_intents = [
            payload for label, payload in entries if label == "provider-mutation-intent"
        ]
        emergency_provider_intents = journal.emergency_provider_intents()
        for emergency_intent in emergency_provider_intents:
            if emergency_intent not in provider_intents:
                provider_intents.append(emergency_intent)
        completed_provider_requests = {
            str(payload.get("request_sha256"))
            for label, payload in entries
            if (
                label
                in {
                    "provider-restrict-global",
                    "provider-create-ruleset",
                    "provider-launch",
                    "provider-terminate",
                    "provider-delete-ruleset",
                    "provider-restore-global",
                }
                or label.endswith("-ambiguous")
            )
            and type(payload.get("request_sha256")) is str
        }
        unresolved_provider = [
            item
            for item in provider_intents
            if item.get("request_sha256") not in completed_provider_requests
        ]
        mutation_order = tuple(live.MUTATION_PATHS)
        validated_unresolved: list[live.MutationIntent] = []
        for provider_intent_mapping in unresolved_provider:
            provider_intent = live.MutationIntent.from_public_mapping(provider_intent_mapping)
            _require(
                provider_intent.plan_sha256 == plan.sha256
                and (
                    any(
                        payload["binding_sha256"] == provider_intent.callback_binding_sha256
                        for payload in prior_provider_sink_bindings
                    )
                    or provider_intent_mapping in emergency_provider_intents
                ),
                "recovery_provider_intent_rejected",
            )
            validated_unresolved.append(provider_intent)
        _require(
            len({item.request_sha256 for item in validated_unresolved}) == len(validated_unresolved)
            and [mutation_order.index(item.operation) for item in validated_unresolved]
            == sorted(mutation_order.index(item.operation) for item in validated_unresolved),
            "recovery_provider_intent_sequence_rejected",
        )
        if unresolved_provider:
            provider_intent_mapping = unresolved_provider[-1]
            instance._provider_crash_ambiguity = provider.ambiguity_from_persisted_intent(
                provider_intent_mapping
            )

        if any(label == "phase-settlement" for label, _ in entries):
            instance._state = "phase-complete"
        journal.record(
            "provider-intent-sink-bound",
            {
                "plan_sha256": plan.sha256,
                "binding_sha256": provider_intent_binding_sha256,
                "sink": "evidence-journal",
                "bound_before_observation": True,
                "recovery_process": True,
            },
        )
        journal.record(
            "crash-recovery-resume",
            {
                "plan_sha256": plan.sha256,
                "previous_evidence_sha256": journal.last_evidence_sha256,
                "phase": instance._phase,
                "session_recovered": instance._session is not None,
                "dispatch_ambiguity_recovered": instance._dispatch_ambiguity is not None,
                "pending_jit_intent_recovered": instance._crash_jit_intent is not None,
                "mutation_replayed": False,
            },
        )
        return instance

    def _record_evidence(self, label: str, payload: Mapping[str, Any]) -> str:
        try:
            return self._journal.record(label, payload)
        except BaseException as exc:
            errors = self._cleanup_evidence_errors
            if errors is None:
                raise
            errors.append(exc)
            return ""

    def _record_snapshot(self, receipt: live.SnapshotReceipt) -> None:
        self._record_evidence(
            f"provider-{receipt.phase.replace('_', '-')}",
            receipt.to_public_mapping(),
        )

    def _record_progress(self, label: str, payload: Mapping[str, Any]) -> None:
        # Progress includes GitHub mutation write-ahead intents.  Never swallow
        # this failure: the controller must stop before making the request.
        self._journal.record(label, payload)
        if label == "github-dispatch-intent":
            self._dispatch_ambiguity = AmbiguousGitHubMutation(
                "POST",
                str(payload.get("dispatch_path")),
                str(payload.get("request_sha256")),
                "controller_process_exit_after_dispatch_intent",
                reconciliation={
                    "workflow": payload.get("workflow"),
                    "head_sha": payload.get("head_sha"),
                    "expected_runner_nonces": payload.get("expected_runner_nonces"),
                    "pre_dispatch_run_ids": payload.get("pre_dispatch_run_ids"),
                    "dispatch_retried": False,
                },
            )
        elif label == "github-cancel-intent":
            cancellation = AmbiguousGitHubMutation(
                "POST",
                str(payload.get("cancel_path")),
                str(payload.get("request_sha256")),
                "controller_process_exit_after_cancel_intent",
                reconciliation={
                    "run_id": payload.get("run_id"),
                    "reason": payload.get("reason"),
                    "accepted_job_ids": payload.get("accepted_job_ids"),
                    "serviced_job_ids": payload.get("serviced_job_ids"),
                    "response_received": None,
                    "response_sha256": None,
                    "cancel_retried": False,
                },
            )
            if self._session is None and self._dispatch_ambiguity is not None:
                reconciliation = dict(self._dispatch_ambiguity.reconciliation)
                reconciliation.update(
                    {
                        "run_id": payload.get("run_id"),
                        "cancel_already_attempted": True,
                        "cancel_request_sha256": payload.get("request_sha256"),
                        "cancel_response_sha256": None,
                        "cancel_retried": False,
                    }
                )
                original = self._dispatch_ambiguity
                self._dispatch_ambiguity = AmbiguousGitHubMutation(
                    original.method,
                    original.path,
                    original.request_sha256,
                    original.reason_code,
                    reconciliation=reconciliation,
                )
            else:
                self._github_ambiguity = cancellation
        elif label == "github-cancel-settled":
            # Keep the already-attempted marker until the enclosing phase
            # settlement is durable. If that later write fails, abort performs
            # read-only reconciliation rather than a second POST.
            pass
        elif label == "github-jit-intent":
            self._crash_jit_intent = _json_copy(payload)
        elif label == "github-runner-delete-intent":
            self._runner_delete_intent = _json_copy(payload)
        elif label == "github-runner-delete-settled":
            self._runner_delete_intent = None
        elif label == "accepted-actions-job":
            self._crash_jit_intent = None

    @staticmethod
    def _recovery_mapping(receipt: live.RecoveryReceipt) -> dict[str, Any]:
        return {
            "plan_sha256": receipt.plan_sha256,
            "operation": receipt.operation,
            "ambiguous_request_sha256": receipt.ambiguous_request_sha256,
            "inventory_snapshot_sha256": receipt.inventory_snapshot_sha256,
            "outcome": receipt.outcome,
            "ruleset_id": receipt.ruleset_id,
            "instance_id": receipt.instance_id,
        }

    def _observe_until(self, phase: str) -> live.SnapshotReceipt:
        last_error: live.ContractError | None = None
        for attempt in range(self._observation_poll_limit):
            try:
                receipt = self._provider.observe(phase)
            except live.ContractError as exc:
                last_error = exc
                if attempt + 1 < self._observation_poll_limit:
                    self._sleep(5)
                continue
            self._record_snapshot(receipt)
            return receipt
        raise ControllerError(f"provider_{phase}_not_observed") from last_error

    def _transition(
        self,
        operation: str,
        prestate: live.SnapshotReceipt,
        mutation: Callable[[], live.MutationReceipt],
    ) -> live.SnapshotReceipt:
        _require(
            prestate.phase == live.MUTATION_PRESTATE[operation],
            "driver_provider_prestate_mismatch",
        )
        operation_label = operation.replace("_", "-")
        self._record_evidence(
            f"provider-{operation_label}-intent",
            {
                "plan_sha256": self._plan.sha256,
                "operation": operation,
                "prestate": prestate.to_public_mapping(),
                "mutation_retried": False,
            },
        )
        try:
            receipt = mutation()
        except live.AmbiguousMutation as ambiguity:
            self._record_evidence(
                f"provider-{operation_label}-ambiguous",
                {
                    "operation": ambiguity.operation,
                    "request_sha256": ambiguity.request_sha256,
                    "reason_code": ambiguity.reason_code,
                    "mutation_retried": False,
                },
            )
            recovery_snapshot = self._provider.observe("recovery")
            self._record_snapshot(recovery_snapshot)
            recovery = self._provider.recover_ambiguous(ambiguity, recovery_snapshot)
            self._record_evidence(
                f"provider-{operation_label}-recovery", self._recovery_mapping(recovery)
            )
            _require(
                recovery.outcome in {"applied_exactly_once", "applied_in_progress"},
                "provider_ambiguous_mutation_not_applied",
            )
        else:
            self._record_evidence(f"provider-{operation_label}", receipt.to_public_mapping())
        return self._observe_until(self._NEXT_PHASE[operation])

    def _observe_bound_instance(self) -> live.SnapshotReceipt:
        receipt = self._provider.observe("instance_bound")
        self._record_snapshot(receipt)
        return receipt

    def provision(self) -> HostReadinessReceipt:
        _require(self._state == "planned", "driver_provision_state_rejected")
        try:
            self._journal.record("immutable-plan", self._plan.to_mapping())
            self._journal.record(
                "provider-intent-sink-bound",
                {
                    "plan_sha256": self._plan.sha256,
                    "binding_sha256": self._provider_intent_binding_sha256,
                    "sink": "evidence-journal",
                    "bound_before_observation": True,
                    "recovery_process": False,
                },
            )
            self._journal.record("ssh-executable", self._controller.ssh_executable_receipt())
            self._journal.record("github-executable", self._controller.github_executable_receipt())
            _require(
                self._access_identity_evidence is not None,
                "driver_access_identity_evidence_missing",
            )
            assert self._access_identity_evidence is not None
            self._journal.record("ssh-access-identity", self._access_identity_evidence)
            baseline = self._provider.observe("baseline")
            self._record_snapshot(baseline)
            restricted = self._transition(
                "restrict_global", baseline, lambda: self._provider.restrict_global(baseline)
            )
            ruleset = self._transition(
                "create_ruleset", restricted, lambda: self._provider.create_ruleset(restricted)
            )
            instance = self._transition(
                "launch",
                ruleset,
                lambda: self._provider.launch(ruleset, self._identity, self._runtime_bundle),
            )
            _require(instance.instance_public_ipv4 is not None, "driver_instance_ip_missing")
            instance_public_ipv4 = instance.instance_public_ipv4
            assert instance_public_ipv4 is not None
            known_hosts = live.write_public_known_hosts(
                self._known_hosts_path,
                identity=self._identity,
                public_ipv4=instance_public_ipv4,
                evidence_directory_receipt=self._journal.evidence_directory_receipt,
            )
            self._journal.record("known-hosts", known_hosts.to_public_mapping())
            bindings = {
                mode: live.build_strict_ssh_binding(
                    identity=self._identity,
                    public_ipv4=instance_public_ipv4,
                    access_identity_file=self._access_identity_file,
                    known_hosts_file=known_hosts,
                    remote_mode=mode,
                )
                for mode in ("cloud-init", "preflight", "run")
            }
            for mode, binding in bindings.items():
                self._journal.record(f"ssh-{mode}", binding.to_public_mapping())
            readiness = self._controller.establish_host_readiness(
                bindings["cloud-init"],
                bindings["preflight"],
                provider_plan=self._plan,
                known_hosts=known_hosts,
                observe_provider_instance=self._observe_bound_instance,
            )
            self._journal.record(
                "host-readiness",
                self._readiness_mapping(readiness),
            )
        except BaseException:
            if not self._identity.destroyed:
                self._identity.destroy()
            if self._access_identity is not None and not self._access_identity.closed:
                self._controller.close_access_identity()
            self._state = "provision-failed"
            raise
        self._known_hosts = known_hosts
        self._bindings = bindings
        self._readiness = readiness
        self._state = "ready"
        return readiness

    @staticmethod
    def _readiness_mapping(readiness: HostReadinessReceipt) -> dict[str, Any]:
        readiness.validate()
        return {
            "cloud_init": readiness.cloud_init.to_public_mapping(),
            "host_preflight": readiness.preflight.to_public_mapping(),
            "cloud_init_sha256": readiness.cloud_init_sha256,
            "preflight_sha256": readiness.preflight_sha256,
            "cloud_binding": _json_copy(readiness.cloud_binding),
            "preflight_binding": _json_copy(readiness.preflight_binding),
            "ssh_attempts": _json_copy(readiness.ssh_attempts),
            "readiness_evidence_sha256": readiness.evidence_sha256,
        }

    @staticmethod
    def _session_mapping(session: PhaseSession) -> dict[str, Any]:
        return {
            "phase": session.phase,
            "workflow": session.workflow,
            "workflow_path": session.workflow_path,
            "dispatch_ref": session.dispatch_ref,
            "run_ref": session.run_ref,
            "head_sha": session.head_sha,
            "inputs": dict(session.inputs),
            "prior_accepted_cuda_runner_nonces": list(session.prior_accepted_cuda_runner_nonces),
            "prior_authority_evidence_identities": [
                _json_copy(item) for item in session.prior_authority_evidence_identities
            ],
            "run_id": session.run["id"],
            "run_attempt": session.run["run_attempt"],
            "jobs": [dict(job.__dict__) for job in session.jobs],
            "queued_jobs": [dict(job.__dict__) for job in session.queued_jobs],
            "dispatch_receipt": dict(session.dispatch_receipt.__dict__),
        }

    def run_phase(
        self,
        phase: str,
        *,
        supplied_ref: str,
        app_capture_supplier: InstalledAppCaptureSupplier,
        prior_accepted_cuda_runner_nonces: tuple[str, ...] = (),
        preflight_run_id: int | None = None,
        cuda_run_id: int | None = None,
        final_main_acceptance: FinalMainAcceptance | None = None,
        dispatch_poll_limit: int | None = None,
    ) -> PhaseCompletion:
        _require(self._state == "ready", "driver_phase_state_rejected")
        _require(self._phase is None, "driver_second_phase_rejected")
        _require(callable(app_capture_supplier), "driver_app_capture_supplier_not_callable")
        _require(self._readiness is not None, "driver_readiness_missing")
        _require(self._known_hosts is not None, "driver_known_hosts_missing")
        _require(
            set(self._bindings) == {"cloud-init", "preflight", "run"}, "driver_bindings_missing"
        )
        self._phase = phase
        try:
            session = self._controller.dispatch_phase(
                phase,
                head_sha=self._plan.head_sha,
                supplied_ref=supplied_ref,
                prior_accepted_cuda_runner_nonces=prior_accepted_cuda_runner_nonces,
                preflight_run_id=preflight_run_id,
                cuda_run_id=cuda_run_id,
                final_main_acceptance=final_main_acceptance,
                poll_limit=dispatch_poll_limit,
                progress=self._record_progress,
            )
        except AmbiguousGitHubMutation as exc:
            self._journal.record("ambiguous-github-mutation", exc.to_public_mapping())
            self._dispatch_ambiguity = exc
            self._state = "phase-dispatch-ambiguous"
            raise
        except BaseException as exc:
            self._journal.record(
                "phase-failure",
                {
                    "exception_type": type(exc).__name__,
                    "stable_code": (
                        str(exc)
                        if isinstance(exc, (ControllerError, live.ContractError))
                        else "unclassified-local-failure"
                    ),
                    "cleanup_required": True,
                },
            )
            self._state = "phase-failed"
            raise
        _require(session.head_sha == self._plan.head_sha, "driver_dispatch_head_mismatch")
        self._session = session
        self._dispatch_ambiguity = None
        self._state = "phase-active"
        self._journal.record("github-dispatch", self._session_mapping(session))
        readiness = self._readiness
        known_hosts = self._known_hosts
        assert readiness is not None and known_hosts is not None
        try:
            for index, job in enumerate(session.jobs):
                self._active_job = job
                if index:
                    readiness = self._controller.refresh_host_preflight(
                        readiness,
                        self._bindings["preflight"],
                        provider_plan=self._plan,
                        known_hosts=known_hosts,
                        observe_provider_instance=self._observe_bound_instance,
                    )
                    self._journal.record(
                        "host-preflight-refresh",
                        self._readiness_mapping(readiness),
                    )
                capture_mapping, evidence_reader = app_capture_supplier()
                _require(callable(evidence_reader), "driver_app_evidence_reader_not_callable")
                app_capture = TrustedAppCapture.from_mapping(
                    capture_mapping,
                    resources=self._controller.sealed_resources,
                    evidence_reader=evidence_reader,
                )
                archive_receipt = self._journal.archive_installed_app_capture(
                    app_capture, evidence_reader
                )
                self._journal.record("installed-app-raw-archive", archive_receipt)
                self._journal.record("installed-app-authority", app_capture.to_mapping())
                runtime_plan, execution = self._controller.execute_job(
                    session,
                    job.key,
                    app_capture=app_capture,
                    installed_app_evidence_reader=evidence_reader,
                    readiness=readiness,
                    run_binding=self._bindings["run"],
                    control_plane_plan_sha256=self._plan.sha256,
                    progress=self._record_progress,
                )
                self._controller.settle_job(
                    session,
                    runtime_plan,
                    execution,
                    progress=self._record_progress,
                )
                self._active_job = None

            sealed_final: FinalMainAcceptance | None = None
            if phase == "pull-request":
                settlement = self._controller.cancel_pull_request_after_singles(
                    session, progress=self._record_progress
                )
            elif phase == "final-main":
                settlement = self._controller.settle_final_main(session)
                sealed_final = self._controller.seal_final_main_acceptance(session, settlement)
                self._journal.record("final-main-acceptance", sealed_final.to_mapping())
            elif phase == "publication":
                settlement = self._controller.publication_gpu_gate(session)
            else:
                raise ControllerError("driver_phase_rejected")
            final_digest = self._journal.record("phase-settlement", settlement)
        except AmbiguousRemoteExecution as exc:
            self._journal.record("ambiguous-remote-start", exc.receipt)
            self._remote_ambiguity_receipt = dict(exc.receipt)
            self._state = "remote-start-ambiguous"
            raise
        except AmbiguousGitHubMutation as exc:
            self._journal.record("ambiguous-github-mutation", exc.to_public_mapping())
            self._github_ambiguity = exc
            self._state = "github-mutation-ambiguous"
            raise
        except BaseException as exc:
            self._journal.record(
                "phase-failure",
                {
                    "exception_type": type(exc).__name__,
                    "stable_code": (
                        str(exc)
                        if isinstance(exc, (ControllerError, live.ContractError))
                        else "unclassified-local-failure"
                    ),
                    "cleanup_required": True,
                },
            )
            self._state = "phase-failed"
            raise
        self._state = "phase-complete"
        return PhaseCompletion(
            phase=phase,
            head_sha=session.head_sha,
            run_id=int(session.run["id"]),
            settlement=_json_copy(settlement),
            final_evidence_sha256=final_digest,
            final_main_acceptance=sealed_final,
        )

    def _locate_abort_state(self) -> live.SnapshotReceipt:
        for attempt in range(self._observation_poll_limit):
            for phase in (
                "instance_bound",
                "ruleset_ready",
                "instance_absent",
                "global_restricted",
                "ruleset_absent",
                "baseline",
                "restored",
            ):
                try:
                    receipt = self._provider.observe(phase)
                except live.ContractError:
                    continue
                self._record_snapshot(receipt)
                if phase == "ruleset_ready":
                    normalized = self._provider.observe("instance_absent")
                    self._record_snapshot(normalized)
                    return normalized
                if phase == "global_restricted":
                    normalized = self._provider.observe("ruleset_absent")
                    self._record_snapshot(normalized)
                    return normalized
                if phase == "baseline":
                    normalized = self._provider.observe("restored")
                    self._record_snapshot(normalized)
                    return normalized
                return receipt
            if attempt + 1 < self._observation_poll_limit:
                self._sleep(5)
        raise ControllerError("provider_abort_state_unresolved")

    def abort(self) -> str:
        """Reconcile GitHub, then unwind only an exact owned provider state."""

        _require(self._state != "restored", "driver_already_restored")
        self._cleanup_evidence_errors = []
        blocker: BaseException | None = None
        github_cleanup_safe = True
        zero_runners: Mapping[str, Any] | None = None
        if hasattr(self, "_identity") and not self._identity.destroyed:
            try:
                self._identity.destroy()
            except BaseException as exc:
                blocker = exc
            else:
                self._record_evidence(
                    "host-identity-destroyed",
                    {
                        "host_fingerprint": self._plan.host_key_fingerprint,
                        "private_key_destroyed": True,
                        "private_key_persisted": False,
                    },
                )
        if self._access_identity is not None and not self._access_identity.closed:
            try:
                self._controller.close_access_identity()
            except BaseException as exc:
                blocker = blocker or exc
            else:
                self._record_evidence(
                    "ssh-access-identity-closed",
                    {
                        "public_key_sha256": self._plan.ssh_public_key_sha256,
                        "closed": True,
                        "private_path_archived": False,
                        "private_digest_archived": False,
                    },
                )

        if self._provider_crash_ambiguity is not None:
            try:
                recovery_snapshot = self._provider.observe("recovery")
                self._record_snapshot(recovery_snapshot)
                recovery = self._provider.recover_ambiguous(
                    self._provider_crash_ambiguity, recovery_snapshot
                )
                self._record_evidence("provider-crash-recovery", self._recovery_mapping(recovery))
                _require(
                    recovery.outcome
                    in {"not_applied", "applied_exactly_once", "applied_in_progress"},
                    "provider_crash_recovery_outcome_rejected",
                )
                self._provider_crash_ambiguity = None
            except BaseException as exc:
                blocker = exc
                github_cleanup_safe = False
                self._record_evidence(
                    "provider-crash-recovery-blocker",
                    {
                        "exception_type": type(exc).__name__,
                        "stable_code": (
                            str(exc)
                            if isinstance(exc, live.ContractError)
                            else "unclassified-provider-recovery-failure"
                        ),
                    },
                )

        if self._dispatch_ambiguity is not None:
            _require(self._phase is not None, "driver_ambiguous_dispatch_phase_missing")
            assert self._phase is not None
            try:
                reconciliation = self._controller.reconcile_ambiguous_dispatch_for_abort(
                    self._phase,
                    self._plan.head_sha,
                    self._dispatch_ambiguity,
                    progress=self._record_progress,
                )
                self._record_evidence("abort-dispatch-reconciliation", reconciliation)
                self._dispatch_ambiguity = None
            except BaseException as exc:
                blocker = exc
                github_cleanup_safe = False
                self._record_evidence(
                    "abort-github-blocker",
                    {
                        "exception_type": type(exc).__name__,
                        "stable_code": (
                            str(exc)
                            if isinstance(exc, ControllerError)
                            else "unclassified-github-reconciliation-failure"
                        ),
                    },
                )

        state = self._locate_abort_state()
        if state.phase == "instance_bound":
            state = self._transition("terminate", state, lambda: self._provider.terminate(state))

        if (
            self._session is not None
            and self._active_job is not None
            and self._runner_delete_intent is not None
        ):
            try:
                reconciliation = self._controller.reconcile_runner_delete_for_abort(
                    self._session,
                    self._active_job,
                    self._runner_delete_intent,
                )
                self._record_evidence("abort-runner-delete-reconciliation", reconciliation)
                self._runner_delete_intent = None
            except BaseException as exc:
                blocker = blocker or exc
                github_cleanup_safe = False

        if (
            self._runner_delete_intent is None
            and self._session is not None
            and self._remote_ambiguity_receipt is not None
        ):
            try:
                reconciliation = self._controller.reconcile_runner_after_host_stop(
                    self._session,
                    self._remote_ambiguity_receipt,
                    progress=self._record_progress,
                )
                self._record_evidence("abort-runner-reconciliation", reconciliation)
                self._remote_ambiguity_receipt = None
            except BaseException as exc:
                blocker = blocker or exc
                github_cleanup_safe = False
        elif (
            self._runner_delete_intent is None
            and self._session is not None
            and self._active_job is not None
            and self._crash_jit_intent is not None
        ):
            try:
                reconciliation = self._controller.reconcile_crash_jit_after_host_stop(
                    self._session,
                    self._active_job,
                    self._crash_jit_intent,
                    progress=self._record_progress,
                )
                self._record_evidence("abort-crash-jit-reconciliation", reconciliation)
            except BaseException as exc:
                blocker = blocker or exc
                github_cleanup_safe = False
                self._record_evidence(
                    "abort-runner-blocker",
                    {
                        "exception_type": type(exc).__name__,
                        "stable_code": (
                            str(exc)
                            if isinstance(exc, ControllerError)
                            else "unclassified-runner-reconciliation-failure"
                        ),
                    },
                )
        elif (
            self._runner_delete_intent is None
            and self._session is not None
            and self._active_job is not None
            and self._github_ambiguity is not None
            and self._github_ambiguity.reason_code == "runner_delete_unresolved"
        ):
            runner_id = self._github_ambiguity.reconciliation.get("runner_id")
            try:
                reconciliation = self._controller.reconcile_runner_after_host_stop(
                    self._session,
                    {"job_id": self._active_job.job_id, "runner_id": runner_id},
                    progress=self._record_progress,
                )
                self._record_evidence("abort-runner-reconciliation", reconciliation)
            except BaseException as exc:
                blocker = blocker or exc
                github_cleanup_safe = False

        try:
            zero_runners = self._controller.prove_zero_runner_inventory()
            self._record_evidence("github-zero-runners-before-abort", zero_runners)
        except BaseException as exc:
            blocker = blocker or exc
            github_cleanup_safe = False

        if github_cleanup_safe and self._session is not None and self._state != "phase-complete":
            expected = tuple(self._session.accepted)
            complete = tuple(item.key for item in self._session.jobs) == expected
            try:
                cancel_path = (
                    f"/repos/jemsbhai/explainiverse/actions/runs/"
                    f"{self._session.run['id']}/cancel"
                )
                if (
                    self._github_ambiguity is not None
                    and self._github_ambiguity.method == "POST"
                    and self._github_ambiguity.path == cancel_path
                ):
                    settlement = self._controller.reconcile_cancel_for_abort(
                        self._session, self._github_ambiguity
                    )
                    self._record_evidence("abort-cancel-reconciliation", settlement)
                    self._github_ambiguity = None
                elif self._crash_jit_intent is not None:
                    serviced_job_id = self._crash_jit_intent.get("job_id")
                    _require(
                        type(serviced_job_id) is int and serviced_job_id > 0,
                        "driver_crash_serviced_job_id_rejected",
                    )
                    assert isinstance(serviced_job_id, int)
                    settlement = self._controller.cancel_crashed_phase(
                        self._session,
                        serviced_job_ids=(serviced_job_id,),
                        progress=self._record_progress,
                    )
                    self._record_evidence("abort-crash-phase-cancellation", settlement)
                elif complete and self._session.phase == "pull-request":
                    settlement = self._controller.cancel_pull_request_after_singles(
                        self._session, progress=self._record_progress
                    )
                    self._record_evidence("abort-pr-settlement", settlement)
                elif complete and self._session.phase == "final-main":
                    settlement = self._controller.settle_final_main(self._session)
                    sealed = self._controller.seal_final_main_acceptance(self._session, settlement)
                    existing_acceptance = [
                        payload
                        for label, payload in self._journal.verified_entries()
                        if label == "final-main-acceptance"
                    ]
                    _require(
                        len(existing_acceptance) <= 1,
                        "driver_final_acceptance_cardinality",
                    )
                    if existing_acceptance:
                        _require(
                            existing_acceptance[0] == sealed.to_mapping(),
                            "driver_final_acceptance_resume_drift",
                        )
                    else:
                        self._record_evidence("final-main-acceptance", sealed.to_mapping())
                    self._record_evidence("phase-settlement", settlement)
                elif complete and self._session.phase == "publication":
                    settlement = self._controller.publication_gpu_gate(self._session)
                    self._record_evidence("phase-settlement", settlement)
                elif not complete:
                    settlement = self._controller.cancel_failed_phase(
                        self._session, progress=self._record_progress
                    )
                    self._record_evidence("abort-phase-cancellation", settlement)
            except BaseException as exc:
                blocker = blocker or exc

        if state.phase == "instance_absent":
            state = self._transition(
                "delete_ruleset", state, lambda: self._provider.delete_ruleset(state)
            )
        if state.phase == "ruleset_absent":
            state = self._transition(
                "restore_global", state, lambda: self._provider.restore_global(state)
            )
        restored = state
        _require(restored.phase == "restored", "driver_restoration_not_observed")
        evidence_errors_before_final = self._cleanup_evidence_errors or []
        clean_restoration = (
            blocker is None
            and not evidence_errors_before_final
            and zero_runners is not None
            and self._known_hosts is not None
        )
        if clean_restoration:
            assert self._known_hosts is not None
            lifecycle_payload = {
                "plan_sha256": self._plan.sha256,
                "provider_instances": 0,
                "provider_firewall_rulesets": 0,
                "global_firewall_restored": True,
                "repository_runners": 0,
                "known_hosts_sha256": self._known_hosts.content_sha256,
            }
        else:
            lifecycle_payload = {
                "plan_sha256": self._plan.sha256,
                "provider_instances": 0,
                "provider_firewall_rulesets": 0,
                "global_firewall_restored": True,
                "repository_runners_zero_verified": zero_runners is not None,
                "github_reconciliation_blocked": blocker is not None,
                "evidence_archival_error_count": len(evidence_errors_before_final),
            }
        final_digest = self._record_evidence(
            "lifecycle-restored" if clean_restoration else "lifecycle-provider-restored",
            lifecycle_payload,
        )
        if self._access_identity is not None and not self._access_identity.closed:
            try:
                self._controller.close_access_identity()
            except BaseException as exc:
                blocker = blocker or exc
        try:
            self._journal.close()
        except BaseException as exc:
            blocker = blocker or exc
        evidence_errors = self._cleanup_evidence_errors or []
        if evidence_errors and blocker is None:
            blocker = ControllerError("abort_evidence_archival_incomplete")
        if blocker is not None:
            self._state = "abort-blocked"
            raise blocker
        self._state = "restored"
        return final_digest

    def teardown(self) -> str:
        """Normal completion and all recoverable failures use the same fresh abort path."""

        return self.abort()
