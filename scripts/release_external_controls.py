"""Capture and verify the mutable controls around a stable release.

The reviewed policy lives in ``.github/release-control-policy.json``.  An
administrator first captures the relevant GitHub API responses with a local
``gh`` token because a workflow ``GITHUB_TOKEN`` cannot read branch protection
or secret metadata.  The pre-tag workflow accepts only a fresh capture by an
approved principal, binds it to that workflow run, attests it, and fails closed
on any difference.  The
snapshot deliberately records (but cannot certify) the expected PyPI Trusted
Publisher: PyPI does not expose project publisher settings through a public
read API.  A successful token-free OIDC upload remains the publication-time
proof for that separate external control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

_SHA = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TAG = re.compile(r"v\d+\.\d+\.\d+")
_MAX_SNAPSHOT_AGE = timedelta(minutes=30)
_MAX_INSTALLED_APP_CAPTURE_AGE = timedelta(minutes=10)
_MAX_CLOCK_SKEW = timedelta(minutes=1)
_MAX_APP_EVIDENCE_BYTES = 10 * 1024 * 1024
_APP_EVIDENCE_MEDIA_TYPE = "text/plain; charset=utf-8"
_APP_EVIDENCE_KIND_ORDER = {
    "installation-list": 0,
    "installation-configure": 1,
    "permission-update": 2,
}


class ApiNotFoundError(RuntimeError):
    """The requested GitHub object does not exist."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        value = _mapping(value, ".".join(keys))[key]
    return value


def _canonical_names(values: Sequence[Any], name: str) -> list[str]:
    names: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} entries must be non-empty strings")
        names.append(value)
    if len(names) != len(set(names)):
        raise ValueError(f"{name} entries must be unique")
    return sorted(names)


def _aware_utc_timestamp(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an RFC 3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an RFC 3339 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _canonical_app_evidence(value: Any, name: str) -> Mapping[str, Any]:
    evidence = _mapping(value, name)
    expected_keys = {
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
    if set(evidence) != expected_keys:
        raise ValueError(
            f"{name} keys must be exactly {sorted(expected_keys)!r}; "
            f"got {sorted(evidence, key=str)!r}"
        )
    filename = evidence.get("filename")
    if (
        not isinstance(filename, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", filename) is None
        or filename in {".", ".."}
    ):
        raise ValueError(f"{name} filename must be a safe unique basename")
    kind = evidence.get("kind")
    if kind not in _APP_EVIDENCE_KIND_ORDER:
        raise ValueError(f"{name} kind is not recognized")
    installation_id = evidence.get("installation_id")
    if kind == "installation-list":
        if installation_id is not None:
            raise ValueError(f"{name} installation-list id must be null")
    elif (
        isinstance(installation_id, bool)
        or not isinstance(installation_id, int)
        or installation_id < 1
    ):
        raise ValueError(f"{name} installation id must be a positive integer")
    source_url = evidence.get("source_url")
    if not isinstance(source_url, str) or not source_url.startswith("https://github.com/"):
        raise ValueError(f"{name} source_url must be an HTTPS github.com URL")
    captured_at = _aware_utc_timestamp(evidence.get("captured_at"), f"{name} captured_at")
    if evidence.get("media_type") != _APP_EVIDENCE_MEDIA_TYPE:
        raise ValueError(f"{name} media_type must be {_APP_EVIDENCE_MEDIA_TYPE!r}")
    if evidence.get("full_page") is not True:
        raise ValueError(f"{name} full_page must be true")
    byte_count = evidence.get("bytes")
    if (
        isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or not 1 <= byte_count <= _MAX_APP_EVIDENCE_BYTES
    ):
        raise ValueError(f"{name} bytes must be a positive bounded integer")
    digest = evidence.get("sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{name} sha256 must be 64 lowercase hexadecimal characters")
    return {
        "filename": filename,
        "kind": kind,
        "installation_id": installation_id,
        "source_url": source_url,
        "captured_at": captured_at.isoformat(),
        "media_type": _APP_EVIDENCE_MEDIA_TYPE,
        "full_page": True,
        "bytes": byte_count,
        "sha256": digest,
    }


def _app_evidence_header(evidence: Mapping[str, Any]) -> str:
    installation_id = evidence["installation_id"]
    normalized_id = "null" if installation_id is None else str(installation_id)
    return (
        f"source_url={evidence['source_url']}\t"
        f"captured_at={evidence['captured_at']}\t"
        f"kind={evidence['kind']}\t"
        f"installation_id={normalized_id}\n"
    )


def _canonical_app_installation(value: Any, name: str) -> Mapping[str, Any]:
    installation = _mapping(value, name)
    expected_keys = {
        "id",
        "name",
        "repository_selection",
        "repository_access",
        "suspended",
        "danger_zone_action",
        "permission_update_requested",
        "permission_update_review",
        "permissions",
        "requested_additional_permissions",
    }
    if set(installation) != expected_keys:
        raise ValueError(
            f"{name} keys must be exactly {sorted(expected_keys)!r}; "
            f"got {sorted(installation, key=str)!r}"
        )
    installation_id = installation.get("id")
    if isinstance(installation_id, bool) or not isinstance(installation_id, int):
        raise ValueError(f"{name} id must be a positive integer")
    if installation_id < 1:
        raise ValueError(f"{name} id must be a positive integer")
    app_name = installation.get("name")
    if not isinstance(app_name, str) or not app_name:
        raise ValueError(f"{name} name must be a non-empty string")
    repository_selection = installation.get("repository_selection")
    if repository_selection not in {"all", "selected"}:
        raise ValueError(f"{name} repository_selection must be 'all' or 'selected'")
    for field in (
        "repository_access",
        "suspended",
        "permission_update_requested",
    ):
        if not isinstance(installation.get(field), bool):
            raise ValueError(f"{name} {field} must be boolean")
    if repository_selection == "all" and installation.get("repository_access") is not True:
        raise ValueError(f"{name} all-repository selection must include the release repository")
    expected_danger_action = "Unsuspend" if installation.get("suspended") else "Suspend"
    if installation.get("danger_zone_action") != expected_danger_action:
        raise ValueError(
            f"{name} danger_zone_action must be {expected_danger_action!r} for the "
            "recorded suspension state"
        )

    normalized_permissions: dict[str, Mapping[str, list[str]]] = {}
    for field in ("permissions", "requested_additional_permissions"):
        permission_map = _mapping(installation.get(field), f"{name} {field}")
        if set(permission_map) != {"read", "write"}:
            raise ValueError(f"{name} {field} keys must be exactly ['read', 'write']")
        normalized_permissions[field] = {
            access: _canonical_names(
                _sequence(permission_map.get(access), f"{name} {field} {access}"),
                f"{name} {field} {access}",
            )
            for access in ("read", "write")
        }

    requested_permissions = normalized_permissions["requested_additional_permissions"]
    has_displayed_delta = any(requested_permissions[access] for access in ("read", "write"))
    if installation.get("permission_update_requested"):
        expected_update_review = (
            "listed-additional-permissions"
            if has_displayed_delta
            else "pending-no-displayed-repository-permission-delta"
        )
    else:
        expected_update_review = "not-requested"
        if has_displayed_delta:
            raise ValueError(f"{name} cannot list requested permissions without a pending update")
    if installation.get("permission_update_review") != expected_update_review:
        raise ValueError(f"{name} permission_update_review must be {expected_update_review!r}")

    return {
        "id": installation_id,
        "name": app_name,
        "repository_selection": repository_selection,
        "repository_access": installation.get("repository_access"),
        "suspended": installation.get("suspended"),
        "danger_zone_action": expected_danger_action,
        "permission_update_requested": installation.get("permission_update_requested"),
        "permission_update_review": expected_update_review,
        "permissions": normalized_permissions["permissions"],
        "requested_additional_permissions": normalized_permissions[
            "requested_additional_permissions"
        ],
    }


def _require_complete_app_evidence(
    installations: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
    *,
    aggregate_time: datetime,
) -> None:
    expected_roles = {("installation-list", None)}
    expected_urls = {("installation-list", None): "https://github.com/settings/installations"}
    for installation in installations:
        installation_id = installation["id"]
        configure_role = ("installation-configure", installation_id)
        expected_roles.add(configure_role)
        expected_urls[configure_role] = (
            f"https://github.com/settings/installations/{installation_id}"
        )
        if installation["permission_update_requested"]:
            update_role = ("permission-update", installation_id)
            expected_roles.add(update_role)
            expected_urls[update_role] = (
                f"https://github.com/settings/installations/{installation_id}/permissions/update"
            )
    actual_roles = {(value["kind"], value["installation_id"]) for value in evidence}
    if actual_roles != expected_roles or len(evidence) != len(expected_roles):
        raise ValueError(
            "installed App evidence must contain exactly one complete list page, one "
            "Configure page per installation, and one permission-update page exactly "
            "when an update is pending"
        )
    page_times: list[datetime] = []
    for item in evidence:
        role = (item["kind"], item["installation_id"])
        if item["source_url"] != expected_urls[role]:
            raise ValueError(
                f"installed App evidence source URL differs for {role!r}: "
                f"expected {expected_urls[role]!r}, got {item['source_url']!r}"
            )
        page_time = _aware_utc_timestamp(item["captured_at"], "installed App evidence captured_at")
        page_times.append(page_time)
        page_age = aggregate_time - page_time
        if page_age < -_MAX_CLOCK_SKEW:
            raise ValueError(
                f"installed App evidence page postdates the aggregate capture for {role!r}"
            )
        if page_age > _MAX_INSTALLED_APP_CAPTURE_AGE:
            raise ValueError(
                f"installed App evidence page is stale within the capture session for {role!r}"
            )
    if aggregate_time != max(page_times):
        raise ValueError(
            "installed App authority captured_at must equal the latest evidence page timestamp"
        )


def _normalize_installed_app_authority(
    value: Any,
    *,
    repository: Any,
    capture_principal: Any,
    evidence_reader: Callable[[str], bytes] | None = None,
) -> Mapping[str, Any]:
    authority = _mapping(value, "installed App authority capture")
    expected_keys = {
        "schema_version",
        "captured_at",
        "capture_principal",
        "repository",
        "source_url",
        "coverage_complete",
        "installations",
        "evidence",
    }
    if set(authority) != expected_keys:
        raise ValueError(
            "installed App authority capture keys must be exactly "
            f"{sorted(expected_keys)!r}; got {sorted(authority, key=str)!r}"
        )
    if type(authority.get("schema_version")) is not int or authority.get("schema_version") != 1:
        raise ValueError("installed App authority capture schema_version must be 1")
    captured_datetime = _aware_utc_timestamp(
        authority.get("captured_at"), "installed App authority captured_at"
    )
    if authority.get("capture_principal") != capture_principal:
        raise ValueError(
            "installed App authority capture_principal must match the authenticated "
            "GitHub API principal"
        )
    if authority.get("repository") != repository:
        raise ValueError("installed App authority repository must match the release repository")
    if authority.get("coverage_complete") is not True:
        raise ValueError("installed App authority coverage_complete must be true")
    installations = [
        _canonical_app_installation(raw, "installed App installation")
        for raw in _sequence(authority.get("installations"), "installed App installations")
    ]
    installation_ids = [installation["id"] for installation in installations]
    if len(installation_ids) != len(set(installation_ids)):
        raise ValueError("installed App installation ids must be unique")
    installations.sort(key=lambda installation: installation["id"])
    evidence = [
        _canonical_app_evidence(raw, "installed App evidence")
        for raw in _sequence(authority.get("evidence"), "installed App evidence")
    ]
    evidence_roles = [(value["kind"], value["installation_id"]) for value in evidence]
    if len(evidence_roles) != len(set(evidence_roles)):
        raise ValueError("installed App evidence page roles must be unique")
    evidence_filenames = [value["filename"] for value in evidence]
    if len(evidence_filenames) != len(set(evidence_filenames)):
        raise ValueError("installed App evidence filenames must be unique")
    evidence_digests = [value["sha256"] for value in evidence]
    if len(evidence_digests) != len(set(evidence_digests)):
        raise ValueError("installed App evidence file digests must be unique")
    if evidence_reader is not None:
        for item in evidence:
            raw = evidence_reader(item["filename"])
            if not isinstance(raw, bytes):
                raise ValueError("installed App evidence reader must return bytes")
            if len(raw) != item["bytes"]:
                raise ValueError(
                    f"installed App evidence {item['filename']!r} byte count differs "
                    "from the captured manifest"
                )
            if hashlib.sha256(raw).hexdigest() != item["sha256"]:
                raise ValueError(
                    f"installed App evidence {item['filename']!r} digest differs "
                    "from the captured manifest"
                )
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"installed App evidence {item['filename']!r} is not strict UTF-8 text"
                ) from exc
            expected_header = _app_evidence_header(item)
            if not text.startswith(expected_header) or len(text) == len(expected_header):
                raise ValueError(
                    f"installed App evidence {item['filename']!r} does not begin with the "
                    "exact source/time/kind/installation header followed by page content"
                )
    evidence.sort(
        key=lambda item: (
            _APP_EVIDENCE_KIND_ORDER[item["kind"]],
            item["installation_id"] or 0,
        )
    )
    _require_complete_app_evidence(
        installations,
        evidence,
        aggregate_time=captured_datetime,
    )
    return {
        "schema_version": 1,
        "captured_at": captured_datetime.astimezone(timezone.utc).isoformat(),
        "capture_principal": capture_principal,
        "repository": repository,
        "source_url": authority.get("source_url"),
        "coverage_complete": True,
        "installations": installations,
        "evidence": evidence,
    }


def _read_installed_app_evidence_file(directory: Path, filename: str) -> bytes:
    if (
        not isinstance(filename, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", filename) is None
        or filename in {".", ".."}
    ):
        raise ValueError("installed App evidence filename must be a safe basename")
    root = directory.resolve(strict=True)
    candidate = root / filename
    before = candidate.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or candidate.is_symlink():
        raise ValueError(f"installed App evidence {filename!r} must be a single-link regular file")
    if not 1 <= before.st_size <= _MAX_APP_EVIDENCE_BYTES:
        raise ValueError(f"installed App evidence {filename!r} has an invalid size")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(candidate, flags)
    try:
        opened = os.fstat(descriptor)
        identity_fields = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns")
        if any(getattr(before, field) != getattr(opened, field) for field in identity_fields):
            raise ValueError(f"installed App evidence {filename!r} changed while opening")
        chunks: list[bytes] = []
        remaining = _MAX_APP_EVIDENCE_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if any(getattr(opened, field) != getattr(after, field) for field in identity_fields):
            raise ValueError(f"installed App evidence {filename!r} changed while reading")
        if len(raw) != opened.st_size:
            raise ValueError(f"installed App evidence {filename!r} could not be read completely")
        return raw
    finally:
        os.close(descriptor)


def verify_installed_app_restoration(
    *,
    before: Mapping[str, Any],
    restored: Mapping[str, Any],
    repository: str,
    capture_principal: str,
    before_evidence_reader: Callable[[str], bytes],
    restored_evidence_reader: Callable[[str], bytes],
    now: datetime | None = None,
) -> Mapping[str, Any]:
    """Prove that App access returned exactly to its separately captured pre-window state."""
    normalized_before = _normalize_installed_app_authority(
        before,
        repository=repository,
        capture_principal=capture_principal,
        evidence_reader=before_evidence_reader,
    )
    normalized_restored = _normalize_installed_app_authority(
        restored,
        repository=repository,
        capture_principal=capture_principal,
        evidence_reader=restored_evidence_reader,
    )
    if normalized_before["source_url"] != normalized_restored["source_url"]:
        raise ValueError("restored installed App source URL differs from the pre-window record")
    if normalized_before["installations"] != normalized_restored["installations"]:
        raise ValueError("restored installed App state differs from the pre-window record")
    before_digests = {item["sha256"] for item in normalized_before["evidence"]}
    restored_digests = {item["sha256"] for item in normalized_restored["evidence"]}
    if before_digests.intersection(restored_digests):
        raise ValueError("restored installed App evidence reuses a pre-window page capture")
    before_time = _aware_utc_timestamp(
        normalized_before["captured_at"], "pre-window installed App captured_at"
    )
    restored_time = _aware_utc_timestamp(
        normalized_restored["captured_at"], "restored installed App captured_at"
    )
    if restored_time <= before_time:
        raise ValueError("restored installed App capture must postdate the pre-window capture")
    verified_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if restored_time - verified_at > _MAX_CLOCK_SKEW:
        raise ValueError("restored installed App capture is in the future")
    if verified_at - restored_time > _MAX_SNAPSHOT_AGE:
        raise ValueError("restored installed App capture is stale; recapture within 30 minutes")
    before_pages = {
        (item["kind"], item["installation_id"]): item for item in normalized_before["evidence"]
    }
    for item in normalized_restored["evidence"]:
        role = (item["kind"], item["installation_id"])
        restored_page_time = _aware_utc_timestamp(
            item["captured_at"], "restored installed App evidence captured_at"
        )
        page_age = verified_at - restored_page_time
        if page_age < -_MAX_CLOCK_SKEW:
            raise ValueError(f"restored installed App evidence page is in the future for {role!r}")
        if page_age > _MAX_SNAPSHOT_AGE:
            raise ValueError(
                f"restored installed App evidence page is stale for {role!r}; "
                "recapture within 30 minutes"
            )
        before_page_time = _aware_utc_timestamp(
            before_pages[role]["captured_at"],
            "pre-window installed App evidence captured_at",
        )
        if restored_page_time <= before_page_time:
            raise ValueError(
                f"restored installed App evidence page must postdate the matching "
                f"pre-window page for {role!r}"
            )

    def digest(value: Mapping[str, Any]) -> str:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    return {
        "schema_version": 1,
        "verified_at": verified_at.isoformat(),
        "repository": repository,
        "capture_principal": capture_principal,
        "pre_window_capture_sha256": digest(normalized_before),
        "restored_capture_sha256": digest(normalized_restored),
        "restoration_exact": True,
        "installations": normalized_restored["installations"],
    }


def _manifest_input_paths(path: Path, manifest: Mapping[str, Any]) -> list[Path]:
    paths = [path]
    for raw in _sequence(manifest.get("evidence"), "installed App evidence"):
        item = _mapping(raw, "installed App evidence")
        filename = item.get("filename")
        if (
            not isinstance(filename, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", filename) is None
            or filename in {".", ".."}
        ):
            raise ValueError("installed App evidence filename must be a safe basename")
        paths.append(path.parent / filename)
    return paths


def _reject_output_aliases(output: Path, protected: Sequence[Path]) -> None:
    output_paths = (output, output.with_suffix(output.suffix + ".sha256"))
    protected_paths = {path.resolve(strict=False) for path in protected}
    for path in output_paths:
        if path.resolve(strict=False) in protected_paths:
            raise ValueError(
                f"output path {str(path)!r} must not overwrite an input or retained evidence file"
            )


def load_policy(path: Path) -> tuple[Mapping[str, Any], str]:
    """Load a policy and return it with its exact-file SHA-256 digest."""
    raw = path.read_bytes()
    policy = _mapping(json.loads(raw), "release control policy")
    if policy.get("schema_version") != 1:
        raise ValueError("release control policy schema_version must be 1")
    return policy, hashlib.sha256(raw).hexdigest()


def evaluate_controls(policy: Mapping[str, Any], observation: Mapping[str, Any]) -> list[str]:
    """Return every policy difference in deterministic order."""
    violations: list[str] = []
    expected_repository = policy.get("repository")
    if observation.get("repository") != expected_repository:
        violations.append(
            f"repository: expected {expected_repository!r}, got {observation.get('repository')!r}"
        )
    expected_branch = policy.get("default_branch")
    if observation.get("default_branch") != expected_branch:
        violations.append(
            f"default_branch: expected {expected_branch!r}, got {observation.get('default_branch')!r}"
        )
    if observation.get("tag_exists") is not False:
        violations.append("release tag already exists; the control snapshot must precede tagging")

    fork_approval_policy = _mapping(
        policy.get("fork_pr_contributor_approval"),
        "fork pull-request contributor approval policy",
    )
    if set(fork_approval_policy) != {"approval_policy"}:
        violations.append(
            "fork_pr_contributor_approval policy keys must be exactly ['approval_policy']"
        )
    expected_fork_approval = fork_approval_policy.get("approval_policy")
    if expected_fork_approval != "all_external_contributors":
        violations.append(
            "fork_pr_contributor_approval policy must require "
            f"'all_external_contributors'; got {expected_fork_approval!r}"
        )
    fork_approval = _mapping(
        observation.get("fork_pr_contributor_approval"),
        "fork pull-request contributor approval observation",
    )
    if set(fork_approval) != {"approval_policy"}:
        violations.append(
            "fork_pr_contributor_approval observation keys must be exactly ['approval_policy']"
        )
    actual_fork_approval = fork_approval.get("approval_policy")
    if actual_fork_approval != expected_fork_approval:
        violations.append(
            "fork_pr_contributor_approval.approval_policy: expected "
            f"{expected_fork_approval!r}, got {actual_fork_approval!r}"
        )

    expected_principals = _canonical_names(
        _sequence(policy.get("admin_snapshot_principals"), "admin_snapshot_principals"),
        "admin_snapshot_principals",
    )
    if observation.get("capture_principal") not in expected_principals:
        violations.append(
            f"capture_principal: expected one of {expected_principals!r}, "
            f"got {observation.get('capture_principal')!r}"
        )

    authority_policy = _mapping(
        policy.get("release_runner_authority"), "release runner authority policy"
    )
    expected_collaborator_logins = _canonical_names(
        _sequence(
            authority_policy.get("allowed_collaborator_logins"),
            "release runner allowed collaborator logins",
        ),
        "release runner allowed collaborator logins",
    )
    expected_invitations = list(
        _sequence(
            authority_policy.get("pending_invitations"),
            "release runner pending invitations policy",
        )
    )
    if expected_invitations:
        violations.append(
            "release_runner_authority.pending_invitations policy must be an empty array"
        )

    authority = _mapping(
        observation.get("release_runner_authority"),
        "release runner authority observation",
    )
    actual_collaborator_logins: list[str] = []
    actual_write_logins: list[str] = []
    seen_collaborators: set[str] = set()
    for raw_collaborator in _sequence(
        authority.get("collaborators"), "release runner collaborators"
    ):
        collaborator = _mapping(raw_collaborator, "release runner collaborator")
        login = collaborator.get("login")
        if not isinstance(login, str) or not login:
            violations.append("release runner collaborator login must be a non-empty string")
            continue
        if login in seen_collaborators:
            violations.append(f"release runner collaborator {login!r} is duplicated")
            continue
        seen_collaborators.add(login)
        actual_collaborator_logins.append(login)
        permissions = _mapping(
            collaborator.get("permissions"),
            f"release runner collaborator {login!r} permissions",
        )
        effective_write = False
        for permission_name in ("admin", "maintain", "push"):
            permission_value = permissions.get(permission_name)
            if not isinstance(permission_value, bool):
                violations.append(
                    f"release runner collaborator {login!r} permission "
                    f"{permission_name!r} must be boolean"
                )
            elif permission_value:
                effective_write = True
        if effective_write:
            actual_write_logins.append(login)
    actual_collaborator_logins.sort()
    actual_write_logins.sort()
    if actual_collaborator_logins != expected_collaborator_logins:
        violations.append(
            "release_runner_authority.allowed_collaborator_logins: expected "
            f"{expected_collaborator_logins!r}, got {actual_collaborator_logins!r}"
        )
    missing_write_logins = sorted(set(expected_collaborator_logins) - set(actual_write_logins))
    if missing_write_logins:
        violations.append(
            "release_runner_authority.required_write_logins: expected effective write for "
            f"{expected_collaborator_logins!r}, missing {missing_write_logins!r}"
        )

    pending_invitations = list(
        _sequence(
            authority.get("pending_invitations"),
            "release runner pending invitations",
        )
    )
    if pending_invitations != expected_invitations:
        violations.append(
            "release_runner_authority.pending_invitations: expected "
            f"{expected_invitations!r}, got {pending_invitations!r}"
        )

    for field in ("registered_runners", "repository_variable_names"):
        expected_values = list(
            _sequence(
                authority_policy.get(field),
                f"release runner {field} policy",
            )
        )
        if expected_values:
            violations.append(f"release_runner_authority.{field} policy must be an empty array")
        actual_values = list(_sequence(authority.get(field), f"release runner {field}"))
        if actual_values != expected_values:
            violations.append(
                f"release_runner_authority.{field}: expected "
                f"{expected_values!r}, got {actual_values!r}"
            )

    installed_apps_policy = _mapping(
        authority_policy.get("installed_apps"),
        "release runner installed Apps policy",
    )
    if set(installed_apps_policy) != {"source_url", "expected_installations"}:
        violations.append(
            "release_runner_authority.installed_apps policy keys must be exactly "
            "['expected_installations', 'source_url']"
        )
    expected_installations = [
        _canonical_app_installation(value, "expected installed App installation")
        for value in _sequence(
            installed_apps_policy.get("expected_installations"),
            "expected installed App installations",
        )
    ]
    expected_installations.sort(key=lambda installation: installation["id"])
    expected_installation_ids = [value["id"] for value in expected_installations]
    if len(expected_installation_ids) != len(set(expected_installation_ids)):
        violations.append("expected installed App installation ids must be unique")
    sensitive_permissions = {"actions", "administration", "workflows"}
    for installation in expected_installations:
        effective_sensitive = sensitive_permissions.intersection(
            installation["permissions"]["write"]
        )
        requested_sensitive = sensitive_permissions.intersection(
            installation["requested_additional_permissions"]["write"]
        )
        if (
            installation["repository_access"]
            and not installation["suspended"]
            and (effective_sensitive or requested_sensitive)
        ):
            violations.append(
                "release_runner_authority.installed_apps policy leaves active runner "
                f"authority for {installation['name']!r}: "
                f"{sorted(effective_sensitive | requested_sensitive)!r}"
            )
        if installation["permission_update_requested"] and not installation["suspended"]:
            violations.append(
                "release_runner_authority.installed_apps policy leaves an unresolved "
                f"permission update active for {installation['name']!r}"
            )
    try:
        installed_apps = _normalize_installed_app_authority(
            authority.get("installed_apps"),
            repository=observation.get("repository"),
            capture_principal=observation.get("capture_principal"),
        )
    except ValueError as exc:
        violations.append(f"release_runner_authority.installed_apps: invalid ({exc})")
    else:
        if installed_apps.get("source_url") != installed_apps_policy.get("source_url"):
            violations.append(
                "release_runner_authority.installed_apps.source_url: expected "
                f"{installed_apps_policy.get('source_url')!r}, "
                f"got {installed_apps.get('source_url')!r}"
            )
        if installed_apps.get("installations") != expected_installations:
            violations.append(
                "release_runner_authority.installed_apps.installations differ from the "
                "complete reviewed installation policy"
            )

    branch = _mapping(observation.get("branch_protection"), "branch_protection")
    branch_policy = _mapping(policy.get("branch_protection"), "branch_protection policy")
    branch_fields = {
        "admin_enforcement": ("enforce_admins", "enabled"),
        "strict_required_checks": ("required_status_checks", "strict"),
        "resolved_conversations": ("required_conversation_resolution", "enabled"),
        "allow_force_pushes": ("allow_force_pushes", "enabled"),
        "allow_deletions": ("allow_deletions", "enabled"),
    }
    for policy_name, path in branch_fields.items():
        expected = branch_policy.get(policy_name)
        try:
            actual = _nested(branch, *path)
        except (KeyError, ValueError):
            actual = "<missing>"
        if actual is not expected:
            violations.append(f"main.{policy_name}: expected {expected!r}, got {actual!r}")

    expected_checks = _canonical_names(
        _sequence(policy.get("required_checks"), "required_checks"), "required_checks"
    )
    try:
        actual_checks = _canonical_names(
            _sequence(
                _nested(branch, "required_status_checks", "contexts"),
                "branch required check contexts",
            ),
            "branch required check contexts",
        )
    except (KeyError, ValueError) as exc:
        violations.append(f"main.required_checks: invalid or missing ({exc})")
        actual_checks = []
    if actual_checks != expected_checks:
        violations.append(
            f"main.required_checks: expected {expected_checks!r}, got {actual_checks!r}"
        )

    provider_policy = _mapping(
        policy.get("required_check_provider"), "required_check_provider policy"
    )
    expected_app_id = provider_policy.get("app_id")
    expected_app_slug = provider_policy.get("slug")
    if not isinstance(expected_app_id, int) or expected_app_id <= 0:
        violations.append("required_check_provider.app_id must be a positive integer")
    if not isinstance(expected_app_slug, str) or not expected_app_slug:
        violations.append("required_check_provider.slug must be a non-empty string")
    try:
        branch_bindings = [
            _mapping(value, "branch required check binding")
            for value in _sequence(
                _nested(branch, "required_status_checks", "checks"),
                "branch required check bindings",
            )
        ]
        actual_bindings = sorted(
            (value.get("context"), value.get("app_id")) for value in branch_bindings
        )
    except (KeyError, ValueError) as exc:
        violations.append(f"main.required_check_bindings: invalid or missing ({exc})")
        actual_bindings = []
    expected_bindings = sorted((name, expected_app_id) for name in expected_checks)
    if actual_bindings != expected_bindings:
        violations.append(
            f"main.required_check_bindings: expected {expected_bindings!r}, "
            f"got {actual_bindings!r}"
        )

    workflow_policy = _mapping(
        policy.get("required_check_workflows"), "required check workflows policy"
    )
    if set(workflow_policy) != set(expected_checks):
        violations.append(
            "required_check_workflows: keys must exactly match required_checks; "
            f"expected {sorted(expected_checks)!r}, got {sorted(workflow_policy)!r}"
        )

    immutable_policy = _mapping(policy.get("immutable_releases"), "immutable releases policy")
    immutable_releases = _mapping(
        observation.get("immutable_releases"), "immutable releases observation"
    )
    if immutable_policy.get("enabled") is not True:
        violations.append(
            "immutable_releases policy.enabled must be the JSON boolean true; "
            f"got {immutable_policy.get('enabled')!r}"
        )
    if immutable_releases.get("enabled") is not True:
        violations.append(
            "immutable_releases.enabled: expected true, got "
            f"{immutable_releases.get('enabled')!r}"
        )

    latest_checks: dict[str, Mapping[str, Any]] = {}
    for raw_check in _sequence(observation.get("check_runs"), "check_runs"):
        check = _mapping(raw_check, "check run")
        name = check.get("name")
        if isinstance(name, str) and name not in latest_checks:
            latest_checks[name] = check
    for name in expected_checks:
        latest_check = latest_checks.get(name)
        if latest_check is None:
            violations.append(f"release commit check {name!r}: missing")
        elif (
            latest_check.get("status") != "completed" or latest_check.get("conclusion") != "success"
        ):
            violations.append(
                f"release commit check {name!r}: expected completed/success, got "
                f"{latest_check.get('status')!r}/{latest_check.get('conclusion')!r}"
            )
        else:
            app = latest_check.get("app")
            if not isinstance(app, Mapping):
                violations.append(f"release commit check {name!r}: missing check provider app")
            elif app.get("id") != expected_app_id or app.get("slug") != expected_app_slug:
                violations.append(
                    f"release commit check {name!r}: expected provider "
                    f"{expected_app_slug!r}/{expected_app_id!r}, got "
                    f"{app.get('slug')!r}/{app.get('id')!r}"
                )
            else:
                expected_workflow = _mapping(
                    workflow_policy.get(name), f"required workflow policy for {name!r}"
                )
                workflow_run = latest_check.get("workflow_run")
                if not isinstance(workflow_run, Mapping):
                    violations.append(
                        f"release commit check {name!r}: missing bound Actions workflow run"
                    )
                    continue
                expected_run_fields = {
                    "repository": policy.get("repository"),
                    "path": expected_workflow.get("path"),
                    "event": expected_workflow.get("event"),
                    "head_branch": expected_workflow.get("head_branch"),
                    "head_sha": observation.get("release_commit"),
                    "status": "completed",
                    "conclusion": "success",
                }
                for field, expected in expected_run_fields.items():
                    if workflow_run.get(field) != expected:
                        violations.append(
                            f"release commit check {name!r}: workflow run {field} expected "
                            f"{expected!r}, got {workflow_run.get(field)!r}"
                        )

    tag_policy = _mapping(policy.get("tag_ruleset"), "tag_ruleset policy")
    matching_rulesets = [
        _mapping(value, "ruleset")
        for value in _sequence(observation.get("rulesets"), "rulesets")
        if isinstance(value, Mapping) and value.get("name") == tag_policy.get("name")
    ]
    if len(matching_rulesets) != 1:
        violations.append(
            f"tag ruleset {tag_policy.get('name')!r}: expected exactly one, got "
            f"{len(matching_rulesets)}"
        )
    else:
        ruleset = matching_rulesets[0]
        scalar_fields = ("target", "enforcement", "current_user_can_bypass")
        for field in scalar_fields:
            if ruleset.get(field) != tag_policy.get(field):
                violations.append(
                    f"tag_ruleset.{field}: expected {tag_policy.get(field)!r}, "
                    f"got {ruleset.get(field)!r}"
                )
        conditions = _mapping(ruleset.get("conditions"), "ruleset conditions")
        ref_name = _mapping(conditions.get("ref_name"), "ruleset ref_name")
        for field in ("include", "exclude"):
            expected = sorted(_sequence(tag_policy.get(field), f"tag policy {field}"))
            actual = sorted(_sequence(ref_name.get(field), f"tag ruleset {field}"))
            if actual != expected:
                violations.append(f"tag_ruleset.{field}: expected {expected!r}, got {actual!r}")
        actual_rule_types = _canonical_names(
            [
                _mapping(value, "tag rule").get("type")
                for value in _sequence(ruleset.get("rules"), "tag rules")
            ],
            "tag rule types",
        )
        expected_rule_types = _canonical_names(
            _sequence(tag_policy.get("rule_types"), "tag policy rule_types"),
            "tag policy rule types",
        )
        if actual_rule_types != expected_rule_types:
            violations.append(
                "tag_ruleset.rule_types: expected "
                f"{expected_rule_types!r}, got {actual_rule_types!r}"
            )
        if ruleset.get("bypass_actors") != tag_policy.get("bypass_actors"):
            violations.append(
                f"tag_ruleset.bypass_actors: expected {tag_policy.get('bypass_actors')!r}, "
                f"got {ruleset.get('bypass_actors')!r}"
            )

    environment_policy = _mapping(policy.get("pypi_environment"), "pypi environment policy")
    environment = _mapping(observation.get("pypi_environment"), "pypi_environment")
    for field in ("name", "can_admins_bypass", "deployment_branch_policy"):
        if environment.get(field) != environment_policy.get(field):
            violations.append(
                f"pypi_environment.{field}: expected {environment_policy.get(field)!r}, "
                f"got {environment.get(field)!r}"
            )

    reviewer_rules = [
        _mapping(value, "environment protection rule")
        for value in _sequence(environment.get("protection_rules"), "environment protection_rules")
        if isinstance(value, Mapping) and value.get("type") == "required_reviewers"
    ]
    if len(reviewer_rules) != 1:
        violations.append(
            f"pypi_environment.required_reviewers: expected one rule, got {len(reviewer_rules)}"
        )
    else:
        reviewer_rule = reviewer_rules[0]
        reviewer_logins = sorted(
            _mapping(value, "environment reviewer").get("reviewer", {}).get("login")
            for value in _sequence(reviewer_rule.get("reviewers"), "environment reviewers")
        )
        expected_logins = sorted(
            _sequence(
                environment_policy.get("required_reviewer_logins"),
                "required reviewer logins",
            )
        )
        if reviewer_logins != expected_logins:
            violations.append(
                f"pypi_environment.reviewers: expected {expected_logins!r}, "
                f"got {reviewer_logins!r}"
            )
        if reviewer_rule.get("prevent_self_review") is not environment_policy.get(
            "prevent_self_review"
        ):
            violations.append(
                "pypi_environment.prevent_self_review: expected "
                f"{environment_policy.get('prevent_self_review')!r}, "
                f"got {reviewer_rule.get('prevent_self_review')!r}"
            )

    actual_deployment_policies = sorted(
        [
            {"name": item.get("name"), "type": item.get("type")}
            for item in (
                _mapping(value, "deployment policy")
                for value in _sequence(
                    observation.get("deployment_policies"), "deployment_policies"
                )
            )
        ],
        key=lambda item: (str(item["type"]), str(item["name"])),
    )
    expected_deployment_policies = sorted(
        [
            dict(_mapping(value, "expected deployment policy"))
            for value in _sequence(
                environment_policy.get("deployment_policies"), "expected deployment policies"
            )
        ],
        key=lambda item: (str(item["type"]), str(item["name"])),
    )
    if actual_deployment_policies != expected_deployment_policies:
        violations.append(
            "pypi_environment.deployment_policies: expected "
            f"{expected_deployment_policies!r}, got {actual_deployment_policies!r}"
        )

    for observation_name, policy_name in (
        ("repository_secret_names", "repository_secret_names"),
        ("environment_secret_names", "environment_secret_names"),
    ):
        actual = sorted(_sequence(observation.get(observation_name), observation_name))
        expected = sorted(_sequence(environment_policy.get(policy_name), policy_name))
        if actual != expected:
            violations.append(f"{observation_name}: expected {expected!r}, got {actual!r}")
    return violations


class GitHubApi:
    """Small fail-closed GitHub JSON client used only by the preflight."""

    def __init__(self, *, token: str, api_url: str = "https://api.github.com"):
        if not token:
            raise ValueError("a GitHub token is required for the external-control preflight")
        self._token = token
        self._api_url = api_url.rstrip("/")

    def get(self, path: str) -> Any:
        request = urllib.request.Request(
            f"{self._api_url}/{path.lstrip('/')}",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self._token}",
                "User-Agent": "explainiverse-release-preflight",
                "X-GitHub-Api-Version": "2026-03-10",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise ApiNotFoundError(path) from exc
            raise RuntimeError(f"GitHub API {path!r} returned HTTP {exc.code}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"GitHub API {path!r} could not be read: {exc}") from exc


def capture_observation(
    *,
    policy: Mapping[str, Any],
    release_tag: str,
    release_commit: str,
    get_json: Callable[[str], Any],
    installed_app_authority: Mapping[str, Any],
    installed_app_evidence_reader: Callable[[str], bytes],
) -> Mapping[str, Any]:
    """Capture all policy-controlled GitHub state through an injected client."""
    repository = str(policy["repository"])
    branch = str(policy["default_branch"])
    environment_name = str(_mapping(policy["pypi_environment"], "environment")["name"])
    root = f"repos/{repository}"

    try:
        immutable_releases = _mapping(
            get_json(f"{root}/immutable-releases"), "immutable releases response"
        )
    except ApiNotFoundError:
        immutable_releases = {"enabled": False, "enforced_by_owner": False}
    ruleset_summaries = _sequence(get_json(f"{root}/rulesets"), "ruleset summaries")
    rulesets = [
        get_json(f"{root}/rulesets/{_mapping(value, 'ruleset summary')['id']}")
        for value in ruleset_summaries
    ]
    deployment_response = _mapping(
        get_json(f"{root}/environments/{environment_name}/deployment-branch-policies"),
        "deployment policy response",
    )
    repository_secrets = _mapping(
        get_json(f"{root}/actions/secrets"), "repository secrets response"
    )
    environment_secrets = _mapping(
        get_json(f"{root}/environments/{environment_name}/secrets"),
        "environment secrets response",
    )
    raw_fork_approval = _mapping(
        get_json(f"{root}/actions/permissions/fork-pr-contributor-approval"),
        "fork pull-request contributor approval response",
    )
    raw_collaborators = list(
        _sequence(
            get_json(f"{root}/collaborators?affiliation=all&per_page=100"),
            "repository collaborators response",
        )
    )
    if len(raw_collaborators) >= 100:
        raise ValueError(
            "repository collaborators capture may be incomplete at the 100-entry page limit"
        )
    raw_invitations = list(
        _sequence(
            get_json(f"{root}/invitations?per_page=100"),
            "repository invitations response",
        )
    )
    if len(raw_invitations) >= 100:
        raise ValueError(
            "repository invitations capture may be incomplete at the 100-entry page limit"
        )
    raw_runners_response = _mapping(
        get_json(f"{root}/actions/runners?per_page=100"),
        "repository runners response",
    )
    raw_runners = list(_sequence(raw_runners_response.get("runners"), "repository runners"))
    if raw_runners_response.get("total_count") != len(raw_runners):
        raise ValueError(
            "repository runners capture is incomplete: "
            f"total_count={raw_runners_response.get('total_count')!r}, "
            f"captured={len(raw_runners)}"
        )
    raw_variables_response = _mapping(
        get_json(f"{root}/actions/variables?per_page=100"),
        "repository variables response",
    )
    raw_variables = list(_sequence(raw_variables_response.get("variables"), "repository variables"))
    if raw_variables_response.get("total_count") != len(raw_variables):
        raise ValueError(
            "repository variables capture is incomplete: "
            f"total_count={raw_variables_response.get('total_count')!r}, "
            f"captured={len(raw_variables)}"
        )
    check_response = _mapping(
        get_json(f"{root}/commits/{release_commit}/check-runs?per_page=100"),
        "check runs response",
    )
    raw_checks = [
        _mapping(value, "check run")
        for value in _sequence(check_response.get("check_runs"), "check runs")
    ]
    total_checks = check_response.get("total_count")
    if total_checks is not None and total_checks != len(raw_checks):
        raise ValueError(
            "check-runs capture is incomplete: "
            f"total_count={total_checks!r}, captured={len(raw_checks)}"
        )
    try:
        get_json(f"{root}/git/ref/tags/{urllib.parse.quote(release_tag, safe='')}")
    except ApiNotFoundError:
        tag_exists = False
    else:
        tag_exists = True

    def secret_names(response: Mapping[str, Any]) -> list[str]:
        return sorted(
            str(_mapping(value, "secret metadata")["name"])
            for value in _sequence(response.get("secrets"), "secret metadata")
        )

    normalized_collaborators = []
    for raw_value in raw_collaborators:
        value = _mapping(raw_value, "repository collaborator")
        permissions = _mapping(value.get("permissions"), "repository collaborator permissions")
        normalized_collaborators.append(
            {
                "login": value.get("login"),
                "role_name": value.get("role_name"),
                "permissions": {
                    name: permissions.get(name) for name in ("admin", "maintain", "push")
                },
            }
        )
    normalized_collaborators.sort(key=lambda value: str(value["login"]))

    normalized_invitations = []
    for raw_value in raw_invitations:
        value = _mapping(raw_value, "repository invitation")
        invitee = _mapping(value.get("invitee"), "repository invitation invitee")
        normalized_invitations.append(
            {
                "id": value.get("id"),
                "invitee": invitee.get("login"),
                "permissions": value.get("permissions"),
            }
        )
    normalized_invitations.sort(key=lambda value: (str(value["invitee"]), str(value["id"])))

    normalized_runners = []
    for raw_value in raw_runners:
        value = _mapping(raw_value, "repository runner")
        normalized_runners.append(
            {
                "id": value.get("id"),
                "name": value.get("name"),
                "os": value.get("os"),
                "status": value.get("status"),
                "busy": value.get("busy"),
                "labels": [
                    {
                        "id": label.get("id"),
                        "name": label.get("name"),
                        "type": label.get("type"),
                    }
                    for label in (
                        _mapping(item, "repository runner label")
                        for item in _sequence(value.get("labels"), "repository runner labels")
                    )
                ],
            }
        )
    normalized_runners.sort(key=lambda value: (str(value["name"]), str(value["id"])))

    repository_variable_names = sorted(
        str(_mapping(value, "repository variable metadata")["name"]) for value in raw_variables
    )

    raw_branch = _mapping(
        get_json(f"{root}/branches/{branch}/protection"), "branch protection response"
    )
    raw_environment = _mapping(
        get_json(f"{root}/environments/{environment_name}"), "environment response"
    )
    principal = _mapping(get_json("user"), "authenticated GitHub user").get("login")
    if not isinstance(principal, str) or not principal:
        raise ValueError("authenticated GitHub user response has no login")
    normalized_installed_apps = _normalize_installed_app_authority(
        installed_app_authority,
        repository=repository,
        capture_principal=principal,
        evidence_reader=installed_app_evidence_reader,
    )

    def enabled_field(name: str) -> Mapping[str, Any]:
        value = _mapping(raw_branch.get(name), f"branch protection {name}")
        return {"enabled": value.get("enabled")}

    required_status_checks = _mapping(
        raw_branch.get("required_status_checks"), "required status checks"
    )
    normalized_branch = {
        "enforce_admins": enabled_field("enforce_admins"),
        "required_status_checks": {
            "strict": required_status_checks.get("strict"),
            "contexts": required_status_checks.get("contexts"),
            "checks": required_status_checks.get("checks"),
        },
        "required_conversation_resolution": enabled_field("required_conversation_resolution"),
        "allow_force_pushes": enabled_field("allow_force_pushes"),
        "allow_deletions": enabled_field("allow_deletions"),
    }
    normalized_rulesets = []
    for raw_value in rulesets:
        value = _mapping(raw_value, "ruleset detail")
        normalized_rulesets.append(
            {
                field: value.get(field)
                for field in (
                    "name",
                    "target",
                    "enforcement",
                    "conditions",
                    "rules",
                    "bypass_actors",
                    "current_user_can_bypass",
                )
            }
        )
    normalized_protection_rules = []
    for raw_value in _sequence(
        raw_environment.get("protection_rules"), "environment protection rules"
    ):
        value = _mapping(raw_value, "environment protection rule")
        if value.get("type") == "required_reviewers":
            normalized_protection_rules.append(
                {
                    "type": "required_reviewers",
                    "prevent_self_review": value.get("prevent_self_review"),
                    "reviewers": [
                        {
                            "type": reviewer.get("type"),
                            "reviewer": {
                                "login": _mapping(
                                    reviewer.get("reviewer"), "environment reviewer identity"
                                ).get("login")
                            },
                        }
                        for reviewer in (
                            _mapping(item, "environment reviewer")
                            for item in _sequence(value.get("reviewers"), "environment reviewers")
                        )
                    ],
                }
            )
        else:
            normalized_protection_rules.append({"type": value.get("type")})
    normalized_environment = {
        "name": raw_environment.get("name"),
        "can_admins_bypass": raw_environment.get("can_admins_bypass"),
        "deployment_branch_policy": raw_environment.get("deployment_branch_policy"),
        "protection_rules": normalized_protection_rules,
    }
    required_names = set(
        _canonical_names(
            _sequence(policy.get("required_checks"), "required checks policy"),
            "required checks policy",
        )
    )
    actions_run_cache: dict[str, Mapping[str, Any]] = {}
    normalized_checks = []
    details_pattern = re.compile(
        rf"^https://github\.com/{re.escape(repository)}/actions/runs/([1-9][0-9]*)(?:/job/[1-9][0-9]*)?(?:\?.*)?$"
    )
    for check in raw_checks:
        details_url = check.get("details_url")
        workflow_run = None
        if check.get("name") in required_names and isinstance(details_url, str):
            match = details_pattern.fullmatch(details_url)
            if match is not None:
                run_id = match.group(1)
                if run_id not in actions_run_cache:
                    actions_run_cache[run_id] = _mapping(
                        get_json(f"{root}/actions/runs/{run_id}"),
                        "required check Actions run",
                    )
                raw_run = actions_run_cache[run_id]
                raw_repository = raw_run.get("repository")
                workflow_run = {
                    "id": raw_run.get("id"),
                    "repository": (
                        raw_repository.get("full_name")
                        if isinstance(raw_repository, Mapping)
                        else None
                    ),
                    "path": raw_run.get("path"),
                    "event": raw_run.get("event"),
                    "head_branch": raw_run.get("head_branch"),
                    "head_sha": raw_run.get("head_sha"),
                    "status": raw_run.get("status"),
                    "conclusion": raw_run.get("conclusion"),
                    "run_attempt": raw_run.get("run_attempt"),
                }
        normalized_checks.append(
            {
                "name": check.get("name"),
                "status": check.get("status"),
                "conclusion": check.get("conclusion"),
                "completed_at": check.get("completed_at"),
                "details_url": details_url,
                "app": {
                    "id": (
                        check.get("app", {}).get("id")
                        if isinstance(check.get("app"), Mapping)
                        else None
                    ),
                    "slug": (
                        check.get("app", {}).get("slug")
                        if isinstance(check.get("app"), Mapping)
                        else None
                    ),
                },
                "workflow_run": workflow_run,
            }
        )

    return {
        "repository": repository,
        "default_branch": branch,
        "capture_principal": principal,
        "release_runner_authority": {
            "collaborators": normalized_collaborators,
            "pending_invitations": normalized_invitations,
            "registered_runners": normalized_runners,
            "repository_variable_names": repository_variable_names,
            "installed_apps": normalized_installed_apps,
        },
        "release_tag": release_tag,
        "release_commit": release_commit,
        "tag_exists": tag_exists,
        "fork_pr_contributor_approval": {
            "approval_policy": raw_fork_approval.get("approval_policy")
        },
        "immutable_releases": {
            "enabled": immutable_releases.get("enabled"),
            "enforced_by_owner": immutable_releases.get("enforced_by_owner"),
        },
        "branch_protection": normalized_branch,
        "rulesets": normalized_rulesets,
        "pypi_environment": normalized_environment,
        "deployment_policies": deployment_response.get("branch_policies"),
        "repository_secret_names": secret_names(repository_secrets),
        "environment_secret_names": secret_names(environment_secrets),
        "check_runs": normalized_checks,
    }


def make_snapshot(
    *,
    policy: Mapping[str, Any],
    policy_sha256: str,
    observation: Mapping[str, Any],
    workflow_run: Mapping[str, Any],
    now: datetime | None = None,
) -> Mapping[str, Any]:
    violations = evaluate_controls(policy, observation)
    observed_at = now or datetime.now(timezone.utc)
    snapshot = {
        "schema_version": 1,
        "observed_at": observed_at.astimezone(timezone.utc).isoformat(),
        "policy_sha256": policy_sha256,
        "workflow_run": dict(workflow_run),
        "observation": dict(observation),
        "pypi_trusted_publisher": {
            "expected": dict(
                _mapping(policy.get("pypi_trusted_publisher"), "trusted publisher policy")
            ),
            "verification_status": "blocked_no_public_read_api",
        },
    }
    try:
        verify_snapshot_freshness(snapshot, now=observed_at)
    except ValueError as exc:
        violations.append(f"snapshot freshness: {exc}")
    snapshot["repository_controls_accepted"] = not violations
    snapshot["violations"] = violations
    return snapshot


def verify_snapshot_freshness(
    snapshot: Mapping[str, Any],
    *,
    now: datetime | None = None,
    max_age: timedelta = _MAX_SNAPSHOT_AGE,
) -> None:
    """Reject future or stale observations whenever a snapshot is consumed."""
    observed_at = snapshot.get("observed_at")
    if not isinstance(observed_at, str):
        raise ValueError("external-control snapshot has no observed_at timestamp")
    try:
        observed = datetime.fromisoformat(observed_at)
    except ValueError as exc:
        raise ValueError("external-control snapshot observed_at is not ISO-8601") from exc
    if observed.tzinfo is None or observed.utcoffset() is None:
        raise ValueError("external-control snapshot observed_at must include a timezone")
    current = now or datetime.now(timezone.utc)
    age = current.astimezone(timezone.utc) - observed.astimezone(timezone.utc)
    if age < -_MAX_CLOCK_SKEW:
        raise ValueError("external-control snapshot observed_at is in the future")
    if age > max_age:
        raise ValueError(f"external-control snapshot is stale ({age}); recapture within {max_age}")

    observation = _mapping(snapshot.get("observation"), "snapshot observation")
    authority = _mapping(
        observation.get("release_runner_authority"),
        "snapshot release runner authority",
    )
    installed_apps = _mapping(
        authority.get("installed_apps"),
        "snapshot installed App authority capture",
    )
    app_captured_at = installed_apps.get("captured_at")
    if not isinstance(app_captured_at, str):
        raise ValueError("installed App authority capture has no captured_at timestamp")
    try:
        app_captured = datetime.fromisoformat(app_captured_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            "installed App authority captured_at is not an RFC 3339 timestamp"
        ) from exc
    if app_captured.tzinfo is None or app_captured.utcoffset() is None:
        raise ValueError("installed App authority captured_at must include a timezone")
    app_age_at_snapshot = observed.astimezone(timezone.utc) - app_captured.astimezone(timezone.utc)
    if app_age_at_snapshot < -_MAX_CLOCK_SKEW:
        raise ValueError("installed App authority captured_at is after the control snapshot")
    if app_age_at_snapshot > _MAX_INSTALLED_APP_CAPTURE_AGE:
        raise ValueError(
            "installed App authority capture is stale at snapshot creation "
            f"({app_age_at_snapshot}); recapture within {_MAX_INSTALLED_APP_CAPTURE_AGE}"
        )
    app_age_at_verification = current.astimezone(timezone.utc) - app_captured.astimezone(
        timezone.utc
    )
    if app_age_at_verification < -_MAX_CLOCK_SKEW:
        raise ValueError("installed App authority captured_at is in the future")
    if app_age_at_verification > max_age:
        raise ValueError(
            "installed App authority capture is stale at verification "
            f"({app_age_at_verification}); recapture within {max_age}"
        )
    for item in _sequence(installed_apps.get("evidence"), "installed App evidence"):
        page = _mapping(item, "installed App evidence")
        page_captured = _aware_utc_timestamp(
            page.get("captured_at"), "installed App evidence captured_at"
        )
        page_age_at_verification = current.astimezone(timezone.utc) - page_captured
        if page_age_at_verification < -_MAX_CLOCK_SKEW:
            raise ValueError("installed App evidence captured_at is in the future")
        if page_age_at_verification > max_age:
            raise ValueError(
                "installed App evidence page is stale at verification "
                f"({page_age_at_verification}); recapture within {max_age}"
            )


def verify_snapshot(
    *,
    policy: Mapping[str, Any],
    policy_sha256: str,
    snapshot: Mapping[str, Any],
    repository: str,
    release_tag: str,
    release_commit: str,
    now: datetime | None = None,
    max_age: timedelta = _MAX_SNAPSHOT_AGE,
) -> None:
    """Verify a preflight artifact before a tag workflow may build or publish."""
    if snapshot.get("schema_version") != 1:
        raise ValueError("external-control snapshot schema_version must be 1")
    if snapshot.get("policy_sha256") != policy_sha256:
        raise ValueError("external-control snapshot was produced from a different policy file")
    verify_snapshot_freshness(snapshot, now=now, max_age=max_age)
    observation = _mapping(snapshot.get("observation"), "snapshot observation")
    expected = {
        "repository": repository,
        "release_tag": release_tag,
        "release_commit": release_commit,
    }
    for field, expected_value in expected.items():
        if observation.get(field) != expected_value:
            raise ValueError(
                f"external-control snapshot {field} mismatch: expected "
                f"{expected_value!r}, got {observation.get(field)!r}"
            )
    violations = evaluate_controls(policy, observation)
    if violations:
        raise ValueError("external-control snapshot fails policy: " + "; ".join(violations))
    if snapshot.get("repository_controls_accepted") is not True:
        raise ValueError("external-control snapshot is not marked accepted")
    if snapshot.get("violations") != []:
        raise ValueError("external-control snapshot contains recorded policy violations")


def _complete_jobs_response(jobs_response: Mapping[str, Any], name: str) -> list[Mapping[str, Any]]:
    if jobs_response.get("query_filter") != "all":
        raise ValueError(f"{name} jobs must be queried with filter=all")
    if jobs_response.get("pagination_complete") is not True:
        raise ValueError(f"{name} jobs response does not prove complete pagination")
    return [
        _mapping(value, f"{name} job")
        for value in _sequence(jobs_response.get("jobs"), f"{name} jobs")
    ]


def _require_first_attempt(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != 1:
        raise ValueError(f"{name} must be the integer 1; got {value!r}")


def _require_positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return value


def _cuda_required_runner_labels(
    cuda_policy: Mapping[str, Any], required_jobs: Sequence[str]
) -> Mapping[str, str]:
    raw_labels = _mapping(
        cuda_policy.get("required_runner_labels"),
        "CUDA evidence required runner labels",
    )
    expected_keys = set(required_jobs)
    actual_keys = set(raw_labels)
    if actual_keys != expected_keys:
        raise ValueError(
            "CUDA evidence required runner label keys must exactly match required jobs: "
            f"expected {sorted(expected_keys)!r}, got {sorted(actual_keys, key=str)!r}"
        )

    labels: dict[str, str] = {}
    for job_name in required_jobs:
        label = raw_labels[job_name]
        if not isinstance(label, str) or not label:
            raise ValueError(
                "CUDA evidence required runner label values must be non-empty strings: "
                f"{job_name!r} has {label!r}"
            )
        if job_name.startswith("CUDA single-GPU "):
            expected_label = "explainiverse-cuda-single"
        elif job_name.startswith("CUDA two-GPU scheduled "):
            expected_label = "explainiverse-cuda-two"
        else:
            raise ValueError(f"CUDA evidence required job {job_name!r} has no supported topology")
        if label != expected_label:
            raise ValueError(
                f"CUDA evidence required runner label for {job_name!r} must be "
                f"{expected_label!r}, got {label!r}"
            )
        labels[job_name] = label
    return labels


def verify_cuda_evidence(
    policy: Mapping[str, Any],
    run: Mapping[str, Any],
    jobs_response: Mapping[str, Any],
    *,
    run_id: str,
    repository: str,
    release_commit: str,
) -> Mapping[str, Any]:
    """Verify and normalize an exact-commit, all-attempt CUDA hardware run."""
    cuda_policy = _mapping(policy.get("cuda_evidence"), "CUDA evidence policy")
    actual_repository = _mapping(run.get("repository"), "CUDA run repository").get("full_name")
    source_run_id = _require_positive_integer(run.get("id"), "CUDA evidence run id")
    expected_fields = {
        "id": (str(source_run_id), str(run_id)),
        "repository": (actual_repository, repository),
        "workflow path": (run.get("path"), cuda_policy.get("workflow_path")),
        "event": (run.get("event"), cuda_policy.get("event")),
        "head branch": (run.get("head_branch"), cuda_policy.get("head_branch")),
        "head SHA": (run.get("head_sha"), release_commit),
        "status": (run.get("status"), "completed"),
        "conclusion": (run.get("conclusion"), "success"),
    }
    for label, (actual, expected) in expected_fields.items():
        if actual != expected:
            raise ValueError(
                f"CUDA evidence run {label} mismatch: expected {expected!r}, got {actual!r}"
            )
    _require_first_attempt(run.get("run_attempt"), "CUDA evidence run attempt")

    jobs = _complete_jobs_response(jobs_response, "CUDA evidence")
    required_jobs = _canonical_names(
        _sequence(cuda_policy.get("required_jobs"), "CUDA evidence required jobs"),
        "CUDA evidence required jobs",
    )
    required_runner_labels = _cuda_required_runner_labels(cuda_policy, required_jobs)
    accepted_jobs: list[Mapping[str, Any]] = []
    accepted_runner_ids: dict[int, str] = {}
    accepted_runner_names: dict[str, str] = {}
    for job_name in required_jobs:
        matches = [job for job in jobs if job.get("name") == job_name]
        if len(matches) != 1:
            raise ValueError(
                f"CUDA evidence must contain exactly one all-attempt {job_name!r} job; "
                f"got {len(matches)}"
            )
        job = matches[0]
        _require_positive_integer(job.get("id"), f"CUDA evidence job {job_name!r} id")
        job_run_id = _require_positive_integer(
            job.get("run_id"), f"CUDA evidence job {job_name!r} run id"
        )
        if job_run_id != source_run_id:
            raise ValueError(
                f"CUDA evidence job {job_name!r} run id mismatch: "
                f"expected {source_run_id!r}, got {job_run_id!r}"
            )
        _require_first_attempt(
            job.get("run_attempt"), f"CUDA evidence job {job_name!r} run attempt"
        )
        if job.get("status") != "completed" or job.get("conclusion") != "success":
            raise ValueError(
                f"CUDA evidence job {job_name!r} did not complete successfully: "
                f"{job.get('status')!r}/{job.get('conclusion')!r}"
            )
        job_head_sha = job.get("head_sha")
        if job_head_sha != release_commit:
            raise ValueError(
                f"CUDA evidence job {job_name!r} head SHA mismatch: "
                f"expected {release_commit!r}, got {job_head_sha!r}"
            )
        runner_id = _require_positive_integer(
            job.get("runner_id"), f"CUDA evidence job {job_name!r} runner id"
        )
        previous_job = accepted_runner_ids.get(runner_id)
        if previous_job is not None:
            raise ValueError(
                f"CUDA evidence runner id {runner_id!r} is reused by required jobs "
                f"{previous_job!r} and {job_name!r}; each one-job JIT runner may execute "
                "at most one required job"
            )
        accepted_runner_ids[runner_id] = job_name
        runner_name = job.get("runner_name")
        if not isinstance(runner_name, str) or not runner_name.strip():
            raise ValueError(
                f"CUDA evidence job {job_name!r} runner name must be a non-empty string; "
                f"got {runner_name!r}"
            )
        required_runner_label = required_runner_labels[job_name]
        expected_runner_name = re.compile(rf"{re.escape(required_runner_label)}-jit-[a-f0-9]{{16}}")
        if expected_runner_name.fullmatch(runner_name) is None:
            raise ValueError(
                f"CUDA evidence job {job_name!r} runner name must identify the exact "
                f"reviewed one-job JIT route {required_runner_label!r}; got {runner_name!r}"
            )
        previous_name_job = accepted_runner_names.get(runner_name)
        if previous_name_job is not None:
            raise ValueError(
                f"CUDA evidence runner name {runner_name!r} is reused by required jobs "
                f"{previous_name_job!r} and {job_name!r}; every one-job JIT runner must "
                "have a distinct generated name"
            )
        accepted_runner_names[runner_name] = job_name
        runner_group_id = _require_positive_integer(
            job.get("runner_group_id"),
            f"CUDA evidence job {job_name!r} runner group id",
        )
        if runner_group_id != 1:
            raise ValueError(
                f"CUDA evidence job {job_name!r} runner group id must be the reviewed "
                f"default group 1; got {runner_group_id!r}"
            )
        runner_group_name = job.get("runner_group_name")
        if runner_group_name != "Default":
            raise ValueError(
                f"CUDA evidence job {job_name!r} runner group name must be the reviewed "
                f"default group 'Default'; got {runner_group_name!r}"
            )
        labels = _canonical_names(
            _sequence(job.get("labels"), f"CUDA evidence job {job_name!r} labels"),
            f"CUDA evidence job {job_name!r} labels",
        )
        if labels != [runner_name]:
            raise ValueError(
                f"CUDA evidence job {job_name!r} labels must be exactly the one-use "
                f"runner name {runner_name!r}; got {labels!r}"
            )
        accepted_job = {
            field: job.get(field)
            for field in (
                "id",
                "run_id",
                "name",
                "status",
                "conclusion",
                "run_attempt",
                "head_sha",
                "runner_id",
                "runner_name",
                "runner_group_id",
                "runner_group_name",
            )
        }
        accepted_job["labels"] = labels
        accepted_jobs.append(accepted_job)

    return {
        "schema_version": 1,
        "query_filter": "all",
        "pagination_complete": True,
        "run": {
            field: run.get(field)
            for field in (
                "id",
                "path",
                "event",
                "head_branch",
                "head_sha",
                "status",
                "conclusion",
                "run_attempt",
                "created_at",
                "updated_at",
            )
        },
        "jobs": accepted_jobs,
    }


def verify_preflight_source_run(
    run: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    *,
    run_id: str,
    repository: str,
    release_commit: str,
) -> None:
    """Bind a downloaded snapshot to its successful pre-tag Actions run."""
    actual_repository = _mapping(run.get("repository"), "preflight run repository").get("full_name")
    expected_fields = {
        "id": (str(run.get("id")), str(run_id)),
        "repository": (actual_repository, repository),
        "workflow path": (run.get("path"), ".github/workflows/release-preflight.yml"),
        "event": (run.get("event"), "workflow_dispatch"),
        "head branch": (run.get("head_branch"), "main"),
        "head SHA": (run.get("head_sha"), release_commit),
        "status": (run.get("status"), "completed"),
        "conclusion": (run.get("conclusion"), "success"),
    }
    for label, (actual, expected) in expected_fields.items():
        if actual != expected:
            raise ValueError(
                f"preflight source run {label} mismatch: expected {expected!r}, got {actual!r}"
            )
    workflow_run = _mapping(snapshot.get("workflow_run"), "snapshot workflow_run")
    run_attempt = _validated_run_id(
        workflow_run.get("run_attempt"), "snapshot workflow run attempt"
    )
    capture_actor = workflow_run.get("actor")
    triggering_actor = workflow_run.get("triggering_actor")
    capture_principal = _mapping(snapshot.get("observation"), "snapshot observation").get(
        "capture_principal"
    )
    if capture_actor != capture_principal or triggering_actor != capture_principal:
        raise ValueError(
            "snapshot workflow actor and triggering actor must both match the authenticated "
            "capture principal"
        )
    api_actor = _mapping(run.get("actor"), "preflight source run actor").get("login")
    api_triggering_actor = _mapping(
        run.get("triggering_actor"), "preflight source run triggering_actor"
    ).get("login")
    snapshot_fields = {
        "id": (str(workflow_run.get("id")), str(run_id)),
        "ref": (workflow_run.get("ref"), "refs/heads/main"),
        "sha": (workflow_run.get("sha"), release_commit),
        "run attempt": (str(run.get("run_attempt")), run_attempt),
        "actor": (api_actor, capture_actor),
        "triggering actor": (api_triggering_actor, triggering_actor),
    }
    for snapshot_label, (actual_value, expected_value) in snapshot_fields.items():
        if actual_value != expected_value:
            raise ValueError(
                f"snapshot workflow run {snapshot_label} mismatch: expected "
                f"{expected_value!r}, got {actual_value!r}"
            )


def bind_snapshot_to_workflow(
    *,
    policy: Mapping[str, Any],
    policy_sha256: str,
    snapshot: Mapping[str, Any],
    repository: str,
    release_tag: str,
    release_commit: str,
    workflow_run: Mapping[str, Any],
    cuda_run: Mapping[str, Any],
    cuda_jobs: Mapping[str, Any],
    cuda_run_id: str,
    now: datetime | None = None,
    max_age: timedelta = _MAX_SNAPSHOT_AGE,
) -> Mapping[str, Any]:
    """Accept a fresh admin capture and bind it to an auditable Actions run."""
    verify_snapshot(
        policy=policy,
        policy_sha256=policy_sha256,
        snapshot=snapshot,
        repository=repository,
        release_tag=release_tag,
        release_commit=release_commit,
        now=now,
        max_age=max_age,
    )
    observation = _mapping(snapshot.get("observation"), "snapshot observation")
    capture_principal = observation.get("capture_principal")
    actor = workflow_run.get("actor")
    triggering_actor = workflow_run.get("triggering_actor")
    if actor != capture_principal or triggering_actor != capture_principal:
        raise ValueError(
            "preflight dispatch actor and triggering actor must both match the authenticated "
            "admin capture principal: "
            f"actor={actor!r}, triggering_actor={triggering_actor!r}, "
            f"capture_principal={capture_principal!r}"
        )
    _validated_run_id(workflow_run.get("run_attempt"), "preflight workflow run attempt")
    bound = dict(snapshot)
    bound["workflow_run"] = dict(workflow_run)
    bound["cuda_evidence"] = verify_cuda_evidence(
        policy,
        cuda_run,
        cuda_jobs,
        run_id=cuda_run_id,
        repository=repository,
        release_commit=release_commit,
    )
    return bound


def verify_bound_cuda_evidence(
    *,
    policy: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    cuda_run: Mapping[str, Any],
    cuda_jobs: Mapping[str, Any],
    cuda_run_id: str,
    repository: str,
    release_commit: str,
) -> None:
    """Re-query and require byte-equivalent normalized CUDA evidence at publish time."""
    embedded = _mapping(snapshot.get("cuda_evidence"), "snapshot CUDA evidence")
    live = verify_cuda_evidence(
        policy,
        cuda_run,
        cuda_jobs,
        run_id=cuda_run_id,
        repository=repository,
        release_commit=release_commit,
    )
    if dict(embedded) != dict(live):
        raise ValueError("live CUDA evidence differs from the attested preflight evidence")


def _validated_release_values(tag: str, commit: str) -> tuple[str, str]:
    normalized_commit = commit.strip().lower()
    if _TAG.fullmatch(tag) is None:
        raise ValueError("release tag must have the form vMAJOR.MINOR.PATCH")
    if _SHA.fullmatch(normalized_commit) is None:
        raise ValueError("release commit must be a complete 40-character lowercase SHA")
    return tag, normalized_commit


def _validated_run_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[1-9][0-9]*", value) is None:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _current_workflow_run() -> Mapping[str, Any]:
    return {
        "id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "ref": os.environ.get("GITHUB_REF"),
        "sha": os.environ.get("GITHUB_SHA"),
        "actor": os.environ.get("GITHUB_ACTOR"),
        "triggering_actor": os.environ.get("GITHUB_TRIGGERING_ACTOR"),
        "workflow": os.environ.get("GITHUB_WORKFLOW"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--policy", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--repository", required=True)
    capture.add_argument("--tag", required=True)
    capture.add_argument("--commit", required=True)
    capture.add_argument("--installed-app-authority", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--policy", type=Path, required=True)
    verify.add_argument("--snapshot", type=Path, required=True)
    verify.add_argument("--repository", required=True)
    verify.add_argument("--tag", required=True)
    verify.add_argument("--commit", required=True)
    verify.add_argument("--run-json", type=Path)
    verify.add_argument("--run-id")
    verify.add_argument("--cuda-run-json", type=Path, required=True)
    verify.add_argument("--cuda-jobs-json", type=Path, required=True)
    verify.add_argument("--cuda-run-id", required=True)
    bind = subparsers.add_parser("bind")
    bind.add_argument("--policy", type=Path, required=True)
    bind.add_argument("--snapshot", type=Path, required=True)
    bind.add_argument("--output", type=Path, required=True)
    bind.add_argument("--repository", required=True)
    bind.add_argument("--tag", required=True)
    bind.add_argument("--commit", required=True)
    bind.add_argument("--cuda-run-json", type=Path, required=True)
    bind.add_argument("--cuda-jobs-json", type=Path, required=True)
    bind.add_argument("--cuda-run-id", required=True)
    restoration = subparsers.add_parser("verify-app-restoration")
    restoration.add_argument("--before", type=Path, required=True)
    restoration.add_argument("--restored", type=Path, required=True)
    restoration.add_argument("--output", type=Path, required=True)
    restoration.add_argument("--repository", required=True)
    restoration.add_argument("--capture-principal", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "verify-app-restoration":
            before = _mapping(
                json.loads(args.before.read_text(encoding="utf-8")),
                "pre-window installed App capture",
            )
            restored = _mapping(
                json.loads(args.restored.read_text(encoding="utf-8")),
                "restored installed App capture",
            )
            _reject_output_aliases(
                args.output,
                [
                    *_manifest_input_paths(args.before, before),
                    *_manifest_input_paths(args.restored, restored),
                ],
            )
            report = verify_installed_app_restoration(
                before=before,
                restored=restored,
                repository=args.repository,
                capture_principal=args.capture_principal,
                before_evidence_reader=lambda filename: _read_installed_app_evidence_file(
                    args.before.parent, filename
                ),
                restored_evidence_reader=lambda filename: _read_installed_app_evidence_file(
                    args.restored.parent, filename
                ),
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
            args.output.write_text(encoded, encoding="utf-8")
            args.output.with_suffix(args.output.suffix + ".sha256").write_text(
                f"{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}  {args.output.name}\n",
                encoding="utf-8",
            )
            return 0
        tag, commit = _validated_release_values(args.tag, args.commit)
        policy, policy_sha256 = load_policy(args.policy)
        if policy.get("repository") != args.repository:
            raise ValueError(
                f"workflow repository {args.repository!r} does not match reviewed policy "
                f"{policy.get('repository')!r}"
            )
        if args.command in {"verify", "bind"}:
            snapshot = _mapping(json.loads(args.snapshot.read_text(encoding="utf-8")), "snapshot")
            cuda_run_id = _validated_run_id(args.cuda_run_id, "cuda-run-id")
            cuda_run = _mapping(
                json.loads(args.cuda_run_json.read_text(encoding="utf-8")),
                "CUDA run JSON",
            )
            cuda_jobs = _mapping(
                json.loads(args.cuda_jobs_json.read_text(encoding="utf-8")),
                "CUDA jobs JSON",
            )
        if args.command == "bind":
            bound = bind_snapshot_to_workflow(
                policy=policy,
                policy_sha256=policy_sha256,
                snapshot=snapshot,
                repository=args.repository,
                release_tag=tag,
                release_commit=commit,
                workflow_run=_current_workflow_run(),
                cuda_run=cuda_run,
                cuda_jobs=cuda_jobs,
                cuda_run_id=cuda_run_id,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            encoded = json.dumps(bound, indent=2, sort_keys=True) + "\n"
            args.output.write_text(encoded, encoding="utf-8")
            args.output.with_suffix(args.output.suffix + ".sha256").write_text(
                f"{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}  {args.output.name}\n",
                encoding="utf-8",
            )
            return 0
        if args.command == "verify":
            verify_snapshot(
                policy=policy,
                policy_sha256=policy_sha256,
                snapshot=snapshot,
                repository=args.repository,
                release_tag=tag,
                release_commit=commit,
            )
            if (args.run_json is None) != (args.run_id is None):
                raise ValueError("--run-json and --run-id must be supplied together")
            if args.run_json is not None:
                run_id = _validated_run_id(args.run_id, "run-id")
                run = _mapping(
                    json.loads(args.run_json.read_text(encoding="utf-8")),
                    "preflight run JSON",
                )
                verify_preflight_source_run(
                    run,
                    snapshot,
                    run_id=run_id,
                    repository=args.repository,
                    release_commit=commit,
                )
            verify_bound_cuda_evidence(
                policy=policy,
                snapshot=snapshot,
                cuda_run=cuda_run,
                cuda_jobs=cuda_jobs,
                cuda_run_id=cuda_run_id,
                repository=args.repository,
                release_commit=commit,
            )
            return 0

        token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or ""
        api = GitHubApi(
            token=token, api_url=os.environ.get("GITHUB_API_URL", "https://api.github.com")
        )
        installed_app_authority = _mapping(
            json.loads(args.installed_app_authority.read_text(encoding="utf-8")),
            "installed App authority capture",
        )
        _reject_output_aliases(
            args.output,
            _manifest_input_paths(args.installed_app_authority, installed_app_authority),
        )
        installed_app_evidence_directory = args.installed_app_authority.parent
        observation = capture_observation(
            policy=policy,
            release_tag=tag,
            release_commit=commit,
            get_json=api.get,
            installed_app_authority=installed_app_authority,
            installed_app_evidence_reader=lambda filename: _read_installed_app_evidence_file(
                installed_app_evidence_directory, filename
            ),
        )
        snapshot = make_snapshot(
            policy=policy,
            policy_sha256=policy_sha256,
            observation=observation,
            workflow_run=_current_workflow_run(),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(snapshot, indent=2, sort_keys=True) + "\n"
        args.output.write_text(encoded, encoding="utf-8")
        args.output.with_suffix(args.output.suffix + ".sha256").write_text(
            f"{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}  {args.output.name}\n",
            encoding="utf-8",
        )
        if snapshot["violations"]:
            for violation in snapshot["violations"]:
                print(f"policy violation: {violation}", file=sys.stderr)
            return 1
        return 0
    except (KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
