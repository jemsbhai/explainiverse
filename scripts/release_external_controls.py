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
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

_SHA = re.compile(r"[0-9a-f]{40}")
_TAG = re.compile(r"v\d+\.\d+\.\d+")
_MAX_SNAPSHOT_AGE = timedelta(minutes=30)
_CUDA_EXCEPTION_ID = "EXPLAINIVERSE-v0.15.0-CPU-ONLY"
_CUDA_EXCEPTION_TAG = "v0.15.0"
_CUDA_EXCEPTION_VERSION = "0.15.0"
_CUDA_EXCEPTION_PULL_REQUEST = 5
_CUDA_EXCEPTION_APPROVED_AT = "2026-09-03"
_CUDA_EXCEPTION_OMITTED_CHECKS = (
    "CUDA single-GPU (Torch latest)",
    "CUDA single-GPU (Torch minimum)",
)
_CUDA_EXCEPTION_OMITTED_JOBS = (
    "CUDA single-GPU (Torch latest)",
    "CUDA single-GPU (Torch minimum)",
    "CUDA two-GPU scheduled (Torch latest)",
    "CUDA two-GPU scheduled (Torch minimum)",
)


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


def _validated_cuda_release_exception(policy: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the single reviewed exception after rejecting any scope drift."""
    exception = _mapping(policy.get("cuda_release_exception"), "CUDA release exception policy")
    expected_keys = {
        "id",
        "release_tag",
        "package_version",
        "merge_pull_request",
        "omitted_required_checks",
        "omitted_cuda_jobs",
        "hardware_evidence_collected",
        "cuda_release_verified",
        "authorized_by",
        "approved_at",
        "reason",
        "disclosure",
    }
    if set(exception) != expected_keys:
        raise ValueError(
            "CUDA release exception policy fields differ from the reviewed schema: "
            f"expected {sorted(expected_keys)!r}, got {sorted(exception, key=str)!r}"
        )

    exact_fields = {
        "id": _CUDA_EXCEPTION_ID,
        "release_tag": _CUDA_EXCEPTION_TAG,
        "package_version": _CUDA_EXCEPTION_VERSION,
        "merge_pull_request": _CUDA_EXCEPTION_PULL_REQUEST,
        "approved_at": _CUDA_EXCEPTION_APPROVED_AT,
        "hardware_evidence_collected": False,
        "cuda_release_verified": False,
    }
    for field, expected in exact_fields.items():
        if exception.get(field) != expected:
            raise ValueError(
                f"CUDA release exception {field} must be exactly {expected!r}, "
                f"got {exception.get(field)!r}"
            )

    omitted_checks = _canonical_names(
        _sequence(
            exception.get("omitted_required_checks"),
            "CUDA release exception omitted required checks",
        ),
        "CUDA release exception omitted required checks",
    )
    if omitted_checks != sorted(_CUDA_EXCEPTION_OMITTED_CHECKS):
        raise ValueError(
            "CUDA release exception may omit exactly the two reviewed CUDA check contexts"
        )
    baseline_checks = _canonical_names(
        _sequence(policy.get("required_checks"), "required checks policy"),
        "required checks policy",
    )
    if not set(omitted_checks) <= set(baseline_checks):
        raise ValueError("CUDA release exception checks are absent from required_checks")

    omitted_jobs = _canonical_names(
        _sequence(
            exception.get("omitted_cuda_jobs"),
            "CUDA release exception omitted CUDA jobs",
        ),
        "CUDA release exception omitted CUDA jobs",
    )
    if omitted_jobs != sorted(_CUDA_EXCEPTION_OMITTED_JOBS):
        raise ValueError("CUDA release exception must omit exactly the four reviewed CUDA jobs")
    cuda_policy = _mapping(policy.get("cuda_evidence"), "CUDA evidence policy")
    required_jobs = _canonical_names(
        _sequence(cuda_policy.get("required_jobs"), "CUDA evidence required jobs"),
        "CUDA evidence required jobs",
    )
    if omitted_jobs != required_jobs:
        raise ValueError(
            "CUDA release exception omitted jobs must exactly match CUDA evidence required jobs"
        )

    authorized_by = _canonical_names(
        _sequence(exception.get("authorized_by"), "CUDA release exception authorized_by"),
        "CUDA release exception authorized_by",
    )
    principals = _canonical_names(
        _sequence(policy.get("admin_snapshot_principals"), "admin_snapshot_principals"),
        "admin_snapshot_principals",
    )
    if authorized_by != principals or authorized_by != ["jemsbhai"]:
        raise ValueError(
            "CUDA release exception authorization must exactly match the reviewed principal"
        )

    reason = exception.get("reason")
    disclosure = exception.get("disclosure")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("CUDA release exception reason must be a non-empty string")
    if not isinstance(disclosure, str) or not disclosure.strip():
        raise ValueError("CUDA release exception disclosure must be a non-empty string")
    required_disclosure_fragments = (
        "0.15.0",
        "CPU-verified",
        "CUDA hardware validation was not performed",
        "no CUDA release-verification claim",
    )
    if any(fragment not in disclosure for fragment in required_disclosure_fragments):
        raise ValueError(
            "CUDA release exception disclosure must retain the reviewed CPU-only warning"
        )

    normalized = dict(exception)
    normalized["omitted_required_checks"] = omitted_checks
    normalized["omitted_cuda_jobs"] = omitted_jobs
    normalized["authorized_by"] = authorized_by
    return normalized


def _resolve_cuda_release_exception(
    policy: Mapping[str, Any], *, release_tag: Any, requested_exception_id: Any
) -> Mapping[str, Any] | None:
    """Resolve only an explicit request for the one exact, reviewed exception."""
    exception = _validated_cuda_release_exception(policy)
    if requested_exception_id in (None, ""):
        return None
    if not isinstance(requested_exception_id, str):
        raise ValueError("CUDA release exception id must be a string")
    if requested_exception_id != exception["id"]:
        raise ValueError(
            f"CUDA release exception id must be exactly {exception['id']!r}, "
            f"got {requested_exception_id!r}"
        )
    if release_tag != exception["release_tag"]:
        raise ValueError(
            "CUDA release exception is restricted to "
            f"{exception['release_tag']!r}, got {release_tag!r}"
        )
    return exception


def load_policy(path: Path) -> tuple[Mapping[str, Any], str]:
    """Load a policy and return it with its exact-file SHA-256 digest."""
    raw = path.read_bytes()
    policy = _mapping(json.loads(raw), "release control policy")
    if policy.get("schema_version") != 1:
        raise ValueError("release control policy schema_version must be 1")
    _validated_cuda_release_exception(policy)
    return policy, hashlib.sha256(raw).hexdigest()


def evaluate_controls(policy: Mapping[str, Any], observation: Mapping[str, Any]) -> list[str]:
    """Return every policy difference in deterministic order."""
    violations: list[str] = []
    selected_exception = _resolve_cuda_release_exception(
        policy,
        release_tag=observation.get("release_tag"),
        requested_exception_id=observation.get("cuda_exception_id"),
    )
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
    expected_principals = _canonical_names(
        _sequence(policy.get("admin_snapshot_principals"), "admin_snapshot_principals"),
        "admin_snapshot_principals",
    )
    if observation.get("capture_principal") not in expected_principals:
        violations.append(
            f"capture_principal: expected one of {expected_principals!r}, "
            f"got {observation.get('capture_principal')!r}"
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

    baseline_checks = _canonical_names(
        _sequence(policy.get("required_checks"), "required_checks"), "required_checks"
    )
    omitted_checks = (
        set(selected_exception["omitted_required_checks"]) if selected_exception else set()
    )
    expected_checks = [name for name in baseline_checks if name not in omitted_checks]
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
    if set(workflow_policy) != set(baseline_checks):
        violations.append(
            "required_check_workflows: keys must exactly match required_checks; "
            f"expected {sorted(baseline_checks)!r}, got {sorted(workflow_policy)!r}"
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
    cuda_exception_id: str | None = None,
) -> Mapping[str, Any]:
    """Capture all policy-controlled GitHub state through an injected client."""
    _resolve_cuda_release_exception(
        policy,
        release_tag=release_tag,
        requested_exception_id=cuda_exception_id,
    )
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

    raw_branch = _mapping(
        get_json(f"{root}/branches/{branch}/protection"), "branch protection response"
    )
    raw_environment = _mapping(
        get_json(f"{root}/environments/{environment_name}"), "environment response"
    )
    principal = _mapping(get_json("user"), "authenticated GitHub user").get("login")
    if not isinstance(principal, str) or not principal:
        raise ValueError("authenticated GitHub user response has no login")

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
        "release_tag": release_tag,
        "release_commit": release_commit,
        "cuda_exception_id": cuda_exception_id,
        "tag_exists": tag_exists,
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
) -> Mapping[str, Any]:
    selected_exception = _resolve_cuda_release_exception(
        policy,
        release_tag=observation.get("release_tag"),
        requested_exception_id=observation.get("cuda_exception_id"),
    )
    violations = evaluate_controls(policy, observation)
    return {
        "schema_version": 1,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "policy_sha256": policy_sha256,
        "workflow_run": dict(workflow_run),
        "observation": dict(observation),
        "repository_controls_accepted": not violations,
        "violations": violations,
        "cuda_release_exception": (
            dict(selected_exception) if selected_exception is not None else None
        ),
        "pypi_trusted_publisher": {
            "expected": dict(
                _mapping(policy.get("pypi_trusted_publisher"), "trusted publisher policy")
            ),
            "verification_status": "blocked_no_public_read_api",
        },
    }


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
    if age < timedelta(minutes=-1):
        raise ValueError("external-control snapshot observed_at is in the future")
    if age > max_age:
        raise ValueError(f"external-control snapshot is stale ({age}); recapture within {max_age}")


def verify_snapshot(
    *,
    policy: Mapping[str, Any],
    policy_sha256: str,
    snapshot: Mapping[str, Any],
    repository: str,
    release_tag: str,
    release_commit: str,
    cuda_exception_id: str | None = None,
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
    observed_exception_id = observation.get("cuda_exception_id")
    if observed_exception_id != cuda_exception_id:
        raise ValueError(
            "external-control snapshot CUDA exception id mismatch: expected "
            f"{cuda_exception_id!r}, got {observed_exception_id!r}"
        )
    selected_exception = _resolve_cuda_release_exception(
        policy,
        release_tag=release_tag,
        requested_exception_id=cuda_exception_id,
    )
    expected_exception = dict(selected_exception) if selected_exception is not None else None
    if snapshot.get("cuda_release_exception") != expected_exception:
        raise ValueError("external-control snapshot CUDA release exception differs from policy")
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
    expected_fields = {
        "id": (str(run.get("id")), str(run_id)),
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

    jobs = _complete_jobs_response(jobs_response, "CUDA evidence")
    required_jobs = _canonical_names(
        _sequence(cuda_policy.get("required_jobs"), "CUDA evidence required jobs"),
        "CUDA evidence required jobs",
    )
    required_runner_labels = _cuda_required_runner_labels(cuda_policy, required_jobs)
    accepted_jobs: list[Mapping[str, Any]] = []
    for job_name in required_jobs:
        matches = [job for job in jobs if job.get("name") == job_name]
        if len(matches) != 1:
            raise ValueError(
                f"CUDA evidence must contain exactly one all-attempt {job_name!r} job; "
                f"got {len(matches)}"
            )
        job = matches[0]
        if job.get("status") != "completed" or job.get("conclusion") != "success":
            raise ValueError(
                f"CUDA evidence job {job_name!r} did not complete successfully: "
                f"{job.get('status')!r}/{job.get('conclusion')!r}"
            )
        job_head_sha = job.get("head_sha")
        if job_head_sha is not None and job_head_sha != release_commit:
            raise ValueError(
                f"CUDA evidence job {job_name!r} head SHA mismatch: "
                f"expected {release_commit!r}, got {job_head_sha!r}"
            )
        labels = _canonical_names(
            _sequence(job.get("labels"), f"CUDA evidence job {job_name!r} labels"),
            f"CUDA evidence job {job_name!r} labels",
        )
        required_runner_label = required_runner_labels[job_name]
        if required_runner_label not in labels:
            raise ValueError(
                f"CUDA evidence job {job_name!r} labels must include expected custom "
                f"runner label {required_runner_label!r}; got {labels!r}"
            )
        accepted_job = {
            field: job.get(field)
            for field in (
                "id",
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


def resolve_cuda_release_gate(
    *,
    policy: Mapping[str, Any],
    release_tag: str,
    release_commit: str,
    cuda_run: Mapping[str, Any] | None,
    cuda_jobs: Mapping[str, Any] | None,
    cuda_run_id: str | None,
    cuda_exception_id: str | None,
    repository: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any] | None]:
    """Resolve the normal hardware path or the one exact CPU-only exception."""
    selected_exception = _resolve_cuda_release_exception(
        policy,
        release_tag=release_tag,
        requested_exception_id=cuda_exception_id,
    )
    cuda_values = (cuda_run, cuda_jobs, cuda_run_id)
    supplied_cuda_values = tuple(value not in (None, "") for value in cuda_values)

    if selected_exception is not None:
        if any(supplied_cuda_values):
            raise ValueError(
                "CUDA hardware evidence inputs must be absent when the CPU-only exception is used"
            )
        gate = {
            "schema_version": 1,
            "mode": "cpu_only_exception",
            "status": "not_run",
            "exception_id": selected_exception["id"],
            "release_tag": release_tag,
            "release_commit": release_commit,
            "package_version": selected_exception["package_version"],
            "merge_pull_request": selected_exception["merge_pull_request"],
            "hardware_evidence_collected": False,
            "cuda_release_verified": False,
            "omitted_required_checks": list(selected_exception["omitted_required_checks"]),
            "omitted_cuda_jobs": list(selected_exception["omitted_cuda_jobs"]),
            "authorized_by": list(selected_exception["authorized_by"]),
            "approved_at": selected_exception["approved_at"],
            "reason": selected_exception["reason"],
            "disclosure": selected_exception["disclosure"],
        }
        return gate, None

    if not all(supplied_cuda_values):
        raise ValueError(
            "CUDA hardware mode requires --cuda-run-id, --cuda-run-json, and "
            "--cuda-jobs-json together"
        )
    assert cuda_run is not None
    assert cuda_jobs is not None
    assert cuda_run_id is not None
    validated_run_id = _validated_run_id(cuda_run_id, "cuda-run-id")
    evidence = verify_cuda_evidence(
        policy,
        cuda_run,
        cuda_jobs,
        run_id=validated_run_id,
        repository=repository,
        release_commit=release_commit,
    )
    gate = {
        "schema_version": 1,
        "mode": "hardware_evidence",
        "status": "verified",
        "exception_id": None,
        "release_tag": release_tag,
        "release_commit": release_commit,
        "hardware_evidence_collected": True,
        "cuda_release_verified": True,
        "cuda_run_id": validated_run_id,
    }
    return gate, evidence


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
    cuda_run: Mapping[str, Any] | None = None,
    cuda_jobs: Mapping[str, Any] | None = None,
    cuda_run_id: str | None = None,
    cuda_exception_id: str | None = None,
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
        cuda_exception_id=cuda_exception_id,
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
    gate, evidence = resolve_cuda_release_gate(
        policy=policy,
        release_tag=release_tag,
        release_commit=release_commit,
        cuda_run=cuda_run,
        cuda_jobs=cuda_jobs,
        cuda_run_id=cuda_run_id,
        cuda_exception_id=cuda_exception_id,
        repository=repository,
    )
    bound["cuda_release_gate"] = gate
    if evidence is None:
        bound.pop("cuda_evidence", None)
    else:
        bound["cuda_evidence"] = evidence
    return bound


def verify_bound_cuda_release_gate(
    *,
    policy: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    release_tag: str,
    release_commit: str,
    repository: str,
    cuda_run: Mapping[str, Any] | None = None,
    cuda_jobs: Mapping[str, Any] | None = None,
    cuda_run_id: str | None = None,
    cuda_exception_id: str | None = None,
) -> Mapping[str, Any]:
    """Re-resolve and exactly match the bound release-gate decision."""
    expected_gate, live_evidence = resolve_cuda_release_gate(
        policy=policy,
        release_tag=release_tag,
        release_commit=release_commit,
        cuda_run=cuda_run,
        cuda_jobs=cuda_jobs,
        cuda_run_id=cuda_run_id,
        cuda_exception_id=cuda_exception_id,
        repository=repository,
    )
    embedded_gate = _mapping(snapshot.get("cuda_release_gate"), "snapshot CUDA release gate")
    if dict(embedded_gate) != dict(expected_gate):
        raise ValueError("live CUDA release gate differs from the attested preflight gate")
    if live_evidence is None:
        if "cuda_evidence" in snapshot:
            raise ValueError("CPU-only exception snapshot must not contain CUDA hardware evidence")
    else:
        embedded_evidence = _mapping(snapshot.get("cuda_evidence"), "snapshot CUDA evidence")
        if dict(embedded_evidence) != dict(live_evidence):
            raise ValueError("live CUDA evidence differs from the attested preflight evidence")
    return embedded_gate


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
    verify_bound_cuda_release_gate(
        policy=policy,
        snapshot=snapshot,
        release_tag=str(
            _mapping(snapshot.get("observation"), "snapshot observation").get("release_tag")
        ),
        release_commit=release_commit,
        repository=repository,
        cuda_run=cuda_run,
        cuda_jobs=cuda_jobs,
        cuda_run_id=cuda_run_id,
        cuda_exception_id=None,
    )


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


def _write_json_with_sha256(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.write_text(encoded, encoding="utf-8")
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}  {path.name}\n",
        encoding="utf-8",
    )


def _write_github_outputs(path: Path | None, gate: Mapping[str, Any]) -> None:
    if path is None:
        return
    mode = gate.get("mode")
    if mode not in {"hardware_evidence", "cpu_only_exception"}:
        raise ValueError(f"cannot emit unsupported CUDA release mode {mode!r}")
    run_id = gate.get("cuda_run_id", "")
    exception_id = gate.get("exception_id") or ""
    values = {
        "cuda_mode": mode,
        "cuda_run_id": run_id,
        "cuda_exception_id": exception_id,
    }
    if any(
        not isinstance(value, str) or "\n" in value or "\r" in value for value in values.values()
    ):
        raise ValueError("CUDA release GitHub output values must be single-line strings")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        for name, value in values.items():
            stream.write(f"{name}={value}\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--policy", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--repository", required=True)
    capture.add_argument("--tag", required=True)
    capture.add_argument("--commit", required=True)
    capture.add_argument("--cuda-exception-id")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--policy", type=Path, required=True)
    verify.add_argument("--snapshot", type=Path, required=True)
    verify.add_argument("--repository", required=True)
    verify.add_argument("--tag", required=True)
    verify.add_argument("--commit", required=True)
    verify.add_argument("--run-json", type=Path)
    verify.add_argument("--run-id")
    verify.add_argument("--cuda-run-json", type=Path)
    verify.add_argument("--cuda-jobs-json", type=Path)
    verify.add_argument("--cuda-run-id")
    verify.add_argument("--cuda-exception-id")
    verify.add_argument("--github-output", type=Path)
    bind = subparsers.add_parser("bind")
    bind.add_argument("--policy", type=Path, required=True)
    bind.add_argument("--snapshot", type=Path, required=True)
    bind.add_argument("--output", type=Path, required=True)
    bind.add_argument("--repository", required=True)
    bind.add_argument("--tag", required=True)
    bind.add_argument("--commit", required=True)
    bind.add_argument("--cuda-run-json", type=Path)
    bind.add_argument("--cuda-jobs-json", type=Path)
    bind.add_argument("--cuda-run-id")
    bind.add_argument("--cuda-exception-id")
    bind.add_argument("--github-output", type=Path)
    bind.add_argument("--cuda-gate-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        tag, commit = _validated_release_values(args.tag, args.commit)
        policy, policy_sha256 = load_policy(args.policy)
        if policy.get("repository") != args.repository:
            raise ValueError(
                f"workflow repository {args.repository!r} does not match reviewed policy "
                f"{policy.get('repository')!r}"
            )
        if args.command in {"verify", "bind"}:
            snapshot = _mapping(json.loads(args.snapshot.read_text(encoding="utf-8")), "snapshot")
            cuda_run_id = args.cuda_run_id or None
            cuda_run = (
                _mapping(
                    json.loads(args.cuda_run_json.read_text(encoding="utf-8")),
                    "CUDA run JSON",
                )
                if args.cuda_run_json is not None
                else None
            )
            cuda_jobs = (
                _mapping(
                    json.loads(args.cuda_jobs_json.read_text(encoding="utf-8")),
                    "CUDA jobs JSON",
                )
                if args.cuda_jobs_json is not None
                else None
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
                cuda_exception_id=args.cuda_exception_id,
            )
            _write_json_with_sha256(args.output, bound)
            gate = _mapping(bound.get("cuda_release_gate"), "bound CUDA release gate")
            if args.cuda_gate_output is not None:
                _write_json_with_sha256(args.cuda_gate_output, gate)
            _write_github_outputs(args.github_output, gate)
            return 0
        if args.command == "verify":
            verify_snapshot(
                policy=policy,
                policy_sha256=policy_sha256,
                snapshot=snapshot,
                repository=args.repository,
                release_tag=tag,
                release_commit=commit,
                cuda_exception_id=args.cuda_exception_id,
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
            gate = verify_bound_cuda_release_gate(
                policy=policy,
                snapshot=snapshot,
                release_tag=tag,
                release_commit=commit,
                repository=args.repository,
                cuda_run=cuda_run,
                cuda_jobs=cuda_jobs,
                cuda_run_id=cuda_run_id,
                cuda_exception_id=args.cuda_exception_id,
            )
            _write_github_outputs(args.github_output, gate)
            return 0

        token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or ""
        api = GitHubApi(
            token=token, api_url=os.environ.get("GITHUB_API_URL", "https://api.github.com")
        )
        observation = capture_observation(
            policy=policy,
            release_tag=tag,
            release_commit=commit,
            get_json=api.get,
            cuda_exception_id=args.cuda_exception_id,
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
