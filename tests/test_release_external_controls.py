"""Security and replay tests for the stable-release control snapshot."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "release_external_controls.py"
SPEC = importlib.util.spec_from_file_location("release_external_controls", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
controls = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = controls
SPEC.loader.exec_module(controls)

POLICY_PATH = ROOT / ".github" / "release-control-policy.json"
SHA = "a" * 40


def _policy():
    return controls.load_policy(POLICY_PATH)


def _matching_installed_app_authority(*, captured_at=None, evidence_prefix="capture"):
    policy, _ = _policy()
    installed_apps = policy["release_runner_authority"]["installed_apps"]
    capture_time = captured_at or datetime.now(timezone.utc).isoformat()
    installations = sorted(
        copy.deepcopy(installed_apps["expected_installations"]), key=lambda value: value["id"]
    )
    evidence_roles = [("installation-list", None)]
    evidence_roles.extend(("installation-configure", value["id"]) for value in installations)
    evidence_roles.extend(
        ("permission-update", value["id"])
        for value in installations
        if value["permission_update_requested"]
    )
    evidence = []
    for kind, installation_id in evidence_roles:
        suffix = "list" if installation_id is None else str(installation_id)
        filename = f"{evidence_prefix}-{kind}-{suffix}.txt"
        if kind == "installation-list":
            source_url = "https://github.com/settings/installations"
        elif kind == "installation-configure":
            source_url = f"https://github.com/settings/installations/{installation_id}"
        else:
            source_url = (
                f"https://github.com/settings/installations/{installation_id}/permissions/update"
            )
        item = {
            "filename": filename,
            "kind": kind,
            "installation_id": installation_id,
            "source_url": source_url,
            "captured_at": capture_time,
            "media_type": "text/plain; charset=utf-8",
            "full_page": True,
        }
        raw = _installed_app_evidence_bytes(item)
        item.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        evidence.append(item)
    return {
        "schema_version": 1,
        "captured_at": capture_time,
        "capture_principal": "jemsbhai",
        "repository": policy["repository"],
        "source_url": installed_apps["source_url"],
        "coverage_complete": True,
        "installations": installations,
        "evidence": evidence,
    }


def _installed_app_evidence_bytes(item):
    return (
        controls._app_evidence_header(item)
        + f"full owner-authenticated page capture: {item['filename']}\n"
    ).encode()


def _evidence_reader_for(capture):
    raw_by_filename = {
        item["filename"]: _installed_app_evidence_bytes(item) for item in capture["evidence"]
    }
    return raw_by_filename.__getitem__


def _refresh_evidence_manifest(capture):
    for item in capture["evidence"]:
        raw = _installed_app_evidence_bytes(item)
        item.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def _active_installed_app_authority(*, captured_at, evidence_prefix):
    capture = _matching_installed_app_authority(
        captured_at=captured_at,
        evidence_prefix=evidence_prefix,
    )
    for installation in capture["installations"]:
        installation.update(suspended=False, danger_zone_action="Suspend")
    return capture


def _preflight_workflow_run(run_id="123", run_attempt="1", actor="jemsbhai"):
    return {
        "id": run_id,
        "run_attempt": run_attempt,
        "ref": "refs/heads/main",
        "sha": SHA,
        "actor": actor,
        "triggering_actor": actor,
        "workflow": "Snapshot mutable controls before a stable tag",
    }


def _preflight_api_run(run_id=123, run_attempt=1, actor="jemsbhai"):
    policy, _ = _policy()
    return {
        "id": run_id,
        "repository": {"full_name": policy["repository"]},
        "path": ".github/workflows/release-preflight.yml",
        "event": "workflow_dispatch",
        "head_branch": "main",
        "head_sha": SHA,
        "status": "completed",
        "conclusion": "success",
        "run_attempt": run_attempt,
        "actor": {"login": actor},
        "triggering_actor": {"login": actor},
    }


def test_current_workflow_run_records_attempt_and_triggering_actor(monkeypatch):
    values = {
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_SHA": SHA,
        "GITHUB_ACTOR": "jemsbhai",
        "GITHUB_TRIGGERING_ACTOR": "jemsbhai",
        "GITHUB_WORKFLOW": "Snapshot mutable controls before a stable tag",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    assert controls._current_workflow_run() == {
        "id": "123",
        "run_attempt": "2",
        "ref": "refs/heads/main",
        "sha": SHA,
        "actor": "jemsbhai",
        "triggering_actor": "jemsbhai",
        "workflow": "Snapshot mutable controls before a stable tag",
    }


def _matching_observation(*, installed_app_captured_at=None):
    policy, _ = _policy()
    checks = list(policy["required_checks"])
    tag_policy = policy["tag_ruleset"]
    environment_policy = policy["pypi_environment"]
    provider = policy["required_check_provider"]
    return {
        "repository": policy["repository"],
        "default_branch": policy["default_branch"],
        "capture_principal": "jemsbhai",
        "release_runner_authority": {
            "collaborators": [
                {
                    "login": "jemsbhai",
                    "role_name": "admin",
                    "permissions": {"admin": True, "maintain": True, "push": True},
                }
            ],
            "pending_invitations": [],
            "registered_runners": [],
            "repository_variable_names": [],
            "installed_apps": _matching_installed_app_authority(
                captured_at=installed_app_captured_at
            ),
        },
        "release_tag": "v0.15.0",
        "release_commit": SHA,
        "tag_exists": False,
        "fork_pr_contributor_approval": {
            "approval_policy": policy["fork_pr_contributor_approval"]["approval_policy"]
        },
        "immutable_releases": {"enabled": True, "enforced_by_owner": False},
        "branch_protection": {
            "enforce_admins": {"enabled": True},
            "required_status_checks": {
                "strict": True,
                "contexts": checks,
                "checks": [{"context": name, "app_id": provider["app_id"]} for name in checks],
            },
            "required_conversation_resolution": {"enabled": True},
            "allow_force_pushes": {"enabled": False},
            "allow_deletions": {"enabled": False},
        },
        "check_runs": [
            {
                "name": name,
                "status": "completed",
                "conclusion": "success",
                "completed_at": None,
                "details_url": (
                    f"https://github.com/{policy['repository']}/actions/runs/{1000 + index}"
                    f"/job/{2000 + index}"
                ),
                "app": {"id": provider["app_id"], "slug": provider["slug"]},
                "workflow_run": {
                    "id": 1000 + index,
                    "repository": policy["repository"],
                    **policy["required_check_workflows"][name],
                    "head_sha": SHA,
                    "status": "completed",
                    "conclusion": "success",
                    "run_attempt": 1,
                },
            }
            for index, name in enumerate(checks)
        ],
        "rulesets": [
            {
                "name": tag_policy["name"],
                "target": tag_policy["target"],
                "enforcement": tag_policy["enforcement"],
                "conditions": {
                    "ref_name": {
                        "include": tag_policy["include"],
                        "exclude": tag_policy["exclude"],
                    }
                },
                "rules": [{"type": value} for value in tag_policy["rule_types"]],
                "bypass_actors": [],
                "current_user_can_bypass": "never",
            }
        ],
        "pypi_environment": {
            "name": "pypi",
            "can_admins_bypass": False,
            "deployment_branch_policy": environment_policy["deployment_branch_policy"],
            "protection_rules": [
                {
                    "type": "required_reviewers",
                    "prevent_self_review": False,
                    "reviewers": [{"type": "User", "reviewer": {"login": "jemsbhai"}}],
                },
                {"type": "branch_policy"},
            ],
        },
        "deployment_policies": [{"name": "v*", "type": "tag"}],
        "repository_secret_names": [],
        "environment_secret_names": [],
    }


def _cuda_run(run_id=456, commit=SHA):
    policy, _ = _policy()
    cuda_policy = policy["cuda_evidence"]
    return {
        "id": run_id,
        "repository": {"full_name": policy["repository"]},
        "path": cuda_policy["workflow_path"],
        "event": cuda_policy["event"],
        "head_branch": cuda_policy["head_branch"],
        "head_sha": commit,
        "status": "completed",
        "conclusion": "success",
        "run_attempt": 1,
        "created_at": "2026-08-10T19:00:00Z",
        "updated_at": "2026-08-10T19:20:00Z",
    }


def _cuda_jobs(commit=SHA):
    policy, _ = _policy()
    return {
        "query_filter": "all",
        "pagination_complete": True,
        "jobs": [
            {
                "id": 1000 + index,
                "run_id": 456,
                "name": name,
                "status": "completed",
                "conclusion": "success",
                "run_attempt": 1,
                "head_sha": commit,
                "runner_id": 10 + index,
                "runner_name": (
                    policy["cuda_evidence"]["required_runner_labels"][name] + f"-jit-{index:016x}"
                ),
                "runner_group_id": 1,
                "runner_group_name": "Default",
                "labels": [
                    policy["cuda_evidence"]["required_runner_labels"][name] + f"-jit-{index:016x}",
                ],
            }
            for index, name in enumerate(policy["cuda_evidence"]["required_jobs"])
        ],
    }


def test_reviewed_control_policy_has_a_falsifiably_green_fixture():
    policy, _ = _policy()
    assert controls.evaluate_controls(policy, _matching_observation()) == []


@pytest.mark.parametrize("replacement", [None, False, 1, "true"])
def test_immutable_release_policy_requires_explicit_json_true(replacement):
    policy, _ = _policy()
    policy = copy.deepcopy(policy)
    observation = _matching_observation()
    if replacement is None:
        policy["immutable_releases"].pop("enabled")
    else:
        policy["immutable_releases"]["enabled"] = replacement
    violations = controls.evaluate_controls(policy, observation)
    assert any("policy.enabled must be the JSON boolean true" in value for value in violations)


@pytest.mark.parametrize("replacement", [None, False, 1, "true"])
def test_immutable_release_observation_requires_explicit_json_true(replacement):
    policy, _ = _policy()
    observation = _matching_observation()
    if replacement is None:
        observation["immutable_releases"].pop("enabled")
    else:
        observation["immutable_releases"]["enabled"] = replacement
    violations = controls.evaluate_controls(policy, observation)
    assert any("immutable_releases.enabled: expected true" in value for value in violations)


@pytest.mark.parametrize(
    "replacement", [None, "first_time_contributors", "first_time_contributors_new_to_github"]
)
def test_fork_approval_policy_requires_all_external_contributors(replacement):
    policy, _ = _policy()
    policy = copy.deepcopy(policy)
    if replacement is None:
        policy["fork_pr_contributor_approval"].pop("approval_policy")
    else:
        policy["fork_pr_contributor_approval"]["approval_policy"] = replacement

    violations = controls.evaluate_controls(policy, _matching_observation())

    assert any(
        "fork_pr_contributor_approval policy must require 'all_external_contributors'" in value
        for value in violations
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update(tag_exists=True), "must precede tagging"),
        (
            lambda value: value["immutable_releases"].update(enabled=False),
            "immutable_releases.enabled",
        ),
        (
            lambda value: value["fork_pr_contributor_approval"].update(
                approval_policy="first_time_contributors"
            ),
            "fork_pr_contributor_approval.approval_policy",
        ),
        (
            lambda value: value["branch_protection"]["enforce_admins"].update(enabled=False),
            "admin_enforcement",
        ),
        (
            lambda value: value["branch_protection"]["required_status_checks"].update(strict=False),
            "strict_required_checks",
        ),
        (
            lambda value: value["branch_protection"]["required_status_checks"]["contexts"].pop(),
            "required_checks",
        ),
        (
            lambda value: value["branch_protection"]["required_status_checks"]["checks"][0].update(
                app_id=None
            ),
            "required_check_bindings",
        ),
        (
            lambda value: value["check_runs"][0].update(conclusion="failure"),
            "completed/success",
        ),
        (
            lambda value: value["rulesets"][0].update(bypass_actors=[{"actor_id": 1}]),
            "bypass_actors",
        ),
        (
            lambda value: value["pypi_environment"].update(can_admins_bypass=True),
            "can_admins_bypass",
        ),
        (
            lambda value: value["pypi_environment"]["protection_rules"][0].update(reviewers=[]),
            "reviewers",
        ),
        (
            lambda value: value["check_runs"][0]["app"].update(slug="attacker"),
            "expected provider",
        ),
        (
            lambda value: value["check_runs"][0]["workflow_run"].update(
                path=".github/workflows/attacker.yml"
            ),
            "workflow run path",
        ),
        (
            lambda value: value.update(environment_secret_names=["PYPI_API_TOKEN"]),
            "environment_secret_names",
        ),
        (
            lambda value: value["release_runner_authority"]["collaborators"].append(
                {
                    "login": "read-collaborator",
                    "role_name": "read",
                    "permissions": {"admin": False, "maintain": False, "push": False},
                }
            ),
            "allowed_collaborator_logins",
        ),
        (
            lambda value: value["release_runner_authority"]["pending_invitations"].append(
                {"id": 9, "invitee": "pending-writer", "permissions": "push"}
            ),
            "pending_invitations",
        ),
        (
            lambda value: value["release_runner_authority"]["registered_runners"].append(
                {
                    "id": 91,
                    "name": "unexpected-runner",
                    "os": "linux",
                    "status": "offline",
                    "busy": False,
                    "labels": [],
                }
            ),
            "registered_runners",
        ),
        (
            lambda value: value["release_runner_authority"]["repository_variable_names"].append(
                "UNEXPECTED_RUNNER_ROUTE"
            ),
            "repository_variable_names",
        ),
        (
            lambda value: value["release_runner_authority"]["installed_apps"]["installations"][
                1
            ].update(suspended=False, danger_zone_action="Suspend"),
            "installations differ",
        ),
        (
            lambda value: value["release_runner_authority"]["installed_apps"][
                "installations"
            ].pop(),
            "installed App evidence",
        ),
        (
            lambda value: value["release_runner_authority"]["installed_apps"].update(
                coverage_complete=False
            ),
            "coverage_complete",
        ),
        (
            lambda value: value["release_runner_authority"]["installed_apps"]["installations"][5][
                "permissions"
            ]["write"].append("workflows"),
            "installations differ",
        ),
        (
            lambda value: value["release_runner_authority"]["collaborators"][0][
                "permissions"
            ].update(admin=False, maintain=False, push=False),
            "required_write_logins",
        ),
    ],
)
def test_control_policy_fails_closed_on_each_security_regression(mutation, match):
    policy, _ = _policy()
    observation = _matching_observation()
    mutation(observation)
    assert any(match in violation for violation in controls.evaluate_controls(policy, observation))


def test_latest_duplicate_check_run_cannot_be_masked_by_an_older_success():
    policy, _ = _policy()
    observation = _matching_observation()
    name = policy["required_checks"][0]
    observation["check_runs"] = [
        {"name": name, "status": "completed", "conclusion": "failure"},
        *observation["check_runs"],
    ]
    violations = controls.evaluate_controls(policy, observation)
    assert any(name in violation and "failure" in violation for violation in violations)


@pytest.mark.parametrize("immutable_not_found", [False, True])
def test_capture_observation_requires_tag_absence_and_collects_detailed_ruleset(
    immutable_not_found,
):
    policy, _ = _policy()
    observation = _matching_observation()
    root = f"repos/{policy['repository']}"
    responses = {
        f"{root}/immutable-releases": observation["immutable_releases"],
        f"{root}/rulesets": [{"id": 7, "name": policy["tag_ruleset"]["name"]}],
        f"{root}/rulesets/7": observation["rulesets"][0],
        f"{root}/environments/pypi/deployment-branch-policies": {
            "branch_policies": observation["deployment_policies"]
        },
        f"{root}/actions/secrets": {"secrets": []},
        f"{root}/environments/pypi/secrets": {"secrets": []},
        f"{root}/actions/permissions/fork-pr-contributor-approval": observation[
            "fork_pr_contributor_approval"
        ],
        f"{root}/collaborators?affiliation=all&per_page=100": [
            {
                "login": "jemsbhai",
                "role_name": "admin",
                "permissions": {
                    "admin": True,
                    "maintain": True,
                    "push": True,
                    "triage": True,
                    "pull": True,
                },
            }
        ],
        f"{root}/invitations?per_page=100": [],
        f"{root}/actions/runners?per_page=100": {
            "total_count": 0,
            "runners": [],
        },
        f"{root}/actions/variables?per_page=100": {
            "total_count": 0,
            "variables": [],
        },
        f"{root}/commits/{SHA}/check-runs?per_page=100": {
            "total_count": len(observation["check_runs"]),
            "check_runs": observation["check_runs"],
        },
        f"{root}/branches/main/protection": observation["branch_protection"],
        f"{root}/environments/pypi": observation["pypi_environment"],
        "user": {"login": "jemsbhai"},
    }
    for check in observation["check_runs"]:
        run = check["workflow_run"]
        responses[f"{root}/actions/runs/{run['id']}"] = {
            **run,
            "repository": {"full_name": run["repository"]},
        }

    def get_json(path):
        if path == f"{root}/git/ref/tags/v0.15.0":
            raise controls.ApiNotFoundError(path)
        if immutable_not_found and path == f"{root}/immutable-releases":
            raise controls.ApiNotFoundError(path)
        return responses[path]

    installed_apps = copy.deepcopy(observation["release_runner_authority"]["installed_apps"])
    captured = controls.capture_observation(
        policy=policy,
        release_tag="v0.15.0",
        release_commit=SHA,
        get_json=get_json,
        installed_app_authority=installed_apps,
        installed_app_evidence_reader=_evidence_reader_for(installed_apps),
    )
    expected = copy.deepcopy(observation)
    if immutable_not_found:
        expected["immutable_releases"] = {"enabled": False, "enforced_by_owner": False}
    assert captured == expected
    if immutable_not_found:
        violations = controls.evaluate_controls(policy, captured)
        assert any("immutable_releases.enabled" in value for value in violations)


def test_release_runner_authority_rejects_malformed_and_duplicate_collaborators():
    policy, _ = _policy()
    observation = _matching_observation()
    observation["release_runner_authority"]["collaborators"][0]["permissions"]["push"] = "true"
    observation["release_runner_authority"]["collaborators"].append(
        {
            "login": "jemsbhai",
            "role_name": "admin",
            "permissions": {"admin": True, "maintain": True, "push": True},
        }
    )

    violations = controls.evaluate_controls(policy, observation)

    assert any("permission 'push' must be boolean" in value for value in violations)
    assert any("collaborator 'jemsbhai' is duplicated" in value for value in violations)


@pytest.mark.parametrize("full_page", ["collaborators", "invitations"])
def test_capture_observation_rejects_authority_pages_that_may_be_incomplete(full_page):
    policy, _ = _policy()
    root = f"repos/{policy['repository']}"
    responses = {
        f"{root}/immutable-releases": {"enabled": True},
        f"{root}/rulesets": [],
        f"{root}/environments/pypi/deployment-branch-policies": {"branch_policies": []},
        f"{root}/actions/secrets": {"secrets": []},
        f"{root}/environments/pypi/secrets": {"secrets": []},
        f"{root}/actions/permissions/fork-pr-contributor-approval": {
            "approval_policy": "all_external_contributors"
        },
        f"{root}/collaborators?affiliation=all&per_page=100": [],
        f"{root}/invitations?per_page=100": [],
    }
    if full_page == "collaborators":
        responses[f"{root}/collaborators?affiliation=all&per_page=100"] = [
            {"login": f"user-{index}"} for index in range(100)
        ]
    else:
        responses[f"{root}/invitations?per_page=100"] = [{"id": index} for index in range(100)]

    installed_apps = _matching_installed_app_authority()
    with pytest.raises(ValueError, match=f"repository {full_page} capture may be incomplete"):
        controls.capture_observation(
            policy=policy,
            release_tag="v0.15.0",
            release_commit=SHA,
            get_json=responses.__getitem__,
            installed_app_authority=installed_apps,
            installed_app_evidence_reader=_evidence_reader_for(installed_apps),
        )


def test_release_runner_authority_policy_cannot_approve_pending_invitations():
    policy, _ = _policy()
    policy = copy.deepcopy(policy)
    policy["release_runner_authority"]["pending_invitations"] = ["pending-writer"]

    violations = controls.evaluate_controls(policy, _matching_observation())

    assert any("policy must be an empty array" in value for value in violations)


@pytest.mark.parametrize("field", ["registered_runners", "repository_variable_names"])
def test_release_runner_authority_policy_cannot_approve_persistent_surfaces(field):
    policy, _ = _policy()
    policy = copy.deepcopy(policy)
    policy["release_runner_authority"][field] = ["unexpected"]

    violations = controls.evaluate_controls(policy, _matching_observation())

    assert any(f"{field} policy must be an empty array" in value for value in violations)


def test_release_runner_authority_policy_cannot_leave_workflow_capable_app_active():
    policy, _ = _policy()
    policy = copy.deepcopy(policy)
    policy["release_runner_authority"]["installed_apps"]["expected_installations"][0].update(
        suspended=False,
        danger_zone_action="Suspend",
    )

    violations = controls.evaluate_controls(policy, _matching_observation())

    assert any("leaves active runner authority" in value for value in violations)


def test_all_repository_app_cannot_deny_release_repository_access():
    policy, _ = _policy()
    policy = copy.deepcopy(policy)
    observation = _matching_observation()
    for value in (
        policy["release_runner_authority"]["installed_apps"]["expected_installations"][0],
        observation["release_runner_authority"]["installed_apps"]["installations"][0],
    ):
        value.update(repository_access=False)

    with pytest.raises(ValueError, match="all-repository selection must include"):
        controls.evaluate_controls(policy, observation)


def test_pending_permission_update_must_be_suspended_even_without_displayed_delta():
    policy, _ = _policy()
    policy = copy.deepcopy(policy)
    observation = _matching_observation()
    installation_id = 14141661
    for values in (
        policy["release_runner_authority"]["installed_apps"]["expected_installations"],
        observation["release_runner_authority"]["installed_apps"]["installations"],
    ):
        value = next(item for item in values if item["id"] == installation_id)
        value.update(suspended=False, danger_zone_action="Suspend")

    violations = controls.evaluate_controls(policy, observation)

    assert any("leaves an unresolved permission update active" in value for value in violations)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(schema_version=True),
        lambda value: value.update(captured_at="not-a-time"),
        lambda value: value.update(capture_principal="attacker"),
        lambda value: value.update(repository="attacker/explainiverse"),
        lambda value: value.update(source_url="https://attacker.invalid"),
        lambda value: value["installations"].append(copy.deepcopy(value["installations"][0])),
        lambda value: value["installations"][0].update(id=True),
        lambda value: value["installations"][0]["permissions"]["write"].append("actions"),
        lambda value: value["installations"][0].update(unexpected=True),
        lambda value: value["evidence"].pop(),
        lambda value: value["evidence"][0].update(source_url="https://github.com/attacker"),
        lambda value: value["evidence"][1].update(
            kind=value["evidence"][0]["kind"],
            installation_id=value["evidence"][0]["installation_id"],
        ),
        lambda value: value["evidence"][0].update(sha256="A" * 64),
    ],
)
def test_installed_app_authority_capture_is_strict_and_policy_bound(mutation):
    policy, _ = _policy()
    observation = _matching_observation()
    capture = observation["release_runner_authority"]["installed_apps"]
    mutation(capture)

    violations = controls.evaluate_controls(policy, observation)

    assert any("installed_apps" in value for value in violations)


def test_capture_recomputes_installed_app_evidence_bytes_and_digest():
    policy, _ = _policy()
    capture = _matching_installed_app_authority()

    with pytest.raises(ValueError, match="byte count differs|digest differs"):
        controls._normalize_installed_app_authority(
            capture,
            repository=policy["repository"],
            capture_principal="jemsbhai",
            evidence_reader=lambda filename: b"wrong evidence",
        )


def test_installed_app_aggregate_time_must_equal_latest_page_time():
    policy, _ = _policy()
    now = datetime(2026, 8, 10, 20, 0, tzinfo=timezone.utc)
    capture = _matching_installed_app_authority(captured_at=now.isoformat())
    for item in capture["evidence"]:
        item["captured_at"] = (now - timedelta(minutes=5)).isoformat()
    _refresh_evidence_manifest(capture)

    with pytest.raises(ValueError, match="must equal the latest evidence page timestamp"):
        controls._normalize_installed_app_authority(
            capture,
            repository=policy["repository"],
            capture_principal="jemsbhai",
            evidence_reader=_evidence_reader_for(capture),
        )


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        (b"\xff\xfe\x00", "not strict UTF-8 text"),
        (b"wrong header\npage content\n", "does not begin with the exact"),
    ],
)
def test_installed_app_evidence_requires_strict_utf8_and_exact_header(raw, match):
    policy, _ = _policy()
    capture = _matching_installed_app_authority()
    first = capture["evidence"][0]
    first.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    expected_reader = _evidence_reader_for(capture)

    def reader(filename):
        return raw if filename == first["filename"] else expected_reader(filename)

    with pytest.raises(ValueError, match=match):
        controls._normalize_installed_app_authority(
            capture,
            repository=policy["repository"],
            capture_principal="jemsbhai",
            evidence_reader=reader,
        )


def test_installed_app_evidence_file_reader_rejects_symlink_and_reads_exact_file(tmp_path):
    evidence = tmp_path / "installation-list.txt"
    evidence.write_bytes(b"complete page\n")
    assert (
        controls._read_installed_app_evidence_file(tmp_path, evidence.name) == evidence.read_bytes()
    )

    alias = tmp_path / "alias.txt"
    try:
        alias.symlink_to(evidence)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ValueError, match="single-link regular file"):
        controls._read_installed_app_evidence_file(tmp_path, alias.name)


def test_installed_app_restoration_requires_exact_state_and_fresh_distinct_evidence():
    policy, _ = _policy()
    before_time = datetime(2026, 8, 10, 20, 0, tzinfo=timezone.utc)
    restored_time = before_time + timedelta(hours=2)
    before = _active_installed_app_authority(
        captured_at=before_time.isoformat(), evidence_prefix="before"
    )
    restored = _active_installed_app_authority(
        captured_at=restored_time.isoformat(), evidence_prefix="restored"
    )

    report = controls.verify_installed_app_restoration(
        before=before,
        restored=restored,
        repository=policy["repository"],
        capture_principal="jemsbhai",
        before_evidence_reader=_evidence_reader_for(before),
        restored_evidence_reader=_evidence_reader_for(restored),
        now=restored_time + timedelta(minutes=1),
    )
    assert report["restoration_exact"] is True
    assert report["pre_window_capture_sha256"] != report["restored_capture_sha256"]

    changed = copy.deepcopy(restored)
    changed["installations"][0]["permissions"]["write"].append("workflows")
    with pytest.raises(ValueError, match="differs from the pre-window record"):
        controls.verify_installed_app_restoration(
            before=before,
            restored=changed,
            repository=policy["repository"],
            capture_principal="jemsbhai",
            before_evidence_reader=_evidence_reader_for(before),
            restored_evidence_reader=_evidence_reader_for(restored),
            now=restored_time + timedelta(minutes=1),
        )

    replayed = copy.deepcopy(before)
    with pytest.raises(ValueError, match="reuses a pre-window page capture"):
        controls.verify_installed_app_restoration(
            before=before,
            restored=replayed,
            repository=policy["repository"],
            capture_principal="jemsbhai",
            before_evidence_reader=_evidence_reader_for(before),
            restored_evidence_reader=_evidence_reader_for(before),
            now=restored_time + timedelta(minutes=1),
        )


def test_installed_app_restoration_checks_each_page_freshness_and_role_chronology():
    policy, _ = _policy()
    now = datetime(2026, 8, 10, 20, 0, tzinfo=timezone.utc)
    before = _active_installed_app_authority(
        captured_at=(now - timedelta(hours=3)).isoformat(), evidence_prefix="before-freshness"
    )
    restored = _active_installed_app_authority(
        captured_at=(now - timedelta(minutes=29)).isoformat(),
        evidence_prefix="restored-freshness",
    )
    restored["evidence"][0]["captured_at"] = (now - timedelta(minutes=39)).isoformat()
    _refresh_evidence_manifest(restored)

    with pytest.raises(ValueError, match="evidence page is stale"):
        controls.verify_installed_app_restoration(
            before=before,
            restored=restored,
            repository=policy["repository"],
            capture_principal="jemsbhai",
            before_evidence_reader=_evidence_reader_for(before),
            restored_evidence_reader=_evidence_reader_for(restored),
            now=now,
        )

    before = _active_installed_app_authority(
        captured_at=(now - timedelta(minutes=2)).isoformat(), evidence_prefix="before-order"
    )
    restored = _active_installed_app_authority(
        captured_at=(now - timedelta(minutes=1)).isoformat(), evidence_prefix="restored-order"
    )
    restored["evidence"][0]["captured_at"] = (now - timedelta(minutes=3)).isoformat()
    _refresh_evidence_manifest(restored)

    with pytest.raises(ValueError, match="must postdate the matching pre-window page"):
        controls.verify_installed_app_restoration(
            before=before,
            restored=restored,
            repository=policy["repository"],
            capture_principal="jemsbhai",
            before_evidence_reader=_evidence_reader_for(before),
            restored_evidence_reader=_evidence_reader_for(restored),
            now=now,
        )


def test_output_and_digest_sidecar_cannot_alias_retained_inputs(tmp_path):
    authority = tmp_path / "installed-apps.json"
    with pytest.raises(ValueError, match="must not overwrite"):
        controls._reject_output_aliases(authority, [authority])

    output = tmp_path / "report.json"
    with pytest.raises(ValueError, match="must not overwrite"):
        controls._reject_output_aliases(
            output,
            [output.with_suffix(output.suffix + ".sha256")],
        )


@pytest.mark.parametrize(
    ("endpoint", "response_name"),
    [
        ("actions/runners?per_page=100", "repository runners"),
        ("actions/variables?per_page=100", "repository variables"),
    ],
)
def test_capture_observation_rejects_incomplete_runner_surfaces(endpoint, response_name):
    policy, _ = _policy()
    root = f"repos/{policy['repository']}"
    responses = {
        f"{root}/immutable-releases": {"enabled": True},
        f"{root}/rulesets": [],
        f"{root}/environments/pypi/deployment-branch-policies": {"branch_policies": []},
        f"{root}/actions/secrets": {"secrets": []},
        f"{root}/environments/pypi/secrets": {"secrets": []},
        f"{root}/actions/permissions/fork-pr-contributor-approval": {
            "approval_policy": "all_external_contributors"
        },
        f"{root}/collaborators?affiliation=all&per_page=100": [],
        f"{root}/invitations?per_page=100": [],
        f"{root}/actions/runners?per_page=100": {"total_count": 0, "runners": []},
        f"{root}/actions/variables?per_page=100": {"total_count": 0, "variables": []},
    }
    key = "runners" if endpoint.startswith("actions/runners") else "variables"
    responses[f"{root}/{endpoint}"] = {"total_count": 1, key: []}

    installed_apps = _matching_installed_app_authority()
    with pytest.raises(ValueError, match=f"{response_name} capture is incomplete"):
        controls.capture_observation(
            policy=policy,
            release_tag="v0.15.0",
            release_commit=SHA,
            get_json=responses.__getitem__,
            installed_app_authority=installed_apps,
            installed_app_evidence_reader=_evidence_reader_for(installed_apps),
        )


def test_snapshot_is_bound_to_exact_policy_repository_tag_and_commit():
    policy, digest = _policy()
    snapshot = controls.make_snapshot(
        policy=policy,
        policy_sha256=digest,
        observation=_matching_observation(),
        workflow_run={"id": "123"},
    )
    controls.verify_snapshot(
        policy=policy,
        policy_sha256=digest,
        snapshot=snapshot,
        repository=policy["repository"],
        release_tag="v0.15.0",
        release_commit=SHA,
    )
    assert snapshot["pypi_trusted_publisher"]["verification_status"] == (
        "blocked_no_public_read_api"
    )

    for field, replacement in (
        ("repository", "attacker/fork"),
        ("release_tag", "v0.15.1"),
        ("release_commit", "b" * 40),
    ):
        tampered = copy.deepcopy(snapshot)
        tampered["observation"][field] = replacement
        with pytest.raises(ValueError, match=field):
            controls.verify_snapshot(
                policy=policy,
                policy_sha256=digest,
                snapshot=tampered,
                repository=policy["repository"],
                release_tag="v0.15.0",
                release_commit=SHA,
            )


def test_snapshot_is_bound_to_its_successful_pre_tag_workflow_run():
    policy, digest = _policy()
    snapshot = controls.make_snapshot(
        policy=policy,
        policy_sha256=digest,
        observation=_matching_observation(),
        workflow_run=_preflight_workflow_run(),
    )
    run = _preflight_api_run()
    controls.verify_preflight_source_run(
        run,
        snapshot,
        run_id="123",
        repository=policy["repository"],
        release_commit=SHA,
    )
    for field, replacement in (
        ("path", ".github/workflows/attacker.yml"),
        ("event", "pull_request"),
        ("head_branch", "feature"),
        ("head_sha", "b" * 40),
        ("conclusion", "failure"),
    ):
        tampered = copy.deepcopy(run)
        tampered[field] = replacement
        with pytest.raises(ValueError, match="preflight source run"):
            controls.verify_preflight_source_run(
                tampered,
                snapshot,
                run_id="123",
                repository=policy["repository"],
                release_commit=SHA,
            )

    for field, replacement, match in (
        ("run_attempt", 2, "run attempt"),
        ("actor", {"login": "someone-else"}, "actor"),
        ("triggering_actor", {"login": "someone-else"}, "triggering actor"),
    ):
        tampered = copy.deepcopy(run)
        tampered[field] = replacement
        with pytest.raises(ValueError, match=match):
            controls.verify_preflight_source_run(
                tampered,
                snapshot,
                run_id="123",
                repository=policy["repository"],
                release_commit=SHA,
            )

    for field, replacement, match in (
        ("actor", "someone-else", "authenticated capture principal"),
        ("triggering_actor", "someone-else", "authenticated capture principal"),
        ("run_attempt", "2", "run attempt"),
        ("run_attempt", None, "positive integer"),
    ):
        tampered = copy.deepcopy(snapshot)
        tampered["workflow_run"][field] = replacement
        with pytest.raises(ValueError, match=match):
            controls.verify_preflight_source_run(
                run,
                tampered,
                run_id="123",
                repository=policy["repository"],
                release_commit=SHA,
            )


def test_snapshot_workflow_identity_cannot_be_replayed_from_another_run():
    policy, digest = _policy()
    snapshot = controls.make_snapshot(
        policy=policy,
        policy_sha256=digest,
        observation=_matching_observation(),
        workflow_run=_preflight_workflow_run(run_id="999"),
    )
    run = _preflight_api_run()
    with pytest.raises(ValueError, match="snapshot workflow run id"):
        controls.verify_preflight_source_run(
            run,
            snapshot,
            run_id="123",
            repository=policy["repository"],
            release_commit=SHA,
        )


def test_admin_snapshot_binding_requires_freshness_and_same_dispatch_actor():
    policy, digest = _policy()
    now = datetime(2026, 8, 10, 20, 0, tzinfo=timezone.utc)
    snapshot = controls.make_snapshot(
        policy=policy,
        policy_sha256=digest,
        observation=_matching_observation(
            installed_app_captured_at=(now - timedelta(minutes=6)).isoformat()
        ),
        workflow_run={"id": None},
        now=now - timedelta(minutes=5),
    )
    workflow_run = _preflight_workflow_run()
    bound = controls.bind_snapshot_to_workflow(
        policy=policy,
        policy_sha256=digest,
        snapshot=snapshot,
        repository=policy["repository"],
        release_tag="v0.15.0",
        release_commit=SHA,
        workflow_run=workflow_run,
        cuda_run=_cuda_run(),
        cuda_jobs=_cuda_jobs(),
        cuda_run_id="456",
        now=now,
    )
    assert bound["workflow_run"] == workflow_run
    assert [job["name"] for job in bound["cuda_evidence"]["jobs"]] == sorted(
        policy["cuda_evidence"]["required_jobs"]
    )

    for mutation, match in (
        ({"actor": "someone-else"}, "dispatch actor and triggering actor"),
        ({"triggering_actor": "someone-else"}, "dispatch actor and triggering actor"),
        ({"triggering_actor": None}, "dispatch actor and triggering actor"),
        ({"run_attempt": None}, "workflow run attempt"),
        ({"run_attempt": "0"}, "workflow run attempt"),
    ):
        with pytest.raises(ValueError, match=match):
            controls.bind_snapshot_to_workflow(
                policy=policy,
                policy_sha256=digest,
                snapshot=snapshot,
                repository=policy["repository"],
                release_tag="v0.15.0",
                release_commit=SHA,
                workflow_run={**workflow_run, **mutation},
                cuda_run=_cuda_run(),
                cuda_jobs=_cuda_jobs(),
                cuda_run_id="456",
                now=now,
            )
    with pytest.raises(ValueError, match="stale"):
        controls.bind_snapshot_to_workflow(
            policy=policy,
            policy_sha256=digest,
            snapshot=snapshot,
            repository=policy["repository"],
            release_tag="v0.15.0",
            release_commit=SHA,
            workflow_run=workflow_run,
            cuda_run=_cuda_run(),
            cuda_jobs=_cuda_jobs(),
            cuda_run_id="456",
            now=now + timedelta(hours=1),
        )


def test_publish_verification_rejects_replay_after_thirty_minutes():
    policy, digest = _policy()
    now = datetime(2026, 8, 10, 20, 0, tzinfo=timezone.utc)
    snapshot = controls.make_snapshot(
        policy=policy,
        policy_sha256=digest,
        observation=_matching_observation(
            installed_app_captured_at=(now - timedelta(minutes=30)).isoformat()
        ),
        workflow_run={},
        now=now - timedelta(minutes=29),
    )

    controls.verify_snapshot(
        policy=policy,
        policy_sha256=digest,
        snapshot=snapshot,
        repository=policy["repository"],
        release_tag="v0.15.0",
        release_commit=SHA,
        now=now,
    )
    with pytest.raises(ValueError, match="stale"):
        controls.verify_snapshot(
            policy=policy,
            policy_sha256=digest,
            snapshot=snapshot,
            repository=policy["repository"],
            release_tag="v0.15.0",
            release_commit=SHA,
            now=now + timedelta(minutes=2),
        )


@pytest.mark.parametrize(
    ("captured_at_delta", "match"),
    [
        (timedelta(minutes=-11), "installed App authority capture is stale"),
        (timedelta(minutes=2), "installed App authority captured_at is after"),
    ],
)
def test_snapshot_rejects_installed_app_capture_outside_freshness_window(captured_at_delta, match):
    policy, digest = _policy()
    now = datetime(2026, 8, 10, 20, 0, tzinfo=timezone.utc)
    snapshot = controls.make_snapshot(
        policy=policy,
        policy_sha256=digest,
        observation=_matching_observation(
            installed_app_captured_at=(now + captured_at_delta).isoformat()
        ),
        workflow_run={},
        now=now,
    )

    with pytest.raises(ValueError, match=match):
        controls.verify_snapshot(
            policy=policy,
            policy_sha256=digest,
            snapshot=snapshot,
            repository=policy["repository"],
            release_tag="v0.15.0",
            release_commit=SHA,
            now=now,
        )


def test_snapshot_rejects_installed_app_capture_stale_at_verification():
    policy, digest = _policy()
    now = datetime(2026, 8, 10, 20, 0, tzinfo=timezone.utc)
    snapshot_time = now - timedelta(minutes=29)
    snapshot = controls.make_snapshot(
        policy=policy,
        policy_sha256=digest,
        observation=_matching_observation(
            installed_app_captured_at=(snapshot_time - timedelta(minutes=2)).isoformat()
        ),
        workflow_run={},
        now=snapshot_time,
    )
    assert snapshot["repository_controls_accepted"] is True

    with pytest.raises(
        ValueError, match="installed App authority capture is stale at verification"
    ):
        controls.verify_snapshot(
            policy=policy,
            policy_sha256=digest,
            snapshot=snapshot,
            repository=policy["repository"],
            release_tag="v0.15.0",
            release_commit=SHA,
            now=now,
        )


def test_make_snapshot_fails_closed_on_stale_installed_app_capture():
    policy, digest = _policy()
    now = datetime(2026, 8, 10, 20, 0, tzinfo=timezone.utc)
    snapshot = controls.make_snapshot(
        policy=policy,
        policy_sha256=digest,
        observation=_matching_observation(
            installed_app_captured_at=(now - timedelta(minutes=11)).isoformat()
        ),
        workflow_run={},
        now=now,
    )

    assert snapshot["repository_controls_accepted"] is False
    assert any(
        "installed App authority capture is stale" in value for value in snapshot["violations"]
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda run, jobs: run.update(head_sha="b" * 40), "head SHA"),
        (lambda run, jobs: run.update(id=0), "run id must be a positive integer"),
        (lambda run, jobs: run.update(id=True), "run id must be a positive integer"),
        (lambda run, jobs: run.update(id=999), "run id mismatch"),
        (
            lambda run, jobs: jobs["jobs"].append(copy.deepcopy(jobs["jobs"][0])),
            "exactly one all-attempt",
        ),
        (lambda run, jobs: jobs.update(query_filter="latest"), "filter=all"),
        (lambda run, jobs: jobs.update(pagination_complete=False), "complete pagination"),
        (
            lambda run, jobs: jobs["jobs"][0].update(conclusion="failure"),
            "did not complete successfully",
        ),
    ],
)
def test_cuda_evidence_fails_closed_on_wrong_commit_rerun_or_incomplete_query(mutation, match):
    policy, _ = _policy()
    run = _cuda_run()
    jobs = _cuda_jobs()
    mutation(run, jobs)

    with pytest.raises(ValueError, match=match):
        controls.verify_cuda_evidence(
            policy,
            run,
            jobs,
            run_id="456",
            repository=policy["repository"],
            release_commit=SHA,
        )


@pytest.mark.parametrize("attempt", [2, 0, None, True, False, "1"])
def test_cuda_evidence_requires_exactly_first_run_attempt(attempt):
    policy, _ = _policy()
    run = _cuda_run()
    run["run_attempt"] = attempt

    with pytest.raises(ValueError, match="run attempt must be the integer 1"):
        controls.verify_cuda_evidence(
            policy,
            run,
            _cuda_jobs(),
            run_id="456",
            repository=policy["repository"],
            release_commit=SHA,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("run_attempt", 2, "run attempt must be the integer 1"),
        ("run_attempt", True, "run attempt must be the integer 1"),
        ("head_sha", None, "head SHA mismatch"),
        ("head_sha", "b" * 40, "head SHA mismatch"),
        ("id", 0, "id must be a positive integer"),
        ("id", True, "id must be a positive integer"),
        ("run_id", 0, "run id must be a positive integer"),
        ("run_id", True, "run id must be a positive integer"),
        ("run_id", 999, "run id mismatch"),
        ("runner_id", 0, "runner id must be a positive integer"),
        ("runner_id", True, "runner id must be a positive integer"),
        ("runner_name", None, "runner name must be a non-empty string"),
        ("runner_name", "", "runner name must be a non-empty string"),
        ("runner_name", "   ", "runner name must be a non-empty string"),
        ("runner_name", "persistent-gpu-runner", "reviewed one-job JIT route"),
        (
            "runner_name",
            "explainiverse-cuda-two-jit-0000000000000001",
            "reviewed one-job JIT route",
        ),
        (
            "runner_name",
            "explainiverse-cuda-single-jit-AAAAAAAAAAAAAAAA",
            "reviewed one-job JIT route",
        ),
        ("runner_group_id", 0, "runner group id must be a positive integer"),
        ("runner_group_id", True, "runner group id must be a positive integer"),
        ("runner_group_id", 2, "reviewed default group 1"),
        ("runner_group_name", None, "reviewed default group 'Default'"),
        ("runner_group_name", "", "reviewed default group 'Default'"),
        ("runner_group_name", "Attacker", "reviewed default group 'Default'"),
    ],
)
def test_cuda_evidence_requires_bound_job_and_runner_identity(field, replacement, match):
    policy, _ = _policy()
    jobs = _cuda_jobs()
    jobs["jobs"][0][field] = replacement

    with pytest.raises(ValueError, match=match):
        controls.verify_cuda_evidence(
            policy,
            _cuda_run(),
            jobs,
            run_id="456",
            repository=policy["repository"],
            release_commit=SHA,
        )


def test_cuda_evidence_requires_a_distinct_runner_id_for_every_required_job():
    policy, _ = _policy()
    jobs = _cuda_jobs()
    jobs["jobs"][1]["runner_id"] = jobs["jobs"][0]["runner_id"]

    with pytest.raises(ValueError, match="runner id .* is reused"):
        controls.verify_cuda_evidence(
            policy,
            _cuda_run(),
            jobs,
            run_id="456",
            repository=policy["repository"],
            release_commit=SHA,
        )


def test_cuda_evidence_requires_a_distinct_generated_runner_name_for_every_required_job():
    policy, _ = _policy()
    jobs = _cuda_jobs()
    jobs["jobs"][1]["runner_name"] = jobs["jobs"][0]["runner_name"]

    with pytest.raises(ValueError, match="runner name .* is reused"):
        controls.verify_cuda_evidence(
            policy,
            _cuda_run(),
            jobs,
            run_id="456",
            repository=policy["repository"],
            release_commit=SHA,
        )


@pytest.mark.parametrize(
    "replacement",
    [
        [],
        ["explainiverse-cuda-single"],
        ["explainiverse-cuda-single-jit-0000000000000000", "self-hosted"],
        ["explainiverse-cuda-two-jit-0000000000000000"],
    ],
)
def test_cuda_evidence_requires_the_exact_one_use_job_label(replacement):
    policy, _ = _policy()
    jobs = _cuda_jobs()
    jobs["jobs"][0]["labels"] = replacement

    with pytest.raises(ValueError, match="labels must be exactly the one-use runner name"):
        controls.verify_cuda_evidence(
            policy,
            _cuda_run(),
            jobs,
            run_id="456",
            repository=policy["repository"],
            release_commit=SHA,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda cuda_policy: cuda_policy.pop("required_runner_labels"),
        lambda cuda_policy: cuda_policy.update(required_runner_labels=[]),
        lambda cuda_policy: cuda_policy["required_runner_labels"].pop(
            "CUDA single-GPU (Torch latest)"
        ),
        lambda cuda_policy: cuda_policy["required_runner_labels"].update(
            {"unexpected": "explainiverse-cuda-single"}
        ),
        lambda cuda_policy: cuda_policy["required_runner_labels"].update(
            {"CUDA single-GPU (Torch latest)": ""}
        ),
        lambda cuda_policy: cuda_policy["required_runner_labels"].update(
            {"CUDA single-GPU (Torch latest)": "explainiverse-cuda-two"}
        ),
    ],
)
def test_cuda_evidence_rejects_malformed_runner_label_policy(mutation):
    policy, _ = _policy()
    policy = copy.deepcopy(policy)
    mutation(policy["cuda_evidence"])

    with pytest.raises(ValueError, match="CUDA evidence required runner label"):
        controls.verify_cuda_evidence(
            policy,
            _cuda_run(),
            _cuda_jobs(),
            run_id="456",
            repository=policy["repository"],
            release_commit=SHA,
        )


def test_publish_requery_must_equal_attested_cuda_evidence():
    policy, digest = _policy()
    now = datetime(2026, 8, 10, 20, 0, tzinfo=timezone.utc)
    snapshot = controls.make_snapshot(
        policy=policy,
        policy_sha256=digest,
        observation=_matching_observation(
            installed_app_captured_at=(now - timedelta(minutes=2)).isoformat()
        ),
        workflow_run={},
        now=now - timedelta(minutes=1),
    )
    bound = controls.bind_snapshot_to_workflow(
        policy=policy,
        policy_sha256=digest,
        snapshot=snapshot,
        repository=policy["repository"],
        release_tag="v0.15.0",
        release_commit=SHA,
        workflow_run=_preflight_workflow_run(),
        cuda_run=_cuda_run(),
        cuda_jobs=_cuda_jobs(),
        cuda_run_id="456",
        now=now,
    )
    live_jobs = _cuda_jobs()
    replacement_name = "explainiverse-cuda-single-jit-ffffffffffffffff"
    live_jobs["jobs"][0]["runner_name"] = replacement_name
    live_jobs["jobs"][0]["labels"] = [replacement_name]

    with pytest.raises(ValueError, match="differs from the attested"):
        controls.verify_bound_cuda_evidence(
            policy=policy,
            snapshot=bound,
            cuda_run=_cuda_run(),
            cuda_jobs=live_jobs,
            cuda_run_id="456",
            repository=policy["repository"],
            release_commit=SHA,
        )


def test_verify_cli_rejects_policy_digest_substitution(tmp_path):
    policy, digest = _policy()
    snapshot = controls.make_snapshot(
        policy=policy,
        policy_sha256="0" * 64,
        observation=_matching_observation(),
        workflow_run={},
    )
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    cuda_run_path = tmp_path / "cuda-run.json"
    cuda_run_path.write_text(json.dumps(_cuda_run()), encoding="utf-8")
    cuda_jobs_path = tmp_path / "cuda-jobs.json"
    cuda_jobs_path.write_text(json.dumps(_cuda_jobs()), encoding="utf-8")
    assert digest != snapshot["policy_sha256"]
    assert (
        controls.main(
            [
                "verify",
                "--policy",
                str(POLICY_PATH),
                "--snapshot",
                str(snapshot_path),
                "--repository",
                policy["repository"],
                "--tag",
                "v0.15.0",
                "--commit",
                SHA,
                "--cuda-run-json",
                str(cuda_run_path),
                "--cuda-jobs-json",
                str(cuda_jobs_path),
                "--cuda-run-id",
                "456",
            ]
        )
        == 2
    )
