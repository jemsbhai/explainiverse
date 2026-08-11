"""Security and replay tests for the stable-release control snapshot."""

from __future__ import annotations

import copy
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


def _matching_observation():
    policy, _ = _policy()
    checks = list(policy["required_checks"])
    tag_policy = policy["tag_ruleset"]
    environment_policy = policy["pypi_environment"]
    provider = policy["required_check_provider"]
    return {
        "repository": policy["repository"],
        "default_branch": policy["default_branch"],
        "capture_principal": "jemsbhai",
        "release_tag": "v0.15.0",
        "release_commit": SHA,
        "tag_exists": False,
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
                "name": name,
                "status": "completed",
                "conclusion": "success",
                "run_attempt": 1,
                "head_sha": commit,
                "runner_id": 10 + index,
                "runner_name": f"ephemeral-gpu-{index}",
                "runner_group_id": 7,
                "runner_group_name": "approved-gpu",
                "labels": ["self-hosted", "gpu"],
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
    ("mutation", "match"),
    [
        (lambda value: value.update(tag_exists=True), "must precede tagging"),
        (
            lambda value: value["immutable_releases"].update(enabled=False),
            "immutable_releases.enabled",
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

    captured = controls.capture_observation(
        policy=policy,
        release_tag="v0.15.0",
        release_commit=SHA,
        get_json=get_json,
    )
    expected = copy.deepcopy(observation)
    if immutable_not_found:
        expected["immutable_releases"] = {"enabled": False, "enforced_by_owner": False}
    assert captured == expected
    if immutable_not_found:
        violations = controls.evaluate_controls(policy, captured)
        assert any("immutable_releases.enabled" in value for value in violations)


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
        observation=_matching_observation(),
        workflow_run={"id": None},
    )
    snapshot["observed_at"] = (now - timedelta(minutes=5)).isoformat()
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
        observation=_matching_observation(),
        workflow_run={},
    )
    snapshot["observed_at"] = (now - timedelta(minutes=29)).isoformat()

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
    ("mutation", "match"),
    [
        (lambda run, jobs: run.update(head_sha="b" * 40), "head SHA"),
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


def test_publish_requery_must_equal_attested_cuda_evidence():
    policy, digest = _policy()
    now = datetime(2026, 8, 10, 20, 0, tzinfo=timezone.utc)
    snapshot = controls.make_snapshot(
        policy=policy,
        policy_sha256=digest,
        observation=_matching_observation(),
        workflow_run={},
    )
    snapshot["observed_at"] = (now - timedelta(minutes=1)).isoformat()
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
    live_jobs["jobs"][0]["runner_name"] = "different-runner"

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
