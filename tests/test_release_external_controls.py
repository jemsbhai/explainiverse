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
CUDA_EXCEPTION_ID = "EXPLAINIVERSE-v0.15.0-CPU-ONLY"


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


def _matching_observation(cuda_exception_id=None):
    policy, _ = _policy()
    checks = list(policy["required_checks"])
    if cuda_exception_id is not None:
        omitted = set(policy["cuda_release_exception"]["omitted_required_checks"])
        checks = [name for name in checks if name not in omitted]
    tag_policy = policy["tag_ruleset"]
    environment_policy = policy["pypi_environment"]
    provider = policy["required_check_provider"]
    return {
        "repository": policy["repository"],
        "default_branch": policy["default_branch"],
        "capture_principal": "jemsbhai",
        "release_tag": "v0.15.0",
        "release_commit": SHA,
        "cuda_exception_id": cuda_exception_id,
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
                "labels": [
                    "self-hosted",
                    "gpu",
                    policy["cuda_evidence"]["required_runner_labels"][name],
                ],
            }
            for index, name in enumerate(policy["cuda_evidence"]["required_jobs"])
        ],
    }


def test_reviewed_control_policy_has_a_falsifiably_green_fixture():
    policy, _ = _policy()
    assert controls.evaluate_controls(policy, _matching_observation()) == []


def test_explicit_v0150_exception_has_an_exact_21_check_effective_policy():
    policy, _ = _policy()
    observation = _matching_observation(CUDA_EXCEPTION_ID)

    assert len(policy["required_checks"]) == 23
    assert len(observation["branch_protection"]["required_status_checks"]["contexts"]) == 21
    assert controls.evaluate_controls(policy, observation) == []


def test_21_check_policy_is_rejected_without_the_explicit_exception():
    policy, _ = _policy()
    observation = _matching_observation(CUDA_EXCEPTION_ID)
    observation["cuda_exception_id"] = None

    violations = controls.evaluate_controls(policy, observation)

    assert any("main.required_checks" in violation for violation in violations)
    for name in policy["cuda_release_exception"]["omitted_required_checks"]:
        assert any(f"release commit check {name!r}: missing" in value for value in violations)


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("id", "CUDA-ANY-RELEASE", "id must be exactly"),
        ("release_tag", "v0.15.1", "release_tag must be exactly"),
        ("package_version", "0.15.1", "package_version must be exactly"),
        ("merge_pull_request", 6, "merge_pull_request must be exactly"),
        ("hardware_evidence_collected", True, "must be exactly False"),
        ("cuda_release_verified", True, "must be exactly False"),
        (
            "omitted_required_checks",
            ["Full extras, quality, and package"],
            "exactly the two reviewed CUDA check contexts",
        ),
        (
            "omitted_cuda_jobs",
            ["CUDA single-GPU (Torch latest)"],
            "exactly the four reviewed CUDA jobs",
        ),
    ],
)
def test_exception_policy_rejects_scope_or_identity_drift(field, replacement, match):
    policy, _ = _policy()
    policy = copy.deepcopy(policy)
    policy["cuda_release_exception"][field] = replacement

    with pytest.raises(ValueError, match=match):
        controls.evaluate_controls(policy, _matching_observation(CUDA_EXCEPTION_ID))


@pytest.mark.parametrize(
    ("tag", "exception_id", "match"),
    [
        ("v0.15.0", "wrong-exception", "id must be exactly"),
        ("v0.15.1", CUDA_EXCEPTION_ID, "restricted to 'v0.15.0'"),
    ],
)
def test_exception_resolution_rejects_wrong_id_or_any_other_tag(tag, exception_id, match):
    policy, _ = _policy()

    with pytest.raises(ValueError, match=match):
        controls._resolve_cuda_release_exception(
            policy,
            release_tag=tag,
            requested_exception_id=exception_id,
        )


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


def test_exception_snapshot_binds_an_explicit_cpu_only_gate_without_cuda_evidence():
    policy, digest = _policy()
    now = datetime(2026, 8, 10, 20, 0, tzinfo=timezone.utc)
    snapshot = controls.make_snapshot(
        policy=policy,
        policy_sha256=digest,
        observation=_matching_observation(CUDA_EXCEPTION_ID),
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
        cuda_exception_id=CUDA_EXCEPTION_ID,
        now=now,
    )

    gate = bound["cuda_release_gate"]
    assert gate == {
        "schema_version": 1,
        "mode": "cpu_only_exception",
        "status": "not_run",
        "exception_id": CUDA_EXCEPTION_ID,
        "release_tag": "v0.15.0",
        "release_commit": SHA,
        "package_version": "0.15.0",
        "merge_pull_request": 5,
        "hardware_evidence_collected": False,
        "cuda_release_verified": False,
        "omitted_required_checks": sorted(
            policy["cuda_release_exception"]["omitted_required_checks"]
        ),
        "omitted_cuda_jobs": sorted(policy["cuda_evidence"]["required_jobs"]),
        "authorized_by": ["jemsbhai"],
        "approved_at": "2026-09-03",
        "reason": policy["cuda_release_exception"]["reason"],
        "disclosure": policy["cuda_release_exception"]["disclosure"],
    }
    assert "cuda_evidence" not in bound
    assert bound["cuda_release_exception"] == policy["cuda_release_exception"] | {
        "omitted_required_checks": sorted(
            policy["cuda_release_exception"]["omitted_required_checks"]
        ),
        "omitted_cuda_jobs": sorted(policy["cuda_release_exception"]["omitted_cuda_jobs"]),
    }
    assert (
        controls.verify_bound_cuda_release_gate(
            policy=policy,
            snapshot=bound,
            release_tag="v0.15.0",
            release_commit=SHA,
            repository=policy["repository"],
            cuda_exception_id=CUDA_EXCEPTION_ID,
        )["mode"]
        == "cpu_only_exception"
    )


@pytest.mark.parametrize(
    ("cuda_run", "cuda_jobs", "cuda_run_id"),
    [
        ({}, None, None),
        (None, {}, None),
        (None, None, "456"),
        ({}, {}, "456"),
    ],
)
def test_exception_rejects_any_cuda_hardware_evidence_input(cuda_run, cuda_jobs, cuda_run_id):
    policy, _ = _policy()

    with pytest.raises(ValueError, match="must be absent"):
        controls.resolve_cuda_release_gate(
            policy=policy,
            release_tag="v0.15.0",
            release_commit=SHA,
            cuda_run=cuda_run,
            cuda_jobs=cuda_jobs,
            cuda_run_id=cuda_run_id,
            cuda_exception_id=CUDA_EXCEPTION_ID,
            repository=policy["repository"],
        )


@pytest.mark.parametrize(
    ("cuda_run", "cuda_jobs", "cuda_run_id"),
    [
        (None, None, None),
        (_cuda_run(), None, "456"),
        (None, _cuda_jobs(), "456"),
        (_cuda_run(), _cuda_jobs(), None),
    ],
)
def test_normal_mode_requires_the_complete_cuda_evidence_trio(cuda_run, cuda_jobs, cuda_run_id):
    policy, _ = _policy()

    with pytest.raises(ValueError, match="requires --cuda-run-id"):
        controls.resolve_cuda_release_gate(
            policy=policy,
            release_tag="v0.15.0",
            release_commit=SHA,
            cuda_run=cuda_run,
            cuda_jobs=cuda_jobs,
            cuda_run_id=cuda_run_id,
            cuda_exception_id=None,
            repository=policy["repository"],
        )


def test_exception_publish_verification_rejects_gate_or_evidence_tampering():
    policy, digest = _policy()
    now = datetime.now(timezone.utc)
    snapshot = controls.make_snapshot(
        policy=policy,
        policy_sha256=digest,
        observation=_matching_observation(CUDA_EXCEPTION_ID),
        workflow_run={},
    )
    bound = controls.bind_snapshot_to_workflow(
        policy=policy,
        policy_sha256=digest,
        snapshot=snapshot,
        repository=policy["repository"],
        release_tag="v0.15.0",
        release_commit=SHA,
        workflow_run=_preflight_workflow_run(),
        cuda_exception_id=CUDA_EXCEPTION_ID,
        now=now,
    )

    tampered_gate = copy.deepcopy(bound)
    tampered_gate["cuda_release_gate"]["omitted_required_checks"].append(
        "Full extras, quality, and package"
    )
    with pytest.raises(ValueError, match="differs from the attested preflight gate"):
        controls.verify_bound_cuda_release_gate(
            policy=policy,
            snapshot=tampered_gate,
            release_tag="v0.15.0",
            release_commit=SHA,
            repository=policy["repository"],
            cuda_exception_id=CUDA_EXCEPTION_ID,
        )

    fake_evidence = copy.deepcopy(bound)
    fake_evidence["cuda_evidence"] = {"run": {"id": 456}}
    with pytest.raises(ValueError, match="must not contain CUDA hardware evidence"):
        controls.verify_bound_cuda_release_gate(
            policy=policy,
            snapshot=fake_evidence,
            release_tag="v0.15.0",
            release_commit=SHA,
            repository=policy["repository"],
            cuda_exception_id=CUDA_EXCEPTION_ID,
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


@pytest.mark.parametrize(
    "replacement",
    [
        ["self-hosted", "gpu"],
        ["self-hosted", "gpu", "explainiverse-cuda-two"],
    ],
)
def test_cuda_evidence_requires_the_expected_custom_topology_label(replacement):
    policy, _ = _policy()
    jobs = _cuda_jobs()
    jobs["jobs"][0]["labels"] = replacement

    with pytest.raises(ValueError, match="expected custom runner label"):
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


def test_exception_bind_cli_emits_gate_record_and_outputs_only_after_success(tmp_path, monkeypatch):
    policy, digest = _policy()
    snapshot = controls.make_snapshot(
        policy=policy,
        policy_sha256=digest,
        observation=_matching_observation(CUDA_EXCEPTION_ID),
        workflow_run={},
    )
    snapshot_path = tmp_path / "admin-capture.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    output_path = tmp_path / "external-controls.json"
    gate_path = tmp_path / "cuda-gate-record.json"
    github_output = tmp_path / "github-output.txt"
    for name, value in {
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_SHA": SHA,
        "GITHUB_ACTOR": "jemsbhai",
        "GITHUB_TRIGGERING_ACTOR": "jemsbhai",
        "GITHUB_WORKFLOW": "Snapshot mutable controls before a stable tag",
    }.items():
        monkeypatch.setenv(name, value)

    assert (
        controls.main(
            [
                "bind",
                "--policy",
                str(POLICY_PATH),
                "--snapshot",
                str(snapshot_path),
                "--output",
                str(output_path),
                "--repository",
                policy["repository"],
                "--tag",
                "v0.15.0",
                "--commit",
                SHA,
                "--cuda-exception-id",
                CUDA_EXCEPTION_ID,
                "--cuda-gate-output",
                str(gate_path),
                "--github-output",
                str(github_output),
            ]
        )
        == 0
    )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    bound = json.loads(output_path.read_text(encoding="utf-8"))
    assert gate == bound["cuda_release_gate"]
    assert gate_path.with_suffix(".json.sha256").is_file()
    assert output_path.with_suffix(".json.sha256").is_file()
    assert github_output.read_text(encoding="utf-8").splitlines() == [
        "cuda_mode=cpu_only_exception",
        "cuda_run_id=",
        f"cuda_exception_id={CUDA_EXCEPTION_ID}",
    ]

    failed_output = tmp_path / "failed-github-output.txt"
    assert (
        controls.main(
            [
                "verify",
                "--policy",
                str(POLICY_PATH),
                "--snapshot",
                str(output_path),
                "--repository",
                policy["repository"],
                "--tag",
                "v0.15.1",
                "--commit",
                SHA,
                "--cuda-exception-id",
                CUDA_EXCEPTION_ID,
                "--github-output",
                str(failed_output),
            ]
        )
        == 2
    )
    assert not failed_output.exists()
