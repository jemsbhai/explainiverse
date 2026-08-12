"""Recovery-only release flow must reuse and re-verify original artifacts."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_release_recovery.py"
SPEC = importlib.util.spec_from_file_location("verify_release_recovery", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
recovery = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = recovery
SPEC.loader.exec_module(recovery)

SHA = "a" * 40
SOURCE_RUN_ID = "1234"
FILENAMES = ("explainiverse-0.15.0-py3-none-any.whl", "explainiverse-0.15.0.tar.gz")


def _artifacts(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    payloads = {FILENAMES[0]: b"original wheel", FILENAMES[1]: b"original sdist"}
    hashes = {}
    for filename, payload in payloads.items():
        (dist / filename).write_bytes(payload)
        hashes[filename] = hashlib.sha256(payload).hexdigest()
    sums = tmp_path / "SHA256SUMS"
    sums.write_text(
        "".join(f"{digest}  {filename}\n" for filename, digest in hashes.items()),
        encoding="utf-8",
    )
    pypi = {
        "info": {"name": "explainiverse", "version": "0.15.0"},
        "urls": [
            {"filename": filename, "digests": {"sha256": digest}}
            for filename, digest in hashes.items()
        ],
    }
    return dist, sums, hashes, pypi


def _source_run():
    run = {
        "id": int(SOURCE_RUN_ID),
        "run_attempt": 1,
        "actor": {"login": "jemsbhai"},
        "triggering_actor": {"login": "jemsbhai"},
        "repository": {"full_name": "jemsbhai/explainiverse"},
        "path": ".github/workflows/publish-pypi.yml",
        "event": "workflow_dispatch",
        "head_sha": SHA,
        "head_branch": "v0.15.0",
        "status": "completed",
        # The overall run is expected to be failed when GitHub Release creation failed.
        "conclusion": "failure",
    }
    jobs = {
        "query_filter": "all",
        "pagination_complete": True,
        "jobs": [
            {
                "name": name,
                "run_attempt": 1,
                "status": "completed",
                "conclusion": "success",
            }
            for name in (
                "Verify, build once, and inventory",
                "Attest the immutable distributions",
                "Publish through PyPI Trusted Publishing",
            )
        ]
        + [
            {
                "name": "Create the immutable GitHub release",
                "run_attempt": 1,
                "status": "completed",
                "conclusion": "failure",
                "steps": [
                    {"name": "Set up job", "status": "completed", "conclusion": "success"},
                    {
                        "name": "Stage an explicitly requested post-PyPI recovery drill",
                        "status": "completed",
                        "conclusion": "failure",
                    },
                    *[
                        {"name": name, "status": "completed", "conclusion": "skipped"}
                        for name in recovery._POST_PYPI_RELEASE_STEPS
                    ],
                    {"name": "Complete job", "status": "completed", "conclusion": "success"},
                ],
            }
        ],
    }
    return run, jobs


def _governance_record():
    return {
        "schema_version": 1,
        "release": {
            "repository": "jemsbhai/explainiverse",
            "tag": "v0.15.0",
            "commit": SHA,
        },
        "governance": {
            "release_dispatch_actor": "jemsbhai",
            "release_triggering_actor": "jemsbhai",
            "release_run_attempt": "1",
        },
        "evidence": {
            "release_workflow_run_id": SOURCE_RUN_ID,
            "release_workflow_run_url": (
                f"https://github.com/jemsbhai/explainiverse/actions/runs/{SOURCE_RUN_ID}"
            ),
        },
    }


def _verify_governance(record, run):
    recovery.verify_recovery_governance_record(
        record,
        run,
        repository="jemsbhai/explainiverse",
        release_tag="v0.15.0",
        release_commit=SHA,
        source_run_id=SOURCE_RUN_ID,
    )


def test_recovery_governance_record_is_bound_to_the_exact_source_run():
    run, _ = _source_run()
    _verify_governance(_governance_record(), run)


@pytest.mark.parametrize(
    ("target", "path", "replacement", "match"),
    [
        ("record", ("schema_version",), True, "schema_version"),
        ("record", ("release", "repository"), "other/repository", "record repository"),
        ("record", ("release", "tag"), "v0.15.1", "record tag"),
        ("record", ("release", "commit"), "b" * 40, "record commit"),
        ("record", ("evidence", "release_workflow_run_id"), "999", "source run id"),
        (
            "record",
            ("evidence", "release_workflow_run_url"),
            "https://github.com/jemsbhai/explainiverse/actions/runs/999",
            "source run URL",
        ),
        ("record", ("governance", "release_run_attempt"), "2", "source run attempt"),
        ("record", ("governance", "release_dispatch_actor"), "other", "source actor"),
        (
            "record",
            ("governance", "release_triggering_actor"),
            "other",
            "source triggering actor",
        ),
        ("run", ("id",), 999, "source run id mismatch"),
        ("run", ("repository", "full_name"), "other/repository", "source run repository"),
        ("run", ("head_branch",), "v0.15.1", "source run tag"),
        ("run", ("head_sha",), "b" * 40, "source run commit"),
        ("run", ("actor", "login"), "other", "record source actor"),
        (
            "run",
            ("triggering_actor", "login"),
            "other",
            "record source triggering actor",
        ),
    ],
)
def test_recovery_governance_record_rejects_cross_run_or_identity_drift(
    target, path, replacement, match
):
    record = _governance_record()
    run, _ = _source_run()
    value = record if target == "record" else run
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = replacement
    with pytest.raises(ValueError, match=match):
        _verify_governance(record, run)


@pytest.mark.parametrize("run_attempt", [2, 0, None, True, False])
def test_recovery_governance_record_requires_exact_integer_first_attempt(run_attempt):
    run, _ = _source_run()
    run["run_attempt"] = run_attempt
    with pytest.raises(ValueError, match="source run attempt must be the integer 1"):
        _verify_governance(_governance_record(), run)


def test_governance_record_cli_fails_closed_on_retained_record_substitution(tmp_path):
    record_path = tmp_path / "RELEASE_GOVERNANCE.json"
    run_path = tmp_path / "source-run.json"
    record = _governance_record()
    run, _ = _source_run()
    record_path.write_text(json.dumps(record), encoding="utf-8")
    run_path.write_text(json.dumps(run), encoding="utf-8")
    arguments = [
        "governance-record",
        "--record-json",
        str(record_path),
        "--run-json",
        str(run_path),
        "--repository",
        "jemsbhai/explainiverse",
        "--tag",
        "v0.15.0",
        "--commit",
        SHA,
        "--source-run-id",
        SOURCE_RUN_ID,
    ]
    assert recovery.main(arguments) == 0
    record["evidence"]["release_workflow_run_id"] = "999"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    assert recovery.main(arguments) == 2


def test_original_dist_pypi_and_github_assets_must_have_identical_hashes(tmp_path):
    dist, sums, hashes, pypi = _artifacts(tmp_path)
    expected = recovery.parse_sha256sums(sums)
    assert expected == hashes
    recovery.verify_distribution_directory(dist, expected)
    recovery.verify_pypi_json(pypi, project="explainiverse", version="0.15.0", expected=expected)
    release_assets = tmp_path / "github-assets"
    release_assets.mkdir()
    for source in dist.iterdir():
        (release_assets / source.name).write_bytes(source.read_bytes())
    recovery.verify_distribution_directory(release_assets, expected)


def test_github_release_requires_exact_dist_and_provenance_asset_inventory(tmp_path):
    dist, _, _, _ = _artifacts(tmp_path)
    provenance = tmp_path / "provenance"
    provenance.mkdir()
    (provenance / "SHA256SUMS").write_text("reviewed sums", encoding="utf-8")
    (provenance / "build.cdx.json").write_text("{}", encoding="utf-8")
    release_assets = tmp_path / "github-assets"
    release_assets.mkdir()
    for source in (*dist.iterdir(), *provenance.iterdir()):
        (release_assets / source.name).write_bytes(source.read_bytes())
    recovery.verify_release_asset_directory(release_assets, expected_directories=[dist, provenance])
    (release_assets / "unexpected.exe").write_bytes(b"unreviewed")
    with pytest.raises(ValueError, match="asset inventory mismatch"):
        recovery.verify_release_asset_directory(
            release_assets, expected_directories=[dist, provenance]
        )
    (release_assets / "unexpected.exe").unlink()
    (release_assets / "build.cdx.json").write_bytes(b"substituted")
    with pytest.raises(ValueError, match="asset SHA-256 mismatch"):
        recovery.verify_release_asset_directory(
            release_assets, expected_directories=[dist, provenance]
        )


def test_final_release_body_contains_the_exact_retained_governance_disclosure():
    disclosure = "## Release governance\n\nSingle-operator approval is disclosed."
    recovery.verify_release_governance_disclosure(
        {"body": f"Generated notes.\n\n{disclosure}\n"}, disclosure
    )


@pytest.mark.parametrize(
    ("body", "disclosure", "match"),
    [
        (None, "required disclosure", "body must be a string"),
        ("required disclosurE", "required disclosure", "exact governance disclosure"),
        ("anything", "", "must not be empty"),
    ],
)
def test_final_release_body_rejects_missing_or_near_match_disclosures(body, disclosure, match):
    with pytest.raises(ValueError, match=match):
        recovery.verify_release_governance_disclosure({"body": body}, disclosure)


def test_release_body_cli_fails_closed_on_disclosure_tampering(tmp_path):
    release_json = tmp_path / "release.json"
    disclosure_file = tmp_path / "RELEASE_GOVERNANCE.md"
    disclosure_file.write_text("authoritative disclosure\n", encoding="utf-8")
    release_json.write_text(
        json.dumps({"body": "notes\n\nauthoritative disclosure"}), encoding="utf-8"
    )
    arguments = [
        "release-body",
        "--release-json",
        str(release_json),
        "--disclosure",
        str(disclosure_file),
    ]
    assert recovery.main(arguments) == 0
    release_json.write_text(
        json.dumps({"body": "notes\n\nauthoritative disclosurE"}), encoding="utf-8"
    )
    assert recovery.main(arguments) == 2


@pytest.mark.parametrize(
    "line",
    [
        "not-a-hash  artifact.whl\n",
        f"{'0' * 64}  ../artifact.whl\n",
        f"{'0' * 64}  nested/artifact.whl\n",
        f"{'0' * 64}  provenance.json\n",
    ],
)
def test_sha256_manifest_rejects_malformed_or_unsafe_entries(tmp_path, line):
    sums = tmp_path / "SHA256SUMS"
    sums.write_text(line, encoding="utf-8")
    with pytest.raises(ValueError):
        recovery.parse_sha256sums(sums)


def test_sha256_manifest_rejects_duplicates_and_requires_both_distribution_kinds(tmp_path):
    digest = "0" * 64
    sums = tmp_path / "SHA256SUMS"
    sums.write_text(f"{digest}  artifact.whl\n{digest}  artifact.whl\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        recovery.parse_sha256sums(sums)
    sums.write_text(f"{digest}  artifact.whl\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source distribution"):
        recovery.parse_sha256sums(sums)


def test_distribution_verification_rejects_missing_extra_or_changed_files(tmp_path):
    dist, sums, _, _ = _artifacts(tmp_path)
    expected = recovery.parse_sha256sums(sums)
    (dist / FILENAMES[0]).write_bytes(b"substitution")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        recovery.verify_distribution_directory(dist, expected)
    (dist / FILENAMES[0]).write_bytes(b"original wheel")
    (dist / "attacker-1.0.whl").write_bytes(b"extra")
    with pytest.raises(ValueError, match="inventory mismatch"):
        recovery.verify_distribution_directory(dist, expected)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value["info"].update(name="other"), "project mismatch"),
        (lambda value: value["info"].update(version="0.15.1"), "version mismatch"),
        (
            lambda value: value["urls"][0]["digests"].update(sha256="0" * 64),
            "inventory/hash mismatch",
        ),
        (lambda value: value["urls"].pop(), "inventory/hash mismatch"),
    ],
)
def test_pypi_verification_fails_closed_on_identity_or_hash_drift(tmp_path, mutation, match):
    _, sums, _, pypi = _artifacts(tmp_path)
    expected = recovery.parse_sha256sums(sums)
    mutation(pypi)
    with pytest.raises(ValueError, match=match):
        recovery.verify_pypi_json(
            pypi, project="explainiverse", version="0.15.0", expected=expected
        )


def test_failed_overall_source_run_is_accepted_only_after_publish_succeeded():
    run, jobs = _source_run()
    assert (
        recovery.verify_source_run(
            run,
            jobs,
            repository="jemsbhai/explainiverse",
            workflow_path=".github/workflows/publish-pypi.yml",
            release_tag="v0.15.0",
            release_commit=SHA,
        )
        == "staged_drill"
    )
    publish = next(job for job in jobs["jobs"] if job["name"].startswith("Publish"))
    publish["conclusion"] = "failure"
    with pytest.raises(ValueError, match="Publish.*did not complete successfully"):
        recovery.verify_source_run(
            run,
            jobs,
            repository="jemsbhai/explainiverse",
            workflow_path=".github/workflows/publish-pypi.yml",
            release_tag="v0.15.0",
            release_commit=SHA,
        )


def test_source_run_must_prove_failed_overall_and_downstream_release_job():
    run, jobs = _source_run()
    run["conclusion"] = "success"
    with pytest.raises(ValueError, match="conclusion mismatch"):
        recovery.verify_source_run(
            run,
            jobs,
            repository="jemsbhai/explainiverse",
            workflow_path=".github/workflows/publish-pypi.yml",
            release_tag="v0.15.0",
            release_commit=SHA,
        )

    run, jobs = _source_run()
    jobs["jobs"][-1]["conclusion"] = "success"
    with pytest.raises(ValueError, match="demonstrate a downstream failure"):
        recovery.verify_source_run(
            run,
            jobs,
            repository="jemsbhai/explainiverse",
            workflow_path=".github/workflows/publish-pypi.yml",
            release_tag="v0.15.0",
            release_commit=SHA,
        )


def test_source_run_distinguishes_staged_drill_from_unplanned_failure():
    run, jobs = _source_run()
    release_steps = jobs["jobs"][-1]["steps"]
    release_steps[1]["conclusion"] = "skipped"
    release_steps[2]["conclusion"] = "failure"
    assert (
        recovery.verify_source_run(
            run,
            jobs,
            repository="jemsbhai/explainiverse",
            workflow_path=".github/workflows/publish-pypi.yml",
            release_tag="v0.15.0",
            release_commit=SHA,
        )
        == "unplanned_downstream_failure"
    )


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "later-ran", "no-failure"])
def test_source_run_rejects_ambiguous_downstream_failure_evidence(mutation):
    run, jobs = _source_run()
    release_job = jobs["jobs"][-1]
    if mutation == "missing":
        jobs["jobs"].pop()
    elif mutation == "duplicate":
        jobs["jobs"].append(copy.deepcopy(release_job))
    elif mutation == "later-ran":
        release_job["steps"][2]["conclusion"] = "success"
    else:
        release_job["steps"][1]["conclusion"] = "skipped"
    with pytest.raises(ValueError):
        recovery.verify_source_run(
            run,
            jobs,
            repository="jemsbhai/explainiverse",
            workflow_path=".github/workflows/publish-pypi.yml",
            release_tag="v0.15.0",
            release_commit=SHA,
        )


def test_source_run_rejects_hidden_rerun_attempts_and_incomplete_pagination():
    run, jobs = _source_run()
    jobs["jobs"].append(copy.deepcopy(jobs["jobs"][0]))
    with pytest.raises(ValueError, match="exactly one all-attempt"):
        recovery.verify_source_run(
            run,
            jobs,
            repository="jemsbhai/explainiverse",
            workflow_path=".github/workflows/publish-pypi.yml",
            release_tag="v0.15.0",
            release_commit=SHA,
        )

    _, jobs = _source_run()
    jobs["query_filter"] = "latest"
    with pytest.raises(ValueError, match="filter=all"):
        recovery.verify_source_run(
            run,
            jobs,
            repository="jemsbhai/explainiverse",
            workflow_path=".github/workflows/publish-pypi.yml",
            release_tag="v0.15.0",
            release_commit=SHA,
        )

    _, jobs = _source_run()
    jobs["pagination_complete"] = False
    with pytest.raises(ValueError, match="complete pagination"):
        recovery.verify_source_run(
            run,
            jobs,
            repository="jemsbhai/explainiverse",
            workflow_path=".github/workflows/publish-pypi.yml",
            release_tag="v0.15.0",
            release_commit=SHA,
        )


@pytest.mark.parametrize("run_attempt", [2, 0, None, True, False])
def test_source_run_requires_exact_integer_first_attempt(run_attempt):
    run, jobs = _source_run()
    run["run_attempt"] = run_attempt
    with pytest.raises(ValueError, match="source run attempt must be the integer 1"):
        recovery.verify_source_run(
            run,
            jobs,
            repository="jemsbhai/explainiverse",
            workflow_path=".github/workflows/publish-pypi.yml",
            release_tag="v0.15.0",
            release_commit=SHA,
        )


@pytest.mark.parametrize("run_attempt", [2, 0, None, True, False])
@pytest.mark.parametrize(
    "job_name",
    [
        "Verify, build once, and inventory",
        "Attest the immutable distributions",
        "Publish through PyPI Trusted Publishing",
        "Create the immutable GitHub release",
    ],
)
def test_source_run_requires_exact_integer_first_attempt_for_trusted_jobs(job_name, run_attempt):
    run, jobs = _source_run()
    job = next(value for value in jobs["jobs"] if value["name"] == job_name)
    job["run_attempt"] = run_attempt
    expected_error = f"source job {job_name!r} attempt must be the integer 1"
    with pytest.raises(ValueError, match=re.escape(expected_error)):
        recovery.verify_source_run(
            run,
            jobs,
            repository="jemsbhai/explainiverse",
            workflow_path=".github/workflows/publish-pypi.yml",
            release_tag="v0.15.0",
            release_commit=SHA,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("path", ".github/workflows/attacker.yml", "workflow path"),
        ("event", "pull_request", "event"),
        ("head_sha", "b" * 40, "head SHA"),
        ("head_branch", "main", "head branch/tag"),
        ("status", "in_progress", "status"),
    ],
)
def test_source_run_is_bound_to_exact_tag_workflow_and_commit(field, replacement, match):
    run, jobs = _source_run()
    tampered = copy.deepcopy(run)
    tampered[field] = replacement
    with pytest.raises(ValueError, match=match):
        recovery.verify_source_run(
            tampered,
            jobs,
            repository="jemsbhai/explainiverse",
            workflow_path=".github/workflows/publish-pypi.yml",
            release_tag="v0.15.0",
            release_commit=SHA,
        )


def test_recovery_workflow_has_no_pypi_upload_or_skip_existing_escape_hatch():
    workflow = (ROOT / ".github" / "workflows" / "recover-github-release.yml").read_text(
        encoding="utf-8"
    )
    lowered = workflow.lower()
    assert "gh-action-pypi-publish" not in lowered
    assert "skip-existing" not in lowered
    assert "verify_release_recovery.py source-run" in workflow
    assert "jobs?filter=all&per_page=100" in workflow
    assert "gh api --paginate" in workflow
    assert "gh attestation verify" in workflow
    assert "verify_release_recovery.py artifacts" in workflow
    assert "final-github-assets.sha256" in workflow
    assert "Archive complete or partial recovery evidence" in workflow
    assert "if: always()" in workflow
    final_api = workflow.index("> recovery/final-github-release-api.json")
    immutable = workflow.index(
        "'.immutable == true' recovery/final-github-release-api.json", final_api
    )
    final_disclosure = workflow.index("verify_release_recovery.py release-body", immutable)
    assert final_api < immutable < final_disclosure
    assert "--release-json recovery/final-github-release-api.json" in workflow
    assert "--disclosure provenance/RELEASE_GOVERNANCE.md" in workflow


def test_recovery_checks_immutable_release_setting_immediately_before_finalizing():
    workflow = (ROOT / ".github" / "workflows" / "recover-github-release.yml").read_text(
        encoding="utf-8"
    )
    finalize_step = workflow.split(
        "      - name: Finalize the verified draft without any PyPI upload", 1
    )[1].split("      - name: Record and reverify the final GitHub Release inventory", 1)[0]
    draft_guard = finalize_step.index('if [[ "$is_draft" = true ]]')
    setting_query = finalize_step.index(
        'gh api "repos/$GITHUB_REPOSITORY/immutable-releases"', draft_guard
    )
    api_version = finalize_step.index("X-GitHub-Api-Version: 2026-03-10", setting_query)
    explicit_true = finalize_step.index("jq -e '.enabled == true'", api_version)
    publish = finalize_step.index('gh release edit "$RELEASE_TAG"', explicit_true)
    assert draft_guard < setting_query < api_version < explicit_true < publish


def test_recovery_binds_governance_record_before_any_draft_mutation():
    workflow = (ROOT / ".github" / "workflows" / "recover-github-release.yml").read_text(
        encoding="utf-8"
    )
    binding = workflow.index("verify_release_recovery.py governance-record")
    draft_mutation = workflow.index("      - name: Create or inspect a recovery draft")
    assert binding < draft_mutation
    for argument in (
        "--record-json provenance/RELEASE_GOVERNANCE.json",
        "--run-json recovery/source-run.json",
        '--repository "$GITHUB_REPOSITORY"',
        '--tag "$RELEASE_TAG"',
        '--commit "$(git rev-parse HEAD)"',
        '--source-run-id "$SOURCE_RUN_ID"',
    ):
        assert argument in workflow[binding:draft_mutation]


def test_staged_drill_step_contract_matches_every_real_downstream_workflow_step():
    workflow = (ROOT / ".github" / "workflows" / "publish-pypi.yml").read_text(encoding="utf-8")
    release_job = workflow.split("  github-release:", 1)[1]
    step_names = [
        line.split("- name: ", 1)[1]
        for line in release_job.splitlines()
        if line.startswith("      - name: ")
    ]
    stage = "Stage an explicitly requested post-PyPI recovery drill"
    assert step_names[step_names.index(stage) + 1 :] == list(recovery._POST_PYPI_RELEASE_STEPS)
