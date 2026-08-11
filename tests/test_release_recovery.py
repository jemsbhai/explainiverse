"""Recovery-only release flow must reuse and re-verify original artifacts."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
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
            {"name": name, "status": "completed", "conclusion": "success"}
            for name in (
                "Verify, build once, and inventory",
                "Attest the immutable distributions",
                "Publish through PyPI Trusted Publishing",
            )
        ]
        + [
            {
                "name": "Create the immutable GitHub release",
                "status": "completed",
                "conclusion": "failure",
            }
        ],
    }
    return run, jobs


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
    recovery.verify_source_run(
        run,
        jobs,
        repository="jemsbhai/explainiverse",
        workflow_path=".github/workflows/publish-pypi.yml",
        release_tag="v0.15.0",
        release_commit=SHA,
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
