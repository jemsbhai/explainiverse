"""Structural security contracts for P0 release and compatibility workflows."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def _read(name):
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def test_every_external_action_is_pinned_to_a_full_commit_sha():
    for workflow in WORKFLOWS.glob("*.yml"):
        for line in workflow.read_text(encoding="utf-8").splitlines():
            match = re.match(r"\s*uses:\s*([^#\s]+)", line)
            if match is None or match.group(1).startswith("./"):
                continue
            action = match.group(1)
            assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", action), (workflow, action)


def test_preflight_requires_fresh_admin_capture_then_attests_and_binds_it():
    workflow = _read("release-preflight.yml")
    assert "admin_snapshot_base64:" in workflow
    assert "cuda_run_id:" in workflow
    assert "jobs?filter=all&per_page=100" in workflow
    assert "gh api --paginate" in workflow
    assert "release_external_controls.py bind" in workflow
    assert "--cuda-run-json release-preflight/cuda-run.json" in workflow
    assert "--cuda-jobs-json release-preflight/cuda-jobs.json" in workflow
    assert "admin-capture.json" in workflow
    assert "actions/attest-build-provenance@" in workflow
    assert "GH_TOKEN: ${{ github.token }}" in workflow
    assert "if: always()" in workflow


def test_publish_cannot_build_without_preflight_and_real_cuda_edges():
    workflow = _read("publish-pypi.yml")
    assert "preflight_run_id:" in workflow
    assert "needs: [preflight, cuda-release]" in workflow
    assert "gh attestation verify release-preflight/artifact/external-controls.json" in workflow
    assert "release_external_controls.py verify" in workflow
    assert "Release CUDA single-GPU (Torch ${{ matrix.torch-edge }}, zero skips)" in workflow
    assert 'EXPLAINIVERSE_REQUIRED_CUDA_DEVICES: "1"' in workflow
    assert "needs: preflight" in workflow
    assert workflow.count("check_pypi_version_absent.py") == 2
    assert workflow.index("check_pypi_version_absent.py") < workflow.index("  build:")
    assert workflow.rindex("check_pypi_version_absent.py") < workflow.index(
        "pypa/gh-action-pypi-publish@"
    )
    assert "jobs?filter=all&per_page=100" in workflow
    assert "gh api --paginate" in workflow
    assert workflow.count("gh-action-pypi-publish@") == 1
    assert "skip-existing" not in workflow.lower()


def test_recovery_is_idempotent_downstream_only_and_hash_checks_all_services():
    workflow = _read("recover-github-release.yml")
    lowered = workflow.lower()
    assert "gh-action-pypi-publish" not in lowered
    assert "twine upload" not in lowered
    assert "skip-existing" not in lowered
    assert "verify_release_recovery.py source-run" in workflow
    assert "jobs?filter=all&per_page=100" in workflow
    assert "gh api --paginate" in workflow
    assert 'gh attestation verify "$artifact"' in workflow
    assert "https://pypi.org/pypi/explainiverse/$version/json" in workflow
    assert "--github-assets recovery/verified-release-assets" in workflow
    assert "--provenance provenance" in workflow
    assert "--draft=false" in workflow
    assert "final-github-assets.sha256" in workflow
    assert "final-github-release.json" in workflow
    assert "Archive complete or partial recovery evidence" in workflow
    assert "if: always()" in workflow


def test_cuda_workflow_has_required_and_scheduled_minimum_latest_zero_skip_lanes():
    workflow = _read("cuda-ci.yml")
    suite = (ROOT / "tests_cuda" / "test_cuda_release.py").read_text(encoding="utf-8")
    assert "pull_request:" in workflow and "schedule:" in workflow
    assert "CUDA single-GPU (Torch ${{ matrix.torch-edge }})" in workflow
    assert "CUDA two-GPU scheduled (Torch ${{ matrix.torch-edge }})" in workflow
    assert workflow.count("torch-edge: minimum") == 2
    assert workflow.count("torch-edge: latest") == 2
    assert 'EXPLAINIVERSE_REQUIRED_CUDA_DEVICES: "1"' in workflow
    assert 'EXPLAINIVERSE_REQUIRED_CUDA_DEVICES: "2"' in workflow
    assert workflow.count("tests_cuda/test_cuda_release.py") == 2
    assert "pytest.skip" not in suite
    assert "pytest.importorskip" not in suite
    assert "torch.cuda.device_count() >= REQUIRED_CUDA_DEVICES" in suite
    skip_policy = (ROOT / "tests_cuda" / "conftest.py").read_text(encoding="utf-8")
    assert "report.skipped" in skip_policy
    assert "session.exitstatus = pytest.ExitCode.TESTS_FAILED" in skip_policy


def test_dependency_schedule_covers_each_declared_edge_and_next_major_probe():
    workflow = _read("dependency-constraints.yml")
    for case in (
        "python310-direct-floor",
        "captum-minimum",
        "captum-current",
        "shap-xgboost-floor",
        "shap-xgboost-current",
        "python313-latest",
    ):
        assert f"case: {case}" in workflow
    assert "schedule:" in workflow
    assert "select_dependency_prerelease.py" in workflow
    assert "tests/test_localisation_accuracy.py" in workflow
    assert "tests/test_lime_accuracy.py" in workflow
    assert workflow.count("tests/reference/test_ref_deeplift.py") == 2
    assert "import captum" in workflow
    assert "python -m build" in workflow
    assert "python scripts/execute_tutorials.py" in workflow


def test_required_context_policy_matches_new_p0_and_preserved_p1_gates():
    policy = json.loads((ROOT / ".github" / "release-control-policy.json").read_text())
    required = set(policy["required_checks"])
    assert {
        "Artifact byte reproducibility",
        "CUDA single-GPU (Torch minimum)",
        "CUDA single-GPU (Torch latest)",
        "Compatibility-latest (macos-15, Python 3.12)",
        "Quantus reference parity (zero skips)",
    } <= required
    assert policy["admin_snapshot_principals"] == ["jemsbhai"]
    assert set(policy["cuda_evidence"]["required_jobs"]) == {
        "CUDA single-GPU (Torch minimum)",
        "CUDA single-GPU (Torch latest)",
        "CUDA two-GPU scheduled (Torch minimum)",
        "CUDA two-GPU scheduled (Torch latest)",
    }


def test_quantus_job_installs_every_collection_dependency_and_runs_exact_marker():
    workflow = _read("python-ci.yml")
    quantus_job = workflow.split("  quantus-reference:", 1)[1].split("\n  base-install:", 1)[0]
    assert '--editable ".[all]"' in quantus_job
    assert '"quantus>=0.6,<0.7"' in quantus_job
    assert '"grad-cam>=1.5.5,<2"' in quantus_job
    assert "validate_quantus_partition.py" in quantus_job
    assert "-m quantus_reference" in quantus_job


def test_release_runbook_records_legacy_incident_without_fabricating_recovery():
    runbook = (ROOT / "docs" / "RELEASE_OPERATIONS.md").read_text(encoding="utf-8")
    assert "b1b98dfdfc0acbc8dc2113d8db87d40ae9cec2f958ed25b00bc6e30d43db41d4" in runbook
    assert "e2ab525f720d9970f25c307be84b9a5a6bb5feb612a4457ba9d72925cf2af68b" in runbook
    assert "unsigned" in runbook
    assert "cannot recreate original provenance" in runbook
