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
    assert "defaults:\n  run:\n    shell: bash" in workflow
    assert "admin_snapshot_base64:" in workflow
    assert "cuda_run_id:" in workflow
    assert "jobs?filter=all&per_page=100" in workflow
    assert "gh api --paginate" in workflow
    assert "release_external_controls.py bind" in workflow
    assert "--cuda-run-json release-preflight/cuda-run.json" in workflow
    assert "--cuda-jobs-json release-preflight/cuda-jobs.json" in workflow
    assert "admin-capture.json" in workflow
    assert "(cd release-preflight && sha256sum cuda-run.json cuda-jobs.json" in workflow
    assert "(cd release-preflight && sha256sum admin-capture.json" in workflow
    assert "actions/attest-build-provenance@" in workflow
    assert "GH_TOKEN: ${{ github.token }}" in workflow
    assert "GITHUB_RUN_ATTEMPT" in workflow
    assert '"$GITHUB_ACTOR" != "$GITHUB_TRIGGERING_ACTOR"' in workflow
    assert "if: always()" in workflow


def test_publish_cannot_build_without_preflight_and_real_cuda_edges():
    workflow = _read("publish-pypi.yml")
    lowered = workflow.lower()
    assert "preflight_run_id:" in workflow
    assert "needs: [preflight, cuda-release]" in workflow
    assert "gh attestation verify release-preflight/artifact/external-controls.json" in workflow
    assert "release-preflight.yml@refs/heads/main" in workflow
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
    assert "skip-existing" not in lowered
    for forbidden in ("${{ secrets.", "password:", "pypi_api_token", "__token__", "user:"):
        assert forbidden not in lowered
    publisher = workflow.split("  publish:", 1)[1].split("\n  github-release:", 1)[0]
    assert "attestations: true" in publisher
    assert "environment:\n      name: pypi" in publisher
    assert "id-token: write" in publisher
    assert "create_release_governance_record.py" in workflow
    assert "external-controls.json" in workflow
    assert "--notes-file provenance/RELEASE_GOVERNANCE.md" in workflow
    assert "--draft" in workflow and "--draft=false" in workflow
    assert "release-verification/pypi.json" in workflow
    assert "release-verification/draft-assets" in workflow
    assert "release-verification/final-assets" in workflow
    assert workflow.count("verify_release_recovery.py artifacts") == 2
    assert "'.immutable' release-verification/final-release.json" in workflow
    assert (
        'gh api "repos/$GITHUB_REPOSITORY/releases/tags/$RELEASE_TAG" \\\n'
        '            -H "X-GitHub-Api-Version: 2026-03-10"'
    ) in workflow
    assert "Archive normal-path release verification evidence" in workflow
    assert "if: always() && inputs.stage_recovery_drill == false" in workflow
    assert "retention-days: 90" in workflow
    assert "verify_pypi_provenance.py" in workflow
    assert "--output-dir release-verification/pypi-provenance" in workflow
    assert "pypi-attestations verify pypi" in workflow
    assert '--repository "https://github.com/$GITHUB_REPOSITORY"' in workflow
    assert "--provenance-file" in workflow
    assert "pypi-attestations-version.txt" in workflow
    assert "--require-hashes" in workflow
    assert "defaults:\n  run:\n    shell: bash" in workflow
    build_job = workflow.split("  build:", 1)[1].split("\n  attest:", 1)[0]
    assert "actions: read" in build_job and "attestations: read" in build_job
    assert "sha256sum --check admin-capture.json.sha256" in build_job
    assert "sha256sum --check cuda-evidence.sha256" in build_job
    assert "for evidence in" in build_job
    assert "audit_typing_readiness.py --distribution" in build_job
    assert build_job.index("audit_typing_readiness.py --distribution") < build_job.index(
        "Upload immutable distributions"
    )


def test_recovery_is_idempotent_downstream_only_and_hash_checks_all_services():
    workflow = _read("recover-github-release.yml")
    lowered = workflow.lower()
    assert "gh-action-pypi-publish" not in lowered
    assert "twine upload" not in lowered
    assert "skip-existing" not in lowered
    assert "verify_release_recovery.py source-run" in workflow
    assert "require_staged_drill:" in workflow
    assert "--require-staged-drill" in workflow
    assert "jobs?filter=all&per_page=100" in workflow
    assert "gh api --paginate" in workflow
    assert 'gh attestation verify "$artifact"' in workflow
    assert "publish-pypi.yml@refs/tags/$RELEASE_TAG" in workflow
    assert "https://pypi.org/pypi/explainiverse/$version/json" in workflow
    assert "--github-assets recovery/verified-release-assets" in workflow
    assert "--provenance provenance" in workflow
    assert "--draft=false" in workflow
    assert "final-github-assets.sha256" in workflow
    assert "final-github-release.json" in workflow
    assert "Archive complete or partial recovery evidence" in workflow
    assert "if: always()" in workflow
    assert "--notes-file provenance/RELEASE_GOVERNANCE.md" in workflow
    assert "recovery draft omitted the original governance disclosure" in workflow
    assert "'.immutable' recovery/final-github-release-api.json" in workflow
    assert (
        'gh api "repos/$GITHUB_REPOSITORY/releases/tags/$RELEASE_TAG" \\\n'
        '            -H "X-GitHub-Api-Version: 2026-03-10"'
    ) in workflow
    assert "defaults:\n  run:\n    shell: bash" in workflow
    assert "verify_pypi_provenance.py" in workflow
    assert "--output-dir recovery/pypi-provenance" in workflow
    assert "pypi-attestations verify pypi" in workflow
    assert "pypi-attestations-version.txt" in workflow
    assert "--require-hashes" in workflow


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
    assert workflow.count('EXPLAINIVERSE_ENFORCE_CUDA_MANIFEST: "1"') == 2
    assert "pytest.skip" not in suite
    assert "pytest.importorskip" not in suite
    assert "torch.cuda.device_count() == REQUIRED_CUDA_DEVICES" in suite
    assert 'CUDA_VISIBLE_DEVICES: "0"' in workflow
    assert 'CUDA_VISIBLE_DEVICES: "0,1"' in workflow
    assert workflow.count("Require an approved Linux GPU runner") == 2
    assert "defaults:\n  run:\n    shell: bash" in workflow
    skip_policy = (ROOT / "tests_cuda" / "conftest.py").read_text(encoding="utf-8")
    assert "report.skipped" in skip_policy
    assert "session.exitstatus = pytest.ExitCode.TESTS_FAILED" in skip_policy
    publish = _read("publish-pypi.yml")
    assert 'EXPLAINIVERSE_ENFORCE_CUDA_MANIFEST: "1"' in publish
    publish_cuda = publish.split("  cuda-release:", 1)[1].split("\n  build:", 1)[0]
    assert "Require an approved Linux GPU runner" in publish_cuda
    assert 'CUDA_VISIBLE_DEVICES: "0"' in publish_cuda


def test_cuda_cam_matrix_uses_each_family_valid_target_contract():
    suite = (ROOT / "tests_cuda" / "test_cuda_release.py").read_text(encoding="utf-8")
    assert "def _vector_classifier" in suite
    assert "model = nn.Sequential(" in suite
    assert "class _VectorClassifier" not in suite
    assert '("explainer_type", "target_class")' in suite
    assert "(EigenCAMExplainer, None)" in suite
    for explainer in (
        "GradCAMExplainer",
        "HiResCAMExplainer",
        "XGradCAMExplainer",
        "LayerCAMExplainer",
        "ScoreCAMExplainer",
        "EigenGradCAMExplainer",
        "GradCAMElementWiseExplainer",
        "AblationCAMExplainer",
    ):
        assert f"({explainer}, 1)" in suite
    assert ".explain(image, target_class=target_class)" in suite


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
    assert "push:" in workflow and "pull_request:" in workflow
    assert (
        "if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'"
        in workflow
    )
    assert "scikit-image 1.x prerelease availability (blocking monitor)" in workflow
    assert "scikit-image 1.x prerelease compatibility proof" in workflow
    assert "select_dependency_prerelease.py" in workflow
    assert '--metadata "$EVIDENCE_DIR/pypi-metadata.json"' in workflow
    assert "pypi-metadata.sha256" in workflow
    assert "pypi_last_serial" in workflow
    assert 'else "blocked-no-candidate"' in workflow
    assert 'else "discovery-error"' in workflow
    assert "Fail closed while no candidate exists" in workflow
    assert "absence is blocked, not a successful probe" in workflow
    assert "needs.scikit-image-next-major-discovery.result == 'success'" in workflow
    assert "needs.scikit-image-next-major-discovery.outputs.available == 'true'" in workflow
    assert "tests/test_localisation_accuracy.py" in workflow
    assert "tests/test_lime_accuracy.py" in workflow
    assert workflow.count("tests/reference/test_ref_deeplift.py") == 2
    assert "import captum" in workflow
    assert "python -m build" in workflow
    assert "python scripts/execute_tutorials.py" in workflow
    assert "pip-freeze.txt" in workflow
    assert "pre-candidate-pip-check.txt" in workflow
    assert "post-candidate-pip-check.txt" in workflow
    assert "python -m pip uninstall --yes explainiverse" in workflow
    assert "PYTHONPATH: ${{ github.workspace }}/src" in workflow
    assert '--no-deps "scikit-image==$CANDIDATE"' not in workflow
    assert "distribution-sha256.txt" in workflow
    assert "--junitxml artifacts/scikit-image-prerelease/pytest.xml" in workflow
    assert "Archive candidate compatibility proof" in workflow


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
    dependency_contexts = {
        "Dependency constraints (python310-direct-floor)",
        "Dependency constraints (captum-minimum)",
        "Dependency constraints (captum-current)",
        "Dependency constraints (shap-xgboost-floor)",
        "Dependency constraints (shap-xgboost-current)",
        "Dependency constraints (python313-latest)",
    }
    assert dependency_contexts <= required
    assert policy["required_check_provider"] == {"app_id": 15368, "slug": "github-actions"}
    assert set(policy["required_check_workflows"]) == required
    assert all(
        value["path"].startswith(".github/workflows/")
        and value["head_branch"] == "main"
        and value["event"] in {"push", "workflow_dispatch"}
        for value in policy["required_check_workflows"].values()
    )
    assert policy["immutable_releases"] == {"enabled": True}
    assert policy["admin_snapshot_principals"] == ["jemsbhai"]
    assert set(policy["cuda_evidence"]["required_jobs"]) == {
        "CUDA single-GPU (Torch minimum)",
        "CUDA single-GPU (Torch latest)",
        "CUDA two-GPU scheduled (Torch minimum)",
        "CUDA two-GPU scheduled (Torch latest)",
    }


def test_pypi_cryptographic_verifier_is_pinned_in_the_hash_locked_release_graph():
    direct = (ROOT / ".github" / "requirements" / "release-tools.in").read_text()
    locked = (ROOT / ".github" / "requirements" / "release-tools.txt").read_text()
    assert "pypi-attestations==0.0.30" in direct
    pin_line = next(line for line in locked.splitlines() if line.startswith("pypi-attestations=="))
    assert pin_line == "pypi-attestations==0.0.30 " + "\\"
    package_block = locked[locked.index(pin_line) :].split("\n\n", 1)[0]
    assert "--hash=sha256:" in package_block


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
