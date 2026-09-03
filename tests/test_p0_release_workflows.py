"""Structural security contracts for P0 release and compatibility workflows."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def _read(name):
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def _assert_cuda_runner_routing_contract(workflow, publish):
    policy = json.loads(
        (ROOT / ".github" / "release-control-policy.json").read_text(encoding="utf-8")
    )
    required_labels = policy["cuda_evidence"]["required_runner_labels"]
    single_label = required_labels["CUDA single-GPU (Torch latest)"]
    assert required_labels["CUDA single-GPU (Torch minimum)"] == single_label
    two_label = required_labels["CUDA two-GPU scheduled (Torch latest)"]
    assert required_labels["CUDA two-GPU scheduled (Torch minimum)"] == two_label

    routing = workflow.split("  cuda-runner-routing:", 1)[1].split("\n  single-gpu:", 1)[0]
    single = workflow.split("  single-gpu:", 1)[1].split("\n  two-gpu:", 1)[0]
    two = workflow.split("  two-gpu:", 1)[1]
    assert workflow.index("  cuda-runner-routing:") < workflow.index("  single-gpu:")
    assert "runs-on: ubuntu-latest" in routing
    assert "CUDA_SINGLE_RUNNER: ${{ vars.CUDA_SINGLE_RUNNER }}" in routing
    assert "CUDA_TWO_RUNNER: ${{ vars.CUDA_TWO_RUNNER }}" in routing
    assert f'[[ "$CUDA_SINGLE_RUNNER" != "{single_label}" ]]' in routing
    assert f'[[ "$CUDA_TWO_RUNNER" != "{two_label}" ]]' in routing
    assert '"schedule" || "$GITHUB_EVENT_NAME" == "workflow_dispatch"' in routing
    assert routing.count("exit 1") == 2
    assert "continue-on-error" not in routing
    assert "needs: cuda-runner-routing" in single
    assert "if: ${{ always() }}" in single
    assert (
        "${{ needs.cuda-runner-routing.result == 'success' &&\n"
        f"      '{single_label}' || 'ubuntu-latest' }}" in single
    )
    single_reporter = single.split(
        "      - name: Fail the required check when CUDA routing is rejected", 1
    )[1].split("\n      - name: Checkout", 1)[0]
    single_steps = single.split("\n    steps:", 1)[1]
    assert single_steps.index("Fail the required check when CUDA routing is rejected") < (
        single_steps.index("      - name: Checkout")
    )
    assert "if: needs.cuda-runner-routing.result != 'success'" in single_reporter
    assert single_reporter.count("exit 1") == 1
    assert "continue-on-error" not in single_reporter
    assert "if: always()" not in single_steps
    assert "needs: cuda-runner-routing" in two
    assert "always() &&" in two
    assert (
        "${{ needs.cuda-runner-routing.result == 'success' &&\n"
        f"      '{two_label}' || 'ubuntu-latest' }}" in two
    )
    two_reporter = two.split(
        "      - name: Fail the required check when CUDA routing is rejected", 1
    )[1].split("\n      - name: Checkout", 1)[0]
    two_steps = two.split("\n    steps:", 1)[1]
    assert two_steps.index("Fail the required check when CUDA routing is rejected") < (
        two_steps.index("      - name: Checkout")
    )
    assert "if: needs.cuda-runner-routing.result != 'success'" in two_reporter
    assert two_reporter.count("exit 1") == 1
    assert "continue-on-error" not in two_reporter
    assert "if: always()" not in two_steps

    preflight = publish.split("  preflight:", 1)[1].split("\n  cuda-release:", 1)[0]
    publish_cuda = publish.split("  cuda-release:", 1)[1].split("\n  build:", 1)[0]
    assert "Require exact reviewed single-GPU runner routing" in preflight
    publish_routing = preflight.split(
        "      - name: Require exact reviewed single-GPU runner routing", 1
    )[1].split("\n      - name:", 1)[0]
    assert "if: steps.verify.outputs.cuda_mode == 'hardware_evidence'" in publish_routing
    assert "CUDA_SINGLE_RUNNER: ${{ vars.CUDA_SINGLE_RUNNER }}" in publish_routing
    assert f'[[ "$CUDA_SINGLE_RUNNER" != "{single_label}" ]]' in publish_routing
    assert publish_routing.count("exit 1") == 1
    assert "continue-on-error" not in publish_routing
    assert "needs: preflight" in publish_cuda
    assert f"runs-on: {single_label}" in publish_cuda
    assert "ubuntu-latest" not in publish_cuda


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
    assert "cuda_exception_id:" in workflow
    assert workflow.count('default: ""') >= 2
    assert "supply exactly one of cuda_run_id or cuda_exception_id" in workflow
    assert '"EXPLAINIVERSE-v0.15.0-CPU-ONLY"' in workflow
    assert '"$RELEASE_TAG" != "v0.15.0"' in workflow
    assert "jobs?filter=all&per_page=100" in workflow
    assert "gh api --paginate" in workflow
    assert "if: inputs.cuda_run_id != ''" in workflow
    assert "release_external_controls.py bind" in workflow
    assert "--cuda-run-json release-preflight/cuda-run.json" in workflow
    assert "--cuda-jobs-json release-preflight/cuda-jobs.json" in workflow
    assert '--cuda-exception-id "$CUDA_EXCEPTION_ID"' in workflow
    assert "--cuda-gate-output release-preflight/cuda-release-gate.json" in workflow
    assert '--github-output "$GITHUB_OUTPUT"' in workflow
    assert "admin-capture.json" in workflow
    assert "(cd release-preflight && sha256sum cuda-run.json cuda-jobs.json" in workflow
    assert "(cd release-preflight && sha256sum admin-capture.json" in workflow
    assert "actions/attest-build-provenance@" in workflow
    assert "subject-path: release-preflight/*" in workflow
    assert "CUDA release status: NOT RUN" in workflow
    assert "does not claim CUDA or multi-GPU validation" in workflow
    assert "GH_TOKEN: ${{ github.token }}" in workflow
    assert "GITHUB_RUN_ATTEMPT" in workflow
    assert '"$GITHUB_ACTOR" != "$GITHUB_TRIGGERING_ACTOR"' in workflow
    assert "if: always()" in workflow


def test_publish_requires_verified_hardware_or_the_exact_cpu_only_exception():
    workflow = _read("publish-pypi.yml")
    lowered = workflow.lower()
    assert "preflight_run_id:" in workflow
    assert "cuda_exception_id:" in workflow
    assert "supply exactly one of cuda_run_id or cuda_exception_id" in workflow
    assert '"EXPLAINIVERSE-v0.15.0-CPU-ONLY"' in workflow
    assert '"$RELEASE_TAG" != "v0.15.0"' in workflow
    assert "needs: [preflight, cuda-release]" in workflow
    assert "cuda_mode: ${{ steps.verify.outputs.cuda_mode }}" in workflow
    assert "cuda_run_id: ${{ steps.verify.outputs.cuda_run_id }}" in workflow
    assert "cuda_exception_id: ${{ steps.verify.outputs.cuda_exception_id }}" in workflow
    assert "gh attestation verify release-preflight/artifact/external-controls.json" in workflow
    assert "release-preflight.yml@refs/heads/main" in workflow
    assert "release_external_controls.py verify" in workflow
    assert "Release CUDA single-GPU (Torch ${{ matrix.torch-edge }}, zero skips)" in workflow
    assert 'EXPLAINIVERSE_REQUIRED_CUDA_DEVICES: "1"' in workflow
    assert "needs: preflight" in workflow
    cuda_job = workflow.split("  cuda-release:", 1)[1].split("\n  build:", 1)[0]
    assert "if: needs.preflight.outputs.cuda_mode == 'hardware_evidence'" in cuda_job
    assert "cpu_only_exception" not in cuda_job
    assert "continue-on-error" not in cuda_job
    build_job = workflow.split("  build:", 1)[1].split("\n  attest:", 1)[0]
    assert "always() &&" in build_job
    assert "needs.preflight.result == 'success'" in build_job
    assert "needs.preflight.outputs.cuda_mode == 'hardware_evidence'" in build_job
    assert "needs.cuda-release.result == 'success'" in build_job
    assert "needs.preflight.outputs.cuda_mode == 'cpu_only_exception'" in build_job
    assert "needs.cuda-release.result == 'skipped'" in build_job
    assert (
        "needs.preflight.outputs.cuda_exception_id == "
        "'EXPLAINIVERSE-v0.15.0-CPU-ONLY'" in build_job
    )
    assert "inputs.cuda_exception_id" not in build_job
    assert "inputs.cuda_run_id" not in build_job
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
    assert "'.draft == false and .prerelease == false and .immutable == true'" in workflow
    assert (
        'gh api "repos/$GITHUB_REPOSITORY/releases/tags/$RELEASE_TAG" \\\n'
        '            -H "X-GitHub-Api-Version: 2026-03-10"'
    ) in workflow
    immutable_release_precondition = (
        'gh api --method GET "repos/$GITHUB_REPOSITORY/immutable-releases" \\\n'
        '            -H "X-GitHub-Api-Version: 2026-03-10" \\\n'
        "            > release-verification/immutable-releases.json\n"
        "          jq -e '.enabled == true' "
        "release-verification/immutable-releases.json > /dev/null\n"
        '          gh release edit "$RELEASE_TAG" --repo "$GITHUB_REPOSITORY" --draft=false'
    )
    assert immutable_release_precondition in workflow
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
    assert "'.enabled == true' release-verification/immutable-releases.json" in workflow
    assert "actions: read" in build_job and "attestations: read" in build_job
    assert "sha256sum --check admin-capture.json.sha256" in build_job
    assert "sha256sum --check cuda-evidence.sha256" in build_job
    assert "for evidence in" in build_job
    assert "release-source/scripts/audit_typing_readiness.py" in build_job
    assert '--distribution "$artifact"' in build_job
    assert build_job.index("release-source/scripts/audit_typing_readiness.py") < build_job.index(
        "Upload immutable distributions"
    )


def test_publish_derives_cuda_mode_only_after_attested_snapshot_verification():
    workflow = _read("publish-pypi.yml")
    preflight = workflow.split("  preflight:", 1)[1].split("\n  cuda-release:", 1)[0]
    verify = preflight.split(
        "      - name: Bind the verified snapshot and derive the CUDA release mode", 1
    )[1].split("\n      - name:", 1)[0]

    assert "id: verify" in verify
    assert "gh attestation verify release-preflight/artifact/external-controls.json" in verify
    assert verify.index("gh attestation verify") < verify.index(
        "python scripts/release_external_controls.py verify"
    )
    assert '--github-output "$GITHUB_OUTPUT"' in verify
    assert 'cuda_selector=(--cuda-exception-id "$CUDA_EXCEPTION_ID")' in verify
    assert "outputs:\n      cuda_mode: ${{ steps.verify.outputs.cuda_mode }}" in preflight
    assert "Disclose the verified CPU-only release exception" in preflight
    assert "CUDA release status: NOT RUN" in preflight
    assert "does not claim CUDA or multi-GPU validation" in preflight


def test_publish_attests_and_carries_the_selected_cuda_gate_without_fabrication():
    workflow = _read("publish-pypi.yml")
    build_job = workflow.split("  build:", 1)[1].split("\n  attest:", 1)[0]
    governance = build_job.split(
        "      - name: Recreate governance and bind the accepted reproducibility artifacts", 1
    )[1].split("\n      - name:", 1)[0]

    assert "sha256sum --check cuda-release-gate.json.sha256" in governance
    assert "sha256sum --check cuda-evidence.sha256" in governance
    assert 'if [[ "$CUDA_MODE" == "hardware_evidence" ]]' in governance
    assert (
        'elif [[ "$CUDA_MODE" == "cpu_only_exception" && \\\n'
        '                  "$CUDA_EXCEPTION_ID" == "EXPLAINIVERSE-v0.15.0-CPU-ONLY" ]]'
        in governance
    )
    assert 'governance_selector=(--cuda-run-id "$CUDA_RUN_ID")' in governance
    assert 'governance_selector=(--cuda-exception-id "$CUDA_EXCEPTION_ID")' in governance
    assert 'for evidence in "${evidence[@]}"' in governance
    assert '"${governance_selector[@]}"' in governance
    assert "continue-on-error" not in governance


def test_publish_rebuilds_a_clean_tag_and_binds_the_attested_reproducibility_bytes():
    workflow = _read("publish-pypi.yml")
    build_job = workflow.split("  build:", 1)[1].split("\n  attest:", 1)[0]

    candidate_gates = build_job.index("Run the complete experimental JavaScript gate")
    reserved_paths = build_job.index("Refuse release inputs created by candidate-authored gates")
    clean_checkout = build_job.index("Check out a clean release source")
    clean_reverification = build_job.index("Reverify the clean signed release source")
    bind_reproducibility = build_job.index("bind the accepted reproducibility artifacts")
    clean_build = build_job.index("Build once from the clean release checkout")
    assert (
        candidate_gates
        < reserved_paths
        < clean_checkout
        < clean_reverification
        < bind_reproducibility
        < clean_build
    )

    assert 'if [[ -e "$path" || -L "$path" ]]' in build_job
    assert "path: release-source" in build_job
    assert (
        build_job.count(
            'test -z "$(git -C release-source status --porcelain --untracked-files=all)"'
        )
        == 2
    )
    assert "python release-source/scripts/create_release_governance_record.py" in build_job
    assert (
        "cd release-source\n            python scripts/record_release_environment.py" in build_job
    )
    assert "--requirements .github/requirements/release-tools.txt" in build_job
    assert '--output "$GITHUB_WORKSPACE/provenance/release-environment.json"' in build_job

    assert "provenance/reproducibility-expected-run.json" in build_job
    assert "if len(matches) != 1" in build_job
    assert "if set(expected_run) != expected_fields" in build_job
    for field in (
        "id",
        "repository",
        "path",
        "event",
        "head_branch",
        "head_sha",
        "status",
        "conclusion",
        "run_attempt",
    ):
        assert f'"{field}": actual.get("{field}")' in build_job or field == "repository"
    assert 'repository.get("full_name") if isinstance(repository, dict) else None' in build_job
    assert "if actual_normalized != expected" in build_job
    assert "live reproducibility run differs from the attested check run" in build_job
    assert 'gh api "repos/$GITHUB_REPOSITORY/actions/runs/$reproducibility_run_id"' in build_job
    assert build_job.count('gh run download "$reproducibility_run_id"') == 3

    for name, destination in (
        ("reproducibility-one", "reproducibility-proof/one"),
        ("reproducibility-two", "reproducibility-proof/two"),
        ("reproducibility-report", "reproducibility-proof/report"),
    ):
        assert f"--name {name} --dir {destination}" in build_job
    assert "reproducibility-proof/one/provenance/release-environment.json" in build_job
    assert "reproducibility-proof/two/provenance/release-environment.json" in build_job
    assert "reproducibility-proof/one/dist reproducibility-proof/two/dist" in build_job
    assert "find reproducibility-proof/report -mindepth 1 | wc -l" in build_job
    for source, accepted in (
        ("release-environment-one.json", "accepted-reproducibility-environment-one.json"),
        ("release-environment-two.json", "accepted-reproducibility-environment-two.json"),
        (
            "release-environment-comparison.json",
            "accepted-reproducibility-environment-comparison.json",
        ),
        ("reproducibility.json", "accepted-reproducibility-distributions.json"),
    ):
        assert source in build_job
        expected_copy = (
            f"cp reproducibility-proof/report/{source} \\\n" f"            provenance/{accepted}"
        )
        assert expected_copy in build_job
        assert build_job.index(f"provenance/{accepted}") < build_job.index("Upload hashes and SBOM")
    assert 'test -f "reproducibility-proof/report/$report"' in build_job
    assert 'test ! -L "reproducibility-proof/report/$report"' in build_job
    assert build_job.count("cmp reproducibility-proof/report/") == 4

    assert build_job.count("poetry build") == 1
    assert '(cd release-source && poetry build --output "$GITHUB_WORKSPACE/dist")' in build_job
    assert "dist reproducibility-proof/one/dist" in build_job
    assert "dist reproducibility-proof/two/dist" in build_job


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
    assert "'.isDraft == false and .isPrerelease == false'" in workflow
    assert "'.immutable == true' recovery/final-github-release-api.json" in workflow
    assert "verify_release_recovery.py release-body" in workflow
    assert "--release-json recovery/final-github-release-api.json" in workflow
    assert "--disclosure provenance/RELEASE_GOVERNANCE.md" in workflow
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
    assert workflow.count("Require Linux GPU runner OS") == 2
    assert "defaults:\n  run:\n    shell: bash" in workflow
    skip_policy = (ROOT / "tests_cuda" / "conftest.py").read_text(encoding="utf-8")
    assert "report.skipped" in skip_policy
    assert "session.exitstatus = pytest.ExitCode.TESTS_FAILED" in skip_policy
    publish = _read("publish-pypi.yml")
    assert 'EXPLAINIVERSE_ENFORCE_CUDA_MANIFEST: "1"' in publish
    publish_cuda = publish.split("  cuda-release:", 1)[1].split("\n  build:", 1)[0]
    assert "Require Linux GPU runner OS" in publish_cuda
    assert 'CUDA_VISIBLE_DEVICES: "0"' in publish_cuda


def test_cuda_workflows_route_only_through_exact_reviewed_labels():
    _assert_cuda_runner_routing_contract(_read("cuda-ci.yml"), _read("publish-pypi.yml"))


def test_cuda_runner_routing_contract_rejects_fail_open_drift():
    workflow = _read("cuda-ci.yml")
    publish = _read("publish-pypi.yml")
    mutations = (
        (
            workflow.replace(
                "      'explainiverse-cuda-single' || 'ubuntu-latest'",
                "      'ubuntu-latest' || 'ubuntu-latest'",
                1,
            ),
            publish,
        ),
        (
            workflow.replace(
                "      'explainiverse-cuda-two' || 'ubuntu-latest'",
                "      'explainiverse-cuda-single' || 'ubuntu-latest'",
                1,
            ),
            publish,
        ),
        (workflow.replace("    needs: cuda-runner-routing\n", "", 1), publish),
        (
            workflow.replace(
                '[[ "$CUDA_SINGLE_RUNNER" != "explainiverse-cuda-single" ]]',
                '[[ "$CUDA_SINGLE_RUNNER" != "" ]]',
                1,
            ),
            publish,
        ),
        (workflow.replace("            exit 1\n", "            exit 0\n", 1), publish),
        (
            workflow,
            publish.replace(
                "    runs-on: explainiverse-cuda-single",
                "    runs-on: ubuntu-latest",
                1,
            ),
        ),
        (
            workflow,
            publish.replace(
                '[[ "$CUDA_SINGLE_RUNNER" != "explainiverse-cuda-single" ]]',
                '[[ "$CUDA_SINGLE_RUNNER" != "explainiverse-cuda-two" ]]',
                1,
            ),
        ),
    )

    for mutated_workflow, mutated_publish in mutations:
        with pytest.raises(AssertionError):
            _assert_cuda_runner_routing_contract(mutated_workflow, mutated_publish)


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
    assert "scikit-image 1.x prerelease source-compatibility probe" in workflow
    assert "select_dependency_prerelease.py" in workflow
    assert "--current-major 0" in workflow
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
    assert "python scripts/execute_tutorials.py --source-only" in workflow
    assert "pip-freeze.txt" in workflow
    assert "pre-candidate-pip-check.txt" in workflow
    assert "post-candidate-dependencies-pip-check.txt" in workflow
    assert "post-candidate-pip-check.txt" not in workflow
    assert "python -m pip uninstall --yes explainiverse" in workflow
    assert "PYTHONPATH: ${{ github.workspace }}/src" in workflow
    assert '--no-deps "scikit-image==$CANDIDATE"' not in workflow
    assert "distribution-sha256.txt" in workflow
    assert workflow.count("python scripts/record_dependency_candidate_probe.py") == 2
    assert "candidate-source-probe.json" in workflow
    assert "candidate-wheel-metadata.json" in workflow
    assert '--wheel "${wheels[0]}"' in workflow
    assert 'record["source_probe_status"] = "source-probe-passed"' in workflow
    assert 'record["source_probe_commit"] = os.environ["GITHUB_SHA"]' in workflow
    assert "candidate-boundary.sha256" in workflow
    assert "--junitxml artifacts/scikit-image-prerelease/pytest.xml" in workflow
    assert "Archive candidate source-compatibility probe" in workflow


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


def _assert_macos_openmp_contract(workflow):
    compatibility_job = workflow.split("  test:", 1)[1].split(
        "\n  minimum-direct-dependencies:", 1
    )[0]
    arm_check = compatibility_job.index("Require the advertised macOS ARM64 runner")
    openmp = compatibility_job.index("Provision the XGBoost OpenMP runtime on macOS")
    dependency_install = compatibility_job.index(
        "Install all package extras, tests, and tutorial runner dependencies"
    )
    runtime_binding = compatibility_job.index("Use PyTorch's single OpenMP runtime on macOS")
    dependency_import = compatibility_job.index("Require every accuracy-reference dependency")
    coexistence_probe = compatibility_job.index(
        "Prove PyTorch and XGBoost OpenMP coexistence on macOS"
    )
    full_tests = compatibility_job.index("Run all-extras tests")

    assert (
        arm_check
        < openmp
        < dependency_install
        < runtime_binding
        < dependency_import
        < coexistence_probe
        < full_tests
    )
    openmp_step = compatibility_job.split("Provision the XGBoost OpenMP runtime on macOS", 1)[
        1
    ].split("\n      - name:", 1)[0]
    assert "if: matrix.os == 'macos-15'" in openmp_step
    assert 'HOMEBREW_NO_AUTO_UPDATE: "1"' in openmp_step
    assert "brew install libomp" in openmp_step
    assert 'test -f "$(brew --prefix libomp)/lib/libomp.dylib"' in openmp_step
    assert "brew list --versions libomp" in openmp_step

    runtime_step = compatibility_job.split(
        "      - name: Use PyTorch's single OpenMP runtime on macOS", 1
    )[1].split("\n      - name:", 1)[0]
    assert "if: matrix.os == 'macos-15'" in runtime_step
    assert "Path(torch.__file__).resolve().parent / 'lib'" in runtime_step
    assert 'test -f "$torch_lib_dir/libomp.dylib"' in runtime_step
    assert 'echo "DYLD_LIBRARY_PATH=$torch_lib_dir" >> "$GITHUB_ENV"' in runtime_step

    coexistence = compatibility_job.split(
        "      - name: Prove PyTorch and XGBoost OpenMP coexistence on macOS", 1
    )[1].split("\n      - name:", 1)[0]
    assert "if: matrix.os == 'macos-15'" in coexistence
    assert '"import torch, xgboost;' in coexistence
    assert "torch.nn.functional.cross_entropy" in coexistence
    assert "loss.backward()" in coexistence


def test_macos_arm_job_provisions_xgboost_openmp_before_dependency_import():
    workflow = _read("python-ci.yml")
    _assert_macos_openmp_contract(workflow)

    reference_fixtures = (ROOT / "tests" / "reference" / "conftest.py").read_text(encoding="utf-8")
    assert reference_fixtures.index("    import torch") < reference_fixtures.index(
        "import xgboost as xgb"
    )


def test_macos_openmp_contract_rejects_wrong_platform_or_missing_runtime_proof():
    workflow = _read("python-ci.yml")
    mutations = (
        workflow.replace(
            "      - name: Provision the XGBoost OpenMP runtime on macOS\n"
            "        if: matrix.os == 'macos-15'",
            "      - name: Provision the XGBoost OpenMP runtime on macOS\n"
            "        if: matrix.os == 'ubuntu-latest'",
            1,
        ),
        workflow.replace("          brew install libomp\n", "          true\n", 1),
        workflow.replace(
            '          test -f "$(brew --prefix libomp)/lib/libomp.dylib"\n',
            "",
            1,
        ),
        workflow.replace(
            '          test -f "$torch_lib_dir/libomp.dylib"\n',
            "",
            1,
        ),
        workflow.replace(
            '          echo "DYLD_LIBRARY_PATH=$torch_lib_dir" >> "$GITHUB_ENV"\n',
            '          echo "DYLD_LIBRARY_PATH=$(brew --prefix libomp)/lib" >> "$GITHUB_ENV"\n',
            1,
        ),
    )

    for mutated in mutations:
        with pytest.raises(AssertionError):
            _assert_macos_openmp_contract(mutated)


def _assert_quantus_job_contract(workflow):
    quantus_job = workflow.split("  quantus-reference:", 1)[1].split("\n  base-install:", 1)[0]
    manifest = ROOT / ".github" / "constraints" / "quantus-reference-tests.txt"
    manifest_entries = [
        line.strip()
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert len(manifest_entries) == 9
    assert manifest_entries == sorted(set(manifest_entries))
    assert '--editable ".[all]"' in quantus_job
    assert '"quantus>=0.6,<0.7"' in quantus_job
    assert '"grad-cam>=1.5.5,<2"' in quantus_job
    assert "validate_quantus_partition.py" in quantus_job
    assert ".github/constraints/quantus-reference-tests.txt" in quantus_job
    assert "mapfile -t quantus_tests" in quantus_job
    assert "grep -Ev '^[[:space:]]*(#|$)'" in quantus_job
    assert 'test "${#quantus_tests[@]}" -eq 9' in quantus_job
    assert '"${quantus_tests[@]}"' in quantus_job
    assert quantus_job.index("validate_quantus_partition.py") < quantus_job.index(
        "mapfile -t quantus_tests"
    )
    pytest_command = quantus_job.split("python -m pytest", 1)[1]
    assert "-m quantus_reference" in pytest_command
    assert '"${quantus_tests[@]}"' in pytest_command


def test_quantus_job_runs_only_the_exact_fail_closed_reference_manifest():
    _assert_quantus_job_contract(_read("python-ci.yml"))


def test_quantus_job_contract_rejects_full_collection_or_weakened_manifest_count():
    workflow = _read("python-ci.yml")
    mutations = (
        workflow.replace(
            ".github/constraints/quantus-reference-tests.txt",
            "tests",
            1,
        ),
        workflow.replace(
            '          test "${#quantus_tests[@]}" -eq 9',
            '          test "${#quantus_tests[@]}" -ge 1',
            1,
        ),
        workflow.replace(
            '            "${quantus_tests[@]}"',
            "            tests",
            1,
        ),
    )

    for mutated in mutations:
        with pytest.raises(AssertionError):
            _assert_quantus_job_contract(mutated)


def test_release_runbook_records_legacy_incident_without_fabricating_recovery():
    runbook = (ROOT / "docs" / "RELEASE_OPERATIONS.md").read_text(encoding="utf-8")
    assert "b1b98dfdfc0acbc8dc2113d8db87d40ae9cec2f958ed25b00bc6e30d43db41d4" in runbook
    assert "e2ab525f720d9970f25c307be84b9a5a6bb5feb612a4457ba9d72925cf2af68b" in runbook
    assert "unsigned" in runbook
    assert "cannot recreate original provenance" in runbook


def test_release_runbook_dispatches_recovery_from_the_tag_and_defers_to_assets():
    runbook = (ROOT / "docs" / "RELEASE_OPERATIONS.md").read_text(encoding="utf-8")
    assert "gh workflow run recover-github-release.yml --ref $releaseTag" in runbook
    assert "release notes remain mutable" in runbook
    assert "governance assets are authoritative" in runbook
