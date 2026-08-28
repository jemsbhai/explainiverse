"""Structural security contracts for P0 release and compatibility workflows."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
PUBLISH_JOB_IDS = (
    "preflight",
    "cuda-release",
    "build",
    "attest",
    "publish",
    "github-release",
)
EXPECTED_RELEASE_PROVENANCE = {
    "RELEASE_GOVERNANCE.json",
    "RELEASE_GOVERNANCE.md",
    "SHA256SUMS",
    "accepted-reproducibility-distributions.json",
    "accepted-reproducibility-environment-comparison.json",
    "accepted-reproducibility-environment-one.json",
    "accepted-reproducibility-environment-two.json",
    "admin-capture.json",
    "admin-capture.json.sha256",
    "bound-reproducibility-distributions.json",
    "bound-reproducibility-environment.json",
    "cuda-evidence.sha256",
    "cuda-jobs.json",
    "cuda-run.json",
    "explainiverse-build.cdx.json",
    "external-controls.json",
    "external-controls.json.sha256",
    "preflight-source-run.json",
    "publish-vs-reproducibility-one.json",
    "publish-vs-reproducibility-two.json",
    "release-environment.json",
    "reproducibility-expected-run.json",
    "reproducibility-source-run.json",
}


def _read(name):
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def _assert_exact_provenance_inventories(workflow, expected_occurrences):
    marker = "expected_provenance=$(printf '%s\\n' \\\n"
    segments = workflow.split(marker)[1:]
    assert len(segments) == expected_occurrences
    for segment in segments:
        inventory, remainder = segment.split("actual_provenance=", 1)
        names = set()
        for raw_line in inventory.splitlines():
            line = raw_line.strip()
            if line.endswith("\\"):
                name = line[:-1].strip()
            elif line.endswith("| sort)"):
                name = line[: -len("| sort)")].strip()
            else:
                continue
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", name):
                names.add(name)
        assert names == EXPECTED_RELEASE_PROVENANCE
        assert 'test "$actual_provenance" = "$expected_provenance"' in remainder


def _cuda_routing_guard_script(workflow):
    routing = workflow.split("  cuda-runner-routing:", 1)[1].split("\n  single-gpu:", 1)[0]
    script = routing.split("          python3 - <<'PY'\n", 1)[1].split("\n          PY", 1)[0]
    return textwrap.dedent(script)


def _publish_front_door_guard_script(workflow):
    preflight = workflow.split("  preflight:", 1)[1].split("\n  cuda-release:", 1)[0]
    guard = preflight.split(
        "      - name: Require an exact first-attempt owner tag publication",
        1,
    )[1].split("\n      - name:", 1)[0]
    script = guard.split("          python3 - <<'PY'\n", 1)[1].split("\n          PY", 1)[0]
    return textwrap.dedent(script)


def _publish_immutable_source_gate(workflow):
    preflight = workflow.split("  preflight:", 1)[1].split("\n  cuda-release:", 1)[0]
    return preflight.split(
        "      - name: Verify the signed immutable release source before external work",
        1,
    )[1].split("\n      - name:", 1)[0]


def _publish_job_block(workflow, job_id):
    marker = f"  {job_id}:\n"
    remainder = workflow.split(marker, 1)[1]
    next_job = re.search(r"^  [a-z][a-z-]+:\n", remainder, flags=re.MULTILINE)
    return remainder if next_job is None else remainder[: next_job.start()]


def _assert_publish_jobs_are_first_attempt_only(workflow):
    exact_guard = "    if: ${{ github.run_attempt == 1 }}"
    for job_id in PUBLISH_JOB_IDS:
        block = _publish_job_block(workflow, job_id)
        guards = re.findall(r"^    if:.*$", block, flags=re.MULTILINE)
        assert guards == [exact_guard], job_id
        assert block.index(exact_guard) < block.index("    runs-on:"), job_id
        if "    needs:" in block:
            assert block.index("    needs:") < block.index(exact_guard), job_id


def _apply_environment_overrides(environment, overrides):
    for name, value in overrides.items():
        if value is None:
            environment.pop(name, None)
        else:
            environment[name] = value


def _run_cuda_routing_guard(workflow, **overrides):
    environment = os.environ.copy()
    environment.update(
        {
            "GITHUB_REPOSITORY": "jemsbhai/explainiverse",
            "GITHUB_REPOSITORY_OWNER": "jemsbhai",
            "GITHUB_ACTOR": "jemsbhai",
            "GITHUB_TRIGGERING_ACTOR": "jemsbhai",
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "GITHUB_REF": "refs/heads/codex/reviewed-cuda-candidate",
            "GITHUB_RUN_ATTEMPT": "1",
            "SINGLE_MINIMUM_RUNNER_NONCE": "0000000000000001",
            "SINGLE_LATEST_RUNNER_NONCE": "0000000000000002",
            "TWO_MINIMUM_RUNNER_NONCE": "0000000000000003",
            "TWO_LATEST_RUNNER_NONCE": "0000000000000004",
        }
    )
    _apply_environment_overrides(environment, overrides)
    return subprocess.run(
        [sys.executable, "-c", _cuda_routing_guard_script(workflow)],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )


def _run_publish_front_door_guard(workflow, **overrides):
    release_tag = overrides.get("RELEASE_TAG", "v0.15.0")
    environment = os.environ.copy()
    environment.update(
        {
            "GITHUB_REPOSITORY": "jemsbhai/explainiverse",
            "GITHUB_REPOSITORY_OWNER": "jemsbhai",
            "GITHUB_ACTOR": "jemsbhai",
            "GITHUB_TRIGGERING_ACTOR": "jemsbhai",
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "GITHUB_REF": f"refs/tags/{release_tag}",
            "GITHUB_RUN_ATTEMPT": "1",
            "RELEASE_TAG": "v0.15.0",
            "STAGE_RECOVERY_DRILL": "true",
            "SINGLE_MINIMUM_RUNNER_NONCE": "0000000000000001",
            "SINGLE_LATEST_RUNNER_NONCE": "0000000000000002",
        }
    )
    _apply_environment_overrides(environment, overrides)
    return subprocess.run(
        [sys.executable, "-c", _publish_front_door_guard_script(workflow)],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )


def _assert_publish_front_door_contract(workflow):
    exact_main_fetch = "git fetch --no-tags origin '+refs/heads/main:refs/remotes/origin/main'"
    assert workflow.count(exact_main_fetch) == 4
    preflight = workflow.split("  preflight:", 1)[1].split("\n  cuda-release:", 1)[0]
    front_door = "Require an exact first-attempt owner tag publication"
    checkout = "Check out the immutable release tag"
    immutable_source = "Verify the signed immutable release source before external work"
    setup_python = "Set up Python 3.12"
    snapshot = "Download and bind the preflight run and attested snapshot"

    assert preflight.index(front_door) < preflight.index("Validate preflight run input")
    assert preflight.split("    steps:\n", 1)[1].lstrip().startswith(f"- name: {front_door}")
    assert preflight.index(checkout) < preflight.index(immutable_source)
    assert preflight.index(immutable_source) < preflight.index(setup_python)
    assert preflight.index(immutable_source) < preflight.index(snapshot)
    post_checkout = preflight.split(f"      - name: {checkout}", 1)[1]
    assert post_checkout.split("\n      - name:", 1)[1].startswith(f" {immutable_source}")

    guard_script = _publish_front_door_guard_script(workflow)
    assert 'repository == "jemsbhai/explainiverse"' in guard_script
    assert 'repository_owner == "jemsbhai"' in guard_script
    assert "actor == repository_owner" in guard_script
    assert "triggering_actor == repository_owner" in guard_script
    assert 'event_name == "workflow_dispatch"' in guard_script
    assert 'run_attempt == "1"' in guard_script
    assert 're.fullmatch(r"v[0-9]+\\.[0-9]+\\.[0-9]+", release_tag)' in guard_script
    assert 'ref == f"refs/tags/{release_tag}"' in guard_script
    assert 'release_tag == "v0.15.0"' in guard_script
    assert 'stage_recovery_drill != "true"' in guard_script
    assert 're.fullmatch(r"[a-f0-9]{16}", value)' in guard_script
    assert "len(set(runner_nonces)) == len(runner_nonces)" in guard_script
    assert "SINGLE_MINIMUM_RUNNER_NONCE" in preflight
    assert "SINGLE_LATEST_RUNNER_NONCE" in preflight
    assert "raise SystemExit(" in guard_script

    immutable_gate = _publish_immutable_source_gate(workflow)
    assert "checkout_sha=$(git rev-parse HEAD)" in immutable_gate
    assert '[[ "$checkout_sha" != "$GITHUB_SHA" ]]' in immutable_gate
    assert '[[ "$(git cat-file -t "$RELEASE_TAG")" != tag ]]' in immutable_gate
    assert 'git rev-parse "$RELEASE_TAG^{commit}"' in immutable_gate
    assert 'git rev-parse "$RELEASE_TAG^{tag}"' in immutable_gate
    assert 'gh api "repos/$GITHUB_REPOSITORY/git/tags/$tag_object"' in immutable_gate
    assert "--jq '.verification.verified'" in immutable_gate
    assert ')" != true ]]' in immutable_gate
    assert exact_main_fetch in immutable_gate
    assert "if ! git merge-base --is-ancestor HEAD origin/main" in immutable_gate
    assert "release_version=$(sed -n" in immutable_gate
    assert '[[ "$release_version" != "${RELEASE_TAG#v}" ]]' in immutable_gate
    assert "python " not in immutable_gate
    assert "python3 " not in immutable_gate


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
    authorization = routing.split(
        "      - name: Reject untrusted CUDA runner execution contexts", 1
    )[1]
    guard_script = _cuda_routing_guard_script(workflow)
    assert workflow.index("  cuda-runner-routing:") < workflow.index("  single-gpu:")
    assert "runs-on: ubuntu-latest" in routing
    assert "python3 - <<'PY'" in authorization
    assert 'repository == "jemsbhai/explainiverse"' in authorization
    assert 'repository_owner == "jemsbhai"' in authorization
    assert "\n    actor == repository_owner\n" in guard_script
    assert "\n    and triggering_actor == repository_owner\n" in guard_script
    assert 'event_name == "workflow_dispatch"' in authorization
    assert 'event_name == "schedule"' not in authorization
    assert 'run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT")' in authorization
    assert 'first_attempt = run_attempt == "1"' in authorization
    assert 're.fullmatch(r"[a-f0-9]{16}", value)' in authorization
    assert "len(set(runner_nonces)) == len(runner_nonces)" in authorization
    assert "and first_attempt" in authorization
    assert "raise SystemExit(" in authorization
    assert 'event_name == "pull_request"' not in authorization
    assert 'event_name == "push"' not in authorization
    assert "continue-on-error" not in authorization
    assert "Checkout" not in routing
    for topology, edge in (
        ("SINGLE", "MINIMUM"),
        ("SINGLE", "LATEST"),
        ("TWO", "MINIMUM"),
        ("TWO", "LATEST"),
    ):
        input_name = f"{topology.lower()}_{edge.lower()}_runner_nonce"
        assert f"{topology}_{edge}_RUNNER_NONCE: ${{{{ inputs.{input_name} }}}}" in routing
        assert f"      {input_name}:" in workflow
    assert "vars.CUDA_SINGLE_RUNNER" not in workflow
    assert "vars.CUDA_TWO_RUNNER" not in workflow
    assert "continue-on-error" not in routing
    assert "needs: cuda-runner-routing" in single
    assert "if: ${{ always() }}" in single
    assert (
        "${{ github.run_attempt == 1 &&\n"
        "      needs.cuda-runner-routing.result == 'success' &&\n"
        f"      format('{single_label}-jit-{{0}}', matrix.runner_nonce) ||\n"
        "      'ubuntu-latest' }}" in single
    )
    assert "runner_nonce: ${{ inputs.single_minimum_runner_nonce }}" in single
    assert "runner_nonce: ${{ inputs.single_latest_runner_nonce }}" in single
    single_reporter = single.split(
        "      - name: Fail the required check when CUDA routing is rejected", 1
    )[1].split("\n      - name: Checkout", 1)[0]
    single_steps = single.split("\n    steps:", 1)[1]
    assert single_steps.index("Fail the required check when CUDA routing is rejected") < (
        single_steps.index("      - name: Checkout")
    )
    assert (
        "if: github.run_attempt != 1 || needs.cuda-runner-routing.result != 'success'"
        in single_reporter
    )
    assert single_reporter.count("exit 1") == 1
    assert "continue-on-error" not in single_reporter
    assert "if: always()" not in single_steps
    assert "needs: cuda-runner-routing" in two
    assert "always() &&" in two
    assert (
        "${{ github.run_attempt == 1 &&\n"
        "      needs.cuda-runner-routing.result == 'success' &&\n"
        f"      format('{two_label}-jit-{{0}}', matrix.runner_nonce) ||\n"
        "      'ubuntu-latest' }}" in two
    )
    assert "runner_nonce: ${{ inputs.two_minimum_runner_nonce }}" in two
    assert "runner_nonce: ${{ inputs.two_latest_runner_nonce }}" in two
    two_reporter = two.split(
        "      - name: Fail the required check when CUDA routing is rejected", 1
    )[1].split("\n      - name: Checkout", 1)[0]
    two_steps = two.split("\n    steps:", 1)[1]
    assert two_steps.index("Fail the required check when CUDA routing is rejected") < (
        two_steps.index("      - name: Checkout")
    )
    assert (
        "if: github.run_attempt != 1 || needs.cuda-runner-routing.result != 'success'"
        in two_reporter
    )
    assert two_reporter.count("exit 1") == 1
    assert "continue-on-error" not in two_reporter
    assert "if: always()" not in two_steps

    preflight = publish.split("  preflight:", 1)[1].split("\n  cuda-release:", 1)[0]
    publish_cuda = publish.split("  cuda-release:", 1)[1].split("\n  build:", 1)[0]
    assert "Require exact reviewed single-GPU runner routing" not in preflight
    assert "SINGLE_MINIMUM_RUNNER_NONCE: ${{ inputs.single_minimum_runner_nonce }}" in preflight
    assert "SINGLE_LATEST_RUNNER_NONCE: ${{ inputs.single_latest_runner_nonce }}" in preflight
    assert 're.fullmatch(r"[a-f0-9]{16}", value)' in preflight
    assert "len(set(runner_nonces)) == len(runner_nonces)" in preflight
    assert "evidence_nonces & release_nonces" in preflight
    assert "release runner nonces reuse CUDA evidence nonces" in preflight
    assert "needs: preflight" in publish_cuda
    assert (
        f"runs-on: ${{{{ format('{single_label}-jit-{{0}}', matrix.runner_nonce) }}}}"
        in publish_cuda
    )
    assert "runner_nonce: ${{ inputs.single_minimum_runner_nonce }}" in publish_cuda
    assert "runner_nonce: ${{ inputs.single_latest_runner_nonce }}" in publish_cuda
    assert "ubuntu-latest" not in publish_cuda


def test_tagged_repository_python_never_inherits_github_tokens():
    for workflow_name in ("publish-pypi.yml", "recover-github-release.yml"):
        workflow = _read(workflow_name)
        for step in workflow.split("\n      - name:")[1:]:
            header = step.split("\n      - name:", 1)[0]
            if "GH_TOKEN:" not in header and "GITHUB_TOKEN:" not in header:
                continue
            for line in header.splitlines():
                stripped = line.strip()
                if re.search(r"(?:python|python3) (?:release-source/)?scripts/", stripped):
                    assert "env -u GH_TOKEN -u GITHUB_TOKEN" in stripped, (
                        workflow_name,
                        stripped,
                    )


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

    verify_job = workflow.split("  verify:", 1)[1].split("\n  attest:", 1)[0]
    attest_job = workflow.split("  attest:", 1)[1]
    assert "id-token: write" not in verify_job
    assert "attestations: write" not in verify_job
    assert "release_external_controls.py bind" in verify_job
    assert "id: upload" in verify_job
    assert "artifact-id: ${{ steps.upload.outputs.artifact-id }}" in verify_job
    assert "artifact-digest: ${{ steps.upload.outputs.artifact-digest }}" in verify_job

    assert "needs: verify" in attest_job
    assert "id-token: write" in attest_job
    assert "attestations: write" in attest_job
    assert "actions/checkout@" not in attest_job
    assert "actions/setup-python@" not in attest_job
    assert "python scripts/" not in attest_job
    assert "GH_TOKEN: ${{ github.token }}" in attest_job
    assert "actions/artifacts/$ARTIFACT_ID/zip" in attest_job
    assert 'gh api "repos/$GITHUB_REPOSITORY/actions/artifacts/$ARTIFACT_ID"' in attest_job
    assert "--arg name release-control-preflight" in attest_job
    assert ".workflow_run.id == $run and .workflow_run.head_sha == $head" in attest_job
    assert ".digest == $digest" in attest_job
    assert "artifact digest must be 64 lowercase hexadecimal characters" in attest_job
    assert 'test "$actual_artifact_digest" = "$ARTIFACT_DIGEST"' in attest_job
    assert 'archive_entries=$(unzip -Z1 "$artifact_archive" | sort)' in attest_job
    assert "! -type f -print -quit" in attest_job
    assert "sha256sum --check admin-capture.json.sha256" in attest_job
    assert "sha256sum --check cuda-evidence.sha256" in attest_job
    assert "sha256sum --check external-controls.json.sha256" in attest_job
    assert attest_job.index("Download and verify only the exact low-authority artifact") < (
        attest_job.index("actions/attest-build-provenance@")
    )


def test_publish_cannot_build_without_preflight_and_real_cuda_edges():
    workflow = _read("publish-pypi.yml")
    lowered = workflow.lower()
    _assert_publish_front_door_contract(workflow)
    _assert_publish_jobs_are_first_attempt_only(workflow)
    assert "preflight_run_id:" in workflow
    assert "needs: [preflight, cuda-release]" in workflow
    assert "gh attestation verify release-preflight/artifact/external-controls.json" in workflow
    assert "release-preflight.yml@refs/heads/main" in workflow
    assert "release_external_controls.py verify" in workflow
    assert "Release CUDA single-GPU (Torch ${{ matrix.torch-edge }}, zero skips)" in workflow
    assert 'EXPLAINIVERSE_REQUIRED_CUDA_DEVICES: "1"' in workflow
    assert "needs: preflight" in workflow
    assert workflow.count("check_pypi_version_absent.py") == 1
    assert workflow.index("check_pypi_version_absent.py") < workflow.index("  build:")
    assert "jobs?filter=all&per_page=100" in workflow
    assert "gh api --paginate" in workflow
    assert workflow.count("gh-action-pypi-publish@") == 1
    assert "skip-existing" not in lowered
    for forbidden in ("${{ secrets.", "password:", "pypi_api_token", "__token__", "user:"):
        assert forbidden not in lowered
    attester = workflow.split("  attest:", 1)[1].split("\n  publish:", 1)[0]
    publisher = workflow.split("  publish:", 1)[1].split("\n  github-release:", 1)[0]
    release_preparer = workflow.split("  github-release:", 1)[1].split(
        "\n  github-release-finalize:", 1
    )[0]
    release_finalizer = workflow.split("  github-release-finalize:", 1)[1]
    build_job = workflow.split("  build:", 1)[1].split("\n  attest:", 1)[0]

    assert "id: upload_distributions" in build_job
    assert "id: upload_provenance" in build_job
    assert "distributions_artifact_id:" in build_job
    assert "distributions_artifact_digest:" in build_job
    assert "provenance_artifact_id:" in build_job
    assert "provenance_artifact_digest:" in build_job

    assert "actions: read" in attester
    assert "id-token: write" in attester
    assert "attestations: write" in attester
    assert "actions/checkout@" not in attester
    assert "actions/setup-python@" not in attester
    assert "python scripts/" not in attester
    assert "actions/download-artifact@" not in attester
    assert "actions/artifacts/$artifact_id/zip" in attester
    assert 'gh api "repos/$GITHUB_REPOSITORY/actions/artifacts/$artifact_id"' in attester
    assert ".workflow_run.id == $run and .workflow_run.head_sha == $head" in attester
    assert ".digest == $digest" in attester
    assert 'test "$actual_digest" = "$artifact_digest"' in attester
    assert "distribution hash manifest is malformed" in attester
    assert attester.index('test "$actual_digest" = "$artifact_digest"') < attester.index(
        "actions/attest-build-provenance@"
    )
    assert "attestations: true" in publisher
    assert "environment:\n      name: pypi" in publisher
    assert "id-token: write" in publisher
    assert "actions: read" in publisher
    assert "attestations: read" in publisher
    assert "actions/checkout@" not in publisher
    assert "actions/setup-python@" not in publisher
    assert "python scripts/" not in publisher
    assert "python release-source/scripts/" not in publisher
    assert "actions/download-artifact@" not in publisher
    assert "actions/artifacts/$artifact_id/zip" in publisher
    assert 'gh api "repos/$GITHUB_REPOSITORY/actions/artifacts/$artifact_id"' in publisher
    assert ".workflow_run.id == $run and .workflow_run.head_sha == $head" in publisher
    assert ".digest == $digest" in publisher
    assert 'test "$actual_digest" = "$artifact_digest"' in publisher
    assert 'gh attestation verify "$artifact"' in publisher
    assert "status=$(curl --silent --show-error --location" in publisher
    assert 'if [[ "$status" != 404 ]]; then' in publisher
    assert publisher.index('gh attestation verify "$artifact"') < publisher.index(
        "pypa/gh-action-pypi-publish@"
    )
    assert publisher.index("PyPI absence guard requires exact HTTP 404") < publisher.index(
        "pypa/gh-action-pypi-publish@"
    )
    assert "create_release_governance_record.py" in workflow
    assert "external-controls.json" in workflow
    assert "--notes-file finalize-source/release-assets/RELEASE_GOVERNANCE.md" in workflow
    assert "--draft" in workflow and "--draft=false" in workflow
    assert "normal-release-plan/evidence/pre-finalize-pypi.json" in workflow
    assert "normal-release-plan/evidence/draft-assets" in workflow
    assert "normal-release-plan/evidence/final-assets" in workflow
    assert workflow.count("verify_release_recovery.py artifacts") == 1
    assert "'.draft == false and .prerelease == false and .immutable == true'" in workflow
    assert (
        'gh api "repos/$GITHUB_REPOSITORY/releases/tags/$RELEASE_TAG" \\\n'
        '            -H "X-GitHub-Api-Version: 2026-03-10"'
    ) in workflow
    assert release_finalizer.index("live-boundary-immutable-releases.json") < (
        release_finalizer.index('gh release edit "$RELEASE_TAG"')
    )
    assert "Archive normal-path release verification evidence" in workflow
    assert "if: always() && inputs.stage_recovery_drill == false" in workflow
    assert "retention-days: 90" in workflow
    assert "verify_pypi_provenance.py" in workflow
    assert "--output-dir pypi-provenance" in workflow
    assert "pypi-attestations verify pypi" in workflow
    assert '--repository "https://github.com/$GITHUB_REPOSITORY"' in workflow
    assert "--provenance-file" in workflow
    assert "pypi-attestations-version.txt" in workflow
    assert "--require-hashes" in workflow
    assert "defaults:\n  run:\n    shell: bash" in workflow
    assert "normal-release-plan/evidence/immutable-releases.json" in workflow

    assert "contents: write" not in release_preparer
    assert "Verify signed immutable release source before candidate code" in release_preparer
    assert "gh release create" not in release_preparer
    assert "gh release upload" not in release_preparer
    assert "gh release edit" not in release_preparer
    assert "contents: write" in release_finalizer
    assert "needs: [build, github-release]" in release_finalizer
    assert "github.run_attempt == 1 && inputs.stage_recovery_drill == false" in (release_finalizer)
    for forbidden in (
        "actions/checkout@",
        "actions/setup-python@",
        "python scripts/",
        "python3 scripts/",
        "pip install",
        "poetry ",
        "verify_release_recovery.py",
        "verify_pypi_provenance.py",
    ):
        assert forbidden not in release_finalizer
    assert "actions/artifacts/$artifact_id/zip" in release_finalizer
    assert "PLAN_ARTIFACT_ID: ${{ needs.github-release.outputs.plan_artifact_id }}" in (
        release_finalizer
    )
    assert 'test "$actual_digest" = "$artifact_digest"' in release_finalizer
    assert ".workflow_run.id == $run and .workflow_run.head_sha == $head" in release_finalizer
    assert "release-distributions" in release_finalizer
    assert "release-provenance" in release_finalizer
    assert "cmp --silent normal-release-plan/release-assets.sha256" in release_finalizer
    assert "finalize-source/release-assets.sha256" in release_finalizer
    assert ".verification.verified == true" in release_finalizer
    assert "finalize-main-ancestry.json" in release_finalizer
    assert 'gh attestation verify "finalize-source/release-assets/$filename"' in (release_finalizer)
    assert 'gh release upload "$RELEASE_TAG"' in release_finalizer
    assert '"finalize-source/release-assets/$filename"' in release_finalizer
    assert 'gh release upload "$RELEASE_TAG" "normal-release-plan/' not in release_finalizer

    assert "actions: read" in build_job and "attestations: read" in build_job
    assert "sha256sum --check admin-capture.json.sha256" in build_job
    assert "sha256sum --check cuda-evidence.sha256" in build_job
    assert "for evidence in" in build_job
    assert "release-source/scripts/audit_typing_readiness.py" in build_job
    assert '--distribution "$artifact"' in build_job
    assert build_job.index("release-source/scripts/audit_typing_readiness.py") < build_job.index(
        "Upload immutable distributions"
    )


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
    assert (
        "env -u GH_TOKEN -u GITHUB_TOKEN python release-source/scripts/create_release_governance_record.py"
        in build_job
    )
    assert (
        "cd release-source\n            env -u GH_TOKEN -u GITHUB_TOKEN python scripts/record_release_environment.py"
        in build_job
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
    assert "reproducibility_run_attempt=$(" in build_job
    assert "provenance/reproducibility-source-run.json" in build_job
    assert '--expected-run-id "$reproducibility_run_id"' in build_job
    assert '--expected-run-attempt "$reproducibility_run_attempt"' in build_job

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
    assert "release-assets.sha256" in workflow
    assert 'test "$actual_digest" = "$ARTIFACT_DIGEST"' in workflow
    assert "actions/artifacts/$ARTIFACT_ID/zip" in workflow
    assert "actions/runs/$SOURCE_RUN_ID/artifacts?per_page=100" in workflow
    assert "release-distributions" in workflow
    assert "release-provenance" in workflow
    assert ".workflow_run.id == $run and .workflow_run.head_sha == $head" in workflow
    assert 'test "$actual_digest" = "$artifact_digest"' in workflow
    assert "cmp recovery/source-release-assets.sha256 recovery/release-assets.sha256" in workflow
    assert "mutation-tag-object.json" in workflow
    assert "mutation-main-ancestry.json" in workflow
    assert ".verification.verified == true" in workflow
    assert '.conclusion == "failure"' in workflow
    assert "pre-mutation-pypi.json" in workflow
    assert 'test "$actual_pypi" = "$expected_pypi"' in workflow
    assert "--draft=false" in workflow
    assert "final-github-assets.sha256" in workflow
    assert "final-github-release.json" in workflow
    assert "Archive complete or partial recovery evidence for fixed mutation" in workflow
    assert "Archive complete or partial fixed-command recovery evidence" in workflow
    assert "if: always()" in workflow
    assert "--notes-file recovery/source-release-assets/RELEASE_GOVERNANCE.md" in workflow
    assert "for asset in recovery/source-release-assets/*" in workflow
    assert 'gh release upload "$RELEASE_TAG" "$asset"' in workflow
    assert 'gh release create "$RELEASE_TAG" ./recovery/release-assets/*' not in workflow
    assert "recovery draft omitted the original governance disclosure" in workflow
    assert "'.isDraft == false and .isPrerelease == false'" in workflow
    assert "'.immutable == true' recovery/final-github-release-api.json" in workflow
    assert "final GitHub Release omitted the original governance disclosure" in workflow
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

    verify_job = workflow.split("  verify:", 1)[1].split("\n  recover:", 1)[0]
    recover_job = workflow.split("  recover:", 1)[1]
    assert "contents: write" not in verify_job
    assert "verify_release_recovery.py" in verify_job
    assert "verify_pypi_provenance.py" in verify_job
    assert "needs: verify" in recover_job
    assert "contents: write" in recover_job
    assert "recovery/source-release-assets" in recover_job
    assert 'gh attestation verify "recovery/source-release-assets/$filename"' in recover_job
    for forbidden in (
        "actions/checkout@",
        "actions/setup-python@",
        "python scripts/",
        "python3 scripts/",
        "pip install",
        "verify_release_recovery.py",
        "verify_pypi_provenance.py",
    ):
        assert forbidden not in recover_job


def test_privileged_release_jobs_reject_extra_or_missing_provenance_assets():
    publish = _read("publish-pypi.yml")
    recovery = _read("recover-github-release.yml")
    _assert_exact_provenance_inventories(publish, 3)
    _assert_exact_provenance_inventories(recovery, 1)

    for workflow, occurrences in ((publish, 3), (recovery, 1)):
        mutated = workflow.replace(
            "            explainiverse-build.cdx.json \\\n",
            "",
            1,
        )
        assert mutated != workflow
        with pytest.raises(AssertionError):
            _assert_exact_provenance_inventories(mutated, occurrences)


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


@pytest.mark.parametrize(
    ("overrides", "authorized"),
    (
        pytest.param({}, True, id="owner-dispatch-reviewed-branch"),
        pytest.param(
            {"GITHUB_EVENT_NAME": "schedule", "GITHUB_REF": "refs/heads/main"},
            False,
            id="schedule-never-opens-one-use-runner-route",
        ),
        pytest.param(
            {
                "GITHUB_EVENT_NAME": "pull_request",
                "GITHUB_REF": "refs/pull/17/merge",
            },
            False,
            id="pull-request-with-exact-runner-variables",
        ),
        pytest.param(
            {"GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/main"},
            False,
            id="push-with-exact-runner-variables",
        ),
        pytest.param(
            {"GITHUB_ACTOR": "write-collaborator"},
            False,
            id="non-owner-dispatch",
        ),
        pytest.param(
            {"GITHUB_TRIGGERING_ACTOR": "write-collaborator"},
            False,
            id="non-owner-rerun",
        ),
        pytest.param(
            {"GITHUB_EVENT_NAME": "schedule", "GITHUB_REF": "refs/heads/develop"},
            False,
            id="schedule-outside-main",
        ),
        pytest.param(
            {"SINGLE_MINIMUM_RUNNER_NONCE": "AAAAAAAAAAAAAAAA"},
            False,
            id="uppercase-runner-nonce",
        ),
        pytest.param(
            {"SINGLE_MINIMUM_RUNNER_NONCE": "0" * 15},
            False,
            id="short-runner-nonce",
        ),
        pytest.param(
            {"SINGLE_MINIMUM_RUNNER_NONCE": None},
            False,
            id="missing-runner-nonce",
        ),
        pytest.param(
            {
                "SINGLE_MINIMUM_RUNNER_NONCE": "0000000000000002",
                "SINGLE_LATEST_RUNNER_NONCE": "0000000000000002",
            },
            False,
            id="reused-runner-nonce",
        ),
        pytest.param(
            {"GITHUB_RUN_ATTEMPT": "2"},
            False,
            id="rerun-attempt-two",
        ),
        pytest.param(
            {"GITHUB_RUN_ATTEMPT": "0"},
            False,
            id="zero-attempt",
        ),
        pytest.param(
            {"GITHUB_RUN_ATTEMPT": None},
            False,
            id="missing-attempt",
        ),
        pytest.param(
            {"GITHUB_RUN_ATTEMPT": "first"},
            False,
            id="non-numeric-attempt",
        ),
        pytest.param(
            {
                "GITHUB_REPOSITORY": "attacker/explainiverse",
                "GITHUB_REPOSITORY_OWNER": "attacker",
                "GITHUB_ACTOR": "attacker",
                "GITHUB_TRIGGERING_ACTOR": "attacker",
            },
            False,
            id="fork-owner-dispatch",
        ),
    ),
)
def test_cuda_router_authorizes_only_owner_first_dispatch_with_distinct_nonces(
    overrides, authorized
):
    workflow = _read("cuda-ci.yml")
    completed = _run_cuda_routing_guard(workflow, **overrides)

    assert (completed.returncode == 0) is authorized
    if not authorized:
        assert "custom CUDA runners require an owner-triggered first-attempt" in completed.stderr


def test_owner_dispatch_unlocks_both_exact_cuda_runner_routes():
    workflow = _read("cuda-ci.yml")
    assert _run_cuda_routing_guard(workflow).returncode == 0

    single = workflow.split("  single-gpu:", 1)[1].split("\n  two-gpu:", 1)[0]
    two = workflow.split("  two-gpu:", 1)[1]
    assert (
        "${{ github.run_attempt == 1 &&\n"
        "      needs.cuda-runner-routing.result == 'success' &&\n"
        "      format('explainiverse-cuda-single-jit-{0}', matrix.runner_nonce) ||\n"
        "      'ubuntu-latest' }}" in single
    )
    assert (
        "${{ github.run_attempt == 1 &&\n"
        "      needs.cuda-runner-routing.result == 'success' &&\n"
        "      format('explainiverse-cuda-two-jit-{0}', matrix.runner_nonce) ||\n"
        "      'ubuntu-latest' }}" in two
    )


@pytest.mark.parametrize(
    ("overrides", "authorized"),
    (
        pytest.param({}, True, id="v0-15-0-first-attempt-owner-drill"),
        pytest.param(
            {"RELEASE_TAG": "v0.15.1", "STAGE_RECOVERY_DRILL": "false"},
            True,
            id="future-release-normal-path-remains-available",
        ),
        pytest.param(
            {"GITHUB_REF": "refs/heads/main"},
            False,
            id="branch-ref-cannot-select-a-tag-input",
        ),
        pytest.param(
            {"GITHUB_REF": "refs/tags/v0.15.1"},
            False,
            id="different-tag-ref",
        ),
        pytest.param(
            {"GITHUB_EVENT_NAME": "push"},
            False,
            id="non-dispatch-event",
        ),
        pytest.param(
            {"GITHUB_EVENT_NAME": None},
            False,
            id="missing-event",
        ),
        pytest.param(
            {"RELEASE_TAG": "0.15.0"},
            False,
            id="malformed-stable-tag",
        ),
        pytest.param({"GITHUB_RUN_ATTEMPT": "2"}, False, id="rerun-attempt-two"),
        pytest.param({"GITHUB_RUN_ATTEMPT": "0"}, False, id="zero-attempt"),
        pytest.param({"GITHUB_RUN_ATTEMPT": None}, False, id="missing-attempt"),
        pytest.param(
            {"GITHUB_RUN_ATTEMPT": "first"},
            False,
            id="non-numeric-attempt",
        ),
        pytest.param({"GITHUB_ACTOR": "write-collaborator"}, False, id="non-owner-actor"),
        pytest.param(
            {"GITHUB_TRIGGERING_ACTOR": "write-collaborator"},
            False,
            id="non-owner-triggering-actor",
        ),
        pytest.param(
            {"GITHUB_REPOSITORY": "attacker/explainiverse"},
            False,
            id="wrong-repository",
        ),
        pytest.param(
            {"STAGE_RECOVERY_DRILL": "false"},
            False,
            id="v0-15-0-without-drill",
        ),
        pytest.param(
            {"STAGE_RECOVERY_DRILL": None},
            False,
            id="v0-15-0-with-missing-drill-input",
        ),
        pytest.param(
            {"SINGLE_MINIMUM_RUNNER_NONCE": "AAAAAAAAAAAAAAAA"},
            False,
            id="uppercase-release-runner-nonce",
        ),
        pytest.param(
            {"SINGLE_MINIMUM_RUNNER_NONCE": None},
            False,
            id="missing-release-runner-nonce",
        ),
        pytest.param(
            {
                "SINGLE_MINIMUM_RUNNER_NONCE": "0000000000000002",
                "SINGLE_LATEST_RUNNER_NONCE": "0000000000000002",
            },
            False,
            id="reused-release-runner-nonce",
        ),
    ),
)
def test_publish_front_door_requires_first_attempt_owner_and_candidate_drill(overrides, authorized):
    completed = _run_publish_front_door_guard(_read("publish-pypi.yml"), **overrides)

    assert (completed.returncode == 0) is authorized
    if not authorized:
        assert "publication preflight requires" in completed.stderr


def test_publish_front_door_contract_rejects_source_and_ordering_drift():
    workflow = _read("publish-pypi.yml")
    mutations = (
        workflow.replace(
            'event_name == "workflow_dispatch"',
            "event_name == event_name",
            1,
        ),
        workflow.replace(
            'ref == f"refs/tags/{release_tag}"',
            "ref == ref",
            1,
        ),
        workflow.replace(
            're.fullmatch(r"v[0-9]+\\.[0-9]+\\.[0-9]+", release_tag)',
            "release_tag",
            1,
        ),
        workflow.replace(
            're.fullmatch(r"[a-f0-9]{16}", value)',
            "value",
            1,
        ),
        workflow.replace(
            "distinct_nonces = len(set(runner_nonces)) == len(runner_nonces)",
            "distinct_nonces = True",
            1,
        ),
        workflow.replace(
            '[[ "$checkout_sha" != "$GITHUB_SHA" ]]',
            '[[ "$checkout_sha" != "$checkout_sha" ]]',
            1,
        ),
        workflow.replace(
            '[[ "$(git cat-file -t "$RELEASE_TAG")" != tag ]]',
            '[[ "$(git cat-file -t "$RELEASE_TAG")" != commit ]]',
            1,
        ),
        workflow.replace(
            'git rev-parse "$RELEASE_TAG^{commit}"',
            'git rev-parse "$RELEASE_TAG"',
            1,
        ),
        workflow.replace(
            "--jq '.verification.verified'",
            "--jq '.verification.reason'",
            1,
        ),
        workflow.replace(
            "git fetch --no-tags origin '+refs/heads/main:refs/remotes/origin/main'",
            "git fetch --no-tags origin main",
            1,
        ),
        workflow.replace(
            "git merge-base --is-ancestor HEAD origin/main",
            "git merge-base HEAD origin/main",
            1,
        ),
        workflow.replace(
            '[[ "$release_version" != "${RELEASE_TAG#v}" ]]',
            '[[ "$release_version" != "$release_version" ]]',
            1,
        ),
        workflow.replace(
            "      - name: Verify the signed immutable release source before external work",
            "      - name: Verify the signed immutable release source after external work",
            1,
        ),
    )

    for index, mutated in enumerate(mutations):
        assert mutated != workflow, index
        with pytest.raises((AssertionError, IndexError, ValueError)):
            _assert_publish_front_door_contract(mutated)


def test_every_publish_job_rejects_partial_reruns_before_runner_allocation():
    workflow = _read("publish-pypi.yml")
    exact_guard = "    if: ${{ github.run_attempt == 1 }}"
    _assert_publish_jobs_are_first_attempt_only(workflow)

    for job_id in PUBLISH_JOB_IDS:
        marker = f"  {job_id}:\n"
        job_start = workflow.index(marker) + len(marker)
        job_end_match = re.search(r"^  [a-z][a-z-]+:\n", workflow[job_start:], flags=re.MULTILINE)
        job_end = len(workflow) if job_end_match is None else job_start + job_end_match.start()
        guard_start = workflow.index(exact_guard, job_start, job_end)
        guard_end = workflow.index("\n", guard_start) + 1
        mutations = (
            workflow[:guard_start] + workflow[guard_end:],
            workflow[:guard_start]
            + "    if: ${{ github.run_attempt == '1' }}\n"
            + workflow[guard_end:],
            workflow[:guard_start]
            + "    if: ${{ always() && github.run_attempt == 1 }}\n"
            + workflow[guard_end:],
        )
        for mutated in mutations:
            with pytest.raises((AssertionError, IndexError, ValueError)):
                _assert_publish_jobs_are_first_attempt_only(mutated)


def test_cuda_runner_routing_contract_rejects_fail_open_drift():
    workflow = _read("cuda-ci.yml")
    publish = _read("publish-pypi.yml")
    mutations = (
        (
            workflow.replace(
                "      format('explainiverse-cuda-single-jit-{0}', matrix.runner_nonce) ||",
                "      'ubuntu-latest' ||",
                1,
            ),
            publish,
        ),
        (
            workflow.replace(
                "      ${{ github.run_attempt == 1 &&\n"
                "      needs.cuda-runner-routing.result == 'success' &&",
                "      ${{ needs.cuda-runner-routing.result == 'success' &&",
                1,
            ),
            publish,
        ),
        (
            workflow.replace(
                "if: github.run_attempt != 1 || needs.cuda-runner-routing.result != 'success'",
                "if: needs.cuda-runner-routing.result != 'success'",
                1,
            ),
            publish,
        ),
        (
            workflow.replace(
                "      format('explainiverse-cuda-two-jit-{0}', matrix.runner_nonce) ||",
                "      format('explainiverse-cuda-single-jit-{0}', matrix.runner_nonce) ||",
                1,
            ),
            publish,
        ),
        (workflow.replace("    needs: cuda-runner-routing\n", "", 1), publish),
        (
            workflow.replace(
                're.fullmatch(r"[a-f0-9]{16}", value)',
                "value",
                1,
            ),
            publish,
        ),
        (workflow.replace("          exit 1\n", "          exit 0\n", 1), publish),
        (
            workflow,
            publish.replace(
                "    runs-on: ${{ format('explainiverse-cuda-single-jit-{0}', matrix.runner_nonce) }}",
                "    runs-on: ubuntu-latest",
                1,
            ),
        ),
        (
            workflow,
            publish.replace(
                "len(set(runner_nonces)) == len(runner_nonces)",
                "True",
                1,
            ),
        ),
        (
            workflow.replace(
                'repository == "jemsbhai/explainiverse"',
                "repository == repository",
                1,
            ),
            publish,
        ),
        (
            workflow.replace(
                "actor == repository_owner",
                "actor == actor",
                1,
            ),
            publish,
        ),
        (
            workflow.replace(
                "triggering_actor == repository_owner",
                "triggering_actor == triggering_actor",
                1,
            ),
            publish,
        ),
        (
            workflow.replace(
                'approved_event = event_name == "workflow_dispatch"',
                "approved_event = True",
                1,
            ),
            publish,
        ),
        (
            workflow.replace(
                "distinct_nonces = len(set(runner_nonces)) == len(runner_nonces)",
                "distinct_nonces = True",
                1,
            ),
            publish,
        ),
        (
            workflow.replace(
                'first_attempt = run_attempt == "1"',
                "first_attempt = True",
                1,
            ),
            publish,
        ),
    )

    for index, (mutated_workflow, mutated_publish) in enumerate(mutations):
        try:
            _assert_cuda_runner_routing_contract(mutated_workflow, mutated_publish)
        except AssertionError:
            continue
        raise AssertionError(f"CUDA routing mutation {index} escaped the contract")


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


def test_release_workflow_shell_fragments_preserve_arguments_and_option_boundaries():
    dependency = _read("dependency-constraints.yml")
    publish = _read("publish-pypi.yml")
    recovery = _read("recover-github-release.yml")

    assert 'read -r -a test_targets <<< "$TEST_TARGETS"' in dependency
    assert '"${test_targets[@]}"' in dependency
    assert ' -m "not quantus_reference" $TEST_TARGETS' not in dependency
    assert "(cd dist && sha256sum -- *.whl *.tar.gz)" in publish
    assert 'gh release create "$RELEASE_TAG" \\' in publish
    assert 'gh release upload "$RELEASE_TAG" \\' in publish
    assert '"finalize-source/release-assets/$filename"' in publish
    assert 'gh release create "$RELEASE_TAG" ./normal-release-plan/release-assets/*' not in publish
    assert publish.count("sha256sum ./* | sort") == 3
    assert "sha256sum ./* | sort" in recovery


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
    authority = policy["release_runner_authority"]
    assert authority["allowed_collaborator_logins"] == ["jemsbhai"]
    assert authority["pending_invitations"] == []
    assert authority["registered_runners"] == []
    assert authority["repository_variable_names"] == []
    installed_apps = authority["installed_apps"]
    assert installed_apps["source_url"] == "https://github.com/settings/installations"
    assert {value["id"] for value in installed_apps["expected_installations"]} == {
        67312423,
        98967149,
        14141661,
        98315629,
        109872254,
        80585128,
    }
    assert {
        value["name"]: value["suspended"] for value in installed_apps["expected_installations"]
    } == {
        "ChatGPT Codex Connector": True,
        "Claude": True,
        "GitGuardian": True,
        "lovable.dev": True,
        "Socket Security": False,
        "Vercel": True,
    }
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


def test_release_runbook_requires_external_authority_for_mutable_workflows():
    runbook = (ROOT / "docs" / "RELEASE_OPERATIONS.md").read_text(encoding="utf-8")
    assert "guard is defense in depth, not the authority boundary" in runbook
    assert "is the sole collaborator" in runbook
    assert "zero pending invitations" in runbook
    assert "queued or in-progress job targeting any planned nonce-bearing label" in runbook
    assert "re-invite each collaborator at the exact prior permission" in runbook
    assert "Restoration is not complete" in runbook
    assert "Only then directly generate each one-use JIT configuration" in runbook
    assert "administrator/provider controls" in runbook
    assert "disjoint from all four accepted CUDA-evidence nonces" in runbook
    assert "--installed-app-authority $installedAppAuthority" in runbook
    assert "no more than 10 minutes before the JSON snapshot" in runbook
