"""Release provenance must identify the tag checkout, not a dispatch branch."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

VALIDATOR_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "validate_release_attestation_context.py"
)
VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "explainiverse_release_attestation_context", VALIDATOR_PATH
)
assert VALIDATOR_SPEC is not None and VALIDATOR_SPEC.loader is not None
validator = importlib.util.module_from_spec(VALIDATOR_SPEC)
sys.modules[VALIDATOR_SPEC.name] = validator
VALIDATOR_SPEC.loader.exec_module(validator)
validate_release_attestation_context = validator.validate_release_attestation_context

SHA_A = "a" * 40
SHA_B = "b" * 40


def test_release_attestation_context_accepts_the_exact_tag_and_commit():
    validate_release_attestation_context(
        release_tag="v0.15.0",
        github_ref="refs/tags/v0.15.0",
        github_sha=SHA_A.upper(),
        checkout_sha=SHA_A,
    )


@pytest.mark.parametrize(
    ("release_tag", "github_ref", "github_sha", "checkout_sha", "match"),
    [
        ("0.15.0", "refs/tags/0.15.0", SHA_A, SHA_A, "form"),
        ("v0.15.0", "refs/heads/main", SHA_A, SHA_A, "dispatched from"),
        ("v0.15.0", "refs/tags/v0.15.0", SHA_A, SHA_B, "does not match"),
        ("v0.15.0", "refs/tags/v0.15.0", "abc", SHA_A, "40-character"),
    ],
)
def test_release_attestation_context_rejects_mismatched_oidc_identity(
    release_tag, github_ref, github_sha, checkout_sha, match
):
    with pytest.raises(ValueError, match=match):
        validate_release_attestation_context(
            release_tag=release_tag,
            github_ref=github_ref,
            github_sha=github_sha,
            checkout_sha=checkout_sha,
        )


def test_publish_workflow_runs_the_attestation_context_guard():
    workflow = (Path(__file__).parents[1] / ".github" / "workflows" / "publish-pypi.yml").read_text(
        encoding="utf-8"
    )
    assert "python scripts/validate_release_attestation_context.py" in workflow


def test_recovery_workflow_binds_its_dispatch_ref_and_sha_before_remote_evidence():
    workflow = (
        Path(__file__).parents[1] / ".github" / "workflows" / "recover-github-release.yml"
    ).read_text(encoding="utf-8")
    checkout = workflow.index("Check out the immutable release tag")
    guard = workflow.index("python scripts/validate_release_attestation_context.py")
    source_run_query = workflow.index(
        'gh api "repos/$GITHUB_REPOSITORY/actions/runs/$SOURCE_RUN_ID"'
    )
    assert "ref: ${{ inputs.tag }}" in workflow
    dispatch_guard = workflow.index('expected_ref="refs/tags/$RELEASE_TAG"')
    assert dispatch_guard < checkout
    assert checkout < guard < source_run_query
