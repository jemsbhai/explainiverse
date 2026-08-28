from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "deploy-demo.yml"

UPLOAD_PAGES_SHA = "fc324d3547104276b827a68afc52ff2a11cc49c9"
TRANSITIVE_UPLOAD_ARTIFACT_SHA = "bbbca2ddaa5d8feaa63e36b76fdaad77386f024f"
CONFIGURE_PAGES_SHA = "983d7736d9b0ae728b81ab479565c72886d7745b"
DEPLOY_PAGES_SHA = "d6db90164ac5ed86f2b6aed7e0febac5b3c0c03e"

REQUIRED_CONTEXT_EXPRESSIONS = (
    "github.repository == 'jemsbhai/explainiverse'",
    "github.repository_owner == 'jemsbhai'",
    "github.event_name == 'push'",
    "github.ref == 'refs/heads/main'",
    "github.sha == github.event.after",
    "github.run_attempt == 1",
)


def _job(workflow: str, name: str, next_name: str | None) -> str:
    end = rf"(?=^  {re.escape(next_name)}:\n)" if next_name is not None else r"\Z"
    match = re.search(rf"(?ms)^  {re.escape(name)}:\n(.*?){end}", workflow)
    if match is None:
        raise ValueError(f"missing Pages job {name}")
    return match.group(0)


def _verify_pages_authority_contract(workflow: str) -> None:
    trigger = workflow.split("\npermissions:\n", 1)[0].rstrip("\n")
    if trigger != ("name: Deploy Demo to Pages\n\n" "on:\n" '  push:\n    branches: ["main"]'):
        raise ValueError("Pages trigger must be exactly a main push")
    if "workflow_dispatch" in workflow or "pull_request" in trigger:
        raise ValueError("Pages workflow has a non-push trigger")

    authorize = _job(workflow, "authorize", "build")
    build = _job(workflow, "build", "deploy")
    deploy = _job(workflow, "deploy", None)
    for name, block in (("authorize", authorize), ("build", build), ("deploy", deploy)):
        for expression in REQUIRED_CONTEXT_EXPRESSIONS:
            if expression not in block:
                raise ValueError(f"{name} does not bind {expression}")

    if "    permissions: {}\n" not in authorize:
        raise ValueError("authorization job must have zero token permissions")
    if "uses:" in authorize or "actions/checkout" in authorize:
        raise ValueError("authorization job may not execute repository or third-party code")
    required_authorize_environment = (
        "ACTUAL_REPOSITORY: ${{ github.repository }}",
        "ACTUAL_OWNER: ${{ github.repository_owner }}",
        "ACTUAL_EVENT: ${{ github.event_name }}",
        "ACTUAL_REF: ${{ github.ref }}",
        "ACTUAL_SHA: ${{ github.sha }}",
        "EVENT_AFTER: ${{ github.event.after }}",
        "RUN_ATTEMPT: ${{ github.run_attempt }}",
    )
    for binding in required_authorize_environment:
        if binding not in authorize:
            raise ValueError(f"authorization script lacks {binding}")
    if 're.fullmatch(r"[0-9a-f]{40}", sha)' not in authorize:
        raise ValueError("authorization script does not validate the exact SHA shape")
    if 're.fullmatch(r"[0-9a-f]{40}", event_after)' not in authorize:
        raise ValueError("authorization script does not validate event.after")
    if "and sha == event_after" not in authorize:
        raise ValueError("authorization script does not bind SHA to push event.after")

    if "    needs: authorize\n" not in build:
        raise ValueError("build must depend on authorization")
    if "needs.authorize.result == 'success'" not in build:
        raise ValueError("build does not require successful authorization")
    if build.index("needs: authorize") > build.index("actions/checkout@"):
        raise ValueError("build can execute repository code before authorization")
    topology_guard = "Reject links, hardlinks, and special files from the deployable tree"
    final_tree_guard = "Re-prove the exact deployable tree before Pages packaging"
    for sentinel in (
        "find dist/demo -type l -print -quit",
        "find dist/demo -type f -links +1 -print -quit",
        "find dist/demo ! -type d ! -type f -print -quit",
        "find dist/demo -mindepth 1 -type d -empty -print -quit",
        're.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", relative)',
    ):
        if build.count(sentinel) != 2:
            raise ValueError(f"Pages topology guard lacks {sentinel}")
    if topology_guard not in build or build.index(topology_guard) > build.index(
        "Record the exact deployable tree"
    ):
        raise ValueError("Pages topology is not checked before evidence")
    if final_tree_guard not in build:
        raise ValueError("Pages tree is not re-proved before packaging")
    if not (
        build.index("Retain the exact deployment evidence")
        < build.index(final_tree_guard)
        < build.index("actions/upload-pages-artifact@")
    ):
        raise ValueError("Pages tree is not re-proved immediately before packaging")
    if 'cmp --silent artifacts/deployment/demo-files.sha256 "$current_manifest"' not in build:
        raise ValueError("Pages final manifest is not bound to recorded evidence")
    if "sha256sum --check artifacts/deployment/demo-tree.sha256" not in build:
        raise ValueError("Pages final tree hash is not checked")

    if "    needs: [authorize, build]\n" not in deploy:
        raise ValueError("deploy must depend on authorization and build")
    if "needs.authorize.result == 'success'" not in deploy:
        raise ValueError("deploy does not require successful authorization")
    if "needs.build.result == 'success'" not in deploy:
        raise ValueError("deploy does not require a successful build")
    if not deploy.index("    if: >-") < deploy.index("    environment:"):
        raise ValueError("deployment authority is declared before its job-level condition")
    if "      actions: read\n      pages: write\n      id-token: write\n" not in deploy:
        raise ValueError("deployment cannot independently read retained artifacts")
    verifier = "Verify the exact server-retained Pages artifact and deployed tree"
    if verifier not in deploy or deploy.index(verifier) > deploy.index("Setup Pages"):
        raise ValueError("deployment job lacks a pre-authority artifact verifier")
    for binding in (
        "EVIDENCE_ARTIFACT_ID: ${{ needs.build.outputs.evidence_artifact_id }}",
        "EVIDENCE_ARTIFACT_DIGEST: ${{ needs.build.outputs.evidence_artifact_digest }}",
        "PAGES_ARTIFACT_ID: ${{ needs.build.outputs.pages_artifact_id }}",
        "actions/runs/{run_id}/artifacts?{query}",
        'workflow_run.get("id") != run_id',
        'workflow_run.get("head_sha") != head_sha',
        'direct.get("name") != name',
        'direct.get("id") != artifact_id',
        'server_digest != f"sha256:{expected_digest}"',
        "hashlib.sha256(path.read_bytes()).hexdigest() != digest",
        'read_zip(pages_zip, {"artifact.tar"})',
        "member.issym() or member.islnk()",
        "directories != expected_directories",
        "actual != manifest",
    ):
        if binding not in deploy:
            raise ValueError(f"deployment artifact verifier lacks {binding}")
    if "Authorization" not in deploy or "new, unauthenticated request" not in deploy:
        raise ValueError("artifact redirect must not receive the repository token")
    if "artifact_name: github-pages" not in deploy:
        raise ValueError("deploy-pages is not bound to the verified artifact name")

    for output in (
        "evidence_artifact_id: ${{ steps.evidence-upload.outputs.artifact-id }}",
        "evidence_artifact_digest: ${{ steps.evidence-upload.outputs.artifact-digest }}",
        "pages_artifact_id: ${{ steps.pages-upload.outputs.artifact_id }}",
    ):
        if output not in build:
            raise ValueError(f"build does not export {output}")

    expected_upload = f"actions/upload-pages-artifact@{UPLOAD_PAGES_SHA} # v5.0.0"
    if expected_upload not in build:
        raise ValueError("upload-pages-artifact is not pinned to reviewed official v5.0.0")
    if TRANSITIVE_UPLOAD_ARTIFACT_SHA not in build:
        raise ValueError("reviewed upload-artifact transitive pin is not recorded")
    if f"actions/configure-pages@{CONFIGURE_PAGES_SHA} # v5" not in deploy:
        raise ValueError("configure-pages pin drifted")
    if f"actions/deploy-pages@{DEPLOY_PAGES_SHA} # v4" not in deploy:
        raise ValueError("deploy-pages pin drifted")

    for field in (
        "repository: process.env.GITHUB_REPOSITORY",
        "repository_owner: process.env.GITHUB_REPOSITORY_OWNER",
        "event: process.env.GITHUB_EVENT_NAME",
        "ref: process.env.GITHUB_REF",
        "commit: process.env.GITHUB_SHA",
        "event_after: process.env.EVENT_AFTER",
        "workflow_run_attempt: Number(process.env.GITHUB_RUN_ATTEMPT)",
    ):
        if field not in build:
            raise ValueError(f"deployment evidence lacks {field}")


def test_pages_workflow_binds_authority_before_checkout_and_deployment() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    _verify_pages_authority_contract(workflow)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('branches: ["main"]', 'branches: ["develop"]', "main push"),
        ("\npermissions:\n", "\n  workflow_dispatch:\n\npermissions:\n", "main push"),
        (
            "github.repository == 'jemsbhai/explainiverse'",
            "github.repository == 'attacker/fork'",
            "authorize does not bind",
        ),
        (
            "github.repository_owner == 'jemsbhai'",
            "github.repository_owner == 'attacker'",
            "authorize does not bind",
        ),
        ("github.event_name == 'push'", "github.event_name != ''", "authorize does not bind"),
        (
            "github.ref == 'refs/heads/main'",
            "startsWith(github.ref, 'refs/heads/')",
            "authorize does not bind",
        ),
        ("github.sha == github.event.after", "github.sha != ''", "authorize does not bind"),
        ("github.run_attempt == 1", "github.run_attempt > 0", "authorize does not bind"),
        ("    permissions: {}\n", "    permissions:\n      contents: read\n", "zero token"),
        ("    needs: authorize\n", "", "build must depend"),
        ("    needs: [authorize, build]\n", "    needs: build\n", "deploy must depend"),
        (
            f"actions/upload-pages-artifact@{UPLOAD_PAGES_SHA}",
            "actions/upload-pages-artifact@v5",
            "not pinned",
        ),
        (
            TRANSITIVE_UPLOAD_ARTIFACT_SHA,
            "0" * 40,
            "transitive pin",
        ),
        (
            "event_after: process.env.EVENT_AFTER",
            "event_after: null",
            "deployment evidence",
        ),
        (
            "find dist/demo -type l -print -quit",
            "echo disabled-symlink-check",
            "topology guard",
        ),
        (
            'cmp --silent artifacts/deployment/demo-files.sha256 "$current_manifest"',
            "true # manifest comparison removed",
            "final manifest",
        ),
        ("      actions: read\n", "", "independently read"),
        (
            'workflow_run.get("head_sha") != head_sha',
            "False",
            "artifact verifier lacks",
        ),
        (
            "hashlib.sha256(path.read_bytes()).hexdigest() != digest",
            "False",
            "artifact verifier lacks",
        ),
        ("member.issym() or member.islnk()", "False", "artifact verifier lacks"),
        ("actual != manifest", "False", "artifact verifier lacks"),
        ("artifact_name: github-pages", "artifact_name: other", "verified artifact name"),
    ],
)
def test_pages_authority_contract_rejects_adversarial_drift(
    old: str, new: str, message: str
) -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert old in workflow
    changed = workflow.replace(old, new, 1)
    with pytest.raises(ValueError, match=message):
        _verify_pages_authority_contract(changed)
