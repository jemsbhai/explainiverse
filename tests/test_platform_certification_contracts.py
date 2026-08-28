"""Structural contracts for platform, browser, typing, and accessibility gates."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_release_policy_requires_the_exact_new_javascript_contexts():
    policy = json.loads(_read(".github/release-control-policy.json"))
    workflow = _read(".github/workflows/js-ci.yml")
    required = set(policy["required_checks"])
    contexts = {
        "React 18 peer floor (Node.js 20.11.0)",
        "Real-browser demo and automated accessibility (Playwright)",
    }
    assert contexts <= required
    for context in contexts:
        assert f"name: {context}" in workflow
    assert "react@18.0.0 react-dom@18.0.0" in workflow
    assert "npx playwright install --with-deps chromium firefox webkit" in workflow
    assert "npm run test:browser" in workflow


def test_browser_gate_exercises_security_functionality_and_accessibility():
    package = json.loads(_read("packages/js/package.json"))
    assert package["private"] is True
    assert package["type"] == "commonjs"
    assert package["devDependencies"]["@playwright/test"] == "1.62.1"
    assert package["devDependencies"]["@axe-core/playwright"] == "4.12.1"

    config = _read("packages/js/playwright.config.ts")
    assert "retries: 0" in config
    assert "retries: process.env.CI" not in config
    for engine in ("chromium", "firefox", "webkit"):
        assert f'name: "{engine}"' in config

    suite = _read("packages/js/tests/browser/demo.spec.ts")
    for contract in (
        "context.route",
        'route.abort("blockedbyclient")',
        'page.on("pageerror"',
        "AxeBuilder",
        "wcag22aa",
        "320 CSS px",
        "updates synthetic attribution state",
    ):
        assert contract in suite

    html = _read("packages/js/src/demo/index.html")
    assert "Content-Security-Policy" in html
    assert "connect-src 'none'" in html
    assert "script-src 'self'" in html
    assert 'name="referrer" content="no-referrer"' in html

    demo = _read("packages/js/src/demo/main.tsx")
    assert "fontSize: 'clamp(2rem, 10vw, 42px)'" in demo
    assert "overflowWrap: 'anywhere'" in demo

    visualizer = _read("packages/js/src/visualizer/ExplanationVisualizer.tsx")
    assert "flexWrap: 'wrap'" in visualizer
    assert "minWidth: 0" in visualizer
    assert "maxWidth: '100%'" in visualizer

    assert 'document.querySelectorAll<HTMLElement>("*")' in suite
    assert "documentWidth: document.documentElement.scrollWidth" in suite
    assert "offenders: []" in suite


def test_react_peer_floor_avoids_the_shared_npm_cache_post_action():
    workflow = _read(".github/workflows/js-ci.yml")
    build_job = workflow.split("  build:", 1)[1].split("\n  react-peer-floor:", 1)[0]
    peer_floor_job = workflow.split("  react-peer-floor:", 1)[1].split("\n  real-browser-demo:", 1)[
        0
    ]
    browser_job = workflow.split("  real-browser-demo:", 1)[1]

    assert "cache: npm" in build_job
    assert "cache-dependency-path: packages/js/package-lock.json" in build_job
    assert "cache: npm" not in peer_floor_job
    assert "cache-dependency-path:" not in peer_floor_job
    assert "cache: npm" in browser_job
    assert "cache-dependency-path: packages/js/package-lock.json" in browser_job


def test_javascript_ci_publish_and_deploy_gates_run_the_high_severity_audit():
    audited_gates = {
        ".github/workflows/js-ci.yml": "npm test",
        ".github/workflows/publish-pypi.yml": "npm test",
        ".github/workflows/deploy-demo.yml": "npm run build:demo",
    }
    command = "npm audit --audit-level=high"
    for path, downstream_gate in audited_gates.items():
        workflow = _read(path)
        assert workflow.count(command) == 1
        assert workflow.index("npm ci") < workflow.index(command)
        assert workflow.index(command) < workflow.index(downstream_gate)


def test_locked_nanoid_is_outside_the_zero_size_denial_of_service_range():
    lock = json.loads(_read("packages/js/package-lock.json"))
    nanoid = lock["packages"]["node_modules/nanoid"]
    version = tuple(int(part) for part in nanoid["version"].split("."))

    assert version >= (3, 3, 18)
    assert nanoid["dev"] is True
    assert nanoid["resolved"] == (
        f"https://registry.npmjs.org/nanoid/-/nanoid-{nanoid['version']}.tgz"
    )


def test_private_javascript_tarball_and_deployment_evidence_are_fail_closed():
    package = json.loads(_read("packages/js/package.json"))
    allowlist = json.loads(_read("packages/js/npm-pack-allowlist.json"))
    guard = _read("packages/js/scripts/check-package-boundary.mjs")
    assert package["scripts"]["check:package-boundary"] == (
        "node scripts/check-package-boundary.mjs"
    )
    assert len(allowlist) == 16 and len(allowlist) == len(set(allowlist))
    assert "publishConfig" in guard
    assert "FORBIDDEN_LIFECYCLE_SCRIPTS" in guard
    assert "npm tarball differs from reviewed allowlist" in guard
    for workflow_path in (
        ".github/workflows/js-ci.yml",
        ".github/workflows/publish-pypi.yml",
        ".github/workflows/deploy-demo.yml",
    ):
        assert "npm run check:package-boundary" in _read(workflow_path)

    deploy = _read(".github/workflows/deploy-demo.yml")
    for command in ("npm test", "npm run typecheck", "npm run lint", "npm run build"):
        assert deploy.index(command) < deploy.index("npm run build:demo")
    assert "demo-files.sha256" in deploy
    assert "demo-tree.sha256" in deploy
    assert 'test "$demo_file_count" -gt 0' in deploy
    assert "xargs -0 -r sha256sum" in deploy
    assert "deployment-evidence.json" in deploy
    assert "retention-days: 180" in deploy


def test_bundle_budget_and_untyped_distribution_guards_fail_closed():
    budget = json.loads(_read("packages/js/bundle-budget.json"))
    assert budget["maxTotalOutputBytes"] <= 225280
    assert budget["maxTotalJavaScriptGzipBytes"] <= 73728
    assert (
        "npm run check:bundle"
        in json.loads(_read("packages/js/package.json"))["scripts"]["test:browser"]
    )

    typing = json.loads(_read(".github/typing-readiness-policy.json"))
    assert typing["claim_status"] == "blocked"
    assert not (ROOT / typing["forbidden_marker"]).exists()
    python_workflow = _read(".github/workflows/python-ci.yml")
    assert python_workflow.count("scripts/audit_typing_readiness.py") == 2
    assert python_workflow.count('--distribution "$wheel_path"') == 1
    assert python_workflow.count('--distribution "$sdist_path"') == 1


def test_manual_accessibility_gate_cannot_pass_without_both_at_profiles():
    policy = json.loads(_read(".github/accessibility-certification-policy.json"))
    assert policy["claim_status"] == "blocked_pending_manual_evidence"
    assert policy["max_evidence_age_days"] == 180
    assert {profile["id"] for profile in policy["required_profiles"]} == {
        "macos-safari-voiceover",
        "windows-edge-nvda",
    }
    assert policy["required_scenarios"] == [
        "disclosure-and-landmarks",
        "keyboard-order-and-focus",
        "task-and-class-state-change",
        "signed-attribution-semantics",
        "add-and-remove-feature",
        "empty-state-announcement",
        "zoom-and-320-css-px-reflow",
    ]
    assert set(policy["required_artifact_kinds"]) == {
        "interaction-transcript",
        "screen-recording",
    }

    workflow = _read(".github/workflows/accessibility-certification.yml")
    assert "Validate manual assistive-technology evidence" in workflow
    assert "scripts/validate_accessibility_evidence.py" in workflow
    assert "EXPECTED_COMMIT: ${{ github.sha }}" in workflow
    assert '--expected-commit "$EXPECTED_COMMIT"' in workflow
    assert "git ls-files --error-unmatch" in workflow
    assert 'test ! -L "$EVIDENCE_PATH"' in workflow
    assert 'realpath -- "$EVIDENCE_PATH"' in workflow
    assert '"$GITHUB_WORKSPACE"/*' in workflow
    assert "evidence-manifest.json" in workflow
    assert "accessibility-certification-policy.json" in workflow
    assert "evidence-files.sha256" in workflow
    assert "path: artifacts/accessibility/*" in workflow
    assert "retention-days: 180" in workflow
    assert "contents: read" in workflow
