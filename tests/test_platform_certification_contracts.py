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
    assert {profile["id"] for profile in policy["required_profiles"]} == {
        "macos-safari-voiceover",
        "windows-edge-nvda",
    }
    assert set(policy["required_artifact_kinds"]) == {
        "interaction-transcript",
        "screen-recording",
    }

    workflow = _read(".github/workflows/accessibility-certification.yml")
    assert "Validate manual assistive-technology evidence" in workflow
    assert "scripts/validate_accessibility_evidence.py" in workflow
    assert "EXPECTED_COMMIT: ${{ github.sha }}" in workflow
    assert '--expected-commit "$EXPECTED_COMMIT"' in workflow
    assert "retention-days: 180" in workflow
    assert "contents: read" in workflow
