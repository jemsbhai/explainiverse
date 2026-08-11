"""Fail closed unless a release workflow was dispatched from its release tag.

The GitHub artifact-attestation action derives source provenance from the
workflow run's OIDC claims, not from a later checkout ref.  A workflow that
checks out a tag while being dispatched from ``main`` would therefore produce
misleading source claims.  This guard binds the dispatch ref and SHA to the
checked-out release commit before any release gate or build runs.
"""

from __future__ import annotations

import os
import re
import subprocess

_STABLE_TAG = re.compile(r"v\d+\.\d+\.\d+")
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


def validate_release_attestation_context(
    *,
    release_tag: str,
    github_ref: str,
    github_sha: str,
    checkout_sha: str,
) -> None:
    """Validate that GitHub's provenance identity matches the release checkout."""
    if _STABLE_TAG.fullmatch(release_tag) is None:
        raise ValueError("release tag must have the form vMAJOR.MINOR.PATCH")
    expected_ref = f"refs/tags/{release_tag}"
    if github_ref != expected_ref:
        raise ValueError(f"workflow must be dispatched from {expected_ref!r}; got {github_ref!r}")
    normalized_github_sha = github_sha.strip().lower()
    normalized_checkout_sha = checkout_sha.strip().lower()
    if _COMMIT_SHA.fullmatch(normalized_github_sha) is None:
        raise ValueError("GITHUB_SHA must be a complete 40-character commit SHA")
    if _COMMIT_SHA.fullmatch(normalized_checkout_sha) is None:
        raise ValueError("checkout SHA must be a complete 40-character commit SHA")
    if normalized_github_sha != normalized_checkout_sha:
        raise ValueError(
            "workflow dispatch SHA does not match the checked-out release commit: "
            f"{normalized_github_sha!r} != {normalized_checkout_sha!r}"
        )


def main() -> None:
    """Validate the current GitHub Actions environment and checkout."""
    required = ("RELEASE_TAG", "GITHUB_REF", "GITHUB_SHA")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"missing required release environment: {', '.join(missing)}")
    checkout_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    try:
        validate_release_attestation_context(
            release_tag=os.environ["RELEASE_TAG"],
            github_ref=os.environ["GITHUB_REF"],
            github_sha=os.environ["GITHUB_SHA"],
            checkout_sha=checkout_sha,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
