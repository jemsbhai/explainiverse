"""Fail closed unless a stable release version is absent from PyPI.

This guard runs once before release build work and again immediately before the
OIDC publisher action.  HTTP 404 is the only accepted absence signal; an
existing release or any network/API ambiguity fails without invoking a
publisher or relying on ``skip-existing``.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Sequence

_PROJECT = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?")
_TAG = re.compile(r"v(\d+\.\d+\.\d+)")


def version_from_tag(tag: str) -> str:
    """Return a stable public version from an exact semantic-version tag."""
    match = _TAG.fullmatch(tag)
    if match is None:
        raise ValueError("release tag must have the form vMAJOR.MINOR.PATCH")
    return match.group(1)


def require_pypi_version_absent(
    project: str,
    version: str,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> None:
    """Accept only PyPI's authoritative 404 for an unpublished version."""
    if _PROJECT.fullmatch(project) is None:
        raise ValueError(f"invalid PyPI project name {project!r}")
    url = (
        "https://pypi.org/pypi/"
        f"{urllib.parse.quote(project, safe='')}/{urllib.parse.quote(version, safe='')}/json"
    )
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "explainiverse-release-absence-guard",
        },
    )
    try:
        with opener(request, timeout=30) as response:
            status = getattr(response, "status", None)
            if status != 200:
                raise RuntimeError(f"PyPI returned unexpected HTTP status {status!r} for {url}")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return
        raise RuntimeError(f"PyPI release lookup returned HTTP {exc.code}") from exc
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(f"PyPI release lookup could not be completed: {exc}") from exc
    raise ValueError(
        f"PyPI already contains {project} {version}; use the recovery-only workflow and "
        "do not invoke the publisher again"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="explainiverse")
    parser.add_argument("--tag", required=True)
    args = parser.parse_args(argv)
    try:
        version = version_from_tag(args.tag)
        require_pypi_version_absent(args.project, version)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"PyPI absence verified for {args.project} {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
