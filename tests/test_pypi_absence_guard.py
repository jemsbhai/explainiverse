"""A repeat release must fail before the sole PyPI publisher can run."""

from __future__ import annotations

import importlib.util
import sys
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_pypi_version_absent", ROOT / "scripts" / "check_pypi_version_absent.py"
)
assert SPEC is not None and SPEC.loader is not None
guard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = guard
SPEC.loader.exec_module(guard)


class _Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        del args
        return False


def _http_error(code):
    return urllib.error.HTTPError("https://pypi.org/", code, "error", hdrs=None, fp=None)


def test_only_authoritative_404_is_accepted_as_unpublished():
    calls = []

    def absent(request, *, timeout):
        calls.append((request.full_url, request.get_header("User-agent"), timeout))
        raise _http_error(404)

    guard.require_pypi_version_absent("explainiverse", "0.15.0", opener=absent)

    assert calls == [
        (
            "https://pypi.org/pypi/explainiverse/0.15.0/json",
            "explainiverse-release-absence-guard",
            30,
        )
    ]


def test_existing_version_fails_with_recovery_only_guidance():
    with pytest.raises(ValueError, match="already contains.*recovery-only"):
        guard.require_pypi_version_absent(
            "explainiverse", "0.15.0", opener=lambda request, timeout: _Response()
        )


@pytest.mark.parametrize(
    "failure",
    [_http_error(500), urllib.error.URLError("offline")],
    ids=("http-500", "url-error"),
)
def test_network_or_api_ambiguity_fails_closed(failure):
    def unavailable(request, *, timeout):
        del request, timeout
        raise failure

    with pytest.raises(RuntimeError, match="lookup"):
        guard.require_pypi_version_absent("explainiverse", "0.15.0", opener=unavailable)


@pytest.mark.parametrize("tag", ["0.15.0", "v0.15", "v0.15.0.dev0", "v1.2.3/other"])
def test_only_exact_stable_tags_can_be_looked_up(tag):
    with pytest.raises(ValueError, match="vMAJOR.MINOR.PATCH"):
        guard.version_from_tag(tag)
