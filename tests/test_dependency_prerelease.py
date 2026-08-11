"""Selection logic for scheduled next-major compatibility probes."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "select_dependency_prerelease.py"
SPEC = importlib.util.spec_from_file_location("select_dependency_prerelease", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
selector = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = selector
SPEC.loader.exec_module(selector)


def test_selects_latest_non_yanked_prerelease_in_exact_next_major():
    metadata = {
        "releases": {
            "0.25.0": [{"yanked": False}],
            "1.0.0a2": [{"yanked": False}],
            "1.0.0a1": [{"yanked": False}],
            "1.0.0rc1": [{"yanked": False}],
            "1.0.0": [{"yanked": False}],
            "2.0.0a1": [{"yanked": False}],
        }
    }
    assert selector.select_next_major_prerelease(metadata, current_major=0) == "1.0.0rc1"


def test_yanked_or_final_releases_do_not_satisfy_the_prerelease_gate():
    metadata = {
        "releases": {
            "1.0.0a1": [{"yanked": True}],
            "1.0.0": [{"yanked": False}],
        }
    }
    assert selector.select_next_major_prerelease(metadata, current_major=0) is None


@pytest.mark.parametrize("releases", [None, [], "bad"])
def test_malformed_pypi_release_index_fails_closed(releases):
    with pytest.raises(ValueError, match="releases mapping"):
        selector.select_next_major_prerelease({"releases": releases}, current_major=0)


def test_cli_has_distinct_blocked_status_when_no_candidate_exists(tmp_path):
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"releases": {"0.25.0": [{}]}}), encoding="utf-8")
    assert (
        selector.main(
            [
                "--package",
                "scikit-image",
                "--current-major",
                "0",
                "--metadata",
                str(metadata),
            ]
        )
        == 3
    )
