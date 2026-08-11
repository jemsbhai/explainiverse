"""Adversarial tests for the outside-bound dependency source probe."""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "record_dependency_candidate_probe.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "explainiverse_dependency_candidate_probe", SCRIPT_PATH
)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
probe = importlib.util.module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = probe
SCRIPT_SPEC.loader.exec_module(probe)


def _project(tmp_path: Path, requirement: str = "scikit-image>=0.20,<1.0") -> Path:
    path = tmp_path / "pyproject.toml"
    path.write_text(
        "\n".join(
            (
                "[project]",
                'name = "explainiverse"',
                'version = "0.15.0"',
                f'dependencies = ["{requirement}"]',
                "",
            )
        ),
        encoding="utf-8",
    )
    return path


def _wheel(tmp_path: Path, requirement: str = "scikit-image>=0.20,<1.0") -> Path:
    path = tmp_path / "explainiverse-0.15.0-py3-none-any.whl"
    metadata = "\n".join(
        (
            "Metadata-Version: 2.4",
            "Name: explainiverse",
            "Version: 0.15.0",
            f"Requires-Dist: {requirement}",
            "",
        )
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("explainiverse-0.15.0.dist-info/METADATA", metadata)
    return path


def _versions(monkeypatch, *, project: str | None = None, candidate: str = "1.0.0rc1"):
    def installed(name: str) -> str:
        if name == "explainiverse":
            if project is None:
                raise probe.PackageNotFoundError(name)
            return project
        if name == "scikit-image":
            return candidate
        raise probe.PackageNotFoundError(name)

    monkeypatch.setattr(probe, "installed_version", installed)


def test_probe_records_source_and_dependency_only_scope(monkeypatch, tmp_path):
    _versions(monkeypatch)
    record = probe.build_probe_record(
        project_file=_project(tmp_path),
        package="scikit-image",
        candidate="1.0.0rc1",
        wheel=_wheel(tmp_path),
    )

    assert record["probe_scope"] == "source-compatibility-only"
    assert record["post_candidate_graph_scope"] == "dependencies-only"
    assert record["project_distribution_installed"] is False
    assert record["full_distribution_graph_verified"] is False
    assert record["candidate_satisfies_checked_in_requirement"] is False
    assert record["wheel_metadata"] == {
        "path": str(tmp_path / "explainiverse-0.15.0-py3-none-any.whl"),
        "requirement": "scikit-image<1.0,>=0.20",
        "candidate_satisfies_requirement": False,
    }


def test_probe_refuses_an_installed_project_distribution(monkeypatch, tmp_path):
    _versions(monkeypatch, project="0.15.0")
    with pytest.raises(ValueError, match="distribution.*absent"):
        probe.build_probe_record(
            project_file=_project(tmp_path),
            package="scikit-image",
            candidate="1.0.0rc1",
        )


@pytest.mark.parametrize(
    ("candidate", "observed", "message"),
    (
        ("1.0.0", "1.0.0", "must be a prerelease"),
        ("0.99.0rc1", "0.99.0rc1", "unexpectedly satisfies"),
        ("1.0.0rc1", "1.0.0rc2", "does not match candidate"),
    ),
)
def test_probe_rejects_invalid_or_unbound_candidate(
    monkeypatch, tmp_path, candidate, observed, message
):
    _versions(monkeypatch, candidate=observed)
    with pytest.raises(ValueError, match=message):
        probe.build_probe_record(
            project_file=_project(tmp_path),
            package="scikit-image",
            candidate=candidate,
        )


def test_probe_refuses_wheel_metadata_that_widens_the_reviewed_bound(monkeypatch, tmp_path):
    _versions(monkeypatch)
    with pytest.raises(ValueError, match="wheel dependency requirement differs"):
        probe.build_probe_record(
            project_file=_project(tmp_path),
            package="scikit-image",
            candidate="1.0.0rc1",
            wheel=_wheel(tmp_path, "scikit-image>=0.20"),
        )
