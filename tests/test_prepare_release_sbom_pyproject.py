"""Tests for the PEP 621-only release SBOM manifest."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised by the Python 3.10 CI lane
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_release_sbom_pyproject.py"
SPEC = importlib.util.spec_from_file_location("prepare_release_sbom_pyproject", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
sbom_manifest = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sbom_manifest
SPEC.loader.exec_module(sbom_manifest)


def test_repository_manifest_preserves_project_and_removes_all_poetry_tables():
    source = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    prepared = sbom_manifest.prepare_sbom_pyproject(source)

    original_data = tomllib.loads(source)
    prepared_data = tomllib.loads(prepared)
    assert prepared_data["project"] == original_data["project"]
    assert "poetry" not in prepared_data["tool"]
    assert prepared_data["tool"]["pytest"] == original_data["tool"]["pytest"]
    assert prepared_data["tool"]["mypy"] == original_data["tool"]["mypy"]


def test_preparation_preserves_other_tables_after_poetry_array_tables():
    source = """\
[project]
name = "example"
version = "1.0.0"

[tool.poetry]
packages = [{ include = "example", from = "src" }]

[[tool.poetry.include]]
path = "data"

[tool.black]
line-length = 100

[[tool.mypy.overrides]]
module = ["example.*"]
ignore_missing_imports = true
"""

    prepared = sbom_manifest.prepare_sbom_pyproject(source)
    parsed = tomllib.loads(prepared)

    assert parsed["project"]["name"] == "example"
    assert "poetry" not in parsed["tool"]
    assert parsed["tool"]["black"]["line-length"] == 100
    assert parsed["tool"]["mypy"]["overrides"][0]["module"] == ["example.*"]


def test_preparation_preserves_quoted_and_whitespace_separated_table_keys():
    source = """\
[project]
name = "example"
version = "1.0.0"

[tool.poetry]
packages = [{ include = "example", from = "src" }]

[tool . "black"]
line-length = 100

[[tool . 'mypy' . overrides]]
module = ["example.*"]
ignore_missing_imports = true
"""

    prepared = sbom_manifest.prepare_sbom_pyproject(source)
    parsed = tomllib.loads(prepared)

    assert "poetry" not in parsed["tool"]
    assert parsed["tool"]["black"]["line-length"] == 100
    assert parsed["tool"]["mypy"]["overrides"][0]["module"] == ["example.*"]


def test_preparation_fails_closed_if_dotted_poetry_metadata_remains():
    source = """\
tool.poetry.packages = []

[project]
name = "example"
version = "1.0.0"
"""

    with pytest.raises(ValueError, match="still contains tool.poetry"):
        sbom_manifest.prepare_sbom_pyproject(source)


@pytest.mark.parametrize(
    "source",
    [
        "[tool.black]\nline-length = 100\n",
        "[project]\nversion = '1.0.0'\n",
    ],
)
def test_preparation_requires_named_pep621_project(source):
    with pytest.raises(ValueError, match="PEP 621 project metadata"):
        sbom_manifest.prepare_sbom_pyproject(source)


def test_cli_refuses_to_overwrite_the_reviewed_source(tmp_path):
    source = tmp_path / "pyproject.toml"
    source.write_text("[project]\nname='example'\nversion='1.0.0'\n", encoding="utf-8")

    with pytest.raises(ValueError, match="output must differ"):
        sbom_manifest.main(["--source", str(source), "--output", str(source)])
