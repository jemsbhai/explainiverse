"""Accuracy and integrity tests for the reviewed tutorial execution harness."""

from __future__ import annotations

import copy
import importlib.util
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "execute_tutorials.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("explainiverse_tutorial_runner", RUNNER_PATH)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
runner = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = runner
RUNNER_SPEC.loader.exec_module(runner)


def _notebook_with_code(source: str):
    return nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)])


def test_reviewed_manifest_exactly_matches_notebook_inventory():
    runner._validate_manifest_inventory()
    assert {path.name for path in runner.DEFAULT_NOTEBOOKS} == {
        path.name for path in runner.TUTORIAL_DIR.glob("*.ipynb")
    }
    assert {spec.filename for spec in runner.TUTORIAL_SPECS} == {
        "01_lime_tabular.ipynb",
        "02_kernelshap.ipynb",
        "03_treeshap.ipynb",
    }


@pytest.mark.parametrize(
    "source",
    [
        "import socket\nsocket.create_connection(('example.com', 443))",
        "from sklearn.datasets import fetch_california_housing\nfetch_california_housing()",
        "import subprocess\nsubprocess.run(['python', '-m', 'pip', 'install', 'x'])",
        "import os\nos.system('curl https://example.com')",
        "__import__('requests').get('https://example.com')",
    ],
)
def test_source_policy_rejects_network_and_install_bypasses(source):
    with pytest.raises(ValueError, match="forbidden|network|package-install"):
        runner._validate_source(Path("untrusted.ipynb"), _notebook_with_code(source))


def test_source_policy_allows_local_sklearn_dataset():
    notebook = _notebook_with_code(
        "from sklearn.datasets import load_iris\niris = load_iris()\nassert len(iris.data)"
    )
    runner._validate_source(Path("local.ipynb"), notebook)


def test_hidden_socket_guard_rejects_non_loopback_before_connecting():
    check = f"""
{runner.OFFLINE_BOOTSTRAP}
try:
    _explainiverse_socket.create_connection(("example.com", 443), timeout=0.01)
except RuntimeError as exc:
    assert "non-loopback" in str(exc)
else:
    raise AssertionError("network guard did not reject an external address")
"""
    completed = subprocess.run(
        [sys.executable, "-c", check],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr


def test_only_manifest_notebooks_can_be_selected():
    unreviewed = runner.TUTORIAL_DIR / "99_unreviewed.ipynb"
    with pytest.raises(ValueError, match="not in the reviewed execution manifest"):
        runner._resolve_notebooks([str(unreviewed)])


@pytest.mark.parametrize(
    "filename",
    [spec.filename for spec in runner.TUTORIAL_SPECS],
)
def test_published_provenance_binds_source_outputs_and_full_schema(filename):
    path = runner.TUTORIAL_DIR / filename
    notebook = nbformat.read(path, as_version=4)
    runner._validate_published_provenance(path, notebook)

    changed_source = copy.deepcopy(notebook)
    changed_source.cells[0].source += "\nSource tampering."
    with pytest.raises(ValueError, match="notebook_source_sha256"):
        runner._validate_published_provenance(path, changed_source)

    changed_output = copy.deepcopy(notebook)
    code_cell = next(cell for cell in changed_output.cells if cell.cell_type == "code")
    code_cell.outputs = []
    with pytest.raises(ValueError, match="published_outputs_sha256"):
        runner._validate_published_provenance(path, changed_output)

    changed_seed = copy.deepcopy(notebook)
    changed_seed.metadata.explainiverse_execution["deterministic_seed"] += 1
    with pytest.raises(ValueError, match="deterministic_seed"):
        runner._validate_published_provenance(path, changed_seed)

    changed_timestamp = copy.deepcopy(notebook)
    changed_timestamp.metadata.explainiverse_execution["executed_at_utc"] = "not-a-timestamp"
    with pytest.raises(ValueError, match="ISO-8601"):
        runner._validate_published_provenance(path, changed_timestamp)

    non_utc_timestamp = copy.deepcopy(notebook)
    non_utc_timestamp.metadata.explainiverse_execution["executed_at_utc"] = "2026-08-09T12:00:00"
    with pytest.raises(ValueError, match="UTC timezone"):
        runner._validate_published_provenance(path, non_utc_timestamp)

    missing_python = copy.deepcopy(notebook)
    missing_python.metadata.explainiverse_execution["python_version"] = ""
    with pytest.raises(ValueError, match="python_version"):
        runner._validate_published_provenance(path, missing_python)

    missing_platform = copy.deepcopy(notebook)
    missing_platform.metadata.explainiverse_execution["platform"] = ""
    with pytest.raises(ValueError, match="platform"):
        runner._validate_published_provenance(path, missing_platform)


def test_published_output_cannot_disclose_checkout_path():
    path = runner.DEFAULT_NOTEBOOKS[0]
    notebook = nbformat.read(path, as_version=4)
    changed = copy.deepcopy(notebook)
    code_cell = next(cell for cell in changed.cells if cell.cell_type == "code")
    code_cell.outputs = [
        nbformat.v4.new_output("stream", name="stdout", text=f"loaded {runner.ROOT}")
    ]
    with pytest.raises(ValueError, match="local checkout path"):
        runner._validate_published_outputs(path, changed)


def test_checkout_import_and_hashes_are_current_and_stable():
    runner._assert_checkout_import()
    assert runner._lock_digest() == runner._lock_digest()
    assert runner._runner_digest() == runner._runner_digest()
    assert runner._package_source_digest() == runner._package_source_digest()
