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
        "04_finite_estimator_uncertainty.ipynb",
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


def test_clean_execution_must_reproduce_published_outputs():
    path = Path("deterministic.ipynb")
    published = _notebook_with_code("print('fresh deterministic value')")
    published.cells[0].execution_count = 1
    published.cells[0].outputs = [
        nbformat.v4.new_output("stream", name="stdout", text="stale value\n")
    ]
    executed = copy.deepcopy(published)
    executed.cells[0].outputs = [
        nbformat.v4.new_output(
            "stream",
            name="stdout",
            text="fresh deterministic value\n",
        )
    ]

    with pytest.raises(ValueError, match="no longer matches the published outputs") as exc_info:
        runner._validate_reexecuted_outputs(path, published, executed)
    message = str(exc_info.value)
    assert "published_sha256=" in message
    assert "executed_sha256=" in message
    assert "first_differing_code_cell=1" in message
    assert "stale value" in message
    assert "fresh deterministic value" in message

    published.cells[0].outputs = copy.deepcopy(executed.cells[0].outputs)
    runner._validate_reexecuted_outputs(path, published, executed)


def test_clean_execution_ignores_adjacent_same_stream_chunk_boundaries():
    path = Path("stream-chunks.ipynb")
    published = _notebook_with_code("print('first'); print('second')")
    published.cells[0].execution_count = 1
    published.cells[0].outputs = [
        nbformat.v4.new_output(
            "stream",
            name="stdout",
            text="first\nsecond\n",
        )
    ]
    executed = copy.deepcopy(published)
    executed.cells[0].outputs = [
        nbformat.v4.new_output("stream", name="stdout", text="first"),
        nbformat.v4.new_output("stream", name="stdout", text="\nsecond\n"),
    ]

    runner._validate_reexecuted_outputs(path, published, executed)
    assert len(executed.cells[0].outputs) == 2


@pytest.mark.parametrize(
    "executed_outputs",
    [
        [
            nbformat.v4.new_output("stream", name="stdout", text="first"),
            nbformat.v4.new_output("stream", name="stdout", text=" changed\n"),
        ],
        [
            nbformat.v4.new_output("stream", name="stdout", text="first"),
            nbformat.v4.new_output("stream", name="stderr", text="\nsecond\n"),
        ],
        [
            nbformat.v4.new_output("stream", name="stdout", text="first\n"),
            nbformat.v4.new_output(
                "display_data",
                data={"text/plain": "second\n"},
                metadata={},
            ),
        ],
        [
            nbformat.v4.new_output("stream", name="stdout", text="first"),
            nbformat.v4.new_output(
                "display_data",
                data={"text/plain": "intervening output"},
                metadata={},
            ),
            nbformat.v4.new_output("stream", name="stdout", text="\nsecond\n"),
        ],
    ],
    ids=[
        "changed-text",
        "different-stream-name",
        "different-output-type",
        "intervening-output",
    ],
)
def test_stream_chunk_canonicalization_preserves_semantic_boundaries(executed_outputs):
    path = Path("stream-boundaries.ipynb")
    published = _notebook_with_code("print('first'); print('second')")
    published.cells[0].execution_count = 1
    published.cells[0].outputs = [
        nbformat.v4.new_output(
            "stream",
            name="stdout",
            text="first\nsecond\n",
        )
    ]
    executed = copy.deepcopy(published)
    executed.cells[0].outputs = executed_outputs

    with pytest.raises(ValueError, match="no longer matches the published outputs"):
        runner._validate_reexecuted_outputs(path, published, executed)


def test_checkout_import_and_hashes_are_current_and_stable():
    assert runner._assert_checkout_import() == runner._declared_project_version()
    assert runner._lock_digest() == runner._lock_digest()
    assert runner._runner_digest() == runner._runner_digest()
    assert runner._package_source_digest() == runner._package_source_digest()


def test_kernelshap_plot_is_a_platform_neutral_accessible_svg():
    path = runner.TUTORIAL_DIR / "02_kernelshap.ipynb"
    notebook = nbformat.read(path, as_version=4)
    runner._validate_source(path, notebook)
    runner._validate_published_provenance(path, notebook)

    source = "\n".join(cell.source for cell in notebook.cells)
    assert "display(SVG(svg))" in source
    assert "matplotlib.pyplot" not in source

    rich_outputs = [
        output
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
        if output.output_type == "display_data"
    ]
    assert len(rich_outputs) == 1
    assert "image/png" not in rich_outputs[0].data
    svg = rich_outputs[0].data["image/svg+xml"]
    assert 'role="img"' in svg
    assert "<title" in svg
    assert "<desc" in svg


def test_treeshap_plot_is_a_platform_neutral_accessible_svg():
    path = runner.TUTORIAL_DIR / "03_treeshap.ipynb"
    notebook = nbformat.read(path, as_version=4)
    runner._validate_source(path, notebook)

    source = "\n".join(cell.source for cell in notebook.cells)
    assert "display(SVG(svg))" in source
    assert "matplotlib.pyplot" not in source

    rich_outputs = [
        output
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
        if output.output_type == "display_data"
    ]
    assert len(rich_outputs) == 1
    assert "image/png" not in rich_outputs[0].data
    svg = rich_outputs[0].data["image/svg+xml"]
    assert 'role="img"' in svg
    assert "<title" in svg
    assert "<desc" in svg


def test_source_only_verification_requires_absent_distribution_metadata(monkeypatch):
    declared_version = runner._declared_project_version()

    def missing_distribution(_name):
        raise runner.PackageNotFoundError

    monkeypatch.setattr(runner, "version", missing_distribution)
    assert runner._assert_checkout_import(source_only=True) == declared_version

    path = runner.DEFAULT_NOTEBOOKS[0]
    notebook = nbformat.read(path, as_version=4)
    runner._validate_published_provenance(path, notebook, expected_version=declared_version)

    monkeypatch.setattr(runner, "version", lambda _name: declared_version)
    with pytest.raises(RuntimeError, match="distribution.*absent"):
        runner._assert_checkout_import(source_only=True)


def test_source_only_mode_cannot_publish_notebooks():
    with pytest.raises(ValueError, match="source-only.*write"):
        runner.main(["--source-only", "--write"])


def test_checkout_import_is_bound_to_project_metadata(monkeypatch, tmp_path):
    project = tmp_path / "pyproject.toml"
    project.write_text(
        '[project]\nname = "explainiverse"\nversion = "99.0.0"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "PROJECT_FILE", project)
    with pytest.raises(RuntimeError, match="pyproject.toml.*__version__ disagree"):
        runner._assert_checkout_import()


def test_uncertainty_tutorial_publishes_the_scoped_formula_and_fresh_equality_contracts():
    path = runner.TUTORIAL_DIR / "04_finite_estimator_uncertainty.ipynb"
    notebook = nbformat.read(path, as_version=4)
    runner._validate_source(path, notebook)
    runner._validate_published_provenance(path, notebook)

    source = "\n".join(cell.source for cell in notebook.cells)
    assert "load_iris()" in source
    assert "model_probabilities, formula_probabilities" in source
    assert "run_seeded_replicates(" in source
    assert "evaluate_intervention_sensitivity(" in source
    assert 'replicate_report["finite_estimate_is_global_proof"] is False' in source
    assert 'sensitivity_report["universal_default_claimed"] is False' in source
    assert "fresh_replicate_report == replicate_report" in source
    assert "fresh_sensitivity_report == sensitivity_report" in source

    published_text = "\n".join(
        output.get("text", "")
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
        if output.output_type == "stream"
    )
    assert "fresh seeded report is exactly equal: True" in published_text
    assert "same sign across references: False" in published_text
    assert "fresh sensitivity report is exactly equal: True" in published_text
