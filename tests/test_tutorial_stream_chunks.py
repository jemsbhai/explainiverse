"""Tutorial replay compares stream content without transport-chunk artifacts."""

import copy
import importlib.util
import sys
from pathlib import Path

import nbformat
import pytest

RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "execute_tutorials.py"
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "explainiverse_tutorial_stream_runner", RUNNER_PATH
)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
runner = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = runner
RUNNER_SPEC.loader.exec_module(runner)


def _stream(text, name="stdout"):
    return nbformat.v4.new_output("stream", name=name, text=text)


def _notebook(outputs, execution_count=1):
    return nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "print('reviewed output')", execution_count=execution_count, outputs=outputs
            )
        ]
    )


@pytest.mark.parametrize("name", ["stdout", "stderr"])
@pytest.mark.parametrize(
    "chunks",
    [
        ["Fresh explainers with identical seeds produced identical coefficients.", "\n"],
        ["a", "b", "c\n"],
        ["10%", "\r20%", "\n"],
        ["prefix", "", "suffix\n"],
    ],
)
def test_identical_stream_text_may_have_different_chunk_boundaries(name, chunks):
    published = _notebook([_stream(chunk, name) for chunk in chunks])
    executed = _notebook([_stream("".join(chunks), name)])
    before_published = copy.deepcopy(published)
    before_executed = copy.deepcopy(executed)
    runner._validate_reexecuted_outputs(Path("deterministic.ipynb"), published, executed)
    assert published == before_published
    assert executed == before_executed


@pytest.mark.parametrize(
    "published_outputs, executed_outputs",
    [
        ([_stream("a\n")], [_stream("b\n")]),
        ([_stream("a\n")], [_stream("a\n", "stderr")]),
        (
            [_stream("a"), _stream("warning", "stderr"), _stream("b")],
            [_stream("ab"), _stream("warning", "stderr")],
        ),
        (
            [
                _stream("a"),
                nbformat.v4.new_output("display_data", data={"text/plain": "rich"}),
                _stream("b"),
            ],
            [_stream("ab"), nbformat.v4.new_output("display_data", data={"text/plain": "rich"})],
        ),
    ],
)
def test_content_channel_and_output_order_changes_remain_rejected(
    published_outputs, executed_outputs
):
    with pytest.raises(ValueError, match="no longer matches"):
        runner._validate_reexecuted_outputs(
            Path("deterministic.ipynb"), _notebook(published_outputs), _notebook(executed_outputs)
        )


def test_execution_count_changes_remain_rejected():
    with pytest.raises(ValueError, match="no longer matches"):
        runner._validate_reexecuted_outputs(
            Path("deterministic.ipynb"),
            _notebook([_stream("same\n")], execution_count=1),
            _notebook([_stream("same\n")], execution_count=2),
        )


def test_streams_are_not_coalesced_across_code_cells():
    published = _notebook([_stream("a")])
    published.cells.append(
        nbformat.v4.new_code_cell("print('second')", execution_count=2, outputs=[_stream("b")])
    )
    executed = copy.deepcopy(published)
    executed.cells[0].outputs = [_stream("ab")]
    executed.cells[1].outputs = []
    with pytest.raises(ValueError, match="no longer matches"):
        runner._validate_reexecuted_outputs(Path("deterministic.ipynb"), published, executed)


def test_provenance_accepts_rechunking_but_rejects_changed_stream_content():
    path = runner.DEFAULT_NOTEBOOKS[0]
    published = _notebook([_stream("reviewed value"), _stream("\n")])
    declared_version = runner._declared_project_version()
    runner._publish_record(
        path,
        published,
        "2026-11-09T12:00:00+00:00",
        expected_version=declared_version,
    )
    runner._validate_published_provenance(path, published, expected_version=declared_version)
    before = copy.deepcopy(published)
    rechunked = copy.deepcopy(published)
    rechunked.cells[0].outputs = [_stream("reviewed value\n")]
    runner._validate_published_provenance(path, rechunked, expected_version=declared_version)
    assert runner._published_outputs_digest(rechunked) == runner._published_outputs_digest(
        published
    )
    assert published == before

    tampered = copy.deepcopy(rechunked)
    tampered.cells[0].outputs[0].text = "changed value\n"
    with pytest.raises(ValueError, match="published_outputs_sha256"):
        runner._validate_published_provenance(path, tampered, expected_version=declared_version)
