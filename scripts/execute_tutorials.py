"""Publish and verify the repository's reviewed offline tutorial notebooks.

``--write`` executes every selected reviewed notebook and publishes fresh
outputs plus provenance.  The default mode first validates the stored source,
outputs, package tree, runner, dependency lock, and metadata before executing
a clean in-memory copy.  The network guard is defense in depth for reviewed
notebooks; this runner is not a hostile-code security sandbox.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import copy
import hashlib
import json
import os
import platform
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterator, Sequence

import nbformat
from nbclient import NotebookClient
from nbformat import NotebookNode

import explainiverse

ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_DIR = ROOT / "tutorials"
PACKAGE_SOURCE_DIR = ROOT / "src" / "explainiverse"
LOCK_FILE = ROOT / "poetry.lock"
PROJECT_FILE = ROOT / "pyproject.toml"
RUNNER_LABEL = "scripts/execute_tutorials.py"
PROVENANCE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class TutorialSpec:
    """Reviewed execution contract for one promoted tutorial."""

    filename: str
    deterministic_seed: int


TUTORIAL_SPECS = (
    TutorialSpec("01_lime_tabular.ipynb", 42),
    TutorialSpec("02_kernelshap.ipynb", 17),
    TutorialSpec("03_treeshap.ipynb", 17),
    TutorialSpec("04_finite_estimator_uncertainty.ipynb", 31),
)
SPEC_BY_FILENAME = {spec.filename: spec for spec in TUTORIAL_SPECS}
DEFAULT_NOTEBOOKS = tuple(TUTORIAL_DIR / spec.filename for spec in TUTORIAL_SPECS)

FORBIDDEN_TEXT = {
    "package-install command": re.compile(
        r"(?im)^\s*(?:!|%)(?:pip|conda|mamba|uv)\b|\b(?:pip|conda|mamba|uv)\s+install\b"
    ),
    "network URL": re.compile(r"\b(?:https?|ftp)://", re.IGNORECASE),
    "shell download": re.compile(r"(?im)^\s*!?\s*(?:curl|wget)\b"),
}
FORBIDDEN_IMPORT_ROOTS = {
    "aiohttp",
    "boto3",
    "ftplib",
    "httpx",
    "importlib",
    "openml",
    "pip",
    "requests",
    "socket",
    "subprocess",
    "telnetlib",
    "urllib",
    "webbrowser",
}
FORBIDDEN_CALLS = {
    "__import__",
    "eval",
    "exec",
    "os.popen",
    "os.system",
}

OFFLINE_BOOTSTRAP = """
import ipaddress as _explainiverse_ipaddress
import socket as _explainiverse_socket

_explainiverse_original_connect = _explainiverse_socket.socket.connect
_explainiverse_original_connect_ex = _explainiverse_socket.socket.connect_ex
_explainiverse_original_create_connection = _explainiverse_socket.create_connection

def _explainiverse_is_loopback(address):
    if isinstance(address, str):
        return True  # Local Unix-domain socket path.
    if not isinstance(address, tuple) or not address:
        return False
    host = address[0]
    if host == "localhost":
        return True
    try:
        return _explainiverse_ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False

def _explainiverse_guarded_connect(sock, address):
    if not _explainiverse_is_loopback(address):
        raise RuntimeError("tutorial execution blocks non-loopback socket connections")
    return _explainiverse_original_connect(sock, address)

def _explainiverse_guarded_connect_ex(sock, address):
    if not _explainiverse_is_loopback(address):
        raise RuntimeError("tutorial execution blocks non-loopback socket connections")
    return _explainiverse_original_connect_ex(sock, address)

def _explainiverse_guarded_create_connection(address, *args, **kwargs):
    if not _explainiverse_is_loopback(address):
        raise RuntimeError("tutorial execution blocks non-loopback socket connections")
    return _explainiverse_original_create_connection(address, *args, **kwargs)

_explainiverse_socket.socket.connect = _explainiverse_guarded_connect
_explainiverse_socket.socket.connect_ex = _explainiverse_guarded_connect_ex
_explainiverse_socket.create_connection = _explainiverse_guarded_create_connection
""".strip()


def _canonical_text(text: str) -> bytes:
    return text.replace("\r\n", "\n").encode("utf-8")


def _digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_digest(path: Path) -> str:
    return _digest_bytes(_canonical_text(path.read_text(encoding="utf-8")))


def _lock_digest() -> str:
    return _file_digest(LOCK_FILE)


def _runner_digest() -> str:
    return _file_digest(Path(__file__).resolve())


def _package_source_digest() -> str:
    digest = hashlib.sha256()
    paths = sorted(path for path in PACKAGE_SOURCE_DIR.rglob("*.py") if path.is_file())
    if not paths:
        raise RuntimeError(f"No package source files found under {PACKAGE_SOURCE_DIR}")
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_canonical_text(path.read_text(encoding="utf-8")))
        digest.update(b"\0")
    return digest.hexdigest()


def _json_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _digest_bytes(encoded)


def _notebook_source_digest(notebook: NotebookNode) -> str:
    metadata = copy.deepcopy(dict(notebook.metadata))
    metadata.pop("explainiverse_execution", None)
    metadata.pop("widgets", None)
    cells = []
    for cell in notebook.cells:
        cell_payload = {
            "cell_type": cell.cell_type,
            "id": cell.get("id"),
            "metadata": {
                key: value for key, value in dict(cell.metadata).items() if key != "execution"
            },
            "source": cell.source.replace("\r\n", "\n"),
        }
        if cell.cell_type == "markdown":
            cell_payload["attachments"] = copy.deepcopy(cell.get("attachments", {}))
        cells.append(cell_payload)
    return _json_digest(
        {
            "nbformat": notebook.nbformat,
            "nbformat_minor": notebook.nbformat_minor,
            "metadata": metadata,
            "cells": cells,
        }
    )


def _published_outputs_payload(notebook: NotebookNode) -> list[dict[str, Any]]:
    return [
        {
            "execution_count": cell.execution_count,
            "outputs": copy.deepcopy(cell.outputs),
        }
        for cell in notebook.cells
        if cell.cell_type == "code"
    ]


def _published_outputs_digest(notebook: NotebookNode) -> str:
    return _json_digest(_published_outputs_payload(notebook))


def _declared_project_version() -> str:
    in_project_table = False
    declared_versions: list[str] = []
    for raw_line in PROJECT_FILE.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("["):
            if in_project_table:
                break
            in_project_table = stripped == "[project]"
            continue
        if not in_project_table:
            continue
        match = re.fullmatch(r'version\s*=\s*"([^"]+)"\s*(?:#.*)?', stripped)
        if match:
            declared_versions.append(match.group(1))
    if len(declared_versions) != 1:
        raise RuntimeError(
            "pyproject.toml [project] must contain exactly one double-quoted version"
        )
    return declared_versions[0]


def _assert_checkout_import(*, source_only: bool = False) -> str:
    package_file = Path(explainiverse.__file__).resolve()
    expected_root = PACKAGE_SOURCE_DIR.resolve()
    if package_file != expected_root / "__init__.py" and expected_root not in package_file.parents:
        raise RuntimeError(
            f"Explainiverse imported from {package_file}, expected the checkout under {expected_root}"
        )
    declared_version = _declared_project_version()
    if explainiverse.__version__ != declared_version:
        raise RuntimeError("pyproject.toml and explainiverse.__version__ disagree")

    try:
        metadata_version = version("explainiverse")
    except PackageNotFoundError:
        metadata_version = None
    if source_only:
        if metadata_version is not None:
            raise RuntimeError(
                "source-only tutorial verification requires the Explainiverse distribution "
                f"to be absent; found {metadata_version}"
            )
    elif metadata_version is None:
        raise RuntimeError("Explainiverse distribution metadata is missing")
    elif metadata_version != declared_version:
        raise RuntimeError("Package metadata and the declared project version disagree")
    return declared_version


def _validate_manifest_inventory() -> None:
    expected = set(SPEC_BY_FILENAME)
    actual = {path.name for path in TUTORIAL_DIR.glob("*.ipynb") if path.is_file()}
    if actual != expected:
        raise ValueError(
            "Reviewed tutorial manifest does not match tutorials/*.ipynb: "
            f"missing={sorted(expected - actual)}, unreviewed={sorted(actual - expected)}"
        )


def _resolve_notebooks(raw_paths: Sequence[str]) -> list[Path]:
    _validate_manifest_inventory()
    paths = [Path(raw).resolve() for raw in raw_paths] if raw_paths else list(DEFAULT_NOTEBOOKS)
    reviewed = {path.resolve() for path in DEFAULT_NOTEBOOKS}
    if len(paths) != len(set(paths)):
        raise ValueError("Tutorial notebook paths must not be duplicated")
    for path in paths:
        if path not in reviewed:
            raise ValueError(f"Tutorial is not in the reviewed execution manifest: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def _dotted_call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _validate_ast_policy(tree: ast.AST, path: Path, cell_number: int) -> None:
    for node in ast.walk(tree):
        imported: list[str] = []
        if isinstance(node, ast.Import):
            imported = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported = [node.module or ""]
            if any(alias.name.startswith("fetch_") for alias in node.names):
                raise ValueError(
                    f"{path.name}: code cell {cell_number} imports a network dataset loader"
                )
        for module_name in imported:
            if module_name.split(".", 1)[0] in FORBIDDEN_IMPORT_ROOTS:
                raise ValueError(
                    f"{path.name}: code cell {cell_number} imports forbidden module {module_name!r}"
                )

        if isinstance(node, ast.Call):
            call_name = _dotted_call_name(node.func)
            leaf_name = call_name.rsplit(".", 1)[-1]
            if call_name in FORBIDDEN_CALLS or call_name.startswith("os.spawn"):
                raise ValueError(
                    f"{path.name}: code cell {cell_number} calls forbidden function {call_name!r}"
                )
            if leaf_name.startswith("fetch_"):
                raise ValueError(
                    f"{path.name}: code cell {cell_number} calls a network dataset loader"
                )


def _validate_source(path: Path, notebook: NotebookNode) -> None:
    if notebook.nbformat != 4:
        raise ValueError(f"{path.name}: expected notebook format 4, got {notebook.nbformat}")
    nbformat.validate(notebook)

    code_cells = 0
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code" or not cell.source.strip():
            continue
        code_cells += 1
        try:
            tree = ast.parse(cell.source, filename=f"{path.name}:cell-{index + 1}")
        except SyntaxError as exc:
            raise ValueError(f"{path.name}: code cell {index + 1} does not compile") from exc
        for label, pattern in FORBIDDEN_TEXT.items():
            if pattern.search(cell.source):
                raise ValueError(f"{path.name}: code cell {index + 1} contains {label}")
        _validate_ast_policy(tree, path, index + 1)

    if code_cells == 0:
        raise ValueError(f"{path.name}: tutorial contains no executable code")


def _parse_utc_timestamp(value: Any, *, path: Path) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path.name}: executed_at_utc must be a non-empty ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{path.name}: executed_at_utc is not valid ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{path.name}: executed_at_utc must include the UTC timezone")


def _validate_published_outputs(path: Path, notebook: NotebookNode) -> None:
    serialized = json.dumps(
        [cell.outputs for cell in notebook.cells if cell.cell_type == "code"],
        ensure_ascii=False,
    )
    normalised_output = serialized.replace("\\\\", "/").replace("\\", "/").lower()
    local_form = str(ROOT).replace("\\", "/").lower()
    generic_machine_path = re.search(r"(?:\b[a-z]:/|/(?:home|users|workspace)/)", normalised_output)
    if local_form in normalised_output or generic_machine_path:
        raise ValueError(f"{path.name}: published output exposes the local checkout path")
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code" or not cell.source.strip():
            continue
        if cell.execution_count is None:
            raise ValueError(f"{path.name}: code cell {index + 1} has no published execution count")
        if any(output.output_type == "error" for output in cell.outputs):
            raise ValueError(f"{path.name}: code cell {index + 1} contains a published error")


def _validate_reexecuted_outputs(
    path: Path,
    published: NotebookNode,
    executed: NotebookNode,
) -> None:
    """Require a clean execution to reproduce the reviewed stored outputs."""
    _validate_published_outputs(path, executed)
    published_digest = _published_outputs_digest(published)
    executed_digest = _published_outputs_digest(executed)
    if executed_digest != published_digest:
        published_payload = _published_outputs_payload(published)
        executed_payload = _published_outputs_payload(executed)
        differing_cell = next(
            index
            for index, (expected, actual) in enumerate(
                zip(published_payload, executed_payload, strict=True), start=1
            )
            if expected != actual
        )
        expected_output = json.dumps(
            published_payload[differing_cell - 1],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        actual_output = json.dumps(
            executed_payload[differing_cell - 1],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        diagnostic_limit = 2_000
        raise ValueError(
            f"{path.name}: clean execution no longer matches the published outputs; "
            f"published_sha256={published_digest}; executed_sha256={executed_digest}; "
            f"first_differing_code_cell={differing_cell}; "
            f"published={expected_output[:diagnostic_limit]!r}; "
            f"executed={actual_output[:diagnostic_limit]!r}; "
            "review the numerical/API change and run scripts/execute_tutorials.py --write "
            "to publish intentionally updated outputs"
        )


def _validate_published_provenance(
    path: Path,
    notebook: NotebookNode,
    *,
    expected_version: str | None = None,
) -> None:
    spec = SPEC_BY_FILENAME[path.name]
    record = notebook.metadata.get("explainiverse_execution", {})
    if not isinstance(record, dict):
        raise ValueError(f"{path.name}: execution provenance must be a mapping")

    expected = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "status": "verified",
        "explainiverse_version": expected_version or version("explainiverse"),
        "poetry_lock_sha256": _lock_digest(),
        "package_source_sha256": _package_source_digest(),
        "notebook_source_sha256": _notebook_source_digest(notebook),
        "published_outputs_sha256": _published_outputs_digest(notebook),
        "runner": RUNNER_LABEL,
        "runner_sha256": _runner_digest(),
        "network_requirement": "none",
        "execution_network_guard": "python_socket_non_loopback",
        "security_sandbox": False,
        "deterministic_seed": spec.deterministic_seed,
    }
    mismatches = {
        key: (record.get(key), expected_value)
        for key, expected_value in expected.items()
        if record.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"{path.name}: stale or missing execution provenance: {mismatches}")

    _parse_utc_timestamp(record.get("executed_at_utc"), path=path)
    if not isinstance(record.get("python_version"), str) or not re.fullmatch(
        r"\d+\.\d+\.\d+(?:[^\s]*)?", record["python_version"]
    ):
        raise ValueError(f"{path.name}: python_version is missing or malformed")
    if not isinstance(record.get("platform"), str) or not record["platform"].strip():
        raise ValueError(f"{path.name}: platform is missing or malformed")
    _validate_published_outputs(path, notebook)


def _clear_execution(notebook: NotebookNode) -> None:
    notebook.metadata.pop("explainiverse_execution", None)
    notebook.metadata.pop("widgets", None)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        cell.execution_count = None
        cell.outputs = []
        cell.metadata.pop("execution", None)


def _normalise_execution(notebook: NotebookNode) -> None:
    execution_count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        cell.metadata.pop("execution", None)
        if cell.source.strip():
            execution_count += 1
            cell.execution_count = execution_count
        else:
            cell.execution_count = None


@contextmanager
def _deterministic_kernel_environment() -> Iterator[None]:
    values = {
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "HTTP_PROXY": "http://127.0.0.1:9",
        "HTTPS_PROXY": "http://127.0.0.1:9",
        "ALL_PROXY": "http://127.0.0.1:9",
        "NO_PROXY": "localhost,127.0.0.1,::1",
        "PIP_NO_INDEX": "1",
    }
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, old_value in previous.items():
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value


def _execute(notebook: NotebookNode, timeout: int) -> NotebookNode:
    _clear_execution(notebook)
    guard_cell = nbformat.v4.new_code_cell(
        OFFLINE_BOOTSTRAP,
        metadata={"tags": ["explainiverse-internal", "remove-cell"]},
    )
    notebook.cells.insert(0, guard_cell)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(ROOT)}},
    )
    with _deterministic_kernel_environment():
        executed = client.execute()
    executed.cells.pop(0)
    _normalise_execution(executed)
    return executed


def _publish_record(
    path: Path,
    notebook: NotebookNode,
    executed_at: str,
    *,
    expected_version: str,
) -> None:
    spec = SPEC_BY_FILENAME[path.name]
    _parse_utc_timestamp(executed_at, path=path)
    notebook.metadata["explainiverse_execution"] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "status": "verified",
        "executed_at_utc": executed_at,
        "explainiverse_version": expected_version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "poetry_lock_sha256": _lock_digest(),
        "package_source_sha256": _package_source_digest(),
        "notebook_source_sha256": _notebook_source_digest(notebook),
        "published_outputs_sha256": _published_outputs_digest(notebook),
        "runner": RUNNER_LABEL,
        "runner_sha256": _runner_digest(),
        "network_requirement": "none",
        "execution_network_guard": "python_socket_non_loopback",
        "security_sandbox": False,
        "deterministic_seed": spec.deterministic_seed,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebooks", nargs="*", help="Reviewed tutorials (defaults to the complete manifest)"
    )
    parser.add_argument("--write", action="store_true", help="Publish fresh outputs and provenance")
    parser.add_argument(
        "--source-only",
        action="store_true",
        help=(
            "Verify the checkout while requiring installed Explainiverse distribution metadata "
            "to be absent; this mode cannot publish notebooks"
        ),
    )
    parser.add_argument("--timeout", type=int, default=300, help="Per-cell timeout in seconds")
    parser.add_argument(
        "--executed-at",
        help="UTC ISO-8601 timestamp used with --write (defaults to the current time)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if args.executed_at and not args.write:
        raise ValueError("--executed-at requires --write")
    if args.source_only and args.write:
        raise ValueError("--source-only cannot be combined with --write")

    expected_version = _assert_checkout_import(source_only=args.source_only)
    paths = _resolve_notebooks(args.notebooks)
    executed_at = args.executed_at or datetime.now(timezone.utc).isoformat(timespec="seconds")

    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]

    executions: list[tuple[Path, NotebookNode]] = []
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        _validate_source(path, notebook)
        if not args.write:
            _validate_published_provenance(path, notebook, expected_version=expected_version)

        executed = _execute(copy.deepcopy(notebook), args.timeout)
        executions.append((path, executed))
        if not args.write:
            _validate_reexecuted_outputs(path, notebook, executed)
            print(f"verified {path.relative_to(ROOT)}")

    # Execute the entire selected set successfully before changing any notebook.
    if args.write:
        for path, executed in executions:
            _publish_record(path, executed, executed_at, expected_version=expected_version)
            _validate_published_provenance(path, executed, expected_version=expected_version)
            nbformat.write(executed, path)
            print(f"published {path.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
