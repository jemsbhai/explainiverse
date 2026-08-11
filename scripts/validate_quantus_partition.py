"""Audit the exact pytest partition owned by the Quantus reference lane."""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

_DYNAMIC_IMPORTERS = {
    "builtins.__import__",
    "importlib.import_module",
    "pytest.importorskip",
}


def _is_quantus_marker(decorator: ast.expr, aliases: Mapping[str, str]) -> bool:
    if isinstance(decorator, ast.Call):
        decorator = decorator.func
    return _canonical_expr(decorator, aliases) == "pytest.mark.quantus_reference"


def _canonical_expr(node: ast.AST, aliases: Mapping[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        builtin = {
            "__import__": "builtins.__import__",
            "getattr": "builtins.getattr",
        }.get(node.id)
        if builtin is not None:
            return builtin
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        value = _canonical_expr(node.value, aliases)
        if value is not None:
            return f"{value}.{node.attr}"
    if (
        isinstance(node, ast.Call)
        and _canonical_expr(node.func, aliases) == "builtins.getattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    ):
        value = _canonical_expr(node.args[0], aliases)
        if value is not None:
            return f"{value}.{node.args[1].value}"
    return None


def _import_aliases(nodes: Iterable[ast.AST]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in nodes:
        if isinstance(node, ast.Import):
            for value in node.names:
                bound = value.asname or value.name.split(".", 1)[0]
                canonical = value.name if value.asname else value.name.split(".", 1)[0]
                aliases[bound] = canonical
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            for value in node.names:
                if value.name == "*":
                    continue
                aliases[value.asname or value.name] = f"{node.module}.{value.name}"
    return aliases


def _ordered_scope_nodes(nodes: Sequence[ast.AST]) -> list[ast.AST]:
    return sorted(
        (node for node in nodes if hasattr(node, "lineno")),
        key=lambda node: (getattr(node, "lineno", 0), getattr(node, "col_offset", 0)),
    )


def _update_scope_aliases(node: ast.AST, aliases: dict[str, str]) -> None:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        aliases.update(_import_aliases((node,)))
        return
    if isinstance(node, ast.Assign):
        targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        targets = [node.target.id]
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        aliases.pop(node.name, None)
        return
    elif isinstance(node, ast.Delete):
        for target in node.targets:
            if isinstance(target, ast.Name):
                aliases.pop(target.id, None)
        return
    else:
        return

    value = node.value
    canonical = _canonical_expr(value, aliases) if value is not None else None
    for name in targets:
        if canonical is None:
            aliases.pop(name, None)
        else:
            aliases[name] = canonical


def _update_string_aliases(node: ast.AST, values: dict[str, str]) -> None:
    if isinstance(node, ast.Assign):
        targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        targets = [node.target.id]
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        values.pop(node.name, None)
        return
    elif isinstance(node, ast.Delete):
        for target in node.targets:
            if isinstance(target, ast.Name):
                values.pop(target.id, None)
        return
    else:
        return

    value = node.value
    resolved: str | None = None
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        resolved = value.value
    elif isinstance(value, ast.Name):
        resolved = values.get(value.id)
    for name in targets:
        if resolved is None:
            values.pop(name, None)
        else:
            values[name] = resolved


def _scope_aliases(
    nodes: Sequence[ast.AST],
    base: Mapping[str, str] | None = None,
    *,
    shadowed: Iterable[str] = (),
) -> dict[str, str]:
    aliases = dict(base or {})
    for name in shadowed:
        aliases.pop(name, None)

    for node in _ordered_scope_nodes(nodes):
        _update_scope_aliases(node, aliases)
    return aliases


def _string_aliases(
    nodes: Sequence[ast.AST],
    base: Mapping[str, str] | None = None,
    *,
    shadowed: Iterable[str] = (),
) -> dict[str, str]:
    values = dict(base or {})
    for name in shadowed:
        values.pop(name, None)

    for node in _ordered_scope_nodes(nodes):
        _update_string_aliases(node, values)
    return values


def _scope_states(
    nodes: Sequence[ast.AST],
    base_aliases: Mapping[str, str],
    base_strings: Mapping[str, str],
    *,
    shadowed: Iterable[str] = (),
) -> dict[int, tuple[dict[str, str], dict[str, str]]]:
    aliases = dict(base_aliases)
    strings = dict(base_strings)
    for name in shadowed:
        aliases.pop(name, None)
        strings.pop(name, None)
    states: dict[int, tuple[dict[str, str], dict[str, str]]] = {}
    for node in _ordered_scope_nodes(nodes):
        states[id(node)] = (dict(aliases), dict(strings))
        _update_scope_aliases(node, aliases)
        _update_string_aliases(node, strings)
    return states


def _is_quantus_module(value: object) -> bool:
    return isinstance(value, str) and (value == "quantus" or value.startswith("quantus."))


def _literal_or_alias(node: ast.AST, string_aliases: Mapping[str, str]) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return string_aliases.get(node.id)
    return None


def _call_module_name(call: ast.Call, string_aliases: Mapping[str, str]) -> object:
    if call.args:
        return _literal_or_alias(call.args[0], string_aliases)
    for keyword in call.keywords:
        if keyword.arg in {"name", "modname"}:
            return _literal_or_alias(keyword.value, string_aliases)
    return None


def _node_loads_quantus(
    node: ast.AST, aliases: Mapping[str, str], string_aliases: Mapping[str, str]
) -> bool:
    if isinstance(node, ast.Import):
        return any(_is_quantus_module(value.name) for value in node.names)
    if isinstance(node, ast.ImportFrom):
        return _is_quantus_module(node.module)
    if isinstance(node, ast.Call):
        return _canonical_expr(node.func, aliases) in _DYNAMIC_IMPORTERS and _is_quantus_module(
            _call_module_name(node, string_aliases)
        )
    return False


def _walk_without_function_bodies(node: ast.AST) -> Iterable[ast.AST]:
    """Yield module/class execution nodes without descending into function bodies."""
    yield node
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                yield from _walk_without_function_bodies(decorator)
            for default in (*node.args.defaults, *node.args.kw_defaults):
                if default is not None:
                    yield from _walk_without_function_bodies(default)
        return
    for child in ast.iter_child_nodes(node):
        yield from _walk_without_function_bodies(child)


def _function_nodes(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.AST]:
    nodes: list[ast.AST] = []
    for statement in node.body:
        nodes.extend(_walk_without_function_bodies(statement))
    return nodes


def _contains_quantus_import(node: ast.AST) -> bool:
    """Return whether *node* directly imports or dynamically loads Quantus."""
    nodes = list(ast.walk(node))
    states = _scope_states(nodes, {}, {})
    return any(
        _node_loads_quantus(child, *states[id(child)]) for child in nodes if id(child) in states
    )


@dataclass(frozen=True)
class _CallSite:
    canonical: str
    positional: tuple[object, ...]
    keywords: tuple[tuple[str | None, object], ...]


def _call_sites(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    states: Mapping[int, tuple[Mapping[str, str], Mapping[str, str]]],
) -> tuple[_CallSite, ...]:
    calls: list[_CallSite] = []
    for child in _function_nodes(node):
        if not isinstance(child, ast.Call):
            continue
        aliases, string_aliases = states[id(child)]
        canonical = _canonical_expr(child.func, aliases)
        if canonical is not None:
            calls.append(
                _CallSite(
                    canonical=canonical,
                    positional=tuple(
                        _literal_or_alias(value, string_aliases) for value in child.args
                    ),
                    keywords=tuple(
                        (keyword.arg, _literal_or_alias(keyword.value, string_aliases))
                        for keyword in child.keywords
                    ),
                )
            )
    return tuple(calls)


def _called_nested_function_loads_quantus(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    calls: Sequence[_CallSite],
) -> bool:
    nested = {
        statement.name: statement
        for statement in node.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    called = {call.canonical for call in calls}
    return any(
        name in called and _contains_quantus_import(function) for name, function in nested.items()
    )


def _usefixture_names(decorators: Iterable[ast.expr], aliases: Mapping[str, str]) -> set[str]:
    names: set[str] = set()
    for decorator in decorators:
        if not isinstance(decorator, ast.Call):
            continue
        if _canonical_expr(decorator.func, aliases) != "pytest.mark.usefixtures":
            continue
        names.update(
            value.value
            for value in decorator.args
            if isinstance(value, ast.Constant) and isinstance(value.value, str)
        )
    return names


def _fixture_names(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    aliases: Mapping[str, str],
    inherited: Iterable[str] = (),
) -> set[str]:
    arguments = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
    argument_names = {value.arg for value in arguments if value.arg not in {"self", "cls"}}
    names = set(inherited) | argument_names | _usefixture_names(node.decorator_list, aliases)
    for child in _function_nodes(node):
        if not isinstance(child, ast.Call) or not isinstance(child.func, ast.Attribute):
            continue
        if (
            child.func.attr == "getfixturevalue"
            and isinstance(child.func.value, ast.Name)
            and child.func.value.id in argument_names
            and child.args
            and isinstance(child.args[0], ast.Constant)
            and isinstance(child.args[0].value, str)
        ):
            names.add(child.args[0].value)
    return names


def _module_pytest_marks(statements: Sequence[ast.stmt]) -> list[ast.expr]:
    markers: list[ast.expr] = []
    for statement in statements:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        if isinstance(statement, ast.Assign):
            is_pytestmark = any(
                isinstance(target, ast.Name) and target.id == "pytestmark"
                for target in statement.targets
            )
        else:
            is_pytestmark = (
                isinstance(statement.target, ast.Name) and statement.target.id == "pytestmark"
            )
        if not is_pytestmark or statement.value is None:
            continue
        value = statement.value
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            markers.extend(value.elts)
        else:
            markers.append(value)
    return markers


def _dynamic_loader_parameters(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    states: Mapping[int, tuple[Mapping[str, str], Mapping[str, str]]],
) -> frozenset[str]:
    parameters = {
        value.arg
        for value in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
        if value.arg not in {"self", "cls"}
    }
    dynamic: set[str] = set()
    for child in _function_nodes(node):
        if not isinstance(child, ast.Call):
            continue
        aliases, _ = states[id(child)]
        if _canonical_expr(child.func, aliases) not in _DYNAMIC_IMPORTERS:
            continue
        module_expr: ast.AST | None = child.args[0] if child.args else None
        if module_expr is None:
            for keyword in child.keywords:
                if keyword.arg in {"name", "modname"}:
                    module_expr = keyword.value
                    break
        if isinstance(module_expr, ast.Name) and module_expr.id in parameters:
            dynamic.add(module_expr.id)
    return frozenset(dynamic)


def _fixture_name(
    node: ast.FunctionDef | ast.AsyncFunctionDef, aliases: Mapping[str, str]
) -> str | None:
    for decorator in node.decorator_list:
        candidate = decorator.func if isinstance(decorator, ast.Call) else decorator
        if _canonical_expr(candidate, aliases) != "pytest.fixture":
            continue
        if isinstance(decorator, ast.Call):
            for keyword in decorator.keywords:
                if (
                    keyword.arg == "name"
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)
                ):
                    return keyword.value.value
        return node.name
    return None


def _is_autouse_fixture(
    node: ast.FunctionDef | ast.AsyncFunctionDef, aliases: Mapping[str, str]
) -> bool:
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        candidate = decorator.func
        if _canonical_expr(candidate, aliases) != "pytest.fixture":
            continue
        for keyword in decorator.keywords:
            if keyword.arg == "autouse" and isinstance(keyword.value, ast.Constant):
                return keyword.value.value is True
    return False


@dataclass(frozen=True)
class _Function:
    key: str
    relative: str
    qualname: str
    class_names: tuple[str, ...]
    node_id: str | None
    marked: bool
    direct_load: bool
    calls: tuple[_CallSite, ...]
    parameters: tuple[str, ...]
    dynamic_loader_parameters: frozenset[str]
    fixture_names: frozenset[str]
    fixture_name: str | None
    autouse_fixture: bool


def _iter_functions(
    statements: Sequence[ast.stmt],
    *,
    relative: str,
    module_aliases: Mapping[str, str],
    module_string_aliases: Mapping[str, str],
    collect_tests: bool,
    class_names: tuple[str, ...] = (),
    class_marked: bool = False,
    inherited_fixture_names: frozenset[str] = frozenset(),
) -> Iterable[_Function]:
    for statement in statements:
        if isinstance(statement, ast.ClassDef):
            marked = class_marked or any(
                _is_quantus_marker(value, module_aliases) for value in statement.decorator_list
            )
            fixture_names = inherited_fixture_names | _usefixture_names(
                statement.decorator_list, module_aliases
            )
            yield from _iter_functions(
                statement.body,
                relative=relative,
                module_aliases=module_aliases,
                module_string_aliases=module_string_aliases,
                collect_tests=collect_tests,
                class_names=(*class_names, statement.name),
                class_marked=marked,
                inherited_fixture_names=frozenset(fixture_names),
            )
            continue
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        qualname = ".".join((*class_names, statement.name))
        nodes = _function_nodes(statement)
        parameters = tuple(
            value.arg
            for value in (
                *statement.args.posonlyargs,
                *statement.args.args,
                *statement.args.kwonlyargs,
            )
            if value.arg not in {"self", "cls"}
        )
        states = _scope_states(
            nodes,
            module_aliases,
            module_string_aliases,
            shadowed=parameters,
        )
        calls = _call_sites(statement, states)
        direct_load = any(
            _node_loads_quantus(node, *states[id(node)]) for node in nodes if id(node) in states
        ) or _called_nested_function_loads_quantus(statement, calls)
        is_collected_class = not class_names or all(name.startswith("Test") for name in class_names)
        is_test = collect_tests and is_collected_class and statement.name.startswith("test_")
        node_id = "::".join((relative, *class_names, statement.name)) if is_test else None
        marked = class_marked or any(
            _is_quantus_marker(value, module_aliases) for value in statement.decorator_list
        )
        yield _Function(
            key=f"{relative}::{qualname}",
            relative=relative,
            qualname=qualname,
            class_names=class_names,
            node_id=node_id,
            marked=marked,
            direct_load=direct_load,
            calls=calls,
            parameters=parameters,
            dynamic_loader_parameters=_dynamic_loader_parameters(statement, states),
            fixture_names=frozenset(
                _fixture_names(statement, module_aliases, inherited_fixture_names)
            ),
            fixture_name=_fixture_name(statement, module_aliases),
            autouse_fixture=_is_autouse_fixture(statement, module_aliases),
        )


def _local_call_target(
    function: _Function, call: str, functions: Mapping[str, _Function]
) -> str | None:
    candidates: list[str] = []
    if call.startswith(("self.", "cls.")) and function.class_names:
        candidates.append(".".join((*function.class_names, call.split(".", 1)[1])))
    elif "." not in call:
        if function.class_names:
            candidates.append(".".join((*function.class_names, call)))
        candidates.append(call)
    else:
        candidates.append(call)
    for qualname in candidates:
        key = f"{function.relative}::{qualname}"
        if key in functions:
            return key
    return None


def _call_supplies_quantus(call: _CallSite, target: _Function) -> bool:
    keyword_values = dict(call.keywords)
    for index, parameter in enumerate(target.parameters):
        if parameter not in target.dynamic_loader_parameters:
            continue
        value = (
            call.positional[index]
            if index < len(call.positional)
            else keyword_values.get(parameter)
        )
        if _is_quantus_module(value):
            return True
    return False


def _fixture_target(
    function: _Function,
    fixture_name: str,
    fixture_keys: Mapping[str, set[str]],
    functions: Mapping[str, _Function],
) -> set[str]:
    requester = PurePosixPath(function.relative)
    requester_directory = requester.parent
    candidates: list[tuple[int, str]] = []
    for key in fixture_keys.get(fixture_name, ()):
        fixture = functions[key]
        fixture_path = PurePosixPath(fixture.relative)
        if fixture_path == requester:
            candidates.append((10_000, key))
            continue
        if fixture_path.name != "conftest.py":
            continue
        fixture_directory = fixture_path.parent
        if (
            fixture_directory == requester_directory
            or fixture_directory in requester_directory.parents
        ):
            candidates.append((len(fixture_directory.parts), key))
    if not candidates:
        return set()
    priority = max(value for value, _ in candidates)
    return {key for value, key in candidates if value == priority}


def _reachable(start: str, edges: Mapping[str, set[str]]) -> set[str]:
    seen: set[str] = set()
    pending = list(edges.get(start, ()))
    while pending:
        key = pending.pop()
        if key in seen:
            continue
        seen.add(key)
        pending.extend(edges.get(key, ()))
    return seen


def discover_partition(tests_root: Path) -> tuple[list[str], list[str]]:
    """Return marked node IDs and static import/marker contract violations."""
    violations: list[str] = []
    functions: dict[str, _Function] = {}

    for path in sorted(tests_root.rglob("*.py")):
        relative = path.relative_to(tests_root.parent).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except SyntaxError as exc:
            violations.append(f"{relative}: cannot parse: {exc}")
            continue

        module_nodes = list(_walk_without_function_bodies(tree))
        module_aliases = _scope_aliases(module_nodes)
        module_string_aliases = _string_aliases(module_nodes)
        module_states = _scope_states(module_nodes, {}, {})
        module_markers = _module_pytest_marks(tree.body)
        module_marked = any(_is_quantus_marker(value, module_aliases) for value in module_markers)
        module_fixture_names = frozenset(_usefixture_names(module_markers, module_aliases))
        if any(
            _node_loads_quantus(node, *module_states[id(node)])
            for node in module_nodes
            if id(node) in module_states
        ):
            violations.append(f"{relative}: Quantus must not be imported at module scope")

        for function in _iter_functions(
            tree.body,
            relative=relative,
            module_aliases=module_aliases,
            module_string_aliases=module_string_aliases,
            collect_tests=path.name.startswith("test_"),
            class_marked=module_marked,
            inherited_fixture_names=module_fixture_names,
        ):
            functions[function.key] = function

    fixture_keys: dict[str, set[str]] = {}
    for function in functions.values():
        if function.fixture_name is not None:
            fixture_keys.setdefault(function.fixture_name, set()).add(function.key)

    edges: dict[str, set[str]] = {key: set() for key in functions}
    conditional_loading: set[str] = set()
    for function in functions.values():
        for call in function.calls:
            target = _local_call_target(function, call.canonical, functions)
            if target is not None:
                edges[function.key].add(target)
                if _call_supplies_quantus(call, functions[target]):
                    conditional_loading.add(function.key)
        for fixture_name in function.fixture_names:
            edges[function.key].update(
                _fixture_target(function, fixture_name, fixture_keys, functions)
            )

    loading = {
        function.key for function in functions.values() if function.direct_load
    } | conditional_loading
    changed = True
    while changed:
        changed = False
        for key, targets in edges.items():
            if key not in loading and targets & loading:
                loading.add(key)
                changed = True

    marked: list[str] = []
    referenced_loading_helpers: set[str] = set()
    for function in functions.values():
        if function.node_id is None:
            continue
        if function.marked:
            marked.append(function.node_id)

        reachable = _reachable(function.key, edges)
        referenced_loading_helpers.update(reachable & loading)
        uses_quantus = function.key in loading
        if function.marked and not uses_quantus:
            violations.append(
                f"{function.node_id}: quantus_reference test has no direct or audited helper import"
            )
        elif not function.marked and uses_quantus:
            if function.direct_load:
                violations.append(
                    f"{function.node_id}: Quantus use is missing quantus_reference marker"
                )
            else:
                violations.append(
                    f"{function.node_id}: calls or requests a Quantus-loading helper/fixture "
                    "without quantus_reference marker"
                )

    for key in sorted(loading):
        function = functions[key]
        if function.node_id is not None:
            continue
        if function.autouse_fixture:
            violations.append(
                f"{function.relative}::{function.qualname}: Quantus-loading autouse fixtures "
                "cannot define an exact marker partition"
            )
        elif key not in referenced_loading_helpers:
            violations.append(
                f"{function.relative}::{function.qualname}: Quantus-loading helper/fixture is "
                "not referenced by any collected test"
            )

    return sorted(marked), sorted(set(violations))


def load_manifest(path: Path) -> list[str]:
    entries = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if entries != sorted(set(entries)):
        raise ValueError("Quantus reference manifest must be sorted and contain unique node IDs")
    for entry in entries:
        if not entry.startswith("tests/") or "::test_" not in entry:
            raise ValueError(f"invalid Quantus reference node ID {entry!r}")
    return entries


def validate_partition(tests_root: Path, manifest_path: Path) -> None:
    marked, violations = discover_partition(tests_root)
    manifest = load_manifest(manifest_path)
    if marked != manifest:
        missing = sorted(set(marked) - set(manifest))
        stale = sorted(set(manifest) - set(marked))
        if missing:
            violations.append(f"manifest is missing marked tests: {missing!r}")
        if stale:
            violations.append(f"manifest contains unmarked/stale tests: {stale!r}")
    if violations:
        raise ValueError("; ".join(violations))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-root", type=Path, default=Path("tests"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(".github/constraints/quantus-reference-tests.txt"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        validate_partition(args.tests_root, args.manifest)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
