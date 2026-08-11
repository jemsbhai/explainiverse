"""Non-overridable traversal of PyTorch's registered module/tensor graph."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, List

_EXACT_REGISTRY_TYPES = (dict, OrderedDict)


def _exact_registry(module: Any, registry_name: str) -> Any:
    """Return one ordinary PyTorch registry without invoking user mappings.

    PyTorch 2.0 initializes these registries as ``OrderedDict`` while newer
    releases use ``dict``.  Subclasses of either can override iteration and
    expose different registered state on successive reads, so accepting a
    generic Mapping would make snapshots and fail-closed device moves
    unprovable.
    """

    state = object.__getattribute__(module, "__dict__")
    registry = state.get(registry_name)
    if type(registry) not in _EXACT_REGISTRY_TYPES:
        raise RuntimeError(
            f"nn.Module.{registry_name} must be an exact built-in dict or "
            "collections.OrderedDict; custom or missing registries are unsupported"
        )
    return registry


def _require_standard_state_dispatch(module: Any) -> None:
    """Reject Python dispatch that can redefine registered-state semantics."""

    import torch.nn as nn

    for method_name in ("__getattribute__", "__setattr__", "__delattr__"):
        current = getattr(type(module), method_name)
        canonical = getattr(nn.Module, method_name)
        if current is not canonical:
            raise RuntimeError(
                f"{type(module).__name__}.{method_name} overrides nn.Module registered-state "
                "dispatch; custom state access or mutation semantics are unsupported"
            )


def registered_module_graph(root: Any) -> List[Any]:
    """Return registered modules in DFS order without public traversal hooks."""

    import torch.nn as nn

    if not isinstance(root, nn.Module):
        raise TypeError("root must be an nn.Module")
    ordered: List[Any] = []
    pending = [(root, False)]
    seen: set[int] = set()
    active: set[int] = set()
    while pending:
        module, exiting = pending.pop()
        identity = id(module)
        if exiting:
            active.remove(identity)
            continue
        if identity in active:
            raise RuntimeError("nn.Module._modules contains a registration cycle")
        if identity in seen:
            continue
        active.add(identity)
        seen.add(identity)
        _require_standard_state_dispatch(module)
        # Validate all registered-state containers before reading any of them.
        # This keeps every consumer fail closed even when it only needs the
        # module topology: a custom tensor registry can otherwise lie during a
        # later snapshot while model execution still observes the real tensor.
        children = _exact_registry(module, "_modules")
        _exact_registry(module, "_parameters")
        _exact_registry(module, "_buffers")
        ordered.append(module)
        pending.append((module, True))
        for child in reversed(tuple(children.values())):
            if child is None:
                continue
            if not isinstance(child, nn.Module):
                raise RuntimeError("nn.Module._modules contains a non-module child")
            pending.append((child, False))
    return ordered


def _registered_tensors(root: Any, registry_name: str) -> List[Any]:
    tensors: List[Any] = []
    seen: set[int] = set()
    for module in registered_module_graph(root):
        registry = _exact_registry(module, registry_name)
        for tensor in registry.values():
            if tensor is None or id(tensor) in seen:
                continue
            seen.add(id(tensor))
            tensors.append(tensor)
    return tensors


def registered_parameters(root: Any) -> List[Any]:
    """Return identity-deduplicated parameters from the internal graph."""

    return _registered_tensors(root, "_parameters")


def registered_buffers(root: Any) -> List[Any]:
    """Return identity-deduplicated buffers from the internal graph."""

    return _registered_tensors(root, "_buffers")


def registered_tensor_devices(root: Any) -> List[Any]:
    """Return devices for every internally registered parameter and buffer."""

    return [tensor.device for tensor in (*registered_parameters(root), *registered_buffers(root))]
