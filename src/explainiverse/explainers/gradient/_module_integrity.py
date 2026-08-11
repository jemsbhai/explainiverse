"""Fail-closed integrity checks for module-rule attribution backends."""

from __future__ import annotations

import inspect
from collections import OrderedDict
from types import MappingProxyType, MethodType
from typing import Any, Iterable, Mapping

_EXACT_REGISTRY_TYPES = (dict, OrderedDict)
_EXECUTION_HOOK_REGISTRIES = (
    "_forward_pre_hooks",
    "_forward_hooks",
    "_backward_pre_hooks",
    "_backward_hooks",
)
_STATE_IO_HOOK_REGISTRIES = (
    "_state_dict_pre_hooks",
    "_state_dict_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
)
_GLOBAL_EXECUTION_HOOK_REGISTRIES = (
    "_global_forward_pre_hooks",
    "_global_forward_hooks",
    "_global_backward_pre_hooks",
    "_global_backward_hooks",
)


def capture_canonical_forwards(module_types: Iterable[type]) -> Mapping[type, Any]:
    """Capture immutable standard-forward identities at module import time."""

    return MappingProxyType(
        {
            module_type: inspect.getattr_static(module_type, "forward")
            for module_type in module_types
        }
    )


def _require_exact_empty_registry(owner: Any, name: str, *, context: str) -> None:
    state = object.__getattribute__(owner, "__dict__")
    registry = state.get(name)
    if type(registry) not in _EXACT_REGISTRY_TYPES:
        raise RuntimeError(
            f"{context} requires {name} to be an exact built-in dict or "
            "collections.OrderedDict; custom or missing hook registries are unsupported"
        )
    if registry:
        raise RuntimeError(
            f"{context} does not permit pre-existing {name} hooks because they can "
            "change attribution execution or restoration"
        )


def require_no_global_execution_hooks(*, context: str) -> None:
    """Reject process-global module execution hooks at the checked boundary."""

    import torch.nn.modules.module as module_runtime

    for name in _GLOBAL_EXECUTION_HOOK_REGISTRIES:
        registry = getattr(module_runtime, name, None)
        if type(registry) not in _EXACT_REGISTRY_TYPES:
            raise RuntimeError(
                f"{context} cannot verify PyTorch's global {name} registry on this version"
            )
        if registry:
            raise RuntimeError(
                f"{context} does not permit process-global {name} hooks because they "
                "can change attribution execution"
            )


def require_module_integrity(
    module: Any,
    *,
    path: str,
    context: str,
    canonical_forward: Any | None,
    canonical_methods: Mapping[str, Any] | None = None,
    check_state_io: bool = False,
    canonical_state_methods: Mapping[str, Any] | None = None,
) -> Any:
    """Validate one module's callable identity and private hook registries.

    ``canonical_forward=None`` is reserved for a custom DeepLIFT composite:
    its current class-level implementation is accepted at construction, while
    an instance-level shadow is still rejected and the caller's topology token
    detects later class replacement.
    """

    location = path or "<root>"
    qualified = f"{context} module {location} ({type(module).__name__})"
    state = object.__getattribute__(module, "__dict__")
    if "forward" in state:
        raise RuntimeError(
            f"{qualified} has an instance-shadowed forward; only the unshadowed "
            "class implementation is supported"
        )

    current_forward = inspect.getattr_static(type(module), "forward")
    expected_forward = current_forward if canonical_forward is None else canonical_forward
    if current_forward is not expected_forward:
        raise RuntimeError(f"{qualified} no longer has its canonical forward implementation")
    bound_forward = object.__getattribute__(module, "forward")
    if (
        type(bound_forward) is not MethodType
        or bound_forward.__self__ is not module
        or bound_forward.__func__ is not expected_forward
    ):
        raise RuntimeError(f"{qualified} forward binding is not canonical")

    if state.get("_compiled_call_impl") is not None:
        raise RuntimeError(
            f"{qualified} uses a compiled call implementation outside the verified "
            "module-rule contract"
        )

    for method_name, expected_method in (canonical_methods or {}).items():
        if method_name in state:
            raise RuntimeError(
                f"{qualified} has an instance-shadowed {method_name}; canonical "
                "Captum traversal and hook registration cannot be proven"
            )
        current_method = inspect.getattr_static(type(module), method_name)
        bound_method = object.__getattribute__(module, method_name)
        if current_method is not expected_method or (
            type(bound_method) is not MethodType
            or bound_method.__self__ is not module
            or bound_method.__func__ is not expected_method
        ):
            raise RuntimeError(f"{qualified} does not use canonical nn.Module.{method_name}")

    for registry_name in _EXECUTION_HOOK_REGISTRIES:
        _require_exact_empty_registry(module, registry_name, context=qualified)

    if check_state_io:
        for registry_name in _STATE_IO_HOOK_REGISTRIES:
            _require_exact_empty_registry(module, registry_name, context=qualified)
        if canonical_state_methods is None:
            raise RuntimeError("canonical state methods are required for state-I/O validation")
        for method_name, expected_method in canonical_state_methods.items():
            if method_name in state:
                raise RuntimeError(
                    f"{qualified} has an instance-shadowed {method_name}; Captum LRP "
                    "cannot prove state restoration"
                )
            current_method = inspect.getattr_static(type(module), method_name)
            bound_method = object.__getattribute__(module, method_name)
            if current_method is not expected_method or (
                type(bound_method) is not MethodType
                or bound_method.__self__ is not module
                or bound_method.__func__ is not expected_method
            ):
                raise RuntimeError(
                    f"{qualified} does not use canonical nn.Module.{method_name}; "
                    "Captum LRP cannot prove state restoration"
                )

    return current_forward
