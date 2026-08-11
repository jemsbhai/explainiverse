"""Model-state isolation shared by gradient-based explainers."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from contextlib import contextmanager
from threading import Lock, RLock
from types import MethodType
from typing import Any, Iterator
from weakref import WeakKeyDictionary

import numpy as np

from explainiverse._torch_module_graph import (
    registered_buffers,
    registered_module_graph,
    registered_parameters,
)

_LOCK_REGISTRY_GUARD = Lock()
_MODEL_STATE_LOCKS: "WeakKeyDictionary[object, RLock]" = WeakKeyDictionary()
_FALLBACK_MODEL_STATE_LOCK = RLock()
_TORCH_DEFAULT_RNG_LOCK = RLock()


class ModelStateIsolationError(RuntimeError):
    """A declared model-owned state contract could not be preserved."""


def _capture_declared_fingerprint(callback, module) -> dict[str, Any] | None:
    """Capture one detached, name-addressable opt-in state fingerprint."""

    if callback is None:
        return None
    fingerprint = callback(module)
    if not isinstance(fingerprint, Mapping):
        raise TypeError("model_state_fingerprint must return a mapping")
    normalized: dict[str, Any] = {}
    for name, value in fingerprint.items():
        if not isinstance(name, str) or not name.strip():
            raise TypeError("model_state_fingerprint keys must be non-empty strings")
        normalized[name] = copy.deepcopy(value)
    return normalized


def _fingerprint_values_equal(left: Any, right: Any) -> bool:
    """Compare common fingerprint payloads without truth-value ambiguity."""

    try:
        import torch

        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            return bool(
                isinstance(left, torch.Tensor)
                and isinstance(right, torch.Tensor)
                and left.dtype == right.dtype
                and left.device == right.device
                and left.layout == right.layout
                and tuple(left.shape) == tuple(right.shape)
                and torch.equal(left, right)
            )
    except ImportError:  # pragma: no cover - this module is used by Torch paths
        pass
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return bool(
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and np.array_equal(left, right, equal_nan=True)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if set(left) != set(right):
            return False
        return all(_fingerprint_values_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            return False
        return all(_fingerprint_values_equal(a, b) for a, b in zip(left, right))
    try:
        result = left == right
    except Exception:
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _changed_fingerprint_names(
    before: dict[str, Any] | None, after: dict[str, Any] | None
) -> list[str]:
    if before is None and after is None:
        return []
    if before is None or after is None:
        return ["<fingerprint availability>"]
    names = sorted(set(before) | set(after))
    return [
        name
        for name in names
        if name not in before
        or name not in after
        or not _fingerprint_values_equal(before[name], after[name])
    ]


def adapter_model_lock(module) -> RLock:
    """Return the stable re-entrant lock associated with one wrapped module."""

    try:
        with _LOCK_REGISTRY_GUARD:
            lock = _MODEL_STATE_LOCKS.get(module)
            if lock is None:
                lock = RLock()
                _MODEL_STATE_LOCKS[module] = lock
            return lock
    except TypeError:
        # Custom unhashable / non-weak-referenceable module-like objects are
        # rare but can still receive the conservative serialized behavior.
        return _FALLBACK_MODEL_STATE_LOCK


@contextmanager
def adapter_model_operation_lock(module) -> Iterator[None]:
    """Serialize one operation in the shared Torch-RNG/model-state lock order."""

    with _TORCH_DEFAULT_RNG_LOCK, adapter_model_lock(module):
        yield


def _require_standard_mode_dispatch(modules) -> None:
    """Reject virtual train/eval transitions with arbitrary side effects."""

    import torch.nn as nn

    for child in modules:
        for method_name, canonical in (("train", nn.Module.train), ("eval", nn.Module.eval)):
            state = object.__getattribute__(child, "__dict__")
            if method_name in state:
                raise RuntimeError(
                    f"{type(child).__name__} has an instance-shadowed {method_name}; "
                    "model-state isolation requires canonical nn.Module mode dispatch"
                )
            bound = object.__getattribute__(child, method_name)
            if (
                type(bound) is not MethodType
                or bound.__self__ is not child
                or bound.__func__ is not canonical
            ):
                raise RuntimeError(
                    f"{type(child).__name__}.{method_name} overrides canonical nn.Module "
                    "mode dispatch; model-state isolation cannot prove its side effects"
                )


@contextmanager
def preserve_adapter_model_eval(adapter, *, preserve_gradients: bool = True) -> Iterator[None]:
    """Evaluate a wrapped torch module deterministically and restore its state.

    PyTorch adapters enter evaluation mode when they are constructed, but callers
    can subsequently put the wrapped module back into training mode. Attribution
    must not then sample Dropout or update BatchNorm buffers. This context also
    restores mixed per-module training flags and existing parameter-gradient
    objects exactly.
    """

    import torch

    module = getattr(adapter, "model", None)
    if module is None or not isinstance(module, torch.nn.Module):
        yield
        return

    # Model modes, buffers, and Torch's default RNGs are mutable shared state.
    # Serialize Explainiverse attribution contexts so concurrent explainers do
    # not restore stale snapshots over one another. The lock is re-entrant for
    # explainers that nest state-preservation contexts.
    # The per-model lock protects modes/buffers/gradients. The additional RNG
    # lock is process-wide because Torch's default CPU/CUDA generators are
    # process-global even when two independent models are being explained.
    # Acquire the process-wide lock first so a nested explanation of a second
    # model cannot deadlock against another thread already waiting on RNG state.
    with adapter_model_operation_lock(module):
        isolation_complete = False

        def poison_adapter(reason: str) -> None:
            if hasattr(adapter, "_poisoned_reason"):
                adapter._poisoned_reason = reason

        protocol = getattr(adapter, "model_state_protocol", None)
        fingerprint_callback = getattr(adapter, "model_state_fingerprint", None)
        model_generators = tuple(getattr(adapter, "model_generators", ()))
        generator_states = [generator.get_state().clone() for generator in model_generators]
        cpu_rng_state = torch.random.get_rng_state().clone()
        cuda_rng_states = None
        if torch.cuda.is_available():
            # Capture all default CUDA generators even when CUDA was not yet
            # initialized. This prevents an attribution forward that lazily
            # initializes CUDA from leaking its first RNG consumption.
            cuda_rng_states = [state.clone() for state in torch.cuda.get_rng_state_all()]

        def restore_rng_states() -> None:
            try:
                torch.random.set_rng_state(cpu_rng_state)
                if cuda_rng_states is not None:
                    torch.cuda.set_rng_state_all(cuda_rng_states)
                for generator, state in zip(model_generators, generator_states):
                    generator.set_state(state)
            except Exception as error:  # pragma: no cover - device-specific generator failure
                poison_adapter(f"Torch generator restoration raised {type(error).__name__}")
                raise ModelStateIsolationError(
                    "Torch generator state restoration failed; reconstruct the model and "
                    "adapter before further use."
                ) from error

        try:
            # Validate the internal graph and mode dispatch before callbacks or
            # model work. Mode changes below use direct flags, so no virtual
            # ``train`` / ``eval`` / ``children`` implementation can mutate
            # model state while entering the isolation boundary.
            modules = registered_module_graph(module)
            _require_standard_mode_dispatch(modules)
            # RNG ownership begins before user callbacks: snapshot and
            # fingerprint implementations are allowed to consume both default
            # and explicitly injected generators without leaking state.
            declared_snapshot = protocol.snapshot(module) if protocol is not None else None
            restore_rng_states()
            fingerprint_before = _capture_declared_fingerprint(fingerprint_callback, module)
            restore_rng_states()
            training_states = [bool(child.training) for child in modules]
            buffers = registered_buffers(module)
            saved_buffers = []
            for buffer in buffers:
                is_strided = buffer.layout == torch.strided
                storage = buffer.untyped_storage() if is_strided else None
                saved_buffers.append(
                    (
                        buffer.detach().clone(),
                        storage,
                        int(buffer.storage_offset()) if is_strided else 0,
                        tuple(buffer.size()),
                        tuple(buffer.stride()) if is_strided else (),
                    )
                )
            parameters = registered_parameters(module) if preserve_gradients else []
            original_gradients = [parameter.grad for parameter in parameters]
            saved_gradients = [
                None if gradient is None else gradient.detach().clone()
                for gradient in original_gradients
            ]

            try:
                for child in modules:
                    child.training = False
                yield
            finally:
                restoration_errors: list[str] = []
                try:
                    with torch.no_grad():
                        for buffer, snapshot in zip(buffers, saved_buffers):
                            saved, storage, storage_offset, size, stride = snapshot
                            if storage is not None:
                                try:
                                    buffer.set_(storage, storage_offset, size, stride)
                                except (RuntimeError, TypeError):
                                    buffer.resize_as_(saved)
                            elif tuple(buffer.size()) != size:
                                buffer.resize_as_(saved)
                            buffer.copy_(saved)
                        for parameter, original, saved in zip(
                            parameters, original_gradients, saved_gradients
                        ):
                            if original is None:
                                parameter.grad = None
                            else:
                                original.detach().copy_(saved)
                                parameter.grad = original
                except Exception as error:  # pragma: no cover - exotic custom tensors
                    restoration_errors.append(f"registered torch state: {error}")
                # Direct assignment preserves intentionally mixed parent/child modes.
                for child, training in zip(modules, training_states):
                    child.training = training
                if protocol is not None:
                    try:
                        protocol.restore(module, declared_snapshot)
                    except Exception as error:
                        restoration_errors.append(f"declared model_state_protocol: {error}")

                # Make the after-fingerprint observe the same owned RNG state
                # as the before-fingerprint, regardless of model/protocol use.
                restore_rng_states()
                fingerprint_after = _capture_declared_fingerprint(fingerprint_callback, module)
                changed_names = _changed_fingerprint_names(fingerprint_before, fingerprint_after)
                if changed_names:
                    changed = ", ".join(changed_names)
                    raise ModelStateIsolationError(
                        "Declared model-owned state changed during explanation: "
                        f"{changed}. Provide a model_state_protocol that restores these "
                        "components or remove the unsupported mutation."
                    )
                if restoration_errors:
                    details = "; ".join(restoration_errors)
                    raise ModelStateIsolationError(
                        "Model state restoration failed; reconstruct the model and adapter "
                        f"before further use ({details})."
                    )
                isolation_complete = True
        except BaseException as error:
            if not isolation_complete:
                poison_adapter(
                    "model-state isolation did not complete after " f"{type(error).__name__}"
                )
            raise
        finally:
            # Restore last so protocol/fingerprint callbacks cannot leak RNG,
            # including when a callback or the explained model raises.
            restore_rng_states()
