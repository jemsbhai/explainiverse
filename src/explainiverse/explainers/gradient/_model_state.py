"""Model-state isolation shared by gradient-based explainers."""

from __future__ import annotations

from contextlib import contextmanager
from threading import Lock, RLock
from typing import Iterator
from weakref import WeakKeyDictionary

_LOCK_REGISTRY_GUARD = Lock()
_MODEL_STATE_LOCKS: "WeakKeyDictionary[object, RLock]" = WeakKeyDictionary()
_FALLBACK_MODEL_STATE_LOCK = RLock()
_TORCH_DEFAULT_RNG_LOCK = RLock()


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


@contextmanager
def preserve_adapter_model_eval(adapter, *, preserve_gradients: bool = True) -> Iterator[None]:
    """Evaluate a wrapped torch module deterministically and restore its state.

    PyTorch adapters enter evaluation mode when they are constructed, but callers
    can subsequently put the wrapped module back into training mode. Attribution
    must not then sample Dropout or update BatchNorm buffers. This context also
    restores mixed per-module training flags and existing parameter-gradient
    objects exactly.
    """

    module = getattr(adapter, "model", None)
    if module is None or not callable(getattr(module, "modules", None)):
        yield
        return

    import torch

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
        modules = list(module.modules())
        training_states = [bool(child.training) for child in modules]
        buffers = list(module.buffers())
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
        parameters = list(module.parameters()) if preserve_gradients else []
        original_gradients = [parameter.grad for parameter in parameters]
        saved_gradients = [
            None if gradient is None else gradient.detach().clone()
            for gradient in original_gradients
        ]
        cpu_rng_state = torch.random.get_rng_state().clone()
        cuda_rng_states = None
        if torch.cuda.is_available():
            # Capture all default CUDA generators even when CUDA was not yet
            # initialized. This prevents an attribution forward that lazily
            # initializes CUDA from leaking its first RNG consumption.
            cuda_rng_states = [state.clone() for state in torch.cuda.get_rng_state_all()]

        try:
            # The transition itself can execute overridden ``train(False)``
            # code, so it belongs inside the restoration boundary too.
            module.eval()
            yield
        finally:
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
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)
            # Direct assignment preserves intentionally mixed parent/child modes.
            for child, training in zip(modules, training_states):
                child.training = training
