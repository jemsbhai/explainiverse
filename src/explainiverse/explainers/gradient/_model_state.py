"""Model-state isolation shared by gradient-based explainers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


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

    modules = list(module.modules())
    training_states = [bool(child.training) for child in modules]
    buffers = list(module.buffers())
    saved_buffers = [buffer.detach().clone() for buffer in buffers]
    parameters = list(module.parameters()) if preserve_gradients else []
    original_gradients = [parameter.grad for parameter in parameters]
    saved_gradients = [
        None if gradient is None else gradient.detach().clone() for gradient in original_gradients
    ]

    module.eval()
    try:
        yield
    finally:
        with torch.no_grad():
            for buffer, saved in zip(buffers, saved_buffers):
                buffer.copy_(saved)
            for parameter, original, saved in zip(parameters, original_gradients, saved_gradients):
                if original is None:
                    parameter.grad = None
                else:
                    original.detach().copy_(saved)
                    parameter.grad = original
        # Direct assignment preserves intentionally mixed parent/child modes.
        for child, training in zip(modules, training_states):
            child.training = training
