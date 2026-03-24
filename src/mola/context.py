"""Adapter context threading.

The trickiest problem in multi-LoRA: how to tell each layer which adapter to use
during a forward pass, WITHOUT modifying any mlx-lm model code.

Solution: contextvars. Same pattern as Flask's request context or SQLAlchemy's session.
The MultiLoRALinear layer reads the current adapter from the context variable.
The server sets it before each forward pass.

    with adapter_context("solana"):
        output = model(x)  # all MultiLoRALinear layers use the "solana" adapter
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager

_current_adapter: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "mola_adapter", default=None
)


@contextmanager
def adapter_context(adapter_id: str | None):
    """Set the active adapter for the duration of a forward pass."""
    token = _current_adapter.set(adapter_id)
    try:
        yield
    finally:
        _current_adapter.reset(token)


def get_current_adapter() -> str | None:
    """Get the currently active adapter ID. Called by MultiLoRALinear."""
    return _current_adapter.get()
