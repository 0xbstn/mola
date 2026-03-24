"""Adapter context for the current forward pass."""

from __future__ import annotations

import contextvars
from contextlib import contextmanager

_current_adapter: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "mola_adapter", default=None
)
_current_slot_id: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "mola_adapter_slot", default=None
)


@contextmanager
def adapter_context(adapter_id: str | None, slot_id: int | None = None):
    adapter_token = _current_adapter.set(adapter_id)
    slot_token = _current_slot_id.set(slot_id)
    try:
        yield
    finally:
        _current_slot_id.reset(slot_token)
        _current_adapter.reset(adapter_token)


def get_current_adapter() -> str | None:
    return _current_adapter.get()


def get_current_slot_id() -> int | None:
    return _current_slot_id.get()
