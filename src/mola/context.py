"""Adapter context for the current forward pass."""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mola.ports.routed_decode import RoutedLoRADeltaSession

_current_adapter: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "mola_adapter", default=None
)
_current_slot_id: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "mola_adapter_slot", default=None
)
_current_routed_decode_session: contextvars.ContextVar[RoutedLoRADeltaSession | None] = contextvars.ContextVar(
    "mola_routed_decode_session", default=None
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


@contextmanager
def routed_decode_context(session: RoutedLoRADeltaSession | None):
    session_token = _current_routed_decode_session.set(session)
    try:
        yield
    finally:
        _current_routed_decode_session.reset(session_token)


def get_current_adapter() -> str | None:
    return _current_adapter.get()


def get_current_slot_id() -> int | None:
    return _current_slot_id.get()


def get_current_routed_decode_session() -> RoutedLoRADeltaSession | None:
    return _current_routed_decode_session.get()
