"""Adapter context for the current forward pass."""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mola.application.packing import RoutedLayerPackState

_current_adapter: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "mola_adapter", default=None
)
_current_slot_id: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "mola_adapter_slot", default=None
)
_current_routed_pack_state: contextvars.ContextVar[RoutedLayerPackState | None] = contextvars.ContextVar(
    "mola_routed_pack_state", default=None
)
_current_token_slot_ids: contextvars.ContextVar[tuple[int, ...] | None] = contextvars.ContextVar(
    "mola_token_slot_ids", default=None
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
def routed_decode_context(
    layer_pack_state: RoutedLayerPackState | None,
    token_slot_ids: tuple[int, ...] | None,
):
    pack_token = _current_routed_pack_state.set(layer_pack_state)
    slot_ids_token = _current_token_slot_ids.set(token_slot_ids)
    try:
        yield
    finally:
        _current_token_slot_ids.reset(slot_ids_token)
        _current_routed_pack_state.reset(pack_token)


def get_current_adapter() -> str | None:
    return _current_adapter.get()


def get_current_slot_id() -> int | None:
    return _current_slot_id.get()


def get_current_routed_pack_state() -> RoutedLayerPackState | None:
    return _current_routed_pack_state.get()


def get_current_token_slot_ids() -> tuple[int, ...] | None:
    return _current_token_slot_ids.get()
