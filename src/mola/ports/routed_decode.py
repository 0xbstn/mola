from __future__ import annotations

from typing import Any, Protocol

from mola.application.packing import LayerSlotPackView


class RoutedLoRADeltaSession(Protocol):
    def delta(self, layer_name: str, x: Any) -> Any | None: ...


class RoutedLoRADeltaSessionFactory(Protocol):
    def build(
        self,
        views: tuple[LayerSlotPackView, ...],
        token_slot_ids: tuple[int, ...],
    ) -> RoutedLoRADeltaSession: ...