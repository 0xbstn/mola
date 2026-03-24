from __future__ import annotations

from typing import Any, Protocol


class RoutedLoRADeltaSession(Protocol):
    def delta(self, layer_name: str, x: Any) -> Any | None: ...