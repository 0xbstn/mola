from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Sequence, TypeVar

SlotId = TypeVar("SlotId")


@dataclass(frozen=True)
class SlotSchedulingState(Generic[SlotId]):
    slot_id: SlotId
    active_count: int
    pending_count: int
    last_service_ts: float
    last_active_ts: float
    oldest_unstarted_ts: float | None


class WaitingAwareSchedulingPolicy(Generic[SlotId]):
    def order(self, slots: Sequence[SlotSchedulingState[SlotId]]) -> list[SlotId]:
        ranked = sorted(slots, key=self._priority)
        return [slot.slot_id for slot in ranked]

    def _priority(self, slot: SlotSchedulingState[SlotId]) -> tuple:
        wait_anchor = slot.oldest_unstarted_ts
        priority_class = 0
        if wait_anchor is None:
            if slot.pending_count:
                wait_anchor = slot.last_active_ts
                priority_class = 1
            else:
                wait_anchor = slot.last_service_ts or slot.last_active_ts
                priority_class = 2
        return (
            priority_class,
            wait_anchor,
            slot.last_service_ts or 0.0,
            -slot.pending_count,
            -slot.active_count,
        )
