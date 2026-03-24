from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol

from mola.adapter import AdapterSlotBinding


class SlotBoundLayer(Protocol):
    @property
    def slot_ids(self) -> tuple[int, ...]: ...

    def slot_bindings(self) -> list[tuple[int, Any, Any, float]]: ...


@dataclass(frozen=True)
class LayerSlotPackEntry:
    adapter_name: str
    slot_id: int
    lora_a: Any
    lora_b: Any
    scale: float


@dataclass(frozen=True)
class LayerSlotPackView:
    layer_name: str
    slot_ids: tuple[int, ...]
    entries: tuple[LayerSlotPackEntry, ...]


@dataclass(frozen=True)
class MaterializedLayerSlotPack:
    layer_name: str
    slot_ids: tuple[int, ...]
    adapter_names: tuple[str, ...]
    lora_a: Any
    lora_b: Any
    scales: tuple[float, ...]


def build_layer_slot_pack_views(
    active_bindings: Iterable[AdapterSlotBinding],
    layers: Iterable[tuple[str, SlotBoundLayer]],
) -> tuple[LayerSlotPackView, ...]:
    active_by_slot = {
        binding.slot_id: binding
        for binding in active_bindings
    }
    if not active_by_slot:
        return ()

    views: list[LayerSlotPackView] = []
    for layer_name, layer in layers:
        entries = [
            LayerSlotPackEntry(
                adapter_name=active_by_slot[slot_id].name,
                slot_id=slot_id,
                lora_a=lora_a,
                lora_b=lora_b,
                scale=scale,
            )
            for slot_id, lora_a, lora_b, scale in layer.slot_bindings()
            if slot_id in active_by_slot
        ]
        if not entries:
            continue
        entries.sort(key=lambda entry: entry.slot_id)
        views.append(
            LayerSlotPackView(
                layer_name=layer_name,
                slot_ids=tuple(entry.slot_id for entry in entries),
                entries=tuple(entries),
            )
        )

    return tuple(views)


def materialize_layer_slot_packs(
    views: Iterable[LayerSlotPackView],
    stack_fn: Callable[[list[Any]], Any],
) -> tuple[MaterializedLayerSlotPack, ...]:
    packs: list[MaterializedLayerSlotPack] = []
    for view in views:
        entries = list(view.entries)
        if not entries:
            continue
        packs.append(
            MaterializedLayerSlotPack(
                layer_name=view.layer_name,
                slot_ids=view.slot_ids,
                adapter_names=tuple(entry.adapter_name for entry in entries),
                lora_a=stack_fn([entry.lora_a for entry in entries]),
                lora_b=stack_fn([entry.lora_b for entry in entries]),
                scales=tuple(entry.scale for entry in entries),
            )
        )
    return tuple(packs)