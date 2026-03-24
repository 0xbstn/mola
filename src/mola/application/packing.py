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
    scales: Any

    @property
    def slot_row_by_id(self) -> dict[int, int]:
        return build_slot_row_map(self.slot_ids)


@dataclass(frozen=True)
class RoutedDecodeGroup:
    slot_id: int
    pack_row: int
    adapter_name: str
    token_rows: tuple[int, ...]


@dataclass(frozen=True)
class RoutedDecodePlan:
    token_count: int
    sorted_token_rows: tuple[int, ...]
    restore_order: tuple[int, ...]
    groups: tuple[RoutedDecodeGroup, ...]

    @property
    def homogeneous_slot_id(self) -> int | None:
        if len(self.groups) != 1:
            return None
        return self.groups[0].slot_id


@dataclass(frozen=True)
class RoutedLayerPackState:
    packs_by_layer: dict[str, MaterializedLayerSlotPack]

    def get(self, layer_name: str) -> MaterializedLayerSlotPack | None:
        return self.packs_by_layer.get(layer_name)


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


def build_slot_row_map(slot_ids: Iterable[int]) -> dict[int, int]:
    slot_row_by_id: dict[int, int] = {}
    for row, slot_id in enumerate(slot_ids):
        if slot_id in slot_row_by_id:
            raise ValueError(f"duplicate slot_id in pack: {slot_id}")
        slot_row_by_id[slot_id] = row
    return slot_row_by_id


def build_layer_slot_pack_state(
    packs: Iterable[MaterializedLayerSlotPack],
) -> RoutedLayerPackState:
    packs_by_layer: dict[str, MaterializedLayerSlotPack] = {}
    for pack in packs:
        if pack.layer_name in packs_by_layer:
            raise ValueError(f"duplicate layer pack: {pack.layer_name}")
        packs_by_layer[pack.layer_name] = pack
    return RoutedLayerPackState(packs_by_layer=packs_by_layer)


def build_routed_decode_plan(
    pack: MaterializedLayerSlotPack,
    token_slot_ids: Iterable[int],
) -> RoutedDecodePlan:
    token_slot_ids = tuple(token_slot_ids)
    if not token_slot_ids:
        return RoutedDecodePlan(
            token_count=0,
            sorted_token_rows=(),
            restore_order=(),
            groups=(),
        )

    slot_row_by_id = pack.slot_row_by_id
    grouped_rows: dict[int, list[int]] = {}
    for row, slot_id in enumerate(token_slot_ids):
        pack_row = slot_row_by_id.get(slot_id)
        if pack_row is None:
            raise ValueError(
                f"layer '{pack.layer_name}' has no packed binding for slot_id={slot_id}"
            )
        grouped_rows.setdefault(pack_row, []).append(row)

    groups = tuple(
        RoutedDecodeGroup(
            slot_id=pack.slot_ids[pack_row],
            pack_row=pack_row,
            adapter_name=pack.adapter_names[pack_row],
            token_rows=tuple(grouped_rows[pack_row]),
        )
        for pack_row in sorted(grouped_rows)
    )
    sorted_token_rows = tuple(
        row
        for group in groups
        for row in group.token_rows
    )
    restore_order = [0] * len(sorted_token_rows)
    for sorted_pos, row in enumerate(sorted_token_rows):
        restore_order[row] = sorted_pos

    return RoutedDecodePlan(
        token_count=len(token_slot_ids),
        sorted_token_rows=sorted_token_rows,
        restore_order=tuple(restore_order),
        groups=groups,
    )


def flatten_token_rows(
    x: Any,
    *,
    flatten_fn: Callable[[Any], tuple[Any, tuple[int, ...]]],
) -> tuple[Any, tuple[int, ...]]:
    return flatten_fn(x)


def restore_token_rows(
    x: Any,
    original_shape: tuple[int, ...],
    *,
    restore_fn: Callable[[Any, tuple[int, ...]], Any],
) -> Any:
    return restore_fn(x, original_shape)


def routed_decode_delta_reference(
    x: Any,
    pack: MaterializedLayerSlotPack,
    token_slot_ids: Iterable[int],
    *,
    take_rows_fn: Callable[[Any, tuple[int, ...]], Any],
    concat_fn: Callable[[list[Any]], Any],
) -> Any:
    plan = build_routed_decode_plan(pack, token_slot_ids)
    if not plan.groups:
        raise ValueError("routed decode requires at least one token row")

    sorted_chunks = []
    for group in plan.groups:
        x_group = take_rows_fn(x, group.token_rows)
        delta = (x_group @ pack.lora_a[group.pack_row]) @ pack.lora_b[group.pack_row]
        sorted_chunks.append(pack.scales[group.pack_row] * delta)

    sorted_delta = concat_fn(sorted_chunks)
    return take_rows_fn(sorted_delta, plan.restore_order)


def routed_decode_delta_rows_reference(
    x: Any,
    pack: MaterializedLayerSlotPack,
    token_slot_ids: Iterable[int],
    *,
    flatten_fn: Callable[[Any], tuple[Any, tuple[int, ...]]],
    restore_fn: Callable[[Any, tuple[int, ...]], Any],
    take_rows_fn: Callable[[Any, tuple[int, ...]], Any],
    concat_fn: Callable[[list[Any]], Any],
) -> Any:
    flat_x, original_shape = flatten_token_rows(x, flatten_fn=flatten_fn)
    flat_delta = routed_decode_delta_reference(
        flat_x,
        pack,
        token_slot_ids,
        take_rows_fn=take_rows_fn,
        concat_fn=concat_fn,
    )
    return restore_token_rows(flat_delta, original_shape, restore_fn=restore_fn)


def materialize_layer_slot_packs(
    views: Iterable[LayerSlotPackView],
    stack_fn: Callable[[list[Any]], Any],
    scale_fn: Callable[[list[float]], Any] | None = None,
) -> tuple[MaterializedLayerSlotPack, ...]:
    packs: list[MaterializedLayerSlotPack] = []
    for view in views:
        entries = list(view.entries)
        if not entries:
            continue
        scale_values = [entry.scale for entry in entries]
        packs.append(
            MaterializedLayerSlotPack(
                layer_name=view.layer_name,
                slot_ids=view.slot_ids,
                adapter_names=tuple(entry.adapter_name for entry in entries),
                lora_a=stack_fn([entry.lora_a for entry in entries]),
                lora_b=stack_fn([entry.lora_b for entry in entries]),
                scales=scale_fn(scale_values) if scale_fn else tuple(scale_values),
            )
        )
    return tuple(packs)