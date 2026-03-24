from __future__ import annotations

from dataclasses import dataclass

from mola.application.packing import MaterializedLayerSlotPack, RoutedLayerPackState


class RoutedDecodeContractError(ValueError):
    pass


@dataclass(frozen=True)
class RoutedLayerExecution:
    pack: MaterializedLayerSlotPack
    token_slot_ids: tuple[int, ...]


def resolve_routed_layer_execution(
    layer_name: str | None,
    x_shape: tuple[int, ...],
    layer_pack_state: RoutedLayerPackState | None,
    token_slot_ids: tuple[int, ...] | None,
    *,
    strict: bool = False,
) -> RoutedLayerExecution | None:
    if layer_name is None or layer_pack_state is None or token_slot_ids is None:
        return None
    if not token_slot_ids or not x_shape:
        return None

    pack = layer_pack_state.get(layer_name)
    if pack is None:
        if strict:
            raise RoutedDecodeContractError(
                f"missing routed layer pack for layer '{layer_name}'"
            )
        return None

    row_count = 1
    for dim in x_shape[:-1]:
        row_count *= dim
    if row_count != len(token_slot_ids):
        if strict:
            raise RoutedDecodeContractError(
                f"row count mismatch for layer '{layer_name}': "
                f"expected {len(token_slot_ids)}, got {row_count}"
            )
        return None

    return RoutedLayerExecution(
        pack=pack,
        token_slot_ids=token_slot_ids,
    )
