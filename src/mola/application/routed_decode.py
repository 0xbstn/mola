from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from mola.application.packing import MaterializedLayerSlotPack, RoutedDecodePlan, RoutedLayerPackState, build_routed_decode_plan, build_slot_row_map


class RoutedDecodeContractError(ValueError):
    pass


ROUTED_DECODE_SLOT_ID_DTYPE = "int32"
ROUTED_DECODE_REQUIRE_ROW_CONTIGUOUS = True


@dataclass(frozen=True)
class RoutedDecodeLayerABI:
    layer_name: str
    slot_ids: tuple[int, ...]
    slot_count: int
    input_dim: int
    rank: int
    output_dim: int
    slot_id_dtype: str = ROUTED_DECODE_SLOT_ID_DTYPE
    require_row_contiguous: bool = ROUTED_DECODE_REQUIRE_ROW_CONTIGUOUS


@dataclass(frozen=True)
class FrozenRoutedLayerExecution:
    pack: MaterializedLayerSlotPack
    plan: RoutedDecodePlan
    abi: RoutedDecodeLayerABI
    token_slot_ids: tuple[int, ...]


@dataclass(frozen=True)
class RoutedLayerExecution:
    pack: MaterializedLayerSlotPack
    token_slot_ids: tuple[int, ...]


def _normalize_slot_ids(token_slot_ids: Iterable[int]) -> tuple[int, ...]:
    normalized = tuple(token_slot_ids)
    for slot_id in normalized:
        if type(slot_id) is not int:
            raise RoutedDecodeContractError(
                f"slot_ids must be plain ints for ABI v1, got {type(slot_id).__name__}"
            )
    return normalized


def _shape_tuple(name: str, value) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise RoutedDecodeContractError(f"{name} must expose shape for ABI v1")
    return tuple(int(dim) for dim in shape)


def freeze_routed_decode_layer_abi(
    pack: MaterializedLayerSlotPack,
) -> RoutedDecodeLayerABI:
    slot_ids = tuple(pack.slot_ids)
    if not slot_ids:
        raise RoutedDecodeContractError(
            f"layer '{pack.layer_name}' has no slot_ids for routed decode"
        )
    if slot_ids != tuple(sorted(slot_ids)):
        raise RoutedDecodeContractError(
            f"layer '{pack.layer_name}' slot_ids must be sorted for ABI v1"
        )
    build_slot_row_map(slot_ids)

    slot_count = len(slot_ids)
    if len(pack.adapter_names) != slot_count:
        raise RoutedDecodeContractError(
            f"layer '{pack.layer_name}' adapter_names length does not match slot_ids"
        )

    lora_a_shape = _shape_tuple("lora_a", pack.lora_a)
    lora_b_shape = _shape_tuple("lora_b", pack.lora_b)
    scales_shape = _shape_tuple("scales", pack.scales)

    if len(lora_a_shape) != 3:
        raise RoutedDecodeContractError(
            f"layer '{pack.layer_name}' expects lora_a with shape [S, D, R], got {lora_a_shape}"
        )
    if len(lora_b_shape) != 3:
        raise RoutedDecodeContractError(
            f"layer '{pack.layer_name}' expects lora_b with shape [S, R, O], got {lora_b_shape}"
        )
    if len(scales_shape) != 1:
        raise RoutedDecodeContractError(
            f"layer '{pack.layer_name}' expects scales with shape [S], got {scales_shape}"
        )

    if lora_a_shape[0] != slot_count or lora_b_shape[0] != slot_count or scales_shape[0] != slot_count:
        raise RoutedDecodeContractError(
            f"layer '{pack.layer_name}' slot count mismatch across packed tensors"
        )

    input_dim = lora_a_shape[1]
    rank = lora_a_shape[2]
    output_dim = lora_b_shape[2]

    if rank != lora_b_shape[1]:
        raise RoutedDecodeContractError(
            f"layer '{pack.layer_name}' rank mismatch: lora_a={lora_a_shape}, lora_b={lora_b_shape}"
        )
    if input_dim <= 0 or rank <= 0 or output_dim <= 0:
        raise RoutedDecodeContractError(
            f"layer '{pack.layer_name}' packed dims must be positive, got {lora_a_shape} and {lora_b_shape}"
        )

    return RoutedDecodeLayerABI(
        layer_name=pack.layer_name,
        slot_ids=slot_ids,
        slot_count=slot_count,
        input_dim=input_dim,
        rank=rank,
        output_dim=output_dim,
    )


def freeze_routed_layer_execution(
    pack: MaterializedLayerSlotPack,
    token_slot_ids: Iterable[int],
) -> FrozenRoutedLayerExecution:
    normalized_slot_ids = _normalize_slot_ids(token_slot_ids)
    abi = freeze_routed_decode_layer_abi(pack)
    try:
        plan = build_routed_decode_plan(pack, normalized_slot_ids)
    except ValueError as exc:
        raise RoutedDecodeContractError(str(exc)) from exc
    return FrozenRoutedLayerExecution(
        pack=pack,
        plan=plan,
        abi=abi,
        token_slot_ids=normalized_slot_ids,
    )


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
