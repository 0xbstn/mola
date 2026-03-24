from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from mola.application.packing import RoutedLayerPackState, routed_decode_delta_rows_reference
from mola.application.routed_decode import resolve_routed_layer_execution


@dataclass(frozen=True)
class ReferenceRoutedLoRADeltaSession:
    layer_pack_state: RoutedLayerPackState
    token_slot_ids: tuple[int, ...]

    def delta(self, layer_name: str, x: mx.array) -> mx.array | None:
        execution = resolve_routed_layer_execution(
            layer_name,
            tuple(x.shape),
            self.layer_pack_state,
            self.token_slot_ids,
        )
        if execution is None:
            return None

        try:
            return routed_decode_delta_rows_reference(
                x,
                execution.pack,
                execution.token_slot_ids,
                flatten_fn=lambda array: (
                    array.reshape((-1, array.shape[-1])),
                    tuple(array.shape),
                ),
                restore_fn=lambda array, shape: array.reshape(
                    shape[:-1] + (array.shape[-1],)
                ),
                take_rows_fn=lambda array, rows: array[mx.array(rows)],
                concat_fn=lambda chunks: mx.concatenate(chunks, axis=0),
            )
        except ValueError:
            return None