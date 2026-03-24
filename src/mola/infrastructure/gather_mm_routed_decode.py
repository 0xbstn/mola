from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx

from mola.application.packing import LayerSlotPackView, materialize_layer_slot_packs
from mola.application.routed_decode import FrozenRoutedLayerExecution, RoutedDecodeContractError, freeze_routed_layer_execution, resolve_routed_layer_execution


@dataclass(frozen=True)
class GatherMMRoutedLoRADeltaSession:
    layer_executions: dict[str, FrozenRoutedLayerExecution]
    token_slot_ids: tuple[int, ...]
    strict: bool = False

    def delta(self, layer_name: str, x: mx.array) -> mx.array | None:
        execution = self.layer_executions.get(layer_name)
        if execution is None:
            if self.strict:
                raise RoutedDecodeContractError(
                    f"missing routed layer pack for layer '{layer_name}'"
                )
            return None

        validated = resolve_routed_layer_execution(
            layer_name,
            tuple(x.shape),
            type("_PackState", (), {"get": lambda _, __: execution.pack})(),
            self.token_slot_ids,
            strict=self.strict,
        )
        if validated is None:
            return None

        flat_x = x.reshape((-1, x.shape[-1]))
        if flat_x.shape[-1] != execution.abi.input_dim:
            if self.strict:
                raise RoutedDecodeContractError(
                    f"input dim mismatch for layer '{layer_name}': expected {execution.abi.input_dim}, got {flat_x.shape[-1]}"
                )
            return None

        try:
            if execution.plan.homogeneous_slot_id is not None:
                pack_row = execution.plan.groups[0].pack_row
                delta = (
                    flat_x @ execution.pack.lora_a[pack_row]
                ) @ execution.pack.lora_b[pack_row]
                delta = execution.pack.scales[pack_row] * delta
                return delta.reshape(tuple(x.shape[:-1]) + (delta.shape[-1],))

            slot_row_by_id = execution.pack.slot_row_by_id
            rhs_indices = mx.array(
                [slot_row_by_id[slot_id] for slot_id in execution.token_slot_ids],
                dtype=mx.int32,
            )
            gathered_scales = execution.pack.scales[rhs_indices].reshape((-1, 1))
            z = mx.gather_mm(
                mx.expand_dims(flat_x, -2),
                execution.pack.lora_a,
                rhs_indices=rhs_indices,
                sorted_indices=False,
            )
            y = mx.gather_mm(
                z,
                execution.pack.lora_b,
                rhs_indices=rhs_indices,
                sorted_indices=False,
            ).squeeze(-2)
            delta = gathered_scales * y
            return delta.reshape(tuple(x.shape[:-1]) + (delta.shape[-1],))
        except ValueError as exc:
            if self.strict:
                raise RoutedDecodeContractError(str(exc)) from exc
            return None


@dataclass(frozen=True)
class GatherMMRoutedLoRADeltaSessionFactory:
    strict: bool = False
    _pack_cache: dict[tuple, object] = field(default_factory=dict, init=False, repr=False, compare=False)

    def _cache_key(self, view: LayerSlotPackView) -> tuple:
        return (
            view.layer_name,
            view.slot_ids,
            tuple((id(entry.lora_a), id(entry.lora_b), entry.scale) for entry in view.entries),
        )

    def _materialize_pack(self, view: LayerSlotPackView):
        key = self._cache_key(view)
        cached = self._pack_cache.get(key)
        if cached is not None:
            return cached

        pack = materialize_layer_slot_packs(
            (view,),
            stack_fn=lambda values: mx.stack(values, axis=0),
            scale_fn=lambda values: mx.array(values),
        )[0]
        self._pack_cache[key] = pack
        return pack

    def build(
        self,
        views,
        token_slot_ids: tuple[int, ...],
    ) -> GatherMMRoutedLoRADeltaSession:
        layer_executions: dict[str, FrozenRoutedLayerExecution] = {}
        for view in views:
            pack = self._materialize_pack(view)
            try:
                execution = freeze_routed_layer_execution(pack, token_slot_ids)
            except RoutedDecodeContractError:
                if self.strict:
                    raise
                continue
            layer_executions[pack.layer_name] = execution

        return GatherMMRoutedLoRADeltaSession(
            layer_executions=layer_executions,
            token_slot_ids=token_slot_ids,
            strict=self.strict,
        )
