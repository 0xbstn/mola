from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from mola.application.packing import LayerSlotPackView, materialize_layer_slot_packs
from mola.application.routed_decode import FrozenRoutedLayerExecution, RoutedDecodeContractError, freeze_routed_layer_execution, resolve_routed_layer_execution


_METAL_LORA_DELTA_SOURCE = """
uint elem = thread_position_in_grid.x;
uint rows = x_shape[0];
uint in_dim = x_shape[1];
uint rank = a_shape[1];
uint out_dim = b_shape[1];
uint total = rows * out_dim;
if (elem >= total) {
    return;
}

uint row = elem / out_dim;
uint col = elem % out_dim;
T acc = T(0);
for (uint r = 0; r < rank; ++r) {
    T inner = T(0);
    for (uint d = 0; d < in_dim; ++d) {
        inner += T(x[row * in_dim + d]) * T(a[d * rank + r]);
    }
    acc += inner * T(b[r * out_dim + col]);
}
out[elem] = T(scale[0]) * acc;
"""


@dataclass(frozen=True)
class MetalKernelRoutedLoRADeltaSession:
    layer_executions: dict[str, FrozenRoutedLayerExecution]
    token_slot_ids: tuple[int, ...]
    kernel: Any | None
    strict: bool = False

    def _kernel_template_dtype(self, dtype):
        if dtype in (mx.float16, mx.bfloat16):
            return mx.float32
        return dtype

    def _run_group_kernel(
        self,
        flat_x: mx.array,
        execution: FrozenRoutedLayerExecution,
        group_index: int,
    ) -> mx.array:
        if self.kernel is None:
            raise RoutedDecodeContractError("metal routed decode kernel is unavailable")
        group = execution.plan.groups[group_index]
        x_group = flat_x[mx.array(group.token_rows)]
        lora_a = execution.pack.lora_a[group.pack_row]
        lora_b = execution.pack.lora_b[group.pack_row]
        scale = execution.pack.scales[group.pack_row : group.pack_row + 1]

        if lora_a.dtype != x_group.dtype:
            lora_a = lora_a.astype(x_group.dtype)
        if lora_b.dtype != x_group.dtype:
            lora_b = lora_b.astype(x_group.dtype)
        if scale.dtype != x_group.dtype:
            scale = scale.astype(x_group.dtype)

        rows = int(x_group.shape[0])
        out_dim = int(lora_b.shape[1])
        grid_x = max(rows * out_dim, 1)
        threadgroup_x = min(256, grid_x)
        return self.kernel(
            inputs=[x_group, lora_a, lora_b, scale],
            template=[("T", self._kernel_template_dtype(x_group.dtype))],
            grid=(grid_x, 1, 1),
            threadgroup=(threadgroup_x, 1, 1),
            output_shapes=[(rows, out_dim)],
            output_dtypes=[x_group.dtype],
        )[0]

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
                flat_delta = self._run_group_kernel(flat_x, execution, 0)
                return flat_delta.reshape(tuple(x.shape[:-1]) + (flat_delta.shape[-1],))

            sorted_chunks = [
                self._run_group_kernel(flat_x, execution, group_index)
                for group_index in range(len(execution.plan.groups))
            ]
            sorted_delta = mx.concatenate(sorted_chunks, axis=0)
            restored = sorted_delta[mx.array(execution.plan.restore_order)]
            return restored.reshape(tuple(x.shape[:-1]) + (restored.shape[-1],))
        except ValueError as exc:
            if self.strict:
                raise RoutedDecodeContractError(str(exc)) from exc
            return None


@dataclass
class MetalKernelRoutedLoRADeltaSessionFactory:
    strict: bool = False
    _pack_cache: dict[tuple, object] = field(default_factory=dict, init=False, repr=False, compare=False)
    _kernel: Any | None = field(default=None, init=False, repr=False, compare=False)

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

    def _get_kernel(self):
        if self._kernel is None:
            self._kernel = mx.fast.metal_kernel(
                name="mola_routed_lora_delta_v1",
                input_names=["x", "a", "b", "scale"],
                output_names=["out"],
                source=_METAL_LORA_DELTA_SOURCE,
                ensure_row_contiguous=True,
            )
        return self._kernel

    def build(
        self,
        views,
        token_slot_ids: tuple[int, ...],
    ) -> MetalKernelRoutedLoRADeltaSession:
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

        kernel = self._get_kernel() if layer_executions else None
        return MetalKernelRoutedLoRADeltaSession(
            layer_executions=layer_executions,
            token_slot_ids=token_slot_ids,
            kernel=kernel,
            strict=self.strict,
        )
