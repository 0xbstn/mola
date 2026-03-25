from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx

from mola.application.packing import LayerSlotPackView, materialize_layer_slot_packs
from mola.application.routed_decode import (
    FrozenRoutedLayerExecution,
    RoutedDecodeContractError,
    freeze_routed_layer_execution,
    resolve_routed_layer_execution,
)


_METAL_GATHER_MIXED_SOURCE = """
uint lane_x = thread_position_in_threadgroup.x;
uint lane_y = thread_position_in_threadgroup.y;
uint row = threadgroup_position_in_grid.y;
uint rows = x_shape[0];
uint in_dim = x_shape[1];
uint rank = a_shape[2];
uint out_dim = b_shape[2];

if (row >= rows) {
    return;
}

uint slot = uint(slot_rows[row]);
threadgroup T partial[THREADS_Y][THREADS_X];
threadgroup T zbuf[MAX_R];

if (lane_y < rank) {
    T sum = T(0);
    for (uint d = lane_x; d < in_dim; d += THREADS_X) {
        uint a_idx = ((slot * in_dim + d) * rank) + lane_y;
        sum += T(x[row * in_dim + d]) * T(a[a_idx]);
    }
    partial[lane_y][lane_x] = sum;
}

threadgroup_barrier(mem_flags::mem_threadgroup);

if (lane_y < rank && lane_x == 0) {
    T z = T(0);
    for (uint lx = 0; lx < THREADS_X; ++lx) {
        z += partial[lane_y][lx];
    }
    zbuf[lane_y] = z;
}

threadgroup_barrier(mem_flags::mem_threadgroup);

if (lane_y == 0) {
    T scale = T(scales[slot]);
    for (uint out_col = lane_x; out_col < out_dim; out_col += THREADS_X) {
        T acc = T(0);
        for (uint r = 0; r < rank; ++r) {
            uint b_idx = ((slot * rank + r) * out_dim) + out_col;
            acc += zbuf[r] * T(b[b_idx]);
        }
        out[row * out_dim + out_col] = scale * acc;
    }
}
"""


@dataclass(frozen=True)
class MetalGatherRoutedLoRADeltaSession:
    layer_executions: dict[str, FrozenRoutedLayerExecution]
    token_slot_ids: tuple[int, ...]
    kernel: object | None
    strict: bool = False

    def _kernel_template_dtype(self, dtype):
        if dtype in (mx.float16, mx.bfloat16):
            return mx.float32
        return dtype

    def _should_use_metal(self, execution: FrozenRoutedLayerExecution) -> bool:
        return (
            execution.abi.output_dim <= 1024
            and execution.abi.rank * 128 <= 1024
        )

    def _metal_threads_x(
        self,
        execution: FrozenRoutedLayerExecution,
        layer_name: str,
    ) -> int:
        rank = int(execution.abi.rank)
        limit = 1024 // max(rank, 1)
        if layer_name.endswith((".q_proj", ".o_proj")):
            return min(96, limit)
        return min(128, limit)

    def _gather_delta(
        self,
        flat_x: mx.array,
        execution: FrozenRoutedLayerExecution,
    ) -> mx.array:
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
        return gathered_scales * y

    def _metal_delta(
        self,
        flat_x: mx.array,
        execution: FrozenRoutedLayerExecution,
        layer_name: str,
    ) -> mx.array:
        if self.kernel is None:
            raise RoutedDecodeContractError("metal-gather routed decode kernel is unavailable")
        rank = int(execution.abi.rank)
        threads_x = self._metal_threads_x(execution, layer_name)
        slot_row_by_id = execution.pack.slot_row_by_id
        slot_rows = mx.array(
            [slot_row_by_id[slot_id] for slot_id in execution.token_slot_ids],
            dtype=mx.int32,
        )
        return self.kernel(
            inputs=[
                flat_x,
                execution.pack.lora_a,
                execution.pack.lora_b,
                execution.pack.scales,
                slot_rows,
            ],
            template=[
                ("T", self._kernel_template_dtype(flat_x.dtype)),
                ("MAX_R", rank),
                ("THREADS_X", threads_x),
                ("THREADS_Y", rank),
            ],
            grid=(threads_x, int(flat_x.shape[0]) * rank, 1),
            threadgroup=(threads_x, rank, 1),
            output_shapes=[(int(flat_x.shape[0]), execution.abi.output_dim)],
            output_dtypes=[flat_x.dtype],
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
                pack_row = execution.plan.groups[0].pack_row
                delta = (
                    flat_x @ execution.pack.lora_a[pack_row]
                ) @ execution.pack.lora_b[pack_row]
                delta = execution.pack.scales[pack_row] * delta
                return delta.reshape(tuple(x.shape[:-1]) + (delta.shape[-1],))

            delta = (
                self._metal_delta(flat_x, execution, layer_name)
                if self._should_use_metal(execution)
                else self._gather_delta(flat_x, execution)
            )
            return delta.reshape(tuple(x.shape[:-1]) + (delta.shape[-1],))
        except ValueError as exc:
            if self.strict:
                raise RoutedDecodeContractError(str(exc)) from exc
            return None


@dataclass
class MetalGatherRoutedLoRADeltaSessionFactory:
    strict: bool = False
    _pack_cache: dict[tuple, object] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _kernel: object | None = field(default=None, init=False, repr=False, compare=False)

    def _cache_key(self, view: LayerSlotPackView) -> tuple:
        return (
            view.layer_name,
            view.slot_ids,
            tuple(
                (id(entry.lora_a), id(entry.lora_b), entry.scale)
                for entry in view.entries
            ),
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
                name="mola_routed_lora_delta_metal_gather_v1",
                input_names=["x", "a", "b", "scales", "slot_rows"],
                output_names=["out"],
                source=_METAL_GATHER_MIXED_SOURCE,
                ensure_row_contiguous=True,
            )
        return self._kernel

    def build(
        self,
        views,
        token_slot_ids: tuple[int, ...],
    ) -> MetalGatherRoutedLoRADeltaSession:
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
        return MetalGatherRoutedLoRADeltaSession(
            layer_executions=layer_executions,
            token_slot_ids=token_slot_ids,
            kernel=kernel,
            strict=self.strict,
        )
