from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

from mola.application.packing import (
    LayerSlotPackEntry,
    LayerSlotPackView,
    materialize_layer_slot_packs,
    routed_decode_delta_rows_reference,
)


def _view():
    return LayerSlotPackView(
        layer_name="layers.0.q_proj",
        slot_ids=(1, 2),
        entries=(
            LayerSlotPackEntry(
                adapter_name="rust",
                slot_id=1,
                lora_a=np.array([[1.0], [0.0]]),
                lora_b=np.array([[2.0, 3.0]]),
                scale=2.0,
            ),
            LayerSlotPackEntry(
                adapter_name="sql",
                slot_id=2,
                lora_a=np.array([[0.0], [1.0]]),
                lora_b=np.array([[4.0, 1.0]]),
                scale=0.5,
            ),
        ),
    )


def _gate_view():
    return LayerSlotPackView(
        layer_name="layers.0.gate_proj",
        slot_ids=(1, 2),
        entries=(
            LayerSlotPackEntry(
                adapter_name="rust",
                slot_id=1,
                lora_a=np.array([[1.0], [0.0]]),
                lora_b=np.array([[2.0, 3.0]]),
                scale=2.0,
            ),
            LayerSlotPackEntry(
                adapter_name="sql",
                slot_id=2,
                lora_a=np.array([[0.0], [1.0]]),
                lora_b=np.array([[4.0, 1.0]]),
                scale=0.5,
            ),
        ),
    )


def _other_view():
    return LayerSlotPackView(
        layer_name="layers.0.other_proj",
        slot_ids=(1, 2),
        entries=(
            LayerSlotPackEntry(
                adapter_name="rust",
                slot_id=1,
                lora_a=np.array([[1.0], [0.0]]),
                lora_b=np.array([[2.0, 3.0]]),
                scale=2.0,
            ),
            LayerSlotPackEntry(
                adapter_name="sql",
                slot_id=2,
                lora_a=np.array([[0.0], [1.0]]),
                lora_b=np.array([[4.0, 1.0]]),
                scale=0.5,
            ),
        ),
    )


def _wide_view():
    return LayerSlotPackView(
        layer_name="layers.0.up_proj",
        slot_ids=(1, 2),
        entries=(
            LayerSlotPackEntry(
                adapter_name="rust",
                slot_id=1,
                lora_a=np.array([[1.0], [0.0]]),
                lora_b=np.ones((1, 1025), dtype=np.float32),
                scale=2.0,
            ),
            LayerSlotPackEntry(
                adapter_name="sql",
                slot_id=2,
                lora_a=np.array([[0.0], [1.0]]),
                lora_b=np.full((1, 1025), 0.5, dtype=np.float32),
                scale=0.5,
            ),
        ),
    )


def _expected(view, token_slot_ids, x):
    pack = materialize_layer_slot_packs(
        (view,),
        stack_fn=lambda values: np.stack(values, axis=0),
        scale_fn=lambda values: np.array(values),
    )[0]
    return routed_decode_delta_rows_reference(
        x,
        pack,
        token_slot_ids,
        flatten_fn=lambda array: (array.reshape(-1, array.shape[-1]), array.shape),
        restore_fn=lambda array, shape: array.reshape(shape[:-1] + (array.shape[-1],)),
        take_rows_fn=lambda array, rows: array[list(rows)],
        concat_fn=lambda chunks: np.concatenate(chunks, axis=0),
    )


def _load_backend_module(monkeypatch, *, kernel_raises: bool = False):
    gather_calls = []
    kernel_calls = []

    def gather_mm(lhs, rhs, *, rhs_indices, sorted_indices=False):
        lhs = np.asarray(lhs)
        rhs = np.asarray(rhs)
        rhs_indices = np.asarray(rhs_indices, dtype=np.int32)
        gather_calls.append(rhs_indices.tolist())
        out = []
        for row, rhs_index in enumerate(rhs_indices):
            out.append(lhs[row] @ rhs[rhs_index])
        return np.stack(out, axis=0)

    def fake_kernel(*, inputs, template, grid, threadgroup, output_shapes, output_dtypes):
        kernel_calls.append(
            {
                "grid": grid,
                "threadgroup": threadgroup,
                "template": template,
                "output_shapes": output_shapes,
            }
        )
        x, a, b, scales, slot_rows = inputs
        rows = []
        for row, slot_row in enumerate(np.asarray(slot_rows, dtype=np.int32).tolist()):
            rows.append(scales[slot_row] * ((x[row] @ a[slot_row]) @ b[slot_row]))
        return [np.stack(rows, axis=0).astype(output_dtypes[0])]

    def load_kernel(**kwargs):
        if kernel_raises:
            raise RuntimeError("no metal")
        return fake_kernel

    mlx_pkg = ModuleType("mlx")
    core_mod = ModuleType("mlx.core")
    core_mod.array = lambda values, dtype=None: np.array(values, dtype=dtype)
    core_mod.concatenate = lambda values, axis=0: np.concatenate(values, axis=axis)
    core_mod.stack = lambda values, axis=0: np.stack(values, axis=axis)
    core_mod.expand_dims = np.expand_dims
    core_mod.gather_mm = gather_mm
    core_mod.matmul = np.matmul
    core_mod.float16 = np.dtype("float16")
    core_mod.bfloat16 = "bfloat16"
    core_mod.float32 = np.dtype("float32")
    core_mod.int32 = np.int32
    core_mod.fast = SimpleNamespace(metal_kernel=load_kernel)
    mlx_pkg.core = core_mod
    monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", core_mod)

    module_name = "mola_test_metal_gather_routed_decode"
    path = (
        Path(__file__).resolve().parents[1]
        / "src/mola/infrastructure/metal_gather_routed_decode.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module, gather_calls, kernel_calls


class TestMetalGatherRoutedDecodeBackend:
    def test_homogeneous_delta_matches_reference_oracle(self, monkeypatch):
        module, gather_calls, kernel_calls = _load_backend_module(monkeypatch)
        factory = module.MetalGatherRoutedLoRADeltaSessionFactory(strict=True)
        session = factory.build((_view(),), (1, 1, 1))
        x = np.array([[10.0, 1.0], [2.0, 5.0], [7.0, 3.0]], dtype=np.float32)

        delta = session.delta("layers.0.q_proj", x)

        assert np.allclose(delta, _expected(_view(), (1, 1, 1), x))
        assert gather_calls == []
        assert kernel_calls == []

    def test_mixed_small_output_uses_metal_kernel(self, monkeypatch):
        module, gather_calls, kernel_calls = _load_backend_module(monkeypatch)
        factory = module.MetalGatherRoutedLoRADeltaSessionFactory(strict=True)
        token_slot_ids = tuple(([2, 1] * 8))
        session = factory.build((_view(),), token_slot_ids)
        x = np.tile(np.array([[10.0, 1.0], [2.0, 5.0]], dtype=np.float32), (8, 1))

        delta = session.delta("layers.0.q_proj", x)

        assert np.allclose(delta, _expected(_view(), token_slot_ids, x))
        assert gather_calls == []
        assert len(kernel_calls) == 1
        assert kernel_calls[0]["output_shapes"] == [(16, 2)]

    def test_mixed_large_output_uses_metal_kernel(self, monkeypatch):
        module, gather_calls, kernel_calls = _load_backend_module(monkeypatch)
        factory = module.MetalGatherRoutedLoRADeltaSessionFactory(strict=True)
        session = factory.build((_wide_view(),), (2, 1, 2))
        x = np.array([[10.0, 1.0], [2.0, 5.0], [7.0, 3.0]], dtype=np.float32)

        delta = session.delta("layers.0.up_proj", x)

        assert np.allclose(delta, _expected(_wide_view(), (2, 1, 2), x))
        assert gather_calls == []
        assert len(kernel_calls) == 1
        assert kernel_calls[0]["output_shapes"] == [(3, 1025)]

    def test_gate_proj_uses_metal_when_small(self, monkeypatch):
        module, gather_calls, kernel_calls = _load_backend_module(monkeypatch)
        factory = module.MetalGatherRoutedLoRADeltaSessionFactory(strict=True)
        token_slot_ids = tuple(([2, 1] * 8))
        session = factory.build((_gate_view(),), token_slot_ids)
        x = np.tile(np.array([[10.0, 1.0], [2.0, 5.0]], dtype=np.float32), (8, 1))

        delta = session.delta("layers.0.gate_proj", x)

        assert np.allclose(delta, _expected(_gate_view(), token_slot_ids, x))
        assert gather_calls == []
        assert len(kernel_calls) == 1

    def test_unknown_layer_family_stays_on_gather(self, monkeypatch):
        module, gather_calls, kernel_calls = _load_backend_module(monkeypatch)
        factory = module.MetalGatherRoutedLoRADeltaSessionFactory(strict=True)
        token_slot_ids = tuple(([2, 1] * 8))
        session = factory.build((_other_view(),), token_slot_ids)
        x = np.tile(np.array([[10.0, 1.0], [2.0, 5.0]], dtype=np.float32), (8, 1))

        delta = session.delta("layers.0.other_proj", x)

        assert np.allclose(delta, _expected(_other_view(), token_slot_ids, x))
        assert gather_calls == [[1, 0] * 8, [1, 0] * 8]
        assert kernel_calls == []

    def test_kernel_creation_failure_falls_back_to_gather(self, monkeypatch):
        module, gather_calls, kernel_calls = _load_backend_module(
            monkeypatch, kernel_raises=True
        )
        factory = module.MetalGatherRoutedLoRADeltaSessionFactory(strict=True)
        token_slot_ids = tuple(([2, 1] * 8))
        session = factory.build((_view(),), token_slot_ids)
        x = np.tile(np.array([[10.0, 1.0], [2.0, 5.0]], dtype=np.float32), (8, 1))

        delta = session.delta("layers.0.q_proj", x)

        assert np.allclose(delta, _expected(_view(), token_slot_ids, x))
        assert gather_calls == [[1, 0] * 8, [1, 0] * 8]
        assert kernel_calls == []
