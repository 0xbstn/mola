from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

from mola.application.packing import LayerSlotPackEntry, LayerSlotPackView, materialize_layer_slot_packs, routed_decode_delta_rows_reference


def _load_backend_module(monkeypatch, kernel_impl):
    mlx_pkg = ModuleType("mlx")
    core_mod = ModuleType("mlx.core")
    core_mod.array = lambda values, dtype=None: np.array(values, dtype=dtype)
    core_mod.concatenate = lambda values, axis=0: np.concatenate(values, axis=axis)
    core_mod.stack = lambda values, axis=0: np.stack(values, axis=axis)
    core_mod.float16 = np.dtype("float16")
    core_mod.bfloat16 = "bfloat16"
    core_mod.float32 = np.dtype("float32")
    core_mod.fast = SimpleNamespace(metal_kernel=lambda **kwargs: kernel_impl)
    mlx_pkg.core = core_mod
    monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", core_mod)

    module_name = "mola_test_metal_routed_decode"
    path = Path(__file__).resolve().parents[1] / "src/mola/infrastructure/metal_routed_decode.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


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


def _expected(token_slot_ids, x):
    pack = materialize_layer_slot_packs(
        (_view(),),
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


class TestMetalKernelRoutedDecodeBackend:
    def test_homogeneous_delta_matches_reference_oracle(self, monkeypatch):
        calls = []

        def fake_kernel(*, inputs, template, grid, threadgroup, output_shapes, output_dtypes):
            calls.append(
                {
                    "grid": grid,
                    "threadgroup": threadgroup,
                    "output_shapes": output_shapes,
                    "template": template,
                }
            )
            x, a, b, scale = inputs
            out = (scale[0] * ((x @ a) @ b)).astype(output_dtypes[0])
            return [out]

        module = _load_backend_module(monkeypatch, fake_kernel)
        factory = module.MetalKernelRoutedLoRADeltaSessionFactory(strict=True)
        session = factory.build((_view(),), (1, 1, 1))
        x = np.array([[10.0, 1.0], [2.0, 5.0], [7.0, 3.0]], dtype=np.float32)

        delta = session.delta("layers.0.q_proj", x)

        assert np.allclose(delta, _expected((1, 1, 1), x))
        assert len(calls) == 1
        assert calls[0]["output_shapes"] == [(3, 2)]

    def test_mixed_delta_matches_reference_oracle_and_restores_order(self, monkeypatch):
        calls = []

        def fake_kernel(*, inputs, template, grid, threadgroup, output_shapes, output_dtypes):
            calls.append(output_shapes[0])
            x, a, b, scale = inputs
            out = (scale[0] * ((x @ a) @ b)).astype(output_dtypes[0])
            return [out]

        module = _load_backend_module(monkeypatch, fake_kernel)
        factory = module.MetalKernelRoutedLoRADeltaSessionFactory(strict=True)
        session = factory.build((_view(),), (2, 1, 2))
        x = np.array(
            [
                [10.0, 1.0],
                [2.0, 5.0],
                [7.0, 3.0],
            ],
            dtype=np.float32,
        )

        delta = session.delta("layers.0.q_proj", x)

        assert np.allclose(delta, _expected((2, 1, 2), x))
        assert calls == [(1, 2), (2, 2)]
