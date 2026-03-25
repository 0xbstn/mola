"""Tests for the batched generation engine."""

from __future__ import annotations

import asyncio
import queue
import sys
import threading
import time
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mola.application.packing import build_layer_slot_pack_state, materialize_layer_slot_packs
from mola.application.routed_decode import RoutedDecodeContractError
from mola.engine import (
    AdmissionRejected,
    DecodeRowBinding,
    EngineConfig,
    EngineMetrics,
    GenerateRequest,
    MIXED_DECODE_ADAPTER_ID,
    MOLAEngine,
    RuntimeSlotLayout,
    _AdapterSlot,
    _get_stop_tokens,
)
from mola.adapter import AdapterSlotBinding
from mola.ports.generator import GeneratorHandle, GeneratorState


def _tuple_routed_session_factory():
    return SimpleNamespace(
        build=lambda views, token_slot_ids: (
            build_layer_slot_pack_state(
                materialize_layer_slot_packs(
                    views,
                    stack_fn=lambda values: tuple(values),
                    scale_fn=lambda values: tuple(values),
                )
            ),
            token_slot_ids,
        )
    )


def _make_engine(**overrides):
    mola_model = MagicMock()
    mola_model.tokenizer.eos_token_id = 0
    mola_model.adapter_slot_id.return_value = None
    mola_model.adapter_slot_bindings.return_value = []
    mola_model.iter_slot_bound_lora_layers.return_value = iter(())
    mola_model.iter_routed_decode_lora_layers.return_value = iter(())
    config = overrides.pop("config", None)
    generator_factory = overrides.pop("generator_factory", None)
    routed_decode_session_factory = overrides.pop("routed_decode_session_factory", None)
    return MOLAEngine(
        mola_model,
        config,
        generator_factory=generator_factory,
        routed_decode_session_factory=routed_decode_session_factory,
    )


class TestEngineMetrics:
    def test_empty_snapshot(self):
        m = EngineMetrics()
        snap = m.snapshot()
        assert snap["queued_requests"] == 0
        assert snap["total_step_lock_wait_ms"] == 0
        assert snap["total_insert_lock_wait_ms"] == 0
        assert snap["completion_batch_size_limit"] == 0
        assert snap["prefill_batch_size_limit"] == 0
        assert snap["avg_ttft_ms"] == 0
        assert snap["avg_tps"] == 0
        assert snap["routed_decode_backend"] == "reference"
        assert snap["mixed_decode_migration_events"] == 0
        assert snap["mixed_decode_migrated_sequences"] == 0
        assert snap["mixed_decode_steps"] == 0
        assert snap["mixed_decode_rows"] == 0
        assert snap["avg_mixed_decode_rows"] == 0

    def test_record_completion(self):
        m = EngineMetrics()
        m.record_completion(ttft=0.1, tps=50.0)
        m.record_completion(ttft=0.2, tps=60.0)
        snap = m.snapshot()
        assert snap["requests_completed"] == 2
        assert snap["avg_ttft_ms"] == pytest.approx(150.0, abs=1)
        assert snap["avg_tps"] == pytest.approx(55.0, abs=1)

    def test_samples_capped_at_100(self):
        m = EngineMetrics()
        for _ in range(200):
            m.record_completion(ttft=0.01, tps=100.0)
        assert len(m._ttft_samples) == 100

    def test_snapshot_defaults(self):
        m = EngineMetrics(token_budget_limit=32768)
        snap = m.snapshot()
        assert snap["queued_requests"] == 0
        assert snap["active_generators"] == 0
        assert snap["active_sequences"] == 0
        assert snap["total_tokens_generated"] == 0
        assert snap["requests_completed"] == 0
        assert snap["requests_rejected"] == 0
        assert snap["inflight_tokens_reserved"] == 0
        assert snap["token_budget_limit"] == 32768
        assert snap["completion_batch_size_limit"] == 0
        assert snap["prefill_batch_size_limit"] == 0
        assert snap["total_step_lock_wait_ms"] == 0
        assert snap["total_insert_lock_wait_ms"] == 0
        assert snap["routed_decode_reference_enabled"] is False
        assert snap["routed_decode_backend"] == "reference"
        assert snap["mixed_decode_migration_events"] == 0
        assert snap["mixed_decode_migrated_sequences"] == 0
        assert snap["mixed_decode_steps"] == 0
        assert snap["mixed_decode_rows"] == 0
        assert snap["avg_mixed_decode_rows"] == 0
        assert snap["avg_ttft_ms"] == 0
        assert snap["avg_tps"] == 0

    def test_completion_records_samples(self):
        m = EngineMetrics()
        m.record_completion(ttft=0.1, tps=50.0)
        snap = m.snapshot()
        assert snap["requests_completed"] == 1
        assert snap["avg_ttft_ms"] == 100.0
        assert snap["avg_tps"] == 50.0

    def test_engine_config_marks_routed_decode_reference_in_metrics(self):
        engine = _make_engine(config=EngineConfig(enable_routed_decode_reference=True))
        assert engine.metrics.snapshot()["routed_decode_reference_enabled"] is True

    def test_engine_config_marks_strict_routed_decode_reference_in_metrics(self):
        engine = _make_engine(
            config=EngineConfig(
                enable_routed_decode_reference=True,
                strict_routed_decode_reference=True,
            )
        )
        snap = engine.metrics.snapshot()
        assert snap["routed_decode_reference_enabled"] is True
        assert snap["routed_decode_reference_strict"] is True

    def test_engine_config_marks_routed_decode_backend_in_metrics(self):
        engine = _make_engine(
            config=EngineConfig(routed_decode_backend="metal-kernel")
        )
        assert engine.metrics.snapshot()["routed_decode_backend"] == "metal-kernel"

    def test_engine_config_marks_batch_limits_in_metrics(self):
        engine = _make_engine(
            config=EngineConfig(max_batch_size=64, prefill_batch_size=12)
        )
        snap = engine.metrics.snapshot()
        assert snap["completion_batch_size_limit"] == 64
        assert snap["prefill_batch_size_limit"] == 12

    def test_engine_config_marks_gather_mm_routed_decode_backend_in_metrics(self):
        engine = _make_engine(
            config=EngineConfig(routed_decode_backend="gather-mm")
        )
        assert engine.metrics.snapshot()["routed_decode_backend"] == "gather-mm"

    def test_engine_config_marks_metal_gather_routed_decode_backend_in_metrics(self):
        engine = _make_engine(
            config=EngineConfig(routed_decode_backend="metal-gather")
        )
        assert engine.metrics.snapshot()["routed_decode_backend"] == "metal-gather"


class TestDefaultRoutedDecodeFactory:
    def test_default_routed_decode_factory_uses_reference_backend(self, monkeypatch):
        fake_factory = object()
        fake_module = SimpleNamespace(
            ReferenceRoutedLoRADeltaSessionFactory=lambda strict: fake_factory
        )
        monkeypatch.setitem(sys.modules, "mola.infrastructure.routed_decode", fake_module)
        engine = _make_engine(config=EngineConfig(routed_decode_backend="reference"))

        factory = engine._default_routed_decode_session_factory()

        assert factory is fake_factory

    def test_default_routed_decode_factory_uses_metal_backend(self, monkeypatch):
        fake_factory = object()
        fake_module = SimpleNamespace(
            MetalKernelRoutedLoRADeltaSessionFactory=lambda strict: fake_factory
        )
        monkeypatch.setitem(sys.modules, "mola.infrastructure.metal_routed_decode", fake_module)
        engine = _make_engine(config=EngineConfig(routed_decode_backend="metal-kernel"))

        factory = engine._default_routed_decode_session_factory()

        assert factory is fake_factory

    def test_default_routed_decode_factory_uses_gather_mm_backend(self, monkeypatch):
        fake_factory = object()
        fake_module = SimpleNamespace(
            GatherMMRoutedLoRADeltaSessionFactory=lambda strict: fake_factory
        )
        monkeypatch.setitem(sys.modules, "mola.infrastructure.gather_mm_routed_decode", fake_module)
        engine = _make_engine(config=EngineConfig(routed_decode_backend="gather-mm"))

        factory = engine._default_routed_decode_session_factory()

        assert factory is fake_factory

    def test_default_routed_decode_factory_uses_metal_gather_backend(self, monkeypatch):
        fake_factory = object()
        fake_module = SimpleNamespace(
            MetalGatherRoutedLoRADeltaSessionFactory=lambda strict: fake_factory
        )
        monkeypatch.setitem(
            sys.modules, "mola.infrastructure.metal_gather_routed_decode", fake_module
        )
        engine = _make_engine(config=EngineConfig(routed_decode_backend="metal-gather"))

        factory = engine._default_routed_decode_session_factory()

        assert factory is fake_factory

    def test_default_routed_decode_factory_rejects_unknown_backend(self):
        engine = _make_engine(config=EngineConfig(routed_decode_backend="wat"))

        with pytest.raises(ValueError, match="unsupported routed_decode_backend"):
            engine._default_routed_decode_session_factory()


class TestGetStopTokens:
    def test_single_eos(self):
        tok = MagicMock()
        tok.eos_token_id = 42
        assert _get_stop_tokens(tok) == {42}

    def test_list_eos(self):
        tok = MagicMock()
        tok.eos_token_id = [1, 2, 3]
        assert _get_stop_tokens(tok) == {1, 2, 3}

    def test_no_eos(self):
        tok = MagicMock(spec=[])
        assert _get_stop_tokens(tok) == set()


class TestGenerateRequest:
    def test_defaults(self):
        req = GenerateRequest([1, 2, 3], "rust", 100, None, asyncio.Queue())
        assert req.cancelled is False
        assert req.first_token_at is None
        assert req.token_count == 0

    def test_estimated_tokens(self):
        req = GenerateRequest([1, 2, 3], "rust", 7, None, asyncio.Queue())
        assert req.estimated_tokens == 10


class TestBackpressure:
    def test_submit_raises_on_full_queue(self):
        engine = _make_engine(config=EngineConfig(max_queued_requests=1))
        engine.submit(GenerateRequest([1], None, 10, None, asyncio.Queue()))
        with pytest.raises(queue.Full):
            engine.submit(GenerateRequest([2], None, 10, None, asyncio.Queue()))

    def test_submit_rejects_when_token_budget_is_exceeded(self):
        engine = _make_engine(config=EngineConfig(max_inflight_tokens=5))
        with pytest.raises(AdmissionRejected):
            engine.submit(GenerateRequest([1, 2, 3], None, 3, None, asyncio.Queue()))
        assert engine.metrics.requests_rejected == 1
        assert engine.metrics.inflight_tokens_reserved == 0


class TestCancellation:
    def test_cancel_pre_uid(self):
        """Cancel works before _drain_requests assigns a UID."""
        engine = _make_engine()
        q = asyncio.Queue()
        req = GenerateRequest([1], None, 10, None, q)
        engine.submit(req)
        engine.cancel(q)
        assert req.cancelled is True

    def test_cancel_post_uid(self):
        """Cancel works after the request has a UID."""
        engine = _make_engine()
        q = asyncio.Queue()
        req = GenerateRequest([1], None, 10, None, q)
        engine._uid_to_request[(None, 42)] = req
        engine.cancel(q)
        assert req.cancelled is True

    def test_cancel_unknown_queue_is_noop(self):
        engine = _make_engine()
        engine.cancel(asyncio.Queue())

    def test_cancel_pending_slot_request(self):
        engine = _make_engine()
        q = asyncio.Queue()
        req = GenerateRequest([1], "active", 10, None, q)
        slot = _AdapterSlot(generator=MagicMock(), adapter_id="active", active_uids={10})
        slot.pending_requests.append(req)
        engine._generators["active"] = slot

        engine.cancel(q)
        engine._process_cancelled()

        assert list(slot.pending_requests) == []
        assert engine.metrics.inflight_tokens_reserved == 0


class TestIdleCleanup:
    def test_removes_idle_and_calls_close(self):
        engine = _make_engine(config=EngineConfig(idle_timeout=0.0))
        mock_gen = MagicMock()
        engine._generators["old"] = _AdapterSlot(
            generator=mock_gen, adapter_id="old", last_active=time.time() - 100
        )
        engine._cleanup_idle()
        assert "old" not in engine._generators
        mock_gen.close.assert_called_once()

    def test_keeps_active_slots(self):
        engine = _make_engine(config=EngineConfig(idle_timeout=60.0))
        mock_gen = MagicMock()
        engine._generators["active"] = _AdapterSlot(
            generator=mock_gen, adapter_id="active", active_uids={1, 2}
        )
        engine._cleanup_idle()
        assert "active" in engine._generators
        mock_gen.close.assert_not_called()

    def test_keeps_slots_with_pending_requests(self):
        engine = _make_engine(config=EngineConfig(idle_timeout=0.0))
        mock_gen = MagicMock()
        slot = _AdapterSlot(generator=MagicMock(), adapter_id="active")
        slot.pending_requests.append(GenerateRequest([1], "active", 10, None, asyncio.Queue()))
        slot.last_active = time.time() - 100
        engine._generators["active"] = slot
        engine._cleanup_idle()
        assert "active" in engine._generators
        mock_gen.close.assert_not_called()


class TestAdapterSlotResolution:
    class _FakeLayer:
        def __init__(self, bindings):
            self._bindings = bindings

        @property
        def slot_ids(self):
            return tuple(slot_id for slot_id, *_ in self._bindings)

        def slot_bindings(self):
            return list(self._bindings)

    def test_slot_id_uses_model_resolver(self):
        engine = _make_engine()
        engine.mola_model.adapter_slot_id.return_value = 7
        assert engine._adapter_slot_id("rust") == 7

    def test_slot_id_rejects_non_int_mock_values(self):
        engine = _make_engine()
        engine.mola_model.adapter_slot_id.return_value = MagicMock()
        assert engine._adapter_slot_id("rust") is None

    def test_runtime_slot_layout_tracks_loaded_active_and_pending_slots(self):
        engine = _make_engine()
        engine.mola_model.adapter_slot_bindings.return_value = [
            MagicMock(slot_id=1),
            MagicMock(slot_id=2),
            MagicMock(slot_id=3),
        ]

        rust_slot = _AdapterSlot(generator=MagicMock(), adapter_id="rust", active_uids={10})
        sql_slot = _AdapterSlot(generator=MagicMock(), adapter_id="sql")
        sql_slot.pending_requests.append(GenerateRequest([1], "sql", 10, None, asyncio.Queue()))
        engine._generators = {"rust": rust_slot, "sql": sql_slot, None: _AdapterSlot(generator=MagicMock(), adapter_id=None, active_uids={1})}

        def slot_id(adapter_id):
            return {"rust": 1, "sql": 2}.get(adapter_id)

        engine.mola_model.adapter_slot_id.side_effect = slot_id

        assert engine.runtime_slot_layout() == RuntimeSlotLayout(
            loaded_slot_ids=(1, 2, 3),
            active_slot_ids=(1,),
            pending_slot_ids=(2,),
        )

    def test_active_slot_bindings_filters_loaded_bindings_by_runtime_state(self):
        engine = _make_engine()
        engine.mola_model.adapter_slot_bindings.return_value = [
            AdapterSlotBinding("rust", 1, 8, 16.0, 1, ("q_proj",), "/fake/rust"),
            AdapterSlotBinding("sql", 2, 8, 16.0, 1, ("q_proj",), "/fake/sql"),
            AdapterSlotBinding("legal", 3, 8, 16.0, 1, ("q_proj",), "/fake/legal"),
        ]

        rust_slot = _AdapterSlot(generator=MagicMock(), adapter_id="rust", active_uids={10})
        sql_slot = _AdapterSlot(generator=MagicMock(), adapter_id="sql")
        sql_slot.pending_requests.append(GenerateRequest([1], "sql", 10, None, asyncio.Queue()))
        engine._generators = {"rust": rust_slot, "sql": sql_slot}

        engine.mola_model.adapter_slot_id.side_effect = lambda adapter_id: {"rust": 1, "sql": 2}.get(adapter_id)

        assert engine.active_slot_bindings() == (
            AdapterSlotBinding("rust", 1, 8, 16.0, 1, ("q_proj",), "/fake/rust"),
            AdapterSlotBinding("sql", 2, 8, 16.0, 1, ("q_proj",), "/fake/sql"),
        )

    def test_active_slot_bindings_are_sorted(self):
        engine = _make_engine()
        engine.mola_model.adapter_slot_bindings.return_value = [
            AdapterSlotBinding("sql", 2, 8, 16.0, 1, ("q_proj",), "/fake/sql"),
            AdapterSlotBinding("rust", 1, 8, 16.0, 1, ("q_proj",), "/fake/rust"),
        ]
        engine.mola_model.adapter_slot_id.side_effect = lambda adapter_id: {
            "sql": 2,
            "rust": 1,
        }.get(adapter_id)
        engine._generators["sql"] = _AdapterSlot(
            generator=MagicMock(),
            adapter_id="sql",
            pending_requests=deque([GenerateRequest([1], "sql", 8, None, asyncio.Queue())]),
        )
        engine._generators["rust"] = _AdapterSlot(
            generator=MagicMock(),
            adapter_id="rust",
            active_uids={1, 2},
        )

        assert engine.active_slot_bindings() == (
            AdapterSlotBinding("rust", 1, 8, 16.0, 1, ("q_proj",), "/fake/rust"),
            AdapterSlotBinding("sql", 2, 8, 16.0, 1, ("q_proj",), "/fake/sql"),
        )

    def test_decode_active_slot_bindings_use_only_decode_rows(self):
        engine = _make_engine()
        engine.mola_model.adapter_slot_bindings.return_value = [
            AdapterSlotBinding("rust", 1, 8, 16.0, 1, ("q_proj",), "/fake/rust"),
            AdapterSlotBinding("sql", 2, 8, 16.0, 1, ("q_proj",), "/fake/sql"),
            AdapterSlotBinding("legal", 3, 8, 16.0, 1, ("q_proj",), "/fake/legal"),
        ]
        rust_slot = _AdapterSlot(generator=MagicMock(), adapter_id="rust", active_uids={1})
        rust_slot.generator.active_handles.return_value = (MagicMock(uid=21),)
        sql_slot = _AdapterSlot(generator=MagicMock(), adapter_id="sql")
        sql_slot.pending_requests.append(GenerateRequest([1], "sql", 10, None, asyncio.Queue()))
        engine._generators = {"rust": rust_slot, "sql": sql_slot}
        engine.mola_model.adapter_slot_id.side_effect = lambda adapter_id: {"rust": 1, "sql": 2}.get(adapter_id)

        assert engine.decode_active_slot_bindings() == (
            AdapterSlotBinding("rust", 1, 8, 16.0, 1, ("q_proj",), "/fake/rust"),
        )

    def test_layer_slot_pack_views_builds_from_runtime_and_model(self):
        engine = _make_engine()
        engine.mola_model.adapter_slot_bindings.return_value = [
            AdapterSlotBinding("rust", 1, 8, 16.0, 1, ("q_proj",), "/fake/rust"),
            AdapterSlotBinding("sql", 2, 8, 16.0, 1, ("q_proj",), "/fake/sql"),
            AdapterSlotBinding("legal", 3, 8, 16.0, 1, ("q_proj",), "/fake/legal"),
        ]
        engine.mola_model.iter_slot_bound_lora_layers.return_value = iter([
            (
                "layers.0.q_proj",
                self._FakeLayer([
                    (3, "a-legal", "b-legal", 16.0),
                    (1, "a-rust", "b-rust", 16.0),
                    (2, "a-sql", "b-sql", 16.0),
                ]),
            ),
        ])

        rust_slot = _AdapterSlot(generator=MagicMock(), adapter_id="rust", active_uids={10})
        sql_slot = _AdapterSlot(generator=MagicMock(), adapter_id="sql", active_uids={10})
        sql_slot.pending_requests.append(GenerateRequest([1], "sql", 10, None, asyncio.Queue()))
        engine._generators = {"rust": rust_slot, "sql": sql_slot}
        engine.mola_model.adapter_slot_id.side_effect = lambda adapter_id: {"rust": 1, "sql": 2}.get(adapter_id)

        views = engine.layer_slot_pack_views()

        assert len(views) == 1
        assert views[0].layer_name == "layers.0.q_proj"
        assert views[0].slot_ids == (1, 2)
        assert tuple(entry.adapter_name for entry in views[0].entries) == ("rust", "sql")

    def test_materialize_layer_slot_packs_uses_runtime_state(self):
        engine = _make_engine()
        engine.mola_model.adapter_slot_bindings.return_value = [
            AdapterSlotBinding("rust", 1, 8, 16.0, 1, ("q_proj",), "/fake/rust"),
            AdapterSlotBinding("sql", 2, 8, 16.0, 1, ("q_proj",), "/fake/sql"),
        ]
        engine.mola_model.iter_slot_bound_lora_layers.return_value = iter([
            (
                "layers.0.q_proj",
                self._FakeLayer([
                    (2, "a-sql", "b-sql", 20.0),
                    (1, "a-rust", "b-rust", 16.0),
                ]),
            ),
        ])
        engine._generators = {
            "rust": _AdapterSlot(generator=MagicMock(), adapter_id="rust", active_uids={1}),
            "sql": _AdapterSlot(generator=MagicMock(), adapter_id="sql", pending_requests=deque([GenerateRequest([1], "sql", 10, None, asyncio.Queue())])),
        }
        engine.mola_model.adapter_slot_id.side_effect = lambda adapter_id: {"rust": 1, "sql": 2}.get(adapter_id)

        packs = engine.materialize_layer_slot_packs(stack_fn=lambda values: tuple(values))

        assert len(packs) == 1
        assert packs[0].layer_name == "layers.0.q_proj"
        assert packs[0].slot_ids == (1, 2)
        assert packs[0].adapter_names == ("rust", "sql")
        assert packs[0].lora_a == ("a-rust", "a-sql")

    def test_materialize_layer_slot_packs_can_materialize_scales(self):
        engine = _make_engine()
        engine.mola_model.adapter_slot_bindings.return_value = [
            AdapterSlotBinding("rust", 1, 8, 16.0, 1, ("q_proj",), "/fake/rust"),
            AdapterSlotBinding("sql", 2, 8, 16.0, 1, ("q_proj",), "/fake/sql"),
        ]
        engine.mola_model.iter_slot_bound_lora_layers.return_value = iter([
            (
                "layers.0.q_proj",
                self._FakeLayer([
                    (2, "a-sql", "b-sql", 20.0),
                    (1, "a-rust", "b-rust", 16.0),
                ]),
            ),
        ])
        engine._generators = {
            "rust": _AdapterSlot(generator=MagicMock(), adapter_id="rust", active_uids={1}),
            "sql": _AdapterSlot(generator=MagicMock(), adapter_id="sql", pending_requests=deque([GenerateRequest([1], "sql", 10, None, asyncio.Queue())])),
        }
        engine.mola_model.adapter_slot_id.side_effect = lambda adapter_id: {"rust": 1, "sql": 2}.get(adapter_id)

        packs = engine.materialize_layer_slot_packs(
            stack_fn=lambda values: tuple(values),
            scale_fn=lambda values: tuple(value * 2 for value in values),
        )

        assert packs[0].scales == (32.0, 40.0)

    def test_materialize_layer_slot_pack_state_indexes_by_layer_name(self):
        engine = _make_engine()
        engine.mola_model.adapter_slot_bindings.return_value = [
            AdapterSlotBinding("rust", 1, 8, 16.0, 1, ("q_proj",), "/fake/rust"),
            AdapterSlotBinding("sql", 2, 8, 16.0, 1, ("q_proj",), "/fake/sql"),
        ]
        engine.mola_model.iter_routed_decode_lora_layers.return_value = iter([
            (
                "layers.0.q_proj",
                self._FakeLayer([
                    (2, "a-sql", "b-sql", 20.0),
                    (1, "a-rust", "b-rust", 16.0),
                ]),
            ),
        ])
        engine._generators = {
            "rust": _AdapterSlot(generator=MagicMock(), adapter_id="rust", active_uids={1}),
            "sql": _AdapterSlot(generator=MagicMock(), adapter_id="sql", pending_requests=deque([GenerateRequest([1], "sql", 10, None, asyncio.Queue())])),
        }
        engine._generators["rust"].generator.active_handles.return_value = (MagicMock(uid=21),)
        engine._ordered_slots = lambda: [engine._generators["rust"], engine._generators["sql"]]
        engine.mola_model.adapter_slot_id.side_effect = lambda adapter_id: {"rust": 1, "sql": 2}.get(adapter_id)

        state = engine.materialize_layer_slot_pack_state(
            stack_fn=lambda values: tuple(values),
            scale_fn=lambda values: tuple(values),
        )

        pack = state.get("layers.0.q_proj")
        assert pack is not None
        assert pack.adapter_names == ("rust",)
        assert pack.slot_ids == (1,)
        assert state.get("layers.9.q_proj") is None

    def test_build_routed_decode_session_filters_pack_state_to_explicit_slot_ids(self):
        engine = _make_engine(routed_decode_session_factory=_tuple_routed_session_factory())
        engine.mola_model.adapter_slot_bindings.return_value = [
            AdapterSlotBinding("rust", 1, 8, 16.0, 1, ("q_proj",), "/fake/rust"),
            AdapterSlotBinding("sql", 2, 8, 16.0, 1, ("q_proj",), "/fake/sql"),
        ]
        engine.mola_model.iter_routed_decode_lora_layers.return_value = iter([
            (
                "layers.0.q_proj",
                self._FakeLayer([
                    (2, "a-sql", "b-sql", 20.0),
                    (1, "a-rust", "b-rust", 16.0),
                ]),
            ),
        ])

        state, token_slot_ids = engine.build_routed_decode_session(
            (2, 2, 2),
        )

        pack = state.get("layers.0.q_proj")
        assert token_slot_ids == (2, 2, 2)
        assert pack is not None
        assert pack.adapter_names == ("sql",)
        assert pack.slot_ids == (2,)

    def test_build_routed_decode_session_rejects_unknown_explicit_slot_ids(self):
        engine = _make_engine(routed_decode_session_factory=_tuple_routed_session_factory())
        engine.mola_model.adapter_slot_bindings.return_value = [
            AdapterSlotBinding("rust", 1, 8, 16.0, 1, ("q_proj",), "/fake/rust"),
        ]
        engine.mola_model.iter_routed_decode_lora_layers.return_value = iter([
            ("layers.0.q_proj", self._FakeLayer([(1, "a-rust", "b-rust", 16.0)])),
        ])

        with pytest.raises(ValueError, match="missing adapter slot bindings"):
            engine.build_routed_decode_session(
                (1, 2),
            )

    def test_materialize_layer_slot_pack_state_uses_routed_layer_iterator(self):
        engine = _make_engine()
        engine.mola_model.adapter_slot_bindings.return_value = [
            AdapterSlotBinding("rust", 1, 8, 16.0, 1, ("q_proj",), "/fake/rust"),
            AdapterSlotBinding("sql", 2, 8, 16.0, 1, ("q_proj",), "/fake/sql"),
        ]
        engine.mola_model.iter_slot_bound_lora_layers.return_value = iter([
            ("layers.9.q_proj", self._FakeLayer([(1, "a-other", "b-other", 16.0)])),
        ])
        engine.mola_model.iter_routed_decode_lora_layers.return_value = iter([
            ("layers.0.q_proj", self._FakeLayer([(1, "a-rust", "b-rust", 16.0), (2, "a-sql", "b-sql", 20.0)])),
        ])
        engine._generators = {
            "rust": _AdapterSlot(generator=MagicMock(), adapter_id="rust", active_uids={1}),
            "sql": _AdapterSlot(generator=MagicMock(), adapter_id="sql", pending_requests=deque([GenerateRequest([1], "sql", 10, None, asyncio.Queue())])),
        }
        engine._generators["rust"].generator.active_handles.return_value = (MagicMock(uid=21),)
        engine._ordered_slots = lambda: [engine._generators["rust"], engine._generators["sql"]]
        engine.mola_model.adapter_slot_id.side_effect = lambda adapter_id: {"rust": 1}.get(adapter_id)

        state = engine.materialize_layer_slot_pack_state(
            stack_fn=lambda values: tuple(values),
            scale_fn=lambda values: tuple(values),
        )

        assert state.get("layers.0.q_proj") is not None
        assert state.get("layers.0.q_proj").slot_ids == (1,)
        assert state.get("layers.9.q_proj") is None

    def test_build_routed_decode_session_preserves_token_slot_ids(self):
        engine = _make_engine(routed_decode_session_factory=_tuple_routed_session_factory())
        engine.mola_model.adapter_slot_bindings.return_value = [
            AdapterSlotBinding("rust", 1, 8, 16.0, 1, ("q_proj",), "/fake/rust"),
        ]
        engine.mola_model.iter_routed_decode_lora_layers.return_value = iter([
            (
                "layers.0.q_proj",
                self._FakeLayer([
                    (1, "a-rust", "b-rust", 16.0),
                ]),
            ),
        ])
        engine._generators = {
            "rust": _AdapterSlot(generator=MagicMock(), adapter_id="rust", active_uids={1}),
        }
        engine.mola_model.adapter_slot_id.side_effect = lambda adapter_id: {"rust": 1}.get(adapter_id)

        state, token_slot_ids = engine.build_routed_decode_session(
            (1, 1, 1),
        )

        assert token_slot_ids == (1, 1, 1)
        assert state.get("layers.0.q_proj") is not None

    def test_build_homogeneous_decode_session_repeats_runtime_slot_id(self):
        engine = _make_engine(routed_decode_session_factory=_tuple_routed_session_factory())
        engine.mola_model.adapter_slot_bindings.return_value = [
            AdapterSlotBinding("rust", 1, 8, 16.0, 1, ("q_proj",), "/fake/rust"),
        ]
        engine.mola_model.iter_routed_decode_lora_layers.return_value = iter([
            (
                "layers.0.q_proj",
                self._FakeLayer([
                    (1, "a-rust", "b-rust", 16.0),
                ]),
            ),
        ])
        engine._generators = {
            "rust": _AdapterSlot(generator=MagicMock(), adapter_id="rust", active_uids={1}),
        }
        engine.mola_model.adapter_slot_id.side_effect = lambda adapter_id: {"rust": 1}.get(adapter_id)

        state, token_slot_ids = engine.build_homogeneous_decode_session(
            "rust",
            3,
        )

        assert token_slot_ids == (1, 1, 1)
        assert state.get("layers.0.q_proj") is not None

    def test_build_homogeneous_decode_routed_session_for_slot_uses_active_handle_order(self):
        engine = _make_engine(routed_decode_session_factory=_tuple_routed_session_factory())
        engine.mola_model.adapter_slot_bindings.return_value = [
            AdapterSlotBinding("rust", 1, 8, 16.0, 1, ("q_proj",), "/fake/rust"),
        ]
        engine.mola_model.iter_routed_decode_lora_layers.return_value = iter([
            (
                "layers.0.q_proj",
                self._FakeLayer([
                    (1, "a-rust", "b-rust", 16.0),
                ]),
            ),
        ])
        slot = _AdapterSlot(generator=MagicMock(), adapter_id="rust", active_uids={1, 2, 3})
        slot.generator.active_handles.return_value = (
            MagicMock(uid=11),
            MagicMock(uid=12),
            MagicMock(uid=13),
        )
        engine._generators = {"rust": slot}
        engine.mola_model.adapter_slot_id.side_effect = lambda adapter_id: {"rust": 1}.get(adapter_id)

        state, token_slot_ids = engine._build_homogeneous_decode_routed_session_for_slot(
            slot,
        )

        assert token_slot_ids == (1, 1, 1)
        assert state is not None
        assert state.get("layers.0.q_proj") is not None

    def test_build_decode_routed_session_for_slot_skips_base_slot(self):
        engine = _make_engine(routed_decode_session_factory=_tuple_routed_session_factory())
        slot = _AdapterSlot(generator=MagicMock(), adapter_id=None, active_uids={1, 2})
        slot.generator.active_handles.return_value = (MagicMock(uid=1), MagicMock(uid=2))

        session = engine._build_homogeneous_decode_routed_session_for_slot(
            slot,
        )

        assert session is None

    def test_build_decode_routed_session_for_slot_skips_empty_generator_batch(self):
        engine = _make_engine(routed_decode_session_factory=_tuple_routed_session_factory())
        slot = _AdapterSlot(generator=MagicMock(), adapter_id="rust", active_uids={1, 2})
        slot.generator.active_handles.return_value = ()

        session = engine._build_homogeneous_decode_routed_session_for_slot(
            slot,
        )

        assert session is None

    def test_maybe_build_decode_routed_session_skips_prefill_stage(self):
        engine = _make_engine(
            config=EngineConfig(enable_routed_decode_reference=True),
            routed_decode_session_factory=_tuple_routed_session_factory(),
        )
        engine.mola_model.adapter_slot_bindings.return_value = [
            AdapterSlotBinding("rust", 1, 8, 16.0, 1, ("q_proj",), "/fake/rust"),
        ]
        engine.mola_model.iter_routed_decode_lora_layers.return_value = iter([
            (
                "layers.0.q_proj",
                self._FakeLayer([
                    (1, "a-rust", "b-rust", 16.0),
                ]),
            ),
        ])
        engine.mola_model.adapter_slot_id.side_effect = lambda adapter_id: {"rust": 1}.get(adapter_id)
        slot = _AdapterSlot(generator=MagicMock(), adapter_id="rust", active_uids={7})
        slot.generator.active_handles.return_value = (MagicMock(uid=7),)
        engine._uid_to_request[("rust", 7)] = GenerateRequest([1, 2, 3], "rust", 16, None, asyncio.Queue())

        session = engine._maybe_build_homogeneous_decode_routed_session_for_slot_locked(slot)

        assert session is None

    def test_maybe_build_decode_routed_session_allows_decode_stage(self):
        engine = _make_engine(
            config=EngineConfig(enable_routed_decode_reference=True),
            routed_decode_session_factory=_tuple_routed_session_factory(),
        )
        engine.mola_model.adapter_slot_bindings.return_value = [
            AdapterSlotBinding("rust", 1, 8, 16.0, 1, ("q_proj",), "/fake/rust"),
        ]
        engine.mola_model.iter_routed_decode_lora_layers.return_value = iter([
            (
                "layers.0.q_proj",
                self._FakeLayer([
                    (1, "a-rust", "b-rust", 16.0),
                ]),
            ),
        ])
        engine.mola_model.adapter_slot_id.side_effect = lambda adapter_id: {"rust": 1}.get(adapter_id)
        slot = _AdapterSlot(generator=MagicMock(), adapter_id="rust", active_uids={7})
        slot.generator.active_handles.return_value = (MagicMock(uid=7),)
        req = GenerateRequest([1, 2, 3], "rust", 16, None, asyncio.Queue())
        req.first_token_at = time.time()
        engine._uid_to_request[("rust", 7)] = req

        state, token_slot_ids = engine._maybe_build_homogeneous_decode_routed_session_for_slot_locked(slot)

        assert token_slot_ids == (1,)
        assert state.get("layers.0.q_proj") is not None

    def test_decode_row_bindings_follow_scheduler_and_generator_order(self):
        engine = _make_engine()
        rust_slot = _AdapterSlot(generator=MagicMock(), adapter_id="rust", active_uids={1, 2})
        rust_slot.generator.active_handles.return_value = (MagicMock(uid=21), MagicMock(uid=22))
        sql_slot = _AdapterSlot(generator=MagicMock(), adapter_id="sql", active_uids={3})
        sql_slot.generator.active_handles.return_value = (MagicMock(uid=31),)
        engine._ordered_slots = lambda: [sql_slot, rust_slot]
        engine.mola_model.adapter_slot_id.side_effect = lambda adapter_id: {"rust": 1, "sql": 2}.get(adapter_id)

        bindings = engine.decode_row_bindings()

        assert bindings == (
            DecodeRowBinding("sql", 2, 31),
            DecodeRowBinding("rust", 1, 21),
            DecodeRowBinding("rust", 1, 22),
        )

    def test_build_active_decode_session_uses_decode_row_bindings(self):
        engine = _make_engine(routed_decode_session_factory=_tuple_routed_session_factory())
        engine.mola_model.adapter_slot_bindings.return_value = [
            AdapterSlotBinding("rust", 1, 8, 16.0, 1, ("q_proj",), "/fake/rust"),
            AdapterSlotBinding("sql", 2, 8, 16.0, 1, ("q_proj",), "/fake/sql"),
        ]
        engine.mola_model.iter_routed_decode_lora_layers.return_value = iter([
            (
                "layers.0.q_proj",
                self._FakeLayer([
                    (2, "a-sql", "b-sql", 20.0),
                    (1, "a-rust", "b-rust", 16.0),
                ]),
            ),
        ])
        rust_slot = _AdapterSlot(generator=MagicMock(), adapter_id="rust", active_uids={1, 2})
        rust_slot.generator.active_handles.return_value = (MagicMock(uid=21), MagicMock(uid=22))
        sql_slot = _AdapterSlot(generator=MagicMock(), adapter_id="sql", active_uids={3})
        sql_slot.generator.active_handles.return_value = (MagicMock(uid=31),)
        engine._generators = {"sql": sql_slot, "rust": rust_slot}
        engine._ordered_slots = lambda: [sql_slot, rust_slot]
        engine.mola_model.adapter_slot_id.side_effect = lambda adapter_id: {"rust": 1, "sql": 2}.get(adapter_id)

        state, token_slot_ids = engine.build_active_decode_session(
        )

        assert token_slot_ids == (2, 1, 1)
        assert state.get("layers.0.q_proj") is not None

    def test_step_slot_builds_homogeneous_routed_session_under_model_lock(self):
        engine = _make_engine(config=EngineConfig(enable_routed_decode_reference=True))
        slot = _AdapterSlot(generator=MagicMock(), adapter_id="rust", active_uids={1})
        slot.generator.step.return_value = []

        class FakeLock:
            def __init__(self):
                self.held = False

            def __enter__(self):
                self.held = True
                return self

            def __exit__(self, exc_type, exc, tb):
                self.held = False
                return False

        fake_lock = FakeLock()
        engine.model_lock = fake_lock
        calls = []

        def fake_builder(target_slot):
            calls.append(target_slot)
            assert fake_lock.held is True
            return None

        engine._maybe_build_homogeneous_decode_routed_session_for_slot_locked = fake_builder

        engine._step_slot(slot)

        assert calls == [slot]

    def test_step_slot_passes_routed_session_into_context(self, monkeypatch):
        session = object()
        engine = _make_engine(config=EngineConfig(enable_routed_decode_reference=True))
        slot = _AdapterSlot(generator=MagicMock(), adapter_id="rust", active_uids={1})
        slot.generator.step.return_value = []
        engine._maybe_build_homogeneous_decode_routed_session_for_slot_locked = lambda target_slot: session
        seen = []

        from contextlib import contextmanager

        @contextmanager
        def fake_routed_context(active_session):
            seen.append(active_session)
            yield

        monkeypatch.setattr("mola.engine.routed_decode_context", fake_routed_context)

        engine._step_slot(slot)

        assert seen == [session]


class TestModelLock:
    def test_is_threading_lock(self):
        engine = _make_engine()
        assert isinstance(engine.model_lock, threading.Lock)

    def test_acquirable(self):
        engine = _make_engine()
        with engine.model_lock:
            pass


class TestStop:
    def test_clears_state_and_closes_generators(self):
        engine = _make_engine()
        mock_gen = MagicMock()
        slot = _AdapterSlot(generator=mock_gen, adapter_id="test")
        slot.pending_requests.append(GenerateRequest([2], "test", 10, None, asyncio.Queue()))
        engine._generators["test"] = slot
        req = GenerateRequest(
            [1], "test", 10, None, asyncio.Queue()
        )
        engine._uid_to_request[("test", 0)] = req
        engine.metrics.inflight_tokens_reserved = req.estimated_tokens + slot.pending_requests[0].estimated_tokens
        engine._reserved_tokens = engine.metrics.inflight_tokens_reserved
        engine.stop()
        assert len(engine._generators) == 0
        assert len(engine._uid_to_request) == 0
        assert engine.metrics.inflight_tokens_reserved == 0
        mock_gen.close.assert_called_once()


class TestPendingTracking:
    def test_submit_registers_pending(self):
        engine = _make_engine()
        q = asyncio.Queue()
        engine.submit(GenerateRequest([1], None, 10, None, q))
        assert id(q) in engine._pending_by_queue

    def test_drain_removes_pending(self):
        engine = _make_engine()
        q = asyncio.Queue()
        engine.submit(GenerateRequest([1], None, 10, None, q))

        mock_gen = MagicMock()
        engine._generators[None] = _AdapterSlot(generator=mock_gen, adapter_id=None)

        engine._drain_requests()
        assert id(q) not in engine._pending_by_queue
        assert len(engine._generators[None].pending_requests) == 1
        assert engine.metrics.queued_requests == 1

    def test_insert_pending_batches_same_adapter_requests(self):
        engine = _make_engine(config=EngineConfig(prefill_batch_size=2))
        q1 = asyncio.Queue()
        q2 = asyncio.Queue()
        engine.submit(GenerateRequest([1], "rust", 10, None, q1))
        engine.submit(GenerateRequest([2], "rust", 12, None, q2))

        mock_gen = MagicMock()
        mock_gen.submit_batch.return_value = [MagicMock(uid=0), MagicMock(uid=1)]
        slot = _AdapterSlot(generator=mock_gen, adapter_id="rust")
        engine._generators["rust"] = slot

        engine._drain_requests()
        inserted = engine._insert_pending(slot)

        assert inserted is True
        assert slot.active_uids == {0, 1}
        assert list(slot.pending_requests) == []
        assert ("rust", 0) in engine._uid_to_request
        assert ("rust", 1) in engine._uid_to_request
        mock_gen.submit_batch.assert_called_once()

    def test_insert_pending_records_insert_metrics(self):
        engine = _make_engine(config=EngineConfig(prefill_batch_size=1))
        q = asyncio.Queue()
        req = GenerateRequest([1], "rust", 10, None, q)
        mock_gen = MagicMock()
        mock_gen.submit_batch.return_value = [MagicMock(uid=0)]
        slot = _AdapterSlot(generator=mock_gen, adapter_id="rust")
        slot.pending_requests.append(req)
        engine._generators["rust"] = slot

        inserted = engine._insert_pending(slot)

        assert inserted is True
        snap = slot.slot_metrics.snapshot()
        assert snap["insert_calls"] == 1
        assert snap["inserted_requests"] == 1
        assert snap["avg_insert_ms"] >= 0
        assert snap["avg_insert_lock_wait_ms"] >= 0


class TestProfilingMetrics:
    def test_slot_snapshot_includes_lock_wait_metrics(self):
        engine = _make_engine()
        slot = _AdapterSlot(generator=MagicMock(), adapter_id="rust")
        slot.slot_metrics.record_insert(9.5, 1.5, 2)
        slot.slot_metrics.record_step(4.0, 0.5, 3)
        engine._generators["rust"] = slot

        snap = engine.slot_snapshots()["rust"]

        assert snap["insert_calls"] == 1
        assert snap["inserted_requests"] == 2
        assert snap["avg_insert_lock_wait_ms"] == pytest.approx(1.5, abs=0.01)
        assert snap["avg_step_lock_wait_ms"] == pytest.approx(0.5, abs=0.01)

    def test_step_slot_records_lock_wait_metrics(self):
        engine = _make_engine()
        req = GenerateRequest([1], "rust", 10, None, asyncio.Queue())
        engine._uid_to_request[("rust", 0)] = req
        engine._send_to_queue = lambda *_args, **_kwargs: True

        response = MagicMock()
        response.handle.uid = 0
        response.token = 7
        response.finish_reason = None

        mock_gen = MagicMock()
        mock_gen.step.return_value = [response]
        slot = _AdapterSlot(generator=mock_gen, adapter_id="rust", active_uids={0})

        engine._step_slot(slot)

        snap = slot.slot_metrics.snapshot()
        assert snap["steps"] == 1
        assert snap["avg_step_lock_wait_ms"] >= 0
        assert engine.metrics.snapshot()["total_step_lock_wait_ms"] >= 0


class TestUidNamespacing:
    def test_dispatch_routes_duplicate_uids_to_correct_queues(self):
        engine = _make_engine()
        rust_req = GenerateRequest([1], "rust", 10, None, asyncio.Queue())
        sql_req = GenerateRequest([2], "sql", 10, None, asyncio.Queue())
        engine._uid_to_request[("rust", 0)] = rust_req
        engine._uid_to_request[("sql", 0)] = sql_req
        engine._send_to_queue = lambda req, data: (req.response_queue.put_nowait(data), True)[1]

        engine._dispatch_token("rust", 0, 11, None)
        engine._dispatch_token("sql", 0, 22, None)

        assert rust_req.response_queue.get_nowait()["token"] == 11
        assert sql_req.response_queue.get_nowait()["token"] == 22

    def test_dispatch_routes_using_generator_key_not_adapter_id(self):
        engine = _make_engine()
        req = GenerateRequest([1], "rust", 10, None, asyncio.Queue())
        engine._uid_to_request[("adapted", 0)] = req
        engine._send_to_queue = lambda target_req, data: (
            target_req.response_queue.put_nowait(data),
            True,
        )[1]

        engine._dispatch_token("adapted", 0, 11, None)

        assert req.response_queue.get_nowait()["token"] == 11

    def test_process_cancelled_uses_generator_key_to_find_slot(self):
        engine = _make_engine()
        req = GenerateRequest([1], "rust", 10, None, asyncio.Queue())
        req.cancelled = True
        slot = _AdapterSlot(
            generator=MagicMock(),
            adapter_id="rust",
            generator_key="adapted",
            active_uids={0},
        )
        engine._generators["rust"] = slot
        engine._uid_to_request[("adapted", 0)] = req

        engine._process_cancelled()

        slot.generator.cancel.assert_called_once_with([GeneratorHandle(uid=0)])
        assert ("adapted", 0) not in engine._uid_to_request


class TestMixedDecodeMigration:
    def test_migrate_decode_ready_request_moves_state_to_shared_slot(self):
        engine = _make_engine(
            config=EngineConfig(
                enable_routed_decode_reference=True,
                enable_mixed_decode_migration=True,
            )
        )
        req = GenerateRequest([1], "rust", 10, None, asyncio.Queue())
        req.first_token_at = time.time()
        source_generator = MagicMock()
        source_generator.active_handles.return_value = (GeneratorHandle(uid=7),)
        source_generator.take_states.return_value = [
            GeneratorState(
                handle=GeneratorHandle(uid=7),
                next_token=101,
                logprobs="lp",
                max_tokens=10,
                num_tokens=1,
                cache=["cache"],
                sampler=None,
                logits_processors=[],
                tokens=[1, 2, 3],
            )
        ]
        source_slot = _AdapterSlot(
            generator=source_generator,
            adapter_id="rust",
            active_uids={7},
        )
        shared_generator = MagicMock()
        shared_generator.restore_states.return_value = [GeneratorHandle(uid=70)]
        shared_slot = _AdapterSlot(
            generator=shared_generator,
            adapter_id=MIXED_DECODE_ADAPTER_ID,
            generator_key=MIXED_DECODE_ADAPTER_ID,
            active_uids={99},
        )
        engine._uid_to_request[("rust", 7)] = req
        engine._generators[MIXED_DECODE_ADAPTER_ID] = shared_slot
        engine._get_or_create_mixed_decode_slot = lambda: shared_slot

        engine._migrate_decode_ready_from_slot(source_slot)

        source_generator.take_states.assert_called_once()
        shared_generator.restore_states.assert_called_once()
        assert source_slot.active_uids == set()
        assert shared_slot.active_uids == {70, 99}
        assert req.uid == 70
        assert ("rust", 7) not in engine._uid_to_request
        assert (MIXED_DECODE_ADAPTER_ID, 70) in engine._uid_to_request
        snap = engine.metrics.snapshot()
        assert snap["mixed_decode_migration_events"] == 1
        assert snap["mixed_decode_migrated_sequences"] == 1

    def test_does_not_migrate_single_decode_ready_slot_without_mixed_opportunity(self):
        engine = _make_engine(
            config=EngineConfig(
                enable_routed_decode_reference=True,
                enable_mixed_decode_migration=True,
            )
        )
        req = GenerateRequest([1], "rust", 10, None, asyncio.Queue())
        req.first_token_at = time.time()
        source_generator = MagicMock()
        source_generator.active_handles.return_value = (GeneratorHandle(uid=7),)
        source_slot = _AdapterSlot(
            generator=source_generator,
            adapter_id="rust",
            active_uids={7},
        )
        engine._uid_to_request[("rust", 7)] = req
        engine._generators["rust"] = source_slot

        engine._migrate_decode_ready_from_slot(source_slot)

        source_generator.take_states.assert_not_called()
        assert source_slot.active_uids == {7}
        assert ("rust", 7) in engine._uid_to_request

    def test_restore_failure_rebinds_request_back_to_source_generator(self):
        engine = _make_engine(
            config=EngineConfig(
                enable_routed_decode_reference=True,
                enable_mixed_decode_migration=True,
            )
        )
        req = GenerateRequest([1], "rust", 10, None, asyncio.Queue())
        req.first_token_at = time.time()
        source_generator = MagicMock()
        source_generator.active_handles.return_value = (GeneratorHandle(uid=7),)
        source_generator.take_states.return_value = [
            GeneratorState(
                handle=GeneratorHandle(uid=7),
                next_token=101,
                logprobs="lp",
                max_tokens=10,
                num_tokens=1,
                cache=["cache"],
                sampler=None,
                logits_processors=[],
                tokens=[1, 2, 3],
            )
        ]
        source_generator.restore_states.return_value = [GeneratorHandle(uid=17)]
        source_slot = _AdapterSlot(
            generator=source_generator,
            adapter_id="rust",
            active_uids={7},
        )
        shared_generator = MagicMock()
        shared_generator.restore_states.side_effect = RuntimeError("boom")
        shared_slot = _AdapterSlot(
            generator=shared_generator,
            adapter_id=MIXED_DECODE_ADAPTER_ID,
            generator_key=MIXED_DECODE_ADAPTER_ID,
            active_uids={99},
        )
        engine._uid_to_request[("rust", 7)] = req
        engine._generators[MIXED_DECODE_ADAPTER_ID] = shared_slot
        engine._get_or_create_mixed_decode_slot = lambda: shared_slot

        with pytest.raises(RuntimeError, match="boom"):
            engine._migrate_decode_ready_from_slot(source_slot)

        source_generator.restore_states.assert_called_once()
        assert source_slot.active_uids == {17}
        assert req.uid == 17
        assert ("rust", 7) not in engine._uid_to_request
        assert ("rust", 17) in engine._uid_to_request

    def test_decode_row_bindings_expand_shared_slot_requests(self):
        engine = _make_engine()
        shared_generator = MagicMock()
        shared_generator.active_handles.return_value = (
            GeneratorHandle(uid=70),
            GeneratorHandle(uid=71),
        )
        shared_slot = _AdapterSlot(
            generator=shared_generator,
            adapter_id=MIXED_DECODE_ADAPTER_ID,
            generator_key=MIXED_DECODE_ADAPTER_ID,
            active_uids={70, 71},
        )
        rust_req = GenerateRequest([1], "rust", 10, None, asyncio.Queue())
        sql_req = GenerateRequest([2], "sql", 10, None, asyncio.Queue())
        engine._uid_to_request[(MIXED_DECODE_ADAPTER_ID, 70)] = rust_req
        engine._uid_to_request[(MIXED_DECODE_ADAPTER_ID, 71)] = sql_req
        engine._ordered_slots = lambda: [shared_slot]
        engine.mola_model.adapter_slot_id.side_effect = (
            lambda adapter_id: {"rust": 1, "sql": 2}.get(adapter_id)
        )

        bindings = engine.decode_row_bindings()

        assert bindings == (
            DecodeRowBinding("rust", 1, 70),
            DecodeRowBinding("sql", 2, 71),
        )

    def test_step_slot_does_not_kill_engine_on_migration_failure(self):
        engine = _make_engine(
            config=EngineConfig(enable_mixed_decode_migration=True)
        )
        slot = _AdapterSlot(
            generator=MagicMock(),
            adapter_id="rust",
            active_uids={7},
        )
        slot.generator.step.return_value = []
        engine._migrate_decode_ready_from_slot = MagicMock(
            side_effect=RuntimeError("boom")
        )

        engine._step_slot(slot)

        engine._migrate_decode_ready_from_slot.assert_called_once_with(slot)

    def test_step_mixed_decode_slot_records_shared_batch_metrics(self):
        engine = _make_engine(config=EngineConfig(enable_routed_decode_reference=True))
        req = GenerateRequest([1], "rust", 10, None, asyncio.Queue())
        req.first_token_at = time.time()
        shared_generator = MagicMock()
        shared_generator.active_handles.return_value = (GeneratorHandle(uid=70),)
        shared_generator.step.return_value = [
            SimpleNamespace(
                handle=GeneratorHandle(uid=70),
                token="ok",
                finish_reason=None,
            )
        ]
        shared_slot = _AdapterSlot(
            generator=shared_generator,
            adapter_id=MIXED_DECODE_ADAPTER_ID,
            generator_key=MIXED_DECODE_ADAPTER_ID,
            active_uids={70},
        )
        engine._uid_to_request[(MIXED_DECODE_ADAPTER_ID, 70)] = req
        engine.mola_model.adapter_slot_id.side_effect = lambda adapter_id: {"rust": 1}.get(
            adapter_id
        )
        engine._dispatch_token = MagicMock()
        engine._build_mixed_decode_routed_session_for_slot_locked = lambda _slot: object()

        engine._step_mixed_decode_slot(shared_slot)

        snap = engine.metrics.snapshot()
        assert snap["mixed_decode_steps"] == 1
        assert snap["mixed_decode_rows"] == 1
        assert snap["avg_mixed_decode_rows"] == 1.0


class TestScheduling:
    def test_orders_slots_by_oldest_unstarted_request(self):
        engine = _make_engine()
        now = time.time()

        rust_req = GenerateRequest([1], "rust", 10, None, asyncio.Queue())
        rust_req.created_at = now - 1
        rust_req.first_token_at = now - 0.5

        sql_req = GenerateRequest([2], "sql", 10, None, asyncio.Queue())
        sql_req.created_at = now - 5

        rust_slot = _AdapterSlot(generator=MagicMock(), adapter_id="rust", active_uids={0})
        rust_slot.last_service_ts = now - 2
        sql_slot = _AdapterSlot(generator=MagicMock(), adapter_id="sql", active_uids={0})
        sql_slot.last_service_ts = now - 1

        engine._generators["rust"] = rust_slot
        engine._generators["sql"] = sql_slot
        engine._uid_to_request[("rust", 0)] = rust_req
        engine._uid_to_request[("sql", 0)] = sql_req

        ordered = engine._ordered_slots()
        assert [slot.adapter_id for slot in ordered] == ["sql", "rust"]

    def test_accrues_service_debt_for_active_decode(self):
        engine = _make_engine()
        slot = _AdapterSlot(generator=MagicMock(), adapter_id="rust", active_uids={0})
        engine._generators["rust"] = slot
        engine._uid_to_request[("rust", 0)] = GenerateRequest([1], "rust", 10, None, asyncio.Queue())

        with engine._state_lock:
            engine._accrue_service_debt_locked()

        assert slot.service_debt == 1.5

    def test_prefill_runs_every_interval_when_decode_active(self):
        engine = _make_engine(config=EngineConfig(prefill_interval=3))
        assert engine._should_run_prefill(iteration=1, has_decode=True) is False
        assert engine._should_run_prefill(iteration=2, has_decode=True) is False
        assert engine._should_run_prefill(iteration=3, has_decode=True) is True

    def test_prefill_always_runs_without_decode(self):
        engine = _make_engine(config=EngineConfig(prefill_interval=99))
        assert engine._should_run_prefill(iteration=1, has_decode=False) is True

    def test_completion_releases_token_budget(self):
        engine = _make_engine()
        req = GenerateRequest([1, 2], "rust", 10, None, asyncio.Queue())
        engine._reserved_tokens = req.estimated_tokens
        engine.metrics.inflight_tokens_reserved = req.estimated_tokens
        engine._uid_to_request[("rust", 0)] = req
        engine._send_to_queue = lambda *_args, **_kwargs: True

        engine._dispatch_token("rust", 0, 11, "stop")

        assert engine.metrics.inflight_tokens_reserved == 0
        assert ("rust", 0) not in engine._uid_to_request


class TestRoutedDecodeStepFailureHandling:
    def test_step_slot_fails_slot_on_routed_contract_error(self):
        engine = _make_engine(config=EngineConfig(enable_routed_decode_reference=True))
        slot = _AdapterSlot(generator=MagicMock(), adapter_id="rust", active_uids={1})
        failed = []
        engine._fail_slot = lambda target_slot, message: failed.append((target_slot, message))
        engine._maybe_build_homogeneous_decode_routed_session_for_slot_locked = (
            lambda target_slot: (_ for _ in ()).throw(RoutedDecodeContractError("boom"))
        )

        engine._step_slot(slot)

        assert failed == [(slot, "internal error")]
        slot.generator.step.assert_not_called()

    def test_step_slot_falls_back_when_routed_backend_is_unavailable(self, monkeypatch):
        engine = _make_engine(config=EngineConfig(enable_routed_decode_reference=True))
        slot = _AdapterSlot(generator=MagicMock(), adapter_id="rust", active_uids={1})
        slot.generator.step.return_value = []
        engine._maybe_build_homogeneous_decode_routed_session_for_slot_locked = (
            lambda target_slot: (_ for _ in ()).throw(ImportError("metal unavailable"))
        )
        seen = []

        from contextlib import contextmanager

        @contextmanager
        def fake_routed_context(active_session):
            seen.append(active_session)
            yield

        monkeypatch.setattr("mola.engine.routed_decode_context", fake_routed_context)

        engine._step_slot(slot)

        assert seen == [None]
        slot.generator.step.assert_called_once()
