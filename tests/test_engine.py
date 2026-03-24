"""Tests for the batched generation engine."""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from mola.engine import (
    AdmissionRejected,
    EngineConfig,
    EngineMetrics,
    GenerateRequest,
    MOLAEngine,
    _AdapterSlot,
    _get_stop_tokens,
)


def _make_engine(**overrides):
    mola_model = MagicMock()
    mola_model.tokenizer.eos_token_id = 0
    config = overrides.pop("config", None)
    generator_factory = overrides.pop("generator_factory", None)
    return MOLAEngine(mola_model, config, generator_factory=generator_factory)


class TestEngineMetrics:
    def test_empty_snapshot(self):
        m = EngineMetrics()
        snap = m.snapshot()
        assert snap["queued_requests"] == 0
        assert snap["avg_ttft_ms"] == 0
        assert snap["avg_tps"] == 0

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
        req = GenerateRequest([1], "rust", 10, None, q)
        slot = _AdapterSlot(generator=MagicMock(), adapter_id="rust")
        slot.pending_requests.append(req)
        engine._generators["rust"] = slot

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
        slot = _AdapterSlot(generator=mock_gen, adapter_id="active")
        slot.pending_requests.append(GenerateRequest([1], "active", 10, None, asyncio.Queue()))
        slot.last_active = time.time() - 100
        engine._generators["active"] = slot
        engine._cleanup_idle()
        assert "active" in engine._generators
        mock_gen.close.assert_not_called()


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
