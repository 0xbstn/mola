"""Tests for the batched generation engine."""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from mola.engine import (
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
    with patch("mola.engine.BatchGenerator"):
        engine = MOLAEngine(mola_model, config)
    return engine


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


class TestBackpressure:
    def test_submit_raises_on_full_queue(self):
        engine = _make_engine(config=EngineConfig(max_queued_requests=1))
        engine.submit(GenerateRequest([1], None, 10, None, asyncio.Queue()))
        with pytest.raises(queue.Full):
            engine.submit(GenerateRequest([2], None, 10, None, asyncio.Queue()))


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
        engine._uid_to_request[42] = req
        engine.cancel(q)
        assert req.cancelled is True

    def test_cancel_unknown_queue_is_noop(self):
        engine = _make_engine()
        engine.cancel(asyncio.Queue())


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
        engine._generators["test"] = _AdapterSlot(generator=mock_gen, adapter_id="test")
        engine._uid_to_request[0] = GenerateRequest([1], None, 10, None, asyncio.Queue())
        engine.stop()
        assert len(engine._generators) == 0
        assert len(engine._uid_to_request) == 0
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
        mock_gen.insert.return_value = [0]
        engine._generators[None] = _AdapterSlot(generator=mock_gen, adapter_id=None)

        engine._drain_requests()
        assert id(q) not in engine._pending_by_queue
        assert 0 in engine._uid_to_request
