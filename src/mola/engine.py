"""Batched generation engine with per-adapter BatchGenerators.

Each active adapter gets its own BatchGenerator (created on demand,
destroyed after idle timeout). The engine runs in a background thread,
round-robining between generators. Same-adapter requests are batched
in a single forward pass via BatchGenerator.insert (continuous batching).

Lifecycle:
- cancel() works pre-UID (queued) and post-UID (generating)
- model_lock prevents adapter load/unload from racing with forward passes
- generators are closed explicitly on cleanup (Metal resource release)
- stop() drains in-flight requests before shutting down
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mlx.core as mx
from mlx_lm.generate import BatchGenerator

from mola.context import adapter_context

if TYPE_CHECKING:
    from mola.model import MOLAModel

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    max_queued_requests: int = 128
    max_batch_size: int = 32
    prefill_batch_size: int = 8
    idle_timeout: float = 60.0
    response_queue_size: int = 1024


@dataclass
class EngineMetrics:
    queued_requests: int = 0
    active_generators: int = 0
    active_sequences: int = 0
    total_tokens_generated: int = 0
    requests_completed: int = 0
    _ttft_samples: list[float] = field(default_factory=list)
    _tps_samples: list[float] = field(default_factory=list)

    def record_completion(self, ttft: float, tps: float):
        self._ttft_samples.append(ttft)
        self._tps_samples.append(tps)
        self.requests_completed += 1
        if len(self._ttft_samples) > 100:
            self._ttft_samples = self._ttft_samples[-100:]
            self._tps_samples = self._tps_samples[-100:]

    def snapshot(self) -> dict:
        return {
            "queued_requests": self.queued_requests,
            "active_generators": self.active_generators,
            "active_sequences": self.active_sequences,
            "total_tokens_generated": self.total_tokens_generated,
            "requests_completed": self.requests_completed,
            "avg_ttft_ms": round(
                sum(self._ttft_samples) / len(self._ttft_samples) * 1000, 1
            )
            if self._ttft_samples
            else 0,
            "avg_tps": round(
                sum(self._tps_samples) / len(self._tps_samples), 1
            )
            if self._tps_samples
            else 0,
        }


@dataclass
class GenerateRequest:
    prompt_tokens: list[int]
    adapter_id: str | None
    max_tokens: int
    sampler: object | None
    response_queue: asyncio.Queue
    created_at: float = field(default_factory=time.time)
    cancelled: bool = False
    first_token_at: float | None = None
    token_count: int = 0


@dataclass
class _AdapterSlot:
    generator: BatchGenerator
    adapter_id: str | None
    active_uids: set[int] = field(default_factory=set)
    last_active: float = field(default_factory=time.time)


class MOLAEngine:
    """Batched multi-adapter generation engine.

    model_lock is held during forward passes and must be acquired
    by admin operations (adapter load/unload) to prevent racing.
    """

    def __init__(self, mola_model: MOLAModel, config: EngineConfig | None = None):
        self.mola_model = mola_model
        self.config = config or EngineConfig()
        self.metrics = EngineMetrics()
        self.model_lock = threading.Lock()

        self._request_queue: queue.Queue[GenerateRequest] = queue.Queue(
            maxsize=self.config.max_queued_requests
        )
        self._generators: dict[str | None, _AdapterSlot] = {}
        self._uid_to_request: dict[int, GenerateRequest] = {}
        self._pending_by_queue: dict[int, GenerateRequest] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._stop_tokens = _get_stop_tokens(mola_model.tokenizer)

    def start(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="mola-engine"
        )
        self._thread.start()
        logger.info("Engine started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        for uid, req in list(self._uid_to_request.items()):
            self._send_to_queue(req, {"error": "server shutting down"})
            self._send_to_queue(req, None)
        self._uid_to_request.clear()
        for slot in self._generators.values():
            self._close_generator(slot)
        self._generators.clear()
        self._pending_by_queue.clear()
        logger.info("Engine stopped")

    def submit(self, request: GenerateRequest):
        """Raises queue.Full on backpressure."""
        self._request_queue.put_nowait(request)
        self._pending_by_queue[id(request.response_queue)] = request
        self.metrics.queued_requests = self._request_queue.qsize()

    def cancel(self, response_queue: asyncio.Queue):
        """Works pre-UID (still queued) and post-UID (in a generator)."""
        req = self._pending_by_queue.get(id(response_queue))
        if req:
            req.cancelled = True
            return
        for req in self._uid_to_request.values():
            if req.response_queue is response_queue:
                req.cancelled = True
                return

    def _run(self):
        cleanup_interval = 10.0
        last_cleanup = time.time()

        while self._running:
            self._drain_requests()
            self._process_cancelled()

            any_active = False
            for adapter_id, slot in list(self._generators.items()):
                if not slot.active_uids:
                    continue

                any_active = True
                with self.model_lock:
                    with adapter_context(adapter_id):
                        try:
                            responses = slot.generator.next()
                        except StopIteration:
                            continue
                        except Exception:
                            logger.exception(
                                f"BatchGenerator error for adapter '{adapter_id}'"
                            )
                            self._fail_slot(slot, "internal error")
                            continue

                slot.last_active = time.time()
                finished_uids = set()
                for resp in responses:
                    self._dispatch_token(resp.uid, resp.token, resp.finish_reason)
                    if resp.finish_reason is not None:
                        finished_uids.add(resp.uid)

                slot.active_uids -= finished_uids
                self._update_sequence_count()

            now = time.time()
            if now - last_cleanup > cleanup_interval:
                self._cleanup_idle()
                last_cleanup = now

            if not any_active:
                time.sleep(0.001)

    def _drain_requests(self):
        drained = 0
        while True:
            try:
                req = self._request_queue.get_nowait()
            except queue.Empty:
                break

            self._pending_by_queue.pop(id(req.response_queue), None)
            if req.cancelled:
                continue

            slot = self._get_or_create_slot(req.adapter_id)
            with self.model_lock:
                with adapter_context(req.adapter_id):
                    uids = slot.generator.insert(
                        prompts=[req.prompt_tokens],
                        max_tokens=[req.max_tokens],
                        samplers=[req.sampler] if req.sampler else None,
                    )
            uid = uids[0]
            slot.active_uids.add(uid)
            self._uid_to_request[uid] = req
            slot.last_active = time.time()
            drained += 1

        if drained:
            self.metrics.queued_requests = self._request_queue.qsize()
            self._update_sequence_count()

    def _process_cancelled(self):
        to_remove = [
            uid for uid, req in self._uid_to_request.items() if req.cancelled
        ]
        for uid in to_remove:
            for slot in self._generators.values():
                if uid in slot.active_uids:
                    try:
                        slot.generator.remove([uid])
                    except Exception:
                        pass
                    slot.active_uids.discard(uid)
                    break
            del self._uid_to_request[uid]
        if to_remove:
            self._update_sequence_count()

    def _dispatch_token(self, uid: int, token: int, finish_reason: str | None):
        req = self._uid_to_request.get(uid)
        if not req or req.cancelled:
            return

        now = time.time()
        if req.first_token_at is None:
            req.first_token_at = now

        if not self._send_to_queue(req, {"token": token, "finish_reason": finish_reason}):
            req.cancelled = True
            return

        req.token_count += 1
        self.metrics.total_tokens_generated += 1

        if finish_reason is not None:
            self._send_to_queue(req, None)
            ttft = (req.first_token_at - req.created_at) if req.first_token_at else 0
            elapsed = now - req.first_token_at if req.first_token_at and req.first_token_at != now else 0.001
            tps = req.token_count / elapsed if elapsed > 0 else 0
            self.metrics.record_completion(ttft, tps)
            self._uid_to_request.pop(uid, None)

    def _send_to_queue(self, req: GenerateRequest, data) -> bool:
        """Thread-safe dispatch to the async response queue. Returns False on failure."""
        if not self._loop:
            return False
        try:
            self._loop.call_soon_threadsafe(req.response_queue.put_nowait, data)
            return True
        except (asyncio.QueueFull, RuntimeError):
            return False

    def _fail_slot(self, slot: _AdapterSlot, error_msg: str):
        for uid in list(slot.active_uids):
            req = self._uid_to_request.pop(uid, None)
            if req:
                self._send_to_queue(req, {"error": error_msg})
                self._send_to_queue(req, None)
        slot.active_uids.clear()
        self._update_sequence_count()

    def _get_or_create_slot(self, adapter_id: str | None) -> _AdapterSlot:
        if adapter_id not in self._generators:
            gen = BatchGenerator(
                self.mola_model.model,
                max_tokens=4096,
                stop_tokens=self._stop_tokens,
                completion_batch_size=self.config.max_batch_size,
                prefill_batch_size=self.config.prefill_batch_size,
            )
            self._generators[adapter_id] = _AdapterSlot(
                generator=gen, adapter_id=adapter_id
            )
            logger.info(f"Created BatchGenerator for adapter '{adapter_id}'")
            self.metrics.active_generators = len(self._generators)
        return self._generators[adapter_id]

    def _close_generator(self, slot: _AdapterSlot):
        try:
            slot.generator.close()
        except Exception:
            pass

    def _cleanup_idle(self):
        now = time.time()
        to_remove = [
            aid
            for aid, slot in self._generators.items()
            if not slot.active_uids
            and (now - slot.last_active) > self.config.idle_timeout
        ]
        for aid in to_remove:
            self._close_generator(self._generators[aid])
            del self._generators[aid]
            logger.info(f"Destroyed idle BatchGenerator for adapter '{aid}'")
        if to_remove:
            self.metrics.active_generators = len(self._generators)

    def _update_sequence_count(self):
        self.metrics.active_sequences = sum(
            len(s.active_uids) for s in self._generators.values()
        )


def _get_stop_tokens(tokenizer) -> set[int]:
    stop = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        if isinstance(eos, list):
            stop.update(eos)
        else:
            stop.add(eos)
    return stop
