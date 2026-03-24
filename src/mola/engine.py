"""Batched multi-adapter generation engine."""

from __future__ import annotations

import asyncio
from collections import deque
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from mola.adapter import AdapterSlotBinding
from mola.application.packing import LayerSlotPackView, MaterializedLayerSlotPack, RoutedLayerPackState, build_layer_slot_pack_state, build_layer_slot_pack_views, materialize_layer_slot_packs
from mola.application.admission import AdmissionRejected, TokenBudgetAdmissionPolicy
from mola.application.scheduling import SlotSchedulingState, WaitingAwareSchedulingPolicy
from mola.context import adapter_context, routed_decode_context
from mola.ports.generator import GeneratorHandle, GeneratorPort, GeneratorSubmission
from mola.ports.routed_decode import RoutedLoRADeltaSession, RoutedLoRADeltaSessionFactory

if TYPE_CHECKING:
    from mola.model import MOLAModel

logger = logging.getLogger(__name__)

RequestKey = tuple[str | None, int]


@dataclass(frozen=True)
class RuntimeSlotLayout:
    loaded_slot_ids: tuple[int, ...]
    active_slot_ids: tuple[int, ...]
    pending_slot_ids: tuple[int, ...]


@dataclass(frozen=True)
class DecodeRowBinding:
    adapter_id: str | None
    slot_id: int
    uid: int


@dataclass
class EngineConfig:
    max_queued_requests: int = 128
    max_batch_size: int = 32
    prefill_batch_size: int = 8
    idle_timeout: float = 60.0
    response_queue_size: int = 1024
    max_inflight_tokens: int = 32768
    prefill_interval: int = 2
    prefill_slot_limit: int = 1
    enable_routed_decode_reference: bool = False


@dataclass
class SlotMetrics:
    steps_called: int = 0
    tokens_emitted: int = 0
    total_next_ms: float = 0.0
    total_step_lock_wait_ms: float = 0.0
    max_step_lock_wait_ms: float = 0.0
    insert_calls: int = 0
    inserted_requests: int = 0
    total_insert_ms: float = 0.0
    total_insert_lock_wait_ms: float = 0.0
    max_insert_lock_wait_ms: float = 0.0
    last_step_ts: float = 0.0
    max_gap_ms: float = 0.0

    def record_step(self, duration_ms: float, lock_wait_ms: float, tokens: int):
        now = time.time()
        if self.last_step_ts > 0:
            gap = (now - self.last_step_ts) * 1000
            if gap > self.max_gap_ms:
                self.max_gap_ms = gap
        self.last_step_ts = now
        self.steps_called += 1
        self.tokens_emitted += tokens
        self.total_next_ms += duration_ms
        self.total_step_lock_wait_ms += lock_wait_ms
        if lock_wait_ms > self.max_step_lock_wait_ms:
            self.max_step_lock_wait_ms = lock_wait_ms

    def record_insert(self, duration_ms: float, lock_wait_ms: float, requests: int):
        self.insert_calls += 1
        self.inserted_requests += requests
        self.total_insert_ms += duration_ms
        self.total_insert_lock_wait_ms += lock_wait_ms
        if lock_wait_ms > self.max_insert_lock_wait_ms:
            self.max_insert_lock_wait_ms = lock_wait_ms

    def snapshot(self) -> dict:
        return {
            "steps": self.steps_called,
            "tokens": self.tokens_emitted,
            "avg_next_ms": round(self.total_next_ms / self.steps_called, 2) if self.steps_called else 0,
            "avg_step_lock_wait_ms": round(self.total_step_lock_wait_ms / self.steps_called, 2) if self.steps_called else 0,
            "max_step_lock_wait_ms": round(self.max_step_lock_wait_ms, 2),
            "insert_calls": self.insert_calls,
            "inserted_requests": self.inserted_requests,
            "avg_insert_ms": round(self.total_insert_ms / self.insert_calls, 2) if self.insert_calls else 0,
            "avg_insert_lock_wait_ms": round(self.total_insert_lock_wait_ms / self.insert_calls, 2) if self.insert_calls else 0,
            "max_insert_lock_wait_ms": round(self.max_insert_lock_wait_ms, 2),
            "max_gap_ms": round(self.max_gap_ms, 1),
        }


@dataclass
class EngineMetrics:
    queued_requests: int = 0
    active_generators: int = 0
    active_sequences: int = 0
    total_tokens_generated: int = 0
    requests_completed: int = 0
    requests_rejected: int = 0
    inflight_tokens_reserved: int = 0
    token_budget_limit: int = 0
    total_step_lock_wait_ms: float = 0.0
    total_insert_lock_wait_ms: float = 0.0
    routed_decode_reference_enabled: bool = False
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
            "requests_rejected": self.requests_rejected,
            "inflight_tokens_reserved": self.inflight_tokens_reserved,
            "token_budget_limit": self.token_budget_limit,
            "total_step_lock_wait_ms": round(self.total_step_lock_wait_ms, 2),
            "total_insert_lock_wait_ms": round(self.total_insert_lock_wait_ms, 2),
            "routed_decode_reference_enabled": self.routed_decode_reference_enabled,
            "avg_ttft_ms": round(
                sum(self._ttft_samples) / len(self._ttft_samples) * 1000, 1
            ) if self._ttft_samples else 0,
            "avg_tps": round(
                sum(self._tps_samples) / len(self._tps_samples), 1
            ) if self._tps_samples else 0,
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
    inserted_at: float | None = None
    uid: int | None = None
    terminal_sent_at: float | None = None
    budget_released: bool = False

    @property
    def qid(self) -> int:
        return id(self.response_queue)

    @property
    def estimated_tokens(self) -> int:
        return len(self.prompt_tokens) + max(self.max_tokens, 0)

    def to_submission(self) -> GeneratorSubmission:
        return GeneratorSubmission(
            prompt_tokens=self.prompt_tokens,
            max_tokens=self.max_tokens,
            sampler=self.sampler,
        )


@dataclass
class _AdapterSlot:
    generator: GeneratorPort
    adapter_id: str | None
    active_uids: set[int] = field(default_factory=set)
    pending_requests: deque[GenerateRequest] = field(default_factory=deque)
    service_debt: float = 0.0
    last_active: float = field(default_factory=time.time)
    last_service_ts: float = 0.0
    slot_metrics: SlotMetrics = field(default_factory=SlotMetrics)


class MOLAEngine:
    def __init__(
        self,
        mola_model: MOLAModel,
        config: EngineConfig | None = None,
        generator_factory: Callable[..., GeneratorPort] | None = None,
        routed_decode_session_factory: RoutedLoRADeltaSessionFactory | None = None,
    ):
        self.mola_model = mola_model
        self.config = config or EngineConfig()
        self.metrics = EngineMetrics()
        self.metrics.token_budget_limit = self.config.max_inflight_tokens
        self.metrics.routed_decode_reference_enabled = (
            self.config.enable_routed_decode_reference
        )
        self.admission_policy = TokenBudgetAdmissionPolicy(self.config.max_inflight_tokens)
        self.scheduling_policy: WaitingAwareSchedulingPolicy[str | None] = (
            WaitingAwareSchedulingPolicy()
        )
        self._generator_factory = generator_factory or self._default_generator_factory
        self._routed_decode_session_factory = routed_decode_session_factory
        self.model_lock = threading.Lock()
        self._state_lock = threading.Lock()

        self._request_queue: queue.Queue[GenerateRequest] = queue.Queue(
            maxsize=self.config.max_queued_requests
        )
        self._generators: dict[str | None, _AdapterSlot] = {}
        self._uid_to_request: dict[RequestKey, GenerateRequest] = {}
        self._pending_by_queue: dict[int, GenerateRequest] = {}
        self._reserved_tokens = 0
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
        logger.debug(
            f"engine_stop inflight={len(self._uid_to_request)} "
            f"pending={len(self._pending_by_queue)} "
            f"generators={len(self._generators)}"
        )
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

        with self._state_lock:
            pending = {req.qid: req for req in self._pending_by_queue.values()}
            for slot in self._generators.values():
                for req in slot.pending_requests:
                    pending[req.qid] = req
            active = list(self._uid_to_request.values())
            slots = list(self._generators.values())
            self._uid_to_request.clear()
            self._pending_by_queue.clear()
            self._generators.clear()
            self._reserved_tokens = 0
            self.metrics.inflight_tokens_reserved = 0
            self.metrics.queued_requests = 0
            self.metrics.active_generators = 0
            self.metrics.active_sequences = 0

        for req in [*active, *pending.values()]:
            self._send_to_queue(req, {"error": "server shutting down"})
            self._send_to_queue(req, None)
        for slot in slots:
            self._close_generator(slot)
        logger.info("Engine stopped")

    def submit(self, request: GenerateRequest):
        with self._state_lock:
            decision = self.admission_policy.evaluate(
                reserved_tokens=self._reserved_tokens,
                request_tokens=request.estimated_tokens,
            )
            if not decision.accepted:
                self.metrics.requests_rejected += 1
                raise AdmissionRejected(decision.reason or "Server overloaded")
            self._reserved_tokens = decision.projected_tokens
            self.metrics.inflight_tokens_reserved = self._reserved_tokens
            self._pending_by_queue[request.qid] = request

        try:
            self._request_queue.put_nowait(request)
        except queue.Full:
            with self._state_lock:
                self._pending_by_queue.pop(request.qid, None)
                self._release_budget_locked(request)
                self.metrics.queued_requests = self._queued_request_count_locked()
            raise

        with self._state_lock:
            self.metrics.queued_requests = self._queued_request_count_locked()
        logger.debug(
            f"submit queue={request.qid} adapter={request.adapter_id} "
            f"prompt_len={len(request.prompt_tokens)} max_tokens={request.max_tokens} "
            f"qsize={self.metrics.queued_requests}"
        )

    def cancel(self, response_queue: asyncio.Queue):
        qid = id(response_queue)
        with self._state_lock:
            req = self._pending_by_queue.get(qid)
            if req:
                req.cancelled = True
                logger.debug(f"cancel queue={qid} state=pending")
                return
            for slot in self._generators.values():
                for pending_req in slot.pending_requests:
                    if pending_req.response_queue is response_queue:
                        pending_req.cancelled = True
                        logger.debug(f"cancel queue={qid} state=pending")
                        return
            for req in self._uid_to_request.values():
                if req.response_queue is response_queue:
                    req.cancelled = True
                    logger.debug(f"cancel queue={qid} uid={req.uid} state=active")
                    return
        logger.debug(f"cancel queue={qid} state=unknown")

    def slot_snapshots(self) -> dict:
        now = time.time()
        with self._state_lock:
            return {
                str(aid): {
                    "runtime_slot_id": self._adapter_slot_id(aid),
                    "active_uids": len(slot.active_uids),
                    "batch_size": len(slot.active_uids),
                    "pending_requests": len(slot.pending_requests),
                    "service_debt": round(slot.service_debt, 2),
                    "oldest_wait_ms": round(
                        max(0.0, now - slot.pending_requests[0].created_at) * 1000, 1
                    ) if slot.pending_requests else 0,
                    **slot.slot_metrics.snapshot(),
                }
                for aid, slot in self._generators.items()
            }

    def runtime_slot_layout(self) -> RuntimeSlotLayout:
        bindings = getattr(self.mola_model, "adapter_slot_bindings", None)
        loaded_slot_ids: tuple[int, ...] = ()
        if callable(bindings):
            loaded_slot_ids = tuple(
                binding.slot_id for binding in bindings() if binding.slot_id is not None
            )

        active_slot_ids: list[int] = []
        pending_slot_ids: list[int] = []
        with self._state_lock:
            for adapter_id, slot in self._generators.items():
                runtime_slot_id = self._adapter_slot_id(adapter_id)
                if runtime_slot_id is None:
                    continue
                if slot.active_uids:
                    active_slot_ids.append(runtime_slot_id)
                if slot.pending_requests:
                    pending_slot_ids.append(runtime_slot_id)

        return RuntimeSlotLayout(
            loaded_slot_ids=tuple(sorted(set(loaded_slot_ids))),
            active_slot_ids=tuple(sorted(set(active_slot_ids))),
            pending_slot_ids=tuple(sorted(set(pending_slot_ids))),
        )

    def active_slot_bindings(self) -> tuple[AdapterSlotBinding, ...]:
        bindings = getattr(self.mola_model, "adapter_slot_bindings", None)
        if not callable(bindings):
            return ()

        layout = self.runtime_slot_layout()
        relevant_slot_ids = set(layout.active_slot_ids) | set(layout.pending_slot_ids)
        if not relevant_slot_ids:
            return ()

        return tuple(sorted(
            (
                binding
                for binding in bindings()
                if binding.slot_id in relevant_slot_ids
            ),
            key=lambda binding: binding.slot_id,
        ))

    def slot_bindings_for_slot_ids(
        self,
        slot_ids: list[int] | tuple[int, ...] | set[int],
        *,
        strict: bool = False,
    ) -> tuple[AdapterSlotBinding, ...]:
        bindings = getattr(self.mola_model, "adapter_slot_bindings", None)
        if not callable(bindings):
            return ()
        slot_ids = set(slot_ids)
        if not slot_ids:
            return ()
        matched = tuple(
            sorted(
                (binding for binding in bindings() if binding.slot_id in slot_ids),
                key=lambda binding: binding.slot_id,
            )
        )
        if strict and len(matched) != len(slot_ids):
            missing_slot_ids = sorted(slot_ids - {binding.slot_id for binding in matched})
            raise ValueError(f"missing adapter slot bindings for slot_ids={missing_slot_ids}")
        return matched

    def decode_active_slot_bindings(self) -> tuple[AdapterSlotBinding, ...]:
        active_slot_ids = {binding.slot_id for binding in self.decode_row_bindings()}
        return self.slot_bindings_for_slot_ids(active_slot_ids)

    def layer_slot_pack_views(self) -> tuple[LayerSlotPackView, ...]:
        iter_layers = getattr(self.mola_model, "iter_slot_bound_lora_layers", None)
        if not callable(iter_layers):
            return ()
        return build_layer_slot_pack_views(
            self.active_slot_bindings(),
            iter_layers(),
        )

    def routed_layer_slot_pack_views(
        self,
        token_slot_ids: list[int] | tuple[int, ...] | None = None,
    ) -> tuple[LayerSlotPackView, ...]:
        iter_layers = getattr(self.mola_model, "iter_routed_decode_lora_layers", None)
        active_bindings = (
            self.decode_active_slot_bindings()
            if token_slot_ids is None
            else self.slot_bindings_for_slot_ids(token_slot_ids, strict=True)
        )
        if not active_bindings:
            return ()
        if callable(iter_layers):
            return build_layer_slot_pack_views(
                active_bindings,
                iter_layers(),
            )
        return self.layer_slot_pack_views()

    def materialize_layer_slot_packs(
        self,
        stack_fn: Callable[[list[Any]], Any],
        scale_fn: Callable[[list[float]], Any] | None = None,
    ) -> tuple[MaterializedLayerSlotPack, ...]:
        return materialize_layer_slot_packs(
            self.layer_slot_pack_views(),
            stack_fn=stack_fn,
            scale_fn=scale_fn,
        )

    def materialize_routed_layer_slot_packs(
        self,
        stack_fn: Callable[[list[Any]], Any],
        scale_fn: Callable[[list[float]], Any] | None = None,
        token_slot_ids: list[int] | tuple[int, ...] | None = None,
    ) -> tuple[MaterializedLayerSlotPack, ...]:
        return materialize_layer_slot_packs(
            self.routed_layer_slot_pack_views(token_slot_ids=token_slot_ids),
            stack_fn=stack_fn,
            scale_fn=scale_fn,
        )

    def materialize_layer_slot_pack_state(
        self,
        stack_fn: Callable[[list[Any]], Any],
        scale_fn: Callable[[list[float]], Any] | None = None,
        token_slot_ids: list[int] | tuple[int, ...] | None = None,
    ) -> RoutedLayerPackState:
        return build_layer_slot_pack_state(
            self.materialize_routed_layer_slot_packs(
                stack_fn=stack_fn,
                scale_fn=scale_fn,
                token_slot_ids=token_slot_ids,
            )
        )

    def build_routed_decode_session(
        self,
        token_slot_ids: list[int] | tuple[int, ...],
    ) -> RoutedLoRADeltaSession:
        token_slot_ids = tuple(token_slot_ids)
        return self._get_routed_decode_session_factory().build(
            self.routed_layer_slot_pack_views(token_slot_ids=token_slot_ids),
            token_slot_ids,
        )

    def decode_row_bindings(self) -> tuple[DecodeRowBinding, ...]:
        bindings: list[DecodeRowBinding] = []
        for slot in self._ordered_slots():
            if not slot.active_uids:
                continue
            runtime_slot_id = self._adapter_slot_id(slot.adapter_id)
            if runtime_slot_id is None:
                continue
            for handle in slot.generator.active_handles():
                bindings.append(
                    DecodeRowBinding(
                        adapter_id=slot.adapter_id,
                        slot_id=runtime_slot_id,
                        uid=handle.uid,
                    )
                )
        return tuple(bindings)

    def build_active_decode_session(
        self,
    ) -> RoutedLoRADeltaSession:
        row_bindings = self.decode_row_bindings()
        token_slot_ids = tuple(binding.slot_id for binding in row_bindings)
        return self.build_routed_decode_session(
            token_slot_ids,
        )

    def build_homogeneous_decode_session(
        self,
        adapter_id: str | None,
        token_count: int,
    ) -> RoutedLoRADeltaSession:
        runtime_slot_id = self._adapter_slot_id(adapter_id)
        token_slot_ids = () if runtime_slot_id is None else (runtime_slot_id,) * token_count
        return self.build_routed_decode_session(
            token_slot_ids,
        )

    def _build_homogeneous_decode_routed_session_for_slot(
        self,
        slot: _AdapterSlot,
    ) -> RoutedLoRADeltaSession | None:
        handles = slot.generator.active_handles()
        if not handles:
            return None
        runtime_slot_id = self._adapter_slot_id(slot.adapter_id)
        if runtime_slot_id is None:
            return None
        return self.build_homogeneous_decode_session(
            slot.adapter_id,
            len(handles),
        )

    def _maybe_build_homogeneous_decode_routed_session_for_slot_locked(
        self,
        slot: _AdapterSlot,
    ) -> RoutedLoRADeltaSession | None:
        if not self.config.enable_routed_decode_reference:
            return None
        return self._build_homogeneous_decode_routed_session_for_slot(slot)

    def _adapter_slot_id(self, adapter_id: str | None) -> int | None:
        resolver = getattr(self.mola_model, "adapter_slot_id", None)
        if not callable(resolver):
            return None
        slot_id = resolver(adapter_id)
        if slot_id is None or isinstance(slot_id, int):
            return slot_id
        return None

    def _run(self):
        cleanup_interval = 10.0
        last_cleanup = time.time()
        iteration = 0

        while self._running:
            iteration += 1
            self._drain_requests()
            self._process_cancelled()
            with self._state_lock:
                self._accrue_service_debt_locked()

            any_active = False
            ordered_slots = self._ordered_slots()
            decode_slots = [slot for slot in ordered_slots if slot.active_uids]
            prefill_slots = [slot for slot in ordered_slots if slot.pending_requests]

            for slot in decode_slots:
                self._step_slot(slot)
                any_active = True

            allow_prefill = self._should_run_prefill(
                iteration=iteration,
                has_decode=bool(decode_slots),
            )
            inserted_slots: list[_AdapterSlot] = []
            if allow_prefill:
                limit = self._prefill_limit(has_decode=bool(decode_slots))
                for slot in prefill_slots[:limit]:
                    if self._insert_pending(slot):
                        inserted_slots.append(slot)
                        any_active = True

            if not decode_slots:
                for slot in inserted_slots:
                    self._step_slot(slot)
                    any_active = True

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

            with self._state_lock:
                self._pending_by_queue.pop(req.qid, None)
            if req.cancelled:
                logger.debug(f"drain_skip queue={req.qid} adapter={req.adapter_id} cancelled=true")
                with self._state_lock:
                    self._release_budget_locked(req)
                    self.metrics.queued_requests = self._queued_request_count_locked()
                continue

            slot = self._get_or_create_slot(req.adapter_id)
            with self._state_lock:
                slot.pending_requests.append(req)
                slot.last_active = time.time()
                self.metrics.queued_requests = self._queued_request_count_locked()
            drained += 1

        if drained:
            with self._state_lock:
                self._update_sequence_count_locked()

    def _process_cancelled(self):
        with self._state_lock:
            for slot in self._generators.values():
                if not slot.pending_requests:
                    continue
                kept = deque()
                while slot.pending_requests:
                    req = slot.pending_requests.popleft()
                    if req.cancelled:
                        logger.debug(
                            f"cancel_remove uid=pending adapter={slot.adapter_id} removed=false"
                        )
                        self._release_budget_locked(req)
                        continue
                    kept.append(req)
                slot.pending_requests = kept
                if not slot.active_uids and not slot.pending_requests:
                    slot.service_debt = 0.0
            to_remove = [
                key for key, req in self._uid_to_request.items() if req.cancelled
            ]
            self.metrics.queued_requests = self._queued_request_count_locked()

        for adapter_id, uid in to_remove:
            removed = False
            with self._state_lock:
                slot = self._generators.get(adapter_id)
            if slot and uid in slot.active_uids:
                try:
                    slot.generator.cancel([GeneratorHandle(uid=uid)])
                    removed = True
                except Exception:
                    pass
                with self._state_lock:
                    slot.active_uids.discard(uid)
            logger.debug(
                f"cancel_remove uid={uid} adapter={adapter_id} removed={removed}"
            )
            with self._state_lock:
                req = self._uid_to_request.pop((adapter_id, uid), None)
                if req:
                    self._send_to_queue(req, {"error": "internal error"})
                    self._send_to_queue(req, None)
                    with self._state_lock:
                        self._release_budget_locked(req)
                if slot and not slot.active_uids and not slot.pending_requests:
                    slot.service_debt = 0.0
                self._update_sequence_count_locked()

    def _dispatch_token(
        self,
        adapter_id: str | None,
        uid: int,
        token: int,
        finish_reason: str | None,
    ):
        with self._state_lock:
            req = self._uid_to_request.get((adapter_id, uid))
            if not req or req.cancelled:
                return

        now = time.time()
        if req.first_token_at is None:
            req.first_token_at = now

        ok = self._send_to_queue(req, {"token": token, "finish_reason": finish_reason})
        if not ok:
            logger.debug(
                f"dispatch_fail uid={uid} queue={req.qid} adapter={req.adapter_id} "
                f"token_count={req.token_count}"
            )
            req.cancelled = True
            return

        req.token_count += 1
        self.metrics.total_tokens_generated += 1
        logger.debug(
            f"dispatch_token uid={uid} queue={req.qid} adapter={req.adapter_id} "
            f"finish={finish_reason} token_count={req.token_count} cancelled={req.cancelled}"
        )

        if finish_reason is not None:
            ok = self._send_to_queue(req, None)
            if ok:
                req.terminal_sent_at = time.time()
                logger.debug(
                    f"dispatch_terminal uid={uid} queue={req.qid} adapter={req.adapter_id}"
                )
            else:
                logger.warning(f"dispatch_terminal_fail uid={uid} queue={req.qid}")

            ttft = (req.first_token_at - req.created_at) if req.first_token_at else 0
            elapsed = now - req.first_token_at if req.first_token_at and req.first_token_at != now else 0.001
            tps = req.token_count / elapsed if elapsed > 0 else 0
            self.metrics.record_completion(ttft, tps)
            logger.debug(
                f"dispatch_finish uid={uid} adapter={req.adapter_id} reason={finish_reason} "
                f"tokens={req.token_count} ttft={ttft*1000:.0f}ms tps={tps:.0f} "
                f"total={now - req.created_at:.3f}s"
            )
            with self._state_lock:
                self._uid_to_request.pop((adapter_id, uid), None)
                self._release_budget_locked(req)
                self._update_sequence_count_locked()

    def _send_to_queue(self, req: GenerateRequest, data) -> bool:
        if not self._loop:
            return False
        try:
            self._loop.call_soon_threadsafe(req.response_queue.put_nowait, data)
            return True
        except (asyncio.QueueFull, RuntimeError) as e:
            logger.debug(f"send_to_queue_fail queue={req.qid} error={e}")
            return False

    def _fail_slot(self, slot: _AdapterSlot, error_msg: str):
        with self._state_lock:
            uids = list(slot.active_uids)
            pending = list(slot.pending_requests)
            slot.pending_requests.clear()
        logger.debug(f"slot_fail adapter={slot.adapter_id} uids={uids}")
        for req in pending:
            self._send_to_queue(req, {"error": error_msg})
            self._send_to_queue(req, None)
            with self._state_lock:
                self._release_budget_locked(req)
        for uid in uids:
            with self._state_lock:
                req = self._uid_to_request.pop((slot.adapter_id, uid), None)
            if req:
                self._send_to_queue(req, {"error": error_msg})
                self._send_to_queue(req, None)
                with self._state_lock:
                    self._release_budget_locked(req)
        with self._state_lock:
            slot.active_uids.clear()
            slot.service_debt = 0.0
            self.metrics.queued_requests = self._queued_request_count_locked()
            self._update_sequence_count_locked()

    def _get_or_create_slot(self, adapter_id: str | None) -> _AdapterSlot:
        with self._state_lock:
            slot = self._generators.get(adapter_id)
        if slot is not None:
            return slot
        gen = self._generator_factory(
            self.mola_model.model,
            max_tokens=4096,
            stop_tokens=self._stop_tokens,
            completion_batch_size=self.config.max_batch_size,
            prefill_batch_size=self.config.prefill_batch_size,
        )
        slot = _AdapterSlot(generator=gen, adapter_id=adapter_id)
        with self._state_lock:
            if adapter_id in self._generators:
                self._close_generator(slot)
                return self._generators[adapter_id]
            self._generators[adapter_id] = slot
            self.metrics.active_generators = len(self._generators)
        logger.info(f"Created BatchGenerator for adapter '{adapter_id}'")
        return slot

    def _default_generator_factory(self, *args, **kwargs) -> GeneratorPort:
        from mola.infrastructure.mlx_generator import MLXBatchGeneratorPort

        return MLXBatchGeneratorPort(*args, **kwargs)

    def _default_routed_decode_session_factory(self) -> RoutedLoRADeltaSessionFactory:
        from mola.infrastructure.routed_decode import ReferenceRoutedLoRADeltaSessionFactory

        return ReferenceRoutedLoRADeltaSessionFactory()

    def _get_routed_decode_session_factory(self) -> RoutedLoRADeltaSessionFactory:
        if self._routed_decode_session_factory is None:
            self._routed_decode_session_factory = self._default_routed_decode_session_factory()
        return self._routed_decode_session_factory

    def _close_generator(self, slot: _AdapterSlot):
        try:
            slot.generator.close()
        except Exception:
            pass

    def _ordered_slots(self) -> list[_AdapterSlot]:
        with self._state_lock:
            slots = {
                aid: slot
                for aid, slot in self._generators.items()
                if slot.active_uids or slot.pending_requests
            }
            states = [
                self._slot_state_locked(aid, slot)
                for aid, slot in slots.items()
            ]
        ordered_ids = self.scheduling_policy.order(states)
        return [slots[adapter_id] for adapter_id in ordered_ids]

    def _slot_state_locked(
        self,
        adapter_id: str | None,
        slot: _AdapterSlot,
    ) -> SlotSchedulingState[str | None]:
        return SlotSchedulingState(
            slot_id=adapter_id,
            active_count=len(slot.active_uids),
            pending_count=len(slot.pending_requests),
            last_service_ts=slot.last_service_ts,
            last_active_ts=slot.last_active,
            oldest_unstarted_ts=self._oldest_unstarted_ts_locked(slot),
        )

    def _step_slot(self, slot: _AdapterSlot):
        with self._state_lock:
            active_count = len(slot.active_uids)
        if not active_count:
            return

        logger.debug(
            f"slot_step_start adapter={slot.adapter_id} active={active_count}"
        )
        runtime_slot_id = self._adapter_slot_id(slot.adapter_id)
        lock_wait_start = time.monotonic()
        with self.model_lock:
            lock_wait_ms = (time.monotonic() - lock_wait_start) * 1000
            routed_session = self._maybe_build_homogeneous_decode_routed_session_for_slot_locked(slot)
            t0 = time.monotonic()
            with adapter_context(slot.adapter_id, slot_id=runtime_slot_id):
                with routed_decode_context(routed_session):
                    try:
                        responses = slot.generator.step()
                    except StopIteration:
                        return
                    except Exception:
                        logger.exception(
                            f"BatchGenerator error for adapter '{slot.adapter_id}'"
                        )
                        self._fail_slot(slot, "internal error")
                        return
        step_ms = (time.monotonic() - t0) * 1000

        finished_uids = set()
        resp_uids = []
        resp_reasons = []
        for resp in responses:
            resp_uids.append(resp.handle.uid)
            resp_reasons.append(resp.finish_reason)
            self._dispatch_token(
                slot.adapter_id, resp.handle.uid, resp.token, resp.finish_reason
            )
            if resp.finish_reason is not None:
                finished_uids.add(resp.handle.uid)

        with self._state_lock:
            slot.slot_metrics.record_step(step_ms, lock_wait_ms, len(responses))
            self.metrics.total_step_lock_wait_ms += lock_wait_ms
            slot.service_debt = max(0.0, slot.service_debt - 1.0)
            slot.last_active = time.time()
            slot.last_service_ts = slot.last_active
            slot.active_uids -= finished_uids
            if not slot.active_uids and not slot.pending_requests:
                slot.service_debt = 0.0
            self._update_sequence_count_locked()
        logger.debug(
            f"slot_step_done adapter={slot.adapter_id} responses={len(responses)} "
            f"uids={resp_uids} reasons={resp_reasons} next_ms={step_ms:.1f} "
            f"lock_wait_ms={lock_wait_ms:.1f} runtime_slot_id={runtime_slot_id}"
        )

    def _should_run_prefill(self, *, iteration: int, has_decode: bool) -> bool:
        if not has_decode:
            return True
        return iteration % max(self.config.prefill_interval, 1) == 0

    def _prefill_limit(self, *, has_decode: bool) -> int:
        if not has_decode:
            return self.config.max_batch_size
        return max(1, self.config.prefill_slot_limit)

    def _accrue_service_debt_locked(self):
        for slot in self._generators.values():
            if slot.active_uids:
                slot.service_debt += 1.0
                if self._oldest_unstarted_ts_locked(slot) is not None:
                    slot.service_debt += 0.5
            elif slot.pending_requests:
                slot.service_debt += 0.25

    def _oldest_unstarted_ts_locked(self, slot: _AdapterSlot) -> float | None:
        candidates = [req.created_at for req in slot.pending_requests]
        for uid in slot.active_uids:
            req = self._uid_to_request.get((slot.adapter_id, uid))
            if req and req.first_token_at is None:
                candidates.append(req.created_at)
        return min(candidates) if candidates else None

    def _insert_pending(self, slot: _AdapterSlot) -> bool:
        with self._state_lock:
            capacity = self.config.max_batch_size - len(slot.active_uids)
            if capacity <= 0:
                return False
            limit = min(self.config.prefill_batch_size, capacity)
            batch: list[GenerateRequest] = []
            while slot.pending_requests and len(batch) < limit:
                req = slot.pending_requests.popleft()
                if req.cancelled:
                    logger.debug(
                        f"drain_skip queue={req.qid} adapter={req.adapter_id} cancelled=true"
                    )
                    self._release_budget_locked(req)
                    continue
                batch.append(req)
            self.metrics.queued_requests = self._queued_request_count_locked()

        if not batch:
            return False

        try:
            runtime_slot_id = self._adapter_slot_id(slot.adapter_id)
            lock_wait_start = time.monotonic()
            with self.model_lock:
                lock_wait_ms = (time.monotonic() - lock_wait_start) * 1000
                t0 = time.monotonic()
                with adapter_context(slot.adapter_id, slot_id=runtime_slot_id):
                    handles = slot.generator.submit_batch(
                        [req.to_submission() for req in batch]
                    )
            insert_ms = (time.monotonic() - t0) * 1000
        except Exception:
            logger.exception(f"BatchGenerator insert error for adapter '{slot.adapter_id}'")
            for req in batch:
                self._send_to_queue(req, {"error": "internal error"})
                self._send_to_queue(req, None)
                with self._state_lock:
                    self._release_budget_locked(req)
            with self._state_lock:
                self.metrics.queued_requests = self._queued_request_count_locked()
            return False

        inserted_at = time.time()
        with self._state_lock:
            slot.slot_metrics.record_insert(insert_ms, lock_wait_ms, len(batch))
            self.metrics.total_insert_lock_wait_ms += lock_wait_ms
            for req, handle in zip(batch, handles, strict=True):
                uid = handle.uid
                req.uid = uid
                req.inserted_at = inserted_at
                slot.active_uids.add(uid)
                self._uid_to_request[(slot.adapter_id, uid)] = req
                logger.debug(
                    f"drain_insert queue={req.qid} adapter={req.adapter_id} uid={uid} "
                    f"slot_active={len(slot.active_uids)} generators={len(self._generators)} "
                    f"queue_wait={req.inserted_at - req.created_at:.3f}s"
                )
            slot.last_active = inserted_at
            slot.last_service_ts = inserted_at
            self.metrics.queued_requests = self._queued_request_count_locked()
            self._update_sequence_count_locked()
        logger.debug(
            f"slot_insert_done adapter={slot.adapter_id} batch={len(batch)} insert_ms={insert_ms:.1f} "
            f"lock_wait_ms={lock_wait_ms:.1f} runtime_slot_id={runtime_slot_id}"
        )
        return True

    def _queued_request_count_locked(self) -> int:
        return self._request_queue.qsize() + sum(
            len(slot.pending_requests) for slot in self._generators.values()
        )

    def _release_budget_locked(self, req: GenerateRequest):
        if req.budget_released:
            return
        req.budget_released = True
        self._reserved_tokens = max(0, self._reserved_tokens - req.estimated_tokens)
        self.metrics.inflight_tokens_reserved = self._reserved_tokens

    def _update_sequence_count_locked(self):
        self.metrics.active_sequences = sum(
            len(s.active_uids) for s in self._generators.values()
        )

    def _cleanup_idle(self):
        now = time.time()
        with self._state_lock:
            to_remove = [
                aid
                for aid, slot in self._generators.items()
                if not slot.active_uids
                and not slot.pending_requests
                and (now - slot.last_active) > self.config.idle_timeout
            ]
        for aid in to_remove:
            with self._state_lock:
                slot = self._generators.pop(aid)
            self._close_generator(slot)
            logger.info(f"Destroyed idle BatchGenerator for adapter '{aid}'")
        if to_remove:
            with self._state_lock:
                self.metrics.active_generators = len(self._generators)


def _get_stop_tokens(tokenizer) -> set[int]:
    stop = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        if isinstance(eos, list):
            stop.update(eos)
        else:
            stop.add(eos)
    return stop
