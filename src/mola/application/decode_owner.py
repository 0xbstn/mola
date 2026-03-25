"""Decode ownership seam for shared mixed decode.

Current coupling to ``engine`` is intentionally explicit:
- request bookkeeping still lives in ``engine._uid_to_request``
- slot lifecycle and cache invalidation helpers still live in ``engine``
- the owner still drives an upstream ``BatchGenerator`` through the current
  ``GeneratorPort`` contract (`step`, `take_states`, `restore_states`)
- the next architectural widening point is batch-level ownership transfer on
  the generator/runtime boundary, not more engine-local migration heuristics

This module exists to move mixed decode ownership behind a replaceable boundary
before any deeper runtime/generator refactor. It is not a performance layer by
itself.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mola.application.routed_decode import RoutedDecodeContractError
from mola.context import adapter_context, lora_delta_context, routed_decode_context
from mola.ports.generator import (
    GeneratorDetachedBatch,
    GenerationEvent,
    GeneratorBatchSnapshot,
    GeneratorHandle,
)

if TYPE_CHECKING:
    from mola.engine import MOLAEngine, _AdapterSlot

logger = logging.getLogger(__name__)
MIXED_DECODE_ADAPTER_ID = "__mixed_decode__"


class DecodeOwner(Protocol):
    def active_handles(self, slot: _AdapterSlot) -> tuple[GeneratorHandle, ...]: ...
    def build_routed_session(self, slot: _AdapterSlot): ...
    def restore_to_source_generators(self, slot: _AdapterSlot) -> bool: ...
    def admit_decode_ready_from_slot(self, slot: _AdapterSlot) -> None: ...
    def prestep_admit_decode_ready_slots(self) -> None: ...
    def step(self, slot: _AdapterSlot) -> None: ...


@dataclass(slots=True)
class SharedMixedSlotDecodeOwner:
    engine: MOLAEngine
    shared_detached_batch: GeneratorDetachedBatch | None = None

    def _detached_shared_decode_enabled(self) -> bool:
        return self.engine.config.detached_shared_decode_owner

    def _current_shared_detached_batch_locked(
        self,
        slot: _AdapterSlot,
    ) -> GeneratorDetachedBatch | None:
        batch = self.shared_detached_batch
        if batch is None:
            return None
        handle_uids = tuple(handle.uid for handle in batch.handles)
        if __debug__:
            assert len(handle_uids) == len(set(handle_uids)), (
                "duplicate handles in detached shared batch"
            )
            assert set(handle_uids) == slot.active_uids, (
                f"detached shared batch ownership mismatch for slot={slot.adapter_id}"
            )
        return self.shared_detached_batch

    def _current_shared_detached_batch(
        self,
        slot: _AdapterSlot,
    ) -> GeneratorDetachedBatch | None:
        with self.engine._state_lock:
            return self._current_shared_detached_batch_locked(slot)

    def _attach_source_batch_to_shared_owner(
        self,
        shared_slot: _AdapterSlot,
        batch: GeneratorBatchSnapshot,
        existing_batch: GeneratorDetachedBatch | None,
    ) -> GeneratorDetachedBatch:
        remapped_batch, restored_handles = shared_slot.generator.extend_detached_batch(
            existing_batch,
            batch,
        )
        assert len(remapped_batch.handles) == (
            len(existing_batch.handles) if existing_batch is not None else 0
        ) + len(restored_handles), (
            "shared owner detached batch size mismatch"
        )
        return remapped_batch

    def active_handles(self, slot: _AdapterSlot) -> tuple[GeneratorHandle, ...]:
        if (
            slot.adapter_id == MIXED_DECODE_ADAPTER_ID
            and self._detached_shared_decode_enabled()
        ):
            batch = self._current_shared_detached_batch(slot)
            if batch is None:
                return ()
            return batch.handles
        return slot.generator.active_handles()

    def _take_owned_batch_snapshot(
        self,
        slot: _AdapterSlot,
    ) -> GeneratorBatchSnapshot | None:
        batch = slot.generator.take_active_batch()
        if isinstance(batch, GeneratorBatchSnapshot):
            if __debug__:
                snapshot_uids = tuple(state.handle.uid for state in batch.states)
                assert len(snapshot_uids) == len(set(snapshot_uids)), (
                    f"duplicate uids in batch snapshot for slot={slot.adapter_id}"
                )
                assert set(snapshot_uids) == slot.active_uids, (
                    f"batch snapshot ownership mismatch for slot={slot.adapter_id}"
                )
            return batch

        handles = [
            handle
            for handle in slot.generator.active_handles()
            if handle.uid in slot.active_uids
        ]
        if not handles:
            return None
        states = slot.generator.take_states(handles)
        if not states:
            return None
        return GeneratorBatchSnapshot(states=tuple(states))

    def _restore_batch_snapshot(
        self,
        slot: _AdapterSlot,
        batch: GeneratorBatchSnapshot,
    ) -> tuple[GeneratorHandle, ...]:
        restored = slot.generator.restore_active_batch(batch)
        if (
            isinstance(restored, (list, tuple))
            and len(restored) == len(batch.states)
            and all(isinstance(handle, GeneratorHandle) for handle in restored)
        ):
            return tuple(restored)
        return tuple(slot.generator.restore_states(batch.states))

    def _step_homogeneous(self, slot: _AdapterSlot) -> None:
        with self.engine._state_lock:
            active_count = len(slot.active_uids)
        if not active_count:
            return

        logger.debug(
            f"slot_step_start adapter={slot.adapter_id} active={active_count}"
        )
        runtime_slot_id = self.engine._adapter_slot_id(slot.adapter_id)
        lock_wait_start = time.monotonic()
        with self.engine.model_lock:
            lock_wait_ms = (time.monotonic() - lock_wait_start) * 1000
            try:
                routed_session = (
                    self.engine._maybe_build_homogeneous_decode_routed_session_for_slot_locked(slot)
                )
            except RoutedDecodeContractError:
                logger.exception(
                    "Routed decode contract error for adapter '%s'",
                    slot.adapter_id,
                )
                self.engine._fail_slot(slot, "internal error")
                return
            except Exception:
                logger.exception(
                    "Routed decode backend unavailable for adapter '%s'; falling back",
                    slot.adapter_id,
                )
                routed_session = None
            t0 = time.monotonic()
            with adapter_context(slot.adapter_id, slot_id=runtime_slot_id):
                with lora_delta_context(
                    neutralize=self.engine.config.neutralize_lora_delta
                ):
                    with routed_decode_context(routed_session):
                        try:
                            responses = slot.generator.step()
                        except StopIteration:
                            return
                        except Exception:
                            logger.exception(
                                "BatchGenerator error for adapter '%s'",
                                slot.adapter_id,
                            )
                            self.engine._fail_slot(slot, "internal error")
                            return
        step_ms = (time.monotonic() - t0) * 1000

        finished_uids = set()
        resp_uids = []
        resp_reasons = []
        for resp in responses:
            resp_uids.append(resp.handle.uid)
            resp_reasons.append(resp.finish_reason)
            self.engine._dispatch_token(
                self.engine._slot_generator_key(slot),
                resp.handle.uid,
                resp.token,
                resp.finish_reason,
            )
            if resp.finish_reason is not None:
                finished_uids.add(resp.handle.uid)

        with self.engine._state_lock:
            slot.slot_metrics.record_step(step_ms, lock_wait_ms, len(responses))
            self.engine.metrics.total_step_lock_wait_ms += lock_wait_ms
            self.engine.metrics.homogeneous_decode_steps += 1
            self.engine.metrics.homogeneous_decode_rows += len(responses)
            self.engine.metrics.decode_step_samples += 1
            self.engine.metrics.decode_active_generators_total += (
                self.engine._active_decode_generator_count_locked()
            )
            slot.service_debt = max(0.0, slot.service_debt - 1.0)
            slot.last_active = time.time()
            slot.last_service_ts = slot.last_active
            slot.active_uids -= finished_uids
            if not slot.active_uids and not slot.pending_requests:
                slot.service_debt = 0.0
                self.engine._clear_slot_routed_decode_session_cache(slot)
            self.engine._update_sequence_count_locked()
        logger.debug(
            f"slot_step_done adapter={slot.adapter_id} responses={len(responses)} "
            f"uids={resp_uids} reasons={resp_reasons} next_ms={step_ms:.1f} "
            f"lock_wait_ms={lock_wait_ms:.1f} runtime_slot_id={runtime_slot_id}"
        )
        try:
            self.admit_decode_ready_from_slot(slot)
        except Exception:
            logger.exception(
                "Mixed decode migration failed for adapter '%s'",
                slot.adapter_id,
            )

    def _invalidate_slot_cache_if_composition_changed(
        self,
        slot: _AdapterSlot,
        previous_active_uids: tuple[int, ...],
    ) -> None:
        if tuple(sorted(slot.active_uids)) != previous_active_uids:
            self.engine._clear_slot_routed_decode_session_cache(slot)

    def _debug_assert_slot_request_coherence(self, slot: _AdapterSlot) -> None:
        if not __debug__:
            return
        request_keys = []
        for uid in slot.active_uids:
            key = self.engine._slot_request_key(slot, uid)
            req = self.engine._uid_to_request.get(key)
            assert req is not None, (
                f"missing request binding for active uid={uid} slot={slot.adapter_id}"
            )
            request_keys.append(key)
            if slot.adapter_id == MIXED_DECODE_ADAPTER_ID:
                assert req.adapter_id is not None, (
                    f"shared decode slot lost adapter binding for uid={uid}"
                )
        assert len(request_keys) == len(set(request_keys)), (
            f"duplicate request keys detected for slot={slot.adapter_id}"
        )

    def _debug_assert_unique_request_ownership(self, req: object) -> None:
        if not __debug__:
            return
        matches = sum(1 for candidate in self.engine._uid_to_request.values() if candidate is req)
        assert matches <= 1, "request is bound to multiple decode owners"

    def _debug_assert_finished_requests_released(
        self,
        slot: _AdapterSlot,
        finished_uids: set[int],
    ) -> None:
        if not __debug__:
            return
        for uid in finished_uids:
            assert uid not in slot.active_uids, (
                f"finished uid={uid} still marked active in slot={slot.adapter_id}"
            )
            assert (
                self.engine._uid_to_request.get(self.engine._slot_request_key(slot, uid))
                is None
            ), f"finished uid={uid} still bound in request map"

    def build_routed_session(self, slot: _AdapterSlot):
        if slot.adapter_id != MIXED_DECODE_ADAPTER_ID:
            return None
        if not self.engine.config.enable_routed_decode_reference:
            return None

        token_slot_ids: list[int] = []
        for handle in self.active_handles(slot):
            req = self.engine._uid_to_request.get(
                self.engine._slot_request_key(slot, handle.uid)
            )
            if req is None or req.adapter_id is None or req.first_token_at is None:
                return None
            runtime_slot_id = self.engine._adapter_slot_id(req.adapter_id)
            if runtime_slot_id is None:
                return None
            token_slot_ids.append(runtime_slot_id)
        if not token_slot_ids:
            return None
        token_slot_ids_tuple = tuple(token_slot_ids)
        cached = self.engine._get_slot_cached_routed_decode_session(
            slot,
            token_slot_ids_tuple,
            mixed=True,
        )
        if cached is not None:
            return cached
        session = self.engine.build_routed_decode_session(
            token_slot_ids_tuple,
            mixed=True,
        )
        self.engine._store_slot_routed_decode_session(
            slot,
            token_slot_ids_tuple,
            session,
        )
        return session

    def restore_to_source_generators(self, slot: _AdapterSlot) -> bool:
        if slot.adapter_id != MIXED_DECODE_ADAPTER_ID:
            return False
        previous_shared_uids = tuple(sorted(slot.active_uids))
        if self._detached_shared_decode_enabled():
            detached_batch = self._current_shared_detached_batch(slot)
            batch = (
                slot.generator.snapshot_detached_batch(detached_batch)
                if detached_batch is not None
                else None
            )
        else:
            batch = self._take_owned_batch_snapshot(slot)
        if batch is None:
            return True
        states = batch.states
        handles = tuple(state.handle for state in states)

        handle_reqs: list = []
        with self.engine._state_lock:
            for handle in handles:
                req = self.engine._uid_to_request.get(
                    self.engine._slot_request_key(slot, handle.uid)
                )
                if req is None or req.adapter_id is None:
                    logger.error(
                        "mixed decode restore missing request binding for uid=%s",
                        handle.uid,
                    )
                    rollback_handles = self._restore_batch_snapshot(slot, batch)
                    for old_handle, rollback_handle in zip(
                        handles, rollback_handles, strict=True
                    ):
                        req = self.engine._uid_to_request.pop(
                            self.engine._slot_request_key(slot, old_handle.uid), None
                        )
                        if req is None:
                            continue
                        slot.active_uids.discard(old_handle.uid)
                        req.uid = rollback_handle.uid
                        self.engine._uid_to_request[
                            self.engine._slot_request_key(slot, rollback_handle.uid)
                        ] = req
                        slot.active_uids.add(rollback_handle.uid)
                    self.engine._update_sequence_count_locked()
                    return False
                handle_reqs.append(req)

        groups: dict[str, list[tuple[object, object]]] = {}
        for state, req in zip(states, handle_reqs, strict=True):
            assert req.adapter_id is not None
            groups.setdefault(req.adapter_id, []).append((state, req))

        restored_groups: list[tuple[_AdapterSlot, list[tuple[object, object]], list[object]]] = []
        try:
            for adapter_id, entries in groups.items():
                source_slot = self.engine._get_or_create_slot(adapter_id)
                restored_handles = self._restore_batch_snapshot(
                    source_slot,
                    GeneratorBatchSnapshot(
                        states=tuple(state for state, _ in entries)
                    ),
                )
                restored_groups.append((source_slot, entries, restored_handles))
        except Exception:
            logger.exception("mixed decode restore to source generators failed")
            for source_slot, _entries, restored_handles in restored_groups:
                try:
                    source_slot.generator.take_states(restored_handles)
                except Exception:
                    logger.exception(
                        "mixed decode rollback cleanup failed for adapter '%s'",
                        source_slot.adapter_id,
                    )
            try:
                rollback_handles = self._restore_batch_snapshot(slot, batch)
            except Exception:
                logger.exception("mixed decode rollback to shared slot failed")
                return False
            with self.engine._state_lock:
                for old_handle, rollback_handle, req in zip(
                    handles, rollback_handles, handle_reqs, strict=True
                ):
                    self.engine._uid_to_request.pop(
                        self.engine._slot_request_key(slot, old_handle.uid), None
                    )
                    slot.active_uids.discard(old_handle.uid)
                    req.uid = rollback_handle.uid
                    self.engine._uid_to_request[
                        self.engine._slot_request_key(slot, rollback_handle.uid)
                    ] = req
                    slot.active_uids.add(rollback_handle.uid)
                    self._debug_assert_unique_request_ownership(req)
                self.engine._update_sequence_count_locked()
            return False

        now = time.time()
        with self.engine._state_lock:
            for old_handle in handles:
                self.engine._uid_to_request.pop(
                    self.engine._slot_request_key(slot, old_handle.uid), None
                )
                slot.active_uids.discard(old_handle.uid)
            if self._detached_shared_decode_enabled():
                self.shared_detached_batch = None
            if not slot.active_uids and not slot.pending_requests:
                slot.service_debt = 0.0
            self._invalidate_slot_cache_if_composition_changed(slot, previous_shared_uids)
            for source_slot, entries, restored_handles in restored_groups:
                previous_source_uids = tuple(sorted(source_slot.active_uids))
                for (_state, req), restored_handle in zip(
                    entries, restored_handles, strict=True
                ):
                    req.uid = restored_handle.uid
                    self.engine._uid_to_request[
                        self.engine._slot_request_key(source_slot, restored_handle.uid)
                    ] = req
                    source_slot.active_uids.add(restored_handle.uid)
                    source_slot.last_active = now
                    source_slot.last_service_ts = now
                    self._debug_assert_unique_request_ownership(req)
                self._invalidate_slot_cache_if_composition_changed(
                    source_slot,
                    previous_source_uids,
                )
                self._debug_assert_slot_request_coherence(source_slot)
            self.engine.metrics.active_generators = self.engine._generator_count_locked()
            self.engine._update_sequence_count_locked()
            self._debug_assert_slot_request_coherence(slot)
        logger.debug(
            "mixed_decode_restore count=%s source_slots=%s",
            len(handles),
            [slot.adapter_id for slot, *_ in restored_groups],
        )
        return True

    def admit_decode_ready_from_slot(self, slot: _AdapterSlot) -> None:
        if not self.engine._should_use_mixed_decode_migration(slot.adapter_id):
            return
        if slot.adapter_id is None or slot.adapter_id == MIXED_DECODE_ADAPTER_ID:
            return
        previous_source_uids = tuple(sorted(slot.active_uids))

        with self.engine._state_lock:
            if not self.engine._should_migrate_decode_slot_locked(slot):
                return

        generator_key = self.engine._slot_generator_key(slot)
        shared_slot = self.engine._get_or_create_mixed_decode_slot()
        if self._detached_shared_decode_enabled():
            source_batch = slot.generator.take_active_batch_handle()
            if source_batch is None:
                return
            handles = source_batch.handles
            current_shared_batch = self._current_shared_detached_batch(shared_slot)
            batch = None
        else:
            batch = self._take_owned_batch_snapshot(slot)
            if batch is None:
                return
            states = batch.states
            handles = tuple(state.handle for state in states)
            current_shared_batch = None
        try:
            if self._detached_shared_decode_enabled():
                remapped_batch, restored_handles = shared_slot.generator.promote_detached_batch(
                    current_shared_batch,
                    source_batch,
                )
            else:
                remapped_batch = None
                restored_handles = self._restore_batch_snapshot(shared_slot, batch)
        except Exception:
            if self._detached_shared_decode_enabled():
                slot.generator.restore_detached_batch(source_batch)
                rollback_handles = handles
            else:
                rollback_handles = self._restore_batch_snapshot(slot, batch)
            with self.engine._state_lock:
                for old_handle, rollback_handle in zip(
                    handles, rollback_handles, strict=True
                ):
                    req = self.engine._uid_to_request.pop(
                        self.engine._request_key(generator_key, old_handle.uid), None
                    )
                    if req is None:
                        continue
                    slot.active_uids.discard(old_handle.uid)
                    req.uid = rollback_handle.uid
                    self.engine._uid_to_request[
                        self.engine._slot_request_key(slot, rollback_handle.uid)
                    ] = req
                    slot.active_uids.add(rollback_handle.uid)
                    self._debug_assert_unique_request_ownership(req)
                if self._detached_shared_decode_enabled():
                    self.shared_detached_batch = current_shared_batch
                self.engine._update_sequence_count_locked()
            raise

        now = time.time()
        with self.engine._state_lock:
            previous_shared_uids = tuple(sorted(shared_slot.active_uids))
            for old_handle, new_handle in zip(handles, restored_handles, strict=True):
                req = self.engine._uid_to_request.pop(
                    self.engine._request_key(generator_key, old_handle.uid), None
                )
                if req is None:
                    continue
                if req.inserted_at is not None:
                    self.engine.metrics.mixed_decode_migration_since_insert_ms_total += (
                        now - req.inserted_at
                    ) * 1000
                if req.first_token_at is not None:
                    self.engine.metrics.mixed_decode_migration_since_first_token_ms_total += (
                        now - req.first_token_at
                    ) * 1000
                    self.engine.metrics.mixed_decode_migration_tokens_before_shared += max(
                        req.token_count - 1, 0
                    )
                slot.active_uids.discard(old_handle.uid)
                req.uid = new_handle.uid
                self.engine._uid_to_request[
                    self.engine._slot_request_key(shared_slot, new_handle.uid)
                ] = req
                shared_slot.active_uids.add(new_handle.uid)
                self._debug_assert_unique_request_ownership(req)
            if self._detached_shared_decode_enabled():
                assert remapped_batch is not None
                self.shared_detached_batch = remapped_batch
            if not slot.active_uids and not slot.pending_requests:
                slot.service_debt = 0.0
            self._invalidate_slot_cache_if_composition_changed(slot, previous_source_uids)
            shared_slot.last_active = now
            shared_slot.last_service_ts = now
            self._invalidate_slot_cache_if_composition_changed(
                shared_slot,
                previous_shared_uids,
            )
            self.engine.metrics.mixed_decode_migration_events += 1
            self.engine.metrics.mixed_decode_migrated_sequences += len(restored_handles)
            self.engine.metrics.active_generators = self.engine._generator_count_locked()
            self.engine._update_sequence_count_locked()
            self._debug_assert_slot_request_coherence(slot)
            self._debug_assert_slot_request_coherence(shared_slot)
        logger.debug(
            "migrate_decode adapter=%s count=%s shared_active=%s",
            slot.adapter_id,
            len(restored_handles),
            len(shared_slot.active_uids),
        )

    def prestep_admit_decode_ready_slots(self) -> None:
        if not self.engine.config.prestep_mixed_decode_migration:
            return
        if not self.engine.config.enable_mixed_decode_migration:
            return

        while True:
            with self.engine._state_lock:
                candidates = [
                    slot
                    for slot in self.engine._generators.values()
                    if slot.adapter_id not in (None, MIXED_DECODE_ADAPTER_ID)
                    and self.engine._should_migrate_decode_slot_locked(slot)
                ]
            if not candidates:
                return

            migrated_any = False
            for slot in candidates:
                try:
                    before = tuple(sorted(slot.active_uids))
                    self.admit_decode_ready_from_slot(slot)
                    with self.engine._state_lock:
                        after = tuple(sorted(slot.active_uids))
                    if before != after:
                        migrated_any = True
                except Exception:
                    logger.exception(
                        "Pre-step mixed decode migration failed for adapter '%s'",
                        slot.adapter_id,
                    )
            if not migrated_any:
                return

    def step(self, slot: _AdapterSlot) -> None:
        if slot.adapter_id != MIXED_DECODE_ADAPTER_ID:
            self._step_homogeneous(slot)
            return

        detached_batch = None
        if self._detached_shared_decode_enabled():
            detached_batch = self._current_shared_detached_batch(slot)
            active_count = len(detached_batch.handles) if detached_batch else 0
        else:
            with self.engine._state_lock:
                active_count = len(slot.active_uids)
        if not active_count:
            return

        logger.debug(
            f"slot_step_start adapter={slot.adapter_id} active={active_count}"
        )
        lock_wait_start = time.monotonic()
        with self.engine.model_lock:
            lock_wait_ms = (time.monotonic() - lock_wait_start) * 1000
            try:
                routed_session = self.engine._build_mixed_decode_routed_session_for_slot_locked(
                    slot
                )
            except RoutedDecodeContractError:
                logger.exception("Mixed routed decode contract error")
                if self.engine._restore_mixed_decode_slot_to_source_generators(slot):
                    return
                self.engine._fail_slot(slot, "internal error")
                return
            except Exception:
                logger.exception("Mixed routed decode backend unavailable; falling back")
                if self.engine._restore_mixed_decode_slot_to_source_generators(slot):
                    return
                self.engine._fail_slot(slot, "internal error")
                return
            if routed_session is None:
                if self.engine._restore_mixed_decode_slot_to_source_generators(slot):
                    return
                self.engine._fail_slot(slot, "internal error")
                return
            t0 = time.monotonic()
            with adapter_context(None, slot_id=None):
                with lora_delta_context(
                    neutralize=self.engine.config.neutralize_lora_delta
                ):
                    with routed_decode_context(routed_session):
                        try:
                            if detached_batch is None:
                                responses = slot.generator.step()
                            else:
                                batch_result = slot.generator.step_detached_batch(detached_batch)
                                responses = [
                                    GenerationEvent(
                                        handle=event.handle,
                                        token=event.token,
                                        finish_reason=event.finish_reason,
                                    )
                                    for event in batch_result.events
                                ]
                        except StopIteration:
                            return
                        except Exception:
                            logger.exception("Mixed decode BatchGenerator error")
                            if self.engine._restore_mixed_decode_slot_to_source_generators(slot):
                                return
                            self.engine._fail_slot(slot, "internal error")
                            return
        step_ms = (time.monotonic() - t0) * 1000

        finished_uids = set()
        resp_uids = []
        resp_reasons = []
        for resp in responses:
            resp_uids.append(resp.handle.uid)
            resp_reasons.append(resp.finish_reason)
            self.engine._dispatch_token(
                self.engine._slot_generator_key(slot),
                resp.handle.uid,
                resp.token,
                resp.finish_reason,
            )
            if resp.finish_reason is not None:
                finished_uids.add(resp.handle.uid)

        with self.engine._state_lock:
            previous_active_uids = tuple(sorted(slot.active_uids))
            slot.slot_metrics.record_step(step_ms, lock_wait_ms, len(responses))
            self.engine.metrics.total_step_lock_wait_ms += lock_wait_ms
            self.engine.metrics.mixed_decode_steps += 1
            self.engine.metrics.mixed_decode_rows += len(responses)
            self.engine.metrics.decode_step_samples += 1
            self.engine.metrics.decode_active_generators_total += (
                self.engine._active_decode_generator_count_locked()
            )
            slot.service_debt = max(0.0, slot.service_debt - 1.0)
            slot.last_active = time.time()
            slot.last_service_ts = slot.last_active
            if detached_batch is None:
                slot.active_uids -= finished_uids
            else:
                self.shared_detached_batch = batch_result.batch
                slot.active_uids = (
                    {handle.uid for handle in batch_result.batch.handles}
                    if batch_result.batch is not None
                    else set()
                )
            if not slot.active_uids and not slot.pending_requests:
                slot.service_debt = 0.0
            self._invalidate_slot_cache_if_composition_changed(slot, previous_active_uids)
            self.engine._update_sequence_count_locked()
            self._debug_assert_slot_request_coherence(slot)
            self._debug_assert_finished_requests_released(slot, finished_uids)
        logger.debug(
            f"slot_step_done adapter={slot.adapter_id} responses={len(responses)} "
            f"uids={resp_uids} reasons={resp_reasons} next_ms={step_ms:.1f} "
            f"lock_wait_ms={lock_wait_ms:.1f}"
        )
