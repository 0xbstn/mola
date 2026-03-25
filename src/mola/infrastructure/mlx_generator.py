from __future__ import annotations

from collections.abc import Sequence
import numpy as np

from mola.ports.generator import (
    GeneratorDetachedBatch,
    GeneratorDetachedBatchStepResult,
    GeneratorBatchSnapshot,
    GeneratorBatchStepResult,
    GenerationEvent,
    GeneratorHandle,
    GeneratorPort,
    GeneratorSubmission,
    GeneratorState,
)


def _load_batch_generator_cls():
    from mlx_lm.generate import BatchGenerator

    return BatchGenerator


def _load_batch_state_symbols():
    from mlx_lm.generate import Batch, _merge_caches

    return Batch, _merge_caches


class MLXBatchGeneratorPort(GeneratorPort):
    def __init__(
        self,
        model,
        *,
        max_tokens: int,
        stop_tokens: set[int],
        completion_batch_size: int,
        prefill_batch_size: int,
    ):
        batch_generator_cls = _load_batch_generator_cls()
        self._generator = batch_generator_cls(
            model,
            max_tokens=max_tokens,
            stop_tokens=stop_tokens,
            completion_batch_size=completion_batch_size,
            prefill_batch_size=prefill_batch_size,
        )

    def _supports_public_detached_api(self) -> bool:
        return all(
            hasattr(self._generator, name)
            for name in (
                "detach_active_batch",
                "restore_detached_batch",
                "step_detached_batch",
                "promote_detached_batch",
            )
        )

    def _build_batch(
        self,
        states: Sequence[GeneratorState],
        *,
        preserve_uids: bool,
        existing_uids: Sequence[int] = (),
    ):
        if not states:
            return None

        Batch, merge_caches = _load_batch_state_symbols()
        import mlx.core as mx

        if preserve_uids:
            handles = [state.handle for state in states]
        else:
            occupied_uids = set(existing_uids)
            if self._generator.active_batch is not None:
                occupied_uids.update(self._generator.active_batch.uids)
            next_uid = max(
                getattr(self._generator, "uid_count", 0),
                (max(occupied_uids) + 1) if occupied_uids else 0,
            )
            handles = [
                GeneratorHandle(uid=next_uid + index) for index in range(len(states))
            ]
            if hasattr(self._generator, "uid_count"):
                self._generator.uid_count = next_uid + len(states)

        token_dtype = np.int32
        if hasattr(states[0].tokens, "tolist"):
            token_dtype = np.asarray(states[0].tokens.tolist()).dtype
        y_values = np.array([state.next_token for state in states], dtype=token_dtype)
        batch = Batch(
            uids=[handle.uid for handle in handles],
            y=mx.array(y_values),
            logprobs=[state.logprobs for state in states],
            max_tokens=[state.max_tokens for state in states],
            num_tokens=[state.num_tokens for state in states],
            cache=merge_caches([state.cache for state in states]),
            samplers=[state.sampler for state in states],
            logits_processors=[state.logits_processors for state in states],
            tokens=[state.tokens for state in states],
        )
        return batch, handles

    def _snapshot_from_batch(self, batch) -> GeneratorBatchSnapshot:
        states: list[GeneratorState] = []
        for index, uid in enumerate(batch.uids):
            next_token = batch.y[index]
            if hasattr(next_token, "item"):
                next_token = int(next_token.item())
            else:
                next_token = int(next_token)
            states.append(
                GeneratorState(
                    handle=GeneratorHandle(uid=uid),
                    next_token=next_token,
                    logprobs=batch.logprobs[index],
                    max_tokens=batch.max_tokens[index],
                    num_tokens=batch.num_tokens[index],
                    cache=batch.extract_cache(index),
                    sampler=batch.samplers[index],
                    logits_processors=batch.logits_processors[index],
                    tokens=batch.tokens[index],
                )
            )
        return GeneratorBatchSnapshot(states=tuple(states))

    def _handles_from_batch(self, batch) -> tuple[GeneratorHandle, ...]:
        return tuple(GeneratorHandle(uid=uid) for uid in batch.uids)

    def submit_batch(
        self, requests: Sequence[GeneratorSubmission]
    ) -> list[GeneratorHandle]:
        samplers = None
        if any(req.sampler is not None for req in requests):
            samplers = [req.sampler for req in requests]
        uids = self._generator.insert(
            prompts=[req.prompt_tokens for req in requests],
            max_tokens=[req.max_tokens for req in requests],
            samplers=samplers,
        )
        return [GeneratorHandle(uid=uid) for uid in uids]

    def active_handles(self) -> tuple[GeneratorHandle, ...]:
        batch = self._generator.active_batch
        if batch is None:
            return ()
        return tuple(GeneratorHandle(uid=uid) for uid in batch.uids)

    def step(self) -> list[GenerationEvent]:
        return [
            GenerationEvent(
                handle=GeneratorHandle(uid=response.uid),
                token=response.token,
                finish_reason=response.finish_reason,
            )
            for response in self._generator.next()
        ]

    def cancel(self, handles: Sequence[GeneratorHandle]) -> None:
        self._generator.remove([handle.uid for handle in handles])

    def take_states(
        self, handles: Sequence[GeneratorHandle]
    ) -> list[GeneratorState]:
        if not handles:
            return []
        batch = self._generator.active_batch
        if batch is None:
            return []

        index_by_uid = {uid: index for index, uid in enumerate(batch.uids)}
        missing = [handle.uid for handle in handles if handle.uid not in index_by_uid]
        if missing:
            raise ValueError(f"cannot extract unknown active handle(s): {missing}")

        states: list[GeneratorState] = []
        for handle in handles:
            index = index_by_uid[handle.uid]
            next_token = batch.y[index]
            if hasattr(next_token, "item"):
                next_token = int(next_token.item())
            else:
                next_token = int(next_token)
            states.append(
                GeneratorState(
                    handle=handle,
                    next_token=next_token,
                    logprobs=batch.logprobs[index],
                    max_tokens=batch.max_tokens[index],
                    num_tokens=batch.num_tokens[index],
                    cache=batch.extract_cache(index),
                    sampler=batch.samplers[index],
                    logits_processors=batch.logits_processors[index],
                    tokens=batch.tokens[index],
                )
            )

        self._generator.remove([handle.uid for handle in handles])
        return states

    def restore_states(
        self, states: Sequence[GeneratorState]
    ) -> list[GeneratorHandle]:
        if not states:
            return []
        batch, restored_handles = self._build_batch(states, preserve_uids=False)

        if self._generator.active_batch is None:
            self._generator.active_batch = batch
        else:
            self._generator.active_batch.extend(batch)

        return restored_handles

    def take_active_batch(self) -> GeneratorBatchSnapshot | None:
        handles = self.active_handles()
        if not handles:
            return None
        return GeneratorBatchSnapshot(states=tuple(self.take_states(handles)))

    def restore_active_batch(
        self, batch: GeneratorBatchSnapshot
    ) -> tuple[GeneratorHandle, ...]:
        if not batch.states:
            return ()
        return tuple(self.restore_states(batch.states))

    def detach_active_batch(self) -> GeneratorDetachedBatch | None:
        if self._supports_public_detached_api():
            detached = self._generator.detach_active_batch()
            if detached is None:
                return None
            return GeneratorDetachedBatch(
                handles=tuple(GeneratorHandle(uid=uid) for uid in detached.uids),
                backend_batch=detached,
            )
        batch = self._generator.active_batch
        if batch is None:
            return None
        self._generator.active_batch = None
        return GeneratorDetachedBatch(
            handles=self._handles_from_batch(batch),
            backend_batch=batch,
        )

    def restore_detached_batch(self, batch: GeneratorDetachedBatch) -> None:
        if not batch.handles:
            return
        if self._supports_public_detached_api():
            self._generator.restore_detached_batch(batch.backend_batch)
            return
        if self._generator.active_batch is not None:
            raise RuntimeError("cannot restore detached batch while active_batch is populated")
        self._generator.active_batch = batch.backend_batch

    def snapshot_detached_batch(
        self, batch: GeneratorDetachedBatch
    ) -> GeneratorBatchSnapshot:
        if self._supports_public_detached_api() and hasattr(
            self._generator, "snapshot_detached_batch"
        ):
            raw_batch = self._generator.snapshot_detached_batch(batch.backend_batch)
            return self._snapshot_from_batch(raw_batch)
        return self._snapshot_from_batch(batch.backend_batch)

    def extend_detached_batch(
        self,
        batch: GeneratorDetachedBatch | None,
        incoming: GeneratorBatchSnapshot,
    ) -> tuple[GeneratorDetachedBatch, tuple[GeneratorHandle, ...]]:
        if not incoming.states:
            if batch is None:
                raise ValueError("cannot create detached batch from empty snapshot")
            return batch, ()

        existing_batch = batch.backend_batch if batch is not None else None
        existing_uids = existing_batch.uids if existing_batch is not None else ()
        extension_batch, restored_handles = self._build_batch(
            incoming.states,
            preserve_uids=False,
            existing_uids=existing_uids,
        )
        if existing_batch is None:
            merged_batch = extension_batch
        else:
            existing_batch.extend(extension_batch)
            merged_batch = existing_batch
        return (
            GeneratorDetachedBatch(
                handles=self._handles_from_batch(merged_batch),
                backend_batch=merged_batch,
            ),
            tuple(restored_handles),
        )

    def promote_detached_batch(
        self,
        batch: GeneratorDetachedBatch | None,
        incoming: GeneratorDetachedBatch,
    ) -> tuple[GeneratorDetachedBatch, tuple[GeneratorHandle, ...]]:
        if self._supports_public_detached_api():
            detached_batch, promoted_uids = self._generator.promote_detached_batch(
                batch.backend_batch if batch is not None else None,
                incoming.backend_batch,
            )
            return (
                GeneratorDetachedBatch(
                    handles=tuple(
                        GeneratorHandle(uid=uid) for uid in detached_batch.uids
                    ),
                    backend_batch=detached_batch,
                ),
                tuple(GeneratorHandle(uid=uid) for uid in promoted_uids),
            )
        if not incoming.handles:
            if batch is None:
                raise ValueError("cannot create detached batch from empty detached source")
            return batch, ()

        incoming_batch = incoming.backend_batch
        existing_batch = batch.backend_batch if batch is not None else None
        existing_uids = existing_batch.uids if existing_batch is not None else ()
        next_uid = max(
            getattr(self._generator, "uid_count", 0),
            (max(existing_uids) + 1) if existing_uids else 0,
        )
        promoted_handles = tuple(
            GeneratorHandle(uid=next_uid + index)
            for index in range(len(incoming_batch.uids))
        )
        incoming_batch.uids = [handle.uid for handle in promoted_handles]
        if hasattr(self._generator, "uid_count"):
            self._generator.uid_count = next_uid + len(promoted_handles)

        if existing_batch is None:
            merged_batch = incoming_batch
        else:
            existing_batch.extend(incoming_batch)
            merged_batch = existing_batch

        return (
            GeneratorDetachedBatch(
                handles=self._handles_from_batch(merged_batch),
                backend_batch=merged_batch,
            ),
            promoted_handles,
        )

    def step_batch(
        self, batch: GeneratorBatchSnapshot
    ) -> GeneratorBatchStepResult:
        if not batch.states:
            return GeneratorBatchStepResult(batch=None, events=())

        import mlx.core as mx

        detached_batch, _handles = self._build_batch(batch.states, preserve_uids=True)
        y = detached_batch.y
        for i, toks in enumerate(detached_batch.tokens):
            next_token = y[i : i + 1]
            if isinstance(next_token, np.ndarray):
                detached_batch.tokens[i] = np.concatenate(
                    [np.asarray(toks), next_token]
                )
            else:
                detached_batch.tokens[i] = mx.concatenate(
                    [mx.array(toks), next_token]
                )
        detached_batch.y, detached_batch.logprobs = self._generator._step(
            y[:, None],
            detached_batch.cache,
            detached_batch.samplers,
            detached_batch.logits_processors,
            detached_batch.tokens,
        )
        mx.async_eval(detached_batch.y, detached_batch.logprobs, detached_batch.tokens)

        y_list = y.tolist()
        keep_idx: list[int] = []
        events: list[GenerationEvent] = []

        for e, (token, uid, num_tok, max_tok) in enumerate(
            zip(
                y_list,
                detached_batch.uids,
                detached_batch.num_tokens,
                detached_batch.max_tokens,
            )
        ):
            num_tok += 1
            detached_batch.num_tokens[e] = num_tok
            if token in self._generator.stop_tokens:
                finish_reason = "stop"
            elif num_tok >= max_tok:
                finish_reason = "length"
            else:
                finish_reason = None
                keep_idx.append(e)
            events.append(
                GenerationEvent(
                    handle=GeneratorHandle(uid=uid),
                    token=token,
                    finish_reason=finish_reason,
                )
            )

        next_batch: GeneratorBatchSnapshot | None = None
        if keep_idx:
            detached_batch.filter(keep_idx)
            next_batch = self._snapshot_from_batch(detached_batch)

        return GeneratorBatchStepResult(
            batch=next_batch,
            events=tuple(events),
        )

    def step_detached_batch(
        self, batch: GeneratorDetachedBatch
    ) -> GeneratorDetachedBatchStepResult:
        if not batch.handles:
            return GeneratorDetachedBatchStepResult(batch=None, events=())

        if self._supports_public_detached_api():
            detached_batch, responses = self._generator.step_detached_batch(batch.backend_batch)
            next_batch = (
                GeneratorDetachedBatch(
                    handles=tuple(
                        GeneratorHandle(uid=uid) for uid in detached_batch.uids
                    ),
                    backend_batch=detached_batch,
                )
                if detached_batch is not None
                else None
            )
            return GeneratorDetachedBatchStepResult(
                batch=next_batch,
                events=tuple(
                    GenerationEvent(
                        handle=GeneratorHandle(uid=response.uid),
                        token=response.token,
                        finish_reason=response.finish_reason,
                    )
                    for response in responses
                ),
            )

        import mlx.core as mx

        detached_batch = batch.backend_batch
        y = detached_batch.y
        for i, toks in enumerate(detached_batch.tokens):
            next_token = y[i : i + 1]
            if isinstance(next_token, np.ndarray):
                detached_batch.tokens[i] = np.concatenate(
                    [np.asarray(toks), next_token]
                )
            else:
                detached_batch.tokens[i] = mx.concatenate(
                    [mx.array(toks), next_token]
                )
        detached_batch.y, detached_batch.logprobs = self._generator._step(
            y[:, None],
            detached_batch.cache,
            detached_batch.samplers,
            detached_batch.logits_processors,
            detached_batch.tokens,
        )
        mx.async_eval(detached_batch.y, detached_batch.logprobs, detached_batch.tokens)

        y_list = y.tolist()
        keep_idx: list[int] = []
        events: list[GenerationEvent] = []

        for e, (token, uid, num_tok, max_tok) in enumerate(
            zip(
                y_list,
                detached_batch.uids,
                detached_batch.num_tokens,
                detached_batch.max_tokens,
            )
        ):
            num_tok += 1
            detached_batch.num_tokens[e] = num_tok
            if token in self._generator.stop_tokens:
                finish_reason = "stop"
            elif num_tok >= max_tok:
                finish_reason = "length"
            else:
                finish_reason = None
                keep_idx.append(e)
            events.append(
                GenerationEvent(
                    handle=GeneratorHandle(uid=uid),
                    token=token,
                    finish_reason=finish_reason,
                )
            )

        next_batch: GeneratorDetachedBatch | None = None
        if keep_idx:
            detached_batch.filter(keep_idx)
            next_batch = GeneratorDetachedBatch(
                handles=self._handles_from_batch(detached_batch),
                backend_batch=detached_batch,
            )

        return GeneratorDetachedBatchStepResult(
            batch=next_batch,
            events=tuple(events),
        )

    def close(self) -> None:
        self._generator.close()
