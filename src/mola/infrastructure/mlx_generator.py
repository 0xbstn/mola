from __future__ import annotations

from collections.abc import Sequence
import numpy as np

from mola.ports.generator import (
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

        Batch, merge_caches = _load_batch_state_symbols()
        import mlx.core as mx

        existing_uids = set()
        if self._generator.active_batch is not None:
            existing_uids = set(self._generator.active_batch.uids)
        next_uid = max(
            getattr(self._generator, "uid_count", 0),
            (max(existing_uids) + 1) if existing_uids else 0,
        )
        restored_handles = [
            GeneratorHandle(uid=next_uid + index) for index in range(len(states))
        ]

        token_dtype = np.int32
        if hasattr(states[0].tokens, "tolist"):
            token_dtype = np.asarray(states[0].tokens.tolist()).dtype
        y_values = np.array([state.next_token for state in states], dtype=token_dtype)
        batch = Batch(
            uids=[handle.uid for handle in restored_handles],
            y=mx.array(y_values),
            logprobs=[state.logprobs for state in states],
            max_tokens=[state.max_tokens for state in states],
            num_tokens=[state.num_tokens for state in states],
            cache=merge_caches([state.cache for state in states]),
            samplers=[state.sampler for state in states],
            logits_processors=[state.logits_processors for state in states],
            tokens=[state.tokens for state in states],
        )

        if self._generator.active_batch is None:
            self._generator.active_batch = batch
        else:
            self._generator.active_batch.extend(batch)

        if hasattr(self._generator, "uid_count"):
            self._generator.uid_count = next_uid + len(states)

        return restored_handles

    def close(self) -> None:
        self._generator.close()
