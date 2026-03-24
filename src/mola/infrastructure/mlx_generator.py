from __future__ import annotations

from collections.abc import Sequence

from mola.ports.generator import (
    GenerationEvent,
    GeneratorHandle,
    GeneratorPort,
    GeneratorSubmission,
)


def _load_batch_generator_cls():
    from mlx_lm.generate import BatchGenerator

    return BatchGenerator


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

    def close(self) -> None:
        self._generator.close()
