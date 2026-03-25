from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class GeneratorSubmission:
    prompt_tokens: list[int]
    max_tokens: int
    sampler: object | None


@dataclass(frozen=True)
class GeneratorHandle:
    uid: int


@dataclass(frozen=True)
class GenerationEvent:
    handle: GeneratorHandle
    token: int
    finish_reason: str | None


@dataclass(frozen=True)
class GeneratorState:
    handle: GeneratorHandle
    next_token: int
    logprobs: object
    max_tokens: int
    num_tokens: int
    cache: object
    sampler: object | None
    logits_processors: object
    tokens: object


@dataclass(frozen=True)
class GeneratorBatchSnapshot:
    """Decode-row-ordered snapshot of a generator active batch."""

    states: tuple[GeneratorState, ...]


@dataclass(frozen=True)
class GeneratorDetachedBatch:
    handles: tuple[GeneratorHandle, ...]
    opaque: object


@dataclass(frozen=True)
class GeneratorBatchStepResult:
    batch: GeneratorBatchSnapshot | None
    events: tuple[GenerationEvent, ...]


@dataclass(frozen=True)
class GeneratorDetachedBatchStepResult:
    batch: GeneratorDetachedBatch | None
    events: tuple[GenerationEvent, ...]


class GeneratorPort(Protocol):
    def submit_batch(
        self, requests: Sequence[GeneratorSubmission]
    ) -> list[GeneratorHandle]: ...

    def active_handles(self) -> tuple[GeneratorHandle, ...]:
        """Return handles in the decode row order that the next step() will use."""

    def step(self) -> list[GenerationEvent]: ...

    def cancel(self, handles: Sequence[GeneratorHandle]) -> None: ...

    def take_states(self, handles: Sequence[GeneratorHandle]) -> list[GeneratorState]: ...

    def restore_states(
        self, states: Sequence[GeneratorState]
    ) -> list[GeneratorHandle]: ...

    def take_active_batch(self) -> GeneratorBatchSnapshot | None: ...

    def restore_active_batch(
        self, batch: GeneratorBatchSnapshot
    ) -> tuple[GeneratorHandle, ...]: ...

    def take_active_batch_handle(self) -> GeneratorDetachedBatch | None: ...

    def restore_detached_batch(self, batch: GeneratorDetachedBatch) -> None: ...

    def snapshot_detached_batch(
        self, batch: GeneratorDetachedBatch
    ) -> GeneratorBatchSnapshot: ...

    def extend_detached_batch(
        self,
        batch: GeneratorDetachedBatch | None,
        incoming: GeneratorBatchSnapshot,
    ) -> tuple[GeneratorDetachedBatch, tuple[GeneratorHandle, ...]]: ...

    def promote_detached_batch(
        self,
        batch: GeneratorDetachedBatch | None,
        incoming: GeneratorDetachedBatch,
    ) -> tuple[GeneratorDetachedBatch, tuple[GeneratorHandle, ...]]: ...

    def step_batch(
        self, batch: GeneratorBatchSnapshot
    ) -> GeneratorBatchStepResult: ...

    def step_detached_batch(
        self, batch: GeneratorDetachedBatch
    ) -> GeneratorDetachedBatchStepResult: ...

    def close(self) -> None: ...
