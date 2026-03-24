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

    def close(self) -> None: ...
