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


class GeneratorPort(Protocol):
    def submit_batch(
        self, requests: Sequence[GeneratorSubmission]
    ) -> list[GeneratorHandle]: ...

    def step(self) -> list[GenerationEvent]: ...

    def cancel(self, handles: Sequence[GeneratorHandle]) -> None: ...

    def close(self) -> None: ...
