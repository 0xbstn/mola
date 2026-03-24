from __future__ import annotations

from dataclasses import dataclass


class AdmissionRejected(Exception):
    pass


@dataclass(frozen=True)
class AdmissionDecision:
    accepted: bool
    projected_tokens: int
    reason: str | None = None


class TokenBudgetAdmissionPolicy:
    def __init__(self, max_inflight_tokens: int):
        self.max_inflight_tokens = max_inflight_tokens

    def evaluate(self, *, reserved_tokens: int, request_tokens: int) -> AdmissionDecision:
        projected = reserved_tokens + request_tokens
        if projected > self.max_inflight_tokens:
            return AdmissionDecision(
                accepted=False,
                projected_tokens=projected,
                reason="Server overloaded: token budget exceeded, try again later",
            )
        return AdmissionDecision(accepted=True, projected_tokens=projected)
