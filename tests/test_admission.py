from mola.application.admission import TokenBudgetAdmissionPolicy


def test_accepts_request_within_budget():
    policy = TokenBudgetAdmissionPolicy(max_inflight_tokens=10)
    decision = policy.evaluate(reserved_tokens=4, request_tokens=5)
    assert decision.accepted is True
    assert decision.projected_tokens == 9
    assert decision.reason is None


def test_rejects_request_over_budget():
    policy = TokenBudgetAdmissionPolicy(max_inflight_tokens=10)
    decision = policy.evaluate(reserved_tokens=7, request_tokens=5)
    assert decision.accepted is False
    assert decision.projected_tokens == 12
    assert "token budget exceeded" in decision.reason
