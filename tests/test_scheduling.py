from mola.application.scheduling import SlotSchedulingState, WaitingAwareSchedulingPolicy


def test_prioritizes_oldest_unstarted_work():
    policy: WaitingAwareSchedulingPolicy[str] = WaitingAwareSchedulingPolicy()
    ordered = policy.order(
        [
            SlotSchedulingState(
                slot_id="rust",
                active_count=1,
                pending_count=0,
                last_service_ts=10.0,
                last_active_ts=10.0,
                oldest_unstarted_ts=5.0,
            ),
            SlotSchedulingState(
                slot_id="sql",
                active_count=1,
                pending_count=0,
                last_service_ts=11.0,
                last_active_ts=11.0,
                oldest_unstarted_ts=2.0,
            ),
        ]
    )
    assert ordered == ["sql", "rust"]


def test_deprioritizes_fully_served_slots():
    policy: WaitingAwareSchedulingPolicy[str] = WaitingAwareSchedulingPolicy()
    ordered = policy.order(
        [
            SlotSchedulingState(
                slot_id="decode",
                active_count=1,
                pending_count=0,
                last_service_ts=20.0,
                last_active_ts=20.0,
                oldest_unstarted_ts=None,
            ),
            SlotSchedulingState(
                slot_id="pending",
                active_count=0,
                pending_count=1,
                last_service_ts=0.0,
                last_active_ts=5.0,
                oldest_unstarted_ts=1.0,
            ),
        ]
    )
    assert ordered == ["pending", "decode"]
