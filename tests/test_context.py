"""Tests for adapter context threading."""

from mola.context import adapter_context, get_current_adapter, get_current_neutralize_lora_delta, get_current_routed_decode_session, get_current_slot_id, lora_delta_context, routed_decode_context


class TestAdapterContext:
    def test_default_is_none(self):
        assert get_current_adapter() is None
        assert get_current_slot_id() is None
        assert get_current_routed_decode_session() is None
        assert get_current_neutralize_lora_delta() is False

    def test_context_sets_adapter(self):
        with adapter_context("solana", slot_id=3):
            assert get_current_adapter() == "solana"
            assert get_current_slot_id() == 3
        assert get_current_adapter() is None
        assert get_current_slot_id() is None

    def test_nested_contexts(self):
        with adapter_context("solana", slot_id=3):
            assert get_current_adapter() == "solana"
            assert get_current_slot_id() == 3
            with adapter_context("code", slot_id=7):
                assert get_current_adapter() == "code"
                assert get_current_slot_id() == 7
            assert get_current_adapter() == "solana"
            assert get_current_slot_id() == 3
        assert get_current_adapter() is None
        assert get_current_slot_id() is None

    def test_none_adapter(self):
        with adapter_context(None):
            assert get_current_adapter() is None
            assert get_current_slot_id() is None

    def test_routed_decode_context_sets_session(self):
        session = object()
        with routed_decode_context(session):
            assert get_current_routed_decode_session() is session
        assert get_current_routed_decode_session() is None

    def test_nested_routed_decode_contexts_restore_outer_session(self):
        outer = object()
        inner = object()
        with routed_decode_context(outer):
            assert get_current_routed_decode_session() is outer
            with routed_decode_context(inner):
                assert get_current_routed_decode_session() is inner
            assert get_current_routed_decode_session() is outer
        assert get_current_routed_decode_session() is None

    def test_lora_delta_context_sets_flag(self):
        with lora_delta_context(neutralize=True):
            assert get_current_neutralize_lora_delta() is True
        assert get_current_neutralize_lora_delta() is False

    def test_nested_lora_delta_contexts_restore_outer_flag(self):
        with lora_delta_context(neutralize=True):
            assert get_current_neutralize_lora_delta() is True
            with lora_delta_context(neutralize=False):
                assert get_current_neutralize_lora_delta() is False
            assert get_current_neutralize_lora_delta() is True
        assert get_current_neutralize_lora_delta() is False
