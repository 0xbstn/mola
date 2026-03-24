"""Tests for adapter context threading."""

from mola.context import adapter_context, get_current_adapter, get_current_slot_id


class TestAdapterContext:
    def test_default_is_none(self):
        assert get_current_adapter() is None
        assert get_current_slot_id() is None

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
