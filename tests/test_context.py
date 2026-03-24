"""Tests for adapter context threading."""

from mola.context import adapter_context, get_current_adapter


class TestAdapterContext:
    def test_default_is_none(self):
        assert get_current_adapter() is None

    def test_context_sets_adapter(self):
        with adapter_context("solana"):
            assert get_current_adapter() == "solana"
        assert get_current_adapter() is None

    def test_nested_contexts(self):
        with adapter_context("solana"):
            assert get_current_adapter() == "solana"
            with adapter_context("code"):
                assert get_current_adapter() == "code"
            assert get_current_adapter() == "solana"
        assert get_current_adapter() is None

    def test_none_adapter(self):
        with adapter_context(None):
            assert get_current_adapter() is None
