"""Tests for adapter context threading."""

from mola.application.packing import RoutedLayerPackState
from mola.context import adapter_context, get_current_adapter, get_current_routed_pack_state, get_current_slot_id, get_current_token_slot_ids, routed_decode_context


class TestAdapterContext:
    def test_default_is_none(self):
        assert get_current_adapter() is None
        assert get_current_slot_id() is None
        assert get_current_routed_pack_state() is None
        assert get_current_token_slot_ids() is None

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

    def test_routed_decode_context_sets_state(self):
        state = RoutedLayerPackState(packs_by_layer={})
        with routed_decode_context(state, (1, 2, 1)):
            assert get_current_routed_pack_state() is state
            assert get_current_token_slot_ids() == (1, 2, 1)
        assert get_current_routed_pack_state() is None
        assert get_current_token_slot_ids() is None

    def test_nested_routed_decode_contexts_restore_outer_state(self):
        outer = RoutedLayerPackState(packs_by_layer={"outer": object()})
        inner = RoutedLayerPackState(packs_by_layer={"inner": object()})
        with routed_decode_context(outer, (1, 1)):
            assert get_current_routed_pack_state() is outer
            assert get_current_token_slot_ids() == (1, 1)
            with routed_decode_context(inner, (2,)):
                assert get_current_routed_pack_state() is inner
                assert get_current_token_slot_ids() == (2,)
            assert get_current_routed_pack_state() is outer
            assert get_current_token_slot_ids() == (1, 1)
        assert get_current_routed_pack_state() is None
        assert get_current_token_slot_ids() is None
