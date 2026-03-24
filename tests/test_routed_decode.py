from mola.application.packing import MaterializedLayerSlotPack, RoutedLayerPackState
from mola.application.routed_decode import RoutedDecodeContractError, resolve_routed_layer_execution


class TestResolveRoutedLayerExecution:
    def _pack_state(self):
        return RoutedLayerPackState(
            packs_by_layer={
                "layers.0.q_proj": MaterializedLayerSlotPack(
                    layer_name="layers.0.q_proj",
                    slot_ids=(1, 2),
                    adapter_names=("rust", "sql"),
                    lora_a=("a-rust", "a-sql"),
                    lora_b=("b-rust", "b-sql"),
                    scales=(16.0, 20.0),
                )
            }
        )

    def test_returns_none_without_layer_name(self):
        assert resolve_routed_layer_execution(None, (2, 8), self._pack_state(), (1, 1)) is None

    def test_returns_none_when_pack_missing(self):
        assert resolve_routed_layer_execution(
            "layers.1.q_proj",
            (2, 8),
            self._pack_state(),
            (1, 1),
        ) is None

    def test_raises_when_pack_missing_in_strict_mode(self):
        try:
            resolve_routed_layer_execution(
                "layers.1.q_proj",
                (2, 8),
                self._pack_state(),
                (1, 1),
                strict=True,
            )
        except RoutedDecodeContractError as exc:
            assert "missing routed layer pack" in str(exc)
        else:
            raise AssertionError("strict mode should raise on missing pack")

    def test_returns_none_when_row_count_mismatches_token_slots(self):
        assert resolve_routed_layer_execution(
            "layers.0.q_proj",
            (2, 8),
            self._pack_state(),
            (1,),
        ) is None

    def test_raises_when_row_count_mismatches_token_slots_in_strict_mode(self):
        try:
            resolve_routed_layer_execution(
                "layers.0.q_proj",
                (2, 8),
                self._pack_state(),
                (1,),
                strict=True,
            )
        except RoutedDecodeContractError as exc:
            assert "row count mismatch" in str(exc)
        else:
            raise AssertionError("strict mode should raise on row mismatch")

    def test_resolves_pack_and_token_slot_ids(self):
        execution = resolve_routed_layer_execution(
            "layers.0.q_proj",
            (2, 8),
            self._pack_state(),
            (2, 1),
        )

        assert execution is not None
        assert execution.pack.adapter_names == ("rust", "sql")
        assert execution.token_slot_ids == (2, 1)