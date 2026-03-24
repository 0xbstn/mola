from mola.application.packing import MaterializedLayerSlotPack, RoutedLayerPackState
from mola.application.routed_decode import FrozenRoutedLayerExecution, RoutedDecodeContractError, freeze_routed_decode_layer_abi, freeze_routed_layer_execution, resolve_routed_layer_execution


class TestFreezeRoutedDecodeLayerABI:
    def _valid_pack(self):
        return MaterializedLayerSlotPack(
            layer_name="layers.0.q_proj",
            slot_ids=(1, 2),
            adapter_names=("rust", "sql"),
            lora_a=__import__("numpy").array(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[2.0, 0.0], [0.0, 2.0]],
                ]
            ),
            lora_b=__import__("numpy").array(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ]
            ),
            scales=__import__("numpy").array([16.0, 20.0]),
        )

    def test_freezes_expected_dims_and_slot_dtype(self):
        abi = freeze_routed_decode_layer_abi(self._valid_pack())

        assert abi.layer_name == "layers.0.q_proj"
        assert abi.slot_ids == (1, 2)
        assert abi.slot_count == 2
        assert abi.input_dim == 2
        assert abi.rank == 2
        assert abi.output_dim == 2
        assert abi.slot_id_dtype == "int32"
        assert abi.require_row_contiguous is True

    def test_rejects_unsorted_slot_ids(self):
        pack = self._valid_pack()
        pack = MaterializedLayerSlotPack(
            layer_name=pack.layer_name,
            slot_ids=(2, 1),
            adapter_names=pack.adapter_names,
            lora_a=pack.lora_a,
            lora_b=pack.lora_b,
            scales=pack.scales,
        )

        try:
            freeze_routed_decode_layer_abi(pack)
        except RoutedDecodeContractError as exc:
            assert "slot_ids must be sorted" in str(exc)
        else:
            raise AssertionError("unsorted slot_ids should be rejected")

    def test_rejects_rank_mismatch(self):
        pack = self._valid_pack()
        bad_b = __import__("numpy").zeros((2, 3, 2))
        pack = MaterializedLayerSlotPack(
            layer_name=pack.layer_name,
            slot_ids=pack.slot_ids,
            adapter_names=pack.adapter_names,
            lora_a=pack.lora_a,
            lora_b=bad_b,
            scales=pack.scales,
        )

        try:
            freeze_routed_decode_layer_abi(pack)
        except RoutedDecodeContractError as exc:
            assert "rank mismatch" in str(exc)
        else:
            raise AssertionError("rank mismatch should be rejected")

    def test_rejects_bad_scales_shape(self):
        pack = self._valid_pack()
        bad_scales = __import__("numpy").zeros((2, 1))
        pack = MaterializedLayerSlotPack(
            layer_name=pack.layer_name,
            slot_ids=pack.slot_ids,
            adapter_names=pack.adapter_names,
            lora_a=pack.lora_a,
            lora_b=pack.lora_b,
            scales=bad_scales,
        )

        try:
            freeze_routed_decode_layer_abi(pack)
        except RoutedDecodeContractError as exc:
            assert "expects scales with shape [S]" in str(exc)
        else:
            raise AssertionError("bad scales shape should be rejected")


class TestFreezeRoutedLayerExecution:
    def _valid_pack(self):
        return TestFreezeRoutedDecodeLayerABI()._valid_pack()

    def test_freezes_plan_and_token_slot_ids(self):
        execution = freeze_routed_layer_execution(self._valid_pack(), (2, 1, 2))

        assert isinstance(execution, FrozenRoutedLayerExecution)
        assert execution.token_slot_ids == (2, 1, 2)
        assert execution.plan.sorted_token_rows == (1, 0, 2)
        assert execution.plan.restore_order == (1, 0, 2)
        assert execution.abi.slot_ids == (1, 2)

    def test_rejects_non_int_slot_ids(self):
        try:
            freeze_routed_layer_execution(self._valid_pack(), (1, "2"))
        except RoutedDecodeContractError as exc:
            assert "plain ints" in str(exc)
        else:
            raise AssertionError("non-int slot ids should be rejected")

    def test_rejects_unknown_slot_ids(self):
        try:
            freeze_routed_layer_execution(self._valid_pack(), (1, 9))
        except RoutedDecodeContractError as exc:
            assert "slot_id=9" in str(exc)
        else:
            raise AssertionError("unknown slot ids should be rejected")


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