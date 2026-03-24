from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mola.adapter import AdapterSlotBinding
from mola.application.packing import build_layer_slot_pack_state, build_layer_slot_pack_views, build_routed_decode_plan, build_slot_row_map, materialize_layer_slot_packs, routed_decode_delta_reference, routed_decode_delta_rows_reference


@dataclass
class FakeLayer:
    bindings: list[tuple[int, str, str, float]]

    @property
    def slot_ids(self) -> tuple[int, ...]:
        return tuple(slot_id for slot_id, *_ in self.bindings)

    def slot_bindings(self) -> list[tuple[int, str, str, float]]:
        return list(self.bindings)


class TestBuildLayerSlotPackViews:
    def test_returns_empty_when_no_active_bindings(self):
        layer = FakeLayer([(1, "a1", "b1", 16.0)])
        assert build_layer_slot_pack_views([], [("layers.0.q_proj", layer)]) == ()

    def test_filters_to_active_slots_and_sorts_entries(self):
        active_bindings = [
            AdapterSlotBinding("sql", 2, 8, 16.0, 1, ("q_proj",), "/fake/sql"),
            AdapterSlotBinding("rust", 1, 8, 16.0, 1, ("q_proj",), "/fake/rust"),
        ]
        layers = [
            (
                "layers.0.q_proj",
                FakeLayer([
                    (2, "a-sql", "b-sql", 16.0),
                    (3, "a-legal", "b-legal", 16.0),
                    (1, "a-rust", "b-rust", 16.0),
                ]),
            ),
            (
                "layers.1.k_proj",
                FakeLayer([
                    (3, "a-legal", "b-legal", 16.0),
                ]),
            ),
        ]

        views = build_layer_slot_pack_views(active_bindings, layers)

        assert len(views) == 1
        assert views[0].layer_name == "layers.0.q_proj"
        assert views[0].slot_ids == (1, 2)
        assert tuple(entry.adapter_name for entry in views[0].entries) == ("rust", "sql")
        assert tuple(entry.lora_a for entry in views[0].entries) == ("a-rust", "a-sql")

    def test_preserves_layer_order(self):
        active_bindings = [
            AdapterSlotBinding("rust", 1, 8, 16.0, 1, ("q_proj",), "/fake/rust"),
        ]
        layers = [
            ("layers.1", FakeLayer([(1, "a1", "b1", 16.0)])),
            ("layers.7", FakeLayer([(1, "a7", "b7", 16.0)])),
        ]

        views = build_layer_slot_pack_views(active_bindings, layers)

        assert tuple(view.layer_name for view in views) == ("layers.1", "layers.7")

    def test_materialize_layer_slot_packs_uses_stack_fn(self):
        active_bindings = [
            AdapterSlotBinding("rust", 1, 8, 16.0, 1, ("q_proj",), "/fake/rust"),
            AdapterSlotBinding("sql", 2, 8, 16.0, 1, ("q_proj",), "/fake/sql"),
        ]
        views = build_layer_slot_pack_views(
            active_bindings,
            [
                (
                    "layers.0.q_proj",
                    FakeLayer([
                        (2, "a-sql", "b-sql", 20.0),
                        (1, "a-rust", "b-rust", 16.0),
                    ]),
                )
            ],
        )

        packs = materialize_layer_slot_packs(views, stack_fn=lambda values: tuple(values))

        assert len(packs) == 1
        assert packs[0].layer_name == "layers.0.q_proj"
        assert packs[0].slot_ids == (1, 2)
        assert packs[0].slot_row_by_id == {1: 0, 2: 1}
        assert packs[0].adapter_names == ("rust", "sql")
        assert packs[0].lora_a == ("a-rust", "a-sql")
        assert packs[0].lora_b == ("b-rust", "b-sql")
        assert packs[0].scales == (16.0, 20.0)

    def test_materialize_layer_slot_packs_can_materialize_scales(self):
        active_bindings = [
            AdapterSlotBinding("rust", 1, 8, 16.0, 1, ("q_proj",), "/fake/rust"),
            AdapterSlotBinding("sql", 2, 8, 16.0, 1, ("q_proj",), "/fake/sql"),
        ]
        views = build_layer_slot_pack_views(
            active_bindings,
            [
                (
                    "layers.0.q_proj",
                    FakeLayer([
                        (2, "a-sql", "b-sql", 20.0),
                        (1, "a-rust", "b-rust", 16.0),
                    ]),
                )
            ],
        )

        packs = materialize_layer_slot_packs(
            views,
            stack_fn=lambda values: tuple(values),
            scale_fn=lambda values: tuple(value * 2 for value in values),
        )

        assert packs[0].scales == (32.0, 40.0)


class TestSlotRowMap:
    def test_builds_dense_row_lookup(self):
        assert build_slot_row_map((3, 7, 11)) == {3: 0, 7: 1, 11: 2}

    def test_rejects_duplicate_slot_ids(self):
        try:
            build_slot_row_map((3, 7, 3))
        except ValueError as exc:
            assert "duplicate slot_id" in str(exc)
        else:
            raise AssertionError("duplicate slot ids should be rejected")


class TestBuildRoutedDecodePlan:
    def _pack(self):
        active_bindings = [
            AdapterSlotBinding("rust", 1, 8, 16.0, 1, ("q_proj",), "/fake/rust"),
            AdapterSlotBinding("sql", 2, 8, 16.0, 1, ("q_proj",), "/fake/sql"),
            AdapterSlotBinding("legal", 5, 8, 16.0, 1, ("q_proj",), "/fake/legal"),
        ]
        views = build_layer_slot_pack_views(
            active_bindings,
            [
                (
                    "layers.0.q_proj",
                    FakeLayer([
                        (5, "a-legal", "b-legal", 24.0),
                        (2, "a-sql", "b-sql", 20.0),
                        (1, "a-rust", "b-rust", 16.0),
                    ]),
                )
            ],
        )
        return materialize_layer_slot_packs(views, stack_fn=lambda values: tuple(values))[0]

    def test_groups_rows_by_pack_order_and_builds_restore_order(self):
        pack = self._pack()

        plan = build_routed_decode_plan(pack, (5, 1, 5, 2, 1))

        assert plan.token_count == 5
        assert plan.sorted_token_rows == (1, 4, 3, 0, 2)
        assert plan.restore_order == (3, 0, 4, 2, 1)
        assert tuple((group.slot_id, group.pack_row, group.adapter_name, group.token_rows) for group in plan.groups) == (
            (1, 0, "rust", (1, 4)),
            (2, 1, "sql", (3,)),
            (5, 2, "legal", (0, 2)),
        )
        assert plan.homogeneous_slot_id is None

    def test_marks_homogeneous_plan(self):
        pack = self._pack()

        plan = build_routed_decode_plan(pack, (2, 2, 2))

        assert plan.sorted_token_rows == (0, 1, 2)
        assert plan.restore_order == (0, 1, 2)
        assert len(plan.groups) == 1
        assert plan.homogeneous_slot_id == 2

    def test_rejects_unknown_slots(self):
        pack = self._pack()

        try:
            build_routed_decode_plan(pack, (1, 9))
        except ValueError as exc:
            assert "slot_id=9" in str(exc)
        else:
            raise AssertionError("unknown slot ids should be rejected")


class TestRoutedDecodeDeltaReference:
    def test_applies_slot_specific_delta_and_restores_original_order(self):
        active_bindings = [
            AdapterSlotBinding("rust", 1, 1, 2.0, 1, ("q_proj",), "/fake/rust"),
            AdapterSlotBinding("sql", 2, 1, 0.5, 1, ("q_proj",), "/fake/sql"),
        ]
        views = build_layer_slot_pack_views(
            active_bindings,
            [
                (
                    "layers.0.q_proj",
                    FakeLayer([
                        (2, np.array([[0.0], [1.0]]), np.array([[4.0, 1.0]]), 0.5),
                        (1, np.array([[1.0], [0.0]]), np.array([[2.0, 3.0]]), 2.0),
                    ]),
                )
            ],
        )
        pack = materialize_layer_slot_packs(
            views,
            stack_fn=lambda values: np.stack(values, axis=0),
            scale_fn=lambda values: np.array(values),
        )[0]
        x = np.array(
            [
                [10.0, 1.0],
                [2.0, 5.0],
                [7.0, 3.0],
            ]
        )

        delta = routed_decode_delta_reference(
            x,
            pack,
            (2, 1, 2),
            take_rows_fn=lambda array, rows: array[list(rows)],
            concat_fn=lambda chunks: np.concatenate(chunks, axis=0),
        )

        expected = np.array(
            [
                [2.0, 0.5],
                [8.0, 12.0],
                [6.0, 1.5],
            ]
        )
        assert np.allclose(delta, expected)

    def test_rejects_empty_token_batch(self):
        active_bindings = [
            AdapterSlotBinding("rust", 1, 1, 2.0, 1, ("q_proj",), "/fake/rust"),
        ]
        views = build_layer_slot_pack_views(
            active_bindings,
            [
                (
                    "layers.0.q_proj",
                    FakeLayer([
                        (1, np.array([[1.0], [0.0]]), np.array([[2.0, 3.0]]), 2.0),
                    ]),
                )
            ],
        )
        pack = materialize_layer_slot_packs(
            views,
            stack_fn=lambda values: np.stack(values, axis=0),
            scale_fn=lambda values: np.array(values),
        )[0]

        try:
            routed_decode_delta_reference(
                np.zeros((0, 2)),
                pack,
                (),
                take_rows_fn=lambda array, rows: array[list(rows)],
                concat_fn=lambda chunks: np.concatenate(chunks, axis=0),
            )
        except ValueError as exc:
            assert "at least one token row" in str(exc)
        else:
            raise AssertionError("empty routed decode batches should be rejected")

    def test_supports_row_major_flatten_and_restore(self):
        active_bindings = [
            AdapterSlotBinding("rust", 1, 1, 2.0, 1, ("q_proj",), "/fake/rust"),
            AdapterSlotBinding("sql", 2, 1, 0.5, 1, ("q_proj",), "/fake/sql"),
        ]
        views = build_layer_slot_pack_views(
            active_bindings,
            [
                (
                    "layers.0.q_proj",
                    FakeLayer([
                        (2, np.array([[0.0], [1.0]]), np.array([[4.0, 1.0]]), 0.5),
                        (1, np.array([[1.0], [0.0]]), np.array([[2.0, 3.0]]), 2.0),
                    ]),
                )
            ],
        )
        pack = materialize_layer_slot_packs(
            views,
            stack_fn=lambda values: np.stack(values, axis=0),
            scale_fn=lambda values: np.array(values),
        )[0]
        x = np.array(
            [
                [[10.0, 1.0]],
                [[2.0, 5.0]],
                [[7.0, 3.0]],
            ]
        )

        delta = routed_decode_delta_rows_reference(
            x,
            pack,
            (2, 1, 2),
            flatten_fn=lambda array: (array.reshape(-1, array.shape[-1]), array.shape),
            restore_fn=lambda array, shape: array.reshape(shape[:-1] + (array.shape[-1],)),
            take_rows_fn=lambda array, rows: array[list(rows)],
            concat_fn=lambda chunks: np.concatenate(chunks, axis=0),
        )

        expected = np.array(
            [
                [[2.0, 0.5]],
                [[8.0, 12.0]],
                [[6.0, 1.5]],
            ]
        )
        assert np.allclose(delta, expected)


class TestRoutedLayerPackState:
    def test_builds_lookup_by_layer_name(self):
        active_bindings = [
            AdapterSlotBinding("rust", 1, 8, 16.0, 1, ("q_proj",), "/fake/rust"),
            AdapterSlotBinding("sql", 2, 8, 16.0, 1, ("q_proj",), "/fake/sql"),
        ]
        views = build_layer_slot_pack_views(
            active_bindings,
            [
                ("layers.0.q_proj", FakeLayer([(1, "a-rust", "b-rust", 16.0)])),
                ("layers.1.q_proj", FakeLayer([(2, "a-sql", "b-sql", 20.0)])),
            ],
        )
        packs = materialize_layer_slot_packs(views, stack_fn=lambda values: tuple(values))

        state = build_layer_slot_pack_state(packs)

        assert state.get("layers.0.q_proj") is packs[0]
        assert state.get("layers.1.q_proj") is packs[1]
        assert state.get("layers.9.q_proj") is None

    def test_rejects_duplicate_layer_names(self):
        pack = materialize_layer_slot_packs(
            [
                build_layer_slot_pack_views(
                    [AdapterSlotBinding("rust", 1, 8, 16.0, 1, ("q_proj",), "/fake/rust")],
                    [("layers.0.q_proj", FakeLayer([(1, "a-rust", "b-rust", 16.0)]))],
                )[0],
            ],
            stack_fn=lambda values: tuple(values),
        )[0]

        try:
            build_layer_slot_pack_state((pack, pack))
        except ValueError as exc:
            assert "duplicate layer pack" in str(exc)
        else:
            raise AssertionError("duplicate layer names should be rejected")
