from __future__ import annotations

from dataclasses import dataclass

from mola.adapter import AdapterSlotBinding
from mola.application.packing import build_layer_slot_pack_views, materialize_layer_slot_packs


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
        assert packs[0].adapter_names == ("rust", "sql")
        assert packs[0].lora_a == ("a-rust", "a-sql")
        assert packs[0].lora_b == ("b-rust", "b-sql")
        assert packs[0].scales == (16.0, 20.0)
