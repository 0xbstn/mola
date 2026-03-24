"""Tests for adapter loading and management."""

from unittest.mock import patch

import pytest

from mola.adapter import AdapterConfig, AdapterManager, AdapterSlotBinding, AdapterWeights


class TestAdapterManager:
    def _fake_load(self, mgr: AdapterManager):
        config = AdapterConfig(
            rank=8,
            scale=16.0,
            dropout=0.0,
            num_layers=1,
            target_modules=["self_attn.q_proj"],
        )
        weights = AdapterWeights(weights={})
        return patch("mola.adapter.AdapterConfig.from_file", return_value=config), patch.object(
            mgr, "_load_weights", return_value=weights
        )

    def test_empty_manager(self):
        mgr = AdapterManager()
        assert mgr.list_adapters() == []

    def test_max_adapters_enforced(self):
        mgr = AdapterManager(max_adapters=0)
        with pytest.raises(RuntimeError, match="Max adapters"):
            mgr.load("test", "/nonexistent")

    def test_unload_unknown_raises(self):
        mgr = AdapterManager()
        with pytest.raises(KeyError):
            mgr.unload("nonexistent")

    def test_assigns_slot_ids_and_exposes_them(self):
        mgr = AdapterManager()
        cfg_patch, weights_patch = self._fake_load(mgr)
        with cfg_patch, weights_patch:
            a1 = mgr.load("rust", "/fake/rust")
            a2 = mgr.load("sql", "/fake/sql")

        assert a1.slot_id == 0
        assert a2.slot_id == 1
        assert mgr.slot_id("rust") == 0
        assert mgr.slot_id("sql") == 1
        assert mgr.name_for_slot_id(0) == "rust"
        assert mgr.name_for_slot_id(1) == "sql"
        assert mgr.slot_id(None) is None
        assert mgr.slot_id("missing") is None
        assert mgr.name_for_slot_id(None) is None
        assert mgr.name_for_slot_id(99) is None
        assert mgr.slot_bindings() == [
            AdapterSlotBinding(
                name="rust",
                slot_id=0,
                rank=8,
                scale=16.0,
                num_layers=1,
                target_modules=("self_attn.q_proj",),
                source_path="/fake/rust",
            ),
            AdapterSlotBinding(
                name="sql",
                slot_id=1,
                rank=8,
                scale=16.0,
                num_layers=1,
                target_modules=("self_attn.q_proj",),
                source_path="/fake/sql",
            ),
        ]
        assert mgr.list_adapters() == [
            {
                "name": "rust",
                "slot_id": 0,
                "rank": 8,
                "scale": 16.0,
                "num_layers": 1,
                "target_modules": ["self_attn.q_proj"],
                "memory_mb": 0.0,
                "source": "/fake/rust",
            },
            {
                "name": "sql",
                "slot_id": 1,
                "rank": 8,
                "scale": 16.0,
                "num_layers": 1,
                "target_modules": ["self_attn.q_proj"],
                "memory_mb": 0.0,
                "source": "/fake/sql",
            },
        ]

    def test_reuses_slot_ids_after_unload(self):
        mgr = AdapterManager()
        cfg_patch, weights_patch = self._fake_load(mgr)
        with cfg_patch, weights_patch:
            first = mgr.load("rust", "/fake/rust")
            second = mgr.load("sql", "/fake/sql")
        assert first.slot_id == 0
        assert second.slot_id == 1

        mgr.unload("rust")
        assert mgr.name_for_slot_id(0) is None

        cfg_patch, weights_patch = self._fake_load(mgr)
        with cfg_patch, weights_patch:
            third = mgr.load("legal", "/fake/legal")

        assert third.slot_id == 0
        assert mgr.name_for_slot_id(0) == "legal"
        assert [binding.slot_id for binding in mgr.slot_bindings()] == [0, 1]

    def test_slot_bindings_are_sorted_and_stable(self):
        mgr = AdapterManager()
        cfg_patch, weights_patch = self._fake_load(mgr)
        with cfg_patch, weights_patch:
            mgr.load("rust", "/fake/rust")
            mgr.load("sql", "/fake/sql")

        assert mgr.slot_bindings() == [
            AdapterSlotBinding(
                name="rust",
                slot_id=0,
                rank=8,
                scale=16.0,
                num_layers=1,
                target_modules=("self_attn.q_proj",),
                source_path="/fake/rust",
            ),
            AdapterSlotBinding(
                name="sql",
                slot_id=1,
                rank=8,
                scale=16.0,
                num_layers=1,
                target_modules=("self_attn.q_proj",),
                source_path="/fake/sql",
            ),
        ]
