"""Tests for adapter loading and management."""

import pytest

from mola.adapter import AdapterManager


class TestAdapterManager:
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
