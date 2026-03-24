"""Tests for lora module: MultiLoRALinear, inject/eject weights, load_adapter rollback."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import pytest

from mola.context import adapter_context
from mola.lora import (
    MultiLoRALinear,
    apply_multi_lora,
    eject_adapter_weights,
    inject_adapter_weights,
)


# --- MultiLoRALinear ---


class TestMultiLoRALinear:
    def _make_layer(self, in_dim=8, out_dim=4):
        base = nn.Linear(in_dim, out_dim)
        return MultiLoRALinear(base)

    def test_forward_no_adapter(self):
        layer = self._make_layer()
        x = mx.ones((1, 8))
        y = layer(x)
        assert y.shape == (1, 4)

    def test_forward_with_adapter(self):
        layer = self._make_layer(8, 4)
        rank = 2
        lora_a = mx.ones((8, rank)) * 0.1
        lora_b = mx.ones((rank, 4)) * 0.1
        layer.add_adapter("test", lora_a, lora_b, scale=1.0)

        x = mx.ones((1, 8))

        # Without adapter context — base only
        y_base = layer(x)

        # With adapter context — base + delta
        with adapter_context("test"):
            y_adapted = layer(x)

        # The adapted output should differ from base
        diff = mx.abs(y_adapted - y_base).sum().item()
        assert diff > 0, "Adapter should change the output"

    def test_switch_between_adapters(self):
        layer = self._make_layer(8, 4)
        rank = 2

        lora_a1 = mx.ones((8, rank)) * 0.1
        lora_b1 = mx.ones((rank, 4)) * 0.1
        layer.add_adapter("a1", lora_a1, lora_b1, scale=1.0)

        lora_a2 = mx.ones((8, rank)) * 0.5
        lora_b2 = mx.ones((rank, 4)) * 0.5
        layer.add_adapter("a2", lora_a2, lora_b2, scale=1.0)

        x = mx.ones((1, 8))

        with adapter_context("a1"):
            y1 = layer(x)
        with adapter_context("a2"):
            y2 = layer(x)

        diff = mx.abs(y1 - y2).sum().item()
        assert diff > 0, "Different adapters should produce different outputs"

    def test_remove_adapter(self):
        layer = self._make_layer()
        lora_a = mx.ones((8, 2)) * 0.1
        lora_b = mx.ones((2, 4)) * 0.1
        layer.add_adapter("tmp", lora_a, lora_b, scale=1.0)
        assert layer.has_adapter("tmp")
        layer.remove_adapter("tmp")
        assert not layer.has_adapter("tmp")

    def test_adapter_names(self):
        layer = self._make_layer()
        assert layer.adapter_names == []
        layer.add_adapter("a", mx.zeros((8, 2)), mx.zeros((2, 4)), 1.0)
        layer.add_adapter("b", mx.zeros((8, 2)), mx.zeros((2, 4)), 1.0)
        assert sorted(layer.adapter_names) == ["a", "b"]

    def test_rejects_rank_mismatch(self):
        layer = self._make_layer(8, 4)
        lora_a = mx.zeros((8, 2))  # rank 2
        lora_b = mx.zeros((3, 4))  # rank 3 — mismatch
        with pytest.raises(ValueError, match="rank mismatch"):
            layer.add_adapter("bad", lora_a, lora_b, 1.0)

    def test_rejects_wrong_input_dim(self):
        layer = self._make_layer(8, 4)
        lora_a = mx.zeros((16, 2))  # input dim 16, base expects 8
        lora_b = mx.zeros((2, 4))
        with pytest.raises(ValueError, match="input dim mismatch"):
            layer.add_adapter("bad", lora_a, lora_b, 1.0)

    def test_rejects_wrong_output_dim(self):
        layer = self._make_layer(8, 4)
        lora_a = mx.zeros((8, 2))
        lora_b = mx.zeros((2, 16))  # output dim 16, base expects 4
        with pytest.raises(ValueError, match="output dim mismatch"):
            layer.add_adapter("bad", lora_a, lora_b, 1.0)


# --- inject/eject with a tiny mock model ---


def _make_tiny_model():
    """Build a minimal nn.Module tree with named Linear layers."""

    class TinyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = type("SA", (nn.Module,), {})()
            self.self_attn.q_proj = nn.Linear(8, 8)
            self.self_attn.v_proj = nn.Linear(8, 8)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [TinyBlock()]

    return TinyModel()


class TestInjectEject:
    def test_apply_and_inject(self):
        model = _make_tiny_model()
        targets = ["self_attn.q_proj", "self_attn.v_proj"]

        apply_multi_lora(model, targets)

        # After surgery, targeted layers should be MultiLoRALinear
        for name, module in model.named_modules():
            if any(name.endswith(t) for t in targets):
                assert isinstance(module, MultiLoRALinear), f"{name} was not wrapped"

        # Inject adapter weights
        rank = 2
        weights = {}
        for name, module in model.named_modules():
            if isinstance(module, MultiLoRALinear):
                weights[name] = (mx.ones((8, rank)) * 0.1, mx.ones((rank, 8)) * 0.1)

        count = inject_adapter_weights(model, "test_adapter", weights, scale=1.0)
        assert count == 2

    def test_eject_removes_adapter(self):
        model = _make_tiny_model()
        targets = ["self_attn.q_proj", "self_attn.v_proj"]
        apply_multi_lora(model, targets)

        weights = {}
        for name, module in model.named_modules():
            if isinstance(module, MultiLoRALinear):
                weights[name] = (mx.ones((8, 2)) * 0.1, mx.ones((2, 8)) * 0.1)

        inject_adapter_weights(model, "adapter_x", weights, scale=1.0)
        ejected = eject_adapter_weights(model, "adapter_x")
        assert ejected == 2

        # Verify adapter is gone
        for _, module in model.named_modules():
            if isinstance(module, MultiLoRALinear):
                assert not module.has_adapter("adapter_x")

    def test_inject_rollback_on_shape_error(self):
        """If add_adapter() raises mid-injection (e.g. shape mismatch on the
        2nd layer), already-injected layers must be cleaned up by eject."""
        model = _make_tiny_model()
        targets = ["self_attn.q_proj", "self_attn.v_proj"]
        apply_multi_lora(model, targets)

        # Build weights where q_proj is correct but v_proj has wrong shape
        weights = {}
        layer_names = []
        for name, module in model.named_modules():
            if isinstance(module, MultiLoRALinear):
                layer_names.append(name)

        # First layer: correct shapes (8, 2) and (2, 8)
        weights[layer_names[0]] = (mx.ones((8, 2)) * 0.1, mx.ones((2, 8)) * 0.1)
        # Second layer: wrong input dim — will raise ValueError in add_adapter
        weights[layer_names[1]] = (mx.ones((999, 2)) * 0.1, mx.ones((2, 8)) * 0.1)

        with pytest.raises(ValueError, match="input dim mismatch"):
            inject_adapter_weights(model, "bad_adapter", weights, scale=1.0)

        # After the error, eject should clean up any partial state
        eject_adapter_weights(model, "bad_adapter")

        # Verify: no layer has the failed adapter
        for _, module in model.named_modules():
            if isinstance(module, MultiLoRALinear):
                assert not module.has_adapter("bad_adapter")


# --- MOLAModel.load_adapter integration test ---


class TestLoadAdapterRollback:
    """Integration test: exercises MOLAModel.load_adapter() through the full
    stack (surgery → inject → rollback) with a real tiny model, mocking only
    mlx_lm.load and the adapter file I/O."""

    def _make_mola_model(self):
        """Build a MOLAModel with a real tiny nn.Module tree."""
        from mola.model import MOLAModel

        tiny = _make_tiny_model()
        tokenizer = MagicMock()

        with patch("mola.model.mlx_lm.load", return_value=(tiny, tokenizer)):
            return MOLAModel("test-model")

    def _make_fake_adapter(self, adapter_mgr, name, weights_dict, target_modules):
        """Register a fake adapter in the manager without touching disk."""
        from mola.adapter import Adapter, AdapterConfig, AdapterWeights

        config = AdapterConfig(
            rank=2, scale=1.0, dropout=0.0, num_layers=1,
            target_modules=target_modules,
        )
        adapter_weights = AdapterWeights(weights=weights_dict)
        adapter = Adapter(name=name, config=config, weights=adapter_weights, source_path="/fake")
        adapter_mgr.adapters[name] = adapter
        return adapter

    def test_successful_load(self):
        mola = self._make_mola_model()

        # Prepare correct weights for both layers
        weights = {}
        for name, module in mola.model.named_modules():
            if isinstance(module, nn.Linear):
                weights[name] = (mx.ones((8, 2)) * 0.1, mx.ones((2, 8)) * 0.1)

        targets = ["self_attn.q_proj", "self_attn.v_proj"]

        with patch.object(mola.adapter_manager, "load") as mock_load:
            mock_load.return_value = self._make_fake_adapter(
                mola.adapter_manager, "good", weights, targets,
            )
            adapter = mola.load_adapter("good", "/fake")

        assert adapter.name == "good"
        assert mola.adapter_manager.get("good") is not None

    def test_rollback_on_shape_mismatch(self):
        mola = self._make_mola_model()

        # First layer correct, second layer wrong shape
        layer_names = [
            name for name, m in mola.model.named_modules()
            if isinstance(m, nn.Linear)
        ]
        weights = {
            layer_names[0]: (mx.ones((8, 2)) * 0.1, mx.ones((2, 8)) * 0.1),
            layer_names[1]: (mx.ones((999, 2)) * 0.1, mx.ones((2, 8)) * 0.1),
        }
        targets = ["self_attn.q_proj", "self_attn.v_proj"]

        with patch.object(mola.adapter_manager, "load") as mock_load:
            mock_load.return_value = self._make_fake_adapter(
                mola.adapter_manager, "bad", weights, targets,
            )
            with pytest.raises(ValueError, match="input dim mismatch"):
                mola.load_adapter("bad", "/fake")

        # Adapter must be fully rolled back
        assert mola.adapter_manager.get("bad") is None
        for _, module in mola.model.named_modules():
            if isinstance(module, MultiLoRALinear):
                assert not module.has_adapter("bad")

    def test_reserved_name_rejected(self):
        mola = self._make_mola_model()
        with pytest.raises(ValueError, match="reserved"):
            mola.load_adapter("base", "/fake")
        with pytest.raises(ValueError, match="reserved"):
            mola.load_adapter("test-model", "/fake")
