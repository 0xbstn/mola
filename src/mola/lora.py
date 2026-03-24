"""Multi-LoRA layer implementations and model surgery.

The key technical challenge: mlx-lm's LoRALinear holds ONE adapter.
We need a layer that holds N adapters and picks the right one per request.

Solution: MultiLoRALinear reads the active adapter from context (contextvars)
and applies the corresponding delta. The model architecture code stays untouched.

    # mlx-lm does this (single adapter):
    y = self.linear(x) + scale * (x @ lora_a) @ lora_b

    # MOLA does this (multi adapter, selected by context):
    adapter_id = get_current_adapter()
    lora_a, lora_b, scale = self.adapters[adapter_id]
    y = self.linear(x) + scale * (x @ lora_a) @ lora_b
"""

from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn

from mola.application.packing import routed_decode_delta_rows_reference
from mola.context import get_current_adapter, get_current_routed_pack_state, get_current_slot_id, get_current_token_slot_ids

logger = logging.getLogger(__name__)


class MultiLoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear that supports multiple LoRA adapters.

    Keeps the base linear layer intact. Holds a dict of adapter weights.
    At each forward pass, checks the context for the active adapter
    and applies its delta if present.
    """

    def __init__(self, base_linear: nn.Module, layer_name: str | None = None):
        super().__init__()
        self.linear = base_linear
        self.layer_name = layer_name
        self._adapters: dict[str, tuple[mx.array, mx.array, float]] = {}
        self._adapters_by_slot: dict[int, tuple[mx.array, mx.array, float]] = {}
        self._slot_by_adapter: dict[str, int] = {}

    def add_adapter(
        self,
        name: str,
        lora_a: mx.array,
        lora_b: mx.array,
        scale: float,
        slot_id: int | None = None,
    ):
        if lora_a.shape[1] != lora_b.shape[0]:
            raise ValueError(
                f"LoRA rank mismatch for adapter '{name}': "
                f"lora_a {tuple(lora_a.shape)} vs lora_b {tuple(lora_b.shape)}"
            )
        in_features, out_features = self._base_dims()
        if in_features is not None:
            if lora_a.shape[0] != in_features:
                raise ValueError(
                    f"LoRA input dim mismatch for adapter '{name}': "
                    f"lora_a has {lora_a.shape[0]}, base layer expects {in_features}"
                )
            if lora_b.shape[1] != out_features:
                raise ValueError(
                    f"LoRA output dim mismatch for adapter '{name}': "
                    f"lora_b has {tuple(lora_b.shape)} vs base layer expects {out_features}"
                )
        binding = (lora_a, lora_b, scale)
        self._adapters[name] = binding
        if slot_id is not None:
            previous_slot = self._slot_by_adapter.get(name)
            if previous_slot is not None and previous_slot != slot_id:
                self._adapters_by_slot.pop(previous_slot, None)
            self._adapters_by_slot[slot_id] = binding
            self._slot_by_adapter[name] = slot_id

    def _base_dims(self) -> tuple[int | None, int | None]:
        """Extract (input_dims, output_dims) from the wrapped linear layer."""
        base = self.linear
        if hasattr(base, "input_dims") and hasattr(base, "output_dims"):
            return base.input_dims, base.output_dims
        # Quantized weights are packed: shape[1] = real_in * bits / 32
        if hasattr(base, "bits") and hasattr(base, "weight") and base.weight.ndim == 2:
            out_features = base.weight.shape[0]
            in_features = base.weight.shape[1] * 32 // base.bits
            return in_features, out_features
        if hasattr(base, "weight") and base.weight.ndim == 2:
            return base.weight.shape[1], base.weight.shape[0]
        return None, None

    def remove_adapter(self, name: str):
        if self._adapters.pop(name, None) is None:
            return
        slot_id = self._slot_by_adapter.pop(name, None)
        if slot_id is not None:
            self._adapters_by_slot.pop(slot_id, None)

    def has_adapter(self, name: str) -> bool:
        return name in self._adapters

    @property
    def adapter_names(self) -> list[str]:
        return list(self._adapters.keys())

    @property
    def slot_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._adapters_by_slot))

    def slot_bindings(self) -> list[tuple[int, mx.array, mx.array, float]]:
        return [
            (slot_id, *self._adapters_by_slot[slot_id])
            for slot_id in self.slot_ids
        ]

    def _active_binding(self) -> tuple[mx.array, mx.array, float] | None:
        slot_id = get_current_slot_id()
        if slot_id is not None:
            binding = self._adapters_by_slot.get(slot_id)
            if binding is not None:
                return binding

        adapter_id = get_current_adapter()
        if adapter_id is None:
            return None
        return self._adapters.get(adapter_id)

    def _routed_delta(self, x: mx.array) -> mx.array | None:
        if self.layer_name is None:
            return None
        layer_pack_state = get_current_routed_pack_state()
        token_slot_ids = get_current_token_slot_ids()
        if layer_pack_state is None or token_slot_ids is None:
            return None
        pack = layer_pack_state.get(self.layer_name)
        if pack is None:
            return None
        if not token_slot_ids:
            return None
        if not x.shape:
            return None

        row_count = 1
        for dim in x.shape[:-1]:
            row_count *= dim
        if row_count != len(token_slot_ids):
            return None

        try:
            return routed_decode_delta_rows_reference(
                x,
                pack,
                token_slot_ids,
                flatten_fn=lambda array: (array.reshape((-1, array.shape[-1])), tuple(array.shape)),
                restore_fn=lambda array, shape: array.reshape(shape[:-1] + (array.shape[-1],)),
                take_rows_fn=lambda array, rows: array[mx.array(rows)],
                concat_fn=lambda chunks: mx.concatenate(chunks, axis=0),
            )
        except ValueError:
            return None

    def __call__(self, x: mx.array) -> mx.array:
        y = self.linear(x)

        delta = self._routed_delta(x)
        if delta is None:
            binding = self._active_binding()
            if binding is not None:
                lora_a, lora_b, scale = binding
                delta = scale * ((x @ lora_a) @ lora_b)

        if delta is not None:
            y = y + delta.astype(y.dtype)

        return y


class MultiLoRASwitchLinear(nn.Module):
    """Multi-LoRA for MoE expert layers (SwitchLinear).

    Same concept but uses gather_mm for per-expert LoRA application.
    Handles the case where LoRA targets individual experts in a MoE layer.
    """

    def __init__(self, base_linear: nn.Module, layer_name: str | None = None):
        super().__init__()
        self.linear = base_linear
        self.layer_name = layer_name
        self._adapters: dict[str, tuple[mx.array, mx.array, float]] = {}
        self._adapters_by_slot: dict[int, tuple[mx.array, mx.array, float]] = {}
        self._slot_by_adapter: dict[str, int] = {}

    def add_adapter(
        self,
        name: str,
        lora_a: mx.array,
        lora_b: mx.array,
        scale: float,
        slot_id: int | None = None,
    ):
        binding = (lora_a, lora_b, scale)
        self._adapters[name] = binding
        if slot_id is not None:
            previous_slot = self._slot_by_adapter.get(name)
            if previous_slot is not None and previous_slot != slot_id:
                self._adapters_by_slot.pop(previous_slot, None)
            self._adapters_by_slot[slot_id] = binding
            self._slot_by_adapter[name] = slot_id

    def remove_adapter(self, name: str):
        if self._adapters.pop(name, None) is None:
            return
        slot_id = self._slot_by_adapter.pop(name, None)
        if slot_id is not None:
            self._adapters_by_slot.pop(slot_id, None)

    def has_adapter(self, name: str) -> bool:
        return name in self._adapters

    @property
    def slot_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._adapters_by_slot))

    def slot_bindings(self) -> list[tuple[int, mx.array, mx.array, float]]:
        return [
            (slot_id, *self._adapters_by_slot[slot_id])
            for slot_id in self.slot_ids
        ]

    def _active_binding(self) -> tuple[mx.array, mx.array, float] | None:
        slot_id = get_current_slot_id()
        if slot_id is not None:
            binding = self._adapters_by_slot.get(slot_id)
            if binding is not None:
                return binding

        adapter_id = get_current_adapter()
        if adapter_id is None:
            return None
        return self._adapters.get(adapter_id)

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        y = self.linear(x, indices)

        binding = self._active_binding()
        if binding is not None:
            lora_a, lora_b, scale = binding
            # Per-expert LoRA: lora_a is [num_experts, r, d_in]
            delta = mx.gather_mm(
                mx.gather_mm(x, lora_a, rhs_indices=indices),
                lora_b,
                rhs_indices=indices,
            )
            y = y + (scale * delta).astype(y.dtype)

        return y


def apply_multi_lora(model: nn.Module, target_modules: list[str] | None = None):
    """Replace target Linear layers with MultiLoRALinear.

    Walks the model tree and wraps matching layers. Does NOT load any adapter
    weights — that's done separately via inject_adapter_weights().

    Args:
        model: The mlx-lm model to modify
        target_modules: Layer name suffixes to target (e.g. ["self_attn.q_proj"]).
                       If None, targets all attention projections.
    """
    if target_modules is None:
        target_modules = [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        ]

    replacements = []
    for name, module in model.named_modules():
        if any(name.endswith(target) for target in target_modules):
            if isinstance(module, (nn.Linear, nn.QuantizedLinear)):
                replacements.append((name, MultiLoRALinear(module, layer_name=name)))
            # TODO: handle SwitchLinear for MoE expert LoRA

    if replacements:
        from mlx.utils import tree_unflatten

        model.update_modules(tree_unflatten(replacements))
        logger.info(f"Applied multi-LoRA wrappers to {len(replacements)} layers")

    return model


def inject_adapter_weights(
    model: nn.Module,
    adapter_name: str,
    weights: dict[str, tuple[mx.array, mx.array]],
    scale: float,
    slot_id: int | None = None,
):
    """Load an adapter's weights into the model's MultiLoRALinear layers.

    Args:
        model: Model with MultiLoRALinear layers (after apply_multi_lora)
        adapter_name: Name for this adapter
        weights: Dict mapping layer paths to (lora_a, lora_b) tuples
        scale: LoRA scaling factor (alpha / rank)
    """
    injected = 0
    for name, module in model.named_modules():
        if isinstance(module, (MultiLoRALinear, MultiLoRASwitchLinear)):
            if name in weights:
                lora_a, lora_b = weights[name]
                module.add_adapter(adapter_name, lora_a, lora_b, scale, slot_id=slot_id)
                injected += 1

    logger.info(f"Injected adapter '{adapter_name}' into {injected} layers")
    return injected


def eject_adapter_weights(model: nn.Module, adapter_name: str):
    """Remove an adapter's weights from all MultiLoRALinear layers."""
    ejected = 0
    for _, module in model.named_modules():
        if isinstance(module, (MultiLoRALinear, MultiLoRASwitchLinear)):
            if module.has_adapter(adapter_name):
                module.remove_adapter(adapter_name)
                ejected += 1

    logger.info(f"Ejected adapter '{adapter_name}' from {ejected} layers")
    return ejected
