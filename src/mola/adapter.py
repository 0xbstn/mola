"""Adapter loading and management.

Handles loading LoRA adapters from local paths or HuggingFace,
parsing PEFT-format configs, and managing adapter lifecycle in memory.

Adapter format (standard PEFT/mlx-lm):
    adapters/
        adapter_config.json     # rank, scale, target modules, num_layers
        adapters.safetensors    # lora_a and lora_b weights per layer
"""

from __future__ import annotations

import heapq
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AdapterConfig:
    rank: int
    scale: float
    dropout: float
    num_layers: int
    target_modules: list[str] | None
    fine_tune_type: str = "lora"

    @classmethod
    def from_file(cls, path: Path) -> AdapterConfig:
        with open(path / "adapter_config.json") as f:
            raw = json.load(f)

        lora_params = raw.get("lora_parameters", {})
        return cls(
            rank=lora_params.get("rank", 8),
            scale=lora_params.get("scale", 20.0),
            dropout=lora_params.get("dropout", 0.0),
            num_layers=raw.get("num_layers", 16),
            target_modules=lora_params.get("keys", None),
            fine_tune_type=raw.get("fine_tune_type", "lora"),
        )


@dataclass
class AdapterWeights:
    """LoRA weights for a single adapter, organized by layer path.

    weights maps layer paths to (lora_a, lora_b) tuples:
        "layers.16.self_attn.q_proj" -> (mx.array[d_in, r], mx.array[r, d_out])
    """

    weights: dict[str, tuple[mx.array, mx.array]] = field(default_factory=dict)

    @property
    def memory_bytes(self) -> int:
        total = 0
        for lora_a, lora_b in self.weights.values():
            total += lora_a.nbytes + lora_b.nbytes
        return total


@dataclass
class Adapter:
    name: str
    slot_id: int | None
    config: AdapterConfig
    weights: AdapterWeights
    source_path: str


@dataclass(frozen=True)
class AdapterSlotBinding:
    name: str
    slot_id: int
    rank: int
    scale: float
    num_layers: int
    target_modules: tuple[str, ...]
    source_path: str


class AdapterManager:
    """Manages loading, caching, and lifecycle of LoRA adapters.

    Keeps adapters in memory as MLX arrays. On a 64GB Mac with an 18GB base model,
    you can hold ~200 adapters of 150MB each. No eviction needed for typical use.
    """

    def __init__(self, max_adapters: int = 50):
        self.adapters: dict[str, Adapter] = {}
        self.max_adapters = max_adapters
        self._free_slot_ids: list[int] = []
        self._next_slot_id = 0
        self._slot_to_name: dict[int, str] = {}

    def _allocate_slot_id(self) -> int:
        if self._free_slot_ids:
            return heapq.heappop(self._free_slot_ids)
        slot_id = self._next_slot_id
        self._next_slot_id += 1
        return slot_id

    def _release_slot_id(self, slot_id: int | None) -> None:
        if slot_id is None:
            return
        heapq.heappush(self._free_slot_ids, slot_id)

    def load(self, name: str, path: str) -> Adapter:
        """Load an adapter from a local path.

        Args:
            name: Identifier for this adapter (used in API requests)
            path: Local path to adapter directory (must contain adapter_config.json
                  and adapters.safetensors)
        """
        if name in self.adapters:
            logger.info(f"Adapter '{name}' already loaded, skipping")
            return self.adapters[name]

        if len(self.adapters) >= self.max_adapters:
            raise RuntimeError(
                f"Max adapters ({self.max_adapters}) reached. "
                f"Unload an adapter first."
            )

        adapter_path = Path(path)
        config = AdapterConfig.from_file(adapter_path)
        weights = self._load_weights(adapter_path, config)
        slot_id = self._allocate_slot_id()

        adapter = Adapter(
            name=name,
            slot_id=slot_id,
            config=config,
            weights=weights,
            source_path=path,
        )
        self.adapters[name] = adapter
        self._slot_to_name[slot_id] = name

        mb = weights.memory_bytes / 1024 / 1024
        logger.info(
            f"Loaded adapter '{name}' (slot={slot_id}, {mb:.1f} MB, rank={config.rank})"
        )
        return adapter

    def unload(self, name: str) -> None:
        if name not in self.adapters:
            raise KeyError(f"Adapter '{name}' not loaded")
        adapter = self.adapters.pop(name)
        if adapter.slot_id is not None:
            self._slot_to_name.pop(adapter.slot_id, None)
        self._release_slot_id(adapter.slot_id)
        logger.info(f"Unloaded adapter '{name}'")

    def get(self, name: str) -> Adapter | None:
        return self.adapters.get(name)

    def slot_id(self, name: str | None) -> int | None:
        if name is None:
            return None
        adapter = self.adapters.get(name)
        return adapter.slot_id if adapter else None

    def name_for_slot_id(self, slot_id: int | None) -> str | None:
        if slot_id is None:
            return None
        return self._slot_to_name.get(slot_id)

    def slot_bindings(self) -> list[AdapterSlotBinding]:
        bindings: list[AdapterSlotBinding] = []
        for adapter in self.adapters.values():
            if adapter.slot_id is None:
                continue
            bindings.append(
                AdapterSlotBinding(
                    name=adapter.name,
                    slot_id=adapter.slot_id,
                    rank=adapter.config.rank,
                    scale=adapter.config.scale,
                    num_layers=adapter.config.num_layers,
                    target_modules=tuple(adapter.config.target_modules or ()),
                    source_path=adapter.source_path,
                )
            )
        return sorted(bindings, key=lambda binding: binding.slot_id)

    def list_adapters(self) -> list[dict]:
        return [
            {
                "name": a.name,
                "slot_id": a.slot_id,
                "rank": a.config.rank,
                "scale": a.config.scale,
                "num_layers": a.config.num_layers,
                "target_modules": a.config.target_modules,
                "memory_mb": round(a.weights.memory_bytes / 1024 / 1024, 1),
                "source": a.source_path,
            }
            for a in self.adapters.values()
        ]

    def _load_weights(
        self, adapter_path: Path, config: AdapterConfig
    ) -> AdapterWeights:
        """Load lora_a/lora_b weight pairs from safetensors.

        mlx-lm saves adapter weights with keys like:
            model.layers.16.self_attn.q_proj.lora_a  -> shape [input_dims, rank]
            model.layers.16.self_attn.q_proj.lora_b  -> shape [rank, output_dims]

        Also derives target_modules from weight keys when not in config.
        """
        import mlx.core as mx

        safetensors_path = adapter_path / "adapters.safetensors"
        raw_weights = mx.load(str(safetensors_path))

        # Group lora_a and lora_b by layer path
        paired: dict[str, tuple[mx.array, mx.array]] = {}
        for key, tensor in raw_weights.items():
            if key.endswith(".lora_a"):
                layer_path = key[: -len(".lora_a")]
                b_key = layer_path + ".lora_b"
                if b_key in raw_weights:
                    paired[layer_path] = (tensor, raw_weights[b_key])

        # Derive target_modules from weight keys when config doesn't specify
        if config.target_modules is None and paired:
            import re
            suffixes = set()
            for layer_path in paired:
                # Strip "model.layers.N." prefix → "self_attn.q_proj"
                match = re.sub(r"^(model\.)?layers\.\d+\.", "", layer_path)
                suffixes.add(match)
            config.target_modules = sorted(suffixes)
            logger.info(
                f"Derived target_modules from weights: {config.target_modules}"
            )

        return AdapterWeights(weights=paired)
