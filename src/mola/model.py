"""Model loading and adapter orchestration.

Wraps mlx-lm's model loading with multi-LoRA support.
This is the main entry point that ties everything together.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlx.core as mx
import mlx_lm

from mola.adapter import Adapter, AdapterManager
from mola.context import adapter_context
from mola.lora import apply_multi_lora, eject_adapter_weights, inject_adapter_weights

logger = logging.getLogger(__name__)


class MOLAModel:
    """Base model + multiple LoRA adapters, ready to serve.

    Usage:
        model = MOLAModel("mlx-community/Qwen3.5-35B-A3B-4bit")
        model.load_adapter("rust", "./adapters/rust-lora")
        model.load_adapter("sql", "./adapters/sql-lora")

        # Generate with a specific adapter
        text = model.generate("Implement a lock-free queue", adapter_id="rust")

        # Generate with base model (no adapter)
        text = model.generate("Hello", adapter_id=None)
    """

    def __init__(self, model_path: str):
        logger.info(f"Loading base model: {model_path}")
        self.model, self.tokenizer = mlx_lm.load(model_path)
        self.model_path = model_path
        self.adapter_manager = AdapterManager()
        self._wrapped_modules: set[str] = set()

    def load_adapter(self, name: str, path: str) -> Adapter:
        """Load a LoRA adapter and inject its weights into the model.

        First call triggers the model surgery (wrapping Linear -> MultiLoRALinear).
        Subsequent calls just inject new adapter weights.
        """
        _RESERVED_NAMES = {"base", self.model_path}
        if name in _RESERVED_NAMES:
            raise ValueError(
                f"Adapter name '{name}' is reserved — "
                f"choose a different name"
            )
        adapter = self.adapter_manager.load(name, path)

        needed = set(adapter.config.target_modules or [])
        new_modules = needed - self._wrapped_modules
        if new_modules:
            if self._wrapped_modules:
                logger.warning(
                    f"Adapter '{name}' requires new target modules {new_modules} "
                    f"not covered by previous adapters — extending surgery"
                )
            apply_multi_lora(self.model, list(new_modules))
            self._wrapped_modules |= new_modules

        expected = len(adapter.weights.weights)
        if expected == 0:
            self.adapter_manager.unload(name)
            raise RuntimeError(
                f"Adapter '{name}': no valid LoRA weight pairs found in safetensors"
            )
        try:
            injected = inject_adapter_weights(
                self.model,
                adapter_name=name,
                weights=adapter.weights.weights,
                scale=adapter.config.scale,
            )
        except Exception:
            # Rollback: some layers may already have the adapter injected
            eject_adapter_weights(self.model, name)
            self.adapter_manager.unload(name)
            raise
        if injected < expected:
            eject_adapter_weights(self.model, name)
            self.adapter_manager.unload(name)
            raise RuntimeError(
                f"Adapter '{name}': only {injected}/{expected} weight pairs matched "
                f"wrapped layers — adapter rejected to prevent incorrect outputs"
            )
        return adapter

    def unload_adapter(self, name: str) -> None:
        eject_adapter_weights(self.model, name)
        self.adapter_manager.unload(name)

    def list_adapters(self) -> list[dict]:
        return self.adapter_manager.list_adapters()

    @staticmethod
    def _make_sampler(temp: float, top_p: float):
        """Build an mlx-lm sampler from temperature and top_p."""
        from mlx_lm.sample_utils import make_sampler

        return make_sampler(temp=temp, top_p=top_p)

    def generate(
        self,
        prompt: str,
        adapter_id: str | None = None,
        max_tokens: int = 256,
        temp: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate text with an optional adapter."""
        sampler = self._make_sampler(temp, top_p)
        with adapter_context(adapter_id):
            response = mlx_lm.generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
            )
        return response

    def generate_step(
        self,
        prompt: str,
        adapter_id: str | None = None,
        max_tokens: int = 256,
        temp: float = 0.7,
        top_p: float = 0.9,
    ):
        """Yield tokens one by one for streaming. Used by the server."""
        sampler = self._make_sampler(temp, top_p)
        with adapter_context(adapter_id):
            for step in mlx_lm.stream_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
            ):
                yield step
