"""OpenAI-compatible HTTP server with per-request adapter selection.

API design follows OpenAI format with adapter selection via the model field:
    "model": "base"              -> base model (reserved keyword)
    "model": "mlx-community/..."  -> base model (exact model_path from /health)
    "model": "code-assist"       -> "code-assist" adapter (direct name match)
    "model": "qwen/code-assist"  -> "code-assist" adapter (base/adapter pattern)
    "model": "qwen/typo"         -> 404 error ("typo" is not a loaded adapter)
    "model": "sqll"              -> 404 error (unknown bare name = typo caught)

Additional endpoints for adapter management:
    GET    /v1/adapters              -> list loaded adapters
    POST   /v1/adapters              -> hot-load a new adapter
    DELETE /v1/adapters/{name}       -> unload an adapter
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from mola.model import MOLAModel

logger = logging.getLogger(__name__)


# --- Request/Response models ---


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False


class AddAdapterRequest(BaseModel):
    name: str
    path: str


# --- Adapter ID extraction ---


BASE_MODEL_SELECTOR = "base"


def extract_adapter_id(model_field: str, mola_model: MOLAModel) -> str | None:
    """Extract adapter ID from the 'model' field in the request.

    The model field is a strict selector. Every value must resolve
    unambiguously — unknown names are errors, not silent fallbacks.

    Valid values:
        "base"                                  -> base model (reserved keyword)
        "mlx-community/Qwen3.5-35B-A3B-4bit"   -> base model (exact model_path)
        "code-assist"                           -> adapter (direct name)
        "qwen/code-assist"                      -> adapter (suffix-based: only the
                                                   part after the last "/" matters)
        "anything/code-assist"                  -> same — prefix is ignored
        "qwen/typo"                             -> 404 error
        "sqll"                                  -> 404 error (typo caught)

    The prefix in "prefix/adapter" is not validated — any prefix works as long
    as the suffix is a loaded adapter name. This is intentional: clients can use
    any convention they want for the prefix (model name, org, etc.).

    Raises ValueError for anything that doesn't resolve.
    """
    # 1. Reserved base model keyword
    if model_field == BASE_MODEL_SELECTOR:
        return None
    # 2. Direct adapter name match
    if mola_model.adapter_manager.get(model_field) is not None:
        return model_field
    # 3. Exact base model path — clients can use the value from /health
    if model_field == mola_model.model_path:
        return None
    # 4. base/adapter pattern — suffix must be a known adapter
    if "/" in model_field:
        _, candidate = model_field.rsplit("/", 1)
        if mola_model.adapter_manager.get(candidate) is not None:
            return candidate
        raise ValueError(
            f"'{candidate}' is not a loaded adapter "
            f"(from model field '{model_field}'). "
            f"Valid base model selectors: '{BASE_MODEL_SELECTOR}' "
            f"or '{mola_model.model_path}'."
        )
    # 5. Unknown bare name — not an adapter, not base → error
    adapters = [a["name"] for a in mola_model.list_adapters()]
    valid = [f"'{BASE_MODEL_SELECTOR}' (base model)"]
    valid += [f"'{n}' (adapter)" for n in adapters]
    raise ValueError(
        f"Unknown model selector '{model_field}'. "
        f"Valid values: {', '.join(valid)}. "
        f"Use GET /v1/adapters to list loaded adapters."
    )


# --- SSE streaming ---


async def generate_stream(
    request: ChatRequest, adapter_id: str | None, mola_model: MOLAModel
) -> AsyncGenerator[str, None]:
    """Stream tokens as SSE events in OpenAI format."""
    prompt = _format_chat(request.messages, tokenizer=mola_model.tokenizer)
    request_id = f"chatcmpl-{int(time.time())}"

    for step in mola_model.generate_step(
        prompt,
        adapter_id=adapter_id,
        max_tokens=request.max_tokens,
        temp=request.temperature,
        top_p=request.top_p,
    ):
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": step.text},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Final chunk
    final = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


def _format_chat(messages: list[Message], tokenizer=None) -> str:
    """Format chat messages into a prompt string.

    Uses the tokenizer's chat template when available, falls back to ChatML.
    """
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": m.role, "content": m.content} for m in messages],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass  # fall through to ChatML

    parts = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


# --- App factory ---


def create_app(mola_model: MOLAModel) -> FastAPI:
    app = FastAPI(title="MOLA", version="0.1.0")
    _lock = asyncio.Lock()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        try:
            adapter_id = extract_adapter_id(request.model, mola_model)
        except ValueError as e:
            raise HTTPException(404, str(e))

        if request.stream:
            async def locked_stream():
                async with _lock:
                    async for chunk in generate_stream(request, adapter_id, mola_model):
                        yield chunk

            return StreamingResponse(
                locked_stream(),
                media_type="text/event-stream",
            )

        async with _lock:
            result = mola_model.generate(
                prompt=_format_chat(request.messages, tokenizer=mola_model.tokenizer),
                adapter_id=adapter_id,
                max_tokens=request.max_tokens,
                temp=request.temperature,
                top_p=request.top_p,
            )
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result},
                    "finish_reason": "stop",
                }
            ],
        }

    @app.get("/v1/adapters")
    async def list_adapters():
        return {"adapters": mola_model.list_adapters()}

    @app.post("/v1/adapters")
    async def add_adapter(request: AddAdapterRequest):
        async with _lock:
            try:
                mola_model.load_adapter(request.name, request.path)
            except Exception as e:
                raise HTTPException(400, str(e))
        return {"status": "loaded", "name": request.name}

    @app.delete("/v1/adapters/{name}")
    async def remove_adapter(name: str):
        async with _lock:
            try:
                mola_model.unload_adapter(name)
            except KeyError:
                raise HTTPException(404, f"Adapter '{name}' not loaded")
        return {"status": "unloaded", "name": name}

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": mola_model.model_path,
            "adapters_loaded": len(mola_model.adapter_manager.adapters),
        }

    return app
