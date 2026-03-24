"""OpenAI-compatible HTTP server with per-request adapter selection.

The model field is a strict selector:
    "base"              -> base model (reserved keyword)
    "mlx-community/..."  -> base model (exact model_path from /health)
    "code-assist"       -> adapter (direct name match)
    "qwen/code-assist"  -> adapter (suffix-based, prefix ignored)
    "qwen/typo"         -> 404
    "sqll"              -> 404 (unknown bare name)
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import time
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from mola.engine import AdmissionRejected, EngineConfig, GenerateRequest, MOLAEngine
from mola.model import MOLAModel

logger = logging.getLogger(__name__)

BASE_MODEL_SELECTOR = "base"


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


def extract_adapter_id(model_field: str, mola_model: MOLAModel) -> str | None:
    """Strict model selector. Raises ValueError for anything that doesn't resolve.

    The prefix in "prefix/adapter" is not validated — any prefix works
    as long as the suffix is a loaded adapter name.
    """
    if model_field == BASE_MODEL_SELECTOR:
        return None
    if mola_model.adapter_manager.get(model_field) is not None:
        return model_field
    if model_field == mola_model.model_path:
        return None
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
    adapters = [a["name"] for a in mola_model.list_adapters()]
    valid = [f"'{BASE_MODEL_SELECTOR}' (base model)"]
    valid += [f"'{n}' (adapter)" for n in adapters]
    raise ValueError(
        f"Unknown model selector '{model_field}'. "
        f"Valid values: {', '.join(valid)}. "
        f"Use GET /v1/adapters to list loaded adapters."
    )


def _format_chat(messages: list[Message], tokenizer=None) -> str:
    """Use tokenizer's chat template when available, fall back to ChatML."""
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": m.role, "content": m.content} for m in messages],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    parts = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


async def _stream_tokens(
    engine: MOLAEngine,
    response_queue: asyncio.Queue,
    chat_request: ChatRequest,
    tokenizer,
    raw_request: Request,
) -> AsyncGenerator[str, None]:
    qid = id(response_queue)
    request_id = f"chatcmpl-{int(time.time())}"
    tokens: list[int] = []
    prev_text = ""

    while True:
        if await raw_request.is_disconnected():
            logger.debug(f"http_disconnect queue={qid} stream=true")
            engine.cancel(response_queue)
            return
        try:
            data = await asyncio.wait_for(response_queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        if data is None:
            logger.debug(f"stream_recv queue={qid} kind=none")
            break
        if "error" in data:
            logger.debug(f"stream_recv queue={qid} kind=error")
            yield f"data: {json.dumps({'error': data['error']})}\n\n"
            return

        logger.debug(f"stream_recv queue={qid} kind=token")
        tokens.append(data["token"])
        full_text = tokenizer.decode(tokens)
        new_text = full_text[len(prev_text):]
        prev_text = full_text

        if new_text:
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": chat_request.model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": new_text},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

    yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': chat_request.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
    yield "data: [DONE]\n\n"
    logger.debug(f"stream_done queue={qid}")


def create_app(
    mola_model: MOLAModel, engine_config: EngineConfig | None = None
) -> FastAPI:
    app = FastAPI(title="MOLA", version="0.2.0")
    engine = MOLAEngine(mola_model, engine_config)

    @app.on_event("startup")
    async def startup():
        engine.start(asyncio.get_event_loop())

    @app.on_event("shutdown")
    async def shutdown():
        engine.stop()

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatRequest, request: Request):
        try:
            adapter_id = extract_adapter_id(body.model, mola_model)
        except ValueError as e:
            raise HTTPException(404, str(e))

        prompt = _format_chat(body.messages, tokenizer=mola_model.tokenizer)
        prompt_tokens = mola_model.tokenizer.encode(prompt)
        sampler = mola_model._make_sampler(body.temperature, body.top_p)

        response_queue = asyncio.Queue(maxsize=engine.config.response_queue_size)
        gen_request = GenerateRequest(
            prompt_tokens=prompt_tokens,
            adapter_id=adapter_id,
            max_tokens=body.max_tokens,
            sampler=sampler,
            response_queue=response_queue,
        )

        try:
            engine.submit(gen_request)
        except AdmissionRejected as e:
            raise HTTPException(503, str(e))
        except queue.Full:
            raise HTTPException(503, "Server overloaded, try again later")

        qid = id(response_queue)
        logger.debug(f"http_submit queue={qid} adapter={adapter_id} stream={body.stream}")

        if body.stream:
            return StreamingResponse(
                _stream_tokens(engine, response_queue, body, mola_model.tokenizer, request),
                media_type="text/event-stream",
            )

        tokens: list[int] = []
        t_wait_start = time.time()
        while True:
            if await request.is_disconnected():
                logger.debug(f"http_disconnect queue={qid} stream=false")
                engine.cancel(response_queue)
                raise HTTPException(499, "Client disconnected")
            try:
                data = await asyncio.wait_for(response_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if data is None:
                logger.debug(f"http_recv queue={qid} kind=none")
                logger.debug(
                    f"http_done queue={qid} tokens={len(tokens)} "
                    f"wait={time.time() - t_wait_start:.2f}s"
                )
                break
            if "error" in data:
                logger.debug(f"http_recv queue={qid} kind=error")
                raise HTTPException(500, data["error"])
            logger.debug(f"http_recv queue={qid} kind=token")

            tokens.append(data["token"])

        text = mola_model.tokenizer.decode(tokens)
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
        }

    @app.get("/v1/adapters")
    async def list_adapters():
        return {"adapters": mola_model.list_adapters()}

    @app.post("/v1/adapters")
    async def add_adapter(body: AddAdapterRequest):
        def _do_load():
            with engine.model_lock:
                mola_model.load_adapter(body.name, body.path)

        try:
            await asyncio.to_thread(_do_load)
        except Exception as e:
            raise HTTPException(400, str(e))
        return {"status": "loaded", "name": body.name}

    @app.delete("/v1/adapters/{name}")
    async def remove_adapter(name: str):
        def _do_unload():
            with engine.model_lock:
                mola_model.unload_adapter(name)

        try:
            await asyncio.to_thread(_do_unload)
        except KeyError:
            raise HTTPException(404, f"Adapter '{name}' not loaded")
        return {"status": "unloaded", "name": name}

    @app.get("/v1/models")
    async def list_models():
        models = [
            {"id": BASE_MODEL_SELECTOR, "object": "model", "owned_by": "mola"},
            {"id": mola_model.model_path, "object": "model", "owned_by": "mola"},
        ]
        for adapter in mola_model.list_adapters():
            models.append({"id": adapter["name"], "object": "model", "owned_by": "mola"})
        return {"object": "list", "data": models}

    @app.get("/v1/engine/metrics")
    async def engine_metrics():
        snap = engine.metrics.snapshot()
        snap["slots"] = engine.slot_snapshots()
        return snap

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": mola_model.model_path,
            "adapters_loaded": len(mola_model.adapter_manager.adapters),
        }

    return app
