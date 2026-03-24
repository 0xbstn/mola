<p align="center">
  <h1 align="center">MOLA</h1>
  <p align="center"><b>Multi-adapter Orchestration LoRA Apple</b></p>
  <p align="center">Multi-LoRA inference server for Apple Silicon -- one base model, many adapters, zero reload.</p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-alpha-red" alt="Status: Alpha">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple" alt="Apple Silicon">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue?logo=python" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License">
  <img src="https://img.shields.io/badge/MLX-native-red" alt="MLX Native">
</p>

---

MOLA serves multiple LoRA adapters from a single base model on Apple Silicon. Load your base model once, hot-swap adapters per request, no reloading. Same-adapter requests are batched in a single GPU forward pass.

```
Base model (18 GB, loaded once)
├── Adapter "rust"        (+150 MB)  -- Rust systems programming
├── Adapter "bioml"       (+150 MB)  -- protein structure prediction
├── Adapter "k8s-ops"     (+150 MB)  -- Kubernetes troubleshooting
└── Adapter "sql"         (+150 MB)  -- SQL query optimization

Total memory: ~18.6 GB instead of 72 GB for 4 separate models.
Switch between adapters: instant.
Same-adapter requests: batched.
```

> **Status:** Alpha. Core serving works, same-adapter batching is healthy, and the API contract is stable enough for experimentation. Mixed-adapter decode is still the main performance ceiling. Tested on mlx-lm 0.31.1.

## Why

On CUDA, [vLLM](https://github.com/vllm-project/vllm) supports multi-LoRA serving with `--enable-lora`. On Apple Silicon with MLX, switching adapters means reloading the entire base model -- 18+ GB, 30+ seconds each time.

MOLA brings multi-LoRA serving to MLX. The base model weights stay in memory untouched. Adapter deltas are applied dynamically during the forward pass. Switching adapters is just swapping which delta matrices are active -- zero reload, zero downtime.

## Features

- **Per-request adapter selection** -- choose which adapter via the `model` field
- **Same-adapter batching** -- concurrent requests with the same adapter are batched in a single GPU forward pass via `BatchGenerator`
- **Continuous batching** -- new requests can join active adapter batches while capacity is available
- **Hot-load / hot-unload** -- add or remove adapters at runtime without restarting
- **OpenAI-compatible API** -- OpenAI-style chat completions subset; works with many OpenAI-compatible clients (Msty, Cursor, etc.)
- **SSE streaming** -- token-by-token streaming
- **Supports quantized models** -- works with 4-bit and 8-bit MLX models
- **MoE base models supported** -- serves MoE models (Qwen3.5, DeepSeek, Mixtral) with attention-layer LoRA adapters
- **Standard PEFT format** -- loads adapters trained with mlx-lm, mlx-tune, or any PEFT-compatible tool
- **Token-budget admission control** -- `--max-inflight-tokens` limits total in-flight work using `prompt_tokens + max_tokens`
- **Engine metrics** -- TTFT, tok/s, queue depth, active sequences, and lock-wait counters via `/v1/engine/metrics`
- **Backpressure** -- returns 503 when overloaded instead of queuing unboundedly
- **Client disconnect handling** -- cancelled requests are removed from the engine

## Quickstart

### Install

```bash
git clone https://github.com/0xbstn/mola.git
cd mola
pip install -e ".[dev]"
```

### Serve

```bash
mola serve \
  --model mlx-community/Qwen3.5-35B-A3B-4bit \
  --adapter rust ./adapters/rust-lora \
  --adapter sql ./adapters/sql-lora \
  --max-inflight-tokens 32768 \
  --port 8000
```

`--max-inflight-tokens` sets a global admission budget based on `prompt_tokens + max_tokens` per request. It matters under heavy prefill load and should be tuned against available unified memory.

### Query

The `model` field is a **strict selector** -- unknown names return **404**, typos are caught immediately.

```bash
# Use the "rust" adapter (base/adapter pattern)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen/rust",
    "messages": [{"role": "user", "content": "Implement a lock-free queue with crossbeam"}],
    "stream": true
  }'

# Use the "sql" adapter (direct name)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sql",
    "messages": [{"role": "user", "content": "Optimize this slow JOIN query"}]
  }'

# Base model
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "base",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Manage adapters at runtime

```bash
# List loaded adapters
curl http://localhost:8000/v1/adapters

# Hot-load a new adapter
curl -X POST http://localhost:8000/v1/adapters \
  -H "Content-Type: application/json" \
  -d '{"name": "medical", "path": "./adapters/medical-lora"}'

# Unload an adapter
curl -X DELETE http://localhost:8000/v1/adapters/medical
```

## Architecture

```
HTTP request
    │
    ▼
extract_adapter_id()        ─── strict model selector
    │
    ▼
MOLAEngine.submit()         ─── backpressure (503 if full)
    │
    ▼
┌───────────────────────────────────────────┐
│ Engine thread (round-robin)               │
│                                           │
│  BatchGenerator "rust"  ◄── same-adapter  │
│  BatchGenerator "sql"       batching      │
│  BatchGenerator base                      │
│                                           │
│  model_lock ◄── prevents race with        │
│                  adapter load/unload       │
└───────────────────────────────────────────┘
    │
    ▼
MultiLoRALinear             ─── reads adapter from contextvars
    │                           applies the right delta per layer
    ▼
Tokens dispatched via async queue ──► SSE stream
```

## API Reference

### `POST /v1/chat/completions`

OpenAI-compatible chat completions. Adapter selection via the `model` field:

| Model field | Behavior |
|---|---|
| `"base"` | Base model, no adapter (reserved keyword) |
| `"mlx-community/Qwen3.5-35B-A3B-4bit"` | Base model (exact `model_path` from `/health`) |
| `"rust"` | `rust` adapter (direct name match) |
| `"qwen/rust"` | `rust` adapter (suffix-based, prefix ignored) |
| `"qwen/typo"` | **404** -- `typo` is not a loaded adapter |
| `"sqll"` | **404** -- unknown bare name (typo caught) |

Supports `stream: true` for SSE streaming.

### `GET /v1/models`

OpenAI-compatible model listing. Returns `base`, model path, and all loaded adapter names. Used by clients like Msty for auto-discovery.

### `GET /v1/adapters`

List all loaded adapters with metadata (rank, scale, memory usage).

### `POST /v1/adapters`

Hot-load a new adapter. Body: `{"name": "...", "path": "..."}`. Adapter load/unload is synchronized with the engine via `model_lock`.

### `DELETE /v1/adapters/{name}`

Unload an adapter and free its memory.

### `GET /v1/engine/metrics`

Engine runtime metrics:

```json
{
  "queued_requests": 0,
  "active_generators": 2,
  "active_sequences": 0,
  "total_tokens_generated": 552,
  "requests_completed": 11,
  "requests_rejected": 0,
  "inflight_tokens_reserved": 0,
  "token_budget_limit": 32768,
  "total_step_lock_wait_ms": 0.04,
  "total_insert_lock_wait_ms": 0.0,
  "avg_ttft_ms": 53.8,
  "avg_tps": 225.6
}
```

### `GET /health`

Server health check with model and adapter count.

## Benchmarks

Light benchmark snapshot measured on Qwen2.5-0.5B-Instruct-4bit with rust + sql adapters (Apple Silicon):

| Scenario | conc=8 | conc=16 |
|---|---|---|
| Base | 21.4 req/s, 1368 tok/s, p95 375 ms | 26.3 req/s, 1681 tok/s, p95 610 ms |
| Same adapter | 13.6 req/s, 786 tok/s, p95 597 ms | 18.1 req/s, 1048 tok/s, p95 889 ms |
| Mixed | 4.8 req/s, 246 tok/s, p95 1724 ms | 7.0 req/s, 346 tok/s, p95 2334 ms |

Same-adapter batching scales well. Mixed-adapter traffic is correct and usable, but mixed decode remains the main throughput and latency ceiling.

## One-shot generation (no server)

```bash
mola generate \
  --model mlx-community/Qwen3.5-35B-A3B-4bit \
  --adapter-name rust \
  --adapter-path ./adapters/rust-lora \
  "Implement a thread-safe cache with TTL eviction"
```

## Adapter format

MOLA loads standard PEFT / mlx-lm adapter directories:

```
my-adapter/
├── adapter_config.json      # rank, scale, target modules
└── adapters.safetensors     # lora_a and lora_b weights
```

Train adapters with [mlx-lm](https://github.com/ml-explore/mlx-lm), [mlx-tune](https://github.com/ARahim3/mlx-tune), or any tool that outputs PEFT-compatible safetensors. LoRA, QLoRA -- same output format, both work.

## Requirements

- Apple Silicon Mac (M1 or later)
- Python 3.11+
- macOS 13+
- MLX 0.25+ / mlx-lm 0.25+ (tested on 0.31.1)

## Memory usage

| Component | Size (Qwen3.5-35B-A3B 4-bit) |
|---|---|
| Base model | ~18 GB |
| Each LoRA adapter (rank 64) | ~150 MB |
| 10 adapters | ~1.5 GB |
| **Total (base + 10 adapters)** | **~19.5 GB** |

Compare: loading 10 separate fine-tuned models would require ~180 GB.

## Limitations

- **Alpha** -- API contract is stable but internals may change
- **Apple Silicon only** -- requires MLX (no CUDA, no CPU fallback)
- **No cross-adapter batching** -- requests with different adapters are round-robined, not batched in the same forward pass. Same-adapter requests ARE batched.
- **KV cache** -- switching adapters mid-conversation invalidates the KV cache. Each conversation should use one adapter throughout.
- **Strict adapter validation** -- an adapter must inject into all its expected target layers. Partially compatible adapters are rejected at load time.
- **No custom kernels** -- uses standard MLX matmul. Cross-adapter batching (S-LoRA style) would require a custom Metal kernel (BGMV equivalent).

## Contributing

```bash
git clone https://github.com/0xbstn/mola.git
cd mola
pip install -e ".[dev]"
pytest
```

## License

Apache 2.0
