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

> **Status:** Alpha. The current architecture winner is now the detached shared decode owner runtime on top of the reproducible `mlx-lm` detached-batch patch, with `metal-gather` as the best routed compute backend. Same-adapter and mixed-adapter serving are both stable. The routed decode reference path remains a correctness scaffold, not the default fast path. Tested on `mlx-lm 0.31.1`.

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
- **Experimental routed decode reference** -- `--enable-routed-decode-reference` prepares a homogeneous decode-only routed LoRA path for pre-kernel validation; it is a correctness/reference path, not a faster runtime path today
- **Detached shared decode owner** -- the current winner architecture centralizes mixed decode ownership behind `DecodeOwner` instead of leaving decode fully generator-owned
- **Reproducible `mlx-lm` patch flow** -- winner runtime is reproducible via a tracked patcher and tracked launcher instead of manual edits
- **Backpressure** -- returns 503 when overloaded instead of queuing unboundedly
- **Client disconnect handling** -- cancelled requests are removed from the engine

## Quickstart

### Install

```bash
git clone https://github.com/0xbstn/mola.git
cd mola
python3 -m venv .venv
./.venv/bin/python -m ensurepip --upgrade
./.venv/bin/python -m pip install -e ".[dev]"
./.venv/bin/python devtools/apply_mlx_lm_detached_batch_api.py
```

### Winner Current Architecture

The current architecture winner is:

- detached shared decode owner
- public detached-batch API patched into local `mlx-lm`
- routed mixed decode backend `metal-gather`
- mixed decode migration + pre-step migration
- routed session caching

This is the exact reproducible winner launcher:

```bash
cd mola
./.venv/bin/python devtools/run_mola_winner.py start --port 8000
```

Equivalent explicit server command:

```bash
./.venv/bin/python -m mola.cli -v serve \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --adapter rust ./adapters/rust-lora \
  --adapter sql ./adapters/sql-lora \
  --adapter medical ./adapters/medical-lora \
  --adapter cyber ./adapters/cyber-lora \
  --adapter solidity ./adapters/solidity-lora \
  --adapter devops ./adapters/devops-lora \
  --adapter math ./adapters/math-lora \
  --adapter legal ./adapters/legal-lora \
  --max-inflight-tokens 131072 \
  --max-batch-size 128 \
  --prefill-batch-size 32 \
  --enable-routed-decode-reference \
  --strict-routed-decode-reference \
  --routed-decode-backend metal-gather \
  --enable-mixed-decode-migration \
  --prestep-mixed-decode-migration \
  --cache-routed-decode-sessions \
  --detached-shared-decode-owner \
  --port 8000
```

To stop or restart the winner:

```bash
./.venv/bin/python devtools/run_mola_winner.py stop --port 8000
./.venv/bin/python devtools/run_mola_winner.py restart --port 8000
```

### Serve

```bash
./.venv/bin/python devtools/run_mola_winner.py start --port 8000
```

`--max-inflight-tokens` sets a global admission budget based on `prompt_tokens + max_tokens` per request. It matters under heavy prefill load and should be tuned against available unified memory.

`devtools/run_mola_winner.py` first applies the tracked `mlx-lm` detached-batch patch if needed, then starts the current winner configuration.

`--enable-routed-decode-reference` still turns on the routed decode validation seam, but in the current winner architecture it is paired with the `metal-gather` backend and the detached decode owner runtime.

`--strict-routed-decode-reference` is the fail-closed variant for backend validation. With it enabled, routed decode contract mismatches are surfaced instead of silently falling back to the default adapter path.

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
│ Engine thread + DecodeOwner               │
│                                           │
│  Adapter-local BatchGenerators            │
│    └── prefill / ramp-up feeders          │
│                                           │
│  Shared detached DecodeOwner              │
│    └── mixed decode owner                 │
│                                           │
│  model_lock ◄── prevents race with        │
│                  adapter load/unload       │
└───────────────────────────────────────────┘
    │
    ▼
MultiLoRALinear             ─── routed session + slot metadata
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

Canonical current-architecture benchmark:

```bash
./.venv/bin/python scripts/bench_server.py \
  --routed-validation \
  --concurrency 64,128 \
  --repeats 3 \
  --json-out benchmark/current-architecture-winner-routed-validation-r3.json
```

Measured on `mlx-community/Qwen2.5-0.5B-Instruct-4bit` with 8 resident adapters on Apple Silicon:

| Scenario | conc=64 | conc=128 |
|---|---|---|
| Same | 33.6 req/s, 1984.8 tok/s, p95 1927.3 ms | 31.9 req/s, 1937.4 tok/s, p95 3957.6 ms |
| Mixed | 27.8 req/s, 1351.4 tok/s, p95 2284.9 ms | 26.7 req/s, 1311.5 tok/s, p95 4614.4 ms |
| Long decode mixed | 12.2 req/s, 1752.8 tok/s, p95 5290.1 ms | 13.4 req/s, 1996.5 tok/s, p95 9425.9 ms |
| Fairness | 29.7 req/s, 1533.4 tok/s, p95 2184.8 ms | 28.4 req/s, 1472.7 tok/s, p95 4346.7 ms |

Key ratios on the winner:

- `mixed / same @64` ≈ `0.83`
- `mixed / same @128` ≈ `0.84`

Benchmark artifact committed in the repo:

- [current-architecture-winner-routed-validation-r3.json](/Users/bastienbouge/Documents/dev/mola/benchmark/current-architecture-winner-routed-validation-r3.json)

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
- **Local `mlx-lm` patch required** -- the current winner depends on a reproducible local patch to `mlx_lm.generate.BatchGenerator`; this is scripted, but not upstreamed yet
- **Mixed prefill is not solved** -- the current winner broadens mixed decode materially, but prefill/KV ownership is still not a first-class subsystem
- **No paged KV / residency controller yet** -- adapter+KV ownership, paged base KV, and deeper prefill-owner work remain future architecture work
- **KV cache** -- switching adapters mid-conversation invalidates the KV cache. Each conversation should use one adapter throughout.
- **Strict adapter validation** -- an adapter must inject into all its expected target layers. Partially compatible adapters are rejected at load time.
- **Still not a CUDA-class multi-LoRA engine** -- this winner uses routed MLX ops and a detached decode owner; it is materially better than the older generator-owned runtime, but it is not yet a paged-KV heterogeneous scheduler like vLLM/S-LoRA-class systems.

## What Is Merged vs Lab

Merged in this current architecture candidate:

- detached shared decode owner runtime
- reproducible `mlx-lm` detached-batch patch flow
- `metal-gather` as the current routed backend winner
- reproducible launcher and canonical benchmark command

Kept out of this merge candidate:

- chunked prefill + decode interleaving lab line
- Qwen3.5-9B baseline work
- falsified `metal-gather` local refinements
- falsified alternate routed backend families
- deeper prefill-owner / paged-KV / residency experiments

## Contributing

```bash
git clone https://github.com/0xbstn/mola.git
cd mola
pip install -e ".[dev]"
pytest
```

## License

Apache 2.0
