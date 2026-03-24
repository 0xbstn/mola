<p align="center">
  <h1 align="center">MOLA</h1>
  <p align="center"><b>Multi-adapter Orchestration LoRA Apple</b></p>
  <p align="center">Multi-LoRA inference server for Apple Silicon — one base model, many adapters, zero reload.</p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-alpha-red" alt="Status: Alpha">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple" alt="Apple Silicon">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue?logo=python" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License">
  <img src="https://img.shields.io/badge/MLX-native-red" alt="MLX Native">
</p>

---

MOLA serves multiple LoRA adapters from a single base model on Apple Silicon. Load your base model once, hot-swap adapters per request, no reloading.

```
Base model (18 GB, loaded once)
├── Adapter "rust"        (+150 MB)  — Rust systems programming
├── Adapter "bioml"       (+150 MB)  — protein structure prediction
├── Adapter "k8s-ops"     (+150 MB)  — Kubernetes troubleshooting
└── Adapter "sql"         (+150 MB)  — SQL query optimization

Total memory: ~18.6 GB instead of 72 GB for 4 separate models.
Switch between adapters: instant.
```

> **Status:** MOLA is alpha. Core functionality works, API contract is stable, but no production benchmarks yet. Contributions welcome.

## Why

On CUDA, [vLLM](https://github.com/vllm-project/vllm) supports multi-LoRA serving with `--enable-lora`. On Apple Silicon with MLX, switching adapters means reloading the entire base model — 18+ GB, 30+ seconds each time.

MOLA brings multi-LoRA serving to MLX. The base model weights stay in memory untouched. Adapter deltas are applied dynamically during the forward pass. Switching adapters is just swapping which delta matrices are active — zero reload, zero downtime.

## Features

- **Per-request adapter selection** — choose which adapter via the `model` field in the API
- **Hot-load / hot-unload** — add or remove adapters at runtime without restarting
- **OpenAI-compatible API** — works with any client that speaks OpenAI format
- **SSE streaming** — token-by-token streaming out of the box
- **Minimal overhead** — adapter deltas are two small matrix multiplies per layer
- **Supports quantized models** — works with 4-bit and 8-bit MLX models
- **MoE base models supported** — serves MoE models (Qwen3.5, DeepSeek, Mixtral) with attention-layer LoRA adapters. Full expert-layer LoRA (SwitchLinear) is on the roadmap.
- **Standard PEFT format** — loads adapters trained with mlx-lm, mlx-tune, or any PEFT-compatible tool

## Quickstart

### Install

```bash
pip install mola
```

Or from source:

```bash
git clone https://github.com/YOUR_USERNAME/mola.git
cd mola
pip install -e ".[dev]"
```

### Serve

```bash
mola serve \
  --model mlx-community/Qwen3.5-35B-A3B-4bit \
  --adapter rust ./adapters/rust-lora \
  --adapter sql ./adapters/sql-lora \
  --port 8000
```

### Query

The `model` field is a **strict selector** — every value must resolve unambiguously. Unknown names return **404** instead of silently falling back to the base model, so typos are caught immediately.

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

# Base model — reserved keyword
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "base",
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# Base model — full model path (same value returned by /health)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.5-35B-A3B-4bit",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Manage adapters at runtime

```bash
# List loaded adapters
curl http://localhost:8000/v1/adapters

# Hot-load a new adapter (no restart needed)
curl -X POST http://localhost:8000/v1/adapters \
  -H "Content-Type: application/json" \
  -d '{"name": "medical", "path": "./adapters/medical-lora"}'

# Unload an adapter
curl -X DELETE http://localhost:8000/v1/adapters/medical
```

## How it works

MOLA wraps the base model's linear layers with `MultiLoRALinear` — a drop-in replacement that holds multiple adapter weight sets and selects the right one per request.

```python
# Standard mlx-lm LoRA (single adapter, applied at load time):
y = base_linear(x) + scale * (x @ lora_a) @ lora_b

# MOLA (multiple adapters, selected per request via context):
adapter_id = get_current_adapter()   # read from request context
lora_a, lora_b, scale = adapters[adapter_id]
y = base_linear(x) + scale * (x @ lora_a) @ lora_b
```

The base weights are never modified. Each adapter adds two small matrices per target layer (~50-200 MB total per adapter, depending on rank). Adapter selection uses Python's `contextvars` — no model code modifications needed.

```
Request ──► Server extracts adapter_id from "model" field
               │
               ▼
            Sets adapter context (contextvars)
               │
               ▼
            Forward pass: each MultiLoRALinear reads the context
            and applies the matching adapter delta
               │
               ▼
            Tokens streamed back via SSE
```

## API Reference

### `POST /v1/chat/completions`

OpenAI-compatible chat completions. Adapter selection via the `model` field:

| Model field | Behavior |
|---|---|
| `"base"` | Base model, no adapter (reserved keyword) |
| `"mlx-community/Qwen3.5-35B-A3B-4bit"` | Base model (exact `model_path` from `/health`) |
| `"rust"` | `rust` adapter (direct name match) |
| `"qwen/rust"` | `rust` adapter (`base/adapter` pattern) |
| `"qwen/typo"` | **404** — `typo` is not a loaded adapter |
| `"sqll"` | **404** — unknown bare name (typo caught) |

Supports `stream: true` for SSE streaming.

### `GET /v1/adapters`

List all loaded adapters with metadata (rank, scale, memory usage).

### `POST /v1/adapters`

Hot-load a new adapter. Body: `{"name": "...", "path": "..."}`.

### `DELETE /v1/adapters/{name}`

Unload an adapter and free its memory.

### `GET /health`

Server health check with model and adapter count.

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

Train adapters with [mlx-lm](https://github.com/ml-explore/mlx-lm), [mlx-tune](https://github.com/ARahim3/mlx-tune), or any tool that outputs PEFT-compatible safetensors.

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

- **Alpha** — API contract is stable but internals may change
- **Apple Silicon only** — requires MLX (no CUDA, no CPU fallback)
- **KV cache** — switching adapters mid-conversation invalidates the KV cache (the conversation restarts). Each conversation should use one adapter throughout.
- **No batching across adapters** — concurrent requests with different adapters are served sequentially (single async lock). For single-user local inference this is not a bottleneck.
- **Strict adapter validation** — an adapter must inject into all its expected target layers. Partially compatible adapters are rejected at load time to prevent silent incorrect outputs.
- **No custom kernels** — uses standard MLX `matmul` operations. Performance is good for local use but not optimized for high-throughput serving.

## Roadmap

- [ ] HuggingFace Hub adapter loading (load by ID, not just local path)
- [ ] Adapter merge strategies (TIES, DARE — combine multiple adapters per request)
- [ ] Per-adapter KV cache (avoid invalidation on switch)
- [ ] Expert-layer LoRA for MoE models (SwitchLinear wrapping)
- [ ] QLoRA / DoRA adapter support
- [ ] Prometheus metrics endpoint
- [ ] Benchmarks (latency overhead per adapter, memory scaling)

## Contributing

MOLA is in early development. Contributions, bug reports, and feedback are welcome.

```bash
git clone https://github.com/YOUR_USERNAME/mola.git
cd mola
pip install -e ".[dev]"
pytest
```

## License

Apache 2.0
