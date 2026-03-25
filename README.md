<p align="center">
  <h1 align="center">MOLA</h1>
  <p align="center"><b>Multi-adapter Orchestration LoRA Apple</b></p>
  <p align="center">Multi-LoRA inference server for Apple Silicon: one base model, many adapters, no reload.</p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-alpha-red" alt="Status: Alpha">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple" alt="Apple Silicon">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue?logo=python" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License">
  <img src="https://img.shields.io/badge/MLX-native-red" alt="MLX Native">
</p>

---

MOLA serves multiple LoRA adapters from one MLX base model on Apple Silicon. The base model stays resident in memory, adapters are selected per request, and same-adapter traffic is batched automatically.

> **Status:** Alpha. The published benchmark below uses `mlx-community/Qwen3.5-9B-MLX-4bit` with 8 resident adapters on `mlx-lm 0.31.1`.

| Approach | Runtime shape |
|---|---|
| Separate fine-tuned models | one full model per specialty, higher memory, reloads when switching |
| MOLA | one base model plus many LoRA adapters, lower memory, no model reloads |

This is the practical tradeoff MOLA is built for: keep one base model resident, switch adapters per request, and avoid reloading full fine-tuned checkpoints.

## Features

- OpenAI-compatible chat completions API
- Per-request adapter selection via the `model` field
- Multiple LoRA adapters loaded at once on one base model
- Same-adapter batching and stable mixed-adapter serving
- Hot-load and hot-unload adapters at runtime
- Streaming responses
- Runtime metrics via `/v1/engine/metrics`

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

### Recommended Setup

Start the recommended runtime profile:

```bash
./.venv/bin/python devtools/run_mola_current_architecture.py start \
  --model mlx-community/Qwen3.5-9B-MLX-4bit \
  --adapter rust ./path/to/rust-adapter \
  --adapter sql ./path/to/sql-adapter \
  --port 8000
```

Equivalent explicit command:

```bash
./.venv/bin/python -m mola.cli -v serve \
  --model mlx-community/Qwen3.5-9B-MLX-4bit \
  --adapter rust ./path/to/rust-adapter \
  --adapter sql ./path/to/sql-adapter \
  --adapter support ./path/to/support-adapter \
  --max-inflight-tokens 131072 \
  --max-batch-size 128 \
  --prefill-batch-size 32 \
  --enable-routed-decode-reference \
  --strict-routed-decode-reference \
  --routed-decode-backend gather-mm \
  --enable-mixed-decode-migration \
  --prestep-mixed-decode-migration \
  --cache-routed-decode-sessions \
  --detached-shared-decode-owner \
  --port 8000
```

Stop or restart:

```bash
./.venv/bin/python devtools/run_mola_current_architecture.py stop --port 8000
./.venv/bin/python devtools/run_mola_current_architecture.py restart --port 8000
```

### Query

The `model` field is a strict selector. Unknown adapter names return `404`.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rust",
    "messages": [{"role": "user", "content": "Implement a lock-free queue with crossbeam"}],
    "stream": true
  }'
```

### Manage Adapters At Runtime

```bash
curl http://localhost:8000/v1/adapters

curl -X POST http://localhost:8000/v1/adapters \
  -H "Content-Type: application/json" \
  -d '{"name": "medical", "path": "./path/to/medical-adapter"}'

curl -X DELETE http://localhost:8000/v1/adapters/medical
```

## API Summary

- `POST /v1/chat/completions`: OpenAI-compatible chat completions
- `GET /v1/models`: base model and loaded adapters
- `GET /v1/adapters`: loaded adapter metadata
- `POST /v1/adapters`: hot-load an adapter
- `DELETE /v1/adapters/{name}`: unload an adapter
- `GET /v1/engine/metrics`: runtime counters
- `GET /health`: health and model summary

## Published Benchmark

Reproduce the published benchmark with:

```bash
./.venv/bin/python scripts/bench_server.py \
  --routed-validation \
  --concurrency 1,16,64 \
  --repeats 3 \
  --json-out /tmp/mola-qwen35-9b-current-architecture-bench-1-16-64.json
```

Published profile:
- model: `mlx-community/Qwen3.5-9B-MLX-4bit`
- adapters: `rust`, `sql`, `medical`, `cyber`, `solidity`, `devops`, `math`, `legal`
- backend: `gather-mm`
- batch sizes: `128 / 32`
- machine: `Apple M5 Max 64GB`

This benchmark shows how much performance MOLA keeps when traffic moves from one adapter at a time to a real mixed multi-adapter workload.

| Concurrency | Same tok/s | Mixed tok/s | Multi-LoRA overhead | Mixed p95 |
|---|---:|---:|---:|---:|
| 1 | 76.4 | 76.4 | 0% | 843 ms |
| 16 | 308.8 | 241.4 | -22% | 4220 ms |
| 64 | 732.3 | 555.5 | -24% | 7372 ms |

At concurrency 1, same and mixed are effectively the same shape; the useful signal starts once requests overlap.

At moderate to high load, mixed multi-adapter traffic adds about 22-24% throughput overhead relative to same-adapter traffic.

From the same run:

- `long-decode-mixed` tok/s: `81.1` at `1`, `283.4` at `16`, `691.1` at `64`
- `hot/cold skew mix` tok/s: `77.4` at `1`, `241.3` at `16`, `559.1` at `64`

If you benchmark MOLA on another Apple Silicon machine or model, feel free to open an issue with your hardware, model, and results.

## Adapter Format

MOLA loads standard PEFT / mlx-lm adapter directories:

```text
my-adapter/
├── adapter_config.json
└── adapters.safetensors
```

Train adapters with [mlx-lm](https://github.com/ml-explore/mlx-lm), [mlx-tune](https://github.com/ARahim3/mlx-tune), or any tool that outputs PEFT-compatible safetensors.

## Requirements

- Apple Silicon Mac (M1 or later)
- Python 3.11+
- macOS 13+
- `mlx-lm 0.31.1` recommended

## Limitations

- Alpha release
- Apple Silicon only
- A local `mlx-lm` patch is still required for the recommended setup
- Switching adapters inside one conversation invalidates KV cache reuse
- Mixed prefill and deeper KV/adaptor residency management are still open problems

## License

Apache 2.0
