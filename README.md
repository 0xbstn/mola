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

> **Status:** Alpha. The published benchmark below uses `mlx-community/Qwen2.5-0.5B-Instruct-4bit` with 8 resident adapters on Apple Silicon and `mlx-lm 0.31.1`.

## What MOLA Does

- Serve one base model with many LoRA adapters loaded at the same time
- Select the adapter per request through an OpenAI-compatible chat API
- Batch same-adapter traffic and keep mixed-adapter serving stable
- Hot-load and hot-unload adapters without restarting the server

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
./.venv/bin/python devtools/run_mola_current_architecture.py start --port 8000
```

Equivalent explicit command:

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
  -d '{"name": "medical", "path": "./adapters/medical-lora"}'

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
  --concurrency 64,128 \
  --repeats 3 \
  --json-out /tmp/mola-current-architecture-bench.json
```

Published profile:
- model: `mlx-community/Qwen2.5-0.5B-Instruct-4bit`
- adapters: `rust`, `sql`, `medical`, `cyber`, `solidity`, `devops`, `math`, `legal`
- backend: `metal-gather`
- batch sizes: `128 / 32`

| Scenario | conc=64 | conc=128 |
|---|---|---|
| Same | 33.6 req/s, 1984.8 tok/s, p95 1927.3 ms | 31.9 req/s, 1937.4 tok/s, p95 3957.6 ms |
| Mixed | 27.8 req/s, 1351.4 tok/s, p95 2284.9 ms | 26.7 req/s, 1311.5 tok/s, p95 4614.4 ms |
| Long decode mixed | 12.2 req/s, 1752.8 tok/s, p95 5290.1 ms | 13.4 req/s, 1996.5 tok/s, p95 9425.9 ms |
| Hot/cold skew mix | 29.7 req/s, 1533.4 tok/s, p95 2184.8 ms | 28.4 req/s, 1472.7 tok/s, p95 4346.7 ms |

Key ratios:

- `mixed / same @64` ≈ `0.83`
- `mixed / same @128` ≈ `0.84`

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

## Contributing

```bash
git clone https://github.com/0xbstn/mola.git
cd mola
./.venv/bin/python -m pip install -e ".[dev]"
./.venv/bin/python -m pytest
```

## License

Apache 2.0
