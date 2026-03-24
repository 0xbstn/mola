# MOLA Kernel Roadmap

Date: 2026-03-24

Scope: practical roadmap from the current `v0.3.2` runtime to a first kernel-backed mixed-adapter path.

This note combines:
- current MOLA benchmark evidence
- official MLX / Metal documentation
- upstream `mlx_lm` / MLX source code
- public CUDA multi-LoRA serving systems (`Punica`, `S-LoRA`, `vLLM`, `SGLang`, `TensorRT-LLM`, `LoRAX`)

When a statement is a conclusion rather than a directly documented fact, it is marked `[Inference]`.

## 1. What We Know Now

### 1.0 Current pre-kernel ABI work already landed

The runtime now has the first stable pieces of an adapter-slot ABI:
- adapters get dense integer `slot_id`s at load time
- `AdapterManager` exposes both `adapter_id -> slot_id` and `slot_id -> adapter_id`
- `AdapterManager.slot_bindings()` exposes a stable, sorted control-plane snapshot for later packing work
- `MOLAModel` propagates `slot_id` through `adapter_context(...)`
- `MOLAEngine` resolves `runtime_slot_id` for active slots and exposes it in slot metrics
- `MultiLoRALinear` / `MultiLoRASwitchLinear` maintain slot-based bindings in addition to name-based bindings
- `MOLAEngine.runtime_slot_layout()` exposes loaded vs active vs pending slot IDs
- `MOLAEngine.active_slot_bindings()` filters loaded slot bindings to the slots that matter for the current runtime state
- `MOLAModel.iter_slot_bound_lora_layers()` exposes slot-bound LoRA layers without leaking layer internals into unrelated modules
- `mola.application.packing.build_layer_slot_pack_views(...)` builds per-layer packer inputs from active slots plus slot-bound layers
- `mola.application.packing.materialize_layer_slot_packs(...)` turns those views into backend-agnostic packed tensors via an injected `stack_fn`

[Inference]
This is enough to stop treating adapter names as the only execution identity. It is not yet packed execution, but it is the minimum ABI cleanup needed before packed layer buffers or a routed decode operator.

### 1.0.1 Executable pack contract now exists

The pack layer now exposes the missing execution-time helpers:
- `MaterializedLayerSlotPack.slot_row_by_id` gives the explicit sparse `slot_id -> pack row` map
- `materialize_layer_slot_packs(..., scale_fn=...)` can materialize per-slot scales as arrays instead of leaving them as Python tuples
- `build_routed_decode_plan(...)` groups decode rows by packed slot row, returns the grouped order, and provides `restore_order` for unsorting after a grouped compute path
- `routed_decode_delta_reference(...)` is now the decode-only correctness oracle that consumes a packed layer plus `token_slot_ids`
- `routed_decode_delta_rows_reference(...)` handles the row-major flatten/restore case expected by decode hidden states
- `build_layer_slot_pack_state(...)` gives a stable `layer_name -> pack` lookup surface for the forward path
- `GeneratorPort.active_handles()` now exposes the current decode batch row order without waiting for post-step events

[Inference]
This is the first point where MOLA has enough information to implement a routed decode reference path without inventing another runtime identity layer.

### 1.0.2 Routed execution now sits behind an explicit session boundary

The experimental routed path no longer threads raw pack state and token slot IDs through `MultiLoRALinear`.
The runtime now has:
- `RoutedLoRADeltaSession` as the narrow pre-kernel execution port
- `ReferenceRoutedLoRADeltaSession` as the MLX-backed reference implementation
- a pure `resolve_routed_layer_execution(...)` helper that validates layer lookup and row-count compatibility before a backend runs
- `routed_decode_context(session)` in `mola.context`, so the forward path depends on one opaque execution object instead of multiple low-level values
- `MOLAEngine.build_routed_decode_session(...)`, `build_active_decode_session(...)`, and `build_homogeneous_decode_session(...)`
- explicit failure when a requested routed `slot_id` has no current binding
- an explicit `GeneratorPort.active_handles()` contract that its order must match the decode row order used by the next `step()`

[Inference]
This is the right boundary to preserve before any kernel work. The next backend can replace the session implementation without changing `MultiLoRALinear` again.

### 1.0.3 Live validation of the routed reference path is negative on performance

On March 24, 2026, the current homogeneous routed reference path was validated live on the `Qwen2.5-0.5B-Instruct-4bit` server with eight resident adapters and `routed_decode_reference_enabled=true`.

Compared to the stored `v0.3.2` baseline, the routed reference path was slower across the relevant scenarios:
- `same @8`: `13.58 req/s` -> `10.24 req/s`
- `mixed @8`: `4.84 req/s` -> `2.37 req/s`
- `fairness @8`: `6.88 req/s` -> `3.63 req/s`
- `long-decode-mixed @8`: `2.84 req/s` -> `1.41 req/s`
- `same @16`: `18.12 req/s` -> `14.58 req/s`
- `mixed @16`: `6.98 req/s` -> `3.62 req/s`
- `fairness @16`: `11.20 req/s` -> `5.97 req/s`
- `long-decode-mixed @16`: `3.92 req/s` -> `2.18 req/s`

[Inference]
This is the expected result for the current routed session path. It proves that the execution seam works and that the routed path can be exercised under load, but it does not provide a faster runtime yet. The routed reference path should therefore be treated as a correctness and ABI scaffold for the future kernel, not as a performance feature to enable by default.

### 1.1 The real bottleneck is still mixed decode

From the current `v0.3.2` benchmark notes in `benchmark/0.3.2/notes.md`:
- `same` traffic is healthy
- `mixed` traffic is usable
- `long-decode-mixed` is still the strongest ceiling

Representative numbers:
- `mixed @16`: `7.0 req/s`, `346.4 tok/s`, `p95 2334.4 ms`
- `long-decode-mixed @16`: `3.9 req/s`, `358.1 tok/s`, `p95 4326.4 ms`

### 1.2 The Python `model_lock` does not appear to be the primary wall

We added lock-wait instrumentation and saved a targeted run in:
- `benchmark/0.3.2/bench-20260324-200928-lockwait16.json`

At `conc=16`:
- `base`: `engine_step_lock_wait_ms = 0.01`
- `same`: `engine_step_lock_wait_ms = 0.0`
- `mixed`: `engine_step_lock_wait_ms = 0.04`
- `engine_insert_lock_wait_ms = 0.0` across the scenarios

Interpretation:
- the current mixed slowdown is not mostly explained by Python-side lock contention
- the remaining problem is more likely in operator shape, cache/memory behavior, or structural loss of batching

### 1.3 Admission still matters under heavy prefill

At `conc=32`, the default token budget (`32768`) caused many `long-prefill` rejections, while `65536` removed them.
Saved run:
- `benchmark/0.3.2/bench-20260324-200439-65536.json`

Interpretation:
- pre-kernel work is not over
- `max_inflight_tokens` is a real deployment knob
- unified-memory pressure must be treated as part of the kernel decision, not as an afterthought

## 1.4 Ecosystem calibration: who is actually doing the same thing as MOLA?

The current Apple-Silicon kernel ecosystem is useful for MOLA, but most public projects are not actually "multi-LoRA serving on Apple Silicon". They split into three buckets:

### Same problem as MOLA

- **No direct public equivalent found in this pass.**

[Inference]
There are strong Apple-Silicon inference projects and strong low-level Metal kernel projects, but I did not find a public project that matches all of MOLA's constraints at once:
- Apple Silicon first
- OpenAI-compatible serving
- one base model plus many resident LoRA adapters
- adapter selection per request
- mixed-adapter batching as the main optimization target

### Adjacent serving engines

- **`vllm-metal`** is the closest serving reference on Apple Silicon, but it is not a multi-LoRA server. It is a hardware plugin that runs `vLLM` on Apple Silicon using MLX, keeps the OpenAI-compatible API, and adds experimental paged attention plus memory-fraction controls. Its current Metal paged-attention kernels are adapted from `mistral.rs`, not invented from scratch in this repo.
- **MetalRT** is also adjacent, but not the same product. It is a native Apple-Silicon inference engine with direct Metal GPU programming and self-reported decode wins of about `1.10-1.19x` over `mlx-lm` on the same model files. It is valuable as evidence that hand-tuned Apple-Silicon inference still has headroom over MLX baseline engines.
- **RunAnywhere** is adjacent as an on-device SDK/platform rather than a multi-LoRA server. Its public docs show a plugin architecture, a native Swift SDK with Metal acceleration on Apple Silicon, and LoRA hot-swap plus stackable adapters in the Kotlin SDK. That is useful for adapter lifecycle and packaging ideas, but it is still not a routed mixed-adapter serving runtime.

### Kernel inspiration, not the same product

- **OpenEvolve** is not a serving engine. It is an automated kernel-discovery workflow that evolved a faster MLX/Metal attention kernel and reported `+12.5%` average decode improvement with full maintained-output accuracy on the benchmarked workloads.
- **Gimlet Labs** is also not a serving engine. It is a benchmark showing that frontier models can synthesize faster Metal kernels than a PyTorch eager baseline on many tasks; their report shows `1.22x` average speedup on KernelBench v0.1, with GPT-5 leading the individual models and an agentic swarm doing even better.
- **`pmetal`** is not the same serving problem either. It is a broader Apple-Silicon fine-tuning / inference framework, but it is highly relevant because it openly documents a manual kernel stack including FlashAttention, fused LoRA, fused RMSNorm+LoRA, and tier-aware threadgroup tuning.
- **Official MLX custom kernel paths** are not a project competitor, but they are the canonical integration path MOLA should target first: `mx.fast.metal_kernel` for the first prototype, then a custom `Primitive` / extension if the Python-level prototype proves worthwhile.

### Compact verdict grid

| Project | Verdict vs MOLA | Why | What to reuse |
| --- | --- | --- | --- |
| OpenEvolve | `adjacent` | Optimizes one GPU kernel in a loop; it is not a serving runtime | Loop `variant -> compile -> bench -> keep/drop` |
| Gimlet Labs | `inspiration kernel` | Benchmarks AI-generated Metal kernels against a baseline; not a serving system | Kernel generation benchmarks and model-vs-baseline comparison |
| MetalRT | `adjacent` | General Apple-Silicon inference engine with direct Metal GPU programming | Benchmark discipline, direct Metal hot path |
| `pmetal` | `adjacent` | Broader Apple-Silicon ML platform with manual fused kernels and LoRA tooling | Manual fused-kernel inventory, device-aware tuning, LoRA fusion ideas |
| `vllm-metal` | `adjacent` | Apple-Silicon serving plugin for vLLM, not a multi-adapter LoRA runtime | Memory controls, serving separation, pragmatic kernel adaptation |
| MLX official | `inspiration kernel` | Canonical backend and extension path, not a competing product | `mx.fast.metal_kernel`, `Primitive`, routed ops like `gather_mm` |

[Inference]
There is no public project in this pass that matches all of MOLA's constraints at once:
- Apple Silicon first
- OpenAI-compatible serving
- one base model plus many resident LoRA adapters
- adapter selection per request
- mixed-adapter batching as the main optimization target

## 1.5 What these projects still do by hand

The useful lesson from the current ecosystem is not "AI replaces expert GPU work". It is:
- AI is increasingly used to accelerate iteration
- the winning projects still preserve a narrow, hand-verified kernel contract
- the best public Apple-Silicon work still does the final performance shaping manually

Concrete patterns:

- **MetalRT** explicitly positions itself as "direct Metal GPU programming" with no wrapper layer. The lesson for MOLA is that serious Apple-Silicon wins still come from owning the operator path directly.
- **RunAnywhere** is the clearest public example in this pass of adapter lifecycle done as a product feature: hot-swap, adapter catalog, stackable adapters, and modular backends. The MOLA-relevant lesson is to keep adapter residency and control-plane policy explicit, not implicit.
- **`pmetal`** exposes the most MOLA-relevant manual ideas in public:
  - tier-aware tuning by GPU family / device class
  - fused LoRA and fused RMSNorm+LoRA
  - explicit fused-kernel inventory instead of generic abstraction
- **`vllm-metal`** shows a pragmatic middle path:
  - adapt a proven kernel first (`mistral.rs` paged attention)
  - keep a product-level memory model and serving interface around it
  - develop custom kernels later only where they clearly matter
- **OpenEvolve** shows the most useful AI pattern for MOLA:
  - not one-shot prompting
  - a closed loop of propose -> compile -> benchmark -> evolve

[Inference]
For MOLA, the right reading is:
- do not expect one prompt to yield the final routed LoRA kernel
- do expect AI to help heavily with candidate generation, variants, and harness iteration
- keep the first kernel contract very small and very explicit so human review stays tractable

## 1.6 What is directly useful for MOLA

From these projects, the most transferable ideas are:

1. **Adopt-and-adapt beats invent-from-zero**
- `vllm-metal` adapted paged-attention kernels from `mistral.rs`
- MOLA should likewise prefer a narrow routed LoRA decode kernel before attempting any full fused engine rewrite

2. **Keep the first kernel small**
- OpenEvolve succeeded on one attention kernel, not a whole inference runtime
- MOLA should target decode-only LoRA delta first, not base matmul fusion

3. **Treat tuning as device-aware**
- `pmetal` explicitly tunes by GPU family and device tier
- `[Inference]` MOLA should expect Apple GPU family / tier differences to matter for threadgroup sizes and layout choices if it reaches a real custom-kernel phase

4. **Keep serving controls outside the kernel**
- `vllm-metal` separates memory-fraction controls, prefix cache, and paged attention configuration from the low-level operator implementation
- MOLA should keep admission, fairness, adapter residency, and API behavior in the runtime, not inside kernel logic

5. **Expect a mix of AI and manual work**
- Gimlet and OpenEvolve are the best evidence that AI can discover useful Metal kernels
- MetalRT and `pmetal` are the best evidence that the final winning path is still narrow, benchmarked, and manually shaped

Primary sources:
- Gimlet Labs: `https://gimletlabs.ai/blog/ai-generated-metal-kernels`
- OpenEvolve HF blog: `https://huggingface.co/blog/codelion/openevolve-gpu-kernel-discovery`
- MetalRT HF blog: `https://huggingface.co/blog/runanywhere/metalrt-fastest-inference-apple-silicon`
- `pmetal`: `https://github.com/Epistates/pmetal`
- `vllm-metal`: `https://github.com/vllm-project/vllm-metal`
- MLX custom kernels: `https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html`
- MLX extensions: `https://ml-explore.github.io/mlx/build/html/dev/extensions.html`

## 2. Best Pre-Kernel Path

## 2.1 Do not jump straight to a Metal kernel

The strongest current path before custom kernels is:
1. stabilize runtime adapter-slot representation
2. pack adapter weights by slot and layer
3. prototype mixed-adapter math using existing MLX ops where possible
4. only then write a custom Metal kernel if the prototype leaves clear headroom

Why:
- `mlx_lm` already uses routed matrix operators for MoE via `mx.gather_mm` and `mx.gather_qmm`
- `LoRASwitchLinear` already shows a no-new-kernel composition path for routed LoRA
- current benchmarks say the problem is structural mixed decode, not a Python lock

## 2.2 Most plausible no-kernel prototype

The cleanest no-kernel prototype is not to patch `BatchGenerator` first.
It is to change the layer contract in MOLA from:
- "one adapter selected by contextvars for the whole forward"

to:
- "a batch carries integer `slot_ids`, and the LoRA delta path can route by slot"

Closest upstream references:
- `mlx_lm.models.switch_layers.SwitchLinear` uses `mx.gather_mm(..., rhs_indices=indices, sorted_indices=sorted_indices)`
- `mlx_lm.models.switch_layers.QuantizedSwitchLinear` uses `mx.gather_qmm(..., rhs_indices=indices, sorted_indices=sorted_indices)`
- `mlx_lm.tuner.lora.LoRASwitchLinear` applies the LoRA delta as two routed gathered matmuls, not as a fused kernel
- the upstream MoE blocks sort routed indices once enough tokens are present, then unsort after the routed compute

[Inference]
For MOLA, the best no-kernel prototype is a serving-time `LoRAMixtureLinear` that:
- keeps the base dense path unchanged
- stores packed LoRA A/B tensors by active GPU slot
- takes token/request slot IDs as metadata
- uses `gather_mm`-style routed low-rank projections for mixed batches

This is not expected to be the final fast path. It is the best correctness and architecture bridge before kernel work.

## 3. Runtime Contract MOLA Needs First

Before any kernel, MOLA needs a stable runtime ABI.

### 3.1 Identity levels

- `adapter_id`: stable global identity, user-facing, control-plane key
- `slot_id`: small integer GPU/runtime identity used in the current batch/window

The loaded-vs-active split should be explicit:
- CPU cache / loaded adapters can be larger
- GPU active slot pool should be small and dense

This is the most transferable pattern from:
- `vLLM`
- `SGLang`
- `TensorRT-LLM`
- `LoRAX`

### 3.2 Packed runtime tensors

Per adapted layer, MOLA should move toward slot-major buffers:
- `A_buffer[layer][slot, ...]`
- `B_buffer[layer][slot, ...]`
- `slot_scale[slot]`
- `slot_rank[slot]`

Exact layout depends on whether the first prototype is dense, packed-QKV, or MoE-like.
But the important point is this:
- no adapter-name dict lookups in the hot path
- no per-request object chasing in the hot path

### 3.3 Batch metadata views

Two views are needed:

Decode view:
- `token_slot_ids[num_tokens]`

Prefill view:
- grouped tokens by slot, or
- segment metadata such as `segment_indptr`

This matches the public split across systems:
- `BGMV`-like token-to-adapter mapping for decode
- `SGMV`-like grouped/segmented batching for prefill

## 4. First Kernel Target

## 4.1 Target the decode delta, not the base model

The first custom kernel should be:
- decode-only
- LoRA-delta-only
- not fused with the base matmul

Reason:
- `long-decode-mixed` is the clearest current wall
- decode shape is simpler: one token per sequence is common
- the kernel contract can be kept small
- it avoids coupling the first kernel to the whole base-model matmul path

This matches the advice from current system evidence and your own benchmark profile.

## 4.2 First kernel ABI

A practical first-kernel ABI for MOLA decode is:

Inputs:
- `x`: token features for active decode rows, shape `[T, H]`
- `slot_ids`: integer slot id per row, shape `[T]`
- `lora_a`: packed slot-major LoRA A for one layer, shape `[S, H, R]` or equivalent packed variant
- `lora_b`: packed slot-major LoRA B for one layer, shape `[S, R, O]`
- `scales`: per-slot scaling, shape `[S]`
- optional `ranks`: per-slot rank, shape `[S]` if heterogeneous rank is allowed early

Output:
- `delta`: shape `[T, O]`

Semantics:
- for each row `t`, use `slot_ids[t]` to choose the adapter slot
- compute the low-rank delta only
- base output remains separate for the first version

This is conceptually the MOLA equivalent of a `BGMV`-style routed LoRA decode operator.

## 4.3 Grouped fast path vs unsorted path

The first kernel path should probably support two modes:

1. homogeneous fast path
- all rows use the same slot
- simplest and cheapest case

2. mixed decode path
- rows carry slot ids
- kernel gathers/routs by slot id

[Inference]
If decode rows can be sorted or grouped by slot cheaply, a grouped path may outperform a purely random per-row gather path on Apple GPU. Public MLX Metal kernels already show gathered and segmented GEMM patterns, and Apple GPU benefits from coherent access.

## 5. MLX/Metal Implementation Order

## 5.1 Phase A: Python-level `metal_kernel` prototype

Use `mx.fast.metal_kernel` first.

Why:
- official MLX path
- accepts multiple tensor inputs and metadata tensors
- supports small runtime metadata as ordinary input arrays
- quickest way to validate kernel shape and launch contract

Important documented facts:
- `metal_kernel` auto-generates the signature from input/output tensors
- extra runtime metadata can be passed as additional input arrays
- `ensure_row_contiguous=True` is available
- `atomic_outputs` is supported if needed
- upstream MLX already has specialized GPU paths for gathered RHS matmul when decode shape is favorable (`M == 1`, sorted RHS indices)

Implication for MOLA:
- `slot_ids` can be passed as just another input tensor
- the reference path should preserve grouped/sorted decode order so a later kernel or MLX gathered path can exploit coherent RHS access
- this is enough for a first decode-delta experiment

## 5.2 Phase B: real MLX custom `Primitive`

If the Python-level prototype shows real gains, move to a custom MLX extension:
- `Primitive`
- `eval_gpu()`
- custom `.metallib`
- `nanobind`

Why:
- better control over dispatch and integration
- less Python glue in the hot path
- better foundation for multiple kernels later

This is the path to a maintainable kernel-backed runtime.

## 5.3 Do not start with full fusion

Do **not** begin with:
- base matmul + LoRA fused together
- prefill mixed kernels
- fully paged heterogeneous-rank kernels

Those are phase-2 or phase-3 problems.

## 6. Public Signal: AI-Assisted Metal/MLX Kernel Work

The strongest public signal I found is not that AI replaces kernel engineering. It is that AI helps a lot when the team already has a tight correctness/perf loop.

### 6.1 OpenEvolve

Source:
- Hugging Face blog: https://huggingface.co/blog/codelion/openevolve-gpu-kernel-discovery

What it is:
- An evolution loop that mutates kernel code with LLMs and benchmarks the candidates.
- The blog shows MLX/Metal attention work, with AI driving the search and the benchmark loop.

Why it matters:
- This is the closest public proof that "LLM + benchmark loop" works for MLX/Metal kernel discovery.
- It is not MOLA's exact problem. It is adjacent: one operator, not a multi-adapter serving runtime.

### 6.2 Gimlet Labs

Source:
- Blog: https://gimletlabs.ai/blog/ai-generated-metal-kernels

What it is:
- A benchmark of AI-generated Metal kernels for PyTorch/MLX-style workloads on Apple Silicon.
- The post explicitly evaluates GPT-5 and Claude-class models as kernel generators.

Why it matters:
- It confirms that public kernel work on Apple Silicon is already being accelerated by AI.
- The project is still adjacent to MOLA, not the same system problem.

### 6.3 metalQwen3

Source:
- Repo: https://github.com/BoltzmannEntropy/metalQwen3

What it is:
- A Metal-based Qwen3 inference project.
- The repo README explicitly says Claude Code accelerated development by several times.

Why it matters:
- This is strong evidence that even small Apple Silicon kernel/inference projects are using AI as a force multiplier.
- It is closer to MOLA than generic kernel-bench blogs, but still not the same problem: it is a single-model inference stack, not multi-LoRA serving.

### 6.4 PMetal / Epistates

Source:
- Repo: https://github.com/Epistates/pmetal

What it is:
- A broad Apple Silicon ML SDK/framework/app suite in Rust.
- The README describes custom Metal GPU kernels, ANE integration, training APIs, GUI/TUI, LoRA/QLoRA, quantization, fusion, and an OpenAI-compatible server.
- The interesting part for MOLA is not the product surface, but the implementation style: explicit hardware-tuned Metal paths, sequence packing, fused LoRA, fused RMSNorm+LoRA, tier-aware kernel tuning, and backend separation.
- The repository also exposes a distributed-training crate, but that is separate from the serving path and is not the part MOLA should copy first.

Why it matters:
- This is the clearest example in the set of a more manual, expert-driven Metal codebase rather than an AI-evolve-first workflow.
- It is not the same problem as MOLA. PMetal is a full ML platform; MOLA is a multi-LoRA serving runtime.
- The training/fine-tuning side is interesting as reference material, but it is not the same serving problem as MOLA.
- The distributed crate is useful only as a future reference if MOLA ever wants multi-Mac training or distributed collective primitives. It is not a direct serving-path reference today.

Classification:
- `adjacent` for kernel and optimization ideas
- `not pertinent` for multi-LoRA serving architecture

### 6.5 vllm-metal

Source:
- Repo: https://github.com/vllm-project/vllm-metal

What it is:
- A community-maintained hardware plugin for vLLM on Apple Silicon.
- The README says MLX is the primary compute backend, vLLM provides the engine/scheduler/API layer, and paged attention is experimental and opt-in.
- The README also says the Metal paged-attention kernels are adapted from `mistral.rs` via `kernels-community`, with future custom kernels planned.

Why it matters:
- Useful for MOLA as a reference for Apple Silicon backend integration, memory knobs, and how a serving stack can sit above MLX.
- Useful as a reminder that paged-attention support is not the default path yet; it is an opt-in backend feature with kernel work still in flux.
- It is not the same problem as MOLA. It is a vLLM plugin/runtime integration effort, not a multi-LoRA serving runtime.
- Transferable ideas for MOLA are architectural, not product-level: keep a hardware tier matrix, isolate backend-specific fast paths, and treat sequence packing as a first-class optimization.

Classification:
- `adjacent`
- strong reference for Apple Silicon serving integration
- not a multi-LoRA runtime

### 6.6 Short classification against MOLA

Closest to MOLA's exact serving problem:
- none found in the public GitHub projects checked here

Adjacent and useful:
- `pmetal` for manual Metal kernels, fused LoRA, and Apple Silicon hardware-tuned training/inference
- `vllm-metal` for backend integration, memory controls, and serving-stack structure on Apple Silicon
- `MetalRT` for direct Metal inference engine design and hand-tuned Apple Silicon performance
- `OpenEvolve` for AI-assisted kernel-discovery workflow
- `metalQwen3` for small-team, AI-accelerated Metal inference development

Not pertinent to MOLA's core serving problem:
- general training-only or full-platform projects that do not expose multi-adapter serving as the main runtime problem

### 6.5 What This Suggests For MOLA

- No public project I found is doing MOLA's exact full problem end-to-end: multi-LoRA serving, adapter scheduling, Apple Silicon runtime, and a path toward routed mixed-adapter execution.
- The public evidence does suggest the workflow we should use:
  - write a tight benchmark loop
  - generate/modify one operator at a time
  - keep a human-owned correctness harness
  - use AI for variants, boilerplate, and low-level iteration
- The projects that look "manual" rather than AI-heavy are still useful to MOLA, but mostly as architecture references, not as proof that we need to write the whole stack by hand.

### 6.6 MLX Official Pieces Worth Reusing

Source set:
- MLX examples repo: <https://github.com/ml-explore/mlx-examples>
- MLX-LM switch layers: <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/switch_layers.py>
- MLX-LM LoRA tuner: <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/lora.py>
- MLX custom Metal kernels: <https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html>
- MLX custom extensions: <https://ml-explore.github.io/mlx/build/html/dev/extensions.html>
- MLX Metal debugger: <https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html>

Classification for MOLA:

- `mlx-examples` LoRA / QLoRA example code: `adjacent`
- `mlx_lm.models.switch_layers.SwitchLinear` and `QuantizedSwitchLinear`: `adjacent`
- `mlx_lm.tuner.lora.LoRASwitchLinear`: `adjacent`
- `mlx_lm.generate.BatchGenerator`: `adjacent`
- `mx.gather_mm` / `mx.gather_qmm` routed-expert patterns: `adjacent`
- `mx.fast.metal_kernel`: `adjacent`
- MLX `Primitive` / custom extensions / `eval_gpu()`: `adjacent`
- MLX Metal debugger: `adjacent`
- General MLX demo projects unrelated to text inference, routing, or LoRA: `not pertinent`

Why this matters:
- None of the official MLX pieces is the exact same problem as MOLA's multi-LoRA serving runtime.
- The closest no-kernel template is the routed expert path in `switch_layers.py` and the routed LoRA composition in `tuner/lora.py`.
- `BatchGenerator` is the serving baseline to keep in mind, but it is still a single-adapter execution model.
- The closest kernel path is `mx.fast.metal_kernel` plus the `Primitive` extension path, but those are still just mechanisms, not a multi-LoRA serving solution.

### 6.7 What to steal manually first

The most useful manual ideas for MOLA, in priority order, are:

- **From MetalRT:** own the hot path directly and compare against the same model files, not different model formats.
- **From pmetal:** keep a literal inventory of fused kernels and hardware-specific paths, especially fused LoRA and fused Norm+LoRA.
- **From vllm-metal:** treat memory and paging as first-class serving concerns, but keep the backend/plugin boundary clean.
- **From OpenEvolve:** use a strict propose -> compile -> benchmark -> keep/reject loop, not a vague prompt-only workflow.
- **From Gimlet:** treat kernel generation as search over a benchmark, and expect the AI-generated variant to need human review and pruning.

This is the right reading of the ecosystem for MOLA:
- AI is a force multiplier for kernel iteration.
- the final performance shape still tends to be manual.
- the serving architecture around the kernel remains a separate product problem.

## 7. Practical Next Steps

### 7.1 Before the first kernel

1. Keep the current `v0.3.2` baseline.
2. Keep the new lock-wait instrumentation.
3. Add more profiling evidence if possible:
   - step time split
   - MLX/Metal capture
   - memory counters / working-set pressure
4. Refactor the runtime toward slot-based representation:
   - `adapter_id -> slot_id`
   - packed per-layer A/B buffers
   - batch metadata with slot ids

### 7.2 First no-kernel prototype

Implement a prototype that:
- keeps base dense path unchanged
- introduces slot ids and packed adapter storage
- routes mixed-adapter LoRA with existing MLX operators where feasible
- benchmarks:
  - `mixed-8-distinct`
  - `long-decode-mixed`
  - `fairness`

Success criterion:
- the runtime contract becomes stable
- batch metadata and layer plumbing are no longer the blocker

### 7.3 First kernel prototype

Implement one decode-only delta kernel with `mx.fast.metal_kernel`.

Benchmark again on:
- `mixed-8-distinct`
- `long-decode-mixed`
- `fairness`
- a homogeneous fast-path case for sanity

Success criterion:
- clear improvement on mixed decode without destabilizing the runtime

### 7.4 Only then: escalate

If that works:
- move to MLX `Primitive`
- consider grouped prefill / segmented path next
- consider hetero-rank and paging later

## 8. Recommendation

Current recommendation for MOLA:

- do not spend more time on lightweight scheduler heuristics
- do not blame the Python `model_lock` without stronger evidence; current data says it is negligible
- do not jump straight to a fused kernel

Instead:
1. prepare `slot_id + packed buffer` runtime state
2. build a no-kernel mixed-adapter prototype where possible
3. build one decode-only LoRA-delta kernel with `mx.fast.metal_kernel`
4. re-benchmark the same mixed-decode scenarios

In one sentence:

**The best next move is not “write a huge kernel,” but “stabilize a slot-based mixed-adapter ABI, then prototype a decode-only LoRA-delta kernel on top of it.”**

## 8. OpenEvolve: What Transfers To MOLA

Primary sources:
- Hugging Face blog: https://huggingface.co/blog/codelion/openevolve-gpu-kernel-discovery
- OpenEvolve repo: https://github.com/algorithmicsuperintelligence/openevolve

### What they did

- Target: Qwen3-0.6B GQA attention on Apple Silicon.
- Baseline: MLX `scaled_dot_product_attention`.
- Method: OpenEvolve evolved only the Metal kernel source, not the MLX integration layer.
- Search setup: 25 generations, population size 25, 5 islands, `gemini-2.5-flash` for exploration and `gemini-2.5-pro` for deeper optimization.
- Safety: "bulletproof" evaluation with command-buffer protection, memory-violation handling, retries, and fallback.

### Reported numbers

- Average decode improvement: `12.5%`.
- Peak decode improvement: `106%` on a repetitive-pattern benchmark.
- Bench breakdown in the blog:
  - 7/20 benchmarks with significant gains (`>25%`)
  - 3/20 with moderate gains (`5-25%`)
  - 4/20 neutral
  - 6/20 regressions
- The blog also reports `2.8x` speedup on an Apple M1 Pro for the illustrated kernel evolution path.

### What is transferable to MOLA

- The evaluation loop matters more than the prompt. OpenEvolve works because it has a strict compile/test/perf loop with rollback on GPU errors.
- This matches MOLA's likely kernel phase: generate, validate, benchmark, keep the good variant, discard the rest.
- The strongest transferable pattern is not "let the LLM invent the whole server"; it is "let the LLM explore kernel variants while the harness enforces correctness and throughput."
- For MOLA, that means the same harness style around a single decode-only LoRA delta op, with a tight benchmark set such as `mixed-8-distinct`, `long-decode-mixed`, and `fairness`.

### Same problem or adjacent

- Adjacent, not the same.
- OpenEvolve is about automated discovery of GPU kernels for a specific attention workload.
- MOLA is a serving runtime with adapter scheduling, token admission, and mixed-adapter execution. The kernel question is only one layer of the problem.
- The overlap is real only at the "discover or optimize a decode kernel under strict correctness/perf evaluation" layer.

## 10. Public Apple Silicon / Metal Projects Worth Copying

Verdict first:

- `same problem`: none verified publicly.
- `adjacent`: most of the useful public work is adjacent rather than identical.
- `not pertinent`: general AI tooling that does not touch Apple Silicon kernel/runtime constraints.

### 10.1 Adjacent projects

| Project | Classification | Why it matters for MOLA |
| --- | --- | --- |
| [MLX](https://github.com/ml-explore/mlx) | adjacent | Official Apple-silicon runtime baseline. Useful for `gather_mm`, `Primitive`, `metal_kernel`, and the MLX extension path. |
| [mlx-lm](https://github.com/ml-explore/mlx-lm) | adjacent | Best MLX-LM reference for `BatchGenerator`, `SwitchLinear` / `QuantizedSwitchLinear`, and `LoRASwitchLinear`. This is the closest no-kernel routed path. |
| [vllm-project/vllm-metal](https://github.com/vllm-project/vllm-metal) | adjacent | Apple Silicon plugin architecture for a general serving engine. Good for engine/scheduler integration ideas, not for MOLA-style multi-LoRA routing. |
| [Epistates/pmetal](https://github.com/Epistates/pmetal) | adjacent | Manual Metal tuning with fused LoRA and other fused kernels. Good inspiration for kernel layout and operator fusion, but not a multi-adapter serving clone. |
| [RunAnywhere MetalRT](https://www.runanywhere.ai/blog/metalrt-fastest-llm-decode-engine-apple-silicon) | adjacent | Direct-Metal decode engine with strong benchmark discipline. Good for “less abstraction, tighter benchmark loop” lessons. |
| [RunAnywhere docs](https://docs.runanywhere.ai/kotlin/introduction) | adjacent | LoRA hot-swap, stack multiple adapters, adapter catalog. Useful control-plane ideas, but this is still a device SDK, not MOLA’s server runtime. |
| [OpenEvolve](https://huggingface.co/blog/codelion/openevolve-gpu-kernel-discovery) | adjacent | The strongest public example of an AI loop that mutates Metal kernels under a strict benchmark harness. Useful for MOLA kernel iteration. |
| [metalQwen3](https://github.com/BoltzmannEntropy/metalQwen3) | adjacent | Single-model Metal inference engine with explicit AI-assisted development notes. Good signal that AI can accelerate Metal shader work. |
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | adjacent | Metal backend + LoRA support make it a useful runtime reference, but it is not MOLA’s multi-LoRA Apple Silicon server problem. |

### 10.2 What to copy, what to ignore

- Copy from MLX: `SwitchLinear`, `LoRASwitchLinear`, `gather_mm`, `gather_qmm`, `Primitive`, `metal_kernel`.
- Copy from `pmetal`: explicit fused kernel boundaries, fast path naming, and the willingness to keep kernel code manual and benchmark-driven.
- Copy from `MetalRT`: brutal benchmark focus and minimal abstraction in the hot path.
- Copy from `RunAnywhere`: adapter catalog, hot-swap, and stackable adapters as first-class control-plane concepts.
- Copy from `OpenEvolve`: the compile-test-benchmark loop and checkpointed search over kernel variants.
- Copy from `metalQwen3`: the "mostly manual, benchmark-driven" Metal workflow.
- Ignore the product shapes that do not match MOLA: mobile SDK packaging, training UI, general inference platforms, or single-model engines without multi-LoRA routing.

## 9. Public Project Map: What Is Closest To MOLA

Short verdict:
- no public project I verified is solving MOLA end-to-end
- the closest matches are adjacent at the kernel/runtime boundary
- the most useful projects for MOLA are the ones that prove either:
  - a strict kernel-evolution loop works, or
  - a hand-tuned Apple Silicon inference path can still beat baseline MLX

### Same problem

- None verified in this pass.
- I did not find a public Apple Silicon/Metal project that combines multi-LoRA serving, adapter scheduling, routed decode, and an MOLA-like control plane end-to-end.

### Adjacent, high value

- **OpenEvolve** - [GitHub](https://github.com/algorithmicsuperintelligence/openevolve), [HF blog](https://huggingface.co/blog/codelion/openevolve-gpu-kernel-discovery). This is the best example of an AI/kernel-evolution loop. It is not MOLA, but it is directly useful as the pattern for kernel discovery once MOLA has a tight harness.
- **MetalRT** - [RunAnywhere blog](https://www.runanywhere.ai/blog/metalrt-fastest-llm-decode-engine-apple-silicon), [Hugging Face org](https://huggingface.co/runanywhere). This is a hand-tuned Apple Silicon inference engine. Not MOLA's problem, but useful as evidence that a fully manual Metal path can still beat MLX baselines on decode.
- **MLX custom kernels/extensions** - [custom kernels docs](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html), [extensions docs](https://ml-explore.github.io/mlx/build/html/dev/extensions.html). This is the closest implementation substrate for a MOLA kernel prototype. It is adjacent because it gives us the path, not the serving problem.
- **vllm-metal** - [GitHub](https://github.com/vllm-project/vllm-metal). This is a Metal/MLX Apple Silicon backend for vLLM with paged attention and GQA. It is adjacent because it solves Apple Silicon serving and KV behavior, but not MOLA's multi-LoRA adapter scheduling problem.
- **llama.cpp** - [GitHub](https://github.com/ggml-org/llama.cpp). This is an adjacent serving runtime with Metal support, OpenAI-compatible server, and LoRA support. It is useful for server shape, batching, and model-loading ideas, but it is still not MOLA's multi-adapter runtime.
- **metalQwen3** - [GitHub](https://github.com/BoltzmannEntropy/metalQwen3). This is a manually engineered Metal transformer for a single model, with explicit AI-agent assistance noted in the repo. It is not the same problem, but it is a useful example of the "mostly manual, benchmark-driven" Metal workflow.

### Lower priority / not the main signal

- **pmetal** - I did not confirm a reliable primary source in this pass, so I am not using it as evidence here.
- Other Apple Silicon projects that only add MPS/Core ML wrappers or generic inference tooling are useful background, but they are not close enough to MOLA's multi-LoRA problem to drive kernel decisions.

### What this means for MOLA

- The useful public evidence points to two viable routes:
  - AI-assisted kernel evolution with strict correctness/perf harnesses
  - manual expert Metal tuning when the runtime already has a narrow ABI
- For MOLA, that reinforces the current direction:
  - keep the slot-based ABI
  - keep the packer
  - write one decode-oriented kernel or proto at a time
  - keep the benchmark loop tight enough that AI can iterate safely

## 10. RunAnywhere / MetalRT: What Transfers To MOLA

Primary sources:
- RunAnywhere blog: https://www.runanywhere.ai/blog/metalrt-fastest-llm-decode-engine-apple-silicon
- RunAnywhere GitHub org / SDKs: https://github.com/RunanywhereAI and https://github.com/RunanywhereAI/runanywhere-sdks

### What they do

- RunAnywhere is a broader on-device AI SDK/company, not a multi-LoRA serving runtime.
- MetalRT is their Apple Silicon decode engine. The public blog frames it as "straight to the metal" inference for Apple Silicon, with a native binary stack rather than an abstraction-heavy server.
- The public SDK repo positions the product as a privacy-first on-device AI stack with iOS/Android SDKs, on-device text generation, structured outputs, voice pipeline support, and routing between on-device and cloud.

### Manual vs IA

- The official materials do not describe an AI-generated kernel workflow.
- The visible evidence points to expert/manual systems engineering: direct Metal/C++ engine work, Apple Silicon-specific tuning, and a productized SDK architecture.
- `[Inference]` If AI is used internally, it is not part of the public technical pitch; the public story is still primarily hand-built engine + SDK engineering.

### Reported claims

- Peak decode throughput: `658 tok/s` on Qwen3-0.6B.
- Reported decode speedup: `1.10-1.19x` vs `mlx-lm` on the same model files.
- Reported decode speedup: `1.35-2.14x` vs `llama.cpp`.
- Reported decode speedup: `1.41-2.40x` vs `Ollama`.
- Reported TTFT: `6.6 ms` on Qwen3-0.6B.
- Hardware and protocol details in the blog matter: M4 Max, 64 GB unified memory, 4-bit quantized models, 5 runs, best reported.

### Architecture

- MetalRT is a general inference engine, not a LoRA-specialized runtime.
- The public RunAnywhere SDK is a broader product layer around local inference, model management, analytics, routing, and app integration.
- The architecture is therefore closer to "general Apple Silicon serving stack" than to "mixed-adapter LoRA scheduler".

### What transfers to MOLA

- The product framing: Apple Silicon users care about a native stack, low TTFT, and simple integration, not just kernel speed.
- The benchmark discipline: compare engines on identical model files, fixed hardware, repeat runs, and clear decode/TTFT metrics.
- The implementation style: a small, aggressive hot path with the rest kept in a product layer is a valid pattern for MOLA too.
- The "no wrappers" mindset is useful, but only after the MOLA runtime contract is stable.

### Same problem or adjacent

- Adjacent, not the same.
- MetalRT is a general on-device inference engine for Apple Silicon.
- MOLA is a multi-LoRA serving runtime with adapter scheduling, token admission, and mixed-adapter execution.
- The useful overlap is: Apple Silicon Metal performance engineering, not adapter routing logic.

### Practical takeaway for MOLA

- Treat MetalRT as a proof that aggressive Apple Silicon-native serving can beat more generic stacks on decode.
- Do not copy its product boundary directly; MOLA still needs adapter-slot/runtime semantics that MetalRT does not appear to target.
- If MOLA reaches a kernel-backed mixed-adapter path, the comparison should be against "best Apple Silicon serving stacks" and not against itself in the same architecture class.

## 11. Official MLX/Metal Path For MOLA

Official sources:
- MLX custom Metal kernels: https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
- MLX custom extensions / Primitive: https://ml-explore.github.io/mlx/build/html/dev/extensions.html
- Apple Metal resources / MSL spec: https://developer.apple.com/metal/resources/

Practical order for MOLA:
1. Prototype the routed LoRA delta with `mx.fast.metal_kernel` in Python first.
2. Keep the kernel body small and pass `slot_ids` as ordinary input arrays.
3. Use the MLX `Primitive` / `eval_gpu()` path only if the Python-level kernel shows clear benefit.
4. Keep the routed decode ABI stable before moving any more logic into Metal.

Why this order:
- `fast.metal_kernel()` is the quickest way to validate shapes, strides, and performance on Apple Silicon.
- MLX custom extensions are the right path when the prototype works but still needs tighter integration or a reusable primitive.
- Apple’s MSL docs are the source of truth for the low-level kernel language, threadgroup behavior, and device constraints.
