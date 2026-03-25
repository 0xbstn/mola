# Live Mixed Decode Prototype

Date: 2026-03-25

This note records the first live runtime prototype that actually exercises a routed mixed decode session, instead of only validating the routed session seam in isolation.

## What changed

The prototype adds:
- `GeneratorPort.take_states(...)`
- `GeneratorPort.restore_states(...)`
- a shared mixed decode generator keyed internally as `__mixed_decode__`
- an opt-in engine path that migrates decode-ready adapted requests into that shared generator
- a routed decode session built from the shared generator row order
- `gather-mm` as the routed backend behind that shared decode step

The important difference versus earlier routed validation is:
- old routed validation still stepped one `BatchGenerator` per adapter
- this prototype can execute a real mixed routed decode step in the live engine

## Product decision

This path stays experimental.
It is not the default runtime.

The useful result is not "routed reference is correct".
The useful result is:
- live mixed decode migration is feasible on top of `BatchGenerator`
- the `gather-mm` backend starts to pay off once the runtime actually feeds it a mixed decode batch

## Key benchmark files

Ignored local benchmark outputs used for this note:
- `benchmark/kernel-lab/server-mixed-migration-reference-8-16-r2.json`
- `benchmark/kernel-lab/server-mixed-migration-gather-8-16-r2.json`
- `benchmark/kernel-lab/server-mixed-migration-gather-gated-8-16-r2.json`
- `benchmark/kernel-lab/server-mixed-migration-gather-32.json`
- `benchmark/kernel-lab/server-mixed-migration-gather-gated-32-r2.json`

## Main result

The ungated `gather-mm + migration` prototype already improves the real mixed workloads over the old non-migration baseline.

Then a smaller runtime change improved the product shape:
- do not migrate homogeneous decode traffic into the shared mixed generator
- migrate only when there is a real mixed decode opportunity or an already-active shared mixed batch

This preserves more of the `same` path while keeping the mixed gains.

## Routed migration: `reference` vs `gather-mm`

At `conc=16`, `repeats=2`:

- `reference + migration`
  - `mixed`: `6.4 req/s`, `294.6 tok/s`, `p95 1517.1 ms`
  - `long-decode-mixed`: `3.9 req/s`, `343.8 tok/s`, `p95 2259.0 ms`
  - `fairness`: `8.3 req/s`, `441.3 tok/s`, `p95 1289.1 ms`

- `gather-mm + migration`
  - `mixed`: `10.5 req/s`, `541.0 tok/s`, `p95 1517.1 ms`
  - `long-decode-mixed`: `7.0 req/s`, `653.0 tok/s`, `p95 2259.0 ms`
  - `fairness`: `12.4 req/s`, `672.5 tok/s`, `p95 1289.1 ms`

[Inference]
The routed backend starts to matter only after the live runtime stops forcing one homogeneous decode step per adapter.

## Gated migration results

The gated version keeps homogeneous decode on the normal path until a real mixed opportunity exists.

Compared against the ungated `gather-mm + migration` run:

### `conc=8`

- `same`
  - `10.8 -> 11.7 req/s`
  - `689.6 -> 718.6 tok/s`
  - `748.0 -> 689.9 ms p95`

- `mixed`
  - `7.7 -> 7.3 req/s`
  - `341.9 -> 382.4 tok/s`
  - `1067.4 -> 1135.4 ms p95`

- `long-decode-mixed`
  - `4.9 -> 4.8 req/s`
  - `423.2 -> 402.8 tok/s`
  - `1704.7 -> 1776.1 ms p95`

- `fairness`
  - `7.9 -> 8.5 req/s`
  - `426.5 -> 406.3 tok/s`
  - `1031.3 -> 946.5 ms p95`

### `conc=16`

- `same`
  - `14.4 -> 15.7 req/s`
  - `865.7 -> 937.5 tok/s`
  - `1113.2 -> 1020.5 ms p95`

- `mixed`
  - `10.5 -> 11.2 req/s`
  - `541.0 -> 527.6 tok/s`
  - `1517.1 -> 1445.6 ms p95`

- `long-decode-mixed`
  - `7.0 -> 7.2 req/s`
  - `653.0 -> 674.4 tok/s`
  - `2259.0 -> 2240.1 ms p95`

- `fairness`
  - `12.4 -> 12.4 req/s`
  - `672.5 -> 642.0 tok/s`
  - `1289.1 -> 1287.0 ms p95`

### `conc=32`

- `same`
  - `18.5 -> 18.9 req/s`
  - `1029.5 -> 1154.5 tok/s`
  - `1730.7 -> 1691.3 ms p95`

- `mixed`
  - `15.8 -> 15.5 req/s`
  - `756.7 -> 779.1 tok/s`
  - `2019.1 -> 2056.5 ms p95`

- `long-decode-mixed`
  - `10.8 -> 10.4 req/s`
  - `955.1 -> 982.8 tok/s`
  - `2966.2 -> 3086.0 ms p95`

- `fairness`
  - `17.4 -> 17.4 req/s`
  - `868.5 -> 868.9 tok/s`
  - `1836.8 -> 1836.4 ms p95`

[Inference]
The gated runtime is the better shape:
- clearly better on `same`
- better or neutral on `fairness`
- roughly neutral to slightly better on the discriminating mixed cases at `conc=16`
- no sign that the gating change breaks the scaling shape at `conc=32`

## What the new live metrics say

The gated prototype now exports:
- migration event count
- migrated sequence count
- mixed decode step count
- mixed decode row count
- average rows per mixed decode step

Representative `repeats=1` live run:

- `same @8`
  - `migration_events=0`
  - `migrated_sequences=0`
  - `mixed_steps=0`
  - `avg_rows=0.0`

- `mixed @8`
  - `migration_events=8`
  - `migrated_sequences=8`
  - `mixed_steps=74`
  - `mixed_rows=381`
  - `avg_rows=5.15`

- `long-decode-mixed @8`
  - `migration_events=8`
  - `migrated_sequences=8`
  - `mixed_steps=132`
  - `mixed_rows=699`
  - `avg_rows=5.30`

- `mixed @16`
  - `migration_events=9`
  - `migrated_sequences=16`
  - `mixed_steps=76`
  - `mixed_rows=776`
  - `avg_rows=10.21`

- `long-decode-mixed @16`
  - `migration_events=9`
  - `migrated_sequences=16`
  - `mixed_steps=140`
  - `mixed_rows=1357`
  - `avg_rows=9.69`

- `mixed @32`
  - `migration_events=10`
  - `migrated_sequences=32`
  - `mixed_steps=75`
  - `mixed_rows=1520`
  - `avg_rows=20.27`

- `long-decode-mixed @32`
  - `migration_events=9`
  - `migrated_sequences=32`
  - `mixed_steps=139`
  - `mixed_rows=2804`
  - `avg_rows=20.17`

[Inference]
This is a useful threshold result.
The current runtime is no longer “barely exercising” the mixed shared path:
- homogeneous traffic stays off it entirely
- mixed traffic moves most adapted decode rows into it
- the shared batch is already reasonably wide for decode

[Inference]
That makes the next likely ceiling more about the routed compute backend than about a low migration rate.

## Upstream MLX signal on sorting

Reading upstream MLX / MLX-LM clarified when sorted routed matmul is expected to help:

- MLX’s `gather_mm` fast path for `sorted_indices=True` is only meaningful in the one-sided indexed case
- MLX-LM’s routed expert layers explicitly sort/unsort rows around expert matmuls
- upstream only enables that sorted path when the grouped routed batch is already large enough
- in `mlx_lm.models.switch_layers`, the explicit sorting threshold is `indices.size >= 64`

[Inference]
That makes sorted routed execution a poor fit for the current MOLA prototype:
- the live shared mixed slot already helps without sorting
- current average mixed decode row counts are around `5` to `20`, not `64+`
- MOLA’s present routed LoRA path would have to pay extra sort/unsort work before it could even reach the upstream fast path shape

Decision:
- do not pursue sorted routed decode in the current runtime
- keep `gather-mm` unsorted as the meaningful non-kernel backend
- if sorted routing is revisited later, it should be paired with a runtime that can actually feed much larger grouped batches

One extra runtime check made this even clearer:
- on a tiny real `gather_mm` routed LoRA example, `sorted_indices=True` produced the wrong result in `float16`
- the same shape in `float32` failed because the expected Metal function was missing

[Inference]
So today this is not just a “not worth it yet” path.
For MOLA’s routed LoRA delta shape on this stack, it is currently not a trustworthy backend candidate.

## Batch-size knob result

The current runtime can now expose `max_batch_size` and `prefill_batch_size` directly from the CLI.

The useful experiment was:
- keep the same `gather-mm + migration` runtime
- compare `max_batch_size=32` vs `max_batch_size=64`

### High concurrency (`conc=64`)

`max_batch_size=64` is clearly better than `32`:

- `same`
  - `21.9 -> 27.2 req/s`
  - `1258.3 -> 1648.8 tok/s`
  - `2910.0 -> 2348.8 ms p95`

- `mixed`
  - `22.8 -> 23.6 req/s`
  - `1201.4 -> 1150.4 tok/s`
  - `2783.8 -> 2695.8 ms p95`

- `long-decode-mixed`
  - `15.4 -> 15.2 req/s`
  - `1416.1 -> 1493.3 tok/s`
  - `4142.9 -> 4206.4 ms p95`

- `fairness`
  - `23.3 -> 23.5 req/s`
  - `1211.1 -> 1226.2 tok/s`
  - `2723.0 -> 2698.7 ms p95`

### Lower concurrency (`conc=8/16`)

`max_batch_size=64` does not look like a new default:
- generally neutral to slightly worse on `req/s` and `p95`
- sometimes better on `tok/s`
- not a consistent win at lower concurrency

[Inference]
This makes `max_batch_size` a real deployment knob, not yet a new global default:
- at higher concurrency it materially helps
- at lower concurrency it is not reliably better

Decision:
- keep the knob exposed
- do not change the default from `32` yet

## High-concurrency profile exploration

Two follow-up sweeps mattered more than expected:

### `max_batch_size=64`, `prefill_batch_size=32`

At `conc=64`, compared to the `b64/p8` profile:

- `same`
  - `27.2 -> 29.9 req/s`
  - `1588.8 -> 1761.6 tok/s`
  - `2345.7 -> 2133.7 ms p95`

- `mixed`
  - `23.6 -> 22.7 req/s`
  - `1084.7 -> 1117.1 tok/s`
  - `2694.6 -> 2791.7 ms p95`

- `long-decode-mixed`
  - `15.2 -> 15.2 req/s`
  - `1429.2 -> 1441.8 tok/s`
  - `4204.8 -> 4188.0 ms p95`

- `fairness`
  - `23.8 -> 25.3 req/s`
  - `1248.4 -> 1309.3 tok/s`
  - `2671.1 -> 2485.1 ms p95`

[Inference]
At higher load, a larger prefill wave is not just a prompt-side knob.
It helps the runtime fill the larger decode batches faster and can materially improve the `same` and `fairness` shapes.

### `max_batch_size=128`, `prefill_batch_size=32`

At `conc=128`, compared to `b64/p32`:

- `same`
  - `30.2 -> 32.0 req/s`
  - `1826.9 -> 1909.4 tok/s`
  - `4228.9 -> 3986.4 ms p95`

- `mixed`
  - `22.8 -> 23.3 req/s`
  - `1125.6 -> 1158.1 tok/s`
  - `5376.2 -> 5272.7 ms p95`

- `long-decode-mixed`
  - `15.4 -> 16.8 req/s`
  - `1444.5 -> 1536.7 tok/s`
  - `8067.7 -> 7483.7 ms p95`

- `fairness`
  - `24.1 -> 23.5 req/s`
  - `1228.7 -> 1216.7 tok/s`
  - `5051.6 -> 5270.4 ms p95`

[Inference]
There is now a credible high-concurrency profile that is better than the default experimental shape:
- `max_batch_size=64` or `128`
- `prefill_batch_size=32`

But it is still a profile, not a new universal default:
- it helps most under high load
- it can trade away some `fairness` behavior depending on the setting
- it should stay opt-in until more sweeps are collected

## What this means

The current best experimental story is:
- keep the default production path unchanged
- keep routed decode behind flags
- treat `reference` as the correctness backend
- treat `gather-mm + migration` as the first live mixed decode prototype worth continuing

The important architectural result is:
- the next throughput gains no longer need to come from scheduler heuristics
- they can now come from improving the actual mixed decode compute path and migration policy

## Risks still open

- a mixed shared decode slot can still widen one bad routed row into a shared failure
- migration remains an invasive runtime path and should stay opt-in
- this is still built on top of `BatchGenerator`, not a purpose-built mixed decode runtime

## Next useful experiments

1. compare grouped migration thresholds
- e.g. require `>= 2` adapted slots ready before the first migration wave

2. microbench the live mixed shapes that the shared generator actually sees
- row counts after migration
- layer distribution
- homogeneous vs mixed share per step

3. prototype narrower failure handling for the shared mixed slot
- isolate a bad migrated row without killing the whole shared decode batch if practical

## Rejected experiment

One backend tweak looked attractive on paper and did not pay off live:
- precompute `rhs_indices`
- pre-scale `lora_b` by the adapter scale
- remove the final gathered-scale multiply in `gather-mm`

[Inference]
This did not improve the real server path in a reliable way. It was neutral to worse on the mixed cases and clearly worse on `same` / `fairness` in repeated live runs.

Decision:
- do not keep that variant
- keep the simpler `gather-mm` backend until a stronger compute-path change proves itself
