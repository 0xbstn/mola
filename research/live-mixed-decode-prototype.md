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

## Rejected backend candidate: indexed batched matmul

Another candidate was tested in the lab:
- gather `A` and `B` by routed row index
- then run a regular batched matmul

The isolated microbench looked plausible on some shapes, so it was wired as a real backend candidate.

Live result:
- it booted
- but it did not survive a real routed validation run cleanly enough to treat as a stable candidate

Decision:
- drop the backend from the repo
- keep only the conclusion

[Inference]
For now, `gather-mm` remains the only routed non-kernel backend worth carrying in the live code path.

## Experimental backend: `metal-gather`

After the shared mixed decode path was stable enough to benchmark at higher concurrency, a second routed compute backend was added:
- keep `gather-mm` for the routed `A` contraction
- use a custom Metal kernel for the routed `B` expansion on selected layer families

The first safe version only covered:
- `q/k/v/o`
- `down`

That backend was already slightly better than plain `gather-mm` at high concurrency, especially on `long-decode-mixed`.

Then the Metal path was generalized to large-output families too:
- `up`
- `gate`

The generalized kernel changed from:
- one reduction thread-row per rank lane
- only `lane_y == 0` writing outputs

to:
- cooperative reduction into `zbuf`
- all threads participating in the output write loop
- fixed launch shape `64 x max(16, rank)`

[Inference]
That generalization matters because the remaining routed compute gap is mostly in the large-output MLP projections, not in the smaller attention projections.

## `metal-gather`: live result after large-output generalization

Bench file:
- `/tmp/server-metal-gather-wide-b128-p32-r1.json`

Compared against the earlier `metal-gather` and `gather-mm` runs:

### `conc=128`

- `same`
  - new generalized `metal-gather`: `32.43 req/s`, `1930.8 tok/s`, `3933 ms p95`
  - earlier `metal-gather`: `31.79 req/s`, `1892.8 tok/s`, `4072 ms p95`
  - `gather-mm`: `32.24 req/s`, `1915.6 tok/s`, `3911 ms p95`

- `mixed`
  - new generalized `metal-gather`: `23.16 req/s`, `1151.2 tok/s`, `5321 ms p95`
  - earlier `metal-gather`: `23.12 req/s`, `1163.4 tok/s`, `5378 ms p95`
  - `gather-mm`: `23.19 req/s`, `1147.3 tok/s`, `5323 ms p95`

- `long-decode-mixed`
  - new generalized `metal-gather`: `16.68 req/s`, `1563.3 tok/s`, `7397 ms p95`
  - earlier `metal-gather`: `16.03 req/s`, `1516.6 tok/s`, `7759 ms p95`
  - `gather-mm`: `16.00 req/s`, `1487.4 tok/s`, `7721 ms p95`

- `fairness`
  - new generalized `metal-gather`: `24.48 req/s`, `1296.6 tok/s`, `5005 ms p95`
  - earlier `metal-gather`: `24.57 req/s`, `1271.4 tok/s`, `5035 ms p95`
  - `gather-mm`: `24.17 req/s`, `1241.9 tok/s`, `5109 ms p95`

### `conc=256`

- `same`
  - new generalized `metal-gather`: `33.27 req/s`, `1989.1 tok/s`, `7576 ms p95`
  - earlier `metal-gather`: `33.48 req/s`, `2013.1 tok/s`, `7619 ms p95`
  - `gather-mm`: `32.76 req/s`, `1959.2 tok/s`, `7787 ms p95`

- `mixed`
  - new generalized `metal-gather`: `23.31 req/s`, `1151.7 tok/s`, `10655 ms p95`
  - earlier `metal-gather`: `23.39 req/s`, `1153.4 tok/s`, `10644 ms p95`
  - `gather-mm`: `23.22 req/s`, `1127.9 tok/s`, `10717 ms p95`

- `long-decode-mixed`
  - new generalized `metal-gather`: `15.60 req/s`, `1490.0 tok/s`, `15999 ms p95`
  - earlier `metal-gather`: `15.46 req/s`, `1448.6 tok/s`, `16168 ms p95`
  - `gather-mm`: `15.17 req/s`, `1441.9 tok/s`, `16384 ms p95`

- `fairness`
  - new generalized `metal-gather`: `23.96 req/s`, `1238.8 tok/s`, `10395 ms p95`
  - earlier `metal-gather`: `23.67 req/s`, `1202.0 tok/s`, `10520 ms p95`
  - `gather-mm`: `22.92 req/s`, `1193.7 tok/s`, `10833 ms p95`

Decision:
- keep the generalized `metal-gather` backend
- do not make it the default global runtime backend
- keep using it as the best experimental high-concurrency routed backend

[Inference]
This is the first routed compute backend in the repo that:
- survives live shared mixed decode
- stays correct under migration and restore checks
- and gives a consistent end-to-end improvement over `gather-mm` on the discriminating mixed decode cases

[Inference]
The remaining opportunity is no longer “should there be a routed compute backend at all”.
It is now mostly:
- per-layer-family tuning
- especially the large-output MLP path

## `metal-gather`: launch-shape tuning by layer family

The next local iteration kept the same kernel structure but stopped using one launch shape for every layer family.

Current experimental launch policy:
- `q/o`: `128 x rank`
- `k/v`: `64 x max(16, rank)`
- `down`: `64 x rank`
- `up/gate`: `128 x rank`

Bench file:
- `/tmp/server-metal-gather-wide2-b128-p32-r1.json`

Compared against the previous generalized `metal-gather` run:

### `conc=128`

- `same`
  - `32.43 -> 32.14 req/s`
  - `1930.8 -> 1924.1 tok/s`
  - `3933 -> 3967 ms p95`

- `mixed`
  - `23.16 -> 23.67 req/s`
  - `1151.2 -> 1169.5 tok/s`
  - `5321 -> 5204 ms p95`

- `long-decode-mixed`
  - `16.68 -> 17.54 req/s`
  - `1563.3 -> 1598.7 tok/s`
  - `7397 -> 7075 ms p95`

- `fairness`
  - `24.48 -> 24.96 req/s`
  - `1296.6 -> 1275.5 tok/s`
  - `5005 -> 4957 ms p95`

### `conc=256`

- `same`
  - `33.27 -> 33.83 req/s`
  - `1989.1 -> 2033.7 tok/s`
  - `7576 -> 7441 ms p95`

- `mixed`
  - `23.31 -> 23.69 req/s`
  - `1151.7 -> 1144.1 tok/s`
  - `10655 -> 10491 ms p95`

- `long-decode-mixed`
  - `15.60 -> 15.64 req/s`
  - `1490.0 -> 1463.6 tok/s`
  - `15999 -> 15999 ms p95`

- `fairness`
  - `23.96 -> 24.56 req/s`
  - `1238.8 -> 1241.6 tok/s`
  - `10395 -> 10147 ms p95`

Decision:
- keep the layer-family launch-shape tuning
- treat it as the best current `metal-gather` runtime profile
- do not change the default global backend yet

[Inference]
This is still the same backend family, not a new algorithm.
But it is now clearly better than the earlier generalized launch policy on the important live mixed workloads.

## Rejected backend candidate: `metal-bexpand`

Another compute-only backend was tried after `metal-gather`:
- keep `gather-mm` for the routed `A` contraction
- transpose `B` once at session build time
- run a custom Metal kernel only for the `B` expansion

This backend was attractive because it reduced the amount of custom logic:
- one `gather-mm` for `A`
- one Metal kernel for `z @ B`
- no need to custom-code the `A` reduction inside Metal

Live result at `conc=128`:
- `same`: `31.7 req/s`, `1891.7 tok/s`, `4023 ms p95`
- `mixed`: `19.6 req/s`, `1280.8 tok/s`, `6173 ms p95`
- `long-decode-mixed`: `12.1 req/s`, `1585.9 tok/s`, `9955 ms p95`
- `fairness`: `20.8 req/s`, `1363.5 tok/s`, `6021 ms p95`

Compared with the tuned `metal-gather` baseline at the same profile:
- much worse on `mixed`
- much worse on `long-decode-mixed`
- much worse on `fairness`
- only roughly comparable on `same`

Decision:
- drop `metal-bexpand` from the repo
- keep only the conclusion

[Inference]
Splitting the routed delta into:
- `gather-mm` for `A`
- custom Metal only for `B`

still leaves too much of the routed overhead in place.
For this runtime, the better direction remains:
- bucket rows by slot
- keep the fused routed delta in one backend

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

## Rejected backend candidate: hybrid-ba

Another routed backend was tested after the first live mixed decode prototype:
- keep the current unsorted `gather_mm` first pass for `A`
- replace the second routed `gather_mm` with a dense batched matmul against row-gathered `B`

This looked plausible in the isolated op lab:
- it stayed numerically correct
- on some attention-shaped decode cases it matched or slightly beat `gather-mm`

But the live runtime comparison on the same `mixed decode migration` profile did not justify keeping it as a supported backend.

Profile compared:
- `--routed-decode-backend gather-mm`
- `--enable-mixed-decode-migration`
- `--max-batch-size 128`
- `--prefill-batch-size 32`

Live result against `gather-mm` on the same server benchmark battery:
- `same @64`
  - small improvement (`31.5 -> 31.8 req/s`)
- `mixed @64`
  - worse (`23.7 -> 22.8 req/s`)
- `long-decode-mixed @64`
  - worse (`15.6 -> 15.2 req/s`)
- `fairness @64`
  - slightly worse (`25.5 -> 25.4 req/s`)
- `same @128`
  - worse (`31.7 -> 31.1 req/s`)
- `mixed @128`
  - worse (`22.9 -> 22.3 req/s`)
- `long-decode-mixed @128`
  - worse (`16.7 -> 16.5 req/s`)
- `fairness @128`
  - worse (`24.0 -> 23.3 req/s`)

[Inference]
The isolated op win was too narrow. Once it paid the real runtime costs around migration, grouping, and the rest of the decode path, it did not beat the existing `gather-mm` backend.

Decision:
- do not keep `hybrid-ba` as a routed decode backend
- keep it only as a lab result
- continue from `gather-mm` for the live runtime until a stronger compute-path change proves itself

## New experimental backend: metal-gather

The next useful step was not another MLX-level backend variant.
It was a narrower compute-path change:
- keep the same live routed runtime
- keep `gather-mm` as the baseline routed backend
- add a new backend that uses a custom Metal kernel only on the layer families where the isolated kernel lab already beats `gather-mm`
- fall back to `gather-mm` for the rest

The first useful kernel shape was not the naive one-thread-per-output kernel.
That version stayed too slow because it still underused the threadgroup and effectively paid too much work per row.

The better candidate was a 2D reduction kernel:
- `threadgroup.x` splits the `D` dimension reduction
- `threadgroup.y` maps the LoRA rank lanes
- the threadgroup cooperatively computes `z = x[row] @ A[slot]`
- only `lane_y == 0` writes the output columns

[Inference]
This is the first routed Metal design that actually attacks the right bottleneck:
- parallelize the `D` reduction
- then reuse the reduced `z` across the whole output row

### Isolated kernel-lab result

With the 2D reduction kernel at `T=32`, `float16`, routed mixed rows:

- `attn_q_proj`
  - `gather-mm`: `0.1907 ms`
  - metal reduce: `0.1660-0.1751 ms` depending on pattern
- `attn_kv_proj`
  - `gather-mm`: `0.1841-0.1907 ms`
  - metal reduce: `0.1606-0.1724 ms`
- `mlp_down_proj`
  - `gather-mm`: `0.2104-0.2205 ms`
  - metal reduce: `0.1770-0.1865 ms`
- `mlp_up_gate_proj`
  - `gather-mm`: still better
  - metal reduce: `0.2222-0.2571 ms` vs `0.1770-0.2195 ms`

Decision from the lab:
- use Metal only for:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
  - `down_proj`
- keep `gather-mm` for:
  - `up_proj`
  - `gate_proj`

Implementation detail:
- `q_proj` / `o_proj`: `threads_x = 96`
- `k_proj` / `v_proj` / `down_proj`: `threads_x = 128`
- `up_proj` / `gate_proj`: no Metal path

Safety behavior:
- if Metal kernel creation fails, the backend now degrades to `gather-mm` instead of failing the routed session factory
- if a mixed shared decode batch cannot build or run its routed session, the migrated requests are restored back to their per-adapter generators instead of being failed in place

[Inference]
This matters more than it first seemed.
Without those guards, a backend experiment can look acceptable in steady-state benchmarks while still being too fragile to keep around as a real runtime option.

### Live result at `128`, `repeats=2`

Compared against `gather-mm` on the same runtime profile:
- `--enable-mixed-decode-migration`
- `--max-batch-size 128`
- `--prefill-batch-size 32`

`metal-gather` tuned vs `gather-mm`:

- `same`
  - `32.2 -> 31.7 req/s`
  - `1915.6 -> 1911.8 tok/s`
  - `3910.5 -> 3992.4 ms p95`
- `mixed`
  - `23.2 -> 23.6 req/s`
  - `1147.3 -> 1129.6 tok/s`
  - `5323.0 -> 5228.8 ms p95`
- `long-decode-mixed`
  - `16.0 -> 16.8 req/s`
  - `1487.4 -> 1558.0 tok/s`
  - `7720.7 -> 7369.9 ms p95`
- `fairness`
  - `24.2 -> 24.3 req/s`
  - `1241.9 -> 1246.6 tok/s`
  - `5108.6 -> 5062.5 ms p95`

[Inference]
At `conc=128`, this backend is not a universal win.
But it is the first routed compute backend that improves the key mixed decode shape without collapsing the rest of the runtime.

### Live result at `256`, `repeats=1`

Compared against `gather-mm` on the same profile:

- `same`
  - `32.8 -> 33.5 req/s`
  - `1959.2 -> 2013.1 tok/s`
  - `7787.0 -> 7618.9 ms p95`
- `mixed`
  - `23.2 -> 23.4 req/s`
  - `1127.9 -> 1153.4 tok/s`
  - `10717.3 -> 10643.9 ms p95`
- `long-decode-mixed`
  - `15.2 -> 15.5 req/s`
  - `1441.9 -> 1448.6 tok/s`
  - `16383.8 -> 16167.8 ms p95`
- `fairness`
  - `22.9 -> 23.7 req/s`
  - `1193.7 -> 1202.0 tok/s`
  - `10832.5 -> 10520.3 ms p95`

[Inference]
The advantage becomes clearer as the shared mixed decode batches get larger.
This backend is now best understood as a higher-load experimental path, not as a replacement for the default routed backend at every traffic level.

Decision:
- keep `metal-gather` as an experimental routed backend
- do not make it the default
- keep `gather-mm` as the simpler experimental baseline
- continue future compute work from this backend, not from the older naive `metal-kernel`

## Large-output Metal generalization

The first `metal-gather` revision only used the Metal path for:
- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `down_proj`

It kept `gather-mm` for:
- `up_proj`
- `gate_proj`

That was safe, but it also left a lot of routed decode work on the table because:
- `up_proj` / `gate_proj` dominate some decoder layers
- the old kernel shape only had one `lane_y == 0` stripe write outputs, which is a poor fit for large `out_dim`

The next experiment changed the kernel shape:
- keep the same `A` reduction into `zbuf`
- but let all threads participate in the output projection
- use:
  - `THREADS_X = 64`
  - `THREADS_Y = max(16, rank)`
  - `lane = lane_y * THREADS_X + lane_x`
  - `stride = THREADS_X * THREADS_Y`

[Inference]
This is still not a “full custom fused kernel”.
It is a better output-parallel version of the same routed LoRA delta idea.

### Live result with generalized large-output Metal path

The generalized backend now covers:
- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `down_proj`
- `up_proj`
- `gate_proj`

Compared against the previous tuned `metal-gather` backend:

### `conc=128`, `repeats=1`

- `same`
  - `31.8 -> 32.4 req/s`
  - `1892.8 -> 1930.8 tok/s`
  - `4071.7 -> 3933.2 ms p95`
- `mixed`
  - `23.1 -> 23.2 req/s`
  - `1163.4 -> 1151.2 tok/s`
  - `5377.8 -> 5321.2 ms p95`
- `long-decode-mixed`
  - `16.0 -> 16.7 req/s`
  - `1516.6 -> 1563.3 tok/s`
  - `7759.2 -> 7396.9 ms p95`
- `fairness`
  - `24.6 -> 24.5 req/s`
  - `1271.4 -> 1296.6 tok/s`
  - `5035.3 -> 5005.3 ms p95`

Compared against `gather-mm` at the same profile:

- `same`
  - `32.2 -> 32.4 req/s`
  - `1915.6 -> 1930.8 tok/s`
  - `3910.5 -> 3933.2 ms p95`
- `mixed`
  - `23.2 -> 23.2 req/s`
  - `1147.3 -> 1151.2 tok/s`
  - `5323.0 -> 5321.2 ms p95`
- `long-decode-mixed`
  - `16.0 -> 16.7 req/s`
  - `1487.4 -> 1563.3 tok/s`
  - `7720.7 -> 7396.9 ms p95`
- `fairness`
  - `24.2 -> 24.5 req/s`
  - `1241.9 -> 1296.6 tok/s`
  - `5108.6 -> 5005.3 ms p95`

### `conc=256`, `repeats=1`

Compared against the previous tuned `metal-gather` backend:

- `same`
  - `33.5 -> 33.3 req/s`
  - `2013.1 -> 1989.1 tok/s`
  - `7618.9 -> 7575.9 ms p95`
- `mixed`
  - `23.4 -> 23.3 req/s`
  - `1153.4 -> 1151.7 tok/s`
  - `10643.9 -> 10654.7 ms p95`
- `long-decode-mixed`
  - `15.5 -> 15.6 req/s`
  - `1448.6 -> 1490.0 tok/s`
  - `16167.8 -> 15999.2 ms p95`
- `fairness`
  - `23.7 -> 24.0 req/s`
  - `1202.0 -> 1238.8 tok/s`
  - `10520.3 -> 10394.5 ms p95`

Compared against `gather-mm`:

- `same`
  - `32.8 -> 33.3 req/s`
  - `1959.2 -> 1989.1 tok/s`
  - `7787.0 -> 7575.9 ms p95`
- `mixed`
  - `23.2 -> 23.3 req/s`
  - `1127.9 -> 1151.7 tok/s`
  - `10717.3 -> 10654.7 ms p95`
- `long-decode-mixed`
  - `15.2 -> 15.6 req/s`
  - `1441.9 -> 1490.0 tok/s`
  - `16383.8 -> 15999.2 ms p95`
- `fairness`
  - `22.9 -> 24.0 req/s`
  - `1193.7 -> 1238.8 tok/s`
  - `10832.5 -> 10394.5 ms p95`

[Inference]
This is the first version of `metal-gather` that is:
- clearly better on the discriminating `long-decode-mixed` case
- roughly neutral to positive on `mixed`
- still competitive on `same`
- and more consistently ahead at higher concurrency

Decision:
- keep the generalized `metal-gather` backend
- keep it experimental
- use it as the current best high-load routed compute backend
- do not change the global default yet

## Compile-time rank specialization

The next useful experiment stayed inside the same `metal-gather` backend.
No runtime changes, no new backend, no new routing policy.

Only the shader changed:
- remove the runtime `rank` variable from the mixed kernel
- specialize directly on `MAX_R`
- use `MAX_R` in the packed `A` / `B` indexing
- explicitly ask the compiler to unroll the final `r` loop

[Inference]
This is the first compute tweak after the large-output generalization that meaningfully improved the live routed path without changing the runtime shape again.

### Why this was worth trying

For MOLA today:
- LoRA rank is effectively fixed at a small value in the live experiments
- the mixed routed kernel already compiles with `MAX_R` in the template
- but the shader still paid a runtime-style `rank` loop in the hot path

[Inference]
That made compile-time specialization a low-risk kernel experiment:
- same ABI
- same backend
- same mixed decode migration path
- only less dynamic work inside the shader

### Live result at `conc=128`, `repeats=2`

Compared against the previous generalized `metal-gather` run:

- `same`
  - `31.8 -> 30.9 req/s`
  - `1892.8 -> 1856.4 tok/s`
  - `4071.7 -> 4056.4 ms p95`
- `mixed`
  - `23.1 -> 23.9 req/s`
  - `1163.4 -> 1177.2 tok/s`
  - `5377.8 -> 5177.3 ms p95`
- `long-decode-mixed`
  - `16.0 -> 16.3 req/s`
  - `1516.6 -> 1547.4 tok/s`
  - `7759.2 -> 7600.4 ms p95`
- `fairness`
  - `24.6 -> 25.0 req/s`
  - `1271.4 -> 1275.3 tok/s`
  - `5035.3 -> 4932.5 ms p95`

### Live result at `conc=256`, `repeats=1`

Compared against the previous generalized `metal-gather` run:

- `same`
  - `33.3 -> 33.2 req/s`
  - `1989.1 -> 1920.9 tok/s`
  - `7575.9 -> 7639.4 ms p95`
- `mixed`
  - `23.3 -> 23.8 req/s`
  - `1151.7 -> 1170.9 tok/s`
  - `10654.7 -> 10449.5 ms p95`
- `long-decode-mixed`
  - `15.6 -> 16.3 req/s`
  - `1490.0 -> 1534.6 tok/s`
  - `15999.2 -> 15338.7 ms p95`
- `fairness`
  - `24.0 -> 24.3 req/s`
  - `1238.8 -> 1243.4 tok/s`
  - `10394.5 -> 10241.7 ms p95`

[Inference]
This is not a universal win on every metric.
But it is the first post-`metal-gather` shader tweak that looks worth keeping:
- `same` is slightly softer
- `mixed`, `long-decode-mixed`, and `fairness` improve
- the improvement holds most clearly in the discriminating mixed decode cases

Decision:
- keep the compile-time rank specialization inside `metal-gather`
- treat it as the current best routed compute backend variant
- continue future kernel work from here, not from the rejected split `A/B` paths
