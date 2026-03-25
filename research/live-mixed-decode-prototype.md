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
