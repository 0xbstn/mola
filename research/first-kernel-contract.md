# First Kernel Contract

Date: 2026-03-24

This note is intentionally narrow.
It describes the first kernel-sized contract that can replace the current routed reference session without changing the rest of the runtime again.

## Current swap point

The current routed execution seam is:
- `mola.ports.routed_decode.RoutedLoRADeltaSession`
- `mola.ports.routed_decode.RoutedLoRADeltaSessionFactory`
- `mola.infrastructure.routed_decode.ReferenceRoutedLoRADeltaSessionFactory`

The engine now does three things only:
1. decide which `slot_id`s are active for the next decode step
2. build per-layer routed pack views for those slots
3. ask the session factory to build an opaque routed session

The layer path does one thing only:
- `MultiLoRALinear` calls `session.delta(layer_name, x)`

That means the future kernel can replace the reference path behind the factory without changing `engine.py`, `context.py`, or `lora.py` again.

## First kernel target

Do not start with full fusion.
Do not start with prefill.
Do not start with MoE.

Start with decode-only routed LoRA delta for `MultiLoRALinear`.

Target shape:
- `x`: `[T, D]`
- `slot_ids`: `[T]`
- `A`: packed by active slot for one layer
- `B`: packed by active slot for one layer
- `scales`: packed by active slot for one layer
- output `delta`: `[T, O]`

Per token `t`:
- `s = slot_ids[t]`
- `delta[t] = scale[s] * ((x[t] @ A[s]) @ B[s])`

## First backend behavior

The first backend should support two modes:

1. homogeneous fast path
- all rows share one `slot_id`
- acceptable to dispatch to a simpler path
- useful for comparing the kernel against the current homogeneous routed reference path

2. mixed routed path
- rows may map to different `slot_id`s
- the backend owns any sort/group/restore behavior needed for execution

## Inputs the backend already has

From the current factory seam, the backend receives:
- routed layer pack views for the active slots only
- `token_slot_ids` for the next decode step only

This is enough to:
- materialize packed tensors once per step in the reference backend
- or cache/reuse layer-local packed tensors in a future kernel backend

## What the first kernel does not need

The first kernel does not need:
- base matmul fusion
- prefill support
- `MultiLoRASwitchLinear`
- heterogeneous rank support
- adapter load/unload policy
- scheduler changes

## Success criteria

The first kernel should be judged on:
- `same`
- `mixed`
- `long-decode-mixed`
- `fairness`

And specifically on:
- `req/s`
- `tok/s`
- `p95`
- correctness against the current routed reference path

## Current product decision

The current routed reference path is slower than the default runtime.
That is acceptable.
It should be treated as:
- correctness scaffold
- ABI scaffold
- kernel replacement seam

It should not be treated as a production optimization.
