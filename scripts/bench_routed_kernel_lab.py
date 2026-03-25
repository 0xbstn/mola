#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx


@dataclass(frozen=True)
class Case:
    name: str
    in_dim: int
    rank: int
    out_dim: int


@dataclass
class Result:
    case: str
    tokens: int
    distinct_slots: int
    pattern: str
    dtype: str
    ref_ms: float
    gather_ms: float
    hybrid_ba_ms: float
    metal_tiled_ms: float | None
    gather_speedup: float
    hybrid_ba_speedup: float
    metal_tiled_speedup: float | None
    gather_err: float
    hybrid_ba_err: float
    metal_tiled_err: float | None
    gather_ok: bool
    hybrid_ba_ok: bool
    metal_tiled_ok: bool | None


DEFAULT_CASES = [
    Case("attn_q_proj", in_dim=896, rank=8, out_dim=896),
    Case("attn_kv_proj", in_dim=896, rank=8, out_dim=128),
    Case("mlp_down_proj", in_dim=4864, rank=8, out_dim=896),
    Case("mlp_up_gate_proj", in_dim=896, rank=8, out_dim=4864),
]


_METAL_REDUCE_MIXED_SOURCE = """
uint lane_x = thread_position_in_threadgroup.x;
uint lane_y = thread_position_in_threadgroup.y;
uint row = threadgroup_position_in_grid.y;
uint rows = x_shape[0];
uint in_dim = x_shape[1];
uint rank = a_shape[2];
uint out_dim = b_shape[2];

if (row >= rows) {
    return;
}

uint slot = uint(slot_rows[row]);
threadgroup T partial[THREADS_Y][THREADS_X];
threadgroup T zbuf[MAX_R];

if (lane_y < rank) {
    T sum = T(0);
    for (uint d = lane_x; d < in_dim; d += THREADS_X) {
        uint a_idx = ((slot * in_dim + d) * rank) + lane_y;
        sum += T(x[row * in_dim + d]) * T(a[a_idx]);
    }
    partial[lane_y][lane_x] = sum;
}

threadgroup_barrier(mem_flags::mem_threadgroup);

if (lane_y < rank && lane_x == 0) {
    T z = T(0);
    for (uint lx = 0; lx < THREADS_X; ++lx) {
        z += partial[lane_y][lx];
    }
    zbuf[lane_y] = z;
}

threadgroup_barrier(mem_flags::mem_threadgroup);

if (lane_y == 0) {
    T scale = T(scales[slot]);
    for (uint out_col = lane_x; out_col < out_dim; out_col += THREADS_X) {
        T acc = T(0);
        for (uint r = 0; r < rank; ++r) {
            uint b_idx = ((slot * rank + r) * out_dim) + out_col;
            acc += zbuf[r] * T(b[b_idx]);
        }
        out[row * out_dim + out_col] = scale * acc;
    }
}
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default="4,8,16,32")
    parser.add_argument("--distinct-slots", default="2,4,8")
    parser.add_argument("--patterns", default="cycle,blocks,hot80")
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--scale", type=float, default=4.0)
    parser.add_argument("--value-low", type=float, default=-0.25)
    parser.add_argument("--value-high", type=float, default=0.25)
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default="float16",
    )
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


def _parse_csv_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one integer value")
    return values


def _parse_patterns(value: str) -> list[str]:
    patterns = [item.strip() for item in value.split(",") if item.strip()]
    supported = {"cycle", "blocks", "hot80"}
    unknown = sorted(set(patterns) - supported)
    if unknown:
        raise ValueError(f"unsupported pattern(s): {unknown}")
    if not patterns:
        raise ValueError("expected at least one pattern")
    return patterns


def _dtype(name: str):
    if name == "float16":
        return mx.float16
    if name == "float32":
        return mx.float32
    raise ValueError(f"unsupported dtype: {name}")


def _template_dtype(dtype):
    if dtype in (mx.float16, mx.bfloat16):
        return mx.float32
    return dtype


def _sample(shape, dtype, *, seed: int, low: float, high: float):
    return mx.random.uniform(
        low=low,
        high=high,
        shape=shape,
        dtype=dtype,
        key=mx.random.key(seed),
    )


def _sync(value):
    mx.eval(value)
    return value


def _bench(fn, warmup: int, iters: int) -> float:
    out = None
    for _ in range(warmup):
        out = fn()
        _sync(out)
    started = time.perf_counter()
    for _ in range(iters):
        out = fn()
        _sync(out)
    elapsed = time.perf_counter() - started
    return elapsed * 1000.0 / iters


def _max_abs_err(a, b) -> float:
    diff = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
    return float(mx.max(diff).item())


def _build_slot_rows(tokens: int, distinct_slots: int, pattern: str) -> mx.array:
    if pattern == "cycle":
        values = [row % distinct_slots for row in range(tokens)]
    elif pattern == "blocks":
        block = max(tokens // distinct_slots, 1)
        values = [min(row // block, distinct_slots - 1) for row in range(tokens)]
    elif pattern == "hot80":
        hot_rows = max(int(tokens * 0.8), 1)
        values = [0] * hot_rows
        if distinct_slots == 1:
            values.extend([0] * (tokens - hot_rows))
        else:
            for row in range(tokens - hot_rows):
                values.append(1 + (row % (distinct_slots - 1)))
    else:
        raise ValueError(f"unsupported pattern: {pattern}")
    return mx.array(values[:tokens], dtype=mx.int32)


def _reference_mixed_delta(x, a, b, scales, slot_rows):
    rows = []
    for row, slot_row in enumerate(slot_rows.tolist()):
        delta = (x[row] @ a[slot_row]) @ b[slot_row]
        rows.append(scales[slot_row] * delta)
    return mx.stack(rows, axis=0)


def _gather_mixed_delta(x, a, b, scales, slot_rows):
    x3 = mx.expand_dims(x, -2)
    z = mx.gather_mm(x3, a, rhs_indices=slot_rows, sorted_indices=False)
    y = mx.gather_mm(z, b, rhs_indices=slot_rows, sorted_indices=False).squeeze(-2)
    return scales[slot_rows].reshape((-1, 1)) * y


def _hybrid_ba_mixed_delta(x, a, b, scales, slot_rows):
    x3 = mx.expand_dims(x, -2)
    z = mx.gather_mm(x3, a, rhs_indices=slot_rows, sorted_indices=False)
    gathered_b = b[slot_rows]
    y = mx.matmul(z, gathered_b).squeeze(-2)
    return scales[slot_rows].reshape((-1, 1)) * y


def _build_mixed_tiled_kernel():
    return mx.fast.metal_kernel(
        name="mola_routed_lora_delta_mixed_reduce_v1_bench",
        input_names=["x", "a", "b", "scales", "slot_rows"],
        output_names=["out"],
        source=_METAL_REDUCE_MIXED_SOURCE,
        ensure_row_contiguous=True,
    )


def _metal_tiled_mixed_delta(kernel, x, a, b, scales, slot_rows):
    rows = int(x.shape[0])
    rank = int(a.shape[2])
    threads_x = min(128, 1024 // max(rank, 1))
    return kernel(
        inputs=[x, a, b, scales, slot_rows],
        template=[
            ("T", _template_dtype(x.dtype)),
            ("MAX_R", rank),
            ("THREADS_X", threads_x),
            ("THREADS_Y", rank),
        ],
        grid=(threads_x, rows * rank, 1),
        threadgroup=(threads_x, rank, 1),
        output_shapes=[(rows, int(b.shape[2]))],
        output_dtypes=[x.dtype],
    )[0]


def _run_case(
    kernel,
    case: Case,
    tokens: int,
    distinct_slots: int,
    pattern: str,
    dtype_name: str,
    warmup: int,
    iters: int,
    slots: int,
    scale_value: float,
    value_low: float,
    value_high: float,
) -> Result:
    dtype = _dtype(dtype_name)
    seed = (hash((case.name, tokens, distinct_slots, pattern, dtype_name)) & 0xFFFFFFFF) or 1
    x = _sample((tokens, case.in_dim), dtype, seed=seed, low=value_low, high=value_high)
    a = _sample((slots, case.in_dim, case.rank), dtype, seed=seed + 1, low=value_low, high=value_high)
    b = _sample((slots, case.rank, case.out_dim), dtype, seed=seed + 2, low=value_low, high=value_high)
    scales = mx.array([scale_value + (0.25 * slot) for slot in range(slots)], dtype=dtype)
    slot_rows = _build_slot_rows(tokens, distinct_slots, pattern)

    ref = _sync(_reference_mixed_delta(x, a, b, scales, slot_rows))
    gather = _sync(_gather_mixed_delta(x, a, b, scales, slot_rows))
    hybrid_ba = _sync(_hybrid_ba_mixed_delta(x, a, b, scales, slot_rows))

    metal = None
    metal_err = None
    metal_ok = None
    metal_ms = None
    try:
        metal = _sync(_metal_tiled_mixed_delta(kernel, x, a, b, scales, slot_rows))
        metal_err = _max_abs_err(ref, metal)
        metal_ok = bool(
            mx.allclose(
                ref.astype(mx.float32),
                metal.astype(mx.float32),
                atol=5e-2,
                rtol=5e-2,
            ).item()
        )
        metal_ms = _bench(
            lambda: _metal_tiled_mixed_delta(kernel, x, a, b, scales, slot_rows),
            warmup,
            iters,
        )
    except Exception:
        metal = None

    ref_ms = _bench(lambda: _reference_mixed_delta(x, a, b, scales, slot_rows), warmup, iters)
    gather_ms = _bench(lambda: _gather_mixed_delta(x, a, b, scales, slot_rows), warmup, iters)
    hybrid_ba_ms = _bench(
        lambda: _hybrid_ba_mixed_delta(x, a, b, scales, slot_rows),
        warmup,
        iters,
    )

    return Result(
        case=case.name,
        tokens=tokens,
        distinct_slots=distinct_slots,
        pattern=pattern,
        dtype=dtype_name,
        ref_ms=ref_ms,
        gather_ms=gather_ms,
        hybrid_ba_ms=hybrid_ba_ms,
        metal_tiled_ms=metal_ms,
        gather_speedup=(ref_ms / gather_ms) if gather_ms > 0 else math.inf,
        hybrid_ba_speedup=(ref_ms / hybrid_ba_ms) if hybrid_ba_ms > 0 else math.inf,
        metal_tiled_speedup=(ref_ms / metal_ms) if metal_ms else None,
        gather_err=_max_abs_err(ref, gather),
        hybrid_ba_err=_max_abs_err(ref, hybrid_ba),
        metal_tiled_err=metal_err,
        gather_ok=bool(mx.allclose(ref.astype(mx.float32), gather.astype(mx.float32), atol=5e-2, rtol=5e-2).item()),
        hybrid_ba_ok=bool(
            mx.allclose(ref.astype(mx.float32), hybrid_ba.astype(mx.float32), atol=5e-2, rtol=5e-2).item()
        ),
        metal_tiled_ok=metal_ok,
    )


def _print_results(results: list[Result]) -> None:
    print(
        f"{'case':<18} {'T':>3} {'S':>3} {'pattern':<8} {'dtype':>8} "
        f"{'ref_ms':>9} {'gather':>9} {'hyb_ba':>9} {'metal':>9} "
        f"{'gx':>7} {'hbx':>7} {'mx':>7} "
        f"{'g_err':>9} {'hb_err':>9} {'m_err':>9} "
        f"{'g_ok':>5} {'hb_ok':>6} {'m_ok':>5}"
    )
    print("-" * 168)
    for row in results:
        print(
            f"{row.case:<18} {row.tokens:>3} {row.distinct_slots:>3} {row.pattern:<8} {row.dtype:>8} "
            f"{row.ref_ms:>9.4f} {row.gather_ms:>9.4f} {row.hybrid_ba_ms:>9.4f} "
            f"{(row.metal_tiled_ms if row.metal_tiled_ms is not None else float('nan')):>9.4f} "
            f"{row.gather_speedup:>7.3f} {row.hybrid_ba_speedup:>7.3f} "
            f"{(row.metal_tiled_speedup if row.metal_tiled_speedup is not None else float('nan')):>7.3f} "
            f"{row.gather_err:>9.5f} {row.hybrid_ba_err:>9.5f} "
            f"{(row.metal_tiled_err if row.metal_tiled_err is not None else float('nan')):>9.5f} "
            f"{str(row.gather_ok):>5} {str(row.hybrid_ba_ok):>6} {str(row.metal_tiled_ok if row.metal_tiled_ok is not None else ''):>5}"
        )


def main() -> int:
    args = _parse_args()
    token_levels = _parse_csv_ints(args.tokens)
    distinct_levels = _parse_csv_ints(args.distinct_slots)
    patterns = _parse_patterns(args.patterns)
    kernel = _build_mixed_tiled_kernel()

    results: list[Result] = []
    for case in DEFAULT_CASES:
        for tokens in token_levels:
            for distinct_slots in distinct_levels:
                for pattern in patterns:
                    results.append(
                        _run_case(
                            kernel=kernel,
                            case=case,
                            tokens=tokens,
                            distinct_slots=min(distinct_slots, args.slots),
                            pattern=pattern,
                            dtype_name=args.dtype,
                            warmup=args.warmup,
                            iters=args.iters,
                            slots=args.slots,
                            scale_value=args.scale,
                            value_low=args.value_low,
                            value_high=args.value_high,
                        )
                    )

    _print_results(results)

    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(row) for row in results], indent=2))
        print(f"\nWrote JSON results to {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
