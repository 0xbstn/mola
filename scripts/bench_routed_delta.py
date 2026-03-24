#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx

from mola.infrastructure.metal_routed_decode import _METAL_LORA_DELTA_SOURCE


@dataclass(frozen=True)
class Case:
    name: str
    in_dim: int
    rank: int
    out_dim: int


@dataclass
class Result:
    mode: str
    case: str
    tokens: int
    dtype: str
    ref_ms: float
    gather_ms: float | None
    gather_sorted_ms: float | None
    metal_ms: float | None
    gather_speedup: float | None
    gather_sorted_speedup: float | None
    metal_speedup: float | None
    max_abs_err_gather: float
    max_abs_err_gather_sorted: float | None
    max_abs_err_metal: float | None
    gather_ok: bool
    gather_sorted_ok: bool | None
    metal_ok: bool | None


DEFAULT_CASES = [
    Case("attn_q_proj", in_dim=896, rank=8, out_dim=896),
    Case("attn_kv_proj", in_dim=896, rank=8, out_dim=128),
    Case("mlp_down_proj", in_dim=4864, rank=8, out_dim=896),
    Case("mlp_up_gate_proj", in_dim=896, rank=8, out_dim=4864),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default="1,2,4,8")
    parser.add_argument("--modes", default="homogeneous,mixed")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--slots", type=int, default=8)
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


def _parse_tokens(value: str) -> list[int]:
    levels = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not levels:
        raise ValueError("at least one token count is required")
    return levels


def _parse_modes(value: str) -> list[str]:
    modes = [item.strip() for item in value.split(",") if item.strip()]
    supported = {"homogeneous", "mixed"}
    unknown = sorted(set(modes) - supported)
    if unknown:
        raise ValueError(f"unsupported mode(s): {unknown}")
    if not modes:
        raise ValueError("at least one mode is required")
    return modes


def _dtype(name: str):
    if name == "float16":
        return mx.float16
    if name == "float32":
        return mx.float32
    raise ValueError(f"unsupported dtype: {name}")


def _build_kernel():
    return mx.fast.metal_kernel(
        name="mola_routed_lora_delta_v1_bench",
        input_names=["x", "a", "b", "scale"],
        output_names=["out"],
        source=_METAL_LORA_DELTA_SOURCE,
        ensure_row_contiguous=True,
    )


def _template_dtype(dtype):
    if dtype in (mx.float16, mx.bfloat16):
        return mx.float32
    return dtype


def _reference_delta(x, a, b, scale):
    return scale * ((x @ a) @ b)


def _gather_delta(x, a, b, scale):
    rows = int(x.shape[0])
    idx = mx.zeros((rows,), dtype=mx.int32)
    x3 = mx.expand_dims(x, -2)
    z = mx.gather_mm(x3, mx.expand_dims(a, 0), rhs_indices=idx, sorted_indices=True)
    y = mx.gather_mm(z, mx.expand_dims(b, 0), rhs_indices=idx, sorted_indices=True)
    return scale * y.squeeze(-2)


def _metal_delta(kernel, x, a, b, scale):
    rows = int(x.shape[0])
    out_dim = int(b.shape[1])
    grid_x = max(rows * out_dim, 1)
    threadgroup_x = min(256, grid_x)
    return kernel(
        inputs=[x, a, b, scale],
        template=[("T", _template_dtype(x.dtype))],
        grid=(grid_x, 1, 1),
        threadgroup=(threadgroup_x, 1, 1),
        output_shapes=[(rows, out_dim)],
        output_dtypes=[x.dtype],
    )[0]


def _sync(value):
    mx.eval(value)
    return value


def _sample(shape, dtype, *, seed: int, low: float, high: float):
    return mx.random.uniform(
        low=low,
        high=high,
        shape=shape,
        dtype=dtype,
        key=mx.random.key(seed),
    )


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


def _gather_mixed_sorted_delta(x, a, b, scales, slot_rows):
    order = mx.argsort(slot_rows)
    inv_order = mx.argsort(order)
    sorted_rows = slot_rows[order]
    sorted_x = mx.expand_dims(x[order], -2)
    z = mx.gather_mm(sorted_x, a, rhs_indices=sorted_rows, sorted_indices=True)
    y = mx.gather_mm(z, b, rhs_indices=sorted_rows, sorted_indices=True).squeeze(-2)
    return (scales[sorted_rows].reshape((-1, 1)) * y)[inv_order]


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


def _run_case(
    kernel,
    case: Case,
    tokens: int,
    mode: str,
    dtype_name: str,
    warmup: int,
    iters: int,
    slots: int,
    scale_value: float,
    value_low: float,
    value_high: float,
) -> Result:
    dtype = _dtype(dtype_name)
    seed = (hash((case.name, tokens, dtype_name, mode)) & 0xFFFFFFFF) or 1
    x = _sample(
        (tokens, case.in_dim),
        dtype,
        seed=seed,
        low=value_low,
        high=value_high,
    )

    if mode == "homogeneous":
        a = _sample(
            (case.in_dim, case.rank),
            dtype,
            seed=seed + 1,
            low=value_low,
            high=value_high,
        )
        b = _sample(
            (case.rank, case.out_dim),
            dtype,
            seed=seed + 2,
            low=value_low,
            high=value_high,
        )
        scale = mx.array([scale_value], dtype=dtype)

        ref = _sync(_reference_delta(x, a, b, scale))
        gather = _sync(_gather_delta(x, a, b, scale))
        metal = _sync(_metal_delta(kernel, x, a, b, scale))

        return Result(
            mode=mode,
            case=case.name,
            tokens=tokens,
            dtype=dtype_name,
            ref_ms=_bench(lambda: _reference_delta(x, a, b, scale), warmup, iters),
            gather_ms=_bench(lambda: _gather_delta(x, a, b, scale), warmup, iters),
            gather_sorted_ms=None,
            metal_ms=_bench(lambda: _metal_delta(kernel, x, a, b, scale), warmup, iters),
            gather_speedup=None,
            gather_sorted_speedup=None,
            metal_speedup=None,
            max_abs_err_gather=_max_abs_err(ref, gather),
            max_abs_err_gather_sorted=None,
            max_abs_err_metal=_max_abs_err(ref, metal),
            gather_ok=bool(mx.allclose(ref.astype(mx.float32), gather.astype(mx.float32), atol=5e-2, rtol=5e-2).item()),
            gather_sorted_ok=None,
            metal_ok=bool(mx.allclose(ref.astype(mx.float32), metal.astype(mx.float32), atol=5e-2, rtol=5e-2).item()),
        )

    a = _sample(
        (slots, case.in_dim, case.rank),
        dtype,
        seed=seed + 1,
        low=value_low,
        high=value_high,
    )
    b = _sample(
        (slots, case.rank, case.out_dim),
        dtype,
        seed=seed + 2,
        low=value_low,
        high=value_high,
    )
    scales = mx.array(
        [scale_value + (0.25 * slot) for slot in range(slots)],
        dtype=dtype,
    )
    slot_rows = mx.array([(row * 3) % slots for row in range(tokens)], dtype=mx.int32)

    ref = _sync(_reference_mixed_delta(x, a, b, scales, slot_rows))
    gather = _sync(_gather_mixed_delta(x, a, b, scales, slot_rows))
    gather_sorted = _sync(_gather_mixed_sorted_delta(x, a, b, scales, slot_rows))

    ref_ms = _bench(lambda: _reference_mixed_delta(x, a, b, scales, slot_rows), warmup, iters)
    gather_ms = _bench(lambda: _gather_mixed_delta(x, a, b, scales, slot_rows), warmup, iters)
    gather_sorted_ms = _bench(
        lambda: _gather_mixed_sorted_delta(x, a, b, scales, slot_rows),
        warmup,
        iters,
    )

    return Result(
        mode=mode,
        case=case.name,
        tokens=tokens,
        dtype=dtype_name,
        ref_ms=ref_ms,
        gather_ms=gather_ms,
        gather_sorted_ms=gather_sorted_ms,
        metal_ms=None,
        gather_speedup=(ref_ms / gather_ms) if gather_ms > 0 else math.inf,
        gather_sorted_speedup=(ref_ms / gather_sorted_ms) if gather_sorted_ms > 0 else math.inf,
        metal_speedup=None,
        max_abs_err_gather=_max_abs_err(ref, gather),
        max_abs_err_gather_sorted=_max_abs_err(ref, gather_sorted),
        max_abs_err_metal=None,
        gather_ok=bool(mx.allclose(ref.astype(mx.float32), gather.astype(mx.float32), atol=5e-2, rtol=5e-2).item()),
        gather_sorted_ok=bool(
            mx.allclose(ref.astype(mx.float32), gather_sorted.astype(mx.float32), atol=5e-2, rtol=5e-2).item()
        ),
        metal_ok=None,
    )


def _print_results(results: list[Result]) -> None:
    print(
        f"{'mode':<12} {'case':<18} {'T':>3} {'dtype':>8} {'ref_ms':>9} {'gather_ms':>10} {'g_sorted':>10} {'metal_ms':>10} {'gx':>7} {'gsx':>7} {'mx':>7} {'g_err':>10} {'gs_err':>10} {'m_err':>10} {'g_ok':>5} {'gs_ok':>5} {'m_ok':>5}"
    )
    print("-" * 196)
    for row in results:
        print(
            f"{row.mode:<12} {row.case:<18} {row.tokens:>3} {row.dtype:>8} "
            f"{row.ref_ms:>9.4f} "
            f"{(row.gather_ms if row.gather_ms is not None else float('nan')):>10.4f} "
            f"{(row.gather_sorted_ms if row.gather_sorted_ms is not None else float('nan')):>10.4f} "
            f"{(row.metal_ms if row.metal_ms is not None else float('nan')):>10.4f} "
            f"{(row.gather_speedup if row.gather_speedup is not None else row.ref_ms / row.gather_ms):>7.3f} "
            f"{(row.gather_sorted_speedup if row.gather_sorted_speedup is not None else float('nan')):>7.3f} "
            f"{(row.metal_speedup if row.metal_speedup is not None else (row.ref_ms / row.metal_ms if row.metal_ms else float('nan'))):>7.3f} "
            f"{row.max_abs_err_gather:>10.5f} "
            f"{(row.max_abs_err_gather_sorted if row.max_abs_err_gather_sorted is not None else float('nan')):>10.5f} "
            f"{(row.max_abs_err_metal if row.max_abs_err_metal is not None else float('nan')):>10.5f} "
            f"{str(row.gather_ok):>5} "
            f"{str(row.gather_sorted_ok if row.gather_sorted_ok is not None else ''):>5} "
            f"{str(row.metal_ok if row.metal_ok is not None else ''):>5}"
        )


def main() -> int:
    args = _parse_args()
    token_levels = _parse_tokens(args.tokens)
    modes = _parse_modes(args.modes)
    kernel = _build_kernel()

    results: list[Result] = []
    for mode in modes:
        for case in DEFAULT_CASES:
            for tokens in token_levels:
                results.append(
                    _run_case(
                        kernel=kernel,
                        case=case,
                        tokens=tokens,
                        mode=mode,
                        dtype_name=args.dtype,
                        warmup=args.warmup,
                        iters=args.iters,
                        slots=args.slots,
                        scale_value=args.scale,
                        value_low=args.value_low,
                        value_high=args.value_high,
                    )
                )

    for row in results:
        if row.gather_ms is not None:
            row.gather_speedup = (row.ref_ms / row.gather_ms) if row.gather_ms > 0 else math.inf
        if row.gather_sorted_ms is not None:
            row.gather_sorted_speedup = (row.ref_ms / row.gather_sorted_ms) if row.gather_sorted_ms > 0 else math.inf
        if row.metal_ms is not None:
            row.metal_speedup = (row.ref_ms / row.metal_ms) if row.metal_ms > 0 else math.inf

    _print_results(results)

    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(row) for row in results], indent=2))
        print(f"\nWrote JSON results to {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
