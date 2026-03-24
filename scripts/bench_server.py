"""Benchmark a live MOLA server under concurrency.

This script is meant for MOLA v2 validation:
  - compare base vs same-adapter vs mixed-adapter traffic
  - sweep concurrency levels
  - capture end-to-end latency plus engine metric deltas

Examples:
    python scripts/bench_server.py
    python scripts/bench_server.py --concurrency 1,2,4,8,16 --repeats 3
    python scripts/bench_server.py --same-model rust --mixed-models rust,sql
    python scripts/bench_server.py --extended --concurrency 4,8 --repeats 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_BENCHMARK_VERSION = "0.3.1"

PROMPTS = {
    "base": [
        "Explain in one paragraph what a distributed lock is.",
        "Write a short summary of why caching improves latency.",
        "What is the difference between a thread and a process?",
    ],
    "rust": [
        "Write a Rust function to reverse a string.",
        "Implement binary search in Rust.",
        "Write a Rust struct for a stack.",
    ],
    "sql": [
        "Given CREATE TABLE t (id INT, val INT), return SQL to get max val.",
        "Given a users table with email, return SQL to find duplicate emails.",
        "Given orders(id, customer_id, amount), return SQL for top 5 customers by spend.",
    ],
    "default": [
        "Explain what this model specializes in.",
        "Give a short, direct answer to this prompt.",
        "Write a concise example related to this task.",
    ],
}


@dataclass
class RequestResult:
    model: str
    ok: bool
    status_code: int
    latency_s: float
    chars: int
    error: str | None = None


@dataclass
class ScenarioResult:
    scenario: str
    concurrency: int
    requests: int
    wall_s: float
    p50_ms: float
    p95_ms: float
    req_s: float
    avg_chars: float
    errors: int
    engine_tokens: int
    engine_completed: int
    engine_tok_s: float
    engine_step_lock_wait_ms: float
    engine_insert_lock_wait_ms: float
    engine_routed_decode_reference_enabled: bool
    models: list[str]


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    models: list[str]
    prompt_mode: str = "default"
    max_tokens: int | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--same-model", default=None)
    parser.add_argument("--mixed-models", default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--routed-validation",
        action="store_true",
        help="Run the focused routed decode validation scenario set and require routed_decode_reference_enabled=true",
    )
    parser.add_argument(
        "--require-routed-decode-reference",
        action="store_true",
        help="Fail fast unless /v1/engine/metrics reports routed_decode_reference_enabled=true",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Include long-prefill, long-decode, and fairness scenarios",
    )
    parser.add_argument(
        "--long-prefill-tokens",
        type=int,
        default=16,
        help="Decode length for long-prefill scenarios",
    )
    parser.add_argument(
        "--long-decode-tokens",
        type=int,
        default=256,
        help="Decode length for long-decode scenarios",
    )
    parser.add_argument("--json-out", default=None)
    parser.add_argument(
        "--save-version",
        default=DEFAULT_BENCHMARK_VERSION,
        help="Save results under benchmark/<version>/ when --json-out is not set",
    )
    return parser.parse_args()


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_concurrency(value: str) -> list[int]:
    levels = [int(item) for item in _parse_csv(value)]
    if not levels:
        raise ValueError("At least one concurrency value is required")
    return levels


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1))
    return ordered[rank]


def _pick_prompt(model: str, index: int, base_model_path: str, adapter_names: set[str]) -> str:
    if model == "base" or model == base_model_path:
        prompts = PROMPTS["base"]
    elif model in adapter_names:
        prompts = PROMPTS.get(model, PROMPTS["default"])
    else:
        prompts = PROMPTS["default"]
    return prompts[index % len(prompts)]


LONG_PREFILL_CONTEXT = "\n".join(
    f"Reference paragraph {i}: caching, retries, observability, scheduling, and API ergonomics all interact in production systems."
    for i in range(1, 65)
)


def _build_prompt(
    model: str,
    index: int,
    base_model_path: str,
    adapter_names: set[str],
    prompt_mode: str,
) -> str:
    prompt = _pick_prompt(model, index, base_model_path, adapter_names)
    if prompt_mode == "long_prefill":
        return (
            "Read the following background carefully before answering.\n\n"
            f"{LONG_PREFILL_CONTEXT}\n\n"
            f"Task: {prompt}\n"
            "Keep the final answer concise."
        )
    if prompt_mode == "long_decode":
        return (
            f"{prompt}\n"
            "Answer in a detailed, structured way with numbered steps, rationale, examples, and edge cases."
        )
    return prompt


async def _get_json(client: httpx.AsyncClient, path: str) -> dict[str, Any]:
    response = await client.get(path)
    response.raise_for_status()
    return response.json()


async def _post_chat(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> RequestResult:
    started = time.perf_counter()
    try:
        response = await client.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            },
        )
        latency_s = time.perf_counter() - started
        payload = response.json()
        if response.status_code != 200:
            return RequestResult(
                model=model,
                ok=False,
                status_code=response.status_code,
                latency_s=latency_s,
                chars=0,
                error=json.dumps(payload),
            )

        text = payload["choices"][0]["message"]["content"]
        return RequestResult(
            model=model,
            ok=True,
            status_code=response.status_code,
            latency_s=latency_s,
            chars=len(text),
        )
    except Exception as exc:
        return RequestResult(
            model=model,
            ok=False,
            status_code=0,
            latency_s=time.perf_counter() - started,
            chars=0,
            error=str(exc),
        )


async def _run_wave(
    client: httpx.AsyncClient,
    base_url: str,
    models: list[str],
    base_model_path: str,
    adapter_names: set[str],
    concurrency: int,
    max_tokens: int,
    wave_index: int,
    prompt_mode: str = "default",
) -> list[RequestResult]:
    tasks = []
    for i in range(concurrency):
        model = models[i % len(models)]
        prompt = _build_prompt(
            model,
            wave_index * concurrency + i,
            base_model_path,
            adapter_names,
            prompt_mode,
        )
        tasks.append(_post_chat(client, base_url, model, prompt, max_tokens))
    return await asyncio.gather(*tasks)


async def _run_scenario(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    scenario: str,
    models: list[str],
    base_model_path: str,
    adapter_names: set[str],
    concurrency: int,
    repeats: int,
    max_tokens: int,
    prompt_mode: str = "default",
) -> ScenarioResult:
    before = await _get_json(client, f"{base_url}/v1/engine/metrics")
    started = time.perf_counter()
    results: list[RequestResult] = []

    for wave in range(repeats):
        results.extend(
            await _run_wave(
                client,
                base_url,
                models,
                base_model_path,
                adapter_names,
                concurrency,
                max_tokens,
                wave,
                prompt_mode,
            )
        )

    wall_s = time.perf_counter() - started
    after = await _get_json(client, f"{base_url}/v1/engine/metrics")

    latencies = [r.latency_s for r in results]
    errors = sum(1 for r in results if not r.ok)
    avg_chars = sum(r.chars for r in results) / len(results) if results else 0.0
    token_delta = after["total_tokens_generated"] - before["total_tokens_generated"]
    completed_delta = after["requests_completed"] - before["requests_completed"]
    step_lock_wait_delta = (
        after.get("total_step_lock_wait_ms", 0.0) - before.get("total_step_lock_wait_ms", 0.0)
    )
    insert_lock_wait_delta = (
        after.get("total_insert_lock_wait_ms", 0.0) - before.get("total_insert_lock_wait_ms", 0.0)
    )

    return ScenarioResult(
        scenario=scenario,
        concurrency=concurrency,
        requests=len(results),
        wall_s=wall_s,
        p50_ms=_percentile(latencies, 0.50) * 1000,
        p95_ms=_percentile(latencies, 0.95) * 1000,
        req_s=(len(results) / wall_s) if wall_s > 0 else 0.0,
        avg_chars=avg_chars,
        errors=errors,
        engine_tokens=token_delta,
        engine_completed=completed_delta,
        engine_tok_s=(token_delta / wall_s) if wall_s > 0 else 0.0,
        engine_step_lock_wait_ms=step_lock_wait_delta,
        engine_insert_lock_wait_ms=insert_lock_wait_delta,
        engine_routed_decode_reference_enabled=bool(
            after.get("routed_decode_reference_enabled", False)
        ),
        models=models,
    )


def _choose_same_model(adapter_names: list[str], requested: str | None) -> str:
    if requested:
        return requested
    if adapter_names:
        return adapter_names[0]
    return "base"


def _choose_mixed_models(
    adapter_names: list[str],
    base_model_path: str,
    requested: str | None,
    same_model: str,
) -> list[str]:
    if requested:
        return _parse_csv(requested)
    if len(adapter_names) >= 2:
        return adapter_names
    if adapter_names:
        return ["base", adapter_names[0]]
    return ["base", base_model_path]


def _build_fairness_models(
    adapter_names: list[str],
    same_model: str,
    base_model_path: str,
) -> list[str]:
    hot = same_model if same_model in adapter_names else (
        adapter_names[0] if adapter_names else "base"
    )
    cold = [name for name in adapter_names if name != hot][:4]
    if not cold:
        cold = ["base"] if hot != "base" else [base_model_path]
    return [hot, hot, hot, hot, *cold]


def _build_scenarios(
    *,
    adapter_names: list[str],
    base_model_path: str,
    same_model: str,
    mixed_models: list[str],
    max_tokens: int,
    extended: bool,
    long_prefill_tokens: int,
    long_decode_tokens: int,
) -> list[ScenarioSpec]:
    scenarios = [
        ScenarioSpec("base", ["base"], max_tokens=max_tokens),
        ScenarioSpec("same", [same_model], max_tokens=max_tokens),
        ScenarioSpec("mixed", mixed_models, max_tokens=max_tokens),
    ]
    if not extended:
        return scenarios

    scenarios.extend(
        [
            ScenarioSpec(
                "long-prefill-same",
                [same_model],
                prompt_mode="long_prefill",
                max_tokens=min(max_tokens, long_prefill_tokens),
            ),
            ScenarioSpec(
                "long-prefill-mixed",
                mixed_models,
                prompt_mode="long_prefill",
                max_tokens=min(max_tokens, long_prefill_tokens),
            ),
            ScenarioSpec(
                "long-decode-same",
                [same_model],
                prompt_mode="long_decode",
                max_tokens=max(max_tokens, long_decode_tokens),
            ),
            ScenarioSpec(
                "long-decode-mixed",
                mixed_models,
                prompt_mode="long_decode",
                max_tokens=max(max_tokens, long_decode_tokens),
            ),
            ScenarioSpec(
                "fairness",
                _build_fairness_models(adapter_names, same_model, base_model_path),
                max_tokens=max_tokens,
            ),
        ]
    )
    return scenarios


def _build_routed_validation_scenarios(
    *,
    same_model: str,
    mixed_models: list[str],
    adapter_names: list[str],
    base_model_path: str,
    max_tokens: int,
    long_decode_tokens: int,
) -> list[ScenarioSpec]:
    return [
        ScenarioSpec("same", [same_model], max_tokens=max_tokens),
        ScenarioSpec("mixed", mixed_models, max_tokens=max_tokens),
        ScenarioSpec(
            "long-decode-mixed",
            mixed_models,
            prompt_mode="long_decode",
            max_tokens=max(max_tokens, long_decode_tokens),
        ),
        ScenarioSpec(
            "fairness",
            _build_fairness_models(adapter_names, same_model, base_model_path),
            max_tokens=max_tokens,
        ),
    ]


def _print_header(
    base_url: str,
    health: dict,
    adapters: list[dict],
    scenarios: list[ScenarioSpec],
    engine_metrics: dict[str, Any],
):
    print("=" * 88)
    print("MOLA v2 Benchmark")
    print("=" * 88)
    print(f"Base URL: {base_url}")
    print(f"Model:    {health['model']}")
    print(f"Adapters: {', '.join(a['name'] for a in adapters) if adapters else '(none)'}")
    print(
        "Routed decode reference: "
        f"{engine_metrics.get('routed_decode_reference_enabled', False)}"
    )
    print("Scenarios:")
    for scenario in scenarios:
        suffix = ""
        if scenario.prompt_mode != "default":
            suffix = f" [{scenario.prompt_mode}]"
        print(f"  - {scenario.name}: {', '.join(scenario.models)}{suffix}")
    print()


def _print_results(results: list[ScenarioResult]):
    print(
        f"{'scenario':<14} {'conc':>4} {'reqs':>4} {'wall_s':>8} "
        f"{'p50_ms':>8} {'p95_ms':>8} {'req/s':>8} {'tok/s':>8} {'errs':>5}"
    )
    print("-" * 88)
    for row in results:
        print(
            f"{row.scenario:<14} {row.concurrency:>4} {row.requests:>4} "
            f"{row.wall_s:>8.2f} {row.p50_ms:>8.1f} {row.p95_ms:>8.1f} "
            f"{row.req_s:>8.1f} {row.engine_tok_s:>8.1f} {row.errors:>5}"
        )


def _default_output_path(version: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("benchmark") / version / f"bench-{timestamp}.json"


async def main():
    args = _parse_args()
    concurrency_levels = _parse_concurrency(args.concurrency)
    require_routed_decode_reference = (
        args.require_routed_decode_reference or args.routed_validation
    )

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        health = await _get_json(client, f"{args.base_url}/health")
        engine_metrics = await _get_json(client, f"{args.base_url}/v1/engine/metrics")
        if require_routed_decode_reference and not engine_metrics.get(
            "routed_decode_reference_enabled", False
        ):
            raise SystemExit(
                "Server is not running with routed_decode_reference_enabled=true"
            )
        adapters_payload = await _get_json(client, f"{args.base_url}/v1/adapters")
        adapters = adapters_payload["adapters"]
        adapter_names = [a["name"] for a in adapters]
        adapter_name_set = set(adapter_names)

        same_model = _choose_same_model(adapter_names, args.same_model)
        mixed_models = _choose_mixed_models(
            adapter_names,
            health["model"],
            args.mixed_models,
            same_model,
        )
        scenarios = (
            _build_routed_validation_scenarios(
                same_model=same_model,
                mixed_models=mixed_models,
                adapter_names=adapter_names,
                base_model_path=health["model"],
                max_tokens=args.max_tokens,
                long_decode_tokens=args.long_decode_tokens,
            )
            if args.routed_validation
            else _build_scenarios(
                adapter_names=adapter_names,
                base_model_path=health["model"],
                same_model=same_model,
                mixed_models=mixed_models,
                max_tokens=args.max_tokens,
                extended=args.extended,
                long_prefill_tokens=args.long_prefill_tokens,
                long_decode_tokens=args.long_decode_tokens,
            )
        )

        _print_header(args.base_url, health, adapters, scenarios, engine_metrics)

        if args.warmup > 0:
            print(f"Warming up with {args.warmup} request(s) on {same_model}...")
            for wave in range(args.warmup):
                await _run_wave(
                    client,
                    args.base_url,
                    [same_model],
                    health["model"],
                    adapter_name_set,
                    1,
                    min(args.max_tokens, 32),
                    wave,
                )
            print()

        results: list[ScenarioResult] = []
        for concurrency in concurrency_levels:
            for scenario in scenarios:
                results.append(
                    await _run_scenario(
                        client,
                        base_url=args.base_url,
                        scenario=scenario.name,
                        models=scenario.models,
                        base_model_path=health["model"],
                        adapter_names=adapter_name_set,
                        concurrency=concurrency,
                        repeats=args.repeats,
                        max_tokens=scenario.max_tokens or args.max_tokens,
                        prompt_mode=scenario.prompt_mode,
                    )
                )

        _print_results(results)

        output_path: Path | None = None
        if args.json_out:
            output_path = Path(args.json_out)
        elif args.save_version:
            output_path = _default_output_path(args.save_version)

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps([asdict(result) for result in results], indent=2)
            )
            print(f"\nWrote JSON results to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
