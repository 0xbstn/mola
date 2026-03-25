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
import random
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
    arrival_mode: str
    arrival_rate: float
    active_duration_s: float
    drain_s: float
    concurrency: int
    requests: int
    wall_s: float
    p50_ms: float
    p95_ms: float
    req_s: float
    avg_chars: float
    errors: int
    client_avg_inflight: float
    client_peak_inflight: int
    engine_tokens: int
    engine_completed: int
    engine_tok_s: float
    engine_step_lock_wait_ms: float
    engine_insert_lock_wait_ms: float
    engine_completion_batch_size_limit: int
    engine_prefill_batch_size_limit: int
    engine_routed_decode_reference_enabled: bool
    engine_routed_decode_reference_strict: bool
    engine_routed_decode_backend: str
    engine_mixed_decode_migration_enabled: bool
    engine_cache_routed_decode_sessions: bool
    engine_neutralize_lora_delta: bool
    engine_mixed_decode_migration_events: int
    engine_mixed_decode_migrated_sequences: int
    engine_mixed_decode_steps: int
    engine_mixed_decode_rows: int
    engine_avg_mixed_decode_rows: float
    engine_homogeneous_decode_steps: int
    engine_homogeneous_decode_rows: int
    engine_avg_homogeneous_decode_rows: float
    engine_decode_step_samples: int
    engine_decode_active_generators_total: int
    engine_avg_decode_generators_active: float
    engine_prefill_insert_batches: int
    engine_prefill_insert_requests: int
    engine_avg_prefill_insert_batch: float
    engine_prefill_insert_batches_while_decode: int
    engine_prefill_insert_requests_while_decode: int
    engine_avg_prefill_insert_batch_while_decode: float
    engine_prefill_insert_single_batches: int
    engine_prefill_insert_single_batch_ratio: float
    engine_homogeneous_routed_decode_session_builds: int
    engine_homogeneous_routed_decode_session_cache_hits: int
    engine_avg_homogeneous_routed_decode_session_build_ms: float
    engine_mixed_routed_decode_session_builds: int
    engine_mixed_routed_decode_session_cache_hits: int
    engine_avg_mixed_routed_decode_session_build_ms: float
    engine_avg_mixed_decode_migration_since_insert_ms: float
    engine_avg_mixed_decode_migration_since_first_token_ms: float
    engine_avg_mixed_decode_tokens_before_shared: float
    engine_delta_total_calls: int
    engine_delta_total_rows: int
    engine_delta_avg_rows: float
    engine_delta_direct_calls: int
    engine_delta_direct_rows: int
    engine_delta_avg_direct_rows: float
    engine_delta_routed_calls: int
    engine_delta_routed_rows: int
    engine_delta_avg_routed_rows: float
    engine_delta_neutralized_direct_calls: int
    engine_delta_neutralized_direct_rows: int
    engine_delta_neutralized_routed_calls: int
    engine_delta_neutralized_routed_rows: int
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
    parser.add_argument(
        "--arrival-mode",
        choices=["burst", "poisson"],
        default="burst",
        help="burst runs fixed concurrent waves; poisson runs continuous arrivals with a max in-flight cap",
    )
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=None,
        help="Average Poisson arrival rate in requests/sec; defaults to the concurrency value in poisson mode",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=20.0,
        help="Per-repeat active arrival duration for poisson mode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for poisson arrivals",
    )
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


@dataclass
class _InflightTracker:
    current: int = 0
    peak: int = 0
    area: float = 0.0
    last_ts: float = 0.0

    def start(self, now: float):
        self.last_ts = now

    def update(self, new_current: int, now: float):
        self.area += self.current * (now - self.last_ts)
        self.current = new_current
        if self.current > self.peak:
            self.peak = self.current
        self.last_ts = now

    def average(self, total_s: float) -> float:
        if total_s <= 0:
            return 0.0
        return self.area / total_s


async def _run_poisson_window(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    models: list[str],
    base_model_path: str,
    adapter_names: set[str],
    max_inflight: int,
    arrival_rate: float,
    duration_s: float,
    max_tokens: int,
    prompt_mode: str,
    request_index_offset: int,
    rng: random.Random,
) -> tuple[list[RequestResult], float, float, float, int]:
    if arrival_rate <= 0:
        raise ValueError("arrival_rate must be > 0 for poisson mode")

    semaphore = asyncio.Semaphore(max_inflight)
    tracker = _InflightTracker()
    lock = asyncio.Lock()
    results: list[RequestResult] = []
    tasks: list[asyncio.Task[None]] = []
    launches = 0
    started = time.perf_counter()
    tracker.start(started)

    async def _one_request(local_index: int):
        await semaphore.acquire()
        async with lock:
            tracker.update(tracker.current + 1, time.perf_counter())
        try:
            model = models[local_index % len(models)]
            prompt = _build_prompt(
                model,
                request_index_offset + local_index,
                base_model_path,
                adapter_names,
                prompt_mode,
            )
            result = await _post_chat(client, base_url, model, prompt, max_tokens)
            results.append(result)
        finally:
            async with lock:
                tracker.update(max(0, tracker.current - 1), time.perf_counter())
            semaphore.release()

    while True:
        now = time.perf_counter()
        if now - started >= duration_s:
            break
        tasks.append(asyncio.create_task(_one_request(launches)))
        launches += 1
        delay = rng.expovariate(arrival_rate)
        remaining = duration_s - (time.perf_counter() - started)
        if remaining <= 0:
            break
        await asyncio.sleep(min(delay, remaining))

    active_duration_s = time.perf_counter() - started
    if tasks:
        await asyncio.gather(*tasks)
    finished = time.perf_counter()
    tracker.update(tracker.current, finished)
    wall_s = finished - started
    drain_s = max(0.0, wall_s - active_duration_s)
    return results, wall_s, active_duration_s, drain_s, tracker.peak, tracker.average(
        wall_s
    )


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
    arrival_mode: str = "burst",
    arrival_rate: float | None = None,
    duration_s: float = 20.0,
    seed: int = 7,
) -> ScenarioResult:
    before = await _get_json(client, f"{base_url}/v1/engine/metrics")
    results: list[RequestResult] = []
    active_duration_total = 0.0
    drain_total = 0.0
    peak_inflight = 0
    weighted_avg_inflight_area = 0.0
    started = time.perf_counter()

    if arrival_mode == "burst":
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
    else:
        rate = arrival_rate if arrival_rate is not None else float(concurrency)
        rng = random.Random(seed)
        request_offset = 0
        for _wave in range(repeats):
            (
                wave_results,
                wave_wall_s,
                wave_active_s,
                wave_drain_s,
                wave_peak,
                wave_avg_inflight,
            ) = await _run_poisson_window(
                client,
                base_url=base_url,
                models=models,
                base_model_path=base_model_path,
                adapter_names=adapter_names,
                max_inflight=concurrency,
                arrival_rate=rate,
                duration_s=duration_s,
                max_tokens=max_tokens,
                prompt_mode=prompt_mode,
                request_index_offset=request_offset,
                rng=rng,
            )
            request_offset += len(wave_results)
            results.extend(wave_results)
            active_duration_total += wave_active_s
            drain_total += wave_drain_s
            peak_inflight = max(peak_inflight, wave_peak)
            weighted_avg_inflight_area += wave_avg_inflight * wave_wall_s

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
    mixed_migrated_delta = int(
        after.get("mixed_decode_migrated_sequences", 0)
        - before.get("mixed_decode_migrated_sequences", 0)
    )
    mixed_steps_delta = int(
        after.get("mixed_decode_steps", 0)
        - before.get("mixed_decode_steps", 0)
    )
    mixed_rows_delta = int(
        after.get("mixed_decode_rows", 0)
        - before.get("mixed_decode_rows", 0)
    )
    homogeneous_steps_delta = int(
        after.get("homogeneous_decode_steps", 0)
        - before.get("homogeneous_decode_steps", 0)
    )
    homogeneous_rows_delta = int(
        after.get("homogeneous_decode_rows", 0)
        - before.get("homogeneous_decode_rows", 0)
    )
    decode_step_samples_delta = int(
        after.get("decode_step_samples", 0)
        - before.get("decode_step_samples", 0)
    )
    decode_active_generators_total_delta = int(
        after.get("decode_active_generators_total", 0)
        - before.get("decode_active_generators_total", 0)
    )
    prefill_insert_batches_delta = int(
        after.get("prefill_insert_batches", 0)
        - before.get("prefill_insert_batches", 0)
    )
    prefill_insert_requests_delta = int(
        after.get("prefill_insert_requests", 0)
        - before.get("prefill_insert_requests", 0)
    )
    prefill_insert_batches_while_decode_delta = int(
        after.get("prefill_insert_batches_while_decode", 0)
        - before.get("prefill_insert_batches_while_decode", 0)
    )
    prefill_insert_requests_while_decode_delta = int(
        after.get("prefill_insert_requests_while_decode", 0)
        - before.get("prefill_insert_requests_while_decode", 0)
    )
    prefill_insert_single_batches_delta = int(
        after.get("prefill_insert_single_batches", 0)
        - before.get("prefill_insert_single_batches", 0)
    )
    mixed_migration_since_insert_ms_total_delta = (
        after.get("mixed_decode_migration_since_insert_ms_total", 0.0)
        - before.get("mixed_decode_migration_since_insert_ms_total", 0.0)
    )
    mixed_migration_since_first_token_ms_total_delta = (
        after.get("mixed_decode_migration_since_first_token_ms_total", 0.0)
        - before.get("mixed_decode_migration_since_first_token_ms_total", 0.0)
    )
    mixed_migration_tokens_before_shared_delta = int(
        after.get("mixed_decode_migration_tokens_before_shared", 0)
        - before.get("mixed_decode_migration_tokens_before_shared", 0)
    )
    delta_total_calls_delta = int(
        after.get("delta_total_calls", 0)
        - before.get("delta_total_calls", 0)
    )
    delta_total_rows_delta = int(
        after.get("delta_total_rows", 0)
        - before.get("delta_total_rows", 0)
    )
    delta_direct_calls_delta = int(
        after.get("delta_direct_calls", 0)
        - before.get("delta_direct_calls", 0)
    )
    delta_direct_rows_delta = int(
        after.get("delta_direct_rows", 0)
        - before.get("delta_direct_rows", 0)
    )
    delta_routed_calls_delta = int(
        after.get("delta_routed_calls", 0)
        - before.get("delta_routed_calls", 0)
    )
    delta_routed_rows_delta = int(
        after.get("delta_routed_rows", 0)
        - before.get("delta_routed_rows", 0)
    )
    delta_neutralized_direct_calls_delta = int(
        after.get("delta_neutralized_direct_calls", 0)
        - before.get("delta_neutralized_direct_calls", 0)
    )
    delta_neutralized_direct_rows_delta = int(
        after.get("delta_neutralized_direct_rows", 0)
        - before.get("delta_neutralized_direct_rows", 0)
    )
    delta_neutralized_routed_calls_delta = int(
        after.get("delta_neutralized_routed_calls", 0)
        - before.get("delta_neutralized_routed_calls", 0)
    )
    delta_neutralized_routed_rows_delta = int(
        after.get("delta_neutralized_routed_rows", 0)
        - before.get("delta_neutralized_routed_rows", 0)
    )

    if arrival_mode == "burst":
        active_duration_total = wall_s
        drain_total = 0.0
        peak_inflight = concurrency if results else 0
        avg_inflight = concurrency if results else 0.0
    else:
        avg_inflight = (
            weighted_avg_inflight_area / wall_s if wall_s > 0 else 0.0
        )

    return ScenarioResult(
        scenario=scenario,
        arrival_mode=arrival_mode,
        arrival_rate=float(arrival_rate if arrival_rate is not None else concurrency),
        active_duration_s=active_duration_total,
        drain_s=drain_total,
        concurrency=concurrency,
        requests=len(results),
        wall_s=wall_s,
        p50_ms=_percentile(latencies, 0.50) * 1000,
        p95_ms=_percentile(latencies, 0.95) * 1000,
        req_s=(len(results) / wall_s) if wall_s > 0 else 0.0,
        avg_chars=avg_chars,
        errors=errors,
        client_avg_inflight=round(avg_inflight, 2),
        client_peak_inflight=peak_inflight,
        engine_tokens=token_delta,
        engine_completed=completed_delta,
        engine_tok_s=(token_delta / wall_s) if wall_s > 0 else 0.0,
        engine_step_lock_wait_ms=step_lock_wait_delta,
        engine_insert_lock_wait_ms=insert_lock_wait_delta,
        engine_completion_batch_size_limit=int(
            after.get("completion_batch_size_limit", 0)
        ),
        engine_prefill_batch_size_limit=int(
            after.get("prefill_batch_size_limit", 0)
        ),
        engine_routed_decode_reference_enabled=bool(
            after.get("routed_decode_reference_enabled", False)
        ),
        engine_routed_decode_reference_strict=bool(
            after.get("routed_decode_reference_strict", False)
        ),
        engine_routed_decode_backend=str(
            after.get("routed_decode_backend", "reference")
        ),
        engine_mixed_decode_migration_enabled=bool(
            after.get("mixed_decode_migration_enabled", False)
        ),
        engine_cache_routed_decode_sessions=bool(
            after.get("cache_routed_decode_sessions", False)
        ),
        engine_neutralize_lora_delta=bool(
            after.get("neutralize_lora_delta", False)
        ),
        engine_mixed_decode_migration_events=int(
            after.get("mixed_decode_migration_events", 0)
            - before.get("mixed_decode_migration_events", 0)
        ),
        engine_mixed_decode_migrated_sequences=mixed_migrated_delta,
        engine_mixed_decode_steps=mixed_steps_delta,
        engine_mixed_decode_rows=mixed_rows_delta,
        engine_avg_mixed_decode_rows=(
            round(mixed_rows_delta / mixed_steps_delta, 2)
            if mixed_steps_delta
            else 0.0
        ),
        engine_homogeneous_decode_steps=homogeneous_steps_delta,
        engine_homogeneous_decode_rows=homogeneous_rows_delta,
        engine_avg_homogeneous_decode_rows=(
            round(homogeneous_rows_delta / homogeneous_steps_delta, 2)
            if homogeneous_steps_delta
            else 0.0
        ),
        engine_decode_step_samples=decode_step_samples_delta,
        engine_decode_active_generators_total=decode_active_generators_total_delta,
        engine_avg_decode_generators_active=(
            round(
                decode_active_generators_total_delta / decode_step_samples_delta,
                2,
            )
            if decode_step_samples_delta
            else 0.0
        ),
        engine_prefill_insert_batches=prefill_insert_batches_delta,
        engine_prefill_insert_requests=prefill_insert_requests_delta,
        engine_avg_prefill_insert_batch=(
            round(
                prefill_insert_requests_delta / prefill_insert_batches_delta,
                2,
            )
            if prefill_insert_batches_delta
            else 0.0
        ),
        engine_prefill_insert_batches_while_decode=prefill_insert_batches_while_decode_delta,
        engine_prefill_insert_requests_while_decode=prefill_insert_requests_while_decode_delta,
        engine_avg_prefill_insert_batch_while_decode=(
            round(
                prefill_insert_requests_while_decode_delta
                / prefill_insert_batches_while_decode_delta,
                2,
            )
            if prefill_insert_batches_while_decode_delta
            else 0.0
        ),
        engine_prefill_insert_single_batches=prefill_insert_single_batches_delta,
        engine_prefill_insert_single_batch_ratio=(
            round(
                prefill_insert_single_batches_delta / prefill_insert_batches_delta,
                3,
            )
            if prefill_insert_batches_delta
            else 0.0
        ),
        engine_homogeneous_routed_decode_session_builds=int(
            after.get("homogeneous_routed_decode_session_builds", 0)
            - before.get("homogeneous_routed_decode_session_builds", 0)
        ),
        engine_homogeneous_routed_decode_session_cache_hits=int(
            after.get("homogeneous_routed_decode_session_cache_hits", 0)
            - before.get("homogeneous_routed_decode_session_cache_hits", 0)
        ),
        engine_avg_homogeneous_routed_decode_session_build_ms=float(
            after.get("avg_homogeneous_routed_decode_session_build_ms", 0.0)
        ),
        engine_mixed_routed_decode_session_builds=int(
            after.get("mixed_routed_decode_session_builds", 0)
            - before.get("mixed_routed_decode_session_builds", 0)
        ),
        engine_mixed_routed_decode_session_cache_hits=int(
            after.get("mixed_routed_decode_session_cache_hits", 0)
            - before.get("mixed_routed_decode_session_cache_hits", 0)
        ),
        engine_avg_mixed_routed_decode_session_build_ms=float(
            after.get("avg_mixed_routed_decode_session_build_ms", 0.0)
        ),
        engine_avg_mixed_decode_migration_since_insert_ms=(
            round(mixed_migration_since_insert_ms_total_delta / mixed_migrated_delta, 2)
            if mixed_migrated_delta
            else 0.0
        ),
        engine_avg_mixed_decode_migration_since_first_token_ms=(
            round(
                mixed_migration_since_first_token_ms_total_delta / mixed_migrated_delta,
                2,
            )
            if mixed_migrated_delta
            else 0.0
        ),
        engine_avg_mixed_decode_tokens_before_shared=(
            round(
                mixed_migration_tokens_before_shared_delta / mixed_migrated_delta,
                2,
            )
            if mixed_migrated_delta
            else 0.0
        ),
        engine_delta_total_calls=delta_total_calls_delta,
        engine_delta_total_rows=delta_total_rows_delta,
        engine_delta_avg_rows=(
            round(delta_total_rows_delta / delta_total_calls_delta, 2)
            if delta_total_calls_delta
            else 0.0
        ),
        engine_delta_direct_calls=delta_direct_calls_delta,
        engine_delta_direct_rows=delta_direct_rows_delta,
        engine_delta_avg_direct_rows=(
            round(delta_direct_rows_delta / delta_direct_calls_delta, 2)
            if delta_direct_calls_delta
            else 0.0
        ),
        engine_delta_routed_calls=delta_routed_calls_delta,
        engine_delta_routed_rows=delta_routed_rows_delta,
        engine_delta_avg_routed_rows=(
            round(delta_routed_rows_delta / delta_routed_calls_delta, 2)
            if delta_routed_calls_delta
            else 0.0
        ),
        engine_delta_neutralized_direct_calls=delta_neutralized_direct_calls_delta,
        engine_delta_neutralized_direct_rows=delta_neutralized_direct_rows_delta,
        engine_delta_neutralized_routed_calls=delta_neutralized_routed_calls_delta,
        engine_delta_neutralized_routed_rows=delta_neutralized_routed_rows_delta,
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
    *,
    arrival_mode: str,
    arrival_rate: float | None,
    duration_s: float,
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
    print(
        "Routed decode strict: "
        f"{engine_metrics.get('routed_decode_reference_strict', False)}"
    )
    print(
        "Routed decode backend: "
        f"{engine_metrics.get('routed_decode_backend', 'reference')}"
    )
    print(
        "Neutralize LoRA delta: "
        f"{engine_metrics.get('neutralize_lora_delta', False)}"
    )
    if arrival_mode == "poisson":
        print(
            "Traffic mode: "
            f"poisson arrivals, rate={arrival_rate if arrival_rate is not None else 'auto'} req/s, "
            f"window={duration_s:.1f}s"
        )
    else:
        print("Traffic mode: burst waves")
    print("Scenarios:")
    for scenario in scenarios:
        suffix = ""
        if scenario.prompt_mode != "default":
            suffix = f" [{scenario.prompt_mode}]"
        print(f"  - {scenario.name}: {', '.join(scenario.models)}{suffix}")
    print()


def _print_results(results: list[ScenarioResult]):
    print(
        f"{'scenario':<18} {'conc':>4} {'reqs':>5} {'wall_s':>8} "
        f"{'p50_ms':>8} {'p95_ms':>8} {'req/s':>8} {'tok/s':>8} {'inflt':>7} {'errs':>5}"
    )
    print("-" * 88)
    for row in results:
        print(
            f"{row.scenario:<18} {row.concurrency:>4} {row.requests:>5} "
            f"{row.wall_s:>8.2f} {row.p50_ms:>8.1f} {row.p95_ms:>8.1f} "
            f"{row.req_s:>8.1f} {row.engine_tok_s:>8.1f} {row.client_avg_inflight:>7.1f} {row.errors:>5}"
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

        _print_header(
            args.base_url,
            health,
            adapters,
            scenarios,
            engine_metrics,
            arrival_mode=args.arrival_mode,
            arrival_rate=args.arrival_rate,
            duration_s=args.duration_s,
        )

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
                        arrival_mode=args.arrival_mode,
                        arrival_rate=args.arrival_rate,
                        duration_s=args.duration_s,
                        seed=args.seed,
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
