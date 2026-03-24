"""Test concurrent requests through the MOLA engine."""

import asyncio
import time

import httpx

BASE = "http://localhost:8000/v1/chat/completions"


async def query(client, model, prompt, max_tokens=60):
    t0 = time.time()
    resp = await client.post(
        BASE,
        json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens},
    )
    elapsed = time.time() - t0
    data = resp.json()
    text = data["choices"][0]["message"]["content"][:120]
    return model, elapsed, text


async def main():
    async with httpx.AsyncClient(timeout=60) as client:
        # Sequential baseline
        print("=== SEQUENTIAL (one at a time) ===")
        t0 = time.time()
        for model, prompt in [
            ("rust", "Write a Rust struct for a stack"),
            ("rust", "Implement binary search in Rust"),
            ("sql", "Given: CREATE TABLE t (id INT, val INT)\nFind max val"),
        ]:
            _, elapsed, text = await query(client, model, prompt)
            print(f"  [{model}] {elapsed:.2f}s — {text}")
        seq_total = time.time() - t0
        print(f"  Total: {seq_total:.2f}s\n")

        # Concurrent
        print("=== CONCURRENT (all at once) ===")
        t0 = time.time()
        results = await asyncio.gather(
            query(client, "rust", "Write a Rust struct for a stack"),
            query(client, "rust", "Implement binary search in Rust"),
            query(client, "sql", "Given: CREATE TABLE t (id INT, val INT)\nFind max val"),
        )
        conc_total = time.time() - t0

        for model, elapsed, text in results:
            print(f"  [{model}] {elapsed:.2f}s — {text}")
        print(f"  Total: {conc_total:.2f}s\n")

        print(f"Speedup: {seq_total / conc_total:.2f}x")

        # Metrics
        resp = await client.get("http://localhost:8000/v1/engine/metrics")
        print(f"\nEngine metrics: {resp.json()}")


if __name__ == "__main__":
    asyncio.run(main())
