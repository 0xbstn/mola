from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mola.infrastructure import mlx_generator
from mola.ports.generator import GeneratorHandle, GeneratorState, GeneratorSubmission


class FakeBatchGenerator:
    def __init__(self, *args, **kwargs):
        self.insert_calls = []
        self.next_responses = []
        self.remove_calls = []
        self.closed = False
        self.active_batch = None
        self.uid_count = 0

    def insert(self, prompts, max_tokens, samplers=None):
        self.insert_calls.append((prompts, max_tokens, samplers))
        return [10 + i for i in range(len(prompts))]

    def next(self):
        return list(self.next_responses)

    def remove(self, uids):
        self.remove_calls.append(list(uids))

    def close(self):
        self.closed = True


class FakeActiveBatch:
    def __init__(
        self,
        *,
        uids,
        y,
        logprobs,
        max_tokens,
        num_tokens,
        cache,
        samplers,
        logits_processors,
        tokens,
    ):
        self.uids = list(uids)
        self.y = np.array(y, dtype=np.int32)
        self.logprobs = list(logprobs)
        self.max_tokens = list(max_tokens)
        self.num_tokens = list(num_tokens)
        self.cache = list(cache)
        self.samplers = list(samplers)
        self.logits_processors = list(logits_processors)
        self.tokens = list(tokens)

    def extract_cache(self, idx):
        return self.cache[idx]

    def extend(self, other):
        self.uids.extend(other.uids)
        self.y = np.concatenate([self.y, other.y])
        self.logprobs.extend(other.logprobs)
        self.max_tokens.extend(other.max_tokens)
        self.num_tokens.extend(other.num_tokens)
        self.cache.extend(other.cache)
        self.samplers.extend(other.samplers)
        self.logits_processors.extend(other.logits_processors)
        self.tokens.extend(other.tokens)


class TestMLXBatchGeneratorPort:
    def test_submit_batch_wraps_uids(self, monkeypatch):
        monkeypatch.setattr(mlx_generator, "_load_batch_generator_cls", lambda: FakeBatchGenerator)
        port = mlx_generator.MLXBatchGeneratorPort(
            object(),
            max_tokens=128,
            stop_tokens={0},
            completion_batch_size=8,
            prefill_batch_size=4,
        )

        handles = port.submit_batch(
            [
                GeneratorSubmission([1, 2], 10, None),
                GeneratorSubmission([3], 12, "sampler"),
            ]
        )

        assert handles == [GeneratorHandle(uid=10), GeneratorHandle(uid=11)]
        assert port._generator.insert_calls == [
            ([[1, 2], [3]], [10, 12], [None, "sampler"]),
        ]

    def test_active_handles_uses_batch_order(self, monkeypatch):
        monkeypatch.setattr(mlx_generator, "_load_batch_generator_cls", lambda: FakeBatchGenerator)
        port = mlx_generator.MLXBatchGeneratorPort(
            object(),
            max_tokens=128,
            stop_tokens={0},
            completion_batch_size=8,
            prefill_batch_size=4,
        )
        port._generator.active_batch = SimpleNamespace(uids=[7, 3, 9])

        assert port.active_handles() == (
            GeneratorHandle(uid=7),
            GeneratorHandle(uid=3),
            GeneratorHandle(uid=9),
        )

    def test_active_handles_empty_without_active_batch(self, monkeypatch):
        monkeypatch.setattr(mlx_generator, "_load_batch_generator_cls", lambda: FakeBatchGenerator)
        port = mlx_generator.MLXBatchGeneratorPort(
            object(),
            max_tokens=128,
            stop_tokens={0},
            completion_batch_size=8,
            prefill_batch_size=4,
        )

        assert port.active_handles() == ()

    def test_cancel_and_close_delegate(self, monkeypatch):
        monkeypatch.setattr(mlx_generator, "_load_batch_generator_cls", lambda: FakeBatchGenerator)
        port = mlx_generator.MLXBatchGeneratorPort(
            object(),
            max_tokens=128,
            stop_tokens={0},
            completion_batch_size=8,
            prefill_batch_size=4,
        )

        port.cancel([GeneratorHandle(uid=1), GeneratorHandle(uid=4)])
        port.close()

        assert port._generator.remove_calls == [[1, 4]]
        assert port._generator.closed is True

    def test_take_states_extracts_active_batch_entries(self, monkeypatch):
        monkeypatch.setattr(mlx_generator, "_load_batch_generator_cls", lambda: FakeBatchGenerator)
        port = mlx_generator.MLXBatchGeneratorPort(
            object(),
            max_tokens=128,
            stop_tokens={0},
            completion_batch_size=8,
            prefill_batch_size=4,
        )
        port._generator.active_batch = FakeActiveBatch(
            uids=[7, 9],
            y=[101, 102],
            logprobs=["lp-7", "lp-9"],
            max_tokens=[32, 64],
            num_tokens=[3, 5],
            cache=[["cache-7"], ["cache-9"]],
            samplers=["sampler-7", "sampler-9"],
            logits_processors=[["lp7"], ["lp9"]],
            tokens=[np.array([1, 2, 3]), np.array([4, 5])],
        )

        states = port.take_states([GeneratorHandle(uid=9)])

        assert len(states) == 1
        assert states[0].handle == GeneratorHandle(uid=9)
        assert states[0].next_token == 102
        assert states[0].logprobs == "lp-9"
        assert states[0].max_tokens == 64
        assert states[0].num_tokens == 5
        assert states[0].cache == ["cache-9"]
        assert states[0].sampler == "sampler-9"
        assert states[0].logits_processors == ["lp9"]
        assert np.array_equal(states[0].tokens, np.array([4, 5]))
        assert port._generator.remove_calls == [[9]]

    def test_restore_states_sets_active_batch(self, monkeypatch):
        monkeypatch.setattr(mlx_generator, "_load_batch_generator_cls", lambda: FakeBatchGenerator)
        monkeypatch.setattr(
            mlx_generator,
            "_load_batch_state_symbols",
            lambda: (FakeActiveBatch, lambda caches: [list(cache) for cache in caches]),
        )
        port = mlx_generator.MLXBatchGeneratorPort(
            object(),
            max_tokens=128,
            stop_tokens={0},
            completion_batch_size=8,
            prefill_batch_size=4,
        )

        handles = port.restore_states(
            [
                GeneratorState(
                    handle=GeneratorHandle(uid=7),
                    next_token=101,
                    logprobs="lp-7",
                    max_tokens=32,
                    num_tokens=3,
                    cache=["cache-7"],
                    sampler="sampler-7",
                    logits_processors=["lp7"],
                    tokens=np.array([1, 2, 3], dtype=np.int32),
                ),
                GeneratorState(
                    handle=GeneratorHandle(uid=9),
                    next_token=102,
                    logprobs="lp-9",
                    max_tokens=64,
                    num_tokens=5,
                    cache=["cache-9"],
                    sampler="sampler-9",
                    logits_processors=["lp9"],
                    tokens=np.array([4, 5], dtype=np.int32),
                ),
            ]
        )

        assert handles == [GeneratorHandle(uid=0), GeneratorHandle(uid=1)]
        assert port._generator.active_batch.uids == [0, 1]
        assert port._generator.active_batch.y.tolist() == [101, 102]
        assert port._generator.active_batch.max_tokens == [32, 64]
        assert port._generator.uid_count == 2

    def test_restore_states_assigns_fresh_uids_when_active_batch_exists(self, monkeypatch):
        monkeypatch.setattr(mlx_generator, "_load_batch_generator_cls", lambda: FakeBatchGenerator)
        monkeypatch.setattr(
            mlx_generator,
            "_load_batch_state_symbols",
            lambda: (FakeActiveBatch, lambda caches: [list(cache) for cache in caches]),
        )
        port = mlx_generator.MLXBatchGeneratorPort(
            object(),
            max_tokens=128,
            stop_tokens={0},
            completion_batch_size=8,
            prefill_batch_size=4,
        )
        port._generator.active_batch = FakeActiveBatch(
            uids=[7],
            y=[101],
            logprobs=["lp-7"],
            max_tokens=[32],
            num_tokens=[3],
            cache=[["cache-7"]],
            samplers=["sampler-7"],
            logits_processors=[["lp7"]],
            tokens=[np.array([1, 2, 3], dtype=np.int32)],
        )

        handles = port.restore_states(
            [
                GeneratorState(
                    handle=GeneratorHandle(uid=7),
                    next_token=102,
                    logprobs="lp-7-new",
                    max_tokens=64,
                    num_tokens=5,
                    cache=["cache-7-new"],
                    sampler="sampler-7-new",
                    logits_processors=["lp7-new"],
                    tokens=np.array([4, 5], dtype=np.int32),
                )
            ]
        )

        assert handles == [GeneratorHandle(uid=8)]
        assert port._generator.active_batch.uids == [7, 8]
        assert port._generator.uid_count == 9
