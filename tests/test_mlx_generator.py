from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mola.infrastructure import mlx_generator
from mola.ports.generator import (
    GeneratorDetachedBatch,
    GeneratorDetachedBatchStepResult,
    GeneratorBatchSnapshot,
    GeneratorBatchStepResult,
    GeneratorHandle,
    GeneratorState,
    GeneratorSubmission,
)


class FakeBatchGenerator:
    def __init__(self, *args, **kwargs):
        self.insert_calls = []
        self.next_responses = []
        self.remove_calls = []
        self.closed = False
        self.active_batch = None
        self.uid_count = 0
        self.stop_tokens = kwargs.get("stop_tokens", set())

    def insert(self, prompts, max_tokens, samplers=None):
        self.insert_calls.append((prompts, max_tokens, samplers))
        return [10 + i for i in range(len(prompts))]

    def next(self):
        return list(self.next_responses)

    def _step(self, input_tokens, prompt_cache, samplers, logits_processors, tokens):
        batch_size = input_tokens.shape[0]
        sampled = mx.array([200 + i for i in range(batch_size)], dtype=mx.int32)
        logprobs = [f"logprobs-{i}" for i in range(batch_size)]
        return sampled, logprobs

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

    def filter(self, keep_idx):
        self.uids = [self.uids[k] for k in keep_idx]
        self.logprobs = [self.logprobs[k] for k in keep_idx]
        self.max_tokens = [self.max_tokens[k] for k in keep_idx]
        self.num_tokens = [self.num_tokens[k] for k in keep_idx]
        self.cache = [self.cache[k] for k in keep_idx]
        self.samplers = [self.samplers[k] for k in keep_idx]
        self.logits_processors = [self.logits_processors[k] for k in keep_idx]
        self.tokens = [self.tokens[k] for k in keep_idx]
        keep_idx_np = np.array(keep_idx, dtype=np.int32)
        if isinstance(self.y, np.ndarray):
            self.y = self.y[keep_idx_np]
        else:
            self.y = self.y[mx.array(keep_idx_np, dtype=mx.int32)]

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

    def test_take_active_batch_returns_decode_ordered_snapshot(self, monkeypatch):
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

        batch = port.take_active_batch()

        assert isinstance(batch, GeneratorBatchSnapshot)
        assert tuple(state.handle.uid for state in batch.states) == (7, 9)
        assert tuple(state.next_token for state in batch.states) == (101, 102)
        assert tuple(state.logprobs for state in batch.states) == ("lp-7", "lp-9")
        assert tuple(state.max_tokens for state in batch.states) == (32, 64)
        assert tuple(state.num_tokens for state in batch.states) == (3, 5)
        assert batch.states[0].cache == ["cache-7"]
        assert batch.states[1].cache == ["cache-9"]
        assert batch.states[0].sampler == "sampler-7"
        assert batch.states[1].sampler == "sampler-9"
        assert batch.states[0].logits_processors == ["lp7"]
        assert batch.states[1].logits_processors == ["lp9"]
        assert np.array_equal(batch.states[0].tokens, np.array([1, 2, 3]))
        assert np.array_equal(batch.states[1].tokens, np.array([4, 5]))
        assert port._generator.remove_calls == [[7, 9]]

    def test_take_active_batch_returns_none_without_active_batch(self, monkeypatch):
        monkeypatch.setattr(mlx_generator, "_load_batch_generator_cls", lambda: FakeBatchGenerator)
        port = mlx_generator.MLXBatchGeneratorPort(
            object(),
            max_tokens=128,
            stop_tokens={0},
            completion_batch_size=8,
            prefill_batch_size=4,
        )

        assert port.take_active_batch() is None

    def test_restore_active_batch_restores_snapshot_rows(self, monkeypatch):
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
        batch = GeneratorBatchSnapshot(
            states=(
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
            )
        )

        handles = port.restore_active_batch(batch)

        assert handles == (GeneratorHandle(uid=0), GeneratorHandle(uid=1))
        assert port._generator.active_batch.uids == [0, 1]
        assert port._generator.active_batch.y.tolist() == [101, 102]

    def test_take_active_batch_handle_detaches_live_batch(self, monkeypatch):
        monkeypatch.setattr(mlx_generator, "_load_batch_generator_cls", lambda: FakeBatchGenerator)
        port = mlx_generator.MLXBatchGeneratorPort(
            object(),
            max_tokens=128,
            stop_tokens={0},
            completion_batch_size=8,
            prefill_batch_size=4,
        )
        active_batch = FakeActiveBatch(
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
        port._generator.active_batch = active_batch

        batch = port.take_active_batch_handle()

        assert isinstance(batch, GeneratorDetachedBatch)
        assert batch.handles == (GeneratorHandle(uid=7), GeneratorHandle(uid=9))
        assert batch.opaque is active_batch
        assert port._generator.active_batch is None

    def test_restore_detached_batch_reinstalls_live_batch(self, monkeypatch):
        monkeypatch.setattr(mlx_generator, "_load_batch_generator_cls", lambda: FakeBatchGenerator)
        port = mlx_generator.MLXBatchGeneratorPort(
            object(),
            max_tokens=128,
            stop_tokens={0},
            completion_batch_size=8,
            prefill_batch_size=4,
        )
        active_batch = FakeActiveBatch(
            uids=[7],
            y=[101],
            logprobs=["lp-7"],
            max_tokens=[32],
            num_tokens=[3],
            cache=[["cache-7"]],
            samplers=["sampler-7"],
            logits_processors=[["lp7"]],
            tokens=[np.array([1, 2, 3])],
        )
        batch = GeneratorDetachedBatch(
            handles=(GeneratorHandle(uid=7),),
            opaque=active_batch,
        )

        port.restore_detached_batch(batch)

        assert port._generator.active_batch is active_batch

    def test_snapshot_detached_batch_materializes_generator_state_view(self, monkeypatch):
        monkeypatch.setattr(mlx_generator, "_load_batch_generator_cls", lambda: FakeBatchGenerator)
        port = mlx_generator.MLXBatchGeneratorPort(
            object(),
            max_tokens=128,
            stop_tokens={0},
            completion_batch_size=8,
            prefill_batch_size=4,
        )
        batch = GeneratorDetachedBatch(
            handles=(GeneratorHandle(uid=7), GeneratorHandle(uid=9)),
            opaque=FakeActiveBatch(
                uids=[7, 9],
                y=[101, 102],
                logprobs=["lp-7", "lp-9"],
                max_tokens=[32, 64],
                num_tokens=[3, 5],
                cache=[["cache-7"], ["cache-9"]],
                samplers=["sampler-7", "sampler-9"],
                logits_processors=[["lp7"], ["lp9"]],
                tokens=[np.array([1, 2, 3]), np.array([4, 5])],
            ),
        )

        snapshot = port.snapshot_detached_batch(batch)

        assert tuple(state.handle.uid for state in snapshot.states) == (7, 9)
        assert tuple(state.next_token for state in snapshot.states) == (101, 102)

    def test_extend_detached_batch_appends_rows_without_active_batch_roundtrip(self, monkeypatch):
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
        existing = GeneratorDetachedBatch(
            handles=(GeneratorHandle(uid=99),),
            opaque=FakeActiveBatch(
                uids=[99],
                y=[111],
                logprobs=["lp-99"],
                max_tokens=[32],
                num_tokens=[3],
                cache=[["cache-99"]],
                samplers=["sampler-99"],
                logits_processors=[["lp99"]],
                tokens=[np.array([9, 9, 9], dtype=np.int32)],
            ),
        )
        incoming = GeneratorBatchSnapshot(
            states=(
                GeneratorState(
                    handle=GeneratorHandle(uid=7),
                    next_token=101,
                    logprobs="lp-7",
                    max_tokens=16,
                    num_tokens=1,
                    cache=["cache-7"],
                    sampler="sampler-7",
                    logits_processors=["lp7"],
                    tokens=np.array([1, 2, 3], dtype=np.int32),
                ),
            )
        )

        merged, restored_handles = port.extend_detached_batch(existing, incoming)

        assert restored_handles == (GeneratorHandle(uid=100),)
        assert merged.handles == (GeneratorHandle(uid=99), GeneratorHandle(uid=100))
        assert merged.opaque.uids == [99, 100]
        assert merged.opaque.y.tolist() == [111, 101]
        assert port._generator.active_batch is None

    def test_promote_detached_batch_rekeys_incoming_live_batch_and_merges_in_place(self, monkeypatch):
        monkeypatch.setattr(mlx_generator, "_load_batch_generator_cls", lambda: FakeBatchGenerator)
        port = mlx_generator.MLXBatchGeneratorPort(
            object(),
            max_tokens=128,
            stop_tokens={0},
            completion_batch_size=8,
            prefill_batch_size=4,
        )
        existing = GeneratorDetachedBatch(
            handles=(GeneratorHandle(uid=99),),
            opaque=FakeActiveBatch(
                uids=[99],
                y=[111],
                logprobs=["lp-99"],
                max_tokens=[32],
                num_tokens=[3],
                cache=[["cache-99"]],
                samplers=["sampler-99"],
                logits_processors=[["lp99"]],
                tokens=[np.array([9, 9, 9], dtype=np.int32)],
            ),
        )
        incoming_batch = FakeActiveBatch(
            uids=[7],
            y=[101],
            logprobs=["lp-7"],
            max_tokens=[16],
            num_tokens=[1],
            cache=[["cache-7"]],
            samplers=["sampler-7"],
            logits_processors=[["lp7"]],
            tokens=[np.array([1, 2, 3], dtype=np.int32)],
        )
        incoming = GeneratorDetachedBatch(
            handles=(GeneratorHandle(uid=7),),
            opaque=incoming_batch,
        )

        merged, promoted_handles = port.promote_detached_batch(existing, incoming)

        assert promoted_handles == (GeneratorHandle(uid=100),)
        assert merged.handles == (GeneratorHandle(uid=99), GeneratorHandle(uid=100))
        assert incoming_batch.uids == [100]
        assert merged.opaque.uids == [99, 100]
        assert merged.opaque.y.tolist() == [111, 101]
        assert port._generator.active_batch is None

    def test_step_batch_steps_detached_batch_without_touching_active_batch(self, monkeypatch):
        monkeypatch.setattr(mlx_generator, "_load_batch_generator_cls", lambda: FakeBatchGenerator)
        monkeypatch.setattr(
            mlx_generator,
            "_load_batch_state_symbols",
            lambda: (FakeActiveBatch, lambda caches: [list(cache) for cache in caches]),
        )
        port = mlx_generator.MLXBatchGeneratorPort(
            object(),
            max_tokens=128,
            stop_tokens={999},
            completion_batch_size=8,
            prefill_batch_size=4,
        )
        batch = GeneratorBatchSnapshot(
            states=(
                GeneratorState(
                    handle=GeneratorHandle(uid=7),
                    next_token=101,
                    logprobs="lp-7",
                    max_tokens=4,
                    num_tokens=1,
                    cache=["cache-7"],
                    sampler="sampler-7",
                    logits_processors=["lp7"],
                    tokens=mx.array([1, 2, 3], dtype=mx.int32),
                ),
                GeneratorState(
                    handle=GeneratorHandle(uid=9),
                    next_token=102,
                    logprobs="lp-9",
                    max_tokens=2,
                    num_tokens=1,
                    cache=["cache-9"],
                    sampler="sampler-9",
                    logits_processors=["lp9"],
                    tokens=mx.array([4, 5], dtype=mx.int32),
                ),
            )
        )

        result = port.step_batch(batch)

        assert isinstance(result, GeneratorBatchStepResult)
        assert result.events == (
            mlx_generator.GenerationEvent(
                handle=GeneratorHandle(uid=7),
                token=101,
                finish_reason=None,
            ),
            mlx_generator.GenerationEvent(
                handle=GeneratorHandle(uid=9),
                token=102,
                finish_reason="length",
            ),
        )
        assert result.batch is not None
        assert tuple(state.handle.uid for state in result.batch.states) == (7,)
        assert result.batch.states[0].next_token == 200
        assert result.batch.states[0].logprobs == "logprobs-0"
        assert result.batch.states[0].max_tokens == 4
        assert result.batch.states[0].num_tokens == 2
        assert result.batch.states[0].cache == ["cache-7"]
        assert result.batch.states[0].sampler == "sampler-7"
        assert result.batch.states[0].logits_processors == ["lp7"]
        assert mx.array_equal(
            result.batch.states[0].tokens,
            mx.array([1, 2, 3, 101], dtype=mx.int32),
        )
        assert port._generator.active_batch is None

    def test_step_detached_batch_steps_live_batch_without_rebuild(self, monkeypatch):
        monkeypatch.setattr(mlx_generator, "_load_batch_generator_cls", lambda: FakeBatchGenerator)
        port = mlx_generator.MLXBatchGeneratorPort(
            object(),
            max_tokens=128,
            stop_tokens={999},
            completion_batch_size=8,
            prefill_batch_size=4,
        )
        active_batch = FakeActiveBatch(
            uids=[7, 9],
            y=[101, 102],
            logprobs=["lp-7", "lp-9"],
            max_tokens=[4, 2],
            num_tokens=[1, 1],
            cache=[["cache-7"], ["cache-9"]],
            samplers=["sampler-7", "sampler-9"],
            logits_processors=[["lp7"], ["lp9"]],
            tokens=[
                mx.array([1, 2, 3], dtype=mx.int32),
                mx.array([4, 5], dtype=mx.int32),
            ],
        )
        batch = GeneratorDetachedBatch(
            handles=(GeneratorHandle(uid=7), GeneratorHandle(uid=9)),
            opaque=active_batch,
        )

        result = port.step_detached_batch(batch)

        assert isinstance(result, GeneratorDetachedBatchStepResult)
        assert result.events == (
            mlx_generator.GenerationEvent(
                handle=GeneratorHandle(uid=7),
                token=101,
                finish_reason=None,
            ),
            mlx_generator.GenerationEvent(
                handle=GeneratorHandle(uid=9),
                token=102,
                finish_reason="length",
            ),
        )
        assert result.batch is not None
        assert result.batch.handles == (GeneratorHandle(uid=7),)
        assert result.batch.opaque is active_batch
        assert result.batch.opaque.uids == [7]
        if isinstance(result.batch.opaque.y, np.ndarray):
            assert result.batch.opaque.y.tolist() == [200]
        else:
            assert result.batch.opaque.y.tolist() == [200]
        assert result.batch.opaque.logprobs == ["logprobs-0"]
        assert result.batch.opaque.max_tokens == [4]
        assert result.batch.opaque.num_tokens == [2]
        assert result.batch.opaque.cache == [["cache-7"]]
        assert result.batch.opaque.samplers == ["sampler-7"]
        assert result.batch.opaque.logits_processors == [["lp7"]]
        assert mx.array_equal(
            result.batch.opaque.tokens[0],
            mx.array([1, 2, 3, 101], dtype=mx.int32),
        )
        assert port._generator.active_batch is None
