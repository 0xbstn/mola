from __future__ import annotations

from types import SimpleNamespace

from mola.infrastructure import mlx_generator
from mola.ports.generator import GeneratorHandle, GeneratorSubmission


class FakeBatchGenerator:
    def __init__(self, *args, **kwargs):
        self.insert_calls = []
        self.next_responses = []
        self.remove_calls = []
        self.closed = False
        self.active_batch = None

    def insert(self, prompts, max_tokens, samplers=None):
        self.insert_calls.append((prompts, max_tokens, samplers))
        return [10 + i for i in range(len(prompts))]

    def next(self):
        return list(self.next_responses)

    def remove(self, uids):
        self.remove_calls.append(list(uids))

    def close(self):
        self.closed = True


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
