from unittest.mock import MagicMock, patch

from mola.infrastructure.mlx_generator import MLXBatchGeneratorPort
from mola.ports.generator import GeneratorHandle, GeneratorSubmission


@patch("mola.infrastructure.mlx_generator._load_batch_generator_cls")
def test_submit_batch_wraps_insert(mock_loader):
    batch_generator_cls = MagicMock()
    generator = batch_generator_cls.return_value
    mock_loader.return_value = batch_generator_cls
    generator.insert.return_value = [3, 4]
    port = MLXBatchGeneratorPort(
        MagicMock(),
        max_tokens=64,
        stop_tokens={0},
        completion_batch_size=8,
        prefill_batch_size=4,
    )

    handles = port.submit_batch(
        [
            GeneratorSubmission(prompt_tokens=[1], max_tokens=10, sampler=None),
            GeneratorSubmission(prompt_tokens=[2], max_tokens=12, sampler=None),
        ]
    )

    assert handles == [GeneratorHandle(uid=3), GeneratorHandle(uid=4)]
    generator.insert.assert_called_once()


@patch("mola.infrastructure.mlx_generator._load_batch_generator_cls")
def test_step_maps_batch_generator_responses(mock_loader):
    response = MagicMock(uid=7, token=42, finish_reason="stop")
    batch_generator_cls = MagicMock()
    generator = batch_generator_cls.return_value
    mock_loader.return_value = batch_generator_cls
    generator.next.return_value = [response]
    port = MLXBatchGeneratorPort(
        MagicMock(),
        max_tokens=64,
        stop_tokens={0},
        completion_batch_size=8,
        prefill_batch_size=4,
    )

    events = port.step()

    assert events[0].handle.uid == 7
    assert events[0].token == 42
    assert events[0].finish_reason == "stop"
