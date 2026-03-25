from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "devtools"
    / "run_mola_current_architecture.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_mola_current_architecture", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_args_defaults_to_current_architecture_profile() -> None:
    args = MODULE.parse_args(["start"])
    assert args.max_inflight_tokens == 131072
    assert args.max_batch_size == 128
    assert args.prefill_batch_size == 32
    assert args.adapter is None
    assert args.use_local_benchmark_adapters is False


def test_build_command_contains_current_architecture_flags() -> None:
    args = MODULE.parse_args(
        [
            "start",
            "--port",
            "8001",
            "--adapter",
            "rust",
            "./adapters/rust-lora",
            "--adapter",
            "sql",
            "./adapters/sql-lora",
        ]
    )
    cmd, log_path = MODULE.build_command(args)
    assert cmd[:4] == [str(MODULE.PYTHON), "-m", "mola.cli", "-v"]
    assert "--enable-routed-decode-reference" in cmd
    assert "--strict-routed-decode-reference" in cmd
    assert "--routed-decode-backend" in cmd
    assert "--enable-mixed-decode-migration" in cmd
    assert "--prestep-mixed-decode-migration" in cmd
    assert "--cache-routed-decode-sessions" in cmd
    assert "--detached-shared-decode-owner" in cmd
    assert "--max-inflight-tokens" in cmd
    assert "--max-batch-size" in cmd
    assert "--prefill-batch-size" in cmd
    assert cmd.count("--adapter") == 2
    assert "rust" in cmd
    assert "sql" in cmd
    assert log_path == Path("/tmp/mola-current-architecture-8001.log")


def test_build_command_requires_explicit_adapters_by_default() -> None:
    args = MODULE.parse_args(["start", "--port", "8001"])
    try:
        MODULE.build_command(args)
    except ValueError as exc:
        assert "No adapters configured" in str(exc)
    else:
        raise AssertionError("expected build_command to reject missing adapters")


def test_build_command_can_use_local_benchmark_adapters() -> None:
    args = MODULE.parse_args(["start", "--use-local-benchmark-adapters"])
    cmd, _log_path = MODULE.build_command(args)
    assert cmd.count("--adapter") == len(MODULE.LOCAL_BENCHMARK_ADAPTERS)


def test_restart_without_adapters_fails_before_stop(monkeypatch) -> None:
    args = argparse.Namespace(
        action="restart",
        port=8000,
        log=None,
        model=MODULE.DEFAULT_MODEL,
        adapter=None,
        use_local_benchmark_adapters=False,
        max_inflight_tokens=131072,
        max_batch_size=128,
        prefill_batch_size=32,
    )
    stop_calls: list[int] = []
    monkeypatch.setattr(MODULE, "parse_args", lambda: args)
    monkeypatch.setattr(MODULE, "stop_server", lambda port: stop_calls.append(port) or 0)

    rc = MODULE.main()

    assert rc == 2
    assert stop_calls == []
