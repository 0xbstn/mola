from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "devtools" / "run_mola_winner.py"
SPEC = importlib.util.spec_from_file_location("run_mola_winner", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_args_defaults_to_winner_profile() -> None:
    args = MODULE.parse_args(["start"])
    assert args.max_inflight_tokens == 131072
    assert args.max_batch_size == 128
    assert args.prefill_batch_size == 32


def test_build_command_contains_winner_flags() -> None:
    args = MODULE.parse_args(["start", "--port", "8001"])
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
    assert log_path == Path("/tmp/mola-winner-8001.log")
