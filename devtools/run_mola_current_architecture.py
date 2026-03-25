#!/usr/bin/env python3
from __future__ import annotations

import argparse
import errno
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
PATCHER = ROOT / "devtools" / "apply_mlx_lm_detached_batch_api.py"
DEFAULT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
DEFAULT_ADAPTERS = [
    ("rust", ROOT / "adapters" / "rust-lora"),
    ("sql", ROOT / "adapters" / "sql-lora"),
    ("medical", ROOT / "adapters" / "medical-lora"),
    ("cyber", ROOT / "adapters" / "cyber-lora"),
    ("solidity", ROOT / "adapters" / "solidity-lora"),
    ("devops", ROOT / "adapters" / "devops-lora"),
    ("math", ROOT / "adapters" / "math-lora"),
    ("legal", ROOT / "adapters" / "legal-lora"),
]


def pid_list(port: int) -> list[int]:
    proc = subprocess.run(
        ["lsof", "-ti", f"tcp:{port}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return [int(line) for line in proc.stdout.splitlines() if line.strip()]


def wait_for_port_clear(port: int, timeout_s: float = 10.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not pid_list(port):
            return True
        time.sleep(0.5)
    return False


def wait_for_http(port: int, timeout_s: float = 15.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urlopen(f"http://127.0.0.1:{port}/v1/engine/metrics", timeout=2):
                return True
        except Exception:
            time.sleep(0.5)
    return False


def stop_server(port: int) -> int:
    pids = pid_list(port)
    if not pids:
        print(f"No server running on port {port}")
        return 0

    denied: list[int] = []
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as exc:
            if exc.errno == errno.EPERM:
                denied.append(pid)
    time.sleep(1.0)
    if pid_list(port):
        for pid in pid_list(port):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError as exc:
                if exc.errno == errno.EPERM and pid not in denied:
                    denied.append(pid)
    if denied:
        print(
            f"Permission denied while stopping port {port}; surviving pid(s): {denied}",
            file=sys.stderr,
        )
    if not wait_for_port_clear(port):
        print(f"Port {port} is still busy after stop", file=sys.stderr)
        return 1
    print(f"Stopped server on port {port}")
    return 0


def apply_patch_if_needed() -> int:
    proc = subprocess.run(
        [sys.executable, str(PATCHER)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.stdout.strip():
        print(proc.stdout.strip())
    if proc.returncode != 0:
        if proc.stderr.strip():
            print(proc.stderr.strip(), file=sys.stderr)
        return proc.returncode
    return 0


def build_command(args: argparse.Namespace) -> tuple[list[str], Path]:
    log_path = (
        Path(args.log)
        if args.log
        else Path(f"/tmp/mola-current-architecture-{args.port}.log")
    )
    cmd = [
        str(PYTHON),
        "-m",
        "mola.cli",
        "-v",
        "serve",
        "--model",
        args.model,
        "--port",
        str(args.port),
        "--max-inflight-tokens",
        str(args.max_inflight_tokens),
        "--max-batch-size",
        str(args.max_batch_size),
        "--prefill-batch-size",
        str(args.prefill_batch_size),
        "--enable-routed-decode-reference",
        "--strict-routed-decode-reference",
        "--routed-decode-backend",
        "metal-gather",
        "--enable-mixed-decode-migration",
        "--prestep-mixed-decode-migration",
        "--cache-routed-decode-sessions",
        "--detached-shared-decode-owner",
    ]
    for name, path in DEFAULT_ADAPTERS:
        cmd.extend(["--adapter", name, str(path)])
    return cmd, log_path


def start_server(args: argparse.Namespace) -> int:
    rc = apply_patch_if_needed()
    if rc != 0:
        return rc
    if pid_list(args.port):
        print(f"Server already running on port {args.port}: {pid_list(args.port)}", file=sys.stderr)
        return 1
    cmd, log_path = build_command(args)
    with log_path.open("wb") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    time.sleep(2.0)
    if proc.poll() is not None:
        print(
            f"Current architecture server failed to start; recent log from {log_path}:",
            file=sys.stderr,
        )
        try:
            print(log_path.read_text()[-8000:], file=sys.stderr)
        except Exception:
            pass
        return 1
    if not wait_for_http(args.port):
        print(
            f"Current architecture server did not become healthy on port {args.port}; see {log_path}",
            file=sys.stderr,
        )
        return 1
    print(f"Started MOLA current architecture on port {args.port}")
    print(f"PID: {proc.pid}")
    print(f"Log: {log_path}")
    return 0


def status_server(port: int) -> int:
    pids = pid_list(port)
    if not pids:
        print(f"No server running on port {port}")
        return 1
    print(f"Server running on port {port}: {pids}")
    return 0


def logs(path: str | None, port: int) -> int:
    log_path = (
        Path(path)
        if path
        else Path(f"/tmp/mola-current-architecture-{port}.log")
    )
    if not log_path.exists():
        print(f"Missing log file: {log_path}", file=sys.stderr)
        return 1
    print(log_path.read_text()[-8000:])
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "stop", "restart", "status", "logs"])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-inflight-tokens", type=int, default=131072)
    parser.add_argument("--max-batch-size", type=int, default=128)
    parser.add_argument("--prefill-batch-size", type=int, default=32)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if args.action == "start":
        return start_server(args)
    if args.action == "stop":
        return stop_server(args.port)
    if args.action == "restart":
        rc = stop_server(args.port)
        if rc != 0 and pid_list(args.port):
            return rc
        return start_server(args)
    if args.action == "status":
        return status_server(args.port)
    return logs(args.log, args.port)


if __name__ == "__main__":
    raise SystemExit(main())
