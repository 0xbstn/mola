from __future__ import annotations

from dataclasses import dataclass
import threading


@dataclass
class DeltaRuntimeMetrics:
    direct_calls: int = 0
    direct_rows: int = 0
    routed_calls: int = 0
    routed_rows: int = 0
    neutralized_direct_calls: int = 0
    neutralized_direct_rows: int = 0
    neutralized_routed_calls: int = 0
    neutralized_routed_rows: int = 0


_delta_metrics = DeltaRuntimeMetrics()
_delta_metrics_lock = threading.Lock()


def record_delta_invocation(
    kind: str,
    rows: int,
    *,
    neutralized: bool,
) -> None:
    rows = max(int(rows), 0)
    with _delta_metrics_lock:
        if kind == "direct":
            _delta_metrics.direct_calls += 1
            _delta_metrics.direct_rows += rows
            if neutralized:
                _delta_metrics.neutralized_direct_calls += 1
                _delta_metrics.neutralized_direct_rows += rows
            return
        if kind == "routed":
            _delta_metrics.routed_calls += 1
            _delta_metrics.routed_rows += rows
            if neutralized:
                _delta_metrics.neutralized_routed_calls += 1
                _delta_metrics.neutralized_routed_rows += rows
            return
        raise ValueError(f"unsupported delta invocation kind: {kind}")


def snapshot_delta_runtime_metrics() -> dict[str, int | float]:
    with _delta_metrics_lock:
        direct_calls = _delta_metrics.direct_calls
        direct_rows = _delta_metrics.direct_rows
        routed_calls = _delta_metrics.routed_calls
        routed_rows = _delta_metrics.routed_rows
        neutralized_direct_calls = _delta_metrics.neutralized_direct_calls
        neutralized_direct_rows = _delta_metrics.neutralized_direct_rows
        neutralized_routed_calls = _delta_metrics.neutralized_routed_calls
        neutralized_routed_rows = _delta_metrics.neutralized_routed_rows
    total_calls = direct_calls + routed_calls
    total_rows = direct_rows + routed_rows
    return {
        "delta_direct_calls": direct_calls,
        "delta_direct_rows": direct_rows,
        "delta_routed_calls": routed_calls,
        "delta_routed_rows": routed_rows,
        "delta_total_calls": total_calls,
        "delta_total_rows": total_rows,
        "delta_neutralized_direct_calls": neutralized_direct_calls,
        "delta_neutralized_direct_rows": neutralized_direct_rows,
        "delta_neutralized_routed_calls": neutralized_routed_calls,
        "delta_neutralized_routed_rows": neutralized_routed_rows,
        "delta_avg_direct_rows": round(direct_rows / direct_calls, 2) if direct_calls else 0.0,
        "delta_avg_routed_rows": round(routed_rows / routed_calls, 2) if routed_calls else 0.0,
        "delta_avg_rows": round(total_rows / total_calls, 2) if total_calls else 0.0,
    }


def reset_delta_runtime_metrics() -> None:
    with _delta_metrics_lock:
        _delta_metrics.direct_calls = 0
        _delta_metrics.direct_rows = 0
        _delta_metrics.routed_calls = 0
        _delta_metrics.routed_rows = 0
        _delta_metrics.neutralized_direct_calls = 0
        _delta_metrics.neutralized_direct_rows = 0
        _delta_metrics.neutralized_routed_calls = 0
        _delta_metrics.neutralized_routed_rows = 0
