from __future__ import annotations

import threading
import time
import tracemalloc
from typing import Any, Callable, Dict, Iterable, Optional, TypeVar

import pandas as pd
import psutil

T = TypeVar("T")
_SAMPLE_INTERVAL_SECONDS = 0.02


def dataframe_size_bytes(dataframe: Optional[pd.DataFrame]) -> int:
    if dataframe is None:
        return 0
    return int(dataframe.memory_usage(index=True, deep=True).sum())


def system_resource_snapshot() -> Dict[str, Any]:
    virtual_memory = psutil.virtual_memory()
    disk = psutil.disk_io_counters()
    network = psutil.net_io_counters()
    return {
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "system_cpu_percent_since_previous_sample": psutil.cpu_percent(interval=None),
        "memory_total_bytes": int(virtual_memory.total),
        "memory_available_bytes": int(virtual_memory.available),
        "memory_percent_used": float(virtual_memory.percent),
        "disk_io_counters_system_wide": (
            {
                "read_bytes": int(disk.read_bytes),
                "write_bytes": int(disk.write_bytes),
            }
            if disk is not None
            else None
        ),
        "network_io_counters_system_wide": (
            {
                "bytes_sent": int(network.bytes_sent),
                "bytes_recv": int(network.bytes_recv),
            }
            if network is not None
            else None
        ),
        "gpu_utilization_percent": None,
        "gpu_note": "GPU utilization is not measured by this profiler.",
    }


def profile_callable(
    name: str,
    stage: str,
    operation: Callable[[], T],
    *,
    dataframe_in: Optional[pd.DataFrame] = None,
    records_in: Optional[int] = None,
    bytes_in: Optional[int] = None,
) -> tuple[T, Dict[str, Any]]:
    process = psutil.Process()
    rss_before = int(process.memory_info().rss)
    input_bytes = dataframe_size_bytes(dataframe_in) if dataframe_in is not None else (bytes_in or 0)
    records_in = len(dataframe_in) if dataframe_in is not None else records_in
    trace_started_here = not tracemalloc.is_tracing()
    if trace_started_here:
        tracemalloc.start()
    trace_start_bytes = tracemalloc.get_traced_memory()[0] if trace_started_here else None

    sample_stop = threading.Event()
    sampled_peak_rss = [rss_before]
    sample_errors: list[str] = []

    def sample_rss() -> None:
        while not sample_stop.wait(_SAMPLE_INTERVAL_SECONDS):
            try:
                sampled_peak_rss[0] = max(sampled_peak_rss[0], int(process.memory_info().rss))
            except psutil.Error as exc:
                sample_errors.append(str(exc))
                return

    sampler = threading.Thread(target=sample_rss, name="etl-rss-profiler", daemon=True)
    sampler.start()
    start_time = time.perf_counter()
    process_cpu_start = sum(process.cpu_times()[:2])
    error: Optional[BaseException] = None
    result: Optional[T] = None
    try:
        result = operation()
    except BaseException as exc:
        error = exc
    finally:
        elapsed_seconds = time.perf_counter() - start_time
        process_cpu_seconds = max(0.0, sum(process.cpu_times()[:2]) - process_cpu_start)
        try:
            rss_after = int(process.memory_info().rss)
            sampled_peak_rss[0] = max(sampled_peak_rss[0], rss_after)
        except psutil.Error as exc:
            rss_after = None
            sample_errors.append(str(exc))
        sample_stop.set()
        sampler.join(timeout=_SAMPLE_INTERVAL_SECONDS * 2)

    dataframe_out = result if isinstance(result, pd.DataFrame) else None
    output_bytes = dataframe_size_bytes(dataframe_out)
    records_out = len(dataframe_out) if dataframe_out is not None else None
    python_peak_bytes = None
    if trace_started_here:
        _, peak_bytes = tracemalloc.get_traced_memory()
        python_peak_bytes = max(0, int(peak_bytes - (trace_start_bytes or 0)))
        tracemalloc.stop()

    disk_after = psutil.disk_io_counters()
    network_after = psutil.net_io_counters()
    # System-wide counters are sampled after the operation; per-operation deltas
    # are intentionally omitted because counters are not attributed to this process.
    metrics: Dict[str, Any] = {
        "name": name,
        "stage": stage,
        "status": "failed" if error is not None else "completed",
        "duration_seconds": elapsed_seconds,
        "process_cpu_seconds": process_cpu_seconds,
        "process_cpu_percent_of_one_core": (
            process_cpu_seconds / elapsed_seconds * 100 if elapsed_seconds > 0 else 0.0
        ),
        "rss_bytes_before": rss_before,
        "rss_bytes_after": rss_after,
        "sampled_peak_rss_bytes": sampled_peak_rss[0],
        "rss_sampling_interval_seconds": _SAMPLE_INTERVAL_SECONDS,
        "python_peak_allocation_bytes": python_peak_bytes,
        "python_peak_allocation_note": (
            "Python allocations only; native NumPy/Pandas buffers may not be fully tracked."
            if python_peak_bytes is not None
            else "Not isolated because another tracemalloc session was already active."
        ),
        "dataframe_input_bytes": input_bytes,
        "dataframe_output_bytes": output_bytes if dataframe_out is not None else None,
        "records_in": records_in,
        "records_out": records_out,
        "records_per_second": (
            records_out / elapsed_seconds
            if records_out is not None and elapsed_seconds > 0
            else None
        ),
        "input_bytes_per_second": (
            input_bytes / elapsed_seconds if input_bytes and elapsed_seconds > 0 else None
        ),
        "disk_io_counters_system_wide_after": (
            {
                "read_bytes": int(disk_after.read_bytes),
                "write_bytes": int(disk_after.write_bytes),
            }
            if disk_after is not None
            else None
        ),
        "network_io_counters_system_wide_after": (
            {
                "bytes_sent": int(network_after.bytes_sent),
                "bytes_recv": int(network_after.bytes_recv),
            }
            if network_after is not None
            else None
        ),
        "system_io_note": "Disk and network counters are whole-system snapshots, not process-attributed deltas.",
        "measurement_note": "Elapsed time includes the operation but excludes profiler setup and post-operation summaries.",
    }
    if sample_errors:
        metrics["resource_sampling_errors"] = sample_errors

    if error is not None:
        raise error
    return result, metrics  # type: ignore[return-value]


def identify_bottlenecks(
    metrics: Iterable[Dict[str, Any]],
    *,
    share_threshold: float = 0.25,
) -> list[Dict[str, Any]]:
    measured = [item for item in metrics if item.get("status") == "completed"]
    total_seconds = sum(float(item.get("duration_seconds") or 0.0) for item in measured)
    if len(measured) < 2 or total_seconds <= 0:
        return []

    bottlenecks = []
    for item in measured:
        duration = float(item.get("duration_seconds") or 0.0)
        share = duration / total_seconds
        if share >= share_threshold:
            bottlenecks.append(
                {
                    "name": item.get("name", "unnamed"),
                    "stage": item.get("stage", "unknown"),
                    "duration_seconds": duration,
                    "share_of_measured_time": share,
                    "evidence": "Measured elapsed time share across the supplied completed operations.",
                }
            )
    return sorted(bottlenecks, key=lambda item: item["duration_seconds"], reverse=True)
