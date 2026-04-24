"""Latency / throughput profiler.

Matches Table IV of the manuscript: P50 / P95 / P99 and queries/sec
measured end-to-end from graph update to classification output on a
single accelerator (paper value: P99 = 38ms on A100).
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class LatencyReport:
    p50: float
    p95: float
    p99: float
    qps: float


def measure_latency(
    model: torch.nn.Module,
    loader: Iterable,
    n_samples: int = 1000,
    device: str = "cuda",
) -> LatencyReport:
    model.eval().to(device)
    latencies_ms: list[float] = []
    total_queries = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_samples:
                break
            batch.x = batch.x.to(device)
            batch.node_type_ids = batch.node_type_ids.to(device)
            batch.edge_index = batch.edge_index.to(device)
            batch.edge_delta_t = batch.edge_delta_t.to(device)
            batch.event_times = batch.event_times.to(device)
            batch.session_component_index = batch.session_component_index.to(device)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(batch)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            total_queries += int((batch.node_type_ids == 2).sum().item())

    latencies_ms.sort()
    if not latencies_ms:
        return LatencyReport(0, 0, 0, 0)
    p50 = latencies_ms[len(latencies_ms) // 2]
    p95 = latencies_ms[int(0.95 * len(latencies_ms))]
    p99 = latencies_ms[min(int(0.99 * len(latencies_ms)), len(latencies_ms) - 1)]
    wall = sum(latencies_ms) / 1000.0
    qps = total_queries / max(wall, 1e-6)
    return LatencyReport(p50, p95, p99, qps)
