"""
Autonomous Tool: Micro-Benchmark Performance Profiler.
Measures execution time, latency percentiles, and throughput for functions.
"""

import time
from typing import Dict, Any, Callable, Optional


def run(iterations: int = 1000, target_func: Optional[Callable] = None, **kwargs) -> Dict[str, Any]:
    """
    Profiles execution latency and operations per second.
    """
    if target_func is None:
        # Default benchmark: tokenize pipeline test
        try:
            from project.src.ai_pipeline import TextPipeline
            pipe = TextPipeline()
            target_func = lambda: pipe.process("The quick brown fox jumps over the lazy dog and explores the world.")
        except Exception as e:
            return {"error": f"Default benchmark target unavailable: {e}", "success": False}

    latencies = []
    start_total = time.perf_counter()

    for _ in range(iterations):
        t0 = time.perf_counter()
        target_func()
        latencies.append((time.perf_counter() - t0) * 1000.0)  # ms

    total_time = time.perf_counter() - start_total
    latencies.sort()

    avg_ms = sum(latencies) / len(latencies)
    p50_ms = latencies[len(latencies) // 2]
    p95_ms = latencies[int(len(latencies) * 0.95)]
    p99_ms = latencies[int(len(latencies) * 0.99)]
    ops_per_sec = iterations / total_time if total_time > 0 else 0.0

    return {
        "success": True,
        "iterations": iterations,
        "total_time_seconds": round(total_time, 4),
        "throughput_ops_per_sec": round(ops_per_sec, 2),
        "latency_avg_ms": round(avg_ms, 4),
        "latency_p50_ms": round(p50_ms, 4),
        "latency_p95_ms": round(p95_ms, 4),
        "latency_p99_ms": round(p99_ms, 4),
    }
