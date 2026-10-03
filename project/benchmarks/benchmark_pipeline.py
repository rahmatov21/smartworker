"""
Performance Benchmark for AI Text Pipeline.
Outputs JSON formatted metrics for the Agent Evaluator.
"""

import json
import sys
import time
from pathlib import Path

# Ensure workspace root is in sys.path
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from project.src.ai_pipeline import TextPipeline


def run_benchmark():
    pipeline = TextPipeline()
    corpus = [
        "Autonomous self-improving agents represent the future of software engineering.",
        "Reliability and recoverability are paramount in long-running computational systems.",
        "Deep reinforcement learning allows policies to adapt under non-stationary distributions.",
        "High performance computing requires caching, algorithmic efficiency, and memory profiling.",
    ] * 25  # 100 documents

    query = "autonomous systems and performance caching"

    # Measure search latency
    start = time.perf_counter()
    iterations = 50
    for _ in range(iterations):
        pipeline.search(query, corpus)
    total_time = time.perf_counter() - start

    avg_latency_ms = round((total_time / iterations) * 1000, 3)
    ops_per_second = round(iterations / total_time, 1)

    result = {
        "avg_latency_ms": avg_latency_ms,
        "ops_per_second": ops_per_second,
        "corpus_size": len(corpus),
        "iterations": iterations,
    }

    print(json.dumps(result))


if __name__ == "__main__":
    run_benchmark()
