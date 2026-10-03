"""
Process & Resource Watchdog (Layer A).
Monitors the agent process, detects hangs, infinite loops, memory leaks,
and runaway child processes.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any
import psutil


@dataclass
class WatchdogStatus:
    status: str  # OK, DEAD, HEARTBEAT_TIMEOUT, OOM, HIGH_CPU, RUNTIME_EXCEEDED
    is_healthy: bool
    agent_pid: Optional[int]
    cpu_percent: float
    memory_mb: float
    child_count: int
    heartbeat_age_seconds: float
    details: str


class Watchdog:
    def __init__(
        self,
        heartbeat_file: Path,
        heartbeat_timeout_seconds: float = 45.0,
        max_runtime_per_objective_seconds: float = 300.0,
        max_memory_mb: float = 1024.0,
        max_cpu_percent: float = 95.0,
        logger: Optional[logging.Logger] = None,
    ):
        self.heartbeat_file = heartbeat_file
        self.heartbeat_timeout = heartbeat_timeout_seconds
        self.max_runtime = max_runtime_per_objective_seconds
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self.logger = logger or logging.getLogger("SupervisorWatchdog")
        self.objective_start_time = time.time()

    def reset_objective_timer(self) -> None:
        """Resets the timer when an objective begins."""
        self.objective_start_time = time.time()

    def get_heartbeat_age(self) -> float:
        """Returns the age in seconds of the latest agent heartbeat."""
        if not self.heartbeat_file.exists():
            return float("inf")
        try:
            with open(self.heartbeat_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                ts = data.get("timestamp", 0)
                return max(0.0, time.time() - ts)
        except Exception as e:
            self.logger.warning(f"Failed to read heartbeat file: {e}")
            return float("inf")

    def inspect_process_tree(self, proc: psutil.Process) -> Dict[str, Any]:
        """Calculates total CPU%, RAM (MB), and child process count for proc and children."""
        try:
            mem = proc.memory_info().rss
            cpu = proc.cpu_percent(interval=0.0)
            children = proc.children(recursive=True)
            for child in children:
                try:
                    mem += child.memory_info().rss
                    cpu += child.cpu_percent(interval=0.0)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return {
                "memory_mb": round(mem / (1024 * 1024), 2),
                "cpu_percent": round(cpu, 1),
                "child_count": len(children),
                "children": children,
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {"memory_mb": 0.0, "cpu_percent": 0.0, "child_count": 0, "children": []}

    def check(self, process: Optional[psutil.Process]) -> WatchdogStatus:
        """
        Evaluates the health and resource consumption of the agent process.
        """
        if process is None:
            return WatchdogStatus(
                status="DEAD",
                is_healthy=False,
                agent_pid=None,
                cpu_percent=0.0,
                memory_mb=0.0,
                child_count=0,
                heartbeat_age_seconds=float("inf"),
                details="No agent process registered.",
            )

        # 1. Process liveness check
        if not process.is_running():
            return WatchdogStatus(
                status="DEAD",
                is_healthy=False,
                agent_pid=process.pid,
                cpu_percent=0.0,
                memory_mb=0.0,
                child_count=0,
                heartbeat_age_seconds=float("inf"),
                details=f"Agent process PID {process.pid} is not running.",
            )

        # 2. Resource usage check
        metrics = self.inspect_process_tree(process)
        mem_mb = metrics["memory_mb"]
        cpu_pct = metrics["cpu_percent"]
        child_count = metrics["child_count"]

        if mem_mb > self.max_memory_mb:
            return WatchdogStatus(
                status="OOM",
                is_healthy=False,
                agent_pid=process.pid,
                cpu_percent=cpu_pct,
                memory_mb=mem_mb,
                child_count=child_count,
                heartbeat_age_seconds=0.0,
                details=f"Memory limit exceeded: {mem_mb} MB > {self.max_memory_mb} MB",
            )

        # 3. Heartbeat check
        hb_age = self.get_heartbeat_age()
        if hb_age > self.heartbeat_timeout:
            return WatchdogStatus(
                status="HEARTBEAT_TIMEOUT",
                is_healthy=False,
                agent_pid=process.pid,
                cpu_percent=cpu_pct,
                memory_mb=mem_mb,
                child_count=child_count,
                heartbeat_age_seconds=hb_age,
                details=f"Agent heartbeat expired ({hb_age:.1f}s > {self.heartbeat_timeout}s). Possible freeze or infinite loop.",
            )

        # 4. Objective max runtime check
        runtime = time.time() - self.objective_start_time
        if runtime > self.max_runtime:
            return WatchdogStatus(
                status="RUNTIME_EXCEEDED",
                is_healthy=False,
                agent_pid=process.pid,
                cpu_percent=cpu_pct,
                memory_mb=mem_mb,
                child_count=child_count,
                heartbeat_age_seconds=hb_age,
                details=f"Max objective execution time exceeded ({runtime:.1f}s > {self.max_runtime}s).",
            )

        return WatchdogStatus(
            status="OK",
            is_healthy=True,
            agent_pid=process.pid,
            cpu_percent=cpu_pct,
            memory_mb=mem_mb,
            child_count=child_count,
            heartbeat_age_seconds=hb_age,
            details="Process operating normally within resource limits.",
        )

    def terminate_process_tree(self, proc: Optional[psutil.Process], timeout: float = 5.0) -> None:
        """
        Terminates the agent process and all child processes cleanly, then forcefully if needed.
        """
        if proc is None:
            return
        try:
            if not proc.is_running():
                return
            self.logger.info(f"Terminating agent PID {proc.pid} and all descendants...")
            children = proc.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            proc.terminate()

            gone, alive = psutil.wait_procs([proc] + children, timeout=timeout)
            for p in alive:
                try:
                    self.logger.warning(f"Force killing PID {p.pid}...")
                    p.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
