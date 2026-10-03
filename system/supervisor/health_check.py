"""
Health Check System (Layer A).
Performs periodic comprehensive diagnostics on the system and agent.
Evaluates status as HEALTHY, DEGRADED, or FAILED.
"""

import importlib
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import psutil

from .protection import verify_integrity


@dataclass
class HealthStatus:
    status: str  # HEALTHY, DEGRADED, FAILED
    timestamp: float
    checks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def is_failed(self) -> bool:
        return self.status == "FAILED"


class HealthChecker:
    def __init__(
        self,
        workspace_root: Path,
        baseline_hashes: Dict[str, str],
        required_files: Optional[List[str]] = None,
        required_packages: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.workspace_root = workspace_root.resolve()
        self.baseline_hashes = baseline_hashes
        self.required_files = required_files or [
            "agent/main.py",
            "agent/planner.py",
            "agent/objective_manager.py",
            "agent/memory.py",
            "agent/evaluator.py",
            "agent/tools.py",
            "agent/llm_client.py",
        ]
        self.required_packages = required_packages or ["psutil", "requests", "yaml", "pytest"]
        self.logger = logger or logging.getLogger("HealthChecker")

    def run_all_checks(
        self,
        agent_proc: Optional[psutil.Process] = None,
        check_api: bool = True,
    ) -> HealthStatus:
        """Runs all 10 health checks and assigns an overall health grade."""
        checks: Dict[str, Dict[str, Any]] = {}
        failures: List[str] = []
        warnings: List[str] = []

        # 1. Supervisor Integrity (Critical)
        system_dir = self.workspace_root / "system"
        valid, violations = verify_integrity(system_dir, self.baseline_hashes)
        checks["supervisor_integrity"] = {"passed": valid, "violations": violations}
        if not valid:
            failures.append(f"Supervisor integrity violation: {', '.join(violations)}")

        # 2. Agent Process Liveness
        if agent_proc is not None:
            is_alive = agent_proc.is_running() and agent_proc.status() != psutil.STATUS_ZOMBIE
            checks["agent_liveness"] = {"passed": is_alive, "pid": agent_proc.pid}
            if not is_alive:
                failures.append(f"Agent process PID {agent_proc.pid} is dead or zombie.")
        else:
            checks["agent_liveness"] = {"passed": False, "details": "Agent not running"}

        # 3. Required Files Existence (Critical)
        missing_files = []
        for rel_file in self.required_files:
            file_path = self.workspace_root / rel_file
            if not file_path.exists():
                missing_files.append(rel_file)
        file_check_passed = len(missing_files) == 0
        checks["required_files"] = {"passed": file_check_passed, "missing": missing_files}
        if not file_check_passed:
            failures.append(f"Missing required agent files: {', '.join(missing_files)}")

        # 4. Required Python Packages (Critical)
        missing_pkgs = []
        for pkg in self.required_packages:
            try:
                importlib.import_module(pkg)
            except ImportError:
                missing_pkgs.append(pkg)
        pkg_passed = len(missing_pkgs) == 0
        checks["required_packages"] = {"passed": pkg_passed, "missing": missing_pkgs}
        if not pkg_passed:
            failures.append(f"Missing required Python packages: {', '.join(missing_pkgs)}")

        # 5. State & Database Accessibility
        state_dir = self.workspace_root / "state"
        state_accessible = True
        corrupt_state = []
        if state_dir.exists():
            for f in state_dir.glob("*.json"):
                try:
                    with open(f, "r", encoding="utf-8") as s_file:
                        json.load(s_file)
                except Exception as e:
                    corrupt_state.append(f"{f.name}: {str(e)}")
                    state_accessible = False
        checks["state_accessibility"] = {
            "passed": state_accessible,
            "corrupted_files": corrupt_state,
        }
        if not state_accessible:
            warnings.append(f"Corrupted state JSON files: {', '.join(corrupt_state)}")

        # 6. Memory System Operational
        lessons_file = state_dir / "lessons.json"
        experiments_file = state_dir / "experiments.json"
        memory_ok = True
        try:
            state_dir.mkdir(parents=True, exist_ok=True)
            test_file = state_dir / ".health_rw_test.tmp"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink()
        except Exception as e:
            memory_ok = False
            failures.append(f"State disk read/write failed: {e}")
        checks["memory_system"] = {"passed": memory_ok}

        # 7. Test Execution Capability (Pytest runnable)
        test_exec_ok = True
        try:
            res = subprocess.run(
                [sys.executable, "-m", "pytest", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            test_exec_ok = res.returncode == 0
        except Exception as e:
            test_exec_ok = False
        checks["test_execution"] = {"passed": test_exec_ok}
        if not test_exec_ok:
            failures.append("Pytest test runner is not executable.")

        # 8. API Connection Check
        if check_api:
            api_ok, api_msg = self._check_api_connectivity()
            checks["api_connectivity"] = {"passed": api_ok, "details": api_msg}
            if not api_ok:
                warnings.append(f"API connectivity degraded: {api_msg}")
        else:
            checks["api_connectivity"] = {"passed": True, "details": "Skipped"}

        # 9. Resource Limits Check
        resource_ok = True
        res_details = "OK"
        if agent_proc is not None and agent_proc.is_running():
            try:
                mem_mb = agent_proc.memory_info().rss / (1024 * 1024)
                if mem_mb > 1024:
                    resource_ok = False
                    res_details = f"High RAM usage: {mem_mb:.1f} MB"
                    failures.append(res_details)
            except Exception:
                pass
        checks["resource_usage"] = {"passed": resource_ok, "details": res_details}

        # 10. Runaway Child Processes Check
        child_ok = True
        child_count = 0
        if agent_proc is not None and agent_proc.is_running():
            try:
                children = agent_proc.children(recursive=True)
                child_count = len(children)
                if child_count > 15:
                    child_ok = False
                    warnings.append(f"Elevated child process count: {child_count}")
            except Exception:
                pass
        checks["child_processes"] = {"passed": child_ok, "count": child_count}

        # Determine overall grade
        if len(failures) > 0:
            status = "FAILED"
        elif len(warnings) > 0:
            status = "DEGRADED"
        else:
            status = "HEALTHY"

        result = HealthStatus(
            status=status,
            timestamp=time.time(),
            checks=checks,
            failures=failures,
            warnings=warnings,
        )

        self._log_health_result(result)
        return result

    def _check_api_connectivity(self) -> Tuple[bool, str]:
        """Verifies OpenRouter API or local mock configuration."""
        mock_env = os.environ.get("MOCK_LLM", "false").lower() in ("true", "1", "yes")
        if mock_env:
            return True, "Mock LLM active (offline mode)"

        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY_1")
        if not api_key:
            return True, "No API key configured yet (operating in fallback/mock mode)"

        # Check OpenRouter /auth/key or models endpoint
        try:
            import urllib.request
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/models",
                headers={"User-Agent": "SelfImprover-HealthCheck/1.0"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True, "OpenRouter API endpoint reachable"
                return False, f"OpenRouter returned status {resp.status}"
        except Exception as e:
            return False, f"Could not reach OpenRouter: {e}"

    def _log_health_result(self, result: HealthStatus) -> None:
        """Appends health status to logs/health.log."""
        log_dir = self.workspace_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "health.log"
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(result.timestamp)),
            "status": result.status,
            "failures": result.failures,
            "warnings": result.warnings,
        }
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write to health.log: {e}")
