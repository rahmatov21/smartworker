"""
Evaluator & Benchmark System (Layer B).
Enforces the Test-Before-Acceptance rule:
CHANGE -> TEST -> EVALUATE -> ACCEPT OR ROLLBACK.
Validates zero test regressions, benchmark performance, and supervisor integrity.
"""

import logging
import py_compile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

from system.supervisor.protection import verify_integrity, compute_supervisor_hashes
from .tools import AgentTools


@dataclass
class EvaluationReport:
    passed: bool
    should_rollback: bool
    tests_passed: int
    tests_failed: int
    tests_errors: int
    test_duration: float
    integrity_ok: bool
    benchmark_metrics: Dict[str, Any] = field(default_factory=dict)
    failure_reason: Optional[str] = None
    diagnostics: List[str] = field(default_factory=list)


class Evaluator:
    def __init__(
        self,
        workspace_root: Optional[Path] = None,
        tools: Optional[AgentTools] = None,
        baseline_hashes: Optional[Dict[str, str]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.workspace_root = (workspace_root or Path(__file__).resolve().parent.parent).resolve()
        self.tools = tools or AgentTools(self.workspace_root)
        self.logger = logger or logging.getLogger("Evaluator")

        # Load baseline hashes for supervisor integrity verification
        system_dir = self.workspace_root / "system"
        if baseline_hashes:
            self.baseline_hashes = baseline_hashes
        else:
            self.baseline_hashes = compute_supervisor_hashes(system_dir)

    def evaluate_change(
        self,
        baseline_metrics: Optional[Dict[str, Any]] = None,
        test_path: str = "project/tests",
        benchmark_path: str = "project/benchmarks",
    ) -> EvaluationReport:
        """
        Executes comprehensive evaluation of the modified codebase.
        Enforces zero test failures, supervisor immutability, and syntax correctness.
        """
        diagnostics = []

        # 1. Syntax Check on all Python files
        syntax_ok = True
        for root_dir in ["agent", "project"]:
            target_path = self.workspace_root / root_dir
            if not target_path.exists():
                continue
            for py_file in target_path.rglob("*.py"):
                try:
                    py_compile.compile(str(py_file), doraise=True)
                except py_compile.PyCompileError as e:
                    syntax_ok = False
                    diagnostics.append(f"Syntax error in {py_file.name}: {e}")

        if not syntax_ok:
            return EvaluationReport(
                passed=False,
                should_rollback=True,
                tests_passed=0,
                tests_failed=0,
                tests_errors=1,
                test_duration=0.0,
                integrity_ok=True,
                failure_reason="Syntax validation failed.",
                diagnostics=diagnostics,
            )

        # 2. Supervisor Integrity Check (CRITICAL SAFETY RULE)
        system_dir = self.workspace_root / "system"
        integrity_ok, violations = verify_integrity(system_dir, self.baseline_hashes)
        if not integrity_ok:
            diagnostics.append(f"CRITICAL: Supervisor directory was tampered with! Violations: {violations}")
            return EvaluationReport(
                passed=False,
                should_rollback=True,
                tests_passed=0,
                tests_failed=0,
                tests_errors=1,
                test_duration=0.0,
                integrity_ok=False,
                failure_reason="Supervisor integrity violation.",
                diagnostics=diagnostics,
            )

        # 3. Test Suite Execution
        test_res = self.tools.run_tests(test_path)
        passed = test_res["passed"]
        failed = test_res["failed"]
        errors = test_res["errors"]
        duration = test_res["duration"]

        if not test_res["success"] or failed > 0 or errors > 0:
            diagnostics.append(f"Test failures detected: {failed} failed, {errors} errors.")
            diagnostics.append(f"Pytest output:\n{test_res['stdout'][-800:]}")
            return EvaluationReport(
                passed=False,
                should_rollback=True,
                tests_passed=passed,
                tests_failed=failed,
                tests_errors=errors,
                test_duration=duration,
                integrity_ok=True,
                failure_reason=f"Tests failed ({failed} failures, {errors} errors).",
                diagnostics=diagnostics,
            )

        # 4. Anti-Regression: Prevent test suite shrinkage
        if baseline_metrics:
            prev_tests = baseline_metrics.get("tests_passed", 0)
            if passed < prev_tests:
                diagnostics.append(
                    f"Test count decreased from {prev_tests} to {passed}. Potential test deletion."
                )
                return EvaluationReport(
                    passed=False,
                    should_rollback=True,
                    tests_passed=passed,
                    tests_failed=failed,
                    tests_errors=errors,
                    test_duration=duration,
                    integrity_ok=True,
                    failure_reason="Test count decreased (anti-regression check failed).",
                    diagnostics=diagnostics,
                )

        # 5. Benchmark Performance
        bench_res = self.tools.run_benchmarks(benchmark_path)
        benchmarks = bench_res.get("results", {})

        self.logger.info(
            f"Evaluation PASSED. {passed} tests passed in {duration}s. Integrity verified."
        )

        return EvaluationReport(
            passed=True,
            should_rollback=False,
            tests_passed=passed,
            tests_failed=0,
            tests_errors=0,
            test_duration=duration,
            integrity_ok=True,
            benchmark_metrics=benchmarks,
            diagnostics=diagnostics,
        )
