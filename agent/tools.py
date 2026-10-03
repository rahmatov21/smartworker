"""
Agent Sandboxed Tool Suite (Layer B).
Provides autonomous capabilities for file I/O, Git checkpointing, test execution,
and code analysis.
Strictly enforces supervisor immutability (Layer A is untouchable).
"""

import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from system.supervisor.protection import is_path_protected, assert_path_allowed, ProtectionError


class ToolExecutionError(Exception):
    pass


class AgentTools:
    def __init__(
        self,
        workspace_root: Optional[Path] = None,
        unsecured: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        self.workspace_root = (workspace_root or Path(__file__).resolve().parent.parent).resolve()
        self.unsecured = unsecured or os.environ.get("UNSECURED_MODE", "0") == "1"
        self.logs_dir = self.workspace_root / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger("AgentTools")
        self.tool_log_file = self.logs_dir / "tool_calls.log"

    def _log_tool_call(self, tool_name: str, args: Dict[str, Any], result_summary: str, success: bool) -> None:
        """Appends tool execution event to logs/tool_calls.log."""
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tool": tool_name,
            "args": {k: str(v)[:200] for k, v in args.items()},
            "success": success,
            "result": result_summary[:300],
        }
        try:
            with open(self.tool_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    def read_file(self, filepath: str) -> str:
        """Reads contents of a file in the workspace."""
        path = (self.workspace_root / filepath).resolve()
        try:
            if not self.unsecured:
                assert_path_allowed(path, self.workspace_root)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {filepath}")
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self._log_tool_call("read_file", {"filepath": filepath}, f"Read {len(content)} chars", True)
            return content
        except Exception as e:
            self._log_tool_call("read_file", {"filepath": filepath}, str(e), False)
            raise

    def write_file(self, filepath: str, content: str) -> None:
        """
        Writes content to a file in the workspace.
        Protected from touching /system/supervisor/ unless in unsecured mode.
        """
        path = (self.workspace_root / filepath).resolve()
        try:
            if not self.unsecured:
                assert_path_allowed(path, self.workspace_root)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            self._log_tool_call("write_file", {"filepath": filepath}, f"Wrote {len(content)} chars", True)
        except Exception as e:
            self._log_tool_call("write_file", {"filepath": filepath}, str(e), False)
            raise

    def delete_file(self, filepath: str) -> None:
        """Deletes a file in the workspace, with supervisor protection unless unsecured."""
        path = (self.workspace_root / filepath).resolve()
        try:
            if not self.unsecured:
                assert_path_allowed(path, self.workspace_root)
            if path.exists():
                path.unlink()
            self._log_tool_call("delete_file", {"filepath": filepath}, "Deleted successfully", True)
        except Exception as e:
            self._log_tool_call("delete_file", {"filepath": filepath}, str(e), False)
            raise

    def list_files(self, directory: str = ".") -> List[str]:
        """Lists files recursively under a workspace subdirectory."""
        target_dir = (self.workspace_root / directory).resolve()
        try:
            results = []
            for root, _, files in os.walk(target_dir):
                for f in files:
                    full = Path(root) / f
                    # Do not leak supervisor files into agent listing unless unsecured
                    if not self.unsecured and is_path_protected(full, self.workspace_root):
                        continue
                    results.append(full.relative_to(self.workspace_root).as_posix())
            self._log_tool_call("list_files", {"directory": directory}, f"Found {len(results)} files", True)
            return sorted(results)
        except Exception as e:
            self._log_tool_call("list_files", {"directory": directory}, str(e), False)
            raise

    def search_code(self, query: str, directory: str = ".") -> List[Dict[str, Any]]:
        """Searches for text pattern in workspace source files."""
        matches = []
        files = self.list_files(directory)
        for rel_file in files:
            if not rel_file.endswith((".py", ".json", ".yaml", ".md", ".txt")):
                continue
            try:
                full_path = self.workspace_root / rel_file
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line_idx, line in enumerate(f, start=1):
                        if query.lower() in line.lower():
                            matches.append({
                                "file": rel_file,
                                "line_number": line_idx,
                                "content": line.strip()[:150],
                            })
            except Exception:
                continue
        self._log_tool_call("search_code", {"query": query, "dir": directory}, f"{len(matches)} matches", True)
        return matches

    def run_command(self, cmd: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Executes an approved shell command in the workspace directory.
        Blocks commands that attempt to tamper with supervisor unless in unsecured mode.
        """
        # Security sanitization (bypassed in unsecured mode)
        if not self.unsecured:
            dangerous_patterns = [
                r"\bsystem[\\/]",
                r"\bsupervisor\b",
                r"\bkill_switch\b",
                r"\bformat\b",
                r"\bdiskpart\b",
                r"\brm\s+-rf\s+/",
                r"\bdel\s+.*system",
                r"\brmdir\s+.*system",
            ]
            for pattern in dangerous_patterns:
                if re.search(pattern, cmd, re.IGNORECASE):
                    err = f"Security Violation: Command contains forbidden pattern matching supervisor protection: {cmd}"
                    self._log_tool_call("run_command", {"cmd": cmd}, err, False)
                    raise ProtectionError(err)

        start = time.time()
        try:
            res = subprocess.run(
                cmd,
                cwd=self.workspace_root,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            duration = time.time() - start
            out = {
                "stdout": res.stdout,
                "stderr": res.stderr,
                "returncode": res.returncode,
                "duration": round(duration, 2),
                "success": res.returncode == 0,
            }
            self._log_tool_call("run_command", {"cmd": cmd}, f"Exit: {res.returncode}", res.returncode == 0)
            return out
        except subprocess.TimeoutExpired:
            self._log_tool_call("run_command", {"cmd": cmd}, f"Timeout after {timeout}s", False)
            return {"stdout": "", "stderr": f"Command timed out after {timeout}s", "returncode": -1, "duration": timeout, "success": False}
        except Exception as e:
            self._log_tool_call("run_command", {"cmd": cmd}, str(e), False)
            return {"stdout": "", "stderr": str(e), "returncode": -1, "duration": 0.0, "success": False}

    def run_tests(self, test_path: str = "project/tests") -> Dict[str, Any]:
        """
        Executes pytest suite and parses test statistics.
        """
        full_test_dir = self.workspace_root / test_path
        if not full_test_dir.exists():
            return {
                "success": True,
                "passed": 0,
                "failed": 0,
                "errors": 0,
                "total": 0,
                "duration": 0.0,
                "stdout": f"No test directory found at {test_path}",
            }

        start = time.time()
        res = subprocess.run(
            [sys.executable, "-m", "pytest", test_path, "-v", "--tb=short"],
            cwd=self.workspace_root,
            capture_output=True,
            text=True,
            timeout=60,
        )
        duration = round(time.time() - start, 2)
        output = res.stdout + "\n" + res.stderr

        # Parse pytest output
        passed = 0
        failed = 0
        errors = 0

        passed_match = re.search(r"(\d+)\s+passed", output)
        if passed_match:
            passed = int(passed_match.group(1))

        failed_match = re.search(r"(\d+)\s+failed", output)
        if failed_match:
            failed = int(failed_match.group(1))

        error_match = re.search(r"(\d+)\s+error", output)
        if error_match:
            errors = int(error_match.group(1))

        total = passed + failed + errors
        success = (res.returncode == 0) and (failed == 0) and (errors == 0)

        result = {
            "success": success,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "total": total,
            "duration": duration,
            "stdout": output,
        }
        self._log_tool_call(
            "run_tests",
            {"test_path": test_path},
            f"Passed: {passed}, Failed: {failed}, Errors: {errors}",
            success,
        )
        return result

    def run_benchmarks(self, benchmark_path: str = "project/benchmarks") -> Dict[str, Any]:
        """Runs project benchmark scripts and extracts latency/throughput."""
        bench_dir = self.workspace_root / benchmark_path
        if not bench_dir.exists():
            return {"benchmarks_ran": 0, "results": {}}

        results = {}
        for py_bench in bench_dir.glob("*.py"):
            try:
                res = subprocess.run(
                    [sys.executable, str(py_bench)],
                    cwd=self.workspace_root,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if res.returncode == 0:
                    try:
                        # Expect JSON output from benchmark script
                        results[py_bench.stem] = json.loads(res.stdout.strip())
                    except Exception:
                        results[py_bench.stem] = {"raw_output": res.stdout.strip()}
                else:
                    results[py_bench.stem] = {"error": res.stderr.strip()}
            except Exception as e:
                results[py_bench.stem] = {"error": str(e)}

        self._log_tool_call("run_benchmarks", {"benchmark_path": benchmark_path}, f"Ran {len(results)} benches", True)
        return {"benchmarks_ran": len(results), "results": results}

    # Git Operations
    def git_status(self) -> Dict[str, Any]:
        """Returns clean/dirty state and current branch."""
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=self.workspace_root,
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
        return {"clean": len(lines) == 0, "modified_files": lines}

    def git_checkpoint(self, checkpoint_name: str) -> str:
        """Creates a dedicated checkpoint branch or tag before making changes."""
        # Stage non-supervisor changes
        subprocess.run(["git", "add", "agent", "project", "state"], cwd=self.workspace_root, capture_output=True)
        msg = f"checkpoint: {checkpoint_name}"
        subprocess.run(["git", "commit", "-m", msg, "--allow-empty"], cwd=self.workspace_root, capture_output=True)
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.workspace_root, capture_output=True, text=True)
        commit = res.stdout.strip()
        self._log_tool_call("git_checkpoint", {"name": checkpoint_name}, f"Commit: {commit[:8]}", True)
        return commit

    def git_commit(self, message: str) -> str:
        """Commits all staged changes in agent and project."""
        subprocess.run(["git", "add", "agent", "project", "state"], cwd=self.workspace_root, capture_output=True)
        res = subprocess.run(["git", "commit", "-m", message], cwd=self.workspace_root, capture_output=True, text=True)
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.workspace_root, capture_output=True, text=True)
        commit = rev.stdout.strip()
        self._log_tool_call("git_commit", {"message": message}, f"Commit: {commit[:8]}", True)
        return commit

    def git_rollback(self, commit_hash: str) -> bool:
        """Rolls back agent and project files to a previous commit."""
        try:
            subprocess.run(["git", "reset", "--hard", commit_hash], cwd=self.workspace_root, check=True, capture_output=True)
            subprocess.run(["git", "clean", "-fd", "--", "agent", "project"], cwd=self.workspace_root, check=True, capture_output=True)
            self._log_tool_call("git_rollback", {"target_commit": commit_hash}, "Rolled back successfully", True)
            return True
        except Exception as e:
            self._log_tool_call("git_rollback", {"target_commit": commit_hash}, str(e), False)
            return False
