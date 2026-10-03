"""
Autonomous System Auto-Healer & Progress Monitor Service (Layer A).
Continuously inspects the system for stagnation, deadlocks, frozen processes,
metric plateaus, or abandoned objectives, and automatically performs
remedial healing actions to get the system moving and improving again.
"""

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

from .protection import verify_integrity, compute_supervisor_hashes


@dataclass
class HealingAction:
    issue_type: str
    action_taken: str
    timestamp: float
    success: bool
    details: str


@dataclass
class DiagnosisReport:
    timestamp: float
    is_healthy: bool
    stuck_issues: List[str] = field(default_factory=list)
    metrics_summary: Dict[str, Any] = field(default_factory=dict)
    recommended_fixes: List[str] = field(default_factory=list)


class AutoHealer:
    """
    Intelligent watchdog and self-healing service.
    Detects when progress has stopped, frozen, or stagnated, and actively fixes it.
    """

    def __init__(
        self,
        workspace_root: Optional[Path] = None,
        max_freeze_seconds: float = 60.0,
        max_objective_age_seconds: float = 300.0,
        stagnation_cycle_limit: int = 3,
        auto_restart_on_heal: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        self.workspace_root = (workspace_root or Path(__file__).resolve().parent.parent.parent).resolve()
        self.state_dir = self.workspace_root / "state"
        self.logs_dir = self.workspace_root / "logs"
        self.system_dir = self.workspace_root / "system"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.max_freeze_seconds = max_freeze_seconds
        self.max_objective_age_seconds = max_objective_age_seconds
        self.stagnation_cycle_limit = stagnation_cycle_limit
        self.auto_restart_on_heal = auto_restart_on_heal

        self.logger = logger or self._setup_logger()
        self.healer_log_file = self.logs_dir / "healer.log"

    def _setup_logger(self) -> logging.Logger:
        """Sets up dedicated auto-healer logger."""
        logger = logging.getLogger("AutoHealer")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [AUTO-HEALER] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            logger.addHandler(ch)

            fh = logging.FileHandler(self.logs_dir / "healer.log", encoding="utf-8")
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        return logger

    def _log_healing_action(self, action: HealingAction) -> None:
        """Records a healing event to logs/healer.log and state/lessons.json."""
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(action.timestamp)),
            "issue": action.issue_type,
            "action": action.action_taken,
            "success": action.success,
            "details": action.details,
        }
        try:
            with open(self.healer_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

        # Record as lesson in memory
        lessons_file = self.state_dir / "lessons.json"
        lessons = []
        if lessons_file.exists():
            try:
                with open(lessons_file, "r", encoding="utf-8") as f:
                    lessons = json.load(f)
            except Exception:
                lessons = []

        lessons.append({
            "timestamp": time.time(),
            "event": "AUTO_HEALER_ACTION",
            "topic": f"Auto-heal: {action.issue_type}",
            "lesson": f"Resolved system stall via {action.action_taken}: {action.details}",
            "outcome": "SUCCESS" if action.success else "FAILURE",
        })

        try:
            with open(lessons_file, "w", encoding="utf-8") as f:
                json.dump(lessons, f, indent=2)
        except Exception:
            pass

    # ---------------------------------------------------------
    # Diagnostic Inspections
    # ---------------------------------------------------------
    def diagnose(self, agent_proc: Optional[psutil.Process] = None) -> DiagnosisReport:
        """
        Comprehensive inspection to determine if the system is stopped, frozen,
        or failing to make progress.
        """
        issues = []
        recommendations = []
        metrics_summary = {}

        # 1. Check Process & Heartbeat Freeze
        heartbeat_file = self.state_dir / "heartbeat.json"
        hb_age = float("inf")
        if heartbeat_file.exists():
            try:
                with open(heartbeat_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    hb_age = time.time() - data.get("timestamp", 0)
            except Exception:
                pass

        if agent_proc is not None and agent_proc.is_running():
            if hb_age > self.max_freeze_seconds:
                issues.append("PROCESS_HEARTBEAT_FROZEN")
                recommendations.append("Terminate frozen process and restart agent.")
        elif agent_proc is not None and not agent_proc.is_running():
            issues.append("AGENT_PROCESS_DEAD")
            recommendations.append("Relaunch dead agent subprocess.")

        # 2. Check for Stalled In-Flight Objective
        current_obj_file = self.state_dir / "current_objective.json"
        if current_obj_file.exists():
            try:
                mtime = os.path.getmtime(current_obj_file)
                age = time.time() - mtime
                with open(current_obj_file, "r", encoding="utf-8") as f:
                    obj_data = json.load(f)
                status = obj_data.get("status", "")
                if status in ("in_progress", "pending", "testing") and age > self.max_objective_age_seconds:
                    issues.append("OBJECTIVE_STALLED")
                    recommendations.append(
                        f"Abandon stalled objective '{obj_data.get('title', 'unknown')}' ({age:.0f}s elapsed)."
                    )
            except Exception as e:
                issues.append("CORRUPTED_CURRENT_OBJECTIVE")
                recommendations.append(f"Reset corrupted current_objective.json: {e}")

        # 3. Check for Metric Plateau / Stagnation (Stopped Increasing)
        experiments_file = self.state_dir / "experiments.json"
        if experiments_file.exists():
            try:
                with open(experiments_file, "r", encoding="utf-8") as f:
                    experiments = json.load(f)
                if len(experiments) >= self.stagnation_cycle_limit:
                    recent = experiments[-self.stagnation_cycle_limit :]
                    # Check if all recent experiments failed
                    all_failed = all(not e.get("passed", False) for e in recent)

                    # Check test count progress
                    test_counts = [
                        e.get("metrics_after", {}).get("tests_passed", 0)
                        for e in recent
                        if "metrics_after" in e
                    ]
                    no_test_growth = (
                        len(test_counts) >= self.stagnation_cycle_limit
                        and max(test_counts) <= min(test_counts)
                    )

                    if all_failed:
                        issues.append("REPEATED_EXPERIMENT_FAILURES")
                        recommendations.append(
                            f"Break failure loop: Inject fresh objective direction after {self.stagnation_cycle_limit} failed attempts."
                        )
                    elif no_test_growth:
                        issues.append("METRIC_STAGNATION_PLATEAU")
                        recommendations.append(
                            "Inject divergent architectural breakthrough prompt to escape local optimum."
                        )

                metrics_summary["total_experiments"] = len(experiments)
                metrics_summary["recent_experiments_passed"] = sum(
                    1 for e in experiments[-self.stagnation_cycle_limit :] if e.get("passed", False)
                )
            except Exception:
                pass

        # 4. Check for Orphaned Git Lockfiles
        git_lock = self.workspace_root / ".git" / "index.lock"
        if git_lock.exists():
            issues.append("GIT_INDEX_LOCKED")
            recommendations.append("Remove orphaned .git/index.lock file.")

        # 5. Check for Corrupted State JSON Files
        for f in self.state_dir.glob("*.json"):
            try:
                with open(f, "r", encoding="utf-8") as jf:
                    json.load(jf)
            except Exception as e:
                issues.append(f"CORRUPT_STATE_FILE_{f.name}")
                recommendations.append(f"Regenerate corrupted state file {f.name}.")

        is_healthy = len(issues) == 0
        return DiagnosisReport(
            timestamp=time.time(),
            is_healthy=is_healthy,
            stuck_issues=issues,
            metrics_summary=metrics_summary,
            recommended_fixes=recommendations,
        )

    # ---------------------------------------------------------
    # Auto-Healing Actions
    # ---------------------------------------------------------
    def heal(
        self,
        diagnosis: DiagnosisReport,
        agent_proc: Optional[psutil.Process] = None,
        supervisor: Optional[Any] = None,
    ) -> List[HealingAction]:
        """
        Executes targeted repairs based on identified issues to get the system
        moving and improving again.
        """
        actions = []
        self.logger.warning(
            f"Initiating auto-healing for {len(diagnosis.stuck_issues)} detected issues: "
            f"{', '.join(diagnosis.stuck_issues)}"
        )

        for issue in diagnosis.stuck_issues:
            # Fix 1: Remove orphaned Git lock
            if issue == "GIT_INDEX_LOCKED":
                action = self._heal_git_lock()
                actions.append(action)

            # Fix 2: Stalled Objective
            elif issue in ("OBJECTIVE_STALLED", "CORRUPTED_CURRENT_OBJECTIVE"):
                action = self._heal_stalled_objective()
                actions.append(action)

            # Fix 3: Metric Stagnation Plateau or Repeated Failures
            elif issue in ("METRIC_STAGNATION_PLATEAU", "REPEATED_EXPERIMENT_FAILURES"):
                action = self._heal_metric_stagnation(issue)
                actions.append(action)

            # Fix 4: Process Frozen or Dead
            elif issue in ("PROCESS_HEARTBEAT_FROZEN", "AGENT_PROCESS_DEAD"):
                action = self._heal_frozen_process(agent_proc, supervisor)
                actions.append(action)

            # Fix 5: Corrupt State Files
            elif issue.startswith("CORRUPT_STATE_FILE_"):
                filename = issue.replace("CORRUPT_STATE_FILE_", "")
                action = self._heal_corrupt_state_file(filename)
                actions.append(action)

        self.logger.info(f"Auto-healing completed: {len(actions)} actions applied.")
        return actions

    def _heal_git_lock(self) -> HealingAction:
        """Removes stale .git/index.lock preventing git operations."""
        lock_file = self.workspace_root / ".git" / "index.lock"
        try:
            if lock_file.exists():
                lock_file.unlink()
                action = HealingAction(
                    issue_type="GIT_INDEX_LOCKED",
                    action_taken="Removed stale .git/index.lock",
                    timestamp=time.time(),
                    success=True,
                    details="Unblocked Git repository from lock contention.",
                )
            else:
                action = HealingAction(
                    issue_type="GIT_INDEX_LOCKED",
                    action_taken="Lock file already clear",
                    timestamp=time.time(),
                    success=True,
                    details="No action needed.",
                )
        except Exception as e:
            action = HealingAction(
                issue_type="GIT_INDEX_LOCKED",
                action_taken="Failed to remove .git/index.lock",
                timestamp=time.time(),
                success=False,
                details=str(e),
            )
        self._log_healing_action(action)
        return action

    def _heal_stalled_objective(self) -> HealingAction:
        """
        Clears or archives an objective that has been stuck in progress
        longer than max_objective_age_seconds without completing.
        """
        current_obj_file = self.state_dir / "current_objective.json"
        objectives_history_file = self.state_dir / "objectives.json"

        try:
            title = "unknown"
            if current_obj_file.exists():
                try:
                    with open(current_obj_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    title = data.get("title", "unknown")
                    data["status"] = "stalled_abandoned_by_healer"
                    data["abandoned_at"] = time.time()

                    # Save to historical objectives
                    history = {"completed": [], "failed": [], "pending": []}
                    if objectives_history_file.exists():
                        try:
                            with open(objectives_history_file, "r", encoding="utf-8") as hf:
                                history = json.load(hf)
                        except Exception:
                            pass
                    if "failed" not in history:
                        history["failed"] = []
                    history["failed"].append(data)
                    with open(objectives_history_file, "w", encoding="utf-8") as hf:
                        json.dump(history, hf, indent=2)
                except Exception:
                    pass

                current_obj_file.unlink(missing_ok=True)

            action = HealingAction(
                issue_type="OBJECTIVE_STALLED",
                action_taken="Cleared stalled objective and archived to history",
                timestamp=time.time(),
                success=True,
                details=f"Unblocked agent by abandoning stalled objective: '{title}'",
            )
        except Exception as e:
            action = HealingAction(
                issue_type="OBJECTIVE_STALLED",
                action_taken="Failed to clear stalled objective",
                timestamp=time.time(),
                success=False,
                details=str(e),
            )
        self._log_healing_action(action)
        return action

    def _heal_metric_stagnation(self, issue_type: str) -> HealingAction:
        """
        Breaks metric plateaus and deadlocks by injecting a divergent
        exploration directive into state/divergent_directive.json.
        """
        directive_file = self.state_dir / "divergent_directive.json"
        try:
            payload = {
                "active": True,
                "timestamp": time.time(),
                "reason": issue_type,
                "directive": (
                    "STAGNATION_BREAKER_INJECTION: The system has plateaued on incremental changes. "
                    "Shift focus immediately from minor edits to macro-architectural improvements, "
                    "novel algorithms, deeper edge-case coverage, or new AI retrieval capabilities."
                ),
                "boost_categories": [
                    "performance_optimization",
                    "algorithmic_expansion",
                    "architecture_refactor",
                    "testing_and_quality",
                ],
            }
            with open(directive_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            action = HealingAction(
                issue_type=issue_type,
                action_taken="Injected divergent exploration directive to break metric plateau",
                timestamp=time.time(),
                success=True,
                details="Activated STAGNATION_BREAKER directive to encourage high-impact exploration.",
            )
        except Exception as e:
            action = HealingAction(
                issue_type=issue_type,
                action_taken="Failed to inject stagnation breaker directive",
                timestamp=time.time(),
                success=False,
                details=str(e),
            )
        self._log_healing_action(action)
        return action

    def _heal_frozen_process(
        self,
        agent_proc: Optional[psutil.Process],
        supervisor: Optional[Any] = None,
    ) -> HealingAction:
        """Terminates frozen process and restarts with clean heartbeat."""
        try:
            pid = agent_proc.pid if agent_proc else "N/A"
            if agent_proc is not None and agent_proc.is_running():
                # Terminate children
                for child in agent_proc.children(recursive=True):
                    try:
                        child.kill()
                    except Exception:
                        pass
                agent_proc.kill()

            # Refresh heartbeat
            hb_file = self.state_dir / "heartbeat.json"
            with open(hb_file, "w", encoding="utf-8") as f:
                json.dump({
                    "timestamp": time.time(),
                    "pid": os.getpid(),
                    "status": "healed_restarting",
                }, f, indent=2)

            # If supervisor provided, trigger restart
            if supervisor is not None:
                supervisor.stop_agent("AutoHealer: terminating frozen process")
                supervisor.start_agent()

            action = HealingAction(
                issue_type="PROCESS_FROZEN",
                action_taken="Terminated frozen process and refreshed heartbeat",
                timestamp=time.time(),
                success=True,
                details=f"Terminated frozen PID {pid} and restored operational state.",
            )
        except Exception as e:
            action = HealingAction(
                issue_type="PROCESS_FROZEN",
                action_taken="Failed to terminate frozen process",
                timestamp=time.time(),
                success=False,
                details=str(e),
            )
        self._log_healing_action(action)
        return action

    def _heal_corrupt_state_file(self, filename: str) -> HealingAction:
        """Restores a corrupted state JSON file to its clean default structure."""
        file_path = self.state_dir / filename
        defaults = {
            "current_objective.json": None,
            "objectives.json": {"completed": [], "failed": [], "pending": []},
            "experiments.json": [],
            "lessons.json": [],
            "metrics.json": {"history": {}, "latest": {}},
            "heartbeat.json": {"timestamp": time.time(), "status": "repaired"},
        }
        try:
            default_val = defaults.get(filename, {})
            if default_val is None:
                file_path.unlink(missing_ok=True)
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(default_val, f, indent=2)

            action = HealingAction(
                issue_type="STATE_CORRUPTION",
                action_taken=f"Repaired corrupted state file {filename}",
                timestamp=time.time(),
                success=True,
                details=f"Reset {filename} to verified valid schema.",
            )
        except Exception as e:
            action = HealingAction(
                issue_type="STATE_CORRUPTION",
                action_taken=f"Failed to repair state file {filename}",
                timestamp=time.time(),
                success=False,
                details=str(e),
            )
        self._log_healing_action(action)
        return action

    def check_and_heal(
        self,
        agent_proc: Optional[psutil.Process] = None,
        supervisor: Optional[Any] = None,
    ) -> DiagnosisReport:
        """
        One-stop diagnostic and healing run.
        Diagnoses system state and immediately applies fixes if issues are detected.
        """
        report = self.diagnose(agent_proc)
        if not report.is_healthy:
            self.heal(report, agent_proc=agent_proc, supervisor=supervisor)
        return report

    def run_monitor_loop(self, poll_interval_seconds: float = 10.0, max_cycles: Optional[int] = None) -> None:
        """
        Continuous auto-healing monitoring daemon loop.
        """
        self.logger.info(
            f"AutoHealer monitoring service started (poll interval: {poll_interval_seconds}s)..."
        )
        cycle = 0
        try:
            while True:
                if max_cycles and cycle >= max_cycles:
                    break
                cycle += 1
                report = self.check_and_heal()
                time.sleep(poll_interval_seconds)
        except KeyboardInterrupt:
            self.logger.info("AutoHealer monitor interrupted by user. Stopping.")


if __name__ == "__main__":
    healer = AutoHealer()
    healer.run_monitor_loop()
