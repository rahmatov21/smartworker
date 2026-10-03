"""
Protected Supervisor (Layer A).
The unmodifiable supervisor that controls, monitors, protects,
and rolls back the autonomous agent.
"""

import json
import logging
import os
import py_compile
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any
import psutil
import yaml

from .protection import (
    compute_supervisor_hashes,
    save_baseline_hashes,
    verify_integrity,
    ProtectionError,
)
from .watchdog import Watchdog
from .health_check import HealthChecker, HealthStatus
from .auto_healer import AutoHealer

try:
    from system.env_loader import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent)
except Exception:
    pass


class Supervisor:
    def __init__(
        self,
        workspace_root: Optional[Path] = None,
        config_path: Optional[Path] = None,
        unsecured: bool = False,
    ):
        self.workspace_root = (workspace_root or Path(__file__).resolve().parent.parent.parent).resolve()
        self.system_dir = self.workspace_root / "system"
        self.supervisor_dir = self.system_dir / "supervisor"
        self.state_dir = self.workspace_root / "state"
        self.logs_dir = self.workspace_root / "logs"
        self.backups_dir = self.workspace_root / "backups"
        self.unsecured = unsecured

        # Create directories
        for d in [self.state_dir, self.logs_dir, self.backups_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.config_path = config_path or (self.supervisor_dir / "config.yaml")
        self.config = self._load_config()


        self._setup_logging()

        # Cryptographic baseline
        self.baseline_hash_file = self.workspace_root / self.config["supervisor"].get(
            "baseline_hash_file", "system/supervisor/baseline_hashes.json"
        )
        self.baseline_hashes = self._init_baseline_hashes()

        # Subsystems
        self.watchdog = Watchdog(
            heartbeat_file=self.state_dir / "heartbeat.json",
            heartbeat_timeout_seconds=self.config["supervisor"].get("heartbeat_timeout_seconds", 45.0),
            max_runtime_per_objective_seconds=self.config["supervisor"].get("max_runtime_per_objective_seconds", 300.0),
            max_memory_mb=self.config["supervisor"].get("max_memory_mb", 1024.0),
            max_cpu_percent=self.config["supervisor"].get("max_cpu_percent", 95.0),
            logger=self.logger,
        )

        self.health_checker = HealthChecker(
            workspace_root=self.workspace_root,
            baseline_hashes=self.baseline_hashes,
            required_files=self.config.get("health_check", {}).get("required_files"),
            required_packages=self.config.get("health_check", {}).get("required_packages"),
            logger=self.logger,
        )

        self.auto_healer = AutoHealer(
            workspace_root=self.workspace_root,
            max_freeze_seconds=self.config.get("auto_healer", {}).get("max_freeze_seconds", 60.0),
            max_objective_age_seconds=self.config.get("auto_healer", {}).get("max_objective_age_seconds", 180.0),
            stagnation_cycle_limit=self.config.get("auto_healer", {}).get("stagnation_cycle_limit", 3),
            logger=self.logger,
        )

        # State tracking
        self.agent_process: Optional[subprocess.Popen] = None
        self.agent_psutil_proc: Optional[psutil.Process] = None
        self.consecutive_failures = 0
        self.max_consecutive_failures = self.config["supervisor"].get("max_consecutive_failures", 3)
        self.kill_switch_file = self.workspace_root / self.config["supervisor"].get(
            "kill_switch_file", "system/kill_switch.flag"
        )

        self._ensure_known_good_version()

        # Integrated Telegram Bot listener
        self.telegram_bot = None
        self._init_telegram_bot()

    def _init_telegram_bot(self) -> None:
        """Launches Telegram Bot monitoring daemon if TELEGRAM_BOT_TOKEN is configured."""
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        if token:
            try:
                from system.telegram_bot import TelegramBot
                self.telegram_bot = TelegramBot(workspace_root=self.workspace_root)
                self.telegram_bot.start_in_background()
                self.logger.info("Integrated Telegram Bot daemon started successfully.")
            except Exception as e:
                self.logger.warning(f"Could not launch Telegram Bot daemon: {e}")

    def _load_config(self) -> Dict[str, Any]:
        """Loads supervisor config from YAML."""
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {
            "supervisor": {
                "check_interval_seconds": 2,
                "heartbeat_timeout_seconds": 45,
                "max_runtime_per_objective_seconds": 300,
                "max_consecutive_failures": 3,
                "max_memory_mb": 1024,
                "max_cpu_percent": 95.0,
                "kill_switch_file": "system/kill_switch.flag",
                "agent_entrypoint": "agent/main.py",
            }
        }

    def _setup_logging(self) -> None:
        """Sets up supervisor-specific logger with rotation."""
        self.logger = logging.getLogger("Supervisor")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [SUPERVISOR] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            # Console handler
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

            # File handler
            log_file = self.logs_dir / "supervisor.log"
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

            # Error handler
            err_file = self.logs_dir / "errors.log"
            eh = logging.FileHandler(err_file, encoding="utf-8")
            eh.setLevel(logging.ERROR)
            eh.setFormatter(formatter)
            self.logger.addHandler(eh)

    def _init_baseline_hashes(self) -> Dict[str, str]:
        """Initializes or loads supervisor integrity baseline hashes."""
        if self.baseline_hash_file.exists():
            try:
                with open(self.baseline_hash_file, "r", encoding="utf-8") as f:
                    hashes = json.load(f)
                    self.logger.info(f"Loaded {len(hashes)} supervisor baseline hashes.")
                    return hashes
            except Exception as e:
                self.logger.warning(f"Failed to read baseline hashes, regenerating: {e}")

        hashes = save_baseline_hashes(self.system_dir, self.baseline_hash_file)
        self.logger.info(f"Generated new baseline hashes for {len(hashes)} supervisor files.")
        return hashes

    def get_git_commit(self) -> Optional[str]:
        """Returns the current Git HEAD commit hash."""
        try:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                check=True,
            )
            return res.stdout.strip()
        except Exception:
            return None

    def _ensure_known_good_version(self) -> None:
        """Ensures state/known_good_version.json exists with the initial clean commit."""
        kg_file = self.state_dir / "known_good_version.json"
        commit = self.get_git_commit()
        if not kg_file.exists() and commit:
            data = {
                "commit": commit,
                "timestamp": time.time(),
                "description": "Initial baseline commit",
                "verified": True,
            }
            with open(kg_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self.logger.info(f"Established initial known-good version: {commit[:8]}")

    def update_known_good_version(self, commit: str, description: str) -> None:
        """Records a newly verified commit as the known-good version."""
        kg_file = self.state_dir / "known_good_version.json"
        data = {
            "commit": commit,
            "timestamp": time.time(),
            "description": description,
            "verified": True,
        }
        with open(kg_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self.logger.info(f"Known-good version updated to commit {commit[:8]} ({description})")

    def rollback_to_known_good(self, reason: str) -> bool:
        """
        Rolls back the workspace to the last verified known-good Git commit.
        Protects logs/ and state/ from being deleted during rollback.
        """
        if self.unsecured:
            self.logger.warning(f"Unsecured mode active: Rollback bypassed for reason '{reason}'.")
            return True

        kg_file = self.state_dir / "known_good_version.json"
        if not kg_file.exists():
            self.logger.error("Rollback failed: No known-good version file found.")
            return False


        try:
            with open(kg_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                target_commit = data.get("commit")

            if not target_commit:
                self.logger.error("Rollback failed: Empty commit in known-good record.")
                return False

            self.logger.warning(
                f"EXECUTING EMERGENCY ROLLBACK to commit {target_commit[:8]}! Reason: {reason}"
            )

            # Reset workspace git
            subprocess.run(
                ["git", "reset", "--hard", target_commit],
                cwd=self.workspace_root,
                check=True,
                capture_output=True,
            )

            # Clean untracked files in agent/ and project/
            subprocess.run(
                ["git", "clean", "-fd", "--", "agent", "project"],
                cwd=self.workspace_root,
                check=True,
                capture_output=True,
            )

            # Verify supervisor integrity after rollback
            valid, violations = verify_integrity(self.system_dir, self.baseline_hashes)
            if not valid:
                self.logger.critical(f"Supervisor corrupted after rollback: {violations}")

            # Record incident in state/lessons.json
            self._record_rollback_incident(reason, target_commit)
            self.logger.info(f"Successfully rolled back workspace to {target_commit[:8]}")
            return True

        except Exception as e:
            self.logger.critical(f"Fatal error during rollback execution: {e}", exc_info=True)
            return False

    def _record_rollback_incident(self, reason: str, target_commit: str) -> None:
        """Logs rollback incident into state/lessons.json."""
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
            "event": "SUPERVISOR_ROLLBACK",
            "reason": reason,
            "restored_commit": target_commit,
            "lesson": f"Failure triggered supervisor rollback. Avoid repeating code pattern: {reason}",
        })

        try:
            with open(lessons_file, "w", encoding="utf-8") as f:
                json.dump(lessons, f, indent=2)
        except Exception:
            pass

    def check_kill_switch(self) -> bool:
        """Checks if the emergency kill switch has been activated."""
        return self.kill_switch_file.exists()

    def activate_kill_switch(self) -> None:
        """Creates the kill switch file to order immediate system halt."""
        self.kill_switch_file.write_text("HALT", encoding="utf-8")
        self.logger.warning("Kill switch manually triggered.")

    def deactivate_kill_switch(self) -> None:
        """Removes the kill switch file."""
        if self.kill_switch_file.exists():
            self.kill_switch_file.unlink()

    def start_agent(self) -> bool:
        """Launches the autonomous agent as a managed subprocess."""
        if self.check_kill_switch():
            self.logger.warning("Cannot start agent: Kill switch is active.")
            return False

        entrypoint = self.workspace_root / self.config["supervisor"].get("agent_entrypoint", "agent/main.py")
        if not entrypoint.exists():
            self.logger.error(f"Agent entrypoint not found: {entrypoint}")
            return False

        # Validate python syntax of agent files before running
        agent_dir = self.workspace_root / "agent"
        for py_file in agent_dir.glob("*.py"):
            try:
                py_compile.compile(str(py_file), doraise=True)
            except py_compile.PyCompileError as e:
                self.logger.error(f"Syntax error in agent file {py_file.name}: {e}")
                self.rollback_to_known_good(f"Syntax error in {py_file.name}")
                return False

        self.logger.info("Spawning autonomous agent subprocess...")
        try:
            # Set unbuffered Python execution
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["SUPERVISOR_ACTIVE"] = "1"

            cmd = [sys.executable, "-u", str(entrypoint)]
            if self.unsecured:
                cmd.append("--unsecured")
                env["AGENT_UNSECURED"] = "1"

            self.agent_process = subprocess.Popen(
                cmd,
                cwd=self.workspace_root,
                env=env,
            )

            self.agent_psutil_proc = psutil.Process(self.agent_process.pid)
            self.watchdog.reset_objective_timer()
            self.logger.info(f"Agent running under PID {self.agent_process.pid}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start agent subprocess: {e}", exc_info=True)
            return False

    def stop_agent(self, reason: str = "Normal shutdown") -> None:
        """Gracefully terminates the agent and all its child processes."""
        if self.agent_process is not None:
            self.logger.info(f"Stopping agent (PID: {self.agent_process.pid}). Reason: {reason}")
            self.watchdog.terminate_process_tree(self.agent_psutil_proc)
            self.agent_process = None
            self.agent_psutil_proc = None

    def check_restart_request(self) -> Optional[Dict[str, Any]]:
        """Checks if agent requested a restart via state/restart_signal.json."""
        signal_file = self.state_dir / "restart_signal.json"
        if signal_file.exists():
            try:
                with open(signal_file, "r", encoding="utf-8") as f:
                    signal = json.load(f)
                return signal
            except Exception as e:
                self.logger.warning(f"Corrupted restart signal: {e}")
        return None

    def clear_restart_request(self) -> None:
        """Removes the restart signal file."""
        signal_file = self.state_dir / "restart_signal.json"
        if signal_file.exists():
            try:
                signal_file.unlink()
            except Exception:
                pass

    def handle_restart_request(self, signal: Dict[str, Any]) -> None:
        """
        Executes safe self-restart protocol:
        1. Validate syntax of modified files.
        2. Terminate running agent.
        3. Clear restart signal.
        4. Launch agent with persisted state.
        5. Verify health.
        """
        reason = signal.get("reason", "Agent requested restart")
        self.logger.info(f"Handling agent restart request: {reason}")

        # 1. Stop current agent
        self.stop_agent("Restart requested by agent")

        # 2. Compile check
        agent_dir = self.workspace_root / "agent"
        syntax_ok = True
        for py_file in agent_dir.glob("*.py"):
            try:
                py_compile.compile(str(py_file), doraise=True)
            except py_compile.PyCompileError as e:
                self.logger.error(f"Syntax validation failed on modified file {py_file.name}: {e}")
                syntax_ok = False
                break

        if not syntax_ok:
            self.logger.warning("Modified code failed syntax check. Rolling back to known-good!")
            self.rollback_to_known_good("Syntax error in modified agent code during restart")
            self.clear_restart_request()
            self.start_agent()
            return

        # 3. Clear signal
        self.clear_restart_request()

        # 4. Start agent
        if not self.start_agent():
            self.logger.error("Failed to start agent after restart request. Rolling back...")
            self.rollback_to_known_good("Agent failed to start after self-modification")
            self.start_agent()
            return

        # 5. Wait for agent initialization and verify health
        time.sleep(3)
        health = self.health_checker.run_all_checks(self.agent_psutil_proc)
        if health.is_failed():
            self.logger.error(f"Health check failed after restart: {health.failures}. Rolling back!")
            self.stop_agent("Post-restart health check failed")
            self.rollback_to_known_good("Post-restart health check failure")
            self.start_agent()
        else:
            self.logger.info(f"Agent successfully restarted and verified. Health: {health.status}")

    def run(self, max_cycles: Optional[int] = None) -> None:
        """
        Master Supervisor execution loop.
        Monitors watchdog, health checks, self-restarts, crashes, and kill switch.
        """
        self.logger.info("Starting Protected Supervisor (Layer A)...")
        if not self.start_agent():
            self.logger.critical("Could not initialize agent. Supervisor halting.")
            return

        cycle = 0
        poll_interval = self.config["supervisor"].get("check_interval_seconds", 2)

        try:
            while True:
                if max_cycles and cycle >= max_cycles:
                    self.logger.info(f"Reached specified supervisor max cycles ({max_cycles}). Stopping.")
                    break

                cycle += 1

                # 1. Check emergency kill switch
                if self.check_kill_switch():
                    self.logger.warning("Kill switch detected. Terminating agent and exiting supervisor.")
                    self.stop_agent("Kill switch active")
                    break

                # 2. Check for self-restart request
                restart_signal = self.check_restart_request()
                if restart_signal:
                    self.handle_restart_request(restart_signal)
                    time.sleep(poll_interval)
                    continue

                # 3. Evaluate process watchdog
                watchdog_status = self.watchdog.check(self.agent_psutil_proc)
                if not watchdog_status.is_healthy:
                    self.logger.error(
                        f"Watchdog trigger: {watchdog_status.status} - {watchdog_status.details}"
                    )
                    self.consecutive_failures += 1

                    self.stop_agent(f"Watchdog failure: {watchdog_status.status}")

                    if self.consecutive_failures >= self.max_consecutive_failures:
                        self.logger.critical(
                            f"Circuit breaker tripped: {self.consecutive_failures} consecutive failures! "
                            "Performing emergency rollback to known-good version."
                        )
                        self.rollback_to_known_good("Circuit breaker tripped: Repeated failures")
                        self.consecutive_failures = 0
                        time.sleep(5)  # Cooldown

                    self.logger.info("Restarting agent process...")
                    self.start_agent()
                    time.sleep(poll_interval)
                    continue

                # If healthy, reset failure counter
                if self.consecutive_failures > 0:
                    self.consecutive_failures = 0

                # 4. Periodic health check (every 5 cycles)
                if cycle % 5 == 0:
                    health = self.health_checker.run_all_checks(self.agent_psutil_proc)
                    if health.is_failed():
                        self.logger.error(f"Health check failed: {health.failures}")
                        self.stop_agent("Health check failure")
                        self.rollback_to_known_good(f"Health check failed: {'; '.join(health.failures)}")
                        self.start_agent()

                # 5. Periodic auto-healer inspection and remediation (every 3 cycles)
                if cycle % 3 == 0:
                    self.auto_healer.check_and_heal(self.agent_psutil_proc, supervisor=self)

                time.sleep(poll_interval)


        except KeyboardInterrupt:
            self.logger.info("Supervisor received interrupt signal (Ctrl+C). Cleaning up...")
        finally:
            self.stop_agent("Supervisor exiting")
            self.logger.info("Supervisor shut down cleanly.")


if __name__ == "__main__":
    supervisor = Supervisor()
    supervisor.run()
