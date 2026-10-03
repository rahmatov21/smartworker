"""
Autonomous Self-Improving AI Coding Agent Entry Point (Layer B).
Runs the continuous improvement loop:
INSPECT -> OBJECTIVE -> PLAN -> MODIFY -> TEST -> EVALUATE -> ACCEPT/ROLLBACK -> RECORD.
Maintains continuous heartbeat telemetry for Supervisor watchdog.
"""

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

# Ensure workspace root is in sys.path when executed directly as script
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

try:
    from .evaluator import Evaluator
    from .llm_client import OpenRouterClient
    from .memory import AgentMemory
    from .objective_manager import ObjectiveManager
    from .planner import Planner
    from .tools import AgentTools
except (ImportError, ValueError):
    from agent.evaluator import Evaluator
    from agent.llm_client import OpenRouterClient
    from agent.memory import AgentMemory
    from agent.objective_manager import ObjectiveManager
    from agent.planner import Planner
    from agent.tools import AgentTools



class AutonomousAgent:
    def __init__(
        self,
        workspace_root: Optional[Path] = None,
        config_path: Optional[Path] = None,
        unsecured: bool = False,
    ):
        self.workspace_root = (workspace_root or Path(__file__).resolve().parent.parent).resolve()
        self.unsecured = unsecured or os.environ.get("UNSECURED_MODE", "0") == "1"
        self.agent_dir = self.workspace_root / "agent"
        self.state_dir = self.workspace_root / "state"
        self.logs_dir = self.workspace_root / "logs"

        for d in [self.state_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.config_path = config_path or (self.agent_dir / "agent_config.yaml")
        self.config = self._load_config()

        self._setup_logging()

        if self.unsecured:
            self.logger.warning(
                "AGENT INITIALIZED IN UNSECURED / UNRESTRICTED MODE! "
                "Rollbacks and Layer A protection constraints are disabled."
            )

        # Initialize subcomponents
        self.tools = AgentTools(self.workspace_root, unsecured=self.unsecured, logger=self.logger)
        self.llm = OpenRouterClient(
            model=self.config.get("llm", {}).get("model", "qwen/qwen-2.5-72b-instruct"),
            logger=self.logger,
        )
        self.memory = AgentMemory(self.workspace_root, logger=self.logger)
        self.evaluator = Evaluator(self.workspace_root, tools=self.tools, unsecured=self.unsecured, logger=self.logger)
        self.obj_manager = ObjectiveManager(
            workspace_root=self.workspace_root,
            llm_client=self.llm,
            memory=self.memory,
            tools=self.tools,
            logger=self.logger,
        )
        self.planner = Planner(
            workspace_root=self.workspace_root,
            tools=self.tools,
            evaluator=self.evaluator,
            memory=self.memory,
            llm=self.llm,
            max_fix_attempts=self.config.get("agent", {}).get("max_fix_attempts", 2),
            unsecured=self.unsecured,
            logger=self.logger,
        )

        self.heartbeat_file = self.state_dir / "heartbeat.json"
        self.kill_switch_file = self.workspace_root / "system" / "kill_switch.flag"

        self.is_running = False
        self._heartbeat_thread: Optional[threading.Thread] = None

    def _load_config(self) -> Dict[str, Any]:
        """Loads agent configuration from YAML."""
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _setup_logging(self) -> None:
        """Configures agent-specific logger."""
        self.logger = logging.getLogger("Agent")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [AGENT] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

            fh = logging.FileHandler(self.logs_dir / "agent.log", encoding="utf-8")
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def _emit_heartbeat(self, current_task: str = "idle") -> None:
        """Writes timestamped heartbeat to state/heartbeat.json for Supervisor Watchdog."""
        data = {
            "timestamp": time.time(),
            "pid": os.getpid(),
            "task": current_task,
            "status": "healthy",
        }
        try:
            with open(self.heartbeat_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to write heartbeat: {e}")

    def _start_heartbeat_loop(self) -> None:
        """Starts asynchronous heartbeat emitter thread."""
        def run():
            interval = self.config.get("agent", {}).get("heartbeat_interval_seconds", 5)
            while self.is_running:
                self._emit_heartbeat(current_task="active_cycle")
                time.sleep(interval)

        self._heartbeat_thread = threading.Thread(target=run, daemon=True)
        self._heartbeat_thread.start()

    def check_kill_switch(self) -> bool:
        """Returns True if the kill switch flag exists."""
        return self.kill_switch_file.exists()

    def run_cycle(self) -> bool:
        """
        Executes a single autonomous self-improvement cycle:
        1. Inspect current state & project
        2. Generate & select next objective
        3. Formulate plan
        4. Execute with checkpoints and test-before-acceptance
        5. Record results
        Returns True if agent should continue, False if restart requested or halted.
        """
        if self.check_kill_switch():
            self.logger.warning("Kill switch detected. Terminating cycle.")
            return False

        mission = self.config.get("agent", {}).get("mission", "Continuously improve this AI system.")
        self.logger.info(f"=== Beginning Autonomous Improvement Cycle | Mission: '{mission}' ===")

        # 1. Objective Selection
        self._emit_heartbeat("selecting_objective")
        objective = self.obj_manager.select_next_objective(mission)

        # 2. Plan & Execute Objective
        self._emit_heartbeat(f"executing_objective_{objective.id}")
        result = self.planner.execute_objective(objective)

        # 3. Record Outcome
        self.obj_manager.record_objective_completion(objective, success=result.success)

        self.logger.info(
            f"Cycle finished. Success: {result.success} | Rolled back: {result.rolled_back} | "
            f"Restart required: {result.restart_required}"
        )

        # 4. Handle self-restart if required
        if result.restart_required:
            self.logger.info("Self-restart signaled. Exiting agent process cleanly to allow Supervisor restart.")
            return False

        return True

    def run(self, max_cycles: Optional[int] = None) -> None:
        """Runs the continuous autonomous loop indefinitely."""
        self.is_running = True
        self._start_heartbeat_loop()
        self.logger.info(f"Autonomous Agent online (PID: {os.getpid()}).")

        cycle_count = 0
        cooldown = self.config.get("agent", {}).get("cooldown_between_objectives_seconds", 3)

        try:
            while self.is_running:
                if max_cycles and cycle_count >= max_cycles:
                    self.logger.info(f"Completed {cycle_count} cycles. Exiting.")
                    break

                cycle_count += 1
                should_continue = self.run_cycle()
                if not should_continue:
                    break

                time.sleep(cooldown)

        except KeyboardInterrupt:
            self.logger.info("Agent received interrupt signal. Shutting down...")
        finally:
            self.is_running = False
            self.logger.info("Agent shutdown complete.")


if __name__ == "__main__":
    agent = AutonomousAgent()
    agent.run()
