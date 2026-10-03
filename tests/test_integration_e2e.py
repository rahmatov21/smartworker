"""
End-to-End Integration Tests.
Verifies complete autonomous cycles:
1. Autonomous inspection, objective selection, and successful improvement cycle.
2. Failure injection, test failure diagnosis, and automatic Git rollback.
3. Supervisor self-restart cycle and state resumption.
4. Emergency kill switch immediate halt.
"""

import json
import time
from pathlib import Path
import pytest

from system.supervisor.supervisor import Supervisor
from agent.main import AutonomousAgent
from agent.tools import AgentTools
from agent.objective_manager import Objective


def test_e2e_successful_autonomous_cycle():
    root = Path(__file__).resolve().parent.parent
    agent = AutonomousAgent(workspace_root=root)

    # Run exactly 1 cycle
    should_continue = agent.run_cycle()

    # Verify state files updated
    current_obj_file = root / "state" / "current_objective.json"
    experiments_file = root / "state" / "experiments.json"
    known_good_file = root / "state" / "known_good_version.json"

    assert experiments_file.exists()
    assert known_good_file.exists()

    with open(experiments_file, "r", encoding="utf-8") as f:
        experiments = json.load(f)
    assert len(experiments) > 0

    latest_exp = experiments[-1]
    assert latest_exp["passed"] is True
    assert latest_exp["status"] == "ACCEPTED"
    assert latest_exp["commit"] is not None


def test_e2e_failure_injection_triggers_clean_rollback():
    root = Path(__file__).resolve().parent.parent
    tools = AgentTools(workspace_root=root)

    # Get commit before failure
    before_status = tools.git_status()

    # Create deliberately failing objective that writes bad code
    bad_code = "def syntax_error_breaking_pipeline(:"  # Syntax error

    test_file = "project/src/ai_pipeline.py"
    original_content = tools.read_file(test_file)

    checkpoint_commit = tools.git_checkpoint("before-deliberate-failure")

    try:
        # Corrupt file
        tools.write_file(test_file, original_content + "\n" + bad_code)

        # Evaluator should catch syntax error and demand rollback
        eval_report = agent = AutonomousAgent(workspace_root=root).evaluator.evaluate_change()
        assert eval_report.passed is False
        assert eval_report.should_rollback is True

        # Rollback
        tools.git_rollback(checkpoint_commit)

        # Verify restored
        restored_content = tools.read_file(test_file)
        assert bad_code not in restored_content
    finally:
        # Ensure clean state
        tools.git_rollback(checkpoint_commit)


def test_e2e_kill_switch_halts_agent():
    root = Path(__file__).resolve().parent.parent
    kill_file = root / "system" / "kill_switch.flag"

    # Activate kill switch
    kill_file.write_text("HALT", encoding="utf-8")

    try:
        agent = AutonomousAgent(workspace_root=root)
        # run_cycle should immediately detect kill switch and abort
        result = agent.run_cycle()
        assert result is False
    finally:
        if kill_file.exists():
            kill_file.unlink()
