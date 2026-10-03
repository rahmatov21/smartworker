"""
Unit tests for Auto-Healer & Progress Monitor Service (Layer A).
Verifies detection and automatic remediation of frozen processes,
stalled objectives, metric plateaus, git locks, and corrupted state files.
"""

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock
import pytest
import psutil

from system.supervisor.auto_healer import AutoHealer, HealingAction, DiagnosisReport


def test_auto_healer_diagnose_healthy(tmp_path):
    # Set up clean workspace
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    hb = state_dir / "heartbeat.json"
    with open(hb, "w") as f:
        json.dump({"timestamp": time.time(), "status": "running"}, f)

    healer = AutoHealer(workspace_root=tmp_path, max_freeze_seconds=60.0)
    report = healer.diagnose()
    assert report.is_healthy is True
    assert len(report.stuck_issues) == 0


def test_auto_healer_detects_and_clears_git_lock(tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    lock_file = git_dir / "index.lock"
    lock_file.write_text("locked", encoding="utf-8")

    healer = AutoHealer(workspace_root=tmp_path)
    report = healer.diagnose()
    assert "GIT_INDEX_LOCKED" in report.stuck_issues

    actions = healer.heal(report)
    assert any(a.issue_type == "GIT_INDEX_LOCKED" and a.success for a in actions)
    assert not lock_file.exists()


def test_auto_healer_detects_and_abandons_stalled_objective(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    current_obj = state_dir / "current_objective.json"
    stuck_obj_data = {
        "id": "stalled-obj-999",
        "title": "Stuck Infinite Loop Task",
        "status": "in_progress",
    }
    with open(current_obj, "w") as f:
        json.dump(stuck_obj_data, f)

    # Set mtime to 300 seconds ago
    old_time = time.time() - 300
    os.utime(current_obj, (old_time, old_time))

    healer = AutoHealer(workspace_root=tmp_path, max_objective_age_seconds=100.0)
    report = healer.diagnose()
    assert "OBJECTIVE_STALLED" in report.stuck_issues

    actions = healer.heal(report)
    assert any(a.issue_type == "OBJECTIVE_STALLED" and a.success for a in actions)

    # current_objective should be cleared
    assert not current_obj.exists()

    # archived in objectives.json
    history_file = state_dir / "objectives.json"
    assert history_file.exists()
    with open(history_file, "r") as f:
        history = json.load(f)
    assert any("stalled_abandoned" in o.get("status", "") for o in history.get("failed", []))


def test_auto_healer_breaks_metric_plateau(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    exp_file = state_dir / "experiments.json"

    # 3 consecutive failed experiments
    stagnant_exps = [
        {"id": f"exp-0{i}", "passed": False, "metrics_after": {"tests_passed": 5}}
        for i in range(3)
    ]
    with open(exp_file, "w") as f:
        json.dump(stagnant_exps, f)

    healer = AutoHealer(workspace_root=tmp_path, stagnation_cycle_limit=3)
    report = healer.diagnose()
    assert "REPEATED_EXPERIMENT_FAILURES" in report.stuck_issues

    actions = healer.heal(report)
    assert any("FAILURES" in a.issue_type and a.success for a in actions)

    # Verify divergent directive was injected
    directive_file = state_dir / "divergent_directive.json"
    assert directive_file.exists()
    with open(directive_file, "r") as f:
        directive = json.load(f)
    assert directive.get("active") is True
    assert "STAGNATION_BREAKER" in directive.get("directive", "")


def test_auto_healer_repairs_corrupted_state_file(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    metrics_file = state_dir / "metrics.json"
    metrics_file.write_text("{ corrupt json !!", encoding="utf-8")

    healer = AutoHealer(workspace_root=tmp_path)
    report = healer.diagnose()
    assert any("CORRUPT_STATE_FILE_metrics.json" in issue for issue in report.stuck_issues)

    actions = healer.heal(report)
    assert any(a.issue_type == "STATE_CORRUPTION" and a.success for a in actions)

    # File should now be valid JSON
    with open(metrics_file, "r") as f:
        data = json.load(f)
    assert "history" in data


def test_auto_healer_heals_frozen_process(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    hb = state_dir / "heartbeat.json"
    # Heartbeat from 120s ago
    with open(hb, "w") as f:
        json.dump({"timestamp": time.time() - 120.0, "status": "running"}, f)

    mock_proc = MagicMock(spec=psutil.Process)
    mock_proc.pid = 9876
    mock_proc.is_running.return_value = True
    mock_proc.children.return_value = []

    mock_supervisor = MagicMock()

    healer = AutoHealer(workspace_root=tmp_path, max_freeze_seconds=60.0)
    report = healer.diagnose(agent_proc=mock_proc)
    assert "PROCESS_HEARTBEAT_FROZEN" in report.stuck_issues

    actions = healer.heal(report, agent_proc=mock_proc, supervisor=mock_supervisor)
    assert any(a.issue_type == "PROCESS_FROZEN" and a.success for a in actions)

    # Process was killed and supervisor restarted
    mock_proc.kill.assert_called_once()
    mock_supervisor.stop_agent.assert_called_once()
    mock_supervisor.start_agent.assert_called_once()
