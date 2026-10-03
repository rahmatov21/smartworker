"""
Unit tests for Health Check System (Layer A).
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest
import psutil

from system.supervisor.protection import compute_supervisor_hashes
from system.supervisor.health_check import HealthChecker


def test_health_check_healthy_status():
    root = Path(__file__).resolve().parent.parent
    system_dir = root / "system"
    baseline = compute_supervisor_hashes(system_dir)

    checker = HealthChecker(workspace_root=root, baseline_hashes=baseline)

    # Mock running process
    mock_proc = MagicMock(spec=psutil.Process)
    mock_proc.pid = 4321
    mock_proc.is_running.return_value = True
    mock_proc.status.return_value = psutil.STATUS_RUNNING
    mock_proc.memory_info.return_value.rss = 50 * 1024 * 1024
    mock_proc.children.return_value = []

    status = checker.run_all_checks(agent_proc=mock_proc, check_api=False)
    assert status.status in ("HEALTHY", "DEGRADED")
    assert status.is_failed() is False
    assert status.checks["supervisor_integrity"]["passed"] is True
    assert status.checks["required_files"]["passed"] is True
    assert status.checks["required_packages"]["passed"] is True


def test_health_check_missing_required_file_triggers_failed():
    root = Path(__file__).resolve().parent.parent
    baseline = {}

    checker = HealthChecker(
        workspace_root=root,
        baseline_hashes=baseline,
        required_files=["non_existent_file_xyz.py"],
    )

    status = checker.run_all_checks(agent_proc=None, check_api=False)
    assert status.is_failed() is True
    assert any("non_existent_file_xyz.py" in f for f in status.failures)
