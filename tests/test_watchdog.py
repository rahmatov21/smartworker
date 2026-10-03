"""
Unit tests for Supervisor Watchdog & Resource Monitor.
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock
import pytest
import psutil

from system.supervisor.watchdog import Watchdog, WatchdogStatus


def test_watchdog_heartbeat_fresh(tmp_path):
    hb_file = tmp_path / "heartbeat.json"
    with open(hb_file, "w") as f:
        json.dump({"timestamp": time.time(), "status": "running"}, f)

    watchdog = Watchdog(heartbeat_file=hb_file, heartbeat_timeout_seconds=30.0)
    age = watchdog.get_heartbeat_age()
    assert age < 5.0


def test_watchdog_heartbeat_expired(tmp_path):
    hb_file = tmp_path / "heartbeat.json"
    with open(hb_file, "w") as f:
        json.dump({"timestamp": time.time() - 100.0, "status": "running"}, f)

    watchdog = Watchdog(heartbeat_file=hb_file, heartbeat_timeout_seconds=30.0)
    age = watchdog.get_heartbeat_age()
    assert age >= 99.0

    # Mock running process
    mock_proc = MagicMock(spec=psutil.Process)
    mock_proc.pid = 1234
    mock_proc.is_running.return_value = True
    mock_proc.memory_info.return_value.rss = 10 * 1024 * 1024
    mock_proc.cpu_percent.return_value = 5.0
    mock_proc.children.return_value = []

    status = watchdog.check(mock_proc)
    assert status.status == "HEARTBEAT_TIMEOUT"
    assert status.is_healthy is False


def test_watchdog_dead_process(tmp_path):
    hb_file = tmp_path / "heartbeat.json"
    watchdog = Watchdog(heartbeat_file=hb_file)

    mock_proc = MagicMock(spec=psutil.Process)
    mock_proc.pid = 5678
    mock_proc.is_running.return_value = False

    status = watchdog.check(mock_proc)
    assert status.status == "DEAD"
    assert status.is_healthy is False


def test_watchdog_oom_detection(tmp_path):
    hb_file = tmp_path / "heartbeat.json"
    with open(hb_file, "w") as f:
        json.dump({"timestamp": time.time()}, f)

    watchdog = Watchdog(heartbeat_file=hb_file, max_memory_mb=500.0)

    mock_proc = MagicMock(spec=psutil.Process)
    mock_proc.pid = 9999
    mock_proc.is_running.return_value = True
    # 600 MB usage
    mock_proc.memory_info.return_value.rss = 600 * 1024 * 1024
    mock_proc.cpu_percent.return_value = 10.0
    mock_proc.children.return_value = []

    status = watchdog.check(mock_proc)
    assert status.status == "OOM"
    assert status.is_healthy is False
