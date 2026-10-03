"""
Unit tests for Agent Sandboxed Tools and Protection Enforcement.
"""

from pathlib import Path
import pytest

from agent.tools import AgentTools
from system.supervisor.protection import ProtectionError


def test_agent_tools_read_write_project_allowed(tmp_path):
    root = Path(__file__).resolve().parent.parent
    tools = AgentTools(workspace_root=root)

    test_file = "project/test_scratch.tmp"
    tools.write_file(test_file, "sandbox content verification")
    content = tools.read_file(test_file)
    assert content == "sandbox content verification"
    tools.delete_file(test_file)


def test_agent_tools_write_to_supervisor_strictly_forbidden():
    root = Path(__file__).resolve().parent.parent
    tools = AgentTools(workspace_root=root)

    with pytest.raises(ProtectionError) as exc_info:
        tools.write_file("system/supervisor/supervisor.py", "# malicious overwrite")
    assert "SECURITY VIOLATION" in str(exc_info.value)


def test_agent_tools_delete_supervisor_strictly_forbidden():
    root = Path(__file__).resolve().parent.parent
    tools = AgentTools(workspace_root=root)

    with pytest.raises(ProtectionError) as exc_info:
        tools.delete_file("system/supervisor/config.yaml")
    assert "SECURITY VIOLATION" in str(exc_info.value)


def test_agent_tools_dangerous_command_blocked():
    root = Path(__file__).resolve().parent.parent
    tools = AgentTools(workspace_root=root)

    with pytest.raises(ProtectionError):
        tools.run_command("del /f system\\supervisor\\supervisor.py")


def test_agent_tools_run_tests_parses_results():
    root = Path(__file__).resolve().parent.parent
    tools = AgentTools(workspace_root=root)

    res = tools.run_tests("project/tests")
    assert res["success"] is True
    assert res["passed"] >= 5
    assert res["failed"] == 0
