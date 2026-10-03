"""
Unit tests for Objective Manager, Planner, and Rollback Engine.
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from agent.objective_manager import ObjectiveManager, Objective
from agent.planner import Planner
from agent.evaluator import Evaluator, EvaluationReport
from agent.memory import AgentMemory
from agent.tools import AgentTools


def test_objective_prioritization_scoring():
    mgr = ObjectiveManager()
    high_priority = mgr.calculate_priority_score(
        impact=9.0, feasibility=9.0, safety=9.0, testability=9.0, alignment=9.0, redundancy_penalty=0.0
    )
    low_priority = mgr.calculate_priority_score(
        impact=4.0, feasibility=4.0, safety=4.0, testability=4.0, alignment=4.0, redundancy_penalty=3.0
    )
    assert high_priority > low_priority
    assert low_priority >= 0.0


def test_redundancy_penalty_lowers_priority():
    mgr = ObjectiveManager()
    clean_score = mgr.calculate_priority_score(8.0, 8.0, 8.0, 8.0, 8.0, redundancy_penalty=0.0)
    penalized_score = mgr.calculate_priority_score(8.0, 8.0, 8.0, 8.0, 8.0, redundancy_penalty=4.0)
    assert penalized_score == clean_score - 4.0


def test_planner_rollback_on_test_failure():
    root = Path(__file__).resolve().parent.parent
    tools = AgentTools(workspace_root=root)
    memory = AgentMemory(workspace_root=root)

    # Mock evaluator that reports failure
    mock_evaluator = MagicMock(spec=Evaluator)
    failing_report = EvaluationReport(
        passed=False,
        should_rollback=True,
        tests_passed=0,
        tests_failed=2,
        tests_errors=0,
        test_duration=0.5,
        integrity_ok=True,
        failure_reason="Unit tests crashed",
        diagnostics=["AssertionError: expected 10 got 20"],
    )
    mock_evaluator.evaluate_change.return_value = failing_report

    planner = Planner(
        workspace_root=root,
        tools=tools,
        evaluator=mock_evaluator,
        memory=memory,
        max_fix_attempts=1,
    )

    test_obj = Objective(
        id="test-obj-fail-01",
        title="Test Objective Destined to Fail",
        category="testing",
        hypothesis="Testing failure handling",
        target_files=["project/tests/test_ai_pipeline.py"],
        impact_score=5.0,
        feasibility_score=5.0,
        safety_score=5.0,
        testability_score=5.0,
        alignment_score=5.0,
        redundancy_penalty=0.0,
        priority_score=5.0,
        requires_agent_restart=False,
        status="in_progress",
    )

    result = planner.execute_objective(test_obj)

    # Assertions
    assert result.success is False
    assert result.rolled_back is True
    assert result.final_commit is None
    assert "Rolled back" in result.details


def test_planner_accept_on_test_success():
    root = Path(__file__).resolve().parent.parent
    tools = AgentTools(workspace_root=root)
    memory = AgentMemory(workspace_root=root)

    # Mock evaluator that reports success
    mock_evaluator = MagicMock(spec=Evaluator)
    passing_report = EvaluationReport(
        passed=True,
        should_rollback=False,
        tests_passed=5,
        tests_failed=0,
        tests_errors=0,
        test_duration=0.2,
        integrity_ok=True,
        benchmark_metrics={"avg_latency_ms": 1.5},
        failure_reason=None,
    )
    mock_evaluator.evaluate_change.return_value = passing_report

    planner = Planner(
        workspace_root=root,
        tools=tools,
        evaluator=mock_evaluator,
        memory=memory,
        max_fix_attempts=1,
    )

    test_obj = Objective(
        id="test-obj-success-01",
        title="Test Objective Destined to Succeed",
        category="testing",
        hypothesis="Testing success path",
        target_files=["project/tests/test_ai_pipeline.py"],
        impact_score=8.0,
        feasibility_score=8.0,
        safety_score=8.0,
        testability_score=8.0,
        alignment_score=8.0,
        redundancy_penalty=0.0,
        priority_score=8.0,
        requires_agent_restart=False,
        status="in_progress",
    )

    result = planner.execute_objective(test_obj)

    # Assertions
    assert result.success is True
    assert result.rolled_back is False
    assert result.final_commit is not None
    assert "Known-good version advanced" in result.details
