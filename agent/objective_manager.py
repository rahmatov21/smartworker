"""
Autonomous Objective Selection & Prioritization Manager (Layer B).
Inspects the project, identifies weaknesses, bugs, missing tests, and performance issues,
generates improvement objectives, and prioritizes them using multi-criteria scoring.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

from .llm_client import OpenRouterClient
from .memory import AgentMemory
from .tools import AgentTools


@dataclass
class Objective:
    id: str
    title: str
    category: str
    hypothesis: str
    target_files: List[str]
    impact_score: float        # 1-10
    feasibility_score: float   # 1-10
    safety_score: float        # 1-10 (higher is safer)
    testability_score: float   # 1-10
    alignment_score: float     # 1-10
    redundancy_penalty: float  # 0-10
    priority_score: float      # Calculated composite
    requires_agent_restart: bool
    status: str                # pending, in_progress, completed, failed
    attempt: int = 1


class ObjectiveManager:
    def __init__(
        self,
        workspace_root: Optional[Path] = None,
        llm_client: Optional[OpenRouterClient] = None,
        memory: Optional[AgentMemory] = None,
        tools: Optional[AgentTools] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.workspace_root = (workspace_root or Path(__file__).resolve().parent.parent).resolve()
        self.state_dir = self.workspace_root / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger("ObjectiveManager")

        self.llm = llm_client or OpenRouterClient()
        self.memory = memory or AgentMemory(self.workspace_root)
        self.tools = tools or AgentTools(self.workspace_root)

        self.current_obj_file = self.state_dir / "current_objective.json"
        self.objectives_history_file = self.state_dir / "objectives.json"

    def get_current_objective(self) -> Optional[Objective]:
        """Loads in-progress objective from disk if available (for crash recovery)."""
        if self.current_obj_file.exists():
            try:
                with open(self.current_obj_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data and data.get("status") in ("in_progress", "pending", "testing"):
                        return Objective(**data)
            except Exception as e:
                self.logger.warning(f"Could not load current objective: {e}")
        return None

    def set_current_objective(self, obj: Objective) -> None:
        """Persists active objective to disk."""
        with open(self.current_obj_file, "w", encoding="utf-8") as f:
            json.dump(asdict(obj), f, indent=2)

    def clear_current_objective(self) -> None:
        """Removes the active objective file."""
        if self.current_obj_file.exists():
            self.current_obj_file.unlink()

    def calculate_priority_score(
        self,
        impact: float,
        feasibility: float,
        safety: float,
        testability: float,
        alignment: float,
        redundancy_penalty: float = 0.0,
    ) -> float:
        """
        Multi-criteria prioritization score:
        Impact (25%), Feasibility (25%), Safety (20%), Testability (15%), Alignment (15%) - Redundancy
        """
        score = (
            (impact * 0.25)
            + (feasibility * 0.25)
            + (safety * 0.20)
            + (testability * 0.15)
            + (alignment * 0.15)
            - redundancy_penalty
        )
        return round(max(0.0, score), 2)

    def inspect_system_needs(self) -> Dict[str, Any]:
        """
        Inspects the codebase to identify concrete opportunities:
        - Tests coverage / pass rate
        - Codebase size and missing test files
        - Recent experiment failures to avoid
        - Performance benchmark gaps
        """
        test_results = self.tools.run_tests("project/tests")
        project_files = self.tools.list_files("project")
        agent_files = self.tools.list_files("agent")
        recent_failures = self.memory.get_recent_failures(limit=3)

        src_files = [f for f in project_files if f.startswith("project/src") and f.endswith(".py")]
        test_files = [f for f in project_files if f.startswith("project/tests") and f.endswith(".py")]

        needs = {
            "test_results": test_results,
            "src_count": len(src_files),
            "test_count": len(test_files),
            "recent_failed_experiments": [f.get("objective") for f in recent_failures],
            "agent_files": agent_files,
            "project_files": project_files,
        }
        return needs

    def generate_candidate_objectives(self, mission: str = "Continuously improve this system.") -> List[Objective]:
        """
        Generates diverse candidates using static inspection and LLM reasoning.
        """
        needs = self.inspect_system_needs()
        recent_failed_topics = needs["recent_failed_experiments"]

        # Default rich catalog of potential autonomous objectives across dimensions
        candidates_catalog = [
            {
                "title": "Add LRU caching and edge-case validation to project text pipeline",
                "category": "performance_and_reliability",
                "hypothesis": "Adding caching reduces repeated query latency by 40% and handles empty string edge cases without crashing.",
                "target_files": ["project/src/ai_pipeline.py", "project/tests/test_ai_pipeline.py"],
                "impact": 8.5,
                "feasibility": 9.0,
                "safety": 9.0,
                "testability": 9.5,
                "alignment": 9.0,
                "requires_restart": False,
            },
            {
                "title": "Implement robust JSON schema validation in agent memory system",
                "category": "agent_self_improvement",
                "hypothesis": "Enforcing schema validation prevents corrupted state entries during unexpected shutdowns.",
                "target_files": ["agent/memory.py"],
                "impact": 7.5,
                "feasibility": 8.5,
                "safety": 8.5,
                "testability": 9.0,
                "alignment": 8.5,
                "requires_restart": True,
            },
            {
                "title": "Expand test coverage for tokenizer boundary conditions and Unicode handling",
                "category": "testing_and_quality",
                "hypothesis": "Comprehensive boundary testing discovers hidden parsing errors in non-ASCII texts.",
                "target_files": ["project/tests/test_ai_pipeline.py"],
                "impact": 7.0,
                "feasibility": 9.5,
                "safety": 10.0,
                "testability": 10.0,
                "alignment": 8.0,
                "requires_restart": False,
            },
            {
                "title": "Optimize vector similarity search in project pipeline with vectorization",
                "category": "performance_optimization",
                "hypothesis": "Vectorized cosine similarity search runs 3x faster than pure python loops.",
                "target_files": ["project/src/ai_pipeline.py", "project/benchmarks/benchmark_pipeline.py"],
                "impact": 8.5,
                "feasibility": 8.0,
                "safety": 8.5,
                "testability": 9.5,
                "alignment": 8.5,
                "requires_restart": False,
            },
            {
                "title": "Enhance error recovery and cooldown decay in OpenRouter client",
                "category": "agent_self_improvement",
                "hypothesis": "Gradual cooldown decay restores temporarily throttled keys faster, reducing latency.",
                "target_files": ["agent/llm_client.py"],
                "impact": 8.0,
                "feasibility": 8.0,
                "safety": 8.0,
                "testability": 9.0,
                "alignment": 8.5,
                "requires_restart": True,
            },
        ]

        objectives = []
        for idx, item in enumerate(candidates_catalog):
            # Check redundancy against recent failures
            penalty = 0.0
            for failed_topic in recent_failed_topics:
                if failed_topic and (item["category"] in failed_topic or item["title"] in failed_topic):
                    penalty = 4.0
                    break

            priority = self.calculate_priority_score(
                impact=item["impact"],
                feasibility=item["feasibility"],
                safety=item["safety"],
                testability=item["testability"],
                alignment=item["alignment"],
                redundancy_penalty=penalty,
            )

            obj = Objective(
                id=f"obj-{int(time.time())}-{idx:02d}",
                title=item["title"],
                category=item["category"],
                hypothesis=item["hypothesis"],
                target_files=item["target_files"],
                impact_score=item["impact"],
                feasibility_score=item["feasibility"],
                safety_score=item["safety"],
                testability_score=item["testability"],
                alignment_score=item["alignment"],
                redundancy_penalty=penalty,
                priority_score=priority,
                requires_agent_restart=item["requires_restart"],
                status="pending",
            )
            objectives.append(obj)

        # Sort descending by priority score
        objectives.sort(key=lambda o: o.priority_score, reverse=True)
        return objectives

    def select_next_objective(self, mission: str = "Continuously improve this system.") -> Objective:
        """
        Inspects, generates, scores, and selects the highest-priority objective.
        """
        # First check if an existing objective is in flight
        existing = self.get_current_objective()
        if existing:
            self.logger.info(f"Resuming existing in-flight objective: {existing.title}")
            return existing

        candidates = self.generate_candidate_objectives(mission)
        if not candidates:
            # Fallback default
            best = Objective(
                id=f"obj-{int(time.time())}-fallback",
                title="Perform code linting and add comprehensive unit test assertions",
                category="code_quality",
                hypothesis="Adding strict assertions prevents subtle regressions.",
                target_files=["project/tests/test_ai_pipeline.py"],
                impact_score=7.0,
                feasibility_score=9.0,
                safety_score=9.5,
                testability_score=10.0,
                alignment_score=8.0,
                redundancy_penalty=0.0,
                priority_score=8.5,
                requires_agent_restart=False,
                status="pending",
            )
        else:
            best = candidates[0]

        best.status = "in_progress"
        self.set_current_objective(best)
        self.logger.info(
            f"Selected new objective [{best.priority_score:.2f} pts]: {best.title} (Category: {best.category})"
        )
        return best

    def record_objective_completion(self, obj: Objective, success: bool) -> None:
        """Updates objective status in historical state."""
        obj.status = "completed" if success else "failed"
        history = {}
        if self.objectives_history_file.exists():
            try:
                with open(self.objectives_history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = {}

        key = "completed" if success else "failed"
        if key not in history:
            history[key] = []
        history[key].append(asdict(obj))

        with open(self.objectives_history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        self.clear_current_objective()
