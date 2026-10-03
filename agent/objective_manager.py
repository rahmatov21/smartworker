"""
Autonomous Objective Selection & Prioritization Manager (Layer B).
Inspects the project, identifies weaknesses, bugs, missing tests, and performance issues,
generates improvement objectives, and prioritizes them using multi-criteria scoring.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any


def _canonical_title(title: str) -> str:
    """Normalizes title by removing IDs, #numbers, punctuation and extra spaces."""
    if not title:
        return ""
    clean = re.sub(r"#\d+", "", title)
    clean = re.sub(r"[^\w\s]", " ", clean).strip().lower()
    return " ".join(clean.split())


try:
    from .llm_client import OpenRouterClient
    from .memory import AgentMemory
    from .tools import AgentTools
except (ImportError, ValueError):
    from agent.llm_client import OpenRouterClient
    from agent.memory import AgentMemory
    from agent.tools import AgentTools



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
    source: str = "catalog"    # catalog, ai_autonomous_choice, user_directed


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

        # Check for stagnation breaker directive from AutoHealer
        divergent_directive_file = self.state_dir / "divergent_directive.json"
        boosted_categories = []
        if divergent_directive_file.exists():
            try:
                with open(divergent_directive_file, "r", encoding="utf-8") as ddf:
                    directive_data = json.load(ddf)
                if directive_data.get("active", False):
                    boosted_categories = directive_data.get("boost_categories", [])
                    self.logger.info(f"Applying AutoHealer stagnation-breaker boost for categories: {boosted_categories}")
            except Exception:
                pass

        # Check for user-steered decision preferences (from Telegram /steer or /prioritize)
        decision_pref_file = self.state_dir / "decision_preferences.json"
        steered_boosts = {}
        priority_keyword = ""
        if decision_pref_file.exists():
            try:
                with open(decision_pref_file, "r", encoding="utf-8") as dpf:
                    steered_data = json.load(dpf)
                steered_boosts = steered_data.get("category_boosts", {})
                priority_keyword = steered_data.get("priority_keyword", "").lower()
            except Exception:
                pass

        # Collect recently completed and failed objectives to prevent repeating
        completed_titles = set()
        if self.objectives_history_file.exists():
            try:
                with open(self.objectives_history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
                for item in history.get("completed", []):
                    completed_titles.add(item.get("title", ""))
            except Exception:
                pass

        experiments_file = self.state_dir / "experiments.json"
        if experiments_file.exists():
            try:
                with open(experiments_file, "r", encoding="utf-8") as ef:
                    exps = json.load(ef)
                for e in exps:
                    if e.get("passed"):
                        completed_titles.add(e.get("objective", ""))
            except Exception:
                pass

        # Build canonical titles set to match titles regardless of suffixes like #123, case, or formatting
        completed_canonical = {_canonical_title(t) for t in completed_titles if t}

        # Autonomous AI Objective Brainstorming: Ask LLM to inspect project and formulate next objective
        ai_obj_data = self._generate_llm_objective(mission, needs, completed_canonical)
        if ai_obj_data:
            candidates_catalog.insert(0, ai_obj_data)

        # If all candidates in catalog are completed, add fresh dynamic candidates
        uncompleted_in_catalog = [c for c in candidates_catalog if _canonical_title(c["title"]) not in completed_canonical]
        if not uncompleted_in_catalog:
            candidates_catalog.extend(self._generate_dynamic_candidates(completed_canonical))

        objectives = []

        # Check for user-directed commands (from Telegram /build or CLI)
        user_directives_file = self.state_dir / "user_directives.json"
        if user_directives_file.exists():
            try:
                with open(user_directives_file, "r", encoding="utf-8") as udf:
                    user_directives = json.load(udf)
                pending_directives = [d for d in user_directives if d.get("status") == "pending"]
                for p_idx, p_dir in enumerate(pending_directives):
                    user_obj = Objective(
                        id=p_dir.get("id", f"user-{int(time.time())}-{p_idx:02d}"),
                        title=p_dir.get("title", "User Directed Task"),
                        category="user_directed",
                        hypothesis=p_dir.get("hypothesis", p_dir.get("description", "Execute user build command")),
                        target_files=p_dir.get("target_files", ["project/src/ai_pipeline.py", "project/tests/test_ai_pipeline.py"]),
                        impact_score=10.0,
                        feasibility_score=10.0,
                        safety_score=10.0,
                        testability_score=10.0,
                        alignment_score=10.0,
                        redundancy_penalty=0.0,
                        priority_score=99.0,  # Top priority: user-directed builds execute first
                        requires_agent_restart=p_dir.get("requires_restart", False),
                        status="pending",
                    )
                    objectives.append(user_obj)
            except Exception as e:
                self.logger.warning(f"Failed to load user directives: {e}")

        for idx, item in enumerate(candidates_catalog):
            penalty = 0.0
            canonical = _canonical_title(item["title"])
            # Redundancy penalty for completed objectives: decisively demote already solved tasks
            if canonical in completed_canonical:
                penalty += 25.0

            # Redundancy penalty against recent failed topics
            for failed_topic in recent_failed_topics:
                if failed_topic and (item["category"] in failed_topic or item["title"] in failed_topic):
                    penalty += 4.0
                    break

            # Apply stagnation breaker and user-steered boosts
            boost = 3.0 if item["category"] in boosted_categories else 0.0
            boost += steered_boosts.get(item["category"], 0.0)
            if priority_keyword and (priority_keyword in item["title"].lower() or priority_keyword in item["category"].lower()):
                boost += 6.0

            priority = self.calculate_priority_score(
                impact=item["impact"],
                feasibility=item["feasibility"],
                safety=item["safety"],
                testability=item["testability"],
                alignment=item["alignment"],
                redundancy_penalty=penalty,
            ) + boost

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
                source=item.get("source", "catalog"),
            )
            objectives.append(obj)

        # Sort descending by priority score
        objectives.sort(key=lambda o: o.priority_score, reverse=True)
        return objectives

    def _generate_dynamic_candidates(self, completed_canonical: set) -> List[Dict[str, Any]]:
        """Generates dynamic objectives to keep autonomous improvement expanding."""
        pool = [
            {
                "title": "Implement Levenshtein distance and fuzzy string matching in TextPipeline",
                "category": "performance_and_algorithms",
                "hypothesis": "Fuzzy string matching enables typo-tolerant search and semantic query retrieval.",
                "target_files": ["project/src/ai_pipeline.py", "project/tests/test_ai_pipeline.py"],
                "impact": 8.5,
                "feasibility": 9.5,
                "safety": 9.5,
                "testability": 9.5,
                "alignment": 9.0,
                "requires_restart": False,
            },
            {
                "title": "Add n-gram tokenization supporting bigrams and trigrams to TextPipeline",
                "category": "nlp_feature_engineering",
                "hypothesis": "N-gram extraction captures multi-word phrases and contextual semantics beyond single words.",
                "target_files": ["project/src/ai_pipeline.py", "project/tests/test_ai_pipeline.py"],
                "impact": 8.5,
                "feasibility": 9.0,
                "safety": 9.5,
                "testability": 9.5,
                "alignment": 9.0,
                "requires_restart": False,
            },
            {
                "title": "Implement Shannon entropy and lexical diversity analytics in TextPipeline",
                "category": "analytics_and_metrics",
                "hypothesis": "Information entropy quantifies vocabulary richness and complexity across document corpuses.",
                "target_files": ["project/src/ai_pipeline.py", "project/tests/test_ai_pipeline.py"],
                "impact": 8.0,
                "feasibility": 9.5,
                "safety": 9.5,
                "testability": 9.5,
                "alignment": 8.5,
                "requires_restart": False,
            },
            {
                "title": "Add in-memory LRU search cache with hit and miss statistics",
                "category": "performance_optimization",
                "hypothesis": "Caching search results eliminates redundant compute and cuts repeat query latency to under 1ms.",
                "target_files": ["project/src/ai_pipeline.py", "project/tests/test_ai_pipeline.py"],
                "impact": 8.5,
                "feasibility": 9.0,
                "safety": 9.0,
                "testability": 9.5,
                "alignment": 9.0,
                "requires_restart": False,
            },
            {
                "title": "Implement batch similarity matrix calculation across document pairs",
                "category": "performance_optimization",
                "hypothesis": "Pairwise matrix comparison identifies duplicate documents and clusters similar contents.",
                "target_files": ["project/src/ai_pipeline.py", "project/tests/test_ai_pipeline.py"],
                "impact": 8.5,
                "feasibility": 9.0,
                "safety": 9.0,
                "testability": 9.5,
                "alignment": 9.0,
                "requires_restart": False,
            },
            {
                "title": "Add custom stop words configuration and punctuation filtering options",
                "category": "customization_and_flexibility",
                "hypothesis": "Customizable stop words allow domain-specific vocabulary tuning for specialized text retrieval.",
                "target_files": ["project/src/ai_pipeline.py", "project/tests/test_ai_pipeline.py"],
                "impact": 7.5,
                "feasibility": 9.5,
                "safety": 10.0,
                "testability": 9.5,
                "alignment": 8.5,
                "requires_restart": False,
            },
            {
                "title": "Add atomic write locks and corrupted state recovery in agent memory",
                "category": "agent_self_improvement",
                "hypothesis": "Transactional updates guarantee zero state corruption during sudden restarts.",
                "target_files": ["agent/memory.py"],
                "impact": 8.0,
                "feasibility": 8.5,
                "safety": 8.5,
                "testability": 9.0,
                "alignment": 9.0,
                "requires_restart": True,
            },
        ]
        uncompleted = [c for c in pool if _canonical_title(c["title"]) not in completed_canonical]
        if uncompleted:
            return uncompleted

        # If all predefined capabilities are complete, dynamically generate phase milestones
        phase_idx = len(completed_canonical) + 1
        return [
            {
                "title": f"Expand capability test matrix - Phase {phase_idx}",
                "category": "testing_and_quality",
                "hypothesis": f"Continuous edge-case testing cycle {phase_idx} increases codebase robustness.",
                "target_files": ["project/tests/test_ai_pipeline.py"],
                "impact": 7.0,
                "feasibility": 9.5,
                "safety": 10.0,
                "testability": 10.0,
                "alignment": 8.0,
                "requires_restart": False,
            }
        ]

    def _generate_llm_objective(
        self,
        mission: str,
        needs: Dict[str, Any],
        completed_canonical: set,
    ) -> Optional[Dict[str, Any]]:
        """
        Uses the LLM to inspect project state and autonomously formulate a novel objective.
        """
        if self.llm.mock_mode:
            return None

        # Gather brief summary of current project files and tests
        project_src = self.tools.list_files("project/src")
        project_tests = self.tools.list_files("project/tests")

        pipeline_preview = ""
        try:
            full_pipeline = self.tools.read_file("project/src/ai_pipeline.py")
            pipeline_preview = full_pipeline[:1500]
        except Exception:
            pass

        completed_sample = list(completed_canonical)[-10:] if completed_canonical else ["none yet"]

        prompt = f"""You are the autonomous decision-making brain of an AI self-improving coding agent.
Mission: {mission}

Current Project Files:
- Source Files: {project_src}
- Test Files: {project_tests}
- Current Tests Passing: {needs.get('test_results', {}).get('passed', 0)}

Codebase Preview (project/src/ai_pipeline.py):
```python
{pipeline_preview}
```

Already Completed Improvements (DO NOT REPEAT ANY OF THESE):
{json.dumps(completed_sample, indent=2)}

Task:
Analyze what capabilities, performance optimizations, algorithms, edge-case handlers, or unit tests this codebase is missing.
Formulate the SINGLE most impactful, innovative, and concrete next improvement step.

You MUST respond strictly with a single JSON object in the following format (no other text):
{{
  "title": "Clear, concise action title (e.g. Implement Levenshtein edit distance in TextPipeline)",
  "category": "performance | algorithms | reliability | feature | code_quality",
  "hypothesis": "Concrete explanation of how and why this improves the system",
  "target_files": ["project/src/ai_pipeline.py", "project/tests/test_ai_pipeline.py"],
  "impact": 9.0,
  "feasibility": 9.0,
  "safety": 9.5,
  "testability": 9.5,
  "alignment": 9.0,
  "requires_restart": false
}}"""

        system_prompt = "You are an autonomous AI software architect. Output valid JSON only."

        try:
            self.logger.info("Querying OpenRouter LLM to autonomously formulate next improvement objective...")
            resp = self.llm.generate(prompt=prompt, system_prompt=system_prompt, temperature=0.3)
            clean_json = resp.strip()
            if "```json" in clean_json:
                clean_json = clean_json.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_json:
                clean_json = clean_json.split("```")[1].split("```")[0].strip()

            data = json.loads(clean_json)
            if isinstance(data, dict) and "title" in data and "target_files" in data:
                canonical = _canonical_title(data["title"])
                if canonical not in completed_canonical:
                    data["source"] = "ai_autonomous_choice"
                    self.logger.info(f"AI autonomously formulated new objective: '{data['title']}'")
                    return data
                else:
                    self.logger.warning(f"AI proposed an objective that was already completed: '{data['title']}'. Skipping.")
        except Exception as e:
            self.logger.warning(f"LLM objective formulation fallback to catalog: {e}")

        return None

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

        # If this was a user-directed task, mark it finished in user_directives.json
        if obj.category == "user_directed" or obj.id.startswith("user-"):
            user_directives_file = self.state_dir / "user_directives.json"
            if user_directives_file.exists():
                try:
                    with open(user_directives_file, "r", encoding="utf-8") as udf:
                        directives = json.load(udf)
                    for d in directives:
                        if d.get("id") == obj.id or (d.get("status") == "pending" and d.get("title") == obj.title):
                            d["status"] = "completed" if success else "failed"
                            d["completed_at"] = time.time()
                    with open(user_directives_file, "w", encoding="utf-8") as udf:
                        json.dump(directives, udf, indent=2)
                except Exception:
                    pass

        self.clear_current_objective()

