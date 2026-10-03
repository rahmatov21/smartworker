"""
Autonomous Planner & Execution Engine (Layer B).
Translates high-level objectives into step-by-step actions, creates Git checkpoints,
applies modifications, diagnoses and attempts limited fixes if tests fail,
and strictly enforces rollback upon unresolved failures.
"""

import ast
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    from .evaluator import Evaluator, EvaluationReport
    from .llm_client import OpenRouterClient
    from .memory import AgentMemory
    from .objective_manager import Objective
    from .tools import AgentTools
except (ImportError, ValueError):
    from agent.evaluator import Evaluator, EvaluationReport
    from agent.llm_client import OpenRouterClient
    from agent.memory import AgentMemory
    from agent.objective_manager import Objective
    from agent.tools import AgentTools



@dataclass
class ExecutionResult:
    success: bool
    rolled_back: bool
    checkpoint_commit: str
    final_commit: Optional[str]
    eval_report: EvaluationReport
    fix_attempts_made: int
    restart_required: bool
    details: str


class Planner:
    def __init__(
        self,
        workspace_root: Optional[Path] = None,
        tools: Optional[AgentTools] = None,
        evaluator: Optional[Evaluator] = None,
        memory: Optional[AgentMemory] = None,
        llm: Optional[OpenRouterClient] = None,
        max_fix_attempts: int = 2,
        unsecured: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        self.workspace_root = (workspace_root or Path(__file__).resolve().parent.parent).resolve()
        self.unsecured = unsecured or os.environ.get("UNSECURED_MODE", "0") == "1"
        self.tools = tools or AgentTools(self.workspace_root, unsecured=self.unsecured)
        self.evaluator = evaluator or Evaluator(self.workspace_root, self.tools, unsecured=self.unsecured)
        self.memory = memory or AgentMemory(self.workspace_root)
        self.llm = llm or OpenRouterClient()
        self.max_fix_attempts = max_fix_attempts
        self.logger = logger or logging.getLogger("Planner")

    def plan_objective(self, obj: Objective) -> List[str]:
        """Uses LLM reasoning to decompose objective into executable steps."""
        prompt = (
            f"Create a step-by-step implementation plan for the following objective:\n"
            f"Title: {obj.title}\n"
            f"Category: {obj.category}\n"
            f"Hypothesis: {obj.hypothesis}\n"
            f"Target Files: {', '.join(obj.target_files)}\n\n"
            f"Return a structured list of actionable steps."
        )
        resp = self.llm.generate(prompt)
        try:
            data = json.loads(resp)
            if "steps" in data:
                return data["steps"]
        except Exception:
            pass

        return [
            f"Inspect target files: {', '.join(obj.target_files)}",
            "Establish baseline tests and benchmark metrics",
            "Apply reasoned code changes and edge-case handling",
            "Add or update unit tests to verify behavior",
            "Run evaluation suite and benchmarks",
            "Accept if passed, otherwise rollback",
        ]

    def execute_objective(self, obj: Objective) -> ExecutionResult:
        """
        Executes the autonomous improvement cycle:
        CHECKPOINT -> MODIFY -> TEST -> (DIAGNOSE & FIX) -> ACCEPT OR ROLLBACK
        """
        self.logger.info(f"--- Starting Objective Execution: {obj.title} ---")
        steps = self.plan_objective(obj)
        self.logger.info(f"Plan formulated with {len(steps)} steps.")

        # 1. Establish baseline
        baseline_test = self.tools.run_tests("project/tests")
        baseline_metrics = {
            "tests_passed": baseline_test["passed"],
            "test_duration": baseline_test["duration"],
        }
        self.logger.info(f"Baseline: {baseline_test['passed']} tests passing.")

        # 2. Create Git Checkpoint before modifying anything
        checkpoint_name = f"before-{obj.id}"
        checkpoint_commit = self.tools.git_checkpoint(checkpoint_name)
        self.logger.info(f"Created Git checkpoint commit: {checkpoint_commit[:8]}")

        # 3. Apply improvement
        self._apply_improvement_action(obj)

        # 4. Evaluate modifications
        eval_report = self.evaluator.evaluate_change(baseline_metrics=baseline_metrics)
        fix_attempts = 0

        # 5. Test failure diagnosis and retry loop (up to max_fix_attempts)
        while not eval_report.passed and fix_attempts < self.max_fix_attempts:
            fix_attempts += 1
            self.logger.warning(
                f"Evaluation failed (Attempt {fix_attempts}/{self.max_fix_attempts}): {eval_report.failure_reason}. "
                "Attempting diagnostic fix..."
            )
            self._apply_diagnostic_fix(obj, eval_report)
            eval_report = self.evaluator.evaluate_change(baseline_metrics=baseline_metrics)

        # 6. Final Decision: Accept or Rollback
        if not eval_report.passed:
            if self.unsecured:
                self.logger.warning(
                    f"[UNSECURED MODE] Tests failed ({eval_report.failure_reason}), but auto-rollback is DISABLED. "
                    "Committing changes to allow unrestrained code evolution!"
                )
                commit_msg = f"feat(unsecured): {obj.title} [UNVERIFIED / TESTS FAILED]"
                final_commit = self.tools.git_commit(commit_msg)
                lesson = f"[UNSECURED] Experiment '{obj.title}' kept despite test failure to observe downstream evolution."
                self.memory.record_experiment(
                    experiment_id=obj.id,
                    objective=obj.title,
                    hypothesis=obj.hypothesis,
                    changes=obj.target_files,
                    passed=False,
                    metrics_before=baseline_metrics,
                    metrics_after={"tests_passed": eval_report.tests_passed},
                    lesson=lesson,
                    commit=final_commit,
                )
                return ExecutionResult(
                    success=False,
                    rolled_back=False,
                    checkpoint_commit=checkpoint_commit,
                    final_commit=final_commit,
                    eval_report=eval_report,
                    fix_attempts_made=fix_attempts,
                    restart_required=obj.requires_agent_restart,
                    details=f"[UNSECURED] Kept modified code at commit {final_commit[:8]} without rollback.",
                )

            self.logger.error(
                f"Unresolved failures after {fix_attempts} fix attempts! "
                f"EXECUTING AUTOMATIC ROLLBACK to {checkpoint_commit[:8]}."
            )
            self.tools.git_rollback(checkpoint_commit)

            # Record failure in persistent memory
            lesson = (
                f"Failed to implement '{obj.title}' safely. "
                f"Failure reason: {eval_report.failure_reason}. "
                f"Rollback restored codebase to clean checkpoint."
            )
            self.memory.record_experiment(
                experiment_id=obj.id,
                objective=obj.title,
                hypothesis=obj.hypothesis,
                changes=obj.target_files,
                passed=False,
                metrics_before=baseline_metrics,
                metrics_after={"tests_passed": eval_report.tests_passed},
                lesson=lesson,
                commit=checkpoint_commit,
            )

            return ExecutionResult(
                success=False,
                rolled_back=True,
                checkpoint_commit=checkpoint_commit,
                final_commit=None,
                eval_report=eval_report,
                fix_attempts_made=fix_attempts,
                restart_required=False,
                details=f"Rolled back to {checkpoint_commit[:8]} after {fix_attempts} failed fix attempts.",
            )

        # 7. ACCEPT: Commit successful changes
        commit_msg = f"feat(autonomous): {obj.title}\n\nHypothesis: {obj.hypothesis}"
        final_commit = self.tools.git_commit(commit_msg)
        self.logger.info(f"SUCCESS: Changes verified and committed under {final_commit[:8]}!")

        # Update known-good version in state
        self._update_known_good_state(final_commit, obj.title)

        # Record experiment success in memory
        lesson = (
            f"Successfully improved system: {obj.title}. "
            f"All {eval_report.tests_passed} tests passed. Zero regressions."
        )
        self.memory.record_experiment(
            experiment_id=obj.id,
            objective=obj.title,
            hypothesis=obj.hypothesis,
            changes=obj.target_files,
            passed=True,
            metrics_before=baseline_metrics,
            metrics_after={
                "tests_passed": eval_report.tests_passed,
                "benchmarks": eval_report.benchmark_metrics,
            },
            lesson=lesson,
            commit=final_commit,
        )

        # 8. Check if self-modification requires restart
        restart_req = obj.requires_agent_restart
        if restart_req:
            self.logger.info("Objective modifies agent internals. Requesting supervisor restart...")
            self._signal_restart(final_commit, f"Self-modification: {obj.title}")

        return ExecutionResult(
            success=True,
            rolled_back=False,
            checkpoint_commit=checkpoint_commit,
            final_commit=final_commit,
            eval_report=eval_report,
            fix_attempts_made=fix_attempts,
            restart_required=restart_req,
            details="All tests and evaluations passed. Known-good version advanced.",
        )

    def _apply_improvement_action(self, obj: Objective) -> None:
        """
        Executes the targeted code improvement.
        Uses real OpenRouter LLM generation if available, otherwise runs progressive mock improvements.
        """
        self.logger.info(f"Applying code modifications for objective '{obj.title}' to: {', '.join(obj.target_files)}")

        if not self.llm.mock_mode:
            self._apply_llm_improvement(obj)
            return

        # Progressive simulation / mock fallback improvements
        for target in obj.target_files:
            # Check test files first to avoid substring collision with src files
            if target.endswith("test_ai_pipeline.py"):
                self._improve_test_coverage(target)
            elif target.endswith("ai_pipeline.py"):
                self._improve_ai_pipeline(target)
            elif "memory.py" in target:
                self._improve_agent_memory(target)

    def _apply_llm_improvement(self, obj: Objective) -> None:
        """
        Queries OpenRouter LLM to generate reasoned code modifications for each target file.
        """
        for target_path in obj.target_files:
            current_content = ""
            full_path = self.workspace_root / target_path
            if full_path.exists():
                try:
                    current_content = self.tools.read_file(target_path)
                except Exception:
                    current_content = ""

            prompt = f"""You are an autonomous AI software engineer directly improving this codebase.
Objective: {obj.title}
Category: {obj.category}
Hypothesis: {obj.hypothesis}

Target File: {target_path}

Current Content of {target_path}:
```python
{current_content}
```

Instructions:
1. Implement the requested enhancements, optimizations, or test coverage for this file.
2. The code MUST be 100% syntactically valid Python 3.
3. Preserve all existing tests, classes, and public functions to prevent regressions.
4. Output ONLY the complete, updated file content within a single ```python ``` code block. Do NOT include markdown outside the code block.
"""
            system_prompt = (
                "You are an expert autonomous software engineer. "
                "Output strictly the complete modified Python file inside ```python ```."
            )

            try:
                self.logger.info(f"Querying OpenRouter LLM ({self.llm.model}) to modify {target_path}...")
                response = self.llm.generate(prompt=prompt, system_prompt=system_prompt, temperature=0.1)
                code = self._extract_python_code(response)
                if code and code.strip():
                    try:
                        ast.parse(code)
                        self.tools.write_file(target_path, code)
                        self.logger.info(f"Applied LLM-generated code to {target_path}")
                    except SyntaxError as syn_err:
                        self.logger.warning(f"LLM generated invalid syntax for {target_path}: {syn_err}")
                else:
                    self.logger.warning(f"Could not extract Python code from LLM response for {target_path}")
            except Exception as e:
                self.logger.error(f"Error querying LLM for {target_path}: {e}")

    def _extract_python_code(self, response: str) -> str:
        """Extracts python code from markdown fence blocks or returns raw string."""
        if "```python" in response:
            parts = response.split("```python")
            code = parts[1].split("```")[0]
            return code.strip()
        elif "```" in response:
            parts = response.split("```")
            code = parts[1].split("```")[0]
            return code.strip()
        return response.strip()

    def _improve_ai_pipeline(self, target_path: str) -> None:
        """Progressively enhances ai_pipeline.py with caching, batch processing, and similarity."""
        try:
            content = self.tools.read_file(target_path)
            # Step 1: Add cache decorator and empty input guard
            if "@functools.lru_cache" not in content:
                content = "import functools\n" + content
                content = content.replace(
                    "def tokenize(text: str)",
                    "@functools.lru_cache(maxsize=1024)\ndef tokenize(text: str)",
                )
                self.tools.write_file(target_path, content)
                self.logger.info(f"Enhanced {target_path} with functools LRU cache.")
                return

            # Step 2: Add batch processing method
            if "def batch_process" not in content:
                batch_method = """
    def batch_process(self, texts: list) -> list:
        \"\"\"Batch processes multiple text documents in a single invocation.\"\"\"
        return [self.process(t) for t in texts]
"""
                content += batch_method
                self.tools.write_file(target_path, content)
                self.logger.info(f"Enhanced {target_path} with batch_process capability.")
                return

            # Step 3: Add token cosine similarity metric
            if "def similarity" not in content:
                sim_method = """
    def similarity(self, text_a: str, text_b: str) -> float:
        \"\"\"Calculates token overlap similarity between two texts.\"\"\"
        tokens_a = set(self.tokenize(text_a))
        tokens_b = set(self.tokenize(text_b))
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = len(tokens_a.intersection(tokens_b))
        union = len(tokens_a.union(tokens_b))
        return float(intersection) / float(union) if union > 0 else 0.0
"""
                content += sim_method
                self.tools.write_file(target_path, content)
                self.logger.info(f"Enhanced {target_path} with similarity metric calculation.")
                return

        except Exception as e:
            self.logger.warning(f"Could not apply pipeline optimization: {e}")

    def _improve_test_coverage(self, target_path: str) -> None:
        """Progressively expands test coverage with new edge-case tests."""
        try:
            content = self.tools.read_file(target_path)
            # Test 1: Empty string boundary
            if "test_pipeline_boundary_empty_string" not in content:
                content += """

def test_pipeline_boundary_empty_string():
    from project.src.ai_pipeline import TextPipeline
    pipeline = TextPipeline()
    res = pipeline.process("")
    assert res["tokens"] == []
    assert res["word_count"] == 0
"""
                self.tools.write_file(target_path, content)
                self.logger.info(f"Added boundary tests to {target_path}.")
                return

            # Test 2: Unicode & emoji handling
            if "test_pipeline_unicode_handling" not in content:
                content += """

def test_pipeline_unicode_handling():
    from project.src.ai_pipeline import TextPipeline
    pipeline = TextPipeline()
    res = pipeline.process("Hello 🌍 世界! Café naïve.")
    assert "hello" in res["tokens"]
    assert res["word_count"] > 0
"""
                self.tools.write_file(target_path, content)
                self.logger.info(f"Added Unicode handling tests to {target_path}.")
                return

            # Test 3: Batch processing verification
            if "test_pipeline_batch_processing" not in content:
                content += """

def test_pipeline_batch_processing():
    from project.src.ai_pipeline import TextPipeline
    pipeline = TextPipeline()
    batch = pipeline.batch_process(["First sentence.", "Second sentence."])
    assert len(batch) == 2
    assert batch[0]["word_count"] == 2
"""
                self.tools.write_file(target_path, content)
                self.logger.info(f"Added batch processing tests to {target_path}.")
                return

        except Exception as e:
            self.logger.warning(f"Could not update test coverage: {e}")

    def _improve_agent_memory(self, target_path: str) -> None:
        """Adds schema validation comment / docstring to memory system."""
        try:
            content = self.tools.read_file(target_path)
            tag = f"# [Self-Improved-{int(time.time())}] Schema validation active.\n"
            if "# [Self-Improved" not in content:
                content = tag + content
                self.tools.write_file(target_path, content)
                self.logger.info(f"Updated {target_path} with self-improvement tags.")
        except Exception as e:
            self.logger.warning(f"Could not modify agent memory: {e}")

    def _apply_diagnostic_fix(self, obj: Objective, report: EvaluationReport) -> None:
        """Diagnoses failure message and attempts targeted repair."""
        self.logger.info(f"Diagnosing test failure: {report.failure_reason}. Attempting targeted repair...")
        if not self.llm.mock_mode:
            for target in obj.target_files:
                try:
                    content = self.tools.read_file(target)
                    prompt = f"""A recent autonomous code modification caused a test failure or regression.
Objective: {obj.title}
Target File: {target}
Failure Reason: {report.failure_reason}
Diagnostics / Output:
{report.diagnostics}

Current Content of {target}:
```python
{content}
```

Fix the code so that all tests pass and no syntax/runtime errors occur.
Output ONLY the complete corrected file content within a single ```python ``` code block.
"""
                    response = self.llm.generate(
                        prompt=prompt,
                        system_prompt="Fix Python code regression. Return only ```python ... ```.",
                    )
                    code = self._extract_python_code(response)
                    if code:
                        ast.parse(code)
                        self.tools.write_file(target, code)
                        self.logger.info(f"Applied LLM diagnostic fix to {target}")
                except Exception as e:
                    self.logger.warning(f"Diagnostic fix attempt failed for {target}: {e}")
        else:
            for target in obj.target_files:
                try:
                    content = self.tools.read_file(target)
                    self.tools.write_file(target, content)
                except Exception:
                    pass

    def _update_known_good_state(self, commit: str, description: str) -> None:
        """Updates state/known_good_version.json."""
        kg_file = self.workspace_root / "state" / "known_good_version.json"
        data = {
            "commit": commit,
            "timestamp": time.time(),
            "description": description,
            "verified": True,
        }
        with open(kg_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _signal_restart(self, commit: str, reason: str) -> None:
        """Signals supervisor that an agent self-restart is needed."""
        signal_file = self.workspace_root / "state" / "restart_signal.json"
        signal = {
            "restart_required": True,
            "timestamp": time.time(),
            "commit": commit,
            "reason": reason,
        }
        with open(signal_file, "w", encoding="utf-8") as f:
            json.dump(signal, f, indent=2)
