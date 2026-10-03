"""
Episodic & Semantic Memory System (Layer B).
Maintains persistent records of experiments, lessons learned,
performance metrics, and objective histories across restarts and crashes.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any


class AgentMemory:
    def __init__(self, workspace_root: Optional[Path] = None, logger: Optional[logging.Logger] = None):
        self.workspace_root = (workspace_root or Path(__file__).resolve().parent.parent).resolve()
        self.state_dir = self.workspace_root / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger("AgentMemory")

        self.lessons_file = self.state_dir / "lessons.json"
        self.experiments_file = self.state_dir / "experiments.json"
        self.metrics_file = self.state_dir / "metrics.json"
        self.objectives_file = self.state_dir / "objectives.json"

        self._init_files()

    def _init_files(self) -> None:
        """Ensures all state JSON files exist with valid initial structure."""
        for path, default_val in [
            (self.lessons_file, []),
            (self.experiments_file, []),
            (self.metrics_file, {"history": {}, "latest": {}}),
            (self.objectives_file, {"completed": [], "failed": [], "pending": []}),
        ]:
            if not path.exists():
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(default_val, f, indent=2)
                except Exception as e:
                    self.logger.error(f"Failed to initialize memory file {path.name}: {e}")

    def _read_json(self, path: Path, default_val: Any) -> Any:
        try:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Error reading {path.name}: {e}")
        return default_val

    def _write_json(self, path: Path, data: Any) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving {path.name}: {e}")

    # Lessons Learned
    def record_lesson(
        self,
        topic: str,
        lesson: str,
        outcome: str,  # SUCCESS or FAILURE
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Records a new architectural or behavioral lesson."""
        lessons = self._read_json(self.lessons_file, [])
        entry = {
            "id": f"lesson-{len(lessons) + 1:04d}",
            "timestamp": time.time(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "topic": topic,
            "lesson": lesson,
            "outcome": outcome,
            "context": context or {},
        }
        lessons.append(entry)
        self._write_json(self.lessons_file, lessons)
        self.logger.info(f"Recorded new lesson [{outcome}]: {topic} -> {lesson}")

    def get_lessons(self, topic_keyword: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves past lessons, optionally filtering by topic keyword."""
        lessons = self._read_json(self.lessons_file, [])
        if not topic_keyword:
            return lessons
        kw = topic_keyword.lower()
        return [
            l for l in lessons
            if kw in l.get("topic", "").lower() or kw in l.get("lesson", "").lower()
        ]

    # Experiments History
    def record_experiment(
        self,
        experiment_id: str,
        objective: str,
        hypothesis: str,
        changes: List[str],
        passed: bool,
        metrics_before: Dict[str, Any],
        metrics_after: Dict[str, Any],
        lesson: str,
        commit: Optional[str] = None,
    ) -> None:
        """Stores comprehensive outcome of a modification experiment."""
        experiments = self._read_json(self.experiments_file, [])
        entry = {
            "id": experiment_id,
            "timestamp": time.time(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "objective": objective,
            "hypothesis": hypothesis,
            "changes": changes,
            "passed": passed,
            "status": "ACCEPTED" if passed else "ROLLED_BACK",
            "commit": commit,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "lesson": lesson,
        }
        experiments.append(entry)
        self._write_json(self.experiments_file, experiments)

        # Also store the lesson
        self.record_lesson(
            topic=objective,
            lesson=lesson,
            outcome="SUCCESS" if passed else "FAILURE",
            context={"experiment_id": experiment_id, "changes": changes},
        )

    def get_recent_failures(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Returns the most recent failed experiments to avoid repeating errors."""
        experiments = self._read_json(self.experiments_file, [])
        failures = [e for e in experiments if not e.get("passed", False)]
        return failures[-limit:]

    # Metrics Tracking
    def update_metrics(self, new_metrics: Dict[str, Any]) -> None:
        """Updates continuous performance and test metrics."""
        data = self._read_json(self.metrics_file, {"history": {}, "latest": {}})
        latest = data.get("latest", {})
        history = data.get("history", {})

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        for k, v in new_metrics.items():
            latest[k] = v
            if k not in history:
                history[k] = []
            history[k].append({"timestamp": ts, "value": v})
            # Cap history entries
            if len(history[k]) > 100:
                history[k] = history[k][-100:]

        data["latest"] = latest
        data["history"] = history
        self._write_json(self.metrics_file, data)

    def get_latest_metrics(self) -> Dict[str, Any]:
        """Returns latest metric readings."""
        data = self._read_json(self.metrics_file, {"latest": {}})
        return data.get("latest", {})
