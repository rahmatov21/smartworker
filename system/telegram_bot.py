"""
Telegram Bot Control & Monitoring Service for Autonomous Self-Improving Agent.

Supports:
1. Viewing real-time and historical logs (/logs [target] [lines])
2. Inspecting decisions, in-flight objectives, and goals (/decision, /goal)
3. Directing the agent to build new capabilities (/build <description>)
4. Arranging and steering autonomous decision weights (/steer <category>, /prioritize <keyword>)
5. System diagnostics, emergency kill switch, and instant auto-healing (/status, /heal, /kill, /resume)
"""

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import requests

try:
    from .supervisor.health_check import HealthChecker
    from .supervisor.protection import compute_supervisor_hashes
    from .supervisor.auto_healer import AutoHealer
except (ImportError, ValueError):
    from system.supervisor.health_check import HealthChecker
    from system.supervisor.protection import compute_supervisor_hashes
    from system.supervisor.auto_healer import AutoHealer



class TelegramBot:
    def __init__(
        self,
        token: Optional[str] = None,
        authorized_chat_id: Optional[str] = None,
        workspace_root: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.workspace_root = (workspace_root or Path(__file__).resolve().parent.parent).resolve()
        self.state_dir = self.workspace_root / "state"
        self.logs_dir = self.workspace_root / "logs"
        self.system_dir = self.workspace_root / "system"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.system_dir.mkdir(parents=True, exist_ok=True)


        # Load environment configuration
        self._load_env_if_needed()
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        raw_chat_id = authorized_chat_id or os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        self.authorized_chat_id = str(raw_chat_id) if raw_chat_id else None

        self.logger = logger or self._setup_logger()
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.is_running = False
        self.last_update_id = 0

        # State tracking for notifications
        self._last_notified_exp_count = self._get_experiment_count()
        self._last_notified_objective_id: Optional[str] = None

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("TelegramBot")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [TELEGRAM-BOT] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            logger.addHandler(ch)

            fh = logging.FileHandler(self.logs_dir / "telegram_bot.log", encoding="utf-8")
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        return logger

    def _load_env_if_needed(self) -> None:
        """Loads .env file from workspace root if present."""
        env_file = self.workspace_root / ".env"
        if env_file.exists():
            try:
                with open(env_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k, v = k.strip(), v.strip().strip("'\"")
                            if k not in os.environ:
                                os.environ[k] = v
            except Exception:
                pass

    def _get_experiment_count(self) -> int:
        exp_file = self.state_dir / "experiments.json"
        if exp_file.exists():
            try:
                with open(exp_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return len(data)
            except Exception:
                pass
        return 0

    # ---------------------------------------------------------
    # Telegram API Communication
    # ---------------------------------------------------------
    def verify_token(self) -> Tuple[bool, str]:
        """
        Validates the Telegram Bot Token against api.telegram.org/bot<token>/getMe.
        Returns (is_valid, bot_username_or_error_message).
        """
        if not self.token:
            return False, "TELEGRAM_BOT_TOKEN is not configured"
        try:
            resp = requests.get(f"{self.base_url}/getMe", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("ok"):
                    bot_username = data.get("result", {}).get("username", "unknown_bot")
                    return True, bot_username
                return False, data.get("description", "Unknown error")
            elif resp.status_code == 401:
                return False, "Invalid Bot Token (Unauthorized 401)"
            else:
                return False, f"HTTP {resp.status_code}: {resp.text}"
        except Exception as e:
            return False, str(e)

    def send_message(self, chat_id: str | int, text: str, parse_mode: Optional[str] = "Markdown") -> bool:
        """Sends a message to Telegram with auto-chunking and Markdown error fallback."""
        if not self.token:
            self.logger.warning("Cannot send Telegram message: TELEGRAM_BOT_TOKEN is not configured.")
            return False

        # Telegram message length limit is 4096 characters
        chunks = [text[i : i + 4000] for i in range(0, len(text), 4000)]
        success = True

        for chunk in chunks:
            url = f"{self.base_url}/sendMessage"
            payload: Dict[str, Any] = {"chat_id": chat_id, "text": chunk}
            if parse_mode:
                payload["parse_mode"] = parse_mode

            try:
                resp = requests.post(url, json=payload, timeout=15)
                if not resp.ok and parse_mode:
                    # Retry without Markdown formatting if entity parsing fails
                    payload.pop("parse_mode", None)
                    resp = requests.post(url, json=payload, timeout=15)
                if not resp.ok:
                    self.logger.error(f"Telegram sendMessage failed: {resp.status_code} - {resp.text}")
                    success = False
            except Exception as e:
                self.logger.error(f"Error sending Telegram message: {e}")
                success = False

        return success

    def broadcast(self, text: str, parse_mode: Optional[str] = "Markdown") -> None:
        """Sends a notification to the authorized chat ID."""
        if self.authorized_chat_id:
            self.send_message(self.authorized_chat_id, text, parse_mode=parse_mode)

    def get_updates(self, offset: Optional[int] = None, timeout: int = 20) -> List[Dict[str, Any]]:
        """Long-polls for updates from Telegram."""
        if not self.token:
            return []
        url = f"{self.base_url}/getUpdates"
        params: Dict[str, Any] = {"timeout": timeout}
        if offset is not None:
            params["offset"] = offset

        try:
            resp = requests.get(url, params=params, timeout=timeout + 5)
            if resp.ok:
                return resp.json().get("result", [])
            else:
                self.logger.warning(f"getUpdates error: {resp.status_code} - {resp.text}")
        except requests.exceptions.Timeout:
            pass
        except Exception as e:
            self.logger.error(f"Error fetching updates: {e}")
        return []

    # ---------------------------------------------------------
    # Command Handlers
    # ---------------------------------------------------------
    def handle_command(self, chat_id: str | int, command_text: str) -> None:
        """Parses and executes a user command received from Telegram."""
        parts = command_text.strip().split()
        if not parts:
            return

        cmd = parts[0].lower().split("@")[0]  # Strip bot handle if present
        args = parts[1:]

        # Access check
        if self.authorized_chat_id and str(chat_id) != str(self.authorized_chat_id):
            self.send_message(
                chat_id,
                f"⛔ *Unauthorized Access*\nYour Chat ID `{chat_id}` is not authorized.\n"
                f"Add `TELEGRAM_CHAT_ID={chat_id}` in your server `.env` to enable access.",
            )
            return

        # If authorized_chat_id was not explicitly configured, bind to the first user
        if not self.authorized_chat_id:
            self.authorized_chat_id = str(chat_id)
            self.logger.info(f"Automatically bound authorized chat ID to {chat_id}")

        if cmd in ("/start", "/help"):
            self._cmd_help(chat_id)
        elif cmd == "/logs":
            self._cmd_logs(chat_id, args)
        elif cmd in ("/decision", "/goal", "/target"):
            self._cmd_decision(chat_id)
        elif cmd == "/build":
            self._cmd_build(chat_id, args)
        elif cmd == "/steer":
            self._cmd_steer(chat_id, args)
        elif cmd == "/prioritize":
            self._cmd_prioritize(chat_id, args)
        elif cmd == "/status":
            self._cmd_status(chat_id)
        elif cmd == "/heal":
            self._cmd_heal(chat_id)
        elif cmd == "/kill":
            self._cmd_kill(chat_id)
        elif cmd == "/resume":
            self._cmd_resume(chat_id)
        else:
            self.send_message(
                chat_id,
                f"❓ Unknown command `{cmd}`. Use /help to see all available commands.",
            )

    def _cmd_help(self, chat_id: str | int) -> None:
        msg = (
            "🤖 *Autonomous Self-Improving Agent Control Bot*\n"
            "=========================================\n\n"
            "📜 *1. Inspect Logs:*\n"
            "• `/logs` - Show recent 20 lines of agent log\n"
            "• `/logs agent 40` - Show last 40 lines of agent log\n"
            "• `/logs supervisor 30` - Show supervisor audit log\n"
            "• `/logs healer 20` - Show auto-healer actions log\n\n"
            "🎯 *2. Decisions & Goals:*\n"
            "• `/decision` (or `/goal`) - View in-flight objective, its target goal, hypothesis, and upcoming candidate decisions\n\n"
            "🛠️ *3. Commands & Decision Steering:*\n"
            "• `/build <description>` - Command the agent to build/improve something for itself (queued with top priority)\n"
            "• `/steer <category>` - Steer decisions towards a category (`performance`, `quality`, `agent`, or `reset`)\n"
            "• `/prioritize <keyword>` - Give a priority boost to any candidate matching keyword\n\n"
            "⚙️ *System Operations:*\n"
            "• `/status` - 10-point system health check & diagnostics\n"
            "• `/heal` - Run immediate AutoHealer diagnostic scan & fix\n"
            "• `/kill` - Emergency halt (kill switch)\n"
            "• `/resume` - Clear kill switch & resume\n"
        )
        self.send_message(chat_id, msg)

    def _cmd_logs(self, chat_id: str | int, args: List[str]) -> None:
        """Returns recent log lines."""
        target = "agent"
        lines_count = 20

        if args:
            if args[0].lower() in ("agent", "supervisor", "healer", "errors", "tool_calls"):
                target = args[0].lower()
                if len(args) > 1 and args[1].isdigit():
                    lines_count = min(int(args[1]), 80)
            elif args[0].isdigit():
                lines_count = min(int(args[0]), 80)

        log_file = self.logs_dir / f"{target}.log"
        if not log_file.exists():
            self.send_message(chat_id, f"ℹ️ Log file `{target}.log` does not exist yet.")
            return

        try:
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
            recent = all_lines[-lines_count:]
            content = "".join(recent).strip()
            if not content:
                content = "(log file is currently empty)"

            header = f"📜 *Log: {target}.log (Last {len(recent)} lines)*\n"
            msg = f"{header}```text\n{content}\n```"
            self.send_message(chat_id, msg)
        except Exception as e:
            self.send_message(chat_id, f"❌ Failed to read log file: {e}")

    def _cmd_decision(self, chat_id: str | int) -> None:
        """Displays the current in-flight objective, its goal/hypothesis, and upcoming candidates."""
        current_obj_file = self.state_dir / "current_objective.json"
        decision_pref_file = self.state_dir / "decision_preferences.json"

        lines = ["🎯 *Agent Decisions & Goal Trajectory*", "=================================="]

        # 1. Current In-Flight Objective
        if current_obj_file.exists():
            try:
                with open(current_obj_file, "r", encoding="utf-8") as f:
                    curr = json.load(f)
                lines.append("\n📍 *Current Active Objective:*")
                lines.append(f"• *Title:* {curr.get('title', 'N/A')}")
                lines.append(f"• *Category:* `{curr.get('category', 'N/A')}`")
                lines.append(f"• *Goal / The Point It Wants to Reach:*")
                lines.append(f"  👉 _{curr.get('hypothesis', 'N/A')}_")
                target_files = curr.get("target_files", [])
                lines.append(f"• *Target Files:* `{', '.join(target_files) if target_files else 'N/A'}`")
                lines.append(f"• *Status:* `{curr.get('status', 'in_progress')}`")
                lines.append(f"• *Priority Score:* `{curr.get('priority_score', 'N/A')} pts`")
            except Exception as e:
                lines.append(f"\n⚠️ Error reading current objective: {e}")
        else:
            lines.append("\n📍 *Current Active Objective:* None (idle or evaluating next cycle)")

        # 2. Active Steering & Decision Preferences
        if decision_pref_file.exists():
            try:
                with open(decision_pref_file, "r", encoding="utf-8") as dpf:
                    prefs = json.load(dpf)
                cat_boosts = prefs.get("category_boosts", {})
                kw = prefs.get("priority_keyword", "")
                if cat_boosts or kw:
                    lines.append("\n🧭 *Active User Decision Steering:*")
                    if cat_boosts:
                        for cat, boost in cat_boosts.items():
                            lines.append(f"  • Category `{cat}`: `+{boost} pts`")
                    if kw:
                        lines.append(f"  • Keyword focus: `{kw}` (`+6.0 pts`)")
            except Exception:
                pass

        # 3. Recent Outcomes
        exp_file = self.state_dir / "experiments.json"
        if exp_file.exists():
            try:
                with open(exp_file, "r", encoding="utf-8") as f:
                    exps = json.load(f)
                if exps:
                    lines.append("\n🏁 *Last 2 Completed Decisions:*")
                    for e in exps[-2:]:
                        status_sym = "✅ ACCEPTED" if e.get("passed") else "❌ ROLLED BACK"
                        lines.append(f"  • {status_sym}: {e.get('objective', '')[:50]}")
            except Exception:
                pass

        self.send_message(chat_id, "\n".join(lines))

    def _cmd_build(self, chat_id: str | int, args: List[str]) -> None:
        """Queues a user-directed objective to build something for itself or project."""
        if not args:
            self.send_message(
                chat_id,
                "⚠️ Please specify what you want the agent to build.\n"
                "Example: `/build implement LRU caching for vector query lookups`\n"
                "Example: `/build add schema validation to agent memory module`",
            )
            return

        directive_text = " ".join(args).strip()
        user_directives_file = self.state_dir / "user_directives.json"

        directives = []
        if user_directives_file.exists():
            try:
                with open(user_directives_file, "r", encoding="utf-8") as udf:
                    directives = json.load(udf)
            except Exception:
                directives = []

        new_id = f"user-{int(time.time())}"
        new_entry = {
            "id": new_id,
            "title": f"Build: {directive_text}",
            "description": directive_text,
            "hypothesis": f"User commanded build: {directive_text}. Successfully implement and verify with unit tests.",
            "status": "pending",
            "created_at": time.time(),
            "target_files": ["project/src/ai_pipeline.py", "project/tests/test_ai_pipeline.py"],
            "requires_restart": False,
        }
        directives.append(new_entry)

        try:
            with open(user_directives_file, "w", encoding="utf-8") as udf:
                json.dump(directives, udf, indent=2)

            msg = (
                f"🛠️ *Build Command Accepted!*\n\n"
                f"• *Directive ID:* `{new_id}`\n"
                f"• *Task:* `{directive_text}`\n"
                f"• *Priority:* `Top Priority (#1 - 99.0 pts)`\n\n"
                f"The agent will automatically pick this up as its next objective!"
            )
            self.send_message(chat_id, msg)
        except Exception as e:
            self.send_message(chat_id, f"❌ Failed to save build directive: {e}")

    def _cmd_steer(self, chat_id: str | int, args: List[str]) -> None:
        """Arranges decision priorities by boosting selected categories."""
        if not args:
            self.send_message(
                chat_id,
                "🧭 *Arrange Agent Decisions:*\n"
                "Use `/steer <category>` to prioritize certain types of improvements:\n\n"
                "• `/steer performance` - Prioritize speed, caching & latency\n"
                "• `/steer quality` - Prioritize tests, edge-cases & coverage\n"
                "• `/steer agent` - Prioritize self-improvement of agent engine\n"
                "• `/steer reset` - Reset decision weights to default balanced score",
            )
            return

        choice = args[0].lower()
        decision_pref_file = self.state_dir / "decision_preferences.json"

        category_map = {
            "performance": "performance_optimization",
            "speed": "performance_optimization",
            "quality": "testing_and_quality",
            "testing": "testing_and_quality",
            "tests": "testing_and_quality",
            "agent": "agent_self_improvement",
            "self": "agent_self_improvement",
            "reliability": "performance_and_reliability",
        }

        if choice in ("reset", "clear", "default"):
            try:
                decision_pref_file.unlink(missing_ok=True)
                self.send_message(chat_id, "🧭 *Decision steering reset to default balanced weights.*")
            except Exception as e:
                self.send_message(chat_id, f"❌ Failed to reset steering: {e}")
            return

        target_cat = category_map.get(choice, choice)
        payload = {
            "updated_at": time.time(),
            "category_boosts": {target_cat: 5.0},
            "active_category": target_cat,
        }

        try:
            with open(decision_pref_file, "w", encoding="utf-8") as dpf:
                json.dump(payload, dpf, indent=2)
            self.send_message(
                chat_id,
                f"🎯 *Decision Steered Successfully!*\n"
                f"Category `{target_cat}` received a `+5.0 priority boost`.\n"
                f"The agent will now heavily favor objectives in this domain.",
            )
        except Exception as e:
            self.send_message(chat_id, f"❌ Failed to save decision preferences: {e}")

    def _cmd_prioritize(self, chat_id: str | int, args: List[str]) -> None:
        """Boosts candidate objectives matching a keyword."""
        if not args:
            self.send_message(chat_id, "⚠️ Specify a keyword to prioritize. Example: `/prioritize cache`")
            return

        kw = " ".join(args).strip().lower()
        decision_pref_file = self.state_dir / "decision_preferences.json"

        data = {}
        if decision_pref_file.exists():
            try:
                with open(decision_pref_file, "r", encoding="utf-8") as dpf:
                    data = json.load(dpf)
            except Exception:
                data = {}

        data["priority_keyword"] = kw
        data["updated_at"] = time.time()

        try:
            with open(decision_pref_file, "w", encoding="utf-8") as dpf:
                json.dump(data, dpf, indent=2)
            self.send_message(
                chat_id,
                f"🚀 *Priority Boost Applied!*\n"
                f"Any candidate objective mentioning `{kw}` now receives `+6.0 points`.",
            )
        except Exception as e:
            self.send_message(chat_id, f"❌ Failed to update priority: {e}")

    def _cmd_status(self, chat_id: str | int) -> None:
        """Returns 10-point health check, diagnostics, and recent progress."""
        baseline = compute_supervisor_hashes(self.system_dir)
        checker = HealthChecker(workspace_root=self.workspace_root, baseline_hashes=baseline)
        health = checker.run_all_checks()

        healer = AutoHealer(workspace_root=self.workspace_root)
        heal_report = healer.diagnose()

        lines = [
            "📊 *System Diagnostic Status*",
            "==========================",
            f"• *Overall Health:* `{health.status}`",
            f"• *AutoHealer Status:* `{'HEALTHY' if heal_report.is_healthy else 'STALLED/ATTENTION'}`",
        ]

        if not heal_report.is_healthy:
            for iss in heal_report.stuck_issues:
                lines.append(f"  ⚠️ {iss}")

        # Kill switch
        kill_file = self.system_dir / "kill_switch.flag"
        if kill_file.exists():
            lines.append("• *Kill Switch:* 🔴 `ACTIVE (Agent Halted)`")
        else:
            lines.append("• *Kill Switch:* 🟢 `INACTIVE (Running)`")

        # Experiments
        exp_file = self.state_dir / "experiments.json"
        if exp_file.exists():
            try:
                with open(exp_file, "r", encoding="utf-8") as f:
                    exps = json.load(f)
                lines.append(f"• *Completed Cycles:* `{len(exps)}`")
                if exps:
                    last_pass = sum(1 for e in exps if e.get("passed"))
                    lines.append(f"• *Successful Improvements:* `{last_pass}/{len(exps)}`")
            except Exception:
                pass

        self.send_message(chat_id, "\n".join(lines))

    def _cmd_heal(self, chat_id: str | int) -> None:
        """Triggers AutoHealer diagnostic scan & repair immediately."""
        self.send_message(chat_id, "🩺 *Running AutoHealer diagnostic scan & repair...*")
        healer = AutoHealer(workspace_root=self.workspace_root)
        report = healer.diagnose()

        if report.is_healthy:
            self.send_message(chat_id, "✅ *System Healthy:* No frozen processes, deadlocks, or stalls detected.")
            return

        actions = healer.heal(report)
        lines = [f"🩺 *Applied {len(actions)} Healing Actions:*"]
        for a in actions:
            status_sym = "✅ FIXED" if a.success else "❌ FAILED"
            lines.append(f"• {status_sym} `{a.issue_type}`: {a.action_taken} ({a.details})")

        self.send_message(chat_id, "\n".join(lines))

    def _cmd_kill(self, chat_id: str | int) -> None:
        """Activates emergency kill switch."""
        kill_file = self.system_dir / "kill_switch.flag"
        try:
            kill_file.write_text("HALT", encoding="utf-8")
            self.send_message(
                chat_id,
                "🔴 *Emergency Kill Switch ACTIVATED!*\n"
                "The supervisor will terminate running agent processes.\n"
                "To resume, send `/resume`.",
            )
        except Exception as e:
            self.send_message(chat_id, f"❌ Failed to activate kill switch: {e}")

    def _cmd_resume(self, chat_id: str | int) -> None:
        """Deactivates emergency kill switch."""
        kill_file = self.system_dir / "kill_switch.flag"
        try:
            kill_file.unlink(missing_ok=True)
            self.send_message(
                chat_id,
                "🟢 *Emergency Kill Switch DEACTIVATED.*\n"
                "System is cleared to resume autonomous cycles.",
            )
        except Exception as e:
            self.send_message(chat_id, f"❌ Failed to deactivate kill switch: {e}")

    # ---------------------------------------------------------
    # Background Notification Daemon
    # ---------------------------------------------------------
    def _notification_loop(self) -> None:
        """Monitors state changes and sends alerts to Telegram."""
        while self.is_running:
            time.sleep(5)
            if not self.authorized_chat_id:
                continue

            try:
                # 1. Notify on new completed experiment
                exp_file = self.state_dir / "experiments.json"
                if exp_file.exists():
                    with open(exp_file, "r", encoding="utf-8") as f:
                        experiments = json.load(f)
                    current_count = len(experiments)
                    if current_count > self._last_notified_exp_count:
                        new_exps = experiments[self._last_notified_exp_count :]
                        for exp in new_exps:
                            passed = exp.get("passed", False)
                            sym = "✅ *Improvement Accepted*" if passed else "❌ *Improvement Rolled Back*"
                            title = exp.get("objective", "Unknown objective")
                            commit = exp.get("commit", "N/A")
                            tests_passed = exp.get("metrics_after", {}).get("tests_passed", "N/A")
                            msg = (
                                f"{sym}\n"
                                f"• *Task:* {title}\n"
                                f"• *Commit:* `{commit[:8] if commit else 'N/A'}`\n"
                                f"• *Tests Passed:* `{tests_passed}`"
                            )
                            self.send_message(self.authorized_chat_id, msg)
                        self._last_notified_exp_count = current_count

                # 2. Notify on new objective start
                curr_file = self.state_dir / "current_objective.json"
                if curr_file.exists():
                    with open(curr_file, "r", encoding="utf-8") as f:
                        curr_data = json.load(f)
                    obj_id = curr_data.get("id")
                    if obj_id and obj_id != self._last_notified_objective_id:
                        self._last_notified_objective_id = obj_id
                        title = curr_data.get("title", "")
                        hypothesis = curr_data.get("hypothesis", "")
                        source = curr_data.get("source", "catalog")
                        category = curr_data.get("category", "improvement")
                        source_label = (
                            "🧠 *AI Autonomous Decision (LLM Chosen)*"
                            if source == "ai_autonomous_choice"
                            else (
                                "👤 *User Directed Build*"
                                if source == "user_directed"
                                else "⚙️ *Autonomous Continuous Improvement*"
                            )
                        )
                        msg = (
                            f"{source_label}\n"
                            f"• *Objective:* {title}\n"
                            f"• *Category:* `{category}`\n"
                            f"• *Goal:* _{hypothesis}_"
                        )
                        self.send_message(self.authorized_chat_id, msg)

            except Exception:
                pass

    # ---------------------------------------------------------
    # Main Polling Loop
    # ---------------------------------------------------------
    def run(self) -> None:
        """Starts the Telegram polling bot service."""
        if not self.token:
            self.logger.error("TELEGRAM_BOT_TOKEN is not configured. Telegram bot cannot start.")
            return

        self.logger.info("Starting Telegram Bot listener service...")
        self.is_running = True

        # Start notification monitor thread
        notifier_thread = threading.Thread(target=self._notification_loop, daemon=True)
        notifier_thread.start()

        # Send online alert if chat ID is known
        if self.authorized_chat_id:
            self.send_message(
                self.authorized_chat_id,
                "🤖 *Agent Control Bot Online!*\n"
                "System is listening for commands. Send /help for full menu.",
            )

        try:
            while self.is_running:
                updates = self.get_updates(offset=self.last_update_id + 1, timeout=15)
                for update in updates:
                    update_id = update.get("update_id", 0)
                    self.last_update_id = max(self.last_update_id, update_id)

                    msg = update.get("message", {})
                    chat = msg.get("chat", {})
                    chat_id = chat.get("id")
                    text = msg.get("text", "")

                    if chat_id and text:
                        if not self.authorized_chat_id:
                            self.authorized_chat_id = str(chat_id)
                            self.logger.info(f"Auto-captured authorized Telegram chat ID: {self.authorized_chat_id}")
                        self.logger.info(f"Received Telegram command from {chat_id}: {text}")
                        self.handle_command(chat_id, text)

        except KeyboardInterrupt:
            self.logger.info("Telegram Bot stopped by operator.")
        finally:
            self.is_running = False

    def start_in_background(self) -> Optional[threading.Thread]:
        """Starts the Telegram bot polling loop in a background daemon thread."""
        if not self.token:
            self.logger.warning("No TELEGRAM_BOT_TOKEN configured. Telegram bot daemon not started.")
            return None
        thread = threading.Thread(target=self.run, name="TelegramBotDaemon", daemon=True)
        thread.start()
        self.logger.info("Telegram Bot successfully launched in background daemon thread.")
        return thread


def main():
    bot = TelegramBot()
    bot.run()


if __name__ == "__main__":
    main()
