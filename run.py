"""
Autonomous Self-Improving AI Coding Agent Runner & CLI Controller.
Provides commands to start the supervisor, run cycles, monitor status,
trigger emergency rollbacks, and activate the kill switch.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add workspace root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from system.env_loader import load_dotenv
load_dotenv(root_dir)

from system.supervisor.supervisor import Supervisor
from system.supervisor.health_check import HealthChecker
from system.supervisor.protection import compute_supervisor_hashes
from system.supervisor.auto_healer import AutoHealer
from agent.main import AutonomousAgent



def print_banner():
    banner = """
================================================================================
          AUTONOMOUS SELF-IMPROVING AI CODING AGENT SYSTEM (AGY)
     Layer A: Protected Supervisor  |  Layer B: Autonomous Agent Engine
================================================================================
"""
    print(banner)


def show_status(root: Path):
    """Displays comprehensive system diagnostics, state, and metrics."""
    print("\n--- [System Diagnostic Status] ---")
    system_dir = root / "system"
    baseline = compute_supervisor_hashes(system_dir)
    checker = HealthChecker(workspace_root=root, baseline_hashes=baseline)
    health = checker.run_all_checks()

    print(f"Overall Health: {health.status}")
    for name, res in health.checks.items():
        status_sym = "[OK]" if res.get("passed", False) else "[WARN/FAIL]"
        print(f"  {status_sym:12s} {name:25s}")

    # Known-good commit
    kg_file = root / "state" / "known_good_version.json"
    if kg_file.exists():
        with open(kg_file, "r", encoding="utf-8") as f:
            kg = json.load(f)
        print(f"\nLast Known-Good Commit: {kg.get('commit', 'N/A')[:8]} ({kg.get('description', 'N/A')})")

    # Experiments
    exp_file = root / "state" / "experiments.json"
    if exp_file.exists():
        with open(exp_file, "r", encoding="utf-8") as f:
            exps = json.load(f)
        print(f"Total Completed Experiments: {len(exps)}")
        if exps:
            print("Recent Experiments:")
            for e in exps[-3:]:
                res_tag = "[PASS]" if e.get("passed") else "[FAIL]"
                print(f"  {res_tag} {e.get('objective', '')[:50]}")

    # Lessons
    lessons_file = root / "state" / "lessons.json"
    if lessons_file.exists():
        with open(lessons_file, "r", encoding="utf-8") as f:
            lessons = json.load(f)
        print(f"Persistent Lessons Learned: {len(lessons)}")

    # Auto-Healer Stagnation & Stuck State
    healer = AutoHealer(workspace_root=root)
    heal_report = healer.diagnose()
    if heal_report.is_healthy:
        print("\nAuto-Healer Status: HEALTHY (No stalls, freezes, or plateaus)")
    else:
        print(f"\nAuto-Healer Status: ATTENTION REQUIRED ({len(heal_report.stuck_issues)} detected)")
        for iss in heal_report.stuck_issues:
            print(f"  [!] {iss}")

    # Divergent directive
    dd_file = root / "state" / "divergent_directive.json"
    if dd_file.exists():
        try:
            with open(dd_file, "r", encoding="utf-8") as ddf:
                dd_data = json.load(ddf)
            if dd_data.get("active", False):
                print(f"Active Stagnation-Breaker Directive: ACTIVE ({dd_data.get('reason', 'N/A')})")
        except Exception:
            pass

    # Telegram Bot Status
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if token:
        try:
            from system.telegram_bot import TelegramBot
            bot = TelegramBot(workspace_root=root)
            is_valid, bot_name = bot.verify_token()
            if is_valid:
                print(f"\nTelegram Bot: CONNECTED (@{bot_name})")
            else:
                print(f"\nTelegram Bot: TOKEN ERROR ({bot_name})")
        except Exception as e:
            print(f"\nTelegram Bot: ERROR ({e})")
    else:
        print("\nTelegram Bot: NOT CONFIGURED (Optional: set TELEGRAM_BOT_TOKEN in .env)")

    # Kill switch
    kill_file = root / "system" / "kill_switch.flag"
    if kill_file.exists():
        print("\n[!] WARNING: Emergency Kill Switch is ACTIVE.")
    else:
        print("\nEmergency Kill Switch: INACTIVE (System Ready)")


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Self-Improving AI Coding Agent Controller"
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="Number of autonomous improvement cycles to run (default: continuous)",
    )
    parser.add_argument(
        "--agent-only",
        action="store_true",
        help="Run autonomous agent directly without supervisor (for debugging)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Display system status, health checks, and state statistics",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Perform emergency rollback to last known-good Git version",
    )
    parser.add_argument(
        "--kill",
        action="store_true",
        help="Trigger the emergency kill switch to stop the agent",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Clear the kill switch and allow the system to resume",
    )
    parser.add_argument(
        "--heal-now",
        action="store_true",
        help="Inspect system for frozen processes, stalled tasks, or metric plateaus and apply immediate fixes",
    )
    parser.add_argument(
        "--healer",
        action="store_true",
        help="Run the AutoHealer continuous monitoring daemon loop",
    )
    parser.add_argument(
        "--unsecured",
        action="store_true",
        help="UNSECURED / UNBOUND MODE: Disables supervisor rollback safeguards, test-before-acceptance constraints, and Layer A immutability for testing in a dedicated sandbox.",
    )
    parser.add_argument(
        "--telegram",
        action="store_true",
        help="Run standalone Telegram Bot monitoring service.",
    )
    parser.add_argument(
        "--test-telegram",
        action="store_true",
        help="Validate Telegram Bot token and test connectivity.",
    )


    args = parser.parse_args()
    print_banner()

    root = Path(__file__).resolve().parent

    if args.unsecured:
        print("""
================================================================================
                    [!]  WARNING: UNSECURED MODE ACTIVE  [!]
  Supervisor guardrails, auto-rollback on test failures, and Layer A immutability
  are DISABLED. The agent is permitted full unbounded autonomous self-modification.
  Use only in an isolated sandbox or dedicated testing environment.
================================================================================
""")

    if args.test_telegram:
        print("[*] Testing Telegram Bot configuration and connectivity...")
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            print("[-] TELEGRAM_BOT_TOKEN is not set in .env!")
            print("    Please add TELEGRAM_BOT_TOKEN=<your_token> in .env and try again.")
            return

        try:
            from system.telegram_bot import TelegramBot
            bot = TelegramBot(workspace_root=root)
            is_valid, bot_name = bot.verify_token()
            if is_valid:
                print(f"[+] SUCCESS: Connected to Telegram as @{bot_name}!")
                chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
                if chat_id:
                    print(f"[*] Sending test message to Chat ID {chat_id}...")
                    sent = bot.send_message(
                        chat_id,
                        "🤖 *Self-Improver Test Message*\nYour Telegram Bot is properly configured and working!",
                    )
                    if sent:
                        print("[+] Test message delivered successfully!")
                    else:
                        print("[-] Could not deliver test message. Verify TELEGRAM_CHAT_ID.")
                else:
                    print("[i] TELEGRAM_CHAT_ID is not set. Bot will auto-detect your chat ID when you message it.")
            else:
                print(f"[-] Token verification failed: {bot_name}")
        except Exception as e:
            print(f"[-] Error connecting to Telegram: {e}")
        return

    if args.telegram:
        print("[*] Starting Telegram Bot listener service...")
        try:
            from system.telegram_bot import TelegramBot
            bot = TelegramBot(workspace_root=root)
            bot.run()
        except Exception as e:
            print(f"[!] Error launching Telegram Bot: {e}")
        return

    if args.status:
        show_status(root)
        return

    if args.kill:
        supervisor = Supervisor(workspace_root=root, unsecured=args.unsecured)
        supervisor.activate_kill_switch()
        print("[!] Emergency Kill Switch ACTIVATED. Agent processes will be terminated.")
        return

    if args.resume:
        supervisor = Supervisor(workspace_root=root, unsecured=args.unsecured)
        supervisor.deactivate_kill_switch()
        print("[+] Emergency Kill Switch DEACTIVATED. System is ready to run.")
        return

    if args.heal_now:
        print("[*] Running AutoHealer diagnostic scan & repair...")
        healer = AutoHealer(workspace_root=root)
        report = healer.diagnose()
        print(f"System State: {'HEALTHY' if report.is_healthy else 'STALLED/DEGRADED'}")
        if report.is_healthy:
            print("[+] No stalled tasks, frozen processes, or metric deadlocks detected.")
        else:
            print(f"[!] Detected {len(report.stuck_issues)} issue(s):")
            for issue in report.stuck_issues:
                print(f"  - {issue}")
            print("\n[*] Applying targeted self-healing actions...")
            actions = healer.heal(report)
            for a in actions:
                status_str = "[FIXED]" if a.success else "[FAILED]"
                print(f"  {status_str} {a.issue_type}: {a.action_taken} ({a.details})")
        return

    if args.healer:
        print("[*] Starting AutoHealer continuous monitoring daemon...")
        healer = AutoHealer(workspace_root=root)
        healer.run_monitor_loop()
        return

    if args.rollback:

        supervisor = Supervisor(workspace_root=root, unsecured=args.unsecured)
        print("[*] Initiating manual rollback to known-good version...")
        success = supervisor.rollback_to_known_good("Manual operator request")
        if success:
            print("[+] Successfully restored workspace to known-good version.")
        else:
            print("[-] Rollback failed. Check supervisor.log.")
        return

    if args.agent_only:
        print("[*] Launching Autonomous Agent in standalone mode...")
        agent = AutonomousAgent(workspace_root=root, unsecured=args.unsecured)
        agent.run(max_cycles=args.cycles)
        return

    # Default: Run Protected Supervisor (Layer A)
    print("[*] Starting Protected Supervisor (Layer A)...")
    supervisor = Supervisor(workspace_root=root, unsecured=args.unsecured)
    supervisor.run(max_cycles=args.cycles)


if __name__ == "__main__":
    main()
