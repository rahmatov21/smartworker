"""
Unit tests for Telegram Bot Control & Monitoring Service.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from system.telegram_bot import TelegramBot


@pytest.fixture
def mock_bot(tmp_path):
    bot = TelegramBot(
        token="123456:mock_token_abc",
        authorized_chat_id="999888777",
        workspace_root=tmp_path,
    )
    bot.send_message = MagicMock(return_value=True)
    return bot


def test_telegram_bot_unauthorized_user_blocked(mock_bot):
    mock_bot.handle_command(chat_id="111222333", command_text="/status")
    mock_bot.send_message.assert_called_once()
    assert "Unauthorized" in mock_bot.send_message.call_args[0][1]


def test_telegram_bot_help_command(mock_bot):
    mock_bot.handle_command(chat_id="999888777", command_text="/help")
    mock_bot.send_message.assert_called_once()
    assert "/logs" in mock_bot.send_message.call_args[0][1]
    assert "/build" in mock_bot.send_message.call_args[0][1]
    assert "/decision" in mock_bot.send_message.call_args[0][1]


def test_telegram_bot_build_command_queues_directive(mock_bot, tmp_path):
    mock_bot.handle_command(
        chat_id="999888777",
        command_text="/build implement Redis LRU cache for embedding lookups",
    )
    mock_bot.send_message.assert_called_once()
    assert "Build Command Accepted" in mock_bot.send_message.call_args[0][1]

    # Verify user_directives.json was created
    directives_file = tmp_path / "state" / "user_directives.json"
    assert directives_file.exists()
    with open(directives_file, "r") as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["status"] == "pending"
    assert "Redis LRU cache" in data[0]["description"]


def test_telegram_bot_steer_command(mock_bot, tmp_path):
    mock_bot.handle_command(chat_id="999888777", command_text="/steer performance")
    mock_bot.send_message.assert_called_once()
    assert "Decision Steered Successfully" in mock_bot.send_message.call_args[0][1]

    # Verify decision_preferences.json was created
    prefs_file = tmp_path / "state" / "decision_preferences.json"
    assert prefs_file.exists()
    with open(prefs_file, "r") as f:
        data = json.load(f)
    assert data.get("active_category") == "performance_optimization"
    assert data.get("category_boosts", {}).get("performance_optimization") == 5.0


def test_telegram_bot_decision_command(mock_bot, tmp_path):
    curr_file = tmp_path / "state" / "current_objective.json"
    curr_file.parent.mkdir(parents=True, exist_ok=True)
    with open(curr_file, "w") as f:
        json.dump({
            "id": "obj-test-123",
            "title": "Add caching layer",
            "category": "performance",
            "hypothesis": "Caches will reduce latency by 50%",
            "target_files": ["project/src/ai_pipeline.py"],
            "status": "in_progress",
            "priority_score": 9.2,
        }, f)

    mock_bot.handle_command(chat_id="999888777", command_text="/decision")
    mock_bot.send_message.assert_called_once()
    msg = mock_bot.send_message.call_args[0][1]
    assert "Add caching layer" in msg
    assert "Caches will reduce latency by 50%" in msg


def test_telegram_bot_kill_and_resume(mock_bot, tmp_path):
    kill_file = tmp_path / "system" / "kill_switch.flag"

    mock_bot.handle_command(chat_id="999888777", command_text="/kill")
    assert kill_file.exists()

    mock_bot.handle_command(chat_id="999888777", command_text="/resume")
    assert not kill_file.exists()


def test_telegram_bot_tools_command(mock_bot, tmp_path):
    # Setup custom_tools directory in tmp_path
    tools_dir = tmp_path / "agent" / "custom_tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    sample_tool = tools_dir / "sample_test_tool.py"
    sample_tool.write_text(
        'TOOL_SPEC = {"name": "sample_test_tool", "description": "A sample tool"}\n'
        'def run(): return {"ok": True}\n',
        encoding="utf-8",
    )

    mock_bot.handle_command(chat_id="999888777", command_text="/tools")
    mock_bot.send_message.assert_called_once()
    msg = mock_bot.send_message.call_args[0][1]
    assert "Real-Time Dynamic Agent Tools" in msg
    assert "sample_test_tool" in msg

