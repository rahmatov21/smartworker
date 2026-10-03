"""
Unit tests for OpenRouter Multi-Key Failover Client.
"""

from unittest.mock import patch, MagicMock
import requests
import pytest

from agent.llm_client import OpenRouterClient, KeyStatus


def test_client_initialization_with_multiple_keys():
    keys = ["sk-key-1", "sk-key-2", "sk-key-3", "sk-key-4"]
    client = OpenRouterClient(api_keys=keys, mock_mode=False)
    assert len(client.keys) == 4
    active = client.get_active_key()
    assert active.key == "sk-key-1"


def test_key_rotation_on_failure():
    keys = ["sk-key-1", "sk-key-2", "sk-key-3"]
    client = OpenRouterClient(api_keys=keys, cooldown_seconds=60.0, mock_mode=False)

    # Initial key
    assert client.get_active_key().key == "sk-key-1"

    # Simulate rate-limit 429 on key 1
    next_key = client.rotate_to_next_key("Rate limited", status_code=429)
    assert next_key.key == "sk-key-2"
    assert client.get_active_key().key == "sk-key-2"

    # Simulate quota exhaustion 402 on key 2
    next_key = client.rotate_to_next_key("Credits depleted", status_code=402)
    assert next_key.key == "sk-key-3"
    assert client.get_active_key().key == "sk-key-3"


@patch("requests.post")
def test_generate_automatic_failover(mock_post):
    # Key 1 returns 429, Key 2 returns 200 with valid completion
    resp_429 = MagicMock()
    resp_429.status_code = 429
    resp_429.text = "Too Many Requests"

    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.json.return_value = {
        "choices": [{"message": {"content": "Autonomous improvement plan generated."}}]
    }

    mock_post.side_effect = [resp_429, resp_200]

    keys = ["sk-bad-key", "sk-good-key"]
    client = OpenRouterClient(api_keys=keys, mock_mode=False)

    result = client.generate("Plan next objective")
    assert "Autonomous improvement plan generated." in result
    assert mock_post.call_count == 2
    # Verify second call used Key 2
    auth_header = mock_post.call_args_list[1][1]["headers"]["Authorization"]
    assert auth_header == "Bearer sk-good-key"


def test_mock_mode_generator():
    client = OpenRouterClient(mock_mode=True)
    resp = client.generate("Generate objective for project")
    assert "hypothesis" in resp
    assert "target_files" in resp
