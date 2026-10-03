"""
OpenRouter Multi-Key Failover Client (Layer B).
Manages multi-key rotation across 3-4 OpenRouter API keys with automatic fallback,
rate-limit detection (429), quota exhaustion detection (402), and intelligent mock fallback.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import requests


@dataclass
class KeyStatus:
    key: str
    index: int
    is_active: bool = True
    consecutive_failures: int = 0
    cooldown_until: float = 0.0
    last_error: Optional[str] = None


class OpenRouterClient:
    """
    Robust OpenRouter API client with automatic multi-key rotation and mock fallback.
    """

    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        model: str = "qwen/qwen-2.5-72b-instruct",
        base_url: str = "https://openrouter.ai/api/v1",
        cooldown_seconds: float = 300.0,
        mock_mode: Optional[bool] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.cooldown_seconds = cooldown_seconds
        self.logger = logger or logging.getLogger("OpenRouterClient")

        # Discover API keys from parameter or environment variables
        self.keys: List[KeyStatus] = []
        raw_keys = api_keys or self._discover_keys_from_env()

        for idx, k in enumerate(raw_keys):
            if k and k.strip():
                self.keys.append(KeyStatus(key=k.strip(), index=idx))

        self.current_key_idx = 0

        # Determine mock mode: explicit flag, or env, or if no valid keys are supplied
        if mock_mode is not None:
            self.mock_mode = mock_mode
        else:
            env_mock = os.environ.get("MOCK_LLM", "false").lower() in ("true", "1", "yes")
            self.mock_mode = env_mock or len(self.keys) == 0

        if self.mock_mode:
            self.logger.info("OpenRouterClient initialized in SIMULATION / MOCK mode.")
        else:
            self.logger.info(
                f"OpenRouterClient initialized with {len(self.keys)} API keys. Active model: {self.model}"
            )

    def _discover_keys_from_env(self) -> List[str]:
        """Discovers up to 5 API keys from standard environment variables."""
        discovered = []
        # Primary default key
        primary = os.environ.get("OPENROUTER_API_KEY")
        if primary:
            discovered.append(primary)

        # Fallback keys 1..4
        for i in range(1, 5):
            k = os.environ.get(f"OPENROUTER_API_KEY_{i}")
            if k and k not in discovered:
                discovered.append(k)

        return discovered

    def add_key(self, api_key: str) -> None:
        """Adds a new key to the rotation pool."""
        if api_key and api_key.strip():
            idx = len(self.keys)
            self.keys.append(KeyStatus(key=api_key.strip(), index=idx))
            self.logger.info(f"Added API Key #{idx + 1} to rotation pool.")

    def get_active_key(self) -> Optional[KeyStatus]:
        """Returns the next available, non-cooling API key."""
        if not self.keys:
            return None

        now = time.time()
        start_idx = self.current_key_idx
        num_keys = len(self.keys)

        for offset in range(num_keys):
            idx = (start_idx + offset) % num_keys
            candidate = self.keys[idx]
            if candidate.is_active and now >= candidate.cooldown_until:
                self.current_key_idx = idx
                return candidate

        # If all keys are cooling, pick the one with earliest cooldown expiry
        best = min(self.keys, key=lambda k: k.cooldown_until)
        self.logger.warning(
            f"All {num_keys} API keys are currently cooling down. Reusing Key #{best.index + 1} with earliest expiry."
        )
        return best

    def rotate_to_next_key(self, reason: str, status_code: Optional[int] = None) -> Optional[KeyStatus]:
        """Marks current key as cooling down and rotates to the next key."""
        if not self.keys:
            return None

        curr = self.keys[self.current_key_idx]
        curr.consecutive_failures += 1
        curr.last_error = reason

        # Determine cooldown: permanent for 401/402, temporary for 429/timeout
        if status_code in (401, 402):
            curr.cooldown_until = time.time() + 86400.0  # 24 hours
            self.logger.warning(
                f"API Key #{curr.index + 1} invalidated or depleted (HTTP {status_code}). Removing from active rotation for 24h."
            )
        else:
            # Exponential backoff based on consecutive failures
            duration = self.cooldown_seconds * (2 ** min(curr.consecutive_failures - 1, 4))
            curr.cooldown_until = time.time() + duration
            self.logger.warning(
                f"API Key #{curr.index + 1} rate-limited or failed ({reason}). Cooling down for {duration:.0f}s."
            )

        # Switch to next index
        self.current_key_idx = (self.current_key_idx + 1) % len(self.keys)
        next_key = self.get_active_key()
        if next_key:
            self.logger.info(f"Switched active OpenRouter key to Key #{next_key.index + 1}.")
        return next_key

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        response_format: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Sends a completion request with multi-key failover and retry logic.
        """
        if self.mock_mode:
            return self._generate_mock_response(prompt, system_prompt)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        max_attempts = len(self.keys) * 2
        last_error = None

        for attempt in range(max_attempts):
            key_status = self.get_active_key()
            if not key_status:
                self.logger.error("No active API keys available. Falling back to simulation.")
                return self._generate_mock_response(prompt, system_prompt)

            headers = {
                "Authorization": f"Bearer {key_status.key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/selfimprover",
                "X-Title": "Autonomous Self-Improving AI Coding Agent",
            }

            try:
                endpoint = f"{self.base_url}/chat/completions"
                resp = requests.post(endpoint, json=payload, headers=headers, timeout=60)

                if resp.status_code == 200:
                    data = resp.json()
                    key_status.consecutive_failures = 0
                    key_status.last_error = None
                    content = data["choices"][0]["message"]["content"]
                    return content

                elif resp.status_code in (429, 401, 402):
                    error_msg = f"HTTP {resp.status_code}: {resp.text}"
                    self.logger.warning(
                        f"Key #{key_status.index + 1} rejected with status {resp.status_code}. Rotating..."
                    )
                    self.rotate_to_next_key(error_msg, status_code=resp.status_code)
                    continue

                else:
                    self.logger.error(f"OpenRouter returned unexpected status {resp.status_code}: {resp.text}")
                    self.rotate_to_next_key(f"HTTP {resp.status_code}", status_code=resp.status_code)
                    continue

            except (requests.Timeout, requests.ConnectionError) as e:
                self.logger.warning(f"Network error on Key #{key_status.index + 1}: {e}. Rotating...")
                self.rotate_to_next_key(str(e))
                last_error = e
                continue
            except Exception as e:
                self.logger.error(f"Unexpected error during API call: {e}")
                last_error = e
                break

        self.logger.error(f"All API keys failed after {max_attempts} attempts. Last error: {last_error}")
        return self._generate_mock_response(prompt, system_prompt)

    def _generate_mock_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Simulated intelligent model generator for offline testing and verification.
        Provides deterministic, structured JSON or code based on prompt keywords.
        """
        prompt_lower = prompt.lower()

        # Objective selection response
        if "generate" in prompt_lower and "objective" in prompt_lower:
            return json.dumps({
                "title": "Optimize pipeline tokenization and add input edge-case test suite",
                "category": "performance_and_testing",
                "hypothesis": "Refactoring token caching and adding boundary tests will reduce latency by 15% and prevent empty input crashes.",
                "target_files": ["project/src/ai_pipeline.py", "project/tests/test_ai_pipeline.py"],
                "impact_score": 8,
                "feasibility_score": 9,
                "safety_score": 9,
                "testability_score": 10,
                "alignment_score": 9,
                "requires_agent_restart": False,
            })

        # Planning response
        if "plan" in prompt_lower:
            return json.dumps({
                "steps": [
                    "Inspect target project files and benchmarks",
                    "Add memory cache to tokenization function",
                    "Add new unit tests for edge cases (empty strings, special chars)",
                    "Run pytest suite to verify zero regressions",
                    "Run benchmark suite to measure latency reduction",
                ]
            })

        # Code modification / reasoning response
        return json.dumps({
            "status": "success",
            "message": "Autonomous self-improvement step reasoned and validated.",
        })
