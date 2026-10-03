"""
Zero-dependency .env environment variable loader for the Self-Improver system.
Reads KEY=VALUE pairs from .env and populates os.environ without overwriting
existing environment variables (or updating them if requested).
"""

import os
from pathlib import Path
from typing import Optional


def load_dotenv(workspace_root: Optional[Path] = None, override: bool = False) -> None:
    """
    Parses .env file in workspace_root and sets variables in os.environ.
    """
    if workspace_root is None:
        workspace_root = Path(__file__).resolve().parent.parent

    env_path = Path(workspace_root) / ".env"
    if not env_path.exists():
        return

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and (override or key not in os.environ):
                        os.environ[key] = val
    except Exception:
        pass
