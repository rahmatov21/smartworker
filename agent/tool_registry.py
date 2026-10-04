"""
Dynamic Tool Registry & Synthesis Engine (Layer B).
Enables the autonomous agent to build, register, inspect, and execute
custom tools in real time.
"""

import ast
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable


class ToolRegistry:
    """
    Manages custom tools synthesized by the agent at runtime.
    Stored in agent/custom_tools/ and hot-reloaded dynamically.
    """

    def __init__(self, workspace_root: Optional[Path] = None, logger: Optional[logging.Logger] = None):
        self.workspace_root = (workspace_root or Path(__file__).resolve().parent.parent).resolve()
        self.tools_dir = self.workspace_root / "agent" / "custom_tools"
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.tools_dir / "registry.json"
        self.logger = logger or logging.getLogger("ToolRegistry")
        self._loaded_tools: Dict[str, Callable] = {}
        self._ensure_init_file()
        self.load_all_tools()

    def _ensure_init_file(self) -> None:
        init_py = self.tools_dir / "__init__.py"
        if not init_py.exists():
            init_py.write_text('"""Custom tools dynamically created by the agent."""\n', encoding="utf-8")

    def _load_metadata(self) -> Dict[str, Any]:
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_metadata(self, data: Dict[str, Any]) -> None:
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def register_tool(self, name: str, code: str, description: str = "") -> Dict[str, Any]:
        """
        Synthesizes and registers a new tool in real time:
        1. Validates Python syntax using AST parsing.
        2. Writes tool module to agent/custom_tools/{name}.py.
        3. Dynamically loads and verifies callable entrypoint.
        4. Updates registry metadata.
        """
        clean_name = "".join(c for c in name if c.isalnum() or c == "_").strip().lower()
        if not clean_name:
            raise ValueError(f"Invalid tool name: '{name}'")

        # 1. Syntax validation
        try:
            ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Tool syntax validation failed: {e}")

        # 2. Write tool file
        tool_file = self.tools_dir / f"{clean_name}.py"
        with open(tool_file, "w", encoding="utf-8") as f:
            f.write(code)

        # 3. Dynamic import & validation
        spec = importlib.util.spec_from_file_location(f"agent.custom_tools.{clean_name}", tool_file)
        if not spec or not spec.loader:
            raise ImportError(f"Could not load spec for tool {clean_name}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[f"agent.custom_tools.{clean_name}"] = module
        spec.loader.exec_module(module)

        entrypoint = getattr(module, "run", None)
        if entrypoint is None or not callable(entrypoint):
            # Check for function with matching name
            entrypoint = getattr(module, clean_name, None)
            if entrypoint is None or not callable(entrypoint):
                raise AttributeError(f"Tool '{clean_name}' must define a callable 'run(**kwargs)' or '{clean_name}(**kwargs)'")

        self._loaded_tools[clean_name] = entrypoint

        # 4. Save metadata
        meta = self._load_metadata()
        meta[clean_name] = {
            "name": clean_name,
            "description": description or getattr(entrypoint, "__doc__", "") or "Custom autonomous tool",
            "file": f"agent/custom_tools/{clean_name}.py",
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        self._save_metadata(meta)

        self.logger.info(f"Registered and activated custom tool: '{clean_name}'")
        return meta[clean_name]

    def call_tool(self, name: str, **kwargs) -> Any:
        """Executes a registered custom tool by name."""
        clean_name = name.strip().lower()
        if clean_name not in self._loaded_tools:
            # Try reloading in case it was created recently
            self.load_tool(clean_name)

        if clean_name not in self._loaded_tools:
            raise KeyError(f"Tool '{clean_name}' is not registered. Available tools: {list(self._loaded_tools.keys())}")

        return self._loaded_tools[clean_name](**kwargs)

    def load_tool(self, name: str) -> bool:
        """Loads a single tool from disk."""
        tool_file = self.tools_dir / f"{name}.py"
        if not tool_file.exists():
            return False
        try:
            spec = importlib.util.spec_from_file_location(f"agent.custom_tools.{name}", tool_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[f"agent.custom_tools.{name}"] = module
                spec.loader.exec_module(module)
                entrypoint = getattr(module, "run", None) or getattr(module, name, None)
                if callable(entrypoint):
                    self._loaded_tools[name] = entrypoint
                    return True
        except Exception as e:
            self.logger.warning(f"Failed to load custom tool '{name}': {e}")
        return False

    def load_all_tools(self) -> Dict[str, Callable]:
        """Scans agent/custom_tools/ and loads all valid tools."""
        for py_file in self.tools_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                continue
            self.load_tool(py_file.stem)
        return self._loaded_tools

    def list_tools(self) -> List[Dict[str, Any]]:
        """Returns list of registered tools with metadata."""
        meta = self._load_metadata()
        results = []
        for name, entrypoint in self._loaded_tools.items():
            info = meta.get(name, {})
            desc = info.get("description") or getattr(entrypoint, "__doc__", "Custom agent tool")
            results.append({
                "name": name,
                "description": desc.strip().split("\n")[0] if desc else "Custom agent tool",
                "file": f"agent/custom_tools/{name}.py",
                "callable": True,
            })
        return results

    def delete_tool(self, name: str) -> bool:
        """Removes a custom tool from disk and registry."""
        clean_name = name.strip().lower()
        tool_file = self.tools_dir / f"{clean_name}.py"
        if tool_file.exists():
            tool_file.unlink()
        self._loaded_tools.pop(clean_name, None)
        meta = self._load_metadata()
        if clean_name in meta:
            del meta[clean_name]
            self._save_metadata(meta)
        return True


if __name__ == "__main__":
    reg = ToolRegistry()
    tools = reg.list_tools()
    print(f"Registered Dynamic Tools ({len(tools)}):")
    for t in tools:
        print(f"  - {t['name']}: {t['description']}")

