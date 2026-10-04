"""
Tests for Dynamic Custom Tool Registry and Runtime Tool Synthesis.
"""

from pathlib import Path
from agent.tools import AgentTools
from agent.tool_registry import ToolRegistry


def test_tool_registry_discovers_builtin_tools(tmp_path):
    root = Path(__file__).resolve().parent.parent
    registry = ToolRegistry(workspace_root=root)
    tools = registry.list_tools()
    tool_names = [t["name"] for t in tools]

    assert "code_ast_inspector" in tool_names
    assert "performance_profiler" in tool_names
    assert "synthetic_corpus_generator" in tool_names
    assert "sqlite_storage" in tool_names


def test_call_builtin_custom_tools():
    root = Path(__file__).resolve().parent.parent
    tools = AgentTools(workspace_root=root)

    # 1. Test AST Inspector
    ast_res = tools.call_custom_tool("code_ast_inspector", filepath="project/src/ai_pipeline.py")
    assert ast_res["success"] is True
    assert ast_res["lines_of_code"] > 0
    assert ast_res["classes_count"] >= 1

    # 2. Test Corpus Generator
    corpus_res = tools.call_custom_tool("synthetic_corpus_generator", size=5)
    assert corpus_res["success"] is True
    assert corpus_res["total_documents"] >= 5

    # 3. Test SQLite Storage Tool
    sql_res = tools.call_custom_tool("sqlite_storage", sql="SELECT 42 as num;")
    assert sql_res["success"] is True
    assert sql_res["results"][0]["num"] == 42


def test_runtime_tool_synthesis_and_execution(tmp_path):
    registry = ToolRegistry(workspace_root=tmp_path)

    # Agent writes a new tool in real time
    tool_code = """
def run(x: int = 10, y: int = 20, **kwargs):
    return {"sum": x + y, "product": x * y, "success": True}
"""
    registry.register_tool(
        name="math_calculator",
        code=tool_code,
        description="Calculates arithmetic sum and product",
    )

    res = registry.call_tool("math_calculator", x=7, y=8)
    assert res["success"] is True
    assert res["sum"] == 15
    assert res["product"] == 56

    # Verify tool appears in registry listing
    all_tools = registry.list_tools()
    names = [t["name"] for t in all_tools]
    assert "math_calculator" in names

    # Clean up test tool
    registry.delete_tool("math_calculator")
