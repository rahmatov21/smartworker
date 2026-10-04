"""
Autonomous Tool: AST Code Inspector.
Analyzes Python source files to extract AST metrics, cyclomatic complexity,
function signatures, and code health indicators.
"""

import ast
from pathlib import Path
from typing import Dict, Any, List


class ComplexityVisitor(ast.NodeVisitor):
    def __init__(self):
        self.complexity = 1
        self.functions = []
        self.classes = []

    def visit_FunctionDef(self, node):
        self.functions.append({
            "name": node.name,
            "line": node.lineno,
            "args_count": len(node.args.args),
            "has_docstring": ast.get_docstring(node) is not None,
        })
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        self.classes.append({
            "name": node.name,
            "line": node.lineno,
            "has_docstring": ast.get_docstring(node) is not None,
        })
        self.generic_visit(node)

    def visit_If(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        self.complexity += 1
        self.generic_visit(node)


def run(filepath: str = "project/src/ai_pipeline.py", **kwargs) -> Dict[str, Any]:
    """
    Inspects a Python source file and returns detailed AST analysis.
    """
    path = Path(filepath)
    if not path.exists():
        return {"error": f"File not found: {filepath}", "success": False}

    try:
        content = path.read_text(encoding="utf-8")
        tree = ast.parse(content)
        visitor = ComplexityVisitor()
        visitor.visit(tree)

        lines = content.splitlines()
        loc = len(lines)
        code_lines = len([l for l in lines if l.strip() and not l.strip().startswith("#")])

        return {
            "success": True,
            "file": filepath,
            "lines_of_code": loc,
            "executable_lines": code_lines,
            "cyclomatic_complexity": visitor.complexity,
            "classes_count": len(visitor.classes),
            "functions_count": len(visitor.functions),
            "classes": visitor.classes,
            "functions": visitor.functions,
        }
    except Exception as e:
        return {"error": str(e), "success": False}
