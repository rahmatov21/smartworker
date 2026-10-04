"""
Autonomous Tool: Embedded SQLite Relational Storage.
Provides lightweight relational storage, indexing, and SQL query capabilities
for autonomous agent analytics and experiment caching.
"""

import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional


def run(
    action: str = "query",
    sql: str = "SELECT 1 as alive;",
    params: Optional[List[Any]] = None,
    db_name: str = "agent_analytics.db",
    **kwargs,
) -> Dict[str, Any]:
    """
    Executes relational SQL commands against embedded SQLite store.
    """
    db_path = Path("state") / db_name
    db_path.parent.mkdir(parents=True, exist_ok=True)

    params = params or []
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Ensure default tables exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS autonomous_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    metric_name TEXT,
                    metric_value REAL,
                    details TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tool_invocations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    tool_name TEXT,
                    duration_ms REAL,
                    success INTEGER
                )
            """)
            conn.commit()

            cursor.execute(sql, params)
            if sql.strip().upper().startswith(("SELECT", "PRAGMA")):
                columns = [col[0] for col in cursor.description] if cursor.description else []
                rows = cursor.fetchall()
                results = [dict(zip(columns, row)) for row in rows]
                return {
                    "success": True,
                    "action": "query",
                    "row_count": len(results),
                    "columns": columns,
                    "results": results,
                }
            else:
                conn.commit()
                return {
                    "success": True,
                    "action": "execute",
                    "rows_affected": cursor.rowcount,
                }
    except Exception as e:
        return {"error": str(e), "success": False}
