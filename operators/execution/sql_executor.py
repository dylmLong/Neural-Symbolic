"""
SQL Execution Operator (Physical Operator)

Executes a single SQL query and returns result list (in dictionary form).
"""
from pathlib import Path
from typing import List
import duckdb

from common.config_loader import load_config

# Cache database connection (avoid repeated creation)
_duckdb_conn = None


def get_duckdb_conn():
    """Create and configure DuckDB database connection"""
    global _duckdb_conn
    if _duckdb_conn is None:
        config = load_config()
        project_root = Path(__file__).resolve().parents[2]  # operators/execution/sql_executor.py -> project root directory
        db_path = (project_root / config['duckdb']['path']).resolve()
        _duckdb_conn = duckdb.connect(db_path)
        _duckdb_conn.execute("INSTALL spatial;")
        _duckdb_conn.execute("LOAD spatial;")
    return _duckdb_conn


def execute_sql_query(sql: str) -> List[dict]:
    """
    Execute a single SQL query and return result list (in dictionary form).
    
    :param sql: SQL statement
    :return: Query results
    """
    try:
        conn = get_duckdb_conn()  # Open connection
        result = conn.execute(sql).fetchall()  # Get single row execution result
        columns = [desc[0] for desc in conn.description]
        # Convert to list[dict]
        return [dict(zip(columns, row)) for row in result]  # Convert query results to dictionary list
    except Exception as e:
        return [{"error": str(e)}]

