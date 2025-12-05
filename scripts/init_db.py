import duckdb
from pathlib import Path
from common.config_loader import load_config

def get_duckdb_conn():
    config = load_config()
    project_root = Path(__file__).resolve().parents[1]
    db_path = (project_root / config['duckdb']['path']).resolve()
    print(f"Using database file: {db_path}")
    conn = duckdb.connect(str(db_path))
    return conn

def run_sql_file(conn, sql_file: Path):
    print(f"Executing SQL file: {sql_file}")
    with sql_file.open("r", encoding="utf-8") as f:
        sql_content = f.read()
        # DuckDB supports executing multiple SQL statements
        conn.execute(sql_content)

def init_db():
    conn = get_duckdb_conn()

    # Install and load spatial extension
    conn.execute("INSTALL spatial;")
    conn.execute("LOAD spatial;")

    # Load SQL files sequentially
    sql_dir = Path(__file__).resolve().parent / "sql"
    sql_files = sorted(sql_dir.glob("*.sql"))  # Execute in filename order

    for sql_file in sql_files:
        run_sql_file(conn, sql_file)
    conn.close()
    print("Database initialization completed.")


if __name__ == '__main__':
    init_db()