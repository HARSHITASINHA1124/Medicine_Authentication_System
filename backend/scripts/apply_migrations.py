"""Apply PostgreSQL migrations using the `POSTGRES_URL` environment variable.

This script reads `postgres_migrations/create_tables.sql` and executes it
against the configured `POSTGRES_URL`. It avoids printing credentials.

Usage:
  # set POSTGRES_URL in your environment or .env file
  python scripts/apply_migrations.py
"""
import os
import sys
from sqlalchemy import text
from app.database.session import get_postgres_engine


def main():
    engine = get_postgres_engine()
    if not engine:
        print("POSTGRES_URL is not configured. Set the POSTGRES_URL environment variable and retry.")
        sys.exit(1)

    sql_file = os.path.join(os.path.dirname(__file__), "..", "postgres_migrations", "create_tables.sql")
    sql_file = os.path.normpath(sql_file)
    if not os.path.exists(sql_file):
        print(f"Migration file not found: {sql_file}")
        sys.exit(1)

    with open(sql_file, "r", encoding="utf-8") as fh:
        sql = fh.read()

    try:
        with engine.begin() as conn:
            # exec_driver_sql allows executing multiple statements in one string
            conn.exec_driver_sql(sql)
        print("Migrations applied successfully.")
    except Exception as e:
        print("Failed to apply migrations:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
