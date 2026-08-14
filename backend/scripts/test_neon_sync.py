"""Interactive test for Neon Postgres connection and SQLite->Postgres sync.

This script prompts for your Neon `POSTGRES_URL` (hidden input), applies migrations,
creates a mock batch and one pending scan in the local SQLite DB, runs the sync,
and reports whether the local scan was marked `synced`.

It does NOT print or store your connection string. It only prints sanitized status messages.

Usage:
  python scripts/test_neon_sync.py

When prompted, paste your Neon connection string. The script will run migrations and attempt a sync.
"""
import os
import sys
from datetime import datetime, timezone

# Ensure the repository `backend` directory is on sys.path so `import app` works
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import settings
from app.database.session import get_postgres_engine, SessionLocal, engine
from app.database import models


def apply_migrations_with_engine(pg_engine):
    # Run the SQL migration file using the provided SQLAlchemy engine
    sql_file = os.path.join(os.path.dirname(__file__), "..", "postgres_migrations", "create_tables.sql")
    sql_file = os.path.normpath(sql_file)
    if not os.path.exists(sql_file):
        print("Migration file not found:", sql_file)
        return False

    with open(sql_file, "r", encoding="utf-8") as fh:
        sql = fh.read()

    try:
        with pg_engine.begin() as conn:
            conn.exec_driver_sql(sql)
        return True
    except Exception as e:
        print("Migration failed:", e)
        return False


def create_local_mock_records():
    # Ensure local tables exist
    models.Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        # Create mock batch
        batch = models.Batch(
            batch_id="TEST_B001",
            medicine_name="TestMed",
            manufacturer="TestMan",
            batch_number="T-001",
            manufacturing_date=None,
            expiry_date=None,
        )
        db.add(batch)
        db.commit()
        db.refresh(batch)

        # Create pending scan
        scan = models.Scan(
            device_id="TEST_DEVICE",
            batch_id=batch.batch_id,
            medicine_name="TestMed",
            classification="Genuine",
            confidence_score=0.99,
            anomaly_score=0.02,
            sync_status="pending",
            timestamp=datetime.now(timezone.utc),
        )
        db.add(scan)
        db.commit()
        db.refresh(scan)
        return scan.scan_id
    except Exception as e:
        print("Failed to create local records:", e)
        db.rollback()
        return None
    finally:
        db.close()


def run_sync_with_postgres_url(pg_url):
    # Temporarily set settings.postgres_url for get_postgres_engine to pick up if needed
    # But we will call sync logic directly by creating engine
    from app.core.sync_worker import sync_batch

    # Monkey-patch get_postgres_engine to return our engine for this run
    from app.database import session as dbsession
    original_get_pg = dbsession.get_postgres_engine

    try:
        dbsession.get_postgres_engine = lambda: __import__('sqlalchemy').create_engine(pg_url)
        result = sync_batch()
        return result
    finally:
        dbsession.get_postgres_engine = original_get_pg


def check_local_scan_status(scan_id):
    db = SessionLocal()
    try:
        scan = db.query(models.Scan).filter(models.Scan.scan_id == scan_id).first()
        if not scan:
            return None
        return scan.sync_status
    finally:
        db.close()


def main():
    print("This script will test Neon Postgres migration and a single sync operation.")

    # Use existing application configuration
    pg_engine = get_postgres_engine()
    if not pg_engine:
        print("ERROR: POSTGRES_URL is not configured in application settings (.env). Aborting.")
        sys.exit(1)

    print("Applying migrations to Neon Postgres...")
    ok = apply_migrations_with_engine(pg_engine)
    if not ok:
        print("Migrations failed. Aborting test.")
        sys.exit(1)
    print("Migrations applied successfully.")

    print("Creating local mock batch and pending scan in SQLite...")
    scan_id = create_local_mock_records()
    if scan_id is None:
        print("Failed to create local records. Aborting.")
        sys.exit(1)
    print(f"Created local pending scan with scan_id={scan_id}")

    print("Running synchronization (SQLite -> Neon Postgres)...")
    try:
        # sync_batch uses get_postgres_engine internally, which will return our engine
        from app.core.sync_worker import sync_batch
        synced = sync_batch()
    except Exception as e:
        print("Synchronization raised an exception:", e)
        synced = False

    status = check_local_scan_status(scan_id)
    print("Sync function returned:", bool(synced))
    print("Local scan sync_status:", status)

    if synced and status == "synced":
        print("SUCCESS: Scan uploaded and local record marked 'synced'.")
    elif not synced:
        print("FAILED: sync_batch reported failure. Check network and Neon credentials.")
    else:
        print("Partial: sync_batch returned True but local record not marked 'synced'.")


if __name__ == "__main__":
    main()
