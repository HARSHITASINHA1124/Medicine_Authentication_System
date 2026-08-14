import asyncio
import logging

# pyrefly: ignore [missing-import]
from sqlalchemy import create_engine, text
# pyrefly: ignore [missing-import]
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.database.session import SessionLocal, get_postgres_engine
from app.database.models import Scan

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# This event is set by the API when a new scan arrives
sync_event = asyncio.Event()


def get_sync_wait_seconds(failure_count: int = 0) -> int:
    """Return a bounded retry delay after sync failures.

    - immediate sync when a scan arrives or after a successful sync
    - exponential backoff on failed attempts, capped to a reasonable ceiling
    """
    if failure_count <= 0:
        return settings.sync_interval_seconds
    backoff = settings.sync_retry_backoff_seconds * (2 ** (failure_count - 1))
    return min(backoff, settings.sync_retry_max_seconds)


def sync_batch():
    """Synchronizes a batch of pending local records to PostgreSQL."""
    pg_engine = get_postgres_engine()
    if not pg_engine:
        logger.debug("No PostgreSQL URL configured. Skipping sync.")
        return False

    local_db = SessionLocal()

    try:
        # Get pending records (bounded batch)
        pending_scans = local_db.query(Scan).filter(Scan.sync_status == "pending").limit(100).all()
        if not pending_scans:
            return False

        logger.info(f"Attempting to sync {len(pending_scans)} scans.")

        insert_query = text("""
            INSERT INTO scans
            (device_id, scan_id, batch_id, medicine_name, timestamp, classification, confidence_score, anomaly_score, sync_status)
            VALUES
            (:device_id, :scan_id, :batch_id, :medicine_name, :timestamp, :classification, :confidence_score, :anomaly_score, :sync_status)
            ON CONFLICT (device_id, scan_id) DO NOTHING
        """)

        params = []
        for scan in pending_scans:
            params.append({
                "device_id": scan.device_id,
                "scan_id": scan.scan_id,
                "batch_id": scan.batch_id,
                "medicine_name": scan.medicine_name,
                "timestamp": scan.timestamp,
                "classification": scan.classification,
                "confidence_score": scan.confidence_score,
                "anomaly_score": scan.anomaly_score,
                "sync_status": scan.sync_status,
            })

        with pg_engine.begin() as pg_conn:
            pg_conn.execute(insert_query, params)

        # Mark local as synced only after successful upload
        for scan in pending_scans:
            scan.sync_status = "synced"

        local_db.commit()
        logger.info(f"Successfully synced {len(pending_scans)} scans.")
        return True

    except Exception as e:
        logger.error(f"Error during synchronization: {e}")
        # We do not rollback local changes; pending remains pending
        return False
    finally:
        local_db.close()


async def persistent_sync_loop():
    """Runs forever, triggering on an event or periodic timeout."""
    logger.info("Starting persistent synchronization loop...")
    failure_count = 0

    while True:
        wait_seconds = get_sync_wait_seconds(failure_count)
        try:
            await asyncio.wait_for(sync_event.wait(), timeout=wait_seconds)
            sync_event.clear()
            failure_count = 0
        except asyncio.TimeoutError:
            pass

        synced_any = await asyncio.to_thread(sync_batch)
        if synced_any:
            failure_count = 0
        else:
            failure_count += 1
