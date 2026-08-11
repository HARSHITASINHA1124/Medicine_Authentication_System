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

def sync_batch():
    """Synchronizes a batch of pending local records to PostgreSQL."""
    pg_engine = get_postgres_engine()
    if not pg_engine:
        logger.debug("No PostgreSQL URL configured. Skipping sync.")
        return False
        
    local_db = SessionLocal()
    
    try:
        # Get pending records
        pending_scans = local_db.query(Scan).filter(Scan.sync_status == "pending").limit(100).all()
        if not pending_scans:
            return False
            
        logger.info(f"Attempting to sync {len(pending_scans)} scans.")
        
        # In a real setup, you'd use SQLAlchemy ORM with PostgreSQL too.
        # We will do a raw insert using ON CONFLICT DO NOTHING for idempotency.
        # Requires the cloud DB to have (device_id, scan_id) as UNIQUE.
        insert_query = text("""
            INSERT INTO scans 
            (device_id, scan_id, batch_id, medicine_name, timestamp, classification, confidence_score, anomaly_score)
            VALUES 
            (:device_id, :scan_id, :batch_id, :medicine_name, :timestamp, :classification, :confidence_score, :anomaly_score)
            ON CONFLICT (device_id, scan_id) DO NOTHING
        """)
        
        with pg_engine.begin() as pg_conn:
            for scan in pending_scans:
                pg_conn.execute(insert_query, {
                    "device_id": scan.device_id,
                    "scan_id": scan.scan_id,
                    "batch_id": scan.batch_id,
                    "medicine_name": scan.medicine_name,
                    "timestamp": scan.timestamp,
                    "classification": scan.classification,
                    "confidence_score": scan.confidence_score,
                    "anomaly_score": scan.anomaly_score
                })
        
        # Mark local as synced
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
    while True:
        try:
            # Wait for either the event to be set (new scan) or timeout (periodic fallback)
            await asyncio.wait_for(sync_event.wait(), timeout=settings.sync_interval_seconds)
            # Event was set, meaning immediate sync requested
            sync_event.clear()
        except asyncio.TimeoutError:
            # Timeout reached, run periodic sync
            pass
            
        # Run synchronous blocking DB code in a thread pool to avoid blocking asyncio event loop
        synced_any = await asyncio.to_thread(sync_batch)
        
        # If we successfully synced a batch and it might have been a full batch (e.g. 100),
        # we might want to immediately sync again without waiting for timeout to drain the queue.
        # But for now, we just loop around.
