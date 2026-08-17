from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, timezone
from app.database.session import get_db
from app.database.models import Scan
from app.schemas.schemas import ScanCreate, ScanResponse
from app.core.config import settings

router = APIRouter()

def trigger_sync():
    """Placeholder to signal the background sync worker.
    In a threaded approach, this could set an Event flag. 
    For now, we'll implement the persistent sync loop to poll or wait.
    """
    pass

@router.post("/", response_model=ScanResponse, status_code=201)
def create_scan(scan: ScanCreate, db: Session = Depends(get_db)):
    payload = scan.model_dump(exclude_none=True)
    requested_scan_id = payload.get("scan_id")

    # Create the local record
    new_scan = Scan(
        **payload,
        device_id=settings.device_id,
        sync_status="pending",
        timestamp=datetime.now(timezone.utc)
    )
    db.add(new_scan)
    db.commit()
    db.refresh(new_scan)

    if requested_scan_id is not None:
        new_scan.scan_id = requested_scan_id
    elif new_scan.scan_id is None:
        new_scan.scan_id = new_scan.id
    db.commit()
    db.refresh(new_scan)

    # Notify the sync worker to attempt sync (implementation will depend on worker design)
    trigger_sync()

    return new_scan

@router.get("/", response_model=List[ScanResponse])
def get_scans(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    return db.query(Scan).order_by(Scan.timestamp.desc()).offset(skip).limit(limit).all()

@router.get("/{scan_id}", response_model=ScanResponse)
def get_scan(scan_id: int, db: Session = Depends(get_db)):
    scan = db.query(Scan).filter(Scan.scan_id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    return scan
