from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta, timezone
from app.database.session import get_db
from app.database.models import Scan

router = APIRouter()

@router.get("/stats")
def get_dashboard_stats(db: Session = Depends(get_db)):
    total_scans = db.query(func.count(Scan.scan_id)).scalar()
    genuine = db.query(func.count(Scan.scan_id)).filter(Scan.classification == "Genuine").scalar()
    counterfeit = db.query(func.count(Scan.scan_id)).filter(Scan.classification == "Counterfeit").scalar()
    suspicious = db.query(func.count(Scan.scan_id)).filter(Scan.classification == "Suspicious").scalar()

    return {
        "total_scans": total_scans,
        "genuine": genuine,
        "counterfeit": counterfeit,
        "suspicious": suspicious,
        "pending_sync": db.query(func.count(Scan.scan_id)).filter(Scan.sync_status == "pending").scalar()
    }

@router.get("/trends")
def get_dashboard_trends(days: int = 7, db: Session = Depends(get_db)):
    # Simple trend: count per day for the last N days
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    
    # SQLite friendly grouping (truncating datetime to date requires func.date)
    results = db.query(
        func.date(Scan.timestamp).label("day"),
        func.count(Scan.scan_id).label("count")
    ).filter(Scan.timestamp >= cutoff).group_by(func.date(Scan.timestamp)).all()
    
    trends = [{"date": row.day, "count": row.count} for row in results]
    return {"trends": trends}

@router.get("/risk")
def get_dashboard_risk(db: Session = Depends(get_db)):
    # Average anomaly score overall
    avg_anomaly = db.query(func.avg(Scan.anomaly_score)).scalar() or 0.0
    return {"average_anomaly_score": round(avg_anomaly, 4)}
