# pyrefly: ignore [missing-import]
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Date
# pyrefly: ignore [missing-import]
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.database.session import Base

class Batch(Base):
    __tablename__ = "batches"

    batch_id = Column(String, primary_key=True, index=True)
    medicine_name = Column(String, nullable=True)
    manufacturer = Column(String, nullable=True)
    batch_number = Column(String, nullable=True)
    manufacturing_date = Column(Date, nullable=True)
    expiry_date = Column(Date, nullable=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    scans = relationship("Scan", back_populates="batch")

class Scan(Base):
    __tablename__ = "scans"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    scan_id = Column(Integer, unique=True, nullable=True, index=True)
    device_id = Column(String, nullable=False)
    batch_id = Column(String, ForeignKey("batches.batch_id"), nullable=True)
    medicine_name = Column(String, nullable=True)
    timestamp = Column(DateTime, nullable=False, index=True, default=lambda: datetime.now(timezone.utc))
    channel_1 = Column(Float, nullable=True)
    channel_2 = Column(Float, nullable=True)
    channel_3 = Column(Float, nullable=True)
    channel_4 = Column(Float, nullable=True)
    channel_5 = Column(Float, nullable=True)
    channel_6 = Column(Float, nullable=True)
    classification = Column(String, nullable=True)
    confidence_score = Column(Float, nullable=True)
    anomaly_score = Column(Float, nullable=True)
    sync_status = Column(String, nullable=False, index=True, default="pending")

    batch = relationship("Batch", back_populates="scans")
