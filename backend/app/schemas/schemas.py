from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import date, datetime

class BatchBase(BaseModel):
    batch_id: str
    medicine_name: Optional[str] = None
    manufacturer: Optional[str] = None
    batch_number: Optional[str] = None
    manufacturing_date: Optional[date] = None
    expiry_date: Optional[date] = None

class BatchCreate(BatchBase):
    pass

class BatchResponse(BatchBase):
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class ScanBase(BaseModel):
    batch_id: Optional[str] = None
    medicine_name: Optional[str] = None
    classification: Optional[str] = None
    confidence_score: Optional[float] = None
    anomaly_score: Optional[float] = None

class ScanCreate(ScanBase):
    pass

class ScanResponse(ScanBase):
    scan_id: int
    device_id: str
    timestamp: datetime
    sync_status: str
    model_config = ConfigDict(from_attributes=True)
