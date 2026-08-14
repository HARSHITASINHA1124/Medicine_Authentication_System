from pydantic import BaseModel, ConfigDict, Field
from typing import Literal, Optional
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
    classification: Optional[Literal["Genuine", "Counterfeit"]] = None
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    anomaly_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

class ScanCreate(ScanBase):
    pass

class ScanResponse(ScanBase):
    scan_id: int
    device_id: str
    timestamp: datetime
    sync_status: str
    model_config = ConfigDict(from_attributes=True)
