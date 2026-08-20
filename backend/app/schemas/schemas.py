from pydantic import BaseModel, ConfigDict, Field
from typing import Dict, List, Literal, Optional
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


class MLClassification(BaseModel):
    medicine: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    status: Optional[Literal["HIGH", "MEDIUM", "LOW"]] = None
    explanation: Optional[str] = None


class MLAnomaly(BaseModel):
    score: Optional[float] = Field(default=None, ge=0.0)
    genuine_threshold: Optional[float] = None
    counterfeit_threshold: Optional[float] = None
    status: Optional[Literal["GENUINE", "COUNTERFEIT", "SUSPICIOUS"]] = None
    distance_from_genuine: Optional[float] = None
    distance_from_counterfeit: Optional[float] = None
    explanation: Optional[str] = None


class MLMeasurement(BaseModel):
    stability_cv: Optional[float] = Field(default=None, ge=0.0)
    stability_status: Optional[Literal["STABLE", "UNSTABLE"]] = None


class MLFeature(BaseModel):
    feature: str
    value: float
    absolute_value: Optional[float] = None


class MLExplainability(BaseModel):
    classification: Optional[MLClassification] = None
    anomaly: Optional[MLAnomaly] = None
    measurement: Optional[MLMeasurement] = None
    top_engineered_features: List[MLFeature] = []


class MLScanData(BaseModel):
    n_readings: Optional[int] = Field(default=None, ge=1)
    aggregated_reading: Dict[str, float] = {}
    channel_std: Dict[str, float] = {}
    stability_cv: Optional[float] = Field(default=None, ge=0.0)


class ScanAnalysisRequest(BaseModel):
    readings: List[Dict[str, float]]

class ScanBase(BaseModel):
    scan_id: Optional[int] = None
    batch_id: Optional[str] = None
    medicine_name: Optional[str] = None
    channel_1: Optional[float] = None
    channel_2: Optional[float] = None
    channel_3: Optional[float] = None
    channel_4: Optional[float] = None
    channel_5: Optional[float] = None
    channel_6: Optional[float] = None
    classification: Optional[Literal["Genuine", "Counterfeit", "Suspicious"]] = None
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    anomaly_score: Optional[float] = Field(default=None, ge=0.0)

class ScanCreate(ScanBase):
    medicine: Optional[str] = None
    classification_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    classification_status: Optional[Literal["HIGH", "MEDIUM", "LOW"]] = None
    anomaly_status: Optional[Literal["GENUINE", "COUNTERFEIT", "SUSPICIOUS"]] = None
    final_status: Optional[Literal["GENUINE", "COUNTERFEIT", "SUSPICIOUS"]] = None
    explainability: Optional[MLExplainability] = None
    scan_data: Optional[MLScanData] = None
    classification_probabilities: Dict[str, float] = {}

class ScanResponse(ScanBase):
    id: int
    scan_id: int
    device_id: str
    timestamp: datetime
    sync_status: str
    medicine: Optional[str] = None
    classification_confidence: Optional[float] = None
    classification_status: Optional[str] = None
    anomaly_status: Optional[str] = None
    final_status: Optional[str] = None
    number_of_readings: Optional[int] = None
    stability_cv: Optional[float] = None
    ml_result: Optional[dict] = None
    model_config = ConfigDict(from_attributes=True)
