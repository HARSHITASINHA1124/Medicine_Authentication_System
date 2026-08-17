from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.database.session as db_session
from app.database import models
from app.main import app


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch, tmp_path):
    db_path = tmp_path / "api_contract.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    models.Base.metadata.create_all(bind=engine)

    monkeypatch.setattr(db_session, "SessionLocal", SessionLocal)
    monkeypatch.setattr(db_session, "engine", engine)


client = TestClient(app)


@pytest.fixture
def batch_payload():
    return {
        "batch_id": "BATCH_API_001",
        "medicine_name": "Paracetamol 500mg",
        "manufacturer": "MediCore",
        "batch_number": "MC-1001",
        "manufacturing_date": "2025-01-01",
        "expiry_date": "2028-01-01",
    }


@pytest.fixture
def scan_payload():
    return {
        "batch_id": "BATCH_API_001",
        "medicine_name": "Paracetamol 500mg",
        "classification": "Genuine",
        "confidence_score": 0.97,
        "anomaly_score": 0.03,
    }


def test_valid_scan_creation(scan_payload):
    response = client.post("/api/scans/", json=scan_payload)
    assert response.status_code == 201
    body = response.json()
    assert body["batch_id"] == "BATCH_API_001"
    assert body["classification"] == "Genuine"
    assert body["sync_status"] == "pending"


def test_scan_with_nullable_fields():
    payload = {
        "batch_id": None,
        "medicine_name": None,
        "classification": None,
        "confidence_score": None,
        "anomaly_score": None,
    }
    response = client.post("/api/scans/", json=payload)
    assert response.status_code == 201
    body = response.json()
    assert body["batch_id"] is None
    assert body["medicine_name"] is None
    assert body["classification"] is None
    assert body["confidence_score"] is None
    assert body["anomaly_score"] is None


def test_scan_accepts_six_channel_inputs_and_suspicious_label():
    payload = {
        "scan_id": 987,
        "batch_id": "BATCH_API_002",
        "medicine_name": "Ibuprofen 200mg",
        "channel_1": 1.2,
        "channel_2": 2.3,
        "channel_3": 3.4,
        "channel_4": 4.5,
        "channel_5": 5.6,
        "channel_6": 6.7,
        "classification": "Suspicious",
        "confidence_score": 0.61,
        "anomaly_score": 0.47,
    }

    response = client.post("/api/scans/", json=payload)
    assert response.status_code == 201
    body = response.json()
    assert body["scan_id"] == 987
    assert body["classification"] == "Suspicious"
    assert body["channel_1"] == 1.2
    assert body["channel_6"] == 6.7


def test_invalid_scan_data_rejected():
    response = client.post("/api/scans/", json={
        "classification": "Unknown",
        "confidence_score": 1.5,
        "anomaly_score": -0.1,
    })
    assert response.status_code == 422


def test_get_scan_and_pagination():
    for _ in range(3):
        client.post("/api/scans/", json={
            "batch_id": "BATCH_API_001",
            "medicine_name": "Paracetamol 500mg",
            "classification": "Genuine",
            "confidence_score": 0.95,
            "anomaly_score": 0.05,
        })

    response = client.get("/api/scans/?skip=0&limit=2")
    assert response.status_code == 200
    assert len(response.json()) <= 2

    detail = client.get("/api/scans/1")
    assert detail.status_code == 200
    assert "scan_id" in detail.json()


def test_batch_creation_and_retrieval(batch_payload):
    response = client.post("/api/batches/", json=batch_payload)
    assert response.status_code == 201
    body = response.json()
    assert body["batch_id"] == "BATCH_API_001"

    get_all = client.get("/api/batches/")
    assert get_all.status_code == 200
    assert any(item["batch_id"] == "BATCH_API_001" for item in get_all.json())

    get_one = client.get("/api/batches/BATCH_API_001")
    assert get_one.status_code == 200
    assert get_one.json()["batch_id"] == "BATCH_API_001"


def test_dashboard_endpoints():
    stats = client.get("/api/dashboard/stats")
    assert stats.status_code == 200
    assert set(stats.json()).issuperset({"total_scans", "genuine", "counterfeit", "pending_sync"})

    trends = client.get("/api/dashboard/trends?days=7")
    assert trends.status_code == 200
    assert "trends" in trends.json()

    risk = client.get("/api/dashboard/risk")
    assert risk.status_code == 200
    assert "average_anomaly_score" in risk.json()


def test_error_responses_for_missing_records():
    missing_scan = client.get("/api/scans/999999")
    assert missing_scan.status_code == 404

    missing_batch = client.get("/api/batches/NOT_FOUND_BATCH")
    assert missing_batch.status_code == 404
