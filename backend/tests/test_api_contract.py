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


def test_scan_accepts_ml_result_schema():
    payload = {
        "medicine": "Dolonex",
        "classification_confidence": 0.5481055928992532,
        "anomaly_score": 51430928427314.336,
        "classification_status": "LOW",
        "anomaly_status": "COUNTERFEIT",
        "final_status": "COUNTERFEIT",
        "explainability": {
            "classification": {
                "medicine": "Dolonex",
                "confidence": 0.5481055928992532,
                "status": "LOW",
                "explanation": "Low confidence",
            },
            "anomaly": {
                "score": 51430928427314.336,
                "genuine_threshold": 24.67,
                "counterfeit_threshold": 40.53,
                "status": "COUNTERFEIT",
                "distance_from_genuine": 51430928427289.664,
                "distance_from_counterfeit": 51430928427273.805,
                "explanation": "Counterfeit boundary crossed",
            },
            "measurement": {"stability_cv": 0.0057, "stability_status": "STABLE"},
            "top_engineered_features": [{"feature": "ratio_600_650", "value": 1.44, "absolute_value": 1.44}],
        },
        "scan_data": {
            "n_readings": 10,
            "aggregated_reading": {
                "ch450": 1000.0,
                "ch500": 1200.0,
                "ch550": 1400.0,
                "ch570": 1500.0,
                "ch600": 1300.0,
                "ch650": 900.0,
            },
            "channel_std": {
                "ch450": 5.7,
                "ch500": 6.9,
                "ch550": 8.0,
                "ch570": 8.6,
                "ch600": 7.4,
                "ch650": 5.2,
            },
            "stability_cv": 0.0057,
        },
        "classification_probabilities": {"Dolonex": 0.5481, "Etoricoxib": 0.4513},
    }

    response = client.post("/api/scans/", json=payload)
    assert response.status_code == 201
    body = response.json()
    assert body["medicine_name"] == "Dolonex"
    assert body["classification"] == "Counterfeit"
    assert body["confidence_score"] == 0.5481055928992532
    assert body["anomaly_score"] == 51430928427314.336
    assert body["channel_1"] == 1000.0
    assert body["channel_6"] == 900.0
    assert body["number_of_readings"] == 10


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
