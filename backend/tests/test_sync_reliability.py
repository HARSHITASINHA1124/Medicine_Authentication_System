from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import models
from app.database import session as db_session
from app.core import sync_worker


class FakePGEngine:
    def __init__(self, fail_on_begin=False, fail_on_execute=False):
        self.fail_on_begin = fail_on_begin
        self.fail_on_execute = fail_on_execute
        self.executed = []

    def begin(self):
        if self.fail_on_begin:
            raise ConnectionError("Neon unavailable")
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params):
        if self.fail_on_execute:
            raise RuntimeError("Network failure during synchronization")
        self.executed.extend(params)
        return None


@pytest.fixture
def sqlite_session_factory(monkeypatch, tmp_path):
    db_path = tmp_path / "test_local.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    models.Base.metadata.create_all(bind=engine)

    monkeypatch.setattr(db_session, "SessionLocal", SessionLocal)
    monkeypatch.setattr(db_session, "engine", engine)
    monkeypatch.setattr(sync_worker, "SessionLocal", SessionLocal)

    return SessionLocal


def make_pending_scan(SessionLocal, **kwargs):
    db = SessionLocal()
    try:
        batch = models.Batch(
            batch_id=kwargs.pop("batch_id", "BATCH_TEST_1"),
            medicine_name=kwargs.pop("medicine_name", "Paracetamol"),
            manufacturer=kwargs.pop("manufacturer", "Acme Pharma"),
            batch_number=kwargs.pop("batch_number", "ABC-001"),
        )
        db.add(batch)
        db.commit()

        scan = models.Scan(
            device_id=kwargs.pop("device_id", "device-1"),
            batch_id=batch.batch_id,
            medicine_name=kwargs.pop("medicine_name", None),
            timestamp=kwargs.pop("timestamp", datetime.now(timezone.utc)),
            classification=kwargs.pop("classification", "Genuine"),
            confidence_score=kwargs.pop("confidence_score", 0.99),
            anomaly_score=kwargs.pop("anomaly_score", 0.02),
            sync_status=kwargs.pop("sync_status", "pending"),
            **kwargs,
        )
        db.add(scan)
        db.commit()
        db.refresh(scan)
        return scan
    finally:
        db.close()


def test_neon_unavailable_keeps_records_pending(sqlite_session_factory, monkeypatch):
    monkeypatch.setattr(sync_worker, "get_postgres_engine", lambda: None)
    scan = make_pending_scan(sqlite_session_factory)

    result = sync_worker.sync_batch()

    assert result is False
    db = sqlite_session_factory()
    try:
        fresh = db.query(models.Scan).filter(models.Scan.scan_id == scan.scan_id).first()
        assert fresh is not None
        assert fresh.sync_status == "pending"
    finally:
        db.close()


def test_network_failure_during_sync_does_not_lose_data(sqlite_session_factory, monkeypatch):
    fake_engine = FakePGEngine(fail_on_begin=True)
    monkeypatch.setattr(sync_worker, "get_postgres_engine", lambda: fake_engine)
    scan = make_pending_scan(sqlite_session_factory)

    result = sync_worker.sync_batch()

    assert result is False
    db = sqlite_session_factory()
    try:
        fresh = db.query(models.Scan).filter(models.Scan.scan_id == scan.scan_id).first()
        assert fresh is not None
        assert fresh.sync_status == "pending"
    finally:
        db.close()


def test_rerunning_sync_does_not_duplicate_records(sqlite_session_factory, monkeypatch):
    fake_engine = FakePGEngine()
    monkeypatch.setattr(sync_worker, "get_postgres_engine", lambda: fake_engine)
    scan = make_pending_scan(sqlite_session_factory)

    first = sync_worker.sync_batch()
    second = sync_worker.sync_batch()

    assert first is True
    assert second is False
    assert len(fake_engine.executed) == 1
    assert fake_engine.executed[0]["device_id"] == "device-1"
    assert fake_engine.executed[0]["scan_id"] == scan.scan_id

    db = sqlite_session_factory()
    try:
        fresh = db.query(models.Scan).filter(models.Scan.scan_id == scan.scan_id).first()
        assert fresh.sync_status == "synced"
    finally:
        db.close()


def test_restart_with_pending_record_can_sync_afterward(sqlite_session_factory, monkeypatch):
    fake_engine = FakePGEngine()
    monkeypatch.setattr(sync_worker, "get_postgres_engine", lambda: fake_engine)
    scan = make_pending_scan(sqlite_session_factory)

    # Simulate a fresh app start by creating a new session after the pending record already exists.
    db = sqlite_session_factory()
    try:
        pending = db.query(models.Scan).filter(models.Scan.scan_id == scan.scan_id).first()
        assert pending.sync_status == "pending"
    finally:
        db.close()

    result = sync_worker.sync_batch()

    assert result is True
    db = sqlite_session_factory()
    try:
        synced = db.query(models.Scan).filter(models.Scan.scan_id == scan.scan_id).first()
        assert synced.sync_status == "synced"
    finally:
        db.close()


def test_multiple_batches_and_scans_sync_relationships(sqlite_session_factory, monkeypatch):
    fake_engine = FakePGEngine()
    monkeypatch.setattr(sync_worker, "get_postgres_engine", lambda: fake_engine)

    db = sqlite_session_factory()
    try:
        batch_a = models.Batch(batch_id="BATCH_A", medicine_name="Alpha", manufacturer="M1", batch_number="A1")
        batch_b = models.Batch(batch_id="BATCH_B", medicine_name="Beta", manufacturer="M2", batch_number="B1")
        db.add_all([batch_a, batch_b])
        db.commit()

        scan_a = models.Scan(
            device_id="device-2",
            batch_id="BATCH_A",
            medicine_name="Alpha",
            timestamp=datetime.now(timezone.utc),
            classification="Genuine",
            confidence_score=0.8,
            anomaly_score=0.1,
            sync_status="pending",
        )
        scan_b = models.Scan(
            device_id="device-2",
            batch_id="BATCH_B",
            medicine_name="Beta",
            timestamp=datetime.now(timezone.utc),
            classification="Counterfeit",
            confidence_score=0.65,
            anomaly_score=0.8,
            sync_status="pending",
        )
        db.add_all([scan_a, scan_b])
        db.commit()
        db.refresh(scan_a)
        db.refresh(scan_b)
    finally:
        db.close()

    result = sync_worker.sync_batch()

    assert result is True
    assert len(fake_engine.executed) == 2
    batch_ids = {row["batch_id"] for row in fake_engine.executed}
    assert batch_ids == {"BATCH_A", "BATCH_B"}


def test_nullable_fields_are_allowed_in_local_and_sync(sqlite_session_factory, monkeypatch):
    fake_engine = FakePGEngine()
    monkeypatch.setattr(sync_worker, "get_postgres_engine", lambda: fake_engine)

    db = sqlite_session_factory()
    try:
        batch = models.Batch(batch_id="BATCH_NULL", medicine_name=None, manufacturer=None, batch_number=None)
        db.add(batch)
        db.commit()

        scan = models.Scan(
            device_id="device-3",
            batch_id="BATCH_NULL",
            medicine_name=None,
            timestamp=datetime.now(timezone.utc),
            classification=None,
            confidence_score=None,
            anomaly_score=None,
            sync_status="pending",
        )
        db.add(scan)
        db.commit()
        db.refresh(scan)
    finally:
        db.close()

    result = sync_worker.sync_batch()

    assert result is True
    assert len(fake_engine.executed) == 1
    assert fake_engine.executed[0]["medicine_name"] is None
    assert fake_engine.executed[0]["classification"] is None
    assert fake_engine.executed[0]["confidence_score"] is None
    assert fake_engine.executed[0]["anomaly_score"] is None

    db = sqlite_session_factory()
    try:
        synced = db.query(models.Scan).filter(models.Scan.scan_id == scan.scan_id).first()
        assert synced.sync_status == "synced"
    finally:
        db.close()


def test_retry_backoff_is_bounded_and_reasonable(monkeypatch):
    monkeypatch.setattr(sync_worker.settings, "sync_interval_seconds", 10)
    monkeypatch.setattr(sync_worker.settings, "sync_retry_backoff_seconds", 30)
    monkeypatch.setattr(sync_worker.settings, "sync_retry_max_seconds", 300)

    assert sync_worker.get_sync_wait_seconds(0) == 10
    assert sync_worker.get_sync_wait_seconds(1) == 30
    assert sync_worker.get_sync_wait_seconds(2) == 60
    assert sync_worker.get_sync_wait_seconds(5) == 300
    assert sync_worker.get_sync_wait_seconds(99) == 300


def test_device_id_scan_id_idempotency_is_enforced_by_unique_key(sqlite_session_factory, monkeypatch):
    fake_engine = FakePGEngine()
    monkeypatch.setattr(sync_worker, "get_postgres_engine", lambda: fake_engine)
    scan = make_pending_scan(sqlite_session_factory, device_id="device-dup", batch_id="BATCH_DUP")

    first = sync_worker.sync_batch()
    second = sync_worker.sync_batch()

    assert first is True
    assert second is False
    assert len(fake_engine.executed) == 1
    assert fake_engine.executed[0]["device_id"] == "device-dup"
    assert fake_engine.executed[0]["scan_id"] == scan.scan_id
