-- PostgreSQL schema for central cloud database
-- Run with: psql <connection_string> -f create_tables.sql

CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT PRIMARY KEY,
    medicine_name TEXT,
    manufacturer TEXT,
    batch_number TEXT,
    manufacturing_date DATE,
    expiry_date DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

CREATE TABLE IF NOT EXISTS scans (
    id BIGSERIAL PRIMARY KEY,
    scan_id BIGINT NOT NULL,
    device_id TEXT NOT NULL,
    batch_id TEXT,
    medicine_name TEXT,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    channel_1 DOUBLE PRECISION,
    channel_2 DOUBLE PRECISION,
    channel_3 DOUBLE PRECISION,
    channel_4 DOUBLE PRECISION,
    channel_5 DOUBLE PRECISION,
    channel_6 DOUBLE PRECISION,
    classification TEXT,
    confidence_score DOUBLE PRECISION,
    anomaly_score DOUBLE PRECISION,
    classification_status TEXT,
    anomaly_status TEXT,
    final_status TEXT,
    number_of_readings INTEGER,
    stability_cv DOUBLE PRECISION,
    ml_result JSONB,
    sync_status TEXT,
    UNIQUE (device_id, scan_id)
);

ALTER TABLE scans ADD COLUMN IF NOT EXISTS classification_status TEXT;
ALTER TABLE scans ADD COLUMN IF NOT EXISTS anomaly_status TEXT;
ALTER TABLE scans ADD COLUMN IF NOT EXISTS final_status TEXT;
ALTER TABLE scans ADD COLUMN IF NOT EXISTS number_of_readings INTEGER;
ALTER TABLE scans ADD COLUMN IF NOT EXISTS stability_cv DOUBLE PRECISION;
ALTER TABLE scans ADD COLUMN IF NOT EXISTS ml_result JSONB;

-- Optional index for queries by timestamp / sync_status
CREATE INDEX IF NOT EXISTS idx_scans_timestamp ON scans (timestamp);
CREATE INDEX IF NOT EXISTS idx_scans_sync_status ON scans (sync_status);
