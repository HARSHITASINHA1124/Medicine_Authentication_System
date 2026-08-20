# ML + Frontend Handoff

This document reflects the current backend contract for the Raspberry Pi + ML + cloud sync flow.

## 1) ML -> Backend Integration

### Endpoint
POST /api/scans/

### Exact request JSON
```json
{
  "medicine": "Dolonex",
  "classification_confidence": 0.5481,
  "anomaly_score": 51430928427314.336,
  "classification_status": "LOW",
  "anomaly_status": "COUNTERFEIT",
  "final_status": "COUNTERFEIT",
  "explainability": {},
  "scan_data": {
    "n_readings": 10,
    "aggregated_reading": {
      "ch450": 1000.0,
      "ch500": 1200.0,
      "ch550": 1400.0,
      "ch570": 1500.0,
      "ch600": 1300.0,
      "ch650": 900.0
    },
    "channel_std": {
      "ch450": 5.74,
      "ch500": 6.89,
      "ch550": 8.04,
      "ch570": 8.62,
      "ch600": 7.47,
      "ch650": 5.17
    },
    "stability_cv": 0.0057
  },
  "classification_probabilities": {
    "Dolonex": 0.5481,
    "Etoricoxib": 0.4513
  }
}
```

### Field definitions
| Field | Type | Nullable | Notes |
|---|---|---:|---|
| `medicine` | string | Yes | Predicted medicine name; stored locally as `medicine_name` |
| `classification_confidence` | number | Yes | Range: 0.0 to 1.0 |
| `anomaly_score` | number | Yes | Non-negative model score; it is not limited to 0.0-1.0 |
| `classification_status` | string | Yes | `HIGH`, `MEDIUM`, or `LOW` |
| `anomaly_status` | string | Yes | `GENUINE`, `COUNTERFEIT`, or `SUSPICIOUS` |
| `final_status` | string | Yes | `GENUINE`, `COUNTERFEIT`, or `SUSPICIOUS` |
| `scan_data.n_readings` | integer | Yes | Number of sensor rows; normally 10 |
| `scan_data.aggregated_reading` | object | Yes | Six readings: `ch450`, `ch500`, `ch550`, `ch570`, `ch600`, `ch650` |
| `scan_data.channel_std` | object | Yes | Standard deviation for each channel |
| `scan_data.stability_cv` | number | Yes | Non-negative measurement stability value |
| `explainability` | object | Yes | Classification/anomaly explanations and top engineered features |
| `classification_probabilities` | object | Yes | Medicine-to-probability map; values range from 0.0 to 1.0 |

### Validation rules
- `final_status` and `anomaly_status` must be one of `"GENUINE"`, `"COUNTERFEIT"`, or `"SUSPICIOUS"` when provided.
- `confidence_score` must be between `0.0` and `1.0` when provided.
- `classification_confidence` must be between `0.0` and `1.0` when provided.
- `anomaly_score` may be any non-negative number produced by the anomaly model.
- All ML fields are nullable and may be omitted when unavailable.
- Nulls are valid for incomplete or unavailable ML results.

### Raspberry Pi / ML data pipeline
- The Raspberry Pi produces 6 input channels for one reading.
- A single sample may include 10 rows from the sensor pipeline before ML evaluation.
- The backend expects the final ML result payload after aggregation/processing, not the raw 10-row sensor stream.
- The ML service returns the final status, confidence/anomaly values, six aggregate channel readings, standard deviations, stability, probabilities, and explainability.
- The backend maps `ch450` through `ch650` to its six channel columns and preserves the complete received object in `ml_result`.

### Backend-controlled fields
These are created by the backend and should not be provided by ML clients:
- `id` — internal auto-increment database primary key
- `device_id` — from `DEVICE_ID` in `.env`
- `timestamp` — server-generated UTC timestamp
- `sync_status` — backend-managed local status (`pending` initially)

### Example valid request
```json
{
  "scan_id": 101,
  "batch_id": "B12345",
  "medicine_name": "Paracetamol",
  "channel_1": 0.92,
  "channel_2": 0.88,
  "channel_3": 0.76,
  "channel_4": 0.81,
  "channel_5": 0.90,
  "channel_6": 0.87,
  "classification": "Genuine",
  "confidence_score": 0.96,
  "anomaly_score": 0.04
}
```

### Example nullable request
```json
{
  "scan_id": null,
  "batch_id": null,
  "medicine_name": null,
  "channel_1": null,
  "channel_2": null,
  "channel_3": null,
  "channel_4": null,
  "channel_5": null,
  "channel_6": null,
  "classification": null,
  "confidence_score": null,
  "anomaly_score": null
}
```

---

## 2) Backend -> Frontend Integration

### GET /api/scans/
List scans with pagination.

Query parameters:
- `skip` (int, default `0`)
- `limit` (int, default `50`)

Example response:
```json
[
  {
    "id": 1,
    "scan_id": 1,
    "device_id": "PI_001",
    "batch_id": "B12345",
    "medicine_name": "Paracetamol",
    "channel_1": 0.92,
    "channel_2": 0.88,
    "channel_3": 0.76,
    "channel_4": 0.81,
    "channel_5": 0.90,
    "channel_6": 0.87,
    "timestamp": "2026-08-14T12:00:00Z",
    "classification": "Genuine",
    "confidence_score": 0.96,
    "anomaly_score": 0.04,
    "sync_status": "pending"
  }
]
```

Pagination behavior:
- Response is a JSON array.
- Use `skip` and `limit` to page results.
- `limit` caps the number of returned rows.
- No pagination metadata wrapper is included.

### GET /api/scans/{scan_id}
Example response:
```json
{
  "id": 3,
  "scan_id": 3,
  "device_id": "PI_001",
  "batch_id": "B12345",
  "medicine_name": "Paracetamol",
  "channel_1": 0.92,
  "channel_2": 0.88,
  "channel_3": 0.76,
  "channel_4": 0.81,
  "channel_5": 0.90,
  "channel_6": 0.87,
  "timestamp": "2026-08-14T12:00:00Z",
  "classification": "Genuine",
  "confidence_score": 0.96,
  "anomaly_score": 0.04,
  "sync_status": "pending"
}
```

### GET /api/batches/
Query parameters:
- `skip` (int, default `0`)
- `limit` (int, default `50`)

Example response:
```json
[
  {
    "batch_id": "B12345",
    "medicine_name": "Paracetamol",
    "manufacturer": "Alpha Pharma",
    "batch_number": "ALPHA-001",
    "manufacturing_date": "2025-01-01",
    "expiry_date": "2028-01-01",
    "created_at": "2026-08-14T12:00:00Z"
  }
]
```

### GET /api/batches/{batch_id}
Example response:
```json
{
  "batch_id": "B12345",
  "medicine_name": "Paracetamol",
  "manufacturer": "Alpha Pharma",
  "batch_number": "ALPHA-001",
  "manufacturing_date": "2025-01-01",
  "expiry_date": "2028-01-01",
  "created_at": "2026-08-14T12:00:00Z"
}
```

### GET /api/dashboard/stats
Example response:
```json
{
  "total_scans": 42,
  "genuine": 30,
  "counterfeit": 12,
  "suspicious": 5,
  "pending_sync": 3
}
```

### GET /api/dashboard/trends
Query parameters:
- `days` (int, default `7`)

Example response:
```json
{
  "trends": [
    {"date": "2026-08-08", "count": 12},
    {"date": "2026-08-09", "count": 7},
    {"date": "2026-08-10", "count": 9}
  ]
}
```

### GET /api/dashboard/risk
Example response:
```json
{
  "average_anomaly_score": 0.2143
}
```

### Error responses
- `404 Not Found` — scan or batch does not exist
- `422 Unprocessable Entity` — invalid request body / validation failure
- `400 Bad Request` — duplicate batch ID on creation

Example 404 response:
```json
{
  "detail": "Scan not found"
}
```

Example 422 response:
```json
{
  "detail": [
    {
      "loc": ["body", "confidence_score"],
      "msg": "Input should be less than or equal to 1",
      "type": "less_than_equal"
    }
  ]
}
```

---

## 3) Backend-controlled fields

The following values are backend-controlled and should not be set or modified by the ML team or frontend clients unless explicitly required by the backend contract.

- `id` — internal database primary key used by SQLAlchemy
- `device_id` — device identity, loaded from `DEVICE_ID` in `.env`
- `scan_id` — application-level identifier, usually provided by the ML/PI pipeline or auto-filled when missing
- `timestamp` — UTC timestamp assigned by backend at creation time
- `sync_status` — backend synchronization state (`pending` or `synced`)

Frontend/ML clients should treat these values as read-only unless the backend explicitly exposes an update path.

---

## 4) Database behavior

### SQLite local/offline storage
- SQLite is the local source of truth.
- Scans and batches are saved locally before cloud sync.
- The app works fully offline.
- No cloud database is required for scan creation or local result retrieval.

### Neon PostgreSQL cloud storage
- Neon is used for centralized cloud storage and backup.
- It is not required for local operation.
- The cloud database stores the same core data model for central aggregation and analysis.

### Synchronization flow
1. Scan is created locally in SQLite.
2. Local record is marked `pending`.
3. Backend immediately returns the result to the caller.
4. Background sync worker uploads pending scans to Neon.
5. After successful confirmation, local row is set to `synced`.
6. The SQLite record is not deleted until cloud confirmation succeeds.

### Idempotency
The cloud sync uses the uniqueness of `(device_id, scan_id)`.

This prevents duplicate Neon rows on repeated sync attempts and app restarts.
The internal `id` column is the database primary key, while `scan_id` remains the application identifier used for idempotent cloud writes.

---

## 5) Running the backend locally

1. Open a terminal in the backend folder.
2. Create and activate a virtual environment.
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Copy `.env.example` to `.env` and set the values.
5. Start the server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```
6. Open the docs at:
```text
http://127.0.0.1:8000/docs
```

### Example `.env`
```env
DEVICE_ID=PI_001
DATABASE_URL=sqlite:///./local_scans.db
POSTGRES_URL=postgresql://user:password@host:5432/dbname?sslmode=require
SYNC_INTERVAL_SECONDS=300
```

---

## 6) Example curl / Postman requests

### Create a scan
```bash
curl -X POST "http://127.0.0.1:8000/api/scans/" \
  -H "Content-Type: application/json" \
  -d '{
    "batch_id": "B12345",
    "medicine_name": "Paracetamol",
    "classification": "Genuine",
    "confidence_score": 0.96,
    "anomaly_score": 0.04
  }'
```

### Get scans
```bash
curl "http://127.0.0.1:8000/api/scans/?skip=0&limit=10"
```

### Get one scan
```bash
curl "http://127.0.0.1:8000/api/scans/1"
```

### Create a batch
```bash
curl -X POST "http://127.0.0.1:8000/api/batches/" \
  -H "Content-Type: application/json" \
  -d '{
    "batch_id": "B12345",
    "medicine_name": "Paracetamol",
    "manufacturer": "Alpha Pharma",
    "batch_number": "ALPHA-001",
    "manufacturing_date": "2025-01-01",
    "expiry_date": "2028-01-01"
  }'
```

### Get batches
```bash
curl "http://127.0.0.1:8000/api/batches/"
```

### Dashboard stats
```bash
curl "http://127.0.0.1:8000/api/dashboard/stats"
```

### Dashboard trends
```bash
curl "http://127.0.0.1:8000/api/dashboard/trends?days=7"
```

### Dashboard risk
```bash
curl "http://127.0.0.1:8000/api/dashboard/risk"
```

---

## 7) Assumptions to confirm with ML / Frontend team

- Whether `classification` is always one of `Genuine` / `Counterfeit` or if a third value may be introduced later.
- Whether `batch_id` is always provided by the ML pipeline or sometimes absent.
- Whether the frontend wants to display `sync_status` directly.
- Whether the frontend needs raw scan ID ordering or a separate display-friendly field.
- Whether batch metadata is created separately before or after scan processing.
- Whether additional cloud-only metadata is expected in future backend versions.

## Contact / Handoff summary
- The backend is already offline-first and sync-safe.
- ML should send scan results via `POST /api/scans/` using the contract above.
- Frontend should consume the GET endpoints above and treat backend-created fields as read-only.
- The SQLite → Neon sync uses idempotent cloud inserts and preserves local pending records until confirmation succeeds.
